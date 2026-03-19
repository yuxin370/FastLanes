// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/benchmark/table.cu
// ────────────────────────────────────────────────────────
#include "engine/benchmark/table.cuh"
#include "engine/execution/internal/materialize.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/expression.cuh"
#include "engine/reader.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/datatype_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include <chrono>
#include <stdexcept>

namespace dispatch {
namespace {

size_t data_type_size(const fastlanes::DataType dt) {
	using fastlanes::DataType;
	switch (dt) {
	case DataType::INT8:
	case DataType::UINT8:
	case DataType::BOOLEAN:
		return 1;
	case DataType::INT16:
	case DataType::UINT16:
		return 2;
	case DataType::INT32:
	case DataType::UINT32:
	case DataType::FLOAT:
	case DataType::DATE:
		return 4;
	case DataType::INT64:
	case DataType::UINT64:
	case DataType::DOUBLE:
	case DataType::TIMESTAMP:
		return 8;
	default:
		return 0;
	}
}

fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};

	fastlanes::FileHeader::Load(header, file_path);
	fastlanes::FileFooter::Load(footer, file_path);

	if (header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file_path, footer.table_descriptor_offset, footer.table_descriptor_size, /*verify=*/true);
	}

	return fastlanes::TableDescriptorHandle::FromFile(file_path.parent_path() / "table_descriptor.fbb",
	                                                  /*verify=*/true);
}

double measure_rowgroup_io_ms(const std::filesystem::path& file_path, const fastlanes::RowgroupDescriptor* rg) {
	if (!rg) {
		return 0.0;
	}
	using Clock = std::chrono::steady_clock;
	fastlanes::Buf buf(rg->m_size());
	fastlanes::io  io    = fastlanes::make_unique<fastlanes::File>(file_path);
	const auto     start = Clock::now();
	fastlanes::IO::range_read(io, buf, rg->m_offset(), rg->m_size());
	const auto end = Clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

size_t rowgroup_bytes(const fastlanes::RowgroupDescriptor* rg) {
	if (!rg) {
		return 0;
	}
	size_t      rg_bytes = 0;
	const auto* cols     = rg->m_column_descriptors();
	if (!cols) {
		return 0;
	}
	for (const auto* col_desc : *cols) {
		if (!col_desc) {
			continue;
		}
		const size_t element_size = data_type_size(col_desc->data_type());
		if (element_size == 0) {
			continue;
		}
		rg_bytes += static_cast<size_t>(rg->m_n_tuples()) * element_size;
	}
	return rg_bytes;
}

void check_rowgroup_index(const size_t n_rowgroups, const std::optional<size_t>& rowgroup) {
	if (rowgroup.has_value() && *rowgroup >= n_rowgroups) {
		throw std::out_of_range("rowgroup index out of range");
	}
}

size_t count_active_columns(const std::vector<expr::Expression>& expressions) {
	return runtime::count_active_columns(expressions);
}

struct AppendExpressionsProfile {
	double total_ms        = 0.0;
	double resolve_cpu_ms  = 0.0;
	double dispatch_cpu_ms = 0.0;
	double payload_h2d_ms  = 0.0;
};

template <typename T, typename HostColT>
void add_expression_to_batch_profiled(const size_t    expr_index,
                                      const HostColT& host_col,
                                      const PlanKind  plan,
                                      Batch<T>&       batch,
                                      const ExecutionConfig& cfg,
                                      AppendExpressionsProfile& profile) {
	const auto expr_start = std::chrono::steady_clock::now();
	DeviceExpression<T> expr {};
	expr.plan     = plan;
	expr.n_values = host_col.get_n_values();
	batch.device_outputs.emplace_back(expr.n_values);
	expr.out = batch.device_outputs.back().get();

	bool use_freq_extended = false;
	if constexpr (std::is_same_v<HostColT, flsgpu::host::FREQColumn<T>>) {
		use_freq_extended = detail::should_use_freq_extended(
		    host_col, cfg.freq_prefetch_all_branchless, cfg.freq_hybrid_patcher, cfg.freq_branchless_threshold);
	}

	const auto payload_start = std::chrono::steady_clock::now();
	detail::fill_device_expr(expr, host_col, plan, use_freq_extended);
	const auto payload_end = std::chrono::steady_clock::now();
	const auto payload_ms = std::chrono::duration<double, std::milli>(payload_end - payload_start).count();
	profile.payload_h2d_ms += payload_ms;

	const auto device_idx = static_cast<uint32_t>(batch.device_exprs.size());
	batch.device_exprs.push_back(expr);
	batch.expr_indices.push_back(expr_index);

	const size_t n_vecs = utils::get_n_vecs_from_size(expr.n_values);
	for (size_t vec = 0; vec < n_vecs; ++vec) {
		batch.work_items.push_back(WorkItemAny {device_idx, static_cast<uint32_t>(vec), type_tag_for<T>()});
	}

	const auto expr_end = std::chrono::steady_clock::now();
	const auto expr_total_ms = std::chrono::duration<double, std::milli>(expr_end - expr_start).count();
	const auto dispatch_cpu_ms = expr_total_ms - payload_ms;
	if (dispatch_cpu_ms > 0.0) {
		profile.dispatch_cpu_ms += dispatch_cpu_ms;
	}
}

AppendExpressionsProfile append_expressions_profiled(runtime::ExecutionWorkset&      workset,
                                                     std::vector<expr::Expression>& expressions,
                                                     const ExecutionConfig&         cfg,
                                                     size_t*                        out_total_bytes = nullptr,
                                                     size_t*                        out_n_exprs     = nullptr) {
	AppendExpressionsProfile profile {};
	const auto               start = std::chrono::steady_clock::now();

	const auto resolve_start = std::chrono::steady_clock::now();
	resolve_dict_refs(expressions);
	const auto resolve_end = std::chrono::steady_clock::now();
	profile.resolve_cpu_ms += std::chrono::duration<double, std::milli>(resolve_end - resolve_start).count();

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		if (out_n_exprs) {
			++(*out_n_exprs);
		}

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT      = std::decay_t<decltype(host_col)>;
			    using T             = typename host_value_type<HostColT>::type;
			    constexpr auto plan = detail::plan_for_host_col<HostColT>();
			    if constexpr (is_supported_type_v<T>) {
				    if (out_total_bytes) {
					    *out_total_bytes += host_col.get_n_values() * sizeof(T);
				    }
				    add_expression_to_batch_profiled<T>(i, host_col, plan, workset.host_batches.template get<T>(), cfg, profile);
			    }
		    },
		    expr.column->host);
	}

	const auto end = std::chrono::steady_clock::now();
	profile.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	return profile;
}

void accumulate_rowgroup_stats(TableBenchmarkResult& result,
                               const size_t          rg_columns,
                               const size_t          rg_vectors,
                               const size_t          bytes,
                               const double          setup_ms,
                               const AppendExpressionsProfile& append_profile,
                               const double          upload_h2d_ms,
                               const double          kernel_ms,
                               const double          teardown_ms,
                               const size_t          launch_grid,
                               const size_t          launches) {
	result.end_to_end_ms += setup_ms + append_profile.total_ms + upload_h2d_ms + kernel_ms;
	result.kernel_ms += kernel_ms;
	result.setup_ms += setup_ms;
	result.h2d_ms += append_profile.total_ms + upload_h2d_ms;
	result.pure_h2d_ms += append_profile.payload_h2d_ms + upload_h2d_ms;
	result.payload_h2d_ms += append_profile.payload_h2d_ms;
	result.dispatch_h2d_ms += upload_h2d_ms;
	result.resolve_cpu_ms += append_profile.resolve_cpu_ms;
	result.cpu_dispatch_ms += append_profile.dispatch_cpu_ms;
	result.cpu_dispatch_resolve_ms += append_profile.resolve_cpu_ms + append_profile.dispatch_cpu_ms;
	result.teardown_ms += teardown_ms;
	result.total_launches += launches;
	result.total_launch_grid += launch_grid * launches;
	result.total_columns += rg_columns;
	result.total_items += rg_vectors;
	result.total_bytes += bytes;
	++result.total_rgs;
}

} // namespace

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg) {
	TableBenchmarkResult out {};
	out.samples = cfg.samples;

	reader::reader rdr(fls_path);
	const size_t   n_rowgroups = rdr.rowgroup_count();
	check_rowgroup_index(n_rowgroups, cfg.rowgroup);

	size_t start = 0;
	size_t end   = n_rowgroups;
	if (cfg.rowgroup.has_value()) {
		start = *cfg.rowgroup;
		end   = start + 1;
	}

	const auto  td_handle = load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	if (!td) {
		throw std::runtime_error("failed to load table descriptor");
	}

	const bool whole_table = !cfg.rowgroup.has_value() && cfg.aggregation_scope == AggregationScope::WholeTable;
	if (!whole_table) {
		for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
			const auto*  rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			const double io_ms = measure_rowgroup_io_ms(fls_path, rg);
			const size_t bytes = rowgroup_bytes(rg);

			const auto setup_start = std::chrono::steady_clock::now();
			auto       rowgroup    = rdr.read_rowgroup(rg_idx);
			auto       expressions = expr::assemble(rowgroup);
			const auto setup_end   = std::chrono::steady_clock::now();

			const double setup_total_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
			double       setup_ms       = setup_total_ms - io_ms;
			if (setup_ms < 0.0) {
				setup_ms = 0.0;
			}

			const size_t rg_columns = count_active_columns(expressions);
			const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

			runtime::ExecutionWorkset      workset {};
			runtime::ExecutionWorksetGuard guard(workset);
			const auto                     append_profile = append_expressions_profiled(workset, expressions, cfg.execution);
			const auto                     upload_start   = std::chrono::steady_clock::now();
			runtime::upload_workset(workset);
			const auto   upload_end    = std::chrono::steady_clock::now();
			const double upload_h2d_ms = std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
			size_t                         rg_launch_grid = 0;
			size_t                         rg_launches    = 0;
			const double                   kernel_ms =
			    runtime::run_workset(workset, cfg.samples, cfg.execution, &rg_launch_grid, &rg_launches, true);
			const auto teardown_start = std::chrono::steady_clock::now();
			guard.dismiss();
			runtime::release_workset(workset);
			const auto   teardown_end = std::chrono::steady_clock::now();
			const double teardown_ms = std::chrono::duration<double, std::milli>(teardown_end - teardown_start).count();

			accumulate_rowgroup_stats(out,
			                          rg_columns,
			                          rg_vectors,
			                          bytes,
			                          setup_ms,
			                          append_profile,
			                          upload_h2d_ms,
			                          kernel_ms,
			                          teardown_ms,
			                          rg_launch_grid,
			                          rg_launches);
			free_rowgroup(rowgroup);
		}
		return out;
	}

	runtime::ExecutionWorkset      workset {};
	runtime::ExecutionWorksetGuard guard(workset);
	for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
		const auto*  rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
		const double io_ms = measure_rowgroup_io_ms(fls_path, rg);
		const size_t bytes = rowgroup_bytes(rg);

		const auto setup_start = std::chrono::steady_clock::now();
		auto       rowgroup    = rdr.read_rowgroup(rg_idx);
		auto       expressions = expr::assemble(rowgroup);
		const auto setup_end   = std::chrono::steady_clock::now();

		const double setup_total_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
		double       setup_ms       = setup_total_ms - io_ms;
		if (setup_ms < 0.0) {
			setup_ms = 0.0;
		}

		const size_t rg_columns = count_active_columns(expressions);
		const size_t rg_vectors = rg_columns * rowgroup.n_vecs;
		const auto   append_profile = append_expressions_profiled(workset, expressions, cfg.execution);
		out.setup_ms += setup_ms;
		out.h2d_ms += append_profile.total_ms;
		out.pure_h2d_ms += append_profile.payload_h2d_ms;
		out.payload_h2d_ms += append_profile.payload_h2d_ms;
		out.resolve_cpu_ms += append_profile.resolve_cpu_ms;
		out.cpu_dispatch_ms += append_profile.dispatch_cpu_ms;
		out.cpu_dispatch_resolve_ms += append_profile.resolve_cpu_ms + append_profile.dispatch_cpu_ms;
		out.total_columns += rg_columns;
		out.total_items += rg_vectors;
		out.total_bytes += bytes;
		++out.total_rgs;
		free_rowgroup(rowgroup);
	}

	const auto upload_start = std::chrono::steady_clock::now();
	runtime::upload_workset(workset);
	const auto   upload_end    = std::chrono::steady_clock::now();
	const double upload_h2d_ms = std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
	out.h2d_ms += upload_h2d_ms;
	out.pure_h2d_ms += upload_h2d_ms;
	out.dispatch_h2d_ms += upload_h2d_ms;
	size_t launch_grid  = 0;
	size_t launch_count = 0;
	out.kernel_ms       = runtime::run_workset(workset, cfg.samples, cfg.execution, &launch_grid, &launch_count, true);
	if (launch_count > 0) {
		out.total_launches    = launch_count;
		out.total_launch_grid = launch_grid * launch_count;
	}

	const auto teardown_start = std::chrono::steady_clock::now();
	guard.dismiss();
	runtime::release_workset(workset);
	const auto teardown_end = std::chrono::steady_clock::now();
	out.teardown_ms += std::chrono::duration<double, std::milli>(teardown_end - teardown_start).count();

	out.end_to_end_ms = out.setup_ms + out.h2d_ms + out.kernel_ms;
	return out;
}

} // namespace dispatch
