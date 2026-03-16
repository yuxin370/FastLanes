// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/execution/table.cu
// ────────────────────────────────────────────────────────
#include "engine/data/value-store.cuh"
#include "engine/execution/dict_ref_resolver.cuh"
#include "engine/execution/table.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/datatype_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include <chrono>
#include <unordered_set>

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

void validate_unique_work_items(const std::vector<dispatch::WorkItemAny>& items) {
	std::unordered_set<uint64_t> seen;
	seen.reserve(items.size() * 2 + 1);
	for (const auto& w : items) {
		const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(w.type)) << 56) |
		                     (static_cast<uint64_t>(w.expr_index) << 28) | static_cast<uint64_t>(w.vector_index);
		if (!seen.insert(key).second) {
			throw std::runtime_error("duplicate work item detected (type,expr_index,vector_index)");
		}
	}
}

} // namespace

double append_expressions(BenchmarkWorkset&              workset,
                          std::vector<expr::Expression>& expressions,
                          size_t*                        out_total_bytes,
                          size_t*                        out_n_exprs) {
	using namespace dispatch;
	using namespace dispatch::detail;

	const auto start = std::chrono::steady_clock::now();
	dispatch::resolve_dict_refs(expressions);
	// dispatch::sync_expression_ops_after_resolve(expressions); // only for validation

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
			    using T             = typename dispatch::host_value_type<HostColT>::type;
			    constexpr auto plan = dispatch::detail::plan_for_host_col<HostColT>();
			    if constexpr (dispatch::is_supported_type_v<T>) {
				    if (out_total_bytes) {
					    *out_total_bytes += host_col.get_n_values() * sizeof(T);
				    }
				    add_expression_to_batch<T>(i,
				                               host_col,
				                               plan,
				                               workset.host_batches.template get<T>(),
				                               workset.freq_prefetch_all_branchless,
				                               workset.freq_hybrid_patcher,
				                               workset.freq_branchless_threshold);
			    }
		    },
		    expr.column->host);
	}

	const auto end = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

double prepare_dispatch_buffers(BenchmarkWorkset& workset) {
	const auto start = std::chrono::steady_clock::now();

	workset.d_items.reset();
	workset.work_items.clear();

	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T          = typename decltype(tag)::type;
		auto& host_batch = workset.host_batches.template get<T>();
		if (!host_batch.device_exprs.empty() && !host_batch.work_items.empty()) {
			auto& dev_batch = workset.device_batches.template get<T>();
			dev_batch.d_exprs.emplace(host_batch.device_exprs.size(), host_batch.device_exprs.data());
			dev_batch.d_items.emplace(host_batch.work_items.size(), host_batch.work_items.data());
			dev_batch.n_items = host_batch.work_items.size();
			workset.work_items.reserve(workset.work_items.size() + host_batch.work_items.size());
			workset.work_items.insert(
			    workset.work_items.end(), host_batch.work_items.begin(), host_batch.work_items.end());
		}
	});

	if (!workset.work_items.empty()) {
		validate_unique_work_items(workset.work_items);
		workset.d_items.emplace(workset.work_items.size(), workset.work_items.data());
	}

	const auto end = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

double run_kernel(BenchmarkWorkset& workset,
                  uint32_t          samples,
                  const bool        gpu_dispatch_kernel,
                  const bool        write_out,
                  size_t*           out_grid,
                  size_t*           out_launches) {
	const size_t n_items = workset.work_items.size();
	if (n_items == 0) {
		return 0.0;
	}
	if (gpu_dispatch_kernel && !workset.d_items.has_value()) {
		return 0.0;
	}

	bool   has_any_expr                 = false;
	size_t launches_per_sample          = 0;
	size_t typed_total_items_per_sample = 0;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		auto& d = workset.device_batches.template get<T>();
		if (d.d_exprs.has_value() && d.d_items.has_value()) {
			has_any_expr = true;
			++launches_per_sample;
			typed_total_items_per_sample += d.n_items;
		}
	});
	if (!has_any_expr) {
		if (out_launches) {
			*out_launches = 0;
		}
		return 0.0;
	}
	constexpr uint32_t lanes_i8   = utils::get_n_lanes<int8_t>();
	constexpr uint32_t lanes_i16  = utils::get_n_lanes<int16_t>();
	constexpr uint32_t warp_lanes = (lanes_i8 > lanes_i16) ? lanes_i8 : lanes_i16;

	const size_t mega_launches_per_sample = gpu_dispatch_kernel ? 1 : launches_per_sample;

	if (out_grid) {
		*out_grid = gpu_dispatch_kernel
		                ? ((n_items + 255) / 256)
		                : (launches_per_sample > 0 ? typed_total_items_per_sample / launches_per_sample : 0);
	}
	if (out_launches) {
		*out_launches =
		    (gpu_dispatch_kernel ? mega_launches_per_sample : launches_per_sample) * static_cast<size_t>(samples);
	}

	flsgpu::memory::sync_h2d();

	cudaStream_t stream {};
	CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	cudaEvent_t start {};
	cudaEvent_t stop {};
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	const auto launch_mixed = [&](const dispatch::DeviceExpression<int8_t>*  exprs_i8,
	                              const dispatch::DeviceExpression<int16_t>* exprs_i16,
	                              const dim3&                                grid,
	                              const dim3&                                block) {
		const uint32_t i16_start_index = static_cast<uint32_t>(workset.device_batches.template get<int8_t>().n_items);
		if (write_out) {
			kernels::device::decompress_dispatch_mixed<1, 1, true>
			    <<<grid, block, 0, stream>>>(exprs_i8, exprs_i16, workset.d_items->get(), n_items, i16_start_index);
		} else {
			kernels::device::decompress_dispatch_mixed<1, 1, false>
			    <<<grid, block, 0, stream>>>(exprs_i8, exprs_i16, workset.d_items->get(), n_items, i16_start_index);
		}
		CUDA_SAFE_CALL(cudaGetLastError());
	};

	// Warm up at least once so timing is less biased by first-launch effects.
	if (gpu_dispatch_kernel) {
		const auto*        exprs_i8      = workset.device_batches.template get<int8_t>().d_exprs.has_value()
		                                       ? workset.device_batches.template get<int8_t>().d_exprs->get()
		                                       : nullptr;
		const auto*        exprs_i16     = workset.device_batches.template get<int16_t>().d_exprs.has_value()
		                                       ? workset.device_batches.template get<int16_t>().d_exprs->get()
		                                       : nullptr;
		constexpr uint32_t block_threads = 256;
		const dim3         block(block_threads);
		const size_t       n_threads = n_items * static_cast<size_t>(warp_lanes);
		const dim3         grid(static_cast<unsigned>((n_threads + block_threads - 1) / block_threads));
		launch_mixed(exprs_i8, exprs_i16, grid, block);
	} else {
		dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
			using T            = typename decltype(tag)::type;
			auto& host_batch   = workset.host_batches.template get<T>();
			auto& device_batch = workset.device_batches.template get<T>();
			if (!device_batch.d_exprs.has_value() || !device_batch.d_items.has_value()) {
				return;
			}
			if (write_out) {
				dispatch::detail::launch_batch_no_sync<T, true>(
				    host_batch, device_batch.d_exprs->get(), device_batch.d_items->get(), device_batch.n_items, stream);
			} else {
				dispatch::detail::launch_batch_no_sync<T, false>(
				    host_batch, device_batch.d_exprs->get(), device_batch.d_items->get(), device_batch.n_items, stream);
			}
		});
	}
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

	CUDA_SAFE_CALL(cudaEventRecord(start, stream));

	for (uint32_t sample = 0; sample < samples; ++sample) {
		if (gpu_dispatch_kernel) {
			const auto*        exprs_i8      = workset.device_batches.template get<int8_t>().d_exprs.has_value()
			                                       ? workset.device_batches.template get<int8_t>().d_exprs->get()
			                                       : nullptr;
			const auto*        exprs_i16     = workset.device_batches.template get<int16_t>().d_exprs.has_value()
			                                       ? workset.device_batches.template get<int16_t>().d_exprs->get()
			                                       : nullptr;
			constexpr uint32_t block_threads = 256;
			const dim3         block(block_threads);
			const size_t       n_threads = n_items * static_cast<size_t>(warp_lanes);
			const dim3         grid(static_cast<unsigned>((n_threads + block_threads - 1) / block_threads));
			launch_mixed(exprs_i8, exprs_i16, grid, block);
			continue;
		}
		dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
			using T            = typename decltype(tag)::type;
			auto& host_batch   = workset.host_batches.template get<T>();
			auto& device_batch = workset.device_batches.template get<T>();
			if (!device_batch.d_exprs.has_value() || !device_batch.d_items.has_value()) {
				return;
			}
			if (write_out) {
				dispatch::detail::launch_batch_no_sync<T, true>(
				    host_batch, device_batch.d_exprs->get(), device_batch.d_items->get(), device_batch.n_items, stream);
			} else {
				dispatch::detail::launch_batch_no_sync<T, false>(
				    host_batch, device_batch.d_exprs->get(), device_batch.d_items->get(), device_batch.n_items, stream);
			}
		});
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop, stream));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));
	CUDA_SAFE_CALL(cudaStreamDestroy(stream));

	return static_cast<double>(ms);
}

void free_batches(BenchmarkWorkset& workset) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T          = typename decltype(tag)::type;
		auto& host_batch = workset.host_batches.template get<T>();
		for (auto& expr : host_batch.device_exprs) {
			dispatch::detail::free_device_expr(expr);
		}
	});
}

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

	const bool use_mega_kernel = (!cfg.rowgroup.has_value() && cfg.mega_kernel);
	if (!use_mega_kernel) {
		for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
			const auto*  rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			const double io_ms = measure_rowgroup_io_ms(fls_path, rg);
			const size_t bytes = rowgroup_bytes(rg);

			const auto setup_start = std::chrono::steady_clock::now();
			auto       rowgroup    = rdr.read_rowgroup(rg_idx);
			auto       expressions = expr::assemble(rowgroup);
			const auto setup_end   = std::chrono::steady_clock::now();

			const double setup_total_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
			double       setup          = setup_total_ms - io_ms;
			if (setup < 0.0) {
				setup = 0.0;
			}

			size_t rg_columns = 0;
			for (const auto& expr : expressions) {
				if (expr.column && !expr.column->skip_decompress) {
					++rg_columns;
				}
			}
			const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

			dispatch::BenchmarkWorkset workset;
			workset.freq_prefetch_all_branchless = cfg.freq_prefetch_all_branchless;
			workset.freq_hybrid_patcher          = cfg.freq_hybrid_patcher;
			workset.freq_branchless_threshold    = cfg.freq_branchless_threshold;
			const double rg_h2d_ms               = dispatch::append_expressions(workset, expressions);
			const double rg_finalize_ms          = dispatch::prepare_dispatch_buffers(workset);
			size_t       rg_launches             = 0;
			size_t       rg_launch_grid          = 0;
			const double rg_kernel_ms            = dispatch::run_kernel(
                workset, cfg.samples, cfg.gpu_dispatch_kernel, cfg.write_out, &rg_launch_grid, &rg_launches);
			const auto teardown_start = std::chrono::steady_clock::now();
			dispatch::free_batches(workset);
			const auto   teardown_end = std::chrono::steady_clock::now();
			const double rg_teardown_ms =
			    std::chrono::duration<double, std::milli>(teardown_end - teardown_start).count();

			out.end_to_end_ms += setup + rg_h2d_ms + rg_finalize_ms + rg_kernel_ms;
			out.kernel_ms += rg_kernel_ms;
			out.setup_ms += setup;
			out.h2d_ms += (rg_h2d_ms + rg_finalize_ms);
			out.teardown_ms += rg_teardown_ms;
			out.total_launches += rg_launches;
			out.total_launch_grid += rg_launch_grid * rg_launches;
			out.total_columns += rg_columns;
			out.total_items += rg_vectors;
			out.total_bytes += bytes;
			++out.total_rgs;

			dispatch::free_rowgroup(rowgroup);
		}
		return out;
	}

	dispatch::BenchmarkWorkset workset;
	workset.freq_prefetch_all_branchless = cfg.freq_prefetch_all_branchless;
	workset.freq_hybrid_patcher          = cfg.freq_hybrid_patcher;
	workset.freq_branchless_threshold    = cfg.freq_branchless_threshold;
	for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
		const auto*  rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
		const double io_ms = measure_rowgroup_io_ms(fls_path, rg);
		const size_t bytes = rowgroup_bytes(rg);

		const auto setup_start = std::chrono::steady_clock::now();
		auto       rowgroup    = rdr.read_rowgroup(rg_idx);
		auto       expressions = expr::assemble(rowgroup);
		const auto setup_end   = std::chrono::steady_clock::now();

		const double setup_total_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
		double       setup          = setup_total_ms - io_ms;
		if (setup < 0.0) {
			setup = 0.0;
		}
		out.setup_ms += setup;

		size_t rg_columns = 0;
		for (const auto& expr : expressions) {
			if (expr.column && !expr.column->skip_decompress) {
				++rg_columns;
			}
		}
		const size_t rg_vectors = rg_columns * rowgroup.n_vecs;
		out.h2d_ms += dispatch::append_expressions(workset, expressions);

		out.total_columns += rg_columns;
		out.total_items += rg_vectors;
		out.total_bytes += bytes;
		++out.total_rgs;
		dispatch::free_rowgroup(rowgroup);
	}

	out.h2d_ms += dispatch::prepare_dispatch_buffers(workset);
	size_t launch_grid  = 0;
	size_t launch_count = 0;
	out.kernel_ms =
	    dispatch::run_kernel(workset, cfg.samples, cfg.gpu_dispatch_kernel, cfg.write_out, &launch_grid, &launch_count);
	if (!workset.work_items.empty()) {
		out.total_launches    = launch_count;
		out.total_launch_grid = launch_grid * launch_count;
	}

	const auto teardown_start = std::chrono::steady_clock::now();
	dispatch::free_batches(workset);
	workset.d_items.reset();
	const auto teardown_end = std::chrono::steady_clock::now();
	out.teardown_ms += std::chrono::duration<double, std::milli>(teardown_end - teardown_start).count();

	out.end_to_end_ms = out.setup_ms + out.h2d_ms + out.kernel_ms;
	return out;
}

} // namespace dispatch
