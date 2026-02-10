// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/galp_cli.cu
// ────────────────────────────────────────────────────────
#include "engine/dispatch/common.cuh"
#include "engine/dispatch/rowgroup.cuh"
#include "engine/dispatch/table.cuh"
#include "engine/expression.cuh"
#include "engine/pipeline.cuh"
#include "engine/reader.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/datatype_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

enum class Mode {
	ReadTable,
	Benchmark,
	MeasureLaunch,
};

struct Options {
	Mode                                 mode = Mode::ReadTable;
	std::filesystem::path                input;
	std::optional<std::filesystem::path> output;
	std::optional<size_t>                rowgroup;
	uint32_t                             samples         = 1;
	bool                                 header          = true;
	uint32_t                             launch_iters    = 100000;
	uint32_t                             launch_grid     = 1;
	uint32_t                             launch_block    = 1;
	bool                                 estimate_launch = false;
	uint32_t                             estimate_iters  = 10000;
	bool                                 mega_kernel     = true;
};

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

std::string format_bytes(double bytes) {
	static constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
	size_t                       unit     = 0;
	while (bytes >= 1024.0 && unit < (sizeof(kUnits) / sizeof(kUnits[0]) - 1)) {
		bytes /= 1024.0;
		++unit;
	}
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(bytes < 10.0 ? 2 : (bytes < 100.0 ? 1 : 0)) << bytes << " " << kUnits[unit];
	return oss.str();
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

template <typename T>
struct DeviceBatch {
	dispatch::Batch<T>                                     batch;
	std::optional<GPUArray<dispatch::DeviceExpression<T>>> d_exprs;
	std::optional<GPUArray<dispatch::WorkItemAny>>         d_items;
};

template <typename... Ts>
struct DeviceBatchSet {
	std::tuple<DeviceBatch<Ts>...> batches;

	template <typename T>
	DeviceBatch<T>& get() {
		return std::get<DeviceBatch<T>>(batches);
	}
};

template <typename List>
struct DeviceBatchSetFromList;

template <typename... Ts>
struct DeviceBatchSetFromList<dispatch::TypeList<Ts...>> {
	using type = DeviceBatchSet<Ts...>;
};

template <typename T>
void launch_batch_no_sync(DeviceBatch<T>& db) {
	if (!db.d_exprs.has_value() || !db.d_items.has_value()) {
		return;
	}
	if (db.batch.device_exprs.empty() || db.batch.work_items.empty()) {
		return;
	}
	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	const auto         launch           = make_workitem_launch_config<T, UNPACK_N_VECTORS>(db.batch.work_items.size());

	kernels::device::decompress_rowgroup<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
	    <<<launch.grid, launch.block>>>(db.d_exprs->get(), db.d_items->get(), db.batch.work_items.size());
	CUDA_SAFE_CALL(cudaGetLastError());
}

template <typename BatchesT>
bool batches_have_work(BatchesT& batches) {
	bool has_work = false;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		if (!batch.batch.device_exprs.empty() && !batch.batch.work_items.empty()) {
			has_work = true;
		}
	});
	return has_work;
}

template <typename BatchesT>
double build_gpu_batches(const std::vector<expr::Expression>& expressions, BatchesT& batches) {
	using namespace dispatch;
	using namespace dispatch::detail;

	const auto start = std::chrono::steady_clock::now();

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		const auto plan = plan_for_ops(expr.ops);

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT = std::decay_t<decltype(host_col)>;
			    using T        = typename dispatch::host_value_type<HostColT>::type;
			    if constexpr (dispatch::is_supported_type_v<T>) {
				    add_expression_to_batch<T>(i, host_col, plan, batches.template get<T>().batch);
			    }
		    },
		    expr.column->host);
	}

	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		if (!batch.batch.device_exprs.empty() && !batch.batch.work_items.empty()) {
			batch.d_exprs.emplace(batch.batch.device_exprs.size(), batch.batch.device_exprs.data());
			batch.d_items.emplace(batch.batch.work_items.size(), batch.batch.work_items.data());
		}
	});

	const auto end = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename BatchesT>
double run_gpu_kernels(BatchesT& batches, uint32_t samples) {
	if (!batches_have_work(batches)) {
		return 0.0;
	}
	flsgpu::memory::sync_h2d();

	cudaEvent_t start {};
	cudaEvent_t stop {};
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	for (uint32_t sample = 0; sample < samples; ++sample) {
		dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
			using T     = typename decltype(tag)::type;
			auto& batch = batches.template get<T>();
			if (!batch.batch.device_exprs.empty() && !batch.batch.work_items.empty()) {
				launch_batch_no_sync(batch);
			}
		});
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));

	return static_cast<double>(ms);
}

template <typename BatchesT>
void free_gpu_batches(BatchesT& batches) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		for (auto& expr : batch.batch.device_exprs) {
			dispatch::detail::free_device_expr(expr);
		}
	});
}

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " read_table <input.fls> [output.csv] [--rowgroup N] [--no-header]\n"
	          << "  " << prog << " benchmark <input.fls> [--rowgroup N] [--samples N]\n"
	          << "  " << prog << " measure_launch [--iters N] [--grid N] [--block N]\n"
	          << "\n"
	          << "Options:\n"
	          << "  --rowgroup N   Only process the given rowgroup\n"
	          << "  --samples N    Number of benchmark repetitions (default: 1)\n"
	          << "  --no-header    Skip CSV header\n"
	          << "  --out PATH     Output CSV path (read_table mode)\n"
	          << "  --iters N      Launch measurement iterations (default: 100000)\n"
	          << "  --grid N       Launch grid size for measurement (default: 1)\n"
	          << "  --block N      Launch block size for measurement (default: 1)\n"
	          << "  --estimate-launch  Estimate launch overhead during benchmark\n"
	          << "  --launch-iters N   Iterations for launch estimate (default: 10000)\n"
	          << "  --no-mega-kernel   Benchmark full table using per-rowgroup kernels\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
	if (argc < 2) {
		return false;
	}

	std::string_view mode_arg = argv[1];
	if (mode_arg == "read_table" || mode_arg == "read") {
		opt.mode = Mode::ReadTable;
	} else if (mode_arg == "benchmark" || mode_arg == "bench") {
		opt.mode = Mode::Benchmark;
	} else if (mode_arg == "measure_launch" || mode_arg == "launch") {
		opt.mode = Mode::MeasureLaunch;
	} else if (mode_arg == "--help" || mode_arg == "-h") {
		return false;
	} else {
		return false;
	}

	for (int i = 2; i < argc; ++i) {
		std::string_view arg = argv[i];
		if (arg == "--help" || arg == "-h") {
			return false;
		}
		if (arg == "--rowgroup" && i + 1 < argc) {
			opt.rowgroup = static_cast<size_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--samples" && i + 1 < argc) {
			opt.samples = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--estimate-launch") {
			opt.estimate_launch = true;
			continue;
		}
		if (arg == "--no-mega-kernel") {
			opt.mega_kernel = false;
			continue;
		}
		if (arg == "--launch-iters" && i + 1 < argc) {
			opt.estimate_iters = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--iters" && i + 1 < argc) {
			opt.launch_iters = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--grid" && i + 1 < argc) {
			opt.launch_grid = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--block" && i + 1 < argc) {
			opt.launch_block = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--no-header") {
			opt.header = false;
			continue;
		}
		if (arg == "--out" && i + 1 < argc) {
			opt.output = std::filesystem::path(argv[++i]);
			continue;
		}

		if (opt.input.empty()) {
			opt.input = std::filesystem::path(arg);
			continue;
		}

		if (opt.mode == Mode::ReadTable && !opt.output.has_value()) {
			opt.output = std::filesystem::path(arg);
			continue;
		}

		return false;
	}

	if (opt.mode == Mode::MeasureLaunch) {
		return true;
	}
	return !opt.input.empty();
}

template <typename PtrT>
void write_cell(std::ostream& out, const PtrT& ptr, size_t row) {
	using T = typename PtrT::element_type;
	if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
		out << static_cast<int>(ptr[row]);
	} else {
		out << static_cast<int64_t>(ptr[row]);
	}
}

void write_row(std::ostream&                             out,
               const std::vector<size_t>&                col_indices,
               const dispatch::RowgroupDecompressResult& result,
               const size_t                              row) {
	for (size_t ci = 0; ci < col_indices.size(); ++ci) {
		if (ci > 0) {
			out << "|";
		}
		const auto  col_idx = col_indices[ci];
		const auto& opt     = result.columns[col_idx];
		if (!opt.has_value()) {
			continue;
		}
		std::visit([&](const auto& ptr) { write_cell(out, ptr, row); }, *opt);
	}
	out << "\n";
}

__global__ void empty_kernel() {
}

double measure_gpu_launch_us(uint32_t iters, dim3 grid, dim3 block) {
	// warmup
	for (int i = 0; i < 100; ++i) {
		empty_kernel<<<grid, block>>>();
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEvent_t start {};
	cudaEvent_t stop {};
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));
	for (uint32_t i = 0; i < iters; ++i) {
		empty_kernel<<<grid, block>>>();
	}
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));

	return (ms * 1000.0) / static_cast<double>(iters);
}

double measure_cpu_launch_us(uint32_t iters, dim3 grid, dim3 block) {
	for (int i = 0; i < 100; ++i) {
		empty_kernel<<<grid, block>>>();
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	const auto t0 = std::chrono::steady_clock::now();
	for (uint32_t i = 0; i < iters; ++i) {
		empty_kernel<<<grid, block>>>();
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	const auto t1 = std::chrono::steady_clock::now();

	const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	return (ms * 1000.0) / static_cast<double>(iters);
}

} // namespace

int main(int argc, char** argv) {
	Options opt;
	if (!parse_args(argc, argv, opt)) {
		print_usage(argv[0]);
		return 1;
	}

	try {
		if (opt.mode == Mode::MeasureLaunch) {
			const uint32_t iters = opt.launch_iters;
			const dim3     grid(opt.launch_grid);
			const dim3     block(opt.launch_block);
			const double   gpu_us = measure_gpu_launch_us(iters, grid, block);
			const double   cpu_us = measure_cpu_launch_us(iters, grid, block);
			std::cout << "Launch overhead (" << iters << " iters, grid=" << opt.launch_grid
			          << ", block=" << opt.launch_block << "):\n";
			std::cout << "  gpu_event_us: " << gpu_us << "\n";
			std::cout << "  cpu_wall_us:  " << cpu_us << "\n";
			return 0;
		}

		reader::reader rdr(opt.input);
		const size_t   n_rowgroups = rdr.rowgroup_count();

		size_t start = 0;
		size_t end   = n_rowgroups;
		if (opt.rowgroup.has_value()) {
			if (*opt.rowgroup >= n_rowgroups) {
				throw std::out_of_range("rowgroup index out of range");
			}
			start = *opt.rowgroup;
			end   = start + 1;
		}

		if (opt.mode == Mode::ReadTable) {
			std::ofstream out_file;
			std::ostream* out = &std::cout;
			if (opt.output.has_value() && *opt.output != "-") {
				out_file.open(*opt.output);
				if (!out_file) {
					throw std::runtime_error("failed to open output file");
				}
				out = &out_file;
			}

			bool header_written = false;

			for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
				auto rowgroup    = rdr.read_rowgroup(rg_idx);
				auto expressions = expr::assemble(rowgroup);
				auto result      = dispatch::decompress_rowgroup(expressions);

				std::vector<size_t>      col_indices;
				std::vector<std::string> col_names;
				for (size_t i = 0; i < expressions.size(); ++i) {
					const auto& expr = expressions[i];
					if (!expr.column || expr.column->skip_decompress) {
						continue;
					}
					col_indices.push_back(i);
					auto name = expr.column->name;
					if (name.empty()) {
						name = "col_" + std::to_string(i);
					}
					col_names.push_back(std::move(name));
				}

				if (opt.header && !header_written) {
					for (size_t i = 0; i < col_names.size(); ++i) {
						if (i > 0) {
							*out << ",";
						}
						*out << col_names[i];
					}
					*out << "\n";
					header_written = true;
				}

				const size_t n_values = rowgroup.n_values;
				for (size_t row = 0; row < n_values; ++row) {
					write_row(*out, col_indices, result, row);
				}

				pipeline::free_rowgroup(rowgroup);
			}
			return 0;
		}

		if (opt.mode == Mode::Benchmark) {
			double end_to_end_ms     = 0.0;
			double kernel_ms         = 0.0;
			double setup_ms          = 0.0;
			double h2d_ms            = 0.0;
			double teardown_ms       = 0.0;
			size_t total_launches    = 0;
			size_t total_launch_grid = 0;
			size_t total_columns     = 0;
			size_t total_items       = 0;
			size_t total_bytes       = 0;
			size_t total_rgs         = 0;

			const auto  td_handle = load_table_descriptor(opt.input);
			const auto* td        = td_handle.Get();
			if (!td) {
				throw std::runtime_error("failed to load table descriptor");
			}

			const bool use_mega_kernel = (!opt.rowgroup.has_value() && opt.mega_kernel);
			if (!use_mega_kernel) {
				for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
					const auto*  rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
					const double io_ms = measure_rowgroup_io_ms(opt.input, rg);

					size_t rg_bytes = 0;
					if (rg) {
						const auto* cols = rg->m_column_descriptors();
						if (cols) {
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
						}
					}

					const auto setup_start = std::chrono::steady_clock::now();
					auto       rowgroup    = rdr.read_rowgroup(rg_idx);
					auto       expressions = expr::assemble(rowgroup);
					const auto setup_end   = std::chrono::steady_clock::now();

					const double setup_total_ms =
					    std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
					double setup = setup_total_ms - io_ms;
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

					using Batches = typename DeviceBatchSetFromList<dispatch::SupportedTypes>::type;
					Batches      batches;
					const double rg_h2d_ms      = build_gpu_batches(expressions, batches);
					size_t       rg_launches    = 0;
					size_t       rg_launch_grid = 0;
					dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
						using T     = typename decltype(tag)::type;
						auto& batch = batches.template get<T>();
						if (!batch.batch.device_exprs.empty() && !batch.batch.work_items.empty()) {
							++rg_launches;
							rg_launch_grid += batch.batch.work_items.size();
						}
					});
					const double rg_kernel_ms   = run_gpu_kernels(batches, opt.samples);
					const auto   teardown_start = std::chrono::steady_clock::now();
					free_gpu_batches(batches);
					const auto   teardown_end = std::chrono::steady_clock::now();
					const double rg_teardown_ms =
					    std::chrono::duration<double, std::milli>(teardown_end - teardown_start).count();

					end_to_end_ms += setup + rg_h2d_ms + rg_kernel_ms;
					kernel_ms += rg_kernel_ms;
					setup_ms += setup;
					h2d_ms += rg_h2d_ms;
					teardown_ms += rg_teardown_ms;
					total_launches += rg_launches * static_cast<size_t>(opt.samples);
					total_launch_grid += rg_launch_grid * static_cast<size_t>(opt.samples);
					total_columns += rg_columns;
					total_items += rg_vectors;
					total_bytes += rg_bytes;
					++total_rgs;

					pipeline::free_rowgroup(rowgroup);
				}
			} else {
				dispatch::table::TableBatches table_batches;

				for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
					const auto*  rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
					const double io_ms = measure_rowgroup_io_ms(opt.input, rg);

					size_t rg_bytes = 0;
					if (rg) {
						const auto* cols = rg->m_column_descriptors();
						if (cols) {
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
						}
					}

					const auto setup_start = std::chrono::steady_clock::now();
					auto       rowgroup    = rdr.read_rowgroup(rg_idx);
					auto       expressions = expr::assemble(rowgroup);
					const auto setup_end   = std::chrono::steady_clock::now();

					const double setup_total_ms =
					    std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
					double setup = setup_total_ms - io_ms;
					if (setup < 0.0) {
						setup = 0.0;
					}
					setup_ms += setup;

					size_t rg_columns = 0;
					for (const auto& expr : expressions) {
						if (expr.column && !expr.column->skip_decompress) {
							++rg_columns;
						}
					}
					const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

					h2d_ms += dispatch::table::append_expressions(table_batches, expressions);

					total_columns += rg_columns;
					total_items += rg_vectors;
					total_bytes += rg_bytes;
					++total_rgs;

					pipeline::free_rowgroup(rowgroup);
				}

				h2d_ms += dispatch::table::finalize_batches(table_batches);

				size_t       launch_grid = 0;
				const size_t n_items     = table_batches.work_items.size();
				kernel_ms                = dispatch::table::run_kernel(table_batches, opt.samples, &launch_grid);
				total_launches           = (n_items > 0) ? static_cast<size_t>(opt.samples) : 0;
				total_launch_grid        = (n_items > 0) ? (n_items * static_cast<size_t>(opt.samples)) : 0;

				const auto teardown_start = std::chrono::steady_clock::now();
				dispatch::table::free_batches(table_batches);
				table_batches.d_items.reset();
				const auto teardown_end = std::chrono::steady_clock::now();
				teardown_ms += std::chrono::duration<double, std::milli>(teardown_end - teardown_start).count();

				end_to_end_ms = setup_ms + h2d_ms + kernel_ms;
			}

			const double total_samples = (total_rgs > 0) ? (static_cast<double>(opt.samples) * total_rgs) : 0.0;
			const double avg_us        = (total_samples > 0.0) ? (kernel_ms * 1000.0 / total_samples) : 0.0;
			const double total_bytes_processed =
			    (total_rgs > 0) ? (static_cast<double>(total_bytes) * static_cast<double>(opt.samples)) : 0.0;
			const double kernel_seconds = kernel_ms / 1000.0;
			const double kernel_throughput_bps =
			    (kernel_seconds > 0.0) ? (total_bytes_processed / kernel_seconds) : 0.0;
			const double kernel_throughput_gbps = kernel_throughput_bps / 1e9;
			const double kernel_throughput_gibps =
			    (kernel_seconds > 0.0) ? (total_bytes_processed / (1024.0 * 1024.0 * 1024.0 * kernel_seconds)) : 0.0;
			const double e2e_no_teardown_seconds = end_to_end_ms / 1000.0;
			const double e2e_no_teardown_bps =
			    (e2e_no_teardown_seconds > 0.0) ? (total_bytes_processed / e2e_no_teardown_seconds) : 0.0;
			const double e2e_no_teardown_gbps = e2e_no_teardown_bps / 1e9;
			const double e2e_no_teardown_gibps =
			    (e2e_no_teardown_seconds > 0.0)
			        ? (total_bytes_processed / (1024.0 * 1024.0 * 1024.0 * e2e_no_teardown_seconds))
			        : 0.0;

			const double end_to_end_with_teardown_ms = end_to_end_ms + teardown_ms;
			const double e2e_with_teardown_seconds   = end_to_end_with_teardown_ms / 1000.0;
			const double e2e_with_teardown_bps =
			    (e2e_with_teardown_seconds > 0.0) ? (total_bytes_processed / e2e_with_teardown_seconds) : 0.0;
			const double e2e_with_teardown_gbps = e2e_with_teardown_bps / 1e9;
			const double e2e_with_teardown_gibps =
			    (e2e_with_teardown_seconds > 0.0)
			        ? (total_bytes_processed / (1024.0 * 1024.0 * 1024.0 * e2e_with_teardown_seconds))
			        : 0.0;
			const double cols_per_rg =
			    (total_rgs > 0) ? (static_cast<double>(total_columns) / static_cast<double>(total_rgs)) : 0.0;
			const double vectors_per_rg =
			    (total_rgs > 0) ? (static_cast<double>(total_items) / static_cast<double>(total_rgs)) : 0.0;
			const double avg_grid_per_launch =
			    (total_launches > 0) ? (static_cast<double>(total_launch_grid) / static_cast<double>(total_launches))
			                         : 0.0;

			std::cout << "Benchmark results:\n";
			std::cout << "  rowgroups: " << total_rgs << "\n";
			std::cout << "  columns:   " << total_columns << " (avg " << cols_per_rg << " per rowgroup)\n";
			std::cout << "  vectors: " << total_items << " (avg " << vectors_per_rg << " per rowgroup)\n";
			std::cout << "  samples:   " << opt.samples << "\n";
			std::cout << "  bytes:     " << total_bytes << " (" << format_bytes(static_cast<double>(total_bytes))
			          << ")\n";
			std::cout << "  end_to_end_ms: " << end_to_end_with_teardown_ms << "\n";
			std::cout << "  end_to_end_ms (no teardown): " << end_to_end_ms << "\n";
			std::cout << "  kernel_ms:     " << kernel_ms << "\n";
			std::cout << "  setup_ms:      " << setup_ms << "\n";
			std::cout << "  h2d_ms:        " << h2d_ms << "\n";
			std::cout
			    << "  teardown_ms:   " << teardown_ms
			    << "\n"; //  cudaFree on device buffers, any implicit synchronization caused by freeing those buffers
			std::cout << "  avg_us:    " << avg_us << " (per rowgroup per sample)\n";
			std::cout << "  kernel_throughput: " << kernel_throughput_gbps << " (GB/s), " << kernel_throughput_gibps
			          << " (GiB/s)\n";
			std::cout << "  end_to_end_throughput: " << e2e_with_teardown_gbps << " (GB/s), " << e2e_with_teardown_gibps
			          << " (GiB/s)\n";
			std::cout << "  end_to_end_throughput (no teardown): " << e2e_no_teardown_gbps << " (GB/s), "
			          << e2e_no_teardown_gibps << " (GiB/s)\n";
			std::cout << "  kernel_launches: " << total_launches << "\n";
			std::cout << "  avg_grid_per_launch: " << avg_grid_per_launch << "\n";
			if (opt.estimate_launch && total_launches > 0) {
				const uint32_t block = utils::get_n_lanes<int8_t>();
				uint32_t       grid  = static_cast<uint32_t>(avg_grid_per_launch);
				if (grid == 0) {
					grid = 1;
				}
				const double launch_us          = measure_gpu_launch_us(opt.estimate_iters, dim3(grid), dim3(block));
				const double launch_overhead_ms = (launch_us * static_cast<double>(total_launches)) / 1000.0;
				const double launch_pct         = (kernel_ms > 0.0) ? (launch_overhead_ms * 100.0 / kernel_ms) : 0.0;
				std::cout << "  launch_overhead_ms (est): " << launch_overhead_ms << "\n";
				std::cout << "  launch_overhead_pct (est): " << launch_pct << "%\n";
				std::cout << "  launch_overhead_us (per kernel, est): " << launch_us << "\n";
			}
			return 0;
		}

		return 1;
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
