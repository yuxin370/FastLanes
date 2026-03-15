// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/galp_cli.cu
// ────────────────────────────────────────────────────────
#include "engine/execution/common.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/execution/table.cuh"
#include "engine/expression.cuh"
#include "engine/io/to_csv.cuh"
#include "engine/reader.cuh"
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
	bool                                 gpu_dispatch_kernel          = false;
	bool                                 write_back                   = false;
	bool                                 freq_prefetch_all_branchless = false;
	bool                                 freq_hybrid_patcher          = false;
	float                                freq_branchless_threshold    = 6.0f;
};

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
		          << "  --no-mega-kernel   Benchmark full table using per-rowgroup kernels\n"
		          << "  --gpu-dispatch-kernel  Use one mixed-type kernel launch per sample in mega mode\n"
		          << "  --write-back   Enable global write-back during benchmark kernel execution\n"
		          << "  --freq-prefetch-all-branchless  Use FREQ extended format + PrefetchAllBranchless patcher\n"
		          << "  --freq-hybrid-patcher  Use hybrid FREQ patcher selection by exception density\n"
		          << "  --freq-branchless-threshold N  Hybrid threshold: avg exceptions per vec (default: 6)\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
	if (argc < 2) {
		return false;
	}

	std::string_view mode_arg = argv[1];
	if (mode_arg == "read_table" || mode_arg == "read") {
		opt.mode = Mode::ReadTable;
		opt.gpu_dispatch_kernel = true;
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
		if (arg == "--gpu-dispatch-kernel") {
			opt.gpu_dispatch_kernel = true;
			continue;
		}
		if (arg == "--write-back") {
			opt.write_back = true;
			continue;
		}
		if (arg == "--freq-prefetch-all-branchless") {
			opt.freq_prefetch_all_branchless = true;
			continue;
		}
		if (arg == "--freq-hybrid-patcher") {
			opt.freq_hybrid_patcher = true;
			continue;
		}
		if (arg == "--freq-branchless-threshold" && i + 1 < argc) {
			opt.freq_branchless_threshold = std::stof(argv[++i]);
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

			dispatch::Config decode_cfg {};
			decode_cfg.gpu_dispatch_kernel = opt.gpu_dispatch_kernel;
			io::read_rowgroups_to_csv(rdr, *out, start, end, opt.header, decode_cfg);
			return 0;
		}

		if (opt.mode == Mode::Benchmark) {
			dispatch::TableBenchmarkConfig bench_cfg {};
			bench_cfg.samples                      = opt.samples;
			bench_cfg.mega_kernel                  = opt.mega_kernel;
			bench_cfg.gpu_dispatch_kernel          = opt.gpu_dispatch_kernel;
			bench_cfg.write_out                    = opt.write_back;
			bench_cfg.freq_prefetch_all_branchless = opt.freq_prefetch_all_branchless;
			bench_cfg.freq_hybrid_patcher          = opt.freq_hybrid_patcher;
			bench_cfg.freq_branchless_threshold    = opt.freq_branchless_threshold;
			bench_cfg.rowgroup                     = opt.rowgroup;
			const auto result     = dispatch::benchmark_table(opt.input, bench_cfg);

			const double end_to_end_ms     = result.end_to_end_ms;
			const double kernel_ms         = result.kernel_ms;
			const double setup_ms          = result.setup_ms;
			const double h2d_ms            = result.h2d_ms;
			const double teardown_ms       = result.teardown_ms;
			const size_t total_launches    = result.total_launches;
			const size_t total_launch_grid = result.total_launch_grid;
			const size_t total_columns     = result.total_columns;
			const size_t total_items       = result.total_items;
			const size_t total_bytes       = result.total_bytes;
			const size_t total_rgs         = result.total_rgs;

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
				std::cout << "  gpu_dispatch_kernel: " << (opt.gpu_dispatch_kernel ? 1 : 0) << "\n";
				std::cout << "  write_back: " << (opt.write_back ? 1 : 0) << "\n";
				std::cout << "  freq_prefetch_all_branchless: " << (opt.freq_prefetch_all_branchless ? 1 : 0) << "\n";
				std::cout << "  freq_hybrid_patcher: " << (opt.freq_hybrid_patcher ? 1 : 0) << "\n";
				std::cout << "  freq_branchless_threshold: " << opt.freq_branchless_threshold << "\n";
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
