// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/galp_cli/galp_cli.cu
// ────────────────────────────────────────────────────────
#include "core/expression.cuh"
#include "engine/config.cuh"
#include "engine/operators/rowgroup.cuh"
#include "engine/table/table_options.cuh"
#include "format/reader.cuh"
#include "galp_tools/benchmark_support/table.cuh"
#include "io/csv_writer.cuh"
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
	uint32_t                             samples                    = 5;
	uint32_t                             kernel_samples             = 1;
	bool                                 header                     = true;
	uint32_t                             launch_iters               = 100000;
	uint32_t                             launch_grid                = 1;
	uint32_t                             launch_block               = 1;
	bool                                 per_rowgroup_workset       = false;
	bool                                 mixed_dispatch             = true;
	uint32_t                             unpack_n_vectors           = 1;
	uint32_t                             unpack_n_values            = 1;
	galp::execution::FreqPatcher         freq_patcher               = galp::execution::FreqPatcher::Stateful;
	float                                freq_branchless_threshold  = galp::execution::kFreqHybridBranchlessThreshold;
	bool                                 enable_rowgroup_prefetch   = true;
	bool                                 include_materialize        = false;
	bool                                 reuse_table_resources      = false;
	size_t                               prefetch_depth             = 4;
	size_t                               prefetch_workers           = 0;
	size_t                               max_prefetch_storage_bytes = 0;
	size_t                               stream_target_work_items   = 1u << 18;
	size_t                               stream_max_rowgroups       = 4;
	size_t                               compute_inflight_chunks    = 0;
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

const char* freq_patcher_name(const galp::execution::FreqPatcher patcher) {
	switch (patcher) {
	case galp::execution::FreqPatcher::Stateful:
		return "stateful";
	case galp::execution::FreqPatcher::Branchless:
		return "branchless";
	case galp::execution::FreqPatcher::Hybrid:
		return "hybrid";
	}
	return "unknown";
}

size_t effective_prefetch_workers(const size_t requested, const size_t rowgroups, const size_t max_rowgroups_per_chunk) {
	if (requested != 0) {
		return std::max<size_t>(1, requested);
	}
	if (max_rowgroups_per_chunk >= 4) {
		return 2;
	}
	if (rowgroups >= 128) {
		return 4;
	}
	if (rowgroups >= 32) {
		return 3;
	}
	return 2;
}

struct MetricStats {
	double min    = 0.0;
	double median = 0.0;
	double mean   = 0.0;
};

template <typename Getter>
MetricStats summarize_metric(const std::vector<galp::execution::TableBenchmarkResult>& results, Getter getter) {
	if (results.empty()) {
		return {};
	}
	std::vector<double> values;
	values.reserve(results.size());
	double sum = 0.0;
	for (const auto& result : results) {
		const double value = getter(result);
		values.push_back(value);
		sum += value;
	}
	std::sort(values.begin(), values.end());
	return MetricStats {
	    values.front(),
	    values[values.size() / 2],
	    sum / static_cast<double>(values.size()),
	};
}

size_t median_result_index(const std::vector<galp::execution::TableBenchmarkResult>& results) {
	if (results.empty()) {
		return 0;
	}
	std::vector<size_t> indices(results.size());
	for (size_t i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}
	std::sort(indices.begin(), indices.end(), [&](const size_t lhs, const size_t rhs) {
		return results[lhs].end_to_end_ms < results[rhs].end_to_end_ms;
	});
	return indices[indices.size() / 2];
}

bool parse_freq_patcher(std::string_view value, Options& opt) {
	std::optional<float> threshold;
	const auto           colon = value.find(':');
	if (colon != std::string_view::npos) {
		try {
			threshold = std::stof(std::string(value.substr(colon + 1)));
		} catch (...) { return false; }
		value = value.substr(0, colon);
	}

	if (value == "stateful") {
		opt.freq_patcher = galp::execution::FreqPatcher::Stateful;
	} else if (value == "branchless") {
		opt.freq_patcher = galp::execution::FreqPatcher::Branchless;
	} else if (value == "hybrid") {
		opt.freq_patcher = galp::execution::FreqPatcher::Hybrid;
	} else {
		return false;
	}

	if (threshold.has_value()) {
		if (opt.freq_patcher != galp::execution::FreqPatcher::Hybrid) {
			return false;
		}
		opt.freq_branchless_threshold = *threshold;
	}
	return true;
}

void print_usage(const char* prog) {
	std::cerr
	    << "Usage:\n"
	    << "  " << prog
	    << " read_table <input.fls> [output.csv] [--rowgroup N] [--no-header] [--per-rowgroup-workset]\n"
	    << "  " << prog
	    << " benchmark <input.fls> [--rowgroup N] [--samples N] [--kernel-samples N] [--include-materialize]\n"
	    << "  " << prog << " measure_launch [--iters N] [--grid N] [--block N]\n"
	    << "\n"
	    << "Options:\n"
	    << "  --rowgroup N   Only process the given rowgroup\n"
	    << "  --samples N    Number of independent benchmark samples for min/median/mean (default: 5)\n"
	    << "  --kernel-samples N  Kernel replays inside each benchmark sample (default: 1)\n"
	    << "  --no-header    Skip CSV header\n"
	    << "  --out PATH     Output CSV path (read_table mode)\n"
	    << "  --iters N      Launch measurement iterations (default: 100000)\n"
	    << "  --grid N       Launch grid size for measurement (default: 1)\n"
	    << "  --block N      Launch block size for measurement (default: 1)\n"
	    << "  --per-rowgroup-workset  Process each rowgroup as an independent workset\n"
	    << "  --no-mixed-dispatch  Use typed-batch launches instead of mixed-dispatch (default: mixed)\n"
	    << "  --unpack-n-vectors N  Runtime decode tile size in vectors (supported: 1 or 4)\n"
	    << "  --unpack-n-values N   Runtime decode tile size in values (currently only 1)\n"
	    << "  --no-rowgroup-prefetch  Disable background rowgroup prefetch in whole-table execution\n"
	    << "  --prefetch-depth N  Number of prefetched rowgroups to queue ahead (default: 4)\n"
	    << "  --prefetch-workers N  Number of parallel prefetch threads (default: 0=auto)\n"
	    << "  --max-prefetch-storage-bytes N  Fused prefetch compressed-byte budget "
	       "(default: prefetch_depth * max rowgroup bytes)\n"
	    << "  --stream-target-work-items N  Chunk flush threshold by work_items in whole-table execution "
	       "(default: 262144)\n"
	    << "  --stream-max-rowgroups N  Chunk flush threshold by rowgroups in whole-table execution (default: 4, "
	       "0 disables)\n"
	    << "  --compute-inflight-chunks N  Whole-table compute chunks allowed in flight (default: 0=auto, max: 8)\n"
	    << "  --include-materialize  Benchmark also materializes results to host pinned memory (D2H included)\n"
	    << "  --reuse-table-resources  Benchmark steady-state query after reader/pinned-pool prepare\n"
	    << "  --freq-patcher MODE  FREQ patcher: stateful, branchless, or hybrid[:threshold] (default: stateful)\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
	if (argc < 2) {
		return false;
	}

	std::string_view mode_arg = argv[1];
	if (mode_arg == "read_table") {
		opt.mode = Mode::ReadTable;
	} else if (mode_arg == "benchmark") {
		opt.mode = Mode::Benchmark;
	} else if (mode_arg == "measure_launch") {
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
		if (arg == "--kernel-samples" && i + 1 < argc) {
			opt.kernel_samples = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--per-rowgroup-workset") {
			opt.per_rowgroup_workset = true;
			continue;
		}
		if (arg == "--no-mixed-dispatch") {
			opt.mixed_dispatch = false;
			continue;
		}
		if (arg == "--unpack-n-vectors" && i + 1 < argc) {
			opt.unpack_n_vectors = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--unpack-n-values" && i + 1 < argc) {
			opt.unpack_n_values = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--no-rowgroup-prefetch") {
			opt.enable_rowgroup_prefetch = false;
			continue;
		}
		if (arg == "--prefetch-depth" && i + 1 < argc) {
			opt.prefetch_depth = static_cast<size_t>(std::stoull(argv[++i]));
			continue;
		}
		if (arg == "--prefetch-workers" && i + 1 < argc) {
			opt.prefetch_workers = static_cast<size_t>(std::stoull(argv[++i]));
			continue;
		}
		if (arg == "--max-prefetch-storage-bytes" && i + 1 < argc) {
			opt.max_prefetch_storage_bytes = static_cast<size_t>(std::stoull(argv[++i]));
			continue;
		}
		if (arg == "--stream-target-work-items" && i + 1 < argc) {
			opt.stream_target_work_items = static_cast<size_t>(std::stoull(argv[++i]));
			continue;
		}
		if (arg == "--stream-max-rowgroups" && i + 1 < argc) {
			opt.stream_max_rowgroups = static_cast<size_t>(std::stoull(argv[++i]));
			continue;
		}
		if (arg == "--compute-inflight-chunks" && i + 1 < argc) {
			opt.compute_inflight_chunks = static_cast<size_t>(std::stoull(argv[++i]));
			continue;
		}
		if (arg == "--include-materialize") {
			opt.include_materialize = true;
			continue;
		}
		if (arg == "--reuse-table-resources") {
			opt.reuse_table_resources = true;
			continue;
		}
		if (arg == "--freq-patcher" && i + 1 < argc) {
			if (!parse_freq_patcher(argv[++i], opt)) {
				return false;
			}
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

galp::execution::TableDecompressionScope table_scope_from_options(const Options& opt) {
	return opt.per_rowgroup_workset ? galp::execution::TableDecompressionScope::PerRowgroup
	                                : galp::execution::TableDecompressionScope::WholeTable;
}

template <typename Config>
void apply_table_options(Config& cfg, const Options& opt) {
	cfg.scope                               = table_scope_from_options(opt);
	cfg.execution.unpack_n_vectors          = opt.unpack_n_vectors;
	cfg.execution.unpack_n_values           = opt.unpack_n_values;
	cfg.execution.launch_strategy           = opt.mixed_dispatch ? galp::execution::LaunchStrategy::MixedDispatch
	                                                             : galp::execution::LaunchStrategy::TypedBatches;
	cfg.execution.freq_patcher              = opt.freq_patcher;
	cfg.execution.freq_branchless_threshold = opt.freq_branchless_threshold;
	galp::execution::apply_table_streaming_options(cfg,
	                                               opt.enable_rowgroup_prefetch,
	                                               opt.prefetch_depth,
	                                               opt.prefetch_workers,
	                                               opt.max_prefetch_storage_bytes,
	                                               opt.stream_target_work_items,
	                                               opt.stream_max_rowgroups,
	                                               opt.compute_inflight_chunks);
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

			galp::execution::TableDecompressionConfig decode_cfg {};
			apply_table_options(decode_cfg, opt);
			galp::io::read_table_to_csv(opt.input, *out, opt.header, decode_cfg, opt.rowgroup);
			return 0;
		}

		if (opt.mode == Mode::Benchmark) {
			const uint32_t                        benchmark_samples = std::max<uint32_t>(1, opt.samples);
			const uint32_t                        kernel_samples    = std::max<uint32_t>(1, opt.kernel_samples);
			galp::execution::TableBenchmarkConfig bench_cfg {};
			bench_cfg.samples = kernel_samples;
			apply_table_options(bench_cfg, opt);
			bench_cfg.execution.write_out   = opt.include_materialize;
			bench_cfg.rowgroup              = opt.rowgroup;
			bench_cfg.include_materialize   = opt.include_materialize;
			bench_cfg.reuse_table_resources = opt.reuse_table_resources;

			std::vector<galp::execution::TableBenchmarkResult> results;
			results.reserve(benchmark_samples);
			for (uint32_t sample_idx = 0; sample_idx < benchmark_samples; ++sample_idx) {
				results.push_back(galp::execution::benchmark_table(opt.input, bench_cfg));
			}

			const auto& result          = results[median_result_index(results)];
			const auto  benchmark_stats = summarize_metric(results, [](const auto& r) { return r.end_to_end_ms; });
			const auto  query_stats     = summarize_metric(results, [](const auto& r) { return r.query_wall_ms; });
			const auto  kernel_stats    = summarize_metric(results, [](const auto& r) { return r.kernel_ms; });

			const double benchmark_wall_ms         = result.end_to_end_ms;
			const double resource_prepare_ms       = result.resource_prepare_ms;
			const double query_wall_ms             = result.query_wall_ms;
			const double pipeline_active_ms        = result.pipeline_active_ms;
			const double read_rowgroup_ms          = result.read_rowgroup_ms;
			const double file_read_ms              = result.file_read_ms;
			const double rowgroup_build_ms         = result.rowgroup_build_ms;
			const double pinned_acquire_ms         = result.pinned_acquire_ms;
			const double pread_ms                  = result.pread_ms;
			const double zero_copy_view_setup_ms   = result.zero_copy_view_setup_ms;
			const double prefetch_depth_block_ms   = result.prefetch_depth_block_ms;
			const double prefetch_byte_block_ms    = result.prefetch_byte_block_ms;
			const double read_wall_ms              = result.read_wall_ms;
			const double pread_wall_ms             = result.pread_wall_ms;
			const double file_read_wall_ms         = result.file_read_wall_ms;
			const double assemble_expr_ms          = result.assemble_expr_ms;
			const double append_expr_ms            = result.append_expr_ms;
			const double upload_workset_ms         = result.upload_workset_ms;
			const double run_submit_wall_ms        = result.run_submit_wall_ms;
			const double wait_workset_wall_ms      = result.wait_workset_wall_ms;
			const double event_sync_wall_ms        = result.event_sync_wall_ms;
			const double pre_kernel_event_ms       = result.pre_kernel_event_ms;
			const double warmup_wall_ms            = result.warmup_wall_ms;
			const double timing_event_create_ms    = result.timing_event_create_ms;
			const double timing_event_destroy_ms   = result.timing_event_destroy_ms;
			const double kernel_event_ms           = result.kernel_ms;
			const double release_device_ms         = result.release_device_ms;
			const double free_rowgroup_ms          = result.free_rowgroup_ms;
			const double prefetch_wait_ms          = result.prefetch_wait_ms;
			const double reader_open_ms            = result.reader_open_ms;
			const double descriptor_load_ms        = result.descriptor_load_ms;
			const double pinned_pool_create_ms     = result.pinned_pool_create_ms;
			const double max_storage_scan_ms       = result.max_storage_scan_ms;
			const double pinned_pool_prewarm_ms    = result.pinned_pool_prewarm_ms;
			const double prefetch_queue_start_ms   = result.prefetch_queue_start_ms;
			const double pipeline_setup_total_ms   = result.pipeline_setup_total_ms;
			const size_t total_launches            = result.total_launches;
			const size_t total_launch_grid         = result.total_launch_grid;
			const size_t total_columns             = result.total_columns;
			const size_t total_items               = result.total_items;
			const size_t total_bytes               = result.total_bytes;
			const size_t total_storage_bytes       = result.total_storage_bytes;
			const size_t total_payload_arena_bytes = result.total_payload_arena_bytes;
			const size_t total_output_arena_bytes  = result.total_output_arena_bytes;
			const size_t total_h2d_bytes           = result.total_h2d_bytes;
			const size_t total_h2d_copies          = result.total_h2d_copies;
			const size_t prefetched_rowgroups      = result.prefetched_rowgroups;
			const size_t compute_inflight_chunks   = result.compute_inflight_chunks;
			const size_t rowgroup_prefetch_depth   = result.rowgroup_prefetch_depth;
			const size_t wait_ready_chunks         = result.wait_ready_chunks;
			const size_t wait_blocking_chunks      = result.wait_blocking_chunks;
			const size_t total_rgs                 = result.total_rgs;
			const bool   whole_table_prefetch      = opt.enable_rowgroup_prefetch &&
			                                  bench_cfg.scope == galp::execution::TableDecompressionScope::WholeTable &&
			                                  !opt.rowgroup.has_value() && total_rgs != 0;
			const size_t effective_stream_max_rowgroups =
			    opt.stream_max_rowgroups > 0 ? opt.stream_max_rowgroups : std::max<size_t>(1, total_rgs);
			const size_t actual_prefetch_workers =
			    whole_table_prefetch
			        ? std::min(
			              effective_prefetch_workers(opt.prefetch_workers, total_rgs, effective_stream_max_rowgroups),
			              total_rgs)
			        : 0;
			const double avg_grid_per_launch =
			    (total_launches > 0) ? (static_cast<double>(total_launch_grid) / static_cast<double>(total_launches))
			                         : 0.0;
			const double storage_read_gbps =
			    (pread_wall_ms > 0.0) ? (static_cast<double>(total_storage_bytes) / (pread_wall_ms / 1000.0) / 1.0e9)
			                          : 0.0;

			std::cout << "Benchmark results:\n";
			std::cout << "  rowgroups: " << total_rgs << "\n";
			std::cout << "  columns: " << total_columns << "\n";
			std::cout << "  vectors: " << total_items << "\n";
			std::cout << "  samples: " << benchmark_samples << "\n";
			std::cout << "  kernel_samples: " << kernel_samples << "\n";
			std::cout << "  bytes: " << total_bytes << " (" << format_bytes(static_cast<double>(total_bytes)) << ")\n";
			std::cout << "  storage_bytes: " << total_storage_bytes << " ("
			          << format_bytes(static_cast<double>(total_storage_bytes)) << ")\n";
			std::cout << "  benchmark_wall_ms: " << benchmark_wall_ms << "\n";
			std::cout << "  benchmark_wall_ms_min: " << benchmark_stats.min << "\n";
			std::cout << "  benchmark_wall_ms_median: " << benchmark_stats.median << "\n";
			std::cout << "  benchmark_wall_ms_mean: " << benchmark_stats.mean << "\n";
			std::cout << "  resource_prepare_ms: " << resource_prepare_ms << "\n";
			std::cout << "  query_wall_ms: " << query_wall_ms << "\n";
			std::cout << "  query_wall_ms_min: " << query_stats.min << "\n";
			std::cout << "  query_wall_ms_median: " << query_stats.median << "\n";
			std::cout << "  query_wall_ms_mean: " << query_stats.mean << "\n";
			std::cout << "  pipeline_active_ms: " << pipeline_active_ms << "\n";
			std::cout << "  read_rowgroup_ms: " << read_rowgroup_ms << "\n";
			std::cout << "  file_read_ms: " << file_read_ms << "\n";
			std::cout << "  rowgroup_build_ms: " << rowgroup_build_ms << "\n";
			std::cout << "  pinned_acquire_ms: " << pinned_acquire_ms << "\n";
			std::cout << "  pread_ms: " << pread_ms << "\n";
			std::cout << "  pread_wall_ms: " << pread_wall_ms << "\n";
			std::cout << "  zero_copy_view_setup_ms: " << zero_copy_view_setup_ms << "\n";
			std::cout << "  read_wall_ms: " << read_wall_ms << "\n";
			std::cout << "  file_read_wall_ms: " << file_read_wall_ms << "\n";
			std::cout << "  storage_read_gbps: " << storage_read_gbps << "\n";
			std::cout << "  assemble_expr_ms: " << assemble_expr_ms << "\n";
			std::cout << "  append_expr_ms: " << append_expr_ms << "\n";
			std::cout << "  upload_workset_ms: " << upload_workset_ms << "\n";
			std::cout << "    upload_prep_ms: " << result.upload_prep_ms << "\n";
			std::cout << "      upload_prep_reset_ms: " << result.upload_prep_reset_ms << "\n";
			std::cout << "      upload_prep_output_arena_ms: " << result.upload_prep_output_arena_ms << "\n";
			std::cout << "      upload_prep_bind_ms: " << result.upload_prep_bind_ms << "\n";
			std::cout << "      upload_prep_slots_ms: " << result.upload_prep_slots_ms << "\n";
			std::cout << "    upload_arena_pack_ms: " << result.upload_arena_pack_ms << "\n";
			std::cout << "    upload_layout_ms: " << result.upload_layout_ms << "\n";
			std::cout << "    upload_alloc_ms: " << result.upload_alloc_ms << "\n";
			std::cout << "    upload_resolve_ms: " << result.upload_resolve_ms << "\n";
			std::cout << "    upload_pack_ms: " << result.upload_pack_ms << "\n";
			std::cout << "    upload_dma_issue_ms: " << result.upload_dma_issue_ms << "\n";
			std::cout << "    upload_dma_gpu_ms: " << result.upload_dma_gpu_ms << "\n";
			std::cout << "    upload_event_ms: " << result.upload_event_ms << "\n";
			std::cout << "  run_submit_wall_ms: " << run_submit_wall_ms << "\n";
			std::cout << "    timing_event_create_ms: " << timing_event_create_ms << "\n";
			std::cout << "    warmup_wall_ms: " << warmup_wall_ms << "\n";
			std::cout << "  wait_workset_wall_ms: " << wait_workset_wall_ms << "\n";
			std::cout << "    event_sync_wall_ms: " << event_sync_wall_ms << "\n";
			std::cout << "    pre_kernel_event_ms: " << pre_kernel_event_ms << "\n";
			std::cout << "    wait_ready_chunks: " << wait_ready_chunks << "\n";
			std::cout << "    wait_blocking_chunks: " << wait_blocking_chunks << "\n";
			std::cout << "  payload_arena_bytes: " << total_payload_arena_bytes << "\n";
			std::cout << "  output_arena_bytes: " << total_output_arena_bytes << "\n";
			std::cout << "  h2d_bytes: " << total_h2d_bytes << "\n";
			std::cout << "  h2d_copies: " << total_h2d_copies << "\n";
			std::cout << "  kernel_event_ms: " << kernel_event_ms << "\n";
			std::cout << "  kernel_event_ms_min: " << kernel_stats.min << "\n";
			std::cout << "  kernel_event_ms_median: " << kernel_stats.median << "\n";
			std::cout << "  kernel_event_ms_mean: " << kernel_stats.mean << "\n";
			std::cout << "  release_device_ms: " << release_device_ms << "\n";
			std::cout << "    timing_event_destroy_ms: " << timing_event_destroy_ms << "\n";
			std::cout << "  free_rowgroup_ms: " << free_rowgroup_ms << "\n";
			std::cout << "  prefetch_wait_ms: " << prefetch_wait_ms << "\n";
			std::cout << "  prefetch_depth_block_ms: " << prefetch_depth_block_ms << "\n";
			std::cout << "  prefetch_byte_block_ms: " << prefetch_byte_block_ms << "\n";
			std::cout << "  prefetch_pool_owner_reuses: " << result.prefetch_pool_owner_reuses << "\n";
			std::cout << "  prefetch_pool_owner_migrations: " << result.prefetch_pool_owner_migrations << "\n";
			std::cout << "  prefetch_pool_allocations: " << result.prefetch_pool_allocations << "\n";
			std::cout << "  reader_open_ms: " << reader_open_ms << "\n";
			std::cout << "  descriptor_load_ms: " << descriptor_load_ms << "\n";
			std::cout << "  pinned_pool_create_ms: " << pinned_pool_create_ms << "\n";
			std::cout << "  max_storage_scan_ms: " << max_storage_scan_ms << "\n";
			std::cout << "  pinned_pool_prewarm_ms: " << pinned_pool_prewarm_ms << "\n";
			std::cout << "  prefetch_queue_start_ms: " << prefetch_queue_start_ms << "\n";
			std::cout << "  pipeline_setup_total_ms: " << pipeline_setup_total_ms << "\n";
			std::cout << "  first_rowgroup_read_start_ms: " << result.first_rowgroup_read_start_ms << "\n";
			std::cout << "  first_rowgroup_ready_ms: " << result.first_rowgroup_ready_ms << "\n";
			std::cout << "  prefetched_rowgroups: " << prefetched_rowgroups << "\n";
			// Alias metrics used by existing scripts.
			std::cout << "  end_to_end_ms: " << benchmark_wall_ms << "\n";
			std::cout << "  kernel_ms: " << kernel_event_ms << "\n";
			std::cout << "  kernel_launches: " << total_launches << "\n";
			std::cout << "  avg_grid_per_launch: " << avg_grid_per_launch << "\n";
			std::cout << "  mixed_dispatch: " << (opt.mixed_dispatch ? 1 : 0) << "\n";
			std::cout << "  per_rowgroup_workset: " << (opt.per_rowgroup_workset ? 1 : 0) << "\n";
			std::cout << "  stream_target_work_items: " << opt.stream_target_work_items << "\n";
			std::cout << "  stream_max_rowgroups: " << opt.stream_max_rowgroups << "\n";
			std::cout << "  compute_inflight_chunks: " << compute_inflight_chunks << "\n";
			std::cout << "  compute_inflight_chunks_requested: " << opt.compute_inflight_chunks << "\n";
			std::cout << "  prefetch_depth_effective: " << rowgroup_prefetch_depth << "\n";
			std::cout << "  rowgroup_prefetch: " << (opt.enable_rowgroup_prefetch ? 1 : 0) << "\n";
			std::cout << "  prefetch_depth: " << opt.prefetch_depth << "\n";
			std::cout << "  prefetch_workers: " << actual_prefetch_workers << "\n";
			std::cout << "  prefetch_workers_requested: " << opt.prefetch_workers << "\n";
			std::cout << "  max_prefetch_storage_bytes: " << opt.max_prefetch_storage_bytes << "\n";
			std::cout << "  unpack_n_vectors: " << opt.unpack_n_vectors << "\n";
			std::cout << "  unpack_n_values: " << opt.unpack_n_values << "\n";
			std::cout << "  write_back: " << (bench_cfg.execution.write_out ? 1 : 0) << "\n";
			std::cout << "  include_materialize: " << (bench_cfg.include_materialize ? 1 : 0) << "\n";
			std::cout << "  consume_only: " << (bench_cfg.include_materialize ? 0 : 1) << "\n";
			std::cout << "  write_back_free: " << (bench_cfg.execution.write_out ? 0 : 1) << "\n";
			std::cout << "  reuse_table_resources: " << (opt.reuse_table_resources ? 1 : 0) << "\n";
			std::cout << "  freq_patcher: " << freq_patcher_name(opt.freq_patcher) << "\n";
			std::cout << "  freq_hybrid_threshold: " << opt.freq_branchless_threshold << "\n";
			return 0;
		}

		return 1;
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
