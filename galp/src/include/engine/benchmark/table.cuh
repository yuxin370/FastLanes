// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/benchmark/table.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_BENCHMARK_TABLE_CUH
#define ENGINE_BENCHMARK_TABLE_CUH

#include "engine/execution/common.cuh"
#include <filesystem>
#include <optional>

namespace dispatch {

enum class AggregationScope {
	PerRowgroup,
	WholeTable,
};

struct TableBenchmarkConfig {
	uint32_t         samples           = 1;
	AggregationScope aggregation_scope = AggregationScope::WholeTable;
	ExecutionConfig  execution         = [] {
        ExecutionConfig cfg {};
        cfg.write_out = false;
        return cfg;
	}();
	std::optional<size_t> rowgroup;
};

struct TableBenchmarkResult {
	double end_to_end_ms = 0.0; // excludes teardown
	double kernel_ms     = 0.0;
	double setup_ms      = 0.0;
	double h2d_ms        = 0.0;
	double pure_h2d_ms    = 0.0;
	double payload_h2d_ms = 0.0;
	double dispatch_h2d_ms = 0.0;
	double resolve_cpu_ms = 0.0;
	double cpu_dispatch_ms = 0.0;
	double cpu_dispatch_resolve_ms = 0.0;
	double teardown_ms   = 0.0;

	size_t   total_launches    = 0;
	size_t   total_launch_grid = 0;
	size_t   total_columns     = 0;
	size_t   total_items       = 0;
	size_t   total_bytes       = 0;
	size_t   total_rgs         = 0;
	uint32_t samples           = 1;
};

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg = {});

} // namespace dispatch

#endif // ENGINE_BENCHMARK_TABLE_CUH
