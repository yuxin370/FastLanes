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
	bool                  use_zero_copy_parse         = true;
	bool                  enable_streaming            = true;
	size_t                streaming_target_work_items = 1u << 18;
	size_t                streaming_target_rowgroups  = 8; // 0 means disable rowgroup-cap flushing.
	std::optional<size_t> rowgroup;
};

struct TableBenchmarkResult {
	double end_to_end_ms     = 0.0; // wall clock of the whole benchmark run
	double read_rowgroup_ms  = 0.0; // reader::read_rowgroup* stage
	double assemble_expr_ms  = 0.0; // expr::assemble stage
	double append_expr_ms    = 0.0; // runtime::append_expressions stage
	double upload_workset_ms = 0.0; // runtime::upload_workset stage
	double kernel_ms         = 0.0; // accumulated GPU event time returned by run_workset*
	double release_device_ms = 0.0; // runtime::release_workset stage
	double free_rowgroup_ms  = 0.0; // free_rowgroup stage

	size_t   total_launches    = 0;
	size_t   total_launch_grid = 0;
	size_t   total_columns     = 0;
	size_t   total_items       = 0;
	size_t   total_bytes       = 0;
	size_t   total_payload_arena_bytes = 0;
	size_t   total_rgs         = 0;
	uint32_t samples           = 1;
};

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg = {});

} // namespace dispatch

#endif // ENGINE_BENCHMARK_TABLE_CUH
