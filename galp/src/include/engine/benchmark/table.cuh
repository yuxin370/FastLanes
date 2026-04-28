// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/benchmark/table.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_BENCHMARK_TABLE_CUH
#define ENGINE_BENCHMARK_TABLE_CUH

#include "engine/execution/table.cuh"
#include <cstdint>
#include <filesystem>
#include <optional>

namespace dispatch {

using AggregationScope = TableDecompressionScope;

struct TableBenchmarkConfig : TableDecompressionConfig {
	uint32_t              samples             = 1;
	std::optional<size_t> rowgroup;
	// When true, the benchmark also materializes results back to host pinned
	// memory (one cudaMemcpyAsync(D2H) per chunk via the pinned-D2H path) so
	// the wall clock reflects the full output-producing decompression cost.
	// Off by default because the tuning scripts track the GPU-consume/discard
	// path separately and record include_materialize/write_back in the CSV.
	bool                  include_materialize = false;

	TableBenchmarkConfig() {
		scope               = TableDecompressionScope::WholeTable;
		execution.write_out = false;
	}
};

struct TableBenchmarkResult {
	double end_to_end_ms     = 0.0; // wall clock of the whole benchmark run
	double read_rowgroup_ms  = 0.0; // reader::read_rowgroup* stage
	double file_read_ms      = 0.0; // file IO only
	double rowgroup_build_ms = 0.0; // rowgroup/column construction after file IO
	double assemble_expr_ms  = 0.0; // expr::assemble stage
	double append_expr_ms    = 0.0; // runtime::append_expressions stage
	double upload_workset_ms = 0.0; // runtime::upload_workset stage
	// Sub-stage breakdown of upload_workset_ms (sums to ~upload_workset_ms).
	double upload_prep_ms               = 0.0; // output arena, bind pointers, build mixed slots
	double upload_prep_reset_ms         = 0.0; // workset + device-batch reset loop
	double upload_prep_output_arena_ms  = 0.0; // ensure_workset_output_arena
	double upload_prep_bind_ms          = 0.0; // bind_workset_output_pointers per-type loop
	double upload_prep_slots_ms         = 0.0; // build_mixed_slots
	double upload_arena_pack_ms         = 0.0; // arena.add + resolve_to for metadata
	double upload_layout_ms             = 0.0; // arena layout pass
	double upload_alloc_ms              = 0.0; // ensure_capacity + ensure_pinned_capacity
	double upload_resolve_ms            = 0.0; // resolver targets + callback resolvers
	double upload_pack_ms               = 0.0; // std::memcpy into pinned
	double upload_dma_issue_ms          = 0.0; // cudaMemcpyAsync for regions + staged
	double upload_dma_gpu_ms            = 0.0; // optional GPU-event H2D duration (GALP_MEASURE_H2D=1)
	double upload_event_ms              = 0.0; // cudaEventRecord for h2d ready
	double kernel_ms         = 0.0; // accumulated GPU event time returned by run_workset*
	double release_device_ms = 0.0; // runtime::release_workset stage
	double free_rowgroup_ms  = 0.0; // free_rowgroup stage
	double prefetch_wait_ms  = 0.0; // time waiting for prefetched rowgroups to become available

	size_t   total_launches    = 0;
	size_t   total_launch_grid = 0;
	size_t   total_columns     = 0;
	size_t   total_items       = 0;
	size_t   total_bytes       = 0;
	size_t   total_payload_arena_bytes = 0;
	size_t   total_output_arena_bytes  = 0;
	size_t   total_h2d_bytes           = 0;
	size_t   total_h2d_copies          = 0;
	size_t   prefetched_rowgroups      = 0;
	size_t   total_rgs         = 0;
	uint32_t samples           = 1;
};

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg = {});

} // namespace dispatch

#endif // ENGINE_BENCHMARK_TABLE_CUH
