// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/benchmark_support/include/galp_tools/benchmark_support/table.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_SUPPORT_BENCHMARK_TABLE_CUH
#define GALP_SUPPORT_BENCHMARK_TABLE_CUH

#include "execution/table.cuh"
#include <cstdint>
#include <filesystem>
#include <optional>

namespace galp::execution {

using AggregationScope = TableDecompressionScope;

struct TableBenchmarkConfig : TableDecompressionConfig {
	uint32_t              samples = 1;
	std::optional<size_t> rowgroup;
	// When true, the benchmark also materializes results back to host pinned
	// memory (one D2H materialize per chunk) so the wall clock reflects the
	// output-producing decompression path. Off by default for consume-only
	// pipeline measurements.
	bool include_materialize = false;
	// When true, reader construction and pinned rowgroup pool prewarm happen
	// before the query wall-clock timer. benchmark_wall_ms still includes both.
	bool reuse_table_resources = false;

	TableBenchmarkConfig() {
		scope               = TableDecompressionScope::WholeTable;
		// The default benchmark path is consume-only; include_materialize flips
		// write_out on for full output-producing timings.
		execution.write_out = false;
	}
};

struct TableBenchmarkResult {
	double end_to_end_ms           = 0.0; // wall clock of the whole benchmark run
	double resource_prepare_ms     = 0.0; // reusable reader/pinned-pool setup before steady-state query
	double query_wall_ms           = 0.0; // wall clock of the query run after optional resource prepare
	double pipeline_active_ms      = 0.0; // query wall minus pipeline setup
	double read_rowgroup_ms        = 0.0; // galp::format::read_rowgroup* stage
	double file_read_ms            = 0.0; // rowgroup read/setup stage before rowgroup build
	double rowgroup_build_ms       = 0.0; // rowgroup/column construction after file IO
	double pinned_acquire_ms       = 0.0; // pinned rowgroup buffer lease time before file IO
	double pread_ms                = 0.0; // time spent in File::ReadRangeUnchecked / pread
	double zero_copy_view_setup_ms = 0.0; // zero-copy RowgroupView/descriptor setup after pread
	double prefetch_depth_block_ms = 0.0; // worker time blocked by prefetch_depth back-pressure
	double prefetch_byte_block_ms  = 0.0; // worker time blocked by byte-level prefetch budget
	double read_wall_ms            = 0.0; // wall span from first rowgroup read start to last rowgroup build end
	double pread_wall_ms           = 0.0; // wall span from first pread start to last pread end
	double file_read_wall_ms       = 0.0; // wall span of the file_read_ms stage
	double assemble_expr_ms        = 0.0; // galp::expression::assemble stage
	double append_expr_ms          = 0.0; // runtime::append_expressions stage
	double upload_workset_ms       = 0.0; // runtime::upload_workset stage
	// Sub-stage breakdown of upload_workset_ms (sums to ~upload_workset_ms).
	double upload_prep_ms              = 0.0; // output arena, bind pointers, build mixed slots
	double upload_prep_reset_ms        = 0.0; // workset + device-batch reset loop
	double upload_prep_output_arena_ms = 0.0; // ensure_workset_output_arena
	double upload_prep_bind_ms         = 0.0; // bind_workset_output_pointers per-type loop
	double upload_prep_slots_ms        = 0.0; // build_mixed_slots
	double upload_arena_pack_ms        = 0.0; // arena.add + resolve_to for metadata
	double upload_layout_ms            = 0.0; // arena layout pass
	double upload_alloc_ms             = 0.0; // ensure_capacity + ensure_pinned_capacity
	double upload_resolve_ms           = 0.0; // resolver targets + callback resolvers
	double upload_pack_ms              = 0.0; // std::memcpy into pinned
	double upload_dma_issue_ms         = 0.0; // cudaMemcpyAsync for regions + staged
	double upload_dma_gpu_ms           = 0.0; // optional GPU-event H2D duration (GALP_MEASURE_H2D=1)
	double upload_event_ms             = 0.0; // cudaEventRecord for h2d ready
	double kernel_ms                   = 0.0; // accumulated GPU event time returned by run_workset*
	double release_device_ms           = 0.0; // runtime::release_workset stage
	double free_rowgroup_ms            = 0.0; // free_rowgroup stage
	double prefetch_wait_ms            = 0.0; // time waiting for prefetched rowgroups to become available
	double reader_open_ms              = 0.0; // reader construction + rowgroup_count setup
	double descriptor_load_ms          = 0.0; // descriptor access used for benchmark logical-byte accounting
	double pinned_pool_create_ms       = 0.0; // pinned rowgroup pool object creation
	double max_storage_scan_ms         = 0.0; // scan rowgroup storage sizes before pool prewarm
	double pinned_pool_prewarm_ms      = 0.0; // pinned rowgroup buffer prewarm
	double prefetch_queue_start_ms     = 0.0; // prefetch queue construction + worker start
	double pipeline_setup_total_ms     = 0.0; // setup before the first rowgroup fetch loop
	double first_rowgroup_read_start_ms = 0.0; // first rowgroup read start relative to benchmark wall start
	double first_rowgroup_ready_ms      = 0.0; // first rowgroup ready_push relative to benchmark wall start

	size_t   total_launches            = 0;
	size_t   total_launch_grid         = 0;
	size_t   total_columns             = 0;
	size_t   total_items               = 0;
	size_t   total_bytes               = 0;
	size_t   total_storage_bytes       = 0;
	size_t   total_payload_arena_bytes = 0;
	size_t   total_output_arena_bytes  = 0;
	size_t   total_h2d_bytes           = 0;
	size_t   total_h2d_copies          = 0;
	size_t   prefetched_rowgroups      = 0;
	size_t   prefetch_pool_owner_reuses = 0;
	size_t   prefetch_pool_owner_migrations = 0;
	size_t   prefetch_pool_allocations = 0;
	size_t   total_rgs                 = 0;
	uint32_t samples                   = 1;
};

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg = {});

} // namespace galp::execution

#endif // GALP_SUPPORT_BENCHMARK_TABLE_CUH
