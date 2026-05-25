// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/table/resources.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_TABLE_RESOURCES_CUH
#define ENGINE_RUNTIME_TABLE_RESOURCES_CUH

#include "engine/operators/rowgroup.cuh"
#include "engine/pipeline/pinned_rowgroup_pool.cuh"
#include "engine/pipeline/rowgroup_prefetch_types.cuh"
#include "engine/table/table.cuh"
#include "engine/workset/model.cuh"
#include "format/reader.cuh"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace galp::runtime {

using galp::execution::ExecutionConfig;
using galp::execution::free_rowgroup;
using galp::execution::MaterializedColumn;
using galp::execution::RowgroupData;
using galp::execution::TableData;
using galp::execution::TableDecompressionConfig;
using galp::execution::TableDecompressionScope;

struct TableExecutionRequest {
	TableDecompressionConfig config {};
	uint32_t                 samples = 1;
	std::optional<size_t>    rowgroup;
	std::vector<size_t>      rowgroup_schedule;
	bool                     materialize_results          = true;
	bool                     direct_append_no_materialize = false;
	bool                     warmup_first_run             = false;
	bool                     load_column_names            = true;
};

struct PreparedTableResources {
	std::shared_ptr<galp::format::FlsReader>  shared_rdr;
	size_t                                    n_rowgroups                   = 0;
	size_t                                    start                         = 0;
	size_t                                    end                           = 0;
	const fastlanes::TableDescriptor*         table_descriptor              = nullptr;
	bool                                      whole_table                   = false;
	bool                                      use_rowgroup_prefetch         = false;
	size_t                                    max_rowgroups_per_chunk       = 1;
	size_t                                    compute_inflight_chunks       = 1;
	size_t                                    rowgroup_prefetch_depth       = 1;
	size_t                                    prefetch_workers              = 1;
	size_t                                    max_storage_bytes             = 0;
	size_t                                    fused_prefetch_storage_budget = 0;
	std::shared_ptr<PinnedRowgroupBufferPool> pinned_rowgroup_pool;
};

struct NoopTableExecutionObserver {
	void on_rowgroup_read(const RowgroupReadResult&, bool) {
	}
	void on_assemble_expr(double) {
	}
	void on_rowgroup_stats(size_t, size_t, size_t) {
	}
	void on_append_expr(double) {
	}
	void on_upload_workset(double, const UploadBreakdown&, size_t, size_t) {
	}
	void on_rowgroup_upload(size_t,
	                        const std::chrono::steady_clock::time_point&,
	                        const std::chrono::steady_clock::time_point&) {
	}
	void on_compute_inflight_chunks(size_t) {
	}
	void on_rowgroup_prefetch_depth(size_t) {
	}
	void on_run_submit(double, double, double) {
	}
	void on_wait_workset(double, double, double, bool) {
	}
	void on_kernel(double, size_t, size_t) {
	}
	void on_release_workset(double, double) {
	}
	void on_free_rowgroup(double) {
	}
	void on_prefetch_wait(double) {
	}
	void on_reader_open(double) {
	}
	void on_descriptor_load(double) {
	}
	void on_pinned_pool_create(double) {
	}
	void on_max_storage_scan(double) {
	}
	void on_pinned_pool_prewarm(double) {
	}
	void on_prefetch_queue_start(double) {
	}
	void on_pipeline_setup_total(double) {
	}
};

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_TABLE_RESOURCES_CUH
