// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/table/chunk_state.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_TABLE_CHUNK_STATE_CUH
#define ENGINE_RUNTIME_TABLE_CHUNK_STATE_CUH

#include "cuda/launch/launch.cuh"
#include "engine/pipeline/streaming_pipeline.cuh"
#include "engine/materialization/types.cuh"
#include "engine/table/resources.cuh"
#include <vector>

namespace galp::runtime::detail {

struct GlobalExprLocation {
	size_t rowgroup_slot    = 0;
	size_t local_expr_index = 0;
};

struct PendingRowgroup {
	size_t                                    rowgroup_index = 0;
	size_t                                    logical_bytes  = 0;
	size_t                                    active_columns = 0;
	galp::format::Rowgroup                    rowgroup {};
	std::vector<galp::expression::Expression> expressions;
	RowgroupData                              materialized;
	bool                                      direct_append = false;
};

struct TableChunkState {
	ExecutionWorkset                workset {};
	AsyncWorksetRun                 run {};
	std::vector<PendingRowgroup>    rowgroups;
	std::vector<GlobalExprLocation> expr_locations;
	size_t                          work_items     = 0;
	size_t                          active_columns = 0;
	size_t                          launch_grid    = 0;
	size_t                          launches       = 0;
	bool                            submitted      = false;
	runtime::PendingMaterialize     pending_materialize {};

	TableChunkState()                                  = default;
	TableChunkState(const TableChunkState&)            = delete;
	TableChunkState& operator=(const TableChunkState&) = delete;
	~TableChunkState() noexcept;

	void cleanup_noexcept() noexcept;
};

void reset_chunk(TableChunkState& chunk);

} // namespace galp::runtime::detail

#endif // ENGINE_RUNTIME_TABLE_CHUNK_STATE_CUH
