// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/table/runner.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_TABLE_RUNNER_CUH
#define ENGINE_RUNTIME_TABLE_RUNNER_CUH

#include "cuda/launch/launch.cuh"
#include "engine/materialization/metadata.cuh"
#include "engine/materialization/pinned_d2h.cuh"
#include "engine/table/pipeline_state.cuh"
#include "engine/workset/append.cuh"
#include "engine/workset/upload.cuh"
#include <chrono>
#include <stdexcept>
#include <utility>

namespace galp::runtime::detail {

inline auto make_chunk_target_resolver(std::vector<PendingRowgroup>&          rowgroups,
                                       const std::vector<GlobalExprLocation>& expr_locations) {
	return [&](size_t global_expr_index) -> MaterializedColumn* {
		if (global_expr_index >= expr_locations.size()) {
			throw std::out_of_range("table materialization expr index out of range");
		}
		const auto& location = expr_locations[global_expr_index];
		if (location.rowgroup_slot >= rowgroups.size()) {
			throw std::out_of_range("table materialization rowgroup slot out of range");
		}
		auto& rowgroup = rowgroups[location.rowgroup_slot];
		if (location.local_expr_index >= rowgroup.materialized.columns.size()) {
			throw std::out_of_range("table materialization local expr index out of range");
		}
		auto& slot = rowgroup.materialized.columns[location.local_expr_index];
		if (!slot.has_value()) {
			slot.emplace();
		}
		slot->meta.column_index = location.local_expr_index;
		return &(*slot);
	};
}

inline void materialize_table_workset(ExecutionWorkset&                      workset,
                                      std::vector<PendingRowgroup>&          rowgroups,
                                      const std::vector<GlobalExprLocation>& expr_locations,
                                      const ExecutionConfig&                 cfg) {
	runtime::materialize_outputs_via_pinned_d2h(workset, make_chunk_target_resolver(rowgroups, expr_locations));

	for (auto& rowgroup : rowgroups) {
		runtime::apply_aliases(rowgroup.materialized, rowgroup.expressions, cfg);
		runtime::populate_materialized_metadata(rowgroup.materialized, rowgroup.expressions, cfg);
	}
}

template <typename Observer>
inline void prepare_chunk_workset(TableChunkState& chunk, const TableExecutionRequest& request, Observer& observer) {
	if (chunk.rowgroups.empty()) {
		return;
	}

	runtime::begin_workset_chunk_arena(chunk.workset, chunk.active_columns);
	const auto append_start    = std::chrono::steady_clock::now();
	size_t     expr_index_base = 0;
	for (auto& pending : chunk.rowgroups) {
		if (pending.direct_append) {
			runtime::append_rowgroup_columns(
			    chunk.workset, pending.rowgroup, request.config.execution, expr_index_base, true);
		} else {
			runtime::append_expressions(
			    chunk.workset, pending.expressions, request.config.execution, nullptr, nullptr, expr_index_base, true);
		}
		expr_index_base += pending.active_columns;
	}
	const auto append_end = std::chrono::steady_clock::now();
	observer.on_append_expr(std::chrono::duration<double, std::milli>(append_end - append_start).count());

	const auto upload_start     = std::chrono::steady_clock::now();
	const auto upload_breakdown = runtime::upload_workset(chunk.workset, request.config.execution);
	const auto upload_end       = std::chrono::steady_clock::now();
	observer.on_upload_workset(std::chrono::duration<double, std::milli>(upload_end - upload_start).count(),
	                           upload_breakdown,
	                           chunk.workset.buffers.payload_arena_bytes,
	                           chunk.workset.outputs.used_bytes);
	for (const auto& pending : chunk.rowgroups) {
		observer.on_rowgroup_upload(pending.rowgroup_index, upload_start, upload_end);
	}
}

template <typename RowgroupCallback, typename Observer>
inline void consume_completed_chunk(TableChunkState&             chunk,
                                    const TableExecutionRequest& request,
                                    RowgroupCallback&&           on_rowgroup,
                                    Observer&                    observer,
                                    const bool                   preserve_resources) {
	if (!chunk.submitted || chunk.rowgroups.empty()) {
		return;
	}

	const bool run_was_active = chunk.run.active;
	const bool wait_ready     = runtime::workset_run_is_ready(chunk.run);
	const auto wait_start     = std::chrono::steady_clock::now();
	runtime::wait_workset_async(chunk.run);
	const auto wait_end = std::chrono::steady_clock::now();
	observer.on_wait_workset(std::chrono::duration<double, std::milli>(wait_end - wait_start).count(),
	                         chunk.run.event_sync_wall_ms,
	                         chunk.run.pre_kernel_event_ms,
	                         wait_ready);
	observer.on_kernel(chunk.run.elapsed_ms, chunk.launch_grid, chunk.launches);

	if (request.materialize_results) {
		if (chunk.pending_materialize.active) {
			runtime::finalize_pinned_d2h_materialize(chunk.pending_materialize,
			                                         make_chunk_target_resolver(chunk.rowgroups, chunk.expr_locations));
			for (auto& rowgroup : chunk.rowgroups) {
				runtime::apply_aliases(rowgroup.materialized, rowgroup.expressions, request.config.execution);
				runtime::populate_materialized_metadata(
				    rowgroup.materialized, rowgroup.expressions, request.config.execution);
			}
		} else {
			materialize_table_workset(chunk.workset, chunk.rowgroups, chunk.expr_locations, request.config.execution);
		}
	}

	const auto release_start = std::chrono::steady_clock::now();
	double     timing_event_destroy_ms = 0.0;
	runtime::release_workset(chunk.workset, preserve_resources, run_was_active, &timing_event_destroy_ms);
	const auto release_end = std::chrono::steady_clock::now();
	observer.on_release_workset(std::chrono::duration<double, std::milli>(release_end - release_start).count(),
	                            timing_event_destroy_ms);

	for (auto& pending : chunk.rowgroups) {
		on_rowgroup(pending.rowgroup_index,
		            pending.rowgroup,
		            pending.expressions,
		            request.materialize_results ? &pending.materialized : nullptr);

		const auto free_start = std::chrono::steady_clock::now();
		free_rowgroup(pending.rowgroup);
		const auto free_end = std::chrono::steady_clock::now();
		observer.on_free_rowgroup(std::chrono::duration<double, std::milli>(free_end - free_start).count());
	}

	reset_chunk(chunk);
}

template <typename RowgroupCallback, typename Observer>
inline void run_chunk_sync(TableChunkState&             chunk,
                           const TableExecutionRequest& request,
                           RowgroupCallback&&           on_rowgroup,
                           Observer&                    observer,
                           bool&                        did_warmup) {
	if (chunk.rowgroups.empty()) {
		return;
	}

	prepare_chunk_workset(chunk, request, observer);

	const bool warmup           = request.warmup_first_run && !did_warmup;
	const auto run_submit_start = std::chrono::steady_clock::now();
	auto       run              = runtime::run_workset_async(
        chunk.workset, request.samples, request.config.execution, &chunk.launch_grid, &chunk.launches, warmup);
	const auto run_submit_end = std::chrono::steady_clock::now();
	observer.on_run_submit(std::chrono::duration<double, std::milli>(run_submit_end - run_submit_start).count(),
	                       run.timing_event_create_ms,
	                       run.warmup_wall_ms);
	did_warmup                = did_warmup || warmup;
	const bool run_was_active = run.active;
	const bool wait_ready     = runtime::workset_run_is_ready(run);
	const auto wait_start     = std::chrono::steady_clock::now();
	runtime::wait_workset_async(run);
	const auto wait_end = std::chrono::steady_clock::now();
	observer.on_wait_workset(std::chrono::duration<double, std::milli>(wait_end - wait_start).count(),
	                         run.event_sync_wall_ms,
	                         run.pre_kernel_event_ms,
	                         wait_ready);
	const double kernel_ms = run.elapsed_ms;
	observer.on_kernel(kernel_ms, chunk.launch_grid, chunk.launches);

	if (request.materialize_results) {
		materialize_table_workset(chunk.workset, chunk.rowgroups, chunk.expr_locations, request.config.execution);
	}

	const auto release_start = std::chrono::steady_clock::now();
	double     timing_event_destroy_ms = 0.0;
	runtime::release_workset(chunk.workset, false, run_was_active, &timing_event_destroy_ms);
	const auto release_end = std::chrono::steady_clock::now();
	observer.on_release_workset(std::chrono::duration<double, std::milli>(release_end - release_start).count(),
	                            timing_event_destroy_ms);

	for (auto& pending : chunk.rowgroups) {
		on_rowgroup(pending.rowgroup_index,
		            pending.rowgroup,
		            pending.expressions,
		            request.materialize_results ? &pending.materialized : nullptr);

		const auto free_start = std::chrono::steady_clock::now();
		free_rowgroup(pending.rowgroup);
		const auto free_end = std::chrono::steady_clock::now();
		observer.on_free_rowgroup(std::chrono::duration<double, std::milli>(free_end - free_start).count());
	}

	reset_chunk(chunk);
}

template <typename Observer>
inline void add_rowgroup_to_chunk(TableChunkState&             chunk,
                                  PendingRowgroup&&            pending,
                                  TableData&                   out,
                                  const TableExecutionRequest& request,
                                  Observer&                    observer) {
	const size_t rowgroup_slot  = chunk.rowgroups.size();
	const size_t active_columns = pending.active_columns;

	if (request.materialize_results) {
		pending.materialized.columns.resize(pending.expressions.size());
		chunk.expr_locations.reserve(chunk.expr_locations.size() + active_columns);
		for (size_t expr_idx = 0; expr_idx < pending.expressions.size(); ++expr_idx) {
			const auto& expression = pending.expressions[expr_idx];
			if (expression.column && !expression.column->skip_decompress) {
				chunk.expr_locations.push_back(GlobalExprLocation {rowgroup_slot, expr_idx});
			}
		}
	}

	chunk.work_items += active_columns * pending.rowgroup.n_vecs;
	chunk.active_columns += active_columns;

	out.rowgroups += 1;
	out.total_columns += pending.rowgroup.columns.size();
	out.column_counts.push_back(pending.rowgroup.columns.size());

	observer.on_rowgroup_stats(active_columns, active_columns * pending.rowgroup.n_vecs, pending.logical_bytes);
	chunk.rowgroups.push_back(std::move(pending));
}

} // namespace galp::runtime::detail

#endif // ENGINE_RUNTIME_TABLE_RUNNER_CUH
