// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/runtime/table/runner.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_TABLE_RUNNER_CUH
#define ENGINE_RUNTIME_TABLE_RUNNER_CUH

#include "execution/internal/launch.cuh"
#include "runtime/materialize/metadata.cuh"
#include "runtime/materialize/pinned_d2h.cuh"
#include "runtime/table/helpers.cuh"
#include "runtime/workset/append.cuh"
#include "runtime/workset/upload.cuh"
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

	runtime::wait_workset_async(chunk.run);
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
	runtime::release_workset(chunk.workset, preserve_resources);
	const auto release_end = std::chrono::steady_clock::now();
	observer.on_release_workset(std::chrono::duration<double, std::milli>(release_end - release_start).count());

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

	const bool   warmup    = request.warmup_first_run && !did_warmup;
	const double kernel_ms = runtime::run_workset(
	    chunk.workset, request.samples, request.config.execution, &chunk.launch_grid, &chunk.launches, warmup);
	did_warmup = did_warmup || warmup;
	observer.on_kernel(kernel_ms, chunk.launch_grid, chunk.launches);

	if (request.materialize_results) {
		materialize_table_workset(chunk.workset, chunk.rowgroups, chunk.expr_locations, request.config.execution);
	}

	const auto release_start = std::chrono::steady_clock::now();
	runtime::release_workset(chunk.workset);
	const auto release_end = std::chrono::steady_clock::now();
	observer.on_release_workset(std::chrono::duration<double, std::milli>(release_end - release_start).count());

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
