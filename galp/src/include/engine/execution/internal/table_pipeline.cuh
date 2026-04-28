// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/table_pipeline.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_TABLE_PIPELINE_CUH
#define ENGINE_EXECUTION_INTERNAL_TABLE_PIPELINE_CUH

#include "engine/execution/internal/materialize.cuh"
#include "engine/execution/internal/pinned_rowgroup_pool.cuh"
#include "engine/execution/internal/rowgroup_prefetch_queue.cuh"
#include "engine/execution/internal/streaming_pipeline.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/execution/table.cuh"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace dispatch::runtime {

struct TableExecutionRequest {
	TableDecompressionConfig config {};
	uint32_t                 samples = 1;
	std::optional<size_t>    rowgroup;
	bool                     materialize_results = true;
	bool                     direct_append_no_materialize = false;
	bool                     warmup_first_run    = false;
	bool                     load_column_names   = true;
};

struct NoopTableExecutionObserver {
	void on_rowgroup_read(const RowgroupReadResult&, const bool) {
	}

	void on_assemble_expr(const double) {
	}

	void on_rowgroup_stats(const size_t, const size_t, const size_t) {
	}

	void on_append_expr(const double) {
	}

	void on_upload_workset(const double, const UploadBreakdown&, const size_t, const size_t) {
	}

	void on_kernel(const double, const size_t, const size_t) {
	}

	void on_release_workset(const double) {
	}

	void on_free_rowgroup(const double) {
	}

	void on_prefetch_wait(const double) {
	}
};

namespace detail {

struct GlobalExprLocation {
	size_t rowgroup_slot    = 0;
	size_t local_expr_index = 0;
};

struct PendingRowgroup {
	size_t                        rowgroup_index = 0;
	size_t                        logical_bytes  = 0;
	size_t                        active_columns = 0;
	reader::Rowgroup              rowgroup {};
	std::vector<expr::Expression> expressions;
	RowgroupData                  materialized;
	bool                          direct_append = false;
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
	// In-flight pinned-D2H job tied to this chunk's outputs arena. Populated
	// at submit time when materialize_results=true so the D2H runs in parallel
	// with the next chunk's H2D + kernel; consumed at the consume-phase.
	runtime::PendingMaterialize pending_materialize {};
};

inline size_t data_type_size(const fastlanes::DataType dt) {
	using fastlanes::DataType;
	switch (dt) {
	case DataType::INT8:
	case DataType::UINT8:
	case DataType::BOOLEAN:
		return 1;
	case DataType::INT16:
	case DataType::UINT16:
		return 2;
	case DataType::INT32:
	case DataType::UINT32:
	case DataType::FLOAT:
	case DataType::DATE:
		return 4;
	case DataType::INT64:
	case DataType::UINT64:
	case DataType::DOUBLE:
	case DataType::TIMESTAMP:
		return 8;
	default:
		return 0;
	}
}

inline size_t rowgroup_logical_bytes(const fastlanes::RowgroupDescriptor* rg) {
	if (!rg) {
		return 0;
	}

	size_t      bytes = 0;
	const auto* cols  = rg->m_column_descriptors();
	if (!cols) {
		return 0;
	}

	for (const auto* col_desc : *cols) {
		if (!col_desc) {
			continue;
		}
		const size_t element_size = data_type_size(col_desc->data_type());
		if (element_size == 0) {
			continue;
		}
		bytes += static_cast<size_t>(rg->m_n_tuples()) * element_size;
	}
	return bytes;
}

inline size_t count_active_columns(const std::vector<expr::Expression>& expressions) {
	return runtime::count_active_columns(expressions);
}

inline size_t max_rowgroup_storage_bytes(reader::reader& rdr, const size_t start, const size_t end) {
	size_t max_bytes = 0;
	for (size_t rowgroup_index = start; rowgroup_index < end; ++rowgroup_index) {
		max_bytes = std::max(max_bytes, rdr.rowgroup_storage_bytes(rowgroup_index));
	}
	return max_bytes;
}

inline void reset_chunk(TableChunkState& chunk) {
	chunk.rowgroups.clear();
	chunk.expr_locations.clear();
	chunk.work_items     = 0;
	chunk.active_columns = 0;
	chunk.launch_grid    = 0;
	chunk.launches       = 0;
	chunk.submitted      = false;
}

inline void check_rowgroup_index(const size_t n_rowgroups, const std::optional<size_t>& rowgroup) {
	if (rowgroup.has_value() && *rowgroup >= n_rowgroups) {
		throw std::out_of_range("rowgroup index out of range");
	}
}

inline void validate_table_request(const TableExecutionRequest& request) {
	if (request.config.streaming_target_work_items == 0) {
		throw std::invalid_argument("streaming_target_work_items must be > 0");
	}
	if (request.config.prefetch_depth == 0) {
		throw std::invalid_argument("prefetch_depth must be > 0");
	}
}

inline bool use_whole_table_pipeline(const TableExecutionRequest& request) {
	return !request.rowgroup.has_value() && request.config.scope == TableDecompressionScope::WholeTable;
}

inline RowgroupReadResult read_rowgroup(reader::reader&                                  rdr,
                                        const size_t                                     rowgroup_index,
                                        const std::shared_ptr<PinnedRowgroupBufferPool>& pinned_pool = {}) {
	RowgroupReadResult       result {};
	const auto               file_read_start = std::chrono::steady_clock::now();
	reader::ZeroCopyRowgroup zero_copy {};
	if (pinned_pool) {
		auto lease = pinned_pool->acquire(rdr.rowgroup_storage_bytes(rowgroup_index));
		zero_copy  = rdr.read_rowgroup_zero_copy_into(
            rowgroup_index, std::move(lease.owner), lease.data, lease.capacity, /*backing_is_pinned=*/true);
	} else {
		zero_copy = rdr.read_rowgroup_zero_copy(rowgroup_index);
	}
	const auto file_read_end = std::chrono::steady_clock::now();
	const auto build_start   = std::chrono::steady_clock::now();
	result.rowgroup          = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
	const auto build_end     = std::chrono::steady_clock::now();
	result.file_read_ms      = std::chrono::duration<double, std::milli>(file_read_end - file_read_start).count();
	result.rowgroup_build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
	result.read_ms           = result.file_read_ms + result.rowgroup_build_ms;
	return result;
}

// Resolver shared by the kick (workset state still alive) and finalize (pinned
// buffer reborn into per-column shared_ptrs) phases of pinned-D2H materialize.
// Returns nullptr when the global expr maps to a skip_decompress column.
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
		// The legacy materializer overwrote column_index with location.local_expr_index;
		// populate_materialized_metadata below restores it from the expression list.
		slot->meta.column_index = location.local_expr_index;
		return &(*slot);
	};
}

// Table chunk materialization: one async pinned D2H of the entire chunk's
// outputs arena, then per-expression aliased shared_ptr<T[]> views into the
// pinned buffer. No per-column DMA, no pageable destinations. The pinned slot
// is reference-counted via the aliasing constructor so it lives as long as any
// column view from this chunk.
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

	// When materialize_results=true the streaming submitter already kicked the
	// D2H (kernel-stop event → d2h_stream → pinned buffer). The kernel-event
	// elapsed is recorded by run_workset_async / wait_workset_async; for the
	// async-D2H path we still need wait_workset_async to populate elapsed_ms.
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

	observer.on_rowgroup_stats(active_columns, active_columns * pending.rowgroup.n_vecs, pending.logical_bytes);
	chunk.rowgroups.push_back(std::move(pending));
}

} // namespace detail

template <typename RowgroupPredicate, typename RowgroupCallback, typename Observer>
inline TableData execute_table_pipeline(const std::filesystem::path& fls_path,
                                        const TableExecutionRequest& request,
                                        RowgroupPredicate&&          should_process,
                                        RowgroupCallback&&           on_rowgroup,
                                        Observer&                    observer) {
	if (request.materialize_results && !request.config.execution.write_out) {
		throw std::invalid_argument("table materialization requires execution.write_out=true");
	}

	detail::validate_table_request(request);
	runtime::validate_unpack_config(request.config.execution);

	auto            shared_rdr  = std::make_shared<reader::reader>(fls_path, request.load_column_names);
	reader::reader& rdr         = *shared_rdr;
	const size_t    n_rowgroups = rdr.rowgroup_count();
	detail::check_rowgroup_index(n_rowgroups, request.rowgroup);

	size_t start = 0;
	size_t end   = n_rowgroups;
	if (request.rowgroup.has_value()) {
		start = *request.rowgroup;
		end   = start + 1;
	}

	constexpr bool collect_logical_bytes = !std::is_same_v<std::decay_t<Observer>, NoopTableExecutionObserver>;
	const fastlanes::TableDescriptor*               td = nullptr;
	std::optional<fastlanes::TableDescriptorHandle> td_handle;
	if constexpr (collect_logical_bytes) {
		td_handle.emplace(reader::detail::load_table_descriptor(fls_path));
		td = td_handle->Get();
		if (!td) {
			throw std::runtime_error("failed to load table descriptor");
		}
	}

	const bool   whole_table             = detail::use_whole_table_pipeline(request);
	const bool   use_rowgroup_prefetch   = whole_table && request.config.enable_rowgroup_prefetch;
	const size_t max_rowgroups_per_chunk = (request.config.streaming_target_rowgroups > 0)
	                                           ? request.config.streaming_target_rowgroups
	                                           : (n_rowgroups > 0 ? n_rowgroups : 1U);

	std::shared_ptr<PinnedRowgroupBufferPool> pinned_rowgroup_pool;
	if (whole_table) {
		const size_t pooled_slots = (2U * max_rowgroups_per_chunk) + request.config.prefetch_depth +
		                            std::max<size_t>(1, request.config.prefetch_workers) + 2U;
		pinned_rowgroup_pool = PinnedRowgroupBufferPool::create(pooled_slots);
		if (use_rowgroup_prefetch) {
			const size_t prefetch_slots =
			    request.config.prefetch_depth + std::max<size_t>(1, request.config.prefetch_workers) + 2U;
			const size_t warm_slots = std::min({pooled_slots, max_rowgroups_per_chunk + prefetch_slots, end - start});
			pinned_rowgroup_pool->prewarm(detail::max_rowgroup_storage_bytes(rdr, start, end), warm_slots);
		}
	}

	std::unique_ptr<RowgroupPrefetchQueue> prefetch_queue;
	if (use_rowgroup_prefetch) {
		prefetch_queue = std::make_unique<RowgroupPrefetchQueue>(shared_rdr,
		                                                         start,
		                                                         end,
		                                                         request.config.prefetch_depth,
		                                                         request.config.prefetch_workers,
		                                                         pinned_rowgroup_pool);
	}

	const auto fetch_rowgroup = [&](const size_t rowgroup_index) -> RowgroupReadResult {
		if (prefetch_queue) {
			return prefetch_queue->pop();
		}

		return detail::read_rowgroup(rdr, rowgroup_index, pinned_rowgroup_pool);
	};

	const auto build_pending = [&](const size_t       rowgroup_index,
	                               RowgroupReadResult read_result,
	                               const size_t       logical_bytes) -> detail::PendingRowgroup {
		detail::PendingRowgroup pending {};
		pending.rowgroup_index = rowgroup_index;
		pending.logical_bytes  = logical_bytes;
		pending.rowgroup       = std::move(read_result.rowgroup);
		if (request.direct_append_no_materialize && !request.materialize_results &&
		    runtime::can_direct_append_rowgroup(pending.rowgroup)) {
			pending.active_columns = runtime::count_active_columns(pending.rowgroup);
			pending.direct_append  = true;
			observer.on_assemble_expr(0.0);
			return pending;
		}

		const auto assemble_start = std::chrono::steady_clock::now();
		auto       expressions    = expr::assemble(pending.rowgroup);
		const auto assemble_end   = std::chrono::steady_clock::now();
		observer.on_assemble_expr(std::chrono::duration<double, std::milli>(assemble_end - assemble_start).count());

		pending.active_columns = detail::count_active_columns(expressions);
		pending.expressions    = std::move(expressions);
		return pending;
	};

	TableData  out {};
	bool       did_warmup        = !request.warmup_first_run;
	const auto logical_bytes_for = [&](const size_t rowgroup_index) -> size_t {
		if constexpr (!collect_logical_bytes) {
			(void)rowgroup_index;
			return 0;
		}

		const auto* rg_desc = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
		return detail::rowgroup_logical_bytes(rg_desc);
	};

	if (!whole_table) {
		for (size_t rowgroup_index = start; rowgroup_index < end; ++rowgroup_index) {
			if (!should_process(rowgroup_index)) {
				continue;
			}

			auto read_result = fetch_rowgroup(rowgroup_index);
			observer.on_rowgroup_read(read_result, prefetch_queue != nullptr);
			auto pending = build_pending(rowgroup_index, std::move(read_result), logical_bytes_for(rowgroup_index));

			detail::TableChunkState chunk {};
			detail::add_rowgroup_to_chunk(chunk, std::move(pending), out, request, observer);
			detail::run_chunk_sync(chunk, request, on_rowgroup, observer, did_warmup);
		}

		if (prefetch_queue) {
			observer.on_prefetch_wait(prefetch_queue->wait_ms());
		}
		return out;
	}

	StreamingDoubleBuffer<detail::TableChunkState> pipeline {};

	const auto submit_chunk = [&](detail::TableChunkState& chunk) {
		if (chunk.rowgroups.empty()) {
			return;
		}

		detail::prepare_chunk_workset(chunk, request, observer);
		const bool warmup = request.warmup_first_run && !did_warmup;
		chunk.run         = runtime::run_workset_async(
            chunk.workset, request.samples, request.config.execution, &chunk.launch_grid, &chunk.launches, warmup);
		chunk.submitted = true;
		did_warmup      = did_warmup || warmup;

		// Kick the chunk's D2H *now* so it runs concurrently with the next
		// chunk's prepare_chunk_workset / H2D / kernel on different streams.
		// d2h_stream waits on the kernel-stop event published inside
		// kick_pinned_d2h_materialize so device data is consistent.
		if (request.materialize_results) {
			chunk.pending_materialize = runtime::kick_pinned_d2h_materialize(chunk.workset);
		}
	};

	const auto consume_chunk = [&](detail::TableChunkState& chunk) {
		detail::consume_completed_chunk(chunk, request, on_rowgroup, observer, /*preserve_resources=*/true);
	};

	for (size_t rowgroup_index = start; rowgroup_index < end; ++rowgroup_index) {
		if (!prefetch_queue && !should_process(rowgroup_index)) {
			continue;
		}

		auto read_result = fetch_rowgroup(rowgroup_index);
		if (!should_process(rowgroup_index)) {
			free_rowgroup(read_result.rowgroup);
			continue;
		}
		observer.on_rowgroup_read(read_result, prefetch_queue != nullptr);

		auto  pending = build_pending(rowgroup_index, std::move(read_result), logical_bytes_for(rowgroup_index));
		auto& chunk   = pipeline.build_chunk();
		detail::add_rowgroup_to_chunk(chunk, std::move(pending), out, request, observer);

		const bool reach_items     = chunk.work_items >= request.config.streaming_target_work_items;
		const bool reach_rowgroups = (request.config.streaming_target_rowgroups > 0) &&
		                             (chunk.rowgroups.size() >= request.config.streaming_target_rowgroups);
		if (reach_items || reach_rowgroups) {
			pipeline.submit_build_and_rotate(submit_chunk, consume_chunk);
		}
	}

	pipeline.flush(submit_chunk, consume_chunk);
	runtime::release_workset(pipeline.build_chunk().workset);
	runtime::release_workset(pipeline.other_chunk().workset);
	if (prefetch_queue) {
		observer.on_prefetch_wait(prefetch_queue->wait_ms());
	}
	return out;
}

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_TABLE_PIPELINE_CUH
