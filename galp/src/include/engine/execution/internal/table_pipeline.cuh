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
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
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
	bool                     materialize_results          = true;
	bool                     direct_append_no_materialize = false;
	bool                     warmup_first_run             = false;
	bool                     load_column_names            = true;
};

struct PreparedTableResources {
	std::shared_ptr<reader::reader>              shared_rdr;
	size_t                                       n_rowgroups                   = 0;
	size_t                                       start                         = 0;
	size_t                                       end                           = 0;
	const fastlanes::TableDescriptor*            table_descriptor              = nullptr;
	bool                                         whole_table                   = false;
	bool                                         use_rowgroup_prefetch         = false;
	size_t                                       max_rowgroups_per_chunk       = 1;
	size_t                                       prefetch_workers              = 1;
	size_t                                       max_storage_bytes             = 0;
	size_t                                       fused_prefetch_storage_budget = 0;
	std::shared_ptr<PinnedRowgroupBufferPool>    pinned_rowgroup_pool;
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

	void on_rowgroup_upload(const size_t,
	                        const std::chrono::steady_clock::time_point&,
	                        const std::chrono::steady_clock::time_point&) {
	}

	void on_kernel(const double, const size_t, const size_t) {
	}

	void on_release_workset(const double) {
	}

	void on_free_rowgroup(const double) {
	}

	void on_prefetch_wait(const double) {
	}

	void on_reader_open(const double) {
	}

	void on_descriptor_load(const double) {
	}

	void on_pinned_pool_create(const double) {
	}

	void on_max_storage_scan(const double) {
	}

	void on_pinned_pool_prewarm(const double) {
	}

	void on_prefetch_queue_start(const double) {
	}

	void on_pipeline_setup_total(const double) {
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

inline size_t choose_prefetch_workers(const size_t requested, const size_t rowgroup_count) {
	if (requested != 0) {
		return std::max<size_t>(1, requested);
	}
	if (rowgroup_count >= 128) {
		return 4;
	}
	if (rowgroup_count >= 32) {
		return 3;
	}
	return 2;
}

inline std::optional<size_t> env_size_value(const char* name) {
	const char* raw = std::getenv(name);
	if (raw == nullptr || *raw == '\0') {
		return std::nullopt;
	}
	char*                    end   = nullptr;
	const unsigned long long value = std::strtoull(raw, &end, 10);
	if (end == raw) {
		return std::nullopt;
	}
	if (value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
		return std::numeric_limits<size_t>::max();
	}
	return static_cast<size_t>(value);
}

inline bool env_mode_is(const char* raw, const char* a, const char* b, const char* c = "") {
	return raw != nullptr &&
	       (std::strcmp(raw, a) == 0 || std::strcmp(raw, b) == 0 || (c[0] != '\0' && std::strcmp(raw, c) == 0));
}

inline size_t choose_pinned_prewarm_slots(const size_t pooled_slots,
                                          const size_t total_rowgroups,
                                          const size_t max_rowgroups_per_chunk,
                                          const size_t prefetch_depth_slots,
                                          const size_t active_io_workers) {
	const size_t full_prefetch_slots = prefetch_depth_slots + active_io_workers + 2U;
	const size_t full_slots = std::min({pooled_slots, max_rowgroups_per_chunk + full_prefetch_slots, total_rowgroups});
	if (full_slots == 0) {
		return 0;
	}

	if (const auto requested = env_size_value("GALP_PINNED_ROWGROUP_PREWARM_SLOTS"); requested.has_value()) {
		return std::min(*requested, full_slots);
	}

	const char* mode = std::getenv("GALP_PINNED_ROWGROUP_PREWARM_MODE");
	if (env_mode_is(mode, "full", "eager", "all")) {
		return full_slots;
	}
	if (env_mode_is(mode, "off", "none", "lazy")) {
		return 0;
	}

	const size_t consumer_slots = std::min(max_rowgroups_per_chunk, total_rowgroups);
	const size_t worker_slots   = std::max<size_t>(1, active_io_workers);
	const size_t target_slots   = consumer_slots + worker_slots + prefetch_depth_slots;
	return std::min(full_slots, std::max<size_t>(1, target_slots));
}

inline size_t choose_direct_pinned_prewarm_slots(const size_t pooled_slots,
                                                 const size_t total_rowgroups,
                                                 const size_t max_rowgroups_per_chunk) {
	if (pooled_slots == 0 || total_rowgroups == 0 || max_rowgroups_per_chunk == 0) {
		return 0;
	}

	const size_t chunk_slots = std::min(max_rowgroups_per_chunk, total_rowgroups);
	const size_t double_buffer_slots =
	    chunk_slots > std::numeric_limits<size_t>::max() / 2U ? std::numeric_limits<size_t>::max()
	                                                          : 2U * chunk_slots;
	const size_t live_slots = std::min({pooled_slots, total_rowgroups, double_buffer_slots});
	if (live_slots == 0) {
		return 0;
	}

	if (const auto requested = env_size_value("GALP_PINNED_ROWGROUP_PREWARM_SLOTS"); requested.has_value()) {
		return std::min(*requested, live_slots);
	}

	const char* mode = std::getenv("GALP_PINNED_ROWGROUP_PREWARM_MODE");
	if (env_mode_is(mode, "off", "none", "lazy")) {
		return 0;
	}

	return live_slots;
}

inline RowgroupReadResult read_rowgroup(reader::reader&                                  rdr,
                                        const size_t                                     rowgroup_index,
                                        const std::shared_ptr<PinnedRowgroupBufferPool>& pinned_pool = {}) {
	RowgroupReadResult                    result {};
	const auto                            read_start = std::chrono::steady_clock::now();
	reader::ZeroCopyRowgroup              zero_copy {};
	reader::ZeroCopyReadTiming            io_timing {};
	std::chrono::steady_clock::time_point file_read_start {};
	const size_t                          storage_bytes = rdr.rowgroup_storage_bytes(rowgroup_index);
	result.rowgroup_index                               = rowgroup_index;
	result.storage_bytes                                = storage_bytes;
	if (pinned_pool) {
		const auto acquire_start = std::chrono::steady_clock::now();
		auto       lease         = pinned_pool->acquire(storage_bytes);
		const auto acquire_end   = std::chrono::steady_clock::now();
		result.timing.pinned_acquire_ms =
		    std::chrono::duration<double, std::milli>(acquire_end - acquire_start).count();
		file_read_start = acquire_end;
		zero_copy       = rdr.read_rowgroup_zero_copy_into(rowgroup_index,
                                                     std::move(lease.owner),
                                                     lease.data,
                                                     lease.capacity,
                                                     /*backing_is_pinned=*/true,
                                                     &io_timing);
	} else {
		file_read_start = std::chrono::steady_clock::now();
		zero_copy       = rdr.read_rowgroup_zero_copy(rowgroup_index, &io_timing);
	}
	const auto file_read_end   = std::chrono::steady_clock::now();
	const auto build_start     = std::chrono::steady_clock::now();
	result.rowgroup            = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
	const auto build_end       = std::chrono::steady_clock::now();
	result.timing.file_read_ms = std::chrono::duration<double, std::milli>(file_read_end - file_read_start).count();
	result.timing.rowgroup_build_ms        = std::chrono::duration<double, std::milli>(build_end - build_start).count();
	result.timing.pread_ms                 = io_timing.pread_ms;
	result.timing.zero_copy_view_setup_ms  = io_timing.zero_copy_view_setup_ms;
	result.timing.timeline.pread_start     = io_timing.pread_start;
	result.timing.timeline.pread_end       = io_timing.pread_end;
	result.timing.timeline.read_submit =
	    (io_timing.pread_start == std::chrono::steady_clock::time_point {}) ? file_read_start : io_timing.pread_start;
	result.timing.read_ms                  = std::chrono::duration<double, std::milli>(build_end - read_start).count();
	result.timing.timeline.read_start      = read_start;
	result.timing.timeline.file_read_start = file_read_start;
	result.timing.timeline.file_read_end   = file_read_end;
	result.timing.timeline.build_start     = build_start;
	result.timing.timeline.build_end       = build_end;
	result.timing.timeline.ready_push      = build_end;
	result.timing.timeline.consumer_wait_start = build_end;
	result.timing.timeline.consumer_wait_end   = build_end;
	result.timing.timeline.consumer_pop    = build_end;
	result.timing.timeline.read_end        = build_end;
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

template <typename Observer>
inline PreparedTableResources prepare_table_resources(const std::filesystem::path& fls_path,
                                                      const TableExecutionRequest& request,
                                                      Observer&                    observer) {
	if (request.materialize_results && !request.config.execution.write_out) {
		throw std::invalid_argument("table materialization requires execution.write_out=true");
	}

	detail::validate_table_request(request);
	runtime::validate_unpack_config(request.config.execution);

	PreparedTableResources resources {};
	const auto reader_open_start = std::chrono::steady_clock::now();
	resources.shared_rdr        = std::make_shared<reader::reader>(fls_path, request.load_column_names);
	reader::reader& rdr         = *resources.shared_rdr;
	resources.n_rowgroups       = rdr.rowgroup_count();
	const auto reader_open_end = std::chrono::steady_clock::now();
	observer.on_reader_open(std::chrono::duration<double, std::milli>(reader_open_end - reader_open_start).count());
	detail::check_rowgroup_index(resources.n_rowgroups, request.rowgroup);

	resources.start = 0;
	resources.end   = resources.n_rowgroups;
	if (request.rowgroup.has_value()) {
		resources.start = *request.rowgroup;
		resources.end   = resources.start + 1;
	}

	constexpr bool collect_logical_bytes = !std::is_same_v<std::decay_t<Observer>, NoopTableExecutionObserver>;
	if constexpr (collect_logical_bytes) {
		const auto descriptor_load_start = std::chrono::steady_clock::now();
		resources.table_descriptor = rdr.table_descriptor();
		const auto descriptor_load_end = std::chrono::steady_clock::now();
		observer.on_descriptor_load(
		    std::chrono::duration<double, std::milli>(descriptor_load_end - descriptor_load_start).count());
	}

	resources.whole_table           = detail::use_whole_table_pipeline(request);
	resources.use_rowgroup_prefetch = resources.whole_table && request.config.enable_rowgroup_prefetch;
	resources.max_rowgroups_per_chunk = (request.config.streaming_target_rowgroups > 0)
	                                        ? request.config.streaming_target_rowgroups
	                                        : (resources.n_rowgroups > 0 ? resources.n_rowgroups : 1U);
	resources.prefetch_workers = detail::choose_prefetch_workers(request.config.prefetch_workers,
	                                                             resources.end - resources.start);

	if (resources.whole_table) {
		const size_t active_io_workers    = resources.prefetch_workers;
		const size_t prefetch_depth_slots = request.config.prefetch_depth;
		const size_t pooled_slots =
		    (2U * resources.max_rowgroups_per_chunk) + prefetch_depth_slots + active_io_workers + 2U;
		const auto pool_create_start = std::chrono::steady_clock::now();
		resources.pinned_rowgroup_pool = PinnedRowgroupBufferPool::create(pooled_slots);
		const auto pool_create_end = std::chrono::steady_clock::now();
		observer.on_pinned_pool_create(
		    std::chrono::duration<double, std::milli>(pool_create_end - pool_create_start).count());
		const size_t total_rowgroups = resources.end - resources.start;
		const size_t warm_slots =
		    resources.use_rowgroup_prefetch
		        ? detail::choose_pinned_prewarm_slots(pooled_slots,
		                                              total_rowgroups,
		                                              resources.max_rowgroups_per_chunk,
		                                              prefetch_depth_slots,
		                                              active_io_workers)
		        : detail::choose_direct_pinned_prewarm_slots(pooled_slots,
		                                                     total_rowgroups,
		                                                     resources.max_rowgroups_per_chunk);
		if (resources.use_rowgroup_prefetch || warm_slots != 0) {
			const auto max_storage_scan_start = std::chrono::steady_clock::now();
			resources.max_storage_bytes       = detail::max_rowgroup_storage_bytes(rdr, resources.start, resources.end);
			const auto max_storage_scan_end   = std::chrono::steady_clock::now();
			observer.on_max_storage_scan(
			    std::chrono::duration<double, std::milli>(max_storage_scan_end - max_storage_scan_start).count());
			if (resources.use_rowgroup_prefetch) {
				if (request.config.max_prefetch_storage_bytes != 0) {
					resources.fused_prefetch_storage_budget = request.config.max_prefetch_storage_bytes;
				} else if (resources.max_storage_bytes != 0) {
					const size_t depth = std::max<size_t>(1, request.config.prefetch_depth);
					resources.fused_prefetch_storage_budget =
					    resources.max_storage_bytes > std::numeric_limits<size_t>::max() / depth
					        ? std::numeric_limits<size_t>::max()
					        : resources.max_storage_bytes * depth;
				}
			}
			const auto prewarm_start = std::chrono::steady_clock::now();
			const size_t prewarm_owner_count =
			    resources.use_rowgroup_prefetch ? resources.prefetch_workers : 0U;
			resources.pinned_rowgroup_pool->prewarm(resources.max_storage_bytes,
			                                        warm_slots,
			                                        prewarm_owner_count);
			const auto prewarm_end = std::chrono::steady_clock::now();
			observer.on_pinned_pool_prewarm(
			    std::chrono::duration<double, std::milli>(prewarm_end - prewarm_start).count());
		}
	}

	return resources;
}

template <typename RowgroupPredicate, typename RowgroupCallback, typename Observer>
inline TableData execute_prepared_table_pipeline(PreparedTableResources&                         resources,
                                                 const TableExecutionRequest&                    request,
                                                 RowgroupPredicate&&                             should_process,
                                                 RowgroupCallback&&                              on_rowgroup,
                                                 Observer&                                       observer,
                                                 const std::chrono::steady_clock::time_point&    setup_start) {
	if (request.materialize_results && !request.config.execution.write_out) {
		throw std::invalid_argument("table materialization requires execution.write_out=true");
	}

	detail::validate_table_request(request);
	runtime::validate_unpack_config(request.config.execution);
	detail::check_rowgroup_index(resources.n_rowgroups, request.rowgroup);

	reader::reader& rdr = *resources.shared_rdr;
	constexpr bool collect_logical_bytes = !std::is_same_v<std::decay_t<Observer>, NoopTableExecutionObserver>;
	if constexpr (collect_logical_bytes) {
		if (resources.table_descriptor == nullptr) {
			resources.table_descriptor = rdr.table_descriptor();
		}
	}

	std::unique_ptr<RowgroupPrefetchQueue> prefetch_queue;
	if (resources.use_rowgroup_prefetch) {
		const auto prefetch_queue_start = std::chrono::steady_clock::now();
		prefetch_queue = std::make_unique<RowgroupPrefetchQueue>(resources.shared_rdr,
		                                                         resources.start,
		                                                         resources.end,
		                                                         request.config.prefetch_depth,
		                                                         resources.prefetch_workers,
		                                                         resources.pinned_rowgroup_pool,
		                                                         resources.fused_prefetch_storage_budget);
		const auto prefetch_queue_end = std::chrono::steady_clock::now();
		observer.on_prefetch_queue_start(
		    std::chrono::duration<double, std::milli>(prefetch_queue_end - prefetch_queue_start).count());
	}
	const auto setup_end = std::chrono::steady_clock::now();
	observer.on_pipeline_setup_total(std::chrono::duration<double, std::milli>(setup_end - setup_start).count());
	const auto from_prefetch = [&]() {
		return prefetch_queue != nullptr;
	};

	const auto fetch_rowgroup = [&](const size_t rowgroup_index) -> RowgroupReadResult {
		if (prefetch_queue) {
			return prefetch_queue->pop();
		}

		return detail::read_rowgroup(rdr, rowgroup_index, resources.pinned_rowgroup_pool);
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

		const auto* rg_desc =
		    resources.table_descriptor->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
		return detail::rowgroup_logical_bytes(rg_desc);
	};

	if (!resources.whole_table) {
		for (size_t rowgroup_index = resources.start; rowgroup_index < resources.end; ++rowgroup_index) {
			if (!should_process(rowgroup_index)) {
				continue;
			}

			auto read_result = fetch_rowgroup(rowgroup_index);
			observer.on_rowgroup_read(read_result, from_prefetch());
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

	for (size_t rowgroup_index = resources.start; rowgroup_index < resources.end; ++rowgroup_index) {
		if (!from_prefetch() && !should_process(rowgroup_index)) {
			continue;
		}

		auto read_result = fetch_rowgroup(rowgroup_index);
		if (!should_process(rowgroup_index)) {
			free_rowgroup(read_result.rowgroup);
			continue;
		}
		observer.on_rowgroup_read(read_result, from_prefetch());

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

template <typename RowgroupPredicate, typename RowgroupCallback, typename Observer>
inline TableData execute_table_pipeline(const std::filesystem::path& fls_path,
                                        const TableExecutionRequest& request,
                                        RowgroupPredicate&&          should_process,
                                        RowgroupCallback&&           on_rowgroup,
                                        Observer&                    observer) {
	const auto setup_start = std::chrono::steady_clock::now();
	auto resources = prepare_table_resources(fls_path, request, observer);
	return execute_prepared_table_pipeline(resources,
	                                       request,
	                                       std::forward<RowgroupPredicate>(should_process),
	                                       std::forward<RowgroupCallback>(on_rowgroup),
	                                       observer,
	                                       setup_start);
}

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_TABLE_PIPELINE_CUH
