// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/runtime/pipeline.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_PIPELINE_CUH
#define ENGINE_RUNTIME_PIPELINE_CUH

#include "runtime/table/runner.cuh"
#include "execution/internal/rowgroup_prefetch_queue.cuh"
#include "core/expression.cuh"
#include <chrono>
#include <flatbuffers/base.h>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace galp::runtime {

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
	const auto             reader_open_start = std::chrono::steady_clock::now();
	resources.shared_rdr         = std::make_shared<galp::format::FlsReader>(fls_path, request.load_column_names);
	galp::format::FlsReader& rdr = *resources.shared_rdr;
	resources.n_rowgroups        = rdr.rowgroup_count();
	const auto reader_open_end   = std::chrono::steady_clock::now();
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
		resources.table_descriptor       = rdr.table_descriptor();
		const auto descriptor_load_end   = std::chrono::steady_clock::now();
		observer.on_descriptor_load(
		    std::chrono::duration<double, std::milli>(descriptor_load_end - descriptor_load_start).count());
	}

	resources.whole_table             = detail::use_whole_table_pipeline(request);
	resources.use_rowgroup_prefetch   = resources.whole_table && request.config.enable_rowgroup_prefetch;
	resources.max_rowgroups_per_chunk = (request.config.streaming_target_rowgroups > 0)
	                                        ? request.config.streaming_target_rowgroups
	                                        : (resources.n_rowgroups > 0 ? resources.n_rowgroups : 1U);
	resources.prefetch_workers =
	    detail::choose_prefetch_workers(request.config.prefetch_workers, resources.end - resources.start);

	if (resources.whole_table) {
		const size_t active_io_workers    = resources.prefetch_workers;
		const size_t prefetch_depth_slots = request.config.prefetch_depth;
		const size_t pooled_slots =
		    (2U * resources.max_rowgroups_per_chunk) + prefetch_depth_slots + active_io_workers + 2U;
		const auto pool_create_start   = std::chrono::steady_clock::now();
		resources.pinned_rowgroup_pool = PinnedRowgroupBufferPool::create(pooled_slots);
		const auto pool_create_end     = std::chrono::steady_clock::now();
		observer.on_pinned_pool_create(
		    std::chrono::duration<double, std::milli>(pool_create_end - pool_create_start).count());
		const size_t total_rowgroups = resources.end - resources.start;
		const size_t warm_slots      = resources.use_rowgroup_prefetch
		                                   ? detail::choose_pinned_prewarm_slots(pooled_slots,
                                                                            total_rowgroups,
                                                                            resources.max_rowgroups_per_chunk,
                                                                            prefetch_depth_slots,
                                                                            active_io_workers)
		                                   : detail::choose_direct_pinned_prewarm_slots(
                                            pooled_slots, total_rowgroups, resources.max_rowgroups_per_chunk);
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
			const auto   prewarm_start  = std::chrono::steady_clock::now();
			const size_t prewarm_owners = resources.use_rowgroup_prefetch ? resources.prefetch_workers : 0U;
			resources.pinned_rowgroup_pool->prewarm(resources.max_storage_bytes, warm_slots, prewarm_owners);
			const auto prewarm_end = std::chrono::steady_clock::now();
			observer.on_pinned_pool_prewarm(
			    std::chrono::duration<double, std::milli>(prewarm_end - prewarm_start).count());
		}
	}

	return resources;
}

template <typename RowgroupPredicate, typename RowgroupCallback, typename Observer>
inline TableData execute_prepared_table_pipeline(PreparedTableResources&                      resources,
                                                 const TableExecutionRequest&                 request,
                                                 RowgroupPredicate&&                          should_process,
                                                 RowgroupCallback&&                           on_rowgroup,
                                                 Observer&                                    observer,
                                                 const std::chrono::steady_clock::time_point& setup_start) {
	if (request.materialize_results && !request.config.execution.write_out) {
		throw std::invalid_argument("table materialization requires execution.write_out=true");
	}

	detail::validate_table_request(request);
	runtime::validate_unpack_config(request.config.execution);
	detail::check_rowgroup_index(resources.n_rowgroups, request.rowgroup);

	galp::format::FlsReader& rdr         = *resources.shared_rdr;
	constexpr bool collect_logical_bytes = !std::is_same_v<std::decay_t<Observer>, NoopTableExecutionObserver>;
	if constexpr (collect_logical_bytes) {
		if (resources.table_descriptor == nullptr) {
			resources.table_descriptor = rdr.table_descriptor();
		}
	}

	std::unique_ptr<RowgroupPrefetchQueue> prefetch_queue;
	if (resources.use_rowgroup_prefetch) {
		const auto prefetch_queue_start = std::chrono::steady_clock::now();
		prefetch_queue                  = std::make_unique<RowgroupPrefetchQueue>(resources.shared_rdr,
                                                                 resources.start,
                                                                 resources.end,
                                                                 request.config.prefetch_depth,
                                                                 resources.prefetch_workers,
                                                                 resources.pinned_rowgroup_pool,
                                                                 resources.fused_prefetch_storage_budget);
		const auto prefetch_queue_end   = std::chrono::steady_clock::now();
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
		auto       expressions    = galp::expression::assemble(pending.rowgroup);
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

		const auto* rg_desc = resources.table_descriptor->m_rowgroup_descriptors()->Get(
		    static_cast<flatbuffers::uoffset_t>(rowgroup_index));
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
	auto       resources   = prepare_table_resources(fls_path, request, observer);
	return execute_prepared_table_pipeline(resources,
	                                       request,
	                                       std::forward<RowgroupPredicate>(should_process),
	                                       std::forward<RowgroupCallback>(on_rowgroup),
	                                       observer,
	                                       setup_start);
}

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_PIPELINE_CUH
