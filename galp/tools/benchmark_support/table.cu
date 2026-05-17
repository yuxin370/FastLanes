// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/benchmark_support/table.cu
// ────────────────────────────────────────────────────────
#include "galp_tools/benchmark_support/table.cuh"
#include "runtime/pipeline.cuh"
#include "cuda/memory/cuda_macros.cuh"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace galp::execution {
namespace {

__global__ void benchmark_warmup_kernel() {
}

void warmup_cuda_runtime_once() {
	static std::once_flag once;
	std::call_once(once, []() {
		CUDA_SAFE_CALL(cudaFree(0));

		cudaStream_t stream {};
		cudaEvent_t  done {};
		CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

		for (const size_t bytes : {1ULL << 20, 4ULL << 20, 16ULL << 20, 64ULL << 20}) {
			void* ptr    = nullptr;
			auto  status = cudaMallocAsync(&ptr, bytes, stream);
			if (status == cudaSuccess) {
				CUDA_SAFE_CALL(cudaFreeAsync(ptr, stream));
			} else {
				ptr = nullptr;
				CUDA_SAFE_CALL(cudaMalloc(&ptr, bytes));
				CUDA_SAFE_CALL(cudaFree(ptr));
			}
		}

		for (int i = 0; i < 32; ++i) {
			benchmark_warmup_kernel<<<1, 1, 0, stream>>>();
		}
		CUDA_SAFE_CALL(cudaGetLastError());
		CUDA_SAFE_CALL(cudaEventRecord(done, stream));
		CUDA_SAFE_CALL(cudaEventSynchronize(done));
		CUDA_SAFE_CALL(cudaEventDestroy(done));
		CUDA_SAFE_CALL(cudaStreamDestroy(stream));
	});
}

struct BenchmarkObserver {
	TableBenchmarkResult&                 out;
	struct TimelineRow {
		size_t                         rowgroup_index = 0;
		bool                           from_prefetch  = false;
		size_t                         storage_bytes  = 0;
		runtime::RowgroupReadTiming     timing {};
		runtime::RowgroupPrefetchTiming prefetch {};
	};

	bool                                  have_read_times  = false;
	bool                                  have_pread_times = false;
	bool                                  have_first_rowgroup_read_start = false;
	bool                                  have_first_rowgroup_ready      = false;
	bool                                  timeline_enabled = false;
	std::filesystem::path                 timeline_path {};
	std::chrono::steady_clock::time_point timeline_base {};
	std::chrono::steady_clock::time_point first_read_start {};
	std::chrono::steady_clock::time_point first_pread_start {};
	std::chrono::steady_clock::time_point last_pread_end {};
	std::chrono::steady_clock::time_point first_file_read_start {};
	std::chrono::steady_clock::time_point last_file_read_end {};
	std::chrono::steady_clock::time_point last_read_end {};
	std::vector<TimelineRow>              timeline_rows;
	std::unordered_map<size_t, size_t>     timeline_row_by_rg;

	void configure_timeline(const std::chrono::steady_clock::time_point base) {
		timeline_base = base;
		const char* path = std::getenv("GALP_ROWGROUP_TIMELINE_CSV");
		if (path != nullptr && path[0] != '\0') {
			timeline_enabled = true;
			timeline_path    = path;
		}
	}

	TimelineRow& timeline_row_for(const size_t rowgroup_index) {
		const auto existing = timeline_row_by_rg.find(rowgroup_index);
		if (existing != timeline_row_by_rg.end()) {
			return timeline_rows[existing->second];
		}

		const size_t slot = timeline_rows.size();
		timeline_row_by_rg.emplace(rowgroup_index, slot);
		timeline_rows.push_back(TimelineRow {});
		timeline_rows.back().rowgroup_index = rowgroup_index;
		return timeline_rows.back();
	}

	double ms_since_base(const std::chrono::steady_clock::time_point tp) const {
		if (tp == std::chrono::steady_clock::time_point {}) {
			return 0.0;
		}
		return std::chrono::duration<double, std::milli>(tp - timeline_base).count();
	}

	void on_rowgroup_read(const runtime::RowgroupReadResult& result, const bool from_prefetch) {
		const auto& timing   = result.timing;
		const auto& timeline = timing.timeline;

		out.read_rowgroup_ms += timing.read_ms;
		out.file_read_ms += timing.file_read_ms;
		out.rowgroup_build_ms += timing.rowgroup_build_ms;
		out.pinned_acquire_ms += timing.pinned_acquire_ms;
		out.pread_ms += timing.pread_ms;
		out.zero_copy_view_setup_ms += timing.zero_copy_view_setup_ms;
		out.prefetch_depth_block_ms += result.prefetch.depth_block_ms;
		out.prefetch_byte_block_ms += result.prefetch.byte_block_ms;
		out.total_storage_bytes += result.storage_bytes;
		if (result.prefetch.pool_slot_owner_reused) {
			++out.prefetch_pool_owner_reuses;
		}
		if (result.prefetch.pool_slot_owner_migrated) {
			++out.prefetch_pool_owner_migrations;
		}
		if (result.prefetch.pool_slot_allocated) {
			++out.prefetch_pool_allocations;
		}
		if (timeline.has_read_span()) {
			if (!have_read_times) {
				first_read_start      = timeline.read_start;
				first_file_read_start = timeline.file_read_start;
				last_file_read_end    = timeline.file_read_end;
				last_read_end         = timeline.read_end;
				have_read_times       = true;
			} else {
				first_read_start      = std::min(first_read_start, timeline.read_start);
				first_file_read_start = std::min(first_file_read_start, timeline.file_read_start);
				last_file_read_end    = std::max(last_file_read_end, timeline.file_read_end);
				last_read_end         = std::max(last_read_end, timeline.read_end);
			}
			out.file_read_wall_ms =
			    std::chrono::duration<double, std::milli>(last_file_read_end - first_file_read_start).count();
			out.read_wall_ms = std::chrono::duration<double, std::milli>(last_read_end - first_read_start).count();
		}
		if (!have_first_rowgroup_read_start && timeline.read_start != std::chrono::steady_clock::time_point {}) {
			out.first_rowgroup_read_start_ms = ms_since_base(timeline.read_start);
			have_first_rowgroup_read_start   = true;
		}
		if (!have_first_rowgroup_ready && timeline.ready_push != std::chrono::steady_clock::time_point {}) {
			out.first_rowgroup_ready_ms = ms_since_base(timeline.ready_push);
			have_first_rowgroup_ready   = true;
		}
		if (timeline.has_pread_span()) {
			if (!have_pread_times) {
				first_pread_start = timeline.pread_start;
				last_pread_end    = timeline.pread_end;
				have_pread_times  = true;
			} else {
				first_pread_start = std::min(first_pread_start, timeline.pread_start);
				last_pread_end    = std::max(last_pread_end, timeline.pread_end);
			}
			out.pread_wall_ms = std::chrono::duration<double, std::milli>(last_pread_end - first_pread_start).count();
		}
		if (from_prefetch) {
			++out.prefetched_rowgroups;
		}
		if (timeline_enabled) {
			auto& row         = timeline_row_for(result.rowgroup_index);
			row.from_prefetch = from_prefetch;
			row.storage_bytes = result.storage_bytes;
			row.timing        = timing;
			row.prefetch      = result.prefetch;
		}
	}

	void on_assemble_expr(const double ms) {
		out.assemble_expr_ms += ms;
	}

	void on_rowgroup_stats(const size_t active_columns, const size_t work_items, const size_t logical_bytes) {
		out.total_columns += active_columns;
		out.total_items += work_items;
		out.total_bytes += logical_bytes;
		++out.total_rgs;
	}

	void on_append_expr(const double ms) {
		out.append_expr_ms += ms;
	}

	void on_upload_workset(const double                    ms,
	                       const runtime::UploadBreakdown& breakdown,
	                       const size_t                    payload_arena_bytes,
	                       const size_t                    output_arena_bytes) {
		out.upload_workset_ms += ms;
		out.upload_prep_ms += breakdown.prep_ms;
		out.upload_prep_reset_ms += breakdown.prep_reset_ms;
		out.upload_prep_output_arena_ms += breakdown.prep_output_arena_ms;
		out.upload_prep_bind_ms += breakdown.prep_bind_ms;
		out.upload_prep_slots_ms += breakdown.prep_slots_ms;
		out.upload_arena_pack_ms += breakdown.arena_pack_ms;
		out.upload_layout_ms += breakdown.arena.layout_ms;
		out.upload_alloc_ms += breakdown.arena.alloc_ms;
		out.upload_resolve_ms += breakdown.arena.resolve_ms;
		out.upload_pack_ms += breakdown.arena.pack_ms;
		out.upload_dma_issue_ms += breakdown.arena.dma_issue_ms;
		out.upload_dma_gpu_ms += breakdown.arena.dma_gpu_ms;
		out.upload_event_ms += breakdown.event_record_ms;
		out.total_payload_arena_bytes += payload_arena_bytes;
		out.total_output_arena_bytes += output_arena_bytes;
		out.total_h2d_bytes += breakdown.arena.dma_bytes;
		out.total_h2d_copies += breakdown.arena.dma_count;
	}

	void on_rowgroup_upload(const size_t                                      rowgroup_index,
	                        const std::chrono::steady_clock::time_point& upload_start,
	                        const std::chrono::steady_clock::time_point& upload_end) {
		if (!timeline_enabled) {
			return;
		}
		auto& row                       = timeline_row_for(rowgroup_index);
		row.timing.timeline.upload_start = upload_start;
		row.timing.timeline.upload_end   = upload_end;
	}

	void on_kernel(const double ms, const size_t launch_grid, const size_t launches) {
		out.kernel_ms += ms;
		out.total_launches += launches;
		out.total_launch_grid += launch_grid * launches;
	}

	void on_release_workset(const double ms) {
		out.release_device_ms += ms;
	}

	void on_free_rowgroup(const double ms) {
		out.free_rowgroup_ms += ms;
	}

	void on_prefetch_wait(const double ms) {
		out.prefetch_wait_ms += ms;
	}

	void on_reader_open(const double ms) {
		out.reader_open_ms += ms;
	}

	void on_descriptor_load(const double ms) {
		out.descriptor_load_ms += ms;
	}

	void on_pinned_pool_create(const double ms) {
		out.pinned_pool_create_ms += ms;
	}

	void on_max_storage_scan(const double ms) {
		out.max_storage_scan_ms += ms;
	}

	void on_pinned_pool_prewarm(const double ms) {
		out.pinned_pool_prewarm_ms += ms;
	}

	void on_prefetch_queue_start(const double ms) {
		out.prefetch_queue_start_ms += ms;
	}

	void on_pipeline_setup_total(const double ms) {
		out.pipeline_setup_total_ms += ms;
	}

	void write_timeline_csv() const {
		if (!timeline_enabled) {
			return;
		}

		std::ofstream csv(timeline_path);
		if (!csv) {
			throw std::runtime_error("failed to open GALP_ROWGROUP_TIMELINE_CSV: " + timeline_path.string());
		}

		const auto fmt_time = [&](const std::chrono::steady_clock::time_point tp) {
			if (tp == std::chrono::steady_clock::time_point {}) {
				return std::string {};
			}
			const double ms = std::chrono::duration<double, std::milli>(tp - timeline_base).count();
			std::ostringstream out;
			out << std::fixed << std::setprecision(6) << ms;
			return out.str();
		};
		const auto fmt_duration = [](const std::chrono::steady_clock::time_point start,
		                             const std::chrono::steady_clock::time_point end) {
			if (start == std::chrono::steady_clock::time_point {} ||
			    end == std::chrono::steady_clock::time_point {}) {
				return std::string {};
			}
			const double ms = std::chrono::duration<double, std::milli>(end - start).count();
			std::ostringstream out;
			out << std::fixed << std::setprecision(6) << ms;
			return out.str();
		};

		csv << "rowgroup_index,from_prefetch,storage_bytes,"
		       "worker_id,pool_slot_owner_reused,pool_slot_owner_migrated,pool_slot_allocated,"
		       "read_submit_ms,read_start_ms,file_read_start_ms,pread_start_ms,pread_end_ms,file_read_end_ms,"
		       "raw_ready_push_ms,raw_ready_pop_ms,build_start_ms,build_end_ms,ready_push_ms,consumer_pop_ms,"
		       "consumer_wait_start_ms,consumer_wait_end_ms,upload_start_ms,upload_end_ms,read_end_ms,"
		       "depth_block_ms,byte_block_ms,pinned_acquire_ms,pread_ms,zero_copy_view_setup_ms,rowgroup_build_ms,read_ms,"
		       "raw_queue_wait_ms,read_complete_to_build_start_ms,build_end_to_ready_push_ms,"
		       "ready_to_consumer_pop_ms,consumer_wait_ms,consumer_pop_to_upload_start_ms,upload_ms\n";
		for (const auto& row : timeline_rows) {
			const auto& timeline = row.timing.timeline;
			csv << row.rowgroup_index << ',' << (row.from_prefetch ? 1 : 0) << ',' << row.storage_bytes << ','
			    << row.prefetch.worker_id << ',' << (row.prefetch.pool_slot_owner_reused ? 1 : 0) << ','
			    << (row.prefetch.pool_slot_owner_migrated ? 1 : 0) << ','
			    << (row.prefetch.pool_slot_allocated ? 1 : 0) << ','
			    << fmt_time(timeline.read_submit) << ',' << fmt_time(timeline.read_start) << ','
			    << fmt_time(timeline.file_read_start) << ',' << fmt_time(timeline.pread_start) << ','
			    << fmt_time(timeline.pread_end) << ',' << fmt_time(timeline.file_read_end) << ','
			    << fmt_time(timeline.raw_ready_push) << ',' << fmt_time(timeline.raw_ready_pop) << ','
			    << fmt_time(timeline.build_start) << ',' << fmt_time(timeline.build_end) << ','
			    << fmt_time(timeline.ready_push) << ',' << fmt_time(timeline.consumer_pop) << ','
			    << fmt_time(timeline.consumer_wait_start) << ',' << fmt_time(timeline.consumer_wait_end) << ','
			    << fmt_time(timeline.upload_start) << ',' << fmt_time(timeline.upload_end) << ','
			    << fmt_time(timeline.read_end) << ',' << row.prefetch.depth_block_ms << ','
			    << row.prefetch.byte_block_ms << ','
			    << row.timing.pinned_acquire_ms << ',' << row.timing.pread_ms << ','
			    << row.timing.zero_copy_view_setup_ms << ',' << row.timing.rowgroup_build_ms << ','
			    << row.timing.read_ms << ',' << fmt_duration(timeline.raw_ready_push, timeline.raw_ready_pop) << ','
			    << fmt_duration(timeline.file_read_end, timeline.build_start) << ','
			    << fmt_duration(timeline.build_end, timeline.ready_push) << ','
			    << fmt_duration(timeline.ready_push, timeline.consumer_pop) << ','
			    << fmt_duration(timeline.consumer_wait_start, timeline.consumer_wait_end) << ','
			    << fmt_duration(timeline.consumer_pop, timeline.upload_start) << ','
			    << fmt_duration(timeline.upload_start, timeline.upload_end) << '\n';
		}
	}
};

} // namespace

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg) {
	warmup_cuda_runtime_once();

	TableBenchmarkResult out {};
	out.samples = cfg.samples;

	runtime::TableExecutionRequest request {};
	request.config                       = cfg;
	request.config.execution.write_out   = cfg.include_materialize || cfg.execution.write_out;
	request.samples                      = cfg.samples;
	request.rowgroup                     = cfg.rowgroup;
	request.materialize_results          = cfg.include_materialize;
	request.direct_append_no_materialize = !cfg.include_materialize;
	request.warmup_first_run             = true;
	request.load_column_names            = false;

	BenchmarkObserver observer {out};
	const auto        wall_start = std::chrono::steady_clock::now();
	if (cfg.reuse_table_resources) {
		const auto prepare_start = std::chrono::steady_clock::now();
		auto       resources     = runtime::prepare_table_resources(fls_path, request, observer);
		const auto prepare_end   = std::chrono::steady_clock::now();
		out.resource_prepare_ms = std::chrono::duration<double, std::milli>(prepare_end - prepare_start).count();

		const auto query_start = std::chrono::steady_clock::now();
		observer.configure_timeline(query_start);
		runtime::execute_prepared_table_pipeline(
		    resources,
		    request,
		    [](size_t) { return true; },
		    [](size_t, galp::format::Rowgroup&, const std::vector<galp::expression::Expression>&, const RowgroupData*) {},
		    observer,
		    query_start);
		const auto query_end = std::chrono::steady_clock::now();
		out.query_wall_ms = std::chrono::duration<double, std::milli>(query_end - query_start).count();
	} else {
		observer.configure_timeline(wall_start);
		runtime::execute_table_pipeline(
		    fls_path,
		    request,
		    [](size_t) { return true; },
		    [](size_t, galp::format::Rowgroup&, const std::vector<galp::expression::Expression>&, const RowgroupData*) {},
		    observer);
		const auto query_end = std::chrono::steady_clock::now();
		out.query_wall_ms = std::chrono::duration<double, std::milli>(query_end - wall_start).count();
	}
	const auto wall_end = std::chrono::steady_clock::now();
	observer.write_timeline_csv();
	out.end_to_end_ms   = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
	out.pipeline_active_ms = out.query_wall_ms > out.pipeline_setup_total_ms ? out.query_wall_ms - out.pipeline_setup_total_ms
	                                                                         : 0.0;
	return out;
}

} // namespace galp::execution
