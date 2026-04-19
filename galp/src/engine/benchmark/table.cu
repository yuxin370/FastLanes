// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/benchmark/table.cu
// ────────────────────────────────────────────────────────
#include "engine/benchmark/table.cuh"
#include "engine/execution/internal/materialize.cuh"
#include "engine/execution/internal/streaming_pipeline.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/expression.cuh"
#include "engine/reader.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/datatype_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "flsgpu/host-utils.cuh"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dispatch {
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

size_t data_type_size(const fastlanes::DataType dt) {
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

fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};

	fastlanes::FileHeader::Load(header, file_path);
	fastlanes::FileFooter::Load(footer, file_path);

	if (header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file_path, footer.table_descriptor_offset, footer.table_descriptor_size, /*verify=*/true);
	}

	return fastlanes::TableDescriptorHandle::FromFile(file_path.parent_path() / "table_descriptor.fbb",
	                                                  /*verify=*/true);
}

size_t rowgroup_bytes(const fastlanes::RowgroupDescriptor* rg) {
	if (!rg) {
		return 0;
	}
	size_t      rg_bytes = 0;
	const auto* cols     = rg->m_column_descriptors();
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
		rg_bytes += static_cast<size_t>(rg->m_n_tuples()) * element_size;
	}
	return rg_bytes;
}

void check_rowgroup_index(const size_t n_rowgroups, const std::optional<size_t>& rowgroup) {
	if (rowgroup.has_value() && *rowgroup >= n_rowgroups) {
		throw std::out_of_range("rowgroup index out of range");
	}
}

size_t count_active_columns(const std::vector<expr::Expression>& expressions) {
	return runtime::count_active_columns(expressions);
}

struct StreamingBenchmarkChunk {
	runtime::ExecutionWorkset  workset {};
	runtime::AsyncWorksetRun   run {};
	struct PendingRowgroup {
		reader::Rowgroup              rowgroup {};
		std::vector<expr::Expression> expressions;
		size_t                        active_columns = 0;
	};
	std::vector<PendingRowgroup> pending_rowgroups;
	size_t                     work_items  = 0;
	size_t                     rowgroups   = 0;
	size_t                     active_columns = 0;
	size_t                     launch_grid = 0;
	size_t                     launches    = 0;
	bool                       submitted   = false;
};

struct RowgroupReadResult {
	reader::Rowgroup rowgroup          {};
	double           read_ms           = 0.0;
	double           file_read_ms      = 0.0;
	double           rowgroup_build_ms = 0.0;
};

class PinnedRowgroupBufferPool : public std::enable_shared_from_this<PinnedRowgroupBufferPool> {
public:
	struct Lease {
		std::shared_ptr<void> owner;
		std::byte*            data     = nullptr;
		size_t                capacity = 0;
	};

	// Construct only via create() — enable_shared_from_this requires the
	// instance be owned by a shared_ptr before shared_from_this() is called.
	static std::shared_ptr<PinnedRowgroupBufferPool> create(const size_t slots) {
		return std::shared_ptr<PinnedRowgroupBufferPool>(new PinnedRowgroupBufferPool(slots));
	}

	~PinnedRowgroupBufferPool() {
		for (auto& slot : slots_) {
			if (slot.ptr != nullptr) {
				flsgpu::memory::DevicePool::instance().release_pinned(slot.ptr);
			}
		}
	}

	// Wake any thread currently blocked in acquire() so it can exit with an
	// exception instead of deadlocking during teardown (e.g. if a prefetch
	// worker filled the pool and another worker errored before it drained).
	void request_stop() {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stopping_ = true;
		}
		cv_.notify_all();
	}

	Lease acquire(const size_t min_bytes) {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [&]() {
			return stopping_ || std::any_of(slots_.begin(), slots_.end(), [](const Slot& slot) { return !slot.in_use; });
		});
		if (stopping_) {
			throw std::runtime_error("pinned rowgroup buffer pool stopped");
		}

		for (size_t idx = 0; idx < slots_.size(); ++idx) {
			auto& slot = slots_[idx];
			if (slot.in_use) {
				continue;
			}
			if (slot.capacity < min_bytes) {
				if (slot.ptr != nullptr) {
					flsgpu::memory::DevicePool::instance().release_pinned(slot.ptr);
				}
				const size_t alloc_bytes = round_up_capacity(min_bytes);
				slot.ptr                 = flsgpu::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
				slot.capacity            = alloc_bytes;
			}
			slot.in_use = true;
			// Capture a weak_ptr in the deleter so a lease that outlives the
			// pool (during abnormal teardown) skips release() on a destroyed
			// object instead of dereferencing a dangling `this`. Under normal
			// shutdown the pool is alive until all leases drop.
			std::weak_ptr<PinnedRowgroupBufferPool> weak_self = weak_from_this();
			return Lease {
			    std::shared_ptr<void>(slot.ptr,
			                          [weak_self, idx](void*) {
				                          if (auto self = weak_self.lock()) {
					                          self->release(idx);
				                          }
			                          }),
			    reinterpret_cast<std::byte*>(slot.ptr),
			    slot.capacity,
			};
		}

		throw std::runtime_error("pinned rowgroup buffer pool exhausted");
	}

private:
	explicit PinnedRowgroupBufferPool(const size_t slots) : slots_(std::max<size_t>(1, slots)) {}
	struct Slot {
		void*  ptr      = nullptr;
		size_t capacity = 0;
		bool   in_use   = false;
	};

	static size_t round_up_capacity(const size_t bytes) {
		constexpr size_t kAlign = 64U * 1024U;
		if (bytes == 0) {
			return kAlign;
		}
		return ((bytes + kAlign - 1U) / kAlign) * kAlign;
	}

	void release(const size_t idx) {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			slots_[idx].in_use = false;
		}
		cv_.notify_one();
	}

	std::mutex                  mutex_;
	std::condition_variable     cv_;
	std::vector<Slot>           slots_;
	bool                        stopping_ = false;
};

class RowgroupPrefetchQueue {
public:
	RowgroupPrefetchQueue(const std::filesystem::path& file_path,
	                      const size_t                start,
	                      const size_t                end,
	                      const bool                  use_zero_copy_parse,
	                      const size_t                depth,
	                      const size_t                num_workers,
	                      std::shared_ptr<PinnedRowgroupBufferPool> pinned_pool = {})
	    : start_(start)
	    , end_(end)
	    , depth_(std::max<size_t>(1, depth))
	    , next_claim_(start)
	    , next_out_(start)
	    // Ring buffer sized to the configured depth — not the full table.
	    // cv_not_full_ gates a worker from writing a slot until the consumer
	    // has drained the previous occupant (rg_idx - next_out < depth), so
	    // each slot index (rg_idx % depth) is exclusive at any moment.
	    , slots_(std::max<size_t>(1, depth))
	    , pinned_pool_(pinned_pool) {
		worker_count_ = std::max<size_t>(1, std::min<size_t>(num_workers, end > start ? end - start : 1));
		workers_.reserve(worker_count_);
		for (size_t w = 0; w < worker_count_; ++w) {
			workers_.emplace_back(
			    [this, file_path, use_zero_copy_parse, pinned_pool]() {
				    try {
					    reader::reader rdr(file_path);
					    while (true) {
						    const size_t rg_idx = next_claim_.fetch_add(1, std::memory_order_relaxed);
						    if (rg_idx >= end_) {
							    break;
						    }
						    {
							    std::unique_lock<std::mutex> lock(mutex_);
							    cv_not_full_.wait(lock, [&]() {
								    return stop_ || (rg_idx - next_out_.load(std::memory_order_relaxed) < depth_);
							    });
							    if (stop_) {
								    return;
							    }
						    }

						    Prefetched prefetched {};
						    if (use_zero_copy_parse) {
							    const auto file_read_start = std::chrono::steady_clock::now();
							    reader::ZeroCopyRowgroup zero_copy {};
							    if (pinned_pool) {
								    auto lease = pinned_pool->acquire(rdr.rowgroup_storage_bytes(rg_idx));
								    zero_copy  = rdr.read_rowgroup_zero_copy_into(
								        rg_idx, std::move(lease.owner), lease.data, lease.capacity,
								        /*backing_is_pinned=*/true);
							    } else {
								    zero_copy = rdr.read_rowgroup_zero_copy(rg_idx);
							    }
							    const auto file_read_end = std::chrono::steady_clock::now();
							    const auto build_start   = std::chrono::steady_clock::now();
							    prefetched.rowgroup      = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
							    const auto build_end     = std::chrono::steady_clock::now();
							    prefetched.file_read_ms =
							        std::chrono::duration<double, std::milli>(file_read_end - file_read_start).count();
							    prefetched.rowgroup_build_ms =
							        std::chrono::duration<double, std::milli>(build_end - build_start).count();
							    prefetched.read_ms = prefetched.file_read_ms + prefetched.rowgroup_build_ms;
						    } else {
							    const auto read_start    = std::chrono::steady_clock::now();
							    prefetched.rowgroup      = rdr.read_rowgroup(rg_idx);
							    const auto read_end      = std::chrono::steady_clock::now();
							    prefetched.rowgroup_build_ms =
							        std::chrono::duration<double, std::milli>(read_end - read_start).count();
							    prefetched.read_ms = prefetched.rowgroup_build_ms;
						    }

						    {
							    std::lock_guard<std::mutex> lock(mutex_);
							    if (stop_) {
								    return;
							    }
							    slots_[(rg_idx - start_) % depth_] = std::move(prefetched);
							    ++ready_count_;
							    cv_not_empty_.notify_all();
						    }
					    }
				    } catch (...) {
					    std::lock_guard<std::mutex> lock(mutex_);
					    if (!error_) {
						    error_ = std::current_exception();
					    }
					    cv_not_empty_.notify_all();
				    }

				    std::lock_guard<std::mutex> lock(mutex_);
				    if (++workers_finished_ == worker_count_) {
					    done_ = true;
					    cv_not_empty_.notify_all();
				    }
			    });
		}
	}

	~RowgroupPrefetchQueue() {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stop_ = true;
		}
		cv_not_full_.notify_all();
		cv_not_empty_.notify_all();
		// Wake any worker blocked inside pinned_pool_->acquire() so shutdown can
		// proceed instead of deadlocking on a lease that will never return.
		if (pinned_pool_) {
			pinned_pool_->request_stop();
		}
		for (auto& w : workers_) {
			if (w.joinable()) {
				w.join();
			}
		}
	}

	RowgroupReadResult pop() {
		const auto         wait_start = std::chrono::steady_clock::now();
		std::exception_ptr pending_error;
		Prefetched         item;
		bool               underflow = false;
		const size_t       slot      = (next_out_.load(std::memory_order_relaxed) - start_) % depth_;
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cv_not_empty_.wait(lock, [&]() {
				return stop_ || error_ != nullptr || slots_[slot].has_value() || done_;
			});
			const auto wait_end = std::chrono::steady_clock::now();
			wait_ms_ += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();

			if (error_ != nullptr) {
				pending_error = error_;
			} else if (!slots_[slot].has_value()) {
				underflow = true;
			} else {
				item = std::move(*slots_[slot]);
				slots_[slot].reset();
				--ready_count_;
				next_out_.fetch_add(1, std::memory_order_relaxed);
			}
		}
		cv_not_full_.notify_all();

		if (pending_error) {
			std::rethrow_exception(pending_error);
		}
		if (underflow) {
			throw std::runtime_error("rowgroup prefetch queue underflow");
		}
		return RowgroupReadResult {std::move(item.rowgroup), item.read_ms, item.file_read_ms, item.rowgroup_build_ms};
	}

	// Only safe to call from the consumer thread (the one that owns pop()).
	double wait_ms() const {
		return wait_ms_;
	}

private:
	struct Prefetched {
		reader::Rowgroup rowgroup          {};
		double           read_ms           = 0.0;
		double           file_read_ms      = 0.0;
		double           rowgroup_build_ms = 0.0;
	};

	const size_t                           start_     = 0;
	const size_t                           end_       = 0;
	const size_t                           depth_     = 1;
	std::atomic<size_t>                    next_claim_;
	std::atomic<size_t>                    next_out_;
	std::vector<std::thread>               workers_;
	size_t                                 worker_count_ = 0;
	mutable std::mutex                     mutex_;
	std::condition_variable                cv_not_empty_;
	std::condition_variable                cv_not_full_;
	std::vector<std::optional<Prefetched>> slots_;
	size_t                                 ready_count_      = 0;
	size_t                                 workers_finished_ = 0;
	std::exception_ptr                     error_;
	bool                                   done_    = false;
	bool                                   stop_    = false;
	double                                 wait_ms_ = 0.0;
	std::shared_ptr<PinnedRowgroupBufferPool> pinned_pool_;
};

void reset_streaming_chunk(StreamingBenchmarkChunk& chunk) {
	chunk.pending_rowgroups.clear();
	chunk.work_items  = 0;
	chunk.rowgroups   = 0;
	chunk.active_columns = 0;
	chunk.launch_grid = 0;
	chunk.launches    = 0;
	chunk.submitted   = false;
}

void accumulate_rowgroup_stats(TableBenchmarkResult& result,
                               const size_t          rg_columns,
                               const size_t          rg_vectors,
                               const size_t          bytes,
                               const size_t          payload_arena_bytes,
                               const size_t          output_arena_bytes,
                               const double          read_rowgroup_ms,
                               const double          file_read_ms,
                               const double          rowgroup_build_ms,
                               const double          assemble_expr_ms,
                               const double          append_expr_ms,
                               const double          upload_workset_ms,
                               const double          kernel_ms,
                               const double          release_device_ms,
                               const double          free_rowgroup_ms,
                               const size_t          launch_grid,
                               const size_t          launches) {
	result.read_rowgroup_ms += read_rowgroup_ms;
	result.file_read_ms += file_read_ms;
	result.rowgroup_build_ms += rowgroup_build_ms;
	result.assemble_expr_ms += assemble_expr_ms;
	result.append_expr_ms += append_expr_ms;
	result.upload_workset_ms += upload_workset_ms;
	result.kernel_ms += kernel_ms;
	result.release_device_ms += release_device_ms;
	result.free_rowgroup_ms += free_rowgroup_ms;
	result.total_launches += launches;
	result.total_launch_grid += launch_grid * launches;
	result.total_columns += rg_columns;
	result.total_items += rg_vectors;
	result.total_bytes += bytes;
	result.total_payload_arena_bytes += payload_arena_bytes;
	result.total_output_arena_bytes += output_arena_bytes;
	++result.total_rgs;
}

void accumulate_upload_breakdown(TableBenchmarkResult& out, const runtime::ExecutionWorkset& workset) {
	const auto& ub = workset.last_upload_breakdown;
	out.upload_prep_ms               += ub.prep_ms;
	out.upload_prep_reset_ms         += ub.prep_reset_ms;
	out.upload_prep_output_arena_ms  += ub.prep_output_arena_ms;
	out.upload_prep_bind_ms          += ub.prep_bind_ms;
	out.upload_prep_slots_ms         += ub.prep_slots_ms;
	out.upload_arena_pack_ms         += ub.arena_pack_ms;
	out.upload_event_ms              += ub.event_record_ms;
	if (workset.chunk_arena) {
		const auto& phase = workset.chunk_arena->last_upload_phase_ms;
		out.upload_layout_ms    += phase.layout_ms;
		out.upload_alloc_ms     += phase.alloc_ms;
		out.upload_resolve_ms   += phase.resolve_ms;
		out.upload_pack_ms      += phase.pack_ms;
		out.upload_dma_issue_ms += phase.dma_issue_ms;
	}
}

} // namespace

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg) {
	TableBenchmarkResult out {};
	out.samples = cfg.samples;

	reader::reader rdr(fls_path);
	const size_t   n_rowgroups = rdr.rowgroup_count();
	check_rowgroup_index(n_rowgroups, cfg.rowgroup);

	size_t start = 0;
	size_t end   = n_rowgroups;
	if (cfg.rowgroup.has_value()) {
		start = *cfg.rowgroup;
		end   = start + 1;
	}

	const auto  td_handle = load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	if (!td) {
		throw std::runtime_error("failed to load table descriptor");
	}

	warmup_cuda_runtime_once();
	const auto wall_start = std::chrono::steady_clock::now();
	const bool whole_table = !cfg.rowgroup.has_value() && cfg.aggregation_scope == AggregationScope::WholeTable;
	if (cfg.streaming_target_work_items == 0) {
		throw std::invalid_argument("streaming_target_work_items must be > 0");
	}
	if (cfg.prefetch_depth == 0) {
		throw std::invalid_argument("prefetch_depth must be > 0");
	}
	const size_t stream_target_work_items = cfg.streaming_target_work_items;
	const size_t stream_target_rowgroups  = cfg.streaming_target_rowgroups;
	const bool   use_rowgroup_prefetch    = whole_table && cfg.enable_rowgroup_prefetch;
	// Without a per-chunk rowgroup cap, the items-only threshold can let a
	// single chunk grow to the full table. Size the pool to that worst case so
	// the main thread never blocks on acquire() with every lease held by its
	// own unflushed chunk (deadlock: nothing else can call consume_chunk()).
	const size_t max_rowgroups_per_chunk = (stream_target_rowgroups > 0)
	                                           ? stream_target_rowgroups
	                                           : (n_rowgroups > 0 ? n_rowgroups : 1U);
	std::shared_ptr<PinnedRowgroupBufferPool> pinned_rowgroup_pool;
	if (whole_table && cfg.use_zero_copy_parse) {
		const size_t pooled_slots =
		    (cfg.enable_streaming ? (2U * max_rowgroups_per_chunk) : max_rowgroups_per_chunk)
		    + cfg.prefetch_depth + std::max<size_t>(1, cfg.prefetch_workers) + 2U;
		pinned_rowgroup_pool = PinnedRowgroupBufferPool::create(pooled_slots);
	}

	if (!whole_table) {
		for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
			const auto* rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			const size_t bytes = rowgroup_bytes(rg);

			RowgroupReadResult read_result {};
			if (cfg.use_zero_copy_parse) {
				const auto file_read_start = std::chrono::steady_clock::now();
				auto       zero_copy       = rdr.read_rowgroup_zero_copy(rg_idx);
				const auto file_read_end   = std::chrono::steady_clock::now();
				const auto build_start     = std::chrono::steady_clock::now();
				read_result.rowgroup       = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
				const auto build_end       = std::chrono::steady_clock::now();
				read_result.file_read_ms =
				    std::chrono::duration<double, std::milli>(file_read_end - file_read_start).count();
				read_result.rowgroup_build_ms =
				    std::chrono::duration<double, std::milli>(build_end - build_start).count();
			} else {
				const auto read_start = std::chrono::steady_clock::now();
				read_result.rowgroup  = rdr.read_rowgroup(rg_idx);
				const auto read_end   = std::chrono::steady_clock::now();
				read_result.rowgroup_build_ms =
				    std::chrono::duration<double, std::milli>(read_end - read_start).count();
			}
			read_result.read_ms = read_result.file_read_ms + read_result.rowgroup_build_ms;

			const auto assemble_start = std::chrono::steady_clock::now();
			auto       expressions    = expr::assemble(read_result.rowgroup);
			const auto assemble_end   = std::chrono::steady_clock::now();
			const double assemble_expr_ms =
			    std::chrono::duration<double, std::milli>(assemble_end - assemble_start).count();

			const size_t rg_columns = count_active_columns(expressions);
			const size_t rg_vectors = rg_columns * read_result.rowgroup.n_vecs;

			runtime::ExecutionWorkset      workset {};
			runtime::ExecutionWorksetGuard guard(workset);
			const auto append_start = std::chrono::steady_clock::now();
			runtime::append_expressions(workset, expressions, cfg.execution);
			const auto append_end     = std::chrono::steady_clock::now();
			const double append_expr_ms = std::chrono::duration<double, std::milli>(append_end - append_start).count();

			const auto upload_start = std::chrono::steady_clock::now();
			runtime::upload_workset(workset);
			const auto upload_end       = std::chrono::steady_clock::now();
			const double upload_workset_ms =
			    std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
			accumulate_upload_breakdown(out, workset);

			size_t       rg_launch_grid = 0;
			size_t       rg_launches    = 0;
			const double kernel_ms =
			    runtime::run_workset(workset, cfg.samples, cfg.execution, &rg_launch_grid, &rg_launches, true);

			// release_workset() clears output_arena_used_bytes; snapshot before.
			const size_t output_arena_bytes_snapshot = workset.output_arena_used_bytes;
			const auto release_start = std::chrono::steady_clock::now();
			guard.dismiss();
			runtime::release_workset(workset);
			const auto release_end = std::chrono::steady_clock::now();
			const double release_device_ms = std::chrono::duration<double, std::milli>(release_end - release_start).count();

			const auto free_start = std::chrono::steady_clock::now();
			free_rowgroup(read_result.rowgroup);
			const auto free_end = std::chrono::steady_clock::now();
			const double free_rowgroup_ms = std::chrono::duration<double, std::milli>(free_end - free_start).count();

			accumulate_rowgroup_stats(out,
			                          rg_columns,
			                          rg_vectors,
			                          bytes,
			                          workset.payload_arena_bytes,
			                          output_arena_bytes_snapshot,
			                          read_result.read_ms,
			                          read_result.file_read_ms,
			                          read_result.rowgroup_build_ms,
			                          assemble_expr_ms,
			                          append_expr_ms,
			                          upload_workset_ms,
			                          kernel_ms,
			                          release_device_ms,
			                          free_rowgroup_ms,
			                          rg_launch_grid,
			                          rg_launches);
		}
		const auto wall_end = std::chrono::steady_clock::now();
		out.end_to_end_ms   = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
		return out;
	}

	std::unique_ptr<RowgroupPrefetchQueue> prefetch_queue;
	if (use_rowgroup_prefetch) {
		prefetch_queue = std::make_unique<RowgroupPrefetchQueue>(
		    fls_path, start, end, cfg.use_zero_copy_parse, cfg.prefetch_depth,
		    cfg.prefetch_workers, pinned_rowgroup_pool);
	}

	const auto fetch_rowgroup = [&](const size_t rg_idx) -> RowgroupReadResult {
		RowgroupReadResult result {};
		if (prefetch_queue) {
			result = prefetch_queue->pop();
			++out.prefetched_rowgroups;
		} else {
			if (cfg.use_zero_copy_parse) {
				const auto file_read_start = std::chrono::steady_clock::now();
				reader::ZeroCopyRowgroup zero_copy {};
				if (pinned_rowgroup_pool) {
					auto lease = pinned_rowgroup_pool->acquire(rdr.rowgroup_storage_bytes(rg_idx));
					zero_copy  = rdr.read_rowgroup_zero_copy_into(
					    rg_idx, std::move(lease.owner), lease.data, lease.capacity,
					    /*backing_is_pinned=*/true);
				} else {
					zero_copy = rdr.read_rowgroup_zero_copy(rg_idx);
				}
				const auto file_read_end   = std::chrono::steady_clock::now();
				const auto build_start     = std::chrono::steady_clock::now();
				result.rowgroup            = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
				const auto build_end       = std::chrono::steady_clock::now();
				result.file_read_ms =
				    std::chrono::duration<double, std::milli>(file_read_end - file_read_start).count();
				result.rowgroup_build_ms =
				    std::chrono::duration<double, std::milli>(build_end - build_start).count();
			} else {
				const auto read_start = std::chrono::steady_clock::now();
				result.rowgroup       = rdr.read_rowgroup(rg_idx);
				const auto read_end   = std::chrono::steady_clock::now();
				result.rowgroup_build_ms =
				    std::chrono::duration<double, std::milli>(read_end - read_start).count();
			}
			result.read_ms = result.file_read_ms + result.rowgroup_build_ms;
		}
		out.read_rowgroup_ms += result.read_ms;
		out.file_read_ms += result.file_read_ms;
		out.rowgroup_build_ms += result.rowgroup_build_ms;
		return std::move(result);
	};

	if (!cfg.enable_streaming) {
		runtime::ExecutionWorkset   workset {};
		std::vector<StreamingBenchmarkChunk::PendingRowgroup> pending_rowgroups;
		size_t                      chunk_work_items = 0;
		size_t                      chunk_rowgroups  = 0;
		size_t                      chunk_active_columns = 0;
		bool                        did_warmup       = false;

		const auto run_sync_chunk = [&]() {
			if (chunk_rowgroups == 0) {
				return;
			}

			runtime::begin_workset_chunk_arena(workset, chunk_active_columns);
			const auto append_start = std::chrono::steady_clock::now();
			for (auto& pending : pending_rowgroups) {
				runtime::append_expressions(workset, pending.expressions, cfg.execution);
			}
			const auto append_end = std::chrono::steady_clock::now();
			out.append_expr_ms += std::chrono::duration<double, std::milli>(append_end - append_start).count();

			const auto upload_start = std::chrono::steady_clock::now();
			runtime::upload_workset(workset);
			const auto upload_end = std::chrono::steady_clock::now();
			out.upload_workset_ms += std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
			accumulate_upload_breakdown(out, workset);
			out.total_payload_arena_bytes += workset.payload_arena_bytes;
			out.total_output_arena_bytes += workset.output_arena_used_bytes;

			size_t chunk_launch_grid = 0;
			size_t chunk_launches    = 0;
			const bool warmup_once   = !did_warmup;
			out.kernel_ms += runtime::run_workset(
			    workset, cfg.samples, cfg.execution, &chunk_launch_grid, &chunk_launches, warmup_once);
			did_warmup = true;
			out.total_launches += chunk_launches;
			out.total_launch_grid += chunk_launch_grid * chunk_launches;

			const auto release_start = std::chrono::steady_clock::now();
			runtime::release_workset(workset);
			const auto release_end = std::chrono::steady_clock::now();
			out.release_device_ms += std::chrono::duration<double, std::milli>(release_end - release_start).count();

			for (auto& pending : pending_rowgroups) {
				const auto free_start = std::chrono::steady_clock::now();
				free_rowgroup(pending.rowgroup);
				const auto free_end = std::chrono::steady_clock::now();
				out.free_rowgroup_ms += std::chrono::duration<double, std::milli>(free_end - free_start).count();
			}
			pending_rowgroups.clear();
			chunk_work_items = 0;
			chunk_rowgroups  = 0;
			chunk_active_columns = 0;
		};

		for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
			const auto* rg     = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			const size_t bytes = rowgroup_bytes(rg);

			auto read_result = fetch_rowgroup(rg_idx);
			auto rowgroup    = std::move(read_result.rowgroup);

			const auto assemble_start = std::chrono::steady_clock::now();
			auto       expressions    = expr::assemble(rowgroup);
			const auto assemble_end   = std::chrono::steady_clock::now();
			out.assemble_expr_ms += std::chrono::duration<double, std::milli>(assemble_end - assemble_start).count();

			const size_t rg_columns = count_active_columns(expressions);
			const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

			chunk_work_items += rg_vectors;
			++chunk_rowgroups;
			chunk_active_columns += rg_columns;
			pending_rowgroups.push_back(
			    StreamingBenchmarkChunk::PendingRowgroup {std::move(rowgroup), std::move(expressions), rg_columns});

			out.total_columns += rg_columns;
			out.total_items += rg_vectors;
			out.total_bytes += bytes;
			++out.total_rgs;

			const bool reach_items     = chunk_work_items >= stream_target_work_items;
			const bool reach_rowgroups = (stream_target_rowgroups > 0) && (chunk_rowgroups >= stream_target_rowgroups);
			if (reach_items || reach_rowgroups) {
				run_sync_chunk();
			}
		}

		run_sync_chunk();
		if (prefetch_queue) {
			out.prefetch_wait_ms += prefetch_queue->wait_ms();
		}

		const auto wall_end = std::chrono::steady_clock::now();
		out.end_to_end_ms   = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
		return out;
	}

	runtime::StreamingDoubleBuffer<StreamingBenchmarkChunk> pipeline {};
	bool                                                    did_warmup = false;

	const auto submit_chunk = [&](StreamingBenchmarkChunk& chunk) {
		if (chunk.rowgroups == 0) {
			return;
		}
		runtime::begin_workset_chunk_arena(chunk.workset, chunk.active_columns);
		const auto append_start = std::chrono::steady_clock::now();
		for (auto& pending : chunk.pending_rowgroups) {
			runtime::append_expressions(chunk.workset, pending.expressions, cfg.execution);
		}
		const auto append_end = std::chrono::steady_clock::now();
		out.append_expr_ms += std::chrono::duration<double, std::milli>(append_end - append_start).count();

		const auto upload_start = std::chrono::steady_clock::now();
		runtime::upload_workset(chunk.workset);
		const auto upload_end = std::chrono::steady_clock::now();
		out.upload_workset_ms += std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
		accumulate_upload_breakdown(out, chunk.workset);
		out.total_payload_arena_bytes += chunk.workset.payload_arena_bytes;
		out.total_output_arena_bytes += chunk.workset.output_arena_used_bytes;

		const bool warmup_once = !did_warmup;
		chunk.run = runtime::run_workset_async(
		    chunk.workset, cfg.samples, cfg.execution, &chunk.launch_grid, &chunk.launches, warmup_once);
		chunk.submitted = true;
		did_warmup = true;
	};

	const auto consume_chunk = [&](StreamingBenchmarkChunk& chunk) {
		if (chunk.rowgroups == 0) {
			return;
		}
		if (chunk.submitted) {
			runtime::wait_workset_async(chunk.run);
			out.kernel_ms += chunk.run.elapsed_ms;
			out.total_launches += chunk.launches;
			out.total_launch_grid += chunk.launch_grid * chunk.launches;
		}

		const auto release_start = std::chrono::steady_clock::now();
		runtime::release_workset(chunk.workset, /*preserve_resources=*/true);
		const auto release_end = std::chrono::steady_clock::now();
		out.release_device_ms += std::chrono::duration<double, std::milli>(release_end - release_start).count();

		for (auto& pending : chunk.pending_rowgroups) {
			const auto free_start = std::chrono::steady_clock::now();
			free_rowgroup(pending.rowgroup);
			const auto free_end = std::chrono::steady_clock::now();
			out.free_rowgroup_ms += std::chrono::duration<double, std::milli>(free_end - free_start).count();
		}

		reset_streaming_chunk(chunk);
	};

	for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
		const auto* rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
		const size_t bytes = rowgroup_bytes(rg);

		auto read_result = fetch_rowgroup(rg_idx);
		auto rowgroup    = std::move(read_result.rowgroup);

		const auto assemble_start = std::chrono::steady_clock::now();
		auto       expressions    = expr::assemble(rowgroup);
		const auto assemble_end   = std::chrono::steady_clock::now();
		out.assemble_expr_ms += std::chrono::duration<double, std::milli>(assemble_end - assemble_start).count();

		auto&        chunk      = pipeline.build_chunk();
		const size_t rg_columns = count_active_columns(expressions);
		const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

		chunk.work_items += rg_vectors;
		++chunk.rowgroups;
		chunk.active_columns += rg_columns;

		out.total_columns += rg_columns;
		out.total_items += rg_vectors;
		out.total_bytes += bytes;
		++out.total_rgs;
		chunk.pending_rowgroups.push_back(
		    StreamingBenchmarkChunk::PendingRowgroup {std::move(rowgroup), std::move(expressions), rg_columns});

		const bool reach_items     = chunk.work_items >= stream_target_work_items;
		const bool reach_rowgroups = (stream_target_rowgroups > 0) && (chunk.rowgroups >= stream_target_rowgroups);
		if (reach_items || reach_rowgroups) {
			pipeline.submit_build_and_rotate(submit_chunk, consume_chunk);
		}
	}

	pipeline.flush(submit_chunk, consume_chunk);
	if (prefetch_queue) {
		out.prefetch_wait_ms += prefetch_queue->wait_ms();
	}

	const auto wall_end = std::chrono::steady_clock::now();
	out.end_to_end_ms    = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
	return out;
}

} // namespace dispatch
