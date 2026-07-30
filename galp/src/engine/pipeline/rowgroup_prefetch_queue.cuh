// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/pipeline/rowgroup_prefetch_queue.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_QUEUE_CUH
#define ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_QUEUE_CUH

#include "engine/pipeline/pinned_rowgroup_pool.cuh"
#include "engine/pipeline/rowgroup_prefetch_types.cuh"
#include "format/reader.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace galp::runtime {

inline std::vector<size_t> make_contiguous_rowgroup_schedule(const size_t start, const size_t end) {
	std::vector<size_t> rowgroups;
	rowgroups.reserve(end > start ? end - start : 0);
	for (size_t rowgroup_index = start; rowgroup_index < end; ++rowgroup_index) {
		rowgroups.push_back(rowgroup_index);
	}
	return rowgroups;
}

// Bounded ring-buffer prefetcher: worker threads read rowgroups ahead of the
// consumer while slots_[schedule_pos % depth] provides back-pressure.
class RowgroupPrefetchQueue {
public:
	RowgroupPrefetchQueue(std::shared_ptr<galp::format::FlsReader>  shared_reader,
	                      const size_t                              start,
	                      const size_t                              end,
	                      const size_t                              depth,
	                      const size_t                              num_workers,
	                      std::shared_ptr<PinnedRowgroupBufferPool> pinned_pool                = {},
	                      const size_t                              max_inflight_storage_bytes = 0,
	                      std::vector<std::vector<uint32_t>>        selected_vectors            = {})
	    : RowgroupPrefetchQueue(std::move(shared_reader),
	                            make_contiguous_rowgroup_schedule(start, end),
	                            depth,
	                            num_workers,
	                            std::move(pinned_pool),
	                            max_inflight_storage_bytes,
	                            std::move(selected_vectors)) {
	}

	RowgroupPrefetchQueue(std::shared_ptr<galp::format::FlsReader>  shared_reader,
	                      std::vector<size_t>                       rowgroups,
	                      const size_t                              depth,
	                      const size_t                              num_workers,
	                      std::shared_ptr<PinnedRowgroupBufferPool> pinned_pool                = {},
	                      const size_t                              max_inflight_storage_bytes = 0,
	                      std::vector<std::vector<uint32_t>>        selected_vectors            = {})
	    : rowgroups_(std::move(rowgroups))
	    , selected_vectors_(std::move(selected_vectors))
	    , depth_(std::max<size_t>(1, depth))
	    , max_inflight_storage_bytes_(max_inflight_storage_bytes)
	    , next_claim_(0)
	    , next_out_(0)
	    , next_budget_reserve_(0)
	    , slots_(std::max<size_t>(1, depth))
	    , pinned_pool_(pinned_pool)
	    , shared_reader_(std::move(shared_reader)) {
		if (!shared_reader_) {
			throw std::runtime_error("RowgroupPrefetchQueue: shared reader is null");
		}
		if (!selected_vectors_.empty() && selected_vectors_.size() != rowgroups_.size()) {
			throw std::invalid_argument("RowgroupPrefetchQueue: selected-vector schedule size mismatch");
		}
		const size_t reader_rowgroups = shared_reader_->rowgroup_count();
		for (const size_t rowgroup_index : rowgroups_) {
			if (rowgroup_index >= reader_rowgroups) {
				throw std::out_of_range("RowgroupPrefetchQueue: rowgroup schedule index out of range");
			}
		}
		if (rowgroups_.empty()) {
			done_ = true;
			return;
		}
		worker_count_ = std::max<size_t>(1, std::min<size_t>(num_workers, rowgroups_.size()));
		workers_.reserve(worker_count_);
		for (size_t w = 0; w < worker_count_; ++w) {
			workers_.emplace_back([this, pinned_pool, w]() {
				try {
					galp::format::FlsReader& rdr = *shared_reader_;
					while (true) {
						const size_t schedule_pos = next_claim_.fetch_add(1, std::memory_order_relaxed);
						if (schedule_pos >= rowgroups_.size()) {
							break;
						}
						const size_t rg_idx = rowgroups_[schedule_pos];

						const size_t storage_bytes         = rdr.rowgroup_storage_bytes(rg_idx);
						const auto   depth_block_start     = std::chrono::steady_clock::now();
						bool         blocked_by_bytes      = false;
						bool         budget_order_advanced = false;
						{
							std::unique_lock<std::mutex> lock(mutex_);
							const auto                   depth_ready = [&]() {
								return schedule_pos - next_out_.load(std::memory_order_relaxed) < depth_;
							};
							const auto budget_order_ready = [&]() {
								return max_inflight_storage_bytes_ == 0 || schedule_pos == next_budget_reserve_;
							};
							const auto byte_ready = [&]() {
								return max_inflight_storage_bytes_ == 0 || inflight_storage_bytes_ == 0 ||
								       (inflight_storage_bytes_ <= max_inflight_storage_bytes_ &&
								        storage_bytes <= max_inflight_storage_bytes_ - inflight_storage_bytes_);
							};
							blocked_by_bytes = depth_ready() && budget_order_ready() && !byte_ready();
							cv_not_full_.wait(lock, [&]() {
								return stop_ || (depth_ready() && budget_order_ready() && byte_ready());
							});
							if (stop_) {
								return;
							}
							inflight_storage_bytes_ += storage_bytes;
							if (max_inflight_storage_bytes_ != 0) {
								++next_budget_reserve_;
								budget_order_advanced = true;
							}
						}
						if (budget_order_advanced) {
							cv_not_full_.notify_all();
						}
						const auto depth_block_end = std::chrono::steady_clock::now();

						RowgroupReadResult                    prefetched {};
						const auto                            read_start = std::chrono::steady_clock::now();
						galp::format::ZeroCopyRowgroup        zero_copy {};
						galp::format::ZeroCopyReadTiming      io_timing {};
						std::chrono::steady_clock::time_point file_read_start {};
						prefetched.rowgroup_index      = rg_idx;
						prefetched.full_storage_bytes = storage_bytes;
						prefetched.prefetch.depth_block_ms =
						    std::chrono::duration<double, std::milli>(depth_block_end - depth_block_start).count();
						if (blocked_by_bytes) {
							prefetched.prefetch.byte_block_ms = prefetched.prefetch.depth_block_ms;
						}
						prefetched.prefetch.worker_id = w;
						const auto* selected = selected_vectors_.empty() || selected_vectors_[schedule_pos].empty()
						                           ? nullptr
						                           : &selected_vectors_[schedule_pos];
						const bool use_packed_device_scatter =
						    selected != nullptr && std::getenv("GALP_VECTOR_BUNDLE_DEVICE_SCATTER") != nullptr;
						if (use_packed_device_scatter) {
							file_read_start = std::chrono::steady_clock::now();
							zero_copy = rdr.read_rowgroup_zero_copy_selected_vectors_packed(
							    rg_idx, *selected, &io_timing);
						} else if (pinned_pool) {
							const auto                             acquire_start = std::chrono::steady_clock::now();
							PinnedRowgroupBufferPool::AcquireStats acquire_stats {};
							auto                                   lease = pinned_pool->acquire_for_owner_cancelable(
                                w, storage_bytes, &acquire_stats, &stop_requested_);
							const auto acquire_end = std::chrono::steady_clock::now();
							prefetched.timing.pinned_acquire_ms =
							    std::chrono::duration<double, std::milli>(acquire_end - acquire_start).count();
							prefetched.prefetch.pool_slot_owner_reused   = acquire_stats.owner_reused;
							prefetched.prefetch.pool_slot_owner_migrated = acquire_stats.owner_migrated;
							prefetched.prefetch.pool_slot_allocated      = acquire_stats.allocated;
							file_read_start                              = acquire_end;
							zero_copy = selected != nullptr
							                ? rdr.read_rowgroup_zero_copy_selected_vectors_into(rg_idx,
							                                                                    *selected,
							                                                                    std::move(lease.owner),
							                                                                    lease.data,
							                                                                    lease.capacity,
							                                                                    /*backing_is_pinned=*/true,
							                                                                    &io_timing)
							                : rdr.read_rowgroup_zero_copy_into(rg_idx,
							                                                     std::move(lease.owner),
							                                                     lease.data,
							                                                     lease.capacity,
							                                                     /*backing_is_pinned=*/true,
							                                                     &io_timing);
						} else {
							file_read_start = std::chrono::steady_clock::now();
							zero_copy = selected != nullptr
							                ? rdr.read_rowgroup_zero_copy_selected_vectors(rg_idx, *selected, &io_timing)
							                : rdr.read_rowgroup_zero_copy(rg_idx, &io_timing);
						}

						const auto file_read_end = std::chrono::steady_clock::now();
						const auto build_start   = std::chrono::steady_clock::now();
						prefetched.rowgroup      = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
						const auto build_end     = std::chrono::steady_clock::now();
						prefetched.timing.file_read_ms =
						    std::chrono::duration<double, std::milli>(file_read_end - file_read_start).count();
						prefetched.timing.rowgroup_build_ms =
						    std::chrono::duration<double, std::milli>(build_end - build_start).count();
						prefetched.timing.pread_ms                = io_timing.pread_ms;
						prefetched.storage_bytes                  = io_timing.storage_bytes;
						prefetched.full_storage_bytes             = io_timing.full_storage_bytes;
						prefetched.pread_count                    = io_timing.pread_count;
						prefetched.sparse_read_supported          = io_timing.sparse_read_supported;
						prefetched.used_sparse_read               = io_timing.used_sparse_read;
						prefetched.used_pinned_backing             = io_timing.used_pinned_backing;
						prefetched.used_vector_bundle_read        = io_timing.used_vector_bundle_read;
						prefetched.used_vector_bundle_envelope_read = io_timing.used_vector_bundle_envelope_read;
						prefetched.sparse_fallback_reason         = io_timing.sparse_fallback_reason;
						prefetched.timing.zero_copy_view_setup_ms = io_timing.zero_copy_view_setup_ms;
						prefetched.timing.timeline.pread_start    = io_timing.pread_start;
						prefetched.timing.timeline.pread_end      = io_timing.pread_end;
						prefetched.timing.timeline.read_submit =
						    (io_timing.pread_start == std::chrono::steady_clock::time_point {}) ? file_read_start
						                                                                        : io_timing.pread_start;
						prefetched.timing.read_ms =
						    std::chrono::duration<double, std::milli>(build_end - read_start).count();
						prefetched.timing.timeline.read_start      = read_start;
						prefetched.timing.timeline.file_read_start = file_read_start;
						prefetched.timing.timeline.file_read_end   = file_read_end;
						prefetched.timing.timeline.build_start     = build_start;
						prefetched.timing.timeline.build_end       = build_end;
						prefetched.timing.timeline.read_end        = build_end;

						{
							std::lock_guard<std::mutex> lock(mutex_);
							if (stop_) {
								return;
							}
							prefetched.timing.timeline.ready_push = std::chrono::steady_clock::now();
							slots_[schedule_pos % depth_]         = std::move(prefetched);
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
		stop_requested_.store(true, std::memory_order_release);
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stop_ = true;
		}
		cv_not_full_.notify_all();
		cv_not_empty_.notify_all();
		if (pinned_pool_) {
			pinned_pool_->notify_waiters();
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
		RowgroupReadResult item {};
		bool               underflow = false;
		const size_t       slot      = next_out_.load(std::memory_order_relaxed) % depth_;
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cv_not_empty_.wait(lock, [&]() { return stop_ || error_ != nullptr || slots_[slot].has_value() || done_; });
			const auto wait_end = std::chrono::steady_clock::now();
			wait_ms_ += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();

			if (error_ != nullptr) {
				pending_error = error_;
			} else if (!slots_[slot].has_value()) {
				underflow = true;
			} else {
				item                                     = std::move(*slots_[slot]);
				item.timing.timeline.consumer_wait_start = wait_start;
				item.timing.timeline.consumer_wait_end   = wait_end;
				item.timing.timeline.consumer_pop        = wait_end;
				slots_[slot].reset();
				--ready_count_;
				const size_t reserved_storage_bytes =
				    item.full_storage_bytes != 0 ? item.full_storage_bytes : item.storage_bytes;
				inflight_storage_bytes_ = reserved_storage_bytes < inflight_storage_bytes_
				                              ? inflight_storage_bytes_ - reserved_storage_bytes
				                              : 0;
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
		return item;
	}

	double wait_ms() const {
		return wait_ms_;
	}

private:
	const std::vector<size_t>                      rowgroups_;
	const std::vector<std::vector<uint32_t>>       selected_vectors_;
	const size_t                                   depth_                      = 1;
	const size_t                                   max_inflight_storage_bytes_ = 0;
	std::atomic<size_t>                            next_claim_;
	std::atomic<size_t>                            next_out_;
	size_t                                         next_budget_reserve_ = 0;
	std::vector<std::thread>                       workers_;
	size_t                                         worker_count_ = 0;
	mutable std::mutex                             mutex_;
	std::condition_variable                        cv_not_empty_;
	std::condition_variable                        cv_not_full_;
	std::vector<std::optional<RowgroupReadResult>> slots_;
	size_t                                         inflight_storage_bytes_ = 0;
	size_t                                         ready_count_            = 0;
	size_t                                         workers_finished_       = 0;
	std::exception_ptr                             error_;
	std::atomic<bool>                              stop_requested_ {false};
	bool                                           done_    = false;
	bool                                           stop_    = false;
	double                                         wait_ms_ = 0.0;
	std::shared_ptr<PinnedRowgroupBufferPool>      pinned_pool_;
	std::shared_ptr<galp::format::FlsReader>       shared_reader_;
};

} // namespace galp::runtime

#endif // ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_QUEUE_CUH
