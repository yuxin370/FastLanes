// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/rowgroup_prefetch_queue.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_QUEUE_CUH
#define ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_QUEUE_CUH

#include "engine/execution/internal/pinned_rowgroup_pool.cuh"
#include "engine/reader.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace dispatch::runtime {

struct RowgroupReadResult {
	reader::Rowgroup rowgroup {};
	double           read_ms           = 0.0;
	double           file_read_ms      = 0.0;
	double           rowgroup_build_ms = 0.0;
};

// Bounded ring-buffer prefetcher: a pool of worker threads reads rowgroups
// ahead of the consumer; slots_[(idx - start) % depth] provides back-pressure
// so the producer never runs more than `depth` rowgroups ahead.
class RowgroupPrefetchQueue {
public:
	RowgroupPrefetchQueue(std::shared_ptr<reader::reader>           shared_reader,
	                      const size_t                              start,
	                      const size_t                              end,
	                      const size_t                              depth,
	                      const size_t                              num_workers,
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
	    , pinned_pool_(pinned_pool)
	    , shared_reader_(std::move(shared_reader)) {
		if (!shared_reader_) {
			throw std::runtime_error("RowgroupPrefetchQueue: shared reader is null");
		}
		worker_count_ = std::max<size_t>(1, std::min<size_t>(num_workers, end > start ? end - start : 1));
		workers_.reserve(worker_count_);
		// Pre-touch the file fd & TableDescriptor on the calling thread so each
		// worker sees a fully-initialised reader on its first read; pread() is
		// thread-safe over a shared fd and the descriptor view is read-only,
		// so per-worker reader construction (one fd + one FlatBuffers parse
		// each) is no longer needed.
		(void)shared_reader_->rowgroup_count();
		for (size_t w = 0; w < worker_count_; ++w) {
			workers_.emplace_back([this, pinned_pool]() {
				try {
					reader::reader& rdr = *shared_reader_;
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

						Prefetched               prefetched {};
						const auto               file_read_start = std::chrono::steady_clock::now();
						reader::ZeroCopyRowgroup zero_copy {};
						if (pinned_pool) {
							auto lease = pinned_pool->acquire(rdr.rowgroup_storage_bytes(rg_idx));
							zero_copy  = rdr.read_rowgroup_zero_copy_into(rg_idx,
                                                                         std::move(lease.owner),
                                                                         lease.data,
                                                                         lease.capacity,
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
			cv_not_empty_.wait(lock, [&]() { return stop_ || error_ != nullptr || slots_[slot].has_value() || done_; });
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
		reader::Rowgroup rowgroup {};
		double           read_ms           = 0.0;
		double           file_read_ms      = 0.0;
		double           rowgroup_build_ms = 0.0;
	};

	const size_t                              start_ = 0;
	const size_t                              end_   = 0;
	const size_t                              depth_ = 1;
	std::atomic<size_t>                       next_claim_;
	std::atomic<size_t>                       next_out_;
	std::vector<std::thread>                  workers_;
	size_t                                    worker_count_ = 0;
	mutable std::mutex                        mutex_;
	std::condition_variable                   cv_not_empty_;
	std::condition_variable                   cv_not_full_;
	std::vector<std::optional<Prefetched>>    slots_;
	size_t                                    ready_count_      = 0;
	size_t                                    workers_finished_ = 0;
	std::exception_ptr                        error_;
	bool                                      done_    = false;
	bool                                      stop_    = false;
	double                                    wait_ms_ = 0.0;
	std::shared_ptr<PinnedRowgroupBufferPool> pinned_pool_;
	std::shared_ptr<reader::reader>           shared_reader_;
};

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_QUEUE_CUH
