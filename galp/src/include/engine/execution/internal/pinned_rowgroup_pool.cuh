// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/pinned_rowgroup_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_PINNED_ROWGROUP_POOL_CUH
#define ENGINE_EXECUTION_INTERNAL_PINNED_ROWGROUP_POOL_CUH

#include "flsgpu/memory/device_pool.cuh"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace dispatch::runtime {

// Slot-based pool of CUDA pinned host buffers used to stage rowgroup reads.
// Leases reference-count the slot via a shared_ptr; the weak_from_this deleter
// guards against a lease outliving the pool during abnormal teardown.
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
		try {
			for (auto& slot : slots_) {
				if (slot.ptr != nullptr) {
					flsgpu::memory::DevicePool::instance().release_pinned(slot.ptr);
				}
			}
		} catch (const std::exception& e) {
			std::fprintf(stderr, "PinnedRowgroupBufferPool destructor: %s\n", e.what());
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

	size_t prewarm(const size_t min_bytes, const size_t requested_slots) {
		if (min_bytes == 0 || requested_slots == 0) {
			return 0;
		}
		std::unique_lock<std::mutex> lock(mutex_);
		if (stopping_) {
			throw std::runtime_error("pinned rowgroup buffer pool stopped");
		}

		const size_t alloc_bytes = round_up_capacity(min_bytes);
		const size_t byte_budget = prewarm_byte_budget();
		if (byte_budget == 0) {
			return 0;
		}
		const size_t max_slots = std::min({slots_.size(), requested_slots, byte_budget / alloc_bytes});
		if (max_slots == 0) {
			return 0;
		}

		size_t warmed = 0;
		for (auto& slot : slots_) {
			if (warmed >= max_slots) {
				break;
			}
			if (slot.in_use) {
				continue;
			}
			if (slot.capacity < min_bytes) {
				if (slot.ptr != nullptr) {
					flsgpu::memory::DevicePool::instance().release_pinned(slot.ptr);
				}
				slot.ptr      = flsgpu::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
				slot.capacity = alloc_bytes;
			}
			++warmed;
		}
		return warmed;
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

	static size_t prewarm_byte_budget() {
		const char* env = std::getenv("GALP_PINNED_ROWGROUP_PREWARM_BYTES");
		if (env == nullptr || *env == '\0') {
			return 512ULL * 1024ULL * 1024ULL;
		}
		char*                    end   = nullptr;
		const unsigned long long value = std::strtoull(env, &end, 10);
		if (end == env) {
			return 512ULL * 1024ULL * 1024ULL;
		}
		if (value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
			return std::numeric_limits<size_t>::max();
		}
		return static_cast<size_t>(value);
	}

	void release(const size_t idx) {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			slots_[idx].in_use = false;
		}
		cv_.notify_one();
	}

	std::mutex              mutex_;
	std::condition_variable cv_;
	std::vector<Slot>       slots_;
	bool                    stopping_ = false;
};

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_PINNED_ROWGROUP_POOL_CUH
