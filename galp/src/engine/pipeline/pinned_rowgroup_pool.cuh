// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/pipeline/pinned_rowgroup_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_PINNED_ROWGROUP_POOL_CUH
#define ENGINE_EXECUTION_INTERNAL_PINNED_ROWGROUP_POOL_CUH

#include "cuda/memory/device_pool.cuh"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace galp::runtime {

// Slot-based pool of CUDA pinned host buffers used to stage rowgroup reads.
// Leases reference-count the slot via a shared_ptr; the weak_from_this deleter
// guards against a lease outliving the pool during abnormal teardown.
class PinnedRowgroupBufferPool : public std::enable_shared_from_this<PinnedRowgroupBufferPool> {
public:
	static constexpr size_t kNoOwner = std::numeric_limits<size_t>::max();

	struct Lease {
		std::shared_ptr<void> owner;
		std::byte*            data     = nullptr;
		size_t                capacity = 0;
	};

	struct AcquireStats {
		size_t slot_index     = kNoOwner;
		size_t previous_owner = kNoOwner;
		bool   owner_reused   = false;
		bool   owner_migrated = false;
		bool   allocated      = false;
	};
	struct PrewarmStats {
		size_t requested_slots = 0U;
		size_t requested_bytes = 0U;
		size_t warmed_slots    = 0U;
		size_t warmed_bytes    = 0U;
		size_t largest_class   = 0U;
		bool   complete        = false;
	};

	// Construct only via create() — enable_shared_from_this requires the
	// instance be owned by a shared_ptr before shared_from_this() is called.
	static std::shared_ptr<PinnedRowgroupBufferPool> create(const size_t slots) {
		return std::shared_ptr<PinnedRowgroupBufferPool>(new PinnedRowgroupBufferPool(slots));
	}

	~PinnedRowgroupBufferPool() {
		try {
			for (auto& slot : slots_) {
				if (slot.ptr != nullptr && !slot.from_slab) {
					galp::memory::DevicePool::instance().release_pinned(slot.ptr);
				}
			}
			if (slab_ptr_ != nullptr) {
				galp::memory::DevicePool::instance().release_pinned(slab_ptr_);
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

	void notify_waiters() {
		cv_.notify_all();
	}

	Lease acquire(const size_t min_bytes) {
		return acquire_for_owner(kNoOwner, min_bytes, nullptr);
	}

	Lease acquire_for_owner(const size_t owner, const size_t min_bytes, AcquireStats* stats = nullptr) {
		return acquire_for_owner_cancelable(owner, min_bytes, stats, nullptr);
	}

	Lease acquire_for_owner_cancelable(const size_t           owner,
	                                   const size_t           min_bytes,
	                                   AcquireStats*          stats,
	                                   const std::atomic<bool>* cancel_requested) {
		std::unique_lock<std::mutex> lock(mutex_);
		const auto cancelled = [&]() {
			return cancel_requested != nullptr && cancel_requested->load(std::memory_order_acquire);
		};
		cv_.wait(lock, [&]() {
			return stopping_ || cancelled() ||
			       std::any_of(slots_.begin(), slots_.end(), [](const Slot& slot) { return !slot.in_use; });
		});
		if (stopping_) {
			throw std::runtime_error("pinned rowgroup buffer pool stopped");
		}
		if (cancelled()) {
			throw std::runtime_error("pinned rowgroup buffer acquire cancelled");
		}

		const size_t idx = choose_slot(owner, min_bytes);
		if (idx != kNoOwner) {
			auto& slot = slots_[idx];
			if (stats != nullptr) {
				stats->slot_index     = idx;
				stats->previous_owner = slot.owner;
				stats->owner_reused   = owner != kNoOwner && slot.owner == owner;
				stats->owner_migrated = owner != kNoOwner && slot.owner != kNoOwner && slot.owner != owner;
				stats->allocated      = slot.capacity < min_bytes;
			}
			if (slot.capacity < min_bytes) {
				release_slot_allocation(slot);
				const size_t alloc_bytes = round_up_capacity(min_bytes);
				slot.ptr                 = galp::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
				slot.capacity            = alloc_bytes;
				slot.from_slab           = false;
				slot.slab_index          = kNoOwner;
			}
			slot.in_use = true;
			if (owner != kNoOwner) {
				slot.owner = owner;
			}
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

	size_t prewarm(const size_t min_bytes, const size_t requested_slots, const size_t owner_count = 0) {
		if (min_bytes == 0 || requested_slots == 0) {
			return 0;
		}
		std::unique_lock<std::mutex> lock(mutex_);
		if (stopping_) {
			throw std::runtime_error("pinned rowgroup buffer pool stopped");
		}

		const size_t alloc_bytes = round_up_capacity(min_bytes);
		const size_t byte_budget  = prewarm_byte_budget();
		if (byte_budget == 0) {
			return 0;
		}
		const size_t max_slots = std::min({slots_.size(), requested_slots, byte_budget / alloc_bytes});
		if (max_slots == 0) {
			return 0;
		}

		const bool slab_available = ensure_slab(alloc_bytes, max_slots);
		std::vector<bool> slab_used(slab_slot_count_, false);
		if (slab_available) {
			for (const auto& slot : slots_) {
				if (slot.from_slab && slot.slab_index < slab_used.size()) {
					slab_used[slot.slab_index] = true;
				}
			}
		}

		size_t warmed = 0;
		for (auto& slot : slots_) {
			if (warmed >= max_slots) {
				break;
			}
			if (slot.in_use) {
				continue;
			}

			bool assigned_from_slab = false;
			if (slab_available && !slot.from_slab) {
				const size_t slab_index = next_free_slab_index(slab_used);
				if (slab_index != kNoOwner) {
					release_slot_allocation(slot);
					slot.ptr       = static_cast<std::byte*>(slab_ptr_) + slab_index * slab_slot_capacity_;
					slot.capacity  = slab_slot_capacity_;
					slot.from_slab = true;
					slot.slab_index = slab_index;
					slab_used[slab_index] = true;
					assigned_from_slab = true;
				}
			}

			if (!assigned_from_slab && slot.capacity < min_bytes) {
				release_slot_allocation(slot);
				slot.ptr        = galp::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
				slot.capacity   = alloc_bytes;
				slot.from_slab  = false;
				slot.slab_index = kNoOwner;
			}
			if (owner_count != 0) {
				slot.owner = warmed % owner_count;
			}
			++warmed;
		}
		return warmed;
	}

	// Prewarm a reusable best-fit distribution from an observed batch.  The
	// requested classes repeat deterministically until requested_slots is
	// reached, and the global byte budget keeps startup memory bounded.
	PrewarmStats prewarm_size_classes(std::vector<size_t> classes, const size_t requested_slots) {
		classes.erase(std::remove(classes.begin(), classes.end(), 0U), classes.end());
		if (classes.empty() || requested_slots == 0U) {
			return {};
		}
		for (auto& bytes : classes) {
			bytes = round_up_capacity(bytes);
		}
		std::sort(classes.begin(), classes.end(), std::greater<size_t> {});
		std::unique_lock<std::mutex> lock(mutex_);
		if (stopping_) {
			throw std::runtime_error("pinned rowgroup buffer pool stopped");
		}
		const size_t target = std::min(requested_slots, slots_.size());
		const size_t budget = prewarm_byte_budget();
		PrewarmStats result;
		result.requested_slots = target;
		for (size_t index = 0U; index < target; ++index) {
			const size_t desired = classes[index % classes.size()];
			if (desired > std::numeric_limits<size_t>::max() - result.requested_bytes) {
				throw std::overflow_error("pinned rowgroup prewarm byte count overflow");
			}
			result.requested_bytes += desired;
			if (desired > budget - std::min(budget, result.warmed_bytes)) {
				break;
			}
			auto& slot = slots_[index];
			if (slot.in_use) {
				continue;
			}
			if (slot.capacity < desired) {
				release_slot_allocation(slot);
				slot.ptr        = galp::memory::DevicePool::instance().alloc_pinned(desired);
				slot.capacity   = desired;
				slot.from_slab  = false;
				slot.slab_index = kNoOwner;
			}
			slot.owner = kNoOwner;
			++result.warmed_slots;
			result.warmed_bytes += slot.capacity;
			result.largest_class = std::max(result.largest_class, slot.capacity);
		}
		result.complete = result.warmed_slots == result.requested_slots;
		return result;
	}

	// Install an exact, dataset-derived capacity profile. Unlike
	// prewarm_size_classes(), this does not repeat an observed batch pattern;
	// callers provide every simultaneously reusable slot required by their
	// formal concurrency contract.
	PrewarmStats prewarm_capacities(std::vector<size_t> capacities) {
		capacities.erase(std::remove(capacities.begin(), capacities.end(), 0U), capacities.end());
		for (auto& bytes : capacities) {
			bytes = round_up_capacity(bytes);
		}
		std::sort(capacities.begin(), capacities.end(), std::greater<size_t> {});
		std::unique_lock<std::mutex> lock(mutex_);
		if (stopping_) {
			throw std::runtime_error("pinned rowgroup buffer pool stopped");
		}

		PrewarmStats result;
		result.requested_slots = capacities.size();
		for (const auto desired : capacities) {
			if (desired > std::numeric_limits<size_t>::max() - result.requested_bytes) {
				throw std::overflow_error("pinned rowgroup capacity contract overflow");
			}
			result.requested_bytes += desired;
		}
		const size_t target = std::min(capacities.size(), slots_.size());
		const size_t budget = prewarm_byte_budget();
		for (size_t index = 0U; index < target; ++index) {
			const size_t desired = capacities[index];
			if (desired > budget - std::min(budget, result.warmed_bytes)) {
				break;
			}
			auto& slot = slots_[index];
			if (slot.in_use) {
				break;
			}
			if (slot.capacity < desired) {
				release_slot_allocation(slot);
				slot.ptr        = galp::memory::DevicePool::instance().alloc_pinned(desired);
				slot.capacity   = desired;
				slot.from_slab  = false;
				slot.slab_index = kNoOwner;
			}
			slot.owner = kNoOwner;
			++result.warmed_slots;
			result.warmed_bytes += slot.capacity;
			result.largest_class = std::max(result.largest_class, slot.capacity);
		}
		result.complete = result.requested_slots <= slots_.size() &&
		                  result.warmed_slots == result.requested_slots &&
		                  result.requested_bytes <= budget;
		return result;
	}

private:
	explicit PinnedRowgroupBufferPool(const size_t slots) : slots_(std::max<size_t>(1, slots)) {}

	struct Slot {
		void*  ptr       = nullptr;
		size_t capacity   = 0;
		bool   in_use    = false;
		size_t owner     = kNoOwner;
		bool   from_slab = false;
		size_t slab_index = kNoOwner;
	};

	void release_slot_allocation(Slot& slot) {
		if (slot.ptr != nullptr && !slot.from_slab) {
			galp::memory::DevicePool::instance().release_pinned(slot.ptr);
		}
		slot.ptr        = nullptr;
		slot.capacity   = 0;
		slot.from_slab  = false;
		slot.slab_index = kNoOwner;
	}

	bool ensure_slab(const size_t slot_bytes, const size_t slot_count) {
		if (slot_bytes == 0 || slot_count == 0) {
			return false;
		}
		if (slab_ptr_ != nullptr) {
			return slab_slot_capacity_ >= slot_bytes && slab_slot_count_ >= slot_count;
		}
		if (slot_count > std::numeric_limits<size_t>::max() / slot_bytes) {
			return false;
		}
		const size_t slab_bytes = slot_bytes * slot_count;
		slab_ptr_               = galp::memory::DevicePool::instance().alloc_pinned(slab_bytes);
		slab_slot_capacity_     = slot_bytes;
		slab_slot_count_        = slot_count;
		slab_bytes_             = slab_bytes;
		return true;
	}

	static size_t next_free_slab_index(const std::vector<bool>& used) {
		for (size_t idx = 0; idx < used.size(); ++idx) {
			if (!used[idx]) {
				return idx;
			}
		}
		return kNoOwner;
	}

	size_t choose_slot(const size_t owner, const size_t min_bytes) const {
		const auto find_slot = [&](const auto& predicate) {
			for (size_t idx = 0; idx < slots_.size(); ++idx) {
				if (!slots_[idx].in_use && predicate(slots_[idx])) {
					return idx;
				}
			}
			return kNoOwner;
		};
		const auto find_best_fit = [&](const auto& predicate) {
			size_t best = kNoOwner;
			for (size_t idx = 0U; idx < slots_.size(); ++idx) {
				const auto& slot = slots_[idx];
				if (slot.in_use || slot.capacity < min_bytes || !predicate(slot)) {
					continue;
				}
				if (best == kNoOwner || slot.capacity < slots_[best].capacity) {
					best = idx;
				}
			}
			return best;
		};

		if (owner != kNoOwner) {
			size_t idx = find_best_fit([&](const Slot& slot) { return slot.owner == owner; });
			if (idx != kNoOwner) {
				return idx;
			}
			idx = find_best_fit([](const Slot& slot) { return slot.owner == kNoOwner; });
			if (idx != kNoOwner) {
				return idx;
			}
		} else {
			const size_t idx = find_best_fit([](const Slot& slot) { return slot.owner == kNoOwner; });
			if (idx != kNoOwner) {
				return idx;
			}
		}
		if (const auto idx = find_best_fit([](const Slot&) { return true; }); idx != kNoOwner) {
			return idx;
		}

		// No existing size class fits. Prefer growing an owner-affine/unowned
		// slot, preserving deterministic ownership without sacrificing reuse.
		if (owner != kNoOwner) {
			if (const auto idx = find_slot([&](const Slot& slot) { return slot.owner == owner; }); idx != kNoOwner) {
				return idx;
			}
		}
		if (const auto idx = find_slot([](const Slot& slot) { return slot.owner == kNoOwner; }); idx != kNoOwner) {
			return idx;
		}

		return find_slot([](const Slot&) { return true; });
	}

	static size_t round_up_capacity(const size_t bytes) {
		constexpr size_t kMinimum = 64U * 1024U;
		size_t           capacity = kMinimum;
		while (capacity < bytes) {
			if (capacity > std::numeric_limits<size_t>::max() / 2U) {
				throw std::overflow_error("pinned rowgroup buffer capacity overflow");
			}
			capacity *= 2U;
		}
		return capacity;
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
	void*                   slab_ptr_           = nullptr;
	size_t                  slab_bytes_         = 0;
	size_t                  slab_slot_capacity_ = 0;
	size_t                  slab_slot_count_    = 0;
	bool                    stopping_ = false;
};

} // namespace galp::runtime

#endif // ENGINE_EXECUTION_INTERNAL_PINNED_ROWGROUP_POOL_CUH
