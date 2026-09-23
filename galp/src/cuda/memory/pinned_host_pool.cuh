// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/pinned_host_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_PINNED_HOST_POOL_CUH
#define GALP_MEMORY_PINNED_HOST_POOL_CUH

#include "cuda/cuda_macros.cuh"
#include "cuda/memory/memory_diagnostics.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace galp::memory {

struct PinnedHostPoolStats {
	size_t in_use_bytes          = 0;
	size_t peak_in_use_bytes     = 0;
	size_t cached_bytes          = 0; // includes non-reusable pending frees
	size_t allocation_requests   = 0;
	size_t cuda_allocation_count = 0;
	size_t cuda_allocation_bytes = 0;
};

// Manages pinned host staging buffers used to speed up H2D copies. Owned by
// DevicePool but kept as a separate class so pinned lifecycle does not get
// tangled with device allocation, free-list reuse, and transfer tracking.
class PinnedHostPool {
public:
	explicit PinnedHostPool(size_t cache_limit_bytes     = 256ULL * 1024ULL * 1024ULL,
	                        size_t max_reuse_slack_bytes = 64ULL * 1024ULL * 1024ULL)
	    : cache_limit_bytes_(cache_limit_bytes)
	    , max_reuse_slack_bytes_(max_reuse_slack_bytes) {
	}

	void* alloc(size_t bytes) {
		if (bytes == 0) {
			return nullptr;
		}
		{
			std::lock_guard lock(mutex_);
			++allocation_requests_;
			if (use_pinned_) {
				// Best-fit: reuse the smallest free buffer >= bytes. Exact-size
				// matching forced fresh cudaMallocHost when staged_bytes variance
				// pushed a chunk into a nearby size bucket.
				auto it = free_by_size_.lower_bound(bytes);
				if (it != free_by_size_.end() &&
				    (max_reuse_slack_bytes_ == 0 || it->first - bytes <= max_reuse_slack_bytes_)) {
					const size_t bucket_size = it->first;
					void*        ptr         = it->second.back();
					it->second.pop_back();
					if (it->second.empty()) {
						free_by_size_.erase(it);
					}
					cached_bytes_ -= bucket_size;
					in_use_[ptr] = bucket_size;
					record_in_use_allocation_locked(bucket_size, false);
					return ptr;
				}
			}
		}

		void* ptr    = nullptr;
		auto  status = cudaMallocHost(&ptr, bytes);
		if (status != cudaSuccess) {
			ptr = nullptr;
			release_cached();
			status = cudaMallocHost(&ptr, bytes);
		}
		CUDA_SAFE_CALL(status);
		{
			std::lock_guard lock(mutex_);
			in_use_[ptr] = bytes;
			record_in_use_allocation_locked(bytes, true);
		}
		return ptr;
	}

	void release(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::unique_lock lock(mutex_);
		const auto       it = in_use_.find(ptr);
		if (it == in_use_.end()) {
			throw std::invalid_argument("PinnedHostPool::release requires a live pool allocation");
		}
		const size_t bytes = it->second;
		// Until every throwing step succeeds, the caller (including the tracker)
		// retains its in-use allocation. Never publish it to the cache then throw.
		if (use_pinned_ && bytes <= cache_limit_bytes_) {
			evict_until_room_locked(bytes, lock);
			auto [bucket, inserted] = free_by_size_.try_emplace(bytes);
			try {
				bucket->second.push_back(ptr);
			} catch (...) {
				if (inserted) free_by_size_.erase(bucket);
				throw; // keep ptr in-use, and do not leave an empty reusable bucket
			}
			cached_bytes_ += bytes;
		} else {
			lock.unlock();
			CUDA_SAFE_CALL(cudaFreeHost(ptr));
			lock.lock();
		}
		in_use_.erase(ptr); // iterators can be invalidated while mutex_ is unlocked
		in_use_bytes_ -= bytes;
	}

	// Caller is responsible for draining in-flight transfers and asserting
	// idleness before flipping this; we don't re-check here.
	void set_use_pinned(bool use_pinned) {
		{
			std::lock_guard lock(mutex_);
			use_pinned_ = use_pinned;
		}
		if (!use_pinned) {
			release_cached();
		}
	}

	void set_cache_limit_bytes(size_t bytes) {
		std::unique_lock lock(mutex_);
		cache_limit_bytes_ = bytes;
		evict_until_room_locked(0, lock);
	}

	void release_cached() {
		std::lock_guard  cleanup(cleanup_mutex_);
		std::unique_lock lock(mutex_);
		while (pending_free_ || !free_by_size_.empty()) {
			free_one_cached_locked(lock, false);
		}
	}

	bool has_in_use() {
		std::lock_guard lock(mutex_);
		return !in_use_.empty();
	}

	PinnedHostPoolStats stats() {
		std::lock_guard lock(mutex_);
		return PinnedHostPoolStats {in_use_bytes_,
		                            peak_in_use_bytes_,
		                            cached_bytes_,
		                            allocation_requests_,
		                            cuda_allocation_count_,
		                            cuda_allocation_bytes_};
	}

	// Process-exit escape hatch for DevicePool when the CUDA runtime has already
	// torn down. The driver will reclaim these allocations with the context;
	// calling cudaFreeHost at that point is unsafe on some runtime versions.
	void abandon_without_free() noexcept {
		std::lock_guard lock(mutex_);
		cleanup_enabled_ = false;
		in_use_.clear();
		free_by_size_.clear();
		pending_free_.reset();
		in_use_bytes_ = 0;
		cached_bytes_ = 0;
	}

	~PinnedHostPool() {
		if (!cleanup_enabled_) {
			return;
		}
		if (!in_use_.empty()) {
			std::fprintf(stderr,
			             "PinnedHostPool warning: %zu in-use pinned allocations at shutdown; forcing free.\n",
			             in_use_.size());
		}
		for (auto& [ptr, bytes] : in_use_) {
			(void)bytes;
			CUDA_LOG_CALL(cudaFreeHost(ptr));
		}
		in_use_.clear();

		for (auto& [size, list] : free_by_size_) {
			(void)size;
			for (void* ptr : list) {
				CUDA_LOG_CALL(cudaFreeHost(ptr));
			}
		}
		free_by_size_.clear();
		if (pending_free_) {
			CUDA_LOG_CALL(cudaFreeHost(pending_free_->ptr));
		}
	}

private:
	void record_in_use_allocation_locked(size_t bytes, bool cuda_allocation) {
		in_use_bytes_ += bytes;
		peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
		if (cuda_allocation) {
			++cuda_allocation_count_;
			cuda_allocation_bytes_ += bytes;
		}
	}

	using Lock = std::unique_lock<diagnostics::Mutex<diagnostics::Pinned>>;

	void evict_until_room_locked(size_t required_bytes, Lock& lock) {
		if (cached_bytes_ + required_bytes <= cache_limit_bytes_)
			return;
		lock.unlock();
		std::lock_guard cleanup(cleanup_mutex_);
		lock.lock();
		while (cached_bytes_ + required_bytes > cache_limit_bytes_ && (pending_free_ || !free_by_size_.empty())) {
			free_one_cached_locked(lock, true);
		}
	}

	// cleanup_mutex_ serializes only CUDA frees, not allocation/cache reuse.
	// The single pending slot is an allocation-free ownership transfer. It is
	// never reusable, survives exceptions, and remains included in cached_bytes_.
	void free_one_cached_locked(Lock& lock, bool largest) {
		if (!pending_free_) {
			auto it       = largest ? std::prev(free_by_size_.end()) : free_by_size_.begin();
			pending_free_ = PendingFree {it->second.back(), it->first};
			it->second.pop_back();
			if (it->second.empty())
				free_by_size_.erase(it);
		}
		const auto block = *pending_free_;
		lock.unlock();
		CUDA_SAFE_CALL(cudaFreeHost(block.ptr));
		lock.lock();
		cached_bytes_ -= block.bytes;
		pending_free_.reset();
	}

	struct PendingFree {
		void*  ptr;
		size_t bytes;
	};
	std::mutex                              cleanup_mutex_;
	std::optional<PendingFree>              pending_free_;
	diagnostics::Mutex<diagnostics::Pinned> mutex_;
	bool                                    use_pinned_      = true;
	bool                                    cleanup_enabled_ = true;
	size_t                                  cache_limit_bytes_;
	size_t                                  max_reuse_slack_bytes_;
	size_t                                  in_use_bytes_          = 0;
	size_t                                  peak_in_use_bytes_     = 0;
	size_t                                  cached_bytes_          = 0;
	size_t                                  allocation_requests_   = 0;
	size_t                                  cuda_allocation_count_ = 0;
	size_t                                  cuda_allocation_bytes_ = 0;
	std::map<size_t, std::vector<void*>>    free_by_size_;
	std::unordered_map<void*, size_t>       in_use_;
};

} // namespace galp::memory

#endif // GALP_MEMORY_PINNED_HOST_POOL_CUH
