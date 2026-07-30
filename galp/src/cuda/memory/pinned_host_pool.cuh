// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/pinned_host_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_PINNED_HOST_POOL_CUH
#define GALP_MEMORY_PINNED_HOST_POOL_CUH

#include "cuda/cuda_macros.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iterator>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace galp::memory {

struct PinnedHostPoolStats {
	size_t in_use_bytes          = 0;
	size_t peak_in_use_bytes     = 0;
	size_t cached_bytes          = 0;
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
	    , max_reuse_slack_bytes_(max_reuse_slack_bytes) {}

	void* alloc(size_t bytes) {
		if (bytes == 0) {
			return nullptr;
		}
		{
			std::lock_guard<std::mutex> lock(mutex_);
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

		void* ptr = nullptr;
		auto  status = cudaMallocHost(&ptr, bytes);
		if (status != cudaSuccess) {
			ptr = nullptr;
			release_cached();
			status = cudaMallocHost(&ptr, bytes);
		}
		CUDA_SAFE_CALL(status);
		{
			std::lock_guard<std::mutex> lock(mutex_);
			in_use_[ptr] = bytes;
			record_in_use_allocation_locked(bytes, true);
		}
		return ptr;
	}

	void release(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::vector<void*> evicted;
		bool               free_released = true;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			auto                        it = in_use_.find(ptr);
			if (it != in_use_.end()) {
				const size_t bytes = it->second;
				in_use_.erase(it);
				in_use_bytes_ = bytes <= in_use_bytes_ ? in_use_bytes_ - bytes : 0;
				if (use_pinned_ && bytes <= cache_limit_bytes_) {
					evict_until_room_locked(bytes, evicted);
					if (cached_bytes_ + bytes <= cache_limit_bytes_) {
						free_by_size_[bytes].push_back(ptr);
						cached_bytes_ += bytes;
						free_released = false;
					}
				}
			}
		}
		for (void* evicted_ptr : evicted) {
			CUDA_SAFE_CALL(cudaFreeHost(evicted_ptr));
		}
		if (free_released) {
			CUDA_SAFE_CALL(cudaFreeHost(ptr));
		}
	}

	// Caller is responsible for draining in-flight transfers and asserting
	// idleness before flipping this; we don't re-check here.
	void set_use_pinned(bool use_pinned) {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			use_pinned_ = use_pinned;
		}
		if (!use_pinned) {
			release_cached();
		}
	}

	void set_cache_limit_bytes(size_t bytes) {
		std::vector<void*> evicted;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			cache_limit_bytes_ = bytes;
			evict_until_limit_locked(evicted);
		}
		for (void* ptr : evicted) {
			CUDA_SAFE_CALL(cudaFreeHost(ptr));
		}
	}

	void release_cached() {
		std::map<size_t, std::vector<void*>> cached;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			cached.swap(free_by_size_);
			cached_bytes_ = 0;
		}
		for (auto& [size, list] : cached) {
			(void)size;
			for (void* ptr : list) {
				CUDA_SAFE_CALL(cudaFreeHost(ptr));
			}
		}
	}

	bool has_in_use() {
		std::lock_guard<std::mutex> lock(mutex_);
		return !in_use_.empty();
	}

	PinnedHostPoolStats stats() {
		std::lock_guard<std::mutex> lock(mutex_);
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
		std::lock_guard<std::mutex> lock(mutex_);
		cleanup_enabled_ = false;
		in_use_.clear();
		free_by_size_.clear();
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

	void evict_until_room_locked(size_t required_bytes, std::vector<void*>& evicted) {
		while (cached_bytes_ + required_bytes > cache_limit_bytes_ && !free_by_size_.empty()) {
			evict_largest_locked(evicted);
		}
	}

	void evict_until_limit_locked(std::vector<void*>& evicted) {
		while (cached_bytes_ > cache_limit_bytes_ && !free_by_size_.empty()) {
			evict_largest_locked(evicted);
		}
	}

	void evict_largest_locked(std::vector<void*>& evicted) {
		auto  it   = std::prev(free_by_size_.end());
		auto& list = it->second;
		evicted.push_back(list.back());
		list.pop_back();
		cached_bytes_ -= it->first;
		if (list.empty()) {
			free_by_size_.erase(it);
		}
	}

	std::mutex                           mutex_;
	bool                                 use_pinned_            = true;
	bool                                 cleanup_enabled_       = true;
	size_t                               cache_limit_bytes_;
	size_t                               max_reuse_slack_bytes_;
	size_t                               in_use_bytes_          = 0;
	size_t                               peak_in_use_bytes_     = 0;
	size_t                               cached_bytes_          = 0;
	size_t                               allocation_requests_   = 0;
	size_t                               cuda_allocation_count_ = 0;
	size_t                               cuda_allocation_bytes_ = 0;
	std::map<size_t, std::vector<void*>> free_by_size_;
	std::unordered_map<void*, size_t>    in_use_;
};

} // namespace galp::memory

#endif // GALP_MEMORY_PINNED_HOST_POOL_CUH
