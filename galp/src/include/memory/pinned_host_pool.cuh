// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/memory/pinned_host_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_PINNED_HOST_POOL_CUH
#define GALP_MEMORY_PINNED_HOST_POOL_CUH

#include "memory/cuda_macros.cuh"

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace galp::memory {

// Manages pinned host staging buffers used to speed up H2D copies. Owned by
// DevicePool but kept as a separate class so pinned lifecycle does not get
// tangled with device allocation, free-list reuse, and transfer tracking.
class PinnedHostPool {
public:
	void* alloc(size_t bytes) {
		if (bytes == 0) {
			return nullptr;
		}
		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (use_pinned_) {
				// Best-fit: reuse the smallest free buffer >= bytes. Exact-size
				// matching forced fresh cudaMallocHost when staged_bytes variance
				// pushed a chunk into a nearby size bucket.
				for (auto it = free_by_size_.lower_bound(bytes); it != free_by_size_.end(); ++it) {
					if (!it->second.empty()) {
						const size_t bucket_size = it->first;
						void*        ptr         = it->second.back();
						it->second.pop_back();
						in_use_[ptr] = bucket_size;
						return ptr;
					}
				}
			}
		}

		void* ptr = nullptr;
		CUDA_SAFE_CALL(cudaMallocHost(&ptr, bytes));
		{
			std::lock_guard<std::mutex> lock(mutex_);
			in_use_[ptr] = bytes;
		}
		return ptr;
	}

	void release(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = in_use_.find(ptr);
		if (it != in_use_.end()) {
			const size_t bytes = it->second;
			in_use_.erase(it);
			if (use_pinned_) {
				free_by_size_[bytes].push_back(ptr);
			} else {
				CUDA_SAFE_CALL(cudaFreeHost(ptr));
			}
		} else {
			CUDA_SAFE_CALL(cudaFreeHost(ptr));
		}
	}

	// Caller is responsible for draining in-flight transfers and asserting
	// idleness before flipping this; we don't re-check here.
	void set_use_pinned(bool use_pinned) {
		std::lock_guard<std::mutex> lock(mutex_);
		use_pinned_ = use_pinned;
	}

	bool has_in_use() {
		std::lock_guard<std::mutex> lock(mutex_);
		return !in_use_.empty();
	}

	~PinnedHostPool() {
		if (!in_use_.empty()) {
			std::fprintf(stderr,
			             "PinnedHostPool warning: %zu in-use pinned allocations at shutdown; forcing free.\n",
			             in_use_.size());
		}
		for (auto& [ptr, bytes] : in_use_) {
			(void)bytes;
			cudaFreeHost(ptr);
		}
		in_use_.clear();

		for (auto& [size, list] : free_by_size_) {
			(void)size;
			for (void* ptr : list) {
				cudaFreeHost(ptr);
			}
		}
		free_by_size_.clear();
	}

private:
	std::mutex                           mutex_;
	bool                                 use_pinned_ = true;
	std::map<size_t, std::vector<void*>> free_by_size_;
	std::unordered_map<void*, size_t>    in_use_;
};

} // namespace galp::memory

#endif // GALP_MEMORY_PINNED_HOST_POOL_CUH
