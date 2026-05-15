// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/memory/device_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_DEVICE_POOL_CUH
#define GALP_MEMORY_DEVICE_POOL_CUH

#include "memory/cuda_macros.cuh"
#include "memory/pinned_host_pool.cuh"
#include "memory/transfer_tracker.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace galp::memory {

struct DeviceAllocInfo {
	size_t       size         = 0;
	bool         async_alloc  = false;
	cudaStream_t alloc_stream = nullptr;
	bool         sub_alloc    = false; // true for arena sub-pointers (no-op on free)
};

class DevicePool {
public:
	static DevicePool& instance() {
		static DevicePool pool;
		return pool;
	}

	void* alloc(size_t bytes) {
		return alloc_on_stream(bytes, nullptr);
	}

	void* alloc_on_stream(size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return nullptr;
		}
		bool use_async = false;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			use_async = use_async_;
			if (enabled_) {
				if (use_async_) {
					auto& free_list = free_async_by_stream_size_[stream_key(stream)][bytes];
					if (!free_list.empty()) {
						void* ptr = free_list.back();
						free_list.pop_back();
						in_use_[ptr] = DeviceAllocInfo {bytes, true, stream};
						return ptr;
					}
				} else {
					auto& free_list = free_sync_by_size_[bytes];
					if (!free_list.empty()) {
						void* ptr = free_list.back();
						free_list.pop_back();
						in_use_[ptr] = DeviceAllocInfo {bytes, false, nullptr};
						return ptr;
					}
				}
			}
		}

		void* ptr         = nullptr;
		bool  async_alloc = false;
		if (use_async) {
			auto status = cudaMallocAsync(&ptr, bytes, stream);
			if (status == cudaSuccess) {
				async_alloc = true;
			} else {
				ptr = nullptr;
			}
		}
		if (!ptr) {
			CUDA_SAFE_CALL(cudaMalloc(&ptr, bytes));
			async_alloc = false;
		}
		{
			std::lock_guard<std::mutex> lock(mutex_);
			in_use_[ptr] = DeviceAllocInfo {bytes, async_alloc, async_alloc ? stream : nullptr};
		}
		return ptr;
	}

	void register_sub_allocation(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = in_use_.find(ptr);
		if (it != in_use_.end()) {
			// Preserve the real arena-base allocation record when a zero-sized entry
			// aliases offset 0. Re-registering an existing sub-allocation is harmless.
			if (!it->second.sub_alloc) {
				return;
			}
		}
		in_use_[ptr] = DeviceAllocInfo {0, false, nullptr, true};
	}


	// free() calls cudaFree if ptr is not tracked.
	// release_arena_ptr() silently no-ops instead — used by DeviceArena destructor
	// so that sub-pointers and device_base_ already cleaned up by free_device_expr()
	// don't crash via cudaFree on an interior address (cudaErrorInvalidValue) or
	// double-free device_base_ out of the pool free-list.
	void free(void* ptr)              { do_free(ptr, /*fallback_cudafree=*/true);  }
	void release_arena_ptr(void* ptr) { do_free(ptr, /*fallback_cudafree=*/false); }

	void* alloc_pinned(size_t bytes) {
		return pinned_pool_.alloc(bytes);
	}

	void release_pinned(void* ptr) {
		pinned_pool_.release(ptr);
	}

	// Register a stream the caller has already issued async H2D work on so
	// sync_h2d() (no-arg) and sync_h2d(stream) drain it before reset/destroy.
	// Used by DeviceArena::upload for its aggregate DMA path — pinned
	// ownership stays with the caller.
	void register_external_h2d(cudaStream_t stream) {
		tracker_.register_external(stream);
	}

	void copy_h2d(void* dst, const void* src, size_t bytes) {
		copy_h2d_on_stream(dst, src, bytes, nullptr);
	}

	void copy_h2d_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		void* pinned = nullptr;
		bool  use_pinned_staging;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			use_pinned_staging = use_pinned_ && bytes > small_copy_threshold_;
		}
		if (use_pinned_staging) {
			pinned = pinned_pool_.alloc(bytes);
			std::memcpy(pinned, src, bytes);
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice, stream));
		} else {
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
		}
		tracker_.register_transfer(stream, pinned);
	}

	void sync_h2d() {
		tracker_.sync_all(make_release_pinned_fn());
	}

	void sync_h2d(cudaStream_t source_stream) {
		tracker_.sync_stream(source_stream, make_release_pinned_fn());
	}

	void set_enabled(bool enabled) {
		assert_idle_for_reconfiguration("set_enabled");
		std::lock_guard<std::mutex> lock(mutex_);
		enabled_ = enabled;
	}
	void set_use_async(bool use_async) {
		assert_idle_for_reconfiguration("set_use_async");
		std::lock_guard<std::mutex> lock(mutex_);
		use_async_ = use_async;
	}
	void set_use_pinned(bool use_pinned) {
		assert_idle_for_reconfiguration("set_use_pinned");
		{
			std::lock_guard<std::mutex> lock(mutex_);
			use_pinned_ = use_pinned;
		}
		pinned_pool_.set_use_pinned(use_pinned);
	}
	void set_small_copy_threshold(size_t bytes) {
		assert_idle_for_reconfiguration("set_small_copy_threshold");
		std::lock_guard<std::mutex> lock(mutex_);
		small_copy_threshold_ = bytes;
	}

	~DevicePool() {
		try {
			sync_h2d();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "DevicePool destructor: sync_h2d failed: %s\n", e.what());
		}

		{
			size_t real_leaks = 0;
			for (auto& [ptr, info] : in_use_) {
				(void)ptr;
				if (!info.sub_alloc) ++real_leaks;
			}
			if (real_leaks > 0) {
				std::fprintf(stderr,
				             "DevicePool warning: %zu in-use device allocations at shutdown; forcing free.\n",
				             real_leaks);
			}
		}
		for (auto& [ptr, info] : in_use_) {
			if (info.sub_alloc) continue; // interior arena pointer — no standalone cudaFree
			cudaFree(ptr);
		}
		in_use_.clear();

		for (auto& [size, list] : free_sync_by_size_) {
			(void)size;
			for (void* ptr : list) {
				cudaFree(ptr);
			}
		}
		free_sync_by_size_.clear();

		for (auto& [stream_id, buckets] : free_async_by_stream_size_) {
			(void)stream_id;
			for (auto& [size, list] : buckets) {
				(void)size;
				for (void* ptr : list) {
					cudaFree(ptr);
				}
			}
		}
		free_async_by_stream_size_.clear();
	}

private:
	using StreamKey = uintptr_t;

	static StreamKey stream_key(cudaStream_t stream) {
		return reinterpret_cast<StreamKey>(stream);
	}

	TransferTracker::ReleasePinnedFn make_release_pinned_fn() {
		return [this](void* pinned) {
			pinned_pool_.release(pinned);
		};
	}

	void do_free(void* ptr, bool fallback_cudafree) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = in_use_.find(ptr);
		if (it == in_use_.end()) {
			if (fallback_cudafree) {
				CUDA_SAFE_CALL(cudaFree(ptr));
			}
			return;
		}
		if (it->second.sub_alloc) {
			in_use_.erase(it);
			return; // arena sub-pointer: no actual GPU free
		}
		const auto info = it->second;
		in_use_.erase(it);
		if (enabled_) {
			if (info.async_alloc) {
				free_async_by_stream_size_[stream_key(info.alloc_stream)][info.size].push_back(ptr);
			} else {
				free_sync_by_size_[info.size].push_back(ptr);
			}
			return;
		}
		if (info.async_alloc) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, info.alloc_stream));
			return;
		}
		CUDA_SAFE_CALL(cudaFree(ptr));
	}

	// Caller must NOT hold mutex_ — the tracker's release callback delegates
	// into pinned_pool_ (its own mutex), so no cross-lock deadlock, but we
	// still keep the sweep off-lock to keep the hot path simple.
	void assert_idle_for_reconfiguration(const char* api_name) {
		tracker_.reclaim_finished(make_release_pinned_fn());

		std::lock_guard<std::mutex> lock(mutex_);
		bool has_real_allocs = false;
		for (auto& [ptr, info] : in_use_) {
			(void)ptr;
			if (!info.sub_alloc) { has_real_allocs = true; break; }
		}
		if (!tracker_.empty() || has_real_allocs || pinned_pool_.has_in_use()) {
			throw std::runtime_error(std::string("DevicePool::") + api_name +
			                         " requires idle pool (no in-flight copies or live allocations)");
		}
	}

	DevicePool() = default;

	std::mutex   mutex_;
	bool         enabled_              = true;
	bool         use_async_            = true;
	bool         use_pinned_           = true;
	size_t       small_copy_threshold_ = 256 * 1024;

	std::unordered_map<size_t, std::vector<void*>>                                free_sync_by_size_;
	std::unordered_map<StreamKey, std::unordered_map<size_t, std::vector<void*>>> free_async_by_stream_size_;
	std::unordered_map<void*, DeviceAllocInfo>                                    in_use_;

	PinnedHostPool                                                                pinned_pool_;
	TransferTracker                                                               tracker_;
};

inline void* device_malloc(size_t bytes) {
	return DevicePool::instance().alloc(bytes);
}

inline void* device_malloc_on_stream(size_t bytes, cudaStream_t stream) {
	return DevicePool::instance().alloc_on_stream(bytes, stream);
}

inline void device_free(void* ptr) {
	DevicePool::instance().free(ptr);
}

inline void device_memcpy_h2d(void* dst, const void* src, size_t bytes) {
	DevicePool::instance().copy_h2d(dst, src, bytes);
}

inline void device_memcpy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
	DevicePool::instance().copy_h2d_on_stream(dst, src, bytes, stream);
}

inline void sync_h2d() {
	DevicePool::instance().sync_h2d();
}

inline void sync_h2d(cudaStream_t source_stream) {
	DevicePool::instance().sync_h2d(source_stream);
}

} // namespace galp::memory

#endif // GALP_MEMORY_DEVICE_POOL_CUH
