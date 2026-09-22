// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/device_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_DEVICE_POOL_CUH
#define GALP_MEMORY_DEVICE_POOL_CUH

#include "cuda/cuda_macros.cuh"
#include "cuda/memory/pinned_host_pool.cuh"
#include "cuda/memory/transfer_tracker.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iterator>
#include <map>
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
	int          device       = -1;
	bool         h2d_pending  = false;
	cudaStream_t h2d_stream   = nullptr;
};

struct DevicePoolStats {
	size_t in_use_bytes          = 0;
	size_t peak_in_use_bytes     = 0;
	size_t cached_bytes          = 0;
	size_t allocation_requests   = 0;
	size_t cuda_allocation_count = 0;
	size_t cuda_allocation_bytes = 0;
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
		int device = 0;
		CUDA_SAFE_CALL(cudaGetDevice(&device));
		bool use_async = false;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			use_async = use_async_ && stream != nullptr;
			if (enabled_ && !use_async) {
				size_t actual_size = 0;
				void*  cached_ptr  = take_cached_block_locked(bytes, device, actual_size);
				if (cached_ptr != nullptr) {
					in_use_[cached_ptr] = DeviceAllocInfo {actual_size, false, nullptr, false, device};
					record_in_use_allocation_locked(actual_size, /*cuda_allocation=*/false);
					return cached_ptr;
				}
			}
		}

		void* ptr         = nullptr;
		bool  async_alloc = false;
		if (use_async) {
			auto status = cudaMallocAsync(&ptr, bytes, stream);
			if (status != cudaSuccess) {
				ptr = nullptr;
				release_cached();
				status = cudaMallocAsync(&ptr, bytes, stream);
			}
			if (status == cudaSuccess) {
				async_alloc = true;
			} else {
				ptr = nullptr;
			}
		}
		if (!ptr) {
			auto status = cudaMalloc(&ptr, bytes);
			if (status != cudaSuccess) {
				ptr = nullptr;
				release_cached();
				status = cudaMalloc(&ptr, bytes);
			}
			CUDA_SAFE_CALL(status);
			async_alloc = false;
		}
		{
			std::lock_guard<std::mutex> lock(mutex_);
			in_use_[ptr] = DeviceAllocInfo {bytes, async_alloc, async_alloc ? stream : nullptr, false, device};
			record_in_use_allocation_locked(bytes, /*cuda_allocation=*/true);
		}
		return ptr;
	}

	DevicePoolStats stats() {
		std::lock_guard<std::mutex> lock(mutex_);
		return DevicePoolStats {in_use_bytes_,
		                        peak_in_use_bytes_,
		                        free_cached_bytes_,
		                        allocation_requests_,
		                        cuda_allocation_count_,
		                        cuda_allocation_bytes_};
	}

	PinnedHostPoolStats pinned_stats() {
		return pinned_pool_.stats();
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
		in_use_[ptr] = DeviceAllocInfo {0, false, nullptr, true, -1};
	}

	// free() calls cudaFree if ptr is not tracked.
	// release_arena_ptr() silently no-ops instead — used by DeviceArena destructor
	// so that sub-pointers and device_base_ already cleaned up by free_device_expr()
	// don't crash via cudaFree on an interior address (cudaErrorInvalidValue) or
	// double-free device_base_ out of the pool free-list.
	void free(void* ptr) {
		do_free(ptr, /*fallback_cudafree=*/true);
	}
	void release_arena_ptr(void* ptr) {
		do_free(ptr, /*fallback_cudafree=*/false);
	}

	void* alloc_pinned(size_t bytes) {
		return pinned_pool_.alloc(bytes);
	}

	void release_pinned(void* ptr) {
		pinned_pool_.release(ptr);
	}

	// Device memory remains in in_use_ until tracking proves its DMA complete.
	// External pinned backing remains owned by the caller through sync_h2d().
	template <typename Submit>
	void submit_external_h2d(void* dst, cudaStream_t stream, Submit&& issue) {
		track_device_transfer(dst, stream);
		tracker_.submit(stream, [&](void*&) { issue(); });
	}

	void sync_pending_h2d(cudaStream_t stream) {
		tracker_.sync_stream(stream, make_release_pinned_fn(), true);
	}

	void copy_h2d(void* dst, const void* src, size_t bytes) {
		copy_h2d_on_stream(dst, src, bytes, nullptr);
	}

	void copy_h2d_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		bool use_pinned_staging;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			use_pinned_staging = use_pinned_ && bytes > small_copy_threshold_;
		}
		if (dst == nullptr || src == nullptr) {
			CUDA_SAFE_CALL(cudaErrorInvalidValue);
		}
		track_device_transfer(dst, stream);
		tracker_.submit(stream, [&](void*& pinned) {
			if (use_pinned_staging) {
				pinned = pinned_pool_.alloc(bytes);
				std::memcpy(pinned, src, bytes);
			}
			CUDA_SAFE_CALL(
			    cudaMemcpyAsync(dst, pinned != nullptr ? pinned : src, bytes, cudaMemcpyHostToDevice, stream));
		});
	}

	// Issue an async upload from storage whose pinned lifetime is owned by the
	// caller.  Unlike copy_h2d_on_stream(), this path must not acquire another
	// staging allocation or copy the source again.  The transfer tracker still
	// owns stream completion, while the caller keeps the source alive until the
	// corresponding stream work has completed.
	void copy_pinned_h2d_on_stream(void* dst, const void* pinned_src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		if (dst == nullptr || pinned_src == nullptr) {
			throw std::invalid_argument("DevicePool::copy_pinned_h2d_on_stream received a null pointer");
		}
		submit_external_h2d(dst, stream, [&] {
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream));
		});
	}

	void sync_h2d() {
		tracker_.sync_all(make_release_pinned_fn());
	}

	void sync_h2d(cudaStream_t source_stream) {
		tracker_.sync_stream(source_stream, make_release_pinned_fn());
	}

	void complete_h2d(cudaStream_t source_stream) {
		tracker_.complete_stream(source_stream, make_release_pinned_fn());
	}

	void release_cached() {
		sync_h2d();

		std::map<size_t, std::vector<CachedBlock>> sync_free;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			sync_free.swap(free_sync_by_size_);
			free_cached_bytes_ = 0;
		}

		for (auto& [size, list] : sync_free) {
			(void)size;
			for (const CachedBlock& block : list) {
				free_cached_block(block);
			}
		}
		pinned_pool_.release_cached();
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
		// Runs only at process exit (Meyers singleton). The CUDA runtime registers its
		// own atexit handler to tear down the context/stream-ordered memory pool; if that
		// runs first, any free here dereferences freed driver state and segfaults deep in
		// libcuda (cudaFree on cudaMallocAsync memory routes to cuMemFreeAsync). Probe with
		// a cheap call: once the runtime is unloading, skip all frees — the driver reclaims
		// every device allocation when the context is destroyed.
		int device = 0;
		if (cudaGetDevice(&device) != cudaSuccess) {
			pinned_pool_.abandon_without_free();
			return;
		}

		try {
			sync_h2d();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "DevicePool destructor: sync_h2d failed: %s\n", e.what());
			// Completion is unknown. Leave both kinds of allocation to context
			// teardown rather than freeing memory that DMA may still reference.
			pinned_pool_.abandon_without_free();
			return;
		}

		{
			size_t real_leaks = 0;
			for (auto& [ptr, info] : in_use_) {
				(void)ptr;
				if (!info.sub_alloc)
					++real_leaks;
			}
			if (real_leaks > 0) {
				std::fprintf(stderr,
				             "DevicePool warning: %zu in-use device allocations at shutdown; forcing free.\n",
				             real_leaks);
			}
		}
		for (auto& [ptr, info] : in_use_) {
			if (info.sub_alloc)
				continue; // interior arena pointer - no standalone cudaFree
			static_cast<void>(cudaSetDevice(info.device));
			cudaFree(ptr);
		}
		in_use_.clear();

		for (auto& [size, list] : free_sync_by_size_) {
			(void)size;
			for (const CachedBlock& block : list) {
				static_cast<void>(cudaSetDevice(block.device));
				cudaFree(block.ptr);
			}
		}
		free_sync_by_size_.clear();
	}

private:
	struct CachedBlock {
		void* ptr    = nullptr;
		int   device = -1;
	};

	class ScopedDevice {
	public:
		explicit ScopedDevice(int device) {
			CUDA_SAFE_CALL(cudaGetDevice(&previous_));
			if (previous_ != device) {
				CUDA_SAFE_CALL(cudaSetDevice(device));
				restore_ = true;
			}
		}
		~ScopedDevice() {
			if (restore_) {
				CUDA_LOG_CALL(cudaSetDevice(previous_));
			}
		}

		ScopedDevice(const ScopedDevice&)            = delete;
		ScopedDevice& operator=(const ScopedDevice&) = delete;

	private:
		int  previous_ = -1;
		bool restore_  = false;
	};

	void record_in_use_allocation_locked(const size_t bytes, const bool cuda_allocation) {
		in_use_bytes_ += bytes;
		peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
		++allocation_requests_;
		if (cuda_allocation) {
			++cuda_allocation_count_;
			cuda_allocation_bytes_ += bytes;
		}
	}

	static void free_cached_block(const CachedBlock& block) {
		ScopedDevice device(block.device);
		CUDA_SAFE_CALL(cudaFree(block.ptr));
	}

	TransferTracker::ReleasePinnedFn make_release_pinned_fn() {
		return [this](void* pinned) {
			pinned_pool_.release(pinned);
		};
	}

	void track_device_transfer(void* ptr, cudaStream_t stream) {
		std::unique_lock<std::mutex> lock(mutex_);
		auto                         it = in_use_.find(ptr);
		if (it == in_use_.end() || it->second.sub_alloc) {
			return; // External allocations retain their caller's lifetime contract.
		}
		const auto info = it->second;
		if (info.h2d_pending && info.h2d_stream != stream) {
			lock.unlock();
			ScopedDevice device(info.device);
			sync_pending_h2d(info.h2d_stream);
			lock.lock();
			it = in_use_.find(ptr);
		}
		it->second.h2d_pending = true;
		it->second.h2d_stream  = stream;
	}

	void do_free(void* ptr, bool fallback_cudafree) {
		if (ptr == nullptr) {
			return;
		}
		std::unique_lock<std::mutex> lock(mutex_);
		auto                         it = in_use_.find(ptr);
		if (it == in_use_.end()) {
			lock.unlock();
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
		if (info.h2d_pending) {
			lock.unlock();
			ScopedDevice device(info.device);
			sync_pending_h2d(info.h2d_stream);
			lock.lock();
			it = in_use_.find(ptr);
		}
		in_use_.erase(it);
		in_use_bytes_ = info.size <= in_use_bytes_ ? in_use_bytes_ - info.size : 0;
		if (enabled_ && !info.async_alloc) {
			std::vector<CachedBlock> evicted_cached;
			if (info.size <= free_cache_limit_bytes_) {
				evict_cached_until_room_locked(info.size, evicted_cached);
				free_sync_by_size_[info.size].push_back(CachedBlock {ptr, info.device});
				free_cached_bytes_ += info.size;
				lock.unlock();
				for (const CachedBlock& evicted : evicted_cached) {
					free_cached_block(evicted);
				}
				return;
			}
			lock.unlock();
			ScopedDevice device(info.device);
			CUDA_SAFE_CALL(cudaFree(ptr));
			return;
		}
		lock.unlock();
		ScopedDevice device(info.device);
		if (info.async_alloc) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, info.alloc_stream));
			return;
		}
		CUDA_SAFE_CALL(cudaFree(ptr));
	}

	void evict_cached_until_room_locked(size_t required_bytes, std::vector<CachedBlock>& evicted) {
		while (free_cached_bytes_ + required_bytes > free_cache_limit_bytes_) {
			if (!evict_largest_cached_locked(evicted)) {
				return;
			}
		}
	}

	bool evict_largest_cached_locked(std::vector<CachedBlock>& evicted) {
		if (free_sync_by_size_.empty()) {
			return false;
		}

		auto  bucket_it = std::prev(free_sync_by_size_.end());
		auto& list      = bucket_it->second;
		evicted.push_back(list.back());
		list.pop_back();
		free_cached_bytes_ -= bucket_it->first;
		if (list.empty()) {
			free_sync_by_size_.erase(bucket_it);
		}
		return true;
	}

	bool reusable_size(size_t request_bytes, size_t cached_bytes) const {
		if (cached_bytes < request_bytes) {
			return false;
		}
		return max_reuse_slack_bytes_ == 0 || cached_bytes - request_bytes <= max_reuse_slack_bytes_;
	}

	void* take_cached_block_locked(size_t request_bytes, int device, size_t& actual_size) {
		auto it = free_sync_by_size_.lower_bound(request_bytes);
		for (; it != free_sync_by_size_.end() && reusable_size(request_bytes, it->first); ++it) {
			auto& list  = it->second;
			auto  block = std::find_if(list.begin(), list.end(), [device](const CachedBlock& candidate) {
				return candidate.device == device;
			});
			if (block == list.end()) {
				continue;
			}
			void* ptr = block->ptr;
			list.erase(block);
			actual_size = it->first;
			free_cached_bytes_ -= actual_size;
			if (list.empty()) {
				free_sync_by_size_.erase(it);
			}
			return ptr;
		}
		return nullptr;
	}

	// Caller must NOT hold mutex_ - the tracker's release callback delegates
	// into pinned_pool_ (its own mutex), so no cross-lock deadlock, but we
	// still keep the sweep off-lock to keep the hot path simple.
	void assert_idle_for_reconfiguration(const char* api_name) {
		tracker_.reclaim_finished(make_release_pinned_fn());

		std::lock_guard<std::mutex> lock(mutex_);
		bool                        has_real_allocs = false;
		for (auto& [ptr, info] : in_use_) {
			(void)ptr;
			if (!info.sub_alloc) {
				has_real_allocs = true;
				break;
			}
		}
		if (!tracker_.empty() || has_real_allocs || pinned_pool_.has_in_use()) {
			throw std::runtime_error(std::string("DevicePool::") + api_name +
			                         " requires idle pool (no in-flight copies or live allocations)");
		}
	}

	static size_t read_size_env(const char* name, size_t fallback) {
		const char* value = std::getenv(name);
		if (value == nullptr || *value == '\0') {
			return fallback;
		}
		char* end    = nullptr;
		auto  parsed = std::strtoull(value, &end, 10);
		if (end == value || *end != '\0') {
			return fallback;
		}
		return static_cast<size_t>(parsed);
	}

	DevicePool()
	    : free_cache_limit_bytes_(read_size_env("GALP_DEVICE_POOL_CACHE_LIMIT_BYTES", 1024ULL * 1024ULL * 1024ULL))
	    , max_reuse_slack_bytes_(read_size_env("GALP_DEVICE_POOL_MAX_REUSE_SLACK_BYTES", 256ULL * 1024ULL * 1024ULL))
	    , pinned_pool_(read_size_env("GALP_PINNED_HOST_POOL_CACHE_LIMIT_BYTES", 256ULL * 1024ULL * 1024ULL),
	                   read_size_env("GALP_PINNED_HOST_POOL_MAX_REUSE_SLACK_BYTES", 64ULL * 1024ULL * 1024ULL)) {
	}

	std::mutex mutex_;
	bool       enabled_                = true;
	bool       use_async_              = true;
	bool       use_pinned_             = true;
	size_t     small_copy_threshold_   = 256 * 1024;
	size_t     free_cache_limit_bytes_ = 0;
	size_t     max_reuse_slack_bytes_  = 0;
	size_t     free_cached_bytes_      = 0;
	size_t     in_use_bytes_           = 0;
	size_t     peak_in_use_bytes_      = 0;
	size_t     allocation_requests_    = 0;
	size_t     cuda_allocation_count_  = 0;
	size_t     cuda_allocation_bytes_  = 0;

	std::map<size_t, std::vector<CachedBlock>> free_sync_by_size_;
	std::unordered_map<void*, DeviceAllocInfo> in_use_;

	PinnedHostPool  pinned_pool_;
	TransferTracker tracker_;
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

inline void device_release_cached() {
	DevicePool::instance().release_cached();
}

inline DevicePoolStats device_pool_stats() {
	return DevicePool::instance().stats();
}

inline PinnedHostPoolStats pinned_host_pool_stats() {
	return DevicePool::instance().pinned_stats();
}

inline void device_memcpy_h2d(void* dst, const void* src, size_t bytes) {
	DevicePool::instance().copy_h2d(dst, src, bytes);
}

inline void device_memcpy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
	DevicePool::instance().copy_h2d_on_stream(dst, src, bytes, stream);
}

inline void device_memcpy_pinned_h2d_async(void* dst, const void* pinned_src, size_t bytes, cudaStream_t stream) {
	DevicePool::instance().copy_pinned_h2d_on_stream(dst, pinned_src, bytes, stream);
}

inline void sync_h2d() {
	DevicePool::instance().sync_h2d();
}

inline void sync_h2d(cudaStream_t source_stream) {
	DevicePool::instance().sync_h2d(source_stream);
}

inline void complete_h2d(cudaStream_t source_stream) {
	DevicePool::instance().complete_h2d(source_stream);
}

} // namespace galp::memory

#endif // GALP_MEMORY_DEVICE_POOL_CUH
