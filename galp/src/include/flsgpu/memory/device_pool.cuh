// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/memory/device_pool.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_MEMORY_DEVICE_POOL_CUH
#define FLSGPU_MEMORY_DEVICE_POOL_CUH

#include "flsgpu/memory/cuda_macros.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace flsgpu { namespace memory {

struct DeviceAllocInfo {
	size_t       size         = 0;
	bool         async_alloc  = false;
	cudaStream_t alloc_stream = nullptr;
	bool         sub_alloc    = false; // true for arena sub-pointers (no-op on free)
};

struct PendingCopyEvent {
	cudaEvent_t event  = nullptr;
	void*       pinned = nullptr;
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
		std::lock_guard<std::mutex> lock(mutex_);
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

		void* ptr         = nullptr;
		bool  async_alloc = false;
		if (use_async_) {
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
		in_use_[ptr] = DeviceAllocInfo {bytes, async_alloc, async_alloc ? stream : nullptr};
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
		if (bytes == 0) {
			return nullptr;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		return alloc_pinned_locked(bytes);
	}

	void release_pinned(void* ptr) {
		std::lock_guard<std::mutex> lock(mutex_);
		release_pinned_locked(ptr);
	}

	// Register a stream the caller has already issued async H2D work on so
	// sync_h2d() (no-arg) and sync_h2d(stream) drain it before reset/destroy.
	// Use when bypassing queue_staged_h2d/copy_h2d_on_stream for aggregate DMA
	// issue (DeviceArena::upload). pinned ownership stays with the caller.
	void register_external_h2d(cudaStream_t stream) {
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(stream)].push_back(PendingCopyEvent {event, nullptr});
	}

	// Queue a single H2D transfer from an already-pinned buffer.
	// When batch is active, queued as one entry; pinned is freed after batch event.
	// When batch is not active, issued directly with its own event.
	void queue_staged_h2d(void* dst, void* pinned_src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream));
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(stream)].push_back(PendingCopyEvent {event, pinned_src});
	}

	void copy_h2d(void* dst, const void* src, size_t bytes) {
		copy_h2d_on_stream(dst, src, bytes, nullptr);
	}

	void copy_h2d_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		void* pinned = nullptr;
		if (use_pinned_ && bytes > small_copy_threshold_) {
			pinned = alloc_pinned(bytes);
			std::memcpy(pinned, src, bytes);
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice, stream));
		} else {
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
		}
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(stream)].push_back(PendingCopyEvent {event, pinned});
	}

	void sync_h2d() {
		// Collect streams to synchronize while holding the lock, then release
		// the lock before calling cudaStreamSynchronize (which may block).
		std::vector<cudaStream_t> streams_to_sync;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			for (const auto& [stream_id, events] : pending_copy_events_) {
				(void)events;
				streams_to_sync.push_back(stream_from_key(stream_id));
			}
		}

		for (auto* s : streams_to_sync) {
			CUDA_SAFE_CALL(cudaStreamSynchronize(s));
		}

		// Now reclaim events and pinned memory under the lock.
		std::lock_guard<std::mutex> lock(mutex_);
		for (auto& [stream_id, events] : pending_copy_events_) {
			(void)stream_id;
			for (auto& pending : events) {
				if (pending.event != nullptr) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
				}
				release_pinned_locked(pending.pinned);
			}
		}
		pending_copy_events_.clear();
	}

	void sync_h2d(cudaStream_t source_stream) {
		CUDA_SAFE_CALL(cudaStreamSynchronize(source_stream));

		// Erase all pending records keyed by this stream before the caller destroys it.
		// This prevents stale-key reuse if CUDA later recycles the same stream handle value.
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = pending_copy_events_.find(stream_key(source_stream));
		if (it != pending_copy_events_.end()) {
			for (auto& pending : it->second) {
				if (pending.event != nullptr) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
				}
				release_pinned_locked(pending.pinned);
			}
			pending_copy_events_.erase(it);
		}
	}

	void set_enabled(bool enabled) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_enabled");
		enabled_ = enabled;
	}
	void set_use_async(bool use_async) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_use_async");
		use_async_ = use_async;
	}
	void set_use_pinned(bool use_pinned) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_use_pinned");
		use_pinned_ = use_pinned;
	}
	void set_small_copy_threshold(size_t bytes) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_small_copy_threshold");
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
				fprintf(stderr, "DevicePool warning: %zu in-use device allocations at shutdown; forcing free.\n",
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

		if (!pinned_in_use_.empty()) {
			fprintf(stderr, "DevicePool warning: %zu in-use pinned allocations at shutdown; forcing free.\n",
			        pinned_in_use_.size());
		}
		for (auto& [ptr, bytes] : pinned_in_use_) {
			(void)bytes;
			cudaFreeHost(ptr);
		}
		pinned_in_use_.clear();

		for (auto& [size, list] : pinned_free_by_size_) {
			(void)size;
			for (void* ptr : list) {
				cudaFreeHost(ptr);
			}
		}
		pinned_free_by_size_.clear();
	}

private:
	using StreamKey = uintptr_t;

	static StreamKey stream_key(cudaStream_t stream) {
		return reinterpret_cast<StreamKey>(stream);
	}

	static cudaStream_t stream_from_key(StreamKey key) {
		return reinterpret_cast<cudaStream_t>(key);
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

	void release_pinned_locked(void*& pinned) {
		if (pinned == nullptr) {
			return;
		}
		auto pinned_it = pinned_in_use_.find(pinned);
		if (pinned_it != pinned_in_use_.end()) {
			const size_t bytes = pinned_it->second;
			pinned_in_use_.erase(pinned_it);
			if (use_pinned_) {
				pinned_free_by_size_[bytes].push_back(pinned);
			} else {
				CUDA_SAFE_CALL(cudaFreeHost(pinned));
			}
		} else {
			CUDA_SAFE_CALL(cudaFreeHost(pinned));
		}
		pinned = nullptr;
	}

	void* alloc_pinned_locked(size_t bytes) {
		if (use_pinned_) {
			// Best-fit: reuse the smallest free buffer >= bytes. Exact-size
			// matching forced fresh cudaMallocHost when staged_bytes variance
			// pushed a chunk into a nearby size bucket.
			for (auto it = pinned_free_by_size_.lower_bound(bytes); it != pinned_free_by_size_.end(); ++it) {
				if (!it->second.empty()) {
					const size_t bucket_size = it->first;
					void*        ptr         = it->second.back();
					it->second.pop_back();
					pinned_in_use_[ptr] = bucket_size;
					return ptr;
				}
			}
		}

		void* ptr = nullptr;
		CUDA_SAFE_CALL(cudaMallocHost(&ptr, bytes));
		pinned_in_use_[ptr] = bytes;
		return ptr;
	}

	void assert_idle_for_reconfiguration_locked(const char* api_name) {
		reclaim_finished_locked();
		// sub_alloc entries are phantom bookkeeping entries for arena interior pointers
		// and do not represent actual live GPU allocations — exclude them from the check.
		bool has_real_allocs = false;
		for (auto& [ptr, info] : in_use_) {
			(void)ptr;
			if (!info.sub_alloc) { has_real_allocs = true; break; }
		}
		if (!pending_copy_events_.empty() || has_real_allocs || !pinned_in_use_.empty()) {
			throw std::runtime_error(std::string("DevicePool::") + api_name +
			                         " requires idle pool (no in-flight copies or live allocations)");
		}
	}

	void reclaim_finished_locked() {
		for (auto it = pending_copy_events_.begin(); it != pending_copy_events_.end();) {
			auto& events = it->second;
			for (auto event_it = events.begin(); event_it != events.end();) {
				auto& pending = *event_it;
				if (pending.event == nullptr) {
					event_it = events.erase(event_it);
					continue;
				}
				auto status = cudaEventQuery(pending.event);
				if (status == cudaSuccess) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
					release_pinned_locked(pending.pinned);
					event_it = events.erase(event_it);
				} else if (status == cudaErrorNotReady) {
					++event_it;
				} else {
					CUDA_SAFE_CALL(status);
				}
			}
			if (events.empty()) {
				it = pending_copy_events_.erase(it);
			} else {
				++it;
			}
		}
	}

	DevicePool() = default;

	std::mutex   mutex_;
	bool         enabled_              = true;
	bool         use_async_            = true;
	bool         use_pinned_           = true;
	size_t       small_copy_threshold_ = 256 * 1024;

	std::unordered_map<size_t, std::vector<void*>>                                      free_sync_by_size_;
	std::unordered_map<StreamKey, std::unordered_map<size_t, std::vector<void*>>> free_async_by_stream_size_;
	std::unordered_map<void*, DeviceAllocInfo>     in_use_;

	std::map<size_t, std::vector<void*>>           pinned_free_by_size_;
	std::unordered_map<void*, size_t>              pinned_in_use_;
	std::unordered_map<StreamKey, std::vector<PendingCopyEvent>> pending_copy_events_;
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

}} // namespace flsgpu::memory

#endif // FLSGPU_MEMORY_DEVICE_POOL_CUH
