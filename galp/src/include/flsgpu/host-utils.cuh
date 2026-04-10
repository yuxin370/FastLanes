// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/host-utils.cuh
// ────────────────────────────────────────────────────────
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#define CUDA_SAFE_CALL(call)                                                                                           \
	do {                                                                                                               \
		cudaError_t err = call;                                                                                        \
		if (cudaSuccess != err) {                                                                                      \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err));  \
			exit(EXIT_FAILURE);                                                                                        \
		}                                                                                                              \
	} while (0)

#define CUDA_SAFE_CALL_TRACED(call)                                                                                    \
	do {                                                                                                               \
		fprintf(stderr, "Start CUDA_CALL ['%s': line %i\n", __FILE__, __LINE__);                                       \
		cudaError_t err = call;                                                                                        \
		fprintf(stderr, "End CUDA_CALL ['%s': line %i\n", __FILE__, __LINE__);                                         \
		if (cudaSuccess != err) {                                                                                      \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err));  \
			exit(EXIT_FAILURE);                                                                                        \
		}                                                                                                              \
	} while (0)

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
		return alloc_on_stream(bytes, stream_);
	}

	void* alloc_on_stream(size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return nullptr;
		}
		const auto use_stream = stream != nullptr ? stream : stream_;
		std::lock_guard<std::mutex> lock(mutex_);
		if (enabled_) {
			if (use_async_) {
				auto& free_list = free_async_by_stream_size_[stream_key(use_stream)][bytes];
				if (!free_list.empty()) {
					void* ptr = free_list.back();
					free_list.pop_back();
					in_use_[ptr] = DeviceAllocInfo {bytes, true, use_stream};
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
			auto status = cudaMallocAsync(&ptr, bytes, use_stream);
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
		in_use_[ptr] = DeviceAllocInfo {bytes, async_alloc, async_alloc ? use_stream : nullptr};
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


	void free(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = in_use_.find(ptr);
		if (it != in_use_.end()) {
			if (it->second.sub_alloc) {
				in_use_.erase(it);
				return; // arena sub-pointer: no actual free
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
				CUDA_SAFE_CALL(cudaFreeAsync(ptr, info.alloc_stream != nullptr ? info.alloc_stream : stream_));
				return;
			}
			CUDA_SAFE_CALL(cudaFree(ptr));
			return;
		}

		CUDA_SAFE_CALL(cudaFree(ptr));
	}

	void* alloc_pinned(size_t bytes) {
		if (bytes == 0) {
			return nullptr;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		return alloc_pinned_locked(bytes);
	}

	// Queue a single H2D transfer from an already-pinned buffer.
	// When batch is active, queued as one entry; pinned is freed after batch event.
	// When batch is not active, issued directly with its own event.
	void queue_staged_h2d(void* dst, void* pinned_src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		const auto use_stream = stream != nullptr ? stream : stream_;
		CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned_src, bytes, cudaMemcpyHostToDevice, use_stream));
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, use_stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(use_stream)].push_back(PendingCopyEvent {event, pinned_src});
	}

	void copy_h2d(void* dst, const void* src, size_t bytes) {
		copy_h2d_on_stream(dst, src, bytes, stream_);
	}

	void copy_h2d_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		const auto use_stream = stream != nullptr ? stream : stream_;

		void* pinned = nullptr;
		if (use_pinned_ && bytes > small_copy_threshold_) {
			pinned = alloc_pinned(bytes);
			std::memcpy(pinned, src, bytes);
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice, use_stream));
		} else {
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, use_stream));
		}

		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, use_stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(use_stream)].push_back(PendingCopyEvent {event, pinned});
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
		const auto src_stream = source_stream != nullptr ? source_stream : stream_;
		CUDA_SAFE_CALL(cudaStreamSynchronize(src_stream));

		// Erase all pending records keyed by this stream before the caller destroys it.
		// This prevents stale-key reuse if CUDA later recycles the same stream handle value.
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = pending_copy_events_.find(stream_key(src_stream));
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

	cudaStream_t stream() const {
		return stream_;
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
		sync_h2d();

		if (!in_use_.empty()) {
			fprintf(stderr, "DevicePool warning: %zu in-use device allocations at shutdown; forcing free.\n", in_use_.size());
		}
		for (auto& [ptr, info] : in_use_) {
			if (info.sub_alloc) continue; // skip arena sub-pointers
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
			auto& free_list = pinned_free_by_size_[bytes];
			if (!free_list.empty()) {
				void* ptr = free_list.back();
				free_list.pop_back();
				pinned_in_use_[ptr] = bytes;
				return ptr;
			}
		}

		void* ptr = nullptr;
		CUDA_SAFE_CALL(cudaMallocHost(&ptr, bytes));
		pinned_in_use_[ptr] = bytes;
		return ptr;
	}

	void assert_idle_for_reconfiguration_locked(const char* api_name) {
		reclaim_finished_locked();
		if (!pending_copy_events_.empty() || !in_use_.empty() || !pinned_in_use_.empty()) {
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

	DevicePool() {
		stream_ = 0;
	}

	std::mutex   mutex_;
	bool         enabled_              = true;
	bool         use_async_            = true;
	bool         use_pinned_           = true;
	size_t       small_copy_threshold_ = 256 * 1024;
	cudaStream_t stream_               = nullptr;

	std::unordered_map<size_t, std::vector<void*>>                                      free_sync_by_size_;
	std::unordered_map<StreamKey, std::unordered_map<size_t, std::vector<void*>>> free_async_by_stream_size_;
	std::unordered_map<void*, DeviceAllocInfo>     in_use_;

	std::unordered_map<size_t, std::vector<void*>> pinned_free_by_size_;
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

inline void sync_h2d() {
	DevicePool::instance().sync_h2d();
}

inline void sync_h2d(cudaStream_t source_stream) {
	DevicePool::instance().sync_h2d(source_stream);
}

// ── DeviceArena: aggregated device allocation + staged H2D ──────────
// A workset can append many column sub-arrays into one arena, then perform one
// upload() to allocate/copy the aggregated payload and resolve device pointers.
//
// Host data must remain valid until upload().
// For temporary host columns, use defer_free() to extend their lifetime.
//
// Usage:
//   DeviceArena arena(stream);
//   auto i0 = arena.add<T>(count, host_ptr);
//   arena.upload();
//   T* d0 = arena.get<T>(i0);
class DeviceArena {
	struct Entry {
		size_t      offset     = 0;
		size_t      copy_bytes = 0;
		const void* host_src   = nullptr;
	};

public:
	explicit DeviceArena(cudaStream_t stream) : stream_(stream) {}
	~DeviceArena() {
		run_deferred_frees();
	}

	DeviceArena(const DeviceArena&)            = delete;
	DeviceArena& operator=(const DeviceArena&) = delete;

	template <typename T>
	size_t add(size_t count, const T* host_src, size_t buffer_elements = 0) {
		const size_t copy_bytes  = count * sizeof(T);
		const size_t alloc_bytes = copy_bytes + buffer_elements * sizeof(T);
		// Align offset to 256 bytes for coalesced access
		total_bytes_ = (total_bytes_ + 255U) & ~size_t(255U);
		const size_t idx = entries_.size();
		entries_.push_back(Entry {total_bytes_, copy_bytes, host_src});
		total_bytes_ += alloc_bytes;
		return idx;
	}

	/// Register a callback to be invoked after upload() sets device_base_.
	/// Use this to populate device column pointers after the aggregated upload.
	void add_resolver(std::function<void()> fn) {
		resolvers_.push_back(std::move(fn));
	}

	/// Defer a cleanup action until after upload() completes.
	/// Use this to extend the lifetime of temporary host columns whose data
	/// must remain valid until upload() packs it into the pinned buffer.
	void defer_free(std::function<void()> fn) {
		deferred_frees_.push_back(std::move(fn));
	}

	void upload() {
		if (total_bytes_ == 0 || entries_.empty()) {
			for (auto& fn : resolvers_) {
				fn();
			}
			resolvers_.clear();
			run_deferred_frees();
			return;
		}
		auto& pool = DevicePool::instance();

		// Round up to power-of-2 buckets (min 64KB) to improve DevicePool cache hits.
		// Without this, each chunk's unique total_bytes_ misses the size-keyed free-list,
		// causing expensive cudaMallocAsync/cudaMallocHost on every chunk.
		const size_t alloc_bytes = round_up_pow2(total_bytes_, 65536U);

		// Single device allocation (arena)
		device_base_ = reinterpret_cast<char*>(pool.alloc_on_stream(alloc_bytes, stream_));

		// Pack all sub-arrays into a single pinned buffer, then queue ONE H2D copy.
		void* pinned = pool.alloc_pinned(alloc_bytes);
		for (const auto& e : entries_) {
			if (e.host_src != nullptr && e.copy_bytes > 0) {
				std::memcpy(reinterpret_cast<char*>(pinned) + e.offset, e.host_src, e.copy_bytes);
			}
		}
		pool.queue_staged_h2d(device_base_, pinned, total_bytes_, stream_);

		// Register sub-pointers so that free_device_pointer on them is a no-op.
		// The first entry is at the arena base and keeps the real allocation.
		for (size_t i = 1; i < entries_.size(); ++i) {
			void* sub_ptr = device_base_ + entries_[i].offset;
			pool.register_sub_allocation(sub_ptr);
		}

		// Invoke resolver callbacks to populate device column pointers.
		for (auto& fn : resolvers_) {
			fn();
		}
		resolvers_.clear();

		// Release temporary host data that was kept alive for deferred packing.
		run_deferred_frees();
	}

	template <typename T>
	T* get(size_t idx) const {
		return reinterpret_cast<T*>(device_base_ + entries_[idx].offset);
	}

	size_t total_bytes() const {
		return total_bytes_;
	}

	size_t entry_count() const {
		return entries_.size();
	}

private:
	void run_deferred_frees() {
		for (auto& fn : deferred_frees_) {
			fn();
		}
		deferred_frees_.clear();
	}

	/// Round up to the next power of 2 that is >= min_bucket.
	static size_t round_up_pow2(size_t bytes, size_t min_bucket) {
		if (bytes <= min_bucket) {
			return min_bucket;
		}
		// Next power of 2 >= bytes
		size_t v = bytes - 1;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v |= v >> 32;
		return v + 1;
	}

	cudaStream_t                       stream_      = nullptr;
	char*                              device_base_ = nullptr;
	size_t                             total_bytes_ = 0;
	std::vector<Entry>                 entries_;
	std::vector<std::function<void()>> resolvers_;
	std::vector<std::function<void()>> deferred_frees_;
};

inline void device_memcpy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
	DevicePool::instance().copy_h2d_on_stream(dst, src, bytes, stream);
}

}} // namespace flsgpu::memory

template <typename T>
void free_device_pointer(T*& device_ptr) {
	if (device_ptr != nullptr) {
		flsgpu::memory::device_free(device_ptr);
	}
	device_ptr = nullptr;
}

template <typename T>
class GPUArray {
private:
	size_t allocation_size;
	size_t memory_size;
	T*     device_ptr = nullptr;

	void allocate() {
		device_ptr = reinterpret_cast<T*>(flsgpu::memory::device_malloc(allocation_size));
	}
	void allocate(cudaStream_t stream) {
		device_ptr = reinterpret_cast<T*>(flsgpu::memory::device_malloc_on_stream(allocation_size, stream));
	}

public:
	GPUArray(const size_t count) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate();
	}

	GPUArray(const size_t count, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		if (stream != nullptr) {
			allocate(stream);
		} else {
			allocate();
		}
	}

	GPUArray(const size_t count, const T* host_p) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate();
		flsgpu::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
	}

	GPUArray(const size_t count, const T* host_p, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate(stream);
		flsgpu::memory::device_memcpy_h2d_async(device_ptr, host_p, memory_size, stream);
	}

	GPUArray(const size_t count, const size_t buffer, const T* host_p) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size + buffer * sizeof(T);
		allocate();
		flsgpu::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
	}

	GPUArray(const size_t count, const size_t buffer, const T* host_p, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size + buffer * sizeof(T);
		allocate(stream);
		flsgpu::memory::device_memcpy_h2d_async(device_ptr, host_p, memory_size, stream);
	}

	// Copy constructor
	GPUArray(const GPUArray&) = delete;
	// Assignment operator deleted
	GPUArray& operator=(const GPUArray&) = delete;

	// Move constructor
	GPUArray(GPUArray&& other) noexcept
	    : allocation_size(other.allocation_size)
	    , memory_size(other.memory_size)
	    , device_ptr(other.device_ptr) {
		other.allocation_size = 0;
		other.memory_size     = 0;
		other.device_ptr      = nullptr;
	}

	// Assignment operator
	GPUArray& operator=(GPUArray&& other) noexcept {
		if (this != &other) {
			free_device_pointer(device_ptr);
			allocation_size       = other.allocation_size;
			memory_size           = other.memory_size;
			device_ptr            = other.device_ptr;
			other.allocation_size = 0;
			other.memory_size     = 0;
			other.device_ptr      = nullptr;
		}
		return *this;
	}

	~GPUArray() {
		free_device_pointer(device_ptr);
	}

	void copy_to_host(T* host_p) {
		CUDA_SAFE_CALL(cudaMemcpy(host_p, device_ptr, memory_size, cudaMemcpyDeviceToHost));
	}

	T* get() {
		return device_ptr;
	}

	T* release() {
		auto temp  = device_ptr;
		device_ptr = nullptr;
		return temp;
	}
};

#endif // GPU_UTILS_H
