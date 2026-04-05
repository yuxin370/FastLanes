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
};

struct PendingCopyEvent {
	cudaEvent_t event  = nullptr;
	void*       pinned = nullptr;
};

struct PendingCopyBatchEvent {
	cudaEvent_t        event = nullptr;
	std::vector<void*> pinned_list;
};

struct BatchedCopy {
	void*       dst   = nullptr;
	const void* src   = nullptr;
	size_t      bytes = 0;
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

	void free(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = in_use_.find(ptr);
		if (it != in_use_.end()) {
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

	void copy_h2d(void* dst, const void* src, size_t bytes) {
		copy_h2d_on_stream(dst, src, bytes, stream_);
	}

	void copy_h2d_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		const auto use_stream = stream != nullptr ? stream : stream_;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (batch_active_ && use_stream == batch_stream_) {
				void*       pinned     = nullptr;
				const void* queued_src = src;
				if (use_pinned_ && bytes > small_copy_threshold_) {
					pinned     = alloc_pinned_locked(bytes);
					std::memcpy(pinned, src, bytes);
					queued_src = pinned;
				}
				batched_copies_.push_back(BatchedCopy {dst, queued_src, bytes, pinned});
				return;
			}
		}

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

	void begin_h2d_batch(cudaStream_t stream) {
		const auto use_stream = stream != nullptr ? stream : stream_;
		std::lock_guard<std::mutex> lock(mutex_);
		if (batch_active_) {
			throw std::runtime_error("DevicePool::begin_h2d_batch called while another batch is active");
		}
		batch_active_ = true;
		batch_stream_ = use_stream;
		batched_copies_.clear();
	}

	void flush_h2d_batch() {
		std::vector<BatchedCopy> copies;
		cudaStream_t             batch_stream = nullptr;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (!batch_active_) {
				return;
			}
			batch_stream = batch_stream_;
			copies       = std::move(batched_copies_);
			batched_copies_.clear();
			batch_active_ = false;
			batch_stream_ = nullptr;
		}

		for (const auto& copy : copies) {
			if (copy.bytes == 0) {
				continue;
			}
			CUDA_SAFE_CALL(cudaMemcpyAsync(copy.dst, copy.src, copy.bytes, cudaMemcpyHostToDevice, batch_stream));
		}

		if (!copies.empty()) {
			std::vector<void*> pinned_list;
			pinned_list.reserve(copies.size());
			for (auto& copy : copies) {
				if (copy.pinned != nullptr) {
					pinned_list.push_back(copy.pinned);
					copy.pinned = nullptr;
				}
			}
			cudaEvent_t event {};
			CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
			CUDA_SAFE_CALL(cudaEventRecord(event, batch_stream));
			std::lock_guard<std::mutex> lock(mutex_);
			pending_copy_batch_events_[stream_key(batch_stream)].push_back(
			    PendingCopyBatchEvent {event, std::move(pinned_list)});
		}
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
			for (const auto& [stream_id, events] : pending_copy_batch_events_) {
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
		for (auto& [stream_id, events] : pending_copy_batch_events_) {
			(void)stream_id;
			for (auto& pending : events) {
				if (pending.event != nullptr) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
				}
				for (void*& pinned : pending.pinned_list) {
					release_pinned_locked(pinned);
				}
				pending.pinned_list.clear();
			}
		}
		pending_copy_batch_events_.clear();
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

		auto bit = pending_copy_batch_events_.find(stream_key(src_stream));
		if (bit != pending_copy_batch_events_.end()) {
			for (auto& pending : bit->second) {
				if (pending.event != nullptr) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
				}
				for (void*& pinned : pending.pinned_list) {
					release_pinned_locked(pinned);
				}
				pending.pinned_list.clear();
			}
			pending_copy_batch_events_.erase(bit);
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
			(void)info;
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
		if (!pending_copy_events_.empty() || !pending_copy_batch_events_.empty() || batch_active_ ||
		    !batched_copies_.empty() || !in_use_.empty() || !pinned_in_use_.empty()) {
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
		for (auto it = pending_copy_batch_events_.begin(); it != pending_copy_batch_events_.end();) {
			auto& events = it->second;
			for (auto event_it = events.begin(); event_it != events.end();) {
				auto& pending = *event_it;
				if (pending.event == nullptr) {
					for (void*& pinned : pending.pinned_list) {
						release_pinned_locked(pinned);
					}
					pending.pinned_list.clear();
					event_it = events.erase(event_it);
					continue;
				}
				auto status = cudaEventQuery(pending.event);
				if (status == cudaSuccess) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
					for (void*& pinned : pending.pinned_list) {
						release_pinned_locked(pinned);
					}
					pending.pinned_list.clear();
					event_it = events.erase(event_it);
				} else if (status == cudaErrorNotReady) {
					++event_it;
				} else {
					CUDA_SAFE_CALL(status);
				}
			}
			if (events.empty()) {
				it = pending_copy_batch_events_.erase(it);
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
	bool         batch_active_         = false;
	cudaStream_t batch_stream_         = nullptr;
	std::vector<BatchedCopy> batched_copies_;

	std::unordered_map<size_t, std::vector<void*>>                                      free_sync_by_size_;
	std::unordered_map<StreamKey, std::unordered_map<size_t, std::vector<void*>>> free_async_by_stream_size_;
	std::unordered_map<void*, DeviceAllocInfo>     in_use_;

	std::unordered_map<size_t, std::vector<void*>> pinned_free_by_size_;
	std::unordered_map<void*, size_t>              pinned_in_use_;
	std::unordered_map<StreamKey, std::vector<PendingCopyEvent>> pending_copy_events_;
	std::unordered_map<StreamKey, std::vector<PendingCopyBatchEvent>> pending_copy_batch_events_;
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

inline void begin_h2d_batch(cudaStream_t stream = nullptr) {
	DevicePool::instance().begin_h2d_batch(stream);
}

inline void flush_h2d_batch() {
	DevicePool::instance().flush_h2d_batch();
}

class BatchUploader {
public:
	explicit BatchUploader(cudaStream_t stream = nullptr) {
		DevicePool::instance().begin_h2d_batch(stream);
		active_ = true;
	}

	BatchUploader(const BatchUploader&)            = delete;
	BatchUploader& operator=(const BatchUploader&) = delete;

	BatchUploader(BatchUploader&& other) noexcept
	    : active_(other.active_) {
		other.active_ = false;
	}

	BatchUploader& operator=(BatchUploader&& other) noexcept {
		if (this != &other) {
			if (active_) {
				DevicePool::instance().flush_h2d_batch();
			}
			active_       = other.active_;
			other.active_ = false;
		}
		return *this;
	}

	~BatchUploader() {
		if (active_) {
			DevicePool::instance().flush_h2d_batch();
		}
	}

	void flush() {
		if (!active_) {
			return;
		}
		DevicePool::instance().flush_h2d_batch();
		active_ = false;
	}

private:
	bool active_ = false;
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
