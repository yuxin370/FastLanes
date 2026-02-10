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
	size_t size        = 0;
	bool   async_alloc = false;
};

class DevicePool {
public:
	static DevicePool& instance() {
		static DevicePool pool;
		return pool;
	}

	void* alloc(size_t bytes) {
		if (bytes == 0) {
			return nullptr;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		if (enabled_) {
			auto& free_list = free_by_size_[bytes];
			if (!free_list.empty()) {
				void* ptr = free_list.back();
				free_list.pop_back();
				in_use_[ptr] = DeviceAllocInfo {bytes, use_async_};
				return ptr;
			}
		}

		void* ptr         = nullptr;
		bool  async_alloc = false;
		if (use_async_) {
			auto status = cudaMallocAsync(&ptr, bytes, stream_);
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
		in_use_[ptr] = DeviceAllocInfo {bytes, async_alloc};
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
				free_by_size_[info.size].push_back(ptr);
				return;
			}
			if (info.async_alloc) {
				CUDA_SAFE_CALL(cudaFreeAsync(ptr, stream_));
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

	void free_pinned(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = pinned_in_use_.find(ptr);
		if (it != pinned_in_use_.end()) {
			const size_t bytes = it->second;
			pinned_in_use_.erase(it);
			if (use_pinned_) {
				pinned_free_by_size_[bytes].push_back(ptr);
				return;
			}
			CUDA_SAFE_CALL(cudaFreeHost(ptr));
			return;
		}
		CUDA_SAFE_CALL(cudaFreeHost(ptr));
	}

	void copy_h2d(void* dst, const void* src, size_t bytes) {
		if (bytes == 0) {
			return;
		}
		if (!use_pinned_ || bytes <= small_copy_threshold_) {
			CUDA_SAFE_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
			return;
		}
		void* pinned = alloc_pinned(bytes);
		std::memcpy(pinned, src, bytes);
		CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice, stream_));
		pending_pinned_.push_back(pinned);
	}

	void sync_h2d() {
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
		for (void* ptr : pending_pinned_) {
			free_pinned(ptr);
		}
		pending_pinned_.clear();
	}

	cudaStream_t stream() const {
		return stream_;
	}

	void set_enabled(bool enabled) {
		enabled_ = enabled;
	}
	void set_use_async(bool use_async) {
		use_async_ = use_async;
	}
	void set_use_pinned(bool use_pinned) {
		use_pinned_ = use_pinned;
	}
	void set_small_copy_threshold(size_t bytes) {
		small_copy_threshold_ = bytes;
	}

	~DevicePool() {
		for (auto& [size, list] : free_by_size_) {
			for (void* ptr : list) {
				cudaFree(ptr);
			}
		}
		for (auto& [size, list] : pinned_free_by_size_) {
			for (void* ptr : list) {
				cudaFreeHost(ptr);
			}
		}
	}

private:
	DevicePool() {
		stream_ = 0;
	}

	std::mutex   mutex_;
	bool         enabled_              = true;
	bool         use_async_            = true;
	bool         use_pinned_           = true;
	size_t       small_copy_threshold_ = 256 * 1024;
	cudaStream_t stream_               = nullptr;

	std::unordered_map<size_t, std::vector<void*>> free_by_size_;
	std::unordered_map<void*, DeviceAllocInfo>     in_use_;

	std::unordered_map<size_t, std::vector<void*>> pinned_free_by_size_;
	std::unordered_map<void*, size_t>              pinned_in_use_;
	std::vector<void*>                             pending_pinned_;
};

inline void* device_malloc(size_t bytes) {
	return DevicePool::instance().alloc(bytes);
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

public:
	GPUArray(const size_t count) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate();
	}

	GPUArray(const size_t count, const T* host_p) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate();
		flsgpu::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
	}

	GPUArray(const size_t count, const size_t buffer, const T* host_p) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size + buffer * sizeof(T);
		allocate();
		flsgpu::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
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
