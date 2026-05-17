// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/gpu_array.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_GPU_ARRAY_CUH
#define GALP_MEMORY_GPU_ARRAY_CUH

#include "cuda/memory/cuda_macros.cuh"
#include "cuda/memory/device_pool.cuh"

#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
void free_device_pointer(T*& device_ptr) {
	if (device_ptr != nullptr) {
		galp::memory::device_free(device_ptr);
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
		device_ptr = reinterpret_cast<T*>(galp::memory::device_malloc(allocation_size));
	}
	void allocate(cudaStream_t stream) {
		device_ptr = reinterpret_cast<T*>(galp::memory::device_malloc_on_stream(allocation_size, stream));
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
		galp::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
	}

	GPUArray(const size_t count, const T* host_p, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate(stream);
		galp::memory::device_memcpy_h2d_async(device_ptr, host_p, memory_size, stream);
	}

	GPUArray(const size_t count, const size_t buffer, const T* host_p) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size + buffer * sizeof(T);
		allocate();
		galp::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
	}

	GPUArray(const size_t count, const size_t buffer, const T* host_p, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size + buffer * sizeof(T);
		allocate(stream);
		galp::memory::device_memcpy_h2d_async(device_ptr, host_p, memory_size, stream);
	}

	GPUArray(const GPUArray&)            = delete;
	GPUArray& operator=(const GPUArray&) = delete;

	GPUArray(GPUArray&& other) noexcept
	    : allocation_size(other.allocation_size)
	    , memory_size(other.memory_size)
	    , device_ptr(other.device_ptr) {
		other.allocation_size = 0;
		other.memory_size     = 0;
		other.device_ptr      = nullptr;
	}

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
		try {
			free_device_pointer(device_ptr);
		} catch (const std::exception& e) {
			std::fprintf(stderr, "GPUArray destructor: %s\n", e.what());
		}
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

#endif // GALP_MEMORY_GPU_ARRAY_CUH
