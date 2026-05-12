// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/memory/cuda_raii.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_MEMORY_CUDA_RAII_CUH
#define FLSGPU_MEMORY_CUDA_RAII_CUH

#include "flsgpu/memory/cuda_macros.cuh"
#include <cuda_runtime.h>
#include <utility>

namespace galp::memory {

class CudaStream {
public:
	CudaStream() = default;
	explicit CudaStream(const unsigned flags) {
		create(flags);
	}

	CudaStream(const CudaStream&)            = delete;
	CudaStream& operator=(const CudaStream&) = delete;

	CudaStream(CudaStream&& other) noexcept
	    : stream_(std::exchange(other.stream_, nullptr)) {
	}

	CudaStream& operator=(CudaStream&& other) noexcept {
		if (this != &other) {
			reset();
			stream_ = std::exchange(other.stream_, nullptr);
		}
		return *this;
	}

	~CudaStream() {
		reset();
	}

	void create(const unsigned flags) {
		if (stream_ == nullptr) {
			CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream_, flags));
		}
	}

	void reset(cudaStream_t stream = nullptr) noexcept {
		if (stream_ != nullptr) {
			CUDA_LOG_CALL(cudaStreamDestroy(stream_));
		}
		stream_ = stream;
	}

	cudaStream_t get() const noexcept {
		return stream_;
	}

	explicit operator bool() const noexcept {
		return stream_ != nullptr;
	}

private:
	cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
	CudaEvent() = default;
	explicit CudaEvent(const unsigned flags) {
		create_with_flags(flags);
	}

	CudaEvent(const CudaEvent&)            = delete;
	CudaEvent& operator=(const CudaEvent&) = delete;

	CudaEvent(CudaEvent&& other) noexcept
	    : event_(std::exchange(other.event_, nullptr)) {
	}

	CudaEvent& operator=(CudaEvent&& other) noexcept {
		if (this != &other) {
			reset();
			event_ = std::exchange(other.event_, nullptr);
		}
		return *this;
	}

	~CudaEvent() {
		reset();
	}

	void create() {
		if (event_ == nullptr) {
			CUDA_SAFE_CALL(cudaEventCreate(&event_));
		}
	}

	void create_with_flags(const unsigned flags) {
		if (event_ == nullptr) {
			CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event_, flags));
		}
	}

	void record(cudaStream_t stream = nullptr) {
		CUDA_SAFE_CALL(cudaEventRecord(event_, stream));
	}

	void synchronize() {
		CUDA_SAFE_CALL(cudaEventSynchronize(event_));
	}

	float elapsed_since(const CudaEvent& start) const {
		float ms = 0.0f;
		CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start.get(), event_));
		return ms;
	}

	void reset(cudaEvent_t event = nullptr) noexcept {
		if (event_ != nullptr) {
			CUDA_LOG_CALL(cudaEventDestroy(event_));
		}
		event_ = event;
	}

	cudaEvent_t get() const noexcept {
		return event_;
	}

	explicit operator bool() const noexcept {
		return event_ != nullptr;
	}

private:
	cudaEvent_t event_ = nullptr;
};

} // namespace galp::memory

#endif // FLSGPU_MEMORY_CUDA_RAII_CUH
