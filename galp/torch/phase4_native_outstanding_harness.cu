#include <cuda/atomic>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace py = pybind11;

namespace {

struct GateState final {
	uint32_t release = 0U;
	uint32_t status  = 0U; // 0=waiting, 1=host-released, 2=device-watchdog.
};

void check_cuda(const cudaError_t status, const char* const operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(
		    std::string(operation) + " failed: " + cudaGetErrorString(status));
	}
}

__global__ void wait_for_host_release(
    GateState* const gate,
    const uint64_t watchdog_cycles) {
	cuda::atomic_ref<uint32_t, cuda::thread_scope_system> release(gate->release);
	cuda::atomic_ref<uint32_t, cuda::thread_scope_system> status(gate->status);
	const uint64_t started = clock64();
	while (release.load(cuda::memory_order_acquire) == 0U) {
		if (clock64() - started >= watchdog_cycles) {
			status.store(2U, cuda::memory_order_release);
			return;
		}
		__nanosleep(1000U);
	}
	status.store(1U, cuda::memory_order_release);
}

__global__ void read_first_float(const float* const input, float* const output) {
	if (blockIdx.x == 0U && threadIdx.x == 0U) {
		output[0] = input[0];
	}
}

class OutstandingConsumerHandle final {
public:
	OutstandingConsumerHandle(
	    const int cuda_device,
	    cudaStream_t consumer_stream,
	    GateState* gate,
	    float* checksum,
	    cudaEvent_t completion) noexcept
	    : cuda_device_(cuda_device)
	    , consumer_stream_(consumer_stream)
	    , gate_(gate)
	    , checksum_(checksum)
	    , completion_(completion) {
	}

	~OutstandingConsumerHandle() {
		cleanup_noexcept();
	}

	OutstandingConsumerHandle(const OutstandingConsumerHandle&) = delete;
	OutstandingConsumerHandle& operator=(const OutstandingConsumerHandle&) = delete;

	[[nodiscard]] bool consumer_complete() const {
		std::lock_guard lock(mutex_);
		if (closed_) {
			return true;
		}
		check_cuda(cudaSetDevice(cuda_device_), "cudaSetDevice(test query)");
		const auto status = cudaEventQuery(completion_);
		if (status == cudaSuccess) {
			return true;
		}
		if (status == cudaErrorNotReady) {
			static_cast<void>(cudaGetLastError());
			return false;
		}
		check_cuda(status, "cudaEventQuery(test consumer)");
		return false;
	}

	[[nodiscard]] bool gate_timed_out() const noexcept {
		return gate_status() == 2U;
	}

	[[nodiscard]] uint32_t gate_status() const noexcept {
		if (gate_ == nullptr) {
			return 1U;
		}
		return std::atomic_ref<uint32_t>(gate_->status).load(std::memory_order_acquire);
	}

	void release_gate() noexcept {
		if (gate_ != nullptr) {
			std::atomic_ref<uint32_t>(gate_->release).store(1U, std::memory_order_release);
		}
	}

	bool wait_consumer(const int timeout_milliseconds) const {
		if (timeout_milliseconds <= 0) {
			throw std::invalid_argument("consumer wait timeout must be positive");
		}
		const auto deadline = std::chrono::steady_clock::now() +
		                      std::chrono::milliseconds(timeout_milliseconds);
		while (std::chrono::steady_clock::now() < deadline) {
			if (consumer_complete()) {
				return true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		return consumer_complete();
	}

	float consumer_checksum() const {
		if (!consumer_complete()) {
			throw std::logic_error("consumer checksum requested before completion");
		}
		std::lock_guard lock(mutex_);
		if (closed_ || checksum_ == nullptr) {
			throw std::logic_error("consumer checksum requested after cleanup");
		}
		check_cuda(cudaSetDevice(cuda_device_), "cudaSetDevice(test checksum)");
		float value = 0.0F;
		check_cuda(
		    cudaMemcpy(&value, checksum_, sizeof(value), cudaMemcpyDeviceToHost),
		    "cudaMemcpy(test checksum)");
		return value;
	}

	void cleanup(const int timeout_milliseconds) {
		release_gate();
		if (!wait_consumer(timeout_milliseconds)) {
			throw std::runtime_error("test consumer did not complete before cleanup timeout");
		}
		std::lock_guard lock(mutex_);
		if (closed_) {
			return;
		}
		check_cuda(cudaSetDevice(cuda_device_), "cudaSetDevice(test cleanup)");
		check_cuda(cudaEventDestroy(completion_), "cudaEventDestroy(test completion)");
		check_cuda(cudaFree(checksum_), "cudaFree(test checksum)");
		check_cuda(cudaFreeHost(gate_), "cudaFreeHost(test gate)");
		completion_ = nullptr;
		checksum_   = nullptr;
		gate_       = nullptr;
		closed_     = true;
	}

private:
	void cleanup_noexcept() noexcept {
		try {
			cleanup(7000);
		} catch (...) {
			// The device gate has its own watchdog. If the CUDA context is
			// already unhealthy, leaking test-only resources is safer than
			// freeing memory still referenced by a kernel.
		}
	}

	int          cuda_device_ = -1;
	cudaStream_t consumer_stream_ = nullptr;
	GateState*   gate_ = nullptr;
	float*       checksum_ = nullptr;
	cudaEvent_t  completion_ = nullptr;
	mutable std::mutex mutex_;
	bool               closed_ = false;
};

std::shared_ptr<OutstandingConsumerHandle> arm_outstanding_consumer(
    const uintptr_t input_pointer,
    const uintptr_t consumer_stream_identity,
    const int cuda_device,
    const int watchdog_milliseconds) {
	if (input_pointer == 0U || consumer_stream_identity == 0U) {
		throw std::invalid_argument("test consumer requires non-null input and stream identities");
	}
	if (cuda_device < 0 || watchdog_milliseconds <= 0) {
		throw std::invalid_argument("test consumer requires a valid device and watchdog timeout");
	}
	check_cuda(cudaSetDevice(cuda_device), "cudaSetDevice(test arm)");
	GateState* gate = nullptr;
	float* checksum = nullptr;
	cudaEvent_t completion = nullptr;
	bool gate_launched = false;
	check_cuda(
	    cudaHostAlloc(reinterpret_cast<void**>(&gate), sizeof(GateState), cudaHostAllocMapped),
	    "cudaHostAllocMapped(test gate)");
	gate->release = 0U;
	gate->status  = 0U;
	try {
		void* device_gate = nullptr;
		check_cuda(cudaHostGetDevicePointer(&device_gate, gate, 0U), "cudaHostGetDevicePointer(test gate)");
		check_cuda(cudaMalloc(reinterpret_cast<void**>(&checksum), sizeof(float)), "cudaMalloc(test checksum)");
		check_cuda(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming), "cudaEventCreate(test completion)");
		int clock_rate_khz = 0;
		check_cuda(
		    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, cuda_device),
		    "cudaDeviceGetAttribute(clock rate)");
		const uint64_t watchdog_cycles =
		    static_cast<uint64_t>(clock_rate_khz) * static_cast<uint64_t>(watchdog_milliseconds);
		const auto stream = reinterpret_cast<cudaStream_t>(consumer_stream_identity);
		wait_for_host_release<<<1, 1, 0, stream>>>(static_cast<GateState*>(device_gate), watchdog_cycles);
		check_cuda(cudaGetLastError(), "wait_for_host_release launch");
		gate_launched = true;
		read_first_float<<<1, 1, 0, stream>>>(
		    reinterpret_cast<const float*>(input_pointer), checksum);
		check_cuda(cudaGetLastError(), "read_first_float launch");
		check_cuda(cudaEventRecord(completion, stream), "cudaEventRecord(test completion)");
		return std::make_shared<OutstandingConsumerHandle>(
		    cuda_device, stream, gate, checksum, completion);
	} catch (...) {
		std::atomic_ref<uint32_t>(gate->release).store(1U, std::memory_order_release);
		// Never call a potentially synchronizing free after publishing device
		// work without a completion marker. The host release plus device
		// watchdog make the stream finite; leaking private test resources on
		// this exceptional path is safer than hanging the CUDA context.
		if (gate_launched) {
			throw;
		}
		if (completion != nullptr) {
			static_cast<void>(cudaEventDestroy(completion));
		}
		if (checksum != nullptr) {
			static_cast<void>(cudaFree(checksum));
		}
		static_cast<void>(cudaFreeHost(gate));
		throw;
	}
}

} // namespace

PYBIND11_MODULE(_galp_phase4_outstanding_test, binding) {
	binding.doc() = "Private Phase-4A native outstanding-consumer harness";
	py::class_<OutstandingConsumerHandle, std::shared_ptr<OutstandingConsumerHandle>>(
	    binding, "OutstandingConsumerHandle")
	    .def_property_readonly("consumer_complete", &OutstandingConsumerHandle::consumer_complete)
	    .def_property_readonly("gate_status", &OutstandingConsumerHandle::gate_status)
	    .def_property_readonly("gate_timed_out", &OutstandingConsumerHandle::gate_timed_out)
	    .def("release_gate", &OutstandingConsumerHandle::release_gate)
	    .def(
	        "wait_consumer",
	        &OutstandingConsumerHandle::wait_consumer,
	        py::arg("timeout_milliseconds") = 7000,
	        py::call_guard<py::gil_scoped_release>())
	    .def_property_readonly("consumer_checksum", &OutstandingConsumerHandle::consumer_checksum)
	    .def(
	        "cleanup",
	        &OutstandingConsumerHandle::cleanup,
	        py::arg("timeout_milliseconds") = 7000,
	        py::call_guard<py::gil_scoped_release>());
	binding.def(
	    "arm_outstanding_consumer",
	    &arm_outstanding_consumer,
	    py::arg("input_pointer"),
	    py::arg("consumer_stream_identity"),
	    py::arg("cuda_device"),
	    py::arg("watchdog_milliseconds") = 5000);
}
