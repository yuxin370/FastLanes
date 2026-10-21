// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/cuda/cuda_macros.cuh
// ────────────────────────────────────────────────────────
#ifndef FASTLENS_CUDA_CUDA_MACROS_CUH
#define FASTLENS_CUDA_CUDA_MACROS_CUH

#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace fastlens::memory {

// Exception type raised by CUDA_SAFE_CALL when a CUDA Runtime API call fails.
// Carries the raw cudaError_t so callers can distinguish recoverable
// (e.g. cudaErrorMemoryAllocation) from fatal errors.
class CudaError : public std::runtime_error {
public:
	CudaError(cudaError_t code, const char* expr, const char* file, int line)
	    : std::runtime_error(build_message(code, expr, file, line))
	    , code_(code) {}

	cudaError_t code() const noexcept { return code_; }

private:
	static std::string build_message(cudaError_t code, const char* expr, const char* file, int line) {
		std::string msg = "CUDA error ";
		msg += cudaGetErrorString(code);
		msg += " while evaluating `";
		msg += expr;
		msg += "` at ";
		msg += file;
		msg += ":";
		msg += std::to_string(line);
		return msg;
	}

	cudaError_t code_;
};

} // namespace fastlens::memory

// CUDA_SAFE_CALL throws fastlens::memory::CudaError on failure.
// Use CUDA_LOG_CALL in destructors and other noexcept cleanup paths.
#define CUDA_SAFE_CALL(call)                                                                                           \
	do {                                                                                                               \
		cudaError_t _fastlens_cuda_err = (call);                                                                           \
		if (_fastlens_cuda_err != cudaSuccess) {                                                                           \
			throw ::fastlens::memory::CudaError(_fastlens_cuda_err, #call, __FILE__, __LINE__);                                      \
		}                                                                                                              \
	} while (0)

// CUDA_LOG_CALL logs failures to stderr without throwing.
#define CUDA_LOG_CALL(call)                                                                                            \
	do {                                                                                                               \
		cudaError_t _fastlens_cuda_err = (call);                                                                           \
		if (_fastlens_cuda_err != cudaSuccess) {                                                                           \
			std::fprintf(stderr, "CUDA error in file '%s' at line %d: %s (call: %s)\n", __FILE__, __LINE__,            \
			             cudaGetErrorString(_fastlens_cuda_err), #call);                                                   \
		}                                                                                                              \
	} while (0)

#endif // FASTLENS_CUDA_CUDA_MACROS_CUH
