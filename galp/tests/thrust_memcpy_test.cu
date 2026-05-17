// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tests/thrust_memcpy_test.cu
// ────────────────────────────────────────────────────────
// tests/test_thrust_memcpy.cu

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

namespace {
std::string cuda_unavailable_reason() {
	int        device_count = 0;
	const auto status       = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) {
		return std::string("CUDA device not available for Thrust test: ") + cudaGetErrorString(status);
	}
	if (device_count <= 0) {
		return "CUDA device not available for Thrust test: device count is zero";
	}
	return {};
}
} // namespace

TEST(ThrustMemcpy, HostToDeviceAndBack) {
	if (const auto reason = cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}

	constexpr std::size_t N = 1 << 12; // 4096 ints

	// 1) Host data: 0,1,2,…,N-1
	thrust::host_vector<int> h_src(N);
	thrust::sequence(h_src.begin(), h_src.end());

	// 2) Copy to device (H→D constructor)
	thrust::device_vector<int> d_buf = h_src;

	// 3) Copy back to host (D→H constructor)
	thrust::host_vector<int> h_dst = d_buf;

	// 4) Verify equality with a single call
	bool same = thrust::equal(h_src.begin(), h_src.end(), h_dst.begin());
	EXPECT_TRUE(same) << "Data changed after GPU round-trip";
}
