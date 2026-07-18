#include "codecs/device_ops/unsumer.cuh"
#include "cuda/memory/gpu_array.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

namespace {

std::string delta_cuda_unavailable_reason() {
	int        device_count = 0;
	const auto status       = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) {
		return std::string("CUDA device not available for DELTA test: ") + cudaGetErrorString(status);
	}
	return device_count > 0 ? std::string {} : "CUDA device not available for DELTA test: device count is zero";
}

template <typename UIntT>
struct DeltaUnsumerTestColumn {
	const UIntT* rsum_bases;
};

template <typename UIntT, unsigned UNPACK_N_VECTORS>
struct DeltaUnsumerTestUnpacker {
	const UIntT* deltas;
	unsigned     cursor = 0;

	__device__ __forceinline__ void unpack_next_into(UIntT* __restrict out) {
#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			out[vector] = deltas[cursor * UNPACK_N_VECTORS + vector];
		}
		++cursor;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
__global__ void run_delta_unsumer_test(
    const typename galp::codec::utils::same_width_uint<T>::type* bases,
    const typename galp::codec::utils::same_width_uint<T>::type* deltas,
    T*                                                           out) {
	if (blockIdx.x != 0 || threadIdx.x != 0) {
		return;
	}
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	DeltaUnsumerTestColumn<UIntT>                    column {bases};
	DeltaUnsumerTestUnpacker<UIntT, UNPACK_N_VECTORS> unpacker {deltas};
	galp::codec::device::DeltaUnsumer<T, UNPACK_N_VECTORS> unsumer(column, 0, 0);
	for (unsigned position = 0; position < galp::codec::utils::get_values_per_lane<T>(); ++position) {
		unsumer.unsum_next_into(unpacker, out + position * UNPACK_N_VECTORS);
	}
}

template <typename T>
T signed_bits(const typename galp::codec::utils::same_width_uint<T>::type bits) {
	T result {};
	std::memcpy(&result, &bits, sizeof(T));
	return result;
}

template <typename T, unsigned UNPACK_N_VECTORS>
std::vector<T> run_delta_unsumer(
    const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& vector_bases,
    const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& deltas) {
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr size_t positions = galp::codec::utils::get_values_per_lane<T>();
	constexpr size_t lanes     = galp::codec::utils::get_n_lanes<T>();
	std::vector<UIntT> bases(UNPACK_N_VECTORS * lanes, UIntT {0});
	for (size_t vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
		bases[vector * lanes] = vector_bases[vector];
	}
	GPUArray<UIntT> device_bases(bases.size(), bases.data());
	GPUArray<UIntT> device_deltas(deltas.size(), deltas.data());
	GPUArray<T>     device_out(positions * UNPACK_N_VECTORS);
	run_delta_unsumer_test<T, UNPACK_N_VECTORS>
	    <<<1, 1>>>(device_bases.get(), device_deltas.get(), device_out.get());
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	std::vector<T> output(positions * UNPACK_N_VECTORS);
	device_out.copy_to_host(output.data());
	return output;
}

template <typename T, unsigned UNPACK_N_VECTORS>
std::vector<T> expected_delta_unsum(
    const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& vector_bases,
    const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& deltas) {
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr unsigned positions = galp::codec::utils::get_values_per_lane<T>();
	std::vector<T> output(positions * UNPACK_N_VECTORS);
	for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
		UIntT prefix = vector_bases[vector];
		for (unsigned logical_position = 0; logical_position < positions; ++logical_position) {
			unsigned physical_position = logical_position;
			if constexpr (std::is_same_v<T, int16_t>) {
				physical_position = logical_position < 8U ? logical_position * 2U
				                                                 : (logical_position - 8U) * 2U + 1U;
			}
			prefix = static_cast<UIntT>(
			    prefix + deltas[physical_position * UNPACK_N_VECTORS + vector]);
			output[physical_position * UNPACK_N_VECTORS + vector] = signed_bits<T>(prefix);
		}
	}
	return output;
}

} // namespace

TEST(DeltaUnsumer, I8Prefix) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned vectors = 2;
	std::vector<uint8_t> bases {10U, 90U};
	std::vector<uint8_t> deltas(galp::codec::utils::get_values_per_lane<int8_t>() * vectors);
	for (size_t position = 0; position < galp::codec::utils::get_values_per_lane<int8_t>(); ++position) {
		deltas[position * vectors]      = static_cast<uint8_t>(position + 1U);
		deltas[position * vectors + 1U] = static_cast<uint8_t>(2U * position + 3U);
	}
	EXPECT_EQ((run_delta_unsumer<int8_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int8_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, I8Wraparound) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned vectors = 1;
	std::vector<uint8_t> bases {250U};
	std::vector<uint8_t> deltas {10U, 250U, 17U, 240U, 32U, 225U, 64U, 193U};
	EXPECT_EQ((run_delta_unsumer<int8_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int8_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, I16Reorder) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned vectors = 2;
	std::vector<uint16_t> bases {100U, 2000U};
	std::vector<uint16_t> deltas(galp::codec::utils::get_values_per_lane<int16_t>() * vectors);
	for (size_t position = 0; position < galp::codec::utils::get_values_per_lane<int16_t>(); ++position) {
		deltas[position * vectors]      = static_cast<uint16_t>(position + 1U);
		deltas[position * vectors + 1U] = static_cast<uint16_t>(3U * position + 2U);
	}
	EXPECT_EQ((run_delta_unsumer<int16_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int16_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, I16Wraparound) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned vectors = 1;
	std::vector<uint16_t> bases {65530U};
	std::vector<uint16_t> deltas {
	    10U, 65500U, 73U, 65400U, 211U, 65000U, 997U, 64000U,
	    4093U, 60000U, 8191U, 50000U, 16381U, 40000U, 32749U, 30000U};
	EXPECT_EQ((run_delta_unsumer<int16_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int16_t, vectors>(bases, deltas)));
}
