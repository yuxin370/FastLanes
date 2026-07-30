#include "codecs/device_ops/functors.cuh"
#include "codecs/device_ops/unsumer.cuh"
#include "codecs/device_ops/unpackers.cuh"
#include "cuda/memory/gpu_array.cuh"
#include "engine/unpack_dispatch.cuh"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
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
__global__ void run_delta_unsumer_test(const typename galp::codec::utils::same_width_uint<T>::type* bases,
                                       const typename galp::codec::utils::same_width_uint<T>::type* deltas,
                                       T*                                                           out) {
	if (blockIdx.x != 0 || threadIdx.x != 0) {
		return;
	}
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	DeltaUnsumerTestColumn<UIntT>                          column {bases};
	DeltaUnsumerTestUnpacker<UIntT, UNPACK_N_VECTORS>      unpacker {deltas};
	galp::codec::device::DeltaUnsumer<T, UNPACK_N_VECTORS> unsumer(column, 0, 0);
	for (unsigned position = 0; position < galp::codec::utils::get_values_per_lane<T>(); ++position) {
		unsumer.unsum_next_into(unpacker, out + position * UNPACK_N_VECTORS);
	}
}

template <typename T, unsigned UNPACK_N_VECTORS>
__global__ void run_delta_register_unsumer_test(const typename galp::codec::utils::same_width_uint<T>::type* bases,
                                                const typename galp::codec::utils::same_width_uint<T>::type* deltas,
                                                T*                                                           out) {
	if (blockIdx.x != 0 || threadIdx.x != 0) {
		return;
	}
	using UIntT                             = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr unsigned            positions = galp::codec::utils::get_values_per_lane<T>();
	DeltaUnsumerTestColumn<UIntT> column {bases};
	UIntT                         values[UNPACK_N_VECTORS * positions];
#pragma unroll
	for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
#pragma unroll
		for (unsigned position = 0; position < positions; ++position) {
			values[vector * positions + position] = deltas[position * UNPACK_N_VECTORS + vector];
		}
	}
	galp::codec::device::DeltaRegisterUnsumer<T, UNPACK_N_VECTORS> unsumer(column, 0, 0);
	unsumer.unsum_inplace(values);
#pragma unroll
	for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
#pragma unroll
		for (unsigned position = 0; position < positions; ++position) {
			out[position * UNPACK_N_VECTORS + vector] = static_cast<T>(values[vector * positions + position]);
		}
	}
}

template <typename UIntT, unsigned UNPACK_N_VECTORS>
__global__ void run_delta_lane_tile_unpacker_test(const UIntT*  packed,
	                                               const uint32_t* vector_offsets,
	                                               const vbw_t*    bit_widths,
	                                               UIntT*          out) {
	if (blockIdx.x != 0 || threadIdx.x != 0) {
		return;
	}
	constexpr unsigned positions = galp::codec::utils::get_values_per_lane<UIntT>();
	using UnpackerT = galp::codec::device::BitUnpackerLaneTile<
	    UIntT, UNPACK_N_VECTORS, positions, galp::codec::device::BPFunctor<UIntT>>;
	UnpackerT unpacker(packed,
	                   vector_offsets,
	                   bit_widths,
	                   0,
	                   0,
	                   galp::codec::device::BPFunctor<UIntT> {});
	unpacker.unpack_next_into(out);
}

template <typename T>
T signed_bits(const typename galp::codec::utils::same_width_uint<T>::type bits) {
	T result {};
	std::memcpy(&result, &bits, sizeof(T));
	return result;
}

template <typename T, unsigned UNPACK_N_VECTORS>
std::vector<T> run_delta_unsumer(const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& vector_bases,
                                 const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& deltas) {
	using UIntT                  = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr size_t   positions = galp::codec::utils::get_values_per_lane<T>();
	constexpr size_t   lanes     = galp::codec::utils::get_n_lanes<T>();
	std::vector<UIntT> bases(UNPACK_N_VECTORS * lanes, UIntT {0});
	for (size_t vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
		bases[vector * lanes] = vector_bases[vector];
	}
	GPUArray<UIntT> device_bases(bases.size(), bases.data());
	GPUArray<UIntT> device_deltas(deltas.size(), deltas.data());
	GPUArray<T>     device_out(positions * UNPACK_N_VECTORS);
	run_delta_unsumer_test<T, UNPACK_N_VECTORS><<<1, 1>>>(device_bases.get(), device_deltas.get(), device_out.get());
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	std::vector<T> output(positions * UNPACK_N_VECTORS);
	device_out.copy_to_host(output.data());
	return output;
}

template <typename T, unsigned UNPACK_N_VECTORS>
std::vector<T>
run_delta_register_unsumer(const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& vector_bases,
                           const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& deltas) {
	using UIntT                  = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr size_t   positions = galp::codec::utils::get_values_per_lane<T>();
	constexpr size_t   lanes     = galp::codec::utils::get_n_lanes<T>();
	std::vector<UIntT> bases(UNPACK_N_VECTORS * lanes, UIntT {0});
	for (size_t vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
		bases[vector * lanes] = vector_bases[vector];
	}
	GPUArray<UIntT> device_bases(bases.size(), bases.data());
	GPUArray<UIntT> device_deltas(deltas.size(), deltas.data());
	GPUArray<T>     device_out(positions * UNPACK_N_VECTORS);
	run_delta_register_unsumer_test<T, UNPACK_N_VECTORS>
	    <<<1, 1>>>(device_bases.get(), device_deltas.get(), device_out.get());
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	std::vector<T> output(positions * UNPACK_N_VECTORS);
	device_out.copy_to_host(output.data());
	return output;
}

template <typename T, unsigned UNPACK_N_VECTORS>
std::vector<T>
expected_delta_unsum(const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& vector_bases,
                     const std::vector<typename galp::codec::utils::same_width_uint<T>::type>& deltas) {
	using UIntT                  = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr unsigned positions = galp::codec::utils::get_values_per_lane<T>();
	std::vector<T>     output(positions * UNPACK_N_VECTORS);
	for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
		UIntT prefix = vector_bases[vector];
		for (unsigned logical_position = 0; logical_position < positions; ++logical_position) {
			unsigned physical_position = logical_position;
			if constexpr (std::is_same_v<T, int16_t>) {
				physical_position = logical_position < 8U ? logical_position * 2U : (logical_position - 8U) * 2U + 1U;
			}
			prefix = static_cast<UIntT>(prefix + deltas[physical_position * UNPACK_N_VECTORS + vector]);
			output[physical_position * UNPACK_N_VECTORS + vector] = signed_bits<T>(prefix);
		}
	}
	return output;
}

template <typename UIntT, unsigned UNPACK_N_VECTORS>
std::vector<UIntT> run_delta_lane_tile_unpacker(const std::vector<vbw_t>& bit_widths) {
	constexpr unsigned positions = galp::codec::utils::get_values_per_lane<UIntT>();
	constexpr unsigned lanes     = galp::codec::utils::get_n_lanes<UIntT>();
	constexpr unsigned type_bits = galp::codec::utils::sizeof_in_bits<UIntT>();
	static_assert(UNPACK_N_VECTORS > 0U);
	if (bit_widths.size() != UNPACK_N_VECTORS) {
		throw std::invalid_argument("lane-tile test bit-width count mismatch");
	}

	std::vector<uint32_t> offsets(UNPACK_N_VECTORS);
	size_t                packed_size = 0;
	for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
		offsets[vector] = static_cast<uint32_t>(packed_size);
		packed_size += static_cast<size_t>(bit_widths[vector]) * lanes;
	}
	std::vector<UIntT> packed(std::max<size_t>(packed_size, 1U), UIntT {0});
	std::vector<UIntT> expected(UNPACK_N_VECTORS * positions);

	for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
		const unsigned bit_width = bit_widths[vector];
		const UIntT mask = galp::codec::utils::set_first_n_bits<UIntT>(static_cast<int32_t>(bit_width));
		for (unsigned position = 0; position < positions; ++position) {
			const UIntT value = static_cast<UIntT>((position * 37U + vector * 53U + 5U) & mask);
			expected[vector * positions + position] = value;
			if (bit_width == 0U) {
				continue;
			}
			const unsigned bit_position = position * bit_width;
			const unsigned line         = bit_position / type_bits;
			const unsigned offset       = bit_position % type_bits;
			packed[offsets[vector] + line * lanes] = static_cast<UIntT>(
			    packed[offsets[vector] + line * lanes] | static_cast<UIntT>(value << offset));
			if (offset + bit_width > type_bits) {
				packed[offsets[vector] + (line + 1U) * lanes] = static_cast<UIntT>(
				    packed[offsets[vector] + (line + 1U) * lanes] |
				    static_cast<UIntT>(value >> (type_bits - offset)));
			}
		}
	}

	GPUArray<UIntT>  device_packed(packed.size(), packed.data());
	GPUArray<uint32_t> device_offsets(offsets.size(), offsets.data());
	GPUArray<vbw_t>  device_widths(bit_widths.size(), bit_widths.data());
	GPUArray<UIntT>  device_out(expected.size());
	run_delta_lane_tile_unpacker_test<UIntT, UNPACK_N_VECTORS>
	    <<<1, 1>>>(device_packed.get(), device_offsets.get(), device_widths.get(), device_out.get());
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	std::vector<UIntT> output(expected.size());
	device_out.copy_to_host(output.data());
	EXPECT_EQ(output, expected);
	return output;
}

} // namespace

TEST(DeltaConfig, AutoDefaultsToRegisterAndKeepsExplicitFallback) {
	galp::execution::ExecutionConfig config {};
	EXPECT_EQ(config.delta_decoder, galp::execution::DeltaDecoder::Auto);
	EXPECT_EQ(galp::execution::resolve_delta_decoder(config.delta_decoder),
	          galp::execution::DeltaDecoder::Register);
	EXPECT_TRUE(galp::runtime::with_delta_decoder(config, [](auto decoder) {
		return decltype(decoder)::value == galp::execution::DeltaDecoder::Register;
	}));

	config.delta_decoder = galp::execution::DeltaDecoder::Stateful;
	EXPECT_FALSE(galp::execution::delta_uses_register(config.delta_decoder));
	EXPECT_TRUE(galp::runtime::with_delta_decoder(config, [](auto decoder) {
		return decltype(decoder)::value == galp::execution::DeltaDecoder::Stateful;
	}));
}

TEST(DeltaUnsumer, I8Prefix) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned   vectors = 2;
	std::vector<uint8_t> bases {10U, 90U};
	std::vector<uint8_t> deltas(galp::codec::utils::get_values_per_lane<int8_t>() * vectors);
	for (size_t position = 0; position < galp::codec::utils::get_values_per_lane<int8_t>(); ++position) {
		deltas[position * vectors]      = static_cast<uint8_t>(position + 1U);
		deltas[position * vectors + 1U] = static_cast<uint8_t>(2U * position + 3U);
	}
	EXPECT_EQ((run_delta_unsumer<int8_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int8_t, vectors>(bases, deltas)));
	EXPECT_EQ((run_delta_register_unsumer<int8_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int8_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, I8Wraparound) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned   vectors = 1;
	std::vector<uint8_t> bases {250U};
	std::vector<uint8_t> deltas {10U, 250U, 17U, 240U, 32U, 225U, 64U, 193U};
	EXPECT_EQ((run_delta_unsumer<int8_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int8_t, vectors>(bases, deltas)));
	EXPECT_EQ((run_delta_register_unsumer<int8_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int8_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, I16Reorder) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned    vectors = 2;
	std::vector<uint16_t> bases {100U, 2000U};
	std::vector<uint16_t> deltas(galp::codec::utils::get_values_per_lane<int16_t>() * vectors);
	for (size_t position = 0; position < galp::codec::utils::get_values_per_lane<int16_t>(); ++position) {
		deltas[position * vectors]      = static_cast<uint16_t>(position + 1U);
		deltas[position * vectors + 1U] = static_cast<uint16_t>(3U * position + 2U);
	}
	EXPECT_EQ((run_delta_unsumer<int16_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int16_t, vectors>(bases, deltas)));
	EXPECT_EQ((run_delta_register_unsumer<int16_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int16_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, I16Wraparound) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned    vectors = 1;
	std::vector<uint16_t> bases {65530U};
	std::vector<uint16_t> deltas {10U,
	                              65500U,
	                              73U,
	                              65400U,
	                              211U,
	                              65000U,
	                              997U,
	                              64000U,
	                              4093U,
	                              60000U,
	                              8191U,
	                              50000U,
	                              16381U,
	                              40000U,
	                              32749U,
	                              30000U};
	EXPECT_EQ((run_delta_unsumer<int16_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int16_t, vectors>(bases, deltas)));
	EXPECT_EQ((run_delta_register_unsumer<int16_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int16_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, RegisterI8FourVectorILP) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned   vectors = 4;
	std::vector<uint8_t> bases {3U, 101U, 247U, 64U};
	std::vector<uint8_t> deltas(galp::codec::utils::get_values_per_lane<int8_t>() * vectors);
	for (size_t position = 0; position < galp::codec::utils::get_values_per_lane<int8_t>(); ++position) {
		for (size_t vector = 0; vector < vectors; ++vector) {
			deltas[position * vectors + vector] =
			    static_cast<uint8_t>((position + 1U) * (vector + 3U) + vector);
		}
	}
	EXPECT_EQ((run_delta_register_unsumer<int8_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int8_t, vectors>(bases, deltas)));
}

TEST(DeltaUnsumer, RegisterI16FourVectorILP) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	constexpr unsigned    vectors = 4;
	std::vector<uint16_t> bases {11U, 4093U, 65530U, 32000U};
	std::vector<uint16_t> deltas(galp::codec::utils::get_values_per_lane<int16_t>() * vectors);
	for (size_t position = 0; position < galp::codec::utils::get_values_per_lane<int16_t>(); ++position) {
		for (size_t vector = 0; vector < vectors; ++vector) {
			deltas[position * vectors + vector] =
			    static_cast<uint16_t>((position + 7U) * (vector + 5U) * 997U);
		}
	}
	EXPECT_EQ((run_delta_register_unsumer<int16_t, vectors>(bases, deltas)),
	          (expected_delta_unsum<int16_t, vectors>(bases, deltas)));
}

TEST(DeltaLaneTileUnpacker, I8IndependentWidthsIncludingZeroAndFull) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	(void)run_delta_lane_tile_unpacker<uint8_t, 4>({0U, 1U, 5U, 8U});
}

TEST(DeltaLaneTileUnpacker, I16IndependentWidthsIncludingZeroAndFull) {
	if (const auto reason = delta_cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}
	(void)run_delta_lane_tile_unpacker<uint16_t, 4>({0U, 3U, 9U, 16U});
}
