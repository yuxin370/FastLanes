// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/codecs/device_ops/unsumer.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_DECOMPRESSION_PRIMITIVES_UNSUMER_CUH
#define GALP_DECOMPRESSION_PRIMITIVES_UNSUMER_CUH

#include "codecs/device_types.cuh"
#include "codecs/encodings/all.cuh"
#include "codecs/utils.cuh"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace galp::codec::device {

template <typename IndexT, unsigned UNPACK_N_VALUES>
struct RLEUnsumOrder {
	__device__ __forceinline__ static unsigned code_index(const unsigned logical_pos) {
		return logical_pos;
	}
};

template <>
struct RLEUnsumOrder<uint16_t, 16> {
	__device__ __forceinline__ static unsigned code_index(const unsigned logical_pos) {
		// Matches FastLanes generated rsum<uint16_t> order:
		// 0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15.
		return (logical_pos < 8u) ? (logical_pos << 1u) : (((logical_pos - 8u) << 1u) + 1u);
	}
};

template <typename ValueT, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct RLEUnsumer {
private:
	IndexT                   prefix[UNPACK_N_VECTORS];
	static constexpr int32_t N_LANES = galp::codec::utils::get_n_lanes<IndexT>();

public:
	template <typename ColumnT>
	__device__ __forceinline__ RLEUnsumer(const ColumnT column, const vi_t vector_index, const lane_t lane) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			const IndexT* base_ptr = column.rsum_bases + (vector_index + v) * N_LANES;
			prefix[v]              = base_ptr[lane];
		}
	}

	__device__ __forceinline__ void unsum_inplace(IndexT* __restrict codes) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			IndexT cur = prefix[v];
#pragma unroll
			for (unsigned logical_pos = 0; logical_pos < UNPACK_N_VALUES; ++logical_pos) {
				const unsigned in_lane_pos = RLEUnsumOrder<IndexT, UNPACK_N_VALUES>::code_index(logical_pos);
				const unsigned idx         = in_lane_pos + v * UNPACK_N_VALUES;
				cur += codes[idx];
				codes[idx] = cur;
			}
			prefix[v] = cur;
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct DeltaUnsumer;

// FastLanes I8 DELTA codes arrive in prefix order, so only one running
// prefix per unpacked vector is needed.
template <unsigned UNPACK_N_VECTORS>
struct DeltaUnsumer<int8_t, UNPACK_N_VECTORS> {
private:
	using UIntT                      = uint8_t;
	static constexpr int32_t N_LANES = galp::codec::utils::get_n_lanes<int8_t>();
	UIntT                    prefixes[UNPACK_N_VECTORS];

public:
	template <typename ColumnT>
	__device__ __forceinline__ DeltaUnsumer(const ColumnT column, const vi_t vector_index, const lane_t lane) {
#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			prefixes[vector] = column.rsum_bases[(vector_index + vector) * N_LANES + lane];
		}
	}

	template <typename UnpackerT>
	__device__ __forceinline__ void unsum_next_into(UnpackerT& unpacker, int8_t* __restrict out) {
		UIntT deltas[UNPACK_N_VECTORS];
		unpacker.unpack_next_into(deltas);
#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			prefixes[vector] = static_cast<UIntT>(prefixes[vector] + deltas[vector]);
			out[vector]      = static_cast<int8_t>(prefixes[vector]);
		}
	}
};

// FastLanes I16 DELTA stores each lane in 0,2,...,14,1,3,...,15 prefix
// order while the unpacker emits physical positions 0..15. Buffering the
// complete lane is therefore required before the first physical output.
template <unsigned UNPACK_N_VECTORS>
struct DeltaUnsumer<int16_t, UNPACK_N_VECTORS> {
private:
	using UIntT                                 = uint16_t;
	static constexpr unsigned N_VALUES_PER_LANE = galp::codec::utils::get_values_per_lane<int16_t>();
	static constexpr int32_t  N_LANES           = galp::codec::utils::get_n_lanes<int16_t>();
	static_assert(N_VALUES_PER_LANE == 16);

	UIntT    values[UNPACK_N_VECTORS * N_VALUES_PER_LANE];
	UIntT    prefixes[UNPACK_N_VECTORS];
	unsigned cursor      = 0;
	bool     initialized = false;

	__device__ __forceinline__ static constexpr unsigned code_index(const unsigned logical_position) {
		return logical_position < 8U ? logical_position * 2U : (logical_position - 8U) * 2U + 1U;
	}

	template <typename UnpackerT>
	__device__ __forceinline__ void initialize(UnpackerT& unpacker) {
		if (initialized) {
			return;
		}
#pragma unroll
		for (unsigned position = 0; position < N_VALUES_PER_LANE; ++position) {
			UIntT deltas[UNPACK_N_VECTORS];
			unpacker.unpack_next_into(deltas);
#pragma unroll
			for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
				values[vector * N_VALUES_PER_LANE + position] = deltas[vector];
			}
		}
#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			UIntT prefix = prefixes[vector];
#pragma unroll
			for (unsigned logical_position = 0; logical_position < N_VALUES_PER_LANE; ++logical_position) {
				const unsigned index = vector * N_VALUES_PER_LANE + code_index(logical_position);
				prefix               = static_cast<UIntT>(prefix + values[index]);
				values[index]        = prefix;
			}
		}
		initialized = true;
	}

public:
	template <typename ColumnT>
	__device__ __forceinline__ DeltaUnsumer(const ColumnT column, const vi_t vector_index, const lane_t lane) {
#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			prefixes[vector] = column.rsum_bases[(vector_index + vector) * N_LANES + lane];
		}
	}

	template <typename UnpackerT>
	__device__ __forceinline__ void unsum_next_into(UnpackerT& unpacker, int16_t* __restrict out) {
		initialize(unpacker);
#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			out[vector] = static_cast<int16_t>(values[vector * N_VALUES_PER_LANE + cursor]);
		}
		++cursor;
	}
};

// Whole-lane DELTA unsumer used by DELTARegisterDecompressor. The input and
// output are vector-major lane tiles. All positions are compile-time constants
// after unrolling, so the caller's tile can be scalarized into registers.
template <typename T, unsigned UNPACK_N_VECTORS>
struct DeltaRegisterUnsumer {
	using UIntT                                 = typename galp::codec::utils::same_width_uint<T>::type;
	static constexpr unsigned N_VALUES_PER_LANE = galp::codec::utils::get_values_per_lane<T>();
	static constexpr int32_t  N_LANES           = galp::codec::utils::get_n_lanes<T>();

	UIntT bases[UNPACK_N_VECTORS];

	__device__ __forceinline__ static constexpr unsigned code_index(const unsigned logical_position) {
		if constexpr (std::is_same_v<T, int16_t>) {
			static_assert(N_VALUES_PER_LANE == 16);
			return logical_position < 8U ? logical_position * 2U : (logical_position - 8U) * 2U + 1U;
		}
		return logical_position;
	}

	template <typename ColumnT>
	__device__ __forceinline__ DeltaRegisterUnsumer(const ColumnT column, const vi_t vector_index, const lane_t lane) {
#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			bases[vector] = column.rsum_bases[(vector_index + vector) * N_LANES + lane];
		}
	}

	__device__ __forceinline__ void unsum_inplace(UIntT* __restrict values) const {
		static_assert(N_VALUES_PER_LANE % 2U == 0U);
		constexpr unsigned SEGMENT_SIZE = N_VALUES_PER_LANE / 2U;
		UIntT             first_prefix[UNPACK_N_VECTORS];
		UIntT             second_prefix[UNPACK_N_VECTORS];

#pragma unroll
		for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
			first_prefix[vector]  = bases[vector];
			second_prefix[vector] = UIntT {0};
		}

		// The two half-lane scans are independent. Keeping vector as the inner
		// loop exposes the U=2/U=4 chains as instruction-level parallelism while
		// reducing the critical dependency depth from 8/16 to 4/8 additions.
#pragma unroll
		for (unsigned logical_position = 0; logical_position < SEGMENT_SIZE; ++logical_position) {
#pragma unroll
			for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
				const unsigned physical_position = code_index(logical_position);
				const unsigned index             = vector * N_VALUES_PER_LANE + physical_position;
				first_prefix[vector] = static_cast<UIntT>(first_prefix[vector] + values[index]);
				values[index]       = first_prefix[vector];
			}
		}

#pragma unroll
		for (unsigned logical_position = SEGMENT_SIZE; logical_position < N_VALUES_PER_LANE;
		     ++logical_position) {
#pragma unroll
			for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
				const unsigned physical_position = code_index(logical_position);
				const unsigned index             = vector * N_VALUES_PER_LANE + physical_position;
				second_prefix[vector] = static_cast<UIntT>(second_prefix[vector] + values[index]);
				values[index]        = second_prefix[vector];
			}
		}

#pragma unroll
		for (unsigned logical_position = SEGMENT_SIZE; logical_position < N_VALUES_PER_LANE;
		     ++logical_position) {
#pragma unroll
			for (unsigned vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
				const unsigned physical_position = code_index(logical_position);
				const unsigned index             = vector * N_VALUES_PER_LANE + physical_position;
				values[index] = static_cast<UIntT>(values[index] + first_prefix[vector]);
			}
		}
	}
};

} // namespace galp::codec::device

#endif // GALP_DECOMPRESSION_PRIMITIVES_UNSUMER_CUH
