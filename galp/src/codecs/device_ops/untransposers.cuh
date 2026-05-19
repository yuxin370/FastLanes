// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/codecs/device_ops/untransposers.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_DECOMPRESSION_PRIMITIVES_UNTRANSPOSERS_CUH
#define GALP_DECOMPRESSION_PRIMITIVES_UNTRANSPOSERS_CUH

#include "codecs/consts.cuh"
#include <cstdint>

namespace galp::codec::device {

struct IdentityUntransposer {
	__device__ __forceinline__ static uint32_t map_index(const uint32_t in_idx) {
		return in_idx;
	}
};

struct FastLanesScalarAAVUF1LayoutTag {};

enum class FastLanesUntransposeKind : uint8_t {
	Input,
	Output,
};

__device__ __forceinline__ constexpr uint32_t bit_reverse3(const uint32_t value) {
	return ((value & 0x04u) >> 2) | (value & 0x02u) | ((value & 0x01u) << 2);
}

__device__ __forceinline__ constexpr uint32_t bit_reverse6(const uint32_t value) {
	return ((value & 0x20u) >> 5) | ((value & 0x10u) >> 3) | ((value & 0x08u) >> 1) | ((value & 0x04u) << 1) |
	       ((value & 0x02u) << 3) | ((value & 0x01u) << 5);
}

template <uint32_t ValuesPerVector, FastLanesUntransposeKind Kind, typename LayoutTag = FastLanesScalarAAVUF1LayoutTag>
struct FastLanesUntransposer;

template <>
struct FastLanesUntransposer<1024, FastLanesUntransposeKind::Input, FastLanesScalarAAVUF1LayoutTag> {
	__device__ __forceinline__ static uint32_t map_index(const uint32_t in_idx) {
		static_assert(galp::codec::consts::VALUES_PER_VECTOR == 1024,
		              "FastLanes 1024 input untranspose assumes 1024 values per vector.");
		// Align with generated::untranspose::fallback::scalar::untranspose_i for scalar_aav_uf1.
		const uint32_t low4       = in_idx & 0x0Fu;
		const uint32_t block6     = (in_idx >> 4) & 0x3Fu;
		const uint32_t block_low3 = block6 & 0x07u;
		const uint32_t block_hi3  = (block6 >> 3) & 0x07u;
		return (low4 << 6) | (bit_reverse3(block_low3) << 3) | block_hi3;
	}
};

template <>
struct FastLanesUntransposer<1024, FastLanesUntransposeKind::Output, FastLanesScalarAAVUF1LayoutTag> {
	__device__ __forceinline__ static uint32_t map_index(const uint32_t in_idx) {
		static_assert(galp::codec::consts::VALUES_PER_VECTOR == 1024,
		              "FastLanes 1024 output untranspose assumes 1024 values per vector.");
		// Align with generated::untranspose::fallback::scalar::untranspose_o for scalar_aav_uf1.
		const uint32_t low4       = in_idx & 0x0Fu;
		const uint32_t block6     = (in_idx >> 4) & 0x3Fu;
		const uint32_t block_low3 = block6 & 0x07u;
		const uint32_t block_hi3  = (block6 >> 3) & 0x07u;
		return (low4 << 6) | (bit_reverse3(block_low3) << 3) | block_hi3;
	}
};

using FastLanes1024InputUntransposer =
    FastLanesUntransposer<1024, FastLanesUntransposeKind::Input, FastLanesScalarAAVUF1LayoutTag>;
using FastLanes1024OutputUntransposer =
    FastLanesUntransposer<1024, FastLanesUntransposeKind::Output, FastLanesScalarAAVUF1LayoutTag>;

} // namespace galp::codec::device

#endif // GALP_DECOMPRESSION_PRIMITIVES_UNTRANSPOSERS_CUH
