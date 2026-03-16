// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/fls/untransposers.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_FLS_UNTRANSPOSERS_CUH
#define FLSGPU_FLS_UNTRANSPOSERS_CUH

#include "flsgpu/consts.cuh"
#include <cstdint>

namespace flsgpu { namespace device {

struct IdentityUntransposer {
	__device__ __forceinline__ static uint32_t map_index(const uint32_t in_idx) {
		return in_idx;
	}
};

struct FastLanes1024Untransposer {
	__device__ __forceinline__ static uint32_t map_index(const uint32_t in_idx) {
		static_assert(consts::VALUES_PER_VECTOR == 1024, "FastLanes1024Untransposer assumes 1024 values per vector.");
		// Align with generated::untranspose::fallback::scalar::untranspose_i mapping for 1024 values.
		// Bit permutation (in bits x9..x0 -> out bits y9..y0):
		// y[9:6] = x[3:0], y2=x9, y1=x8, y0=x7, y3=x6, y4=x5, y5=x4.
		const uint32_t low4  = in_idx & 0x0Fu;
		const uint32_t high6 = (in_idx >> 4) & 0x3Fu;
		const uint32_t low6 =
		    ((high6 >> 3) & 0x07u) | ((high6 & 0x04u) << 1) | ((high6 & 0x02u) << 3) | ((high6 & 0x01u) << 5);
		return (low4 << 6) | low6;
	}
};

}} // namespace flsgpu::device

#endif // FLSGPU_FLS_UNTRANSPOSERS_CUH
