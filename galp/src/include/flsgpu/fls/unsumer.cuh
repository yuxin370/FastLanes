// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/fls/unsumer.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_FLS_UNSUMER_CUH
#define FLSGPU_FLS_UNSUMER_CUH

#include "flsgpu/device-types.cuh"
#include "flsgpu/structs.cuh"
#include "flsgpu/utils.cuh"
#include <cstddef>
#include <cstdint>

namespace flsgpu { namespace device {

template <typename CodeT, unsigned UNPACK_N_VALUES>
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

template <typename ValueT, typename CodeT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct RLEUnsumer {
private:
	CodeT                    prefix[UNPACK_N_VECTORS];
	static constexpr int32_t N_LANES = utils::get_n_lanes<CodeT>();

public:
	__device__ __forceinline__
	RLEUnsumer(const flsgpu::device::RLEColumn<ValueT, CodeT> column, const vi_t vector_index, const lane_t lane) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			const CodeT* base_ptr = column.rsum_bases + (vector_index + v) * N_LANES;
			prefix[v]             = base_ptr[lane];
		}
	}

	__device__ __forceinline__ void unsum_inplace(CodeT* __restrict codes) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			CodeT cur = prefix[v];
#pragma unroll
			for (unsigned logical_pos = 0; logical_pos < UNPACK_N_VALUES; ++logical_pos) {
				const unsigned in_lane_pos = RLEUnsumOrder<CodeT, UNPACK_N_VALUES>::code_index(logical_pos);
				const unsigned idx         = in_lane_pos + v * UNPACK_N_VALUES;
				cur += codes[idx];
				codes[idx] = cur;
			}
			prefix[v] = cur;
		}
	}
};

}} // namespace flsgpu::device

#endif // FLSGPU_FLS_UNSUMER_CUH
