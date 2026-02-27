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

namespace flsgpu {
namespace device {

template <typename ValueT, typename CodeT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyRLEUnsumer {
private:
	CodeT      prefix[UNPACK_N_VECTORS];
	const int32_t n_lanes;

public:
	__device__ __forceinline__
	DummyRLEUnsumer(const flsgpu::device::RLEColumn<ValueT, CodeT> column, const vi_t vector_index, const lane_t lane)
	    : n_lanes(utils::get_n_lanes<CodeT>()) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			const CodeT* base_ptr = column.rsum_bases + (vector_index + v) * n_lanes;
			prefix[v]             = base_ptr[lane];
		}
	}

	__device__ __forceinline__ void unsum_inplace(CodeT* __restrict codes) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			CodeT cur = prefix[v];
#pragma unroll
			for (unsigned i = 0; i < UNPACK_N_VALUES; ++i) {
				const unsigned idx = i + v * UNPACK_N_VALUES;
				cur += codes[idx];
				codes[idx] = cur;
			}
			prefix[v] = cur;
		}
	}
};

} // namespace device
} // namespace flsgpu

#endif // FLSGPU_FLS_UNSUMER_CUH
