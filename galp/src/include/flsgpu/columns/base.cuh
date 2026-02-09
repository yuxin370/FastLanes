// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/base.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_BASE_CUH
#define FLSGPU_COLUMNS_BASE_CUH

#include "flsgpu/device-types.cuh"
#include "flsgpu/utils.cuh"
#include <cstddef>
#include <cstdint>

namespace flsgpu { namespace device {

template <typename T>
struct FunctorBase {
	using UINT_T = typename utils::same_width_uint<T>::type;

	virtual __device__ __forceinline__ T operator()(const UINT_T value, [[maybe_unused]] const vi_t vector_index);
};

template <typename T>
struct DecompressorBase {
	/* Constructor, cannot be enforced
  __device__ DecompressorBase(const ColumnT column,
  const vi_t vector_index, const lane_t lane)
  */

	virtual void __device__ unpack_next_into(T* __restrict out);
};

}} // namespace flsgpu::device

#endif // FLSGPU_COLUMNS_BASE_CUH
