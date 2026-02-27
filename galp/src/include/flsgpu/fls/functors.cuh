// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/fls/functors.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_FLS_FUNCTORS_CUH
#define FLSGPU_FLS_FUNCTORS_CUH

#include "flsgpu/device-types.cuh"
#include "flsgpu/old-fls.cuh"
#include "flsgpu/structs.cuh"
#include "flsgpu/utils.cuh"
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace flsgpu {
namespace device {
template <typename T>
struct BPFunctor : FunctorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	__device__ __forceinline__   BPFunctor() {};
	__device__ __forceinline__ T operator()(const UINT_T value, [[maybe_unused]] const vi_t vector_index) override {
		return value;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct FFORFunctor : FunctorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	UINT_T                     bases[UNPACK_N_VECTORS];
	__device__ __forceinline__ FFORFunctor(const UINT_T* a_bases) {
#pragma unroll
		for (int32_t v {0}; v < UNPACK_N_VECTORS; ++v) {
			bases[v] = a_bases[v];
		}
	};

	__device__ __forceinline__ T operator()(const UINT_T value, const vi_t vector_index) override {
		return static_cast<T>(value + bases[vector_index]);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct DICTFunctor : FunctorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	const UINT_T* __restrict__ keys;
	UINT_T bases[UNPACK_N_VECTORS];

	__device__ __forceinline__ DICTFunctor(const UINT_T* a_bases, const UINT_T* a_keys)
	    : keys(a_keys) {
#pragma unroll
		for (int v = 0; v < UNPACK_N_VECTORS; ++v)
			bases[v] = a_bases[v];
	}

	__device__ __forceinline__ T operator()(UINT_T value, vi_t vector_index) override {
		const auto idx = value + bases[vector_index];

		// return __ldg(keys + idx);
		return static_cast<T>(keys[idx]);
	}
};

template <typename T, typename IndexT, typename KeyT = IndexT>
struct DICTIndexFunctor {
	const KeyT* __restrict__ keys;
	__device__ __forceinline__ DICTIndexFunctor(const KeyT* a_keys)
	    : keys(a_keys) {
	}

	__device__ __forceinline__ T operator()(IndexT value, vi_t) {
		return static_cast<T>(keys[value]);
	}
};

template <typename T, typename IndexT, unsigned UNPACK_N_VECTORS>
struct DICTFunctorIdx {
	using KEY_T = typename utils::same_width_uint<T>::type;
	const KEY_T* __restrict__ keys;
	IndexT bases[UNPACK_N_VECTORS];

	__device__ __forceinline__ DICTFunctorIdx(const IndexT* a_bases, const KEY_T* a_keys)
	    : keys(a_keys) {
#pragma unroll
		for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
			bases[v] = a_bases[v];
		}
	}

	__device__ __forceinline__ T operator()(IndexT value, vi_t vector_index) {
		const auto idx = static_cast<size_t>(value + bases[vector_index]);
		return static_cast<T>(keys[idx]);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct DICTShfl32Functor : FunctorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;

	UINT_T k_lane; // key of this lane
	UINT_T bases[UNPACK_N_VECTORS];

	__device__ __forceinline__
	DICTShfl32Functor(const UINT_T* a_bases, const UINT_T* __restrict__ keys, const int32_t key_count) {
		const int lane = (int)(threadIdx.x & 31);

		// each lane loads its own key
		k_lane = lane < key_count ? keys[lane] : 0;

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v)
			bases[v] = a_bases[v];

		// incase warp divergence
		// __syncwarp();
	}

	__device__ __forceinline__ T operator()(UINT_T value, vi_t vector_index) override {
		// key count must be <= 32
		const int idx = (int)(value + bases[vector_index]);
		return static_cast<T>(__shfl_sync(0xFFFFFFFFu, k_lane, idx));
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct DICTAdaptiveFunctor : FunctorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;

	const UINT_T* __restrict__ keys;
	int32_t key_count;

	UINT_T k_lane; // only available when key_count<=32
	UINT_T bases[UNPACK_N_VECTORS];
	bool   use_shuffle;

	__device__ __forceinline__
	DICTAdaptiveFunctor(const UINT_T* a_bases, const UINT_T* __restrict__ keys, int32_t key_count)
	    : keys(keys)
	    , key_count(key_count) {

		const int lane = (int)(threadIdx.x & 31);

		use_shuffle = (key_count <= 32);

		if (use_shuffle) {
			k_lane = lane < key_count ? keys[lane] : 0;
		}

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v)
			bases[v] = a_bases[v];
	}

	__device__ __forceinline__ T operator()(UINT_T value, vi_t vector_index) override {
		const int idx = (int)(value + bases[vector_index]);

		if (use_shuffle) {
			return static_cast<T>(__shfl_sync(0xFFFFFFFFu, k_lane, idx));
		}

		return static_cast<T>(keys[idx]);
	}
};

} // namespace device
} // namespace flsgpu

#endif // FLSGPU_FLS_FUNCTORS_CUH
