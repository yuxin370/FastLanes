// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/codecs/decode/primitives/decompressors.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_DECOMPRESSION_PRIMITIVES_DECOMPRESSORS_CUH
#define GALP_DECOMPRESSION_PRIMITIVES_DECOMPRESSORS_CUH

#include "codecs/device_types.cuh"
#include "codecs/decode/primitives/expanders.cuh"
#include "codecs/decode/primitives/functors.cuh"
#include "codecs/decode/primitives/patchers.cuh"
#include "codecs/decode/primitives/unpackers.cuh"
#include "codecs/decode/primitives/unsumer.cuh"
#include "codecs/columns/all.cuh"
#include "codecs/utils.cuh"
#include <assert.h>
#include <cstdint>
#include <cstdio>

namespace galp::codec::device {
template <typename T, unsigned UNPACK_N_VECTORS, typename UnpackerT, typename ColumnT>
struct BPDecompressor : DecompressorBase<T> {
	UnpackerT                  unpacker;
	__device__ __forceinline__ BPDecompressor(const BPColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.packed_array + column.vector_offsets[vector_index],
	               lane,
	               column.bit_widths[vector_index],
	               BPFunctor<T>()) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename UnpackerT, typename ColumnT>
struct FFORDecompressor : DecompressorBase<T> {
	UnpackerT                  unpacker;
	__device__ __forceinline__ FFORDecompressor(const FFORColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.bp.packed_array + column.bp.vector_offsets[vector_index],
	               lane,
	               column.bp.bit_widths[vector_index],
	               FFORFunctor<T, UNPACK_N_VECTORS>(column.bases + vector_index)) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename ColumnT>
struct CONSTANTDecompressor : DecompressorBase<T> {
	T                          value;
	__device__ __forceinline__ CONSTANTDecompressor(const ColumnT                 column,
	                                                [[maybe_unused]] const vi_t   vector_index,
	                                                [[maybe_unused]] const lane_t lane)
	    : value(column.value) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		constexpr unsigned kCount = UNPACK_N_VECTORS * UNPACK_N_VALUES;
#pragma unroll
		for (unsigned i = 0; i < kCount; ++i) {
			out[i] = value;
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename PatcherT, typename ColumnT>
struct FREQDecompressor : DecompressorBase<T> {
	PatcherT                   patcher;
	__device__ __forceinline__ FREQDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : patcher(column, vector_index, lane) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		patcher.fill_and_patch(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename UnpackerT, typename PatcherT, typename ColumnT>
struct SLPATCHDecompressor : DecompressorBase<T> {
	PatcherT                   patcher;
	UnpackerT                  unpacker;
	__device__ __forceinline__ SLPATCHDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : patcher(column, vector_index, lane)
	    , unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
	               FFORFunctor<T, UNPACK_N_VECTORS>(column.ffor.bases + vector_index)) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
		patcher.patch(out);
	}
};

template <typename T,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename UnpackerT,
          typename PatcherT,
          typename ColumnT,
          typename ProcessorT = DICTFunctor<T, UNPACK_N_VECTORS>>
struct DICTSLPATCHDecompressor : DecompressorBase<T> {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	PatcherT  patcher;
	UnpackerT unpacker;

	__device__ __forceinline__ DICTSLPATCHDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : patcher(column, vector_index, lane)
	    , unpacker(column.index.ffor.bp.packed_array + column.index.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.index.ffor.bp.bit_widths[vector_index],
	               ProcessorT(column.index.ffor.bases + vector_index, column.keys)) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
		patcher.patch(out);
	}
};

template <typename T,
          unsigned UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          typename ProcessorT = DICTFunctor<T, UNPACK_N_VECTORS>>
struct DICTDecompressor : DecompressorBase<T> {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	UnpackerT                  unpacker;
	__device__ __forceinline__ DICTDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
	               ProcessorT(column.ffor.bases + vector_index,
	                          column.keys)) { // column.keys at global memory
	}

	// outer key pointer, which may points to shared memory
	__device__ __forceinline__ DICTDecompressor(const ColumnT column,
	                                            const vi_t    vector_index,
	                                            const lane_t  lane,
	                                            const typename ColumnT::KEY_T* __restrict keys_ptr)
	    : unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
	               ProcessorT(column.ffor.bases + vector_index, keys_ptr)) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename UnpackerT, typename ColumnT>
struct DICTShfl32Decompressor : DecompressorBase<T> {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	UnpackerT unpacker;

	__device__ __forceinline__
	DICTShfl32Decompressor(const DICTFFORColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
		               DICTShfl32Functor<T, UNPACK_N_VECTORS>(
		                   column.ffor.bases + vector_index, reinterpret_cast<const UINT_T*>(column.keys), column.key_count)) {
	}
	__device__ __forceinline__ void unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename ExpanderT, typename ColumnT>
struct CROSSRLEDecompressor : DecompressorBase<T> {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	ExpanderT                  expander;
	__device__ __forceinline__ CROSSRLEDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : expander(column, vector_index, lane) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		expander.expand_run_into(out);
	}
};

template <typename ValueT,
          typename IndexT,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename UnpackerT,
          typename ExpanderT,
          typename ColumnT>
struct RLEDecompressor : DecompressorBase<ValueT> {
	static constexpr unsigned                                     N_VALUES = UNPACK_N_VECTORS * UNPACK_N_VALUES;
	UnpackerT                                                     unpacker;
	RLEUnsumer<ValueT, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES> unsumer;
	ExpanderT                                                     expander;

	__device__ __forceinline__ RLEDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
	               FFORFunctor<IndexT, UNPACK_N_VECTORS>(column.ffor.bases + vector_index))
	    , unsumer(column, vector_index, lane)
	    , expander(column, vector_index, lane) {
	}

	void __device__ unpack_next_into(ValueT* __restrict out) {
		IndexT codes[N_VALUES];
		unpacker.unpack_next_into(codes);
		unsumer.unsum_inplace(codes);
		expander.expand_codes_into(codes, out);
	}
};

} // namespace galp::codec::device

#endif // GALP_DECOMPRESSION_PRIMITIVES_DECOMPRESSORS_CUH
