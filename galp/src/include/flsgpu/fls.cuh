// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/fls.cuh
// ────────────────────────────────────────────────────────
#ifndef FLS_CUH
#define FLS_CUH

#include "device-types.cuh"
#include "old-fls.cuh"
#include "structs.cuh"
#include "utils.cuh"
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace flsgpu { namespace device {

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

template <typename T>
struct BitUnpackerBase {
	/* Constructor, but cannot be enforced
  BitUnpackerBase(
  const UINT_T *__restrict in, const lane_t lane,
  const vbw_t value_bit_width, OutputProcessor processor)
	*/
	virtual __device__ __forceinline__ void unpack_next_into(T* __restrict out);
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename OutputProcessor>
struct BitUnpackerDummy : flsgpu::device::BitUnpackerBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;

	const UINT_T*   in;
	OutputProcessor processor;

	__device__ __forceinline__ BitUnpackerDummy(const UINT_T* __restrict a_in,
	                                            const lane_t                 lane,
	                                            [[maybe_unused]] const vbw_t value_bit_width,
	                                            OutputProcessor              processor)
	    : in(a_in + lane)
	    , processor(processor) {};

	__device__ __forceinline__ void unpack_next_into(T* __restrict out) override {
		constexpr int32_t N_LANES = utils::get_n_lanes<UINT_T>();

#pragma unroll
		for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
			for (int j = 0; j < UNPACK_N_VALUES; ++j) {
				out[v * UNPACK_N_VALUES + j] = processor(in[v * consts::VALUES_PER_VECTOR + j * N_LANES], v);
			}
		}

		in += UNPACK_N_VALUES * N_LANES;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename OutputProcessor>
struct BitUnpackerOldFls : flsgpu::device::BitUnpackerBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;

	const UINT_T*   in;
	const vbw_t     value_bit_width;
	OutputProcessor processor;

	__device__ __forceinline__ BitUnpackerOldFls(const UINT_T* __restrict a_in,
	                                             const lane_t    lane,
	                                             const vbw_t     a_value_bit_width,
	                                             OutputProcessor processor)
	    : in(a_in + lane)
	    , value_bit_width(a_value_bit_width)
	    , processor(processor) {
		static_assert(UNPACK_N_VECTORS == 1, "Old FLS can only unpack 1 at a time");
		static_assert(UNPACK_N_VALUES == utils::get_values_per_lane<T>(), "Old FLS can only unpack entire lanes");
	};

	__device__ __forceinline__ void unpack_next_into(T* __restrict out) override {
		UINT_T* u_out = reinterpret_cast<UINT_T*>(out);
		oldfls::adjusted::unpack(in, u_out, value_bit_width);

		for (int32_t i {0}; i < UNPACK_N_VALUES; ++i) {
			out[i] = processor(u_out[i], 0);
		}
	}
};

template <typename T>
struct LoaderBase {
	using UINT_T = typename utils::same_width_uint<T>::type;

	virtual __device__ __forceinline__ void load_next_into(UINT_T* out);
	virtual __device__ __forceinline__ void next_line();
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct CacheLoader : LoaderBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;

	const UINT_T* in;
	int32_t       vector_offset;

	__device__ __forceinline__ CacheLoader(const UINT_T* in, const int32_t vector_offset)
	    : in(in)
	    , vector_offset(vector_offset) {};

	__device__ __forceinline__ void load_next_into(UINT_T* out) override {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			out[v] = *(in + v * vector_offset);
		}
	}

	__device__ __forceinline__ void next_line() override {
		in += utils::get_n_lanes<T>();
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned BUFFER_SIZE>
struct LocalMemoryLoader : LoaderBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	UINT_T        buffers[UNPACK_N_VECTORS * BUFFER_SIZE];
	const UINT_T* in;
	int32_t       vector_offset;
	int32_t       buffer_index = BUFFER_SIZE;

	__device__ __forceinline__ LocalMemoryLoader(const UINT_T* in, const int32_t vector_offset)
	    : in(in)
	    , vector_offset(vector_offset) {
		next_line();
	};

	__device__ __forceinline__ void load_next_into(UINT_T* out) override {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			out[v] = buffers[v * BUFFER_SIZE + buffer_index];
		}
	}

	__device__ __forceinline__ void next_line() override {
		if (buffer_index >= BUFFER_SIZE - 1) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
				for (int b {0}; b < BUFFER_SIZE; ++b) {
					buffers[v * BUFFER_SIZE + b] = *(in + v * vector_offset + b * utils::get_n_lanes<T>());
				}
			}
			in += BUFFER_SIZE * utils::get_n_lanes<T>();
			buffer_index = 0;
		} else {
			++buffer_index;
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned BUFFER_SIZE>
struct SharedMemoryLoader : LoaderBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	// No syncthreads are needed as threads only read and write their own section
	// Shared memory is allocated per block, so this also depends on block config
	// 4 divide by 32 bits/lanes, multiply by number of warps per block
	const UINT_T* in;
	int32_t       vector_offset;
	int32_t       buffer_index = BUFFER_SIZE;
	UINT_T*       buffers;

	__device__ __forceinline__ SharedMemoryLoader(const UINT_T* in, const int32_t vector_offset)
	    : in(in)
	    , vector_offset(vector_offset) {
		constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
		__shared__ UINT_T  shared_ptr[N_LANES * BUFFER_SIZE * UNPACK_N_VECTORS * (sizeof(T) / 4 * 2)];
		buffers = shared_ptr + threadIdx.x * BUFFER_SIZE * UNPACK_N_VECTORS;
		next_line();
	};

	__device__ __forceinline__ void load_next_into(UINT_T* out) override {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			out[v] = buffers[v * BUFFER_SIZE + buffer_index];
		}
	}

	__device__ __forceinline__ void next_line() override {
		if (buffer_index >= BUFFER_SIZE - 1) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
				for (int b {0}; b < BUFFER_SIZE; ++b) {
					buffers[v * BUFFER_SIZE + b] = *(in + v * vector_offset + b * utils::get_n_lanes<T>());
				}
			}
			in += BUFFER_SIZE * utils::get_n_lanes<T>();
			buffer_index = 0;
		} else {
			++buffer_index;
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned BUFFER_SIZE>
struct RegisterLoader : LoaderBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	UINT_T        buffers[UNPACK_N_VECTORS * BUFFER_SIZE];
	const UINT_T* in;
	int32_t       vector_offset;
	int32_t       buffer_index = BUFFER_SIZE;

	__device__ __forceinline__ RegisterLoader(const UINT_T* in, const int32_t vector_offset)
	    : in(in)
	    , vector_offset(vector_offset) {
		static_assert(BUFFER_SIZE <= 4, "Switch in RegisterLoader is not long enough for this buffer size.");
		next_line();
	};

	__device__ __forceinline__ void load_next_into(UINT_T* out) override {

		switch (buffer_index) {
		case 0: {
			if (0 < BUFFER_SIZE) {
#pragma unroll
				for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
					out[v] = buffers[v * BUFFER_SIZE + 0];
				}
			}
		} break;
		case 1: {
			if (1 < BUFFER_SIZE) {
#pragma unroll
				for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
					out[v] = buffers[v * BUFFER_SIZE + 1];
				}
			}
		} break;
		case 2: {
			if (2 < BUFFER_SIZE) {
#pragma unroll
				for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
					out[v] = buffers[v * BUFFER_SIZE + 2];
				}
			}
		} break;
		case 3: {
			if (3 < BUFFER_SIZE) {
#pragma unroll
				for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
					out[v] = buffers[v * BUFFER_SIZE + 3];
				}
			}
		} break;
		}
	}

	__device__ __forceinline__ void next_line() override {
		if (buffer_index >= BUFFER_SIZE - 1) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
				for (int b {0}; b < BUFFER_SIZE; ++b) {
					buffers[v * BUFFER_SIZE + b] = *(in + v * vector_offset + b * utils::get_n_lanes<T>());
				}
			}
			in += BUFFER_SIZE * utils::get_n_lanes<T>();
			buffer_index = 0;
		} else {
			++buffer_index;
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned BUFFER_SIZE>
struct RegisterBranchlessLoader : LoaderBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	UINT_T        buffers[UNPACK_N_VECTORS * BUFFER_SIZE];
	const UINT_T* in;
	int32_t       vector_offset;
	int32_t       buffer_index = BUFFER_SIZE;

	__device__ __forceinline__ RegisterBranchlessLoader(const UINT_T* in, const int32_t vector_offset)
	    : in(in)
	    , vector_offset(vector_offset) {
		next_line();
	};

	__device__ __forceinline__ void load_next_into(UINT_T* out) override {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			out[v] = buffers[v * BUFFER_SIZE];
		}
	}

	__device__ __forceinline__ void next_line() override {
		if (buffer_index >= BUFFER_SIZE - 1) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
				for (int b {0}; b < BUFFER_SIZE; ++b) {
					buffers[v * BUFFER_SIZE + b] = *(in + v * vector_offset + b * utils::get_n_lanes<T>());
				}
			}
			in += BUFFER_SIZE * utils::get_n_lanes<T>();
			buffer_index = 0;
		} else {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
				for (int b {1}; b < BUFFER_SIZE; ++b) {
					buffers[v * BUFFER_SIZE + b - 1] = buffers[v * BUFFER_SIZE + b];
				}
			}
			++buffer_index;
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct Masker {
	using UINT_T = typename utils::same_width_uint<T>::type;
	const vbw_t value_bit_width;
	const T     value_mask;
	uint16_t    buffer_offset = 0;

	__device__ __forceinline__ Masker(const vbw_t value_bit_width)
	    : value_bit_width(value_bit_width)
	    , value_mask(utils::set_first_n_bits<UINT_T>(value_bit_width)) {};

	__device__ __forceinline__ Masker(const uint16_t buffer_offset, const vbw_t value_bit_width)
	    : buffer_offset(buffer_offset)
	    , value_bit_width(value_bit_width)
	    , value_mask(utils::set_first_n_bits<UINT_T>(value_bit_width)) {};

	__device__ __forceinline__ void mask_and_increment(T* values, const T* buffers) {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			values[v] = (buffers[v] & (value_mask << buffer_offset)) >> buffer_offset;
		}
		buffer_offset += value_bit_width;
	}

	__device__ __forceinline__ void next_line() {
		buffer_offset -= utils::get_lane_bitwidth<T>();
	}
	__device__ __forceinline__ bool is_buffer_empty() const {
		return buffer_offset == utils::get_lane_bitwidth<T>();
	}

	__device__ __forceinline__ bool continues_on_next_line() const {
		return buffer_offset > utils::get_lane_bitwidth<T>();
	}

	__device__ __forceinline__ void mask_and_insert_remaining_value(T* values, const T* buffers) const {
		T buffer_offset_mask = (T {1} << static_cast<T>(buffer_offset)) - T {1};

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			values[v] |= (buffers[v] & buffer_offset_mask) << (value_bit_width - buffer_offset);
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename processor_T, typename LoaderT>
__device__ void unpack_vector_stateless(const typename utils::same_width_uint<T>::type* __restrict in,
                                        T* __restrict out,
                                        const lane_t lane,
                                        const vbw_t  value_bit_width,
                                        const si_t   start_index,
                                        processor_T  processor,
                                        int32_t      vector_offset) {
	using UINT_T                      = typename utils::same_width_uint<T>::type;
	constexpr uint8_t  LANE_BIT_WIDTH = utils::get_lane_bitwidth<UINT_T>();
	constexpr uint32_t N_LANES        = utils::get_n_lanes<UINT_T>();
	uint16_t           preceding_bits = (start_index * value_bit_width);
	uint16_t           buffer_offset  = preceding_bits % LANE_BIT_WIDTH;
	uint16_t           n_input_line   = preceding_bits / LANE_BIT_WIDTH;

	LoaderT                          loader(in + n_input_line * N_LANES + lane, vector_offset);
	Masker<UINT_T, UNPACK_N_VECTORS> masker(buffer_offset, value_bit_width);

	UINT_T values[UNPACK_N_VECTORS];

#pragma unroll
	for (int i = 0; i < UNPACK_N_VALUES; ++i) {
		if (masker.is_buffer_empty()) {
			loader.next_line();
			masker.next_line();
		}

		UINT_T buffers[UNPACK_N_VECTORS];
		loader.load_next_into(buffers);
		masker.mask_and_increment(values, buffers);

		if (masker.continues_on_next_line()) {
			loader.next_line();
			masker.next_line();
			loader.load_next_into(buffers);
			masker.mask_and_insert_remaining_value(values, buffers);
		}

#pragma unroll
		for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
			*(out + i + v * UNPACK_N_VALUES) = processor(values[v], v);
		}
	}
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename OutputProcessor>
struct BitUnpackerStateless : BitUnpackerBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;

	const UINT_T* __restrict in;
	const lane_t    lane;
	const vbw_t     value_bit_width;
	OutputProcessor processor;
	int32_t         vector_offset;

	si_t start_index = 0;

	__device__ __forceinline__ BitUnpackerStateless(const UINT_T* __restrict in,
	                                                const lane_t    lane,
	                                                const vbw_t     value_bit_width,
	                                                OutputProcessor processor)
	    : in(in)
	    , lane(lane)
	    , value_bit_width(value_bit_width)
	    , processor(processor)
	    , vector_offset(utils::get_compressed_vector_size<UINT_T>(value_bit_width)) {
	}

	__device__ __forceinline__ void unpack_next_into(T* __restrict out) override {
		unpack_vector_stateless<T,
		                        UNPACK_N_VECTORS,
		                        UNPACK_N_VALUES,
		                        OutputProcessor,
		                        CacheLoader<T, UNPACK_N_VECTORS>>(
		    in, out, lane, value_bit_width, start_index, processor, vector_offset);
		start_index += UNPACK_N_VALUES;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename processor_T>
__device__ void unpack_vector_stateless_branchless(const typename utils::same_width_uint<T>::type* __restrict in,
                                                   T* __restrict out,
                                                   const lane_t  lane,
                                                   const vbw_t   value_bit_width,
                                                   const si_t    start_index,
                                                   processor_T   processor,
                                                   const int32_t vector_offset) {
	using UINT_T                     = typename utils::same_width_uint<T>::type;
	constexpr int32_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<UINT_T>();
	constexpr int32_t N_LANES        = utils::get_n_lanes<UINT_T>();
	constexpr int32_t BIT_COUNT      = utils::sizeof_in_bits<T>();

	int32_t preceding_bits_first = (start_index * value_bit_width);
	int32_t n_input_line         = preceding_bits_first / LANE_BIT_WIDTH;
	int32_t offset_first         = preceding_bits_first % LANE_BIT_WIDTH;
	int32_t offset_second        = BIT_COUNT - offset_first;
	UINT_T  value_mask           = utils::set_first_n_bits<UINT_T>(value_bit_width);

	UINT_T values[UNPACK_N_VECTORS] = {0};

	in += n_input_line * N_LANES + lane;
#pragma unroll
	for (int32_t v {0}; v < UNPACK_N_VECTORS; v++) {
		const auto v_in = in + v * vector_offset;
		values[v] |= (v_in[0] & (value_mask << offset_first)) >> offset_first;
		values[v] |= (v_in[N_LANES] & (value_mask >> offset_second)) << offset_second;
		out[v] = processor(values[v], v);
	}
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename OutputProcessor>
struct BitUnpackerStatelessBranchless : BitUnpackerBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;

	const UINT_T* __restrict in;
	const lane_t    lane;
	const vbw_t     value_bit_width;
	OutputProcessor processor;
	const int32_t   vector_offset;

	si_t start_index = 0;

	__device__ __forceinline__ BitUnpackerStatelessBranchless(const UINT_T* __restrict in,
	                                                          const lane_t    lane,
	                                                          const vbw_t     value_bit_width,
	                                                          OutputProcessor processor)
	    : in(in)
	    , lane(lane)
	    , value_bit_width(value_bit_width)
	    , processor(processor)
	    , vector_offset(utils::get_compressed_vector_size<UINT_T>(value_bit_width)) {
	}

	__device__ __forceinline__ void unpack_next_into(T* __restrict out) override {
#pragma unroll
		for (int32_t i {0}; i < UNPACK_N_VALUES; i++) {
			unpack_vector_stateless_branchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
			    in, out + i, lane, value_bit_width, start_index + i, processor, vector_offset);
		}
		start_index += UNPACK_N_VALUES;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename OutputProcessor, typename LoaderT>
struct BitUnpackerStateful : BitUnpackerBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	LoaderT                          loader;
	Masker<UINT_T, UNPACK_N_VECTORS> masker;
	OutputProcessor                  processor;

	__device__ __forceinline__ BitUnpackerStateful(const UINT_T* __restrict in,
	                                               const lane_t    lane,
	                                               const vbw_t     value_bit_width,
	                                               OutputProcessor processor)
	    : loader(in + lane, utils::get_compressed_vector_size<UINT_T>(value_bit_width))
	    , masker(value_bit_width)
	    , processor(processor) {
	}

	__device__ __forceinline__ void unpack_next_into(T* __restrict out) override {
		UINT_T values[UNPACK_N_VECTORS];

#pragma unroll
		for (int i = 0; i < UNPACK_N_VALUES; ++i) {
			if (masker.is_buffer_empty()) {
				loader.next_line();
				masker.next_line();
			}

			UINT_T buffers[UNPACK_N_VECTORS];
			loader.load_next_into(buffers);
			masker.mask_and_increment(values, buffers);

			if (masker.continues_on_next_line()) {
				loader.next_line();
				masker.next_line();
				loader.load_next_into(buffers);
				masker.mask_and_insert_remaining_value(values, buffers);
			}

#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
				*(out + i + v * UNPACK_N_VALUES) = processor(values[v], v);
			}
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename OutputProcessor>
struct BitUnpackerStatefulBranchless : BitUnpackerBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	OutputProcessor processor;

	const UINT_T* in;
	const int32_t vector_offset;
	const vbw_t   value_bit_width;

	int32_t offset_first = 0;
	UINT_T  value_mask;

	__device__ __forceinline__ BitUnpackerStatefulBranchless(const UINT_T* __restrict a_in,
	                                                         const lane_t    lane,
	                                                         const vbw_t     value_bit_width,
	                                                         OutputProcessor processor)
	    : in(a_in + lane)
	    , value_bit_width(value_bit_width)
	    , value_mask(utils::set_first_n_bits<UINT_T>(value_bit_width))
	    , vector_offset(utils::get_compressed_vector_size<T>(value_bit_width))
	    , processor(processor) {
	}

	__device__ __forceinline__ void unpack_next_into(T* __restrict out) override {
		constexpr int32_t N_LANES        = utils::get_n_lanes<UINT_T>();
		constexpr int32_t BIT_COUNT      = utils::sizeof_in_bits<T>();
		constexpr int32_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<UINT_T>();

#pragma unroll
		for (int32_t i {0}; i < UNPACK_N_VALUES; i++) {
			const auto offset_second = BIT_COUNT - offset_first;

#pragma unroll
			for (int32_t v {0}; v < UNPACK_N_VECTORS; v++) {
				const auto v_in = in + v * vector_offset;
				out[UNPACK_N_VALUES * v + i] =
				    processor(((v_in[0] >> offset_first) & value_mask) |
				                  ((v_in[N_LANES] & (value_mask >> offset_second)) << offset_second),
				              v);
			}

			in += (offset_second <= value_bit_width) * N_LANES;
			offset_first = (offset_first + value_bit_width) % LANE_BIT_WIDTH;
		}
	}
};

template <typename OutT, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename OutputProcessor>
struct BitUnpackerStatefulBranchlessIdx : BitUnpackerBase<OutT> {
	using UINT_T = typename utils::same_width_uint<IndexT>::type;
	OutputProcessor processor;

	const UINT_T* in;
	const int32_t vector_offset;
	const vbw_t   value_bit_width;

	int32_t offset_first = 0;
	UINT_T  value_mask;

	__device__ __forceinline__ BitUnpackerStatefulBranchlessIdx(const UINT_T* __restrict a_in,
	                                                            const lane_t    lane,
	                                                            const vbw_t     value_bit_width,
	                                                            OutputProcessor processor)
	    : in(a_in + lane)
	    , vector_offset(utils::get_compressed_vector_size<IndexT>(value_bit_width))
	    , value_bit_width(value_bit_width)
	    , value_mask(utils::set_first_n_bits<UINT_T>(value_bit_width))
	    , processor(processor) {
	}

	__device__ __forceinline__ void unpack_next_into(OutT* __restrict out) override {
		constexpr int32_t N_LANES        = utils::get_n_lanes<UINT_T>();
		constexpr int32_t BIT_COUNT      = utils::sizeof_in_bits<IndexT>();
		constexpr int32_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<UINT_T>();

#pragma unroll
		for (int32_t i {0}; i < UNPACK_N_VALUES; i++) {
			const auto offset_second = BIT_COUNT - offset_first;

#pragma unroll
			for (int32_t v {0}; v < UNPACK_N_VECTORS; v++) {
				const auto v_in = in + v * vector_offset;
				const auto raw  = ((v_in[0] >> offset_first) & value_mask) |
				                 ((v_in[N_LANES] & (value_mask >> offset_second)) << offset_second);
				out[UNPACK_N_VALUES * v + i] = processor(static_cast<IndexT>(raw), v);
			}

			in += (offset_second <= value_bit_width) * N_LANES;
			offset_first = (offset_first + value_bit_width) % LANE_BIT_WIDTH;
		}
	}
};

template <typename T>
struct FREQExceptionPatcherBase {
public:
	__device__ __forceinline__ virtual void fill_and_patch(T* out) = 0;
	__device__ virtual ~FREQExceptionPatcherBase()                 = default;
};

template <typename T>
struct SLPATCHExceptionPatcherBase {
public:
	__device__ __forceinline__ virtual void patch(T* out) = 0;
	__device__ virtual ~SLPATCHExceptionPatcherBase()     = default;
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyFREQExceptionPatcher : flsgpu::device::FREQExceptionPatcherBase<T> {

public:
	void __device__ __forceinline__ fill_and_patch(T* out) override {
	}

	__device__ __forceinline__
	DummyFREQExceptionPatcher(const flsgpu::device::FREQColumn<T> column, const vi_t vector_index, const lane_t lane) {
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatelessFREQExceptionPatcher : FREQExceptionPatcherBase<T> {
	using INT_T = typename utils::same_width_int<T>::type;

	si_t         start_index = 0;
	uint16_t     exceptions_count[UNPACK_N_VECTORS];
	uint16_t*    vec_exceptions_positions[UNPACK_N_VECTORS];
	T*           vec_exceptions[UNPACK_N_VECTORS];
	T            frequent_value[UNPACK_N_VECTORS];
	const lane_t lane;

public:
	void __device__ __forceinline__ fill_and_patch(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

		const int first_pos = start_index * N_LANES + lane;
		const int last_pos  = first_pos + N_LANES * (UNPACK_N_VALUES - 1);

		start_index += UNPACK_N_VALUES;

		// 1) broadcast fill (sequential write, no branches)
#pragma unroll
		for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
			const T fv = frequent_value[v];
#pragma unroll
			for (int i = 0; i < UNPACK_N_VALUES; ++i) {
				out[v * UNPACK_N_VALUES + i] = fv;
			}
		}

		// 2) patch exceptions （stateless)
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			for (int i {0}; i < exceptions_count[v]; i++) {
				auto position  = vec_exceptions_positions[v][i];
				auto exception = vec_exceptions[v][i];
				if (position >= first_pos) {
					if (position <= last_pos && position % N_LANES == lane) {
						out[(position - first_pos) / N_LANES + v * UNPACK_N_VALUES] = exception;
					}
					if (position + 1 > last_pos) {
						break;
					}
				}
			}
		}
	}

	__device__ __forceinline__
	StatelessFREQExceptionPatcher(const FREQColumn<T> column, const vi_t first_vector_index, const lane_t lane)
	    : lane(lane) {

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			auto vec_index      = first_vector_index + v;
			exceptions_count[v] = column.counts[vec_index]; // exceotions count in this vector
			vec_exceptions_positions[v] =
			    column.positions +
			    column.exceptions_offsets[vec_index]; // exceptions_offsets corresponds to offests in exceptions array
			vec_exceptions[v] = column.exceptions +
			                    column.exceptions_offsets[vec_index]; // get the first position/exception in this vector
			frequent_value[v] = column.frequent_value[vec_index];     // also the first values in this vector
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulFREQExceptionPatcher : FREQExceptionPatcherBase<T> {
	using INT_T = typename utils::same_width_int<T>::type;

	si_t         start_index = 0;
	uint16_t     exceptions_count[UNPACK_N_VECTORS];
	uint16_t*    vec_exceptions_positions[UNPACK_N_VECTORS];
	T*           vec_exceptions[UNPACK_N_VECTORS];
	const lane_t lane;
	int32_t      exception_index[UNPACK_N_VECTORS] = {0};
	T            frequent_value[UNPACK_N_VECTORS];

public:
	void __device__ __forceinline__ fill_and_patch(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

		const int first_pos = start_index * N_LANES + lane;
		const int last_pos  = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
		start_index += UNPACK_N_VALUES;

		// 1) broadcast fill (sequential write, no branches)
#pragma unroll
		for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
			const T fv = frequent_value[v];
#pragma unroll
			for (int i = 0; i < UNPACK_N_VALUES; ++i) {
				out[v * UNPACK_N_VALUES + i] = fv;
			}
		}

		// 2) patch exceptions (stateful)
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			for (; exception_index[v] < exceptions_count[v]; exception_index[v]++) {
				auto position  = vec_exceptions_positions[v][exception_index[v]];
				auto exception = vec_exceptions[v][exception_index[v]];
				if (position >= first_pos) {
					if (position <= last_pos && position % N_LANES == lane) {
						out[(position - first_pos) / N_LANES + v * UNPACK_N_VALUES] = exception;
					}
					if (position + 1 > last_pos) {
						break;
					}
				}
			}
		}
	}

	__device__ __forceinline__
	StatefulFREQExceptionPatcher(const FREQColumn<T> column, const vi_t first_vector_index, const lane_t lane)
	    : lane(lane) {

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			auto vec_index              = first_vector_index + v;
			exceptions_count[v]         = column.counts[vec_index];
			vec_exceptions_positions[v] = column.positions + column.exceptions_offsets[vec_index];
			vec_exceptions[v]           = column.exceptions + column.exceptions_offsets[vec_index];
			frequent_value[v]           = column.frequent_value[vec_index];
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulSLPATCHExceptionPatcher : SLPATCHExceptionPatcherBase<T> {
	using INT_T = typename utils::same_width_int<T>::type;

	si_t         start_index = 0;
	uint16_t     exceptions_count[UNPACK_N_VECTORS];
	uint16_t*    vec_exceptions_positions[UNPACK_N_VECTORS];
	T*           vec_exceptions[UNPACK_N_VECTORS];
	const lane_t lane;
	int32_t      exception_index[UNPACK_N_VECTORS] = {0};

public:
	void __device__ __forceinline__ patch(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

		const int first_pos = start_index * N_LANES + lane;
		const int last_pos  = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
		start_index += UNPACK_N_VALUES;

		// patch exceptions
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			for (; exception_index[v] < exceptions_count[v]; exception_index[v]++) {
				auto position  = vec_exceptions_positions[v][exception_index[v]];
				auto exception = vec_exceptions[v][exception_index[v]];
				if (position >= first_pos) {
					if (position <= last_pos && position % N_LANES == lane) {
						out[(position - first_pos) / N_LANES + v * UNPACK_N_VALUES] = exception;
					}
					if (position + 1 > last_pos) {
						break;
					}
				}
			}
		}
	}

	__device__ __forceinline__
	StatefulSLPATCHExceptionPatcher(const SLPATCHColumn<T> column, const vi_t first_vector_index, const lane_t lane)
	    : lane(lane) {

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			auto vec_index              = first_vector_index + v;
			exceptions_count[v]         = column.counts[vec_index];
			vec_exceptions_positions[v] = column.positions + column.exceptions_offsets[vec_index];
			vec_exceptions[v]           = column.exceptions + column.exceptions_offsets[vec_index];
		}
	}
};

template <typename T, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulSLPATCHDictExceptionPatcher : SLPATCHExceptionPatcherBase<T> {
	using INT_T = typename utils::same_width_int<T>::type;
	using KEY_T = typename utils::same_width_uint<T>::type;

	si_t                               start_index = 0;
	uint16_t                           exceptions_count[UNPACK_N_VECTORS];
	uint16_t*                          vec_exceptions_positions[UNPACK_N_VECTORS];
	IndexT*                            vec_exceptions[UNPACK_N_VECTORS];
	DICTIndexFunctor<T, IndexT, KEY_T> processor;
	const lane_t                       lane;
	int32_t                            exception_index[UNPACK_N_VECTORS] = {0};

public:
	void __device__ __forceinline__ patch(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

		const int first_pos = start_index * N_LANES + lane;
		const int last_pos  = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
		start_index += UNPACK_N_VALUES;

		// patch exceptions with mapped dictionary values
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			for (; exception_index[v] < exceptions_count[v]; exception_index[v]++) {
				auto position  = vec_exceptions_positions[v][exception_index[v]];
				auto exception = vec_exceptions[v][exception_index[v]];
				if (position >= first_pos) {
					if (position <= last_pos && position % N_LANES == lane) {
						out[(position - first_pos) / N_LANES + v * UNPACK_N_VALUES] = processor(exception, v);
					}
					if (position + 1 > last_pos) {
						break;
					}
				}
			}
		}
	}

	__device__ __forceinline__ StatefulSLPATCHDictExceptionPatcher(const DICTSLPATCHColumn<T, IndexT> column,
	                                                               const vi_t   first_vector_index,
	                                                               const lane_t lane)
	    : processor(column.keys)
	    , lane(lane) {

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			auto vec_index              = first_vector_index + v;
			exceptions_count[v]         = column.index.counts[vec_index];
			vec_exceptions_positions[v] = column.index.positions + column.index.exceptions_offsets[vec_index];
			vec_exceptions[v]           = column.index.exceptions + column.index.exceptions_offsets[vec_index];
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct NaiveFREQExceptionPatcher : FREQExceptionPatcherBase<T> {
private:
	uint16_t  count[UNPACK_N_VECTORS];
	uint16_t* positions[UNPACK_N_VECTORS];
	T*        exceptions[UNPACK_N_VECTORS];
	uint16_t  current_position;
	T         frequent_value[UNPACK_N_VECTORS];

public:
	__device__ __forceinline__
	NaiveFREQExceptionPatcher(const FREQExtendedColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : current_position(lane) {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			const vi_t current_vector_index = vector_index + v;

			const auto offset_count = column.offsets_counts[current_vector_index * utils::get_n_lanes<T>() + lane];
			count[v]                = offset_count >> 10;

			const auto offset = (offset_count & 0x3FF);
			positions[v]      = column.positions + column.exceptions_offsets[current_vector_index] + offset;
			exceptions[v]     = column.exceptions + column.exceptions_offsets[current_vector_index] + offset;
			frequent_value[v] = column.frequent_value[current_vector_index];
		}
	}

	void __device__ __forceinline__ fill_and_patch(T* out) override {
#pragma unroll
		for (int w {0}; w < UNPACK_N_VALUES; ++w) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
				out[v * UNPACK_N_VALUES + w] = frequent_value[v];
				if (count[v] > 0 && current_position == *(positions[v])) {
					out[v * UNPACK_N_VALUES + w] = *(exceptions[v]);
					++(positions[v]);
					++(exceptions[v]);
					--(count[v]);
				}
			}
			current_position += utils::get_n_lanes<T>();
		}
	}
};

template <typename ToT, typename FromT>
ToT __device__ __forceinline__ reinterpret_as(FromT value) {
	ToT* ptr = reinterpret_cast<ToT*>(&value);
	return *ptr;
}

template <typename T>
void __device__ __forceinline__
overwrite_or_fill(T* __restrict buffer, const T fill_value, const T* __restrict new_value, const bool condition) {
	using UINT_T = typename utils::same_width_uint<T>::type;
	*buffer      = reinterpret_as<T>((reinterpret_as<UINT_T>(fill_value) * (!condition)) |
                                (reinterpret_as<UINT_T>(*new_value) * condition));
}

template <typename T>
void __device__ __forceinline__
overwrite_or_fill_mask(T* __restrict buffer, const T fill_value, const T* __restrict new_value, const bool condition) {
	using UINT_T = typename utils::same_width_uint<T>::type;

	// mask = 0 (cond=0) or all-ones (cond=1)
	const UINT_T mask = UINT_T(0) - static_cast<UINT_T>(condition);

	const UINT_T fv_u = reinterpret_as<UINT_T>(fill_value);
	const UINT_T nv_u = reinterpret_as<UINT_T>(*new_value);

	*buffer = reinterpret_as<T>((fv_u & ~mask) | (nv_u & mask));
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct NaiveBranchlessFREQExceptionPatcher : FREQExceptionPatcherBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	uint16_t  count[UNPACK_N_VECTORS];
	uint16_t* positions[UNPACK_N_VECTORS];
	T*        exceptions[UNPACK_N_VECTORS];
	uint16_t  current_position;
	T         frequent_value[UNPACK_N_VECTORS];

public:
	__device__ __forceinline__
	NaiveBranchlessFREQExceptionPatcher(const FREQExtendedColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : current_position(lane) {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			const vi_t current_vector_index = vector_index + v;

			const auto offset_count = column.offsets_counts[current_vector_index * utils::get_n_lanes<T>() + lane];
			count[v]                = offset_count >> 10;

			const auto exceptions_offset = column.exceptions_offsets[current_vector_index];
			const auto lane_offset       = (offset_count & 0x3FF);

			positions[v]      = column.positions + exceptions_offset + lane_offset;
			exceptions[v]     = column.exceptions + exceptions_offset + lane_offset;
			frequent_value[v] = column.frequent_value[current_vector_index];
		}
	}

	void __device__ __forceinline__ fill_and_patch(T* out) override {
#pragma unroll
		for (int w {0}; w < UNPACK_N_VALUES; ++w) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
				// It is possible to easily prefetch *positions[v] here
				bool comp = (count[v] > 0) && (current_position == (*positions[v]));
				overwrite_or_fill<T>(&out[v * UNPACK_N_VALUES + w], frequent_value[v], exceptions[v], comp);
				positions[v] += comp;
				exceptions[v] += comp;
				count[v] -= comp;
			}
			current_position += utils::get_n_lanes<T>();
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct PrefetchPositionFREQExceptionPatcher : FREQExceptionPatcherBase<T> {
private:
	uint16_t  count[UNPACK_N_VECTORS];
	uint16_t* positions[UNPACK_N_VECTORS];
	T*        exceptions[UNPACK_N_VECTORS];
	uint16_t  next_position[UNPACK_N_VECTORS];
	uint16_t  position;
	T         frequent_value[UNPACK_N_VECTORS];

public:
	__device__ __forceinline__ PrefetchPositionFREQExceptionPatcher(const FREQExtendedColumn<T> column,
	                                                                const vi_t                  first_vector_index,
	                                                                const lane_t                lane)
	    : position(lane) {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			const vi_t vector_index = first_vector_index + v;
			const auto offset_count = column.offsets_counts[vector_index * utils::get_n_lanes<T>() + lane];
			count[v]                = offset_count >> 10;

			const auto exceptions_offset = column.exceptions_offsets[vector_index];
			const auto lane_offset       = (offset_count & 0x3FF);
			positions[v]                 = column.positions + exceptions_offset + lane_offset;
			exceptions[v]                = column.exceptions + exceptions_offset + lane_offset;

			next_position[v]  = *positions[v];
			frequent_value[v] = column.frequent_value[vector_index];
		}
	}

	void __device__ __forceinline__ fill_and_patch(T* out) override {
#pragma unroll
		for (int w {0}; w < UNPACK_N_VALUES; ++w) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
				out[v * UNPACK_N_VALUES + w] = frequent_value[v];
				if (count[v] > 0 && position == next_position[v]) {
					out[v * UNPACK_N_VALUES + w] = *exceptions[v];
					++positions[v];
					++exceptions[v];
					--count[v];
					next_position[v] = *positions[v];
				}
			}
			position += utils::get_n_lanes<T>();
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct PrefetchAllFREQExceptionPatcher : FREQExceptionPatcherBase<T> {
private:
	uint16_t  count[UNPACK_N_VECTORS];
	uint16_t* positions[UNPACK_N_VECTORS];
	T*        exceptions[UNPACK_N_VECTORS];

	uint16_t index[UNPACK_N_VECTORS] = {0};
	uint16_t next_position[UNPACK_N_VECTORS];
	T        next_exception[UNPACK_N_VECTORS];
	uint16_t current_position;
	T        frequent_value[UNPACK_N_VECTORS];

public:
	void __device__ __forceinline__ read_next_exception(vi_t v) {
		if (index[v] < count[v]) {
			next_position[v]  = *positions[v];
			next_exception[v] = *exceptions[v];
			++positions[v];
			++exceptions[v];
			++index[v];
		} else {
			next_position[v] = consts::VALUES_PER_VECTOR;
		}
	}

	__device__ __forceinline__ PrefetchAllFREQExceptionPatcher(const FREQExtendedColumn<T> column,
	                                                           const vi_t                  first_vector_index,
	                                                           const lane_t                lane)
	    : current_position(lane) {
		// Parse the data from the column
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			const auto vector_index = first_vector_index + v;
			const auto offset_count = column.offsets_counts[vector_index * utils::get_n_lanes<T>() + lane];
			count[v]                = offset_count >> 10;

			const auto exceptions_offset = column.exceptions_offsets[vector_index];
			const auto lane_offset       = (offset_count & 0x3FF);
			positions[v]                 = column.positions + exceptions_offset + lane_offset;
			exceptions[v]                = column.exceptions + exceptions_offset + lane_offset;

			next_position[v]  = *positions[v];
			next_exception[v] = *exceptions[v];

			// This might be avoided to avoid a branch
			read_next_exception(v);
			frequent_value[v] = column.frequent_value[vector_index];
		}
	}

	void __device__ __forceinline__ fill_and_patch(T* out) override {
#pragma unroll
		for (int w {0}; w < UNPACK_N_VALUES; ++w) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
				out[v * UNPACK_N_VALUES + w] = frequent_value[v];
				if (current_position == next_position[v]) {
					out[v * UNPACK_N_VALUES + w] = next_exception[v];
					read_next_exception(v);
				}
			}
			current_position += utils::get_n_lanes<T>();
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct PrefetchAllBranchlessFREQExceptionPatcher : FREQExceptionPatcherBase<T> {
private:
	uint16_t  count[UNPACK_N_VECTORS];
	uint16_t* positions[UNPACK_N_VECTORS];
	T*        exceptions[UNPACK_N_VECTORS];

	uint16_t index[UNPACK_N_VECTORS] = {0};
	uint16_t next_position[UNPACK_N_VECTORS];
	T        next_exception[UNPACK_N_VECTORS];
	uint16_t current_position;
	T        frequent_value[UNPACK_N_VECTORS];

public:
	void __device__ __forceinline__ read_next_exception() {
	}

	__device__ __forceinline__ PrefetchAllBranchlessFREQExceptionPatcher(const FREQExtendedColumn<T> column,
	                                                                     const vi_t                  first_vector_index,
	                                                                     const lane_t                lane)
	    : current_position(lane) {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			const vi_t vector_index = first_vector_index + v;
			const auto offset_count = column.offsets_counts[vector_index * utils::get_n_lanes<T>() + lane];
			count[v]                = offset_count >> 10;

			const auto exceptions_offset = column.exceptions_offsets[vector_index];
			const auto lane_offset       = (offset_count & 0x3FF);
			positions[v]                 = column.positions + exceptions_offset + lane_offset;
			exceptions[v]                = column.exceptions + exceptions_offset + lane_offset;

			next_position[v]  = *positions[v];
			next_exception[v] = *exceptions[v];

			bool comparison = count[v] > 0;
			next_position[v] += (!comparison) * consts::VALUES_PER_VECTOR;
			frequent_value[v] = column.frequent_value[vector_index];
		}
	}

	void __device__ __forceinline__ fill_and_patch(T* out) override {
		// NOTES: It is probably possible to remove the next_position variable
		// as well as the next_exception, if you use prefetching. This would make
		// it easier to do multiple vectors.

#pragma unroll
		for (int w {0}; w < UNPACK_N_VALUES; ++w) {
#pragma unroll
			for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
				bool comparison = current_position == next_position[v];

				overwrite_or_fill<T>(out + v * UNPACK_N_VALUES + w, frequent_value[v], &next_exception[v], comparison);

				positions[v] += comparison;
				exceptions[v] += comparison;
				next_position[v]  = *positions[v];
				next_exception[v] = *exceptions[v];
				index[v] += comparison;

				comparison = index[v] < count[v];
				next_position[v] += (!comparison) * consts::VALUES_PER_VECTOR;
			}
			current_position += utils::get_n_lanes<T>();
		}
	}
};

template <typename T>
struct CROSSRLEExpanderBase {
public:
	__device__ __forceinline__ virtual void rle_expand(T* out) = 0;
	__device__ virtual ~CROSSRLEExpanderBase()                 = default;
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	si_t      start_index = 0;
	uint32_t  vec_base[UNPACK_N_VECTORS];
	UINT_T*   vec_values[UNPACK_N_VECTORS];
	uint32_t* vec_lengths[UNPACK_N_VECTORS];
	uint32_t  vec_runs_positions[UNPACK_N_VECTORS]; // the starting position of the first run in each vector
	                                                // const lane_t lane;

public:
	void __device__ __forceinline__ rle_expand(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			// the starting position in decompressed arrays of the current run
			uint32_t current_run_position = vec_runs_positions[v];
			uint32_t run_index            = 0;
			uint32_t run_length           = vec_lengths[v][run_index];
			UINT_T   current_value        = vec_values[v][run_index];
#pragma unroll
			for (int i {0}; i < UNPACK_N_VALUES; ++i) {
				const int global_position = static_cast<int>(vec_base[v]) + (start_index + i) * N_LANES;
				// move to the correct run
				while (global_position >= current_run_position + run_length) {
					current_run_position += run_length;
					++run_index;
					run_length    = vec_lengths[v][run_index];
					current_value = vec_values[v][run_index];
				}
				out[v * UNPACK_N_VALUES + i] = current_value;
			}
		}

		start_index += UNPACK_N_VALUES; // will traverse the whole vectors
	}

	__device__ __forceinline__
	DummyCROSSRLEExpander(const flsgpu::device::CROSSRLEColumn<T> column, const vi_t vector_index, const lane_t lane) {
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			auto vec_index        = vector_index + v;
			vec_values[v]         = column.values + column.offsets[vec_index];
			vec_lengths[v]        = column.lengths + column.offsets[vec_index];
			vec_base[v]           = static_cast<int32_t>(vec_index * consts::VALUES_PER_VECTOR) + lane;
			vec_runs_positions[v] = column.run_positions[column.offsets[vec_index]];
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	si_t      start_index = 0;
	uint32_t  vec_base[UNPACK_N_VECTORS];
	UINT_T*   vec_values[UNPACK_N_VECTORS];
	uint32_t* vec_lengths[UNPACK_N_VECTORS];

	// --- persistent per-vector cursor state (Point 1) ---
	uint32_t vec_run_index[UNPACK_N_VECTORS];
	uint32_t vec_run_length[UNPACK_N_VECTORS];
	uint32_t vec_run_pos[UNPACK_N_VECTORS];   // current run start position in decompressed space
	UINT_T   vec_run_value[UNPACK_N_VECTORS]; // current run value (cached)

public:
	void __device__ __forceinline__ rle_expand(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			// load persistent state into locals
			uint32_t run_index  = vec_run_index[v];
			uint32_t run_length = vec_run_length[v];
			uint32_t run_pos    = vec_run_pos[v];
			UINT_T   value      = vec_run_value[v];

#pragma unroll
			for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
				// global position is monotonic increasing across calls (start_index increases)
				const uint32_t global_position =
				    vec_base[v] + static_cast<uint32_t>(start_index + i) * (uint32_t)N_LANES;

				// advance to the correct run
				while (global_position >= run_pos + run_length) {
					run_pos += run_length;
					++run_index;

					// Point 3: keep run_length/index 32-bit. Assumes lengths fit in uint32_t.
					run_length = vec_lengths[v][run_index];
					value      = vec_values[v][run_index];
				}

				out[v * UNPACK_N_VALUES + i] = value;
			}

			// store back persistent state
			vec_run_index[v]  = run_index;
			vec_run_length[v] = run_length;
			vec_run_pos[v]    = run_pos;
			vec_run_value[v]  = value;
		}

		start_index += UNPACK_N_VALUES;
	}

	__device__ __forceinline__ StatefulCROSSRLEExpander(const flsgpu::device::CROSSRLEColumn<T> column,
	                                                    const vi_t                              vector_index,
	                                                    const lane_t                            lane) {
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			auto vec_index = vector_index + v;

			vec_values[v]  = column.values + column.offsets[vec_index];
			vec_lengths[v] = column.lengths + column.offsets[vec_index];

			vec_base[v] = static_cast<uint32_t>(vec_index * consts::VALUES_PER_VECTOR) + (uint32_t)lane;

			// init cursor state at first run
			vec_run_index[v]  = 0;
			vec_run_pos[v]    = column.run_positions[column.offsets[vec_index]];
			vec_run_length[v] = static_cast<uint32_t>(vec_lengths[v][0]); // assumes <= UINT32_MAX
			vec_run_value[v]  = vec_values[v][0];
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct PrefetchStatefulCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	si_t      start_index = 0;
	uint32_t  vec_base[UNPACK_N_VECTORS];
	UINT_T*   vec_values[UNPACK_N_VECTORS];
	uint32_t* vec_lengths[UNPACK_N_VECTORS];

	// run count of each vector
	uint32_t vec_run_count[UNPACK_N_VECTORS];

	// persistent cursor state
	uint32_t vec_run_index[UNPACK_N_VECTORS];
	uint32_t vec_run_end[UNPACK_N_VECTORS];
	UINT_T   vec_run_value[UNPACK_N_VECTORS];

	// --- prefetch/cache next run meta ---
	uint32_t vec_next_length[UNPACK_N_VECTORS];
	UINT_T   vec_next_value[UNPACK_N_VECTORS];

	__device__ __forceinline__ void load_next(int      v,
	                                          uint32_t run_index,
	                                          uint32_t run_count,
	                                          const uint32_t* __restrict__ lengths,
	                                          const UINT_T* __restrict__ values) {
		const uint32_t ni = run_index + 1u;
		if (ni < run_count) {
#if __CUDA_ARCH__ >= 350
			vec_next_length[v] = __ldg(lengths + ni);
			vec_next_value[v]  = __ldg(values + ni);
#else
			vec_next_length[v] = lengths[ni];
			vec_next_value[v]  = values[ni];
#endif
		}
	}

public:
	__device__ __forceinline__ PrefetchStatefulCROSSRLEExpander(const flsgpu::device::CROSSRLEColumn<T> column,
	                                                            const vi_t                              vector_index,
	                                                            const lane_t                            lane) {
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			const auto vec_index = vector_index + (vi_t)v;

			const uint32_t off0 = column.offsets[vec_index];
			const uint32_t off1 = column.offsets[vec_index + 1];

			vec_values[v]    = column.values + off0;
			vec_lengths[v]   = column.lengths + off0;
			vec_run_count[v] = off1 - off0; // >=1

			vec_base[v] = static_cast<uint32_t>(vec_index * consts::VALUES_PER_VECTOR) + (uint32_t)lane;

			// init at first run
			vec_run_index[v] = 0;
			vec_run_value[v] = (UINT_T)vec_values[v][0];

			const uint32_t run0_pos = column.run_positions[off0];
			const uint32_t run0_len = (uint32_t)vec_lengths[v][0];
			vec_run_end[v]          = run0_pos + run0_len; // end of run0

			// prefetch next run meta
			load_next(v, 0u, vec_run_count[v], vec_lengths[v], vec_values[v]);
		}
	}

	void __device__ __forceinline__ rle_expand(T* out) override {
		constexpr uint32_t N_LANES = (uint32_t)utils::get_n_lanes<INT_T>();

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			const UINT_T* __restrict__ values    = vec_values[v];
			const uint32_t* __restrict__ lengths = vec_lengths[v];
			const uint32_t run_count             = vec_run_count[v];

			uint32_t run_index = vec_run_index[v];
			uint32_t run_end   = vec_run_end[v];
			UINT_T   value     = vec_run_value[v];

			uint32_t next_len = vec_next_length[v];
			UINT_T   next_val = vec_next_value[v];

			uint32_t pos = vec_base[v] + (uint32_t)start_index * N_LANES;

			const uint32_t last_pos = pos + (uint32_t)(UNPACK_N_VALUES - 1) * N_LANES;

#pragma unroll
			for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
				while (pos >= run_end && (run_index + 1u) < run_count) {
					++run_index;

					// consume prefetched next
					value = next_val;
					run_end += next_len; // contiguous runs

					// refresh next
					load_next(v, run_index, run_count, lengths, values);
					next_len = vec_next_length[v];
					next_val = vec_next_value[v];
				}

				out[v * UNPACK_N_VALUES + i] = (T)value;
				pos += N_LANES;
			}

			// store back
			vec_run_index[v] = run_index;
			vec_run_end[v]   = run_end;
			vec_run_value[v] = value;

			vec_next_length[v] = next_len;
			vec_next_value[v]  = next_val;
		}

		start_index += UNPACK_N_VALUES;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulAdvanceCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	si_t      start_index = 0;
	uint32_t  vec_base[UNPACK_N_VECTORS];
	UINT_T*   vec_values[UNPACK_N_VECTORS];
	uint32_t* vec_lengths[UNPACK_N_VECTORS];

	// --- persistent per-vector cursor state (Point 1) ---
	uint32_t vec_run_index[UNPACK_N_VECTORS];
	uint32_t vec_run_length[UNPACK_N_VECTORS];
	uint32_t vec_run_pos[UNPACK_N_VECTORS];   // current run start position in decompressed space
	UINT_T   vec_run_value[UNPACK_N_VECTORS]; // current run value (cached)

	__device__ __forceinline__ void advance_if_needed(const uint32_t gp,
	                                                  uint32_t&      run_pos,
	                                                  uint32_t&      run_len,
	                                                  uint32_t&      run_idx,
	                                                  const uint32_t* __restrict__ lengths,
	                                                  const UINT_T* __restrict__ values,
	                                                  UINT_T& run_val) {

		const uint32_t run_end = run_pos + run_len;
		const int      need    = (gp >= run_end); // 0/1

		// predicated update
		const uint32_t new_idx = run_idx + (uint32_t)need;
		const uint32_t new_pos = run_pos + (need ? run_len : 0u);

		// update only when need=1
		const uint32_t next_len = lengths[new_idx];
		const UINT_T   next_val = values[new_idx];

		run_idx = new_idx;
		run_pos = new_pos;
		run_len = need ? next_len : run_len;
		run_val = need ? next_val : run_val;
	}

public:
	__device__ __forceinline__ void rle_expand(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {

			uint32_t run_idx = vec_run_index[v];
			uint32_t run_len = vec_run_length[v];
			uint32_t run_pos = vec_run_pos[v];
			UINT_T   run_val = vec_run_value[v];

			const uint32_t* __restrict__ lengths = vec_lengths[v];
			const UINT_T* __restrict__ values    = vec_values[v];

			uint32_t gp = vec_base[v] + (uint32_t)start_index * (uint32_t)N_LANES;

#pragma unroll
			for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
				// advance 2 times for better ILP
				advance_if_needed(gp, run_pos, run_len, run_idx, lengths, values, run_val);
				advance_if_needed(gp, run_pos, run_len, run_idx, lengths, values, run_val);

				while (gp >= run_pos + run_len) {
					run_pos += run_len;
					++run_idx;
					run_len = lengths[run_idx];
					run_val = values[run_idx];
				}

				out[v * UNPACK_N_VALUES + i] = (T)run_val;
				gp += (uint32_t)N_LANES;
			}

			vec_run_index[v]  = run_idx;
			vec_run_length[v] = run_len;
			vec_run_pos[v]    = run_pos;
			vec_run_value[v]  = run_val;
		}

		start_index += UNPACK_N_VALUES;
	}

	__device__ __forceinline__ StatefulAdvanceCROSSRLEExpander(const flsgpu::device::CROSSRLEColumn<T> column,
	                                                           const vi_t                              vector_index,
	                                                           const lane_t                            lane) {
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			auto vec_index = vector_index + v;

			vec_values[v]  = column.values + column.offsets[vec_index];
			vec_lengths[v] = column.lengths + column.offsets[vec_index];

			vec_base[v] = static_cast<uint32_t>(vec_index * consts::VALUES_PER_VECTOR) + (uint32_t)lane;

			// init cursor state at first run
			vec_run_index[v]  = 0;
			vec_run_pos[v]    = column.run_positions[column.offsets[vec_index]];
			vec_run_length[v] = static_cast<uint32_t>(vec_lengths[v][0]); // assumes <= UINT32_MAX
			vec_run_value[v]  = vec_values[v][0];
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulShuffleCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	si_t   start_index = 0;
	lane_t lane_;

	uint32_t  vec_base0[UNPACK_N_VECTORS];
	UINT_T*   vec_values[UNPACK_N_VECTORS];
	uint32_t* vec_lengths[UNPACK_N_VECTORS];

	uint32_t vec_run_index[UNPACK_N_VECTORS];
	uint32_t vec_run_length[UNPACK_N_VECTORS];
	uint32_t vec_run_pos[UNPACK_N_VECTORS];
	UINT_T   vec_run_value[UNPACK_N_VECTORS];

	static __device__ __forceinline__ UINT_T shfl0(UINT_T x, unsigned mask, int width) {
		if constexpr (sizeof(UINT_T) == 4) {
			return (UINT_T)__shfl_sync(mask, (uint32_t)x, 0, width);
		} else if constexpr (sizeof(UINT_T) == 8) {

			unsigned long long y = (unsigned long long)x;
			y                    = __shfl_sync(mask, y, 0, width);
			return (UINT_T)y;
		} else {
			return x;
		}
	}

public:
	__device__ __forceinline__ void rle_expand(T* __restrict__ out) override {
		constexpr int  N_LANES = utils::get_n_lanes<INT_T>();
		const unsigned mask    = __activemask();
		const int      lane    = (int)lane_;

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {

			uint32_t run_idx = 0;
			uint32_t run_pos = 0;
			uint32_t run_len = 0;
			UINT_T   run_val = 0;

			if (lane == 0) {
				run_idx = vec_run_index[v];
				run_pos = vec_run_pos[v];
				run_len = vec_run_length[v];
				run_val = vec_run_value[v];
			}

			T* __restrict__ outv = out + v * UNPACK_N_VALUES;
			const uint32_t base0 = vec_base0[v];

#pragma unroll
			for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
				// strip covers [strip_start, strip_end) in decompressed space
				const uint32_t strip_start = base0 + (uint32_t)(start_index + i) * (uint32_t)N_LANES;
				const uint32_t strip_end   = strip_start + (uint32_t)N_LANES;

				uint32_t run_end = 0;
				if (lane == 0) {

					run_end = run_pos + run_len;
					while (strip_start >= run_end) {
						run_pos = run_end;
						++run_idx;
						run_len = vec_lengths[v][run_idx];
						run_val = vec_values[v][run_idx];
						run_end = run_pos + run_len;
					}
				}

				const uint32_t b_run_pos = __shfl_sync(mask, run_pos, 0, N_LANES);
				const uint32_t b_run_end = __shfl_sync(mask, run_end, 0, N_LANES);

				// uniform branch: strip fully covered by current run
				if (b_run_pos <= strip_start && b_run_end >= strip_end) {
					const UINT_T b_run_val = shfl0(run_val, mask, N_LANES);
					outv[i]                = (T)b_run_val;
					continue;
				}

				// slow path: strip cover several run -> lane0 generate [seg_begin, seg_end) iteratelly
				for (;;) {
					uint32_t packed_be = 0; // [15:0]=begin, [31:16]=end
					UINT_T   seg_val   = 0;

					if (lane == 0) {
						const uint32_t local_run_end = run_pos + run_len;

						const uint32_t seg_begin_u = (run_pos > strip_start) ? (run_pos - strip_start) : 0u;
						const uint32_t seg_end_u =
						    (local_run_end < strip_end) ? (local_run_end - strip_start) : (uint32_t)N_LANES;

						// one less shfl
						packed_be = (seg_begin_u & 0xFFFFu) | ((seg_end_u & 0xFFFFu) << 16);
						seg_val   = run_val;

						// if not cover the strip，push to next run
						if (seg_end_u < (uint32_t)N_LANES) {
							run_pos = local_run_end;
							++run_idx;
							run_len = vec_lengths[v][run_idx];
							run_val = vec_values[v][run_idx];
						}
					}

					const uint32_t b_be      = __shfl_sync(mask, packed_be, 0, N_LANES);
					const int      seg_begin = (int)(b_be & 0xFFFFu);
					const int      seg_end   = (int)(b_be >> 16);
					const UINT_T   b_val     = shfl0(seg_val, mask, N_LANES);

					if (lane >= seg_begin && lane < seg_end) {
						outv[i] = (T)b_val;
					}

					if (seg_end >= N_LANES)
						break;
				}
			}

			if (lane == 0) {
				vec_run_index[v]  = run_idx;
				vec_run_pos[v]    = run_pos;
				vec_run_length[v] = run_len;
				vec_run_value[v]  = run_val;
			}
		}

		start_index += UNPACK_N_VALUES;
	}

	__device__ __forceinline__ StatefulShuffleCROSSRLEExpander(const flsgpu::device::CROSSRLEColumn<T> column,
	                                                           const vi_t                              vector_index,
	                                                           const lane_t                            lane)
	    : lane_(lane) {
		constexpr int N_LANES = utils::get_n_lanes<INT_T>();

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			const auto vec_index = vector_index + v;

			uint32_t off = (lane_ == 0) ? column.offsets[vec_index] : 0;
			off          = __shfl_sync(0xFFFFFFFFu, off, 0, N_LANES);

			vec_values[v]  = column.values + off;
			vec_lengths[v] = column.lengths + off;

			uint32_t rp    = (lane_ == 0) ? column.run_positions[off] : 0;
			rp             = __shfl_sync(0xFFFFFFFFu, rp, 0, N_LANES);
			vec_run_pos[v] = rp;

			vec_base0[v]      = (uint32_t)(vec_index * consts::VALUES_PER_VECTOR);
			vec_run_index[v]  = 0;
			vec_run_length[v] = vec_lengths[v][0];
			vec_run_value[v]  = vec_values[v][0];
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulExtendedCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	// k-space cursor: UNPACK_N_VALUES k each time
	uint16_t start_k_ = 0;
	lane_t   lane_;

	// per-vector lane-run slice pointers (already offset to this lane)
	const UINT_T*   lane_vals_[UNPACK_N_VECTORS];
	const uint16_t* lane_lens_[UNPACK_N_VECTORS];

	// per-vector lane-run state
	uint16_t run_cnt_[UNPACK_N_VECTORS]; // number of runs in this lane (within this vector)
	uint16_t run_idx_[UNPACK_N_VECTORS];
	uint16_t run_pos_[UNPACK_N_VECTORS]; // start k of current run
	uint16_t run_len_[UNPACK_N_VECTORS]; // length in k-space
	UINT_T   run_val_[UNPACK_N_VECTORS];

public:
	__device__ __forceinline__ StatefulExtendedCROSSRLEExpander(const flsgpu::device::CROSSRLEExtendedColumn<T> col,
	                                                            const vi_t   vec_index,
	                                                            const lane_t lane)
	    : lane_(lane) {
		constexpr uint32_t N_LANES = (uint32_t)utils::get_n_lanes<INT_T>();

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			const uint32_t vec = (uint32_t)(vec_index + v);

			// decode packed per-lane (offset,count) within this vector slice
			const uint32_t packed   = col.offsets_counts[vec * N_LANES + (uint32_t)lane_];
			const uint16_t lane_off = (uint16_t)(packed & 0xFFFFu);
			const uint16_t lane_cnt = (uint16_t)(packed >> 16);

			// global base for this vector slice
			const size_t vec_base = col.lane_runs_offsets[vec];
			const size_t base     = vec_base + (size_t)lane_off;

			lane_vals_[v] = col.lane_values + base;
			lane_lens_[v] = col.lane_lengths + base;

			run_cnt_[v] = lane_cnt;
			run_idx_[v] = 0;
			run_pos_[v] = 0;

			run_len_[v] = lane_lens_[v][0];
			run_val_[v] = lane_vals_[v][0];
		}
	}

	__device__ __forceinline__ void rle_expand(T* __restrict__ out) override {
		// each lane for one stride
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			T* __restrict__ outv = out + v * UNPACK_N_VALUES;

			const UINT_T*   vals    = lane_vals_[v];
			const uint16_t* lens    = lane_lens_[v];
			uint16_t        run_cnt = run_cnt_[v];
			uint16_t        run_idx = run_idx_[v];
			uint16_t        run_pos = run_pos_[v];
			uint16_t        run_len = run_len_[v];
			UINT_T          run_val = run_val_[v];

			uint16_t run_end = (uint16_t)(run_pos + run_len);

#pragma unroll
			for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
				const uint16_t k = (uint16_t)(start_k_ + (uint16_t)i);

				// advance to run covering k
				while (k >= run_end && (uint16_t)(run_idx + 1) < run_cnt) {
					run_pos = run_end;
					++run_idx;
					run_len = lens[run_idx];
					run_val = vals[run_idx];
					run_end = (uint16_t)(run_pos + run_len);
				}

				outv[i] = (T)run_val;
			}

			run_idx_[v] = run_idx;
			run_pos_[v] = run_pos;
			run_len_[v] = run_len;
			run_val_[v] = run_val;
		}

		start_k_ = (uint16_t)(start_k_ + (uint16_t)UNPACK_N_VALUES);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct BranchlessCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	uint16_t start_k_ = 0;
	lane_t   lane_;

	const UINT_T* lane_vals_[UNPACK_N_VECTORS];
	uint64_t      lane_mask_[UNPACK_N_VECTORS];

	__device__ __forceinline__
	BranchlessCROSSRLEExpander(const device::CROSSRLELaneMaskColumn<T> col, vi_t first_vec, lane_t lane)
	    : lane_(lane) {
		constexpr uint32_t N_LANES = (uint32_t)utils::get_n_lanes<INT_T>();
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			const uint32_t vec  = (uint32_t)(first_vec + v);
			const uint32_t id   = vec * N_LANES + (uint32_t)lane_;
			const uint32_t base = col.lane_run_base[id];
			lane_vals_[v]       = col.lane_run_values + base;
			lane_mask_[v]       = col.lane_boundary_mask[id];
		}
	}

	__device__ __forceinline__ void rle_expand(T* __restrict__ out) override {
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			T* __restrict__ outv            = out + v * UNPACK_N_VALUES;
			const UINT_T* __restrict__ vals = lane_vals_[v];
			const uint64_t mask             = lane_mask_[v];

#pragma unroll
			for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
				const uint32_t k = (uint32_t)start_k_ + (uint32_t)i;

				// prefix boundaries in [1..k]
				// use 64-bit safe ops
				const unsigned long long prefix = ((unsigned long long)mask & ((1ull << k) - 1ull));

				const uint32_t run_id = (uint32_t)__popcll(prefix);

				outv[i] = (T)vals[run_id];
			}
		}
		start_k_ += (uint16_t)UNPACK_N_VALUES;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct PrefetchBranchlessCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	uint16_t start_k_ = 0;
	lane_t   lane_;

	const UINT_T* lane_vals_[UNPACK_N_VECTORS];
	uint64_t      lane_mask_[UNPACK_N_VECTORS];
	uint16_t      lane_run_len_[UNPACK_N_VECTORS]; // the run count of this (vec,lane) (length of the CSR segment)

	__device__ __forceinline__
	PrefetchBranchlessCROSSRLEExpander(const device::CROSSRLELaneMaskColumn<T> col, vi_t first_vec, lane_t lane)
	    : lane_(lane) {
		constexpr uint32_t N_LANES = (uint32_t)utils::get_n_lanes<INT_T>();
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			const uint32_t vec = (uint32_t)(first_vec + v);
			const uint32_t id  = vec * N_LANES + (uint32_t)lane_;

			const uint32_t base = col.lane_run_base[id];
			const uint32_t end  = col.lane_run_base[id + 1];

			lane_vals_[v]    = col.lane_run_values + base;
			lane_run_len_[v] = (uint16_t)(end - base); // >=1
			lane_mask_[v]    = col.lane_boundary_mask[id];
		}
	}

	__device__ __forceinline__ void rle_expand(T* __restrict__ out) override {
		const uint32_t k0 = (uint32_t)start_k_;

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			T* __restrict__ outv            = out + v * UNPACK_N_VALUES;
			const UINT_T* __restrict__ vals = lane_vals_[v];
			const uint64_t mask             = lane_mask_[v];

			const uint32_t run_len = (uint32_t)lane_run_len_[v];
			const uint32_t last    = run_len - 1u; // run_len >= 1

			// run_id: prefix bits in [0..k-1]

			uint32_t run_id = (uint32_t)__popcll(mask & ((1ull << k0) - 1ull));

			// prefetch cur / next
			UINT_T cur = vals[run_id];

			uint32_t next_id = run_id + 1u;
			next_id          = (next_id < run_len) ? next_id : last;
			UINT_T next      = vals[next_id];

#pragma unroll
			for (uint32_t i = 0; i < (uint32_t)UNPACK_N_VALUES; ++i) {
				outv[i] = (T)cur;

				// assume that pos < 63 ( VALUES_PER_LANE <= 64 and start_k_ within lane)
				const uint32_t inc = (uint32_t)((mask >> (k0 + i)) & 1ull);

				// run_id += inc
				run_id += inc;

				// branchless: if (inc) cur = next;
				const UINT_T m = (UINT_T)0 - (UINT_T)inc;
				cur            = (cur & ~m) | (next & m);

				// branchless prefetch-all: update next = vals[min(run_id+1, last)]
				uint32_t rn = run_id + 1u;
				rn          = (rn < run_len) ? rn : last;
				next        = vals[rn];
			}
		}

		start_k_ += (uint16_t)UNPACK_N_VALUES;
	}
};

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
          typename IndexT,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename UnpackerT,
          typename ColumnT>
struct RLEDecompressor : DecompressorBase<T> {
	UnpackerT unpacker;

	const T*      rle_values;
	const size_t* rle_offsets;
	const IndexT* rsum_bases;
	IndexT        prefix[UNPACK_N_VECTORS];
	const vi_t    base_vector_index;
	const int32_t n_lanes;

	__device__ __forceinline__ RLEDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
	               FFORFunctor<IndexT, UNPACK_N_VECTORS>(column.ffor.bases + vector_index))
	    , rle_values(column.rle_values)
	    , rle_offsets(column.rle_offsets)
	    , rsum_bases(column.rsum_bases)
	    , base_vector_index(vector_index)
	    , n_lanes(utils::get_n_lanes<IndexT>()) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			const IndexT* base_ptr = rsum_bases + (vector_index + v) * n_lanes;
			prefix[v]              = base_ptr[lane];
		}
	}

	void __device__ unpack_next_into(T* __restrict out) {
		IndexT deltas[UNPACK_N_VALUES * UNPACK_N_VECTORS];
		unpacker.unpack_next_into(deltas);

#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			const size_t base_offset = rle_offsets[base_vector_index + v];
			IndexT       cur         = prefix[v];

#pragma unroll
			for (unsigned i = 0; i < UNPACK_N_VALUES; ++i) {
				cur += deltas[i + v * UNPACK_N_VALUES];
				out[i + v * UNPACK_N_VALUES] = rle_values[base_offset + static_cast<size_t>(cur)];
			}
			prefix[v] = cur;
		}
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
	using UINT_T = typename utils::same_width_uint<T>::type;
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
	using UINT_T = typename utils::same_width_uint<T>::type;
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
	using UINT_T = typename utils::same_width_uint<T>::type;
	UnpackerT unpacker;

	__device__ __forceinline__
	DICTShfl32Decompressor(const DICTFFORColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
	               DICTShfl32Functor<T, UNPACK_N_VECTORS>(
	                   column.ffor.bases + vector_index, (const UINT_T* __restrict__)column.keys, column.key_count)) {
	}
	__device__ __forceinline__ void unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename ExpanderT, typename ColumnT>
struct CROSSRLEDecompressor : DecompressorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	ExpanderT                  expander;
	__device__ __forceinline__ CROSSRLEDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : expander(column, vector_index, lane) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		expander.rle_expand(out);
	}
};

}} // namespace flsgpu::device

#endif // FLS_CUH
