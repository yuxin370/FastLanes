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

	__device__ __forceinline__ UINT_T operator()(const UINT_T value, const vi_t vector_index) override {
		return value + bases[vector_index];
	}
};
// todo: considering using share memory to cache keys
template <typename T, unsigned UNPACK_N_VECTORS>
struct DICTFunctor : FunctorBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const UINT_T* __restrict__ 		keys;
  UINT_T 							bases[UNPACK_N_VECTORS];

  __device__ __forceinline__
  DICTFunctor(const UINT_T* a_bases, const UINT_T* a_keys) : keys(a_keys) {
#pragma unroll
    for (int v=0; v<UNPACK_N_VECTORS; ++v) bases[v] = a_bases[v];
  }

  __device__ __forceinline__
  UINT_T operator()(UINT_T value, vi_t vector_index) override {
    const auto idx = value + bases[vector_index];

    // return __ldg(keys + idx);   
    return keys[idx];

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


template <typename T>
struct FREQExceptionPatcherBase {
public:
	__device__ __forceinline__ virtual void fill_and_patch(T* out) = 0;
  	__device__ virtual ~FREQExceptionPatcherBase() = default;
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
	T      		 frequent_value[UNPACK_N_VECTORS];
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
			auto vec_index              = first_vector_index + v;
			exceptions_count[v]         = column.counts[vec_index]; // exceotions count in this vector
			vec_exceptions_positions[v] = column.positions + column.exceptions_offsets[vec_index]; // exceptions_offsets corresponds to offests in exceptions array
			vec_exceptions[v]           = column.exceptions + column.exceptions_offsets[vec_index]; // get the first position/exception in this vector
			frequent_value[v]			= column.frequent_value[vec_index]; // also the first values in this vector
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
	T      		 frequent_value[UNPACK_N_VECTORS];

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
			frequent_value[v]			= column.frequent_value[vec_index];
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
	T      	  frequent_value[UNPACK_N_VECTORS];

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
	T      	  frequent_value[UNPACK_N_VECTORS];

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

			positions[v]  = column.positions + exceptions_offset + lane_offset;
			exceptions[v] = column.exceptions + exceptions_offset + lane_offset;
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
	T      	  frequent_value[UNPACK_N_VECTORS];

public:
	__device__ __forceinline__ PrefetchPositionFREQExceptionPatcher(const FREQExtendedColumn<T> column,
	                                                               const vi_t                 first_vector_index,
	                                                               const lane_t               lane)
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

			next_position[v] = *positions[v];
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
	T      	 frequent_value[UNPACK_N_VECTORS];

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

	__device__ __forceinline__
	PrefetchAllFREQExceptionPatcher(const FREQExtendedColumn<T> column, const vi_t first_vector_index, const lane_t lane)
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
	T      	 frequent_value[UNPACK_N_VECTORS];


public:
	void __device__ __forceinline__ read_next_exception() {
	}

	__device__ __forceinline__ PrefetchAllBranchlessFREQExceptionPatcher(const FREQExtendedColumn<T> column,
	                                                                    const vi_t                 first_vector_index,
	                                                                    const lane_t               lane)
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
	__device__ __forceinline__ virtual void fill_and_patch(T* out) = 0;
  	__device__ virtual ~CROSSRLEExpanderBase() = default;
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T = typename utils::same_width_int<T>::type;

	si_t         start_index = 0;
	UINT_T*    	 vec_values[UNPACK_N_VECTORS];
	size_t*    	 vec_lengths[UNPACK_N_VECTORS];
	uint32_t*	 vec_runs_positions[UNPACK_N_VECTORS]; // the starting position of the first run in each vector

	const lane_t lane;
	

public:
	void __device__ __forceinline__ rle_expand(T* out) override {
		constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

		const int first_pos = start_index * N_LANES + lane;  // this is the lane in corresponding vectors
		const int last_pos  = first_pos + N_LANES * (UNPACK_N_VALUES - 1); // the last lane in corresponding vectors
		
		start_index += UNPACK_N_VALUES; // will traverse the whole vectors (UNPACK_N_VALUES one time, for UNPACK_N_VECTORS vectors)

		// locate corresponding runs and expand it into out
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			size_t  current_run_position = vec_runs_positions[v]; // the starting position in decompressed arrays of the current run
			size_t  run_index            = 0;
			size_t  run_length           = vec_lengths[v][run_index];
			UINT_T  current_value        = vec_values[v][run_index];
#pragma unroll
			for (int i {0}; i < UNPACK_N_VALUES; ++i) {
				const int global_position = first_pos + v * consts::VALUES_PER_VECTOR + i * N_LANES;
				// move to the correct run
				while (global_position >= current_run_position + run_length) {
					current_run_position += run_length;
					++run_index;
					run_length    = vec_lengths[v][run_index];
					current_value = vec_values[v][run_index];
				}
				out[v * UNPACK_N_VALUES + i] = current_value; // not friendly for branch prediction
			}	
		}
	}

	__device__ __forceinline__
	DummyCROSSRLEExpander(const flsgpu::device::CROSSRLEColumn<T> column, const vi_t vector_index, const lane_t lane) 
		: lane (lane) {
	
#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			auto vec_index              = vector_index + v;
			vec_values[v]               = column.values + column.offsets[vec_index];
			vec_lengths[v]              = column.lengths + column.offsets[vec_index];
			vec_runs_positions[v]       = column.run_positions + column.offsets[vec_index]; 
		}
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

template <typename T, unsigned UNPACK_N_VECTORS, typename PatcherT, typename ColumnT>
struct FREQDecompressor : DecompressorBase<T> {
	PatcherT  					patcher;
	__device__ __forceinline__ FREQDecompressor(const ColumnT column, const vi_t vector_index, const lane_t lane)
	    : patcher(PatcherT(column, vector_index, lane)){
	}

	void __device__ unpack_next_into(T* __restrict out) {
		patcher.fill_and_patch(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename UnpackerT, typename ColumnT>
struct DICTDecompressor : DecompressorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	UnpackerT 					unpacker;
	__device__ __forceinline__ DICTDecompressor(const DICTColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
	               lane,
	               column.ffor.bp.bit_widths[vector_index],
	               DICTFunctor<T, UNPACK_N_VECTORS>(column.ffor.bases + vector_index,
	                                               column.keys)){ // column.keys at global memory
	}

	// outer key pointer, which may points to shared memory
	__device__ __forceinline__ DICTDecompressor(const DICTColumn<T> column, const vi_t vector_index, const lane_t lane, const UINT_T* __restrict keys_ptr)
		: unpacker(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vector_index],
					lane,
					column.ffor.bp.bit_widths[vector_index],
					DICTFunctor<T, UNPACK_N_VECTORS>(column.ffor.bases + vector_index,
													keys_ptr)) {
	}

	void __device__ unpack_next_into(T* __restrict out) {
		unpacker.unpack_next_into(out);
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, typename ExaanderT, typename ColumnT>
struct CROSSRLEDecompressor : DecompressorBase<T> {
	using UINT_T = typename utils::same_width_uint<T>::type;
	ExaanderT 					exaander;
	__device__ __forceinline__ CROSSRLEDecompressor(const CROSSRLEColumn<T> column, const vi_t vector_index, const lane_t lane)
	    : exaander(ExaanderT(column, vector_index, lane)){ 
	}

	void __device__ unpack_next_into(T* __restrict out) {
		exaander.unpack_next_into(out);
	}
};

}} // namespace flsgpu::device

#endif // FLS_CUH
