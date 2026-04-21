// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/fls/patchers.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_FLS_PATCHERS_CUH
#define FLSGPU_FLS_PATCHERS_CUH

#include "flsgpu/device-types.cuh"
#include "flsgpu/fls/functors.cuh"
#include "flsgpu/structs.cuh"
#include "flsgpu/utils.cuh"
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace flsgpu { namespace device {
template <typename T>
struct FREQExceptionPatcherBase {
public:
	__device__ __forceinline__ void fill_and_patch([[maybe_unused]] T* out) {
	}
	__device__ ~FREQExceptionPatcherBase() = default;
};

template <typename T>
struct SLPATCHExceptionPatcherBase {
public:
	__device__ __forceinline__ void patch([[maybe_unused]] T* out) {
	}
	__device__ ~SLPATCHExceptionPatcherBase() = default;
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyFREQExceptionPatcher : flsgpu::device::FREQExceptionPatcherBase<T> {

public:
	void __device__ __forceinline__ fill_and_patch(T* out) {
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
	void __device__ __forceinline__ fill_and_patch(T* out) {
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
			    column.positions_offsets[vec_index]; // position offsets correspond to positions segment layout
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
	void __device__ __forceinline__ fill_and_patch(T* out) {
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
			vec_exceptions_positions[v] = column.positions + column.positions_offsets[vec_index];
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
	void __device__ __forceinline__ patch(T* out) {
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
			vec_exceptions_positions[v] = column.positions + column.positions_offsets[vec_index];
			vec_exceptions[v]           = column.exceptions + column.exceptions_offsets[vec_index];
		}
	}
};

template <typename T, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulSLPATCHDictExceptionPatcher : SLPATCHExceptionPatcherBase<T> {
	using INT_T = typename utils::same_width_int<IndexT>::type;
	using KEY_T = typename utils::same_width_uint<T>::type;

	si_t                               start_index = 0;
	uint16_t                           exceptions_count[UNPACK_N_VECTORS];
	uint16_t*                          vec_exceptions_positions[UNPACK_N_VECTORS];
	IndexT*                            vec_exceptions[UNPACK_N_VECTORS];
	DICTIndexFunctor<T, IndexT, KEY_T> processor;
	const lane_t                       lane;
	int32_t                            exception_index[UNPACK_N_VECTORS] = {0};

public:
	void __device__ __forceinline__ patch(T* out) {
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
			vec_exceptions_positions[v] = column.index.positions + column.index.positions_offsets[vec_index];
			vec_exceptions[v]           = column.index.exceptions + column.index.exceptions_offsets[vec_index];
		}
	}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatelessSLPATCHExceptionPatcher : SLPATCHExceptionPatcherBase<T> {
	using INT_T = typename utils::same_width_int<T>::type;

	si_t         start_index = 0;
	uint16_t     exceptions_count[UNPACK_N_VECTORS];
	uint16_t*    vec_exceptions_positions[UNPACK_N_VECTORS];
	T*           vec_exceptions[UNPACK_N_VECTORS];
	const lane_t lane;

public:
	void __device__ __forceinline__ patch(T* out) {
		constexpr auto N_LANES   = utils::get_n_lanes<INT_T>();
		const int      first_pos = start_index * N_LANES + lane;
		const int      last_pos  = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
		start_index += UNPACK_N_VALUES;

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			for (int i {0}; i < exceptions_count[v]; ++i) {
				const auto position = vec_exceptions_positions[v][i];
				if (position >= first_pos && position <= last_pos && position % N_LANES == lane) {
					out[(position - first_pos) / N_LANES + v * UNPACK_N_VALUES] = vec_exceptions[v][i];
				}
			}
		}
	}

	__device__ __forceinline__
	StatelessSLPATCHExceptionPatcher(const SLPATCHColumn<T> column, const vi_t first_vector_index, const lane_t lane)
	    : lane(lane) {

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			const auto vec_index        = first_vector_index + v;
			exceptions_count[v]         = column.counts[vec_index];
			vec_exceptions_positions[v] = column.positions + column.positions_offsets[vec_index];
			vec_exceptions[v]           = column.exceptions + column.exceptions_offsets[vec_index];
		}
	}
};

template <typename T, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatelessSLPATCHDictExceptionPatcher : SLPATCHExceptionPatcherBase<T> {
	using INT_T = typename utils::same_width_int<IndexT>::type;
	using KEY_T = typename utils::same_width_uint<T>::type;

	si_t                               start_index = 0;
	uint16_t                           exceptions_count[UNPACK_N_VECTORS];
	uint16_t*                          vec_exceptions_positions[UNPACK_N_VECTORS];
	IndexT*                            vec_exceptions[UNPACK_N_VECTORS];
	DICTIndexFunctor<T, IndexT, KEY_T> processor;
	const lane_t                       lane;

public:
	void __device__ __forceinline__ patch(T* out) {
		constexpr auto N_LANES   = utils::get_n_lanes<INT_T>();
		const int      first_pos = start_index * N_LANES + lane;
		const int      last_pos  = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
		start_index += UNPACK_N_VALUES;

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			for (int i {0}; i < exceptions_count[v]; ++i) {
				const auto position = vec_exceptions_positions[v][i];
				if (position >= first_pos && position <= last_pos && position % N_LANES == lane) {
					out[(position - first_pos) / N_LANES + v * UNPACK_N_VALUES] = processor(vec_exceptions[v][i], v);
				}
			}
		}
	}

	__device__ __forceinline__ StatelessSLPATCHDictExceptionPatcher(const DICTSLPATCHColumn<T, IndexT> column,
	                                                                const vi_t   first_vector_index,
	                                                                const lane_t lane)
	    : processor(column.keys)
	    , lane(lane) {

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
			const auto vec_index        = first_vector_index + v;
			exceptions_count[v]         = column.index.counts[vec_index];
			vec_exceptions_positions[v] = column.index.positions + column.index.positions_offsets[vec_index];
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

	void __device__ __forceinline__ fill_and_patch(T* out) {
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

	void __device__ __forceinline__ fill_and_patch(T* out) {
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

	void __device__ __forceinline__ fill_and_patch(T* out) {
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

	void __device__ __forceinline__ fill_and_patch(T* out) {
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

	void __device__ __forceinline__ fill_and_patch(T* out) {
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

}} // namespace flsgpu::device

#endif // FLSGPU_FLS_PATCHERS_CUH
