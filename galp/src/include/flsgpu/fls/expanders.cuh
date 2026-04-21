// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/fls/expanders.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_FLS_EXPANDERS_CUH
#define FLSGPU_FLS_EXPANDERS_CUH

#include "flsgpu/device-types.cuh"
#include "flsgpu/fls/functors.cuh"
#include "flsgpu/structs.cuh"
#include "flsgpu/utils.cuh"
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace flsgpu { namespace device {

template <typename ValueT, typename IndexT>
struct RLEExpanderBase {
public:
	__device__ __forceinline__ void expand_run_into([[maybe_unused]] ValueT* out) {
	}
	__device__ ~RLEExpanderBase() = default;
};

template <typename ValueT, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyRLEExpander : flsgpu::device::RLEExpanderBase<ValueT, IndexT> {
private:
	const ValueT* rle_values;
	const size_t* rle_offsets;
	const vi_t    base_vector_index;

public:
	__device__ __forceinline__ ValueT decode_value(const size_t base_offset, const IndexT code) const {
		return static_cast<ValueT>(rle_values[base_offset + static_cast<size_t>(code)]);
	}

	__device__ __forceinline__ void expand_codes_into(const IndexT* __restrict codes, ValueT* __restrict out) {
#pragma unroll
		for (unsigned v = 0; v < UNPACK_N_VECTORS; ++v) {
			const size_t base_offset = rle_offsets[base_vector_index + v];
#pragma unroll
			for (unsigned i = 0; i < UNPACK_N_VALUES; ++i) {
				const unsigned idx = i + v * UNPACK_N_VALUES;
				out[idx]           = decode_value(base_offset, codes[idx]);
			}
		}
	}

	void __device__ __forceinline__ expand_run_into(ValueT* out) {
		IndexT codes[UNPACK_N_VECTORS * UNPACK_N_VALUES];
#pragma unroll
		for (unsigned i = 0; i < UNPACK_N_VECTORS * UNPACK_N_VALUES; ++i) {
			codes[i] = static_cast<IndexT>(out[i]);
		}
		expand_codes_into(codes, out);
	}

	__device__ __forceinline__
	DummyRLEExpander(const flsgpu::device::RLEColumn<ValueT, IndexT> column, const vi_t vector_index, const lane_t lane)
	    : rle_values(column.rle_values)
	    , rle_offsets(column.rle_offsets)
	    , base_vector_index(vector_index) {
		(void)lane;
	}
};

template <typename T>
struct CROSSRLEExpanderBase {
public:
	__device__ __forceinline__ void expand_run_into([[maybe_unused]] T* out) {
	}
	__device__ ~CROSSRLEExpanderBase() = default;
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
	void __device__ __forceinline__ expand_run_into(T* out) {
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
struct StatefulCacheCROSSRLEExpander : flsgpu::device::CROSSRLEExpanderBase<T> {
private:
	using UINT_T = typename utils::same_width_uint<T>::type;
	using INT_T  = typename utils::same_width_int<T>::type;

	si_t      start_index = 0;
	uint32_t  vec_base[UNPACK_N_VECTORS];
	UINT_T*   vec_values[UNPACK_N_VECTORS];
	uint32_t* vec_lengths[UNPACK_N_VECTORS];

	// --- persistent per-vector cursor state ---
	uint32_t vec_run_index[UNPACK_N_VECTORS];
	uint32_t vec_run_length[UNPACK_N_VECTORS];
	uint32_t vec_run_end[UNPACK_N_VECTORS];   // current run end position (exclusive) in decompressed space
	UINT_T   vec_run_value[UNPACK_N_VECTORS]; // current run value (cached)

public:
	void __device__ __forceinline__ expand_run_into(T* out) {
		constexpr uint32_t N_LANES_U32 = static_cast<uint32_t>(utils::get_n_lanes<INT_T>());

		// Hoist call-invariant base offset (monotonic across calls)
		const uint32_t call_start_offset = static_cast<uint32_t>(start_index) * N_LANES_U32;

#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			// Local aliases (encourage register caching)
			UINT_T* __restrict__ values    = vec_values[v];
			uint32_t* __restrict__ lengths = vec_lengths[v];

			// Load persistent state into locals
			uint32_t run_index  = vec_run_index[v];
			uint32_t run_length = vec_run_length[v];
			uint32_t run_end    = vec_run_end[v]; // exclusive end of current run
			UINT_T   value      = vec_run_value[v];

			// First output position for this vector in this call
			uint32_t pos = vec_base[v] + call_start_offset;

			// ---- Fast path: this entire unpack chunk stays in current run ----
			// Avoid per-element boundary checks / while-loop and any extra run metadata loads.
			{
				const uint32_t last_pos = pos + static_cast<uint32_t>(UNPACK_N_VALUES - 1) * N_LANES_U32;
				if (last_pos < run_end) {
#pragma unroll
					for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
						out[v * UNPACK_N_VALUES + i] = value;
					}

					// State unchanged except logical progress (start_index is global, run cursor state unchanged)
					vec_run_index[v]  = run_index;
					vec_run_length[v] = run_length;
					vec_run_end[v]    = run_end;
					vec_run_value[v]  = value;
					continue;
				}
			}

			// ---- Slow path: may cross one or more run boundaries ----
#pragma unroll
			for (int i = 0; i < (int)UNPACK_N_VALUES; ++i) {
				// Advance to the correct run only when crossing boundary.
				// Uses cached run_end (exclusive), so no hot-path recompute of run_pos + run_length.
				while (pos >= run_end) {
					++run_index;

					// Assumes lengths fit in uint32_t and runs are contiguous in decompressed space.
					run_length = lengths[run_index];
					run_end += run_length;
					value = values[run_index];
				}

				out[v * UNPACK_N_VALUES + i] = value;
				pos += N_LANES_U32;
			}

			// Store back persistent state
			vec_run_index[v]  = run_index;
			vec_run_length[v] = run_length;
			vec_run_end[v]    = run_end;
			vec_run_value[v]  = value;
		}

		start_index += UNPACK_N_VALUES;
	}

	__device__ __forceinline__ StatefulCacheCROSSRLEExpander(const flsgpu::device::CROSSRLEColumn<T> column,
	                                                         const vi_t                              vector_index,
	                                                         const lane_t                            lane) {
#pragma unroll
		for (int v = 0; v < (int)UNPACK_N_VECTORS; ++v) {
			auto vec_index = vector_index + v;
			auto off       = column.offsets[vec_index];

			vec_values[v]  = column.values + off;
			vec_lengths[v] = column.lengths + off;

			vec_base[v] = static_cast<uint32_t>(vec_index * consts::VALUES_PER_VECTOR) + static_cast<uint32_t>(lane);

			// init cursor state at first run
			vec_run_index[v] = 0;

			// current run [run_start, run_end)
			const uint32_t run_start = column.run_positions[off];
			const uint32_t len0      = static_cast<uint32_t>(vec_lengths[v][0]); // assumes <= UINT32_MAX

			vec_run_length[v] = len0;
			vec_run_end[v]    = run_start + len0;
			vec_run_value[v]  = vec_values[v][0];
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
	void __device__ __forceinline__ expand_run_into(T* out) {
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

	void __device__ __forceinline__ expand_run_into(T* out) {
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
	__device__ __forceinline__ void expand_run_into(T* out) {
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
	__device__ __forceinline__ void expand_run_into(T* __restrict__ out) {
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

	__device__ __forceinline__ void expand_run_into(T* __restrict__ out) {
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

	__device__ __forceinline__ void expand_run_into(T* __restrict__ out) {
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

	__device__ __forceinline__ void expand_run_into(T* __restrict__ out) {
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

}} // namespace flsgpu::device

#endif // FLSGPU_FLS_EXPANDERS_CUH
