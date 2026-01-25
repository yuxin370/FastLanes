// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/structs.cuh
// ────────────────────────────────────────────────────────
#ifndef STRUCTS_CUH
#define STRUCTS_CUH

#include "alp.hpp"
#include "device-types.cuh"
#include "host-utils.cuh"
#include "utils.cuh"
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace flsgpu {

namespace device {

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

template <typename T>
struct BPColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	UINT_T* packed_array;
	vbw_t*  bit_widths;
	size_t* vector_offsets;
};

template <typename T>
struct FFORColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t      n_values;
	BPColumn<T> bp;
	UINT_T*     bases;
};

template <typename T>
struct DICTColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t             n_values;
	FFORColumn<UINT_T> ffor; // index stream (FFOR-compressed)

	UINT_T* keys;      // dictionary keys (shared by all vectors)
	size_t  key_count; // number of keys in dictionary
};

template <typename T>
struct CROSSRLEColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	size_t    n_runs; // number of runs : is this needed?
	UINT_T*   values;
	uint32_t* lengths;
	uint32_t* offsets;       //  each vector's start run idx
	uint32_t* run_positions; // runs' start position (offset) in decompressed array
};

template <typename T>
struct CROSSRLEExtendedColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;

	size_t n_values;
	size_t n_vecs;

	size_t    n_lane_runs;       // total runs after lane-projection
	size_t*   lane_runs_offsets; // [n_vecs+1], global offset into lane_values/lane_lengths
	UINT_T*   lane_values;       // [n_lane_runs]
	uint16_t* lane_lengths;      // [n_lane_runs], length in k-space (stride index)
	uint32_t* offsets_counts;    // [n_vecs * N_LANES], packed per-lane (offset,count) within vector slice
};

template <typename T>
struct CROSSRLELaneMaskColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;

	size_t n_values;
	size_t n_vecs;

	size_t    n_lane_runs;        // total runs across all (vec,lane)
	uint32_t* lane_run_base;      // [n_vecs*N_LANES + 1]  CSR base
	UINT_T*   lane_run_values;    // [n_lane_runs]
	uint64_t* lane_boundary_mask; // [n_vecs*N_LANES] bit k=1 => new run begins at k
};

template <typename T>
struct FREQColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	T* frequent_value; // frequent values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* counts;             // number of exceptions per vector
};

template <typename T>
struct FREQExtendedColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	T* frequent_value; // frequent values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* offsets_counts;     // offsets and counts per lane
};

template <typename T>
struct ALPColumn {
	using INT_T  = typename utils::same_width_int<T>::type;
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t             n_values;
	FFORColumn<UINT_T> ffor;

	INT_T*   factors;
	T*       fractions;
	uint8_t* factor_indices;
	uint8_t* fraction_indices;

	size_t    n_exceptions;
	size_t*   exceptions_offsets;
	T*        exceptions;
	uint16_t* positions;
	uint16_t* counts;
};

template <typename T>
struct ALPExtendedColumn {
	using INT_T  = typename utils::same_width_int<T>::type;
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t             n_values;
	FFORColumn<UINT_T> ffor;

	INT_T*   factors;
	T*       fractions;
	uint8_t* factor_indices;
	uint8_t* fraction_indices;

	size_t    n_exceptions;
	size_t*   exceptions_offsets;
	T*        exceptions;
	uint16_t* positions;
	uint16_t* offsets_counts;
};
} // namespace device

namespace host {

template <typename T>
struct BPColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::BPColumn<T>;

	size_t n_values;
	size_t n_packed_values;

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}

	UINT_T* packed_array;
	vbw_t*  bit_widths;
	size_t* vector_offsets;

	device::BPColumn<T> copy_to_device() const {
		const size_t branchless_extra_access_buffer = sizeof(T) * utils::get_n_lanes<T>() * 4;
		return device::BPColumn<T> {
		    n_values,
		    get_n_vecs(),
		    GPUArray<UINT_T>(n_packed_values, branchless_extra_access_buffer, packed_array).release(),
		    GPUArray<vbw_t>(get_n_vecs(), bit_widths).release(),
		    GPUArray<size_t>(get_n_vecs(), vector_offsets).release()};
	}
};

template <typename T>
struct FFORColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FFORColumn<T>;

	BPColumn<T> bp;
	UINT_T*     bases;

	size_t get_n_values() const {
		return bp.n_values;
	}
	size_t get_n_vecs() const {
		return bp.get_n_vecs();
	}

	device::FFORColumn<T> copy_to_device() const {
		return device::FFORColumn<T> {
		    get_n_values(), bp.copy_to_device(), GPUArray<UINT_T>(bp.get_n_vecs(), bases).release()};
	}
};

template <typename T>
struct CROSSRLEExtendedColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::CROSSRLEExtendedColumn<T>;

	size_t n_values;

	size_t    n_lane_runs;       // total runs after lane-projection
	size_t*   lane_runs_offsets; // [n_vecs+1]
	UINT_T*   lane_values;       // [n_lane_runs]
	uint16_t* lane_lengths;      // [n_lane_runs]
	uint32_t* offsets_counts;    // [n_vecs * N_LANES]

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}

	device::CROSSRLEExtendedColumn<T> copy_to_device() const {
		const size_t n_vecs = get_n_vecs();

		return device::CROSSRLEExtendedColumn<T> {
		    n_values,
		    n_vecs,
		    n_lane_runs,
		    GPUArray<size_t>(n_vecs + 1, lane_runs_offsets).release(),
		    GPUArray<UINT_T>(n_lane_runs, lane_values).release(),
		    GPUArray<uint16_t>(n_lane_runs, lane_lengths).release(),
		    GPUArray<uint32_t>(n_vecs * utils::get_n_lanes<T>(), offsets_counts).release(),
		};
	}
};

template <typename T>
struct CROSSRLELaneMaskColumn {

	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::CROSSRLELaneMaskColumn<T>;

	size_t n_values;

	size_t    n_lane_runs;        // total runs across all (vec,lane)
	uint32_t* lane_run_base;      // [n_vecs*N_LANES + 1]  CSR base
	UINT_T*   lane_run_values;    // [n_lane_runs]
	uint64_t* lane_boundary_mask; // [n_vecs*N_LANES] bit k=1 => new run begins at k

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}

	device::CROSSRLELaneMaskColumn<T> copy_to_device() const {
		const size_t n_vecs     = get_n_vecs();
		const size_t total_runs = n_vecs * utils::get_n_lanes<T>();
		return device::CROSSRLELaneMaskColumn<T> {
		    get_n_values(),
		    n_vecs,
		    n_lane_runs,
		    GPUArray<uint32_t>(total_runs + 1, lane_run_base).release(),
		    GPUArray<UINT_T>(n_lane_runs, lane_run_values).release(),
		    GPUArray<uint64_t>(total_runs, lane_boundary_mask).release(),
		};
	}
};

template <typename T>
struct CROSSRLEColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::CROSSRLEColumn<T>;
	size_t n_values;

	size_t    n_runs; // number of runs : is this needed?
	UINT_T*   values;
	uint32_t* lengths;
	uint32_t* offsets;       //  each vector's start run idx
	uint32_t* run_positions; // runs' start position (offset) in decompressed array

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}

	device::CROSSRLEColumn<T> copy_to_device() const {
		const size_t n_vecs = get_n_vecs();

		return device::CROSSRLEColumn<T> {
		    get_n_values(),
		    n_vecs,
		    n_runs,
		    GPUArray<UINT_T>(n_runs, values).release(),
		    GPUArray<uint32_t>(n_runs, lengths).release(),
		    GPUArray<uint32_t>(n_vecs + 1, offsets).release(),
		    GPUArray<uint32_t>(n_runs, run_positions).release(),
		};
	}

	CROSSRLELaneMaskColumn<T> create_lane_mask_column() const {
		constexpr uint32_t N_LANES         = (uint32_t)utils::get_n_lanes<T>();
		constexpr uint32_t VALUES_PER_LANE = (uint32_t)utils::get_values_per_lane<T>();
		constexpr uint32_t VEC_VALUES      = (uint32_t)consts::VALUES_PER_VECTOR;

		static_assert(VALUES_PER_LANE <= 64, "lane_boundary_mask uses uint64_t; extend if >64");
		static_assert(VEC_VALUES == N_LANES * VALUES_PER_LANE, "expect VALUES_PER_VECTOR == N_LANES*VALUES_PER_LANE");

		const size_t n_vecs   = get_n_vecs();
		const size_t n_blocks = n_vecs * (size_t)N_LANES;

		// 第一遍：统计 run_cnt 并生成 CSR base
		auto* lane_run_cnt       = new uint16_t[n_blocks];
		auto* lane_run_base      = new uint32_t[n_blocks + 1];
		auto* lane_boundary_mask = new uint64_t[n_blocks];

		for (size_t i = 0; i < n_blocks; ++i) {
			lane_run_cnt[i]       = 0;
			lane_boundary_mask[i] = 0ull;
		}

		for (size_t vec = 0; vec < n_vecs; ++vec) {
			UINT_T tmp[VEC_VALUES];
#pragma unroll
			for (uint32_t i = 0; i < VEC_VALUES; ++i)
				tmp[i] = (UINT_T)0;

			const uint32_t vec_base = (uint32_t)(vec * (size_t)VEC_VALUES);
			const uint32_t r0       = offsets[vec];
			const uint32_t r1       = offsets[vec + 1];

			for (uint32_t r = r0; r < r1; ++r) {
				const uint32_t run_start_g = run_positions[r];
				const uint32_t run_len     = lengths[r];

				if (run_start_g + run_len <= vec_base)
					continue;
				if (run_start_g >= vec_base + VEC_VALUES)
					continue;

				uint32_t local_start = (run_start_g > vec_base) ? (run_start_g - vec_base) : 0u;
				uint32_t local_end   = run_start_g + run_len - vec_base;
				if (local_end > VEC_VALUES)
					local_end = VEC_VALUES;

				const UINT_T v = values[r];
				for (uint32_t p = local_start; p < local_end; ++p)
					tmp[p] = v;
			}

			for (uint32_t lane = 0; lane < N_LANES; ++lane) {
				uint64_t mask    = 0ull;
				UINT_T   prev    = tmp[lane];
				uint16_t run_cnt = 1;

				for (uint32_t k = 1; k < VALUES_PER_LANE; ++k) {
					const UINT_T   cur     = tmp[lane + k * N_LANES];
					const uint32_t changed = (cur != prev) ? 1u : 0u;

					// IMPORTANT: k can be >= 32, must shift in 64-bit
					mask |= (uint64_t(changed) << k);

					run_cnt += (uint16_t)changed;
					prev = cur;
				}

				const size_t id        = vec * (size_t)N_LANES + (size_t)lane;
				lane_boundary_mask[id] = mask;
				lane_run_cnt[id]       = run_cnt;
			}
		}

		lane_run_base[0]    = 0;
		uint32_t total_runs = 0;
		for (size_t id = 0; id < n_blocks; ++id) {
			total_runs += (uint32_t)lane_run_cnt[id];
			lane_run_base[id + 1] = total_runs;
		}

		// sencond round: lane_run_values
		auto* lane_run_values = (total_runs ? new UINT_T[(size_t)total_runs] : nullptr);

		for (size_t vec = 0; vec < n_vecs; ++vec) {
			UINT_T tmp[VEC_VALUES];
#pragma unroll
			for (uint32_t i = 0; i < VEC_VALUES; ++i)
				tmp[i] = (UINT_T)0;

			const uint32_t vec_base = (uint32_t)(vec * (size_t)VEC_VALUES);
			const uint32_t r0       = offsets[vec];
			const uint32_t r1       = offsets[vec + 1];

			for (uint32_t r = r0; r < r1; ++r) {
				const uint32_t run_start_g = run_positions[r];
				const uint32_t run_len     = lengths[r];

				if (run_start_g + run_len <= vec_base)
					continue;
				if (run_start_g >= vec_base + VEC_VALUES)
					continue;

				uint32_t local_start = (run_start_g > vec_base) ? (run_start_g - vec_base) : 0u;
				uint32_t local_end   = run_start_g + run_len - vec_base;
				if (local_end > VEC_VALUES)
					local_end = VEC_VALUES;

				const UINT_T v = values[r];
				for (uint32_t p = local_start; p < local_end; ++p)
					tmp[p] = v;
			}

			for (uint32_t lane = 0; lane < N_LANES; ++lane) {
				const size_t   id      = vec * (size_t)N_LANES + (size_t)lane;
				const uint32_t base    = lane_run_base[id];
				const uint16_t run_cnt = lane_run_cnt[id];

				uint32_t cursor           = base;
				UINT_T   cur_val          = tmp[lane];
				lane_run_values[cursor++] = cur_val;

				for (uint32_t k = 1; k < VALUES_PER_LANE; ++k) {
					const UINT_T v = tmp[lane + k * N_LANES];
					if (v != cur_val) {
						cur_val                   = v;
						lane_run_values[cursor++] = cur_val;
					}
				}

				(void)run_cnt;
			}
		}

		delete[] lane_run_cnt;

		return CROSSRLELaneMaskColumn<T> {n_values,
		                                  (size_t)total_runs, // n_lane_runs
		                                  lane_run_base,
		                                  lane_run_values,
		                                  lane_boundary_mask};
	}

	CROSSRLEExtendedColumn<T> create_extended_column() const {
		auto [e_lane_runs_offsets, e_values, e_lengths, e_offsets_counts, e_total_runs] =
		    convert_runs_to_lane_divided_format();
		return CROSSRLEExtendedColumn<T> {
		    n_values, e_total_runs, e_lane_runs_offsets, e_values, e_lengths, e_offsets_counts};
	}

	std::tuple<size_t*, UINT_T*, uint16_t*, uint32_t*, size_t> convert_runs_to_lane_divided_format() const {
		constexpr uint32_t N_LANES         = (uint32_t)utils::get_n_lanes<T>();
		constexpr uint32_t VALUES_PER_LANE = (uint32_t)utils::get_values_per_lane<T>();
		constexpr uint32_t VEC_VALUES      = (uint32_t)consts::VALUES_PER_VECTOR;

		const size_t n_vecs = get_n_vecs();

		// count run counts of each (vec,lane), also, count the total vec count
		auto* lane_run_counts = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * n_vecs * N_LANES));
		auto* vec_total_runs  = reinterpret_cast<size_t*>(malloc(sizeof(size_t) * n_vecs));

		for (size_t vec = 0; vec < n_vecs; ++vec) {

			UINT_T tmp[VEC_VALUES];

			const uint32_t vec_base = (uint32_t)(vec * (size_t)VEC_VALUES);
			const uint32_t r0       = offsets[vec];
			const uint32_t r1       = offsets[vec + 1];

			for (uint32_t r = r0; r < r1; ++r) {
				const uint32_t run_start_g = run_positions[r];
				const uint32_t run_len     = lengths[r];

				if (run_start_g + run_len <= vec_base)
					continue;
				if (run_start_g >= vec_base + VEC_VALUES)
					continue;

				uint32_t local_start = (run_start_g > vec_base) ? (run_start_g - vec_base) : 0u;
				uint32_t local_end   = run_start_g + run_len - vec_base;
				if (local_end > VEC_VALUES)
					local_end = VEC_VALUES;

				const UINT_T v = values[r];
				for (uint32_t p = local_start; p < local_end; ++p)
					tmp[p] = v;
			}

			size_t vec_runs = 0;

			for (uint32_t lane = 0; lane < N_LANES; ++lane) {
				// k=0..VALUES_PER_LANE-1
				UINT_T   prev    = tmp[lane];
				uint16_t run_cnt = 1;

				for (uint32_t k = 1; k < VALUES_PER_LANE; ++k) {
					const UINT_T cur = tmp[lane + k * N_LANES];
					run_cnt += (cur != prev);
					prev = cur;
				}

				lane_run_counts[vec * N_LANES + lane] = run_cnt;
				vec_runs += (size_t)run_cnt;
			}

			vec_total_runs[vec] = vec_runs;
		}

		// get lane_runs_offsets, and calculate total runs
		auto*  lane_runs_offsets = reinterpret_cast<size_t*>(malloc(sizeof(size_t) * (n_vecs + 1)));
		size_t total_runs        = 0;
		lane_runs_offsets[0]     = 0;
		for (size_t vec = 0; vec < n_vecs; ++vec) {
			total_runs += vec_total_runs[vec];
			lane_runs_offsets[vec + 1] = total_runs;
		}

		auto* out_values         = reinterpret_cast<UINT_T*>(malloc(sizeof(UINT_T) * total_runs));
		auto* out_lengths        = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * total_runs));
		auto* out_offsets_counts = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * n_vecs * N_LANES));

		// second round: write lane-run stream + offsets_counts
		for (size_t vec = 0; vec < n_vecs; ++vec) {
			UINT_T tmp[VEC_VALUES];
			for (uint32_t i = 0; i < VEC_VALUES; ++i)
				tmp[i] = (UINT_T)0;

			const uint32_t vec_base = (uint32_t)(vec * (size_t)VEC_VALUES);
			const uint32_t r0       = offsets[vec];
			const uint32_t r1       = offsets[vec + 1];

			for (uint32_t r = r0; r < r1; ++r) {
				const uint32_t run_start_g = run_positions[r];
				const uint32_t run_len     = lengths[r];

				if (run_start_g + run_len <= vec_base)
					continue;
				if (run_start_g >= vec_base + VEC_VALUES)
					continue;

				uint32_t local_start = (run_start_g > vec_base) ? (run_start_g - vec_base) : 0u;
				uint32_t local_end   = run_start_g + run_len - vec_base;
				if (local_end > VEC_VALUES)
					local_end = VEC_VALUES;

				const UINT_T v = values[r];
				for (uint32_t p = local_start; p < local_end; ++p)
					tmp[p] = v;
			}

			const size_t vec_out_base = lane_runs_offsets[vec];
			size_t       cursor       = vec_out_base;

			for (uint32_t lane = 0; lane < N_LANES; ++lane) {
				const uint16_t run_cnt = lane_run_counts[vec * N_LANES + lane];

				const uint16_t lane_off = (uint16_t)(cursor - vec_out_base);

				// pack: [31:16]=count, [15:0]=offset
				out_offsets_counts[vec * N_LANES + lane] = (uint32_t(run_cnt) << 16) | uint32_t(lane_off);

				// write RLE of this lane（在 k-space）
				UINT_T   cur_val = tmp[lane];
				uint16_t cur_len = 1;

				for (uint32_t k = 1; k < VALUES_PER_LANE; ++k) {
					const UINT_T v = tmp[lane + k * N_LANES];
					if (v == cur_val) {
						++cur_len;
					} else {
						out_values[cursor]  = cur_val;
						out_lengths[cursor] = cur_len;
						++cursor;
						cur_val = v;
						cur_len = 1;
					}
				}
				out_values[cursor]  = cur_val;
				out_lengths[cursor] = cur_len;
				++cursor;
			}

			// if (cursor != lane_runs_offsets[vec + 1]) { /* error */ }
		}

		free(lane_run_counts);
		free(vec_total_runs);

		return std::make_tuple(lane_runs_offsets, out_values, out_lengths, out_offsets_counts, total_runs);
	}
};

template <typename T>
struct DICTColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::DICTColumn<T>;

	FFORColumn<UINT_T> ffor;      // index stream
	UINT_T*            keys;      // host dictionary keys
	size_t             key_count; // number of keys

	size_t get_n_values() const {
		return ffor.bp.n_values;
	}
	size_t get_n_vecs() const {
		return ffor.bp.get_n_vecs();
	}

	device::DICTColumn<T> copy_to_device() const {
		return device::DICTColumn<T> {
		    get_n_values(), ffor.copy_to_device(), GPUArray<UINT_T>(key_count, keys).release(), key_count};
	}
};

template <typename T>
struct FREQExtendedColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FREQExtendedColumn<T>;
	size_t n_values;

	T*        frequent_value;     // frequent values
	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* offsets_counts;     // offsets and counts per lane

	size_t get_n_values() const {
		return n_values;
	}

	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}

	device::FREQExtendedColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = consts::MAX_UNPACK_N_VECS;
		return device::FREQExtendedColumn<T> {
		    n_values,
		    get_n_vecs(),
		    GPUArray<T>(get_n_vecs(), frequent_value).release(),
		    n_exceptions,
		    GPUArray<size_t>(get_n_vecs(), exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(get_n_vecs() * utils::get_n_lanes<T>(), offsets_counts).release(),
		};
	}
};

template <typename T>
struct FREQColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FREQColumn<T>;
	size_t n_values;

	T*        frequent_value;     // frequent values
	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* counts;             // number of exceptions per vector

	size_t get_n_values() const {
		return n_values;
	}

	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}

	device::FREQColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = consts::MAX_UNPACK_N_VECS;
		return device::FREQColumn<T> {
		    n_values,
		    get_n_vecs(),
		    GPUArray<T>(get_n_vecs(), frequent_value).release(),
		    n_exceptions,
		    GPUArray<size_t>(get_n_vecs(), exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(get_n_vecs(), counts).release(),
		};
	}

	std::tuple<T*, uint16_t*, uint16_t*> convert_exceptions_to_lane_divided_format() const {
		constexpr auto N_LANES         = utils::get_n_lanes<T>();
		constexpr auto VALUES_PER_LANE = utils::get_values_per_lane<T>();

		// New exception allocations
		T*        out_exceptions     = reinterpret_cast<T*>(malloc(sizeof(T) * n_exceptions));
		uint16_t* out_positions      = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * n_exceptions));
		uint16_t* out_offsets_counts = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * get_n_vecs() * N_LANES));

		// Intermediate arrays for reordering positions and exceptions
		T        vec_exceptions[consts::VALUES_PER_VECTOR];
		T        vec_exceptions_positions[consts::VALUES_PER_VECTOR];
		uint16_t lane_counts[N_LANES];

		// Copies of pointers for pointer arithmetic
		T*        c_exceptions         = exceptions;
		uint16_t* c_positions          = positions;
		T*        c_out_exceptions     = out_exceptions;
		uint16_t* c_out_positions      = out_positions;
		uint16_t* c_out_offsets_counts = out_offsets_counts;

		for (size_t vec_index {0}; vec_index < get_n_vecs(); ++vec_index) {
			uint32_t vec_exception_count = counts[vec_index];

			// Reset counts
			for (size_t j {0}; j < N_LANES; ++j) {
				lane_counts[j] = 0;
			}

			// Split all exceptions into lanes
			for (size_t exception_index {0}; exception_index < vec_exception_count; ++exception_index) {
				T        exception = c_exceptions[exception_index];
				uint16_t position  = c_positions[exception_index];

				uint32_t lane                 = position % N_LANES;
				uint32_t lane_exception_count = lane_counts[lane];
				++lane_counts[lane];
				vec_exceptions[lane * VALUES_PER_LANE + lane_exception_count]           = exception;
				vec_exceptions_positions[lane * VALUES_PER_LANE + lane_exception_count] = position;
			}

			// Merge and concatenate all exceptions per lane into single contiguous
			// array
			uint32_t vec_exceptions_counter = 0;
			for (size_t lane {0}; lane < N_LANES; ++lane) {
				uint32_t exc_in_lane_count = lane_counts[lane];
				for (size_t exc_in_lane {0}; exc_in_lane < exc_in_lane_count; ++exc_in_lane) {

					c_out_exceptions[vec_exceptions_counter] = vec_exceptions[lane * VALUES_PER_LANE + exc_in_lane];
					c_out_positions[vec_exceptions_counter] =
					    vec_exceptions_positions[lane * VALUES_PER_LANE + exc_in_lane];
					++vec_exceptions_counter;
				}

				c_out_offsets_counts[lane] = (exc_in_lane_count << 10) | (vec_exceptions_counter - exc_in_lane_count);
			}

			c_exceptions += vec_exception_count;
			c_positions += vec_exception_count;
			c_out_exceptions += vec_exception_count;
			c_out_positions += vec_exception_count;
			c_out_offsets_counts += utils::get_n_lanes<T>();
		}

		return std::make_tuple(out_exceptions, out_positions, out_offsets_counts);
	}

	FREQExtendedColumn<T> create_extended_column() const {
		auto [e_exceptions, e_positions, e_offsets_counts] = convert_exceptions_to_lane_divided_format();
		return FREQExtendedColumn<T> {n_values,
		                              utils::copy_array(frequent_value, get_n_vecs()),
		                              n_exceptions,
		                              utils::copy_array(exceptions_offsets, get_n_vecs()),
		                              e_exceptions,
		                              e_positions,
		                              e_offsets_counts};
	}
};

template <typename T>
struct ALPExtendedColumn {
	using INT_T         = typename utils::same_width_int<T>::type;
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::ALPExtendedColumn<T>;

	FFORColumn<UINT_T> ffor;

	uint8_t* factor_indices;
	uint8_t* fraction_indices;

	size_t    n_exceptions;
	size_t*   exceptions_offsets;
	T*        exceptions;
	uint16_t* positions;
	uint16_t* offsets_counts;

	size_t compressed_size_bytes_alp_extended;

	size_t get_n_values() const {
		return ffor.bp.n_values;
	}
	size_t get_n_vecs() const {
		return ffor.bp.get_n_vecs();
	}

	double get_compression_ratio() const {
		return static_cast<double>(ffor.bp.n_values * sizeof(T)) /
		       static_cast<double>(compressed_size_bytes_alp_extended);
	}

	device::ALPExtendedColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = consts::MAX_UNPACK_N_VECS;
		return device::ALPExtendedColumn<T> {
		    get_n_values(),
		    ffor.copy_to_device(),
		    GPUArray<INT_T>(consts::as<T>::FACT_ARR_COUNT, alp::Constants<T>::FACT_ARR.data()).release(),
		    GPUArray<T>(consts::as<T>::FRAC_ARR_COUNT, alp::Constants<T>::FRAC_ARR.data()).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), factor_indices).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), fraction_indices).release(),
		    n_exceptions,
		    GPUArray<size_t>(ffor.bp.get_n_vecs(), exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(ffor.bp.get_n_vecs() * utils::get_n_lanes<T>(), offsets_counts).release(),
		};
	}
};

template <typename T>
struct ALPColumn {
	using INT_T         = typename utils::same_width_int<T>::type;
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::ALPColumn<T>;

	FFORColumn<UINT_T> ffor;

	uint8_t* factor_indices;
	uint8_t* fraction_indices;

	size_t    n_exceptions;
	size_t*   exceptions_offsets;
	T*        exceptions;
	uint16_t* positions;
	uint16_t* counts;

	size_t compressed_size_bytes_alp;
	size_t compressed_size_bytes_alp_extended;

	size_t get_n_values() const {
		return ffor.bp.n_values;
	}
	size_t get_n_vecs() const {
		return ffor.bp.get_n_vecs();
	}

	double get_compression_ratio() const {
		return static_cast<double>(ffor.bp.n_values * sizeof(T)) / static_cast<double>(compressed_size_bytes_alp);
	}

	device::ALPColumn<T> copy_to_device() const {
		return device::ALPColumn<T> {
		    get_n_values(),
		    ffor.copy_to_device(),
		    GPUArray<INT_T>(consts::as<T>::FACT_ARR_COUNT, alp::Constants<T>::FACT_ARR.data()).release(),
		    GPUArray<T>(consts::as<T>::FRAC_ARR_COUNT, alp::Constants<T>::FRAC_ARR.data()).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), factor_indices).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), fraction_indices).release(),
		    n_exceptions,
		    GPUArray<size_t>(ffor.bp.get_n_vecs(), exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, positions).release(),
		    GPUArray<uint16_t>(ffor.bp.get_n_vecs(), counts).release(),
		};
	}

	std::tuple<T*, uint16_t*, uint16_t*> convert_exceptions_to_lane_divided_format() const {
		constexpr auto N_LANES         = utils::get_n_lanes<T>();
		constexpr auto VALUES_PER_LANE = utils::get_values_per_lane<T>();

		// New exception allocations
		T*        out_exceptions = reinterpret_cast<T*>(malloc(sizeof(T) * n_exceptions));
		uint16_t* out_positions  = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * n_exceptions));
		uint16_t* out_offsets_counts =
		    reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * ffor.get_n_vecs() * N_LANES));

		// Intermediate arrays for reordering positions and exceptions
		T        vec_exceptions[consts::VALUES_PER_VECTOR];
		T        vec_exceptions_positions[consts::VALUES_PER_VECTOR];
		uint16_t lane_counts[N_LANES];

		// Copies of pointers for pointer arithmetic
		T*        c_exceptions         = exceptions;
		uint16_t* c_positions          = positions;
		T*        c_out_exceptions     = out_exceptions;
		uint16_t* c_out_positions      = out_positions;
		uint16_t* c_out_offsets_counts = out_offsets_counts;

		for (size_t vec_index {0}; vec_index < ffor.get_n_vecs(); ++vec_index) {
			uint32_t vec_exception_count = counts[vec_index];

			// Reset counts
			for (size_t j {0}; j < N_LANES; ++j) {
				lane_counts[j] = 0;
			}

			// Split all exceptions into lanes
			for (size_t exception_index {0}; exception_index < vec_exception_count; ++exception_index) {
				T        exception = c_exceptions[exception_index];
				uint16_t position  = c_positions[exception_index];

				uint32_t lane                 = position % N_LANES;
				uint32_t lane_exception_count = lane_counts[lane];
				++lane_counts[lane];
				vec_exceptions[lane * VALUES_PER_LANE + lane_exception_count]           = exception;
				vec_exceptions_positions[lane * VALUES_PER_LANE + lane_exception_count] = position;
			}

			// Merge and concatenate all exceptions per lane into single contiguous
			// array
			uint32_t vec_exceptions_counter = 0;
			for (size_t lane {0}; lane < N_LANES; ++lane) {
				uint32_t exc_in_lane_count = lane_counts[lane];
				for (size_t exc_in_lane {0}; exc_in_lane < exc_in_lane_count; ++exc_in_lane) {

					c_out_exceptions[vec_exceptions_counter] = vec_exceptions[lane * VALUES_PER_LANE + exc_in_lane];
					c_out_positions[vec_exceptions_counter] =
					    vec_exceptions_positions[lane * VALUES_PER_LANE + exc_in_lane];
					++vec_exceptions_counter;
				}

				c_out_offsets_counts[lane] = (exc_in_lane_count << 10) | (vec_exceptions_counter - exc_in_lane_count);
			}

			c_exceptions += vec_exception_count;
			c_positions += vec_exception_count;
			c_out_exceptions += vec_exception_count;
			c_out_positions += vec_exception_count;
			c_out_offsets_counts += utils::get_n_lanes<T>();
		}

		return std::make_tuple(out_exceptions, out_positions, out_offsets_counts);
	}

	ALPExtendedColumn<T> create_extended_column() const {
		auto [e_exceptions, e_positions, e_offsets_counts] = convert_exceptions_to_lane_divided_format();
		return ALPExtendedColumn<T> {FFORColumn<UINT_T> {
		                                 BPColumn<UINT_T> {
		                                     ffor.bp.n_values,
		                                     ffor.bp.n_packed_values,
		                                     utils::copy_array(ffor.bp.packed_array, ffor.bp.n_packed_values),
		                                     utils::copy_array(ffor.bp.bit_widths, get_n_vecs()),
		                                     utils::copy_array(ffor.bp.vector_offsets, get_n_vecs()),
		                                 },
		                                 utils::copy_array(ffor.bases, get_n_vecs()),
		                             },
		                             utils::copy_array(factor_indices, get_n_vecs()),
		                             utils::copy_array(fraction_indices, get_n_vecs()),
		                             n_exceptions,
		                             utils::copy_array(exceptions_offsets, get_n_vecs()),
		                             e_exceptions,
		                             e_positions,
		                             e_offsets_counts,
		                             compressed_size_bytes_alp_extended};
	}
};

template <typename T>
void free_column(BPColumn<T> column) {
	delete[] column.packed_array;
	delete[] column.bit_widths;
	delete[] column.vector_offsets;
}

template <typename T>
void free_column(FFORColumn<T> column) {
	free_column(column.bp);
	delete[] column.bases;
}

template <typename T>
void free_column(DICTColumn<T> column) {
	free_column(column.ffor);
	delete[] column.keys;
}

template <typename T>
void free_column(FREQColumn<T> column) {
	delete[] column.frequent_value;
	delete[] column.exceptions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.counts;
}

template <typename T>
void free_column(FREQExtendedColumn<T> column) {
	delete[] column.frequent_value;
	delete[] column.exceptions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.offsets_counts;
}

template <typename T>
void free_column(CROSSRLEColumn<T> column) {
	delete[] column.values;
	delete[] column.lengths;
	delete[] column.run_positions;
	delete[] column.offsets;
}

template <typename T>
void free_column(CROSSRLEExtendedColumn<T> column) {
	delete[] column.lane_runs_offsets;
	delete[] column.lane_values;
	delete[] column.lane_lengths;
	delete[] column.offsets_counts;
}

template <typename T>
void free_column(CROSSRLELaneMaskColumn<T> column) {
	delete[] column.lane_run_base;
	delete[] column.lane_run_values;
	delete[] column.lane_boundary_mask;
}

template <typename T>
void free_column(ALPColumn<T> column) {
	free_column(column.ffor);
	delete[] column.factor_indices;
	delete[] column.fraction_indices;
	delete[] column.exceptions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.counts;
}

template <typename T>
void free_column(ALPExtendedColumn<T> column) {
	free_column(column.ffor);
	delete[] column.factor_indices;
	delete[] column.fraction_indices;
	delete[] column.exceptions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.offsets_counts;
}

// Structs that are passed to the GPU cannot contain methods,
// that is why there is a separate method for the destructors
template <typename T>
void free_column(device::BPColumn<T> column) {
	free_device_pointer(column.packed_array);
	free_device_pointer(column.bit_widths);
	free_device_pointer(column.vector_offsets);
}

template <typename T>
void free_column(device::FFORColumn<T> column) {
	free_column(column.bp);
	free_device_pointer(column.bases);
}

template <typename T>
void free_column(device::DICTColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.keys);
}

template <typename T>
void free_column(device::FREQColumn<T> column) {
	free_device_pointer(column.frequent_value);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.counts);
}

template <typename T>
void free_column(device::FREQExtendedColumn<T> column) {
	free_device_pointer(column.frequent_value);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.offsets_counts);
}

template <typename T>
void free_column(device::CROSSRLEColumn<T> column) {
	free_device_pointer(column.values);
	free_device_pointer(column.lengths);
	free_device_pointer(column.run_positions);
	free_device_pointer(column.offsets);
}

template <typename T>
void free_column(device::CROSSRLEExtendedColumn<T> column) {
	free_device_pointer(column.lane_runs_offsets);
	free_device_pointer(column.lane_values);
	free_device_pointer(column.lane_lengths);
	free_device_pointer(column.offsets_counts);
}

template <typename T>
void free_column(device::CROSSRLELaneMaskColumn<T> column) {
	free_device_pointer(column.lane_run_base);
	free_device_pointer(column.lane_run_values);
	free_device_pointer(column.lane_boundary_mask);
}

template <typename T>
void free_column(device::ALPColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.factors);
	free_device_pointer(column.fractions);
	free_device_pointer(column.factor_indices);
	free_device_pointer(column.fraction_indices);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.counts);
}

template <typename T>
void free_column(device::ALPExtendedColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.factors);
	free_device_pointer(column.fractions);
	free_device_pointer(column.factor_indices);
	free_device_pointer(column.fraction_indices);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.offsets_counts);
}

} // namespace host
} // namespace flsgpu

#endif // STRUCTS_CUH
