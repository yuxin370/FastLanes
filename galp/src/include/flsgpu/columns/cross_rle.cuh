// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/cross_rle.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_CROSS_RLE_CUH
#define FLSGPU_COLUMNS_CROSS_RLE_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/columns/cross_rle_extended.cuh"
#include "flsgpu/columns/cross_rle_lane_mask.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/host-utils.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>

namespace flsgpu {
namespace device {

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

} // namespace device

namespace host {

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
void free_column(CROSSRLEColumn<T> column) {
	delete[] column.values;
	delete[] column.lengths;
	delete[] column.run_positions;
	delete[] column.offsets;
}

template <typename T>
void free_column(device::CROSSRLEColumn<T> column) {
	free_device_pointer(column.values);
	free_device_pointer(column.lengths);
	free_device_pointer(column.run_positions);
	free_device_pointer(column.offsets);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::CROSSRLEColumn<T>> parse_cross_rle(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 2) {
		throw std::runtime_error("EXP_CROSS_RLE: missing operand tokens");
	}
	const size_t base_idx = ctx.operand_tokens->size() - 1;
	const auto   seg_vals = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto   seg_lens = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	const size_t n_runs        = seg_lens.data_span.size() / sizeof(uint32_t);
	auto*        values        = detail::copy_segment_array<typename utils::same_width_uint<T>::type>(seg_vals);
	auto*        lengths       = detail::copy_segment_array<uint32_t>(seg_lens);
	auto*        run_positions = new uint32_t[n_runs];

	uint32_t pos = 0;
	for (size_t i = 0; i < n_runs; ++i) {
		run_positions[i] = pos;
		pos += lengths[i];
	}

	auto*    offsets = new uint32_t[ctx.n_vecs + 1];
	uint32_t cur     = 0;
	uint32_t idx_run = 0;
	for (size_t v = 0; v < ctx.n_vecs; ++v) {
		const size_t target_start = v * detail::kVecSize;
		while (idx_run < n_runs && cur + lengths[idx_run] <= target_start) {
			cur += lengths[idx_run];
			++idx_run;
		}
		offsets[v] = idx_run;
	}
	offsets[ctx.n_vecs] = static_cast<uint32_t>(n_runs);

	return ParseResultT<flsgpu::host::CROSSRLEColumn<T>> {
	    flsgpu::host::CROSSRLEColumn<T> {ctx.n_values, n_runs, values, lengths, offsets, run_positions}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_CROSS_RLE_CUH
