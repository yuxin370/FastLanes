// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/cross_rle_extended.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_CROSS_RLE_EXTENDED_CUH
#define FLSGPU_COLUMNS_CROSS_RLE_EXTENDED_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/host-utils.cuh"

namespace flsgpu {
namespace device {

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

} // namespace device

namespace host {

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
void free_column(CROSSRLEExtendedColumn<T> column) {
	delete[] column.lane_runs_offsets;
	delete[] column.lane_values;
	delete[] column.lane_lengths;
	delete[] column.offsets_counts;
}

template <typename T>
void free_column(device::CROSSRLEExtendedColumn<T> column) {
	free_device_pointer(column.lane_runs_offsets);
	free_device_pointer(column.lane_values);
	free_device_pointer(column.lane_lengths);
	free_device_pointer(column.offsets_counts);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::CROSSRLEExtendedColumn<T>> parse_cross_rle_extended(const ParseContext& ctx) {
	using UINT_T = typename utils::same_width_uint<T>::type;
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 2) {
		throw std::runtime_error("EXP_CROSS_RLE_EXTENDED: missing operand tokens");
	}
	const size_t base_idx = ctx.operand_tokens->size() - 1;
	const auto seg_vals = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto seg_lens = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	const size_t n_runs = seg_lens.data_span.size() / sizeof(uint32_t);
	auto*        values = detail::copy_segment_array<UINT_T>(seg_vals);
	auto*        lengths = detail::copy_segment_array<uint32_t>(seg_lens);

	auto* run_positions = new uint32_t[n_runs];
	uint32_t pos = 0;
	for (size_t i = 0; i < n_runs; ++i) {
		run_positions[i] = pos;
		pos += lengths[i];
	}

	auto* offsets = new uint32_t[ctx.n_vecs + 1];
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

	constexpr uint32_t N_LANES         = (uint32_t)utils::get_n_lanes<T>();
	constexpr uint32_t VALUES_PER_LANE = (uint32_t)utils::get_values_per_lane<T>();
	constexpr uint32_t VEC_VALUES      = (uint32_t)consts::VALUES_PER_VECTOR;

	auto* lane_run_counts = new uint16_t[ctx.n_vecs * N_LANES];
	auto* vec_total_runs  = new size_t[ctx.n_vecs];

	for (size_t vec = 0; vec < ctx.n_vecs; ++vec) {
		UINT_T tmp[VEC_VALUES];
		for (uint32_t i = 0; i < VEC_VALUES; ++i) {
			tmp[i] = UINT_T {};
		}

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

	auto*  lane_runs_offsets = new size_t[ctx.n_vecs + 1];
	size_t total_runs        = 0;
	lane_runs_offsets[0]     = 0;
	for (size_t vec = 0; vec < ctx.n_vecs; ++vec) {
		total_runs += vec_total_runs[vec];
		lane_runs_offsets[vec + 1] = total_runs;
	}

	auto* lane_values        = (total_runs ? new UINT_T[total_runs] : nullptr);
	auto* lane_lengths       = (total_runs ? new uint16_t[total_runs] : nullptr);
	auto* offsets_counts     = new uint32_t[ctx.n_vecs * N_LANES];

	for (size_t vec = 0; vec < ctx.n_vecs; ++vec) {
		UINT_T tmp[VEC_VALUES];
		for (uint32_t i = 0; i < VEC_VALUES; ++i) {
			tmp[i] = UINT_T {};
		}

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

			offsets_counts[vec * N_LANES + lane] = (uint32_t(run_cnt) << 16) | uint32_t(lane_off);

			UINT_T   cur_val = tmp[lane];
			uint16_t cur_len = 1;

			for (uint32_t k = 1; k < VALUES_PER_LANE; ++k) {
				const UINT_T v = tmp[lane + k * N_LANES];
				if (v == cur_val) {
					++cur_len;
				} else {
					lane_values[cursor]  = cur_val;
					lane_lengths[cursor] = cur_len;
					++cursor;
					cur_val = v;
					cur_len = 1;
				}
			}
			lane_values[cursor]  = cur_val;
			lane_lengths[cursor] = cur_len;
			++cursor;
		}
	}

	delete[] lane_run_counts;
	delete[] vec_total_runs;
	delete[] values;
	delete[] lengths;
	delete[] offsets;
	delete[] run_positions;

	return ParseResultT<flsgpu::host::CROSSRLEExtendedColumn<T>> {
	    flsgpu::host::CROSSRLEExtendedColumn<T> {
	        ctx.n_values,
	        total_runs,
	        lane_runs_offsets,
	        lane_values,
	        lane_lengths,
	        offsets_counts}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_CROSS_RLE_EXTENDED_CUH
