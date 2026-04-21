// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/cross_rle_lane_mask.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_CROSS_RLE_LANE_MASK_CUH
#define FLSGPU_COLUMNS_CROSS_RLE_LANE_MASK_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/memory/device_arena.cuh"
#include "flsgpu/memory/gpu_array.cuh"

namespace flsgpu {
namespace device {

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

} // namespace device

namespace host {

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

	void copy_to_device(flsgpu::memory::DeviceArena& arena, device::CROSSRLELaneMaskColumn<T>& out) const {
		const size_t nv         = get_n_vecs();
		const size_t total_runs = nv * utils::get_n_lanes<T>();
		auto i_base = arena.template add<uint32_t>(total_runs + 1, lane_run_base);
		auto i_vals = arena.template add<UINT_T>(n_lane_runs, lane_run_values);
		auto i_mask = arena.template add<uint64_t>(total_runs, lane_boundary_mask);
		out.n_values    = get_n_values();
		out.n_vecs      = nv;
		out.n_lane_runs = n_lane_runs;
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_run_base), i_base);
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_run_values), i_vals);
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_boundary_mask), i_mask);
	}
};

template <typename T>
void free_column(CROSSRLELaneMaskColumn<T> column) {
	delete[] column.lane_run_base;
	delete[] column.lane_run_values;
	delete[] column.lane_boundary_mask;
}

template <typename T>
void free_column(device::CROSSRLELaneMaskColumn<T> column) {
	free_device_pointer(column.lane_run_base);
	free_device_pointer(column.lane_run_values);
	free_device_pointer(column.lane_boundary_mask);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::CROSSRLELaneMaskColumn<T>> parse_cross_rle_lane_mask(const ParseContext& ctx) {
	using UINT_T = typename utils::same_width_uint<T>::type;
	auto raw = flsgpu::host::detail::parse_cross_rle_raw_runs<T>(ctx);

	constexpr uint32_t N_LANES         = (uint32_t)utils::get_n_lanes<T>();
	constexpr uint32_t VALUES_PER_LANE = (uint32_t)utils::get_values_per_lane<T>();
	constexpr uint32_t VEC_VALUES      = (uint32_t)consts::VALUES_PER_VECTOR;

	static_assert(VALUES_PER_LANE <= 64, "lane_boundary_mask uses uint64_t; extend if >64");
	static_assert(VEC_VALUES == N_LANES * VALUES_PER_LANE, "expect VALUES_PER_VECTOR == N_LANES*VALUES_PER_LANE");

	const size_t n_blocks = ctx.n_vecs * (size_t)N_LANES;

	auto* lane_run_cnt       = new uint16_t[n_blocks];
	auto* lane_run_base      = new uint32_t[n_blocks + 1];
	auto* lane_boundary_mask = new uint64_t[n_blocks];

	for (size_t i = 0; i < n_blocks; ++i) {
		lane_run_cnt[i]       = 0;
		lane_boundary_mask[i] = 0ull;
	}

	for (size_t vec = 0; vec < ctx.n_vecs; ++vec) {
		UINT_T tmp[VEC_VALUES];
		const uint32_t vec_base = (uint32_t)(vec * (size_t)VEC_VALUES);
		flsgpu::host::detail::expand_runs_into_vector<UINT_T, VEC_VALUES>(
		    tmp, vec_base, raw.values, raw.lengths, raw.run_positions, raw.offsets[vec], raw.offsets[vec + 1]);

		for (uint32_t lane = 0; lane < N_LANES; ++lane) {
			uint64_t mask    = 0ull;
			UINT_T   prev    = tmp[lane];
			uint16_t run_cnt = 1;

			for (uint32_t k = 1; k < VALUES_PER_LANE; ++k) {
				const UINT_T   cur     = tmp[lane + k * N_LANES];
				const uint32_t changed = (cur != prev) ? 1u : 0u;
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

	auto* lane_run_values = (total_runs ? new UINT_T[(size_t)total_runs] : nullptr);

	for (size_t vec = 0; vec < ctx.n_vecs; ++vec) {
		UINT_T tmp[VEC_VALUES];
		const uint32_t vec_base = (uint32_t)(vec * (size_t)VEC_VALUES);
		flsgpu::host::detail::expand_runs_into_vector<UINT_T, VEC_VALUES>(
		    tmp, vec_base, raw.values, raw.lengths, raw.run_positions, raw.offsets[vec], raw.offsets[vec + 1]);

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
	delete[] raw.values;
	delete[] raw.lengths;
	delete[] raw.offsets;
	delete[] raw.run_positions;

	return ParseResultT<flsgpu::host::CROSSRLELaneMaskColumn<T>> {flsgpu::host::CROSSRLELaneMaskColumn<T> {
	    ctx.n_values, (size_t)total_runs, lane_run_base, lane_run_values, lane_boundary_mask}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_CROSS_RLE_LANE_MASK_CUH
