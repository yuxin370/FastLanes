// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/codecs/encodings/cross_rle_lane_mask.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_CROSS_RLE_LANE_MASK_CUH
#define GALP_COMPRESSION_COLUMNS_CROSS_RLE_LANE_MASK_CUH

#include "codecs/encodings/base.cuh"
#include "codecs/consts.cuh"
#include "cuda/memory/device_arena.cuh"
#include "cuda/memory/gpu_array.cuh"

namespace galp::codec {
namespace device {

template <typename T>
struct CROSSRLELaneMaskColumn {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;

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
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::CROSSRLELaneMaskColumn<T>;

		size_t n_values;

		size_t    n_lane_runs;        // total runs across all (vec,lane)
		HostArray<uint32_t> lane_run_base;      // [n_vecs*N_LANES + 1]  CSR base
		HostArray<UINT_T>   lane_run_values;    // [n_lane_runs]
		HostArray<uint64_t> lane_boundary_mask; // [n_vecs*N_LANES] bit k=1 => new run begins at k

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return galp::codec::utils::get_n_vecs_from_size(n_values);
	}

	device::CROSSRLELaneMaskColumn<T> copy_to_device() const {
		const size_t n_vecs     = get_n_vecs();
		const size_t total_runs = n_vecs * galp::codec::utils::get_n_lanes<T>();
		return device::CROSSRLELaneMaskColumn<T> {
		    get_n_values(),
		    n_vecs,
		    n_lane_runs,
		    GPUArray<uint32_t>(total_runs + 1, lane_run_base).release(),
		    GPUArray<UINT_T>(n_lane_runs, lane_run_values).release(),
		    GPUArray<uint64_t>(total_runs, lane_boundary_mask).release(),
		};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::CROSSRLELaneMaskColumn<T>& out) const {
		const size_t nv         = get_n_vecs();
		const size_t total_runs = nv * galp::codec::utils::get_n_lanes<T>();
		auto         i_base     = arena.template add<uint32_t>(total_runs + 1, lane_run_base);
		auto         i_vals     = arena.template add<UINT_T>(n_lane_runs, lane_run_values);
		auto         i_mask     = arena.template add<uint64_t>(total_runs, lane_boundary_mask);
		out.n_values            = get_n_values();
		out.n_vecs              = nv;
		out.n_lane_runs         = n_lane_runs;
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_run_base), i_base);
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_run_values), i_vals);
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_boundary_mask), i_mask);
	}
};

template <typename T>
void free_column(CROSSRLELaneMaskColumn<T>& column) {
	column.lane_run_base.reset();
	column.lane_run_values.reset();
	column.lane_boundary_mask.reset();
}

template <typename T>
void free_column(device::CROSSRLELaneMaskColumn<T> column) {
	free_device_pointer(column.lane_run_base);
	free_device_pointer(column.lane_run_values);
	free_device_pointer(column.lane_boundary_mask);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_CROSS_RLE_LANE_MASK_CUH
