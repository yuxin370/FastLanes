// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/columns/cross_rle_extended.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_CROSS_RLE_EXTENDED_CUH
#define GALP_COMPRESSION_COLUMNS_CROSS_RLE_EXTENDED_CUH

#include "compression/columns/base.cuh"
#include "compression/consts.cuh"
#include "memory/device_arena.cuh"
#include "memory/gpu_array.cuh"

namespace galp::codec {
namespace device {

template <typename T>
struct CROSSRLEExtendedColumn {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;

	size_t n_values;
	size_t n_vecs;

	size_t    n_lane_runs;       // total runs after lane-projection
	uint32_t* lane_runs_offsets; // [n_vecs+1], global offset into lane_values/lane_lengths
	UINT_T*   lane_values;       // [n_lane_runs]
	uint16_t* lane_lengths;      // [n_lane_runs], length in k-space (stride index)
	uint32_t* offsets_counts;    // [n_vecs * N_LANES], packed per-lane (offset,count) within vector slice
};

} // namespace device

namespace host {

template <typename T>
struct CROSSRLEExtendedColumn {
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::CROSSRLEExtendedColumn<T>;

	size_t n_values;

	size_t    n_lane_runs;       // total runs after lane-projection
	HostArray<uint32_t> lane_runs_offsets; // [n_vecs+1]
	HostArray<UINT_T>   lane_values;       // [n_lane_runs]
	HostArray<uint16_t> lane_lengths;      // [n_lane_runs]
	HostArray<uint32_t> offsets_counts;    // [n_vecs * N_LANES]

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return galp::codec::utils::get_n_vecs_from_size(n_values);
	}

	device::CROSSRLEExtendedColumn<T> copy_to_device() const {
		const size_t n_vecs = get_n_vecs();

		return device::CROSSRLEExtendedColumn<T> {
		    n_values,
		    n_vecs,
		    n_lane_runs,
		    GPUArray<uint32_t>(n_vecs + 1, lane_runs_offsets).release(),
		    GPUArray<UINT_T>(n_lane_runs, lane_values).release(),
		    GPUArray<uint16_t>(n_lane_runs, lane_lengths).release(),
		    GPUArray<uint32_t>(n_vecs * galp::codec::utils::get_n_lanes<T>(), offsets_counts).release(),
		};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::CROSSRLEExtendedColumn<T>& out) const {
		const size_t nv     = get_n_vecs();
		auto         i_offs = arena.template add<uint32_t>(nv + 1, lane_runs_offsets);
		auto         i_vals = arena.template add<UINT_T>(n_lane_runs, lane_values);
		auto         i_lens = arena.template add<uint16_t>(n_lane_runs, lane_lengths);
		auto         i_oc   = arena.template add<uint32_t>(nv * galp::codec::utils::get_n_lanes<T>(), offsets_counts);
		out.n_values        = n_values;
		out.n_vecs          = nv;
		out.n_lane_runs     = n_lane_runs;
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_runs_offsets), i_offs);
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_values), i_vals);
		arena.resolve_to(reinterpret_cast<void**>(&out.lane_lengths), i_lens);
		arena.resolve_to(reinterpret_cast<void**>(&out.offsets_counts), i_oc);
	}
};

template <typename T>
void free_column(CROSSRLEExtendedColumn<T>& column) {
	column.lane_runs_offsets.reset();
	column.lane_values.reset();
	column.lane_lengths.reset();
	column.offsets_counts.reset();
}

template <typename T>
void free_column(device::CROSSRLEExtendedColumn<T> column) {
	free_device_pointer(column.lane_runs_offsets);
	free_device_pointer(column.lane_values);
	free_device_pointer(column.lane_lengths);
	free_device_pointer(column.offsets_counts);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_CROSS_RLE_EXTENDED_CUH
