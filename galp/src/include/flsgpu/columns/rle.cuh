// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/rle.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_RLE_CUH
#define FLSGPU_COLUMNS_RLE_CUH

#include "flsgpu/columns/ffor.cuh"
#include "flsgpu/memory/device_arena.cuh"
#include "flsgpu/memory/gpu_array.cuh"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace galp::codec {
namespace device {

template <typename T, typename IndexT>
struct RLEColumn {
	size_t             n_values;
	size_t             n_vecs;
	FFORColumn<IndexT> ffor;
	IndexT*            rsum_bases;   // n_vecs * n_lanes(IndexT)
	T*                 rle_values;   // concatenated per-vector values
	uint32_t*          rle_offsets;  // per-vector base offset into rle_values
	size_t             n_rle_values; // total values length
};

} // namespace device

namespace host {

template <typename T, typename IndexT>
struct RLEColumn {
	using DeviceColumnT = typename device::RLEColumn<T, IndexT>;

	size_t             n_values;
	size_t             n_vecs;
	FFORColumn<IndexT> ffor;
	HostArray<IndexT>  rsum_bases;
	HostArray<T>       rle_values;
	HostArray<uint32_t> rle_offsets;
	size_t             n_rle_values;

	size_t get_n_values() const {
		return n_values;
	}

	device::RLEColumn<T, IndexT> copy_to_device() const {
		return device::RLEColumn<T, IndexT> {
		    n_values,
		    n_vecs,
		    ffor.copy_to_device(),
		    GPUArray<IndexT>(n_vecs * galp::codec::utils::get_n_lanes<IndexT>(), rsum_bases).release(),
		    GPUArray<T>(n_rle_values, rle_values).release(),
		    GPUArray<uint32_t>(n_vecs, rle_offsets).release(),
		    n_rle_values};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::RLEColumn<T, IndexT>& out) const {
		using UINT_IDX               = typename galp::codec::utils::same_width_uint<IndexT>::type;
		const size_t bp_buffer_elems = galp::codec::utils::get_n_lanes<IndexT>() * 4;
		auto i_packed    = arena.template add<UINT_IDX>(ffor.bp.n_packed_values, ffor.bp.packed_array, bp_buffer_elems);
		auto i_bw        = arena.template add<vbw_t>(ffor.bp.get_n_vecs(), ffor.bp.bit_widths);
		auto i_bp_off    = arena.template add<uint32_t>(ffor.bp.get_n_vecs(), ffor.bp.vector_offsets);
		auto i_bases     = arena.template add<UINT_IDX>(ffor.bp.get_n_vecs(), ffor.bases);
		auto i_rsum      = arena.template add<IndexT>(n_vecs * galp::codec::utils::get_n_lanes<IndexT>(), rsum_bases);
		auto i_vals      = arena.template add<T>(n_rle_values, rle_values);
		auto i_offs      = arena.template add<uint32_t>(n_vecs, rle_offsets);
		out.n_values     = n_values;
		out.n_vecs       = n_vecs;
		out.n_rle_values = n_rle_values;
		out.ffor.n_values    = ffor.get_n_values();
		out.ffor.bp.n_values = ffor.bp.n_values;
		out.ffor.bp.n_vecs   = ffor.bp.get_n_vecs();
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.packed_array), i_packed);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.bit_widths), i_bw);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.vector_offsets), i_bp_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bases), i_bases);
		arena.resolve_to(reinterpret_cast<void**>(&out.rsum_bases), i_rsum);
		arena.resolve_to(reinterpret_cast<void**>(&out.rle_values), i_vals);
		arena.resolve_to(reinterpret_cast<void**>(&out.rle_offsets), i_offs);
	}
};

template <typename T, typename IndexT>
void free_column(RLEColumn<T, IndexT>& column) {
	free_column(column.ffor);
	column.rsum_bases.reset();
	column.rle_values.reset();
	column.rle_offsets.reset();
}

template <typename T, typename IndexT>
void free_column(device::RLEColumn<T, IndexT> column) {
	free_column(column.ffor);
	free_device_pointer(column.rsum_bases);
	free_device_pointer(column.rle_values);
	free_device_pointer(column.rle_offsets);
}

} // namespace host
} // namespace galp::codec

#endif // FLSGPU_COLUMNS_RLE_CUH
