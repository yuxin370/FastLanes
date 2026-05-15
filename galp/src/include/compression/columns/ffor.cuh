// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/columns/ffor.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_FFOR_CUH
#define GALP_COMPRESSION_COLUMNS_FFOR_CUH

#include "compression/columns/bp.cuh"

namespace galp::codec {
namespace device {

template <typename T>
struct FFORColumn {
		using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
		size_t      n_values;
		BPColumn<T>        bp;
		UINT_T*            bases;
};

} // namespace device

namespace host {

template <typename T>
struct FFORColumn {
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FFORColumn<T>;

		BPColumn<T> bp;
		HostArray<UINT_T> bases;

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

	void copy_to_device(galp::memory::DeviceArena& arena, device::FFORColumn<T>& out) const {
		const size_t bp_buffer_elems = galp::codec::utils::get_n_lanes<T>() * 4;
		auto         i_packed        = arena.template add<UINT_T>(bp.n_packed_values, bp.packed_array, bp_buffer_elems);
		auto         i_bw            = arena.template add<vbw_t>(bp.get_n_vecs(), bp.bit_widths);
		auto         i_offsets       = arena.template add<uint32_t>(bp.get_n_vecs(), bp.vector_offsets);
		auto         i_bases         = arena.template add<UINT_T>(bp.get_n_vecs(), bases);
		out.n_values                 = get_n_values();
		out.bp.n_values              = bp.n_values;
		out.bp.n_vecs                = bp.get_n_vecs();
		arena.resolve_to(reinterpret_cast<void**>(&out.bp.packed_array), i_packed);
		arena.resolve_to(reinterpret_cast<void**>(&out.bp.bit_widths), i_bw);
		arena.resolve_to(reinterpret_cast<void**>(&out.bp.vector_offsets), i_offsets);
		arena.resolve_to(reinterpret_cast<void**>(&out.bases), i_bases);
	}
};

template <typename T>
void free_column(FFORColumn<T>& column) {
	free_column(column.bp);
	column.bases.reset();
}

template <typename T>
void free_column(device::FFORColumn<T> column) {
	free_column(column.bp);
	free_device_pointer(column.bases);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_FFOR_CUH
