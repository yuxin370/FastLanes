// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/columns/dict_ffor.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_DICT_FFOR_CUH
#define GALP_COMPRESSION_COLUMNS_DICT_FFOR_CUH

#include "compression/columns/ffor.cuh"
#include <cstring>
#include <utility>

namespace galp::codec {
namespace device {

template <typename T, typename IndexT = typename galp::codec::utils::same_width_uint<T>::type>
struct DICTFFORColumn {
	using UINT_T  = typename galp::codec::utils::same_width_uint<T>::type;
	using INDEX_T = IndexT;
	using KEY_T   = UINT_T;
	size_t             n_values;
	FFORColumn<IndexT> ffor; // index stream (FFOR-compressed)

	KEY_T* keys;      // dictionary keys (shared by all vectors)
	size_t key_count; // number of keys in dictionary
};

} // namespace device

namespace host {

template <typename T, typename IndexT = typename galp::codec::utils::same_width_uint<T>::type>
struct DICTFFORColumn {
	using KEY_T         = typename galp::codec::utils::same_width_uint<T>::type;
	using INDEX_T       = IndexT;
	using UINT_T        = KEY_T;
	using DeviceColumnT = typename device::DICTFFORColumn<T, IndexT>;

	FFORColumn<IndexT> ffor; // index stream (FFOR-compressed)
	HostArray<KEY_T>   keys;
	size_t             key_count;

	size_t get_n_values() const {
		return ffor.get_n_values();
	}
	size_t get_n_vecs() const {
		return ffor.get_n_vecs();
	}

	device::DICTFFORColumn<T, IndexT> copy_to_device() const {
		return device::DICTFFORColumn<T, IndexT> {
		    get_n_values(), ffor.copy_to_device(), GPUArray<KEY_T>(key_count, keys).release(), key_count};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::DICTFFORColumn<T, IndexT>& out) const {
		using UINT_IDX               = typename galp::codec::utils::same_width_uint<IndexT>::type;
		const size_t bp_buffer_elems = galp::codec::utils::get_n_lanes<IndexT>() * 4;
		auto i_packed = arena.template add<UINT_IDX>(ffor.bp.n_packed_values, ffor.bp.packed_array, bp_buffer_elems);
		auto i_bw     = arena.template add<vbw_t>(ffor.bp.get_n_vecs(), ffor.bp.bit_widths);
		auto i_bp_off = arena.template add<uint32_t>(ffor.bp.get_n_vecs(), ffor.bp.vector_offsets);
		auto i_bases  = arena.template add<UINT_IDX>(ffor.bp.get_n_vecs(), ffor.bases);
		auto i_keys   = arena.template add<KEY_T>(key_count, keys);
		out.n_values  = get_n_values();
		out.key_count = key_count;
		out.ffor.n_values    = ffor.get_n_values();
		out.ffor.bp.n_values = ffor.bp.n_values;
		out.ffor.bp.n_vecs   = ffor.bp.get_n_vecs();
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.packed_array), i_packed);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.bit_widths), i_bw);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.vector_offsets), i_bp_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bases), i_bases);
		arena.resolve_to(reinterpret_cast<void**>(&out.keys), i_keys);
	}
};

template <typename T, typename IndexT>
void free_column(DICTFFORColumn<T, IndexT>& column) {
	free_column(column.ffor);
	column.keys.reset();
}

template <typename T, typename IndexT>
void free_column(device::DICTFFORColumn<T, IndexT> column) {
	free_column(column.ffor);
	free_device_pointer(column.keys);
}

inline galp::codec::host::FFORColumn<uint8_t> make_ffor_u8_from_ffor_i8(const galp::codec::host::FFORColumn<int8_t>& col) {
	auto*                           packed = galp::codec::utils::copy_array(col.bp.packed_array.get(), col.bp.n_packed_values);
	auto*                           bws    = galp::codec::utils::copy_array(col.bp.bit_widths.get(), col.bp.get_n_vecs());
	auto*                           offs   = galp::codec::utils::copy_array(col.bp.vector_offsets.get(), col.bp.get_n_vecs());
	galp::codec::host::BPColumn<uint8_t> bp {col.bp.n_values, col.bp.n_packed_values, packed, bws, offs};

	auto* bases = new uint8_t[col.get_n_vecs()];
	for (size_t i = 0; i < col.get_n_vecs(); ++i) {
		bases[i] = static_cast<uint8_t>(col.bases[i]);
	}
		return galp::codec::host::FFORColumn<uint8_t> {std::move(bp), bases};
}

inline galp::codec::host::FFORColumn<uint8_t> make_ffor_u8_from_bp_i8(const galp::codec::host::BPColumn<int8_t>& col) {
	auto*                           packed = galp::codec::utils::copy_array(col.packed_array.get(), col.n_packed_values);
	auto*                           bws    = galp::codec::utils::copy_array(col.bit_widths.get(), col.get_n_vecs());
	auto*                           offs   = galp::codec::utils::copy_array(col.vector_offsets.get(), col.get_n_vecs());
	galp::codec::host::BPColumn<uint8_t> bp {col.n_values, col.n_packed_values, packed, bws, offs};
	auto*                           bases = new uint8_t[bp.get_n_vecs()];
	std::memset(bases, 0, bp.get_n_vecs() * sizeof(uint8_t));
		return galp::codec::host::FFORColumn<uint8_t> {std::move(bp), bases};
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_DICT_FFOR_CUH
