// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/dict_slpatch.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_DICT_SLPATCH_CUH
#define FLSGPU_COLUMNS_DICT_SLPATCH_CUH

#include "flsgpu/columns/dict_ffor.cuh"
#include "flsgpu/columns/slpatch.cuh"

namespace flsgpu {
namespace device {

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
struct DICTSLPATCHColumn {
	using KEY_T   = typename utils::same_width_uint<T>::type;
	using INDEX_T = IndexT;
	using UINT_T  = KEY_T;
	size_t                n_values;
	SLPATCHColumn<IndexT> index;     // index stream (FFOR+SLPATCH)
	KEY_T*                keys;      // dictionary keys (shared by all vectors)
	size_t                key_count; // number of keys in dictionary
};

} // namespace device

namespace host {

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
struct DICTSLPATCHColumn {
	using KEY_T         = typename utils::same_width_uint<T>::type;
	using INDEX_T       = IndexT;
	using UINT_T        = KEY_T;
	using DeviceColumnT = typename device::DICTSLPATCHColumn<T, IndexT>;

	SLPATCHColumn<IndexT> index;     // index stream (FFOR+SLPATCH)
	KEY_T*                keys;      // host dictionary keys
	size_t                key_count; // number of keys

	size_t get_n_values() const {
		return index.get_n_values();
	}
	size_t get_n_vecs() const {
		return index.get_n_vecs();
	}

	device::DICTSLPATCHColumn<T, IndexT> copy_to_device() const {
		return device::DICTSLPATCHColumn<T, IndexT> {
		    get_n_values(), index.copy_to_device(), GPUArray<KEY_T>(key_count, keys).release(), key_count};
	}

	void copy_to_device(flsgpu::memory::DeviceArena& arena, device::DICTSLPATCHColumn<T, IndexT>& out) const {
		using UINT_IDX               = typename utils::same_width_uint<IndexT>::type;
		const size_t bp_buffer_elems = utils::get_n_lanes<IndexT>() * 4;
		const size_t buf             = consts::MAX_UNPACK_N_VECS;
		const size_t sl_nvecs        = index.n_vecs;
		auto         i_packed =
		    arena.template add<UINT_IDX>(index.ffor.bp.n_packed_values, index.ffor.bp.packed_array, bp_buffer_elems);
		auto i_bw               = arena.template add<vbw_t>(index.ffor.bp.get_n_vecs(), index.ffor.bp.bit_widths);
		auto i_bp_off           = arena.template add<uint32_t>(index.ffor.bp.get_n_vecs(), index.ffor.bp.vector_offsets);
		auto i_bases            = arena.template add<UINT_IDX>(index.ffor.bp.get_n_vecs(), index.ffor.bases);
		auto i_exc_off          = arena.template add<uint32_t>(sl_nvecs, index.exceptions_offsets);
		auto i_exc              = arena.template add<IndexT>(index.n_exceptions, index.exceptions, buf);
		auto i_pos              = arena.template add<uint16_t>(index.n_exceptions, index.positions, buf);
		auto i_cnt              = arena.template add<uint16_t>(sl_nvecs, index.counts);
		auto i_keys             = arena.template add<KEY_T>(key_count, keys);
		out.n_values            = get_n_values();
		out.key_count           = key_count;
		out.index.n_values      = index.n_values;
		out.index.n_vecs        = sl_nvecs;
		out.index.n_exceptions  = index.n_exceptions;
		out.index.ffor.n_values = index.ffor.get_n_values();
		out.index.ffor.bp.n_values = index.ffor.bp.n_values;
		out.index.ffor.bp.n_vecs   = index.ffor.bp.get_n_vecs();
		arena.resolve_to(reinterpret_cast<void**>(&out.index.ffor.bp.packed_array), i_packed);
		arena.resolve_to(reinterpret_cast<void**>(&out.index.ffor.bp.bit_widths), i_bw);
		arena.resolve_to(reinterpret_cast<void**>(&out.index.ffor.bp.vector_offsets), i_bp_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.index.ffor.bases), i_bases);
		arena.resolve_to(reinterpret_cast<void**>(&out.index.exceptions_offsets), i_exc_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.index.exceptions), i_exc);
		arena.resolve_to(reinterpret_cast<void**>(&out.index.positions), i_pos);
		arena.resolve_to(reinterpret_cast<void**>(&out.index.counts), i_cnt);
		arena.resolve_to(reinterpret_cast<void**>(&out.keys), i_keys);
	}
};

template <typename T, typename IndexT>
void free_column(DICTSLPATCHColumn<T, IndexT> column) {
	flsgpu::host::free_column(column.index);
	delete[] column.keys;
}

template <typename T, typename IndexT>
void free_column(device::DICTSLPATCHColumn<T, IndexT> column) {
	flsgpu::host::free_column(column.index);
	free_device_pointer(column.keys);
}

namespace detail {

inline flsgpu::host::SLPATCHColumn<uint8_t>
make_slpatch_u8_from_slpatch_i8(const flsgpu::host::SLPATCHColumn<int8_t>& col) {
	auto index_ffor = detail::make_ffor_u8_from_ffor_i8(col.ffor);

	auto* offsets     = utils::copy_array(col.exceptions_offsets, col.n_vecs);
	auto* pos         = utils::copy_array(col.positions, col.n_exceptions);
	auto* cnt         = utils::copy_array(col.counts, col.n_vecs);
	auto* exc         = new uint8_t[col.n_exceptions];
	for (size_t i = 0; i < col.n_exceptions; ++i) {
		exc[i] = static_cast<uint8_t>(col.exceptions[i]);
	}

	return flsgpu::host::SLPATCHColumn<uint8_t> {
	    col.n_values, col.n_vecs, std::move(index_ffor), col.n_exceptions, offsets, exc, pos, cnt};
}

} // namespace detail

} // namespace host
} // namespace flsgpu

#endif // FLSGPU_COLUMNS_DICT_SLPATCH_CUH
