// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/dict_ffor.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_DICT_FFOR_CUH
#define FLSGPU_COLUMNS_DICT_FFOR_CUH

#include "flsgpu/columns/ffor.cuh"
#include "flsgpu/columns/parse_common.cuh"

namespace flsgpu {
namespace device {

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
struct DICTFFORColumn {
	using UINT_T  = typename utils::same_width_uint<T>::type;
	using INDEX_T = IndexT;
	using KEY_T   = UINT_T;
	size_t             n_values;
	FFORColumn<IndexT> ffor; // index stream (FFOR-compressed)

	KEY_T* keys;      // dictionary keys (shared by all vectors)
	size_t key_count; // number of keys in dictionary
};

} // namespace device

namespace host {

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
struct DICTFFORColumn {
	using KEY_T         = typename utils::same_width_uint<T>::type;
	using INDEX_T       = IndexT;
	using UINT_T        = KEY_T;
	using DeviceColumnT = typename device::DICTFFORColumn<T, IndexT>;

	FFORColumn<IndexT> ffor; // index stream (FFOR-compressed)
	KEY_T*             keys;
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

	void copy_to_device(flsgpu::memory::DeviceArena& arena, device::DICTFFORColumn<T, IndexT>& out) const {
		using UINT_IDX = typename utils::same_width_uint<IndexT>::type;
		const size_t bp_buffer_elems = utils::get_n_lanes<IndexT>() * 4;
		auto i_packed  = arena.template add<UINT_IDX>(ffor.bp.n_packed_values, ffor.bp.packed_array, bp_buffer_elems);
		auto i_bw      = arena.template add<vbw_t>(ffor.bp.get_n_vecs(), ffor.bp.bit_widths);
		auto i_bp_off  = arena.template add<size_t>(ffor.bp.get_n_vecs(), ffor.bp.vector_offsets);
		auto i_bases   = arena.template add<UINT_IDX>(ffor.bp.get_n_vecs(), ffor.bases);
		auto i_keys    = arena.template add<KEY_T>(key_count, keys);
		out.n_values         = get_n_values();
		out.key_count        = key_count;
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
void free_column(DICTFFORColumn<T, IndexT> column) {
	free_column(column.ffor);
	delete[] column.keys;
}

template <typename T, typename IndexT>
void free_column(device::DICTFFORColumn<T, IndexT> column) {
	free_column(column.ffor);
	free_device_pointer(column.keys);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

inline flsgpu::host::FFORColumn<uint8_t> make_ffor_u8_from_ffor_i8(const flsgpu::host::FFORColumn<int8_t>& col) {
	auto*                           packed = utils::copy_array(col.bp.packed_array, col.bp.n_packed_values);
	auto*                           bws    = utils::copy_array(col.bp.bit_widths, col.bp.get_n_vecs());
	auto*                           offs   = utils::copy_array(col.bp.vector_offsets, col.bp.get_n_vecs());
	flsgpu::host::BPColumn<uint8_t> bp {col.bp.n_values, col.bp.n_packed_values, packed, bws, offs};

	auto* bases = new uint8_t[col.get_n_vecs()];
	for (size_t i = 0; i < col.get_n_vecs(); ++i) {
		bases[i] = static_cast<uint8_t>(col.bases[i]);
	}
	return flsgpu::host::FFORColumn<uint8_t> {bp, bases};
}

inline flsgpu::host::FFORColumn<uint8_t> make_ffor_u8_from_bp_i8(const flsgpu::host::BPColumn<int8_t>& col) {
	auto*                           packed = utils::copy_array(col.packed_array, col.n_packed_values);
	auto*                           bws    = utils::copy_array(col.bit_widths, col.get_n_vecs());
	auto*                           offs   = utils::copy_array(col.vector_offsets, col.get_n_vecs());
	flsgpu::host::BPColumn<uint8_t> bp {col.n_values, col.n_packed_values, packed, bws, offs};
	auto*                           bases = new uint8_t[bp.get_n_vecs()];
	std::memset(bases, 0, bp.get_n_vecs() * sizeof(uint8_t));
	return flsgpu::host::FFORColumn<uint8_t> {bp, bases};
}

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
inline ParseResultT<flsgpu::host::DICTFFORColumn<T, IndexT>> parse_dict_ffor(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 4) {
		throw std::runtime_error("EXP_DICT_FFOR: missing operand tokens");
	}
	const auto key_seg_idx = static_cast<uint32_t>(ctx.operand_tokens->Get(0));
	const auto seg_keys    = ctx.column_view.GetSegment(key_seg_idx);
	using KEY_T            = typename utils::same_width_uint<T>::type;
	const size_t key_count = seg_keys.data_span.size() / sizeof(KEY_T);
	auto*        keys      = detail::copy_segment_array<KEY_T>(seg_keys);

	const auto seg_bitpacked = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(1)));
	const auto seg_bw        = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(2)));
	const auto seg_base      = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(3)));

	auto  bp_parts = detail::parse_bp_segments<IndexT>(seg_bitpacked, seg_bw, ctx.n_vecs);
	auto* bases    = detail::copy_segment_array<IndexT>(seg_base);

	flsgpu::host::BPColumn<IndexT> bp_idx {
	    ctx.n_values, bp_parts.n_packed, bp_parts.packed, bp_parts.bit_widths, bp_parts.vector_offsets};
	flsgpu::host::FFORColumn<IndexT> ffor_idx {bp_idx, bases};

	return ParseResultT<flsgpu::host::DICTFFORColumn<T, IndexT>> {
	    flsgpu::host::DICTFFORColumn<T, IndexT> {std::move(ffor_idx), keys, key_count}};
}

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
inline ParseResultT<flsgpu::host::DICTFFORColumn<T, IndexT>>
parse_dict_ffor_with_index(const ParseContext& ctx, flsgpu::host::FFORColumn<IndexT> index_ffor) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 2) {
		throw std::runtime_error("EXP_DICT: missing operand tokens");
	}
	const auto key_seg_idx = static_cast<uint32_t>(ctx.operand_tokens->Get(ctx.operand_tokens->size() - 1));
	const auto seg_keys    = ctx.column_view.GetSegment(key_seg_idx);
	using KEY_T            = typename utils::same_width_uint<T>::type;
	const size_t key_count = seg_keys.data_span.size() / sizeof(KEY_T);
	auto*        keys      = detail::copy_segment_array<KEY_T>(seg_keys);

	return ParseResultT<flsgpu::host::DICTFFORColumn<T, IndexT>> {
	    flsgpu::host::DICTFFORColumn<T, IndexT> {std::move(index_ffor), keys, key_count}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_DICT_FFOR_CUH
