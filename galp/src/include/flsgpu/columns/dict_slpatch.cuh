// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/dict_slpatch.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_DICT_SLPATCH_CUH
#define FLSGPU_COLUMNS_DICT_SLPATCH_CUH

#include "flsgpu/columns/dict_ffor.cuh"
#include "flsgpu/columns/parse_common.cuh"
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
		using UINT_IDX = typename utils::same_width_uint<IndexT>::type;
		const size_t bp_buffer_elems = utils::get_n_lanes<IndexT>() * 4;
		const size_t buf = consts::MAX_UNPACK_N_VECS;
		const size_t sl_nvecs = index.n_vecs;
		auto i_packed  = arena.template add<UINT_IDX>(index.ffor.bp.n_packed_values, index.ffor.bp.packed_array, bp_buffer_elems);
		auto i_bw      = arena.template add<vbw_t>(index.ffor.bp.get_n_vecs(), index.ffor.bp.bit_widths);
		auto i_bp_off  = arena.template add<size_t>(index.ffor.bp.get_n_vecs(), index.ffor.bp.vector_offsets);
		auto i_bases   = arena.template add<UINT_IDX>(index.ffor.bp.get_n_vecs(), index.ffor.bases);
		auto i_exc_off = arena.template add<size_t>(sl_nvecs, index.exceptions_offsets);
		auto i_pos_off = arena.template add<size_t>(sl_nvecs, index.positions_offsets);
		auto i_exc     = arena.template add<IndexT>(index.n_exceptions, index.exceptions, buf);
		auto i_pos     = arena.template add<uint16_t>(index.n_exceptions, index.positions, buf);
		auto i_cnt     = arena.template add<uint16_t>(sl_nvecs, index.counts);
		auto i_keys    = arena.template add<KEY_T>(key_count, keys);
		const size_t bp_nv     = index.ffor.bp.n_values;
		const size_t bp_nvec   = index.ffor.bp.get_n_vecs();
		const size_t ffor_nv   = index.ffor.get_n_values();
		const size_t idx_nv    = index.n_values;
		const size_t idx_nexc  = index.n_exceptions;
		out.n_values  = get_n_values();
		out.key_count = key_count;
		arena.add_resolver([&arena, &out, i_packed, i_bw, i_bp_off, i_bases,
		                     i_exc_off, i_pos_off, i_exc, i_pos, i_cnt, i_keys,
		                     bp_nv, bp_nvec, ffor_nv, idx_nv, sl_nvecs, idx_nexc]() {
			device::BPColumn<IndexT> d_bp {
			    bp_nv, bp_nvec,
			    arena.get<UINT_IDX>(i_packed), arena.get<vbw_t>(i_bw), arena.get<size_t>(i_bp_off)};
			device::FFORColumn<IndexT> d_ffor {ffor_nv, d_bp, arena.get<UINT_IDX>(i_bases)};
			out.index = device::SLPATCHColumn<IndexT> {
			    idx_nv, sl_nvecs, d_ffor, idx_nexc,
			    arena.get<size_t>(i_exc_off), arena.get<size_t>(i_pos_off),
			    arena.get<IndexT>(i_exc), arena.get<uint16_t>(i_pos), arena.get<uint16_t>(i_cnt)};
			out.keys = arena.get<KEY_T>(i_keys);
		});
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

} // namespace host
} // namespace flsgpu

namespace reader::columns {

inline flsgpu::host::SLPATCHColumn<uint8_t>
make_slpatch_u8_from_slpatch_i8(const flsgpu::host::SLPATCHColumn<int8_t>& col) {
	auto index_ffor = make_ffor_u8_from_ffor_i8(col.ffor);

	auto* offsets     = utils::copy_array(col.exceptions_offsets, col.n_vecs);
	auto* pos_offsets = utils::copy_array(col.positions_offsets, col.n_vecs);
	auto* pos         = utils::copy_array(col.positions, col.n_exceptions);
	auto* cnt         = utils::copy_array(col.counts, col.n_vecs);
	auto* exc         = new uint8_t[col.n_exceptions];
	for (size_t i = 0; i < col.n_exceptions; ++i) {
		exc[i] = static_cast<uint8_t>(col.exceptions[i]);
	}

	return flsgpu::host::SLPATCHColumn<uint8_t> {
	    col.n_values, col.n_vecs, std::move(index_ffor), col.n_exceptions, offsets, pos_offsets, exc, pos, cnt};
}

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
inline ParseResultT<flsgpu::host::DICTSLPATCHColumn<T, IndexT>> parse_dict_slpatch(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 7) {
		throw std::runtime_error("EXP_DICT_FFOR_SLPATCH: missing operand tokens");
	}
	const auto key_seg_idx = static_cast<uint32_t>(ctx.operand_tokens->Get(0));
	const auto seg_keys    = ctx.column_view.GetSegment(key_seg_idx);
	using KEY_T            = typename utils::same_width_uint<T>::type;
	const size_t key_count = seg_keys.data_span.size() / sizeof(KEY_T);
	auto*        keys      = detail::copy_segment_array<KEY_T>(seg_keys);

	const auto seg_exc       = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(1)));
	const auto seg_pos       = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(2)));
	const auto seg_cnt       = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(3)));
	const auto seg_bitpacked = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(4)));
	const auto seg_bw        = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(5)));
	const auto seg_base      = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(6)));

	auto  bp_parts = detail::parse_bp_segments<IndexT>(seg_bitpacked, seg_bw, ctx.n_vecs);
	auto* bases    = detail::copy_segment_array<IndexT>(seg_base);

	flsgpu::host::BPColumn<IndexT> bp_idx {
	    ctx.n_values, bp_parts.n_packed, bp_parts.packed, bp_parts.bit_widths, bp_parts.vector_offsets};
	flsgpu::host::FFORColumn<IndexT> ffor_idx {bp_idx, bases};

	auto* counts     = detail::copy_segment_array<uint16_t>(seg_cnt);
	auto* positions  = detail::copy_segment_array<uint16_t>(seg_pos);
	auto* exceptions = detail::copy_segment_array<IndexT>(seg_exc);

	auto exc     = detail::build_exception_offsets_from_segment<IndexT>(seg_exc, ctx.n_vecs);
	auto pos_off = detail::build_exception_offsets_from_segment<uint16_t>(seg_pos, ctx.n_vecs);

	flsgpu::host::SLPATCHColumn<IndexT> slpatch_idx {
	    ctx.n_values, ctx.n_vecs, ffor_idx, exc.total, exc.offsets, pos_off.offsets, exceptions, positions, counts};

	return ParseResultT<flsgpu::host::DICTSLPATCHColumn<T, IndexT>> {
	    flsgpu::host::DICTSLPATCHColumn<T, IndexT> {slpatch_idx, keys, key_count}};
}

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
inline ParseResultT<flsgpu::host::DICTSLPATCHColumn<T, IndexT>>
parse_dict_slpatch_with_index(const ParseContext& ctx, flsgpu::host::SLPATCHColumn<IndexT> index_slpatch) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 2) {
		throw std::runtime_error("EXP_DICT: missing operand tokens");
	}
	const auto key_seg_idx = static_cast<uint32_t>(ctx.operand_tokens->Get(ctx.operand_tokens->size() - 1));
	const auto seg_keys    = ctx.column_view.GetSegment(key_seg_idx);
	using KEY_T            = typename utils::same_width_uint<T>::type;
	const size_t key_count = seg_keys.data_span.size() / sizeof(KEY_T);
	auto*        keys      = detail::copy_segment_array<KEY_T>(seg_keys);

	return ParseResultT<flsgpu::host::DICTSLPATCHColumn<T, IndexT>> {
	    flsgpu::host::DICTSLPATCHColumn<T, IndexT> {std::move(index_slpatch), keys, key_count}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_DICT_SLPATCH_CUH
