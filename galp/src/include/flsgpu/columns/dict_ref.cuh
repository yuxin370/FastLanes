// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/dict_ref.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_DICT_REF_CUH
#define FLSGPU_COLUMNS_DICT_REF_CUH

#include "flsgpu/columns/parse_common.cuh"
#include <cstring>

namespace flsgpu { namespace host {

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
struct DICTREFColumn {
	using KEY_T   = typename utils::same_width_uint<T>::type;
	using INDEX_T = IndexT;

	size_t   n_values;
	uint32_t index_column_index;
	KEY_T*   keys;
	size_t   key_count;

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}
};

template <typename T, typename IndexT>
void free_column(DICTREFColumn<T, IndexT> column) {
	delete[] column.keys;
}

}} // namespace flsgpu::host

namespace reader::columns {

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
inline ParseResultT<flsgpu::host::DICTREFColumn<T, IndexT>> parse_dict_ref(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 2) {
		throw std::runtime_error("EXP_DICT: missing operand tokens");
	}
	const auto index_col_idx = static_cast<uint32_t>(ctx.operand_tokens->Get(0));
	const auto key_seg_idx   = static_cast<uint32_t>(ctx.operand_tokens->Get(ctx.operand_tokens->size() - 1));
	const auto seg_keys      = ctx.column_view.GetSegment(key_seg_idx);
	using KEY_T              = typename utils::same_width_uint<T>::type;
	const size_t key_count   = seg_keys.data_span.size() / sizeof(KEY_T);
	auto*        keys        = detail::copy_segment_array<KEY_T>(seg_keys);

	return ParseResultT<flsgpu::host::DICTREFColumn<T, IndexT>> {
	    flsgpu::host::DICTREFColumn<T, IndexT> {ctx.n_values, index_col_idx, keys, key_count}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_DICT_REF_CUH
