// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/dict_ref.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_DICT_REF_CUH
#define FLSGPU_COLUMNS_DICT_REF_CUH

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

#endif // FLSGPU_COLUMNS_DICT_REF_CUH
