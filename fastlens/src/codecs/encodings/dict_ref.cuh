// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/codecs/encodings/dict_ref.cuh
// ────────────────────────────────────────────────────────
#ifndef FASTLENS_COMPRESSION_COLUMNS_DICT_REF_CUH
#define FASTLENS_COMPRESSION_COLUMNS_DICT_REF_CUH

#include "codecs/encodings/base.cuh"
#include <cstring>

namespace fastlens::codec::host {

template <typename T, typename IndexT = typename fastlens::codec::utils::same_width_uint<T>::type>
struct DICTREFColumn {
	using KEY_T   = typename fastlens::codec::utils::same_width_uint<T>::type;
	using INDEX_T = IndexT;

	size_t   n_values;
	uint32_t index_column_index;
	HostArray<KEY_T> keys;
	size_t           key_count;

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return fastlens::codec::utils::get_n_vecs_from_size(n_values);
	}
};

template <typename T, typename IndexT>
void free_column(DICTREFColumn<T, IndexT>& column) {
	column.keys.reset();
}

} // namespace fastlens::codec::host

#endif // FASTLENS_COMPRESSION_COLUMNS_DICT_REF_CUH
