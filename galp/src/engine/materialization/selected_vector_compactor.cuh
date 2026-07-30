#ifndef GALP_ENGINE_MATERIALIZATION_SELECTED_VECTOR_COMPACTOR_CUH
#define GALP_ENGINE_MATERIALIZATION_SELECTED_VECTOR_COMPACTOR_CUH

#include "codecs/consts.cuh"
#include "core/data/model.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace galp::runtime::detail {

template <typename T>
T* copy_elements(const T* source, const size_t count) {
	if (count == 0U) {
		return nullptr;
	}
	if (source == nullptr) {
		throw std::runtime_error("selected-vector compaction encountered a null source array");
	}
	auto* out = new T[count];
	std::memcpy(out, source, count * sizeof(T));
	return out;
}

inline void validate_selected_vectors(const std::vector<uint32_t>& selected, const size_t n_vecs) {
	if (selected.empty()) {
		throw std::invalid_argument("selected-vector compaction requires at least one vector");
	}
	if (!std::is_sorted(selected.begin(), selected.end()) ||
	    std::adjacent_find(selected.begin(), selected.end()) != selected.end()) {
		throw std::invalid_argument("selected-vector compaction requires sorted unique vectors");
	}
	if (selected.back() >= n_vecs) {
		throw std::out_of_range("selected-vector compaction index exceeds column vectors");
	}
}

template <typename T>
galp::codec::host::BPColumn<T> compact_bp(const galp::codec::host::BPColumn<T>& column,
	                                      const std::vector<uint32_t>&           selected) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	validate_selected_vectors(selected, column.get_n_vecs());
	std::vector<UINT_T>   packed;
	std::vector<vbw_t>    widths;
	std::vector<uint32_t> offsets;
	widths.reserve(selected.size());
	offsets.reserve(selected.size());
	for (const uint32_t vector : selected) {
		const size_t begin = column.vector_offsets[vector];
		const size_t end   = static_cast<size_t>(vector) + 1U < column.get_n_vecs()
		                         ? column.vector_offsets[vector + 1U]
		                         : column.n_packed_values;
		if (end < begin || end > column.n_packed_values) {
			throw std::runtime_error("selected-vector BP offsets are invalid");
		}
		offsets.push_back(static_cast<uint32_t>(packed.size()));
		widths.push_back(column.bit_widths[vector]);
		packed.insert(packed.end(), column.packed_array.get() + begin, column.packed_array.get() + end);
	}
	return galp::codec::host::BPColumn<T> {
	    selected.size() * galp::codec::consts::VALUES_PER_VECTOR,
	    packed.size(),
	    copy_elements(packed.data(), packed.size()),
	    copy_elements(widths.data(), widths.size()),
	    copy_elements(offsets.data(), offsets.size())};
}

template <typename T>
galp::codec::host::FFORColumn<T> compact_ffor(const galp::codec::host::FFORColumn<T>& column,
	                                          const std::vector<uint32_t>&             selected) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto bp = compact_bp(column.bp, selected);
	auto* bases = new UINT_T[selected.size()];
	for (size_t index = 0; index < selected.size(); ++index) {
		bases[index] = column.bases[selected[index]];
	}
	return galp::codec::host::FFORColumn<T> {std::move(bp), bases};
}

template <typename T>
galp::codec::host::SLPATCHColumn<T> compact_slpatch(const galp::codec::host::SLPATCHColumn<T>& column,
	                                                const std::vector<uint32_t>&                selected) {
	auto ffor = compact_ffor(column.ffor, selected);
	std::vector<uint32_t> offsets;
	std::vector<T>        exceptions;
	std::vector<uint16_t> positions;
	std::vector<uint16_t> counts;
	offsets.reserve(selected.size());
	counts.reserve(selected.size());
	for (const uint32_t vector : selected) {
		const size_t count = column.counts[vector];
		const size_t begin = column.exceptions_offsets[vector];
		if (begin > column.n_exceptions || count > column.n_exceptions - begin) {
			throw std::runtime_error("selected-vector SLPATCH exception range is invalid");
		}
		offsets.push_back(static_cast<uint32_t>(exceptions.size()));
		counts.push_back(static_cast<uint16_t>(count));
		exceptions.insert(exceptions.end(), column.exceptions.get() + begin, column.exceptions.get() + begin + count);
		positions.insert(positions.end(), column.positions.get() + begin, column.positions.get() + begin + count);
	}
	return galp::codec::host::SLPATCHColumn<T> {
	    selected.size() * galp::codec::consts::VALUES_PER_VECTOR,
	    selected.size(),
	    std::move(ffor),
	    exceptions.size(),
	    copy_elements(offsets.data(), offsets.size()),
	    copy_elements(exceptions.data(), exceptions.size()),
	    copy_elements(positions.data(), positions.size()),
	    copy_elements(counts.data(), counts.size())};
}

template <typename T>
galp::codec::host::FREQColumn<T> compact_frequency(const galp::codec::host::FREQColumn<T>& column,
	                                               const std::vector<uint32_t>&             selected) {
	validate_selected_vectors(selected, column.get_n_vecs());
	std::vector<uint32_t> offsets;
	std::vector<T>        exceptions;
	std::vector<uint16_t> positions;
	std::vector<uint16_t> counts;
	offsets.reserve(selected.size());
	counts.reserve(selected.size());
	for (const uint32_t vector : selected) {
		const size_t count = column.counts[vector];
		const size_t begin = column.exceptions_offsets[vector];
		if (begin > column.n_exceptions || count > column.n_exceptions - begin) {
			throw std::runtime_error("selected-vector FREQ exception range is invalid");
		}
		offsets.push_back(static_cast<uint32_t>(exceptions.size()));
		counts.push_back(static_cast<uint16_t>(count));
		exceptions.insert(exceptions.end(), column.exceptions.get() + begin, column.exceptions.get() + begin + count);
		positions.insert(positions.end(), column.positions.get() + begin, column.positions.get() + begin + count);
	}
	return galp::codec::host::FREQColumn<T> {
	    selected.size() * galp::codec::consts::VALUES_PER_VECTOR,
	    selected.size(),
	    column.frequent_value,
	    exceptions.size(),
	    copy_elements(offsets.data(), offsets.size()),
	    copy_elements(exceptions.data(), exceptions.size()),
	    copy_elements(positions.data(), positions.size()),
	    copy_elements(counts.data(), counts.size())};
}

template <typename T>
galp::codec::host::DELTAColumn<T> compact_delta(const galp::codec::host::DELTAColumn<T>& column,
	                                            const std::vector<uint32_t>&              selected) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto ffor = compact_ffor(column.ffor, selected);
	constexpr size_t lanes = galp::codec::utils::get_n_lanes<T>();
	auto* bases = new UINT_T[selected.size() * lanes];
	for (size_t index = 0; index < selected.size(); ++index) {
		std::memcpy(bases + index * lanes, column.rsum_bases.get() + selected[index] * lanes, lanes * sizeof(UINT_T));
	}
	return galp::codec::host::DELTAColumn<T> {std::move(ffor), bases};
}

template <typename T, typename IndexT>
galp::codec::host::DICTFFORColumn<T, IndexT>
compact_dict_ffor(const galp::codec::host::DICTFFORColumn<T, IndexT>& column,
	              const std::vector<uint32_t>&                        selected) {
	using KEY_T = typename galp::codec::host::DICTFFORColumn<T, IndexT>::KEY_T;
	return galp::codec::host::DICTFFORColumn<T, IndexT> {
	    compact_ffor(column.ffor, selected), copy_elements<KEY_T>(column.keys.get(), column.key_count), column.key_count};
}

template <typename T, typename IndexT>
galp::codec::host::DICTSLPATCHColumn<T, IndexT>
compact_dict_slpatch(const galp::codec::host::DICTSLPATCHColumn<T, IndexT>& column,
	                 const std::vector<uint32_t>&                           selected) {
	using KEY_T = typename galp::codec::host::DICTSLPATCHColumn<T, IndexT>::KEY_T;
	return galp::codec::host::DICTSLPATCHColumn<T, IndexT> {compact_slpatch(column.index, selected),
	                                                        copy_elements<KEY_T>(column.keys.get(), column.key_count),
	                                                        column.key_count};
}

template <typename T, typename IndexT>
galp::codec::host::RLEColumn<T, IndexT> compact_rle(const galp::codec::host::RLEColumn<T, IndexT>& column,
	                                                const std::vector<uint32_t>&                    selected) {
	validate_selected_vectors(selected, column.n_vecs);
	auto ffor = compact_ffor(column.ffor, selected);
	constexpr size_t lanes = galp::codec::utils::get_n_lanes<IndexT>();
	auto* bases = new IndexT[selected.size() * lanes];
	std::vector<T> values;
	std::vector<uint32_t> offsets;
	for (size_t index = 0; index < selected.size(); ++index) {
		const uint32_t vector = selected[index];
		std::memcpy(bases + index * lanes, column.rsum_bases.get() + vector * lanes, lanes * sizeof(IndexT));
		const size_t begin = column.rle_offsets[vector];
		const size_t end = static_cast<size_t>(vector) + 1U < column.n_vecs ? column.rle_offsets[vector + 1U]
		                                                                       : column.n_rle_values;
		if (end < begin || end > column.n_rle_values) {
			throw std::runtime_error("selected-vector RLE value range is invalid");
		}
		offsets.push_back(static_cast<uint32_t>(values.size()));
		values.insert(values.end(), column.rle_values.get() + begin, column.rle_values.get() + end);
	}
	return galp::codec::host::RLEColumn<T, IndexT> {
	    selected.size() * galp::codec::consts::VALUES_PER_VECTOR,
	    selected.size(),
	    std::move(ffor),
	    bases,
	    copy_elements(values.data(), values.size()),
	    copy_elements(offsets.data(), offsets.size()),
	    values.size()};
}

template <typename T, typename IndexT>
galp::codec::host::RLESLPATCHColumn<T, IndexT>
compact_rle_slpatch(const galp::codec::host::RLESLPATCHColumn<T, IndexT>& column,
	                const std::vector<uint32_t>&                           selected) {
	validate_selected_vectors(selected, column.n_vecs);
	auto index_column = compact_slpatch(column.index, selected);
	constexpr size_t lanes = galp::codec::utils::get_n_lanes<IndexT>();
	auto* bases = new IndexT[selected.size() * lanes];
	std::vector<T> values;
	std::vector<uint32_t> offsets;
	for (size_t index = 0; index < selected.size(); ++index) {
		const uint32_t vector = selected[index];
		std::memcpy(bases + index * lanes, column.rsum_bases.get() + vector * lanes, lanes * sizeof(IndexT));
		const size_t begin = column.rle_offsets[vector];
		const size_t end = static_cast<size_t>(vector) + 1U < column.n_vecs ? column.rle_offsets[vector + 1U]
		                                                                       : column.n_rle_values;
		if (end < begin || end > column.n_rle_values) {
			throw std::runtime_error("selected-vector RLE-SLPATCH value range is invalid");
		}
		offsets.push_back(static_cast<uint32_t>(values.size()));
		values.insert(values.end(), column.rle_values.get() + begin, column.rle_values.get() + end);
	}
	return galp::codec::host::RLESLPATCHColumn<T, IndexT> {
	    selected.size() * galp::codec::consts::VALUES_PER_VECTOR,
	    selected.size(),
	    std::move(index_column),
	    bases,
	    copy_elements(values.data(), values.size()),
	    copy_elements(offsets.data(), offsets.size()),
	    values.size()};
}

template <typename T>
galp::codec::host::CROSSRLEColumn<T>
compact_cross_rle(const galp::codec::host::CROSSRLEColumn<T>& column,
	              const std::vector<uint32_t>&                 selected) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	validate_selected_vectors(selected, column.get_n_vecs());
	std::vector<UINT_T>   values;
	std::vector<uint32_t> lengths;
	std::vector<uint32_t> offsets;
	std::vector<uint32_t> positions;
	for (size_t compact_vector = 0; compact_vector < selected.size(); ++compact_vector) {
		const uint32_t vector = selected[compact_vector];
		const uint64_t source_begin = static_cast<uint64_t>(vector) * galp::codec::consts::VALUES_PER_VECTOR;
		const uint64_t source_end = std::min<uint64_t>(
		    column.n_values, source_begin + galp::codec::consts::VALUES_PER_VECTOR);
		const uint32_t run_begin = column.offsets[vector];
		const uint32_t run_end   = column.offsets[vector + 1U];
		if (run_end < run_begin || run_end > column.n_runs) {
			throw std::runtime_error("selected-vector CROSS_RLE run range is invalid");
		}
		const uint32_t run_limit =
		    run_end < column.n_runs ? static_cast<uint32_t>(run_end + 1U) : run_end;
		offsets.push_back(static_cast<uint32_t>(values.size()));
		uint64_t covered = 0U;
		for (uint32_t run = run_begin; run < run_limit; ++run) {
			const uint64_t begin = column.run_positions[run];
			const uint64_t end   = begin + column.lengths[run];
			const uint64_t clipped_begin = std::max(begin, source_begin);
			const uint64_t clipped_end   = std::min(end, source_end);
			if (clipped_end <= clipped_begin) {
				continue;
			}
			positions.push_back(static_cast<uint32_t>(compact_vector * galp::codec::consts::VALUES_PER_VECTOR +
			                                                 (clipped_begin - source_begin)));
			lengths.push_back(static_cast<uint32_t>(clipped_end - clipped_begin));
			values.push_back(column.values[run]);
			covered += clipped_end - clipped_begin;
		}
		const uint64_t source_values = source_end - source_begin;
		if (covered != source_values) {
			throw std::runtime_error(
			    "selected-vector CROSS_RLE runs do not cover the source vector (vector=" +
			    std::to_string(vector) + ", covered=" + std::to_string(covered) +
			    ", expected=" + std::to_string(source_values) + ", run_begin=" + std::to_string(run_begin) +
			    ", run_end=" + std::to_string(run_end) + ", n_runs=" + std::to_string(column.n_runs) + ")");
		}
		if (source_values < galp::codec::consts::VALUES_PER_VECTOR) {
			positions.push_back(static_cast<uint32_t>(compact_vector * galp::codec::consts::VALUES_PER_VECTOR +
			                                                 source_values));
			lengths.push_back(static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR - source_values));
			values.push_back(UINT_T {});
		}
	}
	offsets.push_back(static_cast<uint32_t>(values.size()));
	return galp::codec::host::CROSSRLEColumn<T> {
	    selected.size() * galp::codec::consts::VALUES_PER_VECTOR,
	    values.size(),
	    copy_elements(values.data(), values.size()),
	    copy_elements(lengths.data(), lengths.size()),
	    copy_elements(offsets.data(), offsets.size()),
	    copy_elements(positions.data(), positions.size())};
}

inline void compact_selected_vectors(galp::execution::Rowgroup& rowgroup,
	                                 const std::vector<uint32_t>& selected) {
	validate_selected_vectors(selected, rowgroup.n_vecs);
	for (auto& column : rowgroup.columns) {
		if (column.skip_decompress) {
			continue;
		}
		column.host = std::visit(
		    [&](auto& host_column) -> galp::execution::EncodedPayload {
			    using ColumnT = std::decay_t<decltype(host_column)>;
			    if constexpr (std::is_same_v<ColumnT, galp::codec::host::BPColumn<int8_t>> ||
			                  std::is_same_v<ColumnT, galp::codec::host::BPColumn<int16_t>>) {
				    return compact_bp(host_column, selected);
			    } else if constexpr (std::is_same_v<ColumnT, galp::codec::host::FFORColumn<int8_t>> ||
			                         std::is_same_v<ColumnT, galp::codec::host::FFORColumn<int16_t>>) {
				    return compact_ffor(host_column, selected);
			    } else if constexpr (std::is_same_v<ColumnT, galp::codec::host::DELTAColumn<int8_t>> ||
			                         std::is_same_v<ColumnT, galp::codec::host::DELTAColumn<int16_t>>) {
				    return compact_delta(host_column, selected);
			    } else if constexpr (std::is_same_v<ColumnT, galp::codec::host::SLPATCHColumn<int8_t>> ||
			                         std::is_same_v<ColumnT, galp::codec::host::SLPATCHColumn<int16_t>>) {
				    return compact_slpatch(host_column, selected);
			    } else if constexpr (std::is_same_v<ColumnT, galp::codec::host::FREQColumn<int8_t>> ||
			                         std::is_same_v<ColumnT, galp::codec::host::FREQColumn<int16_t>>) {
				    return compact_frequency(host_column, selected);
			    } else if constexpr (std::is_same_v<ColumnT, galp::codec::host::CROSSRLEColumn<int8_t>> ||
			                         std::is_same_v<ColumnT, galp::codec::host::CROSSRLEColumn<int16_t>>) {
				    return compact_cross_rle(host_column, selected);
			    } else if constexpr (requires { host_column.ffor; host_column.keys; host_column.key_count; }) {
				    return compact_dict_ffor(host_column, selected);
			    } else if constexpr (requires { host_column.index; host_column.keys; host_column.key_count; }) {
				    return compact_dict_slpatch(host_column, selected);
			    } else if constexpr (requires { host_column.ffor; host_column.rsum_bases; host_column.rle_values; }) {
				    return compact_rle(host_column, selected);
			    } else if constexpr (requires { host_column.index; host_column.rsum_bases; host_column.rle_values; }) {
				    return compact_rle_slpatch(host_column, selected);
			    } else if constexpr (requires { host_column.value; }) {
				    return ColumnT {selected.size() * galp::codec::consts::VALUES_PER_VECTOR, host_column.value};
			    } else {
				    throw std::runtime_error("selected-vector compaction encountered an unresolved dictionary reference");
			    }
		    },
		    column.host);
		column.host_owned_by_backing = false;
		column.backing_base          = nullptr;
		column.backing_bytes         = 0U;
		column.backing_is_pinned     = false;
	}
	rowgroup.n_values = selected.size() * galp::codec::consts::VALUES_PER_VECTOR;
	rowgroup.n_vecs   = selected.size();
	rowgroup.n_tuples = rowgroup.n_values;
	rowgroup.packed_device_payload.reset();
}

} // namespace galp::runtime::detail

#endif
