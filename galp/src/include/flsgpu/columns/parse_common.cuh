// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/parse_common.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_PARSE_COMMON_CUH
#define FLSGPU_COLUMNS_PARSE_COMMON_CUH

#include "fls/footer/column_descriptor_generated.h"
#include "fls/reader/column_view.hpp"
#include "fls/reader/segment.hpp"
#include "flsgpu/consts.cuh"
#include "flsgpu/device-types.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace reader::columns {

struct ParseContext {
	const fastlanes::ColumnView&         column_view;
	const fastlanes::ColumnDescriptor&   col_desc;
	const flatbuffers::Vector<uint64_t>* operand_tokens;
	size_t                               n_values;
	size_t                               n_vecs;
};

template <typename ColT>
struct ParseResultT {
	ColT host;
};

namespace detail {

inline constexpr size_t kVecSize = consts::VALUES_PER_VECTOR;

inline std::vector<uint32_t> extract_entrypoints(const fastlanes::SegmentView& seg) {
	std::vector<uint32_t> out;
	std::visit(
	    [&](auto&& view) {
		    using V = std::decay_t<decltype(view)>;
		    if constexpr (std::is_same_v<V, std::monostate>) {
			    return;
		    } else {
			    out.reserve(view.entrypoint_span.size());
			    for (auto v : view.entrypoint_span) {
				    out.push_back(static_cast<uint32_t>(v));
			    }
		    }
	    },
	    seg.entry_point_view);
	return out;
}

template <typename T>
inline T* copy_segment_array(const fastlanes::SegmentView& seg) {
	const auto n_bytes = seg.data_span.size();
	const auto n_elems = n_bytes / sizeof(T);
	auto*      out     = new T[n_elems];
	std::memcpy(out, seg.data_span.data(), n_elems * sizeof(T));
	return out;
}

template <typename T>
inline std::vector<size_t> build_vector_offsets(const std::vector<uint32_t>& entrypoints_bytes) {
	std::vector<size_t> offsets;
	offsets.reserve(entrypoints_bytes.size());

	size_t prev = 0;
	for (size_t i = 0; i < entrypoints_bytes.size(); ++i) {
		offsets.push_back(prev / sizeof(T));
		prev = entrypoints_bytes[i];
	}
	return offsets;
}

template <typename T>
struct BPParseResult {
	T*      packed;
	vbw_t*  bit_widths;
	size_t  n_packed;
	size_t* vector_offsets;
};

template <typename T>
inline BPParseResult<T> parse_bp_segments(const fastlanes::SegmentView& seg_bitpacked,
                                          const fastlanes::SegmentView& seg_bw,
                                          const size_t                  n_vecs) {
	auto entry = extract_entrypoints(seg_bitpacked);
	if (entry.size() != n_vecs) {
		throw std::runtime_error("bitpacked entrypoint count mismatch");
	}
	auto  offsets = build_vector_offsets<T>(entry);
	auto* packed  = copy_segment_array<T>(seg_bitpacked);
	auto* bws     = copy_segment_array<vbw_t>(seg_bw);

	auto* offsets_arr = new size_t[n_vecs];
	for (size_t i = 0; i < n_vecs; ++i) {
		offsets_arr[i] = offsets[i];
	}

	const size_t n_packed = seg_bitpacked.data_span.size() / sizeof(T);
	return BPParseResult<T> {packed, bws, n_packed, offsets_arr};
}

struct ExceptionOffsets {
	size_t* offsets;
	size_t  total;
};

inline ExceptionOffsets build_exception_offsets(const uint16_t* counts, const size_t n_vecs) {
	auto*  offsets = new size_t[n_vecs];
	size_t acc     = 0;
	for (size_t i = 0; i < n_vecs; ++i) {
		offsets[i] = acc;
		acc += counts[i];
	}
	return ExceptionOffsets {offsets, acc};
}

template <typename T>
inline ExceptionOffsets build_exception_offsets_from_segment(const fastlanes::SegmentView& seg, const size_t n_vecs) {
	const auto entry = extract_entrypoints(seg);
	if (entry.size() != n_vecs) {
		throw std::runtime_error("exception segment entrypoint count mismatch");
	}

	auto*  offsets    = new size_t[n_vecs];
	size_t prev_bytes = 0;
	for (size_t i = 0; i < n_vecs; ++i) {
		const size_t cur_bytes = static_cast<size_t>(entry[i]);
		if (cur_bytes < prev_bytes || (cur_bytes % sizeof(T)) != 0) {
			delete[] offsets;
			throw std::runtime_error("invalid exception segment entrypoints");
		}
		offsets[i] = prev_bytes / sizeof(T);
		prev_bytes = cur_bytes;
	}

	return ExceptionOffsets {offsets, prev_bytes / sizeof(T)};
}

} // namespace detail
} // namespace reader::columns

#endif // FLSGPU_COLUMNS_PARSE_COMMON_CUH
