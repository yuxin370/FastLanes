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

	const size_t segment_bytes = seg.data_span.size();
	auto*        offsets       = new size_t[n_vecs];
	size_t       prev_bytes    = 0;
	for (size_t i = 0; i < n_vecs; ++i) {
		const size_t cur_bytes = static_cast<size_t>(entry[i]);
		if (cur_bytes < prev_bytes || (cur_bytes % sizeof(T)) != 0 || cur_bytes > segment_bytes) {
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

namespace flsgpu::host::detail {

/// Expand global runs into a per-vector decompressed tmp array.
/// Shared by cross_rle, cross_rle_extended, and cross_rle_lane_mask.
template <typename UINT_T, uint32_t VEC_VALUES>
inline void expand_runs_into_vector(UINT_T*          tmp,
                                    const uint32_t   vec_base,
                                    const UINT_T*    values,
                                    const uint32_t*  lengths,
                                    const uint32_t*  run_positions,
                                    const uint32_t   r0,
                                    const uint32_t   r1) {
	for (uint32_t i = 0; i < VEC_VALUES; ++i)
		tmp[i] = UINT_T {};

	for (uint32_t r = r0; r < r1; ++r) {
		const uint32_t run_start_g = run_positions[r];
		const uint32_t run_len     = lengths[r];

		if (run_start_g + run_len <= vec_base)
			continue;
		if (run_start_g >= vec_base + VEC_VALUES)
			continue;

		uint32_t local_start = (run_start_g > vec_base) ? (run_start_g - vec_base) : 0u;
		uint32_t local_end   = run_start_g + run_len - vec_base;
		if (local_end > VEC_VALUES)
			local_end = VEC_VALUES;

		const UINT_T v = values[r];
		for (uint32_t p = local_start; p < local_end; ++p)
			tmp[p] = v;
	}
}

/// Parse raw cross-RLE segments (values + lengths) into run_positions + per-vector offsets.
/// Returns {values, lengths, run_positions, offsets} — caller owns all allocations.
template <typename T>
struct CrossRLERawRuns {
	using UINT_T = typename utils::same_width_uint<T>::type;
	UINT_T*   values;
	uint32_t* lengths;
	uint32_t* run_positions;
	uint32_t* offsets;
	size_t    n_runs;
};

template <typename T>
inline CrossRLERawRuns<T> parse_cross_rle_raw_runs(const reader::columns::ParseContext& ctx) {
	using UINT_T = typename utils::same_width_uint<T>::type;
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 2) {
		throw std::runtime_error("CROSS_RLE: missing operand tokens");
	}
	const size_t base_idx = ctx.operand_tokens->size() - 1;
	const auto   seg_vals = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto   seg_lens = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	const size_t n_runs  = seg_lens.data_span.size() / sizeof(uint32_t);
	auto*        values  = reader::columns::detail::copy_segment_array<UINT_T>(seg_vals);
	auto*        lengths = reader::columns::detail::copy_segment_array<uint32_t>(seg_lens);

	auto*    run_positions = new uint32_t[n_runs];
	uint32_t pos           = 0;
	for (size_t i = 0; i < n_runs; ++i) {
		run_positions[i] = pos;
		pos += lengths[i];
	}

	auto*    offsets = new uint32_t[ctx.n_vecs + 1];
	uint32_t cur     = 0;
	uint32_t idx_run = 0;
	for (size_t v = 0; v < ctx.n_vecs; ++v) {
		const size_t target_start = v * reader::columns::detail::kVecSize;
		while (idx_run < n_runs && cur + lengths[idx_run] <= target_start) {
			cur += lengths[idx_run];
			++idx_run;
		}
		offsets[v] = idx_run;
	}
	offsets[ctx.n_vecs] = static_cast<uint32_t>(n_runs);

	return CrossRLERawRuns<T> {values, lengths, run_positions, offsets, n_runs};
}

} // namespace flsgpu::host::detail

#endif // FLSGPU_COLUMNS_PARSE_COMMON_CUH
