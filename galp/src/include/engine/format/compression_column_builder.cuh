// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/format/compression_column_builder.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_FORMAT_CODEC_COLUMN_BUILDER_CUH
#define GALP_ENGINE_FORMAT_CODEC_COLUMN_BUILDER_CUH

#include "fls/reader/rowgroup_view.hpp"
#include "fls/reader/segment.hpp"
#include "compression/columns/all.cuh"
#include "compression/utils.cuh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace galp::format::detail {

inline constexpr size_t kVecSize = galp::codec::consts::VALUES_PER_VECTOR;

struct SegmentOffsets {
	uint32_t* offsets;
	size_t    total;
};

inline uint32_t checked_u32_offset(const size_t value, const char* field) {
	if (value > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
		std::ostringstream msg;
		msg << field << " exceeds uint32_t range: " << value;
		throw std::runtime_error(msg.str());
	}
	return static_cast<uint32_t>(value);
}

inline size_t checked_add_size(const size_t a, const size_t b, const char* field) {
	if (b > std::numeric_limits<size_t>::max() - a) {
		std::ostringstream msg;
		msg << field << " exceeds size_t range";
		throw std::runtime_error(msg.str());
	}
	return a + b;
}

struct ZeroCopyHostStorage {
	struct ScratchBlock {
		std::unique_ptr<std::byte[]> data;
		size_t                       capacity = 0;
		size_t                       used     = 0;
	};

	static constexpr size_t kScratchBlockBytes = 64U * 1024U;

	std::shared_ptr<void>                    backing_owner;
	std::shared_ptr<fastlanes::RowgroupView> rowgroup_view;
	std::vector<ScratchBlock>                scratch_blocks;

	template <typename T>
	T* allocate_array(const size_t n) {
		if (n == 0) {
			return nullptr;
		}

		const size_t bytes     = n * sizeof(T);
		const size_t alignment = std::max<size_t>(alignof(T), alignof(std::max_align_t));

		for (auto& block : scratch_blocks) {
			const uintptr_t base    = reinterpret_cast<uintptr_t>(block.data.get());
			const uintptr_t current = base + block.used;
			const uintptr_t aligned = (current + alignment - 1U) & ~(alignment - 1U);
			const size_t    next    = static_cast<size_t>(aligned - base) + bytes;
			if (next <= block.capacity) {
				block.used = next;
				return reinterpret_cast<T*>(aligned);
			}
		}

		const size_t capacity = std::max(kScratchBlockBytes, bytes + alignment);
		auto&        block    = scratch_blocks.emplace_back();
		block.data            = std::make_unique<std::byte[]>(capacity);
		block.capacity        = capacity;
		block.used            = 0;

		const uintptr_t base    = reinterpret_cast<uintptr_t>(block.data.get());
		const uintptr_t aligned = (base + alignment - 1U) & ~(alignment - 1U);
		block.used              = static_cast<size_t>(aligned - base) + bytes;
		return reinterpret_cast<T*>(aligned);
	}
};

class SmallBuildState {
public:
	explicit SmallBuildState(const size_t size)
	    : size_(size) {
		if (size_ <= inline_.size()) {
			data_ = inline_.data();
			std::fill(inline_.begin(), inline_.begin() + static_cast<std::ptrdiff_t>(size_), uint8_t {0});
		} else {
			heap_.assign(size_, uint8_t {0});
			data_ = heap_.data();
		}
	}

	uint8_t& operator[](const size_t idx) {
		return data_[idx];
	}

	const uint8_t& operator[](const size_t idx) const {
		return data_[idx];
	}

private:
	static constexpr size_t            kInlineStates = 256;
	size_t                             size_         = 0;
	std::array<uint8_t, kInlineStates> inline_ {};
	std::vector<uint8_t>               heap_;
	uint8_t*                           data_ = nullptr;
};

template <typename T>
inline T* segment_ptr_or_copy(const fastlanes::SegmentView& seg, ZeroCopyHostStorage& storage) {
	const size_t n_bytes = seg.data_span.size();
	if ((n_bytes % sizeof(T)) != 0) {
		throw std::runtime_error("segment byte size is not a multiple of target element size");
	}
	auto* raw = seg.data_span.data();
	if ((reinterpret_cast<uintptr_t>(raw) % alignof(T)) == 0) {
		return const_cast<T*>(reinterpret_cast<const T*>(raw));
	}

	const size_t n_elems = n_bytes / sizeof(T);
	auto*        out     = storage.allocate_array<T>(n_elems);
	std::memcpy(out, raw, n_bytes);
	return out;
}

template <typename Callback>
inline void for_each_entrypoint(const fastlanes::SegmentView& seg,
                                const size_t                  expected,
                                const char*                   segment_tag,
                                Callback&&                    callback) {
	size_t actual = 0;
	bool   ok     = false;
	std::visit(
	    [&](auto&& view) {
		    using V = std::decay_t<decltype(view)>;
		    if constexpr (std::is_same_v<V, std::monostate>) {
			    actual = 0;
		    } else {
			    actual = view.entrypoint_span.size();
			    if (actual != expected) {
				    return;
			    }
			    for (size_t i = 0; i < expected; ++i) {
				    callback(i, static_cast<size_t>(view.entrypoint_span[i]));
			    }
			    ok = true;
		    }
	    },
	    seg.entry_point_view);
	if (!ok) {
		std::ostringstream msg;
		msg << segment_tag << " entrypoint count mismatch: expected " << expected << ", got " << actual;
		throw std::runtime_error(msg.str());
	}
}

template <typename T>
inline galp::codec::host::BPColumn<T> make_bp_zero_copy(const fastlanes::SegmentView& seg_bitpacked,
                                                        const fastlanes::SegmentView& seg_bw,
                                                        const size_t                  n_values,
                                                        const size_t                  n_vecs,
                                                        ZeroCopyHostStorage&          storage) {
	using PackedT = typename galp::codec::utils::same_width_uint<T>::type;

	auto*  vector_offsets = storage.allocate_array<uint32_t>(n_vecs);
	size_t prev_bytes     = 0;
	for_each_entrypoint(seg_bitpacked, n_vecs, "bitpacked segment", [&](const size_t i, const size_t cur_bytes) {
		if (cur_bytes < prev_bytes || (cur_bytes % sizeof(PackedT)) != 0) {
			throw std::runtime_error("invalid bitpacked segment entrypoints");
		}
		vector_offsets[i] = checked_u32_offset(prev_bytes / sizeof(PackedT), "bitpacked vector offset");
		prev_bytes        = cur_bytes;
	});

	auto*        packed     = segment_ptr_or_copy<PackedT>(seg_bitpacked, storage);
	auto*        bit_widths = segment_ptr_or_copy<vbw_t>(seg_bw, storage);
	const size_t n_bp       = seg_bitpacked.data_span.size() / sizeof(PackedT);
	return galp::codec::host::BPColumn<T> {n_values,
	                                       n_bp,
	                                       galp::codec::host::borrow_array(packed),
	                                       galp::codec::host::borrow_array(bit_widths),
	                                       galp::codec::host::borrow_array(vector_offsets)};
}

template <typename T>
inline galp::codec::host::BPColumn<T>
make_uncompressed_zero_copy(const fastlanes::SegmentView& seg, const size_t n_values, ZeroCopyHostStorage& storage) {
	using UINT_T                  = typename galp::codec::utils::same_width_uint<T>::type;
	const size_t n_vecs           = galp::codec::utils::get_n_vecs_from_size(n_values);
	const size_t n_packed_values  = n_vecs * kVecSize;
	const size_t required_bytes   = n_packed_values * sizeof(UINT_T);
	const auto*  segment_data     = seg.data_span.data();
	const bool   segment_is_typed = segment_data != nullptr &&
	                              (reinterpret_cast<uintptr_t>(segment_data) % alignof(UINT_T)) == 0 &&
	                              seg.data_span.size() >= required_bytes;

	UINT_T* packed = nullptr;
	if (segment_is_typed) {
		packed = const_cast<UINT_T*>(reinterpret_cast<const UINT_T*>(segment_data));
	} else {
		packed = storage.allocate_array<UINT_T>(n_packed_values);
		if (required_bytes > 0) {
			std::memset(packed, 0, required_bytes);
		}
		const size_t copy_bytes = std::min(seg.data_span.size(), required_bytes);
		if (copy_bytes > 0) {
			std::memcpy(packed, segment_data, copy_bytes);
		}
	}

	auto*      bit_widths     = storage.allocate_array<vbw_t>(n_vecs);
	auto*      vector_offsets = storage.allocate_array<uint32_t>(n_vecs);
	const auto bw             = static_cast<vbw_t>(sizeof(T) * 8);
	for (size_t vi = 0; vi < n_vecs; ++vi) {
		bit_widths[vi]     = bw;
		vector_offsets[vi] = checked_u32_offset(vi * kVecSize, "uncompressed vector offset");
	}

	return galp::codec::host::BPColumn<T> {n_values,
	                                       n_packed_values,
	                                       galp::codec::host::borrow_array(packed),
	                                       galp::codec::host::borrow_array(bit_widths),
	                                       galp::codec::host::borrow_array(vector_offsets)};
}

template <typename T>
inline galp::codec::host::FFORColumn<T> make_ffor_zero_copy(const fastlanes::SegmentView& seg_bitpacked,
                                                            const fastlanes::SegmentView& seg_bw,
                                                            const fastlanes::SegmentView& seg_base,
                                                            const size_t                  n_values,
                                                            const size_t                  n_vecs,
                                                            ZeroCopyHostStorage&          storage) {
	using BaseT = typename galp::codec::utils::same_width_uint<T>::type;
	auto  bp    = make_bp_zero_copy<T>(seg_bitpacked, seg_bw, n_values, n_vecs, storage);
	auto* bases = segment_ptr_or_copy<BaseT>(seg_base, storage);
	return galp::codec::host::FFORColumn<T> {std::move(bp), galp::codec::host::borrow_array(bases)};
}

inline SegmentOffsets build_entrypoint_offsets(const fastlanes::SegmentView& seg,
                                               const size_t                  n_vecs,
                                               const size_t                  elem_bytes,
                                               ZeroCopyHostStorage&          storage,
                                               const char*                   segment_tag = "exception segment") {
	const size_t segment_bytes = seg.data_span.size();
	auto*        offsets       = storage.allocate_array<uint32_t>(n_vecs);
	size_t       prev_bytes    = 0;
	for_each_entrypoint(seg, n_vecs, segment_tag, [&](const size_t i, const size_t cur_bytes) {
		if (cur_bytes < prev_bytes || (cur_bytes % elem_bytes) != 0 || cur_bytes > segment_bytes) {
			throw std::runtime_error(std::string("invalid ") + segment_tag + " entrypoints");
		}
		offsets[i] = checked_u32_offset(prev_bytes / elem_bytes, segment_tag);
		prev_bytes = cur_bytes;
	});
	(void)checked_u32_offset(prev_bytes / elem_bytes, segment_tag);
	return SegmentOffsets {offsets, prev_bytes / elem_bytes};
}

inline bool validate_shared_position_offsets_enabled() {
	const char* env = std::getenv("GALP_VALIDATE_SHARED_POSITION_OFFSETS");
	return env != nullptr && std::strcmp(env, "0") != 0;
}

inline void validate_matching_entrypoint_offsets(const fastlanes::SegmentView& seg,
                                                 const uint32_t*               expected_offsets,
                                                 const size_t                  n_vecs,
                                                 const size_t                  elem_bytes,
                                                 const size_t                  expected_total,
                                                 const char*                   segment_tag) {
	if (!validate_shared_position_offsets_enabled()) {
		return;
	}

	const size_t segment_bytes = seg.data_span.size();
	size_t       prev_bytes    = 0;
	for_each_entrypoint(seg, n_vecs, segment_tag, [&](const size_t i, const size_t cur_bytes) {
		if (cur_bytes < prev_bytes || (cur_bytes % elem_bytes) != 0 || cur_bytes > segment_bytes) {
			throw std::runtime_error(std::string("invalid ") + segment_tag + " entrypoints");
		}
		const size_t offset = prev_bytes / elem_bytes;
		if (checked_u32_offset(offset, segment_tag) != expected_offsets[i]) {
			throw std::runtime_error(std::string(segment_tag) + " offsets do not match exception offsets");
		}
		prev_bytes = cur_bytes;
	});
	const size_t total = prev_bytes / elem_bytes;
	if (total != expected_total) {
		throw std::runtime_error(std::string(segment_tag) + " total does not match exception total");
	}
}

template <typename T>
inline galp::codec::host::SLPATCHColumn<T> make_slpatch_zero_copy(const fastlanes::SegmentView& seg_exc,
                                                                  const fastlanes::SegmentView& seg_pos,
                                                                  const fastlanes::SegmentView& seg_cnt,
                                                                  const fastlanes::SegmentView& seg_bitpacked,
                                                                  const fastlanes::SegmentView& seg_bw,
                                                                  const fastlanes::SegmentView& seg_base,
                                                                  const size_t                  n_values,
                                                                  const size_t                  n_vecs,
                                                                  ZeroCopyHostStorage&          storage) {
	auto ffor = make_ffor_zero_copy<T>(seg_bitpacked, seg_bw, seg_base, n_values, n_vecs, storage);

	auto exc = build_entrypoint_offsets(seg_exc, n_vecs, sizeof(T), storage);
	validate_matching_entrypoint_offsets(
	    seg_pos, exc.offsets, n_vecs, sizeof(uint16_t), exc.total, "SLPATCH positions segment");

	auto* counts     = segment_ptr_or_copy<uint16_t>(seg_cnt, storage);
	auto* positions  = segment_ptr_or_copy<uint16_t>(seg_pos, storage);
	auto* exceptions = segment_ptr_or_copy<T>(seg_exc, storage);

	return galp::codec::host::SLPATCHColumn<T> {n_values,
	                                            n_vecs,
	                                            std::move(ffor),
	                                            exc.total,
	                                            galp::codec::host::borrow_array(exc.offsets),
	                                            galp::codec::host::borrow_array(exceptions),
	                                            galp::codec::host::borrow_array(positions),
	                                            galp::codec::host::borrow_array(counts)};
}

template <typename T>
inline galp::codec::host::FREQColumn<T> make_frequency_zero_copy(const fastlanes::SegmentView& seg_fv,
                                                                 const fastlanes::SegmentView& seg_exc,
                                                                 const fastlanes::SegmentView& seg_pos,
                                                                 const fastlanes::SegmentView& seg_cnt,
                                                                 const size_t                  n_values,
                                                                 const size_t                  n_vecs,
                                                                 ZeroCopyHostStorage&          storage) {
	if (seg_fv.data_span.size() != sizeof(T)) {
		throw std::runtime_error("EXP_FREQUENCY: invalid frequent value size");
	}
	T fv;
	std::memcpy(&fv, seg_fv.data_span.data(), sizeof(T));

	auto exc = build_entrypoint_offsets(seg_exc, n_vecs, sizeof(T), storage);
	validate_matching_entrypoint_offsets(
	    seg_pos, exc.offsets, n_vecs, sizeof(uint16_t), exc.total, "EXP_FREQUENCY positions segment");

	auto* counts     = segment_ptr_or_copy<uint16_t>(seg_cnt, storage);
	auto* positions  = segment_ptr_or_copy<uint16_t>(seg_pos, storage);
	auto* exceptions = segment_ptr_or_copy<T>(seg_exc, storage);

	return galp::codec::host::FREQColumn<T> {n_values,
	                                         n_vecs,
	                                         fv,
	                                         exc.total,
	                                         galp::codec::host::borrow_array(exc.offsets),
	                                         galp::codec::host::borrow_array(exceptions),
	                                         galp::codec::host::borrow_array(positions),
	                                         galp::codec::host::borrow_array(counts)};
}

template <typename T>
inline galp::codec::host::CROSSRLEColumn<T> make_cross_rle_zero_copy(const fastlanes::SegmentView& seg_vals,
                                                                     const fastlanes::SegmentView& seg_lens,
                                                                     const size_t                  n_values,
                                                                     const size_t                  n_vecs,
                                                                     ZeroCopyHostStorage&          storage) {
	using ValueT = typename galp::codec::utils::same_width_uint<T>::type;
	if ((seg_lens.data_span.size() % sizeof(uint32_t)) != 0) {
		throw std::runtime_error("EXP_CROSS_RLE lengths segment byte size is not uint32_t-aligned");
	}
	if ((seg_vals.data_span.size() % sizeof(ValueT)) != 0) {
		throw std::runtime_error("EXP_CROSS_RLE values segment byte size is not value-aligned");
	}
	const size_t n_runs = seg_lens.data_span.size() / sizeof(uint32_t);
	const size_t n_vals = seg_vals.data_span.size() / sizeof(ValueT);
	if (n_runs == 0 && n_values > 0) {
		throw std::runtime_error("EXP_CROSS_RLE has no runs for a non-empty column");
	}
	if (n_vals < n_runs) {
		std::ostringstream msg;
		msg << "EXP_CROSS_RLE values segment too short: values=" << n_vals << " runs=" << n_runs;
		throw std::runtime_error(msg.str());
	}
	if (n_runs > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
		throw std::runtime_error("EXP_CROSS_RLE run count exceeds uint32_t range");
	}
	if (n_vecs > std::numeric_limits<size_t>::max() / kVecSize) {
		throw std::runtime_error("EXP_CROSS_RLE vector count exceeds size_t range");
	}
	auto* values  = segment_ptr_or_copy<ValueT>(seg_vals, storage);
	auto* lengths = segment_ptr_or_copy<uint32_t>(seg_lens, storage);

	auto*  run_positions = storage.allocate_array<uint32_t>(n_runs);
	size_t pos           = 0;
	for (size_t i = 0; i < n_runs; ++i) {
		if (lengths[i] == 0) {
			throw std::runtime_error("EXP_CROSS_RLE contains a zero-length run");
		}
		run_positions[i] = checked_u32_offset(pos, "EXP_CROSS_RLE run position");
		pos              = checked_add_size(pos, static_cast<size_t>(lengths[i]), "EXP_CROSS_RLE run length total");
	}
	if (pos < n_values) {
		std::ostringstream msg;
		msg << "EXP_CROSS_RLE run lengths cover " << pos << " values, expected at least " << n_values;
		throw std::runtime_error(msg.str());
	}

	auto*  offsets = storage.allocate_array<uint32_t>(n_vecs + 1);
	size_t cur     = 0;
	size_t idx_run = 0;
	for (size_t v = 0; v < n_vecs; ++v) {
		const size_t target_start = v * kVecSize;
		while (idx_run < n_runs &&
		       checked_add_size(cur, static_cast<size_t>(lengths[idx_run]), "EXP_CROSS_RLE vector offset") <=
		           target_start) {
			cur = checked_add_size(cur, static_cast<size_t>(lengths[idx_run]), "EXP_CROSS_RLE vector offset");
			++idx_run;
		}
		offsets[v] = checked_u32_offset(idx_run, "EXP_CROSS_RLE vector run offset");
	}
	offsets[n_vecs] = checked_u32_offset(n_runs, "EXP_CROSS_RLE final run offset");

	return galp::codec::host::CROSSRLEColumn<T> {n_values,
	                                             n_runs,
	                                             galp::codec::host::borrow_array(values),
	                                             galp::codec::host::borrow_array(lengths),
	                                             galp::codec::host::borrow_array(offsets),
	                                             galp::codec::host::borrow_array(run_positions)};
}

template <typename T, typename IndexT = typename galp::codec::utils::same_width_uint<T>::type>
inline galp::codec::host::DICTFFORColumn<T, IndexT>
make_dict_ffor_zero_copy(const fastlanes::SegmentView& seg_keys,
                         const fastlanes::SegmentView& seg_bitpacked,
                         const fastlanes::SegmentView& seg_bw,
                         const fastlanes::SegmentView& seg_base,
                         const size_t                  n_values,
                         const size_t                  n_vecs,
                         ZeroCopyHostStorage&          storage) {
	using KeyT = typename galp::codec::utils::same_width_uint<T>::type;
	auto  ffor = make_ffor_zero_copy<IndexT>(seg_bitpacked, seg_bw, seg_base, n_values, n_vecs, storage);
	auto* keys = segment_ptr_or_copy<KeyT>(seg_keys, storage);
	return galp::codec::host::DICTFFORColumn<T, IndexT> {
	    std::move(ffor), galp::codec::host::borrow_array(keys), seg_keys.data_span.size() / sizeof(KeyT)};
}

template <typename T, typename IndexT = typename galp::codec::utils::same_width_uint<T>::type>
inline galp::codec::host::DICTREFColumn<T, IndexT> make_dict_ref_zero_copy(const fastlanes::SegmentView& seg_keys,
                                                                           const uint32_t                index_col_idx,
                                                                           const size_t                  n_values,
                                                                           ZeroCopyHostStorage&          storage) {
	using KeyT = typename galp::codec::utils::same_width_uint<T>::type;
	auto* keys = segment_ptr_or_copy<KeyT>(seg_keys, storage);
	return galp::codec::host::DICTREFColumn<T, IndexT> {
	    n_values, index_col_idx, galp::codec::host::borrow_array(keys), seg_keys.data_span.size() / sizeof(KeyT)};
}

template <typename T, typename IndexT>
inline galp::codec::host::RLEColumn<T, IndexT> make_rle_zero_copy(const fastlanes::SegmentView& seg_vals,
                                                                  const fastlanes::SegmentView& seg_rsum,
                                                                  const fastlanes::SegmentView& seg_bitpacked,
                                                                  const fastlanes::SegmentView& seg_bw,
                                                                  const fastlanes::SegmentView& seg_base,
                                                                  const size_t                  n_values,
                                                                  const size_t                  n_vecs,
                                                                  ZeroCopyHostStorage&          storage) {
	auto ffor = make_ffor_zero_copy<IndexT>(seg_bitpacked, seg_bw, seg_base, n_values, n_vecs, storage);

	const size_t expected_bases = n_vecs * galp::codec::utils::get_n_lanes<IndexT>();
	if (seg_rsum.data_span.size() / sizeof(IndexT) != expected_bases) {
		throw std::runtime_error("EXP_RLE: rsum bases size mismatch");
	}

	const auto rle_offsets_info =
	    build_entrypoint_offsets(seg_vals, n_vecs, sizeof(T), storage, "EXP_RLE values segment");
	auto* const offsets = rle_offsets_info.offsets;

	auto*        rsum_bases   = segment_ptr_or_copy<IndexT>(seg_rsum, storage);
	auto*        values       = segment_ptr_or_copy<T>(seg_vals, storage);
	const size_t n_rle_values = seg_vals.data_span.size() / sizeof(T);

	return galp::codec::host::RLEColumn<T, IndexT> {n_values,
	                                                n_vecs,
	                                                std::move(ffor),
	                                                galp::codec::host::borrow_array(rsum_bases),
	                                                galp::codec::host::borrow_array(values),
	                                                galp::codec::host::borrow_array(offsets),
	                                                n_rle_values};
}

template <typename T>
inline T* clone_array(const T* in, const size_t n_elements) {
	if (n_elements == 0) {
		return nullptr;
	}
	if (in == nullptr) {
		throw std::runtime_error("cannot clone null column array");
	}
	auto* out = new T[n_elements];
	std::memcpy(out, in, sizeof(T) * n_elements);
	return out;
}

template <typename T>
inline galp::codec::host::BPColumn<T> clone_column(const galp::codec::host::BPColumn<T>& col) {
	return galp::codec::host::BPColumn<T> {col.n_values,
	                                       col.n_packed_values,
	                                       clone_array(col.packed_array.get(), col.n_packed_values),
	                                       clone_array(col.bit_widths.get(), col.get_n_vecs()),
	                                       clone_array(col.vector_offsets.get(), col.get_n_vecs())};
}

template <typename T>
inline galp::codec::host::FFORColumn<T> clone_column(const galp::codec::host::FFORColumn<T>& col) {
	return galp::codec::host::FFORColumn<T> {clone_column(col.bp), clone_array(col.bases.get(), col.get_n_vecs())};
}

template <typename T>
inline galp::codec::host::CONSTANTColumn<T> clone_column(const galp::codec::host::CONSTANTColumn<T>& col) {
	return col;
}

template <typename T>
inline galp::codec::host::FREQColumn<T> clone_column(const galp::codec::host::FREQColumn<T>& col) {
	return galp::codec::host::FREQColumn<T> {col.n_values,
	                                         col.n_vecs,
	                                         col.frequent_value,
	                                         col.n_exceptions,
	                                         clone_array(col.exceptions_offsets.get(), col.n_vecs),
	                                         clone_array(col.exceptions.get(), col.n_exceptions),
	                                         clone_array(col.positions.get(), col.n_exceptions),
	                                         clone_array(col.counts.get(), col.n_vecs)};
}

template <typename T>
inline galp::codec::host::SLPATCHColumn<T> clone_column(const galp::codec::host::SLPATCHColumn<T>& col) {
	return galp::codec::host::SLPATCHColumn<T> {col.n_values,
	                                            col.n_vecs,
	                                            clone_column(col.ffor),
	                                            col.n_exceptions,
	                                            clone_array(col.exceptions_offsets.get(), col.n_vecs),
	                                            clone_array(col.exceptions.get(), col.n_exceptions),
	                                            clone_array(col.positions.get(), col.n_exceptions),
	                                            clone_array(col.counts.get(), col.n_vecs)};
}

template <typename T>
inline galp::codec::host::CROSSRLEColumn<T> clone_column(const galp::codec::host::CROSSRLEColumn<T>& col) {
	const size_t n_vecs = col.get_n_vecs();
	return galp::codec::host::CROSSRLEColumn<T> {col.n_values,
	                                             col.n_runs,
	                                             clone_array(col.values.get(), col.n_runs),
	                                             clone_array(col.lengths.get(), col.n_runs),
	                                             clone_array(col.offsets.get(), n_vecs + 1),
	                                             clone_array(col.run_positions.get(), col.n_runs)};
}

template <typename T, typename IndexT>
inline galp::codec::host::DICTFFORColumn<T, IndexT>
clone_column(const galp::codec::host::DICTFFORColumn<T, IndexT>& col) {
	return galp::codec::host::DICTFFORColumn<T, IndexT> {
	    clone_column(col.ffor), clone_array(col.keys.get(), col.key_count), col.key_count};
}

template <typename T, typename IndexT>
inline galp::codec::host::DICTREFColumn<T, IndexT>
clone_column(const galp::codec::host::DICTREFColumn<T, IndexT>& col) {
	return galp::codec::host::DICTREFColumn<T, IndexT> {
	    col.n_values, col.index_column_index, clone_array(col.keys.get(), col.key_count), col.key_count};
}

template <typename T, typename IndexT>
inline galp::codec::host::DICTSLPATCHColumn<T, IndexT>
clone_column(const galp::codec::host::DICTSLPATCHColumn<T, IndexT>& col) {
	return galp::codec::host::DICTSLPATCHColumn<T, IndexT> {
	    clone_column(col.index), clone_array(col.keys.get(), col.key_count), col.key_count};
}

template <typename T, typename IndexT>
inline galp::codec::host::RLEColumn<T, IndexT> clone_column(const galp::codec::host::RLEColumn<T, IndexT>& col) {
	return galp::codec::host::RLEColumn<T, IndexT> {
	    col.n_values,
	    col.n_vecs,
	    clone_column(col.ffor),
	    clone_array(col.rsum_bases.get(), col.n_vecs * galp::codec::utils::get_n_lanes<IndexT>()),
	    clone_array(col.rle_values.get(), col.n_rle_values),
	    clone_array(col.rle_offsets.get(), col.n_vecs),
	    col.n_rle_values};
}

} // namespace galp::format::detail

#endif // GALP_ENGINE_FORMAT_CODEC_COLUMN_BUILDER_CUH
