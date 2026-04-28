// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/reader.cuh
// ────────────────────────────────────────────────────────
#ifndef FLS_READER_CUH
#define FLS_READER_CUH

#include "engine/data/model.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/operator_token_generated.h"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include "fls/reader/column_view.hpp"
#include "fls/reader/rowgroup_view.hpp"
#include "fls/reader/segment.hpp"
#include "flsgpu/columns/all.cuh"
#include "flsgpu/flsgpu.cuh"
#include "flsgpu/utils.cuh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace reader {

namespace detail {
inline constexpr size_t kVecSize = consts::VALUES_PER_VECTOR;

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

inline fastlanes::TableDescriptorHandle load_table_descriptor(fastlanes::File&             file,
                                                              const std::filesystem::path& file_path) {
	fastlanes::FileHeader file_header {};
	fastlanes::FileFooter file_footer {};

	fastlanes::FileHeader::Load(file_header, file);
	fastlanes::FileFooter::Load(file_footer, file);

	if (file_header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file, file_footer.table_descriptor_offset, file_footer.table_descriptor_size, /*verify=*/true);
	}

	const auto footer_path = file_path.parent_path() / "table_descriptor.fbb";
	return fastlanes::TableDescriptorHandle::FromFile(footer_path, /*verify=*/true);
}

inline fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::File file(file_path);
	return load_table_descriptor(file, file_path);
}

struct ZeroCopyHostStorage {
	struct ScratchBlock {
		std::unique_ptr<std::byte[]> data;
		size_t                       capacity = 0;
		size_t                       used     = 0;
	};

	static constexpr size_t kScratchBlockBytes = 64U * 1024U;

	// Destruction order matters: scratch blocks first, then rowgroup_view
	// (references backing_owner), then backing_owner last.
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
inline flsgpu::host::BPColumn<T> make_bp_zero_copy(const fastlanes::SegmentView& seg_bitpacked,
                                                   const fastlanes::SegmentView& seg_bw,
                                                   const size_t                  n_values,
                                                   const size_t                  n_vecs,
                                                   ZeroCopyHostStorage&          storage) {
	using PackedT = typename utils::same_width_uint<T>::type;

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
	return flsgpu::host::BPColumn<T> {n_values, n_bp, packed, bit_widths, vector_offsets};
}

template <typename T>
inline flsgpu::host::BPColumn<T>
make_uncompressed_zero_copy(const fastlanes::SegmentView& seg, const size_t n_values, ZeroCopyHostStorage& storage) {
	using UINT_T                  = typename utils::same_width_uint<T>::type;
	const size_t n_vecs           = utils::get_n_vecs_from_size(n_values);
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

	return flsgpu::host::BPColumn<T> {n_values, n_packed_values, packed, bit_widths, vector_offsets};
}

template <typename T>
inline flsgpu::host::FFORColumn<T> make_ffor_zero_copy(const fastlanes::SegmentView& seg_bitpacked,
                                                       const fastlanes::SegmentView& seg_bw,
                                                       const fastlanes::SegmentView& seg_base,
                                                       const size_t                  n_values,
                                                       const size_t                  n_vecs,
                                                       ZeroCopyHostStorage&          storage) {
	using BaseT = typename utils::same_width_uint<T>::type;
	auto  bp    = make_bp_zero_copy<T>(seg_bitpacked, seg_bw, n_values, n_vecs, storage);
	auto* bases = segment_ptr_or_copy<BaseT>(seg_base, storage);
	return flsgpu::host::FFORColumn<T> {bp, bases};
}

// Shared helper: parse a segment's entrypoints into per-vector element offsets,
// validating monotonicity, alignment against the element size, and that the
// final cumulative byte count fits within the segment payload. Offsets are
// carved from the zero-copy scratch arena rather than heap-allocated. The
// segment_tag prefixes thrown error messages so operators can identify which
// segment (e.g. "exception segment", "EXP_RLE values") is malformed.
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
inline flsgpu::host::SLPATCHColumn<T> make_slpatch_zero_copy(const fastlanes::SegmentView& seg_exc,
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

	return flsgpu::host::SLPATCHColumn<T> {
	    n_values, n_vecs, ffor, exc.total, exc.offsets, exceptions, positions, counts};
}

template <typename T>
inline flsgpu::host::FREQColumn<T> make_frequency_zero_copy(const fastlanes::SegmentView& seg_fv,
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

	return flsgpu::host::FREQColumn<T> {
	    n_values, n_vecs, fv, exc.total, exc.offsets, exceptions, positions, counts};
}

template <typename T>
inline flsgpu::host::CROSSRLEColumn<T> make_cross_rle_zero_copy(const fastlanes::SegmentView& seg_vals,
                                                                const fastlanes::SegmentView& seg_lens,
                                                                const size_t                  n_values,
                                                                const size_t                  n_vecs,
                                                                ZeroCopyHostStorage&          storage) {
	using ValueT         = typename utils::same_width_uint<T>::type;
	const size_t n_runs  = seg_lens.data_span.size() / sizeof(uint32_t);
	auto*        values  = segment_ptr_or_copy<ValueT>(seg_vals, storage);
	auto*        lengths = segment_ptr_or_copy<uint32_t>(seg_lens, storage);

	auto*    run_positions = storage.allocate_array<uint32_t>(n_runs);
	uint32_t pos           = 0;
	for (size_t i = 0; i < n_runs; ++i) {
		run_positions[i] = pos;
		pos += lengths[i];
	}

	auto*    offsets = storage.allocate_array<uint32_t>(n_vecs + 1);
	uint32_t cur     = 0;
	uint32_t idx_run = 0;
	for (size_t v = 0; v < n_vecs; ++v) {
		const size_t target_start = v * kVecSize;
		while (idx_run < n_runs && cur + lengths[idx_run] <= target_start) {
			cur += lengths[idx_run];
			++idx_run;
		}
		offsets[v] = idx_run;
	}
	offsets[n_vecs] = static_cast<uint32_t>(n_runs);

	return flsgpu::host::CROSSRLEColumn<T> {n_values, n_runs, values, lengths, offsets, run_positions};
}

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
inline flsgpu::host::DICTFFORColumn<T, IndexT> make_dict_ffor_zero_copy(const fastlanes::SegmentView& seg_keys,
                                                                        const fastlanes::SegmentView& seg_bitpacked,
                                                                        const fastlanes::SegmentView& seg_bw,
                                                                        const fastlanes::SegmentView& seg_base,
                                                                        const size_t                  n_values,
                                                                        const size_t                  n_vecs,
                                                                        ZeroCopyHostStorage&          storage) {
	using KeyT = typename utils::same_width_uint<T>::type;
	auto  ffor = make_ffor_zero_copy<IndexT>(seg_bitpacked, seg_bw, seg_base, n_values, n_vecs, storage);
	auto* keys = segment_ptr_or_copy<KeyT>(seg_keys, storage);
	return flsgpu::host::DICTFFORColumn<T, IndexT> {std::move(ffor), keys, seg_keys.data_span.size() / sizeof(KeyT)};
}

template <typename T, typename IndexT = typename utils::same_width_uint<T>::type>
inline flsgpu::host::DICTREFColumn<T, IndexT> make_dict_ref_zero_copy(const fastlanes::SegmentView& seg_keys,
                                                                      const uint32_t                index_col_idx,
                                                                      const size_t                  n_values,
                                                                      ZeroCopyHostStorage&          storage) {
	using KeyT = typename utils::same_width_uint<T>::type;
	auto* keys = segment_ptr_or_copy<KeyT>(seg_keys, storage);
	return flsgpu::host::DICTREFColumn<T, IndexT> {
	    n_values, index_col_idx, keys, seg_keys.data_span.size() / sizeof(KeyT)};
}

template <typename T, typename IndexT>
inline flsgpu::host::RLEColumn<T, IndexT> make_rle_zero_copy(const fastlanes::SegmentView& seg_vals,
                                                             const fastlanes::SegmentView& seg_rsum,
                                                             const fastlanes::SegmentView& seg_bitpacked,
                                                             const fastlanes::SegmentView& seg_bw,
                                                             const fastlanes::SegmentView& seg_base,
                                                             const size_t                  n_values,
                                                             const size_t                  n_vecs,
                                                             ZeroCopyHostStorage&          storage) {
	auto ffor = make_ffor_zero_copy<IndexT>(seg_bitpacked, seg_bw, seg_base, n_values, n_vecs, storage);

	const size_t expected_bases = n_vecs * utils::get_n_lanes<IndexT>();
	if (seg_rsum.data_span.size() / sizeof(IndexT) != expected_bases) {
		throw std::runtime_error("EXP_RLE: rsum bases size mismatch");
	}

	const auto rle_offsets_info =
	    build_entrypoint_offsets(seg_vals, n_vecs, sizeof(T), storage, "EXP_RLE values segment");
	auto* const offsets = rle_offsets_info.offsets;

	auto*        rsum_bases   = segment_ptr_or_copy<IndexT>(seg_rsum, storage);
	auto*        values       = segment_ptr_or_copy<T>(seg_vals, storage);
	const size_t n_rle_values = seg_vals.data_span.size() / sizeof(T);

	return flsgpu::host::RLEColumn<T, IndexT> {
	    n_values, n_vecs, std::move(ffor), rsum_bases, values, offsets, n_rle_values};
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
inline flsgpu::host::BPColumn<T> clone_column(const flsgpu::host::BPColumn<T>& col) {
	return flsgpu::host::BPColumn<T> {col.n_values,
	                                  col.n_packed_values,
	                                  clone_array(col.packed_array, col.n_packed_values),
	                                  clone_array(col.bit_widths, col.get_n_vecs()),
	                                  clone_array(col.vector_offsets, col.get_n_vecs())};
}

template <typename T>
inline flsgpu::host::FFORColumn<T> clone_column(const flsgpu::host::FFORColumn<T>& col) {
	return flsgpu::host::FFORColumn<T> {clone_column(col.bp), clone_array(col.bases, col.get_n_vecs())};
}

template <typename T>
inline flsgpu::host::CONSTANTColumn<T> clone_column(const flsgpu::host::CONSTANTColumn<T>& col) {
	return col;
}

template <typename T>
inline flsgpu::host::FREQColumn<T> clone_column(const flsgpu::host::FREQColumn<T>& col) {
	return flsgpu::host::FREQColumn<T> {col.n_values,
	                                    col.n_vecs,
	                                    col.frequent_value,
	                                    col.n_exceptions,
	                                    clone_array(col.exceptions_offsets, col.n_vecs),
	                                    clone_array(col.exceptions, col.n_exceptions),
	                                    clone_array(col.positions, col.n_exceptions),
	                                    clone_array(col.counts, col.n_vecs)};
}

template <typename T>
inline flsgpu::host::SLPATCHColumn<T> clone_column(const flsgpu::host::SLPATCHColumn<T>& col) {
	return flsgpu::host::SLPATCHColumn<T> {col.n_values,
	                                       col.n_vecs,
	                                       clone_column(col.ffor),
	                                       col.n_exceptions,
	                                       clone_array(col.exceptions_offsets, col.n_vecs),
	                                       clone_array(col.exceptions, col.n_exceptions),
	                                       clone_array(col.positions, col.n_exceptions),
	                                       clone_array(col.counts, col.n_vecs)};
}

template <typename T>
inline flsgpu::host::CROSSRLEColumn<T> clone_column(const flsgpu::host::CROSSRLEColumn<T>& col) {
	const size_t n_vecs = col.get_n_vecs();
	return flsgpu::host::CROSSRLEColumn<T> {col.n_values,
	                                        col.n_runs,
	                                        clone_array(col.values, col.n_runs),
	                                        clone_array(col.lengths, col.n_runs),
	                                        clone_array(col.offsets, n_vecs + 1),
	                                        clone_array(col.run_positions, col.n_runs)};
}

template <typename T, typename IndexT>
inline flsgpu::host::DICTFFORColumn<T, IndexT> clone_column(const flsgpu::host::DICTFFORColumn<T, IndexT>& col) {
	return flsgpu::host::DICTFFORColumn<T, IndexT> {
	    clone_column(col.ffor), clone_array(col.keys, col.key_count), col.key_count};
}

template <typename T, typename IndexT>
inline flsgpu::host::DICTREFColumn<T, IndexT> clone_column(const flsgpu::host::DICTREFColumn<T, IndexT>& col) {
	return flsgpu::host::DICTREFColumn<T, IndexT> {
	    col.n_values, col.index_column_index, clone_array(col.keys, col.key_count), col.key_count};
}

template <typename T, typename IndexT>
inline flsgpu::host::DICTSLPATCHColumn<T, IndexT> clone_column(
    const flsgpu::host::DICTSLPATCHColumn<T, IndexT>& col) {
	return flsgpu::host::DICTSLPATCHColumn<T, IndexT> {
	    clone_column(col.index), clone_array(col.keys, col.key_count), col.key_count};
}

template <typename T, typename IndexT>
inline flsgpu::host::RLEColumn<T, IndexT> clone_column(const flsgpu::host::RLEColumn<T, IndexT>& col) {
	return flsgpu::host::RLEColumn<T, IndexT> {col.n_values,
	                                           col.n_vecs,
	                                           clone_column(col.ffor),
	                                           clone_array(col.rsum_bases,
	                                                       col.n_vecs * utils::get_n_lanes<IndexT>()),
	                                           clone_array(col.rle_values, col.n_rle_values),
	                                           clone_array(col.rle_offsets, col.n_vecs),
	                                           col.n_rle_values};
}

} // namespace detail

using HostColumnVariant = dispatch::EncodedPayload;
using Column            = dispatch::Column;
using Rowgroup          = dispatch::Rowgroup;

struct ZeroCopyColumnPlan {
	size_t                   column_index = 0;
	std::string              name;
	fastlanes::OperatorToken token = fastlanes::OperatorToken::INVALID;
	std::vector<uint32_t>    operand_ids;
	bool                     skip_decompress = false;
	std::optional<size_t>    alias_of;
};

struct ZeroCopySchemaPlan {
	std::vector<ZeroCopyColumnPlan> columns;
	std::vector<size_t>             build_order;
	bool                            enabled = false;
};

struct ZeroCopyColumn {
	size_t                               column_index = 0;
	std::string                          name;
	const std::string*                   name_ref          = nullptr;
	fastlanes::OperatorToken             token             = fastlanes::OperatorToken::INVALID;
	const fastlanes::ColumnDescriptor*   column_descriptor = nullptr;
	const flatbuffers::Vector<uint64_t>* operand_tokens    = nullptr;
	const std::vector<uint32_t>*         operand_ids       = nullptr;
	const fastlanes::ColumnView*         column_view       = nullptr;
	fastlanes::span<std::byte>           column_span;
	bool                                 skip_decompress = false;
	std::optional<size_t>                alias_of;
};

inline size_t zero_copy_operand_count(const ZeroCopyColumn& col) {
	if (col.operand_ids != nullptr) {
		return col.operand_ids->size();
	}
	return col.operand_tokens != nullptr ? col.operand_tokens->size() : 0;
}

inline uint32_t zero_copy_operand(const ZeroCopyColumn& col, const size_t idx) {
	if (col.operand_ids != nullptr) {
		if (idx >= col.operand_ids->size()) {
			throw std::out_of_range("zero-copy operand index out of range");
		}
		return (*col.operand_ids)[idx];
	}
	if (col.operand_tokens == nullptr || idx >= col.operand_tokens->size()) {
		throw std::out_of_range("zero-copy operand index out of range");
	}
	return static_cast<uint32_t>(col.operand_tokens->Get(static_cast<flatbuffers::uoffset_t>(idx)));
}

inline const std::string& zero_copy_column_name(const ZeroCopyColumn& col) {
	if (col.name_ref != nullptr) {
		return *col.name_ref;
	}
	return col.name;
}

inline fastlanes::SegmentView zero_copy_segment(const ZeroCopyColumn& col, const uint32_t segment_idx) {
	if (col.column_view != nullptr) {
		return col.column_view->GetSegment(segment_idx);
	}
	if (col.column_descriptor == nullptr) {
		throw std::runtime_error("zero-copy column descriptor missing");
	}
	const auto* segment_descriptors = col.column_descriptor->segment_descriptors();
	if (segment_descriptors == nullptr || segment_idx >= segment_descriptors->size()) {
		throw std::out_of_range("zero-copy segment index out of range");
	}
	return fastlanes::make_segment_view(col.column_span,
	                                    *segment_descriptors->Get(static_cast<flatbuffers::uoffset_t>(segment_idx)));
}

struct ZeroCopyRowgroup {
	size_t                                                  rowgroup_index = 0;
	size_t                                                  n_values       = 0;
	size_t                                                  n_vecs         = 0;
	size_t                                                  n_tuples       = 0;
	std::shared_ptr<const fastlanes::TableDescriptorHandle> table_descriptor_owner;
	std::shared_ptr<const ZeroCopySchemaPlan>               schema_plan_owner;
	const fastlanes::RowgroupDescriptor*                    rowgroup_descriptor = nullptr;
	const ZeroCopySchemaPlan*                               schema_plan         = nullptr;
	std::shared_ptr<void>                                   backing_owner;
	fastlanes::span<std::byte>                              backing_span;
	// True only if backing_span points into CUDA-pinned memory. Controls
	// whether DeviceArena may issue direct cudaMemcpyAsync from it; registering
	// a pageable buffer as a direct backing forces the async copy to fall back
	// to a synchronous staged transfer internally.
	bool                                     backing_is_pinned = false;
	std::shared_ptr<fastlanes::RowgroupView> rowgroup_view;
	std::vector<ZeroCopyColumn>              columns;
};

inline ZeroCopyColumn make_zero_copy_column_from_plan(const ZeroCopyRowgroup&   rowgroup,
                                                      const ZeroCopyColumnPlan& plan_col) {
	if (rowgroup.rowgroup_descriptor == nullptr) {
		throw std::runtime_error("zero-copy rowgroup plan metadata missing");
	}
	const auto* col_descs = rowgroup.rowgroup_descriptor->m_column_descriptors();
	if (col_descs == nullptr || plan_col.column_index >= col_descs->size()) {
		throw std::out_of_range("zero-copy column plan index out of range");
	}

	ZeroCopyColumn col {};
	col.column_index      = plan_col.column_index;
	col.name_ref          = &plan_col.name;
	col.token             = plan_col.token;
	col.column_descriptor = col_descs->Get(static_cast<flatbuffers::uoffset_t>(plan_col.column_index));
	col.operand_ids       = &plan_col.operand_ids;
	col.column_view       = rowgroup.rowgroup_view != nullptr
	                            ? &(*rowgroup.rowgroup_view)[static_cast<fastlanes::n_t>(plan_col.column_index)]
	                            : nullptr;
	col.column_span       = rowgroup.backing_span;
	col.skip_decompress   = plan_col.skip_decompress;
	col.alias_of          = plan_col.alias_of;
	return col;
}

class reader {
public:
	explicit reader(const std::filesystem::path& file_path, const bool load_column_names = true)
	    : m_file(std::make_shared<fastlanes::File>(file_path))
	    , m_table_descriptor(
	          std::make_shared<fastlanes::TableDescriptorHandle>(detail::load_table_descriptor(*m_file, file_path)))
	    , m_load_column_names(load_column_names) {
		m_zero_copy_schema_plan = std::make_shared<ZeroCopySchemaPlan>(build_shared_zero_copy_schema_plan());
	}

	size_t rowgroup_count() const {
		const auto* td = m_table_descriptor->Get();
		if (!td) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		return static_cast<size_t>(td->m_rowgroup_descriptors()->size());
	}

	size_t rowgroup_storage_bytes(const size_t rowgroup_idx) const {
		const auto* td = m_table_descriptor->Get();
		if (!td) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		const auto n_rgs = td->m_rowgroup_descriptors()->size();
		if (rowgroup_idx >= n_rgs) {
			throw std::out_of_range("rowgroup_idx out of range");
		}
		const auto* rg = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
		return static_cast<size_t>(rg->m_size());
	}

	ZeroCopyRowgroup read_rowgroup_zero_copy_into(const size_t          rowgroup_idx,
	                                              std::shared_ptr<void> backing_owner,
	                                              std::byte* const      backing_data,
	                                              const size_t          backing_capacity,
	                                              const bool            backing_is_pinned = false) {
		const auto* td = m_table_descriptor->Get();
		if (!td) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		const auto n_rgs = td->m_rowgroup_descriptors()->size();
		if (rowgroup_idx >= n_rgs) {
			throw std::out_of_range("rowgroup_idx out of range");
		}

		const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
		const size_t rg_bytes = static_cast<size_t>(rg->m_size());
		if (backing_data == nullptr || backing_capacity < rg_bytes) {
			throw std::runtime_error("external rowgroup backing is null or too small");
		}

		const size_t n_vecs   = static_cast<size_t>(rg->m_n_vec());
		const size_t n_values = n_vecs * consts::VALUES_PER_VECTOR;
		const size_t n_tuples = static_cast<size_t>(rg->m_n_tuples());

		m_file->ReadRangeUnchecked(backing_data, rg->m_offset(), rg->m_size());
		auto        backing_span    = fastlanes::span<std::byte> {backing_data, rg_bytes};
		const auto& col_descs       = *rg->m_column_descriptors();
		const bool  use_schema_plan = m_zero_copy_schema_plan && m_zero_copy_schema_plan->enabled &&
		                             m_zero_copy_schema_plan->columns.size() == col_descs.size();
		std::shared_ptr<fastlanes::RowgroupView> view;
		if (!use_schema_plan) {
			view = std::make_shared<fastlanes::RowgroupView>(backing_span, *rg);
		}

		ZeroCopyRowgroup out {};
		out.rowgroup_index         = rowgroup_idx;
		out.n_values               = n_values;
		out.n_vecs                 = n_vecs;
		out.n_tuples               = n_tuples;
		out.table_descriptor_owner = m_table_descriptor;
		out.rowgroup_descriptor    = rg;
		out.backing_owner          = std::move(backing_owner);
		out.backing_span           = backing_span;
		out.backing_is_pinned      = backing_is_pinned;
		out.rowgroup_view          = view;

		if (use_schema_plan) {
			out.schema_plan_owner = m_zero_copy_schema_plan;
			out.schema_plan       = out.schema_plan_owner.get();
			return out;
		}

		out.columns.reserve(col_descs.size());

		for (size_t col_idx = 0; col_idx < col_descs.size(); ++col_idx) {
			const auto& col_desc = *col_descs.Get(static_cast<flatbuffers::uoffset_t>(col_idx));
			const auto* rpn      = col_desc.encoding_rpn();
			if (!rpn || !rpn->operator_tokens()) {
				throw std::runtime_error("missing encoding_rpn/operator_tokens");
			}
			const auto* ops = rpn->operator_tokens();
			if (ops->size() != 1) {
				std::ostringstream msg;
				msg << "only single-op expressions are supported in zero-copy reader; got ops=[";
				for (size_t i = 0; i < ops->size(); ++i) {
					if (i > 0) {
						msg << ", ";
					}
					msg << fastlanes::token_to_string(ops->Get(static_cast<flatbuffers::uoffset_t>(i)));
				}
				msg << "]";
				throw std::runtime_error(msg.str());
			}

			ZeroCopyColumn col {};
			col.column_index      = col_idx;
			col.name              = (m_load_column_names && col_desc.name()) ? col_desc.name()->str() : std::string {};
			col.token             = ops->Get(0);
			col.column_descriptor = &col_desc;
			col.operand_tokens    = rpn->operand_tokens();
			col.column_view       = &(*view)[static_cast<fastlanes::n_t>(col_idx)];
			col.column_span       = backing_span;
			if (col.token == fastlanes::OperatorToken::EXP_EQUAL && col.operand_tokens &&
			    col.operand_tokens->size() >= 1) {
				col.skip_decompress = true;
				col.alias_of        = static_cast<size_t>(col.operand_tokens->Get(0));
			}
			out.columns.push_back(col);
		}
		return out;
	}

	ZeroCopyRowgroup read_rowgroup_zero_copy(const size_t rowgroup_idx = 0) {
		auto backing = std::make_shared<fastlanes::Buf>(rowgroup_storage_bytes(rowgroup_idx));
		return read_rowgroup_zero_copy_into(rowgroup_idx,
		                                    std::static_pointer_cast<void>(backing),
		                                    reinterpret_cast<std::byte*>(backing->mutable_data()),
		                                    backing->Capacity());
	}

	Rowgroup materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy) const {
		auto storage           = std::make_shared<detail::ZeroCopyHostStorage>();
		storage->backing_owner = zero_copy.backing_owner;
		storage->rowgroup_view = zero_copy.rowgroup_view;

		const auto* schema_plan     = zero_copy.schema_plan;
		const bool  use_schema_plan = schema_plan != nullptr && schema_plan->enabled && schema_plan->columns.size() > 0;
		const size_t column_count   = use_schema_plan ? schema_plan->columns.size() : zero_copy.columns.size();

		Rowgroup out {zero_copy.n_values, zero_copy.n_vecs, zero_copy.n_tuples, {}};
		out.columns.resize(column_count);
		detail::SmallBuildState build_state(column_count);

		const auto get_zero_copy_column = [&](const size_t col_idx) -> ZeroCopyColumn {
			if (use_schema_plan) {
				return make_zero_copy_column_from_plan(zero_copy, schema_plan->columns[col_idx]);
			}
			return zero_copy.columns[col_idx];
		};

		auto build_column = [&](auto&& self, size_t col_idx) -> Column& {
			if (col_idx >= out.columns.size()) {
				throw std::out_of_range("column index out of range");
			}
			if (build_state[col_idx] == 2) {
				return out.columns[col_idx];
			}
			if (build_state[col_idx] == 1) {
				throw std::runtime_error("cycle detected in column dependencies");
			}
			build_state[col_idx] = 1;

			const auto zcol = get_zero_copy_column(col_idx);
			Column     result {};
			result.name  = zero_copy_column_name(zcol);
			result.token = zcol.token;

			if (zcol.skip_decompress) {
				if (!zcol.alias_of.has_value()) {
					throw std::runtime_error("zero-copy alias column missing source");
				}
				const size_t src_col_idx     = *zcol.alias_of;
				auto&        src_col         = self(self, src_col_idx);
				result.host                  = src_col.host;
				result.skip_decompress       = true;
				result.alias_of              = src_col_idx;
				result.host_owned_by_backing = src_col.host_owned_by_backing;
				result.backing_base          = src_col.backing_base;
				result.backing_bytes         = src_col.backing_bytes;
				result.backing_is_pinned     = src_col.backing_is_pinned;
			} else {
				if (zcol.column_descriptor == nullptr) {
					std::ostringstream msg;
					msg << "zero-copy column metadata missing (col_index=" << zcol.column_index
					    << ", name=" << zero_copy_column_name(zcol)
					    << ", token=" << fastlanes::token_to_string(zcol.token) << ")";
					throw std::runtime_error(msg.str());
				}
				switch (zcol.token) {
					using enum fastlanes::OperatorToken;
				case EXP_UNCOMPRESSED_I08: {
					if (zero_copy_operand_count(zcol) < 1) {
						throw std::runtime_error("EXP_UNCOMPRESSED_I08: missing operand tokens");
					}
					const auto seg = zero_copy_segment(
					    zcol, static_cast<uint32_t>(zero_copy_operand(zcol, zero_copy_operand_count(zcol) - 1)));
					result.host = detail::make_uncompressed_zero_copy<int8_t>(seg, zero_copy.n_values, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_CONSTANT_I08: {
					if (zcol.column_descriptor == nullptr || !zcol.column_descriptor->max()) {
						throw std::runtime_error("EXP_CONSTANT_I08: missing max value");
					}
					const auto* bin = zcol.column_descriptor->max()->binary_data();
					if (bin == nullptr || bin->size() != sizeof(int8_t)) {
						throw std::runtime_error("EXP_CONSTANT_I08: invalid constant size");
					}
					const auto value             = *reinterpret_cast<const int8_t*>(bin->data());
					result.host                  = flsgpu::host::CONSTANTColumn<int8_t> {zero_copy.n_values, value};
					result.host_owned_by_backing = false;
					break;
				}
				case EXP_FREQUENCY_I08: {
					if (zero_copy_operand_count(zcol) < 4) {
						throw std::runtime_error("EXP_FREQUENCY_I08: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_fv =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
					const auto seg_exc =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_pos =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_cnt =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host = detail::make_frequency_zero_copy<int8_t>(
					    seg_fv, seg_exc, seg_pos, seg_cnt, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FREQUENCY_I16: {
					if (zero_copy_operand_count(zcol) < 4) {
						throw std::runtime_error("EXP_FREQUENCY_I16: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_fv =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
					const auto seg_exc =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_pos =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_cnt =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host = detail::make_frequency_zero_copy<int16_t>(
					    seg_fv, seg_exc, seg_pos, seg_cnt, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FFOR_I08: {
					if (zero_copy_operand_count(zcol) < 3) {
						throw std::runtime_error("EXP_FFOR_I08: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_bw =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_base =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host = detail::make_ffor_zero_copy<int8_t>(
					    seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FFOR_I16: {
					if (zero_copy_operand_count(zcol) < 3) {
						throw std::runtime_error("EXP_FFOR_I16: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_bw =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_base =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host = detail::make_ffor_zero_copy<int16_t>(
					    seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FFOR_SLPATCH_I08: {
					if (zero_copy_operand_count(zcol) < 6) {
						throw std::runtime_error("EXP_FFOR_SLPATCH_I08: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_exc =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 5)));
					const auto seg_pos =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
					const auto seg_cnt =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_bw =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_base =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host                  = detail::make_slpatch_zero_copy<int8_t>(seg_exc,
                                                                         seg_pos,
                                                                         seg_cnt,
                                                                         seg_bitpacked,
                                                                         seg_bw,
                                                                         seg_base,
                                                                         zero_copy.n_values,
                                                                         zero_copy.n_vecs,
                                                                         *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FFOR_SLPATCH_I16: {
					if (zero_copy_operand_count(zcol) < 6) {
						throw std::runtime_error("EXP_FFOR_SLPATCH_I16: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_exc =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 5)));
					const auto seg_pos =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
					const auto seg_cnt =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_bw =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_base =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host                  = detail::make_slpatch_zero_copy<int16_t>(seg_exc,
                                                                          seg_pos,
                                                                          seg_cnt,
                                                                          seg_bitpacked,
                                                                          seg_bw,
                                                                          seg_base,
                                                                          zero_copy.n_values,
                                                                          zero_copy.n_vecs,
                                                                          *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I08_FFOR_SLPATCH_U08: {
					if (zero_copy_operand_count(zcol) < 7) {
						throw std::runtime_error("EXP_DICT_I08_FFOR_SLPATCH_U08: missing operand tokens");
					}
					const auto seg_keys = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
					const auto seg_exc  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
					const auto seg_pos  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
					const auto seg_cnt  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 4)));
					const auto seg_bw   = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 5)));
					const auto seg_base = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 6)));
					auto       index_slpatch = detail::make_slpatch_zero_copy<uint8_t>(seg_exc,
                                                                                 seg_pos,
                                                                                 seg_cnt,
                                                                                 seg_bitpacked,
                                                                                 seg_bw,
                                                                                 seg_base,
                                                                                 zero_copy.n_values,
                                                                                 zero_copy.n_vecs,
                                                                                 *storage);
					using KeyT               = typename utils::same_width_uint<int8_t>::type;
					const size_t key_count   = seg_keys.data_span.size() / sizeof(KeyT);
					auto*        keys        = detail::segment_ptr_or_copy<KeyT>(seg_keys, *storage);
					result.host = flsgpu::host::DICTSLPATCHColumn<int8_t, uint8_t> {index_slpatch, keys, key_count};
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I08_FFOR_U08: {
					if (zero_copy_operand_count(zcol) < 4) {
						throw std::runtime_error("EXP_DICT_I08_FFOR_U08: missing operand tokens");
					}
					const auto seg_keys = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
					const auto seg_bw   = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
					const auto seg_base = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
					result.host         = detail::make_dict_ffor_zero_copy<int8_t, uint8_t>(
                        seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_U16: {
					if (zero_copy_operand_count(zcol) < 4) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_U16: missing operand tokens");
					}
					const auto seg_keys = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
					const auto seg_bw   = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
					const auto seg_base = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
					result.host         = detail::make_dict_ffor_zero_copy<int16_t, uint16_t>(
                        seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_U08: {
					if (zero_copy_operand_count(zcol) < 4) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_U08: missing operand tokens");
					}
					const auto seg_keys = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
					const auto seg_bw   = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
					const auto seg_base = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
					result.host         = detail::make_dict_ffor_zero_copy<int16_t, uint8_t>(
                        seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I08_U08: {
					if (zero_copy_operand_count(zcol) < 2) {
						throw std::runtime_error("EXP_DICT_I08_U08: missing operand tokens");
					}
					const auto index_col_idx = static_cast<uint32_t>(zero_copy_operand(zcol, 0));
					const auto seg_keys      = zero_copy_segment(
                        zcol, static_cast<uint32_t>(zero_copy_operand(zcol, zero_copy_operand_count(zcol) - 1)));
					result.host = detail::make_dict_ref_zero_copy<int8_t, uint8_t>(
					    seg_keys, index_col_idx, zero_copy.n_values, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_SLPATCH_U16: {
					if (zero_copy_operand_count(zcol) < 7) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_SLPATCH_U16: missing operand tokens");
					}
					const auto seg_keys = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
					const auto seg_exc  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
					const auto seg_pos  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
					const auto seg_cnt  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 4)));
					const auto seg_bw   = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 5)));
					const auto seg_base = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 6)));
					auto       index_slpatch = detail::make_slpatch_zero_copy<uint16_t>(seg_exc,
                                                                                  seg_pos,
                                                                                  seg_cnt,
                                                                                  seg_bitpacked,
                                                                                  seg_bw,
                                                                                  seg_base,
                                                                                  zero_copy.n_values,
                                                                                  zero_copy.n_vecs,
                                                                                  *storage);
					using KeyT               = typename utils::same_width_uint<int16_t>::type;
					const size_t key_count   = seg_keys.data_span.size() / sizeof(KeyT);
					auto*        keys        = detail::segment_ptr_or_copy<KeyT>(seg_keys, *storage);
					result.host = flsgpu::host::DICTSLPATCHColumn<int16_t, uint16_t> {index_slpatch, keys, key_count};
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_SLPATCH_U08: {
					if (zero_copy_operand_count(zcol) < 7) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_SLPATCH_U08: missing operand tokens");
					}
					const auto seg_keys = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
					const auto seg_exc  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
					const auto seg_pos  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
					const auto seg_cnt  = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 4)));
					const auto seg_bw   = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 5)));
					const auto seg_base = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 6)));
					auto       index_slpatch = detail::make_slpatch_zero_copy<uint8_t>(seg_exc,
                                                                                 seg_pos,
                                                                                 seg_cnt,
                                                                                 seg_bitpacked,
                                                                                 seg_bw,
                                                                                 seg_base,
                                                                                 zero_copy.n_values,
                                                                                 zero_copy.n_vecs,
                                                                                 *storage);
					using KeyT               = typename utils::same_width_uint<int16_t>::type;
					const size_t key_count   = seg_keys.data_span.size() / sizeof(KeyT);
					auto*        keys        = detail::segment_ptr_or_copy<KeyT>(seg_keys, *storage);
					result.host = flsgpu::host::DICTSLPATCHColumn<int16_t, uint8_t> {index_slpatch, keys, key_count};
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_CROSS_RLE_I08: {
					if (zero_copy_operand_count(zcol) < 2) {
						throw std::runtime_error("EXP_CROSS_RLE_I08: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_vals =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_lens =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host = detail::make_cross_rle_zero_copy<int8_t>(
					    seg_vals, seg_lens, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_RLE_I08_U16: {
					if (zero_copy_operand_count(zcol) < 5) {
						throw std::runtime_error("EXP_RLE_I08_U16: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_vals =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
					const auto seg_rsum =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_bw =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_base =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host                  = detail::make_rle_zero_copy<int8_t, uint16_t>(seg_vals,
                                                                               seg_rsum,
                                                                               seg_bitpacked,
                                                                               seg_bw,
                                                                               seg_base,
                                                                               zero_copy.n_values,
                                                                               zero_copy.n_vecs,
                                                                               *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_RLE_I16_U16: {
					if (zero_copy_operand_count(zcol) < 5) {
						throw std::runtime_error("EXP_RLE_I16_U16: missing operand tokens");
					}
					const size_t base_idx = zero_copy_operand_count(zcol) - 1;
					const auto   seg_vals =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
					const auto seg_rsum =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
					const auto seg_bitpacked =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
					const auto seg_bw =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
					const auto seg_base =
					    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
					result.host                  = detail::make_rle_zero_copy<int16_t, uint16_t>(seg_vals,
                                                                                seg_rsum,
                                                                                seg_bitpacked,
                                                                                seg_bw,
                                                                                seg_base,
                                                                                zero_copy.n_values,
                                                                                zero_copy.n_vecs,
                                                                                *storage);
					result.host_owned_by_backing = true;
					break;
				}
				default:
					std::ostringstream msg;
					msg << "unsupported operator token for zero-copy materialization: "
					    << fastlanes::token_to_string(zcol.token) << " (col_index=" << zcol.column_index
					    << ", name=" << zero_copy_column_name(zcol) << ")";
					throw std::runtime_error(msg.str());
				}

				if (result.host_owned_by_backing && zero_copy.backing_is_pinned) {
					// Only publish a backing slice when the buffer is CUDA-pinned —
					// DeviceArena's direct cudaMemcpyAsync path requires that.
					// For pageable backings (the default read_rowgroup_zero_copy
					// path, which allocates via fastlanes::Buf / new[]), leave
					// backing_base null so append_expressions() falls back to the
					// staged-pinned copy instead of losing async H2D overlap.
					//
					// Record only this column's slice — using the full rowgroup
					// span would force DeviceArena::upload() to reserve device
					// memory and DMA every column's bytes even under narrow
					// projections.
					if (zcol.column_view != nullptr && !zcol.column_view->column_span.empty()) {
						result.backing_base  = zcol.column_view->column_span.data();
						result.backing_bytes = zcol.column_view->column_span.size();
					} else {
						result.backing_base  = zero_copy.backing_span.data();
						result.backing_bytes = zero_copy.backing_span.size();
					}
					result.backing_is_pinned = true;
				}
			}

			out.columns[col_idx] = std::move(result);
			build_state[col_idx] = 2;
			return out.columns[col_idx];
		};

		if (use_schema_plan) {
			for (const size_t i : schema_plan->build_order) {
				(void)build_column(build_column, i);
			}
		} else {
			for (size_t i = 0; i < zero_copy.columns.size(); ++i) {
				(void)build_column(build_column, i);
			}
		}
		out.backing_storage = std::move(storage);
		return out;
	}

	Rowgroup read_rowgroup_zero_copy_materialized(const size_t rowgroup_idx = 0) {
		return materialize_zero_copy_rowgroup(read_rowgroup_zero_copy(rowgroup_idx));
	}

	Rowgroup read_rowgroup(const size_t rowgroup_idx = 0) {
		return make_owning_rowgroup(read_rowgroup_zero_copy_materialized(rowgroup_idx));
	}

	std::vector<Rowgroup> read_table() {
		const size_t          n_rgs = rowgroup_count();
		std::vector<Rowgroup> out;
		out.reserve(n_rgs);
		for (size_t rg_idx = 0; rg_idx < n_rgs; ++rg_idx) {
			out.emplace_back(read_rowgroup(rg_idx));
		}
		return out;
	}

private:
	static void release_transient_materialized_rowgroup(Rowgroup& rowgroup) {
		for (auto& col : rowgroup.columns) {
			if (col.alias_of.has_value()) {
				continue;
			}
			if (rowgroup.backing_storage && col.host_owned_by_backing) {
				continue;
			}
			std::visit([](auto& host_col) { flsgpu::host::free_column(host_col); }, col.host);
		}
		rowgroup.columns.clear();
		rowgroup.backing_storage.reset();
	}

	static Rowgroup make_owning_rowgroup(Rowgroup rowgroup) {
		Rowgroup out {rowgroup.n_values, rowgroup.n_vecs, rowgroup.n_tuples, {}};
		out.columns.resize(rowgroup.columns.size());

		detail::SmallBuildState build_state(rowgroup.columns.size());
		auto clone_column = [&](auto&& self, const size_t col_idx) -> Column& {
			if (col_idx >= rowgroup.columns.size()) {
				throw std::out_of_range("column index out of range");
			}
			if (build_state[col_idx] == 2) {
				return out.columns[col_idx];
			}
			if (build_state[col_idx] == 1) {
				throw std::runtime_error("cycle detected in column dependencies");
			}
			build_state[col_idx] = 1;

			const auto& src = rowgroup.columns[col_idx];
			auto&       dst = out.columns[col_idx];
			dst.name        = src.name;
			dst.token       = src.token;

			if (src.alias_of.has_value()) {
				const size_t src_col_idx = *src.alias_of;
				auto&        src_col     = self(self, src_col_idx);
				dst.host                 = src_col.host;
				dst.skip_decompress      = true;
				dst.alias_of             = src_col_idx;
			} else {
				dst.host = std::visit(
				    [](const auto& host_col) -> HostColumnVariant { return detail::clone_column(host_col); },
				    src.host);
			}

			dst.host_owned_by_backing = false;
			dst.backing_base          = nullptr;
			dst.backing_bytes         = 0;
			dst.backing_is_pinned     = false;
			build_state[col_idx]      = 2;
			return dst;
		};

		for (size_t i = 0; i < rowgroup.columns.size(); ++i) {
			(void)clone_column(clone_column, i);
		}

		release_transient_materialized_rowgroup(rowgroup);
		return out;
	}

	static std::vector<ZeroCopyColumnPlan> build_zero_copy_column_plan(const fastlanes::RowgroupDescriptor& rg,
	                                                                   const bool load_column_names) {
		const auto* col_descs = rg.m_column_descriptors();
		if (col_descs == nullptr) {
			throw std::runtime_error("missing rowgroup column descriptors");
		}

		std::vector<ZeroCopyColumnPlan> plan;
		plan.reserve(col_descs->size());
		for (size_t col_idx = 0; col_idx < col_descs->size(); ++col_idx) {
			const auto& col_desc = *col_descs->Get(static_cast<flatbuffers::uoffset_t>(col_idx));
			const auto* rpn      = col_desc.encoding_rpn();
			if (!rpn || !rpn->operator_tokens()) {
				throw std::runtime_error("missing encoding_rpn/operator_tokens");
			}
			const auto* ops = rpn->operator_tokens();
			if (ops->size() != 1) {
				std::ostringstream msg;
				msg << "only single-op expressions are supported in zero-copy reader plan; got ops=[";
				for (size_t i = 0; i < ops->size(); ++i) {
					if (i > 0) {
						msg << ", ";
					}
					msg << fastlanes::token_to_string(ops->Get(static_cast<flatbuffers::uoffset_t>(i)));
				}
				msg << "]";
				throw std::runtime_error(msg.str());
			}

			ZeroCopyColumnPlan col {};
			col.column_index = col_idx;
			col.name         = (load_column_names && col_desc.name()) ? col_desc.name()->str() : std::string {};
			col.token        = ops->Get(0);
			if (const auto* operands = rpn->operand_tokens()) {
				col.operand_ids.reserve(operands->size());
				for (size_t i = 0; i < operands->size(); ++i) {
					col.operand_ids.push_back(
					    static_cast<uint32_t>(operands->Get(static_cast<flatbuffers::uoffset_t>(i))));
				}
			}
			if (col.token == fastlanes::OperatorToken::EXP_EQUAL) {
				if (col.operand_ids.empty()) {
					throw std::runtime_error("EXP_EQUAL: missing operand tokens");
				}
				col.skip_decompress = true;
				col.alias_of        = static_cast<size_t>(col.operand_ids[0]);
			}
			plan.push_back(std::move(col));
		}
		return plan;
	}

	static bool rowgroup_matches_zero_copy_plan(const fastlanes::RowgroupDescriptor&   rg,
	                                            const std::vector<ZeroCopyColumnPlan>& plan) {
		const auto* col_descs = rg.m_column_descriptors();
		if (col_descs == nullptr || col_descs->size() != plan.size()) {
			return false;
		}
		for (size_t col_idx = 0; col_idx < col_descs->size(); ++col_idx) {
			const auto& col_desc = *col_descs->Get(static_cast<flatbuffers::uoffset_t>(col_idx));
			const auto* rpn      = col_desc.encoding_rpn();
			if (!rpn || !rpn->operator_tokens()) {
				return false;
			}
			const auto* ops = rpn->operator_tokens();
			if (ops->size() != 1 || ops->Get(0) != plan[col_idx].token) {
				return false;
			}
			const auto*  operands     = rpn->operand_tokens();
			const size_t operand_size = operands != nullptr ? operands->size() : 0;
			if (operand_size != plan[col_idx].operand_ids.size()) {
				return false;
			}
			for (size_t i = 0; i < operand_size; ++i) {
				if (static_cast<uint32_t>(operands->Get(static_cast<flatbuffers::uoffset_t>(i))) !=
				    plan[col_idx].operand_ids[i]) {
					return false;
				}
			}
		}
		return true;
	}

	static std::vector<size_t> build_zero_copy_column_order(const std::vector<ZeroCopyColumnPlan>& columns) {
		std::vector<size_t> order;
		order.reserve(columns.size());
		std::vector<uint8_t> state(columns.size(), 0);

		auto visit = [&](auto&& self, const size_t idx) -> void {
			if (idx >= columns.size()) {
				throw std::out_of_range("zero-copy column alias index out of range");
			}
			if (state[idx] == 2) {
				return;
			}
			if (state[idx] == 1) {
				throw std::runtime_error("cycle detected in zero-copy column aliases");
			}
			state[idx] = 1;
			if (columns[idx].alias_of.has_value()) {
				self(self, *columns[idx].alias_of);
			}
			state[idx] = 2;
			order.push_back(idx);
		};

		for (size_t i = 0; i < columns.size(); ++i) {
			visit(visit, i);
		}
		return order;
	}

	ZeroCopySchemaPlan build_shared_zero_copy_schema_plan() const {
		ZeroCopySchemaPlan plan {};
		const auto*        td = m_table_descriptor->Get();
		if (!td || !td->m_rowgroup_descriptors() || td->m_rowgroup_descriptors()->size() == 0) {
			return plan;
		}

		const auto* rowgroups      = td->m_rowgroup_descriptors();
		const auto* first_rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(0));
		if (first_rowgroup == nullptr) {
			return plan;
		}
		try {
			plan.columns     = build_zero_copy_column_plan(*first_rowgroup, m_load_column_names);
			plan.build_order = build_zero_copy_column_order(plan.columns);
		} catch (const std::exception&) {
			plan.columns.clear();
			plan.build_order.clear();
			return plan;
		}
		plan.enabled = true;
		for (size_t i = 1; i < rowgroups->size(); ++i) {
			const auto* rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(i));
			if (rowgroup == nullptr || !rowgroup_matches_zero_copy_plan(*rowgroup, plan.columns)) {
				plan.enabled = false;
				plan.columns.clear();
				plan.build_order.clear();
				break;
			}
		}
		return plan;
	}

	std::shared_ptr<fastlanes::File>                        m_file;
	std::shared_ptr<const fastlanes::TableDescriptorHandle> m_table_descriptor;
	bool                                                    m_load_column_names = true;
	std::shared_ptr<const ZeroCopySchemaPlan>               m_zero_copy_schema_plan;
};

} // namespace reader

#endif // FLS_READER_CUH
