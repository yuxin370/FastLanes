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
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include <unordered_set>
#include <variant>
#include <vector>

namespace reader {

namespace detail {
inline fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::FileHeader file_header {};
	fastlanes::FileFooter file_footer {};

	fastlanes::FileHeader::Load(file_header, file_path);
	fastlanes::FileFooter::Load(file_footer, file_path);

	if (file_header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file_path, file_footer.table_descriptor_offset, file_footer.table_descriptor_size, /*verify=*/true);
	}

	const auto footer_path = file_path.parent_path() / "table_descriptor.fbb";
	return fastlanes::TableDescriptorHandle::FromFile(footer_path, /*verify=*/true);
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

template <typename T>
inline flsgpu::host::BPColumn<T> make_bp_zero_copy(const fastlanes::SegmentView& seg_bitpacked,
                                                   const fastlanes::SegmentView& seg_bw,
                                                   const size_t                  n_values,
                                                   const size_t                  n_vecs,
                                                   ZeroCopyHostStorage&          storage) {
	using PackedT = typename utils::same_width_uint<T>::type;

	const auto entrypoints = columns::detail::extract_entrypoints(seg_bitpacked);
	if (entrypoints.size() != n_vecs) {
		throw std::runtime_error("bitpacked entrypoint count mismatch");
	}

	auto*  vector_offsets = storage.allocate_array<size_t>(n_vecs);
	size_t prev_bytes     = 0;
	for (size_t i = 0; i < n_vecs; ++i) {
		const size_t cur_bytes = static_cast<size_t>(entrypoints[i]);
		if (cur_bytes < prev_bytes || (cur_bytes % sizeof(PackedT)) != 0) {
			throw std::runtime_error("invalid bitpacked segment entrypoints");
		}
		vector_offsets[i] = prev_bytes / sizeof(PackedT);
		prev_bytes        = cur_bytes;
	}

	auto*        packed     = segment_ptr_or_copy<PackedT>(seg_bitpacked, storage);
	auto*        bit_widths = segment_ptr_or_copy<vbw_t>(seg_bw, storage);
	const size_t n_bp       = seg_bitpacked.data_span.size() / sizeof(PackedT);
	return flsgpu::host::BPColumn<T> {n_values, n_bp, packed, bit_widths, vector_offsets};
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
inline columns::detail::ExceptionOffsets build_entrypoint_offsets(const fastlanes::SegmentView& seg,
                                                                   const size_t                  n_vecs,
                                                                   const size_t                  elem_bytes,
                                                                   ZeroCopyHostStorage&          storage,
                                                                   const char*                   segment_tag = "exception segment") {
	const auto entry = columns::detail::extract_entrypoints(seg);
	if (entry.size() != n_vecs) {
		throw std::runtime_error(std::string(segment_tag) + " entrypoint count mismatch");
	}
	const size_t segment_bytes = seg.data_span.size();
	auto*        offsets       = storage.allocate_array<size_t>(n_vecs);
	size_t       prev_bytes    = 0;
	for (size_t i = 0; i < n_vecs; ++i) {
		const size_t cur_bytes = static_cast<size_t>(entry[i]);
		if (cur_bytes < prev_bytes || (cur_bytes % elem_bytes) != 0 || cur_bytes > segment_bytes) {
			throw std::runtime_error(std::string("invalid ") + segment_tag + " entrypoints");
		}
		offsets[i] = prev_bytes / elem_bytes;
		prev_bytes = cur_bytes;
	}
	return columns::detail::ExceptionOffsets {offsets, prev_bytes / elem_bytes};
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
	auto pos = build_entrypoint_offsets(seg_pos, n_vecs, sizeof(uint16_t), storage);

	auto* counts     = segment_ptr_or_copy<uint16_t>(seg_cnt, storage);
	auto* positions  = segment_ptr_or_copy<uint16_t>(seg_pos, storage);
	auto* exceptions = segment_ptr_or_copy<T>(seg_exc, storage);

	return flsgpu::host::SLPATCHColumn<T> {
	    n_values, n_vecs, ffor, exc.total, exc.offsets, pos.offsets, exceptions, positions, counts};
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
	auto*      fv_arr = storage.allocate_array<T>(n_vecs);
	for (size_t i = 0; i < n_vecs; ++i) {
		fv_arr[i] = fv;
	}

	auto exc = build_entrypoint_offsets(seg_exc, n_vecs, sizeof(T), storage);
	auto pos = build_entrypoint_offsets(seg_pos, n_vecs, sizeof(uint16_t), storage);

	auto* counts     = segment_ptr_or_copy<uint16_t>(seg_cnt, storage);
	auto* positions  = segment_ptr_or_copy<uint16_t>(seg_pos, storage);
	auto* exceptions = segment_ptr_or_copy<T>(seg_exc, storage);

	return flsgpu::host::FREQColumn<T> {
	    n_values, n_vecs, fv_arr, exc.total, exc.offsets, pos.offsets, exceptions, positions, counts};
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
		const size_t target_start = v * columns::detail::kVecSize;
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
	return flsgpu::host::DICTFFORColumn<T, IndexT> {
	    std::move(ffor), keys, seg_keys.data_span.size() / sizeof(KeyT)};
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

	const auto rle_offsets_info = build_entrypoint_offsets(seg_vals, n_vecs, sizeof(T), storage, "EXP_RLE values segment");
	auto* const offsets         = rle_offsets_info.offsets;

	auto* rsum_bases = segment_ptr_or_copy<IndexT>(seg_rsum, storage);
	auto* values     = segment_ptr_or_copy<T>(seg_vals, storage);
	const size_t n_rle_values = seg_vals.data_span.size() / sizeof(T);

	return flsgpu::host::RLEColumn<T, IndexT> {
	    n_values, n_vecs, std::move(ffor), rsum_bases, values, offsets, n_rle_values};
}

} // namespace detail

using HostColumnVariant = dispatch::EncodedPayload;
using Column            = dispatch::Column;
using Rowgroup          = dispatch::Rowgroup;

struct ZeroCopyColumn {
	size_t                               column_index = 0;
	std::string                          name;
	fastlanes::OperatorToken             token             = fastlanes::OperatorToken::INVALID;
	const fastlanes::ColumnDescriptor*   column_descriptor = nullptr;
	const flatbuffers::Vector<uint64_t>* operand_tokens    = nullptr;
	const fastlanes::ColumnView*         column_view       = nullptr;
	bool                                 skip_decompress   = false;
	std::optional<size_t>                alias_of;
};

struct ZeroCopyRowgroup {
	size_t                                   n_values = 0;
	size_t                                   n_vecs   = 0;
	size_t                                   n_tuples = 0;
	std::shared_ptr<void>                    backing_owner;
	fastlanes::span<std::byte>               backing_span;
	// True only if backing_span points into CUDA-pinned memory. Controls
	// whether DeviceArena may issue direct cudaMemcpyAsync from it; registering
	// a pageable buffer as a direct backing forces the async copy to fall back
	// to a synchronous staged transfer internally.
	bool                                     backing_is_pinned = false;
	std::shared_ptr<fastlanes::RowgroupView> rowgroup_view;
	std::vector<ZeroCopyColumn>              columns;
};

class reader {
public:
	explicit reader(const std::filesystem::path& file_path)
	    : m_file_path(file_path)
	    , m_table_descriptor(detail::load_table_descriptor(file_path)) {
	}

	size_t rowgroup_count() const {
		const auto* td = m_table_descriptor.Get();
		if (!td) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		return static_cast<size_t>(td->m_rowgroup_descriptors()->size());
	}

	size_t rowgroup_storage_bytes(const size_t rowgroup_idx) const {
		const auto* td = m_table_descriptor.Get();
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

	ZeroCopyRowgroup read_rowgroup_zero_copy_into(const size_t                rowgroup_idx,
	                                              std::shared_ptr<void>       backing_owner,
	                                              std::byte* const            backing_data,
	                                              const size_t                backing_capacity,
	                                              const bool                  backing_is_pinned = false) {
		const auto* td = m_table_descriptor.Get();
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

		fastlanes::io io = fastlanes::make_unique<fastlanes::File>(m_file_path);
		fastlanes::IO::range_read(io, backing_data, rg->m_offset(), rg->m_size());
		auto backing_span = fastlanes::span<std::byte> {backing_data, rg_bytes};
		auto view         = std::make_shared<fastlanes::RowgroupView>(backing_span, *rg);

		const auto&      col_descs = *rg->m_column_descriptors();
		ZeroCopyRowgroup out {};
		out.n_values      = n_values;
		out.n_vecs        = n_vecs;
		out.n_tuples      = n_tuples;
		out.backing_owner     = std::move(backing_owner);
		out.backing_span      = backing_span;
		out.backing_is_pinned = backing_is_pinned;
		out.rowgroup_view     = view;
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
			col.name              = col_desc.name() ? col_desc.name()->str() : std::string {};
			col.token             = ops->Get(0);
			col.column_descriptor = &col_desc;
			col.operand_tokens    = rpn->operand_tokens();
			col.column_view       = &(*view)[static_cast<fastlanes::n_t>(col_idx)];
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
		auto       storage      = std::make_shared<detail::ZeroCopyHostStorage>();
		storage->backing_owner  = zero_copy.backing_owner;
		storage->rowgroup_view  = zero_copy.rowgroup_view;

		std::vector<std::optional<Column>> built(zero_copy.columns.size());
		std::unordered_set<size_t>         in_progress;

		auto build_column = [&](auto&& self, size_t col_idx) -> Column& {
			if (col_idx >= built.size()) {
				throw std::out_of_range("column index out of range");
			}
			if (built[col_idx].has_value()) {
				return *built[col_idx];
			}
			if (in_progress.count(col_idx)) {
				throw std::runtime_error("cycle detected in column dependencies");
			}
			in_progress.insert(col_idx);

			const auto& zcol = zero_copy.columns[col_idx];
			Column      result {};
			result.name  = zcol.name;
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
				if (!zcol.column_view || !zcol.column_descriptor) {
					std::ostringstream msg;
					msg << "zero-copy column metadata missing (col_index=" << zcol.column_index
					    << ", name=" << zcol.name << ", token=" << fastlanes::token_to_string(zcol.token) << ")";
					throw std::runtime_error(msg.str());
				}
				columns::ParseContext ctx {*zcol.column_view,
				                           *zcol.column_descriptor,
				                           zcol.operand_tokens,
				                           zero_copy.n_values,
				                           zero_copy.n_vecs};

				const auto parse_with_copy_fallback = [&]() {
					switch (zcol.token) {
						using enum fastlanes::OperatorToken;
					case EXP_UNCOMPRESSED_I08: {
						auto parsed = columns::parse_uncompressed<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_CONSTANT_I08: {
						auto parsed = columns::parse_constant<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_FFOR_I08: {
						auto parsed = columns::parse_ffor<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_FFOR_I16: {
						auto parsed = columns::parse_ffor<int16_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_FFOR_SLPATCH_I08: {
						auto parsed = columns::parse_slpatch<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_FFOR_SLPATCH_I16: {
						auto parsed = columns::parse_slpatch<int16_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_FREQUENCY_I08: {
						auto parsed = columns::parse_frequency<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_FREQUENCY_I16: {
						auto parsed = columns::parse_frequency<int16_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_CROSS_RLE_I08: {
						auto parsed = columns::parse_cross_rle<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_DICT_I08_FFOR_SLPATCH_U08: {
						auto parsed = columns::parse_dict_slpatch<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_DICT_I08_FFOR_U08: {
						auto parsed = columns::parse_dict_ffor<int8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_DICT_I16_FFOR_U16: {
						auto parsed = columns::parse_dict_ffor<int16_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_DICT_I16_FFOR_U08: {
						auto parsed = columns::parse_dict_ffor<int16_t, uint8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_DICT_I08_U08: {
						auto parsed = columns::parse_dict_ref<int8_t, uint8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_DICT_I16_FFOR_SLPATCH_U16: {
						auto parsed = columns::parse_dict_slpatch<int16_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_DICT_I16_FFOR_SLPATCH_U08: {
						auto parsed = columns::parse_dict_slpatch<int16_t, uint8_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_RLE_I08_U16: {
						auto parsed = columns::parse_rle<int8_t, uint16_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					case EXP_RLE_I16_U16: {
						auto parsed = columns::parse_rle<int16_t, uint16_t>(ctx);
						result.host = std::move(parsed.host);
						break;
					}
					default: {
						std::ostringstream msg;
						msg << "unsupported operator token for zero-copy materialization fallback: "
						    << fastlanes::token_to_string(zcol.token) << " (col_index=" << zcol.column_index
						    << ", name=" << zcol.name << ")";
						throw std::runtime_error(msg.str());
					}
					}
					result.host_owned_by_backing = false;
				};

				switch (zcol.token) {
					using enum fastlanes::OperatorToken;
				case EXP_FREQUENCY_I08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 4) {
						throw std::runtime_error("EXP_FREQUENCY_I08: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto   seg_fv =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 3)));
					const auto seg_exc =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_pos =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_cnt =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
					result.host = detail::make_frequency_zero_copy<int8_t>(
					    seg_fv, seg_exc, seg_pos, seg_cnt, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FREQUENCY_I16: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 4) {
						throw std::runtime_error("EXP_FREQUENCY_I16: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto   seg_fv =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 3)));
					const auto seg_exc =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_pos =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_cnt =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
					result.host = detail::make_frequency_zero_copy<int16_t>(
					    seg_fv, seg_exc, seg_pos, seg_cnt, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FFOR_I08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 3) {
						throw std::runtime_error("EXP_FFOR_I08: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto   seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
					result.host = detail::make_ffor_zero_copy<int8_t>(
					    seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FFOR_I16: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 3) {
						throw std::runtime_error("EXP_FFOR_I16: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto   seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
					result.host = detail::make_ffor_zero_copy<int16_t>(
					    seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_FFOR_SLPATCH_I08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 6) {
						throw std::runtime_error("EXP_FFOR_SLPATCH_I08: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto   seg_exc =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 5)));
					const auto seg_pos =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 4)));
					const auto seg_cnt =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 3)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
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
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 6) {
						throw std::runtime_error("EXP_FFOR_SLPATCH_I16: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto   seg_exc =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 5)));
					const auto seg_pos =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 4)));
					const auto seg_cnt =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 3)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
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
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 7) {
						throw std::runtime_error("EXP_DICT_I08_FFOR_SLPATCH_U08: missing operand tokens");
					}
					const auto seg_keys =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(0)));
					const auto seg_exc =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(1)));
					const auto seg_pos =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(2)));
					const auto seg_cnt =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(3)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(4)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(5)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(6)));
					auto index_slpatch     = detail::make_slpatch_zero_copy<uint8_t>(seg_exc,
                                                                                 seg_pos,
                                                                                 seg_cnt,
                                                                                 seg_bitpacked,
                                                                                 seg_bw,
                                                                                 seg_base,
                                                                                 zero_copy.n_values,
                                                                                 zero_copy.n_vecs,
                                                                                 *storage);
					using KeyT             = typename utils::same_width_uint<int8_t>::type;
					const size_t key_count = seg_keys.data_span.size() / sizeof(KeyT);
					auto*        keys      = detail::segment_ptr_or_copy<KeyT>(seg_keys, *storage);
					result.host = flsgpu::host::DICTSLPATCHColumn<int8_t, uint8_t> {index_slpatch, keys, key_count};
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I08_FFOR_U08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 4) {
						throw std::runtime_error("EXP_DICT_I08_FFOR_U08: missing operand tokens");
					}
					const auto seg_keys =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(0)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(1)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(2)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(3)));
					result.host = detail::make_dict_ffor_zero_copy<int8_t, uint8_t>(
					    seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_U16: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 4) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_U16: missing operand tokens");
					}
					const auto seg_keys =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(0)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(1)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(2)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(3)));
					result.host = detail::make_dict_ffor_zero_copy<int16_t, uint16_t>(
					    seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_U08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 4) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_U08: missing operand tokens");
					}
					const auto seg_keys =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(0)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(1)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(2)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(3)));
					result.host = detail::make_dict_ffor_zero_copy<int16_t, uint8_t>(
					    seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I08_U08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 2) {
						throw std::runtime_error("EXP_DICT_I08_U08: missing operand tokens");
					}
					const auto index_col_idx = static_cast<uint32_t>(zcol.operand_tokens->Get(0));
					const auto seg_keys =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(
					        zcol.operand_tokens->Get(zcol.operand_tokens->size() - 1)));
					result.host =
					    detail::make_dict_ref_zero_copy<int8_t, uint8_t>(seg_keys, index_col_idx, zero_copy.n_values, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_SLPATCH_U16: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 7) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_SLPATCH_U16: missing operand tokens");
					}
					const auto seg_keys =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(0)));
					const auto seg_exc =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(1)));
					const auto seg_pos =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(2)));
					const auto seg_cnt =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(3)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(4)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(5)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(6)));
					auto index_slpatch     = detail::make_slpatch_zero_copy<uint16_t>(seg_exc,
                                                                                  seg_pos,
                                                                                  seg_cnt,
                                                                                  seg_bitpacked,
                                                                                  seg_bw,
                                                                                  seg_base,
                                                                                  zero_copy.n_values,
                                                                                  zero_copy.n_vecs,
                                                                                  *storage);
					using KeyT             = typename utils::same_width_uint<int16_t>::type;
					const size_t key_count = seg_keys.data_span.size() / sizeof(KeyT);
					auto*        keys      = detail::segment_ptr_or_copy<KeyT>(seg_keys, *storage);
					result.host = flsgpu::host::DICTSLPATCHColumn<int16_t, uint16_t> {index_slpatch, keys, key_count};
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_DICT_I16_FFOR_SLPATCH_U08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 7) {
						throw std::runtime_error("EXP_DICT_I16_FFOR_SLPATCH_U08: missing operand tokens");
					}
					const auto seg_keys =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(0)));
					const auto seg_exc =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(1)));
					const auto seg_pos =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(2)));
					const auto seg_cnt =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(3)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(4)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(5)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(6)));
					auto index_slpatch     = detail::make_slpatch_zero_copy<uint8_t>(seg_exc,
                                                                                 seg_pos,
                                                                                 seg_cnt,
                                                                                 seg_bitpacked,
                                                                                 seg_bw,
                                                                                 seg_base,
                                                                                 zero_copy.n_values,
                                                                                 zero_copy.n_vecs,
                                                                                 *storage);
					using KeyT             = typename utils::same_width_uint<int16_t>::type;
					const size_t key_count = seg_keys.data_span.size() / sizeof(KeyT);
					auto*        keys      = detail::segment_ptr_or_copy<KeyT>(seg_keys, *storage);
					result.host = flsgpu::host::DICTSLPATCHColumn<int16_t, uint8_t> {index_slpatch, keys, key_count};
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_CROSS_RLE_I08: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 2) {
						throw std::runtime_error("EXP_CROSS_RLE_I08: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto   seg_vals =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_lens =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
					result.host = detail::make_cross_rle_zero_copy<int8_t>(
					    seg_vals, seg_lens, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_RLE_I08_U16: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 5) {
						throw std::runtime_error("EXP_RLE_I08_U16: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto seg_vals =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 4)));
					const auto seg_rsum =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 3)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
					result.host = detail::make_rle_zero_copy<int8_t, uint16_t>(
					    seg_vals, seg_rsum, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				case EXP_RLE_I16_U16: {
					if (!zcol.operand_tokens || zcol.operand_tokens->size() < 5) {
						throw std::runtime_error("EXP_RLE_I16_U16: missing operand tokens");
					}
					const size_t base_idx = zcol.operand_tokens->size() - 1;
					const auto seg_vals =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 4)));
					const auto seg_rsum =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 3)));
					const auto seg_bitpacked =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 2)));
					const auto seg_bw =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 1)));
					const auto seg_base =
					    zcol.column_view->GetSegment(static_cast<uint32_t>(zcol.operand_tokens->Get(base_idx - 0)));
					result.host = detail::make_rle_zero_copy<int16_t, uint16_t>(
					    seg_vals, seg_rsum, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
					result.host_owned_by_backing = true;
					break;
				}
				default:
					parse_with_copy_fallback();
					break;
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

			built[col_idx] = std::move(result);
			in_progress.erase(col_idx);
			return *built[col_idx];
		};

		Rowgroup out {zero_copy.n_values, zero_copy.n_vecs, zero_copy.n_tuples, {}};
		out.columns.reserve(zero_copy.columns.size());
		for (size_t i = 0; i < zero_copy.columns.size(); ++i) {
			out.columns.push_back(build_column(build_column, i));
		}
		out.backing_storage = std::move(storage);
		return out;
	}

	Rowgroup read_rowgroup_zero_copy_materialized(const size_t rowgroup_idx = 0) {
		return materialize_zero_copy_rowgroup(read_rowgroup_zero_copy(rowgroup_idx));
	}

	Rowgroup read_rowgroup(const size_t rowgroup_idx = 0) {
		const auto* td = m_table_descriptor.Get();
		if (!td) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		const auto n_rgs = td->m_rowgroup_descriptors()->size();
		if (rowgroup_idx >= n_rgs) {
			throw std::out_of_range("rowgroup_idx out of range");
		}

		const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
		const size_t n_vecs   = static_cast<size_t>(rg->m_n_vec());
		const size_t n_values = n_vecs * consts::VALUES_PER_VECTOR;
		const size_t n_tuples = static_cast<size_t>(rg->m_n_tuples());

		// Read rowgroup bytes
		fastlanes::Buf buf(rg->m_size());
		fastlanes::io  io = fastlanes::make_unique<fastlanes::File>(m_file_path);
		fastlanes::IO::range_read(io, buf, rg->m_offset(), rg->m_size());
		fastlanes::RowgroupView rg_view(buf.Span(), *rg);

		// Build columns (with dependency resolution for DICT_I08_U08)
		const auto&                        col_descs = *rg->m_column_descriptors();
		std::vector<std::optional<Column>> built(col_descs.size());
		std::unordered_set<size_t>         in_progress;

		auto build_column = [&](auto&& self, size_t col_idx) -> Column& {
			if (col_idx >= built.size()) {
				throw std::out_of_range("column index out of range");
			}
			if (built[col_idx].has_value()) {
				return *built[col_idx];
			}
			if (in_progress.count(col_idx)) {
				throw std::runtime_error("cycle detected in column dependencies");
			}

			in_progress.insert(col_idx);

			const auto& col_desc = *col_descs.Get(static_cast<flatbuffers::uoffset_t>(col_idx));
			const auto* rpn      = col_desc.encoding_rpn();
			if (!rpn || !rpn->operator_tokens()) {
				throw std::runtime_error("missing encoding_rpn/operator_tokens");
			}
			const auto* ops = rpn->operator_tokens();
			if (ops->size() != 1) {
				std::ostringstream msg;
				msg << "only single-op expressions are supported in this reader; got ops=[";
				for (size_t i = 0; i < ops->size(); ++i) {
					if (i > 0) {
						msg << ", ";
					}
					msg << fastlanes::token_to_string(ops->Get(static_cast<flatbuffers::uoffset_t>(i)));
				}
				msg << "]";
				throw std::runtime_error(msg.str());
			}

			const auto op_token = ops->Get(0);
			Column     result;
			result.name          = col_desc.name() ? col_desc.name()->str() : std::string {};
			result.token         = op_token;
			auto& column_view    = rg_view[static_cast<fastlanes::n_t>(col_idx)];
			auto* operand_tokens = rpn->operand_tokens();

			columns::ParseContext ctx {column_view, col_desc, operand_tokens, n_values, n_vecs};

			switch (op_token) {
				using enum fastlanes::OperatorToken;
			case EXP_UNCOMPRESSED_I08: {
				auto parsed = columns::parse_uncompressed<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_CONSTANT_I08: {
				auto parsed = columns::parse_constant<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_I08: {
				auto parsed = columns::parse_ffor<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_I16: {
				auto parsed = columns::parse_ffor<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_SLPATCH_I08: {
				auto parsed = columns::parse_slpatch<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_SLPATCH_I16: {
				auto parsed = columns::parse_slpatch<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FREQUENCY_I08: {
				auto parsed = columns::parse_frequency<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FREQUENCY_I16: {
				auto parsed = columns::parse_frequency<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_CROSS_RLE_I08: {
				auto parsed = columns::parse_cross_rle<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I08_FFOR_SLPATCH_U08: {
				auto parsed = columns::parse_dict_slpatch<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I08_FFOR_U08: {
				auto parsed = columns::parse_dict_ffor<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_U16: {
				auto parsed = columns::parse_dict_ffor<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_U08: {
				auto parsed = columns::parse_dict_ffor<int16_t, uint8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I08_U08: {
				auto parsed = columns::parse_dict_ref<int8_t, uint8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_SLPATCH_U16: {
				auto parsed = columns::parse_dict_slpatch<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_SLPATCH_U08: {
				auto parsed = columns::parse_dict_slpatch<int16_t, uint8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_RLE_I08_U16: {
				auto parsed = columns::parse_rle<int8_t, uint16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_RLE_I16_U16: {
				auto parsed = columns::parse_rle<int16_t, uint16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_EQUAL: {
				if (!operand_tokens || operand_tokens->size() < 1) {
					throw std::runtime_error("EXP_EQUAL: missing operand tokens");
				}
				const auto src_col_idx = static_cast<size_t>(operand_tokens->Get(0));
				auto&      src_col     = self(self, src_col_idx);
				result.host            = src_col.host;
				result.skip_decompress = true;
				result.alias_of        = src_col_idx;
				break;
			}
			default: {
				std::ostringstream msg;
				msg << "unsupported operator token for this reader: " << fastlanes::token_to_string(op_token)
				    << " (col_index=" << col_idx
				    << ", name=" << (col_desc.name() ? col_desc.name()->str() : std::string("<unnamed>")) << ")";
				throw std::runtime_error(msg.str());
			}
			}

			built[col_idx] = std::move(result);
			in_progress.erase(col_idx);
			return *built[col_idx];
		};

		Rowgroup out {n_values, n_vecs, n_tuples, {}};
		out.columns.reserve(col_descs.size());
		for (size_t i = 0; i < col_descs.size(); ++i) {
			out.columns.push_back(build_column(build_column, i));
		}

		return out;
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
	std::filesystem::path            m_file_path;
	fastlanes::TableDescriptorHandle m_table_descriptor;
};

} // namespace reader

#endif // FLS_READER_CUH
