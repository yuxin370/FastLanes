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
#include "flsgpu/flsgpu-api.cuh"
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
	// Destruction order matters: owned_arrays first (frees allocated copies),
	// then rowgroup_view (references backing_buffer), then backing_buffer last.
	std::shared_ptr<fastlanes::Buf>          backing_buffer;
	std::shared_ptr<fastlanes::RowgroupView> rowgroup_view;
	std::vector<std::function<void()>>       owned_arrays;

	~ZeroCopyHostStorage() {
		// Explicit teardown in safe order.
		owned_arrays.clear();
		rowgroup_view.reset();
		backing_buffer.reset();
	}

	template <typename T>
	T* allocate_array(const size_t n) {
		auto* ptr = new T[n];
		owned_arrays.emplace_back([ptr]() { delete[] ptr; });
		return ptr;
	}

	template <typename T>
	T* own_array(T* ptr) {
		if (ptr != nullptr) {
			owned_arrays.emplace_back([ptr]() { delete[] ptr; });
		}
		return ptr;
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

	auto exc = columns::detail::build_exception_offsets_from_segment<T>(seg_exc, n_vecs);
	auto pos = columns::detail::build_exception_offsets_from_segment<uint16_t>(seg_pos, n_vecs);
	storage.own_array(exc.offsets);
	storage.own_array(pos.offsets);

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
	const auto fv     = *reinterpret_cast<const T*>(seg_fv.data_span.data());
	auto*      fv_arr = storage.allocate_array<T>(n_vecs);
	for (size_t i = 0; i < n_vecs; ++i) {
		fv_arr[i] = fv;
	}

	auto exc = columns::detail::build_exception_offsets_from_segment<T>(seg_exc, n_vecs);
	auto pos = columns::detail::build_exception_offsets_from_segment<uint16_t>(seg_pos, n_vecs);
	storage.own_array(exc.offsets);
	storage.own_array(pos.offsets);

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
	std::shared_ptr<fastlanes::Buf>          backing_buffer;
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

	ZeroCopyRowgroup read_rowgroup_zero_copy(const size_t rowgroup_idx = 0) {
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

		auto          backing = std::make_shared<fastlanes::Buf>(rg->m_size());
		fastlanes::io io      = fastlanes::make_unique<fastlanes::File>(m_file_path);
		fastlanes::IO::range_read(io, *backing, rg->m_offset(), rg->m_size());
		auto view = std::make_shared<fastlanes::RowgroupView>(backing->Span(), *rg);

		const auto&      col_descs = *rg->m_column_descriptors();
		ZeroCopyRowgroup out {};
		out.n_values       = n_values;
		out.n_vecs         = n_vecs;
		out.n_tuples       = n_tuples;
		out.backing_buffer = backing;
		out.rowgroup_view  = view;
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

	Rowgroup read_rowgroup_zero_copy_materialized(const size_t rowgroup_idx = 0) {
		const auto zero_copy    = read_rowgroup_zero_copy(rowgroup_idx);
		auto       storage      = std::make_shared<detail::ZeroCopyHostStorage>();
		storage->backing_buffer = zero_copy.backing_buffer;
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
				default:
					parse_with_copy_fallback();
					break;
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
