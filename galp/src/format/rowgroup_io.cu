// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/format/rowgroup_io.cu
// ────────────────────────────────────────────────────────
#include "format/rowgroup_io.cuh"
#include "format/compact_descriptor_v3.hpp"
#include "fls/expression/rpn.hpp"
#include "galp/errors.hpp"
#include <flatbuffers/base.h>
#include <limits>
#include <span>
#include <stdexcept>

namespace galp::format {

size_t zero_copy_operand_count(const ZeroCopyColumn& col) {
	if (col.operand_ids != nullptr) {
		return col.operand_ids->size();
	}
	return col.operand_tokens != nullptr ? col.operand_tokens->size() : 0;
}

uint32_t zero_copy_operand(const ZeroCopyColumn& col, const size_t idx) {
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

const std::string& zero_copy_column_name(const ZeroCopyColumn& col) {
	if (col.name_ref != nullptr) {
		return *col.name_ref;
	}
	return col.name;
}

[[noreturn]] void throw_unsupported_zero_copy_token(const fastlanes::OperatorToken token,
                                                    const size_t                   rowgroup_index,
                                                    const size_t                   column_index,
                                                    const std::string&             column_name) {
	throw galp::UnsupportedFormatError(fastlanes::token_to_string(token), rowgroup_index, column_index, column_name);
}

fastlanes::SegmentView zero_copy_segment(const ZeroCopyColumn& col, const uint32_t segment_idx) {
	if (col.column_view != nullptr) {
		return col.column_view->GetSegment(segment_idx);
	}
	if (col.compact_rowgroup != nullptr && col.compact_column != nullptr) {
		const auto& direct_column = *col.compact_column;
		if (segment_idx >= direct_column.segment_count ||
		    direct_column.segment_begin > col.compact_rowgroup->segments.size() ||
		    segment_idx > col.compact_rowgroup->segments.size() - direct_column.segment_begin - 1U) {
			throw std::out_of_range("compact zero-copy segment index out of range");
		}
		const auto& segment = col.compact_rowgroup->segments[direct_column.segment_begin + segment_idx];
		if (segment.entrypoint_offset > col.column_span.size() || segment.entrypoint_size >
		        col.column_span.size() - static_cast<size_t>(segment.entrypoint_offset) ||
		    segment.data_offset > col.column_span.size() ||
		    segment.data_size > col.column_span.size() - static_cast<size_t>(segment.data_offset)) {
			throw std::out_of_range("compact zero-copy segment geometry exceeds rowgroup backing");
		}
		auto entrypoint_span = col.column_span.subspan(static_cast<size_t>(segment.entrypoint_offset),
		                                                static_cast<size_t>(segment.entrypoint_size));
		auto data_span = col.column_span.subspan(static_cast<size_t>(segment.data_offset),
		                                      static_cast<size_t>(segment.data_size));
		switch (segment.entry_point_type) {
		case fastlanes::EntryPointType::UINT8:
			return fastlanes::SegmentView {
			    fastlanes::EntryPointView<uint8_t>(std::span<uint8_t>(
			        reinterpret_cast<uint8_t*>(entrypoint_span.data()), entrypoint_span.size())),
			    data_span};
		case fastlanes::EntryPointType::UINT16:
			if (entrypoint_span.size() % sizeof(uint16_t) != 0U) {
				throw std::runtime_error("compact UINT16 entry-point segment has invalid size");
			}
			return fastlanes::SegmentView {
			    fastlanes::EntryPointView<uint16_t>(std::span<uint16_t>(
			        reinterpret_cast<uint16_t*>(entrypoint_span.data()), entrypoint_span.size() / sizeof(uint16_t))),
			    data_span};
		case fastlanes::EntryPointType::UINT32:
			if (entrypoint_span.size() % sizeof(uint32_t) != 0U) {
				throw std::runtime_error("compact UINT32 entry-point segment has invalid size");
			}
			return fastlanes::SegmentView {
			    fastlanes::EntryPointView<uint32_t>(std::span<uint32_t>(
			        reinterpret_cast<uint32_t*>(entrypoint_span.data()), entrypoint_span.size() / sizeof(uint32_t))),
			    data_span};
		case fastlanes::EntryPointType::UINT64:
		default:
			throw std::runtime_error("compact zero-copy segment uses an unsupported entry-point type");
		}
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

const uint8_t* zero_copy_maximum_data(const ZeroCopyColumn& col) {
	if (col.compact_rowgroup != nullptr && col.compact_column != nullptr) {
		const auto& direct_column = *col.compact_column;
		if (direct_column.maximum_offset > col.compact_rowgroup->maximum_bytes.size() ||
		    direct_column.maximum_size >
		        col.compact_rowgroup->maximum_bytes.size() - direct_column.maximum_offset) {
			throw std::out_of_range("compact zero-copy maximum exceeds decoded scalar bytes");
		}
		return direct_column.maximum_size == 0U
		           ? nullptr
		           : col.compact_rowgroup->maximum_bytes.data() + direct_column.maximum_offset;
	}
	if (col.column_descriptor == nullptr || col.column_descriptor->max() == nullptr ||
	    col.column_descriptor->max()->binary_data() == nullptr) {
		return nullptr;
	}
	return col.column_descriptor->max()->binary_data()->data();
}

size_t zero_copy_maximum_size(const ZeroCopyColumn& col) {
	if (col.compact_rowgroup != nullptr && col.compact_column != nullptr) {
		return col.compact_column->maximum_size;
	}
	if (col.column_descriptor == nullptr || col.column_descriptor->max() == nullptr ||
	    col.column_descriptor->max()->binary_data() == nullptr) {
		return 0U;
	}
	return col.column_descriptor->max()->binary_data()->size();
}

ZeroCopyColumn make_zero_copy_column_from_plan(const ZeroCopyRowgroup& rowgroup, const ZeroCopyColumnPlan& plan_col) {
	if (rowgroup.rowgroup_descriptor == nullptr && rowgroup.compact_direct_owner == nullptr) {
		throw std::runtime_error("zero-copy rowgroup plan metadata missing");
	}

	ZeroCopyColumn col {};
	col.column_index      = plan_col.column_index;
	col.name_ref          = &plan_col.name;
	col.token             = plan_col.token;
	if (rowgroup.compact_direct_owner != nullptr) {
		if (plan_col.column_index >= rowgroup.compact_direct_owner->columns.size()) {
			throw std::out_of_range("compact zero-copy column plan index out of range");
		}
		col.compact_rowgroup  = rowgroup.compact_direct_owner.get();
		col.compact_column    = &rowgroup.compact_direct_owner->columns[plan_col.column_index];
		col.column_descriptor = col.compact_column->schema;
	} else {
		const auto* col_descs = rowgroup.rowgroup_descriptor->m_column_descriptors();
		if (col_descs == nullptr || plan_col.column_index >= col_descs->size()) {
			throw std::out_of_range("zero-copy column plan index out of range");
		}
		col.column_descriptor = col_descs->Get(static_cast<flatbuffers::uoffset_t>(plan_col.column_index));
	}
	col.operand_ids       = &plan_col.operand_ids;
	col.column_view       = rowgroup.rowgroup_view != nullptr
	                            ? &(*rowgroup.rowgroup_view)[static_cast<fastlanes::n_t>(plan_col.column_index)]
	                            : nullptr;
	col.column_span       = rowgroup.backing_span;
	col.skip_decompress   = plan_col.skip_decompress;
	col.alias_of          = plan_col.alias_of;
	return col;
}

} // namespace galp::format
