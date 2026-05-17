// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/storage/rowgroup_io.cu
// ────────────────────────────────────────────────────────
#include "storage/rowgroup_io.cuh"
#include "fls/expression/rpn.hpp"
#include "galp/errors.hpp"
#include <flatbuffers/base.h>
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

ZeroCopyColumn make_zero_copy_column_from_plan(const ZeroCopyRowgroup& rowgroup, const ZeroCopyColumnPlan& plan_col) {
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

} // namespace galp::format
