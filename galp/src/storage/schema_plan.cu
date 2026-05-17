// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/storage/schema_plan.cu
// ────────────────────────────────────────────────────────
#include "storage/schema_plan.cuh"
#include "fls/expression/rpn.hpp"
#include <flatbuffers/base.h>
#include <sstream>
#include <stdexcept>

namespace galp::format::detail {

std::vector<ZeroCopyColumnPlan> build_zero_copy_column_plan(const fastlanes::RowgroupDescriptor& rg,
                                                            const bool                           load_column_names) {
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
				col.operand_ids.push_back(static_cast<uint32_t>(operands->Get(static_cast<flatbuffers::uoffset_t>(i))));
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

bool rowgroup_matches_zero_copy_plan(const fastlanes::RowgroupDescriptor&   rg,
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

std::vector<size_t> build_zero_copy_column_order(const std::vector<ZeroCopyColumnPlan>& columns) {
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

} // namespace galp::format::detail
