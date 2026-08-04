#include "format/compact_read_plan.hpp"
#include "fls/footer/operator_token_generated.h"
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>

namespace galp::format {
namespace {

bool is_external_column_reference(const fastlanes::OperatorToken token) {
	return token == fastlanes::OperatorToken::EXP_EQUAL || token == fastlanes::OperatorToken::EXP_DICT_I08_U08 ||
	       token == fastlanes::OperatorToken::EXP_DICT_I16_U08 || token == fastlanes::OperatorToken::EXP_DICT_I16_U16;
}

void add_column_dependencies(const fastlanes::RowgroupDescriptorT& rowgroup,
                             const size_t                          column_index,
                             std::vector<uint8_t>&                 selected,
                             std::vector<uint8_t>&                 visiting) {
	if (column_index >= rowgroup.m_column_descriptors.size()) {
		throw std::out_of_range("Compact v3 coefficient dependency is outside the rowgroup schema");
	}
	if (selected[column_index] != 0U) {
		return;
	}
	if (visiting[column_index] != 0U) {
		throw std::runtime_error("Compact v3 coefficient dependency cycle detected");
	}
	visiting[column_index] = 1U;
	const auto& column     = rowgroup.m_column_descriptors[column_index];
	if (column == nullptr || column->encoding_rpn == nullptr || column->encoding_rpn->operator_tokens.empty()) {
		throw std::runtime_error("Compact v3 coefficient schema is missing its encoding expression");
	}
	const auto token = column->encoding_rpn->operator_tokens.front();
	if (is_external_column_reference(token)) {
		if (column->encoding_rpn->operand_tokens.empty()) {
			throw std::runtime_error("Compact v3 external coefficient reference has no operand");
		}
		const auto dependency = column->encoding_rpn->operand_tokens.front();
		if (dependency > std::numeric_limits<size_t>::max()) {
			throw std::out_of_range("Compact v3 coefficient dependency exceeds addressable memory");
		}
		add_column_dependencies(rowgroup, static_cast<size_t>(dependency), selected, visiting);
	}
	visiting[column_index] = 0U;
	selected[column_index] = 1U;
}

uint64_t checked_add(const uint64_t left, const uint64_t right, const char* const label) {
	if (right > std::numeric_limits<uint64_t>::max() - left) {
		throw std::overflow_error(std::string("Compact v3 ") + label + " overflow");
	}
	return left + right;
}

} // namespace

CompactReadPlan compile_compact_read_plan(const CompactDescriptorV3&    descriptor,
                                          const std::vector<uint32_t>&  selected_rowgroups,
                                          const std::vector<uint8_t>&   selected_coefficients,
                                          const CompactReadPlanOptions& options) {
	if (options.page_size == 0U || (options.page_size & (options.page_size - 1U)) != 0U) {
		throw std::invalid_argument("Compact v3 read-plan page size must be a non-zero power of two");
	}
	if (selected_rowgroups.empty()) {
		throw std::invalid_argument("Compact v3 read plan requires at least one rowgroup");
	}
	if (selected_coefficients.empty()) {
		throw std::invalid_argument("Compact v3 read plan requires at least one coefficient");
	}

	CompactReadPlan plan;
	plan.rowgroups_              = selected_rowgroups;
	plan.requested_coefficients_ = selected_coefficients;
	std::sort(plan.rowgroups_.begin(), plan.rowgroups_.end());
	plan.rowgroups_.erase(std::unique(plan.rowgroups_.begin(), plan.rowgroups_.end()), plan.rowgroups_.end());
	std::sort(plan.requested_coefficients_.begin(), plan.requested_coefficients_.end());
	plan.requested_coefficients_.erase(
	    std::unique(plan.requested_coefficients_.begin(), plan.requested_coefficients_.end()),
	    plan.requested_coefficients_.end());
	if (plan.rowgroups_.back() >= descriptor.rowgroup_count()) {
		throw std::out_of_range("Compact v3 selected rowgroup is out of range");
	}
	if (plan.requested_coefficients_.back() >= descriptor.column_count()) {
		throw std::out_of_range("Compact v3 selected coefficient is out of range");
	}

	plan.stats_.selected_rowgroup_count    = plan.rowgroups_.size();
	plan.stats_.full_rowgroup_count        = descriptor.rowgroup_count();
	plan.stats_.selected_coefficient_count = plan.requested_coefficients_.size();
	plan.stats_.full_coefficient_count     = descriptor.column_count();
	plan.stats_.selected_vector_ratio =
	    descriptor.rowgroup_count() == 0U
	        ? 0.0
	        : static_cast<double>(plan.rowgroups_.size()) / static_cast<double>(descriptor.rowgroup_count());
	plan.stats_.selected_coefficient_ratio =
	    descriptor.column_count() == 0U
	        ? 0.0
	        : static_cast<double>(plan.requested_coefficients_.size()) / static_cast<double>(descriptor.column_count());

	std::set<uint64_t> covered_pages;
	std::set<uint64_t> full_covered_pages;
	for (const auto rowgroup_index : plan.rowgroups_) {
		const auto record = descriptor.rowgroup(rowgroup_index);
		plan.stats_.full_rowgroup_bytes =
		    checked_add(plan.stats_.full_rowgroup_bytes, record.payload_size, "full-rowgroup byte count");
		if (record.payload_size != 0U) {
			const uint64_t first_full_page = record.payload_offset / options.page_size;
			const uint64_t last_full_page  = (record.payload_offset + record.payload_size - 1U) / options.page_size;
			for (uint64_t page = first_full_page; page <= last_full_page; ++page) {
				full_covered_pages.insert(page);
			}
		}
		const auto           rowgroup = descriptor.unpack_rowgroup(rowgroup_index);
		std::vector<uint8_t> physical_columns(descriptor.column_count(), 0U);
		std::vector<uint8_t> visiting(descriptor.column_count(), 0U);
		for (const auto coefficient : plan.requested_coefficients_) {
			add_column_dependencies(*rowgroup, coefficient, physical_columns, visiting);
		}
		const auto coefficient_ranges = descriptor.coefficient_ranges(rowgroup_index);

		std::vector<CompactReadRange> rowgroup_ranges;
		for (size_t column = 0U; column < physical_columns.size(); ++column) {
			if (physical_columns[column] == 0U) {
				continue;
			}
			const auto coefficient = coefficient_ranges[column];
			if (coefficient.size == 0U) {
				continue;
			}
			plan.stats_.logical_bytes = checked_add(plan.stats_.logical_bytes, coefficient.size, "logical byte count");
			rowgroup_ranges.push_back(
			    {record.payload_offset + coefficient.offset, coefficient.size, rowgroup_index, coefficient.offset});
			const uint64_t first_page = (record.payload_offset + coefficient.offset) / options.page_size;
			const uint64_t last_page =
			    (record.payload_offset + coefficient.offset + coefficient.size - 1U) / options.page_size;
			for (uint64_t page = first_page; page <= last_page; ++page) {
				covered_pages.insert(page);
			}
		}
		plan.stats_.range_count += rowgroup_ranges.size();
		std::sort(rowgroup_ranges.begin(), rowgroup_ranges.end(), [](const auto& left, const auto& right) {
			return left.backing_offset < right.backing_offset;
		});
		for (const auto& range : rowgroup_ranges) {
			if (plan.ranges_.empty() || plan.ranges_.back().rowgroup_index != range.rowgroup_index) {
				plan.ranges_.push_back(range);
				continue;
			}
			auto&          previous     = plan.ranges_.back();
			const uint64_t previous_end = previous.file_offset + previous.size;
			const uint64_t range_end    = range.file_offset + range.size;
			const bool     adjacent     = range.file_offset <= previous_end;
			const bool     same_page    = options.coalesce_ranges_on_same_page && previous.size != 0U &&
			                       (previous_end - 1U) / options.page_size == range.file_offset / options.page_size;
			if (!adjacent && !same_page) {
				plan.ranges_.push_back(range);
				continue;
			}
			const uint64_t merged_end = std::max(previous_end, range_end);
			if (merged_end - previous.file_offset > std::numeric_limits<uint32_t>::max()) {
				throw std::overflow_error("Compact v3 coalesced range exceeds uint32 size");
			}
			previous.size = static_cast<uint32_t>(merged_end - previous.file_offset);
		}
	}
	for (const auto& range : plan.ranges_) {
		plan.stats_.read_bytes = checked_add(plan.stats_.read_bytes, range.size, "read byte count");
	}
	plan.stats_.coalesced_run_count = plan.ranges_.size();
	if (covered_pages.size() > std::numeric_limits<uint64_t>::max() / options.page_size ||
	    full_covered_pages.size() > std::numeric_limits<uint64_t>::max() / options.page_size) {
		throw std::overflow_error("Compact v3 page coverage overflow");
	}
	plan.stats_.physical_page_bytes      = static_cast<uint64_t>(covered_pages.size()) * options.page_size;
	plan.stats_.full_physical_page_bytes = static_cast<uint64_t>(full_covered_pages.size()) * options.page_size;
	plan.stats_.physical_page_ratio      = plan.stats_.full_physical_page_bytes == 0U
	                                           ? 0.0
	                                           : static_cast<double>(plan.stats_.physical_page_bytes) /
                                                static_cast<double>(plan.stats_.full_physical_page_bytes);
	return plan;
}

} // namespace galp::format
