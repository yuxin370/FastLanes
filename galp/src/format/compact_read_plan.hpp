#ifndef GALP_FORMAT_COMPACT_READ_PLAN_HPP
#define GALP_FORMAT_COMPACT_READ_PLAN_HPP

#include "format/compact_descriptor_v3.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace galp::format {

struct CompactReadRange {
	uint64_t file_offset    = 0U;
	uint32_t size           = 0U;
	uint32_t rowgroup_index = 0U;
	uint32_t backing_offset = 0U;
};

struct CompactReadPlanStats {
	uint64_t logical_bytes              = 0U;
	uint64_t read_bytes                 = 0U;
	uint64_t full_rowgroup_bytes        = 0U;
	uint64_t physical_page_bytes        = 0U;
	uint64_t full_physical_page_bytes   = 0U;
	uint64_t range_count                = 0U;
	uint64_t coalesced_run_count        = 0U;
	uint64_t selected_rowgroup_count    = 0U;
	uint64_t full_rowgroup_count        = 0U;
	uint64_t selected_coefficient_count = 0U;
	uint64_t full_coefficient_count     = 0U;
	double   selected_vector_ratio      = 0.0;
	double   selected_coefficient_ratio = 0.0;
	double   physical_page_ratio        = 0.0;
};

struct CompactReadPlanOptions {
	uint32_t page_size                    = 4096U;
	bool     coalesce_ranges_on_same_page = true;
};

class CompactReadPlan {
public:
	[[nodiscard]] const std::vector<CompactReadRange>& ranges() const noexcept {
		return ranges_;
	}
	[[nodiscard]] const std::vector<uint32_t>& rowgroups() const noexcept {
		return rowgroups_;
	}
	[[nodiscard]] const std::vector<uint8_t>& requested_coefficients() const noexcept {
		return requested_coefficients_;
	}
	[[nodiscard]] const CompactReadPlanStats& stats() const noexcept {
		return stats_;
	}

private:
	friend CompactReadPlan        compile_compact_read_plan(const CompactDescriptorV3&,
	                                                        const std::vector<uint32_t>&,
	                                                        const std::vector<uint8_t>&,
	                                                        const CompactReadPlanOptions&);
	std::vector<CompactReadRange> ranges_;
	std::vector<uint32_t>         rowgroups_;
	std::vector<uint8_t>          requested_coefficients_;
	CompactReadPlanStats          stats_;
};

CompactReadPlan compile_compact_read_plan(const CompactDescriptorV3&    descriptor,
                                          const std::vector<uint32_t>&  selected_rowgroups,
                                          const std::vector<uint8_t>&   selected_coefficients,
                                          const CompactReadPlanOptions& options = {});

} // namespace galp::format

#endif // GALP_FORMAT_COMPACT_READ_PLAN_HPP
