// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/format/schema_plan.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_FORMAT_SCHEMA_PLAN_CUH
#define GALP_ENGINE_FORMAT_SCHEMA_PLAN_CUH

#include "fls/footer/operator_token_generated.h"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace galp::format {

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

namespace detail {

std::vector<ZeroCopyColumnPlan> build_zero_copy_column_plan(const fastlanes::RowgroupDescriptor& rg,
                                                            bool                                 load_column_names);

bool rowgroup_matches_zero_copy_plan(const fastlanes::RowgroupDescriptor&   rg,
                                     const std::vector<ZeroCopyColumnPlan>& plan);

std::vector<size_t> build_zero_copy_column_order(const std::vector<ZeroCopyColumnPlan>& columns);

} // namespace detail

} // namespace galp::format

#endif // GALP_ENGINE_FORMAT_SCHEMA_PLAN_CUH
