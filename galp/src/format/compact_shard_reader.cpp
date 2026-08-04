#include "format/compact_shard_reader.hpp"
#include <algorithm>
#include <stdexcept>

namespace galp::format {

CompactShardReader::CompactShardReader(const std::filesystem::path& shard_path)
    : descriptor_(CompactDescriptorV3::Open(shard_path))
    , range_reader_(shard_path) {
}

CompactReadPlan CompactShardReader::Plan(const std::vector<uint32_t>&  rowgroups,
                                         const std::vector<uint8_t>&   coefficients,
                                         const CompactReadPlanOptions& options) const {
	return compile_compact_read_plan(descriptor_, rowgroups, coefficients, options);
}

CompactShardReadResult CompactShardReader::Read(const CompactReadPlan& plan) {
	CompactShardReadResult result;
	result.rowgroups.reserve(plan.rowgroups().size());
	for (const auto rowgroup_index : plan.rowgroups()) {
		CompactRowgroupBacking backing;
		backing.rowgroup_index = rowgroup_index;
		backing.bytes.resize(descriptor_.rowgroup(rowgroup_index).payload_size, std::byte {0});
		result.rowgroups.push_back(std::move(backing));
	}
	const auto stats_before = range_reader_.stats();
	for (const auto& range : plan.ranges()) {
		const auto found =
		    std::lower_bound(result.rowgroups.begin(),
		                     result.rowgroups.end(),
		                     range.rowgroup_index,
		                     [](const auto& backing, const auto index) { return backing.rowgroup_index < index; });
		if (found == result.rowgroups.end() || found->rowgroup_index != range.rowgroup_index ||
		    range.backing_offset > found->bytes.size() || range.size > found->bytes.size() - range.backing_offset) {
			throw std::runtime_error("Compact v3 read range does not fit its rowgroup backing");
		}
		range_reader_.Read(range.file_offset,
		                   range.size,
		                   found->bytes.data() + range.backing_offset,
		                   found->bytes.size() - range.backing_offset);
	}
	const auto stats_after = range_reader_.stats();
	result.io.bytes_read   = stats_after.bytes_read - stats_before.bytes_read;
	result.io.pread_count  = stats_after.pread_count - stats_before.pread_count;
	return result;
}

} // namespace galp::format
