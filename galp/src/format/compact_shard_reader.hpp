#ifndef GALP_FORMAT_COMPACT_SHARD_READER_HPP
#define GALP_FORMAT_COMPACT_SHARD_READER_HPP

#include "format/compact_read_plan.hpp"
#include "format/range_reader.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace galp::format {

struct CompactRowgroupBacking {
	uint32_t               rowgroup_index = 0U;
	std::vector<std::byte> bytes;
};

struct CompactShardReadResult {
	std::vector<CompactRowgroupBacking> rowgroups;
	RangeReaderStats                    io;
};

class CompactShardReader {
public:
	explicit CompactShardReader(const std::filesystem::path& shard_path);
	[[nodiscard]] const CompactDescriptorV3& descriptor() const noexcept {
		return descriptor_;
	}
	[[nodiscard]] CompactReadPlan Plan(const std::vector<uint32_t>&  rowgroups,
	                                   const std::vector<uint8_t>&   coefficients,
	                                   const CompactReadPlanOptions& options = {}) const;
	CompactShardReadResult        Read(const CompactReadPlan& plan);

private:
	CompactDescriptorV3 descriptor_;
	RangeReader         range_reader_;
};

} // namespace galp::format

#endif // GALP_FORMAT_COMPACT_SHARD_READER_HPP
