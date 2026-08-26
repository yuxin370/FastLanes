#ifndef GALP_DIRECT_DCT_PHYSICAL_LAYOUT_PLANNER_HPP
#define GALP_DIRECT_DCT_PHYSICAL_LAYOUT_PLANNER_HPP

#include "direct_dct/logical_types.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace galp::direct_dct {

// A read-only description of how one logical batch intersects physical
// manifest shards. It contains no CUDA, allocator, or scheduling state.
struct SegmentPlan final {
	uint32_t shard_id             = 0U;
	size_t   logical_output_offset = 0U;
	size_t   image_count           = 0U;
	uint32_t first_global_image_id = 0U;
	uint32_t last_global_image_id  = 0U;
};

struct LogicalBatchPhysicalPlan final {
	uint64_t                 request_identity = 0U;
	uint64_t                 batch_ordinal    = 0U;
	size_t                   logical_image_count = 0U;
	std::vector<SegmentPlan> segments;
};

// Canonical native owner of manifest-shard segmentation. DirectDctRuntime
// remains the owner of the detailed rowgroup/layout plan and all I/O/CUDA
// execution; this planner only removes physical shard decisions from Python.
class PhysicalLayoutPlanner final {
public:
	explicit PhysicalLayoutPlanner(const std::filesystem::path& manifest_path);
	explicit PhysicalLayoutPlanner(jpeg::JpegDctShardManifest manifest);

	[[nodiscard]] LogicalBatchPhysicalPlan plan(const LogicalBatchRequest& request) const;
	[[nodiscard]] std::vector<LogicalBatchPhysicalPlan>
	plan(const std::vector<LogicalBatchRequest>& requests) const;
	[[nodiscard]] const jpeg::JpegDctShardManifest& manifest() const noexcept;

private:
	[[nodiscard]] const jpeg::JpegDctShardManifestEntry& locate(uint32_t image_id) const;

	jpeg::JpegDctShardManifest manifest_;
};

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_PHYSICAL_LAYOUT_PLANNER_HPP
