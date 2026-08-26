#include "direct_dct/physical_layout_planner.hpp"
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

galp::jpeg::JpegDctShardManifest manifest() {
	galp::jpeg::JpegDctShardManifest result;
	result.image_count = 2073U;
	result.shards = {
	    {.shard_id = 0U, .first_global_image_index = 0U, .image_count = 1024U},
	    {.shard_id = 1U, .first_global_image_index = 1024U, .image_count = 1024U},
	    {.shard_id = 2U, .first_global_image_index = 2048U, .image_count = 25U},
	};
	return result;
}

galp::direct_dct::LogicalBatchRequest request(const uint32_t first, const size_t count) {
	galp::direct_dct::LogicalBatchRequest result;
	result.request_identity = 17U;
	result.batch_ordinal = 3U;
	result.semantic_profile_id = "rgbnomore-validation-v1";
	result.logical_batch_size = 50U;
	for (size_t index = 0U; index < count; ++index) {
		galp::direct_dct::LogicalBatchRequest::Sample sample;
		sample.image_id = first + static_cast<uint32_t>(index);
		sample.transform.logical_sample_id = std::to_string(sample.image_id);
		result.samples.push_back(std::move(sample));
	}
	return result;
}

TEST(PhysicalLayoutPlanner, SingleShardLogicalBatch) {
	const galp::direct_dct::PhysicalLayoutPlanner planner(manifest());
	const auto plan = planner.plan(request(50U, 50U));
	ASSERT_EQ(plan.segments.size(), 1U);
	EXPECT_EQ(plan.logical_image_count, 50U);
	EXPECT_EQ(plan.segments[0].shard_id, 0U);
	EXPECT_EQ(plan.segments[0].logical_output_offset, 0U);
	EXPECT_EQ(plan.segments[0].image_count, 50U);
	EXPECT_EQ(plan.segments[0].first_global_image_id, 50U);
	EXPECT_EQ(plan.segments[0].last_global_image_id, 99U);
}

TEST(PhysicalLayoutPlanner, BatchFiftyCrossesShardBoundaryExactly) {
	const galp::direct_dct::PhysicalLayoutPlanner planner(manifest());
	const auto plan = planner.plan(request(1000U, 50U));
	ASSERT_EQ(plan.segments.size(), 2U);
	EXPECT_EQ(plan.segments[0].shard_id, 0U);
	EXPECT_EQ(plan.segments[0].logical_output_offset, 0U);
	EXPECT_EQ(plan.segments[0].image_count, 24U);
	EXPECT_EQ(plan.segments[0].first_global_image_id, 1000U);
	EXPECT_EQ(plan.segments[0].last_global_image_id, 1023U);
	EXPECT_EQ(plan.segments[1].shard_id, 1U);
	EXPECT_EQ(plan.segments[1].logical_output_offset, 24U);
	EXPECT_EQ(plan.segments[1].image_count, 26U);
	EXPECT_EQ(plan.segments[1].first_global_image_id, 1024U);
	EXPECT_EQ(plan.segments[1].last_global_image_id, 1049U);
}

TEST(PhysicalLayoutPlanner, PartialFinalBatch) {
	const galp::direct_dct::PhysicalLayoutPlanner planner(manifest());
	const auto plan = planner.plan(request(2048U, 25U));
	ASSERT_EQ(plan.segments.size(), 1U);
	EXPECT_EQ(plan.logical_image_count, 25U);
	EXPECT_EQ(plan.segments[0].shard_id, 2U);
	EXPECT_EQ(plan.segments[0].logical_output_offset, 0U);
	EXPECT_EQ(plan.segments[0].image_count, 25U);
	EXPECT_EQ(plan.segments[0].first_global_image_id, 2048U);
	EXPECT_EQ(plan.segments[0].last_global_image_id, 2072U);
}

} // namespace
