#include "direct_dct/direct_dct_metrics.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace {

using galp::direct_dct::DirectDctMetricsAggregator;
using galp::direct_dct::DirectDctMetricsObservation;

TEST(DirectDctMetrics, DescriptorInventoryIsCompleteAndUnique) {
	const auto descriptors = galp::direct_dct::direct_dct_metric_descriptors();
	std::unordered_set<std::string_view> names;
	for (const auto& descriptor : descriptors) {
		EXPECT_EQ(descriptor.schema_version, galp::direct_dct::kDirectDctMetricsSchemaVersion);
		EXPECT_TRUE(names.insert(descriptor.name).second) << descriptor.name;
	}
	for (const std::string_view required : {
	         "schema", "complete", "consumer_wait_ms", "submit_to_ready_ms",
	         "producer_ms", "planning_ms", "io_ms", "decode_ms", "transform_ms",
	         "logical_bytes", "physical_bytes", "peak_transient_bytes",
	         "consumed_batches", "completed_batches"}) {
		EXPECT_TRUE(names.contains(required)) << required;
	}
	EXPECT_EQ(names.size(), descriptors.size());
}

TEST(DirectDctMetrics, HostAndGpuCompletionAreIndependent) {
	DirectDctMetricsAggregator aggregator;
	DirectDctMetricsObservation host;
	host.consumer_wait_ms = 1.5;
	host.submit_to_ready_ms = 2.5;
	host.producer_ms = 3.5;
	host.planning_ms = 4.5;
	host.io_ms = 5.5;
	host.logical_bytes = 100U;
	host.physical_bytes = 80U;
	host.peak_transient_bytes = 64U;
	aggregator.observe_host(host);

	auto snapshot = aggregator.snapshot();
	EXPECT_TRUE(snapshot.host_snapshot_taken);
	EXPECT_FALSE(snapshot.gpu_timings_finalized);
	EXPECT_EQ(snapshot.consumed_batches, 1U);
	EXPECT_EQ(snapshot.completed_batches, 0U);
	EXPECT_DOUBLE_EQ(snapshot.decode_ms, 0.0);

	aggregator.observe_gpu_completion(6.5, 7.5, 96U);
	snapshot = aggregator.snapshot();
	EXPECT_TRUE(snapshot.gpu_timings_finalized);
	EXPECT_EQ(snapshot.completed_batches, 1U);
	EXPECT_DOUBLE_EQ(snapshot.decode_ms, 6.5);
	EXPECT_DOUBLE_EQ(snapshot.transform_ms, 7.5);
	EXPECT_EQ(snapshot.peak_transient_bytes, 96U);
}

TEST(DirectDctMetrics, NativeReducersMatchStableGolden) {
	DirectDctMetricsAggregator aggregator;
	DirectDctMetricsObservation first;
	first.gpu_timings_finalized = true;
	first.consumer_wait_ms = 1.0;
	first.submit_to_ready_ms = 2.0;
	first.producer_ms = 3.0;
	first.planning_ms = 4.0;
	first.io_ms = 5.0;
	first.decode_ms = 6.0;
	first.transform_ms = 7.0;
	first.logical_bytes = 100U;
	first.physical_bytes = 80U;
	first.peak_transient_bytes = 64U;
	first.completed_batches = 1U;
	aggregator.observe(first);

	auto second = first;
	second.consumer_wait_ms = 10.0;
	second.submit_to_ready_ms = 20.0;
	second.producer_ms = 30.0;
	second.planning_ms = 40.0;
	second.io_ms = 50.0;
	second.decode_ms = 60.0;
	second.transform_ms = 70.0;
	second.logical_bytes = 1000U;
	second.physical_bytes = 800U;
	second.peak_transient_bytes = 32U;
	aggregator.observe(second);

	const auto snapshot = aggregator.snapshot();
	EXPECT_TRUE(snapshot.gpu_timings_finalized);
	EXPECT_DOUBLE_EQ(snapshot.consumer_wait_ms, 11.0);
	EXPECT_DOUBLE_EQ(snapshot.submit_to_ready_ms, 22.0);
	EXPECT_DOUBLE_EQ(snapshot.producer_ms, 33.0);
	EXPECT_DOUBLE_EQ(snapshot.planning_ms, 44.0);
	EXPECT_DOUBLE_EQ(snapshot.io_ms, 55.0);
	EXPECT_DOUBLE_EQ(snapshot.decode_ms, 66.0);
	EXPECT_DOUBLE_EQ(snapshot.transform_ms, 77.0);
	EXPECT_EQ(snapshot.logical_bytes, 1100U);
	EXPECT_EQ(snapshot.physical_bytes, 880U);
	EXPECT_EQ(snapshot.peak_transient_bytes, 64U);
	EXPECT_EQ(snapshot.consumed_batches, 2U);
	EXPECT_EQ(snapshot.completed_batches, 2U);
}

TEST(DirectDctMetrics, PreservesFinalizedSubsetOfIncompletePipelineSnapshot) {
	DirectDctMetricsAggregator aggregator;
	DirectDctMetricsObservation partial;
	partial.gpu_timings_finalized = false;
	partial.decode_ms = 9.0;
	partial.transform_ms = 3.0;
	partial.consumed_batches = 2U;
	partial.completed_batches = 1U;
	aggregator.observe(partial);

	const auto snapshot = aggregator.snapshot();
	EXPECT_FALSE(snapshot.gpu_timings_finalized);
	EXPECT_EQ(snapshot.consumed_batches, 2U);
	EXPECT_EQ(snapshot.completed_batches, 1U);
	EXPECT_DOUBLE_EQ(snapshot.decode_ms, 9.0);
	EXPECT_DOUBLE_EQ(snapshot.transform_ms, 3.0);
}

TEST(DirectDctMetrics, RejectsInconsistentFinalizedSnapshot) {
	DirectDctMetricsAggregator aggregator;
	DirectDctMetricsObservation invalid;
	invalid.gpu_timings_finalized = true;
	invalid.consumed_batches = 2U;
	invalid.completed_batches = 1U;
	EXPECT_THROW(aggregator.observe(invalid), std::invalid_argument);
}

} // namespace
