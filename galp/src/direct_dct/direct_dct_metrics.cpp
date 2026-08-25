#include "direct_dct/direct_dct_metrics.hpp"

#include <algorithm>
#include <stdexcept>

namespace galp::direct_dct {
namespace {

constexpr std::array<MetricDescriptor, 14U> kDescriptors {{
    {"schema", MetricValueType::kString, MetricScope::kPipeline, MetricUnit::kIdentifier,
     MetricReducer::kInvariant, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"complete", MetricValueType::kBoolean, MetricScope::kPipeline, MetricUnit::kBoolean,
     MetricReducer::kInvariant, MetricCompletionRequirement::kGpuCompletion, kDirectDctMetricsSchemaVersion},
    {"consumer_wait_ms", MetricValueType::kFloatingPoint, MetricScope::kPipeline, MetricUnit::kMilliseconds,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"submit_to_ready_ms", MetricValueType::kFloatingPoint, MetricScope::kPipeline, MetricUnit::kMilliseconds,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"producer_ms", MetricValueType::kFloatingPoint, MetricScope::kPipeline, MetricUnit::kMilliseconds,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"planning_ms", MetricValueType::kFloatingPoint, MetricScope::kPipeline, MetricUnit::kMilliseconds,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"io_ms", MetricValueType::kFloatingPoint, MetricScope::kPipeline, MetricUnit::kMilliseconds,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"decode_ms", MetricValueType::kFloatingPoint, MetricScope::kPipeline, MetricUnit::kMilliseconds,
     MetricReducer::kSum, MetricCompletionRequirement::kGpuCompletion, kDirectDctMetricsSchemaVersion},
    {"transform_ms", MetricValueType::kFloatingPoint, MetricScope::kPipeline, MetricUnit::kMilliseconds,
     MetricReducer::kSum, MetricCompletionRequirement::kGpuCompletion, kDirectDctMetricsSchemaVersion},
    {"logical_bytes", MetricValueType::kUnsignedInteger, MetricScope::kPipeline, MetricUnit::kBytes,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"physical_bytes", MetricValueType::kUnsignedInteger, MetricScope::kPipeline, MetricUnit::kBytes,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"peak_transient_bytes", MetricValueType::kUnsignedInteger, MetricScope::kPipeline, MetricUnit::kBytes,
     MetricReducer::kMaximum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"consumed_batches", MetricValueType::kUnsignedInteger, MetricScope::kPipeline, MetricUnit::kCount,
     MetricReducer::kSum, MetricCompletionRequirement::kHostSnapshot, kDirectDctMetricsSchemaVersion},
    {"completed_batches", MetricValueType::kUnsignedInteger, MetricScope::kPipeline, MetricUnit::kCount,
     MetricReducer::kSum, MetricCompletionRequirement::kGpuCompletion, kDirectDctMetricsSchemaVersion},
}};

void validate_observation(const DirectDctMetricsObservation& observation) {
	if (!observation.host_snapshot_taken) {
		throw std::invalid_argument("Direct-DCT metrics observations require a host snapshot");
	}
	if (observation.completed_batches > observation.consumed_batches) {
		throw std::invalid_argument("Direct-DCT completed batch count exceeds consumed batch count");
	}
	if (observation.gpu_timings_finalized &&
	    observation.completed_batches != observation.consumed_batches) {
		throw std::invalid_argument("finalized Direct-DCT metrics must complete every consumed batch");
	}
}

} // namespace

std::span<const MetricDescriptor> direct_dct_metric_descriptors() noexcept {
	return kDescriptors;
}

void DirectDctMetricsAggregator::reset() noexcept {
	values_ = {};
}

void DirectDctMetricsAggregator::observe_host(const DirectDctMetricsObservation& observation) {
	validate_observation(observation);
	values_.host_snapshot_taken = values_.host_snapshot_taken && observation.host_snapshot_taken;
	values_.consumer_wait_ms += observation.consumer_wait_ms;
	values_.submit_to_ready_ms += observation.submit_to_ready_ms;
	values_.producer_ms += observation.producer_ms;
	values_.planning_ms += observation.planning_ms;
	values_.io_ms += observation.io_ms;
	values_.logical_bytes += observation.logical_bytes;
	values_.physical_bytes += observation.physical_bytes;
	values_.peak_transient_bytes =
	    std::max(values_.peak_transient_bytes, observation.peak_transient_bytes);
	values_.consumed_batches += observation.consumed_batches;
}

void DirectDctMetricsAggregator::observe_gpu_completion(
    const double decode_ms,
    const double transform_ms,
    const uint64_t peak_transient_bytes) noexcept {
	values_.decode_ms += decode_ms;
	values_.transform_ms += transform_ms;
	values_.peak_transient_bytes = std::max(values_.peak_transient_bytes, peak_transient_bytes);
	++values_.completed_batches;
}

void DirectDctMetricsAggregator::observe(const DirectDctMetricsObservation& observation) {
	observe_host(observation);
	values_.decode_ms += observation.decode_ms;
	values_.transform_ms += observation.transform_ms;
	values_.completed_batches += observation.completed_batches;
}

DirectDctMetricsSnapshot DirectDctMetricsAggregator::snapshot() const noexcept {
	auto out = values_;
	out.gpu_timings_finalized = out.completed_batches == out.consumed_batches;
	return out;
}

} // namespace galp::direct_dct
