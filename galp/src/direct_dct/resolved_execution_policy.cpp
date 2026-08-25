#include "direct_dct/resolved_execution_policy.hpp"
#include <stdexcept>
#include <string>

namespace galp::direct_dct {
namespace {

ResolvedExecutionPolicy compact_policy(const std::string_view source_profile_id) {
	return ResolvedExecutionPolicy {
	    source_profile_id,
	    "compact-v3-planless-limited-o512-c512-v1",
	    jpeg::JpegDctDeviceLayout::kTransformedDctGrid,
	    0U,
	    64U,
	    jpeg::kDefaultJpegDctDeviceDecodeWorksetCapacityBytes,
	    0U,
	    true,
	    16U,
	    8U,
	    1U,
	    true,
	    jpeg::JpegDctSchedulingPolicy::kLimitedOverlap,
	    512U,
	    512U,
	    true,
	    true,
	    ResolvedExecutionPolicy::SubmissionGatePolicy::kCreatePerAsyncBatch,
	    jpeg::JpegDctBlockMajorDoubleBufferPolicy::kAutomatic,
	    jpeg::JpegDctCropExecutionMode::kAutomatic,
	    1'000'000U,
	    0U,
	    0U,
	};
}

} // namespace

ResolvedExecutionPolicy resolve_execution_policy(const std::string_view semantic_profile_id) {
	if (semantic_profile_id == SemanticProfileRegistry::kProfileIds[0]) {
		return compact_policy(semantic_profile_id);
	}
	if (semantic_profile_id == SemanticProfileRegistry::kProfileIds[1]) {
		auto compact = compact_policy(semantic_profile_id);
		return ResolvedExecutionPolicy {
		    compact.source_profile_id,
		    "block-major-p4-scheduled-bounded-110-v1",
		    compact.layout,
		    compact.cache_capacity_bytes,
		    compact.decode_batch_rowgroups,
		    compact.decode_workset_capacity_bytes,
		    compact.plan_cache_capacity,
		    compact.enable_rowgroup_prefetch,
		    compact.rowgroup_prefetch_depth,
		    compact.rowgroup_prefetch_workers,
		    compact.rowgroup_prefetch_min_decode_batches,
		    compact.enable_planless_execution,
		    compact.scheduling_policy,
		    compact.transform_blocks_per_launch,
		    compact.transform_ctas_per_launch,
		    compact.use_low_priority_streams,
		    compact.async_planless_completion,
		    compact.submission_gate_policy,
		    compact.block_major_double_buffer_policy,
		    jpeg::JpegDctCropExecutionMode::kBoundedIoUringScheduledRangeReadSelectedDecode,
		    1'100'000U,
		    0U,
		    0U,
		};
	}
	if (semantic_profile_id == SemanticProfileRegistry::kProfileIds[2]) {
		auto compact = compact_policy(semantic_profile_id);
		return ResolvedExecutionPolicy {
		    compact.source_profile_id,
		    "block-major-dynamic-crop-p4-bounded-110-v1",
		    compact.layout,
		    compact.cache_capacity_bytes,
		    compact.decode_batch_rowgroups,
		    compact.decode_workset_capacity_bytes,
		    compact.plan_cache_capacity,
		    compact.enable_rowgroup_prefetch,
		    compact.rowgroup_prefetch_depth,
		    compact.rowgroup_prefetch_workers,
		    compact.rowgroup_prefetch_min_decode_batches,
		    compact.enable_planless_execution,
		    compact.scheduling_policy,
		    compact.transform_blocks_per_launch,
		    compact.transform_ctas_per_launch,
		    compact.use_low_priority_streams,
		    compact.async_planless_completion,
		    compact.submission_gate_policy,
		    compact.block_major_double_buffer_policy,
		    jpeg::JpegDctCropExecutionMode::kBoundedIoUringRangeReadSelectedDecode,
		    1'100'000U,
		    0U,
		    0U,
		};
	}
	throw std::invalid_argument("unknown shadow Direct-DCT execution profile '" + std::string(semantic_profile_id) +
	                            "'");
}

jpeg::JpegDctDeviceBatchOptions materialize_shadow_options(const SemanticProfileRegistry::SemanticProfile& semantic,
                                                           const ResolvedExecutionPolicy&                  policy) {
	if (semantic.id != policy.source_profile_id) {
		throw std::invalid_argument("shadow semantic profile and execution policy ids do not match");
	}
	if (semantic.output_layout != policy.layout) {
		throw std::invalid_argument("shadow semantic profile and execution policy layouts do not match");
	}

	jpeg::JpegDctDeviceBatchOptions options;
	options.layout                               = policy.layout;
	options.grid_transform                       = semantic.grid_transform;
	options.coefficient_selection                = semantic.coefficient_selection;
	options.cache_capacity_bytes                 = policy.cache_capacity_bytes;
	options.decode_batch_rowgroups               = policy.decode_batch_rowgroups;
	options.decode_workset_capacity_bytes        = policy.decode_workset_capacity_bytes;
	options.plan_cache_capacity                  = policy.plan_cache_capacity;
	options.enable_rowgroup_prefetch             = policy.enable_rowgroup_prefetch;
	options.rowgroup_prefetch_depth              = policy.rowgroup_prefetch_depth;
	options.rowgroup_prefetch_workers            = policy.rowgroup_prefetch_workers;
	options.rowgroup_prefetch_min_decode_batches = policy.rowgroup_prefetch_min_decode_batches;
	options.enable_planless_execution            = policy.enable_planless_execution;
	options.scheduling_policy                    = policy.scheduling_policy;
	options.transform_blocks_per_launch          = policy.transform_blocks_per_launch;
	options.transform_ctas_per_launch            = policy.transform_ctas_per_launch;
	options.use_low_priority_streams             = policy.use_low_priority_streams;
	options.async_planless_completion            = policy.async_planless_completion;
	options.transform_submission_gate.reset();
	options.block_major_double_buffer_policy     = policy.block_major_double_buffer_policy;
	options.crop_execution_mode                  = policy.crop_execution_mode;
	options.bounded_read_amplification_ppm       = policy.bounded_read_amplification_ppm;
	options.bounded_read_local_amplification_ppm = policy.bounded_read_local_amplification_ppm;
	options.bounded_read_max_run_bytes           = policy.bounded_read_max_run_bytes;
	return options;
}

} // namespace galp::direct_dct
