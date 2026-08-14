#ifndef GALP_PROFILES_DIRECT_DCT_HPP
#define GALP_PROFILES_DIRECT_DCT_HPP

#include "galp/jpeg_dct_device.hpp"

#include <optional>
#include <string_view>

namespace galp::profiles {

// Application-independent execution policy.  These fields describe how a
// Direct-DCT request is scheduled and read; they do not describe model output
// semantics.
struct DirectDctRuntimePolicy {
	std::string_view id;
	size_t           cache_capacity_bytes = 0U;
	size_t           decode_batch_rowgroups = jpeg::kDefaultJpegDctDecodeBatchRowgroups;
	size_t           decode_workset_capacity_bytes = jpeg::kDefaultJpegDctDeviceDecodeWorksetCapacityBytes;
	size_t           plan_cache_capacity = jpeg::kDefaultJpegDctDevicePlanCacheCapacity;
	bool             enable_rowgroup_prefetch = true;
	size_t           rowgroup_prefetch_depth = jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth;
	size_t           rowgroup_prefetch_workers = jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers;
	size_t           rowgroup_prefetch_min_decode_batches =
	    jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches;
	bool                          enable_planless_execution = true;
	jpeg::JpegDctSchedulingPolicy scheduling_policy = jpeg::JpegDctSchedulingPolicy::kFullyOverlapped;
	size_t                        transform_blocks_per_launch = 0U;
	size_t                        transform_ctas_per_launch = 0U;
	bool                          use_low_priority_streams = false;
	bool                          async_planless_completion = false;
	jpeg::JpegDctBlockMajorDoubleBufferPolicy block_major_double_buffer_policy =
	    jpeg::JpegDctBlockMajorDoubleBufferPolicy::kAutomatic;
	jpeg::JpegDctCropExecutionMode crop_execution_mode = jpeg::JpegDctCropExecutionMode::kAutomatic;
	uint32_t bounded_read_amplification_ppm = 1'000'000U;
	uint32_t bounded_read_local_amplification_ppm = 0U;
	size_t   bounded_read_max_run_bytes = 0U;
};

// Application-owned output contract.  It defines tensor layout and numerical
// semantics, but contains no allocator, I/O, stream, or launch tuning.
struct DirectDctOutputProfile {
	std::string_view                         id;
	jpeg::JpegDctDeviceLayout                layout = jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::optional<jpeg::JpegDctGridTransformSpec> grid_transform;
	jpeg::JpegDctCoefficientSelection        coefficient_selection {};
};

// A registered public profile composes one semantic output contract with one
// internal runtime policy.  Python selects the public id; implementation
// policy remains native-owned and can evolve under a new policy id.
struct RegisteredDirectDctProfile {
	std::string_view       id;
	DirectDctOutputProfile output;
	DirectDctRuntimePolicy runtime;
};

inline jpeg::JpegDctDeviceBatchOptions materialize_direct_dct_options(
    const RegisteredDirectDctProfile& profile) {
	jpeg::JpegDctDeviceBatchOptions options;
	options.layout                               = profile.output.layout;
	options.grid_transform                       = profile.output.grid_transform;
	options.coefficient_selection                = profile.output.coefficient_selection;
	options.cache_capacity_bytes                 = profile.runtime.cache_capacity_bytes;
	options.decode_batch_rowgroups               = profile.runtime.decode_batch_rowgroups;
	options.decode_workset_capacity_bytes         = profile.runtime.decode_workset_capacity_bytes;
	options.plan_cache_capacity                  = profile.runtime.plan_cache_capacity;
	options.enable_rowgroup_prefetch             = profile.runtime.enable_rowgroup_prefetch;
	options.rowgroup_prefetch_depth              = profile.runtime.rowgroup_prefetch_depth;
	options.rowgroup_prefetch_workers            = profile.runtime.rowgroup_prefetch_workers;
	options.rowgroup_prefetch_min_decode_batches = profile.runtime.rowgroup_prefetch_min_decode_batches;
	options.enable_planless_execution            = profile.runtime.enable_planless_execution;
	options.scheduling_policy                    = profile.runtime.scheduling_policy;
	options.transform_blocks_per_launch          = profile.runtime.transform_blocks_per_launch;
	options.transform_ctas_per_launch            = profile.runtime.transform_ctas_per_launch;
	options.use_low_priority_streams             = profile.runtime.use_low_priority_streams;
	options.async_planless_completion             = profile.runtime.async_planless_completion;
	options.block_major_double_buffer_policy      = profile.runtime.block_major_double_buffer_policy;
	options.crop_execution_mode                   = profile.runtime.crop_execution_mode;
	options.bounded_read_amplification_ppm        = profile.runtime.bounded_read_amplification_ppm;
	options.bounded_read_local_amplification_ppm  = profile.runtime.bounded_read_local_amplification_ppm;
	options.bounded_read_max_run_bytes            = profile.runtime.bounded_read_max_run_bytes;
	return options;
}

inline constexpr std::string_view kCompactV3RuntimePolicyId =
    "compact-v3-planless-limited-o512-c512-v1";

inline DirectDctRuntimePolicy compact_v3_runtime_policy() {
	DirectDctRuntimePolicy policy;
	policy.id                                   = kCompactV3RuntimePolicyId;
	policy.cache_capacity_bytes                 = 0U;
	policy.decode_batch_rowgroups               = 64U;
	policy.decode_workset_capacity_bytes         = jpeg::kDefaultJpegDctDeviceDecodeWorksetCapacityBytes;
	policy.plan_cache_capacity                  = 0U;
	policy.enable_rowgroup_prefetch             = true;
	policy.rowgroup_prefetch_depth              = 16U;
	policy.rowgroup_prefetch_workers            = 8U;
	policy.rowgroup_prefetch_min_decode_batches = 1U;
	policy.enable_planless_execution            = true;
	policy.scheduling_policy                    = jpeg::JpegDctSchedulingPolicy::kLimitedOverlap;
	policy.transform_blocks_per_launch          = 512U;
	policy.transform_ctas_per_launch            = 512U;
	policy.use_low_priority_streams             = true;
	policy.async_planless_completion             = true;
	return policy;
}

inline constexpr std::string_view kBlockMajorScheduledBoundedRuntimePolicyId =
    "block-major-p4-scheduled-bounded-110-v1";

inline DirectDctRuntimePolicy block_major_scheduled_bounded_runtime_policy() {
	auto policy = compact_v3_runtime_policy();
	policy.id = kBlockMajorScheduledBoundedRuntimePolicyId;
	policy.crop_execution_mode =
	    jpeg::JpegDctCropExecutionMode::kBoundedIoUringScheduledRangeReadSelectedDecode;
	policy.bounded_read_amplification_ppm       = 1'100'000U;
	policy.bounded_read_local_amplification_ppm = 0U;
	policy.bounded_read_max_run_bytes           = 0U;
	return policy;
}

} // namespace galp::profiles

#endif // GALP_PROFILES_DIRECT_DCT_HPP
