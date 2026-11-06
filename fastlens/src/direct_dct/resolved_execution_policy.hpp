#ifndef FASTLENS_DIRECT_DCT_RESOLVED_EXECUTION_POLICY_HPP
#define FASTLENS_DIRECT_DCT_RESOLVED_EXECUTION_POLICY_HPP

#include "direct_dct/profile_registry.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace fastlens::direct_dct {

// Immutable execution settings for a semantic profile.
struct ResolvedExecutionPolicy final {
	enum class SubmissionGatePolicy : uint8_t {
		kDisabled,
		kCreatePerAsyncBatch,
	};

	// Field names used to compare jpeg::JpegDctDeviceBatchOptions in tests.
	enum class BatchOptionField : size_t {
		kLayout,
		kGridTransform,
		kCacheCapacityBytes,
		kDecodeBatchRowgroups,
		kPlanCacheCapacity,
		kEnableRowgroupPrefetch,
		kRowgroupPrefetchDepth,
		kRowgroupPrefetchWorkers,
		kRowgroupPrefetchMinDecodeBatches,
		kCoefficientSelection,
		kEnablePlanlessExecution,
		kSchedulingPolicy,
		kTransformBlocksPerLaunch,
		kTransformCtasPerLaunch,
		kUseLowPriorityStreams,
		kAsyncPlanlessCompletion,
		kTransformSubmissionGate,
		kBlockMajorDoubleBufferPolicy,
		kCropExecutionMode,
		kDecodeWorksetCapacityBytes,
		kBoundedReadAmplificationPpm,
		kBoundedReadLocalAmplificationPpm,
		kBoundedReadMaxRunBytes,
		kCount,
	};

	inline static constexpr size_t kBatchOptionFieldCount = static_cast<size_t>(BatchOptionField::kCount);
	inline static constexpr std::array<std::string_view, kBatchOptionFieldCount> kBatchOptionFieldNames = {
	    "layout",
	    "grid_transform",
	    "cache_capacity_bytes",
	    "decode_batch_rowgroups",
	    "plan_cache_capacity",
	    "enable_rowgroup_prefetch",
	    "rowgroup_prefetch_depth",
	    "rowgroup_prefetch_workers",
	    "rowgroup_prefetch_min_decode_batches",
	    "coefficient_selection",
	    "enable_planless_execution",
	    "scheduling_policy",
	    "transform_blocks_per_launch",
	    "transform_ctas_per_launch",
	    "use_low_priority_streams",
	    "async_planless_completion",
	    "transform_submission_gate",
	    "block_major_double_buffer_policy",
	    "crop_execution_mode",
	    "decode_workset_capacity_bytes",
	    "bounded_read_amplification_ppm",
	    "bounded_read_local_amplification_ppm",
	    "bounded_read_max_run_bytes",
	};

	const std::string_view                          source_profile_id;
	const std::string_view                          runtime_policy_id;
	const jpeg::JpegDctDeviceLayout                 layout;
	const size_t                                    cache_capacity_bytes;
	const size_t                                    decode_batch_rowgroups;
	const size_t                                    decode_workset_capacity_bytes;
	const size_t                                    plan_cache_capacity;
	const bool                                      enable_rowgroup_prefetch;
	const size_t                                    rowgroup_prefetch_depth;
	const size_t                                    rowgroup_prefetch_workers;
	const size_t                                    rowgroup_prefetch_min_decode_batches;
	const bool                                      enable_planless_execution;
	const jpeg::JpegDctSchedulingPolicy             scheduling_policy;
	const size_t                                    transform_blocks_per_launch;
	const size_t                                    transform_ctas_per_launch;
	const bool                                      use_low_priority_streams;
	const bool                                      async_planless_completion;
	const SubmissionGatePolicy                      submission_gate_policy;
	const jpeg::JpegDctBlockMajorDoubleBufferPolicy block_major_double_buffer_policy;
	const jpeg::JpegDctCropExecutionMode            crop_execution_mode;
	const uint32_t                                  bounded_read_amplification_ppm;
	const uint32_t                                  bounded_read_local_amplification_ppm;
	const size_t                                    bounded_read_max_run_bytes;
};

static_assert(ResolvedExecutionPolicy::kBatchOptionFieldNames.size() ==
              ResolvedExecutionPolicy::kBatchOptionFieldCount);

[[nodiscard]] ResolvedExecutionPolicy resolve_execution_policy(std::string_view semantic_profile_id);

// Combine semantic and execution settings without starting runtime work.
[[nodiscard]] jpeg::JpegDctDeviceBatchOptions
materialize_shadow_options(const SemanticProfileRegistry::SemanticProfile& semantic,
                           const ResolvedExecutionPolicy&                  policy);

} // namespace fastlens::direct_dct

#endif // FASTLENS_DIRECT_DCT_RESOLVED_EXECUTION_POLICY_HPP
