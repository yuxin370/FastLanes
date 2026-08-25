#ifndef GALP_DIRECT_DCT_RESOLVED_EXECUTION_POLICY_HPP
#define GALP_DIRECT_DCT_RESOLVED_EXECUTION_POLICY_HPP

#include "direct_dct/profile_registry.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace galp::direct_dct {

// Immutable shadow representation of the execution values materialized by the
// current production profile resolver. Phase 1 does not make this authoritative.
struct ResolvedExecutionPolicy final {
	enum class SubmissionGatePolicy : uint8_t {
		kDisabled,
		kCreatePerAsyncBatch,
	};

	// Explicit completeness inventory for every field in
	// jpeg::JpegDctDeviceBatchOptions. Contract tests compare all 23 fields.
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

// Differential-test materializer only. It reconstructs the legacy option
// aggregate without installing a submission gate or entering the runtime path.
[[nodiscard]] jpeg::JpegDctDeviceBatchOptions
materialize_shadow_options(const SemanticProfileRegistry::SemanticProfile& semantic,
                           const ResolvedExecutionPolicy&                  policy);

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_RESOLVED_EXECUTION_POLICY_HPP
