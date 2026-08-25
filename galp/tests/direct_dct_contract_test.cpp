#include "direct_dct/logical_types.hpp"
#include "direct_dct/profile_registry.hpp"
#include "direct_dct/resolved_execution_policy.hpp"
#include "galp/profiles/registry.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using galp::direct_dct::LogicalBatchRequest;
using galp::direct_dct::ResolvedExecutionPolicy;
using galp::direct_dct::SemanticProfileRegistry;

template <typename Legacy, typename Shadow>
void expect_named_equal(const std::string_view field, const Legacy& legacy, const Shadow& shadow) {
	SCOPED_TRACE(std::string(field));
	EXPECT_EQ(legacy, shadow);
}

std::string_view option_field_name(const ResolvedExecutionPolicy::BatchOptionField field) {
	const auto index = static_cast<size_t>(field);
	return ResolvedExecutionPolicy::kBatchOptionFieldNames.at(index);
}

template <typename Legacy, typename Shadow>
void expect_option_equal(const ResolvedExecutionPolicy::BatchOptionField field,
                         const Legacy&                                   legacy,
                         const Shadow&                                   shadow) {
	expect_named_equal(option_field_name(field), legacy, shadow);
}

void expect_sampling_ratio_equal(const galp::jpeg::JpegDctSamplingRatio& legacy,
                                 const galp::jpeg::JpegDctSamplingRatio& shadow,
                                 const size_t                            index) {
	SCOPED_TRACE("allowed_chroma_sampling_ratios[" + std::to_string(index) + "]");
	expect_named_equal("horizontal_numerator", legacy.horizontal_numerator, shadow.horizontal_numerator);
	expect_named_equal("horizontal_denominator", legacy.horizontal_denominator, shadow.horizontal_denominator);
	expect_named_equal("vertical_numerator", legacy.vertical_numerator, shadow.vertical_numerator);
	expect_named_equal("vertical_denominator", legacy.vertical_denominator, shadow.vertical_denominator);
}

void expect_transform_equal(const galp::jpeg::JpegDctGridTransformSpec& legacy,
                            const galp::jpeg::JpegDctGridTransformSpec& shadow) {
	expect_named_equal("y_output_width_blocks", legacy.y_output_width_blocks, shadow.y_output_width_blocks);
	expect_named_equal("y_output_height_blocks", legacy.y_output_height_blocks, shadow.y_output_height_blocks);
	expect_named_equal("cbcr_output_width_blocks", legacy.cbcr_output_width_blocks, shadow.cbcr_output_width_blocks);
	expect_named_equal("cbcr_output_height_blocks", legacy.cbcr_output_height_blocks, shadow.cbcr_output_height_blocks);
	expect_named_equal(
	    "crop_reference_width_blocks", legacy.crop_reference_width_blocks, shadow.crop_reference_width_blocks);
	expect_named_equal(
	    "crop_reference_height_blocks", legacy.crop_reference_height_blocks, shadow.crop_reference_height_blocks);
	expect_named_equal(
	    "crop_origin_alignment_blocks", legacy.crop_origin_alignment_blocks, shadow.crop_origin_alignment_blocks);
	expect_named_equal("chroma_crop_scale_x", legacy.chroma_crop_scale_x, shadow.chroma_crop_scale_x);
	expect_named_equal("chroma_crop_scale_y", legacy.chroma_crop_scale_y, shadow.chroma_crop_scale_y);
	expect_named_equal("clamp_min", legacy.clamp_min, shadow.clamp_min);
	expect_named_equal("clamp_max", legacy.clamp_max, shadow.clamp_max);
	expect_named_equal("output_data_type", legacy.output_data_type, shadow.output_data_type);
	expect_named_equal("output_add", legacy.output_add, shadow.output_add);
	expect_named_equal("output_scale", legacy.output_scale, shadow.output_scale);
	expect_named_equal("dequantize", legacy.dequantize, shadow.dequantize);
	expect_named_equal("require_all_coefficients", legacy.require_all_coefficients, shadow.require_all_coefficients);
	expect_named_equal("allow_grayscale", legacy.allow_grayscale, shadow.allow_grayscale);
	expect_named_equal("preferred_small_crop_width_blocks",
	                   legacy.preferred_small_crop_width_blocks,
	                   shadow.preferred_small_crop_width_blocks);
	expect_named_equal("preferred_small_crop_height_blocks",
	                   legacy.preferred_small_crop_height_blocks,
	                   shadow.preferred_small_crop_height_blocks);
	expect_named_equal("allowed_chroma_sampling_ratios.size",
	                   legacy.allowed_chroma_sampling_ratios.size(),
	                   shadow.allowed_chroma_sampling_ratios.size());
	const auto common_size =
	    std::min(legacy.allowed_chroma_sampling_ratios.size(), shadow.allowed_chroma_sampling_ratios.size());
	for (size_t index = 0U; index < common_size; ++index) {
		expect_sampling_ratio_equal(
		    legacy.allowed_chroma_sampling_ratios[index], shadow.allowed_chroma_sampling_ratios[index], index);
	}
}

void expect_semantic_equal(const galp::profiles::DirectDctOutputProfile&   legacy,
                           const SemanticProfileRegistry::SemanticProfile& shadow) {
	expect_named_equal("profile.id", legacy.id, shadow.id);
	expect_named_equal("profile.output_layout", legacy.layout, shadow.output_layout);
	expect_named_equal("profile.coefficient_selection",
	                   legacy.coefficient_selection.coefficients,
	                   shadow.coefficient_selection.coefficients);
	expect_named_equal(
	    "profile.grid_transform.has_value", legacy.grid_transform.has_value(), shadow.grid_transform.has_value());
	if (legacy.grid_transform.has_value() && shadow.grid_transform.has_value()) {
		expect_transform_equal(*legacy.grid_transform, *shadow.grid_transform);
	}
}

void expect_batch_options_equal(const galp::jpeg::JpegDctDeviceBatchOptions& legacy,
                                const galp::jpeg::JpegDctDeviceBatchOptions& shadow) {
	using Field = ResolvedExecutionPolicy::BatchOptionField;
	expect_option_equal(Field::kLayout, legacy.layout, shadow.layout);
	expect_option_equal(Field::kGridTransform, legacy.grid_transform.has_value(), shadow.grid_transform.has_value());
	if (legacy.grid_transform.has_value() && shadow.grid_transform.has_value()) {
		SCOPED_TRACE(std::string(option_field_name(Field::kGridTransform)));
		expect_transform_equal(*legacy.grid_transform, *shadow.grid_transform);
	}
	expect_option_equal(Field::kCacheCapacityBytes, legacy.cache_capacity_bytes, shadow.cache_capacity_bytes);
	expect_option_equal(Field::kDecodeBatchRowgroups, legacy.decode_batch_rowgroups, shadow.decode_batch_rowgroups);
	expect_option_equal(Field::kPlanCacheCapacity, legacy.plan_cache_capacity, shadow.plan_cache_capacity);
	expect_option_equal(
	    Field::kEnableRowgroupPrefetch, legacy.enable_rowgroup_prefetch, shadow.enable_rowgroup_prefetch);
	expect_option_equal(Field::kRowgroupPrefetchDepth, legacy.rowgroup_prefetch_depth, shadow.rowgroup_prefetch_depth);
	expect_option_equal(
	    Field::kRowgroupPrefetchWorkers, legacy.rowgroup_prefetch_workers, shadow.rowgroup_prefetch_workers);
	expect_option_equal(Field::kRowgroupPrefetchMinDecodeBatches,
	                    legacy.rowgroup_prefetch_min_decode_batches,
	                    shadow.rowgroup_prefetch_min_decode_batches);
	expect_option_equal(Field::kCoefficientSelection,
	                    legacy.coefficient_selection.coefficients,
	                    shadow.coefficient_selection.coefficients);
	expect_option_equal(
	    Field::kEnablePlanlessExecution, legacy.enable_planless_execution, shadow.enable_planless_execution);
	expect_option_equal(Field::kSchedulingPolicy, legacy.scheduling_policy, shadow.scheduling_policy);
	expect_option_equal(
	    Field::kTransformBlocksPerLaunch, legacy.transform_blocks_per_launch, shadow.transform_blocks_per_launch);
	expect_option_equal(
	    Field::kTransformCtasPerLaunch, legacy.transform_ctas_per_launch, shadow.transform_ctas_per_launch);
	expect_option_equal(
	    Field::kUseLowPriorityStreams, legacy.use_low_priority_streams, shadow.use_low_priority_streams);
	expect_option_equal(
	    Field::kAsyncPlanlessCompletion, legacy.async_planless_completion, shadow.async_planless_completion);
	expect_option_equal(Field::kTransformSubmissionGate,
	                    legacy.transform_submission_gate != nullptr,
	                    shadow.transform_submission_gate != nullptr);
	expect_option_equal(Field::kBlockMajorDoubleBufferPolicy,
	                    legacy.block_major_double_buffer_policy,
	                    shadow.block_major_double_buffer_policy);
	expect_option_equal(Field::kCropExecutionMode, legacy.crop_execution_mode, shadow.crop_execution_mode);
	expect_option_equal(
	    Field::kDecodeWorksetCapacityBytes, legacy.decode_workset_capacity_bytes, shadow.decode_workset_capacity_bytes);
	expect_option_equal(Field::kBoundedReadAmplificationPpm,
	                    legacy.bounded_read_amplification_ppm,
	                    shadow.bounded_read_amplification_ppm);
	expect_option_equal(Field::kBoundedReadLocalAmplificationPpm,
	                    legacy.bounded_read_local_amplification_ppm,
	                    shadow.bounded_read_local_amplification_ppm);
	expect_option_equal(
	    Field::kBoundedReadMaxRunBytes, legacy.bounded_read_max_run_bytes, shadow.bounded_read_max_run_bytes);
}

std::vector<galp::jpeg::JpegDctImageCropRequest> legacy_requests(const size_t count) {
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	requests.reserve(count);
	for (size_t index = 0U; index < count; ++index) {
		galp::jpeg::JpegDctImageCropRequest request;
		request.global_image_index = static_cast<uint32_t>(10U + index);
		requests.push_back(std::move(request));
	}
	return requests;
}

TEST(DirectDctShadowContract, ProfileIdsMatchLegacy) {
	const auto  legacy = galp::profiles::available_direct_dct_profile_ids();
	const auto& shadow = SemanticProfileRegistry::available_profile_ids();
	ASSERT_EQ(legacy.size(), shadow.size());
	for (size_t index = 0U; index < legacy.size(); ++index) {
		SCOPED_TRACE(index);
		EXPECT_EQ(legacy[index], shadow[index]);
	}
}

TEST(DirectDctShadowContract, SemanticGeometryAndTransformsMatchLegacyFieldByField) {
	for (const auto profile_id : SemanticProfileRegistry::available_profile_ids()) {
		SCOPED_TRACE(std::string(profile_id));
		const auto legacy = galp::profiles::resolve_direct_dct_profile(profile_id);
		const auto shadow = SemanticProfileRegistry::resolve(profile_id);
		expect_semantic_equal(legacy.output, shadow);
	}
}

TEST(DirectDctShadowContract, ExecutionPolicyMatchesLegacyMaterializationFieldByField) {
	static_assert(!std::is_copy_assignable_v<ResolvedExecutionPolicy>);
	static_assert(ResolvedExecutionPolicy::kBatchOptionFieldCount == 23U);
	const std::set<std::string_view> inventory(ResolvedExecutionPolicy::kBatchOptionFieldNames.begin(),
	                                           ResolvedExecutionPolicy::kBatchOptionFieldNames.end());
	ASSERT_EQ(inventory.size(), ResolvedExecutionPolicy::kBatchOptionFieldCount);

	for (const auto profile_id : SemanticProfileRegistry::available_profile_ids()) {
		SCOPED_TRACE(std::string(profile_id));
		const auto legacy_profile  = galp::profiles::resolve_direct_dct_profile(profile_id);
		const auto legacy_options  = galp::profiles::materialize_direct_dct_options(legacy_profile);
		const auto shadow_semantic = SemanticProfileRegistry::resolve(profile_id);
		const auto shadow_policy   = galp::direct_dct::resolve_execution_policy(profile_id);
		const auto shadow_options  = galp::direct_dct::materialize_shadow_options(shadow_semantic, shadow_policy);

		expect_named_equal("source_profile_id", legacy_profile.id, shadow_policy.source_profile_id);
		expect_named_equal("runtime_policy_id", legacy_profile.runtime.id, shadow_policy.runtime_policy_id);
		expect_named_equal("resolved.layout", legacy_options.layout, shadow_policy.layout);
		expect_batch_options_equal(legacy_options, shadow_options);
		EXPECT_EQ(shadow_policy.submission_gate_policy,
		          ResolvedExecutionPolicy::SubmissionGatePolicy::kCreatePerAsyncBatch);
	}
}

TEST(LogicalBatchRequestContract, EmptyBatchIsRejected) {
	LogicalBatchRequest request;
	request.semantic_profile_id = "rgbnomore-validation-v1";
	request.logical_batch_size  = 4U;
	request.partial_tail        = true;
	EXPECT_THROW(galp::direct_dct::validate_logical_batch_request(request), std::invalid_argument);
}

TEST(LogicalBatchRequestContract, SingleSamplePreservesOneLogicalBoundary) {
	const auto legacy = legacy_requests(1U);
	const auto request =
	    galp::direct_dct::shadow_convert_legacy_requests(legacy, 1U, "rgbnomore-validation-v1", 100U, 3U);
	ASSERT_EQ(request.samples.size(), 1U);
	EXPECT_EQ(request.request_identity, 100U);
	EXPECT_EQ(request.batch_ordinal, 3U);
	EXPECT_EQ(request.semantic_profile_id, "rgbnomore-validation-v1");
	EXPECT_EQ(request.logical_batch_size, 1U);
	EXPECT_FALSE(request.partial_tail);
	EXPECT_EQ(request.samples[0].image_id, 10U);
}

TEST(LogicalBatchRequestContract, MultiSamplePreservesOrderAndBoundary) {
	const auto legacy  = legacy_requests(3U);
	const auto request = galp::direct_dct::shadow_convert_legacy_requests(legacy, 3U, "rgbnomore-validation-v1");
	ASSERT_EQ(request.samples.size(), 3U);
	EXPECT_EQ(request.samples[0].image_id, 10U);
	EXPECT_EQ(request.samples[1].image_id, 11U);
	EXPECT_EQ(request.samples[2].image_id, 12U);
	EXPECT_FALSE(request.partial_tail);
}

TEST(LogicalBatchRequestContract, PartialTailIsExplicit) {
	const auto legacy  = legacy_requests(2U);
	const auto request = galp::direct_dct::shadow_convert_legacy_requests(legacy, 4U, "rgbnomore-validation-v1");
	EXPECT_EQ(request.logical_batch_size, 4U);
	EXPECT_EQ(request.samples.size(), 2U);
	EXPECT_TRUE(request.partial_tail);
}

TEST(LogicalBatchRequestContract, SemanticTransformAndMetadataArePreserved) {
	auto legacy                 = legacy_requests(1U);
	legacy[0].source_crop       = {8U, 16U, 224U, 224U};
	legacy[0].horizontal_flip   = true;
	legacy[0].logical_sample_id = "sample-10";
	legacy[0].augmentation_key  = "crop-8-16-flip";
	const auto request = galp::direct_dct::shadow_convert_legacy_requests(legacy, 1U, "rgbnomore-validation-v1");
	ASSERT_TRUE(request.samples[0].transform.source_crop.has_value());
	const auto& crop = *request.samples[0].transform.source_crop;
	EXPECT_EQ(crop.x, 8U);
	EXPECT_EQ(crop.y, 16U);
	EXPECT_EQ(crop.width, 224U);
	EXPECT_EQ(crop.height, 224U);
	EXPECT_TRUE(request.samples[0].transform.horizontal_flip);
	EXPECT_EQ(request.samples[0].transform.logical_sample_id, "sample-10");
	EXPECT_EQ(request.samples[0].transform.augmentation_key, "crop-8-16-flip");
}

TEST(LogicalBatchRequestContract, InvalidSemanticCropAndBoundaryAreRejected) {
	auto invalid_crop           = legacy_requests(1U);
	invalid_crop[0].source_crop = {8U, 16U, 0U, 224U};
	EXPECT_THROW(galp::direct_dct::shadow_convert_legacy_requests(invalid_crop, 1U, "rgbnomore-validation-v1"),
	             std::invalid_argument);

	LogicalBatchRequest invalid_boundary;
	invalid_boundary.semantic_profile_id = "rgbnomore-validation-v1";
	invalid_boundary.logical_batch_size  = 2U;
	invalid_boundary.partial_tail        = false;
	invalid_boundary.samples.push_back(LogicalBatchRequest::Sample {});
	EXPECT_THROW(galp::direct_dct::validate_logical_batch_request(invalid_boundary), std::invalid_argument);

	invalid_boundary.semantic_profile_id.clear();
	EXPECT_THROW(galp::direct_dct::validate_logical_batch_request(invalid_boundary), std::invalid_argument);
}

TEST(PipelineTraceEventContract, SchemaCarriesDifferentialOrdinalsWithoutRuntimeHooks) {
	using Event = galp::direct_dct::PipelineTraceEvent;
	static_assert(std::is_trivially_copyable_v<Event>);
	Event event;
	event.request_identity   = 901U;
	event.request_ordinal    = 6U;
	event.batch_ordinal      = 7U;
	event.stage              = Event::Stage::kCompleted;
	event.plan_identity_hash = 0xabcU;
	event.io_identity_hash   = 0xdefU;
	event.prepare_ordinal    = 11U;
	event.stage_ordinal      = 21U;
	event.read_ordinal       = 31U;
	event.submission_ordinal = 41U;
	event.completion_ordinal = 40U;
	EXPECT_EQ(event.request_identity, 901U);
	EXPECT_EQ(event.request_ordinal, 6U);
	EXPECT_EQ(event.batch_ordinal, 7U);
	EXPECT_EQ(event.stage, Event::Stage::kCompleted);
	EXPECT_EQ(event.plan_identity_hash, 0xabcU);
	EXPECT_EQ(event.io_identity_hash, 0xdefU);
	EXPECT_EQ(event.prepare_ordinal, 11U);
	EXPECT_EQ(event.stage_ordinal, 21U);
	EXPECT_EQ(event.read_ordinal, 31U);
	EXPECT_EQ(event.submission_ordinal, 41U);
	EXPECT_EQ(event.completion_ordinal, 40U);
}

TEST(DirectDctShadowContract, UnknownAndMismatchedProfilesAreRejected) {
	EXPECT_THROW(static_cast<void>(SemanticProfileRegistry::resolve("unknown")), std::invalid_argument);
	EXPECT_THROW(static_cast<void>(galp::direct_dct::resolve_execution_policy("unknown")), std::invalid_argument);
	const auto semantic = SemanticProfileRegistry::resolve(SemanticProfileRegistry::kProfileIds[0]);
	const auto policy   = galp::direct_dct::resolve_execution_policy(SemanticProfileRegistry::kProfileIds[1]);
	EXPECT_THROW(static_cast<void>(galp::direct_dct::materialize_shadow_options(semantic, policy)),
	             std::invalid_argument);
}

} // namespace
