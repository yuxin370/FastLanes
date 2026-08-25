#include "direct_dct/native_logical_batch_pipeline.hpp"
#include "direct_dct/native_logical_batch_pipeline_detail.hpp"
#include "direct_dct/profile_registry.hpp"
#include "direct_dct/resolved_execution_policy.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace galp::direct_dct {
namespace {

class StableFingerprint final {
public:
	template <typename T>
	void integral(const T value) {
		static_assert(std::is_integral_v<T>);
		if constexpr (std::is_same_v<T, bool>) {
			byte(value ? uint8_t {1U} : uint8_t {0U});
		} else {
			using Unsigned = std::make_unsigned_t<T>;
			auto bits      = static_cast<Unsigned>(value);
			for (size_t index = 0U; index < sizeof(Unsigned); ++index) {
				byte(static_cast<uint8_t>((bits >> (index * 8U)) & static_cast<Unsigned>(0xffU)));
			}
		}
	}

	template <typename Enum>
	void enumeration(const Enum value) {
		static_assert(std::is_enum_v<Enum>);
		integral(static_cast<std::underlying_type_t<Enum>>(value));
	}

	void floating(const float value) {
		integral(std::bit_cast<uint32_t>(value));
	}

	void floating(const double value) {
		integral(std::bit_cast<uint64_t>(value));
	}

	void string(const std::string_view value) {
		integral(value.size());
		for (const char character : value) {
			byte(static_cast<uint8_t>(character));
		}
	}

	[[nodiscard]] uint64_t value() const noexcept {
		return value_;
	}

private:
	void byte(const uint8_t value) noexcept {
		value_ ^= value;
		value_ *= 1099511628211ULL;
	}

	uint64_t value_ = 14695981039346656037ULL;
};

void hash_crop(StableFingerprint& hash, const jpeg::JpegDctCropBox& crop) {
	hash.integral(crop.x);
	hash.integral(crop.y);
	hash.integral(crop.width);
	hash.integral(crop.height);
}

void hash_requests(StableFingerprint& hash, const std::vector<jpeg::JpegDctImageCropRequest>& requests) {
	hash.integral(requests.size());
	for (const auto& request : requests) {
		hash.integral(request.global_image_index);
		hash_crop(hash, request.source_crop);
		hash.integral(request.horizontal_flip);
		hash.string(request.logical_sample_id);
		hash.string(request.augmentation_key);
	}
}

void hash_transform(StableFingerprint& hash, const jpeg::JpegDctGridTransformSpec& transform) {
	hash.integral(transform.y_output_width_blocks);
	hash.integral(transform.y_output_height_blocks);
	hash.integral(transform.cbcr_output_width_blocks);
	hash.integral(transform.cbcr_output_height_blocks);
	hash.integral(transform.crop_reference_width_blocks);
	hash.integral(transform.crop_reference_height_blocks);
	hash.integral(transform.crop_origin_alignment_blocks);
	hash.integral(transform.chroma_crop_scale_x);
	hash.integral(transform.chroma_crop_scale_y);
	hash.integral(transform.clamp_min);
	hash.integral(transform.clamp_max);
	hash.enumeration(transform.output_data_type);
	hash.floating(transform.output_add);
	hash.floating(transform.output_scale);
	hash.integral(transform.dequantize);
	hash.integral(transform.require_all_coefficients);
	hash.integral(transform.allow_grayscale);
	hash.integral(transform.preferred_small_crop_width_blocks.size());
	for (const auto value : transform.preferred_small_crop_width_blocks) {
		hash.integral(value);
	}
	hash.integral(transform.preferred_small_crop_height_blocks.size());
	for (const auto value : transform.preferred_small_crop_height_blocks) {
		hash.integral(value);
	}
	hash.integral(transform.allowed_chroma_sampling_ratios.size());
	for (const auto& ratio : transform.allowed_chroma_sampling_ratios) {
		hash.integral(ratio.horizontal_numerator);
		hash.integral(ratio.horizontal_denominator);
		hash.integral(ratio.vertical_numerator);
		hash.integral(ratio.vertical_denominator);
	}
}

void hash_options(StableFingerprint& hash, const jpeg::JpegDctDeviceBatchOptions& options) {
	hash.enumeration(options.layout);
	hash.integral(options.grid_transform.has_value());
	if (options.grid_transform.has_value()) {
		hash_transform(hash, *options.grid_transform);
	}
	hash.integral(options.cache_capacity_bytes);
	hash.integral(options.decode_batch_rowgroups);
	hash.integral(options.plan_cache_capacity);
	hash.integral(options.enable_rowgroup_prefetch);
	hash.integral(options.rowgroup_prefetch_depth);
	hash.integral(options.rowgroup_prefetch_workers);
	hash.integral(options.rowgroup_prefetch_min_decode_batches);
	hash.integral(options.coefficient_selection.coefficients.size());
	for (const auto coefficient : options.coefficient_selection.coefficients) {
		hash.integral(coefficient);
	}
	hash.integral(options.enable_planless_execution);
	hash.enumeration(options.scheduling_policy);
	hash.integral(options.transform_blocks_per_launch);
	hash.integral(options.transform_ctas_per_launch);
	hash.integral(options.use_low_priority_streams);
	hash.integral(options.async_planless_completion);
	hash.integral(options.transform_submission_gate != nullptr);
	hash.enumeration(options.block_major_double_buffer_policy);
	hash.enumeration(options.crop_execution_mode);
	hash.integral(options.decode_workset_capacity_bytes);
	hash.integral(options.bounded_read_amplification_ppm);
	hash.integral(options.bounded_read_local_amplification_ppm);
	hash.integral(options.bounded_read_max_run_bytes);
}

void hash_rowgroup(StableFingerprint& hash, const jpeg::JpegDctDeviceRowgroupMetadata& rowgroup) {
	hash.integral(rowgroup.shard_id);
	hash.integral(rowgroup.rowgroup_index);
}

void hash_io_semantics(StableFingerprint& hash, const jpeg::JpegDctDeviceBatchPlanPreview& preview) {
	hash.integral(preview.rowgroups.size());
	for (const auto& rowgroup : preview.rowgroups) {
		hash_rowgroup(hash, rowgroup);
	}
	hash.integral(preview.rowgroup_vector_plans.size());
	for (const auto& vector_plan : preview.rowgroup_vector_plans) {
		hash_rowgroup(hash, vector_plan.rowgroup);
		hash.integral(vector_plan.full_vector_count);
		hash.integral(vector_plan.selected_vectors.size());
		for (const auto selected : vector_plan.selected_vectors) {
			hash.integral(selected);
		}
	}
	hash.integral(preview.decode_workset_capacity_bytes);
	hash.integral(preview.estimated_max_decode_workset_bytes);
	hash.integral(preview.estimated_oversized_decode_rowgroups);
}

void hash_plan_semantics(StableFingerprint& hash, const jpeg::JpegDctDeviceBatchPlanPreview& preview) {
	// Deliberately exclude wall-clock timings and cache hit/miss counters: they
	// describe observation history, not the deterministic physical plan.
	hash.enumeration(preview.layout);
	hash.integral(preview.image_layouts.size());
	for (const auto& layout : preview.image_layouts) {
		hash.integral(layout.global_image_index);
		hash.integral(layout.block_offset);
		hash.integral(layout.block_count);
	}
	hash.integral(preview.block_metadata.size());
	for (const auto& block : preview.block_metadata) {
		hash.integral(block.request_index);
		hash.integral(block.global_image_index);
		hash.integral(block.semantic_slot_id);
		hash.integral(block.block_x);
		hash.integral(block.block_y);
	}
	hash_io_semantics(hash, preview);
	hash.integral(preview.planned_selected_vector_count);
	hash.integral(preview.estimated_selected_vector_count);
	hash.integral(preview.full_vector_count);
	hash.integral(preview.planned_saved_vector_count);
	hash.integral(preview.estimated_saved_vector_count);
	hash.integral(preview.selected_coefficients.size());
	for (const auto coefficient : preview.selected_coefficients) {
		hash.integral(coefficient);
	}
	hash.integral(preview.coefficients_per_block);
	hash.floating(preview.planned_selected_vector_ratio);
	hash.floating(preview.estimated_selected_vector_ratio);
	for (const auto dimension : preview.ycbcr_dct_grid_shape.y) {
		hash.integral(dimension);
	}
	for (const auto dimension : preview.ycbcr_dct_grid_shape.cbcr) {
		hash.integral(dimension);
	}
	hash.integral(preview.uses_planless_fixed_transform);
	hash.integral(preview.compact_image_descriptor_count);
	hash.integral(preview.fixed_transform_component_count);
	hash.integral(preview.fixed_transform_source_block_count);
	hash.integral(preview.fixed_transform_output_block_count);
	hash.integral(preview.host_expanded_transform_items_created);
	hash.integral(preview.host_output_block_source_lists_created);
	hash.integral(preview.host_global_transform_sort_items);
	hash.integral(preview.planless_axis_program_count);
	hash.integral(preview.planless_axis_phase_matrix_count);
	hash.integral(preview.planless_axis_program_bytes);
	hash.integral(preview.compact_plan_bytes);
	hash.integral(preview.compact_plan_peak_bytes);
	hash.integral(preview.coordinate_group_lookup_count);
	hash.integral(preview.coordinate_group_index_entries);
	hash.integral(preview.coordinate_group_index_populated);
	hash.integral(preview.coordinate_group_index_holes);
	hash.integral(preview.coordinate_group_index_bytes);
	hash.floating(preview.coordinate_group_index_density);
	hash.integral(preview.exact_batch_plan_cache_enabled);
	hash.integral(preview.compact_reader_image_locator_bytes);
	hash.integral(preview.compact_reader_shard_index_bytes);
	hash.integral(preview.compact_reader_layout_dictionary_bytes);
	hash.integral(preview.compact_reader_quant_table_dictionary_bytes);
	hash.integral(preview.compact_reader_total_bytes);
	hash.integral(preview.compact_reader_shard_descriptor_bytes);
	hash.integral(preview.compact_reader_shard_index_derived);
	hash.integral(preview.decode_workset_capacity_bytes);
	hash.integral(preview.estimated_max_decode_workset_bytes);
	hash.integral(preview.estimated_oversized_decode_rowgroups);
	hash.integral(preview.planless_axis_program_capacity_contract_bytes);
	hash.integral(preview.planless_axis_program_capacity_contract_complete);
}

void check_cuda(const cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
	}
}

class DirectDctRuntimeAdapter final {
public:
	using PreparedBatch = jpeg::DirectDctPreparedBatch;
	using Batch         = jpeg::DirectDctBatch;

	explicit DirectDctRuntimeAdapter(const std::filesystem::path& manifest_path)
	    : runtime_(std::make_shared<jpeg::DirectDctRuntime>(manifest_path)) {
	}

	explicit DirectDctRuntimeAdapter(std::shared_ptr<jpeg::DirectDctRuntime> runtime)
	    : runtime_(std::move(runtime)) {
		if (!runtime_) {
			throw std::invalid_argument("NativeLogicalBatchPipeline runtime must not be null");
		}
	}

	DirectDctRuntimeAdapter(const DirectDctRuntimeAdapter&)                = delete;
	DirectDctRuntimeAdapter& operator=(const DirectDctRuntimeAdapter&)     = delete;
	DirectDctRuntimeAdapter(DirectDctRuntimeAdapter&&) noexcept            = default;
	DirectDctRuntimeAdapter& operator=(DirectDctRuntimeAdapter&&) noexcept = default;

	[[nodiscard]] int capture_device() const {
		int device_index = 0;
		check_cuda(cudaGetDevice(&device_index), "cudaGetDevice");
		return device_index;
	}

	void activate_device(const int device_index) {
		check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
	}

	[[nodiscard]] detail::NativePlanTraceIdentity
	trace_identity(const std::vector<jpeg::JpegDctImageCropRequest>& requests,
	               const jpeg::JpegDctDeviceBatchOptions&            options) {
		const auto        preview = runtime_->PlanBatch(requests, options);
		StableFingerprint plan;
		hash_requests(plan, requests);
		hash_options(plan, options);
		hash_plan_semantics(plan, preview);
		StableFingerprint io;
		hash_options(io, options);
		hash_io_semantics(io, preview);
		return {plan.value(), io.value()};
	}

	[[nodiscard]] PreparedBatch prepare(const std::vector<jpeg::JpegDctImageCropRequest>& requests,
	                                    const jpeg::JpegDctDeviceBatchOptions&            options) {
		return runtime_->PrepareBatch(requests, options);
	}

	void stage(PreparedBatch& prepared) {
		runtime_->StageBatchIo(prepared);
	}

	[[nodiscard]] Batch read(PreparedBatch prepared) {
		// ReadPreparedBatch deliberately performs the same second StageBatchIo
		// call as the legacy Torch prefetch path. Phase 2 does not normalize it.
		return runtime_->ReadPreparedBatch(std::move(prepared));
	}

private:
	std::shared_ptr<jpeg::DirectDctRuntime> runtime_;
};

jpeg::JpegDctDeviceBatchOptions resolved_shadow_options(const std::string_view profile_id) {
	const auto semantic = SemanticProfileRegistry::resolve(profile_id);
	const auto policy   = resolve_execution_policy(profile_id);
	return materialize_shadow_options(semantic, policy);
}

} // namespace

struct NativeLogicalBatchPipeline::Impl final {
	Impl(const std::filesystem::path& manifest_path,
	     const std::string_view       semantic_profile_id,
	     NativePipelineTraceBuffer*   trace)
	    : core(DirectDctRuntimeAdapter(manifest_path),
	           std::string(semantic_profile_id),
	           resolved_shadow_options(semantic_profile_id),
	           trace) {
	}

	Impl(std::shared_ptr<jpeg::DirectDctRuntime> runtime,
	     const std::string_view                  semantic_profile_id,
	     jpeg::JpegDctDeviceBatchOptions         options,
	     NativePipelineTraceBuffer*              trace)
	    : core(DirectDctRuntimeAdapter(std::move(runtime)),
	           std::string(semantic_profile_id),
	           std::move(options),
	           trace) {
	}

	detail::NativeLogicalBatchPipelineCore<DirectDctRuntimeAdapter> core;
};

NativeLogicalBatchPipeline::NativeLogicalBatchPipeline(const std::filesystem::path& manifest_path,
                                                       const std::string_view       semantic_profile_id,
                                                       NativePipelineTraceBuffer*   trace)
    : impl_(std::make_unique<Impl>(manifest_path, semantic_profile_id, trace)) {
}

NativeLogicalBatchPipeline::NativeLogicalBatchPipeline(std::shared_ptr<jpeg::DirectDctRuntime> runtime,
                                                       const std::string_view semantic_profile_id,
                                                       jpeg::JpegDctDeviceBatchOptions options,
                                                       NativePipelineTraceBuffer* trace)
    : impl_(std::make_unique<Impl>(
          std::move(runtime), semantic_profile_id, std::move(options), trace)) {
}

NativeLogicalBatchPipeline::~NativeLogicalBatchPipeline() = default;

void NativeLogicalBatchPipeline::reset(std::vector<LogicalBatchRequest> requests) {
	impl_->core.reset(std::move(requests));
}

jpeg::DirectDctBatch NativeLogicalBatchPipeline::next() {
	return impl_->core.next();
}

bool NativeLogicalBatchPipeline::ready() const {
	return impl_->core.ready();
}

bool NativeLogicalBatchPipeline::started() const noexcept {
	return impl_->core.started();
}

size_t NativeLogicalBatchPipeline::prefetched_batch_count() const noexcept {
	return impl_->core.prefetched_batch_count();
}

NativePipelineState NativeLogicalBatchPipeline::state() const noexcept {
	return impl_->core.state();
}

NativePipelinePrefetchMetrics NativeLogicalBatchPipeline::prefetch_metrics() const noexcept {
	return impl_->core.prefetch_metrics();
}

size_t NativeLogicalBatchPipeline::close() noexcept {
	return impl_->core.close();
}

} // namespace galp::direct_dct
