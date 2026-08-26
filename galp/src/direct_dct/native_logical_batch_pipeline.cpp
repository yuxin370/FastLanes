#include "direct_dct/native_logical_batch_pipeline.hpp"
#include "direct_dct/native_logical_batch_pipeline_detail.hpp"
#include "direct_dct/profile_registry.hpp"
#include "direct_dct/resolved_execution_policy.hpp"
#include <array>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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

bool native_physical_orchestration_enabled() noexcept {
	const auto* value = std::getenv("GALP_PHASE6_NATIVE_PHYSICAL");
	return value != nullptr && std::string_view(value) != "0";
}

jpeg::JpegDctImageCropRequest lower_sample_copy(const LogicalBatchRequest::Sample& sample) {
	jpeg::JpegDctImageCropRequest request;
	request.global_image_index = sample.image_id;
	if (sample.transform.source_crop.has_value()) {
		request.source_crop = *sample.transform.source_crop;
	}
	request.horizontal_flip   = sample.transform.horizontal_flip;
	request.logical_sample_id = sample.transform.logical_sample_id;
	request.augmentation_key  = sample.transform.augmentation_key;
	return request;
}

void add_prefetch_metrics(NativePipelinePrefetchMetrics& destination,
	                       const NativePipelinePrefetchMetrics& source) noexcept {
	destination.producer_active_nanoseconds += source.producer_active_nanoseconds;
	destination.planning_nanoseconds += source.planning_nanoseconds;
	destination.io_staging_nanoseconds += source.io_staging_nanoseconds;
	destination.ordered_submission_nanoseconds += source.ordered_submission_nanoseconds;
	destination.submit_to_ready_nanoseconds += source.submit_to_ready_nanoseconds;
}

} // namespace

struct NativeLogicalBatchPipeline::Impl final {
	Impl(const std::filesystem::path& manifest_path,
	     const std::string_view       semantic_profile_id,
	     NativePipelineTraceBuffer*   trace)
	    : planner(manifest_path),
	      optimized_physical(native_physical_orchestration_enabled()),
	      core(DirectDctRuntimeAdapter(manifest_path),
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

	Impl(std::shared_ptr<jpeg::DirectDctRuntime> runtime,
	     const std::filesystem::path&             manifest_path,
	     const std::string_view                   semantic_profile_id,
	     jpeg::JpegDctDeviceBatchOptions          options,
	     NativePipelineTraceBuffer*               trace)
	    : planner(manifest_path),
	      optimized_physical(native_physical_orchestration_enabled()),
	      core(DirectDctRuntimeAdapter(std::move(runtime)),
	           std::string(semantic_profile_id),
	           std::move(options),
	           trace) {
	}

	[[nodiscard]] const jpeg::JpegDctShardManifestEntry& shard(const uint32_t shard_id) const {
		if (!planner) {
			throw std::logic_error("native physical orchestration has no manifest planner");
		}
		const auto& shards = planner->manifest().shards;
		const auto found = std::find_if(shards.begin(), shards.end(), [shard_id](const auto& value) {
			return value.shard_id == shard_id;
		});
		if (found == shards.end()) {
			throw std::out_of_range("native physical plan references an unknown shard");
		}
		return *found;
	}

	[[nodiscard]] std::vector<LogicalBatchRequest>
	canonical_shard_requests(const std::vector<LogicalBatchRequest>& requests) {
		std::unordered_map<uint32_t, LogicalBatchRequest::Sample> samples;
		for (const auto& request : requests) {
			for (const auto& sample : request.samples) {
				if (!samples.emplace(sample.image_id, sample).second) {
					throw std::invalid_argument("native physical orchestration requires unique image IDs");
				}
			}
		}
		std::vector<uint32_t> touched_shards;
		for (size_t logical_index = 0U; logical_index < physical_plans.size(); ++logical_index) {
			const auto& plan = physical_plans[logical_index];
			for (const auto& segment : plan.segments) {
				if (std::find(touched_shards.begin(), touched_shards.end(), segment.shard_id) ==
				    touched_shards.end()) {
					touched_shards.push_back(segment.shard_id);
				}
				last_logical_use[segment.shard_id] = logical_index;
			}
		}
		std::vector<LogicalBatchRequest> canonical;
		canonical.reserve(touched_shards.size());
		physical_shard_ids.clear();
		for (size_t ordinal = 0U; ordinal < touched_shards.size(); ++ordinal) {
			const auto shard_id = touched_shards[ordinal];
			const auto& entry = shard(shard_id);
			LogicalBatchRequest request;
			request.request_identity = static_cast<uint64_t>(shard_id) + 1U;
			request.batch_ordinal = ordinal;
			request.semantic_profile_id = requests.front().semantic_profile_id;
			request.logical_batch_size = entry.image_count;
			request.partial_tail = false;
			request.samples.reserve(entry.image_count);
			for (uint64_t offset = 0U; offset < entry.image_count; ++offset) {
				const auto image_id = static_cast<uint32_t>(entry.first_global_image_index + offset);
				const auto found = samples.find(image_id);
				if (found == samples.end()) {
					throw std::invalid_argument(
					    "optimized native physical orchestration requires complete physical-shard coverage");
				}
				request.samples.push_back(found->second);
			}
			validate_logical_batch_request(request);
			physical_shard_ids.push_back(shard_id);
			canonical.push_back(std::move(request));
		}
		return canonical;
	}

	std::shared_ptr<jpeg::DirectDctBatch> load_shard(const uint32_t shard_id) {
		if (const auto found = active_shards.find(shard_id); found != active_shards.end()) {
			return found->second;
		}
		while (next_physical_batch < physical_shard_ids.size()) {
			const auto loaded_id = physical_shard_ids[next_physical_batch++];
			auto loaded = std::make_shared<jpeg::DirectDctBatch>(core.next());
			add_prefetch_metrics(last_logical_prefetch_metrics, core.prefetch_metrics());
			active_shards.emplace(loaded_id, loaded);
			if (loaded_id == shard_id) {
				return loaded;
			}
		}
		throw std::logic_error("native physical shard execution order did not satisfy SegmentPlan");
	}

	std::optional<PhysicalLayoutPlanner> planner;
	bool optimized_physical = false;
	std::vector<LogicalBatchPhysicalPlan> physical_plans;
	std::vector<LogicalBatchRequest> logical_requests;
	std::vector<uint32_t> physical_shard_ids;
	std::unordered_map<uint32_t, size_t> last_logical_use;
	std::unordered_map<uint32_t, std::shared_ptr<jpeg::DirectDctBatch>> active_shards;
	std::unordered_set<uint32_t> reported_shards;
	size_t next_physical_batch = 0U;
	size_t next_logical_batch = 0U;
	NativePipelineState logical_state;
	NativePipelinePrefetchMetrics last_logical_prefetch_metrics;
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

NativeLogicalBatchPipeline::NativeLogicalBatchPipeline(std::shared_ptr<jpeg::DirectDctRuntime> runtime,
                                                       const std::filesystem::path& manifest_path,
                                                       const std::string_view semantic_profile_id,
                                                       jpeg::JpegDctDeviceBatchOptions options,
                                                       NativePipelineTraceBuffer* trace)
    : impl_(std::make_unique<Impl>(
          std::move(runtime), manifest_path, semantic_profile_id, std::move(options), trace)) {
}

NativeLogicalBatchPipeline::~NativeLogicalBatchPipeline() = default;

void NativeLogicalBatchPipeline::reset(std::vector<LogicalBatchRequest> requests) {
	impl_->physical_plans = impl_->planner.has_value()
	                            ? impl_->planner->plan(requests)
	                            : std::vector<LogicalBatchPhysicalPlan> {};
	if (!impl_->optimized_physical) {
		impl_->core.reset(std::move(requests));
		return;
	}
	if (!impl_->planner || requests.empty()) {
		throw std::invalid_argument("optimized native physical orchestration requires a manifest and requests");
	}
	impl_->logical_requests = requests;
	impl_->last_logical_use.clear();
	impl_->active_shards.clear();
	impl_->reported_shards.clear();
	impl_->next_physical_batch = 0U;
	impl_->next_logical_batch = 0U;
	impl_->last_logical_prefetch_metrics = {};
	auto canonical = impl_->canonical_shard_requests(requests);
	impl_->logical_state = {};
	impl_->logical_state.lifecycle = NativePipelineState::Lifecycle::kRunning;
	impl_->logical_state.request_count = requests.size();
	impl_->logical_state.next_request = requests.size();
	impl_->logical_state.pending_count = std::min<size_t>(2U, requests.size());
	impl_->logical_state.max_pending_count = impl_->logical_state.pending_count;
	impl_->logical_state.closed = false;
	impl_->core.reset(std::move(canonical));
}

jpeg::DirectDctBatch NativeLogicalBatchPipeline::next() {
	if (!impl_->optimized_physical) {
		return impl_->core.next();
	}
	if (impl_->logical_state.closed || impl_->next_logical_batch >= impl_->logical_requests.size()) {
		throw std::out_of_range("NativeLogicalBatchPipeline is closed or exhausted");
	}
	impl_->last_logical_prefetch_metrics = {};
	const auto logical_index = impl_->next_logical_batch;
	const auto& request = impl_->logical_requests[logical_index];
	const auto& plan = impl_->physical_plans[logical_index];
	std::vector<jpeg::DirectDctBatch::LogicalSegmentInput> segments;
	segments.reserve(plan.segments.size());
	std::shared_ptr<jpeg::DirectDctBatch> stats_source;
	for (const auto& segment : plan.segments) {
		auto source = impl_->load_shard(segment.shard_id);
		const auto& shard = impl_->shard(segment.shard_id);
		const auto source_offset = static_cast<size_t>(
		    static_cast<uint64_t>(segment.first_global_image_id) - shard.first_global_image_index);
		segments.push_back(jpeg::DirectDctBatch::LogicalSegmentInput {
		    source, source_offset, segment.image_count, segment.logical_output_offset});
		if (!stats_source && impl_->reported_shards.insert(segment.shard_id).second) {
			stats_source = source;
		}
	}
	std::vector<uint32_t> image_ids;
	std::vector<jpeg::JpegDctImageCropRequest> transforms;
	image_ids.reserve(request.samples.size());
	transforms.reserve(request.samples.size());
	for (const auto& sample : request.samples) {
		image_ids.push_back(sample.image_id);
		transforms.push_back(lower_sample_copy(sample));
	}
	auto result = jpeg::DirectDctBatch::MakeLogicalGridBatch(
	    std::move(segments), std::move(image_ids), std::move(transforms), std::move(stats_source));
	for (const auto& segment : plan.segments) {
		const auto found = impl_->last_logical_use.find(segment.shard_id);
		if (found != impl_->last_logical_use.end() && found->second == logical_index) {
			impl_->active_shards.erase(segment.shard_id);
		}
	}
	++impl_->next_logical_batch;
	++impl_->logical_state.completed_request_count;
	impl_->logical_state.pending_count = std::min<size_t>(
	    2U, impl_->logical_requests.size() - impl_->next_logical_batch);
	if (impl_->next_logical_batch == impl_->logical_requests.size()) {
		impl_->logical_state.lifecycle = NativePipelineState::Lifecycle::kDrained;
	}
	return result;
}

bool NativeLogicalBatchPipeline::ready() const {
	return impl_->optimized_physical && impl_->active_shards.size() != 0U ? true : impl_->core.ready();
}

bool NativeLogicalBatchPipeline::started() const noexcept {
	return impl_->core.started();
}

size_t NativeLogicalBatchPipeline::prefetched_batch_count() const noexcept {
	return impl_->optimized_physical ? impl_->logical_state.next_request : impl_->core.prefetched_batch_count();
}

NativePipelineState NativeLogicalBatchPipeline::state() const noexcept {
	return impl_->optimized_physical ? impl_->logical_state : impl_->core.state();
}

NativePipelinePrefetchMetrics NativeLogicalBatchPipeline::prefetch_metrics() const noexcept {
	return impl_->optimized_physical ? impl_->last_logical_prefetch_metrics : impl_->core.prefetch_metrics();
}

const std::vector<LogicalBatchPhysicalPlan>& NativeLogicalBatchPipeline::physical_plans() const noexcept {
	return impl_->physical_plans;
}

size_t NativeLogicalBatchPipeline::close() noexcept {
	const auto cancelled = impl_->core.close();
	if (impl_->optimized_physical) {
		impl_->active_shards.clear();
		impl_->logical_requests.clear();
		impl_->physical_shard_ids.clear();
		impl_->logical_state.request_count = 0U;
		impl_->logical_state.next_request = 0U;
		impl_->logical_state.pending_count = 0U;
		impl_->logical_state.cancelled_request_count = cancelled;
		impl_->logical_state.closed = true;
		impl_->logical_state.lifecycle = NativePipelineState::Lifecycle::kClosed;
	}
	return cancelled;
}

} // namespace galp::direct_dct
