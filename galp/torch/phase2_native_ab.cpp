#include "galp/direct_dct.hpp"
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Deliberately compile the Phase-2 shadow executor into this private test
// module.  The production Torch extension remains linked only to the legacy
// pipeline, and no test hook is added to either production implementation.
#include "../src/direct_dct/native_logical_batch_pipeline.cpp"

namespace py = pybind11;

namespace {

using galp::direct_dct::LogicalBatchRequest;
using galp::direct_dct::NativePipelineState;
using galp::direct_dct::PipelineTraceEvent;

struct CapturedTensor final {
	std::array<size_t, 6> shape {};
	std::array<size_t, 6> strides {};
	std::string           dtype;
	std::string           bytes;
};

struct CapturedBatch final {
	CapturedTensor y;
	CapturedTensor cbcr;
	std::vector<uint32_t> global_image_ids;
	std::vector<galp::jpeg::JpegDctImageCropRequest> transforms;
	std::vector<galp::jpeg::JpegDctDeviceImageLayout> image_layouts;
	std::vector<galp::jpeg::JpegDctDeviceBlockMetadata> block_metadata;
	std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata> rowgroups;
	std::vector<uint8_t> selected_coefficients;
	galp::jpeg::JpegDctDeviceExecutionStats stats;
	galp::jpeg::JpegDctDeviceCacheStats cache;
	size_t block_count = 0U;
	size_t coefficients_per_block = 0U;
	int    cuda_device = -1;
};

struct CapturedRun final {
	std::vector<CapturedBatch> batches;
	std::vector<PipelineTraceEvent> trace;
	std::vector<galp::direct_dct::detail::NativePlanTraceIdentity> production_plan_identities;
	NativePipelineState state_after_reset;
	NativePipelineState state_before_close;
	NativePipelineState state_after_close;
	size_t prefetched_after_reset = 0U;
	size_t close_cancelled = 0U;
	std::vector<double> batch_latency_ms;
	double execution_wall_ms = 0.0;
};

void check_cuda(const cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
	}
}

galp::jpeg::JpegDctCropBox parse_crop(const py::handle value) {
	galp::jpeg::JpegDctCropBox crop;
	if (value.is_none()) {
		return crop;
	}
	const auto dict = py::reinterpret_borrow<py::dict>(value);
	crop.x          = dict["x"].cast<uint32_t>();
	crop.y          = dict["y"].cast<uint32_t>();
	crop.width      = dict["width"].cast<uint32_t>();
	crop.height     = dict["height"].cast<uint32_t>();
	return crop;
}

std::vector<LogicalBatchRequest> parse_requests(const py::list& batches,
                                                const std::string& profile_id,
                                                const size_t logical_batch_size) {
	std::vector<LogicalBatchRequest> result;
	result.reserve(batches.size());
	for (size_t batch_index = 0U; batch_index < batches.size(); ++batch_index) {
		const auto samples = py::reinterpret_borrow<py::list>(batches[batch_index]);
		LogicalBatchRequest request;
		request.request_identity    = 1000U + batch_index;
		request.batch_ordinal       = batch_index;
		request.semantic_profile_id = profile_id;
		request.logical_batch_size  = logical_batch_size;
		request.partial_tail        = samples.size() < logical_batch_size;
		request.samples.reserve(samples.size());
		for (const auto sample_value : samples) {
			const auto sample = py::reinterpret_borrow<py::dict>(sample_value);
			LogicalBatchRequest::Sample parsed;
			parsed.image_id = sample["image_id"].cast<uint32_t>();
			if (sample.contains("crop") && !sample["crop"].is_none()) {
				parsed.transform.source_crop = parse_crop(sample["crop"]);
			}
			if (sample.contains("horizontal_flip")) {
				parsed.transform.horizontal_flip = sample["horizontal_flip"].cast<bool>();
			}
			if (sample.contains("logical_sample_id")) {
				parsed.transform.logical_sample_id = sample["logical_sample_id"].cast<std::string>();
			}
			if (sample.contains("augmentation_key")) {
				parsed.transform.augmentation_key = sample["augmentation_key"].cast<std::string>();
			}
			request.samples.push_back(std::move(parsed));
		}
		galp::direct_dct::validate_logical_batch_request(request);
		result.push_back(std::move(request));
	}
	return result;
}

std::vector<galp::jpeg::JpegDctImageCropRequest> lower_request(const LogicalBatchRequest& logical) {
	std::vector<galp::jpeg::JpegDctImageCropRequest> result;
	result.reserve(logical.samples.size());
	for (const auto& sample : logical.samples) {
		galp::jpeg::JpegDctImageCropRequest request;
		request.global_image_index = sample.image_id;
		if (sample.transform.source_crop.has_value()) {
			request.source_crop = *sample.transform.source_crop;
		}
		request.horizontal_flip   = sample.transform.horizontal_flip;
		request.logical_sample_id = sample.transform.logical_sample_id;
		request.augmentation_key  = sample.transform.augmentation_key;
		result.push_back(std::move(request));
	}
	return result;
}

CapturedTensor capture_tensor(const galp::jpeg::DirectDctGridTensorDescriptor& descriptor,
                              const bool capture_data) {
	CapturedTensor result;
	result.shape   = descriptor.shape;
	result.strides = descriptor.strides;
	const size_t element_size = descriptor.dtype == galp::jpeg::DirectDctTensorDataType::kFloat32
	                                ? sizeof(float)
	                                : sizeof(int16_t);
	result.dtype = descriptor.dtype == galp::jpeg::DirectDctTensorDataType::kFloat32 ? "float32" : "int16";
	if (capture_data && descriptor.raw_data() != nullptr && !descriptor.empty()) {
		result.bytes.resize(descriptor.element_count() * element_size);
		check_cuda(cudaMemcpy(result.bytes.data(),
		                      descriptor.raw_data(),
		                      result.bytes.size(),
		                      cudaMemcpyDeviceToHost),
		           "cudaMemcpy(test-only tensor capture)");
	}
	return result;
}

CapturedBatch capture_batch(galp::jpeg::DirectDctBatch& batch,
                            const bool capture_data,
                            const bool capture_details) {
	batch.synchronize();
	CapturedBatch result;
	if (capture_details) {
		result.y                     = capture_tensor(batch.y_tensor_async(), capture_data);
		result.cbcr                  = capture_tensor(batch.cbcr_tensor_async(), capture_data);
		result.global_image_ids      = batch.global_image_ids();
		result.transforms            = batch.transform_requests();
		result.image_layouts         = batch.image_layouts();
		result.block_metadata        = batch.block_metadata();
		result.rowgroups             = batch.rowgroups();
		result.selected_coefficients = batch.selected_coefficients();
		result.block_count           = batch.block_count();
		result.coefficients_per_block = batch.coefficients_per_block();
		result.cuda_device           = batch.cuda_device();
	}
	result.stats                  = batch.execution_stats();
	result.cache                  = batch.cache_stats();
	return result;
}

std::vector<galp::direct_dct::detail::NativePlanTraceIdentity>
production_plan_identities(const std::filesystem::path& manifest,
                           const std::vector<LogicalBatchRequest>& logical_requests,
                           const galp::jpeg::JpegDctDeviceBatchOptions& options) {
	galp::direct_dct::DirectDctRuntimeAdapter adapter(manifest);
	std::vector<galp::direct_dct::detail::NativePlanTraceIdentity> identities;
	identities.reserve(logical_requests.size());
	for (const auto& logical : logical_requests) {
		auto per_batch_options = options;
		if (per_batch_options.async_planless_completion) {
			per_batch_options.transform_submission_gate =
			    std::make_shared<galp::jpeg::JpegDctDeviceTransformSubmissionGate>();
		}
		identities.push_back(adapter.trace_identity(lower_request(logical), per_batch_options));
	}
	return identities;
}

CapturedRun execute_native(const std::filesystem::path& manifest,
                           const std::string& profile_id,
                           const std::string& dct_coeffs,
                           std::vector<LogicalBatchRequest> logical_requests,
                           const bool trace_enabled,
                           const bool capture_data,
                           const bool capture_details) {
	CapturedRun result;
	const auto semantic = galp::direct_dct::SemanticProfileRegistry::resolve(profile_id);
	const auto policy   = galp::direct_dct::resolve_execution_policy(profile_id);
	auto       options  = galp::direct_dct::materialize_shadow_options(semantic, policy);
	if (!galp::jpeg::parse_jpeg_dct_coefficient_selection(dct_coeffs, options.coefficient_selection)) {
		throw std::invalid_argument(
		    "invalid DCT coefficient selection; expected all, first:N, or list:0,1,...");
	}
	if (trace_enabled) {
		result.production_plan_identities =
		    production_plan_identities(manifest, logical_requests, options);
	}
	galp::direct_dct::NativePipelineTraceBuffer trace;
	auto runtime = std::make_shared<galp::jpeg::DirectDctRuntime>(manifest);
	galp::direct_dct::NativeLogicalBatchPipeline pipeline(
	    std::move(runtime), profile_id, std::move(options), trace_enabled ? &trace : nullptr);
	const size_t batch_count = logical_requests.size();
	const auto execution_begin = std::chrono::steady_clock::now();
	pipeline.reset(std::move(logical_requests));
	result.state_after_reset    = pipeline.state();
	result.prefetched_after_reset = pipeline.prefetched_batch_count();
	result.batches.reserve(batch_count);
	result.batch_latency_ms.reserve(batch_count);
	for (size_t batch_index = 0U; batch_index < batch_count; ++batch_index) {
		const auto batch_begin = std::chrono::steady_clock::now();
		auto batch = pipeline.next();
		result.batches.push_back(capture_batch(batch, capture_data, capture_details));
		result.batch_latency_ms.push_back(
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - batch_begin).count());
	}
	result.state_before_close = pipeline.state();
	result.close_cancelled    = pipeline.close();
	result.state_after_close  = pipeline.state();
	if (trace_enabled) {
		result.trace = trace.snapshot();
	}
	result.execution_wall_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - execution_begin).count();
	return result;
}

const char* stage_name(const PipelineTraceEvent::Stage stage) {
	switch (stage) {
	case PipelineTraceEvent::Stage::kRequestAccepted: return "request_accepted";
	case PipelineTraceEvent::Stage::kPreparing: return "preparing";
	case PipelineTraceEvent::Stage::kPlanReady: return "plan_ready";
	case PipelineTraceEvent::Stage::kStaged: return "staged";
	case PipelineTraceEvent::Stage::kAwaitingPredecessor: return "awaiting_predecessor";
	case PipelineTraceEvent::Stage::kAwaitingOutputSlot: return "awaiting_output_slot";
	case PipelineTraceEvent::Stage::kOutputSlotAcquired: return "output_slot_acquired";
	case PipelineTraceEvent::Stage::kReadStarted: return "read_started";
	case PipelineTraceEvent::Stage::kSubmitted: return "submitted";
	case PipelineTraceEvent::Stage::kCompleted: return "completed";
	case PipelineTraceEvent::Stage::kCancelled: return "cancelled";
	case PipelineTraceEvent::Stage::kFailed: return "failed";
	case PipelineTraceEvent::Stage::kClosed: return "closed";
	}
	return "unknown";
}

const char* lifecycle_name(const NativePipelineState::Lifecycle lifecycle) {
	switch (lifecycle) {
	case NativePipelineState::Lifecycle::kIdle: return "idle";
	case NativePipelineState::Lifecycle::kRunning: return "running";
	case NativePipelineState::Lifecycle::kDrained: return "drained";
	case NativePipelineState::Lifecycle::kFailed: return "failed";
	case NativePipelineState::Lifecycle::kClosed: return "closed";
	}
	return "unknown";
}

py::dict state_to_dict(const NativePipelineState& state) {
	py::dict out;
	out["lifecycle"]               = lifecycle_name(state.lifecycle);
	out["request_count"]           = state.request_count;
	out["next_request"]            = state.next_request;
	out["pending_count"]           = state.pending_count;
	out["max_pending_count"]       = state.max_pending_count;
	out["completed_request_count"] = state.completed_request_count;
	out["cancelled_request_count"] = state.cancelled_request_count;
	out["output_slot_capacity"]    = state.output_slot_capacity;
	out["live_output_slots"]       = state.live_output_slots;
	out["peak_live_output_slots"]  = state.peak_live_output_slots;
	out["output_slot_waiters"]     = state.output_slot_waiters;
	out["live_output_bytes"]       = state.live_output_bytes;
	out["peak_output_bytes"]       = state.peak_output_bytes;
	out["maximum_output_slot_bytes"] = state.maximum_output_slot_bytes;
	out["closed"]                  = state.closed;
	return out;
}

py::dict tensor_to_dict(const CapturedTensor& tensor) {
	py::dict out;
	out["shape"]   = tensor.shape;
	out["strides"] = tensor.strides;
	out["dtype"]   = tensor.dtype;
	out["bytes"]   = py::bytes(tensor.bytes);
	return out;
}

py::dict stats_to_dict(const galp::jpeg::JpegDctDeviceExecutionStats& stats) {
	py::dict out;
#define GALP_PHASE2_STAT(name) out[#name] = stats.name
	GALP_PHASE2_STAT(decode_kernel_launch_count);
	GALP_PHASE2_STAT(gather_kernel_launch_count);
	GALP_PHASE2_STAT(prefix_gather_kernel_launch_count);
	GALP_PHASE2_STAT(cached_gather_kernel_launch_count);
	GALP_PHASE2_STAT(materialize_kernel_launch_count);
	GALP_PHASE2_STAT(planless_transform_kernel_launch_count);
	GALP_PHASE2_STAT(planless_transform_dense_kernel_launch_count);
	GALP_PHASE2_STAT(planless_transform_sparse_kernel_launch_count);
	GALP_PHASE2_STAT(planless_transform_dense_output_block_count);
	GALP_PHASE2_STAT(planless_transform_sparse_output_block_count);
	GALP_PHASE2_STAT(planless_transform_selected_coefficient_count);
	GALP_PHASE2_STAT(planless_transform_compact_binding_count);
	GALP_PHASE2_STAT(planless_transform_dense_binding_equivalent_count);
	GALP_PHASE2_STAT(fixed_grid_finalize_kernel_launch_count);
	GALP_PHASE2_STAT(project_decoded_ycbcr_grid_launch_count);
	GALP_PHASE2_STAT(internal_sync_count);
	GALP_PHASE2_STAT(cached_gather_sync_count);
	GALP_PHASE2_STAT(decoded_batch_sync_count);
	GALP_PHASE2_STAT(cached_gather_event_handoff_count);
	GALP_PHASE2_STAT(fixed_grid_round_event_handoff_count);
	GALP_PHASE2_STAT(decode_to_transform_event_handoff_count);
	GALP_PHASE2_STAT(copy_to_decode_event_handoff_count);
	GALP_PHASE2_STAT(workset_upload_dma_count);
	GALP_PHASE2_STAT(workset_upload_dma_bytes);
	GALP_PHASE2_STAT(workset_upload_count);
	GALP_PHASE2_STAT(scratch_upload_count);
	GALP_PHASE2_STAT(compressed_payload_bytes_read);
	GALP_PHASE2_STAT(selected_compressed_payload_bytes);
	GALP_PHASE2_STAT(rowgroup_storage_bytes_read);
	GALP_PHASE2_STAT(pread_count);
	GALP_PHASE2_STAT(preadv_count);
	GALP_PHASE2_STAT(galp_native_device_in_use_bytes);
	GALP_PHASE2_STAT(galp_native_device_peak_in_use_bytes);
	GALP_PHASE2_STAT(galp_native_device_cached_bytes);
	GALP_PHASE2_STAT(galp_native_device_allocation_requests);
	GALP_PHASE2_STAT(galp_native_device_cuda_allocation_count);
	GALP_PHASE2_STAT(galp_native_device_cuda_allocation_bytes);
	GALP_PHASE2_STAT(galp_native_pinned_in_use_bytes);
	GALP_PHASE2_STAT(galp_native_pinned_peak_in_use_bytes);
	GALP_PHASE2_STAT(galp_native_pinned_cached_bytes);
	GALP_PHASE2_STAT(galp_native_pinned_allocation_requests);
	GALP_PHASE2_STAT(galp_native_pinned_cuda_allocation_count);
	GALP_PHASE2_STAT(galp_native_pinned_cuda_allocation_bytes);
	GALP_PHASE2_STAT(actual_transient_total_used_high_water_bytes);
	GALP_PHASE2_STAT(actual_transient_total_allocated_high_water_bytes);
	GALP_PHASE2_STAT(decode_workset_output_arena_growth_count);
	GALP_PHASE2_STAT(decode_workset_chunk_arena_growth_count);
	GALP_PHASE2_STAT(compact_batch_buffer_growth_count);
	GALP_PHASE2_STAT(compact_batch_buffer_reuse_count);
	GALP_PHASE2_STAT(planless_axis_program_device_growth_count);
	GALP_PHASE2_STAT(planless_axis_program_pinned_growth_count);
	GALP_PHASE2_STAT(rowgroup_count);
	GALP_PHASE2_STAT(workset_count);
	GALP_PHASE2_STAT(planless_transform_threads_per_cta);
	GALP_PHASE2_STAT(planless_transform_max_blocks_per_launch);
	GALP_PHASE2_STAT(planless_transform_max_output_blocks_per_launch);
#undef GALP_PHASE2_STAT
	return out;
}

py::dict batch_to_dict(const CapturedBatch& batch) {
	py::dict out;
	out["y"]                      = tensor_to_dict(batch.y);
	out["cbcr"]                   = tensor_to_dict(batch.cbcr);
	out["global_image_ids"]       = batch.global_image_ids;
	out["selected_coefficients"]  = batch.selected_coefficients;
	out["block_count"]            = batch.block_count;
	out["coefficients_per_block"] = batch.coefficients_per_block;
	out["cuda_device"]            = batch.cuda_device;

	py::list transforms;
	for (const auto& transform : batch.transforms) {
		py::dict crop;
		crop["x"]      = transform.source_crop.x;
		crop["y"]      = transform.source_crop.y;
		crop["width"]  = transform.source_crop.width;
		crop["height"] = transform.source_crop.height;
		crop["unit"]   = "source_pixels";
		py::dict item;
		item["global_image_id"]  = transform.global_image_index;
		item["crop"]             = std::move(crop);
		item["horizontal_flip"]  = transform.horizontal_flip;
		item["logical_sample_id"] = transform.logical_sample_id;
		item["augmentation_key"]  = transform.augmentation_key;
		transforms.append(std::move(item));
	}
	out["transform_descriptors"] = std::move(transforms);

	py::list layouts;
	for (const auto& layout : batch.image_layouts) {
		py::dict item;
		item["global_image_index"] = layout.global_image_index;
		item["block_offset"]       = layout.block_offset;
		item["block_count"]        = layout.block_count;
		layouts.append(std::move(item));
	}
	out["image_layouts"] = std::move(layouts);

	py::list blocks;
	for (const auto& block : batch.block_metadata) {
		py::dict item;
		item["request_index"]      = block.request_index;
		item["global_image_index"] = block.global_image_index;
		item["semantic_slot_id"]   = block.semantic_slot_id;
		item["block_x"]            = block.block_x;
		item["block_y"]            = block.block_y;
		blocks.append(std::move(item));
	}
	out["block_metadata"] = std::move(blocks);

	py::list rowgroups;
	for (const auto& rowgroup : batch.rowgroups) {
		py::dict item;
		item["shard_id"]       = rowgroup.shard_id;
		item["rowgroup_index"] = rowgroup.rowgroup_index;
		rowgroups.append(std::move(item));
	}
	out["rowgroups"] = std::move(rowgroups);
	out["stats"]     = stats_to_dict(batch.stats);
	py::dict cache;
	cache["capacity_bytes"]      = batch.cache.capacity_bytes;
	cache["resident_bytes"]      = batch.cache.resident_bytes;
	cache["resident_rowgroups"]  = batch.cache.resident_rowgroups;
	cache["peak_resident_bytes"] = batch.cache.peak_resident_bytes;
	cache["hits"]                = batch.cache.hits;
	cache["misses"]              = batch.cache.misses;
	cache["inserts"]             = batch.cache.inserts;
	cache["evictions"]           = batch.cache.evictions;
	out["cache"] = std::move(cache);
	return out;
}

py::dict run_to_dict(const CapturedRun& run) {
	py::dict out;
	py::list batches;
	for (const auto& batch : run.batches) {
		batches.append(batch_to_dict(batch));
	}
	out["batches"] = std::move(batches);

	py::list trace;
	for (const auto& event : run.trace) {
		py::dict item;
		item["stage"]                 = stage_name(event.stage);
		item["request_identity"]      = event.request_identity;
		item["request_ordinal"]       = event.request_ordinal;
		item["batch_ordinal"]         = event.batch_ordinal;
		item["plan_identity_hash"]    = event.plan_identity_hash;
		item["io_identity_hash"]      = event.io_identity_hash;
		item["prepare_ordinal"]       = event.prepare_ordinal;
		item["stage_ordinal"]         = event.stage_ordinal;
		item["read_ordinal"]          = event.read_ordinal;
		item["submission_ordinal"]    = event.submission_ordinal;
		item["completion_ordinal"]    = event.completion_ordinal;
		trace.append(std::move(item));
	}
	out["trace"] = std::move(trace);

	py::list identities;
	for (const auto& identity : run.production_plan_identities) {
		py::dict item;
		item["plan_identity_hash"] = identity.plan_identity_hash;
		item["io_identity_hash"]   = identity.io_identity_hash;
		identities.append(std::move(item));
	}
	out["production_plan_identities"] = std::move(identities);
	out["state_after_reset"]          = state_to_dict(run.state_after_reset);
	out["state_before_close"]         = state_to_dict(run.state_before_close);
	out["state_after_close"]          = state_to_dict(run.state_after_close);
	out["prefetched_after_reset"]     = run.prefetched_after_reset;
	out["close_cancelled"]            = run.close_cancelled;
	out["batch_latency_ms"]           = run.batch_latency_ms;
	out["execution_wall_ms"]          = run.execution_wall_ms;
	return out;
}

} // namespace

PYBIND11_MODULE(_galp_phase2_native_ab, module) {
	module.doc() = "Private Phase-2 Legacy/Native GPU A/B probe; never linked into production";
	module.def(
	    "run_native",
	    [](const std::string& manifest_path,
	       const std::string& profile_id,
	       const std::string& dct_coeffs,
	       const py::list& request_batches,
	       const size_t logical_batch_size,
	       const bool trace_enabled,
	       const bool capture_data,
	       const bool capture_details) {
		    auto logical = parse_requests(request_batches, profile_id, logical_batch_size);
		    CapturedRun captured;
		    {
			    py::gil_scoped_release release;
			    captured = execute_native(
			        std::filesystem::path(manifest_path),
			        profile_id,
			        dct_coeffs,
			        std::move(logical),
			        trace_enabled,
			        capture_data,
			        capture_details);
		    }
		    return run_to_dict(captured);
	    },
	    py::arg("manifest_path"),
	    py::arg("profile_id"),
	    py::arg("dct_coeffs"),
	    py::arg("request_batches"),
	    py::arg("logical_batch_size"),
	    py::arg("trace_enabled") = true,
	    py::arg("capture_data") = true,
	    py::arg("capture_details") = true);
}
