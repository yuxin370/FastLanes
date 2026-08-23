#include "galp/direct_dct.hpp"
#include "galp/profiles/registry.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <algorithm>
#include <atomic>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr size_t kBytesPerMiB                         = size_t {1024} * size_t {1024};
constexpr size_t kDefaultDirectDctCacheCapacityMiB    = 0;
constexpr size_t kDefaultDirectDctDecodeWorksetCapacityMiB =
    galp::jpeg::kDefaultJpegDctDeviceDecodeWorksetCapacityBytes / kBytesPerMiB;
constexpr auto   kImmediateFuturePollDuration         = std::chrono::seconds(0);

galp::jpeg::JpegDctCropBox parse_crop(const py::object& crop) {
	galp::jpeg::JpegDctCropBox parsed {};
	if (crop.is_none()) {
		return parsed;
	}
	if (py::isinstance<py::dict>(crop)) {
		const auto dict = py::reinterpret_borrow<py::dict>(crop);
		for (const auto* key : {"x", "y", "width", "height"}) {
			if (!dict.contains(key)) {
				throw std::invalid_argument(std::string("crop dict is missing ") + key);
			}
		}
		parsed.x      = dict["x"].cast<uint32_t>();
		parsed.y      = dict["y"].cast<uint32_t>();
		parsed.width  = dict["width"].cast<uint32_t>();
		parsed.height = dict["height"].cast<uint32_t>();
		return parsed;
	}
	const auto seq = py::reinterpret_borrow<py::sequence>(crop);
	if (seq.size() != 4) {
		throw std::invalid_argument("crop must be None or a 4-item sequence: (x, y, width, height)");
	}
	parsed.x      = seq[0].cast<uint32_t>();
	parsed.y      = seq[1].cast<uint32_t>();
	parsed.width  = seq[2].cast<uint32_t>();
	parsed.height = seq[3].cast<uint32_t>();
	return parsed;
}

std::vector<galp::jpeg::JpegDctImageCropRequest>
parse_transform_requests(const std::vector<uint32_t>& image_ids,
                         const py::object&            transforms,
                         const galp::jpeg::JpegDctCropBox& default_crop) {
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	requests.reserve(image_ids.size());
	if (transforms.is_none()) {
		for (const auto image_id : image_ids) {
			requests.push_back(galp::jpeg::JpegDctImageCropRequest {image_id, default_crop});
		}
		return requests;
	}
	const auto sequence = py::reinterpret_borrow<py::sequence>(transforms);
	if (static_cast<size_t>(sequence.size()) != image_ids.size()) {
		throw std::invalid_argument("transforms must contain exactly one descriptor per image_id");
	}
	for (size_t index = 0; index < image_ids.size(); ++index) {
		if (!py::isinstance<py::dict>(sequence[index])) {
			throw std::invalid_argument("each transform descriptor must be a dict");
		}
		const auto descriptor = py::reinterpret_borrow<py::dict>(sequence[index]);
		galp::jpeg::JpegDctImageCropRequest request;
		request.global_image_index = image_ids[index];
		request.source_crop = descriptor.contains("crop") ? parse_crop(descriptor["crop"]) : default_crop;
		request.horizontal_flip = descriptor.contains("horizontal_flip")
		                              ? descriptor["horizontal_flip"].cast<bool>()
		                              : false;
		request.logical_sample_id = descriptor.contains("logical_sample_id")
		                                ? descriptor["logical_sample_id"].cast<std::string>()
		                                : std::string {};
		request.augmentation_key = descriptor.contains("augmentation_key")
		                               ? descriptor["augmentation_key"].cast<std::string>()
		                               : std::string {};
		requests.push_back(std::move(request));
	}
	return requests;
}

py::list transform_requests_to_python(const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests) {
	py::list result;
	for (const auto& request : requests) {
		py::dict crop;
		crop["x"]      = request.source_crop.x;
		crop["y"]      = request.source_crop.y;
		crop["width"]  = request.source_crop.width;
		crop["height"] = request.source_crop.height;
		crop["unit"]   = "source_pixels";
		py::dict descriptor;
		descriptor["global_image_id"]   = request.global_image_index;
		descriptor["crop"]              = std::move(crop);
		descriptor["horizontal_flip"]   = request.horizontal_flip;
		descriptor["logical_sample_id"] = request.logical_sample_id;
		descriptor["augmentation_key"]  = request.augmentation_key;
		result.append(std::move(descriptor));
	}
	return result;
}

galp::jpeg::JpegDctCoefficientSelection parse_coefficients(const std::string& spec);
galp::jpeg::JpegDctDeviceLayout parse_layout(const std::string& layout);
galp::jpeg::JpegDctSchedulingPolicy parse_scheduling_policy(const std::string& policy);
galp::jpeg::JpegDctBlockMajorDoubleBufferPolicy
parse_block_major_double_buffer_policy(const std::string& policy);

std::optional<galp::jpeg::JpegDctGridTransformSpec> parse_grid_transform(const py::object& value) {
	if (value.is_none()) {
		return std::nullopt;
	}
	if (!py::isinstance<py::dict>(value)) {
		throw std::invalid_argument("grid_transform must be None or a dict");
	}
	const auto dict = py::reinterpret_borrow<py::dict>(value);
	const auto required_u32 = [&](const char* key) {
		if (!dict.contains(key)) {
			throw std::invalid_argument(std::string("grid_transform is missing ") + key);
		}
		return dict[key].cast<uint32_t>();
	};
	const auto optional_bool = [&](const char* key, const bool fallback) {
		return dict.contains(key) ? dict[key].cast<bool>() : fallback;
	};

	galp::jpeg::JpegDctGridTransformSpec spec;
	spec.y_output_width_blocks        = required_u32("y_output_width_blocks");
	spec.y_output_height_blocks       = required_u32("y_output_height_blocks");
	spec.cbcr_output_width_blocks     = required_u32("cbcr_output_width_blocks");
	spec.cbcr_output_height_blocks    = required_u32("cbcr_output_height_blocks");
	spec.crop_reference_width_blocks  = required_u32("crop_reference_width_blocks");
	spec.crop_reference_height_blocks = required_u32("crop_reference_height_blocks");
	spec.crop_origin_alignment_blocks = required_u32("crop_origin_alignment_blocks");
	spec.chroma_crop_scale_x          = required_u32("chroma_crop_scale_x");
	spec.chroma_crop_scale_y          = required_u32("chroma_crop_scale_y");
	if (!dict.contains("clamp_min") || !dict.contains("clamp_max")) {
		throw std::invalid_argument("grid_transform requires clamp_min and clamp_max");
	}
	spec.clamp_min                = dict["clamp_min"].cast<int32_t>();
	spec.clamp_max                = dict["clamp_max"].cast<int32_t>();
	if (dict.contains("output_dtype")) {
		const auto output_dtype = dict["output_dtype"].cast<std::string>();
		if (output_dtype == "int16") {
			spec.output_data_type = galp::jpeg::JpegDctGridOutputDataType::kInt16;
		} else if (output_dtype == "float32") {
			spec.output_data_type = galp::jpeg::JpegDctGridOutputDataType::kFloat32;
		} else {
			throw std::invalid_argument("grid_transform output_dtype must be 'int16' or 'float32'");
		}
	}
	spec.output_add   = dict.contains("output_add") ? dict["output_add"].cast<float>() : 0.0F;
	spec.output_scale = dict.contains("output_scale") ? dict["output_scale"].cast<float>() : 1.0F;
	spec.dequantize               = optional_bool("dequantize", true);
	spec.require_all_coefficients = optional_bool("require_all_coefficients", true);
	spec.allow_grayscale          = optional_bool("allow_grayscale", false);
	if (dict.contains("preferred_small_crop_width_blocks")) {
		spec.preferred_small_crop_width_blocks =
		    dict["preferred_small_crop_width_blocks"].cast<std::vector<uint32_t>>();
	}
	if (dict.contains("preferred_small_crop_height_blocks")) {
		spec.preferred_small_crop_height_blocks =
		    dict["preferred_small_crop_height_blocks"].cast<std::vector<uint32_t>>();
	}
	if (dict.contains("allowed_chroma_sampling_ratios")) {
		for (const auto item : dict["allowed_chroma_sampling_ratios"].cast<py::sequence>()) {
			const auto ratio = py::reinterpret_borrow<py::sequence>(item);
			if (ratio.size() != 4) {
				throw std::invalid_argument("each allowed_chroma_sampling_ratios item must contain four integers");
			}
			spec.allowed_chroma_sampling_ratios.push_back(galp::jpeg::JpegDctSamplingRatio {
			    ratio[0].cast<uint16_t>(), ratio[1].cast<uint16_t>(), ratio[2].cast<uint16_t>(), ratio[3].cast<uint16_t>()});
		}
	}
	return spec;
}

size_t cache_capacity_bytes_from_mib(const size_t cache_capacity_mib) {
	if (cache_capacity_mib > std::numeric_limits<size_t>::max() / kBytesPerMiB) {
		throw std::invalid_argument("cache_capacity_mib is too large");
	}
	return cache_capacity_mib * kBytesPerMiB;
}

galp::jpeg::JpegDctCropExecutionMode parse_crop_execution_mode(const std::string& mode) {
	if (mode == "auto") {
		return galp::jpeg::JpegDctCropExecutionMode::kAutomatic;
	}
	if (mode == "full-rowgroup-decode" || mode == "full_rowgroup_decode") {
		return galp::jpeg::JpegDctCropExecutionMode::kFullRowgroupDecode;
	}
	if (mode == "rowgroup-read-selected-decode" || mode == "rowgroup_read_selected_decode") {
		return galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
	}
	if (mode == "vector-range-read-selected-decode" || mode == "vector_range_read_selected_decode") {
		return galp::jpeg::JpegDctCropExecutionMode::kVectorRangeReadSelectedDecode;
	}
	if (mode == "bounded-range-read-selected-decode" || mode == "bounded_range_read_selected_decode") {
		return galp::jpeg::JpegDctCropExecutionMode::kBoundedRangeReadSelectedDecode;
	}
	if (mode == "bounded-io-uring-range-read-selected-decode" ||
	    mode == "bounded_io_uring_range_read_selected_decode") {
		return galp::jpeg::JpegDctCropExecutionMode::kBoundedIoUringRangeReadSelectedDecode;
	}
	if (mode == "bounded-io-uring-scheduled-range-read-selected-decode" ||
	    mode == "bounded_io_uring_scheduled_range_read_selected_decode") {
		return galp::jpeg::JpegDctCropExecutionMode::kBoundedIoUringScheduledRangeReadSelectedDecode;
	}
	throw std::invalid_argument(
	    "invalid crop_execution_mode; expected auto, full-rowgroup-decode, "
	    "rowgroup-read-selected-decode, vector-range-read-selected-decode, or "
	    "bounded-range-read-selected-decode, bounded-io-uring-range-read-selected-decode, or "
	    "bounded-io-uring-scheduled-range-read-selected-decode");
}

uint32_t amplification_cap_to_ppm(const double cap, const bool allow_inherit, const char* const label) {
	if (allow_inherit && cap == 0.0) {
		return 0U;
	}
	if (!std::isfinite(cap) || cap < 1.0 || cap > 1.10) {
		throw std::invalid_argument(std::string(label) + " must be in [1.0, 1.10]");
	}
	const auto scaled = std::llround(cap * 1'000'000.0);
	if (scaled < 1'000'000LL || scaled > 1'100'000LL) {
		throw std::invalid_argument(std::string(label) + " is outside the supported integer cap range");
	}
	return static_cast<uint32_t>(scaled);
}

galp::jpeg::JpegDctDeviceBatchOptions make_batch_options(const std::string& dct_coeffs,
                                                         const size_t       cache_capacity_mib,
                                                         const size_t       decode_batch_rowgroups,
                                                         const bool         enable_rowgroup_prefetch,
                                                         const size_t       rowgroup_prefetch_depth,
                                                         const size_t       rowgroup_prefetch_workers,
                                                         const size_t       rowgroup_prefetch_min_decode_batches,
                                                         const std::string& layout,
                                                         const py::object&  grid_transform,
	                                                     const size_t       plan_cache_capacity,
	                                                     const bool         enable_planless_execution,
	                                                     const size_t       decode_workset_capacity_mib,
	                                                     const std::string& crop_execution_mode = "auto",
	                                                     const std::string& scheduling_policy = "fully-overlapped",
                                                         const size_t       transform_blocks_per_launch = 0,
                                                         const size_t       transform_ctas_per_launch = 0,
	                                                     const bool         use_low_priority_streams = false,
	                                                     const std::string& block_major_double_buffer = "auto",
	                                                     const bool         async_planless_completion = false,
	                                                     const double       bounded_read_amplification_cap = 1.0,
	                                                     const double       bounded_read_local_amplification_cap = 0.0,
	                                                     const size_t       bounded_read_max_run_bytes = 0U) {
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.coefficient_selection                = parse_coefficients(dct_coeffs);
	options.layout                               = parse_layout(layout);
	options.grid_transform                       = parse_grid_transform(grid_transform);
	options.cache_capacity_bytes                 = cache_capacity_bytes_from_mib(cache_capacity_mib);
	options.decode_batch_rowgroups               = decode_batch_rowgroups;
	options.decode_workset_capacity_bytes         = cache_capacity_bytes_from_mib(decode_workset_capacity_mib);
	options.plan_cache_capacity                  = plan_cache_capacity;
	options.enable_rowgroup_prefetch             = enable_rowgroup_prefetch;
	options.rowgroup_prefetch_depth              = rowgroup_prefetch_depth;
	options.rowgroup_prefetch_workers            = rowgroup_prefetch_workers;
	options.rowgroup_prefetch_min_decode_batches = rowgroup_prefetch_min_decode_batches;
	options.enable_planless_execution            = enable_planless_execution;
	options.scheduling_policy                    = parse_scheduling_policy(scheduling_policy);
	options.transform_blocks_per_launch          = transform_blocks_per_launch;
	options.transform_ctas_per_launch            = transform_ctas_per_launch;
	options.use_low_priority_streams              = use_low_priority_streams;
	options.async_planless_completion             = async_planless_completion;
	options.block_major_double_buffer_policy      =
	    parse_block_major_double_buffer_policy(block_major_double_buffer);
	options.crop_execution_mode                   = parse_crop_execution_mode(crop_execution_mode);
	options.bounded_read_amplification_ppm = amplification_cap_to_ppm(
	    bounded_read_amplification_cap, /*allow_inherit=*/false, "bounded_read_amplification_cap");
	options.bounded_read_local_amplification_ppm = amplification_cap_to_ppm(
	    bounded_read_local_amplification_cap,
	    /*allow_inherit=*/true,
	    "bounded_read_local_amplification_cap");
	options.bounded_read_max_run_bytes = bounded_read_max_run_bytes;
	return options;
}

galp::jpeg::JpegDctCoefficientSelection parse_coefficients(const std::string& spec) {
	galp::jpeg::JpegDctCoefficientSelection selection;
	if (!galp::jpeg::parse_jpeg_dct_coefficient_selection(spec, selection)) {
		throw std::invalid_argument("invalid DCT coefficient selection; expected all, first:N, or list:0,1,...");
	}
	return selection;
}

galp::jpeg::JpegDctDeviceLayout parse_layout(const std::string& layout) {
	if (layout == "compact" || layout == "image-major-component-block-coeff" ||
	    layout == "image_major_component_block_coeff") {
		return galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	}
	if (layout == "ycbcr_dct_grid" || layout == "ycbcr-dct-grid") {
		return galp::jpeg::JpegDctDeviceLayout::kYcbcrDctGrid;
	}
	if (layout == "transformed_dct_grid" || layout == "transformed-dct-grid") {
		return galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	}
	throw std::invalid_argument(
	    "invalid DCT output layout; expected compact, ycbcr_dct_grid, or transformed_dct_grid");
}

galp::jpeg::JpegDctSchedulingPolicy parse_scheduling_policy(const std::string& policy) {
	if (policy == "fully-overlapped" || policy == "fully_overlapped") {
		return galp::jpeg::JpegDctSchedulingPolicy::kFullyOverlapped;
	}
	if (policy == "limited-overlap" || policy == "limited_overlap") {
		return galp::jpeg::JpegDctSchedulingPolicy::kLimitedOverlap;
	}
	if (policy == "serial") {
		return galp::jpeg::JpegDctSchedulingPolicy::kSerial;
	}
	throw std::invalid_argument(
	    "invalid scheduling_policy; expected fully-overlapped, limited-overlap, or serial");
}

galp::jpeg::JpegDctBlockMajorDoubleBufferPolicy
parse_block_major_double_buffer_policy(const std::string& policy) {
	if (policy == "auto" || policy == "automatic") {
		return galp::jpeg::JpegDctBlockMajorDoubleBufferPolicy::kAutomatic;
	}
	if (policy == "on" || policy == "enabled") {
		return galp::jpeg::JpegDctBlockMajorDoubleBufferPolicy::kEnabled;
	}
	if (policy == "off" || policy == "disabled") {
		return galp::jpeg::JpegDctBlockMajorDoubleBufferPolicy::kDisabled;
	}
	throw std::invalid_argument("invalid block_major_double_buffer; expected auto, on, or off");
}

std::string layout_to_string(const galp::jpeg::JpegDctDeviceLayout layout) {
	switch (layout) {
	case galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff:
		return "compact";
	case galp::jpeg::JpegDctDeviceLayout::kYcbcrDctGrid:
		return "ycbcr_dct_grid";
	case galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid:
		return "transformed_dct_grid";
	}
	return "unknown";
}

py::dict direct_dct_profile_info(const galp::profiles::RegisteredDirectDctProfile& profile) {
	py::dict out;
	out["schema"]            = "galp-direct-dct-profile-v1";
	out["id"]                = profile.id;
	out["output_profile_id"] = profile.output.id;
	out["runtime_policy_id"] = profile.runtime.id;
	out["layout"]            = layout_to_string(profile.output.layout);
	if (profile.output.grid_transform.has_value()) {
		const auto& transform = *profile.output.grid_transform;
		out["output_dtype"] = transform.output_data_type == galp::jpeg::JpegDctGridOutputDataType::kFloat32
		                          ? "float32"
		                          : "int16";
		out["crop_reference_blocks"] =
		    py::make_tuple(transform.crop_reference_width_blocks, transform.crop_reference_height_blocks);
		out["y_output_blocks"] =
		    py::make_tuple(transform.y_output_width_blocks, transform.y_output_height_blocks);
		out["cbcr_output_blocks"] =
		    py::make_tuple(transform.cbcr_output_width_blocks, transform.cbcr_output_height_blocks);
	}
	return out;
}

py::dict image_layout_to_dict(const galp::jpeg::JpegDctDeviceImageLayout& layout) {
	py::dict out;
	out["global_image_index"] = layout.global_image_index;
	out["block_offset"]       = layout.block_offset;
	out["block_count"]        = layout.block_count;
	return out;
}

py::dict block_metadata_to_dict(const galp::jpeg::JpegDctDeviceBlockMetadata& block) {
	py::dict out;
	out["request_index"]      = block.request_index;
	out["global_image_index"] = block.global_image_index;
	out["semantic_slot_id"]   = block.semantic_slot_id;
	out["block_x"]            = block.block_x;
	out["block_y"]            = block.block_y;
	return out;
}

	py::dict rowgroup_metadata_to_dict(const galp::jpeg::JpegDctDeviceRowgroupMetadata& rowgroup) {
		py::dict out;
		out["shard_id"]       = rowgroup.shard_id;
		out["rowgroup_index"] = rowgroup.rowgroup_index;
		return out;
	}

	std::vector<galp::jpeg::JpegDctImageCropRequest>
	make_crop_requests(const std::vector<uint32_t>& image_ids, const galp::jpeg::JpegDctCropBox& crop) {
		std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
		requests.reserve(image_ids.size());
		for (const auto image_id : image_ids) {
			requests.push_back(galp::jpeg::JpegDctImageCropRequest {image_id, crop});
		}
		return requests;
	}

	py::dict plan_preview_to_dict(const galp::jpeg::JpegDctDeviceBatchPlanPreview& preview) {
		py::dict out;
		out["layout"]                          = layout_to_string(preview.layout);
		out["image_count"]                     = preview.image_layouts.size();
		out["block_count"]                     = preview.block_metadata.size();
		out["rowgroup_count"]                  = preview.rowgroups.size();
		out["planned_selected_vector_count"]   = preview.planned_selected_vector_count;
		out["estimated_selected_vector_count"] = preview.estimated_selected_vector_count;
		out["full_vector_count"]               = preview.full_vector_count;
		out["planned_saved_vector_count"]      = preview.planned_saved_vector_count;
	    out["estimated_saved_vector_count"]                = preview.estimated_saved_vector_count;
	    out["coefficients_per_block"]                      = preview.coefficients_per_block;
	    out["planned_selected_vector_ratio"]               = preview.planned_selected_vector_ratio;
	    out["estimated_selected_vector_ratio"]             = preview.estimated_selected_vector_ratio;
	    out["planning_ms"]                                 = preview.planning_ms;
	    out["resize_weight_build_ms"]                      = preview.resize_weight_build_ms;
	    out["dct_resize_weight_cache_hits"]                = preview.dct_resize_weight_cache_hits;
	    out["dct_resize_weight_cache_misses"]              = preview.dct_resize_weight_cache_misses;
	    out["dct_conversion_matrix_cache_hits"]            = preview.dct_conversion_matrix_cache_hits;
	    out["dct_conversion_matrix_cache_misses"]          = preview.dct_conversion_matrix_cache_misses;
	    out["uses_planless_fixed_transform"]               = preview.uses_planless_fixed_transform;
	    out["compact_image_descriptor_count"]              = preview.compact_image_descriptor_count;
	    out["fixed_transform_component_count"]             = preview.fixed_transform_component_count;
	    out["fixed_transform_source_block_count"]          = preview.fixed_transform_source_block_count;
	    out["fixed_transform_output_block_count"]          = preview.fixed_transform_output_block_count;
	    out["host_expanded_transform_items_created"]       = preview.host_expanded_transform_items_created;
	    out["host_output_block_source_lists_created"]      = preview.host_output_block_source_lists_created;
	    out["host_global_transform_sort_items"]            = preview.host_global_transform_sort_items;
	    out["planless_axis_program_count"]                 = preview.planless_axis_program_count;
	    out["planless_axis_phase_matrix_count"]            = preview.planless_axis_phase_matrix_count;
	    out["compiled_access_profile_hits"]                = preview.compiled_access_profile_hits;
	    out["compiled_access_profile_misses"]              = preview.compiled_access_profile_misses;
	    out["planless_axis_program_bytes"]                 = preview.planless_axis_program_bytes;
	    out["planless_axis_program_capacity_contract_bytes"] =
	        preview.planless_axis_program_capacity_contract_bytes;
	    out["planless_axis_program_capacity_contract_complete"] =
	        preview.planless_axis_program_capacity_contract_complete;
	    out["compact_plan_bytes"]                          = preview.compact_plan_bytes;
	    out["compact_plan_peak_bytes"]                     = preview.compact_plan_peak_bytes;
	    out["coordinate_group_lookup_count"]               = preview.coordinate_group_lookup_count;
	    out["coordinate_group_index_entries"]              = preview.coordinate_group_index_entries;
	    out["coordinate_group_index_populated"]            = preview.coordinate_group_index_populated;
	    out["coordinate_group_index_holes"]                = preview.coordinate_group_index_holes;
	    out["coordinate_group_index_bytes"]                = preview.coordinate_group_index_bytes;
	    out["coordinate_group_index_density"]              = preview.coordinate_group_index_density;
	    out["exact_batch_plan_cache_enabled"]              = preview.exact_batch_plan_cache_enabled;
	    out["compact_reader_image_locator_bytes"]          = preview.compact_reader_image_locator_bytes;
	    out["compact_reader_shard_index_bytes"]            = preview.compact_reader_shard_index_bytes;
	    out["compact_reader_layout_dictionary_bytes"]      = preview.compact_reader_layout_dictionary_bytes;
	    out["compact_reader_quant_table_dictionary_bytes"] = preview.compact_reader_quant_table_dictionary_bytes;
	    out["compact_reader_total_bytes"]                  = preview.compact_reader_total_bytes;
	    out["compact_reader_shard_descriptor_bytes"]       = preview.compact_reader_shard_descriptor_bytes;
	    out["compact_reader_shard_index_derived"]          = preview.compact_reader_shard_index_derived;

	    py::list selected_coefficients;
	    for (const auto coeff : preview.selected_coefficients) {
		    selected_coefficients.append(coeff);
	    }
	    out["selected_coefficients"] = std::move(selected_coefficients);

		py::list image_layouts;
		for (const auto& layout : preview.image_layouts) {
			image_layouts.append(image_layout_to_dict(layout));
		}
		out["image_layouts"] = std::move(image_layouts);

		py::list block_metadata;
		for (const auto& block : preview.block_metadata) {
			block_metadata.append(block_metadata_to_dict(block));
		}
		out["block_metadata"] = std::move(block_metadata);

		py::list rowgroups;
		for (const auto& rowgroup : preview.rowgroups) {
			rowgroups.append(rowgroup_metadata_to_dict(rowgroup));
		}
		out["rowgroups"] = std::move(rowgroups);
		if (preview.layout == galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid) {
			out["y_shape"] = py::make_tuple(preview.ycbcr_dct_grid_shape.y[0],
			                                preview.ycbcr_dct_grid_shape.y[1],
			                                preview.ycbcr_dct_grid_shape.y[2],
			                                preview.ycbcr_dct_grid_shape.y[3],
			                                preview.ycbcr_dct_grid_shape.y[4],
			                                preview.ycbcr_dct_grid_shape.y[5]);
			out["cbcr_shape"] = py::make_tuple(preview.ycbcr_dct_grid_shape.cbcr[0],
			                                  preview.ycbcr_dct_grid_shape.cbcr[1],
			                                  preview.ycbcr_dct_grid_shape.cbcr[2],
			                                  preview.ycbcr_dct_grid_shape.cbcr[3],
			                                  preview.ycbcr_dct_grid_shape.cbcr[4],
			                                  preview.ycbcr_dct_grid_shape.cbcr[5]);
		}
		return out;
	}

	py::dict reader_initialization_stats_to_dict(const galp::jpeg::JpegDctReaderInitializationStats& stats) {
		py::dict out;
		out["manifest_load_ms"] = stats.manifest_load_ms;
		out["shard_path_validation_ms"] = stats.shard_path_validation_ms;
		out["shard_metadata_load_ms"] = stats.shard_metadata_load_ms;
		out["shard_metadata_index_ms"] = stats.shard_metadata_index_ms;
		out["transform_profile_construction_ms"] = stats.transform_profile_construction_ms;
		out["block_major_companion_index_load_ms"] = stats.block_major_companion_index_load_ms;
		out["total_ms"] = stats.total_ms;
		out["manifest_shard_count"] = stats.manifest_shard_count;
		out["eagerly_loaded_shard_metadata_count"] = stats.eagerly_loaded_shard_metadata_count;
		out["loaded_shard_metadata_count"] = stats.loaded_shard_metadata_count;
		out["block_major_metadata_lazy"] = stats.block_major_metadata_lazy;
		out["block_major_loaded_descriptor_count"] = stats.block_major_loaded_descriptor_count;
		out["block_major_loaded_descriptor_bytes"] = stats.block_major_loaded_descriptor_bytes;
		out["block_major_descriptor_cache_byte_bound"] = stats.block_major_descriptor_cache_byte_bound;
		out["block_major_descriptor_open_ms"] = stats.block_major_descriptor_open_ms;
		out["block_major_descriptor_validation_ms"] = stats.block_major_descriptor_validation_ms;
		return out;
	}

	py::dict image_metadata_to_dict(const galp::jpeg::JpegImageMetadata& image) {
	py::dict out;
	out["source_path"]       = image.source_path.string();
	out["image_width"]       = image.image_width;
	out["image_height"]      = image.image_height;
	out["jpeg_color_space"]  = image.jpeg_color_space;
	out["progressive"]       = image.progressive;
	out["data_precision"]    = image.data_precision;
	out["warning_count"]     = image.warning_count;

	py::list quant_tables;
	for (const auto& table : image.quant_tables) {
		py::dict table_out;
		table_out["table_id"] = table.table_id;
		py::list values;
		for (const auto value : table.values) {
			values.append(value);
		}
		table_out["values"] = std::move(values);
		quant_tables.append(std::move(table_out));
	}
	out["quant_tables"] = std::move(quant_tables);

	py::list components;
	for (const auto& component : image.components) {
		py::dict component_out;
		component_out["component_index"]          = component.component_index;
		component_out["component_id"]             = component.component_id;
		component_out["width_in_blocks"]          = component.width_in_blocks;
		component_out["height_in_blocks"]         = component.height_in_blocks;
		component_out["padded_width_in_blocks"]   = component.padded_width_in_blocks;
		component_out["padded_height_in_blocks"]  = component.padded_height_in_blocks;
		component_out["h_samp_factor"]            = component.h_samp_factor;
		component_out["v_samp_factor"]            = component.v_samp_factor;
		component_out["present"]                  = component.present;
		component_out["semantic_slot_id"]         = component.semantic_slot_id;
		component_out["local_component_index"]    = component.local_component_index;
		component_out["quant_tbl_no"]             = component.quant_tbl_no;
		component_out["quant_table_fingerprint"]  = component.quant_table_fingerprint;
		component_out["encoding_profile_id"]      = component.encoding_profile_id;
		components.append(std::move(component_out));
	}
	out["components"] = std::move(components);
	return out;
}

py::dict cache_stats_to_dict(const galp::jpeg::JpegDctDeviceCacheStats& stats) {
	py::dict out;
	out["capacity_bytes"]     = stats.capacity_bytes;
	out["resident_bytes"]     = stats.resident_bytes;
	out["resident_rowgroups"] = stats.resident_rowgroups;
	out["peak_resident_bytes"] = stats.peak_resident_bytes;
	out["peak_resident_rowgroups"] = stats.peak_resident_rowgroups;
	out["hits"]               = stats.hits;
	out["misses"]             = stats.misses;
	out["inserts"]            = stats.inserts;
	out["evictions"]          = stats.evictions;
	return out;
}

py::dict execution_stats_to_dict(const galp::jpeg::JpegDctDeviceExecutionStats& stats) {
	py::dict out;
	out["planned_selected_vector_count"]                 = stats.planned_selected_vector_count;
	out["selected_vector_count"]                         = stats.selected_vector_count;
	out["full_vector_count"]                             = stats.full_vector_count;
	out["planned_saved_vector_count"]                    = stats.planned_saved_vector_count;
	out["actual_saved_vector_count"]                     = stats.actual_saved_vector_count;
	out["decoded_coefficient_bytes"]                     = stats.decoded_coefficient_bytes;
	out["rowgroup_count"]                                = stats.rowgroup_count;
	out["workset_count"]                                 = stats.workset_count;
	out["decode_kernel_launch_count"]                    = stats.decode_kernel_launch_count;
	out["gather_kernel_launch_count"]                    = stats.gather_kernel_launch_count;
	out["prefix_gather_kernel_launch_count"]             = stats.prefix_gather_kernel_launch_count;
	out["cached_gather_kernel_launch_count"]             = stats.cached_gather_kernel_launch_count;
	out["materialize_kernel_launch_count"]               = stats.materialize_kernel_launch_count;
	out["gather_item_count"]                             = stats.gather_item_count;
	out["decoded_gather_item_count"]                     = stats.decoded_gather_item_count;
	out["cached_gather_item_count"]                      = stats.cached_gather_item_count;
	out["projection_item_count"]                         = stats.projection_item_count;
	out["decoded_projection_item_count"]                 = stats.decoded_projection_item_count;
	out["fixed_transform_item_count"]                    = stats.fixed_transform_item_count;
	out["fixed_transform_image_count"]                   = stats.fixed_transform_image_count;
	out["fixed_transform_component_count"]               = stats.fixed_transform_component_count;
	out["fixed_transform_source_block_count"]            = stats.fixed_transform_source_block_count;
	out["fixed_transform_output_block_count"]            = stats.fixed_transform_output_block_count;
	out["host_expanded_transform_items_created"]         = stats.host_expanded_transform_items_created;
	out["host_output_block_source_lists_created"]        = stats.host_output_block_source_lists_created;
	out["host_global_transform_sort_items"]              = stats.host_global_transform_sort_items;
	out["uses_planless_fixed_transform"]                 = stats.uses_planless_fixed_transform;
	out["planless_image_descriptor_count"]               = stats.planless_image_descriptor_count;
	out["planless_transform_output_block_count"]         = stats.planless_transform_output_block_count;
	out["planless_transform_full_scan_output_block_count"] =
	    stats.planless_transform_full_scan_output_block_count;
	out["planless_transform_skipped_output_block_count"] =
	    stats.planless_transform_skipped_output_block_count;
	out["planless_transform_active_output_index_bytes"] =
	    stats.planless_transform_active_output_index_bytes;
	out["planless_transform_active_output_offset_bytes"] =
	    stats.planless_transform_active_output_offset_bytes;
	out["planless_transform_active_output_schedule_peak_bytes"] =
	    stats.planless_transform_active_output_schedule_peak_bytes;
	out["planless_transform_source_contribution_count"] =
	    stats.planless_transform_source_contribution_count;
	out["planless_transform_source_contribution_visit_count"] =
	    stats.planless_transform_source_contribution_visit_count;
	out["planless_transform_output_workset_ownership_count"] =
	    stats.planless_transform_output_workset_ownership_count;
	out["planless_transform_active_output_workset_count"] =
	    stats.planless_transform_active_output_workset_count;
	out["planless_transform_active_output_schedule_build_count"] =
	    stats.planless_transform_active_output_schedule_build_count;
	out["planless_transform_active_output_offsets_valid"] =
	    stats.planless_transform_active_output_offsets_valid;
	out["planless_transform_active_output_planning_ms"] =
	    stats.planless_transform_active_output_planning_ms;
	out["planless_transform_group_workset_build_ms"] =
	    stats.planless_transform_group_workset_build_ms;
	out["planless_transform_active_output_count_ms"] =
	    stats.planless_transform_active_output_count_ms;
	out["planless_transform_active_output_prefix_ms"] =
	    stats.planless_transform_active_output_prefix_ms;
	out["planless_transform_active_output_fill_ms"] =
	    stats.planless_transform_active_output_fill_ms;
	out["active_output_schedule_sidecar_hit_count"] =
	    stats.active_output_schedule_sidecar_hit_count;
	out["active_output_schedule_sidecar_miss_count"] =
	    stats.active_output_schedule_sidecar_miss_count;
	out["active_output_schedule_sidecar_reject_count"] =
	    stats.active_output_schedule_sidecar_reject_count;
	out["active_output_schedule_sidecar_persist_count"] =
	    stats.active_output_schedule_sidecar_persist_count;
	out["active_output_schedule_sidecar_bytes"] = stats.active_output_schedule_sidecar_bytes;
	out["active_output_schedule_interval_count"] = stats.active_output_schedule_interval_count;
	out["active_output_schedule_mapped_bytes_peak"] =
	    stats.active_output_schedule_mapped_bytes_peak;
	out["active_output_schedule_mmap_capacity_bytes"] =
	    stats.active_output_schedule_mmap_capacity_bytes;
	out["active_output_schedule_mmap_window_count"] =
	    stats.active_output_schedule_mmap_window_count;
	out["active_output_schedule_load_ms"] = stats.active_output_schedule_load_ms;
	out["active_output_schedule_validation_ms"] = stats.active_output_schedule_validation_ms;
	out["active_output_schedule_materialize_ms"] = stats.active_output_schedule_materialize_ms;
	out["active_output_schedule_persist_ms"] = stats.active_output_schedule_persist_ms;
	out["planless_transform_gpu_kernel_ms"] = stats.planless_transform_gpu_kernel_ms;
	out["async_planless_completion_batch_count"] = stats.async_planless_completion_batch_count;
	out["coordinate_group_lookup_count"] = stats.coordinate_group_lookup_count;
	out["coordinate_group_index_entries"] = stats.coordinate_group_index_entries;
	out["coordinate_group_index_populated"] = stats.coordinate_group_index_populated;
	out["coordinate_group_index_holes"] = stats.coordinate_group_index_holes;
	out["coordinate_group_index_bytes"] = stats.coordinate_group_index_bytes;
	out["coordinate_group_index_density"] = stats.coordinate_group_index_density;
	out["rowgroup_storage_bytes_read"]                   = stats.rowgroup_storage_bytes_read;
	out["storage_read_granularity"]                     = stats.storage_read_granularity;
	out["decode_granularity"]                           = stats.decode_granularity;
	out["requested_source_block_count"]                 = stats.requested_source_block_count;
	out["planned_vector_count"]                         = stats.planned_vector_count;
	out["actual_vector_count"]                          = stats.actual_vector_count;
	out["compressed_payload_bytes_read"]                = stats.compressed_payload_bytes_read;
	out["selected_compressed_payload_bytes"]            = stats.selected_compressed_payload_bytes;
	out["full_compressed_payload_bytes"]                = stats.full_compressed_payload_bytes;
	out["pread_count"]                                  = stats.pread_count;
	out["preadv_count"]                                 = stats.preadv_count;
	out["merged_gap_bytes"]                             = stats.merged_gap_bytes;
	out["hole_clear_bytes"]                             = stats.hole_clear_bytes;
	out["static_prefix_restore_bytes"]                  = stats.static_prefix_restore_bytes;
	out["hole_clear_ms"]                                = stats.hole_clear_ms;
	out["static_prefix_restore_ms"]                     = stats.static_prefix_restore_ms;
	out["vector_bundle_rowgroup_count"]                 = stats.vector_bundle_rowgroup_count;
	out["vector_bundle_envelope_rowgroup_count"]        = stats.vector_bundle_envelope_rowgroup_count;
	out["vector_bundle_pread_count"]                    = stats.vector_bundle_pread_count;
	out["pinned_rowgroup_read_count"]                  = stats.pinned_rowgroup_read_count;
	out["pinned_rowgroup_read_bytes"]                  = stats.pinned_rowgroup_read_bytes;
	out["compact_batch_buffer_acquire_count"]          = stats.compact_batch_buffer_acquire_count;
	out["compact_batch_buffer_growth_count"]           = stats.compact_batch_buffer_growth_count;
	out["compact_batch_buffer_reuse_count"]            = stats.compact_batch_buffer_reuse_count;
	out["compact_batch_buffer_requested_bytes"]        = stats.compact_batch_buffer_requested_bytes;
	out["compact_batch_buffer_capacity_bytes"]         = stats.compact_batch_buffer_capacity_bytes;
	out["compact_batch_buffer_high_water_bytes"]       = stats.compact_batch_buffer_high_water_bytes;
	out["compact_batch_buffer_pageable_fallback_count"] = stats.compact_batch_buffer_pageable_fallback_count;
	out["compact_batch_pool_prewarm_performed"]          = stats.compact_batch_pool_prewarm_performed;
	out["compact_batch_pool_prewarmed_slots"]            = stats.compact_batch_pool_prewarmed_slots;
	out["compact_batch_pool_prewarmed_bytes"]            = stats.compact_batch_pool_prewarmed_bytes;
	out["compact_batch_pool_largest_size_class_bytes"]   =
	    stats.compact_batch_pool_largest_size_class_bytes;
	out["compact_batch_pool_capacity_contract_complete"] =
	    stats.compact_batch_pool_capacity_contract_complete;
	out["compact_batch_pool_capacity_contract_images"] =
	    stats.compact_batch_pool_capacity_contract_images;
	out["compact_batch_pool_capacity_contract_groups"] =
	    stats.compact_batch_pool_capacity_contract_groups;
	out["compact_batch_pool_capacity_contract_batches"] =
	    stats.compact_batch_pool_capacity_contract_batches;
	out["compact_batch_pool_capacity_contract_bytes"] =
	    stats.compact_batch_pool_capacity_contract_bytes;
	out["compact_batch_read_group_count"]              = stats.compact_batch_read_group_count;
	out["compact_batch_read_worker_count"]             = stats.compact_batch_read_worker_count;
	out["galp_native_pinned_in_use_bytes"]             = stats.galp_native_pinned_in_use_bytes;
	out["galp_native_pinned_peak_in_use_bytes"]        = stats.galp_native_pinned_peak_in_use_bytes;
	out["galp_native_pinned_cached_bytes"]             = stats.galp_native_pinned_cached_bytes;
	out["galp_native_pinned_allocation_requests"]      = stats.galp_native_pinned_allocation_requests;
	out["galp_native_pinned_cuda_allocation_count"]    = stats.galp_native_pinned_cuda_allocation_count;
	out["galp_native_pinned_cuda_allocation_bytes"]    = stats.galp_native_pinned_cuda_allocation_bytes;
	out["coefficient_range_rowgroup_count"]            = stats.coefficient_range_rowgroup_count;
	out["coefficient_logical_bytes_requested"]         = stats.coefficient_logical_bytes_requested;
	out["coefficient_range_bytes_read"]                = stats.coefficient_range_bytes_read;
	out["physical_page_bytes_covered"]                 = stats.physical_page_bytes_covered;
	out["full_physical_page_bytes"]                    = stats.full_physical_page_bytes;
	out["coalesced_read_run_count"]                    = stats.coalesced_read_run_count;
	out["selected_coefficient_count"]                  = stats.selected_coefficient_count;
	out["full_coefficient_count"]                      = stats.full_coefficient_count;
	out["selected_coefficient_ratio"]                  = stats.selected_coefficient_ratio;
	out["physical_page_coverage_ratio"]                = stats.physical_page_coverage_ratio;
	out["read_amplification"]                           = stats.read_amplification;
	out["duplicate_physical_read_count"]                = stats.duplicate_physical_read_count;
	out["rowgroup_revisit_count"]                       = stats.rowgroup_revisit_count;
	out["vector_run_revisit_count"]                     = stats.vector_run_revisit_count;
	out["physical_read_order_inversions"]               = stats.physical_read_order_inversions;
	out["source_blocks_transformed"]                    = stats.source_blocks_transformed;
	out["sparse_read_supported"]                        = stats.sparse_read_supported;
	out["sparse_read_fallback_rowgroup_count"]          = stats.sparse_read_fallback_rowgroup_count;
	out["sparse_read_fallback_reason"]                  = stats.sparse_read_fallback_reason;
	out["automatic_sparse_storage_candidate_rowgroup_count"] =
	    stats.automatic_sparse_storage_candidate_rowgroup_count;
	out["automatic_sparse_storage_selected_rowgroup_count"] =
	    stats.automatic_sparse_storage_selected_rowgroup_count;
	out["automatic_sparse_storage_rejected_rowgroup_count"] =
	    stats.automatic_sparse_storage_rejected_rowgroup_count;
	out["automatic_sparse_storage_early_rejected_rowgroup_count"] =
	    stats.automatic_sparse_storage_early_rejected_rowgroup_count;
	out["automatic_sparse_storage_full_bytes"] = stats.automatic_sparse_storage_full_bytes;
	out["automatic_sparse_storage_candidate_bytes"] = stats.automatic_sparse_storage_candidate_bytes;
	out["automatic_sparse_storage_candidate_pread_count"] =
	    stats.automatic_sparse_storage_candidate_pread_count;
	out["automatic_sparse_storage_optimistic_bytes"] = stats.automatic_sparse_storage_optimistic_bytes;
	out["automatic_sparse_storage_optimistic_pread_count"] =
	    stats.automatic_sparse_storage_optimistic_pread_count;
	out["automatic_sparse_storage_full_estimated_ns"] = stats.automatic_sparse_storage_full_estimated_ns;
	out["automatic_sparse_storage_candidate_estimated_ns"] =
	    stats.automatic_sparse_storage_candidate_estimated_ns;
	out["adaptive_run_interval_estimated_ns"] = stats.adaptive_run_interval_estimated_ns;
	out["adaptive_bitmap_estimated_ns"] = stats.adaptive_bitmap_estimated_ns;
	out["adaptive_full_rowgroup_estimated_ns"] = stats.adaptive_full_rowgroup_estimated_ns;
	out["adaptive_selected_memory_fit_rowgroup_count"] =
	    stats.adaptive_selected_memory_fit_rowgroup_count;
	out["adaptive_full_memory_fit_rowgroup_count"] = stats.adaptive_full_memory_fit_rowgroup_count;
	out["run_interval_exact_rowgroup_count"] = stats.run_interval_exact_rowgroup_count;
	out["run_interval_bounded_rowgroup_count"] = stats.run_interval_bounded_rowgroup_count;
	out["bitmap_exact_rowgroup_count"]       = stats.bitmap_exact_rowgroup_count;
	out["full_rowgroup_strategy_count"]      = stats.full_rowgroup_strategy_count;
	out["bounded_read_amplification_ppm"] = stats.bounded_read_amplification_ppm;
	out["bounded_read_local_amplification_ppm"] = stats.bounded_read_local_amplification_ppm;
	out["bounded_read_max_run_bytes"] = stats.bounded_read_max_run_bytes;
	out["bounded_io_backend"] = stats.bounded_io_backend;
	out["bounded_io_uring_queue_depth"] = stats.bounded_io_uring_queue_depth;
	out["bounded_exact_storage_bytes"] = stats.bounded_exact_storage_bytes;
	out["bounded_physical_storage_bytes"] = stats.bounded_physical_storage_bytes;
	out["bounded_merged_gap_bytes"] = stats.bounded_merged_gap_bytes;
	out["bounded_exact_extent_count"] = stats.bounded_exact_extent_count;
	out["bounded_physical_run_count"] = stats.bounded_physical_run_count;
	out["bounded_selected_gap_count"] = stats.bounded_selected_gap_count;
	out["bounded_max_run_rejected_gap_count"] = stats.bounded_max_run_rejected_gap_count;
	out["bounded_budget_rejected_gap_count"] = stats.bounded_budget_rejected_gap_count;
	out["io_uring_read_request_count"] = stats.io_uring_read_request_count;
	out["io_uring_completion_count"] = stats.io_uring_completion_count;
	out["io_uring_submit_syscall_count"] = stats.io_uring_submit_syscall_count;
	out["io_uring_wait_syscall_count"] = stats.io_uring_wait_syscall_count;
	out["io_uring_setup_count"] = stats.io_uring_setup_count;
	out["io_uring_ring_mapped_bytes"] = stats.io_uring_ring_mapped_bytes;
	out["io_uring_fallback_count"] = stats.io_uring_fallback_count;
	out["io_uring_read_ms"] = stats.io_uring_read_ms;
	out["bounded_coalesce_ms"] = stats.bounded_coalesce_ms;
	out["decode_workset_capacity_bytes"]     = stats.decode_workset_capacity_bytes;
	out["max_estimated_decode_workset_bytes"] = stats.max_estimated_decode_workset_bytes;
	out["oversized_decode_rowgroup_count"]    = stats.oversized_decode_rowgroup_count;
	out["decode_workset_capacity_plan_image_count"] = stats.decode_workset_capacity_plan_image_count;
	out["decode_workset_output_arena_capacity_plan_bytes"] =
	    stats.decode_workset_output_arena_capacity_plan_bytes;
	out["decode_workset_output_arena_requested_bytes"] =
	    stats.decode_workset_output_arena_requested_bytes;
	out["decode_workset_output_arena_capacity_bytes"] =
	    stats.decode_workset_output_arena_capacity_bytes;
	out["decode_workset_output_arena_growth_count"] =
	    stats.decode_workset_output_arena_growth_count;
	out["decode_workset_output_arena_growth_bytes"] =
	    stats.decode_workset_output_arena_growth_bytes;
	out["decode_workset_chunk_arena_capacity_plan_bytes"] =
	    stats.decode_workset_chunk_arena_capacity_plan_bytes;
	out["decode_workset_chunk_arena_requested_bytes"] =
	    stats.decode_workset_chunk_arena_requested_bytes;
	out["decode_workset_chunk_arena_capacity_bytes"] =
	    stats.decode_workset_chunk_arena_capacity_bytes;
	out["decode_workset_chunk_arena_growth_count"] =
	    stats.decode_workset_chunk_arena_growth_count;
	out["decode_workset_chunk_arena_growth_bytes"] =
	    stats.decode_workset_chunk_arena_growth_bytes;
	out["bounded_double_buffer_enabled"] = stats.bounded_double_buffer_enabled;
	out["bounded_double_buffer_policy"] = stats.bounded_double_buffer_policy;
	out["bounded_double_buffer_candidate"] = stats.bounded_double_buffer_candidate;
	out["bounded_double_buffer_workset_count"] = stats.bounded_double_buffer_workset_count;
	out["bounded_double_buffer_peak_estimated_bytes"] =
	    stats.bounded_double_buffer_peak_estimated_bytes;
	out["actual_transient_current_chunk_compressed_peak_bytes"] =
	    stats.actual_transient_current_chunk_compressed_peak_bytes;
	out["actual_transient_next_chunk_compressed_peak_bytes"] =
	    stats.actual_transient_next_chunk_compressed_peak_bytes;
	out["actual_transient_compressed_backing_live_peak_bytes"] =
	    stats.actual_transient_compressed_backing_live_peak_bytes;
	out["actual_transient_decoded_arena_used_peak_bytes"] =
	    stats.actual_transient_decoded_arena_used_peak_bytes;
	out["actual_transient_decoded_arena_capacity_peak_bytes"] =
	    stats.actual_transient_decoded_arena_capacity_peak_bytes;
	out["actual_transient_ring_fixed_buffer_peak_bytes"] =
	    stats.actual_transient_ring_fixed_buffer_peak_bytes;
	out["actual_transient_active_schedule_peak_bytes"] =
	    stats.actual_transient_active_schedule_peak_bytes;
	out["actual_transient_kernel_referenced_backing_peak_bytes"] =
	    stats.actual_transient_kernel_referenced_backing_peak_bytes;
	out["actual_transient_total_used_high_water_bytes"] =
	    stats.actual_transient_total_used_high_water_bytes;
	out["actual_transient_total_allocated_high_water_bytes"] =
	    stats.actual_transient_total_allocated_high_water_bytes;
	out["actual_transient_memory_gate_passed"] = stats.actual_transient_memory_gate_passed;
	out["galp_native_device_in_use_bytes"]               = stats.galp_native_device_in_use_bytes;
	out["galp_native_device_peak_in_use_bytes"]          = stats.galp_native_device_peak_in_use_bytes;
	out["galp_native_device_cached_bytes"]               = stats.galp_native_device_cached_bytes;
	out["galp_native_device_allocation_requests"]        = stats.galp_native_device_allocation_requests;
	out["galp_native_device_cuda_allocation_count"]      = stats.galp_native_device_cuda_allocation_count;
	out["galp_native_device_cuda_allocation_bytes"]      = stats.galp_native_device_cuda_allocation_bytes;
	out["device_mapping_ms"]                             = stats.device_mapping_ms;
	out["device_mapping_fused"]                          = stats.device_mapping_fused;
	out["planless_axis_program_count"]                   = stats.planless_axis_program_count;
	out["planless_axis_phase_matrix_count"]              = stats.planless_axis_phase_matrix_count;
	out["planless_axis_program_bytes"]                   = stats.planless_axis_program_bytes;
	out["planless_axis_program_capacity_contract_bytes"] =
	    stats.planless_axis_program_capacity_contract_bytes;
	out["planless_axis_program_capacity_contract_complete"] =
	    stats.planless_axis_program_capacity_contract_complete;
	out["planless_axis_program_device_capacity_bytes"] =
	    stats.planless_axis_program_device_capacity_bytes;
	out["planless_axis_program_pinned_capacity_bytes"] =
	    stats.planless_axis_program_pinned_capacity_bytes;
	out["planless_axis_program_device_growth_count"] =
	    stats.planless_axis_program_device_growth_count;
	out["planless_axis_program_pinned_growth_count"] =
	    stats.planless_axis_program_pinned_growth_count;
	out["compact_plan_bytes"]                            = stats.compact_plan_bytes;
	out["compact_plan_peak_bytes"]                       = stats.compact_plan_peak_bytes;
	out["project_decoded_ycbcr_grid_launch_count"]       = stats.project_decoded_ycbcr_grid_launch_count;
	out["jpeg_dct_projection_items_materialized"]        = stats.jpeg_dct_projection_items_materialized;
	out["planless_transform_kernel_launch_count"]        = stats.planless_transform_kernel_launch_count;
	out["planless_transform_max_blocks_per_launch"]      = stats.planless_transform_max_blocks_per_launch;
	out["planless_transform_max_output_blocks_per_launch"] =
	    stats.planless_transform_max_output_blocks_per_launch;
	out["planless_transform_registers_per_thread"]        = stats.planless_transform_registers_per_thread;
	out["planless_transform_static_shared_bytes_per_cta"] =
	    stats.planless_transform_static_shared_bytes_per_cta;
	out["planless_transform_local_bytes_per_thread"]      = stats.planless_transform_local_bytes_per_thread;
	out["planless_transform_threads_per_cta"]             = stats.planless_transform_threads_per_cta;
	out["planless_transform_max_active_ctas_per_sm"]      = stats.planless_transform_max_active_ctas_per_sm;
	out["cuda_max_threads_per_sm"]                        = stats.cuda_max_threads_per_sm;
	out["cuda_warp_size"]                                 = stats.cuda_warp_size;
	out["decode_to_transform_event_handoff_count"]       = stats.decode_to_transform_event_handoff_count;
	out["copy_to_decode_event_handoff_count"]            = stats.copy_to_decode_event_handoff_count;
	out["direct_dct_stream_priority"]                     = stats.direct_dct_stream_priority;
	out["direct_dct_h2d_stream_priority"]                 = stats.direct_dct_h2d_stream_priority;
	out["direct_dct_decode_stream_priority"]              = stats.direct_dct_decode_stream_priority;
	out["direct_dct_transform_stream_priority"]           = stats.direct_dct_transform_stream_priority;
	out["direct_dct_round_stream_priority"]               = stats.direct_dct_round_stream_priority;
	out["cuda_least_stream_priority"]                     = stats.cuda_least_stream_priority;
	out["cuda_greatest_stream_priority"]                  = stats.cuda_greatest_stream_priority;
	out["direct_dct_low_priority_streams"]                = stats.direct_dct_low_priority_streams;
	out["scheduling_policy"]                              = stats.scheduling_policy;
	out["fixed_grid_round_event_handoff_count"]          = stats.fixed_grid_round_event_handoff_count;
	out["fixed_grid_finalize_kernel_launch_count"]       = stats.fixed_grid_finalize_kernel_launch_count;
	out["fixed_grid_output_float32"]                     = stats.fixed_grid_output_float32;
	out["fixed_grid_output_affine_applied"]              = stats.fixed_grid_output_affine_applied;
	out["fixed_grid_output_add"]                         = stats.fixed_grid_output_add;
	out["fixed_grid_output_scale"]                       = stats.fixed_grid_output_scale;
	out["workset_upload_count"]                          = stats.workset_upload_count;
	out["scratch_upload_count"]                          = stats.scratch_upload_count;
	out["scratch_allocation_count"]                      = stats.scratch_allocation_count;
	out["internal_sync_count"]                           = stats.internal_sync_count;
	out["cached_gather_sync_count"]                      = stats.cached_gather_sync_count;
	out["decoded_batch_sync_count"]                      = stats.decoded_batch_sync_count;
	out["cached_gather_event_handoff_count"]             = stats.cached_gather_event_handoff_count;
	out["sparse_vector_cache_hits"]                      = stats.sparse_vector_cache_hits;
	out["sparse_vector_cache_misses"]                    = stats.sparse_vector_cache_misses;
	out["plan_cache_hits"]                               = stats.plan_cache_hits;
	out["plan_cache_misses"]                             = stats.plan_cache_misses;
	out["plan_cache_evictions"]                          = stats.plan_cache_evictions;
	out["exact_batch_plan_cache_enabled"]                = stats.exact_batch_plan_cache_enabled;
	out["runtime_policy_selected_rowgroups"]             = stats.runtime_policy_selected_rowgroups;
	out["runtime_policy_full_rowgroups"]                 = stats.runtime_policy_full_rowgroups;
	out["runtime_policy_tail_full_rowgroups"]            = stats.runtime_policy_tail_full_rowgroups;
	out["runtime_policy_ratio_full_rowgroups"]           = stats.runtime_policy_ratio_full_rowgroups;
	out["runtime_policy_low_saving_full_rowgroups"]      = stats.runtime_policy_low_saving_full_rowgroups;
	out["runtime_policy_forced_full_rowgroups"]          = stats.runtime_policy_forced_full_rowgroups;
	out["runtime_policy_forced_selected_rowgroups"]      = stats.runtime_policy_forced_selected_rowgroups;
	out["prefetch_initial_cache_hit_rowgroup_count"]     = stats.prefetch_initial_cache_hit_rowgroup_count;
	out["prefetch_candidate_rowgroup_count"]             = stats.prefetch_candidate_rowgroup_count;
	out["prefetch_active_shard_count"]                   = stats.prefetch_active_shard_count;
	out["prefetch_config_disabled_shard_count"]          = stats.prefetch_config_disabled_shard_count;
	out["prefetch_all_hit_shard_count"]                  = stats.prefetch_all_hit_shard_count;
	out["prefetch_small_batch_disabled_shard_count"]     = stats.prefetch_small_batch_disabled_shard_count;
	out["prefetch_selected_vector_disabled_shard_count"] = stats.prefetch_selected_vector_disabled_shard_count;
	out["prefetch_selected_vector_miss_rowgroup_count"]  = stats.prefetch_selected_vector_miss_rowgroup_count;
	out["prefetch_initial_hit_runtime_miss_count"]       = stats.prefetch_initial_hit_runtime_miss_count;
	out["prefetch_skipped_repeated_runtime_miss_count"]  = stats.prefetch_skipped_repeated_runtime_miss_count;
	out["prefetched_rowgroup_count"]                     = stats.prefetched_rowgroup_count;
	out["prefetch_consumed_as_hit_count"]                = stats.prefetch_consumed_as_hit_count;
	out["prefetch_skipped_repeated_rowgroup_count"]      = stats.prefetch_skipped_repeated_rowgroup_count;
	out["prefetch_consumed_as_hit_read_ms"]              = stats.prefetch_consumed_as_hit_read_ms;
	out["prefetch_consumed_as_hit_wait_ms"]              = stats.prefetch_consumed_as_hit_wait_ms;
	out["planning_ms"]                                   = stats.planning_ms;
	out["plan_device_batch_ms"]                          = stats.plan_device_batch_ms;
	out["compile_io_plan_ms"]                            = stats.compile_io_plan_ms;
	out["reader_lookup_ms"]                              = stats.reader_lookup_ms;
	out["descriptor_open_ms"]                            = stats.descriptor_open_ms;
	out["schema_plan_build_ms"]                          = stats.schema_plan_build_ms;
	out["static_metadata_wait_ms"]                       = stats.static_metadata_wait_ms;
	out["reader_cache_hit_count"]                        = stats.reader_cache_hit_count;
	out["reader_cache_miss_count"]                       = stats.reader_cache_miss_count;
	out["reader_cache_eviction_count"]                   = stats.reader_cache_eviction_count;
	out["static_metadata_cache_hit_count"]               = stats.static_metadata_cache_hit_count;
	out["static_metadata_cache_miss_count"]              = stats.static_metadata_cache_miss_count;
	out["descriptor_map_count"]                          = stats.descriptor_map_count;
	out["static_metadata_wait_count"]                    = stats.static_metadata_wait_count;
	out["parallel_reader_resolve_ms"]                    = stats.parallel_reader_resolve_ms;
	out["parallel_reader_resolve_workers"]               = stats.parallel_reader_resolve_workers;
	out["dynamic_image_planning_ms"]                      = stats.dynamic_image_planning_ms;
	out["crop_geometry_planning_ms"]                      = stats.crop_geometry_planning_ms;
	out["crop_interval_planning_ms"]                      = stats.crop_interval_planning_ms;
	out["axis_program_planning_ms"]                       = stats.axis_program_planning_ms;
	out["rowgroup_binding_planning_ms"]                   = stats.rowgroup_binding_planning_ms;
	out["plan_finalize_ms"]                               = stats.plan_finalize_ms;
	out["active_reader_count"]                           = stats.active_reader_count;
	out["active_reader_peak_count"]                      = stats.active_reader_peak_count;
	out["static_metadata_count"]                         = stats.static_metadata_count;
	out["static_metadata_peak_count"]                    = stats.static_metadata_peak_count;
	out["static_metadata_bytes"]                         = stats.static_metadata_bytes;
	out["static_metadata_peak_bytes"]                    = stats.static_metadata_peak_bytes;
	out["planning_unique_shard_count"]                   = stats.planning_unique_shard_count;
	out["planning_rowgroup_binding_count"]               = stats.planning_rowgroup_binding_count;
	out["static_metadata_prewarm_performed"]             = stats.static_metadata_prewarm_performed;
	out["static_metadata_prewarm_ms"]                    = stats.static_metadata_prewarm_ms;
	out["static_metadata_prewarm_shards"]                = stats.static_metadata_prewarm_shards;
	out["static_metadata_prewarm_workers"]               = stats.static_metadata_prewarm_workers;
	out["payload_fd_current_count"]                      = stats.payload_fd_current_count;
	out["payload_fd_peak_count"]                         = stats.payload_fd_peak_count;
	out["payload_fd_open_count"]                         = stats.payload_fd_open_count;
	out["payload_fd_close_count"]                        = stats.payload_fd_close_count;
	out["descriptor_unmap_count"]                        = stats.descriptor_unmap_count;
	out["descriptor_mapping_current_count"]              = stats.descriptor_mapping_current_count;
	out["descriptor_mapping_peak_count"]                 = stats.descriptor_mapping_peak_count;
	out["descriptor_map_process_count"]                  = stats.descriptor_map_process_count;
	out["descriptor_mapped_current_bytes"]               = stats.descriptor_mapped_current_bytes;
	out["descriptor_mapped_peak_bytes"]                  = stats.descriptor_mapped_peak_bytes;
	out["canonical_template_hit_count"]                  = stats.canonical_template_hit_count;
	out["canonical_template_miss_count"]                 = stats.canonical_template_miss_count;
	out["canonical_template_sidecar_bytes"]              = stats.canonical_template_sidecar_bytes;
	out["canonical_template_audit_digest"]               = stats.canonical_template_audit_digest;
	out["canonical_template_load_ms"]                    = stats.canonical_template_load_ms;
	out["canonical_template_validation_ms"]              = stats.canonical_template_validation_ms;
	out["sparse_recipe_reader_hit_count"]                 = stats.sparse_recipe_reader_hit_count;
	out["sparse_recipe_reader_miss_count"]                = stats.sparse_recipe_reader_miss_count;
	out["sparse_recipe_rowgroup_hit_count"]               = stats.sparse_recipe_rowgroup_hit_count;
	out["sparse_recipe_rowgroup_miss_count"]              = stats.sparse_recipe_rowgroup_miss_count;
	out["sparse_recipe_sidecar_bytes"]                    = stats.sparse_recipe_sidecar_bytes;
	out["sparse_recipe_record_count"]                     = stats.sparse_recipe_record_count;
	out["sparse_recipe_source_metadata_bytes"]            = stats.sparse_recipe_source_metadata_bytes;
	out["sparse_recipe_source_metadata_pread_count"]      = stats.sparse_recipe_source_metadata_pread_count;
	out["sparse_descriptor_open_ms"]                      = stats.sparse_descriptor_open_ms;
	out["sparse_source_validation_ms"]                    = stats.sparse_source_validation_ms;
	out["sparse_access_index_build_ms"]                   = stats.sparse_access_index_build_ms;
	out["sparse_recipe_load_ms"]                          = stats.sparse_recipe_load_ms;
	out["sparse_recipe_validation_ms"]                    = stats.sparse_recipe_validation_ms;
	out["sparse_recipe_lookup_ms"]                        = stats.sparse_recipe_lookup_ms;
	out["sparse_recipe_rehydrate_ms"]                     = stats.sparse_recipe_rehydrate_ms;
	out["sparse_recipe_rehydrate_service_ms"]             = stats.sparse_recipe_rehydrate_service_ms;
	out["sparse_recipe_rehydrate_workers"]                = stats.sparse_recipe_rehydrate_workers;
	out["sparse_endpoint_resolution_ms"]                  = stats.sparse_endpoint_resolution_ms;
	out["sparse_range_gather_ms"]                         = stats.sparse_range_gather_ms;
	out["sparse_range_sort_exact_coalesce_ms"]            = stats.sparse_range_sort_exact_coalesce_ms;
	out["host_io_staging_ms"]                            = stats.host_io_staging_ms;
	out["host_io_staged_rowgroups"]                      = stats.host_io_staged_rowgroups;
	out["workset_build_ms"]                              = stats.workset_build_ms;
	out["workset_upload_ms"]                             = stats.workset_upload_ms;
	out["workset_upload_prep_ms"]                        = stats.workset_upload_prep_ms;
	out["workset_upload_arena_ms"]                       = stats.workset_upload_arena_ms;
	out["workset_upload_arena_pack_ms"]                  = stats.workset_upload_arena_pack_ms;
	out["workset_upload_arena_layout_ms"]                = stats.workset_upload_arena_layout_ms;
	out["workset_upload_arena_alloc_ms"]                 = stats.workset_upload_arena_alloc_ms;
	out["workset_upload_arena_resolve_ms"]               = stats.workset_upload_arena_resolve_ms;
	out["workset_upload_dma_issue_ms"]                   = stats.workset_upload_dma_issue_ms;
	out["workset_upload_event_record_ms"]                = stats.workset_upload_event_record_ms;
	out["workset_upload_dma_bytes"]                      = stats.workset_upload_dma_bytes;
	out["workset_upload_dma_count"]                      = stats.workset_upload_dma_count;
	out["decode_ms"]                                     = stats.decode_ms;
	out["gather_ms"]                                     = stats.gather_ms;
	out["decoded_gather_ms"]                             = stats.decoded_gather_ms;
	out["cached_gather_ms"]                              = stats.cached_gather_ms;
	out["projection_ms"]                                 = stats.projection_ms;
	out["decoded_projection_ms"]                         = stats.decoded_projection_ms;
	out["projection_item_build_ms"]                      = stats.projection_item_build_ms;
	out["fixed_transform_ms"]                            = stats.fixed_transform_ms;
	out["fixed_grid_round_ms"]                           = stats.fixed_grid_round_ms;
	out["resize_weight_build_ms"]                        = stats.resize_weight_build_ms;
	out["dct_resize_weight_cache_hits"]                  = stats.dct_resize_weight_cache_hits;
	out["dct_resize_weight_cache_misses"]                = stats.dct_resize_weight_cache_misses;
	out["dct_conversion_matrix_cache_hits"]              = stats.dct_conversion_matrix_cache_hits;
	out["dct_conversion_matrix_cache_misses"]            = stats.dct_conversion_matrix_cache_misses;
	out["prefetch_wait_ms"]                              = stats.prefetch_wait_ms;
	out["prefetch_depth_block_ms"]                       = stats.prefetch_depth_block_ms;
	out["prefetch_queue_start_ms"]                       = stats.prefetch_queue_start_ms;
	out["prefetch_rowgroup_read_ms"]                     = stats.prefetch_rowgroup_read_ms;
	out["prefetch_ready_ahead_ms"]                       = stats.prefetch_ready_ahead_ms;
	out["sync_rowgroup_read_ms"]                         = stats.sync_rowgroup_read_ms;
	out["compact_read_group_planning_ms"]                = stats.compact_read_group_planning_ms;
	out["column_binding_ms"]                            = stats.column_binding_ms;
	out["column_binding_expression_scan_count"]         = stats.column_binding_expression_scan_count;
	out["column_binding_rowgroup_count"]                = stats.column_binding_rowgroup_count;
	out["runtime_policy_decision"]                       = stats.runtime_policy_decision;
	out["runtime_policy_reason"]                         = stats.runtime_policy_reason;
	out["cache_enabled"]                                 = stats.cache_enabled;
	return out;
}

int64_t checked_int64(const uint64_t value, const char* field_name) {
	if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
		throw std::overflow_error(std::string(field_name) + " does not fit in torch.int64");
	}
	return static_cast<int64_t>(value);
}

template <typename FillFn>
torch::Tensor make_cuda_int64_tensor(const size_t count, const c10::DeviceIndex device_index, FillFn fill) {
	if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
		throw std::overflow_error("metadata tensor length does not fit in torch.int64");
	}
	auto cpu = torch::empty({static_cast<int64_t>(count)},
	                        torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
	auto* data = cpu.data_ptr<int64_t>();
	for (size_t index = 0; index < count; ++index) {
		data[index] = fill(index);
	}
	auto out = torch::empty({static_cast<int64_t>(count)},
	                        torch::TensorOptions().dtype(torch::kInt64).device(torch::Device(torch::kCUDA, device_index)));
	if (count != 0) {
		out.copy_(cpu, /*non_blocking=*/false);
	}
	return out;
}

py::list image_layouts_to_list(const std::vector<galp::jpeg::JpegDctDeviceImageLayout>& layouts) {
	py::list out;
	for (const auto& layout : layouts) {
		out.append(image_layout_to_dict(layout));
	}
	return out;
}

py::list block_metadata_to_list(const std::vector<galp::jpeg::JpegDctDeviceBlockMetadata>& blocks) {
	py::list out;
	for (const auto& block : blocks) {
		out.append(block_metadata_to_dict(block));
	}
	return out;
}

py::list rowgroups_to_list(const std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata>& rowgroups) {
	py::list out;
	for (const auto& rowgroup : rowgroups) {
		out.append(rowgroup_metadata_to_dict(rowgroup));
	}
	return out;
}

class DeferredDirectDctBatchReleaseQueue {
public:
	static DeferredDirectDctBatchReleaseQueue& instance() {
		static DeferredDirectDctBatchReleaseQueue queue;
		return queue;
	}

	void defer(std::shared_ptr<galp::jpeg::DirectDctBatch> owner, const c10::DeviceIndex device_index) noexcept {
		if (!owner) {
			return;
		}
		try {
			const auto stream = c10::cuda::getCurrentCUDAStream(device_index);
			defer_on_stream(std::move(owner), stream);
		} catch (const std::exception& e) {
			std::fprintf(stderr,
			             "GALP direct-DCT PyTorch tensor deleter: failed to defer CUDA release; "
			             "waiting for the current stream before releasing the batch: %s\n",
			             e.what());
			synchronize_current_stream(device_index);
			owner.reset();
		} catch (...) {
			std::fprintf(stderr,
			             "GALP direct-DCT PyTorch tensor deleter: failed to defer CUDA release; "
			             "waiting for the current stream before releasing the batch.\n");
			synchronize_current_stream(device_index);
			owner.reset();
		}
	}

	void defer_on_stream(std::shared_ptr<galp::jpeg::DirectDctBatch> owner,
	                     const c10::cuda::CUDAStream                  stream) noexcept {
		if (!owner) {
			return;
		}
		try {
			at::cuda::CUDAEvent ready(cudaEventDisableTiming);
			ready.record(stream);
			{
				std::lock_guard<std::mutex> lock(mutex_);
				pending_.emplace_back();
				pending_.back().ready = std::move(ready);
				pending_.back().owner = std::move(owner);
			}
			reclaim_finished();
		} catch (const std::exception& e) {
			std::fprintf(stderr,
			             "GALP direct-DCT PyTorch tensor deleter: failed to defer registered-stream release; "
			             "waiting for that stream before releasing the batch: %s\n",
			             e.what());
			synchronize_stream(stream);
			owner.reset();
		} catch (...) {
			std::fprintf(stderr,
			             "GALP direct-DCT PyTorch tensor deleter: failed to defer registered-stream release; "
			             "waiting for that stream before releasing the batch.\n");
			synchronize_stream(stream);
			owner.reset();
		}
	}

	size_t reclaim_finished() noexcept {
		std::vector<std::shared_ptr<galp::jpeg::DirectDctBatch>> ready;
		try {
			std::lock_guard<std::mutex> lock(mutex_);
			for (auto it = pending_.begin(); it != pending_.end();) {
				if (it->ready.query()) {
					ready.push_back(std::move(it->owner));
					it = pending_.erase(it);
				} else {
					++it;
				}
			}
		} catch (const std::exception& e) {
			std::fprintf(
			    stderr, "GALP direct-DCT PyTorch tensor deleter: deferred release query failed: %s\n", e.what());
		} catch (...) {
			std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: deferred release query failed.\n");
		}
		return ready.size();
	}

	~DeferredDirectDctBatchReleaseQueue() {
		std::deque<PendingRelease> pending;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			pending.swap(pending_);
		}
		for (auto& item : pending) {
			try {
				item.ready.synchronize();
			} catch (const std::exception& e) {
				std::fprintf(stderr,
				             "GALP direct-DCT PyTorch tensor deleter: deferred release shutdown wait failed: %s\n",
				             e.what());
			} catch (...) {
				std::fprintf(stderr,
				             "GALP direct-DCT PyTorch tensor deleter: deferred release shutdown wait failed.\n");
			}
			item.owner.reset();
		}
	}

private:
	struct PendingRelease {
		at::cuda::CUDAEvent                         ready;
		std::shared_ptr<galp::jpeg::DirectDctBatch> owner;
	};

	static void synchronize_current_stream(const c10::DeviceIndex device_index) noexcept {
		try {
			c10::cuda::getCurrentCUDAStream(device_index).synchronize();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: current stream wait failed: %s\n", e.what());
		} catch (...) { std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: current stream wait failed.\n"); }
	}

	static void synchronize_stream(const c10::cuda::CUDAStream stream) noexcept {
		try {
			stream.synchronize();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: registered stream wait failed: %s\n", e.what());
		} catch (...) { std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: registered stream wait failed.\n"); }
	}

	std::mutex                 mutex_;
	std::deque<PendingRelease> pending_;
};

class TorchDirectDctConsumerStreams {
public:
	void register_current(const c10::DeviceIndex device_index) {
		const auto stream = c10::cuda::getCurrentCUDAStream(device_index);
		std::lock_guard<std::mutex> lock(mutex_);
		for (const auto& registered : streams_) {
			if (registered.device_index() == stream.device_index() && registered.stream() == stream.stream()) {
				return;
			}
		}
		streams_.push_back(stream);
	}

	void defer(std::shared_ptr<galp::jpeg::DirectDctBatch> owner,
	           const c10::DeviceIndex                      device_index) const noexcept {
		std::vector<c10::cuda::CUDAStream> streams;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			streams = streams_;
		}
		auto& queue = DeferredDirectDctBatchReleaseQueue::instance();
		if (streams.empty()) {
			queue.defer(std::move(owner), device_index);
			return;
		}
		for (const auto& stream : streams) {
			queue.defer_on_stream(owner, stream);
		}
		owner.reset();
	}

private:
	mutable std::mutex                 mutex_;
	std::vector<c10::cuda::CUDAStream> streams_;
};

struct TorchDirectDctPrefetchTelemetry {
	std::atomic<bool>    started {false};
	// 0=queued, 1=active, 2=finished, 3=cancelled-before-active.
	std::atomic<int>     state {0};
	std::chrono::steady_clock::time_point submitted_at = std::chrono::steady_clock::now();
	std::atomic<int64_t> producer_active_nanoseconds {0};
	std::atomic<int64_t> planning_nanoseconds {0};
	std::atomic<int64_t> io_staging_nanoseconds {0};
	std::atomic<int64_t> ordered_submission_nanoseconds {0};
	std::atomic<int64_t> submit_to_ready_nanoseconds {0};
};

struct TorchDirectDctBatch {
	explicit TorchDirectDctBatch(galp::jpeg::DirectDctBatch batch_in)
	    : batch(std::make_shared<galp::jpeg::DirectDctBatch>(std::move(batch_in))),
	      consumer_streams(std::make_shared<TorchDirectDctConsumerStreams>()) {
	}

	torch::Tensor coefficients() {
		DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
		wait_for_batch_completion();
		register_current_consumer_stream();
		if (tensor.defined()) {
			return tensor;
		}
		const auto desc         = batch->tensor_async();
		const auto device_index = static_cast<c10::DeviceIndex>(desc.cuda_device < 0 ? 0 : desc.cuda_device);
		auto options = torch::TensorOptions().dtype(torch::kInt16).device(torch::Device(torch::kCUDA, device_index));
		if (desc.data == nullptr || desc.empty()) {
			tensor = torch::empty({static_cast<int64_t>(desc.rows()), static_cast<int64_t>(desc.columns())}, options);
			return tensor;
		}
		auto owner = batch;
		auto streams = consumer_streams;
		tensor     = torch::from_blob(
            const_cast<int16_t*>(desc.data),
            {static_cast<int64_t>(desc.rows()), static_cast<int64_t>(desc.columns())},
            {static_cast<int64_t>(desc.strides[0]), static_cast<int64_t>(desc.strides[1])},
			[owner = std::move(owner), streams = std::move(streams), device_index](void*) mutable {
				streams->defer(std::move(owner), device_index);
			},
            options);
		return tensor;
	}

	torch::Tensor y() {
		wait_for_batch_completion();
		register_current_consumer_stream();
		return grid_tensor(batch->y_tensor_async(), y_tensor);
	}

	torch::Tensor cbcr() {
		wait_for_batch_completion();
		register_current_consumer_stream();
		return grid_tensor(batch->cbcr_tensor_async(), cbcr_tensor);
	}

	torch::Tensor image_offsets_tensor() {
		if (image_offsets_tensor_cache.defined()) {
			return image_offsets_tensor_cache;
		}
		const auto& layouts = batch->image_layouts();
		image_offsets_tensor_cache =
		    make_cuda_int64_tensor(layouts.size(), tensor_device_index(), [&layouts](const size_t index) {
			    return checked_int64(layouts[index].block_offset, "image_layout.block_offset");
		    });
		return image_offsets_tensor_cache;
	}

	torch::Tensor image_counts_tensor() {
		if (image_counts_tensor_cache.defined()) {
			return image_counts_tensor_cache;
		}
		const auto& layouts = batch->image_layouts();
		image_counts_tensor_cache =
		    make_cuda_int64_tensor(layouts.size(), tensor_device_index(), [&layouts](const size_t index) {
			    return static_cast<int64_t>(layouts[index].block_count);
		    });
		return image_counts_tensor_cache;
	}

	torch::Tensor block_to_image_tensor() {
		if (block_to_image_tensor_cache.defined()) {
			return block_to_image_tensor_cache;
		}
		const auto counts = image_counts_tensor();
		const auto arange = torch::arange(static_cast<int64_t>(counts.numel()),
		                                  torch::TensorOptions()
		                                      .dtype(torch::kInt64)
		                                      .device(torch::Device(torch::kCUDA, tensor_device_index())));
		block_to_image_tensor_cache = torch::repeat_interleave(arange, counts);
		return block_to_image_tensor_cache;
	}

	[[nodiscard]] py::list image_layouts() const {
		return image_layouts_to_list(batch->image_layouts());
	}

	[[nodiscard]] py::list block_metadata() const {
		return block_metadata_to_list(batch->block_metadata());
	}

	[[nodiscard]] py::list rowgroups() const {
		return rowgroups_to_list(batch->rowgroups());
	}

	[[nodiscard]] py::dict cache_stats() const {
		return cache_stats_to_dict(batch->cache_stats());
	}

	[[nodiscard]] py::dict execution_stats() const {
		return execution_stats_to_dict(batch->execution_stats());
	}

	[[nodiscard]] py::dict execution_stats_snapshot() const {
		// Runtime benchmarks need the host-populated counters without turning
		// every training step into a cudaEventSynchronize. Completion-event-derived
		// decode, fixed-transform, and rounding durations remain optional in this
		// explicitly non-blocking snapshot; execution_stats() finalizes them after
		// the consumer has already waited for batch completion.
		return execution_stats_to_dict(batch->execution_stats_ref());
	}

	[[nodiscard]] py::dict metrics() const {
		// Keep the hot input path non-blocking. Completion-derived durations are
		// finalized only when the native CUDA event is already ready; callers can
		// read the same stable schema again after model completion.
		auto       stats    = batch->execution_stats_ref();
		auto*      event    = batch->cuda_completion_event();
		bool       complete = event == nullptr;
		if (event != nullptr) {
			const auto status = cudaEventQuery(static_cast<cudaEvent_t>(event));
			if (status == cudaSuccess) {
				complete = true;
			} else if (status != cudaErrorNotReady) {
				C10_CUDA_CHECK(status);
			}
		}
		if (complete) {
			stats = batch->execution_stats();
		}
		py::dict   out;
		out["schema"]               = "galp-direct-dct-metrics-v2";
		out["complete"]             = complete;
		out["consumer_wait_ms"]     = consumer_wait_ms;
		out["submit_to_ready_ms"]   = submit_to_ready_ms;
		out["producer_ms"]          = prefetch_telemetry
		                                    ? static_cast<double>(prefetch_telemetry->producer_active_nanoseconds.load(
		                                          std::memory_order_acquire)) /
		                                          1.0e6
		                                    : 0.0;
		out["planning_ms"]          = prefetch_telemetry
		                                    ? static_cast<double>(prefetch_telemetry->planning_nanoseconds.load(
		                                          std::memory_order_acquire)) /
		                                          1.0e6
		                                    : stats.planning_ms;
		out["io_ms"]                = prefetch_telemetry
		                                    ? static_cast<double>(prefetch_telemetry->io_staging_nanoseconds.load(
		                                          std::memory_order_acquire)) /
		                                          1.0e6
		                                    : stats.host_io_staging_ms;
		out["decode_ms"]            = stats.decode_ms;
		out["transform_ms"]         = stats.fixed_transform_ms + stats.fixed_grid_round_ms;
		out["logical_bytes"]        = stats.selected_compressed_payload_bytes;
		out["physical_bytes"]       = stats.compressed_payload_bytes_read;
		out["peak_transient_bytes"] = stats.actual_transient_total_allocated_high_water_bytes;
		return out;
	}

	[[nodiscard]] size_t cache_hits() const noexcept {
		return batch->cache_stats_ref().hits;
	}

	[[nodiscard]] size_t cache_misses() const noexcept {
		return batch->cache_stats_ref().misses;
	}

	[[nodiscard]] size_t cache_inserts() const noexcept {
		return batch->cache_stats_ref().inserts;
	}

	[[nodiscard]] size_t cache_evictions() const noexcept {
		return batch->cache_stats_ref().evictions;
	}

	[[nodiscard]] size_t planned_selected_vector_count() const noexcept {
		return batch->execution_stats_ref().planned_selected_vector_count;
	}

	[[nodiscard]] size_t selected_vector_count() const noexcept {
		return batch->execution_stats_ref().selected_vector_count;
	}

	[[nodiscard]] size_t full_vector_count() const noexcept {
		return batch->execution_stats_ref().full_vector_count;
	}

	[[nodiscard]] size_t actual_saved_vector_count() const noexcept {
		return batch->execution_stats_ref().actual_saved_vector_count;
	}

	[[nodiscard]] size_t rowgroup_count() const noexcept {
		return batch->execution_stats_ref().rowgroup_count;
	}

	[[nodiscard]] size_t workset_count() const noexcept {
		return batch->execution_stats_ref().workset_count;
	}

	[[nodiscard]] size_t decode_kernel_launch_count() const noexcept {
		return batch->execution_stats_ref().decode_kernel_launch_count;
	}

	[[nodiscard]] size_t gather_kernel_launch_count() const noexcept {
		return batch->execution_stats_ref().gather_kernel_launch_count;
	}

	[[nodiscard]] size_t projection_item_count() const noexcept {
		return batch->execution_stats_ref().projection_item_count;
	}

	[[nodiscard]] size_t decoded_projection_item_count() const noexcept {
		return batch->execution_stats_ref().decoded_projection_item_count;
	}

	[[nodiscard]] size_t fixed_transform_item_count() const noexcept {
		return batch->execution_stats_ref().fixed_transform_item_count;
	}

	[[nodiscard]] size_t fixed_transform_image_count() const noexcept {
		return batch->execution_stats_ref().fixed_transform_image_count;
	}

	[[nodiscard]] size_t internal_sync_count() const noexcept {
		return batch->execution_stats_ref().internal_sync_count;
	}

	[[nodiscard]] double planning_ms() const noexcept {
		return batch->execution_stats_ref().planning_ms;
	}

	[[nodiscard]] double decode_ms() const noexcept {
		return batch->execution_stats_ref().decode_ms;
	}

	[[nodiscard]] double gather_ms() const noexcept {
		return batch->execution_stats_ref().gather_ms;
	}

	[[nodiscard]] double projection_ms() const noexcept {
		return batch->execution_stats_ref().projection_ms;
	}

	[[nodiscard]] double prefetch_wait_ms() const noexcept {
		return batch->execution_stats_ref().prefetch_wait_ms;
	}

	[[nodiscard]] double sync_rowgroup_read_ms() const noexcept {
		return batch->execution_stats_ref().sync_rowgroup_read_ms;
	}

	[[nodiscard]] uintptr_t device_data_ptr() const {
		return reinterpret_cast<uintptr_t>(batch->device_data());
	}

	[[nodiscard]] uintptr_t y_device_data_ptr() const noexcept {
		return batch->device_batch().grid_output_data_type() == galp::jpeg::JpegDctGridOutputDataType::kFloat32
		           ? reinterpret_cast<uintptr_t>(batch->y_float_device_data_async())
		           : reinterpret_cast<uintptr_t>(batch->y_device_data_async());
	}

	[[nodiscard]] uintptr_t cbcr_device_data_ptr() const noexcept {
		return batch->device_batch().grid_output_data_type() == galp::jpeg::JpegDctGridOutputDataType::kFloat32
		           ? reinterpret_cast<uintptr_t>(batch->cbcr_float_device_data_async())
		           : reinterpret_cast<uintptr_t>(batch->cbcr_device_data_async());
	}

	[[nodiscard]] std::string layout() const {
		return layout_to_string(batch->device_batch().layout());
	}

	std::shared_ptr<galp::jpeg::DirectDctBatch> batch;
	std::shared_ptr<TorchDirectDctConsumerStreams> consumer_streams;
	torch::Tensor                               tensor;
	torch::Tensor                               y_tensor;
	torch::Tensor                               cbcr_tensor;
	torch::Tensor                               image_offsets_tensor_cache;
	torch::Tensor                               image_counts_tensor_cache;
	torch::Tensor                               block_to_image_tensor_cache;
	std::shared_ptr<TorchDirectDctPrefetchTelemetry> prefetch_telemetry;
	double                                      consumer_wait_ms = 0.0;
	double                                      submit_to_ready_ms = 0.0;

private:
	[[nodiscard]] c10::DeviceIndex tensor_device_index() const noexcept {
		const auto cuda_device = batch->cuda_device();
		return static_cast<c10::DeviceIndex>(cuda_device < 0 ? 0 : cuda_device);
	}

	torch::Tensor grid_tensor(const galp::jpeg::DirectDctGridTensorDescriptor& desc, torch::Tensor& cached) {
		DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
		if (cached.defined()) {
			return cached;
		}
		const auto device_index = static_cast<c10::DeviceIndex>(desc.cuda_device < 0 ? 0 : desc.cuda_device);
		const auto scalar_type = desc.dtype == galp::jpeg::DirectDctTensorDataType::kFloat32 ? torch::kFloat32
		                                                                                     : torch::kInt16;
		auto options = torch::TensorOptions().dtype(scalar_type).device(torch::Device(torch::kCUDA, device_index));
		std::vector<int64_t> shape;
		std::vector<int64_t> strides;
		shape.reserve(desc.shape.size());
		strides.reserve(desc.strides.size());
		for (const auto dim : desc.shape) {
			shape.push_back(static_cast<int64_t>(dim));
		}
		for (const auto stride : desc.strides) {
			strides.push_back(static_cast<int64_t>(stride));
		}
		if (desc.raw_data() == nullptr || desc.empty()) {
			cached = torch::empty(shape, options);
			return cached;
		}
		auto owner = batch;
		auto streams = consumer_streams;
		cached     = torch::from_blob(const_cast<void*>(desc.raw_data()),
                                  shape,
                                  strides,
		                          [owner = std::move(owner), streams = std::move(streams), device_index](void*) mutable {
		                              streams->defer(std::move(owner), device_index);
		                          },
                                  options);
		return cached;
	}

	void register_current_consumer_stream() const {
		consumer_streams->register_current(tensor_device_index());
	}

	void wait_for_batch_completion() const {
		auto* event = batch->cuda_completion_event();
		if (event == nullptr) {
			return;
		}
		const auto device_index = tensor_device_index();
		c10::cuda::CUDAGuard guard(device_index);
		auto stream = c10::cuda::getCurrentCUDAStream(device_index);
		C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), static_cast<cudaEvent_t>(event), 0));
	}
};

struct TorchDirectDctReaderState {
	explicit TorchDirectDctReaderState(const std::string& manifest_path)
	    : runtime(manifest_path) {
	}

	std::mutex                   runtime_mutex;
	std::mutex                   prefetch_mutex;
	std::shared_future<void>     prefetch_tail;
	galp::jpeg::DirectDctRuntime runtime;
};

TorchDirectDctBatch read_batch_from_state(const std::shared_ptr<TorchDirectDctReaderState>& state,
	                                      const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests,
	                                      const galp::jpeg::JpegDctDeviceBatchOptions&      options) {
	auto& release_queue = DeferredDirectDctBatchReleaseQueue::instance();
	release_queue.reclaim_finished();
	auto prepared = state->runtime.PrepareBatch(requests, options);
	state->runtime.StageBatchIo(prepared);
	galp::jpeg::DirectDctBatch raw_batch;
	{
		std::lock_guard<std::mutex> lock(state->runtime_mutex);
		raw_batch = state->runtime.ReadPreparedBatch(std::move(prepared));
	}
	release_queue.reclaim_finished();
	return TorchDirectDctBatch(std::move(raw_batch));
}

class TorchDirectDctPrefetch {
public:
	explicit TorchDirectDctPrefetch(std::future<TorchDirectDctBatch>                 future,
	                                std::shared_ptr<TorchDirectDctPrefetchTelemetry> telemetry,
	                                std::shared_ptr<galp::jpeg::JpegDctDeviceTransformSubmissionGate> submission_gate = nullptr)
	    : future_(std::move(future)), telemetry_(std::move(telemetry)), submission_gate_(std::move(submission_gate)),
	      submission_released_(submission_gate_ == nullptr) {
	}

	~TorchDirectDctPrefetch() {
		release_submission();
	}

	[[nodiscard]] bool ready() const {
		return future_.valid() && future_.wait_for(kImmediateFuturePollDuration) == std::future_status::ready;
	}

	[[nodiscard]] bool started() const noexcept {
		return telemetry_->started.load(std::memory_order_acquire);
	}

	[[nodiscard]] bool active() const noexcept {
		return telemetry_->state.load(std::memory_order_acquire) == 1;
	}

	[[nodiscard]] bool finished() const noexcept {
		const auto state = telemetry_->state.load(std::memory_order_acquire);
		return state == 2 || state == 3;
	}

	[[nodiscard]] double producer_active_ms() const noexcept {
		return static_cast<double>(
		           telemetry_->producer_active_nanoseconds.load(std::memory_order_acquire)) /
		       1.0e6;
	}

	[[nodiscard]] double planning_ms() const noexcept {
		return static_cast<double>(telemetry_->planning_nanoseconds.load(std::memory_order_acquire)) / 1.0e6;
	}

	[[nodiscard]] double io_staging_ms() const noexcept {
		return static_cast<double>(telemetry_->io_staging_nanoseconds.load(std::memory_order_acquire)) / 1.0e6;
	}

	[[nodiscard]] double ordered_submission_ms() const noexcept {
		return static_cast<double>(
		           telemetry_->ordered_submission_nanoseconds.load(std::memory_order_acquire)) /
		       1.0e6;
	}

	bool cancel() noexcept {
		release_submission();
		int queued = 0;
		return telemetry_->state.compare_exchange_strong(
		    queued, 3, std::memory_order_acq_rel, std::memory_order_acquire);
	}

	bool release_submission() noexcept {
		bool expected = false;
		if (!submission_released_.compare_exchange_strong(
		        expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
			return false;
		}
		if (submission_gate_) {
			submission_gate_->release();
		}
		return true;
	}

	TorchDirectDctBatch read() {
		if (!future_.valid()) {
			throw std::runtime_error("DirectDctPrefetch has already been consumed");
		}
		// A production consumer should not need to know about the deferred
		// arena-release queue.  Reclaim batches whose model stream has finished
		// before opening the next transform submission gate.
		DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
		release_submission();
		const auto wait_started = std::chrono::steady_clock::now();
		auto       batch        = future_.get();
		const auto ready_at     = std::chrono::steady_clock::now();
		batch.prefetch_telemetry = telemetry_;
		batch.consumer_wait_ms =
		    std::chrono::duration<double, std::milli>(ready_at - wait_started).count();
		batch.submit_to_ready_ms = static_cast<double>(
		                               telemetry_->submit_to_ready_nanoseconds.load(std::memory_order_acquire)) /
		                           1.0e6;
		DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
		return batch;
	}

private:
	std::future<TorchDirectDctBatch>                 future_;
	std::shared_ptr<TorchDirectDctPrefetchTelemetry> telemetry_;
	std::shared_ptr<galp::jpeg::JpegDctDeviceTransformSubmissionGate> submission_gate_;
	std::atomic<bool>                                submission_released_ {true};
};

std::shared_ptr<TorchDirectDctPrefetch> prefetch_batch_from_state(
    const std::shared_ptr<TorchDirectDctReaderState>&          state,
    std::vector<galp::jpeg::JpegDctImageCropRequest>           requests,
    galp::jpeg::JpegDctDeviceBatchOptions                      options) {
	const auto               device_index = c10::cuda::current_device();
	auto                     state_copy   = state;
	auto                     completion   = std::make_shared<std::promise<void>>();
	auto                     telemetry    = std::make_shared<TorchDirectDctPrefetchTelemetry>();
	auto submission_gate = options.async_planless_completion
	                           ? std::make_shared<galp::jpeg::JpegDctDeviceTransformSubmissionGate>()
	                           : nullptr;
	options.transform_submission_gate = submission_gate;
	std::shared_future<void> predecessor;
	{
		std::lock_guard<std::mutex> lock(state->prefetch_mutex);
		predecessor          = state->prefetch_tail;
		state->prefetch_tail = completion->get_future().share();
	}
	std::future<TorchDirectDctBatch> future;
	try {
		future = std::async(std::launch::async,
		                    [state     = std::move(state_copy),
		                     requests = std::move(requests),
		                     options,
		                     device_index,
		                     predecessor = std::move(predecessor),
		                     completion,
		                     telemetry]() mutable {
			                    telemetry->started.store(true, std::memory_order_release);
			                    try {
				                    int queued = 0;
				                    if (!telemetry->state.compare_exchange_strong(
				                            queued, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
					                    throw std::runtime_error("DirectDctPrefetch was cancelled before execution");
				                    }
				                    const auto active_begin   = std::chrono::steady_clock::now();
				                    const auto planning_begin = active_begin;
				                    auto prepared = state->runtime.PrepareBatch(requests, options);
				                    const auto planning_end = std::chrono::steady_clock::now();
				                    telemetry->planning_nanoseconds.store(
				                        std::chrono::duration_cast<std::chrono::nanoseconds>(planning_end - planning_begin)
				                            .count(),
				                        std::memory_order_release);
				                    const auto io_begin = std::chrono::steady_clock::now();
				                    state->runtime.StageBatchIo(prepared);
				                    const auto io_end = std::chrono::steady_clock::now();
				                    telemetry->io_staging_nanoseconds.store(
				                        std::chrono::duration_cast<std::chrono::nanoseconds>(io_end - io_begin).count(),
				                        std::memory_order_release);
				                    if (predecessor.valid()) {
					                    predecessor.wait();
				                    }
				                    const auto submission_begin = std::chrono::steady_clock::now();
				                    c10::cuda::CUDAGuard guard(device_index);
				                    auto& release_queue = DeferredDirectDctBatchReleaseQueue::instance();
				                    release_queue.reclaim_finished();
				                    galp::jpeg::DirectDctBatch raw_batch;
				                    {
					                    std::lock_guard<std::mutex> lock(state->runtime_mutex);
					                    raw_batch = state->runtime.ReadPreparedBatch(std::move(prepared));
				                    }
				                    release_queue.reclaim_finished();
				                    auto batch = TorchDirectDctBatch(std::move(raw_batch));
				                    const auto submission_end = std::chrono::steady_clock::now();
				                    telemetry->ordered_submission_nanoseconds.store(
				                        std::chrono::duration_cast<std::chrono::nanoseconds>(submission_end - submission_begin)
				                            .count(),
				                        std::memory_order_release);
				                    const auto active_end = std::chrono::steady_clock::now();
				                    telemetry->producer_active_nanoseconds.store(
				                        std::chrono::duration_cast<std::chrono::nanoseconds>(active_end - active_begin)
				                            .count(),
				                        std::memory_order_release);
				                    telemetry->submit_to_ready_nanoseconds.store(
				                        std::chrono::duration_cast<std::chrono::nanoseconds>(
				                            active_end - telemetry->submitted_at).count(),
				                        std::memory_order_release);
				                    telemetry->state.store(2, std::memory_order_release);
				                    completion->set_value();
				                    return batch;
			                    } catch (...) {
				                    if (telemetry->state.load(std::memory_order_acquire) != 3) {
					                    telemetry->state.store(2, std::memory_order_release);
				                    }
				                    try {
					                    completion->set_value();
				                    } catch (const std::future_error&) {}
				                    throw;
			                    }
		                    });
	} catch (...) {
		completion->set_value();
		throw;
	}
	return std::make_shared<TorchDirectDctPrefetch>(
	    std::move(future), std::move(telemetry), std::move(submission_gate));
}

class TorchDirectDctReader {
public:
	explicit TorchDirectDctReader(const std::string& manifest_path)
	{
		const auto started = std::chrono::steady_clock::now();
		state = std::make_shared<TorchDirectDctReaderState>(manifest_path);
		construction_ms_ = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - started).count();
	}

	[[nodiscard]] py::dict initialization_stats() const {
		auto out = reader_initialization_stats_to_dict(state->runtime.InitializationStats());
		out["direct_dct_reader_constructor_ms"] = construction_ms_;
		return out;
	}

	TorchDirectDctBatch read_batch(std::vector<galp::jpeg::JpegDctImageCropRequest> requests,
	                               const galp::jpeg::JpegDctDeviceBatchOptions& options) {
		return read_batch_from_state(state, requests, options);
	}

	py::dict plan_batch(const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests,
	                    const galp::jpeg::JpegDctDeviceBatchOptions& options) {
		return plan_preview_to_dict(state->runtime.PlanBatch(requests, options));
	}

	double prepare_batch_ms(const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests,
	                        const galp::jpeg::JpegDctDeviceBatchOptions& options) {
		const auto started  = std::chrono::steady_clock::now();
		auto       prepared = state->runtime.PrepareBatch(requests, options);
		const auto finished = std::chrono::steady_clock::now();
		if (prepared.empty()) {
			throw std::runtime_error("DirectDct prepare-only diagnostic produced an empty batch");
		}
		return std::chrono::duration<double, std::milli>(finished - started).count();
	}

	std::shared_ptr<TorchDirectDctPrefetch> prefetch_batch(
	                                                       std::vector<galp::jpeg::JpegDctImageCropRequest> requests,
	                                                       galp::jpeg::JpegDctDeviceBatchOptions options) {
		return prefetch_batch_from_state(state, std::move(requests), options);
	}

	TorchDirectDctBatch read_prefetched(const std::shared_ptr<TorchDirectDctPrefetch>& prefetch) {
		if (!prefetch) {
			throw std::invalid_argument("prefetch handle must not be None");
		}
		return prefetch->read();
	}

	size_t manual_reclaim() {
		return DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
	}

	[[nodiscard]] uint64_t image_count() const noexcept {
		return state->runtime.image_count();
	}

	[[nodiscard]] py::dict image_metadata(const uint32_t global_image_index) const {
		std::lock_guard<std::mutex> lock(state->runtime_mutex);
		return image_metadata_to_dict(state->runtime.ImageMetadata(global_image_index));
	}

	[[nodiscard]] uint64_t rowgroup_storage_bytes(const uint32_t               shard_id,
	                                              const std::vector<uint32_t>& rowgroup_indices) const {
		std::lock_guard<std::mutex> lock(state->runtime_mutex);
		return state->runtime.RowgroupStorageBytes(shard_id, rowgroup_indices);
	}

	[[nodiscard]] std::shared_ptr<TorchDirectDctReaderState> shared_state() const noexcept {
		return state;
	}

private:
	std::shared_ptr<TorchDirectDctReaderState> state;
	double construction_ms_ = 0.0;
};

class TorchDirectDctPipelineMetrics {
public:
	void reset() {
		pending_.clear();
		consumed_batches_ = 0;
		completed_batches_ = 0;
		consumer_wait_ms_ = 0.0;
		submit_to_ready_ms_ = 0.0;
		producer_ms_ = 0.0;
		planning_ms_ = 0.0;
		io_ms_ = 0.0;
		decode_ms_ = 0.0;
		transform_ms_ = 0.0;
		logical_bytes_ = 0;
		physical_bytes_ = 0;
		peak_transient_bytes_ = 0;
	}

	void observe(const TorchDirectDctBatch& batch) {
		++consumed_batches_;
		consumer_wait_ms_ += batch.consumer_wait_ms;
		submit_to_ready_ms_ += batch.submit_to_ready_ms;
		if (batch.prefetch_telemetry) {
			producer_ms_ += static_cast<double>(
			                    batch.prefetch_telemetry->producer_active_nanoseconds.load(std::memory_order_acquire)) /
			                1.0e6;
			planning_ms_ += static_cast<double>(
			                    batch.prefetch_telemetry->planning_nanoseconds.load(std::memory_order_acquire)) /
			                1.0e6;
			io_ms_ += static_cast<double>(
			              batch.prefetch_telemetry->io_staging_nanoseconds.load(std::memory_order_acquire)) /
			          1.0e6;
		}
		const auto& stats = batch.batch->execution_stats_ref();
		logical_bytes_ += stats.selected_compressed_payload_bytes;
		physical_bytes_ += stats.compressed_payload_bytes_read;
		peak_transient_bytes_ = std::max(
		    peak_transient_bytes_, stats.actual_transient_total_allocated_high_water_bytes);
		pending_.push_back(batch.batch);
		collect_ready();
	}

	[[nodiscard]] py::dict snapshot() {
		collect_ready();
		py::dict out;
		out["schema"]               = "galp-direct-dct-metrics-v2";
		out["complete"]             = pending_.empty();
		out["consumer_wait_ms"]     = consumer_wait_ms_;
		out["submit_to_ready_ms"]   = submit_to_ready_ms_;
		out["producer_ms"]          = producer_ms_;
		out["planning_ms"]          = planning_ms_;
		out["io_ms"]                = io_ms_;
		out["decode_ms"]            = decode_ms_;
		out["transform_ms"]         = transform_ms_;
		out["logical_bytes"]        = logical_bytes_;
		out["physical_bytes"]       = physical_bytes_;
		out["peak_transient_bytes"] = peak_transient_bytes_;
		out["consumed_batches"]     = consumed_batches_;
		out["completed_batches"]    = completed_batches_;
		return out;
	}

private:
	void collect_ready() {
		for (auto it = pending_.begin(); it != pending_.end();) {
			auto* event = (*it)->cuda_completion_event();
			bool  ready = event == nullptr;
			if (event != nullptr) {
				const auto status = cudaEventQuery(static_cast<cudaEvent_t>(event));
				if (status == cudaSuccess) {
					ready = true;
				} else if (status != cudaErrorNotReady) {
					C10_CUDA_CHECK(status);
				}
			}
			if (!ready) {
				++it;
				continue;
			}
			const auto stats = (*it)->execution_stats();
			decode_ms_ += stats.decode_ms;
			transform_ms_ += stats.fixed_transform_ms + stats.fixed_grid_round_ms;
			peak_transient_bytes_ = std::max(
			    peak_transient_bytes_, stats.actual_transient_total_allocated_high_water_bytes);
			++completed_batches_;
			it = pending_.erase(it);
		}
	}

	std::deque<std::shared_ptr<galp::jpeg::DirectDctBatch>> pending_;
	size_t consumed_batches_ = 0;
	size_t completed_batches_ = 0;
	double consumer_wait_ms_ = 0.0;
	double submit_to_ready_ms_ = 0.0;
	double producer_ms_ = 0.0;
	double planning_ms_ = 0.0;
	double io_ms_ = 0.0;
	double decode_ms_ = 0.0;
	double transform_ms_ = 0.0;
	size_t logical_bytes_ = 0;
	size_t physical_bytes_ = 0;
	size_t peak_transient_bytes_ = 0;
};

class TorchDirectDctPipeline {
public:
	TorchDirectDctPipeline(std::shared_ptr<TorchDirectDctReaderState> state,
	                       galp::jpeg::JpegDctDeviceBatchOptions     options)
	    : state_(std::move(state)), options_(std::move(options)) {
	}

	~TorchDirectDctPipeline() {
		close();
	}

	void reset(const std::vector<std::vector<uint32_t>>& image_id_batches,
	           const py::object&                         transforms_by_batch) {
		const bool has_transforms = !transforms_by_batch.is_none();
		py::list   transform_batches;
		if (has_transforms) {
			transform_batches = transforms_by_batch.cast<py::list>();
			if (static_cast<size_t>(transform_batches.size()) != image_id_batches.size()) {
				throw std::invalid_argument(
				    "transforms_by_batch must contain exactly one entry per image-id batch");
			}
		}
		std::vector<std::vector<galp::jpeg::JpegDctImageCropRequest>> parsed_requests;
		parsed_requests.reserve(image_id_batches.size());
		for (size_t index = 0; index < image_id_batches.size(); ++index) {
			if (image_id_batches[index].empty()) {
				throw std::invalid_argument("DirectDctPipeline batches must not be empty");
			}
			const py::object transforms = has_transforms
			                                  ? py::reinterpret_borrow<py::object>(transform_batches[index])
			                                  : py::none();
			parsed_requests.push_back(parse_transform_requests(
			    image_id_batches[index], transforms, galp::jpeg::JpegDctCropBox {}));
		}
		close();
		metrics_.reset();
		closed_   = false;
		requests_ = std::move(parsed_requests);
		fill_pending();
		if (!pending_.empty()) {
			pending_.front()->release_submission();
		}
	}

	TorchDirectDctBatch next() {
		if (closed_ || pending_.empty()) {
			throw py::stop_iteration();
		}
		auto current = pending_.front();
		current->release_submission();
		try {
			auto batch = current->read();
			last_prefetch_ = std::move(current);
			pending_.pop_front();
			fill_pending();
			metrics_.observe(batch);
			return batch;
		} catch (...) {
			close();
			throw;
		}
	}

	[[nodiscard]] bool ready() const {
		return !pending_.empty() && pending_.front()->ready();
	}

	[[nodiscard]] bool started() const noexcept {
		return !pending_.empty() && pending_.front()->started();
	}

	[[nodiscard]] py::dict prefetch_metrics() const {
		std::shared_ptr<TorchDirectDctPrefetch> source = last_prefetch_;
		if (!source && !pending_.empty()) {
			source = pending_.front();
		}
		py::dict out;
		out["producer_ms"]           = source ? source->producer_active_ms() : 0.0;
		out["planning_ms"]           = source ? source->planning_ms() : 0.0;
		out["io_ms"]                 = source ? source->io_staging_ms() : 0.0;
		out["ordered_submission_ms"] = source ? source->ordered_submission_ms() : 0.0;
		return out;
	}

	[[nodiscard]] size_t prefetched_batch_count() const noexcept {
		return next_request_;
	}

	[[nodiscard]] py::dict metrics() {
		return metrics_.snapshot();
	}

	size_t close() noexcept {
		size_t cancelled = 0;
		for (auto& pending : pending_) {
			cancelled += pending && pending->cancel() ? 1U : 0U;
		}
		pending_.clear();
		requests_.clear();
		last_prefetch_.reset();
		next_request_ = 0;
		closed_       = true;
		return cancelled;
	}

private:
	static constexpr size_t kNativePrefetchDepth = 2U;

	void fill_pending() {
		while (pending_.size() < kNativePrefetchDepth && next_request_ < requests_.size()) {
			pending_.push_back(prefetch_batch_from_state(
			    state_, std::move(requests_[next_request_]), options_));
			++next_request_;
		}
	}

	std::shared_ptr<TorchDirectDctReaderState> state_;
	galp::jpeg::JpegDctDeviceBatchOptions options_;
	std::vector<std::vector<galp::jpeg::JpegDctImageCropRequest>> requests_;
	std::deque<std::shared_ptr<TorchDirectDctPrefetch>> pending_;
	std::shared_ptr<TorchDirectDctPrefetch> last_prefetch_;
	TorchDirectDctPipelineMetrics metrics_;
	size_t next_request_ = 0;
	bool   closed_       = true;
};

} // namespace

PYBIND11_MODULE(_galp_direct_dct, m) {
	m.doc() = "Private GALP Direct-DCT backend; applications must import galp.torch";

	py::class_<TorchDirectDctBatch>(m, "DirectDctBatch")
	    .def_property_readonly("coefficients", &TorchDirectDctBatch::coefficients)
	    .def_property_readonly("y", &TorchDirectDctBatch::y)
	    .def_property_readonly("cbcr", &TorchDirectDctBatch::cbcr)
	    .def_property_readonly("image_offsets_tensor", &TorchDirectDctBatch::image_offsets_tensor)
	    .def_property_readonly("image_counts_tensor", &TorchDirectDctBatch::image_counts_tensor)
	    .def_property_readonly("block_to_image_tensor", &TorchDirectDctBatch::block_to_image_tensor)
	    .def_property_readonly("image_layouts", &TorchDirectDctBatch::image_layouts)
	    .def_property_readonly("block_metadata", &TorchDirectDctBatch::block_metadata)
	    .def_property_readonly("rowgroups", &TorchDirectDctBatch::rowgroups)
	    .def_property_readonly("cache_stats", &TorchDirectDctBatch::cache_stats)
	    .def_property_readonly("execution_stats", &TorchDirectDctBatch::execution_stats)
	    .def_property_readonly("execution_stats_snapshot", &TorchDirectDctBatch::execution_stats_snapshot)
	    .def_property_readonly("metrics", &TorchDirectDctBatch::metrics)
	    .def_property_readonly("cache_hits", &TorchDirectDctBatch::cache_hits)
	    .def_property_readonly("cache_misses", &TorchDirectDctBatch::cache_misses)
	    .def_property_readonly("cache_inserts", &TorchDirectDctBatch::cache_inserts)
	    .def_property_readonly("cache_evictions", &TorchDirectDctBatch::cache_evictions)
	    .def_property_readonly("planned_selected_vector_count", &TorchDirectDctBatch::planned_selected_vector_count)
	    .def_property_readonly("selected_vector_count", &TorchDirectDctBatch::selected_vector_count)
	    .def_property_readonly("full_vector_count", &TorchDirectDctBatch::full_vector_count)
	    .def_property_readonly("actual_saved_vector_count", &TorchDirectDctBatch::actual_saved_vector_count)
	    .def_property_readonly("rowgroup_count", &TorchDirectDctBatch::rowgroup_count)
	    .def_property_readonly("workset_count", &TorchDirectDctBatch::workset_count)
	    .def_property_readonly("decode_kernel_launch_count", &TorchDirectDctBatch::decode_kernel_launch_count)
	    .def_property_readonly("gather_kernel_launch_count", &TorchDirectDctBatch::gather_kernel_launch_count)
	    .def_property_readonly("projection_item_count", &TorchDirectDctBatch::projection_item_count)
	    .def_property_readonly("decoded_projection_item_count", &TorchDirectDctBatch::decoded_projection_item_count)
	    .def_property_readonly("fixed_transform_item_count", &TorchDirectDctBatch::fixed_transform_item_count)
	    .def_property_readonly("fixed_transform_image_count", &TorchDirectDctBatch::fixed_transform_image_count)
	    .def_property_readonly("internal_sync_count", &TorchDirectDctBatch::internal_sync_count)
	    .def_property_readonly("planning_ms", &TorchDirectDctBatch::planning_ms)
	    .def_property_readonly("decode_ms", &TorchDirectDctBatch::decode_ms)
	    .def_property_readonly("gather_ms", &TorchDirectDctBatch::gather_ms)
	    .def_property_readonly("projection_ms", &TorchDirectDctBatch::projection_ms)
	    .def_property_readonly("prefetch_wait_ms", &TorchDirectDctBatch::prefetch_wait_ms)
	    .def_property_readonly("sync_rowgroup_read_ms", &TorchDirectDctBatch::sync_rowgroup_read_ms)
	    .def_property_readonly("layout", &TorchDirectDctBatch::layout)
	    .def_property_readonly("device_data_ptr", &TorchDirectDctBatch::device_data_ptr)
	    .def_property_readonly("y_device_data_ptr", &TorchDirectDctBatch::y_device_data_ptr)
	    .def_property_readonly("cbcr_device_data_ptr", &TorchDirectDctBatch::cbcr_device_data_ptr)
	    .def_property_readonly("global_image_ids",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->global_image_ids(); })
	    .def_property_readonly("transform_descriptors",
	                           [](const TorchDirectDctBatch& batch) {
		                           return transform_requests_to_python(batch.batch->transform_requests());
	                           })
	    .def_property_readonly("selected_coefficients",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->selected_coefficients(); })
	    .def_property_readonly("cuda_device",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->cuda_device(); })
	    .def_property_readonly("block_count",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->block_count(); })
	    .def_property_readonly("coefficients_per_block",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->coefficients_per_block(); })
	    .def_property_readonly("coefficient_count",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->coefficient_count(); })
	    .def_property_readonly("coefficient_bytes",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->coefficient_bytes(); })
	    .def_property_readonly("y_coefficient_count",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->y_coefficient_count(); })
	    .def_property_readonly("cbcr_coefficient_count",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->cbcr_coefficient_count(); });

	py::class_<TorchDirectDctPrefetch, std::shared_ptr<TorchDirectDctPrefetch>>(m, "DirectDctPrefetch")
	    .def_property_readonly("ready", &TorchDirectDctPrefetch::ready)
	    .def_property_readonly("started", &TorchDirectDctPrefetch::started)
	    .def_property_readonly("active", &TorchDirectDctPrefetch::active)
	    .def_property_readonly("finished", &TorchDirectDctPrefetch::finished)
	    .def_property_readonly("producer_active_ms", &TorchDirectDctPrefetch::producer_active_ms)
	    .def_property_readonly("planning_ms", &TorchDirectDctPrefetch::planning_ms)
	    .def_property_readonly("io_staging_ms", &TorchDirectDctPrefetch::io_staging_ms)
	    .def_property_readonly("ordered_submission_ms", &TorchDirectDctPrefetch::ordered_submission_ms)
	    .def("release_submission", &TorchDirectDctPrefetch::release_submission)
	    .def("cancel", &TorchDirectDctPrefetch::cancel)
	    .def("read",
	         &TorchDirectDctPrefetch::read,
	         "Consume the prefetch handle and return the batch; dropping an unread handle waits for the read to finish.",
	         py::call_guard<py::gil_scoped_release>());

	py::class_<TorchDirectDctPipeline, std::shared_ptr<TorchDirectDctPipeline>>(m, "DirectDctPipeline")
	    .def("reset",
	         &TorchDirectDctPipeline::reset,
	         py::arg("image_id_batches"),
	         py::arg("transforms_by_batch") = py::none(),
	         "Replace the logical batch schedule and start native-owned bounded prefetching.")
	    .def("__iter__", [](const std::shared_ptr<TorchDirectDctPipeline>& pipeline) { return pipeline; })
	    .def("__next__",
	         &TorchDirectDctPipeline::next,
	         py::call_guard<py::gil_scoped_release>())
	    .def_property_readonly("ready", &TorchDirectDctPipeline::ready)
	    .def_property_readonly("started", &TorchDirectDctPipeline::started)
	    .def_property_readonly("prefetched_batch_count", &TorchDirectDctPipeline::prefetched_batch_count)
	    .def_property_readonly("metrics", &TorchDirectDctPipeline::metrics)
	    .def_property_readonly("prefetch_metrics", &TorchDirectDctPipeline::prefetch_metrics)
	    .def("close", &TorchDirectDctPipeline::close);

	py::class_<TorchDirectDctReader>(m, "DirectDctReader")
	    .def(py::init<const std::string&>(), py::arg("manifest_path"))
	    .def_property_readonly("image_count", &TorchDirectDctReader::image_count)
	    .def_property_readonly("initialization_stats", &TorchDirectDctReader::initialization_stats)
	    .def(
	        "plan",
	        [](TorchDirectDctReader&        reader,
	           const std::vector<uint32_t>& image_ids,
	           const std::string&           profile_id,
	           const py::object&            transforms) {
		        const auto requests = parse_transform_requests(
		            image_ids, transforms, galp::jpeg::JpegDctCropBox {});
		        const auto profile = galp::profiles::resolve_direct_dct_profile(profile_id);
		        return reader.plan_batch(requests, galp::profiles::materialize_direct_dct_options(profile));
	        },
	        py::arg("image_ids"),
	        py::arg("profile_id"),
	        py::arg("transforms") = py::none(),
	        "Plan a registered semantic Direct-DCT profile using its native-owned runtime policy.")
	    .def(
	        "prefetch",
	        [](TorchDirectDctReader& reader,
	           std::vector<uint32_t> image_ids,
	           const std::string&    profile_id,
	           const py::object&     transforms) {
		        auto requests = parse_transform_requests(
		            image_ids, transforms, galp::jpeg::JpegDctCropBox {});
		        const auto profile = galp::profiles::resolve_direct_dct_profile(profile_id);
		        return reader.prefetch_batch(
		            std::move(requests), galp::profiles::materialize_direct_dct_options(profile));
	        },
	        py::arg("image_ids"),
	        py::arg("profile_id"),
	        py::arg("transforms") = py::none(),
	        "Prefetch a registered semantic Direct-DCT profile using its native-owned runtime policy.")
	    .def(
	        "read",
	        [](TorchDirectDctReader& reader,
	           std::vector<uint32_t> image_ids,
	           const std::string&    profile_id,
	           const py::object&     transforms) {
		        auto requests = parse_transform_requests(
		            image_ids, transforms, galp::jpeg::JpegDctCropBox {});
		        const auto profile = galp::profiles::resolve_direct_dct_profile(profile_id);
		        const auto options = galp::profiles::materialize_direct_dct_options(profile);
		        py::gil_scoped_release release;
		        return reader.read_batch(std::move(requests), options);
	        },
	        py::arg("image_ids"),
	        py::arg("profile_id"),
	        py::arg("transforms") = py::none(),
	        "Read a registered semantic Direct-DCT profile synchronously.")
	    .def(
	        "pipeline",
	        [](TorchDirectDctReader& reader, const std::string& profile_id) {
		        const auto profile = galp::profiles::resolve_direct_dct_profile(profile_id);
		        return std::make_shared<TorchDirectDctPipeline>(
		            reader.shared_state(), galp::profiles::materialize_direct_dct_options(profile));
	        },
	        py::arg("profile_id"),
	        "Create a native-owned bounded pipeline for one semantic profile.")
	    .def("image_metadata",
	         &TorchDirectDctReader::image_metadata,
	         py::arg("global_image_index"),
	         "Return JPEG metadata needed by adapters, including component slots and quantization tables.")
	    .def("rowgroup_storage_bytes",
	         &TorchDirectDctReader::rowgroup_storage_bytes,
	         py::arg("shard_id"),
	         py::arg("rowgroup_indices"),
	         "Return exact compressed FLS bytes for the requested rowgroup records.")
	    .def(
	        "plan_batch",
	        [](TorchDirectDctReader&        reader,
	           const std::vector<uint32_t>& image_ids,
	           const py::object&            crop,
	           const std::string&           dct_coeffs,
	           const size_t                 cache_capacity_mib,
	           const size_t                 decode_batch_rowgroups,
	           const bool                   enable_rowgroup_prefetch,
	           const size_t                 rowgroup_prefetch_depth,
	           const size_t                 rowgroup_prefetch_workers,
	           const size_t                 rowgroup_prefetch_min_decode_batches,
	           const std::string&           layout,
	           const py::object&            grid_transform,
	           const size_t                 plan_cache_capacity,
		           const bool                   enable_planless_execution,
		           const py::object&            transforms,
		           const std::string&           crop_execution_mode,
		           const size_t                 decode_workset_capacity_mib,
		           const double                 bounded_read_amplification_cap,
		           const double                 bounded_read_local_amplification_cap,
		           const size_t                 bounded_read_max_run_bytes) {
		        const auto crop_box = parse_crop(crop);
		        const auto requests = parse_transform_requests(image_ids, transforms, crop_box);
		        const auto options  = make_batch_options(dct_coeffs,
                                                        cache_capacity_mib,
                                                        decode_batch_rowgroups,
                                                        enable_rowgroup_prefetch,
                                                        rowgroup_prefetch_depth,
                                                        rowgroup_prefetch_workers,
                                                        rowgroup_prefetch_min_decode_batches,
                                                        layout,
                                                        grid_transform,
		                                                    plan_cache_capacity,
		                                                    enable_planless_execution,
		                                                    decode_workset_capacity_mib,
		                                                    crop_execution_mode,
		                                                    "fully-overlapped",
		                                                    0U,
		                                                    0U,
		                                                    false,
		                                                    "auto",
		                                                    false,
		                                                    bounded_read_amplification_cap,
		                                                    bounded_read_local_amplification_cap,
		                                                    bounded_read_max_run_bytes);
		        return reader.plan_batch(requests, options);
	        },
	        py::arg("image_ids"),
	        py::arg("crop")                      = py::none(),
	        py::arg("dct_coeffs")                = "all",
	        py::arg("cache_capacity_mib")        = kDefaultDirectDctCacheCapacityMiB,
	        py::arg("decode_batch_rowgroups")    = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups,
	        py::arg("enable_rowgroup_prefetch")  = true,
	        py::arg("rowgroup_prefetch_depth")   = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth,
	        py::arg("rowgroup_prefetch_workers") = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers,
	        py::arg("rowgroup_prefetch_min_decode_batches") =
	            galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches,
	        py::arg("layout")                    = "compact",
	        py::arg("grid_transform")            = py::none(),
	        py::arg("plan_cache_capacity")       = galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity,
	        py::arg("enable_planless_execution") = true,
	        py::arg("transforms")                = py::none(),
		        py::arg("crop_execution_mode")        = "auto",
	        py::arg("decode_workset_capacity_mib") = kDefaultDirectDctDecodeWorksetCapacityMiB,
	        py::arg("bounded_read_amplification_cap") = 1.0,
	        py::arg("bounded_read_local_amplification_cap") = 0.0,
	        py::arg("bounded_read_max_run_bytes") = 0U)
	    .def(
	        "prepare_batch_ms",
	        [](TorchDirectDctReader&        reader,
	           const std::vector<uint32_t>& image_ids,
	           const py::object&            crop,
	           const std::string&           dct_coeffs,
	           const size_t                 cache_capacity_mib,
	           const size_t                 decode_batch_rowgroups,
	           const bool                   enable_rowgroup_prefetch,
	           const size_t                 rowgroup_prefetch_depth,
	           const size_t                 rowgroup_prefetch_workers,
	           const size_t                 rowgroup_prefetch_min_decode_batches,
	           const std::string&           layout,
	           const py::object&            grid_transform,
	           const size_t                 plan_cache_capacity,
	           const bool                   enable_planless_execution,
	           const py::object&            transforms,
	           const std::string&           crop_execution_mode,
	           const size_t                 decode_workset_capacity_mib,
	           const double                 bounded_read_amplification_cap,
	           const double                 bounded_read_local_amplification_cap,
	           const size_t                 bounded_read_max_run_bytes) {
		        const auto crop_box = parse_crop(crop);
		        const auto requests = parse_transform_requests(image_ids, transforms, crop_box);
		        const auto options  = make_batch_options(dct_coeffs,
		                                                 cache_capacity_mib,
		                                                 decode_batch_rowgroups,
		                                                 enable_rowgroup_prefetch,
		                                                 rowgroup_prefetch_depth,
		                                                 rowgroup_prefetch_workers,
		                                                 rowgroup_prefetch_min_decode_batches,
		                                                 layout,
		                                                 grid_transform,
		                                                 plan_cache_capacity,
		                                                 enable_planless_execution,
		                                                 decode_workset_capacity_mib,
		                                                 crop_execution_mode,
		                                                 "fully-overlapped",
		                                                 0U,
		                                                 0U,
		                                                 false,
		                                                 "auto",
		                                                 false,
		                                                 bounded_read_amplification_cap,
		                                                 bounded_read_local_amplification_cap,
		                                                 bounded_read_max_run_bytes);
		        return reader.prepare_batch_ms(requests, options);
	        },
	        py::arg("image_ids"),
	        py::arg("crop")                      = py::none(),
	        py::arg("dct_coeffs")                = "all",
	        py::arg("cache_capacity_mib")        = kDefaultDirectDctCacheCapacityMiB,
	        py::arg("decode_batch_rowgroups")    = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups,
	        py::arg("enable_rowgroup_prefetch")  = true,
	        py::arg("rowgroup_prefetch_depth")   = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth,
	        py::arg("rowgroup_prefetch_workers") = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers,
	        py::arg("rowgroup_prefetch_min_decode_batches") =
	            galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches,
	        py::arg("layout")                    = "compact",
	        py::arg("grid_transform")            = py::none(),
	        py::arg("plan_cache_capacity")       = galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity,
	        py::arg("enable_planless_execution") = true,
	        py::arg("transforms")                = py::none(),
	        py::arg("crop_execution_mode")        = "auto",
	        py::arg("decode_workset_capacity_mib") = kDefaultDirectDctDecodeWorksetCapacityMiB,
	        py::arg("bounded_read_amplification_cap") = 1.0,
	        py::arg("bounded_read_local_amplification_cap") = 0.0,
	        py::arg("bounded_read_max_run_bytes") = 0U,
	        "Measure production PrepareBatch, including I/O-plan compilation, without staging storage or using CUDA.")
	    .def(
	        "read_batch",
	        [](TorchDirectDctReader&        reader,
	           const std::vector<uint32_t>& image_ids,
	           const py::object&            crop,
	           const std::string&           dct_coeffs,
	           const size_t                 cache_capacity_mib,
	           const size_t                 decode_batch_rowgroups,
	           const bool                   enable_rowgroup_prefetch,
	           const size_t                 rowgroup_prefetch_depth,
	           const size_t                 rowgroup_prefetch_workers,
	           const size_t                 rowgroup_prefetch_min_decode_batches,
	           const std::string&           layout,
	           const py::object&            grid_transform,
	           const size_t                 plan_cache_capacity,
	           const bool                   enable_planless_execution,
	           const std::string&           scheduling_policy,
	           const size_t                 transform_blocks_per_launch,
	           const size_t                 transform_ctas_per_launch,
		           const bool                   use_low_priority_streams,
		           const py::object&            transforms,
		           const std::string&           crop_execution_mode,
		           const size_t                 decode_workset_capacity_mib,
		           const std::string&           block_major_double_buffer,
		           const double                 bounded_read_amplification_cap,
		           const double                 bounded_read_local_amplification_cap,
		           const size_t                 bounded_read_max_run_bytes) {
		        const auto             crop_box = parse_crop(crop);
		        auto                   requests = parse_transform_requests(image_ids, transforms, crop_box);
		        const auto             options  = make_batch_options(dct_coeffs,
                                                        cache_capacity_mib,
                                                        decode_batch_rowgroups,
                                                        enable_rowgroup_prefetch,
                                                        rowgroup_prefetch_depth,
                                                        rowgroup_prefetch_workers,
                                                        rowgroup_prefetch_min_decode_batches,
                                                        layout,
                                                        grid_transform,
		                                                    plan_cache_capacity,
		                                                    enable_planless_execution,
		                                                    decode_workset_capacity_mib,
		                                                    crop_execution_mode,
	                                                        scheduling_policy,
                                                        transform_blocks_per_launch,
                                                        transform_ctas_per_launch,
	                                                        use_low_priority_streams,
	                                                        block_major_double_buffer,
	                                                        /*async_planless_completion=*/false,
	                                                        bounded_read_amplification_cap,
	                                                        bounded_read_local_amplification_cap,
	                                                        bounded_read_max_run_bytes);
		        py::gil_scoped_release release;
		        return reader.read_batch(std::move(requests), options);
	        },
	        py::arg("image_ids"),
	        py::arg("crop")                      = py::none(),
	        py::arg("dct_coeffs")                = "all",
	        py::arg("cache_capacity_mib")        = kDefaultDirectDctCacheCapacityMiB,
	        py::arg("decode_batch_rowgroups")    = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups,
	        py::arg("enable_rowgroup_prefetch")  = true,
	        py::arg("rowgroup_prefetch_depth")   = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth,
	        py::arg("rowgroup_prefetch_workers") = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers,
	        py::arg("rowgroup_prefetch_min_decode_batches") =
	            galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches,
	        py::arg("layout")                    = "compact",
	        py::arg("grid_transform")            = py::none(),
	        py::arg("plan_cache_capacity")       = galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity,
	        py::arg("enable_planless_execution") = true,
	        py::arg("scheduling_policy")          = "fully-overlapped",
	        py::arg("transform_blocks_per_launch") = 0,
	        py::arg("transform_ctas_per_launch")  = 0,
	        py::arg("use_low_priority_streams")   = false,
		        py::arg("transforms")                 = py::none(),
	        py::arg("crop_execution_mode")        = "auto",
	        py::arg("decode_workset_capacity_mib") = kDefaultDirectDctDecodeWorksetCapacityMiB,
	        py::arg("block_major_double_buffer")   = "auto",
	        py::arg("bounded_read_amplification_cap") = 1.0,
	        py::arg("bounded_read_local_amplification_cap") = 0.0,
	        py::arg("bounded_read_max_run_bytes") = 0U)
	    .def(
	        "prefetch_batch",
	        [](TorchDirectDctReader& reader,
	           std::vector<uint32_t> image_ids,
	           const py::object&     crop,
	           const std::string&    dct_coeffs,
	           const size_t          cache_capacity_mib,
	           const size_t          decode_batch_rowgroups,
	           const bool            enable_rowgroup_prefetch,
	           const size_t          rowgroup_prefetch_depth,
	           const size_t          rowgroup_prefetch_workers,
	           const size_t          rowgroup_prefetch_min_decode_batches,
	           const std::string&    layout,
	           const py::object&     grid_transform,
	           const size_t          plan_cache_capacity,
	           const bool            enable_planless_execution,
	           const std::string&    scheduling_policy,
	           const size_t          transform_blocks_per_launch,
	           const size_t          transform_ctas_per_launch,
	           const bool            use_low_priority_streams,
	           const py::object&     transforms,
	           const std::string&    crop_execution_mode,
		           const size_t          decode_workset_capacity_mib,
		           const std::string&    block_major_double_buffer,
		           const bool            async_planless_completion,
		           const double          bounded_read_amplification_cap,
		           const double          bounded_read_local_amplification_cap,
		           const size_t          bounded_read_max_run_bytes) {
		        const auto crop_box = parse_crop(crop);
		        auto       requests = parse_transform_requests(image_ids, transforms, crop_box);
		        const auto options  = make_batch_options(dct_coeffs,
                                                        cache_capacity_mib,
                                                        decode_batch_rowgroups,
                                                        enable_rowgroup_prefetch,
                                                        rowgroup_prefetch_depth,
                                                        rowgroup_prefetch_workers,
                                                        rowgroup_prefetch_min_decode_batches,
                                                        layout,
                                                        grid_transform,
		                                                    plan_cache_capacity,
		                                                    enable_planless_execution,
		                                                    decode_workset_capacity_mib,
		                                                    crop_execution_mode,
	                                                        scheduling_policy,
                                                        transform_blocks_per_launch,
	                                                        transform_ctas_per_launch,
	                                                        use_low_priority_streams,
	                                                        block_major_double_buffer,
	                                                        async_planless_completion,
	                                                        bounded_read_amplification_cap,
	                                                        bounded_read_local_amplification_cap,
	                                                        bounded_read_max_run_bytes);
		        return reader.prefetch_batch(std::move(requests), options);
	        },
	        py::arg("image_ids"),
	        py::arg("crop")                      = py::none(),
	        py::arg("dct_coeffs")                = "all",
	        py::arg("cache_capacity_mib")        = kDefaultDirectDctCacheCapacityMiB,
	        py::arg("decode_batch_rowgroups")    = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups,
	        py::arg("enable_rowgroup_prefetch")  = true,
	        py::arg("rowgroup_prefetch_depth")   = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth,
	        py::arg("rowgroup_prefetch_workers") = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers,
	        py::arg("rowgroup_prefetch_min_decode_batches") =
	            galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches,
	        py::arg("layout")                    = "compact",
	        py::arg("grid_transform")            = py::none(),
	        py::arg("plan_cache_capacity")       = galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity,
	        py::arg("enable_planless_execution") = true,
	        py::arg("scheduling_policy")          = "fully-overlapped",
	        py::arg("transform_blocks_per_launch") = 0,
	        py::arg("transform_ctas_per_launch")  = 0,
	        py::arg("use_low_priority_streams")   = false,
		        py::arg("transforms")                 = py::none(),
	        py::arg("crop_execution_mode")        = "auto",
	        py::arg("decode_workset_capacity_mib") = kDefaultDirectDctDecodeWorksetCapacityMiB,
	        py::arg("block_major_double_buffer")   = "auto",
	        py::arg("async_planless_completion")   = false,
	        py::arg("bounded_read_amplification_cap") = 1.0,
	        py::arg("bounded_read_local_amplification_cap") = 0.0,
	        py::arg("bounded_read_max_run_bytes") = 0U)
	    .def(
	        "read_batch_async",
	        [](TorchDirectDctReader& reader,
	           std::vector<uint32_t> image_ids,
	           const py::object&     crop,
	           const std::string&    dct_coeffs,
	           const size_t          cache_capacity_mib,
	           const size_t          decode_batch_rowgroups,
	           const bool            enable_rowgroup_prefetch,
	           const size_t          rowgroup_prefetch_depth,
	           const size_t          rowgroup_prefetch_workers,
	           const size_t          rowgroup_prefetch_min_decode_batches,
	           const std::string&    layout,
	           const py::object&     grid_transform,
	           const size_t          plan_cache_capacity,
		           const bool            enable_planless_execution,
		           const py::object&     transforms,
		           const std::string&    crop_execution_mode,
		           const size_t          decode_workset_capacity_mib,
		           const double          bounded_read_amplification_cap,
		           const double          bounded_read_local_amplification_cap,
		           const size_t          bounded_read_max_run_bytes) {
		        const auto crop_box = parse_crop(crop);
		        auto       requests = parse_transform_requests(image_ids, transforms, crop_box);
		        const auto options  = make_batch_options(dct_coeffs,
                                                        cache_capacity_mib,
                                                        decode_batch_rowgroups,
                                                        enable_rowgroup_prefetch,
                                                        rowgroup_prefetch_depth,
                                                        rowgroup_prefetch_workers,
                                                        rowgroup_prefetch_min_decode_batches,
                                                        layout,
                                                        grid_transform,
		                                                    plan_cache_capacity,
		                                                    enable_planless_execution,
		                                                    decode_workset_capacity_mib,
		                                                    crop_execution_mode,
		                                                    "fully-overlapped",
		                                                    0U,
		                                                    0U,
		                                                    false,
		                                                    "auto",
		                                                    false,
		                                                    bounded_read_amplification_cap,
		                                                    bounded_read_local_amplification_cap,
		                                                    bounded_read_max_run_bytes);
		        return reader.prefetch_batch(std::move(requests), options);
	        },
	        py::arg("image_ids"),
	        py::arg("crop")                      = py::none(),
	        py::arg("dct_coeffs")                = "all",
	        py::arg("cache_capacity_mib")        = kDefaultDirectDctCacheCapacityMiB,
	        py::arg("decode_batch_rowgroups")    = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups,
	        py::arg("enable_rowgroup_prefetch")  = true,
	        py::arg("rowgroup_prefetch_depth")   = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth,
	        py::arg("rowgroup_prefetch_workers") = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers,
	        py::arg("rowgroup_prefetch_min_decode_batches") =
	            galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches,
	        py::arg("layout")                    = "compact",
	        py::arg("grid_transform")            = py::none(),
	        py::arg("plan_cache_capacity")       = galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity,
	        py::arg("enable_planless_execution") = true,
		        py::arg("transforms")                = py::none(),
		        py::arg("crop_execution_mode")        = "auto",
		        py::arg("decode_workset_capacity_mib") = kDefaultDirectDctDecodeWorksetCapacityMiB,
		        py::arg("bounded_read_amplification_cap") = 1.0,
		        py::arg("bounded_read_local_amplification_cap") = 0.0,
		        py::arg("bounded_read_max_run_bytes") = 0U)
	    .def("read_prefetched",
	         &TorchDirectDctReader::read_prefetched,
	         py::arg("prefetch"),
	         py::call_guard<py::gil_scoped_release>())
	    .def("manual_reclaim", &TorchDirectDctReader::manual_reclaim);

	m.attr("DEFAULT_CACHE_CAPACITY_MIB")  = kDefaultDirectDctCacheCapacityMiB;
	m.attr("DEFAULT_PLAN_CACHE_CAPACITY") = galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity;
	m.attr("DIRECT_DCT_BINDING_SCHEMA") = "galp-direct-dct-binding-v2";
	m.attr("DIRECT_DCT_PROFILE_SCHEMA") = "galp-direct-dct-profile-v1";
	m.attr("DIRECT_DCT_METRICS_SCHEMA") = "galp-direct-dct-metrics-v2";
	m.def("available_direct_dct_profiles", &galp::profiles::available_direct_dct_profile_ids);
	m.def("direct_dct_profile_info", [](const std::string& profile_id) {
		return direct_dct_profile_info(galp::profiles::resolve_direct_dct_profile(profile_id));
	}, py::arg("profile_id"));
	m.def("manual_reclaim", []() { return DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished(); });
}
