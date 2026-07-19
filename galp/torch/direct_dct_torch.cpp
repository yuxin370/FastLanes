#include "galp/direct_dct.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
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
constexpr auto   kImmediateFuturePollDuration         = std::chrono::seconds(0);

galp::jpeg::JpegDctCropBox parse_crop(const py::object& crop) {
	galp::jpeg::JpegDctCropBox parsed {};
	if (crop.is_none()) {
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

galp::jpeg::JpegDctCoefficientSelection parse_coefficients(const std::string& spec);
galp::jpeg::JpegDctDeviceLayout parse_layout(const std::string& layout);
galp::jpeg::JpegDctSchedulingPolicy parse_scheduling_policy(const std::string& policy);

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
                                                         const std::string& scheduling_policy = "fully-overlapped",
                                                         const size_t       transform_blocks_per_launch = 0,
                                                         const bool         use_low_priority_streams = false) {
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.coefficient_selection                = parse_coefficients(dct_coeffs);
	options.layout                               = parse_layout(layout);
	options.grid_transform                       = parse_grid_transform(grid_transform);
	options.cache_capacity_bytes                 = cache_capacity_bytes_from_mib(cache_capacity_mib);
	options.decode_batch_rowgroups               = decode_batch_rowgroups;
	options.plan_cache_capacity                  = plan_cache_capacity;
	options.enable_rowgroup_prefetch             = enable_rowgroup_prefetch;
	options.rowgroup_prefetch_depth              = rowgroup_prefetch_depth;
	options.rowgroup_prefetch_workers            = rowgroup_prefetch_workers;
	options.rowgroup_prefetch_min_decode_batches = rowgroup_prefetch_min_decode_batches;
	options.enable_planless_execution            = enable_planless_execution;
	options.scheduling_policy                    = parse_scheduling_policy(scheduling_policy);
	options.transform_blocks_per_launch          = transform_blocks_per_launch;
	options.use_low_priority_streams              = use_low_priority_streams;
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
	    out["planless_axis_program_bytes"]                 = preview.planless_axis_program_bytes;
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
	out["planless_image_descriptor_count"]               = stats.planless_image_descriptor_count;
	out["planless_transform_output_block_count"]         = stats.planless_transform_output_block_count;
	out["rowgroup_storage_bytes_read"]                   = stats.rowgroup_storage_bytes_read;
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
	out["project_decoded_ycbcr_grid_launch_count"]       = stats.project_decoded_ycbcr_grid_launch_count;
	out["jpeg_dct_projection_items_materialized"]        = stats.jpeg_dct_projection_items_materialized;
	out["planless_transform_kernel_launch_count"]        = stats.planless_transform_kernel_launch_count;
	out["planless_transform_max_blocks_per_launch"]      = stats.planless_transform_max_blocks_per_launch;
	out["planless_transform_max_output_blocks_per_launch"] =
	    stats.planless_transform_max_output_blocks_per_launch;
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
			const auto          stream = c10::cuda::getCurrentCUDAStream(device_index);
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

	std::mutex                 mutex_;
	std::deque<PendingRelease> pending_;
};

struct TorchDirectDctBatch {
	explicit TorchDirectDctBatch(galp::jpeg::DirectDctBatch batch_in)
	    : batch(std::make_shared<galp::jpeg::DirectDctBatch>(std::move(batch_in))) {
	}

	torch::Tensor coefficients() {
		DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
		if (tensor.defined()) {
			return tensor;
		}
		wait_for_batch_completion();
		const auto desc         = batch->tensor_async();
		const auto device_index = static_cast<c10::DeviceIndex>(desc.cuda_device < 0 ? 0 : desc.cuda_device);
		auto options = torch::TensorOptions().dtype(torch::kInt16).device(torch::Device(torch::kCUDA, device_index));
		if (desc.data == nullptr || desc.empty()) {
			tensor = torch::empty({static_cast<int64_t>(desc.rows()), static_cast<int64_t>(desc.columns())}, options);
			return tensor;
		}
		auto owner = batch;
		tensor     = torch::from_blob(
            const_cast<int16_t*>(desc.data),
            {static_cast<int64_t>(desc.rows()), static_cast<int64_t>(desc.columns())},
            {static_cast<int64_t>(desc.strides[0]), static_cast<int64_t>(desc.strides[1])},
            [owner = std::move(owner), device_index](void*) mutable {
                DeferredDirectDctBatchReleaseQueue::instance().defer(std::move(owner), device_index);
            },
            options);
		return tensor;
	}

	torch::Tensor y() {
		wait_for_batch_completion();
		return grid_tensor(batch->y_tensor_async(), y_tensor);
	}

	torch::Tensor cbcr() {
		wait_for_batch_completion();
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

	void record_stream() {
		DeferredDirectDctBatchReleaseQueue::instance().defer(batch, tensor_device_index());
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

	[[nodiscard]] uintptr_t device_data_ptr() const noexcept {
		return reinterpret_cast<uintptr_t>(batch->device_data());
	}

	[[nodiscard]] uintptr_t y_device_data_ptr() const noexcept {
		return reinterpret_cast<uintptr_t>(batch->y_device_data());
	}

	[[nodiscard]] uintptr_t cbcr_device_data_ptr() const noexcept {
		return reinterpret_cast<uintptr_t>(batch->cbcr_device_data());
	}

	[[nodiscard]] std::string layout() const {
		return layout_to_string(batch->device_batch().layout());
	}

	std::shared_ptr<galp::jpeg::DirectDctBatch> batch;
	torch::Tensor                               tensor;
	torch::Tensor                               y_tensor;
	torch::Tensor                               cbcr_tensor;
	torch::Tensor                               image_offsets_tensor_cache;
	torch::Tensor                               image_counts_tensor_cache;
	torch::Tensor                               block_to_image_tensor_cache;

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
		auto options = torch::TensorOptions().dtype(torch::kInt16).device(torch::Device(torch::kCUDA, device_index));
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
		if (desc.data == nullptr || desc.empty()) {
			cached = torch::empty(shape, options);
			return cached;
		}
		auto owner = batch;
		cached     = torch::from_blob(const_cast<int16_t*>(desc.data),
                                  shape,
                                  strides,
                                  [owner = std::move(owner), device_index](void*) mutable {
                                      DeferredDirectDctBatchReleaseQueue::instance().defer(std::move(owner), device_index);
                                  },
                                  options);
		return cached;
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
                                          const std::vector<uint32_t>&                      image_ids,
                                          const galp::jpeg::JpegDctCropBox&                 crop,
                                          const galp::jpeg::JpegDctDeviceBatchOptions&      options) {
	auto& release_queue = DeferredDirectDctBatchReleaseQueue::instance();
	release_queue.reclaim_finished();
	galp::jpeg::DirectDctBatch raw_batch;
	{
		std::lock_guard<std::mutex> lock(state->runtime_mutex);
		raw_batch = state->runtime.ReadBatch(image_ids, crop, options);
	}
	release_queue.reclaim_finished();
	return TorchDirectDctBatch(std::move(raw_batch));
}

class TorchDirectDctPrefetch {
public:
	explicit TorchDirectDctPrefetch(std::future<TorchDirectDctBatch> future)
	    : future_(std::move(future)) {
	}

	[[nodiscard]] bool ready() const {
		return future_.valid() && future_.wait_for(kImmediateFuturePollDuration) == std::future_status::ready;
	}

	TorchDirectDctBatch read() {
		if (!future_.valid()) {
			throw std::runtime_error("DirectDctPrefetch has already been consumed");
		}
		return future_.get();
	}

private:
	std::future<TorchDirectDctBatch> future_;
};

class TorchDirectDctReader {
public:
	explicit TorchDirectDctReader(const std::string& manifest_path)
	    : state(std::make_shared<TorchDirectDctReaderState>(manifest_path)) {
	}

	TorchDirectDctBatch read_batch(const std::vector<uint32_t>&                 image_ids,
	                               const galp::jpeg::JpegDctCropBox&            crop,
	                               const galp::jpeg::JpegDctDeviceBatchOptions& options) {
		return read_batch_from_state(state, image_ids, crop, options);
	}

	py::dict plan_batch(const std::vector<uint32_t>&                 image_ids,
	                    const galp::jpeg::JpegDctCropBox&            crop,
	                    const galp::jpeg::JpegDctDeviceBatchOptions& options) {
		std::lock_guard<std::mutex> lock(state->runtime_mutex);
		return plan_preview_to_dict(state->runtime.PlanBatch(make_crop_requests(image_ids, crop), options));
	}

	std::shared_ptr<TorchDirectDctPrefetch> prefetch_batch(std::vector<uint32_t>                       image_ids,
	                                                       const galp::jpeg::JpegDctCropBox            crop,
	                                                       const galp::jpeg::JpegDctDeviceBatchOptions options) {
		const auto               device_index = c10::cuda::current_device();
		auto                     state_copy   = state;
		auto                     completion   = std::make_shared<std::promise<void>>();
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
			                     image_ids = std::move(image_ids),
			                     crop,
			                     options,
			                     device_index,
			                     predecessor = std::move(predecessor),
			                     completion]() mutable {
				                    if (predecessor.valid()) {
					                    predecessor.wait();
				                    }
				                    try {
					                    c10::cuda::CUDAGuard guard(device_index);
					                    auto batch = read_batch_from_state(state, image_ids, crop, options);
					                    completion->set_value();
					                    return batch;
				                    } catch (...) {
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
		return std::make_shared<TorchDirectDctPrefetch>(std::move(future));
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

private:
	std::shared_ptr<TorchDirectDctReaderState> state;
};

} // namespace

PYBIND11_MODULE(_galp_direct_dct, m) {
	m.doc() = "Stay-on-GPU GALP JPEG DCT runtime for PyTorch direct-DCT workloads";

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
	    .def("record_stream",
	         &TorchDirectDctBatch::record_stream,
	         "Keep the underlying DirectDctBatch alive until work on the current CUDA stream reaches this point.")
	    .def_property_readonly("layout", &TorchDirectDctBatch::layout)
	    .def_property_readonly("device_data_ptr", &TorchDirectDctBatch::device_data_ptr)
	    .def_property_readonly("y_device_data_ptr", &TorchDirectDctBatch::y_device_data_ptr)
	    .def_property_readonly("cbcr_device_data_ptr", &TorchDirectDctBatch::cbcr_device_data_ptr)
	    .def_property_readonly("global_image_ids",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->global_image_ids(); })
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
	    .def("read",
	         &TorchDirectDctPrefetch::read,
	         "Consume the prefetch handle and return the batch; dropping an unread handle waits for the read to finish.",
	         py::call_guard<py::gil_scoped_release>());

	py::class_<TorchDirectDctReader>(m, "DirectDctReader")
	    .def(py::init<const std::string&>(), py::arg("manifest_path"))
	    .def_property_readonly("image_count", &TorchDirectDctReader::image_count)
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
	           const bool                   enable_planless_execution) {
		        const auto crop_box = parse_crop(crop);
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
                                                        enable_planless_execution);
		        return reader.plan_batch(image_ids, crop_box, options);
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
	        py::arg("enable_planless_execution") = true)
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
	           const bool                   use_low_priority_streams) {
		        const auto             crop_box = parse_crop(crop);
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
                                                        scheduling_policy,
                                                        transform_blocks_per_launch,
                                                        use_low_priority_streams);
		        py::gil_scoped_release release;
		        return reader.read_batch(image_ids, crop_box, options);
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
	        py::arg("use_low_priority_streams")   = false)
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
	           const bool            use_low_priority_streams) {
		        const auto crop_box = parse_crop(crop);
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
                                                        scheduling_policy,
                                                        transform_blocks_per_launch,
                                                        use_low_priority_streams);
		        return reader.prefetch_batch(std::move(image_ids), crop_box, options);
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
	        py::arg("use_low_priority_streams")   = false)
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
	           const bool            enable_planless_execution) {
		        const auto crop_box = parse_crop(crop);
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
                                                        enable_planless_execution);
		        return reader.prefetch_batch(std::move(image_ids), crop_box, options);
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
	        py::arg("enable_planless_execution") = true)
	    .def("read_prefetched",
	         &TorchDirectDctReader::read_prefetched,
	         py::arg("prefetch"),
	         py::call_guard<py::gil_scoped_release>())
	    .def("manual_reclaim", &TorchDirectDctReader::manual_reclaim);

	m.attr("DEFAULT_CACHE_CAPACITY_MIB")  = kDefaultDirectDctCacheCapacityMiB;
	m.attr("DEFAULT_PLAN_CACHE_CAPACITY") = galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity;
	m.def("manual_reclaim", []() { return DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished(); });
}
