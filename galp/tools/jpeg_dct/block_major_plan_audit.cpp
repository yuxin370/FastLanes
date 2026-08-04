#include "galp/jpeg_dct.hpp"
#include "galp/jpeg_dct_diagnostics.hpp"
#include "galp/profiles/rgbnomore.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

struct Options {
	std::filesystem::path manifest;
	std::filesystem::path descriptor_directory;
	std::filesystem::path output_json;
	uint32_t start = 0U;
	uint32_t count = 128U;
	uint32_t seed  = 20260731U;
	uint32_t decode_batch_rowgroups = 64U;
	uint32_t prefetch_workers       = 2U;
	uint32_t workset_capacity_mib   = 512U;
	std::string pattern = "sequential";
	std::vector<uint32_t> image_ids;
	bool compare_legacy = false;
	bool execute = false;
	bool explicit_crops = false;
	bool require_grayscale = false;
	bool require_cross_shard = false;
};

void usage(const char* program) {
	std::cerr << "Usage: " << program
	          << " MANIFEST --descriptor-dir DIR [--start N] [--count N]"
	             " [--pattern sequential|reverse|random|duplicates] [--seed N]"
	             " [--image-ids N,N,...] [--explicit-crops] [--require-grayscale] [--require-cross-shard]"
	             " [--compare-legacy] [--execute] [--decode-batch-rowgroups N]"
	             " [--prefetch-workers N] [--workset-capacity-mib N] [--output-json PATH]\n";
}

uint32_t parse_u32(const std::string_view name, const char* value) {
	const auto parsed = std::stoull(value);
	if (parsed > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error(std::string(name) + " is outside uint32_t range");
	}
	return static_cast<uint32_t>(parsed);
}

std::vector<uint32_t> parse_image_ids(const char* value) {
	std::vector<uint32_t> result;
	std::string           input(value);
	size_t                begin = 0U;
	while (begin <= input.size()) {
		const auto end = input.find(',', begin);
		const auto token = input.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
		if (token.empty()) {
			throw std::runtime_error("--image-ids contains an empty item");
		}
		result.push_back(parse_u32("--image-ids", token.c_str()));
		if (end == std::string::npos) {
			break;
		}
		begin = end + 1U;
	}
	if (result.empty()) {
		throw std::runtime_error("--image-ids requires at least one image");
	}
	return result;
}

Options parse_options(const int argc, char** argv) {
	Options options;
	for (int index = 1; index < argc; ++index) {
		const std::string_view argument(argv[index]);
		auto require_value = [&](const std::string_view name) -> const char* {
			if (++index >= argc) {
				throw std::runtime_error(std::string(name) + " requires a value");
			}
			return argv[index];
		};
		if (argument == "--help" || argument == "-h") {
			usage(argv[0]);
			std::exit(0);
		}
		if (argument == "--descriptor-dir") {
			options.descriptor_directory = require_value(argument);
		} else if (argument == "--output-json") {
			options.output_json = require_value(argument);
		} else if (argument == "--start") {
			options.start = parse_u32(argument, require_value(argument));
		} else if (argument == "--count") {
			options.count = parse_u32(argument, require_value(argument));
		} else if (argument == "--seed") {
			options.seed = parse_u32(argument, require_value(argument));
		} else if (argument == "--decode-batch-rowgroups") {
			options.decode_batch_rowgroups = parse_u32(argument, require_value(argument));
		} else if (argument == "--prefetch-workers") {
			options.prefetch_workers = parse_u32(argument, require_value(argument));
		} else if (argument == "--workset-capacity-mib") {
			options.workset_capacity_mib = parse_u32(argument, require_value(argument));
		} else if (argument == "--pattern") {
			options.pattern = require_value(argument);
		} else if (argument == "--image-ids") {
			options.image_ids = parse_image_ids(require_value(argument));
		} else if (argument == "--explicit-crops") {
			options.explicit_crops = true;
		} else if (argument == "--require-grayscale") {
			options.require_grayscale = true;
		} else if (argument == "--require-cross-shard") {
			options.require_cross_shard = true;
		} else if (argument == "--compare-legacy") {
			options.compare_legacy = true;
		} else if (argument == "--execute") {
			options.execute        = true;
			options.compare_legacy = true;
		} else if (!argument.empty() && argument.front() == '-') {
			throw std::runtime_error("unknown option: " + std::string(argument));
		} else if (options.manifest.empty()) {
			options.manifest = argv[index];
		} else {
			throw std::runtime_error("multiple manifest paths were supplied");
		}
	}
	if (options.manifest.empty() || options.descriptor_directory.empty() || options.count == 0U ||
	    options.decode_batch_rowgroups == 0U || options.workset_capacity_mib == 0U) {
		usage(argv[0]);
		throw std::runtime_error("manifest, descriptor directory, and positive count are required");
	}
	if (options.pattern != "sequential" && options.pattern != "reverse" && options.pattern != "random" &&
	    options.pattern != "duplicates") {
		throw std::runtime_error("unsupported --pattern");
	}
	return options;
}

using SelectedVector = std::tuple<uint32_t, uint32_t, uint32_t>;

std::set<SelectedVector> compact_vectors(const galp::jpeg::JpegDctBlockMajorCompactPlan& plan) {
	std::set<SelectedVector> result;
	for (const auto& run : plan.vector_runs) {
		for (uint32_t vector = run.first_vector; vector < run.first_vector + run.vector_count; ++vector) {
			result.emplace(run.shard_id, run.rowgroup_index, vector);
		}
	}
	return result;
}

std::set<SelectedVector> legacy_vectors(const galp::jpeg::JpegDctDeviceBatchPlanPreview& preview) {
	std::set<SelectedVector> result;
	for (const auto& rowgroup : preview.rowgroup_vector_plans) {
		for (const auto vector : rowgroup.selected_vectors) {
			result.emplace(rowgroup.rowgroup.shard_id, rowgroup.rowgroup.rowgroup_index, vector);
		}
	}
	return result;
}

struct HostGrid {
	std::vector<int16_t> y;
	std::vector<int16_t> cbcr;
};

struct DeviceRun {
	HostGrid                                 grid;
	galp::jpeg::JpegDctDeviceExecutionStats stats;
	galp::jpeg::JpegDctDeviceCacheStats     cache_stats;
	double                                   wall_ms = 0.0;
};

DeviceRun execute_device_batch(galp::jpeg::JpegDctShardDatasetReader&                    reader,
                               const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests,
                               const galp::jpeg::JpegDctDeviceBatchOptions&            options) {
	const auto begin = std::chrono::steady_clock::now();
	auto       batch = reader.ReadDeviceDctBatch(requests, options);
	batch.synchronize();
	if (batch.grid_output_data_type() != galp::jpeg::JpegDctGridOutputDataType::kInt16) {
		throw std::runtime_error("block-major execution audit requires int16 transformed-grid output");
	}
	DeviceRun result;
	result.grid.y.resize(batch.y_coefficient_count());
	result.grid.cbcr.resize(batch.cbcr_coefficient_count());
	if (!result.grid.y.empty() &&
	    cudaMemcpy(result.grid.y.data(),
	               batch.y_coefficients(),
	               result.grid.y.size() * sizeof(int16_t),
	               cudaMemcpyDeviceToHost) != cudaSuccess) {
		throw std::runtime_error("failed to copy audited Y coefficients from CUDA");
	}
	if (!result.grid.cbcr.empty() &&
	    cudaMemcpy(result.grid.cbcr.data(),
	               batch.cbcr_coefficients(),
	               result.grid.cbcr.size() * sizeof(int16_t),
	               cudaMemcpyDeviceToHost) != cudaSuccess) {
		throw std::runtime_error("failed to copy audited CbCr coefficients from CUDA");
	}
	result.stats       = batch.execution_stats();
	result.cache_stats = batch.cache_stats();
	result.wall_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	return result;
}

uint64_t fnv1a_append(uint64_t hash, const void* data, const size_t bytes) {
	const auto* input = static_cast<const uint8_t*>(data);
	for (size_t index = 0U; index < bytes; ++index) {
		hash ^= input[index];
		hash *= UINT64_C(1099511628211);
	}
	return hash;
}

std::string grid_hash(const HostGrid& grid) {
	uint64_t hash = UINT64_C(14695981039346656037);
	hash = fnv1a_append(hash, grid.y.data(), grid.y.size() * sizeof(int16_t));
	const uint8_t separator = 0xA5U;
	hash = fnv1a_append(hash, &separator, sizeof(separator));
	hash = fnv1a_append(hash, grid.cbcr.data(), grid.cbcr.size() * sizeof(int16_t));
	std::ostringstream output;
	output << std::hex << std::setw(16) << std::setfill('0') << hash;
	return output.str();
}

struct Difference {
	size_t  mismatches = 0U;
	int32_t max_abs    = 0;
};

Difference difference(const HostGrid& lhs, const HostGrid& rhs) {
	if (lhs.y.size() != rhs.y.size() || lhs.cbcr.size() != rhs.cbcr.size()) {
		return {std::numeric_limits<size_t>::max(), std::numeric_limits<int32_t>::max()};
	}
	Difference result;
	const auto compare = [&](const std::vector<int16_t>& left, const std::vector<int16_t>& right) {
		for (size_t index = 0U; index < left.size(); ++index) {
			const int32_t delta = static_cast<int32_t>(left[index]) - static_cast<int32_t>(right[index]);
			const int32_t absolute = delta < 0 ? -delta : delta;
			result.mismatches += absolute != 0;
			result.max_abs = std::max(result.max_abs, absolute);
		}
	};
	compare(lhs.y, rhs.y);
	compare(lhs.cbcr, rhs.cbcr);
	return result;
}

} // namespace

int main(const int argc, char** argv) try {
	const auto options = parse_options(argc, argv);
	const auto load_begin = std::chrono::steady_clock::now();
	galp::jpeg::JpegDctBlockMajorCompactPlanner planner(options.manifest, options.descriptor_directory);
	const auto load_end = std::chrono::steady_clock::now();
	if (options.image_ids.empty() &&
	    (options.start >= planner.image_count() || options.count > planner.image_count())) {
		throw std::runtime_error("request range exceeds dataset image count");
	}
	std::vector<uint32_t> image_ids = options.image_ids;
	image_ids.reserve(options.image_ids.empty() ? options.count : options.image_ids.size());
	if (!options.image_ids.empty()) {
		for (const auto image_id : image_ids) {
			if (image_id >= planner.image_count()) {
				throw std::runtime_error("--image-ids contains an image outside the dataset");
			}
		}
	} else if (options.pattern == "random") {
		std::mt19937 generator(options.seed);
		std::uniform_int_distribution<uint32_t> distribution(0U, static_cast<uint32_t>(planner.image_count() - 1U));
		for (uint32_t index = 0U; index < options.count; ++index) {
			image_ids.push_back(distribution(generator));
		}
	} else if (options.pattern == "duplicates") {
		for (uint32_t index = 0U; index < options.count; ++index) {
			image_ids.push_back(options.start + (index / 3U) % std::max<uint32_t>(1U, options.count / 4U));
		}
	} else {
		if (options.start > planner.image_count() - options.count) {
			throw std::runtime_error("sequential request range exceeds dataset image count");
		}
		for (uint32_t index = 0U; index < options.count; ++index) {
			image_ids.push_back(options.start + index);
		}
		if (options.pattern == "reverse") {
			std::reverse(image_ids.begin(), image_ids.end());
		}
	}
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	requests.reserve(image_ids.size());
	std::unique_ptr<galp::jpeg::JpegDctShardDatasetReader> metadata_reader;
	if (options.explicit_crops) {
		metadata_reader = std::make_unique<galp::jpeg::JpegDctShardDatasetReader>(options.manifest);
	}
	for (size_t request_index = 0U; request_index < image_ids.size(); ++request_index) {
		const auto image = image_ids[request_index];
		galp::jpeg::JpegDctCropBox crop;
		if (metadata_reader) {
			const auto metadata = metadata_reader->ImageMetadata(image);
			crop.width  = std::min<uint32_t>(224U, metadata.image_width);
			crop.height = std::min<uint32_t>(224U, metadata.image_height);
			if (crop.width == 0U || crop.height == 0U) {
				throw std::runtime_error("explicit-crop audit found an image with zero dimensions");
			}
			const auto x_slack = metadata.image_width - crop.width;
			const auto y_slack = metadata.image_height - crop.height;
			crop.x = x_slack == 0U
			             ? 0U
			             : static_cast<uint32_t>((static_cast<uint64_t>(image) * 37U + request_index * 17U) %
			                                     (static_cast<uint64_t>(x_slack) + 1U));
			crop.y = y_slack == 0U
			             ? 0U
			             : static_cast<uint32_t>((static_cast<uint64_t>(image) * 29U + request_index * 23U) %
			                                     (static_cast<uint64_t>(y_slack) + 1U));
		}
		requests.push_back({image, crop, options.explicit_crops && request_index % 2U != 0U, {}, {}});
	}
	const auto transform = galp::profiles::rgbnomore_val_dct_grid_transform();
	const auto plan_begin = std::chrono::steady_clock::now();
	const auto plan = planner.Plan(requests, transform);
	const auto plan_end = std::chrono::steady_clock::now();
	const auto compact_selected = compact_vectors(plan);
	const bool grayscale_requirement_met =
	    !options.require_grayscale ||
	    std::any_of(plan.requests.begin(), plan.requests.end(), [](const auto& request) {
		    return request.components[0].present && !request.components[1].present && !request.components[2].present;
	    });
	std::set<uint32_t> requested_shards;
	for (const auto& request : plan.requests) {
		requested_shards.insert(request.shard_id);
	}
	const bool cross_shard_requirement_met = !options.require_cross_shard || requested_shards.size() > 1U;
	if (!grayscale_requirement_met) {
		throw std::runtime_error("--require-grayscale request set contains no Y-only image");
	}
	if (!cross_shard_requirement_met) {
		throw std::runtime_error("--require-cross-shard request set touches fewer than two shards");
	}

	double legacy_ms = 0.0;
	bool vectors_equal = true;
	size_t legacy_selected_vectors = 0U;
	size_t legacy_expanded_items = 0U;
	size_t legacy_sort_items = 0U;
	if (options.compare_legacy) {
		galp::jpeg::JpegDctShardDatasetReader legacy(options.manifest);
		galp::jpeg::JpegDctDeviceBatchOptions legacy_options;
		legacy_options.layout = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
		legacy_options.grid_transform = transform;
		legacy_options.enable_planless_execution = false;
		legacy_options.crop_execution_mode = galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
		const auto legacy_begin = std::chrono::steady_clock::now();
		const auto preview = legacy.PlanDeviceDctBatch(requests, legacy_options);
		const auto legacy_end = std::chrono::steady_clock::now();
		legacy_ms = std::chrono::duration<double, std::milli>(legacy_end - legacy_begin).count();
		const auto legacy_selected = legacy_vectors(preview);
		vectors_equal = legacy_selected == compact_selected;
		legacy_selected_vectors = legacy_selected.size();
		legacy_expanded_items = preview.host_expanded_transform_items_created;
		legacy_sort_items = preview.host_global_transform_sort_items;
	}

	bool execution_selected = false;
	bool execution_deterministic = false;
	bool execution_within_tolerance = false;
	bool execution_zero_expansion = false;
	bool execution_kernel_launched = false;
	bool execution_strategy_accounting_valid = false;
	bool execution_descriptor_bounds = false;
	bool execution_workset_bounds = false;
	bool execution_io_bounds = false;
	bool execution_resource_bounds = false;
	bool execution_active_output_schedule_valid = false;
	bool execution_cache_contract_valid = false;
	double planless_wall_ms = 0.0;
	double repeat_wall_ms = 0.0;
	double legacy_wall_ms = 0.0;
	std::string planless_hash;
	std::string repeat_hash;
	std::string legacy_hash;
	Difference planless_legacy_difference;
	galp::jpeg::JpegDctDeviceExecutionStats planless_execution_stats;
	galp::jpeg::JpegDctDeviceExecutionStats legacy_execution_stats;
	galp::jpeg::JpegDctDeviceCacheStats     planless_cache_stats;
	if (options.execute) {
		int device_count = 0;
		if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
			throw std::runtime_error("--execute requested but CUDA device is unavailable");
		}
		const auto descriptor_path = std::filesystem::absolute(options.descriptor_directory).string();
		if (setenv("GALP_BLOCK_MAJOR_ACCESS_DIR", descriptor_path.c_str(), 1) != 0) {
			throw std::runtime_error("failed to set GALP_BLOCK_MAJOR_ACCESS_DIR");
		}
		galp::jpeg::JpegDctDeviceBatchOptions device_options;
		device_options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
		device_options.grid_transform            = transform;
		device_options.cache_capacity_bytes      = 0U;
		device_options.plan_cache_capacity       = 0U;
		device_options.decode_batch_rowgroups    = options.decode_batch_rowgroups;
		device_options.enable_rowgroup_prefetch  = options.prefetch_workers > 1U;
		device_options.rowgroup_prefetch_workers = std::max<uint32_t>(1U, options.prefetch_workers);
		device_options.rowgroup_prefetch_depth   = 1U;
		device_options.decode_workset_capacity_bytes =
		    static_cast<size_t>(options.workset_capacity_mib) * 1024U * 1024U;
		device_options.enable_planless_execution = true;
		device_options.crop_execution_mode =
		    galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
		DeviceRun planless;
		DeviceRun repeat;
		{
			galp::jpeg::JpegDctShardDatasetReader planless_reader(options.manifest);
			const auto preview = planless_reader.PlanDeviceDctBatch(requests, device_options);
			execution_selected = preview.uses_planless_fixed_transform;
			execution_zero_expansion = preview.host_expanded_transform_items_created == 0U &&
			                           preview.host_output_block_source_lists_created == 0U &&
			                           preview.host_global_transform_sort_items == 0U;
			planless = execute_device_batch(planless_reader, requests, device_options);
			repeat   = execute_device_batch(planless_reader, requests, device_options);
		}
		if (unsetenv("GALP_BLOCK_MAJOR_ACCESS_DIR") != 0) {
			throw std::runtime_error("failed to clear GALP_BLOCK_MAJOR_ACCESS_DIR before legacy control");
		}
		auto legacy_options                      = device_options;
		legacy_options.enable_planless_execution = false;
		galp::jpeg::JpegDctShardDatasetReader legacy_reader(options.manifest);
		auto legacy = execute_device_batch(legacy_reader, requests, legacy_options);

		planless_wall_ms = planless.wall_ms;
		repeat_wall_ms   = repeat.wall_ms;
		legacy_wall_ms   = legacy.wall_ms;
		planless_hash    = grid_hash(planless.grid);
		repeat_hash      = grid_hash(repeat.grid);
		legacy_hash      = grid_hash(legacy.grid);
		execution_deterministic = planless.grid.y == repeat.grid.y && planless.grid.cbcr == repeat.grid.cbcr;
		planless_legacy_difference = difference(planless.grid, legacy.grid);
		execution_within_tolerance = planless_legacy_difference.max_abs <= 1;
		planless_execution_stats = planless.stats;
		legacy_execution_stats   = legacy.stats;
		planless_cache_stats      = planless.cache_stats;
		execution_zero_expansion = execution_zero_expansion &&
		                           planless.stats.host_expanded_transform_items_created == 0U &&
		                           planless.stats.host_output_block_source_lists_created == 0U &&
		                           planless.stats.host_global_transform_sort_items == 0U;
		execution_kernel_launched = planless.stats.planless_transform_kernel_launch_count > 0U;
		execution_strategy_accounting_valid =
		    planless.stats.rowgroup_count > 0U &&
		    planless.stats.run_interval_exact_rowgroup_count + planless.stats.bitmap_exact_rowgroup_count +
		            planless.stats.full_rowgroup_strategy_count ==
		        planless.stats.rowgroup_count;
		const auto expected_workset_capacity =
		    static_cast<size_t>(options.workset_capacity_mib) * 1024U * 1024U;
		execution_descriptor_bounds =
		    planless.stats.compact_plan_bytes > 0U &&
		    planless.stats.compact_plan_bytes <= planless.stats.compact_plan_peak_bytes &&
		    planless.stats.planless_image_descriptor_count == requests.size();
		execution_active_output_schedule_valid =
		    planless.stats.planless_transform_full_scan_output_block_count > 0U &&
		    planless.stats.planless_transform_output_block_count > 0U &&
		    planless.stats.planless_transform_output_block_count +
		            planless.stats.planless_transform_skipped_output_block_count ==
		        planless.stats.planless_transform_full_scan_output_block_count &&
		    planless.stats.planless_transform_active_output_index_bytes ==
		        planless.stats.planless_transform_output_block_count * sizeof(uint32_t) &&
		    planless.stats.planless_transform_active_output_schedule_build_count == 1U &&
		    planless.stats.planless_transform_active_output_workset_count == planless.stats.workset_count &&
		    planless.stats.planless_transform_active_output_offset_bytes ==
		        (planless.stats.workset_count + 1U) * sizeof(uint64_t) &&
		    planless.stats.planless_transform_active_output_offsets_valid;
		execution_workset_bounds =
		    planless.stats.decode_workset_capacity_bytes == expected_workset_capacity &&
		    planless.stats.max_estimated_decode_workset_bytes <= expected_workset_capacity &&
		    planless.stats.bounded_double_buffer_peak_estimated_bytes <= expected_workset_capacity &&
		    planless.stats.oversized_decode_rowgroup_count == 0U && planless.stats.workset_count > 0U;
		execution_io_bounds =
		    planless.stats.host_io_staged_rowgroups == 0U &&
		    planless.stats.actual_vector_count <= planless.stats.full_vector_count &&
		    planless.stats.compressed_payload_bytes_read <= planless.stats.full_compressed_payload_bytes;
		execution_resource_bounds = execution_descriptor_bounds && execution_workset_bounds && execution_io_bounds &&
		                            execution_active_output_schedule_valid;
		execution_cache_contract_valid =
		    !planless.stats.cache_enabled && !planless.stats.exact_batch_plan_cache_enabled &&
		    planless.stats.plan_cache_hits == 0U && planless.stats.plan_cache_misses == 0U &&
		    planless.stats.plan_cache_evictions == 0U && planless.stats.sparse_vector_cache_hits == 0U &&
		    planless.stats.sparse_vector_cache_misses == 0U && planless.stats.dct_resize_weight_cache_hits == 0U &&
		    planless.stats.dct_resize_weight_cache_misses == 0U &&
		    planless.stats.dct_conversion_matrix_cache_hits == 0U &&
		    planless.stats.dct_conversion_matrix_cache_misses == 0U && planless.cache_stats.capacity_bytes == 0U &&
		    planless.cache_stats.resident_bytes == 0U && planless.cache_stats.peak_resident_bytes == 0U &&
		    planless.cache_stats.resident_rowgroups == 0U && planless.cache_stats.peak_resident_rowgroups == 0U &&
		    planless.cache_stats.hits == 0U && planless.cache_stats.misses == 0U &&
		    planless.cache_stats.inserts == 0U && planless.cache_stats.evictions == 0U;
	}

	const auto load_ms = std::chrono::duration<double, std::milli>(load_end - load_begin).count();
	const auto compact_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();
	std::ostringstream json;
	json << std::setprecision(12)
	     << "{\"schema_version\":\"galp_block_major_plan_audit_v1\",\"pattern\":\"" << options.pattern
	     << "\",\"request_source\":\"" << (options.image_ids.empty() ? "pattern" : "explicit-image-ids")
	     << "\",\"explicit_crops\":" << (options.explicit_crops ? "true" : "false")
	     << ",\"require_grayscale\":" << (options.require_grayscale ? "true" : "false")
	     << ",\"grayscale_requirement_met\":" << (grayscale_requirement_met ? "true" : "false")
	     << ",\"require_cross_shard\":" << (options.require_cross_shard ? "true" : "false")
	     << ",\"cross_shard_requirement_met\":" << (cross_shard_requirement_met ? "true" : "false")
	     << ",\"requested_shard_count\":" << requested_shards.size() << ",\"request_count\":" << requests.size()
	     << ",\"decode_batch_rowgroups\":" << options.decode_batch_rowgroups
	     << ",\"prefetch_workers\":" << options.prefetch_workers
	     << ",\"workset_capacity_mib\":" << options.workset_capacity_mib
	     << ",\"descriptor_load_ms\":" << load_ms
	     << ",\"compact_planning_ms\":" << compact_ms << ",\"compact_plan_bytes\":"
	     << plan.stats.compact_plan_bytes << ",\"compact_plan_peak_bytes\":" << plan.stats.compact_plan_peak_bytes
	     << ",\"unique_image_count\":" << plan.stats.unique_image_count << ",\"duplicate_output_count\":"
	     << plan.stats.duplicate_output_count << ",\"request_sort_items\":" << plan.stats.request_sort_items
	     << ",\"touched_block_groups\":" << plan.stats.touched_block_groups << ",\"group_rank_runs\":"
	     << plan.stats.group_rank_runs << ",\"selected_rowgroups\":" << plan.stats.selected_rowgroups
	     << ",\"selected_vector_runs\":" << plan.stats.selected_vector_runs << ",\"selected_vectors\":"
		     << compact_selected.size() << ",\"touched_rank_cells\":" << plan.stats.touched_rank_cells
		     << ",\"rank_payload_bytes\":" << plan.stats.rank_payload_bytes
		     << ",\"touched_quant_tables\":" << plan.stats.touched_quant_tables
	     << ",\"expanded_transform_items\":" << plan.stats.expanded_transform_items
	     << ",\"global_transform_sort_items\":" << plan.stats.global_transform_sort_items
	     << ",\"legacy_compared\":" << (options.compare_legacy ? "true" : "false")
	     << ",\"legacy_planning_ms\":" << legacy_ms << ",\"legacy_selected_vectors\":"
	     << legacy_selected_vectors << ",\"legacy_expanded_transform_items\":" << legacy_expanded_items
	     << ",\"legacy_sort_items\":" << legacy_sort_items << ",\"selected_vectors_equal\":"
	     << (vectors_equal ? "true" : "false") << ",\"execution_requested\":"
	     << (options.execute ? "true" : "false") << ",\"planless_execution_selected\":"
	     << (execution_selected ? "true" : "false") << ",\"planless_zero_expansion\":"
	     << (execution_zero_expansion ? "true" : "false") << ",\"planless_repeat_deterministic\":"
	     << (execution_deterministic ? "true" : "false") << ",\"planless_legacy_within_tolerance\":"
	     << (execution_within_tolerance ? "true" : "false") << ",\"planless_legacy_mismatch_count\":"
	     << planless_legacy_difference.mismatches << ",\"planless_legacy_max_abs_difference\":"
	     << planless_legacy_difference.max_abs << ",\"execution_kernel_launched\":"
	     << (execution_kernel_launched ? "true" : "false")
	     << ",\"execution_strategy_accounting_valid\":"
	     << (execution_strategy_accounting_valid ? "true" : "false")
	     << ",\"execution_descriptor_bounds\":" << (execution_descriptor_bounds ? "true" : "false")
	     << ",\"execution_workset_bounds\":" << (execution_workset_bounds ? "true" : "false")
	     << ",\"execution_io_bounds\":" << (execution_io_bounds ? "true" : "false")
	     << ",\"execution_resource_bounds\":" << (execution_resource_bounds ? "true" : "false")
	     << ",\"execution_active_output_schedule_valid\":"
	     << (execution_active_output_schedule_valid ? "true" : "false")
	     << ",\"execution_cache_contract_valid\":"
	     << (execution_cache_contract_valid ? "true" : "false") << ",\"planless_wall_ms\":" << planless_wall_ms
	     << ",\"planless_repeat_wall_ms\":" << repeat_wall_ms << ",\"legacy_wall_ms\":" << legacy_wall_ms
	     << ",\"planless_hash\":\"" << planless_hash << "\",\"planless_repeat_hash\":\"" << repeat_hash
	     << "\",\"legacy_hash\":\"" << legacy_hash << "\",\"execution_rowgroups\":"
	     << planless_execution_stats.rowgroup_count << ",\"execution_worksets\":"
	     << planless_execution_stats.workset_count
	     << ",\"execution_planless_image_descriptor_count\":"
	     << planless_execution_stats.planless_image_descriptor_count
	     << ",\"execution_compact_plan_bytes\":" << planless_execution_stats.compact_plan_bytes
	     << ",\"execution_compact_plan_peak_bytes\":" << planless_execution_stats.compact_plan_peak_bytes
	     << ",\"execution_coordinate_group_lookup_count\":"
	     << planless_execution_stats.coordinate_group_lookup_count
	     << ",\"execution_coordinate_group_index_entries\":"
	     << planless_execution_stats.coordinate_group_index_entries
	     << ",\"execution_coordinate_group_index_populated\":"
	     << planless_execution_stats.coordinate_group_index_populated
	     << ",\"execution_coordinate_group_index_holes\":"
	     << planless_execution_stats.coordinate_group_index_holes
	     << ",\"execution_coordinate_group_index_bytes\":"
	     << planless_execution_stats.coordinate_group_index_bytes
	     << ",\"execution_coordinate_group_index_density\":"
	     << planless_execution_stats.coordinate_group_index_density
	     << ",\"execution_host_io_staged_rowgroups\":" << planless_execution_stats.host_io_staged_rowgroups
	     << ",\"execution_actual_vectors\":"
	     << planless_execution_stats.actual_vector_count << ",\"execution_full_vectors\":"
	     << planless_execution_stats.full_vector_count << ",\"execution_compressed_bytes_read\":"
	     << planless_execution_stats.compressed_payload_bytes_read
	     << ",\"execution_full_compressed_bytes\":" << planless_execution_stats.full_compressed_payload_bytes
	     << ",\"execution_read_amplification\":" << planless_execution_stats.read_amplification
	     << ",\"execution_gpu_peak_bytes\":" << planless_execution_stats.galp_native_device_peak_in_use_bytes
	     << ",\"execution_pinned_peak_bytes\":" << planless_execution_stats.galp_native_pinned_peak_in_use_bytes
	     << ",\"legacy_execution_gpu_peak_bytes\":" << legacy_execution_stats.galp_native_device_peak_in_use_bytes
	     << ",\"legacy_execution_pinned_peak_bytes\":"
	     << legacy_execution_stats.galp_native_pinned_peak_in_use_bytes
	     << ",\"execution_decode_workset_capacity_bytes\":"
	     << planless_execution_stats.decode_workset_capacity_bytes
	     << ",\"execution_max_estimated_decode_workset_bytes\":"
	     << planless_execution_stats.max_estimated_decode_workset_bytes
	     << ",\"execution_oversized_decode_rowgroups\":"
	     << planless_execution_stats.oversized_decode_rowgroup_count
	     << ",\"execution_run_interval_exact_rowgroups\":"
	     << planless_execution_stats.run_interval_exact_rowgroup_count
	     << ",\"execution_bitmap_exact_rowgroups\":"
	     << planless_execution_stats.bitmap_exact_rowgroup_count
	     << ",\"execution_full_rowgroup_strategy_count\":"
	     << planless_execution_stats.full_rowgroup_strategy_count
	     << ",\"execution_bounded_double_buffer_enabled\":"
	     << (planless_execution_stats.bounded_double_buffer_enabled ? "true" : "false")
	     << ",\"execution_bounded_double_buffer_policy\":\""
	     << planless_execution_stats.bounded_double_buffer_policy << "\""
	     << ",\"execution_bounded_double_buffer_candidate\":"
	     << (planless_execution_stats.bounded_double_buffer_candidate ? "true" : "false")
	     << ",\"execution_planless_transform_full_scan_output_blocks\":"
	     << planless_execution_stats.planless_transform_full_scan_output_block_count
	     << ",\"execution_planless_transform_active_output_blocks\":"
	     << planless_execution_stats.planless_transform_output_block_count
	     << ",\"execution_planless_transform_skipped_output_blocks\":"
	     << planless_execution_stats.planless_transform_skipped_output_block_count
	     << ",\"execution_planless_transform_active_output_index_bytes\":"
	     << planless_execution_stats.planless_transform_active_output_index_bytes
	     << ",\"execution_planless_transform_active_output_offset_bytes\":"
	     << planless_execution_stats.planless_transform_active_output_offset_bytes
	     << ",\"execution_planless_transform_active_output_schedule_peak_bytes\":"
	     << planless_execution_stats.planless_transform_active_output_schedule_peak_bytes
	     << ",\"execution_planless_transform_source_contribution_count\":"
	     << planless_execution_stats.planless_transform_source_contribution_count
	     << ",\"execution_planless_transform_source_contribution_visit_count\":"
	     << planless_execution_stats.planless_transform_source_contribution_visit_count
	     << ",\"execution_planless_transform_output_workset_ownership_count\":"
	     << planless_execution_stats.planless_transform_output_workset_ownership_count
	     << ",\"execution_planless_transform_active_output_workset_count\":"
	     << planless_execution_stats.planless_transform_active_output_workset_count
	     << ",\"execution_planless_transform_active_output_schedule_build_count\":"
	     << planless_execution_stats.planless_transform_active_output_schedule_build_count
	     << ",\"execution_planless_transform_active_output_offsets_valid\":"
	     << (planless_execution_stats.planless_transform_active_output_offsets_valid ? "true" : "false")
	     << ",\"execution_planless_transform_active_output_planning_ms\":"
	     << planless_execution_stats.planless_transform_active_output_planning_ms
	     << ",\"execution_planless_transform_group_workset_build_ms\":"
	     << planless_execution_stats.planless_transform_group_workset_build_ms
	     << ",\"execution_planless_transform_active_output_count_ms\":"
	     << planless_execution_stats.planless_transform_active_output_count_ms
	     << ",\"execution_planless_transform_active_output_prefix_ms\":"
	     << planless_execution_stats.planless_transform_active_output_prefix_ms
	     << ",\"execution_planless_transform_active_output_fill_ms\":"
	     << planless_execution_stats.planless_transform_active_output_fill_ms
	     << ",\"execution_planless_transform_gpu_kernel_ms\":"
	     << planless_execution_stats.planless_transform_gpu_kernel_ms
	     << ",\"execution_bounded_double_buffer_worksets\":"
	     << planless_execution_stats.bounded_double_buffer_workset_count
	     << ",\"execution_bounded_double_buffer_peak_estimated_bytes\":"
	     << planless_execution_stats.bounded_double_buffer_peak_estimated_bytes
	     << ",\"execution_decoded_cache_capacity_bytes\":" << planless_cache_stats.capacity_bytes
	     << ",\"execution_decoded_cache_current_bytes\":" << planless_cache_stats.resident_bytes
	     << ",\"execution_decoded_cache_peak_bytes\":" << planless_cache_stats.peak_resident_bytes
	     << ",\"execution_decoded_cache_current_entries\":" << planless_cache_stats.resident_rowgroups
	     << ",\"execution_decoded_cache_peak_entries\":" << planless_cache_stats.peak_resident_rowgroups
	     << ",\"execution_decoded_cache_hits\":" << planless_cache_stats.hits
	     << ",\"execution_decoded_cache_misses\":" << planless_cache_stats.misses
	     << ",\"execution_decoded_cache_inserts\":" << planless_cache_stats.inserts
	     << ",\"execution_decoded_cache_evictions\":" << planless_cache_stats.evictions
	     << ",\"execution_plan_cache_hits\":" << planless_execution_stats.plan_cache_hits
	     << ",\"execution_plan_cache_misses\":" << planless_execution_stats.plan_cache_misses
	     << ",\"execution_plan_cache_evictions\":" << planless_execution_stats.plan_cache_evictions
	     << ",\"execution_resize_cache_hits\":" << planless_execution_stats.dct_resize_weight_cache_hits
	     << ",\"execution_resize_cache_misses\":" << planless_execution_stats.dct_resize_weight_cache_misses
	     << ",\"execution_conversion_cache_hits\":"
	     << planless_execution_stats.dct_conversion_matrix_cache_hits
	     << ",\"execution_conversion_cache_misses\":"
	     << planless_execution_stats.dct_conversion_matrix_cache_misses
	     << "}\n";
	std::cout << json.str();
	if (!options.output_json.empty()) {
		std::ofstream output(options.output_json, std::ios::trunc);
		output << json.str();
		if (!output) {
			throw std::runtime_error("failed to write output JSON");
		}
	}
	const bool execution_pass =
	    !options.execute ||
	    (execution_selected && execution_zero_expansion && execution_deterministic && execution_within_tolerance &&
	     execution_kernel_launched && execution_strategy_accounting_valid && execution_resource_bounds &&
	     execution_cache_contract_valid);
	return vectors_equal && execution_pass ? 0 : 3;
} catch (const std::exception& error) {
	std::cerr << "block-major plan audit failed: " << error.what() << '\n';
	return 1;
}
