#include "galp/jpeg_dct.hpp"
#include "galp/profiles/rgbnomore.hpp"
#include "format/reader.cuh"
#include "jpeg/jpeg_dct_metadata.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kDescriptorRssBudgetBytes = 256ULL * 1024ULL * 1024ULL;

struct Options {
	std::filesystem::path manifest;
	std::filesystem::path output;
	std::string           workload;
	uint32_t              first_image       = 0U;
	size_t                batch_size        = 64U;
	size_t                coefficient_count = 64U;
	uint32_t              hold_ms           = 0U;
	size_t                iterations        = 20U;
	size_t                warmup_iterations = 5U;
	bool                  memory_probe      = false;
	bool                  planner_probe     = false;
	bool                  reader_probe      = false;
};

struct ProcessMemory {
	uint64_t rss_bytes      = 0U;
	uint64_t pss_bytes      = 0U;
	uint64_t pss_file_bytes = 0U;
	uint64_t pss_anon_bytes = 0U;
};

struct ProcessIo {
	uint64_t read_bytes = 0U;
	uint64_t rchar      = 0U;
	uint64_t syscr      = 0U;
};

struct MappedFlsMemory {
	uint64_t rss_bytes = 0U;
	uint64_t pss_bytes = 0U;
};

uint64_t parse_u64(const std::string_view name, const char* value) {
	try {
		if (value == nullptr || value[0] == '\0' || value[0] == '-') {
			throw std::invalid_argument("negative or empty value");
		}
		size_t     consumed = 0U;
		const auto parsed   = std::stoull(value, &consumed);
		if (consumed != std::string_view(value).size()) {
			throw std::invalid_argument("trailing characters");
		}
		return parsed;
	} catch (const std::exception&) {
		throw std::invalid_argument(std::string(name) + " requires an unsigned integer");
	}
}

void print_usage(const char* program) {
	std::cerr << "Usage:\n"
	          << "  " << program
	          << " --manifest manifest.bin --memory-probe [--first-image N] [--batch-size N] [--hold-ms N]"
	             " [--output result.json]\n"
	          << "  " << program
	          << " --manifest manifest.bin --planner-probe [--first-image N] [--batch-size N]"
	             " [--warmup N] [--iterations N] [--output result.json]\n"
	          << "  " << program
	          << " --manifest manifest.bin --reader-probe [--first-image N] [--batch-size N]"
	             " [--output result.json]\n"
	          << "  " << program
	          << " --manifest manifest.bin --workload full-all|crop-all|full-prefix|crop-prefix"
	             " [--coefficients 1|4|8|16|32|64] [--first-image N] [--batch-size N]"
	             " [--hold-ms N] [--output result.json]\n"
	          << "Formal cold-I/O measurements should launch exactly one workload per fresh process.\n";
}

bool parse_args(const int argc, char** argv, Options& options) {
	for (int index = 1; index < argc; ++index) {
		const std::string_view argument = argv[index];
		if (argument == "--manifest" && index + 1 < argc) {
			options.manifest = argv[++index];
		} else if (argument == "--output" && index + 1 < argc) {
			options.output = argv[++index];
		} else if (argument == "--memory-probe") {
			options.memory_probe = true;
		} else if (argument == "--planner-probe") {
			options.planner_probe = true;
		} else if (argument == "--reader-probe") {
			options.reader_probe = true;
		} else if (argument == "--workload" && index + 1 < argc) {
			options.workload = argv[++index];
		} else if (argument == "--first-image" && index + 1 < argc) {
			const auto value = parse_u64(argument, argv[++index]);
			if (value > std::numeric_limits<uint32_t>::max()) {
				throw std::invalid_argument("--first-image exceeds uint32 range");
			}
			options.first_image = static_cast<uint32_t>(value);
		} else if (argument == "--batch-size" && index + 1 < argc) {
			const auto value = parse_u64(argument, argv[++index]);
			if (value == 0U || value > std::numeric_limits<size_t>::max()) {
				throw std::invalid_argument("--batch-size must be positive and addressable");
			}
			options.batch_size = static_cast<size_t>(value);
		} else if (argument == "--coefficients" && index + 1 < argc) {
			const auto value = parse_u64(argument, argv[++index]);
			if (value > std::numeric_limits<size_t>::max()) {
				throw std::invalid_argument("--coefficients exceeds size_t range");
			}
			options.coefficient_count = static_cast<size_t>(value);
		} else if (argument == "--hold-ms" && index + 1 < argc) {
			const auto value = parse_u64(argument, argv[++index]);
			if (value > std::numeric_limits<uint32_t>::max()) {
				throw std::invalid_argument("--hold-ms exceeds uint32 range");
			}
			options.hold_ms = static_cast<uint32_t>(value);
		} else if (argument == "--iterations" && index + 1 < argc) {
			const auto value = parse_u64(argument, argv[++index]);
			if (value == 0U || value > std::numeric_limits<size_t>::max()) {
				throw std::invalid_argument("--iterations must be positive and addressable");
			}
			options.iterations = static_cast<size_t>(value);
		} else if (argument == "--warmup" && index + 1 < argc) {
			const auto value = parse_u64(argument, argv[++index]);
			if (value > std::numeric_limits<size_t>::max()) {
				throw std::invalid_argument("--warmup exceeds size_t range");
			}
			options.warmup_iterations = static_cast<size_t>(value);
		} else if (argument == "--help" || argument == "-h") {
			return false;
		} else {
			throw std::invalid_argument("unknown or incomplete argument: " + std::string(argument));
		}
	}
	if (options.manifest.empty()) {
		throw std::invalid_argument("--manifest is required");
	}
	const auto selected_modes = static_cast<unsigned>(options.memory_probe) +
	                            static_cast<unsigned>(options.planner_probe) +
	                            static_cast<unsigned>(options.reader_probe) +
	                            static_cast<unsigned>(!options.workload.empty());
	if (selected_modes != 1U) {
		throw std::invalid_argument(
		    "select exactly one of --memory-probe, --planner-probe, --reader-probe, and --workload");
	}
	if (!options.workload.empty() && options.workload != "full-all" && options.workload != "crop-all" &&
	    options.workload != "full-prefix" && options.workload != "crop-prefix") {
		throw std::invalid_argument("unknown --workload");
	}
	constexpr std::array<size_t, 6> coefficient_counts {1U, 4U, 8U, 16U, 32U, 64U};
	if (std::find(coefficient_counts.begin(), coefficient_counts.end(), options.coefficient_count) ==
	    coefficient_counts.end()) {
		throw std::invalid_argument("--coefficients must be one of 1,4,8,16,32,64");
	}
	if ((options.workload == "full-all" || options.workload == "crop-all") && options.coefficient_count != 64U) {
		throw std::invalid_argument("all-coefficient workloads require --coefficients 64");
	}
	return true;
}

uint64_t saturating_delta(const uint64_t after, const uint64_t before) {
	return after >= before ? after - before : 0U;
}

void assign_kib_field(const std::string_view key, const uint64_t kib, ProcessMemory& memory) {
	const uint64_t bytes = kib * 1024U;
	if (key == "Rss") {
		memory.rss_bytes = bytes;
	} else if (key == "Pss") {
		memory.pss_bytes = bytes;
	} else if (key == "Pss_File") {
		memory.pss_file_bytes = bytes;
	} else if (key == "Pss_Anon") {
		memory.pss_anon_bytes = bytes;
	}
}

ProcessMemory read_process_memory() {
	std::ifstream input("/proc/self/smaps_rollup");
	if (!input) {
		throw std::runtime_error("cannot read /proc/self/smaps_rollup");
	}
	ProcessMemory memory;
	std::string   line;
	while (std::getline(input, line)) {
		const auto colon = line.find(':');
		if (colon == std::string::npos) {
			continue;
		}
		std::istringstream value(line.substr(colon + 1U));
		uint64_t           kib = 0U;
		value >> kib;
		assign_kib_field(std::string_view(line).substr(0U, colon), kib, memory);
	}
	return memory;
}

ProcessIo read_process_io() {
	std::ifstream input("/proc/self/io");
	if (!input) {
		throw std::runtime_error("cannot read /proc/self/io");
	}
	ProcessIo   io;
	std::string line;
	while (std::getline(input, line)) {
		const auto colon = line.find(':');
		if (colon == std::string::npos) {
			continue;
		}
		uint64_t value = 0U;
		std::istringstream(line.substr(colon + 1U)) >> value;
		const auto key = std::string_view(line).substr(0U, colon);
		if (key == "read_bytes") {
			io.read_bytes = value;
		} else if (key == "rchar") {
			io.rchar = value;
		} else if (key == "syscr") {
			io.syscr = value;
		}
	}
	return io;
}

std::string trim(std::string value) {
	const auto first = value.find_first_not_of(" \t");
	if (first == std::string::npos) {
		return {};
	}
	const auto last = value.find_last_not_of(" \t");
	return value.substr(first, last - first + 1U);
}

bool is_mapping_header(const std::string& line) {
	const auto dash = line.find('-');
	return dash != std::string::npos && dash != 0U &&
	       std::all_of(line.begin(), line.begin() + static_cast<std::ptrdiff_t>(dash), [](const unsigned char ch) {
		       return std::isxdigit(ch) != 0;
	       });
}

MappedFlsMemory read_mapped_fls_memory() {
	std::ifstream input("/proc/self/smaps");
	if (!input) {
		throw std::runtime_error("cannot read /proc/self/smaps");
	}
	MappedFlsMemory memory;
	bool            selected = false;
	std::string     line;
	while (std::getline(input, line)) {
		if (is_mapping_header(line)) {
			std::istringstream header(line);
			std::string        address;
			std::string        permissions;
			std::string        offset;
			std::string        device;
			std::string        inode;
			header >> address >> permissions >> offset >> device >> inode;
			std::string path;
			std::getline(header, path);
			path     = trim(std::move(path));
			selected = path.ends_with(".fls") || path.ends_with(".fls (deleted)");
			continue;
		}
		if (!selected) {
			continue;
		}
		const auto colon = line.find(':');
		if (colon == std::string::npos) {
			continue;
		}
		uint64_t kib = 0U;
		std::istringstream(line.substr(colon + 1U)) >> kib;
		const auto key = std::string_view(line).substr(0U, colon);
		if (key == "Rss") {
			memory.rss_bytes += kib * 1024U;
		} else if (key == "Pss") {
			memory.pss_bytes += kib * 1024U;
		}
	}
	return memory;
}

std::string json_escape(const std::string_view value) {
	std::ostringstream output;
	for (const unsigned char ch : value) {
		switch (ch) {
		case '\\':
			output << "\\\\";
			break;
		case '"':
			output << "\\\"";
			break;
		case '\n':
			output << "\\n";
			break;
		case '\r':
			output << "\\r";
			break;
		case '\t':
			output << "\\t";
			break;
		default:
			if (ch < 0x20U) {
				output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(ch)
				       << std::dec << std::setfill(' ');
			} else {
				output << static_cast<char>(ch);
			}
		}
	}
	return output.str();
}

double elapsed_ms(const Clock::time_point begin, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::vector<galp::jpeg::JpegDctImageCropRequest>
make_requests(galp::jpeg::JpegDctShardDatasetReader& reader, const Options& options, const bool cropped) {
	if (options.first_image >= reader.image_count() ||
	    options.batch_size > reader.image_count() - options.first_image) {
		throw std::out_of_range("requested image batch exceeds the manifest");
	}
	if (options.batch_size - 1U > std::numeric_limits<uint32_t>::max() - options.first_image) {
		throw std::out_of_range("requested image batch exceeds the uint32 image-id range");
	}
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	requests.reserve(options.batch_size);
	for (size_t offset = 0U; offset < options.batch_size; ++offset) {
		const auto                 image_id = static_cast<uint32_t>(options.first_image + offset);
		galp::jpeg::JpegDctCropBox crop {};
		if (cropped) {
			const auto metadata = reader.ImageMetadata(image_id);
			if (metadata.image_width == 0U || metadata.image_height == 0U) {
				throw std::runtime_error("crop audit found an image with zero dimensions");
			}
			crop.x      = metadata.image_width / 4U;
			crop.y      = metadata.image_height / 4U;
			crop.width  = std::max(1U, metadata.image_width / 2U);
			crop.height = std::max(1U, metadata.image_height / 2U);
			crop.width  = std::min(crop.width, metadata.image_width - crop.x);
			crop.height = std::min(crop.height, metadata.image_height - crop.y);
		}
		requests.push_back({image_id, crop});
	}
	return requests;
}

galp::jpeg::JpegDctDeviceBatchOptions make_device_options(const size_t coefficient_count) {
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout                    = galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	options.cache_capacity_bytes      = 0U;
	options.plan_cache_capacity       = 0U;
	options.enable_rowgroup_prefetch  = false;
	options.enable_planless_execution = true;
	options.coefficient_selection.coefficients.resize(coefficient_count);
	for (size_t index = 0U; index < coefficient_count; ++index) {
		options.coefficient_selection.coefficients[index] = static_cast<uint8_t>(index);
	}
	return options;
}

galp::jpeg::JpegDctDeviceBatchOptions make_transformed_planner_options() {
	auto options                      = make_device_options(64U);
	options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform            = galp::profiles::rgbnomore_val_dct_grid_transform();
	options.enable_rowgroup_prefetch  = false;
	options.enable_planless_execution = true;
	return options;
}

double sample_mean(const std::vector<double>& samples) {
	double total = 0.0;
	for (const auto sample : samples) {
		total += sample;
	}
	return total / static_cast<double>(samples.size());
}

double sample_percentile(std::vector<double> samples, const double percentile) {
	std::sort(samples.begin(), samples.end());
	const auto rank = static_cast<size_t>(
	    std::ceil(percentile * static_cast<double>(samples.size())));
	return samples[std::min(samples.size() - 1U, std::max<size_t>(1U, rank) - 1U)];
}

void emit_result(const Options& options, const std::string& payload) {
	if (!options.output.empty()) {
		if (std::filesystem::exists(options.output)) {
			throw std::runtime_error("audit output already exists: " + options.output.string());
		}
		if (options.output.has_parent_path()) {
			std::filesystem::create_directories(options.output.parent_path());
		}
		std::ofstream output(options.output);
		if (!output) {
			throw std::runtime_error("cannot create audit output: " + options.output.string());
		}
		output << payload << '\n';
	}
	std::cout << payload << '\n';
	std::cout.flush();
	if (options.hold_ms != 0U) {
		std::this_thread::sleep_for(std::chrono::milliseconds(options.hold_ms));
	}
}

std::string run_memory_probe(const Options& options) {
	const auto                                memory_before = read_process_memory();
	const auto                                begin         = Clock::now();
	galp::jpeg::JpegDctShardDatasetReader     reader(options.manifest);
	const auto                                reader_ready = Clock::now();
	const auto                                requests     = make_requests(reader, options, false);
	const auto                                preview = reader.PlanDeviceDctBatch(requests, make_device_options(64U));
	const auto                                plan_ready = Clock::now();
	std::map<uint32_t, std::vector<uint32_t>> rowgroups_by_shard;
	for (const auto& rowgroup : preview.rowgroups) {
		rowgroups_by_shard[rowgroup.shard_id].push_back(rowgroup.rowgroup_index);
	}
	uint64_t touched_payload_bytes = 0U;
	for (const auto& [shard, rowgroups] : rowgroups_by_shard) {
		touched_payload_bytes += reader.RowgroupStorageBytes(shard, rowgroups);
	}
	const auto descriptor_ready = Clock::now();
	const auto memory_after     = read_process_memory();
	const auto mapped_fls       = read_mapped_fls_memory();
	const auto incremental_rss  = saturating_delta(memory_after.rss_bytes, memory_before.rss_bytes);
	const auto incremental_pss  = saturating_delta(memory_after.pss_bytes, memory_before.pss_bytes);

	std::ostringstream json;
	json << std::boolalpha << "{\n"
	     << "  \"schema\": \"galp-compact-v3-runtime-audit-v1\",\n"
	     << "  \"mode\": \"memory\",\n"
	     << "  \"manifest\": \"" << json_escape(std::filesystem::absolute(options.manifest).string()) << "\",\n"
	     << "  \"image_count\": " << reader.image_count() << ",\n"
	     << "  \"batch_size\": " << options.batch_size << ",\n"
	     << "  \"planned_rowgroup_count\": " << preview.rowgroups.size() << ",\n"
	     << "  \"touched_payload_bytes\": " << touched_payload_bytes << ",\n"
	     << "  \"reader_startup_ms\": " << elapsed_ms(begin, reader_ready) << ",\n"
	     << "  \"first_plan_ms\": " << elapsed_ms(reader_ready, plan_ready) << ",\n"
	     << "  \"descriptor_touch_ms\": " << elapsed_ms(plan_ready, descriptor_ready) << ",\n"
	     << "  \"first_access_ms\": " << elapsed_ms(begin, descriptor_ready) << ",\n"
	     << "  \"rss_before_bytes\": " << memory_before.rss_bytes << ",\n"
	     << "  \"rss_after_bytes\": " << memory_after.rss_bytes << ",\n"
	     << "  \"pss_before_bytes\": " << memory_before.pss_bytes << ",\n"
	     << "  \"pss_after_bytes\": " << memory_after.pss_bytes << ",\n"
	     << "  \"pss_file_after_bytes\": " << memory_after.pss_file_bytes << ",\n"
	     << "  \"pss_anon_after_bytes\": " << memory_after.pss_anon_bytes << ",\n"
	     << "  \"descriptor_and_index_incremental_rss_bytes\": " << incremental_rss << ",\n"
	     << "  \"descriptor_and_index_incremental_pss_bytes\": " << incremental_pss << ",\n"
	     << "  \"actual_fls_mmap_resident_rss_bytes\": " << mapped_fls.rss_bytes << ",\n"
	     << "  \"actual_fls_mmap_resident_pss_bytes\": " << mapped_fls.pss_bytes << ",\n"
	     << "  \"compact_reader_declared_index_bytes\": " << preview.compact_reader_total_bytes << ",\n"
	     << "  \"descriptor_rss_budget_bytes\": " << kDescriptorRssBudgetBytes << ",\n"
	     << "  \"descriptor_rss_gate_pass\": " << (incremental_rss <= kDescriptorRssBudgetBytes) << "\n"
	     << '}';
	return json.str();
}

std::string run_planner_probe(const Options& options) {
	const auto manifest = galp::jpeg::detail::read_jpeg_dct_shard_manifest_file(options.manifest);
	galp::jpeg::JpegDctShardDatasetReader reader(options.manifest);
	auto requests = make_requests(reader, options, true);
	for (size_t index = 0U; index < requests.size(); ++index) {
		requests[index].horizontal_flip = (index & 1U) != 0U;
	}
	const auto device_options = make_transformed_planner_options();
	for (size_t iteration = 0U; iteration < options.warmup_iterations; ++iteration) {
		static_cast<void>(reader.PlanDeviceDctBatch(requests, device_options));
	}

	std::vector<double> internal_samples;
	std::vector<double> wall_samples;
	internal_samples.reserve(options.iterations);
	wall_samples.reserve(options.iterations);
	galp::jpeg::JpegDctDeviceBatchPlanPreview preview;
	for (size_t iteration = 0U; iteration < options.iterations; ++iteration) {
		const auto begin = Clock::now();
		preview          = reader.PlanDeviceDctBatch(requests, device_options);
		const auto end   = Clock::now();
		internal_samples.push_back(preview.planning_ms);
		wall_samples.push_back(elapsed_ms(begin, end));
	}

	const auto planned_rowgroup_count = preview.rowgroup_vector_plans.size();
	std::ostringstream json;
	json << std::boolalpha << "{\n"
	     << "  \"schema\": \"galp-compact-v3-runtime-audit-v1\",\n"
	     << "  \"mode\": \"planner\",\n"
	     << "  \"manifest\": \"" << json_escape(std::filesystem::absolute(options.manifest).string()) << "\",\n"
	     << "  \"manifest_version\": " << manifest.version << ",\n"
	     << "  \"manifest_physical_layout\": \"" << json_escape(manifest.physical_layout) << "\",\n"
	     << "  \"manifest_descriptor_kind\": \"" << json_escape(manifest.descriptor_kind) << "\",\n"
	     << "  \"image_count\": " << reader.image_count() << ",\n"
	     << "  \"first_image\": " << options.first_image << ",\n"
	     << "  \"batch_size\": " << options.batch_size << ",\n"
	     << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n"
	     << "  \"iterations\": " << options.iterations << ",\n"
	     << "  \"crop\": \"center-half\",\n"
	     << "  \"horizontal_flip\": \"alternating\",\n"
	     << "  \"layout\": \"transformed-dct-grid\",\n"
	     << "  \"uses_planless_fixed_transform\": " << preview.uses_planless_fixed_transform << ",\n"
	     << "  \"planning_ms_mean\": " << sample_mean(internal_samples) << ",\n"
	     << "  \"planning_ms_min\": " << *std::min_element(internal_samples.begin(), internal_samples.end()) << ",\n"
	     << "  \"planning_ms_p50\": " << sample_percentile(internal_samples, 0.50) << ",\n"
	     << "  \"planning_ms_p95\": " << sample_percentile(internal_samples, 0.95) << ",\n"
	     << "  \"planning_ms_max\": " << *std::max_element(internal_samples.begin(), internal_samples.end()) << ",\n"
	     << "  \"planning_ms_per_image_mean\": "
	     << sample_mean(internal_samples) / static_cast<double>(options.batch_size) << ",\n"
	     << "  \"wall_ms_mean\": " << sample_mean(wall_samples) << ",\n"
	     << "  \"wall_ms_min\": " << *std::min_element(wall_samples.begin(), wall_samples.end()) << ",\n"
	     << "  \"wall_ms_p50\": " << sample_percentile(wall_samples, 0.50) << ",\n"
	     << "  \"wall_ms_p95\": " << sample_percentile(wall_samples, 0.95) << ",\n"
	     << "  \"wall_ms_max\": " << *std::max_element(wall_samples.begin(), wall_samples.end()) << ",\n"
	     << "  \"wall_ms_per_image_mean\": "
	     << sample_mean(wall_samples) / static_cast<double>(options.batch_size) << ",\n"
	     << "  \"compact_image_descriptor_count\": " << preview.compact_image_descriptor_count << ",\n"
	     << "  \"compact_plan_bytes\": " << preview.compact_plan_bytes << ",\n"
	     << "  \"compact_plan_peak_bytes\": " << preview.compact_plan_peak_bytes << ",\n"
	     << "  \"compiled_access_profile_hits\": " << preview.compiled_access_profile_hits << ",\n"
	     << "  \"compiled_access_profile_misses\": " << preview.compiled_access_profile_misses << ",\n"
	     << "  \"planless_axis_program_count\": " << preview.planless_axis_program_count << ",\n"
	     << "  \"planless_axis_phase_matrix_count\": " << preview.planless_axis_phase_matrix_count << ",\n"
	     << "  \"planless_axis_program_bytes\": " << preview.planless_axis_program_bytes << ",\n"
	     << "  \"resize_weight_build_ms\": " << preview.resize_weight_build_ms << ",\n"
	     << "  \"dct_resize_weight_cache_hits\": " << preview.dct_resize_weight_cache_hits << ",\n"
	     << "  \"dct_resize_weight_cache_misses\": " << preview.dct_resize_weight_cache_misses << ",\n"
	     << "  \"dct_conversion_matrix_cache_hits\": " << preview.dct_conversion_matrix_cache_hits << ",\n"
	     << "  \"dct_conversion_matrix_cache_misses\": " << preview.dct_conversion_matrix_cache_misses << ",\n"
	     << "  \"planned_rowgroup_count\": " << planned_rowgroup_count << ",\n"
	     << "  \"planned_selected_vector_count\": " << preview.planned_selected_vector_count << ",\n"
	     << "  \"full_vector_count\": " << preview.full_vector_count << ",\n"
	     << "  \"planned_saved_vector_count\": " << preview.planned_saved_vector_count << ",\n"
	     << "  \"planned_selected_vector_ratio\": " << preview.planned_selected_vector_ratio << ",\n"
	     << "  \"fixed_transform_component_count\": " << preview.fixed_transform_component_count << ",\n"
	     << "  \"fixed_transform_source_block_count\": " << preview.fixed_transform_source_block_count << ",\n"
	     << "  \"fixed_transform_output_block_count\": " << preview.fixed_transform_output_block_count << ",\n"
	     << "  \"host_expanded_transform_items_created\": "
	     << preview.host_expanded_transform_items_created << ",\n"
	     << "  \"host_output_block_source_lists_created\": "
	     << preview.host_output_block_source_lists_created << ",\n"
	     << "  \"host_global_transform_sort_items\": " << preview.host_global_transform_sort_items << "\n"
	     << '}';
	return json.str();
}

struct CompactReaderProbeCounters {
	size_t rowgroups                = 0U;
	uint64_t storage_bytes          = 0U;
	uint64_t physical_page_bytes    = 0U;
	uint64_t pread_count            = 0U;
	uint64_t preadv_count           = 0U;
	uint64_t coalesced_read_runs    = 0U;
	double   reported_pread_ms      = 0.0;
	double   reader_open_ms         = 0.0;
	double   read_call_wall_ms      = 0.0;
};

CompactReaderProbeCounters read_compact_preview(
    const std::filesystem::path&                       manifest_path,
    const galp::jpeg::JpegDctShardManifest&            manifest,
    const galp::jpeg::JpegDctDeviceBatchPlanPreview&   preview) {
	std::map<uint32_t, std::vector<size_t>> by_shard;
	for (const auto& rowgroup : preview.rowgroups) {
		by_shard[rowgroup.shard_id].push_back(rowgroup.rowgroup_index);
	}
	CompactReaderProbeCounters counters;
	for (auto& [shard_id, rowgroups] : by_shard) {
		std::sort(rowgroups.begin(), rowgroups.end());
		rowgroups.erase(std::unique(rowgroups.begin(), rowgroups.end()), rowgroups.end());
		const auto shard = std::find_if(manifest.shards.begin(), manifest.shards.end(), [&](const auto& candidate) {
			return candidate.shard_id == shard_id;
		});
		if (shard == manifest.shards.end()) {
			throw std::runtime_error("compact reader probe could not resolve a planned shard");
		}
		const auto fls_path = manifest_path.parent_path() / shard->fls_file_name;
		const auto open_begin = Clock::now();
		galp::format::FlsReader fls_reader(fls_path, /*load_column_names=*/false,
		                                   /*enable_sparse_vector_reads=*/false);
		counters.reader_open_ms += elapsed_ms(open_begin, Clock::now());
		if (!fls_reader.is_compact_v3()) {
			throw std::runtime_error("compact reader probe resolved a non-compact FLS shard");
		}
		std::vector<galp::format::ZeroCopyReadTiming> timings;
		const auto read_begin = Clock::now();
		auto views = fls_reader.read_compact_rowgroups_zero_copy_scatter(
		    rowgroups, &timings, /*view_workers=*/4U);
		counters.read_call_wall_ms += elapsed_ms(read_begin, Clock::now());
		if (views.size() != rowgroups.size() || timings.size() != rowgroups.size()) {
			throw std::runtime_error("compact reader probe returned an inconsistent result count");
		}
		counters.rowgroups += rowgroups.size();
		for (const auto& timing : timings) {
			counters.storage_bytes += timing.storage_bytes;
			counters.physical_page_bytes += timing.physical_page_bytes;
			counters.pread_count += timing.pread_count;
			counters.preadv_count += timing.preadv_count;
			counters.coalesced_read_runs += timing.coalesced_read_run_count;
			counters.reported_pread_ms += timing.pread_ms;
		}
	}
	return counters;
}

std::string run_reader_probe(const Options& options) {
	const auto manifest = galp::jpeg::detail::read_jpeg_dct_shard_manifest_file(options.manifest);
	if (!manifest.uses_compact_descriptor() || !manifest.uses_independent_vector_rowgroups()) {
		throw std::invalid_argument("--reader-probe requires a formal compact-v3 manifest");
	}
	galp::jpeg::JpegDctShardDatasetReader reader(options.manifest);
	const auto full_requests = make_requests(reader, options, false);
	auto       selected_requests = make_requests(reader, options, true);
	for (size_t index = 0U; index < selected_requests.size(); ++index) {
		selected_requests[index].horizontal_flip = (index & 1U) != 0U;
	}
	const auto full_preview = reader.PlanDeviceDctBatch(full_requests, make_device_options(64U));
	const auto selected_preview = reader.PlanDeviceDctBatch(selected_requests, make_transformed_planner_options());
	if (!selected_preview.uses_planless_fixed_transform || !selected_preview.block_metadata.empty() ||
	    selected_preview.host_expanded_transform_items_created != 0U ||
	    selected_preview.host_global_transform_sort_items != 0U) {
		throw std::runtime_error("compact reader probe did not receive a compact planless transformed-grid plan");
	}
	if (full_preview.rowgroups.size() != selected_preview.full_vector_count) {
		throw std::runtime_error("compact reader probe full-rowgroup baseline disagrees with the transformed plan");
	}
	const auto selected = read_compact_preview(options.manifest, manifest, selected_preview);
	const auto full     = read_compact_preview(options.manifest, manifest, full_preview);
	if (selected.storage_bytes > full.storage_bytes || selected.rowgroups > full.rowgroups) {
		throw std::runtime_error("compact reader probe selected footprint exceeds its full baseline");
	}
	const auto bytes_avoided = full.storage_bytes - selected.storage_bytes;
	const auto rowgroups_avoided = full.rowgroups - selected.rowgroups;
	const double selected_vector_ratio = full.rowgroups == 0U
	                                         ? 0.0
	                                         : static_cast<double>(selected.rowgroups) /
	                                               static_cast<double>(full.rowgroups);
	const double selected_byte_ratio = full.storage_bytes == 0U
	                                       ? 0.0
	                                       : static_cast<double>(selected.storage_bytes) /
	                                             static_cast<double>(full.storage_bytes);
	const double source_output_amplification = selected_preview.fixed_transform_output_block_count == 0U
	                                               ? 0.0
	                                               : static_cast<double>(
	                                                     selected_preview.fixed_transform_source_block_count) /
	                                                     static_cast<double>(
	                                                         selected_preview.fixed_transform_output_block_count);

	std::ostringstream json;
	json << std::boolalpha << "{\n"
	     << "  \"schema\": \"galp-compact-v3-runtime-audit-v1\",\n"
	     << "  \"mode\": \"reader\",\n"
	     << "  \"manifest\": \"" << json_escape(std::filesystem::absolute(options.manifest).string()) << "\",\n"
	     << "  \"manifest_version\": " << manifest.version << ",\n"
	     << "  \"batch_size\": " << options.batch_size << ",\n"
	     << "  \"crop\": \"center-half\",\n"
	     << "  \"horizontal_flip\": \"alternating\",\n"
	     << "  \"coefficient_selection\": \"all-64\",\n"
	     << "  \"view_workers\": 4,\n"
	     << "  \"page_cache_state\": \"not-controlled; byte and syscall counters are authoritative\",\n"
	     << "  \"selected_rowgroups\": " << selected.rowgroups << ",\n"
	     << "  \"full_rowgroups\": " << full.rowgroups << ",\n"
	     << "  \"actual_saved_rowgroups\": " << rowgroups_avoided << ",\n"
	     << "  \"selected_vector_ratio\": " << selected_vector_ratio << ",\n"
	     << "  \"selected_compressed_bytes\": " << selected.storage_bytes << ",\n"
	     << "  \"full_compressed_bytes\": " << full.storage_bytes << ",\n"
	     << "  \"compressed_bytes_avoided\": " << bytes_avoided << ",\n"
	     << "  \"selected_compressed_byte_ratio\": " << selected_byte_ratio << ",\n"
	     << "  \"selected_physical_page_bytes\": " << selected.physical_page_bytes << ",\n"
	     << "  \"full_physical_page_bytes\": " << full.physical_page_bytes << ",\n"
	     << "  \"selected_pread_count\": " << selected.pread_count << ",\n"
	     << "  \"selected_preadv_count\": " << selected.preadv_count << ",\n"
	     << "  \"selected_coalesced_read_runs\": " << selected.coalesced_read_runs << ",\n"
	     << "  \"full_pread_count\": " << full.pread_count << ",\n"
	     << "  \"full_preadv_count\": " << full.preadv_count << ",\n"
	     << "  \"full_coalesced_read_runs\": " << full.coalesced_read_runs << ",\n"
	     << "  \"selected_reported_pread_ms\": " << selected.reported_pread_ms << ",\n"
	     << "  \"selected_reader_open_ms\": " << selected.reader_open_ms << ",\n"
	     << "  \"selected_read_call_wall_ms\": " << selected.read_call_wall_ms << ",\n"
	     << "  \"full_reported_pread_ms\": " << full.reported_pread_ms << ",\n"
	     << "  \"full_reader_open_ms\": " << full.reader_open_ms << ",\n"
	     << "  \"full_read_call_wall_ms\": " << full.read_call_wall_ms << ",\n"
	     << "  \"planned_decode_vectors\": " << selected_preview.planned_selected_vector_count << ",\n"
	     << "  \"rowgroup_full_read_ratio\": " << (selected.rowgroups == 0U ? 0.0 : 1.0) << ",\n"
	     << "  \"fixed_transform_source_blocks\": "
	     << selected_preview.fixed_transform_source_block_count << ",\n"
	     << "  \"fixed_transform_output_blocks\": "
	     << selected_preview.fixed_transform_output_block_count << ",\n"
	     << "  \"source_output_block_amplification\": " << source_output_amplification << "\n"
	     << '}';
	return json.str();
}

std::string run_io_workload(const Options& options) {
	const bool                            cropped = options.workload == "crop-all" || options.workload == "crop-prefix";
	const auto                            memory_before = read_process_memory();
	const auto                            io_before     = read_process_io();
	const auto                            begin         = Clock::now();
	galp::jpeg::JpegDctShardDatasetReader reader(options.manifest);
	const auto                            full_requests = make_requests(reader, options, false);
	const auto selected_requests = cropped ? make_requests(reader, options, true) : full_requests;
	const auto device_options    = make_device_options(options.coefficient_count);
	const auto full_estimate     = reader.EstimateDeviceDctBatch(full_requests, device_options);
	const auto selected_estimate = reader.EstimateDeviceDctBatch(selected_requests, device_options);
	auto       batch             = reader.ReadDeviceDctBatch(selected_requests, device_options);
	batch.synchronize();
	const auto   finished     = Clock::now();
	const auto   stats        = batch.execution_stats();
	const auto   memory_after = read_process_memory();
	const auto   io_after     = read_process_io();
	const auto   mapped_fls   = read_mapped_fls_memory();
	const double vector_ratio = full_estimate.rowgroups.empty()
	                                ? 0.0
	                                : static_cast<double>(selected_estimate.rowgroups.size()) /
	                                      static_cast<double>(full_estimate.rowgroups.size());

	std::ostringstream json;
	json << std::boolalpha << "{\n"
	     << "  \"schema\": \"galp-compact-v3-runtime-audit-v1\",\n"
	     << "  \"mode\": \"io\",\n"
	     << "  \"manifest\": \"" << json_escape(std::filesystem::absolute(options.manifest).string()) << "\",\n"
	     << "  \"workload\": \"" << json_escape(options.workload) << "\",\n"
	     << "  \"coefficient_count\": " << options.coefficient_count << ",\n"
	     << "  \"batch_size\": " << options.batch_size << ",\n"
	     << "  \"first_batch_latency_ms\": " << elapsed_ms(begin, finished) << ",\n"
	     << "  \"logical_compressed_bytes\": " << stats.coefficient_logical_bytes_requested << ",\n"
	     << "  \"physical_range_bytes_read\": " << stats.compressed_payload_bytes_read << ",\n"
	     << "  \"full_compressed_payload_bytes\": " << stats.full_compressed_payload_bytes << ",\n"
	     << "  \"physical_page_bytes_covered\": " << stats.physical_page_bytes_covered << ",\n"
	     << "  \"full_physical_page_bytes\": " << stats.full_physical_page_bytes << ",\n"
	     << "  \"physical_page_coverage_ratio\": " << stats.physical_page_coverage_ratio << ",\n"
	     << "  \"actual_block_device_read_bytes\": " << saturating_delta(io_after.read_bytes, io_before.read_bytes)
	     << ",\n"
	     << "  \"process_logical_read_characters\": " << saturating_delta(io_after.rchar, io_before.rchar) << ",\n"
	     << "  \"process_read_syscalls\": " << saturating_delta(io_after.syscr, io_before.syscr) << ",\n"
	     << "  \"native_pread_count\": " << stats.pread_count << ",\n"
	     << "  \"native_preadv_count\": " << stats.preadv_count << ",\n"
	     << "  \"coalesced_run_count\": " << stats.coalesced_read_run_count << ",\n"
	     << "  \"selected_vector_count\": " << selected_estimate.rowgroups.size() << ",\n"
	     << "  \"full_vector_count\": " << full_estimate.rowgroups.size() << ",\n"
	     << "  \"selected_vector_ratio\": " << vector_ratio << ",\n"
	     << "  \"selected_coefficient_count\": " << stats.selected_coefficient_count << ",\n"
	     << "  \"full_coefficient_count\": " << stats.full_coefficient_count << ",\n"
	     << "  \"selected_coefficient_ratio\": " << stats.selected_coefficient_ratio << ",\n"
	     << "  \"storage_read_granularity\": \"" << json_escape(stats.storage_read_granularity) << "\",\n"
	     << "  \"decode_granularity\": \"" << json_escape(stats.decode_granularity) << "\",\n"
	     << "  \"rss_before_bytes\": " << memory_before.rss_bytes << ",\n"
	     << "  \"rss_after_bytes\": " << memory_after.rss_bytes << ",\n"
	     << "  \"pss_before_bytes\": " << memory_before.pss_bytes << ",\n"
	     << "  \"pss_after_bytes\": " << memory_after.pss_bytes << ",\n"
	     << "  \"actual_fls_mmap_resident_rss_bytes\": " << mapped_fls.rss_bytes << ",\n"
	     << "  \"actual_fls_mmap_resident_pss_bytes\": " << mapped_fls.pss_bytes << ",\n"
	     << "  \"galp_native_pinned_peak_in_use_bytes\": " << stats.galp_native_pinned_peak_in_use_bytes << ",\n"
	     << "  \"galp_native_device_peak_in_use_bytes\": " << stats.galp_native_device_peak_in_use_bytes << "\n"
	     << '}';
	return json.str();
}

} // namespace

int main(const int argc, char** argv) {
	try {
		Options options;
		if (!parse_args(argc, argv, options)) {
			print_usage(argv[0]);
			return 0;
		}
		const auto payload = options.memory_probe
		                         ? run_memory_probe(options)
		                         : (options.planner_probe
		                                ? run_planner_probe(options)
		                                : (options.reader_probe ? run_reader_probe(options) : run_io_workload(options)));
		emit_result(options, payload);
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "galp_compact_v3_runtime_audit: " << error.what() << '\n';
		return 2;
	}
}
