#include "galp/jpeg_dct_block_major_access.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void usage(const char* program) {
	std::cerr << "Usage: " << program
	          << " MANIFEST --output-dir DIRECTORY [--output-json PATH] [--exhaustive-rank-validation] [--skip-validation]"
	             " [--rank-checkpoint-images N] [--topology-checkpoint-coordinates N]\n";
}

uint16_t parse_u16(const std::string_view name, const char* value) {
	const auto parsed = std::stoull(value);
	if (parsed == 0U || parsed > std::numeric_limits<uint16_t>::max()) {
		throw std::runtime_error(std::string(name) + " must be in [1, 65535]");
	}
	return static_cast<uint16_t>(parsed);
}

void json_string(std::ostream& output, const std::string& value) {
	output << '"';
	for (const auto ch : value) {
		switch (ch) {
		case '\\': output << "\\\\"; break;
		case '"': output << "\\\""; break;
		case '\n': output << "\\n"; break;
		case '\r': output << "\\r"; break;
		case '\t': output << "\\t"; break;
		default: output << ch; break;
		}
	}
	output << '"';
}

} // namespace

int main(const int argc, char** argv) try {
	if (argc < 4) {
		usage(argv[0]);
		return 2;
	}
	std::filesystem::path manifest_path;
	std::filesystem::path output_directory;
	std::filesystem::path output_json;
	galp::jpeg::JpegDctBlockMajorAccessBuildOptions options;
	for (int index = 1; index < argc; ++index) {
		const std::string_view argument(argv[index]);
		if (argument == "--help" || argument == "-h") {
			usage(argv[0]);
			return 0;
		}
		if (argument == "--output-dir") {
			if (++index >= argc) {
				throw std::runtime_error("--output-dir requires a path");
			}
			output_directory = argv[index];
			continue;
		}
		if (argument == "--output-json") {
			if (++index >= argc) {
				throw std::runtime_error("--output-json requires a path");
			}
			output_json = argv[index];
			continue;
		}
		if (argument == "--exhaustive-rank-validation") {
			options.exhaustive_rank_validation = true;
			continue;
		}
		if (argument == "--skip-validation") {
			options.validate_after_write = false;
			continue;
		}
		if (argument == "--rank-checkpoint-images") {
			if (++index >= argc) {
				throw std::runtime_error("--rank-checkpoint-images requires a value");
			}
			options.rank_checkpoint_images = parse_u16("--rank-checkpoint-images", argv[index]);
			continue;
		}
		if (argument == "--topology-checkpoint-coordinates") {
			if (++index >= argc) {
				throw std::runtime_error("--topology-checkpoint-coordinates requires a value");
			}
			options.topology_checkpoint_coordinates =
			    parse_u16("--topology-checkpoint-coordinates", argv[index]);
			continue;
		}
		if (!argument.empty() && argument.front() == '-') {
			throw std::runtime_error("unknown option: " + std::string(argument));
		}
		if (!manifest_path.empty()) {
			throw std::runtime_error("multiple manifest paths were supplied");
		}
		manifest_path = argv[index];
	}
	if (manifest_path.empty() || output_directory.empty()) {
		usage(argv[0]);
		return 2;
	}
	const auto report = galp::jpeg::build_jpeg_dct_block_major_access_dataset(
	    manifest_path, output_directory, options);
	std::ostringstream json;
	json << "{\"schema_version\":\"galp_block_major_access_build_v1\",\"manifest\":";
	json_string(json, std::filesystem::absolute(manifest_path).string());
	json << ",\"index_path\":";
	json_string(json, std::filesystem::absolute(report.index_path).string());
	json << ",\"source_dataset_bytes\":" << report.source_dataset_bytes
	     << ",\"descriptor_bytes\":" << report.descriptor_bytes << ",\"index_bytes\":" << report.index_bytes
	     << ",\"storage_growth_ratio\":" << std::setprecision(12) << report.storage_growth_ratio
	     << ",\"passes_one_percent\":" << (report.passes_one_percent ? "true" : "false")
	     << ",\"passes_half_percent\":" << (report.passes_half_percent ? "true" : "false")
	     << ",\"shards\":[";
	for (size_t index = 0U; index < report.shards.size(); ++index) {
		if (index != 0U) {
			json << ',';
		}
		const auto& shard = report.shards[index];
		json << "{\"shard_id\":" << shard.shard_id << ",\"image_count\":" << shard.image_count
		     << ",\"group_count\":" << shard.group_count << ",\"rank_cell_count\":"
		     << shard.rank_cell_count << ",\"descriptor_bytes\":" << shard.descriptor_bytes
		     << ",\"presence_payload_bytes\":" << shard.presence_payload_bytes
		     << ",\"all_present_cells\":" << shard.all_present_cells << ",\"sparse_cells\":"
		     << shard.sparse_cells << ",\"missing_cells\":" << shard.missing_cells
		     << ",\"bitmap_cells\":" << shard.bitmap_cells << ",\"empty_cells\":" << shard.empty_cells
		     << ",\"groups_checked\":" << shard.validation.groups_checked
		     << ",\"rank_cells_checked\":" << shard.validation.rank_cells_checked
		     << ",\"rank_queries_checked\":" << shard.validation.rank_queries_checked
		     << ",\"select_queries_checked\":" << shard.validation.select_queries_checked
		     << ",\"descriptor_path\":";
		json_string(json, std::filesystem::absolute(shard.descriptor_path).string());
		json << '}';
	}
	json << "]}\n";
	std::cout << json.str();
	if (!output_json.empty()) {
		std::ofstream file(output_json, std::ios::trunc);
		if (!file) {
			throw std::runtime_error("cannot open --output-json path: " + output_json.string());
		}
		file << json.str();
		if (!file) {
			throw std::runtime_error("failed to write --output-json path: " + output_json.string());
		}
	}
	return 0;
} catch (const std::exception& error) {
	std::cerr << "block-major access build failed: " << error.what() << '\n';
	return 1;
}
