#include "fastlens/jpeg_dct_block_major_access.hpp"
#include "jpeg/jpeg_dct_metadata.hpp"
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
	std::cerr
	    << "Usage: " << program
	    << " MANIFEST --output-dir DIRECTORY [--output-json PATH] [--exhaustive-rank-validation] [--skip-validation]"
	       " [--rank-checkpoint-images N] [--topology-checkpoint-coordinates N]\n"
	    << "       " << program << " --inspect MANIFEST\n"
	    << "       " << program << " --inspect-manifest MANIFEST\n"
	    << "       " << program << " --file-crc64 FILE  (complete file, CRC64-ECMA)\n"
	    << "       " << program << " --finalize-store MANIFEST  (unpublished Converter staging only)\n";
}

std::string partition_name(const uint32_t id, const char* suffix) {
	std::ostringstream name;
	name << std::setw(6) << std::setfill('0') << id << suffix;
	return name.str();
}

// This is the published Store contract, not a restriction on the native
// manifest parser used by writers, finalization and experimental layouts.
void validate_store_manifest(const fastlens::jpeg::JpegDctShardManifest& manifest) {
	if (manifest.version != 1U || manifest.shards.empty()) {
		throw std::runtime_error("standard Store requires a nonempty manifest-v1 dataset; prepare it with fastlens convert");
	}
	uint64_t next_image = 0U;
	for (size_t i = 0; i < manifest.shards.size(); ++i) {
		const auto& shard = manifest.shards[i];
		if (shard.shard_id != i ||
		    shard.fls_file_name != "partitions/" + partition_name(shard.shard_id, ".fls") ||
		    shard.metadata_file_name != "partitions/" + partition_name(shard.shard_id, ".images.bin")) {
			throw std::runtime_error("nonstandard Store partition paths or IDs; prepare it with fastlens convert");
		}
		if (shard.image_count == 0U || shard.first_global_image_index != next_image ||
		    next_image > manifest.image_count || shard.image_count > manifest.image_count - next_image) {
			throw std::runtime_error("invalid Store partition image ID range");
		}
		next_image += shard.image_count;
	}
	if (next_image != manifest.image_count) {
		throw std::runtime_error("Store partition image counts disagree with manifest");
	}
}

// The Converter owns this unpublished directory and removes it on failure.
// Keep native generation transactions intact; only finalize their completed
// output here, before any geometry index or annotation binds the manifest.
void finalize_store(const std::filesystem::path& path) {
	using namespace fastlens::jpeg;
	auto manifest = read_jpeg_dct_shard_manifest(path);
	if (manifest.version != 1U || manifest.shards.empty()) {
		throw std::runtime_error("store finalization requires a nonempty manifest-v1 dataset");
	}
	const auto root       = path.parent_path();
	const auto generation = std::filesystem::path(manifest.shards.front().fls_file_name).parent_path();
	if (generation.has_parent_path() || generation.string().rfind("generation_", 0) != 0) {
		throw std::runtime_error("store finalization requires unpublished native generation output");
	}
	for (const auto& shard : manifest.shards) {
		if (std::filesystem::path(shard.fls_file_name).parent_path() != generation ||
		    std::filesystem::path(shard.metadata_file_name).parent_path() != generation) {
			throw std::runtime_error("store finalization requires one complete native generation");
		}
	}
	if (!std::filesystem::create_directory(root / "partitions")) {
		throw std::runtime_error("refusing to replace existing partitions directory");
	}
	for (auto& shard : manifest.shards) {
		const auto fls      = std::filesystem::path("partitions") / partition_name(shard.shard_id, ".fls");
		const auto metadata = std::filesystem::path("partitions") / partition_name(shard.shard_id, ".images.bin");
		std::filesystem::rename(root / shard.fls_file_name, root / fls);
		std::filesystem::rename(root / shard.metadata_file_name, root / metadata);
		shard.fls_file_name      = fls.generic_string();
		shard.metadata_file_name = metadata.generic_string();
	}
	const auto staged_manifest = root / "manifest.bin.tmp";
	write_jpeg_dct_shard_manifest(manifest, staged_manifest);
	std::filesystem::rename(staged_manifest, path);
	std::filesystem::remove(root / generation);
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

// Use the same parsers and source-identity checks as the online reader. No
// coefficient decoding, CUDA initialization, or source JPEG access is needed.
void inspect_dataset(const std::filesystem::path& path) {
	using namespace fastlens::jpeg;
	const auto manifest   = read_jpeg_dct_shard_manifest(path);
	validate_store_manifest(manifest);
	const auto root       = path.parent_path();
	const auto directory  = root / "geometry";
	const auto index_path = directory / "index.bin";
	if (!std::filesystem::is_regular_file(index_path)) {
		throw std::runtime_error("incomplete store: missing geometry/index.bin");
	}
	const auto            index     = read_jpeg_dct_block_major_access_index(index_path, path, manifest);
	uint64_t              fls_bytes = 0U, metadata_bytes = 0U, index_bytes = index.index_bytes;
	std::vector<uint64_t> warning_images;
	for (size_t i = 0; i < manifest.shards.size(); ++i) {
		const auto& shard         = manifest.shards[i];
		const auto  fls_path      = root / shard.fls_file_name;
		const auto  metadata_path = root / shard.metadata_file_name;
		const auto  fls_size      = std::filesystem::file_size(fls_path);
		const auto  metadata_size = std::filesystem::file_size(metadata_path);
		if (fls_size != shard.fls_file_size || metadata_size != shard.metadata_file_size) {
			throw std::runtime_error("shard file size disagrees with manifest: " + shard.fls_file_name);
		}
		fls_bytes += fls_size;
		metadata_bytes += metadata_size;
		if (manifest.version == 1U) {
			const auto metadata = detail::read_jpeg_dct_metadata_file(metadata_path);
			if (metadata.row_ordering != JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor ||
			    metadata.images.size() != shard.image_count) {
				throw std::runtime_error("manifest-v1 requires spatial-major metadata: " + metadata_path.string());
			}
			for (size_t local = 0; local < metadata.images.size(); ++local) {
				if (metadata.images[local].warning_count != 0U) {
					warning_images.push_back(shard.first_global_image_index + local);
				}
			}
		}
		const auto filename   = partition_name(shard.shard_id, ".bin");
		auto       descriptor = JpegDctBlockMajorAccessDescriptor::Open(directory / filename);
		if (descriptor.descriptor_bytes() != index.shards[i].descriptor_bytes ||
		    descriptor.descriptor_crc64() != index.shards[i].descriptor_crc64) {
			throw std::runtime_error("geometry descriptor disagrees with index: " + filename);
		}
		descriptor.ValidateSource(path, shard, metadata_path, fls_path);
		index_bytes += descriptor.descriptor_bytes();
	}
	std::cout << "{\"schema_version\":1,\"manifest_version\":" << manifest.version << ",\"physical_layout\":";
	json_string(std::cout,
	            manifest.version == 1U   ? "spatial-major-image-minor"
	            : manifest.version == 2U ? "image-major"
	                                     : manifest.physical_layout);
	std::cout << ",\"image_count\":" << manifest.image_count << ",\"shard_count\":" << manifest.shards.size()
	          << ",\"rowgroup_vectors\":" << manifest.rowgroup_vectors
	          << ",\"rowgroups_per_shard\":" << manifest.rowgroups_per_shard
	          << ",\"coefficient_file_count\":" << manifest.shards.size()
	          << ",\"metadata_file_count\":" << manifest.shards.size() << ",\"coefficient_bytes\":" << fls_bytes
	          << ",\"metadata_bytes\":" << metadata_bytes << ",\"manifest_bytes\":" << std::filesystem::file_size(path)
	          << ",\"index_bytes\":" << index_bytes << ",\"index_file_count\":" << (manifest.shards.size() + 1U)
	          << ",\"index\":{\"present\":true,\"compatible\":true,\"version\":1}" << ",\"jpeg_warning_image_ids\":[";
	for (size_t i = 0; i < warning_images.size(); ++i) {
		if (i != 0U) {
			std::cout << ',';
		}
		std::cout << warning_images[i];
	}
	std::cout << "]}\n";
}

} // namespace

int main(const int argc, char** argv) try {
	if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
		usage(argv[0]);
		return 0;
	}
	if (argc == 3 && std::string_view(argv[1]) == "--file-crc64") {
		std::cout << std::hex << std::setfill('0') << std::setw(16)
		          << fastlens::jpeg::detail::jpeg_dct_file_crc64(argv[2]) << '\n';
		return 0;
	}
	if (argc == 3 && std::string_view(argv[1]) == "--finalize-store") {
		finalize_store(argv[2]);
		return 0;
	}
	if (argc == 3 && std::string_view(argv[1]) == "--inspect-manifest") {
		const auto manifest = fastlens::jpeg::read_jpeg_dct_shard_manifest(argv[2]);
		validate_store_manifest(manifest);
		std::cout << "{\"image_count\":" << manifest.image_count << ",\"manifest_version\":" << manifest.version
		          << ",\"shard_count\":" << manifest.shards.size() << "}\n";
		return 0;
	}
	if (argc == 3 && std::string_view(argv[1]) == "--inspect") {
		inspect_dataset(argv[2]);
		return 0;
	}
	if (argc < 4) {
		usage(argv[0]);
		return 2;
	}
	std::filesystem::path manifest_path;
	std::filesystem::path output_directory;
	std::filesystem::path output_json;
	fastlens::jpeg::JpegDctBlockMajorAccessBuildOptions options;
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
	const auto report = fastlens::jpeg::build_jpeg_dct_block_major_access_dataset(
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
	std::cerr << "block-major access tool failed: " << error.what() << '\n';
	return 1;
}
