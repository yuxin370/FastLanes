#include "galp/jpeg_dct.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
	std::filesystem::path                 output_fls;
	std::filesystem::path                 output_metadata;
	std::filesystem::path                 output_dir;
	std::vector<std::filesystem::path>    inputs;
	galp::jpeg::JpegDatasetValidationMode policy           = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegMetadataProfile       metadata_profile = galp::jpeg::JpegMetadataProfile::kDctDatasetOnly;
	galp::jpeg::JpegDctShardPreset        shard_preset     = galp::jpeg::JpegDctShardPreset::kBalanced;
	size_t                                shard_images     = 8192;
	uint32_t                              rowgroup_vectors = 128;
	uint32_t                              rowgroups_per_shard           = 256;
	bool                                  shard_mode                    = false;
	bool                                  metadata_profile_specified    = false;
	bool                                  shard_images_specified        = false;
	bool                                  rowgroup_vectors_specified    = false;
	bool                                  rowgroups_per_shard_specified = false;
};

void print_usage(const char* prog) {
	std::cerr
	    << "Usage:\n"
	    << "  " << prog
	    << " --out output.fls --metadata output.metadata.bin [--policy ragged|strict|pad] "
	       "[--metadata-profile dct|reconstruct|preserve] input.jpg\n"
	    << "  " << prog
	    << " --out output.fls --metadata output.metadata.bin [--policy ragged|strict|pad] "
	       "[--metadata-profile dct|reconstruct|preserve] input_dir\n"
	    << "  " << prog
	    << " --out output.fls --metadata output.metadata.bin [--policy ragged|strict|pad] "
	       "[--metadata-profile dct|reconstruct|preserve] input0.jpg [input1.jpg ...]\n"
	    << "  " << prog
	    << " --shard --out-dir output_dct [--policy ragged|strict|pad] [--preset crop-latency|balanced|throughput] "
	       "[--shard-images N] [--rowgroup-vectors N] [--rowgroups-per-shard N] input_dir\n"
	    << "Default: --policy ragged and the legacy metadata format.\n"
	    << "  --policy pad is a dense-layout mode for callers that require fixed num_images rows per block group.\n"
	    << "  --metadata-profile writes the sectioned metadata format; use reconstruct to persist image dimensions "
	       "and quantization tables.\n"
	    << "  --shard writes manifest.bin plus shard_*.fls and shard_*.meta.bin; default preset is balanced.\n";
}

bool is_jpeg_path(const std::filesystem::path& path) {
	auto ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return ext == ".jpg" || ext == ".jpeg" || ext == ".jpe";
}

std::vector<std::filesystem::path> expand_inputs(const std::vector<std::filesystem::path>& inputs) {
	std::vector<std::filesystem::path> expanded;
	for (const auto& input : inputs) {
		if (std::filesystem::is_directory(input)) {
			for (const auto& entry : std::filesystem::recursive_directory_iterator(input)) {
				if (entry.is_regular_file() && is_jpeg_path(entry.path())) {
					expanded.push_back(entry.path());
				}
			}
			continue;
		}
		expanded.push_back(input);
	}

	std::sort(expanded.begin(), expanded.end());
	if (expanded.empty()) {
		throw std::runtime_error("no JPEG files found in input path(s)");
	}
	return expanded;
}

uint32_t parse_u32_arg(const std::string_view name, const char* value) {
	const auto parsed = std::stoull(value);
	if (parsed > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error(std::string(name) + " is outside uint32_t range");
	}
	return static_cast<uint32_t>(parsed);
}

size_t parse_size_arg(const std::string_view name, const char* value) {
	const auto parsed = std::stoull(value);
	if (parsed > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error(std::string(name) + " is outside size_t range");
	}
	return static_cast<size_t>(parsed);
}

void apply_shard_preset(Options& options) {
	size_t   preset_shard_images        = 8192;
	uint32_t preset_rowgroup_vectors    = 128;
	uint32_t preset_rowgroups_per_shard = 256;
	switch (options.shard_preset) {
	case galp::jpeg::JpegDctShardPreset::kCropLatency:
		preset_shard_images     = 4096;
		preset_rowgroup_vectors = 64;
		break;
	case galp::jpeg::JpegDctShardPreset::kBalanced:
		break;
	case galp::jpeg::JpegDctShardPreset::kThroughput:
		preset_rowgroup_vectors = 256;
		break;
	}
	if (!options.shard_images_specified) {
		options.shard_images = preset_shard_images;
	}
	if (!options.rowgroup_vectors_specified) {
		options.rowgroup_vectors = preset_rowgroup_vectors;
	}
	if (!options.rowgroups_per_shard_specified) {
		options.rowgroups_per_shard = preset_rowgroups_per_shard;
	}
}

bool parse_args(const int argc, char** argv, Options& options) {
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--shard") {
			options.shard_mode = true;
			continue;
		}
		if ((arg == "--out" || arg == "-o") && i + 1 < argc) {
			options.output_fls = argv[++i];
			continue;
		}
		if (arg == "--out-dir" && i + 1 < argc) {
			options.output_dir = argv[++i];
			continue;
		}
		if (arg == "--metadata" && i + 1 < argc) {
			options.output_metadata = argv[++i];
			continue;
		}
		if (arg == "--policy" && i + 1 < argc) {
			const std::string_view policy = argv[++i];
			if (policy == "strict") {
				options.policy = galp::jpeg::JpegDatasetValidationMode::kRequireSameComponentGrids;
			} else if (policy == "pad") {
				options.policy = galp::jpeg::JpegDatasetValidationMode::kPadToMaxComponentGrids;
			} else if (policy == "ragged") {
				options.policy = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
			} else {
				throw std::runtime_error("unknown --policy value; expected strict, pad, or ragged");
			}
			continue;
		}
		if (arg == "--metadata-profile" && i + 1 < argc) {
			const std::string_view profile     = argv[++i];
			options.metadata_profile_specified = true;
			if (profile == "dct") {
				options.metadata_profile = galp::jpeg::JpegMetadataProfile::kDctDatasetOnly;
			} else if (profile == "reconstruct") {
				options.metadata_profile = galp::jpeg::JpegMetadataProfile::kReconstructableJpeg;
			} else if (profile == "preserve") {
				options.metadata_profile = galp::jpeg::JpegMetadataProfile::kPreserveOriginalMarkers;
			} else {
				throw std::runtime_error("unknown --metadata-profile value; expected dct, reconstruct, or preserve");
			}
			continue;
		}
		if (arg == "--preset" && i + 1 < argc) {
			const std::string_view preset = argv[++i];
			if (preset == "crop-latency") {
				options.shard_preset = galp::jpeg::JpegDctShardPreset::kCropLatency;
			} else if (preset == "balanced") {
				options.shard_preset = galp::jpeg::JpegDctShardPreset::kBalanced;
			} else if (preset == "throughput") {
				options.shard_preset = galp::jpeg::JpegDctShardPreset::kThroughput;
			} else {
				throw std::runtime_error("unknown --preset value; expected crop-latency, balanced, or throughput");
			}
			continue;
		}
		if (arg == "--shard-images" && i + 1 < argc) {
			options.shard_images           = parse_size_arg(arg, argv[++i]);
			options.shard_images_specified = true;
			continue;
		}
		if (arg == "--rowgroup-vectors" && i + 1 < argc) {
			options.rowgroup_vectors           = parse_u32_arg(arg, argv[++i]);
			options.rowgroup_vectors_specified = true;
			continue;
		}
		if (arg == "--rowgroups-per-shard" && i + 1 < argc) {
			options.rowgroups_per_shard           = parse_u32_arg(arg, argv[++i]);
			options.rowgroups_per_shard_specified = true;
			continue;
		}
		if (arg == "--help" || arg == "-h") {
			return false;
		}
		options.inputs.emplace_back(argv[i]);
	}

	apply_shard_preset(options);
	if (options.shard_mode) {
		return !options.output_dir.empty() && !options.inputs.empty();
	}
	return !options.output_fls.empty() && !options.output_metadata.empty() && !options.inputs.empty();
}

} // namespace

int main(const int argc, char** argv) {
	try {
		Options options;
		if (!parse_args(argc, argv, options)) {
			print_usage(argv[0]);
			return 1;
		}

		options.inputs = expand_inputs(options.inputs);

		galp::jpeg::JpegDctReaderOptions reader_options;
		reader_options.validation_mode = options.policy;
		reader_options.capture_metadata_markers =
		    options.metadata_profile == galp::jpeg::JpegMetadataProfile::kPreserveOriginalMarkers;
		galp::jpeg::JpegDctMetadataWriterOptions writer_options;
		writer_options.profile = options.metadata_profile;
		if (options.shard_mode) {
			galp::jpeg::JpegDctShardOptions shard_options;
			shard_options.shard_images        = options.shard_images;
			shard_options.rowgroup_vectors    = options.rowgroup_vectors;
			shard_options.rowgroups_per_shard = options.rowgroups_per_shard;
			shard_options.preset              = options.shard_preset;
			shard_options.shard_images_specified        = options.shard_images_specified;
			shard_options.rowgroup_vectors_specified    = options.rowgroup_vectors_specified;
			shard_options.rowgroups_per_shard_specified = options.rowgroups_per_shard_specified;
			galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
			    options.inputs, options.output_dir, reader_options, shard_options, writer_options);
			return 0;
		}
		auto table = options.inputs.size() == 1 ? galp::jpeg::read_jpeg_dct_file(options.inputs.front(), reader_options)
		                                        : galp::jpeg::read_jpeg_dct_dataset(options.inputs, reader_options);
		if (options.metadata_profile_specified) {
			galp::jpeg::compress_jpeg_dct_to_fls(table, options.output_fls, options.output_metadata, writer_options);
		} else {
			galp::jpeg::compress_jpeg_dct_to_fls(table, options.output_fls, options.output_metadata);
		}
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "galp_jpeg_dct_tool: " << e.what() << '\n';
		return 2;
	}
}
