#include "galp/jpeg_dct.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
	std::filesystem::path                 output_fls;
	std::filesystem::path                 output_metadata;
	std::vector<std::filesystem::path>    inputs;
	galp::jpeg::JpegDatasetValidationMode policy           = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegMetadataProfile       metadata_profile = galp::jpeg::JpegMetadataProfile::kDctDatasetOnly;
	bool                                  metadata_profile_specified = false;
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
	    << "Default: --policy ragged and the legacy metadata format.\n"
	    << "  --policy pad is a dense-layout mode for callers that require fixed num_images rows per block group.\n"
	    << "  --metadata-profile writes the sectioned metadata format; use reconstruct to persist image dimensions "
	       "and quantization tables.\n";
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

bool parse_args(const int argc, char** argv, Options& options) {
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if ((arg == "--out" || arg == "-o") && i + 1 < argc) {
			options.output_fls = argv[++i];
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
		if (arg == "--help" || arg == "-h") {
			return false;
		}
		options.inputs.emplace_back(argv[i]);
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
