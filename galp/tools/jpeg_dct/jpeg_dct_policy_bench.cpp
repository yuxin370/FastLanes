#include "galp/jpeg_dct.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
	std::filesystem::path          input;
	std::filesystem::path          output_dir                    = "/tmp/galp_jpeg_dct_policy_bench";
	galp::jpeg::JpegDctShardPreset preset                        = galp::jpeg::JpegDctShardPreset::kBalanced;
	size_t                         shard_images                  = 8192;
	uint32_t                       rowgroup_vectors              = 128;
	uint32_t                       rowgroups_per_shard           = 256;
	bool                           csv                           = false;
	bool                           shard_images_specified        = false;
	bool                           rowgroup_vectors_specified    = false;
	bool                           rowgroups_per_shard_specified = false;
};

struct PolicyResult {
	std::string policy;
	uintmax_t   total_output_size          = 0;
	uintmax_t   compressed_data_size       = 0;
	uintmax_t   metadata_size              = 0;
	uintmax_t   source_jpeg_size           = 0;
	uint64_t    real_row_count             = 0;
	uint64_t    padding_row_count          = 0;
	uint64_t    physical_row_count         = 0;
	uint32_t    shard_count                = 0;
	uint32_t    rowgroup_count             = 0;
	uint32_t    block_group_count          = 0;
	double      semantic_compression_ratio = 0.0;
	double      physical_compression_ratio = 0.0;
	double      expansion_vs_jpeg          = 0.0;
	double      encode_ms                  = 0.0;
	double      decode_ms                  = 0.0;
	double      crop_lookup_latency_ns     = 0.0;
	uint64_t    peak_rss_bytes             = 0;
	bool        decode_supported           = true;
	std::string decode_error;
};

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " --input jpeg_dir [--out-dir output_dir] [--csv]\n"
	          << "      [--preset crop-latency|balanced|throughput]\n"
	          << "      [--shard-images N] [--rowgroup-vectors N] [--rowgroups-per-shard N]\n";
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
	switch (options.preset) {
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
		if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
			options.input = argv[++i];
			continue;
		}
		if (arg == "--out-dir" && i + 1 < argc) {
			options.output_dir = argv[++i];
			continue;
		}
		if (arg == "--csv") {
			options.csv = true;
			continue;
		}
		if (arg == "--preset" && i + 1 < argc) {
			const std::string_view preset = argv[++i];
			if (preset == "crop-latency") {
				options.preset = galp::jpeg::JpegDctShardPreset::kCropLatency;
			} else if (preset == "balanced") {
				options.preset = galp::jpeg::JpegDctShardPreset::kBalanced;
			} else if (preset == "throughput") {
				options.preset = galp::jpeg::JpegDctShardPreset::kThroughput;
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
		throw std::runtime_error("unknown argument: " + std::string(arg));
	}
	apply_shard_preset(options);
	return !options.input.empty();
}

bool is_jpeg_path(const std::filesystem::path& path) {
	auto ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return ext == ".jpg" || ext == ".jpeg" || ext == ".jpe";
}

std::vector<std::filesystem::path> collect_jpegs(const std::filesystem::path& input) {
	std::vector<std::filesystem::path> paths;
	if (std::filesystem::is_directory(input)) {
		for (const auto& entry : std::filesystem::recursive_directory_iterator(input)) {
			if (entry.is_regular_file() && is_jpeg_path(entry.path())) {
				paths.push_back(entry.path());
			}
		}
	} else if (is_jpeg_path(input)) {
		paths.push_back(input);
	}
	std::sort(paths.begin(), paths.end());
	if (paths.empty()) {
		throw std::runtime_error("no JPEG files found");
	}
	return paths;
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

uint64_t peak_rss_bytes() {
	rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage) != 0) {
		return 0;
	}
	return static_cast<uint64_t>(usage.ru_maxrss) * 1024U;
}

uintmax_t total_source_jpeg_size(const std::vector<std::filesystem::path>& paths) {
	uintmax_t total = 0;
	for (const auto& path : paths) {
		total += std::filesystem::file_size(path);
	}
	return total;
}

std::string format_bytes(const uintmax_t bytes) {
	static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
	double                       value   = static_cast<double>(bytes);
	size_t                       unit    = 0;
	while (value >= 1024.0 && unit + 1 < (sizeof(units) / sizeof(units[0]))) {
		value /= 1024.0;
		++unit;
	}
	std::ostringstream out;
	out << std::fixed << std::setprecision(value < 10.0 ? 2 : value < 100.0 ? 1 : 0) << value << ' ' << units[unit];
	return out.str();
}

std::string format_count(const uint64_t value) {
	std::ostringstream out;
	out.imbue(std::locale(""));
	out << value;
	return out.str();
}

std::string format_ms(const double value) {
	std::ostringstream out;
	if (value >= 1000.0) {
		out << std::fixed << std::setprecision(2) << (value / 1000.0) << " s";
	} else {
		out << std::fixed << std::setprecision(2) << value << " ms";
	}
	return out.str();
}

std::string format_ns(const double value) {
	std::ostringstream out;
	if (value >= 1000000.0) {
		out << std::fixed << std::setprecision(2) << (value / 1000000.0) << " ms";
	} else if (value >= 1000.0) {
		out << std::fixed << std::setprecision(2) << (value / 1000.0) << " us";
	} else {
		out << std::fixed << std::setprecision(2) << value << " ns";
	}
	return out.str();
}

std::string format_ratio(const double value) {
	std::ostringstream out;
	out << std::fixed << std::setprecision(3) << value << "x";
	return out.str();
}

galp::jpeg::JpegDctShardOptions make_shard_options(const Options& options) {
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.preset              = options.preset;
	shard_options.shard_images        = options.shard_images;
	shard_options.rowgroup_vectors    = options.rowgroup_vectors;
	shard_options.rowgroups_per_shard = options.rowgroups_per_shard;
	shard_options.shard_images_specified        = options.shard_images_specified;
	shard_options.rowgroup_vectors_specified    = options.rowgroup_vectors_specified;
	shard_options.rowgroups_per_shard_specified = options.rowgroups_per_shard_specified;
	return shard_options;
}

double measure_crop_lookup_latency_ns(galp::jpeg::JpegDctShardDatasetReader& reader) {
	const auto start = Clock::now();
	const auto ref   = reader.LocateRow(0, 0, 0, 0);
	if (ref.present) {
		auto group = reader.ReadBlockGroup(ref.shard_id, ref.semantic_slot_id, ref.block_x, ref.block_y);
		if (group.rows.empty()) {
			std::cerr << "";
		}
	}
	const auto end = Clock::now();
	return std::chrono::duration<double, std::nano>(end - start).count();
}

PolicyResult run_policy(const std::vector<std::filesystem::path>&   paths,
                        const std::filesystem::path&                output_dir,
                        const Options&                              cli_options,
                        const std::string&                          policy_name,
                        const galp::jpeg::JpegDatasetValidationMode policy) {
	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = policy;

	const auto policy_dir = output_dir / policy_name;
	std::filesystem::remove_all(policy_dir);

	const auto encode_start = Clock::now();
	const auto manifest     = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
        paths, policy_dir, reader_options, make_shard_options(cli_options));
	const auto encode_end = Clock::now();

	PolicyResult result;
	result.policy           = policy_name;
	result.source_jpeg_size = total_source_jpeg_size(paths);
	result.encode_ms        = elapsed_ms(encode_start, encode_end);
	result.shard_count      = static_cast<uint32_t>(manifest.shards.size());

	for (const auto& shard : manifest.shards) {
		result.compressed_data_size += shard.fls_file_size;
		result.metadata_size += shard.metadata_file_size;
		result.real_row_count += shard.real_row_count;
		result.padding_row_count += shard.padding_row_count;
		result.physical_row_count += shard.physical_row_count;
		result.rowgroup_count += shard.rowgroup_count;
		result.block_group_count += shard.block_group_count;
	}
	result.metadata_size += std::filesystem::file_size(policy_dir / "manifest.bin");
	result.total_output_size = result.compressed_data_size + result.metadata_size;

	const double physical_uncompressed = static_cast<double>(result.physical_row_count) * 64.0 * sizeof(int16_t);
	const double semantic_uncompressed = static_cast<double>(result.real_row_count) * 64.0 * sizeof(int16_t);
	if (result.compressed_data_size != 0) {
		result.physical_compression_ratio = physical_uncompressed / static_cast<double>(result.compressed_data_size);
		result.semantic_compression_ratio = semantic_uncompressed / static_cast<double>(result.compressed_data_size);
	}
	if (result.source_jpeg_size != 0) {
		result.expansion_vs_jpeg =
		    static_cast<double>(result.total_output_size) / static_cast<double>(result.source_jpeg_size);
	}

	try {
		galp::jpeg::JpegDctShardDatasetReader reader(policy_dir / "manifest.bin");
		const auto                            decode_start = Clock::now();
		auto                                  image        = reader.MaterializeImageDct(0);
		if (image.blocks.empty()) {
			std::cerr << "";
		}
		const auto decode_end         = Clock::now();
		result.decode_ms              = elapsed_ms(decode_start, decode_end);
		result.crop_lookup_latency_ns = measure_crop_lookup_latency_ns(reader);
	} catch (const std::exception& e) {
		result.decode_supported = false;
		result.decode_error     = e.what();
	}
	result.peak_rss_bytes = peak_rss_bytes();
	return result;
}

void print_csv_header() {
	std::cout << "policy,total_output_size,semantic_compression_ratio,encode_ms,decode_ms,"
	             "crop_lookup_latency_ns,expansion_vs_jpeg,peak_rss_bytes,"
	             "compressed_data_size,metadata_size,source_jpeg_size,real_row_count,padding_row_count,"
	             "physical_row_count,shard_count,rowgroup_count,block_group_count,physical_compression_ratio,"
	             "decode_supported,decode_error\n";
}

void print_csv_row(const PolicyResult& r) {
	std::cout << r.policy << ',' << r.total_output_size << ',' << r.semantic_compression_ratio << ',' << r.encode_ms
	          << ',' << r.decode_ms << ',' << r.crop_lookup_latency_ns << ',' << r.expansion_vs_jpeg << ','
	          << r.peak_rss_bytes << ',' << r.compressed_data_size << ',' << r.metadata_size << ','
	          << r.source_jpeg_size << ',' << r.real_row_count << ',' << r.padding_row_count << ','
	          << r.physical_row_count << ',' << r.shard_count << ',' << r.rowgroup_count << ',' << r.block_group_count
	          << ',' << r.physical_compression_ratio << ',' << (r.decode_supported ? "true" : "false") << ',' << '"'
	          << r.decode_error << '"' << '\n';
}

void print_human_metric(const char* name, const std::string& pad, const std::string& ragged) {
	std::cout << std::left << std::setw(30) << name << std::right << std::setw(18) << pad << std::setw(18) << ragged
	          << '\n';
}

void print_human_report(const std::vector<std::filesystem::path>& paths,
                        const std::filesystem::path&              output_dir,
                        const PolicyResult&                       pad,
                        const PolicyResult&                       ragged) {
	std::cout << "JPEG DCT sharded policy benchmark\n";
	std::cout << "images: " << format_count(paths.size()) << '\n';
	std::cout << "output_dir: " << output_dir << "\n\n";
	std::cout << std::left << std::setw(30) << "metric" << std::right << std::setw(18) << "pad" << std::setw(18)
	          << "ragged" << '\n';
	std::cout << std::string(66, '-') << '\n';
	print_human_metric("total output", format_bytes(pad.total_output_size), format_bytes(ragged.total_output_size));
	print_human_metric(
	    "compressed data", format_bytes(pad.compressed_data_size), format_bytes(ragged.compressed_data_size));
	print_human_metric("metadata", format_bytes(pad.metadata_size), format_bytes(ragged.metadata_size));
	print_human_metric("source JPEG", format_bytes(pad.source_jpeg_size), format_bytes(ragged.source_jpeg_size));
	print_human_metric(
	    "expansion vs JPEG", format_ratio(pad.expansion_vs_jpeg), format_ratio(ragged.expansion_vs_jpeg));
	print_human_metric("semantic ratio",
	                   format_ratio(pad.semantic_compression_ratio),
	                   format_ratio(ragged.semantic_compression_ratio));
	print_human_metric("encode", format_ms(pad.encode_ms), format_ms(ragged.encode_ms));
	print_human_metric("decode",
	                   pad.decode_supported ? format_ms(pad.decode_ms) : "unsupported",
	                   ragged.decode_supported ? format_ms(ragged.decode_ms) : "unsupported");
	print_human_metric("crop lookup",
	                   pad.decode_supported ? format_ns(pad.crop_lookup_latency_ns) : "unsupported",
	                   ragged.decode_supported ? format_ns(ragged.crop_lookup_latency_ns) : "unsupported");
	print_human_metric("peak RSS", format_bytes(pad.peak_rss_bytes), format_bytes(ragged.peak_rss_bytes));
	print_human_metric("real rows", format_count(pad.real_row_count), format_count(ragged.real_row_count));
	print_human_metric("padding rows", format_count(pad.padding_row_count), format_count(ragged.padding_row_count));
	print_human_metric("physical rows", format_count(pad.physical_row_count), format_count(ragged.physical_row_count));
	print_human_metric("shards", format_count(pad.shard_count), format_count(ragged.shard_count));
	print_human_metric("rowgroups", format_count(pad.rowgroup_count), format_count(ragged.rowgroup_count));
	print_human_metric("block groups", format_count(pad.block_group_count), format_count(ragged.block_group_count));
	print_human_metric("physical ratio",
	                   format_ratio(pad.physical_compression_ratio),
	                   format_ratio(ragged.physical_compression_ratio));
	if (!pad.decode_supported || !ragged.decode_supported) {
		std::cout << "\ndecode note:\n";
		if (!pad.decode_supported) {
			std::cout << "  pad: " << pad.decode_error << '\n';
		}
		if (!ragged.decode_supported) {
			std::cout << "  ragged: " << ragged.decode_error << '\n';
		}
	}
}

} // namespace

int main(const int argc, char** argv) {
	try {
		Options options;
		if (!parse_args(argc, argv, options)) {
			print_usage(argv[0]);
			return 1;
		}

		std::filesystem::create_directories(options.output_dir);
		const auto paths = collect_jpegs(options.input);

		const auto pad = run_policy(
		    paths, options.output_dir, options, "pad", galp::jpeg::JpegDatasetValidationMode::kPadToMaxComponentGrids);
		const auto ragged = run_policy(
		    paths, options.output_dir, options, "ragged", galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor);
		if (options.csv) {
			print_csv_header();
			print_csv_row(pad);
			print_csv_row(ragged);
		} else {
			print_human_report(paths, options.output_dir, pad, ragged);
		}
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "galp_jpeg_dct_policy_bench: " << e.what() << '\n';
		return 2;
	}
}
