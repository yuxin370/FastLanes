#include "galp/jpeg_dct.hpp"
#include "galp/table.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
	std::filesystem::path input;
	std::filesystem::path output_dir = "/tmp/galp_jpeg_dct_policy_bench";
	bool                  csv = false;
};

struct PolicyResult {
	std::string policy;
	uintmax_t   compressed_data_size = 0;
	uintmax_t   metadata_size = 0;
	uintmax_t   total_output_size = 0;
	size_t      real_row_count = 0;
	size_t      padding_row_count = 0;
	size_t      physical_row_count = 0;
	size_t      estimated_block_group_index_size = 0;
	double      physical_compression_ratio = 0.0;
	double      semantic_compression_ratio = 0.0;
	double      encode_ms = 0.0;
	double      decode_ms = 0.0;
	bool        decode_supported = true;
	std::string decode_error;
	double      in_memory_first_block_group_scan_ns = 0.0;
};

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " --input jpeg_dir [--out-dir output_dir] [--csv]\n";
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
		if (arg == "--help" || arg == "-h") {
			return false;
		}
		throw std::runtime_error("unknown argument: " + std::string(arg));
	}
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

std::string format_bytes(const uintmax_t bytes) {
	static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
	double value = static_cast<double>(bytes);
	size_t unit = 0;
	while (value >= 1024.0 && unit + 1 < (sizeof(units) / sizeof(units[0]))) {
		value /= 1024.0;
		++unit;
	}
	std::ostringstream out;
	out << std::fixed << std::setprecision(value < 10.0 ? 2 : value < 100.0 ? 1 : 0) << value << ' ' << units[unit];
	return out.str();
}

std::string format_count(const size_t value) {
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

size_t estimate_block_group_index_size(const galp::jpeg::JpegDctTable& table) {
	return table.block_group_count * sizeof(uint64_t) * 2;
}

double measure_in_memory_first_block_group_scan_ns(const galp::jpeg::JpegDctTable& table, const size_t image_count) {
	const auto rows_to_read = std::min(image_count, table.row_count);
	if (rows_to_read == 0) {
		return 0.0;
	}

	int64_t    sink = 0;
	const auto start = Clock::now();
	for (size_t row = 0; row < rows_to_read; ++row) {
		for (const auto& column : table.columns) {
			sink += column[row];
		}
	}
	const auto end = Clock::now();
	if (sink == std::numeric_limits<int64_t>::min()) {
		std::cerr << "";
	}
	return std::chrono::duration<double, std::nano>(end - start).count();
}

PolicyResult run_policy(const std::vector<std::filesystem::path>& paths,
                        const std::filesystem::path&              output_dir,
                        const std::string&                       policy_name,
                        const galp::jpeg::JpegDatasetValidationMode policy) {
	galp::jpeg::JpegDctReaderOptions options;
	options.validation_mode = policy;

	const auto table = galp::jpeg::read_jpeg_dct_dataset(paths, options);

	const auto fls_path = output_dir / ("jpeg_dct_" + policy_name + ".fls");
	const auto meta_path = output_dir / ("jpeg_dct_" + policy_name + ".metadata.bin");
	if (std::filesystem::exists(fls_path)) {
		std::filesystem::remove(fls_path);
	}
	if (std::filesystem::exists(meta_path)) {
		std::filesystem::remove(meta_path);
	}

	const auto encode_start = Clock::now();
	galp::jpeg::compress_jpeg_dct_to_fls(table, fls_path, meta_path);
	const auto encode_end = Clock::now();

	PolicyResult result;
	result.policy = policy_name;
	result.compressed_data_size = std::filesystem::file_size(fls_path);
	result.metadata_size = std::filesystem::file_size(meta_path);
	result.total_output_size = result.compressed_data_size + result.metadata_size;
	result.real_row_count = table.real_row_count;
	result.padding_row_count = table.padding_row_count;
	result.physical_row_count = table.row_count;
	result.estimated_block_group_index_size = estimate_block_group_index_size(table);
	result.encode_ms = elapsed_ms(encode_start, encode_end);

	const auto decode_start = Clock::now();
	try {
		auto decoded = galp::decompress_table(fls_path);
		(void)decoded;
		const auto decode_end = Clock::now();
		result.decode_ms = elapsed_ms(decode_start, decode_end);
	} catch (const std::exception& e) {
		const auto decode_end = Clock::now();
		result.decode_ms = elapsed_ms(decode_start, decode_end);
		result.decode_supported = false;
		result.decode_error = e.what();
	}

	result.in_memory_first_block_group_scan_ns = measure_in_memory_first_block_group_scan_ns(table, paths.size());

	const double physical_uncompressed = static_cast<double>(table.row_count) * 64.0 * sizeof(int16_t);
	const double semantic_uncompressed = static_cast<double>(table.real_row_count) * 64.0 * sizeof(int16_t);
	if (result.compressed_data_size != 0) {
		result.physical_compression_ratio = physical_uncompressed / static_cast<double>(result.compressed_data_size);
		result.semantic_compression_ratio = semantic_uncompressed / static_cast<double>(result.compressed_data_size);
	}
	return result;
}

void print_csv_header() {
	std::cout << "policy,total_output_size,semantic_compression_ratio,encode_ms,decode_ms,"
	             "in_memory_first_block_group_scan_ns,"
	             "compressed_data_size,metadata_size,real_row_count,padding_row_count,physical_row_count,"
	             "estimated_block_group_index_size,physical_compression_ratio,decode_supported,decode_error\n";
}

void print_csv_row(const PolicyResult& r) {
	std::cout << r.policy << ',' << r.total_output_size << ',' << r.semantic_compression_ratio << ','
	          << r.encode_ms << ',' << r.decode_ms << ',' << r.in_memory_first_block_group_scan_ns << ','
	          << r.compressed_data_size << ',' << r.metadata_size << ',' << r.real_row_count << ','
	          << r.padding_row_count << ',' << r.physical_row_count << ',' << r.estimated_block_group_index_size << ','
	          << r.physical_compression_ratio << ',' << (r.decode_supported ? "true" : "false") << ','
	          << '"' << r.decode_error << '"' << '\n';
}

void print_human_metric(const char* name, const std::string& pad, const std::string& ragged) {
	std::cout << std::left << std::setw(30) << name << std::right << std::setw(18) << pad << std::setw(18) << ragged
	          << '\n';
}

void print_human_report(const std::vector<std::filesystem::path>& paths,
                        const std::filesystem::path&              output_dir,
                        const PolicyResult&                       pad,
                        const PolicyResult&                       ragged) {
	std::cout << "JPEG DCT policy benchmark\n";
	std::cout << "images: " << format_count(paths.size()) << '\n';
	std::cout << "output_dir: " << output_dir << "\n\n";
	std::cout << std::left << std::setw(30) << "metric" << std::right << std::setw(18) << "pad" << std::setw(18)
	          << "ragged" << '\n';
	std::cout << std::string(66, '-') << '\n';
	print_human_metric("total output", format_bytes(pad.total_output_size), format_bytes(ragged.total_output_size));
	print_human_metric("compressed data", format_bytes(pad.compressed_data_size), format_bytes(ragged.compressed_data_size));
	print_human_metric("metadata", format_bytes(pad.metadata_size), format_bytes(ragged.metadata_size));
	print_human_metric("semantic ratio",
	                   format_ratio(pad.semantic_compression_ratio),
	                   format_ratio(ragged.semantic_compression_ratio));
	print_human_metric("encode", format_ms(pad.encode_ms), format_ms(ragged.encode_ms));
	print_human_metric("decode",
	                   pad.decode_supported ? format_ms(pad.decode_ms) : "unsupported",
	                   ragged.decode_supported ? format_ms(ragged.decode_ms) : "unsupported");
	print_human_metric("in-memory group scan",
	                   format_ns(pad.in_memory_first_block_group_scan_ns),
	                   format_ns(ragged.in_memory_first_block_group_scan_ns));
	print_human_metric("real rows", format_count(pad.real_row_count), format_count(ragged.real_row_count));
	print_human_metric("padding rows", format_count(pad.padding_row_count), format_count(ragged.padding_row_count));
	print_human_metric("physical rows", format_count(pad.physical_row_count), format_count(ragged.physical_row_count));
	print_human_metric("est. block-group index",
	                   format_bytes(pad.estimated_block_group_index_size),
	                   format_bytes(ragged.estimated_block_group_index_size));
	print_human_metric("physical ratio",
	                   format_ratio(pad.physical_compression_ratio),
	                   format_ratio(ragged.physical_compression_ratio));
	if (!pad.decode_supported || !ragged.decode_supported) {
		std::cout << "\nDecode note:\n";
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

		const auto pad = run_policy(paths,
		                            options.output_dir,
		                            "pad",
		                            galp::jpeg::JpegDatasetValidationMode::kPadToMaxComponentGrids);
		const auto ragged = run_policy(paths,
		                               options.output_dir,
		                               "ragged",
		                               galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor);
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
