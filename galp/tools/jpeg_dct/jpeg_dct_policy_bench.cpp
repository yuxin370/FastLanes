#include "core/operator_capabilities.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "format/compact_descriptor_v3.hpp"
#include "galp/jpeg_dct.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <set>
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
	galp::jpeg::JpegDctSpatialOrder spatial_order                 = galp::jpeg::JpegDctSpatialOrder::kTiledZ32;
	bool                             spatial_order_specified       = false;
	bool                            compact_v3                    = false;
};

struct TokenStat {
	uint64_t rowgroup_count   = 0;
	uint64_t column_count     = 0;
	uint64_t compressed_bytes = 0;
};

using TokenStats = std::map<fastlanes::OperatorToken, TokenStat>;

struct BenchmarkResult {
	std::string label;
	uintmax_t   total_output_size          = 0;
	uintmax_t   compressed_data_size       = 0;
	uintmax_t   metadata_size              = 0;
	uintmax_t   source_jpeg_size           = 0;
	uintmax_t                  raw_dct_size                     = 0;
	uintmax_t                  compressed_payload_size          = 0;
	uintmax_t                  source_descriptor_size           = 0;
	uintmax_t                  compact_descriptor_size          = 0;
	uintmax_t                  temporary_build_peak_disk_size   = 0;
	uintmax_t                  projected_50000_image_total_size = 0;
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
	double                     reader_startup_ms                = 0.0;
	double                     first_access_ms                  = 0.0;
	double                     hot_decode_ms                    = 0.0;
	double      crop_lookup_latency_ns     = 0.0;
	uint64_t    peak_rss_bytes             = 0;
	uint64_t                   encode_peak_rss_bytes            = 0;
	uint64_t                   first_decode_peak_rss_bytes      = 0;
	bool        decode_supported           = true;
	std::string decode_error;
	TokenStats                 token_stats;
	std::array<TokenStats, 64> coefficient_token_stats;
};

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " --input jpeg_dir [--out-dir output_dir] [--csv]\n"
	          << "      [--preset crop-latency|balanced|throughput|random-access]\n"
	          << "      [--shard-images N] [--rowgroup-vectors N] [--rowgroups-per-shard N]\n"
	          << "      [--spatial-order raster|tiled-raster-32|z-order|tiled-z-32]\n"
	          << "      [--compact-v3]\n";
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
	case galp::jpeg::JpegDctShardPreset::kRandomAccess:
		preset_shard_images        = 8192;
		preset_rowgroup_vectors    = 128;
		preset_rowgroups_per_shard = 8192;
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
		if (arg == "--compact-v3") {
			options.compact_v3 = true;
			continue;
		}
		if (arg == "--spatial-order" && i + 1 < argc) {
			const std::string_view order = argv[++i];
			if (order == "raster") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kRaster;
			} else if (order == "tiled-raster-32") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kTiledRaster32;
			} else if (order == "z-order") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kZOrder;
			} else if (order == "tiled-z-32") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kTiledZ32;
			} else {
				throw std::runtime_error(
				    "unknown --spatial-order value; expected raster, tiled-raster-32, z-order, or tiled-z-32");
			}
			options.spatial_order_specified = true;
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
			} else if (preset == "random-access") {
				options.preset = galp::jpeg::JpegDctShardPreset::kRandomAccess;
			} else {
				throw std::runtime_error(
				    "unknown --preset value; expected crop-latency, balanced, throughput, or random-access");
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
	if (options.compact_v3) {
		shard_options.physical_layout            = galp::jpeg::JpegDctPhysicalLayout::kImageMajorVectorRowgroups;
		shard_options.physical_layout_specified  = true;
		shard_options.rowgroup_vectors           = 1U;
		shard_options.rowgroup_vectors_specified = true;
	} else if (options.spatial_order_specified) {
		shard_options.physical_layout           = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
		shard_options.physical_layout_specified = true;
	}
	return shard_options;
}

std::string spatial_order_name(const galp::jpeg::JpegDctSpatialOrder order) {
	switch (order) {
	case galp::jpeg::JpegDctSpatialOrder::kRaster:
		return "raster";
	case galp::jpeg::JpegDctSpatialOrder::kTiledRaster32:
		return "tiled_raster_32";
	case galp::jpeg::JpegDctSpatialOrder::kZOrder:
		return "z_order";
	case galp::jpeg::JpegDctSpatialOrder::kTiledZ32:
		return "tiled_z_32";
	default:
		throw std::runtime_error("unknown JPEG DCT spatial order");
	}
}

void scan_expression_stats(const std::filesystem::path& fls_path, BenchmarkResult& result) {
	if (galp::format::is_compact_v3_fls(fls_path)) {
		auto descriptor = galp::format::CompactDescriptorV3::Open(fls_path);
		for (size_t rowgroup_index = 0U; rowgroup_index < descriptor.rowgroup_count(); ++rowgroup_index) {
			auto rowgroup = descriptor.unpack_rowgroup(rowgroup_index);
			if (rowgroup->m_column_descriptors.size() > result.coefficient_token_stats.size()) {
				throw std::runtime_error("JPEG DCT audit found more than 64 coefficient columns");
			}
			std::set<fastlanes::OperatorToken> rowgroup_tokens;
			for (size_t coeff_id = 0U; coeff_id < rowgroup->m_column_descriptors.size(); ++coeff_id) {
				const auto& column = rowgroup->m_column_descriptors[coeff_id];
				if (column == nullptr || column->encoding_rpn == nullptr ||
				    column->encoding_rpn->operator_tokens.empty()) {
					throw std::runtime_error("missing operator token while scanning " + fls_path.string());
				}
				const auto token = column->encoding_rpn->operator_tokens.front();
				const auto bytes = static_cast<uint64_t>(column->total_size);
				auto&      total = result.token_stats[token];
				++total.column_count;
				total.compressed_bytes += bytes;
				auto& by_coefficient = result.coefficient_token_stats[coeff_id][token];
				++by_coefficient.column_count;
				++by_coefficient.rowgroup_count;
				by_coefficient.compressed_bytes += bytes;
				rowgroup_tokens.insert(token);
			}
			for (const auto token : rowgroup_tokens) {
				++result.token_stats[token].rowgroup_count;
			}
		}
		return;
	}
	fastlanes::File       file(fls_path);
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	fastlanes::FileHeader::Load(header, file);
	fastlanes::FileFooter::Load(footer, file);
	const auto handle =
	    header.settings.inline_footer
	                        ? fastlanes::TableDescriptorHandle::FromFileSlice(
	                              file, footer.table_descriptor_offset, footer.table_descriptor_size, true)
	        : fastlanes::TableDescriptorHandle::FromFile(fls_path.parent_path() / "table_descriptor.fbb", true);
	const auto* table  = handle.Get();
	if (table == nullptr || table->m_rowgroup_descriptors() == nullptr) {
		throw std::runtime_error("missing table/rowgroup descriptor while scanning " + fls_path.string());
	}
	for (const auto* rowgroup : *table->m_rowgroup_descriptors()) {
		if (rowgroup == nullptr || rowgroup->m_column_descriptors() == nullptr) {
			throw std::runtime_error("missing rowgroup/column descriptor while scanning " + fls_path.string());
		}
		std::set<fastlanes::OperatorToken> rowgroup_tokens;
		for (size_t coeff_id = 0; coeff_id < rowgroup->m_column_descriptors()->size(); ++coeff_id) {
			const auto* column = rowgroup->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(coeff_id));
			if (column == nullptr || column->encoding_rpn() == nullptr ||
			    column->encoding_rpn()->operator_tokens() == nullptr ||
			    column->encoding_rpn()->operator_tokens()->empty()) {
				throw std::runtime_error("missing operator token while scanning " + fls_path.string());
			}
			if (coeff_id >= result.coefficient_token_stats.size()) {
				throw std::runtime_error("JPEG DCT audit found more than 64 coefficient columns");
			}
			const auto token = column->encoding_rpn()->operator_tokens()->Get(0);
			const auto bytes = static_cast<uint64_t>(column->total_size());
			auto&      total = result.token_stats[token];
			++total.column_count;
			total.compressed_bytes += bytes;
			auto& by_coefficient = result.coefficient_token_stats[coeff_id][token];
			++by_coefficient.column_count;
			by_coefficient.compressed_bytes += bytes;
			rowgroup_tokens.insert(token);
		}
		for (const auto token : rowgroup_tokens) {
			++result.token_stats[token].rowgroup_count;
		}
		for (size_t coeff_id = 0; coeff_id < rowgroup->m_column_descriptors()->size(); ++coeff_id) {
			const auto* column = rowgroup->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(coeff_id));
			const auto token = column->encoding_rpn()->operator_tokens()->Get(0);
			++result.coefficient_token_stats[coeff_id][token].rowgroup_count;
		}
	}
}

double measure_crop_lookup_latency_ns(galp::jpeg::JpegDctShardDatasetReader& reader) {
	const auto image = reader.ImageMetadata(0);
	auto component = std::find_if(image.components.begin(), image.components.end(), [](const auto& value) {
		return value.present && value.width_in_blocks != 0 && value.height_in_blocks != 0;
	});
	if (component == image.components.end()) {
		return 0.0;
	}
	constexpr size_t iterations = 10000;
	uint64_t         checksum   = 0;
	const auto start = Clock::now();
	for (size_t iteration = 0; iteration < iterations; ++iteration) {
		const auto x = static_cast<uint32_t>(iteration % component->width_in_blocks);
		const auto y = static_cast<uint32_t>((iteration / component->width_in_blocks) % component->height_in_blocks);
		const auto ref = reader.LocateRow(0, component->semantic_slot_id, x, y);
		checksum += ref.physical_row_index;
	}
	const auto end = Clock::now();
	if (checksum == std::numeric_limits<uint64_t>::max()) {
		std::cerr << "";
	}
	return std::chrono::duration<double, std::nano>(end - start).count() / iterations;
}

BenchmarkResult run_benchmark(const std::vector<std::filesystem::path>& paths,
                              const std::filesystem::path&              output_dir,
                              const Options&                            cli_options,
                              const std::string&                        label) {
	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.image_major_spatial_order = cli_options.spatial_order;

	const auto dataset_dir = output_dir / label;
	std::filesystem::remove_all(dataset_dir);

	const auto encode_start = Clock::now();
	const auto manifest     = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    paths, dataset_dir, reader_options, make_shard_options(cli_options));
	const auto encode_end = Clock::now();

	BenchmarkResult result;
	result.label            = label;
	result.source_jpeg_size = total_source_jpeg_size(paths);
	result.encode_ms        = elapsed_ms(encode_start, encode_end);
	result.shard_count      = static_cast<uint32_t>(manifest.shards.size());
	result.encode_peak_rss_bytes = peak_rss_bytes();

	uintmax_t completed_runtime_bytes = 0U;
	for (const auto& shard : manifest.shards) {
		if (cli_options.compact_v3) {
			constexpr uintmax_t fls_envelope_bytes = 48U;
			const auto standard_temporary_size = fls_envelope_bytes + shard.payload_size + shard.source_descriptor_size;
			const auto concurrent_peak = completed_runtime_bytes + standard_temporary_size + shard.fls_file_size;
			result.temporary_build_peak_disk_size = std::max(result.temporary_build_peak_disk_size, concurrent_peak);
		}
		result.compressed_data_size += shard.fls_file_size;
		result.compressed_payload_size += shard.payload_size;
		result.source_descriptor_size += shard.source_descriptor_size;
		result.compact_descriptor_size += shard.compact_descriptor_size;
		result.metadata_size += shard.metadata_file_size;
		result.real_row_count += shard.real_row_count;
		result.padding_row_count += shard.padding_row_count;
		result.physical_row_count += shard.physical_row_count;
		result.rowgroup_count += shard.rowgroup_count;
		result.block_group_count += shard.block_group_count;
		scan_expression_stats(dataset_dir / shard.fls_file_name, result);
		completed_runtime_bytes += shard.fls_file_size + shard.metadata_file_size;
	}
	result.metadata_size += std::filesystem::file_size(dataset_dir / "manifest.bin");
	result.total_output_size = result.compressed_data_size + result.metadata_size;
	result.temporary_build_peak_disk_size = std::max(result.temporary_build_peak_disk_size, result.total_output_size);
	for (const auto& entry : std::filesystem::recursive_directory_iterator(dataset_dir)) {
		if (entry.is_regular_file() && entry.path().extension() == ".svb") {
			throw std::runtime_error("Compact-v3 benchmark output unexpectedly contains an .svb sidecar");
		}
	}
	result.raw_dct_size = static_cast<uintmax_t>(result.real_row_count) * 64U * sizeof(int16_t);
	if (!paths.empty()) {
		result.projected_50000_image_total_size = static_cast<uintmax_t>(std::ceil(
		    static_cast<long double>(result.total_output_size) * 50000.0L / static_cast<long double>(paths.size())));
	}

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
		const auto                            reader_start = Clock::now();
		galp::jpeg::JpegDctShardDatasetReader reader(dataset_dir / "manifest.bin");
		const auto                            reader_ready = Clock::now();
		const auto                            decode_start = Clock::now();
		auto                                  image        = reader.MaterializeImageDct(0);
		if (image.blocks.empty()) {
			std::cerr << "";
		}
		const auto decode_end         = Clock::now();
		const auto hot_decode_start = Clock::now();
		auto       hot_image        = reader.MaterializeImageDct(0);
		const auto hot_decode_end   = Clock::now();
		if (hot_image.blocks.empty()) {
			std::cerr << "";
		}
		result.reader_startup_ms      = elapsed_ms(reader_start, reader_ready);
		result.decode_ms              = elapsed_ms(decode_start, decode_end);
		result.first_access_ms        = elapsed_ms(reader_start, decode_end);
		result.hot_decode_ms          = elapsed_ms(hot_decode_start, hot_decode_end);
		result.crop_lookup_latency_ns = measure_crop_lookup_latency_ns(reader);
	} catch (const std::exception& e) {
		result.decode_supported = false;
		result.decode_error     = e.what();
	}
	result.peak_rss_bytes = peak_rss_bytes();
	result.first_decode_peak_rss_bytes = result.peak_rss_bytes;
	return result;
}

void write_result_csv_files(const std::filesystem::path& output_dir, const BenchmarkResult& result) {
	{
		std::ofstream out(output_dir / (result.label + "_summary.csv"));
		out << "layout,raw_dct_size,compressed_payload_size,source_descriptor_size,compact_descriptor_size,"
		       "temporary_build_peak_disk_size,projected_50000_image_total_size,"
		       "compressed_data_size,total_output_size,semantic_compression_ratio,encode_ms,reader_startup_ms,"
		       "first_access_ms,decode_ms,hot_decode_ms,row_lookup_ns,"
		       "encode_peak_rss_bytes,first_decode_peak_rss_bytes,rowgroup_count,shard_count\n";
		out << result.label << ',' << result.raw_dct_size << ',' << result.compressed_payload_size << ','
		    << result.source_descriptor_size << ',' << result.compact_descriptor_size << ','
		    << result.temporary_build_peak_disk_size << ',' << result.projected_50000_image_total_size << ','
		    << result.compressed_data_size << ',' << result.total_output_size << ','
		    << result.semantic_compression_ratio << ',' << result.encode_ms << ',' << result.reader_startup_ms << ','
		    << result.first_access_ms << ',' << result.decode_ms << ',' << result.hot_decode_ms << ','
		    << result.crop_lookup_latency_ns << ',' << result.encode_peak_rss_bytes << ','
		    << result.first_decode_peak_rss_bytes << ',' << result.rowgroup_count << ',' << result.shard_count << '\n';
	}
	{
		std::ofstream out(output_dir / (result.label + "_tokens.csv"));
		out << "layout,token,gpu_supported,rowgroup_count,column_count,compressed_bytes\n";
		for (const auto& [token, stat] : result.token_stats) {
			out << result.label << ',' << fastlanes::token_to_string(token) << ','
			    << (galp::expression::is_supported_token(token) ? "true" : "false") << ',' << stat.rowgroup_count << ','
			    << stat.column_count << ',' << stat.compressed_bytes << '\n';
		}
	}
	{
		std::ofstream out(output_dir / (result.label + "_coefficients.csv"));
		out << "layout,coefficient_id,token,gpu_supported,rowgroup_count,column_count,compressed_bytes\n";
		for (size_t coefficient = 0; coefficient < result.coefficient_token_stats.size(); ++coefficient) {
			for (const auto& [token, stat] : result.coefficient_token_stats[coefficient]) {
				out << result.label << ',' << coefficient << ',' << fastlanes::token_to_string(token) << ','
				    << (galp::expression::is_supported_token(token) ? "true" : "false") << ',' << stat.rowgroup_count
				    << ',' << stat.column_count << ',' << stat.compressed_bytes << '\n';
			}
		}
	}
}

void print_wizard_expression_stats(const BenchmarkResult& result) {
	std::cout << "\nWizard-selected root expressions\n";
	std::cout << "token,rowgroups,columns,compressed_bytes,gpu_supported\n";
	for (const auto& [token, stat] : result.token_stats) {
		std::cout << fastlanes::token_to_string(token) << ',' << stat.rowgroup_count << ',' << stat.column_count << ','
		          << stat.compressed_bytes << ',' << (galp::expression::is_supported_token(token) ? "true" : "false")
		          << '\n';
	}
}

void print_csv_header() {
	std::cout << "layout,total_output_size,raw_dct_size,compressed_payload_size,source_descriptor_size,"
	             "compact_descriptor_size,temporary_build_peak_disk_size,projected_50000_image_total_size,"
	             "semantic_compression_ratio,encode_ms,decode_ms,"
	             "reader_startup_ms,first_access_ms,hot_decode_ms,crop_lookup_latency_ns,"
	             "expansion_vs_jpeg,peak_rss_bytes,encode_peak_rss_bytes,"
	             "first_decode_peak_rss_bytes,"
	             "compressed_data_size,metadata_size,source_jpeg_size,real_row_count,padding_row_count,"
	             "physical_row_count,shard_count,rowgroup_count,block_group_count,physical_compression_ratio,"
	             "decode_supported,decode_error\n";
}

void print_csv_row(const BenchmarkResult& r) {
	std::cout << r.label << ',' << r.total_output_size << ',' << r.raw_dct_size << ',' << r.compressed_payload_size
	          << ',' << r.source_descriptor_size << ',' << r.compact_descriptor_size << ','
	          << r.temporary_build_peak_disk_size << ',' << r.projected_50000_image_total_size << ','
	          << r.semantic_compression_ratio << ',' << r.encode_ms << ',' << r.decode_ms << ',' << r.reader_startup_ms
	          << ',' << r.first_access_ms << ',' << r.hot_decode_ms << ',' << r.crop_lookup_latency_ns << ','
	          << r.expansion_vs_jpeg << ',' << r.peak_rss_bytes << ',' << r.encode_peak_rss_bytes << ','
	          << r.first_decode_peak_rss_bytes << ',' << r.compressed_data_size << ',' << r.metadata_size << ','
	          << r.source_jpeg_size << ',' << r.real_row_count << ',' << r.padding_row_count << ','
	          << r.physical_row_count << ',' << r.shard_count << ',' << r.rowgroup_count << ',' << r.block_group_count
	          << ',' << r.physical_compression_ratio << ',' << (r.decode_supported ? "true" : "false") << ',' << '"'
	          << r.decode_error << '"' << '\n';
}

void print_human_metric(const char* name, const std::string& value) {
	std::cout << std::left << std::setw(30) << name << std::right << std::setw(18) << value << '\n';
}

void print_human_report(const std::vector<std::filesystem::path>& paths,
                        const std::filesystem::path&              output_dir,
                        const BenchmarkResult&                    result) {
	std::cout << "JPEG DCT sharded benchmark\n";
	std::cout << "images: " << format_count(paths.size()) << '\n';
	std::cout << "output_dir: " << output_dir << "\n\n";
	std::cout << std::left << std::setw(30) << "metric" << std::right << std::setw(18) << result.label << '\n';
	std::cout << std::string(48, '-') << '\n';
	print_human_metric("total output", format_bytes(result.total_output_size));
	print_human_metric("original raw DCT", format_bytes(result.raw_dct_size));
	print_human_metric("compressed payload", format_bytes(result.compressed_payload_size));
	print_human_metric("source descriptor", format_bytes(result.source_descriptor_size));
	print_human_metric("compact descriptor", format_bytes(result.compact_descriptor_size));
	print_human_metric("temporary build peak", format_bytes(result.temporary_build_peak_disk_size));
	print_human_metric("50K projected total", format_bytes(result.projected_50000_image_total_size));
	print_human_metric("compressed data", format_bytes(result.compressed_data_size));
	print_human_metric("metadata", format_bytes(result.metadata_size));
	print_human_metric("source JPEG", format_bytes(result.source_jpeg_size));
	print_human_metric("expansion vs JPEG", format_ratio(result.expansion_vs_jpeg));
	print_human_metric("semantic ratio", format_ratio(result.semantic_compression_ratio));
	print_human_metric("encode", format_ms(result.encode_ms));
	print_human_metric("reader startup", result.decode_supported ? format_ms(result.reader_startup_ms) : "unsupported");
	print_human_metric("first access", result.decode_supported ? format_ms(result.first_access_ms) : "unsupported");
	print_human_metric("decode", result.decode_supported ? format_ms(result.decode_ms) : "unsupported");
	print_human_metric("hot decode", result.decode_supported ? format_ms(result.hot_decode_ms) : "unsupported");
	print_human_metric("crop lookup",
	                   result.decode_supported ? format_ns(result.crop_lookup_latency_ns) : "unsupported");
	print_human_metric("peak RSS", format_bytes(result.peak_rss_bytes));
	print_human_metric("encode peak RSS", format_bytes(result.encode_peak_rss_bytes));
	print_human_metric("first decode peak RSS", format_bytes(result.first_decode_peak_rss_bytes));
	print_human_metric("real rows", format_count(result.real_row_count));
	print_human_metric("padding rows", format_count(result.padding_row_count));
	print_human_metric("physical rows", format_count(result.physical_row_count));
	print_human_metric("shards", format_count(result.shard_count));
	print_human_metric("rowgroups", format_count(result.rowgroup_count));
	print_human_metric("block groups", format_count(result.block_group_count));
	print_human_metric("physical ratio", format_ratio(result.physical_compression_ratio));
	if (!result.decode_supported) {
		std::cout << "\ndecode note:\n";
		std::cout << "  " << result.label << ": " << result.decode_error << '\n';
	}
	print_wizard_expression_stats(result);
}

} // namespace

int main(const int argc, char** argv) {
	try {
		Options options;
		if (!parse_args(argc, argv, options)) {
			print_usage(argv[0]);
			return 1;
		}
		if (options.compact_v3 && options.spatial_order != galp::jpeg::JpegDctSpatialOrder::kTiledZ32) {
			throw std::invalid_argument("--compact-v3 requires --spatial-order tiled-z-32");
		}

		std::filesystem::create_directories(options.output_dir);
		const auto paths = collect_jpegs(options.input);

		const bool image_major = options.compact_v3 || options.spatial_order_specified ||
		                         options.preset == galp::jpeg::JpegDctShardPreset::kRandomAccess;
		const auto label  = options.compact_v3 ? "compact_v3_" + spatial_order_name(options.spatial_order)
		                    : image_major      ? spatial_order_name(options.spatial_order)
		                                       : "spatial_major";
		const auto result = run_benchmark(paths, options.output_dir, options, label);
		write_result_csv_files(options.output_dir, result);
		if (options.csv) {
			print_csv_header();
			print_csv_row(result);
		} else {
			print_human_report(paths, options.output_dir, result);
		}
		if (options.compact_v3) {
			constexpr uintmax_t kV2HundredImageBytes            = 26914204U;
			constexpr uintmax_t kProjected50000ImageBudgetBytes = 13244000000ULL;
			constexpr uint64_t  kPeakRssBudgetBytes             = 1000000000ULL;
			constexpr double    kColdAccessBudgetMs             = 500.0;
			constexpr double    kLookupBudgetNs                 = 200000.0;
			if (result.source_descriptor_size == 0U ||
			    result.compact_descriptor_size > result.source_descriptor_size / 5U) {
				std::cerr << "compact-v3 gate failed: descriptor reduction is below 80%\n";
				return 3;
			}
			if (paths.size() == 100U && result.total_output_size > kV2HundredImageBytes) {
				std::cerr << "compact-v3 gate failed: 100-image total exceeds the 26,914,204-byte v2 baseline\n";
				return 3;
			}
			if (result.projected_50000_image_total_size > kProjected50000ImageBudgetBytes) {
				std::cerr << "compact-v3 gate failed: projected 50K total exceeds 13.244 GB\n";
				return 3;
			}
			if (!result.decode_supported || result.reader_startup_ms >= kColdAccessBudgetMs ||
			    result.first_access_ms >= kColdAccessBudgetMs) {
				std::cerr << "compact-v3 gate failed: cold reader startup/first access is not below 500 ms\n";
				return 3;
			}
			if (result.crop_lookup_latency_ns >= kLookupBudgetNs) {
				std::cerr << "compact-v3 gate failed: crop lookup is not below 200 us\n";
				return 3;
			}
			if (result.peak_rss_bytes >= kPeakRssBudgetBytes) {
				std::cerr << "compact-v3 gate failed: measured peak RSS is not below 1 GB\n";
				return 3;
			}
		}
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "galp_jpeg_dct_policy_bench: " << e.what() << '\n';
		return 2;
	}
}
