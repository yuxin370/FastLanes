#include "core/operator_capabilities.hpp"
#include "fls/connection.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/table/memory_table.hpp"
#include "galp/direct_dct.hpp"
#include "galp/jpeg_dct.hpp"
#include "galp/profiles/rgbnomore.hpp"
#include "galp_tools/benchmark_support/pipeline.cuh"
#include "jpeg/jpeg_dct_cuda_internal.cuh"
#include "jpeg/jpeg_dct_expression_validation.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <jpeglib.h>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

struct FileCloser {
	void operator()(FILE* file) const {
		if (file != nullptr) {
			std::fclose(file);
		}
	}
};

using FilePtr = std::unique_ptr<FILE, FileCloser>;

struct MetadataHeader {
	std::array<uint8_t, 8> magic {};
	uint16_t               version         = 0;
	uint16_t               profile         = 0;
	uint16_t               row_ordering    = 0;
	uint16_t               validation_mode = 0;
	uint16_t               layout_flags    = 0;
};

struct ShardManifestHeader {
	std::array<uint8_t, 8> magic {};
	uint32_t               version             = 0;
	uint16_t               policy              = 0;
	uint32_t               rowgroup_vectors    = 0;
	uint32_t               rowgroups_per_shard = 0;
	uint64_t               image_count         = 0;
	uint32_t               shard_count         = 0;
};

enum class TestJpegFormat {
	kDefault420,
	kYcbcr444,
	kYcbcr422,
	kYcbcr440,
	kYcbcr411,
	kGrayscale,
};

void write_test_jpeg(const std::filesystem::path& path,
                     const int                    width  = 16,
                     const int                    height = 8,
                     const TestJpegFormat         format = TestJpegFormat::kDefault420) {
	FilePtr file(std::fopen(path.string().c_str(), "wb"));
	if (!file) {
		throw std::runtime_error("failed to create test JPEG");
	}

	jpeg_compress_struct cinfo {};
	jpeg_error_mgr       jerr {};
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, file.get());

	cinfo.image_width      = static_cast<JDIMENSION>(width);
	cinfo.image_height     = static_cast<JDIMENSION>(height);
	cinfo.input_components = format == TestJpegFormat::kGrayscale ? 1 : 3;
	cinfo.in_color_space   = format == TestJpegFormat::kGrayscale ? JCS_GRAYSCALE : JCS_RGB;
	jpeg_set_defaults(&cinfo);
	if (format == TestJpegFormat::kYcbcr444 || format == TestJpegFormat::kYcbcr422 ||
	    format == TestJpegFormat::kYcbcr440 || format == TestJpegFormat::kYcbcr411) {
		for (int component = 0; component < cinfo.num_components; ++component) {
			cinfo.comp_info[component].h_samp_factor = 1;
			cinfo.comp_info[component].v_samp_factor = 1;
		}
		if (format == TestJpegFormat::kYcbcr422) {
			cinfo.comp_info[0].h_samp_factor = 2;
		} else if (format == TestJpegFormat::kYcbcr440) {
			cinfo.comp_info[0].v_samp_factor = 2;
		} else if (format == TestJpegFormat::kYcbcr411) {
			cinfo.comp_info[0].h_samp_factor = 4;
		}
	}
	jpeg_set_quality(&cinfo, 90, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	const auto                 components = static_cast<size_t>(cinfo.input_components);
	std::vector<unsigned char> row(static_cast<size_t>(width) * components);
	while (cinfo.next_scanline < cinfo.image_height) {
		for (int x = 0; x < width; ++x) {
			if (format == TestJpegFormat::kGrayscale) {
				row[static_cast<size_t>(x)] = static_cast<unsigned char>(x * 11 + cinfo.next_scanline * 7);
			} else {
				row[static_cast<size_t>(x) * 3 + 0] = static_cast<unsigned char>(x * 11);
				row[static_cast<size_t>(x) * 3 + 1] = static_cast<unsigned char>(cinfo.next_scanline * 19);
				row[static_cast<size_t>(x) * 3 + 2] = static_cast<unsigned char>(x * 7 + cinfo.next_scanline);
			}
		}
		JSAMPROW row_ptr = row.data();
		jpeg_write_scanlines(&cinfo, &row_ptr, 1);
	}

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
}

MetadataHeader read_metadata_header(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("failed to open JPEG DCT metadata");
	}

	MetadataHeader header;
	in.read(reinterpret_cast<char*>(header.magic.data()), static_cast<std::streamsize>(header.magic.size()));
	const auto read_u16 = [&] {
		std::array<uint8_t, 2> bytes {};
		in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		return static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8U));
	};
	header.version = read_u16();
	const std::array<uint8_t, 8> sectioned_magic {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'};
	if (header.magic == sectioned_magic) {
		header.profile = read_u16();
	}
	header.row_ordering    = read_u16();
	header.validation_mode = read_u16();
	header.layout_flags    = read_u16();
	return header;
}

ShardManifestHeader read_shard_manifest_header(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("failed to open JPEG DCT shard manifest");
	}

	const auto read_u16 = [&] {
		std::array<uint8_t, 2> bytes {};
		in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		return static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8U));
	};
	const auto read_u32 = [&] {
		std::array<uint8_t, 4> bytes {};
		in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		uint32_t value = 0;
		for (unsigned shift = 0; shift < 32; shift += 8) {
			value |= static_cast<uint32_t>(bytes[shift / 8]) << shift;
		}
		return value;
	};
	const auto read_u64 = [&] {
		std::array<uint8_t, 8> bytes {};
		in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		uint64_t value = 0;
		for (unsigned shift = 0; shift < 64; shift += 8) {
			value |= static_cast<uint64_t>(bytes[shift / 8]) << shift;
		}
		return value;
	};

	ShardManifestHeader header;
	in.read(reinterpret_cast<char*>(header.magic.data()), static_cast<std::streamsize>(header.magic.size()));
	header.version             = read_u32();
	header.policy              = read_u16();
	header.rowgroup_vectors    = read_u32();
	header.rowgroups_per_shard = read_u32();
	header.image_count         = read_u64();
	header.shard_count         = read_u32();
	return header;
}

std::vector<uint8_t> all_dct_coefficients() {
	std::vector<uint8_t> coefficients;
	coefficients.reserve(galp::jpeg::detail::kJpegDctCoefficientCount);
	for (uint8_t coeff = 0; coeff < galp::jpeg::detail::kJpegDctCoefficientCount; ++coeff) {
		coefficients.push_back(coeff);
	}
	return coefficients;
}

std::vector<fastlanes::OperatorToken> read_first_rowgroup_root_tokens(const std::filesystem::path& fls_path) {
	fastlanes::File       file(fls_path);
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	fastlanes::FileHeader::Load(header, file);
	fastlanes::FileFooter::Load(footer, file);
	if (!header.settings.inline_footer) {
		throw std::runtime_error("expected inline footer in synthetic mixed-expression shard");
	}
	const auto descriptor = fastlanes::TableDescriptorHandle::FromFileSlice(
	    file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
	if (descriptor.Get() == nullptr || descriptor.Get()->m_rowgroup_descriptors() == nullptr ||
	    descriptor.Get()->m_rowgroup_descriptors()->size() != 1U) {
		throw std::runtime_error("expected one rowgroup in synthetic mixed-expression shard");
	}
	const auto* rowgroup = descriptor.Get()->m_rowgroup_descriptors()->Get(0);
	if (rowgroup == nullptr || rowgroup->m_column_descriptors() == nullptr) {
		throw std::runtime_error("missing synthetic mixed-expression rowgroup descriptor");
	}
	std::vector<fastlanes::OperatorToken> tokens;
	tokens.reserve(rowgroup->m_column_descriptors()->size());
	for (const auto* column : *rowgroup->m_column_descriptors()) {
		if (column == nullptr || column->encoding_rpn() == nullptr ||
		    column->encoding_rpn()->operator_tokens() == nullptr ||
		    column->encoding_rpn()->operator_tokens()->size() != 1U) {
			throw std::runtime_error("invalid synthetic mixed-expression column descriptor");
		}
		tokens.push_back(column->encoding_rpn()->operator_tokens()->Get(0));
	}
	return tokens;
}

void rewrite_image_major_shard_expression(const galp::jpeg::JpegDctTable& table,
	                                      const std::filesystem::path&      fls_path,
	                                      const uint32_t                   shard_id,
	                                      const uint32_t                   rowgroup_vectors,
	                                      const fastlanes::OperatorToken   target_token,
	                                      const bool                       force_expression) {
	std::array<std::vector<int16_t>, 64> values;
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t coefficient = 0; coefficient < columns.size(); ++coefficient) {
		auto& column_values = values[coefficient];
		column_values.resize(table.row_count);
		for (size_t row = 0; row < column_values.size(); ++row) {
			int16_t value = 0;
			switch (target_token) {
			case fastlanes::OperatorToken::EXP_CONSTANT_I16:
				value = static_cast<int16_t>(static_cast<int>(coefficient) - 32);
				break;
			case fastlanes::OperatorToken::EXP_RLE_I16_U16:
				value = static_cast<int16_t>(static_cast<int>((row / 128U + coefficient) % 19U) - 9);
				break;
			case fastlanes::OperatorToken::EXP_FREQUENCY_I16:
				value = static_cast<int16_t>(static_cast<int>((row * 7U + coefficient) % 11U) - 5);
				break;
			case fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08:
				value = static_cast<int16_t>((row * 37U + coefficient * 13U) % 127U);
				break;
			case fastlanes::OperatorToken::EXP_DELTA_I16:
				value = static_cast<int16_t>(-16000 + static_cast<int>(row) + static_cast<int>(coefficient));
				break;
			case fastlanes::OperatorToken::EXP_DICT_I16_U08:
			case fastlanes::OperatorToken::EXP_DICT_I16_U16: {
				const uint32_t cardinality = target_token == fastlanes::OperatorToken::EXP_DICT_I16_U08 ? 127U : 300U;
				uint32_t multiplier = static_cast<uint32_t>(coefficient * 2U + 1U);
				while (std::gcd(multiplier, cardinality) != 1U) {
					multiplier += 2U;
				}
				const uint32_t rank = static_cast<uint32_t>(row % cardinality);
				value               = static_cast<int16_t>(1000 +
                                             static_cast<int>((rank * multiplier + coefficient * 17U) % cardinality));
				break;
			}
			default: {
				const uint32_t hash = static_cast<uint32_t>(row) * 2654435761U +
				                      static_cast<uint32_t>(coefficient) * 2246822519U + shard_id * 3266489917U;
				value = static_cast<int16_t>(hash & 0xffffU);
				break;
			}
			}
			column_values[row] = value;
		}
		columns[coefficient].name = "dct_zz_" + std::to_string(coefficient);
		columns[coefficient].data = std::span<const int16_t>(column_values);
	}

	const std::array<fastlanes::n_t, 1> rowgroup_n_tuples {static_cast<fastlanes::n_t>(table.row_count)};
	fastlanes::MemoryTableOptions options;
	options.n_vectors_per_rowgroup = rowgroup_vectors;
	options.rowgroup_n_tuples = std::span<const fastlanes::n_t>(rowgroup_n_tuples);
	if (force_expression) {
		options.force_schema = true;
		options.forced_schema.assign(columns.size(), target_token);
	}

	const auto staged_path = fls_path.string() + ".mixed.tmp";
	fastlanes::Connection connection;
	fastlanes::load_memory_table(
	    connection, fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn>(columns)}, options);
	connection.inline_footer();
	connection.to_fls(staged_path);
	galp::jpeg::detail::validate_jpeg_dct_fls_gpu_expressions(staged_path, shard_id);
	std::filesystem::rename(staged_path, fls_path);
}

TEST(JpegDct, PublicAggregatesPreserveLegacyPositionalInitialization) {
	galp::jpeg::JpegDctReaderOptions options {galp::jpeg::JpegComponentMode::kSingleComponent,
	                                          0,
	                                          galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor,
	                                          false,
	                                          true};
	EXPECT_FALSE(options.use_zigzag_columns);
	EXPECT_TRUE(options.use_z_curve_block_order);
	EXPECT_EQ(options.image_major_spatial_order, galp::jpeg::JpegDctSpatialOrder::kTiledZ32);

	galp::jpeg::JpegComponentMetadata component {0, 1, 2, 3, 4, 5, 6, 7, false};
	EXPECT_EQ(component.component_index, 0U);
	EXPECT_EQ(component.component_id, 1);
	EXPECT_FALSE(component.present);

	galp::jpeg::JpegImageMetadata image {std::filesystem::path("input.jpg"), 16, 8, 3, true, {component}};
	EXPECT_EQ(image.source_path, std::filesystem::path("input.jpg"));
	EXPECT_TRUE(image.progressive);
	ASSERT_EQ(image.components.size(), 1U);

	galp::jpeg::JpegDctDatasetMetadata metadata {{image},
	                                             galp::jpeg::JpegDctRowOrdering::kSingleImageComponentMajorBlockMajor,
	                                             galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor,
	                                             false,
	                                             true,
	                                             1};
	EXPECT_FALSE(metadata.zigzag_columns);
	EXPECT_TRUE(metadata.z_curve_block_order);
	EXPECT_EQ(metadata.image_count, 1U);

	galp::jpeg::JpegDctDeviceBatchOptions default_batch_options;
	EXPECT_EQ(default_batch_options.layout, galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff);
	EXPECT_FALSE(default_batch_options.grid_transform.has_value());
	EXPECT_EQ(default_batch_options.cache_capacity_bytes, 0U);
	EXPECT_EQ(default_batch_options.decode_batch_rowgroups, galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups);
	EXPECT_TRUE(default_batch_options.coefficient_selection.empty());
	EXPECT_EQ(default_batch_options.coefficient_selection.size(), galp::jpeg::detail::kJpegDctCoefficientCount);
	EXPECT_TRUE(default_batch_options.enable_rowgroup_prefetch);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_depth, galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_depth, 4U);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_workers,
	          galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_workers, 1U);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_min_decode_batches,
	          galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches);

	galp::jpeg::JpegDctDeviceBatchOptions legacy_batch_options {
	    galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff, std::nullopt, 4096U};
	EXPECT_EQ(legacy_batch_options.cache_capacity_bytes, 4096U);
	EXPECT_EQ(legacy_batch_options.decode_batch_rowgroups, galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups);
	EXPECT_TRUE(legacy_batch_options.enable_rowgroup_prefetch);

	galp::jpeg::JpegDctDeviceBatchOptions legacy_prefetch_options {
	    galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff,
	    std::nullopt,
	    4096U,
	    8U,
	    galp::jpeg::kDefaultJpegDctDevicePlanCacheCapacity,
	    false,
	    2U,
	    3U,
	    4U};
	EXPECT_EQ(legacy_prefetch_options.cache_capacity_bytes, 4096U);
	EXPECT_EQ(legacy_prefetch_options.decode_batch_rowgroups, 8U);
	EXPECT_FALSE(legacy_prefetch_options.enable_rowgroup_prefetch);
	EXPECT_EQ(legacy_prefetch_options.rowgroup_prefetch_depth, 2U);
	EXPECT_EQ(legacy_prefetch_options.rowgroup_prefetch_workers, 3U);
	EXPECT_EQ(legacy_prefetch_options.rowgroup_prefetch_min_decode_batches, 4U);
	EXPECT_TRUE(legacy_prefetch_options.coefficient_selection.empty());
}

TEST(JpegDct, SynchronousDeviceAccessPropagatesStoredCompletionFailure) {
	auto batch = galp::jpeg::detail::make_failed_device_batch_for_testing("controlled completion failure");

	EXPECT_EQ(batch.device_coefficients_async(), nullptr);
	EXPECT_EQ(batch.y_coefficients_async(), nullptr);
	EXPECT_EQ(batch.cbcr_coefficients_async(), nullptr);
	EXPECT_EQ(batch.y_float_coefficients_async(), nullptr);
	EXPECT_EQ(batch.cbcr_float_coefficients_async(), nullptr);
	EXPECT_NO_THROW(static_cast<void>(batch.execution_stats_ref()));

	const auto expect_controlled_failure = [](const auto& operation) {
		try {
			operation();
			FAIL() << "synchronous access unexpectedly ignored the completion failure";
		} catch (const std::runtime_error& error) { EXPECT_STREQ(error.what(), "controlled completion failure"); }
	};
	expect_controlled_failure([&] { static_cast<void>(batch.device_coefficients()); });
	expect_controlled_failure([&] { static_cast<void>(batch.y_coefficients()); });
	expect_controlled_failure([&] { static_cast<void>(batch.cbcr_coefficients()); });
	expect_controlled_failure([&] { static_cast<void>(batch.y_float_coefficients()); });
	expect_controlled_failure([&] { static_cast<void>(batch.cbcr_float_coefficients()); });
	expect_controlled_failure([&] { static_cast<void>(batch.execution_stats()); });
	expect_controlled_failure([&] { batch.synchronize(); });
}

TEST(JpegDct, SharedReaderSerializesDeviceSubmissionAcrossHostThreads) {
	int        device_count  = 0;
	const auto device_status = cudaGetDeviceCount(&device_count);
	if (device_status != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}
	int device = 0;
	ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_concurrent_reader_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 16, 16);
	write_test_jpeg(path1, 16, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	galp::jpeg::JpegDctDeviceBatchOptions  options;
	options.cache_capacity_bytes = 1U << 20U;

	std::promise<void> start_promise;
	auto               start = start_promise.get_future().share();
	const auto submit = [&](const uint32_t image_index) {
		return std::async(std::launch::async, [&, image_index] {
			if (cudaSetDevice(device) != cudaSuccess) {
				throw std::runtime_error("failed to select the CUDA device in the reader test worker");
			}
			start.wait();
			auto batch = reader.ReadDeviceDctBatch(
                {galp::jpeg::JpegDctImageCropRequest {image_index, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}}}, options);
			const auto* coefficients = batch.device_coefficients();
			if (coefficients == nullptr || batch.coefficient_count() == 0U) {
				throw std::runtime_error("concurrent reader submission produced an empty device batch");
			}
			return batch.coefficient_count();
		});
	};
	auto first  = submit(0U);
	auto second = submit(1U);
	start_promise.set_value();
	EXPECT_GT(first.get(), 0U);
	EXPECT_GT(second.get(), 0U);
	std::filesystem::remove_all(dir);
}

TEST(JpegDct, SpatialOrderRanksMatchWriterOrderOnRaggedTiles) {
	constexpr uint32_t width  = 67;
	constexpr uint32_t height = 35;
	const std::array<galp::jpeg::JpegDctSpatialOrder, 4> orders {
	    galp::jpeg::JpegDctSpatialOrder::kRaster,
	    galp::jpeg::JpegDctSpatialOrder::kTiledRaster32,
	    galp::jpeg::JpegDctSpatialOrder::kZOrder,
	    galp::jpeg::JpegDctSpatialOrder::kTiledZ32,
	};
	for (const auto spatial_order : orders) {
		const auto block_order = galp::jpeg::detail::make_block_order(width, height, spatial_order);
		ASSERT_EQ(block_order.size(), static_cast<size_t>(width) * height);
		std::vector<bool> seen(block_order.size(), false);
		for (size_t rank = 0; rank < block_order.size(); ++rank) {
			const auto& coord = block_order[rank];
			ASSERT_LT(coord.x, width);
			ASSERT_LT(coord.y, height);
			EXPECT_EQ(galp::jpeg::detail::block_order_rank(width, height, coord.x, coord.y, spatial_order), rank);
			const auto raster_index = static_cast<size_t>(coord.y) * width + coord.x;
			ASSERT_FALSE(seen[raster_index]);
			seen[raster_index] = true;
		}
		EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](const bool value) { return value; }));
	}
}

TEST(JpegDct, DctCoefficientSelectionParserAndNormalization) {
	galp::jpeg::JpegDctCoefficientSelection selection;

	ASSERT_TRUE(galp::jpeg::parse_jpeg_dct_coefficient_selection("all", selection));
	EXPECT_TRUE(selection.coefficients.empty());
	EXPECT_EQ(galp::jpeg::detail::normalize_coefficient_selection(selection), all_dct_coefficients());

	ASSERT_TRUE(galp::jpeg::parse_jpeg_dct_coefficient_selection("first:3", selection));
	EXPECT_EQ(selection.coefficients, (std::vector<uint8_t> {0U, 1U, 2U}));
	EXPECT_EQ(galp::jpeg::detail::normalize_coefficient_selection(selection), (std::vector<uint8_t> {0U, 1U, 2U}));

	ASSERT_TRUE(galp::jpeg::parse_jpeg_dct_coefficient_selection("list:5,0,2", selection));
	const std::vector<uint8_t> listed {5U, 0U, 2U};
	EXPECT_EQ(selection.coefficients, listed);
	EXPECT_EQ(galp::jpeg::detail::normalize_coefficient_selection(selection), listed);

	EXPECT_FALSE(galp::jpeg::parse_jpeg_dct_coefficient_selection("first:0", selection));
	EXPECT_FALSE(galp::jpeg::parse_jpeg_dct_coefficient_selection("first:65", selection));
	EXPECT_FALSE(galp::jpeg::parse_jpeg_dct_coefficient_selection("list:", selection));
	EXPECT_FALSE(galp::jpeg::parse_jpeg_dct_coefficient_selection("list:1,1", selection));
	EXPECT_FALSE(galp::jpeg::parse_jpeg_dct_coefficient_selection("list:64", selection));
	EXPECT_FALSE(galp::jpeg::parse_jpeg_dct_coefficient_selection("list:1,,2", selection));
	EXPECT_FALSE(galp::jpeg::parse_jpeg_dct_coefficient_selection("0,1,2", selection));
	EXPECT_EQ(selection.coefficients, listed);

	galp::jpeg::JpegDctCoefficientSelection duplicate_selection;
	duplicate_selection.coefficients = {1U, 1U};
	EXPECT_THROW((void)galp::jpeg::detail::normalize_coefficient_selection(duplicate_selection), std::invalid_argument);
	galp::jpeg::JpegDctCoefficientSelection out_of_range_selection;
	out_of_range_selection.coefficients = {64U};
	EXPECT_THROW((void)galp::jpeg::detail::normalize_coefficient_selection(out_of_range_selection), std::out_of_range);
}

TEST(JpegDct, ClassifiesNormalizedDctCoefficientSelectionShapes) {
	const auto all = galp::jpeg::detail::normalize_coefficient_selection(galp::jpeg::JpegDctCoefficientSelection {});
	const auto all_shape = galp::jpeg::detail::classify_coefficient_selection(all);
	EXPECT_EQ(all_shape.kind, galp::jpeg::detail::JpegDctCoefficientSelectionKind::kAll);
	EXPECT_EQ(all_shape.count, galp::jpeg::detail::kJpegDctCoefficientCount);
	EXPECT_TRUE(all_shape.is_contiguous_prefix());

	galp::jpeg::JpegDctCoefficientSelection prefix_selection;
	prefix_selection.coefficients = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
	const auto prefix             = galp::jpeg::detail::normalize_coefficient_selection(prefix_selection);
	const auto prefix_shape       = galp::jpeg::detail::classify_coefficient_selection(prefix);
	EXPECT_EQ(prefix_shape.kind, galp::jpeg::detail::JpegDctCoefficientSelectionKind::kPrefix);
	EXPECT_EQ(prefix_shape.count, 8U);
	EXPECT_TRUE(prefix_shape.is_contiguous_prefix());

	galp::jpeg::JpegDctCoefficientSelection list_selection;
	list_selection.coefficients = {0U, 2U, 5U};
	const auto list             = galp::jpeg::detail::normalize_coefficient_selection(list_selection);
	const auto list_shape       = galp::jpeg::detail::classify_coefficient_selection(list);
	EXPECT_EQ(list_shape.kind, galp::jpeg::detail::JpegDctCoefficientSelectionKind::kList);
	EXPECT_EQ(list_shape.count, 3U);
	EXPECT_FALSE(list_shape.is_contiguous_prefix());
}

TEST(JpegDct, DctGatherLayoutCompressesCoefficientAxisBySelectedSlotOrder) {
	const std::vector<uint8_t> selected_coefficients {5U, 0U, 2U};
	const size_t               coefficients_per_block = selected_coefficients.size();
	const size_t               block_count            = 2;

	std::vector<int16_t> full(block_count * galp::jpeg::detail::kJpegDctCoefficientCount);
	for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
		for (size_t coeff_idx = 0; coeff_idx < galp::jpeg::detail::kJpegDctCoefficientCount; ++coeff_idx) {
			full[block_idx * galp::jpeg::detail::kJpegDctCoefficientCount + coeff_idx] =
			    static_cast<int16_t>(block_idx * 100U + coeff_idx);
		}
	}

	std::vector<int16_t> gathered(block_count * coefficients_per_block);
	for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
		for (size_t coeff_slot = 0; coeff_slot < coefficients_per_block; ++coeff_slot) {
			const auto coeff_idx = selected_coefficients[coeff_slot];
			gathered[galp::jpeg::detail::selected_dct_output_offset(block_idx, coeff_slot, coefficients_per_block)] =
			    full[block_idx * galp::jpeg::detail::kJpegDctCoefficientCount + coeff_idx];
		}
	}

	EXPECT_EQ(galp::jpeg::detail::selected_dct_binding_offset(0, 0, coefficients_per_block), 0U);
	EXPECT_EQ(galp::jpeg::detail::selected_dct_binding_offset(1, 2, coefficients_per_block), 5U);
	EXPECT_EQ(galp::jpeg::detail::selected_dct_output_offset(0, 2, coefficients_per_block), 2U);
	EXPECT_EQ(galp::jpeg::detail::selected_dct_output_offset(1, 0, coefficients_per_block), 3U);
	EXPECT_EQ(gathered, (std::vector<int16_t> {5, 0, 2, 105, 100, 102}));
}

TEST(JpegDct, DatasetReadPreservesImageMetadata) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto path =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_metadata_test_" + std::to_string(suffix) + ".jpg");
	write_test_jpeg(path);

	const auto file_table    = galp::jpeg::read_jpeg_dct_file(path);
	const auto dataset_table = galp::jpeg::read_jpeg_dct_dataset({path});

	ASSERT_EQ(file_table.metadata.images.size(), 1);
	ASSERT_EQ(dataset_table.metadata.images.size(), 1);
	const auto& expected = file_table.metadata.images.front();
	const auto& actual   = dataset_table.metadata.images.front();
	EXPECT_EQ(actual.source_path, expected.source_path);
	EXPECT_EQ(actual.image_width, expected.image_width);
	EXPECT_EQ(actual.image_height, expected.image_height);
	EXPECT_EQ(actual.jpeg_color_space, expected.jpeg_color_space);
	EXPECT_EQ(actual.progressive, expected.progressive);
	EXPECT_EQ(actual.data_precision, expected.data_precision);
	EXPECT_EQ(actual.quant_tables.size(), expected.quant_tables.size());
	ASSERT_EQ(actual.components.size(), expected.components.size());

	std::filesystem::remove(path);
}

TEST(JpegDct, MetadataPersistsLayoutFlags) {
	const auto suffix       = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto default_path = std::filesystem::temp_directory_path() /
	                          ("galp_jpeg_dct_default_layout_" + std::to_string(suffix) + ".metadata.bin");
	const auto natural_path = std::filesystem::temp_directory_path() /
	                          ("galp_jpeg_dct_natural_layout_" + std::to_string(suffix) + ".metadata.bin");

	galp::jpeg::JpegDctDatasetMetadata metadata;
	metadata.image_count         = 0;
	metadata.zigzag_columns      = true;
	metadata.z_curve_block_order = true;
	galp::jpeg::write_jpeg_dct_metadata(metadata, default_path);
	const std::array<uint8_t, 8> legacy_magic {'G', 'J', 'D', 'C', 'T', 'M', 'D', '1'};
	auto                         default_header = read_metadata_header(default_path);
	EXPECT_EQ(default_header.magic, legacy_magic);
	EXPECT_EQ(default_header.layout_flags, 3);

	metadata.zigzag_columns      = false;
	metadata.z_curve_block_order = false;
	galp::jpeg::write_jpeg_dct_metadata(metadata, natural_path);
	EXPECT_EQ(read_metadata_header(natural_path).layout_flags, 0);

	std::filesystem::remove(default_path);
	std::filesystem::remove(natural_path);
}

TEST(JpegDct, ExplicitMetadataWriterOptionsUseSectionedFormat) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto path   = std::filesystem::temp_directory_path() /
	                  ("galp_jpeg_dct_sectioned_" + std::to_string(suffix) + ".metadata.bin");

	galp::jpeg::JpegDctDatasetMetadata metadata;
	metadata.image_count = 0;
	galp::jpeg::JpegDctMetadataWriterOptions options;
	options.profile = galp::jpeg::JpegMetadataProfile::kReconstructableJpeg;
	galp::jpeg::write_jpeg_dct_metadata(metadata, path, options);

	const std::array<uint8_t, 8> sectioned_magic {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'};
	auto                         header = read_metadata_header(path);
	EXPECT_EQ(header.magic, sectioned_magic);
	EXPECT_EQ(header.profile, 1);

	std::filesystem::remove(path);
}

TEST(JpegDct, ShardedDatasetWritesManifestAndShardFiles) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir    = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_" + std::to_string(suffix));
	const auto path0  = dir / "input0.jpg";
	const auto path1  = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0);
	write_test_jpeg(path1);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;

	const auto output_dir = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	ASSERT_EQ(manifest.shards.size(), 2U);
	EXPECT_EQ(manifest.shards[0].first_global_image_index, 0U);
	EXPECT_EQ(manifest.shards[1].first_global_image_index, 1U);
	EXPECT_EQ(manifest.shards[0].image_count, 1U);
	EXPECT_EQ(manifest.shards[1].image_count, 1U);
	EXPECT_GT(manifest.shards[0].rowgroup_count, 0U);
	EXPECT_TRUE(std::filesystem::exists(output_dir / "manifest.bin"));
	const auto generation_dir = std::filesystem::path(manifest.shards.front().fls_file_name).parent_path();
	ASSERT_FALSE(generation_dir.empty());
	for (const auto& shard : manifest.shards) {
		EXPECT_EQ(std::filesystem::path(shard.fls_file_name).parent_path(), generation_dir);
		EXPECT_EQ(std::filesystem::path(shard.metadata_file_name).parent_path(), generation_dir);
		EXPECT_TRUE(std::filesystem::exists(output_dir / shard.fls_file_name));
		EXPECT_TRUE(std::filesystem::exists(output_dir / shard.metadata_file_name));
	}
	for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
		EXPECT_NE(entry.path().extension(), ".tmp");
	}

	const std::array<uint8_t, 8> manifest_magic {'G', 'J', 'D', 'C', 'T', 'S', 'H', '1'};
	const auto                   header = read_shard_manifest_header(output_dir / "manifest.bin");
	EXPECT_EQ(header.magic, manifest_magic);
	EXPECT_EQ(header.version, 1U);
	EXPECT_EQ(header.policy, 2U);
	EXPECT_EQ(header.rowgroup_vectors, 1U);
	EXPECT_EQ(header.rowgroups_per_shard, 256U);
	EXPECT_EQ(header.image_count, 2U);
	EXPECT_EQ(header.shard_count, 2U);

	const std::array<uint8_t, 8> sectioned_magic {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'};
	EXPECT_EQ(read_metadata_header(output_dir / manifest.shards.front().metadata_file_name).magic, sectioned_magic);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            row_ref = reader.LocateRow(0, 0, 0, 0);
	EXPECT_TRUE(row_ref.present);
	EXPECT_EQ(row_ref.shard_id, 0U);
	const auto block_group = reader.ReadBlockGroup(row_ref.shard_id, 0, 0, 0);
	EXPECT_GT(block_group.rows.size(), 0U);
	const auto materialized = reader.MaterializeImageDct(0);
	EXPECT_GT(materialized.blocks.size(), 0U);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ShardedDatasetReplacementKeepsPreviousGenerationReadable) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_generation_replace_" + std::to_string(suffix));
	const auto old_path = dir / "old.jpg";
	const auto new_path = dir / "new.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(old_path, 16, 16);
	write_test_jpeg(new_path, 48, 32);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";

	const auto old_manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({old_path}, output_dir, reader_options, shard_options);
	ASSERT_EQ(old_manifest.shards.size(), 1U);
	galp::jpeg::JpegDctShardDatasetReader old_oracle_reader(output_dir / "manifest.bin");
	const auto old_metadata = old_oracle_reader.ImageMetadata(0);
	const auto old_oracle   = old_oracle_reader.MaterializeImageDct(0);
	// Construct this reader before replacement but defer opening its FLS shard
	// until after the new manifest has committed.
	galp::jpeg::JpegDctShardDatasetReader delayed_old_reader(output_dir / "manifest.bin");

	const auto new_manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({new_path}, output_dir, reader_options, shard_options);
	ASSERT_EQ(new_manifest.shards.size(), 1U);
	EXPECT_NE(old_manifest.shards.front().fls_file_name, new_manifest.shards.front().fls_file_name);
	EXPECT_NE(old_manifest.shards.front().metadata_file_name, new_manifest.shards.front().metadata_file_name);
	EXPECT_TRUE(std::filesystem::exists(output_dir / old_manifest.shards.front().fls_file_name));
	EXPECT_TRUE(std::filesystem::exists(output_dir / old_manifest.shards.front().metadata_file_name));
	EXPECT_TRUE(std::filesystem::exists(output_dir / new_manifest.shards.front().fls_file_name));
	EXPECT_TRUE(std::filesystem::exists(output_dir / new_manifest.shards.front().metadata_file_name));

	const auto delayed_old = delayed_old_reader.MaterializeImageDct(0);
	const auto delayed_old_metadata = delayed_old_reader.ImageMetadata(0);
	EXPECT_EQ(delayed_old_metadata.image_width, old_metadata.image_width);
	EXPECT_EQ(delayed_old_metadata.image_height, old_metadata.image_height);
	ASSERT_EQ(delayed_old.blocks.size(), old_oracle.blocks.size());
	for (size_t block = 0; block < delayed_old.blocks.size(); ++block) {
		EXPECT_EQ(delayed_old.blocks[block].semantic_slot_id, old_oracle.blocks[block].semantic_slot_id);
		EXPECT_EQ(delayed_old.blocks[block].block_x, old_oracle.blocks[block].block_x);
		EXPECT_EQ(delayed_old.blocks[block].block_y, old_oracle.blocks[block].block_y);
		EXPECT_EQ(delayed_old.blocks[block].coefficients, old_oracle.blocks[block].coefficients);
	}

	galp::jpeg::JpegDctShardDatasetReader new_reader(output_dir / "manifest.bin");
	const auto new_metadata = new_reader.ImageMetadata(0);
	const auto new_image    = new_reader.MaterializeImageDct(0);
	EXPECT_EQ(new_metadata.image_width, 48U);
	EXPECT_EQ(new_metadata.image_height, 32U);
	EXPECT_NE(new_image.blocks.size(), old_oracle.blocks.size());

	for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
		EXPECT_NE(entry.path().extension(), ".tmp");
	}
	std::filesystem::remove_all(dir);
}

TEST(JpegDct, StagedExpressionValidationRejectsUnsupportedTokenWithFullLocation) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_unsupported_expression_" + std::to_string(suffix));
	std::filesystem::create_directories(dir);
	const auto fls_path = dir / "staged.fls.tmp";

	std::vector<int16_t> values(2U * 1024U);
	for (size_t row = 0; row < values.size(); ++row) {
		values[row] = static_cast<int16_t>(row % 257U);
	}
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t coefficient = 0; coefficient < columns.size(); ++coefficient) {
		columns[coefficient].name = "dct_zz_" + std::to_string(coefficient);
		columns[coefficient].data = std::span<const int16_t>(values);
	}
	fastlanes::MemoryTableOptions options;
	options.n_vectors_per_rowgroup = 2;
	options.force_schema           = true;
	options.forced_schema.assign(columns.size(), fastlanes::OperatorToken::EXP_FFOR_I16);
	options.forced_schema.front() = fastlanes::OperatorToken::EXP_NULL_I16;
	fastlanes::Connection connection;
	fastlanes::load_memory_table(
	    connection, fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn>(columns)}, options);
	connection.inline_footer();
	connection.to_fls(fls_path);

	try {
		galp::jpeg::detail::validate_jpeg_dct_fls_gpu_expressions(fls_path, 7U);
		FAIL() << "unsupported staged expression was accepted";
	} catch (const std::runtime_error& error) {
		const std::string message = error.what();
		EXPECT_NE(message.find("shard=7"), std::string::npos);
		EXPECT_NE(message.find("rowgroup=0"), std::string::npos);
		EXPECT_NE(message.find("column=0"), std::string::npos);
		EXPECT_NE(message.find("coefficient=0"), std::string::npos);
		EXPECT_NE(message.find("token=EXP_NULL_I16"), std::string::npos);
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ShardedReconstructableMetadataPreservesQuantTables) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_reconstruct_" + std::to_string(suffix));
	const auto path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path, 32, 24);

	const auto file_table = galp::jpeg::read_jpeg_dct_file(path);
	ASSERT_EQ(file_table.metadata.images.size(), 1U);
	const auto& expected = file_table.metadata.images.front();
	ASSERT_FALSE(expected.quant_tables.empty());
	ASSERT_FALSE(expected.components.empty());

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	galp::jpeg::JpegDctMetadataWriterOptions metadata_options;
	metadata_options.profile = galp::jpeg::JpegMetadataProfile::kReconstructableJpeg;

	const auto output_dir = dir / "out";
	(void)galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    {path}, output_dir, reader_options, shard_options, metadata_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            actual = reader.ImageMetadata(0);
	EXPECT_EQ(actual.image_width, expected.image_width);
	EXPECT_EQ(actual.image_height, expected.image_height);
	EXPECT_EQ(actual.quant_tables.size(), expected.quant_tables.size());
	ASSERT_EQ(actual.components.size(), expected.components.size());
	ASSERT_FALSE(actual.quant_tables.empty());

	for (size_t i = 0; i < actual.components.size(); ++i) {
		const auto& component          = actual.components[i];
		const auto& expected_component = expected.components[i];
		EXPECT_EQ(component.component_id, expected_component.component_id);
		EXPECT_EQ(component.h_samp_factor, expected_component.h_samp_factor);
		EXPECT_EQ(component.v_samp_factor, expected_component.v_samp_factor);
		EXPECT_EQ(component.quant_tbl_no, expected_component.quant_tbl_no);
		EXPECT_NE(component.quant_table_fingerprint, 0U);
		const auto table = std::find_if(actual.quant_tables.begin(),
		                                actual.quant_tables.end(),
		                                [&](const galp::jpeg::JpegQuantTableMetadata& candidate) {
			                                return candidate.table_id == component.quant_tbl_no;
		                                });
		ASSERT_NE(table, actual.quant_tables.end());
		EXPECT_NE(
		    std::find_if(table->values.begin(), table->values.end(), [](const uint16_t value) { return value != 0; }),
		          table->values.end());
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchReadsCropIntoImageMajorDctBlocks) {
	int        device_count  = 0;
	const auto device_status = cudaGetDeviceCount(&device_count);
	if (device_status != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_device_batch_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 16, 16);
	write_test_jpeg(path1, 16, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader                  reader(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}},
	    galp::jpeg::JpegDctImageCropRequest {1, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}},
	};
	auto batch = reader.ReadDeviceDctBatch(requests);

	ASSERT_EQ(batch.layout(), galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff);
	ASSERT_EQ(batch.image_count(), requests.size());
	ASSERT_NE(batch.device_coefficients(), nullptr);
	ASSERT_EQ(batch.coefficients_per_block(), galp::jpeg::detail::kJpegDctCoefficientCount);
	ASSERT_EQ(batch.selected_coefficients(), all_dct_coefficients());
	ASSERT_EQ(batch.coefficient_count(), batch.block_count() * batch.coefficients_per_block());
	ASSERT_EQ(batch.image_layouts().size(), requests.size());
	ASSERT_EQ(batch.block_metadata().size(), batch.block_count());
	ASSERT_EQ(batch.rowgroups().size(), batch.rowgroup_count());
	EXPECT_GT(batch.rowgroup_count(), 0U);
	EXPECT_EQ(batch.image_layouts()[0].block_offset, 0U);
	EXPECT_GT(batch.image_layouts()[0].block_count, 0U);
	EXPECT_EQ(batch.image_layouts()[1].block_offset, batch.image_layouts()[0].block_count);
	EXPECT_GT(batch.image_layouts()[1].block_count, 0U);

	const auto expect_batch_matches_materialized = [&](const galp::jpeg::JpegDctDeviceBatch& batch_to_check) {
		ASSERT_EQ(batch_to_check.coefficient_count(),
		          batch_to_check.block_count() * batch_to_check.coefficients_per_block());
		ASSERT_EQ(batch_to_check.selected_coefficients().size(), batch_to_check.coefficients_per_block());
		ASSERT_NE(batch_to_check.device_coefficients(), nullptr);

		std::vector<int16_t> host(batch_to_check.coefficient_count());
		ASSERT_EQ(cudaMemcpy(host.data(),
		                     batch_to_check.device_coefficients(),
		                     batch_to_check.coefficient_bytes(),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);

		for (size_t output_block_idx = 0; output_block_idx < batch_to_check.block_metadata().size();
		     ++output_block_idx) {
			const auto& meta         = batch_to_check.block_metadata()[output_block_idx];
			const auto  materialized = reader.MaterializeImageDct(meta.global_image_index);
			const auto* expected     = static_cast<const galp::jpeg::MaterializedJpegDctBlock*>(nullptr);
			for (const auto& block : materialized.blocks) {
				if (block.semantic_slot_id == meta.semantic_slot_id && block.block_x == meta.block_x &&
				    block.block_y == meta.block_y) {
					expected = &block;
					break;
				}
			}
			ASSERT_NE(expected, nullptr);
			for (size_t coeff_slot = 0; coeff_slot < batch_to_check.coefficients_per_block(); ++coeff_slot) {
				const auto coeff_idx = batch_to_check.selected_coefficients()[coeff_slot];
				EXPECT_EQ(host[output_block_idx * batch_to_check.coefficients_per_block() + coeff_slot],
				          expected->coefficients[coeff_idx]);
			}
		}
		};

		expect_batch_matches_materialized(batch);
		const auto dense_stats = batch.execution_stats();
		EXPECT_GT(dense_stats.decoded_gather_item_count, 0U);
		EXPECT_EQ(dense_stats.decoded_projection_item_count, 0U);

		galp::jpeg::JpegDctDeviceBatchOptions selected_options;
		const std::vector<uint8_t>            selected_coefficients {0U, 2U, 5U};
	selected_options.coefficient_selection.coefficients = selected_coefficients;

	auto selected_crop_batch = reader.ReadDeviceDctBatch(requests, selected_options);
	ASSERT_EQ(selected_crop_batch.coefficients_per_block(), selected_coefficients.size());
	ASSERT_EQ(selected_crop_batch.selected_coefficients(), selected_coefficients);
	ASSERT_EQ(selected_crop_batch.coefficient_count(),
	          selected_crop_batch.block_count() * selected_coefficients.size());
	expect_batch_matches_materialized(selected_crop_batch);
	const auto selected_crop_stats = selected_crop_batch.execution_stats();
	EXPECT_EQ(selected_crop_stats.decoded_gather_item_count, 0U);
	EXPECT_EQ(selected_crop_stats.gather_kernel_launch_count, 0U);
	EXPECT_EQ(selected_crop_stats.decoded_projection_item_count,
	          selected_crop_batch.block_count() * selected_coefficients.size());
	EXPECT_GT(selected_crop_stats.materialize_kernel_launch_count, 0U);

	const std::vector<galp::jpeg::JpegDctImageCropRequest> full_image_requests {
	    galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {}},
	};
	auto selected_full_batch = reader.ReadDeviceDctBatch(full_image_requests, selected_options);
	ASSERT_EQ(selected_full_batch.image_count(), 1U);
	ASSERT_EQ(selected_full_batch.coefficients_per_block(), selected_coefficients.size());
	ASSERT_EQ(selected_full_batch.selected_coefficients(), selected_coefficients);
	ASSERT_EQ(selected_full_batch.coefficient_count(),
	          selected_full_batch.block_count() * selected_coefficients.size());
	expect_batch_matches_materialized(selected_full_batch);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchReadsTransformedGridWithRgbNoMoreProfile) {
	int        device_count  = 0;
	const auto device_status = cudaGetDeviceCount(&device_count);
	if (device_status != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_transformed_grid_device_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 32, 32);
	write_test_jpeg(path1, 64, 40);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout               = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform       = galp::profiles::rgbnomore_val_dct_grid_transform();
	options.cache_capacity_bytes = 1U << 20U;

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {}},
	    galp::jpeg::JpegDctImageCropRequest {1, galp::jpeg::JpegDctCropBox {}},
	};
	auto batch = reader.ReadDeviceDctBatch(requests, options);

	EXPECT_EQ(batch.layout(), galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid);
	EXPECT_EQ(batch.image_count(), requests.size());
	EXPECT_EQ(batch.selected_coefficients(), all_dct_coefficients());
	EXPECT_EQ(batch.coefficients_per_block(), galp::jpeg::detail::kJpegDctCoefficientCount);
	EXPECT_EQ(batch.ycbcr_dct_grid_shape().y, (std::array<size_t, 6> {2U, 1U, 28U, 28U, 8U, 8U}));
	EXPECT_EQ(batch.ycbcr_dct_grid_shape().cbcr, (std::array<size_t, 6> {2U, 2U, 14U, 14U, 8U, 8U}));
	EXPECT_EQ(batch.y_coefficient_count(), batch.ycbcr_dct_grid_shape().y_count());
	EXPECT_EQ(batch.cbcr_coefficient_count(), batch.ycbcr_dct_grid_shape().cbcr_count());
	ASSERT_NE(batch.y_coefficients(), nullptr);
	ASSERT_NE(batch.cbcr_coefficients(), nullptr);

	std::vector<int16_t> y_host(batch.y_coefficient_count());
	std::vector<int16_t> cbcr_host(batch.cbcr_coefficient_count());
	ASSERT_EQ(
	    cudaMemcpy(y_host.data(), batch.y_coefficients(), y_host.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	ASSERT_EQ(
	    cudaMemcpy(
	        cbcr_host.data(), batch.cbcr_coefficients(), cbcr_host.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	const auto stats = batch.execution_stats();
	EXPECT_EQ(stats.plan_cache_hits, 0U);
	EXPECT_EQ(stats.plan_cache_misses, 1U);
	EXPECT_EQ(stats.projection_item_count, 0U);
	EXPECT_EQ(stats.decoded_projection_item_count, 0U);
	EXPECT_GT(stats.fixed_transform_item_count, 0U);
	EXPECT_EQ(stats.fixed_transform_image_count, requests.size());
	EXPECT_EQ(stats.fixed_transform_component_count, requests.size() * 3U);
	EXPECT_EQ(stats.fixed_transform_source_block_count, batch.block_count());
	EXPECT_EQ(stats.fixed_transform_output_block_count,
	          (batch.y_coefficient_count() + batch.cbcr_coefficient_count()) / 64U);
	EXPECT_GT(stats.materialize_kernel_launch_count, 0U);
	EXPECT_GT(stats.fixed_grid_round_ms, 0.0);

	auto cached_plan_batch = reader.ReadDeviceDctBatch(requests, options);
	const auto cached_plan_stats = cached_plan_batch.execution_stats();
	EXPECT_EQ(cached_plan_stats.plan_cache_hits, 1U);
	EXPECT_EQ(cached_plan_stats.plan_cache_misses, 0U);
	EXPECT_GT(cached_plan_batch.cache_stats().hits, 0U);
	EXPECT_EQ(cached_plan_stats.fixed_transform_component_count, stats.fixed_transform_component_count);
	EXPECT_EQ(cached_plan_stats.fixed_transform_source_block_count, stats.fixed_transform_source_block_count);
	EXPECT_EQ(cached_plan_stats.fixed_transform_output_block_count, stats.fixed_transform_output_block_count);
	EXPECT_GT(cached_plan_stats.fixed_transform_ms, 0.0);
	EXPECT_EQ(cached_plan_stats.cached_gather_ms, 0.0);
	EXPECT_EQ(cached_plan_stats.resize_weight_build_ms, 0.0);
	EXPECT_EQ(cached_plan_stats.dct_resize_weight_cache_hits, 0U);
	EXPECT_EQ(cached_plan_stats.dct_resize_weight_cache_misses, 0U);
	EXPECT_EQ(cached_plan_stats.dct_conversion_matrix_cache_hits, 0U);
	EXPECT_EQ(cached_plan_stats.dct_conversion_matrix_cache_misses, 0U);

	{
		galp::jpeg::JpegDctShardDatasetReader cache_reader(output_dir / "manifest.bin");
		auto cache_options                 = options;
		cache_options.cache_capacity_bytes = 0;
		cache_options.plan_cache_capacity  = 3;
		const std::vector<galp::jpeg::JpegDctImageCropRequest> requests0 {
		    galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {}},
		};
		const std::vector<galp::jpeg::JpegDctImageCropRequest> requests1 {
		    galp::jpeg::JpegDctImageCropRequest {1, galp::jpeg::JpegDctCropBox {}},
		};
		EXPECT_EQ(cache_reader.ReadDeviceDctBatch(requests0, cache_options).execution_stats().plan_cache_misses, 1U);
		EXPECT_EQ(cache_reader.ReadDeviceDctBatch(requests1, cache_options).execution_stats().plan_cache_misses, 1U);
		EXPECT_EQ(cache_reader.ReadDeviceDctBatch(requests, cache_options).execution_stats().plan_cache_misses, 1U);

		cache_options.plan_cache_capacity = 1;
		const auto shrink_stats = cache_reader.ReadDeviceDctBatch(requests0, cache_options).execution_stats();
		EXPECT_EQ(shrink_stats.plan_cache_hits, 1U);
		EXPECT_EQ(shrink_stats.plan_cache_misses, 0U);
		EXPECT_EQ(shrink_stats.plan_cache_evictions, 2U);

		cache_options.plan_cache_capacity = 0;
		const auto disabled_stats = cache_reader.ReadDeviceDctBatch(requests0, cache_options).execution_stats();
		EXPECT_EQ(disabled_stats.plan_cache_misses, 1U);
		EXPECT_EQ(disabled_stats.plan_cache_evictions, 1U);

		cache_options.plan_cache_capacity = 1;
		const auto reenabled_stats = cache_reader.ReadDeviceDctBatch(requests0, cache_options).execution_stats();
		EXPECT_EQ(reenabled_stats.plan_cache_hits, 0U);
		EXPECT_EQ(reenabled_stats.plan_cache_misses, 1U);
	}

	// A legacy v1 spatial-major manifest still uses the expanded fixed-transform plan. With no decoded
	// cache, identical executions must use the plan-wide grouped reduction rather than unordered atomic
	// accumulation into the same output coefficients. Use a separate reader so this regression does not
	// alter the cache state checked above.
	{
		galp::jpeg::JpegDctShardDatasetReader deterministic_reader(output_dir / "manifest.bin");
		auto                                  deterministic_options = options;
		deterministic_options.cache_capacity_bytes                  = 0U;
		deterministic_options.plan_cache_capacity                   = 0U;
		deterministic_options.decode_batch_rowgroups                = 1U;
		deterministic_options.enable_rowgroup_prefetch              = false;
		const auto copy_transformed_grid                            = [](const galp::jpeg::JpegDctDeviceBatch& source) {
            std::pair<std::vector<int16_t>, std::vector<int16_t>> host {
                std::vector<int16_t>(source.y_coefficient_count()),
                std::vector<int16_t>(source.cbcr_coefficient_count()),
            };
            EXPECT_EQ(cudaMemcpy(host.first.data(),
                                 source.y_coefficients(),
                                 host.first.size() * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(cudaMemcpy(host.second.data(),
                                 source.cbcr_coefficients(),
                                 host.second.size() * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            return host;
		};
		const auto deterministic_batch = deterministic_reader.ReadDeviceDctBatch(requests, deterministic_options);
		const auto deterministic_grid  = copy_transformed_grid(deterministic_batch);
		const auto repeated_batch      = deterministic_reader.ReadDeviceDctBatch(requests, deterministic_options);
		const auto repeated_grid       = copy_transformed_grid(repeated_batch);
		EXPECT_EQ(deterministic_grid, repeated_grid);
	}

	auto empty_batch = reader.ReadDeviceDctBatch({}, options);
	EXPECT_EQ(empty_batch.layout(), galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid);
	EXPECT_EQ(empty_batch.image_count(), 0U);
	EXPECT_EQ(empty_batch.y_coefficient_count(), 0U);
	EXPECT_EQ(empty_batch.cbcr_coefficient_count(), 0U);
	EXPECT_EQ(empty_batch.y_coefficients(), nullptr);
	EXPECT_EQ(empty_batch.cbcr_coefficients(), nullptr);
	EXPECT_EQ(empty_batch.ycbcr_dct_grid_shape().y, (std::array<size_t, 6> {0U, 1U, 28U, 28U, 8U, 8U}));
	EXPECT_EQ(empty_batch.ycbcr_dct_grid_shape().cbcr, (std::array<size_t, 6> {0U, 2U, 14U, 14U, 8U, 8U}));

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, TransformedGridHonorsExplicitPerImageTrainingCrops) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_training_crop_plan_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 64, 64);
	write_test_jpeg(path1, 64, 64);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout         = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform = galp::profiles::rgbnomore_val_dct_grid_transform();
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    {0U, {0U, 0U, 32U, 32U}, false, "train/zero.jpg", "augmentation-zero"},
	    {1U, {32U, 32U, 32U, 32U}, true, "train/one.jpg", "augmentation-one"},
	};
	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto preview = reader.PlanDeviceDctBatch(requests, options);
	bool       first_has_top_left_y     = false;
	bool       second_has_bottom_right_y = false;
	for (const auto& block : preview.block_metadata) {
		if (block.request_index == 0U && block.semantic_slot_id == 0U && block.block_x == 0U && block.block_y == 0U) {
			first_has_top_left_y = true;
		}
		if (block.request_index == 1U && block.semantic_slot_id == 0U && block.block_x == 7U && block.block_y == 7U) {
			second_has_bottom_right_y = true;
		}
	}
	EXPECT_TRUE(first_has_top_left_y);
	EXPECT_TRUE(second_has_bottom_right_y);
	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DirectDctTrainingFlipMatchesDctRuleAndPreservesProvenance) {
	int        device_count  = 0;
	const auto device_status = cudaGetDeviceCount(&device_count);
	if (device_status != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_training_flip_" + std::to_string(suffix));
	const auto path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path, 64, 64);
	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout                 = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform         = galp::profiles::rgbnomore_val_dct_grid_transform();
	options.plan_cache_capacity    = 0;
	options.enable_planless_execution = true;
	galp::jpeg::DirectDctRuntime runtime(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> plain_request {
	    {0U, {0U, 0U, 64U, 64U}, false, "train/sample.jpg", "plain-key"},
	};
	const std::vector<galp::jpeg::JpegDctImageCropRequest> flipped_request {
	    {0U, {0U, 0U, 64U, 64U}, true, "train/sample.jpg", "flipped-key"},
	};
	auto plain   = runtime.ReadBatch(plain_request, options);
	auto flipped = runtime.ReadBatch(flipped_request, options);
	ASSERT_EQ(flipped.transform_requests().size(), 1U);
	EXPECT_TRUE(flipped.transform_requests()[0].horizontal_flip);
	EXPECT_EQ(flipped.transform_requests()[0].logical_sample_id, "train/sample.jpg");
	EXPECT_EQ(flipped.transform_requests()[0].augmentation_key, "flipped-key");

	std::vector<int16_t> plain_y(plain.y_coefficient_count());
	std::vector<int16_t> flipped_y(flipped.y_coefficient_count());
	std::vector<int16_t> plain_cbcr(plain.cbcr_coefficient_count());
	std::vector<int16_t> flipped_cbcr(flipped.cbcr_coefficient_count());
	ASSERT_EQ(
	    cudaMemcpy(plain_y.data(), plain.y_device_data(), plain_y.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	ASSERT_EQ(
	    cudaMemcpy(
	        flipped_y.data(), flipped.y_device_data(), flipped_y.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	ASSERT_EQ(
	    cudaMemcpy(
	        plain_cbcr.data(), plain.cbcr_device_data(), plain_cbcr.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	ASSERT_EQ(cudaMemcpy(flipped_cbcr.data(),
	                     flipped.cbcr_device_data(),
	                     flipped_cbcr.size() * sizeof(int16_t),
	                     cudaMemcpyDeviceToHost),
	          cudaSuccess);
	const auto check_flip = [](const std::vector<int16_t>& source,
	                           const std::vector<int16_t>& actual,
	                           const uint32_t channels,
	                           const uint32_t height,
	                           const uint32_t width) {
		for (uint32_t channel = 0; channel < channels; ++channel) {
			for (uint32_t y = 0; y < height; ++y) {
				for (uint32_t x = 0; x < width; ++x) {
					for (uint32_t coefficient = 0; coefficient < 64U; ++coefficient) {
						const auto source_index =
						    ((static_cast<size_t>(channel) * height + y) * width + (width - 1U - x)) * 64U +
						    coefficient;
						const auto actual_index =
						    ((static_cast<size_t>(channel) * height + y) * width + x) * 64U + coefficient;
						const auto sign = coefficient % 8U % 2U == 0U ? 1 : -1;
						const auto expected = static_cast<int16_t>(sign * source[source_index]);
						if (actual[actual_index] != expected) {
							ADD_FAILURE() << "DCT horizontal flip mismatch at channel=" << channel << ", y=" << y
							              << ", x=" << x << ", coefficient=" << coefficient << ": expected " << expected
							              << ", got " << actual[actual_index];
							return;
						}
					}
				}
			}
		}
	};
	check_flip(plain_y, flipped_y, 1U, 28U, 28U);
	check_flip(plain_cbcr, flipped_cbcr, 2U, 14U, 14U);
	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DirectDctRuntimeExposesStayOnGpuTensorDescriptor) {
	int        device_count  = 0;
	const auto device_status = cudaGetDeviceCount(&device_count);
	if (device_status != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir    = std::filesystem::temp_directory_path() / ("galp_direct_dct_runtime_" + std::to_string(suffix));
	const auto path0  = dir / "input0.jpg";
	const auto path1  = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 16, 16);
	write_test_jpeg(path1, 16, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	const std::vector<uint32_t> image_ids {0U, 1U};
	const std::vector<uint8_t>  selected_coefficients {5U, 0U, 2U};
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.coefficient_selection.coefficients = selected_coefficients;

	galp::jpeg::DirectDctRuntime runtime(output_dir / "manifest.bin");
	auto                         batch = runtime.ReadBatch(image_ids, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}, options);

	EXPECT_EQ(runtime.image_count(), 2U);
	EXPECT_EQ(batch.global_image_ids(), image_ids);
	EXPECT_EQ(batch.image_count(), image_ids.size());
	EXPECT_EQ(batch.selected_coefficients(), selected_coefficients);
	EXPECT_EQ(batch.coefficients_per_block(), selected_coefficients.size());
	EXPECT_EQ(batch.coefficient_count(), batch.block_count() * selected_coefficients.size());
	ASSERT_EQ(batch.image_layouts().size(), image_ids.size());
	ASSERT_EQ(batch.block_metadata().size(), batch.block_count());
	EXPECT_GT(batch.rowgroups().size(), 0U);

	const auto descriptor = batch.tensor();
	EXPECT_EQ(descriptor.data, batch.device_data());
	EXPECT_EQ(descriptor.shape[0], batch.block_count());
	EXPECT_EQ(descriptor.shape[1], selected_coefficients.size());
	EXPECT_EQ(descriptor.strides[0], selected_coefficients.size());
	EXPECT_EQ(descriptor.strides[1], 1U);
	EXPECT_EQ(descriptor.dtype, galp::jpeg::DirectDctTensorDataType::kInt16);
	EXPECT_EQ(descriptor.device, galp::jpeg::DirectDctTensorDevice::kCuda);
	EXPECT_GE(descriptor.cuda_device, 0);
	ASSERT_NE(descriptor.data, nullptr);

	auto moved_batch       = std::move(batch);
	const auto moved_tensor = moved_batch.tensor();
	ASSERT_NE(moved_tensor.data, nullptr);
	EXPECT_EQ(moved_tensor.shape[0], moved_batch.block_count());
	EXPECT_EQ(moved_tensor.shape[1], moved_batch.coefficients_per_block());

	std::vector<int16_t> host(moved_batch.coefficient_count());
	ASSERT_EQ(
	    cudaMemcpy(host.data(), moved_batch.device_data(), moved_batch.coefficient_bytes(), cudaMemcpyDeviceToHost),
	          cudaSuccess);

	galp::jpeg::JpegDctShardDatasetReader reference_reader(output_dir / "manifest.bin");
	for (size_t output_block_idx = 0; output_block_idx < moved_batch.block_metadata().size(); ++output_block_idx) {
		const auto& meta         = moved_batch.block_metadata()[output_block_idx];
		const auto  materialized = reference_reader.MaterializeImageDct(meta.global_image_index);
		const auto* expected     = static_cast<const galp::jpeg::MaterializedJpegDctBlock*>(nullptr);
		for (const auto& block : materialized.blocks) {
			if (block.semantic_slot_id == meta.semantic_slot_id && block.block_x == meta.block_x &&
			    block.block_y == meta.block_y) {
				expected = &block;
				break;
			}
		}
		ASSERT_NE(expected, nullptr);
		for (size_t coeff_slot = 0; coeff_slot < moved_batch.coefficients_per_block(); ++coeff_slot) {
			const auto coeff_idx = moved_batch.selected_coefficients()[coeff_slot];
			EXPECT_EQ(host[output_block_idx * moved_batch.coefficients_per_block() + coeff_slot],
			          expected->coefficients[coeff_idx]);
		}
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DirectDctRuntimeExposesTransformedGridTensorDescriptors) {
	int        device_count  = 0;
	const auto device_status = cudaGetDeviceCount(&device_count);
	if (device_status != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_direct_dct_transformed_grid_runtime_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 32, 32);
	write_test_jpeg(path1, 64, 40);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout         = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform = galp::profiles::rgbnomore_val_dct_grid_transform();

	const std::vector<uint32_t> image_ids {0U, 1U};
	galp::jpeg::DirectDctRuntime runtime(output_dir / "manifest.bin");
	auto                         batch = runtime.ReadBatch(image_ids, galp::jpeg::JpegDctCropBox {}, options);

	EXPECT_EQ(batch.global_image_ids(), image_ids);
	EXPECT_EQ(batch.image_count(), image_ids.size());
	EXPECT_EQ(batch.device_batch().layout(), galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid);
	EXPECT_THROW((void)batch.tensor(), std::logic_error);

	const auto y = batch.y_tensor();
	const auto cbcr = batch.cbcr_tensor();
	EXPECT_EQ(y.data, batch.y_device_data());
	EXPECT_EQ(cbcr.data, batch.cbcr_device_data());
	EXPECT_EQ(y.shape, (std::array<size_t, 6> {2U, 1U, 28U, 28U, 8U, 8U}));
	EXPECT_EQ(cbcr.shape, (std::array<size_t, 6> {2U, 2U, 14U, 14U, 8U, 8U}));
	EXPECT_EQ(y.strides, (std::array<size_t, 6> {50176U, 50176U, 1792U, 64U, 8U, 1U}));
	EXPECT_EQ(cbcr.strides, (std::array<size_t, 6> {25088U, 12544U, 896U, 64U, 8U, 1U}));
	EXPECT_EQ(y.dtype, galp::jpeg::DirectDctTensorDataType::kInt16);
	EXPECT_EQ(cbcr.dtype, galp::jpeg::DirectDctTensorDataType::kInt16);
	EXPECT_EQ(y.device, galp::jpeg::DirectDctTensorDevice::kCuda);
	EXPECT_EQ(cbcr.device, galp::jpeg::DirectDctTensorDevice::kCuda);
	EXPECT_GE(y.cuda_device, 0);
	EXPECT_EQ(cbcr.cuda_device, y.cuda_device);
	EXPECT_EQ(y.element_count(), batch.y_coefficient_count());
	EXPECT_EQ(cbcr.element_count(), batch.cbcr_coefficient_count());
	ASSERT_NE(y.data, nullptr);
	ASSERT_NE(cbcr.data, nullptr);
	EXPECT_EQ(y.float_data, nullptr);
	EXPECT_EQ(cbcr.float_data, nullptr);

	auto float_options                    = options;
	auto float_transform                  = *float_options.grid_transform;
	float_transform.output_data_type      = galp::jpeg::JpegDctGridOutputDataType::kFloat32;
	float_transform.output_add            = 4.0F;
	float_transform.output_scale          = 1.0F / 1020.0F;
	float_options.grid_transform           = float_transform;
	auto float_batch = runtime.ReadBatch(image_ids, galp::jpeg::JpegDctCropBox {}, float_options);

	const auto float_y    = float_batch.y_tensor();
	const auto float_cbcr = float_batch.cbcr_tensor();
	EXPECT_EQ(float_y.data, nullptr);
	EXPECT_EQ(float_cbcr.data, nullptr);
	EXPECT_EQ(float_y.float_data, float_batch.y_float_device_data());
	EXPECT_EQ(float_cbcr.float_data, float_batch.cbcr_float_device_data());
	EXPECT_EQ(float_y.raw_data(), float_y.float_data);
	EXPECT_EQ(float_cbcr.raw_data(), float_cbcr.float_data);
	EXPECT_EQ(float_y.shape, y.shape);
	EXPECT_EQ(float_cbcr.shape, cbcr.shape);
	EXPECT_EQ(float_y.strides, y.strides);
	EXPECT_EQ(float_cbcr.strides, cbcr.strides);
	EXPECT_EQ(float_y.dtype, galp::jpeg::DirectDctTensorDataType::kFloat32);
	EXPECT_EQ(float_cbcr.dtype, galp::jpeg::DirectDctTensorDataType::kFloat32);
	ASSERT_NE(float_y.float_data, nullptr);
	ASSERT_NE(float_cbcr.float_data, nullptr);

	std::vector<int16_t> y_int16(y.element_count());
	std::vector<int16_t> cbcr_int16(cbcr.element_count());
	std::vector<float>   y_float(float_y.element_count());
	std::vector<float>   cbcr_float(float_cbcr.element_count());
	ASSERT_EQ(cudaMemcpy(y_int16.data(), y.data, y_int16.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	ASSERT_EQ(cudaMemcpy(cbcr_int16.data(), cbcr.data, cbcr_int16.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	ASSERT_EQ(cudaMemcpy(y_float.data(), float_y.float_data, y_float.size() * sizeof(float), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	ASSERT_EQ(
	    cudaMemcpy(cbcr_float.data(), float_cbcr.float_data, cbcr_float.size() * sizeof(float), cudaMemcpyDeviceToHost),
	          cudaSuccess);
	const auto expect_scaled = [](const std::vector<int16_t>& integer, const std::vector<float>& actual) {
		ASSERT_EQ(integer.size(), actual.size());
		for (size_t index = 0; index < integer.size(); ++index) {
			const float shifted = static_cast<float>(integer[index]) + 4.0F;
			const float expected = shifted * (1.0F / 1020.0F);
			EXPECT_EQ(actual[index], expected) << "coefficient index " << index;
		}
	};
	expect_scaled(y_int16, y_float);
	expect_scaled(cbcr_int16, cbcr_float);
	const auto float_stats = float_batch.execution_stats();
	EXPECT_EQ(float_stats.plan_cache_hits, 0U);
	EXPECT_EQ(float_stats.plan_cache_misses, 1U);
	EXPECT_TRUE(float_stats.fixed_grid_output_float32);
	EXPECT_TRUE(float_stats.fixed_grid_output_affine_applied);
	EXPECT_EQ(float_stats.fixed_grid_finalize_kernel_launch_count, 1U);
	EXPECT_FLOAT_EQ(float_stats.fixed_grid_output_add, 4.0F);
	EXPECT_FLOAT_EQ(float_stats.fixed_grid_output_scale, 1.0F / 1020.0F);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, PipelineCompareValidatesDctCoefficientSelectionForCropAndFullImage) {
	int        device_count  = 0;
	const auto device_status = cudaGetDeviceCount(&device_count);
	if (device_status != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_pipeline_dct_coeffs_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 16, 16);
	write_test_jpeg(path1, 16, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::execution::PipelineBenchmarkConfig cfg;
	cfg.mode                                    = galp::execution::PipelineBenchmarkMode::Compare;
	cfg.window_images                           = 1;
	cfg.decode_batch_rowgroups                  = 1;
	cfg.enable_rowgroup_prefetch                = false;
	cfg.coefficient_selection.coefficients      = {0U, 2U, 5U};
	constexpr size_t selected_coefficient_count = 3;
	constexpr double selected_coefficient_ratio = static_cast<double>(selected_coefficient_count) /
	                                              static_cast<double>(galp::jpeg::detail::kJpegDctCoefficientCount);
	const auto assert_selected_compare_stage = [&](const galp::execution::PipelineBenchmarkStageResult& stage) {
		EXPECT_EQ(stage.coefficients_per_block, selected_coefficient_count);
		EXPECT_EQ(stage.selected_coefficient_count, selected_coefficient_count);
		EXPECT_EQ(stage.full_coefficient_count, galp::jpeg::detail::kJpegDctCoefficientCount);
		EXPECT_DOUBLE_EQ(stage.selected_coefficient_ratio, selected_coefficient_ratio);
		EXPECT_GT(stage.output_blocks, 0U);
		EXPECT_EQ(stage.output_coefficients, stage.output_blocks * selected_coefficient_count);
		EXPECT_EQ(stage.output_bytes, stage.output_coefficients * sizeof(int16_t));
	};
	const auto assert_dct_pushdown_stage = [&](const galp::execution::PipelineBenchmarkStageResult& stage) {
		assert_selected_compare_stage(stage);
		EXPECT_EQ(stage.decoded_coefficients_per_block, selected_coefficient_count);
		EXPECT_EQ(stage.decoded_coefficients, stage.output_blocks * selected_coefficient_count);
		EXPECT_EQ(stage.decoded_bytes, stage.decoded_coefficients * sizeof(int16_t));
	};
	const auto assert_post_decode_stage = [&](const galp::execution::PipelineBenchmarkStageResult& stage) {
		assert_selected_compare_stage(stage);
		EXPECT_EQ(stage.decoded_coefficients_per_block, galp::jpeg::detail::kJpegDctCoefficientCount);
		EXPECT_EQ(stage.decoded_coefficients, stage.input_blocks * galp::jpeg::detail::kJpegDctCoefficientCount);
		EXPECT_EQ(stage.decoded_bytes, stage.decoded_coefficients * sizeof(int16_t));
	};

	cfg.crop           = galp::jpeg::JpegDctCropBox {0, 0, 8, 8};
	const auto cropped = galp::execution::benchmark_jpeg_dct_pipeline(output_dir / "manifest.bin", cfg);
	EXPECT_TRUE(cropped.outputs_match) << cropped.mismatch;
	assert_dct_pushdown_stage(cropped.pushdown);
	assert_post_decode_stage(cropped.full_then_crop);

	cfg.crop        = galp::jpeg::JpegDctCropBox {};
	const auto full = galp::execution::benchmark_jpeg_dct_pipeline(output_dir / "manifest.bin", cfg);
	EXPECT_TRUE(full.outputs_match) << full.mismatch;
	assert_dct_pushdown_stage(full.pushdown);
	assert_post_decode_stage(full.full_then_crop);

	cfg.mode               = galp::execution::PipelineBenchmarkMode::DctCompare;
	cfg.crop               = galp::jpeg::JpegDctCropBox {0, 0, 8, 8};
	const auto dct_compare = galp::execution::benchmark_jpeg_dct_pipeline(output_dir / "manifest.bin", cfg);
	EXPECT_TRUE(dct_compare.outputs_match) << dct_compare.mismatch;
	EXPECT_EQ(dct_compare.full_then_crop.windows, 0U);
	EXPECT_GT(dct_compare.pushdown.windows, 0U);
	EXPECT_GT(dct_compare.dct_post_decode.windows, 0U);
	assert_dct_pushdown_stage(dct_compare.pushdown);
	assert_post_decode_stage(dct_compare.dct_post_decode);

	cfg.mode                  = galp::execution::PipelineBenchmarkMode::Auto;
	cfg.crop                  = galp::jpeg::JpegDctCropBox {};
		const auto automatic_full = galp::execution::benchmark_jpeg_dct_pipeline(output_dir / "manifest.bin", cfg);
		EXPECT_TRUE(automatic_full.outputs_match) << automatic_full.mismatch;
		EXPECT_EQ(automatic_full.auto_pushdown_windows, 0U);
		EXPECT_EQ(automatic_full.auto_full_then_crop_windows, 0U);
		EXPECT_EQ(automatic_full.auto_no_dct_pushdown_windows, 2U);
		EXPECT_EQ(automatic_full.auto_policy_coefficient_pushdown_windows, 0U);
		EXPECT_EQ(automatic_full.auto_policy_savings_too_small_windows, 2U);
		EXPECT_NE(automatic_full.auto_policy_reason.find("savings_too_small"), std::string::npos);
		EXPECT_EQ(automatic_full.pushdown.windows, 0U);
		EXPECT_EQ(automatic_full.full_then_crop.windows, 0U);
		assert_post_decode_stage(automatic_full.auto_no_dct_pushdown);

		cfg.crop                         = galp::jpeg::JpegDctCropBox {0, 0, 16, 16};
		const auto automatic_full_crop   = galp::execution::benchmark_jpeg_dct_pipeline(output_dir / "manifest.bin", cfg);
		EXPECT_TRUE(automatic_full_crop.outputs_match) << automatic_full_crop.mismatch;
		EXPECT_EQ(automatic_full_crop.auto_policy_fast_gate_windows, 0U);
		EXPECT_EQ(automatic_full_crop.auto_policy_estimate_windows, 2U);
		EXPECT_EQ(automatic_full_crop.auto_pushdown_windows, 0U);
		EXPECT_EQ(automatic_full_crop.auto_full_then_crop_windows, 0U);
		EXPECT_EQ(automatic_full_crop.auto_no_dct_pushdown_windows, 2U);
		EXPECT_EQ(automatic_full_crop.auto_policy_coefficient_pushdown_windows, 0U);
		EXPECT_EQ(automatic_full_crop.auto_policy_savings_too_small_windows, 2U);
		EXPECT_NE(automatic_full_crop.auto_policy_reason.find("savings_too_small"), std::string::npos);
		EXPECT_EQ(automatic_full_crop.pushdown.windows, 0U);
		EXPECT_EQ(automatic_full_crop.full_then_crop.windows, 0U);
		assert_post_decode_stage(automatic_full_crop.auto_no_dct_pushdown);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanPreviewMapsPixelCropsToDctBlocks) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir  = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_plan_preview_" + std::to_string(suffix));
	const auto path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path, 16, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            aligned =
	    reader.PlanDeviceDctBatch({galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}}});
	ASSERT_EQ(aligned.image_layouts.size(), 1U);
	EXPECT_EQ(aligned.image_layouts[0].block_offset, 0U);
	EXPECT_EQ(aligned.image_layouts[0].block_count, 1U);
	ASSERT_EQ(aligned.block_metadata.size(), 1U);
	EXPECT_EQ(aligned.block_metadata[0].semantic_slot_id, 0U);
	EXPECT_EQ(aligned.block_metadata[0].block_x, 0U);
	EXPECT_EQ(aligned.block_metadata[0].block_y, 0U);
	EXPECT_EQ(aligned.rowgroups.size(), 1U);
	EXPECT_EQ(aligned.planned_selected_vector_count, 1U);
	EXPECT_EQ(aligned.estimated_selected_vector_count, 1U);
	EXPECT_EQ(aligned.full_vector_count, 1U);
	EXPECT_EQ(aligned.planned_saved_vector_count, 0U);
	EXPECT_EQ(aligned.estimated_saved_vector_count, 0U);
	EXPECT_DOUBLE_EQ(aligned.planned_selected_vector_ratio, 1.0);
	EXPECT_DOUBLE_EQ(aligned.estimated_selected_vector_ratio, 1.0);
	EXPECT_EQ(aligned.coefficients_per_block, galp::jpeg::detail::kJpegDctCoefficientCount);
	EXPECT_EQ(aligned.selected_coefficients, all_dct_coefficients());

	galp::jpeg::JpegDctDeviceBatchOptions selected_options;
	const std::vector<uint8_t>            selected_coefficients {0U, 2U, 5U};
	selected_options.coefficient_selection.coefficients = selected_coefficients;
	const auto selected                                 = reader.PlanDeviceDctBatch(
        {galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}}}, selected_options);
	ASSERT_EQ(selected.image_layouts.size(), 1U);
	EXPECT_EQ(selected.image_layouts[0].block_count, 1U);
	ASSERT_EQ(selected.block_metadata.size(), 1U);
	EXPECT_EQ(selected.coefficients_per_block, selected_coefficients.size());
	EXPECT_EQ(selected.selected_coefficients, selected_coefficients);
	auto prepared_selected = reader.PrepareDeviceDctBatch(
	    {galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}}}, selected_options);
	EXPECT_FALSE(prepared_selected.empty());
	EXPECT_EQ(prepared_selected.coefficients_per_block(), selected_coefficients.size());
	EXPECT_EQ(prepared_selected.selected_coefficients(), selected_coefficients);

	galp::jpeg::JpegDctDeviceBatchOptions duplicate_options;
	duplicate_options.coefficient_selection.coefficients = {2U, 2U};
	EXPECT_THROW(
	    (void)reader.PlanDeviceDctBatch(
	        {galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}}}, duplicate_options),
	    std::invalid_argument);
	galp::jpeg::JpegDctDeviceBatchOptions out_of_range_options;
	out_of_range_options.coefficient_selection.coefficients = {64U};
	EXPECT_THROW(
	    (void)reader.PlanDeviceDctBatch(
	        {galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}}}, out_of_range_options),
	    std::out_of_range);

	const auto unaligned =
	    reader.PlanDeviceDctBatch({galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {4, 4, 8, 8}}});
	ASSERT_EQ(unaligned.image_layouts.size(), 1U);
	EXPECT_EQ(unaligned.image_layouts[0].block_count, 4U);
	ASSERT_EQ(unaligned.block_metadata.size(), 4U);
	EXPECT_EQ(unaligned.block_metadata[0].block_x, 0U);
	EXPECT_EQ(unaligned.block_metadata[0].block_y, 0U);
	EXPECT_EQ(unaligned.block_metadata[1].block_x, 1U);
	EXPECT_EQ(unaligned.block_metadata[1].block_y, 0U);
	EXPECT_EQ(unaligned.block_metadata[2].block_x, 0U);
	EXPECT_EQ(unaligned.block_metadata[2].block_y, 1U);
	EXPECT_EQ(unaligned.block_metadata[3].block_x, 1U);
	EXPECT_EQ(unaligned.block_metadata[3].block_y, 1U);
	EXPECT_GT(unaligned.rowgroups.size(), 0U);
	EXPECT_LE(unaligned.rowgroups.size(), unaligned.block_metadata.size());
	EXPECT_EQ(unaligned.planned_selected_vector_count, 1U);
	EXPECT_EQ(unaligned.estimated_selected_vector_count, 1U);
	EXPECT_EQ(unaligned.full_vector_count, 1U);
	EXPECT_EQ(unaligned.planned_saved_vector_count, 0U);
	EXPECT_EQ(unaligned.estimated_saved_vector_count, 0U);
	EXPECT_DOUBLE_EQ(unaligned.planned_selected_vector_ratio, 1.0);
	EXPECT_DOUBLE_EQ(unaligned.estimated_selected_vector_ratio, 1.0);

	const auto clipped =
	    reader.PlanDeviceDctBatch({galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {8, 8, 64, 64}}});
	ASSERT_EQ(clipped.image_layouts.size(), 1U);
	EXPECT_EQ(clipped.image_layouts[0].block_count, 1U);
	ASSERT_EQ(clipped.block_metadata.size(), 1U);
	EXPECT_EQ(clipped.block_metadata[0].block_x, 1U);
	EXPECT_EQ(clipped.block_metadata[0].block_y, 1U);
	EXPECT_EQ(clipped.rowgroups.size(), 1U);
	EXPECT_EQ(clipped.planned_selected_vector_count, 1U);
	EXPECT_EQ(clipped.estimated_selected_vector_count, 1U);
	EXPECT_EQ(clipped.full_vector_count, 1U);
	EXPECT_EQ(clipped.planned_saved_vector_count, 0U);
	EXPECT_EQ(clipped.estimated_saved_vector_count, 0U);
	EXPECT_DOUBLE_EQ(clipped.planned_selected_vector_ratio, 1.0);
	EXPECT_DOUBLE_EQ(clipped.estimated_selected_vector_ratio, 1.0);

	const auto full_from_empty_sentinel =
	    reader.PlanDeviceDctBatch({galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 0, 0}}});
	ASSERT_EQ(full_from_empty_sentinel.image_layouts.size(), 1U);
	EXPECT_EQ(full_from_empty_sentinel.image_layouts[0].block_count, 4U);
	ASSERT_EQ(full_from_empty_sentinel.block_metadata.size(), 4U);
	EXPECT_EQ(full_from_empty_sentinel.block_metadata[0].block_x, 0U);
	EXPECT_EQ(full_from_empty_sentinel.block_metadata[0].block_y, 0U);
	EXPECT_EQ(full_from_empty_sentinel.block_metadata[3].block_x, 1U);
	EXPECT_EQ(full_from_empty_sentinel.block_metadata[3].block_y, 1U);

	EXPECT_THROW((void)reader.PlanDeviceDctBatch(
	                 {galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {16, 0, 8, 8}}}),
	             std::out_of_range);
	EXPECT_THROW((void)reader.PlanDeviceDctBatch(
	                 {galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 16, 8, 8}}}),
	             std::out_of_range);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanPreviewMapsCropsAcrossComponents) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_plan_preview_components_" + std::to_string(suffix));
	const auto path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path, 16, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            preview =
	    reader.PlanDeviceDctBatch({galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {4, 4, 8, 8}}});
	ASSERT_EQ(preview.image_layouts.size(), 1U);
	ASSERT_EQ(preview.image_layouts[0].block_offset, 0U);
	EXPECT_EQ(preview.image_layouts[0].block_count, preview.block_metadata.size());
	EXPECT_GT(preview.block_metadata.size(), 0U);
	EXPECT_GT(preview.rowgroups.size(), 0U);

	std::set<uint32_t> semantic_slots;
	for (const auto& block : preview.block_metadata) {
		EXPECT_EQ(block.request_index, 0U);
		EXPECT_EQ(block.global_image_index, 0U);
		semantic_slots.insert(block.semantic_slot_id);
	}
	EXPECT_GE(semantic_slots.size(), 3U);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanPreviewSupportsConfiguredTransformedGrid) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir    = std::filesystem::temp_directory_path() /
	                 ("galp_jpeg_dct_plan_preview_rgbnomore_fixed_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 32, 32);
	write_test_jpeg(path1, 64, 40);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {}},
	    galp::jpeg::JpegDctImageCropRequest {1, galp::jpeg::JpegDctCropBox {}},
	};
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout         = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform = galp::profiles::rgbnomore_val_dct_grid_transform();

	const auto preview = reader.PlanDeviceDctBatch(requests, options);
	EXPECT_EQ(preview.layout, galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid);
	ASSERT_EQ(preview.image_layouts.size(), requests.size());
	EXPECT_GT(preview.block_metadata.size(), 0U);
	EXPECT_GT(preview.rowgroups.size(), 0U);
	EXPECT_EQ(preview.coefficients_per_block, galp::jpeg::detail::kJpegDctCoefficientCount);
	EXPECT_EQ(preview.selected_coefficients, all_dct_coefficients());
	EXPECT_EQ(preview.ycbcr_dct_grid_shape.y, (std::array<size_t, 6> {2U, 1U, 28U, 28U, 8U, 8U}));
	EXPECT_EQ(preview.ycbcr_dct_grid_shape.cbcr, (std::array<size_t, 6> {2U, 2U, 14U, 14U, 8U, 8U}));

	auto custom_transform                     = *options.grid_transform;
	custom_transform.y_output_width_blocks    = 16;
	custom_transform.y_output_height_blocks   = 12;
	custom_transform.cbcr_output_width_blocks = 8;
	custom_transform.cbcr_output_height_blocks = 6;
	custom_transform.clamp_min                 = -500;
	custom_transform.clamp_max                 = 500;
	galp::jpeg::JpegDctDeviceBatchOptions custom_options = options;
	custom_options.grid_transform = custom_transform;
	const auto custom_preview = reader.PlanDeviceDctBatch(requests, custom_options);
	EXPECT_EQ(custom_preview.ycbcr_dct_grid_shape.y, (std::array<size_t, 6> {2U, 1U, 12U, 16U, 8U, 8U}));
	EXPECT_EQ(custom_preview.ycbcr_dct_grid_shape.cbcr, (std::array<size_t, 6> {2U, 2U, 6U, 8U, 8U, 8U}));

	galp::jpeg::JpegDctDeviceBatchOptions missing_transform = options;
	missing_transform.grid_transform.reset();
	EXPECT_THROW((void)reader.PlanDeviceDctBatch(requests, missing_transform), std::runtime_error);

	galp::jpeg::JpegDctDeviceBatchOptions sparse_options = options;
	sparse_options.coefficient_selection.coefficients    = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
	EXPECT_THROW((void)reader.PlanDeviceDctBatch(requests, sparse_options), std::runtime_error);

	auto int16_affine_transform         = *options.grid_transform;
	int16_affine_transform.output_add   = 4.0F;
	auto int16_affine_options           = options;
	int16_affine_options.grid_transform = int16_affine_transform;
	EXPECT_THROW((void)reader.PlanDeviceDctBatch(requests, int16_affine_options), std::runtime_error);

	auto float_transform                    = *options.grid_transform;
	float_transform.output_data_type        = galp::jpeg::JpegDctGridOutputDataType::kFloat32;
	float_transform.output_add              = 4.0F;
	float_transform.output_scale            = 1.0F / 1020.0F;
	auto float_options                  = options;
	float_options.grid_transform        = float_transform;
	const auto float_preview            = reader.PlanDeviceDctBatch(requests, float_options);
	EXPECT_EQ(float_preview.ycbcr_dct_grid_shape.y, preview.ycbcr_dct_grid_shape.y);
	EXPECT_EQ(float_preview.ycbcr_dct_grid_shape.cbcr, preview.ycbcr_dct_grid_shape.cbcr);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, CanonicalImageMajorFixedGridUsesCompactPlanlessDescriptors) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_planless_canonical_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	// Keep enough vectors outside the 448x448 source crop to prove that the
	// compact planless descriptor now selects a strict vector subset.
	write_test_jpeg(path0, 1024, 1024);
	write_test_jpeg(path1, 1024, 1024);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images                  = 1;
	shard_options.shard_images_specified        = true;
	shard_options.rowgroup_vectors              = 8;
	shard_options.rowgroup_vectors_specified    = true;
	shard_options.rowgroups_per_shard           = 256;
	shard_options.rowgroups_per_shard_specified = true;
	shard_options.physical_layout               = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
	shard_options.physical_layout_specified     = true;
	const auto output_dir                       = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);
	ASSERT_EQ(manifest.version, 2U);
	ASSERT_EQ(manifest.shards.size(), 2U);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout               = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform       = galp::profiles::rgbnomore_val_dct_grid_transform();
	options.cache_capacity_bytes = 0U;
	options.plan_cache_capacity  = 0U;
	const auto exact_crop = galp::jpeg::JpegDctCropBox {288U, 288U, 448U, 448U};
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {{1U, exact_crop}, {0U, exact_crop}};

	const auto preview = reader.PlanDeviceDctBatch(requests, options);
	EXPECT_TRUE(preview.uses_planless_fixed_transform);
	EXPECT_EQ(preview.compiled_access_profile_hits, 0U);
	EXPECT_EQ(preview.compiled_access_profile_misses, 6U);
	const auto repeated_preview = reader.PlanDeviceDctBatch(requests, options);
	EXPECT_EQ(repeated_preview.compiled_access_profile_hits, 6U);
	EXPECT_EQ(repeated_preview.compiled_access_profile_misses, 0U);
	EXPECT_EQ(repeated_preview.planned_selected_vector_count, preview.planned_selected_vector_count);
	EXPECT_EQ(preview.compact_image_descriptor_count, requests.size());
	EXPECT_TRUE(preview.block_metadata.empty());
	ASSERT_EQ(preview.image_layouts.size(), requests.size());
	EXPECT_EQ(preview.image_layouts[0].block_offset, 0U);
	EXPECT_EQ(preview.image_layouts[0].block_count, 4704U);
	EXPECT_EQ(preview.image_layouts[1].block_offset, 4704U);
	EXPECT_EQ(preview.image_layouts[1].block_count, 4704U);
	EXPECT_EQ(preview.rowgroups.size(), requests.size());
	EXPECT_EQ(preview.fixed_transform_component_count, 6U);
	EXPECT_EQ(preview.fixed_transform_source_block_count, 9408U);
	EXPECT_EQ(preview.fixed_transform_output_block_count, 2352U);
	EXPECT_EQ(preview.host_expanded_transform_items_created, 0U);
	EXPECT_EQ(preview.host_output_block_source_lists_created, 0U);
	EXPECT_EQ(preview.host_global_transform_sort_items, 0U);
	EXPECT_FALSE(preview.exact_batch_plan_cache_enabled);
	EXPECT_TRUE(preview.planless_axis_program_capacity_contract_complete);
	EXPECT_GT(preview.planless_axis_program_capacity_contract_bytes, preview.planless_axis_program_bytes);
	// The formal v3-capable locator also retains each image's vector-rowgroup
	// count and exact pixel dimensions for image-local binding and crop mapping.
	EXPECT_EQ(preview.compact_reader_image_locator_bytes, 2U * 32U);
	EXPECT_EQ(preview.compact_reader_shard_index_bytes, 0U);
	EXPECT_TRUE(preview.compact_reader_shard_index_derived);
	EXPECT_GT(preview.compact_reader_layout_dictionary_bytes, 0U);
	EXPECT_GT(preview.compact_reader_quant_table_dictionary_bytes, 0U);
	EXPECT_GT(preview.compact_reader_shard_descriptor_bytes, 0U);
	EXPECT_EQ(preview.compact_reader_total_bytes,
	          preview.compact_reader_image_locator_bytes + preview.compact_reader_shard_index_bytes +
	              preview.compact_reader_layout_dictionary_bytes + preview.compact_reader_quant_table_dictionary_bytes +
	              preview.compact_reader_shard_descriptor_bytes);
	uint64_t planned_storage_bytes = 0U;
	for (const auto& rowgroup : preview.rowgroups) {
		planned_storage_bytes += reader.RowgroupStorageBytes(rowgroup.shard_id, {rowgroup.rowgroup_index});
	}
	EXPECT_GT(planned_storage_bytes, 0U);
	EXPECT_GT(preview.planned_selected_vector_count, 0U);
	EXPECT_LT(preview.planned_selected_vector_count, preview.full_vector_count);
	EXPECT_EQ(preview.estimated_selected_vector_count, preview.planned_selected_vector_count);
	EXPECT_EQ(preview.planned_saved_vector_count, preview.full_vector_count - preview.planned_selected_vector_count);
	EXPECT_GT(preview.planned_saved_vector_count, 0U);
	EXPECT_EQ(preview.resize_weight_build_ms, 0.0);
	EXPECT_EQ(preview.dct_resize_weight_cache_misses, 0U);
	EXPECT_EQ(preview.dct_conversion_matrix_cache_misses, 0U);

	auto full_decode_options = options;
	full_decode_options.crop_execution_mode = galp::jpeg::JpegDctCropExecutionMode::kFullRowgroupDecode;
	const auto full_decode_preview = reader.PlanDeviceDctBatch(requests, full_decode_options);
	EXPECT_EQ(full_decode_preview.planned_selected_vector_count, preview.planned_selected_vector_count);
	EXPECT_EQ(full_decode_preview.estimated_selected_vector_count, full_decode_preview.full_vector_count);
	auto rowgroup_crop_options = options;
	rowgroup_crop_options.crop_execution_mode = galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
	const auto rowgroup_crop_preview = reader.PlanDeviceDctBatch(requests, rowgroup_crop_options);
	EXPECT_EQ(rowgroup_crop_preview.estimated_selected_vector_count,
	          rowgroup_crop_preview.planned_selected_vector_count);
	auto vector_crop_options = options;
	vector_crop_options.crop_execution_mode = galp::jpeg::JpegDctCropExecutionMode::kVectorRangeReadSelectedDecode;
	const auto vector_crop_preview = reader.PlanDeviceDctBatch(requests, vector_crop_options);
	EXPECT_EQ(vector_crop_preview.estimated_selected_vector_count, vector_crop_preview.planned_selected_vector_count);
	const std::vector<galp::jpeg::JpegDctImageCropRequest> random_crop_requests {
	    {1U, galp::jpeg::JpegDctCropBox {64U, 160U, 448U, 448U}, true, "sample-1", "crop-a"},
	    {0U, galp::jpeg::JpegDctCropBox {384U, 96U, 448U, 448U}, false, "sample-0", "crop-b"},
	};
	const auto random_crop_preview = reader.PlanDeviceDctBatch(random_crop_requests, vector_crop_options);
	EXPECT_TRUE(random_crop_preview.uses_planless_fixed_transform);
	EXPECT_EQ(random_crop_preview.compact_image_descriptor_count, random_crop_requests.size());
	EXPECT_TRUE(random_crop_preview.block_metadata.empty());
	EXPECT_TRUE(random_crop_preview.planless_axis_program_capacity_contract_complete);
	EXPECT_EQ(random_crop_preview.planless_axis_program_capacity_contract_bytes,
	          preview.planless_axis_program_capacity_contract_bytes);
	EXPECT_GE(random_crop_preview.planless_axis_program_capacity_contract_bytes,
	          random_crop_preview.planless_axis_program_bytes);
	EXPECT_GT(random_crop_preview.planned_selected_vector_count, 0U);
	EXPECT_LT(random_crop_preview.planned_selected_vector_count, random_crop_preview.full_vector_count);
	// 536 source pixels map to 67 luma blocks in this 1024-pixel fixture.
	// The reduced 28/67 output relation exercises production crops beyond the
	// old factor-64 planless limit without requiring device execution.
	const std::vector<galp::jpeg::JpegDctImageCropRequest> large_factor_requests {
	    {0U, galp::jpeg::JpegDctCropBox {0U, 0U, 536U, 536U}, false, "sample-large", "crop-large"}};
	const auto large_factor_preview = reader.PlanDeviceDctBatch(large_factor_requests, vector_crop_options);
	EXPECT_TRUE(large_factor_preview.uses_planless_fixed_transform);
	EXPECT_EQ(large_factor_preview.host_expanded_transform_items_created, 0U);
	EXPECT_TRUE(large_factor_preview.planless_axis_program_capacity_contract_complete);
	EXPECT_EQ(large_factor_preview.planless_axis_program_capacity_contract_bytes,
	          preview.planless_axis_program_capacity_contract_bytes);
	EXPECT_GE(large_factor_preview.planless_axis_program_capacity_contract_bytes,
	          large_factor_preview.planless_axis_program_bytes);
	const auto cross_thread_large_factor_preview = std::async(std::launch::async, [&] {
		galp::jpeg::JpegDctShardDatasetReader thread_reader(output_dir / "manifest.bin");
		return thread_reader.PlanDeviceDctBatch(large_factor_requests, vector_crop_options);
	}).get();
	EXPECT_GT(cross_thread_large_factor_preview.dct_resize_weight_cache_hits, 0U);
	EXPECT_EQ(cross_thread_large_factor_preview.dct_resize_weight_cache_misses, 0U);
	EXPECT_EQ(cross_thread_large_factor_preview.resize_weight_build_ms, 0.0);

	// Compact production planning bypasses the legacy exact-batch cache even
	// and decoded-rowgroup cache even when a caller leaves historical nonzero
	// defaults configured.
	options.plan_cache_capacity  = 128U;
	options.cache_capacity_bytes = 128U * 1024U * 1024U;
	auto       prepared0         = reader.PrepareDeviceDctBatch(requests, options);
	auto       prepared1         = reader.PrepareDeviceDctBatch(requests, options);
	EXPECT_TRUE(prepared0.uses_planless_fixed_transform());
	EXPECT_TRUE(prepared1.uses_planless_fixed_transform());
	EXPECT_FALSE(prepared0.exact_batch_plan_cache_enabled());
	EXPECT_FALSE(prepared1.exact_batch_plan_cache_enabled());
	EXPECT_FALSE(prepared0.decoded_rowgroup_cache_enabled());
	EXPECT_FALSE(prepared1.decoded_rowgroup_cache_enabled());
	EXPECT_FALSE(prepared0.host_io_staged());
	EXPECT_FALSE(prepared1.host_io_staged());
	auto stage0 = std::async(std::launch::async, [&] { reader.StagePreparedDeviceDctBatchIo(prepared0); });
	auto stage1 = std::async(std::launch::async, [&] { reader.StagePreparedDeviceDctBatchIo(prepared1); });
	EXPECT_NO_THROW(stage0.get());
	EXPECT_NO_THROW(stage1.get());
	EXPECT_TRUE(prepared0.host_io_staged());
	EXPECT_TRUE(prepared1.host_io_staged());
	EXPECT_EQ(prepared0.host_io_staged_rowgroups(), requests.size());
	EXPECT_EQ(prepared1.host_io_staged_rowgroups(), requests.size());
	EXPECT_GT(prepared0.host_io_staging_ms(), 0.0);
	EXPECT_GT(prepared1.host_io_staging_ms(), 0.0);
	EXPECT_NO_THROW(reader.StagePreparedDeviceDctBatchIo(prepared1));

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, PlanlessRationalProgramsCoverSamplingShapesShardsAndSpatialOrders) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_planless_generality_" + std::to_string(suffix));
	std::filesystem::create_directories(dir);
	const std::vector<std::filesystem::path> paths {
	    dir / "ycbcr420.jpg",
	    dir / "ycbcr444.jpg",
	    dir / "grayscale.jpg",
	};
	write_test_jpeg(paths[0], 536, 280, TestJpegFormat::kDefault420);
	write_test_jpeg(paths[1], 408, 264, TestJpegFormat::kYcbcr444);
	write_test_jpeg(paths[2], 344, 248, TestJpegFormat::kGrayscale);

	auto transform                               = galp::profiles::rgbnomore_val_dct_grid_transform();
	transform.y_output_width_blocks              = 7U;
	transform.y_output_height_blocks             = 5U;
	transform.cbcr_output_width_blocks           = 3U;
	transform.cbcr_output_height_blocks          = 3U;
	transform.crop_reference_width_blocks        = 1024U;
	transform.crop_reference_height_blocks       = 1024U;
	transform.preferred_small_crop_width_blocks  = {5U};
	transform.preferred_small_crop_height_blocks = {5U};

	struct OrderCase {
		const char*                     name;
		galp::jpeg::JpegDctSpatialOrder order;
	};
	const std::array<OrderCase, 4> cases {{
	    {"raster", galp::jpeg::JpegDctSpatialOrder::kRaster},
	    {"tiled_raster_32", galp::jpeg::JpegDctSpatialOrder::kTiledRaster32},
	    {"z_order", galp::jpeg::JpegDctSpatialOrder::kZOrder},
	    {"tiled_z_32", galp::jpeg::JpegDctSpatialOrder::kTiledZ32},
	}};
	for (const auto& test_case : cases) {
		galp::jpeg::JpegDctReaderOptions reader_options;
		reader_options.validation_mode           = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
		reader_options.image_major_spatial_order = test_case.order;
		galp::jpeg::JpegDctShardOptions shard_options;
		shard_options.shard_images                  = 1U;
		shard_options.shard_images_specified        = true;
		shard_options.rowgroup_vectors              = 8U;
		shard_options.rowgroup_vectors_specified    = true;
		shard_options.rowgroups_per_shard           = 1U;
		shard_options.rowgroups_per_shard_specified = true;
		shard_options.physical_layout               = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
		shard_options.physical_layout_specified     = true;
		const auto output_dir                       = dir / test_case.name;
		const auto manifest =
		    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(paths, output_dir, reader_options, shard_options);
		ASSERT_EQ(manifest.version, 2U) << test_case.name;
		ASSERT_EQ(manifest.shards.size(), paths.size()) << test_case.name;

		galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
		galp::jpeg::JpegDctDeviceBatchOptions options;
		options.layout               = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
		options.grid_transform       = transform;
		options.plan_cache_capacity  = 128U;
		options.cache_capacity_bytes = 64U * 1024U * 1024U;
		// Shuffled request order deliberately crosses all three one-image shards.
		const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {{2U, {}}, {0U, {}}, {1U, {}}};
		const auto                                             preview = reader.PlanDeviceDctBatch(requests, options);
		EXPECT_TRUE(preview.uses_planless_fixed_transform) << test_case.name;
		EXPECT_EQ(preview.compact_image_descriptor_count, requests.size()) << test_case.name;
		EXPECT_TRUE(preview.block_metadata.empty()) << test_case.name;
		EXPECT_EQ(preview.rowgroups.size(), requests.size()) << test_case.name;
		EXPECT_EQ(preview.fixed_transform_component_count, 7U) << test_case.name;
		EXPECT_EQ(preview.fixed_transform_source_block_count, 91U) << test_case.name;
		EXPECT_EQ(preview.fixed_transform_output_block_count, 159U) << test_case.name;
		EXPECT_EQ(preview.host_expanded_transform_items_created, 0U) << test_case.name;
		EXPECT_EQ(preview.host_output_block_source_lists_created, 0U) << test_case.name;
		EXPECT_EQ(preview.host_global_transform_sort_items, 0U) << test_case.name;
		EXPECT_EQ(preview.planless_axis_program_count, 2U) << test_case.name;
		EXPECT_EQ(preview.planless_axis_phase_matrix_count, 15U) << test_case.name;
		EXPECT_EQ(preview.planless_axis_program_bytes, 15U * 64U * sizeof(float)) << test_case.name;
		EXPECT_FALSE(preview.exact_batch_plan_cache_enabled) << test_case.name;

		const auto prepared0 = reader.PrepareDeviceDctBatch(requests, options);
		const auto prepared1 = reader.PrepareDeviceDctBatch(requests, options);
		EXPECT_TRUE(prepared0.uses_planless_fixed_transform()) << test_case.name;
		EXPECT_TRUE(prepared1.uses_planless_fixed_transform()) << test_case.name;
		EXPECT_FALSE(prepared0.exact_batch_plan_cache_enabled()) << test_case.name;
		EXPECT_FALSE(prepared1.exact_batch_plan_cache_enabled()) << test_case.name;
		EXPECT_FALSE(prepared0.decoded_rowgroup_cache_enabled()) << test_case.name;
		EXPECT_FALSE(prepared1.decoded_rowgroup_cache_enabled()) << test_case.name;
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ManifestV3UsesIndependentVectorRowgroupsForCropUpperBound) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_manifest_v3_vectors_" + std::to_string(suffix));
	const auto input_path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(input_path, 1024, 1024);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images                  = 1U;
	shard_options.shard_images_specified        = true;
	shard_options.rowgroup_vectors              = 64U; // v3 deliberately overrides this to one vector.
	shard_options.rowgroup_vectors_specified    = true;
	shard_options.rowgroups_per_shard           = 64U;
	shard_options.rowgroups_per_shard_specified = true;
	shard_options.physical_layout               = galp::jpeg::JpegDctPhysicalLayout::kImageMajorVectorRowgroups;
	shard_options.physical_layout_specified     = true;
	const auto output_dir                       = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({input_path}, output_dir, reader_options, shard_options);
	ASSERT_EQ(manifest.version, 3U);
	EXPECT_TRUE(manifest.uses_independent_vector_rowgroups());
	EXPECT_TRUE(manifest.uses_compact_descriptor());
	EXPECT_EQ(manifest.physical_layout, "image-major-vector-rowgroups");
	EXPECT_EQ(manifest.descriptor_kind, "galp-compact-v1");
	EXPECT_EQ(manifest.vector_size, 1024U);
	EXPECT_EQ(manifest.spatial_order_name, "tiled-z32");
	EXPECT_EQ(manifest.spatial_order, galp::jpeg::JpegDctSpatialOrder::kTiledZ32);
	EXPECT_EQ(manifest.rowgroup_vectors, 1U);
	{
		std::ifstream manifest_stream(output_dir / "manifest.bin", std::ios::binary);
		ASSERT_TRUE(manifest_stream);
		const std::string manifest_bytes {std::istreambuf_iterator<char>(manifest_stream),
		                                  std::istreambuf_iterator<char>()};
		EXPECT_NE(manifest_bytes.find("image-major-vector-rowgroups"), std::string::npos);
		EXPECT_NE(manifest_bytes.find("galp-compact-v1"), std::string::npos);
		EXPECT_NE(manifest_bytes.find("tiled-z32"), std::string::npos);
	}
	ASSERT_EQ(manifest.shards.size(), 1U);
	EXPECT_EQ(manifest.shards[0].rowgroup_count, 24U);
	EXPECT_GT(manifest.shards[0].payload_size, 0U);
	EXPECT_GT(manifest.shards[0].payload_crc64, 0U);
	EXPECT_GT(manifest.shards[0].compact_descriptor_size, 0U);
	EXPECT_GT(manifest.shards[0].source_descriptor_size, 0U);
	EXPECT_LE(manifest.shards[0].compact_descriptor_size * 5U, manifest.shards[0].source_descriptor_size);
	for (const auto& entry : std::filesystem::recursive_directory_iterator(output_dir)) {
		EXPECT_NE(entry.path().extension(), ".svb");
		EXPECT_EQ(entry.path().string().find(".standard.tmp"), std::string::npos);
	}

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            materialized = reader.MaterializeImageDct(0U);
	const auto                            source_table = galp::jpeg::read_jpeg_dct_file(input_path);
	EXPECT_EQ(materialized.blocks.size(), source_table.row_count);
	using SourceBlockKey = std::tuple<uint32_t, uint32_t, uint32_t>;
	std::map<SourceBlockKey, size_t> source_rows;
	for (const auto& group : source_table.metadata.block_group_index) {
		ASSERT_EQ(group.row_count, 1U);
		source_rows.emplace(SourceBlockKey {group.semantic_slot_id, group.block_x, group.block_y}, group.row_start);
	}
	for (const auto& block : materialized.blocks) {
		const auto source_row = source_rows.find(SourceBlockKey {block.semantic_slot_id, block.block_x, block.block_y});
		ASSERT_NE(source_row, source_rows.end());
		galp::jpeg::JpegDctCoefficientRow expected_coefficients {};
		for (size_t coefficient = 0U; coefficient < expected_coefficients.size(); ++coefficient) {
			expected_coefficients[coefficient] = source_table.columns[coefficient][source_row->second];
		}
		EXPECT_EQ(block.coefficients, expected_coefficients);
	}

	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout                    = galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	options.enable_planless_execution = true;
	const std::vector<galp::jpeg::JpegDctImageCropRequest> full_requests {{0U, {}}};
	const std::vector<galp::jpeg::JpegDctImageCropRequest> crop_requests {
	    {0U, galp::jpeg::JpegDctCropBox {0U, 0U, 256U, 256U}}};
	const auto full_preview = reader.PlanDeviceDctBatch(full_requests, options);
	const auto crop_preview = reader.PlanDeviceDctBatch(crop_requests, options);
	EXPECT_FALSE(full_preview.uses_planless_fixed_transform);
	EXPECT_FALSE(crop_preview.uses_planless_fixed_transform);
	EXPECT_EQ(full_preview.rowgroups.size(), 24U);
	EXPECT_LT(crop_preview.rowgroups.size(), full_preview.rowgroups.size());
	EXPECT_EQ(crop_preview.rowgroups.size(), 3U);
	EXPECT_EQ(crop_preview.full_vector_count, 3U);

	auto transformed_options                     = options;
	transformed_options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	transformed_options.grid_transform            = galp::profiles::rgbnomore_val_dct_grid_transform();
	transformed_options.cache_capacity_bytes      = 0U;
	transformed_options.plan_cache_capacity       = 0U;
	transformed_options.enable_rowgroup_prefetch  = false;
	const std::vector<galp::jpeg::JpegDctImageCropRequest> transformed_requests {
	    {0U, galp::jpeg::JpegDctCropBox {288U, 288U, 448U, 448U}, true}};
	const auto transformed_preview = reader.PlanDeviceDctBatch(transformed_requests, transformed_options);
	EXPECT_TRUE(transformed_preview.uses_planless_fixed_transform);
	EXPECT_EQ(transformed_preview.compact_image_descriptor_count, transformed_requests.size());
	EXPECT_TRUE(transformed_preview.block_metadata.empty());
	EXPECT_EQ(transformed_preview.host_expanded_transform_items_created, 0U);
	EXPECT_EQ(transformed_preview.host_output_block_source_lists_created, 0U);
	EXPECT_EQ(transformed_preview.host_global_transform_sort_items, 0U);
	EXPECT_GT(transformed_preview.compact_plan_bytes, 0U);
	EXPECT_EQ(transformed_preview.rowgroups.size(), transformed_preview.planned_selected_vector_count);
	EXPECT_GT(transformed_preview.planned_selected_vector_count, 0U);
	EXPECT_LT(transformed_preview.planned_selected_vector_count, transformed_preview.full_vector_count);
	EXPECT_GT(transformed_preview.planned_saved_vector_count, 0U);

	auto missing_compact_contract = manifest;
	missing_compact_contract.descriptor_kind.clear();
	const auto missing_contract_path = output_dir / "manifest-v3-missing-compact-contract.bin";
	galp::jpeg::write_jpeg_dct_shard_manifest(missing_compact_contract, missing_contract_path);
	EXPECT_THROW((void)galp::jpeg::JpegDctShardDatasetReader {missing_contract_path}, std::runtime_error);
	EXPECT_EQ(crop_preview.planned_selected_vector_count, 3U);

	const auto* run_gpu_tests = std::getenv("GALP_RUN_GPU_TESTS");
	int         device_count  = 0;
	if (run_gpu_tests != nullptr && std::string(run_gpu_tests) == "1" &&
	    cudaGetDeviceCount(&device_count) == cudaSuccess && device_count != 0) {
		using BlockKey = std::tuple<uint32_t, uint32_t, uint32_t>;
		std::map<BlockKey, galp::jpeg::JpegDctCoefficientRow> expected;
		for (const auto& block : materialized.blocks) {
			expected.emplace(BlockKey {block.semantic_slot_id, block.block_x, block.block_y}, block.coefficients);
		}
		const auto execute = [&](const galp::jpeg::JpegDctCropBox& crop, const size_t coefficient_count) {
			galp::jpeg::JpegDctDeviceBatchOptions device_options;
			device_options.layout                    = galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
			device_options.cache_capacity_bytes      = 0U;
			device_options.plan_cache_capacity       = 0U;
			device_options.enable_rowgroup_prefetch  = false;
			device_options.enable_planless_execution = true;
			device_options.coefficient_selection.coefficients.resize(coefficient_count);
			std::iota(device_options.coefficient_selection.coefficients.begin(),
			          device_options.coefficient_selection.coefficients.end(),
			          uint8_t {0U});
			auto batch = reader.ReadDeviceDctBatch(std::vector<galp::jpeg::JpegDctImageCropRequest> {{0U, crop}},
			                                       device_options);
			std::vector<int16_t> host(batch.coefficient_count());
			EXPECT_EQ(
			    cudaMemcpy(
			        host.data(), batch.device_coefficients(), host.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
			    cudaSuccess);
			if (batch.block_metadata().size() * coefficient_count != host.size()) {
				throw std::runtime_error("Compact v3 test batch coefficient shape mismatch");
			}
			for (size_t block_index = 0U; block_index < batch.block_metadata().size(); ++block_index) {
				const auto& metadata = batch.block_metadata()[block_index];
				const auto  found =
				    expected.find(BlockKey {metadata.semantic_slot_id, metadata.block_x, metadata.block_y});
				if (found == expected.end()) {
					ADD_FAILURE() << "Compact v3 device batch returned an unknown block";
					continue;
				}
				for (size_t coefficient = 0U; coefficient < coefficient_count; ++coefficient) {
					EXPECT_EQ(host[block_index * coefficient_count + coefficient], found->second[coefficient]);
				}
			}
			return batch.execution_stats();
		};

		std::map<size_t, galp::jpeg::JpegDctDeviceExecutionStats> full_prefix_stats;
		for (const size_t coefficient_count : {1U, 4U, 8U, 16U, 32U, 64U}) {
			auto stats = execute({}, coefficient_count);
			if (coefficient_count < 64U) {
				EXPECT_EQ(stats.storage_read_granularity, "selected-coefficient-range");
				EXPECT_GT(stats.coefficient_range_rowgroup_count, 0U);
				EXPECT_LT(stats.compressed_payload_bytes_read, stats.full_compressed_payload_bytes);
				EXPECT_EQ(stats.selected_coefficient_ratio, static_cast<double>(coefficient_count) / 64.0);
				EXPECT_LE(stats.physical_page_coverage_ratio, 1.0);
				EXPECT_EQ(stats.pread_count, stats.coalesced_read_run_count);
			} else {
				EXPECT_EQ(stats.storage_read_granularity, "rowgroup");
				EXPECT_EQ(stats.compressed_payload_bytes_read, stats.full_compressed_payload_bytes);
				EXPECT_EQ(stats.coefficient_logical_bytes_requested, stats.compressed_payload_bytes_read);
				EXPECT_EQ(stats.selected_coefficient_ratio, 1.0);
				EXPECT_EQ(stats.physical_page_coverage_ratio, 1.0);
				EXPECT_EQ(stats.pread_count, stats.coalesced_read_run_count);
			}
			full_prefix_stats.emplace(coefficient_count, std::move(stats));
		}
		const auto crop        = galp::jpeg::JpegDctCropBox {0U, 0U, 256U, 256U};
		const auto crop_all    = execute(crop, 64U);
		const auto crop_prefix = execute(crop, 8U);
		EXPECT_LT(crop_all.compressed_payload_bytes_read, full_prefix_stats.at(64U).compressed_payload_bytes_read);
		EXPECT_LT(crop_prefix.compressed_payload_bytes_read, crop_all.compressed_payload_bytes_read);
		EXPECT_LT(crop_prefix.compressed_payload_bytes_read, full_prefix_stats.at(8U).compressed_payload_bytes_read);
		EXPECT_EQ(crop_prefix.storage_read_granularity, "selected-coefficient-range");

		auto legacy_transformed_options                      = transformed_options;
		legacy_transformed_options.enable_planless_execution = false;
		auto transformed_planless = reader.ReadDeviceDctBatch(transformed_requests, transformed_options);
		auto transformed_legacy = reader.ReadDeviceDctBatch(transformed_requests, legacy_transformed_options);
		const auto copy_grid = [](const galp::jpeg::JpegDctDeviceBatch& batch) {
			std::pair<std::vector<int16_t>, std::vector<int16_t>> host {
			    std::vector<int16_t>(batch.y_coefficient_count()),
			    std::vector<int16_t>(batch.cbcr_coefficient_count())};
			EXPECT_EQ(cudaMemcpy(host.first.data(),
			                     batch.y_coefficients(),
			                     host.first.size() * sizeof(int16_t),
			                     cudaMemcpyDeviceToHost),
			          cudaSuccess);
			EXPECT_EQ(cudaMemcpy(host.second.data(),
			                     batch.cbcr_coefficients(),
			                     host.second.size() * sizeof(int16_t),
			                     cudaMemcpyDeviceToHost),
			          cudaSuccess);
			return host;
		};
		EXPECT_EQ(copy_grid(transformed_planless), copy_grid(transformed_legacy));
		const auto transformed_stats = transformed_planless.execution_stats();
		EXPECT_EQ(transformed_stats.planless_image_descriptor_count, transformed_requests.size());
		EXPECT_EQ(transformed_stats.fixed_transform_item_count, 0U);
		EXPECT_EQ(transformed_stats.host_expanded_transform_items_created, 0U);
		EXPECT_EQ(transformed_stats.host_output_block_source_lists_created, 0U);
		EXPECT_EQ(transformed_stats.host_global_transform_sort_items, 0U);
		EXPECT_TRUE(transformed_stats.device_mapping_fused);
		EXPECT_EQ(transformed_stats.workset_count, 1U);
		EXPECT_EQ(transformed_stats.selected_vector_count, transformed_preview.planned_selected_vector_count);
		EXPECT_LT(transformed_stats.selected_vector_count, transformed_stats.full_vector_count);
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ManifestV3PlanlessMatchesLegacyAcrossRaggedShardsAndSampling) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_manifest_v3_planless_matrix_" + std::to_string(suffix));
	std::filesystem::create_directories(dir);
	const std::vector<std::filesystem::path> paths {
	    dir / "ycbcr420.jpg",
	    dir / "ycbcr444.jpg",
	    dir / "grayscale.jpg",
	    dir / "ycbcr422.jpg",
	    dir / "ycbcr440.jpg",
	    dir / "ycbcr411.jpg",
	};
	write_test_jpeg(paths[0], 536, 280, TestJpegFormat::kDefault420);
	write_test_jpeg(paths[1], 408, 264, TestJpegFormat::kYcbcr444);
	write_test_jpeg(paths[2], 344, 248, TestJpegFormat::kGrayscale);
	write_test_jpeg(paths[3], 520, 296, TestJpegFormat::kYcbcr422);
	write_test_jpeg(paths[4], 456, 312, TestJpegFormat::kYcbcr440);
	write_test_jpeg(paths[5], 640, 320, TestJpegFormat::kYcbcr411);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images                  = 2U;
	shard_options.shard_images_specified        = true;
	shard_options.rowgroup_vectors              = 64U;
	shard_options.rowgroup_vectors_specified    = true;
	shard_options.rowgroups_per_shard           = 64U;
	shard_options.rowgroups_per_shard_specified = true;
	shard_options.physical_layout               = galp::jpeg::JpegDctPhysicalLayout::kImageMajorVectorRowgroups;
	shard_options.physical_layout_specified     = true;
	const auto output_dir                       = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(paths, output_dir, reader_options, shard_options);
	ASSERT_EQ(manifest.version, 3U);
	ASSERT_EQ(manifest.shards.size(), 3U);
	ASSERT_TRUE(manifest.uses_independent_vector_rowgroups());

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	galp::jpeg::JpegDctDeviceBatchOptions planless_options;
	planless_options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	planless_options.grid_transform            = galp::profiles::rgbnomore_val_dct_grid_transform();
	planless_options.cache_capacity_bytes      = 0U;
	planless_options.plan_cache_capacity       = 0U;
	planless_options.enable_rowgroup_prefetch  = false;
	planless_options.enable_planless_execution = true;
	auto legacy_options                        = planless_options;
	legacy_options.enable_planless_execution   = false;
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    {5U, galp::jpeg::JpegDctCropBox {32U, 32U, 224U, 224U}, true, "sample-411", "crop-flip"},
	    {0U, galp::jpeg::JpegDctCropBox {8U, 8U, 224U, 224U}, false, "sample-420", "crop"},
	    {3U, galp::jpeg::JpegDctCropBox {24U, 16U, 224U, 224U}, true, "sample-422", "crop-flip"},
	    {2U, {}, true, "sample-gray", "auto-crop-flip"},
	    {1U, galp::jpeg::JpegDctCropBox {16U, 24U, 224U, 224U}, false, "sample-444", "crop"},
	    {4U, galp::jpeg::JpegDctCropBox {40U, 32U, 224U, 224U}, true, "sample-440", "crop-flip"},
	};
	const auto preview = reader.PlanDeviceDctBatch(requests, planless_options);
	EXPECT_TRUE(preview.uses_planless_fixed_transform);
	EXPECT_EQ(preview.compact_image_descriptor_count, requests.size());
	EXPECT_TRUE(preview.block_metadata.empty());
	EXPECT_EQ(preview.host_expanded_transform_items_created, 0U);
	EXPECT_EQ(preview.host_output_block_source_lists_created, 0U);
	EXPECT_EQ(preview.host_global_transform_sort_items, 0U);
	EXPECT_EQ(preview.rowgroups.size(), preview.planned_selected_vector_count);
	EXPECT_GT(preview.planned_selected_vector_count, 0U);
	EXPECT_LT(preview.planned_selected_vector_count, preview.full_vector_count);
	EXPECT_GT(preview.planned_saved_vector_count, 0U);
	EXPECT_GT(preview.compact_plan_bytes, 0U);

	const auto empty_preview = reader.PlanDeviceDctBatch({}, planless_options);
	EXPECT_TRUE(empty_preview.uses_planless_fixed_transform);
	EXPECT_TRUE(empty_preview.rowgroups.empty());
	EXPECT_EQ(empty_preview.compact_image_descriptor_count, 0U);
	EXPECT_EQ(empty_preview.host_expanded_transform_items_created, 0U);
	auto prepared = reader.PrepareDeviceDctBatch(requests, planless_options);
	EXPECT_FALSE(prepared.host_io_staged());
	reader.StagePreparedDeviceDctBatchIo(prepared);
	EXPECT_TRUE(prepared.host_io_staged());
	EXPECT_EQ(prepared.host_io_staged_rowgroups(), preview.rowgroups.size());
	EXPECT_GT(prepared.host_io_staging_ms(), 0.0);

	const auto* run_gpu_tests = std::getenv("GALP_RUN_GPU_TESTS");
	int         device_count  = 0;
	if (run_gpu_tests == nullptr || std::string(run_gpu_tests) != "1" ||
	    cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
		std::filesystem::remove_all(dir);
		GTEST_SKIP() << "set GALP_RUN_GPU_TESTS=1 on the target-GPU host";
	}
	const auto copy_grid = [](const galp::jpeg::JpegDctDeviceBatch& batch) {
		std::pair<std::vector<int16_t>, std::vector<int16_t>> host {
		    std::vector<int16_t>(batch.y_coefficient_count()),
		    std::vector<int16_t>(batch.cbcr_coefficient_count())};
		const auto synchronized = cudaDeviceSynchronize();
		EXPECT_EQ(synchronized, cudaSuccess);
		if (synchronized != cudaSuccess) {
			return host;
		}
		if (!host.first.empty()) {
			EXPECT_EQ(cudaMemcpy(host.first.data(),
			                     batch.y_coefficients(),
			                     host.first.size() * sizeof(int16_t),
			                     cudaMemcpyDeviceToHost),
			          cudaSuccess);
		}
		if (!host.second.empty()) {
			EXPECT_EQ(cudaMemcpy(host.second.data(),
			                     batch.cbcr_coefficients(),
			                     host.second.size() * sizeof(int16_t),
			                     cudaMemcpyDeviceToHost),
			          cudaSuccess);
		}
		return host;
	};
	auto planless = reader.ReadDeviceDctBatch(requests, planless_options);
	auto legacy   = reader.ReadDeviceDctBatch(requests, legacy_options);
	const auto planless_host = copy_grid(planless);
	const auto legacy_host   = copy_grid(legacy);
	const auto expect_component_equal = [&](const std::vector<int16_t>& actual,
	                                        const std::vector<int16_t>& expected,
	                                        const char*                 component) {
		ASSERT_EQ(actual.size(), expected.size());
		ASSERT_EQ(actual.size() % requests.size(), 0U);
		const auto values_per_request = actual.size() / requests.size();
		for (size_t request_index = 0U; request_index < requests.size(); ++request_index) {
			const auto begin = request_index * values_per_request;
			const auto mismatch = std::mismatch(actual.begin() + begin,
			                                    actual.begin() + begin + values_per_request,
			                                    expected.begin() + begin);
			if (mismatch.first != actual.begin() + begin + values_per_request) {
				const auto local_index = static_cast<size_t>(mismatch.first - actual.begin()) - begin;
				ADD_FAILURE() << component << " mismatch for request " << request_index
				              << " (global image " << requests[request_index].global_image_index
				              << ", horizontal_flip=" << requests[request_index].horizontal_flip
				              << ") at local coefficient " << local_index << ": planless=" << *mismatch.first
				              << ", legacy=" << *mismatch.second;
			}
		}
	};
	expect_component_equal(planless_host.first, legacy_host.first, "Y");
	expect_component_equal(planless_host.second, legacy_host.second, "CbCr");
	const auto stats = planless.execution_stats();
	EXPECT_EQ(stats.planless_image_descriptor_count, requests.size());
	EXPECT_EQ(stats.fixed_transform_item_count, 0U);
	EXPECT_EQ(stats.host_expanded_transform_items_created, 0U);
	EXPECT_EQ(stats.host_output_block_source_lists_created, 0U);
	EXPECT_EQ(stats.host_global_transform_sort_items, 0U);
	EXPECT_TRUE(stats.device_mapping_fused);
	EXPECT_EQ(stats.workset_count, 1U);
	EXPECT_EQ(stats.selected_vector_count, preview.planned_selected_vector_count);
	EXPECT_LT(stats.selected_vector_count, stats.full_vector_count);
	EXPECT_GT(stats.actual_saved_vector_count, 0U);
	EXPECT_GT(stats.compact_batch_read_group_count, 0U);
	EXPECT_GT(stats.compact_batch_buffer_acquire_count, 0U);
	EXPECT_GT(stats.coalesced_read_run_count, 0U);
	EXPECT_GT(stats.preadv_count, 0U);
	EXPECT_EQ(stats.column_binding_rowgroup_count, stats.rowgroup_count);
	EXPECT_GT(stats.column_binding_expression_scan_count, 0U);
	EXPECT_LE(stats.column_binding_expression_scan_count,
	          stats.column_binding_rowgroup_count * galp::jpeg::detail::kJpegDctCoefficientCount);
	EXPECT_GT(stats.column_binding_ms, 0.0);
	EXPECT_EQ(stats.decode_workset_capacity_plan_image_count, requests.size());
	EXPECT_GT(stats.decode_workset_output_arena_capacity_plan_bytes, 0U);
	EXPECT_GT(stats.decode_workset_output_arena_requested_bytes, 0U);
	EXPECT_GE(stats.decode_workset_output_arena_capacity_bytes,
	          stats.decode_workset_output_arena_capacity_plan_bytes);
	EXPECT_GT(stats.decode_workset_chunk_arena_capacity_plan_bytes, 0U);
	EXPECT_GT(stats.decode_workset_chunk_arena_requested_bytes, 0U);
	EXPECT_GE(stats.decode_workset_chunk_arena_capacity_bytes,
	          stats.decode_workset_chunk_arena_capacity_plan_bytes);

	auto empty = reader.ReadDeviceDctBatch({}, planless_options);
	EXPECT_EQ(empty.image_count(), 0U);
	EXPECT_EQ(empty.y_coefficient_count(), 0U);
	EXPECT_EQ(empty.cbcr_coefficient_count(), 0U);
	EXPECT_EQ(empty.execution_stats().host_expanded_transform_items_created, 0U);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, CropExecutionModesMatchAndVectorRangeReadsFewerPhysicalBytes) {
	const auto* run_gpu_tests = std::getenv("GALP_RUN_GPU_TESTS");
	if (run_gpu_tests == nullptr || std::string(run_gpu_tests) != "1") {
		GTEST_SKIP() << "set GALP_RUN_GPU_TESTS=1 on the target-GPU host";
	}
	int device_count = 0;
	if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_crop_execution_modes_" + std::to_string(suffix));
	const auto input_path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(input_path, 1024, 1024);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images                  = 1U;
	shard_options.shard_images_specified        = true;
	shard_options.rowgroup_vectors              = 8U;
	shard_options.rowgroup_vectors_specified    = true;
	shard_options.rowgroups_per_shard           = 1U;
	shard_options.rowgroups_per_shard_specified = true;
	shard_options.physical_layout               = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
	shard_options.physical_layout_specified     = true;
	const auto output_dir                       = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({input_path}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    {0U, galp::jpeg::JpegDctCropBox {288U, 288U, 448U, 448U}}};
	galp::jpeg::JpegDctDeviceBatchOptions base_options;
	base_options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	base_options.grid_transform            = galp::profiles::rgbnomore_val_dct_grid_transform();
	base_options.cache_capacity_bytes      = 0U;
	base_options.plan_cache_capacity       = 0U;
	base_options.enable_rowgroup_prefetch  = false;
	base_options.enable_planless_execution = true;
	auto full_options                      = base_options;
	full_options.crop_execution_mode       = galp::jpeg::JpegDctCropExecutionMode::kFullRowgroupDecode;
	auto rowgroup_options                  = base_options;
	rowgroup_options.crop_execution_mode   = galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
	auto vector_options = base_options;
	vector_options.crop_execution_mode     = galp::jpeg::JpegDctCropExecutionMode::kVectorRangeReadSelectedDecode;

	auto full     = reader.ReadDeviceDctBatch(requests, full_options);
	auto rowgroup = reader.ReadDeviceDctBatch(requests, rowgroup_options);
	auto vector   = reader.ReadDeviceDctBatch(requests, vector_options);
	const auto copy_grid = [](const galp::jpeg::JpegDctDeviceBatch& batch) {
		std::pair<std::vector<int16_t>, std::vector<int16_t>> host {
		    std::vector<int16_t>(batch.y_coefficient_count()), std::vector<int16_t>(batch.cbcr_coefficient_count())};
		EXPECT_EQ(
		    cudaMemcpy(
		        host.first.data(), batch.y_coefficients(), host.first.size() * sizeof(int16_t), cudaMemcpyDeviceToHost),
		          cudaSuccess);
		EXPECT_EQ(cudaMemcpy(host.second.data(),
		                     batch.cbcr_coefficients(),
		                     host.second.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		return host;
	};
	const auto full_grid = copy_grid(full);
	EXPECT_EQ(copy_grid(rowgroup), full_grid);
	EXPECT_EQ(copy_grid(vector), full_grid);

	const auto full_stats     = full.execution_stats();
	const auto rowgroup_stats = rowgroup.execution_stats();
	const auto vector_stats   = vector.execution_stats();
	EXPECT_EQ(full_stats.actual_vector_count, full_stats.full_vector_count);
	EXPECT_EQ(full_stats.decode_granularity, "rowgroup");
	EXPECT_EQ(full_stats.storage_read_granularity, "rowgroup");
	EXPECT_EQ(full_stats.compressed_payload_bytes_read, full_stats.full_compressed_payload_bytes);
	EXPECT_GT(rowgroup_stats.actual_vector_count, 0U);
	EXPECT_LT(rowgroup_stats.actual_vector_count, rowgroup_stats.full_vector_count);
	EXPECT_EQ(rowgroup_stats.decode_granularity, "selected-vector");
	EXPECT_EQ(rowgroup_stats.storage_read_granularity, "rowgroup");
	EXPECT_EQ(rowgroup_stats.compressed_payload_bytes_read, rowgroup_stats.full_compressed_payload_bytes);
	EXPECT_EQ(vector_stats.actual_vector_count, rowgroup_stats.actual_vector_count);
	EXPECT_EQ(vector_stats.decode_granularity, "selected-vector");
	EXPECT_EQ(vector_stats.storage_read_granularity, "selected-vector-range")
	    << vector_stats.sparse_read_fallback_reason;
	EXPECT_TRUE(vector_stats.sparse_read_supported) << vector_stats.sparse_read_fallback_reason;
	EXPECT_EQ(vector_stats.sparse_read_fallback_rowgroup_count, 0U) << vector_stats.sparse_read_fallback_reason;
	EXPECT_GT(vector_stats.pread_count, 1U);
	EXPECT_LT(vector_stats.compressed_payload_bytes_read, vector_stats.full_compressed_payload_bytes);
	EXPECT_LT(vector_stats.compressed_payload_bytes_read, rowgroup_stats.compressed_payload_bytes_read);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix) {
	const auto* run_gpu_tests = std::getenv("GALP_RUN_GPU_TESTS");
	if (run_gpu_tests == nullptr || std::string(run_gpu_tests) != "1") {
		GTEST_SKIP() << "set GALP_RUN_GPU_TESTS=1 on the target-GPU host";
	}
	int device_count = 0;
	if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
		GTEST_SKIP() << "CUDA device is not available";
	}
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_planless_device_matrix_" + std::to_string(suffix));
	std::filesystem::create_directories(dir);
	const std::vector<std::filesystem::path> paths {
	    dir / "ycbcr420.jpg",
	    dir / "ycbcr444.jpg",
	    dir / "grayscale.jpg",
	    dir / "ycbcr422.jpg",
	    dir / "ycbcr440.jpg",
	    dir / "ycbcr411.jpg",
	};
	write_test_jpeg(paths[0], 536, 280, TestJpegFormat::kDefault420);
	write_test_jpeg(paths[1], 408, 264, TestJpegFormat::kYcbcr444);
	write_test_jpeg(paths[2], 344, 248, TestJpegFormat::kGrayscale);
	write_test_jpeg(paths[3], 520, 296, TestJpegFormat::kYcbcr422);
	write_test_jpeg(paths[4], 456, 312, TestJpegFormat::kYcbcr440);
	write_test_jpeg(paths[5], 640, 320, TestJpegFormat::kYcbcr411);

	auto rational_transform                               = galp::profiles::rgbnomore_val_dct_grid_transform();
	rational_transform.y_output_width_blocks              = 7U;
	rational_transform.y_output_height_blocks             = 5U;
	rational_transform.cbcr_output_width_blocks           = 3U;
	rational_transform.cbcr_output_height_blocks          = 3U;
	rational_transform.crop_reference_width_blocks        = 1024U;
	rational_transform.crop_reference_height_blocks       = 1024U;
	rational_transform.preferred_small_crop_width_blocks  = {5U};
	rational_transform.preferred_small_crop_height_blocks = {5U};
	const std::array transforms {
	    galp::profiles::rgbnomore_val_dct_grid_transform(),
	    rational_transform,
	};
	struct OrderCase {
		const char*                     name;
		galp::jpeg::JpegDctSpatialOrder order;
	};
	const std::array<OrderCase, 4>                         cases {{
        {"raster", galp::jpeg::JpegDctSpatialOrder::kRaster},
        {"tiled_raster_32", galp::jpeg::JpegDctSpatialOrder::kTiledRaster32},
        {"z_order", galp::jpeg::JpegDctSpatialOrder::kZOrder},
        {"tiled_z_32", galp::jpeg::JpegDctSpatialOrder::kTiledZ32},
    }};
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    {2U, {}},
	    {5U, {}},
	    // The 536-pixel full-width crop maps to 67 luma blocks, exercising a
	    // reduced planless axis factor above the historical limit of 64.
	    {0U, galp::jpeg::JpegDctCropBox {0U, 0U, 536U, 280U}},
	    {3U, {}},
	    {1U, {}},
	    {4U, {}}};
	const auto                                             copy_grid = [](const galp::jpeg::JpegDctDeviceBatch& batch) {
        std::pair<std::vector<int16_t>, std::vector<int16_t>> host {
            std::vector<int16_t>(batch.y_coefficient_count()), std::vector<int16_t>(batch.cbcr_coefficient_count())};
        if (!host.first.empty()) {
            EXPECT_EQ(cudaMemcpy(host.first.data(),
                                 batch.y_coefficients(),
                                 host.first.size() * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
        }
        if (!host.second.empty()) {
            EXPECT_EQ(cudaMemcpy(host.second.data(),
                                 batch.cbcr_coefficients(),
                                 host.second.size() * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
        }
        return host;
	};
	const auto copy_float_grid = [](const galp::jpeg::JpegDctDeviceBatch& batch) {
		std::pair<std::vector<float>, std::vector<float>> host {std::vector<float>(batch.y_coefficient_count()),
		                                                        std::vector<float>(batch.cbcr_coefficient_count())};
		if (!host.first.empty()) {
			EXPECT_EQ(cudaMemcpy(host.first.data(),
			                     batch.y_float_coefficients(),
			                     host.first.size() * sizeof(float),
			                     cudaMemcpyDeviceToHost),
			          cudaSuccess);
		}
		if (!host.second.empty()) {
			EXPECT_EQ(cudaMemcpy(host.second.data(),
			                     batch.cbcr_float_coefficients(),
			                     host.second.size() * sizeof(float),
			                     cudaMemcpyDeviceToHost),
			          cudaSuccess);
		}
		return host;
	};
	const auto expect_float_grid_matches_adapter =
	    [](const std::pair<std::vector<int16_t>, std::vector<int16_t>>& integer,
	       const std::pair<std::vector<float>, std::vector<float>>&     actual) {
		    const auto expect_component = [](const std::vector<int16_t>& integer_component,
		                                     const std::vector<float>&   actual_component) {
			    ASSERT_EQ(integer_component.size(), actual_component.size());
			    for (size_t index = 0; index < integer_component.size(); ++index) {
				    const float shifted  = static_cast<float>(integer_component[index]) + 4.0F;
				    const float expected = shifted * (1.0F / 1020.0F);
				    EXPECT_EQ(actual_component[index], expected) << "coefficient index " << index;
			    }
		    };
		    expect_component(integer.first, actual.first);
		    expect_component(integer.second, actual.second);
	    };

	for (const auto& test_case : cases) {
		galp::jpeg::JpegDctReaderOptions reader_options;
		reader_options.validation_mode           = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
		reader_options.image_major_spatial_order = test_case.order;
		galp::jpeg::JpegDctShardOptions shard_options;
		shard_options.shard_images                  = 1U;
		shard_options.shard_images_specified        = true;
		shard_options.rowgroup_vectors              = 8U;
		shard_options.rowgroup_vectors_specified    = true;
		shard_options.rowgroups_per_shard           = 1U;
		shard_options.rowgroups_per_shard_specified = true;
		shard_options.physical_layout               = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
		shard_options.physical_layout_specified     = true;
		const auto output_dir                       = dir / test_case.name;
		galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(paths, output_dir, reader_options, shard_options);
		galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");

		for (size_t transform_index = 0U; transform_index < transforms.size(); ++transform_index) {
			galp::jpeg::JpegDctDeviceBatchOptions planless_options;
			planless_options.layout                     = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
			planless_options.grid_transform             = transforms[transform_index];
			planless_options.cache_capacity_bytes       = 0U;
			planless_options.plan_cache_capacity        = 0U;
			planless_options.enable_planless_execution  = true;
			auto legacy_options                         = planless_options;
			legacy_options.enable_planless_execution    = false;
			auto chunked_options                        = planless_options;
			chunked_options.scheduling_policy           = galp::jpeg::JpegDctSchedulingPolicy::kLimitedOverlap;
			chunked_options.transform_blocks_per_launch = 512U;
			chunked_options.transform_ctas_per_launch   = 64U;
			chunked_options.use_low_priority_streams    = true;

			auto       planless        = reader.ReadDeviceDctBatch(requests, planless_options);
			auto       legacy          = reader.ReadDeviceDctBatch(requests, legacy_options);
			auto       planless_repeat = reader.ReadDeviceDctBatch(requests, planless_options);
			auto       chunked         = reader.ReadDeviceDctBatch(requests, chunked_options);
			const auto planless_host   = copy_grid(planless);
			const auto legacy_host     = copy_grid(legacy);
			const auto repeat_host     = copy_grid(planless_repeat);
			const auto chunked_host    = copy_grid(chunked);
			EXPECT_EQ(planless_host, legacy_host) << test_case.name << " transform=" << transform_index;
			EXPECT_EQ(planless_host, repeat_host) << test_case.name << " transform=" << transform_index;
			EXPECT_EQ(planless_host, chunked_host) << test_case.name << " transform=" << transform_index;

			auto float_transform                  = transforms[transform_index];
			float_transform.output_data_type      = galp::jpeg::JpegDctGridOutputDataType::kFloat32;
			float_transform.output_add            = 4.0F;
			float_transform.output_scale          = 1.0F / 1020.0F;
			auto float_planless_options           = planless_options;
			float_planless_options.grid_transform = float_transform;
			auto float_legacy_options             = legacy_options;
			float_legacy_options.grid_transform   = float_transform;
			auto       float_planless             = reader.ReadDeviceDctBatch(requests, float_planless_options);
			auto       float_legacy               = reader.ReadDeviceDctBatch(requests, float_legacy_options);
			const auto float_planless_host        = copy_float_grid(float_planless);
			const auto float_legacy_host          = copy_float_grid(float_legacy);
			EXPECT_EQ(float_planless_host, float_legacy_host) << test_case.name << " transform=" << transform_index;
			expect_float_grid_matches_adapter(planless_host, float_planless_host);
			EXPECT_TRUE(float_planless.execution_stats().fixed_grid_output_float32) << test_case.name;
			EXPECT_TRUE(float_planless.execution_stats().fixed_grid_output_affine_applied) << test_case.name;
			EXPECT_EQ(float_planless.execution_stats().fixed_grid_finalize_kernel_launch_count, 1U) << test_case.name;
			EXPECT_TRUE(float_legacy.execution_stats().fixed_grid_output_affine_applied) << test_case.name;
			EXPECT_EQ(float_legacy.execution_stats().fixed_grid_finalize_kernel_launch_count, 1U) << test_case.name;

			const auto planless_stats = planless.execution_stats();
			const auto legacy_stats   = legacy.execution_stats();
			const auto chunked_stats  = chunked.execution_stats();
			EXPECT_EQ(planless_stats.planless_image_descriptor_count, requests.size()) << test_case.name;
			EXPECT_EQ(planless_stats.host_io_staged_rowgroups, requests.size()) << test_case.name;
			EXPECT_GT(planless_stats.host_io_staging_ms, 0.0) << test_case.name;
			EXPECT_EQ(planless_stats.fixed_transform_item_count, 0U) << test_case.name;
			EXPECT_EQ(planless_stats.host_expanded_transform_items_created, 0U) << test_case.name;
			EXPECT_EQ(planless_stats.host_output_block_source_lists_created, 0U) << test_case.name;
			EXPECT_EQ(planless_stats.host_global_transform_sort_items, 0U) << test_case.name;
			EXPECT_FALSE(planless_stats.cache_enabled) << test_case.name;
			EXPECT_FALSE(planless_stats.exact_batch_plan_cache_enabled) << test_case.name;
			EXPECT_TRUE(planless_stats.device_mapping_fused) << test_case.name;
			EXPECT_EQ(planless_stats.device_mapping_ms, 0.0) << test_case.name;
			EXPECT_EQ(planless_stats.workset_count, 1U) << test_case.name;
			EXPECT_EQ(planless_stats.decode_kernel_launch_count, 1U) << test_case.name;
			EXPECT_EQ(planless_stats.fixed_grid_finalize_kernel_launch_count, 1U) << test_case.name;
			EXPECT_EQ(planless_stats.internal_sync_count, 1U) << test_case.name;
			EXPECT_EQ(planless_stats.decoded_batch_sync_count, 1U) << test_case.name;
			EXPECT_EQ(planless_stats.cached_gather_sync_count, 0U) << test_case.name;
			EXPECT_EQ(chunked_stats.planless_transform_max_blocks_per_launch, 64U) << test_case.name;
			EXPECT_EQ(chunked_stats.planless_transform_max_output_blocks_per_launch,
			          std::min<size_t>(512U, chunked_stats.planless_transform_output_block_count))
			    << test_case.name;
			EXPECT_EQ(chunked_stats.planless_transform_kernel_launch_count,
			          (chunked_stats.planless_transform_output_block_count + 511U) / 512U)
			    << test_case.name;
			EXPECT_EQ(chunked_stats.decode_to_transform_event_handoff_count, 1U) << test_case.name;
			EXPECT_TRUE(chunked_stats.direct_dct_low_priority_streams) << test_case.name;
			EXPECT_EQ(chunked_stats.direct_dct_stream_priority, chunked_stats.cuda_least_stream_priority)
			    << test_case.name;
			EXPECT_EQ(chunked_stats.direct_dct_h2d_stream_priority, chunked_stats.cuda_least_stream_priority)
			    << test_case.name;
			EXPECT_EQ(chunked_stats.direct_dct_decode_stream_priority, chunked_stats.cuda_least_stream_priority)
			    << test_case.name;
			EXPECT_EQ(chunked_stats.direct_dct_transform_stream_priority, chunked_stats.cuda_least_stream_priority)
			    << test_case.name;
			EXPECT_EQ(chunked_stats.direct_dct_round_stream_priority, chunked_stats.cuda_least_stream_priority)
			    << test_case.name;
			EXPECT_LT(chunked_stats.cuda_greatest_stream_priority, chunked_stats.cuda_least_stream_priority)
			    << test_case.name;
			EXPECT_EQ(chunked_stats.scheduling_policy, "limited-overlap") << test_case.name;
			EXPECT_GT(legacy_stats.fixed_transform_item_count, 0U) << test_case.name;
			// The explicit 536-pixel crop contributes reduced-factor-67 axis
			// programs to both transform variants. Keep these exact counts so the
			// large-factor regression also verifies bounded program materialization.
			const auto expected_axis_program_count = transform_index == 0U ? 4U : 6U;
			const auto expected_axis_phase_count   = transform_index == 0U ? 140U : 137U;
			EXPECT_EQ(planless_stats.planless_axis_program_count, expected_axis_program_count)
			    << test_case.name;
			EXPECT_EQ(planless_stats.planless_axis_phase_matrix_count, expected_axis_phase_count)
			    << test_case.name;
			EXPECT_EQ(planless_stats.planless_axis_program_bytes,
			          expected_axis_phase_count * 64U * sizeof(float))
			    << test_case.name;
		}
	}
	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanPreviewCanSpanRowgroups) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_plan_preview_rowgroups_" + std::to_string(suffix));
	const auto path = dir / "wide.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path, 8200, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            preview = reader.PlanDeviceDctBatch(
        {galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {8184, 0, 16, 8}}});
	ASSERT_EQ(preview.image_layouts.size(), 1U);
	EXPECT_EQ(preview.image_layouts[0].block_count, 2U);
	ASSERT_EQ(preview.block_metadata.size(), 2U);
	EXPECT_EQ(preview.block_metadata[0].block_x, 1023U);
	EXPECT_EQ(preview.block_metadata[0].block_y, 0U);
	EXPECT_EQ(preview.block_metadata[1].block_x, 1024U);
	EXPECT_EQ(preview.block_metadata[1].block_y, 0U);
	ASSERT_EQ(preview.rowgroups.size(), 2U);
	EXPECT_EQ(preview.rowgroups[0].rowgroup_index, 0U);
	EXPECT_EQ(preview.rowgroups[1].rowgroup_index, 1U);
	EXPECT_EQ(preview.planned_selected_vector_count, 2U);
	EXPECT_EQ(preview.estimated_selected_vector_count, 2U);
	EXPECT_EQ(preview.full_vector_count, 2U);
	EXPECT_DOUBLE_EQ(preview.planned_selected_vector_ratio, 1.0);
	EXPECT_DOUBLE_EQ(preview.estimated_selected_vector_ratio, 1.0);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanPreviewPreservesRequestMajorLayout) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_plan_preview_multi_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 16, 16);
	write_test_jpeg(path1, 16, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader                  reader(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}},
	    galp::jpeg::JpegDctImageCropRequest {1, galp::jpeg::JpegDctCropBox {8, 8, 8, 8}},
	};
	const auto preview = reader.PlanDeviceDctBatch(requests);
	ASSERT_EQ(preview.image_layouts.size(), requests.size());
	ASSERT_EQ(preview.block_metadata.size(), 2U);
	EXPECT_EQ(preview.image_layouts[0].global_image_index, 0U);
	EXPECT_EQ(preview.image_layouts[0].block_offset, 0U);
	EXPECT_EQ(preview.image_layouts[0].block_count, 1U);
	EXPECT_EQ(preview.image_layouts[1].global_image_index, 1U);
	EXPECT_EQ(preview.image_layouts[1].block_offset, 1U);
	EXPECT_EQ(preview.image_layouts[1].block_count, 1U);
	EXPECT_EQ(preview.block_metadata[0].request_index, 0U);
	EXPECT_EQ(preview.block_metadata[0].global_image_index, 0U);
	EXPECT_EQ(preview.block_metadata[0].block_x, 0U);
	EXPECT_EQ(preview.block_metadata[0].block_y, 0U);
	EXPECT_EQ(preview.block_metadata[1].request_index, 1U);
	EXPECT_EQ(preview.block_metadata[1].global_image_index, 1U);
	EXPECT_EQ(preview.block_metadata[1].block_x, 1U);
	EXPECT_EQ(preview.block_metadata[1].block_y, 1U);
	EXPECT_GT(preview.rowgroups.size(), 0U);
	EXPECT_LE(preview.rowgroups.size(), preview.block_metadata.size());

	auto prepared = reader.PrepareDeviceDctBatch(requests);
	EXPECT_FALSE(prepared.empty());
	EXPECT_EQ(prepared.layout(), preview.layout);
	EXPECT_EQ(prepared.image_layouts().size(), preview.image_layouts.size());
	EXPECT_EQ(prepared.block_metadata().size(), preview.block_metadata.size());
	EXPECT_EQ(prepared.rowgroups().size(), preview.rowgroups.size());
	EXPECT_EQ(prepared.planned_selected_vector_count(), preview.planned_selected_vector_count);
	EXPECT_EQ(prepared.estimated_selected_vector_count(), preview.estimated_selected_vector_count);
	EXPECT_EQ(prepared.full_vector_count(), preview.full_vector_count);
	EXPECT_EQ(prepared.planned_saved_vector_count(), preview.planned_saved_vector_count);
	EXPECT_EQ(prepared.estimated_saved_vector_count(), preview.estimated_saved_vector_count);
	EXPECT_DOUBLE_EQ(prepared.planned_selected_vector_ratio(), preview.planned_selected_vector_ratio);
	EXPECT_DOUBLE_EQ(prepared.estimated_selected_vector_ratio(), preview.estimated_selected_vector_ratio);
	galp::jpeg::JpegDctShardDatasetReader other_reader(output_dir / "manifest.bin");
	EXPECT_THROW(other_reader.ReadPreparedDeviceDctBatch(std::move(prepared)), std::runtime_error);
	EXPECT_THROW(reader.ReadPreparedDeviceDctBatch(galp::jpeg::JpegDctDeviceBatchPreparedPlan {}), std::runtime_error);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanEstimateMatchesFullPlanSummary) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_plan_estimate_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 16, 16);
	write_test_jpeg(path1, 32, 16);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader                  reader(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {}},
	    galp::jpeg::JpegDctImageCropRequest {1, galp::jpeg::JpegDctCropBox {}},
	};
	const auto preview  = reader.PlanDeviceDctBatch(requests);
	const auto estimate = reader.EstimateDeviceDctBatch(requests);

	EXPECT_EQ(estimate.layout, preview.layout);
	EXPECT_EQ(estimate.block_count, preview.block_metadata.size());
	EXPECT_EQ(estimate.full_vector_count, preview.full_vector_count);
	ASSERT_EQ(estimate.rowgroups.size(), preview.rowgroups.size());
	for (size_t i = 0; i < estimate.rowgroups.size(); ++i) {
		EXPECT_EQ(estimate.rowgroups[i].shard_id, preview.rowgroups[i].shard_id);
		EXPECT_EQ(estimate.rowgroups[i].rowgroup_index, preview.rowgroups[i].rowgroup_index);
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanEstimateSupportsImageMajorLayout) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_image_major_estimate_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 32, 24);
	write_test_jpeg(path1, 64, 40);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images              = 2;
	shard_options.rowgroup_vectors          = 1;
	shard_options.rowgroups_per_shard       = 2;
	shard_options.physical_layout           = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
	shard_options.physical_layout_specified = true;
	const auto output_dir = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    {1U, {8U, 8U, 32U, 24U}},
	    {0U, {}},
	};
	const auto preview  = reader.PlanDeviceDctBatch(requests);
	const auto estimate = reader.EstimateDeviceDctBatch(requests);

	EXPECT_EQ(estimate.layout, preview.layout);
	EXPECT_EQ(estimate.block_count, preview.block_metadata.size());
	EXPECT_EQ(estimate.full_vector_count, preview.full_vector_count);
	ASSERT_EQ(estimate.rowgroups.size(), preview.rowgroups.size());
	for (size_t rowgroup = 0; rowgroup < estimate.rowgroups.size(); ++rowgroup) {
		EXPECT_EQ(estimate.rowgroups[rowgroup].shard_id, preview.rowgroups[rowgroup].shard_id);
		EXPECT_EQ(estimate.rowgroups[rowgroup].rowgroup_index, preview.rowgroups[rowgroup].rowgroup_index);
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, DeviceBatchPlanPreviewEstimatesVectorSavings) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_plan_preview_vectors_" + std::to_string(suffix));
	const auto path0 = dir / "wide0.jpg";
	const auto path1 = dir / "wide1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 4800, 8);
	write_test_jpeg(path1, 4800, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 2;
	shard_options.rowgroups_per_shard = 256;
	const auto output_dir             = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            preview =
	    reader.PlanDeviceDctBatch({galp::jpeg::JpegDctImageCropRequest {0, galp::jpeg::JpegDctCropBox {0, 0, 8, 8}}});
	ASSERT_EQ(preview.block_metadata.size(), 1U);
	ASSERT_EQ(preview.rowgroups.size(), 1U);
	EXPECT_EQ(preview.planned_selected_vector_count, 1U);
	EXPECT_EQ(preview.estimated_selected_vector_count, 2U);
	EXPECT_EQ(preview.full_vector_count, 2U);
	EXPECT_EQ(preview.planned_saved_vector_count, 1U);
	EXPECT_EQ(preview.estimated_saved_vector_count, 0U);
	EXPECT_DOUBLE_EQ(preview.planned_selected_vector_ratio, 0.5);
	EXPECT_DOUBLE_EQ(preview.estimated_selected_vector_ratio, 1.0);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, RuntimePolicyUsesVectorPushdownOnlyForMeaningfulSavings) {
	using galp::jpeg::detail::choose_jpeg_dct_runtime_policy;
	using galp::jpeg::detail::JpegDctRuntimePolicyDecision;
	using galp::jpeg::detail::JpegDctRuntimePolicyReason;

	{
		const auto policy = choose_jpeg_dct_runtime_policy(8, 32, true);
		EXPECT_EQ(policy.decision, JpegDctRuntimePolicyDecision::kSelectedVectors);
		EXPECT_EQ(policy.reason, JpegDctRuntimePolicyReason::kCropSavesEnoughVectors);
	}
	{
		const auto policy = choose_jpeg_dct_runtime_policy(24, 32, true);
		EXPECT_EQ(policy.decision, JpegDctRuntimePolicyDecision::kFullRowgroup);
		EXPECT_EQ(policy.reason, JpegDctRuntimePolicyReason::kSelectedCoversMostVectors);
	}
	{
		const auto policy = choose_jpeg_dct_runtime_policy(5, 8, true);
		EXPECT_EQ(policy.decision, JpegDctRuntimePolicyDecision::kFullRowgroup);
		EXPECT_EQ(policy.reason, JpegDctRuntimePolicyReason::kSavingsTooSmall);
	}
	{
		const auto policy = choose_jpeg_dct_runtime_policy(8, 32, false);
		EXPECT_EQ(policy.decision, JpegDctRuntimePolicyDecision::kFullRowgroup);
		EXPECT_EQ(policy.reason, JpegDctRuntimePolicyReason::kTailChunkWouldOverrun);
	}
}

TEST(JpegDct, SparseStoragePolicyPricesFragmentationAndMaterialization) {
	using galp::jpeg::detail::choose_jpeg_dct_sparse_storage_policy;
	using galp::jpeg::detail::JpegDctSparseStorageCost;
	using galp::jpeg::detail::JpegDctSparseStoragePolicyReason;

	// Representative of the measured source-range path: modest byte savings
	// cannot pay for hundreds of physical reads per rowgroup.
	const auto fragmented =
	    choose_jpeg_dct_sparse_storage_policy(JpegDctSparseStorageCost {226020U, 181660U, 902U, true});
	EXPECT_FALSE(fragmented.use_sparse_read);
	EXPECT_EQ(fragmented.reason, JpegDctSparseStoragePolicyReason::kFragmentationDominates);
	EXPECT_GT(fragmented.sparse_estimated_ns, fragmented.full_estimated_ns);

	// A compact contiguous representation with substantial byte savings is a
	// legitimate automatic sparse backend even after reconstruction is priced.
	const auto compact =
	    choose_jpeg_dct_sparse_storage_policy(JpegDctSparseStorageCost {1024U * 1024U, 256U * 1024U, 1U, true});
	EXPECT_TRUE(compact.use_sparse_read);
	EXPECT_EQ(compact.reason, JpegDctSparseStoragePolicyReason::kPredictedFaster);
	EXPECT_LT(compact.sparse_estimated_ns, compact.full_estimated_ns);

	// One pread is not sufficient by itself: an envelope that saves few bytes
	// still loses to the required logical-rowgroup clear/scatter pass.
	const auto marginal_envelope =
	    choose_jpeg_dct_sparse_storage_policy(JpegDctSparseStorageCost {226020U, 199600U, 1U, true});
	EXPECT_FALSE(marginal_envelope.use_sparse_read);
	EXPECT_EQ(marginal_envelope.reason, JpegDctSparseStoragePolicyReason::kFragmentationDominates);
}

TEST(JpegDct, AdaptiveReadPolicyComparesRunBitmapFullAndMemoryBudget) {
	using galp::jpeg::detail::choose_jpeg_dct_adaptive_read_policy;
	using galp::jpeg::detail::JpegDctAdaptiveReadCost;
	using galp::jpeg::detail::JpegDctReadStrategy;
	constexpr size_t bytes_per_vector = 1024U * 64U * sizeof(int16_t);

	const auto compact_run = choose_jpeg_dct_adaptive_read_policy(JpegDctAdaptiveReadCost {
	    1024U * 1024U, 256U * 1024U, 1U, 2U, 16U, bytes_per_vector, 64U * 1024U * 1024U, true, true, true});
	EXPECT_EQ(compact_run.strategy, JpegDctReadStrategy::kRunIntervalExact);
	EXPECT_LT(compact_run.run_interval_estimated_ns, compact_run.bitmap_estimated_ns);
	EXPECT_TRUE(compact_run.selected_fits_memory);

	const auto fragmented = choose_jpeg_dct_adaptive_read_policy(JpegDctAdaptiveReadCost {
	    226020U, 181660U, 902U, 2U, 16U, bytes_per_vector, 64U * 1024U * 1024U, true, true, true});
	EXPECT_EQ(fragmented.strategy, JpegDctReadStrategy::kBitmapExact);
	EXPECT_GT(fragmented.run_interval_estimated_ns, fragmented.bitmap_estimated_ns);

	const auto dense = choose_jpeg_dct_adaptive_read_policy(JpegDctAdaptiveReadCost {
	    1024U * 1024U, 0U, 0U, 15U, 16U, bytes_per_vector, 64U * 1024U * 1024U, true, false, true});
	EXPECT_EQ(dense.strategy, JpegDctReadStrategy::kFullRowgroup);

	const auto memory_bounded = choose_jpeg_dct_adaptive_read_policy(JpegDctAdaptiveReadCost {
	    1024U * 1024U, 0U, 0U, 2U, 16U, bytes_per_vector, 3U * 1024U * 1024U, true, false, true});
	EXPECT_EQ(memory_bounded.strategy, JpegDctReadStrategy::kBitmapExact);
	EXPECT_TRUE(memory_bounded.selected_fits_memory);
	EXPECT_FALSE(memory_bounded.full_fits_memory);

	const auto selected_unsupported = choose_jpeg_dct_adaptive_read_policy(JpegDctAdaptiveReadCost {
	    1024U * 1024U, 0U, 0U, 2U, 16U, bytes_per_vector, 64U * 1024U * 1024U, false, false, true});
	EXPECT_EQ(selected_unsupported.strategy, JpegDctReadStrategy::kFullRowgroup);
}

TEST(JpegDct, BatchUnpackWidthFallsBackBeforeBuildingAWorkset) {
	using galp::jpeg::detail::constrain_jpeg_dct_batch_unpack_n_vectors;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPlan;
	using galp::jpeg::detail::JpegDctRuntimePolicyDecision;

	JpegDctDeviceRowgroupPlan divisible_full;
	divisible_full.has_vector_plan         = true;
	divisible_full.full_vector_count       = 12;
	divisible_full.runtime_policy.decision = JpegDctRuntimePolicyDecision::kFullRowgroup;

	JpegDctDeviceRowgroupPlan selected_tail;
	selected_tail.has_vector_plan         = true;
	selected_tail.full_vector_count       = 10;
	selected_tail.runtime_policy.decision = JpegDctRuntimePolicyDecision::kSelectedVectors;

	unsigned unpack_n_vectors = 4;
	unpack_n_vectors = constrain_jpeg_dct_batch_unpack_n_vectors(unpack_n_vectors, divisible_full);
	unpack_n_vectors = constrain_jpeg_dct_batch_unpack_n_vectors(unpack_n_vectors, selected_tail);
	EXPECT_EQ(unpack_n_vectors, 4U);

	JpegDctDeviceRowgroupPlan incompatible_full = selected_tail;
	incompatible_full.runtime_policy.decision = JpegDctRuntimePolicyDecision::kFullRowgroup;
	unpack_n_vectors = constrain_jpeg_dct_batch_unpack_n_vectors(unpack_n_vectors, incompatible_full);
	EXPECT_EQ(unpack_n_vectors, 1U);
	EXPECT_EQ(constrain_jpeg_dct_batch_unpack_n_vectors(4U, JpegDctDeviceRowgroupPlan {}), 1U);
}

TEST(JpegDct, AutoPipelinePolicyAvoidsTinyRowgroupOverhead) {
	using galp::execution::detail::AutoPipelinePolicyReason;
	using galp::execution::detail::choose_auto_coefficient_selection_policy_from_estimates;
	using galp::execution::detail::choose_auto_pipeline_policy_from_block_estimate;
	using galp::execution::detail::choose_auto_pipeline_policy_from_counts;
	using galp::execution::detail::choose_auto_pipeline_policy_from_estimates;

	{
		const auto policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/4096,
		    /*full_blocks=*/4096,
		    /*touched_rowgroups=*/4,
		    /*full_rowgroups=*/4,
		    /*selected_vectors=*/4,
		    /*full_vectors=*/4);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CropCoversFullWindow);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/768, /*full_blocks=*/3072, /*touched_rowgroups=*/1, /*full_rowgroups=*/2);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_EQ(policy.estimated_pushdown_worksets, 1U);
		EXPECT_EQ(policy.estimated_full_worksets, 1U);
		EXPECT_EQ(policy.estimated_pushdown_gather_items, 768U);
		EXPECT_EQ(policy.estimated_full_gather_items, 3072U);
	}
	{
		const auto svhn_policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/1122, /*full_blocks=*/1584, /*touched_rowgroups=*/1, /*full_rowgroups=*/1);
		EXPECT_TRUE(svhn_policy.use_pushdown);
		EXPECT_EQ(svhn_policy.reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);

		const auto cifar_policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/1632, /*full_blocks=*/2304, /*touched_rowgroups=*/1, /*full_rowgroups=*/2);
		EXPECT_TRUE(cifar_policy.use_pushdown);
		EXPECT_EQ(cifar_policy.reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/128,
		    /*full_blocks=*/4096,
		    /*touched_rowgroups=*/1,
		    /*full_rowgroups=*/2,
		    /*selected_vectors=*/1,
		    /*full_vectors=*/32);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
		EXPECT_LT(policy.selected_block_ratio, galp::execution::detail::kAutoVerySmallCropBlockRatio);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/768, /*full_blocks=*/12288, /*touched_rowgroups=*/1, /*full_rowgroups=*/2);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::VerySmallCrop);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(/*selected_blocks=*/3072,
		                                                            /*full_blocks=*/12288,
		                                                            /*touched_rowgroups=*/6,
		                                                            /*full_rowgroups=*/24,
		                                                            /*selected_vectors=*/24,
		                                                            /*full_vectors=*/96);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_LT(policy.avg_full_blocks_per_rowgroup, galp::execution::detail::kAutoMinAvgFullBlocksPerRowgroup);
		EXPECT_LE(policy.touched_rowgroup_ratio,
		          galp::execution::detail::kAutoMaxTinyRowgroupPushdownTouchedRowgroupRatio);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(/*selected_blocks=*/3072,
		                                                            /*full_blocks=*/12288,
		                                                            /*touched_rowgroups=*/12,
		                                                            /*full_rowgroups=*/24,
		                                                            /*selected_vectors=*/24,
		                                                            /*full_vectors=*/96);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::TinyRowgroupsFixedOverhead);
		EXPECT_LT(policy.avg_full_blocks_per_rowgroup, galp::execution::detail::kAutoMinAvgFullBlocksPerRowgroup);
		EXPECT_GT(policy.touched_rowgroup_ratio,
		          galp::execution::detail::kAutoMaxTinyRowgroupPushdownTouchedRowgroupRatio);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(/*selected_blocks=*/102400,
		                                                            /*full_blocks=*/128000,
		                                                            /*touched_rowgroups=*/10,
		                                                            /*full_rowgroups=*/100,
		                                                            /*selected_vectors=*/80,
		                                                            /*full_vectors=*/100);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::LargeWindowAmortizesPushdown);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(/*selected_blocks=*/115200,
		                                                            /*full_blocks=*/128000,
		                                                            /*touched_rowgroups=*/10,
		                                                            /*full_rowgroups=*/100,
		                                                            /*selected_vectors=*/60,
		                                                            /*full_vectors=*/100);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::GatherOutputTooHigh);
		EXPECT_GT(policy.selected_block_ratio, 0.85);
		EXPECT_LT(policy.selected_vector_ratio, 0.85);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/180000, /*full_blocks=*/200000, /*touched_rowgroups=*/90, /*full_rowgroups=*/100);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::TouchesMostRowgroups);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(/*selected_blocks=*/1000,
		                                                            /*full_blocks=*/20000,
		                                                            /*touched_rowgroups=*/1,
		                                                            /*full_rowgroups=*/10,
		                                                            /*selected_vectors=*/90,
		                                                            /*full_vectors=*/100);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::WorksetOverheadTooHigh);
		EXPECT_DOUBLE_EQ(policy.selected_vector_ratio, 0.9);
		EXPECT_EQ(policy.estimated_pushdown_worksets, 1U);
		EXPECT_EQ(policy.estimated_full_worksets, 1U);
		EXPECT_EQ(policy.estimated_pushdown_gather_items, 1000U);
		EXPECT_EQ(policy.estimated_full_gather_items, 20000U);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/768, /*full_blocks=*/3072, /*image_count=*/128);
		ASSERT_TRUE(policy.has_value());
		EXPECT_TRUE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_NE(policy->reason.find("metadata_fast=1"), std::string::npos);
	}
	{
		const auto svhn_policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/1122, /*full_blocks=*/1584, /*image_count=*/128);
		ASSERT_TRUE(svhn_policy.has_value());
		EXPECT_TRUE(svhn_policy->use_pushdown);
		EXPECT_EQ(svhn_policy->reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_NE(svhn_policy->reason.find("metadata_fast=1"), std::string::npos);

		const auto cifar_policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/1632, /*full_blocks=*/2304, /*image_count=*/128);
		ASSERT_TRUE(cifar_policy.has_value());
		EXPECT_TRUE(cifar_policy->use_pushdown);
		EXPECT_EQ(cifar_policy->reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_NE(cifar_policy->reason.find("metadata_fast=1"), std::string::npos);
	}
	{
		const auto svhn_policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/396, /*full_blocks=*/1584, /*image_count=*/128);
		ASSERT_TRUE(svhn_policy.has_value());
		EXPECT_TRUE(svhn_policy->use_pushdown);
		EXPECT_EQ(svhn_policy->reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);

		const auto cifar_policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/576, /*full_blocks=*/2304, /*image_count=*/128);
		ASSERT_TRUE(cifar_policy.has_value());
		EXPECT_TRUE(cifar_policy->use_pushdown);
		EXPECT_EQ(cifar_policy->reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/32000, /*full_blocks=*/100000, /*image_count=*/128);
		ASSERT_TRUE(policy.has_value());
		EXPECT_TRUE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::LargeWindowAmortizesPushdown);
		EXPECT_NE(policy->reason.find("metadata_fast=1"), std::string::npos);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/1000, /*full_blocks=*/200000, /*image_count=*/128);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/512, /*full_blocks=*/8192, /*image_count=*/256);
		ASSERT_TRUE(policy.has_value());
		EXPECT_FALSE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/80000, /*full_blocks=*/100000, /*image_count=*/2000);
		ASSERT_TRUE(policy.has_value());
		EXPECT_FALSE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/10000, /*full_blocks=*/20000, /*image_count=*/400);
		ASSERT_TRUE(policy.has_value());
		EXPECT_FALSE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/102400, /*full_blocks=*/128000, /*image_count=*/128);
		ASSERT_TRUE(policy.has_value());
		EXPECT_FALSE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::GatherOutputTooHigh);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/3072, /*full_blocks=*/12288, /*image_count=*/128);
		ASSERT_TRUE(policy.has_value());
		EXPECT_TRUE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_NE(policy->reason.find("metadata_fast=1"), std::string::npos);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/10000, /*full_blocks=*/20000, /*image_count=*/128);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_counts(/*selected_blocks=*/1000,
		                                                            /*full_blocks=*/200000,
		                                                            /*touched_rowgroups=*/1,
		                                                            /*full_rowgroups=*/128,
		                                                            /*selected_vectors=*/90,
		                                                            /*full_vectors=*/100);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::SavingsTooSmall);
		EXPECT_DOUBLE_EQ(policy.selected_vector_ratio, 0.9);
		EXPECT_EQ(policy.estimated_pushdown_worksets, 1U);
		EXPECT_EQ(policy.estimated_full_worksets, 2U);
	}
	{
		const auto policy =
		    choose_auto_pipeline_policy_from_estimates(/*selected_blocks=*/10000,
		                                                               /*full_blocks=*/20000,
		                                                               /*touched_rowgroups=*/2,
		                                                               /*full_rowgroups=*/10,
		                                                               /*selected_vectors=*/6,
		                                                               /*full_vectors=*/10,
		                                                               /*estimated_pushdown_worksets=*/1,
		                                                               /*estimated_full_worksets=*/2,
		                                                               /*selected_coefficients=*/64,
		                                                               /*active_physical_coefficients=*/64,
		                                                               /*estimated_pushdown_reuse_candidate_rowgroups=*/1,
		                                                               /*estimated_full_reuse_candidate_rowgroups=*/3);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_EQ(policy.estimated_pushdown_worksets, 1U);
		EXPECT_EQ(policy.estimated_full_worksets, 2U);
		EXPECT_EQ(policy.estimated_pushdown_reuse_candidate_rowgroups, 1U);
		EXPECT_EQ(policy.estimated_full_reuse_candidate_rowgroups, 3U);
		EXPECT_DOUBLE_EQ(policy.selected_blocks_per_pushdown_workset, 10000.0);
		EXPECT_NE(policy.reason.find("selected_blocks_per_pushdown_workset=10000.000000"), std::string::npos);
		EXPECT_NE(policy.reason.find("estimated_pushdown_reuse_candidate_rowgroups=1"), std::string::npos);
		EXPECT_NE(policy.reason.find("estimated_full_reuse_candidate_rowgroups=3"), std::string::npos);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_estimates(/*selected_blocks=*/10000,
		                                                               /*full_blocks=*/20000,
		                                                               /*touched_rowgroups=*/2,
		                                                               /*full_rowgroups=*/10,
		                                                               /*selected_vectors=*/6,
		                                                               /*full_vectors=*/10,
		                                                               /*estimated_pushdown_worksets=*/2,
		                                                               /*estimated_full_worksets=*/2);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::WorksetOverheadTooHigh);
		EXPECT_EQ(policy.estimated_pushdown_worksets, 2U);
		EXPECT_EQ(policy.estimated_full_worksets, 2U);
	}
	{
		const auto policy = choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/360000,
		                                                                            /*full_blocks=*/360000,
		                                                                            /*touched_rowgroups=*/89,
		                                                                            /*full_rowgroups=*/89,
		                                                                            /*selected_vectors=*/2809,
		                                                                            /*full_vectors=*/11211,
		                                                                            /*estimated_pushdown_worksets=*/469,
		                                                                            /*estimated_full_worksets=*/469,
		                                                                            /*selected_coefficients=*/8);
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::SavingsTooSmall);
		EXPECT_EQ(policy.selected_coefficient_count, 8U);
		EXPECT_EQ(policy.active_physical_coefficient_count, 8U);
		EXPECT_EQ(policy.estimated_pushdown_decoded_bytes,
		          2809U * galp::codec::consts::VALUES_PER_VECTOR * 8U * sizeof(int16_t));
		EXPECT_EQ(policy.estimated_output_bytes, 360000U * 8U * sizeof(int16_t));
		EXPECT_NE(policy.reason.find("active_physical_coefficients=8"), std::string::npos);
		EXPECT_NE(policy.reason.find("estimated_pushdown_decoded_bytes="), std::string::npos);
		EXPECT_NE(policy.reason.find("estimated_output_bytes="), std::string::npos);
	}
	{
		const auto policy = choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/321702,
		                                                                            /*full_blocks=*/321702,
		                                                                            /*touched_rowgroups=*/256,
		                                                                            /*full_rowgroups=*/256,
		                                                                            /*selected_vectors=*/1144,
		                                                                            /*full_vectors=*/1144,
		                                                                            /*estimated_pushdown_worksets=*/4,
		                                                                            /*estimated_full_worksets=*/4,
		                                                                            /*selected_coefficients=*/8);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CoefficientSelectionPushdown);
		EXPECT_GT(policy.selected_blocks_per_pushdown_workset,
		          galp::execution::detail::kAutoMinCoefficientPushdownBlocksPerWorkset);
		EXPECT_NE(policy.reason.find("coefficient_selection_pushdown"), std::string::npos);
	}
	{
		const auto policy = choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/1528,
		                                                                            /*full_blocks=*/1528,
		                                                                            /*touched_rowgroups=*/6,
		                                                                            /*full_rowgroups=*/6,
		                                                                            /*selected_vectors=*/32,
		                                                                            /*full_vectors=*/32,
		                                                                            /*estimated_pushdown_worksets=*/1,
		                                                                            /*estimated_full_worksets=*/1,
		                                                                            /*selected_coefficients=*/8);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CoefficientSelectionPushdown);
	}
	{
		const auto svhn_policy =
		    choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/396,
		                                                            /*full_blocks=*/396,
		                                                            /*touched_rowgroups=*/1,
		                                                            /*full_rowgroups=*/1,
		                                                            /*selected_vectors=*/2,
		                                                            /*full_vectors=*/2,
		                                                            /*estimated_pushdown_worksets=*/1,
		                                                            /*estimated_full_worksets=*/1,
		                                                            /*selected_coefficients=*/8);
		EXPECT_FALSE(svhn_policy.use_pushdown);
		EXPECT_EQ(svhn_policy.reason_code, AutoPipelinePolicyReason::SavingsTooSmall);
		EXPECT_LT(svhn_policy.selected_blocks_per_pushdown_workset,
		          galp::execution::detail::kAutoMinCoefficientPushdownBlocksPerWorkset);

		const auto cifar_policy =
		    choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/576,
		                                                            /*full_blocks=*/576,
		                                                            /*touched_rowgroups=*/3,
		                                                            /*full_rowgroups=*/3,
		                                                            /*selected_vectors=*/5,
		                                                            /*full_vectors=*/5,
		                                                            /*estimated_pushdown_worksets=*/1,
		                                                            /*estimated_full_worksets=*/1,
		                                                            /*selected_coefficients=*/8);
		EXPECT_FALSE(cifar_policy.use_pushdown);
		EXPECT_EQ(cifar_policy.reason_code, AutoPipelinePolicyReason::SavingsTooSmall);
		EXPECT_LT(cifar_policy.selected_blocks_per_pushdown_workset,
		          galp::execution::detail::kAutoMinCoefficientPushdownBlocksPerWorkset);
	}
	{
		const auto crop_only_policy = choose_auto_pipeline_policy_from_counts(
		    /*selected_blocks=*/768,
		    /*full_blocks=*/12288,
		    /*touched_rowgroups=*/1,
		    /*full_rowgroups=*/2,
		    /*selected_vectors=*/6,
		    /*full_vectors=*/96);
		EXPECT_TRUE(crop_only_policy.use_pushdown);
		EXPECT_EQ(crop_only_policy.reason_code, AutoPipelinePolicyReason::VerySmallCrop);

		const auto policy = choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/768,
		                                                                            /*full_blocks=*/12288,
		                                                                            /*touched_rowgroups=*/1,
		                                                                            /*full_rowgroups=*/2,
		                                                                            /*selected_vectors=*/6,
		                                                                            /*full_vectors=*/96,
		                                                                            /*estimated_pushdown_worksets=*/1,
		                                                                            /*estimated_full_worksets=*/1,
		                                                                            /*selected_coefficients=*/8);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::VerySmallCrop);
		EXPECT_NE(policy.reason.find("very_small_crop"), std::string::npos);
		EXPECT_EQ(policy.selected_coefficient_count, 8U);
		EXPECT_EQ(policy.active_physical_coefficient_count, 8U);
	}
	{
		const auto policy = choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/10000,
		                                                                            /*full_blocks=*/20000,
		                                                                            /*touched_rowgroups=*/2,
		                                                                            /*full_rowgroups=*/10,
		                                                                            /*selected_vectors=*/6,
		                                                                            /*full_vectors=*/10,
		                                                                            /*estimated_pushdown_worksets=*/1,
		                                                                            /*estimated_full_worksets=*/2,
		                                                                            /*selected_coefficients=*/8);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_NE(policy.reason.find("crop_saves_enough_blocks"), std::string::npos);
	}
	{
		const auto crop_only_policy = choose_auto_pipeline_policy_from_counts(/*selected_blocks=*/14237789,
		                                                                      /*full_blocks=*/14237789,
		                                                                      /*touched_rowgroups=*/5425,
		                                                                      /*full_rowgroups=*/5425,
		                                                                      /*selected_vectors=*/52333,
		                                                                      /*full_vectors=*/71999,
		                                                                      /*decode_batch_rowgroups=*/28);
		EXPECT_FALSE(crop_only_policy.use_pushdown);
		EXPECT_EQ(crop_only_policy.reason_code, AutoPipelinePolicyReason::CropCoversFullWindow);

		const auto policy = choose_auto_coefficient_selection_policy_from_estimates(/*selected_blocks=*/14237789,
		                                                                            /*full_blocks=*/14237789,
		                                                                            /*touched_rowgroups=*/5425,
		                                                                            /*full_rowgroups=*/5425,
		                                                                            /*selected_vectors=*/52333,
		                                                                            /*full_vectors=*/71999,
		                                                                            /*estimated_pushdown_worksets=*/193,
		                                                                            /*estimated_full_worksets=*/193,
		                                                                            /*selected_coefficients=*/8,
		                                                                            /*active_physical_coefficients=*/6);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CoefficientSelectionPushdown);
		EXPECT_EQ(policy.selected_coefficient_count, 8U);
		EXPECT_EQ(policy.active_physical_coefficient_count, 6U);
		EXPECT_EQ(policy.estimated_pushdown_decoded_bytes,
		          52333U * galp::codec::consts::VALUES_PER_VECTOR * 6U * sizeof(int16_t));
		EXPECT_EQ(policy.estimated_full_decoded_bytes,
		          71999U * galp::codec::consts::VALUES_PER_VECTOR *
		              galp::execution::detail::kAutoJpegDctCoefficientCount * sizeof(int16_t));
		EXPECT_EQ(policy.estimated_output_bytes, 14237789U * 8U * sizeof(int16_t));
		EXPECT_NE(policy.reason.find("coefficient_selection_pushdown"), std::string::npos);
		EXPECT_NE(policy.reason.find("active_physical_coefficients=6"), std::string::npos);
	}
}

TEST(JpegDct, AutoPipelineReuseCandidateCountsRepeatedPreviewRowgroups) {
	using galp::execution::detail::count_auto_reuse_candidate_rowgroups;
	using galp::execution::detail::estimate_auto_reuse_candidate_rowgroups;
	using galp::execution::detail::estimate_auto_worksets_for_rowgroups;

	std::set<std::pair<uint32_t, uint32_t>>                seen;
	std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata> first_window {
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {0, 7},
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {0, 8},
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {1, 7},
	};
	std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata> second_window {
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {0, 8},
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {1, 7},
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {1, 9},
	};

	EXPECT_EQ(estimate_auto_reuse_candidate_rowgroups(first_window, seen), 0U);
	EXPECT_TRUE(seen.empty());
	EXPECT_EQ(count_auto_reuse_candidate_rowgroups(first_window, seen), 0U);
	EXPECT_EQ(seen.size(), 3U);
	EXPECT_EQ(estimate_auto_reuse_candidate_rowgroups(second_window, seen), 2U);
	EXPECT_EQ(seen.size(), 3U);
	EXPECT_EQ(count_auto_reuse_candidate_rowgroups(second_window, seen), 2U);
	EXPECT_EQ(seen.size(), 4U);
	EXPECT_EQ(estimate_auto_worksets_for_rowgroups(first_window), 2U);
	EXPECT_EQ(estimate_auto_worksets_for_rowgroups(second_window), 2U);
	EXPECT_EQ(estimate_auto_worksets_for_rowgroups(first_window, /*decode_batch_rowgroups=*/1), 3U);
	EXPECT_EQ(estimate_auto_worksets_for_rowgroups(second_window, /*decode_batch_rowgroups=*/2), 2U);

	std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata> many_shards {
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {0, 0},
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {1, 0},
	    galp::jpeg::JpegDctDeviceRowgroupMetadata {2, 0},
	};
	EXPECT_EQ(estimate_auto_worksets_for_rowgroups(many_shards), 3U);

	std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata> large_single_shard;
	large_single_shard.reserve(65);
	for (uint32_t rowgroup_idx = 0; rowgroup_idx < 65; ++rowgroup_idx) {
		large_single_shard.push_back(galp::jpeg::JpegDctDeviceRowgroupMetadata {0, rowgroup_idx});
	}
	EXPECT_EQ(estimate_auto_worksets_for_rowgroups(large_single_shard), 2U);
	EXPECT_EQ(estimate_auto_worksets_for_rowgroups(large_single_shard, /*decode_batch_rowgroups=*/32), 3U);
}

TEST(JpegDct, AutoPipelineExtraPlanMsDoesNotDoubleCountPreparedPlan) {
	using galp::execution::detail::external_plan_overhead_ms;

	EXPECT_DOUBLE_EQ(external_plan_overhead_ms(/*measured_plan_ms=*/12.5, /*executed_plan_planning_ms=*/7.0), 5.5);
	EXPECT_DOUBLE_EQ(external_plan_overhead_ms(/*measured_plan_ms=*/7.0, /*executed_plan_planning_ms=*/7.0), 0.0);
	EXPECT_DOUBLE_EQ(external_plan_overhead_ms(/*measured_plan_ms=*/6.5, /*executed_plan_planning_ms=*/7.0), 0.0);
}

TEST(JpegDct, SelectedDecodeChunksExpandToPhysicalVectorsForSparseReads) {
	using galp::jpeg::detail::expand_selected_decode_chunks;

	EXPECT_EQ(expand_selected_decode_chunks({0U, 8U}, /*rowgroup_n_vecs=*/12, /*unpack_n_vectors=*/4),
	          (std::vector<uint32_t> {0U, 1U, 2U, 3U, 8U, 9U, 10U, 11U}));
	EXPECT_EQ(expand_selected_decode_chunks({8U}, /*rowgroup_n_vecs=*/10, /*unpack_n_vectors=*/4),
	          (std::vector<uint32_t> {8U, 9U}));
}

TEST(JpegDct, LogicalVectorRemapRetainsSelectedTailBeyondCompactCount) {
	using galp::jpeg::detail::build_logical_to_compact_vector_remap;
	const auto missing = std::numeric_limits<uint32_t>::max();
	EXPECT_EQ(build_logical_to_compact_vector_remap(
	              {0U, 1U, 2U, 3U, 5U, 6U}, /*logical_rowgroup_n_vecs=*/9U, /*unpack_n_vectors=*/1U),
	          (std::vector<uint32_t> {0U, 1U, 2U, 3U, missing, 4U, 5U, missing, missing}));
	EXPECT_EQ(build_logical_to_compact_vector_remap({0U, 8U}, /*logical_rowgroup_n_vecs=*/12U, /*unpack_n_vectors=*/4U),
	          (std::vector<uint32_t> {0U, 1U, 2U, 3U, missing, missing, missing, missing, 4U, 5U, 6U, 7U}));
}

TEST(JpegDct, DevicePrefetchPolicyRequiresStructuralOverlap) {
	using galp::jpeg::detail::JpegDctDeviceRowgroupPlan;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPrefetchConfig;
	using galp::jpeg::detail::JpegDctDeviceShardPlan;
	using galp::jpeg::detail::JpegDctRuntimePolicyDecision;
	using galp::jpeg::detail::plan_jpeg_dct_rowgroup_prefetch_from_hits;

	JpegDctDeviceShardPlan shard;
	shard.shard_id = 7;
	for (uint32_t rowgroup_index = 0; rowgroup_index < 5; ++rowgroup_index) {
		JpegDctDeviceRowgroupPlan rowgroup;
		rowgroup.rowgroup_index          = rowgroup_index;
		rowgroup.runtime_policy.decision = JpegDctRuntimePolicyDecision::kFullRowgroup;
		shard.rowgroups.push_back(std::move(rowgroup));
	}

	JpegDctDeviceRowgroupPrefetchConfig config;
	config.min_decode_batches = 2;

	auto plan = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard, {false, false, false, false, false}, config, 5);
	EXPECT_FALSE(plan.enabled);
	EXPECT_TRUE(plan.rowgroup_indices.empty());
	EXPECT_EQ(plan.use_prefetch_for_position.size(), shard.rowgroups.size());
	EXPECT_EQ(plan.candidate_rowgroup_count, 5U);
	EXPECT_TRUE(plan.disabled_by_small_batch_count);

	plan = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard, {false, false, false, false, false}, config, 4);
	ASSERT_TRUE(plan.enabled);
	EXPECT_EQ(plan.rowgroup_indices, (std::vector<size_t> {0, 1, 2, 3, 4}));
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {true, true, true, true, true}));
	EXPECT_EQ(plan.candidate_rowgroup_count, 5U);
	EXPECT_FALSE(plan.disabled_by_small_batch_count);

	config.enabled = false;
	plan           = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard, {false, false, false, false, false}, config, 4);
	EXPECT_FALSE(plan.enabled);
	EXPECT_TRUE(plan.rowgroup_indices.empty());
	EXPECT_EQ(plan.candidate_rowgroup_count, 5U);
	EXPECT_TRUE(plan.disabled_by_config);
}

TEST(JpegDct, DevicePrefetchPolicySkipsCacheHits) {
	using galp::jpeg::detail::JpegDctDeviceRowgroupPlan;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPrefetchConfig;
	using galp::jpeg::detail::JpegDctDeviceShardPlan;
	using galp::jpeg::detail::JpegDctRuntimePolicyDecision;
	using galp::jpeg::detail::plan_jpeg_dct_rowgroup_prefetch_from_hits;

	JpegDctDeviceShardPlan shard;
	for (uint32_t rowgroup_index = 10; rowgroup_index < 15; ++rowgroup_index) {
		JpegDctDeviceRowgroupPlan rowgroup;
		rowgroup.rowgroup_index          = rowgroup_index;
		rowgroup.runtime_policy.decision = JpegDctRuntimePolicyDecision::kFullRowgroup;
		shard.rowgroups.push_back(std::move(rowgroup));
	}

	JpegDctDeviceRowgroupPrefetchConfig config;
	config.min_decode_batches = 2;
	const auto plan = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard, {true, false, true, false, false}, config, 2);
	ASSERT_TRUE(plan.enabled);
	EXPECT_EQ(plan.rowgroup_indices, (std::vector<size_t> {11, 13, 14}));
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {false, true, false, true, true}));
	EXPECT_EQ(plan.initial_cache_hit_for_position, (std::vector<bool> {true, false, true, false, false}));
	EXPECT_EQ(plan.initial_cache_hit_rowgroup_count, 2U);
	EXPECT_EQ(plan.candidate_rowgroup_count, 3U);
}

TEST(JpegDct, DevicePrefetchPolicyReportsAllHitShard) {
	using galp::jpeg::detail::JpegDctDeviceRowgroupPlan;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPrefetchConfig;
	using galp::jpeg::detail::JpegDctDeviceShardPlan;
	using galp::jpeg::detail::JpegDctRuntimePolicyDecision;
	using galp::jpeg::detail::plan_jpeg_dct_rowgroup_prefetch_from_hits;

	JpegDctDeviceShardPlan shard;
	for (uint32_t rowgroup_index = 0; rowgroup_index < 3; ++rowgroup_index) {
		JpegDctDeviceRowgroupPlan rowgroup;
		rowgroup.rowgroup_index          = rowgroup_index;
		rowgroup.runtime_policy.decision = JpegDctRuntimePolicyDecision::kFullRowgroup;
		shard.rowgroups.push_back(std::move(rowgroup));
	}

	const auto plan =
	    plan_jpeg_dct_rowgroup_prefetch_from_hits(shard, {true, true, true}, {}, /*decode_batch_rowgroups=*/2);
	EXPECT_FALSE(plan.enabled);
	EXPECT_TRUE(plan.disabled_by_all_hits);
	EXPECT_EQ(plan.initial_cache_hit_rowgroup_count, 3U);
	EXPECT_EQ(plan.candidate_rowgroup_count, 0U);
	EXPECT_TRUE(plan.rowgroup_indices.empty());
}

TEST(JpegDct, DevicePrefetchPolicyTreatsDisabledCacheAsMisses) {
	using galp::jpeg::detail::JpegDctDeviceDecodedRowgroupCache;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPlan;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPrefetchConfig;
	using galp::jpeg::detail::JpegDctDeviceShardPlan;
	using galp::jpeg::detail::JpegDctRuntimePolicyDecision;
	using galp::jpeg::detail::plan_jpeg_dct_rowgroup_prefetch;

	JpegDctDeviceShardPlan shard;
	shard.shard_id = 3;
	for (uint32_t rowgroup_index = 0; rowgroup_index < 5; ++rowgroup_index) {
		JpegDctDeviceRowgroupPlan rowgroup;
		rowgroup.rowgroup_index          = rowgroup_index;
		rowgroup.runtime_policy.decision = JpegDctRuntimePolicyDecision::kFullRowgroup;
		shard.rowgroups.push_back(std::move(rowgroup));
	}

	JpegDctDeviceDecodedRowgroupCache cache;
	cache.set_capacity(0);

	JpegDctDeviceRowgroupPrefetchConfig config;
	config.min_decode_batches = 2;
	const auto plan = plan_jpeg_dct_rowgroup_prefetch(shard, &cache, config, /*effective_decode_batch_rowgroups=*/2);
	ASSERT_TRUE(plan.enabled);
	EXPECT_EQ(plan.initial_cache_hit_rowgroup_count, 0U);
	EXPECT_EQ(plan.candidate_rowgroup_count, 5U);
	EXPECT_EQ(plan.rowgroup_indices, (std::vector<size_t> {0, 1, 2, 3, 4}));
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {true, true, true, true, true}));
}

TEST(JpegDct, DeviceDecodedCacheReplacementKeepsResidentBytesStable) {
	using galp::jpeg::JpegDctDeviceCacheStats;
	using galp::jpeg::detail::JpegDctDeviceDecodedRowgroupCache;
	using galp::jpeg::detail::JpegDctDeviceDecodedRowgroupCacheEntry;
	using galp::jpeg::detail::JpegDctDeviceDecodedRowgroupCacheKey;

	const auto make_entry = [](const size_t bytes, const uint64_t last_access) {
		auto entry         = std::make_unique<JpegDctDeviceDecodedRowgroupCacheEntry>();
		entry->bytes       = bytes;
		entry->last_access = last_access;
		return entry;
	};

	JpegDctDeviceDecodedRowgroupCache cache;
	cache.set_capacity(250);
	JpegDctDeviceCacheStats                    stats;
	const JpegDctDeviceDecodedRowgroupCacheKey key_a {1, 10};
	const JpegDctDeviceDecodedRowgroupCacheKey key_b {1, 11};
	const JpegDctDeviceDecodedRowgroupCacheKey key_c {1, 12};

	cache.insert_ready_entry(key_a, make_entry(100, 1), stats);
	cache.insert_ready_entry(key_b, make_entry(100, 2), stats);
	ASSERT_EQ(cache.resident_bytes(), 200U);
	ASSERT_EQ(cache.resident_rowgroups(), 2U);

	cache.insert_ready_entry(key_a, make_entry(100, 3), stats);
	EXPECT_EQ(cache.resident_bytes(), 200U);
	EXPECT_EQ(cache.resident_rowgroups(), 2U);
	EXPECT_EQ(stats.inserts, 3U);
	EXPECT_EQ(stats.evictions, 0U);

	cache.insert_ready_entry(key_c, make_entry(100, 4), stats);
	EXPECT_EQ(cache.resident_bytes(), 200U);
	EXPECT_EQ(cache.resident_rowgroups(), 2U);
	EXPECT_NE(cache.entries.find(key_a), cache.entries.end());
	EXPECT_EQ(cache.entries.find(key_b), cache.entries.end());
	EXPECT_NE(cache.entries.find(key_c), cache.entries.end());
	EXPECT_EQ(stats.evictions, 1U);
	EXPECT_EQ(stats.peak_resident_bytes, 200U);
	EXPECT_EQ(stats.peak_resident_rowgroups, 2U);
	EXPECT_LE(stats.peak_resident_bytes, cache.capacity_bytes());

	cache.insert_ready_entry({1, 13}, make_entry(251, 5), stats);
	EXPECT_EQ(cache.resident_bytes(), 200U);
	EXPECT_EQ(cache.resident_rowgroups(), 2U);
	EXPECT_EQ(stats.inserts, 4U);
	EXPECT_EQ(stats.evictions, 1U);
}

TEST(JpegDct, DevicePrefetchPolicyIncludesSelectedVectorMisses) {
	using galp::jpeg::detail::JpegDctDeviceRowgroupPlan;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPrefetchConfig;
	using galp::jpeg::detail::JpegDctDeviceShardPlan;
	using galp::jpeg::detail::plan_jpeg_dct_rowgroup_prefetch_from_hits;

	JpegDctDeviceShardPlan shard;
	for (uint32_t rowgroup_index = 0; rowgroup_index < 5; ++rowgroup_index) {
		JpegDctDeviceRowgroupPlan rowgroup;
		rowgroup.rowgroup_index = rowgroup_index;
		shard.rowgroups.push_back(std::move(rowgroup));
	}

	JpegDctDeviceRowgroupPrefetchConfig config;
	config.min_decode_batches = 2;
	const auto plan = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard, {false, false, false, false, false}, config, 2);
	EXPECT_TRUE(plan.enabled);
	EXPECT_EQ(plan.rowgroup_indices, (std::vector<size_t> {0U, 1U, 2U, 3U, 4U}));
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {true, true, true, true, true}));
	EXPECT_EQ(plan.candidate_rowgroup_count, 5U);
	EXPECT_EQ(plan.selected_vector_miss_rowgroup_count, 5U);
	EXPECT_FALSE(plan.disabled_by_all_hits);
	EXPECT_FALSE(plan.disabled_by_selected_vector_miss);
}

TEST(JpegDct, DevicePrefetchPolicySkipsRepeatedFullMissAfterDecodeBatch) {
	using galp::jpeg::detail::JpegDctDeviceRowgroupPlan;
	using galp::jpeg::detail::JpegDctDeviceRowgroupPrefetchConfig;
	using galp::jpeg::detail::JpegDctDeviceShardPlan;
	using galp::jpeg::detail::JpegDctRuntimePolicyDecision;
	using galp::jpeg::detail::plan_jpeg_dct_rowgroup_prefetch_from_hits;

	JpegDctDeviceShardPlan shard;
	for (const uint32_t rowgroup_index : {10U, 11U, 10U, 12U, 10U}) {
		JpegDctDeviceRowgroupPlan rowgroup;
		rowgroup.rowgroup_index          = rowgroup_index;
		rowgroup.full_vector_count       = 1;
		rowgroup.runtime_policy.decision = JpegDctRuntimePolicyDecision::kFullRowgroup;
		shard.rowgroups.push_back(std::move(rowgroup));
	}

	JpegDctDeviceRowgroupPrefetchConfig config;
	config.min_decode_batches = 2;
	auto plan                 = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard,
	                                                                      {false, false, false, false, false},
                                                          config,
                                                          /*effective_decode_batch_rowgroups=*/2,
                                                          /*decoded_cache_capacity_bytes=*/1U << 20U);
	ASSERT_TRUE(plan.enabled);
	EXPECT_EQ(plan.rowgroup_indices, (std::vector<size_t> {10, 11, 12}));
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {true, true, false, true, false}));
	EXPECT_EQ(plan.skipped_repeated_for_position, (std::vector<bool> {false, false, true, false, true}));
	EXPECT_EQ(plan.candidate_rowgroup_count, 5U);
	EXPECT_EQ(plan.skipped_repeated_rowgroup_count, 2U);

	plan = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard,
	                                                 {false, false, false, false, false},
	                                                 config,
	                                                 /*effective_decode_batch_rowgroups=*/2,
	                                                 /*decoded_cache_capacity_bytes=*/1);
	ASSERT_TRUE(plan.enabled);
	EXPECT_EQ(plan.rowgroup_indices, (std::vector<size_t> {10, 11, 10, 12, 10}));
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {true, true, true, true, true}));
	EXPECT_EQ(plan.skipped_repeated_rowgroup_count, 0U);

	plan = plan_jpeg_dct_rowgroup_prefetch_from_hits(shard,
	                                                 {false, false, false, false, false},
	                                                 config,
	                                                 /*effective_decode_batch_rowgroups=*/2,
	                                                 /*decoded_cache_capacity_bytes=*/0);
	ASSERT_TRUE(plan.enabled);
	EXPECT_EQ(plan.rowgroup_indices, (std::vector<size_t> {10, 11, 10, 12, 10}));
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {true, true, true, true, true}));
	EXPECT_EQ(plan.skipped_repeated_rowgroup_count, 0U);
}

TEST(JpegDct, SelectedVectorPlanningCompactsAndRemapsMultiVectorChunks) {
	using galp::jpeg::detail::JpegDctDeviceGatherItem;
	using galp::jpeg::detail::JpegDctDeviceProjectionItem;
	using galp::jpeg::detail::remap_items_to_selected_vectors;
	using galp::jpeg::detail::remap_projection_items_to_selected_vectors;
	using galp::jpeg::detail::selected_decode_chunks_fit;
	using galp::jpeg::detail::selected_decode_vectors;

	const auto     vec_values       = static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR);
	const unsigned unpack_n_vectors = 4;
	const std::vector<JpegDctDeviceGatherItem> items {
	    JpegDctDeviceGatherItem {0, 5U * vec_values + 7U, 0},
	    JpegDctDeviceGatherItem {0, 4U * vec_values + 3U, 1},
	    JpegDctDeviceGatherItem {0, 9U * vec_values + 11U, 2},
	    JpegDctDeviceGatherItem {0, 5U * vec_values + 31U, 3},
	};

	const auto selected = selected_decode_vectors(items, /*rowgroup_n_vecs=*/12, unpack_n_vectors);
	ASSERT_EQ(selected.size(), 2U);
	EXPECT_EQ(selected[0], 4U);
	EXPECT_EQ(selected[1], 8U);
	EXPECT_TRUE(selected_decode_chunks_fit(selected, /*rowgroup_n_vecs=*/12, unpack_n_vectors));

	const auto remapped = remap_items_to_selected_vectors(items, selected, unpack_n_vectors);
	ASSERT_EQ(remapped.size(), items.size());
	EXPECT_EQ(remapped[0].row_in_rowgroup, 1U * vec_values + 7U);
	EXPECT_EQ(remapped[1].row_in_rowgroup, 0U * vec_values + 3U);
	EXPECT_EQ(remapped[2].row_in_rowgroup, 5U * vec_values + 11U);
	EXPECT_EQ(remapped[3].row_in_rowgroup, 1U * vec_values + 31U);
	EXPECT_EQ(remapped[2].output_block_index, 2U);

	const std::vector<JpegDctDeviceProjectionItem> projection_items {
	    JpegDctDeviceProjectionItem {0, 5U * vec_values + 7U, 12U, 1U, 2U, 3U, 4U, 1U, 0.25F},
	    JpegDctDeviceProjectionItem {0, 9U * vec_values + 11U, 13U, 5U, 6U, 7U, 8U, 2U, -0.5F},
	};
	const auto remapped_projection =
	    remap_projection_items_to_selected_vectors(projection_items, selected, unpack_n_vectors);
	ASSERT_EQ(remapped_projection.size(), projection_items.size());
	EXPECT_EQ(remapped_projection[0].row_in_rowgroup, 1U * vec_values + 7U);
	EXPECT_EQ(remapped_projection[0].output_block_index, 12U);
	EXPECT_EQ(remapped_projection[0].selected_coefficient_slot, 1U);
	EXPECT_EQ(remapped_projection[0].logical_coefficient_id, 2U);
	EXPECT_EQ(remapped_projection[0].physical_coefficient_column_id, 3U);
	EXPECT_EQ(remapped_projection[0].output_coefficient_id, 4U);
	EXPECT_EQ(remapped_projection[0].output_grid_tensor, 1U);
	EXPECT_FLOAT_EQ(remapped_projection[0].weight, 0.25F);
	EXPECT_EQ(remapped_projection[1].row_in_rowgroup, 5U * vec_values + 11U);
	EXPECT_EQ(remapped_projection[1].output_block_index, 13U);
	EXPECT_FLOAT_EQ(remapped_projection[1].weight, -0.5F);
}

TEST(JpegDct, SelectedVectorPlanningCountsMultiVectorChunks) {
	using galp::jpeg::detail::selected_decode_vector_count;

	EXPECT_EQ(selected_decode_vector_count({4U, 8U}, /*rowgroup_n_vecs=*/12, /*unpack_n_vectors_cfg=*/4), 8U);
	EXPECT_EQ(selected_decode_vector_count({0U, 4U}, /*rowgroup_n_vecs=*/6, /*unpack_n_vectors_cfg=*/4), 6U);
	EXPECT_EQ(selected_decode_vector_count({8U}, /*rowgroup_n_vecs=*/10, /*unpack_n_vectors_cfg=*/4), 4U);
}

TEST(JpegDct, SelectedVectorPlanningRejectsTailChunkOverrun) {
	using galp::jpeg::detail::JpegDctDeviceGatherItem;
	using galp::jpeg::detail::remap_items_to_selected_vectors;
	using galp::jpeg::detail::selected_decode_chunks_fit;
	using galp::jpeg::detail::selected_decode_vectors;

	const auto     vec_values       = static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR);
	const unsigned unpack_n_vectors = 4;
	const std::vector<JpegDctDeviceGatherItem> tail_items {
	    JpegDctDeviceGatherItem {0, 9U * vec_values + 5U, 0},
	};

	const auto selected = selected_decode_vectors(tail_items, /*rowgroup_n_vecs=*/10, unpack_n_vectors);
	ASSERT_EQ(selected.size(), 1U);
	EXPECT_EQ(selected[0], 8U);
	EXPECT_FALSE(selected_decode_chunks_fit(selected, /*rowgroup_n_vecs=*/10, unpack_n_vectors));

	EXPECT_THROW((void)selected_decode_vectors(
	                 {JpegDctDeviceGatherItem {0, 10U * vec_values, 0}}, /*rowgroup_n_vecs=*/10, unpack_n_vectors),
	             std::out_of_range);
	EXPECT_THROW((void)remap_items_to_selected_vectors(tail_items, {}, unpack_n_vectors), std::runtime_error);
}

TEST(JpegDct, ShardedReaderIgnoresLegacyStrictValidationIdAsRagged) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_legacy_validation_" + std::to_string(suffix));
	const auto path0 = dir / "small.jpg";
	const auto path1 = dir / "large.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 8, 8);
	write_test_jpeg(path1, 64, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	const auto small_table                  = galp::jpeg::read_jpeg_dct_file(path0, reader_options);
	const auto large_table                  = galp::jpeg::read_jpeg_dct_file(path1, reader_options);
	ASSERT_EQ(small_table.metadata.semantic_components.size(), 1U);
	ASSERT_EQ(large_table.metadata.semantic_components.size(), 1U);
	const auto absent_block_x = large_table.metadata.semantic_components.front().width_in_blocks - 1;
	ASSERT_GE(absent_block_x, small_table.metadata.semantic_components.front().width_in_blocks);

	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;

	const auto output_dir = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);
	ASSERT_EQ(manifest.shards.size(), 2U);
	EXPECT_EQ(manifest.shards[0].padding_row_count, 0U);

	const auto shard_metadata_path = output_dir / manifest.shards[0].metadata_file_name;
	for (const uint16_t legacy_id : {uint16_t {0}, uint16_t {1}}) {
		std::fstream metadata(shard_metadata_path, std::ios::binary | std::ios::in | std::ios::out);
		ASSERT_TRUE(metadata.good());
		metadata.seekp(14);
		const std::array<char, 2> legacy_validation_id {static_cast<char>(legacy_id & 0xffU),
		                                                static_cast<char>((legacy_id >> 8U) & 0xffU)};
		metadata.write(legacy_validation_id.data(), static_cast<std::streamsize>(legacy_validation_id.size()));

		galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
		const auto                            ref = reader.LocateRow(0, 0, absent_block_x, 0);
		EXPECT_FALSE(ref.present);
		EXPECT_EQ(ref.shard_id, 0U);
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ShardedRaggedMissingBlockReturnsAbsentRowRef) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_ragged_absent_" + std::to_string(suffix));
	const auto path0 = dir / "small.jpg";
	const auto path1 = dir / "large.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 8, 8);
	write_test_jpeg(path1, 64, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
	const auto small_table                  = galp::jpeg::read_jpeg_dct_file(path0, reader_options);
	const auto large_table                  = galp::jpeg::read_jpeg_dct_file(path1, reader_options);
	ASSERT_EQ(small_table.metadata.semantic_components.size(), 1U);
	ASSERT_EQ(large_table.metadata.semantic_components.size(), 1U);
	const auto absent_block_x = large_table.metadata.semantic_components.front().width_in_blocks - 1;
	ASSERT_GE(absent_block_x, small_table.metadata.semantic_components.front().width_in_blocks);

	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;

	const auto output_dir = dir / "out";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            ref = reader.LocateRow(0, 0, absent_block_x, 0);
	EXPECT_FALSE(ref.present);
	EXPECT_EQ(ref.shard_id, 0U);
	EXPECT_EQ(ref.local_image_index, 0U);
	EXPECT_EQ(ref.semantic_slot_id, 0U);
	EXPECT_EQ(ref.block_x, absent_block_x);
	EXPECT_EQ(ref.block_y, 0U);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ImageMajorShardsUseOneDirectlyAddressableRowgroupPerImage) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir   = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_image_major_" + std::to_string(suffix));
	const auto path0 = dir / "input0.jpg";
	const auto path1 = dir / "input1.jpg";
	const auto path2 = dir / "input2.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 32, 24);
	write_test_jpeg(path1, 64, 40);
	write_test_jpeg(path2, 48, 32);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;

	galp::jpeg::JpegDctShardOptions legacy_options;
	legacy_options.shard_images        = 3;
	legacy_options.rowgroup_vectors    = 1;
	legacy_options.rowgroups_per_shard = 256;
	const auto legacy_dir              = dir / "legacy";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    {path0, path1, path2}, legacy_dir, reader_options, legacy_options);

	galp::jpeg::JpegDctShardOptions image_major_options;
	image_major_options.shard_images             = 1;
	image_major_options.rowgroup_vectors         = 1;
	image_major_options.rowgroups_per_shard      = 1;
	image_major_options.physical_layout          = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
	image_major_options.physical_layout_specified = true;
	const auto image_major_dir = dir / "image-major";
	const auto manifest = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    {path0, path1, path2}, image_major_dir, reader_options, image_major_options);

	EXPECT_EQ(manifest.version, 2U);
	ASSERT_EQ(manifest.shards.size(), 3U);
	EXPECT_EQ(manifest.shards.front().rowgroup_count, 1U);
	EXPECT_EQ(read_metadata_header(image_major_dir / manifest.shards.front().metadata_file_name).row_ordering, 2U);
	fastlanes::File descriptor_file(image_major_dir / manifest.shards.front().fls_file_name);
	fastlanes::FileHeader descriptor_header {};
	fastlanes::FileFooter descriptor_footer {};
	fastlanes::FileHeader::Load(descriptor_header, descriptor_file);
	fastlanes::FileFooter::Load(descriptor_footer, descriptor_file);
	ASSERT_TRUE(descriptor_header.settings.inline_footer);
	const auto descriptor = fastlanes::TableDescriptorHandle::FromFileSlice(
	    descriptor_file, descriptor_footer.table_descriptor_offset, descriptor_footer.table_descriptor_size);
	ASSERT_NE(descriptor.Get(), nullptr);
	ASSERT_NE(descriptor.Get()->m_rowgroup_descriptors(), nullptr);
	bool saw_non_ffor_expression = false;
	for (const auto* rowgroup : *descriptor.Get()->m_rowgroup_descriptors()) {
		ASSERT_NE(rowgroup, nullptr);
		ASSERT_NE(rowgroup->m_column_descriptors(), nullptr);
		ASSERT_EQ(rowgroup->m_column_descriptors()->size(), 64U);
		for (const auto* column : *rowgroup->m_column_descriptors()) {
			ASSERT_NE(column, nullptr);
			ASSERT_NE(column->encoding_rpn(), nullptr);
			ASSERT_NE(column->encoding_rpn()->operator_tokens(), nullptr);
			ASSERT_GT(column->encoding_rpn()->operator_tokens()->size(), 0U);
			const auto token = column->encoding_rpn()->operator_tokens()->Get(0);
			EXPECT_TRUE(galp::expression::is_supported_token(token)) << fastlanes::token_to_string(token);
			saw_non_ffor_expression = saw_non_ffor_expression || token != fastlanes::OperatorToken::EXP_FFOR_I16;
		}
	}
	EXPECT_TRUE(saw_non_ffor_expression);

	galp::jpeg::JpegDctShardDatasetReader image_major_reader(image_major_dir / "manifest.bin");
	const auto image0_dc = image_major_reader.LocateRow(0, 0, 0, 0);
	const auto                            image0_next = image_major_reader.LocateRow(0, 0, 1, 0);
	const auto                            image1_dc   = image_major_reader.LocateRow(1, 0, 0, 0);
	ASSERT_TRUE(image0_dc.present);
	ASSERT_TRUE(image0_next.present);
	ASSERT_TRUE(image1_dc.present);
	EXPECT_EQ(image0_dc.fls_rowgroup_index, 0U);
	EXPECT_EQ(image0_next.fls_rowgroup_index, image0_dc.fls_rowgroup_index);
	EXPECT_EQ(image0_next.row_offset_in_block_group, image0_dc.row_offset_in_block_group + 1U);
	EXPECT_EQ(image1_dc.shard_id, 1U);
	EXPECT_EQ(image1_dc.fls_rowgroup_index, 0U);

	galp::jpeg::JpegDctDeviceBatchOptions                  plan_options;
	const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    {2U, {}},
	    {0U, {}},
	};
	const auto preview = image_major_reader.PlanDeviceDctBatch(requests, plan_options);
	EXPECT_EQ(preview.rowgroups.size(), requests.size());

	int         device_count  = 0;
	const auto* run_gpu_tests = std::getenv("GALP_RUN_GPU_TESTS");
	if (run_gpu_tests != nullptr && std::string(run_gpu_tests) == "1" &&
	    cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
		galp::jpeg::JpegDctDeviceBatchOptions transformed_options;
		transformed_options.layout               = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
		transformed_options.grid_transform       = galp::profiles::rgbnomore_val_dct_grid_transform();
		transformed_options.cache_capacity_bytes = 0;
		// Force each differently sized image rowgroup into its own decoded
		// workset. The deterministic transform order is plan-wide, so every
		// flush must derive a permutation and group offsets for its source slice.
		transformed_options.decode_batch_rowgroups = 1;
		auto                 transformed_batch = image_major_reader.ReadDeviceDctBatch(requests, transformed_options);
		std::vector<int16_t> y_host(transformed_batch.y_coefficient_count());
		std::vector<int16_t> cbcr_host(transformed_batch.cbcr_coefficient_count());
		ASSERT_EQ(cudaMemcpy(y_host.data(),
		                     transformed_batch.y_coefficients(),
		                     y_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		ASSERT_EQ(cudaMemcpy(cbcr_host.data(),
		                     transformed_batch.cbcr_coefficients(),
		                     cbcr_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		const auto execution_stats = transformed_batch.execution_stats();
		EXPECT_EQ(execution_stats.rowgroup_count, requests.size());
		EXPECT_EQ(execution_stats.workset_count, requests.size());
		EXPECT_EQ(execution_stats.decode_kernel_launch_count, requests.size());
		EXPECT_EQ(execution_stats.internal_sync_count, requests.size());

		auto unified_options                   = transformed_options;
		unified_options.decode_batch_rowgroups = requests.size();
		auto unified_batch = image_major_reader.ReadDeviceDctBatch(requests, unified_options);
		std::vector<int16_t> unified_y_host(unified_batch.y_coefficient_count());
		std::vector<int16_t> unified_cbcr_host(unified_batch.cbcr_coefficient_count());
		ASSERT_EQ(cudaMemcpy(unified_y_host.data(),
		                     unified_batch.y_coefficients(),
		                     unified_y_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		ASSERT_EQ(cudaMemcpy(unified_cbcr_host.data(),
		                     unified_batch.cbcr_coefficients(),
		                     unified_cbcr_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		EXPECT_EQ(y_host, unified_y_host);
		EXPECT_EQ(cbcr_host, unified_cbcr_host);
		EXPECT_EQ(unified_batch.execution_stats().workset_count, 1U);

		// The legacy spatial-major expanded transform and the image-major planless transform consume the same
		// JPEG coefficients. They may differ by one rounded DCT unit, but the legacy result itself must be stable.
		galp::jpeg::JpegDctShardDatasetReader legacy_device_reader(legacy_dir / "manifest.bin");
		auto                 legacy_batch = legacy_device_reader.ReadDeviceDctBatch(requests, unified_options);
		std::vector<int16_t> legacy_y_host(legacy_batch.y_coefficient_count());
		std::vector<int16_t> legacy_cbcr_host(legacy_batch.cbcr_coefficient_count());
		ASSERT_EQ(cudaMemcpy(legacy_y_host.data(),
		                     legacy_batch.y_coefficients(),
		                     legacy_y_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		ASSERT_EQ(cudaMemcpy(legacy_cbcr_host.data(),
		                     legacy_batch.cbcr_coefficients(),
		                     legacy_cbcr_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		const auto max_abs_difference = [](const std::vector<int16_t>& lhs, const std::vector<int16_t>& rhs) {
			if (lhs.size() != rhs.size()) {
				return std::numeric_limits<int32_t>::max();
			}
			int32_t maximum = 0;
			for (size_t index = 0U; index < lhs.size(); ++index) {
				const auto difference = static_cast<int32_t>(lhs[index]) - static_cast<int32_t>(rhs[index]);
				maximum               = std::max(maximum, difference < 0 ? -difference : difference);
			}
			return maximum;
		};
		EXPECT_LE(max_abs_difference(legacy_y_host, unified_y_host), 1);
		EXPECT_LE(max_abs_difference(legacy_cbcr_host, unified_cbcr_host), 1);

		auto                 repeated_legacy_batch = legacy_device_reader.ReadDeviceDctBatch(requests, unified_options);
		std::vector<int16_t> repeated_legacy_y_host(repeated_legacy_batch.y_coefficient_count());
		std::vector<int16_t> repeated_legacy_cbcr_host(repeated_legacy_batch.cbcr_coefficient_count());
		ASSERT_EQ(cudaMemcpy(repeated_legacy_y_host.data(),
		                     repeated_legacy_batch.y_coefficients(),
		                     repeated_legacy_y_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		ASSERT_EQ(cudaMemcpy(repeated_legacy_cbcr_host.data(),
		                     repeated_legacy_batch.cbcr_coefficients(),
		                     repeated_legacy_cbcr_host.size() * sizeof(int16_t),
		                     cudaMemcpyDeviceToHost),
		          cudaSuccess);
		EXPECT_EQ(repeated_legacy_y_host, legacy_y_host);
		EXPECT_EQ(repeated_legacy_cbcr_host, legacy_cbcr_host);
	}

	galp::jpeg::JpegDctShardDatasetReader legacy_reader(legacy_dir / "manifest.bin");
	const auto block_key = [](const galp::jpeg::MaterializedJpegDctBlock& block) {
		return std::tuple {block.semantic_slot_id, block.block_y, block.block_x};
	};
	for (uint32_t image_id = 0; image_id < 3; ++image_id) {
		auto expected = legacy_reader.MaterializeImageDct(image_id).blocks;
		auto actual   = image_major_reader.MaterializeImageDct(image_id).blocks;
		std::sort(expected.begin(), expected.end(), [&](const auto& lhs, const auto& rhs) {
			return block_key(lhs) < block_key(rhs);
		});
		std::sort(actual.begin(), actual.end(), [&](const auto& lhs, const auto& rhs) {
			return block_key(lhs) < block_key(rhs);
		});
		ASSERT_EQ(actual.size(), expected.size());
		for (size_t block_idx = 0; block_idx < actual.size(); ++block_idx) {
			EXPECT_EQ(block_key(actual[block_idx]), block_key(expected[block_idx]));
			EXPECT_EQ(actual[block_idx].coefficients, expected[block_idx].coefficients);
		}
	}

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, RectangleRankIntervalsMatchBlockEnumerationForEverySpatialOrder) {
	using galp::jpeg::detail::block_order_rank;
	using galp::jpeg::detail::block_order_rectangle_rank_intervals;
	const std::array<galp::jpeg::JpegDctSpatialOrder, 4> orders {
	    galp::jpeg::JpegDctSpatialOrder::kRaster,
	    galp::jpeg::JpegDctSpatialOrder::kTiledRaster32,
	    galp::jpeg::JpegDctSpatialOrder::kZOrder,
	    galp::jpeg::JpegDctSpatialOrder::kTiledZ32,
	};
	struct RectangleCase {
		uint32_t grid_width;
		uint32_t grid_height;
		uint32_t x;
		uint32_t y;
		uint32_t width;
		uint32_t height;
	};
	const std::array<RectangleCase, 8> cases {{
	    {5U, 3U, 0U, 0U, 5U, 3U},
	    {5U, 3U, 1U, 1U, 3U, 1U},
	    {33U, 35U, 0U, 0U, 1U, 35U},
	    {33U, 35U, 30U, 30U, 3U, 5U},
	    {67U, 35U, 17U, 9U, 41U, 23U},
	    {67U, 35U, 31U, 0U, 34U, 35U},
	    {67U, 35U, 66U, 34U, 8U, 8U},
	    {67U, 35U, 67U, 35U, 1U, 1U},
	}};
	for (const auto order : orders) {
		for (const auto& test_case : cases) {
			std::vector<bool> expected(static_cast<size_t>(test_case.grid_width) * test_case.grid_height, false);
			const auto        end_x =
			    std::min<uint64_t>(test_case.grid_width, static_cast<uint64_t>(test_case.x) + test_case.width);
			const auto end_y =
			    std::min<uint64_t>(test_case.grid_height, static_cast<uint64_t>(test_case.y) + test_case.height);
			for (uint32_t y = test_case.y; y < end_y; ++y) {
				for (uint32_t x = test_case.x; x < end_x; ++x) {
					expected.at(block_order_rank(test_case.grid_width, test_case.grid_height, x, y, order)) = true;
				}
			}
			std::vector<bool> actual(expected.size(), false);
			const auto intervals = block_order_rectangle_rank_intervals(test_case.grid_width,
			                                                            test_case.grid_height,
			                                                            test_case.x,
			                                                            test_case.y,
			                                                            test_case.width,
			                                                            test_case.height,
			                                                            order);
			uint64_t previous_end = 0U;
			for (const auto& interval : intervals) {
				EXPECT_LT(interval.begin, interval.end);
				EXPECT_GE(interval.begin, previous_end);
				ASSERT_LE(interval.end, actual.size());
				for (uint64_t rank = interval.begin; rank < interval.end; ++rank) {
					actual[rank] = true;
				}
				previous_end = interval.end;
			}
			EXPECT_EQ(actual, expected);
		}
	}
}

TEST(JpegDct, ImageMajorSpatialOrdersRoundtripWithoutChangingDecodeAtoms) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_spatial_orders_" + std::to_string(suffix));
	const auto jpeg_path = dir / "ragged_tiles.jpg";
	std::filesystem::create_directories(dir);
	// 67x35 luma blocks with 4:2:0 defaults: crosses 32x32 tile boundaries
	// and exercises ragged right/bottom tiles in every component.
	write_test_jpeg(jpeg_path, 536, 280);

	galp::jpeg::JpegDctReaderOptions legacy_reader_options;
	galp::jpeg::JpegDctShardOptions  legacy_shard_options;
	legacy_shard_options.shard_images                  = 1;
	legacy_shard_options.rowgroup_vectors              = 4;
	legacy_shard_options.rowgroups_per_shard           = 256;
	legacy_shard_options.shard_images_specified        = true;
	legacy_shard_options.rowgroup_vectors_specified    = true;
	legacy_shard_options.rowgroups_per_shard_specified = true;
	const auto legacy_dir = dir / "legacy";
	galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    {jpeg_path}, legacy_dir, legacy_reader_options, legacy_shard_options);
	galp::jpeg::JpegDctShardDatasetReader legacy_reader(legacy_dir / "manifest.bin");
	auto expected = legacy_reader.MaterializeImageDct(0).blocks;
	const auto block_key = [](const galp::jpeg::MaterializedJpegDctBlock& block) {
		return std::tuple {block.semantic_slot_id, block.block_y, block.block_x};
	};
	std::sort(expected.begin(), expected.end(), [&](const auto& lhs, const auto& rhs) {
		return block_key(lhs) < block_key(rhs);
	});

	struct OrderCase {
		const char*                      name;
		galp::jpeg::JpegDctSpatialOrder order;
	};
	const std::array<OrderCase, 4> cases {{
	    {"raster", galp::jpeg::JpegDctSpatialOrder::kRaster},
	    {"tiled_raster_32", galp::jpeg::JpegDctSpatialOrder::kTiledRaster32},
	    {"z_order", galp::jpeg::JpegDctSpatialOrder::kZOrder},
	    {"tiled_z_32", galp::jpeg::JpegDctSpatialOrder::kTiledZ32},
	}};
	int device_count = 0;
	const bool has_cuda = cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
	std::filesystem::path raster_metadata_path;

	for (const auto& test_case : cases) {
		galp::jpeg::JpegDctReaderOptions reader_options;
		reader_options.image_major_spatial_order = test_case.order;
		galp::jpeg::JpegDctShardOptions shard_options;
		shard_options.preset                         = galp::jpeg::JpegDctShardPreset::kRandomAccess;
		shard_options.shard_images                  = 1;
		shard_options.rowgroup_vectors              = 4;
		shard_options.rowgroups_per_shard           = 1;
		shard_options.shard_images_specified        = true;
		shard_options.rowgroup_vectors_specified    = true;
		shard_options.rowgroups_per_shard_specified = true;
		const auto output_dir = dir / test_case.name;
		const auto manifest = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
		    {jpeg_path}, output_dir, reader_options, shard_options);
		ASSERT_EQ(manifest.version, 2U);
		ASSERT_EQ(manifest.shards.size(), 1U);
		EXPECT_EQ(manifest.shards.front().rowgroup_count, 1U);
		const auto header = read_metadata_header(output_dir / manifest.shards.front().metadata_file_name);
		EXPECT_NE(header.layout_flags & (1U << 2U), 0U) << test_case.name;
		if (test_case.order == galp::jpeg::JpegDctSpatialOrder::kRaster) {
			raster_metadata_path = output_dir / manifest.shards.front().metadata_file_name;
		}

		galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
		auto actual = reader.MaterializeImageDct(0).blocks;
		std::sort(actual.begin(), actual.end(), [&](const auto& lhs, const auto& rhs) {
			return block_key(lhs) < block_key(rhs);
		});
		ASSERT_EQ(actual.size(), expected.size()) << test_case.name;
		for (size_t block_idx = 0; block_idx < actual.size(); ++block_idx) {
			EXPECT_EQ(block_key(actual[block_idx]), block_key(expected[block_idx])) << test_case.name;
			EXPECT_EQ(actual[block_idx].coefficients, expected[block_idx].coefficients) << test_case.name;
		}

		const auto image_metadata = reader.ImageMetadata(0);
		uint64_t component_offset = 0;
		for (const auto& component : image_metadata.components) {
			if (!component.present) {
				continue;
			}
			for (uint32_t y = 0; y < component.height_in_blocks; ++y) {
				for (uint32_t x = 0; x < component.width_in_blocks; ++x) {
					const auto ref = reader.LocateRow(0, component.semantic_slot_id, x, y);
					ASSERT_TRUE(ref.present) << test_case.name;
					EXPECT_EQ(ref.row_offset_in_block_group,
					          component_offset +
					              galp::jpeg::detail::block_order_rank(
					                  component.width_in_blocks, component.height_in_blocks, x, y, test_case.order))
					    << test_case.name;
				}
			}
			component_offset += static_cast<uint64_t>(component.width_in_blocks) * component.height_in_blocks;
		}

		if (has_cuda) {
			const std::vector<galp::jpeg::JpegDctImageCropRequest> requests {{0U, {}}};
			auto batch = reader.ReadDeviceDctBatch(requests);
			std::vector<int16_t> device_values(batch.coefficient_count());
			ASSERT_EQ(cudaMemcpy(device_values.data(),
			                     batch.device_coefficients(),
			                     device_values.size() * sizeof(int16_t),
			                     cudaMemcpyDeviceToHost),
			          cudaSuccess);
			const auto stats = batch.execution_stats();
			EXPECT_EQ(stats.rowgroup_count, 1U) << test_case.name;
			EXPECT_EQ(stats.workset_count, 1U) << test_case.name;
			EXPECT_EQ(stats.decode_kernel_launch_count, 1U) << test_case.name;
			EXPECT_LE(stats.internal_sync_count, 1U) << test_case.name;
			ASSERT_EQ(batch.block_metadata().size() * batch.coefficients_per_block(), device_values.size());
			std::map<std::tuple<uint32_t, uint32_t, uint32_t>, galp::jpeg::JpegDctCoefficientRow> oracle;
			for (const auto& block : expected) {
				oracle.emplace(block_key(block), block.coefficients);
			}
			for (size_t block_idx = 0; block_idx < batch.block_metadata().size(); ++block_idx) {
				const auto& metadata = batch.block_metadata()[block_idx];
				const auto  it = oracle.find(std::tuple {metadata.semantic_slot_id, metadata.block_y, metadata.block_x});
				ASSERT_NE(it, oracle.end());
				for (size_t coefficient = 0; coefficient < batch.coefficients_per_block(); ++coefficient) {
					EXPECT_EQ(device_values[block_idx * batch.coefficients_per_block() + coefficient],
					          it->second[coefficient])
					    << test_case.name;
				}
			}
		}
	}

	// Remove the explicit-order marker from newly written raster metadata to
	// emulate historical image-major v2 metadata. The new reader must retain
	// the historical raster interpretation rather than guessing another order.
	ASSERT_FALSE(raster_metadata_path.empty());
	const auto old_v2_metadata_path = raster_metadata_path;
	{
		std::fstream metadata(old_v2_metadata_path, std::ios::binary | std::ios::in | std::ios::out);
		ASSERT_TRUE(metadata.good());
		metadata.seekg(16);
		std::array<uint8_t, 2> bytes {};
		metadata.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		uint16_t flags = static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8U));
		flags = static_cast<uint16_t>(flags & ~static_cast<uint16_t>(0x1cU));
		bytes[0] = static_cast<uint8_t>(flags & 0xffU);
		bytes[1] = static_cast<uint8_t>(flags >> 8U);
		metadata.seekp(16);
		metadata.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	}
	galp::jpeg::JpegDctShardDatasetReader old_v2_reader(dir / "raster" / "manifest.bin");
	auto old_v2_actual = old_v2_reader.MaterializeImageDct(0).blocks;
	std::sort(old_v2_actual.begin(), old_v2_actual.end(), [&](const auto& lhs, const auto& rhs) {
		return block_key(lhs) < block_key(rhs);
	});
	ASSERT_EQ(old_v2_actual.size(), expected.size());
	for (size_t block_idx = 0; block_idx < expected.size(); ++block_idx) {
		EXPECT_EQ(old_v2_actual[block_idx].coefficients, expected[block_idx].coefficients);
	}

	std::filesystem::remove_all(dir);
}

class JpegDctMixedExpressions : public ::testing::Test {
protected:
	struct ExpressionSpec {
		fastlanes::OperatorToken token;
		bool                     force;
	};
	using CoefficientMap = std::unordered_map<uint64_t, galp::jpeg::JpegDctCoefficientRow>;

	inline static bool                             has_cuda = false;
	inline static std::filesystem::path            root;
	inline static std::filesystem::path            manifest_path;
	inline static std::vector<CoefficientMap>       cpu_oracle;

	static const std::array<ExpressionSpec, 9>& specs() {
		static const std::array<ExpressionSpec, 9> value {{
		    {fastlanes::OperatorToken::EXP_CONSTANT_I16, false},
		    {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16, true},
		    {fastlanes::OperatorToken::EXP_FFOR_I16, true},
		    {fastlanes::OperatorToken::EXP_RLE_I16_U16, true},
		    {fastlanes::OperatorToken::EXP_FREQUENCY_I16, true},
		    {fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08, true},
		    {fastlanes::OperatorToken::EXP_DELTA_I16, true},
		    {fastlanes::OperatorToken::EXP_DICT_I16_U08, false},
		    {fastlanes::OperatorToken::EXP_DICT_I16_U16, false},
		}};
		return value;
	}

	static uint64_t block_key(const uint32_t semantic_slot_id, const uint32_t block_x, const uint32_t block_y) {
		return (static_cast<uint64_t>(semantic_slot_id) << 56U) | (static_cast<uint64_t>(block_y) << 28U) |
		       static_cast<uint64_t>(block_x);
	}

	static void SetUpTestSuite() {
		int device_count = 0;
		has_cuda = cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
		if (!has_cuda) {
			return;
		}

		const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
		root = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_cross_shard_mixed_" + std::to_string(suffix));
		const auto output_dir = root / "out";
		std::filesystem::create_directories(root);
		std::vector<std::filesystem::path> jpeg_paths;
		jpeg_paths.reserve(specs().size());
		for (size_t image = 0; image < specs().size(); ++image) {
			const auto path = root / ("input_" + std::to_string(image) + ".jpg");
			write_test_jpeg(path, 1024, 1024);
			jpeg_paths.push_back(path);
		}

		galp::jpeg::JpegDctReaderOptions reader_options;
		reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;
		galp::jpeg::JpegDctShardOptions shard_options;
		shard_options.shard_images                  = 1;
		shard_options.rowgroup_vectors              = 24;
		shard_options.rowgroups_per_shard           = 1;
		shard_options.shard_images_specified        = true;
		shard_options.rowgroup_vectors_specified    = true;
		shard_options.rowgroups_per_shard_specified = true;
		shard_options.physical_layout               = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
		shard_options.physical_layout_specified     = true;
		auto manifest =
		    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(jpeg_paths, output_dir, reader_options, shard_options);
		ASSERT_EQ(manifest.version, 2U);
		ASSERT_EQ(manifest.shards.size(), specs().size());
		const auto source_table = galp::jpeg::read_jpeg_dct_file(jpeg_paths.front(), reader_options);
		ASSERT_EQ(source_table.row_count, 24U * 1024U);

		for (size_t shard_id = 0; shard_id < specs().size(); ++shard_id) {
			auto& entry = manifest.shards[shard_id];
			ASSERT_EQ(entry.rowgroup_count, 1U);
			const auto fls_path = output_dir / entry.fls_file_name;
			rewrite_image_major_shard_expression(source_table,
			                                     fls_path,
			                                     static_cast<uint32_t>(shard_id),
			                                     manifest.rowgroup_vectors,
			                                     specs()[shard_id].token,
			                                     specs()[shard_id].force);
			entry.fls_file_size = std::filesystem::file_size(fls_path);
			const auto tokens = read_first_rowgroup_root_tokens(fls_path);
			ASSERT_EQ(tokens.size(), 64U);
			for (const auto token : tokens) {
				EXPECT_TRUE(galp::expression::is_supported_token(token)) << fastlanes::token_to_string(token);
				if (specs()[shard_id].force) {
					EXPECT_EQ(token, specs()[shard_id].token);
				} else if (specs()[shard_id].token == fastlanes::OperatorToken::EXP_CONSTANT_I16) {
					EXPECT_TRUE(token == fastlanes::OperatorToken::EXP_CONSTANT_I08 ||
					            token == fastlanes::OperatorToken::EXP_CONSTANT_I16)
					    << fastlanes::token_to_string(token);
				}
			}
			if (specs()[shard_id].token == fastlanes::OperatorToken::EXP_DICT_I16_U08 ||
			    specs()[shard_id].token == fastlanes::OperatorToken::EXP_DICT_I16_U16) {
				std::ostringstream actual_tokens;
				for (const auto token : tokens) {
					actual_tokens << fastlanes::token_to_string(token) << ' ';
				}
				EXPECT_GT(std::count(tokens.begin(), tokens.end(), specs()[shard_id].token), 0U)
				    << "shard=" << shard_id << " expected=" << fastlanes::token_to_string(specs()[shard_id].token)
				    << " actual=" << actual_tokens.str();
			}
		}
		manifest_path = output_dir / "manifest.bin";
		galp::jpeg::write_jpeg_dct_shard_manifest(manifest, manifest_path);

		galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
		cpu_oracle.clear();
		cpu_oracle.resize(specs().size());
		for (uint32_t image_id = 0; image_id < specs().size(); ++image_id) {
			const auto materialized = reader.MaterializeImageDct(image_id);
			auto&      image_oracle = cpu_oracle[image_id];
			image_oracle.reserve(materialized.blocks.size());
			for (const auto& block : materialized.blocks) {
				image_oracle.emplace(block_key(block.semantic_slot_id, block.block_x, block.block_y),
				                     block.coefficients);
			}
		}
	}

	static void TearDownTestSuite() {
		cpu_oracle.clear();
		if (!root.empty()) {
			std::filesystem::remove_all(root);
		}
		root.clear();
		manifest_path.clear();
	}

	void SetUp() override {
		if (!has_cuda) {
			GTEST_SKIP() << "CUDA device is not available";
		}
	}

	static std::vector<galp::jpeg::JpegDctImageCropRequest> full_requests() {
		const std::array<uint32_t, 9> order {8U, 6U, 0U, 4U, 1U, 7U, 5U, 2U, 3U};
		std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
		requests.reserve(order.size());
		for (const auto image_id : order) {
			requests.push_back({image_id, {}});
		}
		return requests;
	}

	static std::vector<galp::jpeg::JpegDctImageCropRequest> selected_requests() {
		auto requests = full_requests();
		for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
			requests[request_index].source_crop = {static_cast<uint32_t>((request_index % 2U) * 256U),
			                                       static_cast<uint32_t>((request_index % 3U) * 128U),
			                                       512U,
			                                       512U};
		}
		return requests;
	}

	static galp::jpeg::JpegDctDeviceBatchOptions no_cache_options() {
		galp::jpeg::JpegDctDeviceBatchOptions options;
		options.cache_capacity_bytes     = 0;
		options.decode_batch_rowgroups   = 64;
		options.enable_rowgroup_prefetch = false;
		return options;
	}

	static void expect_batch_matches_cpu(const galp::jpeg::JpegDctDeviceBatch&                   batch,
	                                     const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests) {
		ASSERT_EQ(batch.image_layouts().size(), requests.size());
		ASSERT_EQ(batch.coefficients_per_block(), 64U);
		std::vector<int16_t> host(batch.coefficient_count());
		ASSERT_EQ(
		    cudaMemcpy(host.data(), batch.device_coefficients(), batch.coefficient_bytes(), cudaMemcpyDeviceToHost),
		          cudaSuccess);
		for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
			const auto& layout = batch.image_layouts()[request_index];
			ASSERT_EQ(layout.global_image_index, requests[request_index].global_image_index);
			for (size_t local_block = 0; local_block < layout.block_count; ++local_block) {
				const size_t output_block = layout.block_offset + local_block;
				ASSERT_LT(output_block, batch.block_metadata().size());
				const auto& metadata = batch.block_metadata()[output_block];
				ASSERT_EQ(metadata.request_index, request_index);
				ASSERT_EQ(metadata.global_image_index, requests[request_index].global_image_index);
				const auto expected = cpu_oracle[metadata.global_image_index].find(
				    block_key(metadata.semantic_slot_id, metadata.block_x, metadata.block_y));
				ASSERT_NE(expected, cpu_oracle[metadata.global_image_index].end());
				for (size_t coefficient = 0; coefficient < 64U; ++coefficient) {
					ASSERT_EQ(host[output_block * 64U + coefficient], expected->second[coefficient])
					    << "request=" << request_index << " block=" << local_block << " coefficient=" << coefficient;
				}
			}
		}
	}
};

TEST_F(JpegDctMixedExpressions, CrossShardWorkset) {
	galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
	const auto requests = full_requests();
	auto       batch    = reader.ReadDeviceDctBatch(requests, no_cache_options());
	expect_batch_matches_cpu(batch, requests);
	const auto stats = batch.execution_stats();
	EXPECT_EQ(stats.rowgroup_count, specs().size());
	EXPECT_EQ(stats.workset_count, 1U);
	EXPECT_EQ(stats.decode_kernel_launch_count, 1U);
	EXPECT_EQ(stats.internal_sync_count, 1U);
}

TEST_F(JpegDctMixedExpressions, SelectedVectors) {
	galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
	const auto requests = selected_requests();
	auto       batch    = reader.ReadDeviceDctBatch(requests, no_cache_options());
	expect_batch_matches_cpu(batch, requests);
	const auto stats = batch.execution_stats();
	EXPECT_EQ(stats.runtime_policy_selected_rowgroups, specs().size());
	EXPECT_EQ(stats.runtime_policy_full_rowgroups, 0U);
	EXPECT_LT(stats.selected_vector_count, stats.full_vector_count);
	EXPECT_GT(stats.actual_saved_vector_count, 0U);
	EXPECT_EQ(stats.workset_count, 1U);
	EXPECT_EQ(stats.decode_kernel_launch_count, 1U);
	EXPECT_EQ(stats.internal_sync_count, 1U);
}

TEST_F(JpegDctMixedExpressions, CacheReuse) {
	galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
	auto cache_options                 = no_cache_options();
	cache_options.cache_capacity_bytes = 64U * 1024U * 1024U;
	const std::vector<galp::jpeg::JpegDctImageCropRequest> warm_requests {
	    {0U, {}}, {2U, {}}, {4U, {}}, {7U, {}}, {8U, {}}};
	auto warm_batch       = reader.ReadDeviceDctBatch(warm_requests, cache_options);
	const auto warm_cache = warm_batch.cache_stats();
	const auto warm_stats = warm_batch.execution_stats();
	EXPECT_EQ(warm_cache.hits, 0U);
	EXPECT_EQ(warm_cache.misses, warm_requests.size());
	EXPECT_EQ(warm_stats.workset_count, 1U);
	EXPECT_EQ(warm_stats.decode_kernel_launch_count, 1U);

	const auto requests = full_requests();
	auto mixed_batch = reader.ReadDeviceDctBatch(requests, cache_options);
	expect_batch_matches_cpu(mixed_batch, requests);
	const auto mixed_cache = mixed_batch.cache_stats();
	const auto mixed_stats = mixed_batch.execution_stats();
	EXPECT_EQ(mixed_cache.hits, warm_requests.size());
	EXPECT_EQ(mixed_cache.misses, specs().size() - warm_requests.size());
	EXPECT_EQ(mixed_stats.workset_count, 1U);
	EXPECT_EQ(mixed_stats.decode_kernel_launch_count, 1U);
	EXPECT_EQ(mixed_stats.cached_gather_kernel_launch_count, 1U);
	EXPECT_EQ(mixed_stats.internal_sync_count, 1U);

	auto hot_batch = reader.ReadDeviceDctBatch(requests, cache_options);
	expect_batch_matches_cpu(hot_batch, requests);
	const auto hot_cache = hot_batch.cache_stats();
	const auto hot_stats = hot_batch.execution_stats();
	EXPECT_EQ(hot_cache.hits, specs().size());
	EXPECT_EQ(hot_cache.misses, 0U);
	EXPECT_EQ(hot_stats.workset_count, 0U);
	EXPECT_EQ(hot_stats.decode_kernel_launch_count, 0U);
	EXPECT_EQ(hot_stats.cached_gather_kernel_launch_count, 1U);
	EXPECT_EQ(hot_stats.internal_sync_count, 1U);
}

TEST(JpegDct, ImageMajorAutomaticallyFitsLargestImageInOneRowgroup) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_image_major_capacity_" + std::to_string(suffix));
	const auto path = dir / "wide.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path, 12000, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;

	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images              = 1;
	shard_options.rowgroup_vectors          = 1;
	shard_options.rowgroups_per_shard       = 1;
	shard_options.physical_layout           = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
	shard_options.physical_layout_specified = true;

	const auto output_dir = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path}, output_dir, reader_options, shard_options);

	EXPECT_EQ(manifest.rowgroup_vectors, 2U);
	ASSERT_EQ(manifest.shards.size(), 1U);
	EXPECT_EQ(manifest.shards.front().rowgroup_count, 1U);
	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	EXPECT_TRUE(reader.LocateRow(0, 0, 1499, 0).present);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, PublicShardPresetAppliesWithoutCliExpansion) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_preset_" + std::to_string(suffix));
	const auto path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;

	galp::jpeg::JpegDctShardOptions throughput_options;
	throughput_options.preset      = galp::jpeg::JpegDctShardPreset::kThroughput;
	const auto throughput_manifest = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    {path}, dir / "throughput", reader_options, throughput_options);
	EXPECT_EQ(throughput_manifest.rowgroup_vectors, 256U);

	galp::jpeg::JpegDctShardOptions crop_options;
	crop_options.preset = galp::jpeg::JpegDctShardPreset::kCropLatency;
	const auto crop_manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path}, dir / "crop", reader_options, crop_options);
	EXPECT_EQ(crop_manifest.rowgroup_vectors, 64U);

	galp::jpeg::JpegDctShardOptions random_access_options;
	random_access_options.preset = galp::jpeg::JpegDctShardPreset::kRandomAccess;
	const auto random_access_manifest = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    {path}, dir / "random-access", reader_options, random_access_options);
	EXPECT_EQ(random_access_manifest.version, 2U);
	EXPECT_EQ(random_access_manifest.rowgroups_per_shard, 8192U);
	ASSERT_EQ(random_access_manifest.shards.size(), 1U);
	EXPECT_EQ(random_access_manifest.shards.front().rowgroup_count, 1U);

	galp::jpeg::JpegDctShardOptions override_options;
	override_options.preset                     = galp::jpeg::JpegDctShardPreset::kThroughput;
	override_options.rowgroup_vectors           = 128;
	override_options.rowgroup_vectors_specified = true;
	const auto override_manifest                = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
        {path}, dir / "override", reader_options, override_options);
	EXPECT_EQ(override_manifest.rowgroup_vectors, 128U);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ShardedWriterSplitsByRowgroupLimit) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_rowgroups_" + std::to_string(suffix));
	const auto path0 = dir / "wide0.jpg";
	const auto path1 = dir / "wide1.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 4800, 8);
	write_test_jpeg(path1, 4800, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;

	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 2;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 1;

	const auto output_dir = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);

	ASSERT_EQ(manifest.shards.size(), 2U);
	EXPECT_EQ(manifest.shards[0].first_global_image_index, 0U);
	EXPECT_EQ(manifest.shards[1].first_global_image_index, 1U);
	EXPECT_EQ(manifest.shards[0].image_count, 1U);
	EXPECT_EQ(manifest.shards[1].image_count, 1U);
	EXPECT_LE(manifest.shards[0].rowgroup_count, 1U);
	EXPECT_LE(manifest.shards[1].rowgroup_count, 1U);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, MalformedInputThrows) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto path =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_malformed_" + std::to_string(suffix) + ".jpg");
	{
		std::ofstream out(path, std::ios::binary);
		out << "not a valid jpeg";
	}

	EXPECT_THROW(galp::jpeg::read_jpeg_dct_file(path), std::runtime_error);

	std::filesystem::remove(path);
}

} // namespace
