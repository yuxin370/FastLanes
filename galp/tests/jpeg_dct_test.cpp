#include "galp/jpeg_dct.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <jpeglib.h>
#include <memory>
#include <stdexcept>
#include <string>
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

void write_test_jpeg(const std::filesystem::path& path, const int width = 16, const int height = 8) {
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
	cinfo.input_components = 3;
	cinfo.in_color_space   = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 90, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	std::vector<unsigned char> row(static_cast<size_t>(width) * 3);
	while (cinfo.next_scanline < cinfo.image_height) {
		for (int x = 0; x < width; ++x) {
			row[static_cast<size_t>(x) * 3 + 0] = static_cast<unsigned char>(x * 11);
			row[static_cast<size_t>(x) * 3 + 1] = static_cast<unsigned char>(cinfo.next_scanline * 19);
			row[static_cast<size_t>(x) * 3 + 2] = static_cast<unsigned char>(x * 7 + cinfo.next_scanline);
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

TEST(JpegDct, PublicAggregatesPreserveLegacyPositionalInitialization) {
	galp::jpeg::JpegDctReaderOptions options {galp::jpeg::JpegComponentMode::kSingleComponent,
	                                          0,
	                                          galp::jpeg::JpegDatasetValidationMode::kPadToMaxComponentGrids,
	                                          false,
	                                          true};
	EXPECT_FALSE(options.use_zigzag_columns);
	EXPECT_TRUE(options.use_z_curve_block_order);

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
	EXPECT_TRUE(std::filesystem::exists(output_dir / "shard_000000.fls"));
	EXPECT_TRUE(std::filesystem::exists(output_dir / "shard_000000.meta.bin"));
	EXPECT_TRUE(std::filesystem::exists(output_dir / "shard_000001.fls"));
	EXPECT_TRUE(std::filesystem::exists(output_dir / "shard_000001.meta.bin"));

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
	EXPECT_EQ(read_metadata_header(output_dir / "shard_000000.meta.bin").magic, sectioned_magic);

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

TEST(JpegDct, ShardedStrictValidationUsesGlobalDatasetLayout) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_strict_" + std::to_string(suffix));
	const auto path0 = dir / "small.jpg";
	const auto path1 = dir / "large.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 8, 8);
	write_test_jpeg(path1, 64, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kRequireSameComponentGrids;

	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;

	EXPECT_THROW(galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	                 {path0, path1}, dir / "out", reader_options, shard_options),
	             std::runtime_error);

	std::filesystem::remove_all(dir);
}

TEST(JpegDct, ShardedPadModeUsesGlobalMaxGridForEveryShard) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir    = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_pad_" + std::to_string(suffix));
	const auto path0  = dir / "small.jpg";
	const auto path1  = dir / "large.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path0, 8, 8);
	write_test_jpeg(path1, 64, 8);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.component_mode           = galp::jpeg::JpegComponentMode::kSingleComponent;
	reader_options.selected_component_index = 0;
	reader_options.validation_mode          = galp::jpeg::JpegDatasetValidationMode::kPadToMaxComponentGrids;
	const auto small_table                  = galp::jpeg::read_jpeg_dct_file(path0, reader_options);
	const auto large_table                  = galp::jpeg::read_jpeg_dct_file(path1, reader_options);
	ASSERT_EQ(small_table.metadata.semantic_components.size(), 1U);
	ASSERT_EQ(large_table.metadata.semantic_components.size(), 1U);
	const auto padded_block_x = large_table.metadata.semantic_components.front().width_in_blocks - 1;
	ASSERT_GE(padded_block_x, small_table.metadata.semantic_components.front().width_in_blocks);

	galp::jpeg::JpegDctShardOptions shard_options;
	shard_options.shard_images        = 1;
	shard_options.rowgroup_vectors    = 1;
	shard_options.rowgroups_per_shard = 256;

	const auto output_dir = dir / "out";
	const auto manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path0, path1}, output_dir, reader_options, shard_options);
	ASSERT_EQ(manifest.shards.size(), 2U);
	EXPECT_GT(manifest.shards[0].padding_row_count, 0U);

	galp::jpeg::JpegDctShardDatasetReader reader(output_dir / "manifest.bin");
	const auto                            ref = reader.LocateRow(0, 0, padded_block_x, 0);
	EXPECT_TRUE(ref.present);
	EXPECT_EQ(ref.shard_id, 0U);
	EXPECT_EQ(ref.row_offset_in_block_group, 0U);

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

TEST(JpegDct, PublicShardPresetAppliesWithoutCliExpansion) {
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir =
	    std::filesystem::temp_directory_path() / ("galp_jpeg_dct_shards_preset_" + std::to_string(suffix));
	const auto path = dir / "input.jpg";
	std::filesystem::create_directories(dir);
	write_test_jpeg(path);

	galp::jpeg::JpegDctReaderOptions reader_options;
	reader_options.validation_mode = galp::jpeg::JpegDatasetValidationMode::kRaggedBlockMajor;

	galp::jpeg::JpegDctShardOptions throughput_options;
	throughput_options.preset = galp::jpeg::JpegDctShardPreset::kThroughput;
	const auto throughput_manifest = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	    {path}, dir / "throughput", reader_options, throughput_options);
	EXPECT_EQ(throughput_manifest.rowgroup_vectors, 256U);

	galp::jpeg::JpegDctShardOptions crop_options;
	crop_options.preset = galp::jpeg::JpegDctShardPreset::kCropLatency;
	const auto crop_manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls({path}, dir / "crop", reader_options, crop_options);
	EXPECT_EQ(crop_manifest.rowgroup_vectors, 64U);

	galp::jpeg::JpegDctShardOptions override_options;
	override_options.preset                     = galp::jpeg::JpegDctShardPreset::kThroughput;
	override_options.rowgroup_vectors           = 128;
	override_options.rowgroup_vectors_specified = true;
	const auto override_manifest = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
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
