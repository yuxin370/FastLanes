#include "galp/jpeg_dct.hpp"
#include "galp_tools/benchmark_support/pipeline.cuh"
#include "jpeg/jpeg_dct_device.cuh"
#include <array>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <jpeglib.h>
#include <memory>
#include <set>
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

	galp::jpeg::JpegDctDeviceBatchOptions default_batch_options;
	EXPECT_EQ(default_batch_options.layout, galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff);
	EXPECT_EQ(default_batch_options.cache_capacity_bytes, 0U);
	EXPECT_EQ(default_batch_options.decode_batch_rowgroups, galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups);
	EXPECT_TRUE(default_batch_options.enable_rowgroup_prefetch);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_depth, galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_workers,
	          galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers);
	EXPECT_EQ(default_batch_options.rowgroup_prefetch_min_decode_batches,
	          galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches);

	galp::jpeg::JpegDctDeviceBatchOptions legacy_batch_options {
	    galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff, 4096U};
	EXPECT_EQ(legacy_batch_options.cache_capacity_bytes, 4096U);
	EXPECT_EQ(legacy_batch_options.decode_batch_rowgroups, galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups);
	EXPECT_TRUE(legacy_batch_options.enable_rowgroup_prefetch);
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
	ASSERT_EQ(batch.coefficient_count(), batch.block_count() * 64U);
	ASSERT_EQ(batch.image_layouts().size(), requests.size());
	ASSERT_EQ(batch.block_metadata().size(), batch.block_count());
	ASSERT_EQ(batch.rowgroups().size(), batch.rowgroup_count());
	EXPECT_GT(batch.rowgroup_count(), 0U);
	EXPECT_EQ(batch.image_layouts()[0].block_offset, 0U);
	EXPECT_GT(batch.image_layouts()[0].block_count, 0U);
	EXPECT_EQ(batch.image_layouts()[1].block_offset, batch.image_layouts()[0].block_count);
	EXPECT_GT(batch.image_layouts()[1].block_count, 0U);

	std::vector<int16_t> host(batch.coefficient_count());
	ASSERT_EQ(cudaMemcpy(host.data(), batch.device_coefficients(), batch.coefficient_bytes(), cudaMemcpyDeviceToHost),
	          cudaSuccess);

	for (size_t output_block_idx = 0; output_block_idx < batch.block_metadata().size(); ++output_block_idx) {
		const auto& meta         = batch.block_metadata()[output_block_idx];
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
		for (size_t coeff_idx = 0; coeff_idx < expected->coefficients.size(); ++coeff_idx) {
			EXPECT_EQ(host[output_block_idx * 64U + coeff_idx], expected->coefficients[coeff_idx]);
		}
	}

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

TEST(JpegDct, AutoPipelinePolicyAvoidsTinyRowgroupOverhead) {
	using galp::execution::detail::AutoPipelinePolicyReason;
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
		EXPECT_FALSE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
		EXPECT_EQ(policy.estimated_pushdown_worksets, 1U);
		EXPECT_EQ(policy.estimated_full_worksets, 1U);
		EXPECT_EQ(policy.estimated_pushdown_gather_items, 768U);
		EXPECT_EQ(policy.estimated_full_gather_items, 3072U);
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
		EXPECT_FALSE(policy->use_pushdown);
		EXPECT_EQ(policy->reason_code, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
		EXPECT_NE(policy->reason.find("metadata_fast=1"), std::string::npos);
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/32000, /*full_blocks=*/100000, /*image_count=*/128);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/1000, /*full_blocks=*/200000, /*image_count=*/128);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/512, /*full_blocks=*/8192, /*image_count=*/256);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/80000, /*full_blocks=*/100000, /*image_count=*/2000);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/10000, /*full_blocks=*/20000, /*image_count=*/400);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/102400, /*full_blocks=*/128000, /*image_count=*/128);
		EXPECT_FALSE(policy.has_value());
	}
	{
		const auto policy = choose_auto_pipeline_policy_from_block_estimate(
		    /*selected_blocks=*/3072, /*full_blocks=*/12288, /*image_count=*/128);
		EXPECT_FALSE(policy.has_value());
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
		const auto policy = choose_auto_pipeline_policy_from_estimates(/*selected_blocks=*/10000,
		                                                               /*full_blocks=*/20000,
		                                                               /*touched_rowgroups=*/2,
		                                                               /*full_rowgroups=*/10,
		                                                               /*selected_vectors=*/6,
		                                                               /*full_vectors=*/10,
		                                                               /*estimated_pushdown_worksets=*/1,
		                                                               /*estimated_full_worksets=*/2);
		EXPECT_TRUE(policy.use_pushdown);
		EXPECT_EQ(policy.reason_code, AutoPipelinePolicyReason::CropSavesEnoughBlocks);
		EXPECT_EQ(policy.estimated_pushdown_worksets, 1U);
		EXPECT_EQ(policy.estimated_full_worksets, 2U);
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
}

TEST(JpegDct, AutoPipelineReuseCandidateCountsRepeatedPreviewRowgroups) {
	using galp::execution::detail::count_auto_reuse_candidate_rowgroups;
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

	EXPECT_EQ(count_auto_reuse_candidate_rowgroups(first_window, seen), 0U);
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
	JpegDctDeviceCacheStats stats;
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
}

TEST(JpegDct, DevicePrefetchPolicySkipsSelectedVectorMisses) {
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
	EXPECT_FALSE(plan.enabled);
	EXPECT_TRUE(plan.rowgroup_indices.empty());
	EXPECT_EQ(plan.use_prefetch_for_position, (std::vector<bool> {false, false, false, false, false}));
	EXPECT_EQ(plan.candidate_rowgroup_count, 0U);
	EXPECT_EQ(plan.selected_vector_miss_rowgroup_count, 5U);
	EXPECT_FALSE(plan.disabled_by_all_hits);
	EXPECT_TRUE(plan.disabled_by_selected_vector_miss);
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
	using galp::jpeg::detail::remap_items_to_selected_vectors;
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
