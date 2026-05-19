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

void write_test_jpeg(const std::filesystem::path& path) {
	FilePtr file(std::fopen(path.string().c_str(), "wb"));
	if (!file) {
		throw std::runtime_error("failed to create test JPEG");
	}

	jpeg_compress_struct cinfo {};
	jpeg_error_mgr       jerr {};
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, file.get());

	constexpr int kWidth   = 16;
	constexpr int kHeight  = 8;
	cinfo.image_width      = kWidth;
	cinfo.image_height     = kHeight;
	cinfo.input_components = 3;
	cinfo.in_color_space   = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 90, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	std::array<unsigned char, kWidth * 3> row {};
	while (cinfo.next_scanline < cinfo.image_height) {
		for (int x = 0; x < kWidth; ++x) {
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
