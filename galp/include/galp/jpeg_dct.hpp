#ifndef GALP_JPEG_DCT_HPP
#define GALP_JPEG_DCT_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace galp::jpeg {

enum class JpegComponentMode {
	kAllComponents,
	kSingleComponent,
};

enum class JpegDatasetValidationMode {
	kRequireSameComponentGrids,
	kPadToMaxComponentGrids,
	kRaggedBlockMajor,
};

struct JpegDctReaderOptions {
	JpegComponentMode          component_mode           = JpegComponentMode::kAllComponents;
	int                        selected_component_index = -1;
	JpegDatasetValidationMode  validation_mode = JpegDatasetValidationMode::kRaggedBlockMajor;
	bool                       use_zigzag_columns       = true;
	bool                       use_z_curve_block_order  = true;
};

struct JpegComponentMetadata {
	size_t   component_index = 0;
	int      component_id    = 0;
	uint32_t width_in_blocks = 0;
	uint32_t height_in_blocks = 0;
	uint32_t padded_width_in_blocks = 0;
	uint32_t padded_height_in_blocks = 0;
	int      h_samp_factor   = 0;
	int      v_samp_factor   = 0;
	bool     present         = true;
};

struct JpegImageMetadata {
	std::filesystem::path              source_path;
	uint32_t                           image_width      = 0;
	uint32_t                           image_height     = 0;
	int                                jpeg_color_space = 0;
	bool                               progressive      = false;
	std::vector<JpegComponentMetadata> components;
};

enum class JpegDctRowOrdering {
	kSingleImageComponentMajorBlockMajor,
	kDatasetComponentMajorBlockMajorImageMinor,
};

struct JpegDctDatasetMetadata {
	std::vector<JpegImageMetadata> images;
	JpegDctRowOrdering            row_ordering = JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	JpegDatasetValidationMode     validation_mode = JpegDatasetValidationMode::kRequireSameComponentGrids;
	bool                          zigzag_columns = true;
	bool                          z_curve_block_order = true;
	size_t                        image_count = 0;
};

struct JpegDctTable {
	size_t                               row_count = 0;
	size_t                               real_row_count = 0;
	size_t                               padding_row_count = 0;
	size_t                               block_group_count = 0;
	std::array<std::vector<int16_t>, 64> columns;
	JpegDctDatasetMetadata               metadata;
};

JpegDctTable read_jpeg_dct_file(const std::filesystem::path& path, const JpegDctReaderOptions& options = {});

JpegDctTable read_jpeg_dct_dataset(const std::vector<std::filesystem::path>& paths,
                                   const JpegDctReaderOptions&               options = {});

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata& metadata, const std::filesystem::path& output_path);

void compress_jpeg_dct_to_fls(const JpegDctTable&           table,
                              const std::filesystem::path&  fls_output_path,
                              const std::filesystem::path&  metadata_output_path);

void compress_jpeg_dct_file_to_fls(const std::filesystem::path& jpeg_path,
                                   const std::filesystem::path& fls_output_path,
                                   const std::filesystem::path& metadata_output_path,
                                   const JpegDctReaderOptions&  options = {});

void compress_jpeg_dct_dataset_to_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                      const std::filesystem::path&              fls_output_path,
                                      const std::filesystem::path&              metadata_output_path,
                                      const JpegDctReaderOptions&               options = {});

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_HPP
