#ifndef GALP_JPEG_DCT_FORMAT_HPP
#define GALP_JPEG_DCT_FORMAT_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <vector>

namespace galp::jpeg {

// Physical order of spatial DCT blocks inside each component of an image-major
// record. Tiles are measured in DCT blocks, not pixels. The image-major rowgroup
// remains the storage/decode atom for every mode.
enum class JpegDctSpatialOrder {
	kRaster,
	kTiledRaster32,
	kZOrder,
	kTiledZ32,
};

enum class JpegComponentMode {
	kAllComponents,
	kSingleComponent,
};

enum class JpegDatasetValidationMode {
	kRequireSameComponentGrids,
	kPadToMaxComponentGrids,
	kRaggedBlockMajor,
};

enum class JpegMetadataProfile {
	kDctDatasetOnly,
	kReconstructableJpeg,
	kPreserveOriginalMarkers,
};

enum class JpegCompressionPartitionPolicy {
	kBySemanticSlot,
	kByEncodingProfile,
};

enum class JpegDctCoefficientEncoding {
	kDense64FastLanes,
	kExpCrossRleI16,
	kDcDeltaDense64FastLanes,
	kDcDeltaAcSparseRle,
	kJpegLikeRunLength,
};

struct JpegDctReaderOptions {
	JpegComponentMode              component_mode               = JpegComponentMode::kAllComponents;
	int                            selected_component_index     = -1;
	JpegDatasetValidationMode      validation_mode              = JpegDatasetValidationMode::kRaggedBlockMajor;
	bool                           use_zigzag_columns           = true;
	bool                           use_z_curve_block_order      = true;
	JpegCompressionPartitionPolicy compression_partition_policy = JpegCompressionPartitionPolicy::kBySemanticSlot;
	JpegDctCoefficientEncoding     coefficient_encoding         = JpegDctCoefficientEncoding::kDense64FastLanes;
	bool                           capture_metadata_markers     = false;
	// Kept last to preserve legacy positional aggregate initialization.
	JpegDctSpatialOrder image_major_spatial_order = JpegDctSpatialOrder::kTiledZ32;
};

struct JpegDctMetadataWriterOptions {
	JpegMetadataProfile profile = JpegMetadataProfile::kDctDatasetOnly;
};

struct JpegComponentMetadata {
	size_t   component_index         = 0;
	int      component_id            = 0;
	uint32_t width_in_blocks         = 0;
	uint32_t height_in_blocks        = 0;
	uint32_t padded_width_in_blocks  = 0;
	uint32_t padded_height_in_blocks = 0;
	int      h_samp_factor           = 0;
	int      v_samp_factor           = 0;
	bool     present                 = true;
	uint32_t semantic_slot_id        = 0;
	size_t   local_component_index   = 0;
	int      quant_tbl_no            = -1;
	uint64_t quant_table_fingerprint = 0;
	uint32_t encoding_profile_id     = std::numeric_limits<uint32_t>::max();
};

struct JpegQuantTableMetadata {
	uint8_t                  table_id = 0;
	std::array<uint16_t, 64> values {};
};

struct JpegMarkerMetadata {
	uint8_t              marker = 0;
	std::vector<uint8_t> payload;
};

struct JpegEncodingProfileMetadata {
	uint32_t                 profile_id              = 0;
	int                      h_samp_factor           = 0;
	int                      v_samp_factor           = 0;
	int                      quant_tbl_no            = -1;
	uint64_t                 quant_table_fingerprint = 0;
	std::array<uint16_t, 64> quant_table_values {};
};

struct JpegImageMetadata {
	std::filesystem::path               source_path;
	uint32_t                            image_width      = 0;
	uint32_t                            image_height     = 0;
	int                                 jpeg_color_space = 0;
	bool                                progressive      = false;
	std::vector<JpegComponentMetadata>  components;
	uint8_t                             data_precision = 8;
	uint32_t                            warning_count  = 0;
	std::vector<JpegQuantTableMetadata> quant_tables;
	std::vector<JpegMarkerMetadata>     markers;
};

enum class JpegDctRowOrdering {
	kSingleImageComponentMajorBlockMajor,
	kDatasetComponentMajorBlockMajorImageMinor,
	kDatasetImageMajorComponentBlockMajor,
};

enum class JpegDctPhysicalLayout {
	kSpatialMajorImageMinor,
	kImageMajor,
	// Experimental manifest-v3 upper-bound layout: image-major rows remain
	// contiguous, but every FastLanes vector is an independently addressable
	// rowgroup. Production datasets should normally use kImageMajor together
	// with the sparse vector-range reader.
	kImageMajorVectorRowgroups,
};

[[nodiscard]] constexpr bool is_image_major_physical_layout(const JpegDctPhysicalLayout layout) noexcept {
	return layout == JpegDctPhysicalLayout::kImageMajor ||
	       layout == JpegDctPhysicalLayout::kImageMajorVectorRowgroups;
}

struct JpegDctBlockGroupIndex {
	uint32_t semantic_slot_id      = 0;
	uint32_t z_order_index         = 0;
	uint32_t block_x               = 0;
	uint32_t block_y               = 0;
	uint64_t row_start             = 0;
	uint32_t row_count             = 0;
	uint32_t fls_rowgroup_index    = 0;
	uint32_t row_start_in_rowgroup = 0;
};

struct JpegDctImageGroupIndex {
	uint32_t local_image_index     = 0;
	uint64_t row_start             = 0;
	uint32_t row_count             = 0;
	uint32_t fls_rowgroup_index    = 0;
	uint32_t row_start_in_rowgroup = 0;
};

struct JpegDctDatasetMetadata {
	std::vector<JpegImageMetadata>     images;
	JpegDctRowOrdering                 row_ordering    = JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	JpegDatasetValidationMode          validation_mode = JpegDatasetValidationMode::kRaggedBlockMajor;
	bool                               zigzag_columns  = true;
	bool                               z_curve_block_order = true;
	size_t                             image_count         = 0;
	std::vector<JpegComponentMetadata> semantic_components;
	std::vector<JpegEncodingProfileMetadata> encoding_profiles;
	std::vector<JpegDctBlockGroupIndex>      block_group_index;
	JpegCompressionPartitionPolicy      compression_partition_policy = JpegCompressionPartitionPolicy::kBySemanticSlot;
	JpegDctCoefficientEncoding          coefficient_encoding         = JpegDctCoefficientEncoding::kDense64FastLanes;
	std::vector<JpegDctImageGroupIndex> image_group_index;
	JpegDctSpatialOrder                 image_major_spatial_order = JpegDctSpatialOrder::kRaster;
};

struct JpegDctTable {
	size_t                               row_count         = 0;
	size_t                               real_row_count    = 0;
	size_t                               padding_row_count = 0;
	size_t                               block_group_count = 0;
	std::array<std::vector<int16_t>, 64> columns;
	JpegDctDatasetMetadata               metadata;
	std::vector<uint64_t>                rowgroup_n_tuples;
};

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_FORMAT_HPP
