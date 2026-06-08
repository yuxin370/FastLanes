#ifndef GALP_JPEG_DCT_HPP
#define GALP_JPEG_DCT_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
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
};

struct JpegDctMetadataWriterOptions {
	// Default writes only the metadata needed to interpret the DCT coefficient table.
	// Use kReconstructableJpeg when JPEG reconstruction fields such as image dimensions
	// and quantization tables must be persisted.
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
};

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

struct JpegDctDatasetMetadata {
	std::vector<JpegImageMetadata>     images;
	JpegDctRowOrdering                 row_ordering    = JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	JpegDatasetValidationMode          validation_mode = JpegDatasetValidationMode::kRequireSameComponentGrids;
	bool                               zigzag_columns  = true;
	bool                               z_curve_block_order = true;
	size_t                             image_count         = 0;
	std::vector<JpegComponentMetadata> semantic_components;
	std::vector<JpegEncodingProfileMetadata> encoding_profiles;
	std::vector<JpegDctBlockGroupIndex>      block_group_index;
	JpegCompressionPartitionPolicy compression_partition_policy = JpegCompressionPartitionPolicy::kBySemanticSlot;
	JpegDctCoefficientEncoding     coefficient_encoding         = JpegDctCoefficientEncoding::kDense64FastLanes;
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

enum class JpegDctShardPreset {
	kCropLatency,
	kBalanced,
	kThroughput,
};

enum class JpegDctDeviceLayout {
	kImageMajorComponentBlockCoeff,
};

struct JpegDctShardOptions {
	size_t             shard_images                  = 8192;
	uint32_t           rowgroup_vectors              = 128;
	uint32_t           rowgroups_per_shard           = 256;
	JpegDctShardPreset preset                        = JpegDctShardPreset::kBalanced;
	bool               shard_images_specified        = false;
	bool               rowgroup_vectors_specified    = false;
	bool               rowgroups_per_shard_specified = false;
};

struct JpegDctShardManifestEntry {
	uint32_t    shard_id                 = 0;
	uint64_t    first_global_image_index = 0;
	uint32_t    image_count              = 0;
	uint64_t    real_row_count           = 0;
	uint64_t    padding_row_count        = 0;
	uint64_t    physical_row_count       = 0;
	uint32_t    rowgroup_count           = 0;
	uint32_t    block_group_count        = 0;
	uint64_t    fls_file_size            = 0;
	uint64_t    metadata_file_size       = 0;
	std::string fls_file_name;
	std::string metadata_file_name;
};

struct JpegDctShardManifest {
	uint32_t                               version             = 1;
	JpegDatasetValidationMode              policy              = JpegDatasetValidationMode::kRaggedBlockMajor;
	uint32_t                               rowgroup_vectors    = 128;
	uint32_t                               rowgroups_per_shard = 256;
	uint64_t                               image_count         = 0;
	std::vector<JpegDctShardManifestEntry> shards;
};

using JpegDctCoefficientRow = std::array<int16_t, 64>;

struct JpegDctRowRef {
	uint32_t shard_id                  = 0;
	uint32_t local_image_index         = 0;
	uint32_t semantic_slot_id          = 0;
	uint32_t block_x                   = 0;
	uint32_t block_y                   = 0;
	uint64_t physical_row_index        = 0;
	uint32_t fls_rowgroup_index        = 0;
	uint32_t row_start_in_rowgroup     = 0;
	uint32_t row_offset_in_block_group = 0;
	bool     present                   = false;
};

struct JpegDctBlockGroup {
	JpegDctBlockGroupIndex             index;
	std::vector<JpegDctCoefficientRow> rows;
};

struct MaterializedJpegDctBlock {
	uint32_t              semantic_slot_id = 0;
	uint32_t              block_x          = 0;
	uint32_t              block_y          = 0;
	JpegDctCoefficientRow coefficients {};
};

struct MaterializedJpegDctImage {
	uint32_t                              global_image_index = 0;
	std::vector<MaterializedJpegDctBlock> blocks;
};

struct JpegDctCropBox {
	uint32_t x      = 0;
	uint32_t y      = 0;
	uint32_t width  = 0;
	uint32_t height = 0;
};

struct JpegDctImageCropRequest {
	uint32_t       global_image_index = 0;
	JpegDctCropBox source_crop {};
};

struct JpegDctDeviceBatchOptions {
	JpegDctDeviceLayout layout               = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	size_t              cache_capacity_bytes = 0;
};

struct JpegDctDeviceImageLayout {
	uint32_t global_image_index = 0;
	uint64_t block_offset       = 0;
	uint32_t block_count        = 0;
};

struct JpegDctDeviceBlockMetadata {
	uint32_t request_index      = 0;
	uint32_t global_image_index = 0;
	uint32_t semantic_slot_id   = 0;
	uint32_t block_x            = 0;
	uint32_t block_y            = 0;
};

struct JpegDctDeviceRowgroupMetadata {
	uint32_t shard_id       = 0;
	uint32_t rowgroup_index = 0;
};

struct JpegDctDeviceCacheStats {
	size_t capacity_bytes     = 0;
	size_t resident_bytes     = 0;
	size_t resident_rowgroups = 0;
	size_t hits               = 0;
	size_t misses             = 0;
	size_t inserts            = 0;
	size_t evictions          = 0;
};

class JpegDctDeviceBatch {
public:
	struct Impl;

	JpegDctDeviceBatch() noexcept;
	explicit JpegDctDeviceBatch(std::unique_ptr<Impl> impl) noexcept;
	~JpegDctDeviceBatch();

	JpegDctDeviceBatch(const JpegDctDeviceBatch&)            = delete;
	JpegDctDeviceBatch& operator=(const JpegDctDeviceBatch&) = delete;
	JpegDctDeviceBatch(JpegDctDeviceBatch&&) noexcept;
	JpegDctDeviceBatch& operator=(JpegDctDeviceBatch&&) noexcept;

	[[nodiscard]] const int16_t*                                    device_coefficients() const noexcept;
	[[nodiscard]] size_t                                            coefficient_count() const noexcept;
	[[nodiscard]] size_t                                            coefficient_bytes() const noexcept;
	[[nodiscard]] size_t                                            block_count() const noexcept;
	[[nodiscard]] size_t                                            image_count() const noexcept;
	[[nodiscard]] size_t                                            rowgroup_count() const noexcept;
	[[nodiscard]] JpegDctDeviceCacheStats                           cache_stats() const noexcept;
	[[nodiscard]] JpegDctDeviceLayout                               layout() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceImageLayout>&      image_layouts() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceBlockMetadata>&    block_metadata() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceRowgroupMetadata>& rowgroups() const noexcept;

private:
	std::unique_ptr<Impl> impl_;
};

class JpegDctShardDatasetReader {
public:
	explicit JpegDctShardDatasetReader(const std::filesystem::path& manifest_path);
	~JpegDctShardDatasetReader();

	JpegDctShardDatasetReader(const JpegDctShardDatasetReader&)            = delete;
	JpegDctShardDatasetReader& operator=(const JpegDctShardDatasetReader&) = delete;
	JpegDctShardDatasetReader(JpegDctShardDatasetReader&&) noexcept;
	JpegDctShardDatasetReader& operator=(JpegDctShardDatasetReader&&) noexcept;

	[[nodiscard]] uint64_t image_count() const noexcept;

	MaterializedJpegDctImage MaterializeImageDct(uint32_t global_image_index);

	JpegDctDeviceBatch ReadDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                      const JpegDctDeviceBatchOptions&            options = {});

	JpegDctBlockGroup ReadBlockGroup(uint32_t shard_id, uint32_t semantic_slot_id, uint32_t block_x, uint32_t block_y);

	JpegDctRowRef LocateRow(uint32_t global_image_index, uint32_t semantic_slot_id, uint32_t block_x, uint32_t block_y);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

JpegDctTable read_jpeg_dct_file(const std::filesystem::path& path, const JpegDctReaderOptions& options = {});

JpegDctTable read_jpeg_dct_dataset(const std::vector<std::filesystem::path>& paths,
                                   const JpegDctReaderOptions&               options = {});

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata& metadata, const std::filesystem::path& output_path);

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata&       metadata,
                             const std::filesystem::path&        output_path,
                             const JpegDctMetadataWriterOptions& options);

void compress_jpeg_dct_to_fls(const JpegDctTable&          table,
                              const std::filesystem::path& fls_output_path,
                              const std::filesystem::path& metadata_output_path);

void compress_jpeg_dct_to_fls(const JpegDctTable&                 table,
                              const std::filesystem::path&        fls_output_path,
                              const std::filesystem::path&        metadata_output_path,
                              const JpegDctMetadataWriterOptions& metadata_options);

void compress_jpeg_dct_file_to_fls(const std::filesystem::path& jpeg_path,
                                   const std::filesystem::path& fls_output_path,
                                   const std::filesystem::path& metadata_output_path,
                                   const JpegDctReaderOptions&  options = {});

void compress_jpeg_dct_file_to_fls(const std::filesystem::path&        jpeg_path,
                                   const std::filesystem::path&        fls_output_path,
                                   const std::filesystem::path&        metadata_output_path,
                                   const JpegDctReaderOptions&         options,
                                   const JpegDctMetadataWriterOptions& metadata_options);

void compress_jpeg_dct_dataset_to_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                      const std::filesystem::path&              fls_output_path,
                                      const std::filesystem::path&              metadata_output_path,
                                      const JpegDctReaderOptions&               options = {});

void compress_jpeg_dct_dataset_to_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                      const std::filesystem::path&              fls_output_path,
                                      const std::filesystem::path&              metadata_output_path,
                                      const JpegDctReaderOptions&               options,
                                      const JpegDctMetadataWriterOptions&       metadata_options);

void write_jpeg_dct_shard_manifest(const JpegDctShardManifest& manifest, const std::filesystem::path& output_path);

JpegDctShardManifest
compress_jpeg_dct_dataset_to_sharded_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                         const std::filesystem::path&              output_dir,
                                         const JpegDctReaderOptions&               options          = {},
                                         const JpegDctShardOptions&                shard_options    = {},
                                         const JpegDctMetadataWriterOptions&       metadata_options = {});

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_HPP
