#ifndef GALP_JPEG_DCT_STORAGE_HPP
#define GALP_JPEG_DCT_STORAGE_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct_device.hpp"
#include "galp/jpeg_dct_format.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace galp::jpeg {

enum class JpegDctShardPreset {
	kCropLatency,
	kBalanced,
	kThroughput,
	kRandomAccess,
};

struct JpegDctShardOptions {
	size_t                shard_images                  = 8192;
	uint32_t              rowgroup_vectors              = 128;
	uint32_t              rowgroups_per_shard           = 256;
	size_t                threads                       = 1;
	size_t                shard_workers                 = 1;
	JpegDctShardPreset    preset                        = JpegDctShardPreset::kBalanced;
	bool                  shard_images_specified        = false;
	bool                  rowgroup_vectors_specified    = false;
	bool                  rowgroups_per_shard_specified = false;
	JpegDctPhysicalLayout physical_layout               = JpegDctPhysicalLayout::kSpatialMajorImageMinor;
	bool                  physical_layout_specified     = false;

	// Independent pipeline controls. `threads` remains the legacy layout/decode
	// control and is mapped to both values when explicitly selected.
	size_t layout_threads                     = 1;
	size_t shard_decode_threads               = 1;
	size_t encoding_workers_per_shard         = 1;
	bool   threads_specified                  = false;
	bool   layout_threads_specified           = false;
	bool   shard_decode_threads_specified     = false;
	bool   encoding_workers_per_shard_specified = false;
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
	uint64_t    payload_size             = 0;
	uint64_t    payload_crc64            = 0;
	uint64_t    compact_descriptor_size  = 0;
	uint64_t    source_descriptor_size   = 0;
	std::string fls_file_name;
	std::string metadata_file_name;
};

struct JpegDctShardManifest {
	uint32_t                               version             = 1;
	uint32_t                               rowgroup_vectors    = 128;
	uint32_t                               rowgroups_per_shard = 256;
	uint64_t                               image_count         = 0;
	std::string                            physical_layout;
	std::string                            descriptor_kind;
	uint32_t                               vector_size = 1024U;
	std::string                            spatial_order_name;
	JpegDctSpatialOrder                    spatial_order = JpegDctSpatialOrder::kRaster;
	std::vector<JpegDctShardManifestEntry> shards;

	[[nodiscard]] bool uses_independent_vector_rowgroups() const noexcept {
		return version == 3U;
	}

	[[nodiscard]] bool uses_compact_descriptor() const noexcept {
		return version == 3U && descriptor_kind == "galp-compact-v1";
	}
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

struct JpegDctReaderInitializationStats {
	double manifest_load_ms                    = 0.0;
	double shard_path_validation_ms            = 0.0;
	double shard_metadata_load_ms               = 0.0;
	double shard_metadata_index_ms              = 0.0;
	double transform_profile_construction_ms    = 0.0;
	double block_major_companion_index_load_ms  = 0.0;
	double total_ms                             = 0.0;
	size_t manifest_shard_count                 = 0U;
	size_t eagerly_loaded_shard_metadata_count  = 0U;
	size_t loaded_shard_metadata_count          = 0U;
	bool   block_major_metadata_lazy            = false;
	size_t block_major_loaded_descriptor_count  = 0U;
	uint64_t block_major_loaded_descriptor_bytes = 0U;
	uint64_t block_major_descriptor_cache_byte_bound = 0U;
	double block_major_descriptor_open_ms        = 0.0;
	double block_major_descriptor_validation_ms  = 0.0;
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

// Thread-safety contract:
// - const metadata, lookup, estimate, and CPU-only planning operations may run
//   concurrently on one reader;
// - PrepareDeviceDctBatch and StagePreparedDeviceDctBatchIo may run
//   concurrently; their plan/profile and host-reader caches are internally
//   synchronized;
// - device resource binding and host submission are mutex-serialized per
//   reader. Before shared scratch/cache state is rebound, an internal CUDA
//   completion fence joins the previous asynchronous device tail. The CPU
//   planning portion of ReadDeviceDctBatch remains outside that boundary;
// - returned batches remain asynchronous and do not have to be synchronized or
//   destroyed before another call on the same reader. A different reader owns
//   independent execution state.
// Destruction or move-assignment requires exclusive ownership.
class JpegDctShardDatasetReader {
public:
	explicit JpegDctShardDatasetReader(const std::filesystem::path& manifest_path);
	~JpegDctShardDatasetReader();

	JpegDctShardDatasetReader(const JpegDctShardDatasetReader&)            = delete;
	JpegDctShardDatasetReader& operator=(const JpegDctShardDatasetReader&) = delete;
	JpegDctShardDatasetReader(JpegDctShardDatasetReader&&) noexcept;
	JpegDctShardDatasetReader& operator=(JpegDctShardDatasetReader&&) noexcept;

	[[nodiscard]] uint64_t          image_count() const noexcept;
	[[nodiscard]] JpegDctReaderInitializationStats InitializationStats() const noexcept;
	[[nodiscard]] JpegImageMetadata ImageMetadata(uint32_t global_image_index) const;
	[[nodiscard]] uint64_t RowgroupStorageBytes(uint32_t shard_id, const std::vector<uint32_t>& rowgroup_indices) const;
	MaterializedJpegDctImage MaterializeImageDct(uint32_t global_image_index);

	JpegDctDeviceBatchPlanPreview  PlanDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                  const JpegDctDeviceBatchOptions&            options = {}) const;
	JpegDctDeviceBatchPlanEstimate EstimateDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                      const JpegDctDeviceBatchOptions& options = {}) const;
	JpegDctDeviceBatchPreparedPlan PrepareDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                     const JpegDctDeviceBatchOptions&            options = {});
	void StagePreparedDeviceDctBatchIo(JpegDctDeviceBatchPreparedPlan& plan);
	JpegDctDeviceBatch             ReadPreparedDeviceDctBatch(JpegDctDeviceBatchPreparedPlan plan);
	JpegDctDeviceBatch             ReadDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
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
// Public, read-only manifest entry point. Native physical schedulers need the
// shard boundaries without depending on src/jpeg private headers.
JpegDctShardManifest read_jpeg_dct_shard_manifest(const std::filesystem::path& path);
JpegDctShardManifest
compress_jpeg_dct_dataset_to_sharded_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                         const std::filesystem::path&              output_dir,
                                         const JpegDctReaderOptions&               options          = {},
                                         const JpegDctShardOptions&                shard_options    = {},
                                         const JpegDctMetadataWriterOptions&       metadata_options = {});

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_STORAGE_HPP
