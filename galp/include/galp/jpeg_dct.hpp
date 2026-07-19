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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace galp::jpeg {

inline constexpr size_t kDefaultJpegDctDecodeBatchRowgroups                   = 64;
inline constexpr size_t kDefaultJpegDctDevicePlanCacheCapacity                = 128;
inline constexpr size_t kDefaultJpegDctDeviceRowgroupPrefetchDepth            = 4;
inline constexpr size_t kDefaultJpegDctDeviceRowgroupPrefetchWorkers          = 1;
inline constexpr size_t kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches = 2;

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

// Physical order of spatial DCT blocks inside each component of an image-major
// record. Tiles are measured in DCT blocks, not pixels. The image-major rowgroup
// remains the storage/decode atom for every mode.
enum class JpegDctSpatialOrder {
	kRaster,
	kTiledRaster32,
	kZOrder,
	kTiledZ32,
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
	// Applies only to the image-major physical layout. Kept at the end to
	// preserve legacy positional aggregate initialization.
	JpegDctSpatialOrder            image_major_spatial_order     = JpegDctSpatialOrder::kTiledZ32;
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
	// Random-access layout: every image is an independently addressable FLS
	// rowgroup and rows inside it are component-major/spatial-order-block-major.
	kDatasetImageMajorComponentBlockMajor,
};

enum class JpegDctPhysicalLayout {
	// Legacy layout optimized for the same spatial block across adjacent images.
	kSpatialMajorImageMinor,
	// Random-batch layout with exactly one independently decodable rowgroup per image.
	kImageMajor,
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
	JpegCompressionPartitionPolicy compression_partition_policy = JpegCompressionPartitionPolicy::kBySemanticSlot;
	JpegDctCoefficientEncoding     coefficient_encoding         = JpegDctCoefficientEncoding::kDense64FastLanes;
	std::vector<JpegDctImageGroupIndex>      image_group_index;
	// Explicit for newly written image-major v2 metadata. Old v2 metadata did
	// not carry this field and is decoded as the historical raster invariant.
	JpegDctSpatialOrder                    image_major_spatial_order = JpegDctSpatialOrder::kRaster;
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
	// Architecture preset for globally shuffled batches. It selects the
	// image-major physical format rather than tuning the legacy layout.
	kRandomAccess,
};

enum class JpegDctDeviceLayout {
	kImageMajorComponentBlockCoeff,
	kYcbcrDctGrid,
	kTransformedDctGrid,
};

struct JpegDctSamplingRatio {
	uint16_t horizontal_numerator   = 1;
	uint16_t horizontal_denominator = 1;
	uint16_t vertical_numerator     = 1;
	uint16_t vertical_denominator   = 1;
};

// Generic parameters for a fused JPEG-DCT grid transform. Application profiles
// provide concrete geometry, numeric policy, and accepted sampling ratios;
// the JPEG planner/executor only lowers this specification to the existing
// batched dequantize/clamp/rational-resize kernel.
struct JpegDctGridTransformSpec {
	uint32_t y_output_width_blocks     = 0;
	uint32_t y_output_height_blocks    = 0;
	uint32_t cbcr_output_width_blocks  = 0;
	uint32_t cbcr_output_height_blocks = 0;
	uint32_t crop_reference_width_blocks  = 0;
	uint32_t crop_reference_height_blocks = 0;
	uint32_t crop_origin_alignment_blocks = 1;
	uint32_t chroma_crop_scale_x           = 1;
	uint32_t chroma_crop_scale_y           = 1;
	int32_t  clamp_min                      = std::numeric_limits<int16_t>::min();
	int32_t  clamp_max                      = std::numeric_limits<int16_t>::max();
	bool     dequantize                     = true;
	bool     require_all_coefficients       = true;
	bool     allow_grayscale                = false;
	std::vector<uint32_t>             preferred_small_crop_width_blocks;
	std::vector<uint32_t>             preferred_small_crop_height_blocks;
	std::vector<JpegDctSamplingRatio> allowed_chroma_sampling_ratios;
};

inline constexpr uint8_t kJpegDctYcbcrDctGridTensorY    = 1;
inline constexpr uint8_t kJpegDctYcbcrDctGridTensorCbCr = 2;

struct JpegDctYcbcrDctGridShape {
	std::array<size_t, 6> y    = {0, 1, 0, 0, 8, 8};
	std::array<size_t, 6> cbcr = {0, 2, 0, 0, 8, 8};

	[[nodiscard]] size_t y_count() const noexcept {
		size_t count = 1;
		for (const auto dim : y) {
			count *= dim;
		}
		return count;
	}

	[[nodiscard]] size_t cbcr_count() const noexcept {
		size_t count = 1;
		for (const auto dim : cbcr) {
			count *= dim;
		}
		return count;
	}
};

struct JpegDctCoefficientSelection {
	std::vector<uint8_t> coefficients;

	[[nodiscard]] bool empty() const noexcept {
		return coefficients.empty();
	}
	[[nodiscard]] size_t size() const noexcept {
		return coefficients.empty() ? 64U : coefficients.size();
	}
};

[[nodiscard]] bool parse_jpeg_dct_coefficient_selection(std::string_view spec, JpegDctCoefficientSelection& selection);

struct JpegDctShardOptions {
	size_t             shard_images                  = 8192;
	uint32_t           rowgroup_vectors              = 128;
	uint32_t           rowgroups_per_shard           = 256;
	size_t             threads                       = 1;
	size_t             shard_workers                 = 1;
	JpegDctShardPreset preset                        = JpegDctShardPreset::kBalanced;
	bool               shard_images_specified        = false;
	bool               rowgroup_vectors_specified    = false;
	bool               rowgroups_per_shard_specified = false;
	JpegDctPhysicalLayout physical_layout            = JpegDctPhysicalLayout::kSpatialMajorImageMinor;
	bool                  physical_layout_specified  = false;
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

enum class JpegDctSchedulingPolicy {
	// Queue the complete transform grid and overlap next-batch Direct-DCT work
	// with the current model invocation.
	kFullyOverlapped,
	// Split the transform into bounded grids on a low-priority stream so a
	// high-priority model stream can claim most SMs between launches.
	kLimitedOverlap,
	// Use the same native event graph as fully-overlapped execution; callers
	// suppress next-batch prefetch until the current model invocation finishes.
	kSerial,
};

struct JpegDctDeviceBatchOptions {
	JpegDctDeviceLayout layout                 = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::optional<JpegDctGridTransformSpec> grid_transform;
	size_t              cache_capacity_bytes   = 0;
	size_t              decode_batch_rowgroups = kDefaultJpegDctDecodeBatchRowgroups;
	// Number of transformed batch plans retained by the reader. Zero disables this cache.
	size_t              plan_cache_capacity    = kDefaultJpegDctDevicePlanCacheCapacity;
	// Advanced rowgroup IO/materialization prefetch controls. Zero-valued sizes are normalized to defaults.
	bool   enable_rowgroup_prefetch             = true;
	size_t rowgroup_prefetch_depth              = kDefaultJpegDctDeviceRowgroupPrefetchDepth;
	size_t rowgroup_prefetch_workers            = kDefaultJpegDctDeviceRowgroupPrefetchWorkers;
	size_t rowgroup_prefetch_min_decode_batches = kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches;
	JpegDctCoefficientSelection coefficient_selection {};
	// Diagnostic A/B switch. Production transformed execution keeps this true;
	// false forces the legacy expanded graph so device tests and benchmark
	// reports can compare both algorithms under an otherwise identical contract.
	bool enable_planless_execution = true;
	JpegDctSchedulingPolicy scheduling_policy = JpegDctSchedulingPolicy::kFullyOverlapped;
	// Zero launches the complete planless transform grid once. A positive value
	// caps output work per launch. Limited-overlap launches at most 64 CTAs and
	// lets each CTA process multiple output blocks to bound active SMs without
	// paying one CUDA launch per 64 outputs.
	size_t transform_blocks_per_launch = 0;
	bool   use_low_priority_streams     = false;
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

struct JpegDctDeviceExecutionStats {
	size_t      planned_selected_vector_count                 = 0;
	size_t      selected_vector_count                         = 0;
	size_t      full_vector_count                             = 0;
	size_t      planned_saved_vector_count                    = 0;
	size_t      actual_saved_vector_count                     = 0;
	size_t      rowgroup_count                                = 0;
	size_t      workset_count                                 = 0;
	size_t      decode_kernel_launch_count                    = 0;
	size_t      gather_kernel_launch_count                    = 0;
	size_t      prefix_gather_kernel_launch_count             = 0;
	size_t      cached_gather_kernel_launch_count             = 0;
	size_t      materialize_kernel_launch_count               = 0;
	size_t      gather_item_count                             = 0;
	size_t      decoded_gather_item_count                     = 0;
	size_t      cached_gather_item_count                      = 0;
	size_t      workset_upload_count                          = 0;
	size_t      scratch_upload_count                          = 0;
	size_t      scratch_allocation_count                      = 0;
	size_t      internal_sync_count                           = 0;
	size_t      cached_gather_sync_count                      = 0;
	size_t      decoded_batch_sync_count                      = 0;
	size_t      cached_gather_event_handoff_count             = 0;
	size_t      sparse_vector_cache_hits                      = 0;
	size_t      sparse_vector_cache_misses                    = 0;
	size_t      plan_cache_hits                               = 0;
	size_t      plan_cache_misses                             = 0;
	size_t      plan_cache_evictions                          = 0;
	size_t      runtime_policy_selected_rowgroups             = 0;
	size_t      runtime_policy_full_rowgroups                 = 0;
	size_t      runtime_policy_tail_full_rowgroups            = 0;
	size_t      runtime_policy_ratio_full_rowgroups           = 0;
	size_t      runtime_policy_low_saving_full_rowgroups      = 0;
	size_t      prefetch_initial_cache_hit_rowgroup_count     = 0;
	size_t      prefetch_candidate_rowgroup_count             = 0;
	size_t      prefetch_active_shard_count                   = 0;
	size_t      prefetch_config_disabled_shard_count          = 0;
	size_t      prefetch_all_hit_shard_count                  = 0;
	size_t      prefetch_small_batch_disabled_shard_count     = 0;
	size_t      prefetch_selected_vector_disabled_shard_count = 0;
	size_t      prefetch_selected_vector_miss_rowgroup_count  = 0;
	size_t      prefetch_initial_hit_runtime_miss_count       = 0;
	size_t      prefetch_skipped_repeated_runtime_miss_count  = 0;
	size_t      prefetched_rowgroup_count                     = 0;
	size_t      prefetch_consumed_as_hit_count                = 0;
	size_t      prefetch_skipped_repeated_rowgroup_count      = 0;
	double      prefetch_consumed_as_hit_read_ms              = 0.0;
	double      prefetch_consumed_as_hit_wait_ms              = 0.0;
	double      planning_ms                                   = 0.0;
	double      workset_build_ms                              = 0.0;
	double      workset_upload_ms                             = 0.0;
	double      workset_upload_prep_ms                        = 0.0;
	double      workset_upload_arena_ms                       = 0.0;
	double      workset_upload_arena_pack_ms                  = 0.0;
	double      workset_upload_arena_layout_ms                = 0.0;
	double      workset_upload_arena_alloc_ms                 = 0.0;
	double      workset_upload_arena_resolve_ms               = 0.0;
	double      workset_upload_dma_issue_ms                   = 0.0;
	double      workset_upload_event_record_ms                = 0.0;
	size_t      workset_upload_dma_bytes                      = 0;
	size_t      workset_upload_dma_count                      = 0;
	double      decode_ms                                     = 0.0;
	double      gather_ms                                     = 0.0;
	double      decoded_gather_ms                             = 0.0;
	double      cached_gather_ms                              = 0.0;
	double      projection_ms                                 = 0.0;
	double      decoded_projection_ms                         = 0.0;
	double      projection_item_build_ms                      = 0.0;
	double      fixed_transform_ms                            = 0.0;
	double      fixed_grid_round_ms                           = 0.0;
	double      resize_weight_build_ms                        = 0.0;
	double      prefetch_wait_ms                              = 0.0;
	double      prefetch_depth_block_ms                       = 0.0;
	double      prefetch_queue_start_ms                       = 0.0;
	double      prefetch_rowgroup_read_ms                     = 0.0;
	double      prefetch_ready_ahead_ms                       = 0.0;
	double      sync_rowgroup_read_ms                         = 0.0;
	std::string runtime_policy_decision;
	std::string runtime_policy_reason;
	size_t      projection_item_count         = 0;
	size_t      decoded_projection_item_count = 0;
	size_t      fixed_transform_item_count    = 0;
	size_t      fixed_transform_image_count   = 0;
	size_t      fixed_transform_component_count    = 0;
	size_t      fixed_transform_source_block_count = 0;
	size_t      fixed_transform_output_block_count = 0;
	size_t      dct_resize_weight_cache_hits       = 0;
	size_t      dct_resize_weight_cache_misses     = 0;
	size_t      dct_conversion_matrix_cache_hits   = 0;
	size_t      dct_conversion_matrix_cache_misses = 0;
	size_t      project_decoded_ycbcr_grid_launch_count = 0;
	size_t      jpeg_dct_projection_items_materialized   = 0;
	size_t      fixed_grid_round_event_handoff_count     = 0;
	bool        cache_enabled                 = false;
	// Phase-2 architecture counters are appended to preserve the positional
	// initialization order of the legacy public aggregate.
	bool        exact_batch_plan_cache_enabled               = false;
	size_t      host_expanded_transform_items_created        = 0;
	size_t      host_output_block_source_lists_created       = 0;
	size_t      host_global_transform_sort_items              = 0;
	size_t      planless_image_descriptor_count              = 0;
	size_t      planless_transform_output_block_count        = 0;
	size_t      planless_axis_program_count                   = 0;
	size_t      planless_axis_phase_matrix_count              = 0;
	size_t      planless_axis_program_bytes                   = 0;
	size_t      rowgroup_storage_bytes_read                   = 0;
	size_t      galp_native_device_in_use_bytes               = 0;
	size_t      galp_native_device_peak_in_use_bytes          = 0;
	size_t      galp_native_device_cached_bytes               = 0;
	size_t      galp_native_device_allocation_requests        = 0;
	size_t      galp_native_device_cuda_allocation_count      = 0;
	size_t      galp_native_device_cuda_allocation_bytes      = 0;
	double      device_mapping_ms                             = 0.0;
	bool        device_mapping_fused                          = false;
	// Scheduling/stream diagnostics (appended for aggregate compatibility).
	size_t      planless_transform_kernel_launch_count        = 0;
	size_t      planless_transform_max_blocks_per_launch      = 0;
	size_t      planless_transform_max_output_blocks_per_launch = 0;
	size_t      decode_to_transform_event_handoff_count       = 0;
	size_t      copy_to_decode_event_handoff_count            = 0;
	int         direct_dct_stream_priority                     = 0;
	int         direct_dct_h2d_stream_priority                 = 0;
	int         direct_dct_decode_stream_priority              = 0;
	int         direct_dct_transform_stream_priority           = 0;
	int         direct_dct_round_stream_priority               = 0;
	int         cuda_least_stream_priority                     = 0;
	int         cuda_greatest_stream_priority                  = 0;
	bool        direct_dct_low_priority_streams                = false;
	std::string scheduling_policy;
};

struct JpegDctDeviceBatchPlanPreview {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	size_t                                     planned_selected_vector_count   = 0;
	size_t                                     estimated_selected_vector_count = 0;
	size_t                                     full_vector_count               = 0;
	size_t                                     planned_saved_vector_count      = 0;
	size_t                                     estimated_saved_vector_count    = 0;
	std::vector<uint8_t>                       selected_coefficients;
	size_t                                     coefficients_per_block          = 64;
	double                                     planned_selected_vector_ratio   = 0.0;
	double                                     estimated_selected_vector_ratio = 0.0;
	double                                     planning_ms                     = 0.0;
	double                                     resize_weight_build_ms          = 0.0;
	size_t                                     dct_resize_weight_cache_hits       = 0;
	size_t                                     dct_resize_weight_cache_misses     = 0;
	size_t                                     dct_conversion_matrix_cache_hits   = 0;
	size_t                                     dct_conversion_matrix_cache_misses = 0;
	JpegDctYcbcrDctGridShape                   ycbcr_dct_grid_shape;
	bool                                       uses_planless_fixed_transform = false;
	size_t                                     compact_image_descriptor_count = 0;
	size_t                                     fixed_transform_component_count = 0;
	size_t                                     fixed_transform_source_block_count = 0;
	size_t                                     fixed_transform_output_block_count = 0;
	size_t                                     host_expanded_transform_items_created = 0;
	size_t                                     host_output_block_source_lists_created = 0;
	size_t                                     host_global_transform_sort_items = 0;
	size_t                                     planless_axis_program_count = 0;
	size_t                                     planless_axis_phase_matrix_count = 0;
	size_t                                     planless_axis_program_bytes = 0;
	bool                                       exact_batch_plan_cache_enabled = false;
	size_t                                     compact_reader_image_locator_bytes = 0;
	size_t                                     compact_reader_shard_index_bytes = 0;
	size_t                                     compact_reader_layout_dictionary_bytes = 0;
	size_t                                     compact_reader_quant_table_dictionary_bytes = 0;
	size_t                                     compact_reader_total_bytes = 0;
	size_t                                     compact_reader_shard_descriptor_bytes = 0;
	bool                                       compact_reader_shard_index_derived = false;
};

struct JpegDctDeviceBatchPlanEstimate {
	JpegDctDeviceLayout                        layout      = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	size_t                                     block_count = 0;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	size_t                                     full_vector_count = 0;
	double                                     planning_ms       = 0.0;
};

class JpegDctDeviceBatchPreparedPlan {
public:
	struct Impl;

	JpegDctDeviceBatchPreparedPlan() noexcept;
	explicit JpegDctDeviceBatchPreparedPlan(std::unique_ptr<Impl> impl) noexcept;
	~JpegDctDeviceBatchPreparedPlan();

	JpegDctDeviceBatchPreparedPlan(const JpegDctDeviceBatchPreparedPlan&)            = delete;
	JpegDctDeviceBatchPreparedPlan& operator=(const JpegDctDeviceBatchPreparedPlan&) = delete;
	JpegDctDeviceBatchPreparedPlan(JpegDctDeviceBatchPreparedPlan&&) noexcept;
	JpegDctDeviceBatchPreparedPlan& operator=(JpegDctDeviceBatchPreparedPlan&&) noexcept;

	[[nodiscard]] bool                                              empty() const noexcept;
	[[nodiscard]] JpegDctDeviceLayout                               layout() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceImageLayout>&      image_layouts() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceBlockMetadata>&    block_metadata() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceRowgroupMetadata>& rowgroups() const noexcept;
	[[nodiscard]] size_t                                            planned_selected_vector_count() const noexcept;
	[[nodiscard]] size_t                                            estimated_selected_vector_count() const noexcept;
	[[nodiscard]] size_t                                            full_vector_count() const noexcept;
	[[nodiscard]] size_t                                            planned_saved_vector_count() const noexcept;
	[[nodiscard]] size_t                                            estimated_saved_vector_count() const noexcept;
	[[nodiscard]] const std::vector<uint8_t>&                       selected_coefficients() const noexcept;
	[[nodiscard]] size_t                                            coefficients_per_block() const noexcept;
	[[nodiscard]] double                                            planned_selected_vector_ratio() const noexcept;
	[[nodiscard]] double                                            estimated_selected_vector_ratio() const noexcept;
	[[nodiscard]] double                                            planning_ms() const noexcept;
	[[nodiscard]] bool                                              uses_planless_fixed_transform() const noexcept;
	[[nodiscard]] bool                                              exact_batch_plan_cache_enabled() const noexcept;
	[[nodiscard]] bool                                              decoded_rowgroup_cache_enabled() const noexcept;

private:
	friend class JpegDctShardDatasetReader;
	std::unique_ptr<Impl> impl_;
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
	[[nodiscard]] const int16_t*                                    y_coefficients() const noexcept;
	[[nodiscard]] const int16_t*                                    cbcr_coefficients() const noexcept;
	[[nodiscard]] const int16_t*                                    device_coefficients_async() const noexcept;
	[[nodiscard]] const int16_t*                                    y_coefficients_async() const noexcept;
	[[nodiscard]] const int16_t*                                    cbcr_coefficients_async() const noexcept;
	void                                                            synchronize() const;
	[[nodiscard]] size_t                                            coefficient_count() const noexcept;
	[[nodiscard]] size_t                                            coefficient_bytes() const noexcept;
	[[nodiscard]] size_t                                            y_coefficient_count() const noexcept;
	[[nodiscard]] size_t                                            cbcr_coefficient_count() const noexcept;
	[[nodiscard]] size_t                                            coefficients_per_block() const noexcept;
	[[nodiscard]] size_t                                            block_count() const noexcept;
	[[nodiscard]] size_t                                            image_count() const noexcept;
	[[nodiscard]] size_t                                            rowgroup_count() const noexcept;
	[[nodiscard]] int                                               cuda_device() const noexcept;
	[[nodiscard]] JpegDctDeviceCacheStats                           cache_stats() const noexcept;
	[[nodiscard]] JpegDctDeviceExecutionStats                       execution_stats() const noexcept;
	[[nodiscard]] const JpegDctDeviceCacheStats&                    cache_stats_ref() const noexcept;
	[[nodiscard]] const JpegDctDeviceExecutionStats&                execution_stats_ref() const noexcept;
	[[nodiscard]] JpegDctDeviceLayout                               layout() const noexcept;
	[[nodiscard]] void*                                             cuda_completion_event() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceImageLayout>&      image_layouts() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceBlockMetadata>&    block_metadata() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceRowgroupMetadata>& rowgroups() const noexcept;
	[[nodiscard]] const std::vector<uint8_t>&                       selected_coefficients() const noexcept;
	[[nodiscard]] JpegDctYcbcrDctGridShape                    ycbcr_dct_grid_shape() const noexcept;

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

	[[nodiscard]] uint64_t          image_count() const noexcept;
	[[nodiscard]] JpegImageMetadata ImageMetadata(uint32_t global_image_index) const;
	[[nodiscard]] uint64_t RowgroupStorageBytes(uint32_t                     shard_id,
	                                            const std::vector<uint32_t>& rowgroup_indices) const;

	MaterializedJpegDctImage MaterializeImageDct(uint32_t global_image_index);

	JpegDctDeviceBatchPlanPreview PlanDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                 const JpegDctDeviceBatchOptions&            options = {}) const;

	JpegDctDeviceBatchPlanEstimate EstimateDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                      const JpegDctDeviceBatchOptions& options = {}) const;

	JpegDctDeviceBatchPreparedPlan PrepareDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                     const JpegDctDeviceBatchOptions&            options = {});

	JpegDctDeviceBatch ReadPreparedDeviceDctBatch(JpegDctDeviceBatchPreparedPlan plan);

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
