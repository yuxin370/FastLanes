#ifndef GALP_JPEG_DCT_PLAN_TYPES_HPP
#define GALP_JPEG_DCT_PLAN_TYPES_HPP

#include "galp/jpeg_dct_device.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

namespace galp::format {
class FlsReader;
class SparseVectorReadPlan;
} // namespace galp::format

namespace galp::jpeg::detail {

struct JpegDctStagedRowgroupRead;

inline constexpr size_t kJpegDctCoefficientCount = 64;

enum class JpegDctRuntimePolicyDecision {
	kSelectedVectors,
	kFullRowgroup,
};

enum class JpegDctRuntimePolicyReason {
	kCropSavesEnoughVectors,
	kTailChunkWouldOverrun,
	kSelectedCoversMostVectors,
	kSavingsTooSmall,
	kForcedFullRowgroup,
	kForcedSelectedVectors,
};

enum class JpegDctReadStrategy {
	kRunIntervalExact,
	kBitmapExact,
	kFullRowgroup,
};

struct JpegDctRuntimePolicyResult {
	JpegDctRuntimePolicyDecision decision = JpegDctRuntimePolicyDecision::kSelectedVectors;
	JpegDctRuntimePolicyReason   reason   = JpegDctRuntimePolicyReason::kCropSavesEnoughVectors;
};

enum class JpegDctCoefficientSelectionKind {
	kAll,
	kPrefix,
	kList,
};

struct JpegDctCoefficientSelectionShape {
	JpegDctCoefficientSelectionKind kind  = JpegDctCoefficientSelectionKind::kAll;
	size_t                          count = kJpegDctCoefficientCount;

	[[nodiscard]] bool is_contiguous_prefix() const noexcept {
		return kind == JpegDctCoefficientSelectionKind::kAll || kind == JpegDctCoefficientSelectionKind::kPrefix;
	}
};

struct JpegDctDeviceGatherItem {
	uint32_t rowgroup_index     = 0;
	uint32_t row_in_rowgroup    = 0;
	uint64_t output_block_index = 0;
};

struct JpegDctDeviceProjectionItem {
	uint32_t rowgroup_index                 = 0;
	uint32_t row_in_rowgroup                = 0;
	uint64_t output_block_index             = 0;
	uint16_t selected_coefficient_slot      = 0;
	uint8_t  logical_coefficient_id         = 0;
	uint8_t  physical_coefficient_column_id = 0;
	uint8_t  output_coefficient_id          = 0;
	uint8_t  output_grid_tensor             = 0;
	float    weight                         = 1.0F;
};

struct JpegDctDeviceFixedTransformItem {
	uint32_t rowgroup_index        = 0;
	uint32_t row_in_rowgroup       = 0;
	uint64_t output_block_index    = 0;
	uint32_t image_index           = 0;
	uint16_t local_block_x         = 0;
	uint16_t local_block_y         = 0;
	uint16_t output_block_x        = 0;
	uint16_t output_block_y        = 0;
	uint8_t  component             = 0;
	bool     zigzag_columns        = false;
	bool     horizontal_flip       = false;
	uint16_t x_factor              = 2;
	uint16_t y_factor              = 2;
	uint8_t  x_subblock            = 0;
	uint8_t  y_subblock            = 0;
	bool     x_upsample            = false;
	bool     y_upsample            = false;
	uint16_t x_up_factor           = 1;
	uint16_t y_up_factor           = 1;
	uint16_t x_down_factor         = 1;
	uint16_t y_down_factor         = 1;
	uint32_t quant_table_index     = 0;
	uint32_t x_weight_matrix_index = 0;
	uint32_t y_weight_matrix_index = 0;
};

// Compact output-driven descriptors are shared values: the CPU planner fills
// them and CUDA execution uploads the exact same definition.
struct JpegDctDevicePlanlessComponentDescriptor {
	uint32_t semantic_slot_id     = 0U;
	uint32_t block_major_coordinate_lookup_index = std::numeric_limits<uint32_t>::max();
	uint32_t component_row_offset = 0;
	uint32_t width_in_blocks      = 0;
	uint32_t height_in_blocks     = 0;
	int32_t  crop_x               = 0;
	int32_t  crop_y               = 0;
	uint32_t crop_width           = 0;
	uint32_t crop_height          = 0;
	uint16_t x_up_factor          = 1;
	uint16_t y_up_factor          = 1;
	uint16_t x_down_factor        = 1;
	uint16_t y_down_factor        = 1;
	uint32_t x_phase_matrix_base  = std::numeric_limits<uint32_t>::max();
	uint32_t y_phase_matrix_base  = std::numeric_limits<uint32_t>::max();
	uint32_t quant_table_index    = 0;
	uint8_t  present              = 0;
};

struct JpegDctDevicePlanlessImageDescriptor {
	uint32_t                                                request_index         = 0;
	uint32_t                                                shard_id              = 0;
	uint32_t                                                local_image_index     = 0;
	uint32_t                                                binding_base          = 0;
	uint32_t                                                row_start_in_rowgroup = 0;
	uint32_t                                                vector_remap_base     = std::numeric_limits<uint32_t>::max();
	uint32_t                                                vector_binding_base   = std::numeric_limits<uint32_t>::max();
	uint32_t                                                vector_binding_count  = 0;
	uint8_t                                                 zigzag_columns        = 0;
	uint8_t                                                 spatial_order         = 0;
	uint8_t                                                 horizontal_flip       = 0;
	uint8_t                                                 reserved              = 0;
	std::array<JpegDctDevicePlanlessComponentDescriptor, 3> components {};
};

// Manifest-v3 stores every image-local vector in a distinct physical
// rowgroup.  The retained batch plan names those physical sources, while the
// execution workset resolves them to its transient coefficient-binding table.
// Keeping this relation at vector granularity avoids materializing any source
// block or source-to-output transform item on the host.
struct JpegDctDevicePlanlessVectorSource {
	uint32_t shard_id       = std::numeric_limits<uint32_t>::max();
	uint32_t rowgroup_index = std::numeric_limits<uint32_t>::max();
};

struct JpegDctDeviceImageMajorPlanlessPlan {
	std::vector<JpegDctDevicePlanlessImageDescriptor> images;
	std::vector<JpegDctDevicePlanlessVectorSource>    vector_sources;
};

struct JpegDctDeviceBlockMajorGroupBinding {
	uint32_t shard_id              = 0U;
	uint32_t semantic_slot_id      = 0U;
	uint32_t block_x               = 0U;
	uint32_t block_y               = 0U;
	uint32_t rowgroup_index        = 0U;
	uint32_t row_start_in_rowgroup = 0U;
	uint32_t rank_cell_index       = 0U;
	uint32_t coefficient_binding_base = std::numeric_limits<uint32_t>::max();
	uint32_t vector_remap_base        = std::numeric_limits<uint32_t>::max();
};

struct JpegDctDeviceBlockMajorRankCell {
	uint32_t image_count             = 0U;
	uint32_t payload_offset          = 0U;
	uint32_t payload_size            = 0U;
	uint16_t present_count           = 0U;
	uint16_t rank_checkpoint_images  = 0U;
	uint8_t  encoding                = 0U;
};

inline constexpr uint32_t kInvalidJpegDctBlockMajorGroupIndex = std::numeric_limits<uint32_t>::max();

// One plan-lifetime, directly addressed coordinate plane. Coordinates stay
// absolute in component block space; origin trims leading crop holes while
// the shared index vector keeps image descriptors compact.
struct JpegDctDeviceBlockMajorCoordinateGroupLookup {
	uint32_t shard_id              = 0U;
	uint32_t semantic_slot_id      = 0U;
	uint32_t origin_x              = 0U;
	uint32_t origin_y              = 0U;
	uint32_t width                 = 0U;
	uint32_t height                = 0U;
	uint32_t stride                = 0U;
	uint64_t group_index_base      = 0U;
	uint32_t populated_group_count = 0U;
};

struct JpegDctDeviceBlockMajorPlanlessPlan {
	std::vector<JpegDctDevicePlanlessImageDescriptor>    images;
	std::vector<JpegDctDeviceBlockMajorGroupBinding>    groups;
	std::vector<JpegDctDeviceBlockMajorCoordinateGroupLookup> coordinate_group_lookups;
	std::vector<uint32_t>                                coordinate_group_indices;
	std::vector<JpegDctDeviceBlockMajorRankCell>        rank_cells;
	std::vector<uint8_t>                                rank_payload;
};

// One rowgroup belongs to exactly one bounded execution workset.  These
// bindings are transient execution-planning input; they are not source-block
// transform items and never expand coefficient contributions.
struct JpegDctDeviceBlockMajorRowgroupWorkset {
	uint32_t shard_id       = 0U;
	uint32_t rowgroup_index = 0U;
	uint32_t workset_index  = 0U;
};

// Output ownership is stored workset-major.  Offsets has workset_count + 1
// entries and indexes active_output_blocks.  Each active entry is one logical
// output block that must run in that workset; source contributions remain
// implicit in the compact group/rank descriptors and are resolved by CUDA.
struct JpegDctDeviceBlockMajorActiveOutputSchedule {
	std::vector<uint64_t> offsets;
	std::vector<uint32_t> active_output_blocks;
	uint64_t              logical_output_block_count = 0U;
	uint64_t              source_contribution_count  = 0U;
	uint64_t              source_contribution_visit_count = 0U;
	uint64_t              output_workset_ownership_count = 0U;
	uint64_t              temporary_bytes_peak       = 0U;
	double                group_workset_build_ms      = 0.0;
	double                active_output_count_ms      = 0.0;
	double                active_output_prefix_ms     = 0.0;
	double                active_output_fill_ms       = 0.0;
	double                total_build_ms              = 0.0;
};

// Build and validate the plan-lifetime coordinate-to-group index after the
// final group order is frozen. Existing lookup storage is replaced.
void build_block_major_coordinate_group_lookup(JpegDctDeviceBlockMajorPlanlessPlan& plan);

[[nodiscard]] uint32_t find_block_major_coordinate_group_lookup(
    const JpegDctDeviceBlockMajorPlanlessPlan& plan,
    uint32_t                                   shard_id,
    uint32_t                                   semantic_slot_id);

[[nodiscard]] JpegDctDeviceBlockMajorActiveOutputSchedule build_block_major_active_output_schedule(
    const JpegDctDeviceBlockMajorPlanlessPlan&                 plan,
    const std::vector<JpegDctDeviceBlockMajorRowgroupWorkset>& rowgroup_worksets,
    const JpegDctGridTransformSpec&                            transform);

struct JpegDctDeviceRowgroupPlan {
	uint32_t                                          rowgroup_index  = 0;
	uint32_t                                          source_shard_id = std::numeric_limits<uint32_t>::max();
	const std::filesystem::path*                      source_fls_path = nullptr;
	std::vector<JpegDctDeviceGatherItem>              items;
	std::vector<JpegDctDeviceProjectionItem>          projection_items;
	std::vector<JpegDctDeviceFixedTransformItem>      fixed_transform_items;
	std::vector<uint32_t>                             selected_vectors;
	std::vector<JpegDctDeviceGatherItem>              selected_gather_items;
	std::vector<JpegDctDeviceProjectionItem>          selected_projection_items;
	std::vector<JpegDctDeviceFixedTransformItem>      selected_fixed_transform_items;
	std::vector<JpegDctDevicePlanlessImageDescriptor> planless_images;
	bool                                              image_major_planless = false;
	std::shared_ptr<const JpegDctDeviceBlockMajorPlanlessPlan> block_major_planless;
	size_t                                            selected_vector_count = 0;
	size_t                                            full_vector_count     = 0;
	bool                                              selected_chunks_fit   = true;
	bool                                              has_vector_plan       = false;
	bool                                              sparse_storage_read   = false;
	bool                                              automatic_sparse_storage_candidate = false;
	JpegDctRuntimePolicyResult                        runtime_policy {};
	JpegDctReadStrategy                               read_strategy = JpegDctReadStrategy::kFullRowgroup;
	std::shared_ptr<galp::format::FlsReader>          prepared_reader;
	std::shared_ptr<const galp::format::SparseVectorReadPlan> compiled_sparse_read_plan;
	std::shared_ptr<JpegDctStagedRowgroupRead>        staged_read;
	size_t                                            estimated_workset_resident_bytes = 0U;
};

struct JpegDctDeviceShardPlan {
	uint32_t                               shard_id              = 0;
	const std::filesystem::path*           fls_path              = nullptr;
	bool                                   mixed_physical_shards = false;
	std::shared_ptr<const JpegDctDeviceImageMajorPlanlessPlan> image_major_planless_owner;
	std::vector<JpegDctDeviceRowgroupPlan> rowgroups;
};

struct JpegDctDeviceRowgroupPrefetchConfig {
	bool   enabled            = true;
	size_t depth              = kDefaultJpegDctDeviceRowgroupPrefetchDepth;
	size_t workers            = kDefaultJpegDctDeviceRowgroupPrefetchWorkers;
	size_t min_decode_batches = kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches;
};

struct JpegDctDeviceScratch;
struct JpegDctDeviceDecodedRowgroupCache;

struct JpegDctDeviceBatchPlan {
	JpegDctDeviceLayout                                  layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::shared_ptr<std::vector<JpegDctDeviceShardPlan>> shards =
	    std::make_shared<std::vector<JpegDctDeviceShardPlan>>();
	std::shared_ptr<std::vector<uint32_t>>  fixed_transform_item_order    = std::make_shared<std::vector<uint32_t>>();
	std::shared_ptr<std::vector<uint32_t>>  fixed_transform_group_offsets = std::make_shared<std::vector<uint32_t>>();
	std::vector<JpegDctDeviceImageLayout>   image_layouts;
	std::vector<JpegDctDeviceBlockMetadata> block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	JpegDctDeviceDecodedRowgroupCache*         cache                                  = nullptr;
	bool                                       cache_enabled                          = false;
	bool                                       compact_v3_storage                     = false;
	JpegDctDeviceScratch*                      scratch                                = nullptr;
	bool                                       unify_rowgroups_across_shards          = false;
	size_t                                     planned_selected_vector_count          = 0;
	size_t                                     estimated_selected_vector_count        = 0;
	size_t                                     full_vector_count                      = 0;
	size_t                                     planned_saved_vector_count             = 0;
	size_t                                     estimated_saved_vector_count           = 0;
	size_t                                     fixed_transform_component_count        = 0;
	size_t                                     fixed_transform_source_block_count     = 0;
	size_t                                     fixed_transform_output_block_count     = 0;
	bool                                       uses_planless_fixed_transform          = false;
	size_t                                     host_expanded_transform_items_created  = 0;
	size_t                                     host_output_block_source_lists_created = 0;
	size_t                                     host_global_transform_sort_items       = 0;
	size_t                                     planless_axis_program_count            = 0;
	size_t                                     planless_axis_phase_matrix_count       = 0;
	size_t                                     compact_plan_bytes                     = 0;
	size_t                                     compact_plan_peak_bytes                = 0;
	size_t                                     coordinate_group_lookup_count          = 0;
	size_t                                     coordinate_group_index_entries         = 0;
	size_t                                     coordinate_group_index_populated       = 0;
	size_t                                     coordinate_group_index_holes           = 0;
	size_t                                     coordinate_group_index_bytes           = 0;
	double                                     coordinate_group_index_density         = 0.0;
	size_t                                     compiled_access_profile_hits           = 0;
	size_t                                     compiled_access_profile_misses         = 0;
	std::vector<uint8_t>                       selected_coefficients;
	std::vector<uint16_t>                      fixed_quant_tables;
	std::vector<float>                         fixed_resize_weight_matrices;
	JpegDctGridTransformSpec                   grid_transform;
	JpegDctCoefficientSelectionShape           coefficient_selection_shape {};
	size_t                                     coefficients_per_block = kJpegDctCoefficientCount;
	JpegDctYcbcrDctGridShape                   ycbcr_dct_grid_shape {};
	double                                     planned_selected_vector_ratio      = 0.0;
	double                                     estimated_selected_vector_ratio    = 0.0;
	double                                     planning_ms                        = 0.0;
	size_t                                     plan_cache_hits                    = 0;
	size_t                                     plan_cache_misses                  = 0;
	size_t                                     plan_cache_evictions               = 0;
	bool                                       exact_batch_plan_cache_enabled     = false;
	double                                     resize_weight_build_ms             = 0.0;
	size_t                                     dct_resize_weight_cache_hits       = 0;
	size_t                                     dct_resize_weight_cache_misses     = 0;
	size_t                                     dct_conversion_matrix_cache_hits   = 0;
	size_t                                     dct_conversion_matrix_cache_misses = 0;
	size_t                                     decode_batch_rowgroups             = kDefaultJpegDctDecodeBatchRowgroups;
	size_t                                     decode_workset_capacity_bytes      =
	    kDefaultJpegDctDeviceDecodeWorksetCapacityBytes;
	size_t                                     estimated_max_decode_workset_bytes   = 0U;
	size_t                                     estimated_oversized_decode_rowgroups = 0U;
	JpegDctDeviceRowgroupPrefetchConfig        rowgroup_prefetch {};
	JpegDctSchedulingPolicy                    scheduling_policy           = JpegDctSchedulingPolicy::kFullyOverlapped;
	size_t                                     transform_blocks_per_launch = 0;
	size_t                                     transform_ctas_per_launch   = 0;
	bool                                       use_low_priority_streams    = false;
	JpegDctBlockMajorDoubleBufferPolicy        block_major_double_buffer_policy =
	    JpegDctBlockMajorDoubleBufferPolicy::kAutomatic;
	size_t automatic_sparse_storage_candidate_rowgroup_count = 0U;
	size_t automatic_sparse_storage_selected_rowgroup_count  = 0U;
	size_t automatic_sparse_storage_rejected_rowgroup_count  = 0U;
	size_t automatic_sparse_storage_early_rejected_rowgroup_count = 0U;
	size_t automatic_sparse_storage_full_bytes               = 0U;
	size_t automatic_sparse_storage_candidate_bytes          = 0U;
	size_t automatic_sparse_storage_candidate_pread_count    = 0U;
	size_t automatic_sparse_storage_optimistic_bytes         = 0U;
	size_t automatic_sparse_storage_optimistic_pread_count   = 0U;
	double automatic_sparse_storage_full_estimated_ns        = 0.0;
	double automatic_sparse_storage_candidate_estimated_ns   = 0.0;
	double adaptive_run_interval_estimated_ns                 = 0.0;
	double adaptive_bitmap_estimated_ns                       = 0.0;
	double adaptive_full_rowgroup_estimated_ns                = 0.0;
	size_t adaptive_selected_memory_fit_rowgroup_count        = 0U;
	size_t adaptive_full_memory_fit_rowgroup_count            = 0U;
	size_t run_interval_exact_rowgroup_count                  = 0U;
	size_t bitmap_exact_rowgroup_count                        = 0U;
	size_t full_rowgroup_strategy_count                       = 0U;
	bool                                       host_io_staged             = false;
	double                                     host_io_staging_ms         = 0.0;
	size_t                                     host_io_staged_rowgroups   = 0U;
	// Compact-v3 host staging runs before ordered CUDA submission. Preserve
	// its batch-read and pinned-arena evidence on the prepared plan so Execute
	// can report the same counters without repeating the physical read.
	size_t compact_batch_buffer_acquire_count           = 0U;
	size_t compact_batch_buffer_growth_count            = 0U;
	size_t compact_batch_buffer_reuse_count             = 0U;
	size_t compact_batch_buffer_requested_bytes         = 0U;
	size_t compact_batch_buffer_capacity_bytes          = 0U;
	size_t compact_batch_buffer_high_water_bytes        = 0U;
	size_t compact_batch_buffer_pageable_fallback_count = 0U;
	size_t compact_batch_read_group_count               = 0U;
	size_t compact_batch_read_worker_count              = 0U;
};

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_PLAN_TYPES_HPP
