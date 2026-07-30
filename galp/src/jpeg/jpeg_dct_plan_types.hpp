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
	uint32_t                                                binding_base          = 0;
	uint32_t                                                row_start_in_rowgroup = 0;
	uint32_t                                                vector_remap_base     = std::numeric_limits<uint32_t>::max();
	uint8_t                                                 zigzag_columns        = 0;
	uint8_t                                                 spatial_order         = 0;
	uint8_t                                                 horizontal_flip       = 0;
	uint8_t                                                 reserved              = 0;
	std::array<JpegDctDevicePlanlessComponentDescriptor, 3> components {};
};

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
	size_t                                            selected_vector_count = 0;
	size_t                                            full_vector_count     = 0;
	bool                                              selected_chunks_fit   = true;
	bool                                              has_vector_plan       = false;
	bool                                              sparse_storage_read   = false;
	bool                                              automatic_sparse_storage_candidate = false;
	JpegDctRuntimePolicyResult                        runtime_policy {};
	std::shared_ptr<galp::format::FlsReader>          prepared_reader;
	std::shared_ptr<const galp::format::SparseVectorReadPlan> compiled_sparse_read_plan;
	std::shared_ptr<JpegDctStagedRowgroupRead>        staged_read;
};

struct JpegDctDeviceShardPlan {
	uint32_t                               shard_id              = 0;
	const std::filesystem::path*           fls_path              = nullptr;
	bool                                   mixed_physical_shards = false;
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
	JpegDctDeviceRowgroupPrefetchConfig        rowgroup_prefetch {};
	JpegDctSchedulingPolicy                    scheduling_policy           = JpegDctSchedulingPolicy::kFullyOverlapped;
	size_t                                     transform_blocks_per_launch = 0;
	size_t                                     transform_ctas_per_launch   = 0;
	bool                                       use_low_priority_streams    = false;
	size_t automatic_sparse_storage_candidate_rowgroup_count = 0U;
	size_t automatic_sparse_storage_selected_rowgroup_count  = 0U;
	size_t automatic_sparse_storage_rejected_rowgroup_count  = 0U;
	size_t automatic_sparse_storage_full_bytes               = 0U;
	size_t automatic_sparse_storage_candidate_bytes          = 0U;
	size_t automatic_sparse_storage_candidate_pread_count    = 0U;
	double automatic_sparse_storage_full_estimated_ns        = 0.0;
	double automatic_sparse_storage_candidate_estimated_ns   = 0.0;
	bool                                       host_io_staged             = false;
	double                                     host_io_staging_ms         = 0.0;
	size_t                                     host_io_staged_rowgroups   = 0U;
};

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_PLAN_TYPES_HPP
