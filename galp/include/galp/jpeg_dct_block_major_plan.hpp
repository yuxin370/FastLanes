#ifndef GALP_JPEG_DCT_BLOCK_MAJOR_PLAN_HPP
#define GALP_JPEG_DCT_BLOCK_MAJOR_PLAN_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct_block_major_access.hpp"
#include "galp/jpeg_dct_device.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace galp::jpeg {

struct JpegDctBlockMajorSupportRectangle {
	uint32_t semantic_slot_id = 0U;
	int32_t  x                = 0;
	int32_t  y                = 0;
	uint32_t width            = 0U;
	uint32_t height           = 0U;
	uint16_t source_width_in_blocks  = 0U;
	uint16_t source_height_in_blocks = 0U;
	uint16_t quant_table_index       = 0U;
	uint8_t  h_samp_factor           = 0U;
	uint8_t  v_samp_factor           = 0U;
	bool     present          = false;
};

struct JpegDctBlockMajorCompactRequest {
	uint32_t global_image_index = 0U;
	uint32_t shard_id           = 0U;
	uint32_t local_image_index  = 0U;
	uint32_t output_slot        = 0U;
	uint32_t unique_image_index = 0U;
	bool     horizontal_flip    = false;
	std::array<JpegDctBlockMajorSupportRectangle, 3> components {};
};

struct JpegDctBlockMajorUniqueImage {
	uint32_t shard_id          = 0U;
	uint32_t local_image_index = 0U;
	uint32_t first_fanout      = 0U;
	uint32_t fanout_count      = 0U;
};

struct JpegDctBlockMajorGroupBinding {
	uint32_t shard_id              = 0U;
	uint32_t semantic_slot_id      = 0U;
	uint32_t block_x               = 0U;
	uint32_t block_y               = 0U;
	uint32_t group_id              = 0U;
	uint32_t rank_cell_id          = 0U;
	uint32_t runtime_rank_cell_index = 0U;
	uint32_t fls_rowgroup_index    = 0U;
	uint32_t row_start_in_rowgroup = 0U;
	uint32_t first_rank_run        = 0U;
	uint32_t rank_run_count        = 0U;
};

struct JpegDctBlockMajorRankRun {
	uint32_t group_binding_index = 0U;
	uint32_t rank_begin          = 0U;
	uint32_t rank_count          = 0U;
};

struct JpegDctBlockMajorVectorRun {
	uint32_t shard_id       = 0U;
	uint32_t rowgroup_index = 0U;
	uint32_t first_vector   = 0U;
	uint32_t vector_count   = 0U;
};

struct JpegDctBlockMajorSelectedRowgroup {
	uint32_t shard_id          = 0U;
	uint32_t rowgroup_index    = 0U;
	uint32_t first_vector_run  = 0U;
	uint32_t vector_run_count  = 0U;
	uint32_t selected_vectors  = 0U;
};

struct JpegDctBlockMajorRuntimeRankCell {
	uint32_t                           shard_id                = 0U;
	uint32_t                           source_cell_id          = 0U;
	uint32_t                           image_count             = 0U;
	uint32_t                           payload_offset          = 0U;
	uint32_t                           payload_size            = 0U;
	uint16_t                           present_count           = 0U;
	uint16_t                           rank_checkpoint_images  = 0U;
	JpegDctBlockMajorPresenceEncoding encoding                = JpegDctBlockMajorPresenceEncoding::kEmpty;
};

struct JpegDctBlockMajorRuntimeQuantTable {
	uint64_t                 fingerprint = 0U;
	std::array<uint16_t, 64> values {};
};

struct JpegDctBlockMajorCompactPlanStats {
	uint64_t request_count                  = 0U;
	uint64_t request_sort_items             = 0U;
	uint64_t unique_image_count             = 0U;
	uint64_t duplicate_output_count         = 0U;
	uint64_t shard_local_request_runs       = 0U;
	uint64_t support_rectangle_count        = 0U;
	uint64_t touched_block_groups           = 0U;
	uint64_t group_rank_runs                = 0U;
	uint64_t selected_rowgroups             = 0U;
	uint64_t selected_vector_runs           = 0U;
	uint64_t selected_vectors               = 0U;
	uint64_t touched_rank_cells             = 0U;
	uint64_t rank_payload_bytes             = 0U;
	uint64_t touched_quant_tables           = 0U;
	uint64_t expanded_transform_items       = 0U;
	uint64_t global_transform_sort_items    = 0U;
	uint64_t compact_plan_bytes             = 0U;
	uint64_t compact_plan_peak_bytes        = 0U;
	bool     input_was_shard_local_monotonic = false;
};

struct JpegDctBlockMajorCompactPlan {
	std::vector<JpegDctBlockMajorCompactRequest> requests;
	std::vector<JpegDctBlockMajorUniqueImage>    unique_images;
	std::vector<uint32_t>                        duplicate_output_slots;
	std::vector<JpegDctBlockMajorGroupBinding>   group_bindings;
	std::vector<JpegDctBlockMajorRankRun>        rank_runs;
	std::vector<JpegDctBlockMajorRuntimeRankCell> rank_cells;
	std::vector<uint8_t>                          rank_payload;
	std::vector<JpegDctBlockMajorRuntimeQuantTable> quant_tables;
	std::vector<JpegDctBlockMajorSelectedRowgroup> rowgroups;
	std::vector<JpegDctBlockMajorVectorRun>        vector_runs;
	JpegDctBlockMajorCompactPlanStats stats;
};

class JpegDctBlockMajorCompactPlanner {
public:
	JpegDctBlockMajorCompactPlanner(const std::filesystem::path& manifest_path,
	                                const std::filesystem::path& descriptor_directory);
	~JpegDctBlockMajorCompactPlanner();

	JpegDctBlockMajorCompactPlanner(const JpegDctBlockMajorCompactPlanner&)            = delete;
	JpegDctBlockMajorCompactPlanner& operator=(const JpegDctBlockMajorCompactPlanner&) = delete;
	JpegDctBlockMajorCompactPlanner(JpegDctBlockMajorCompactPlanner&&) noexcept;
	JpegDctBlockMajorCompactPlanner& operator=(JpegDctBlockMajorCompactPlanner&&) noexcept;

	[[nodiscard]] uint64_t image_count() const noexcept;
	[[nodiscard]] size_t   loaded_descriptor_count() const noexcept;
	[[nodiscard]] uint64_t loaded_descriptor_bytes() const noexcept;
	[[nodiscard]] uint64_t descriptor_cache_byte_bound() const noexcept;
	[[nodiscard]] double   descriptor_open_ms() const noexcept;
	[[nodiscard]] double   descriptor_validation_ms() const noexcept;
	[[nodiscard]] JpegDctBlockMajorCompactPlan Plan(
	    const std::vector<JpegDctImageCropRequest>& requests,
	    const std::optional<JpegDctGridTransformSpec>& grid_transform = std::nullopt) const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_BLOCK_MAJOR_PLAN_HPP
