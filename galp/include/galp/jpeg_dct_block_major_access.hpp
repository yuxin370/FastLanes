#ifndef GALP_JPEG_DCT_BLOCK_MAJOR_ACCESS_HPP
#define GALP_JPEG_DCT_BLOCK_MAJOR_ACCESS_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct_format.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace galp::jpeg {

inline constexpr uint16_t kJpegDctBlockMajorAccessVersion = 1U;

enum class JpegDctBlockMajorPresenceEncoding : uint8_t {
	kEmpty       = 0U,
	kAllPresent  = 1U,
	kSparseList  = 2U,
	kMissingList = 3U,
	kBitmapRank  = 4U,
};

struct JpegDctBlockMajorAccessBuildOptions {
	uint16_t rank_checkpoint_images           = 256U;
	uint16_t topology_checkpoint_coordinates  = 128U;
	bool     validate_after_write              = true;
	bool     exhaustive_rank_validation        = false;
};

struct JpegDctBlockMajorAccessImageRecord {
	uint32_t image_width          = 0U;
	uint32_t image_height         = 0U;
	uint32_t first_component      = 0U;
	uint16_t component_count      = 0U;
	uint16_t present_slot_mask    = 0U;
	uint8_t  data_precision       = 0U;
	int16_t  jpeg_color_space     = 0;
};

struct JpegDctBlockMajorAccessComponentRecord {
	uint16_t semantic_slot_id        = 0U;
	uint16_t local_component_index   = 0U;
	uint16_t width_in_blocks         = 0U;
	uint16_t height_in_blocks        = 0U;
	uint16_t padded_width_in_blocks  = 0U;
	uint16_t padded_height_in_blocks = 0U;
	uint8_t  h_samp_factor           = 0U;
	uint8_t  v_samp_factor           = 0U;
	uint16_t quant_dictionary_id     = 0U;
	uint32_t encoding_profile_id     = 0U;
	int16_t  component_id            = 0;
	int16_t  quant_table_number      = -1;
};

struct JpegDctBlockMajorAccessRank {
	uint32_t cell_id = 0U;
	uint32_t rank    = 0U;
	bool     present = false;
};

struct JpegDctBlockMajorAccessRankInterval {
	uint32_t rank_begin = 0U;
	uint32_t rank_end   = 0U;
};

// A self-contained compressed cell view suitable for copying only the cells
// touched by one runtime batch. The payload keeps the on-disk encoding; it is
// not an expanded bitmap or a coefficient-data copy.
struct JpegDctBlockMajorAccessRankCellRecord {
	uint32_t                            cell_id                = 0U;
	uint32_t                            image_count            = 0U;
	uint16_t                            present_count          = 0U;
	uint16_t                            rank_checkpoint_images = 0U;
	JpegDctBlockMajorPresenceEncoding  encoding               = JpegDctBlockMajorPresenceEncoding::kEmpty;
	std::vector<uint8_t>               payload;
};

struct JpegDctBlockMajorAccessQuantTableRecord {
	uint16_t                 dictionary_id = 0U;
	uint64_t                 fingerprint   = 0U;
	std::array<uint16_t, 64> values {};
};

struct JpegDctBlockMajorAccessGroup {
	uint32_t semantic_slot_id      = 0U;
	uint32_t block_x               = 0U;
	uint32_t block_y               = 0U;
	uint32_t group_id              = 0U;
	uint32_t rank_cell_id          = 0U;
	uint64_t row_start             = 0U;
	uint32_t row_count             = 0U;
	uint32_t fls_rowgroup_index    = 0U;
	uint32_t row_start_in_rowgroup = 0U;
};

struct JpegDctBlockMajorAccessValidationReport {
	uint64_t images_checked        = 0U;
	uint64_t components_checked    = 0U;
	uint64_t groups_checked        = 0U;
	uint64_t rank_cells_checked    = 0U;
	uint64_t rank_queries_checked  = 0U;
	uint64_t select_queries_checked = 0U;
};

struct JpegDctBlockMajorAccessShardReport {
	uint32_t shard_id                 = 0U;
	uint32_t image_count              = 0U;
	uint32_t group_count              = 0U;
	uint32_t rank_cell_count          = 0U;
	uint64_t descriptor_bytes         = 0U;
	uint64_t all_present_cells        = 0U;
	uint64_t sparse_cells             = 0U;
	uint64_t missing_cells            = 0U;
	uint64_t bitmap_cells             = 0U;
	uint64_t empty_cells              = 0U;
	uint64_t presence_payload_bytes   = 0U;
	JpegDctBlockMajorAccessValidationReport validation;
	std::filesystem::path descriptor_path;
};

struct JpegDctBlockMajorAccessDatasetReport {
	uint64_t source_dataset_bytes = 0U;
	uint64_t descriptor_bytes     = 0U;
	uint64_t index_bytes          = 0U;
	double   storage_growth_ratio = 0.0;
	bool     passes_one_percent   = false;
	bool     passes_half_percent  = false;
	std::filesystem::path index_path;
	std::vector<JpegDctBlockMajorAccessShardReport> shards;
};

// The dataset-level companion is deliberately small (one fixed-size record per
// shard).  Readers validate and retain this index at construction time while
// leaving the much larger per-shard descriptors unmapped until first touch.
struct JpegDctBlockMajorAccessIndexRecord {
	uint32_t shard_id                 = 0U;
	uint32_t image_count              = 0U;
	uint64_t first_global_image_index = 0U;
	uint64_t descriptor_bytes         = 0U;
	uint64_t descriptor_crc64         = 0U;
};

struct JpegDctBlockMajorAccessIndex {
	uint64_t manifest_crc64 = 0U;
	uint64_t manifest_bytes = 0U;
	uint64_t index_bytes    = 0U;
	std::vector<JpegDctBlockMajorAccessIndexRecord> shards;
};

[[nodiscard]] JpegDctBlockMajorAccessIndex read_jpeg_dct_block_major_access_index(
    const std::filesystem::path& index_path,
    const std::filesystem::path& manifest_path,
    const JpegDctShardManifest& manifest);

class JpegDctBlockMajorAccessDescriptor {
public:
	static JpegDctBlockMajorAccessDescriptor Open(const std::filesystem::path& descriptor_path);

	JpegDctBlockMajorAccessDescriptor(JpegDctBlockMajorAccessDescriptor&&) noexcept;
	JpegDctBlockMajorAccessDescriptor& operator=(JpegDctBlockMajorAccessDescriptor&&) noexcept;
	~JpegDctBlockMajorAccessDescriptor();

	JpegDctBlockMajorAccessDescriptor(const JpegDctBlockMajorAccessDescriptor&)            = delete;
	JpegDctBlockMajorAccessDescriptor& operator=(const JpegDctBlockMajorAccessDescriptor&) = delete;

	[[nodiscard]] uint32_t shard_id() const noexcept;
	[[nodiscard]] uint64_t first_global_image_index() const noexcept;
	[[nodiscard]] uint32_t image_count() const noexcept;
	[[nodiscard]] uint32_t group_count() const noexcept;
	[[nodiscard]] uint32_t rank_cell_count() const noexcept;
	[[nodiscard]] uint32_t semantic_slot_count() const noexcept;
	[[nodiscard]] size_t   descriptor_bytes() const noexcept;
	[[nodiscard]] uint64_t descriptor_crc64() const noexcept;

	[[nodiscard]] JpegDctBlockMajorAccessImageRecord image(uint32_t local_image_index) const;
	[[nodiscard]] JpegDctBlockMajorAccessComponentRecord component(uint32_t component_index) const;
	[[nodiscard]] JpegDctBlockMajorAccessRankCellRecord rank_cell(uint32_t cell_id) const;
	[[nodiscard]] JpegDctBlockMajorAccessQuantTableRecord quant_table(uint16_t dictionary_id) const;
	[[nodiscard]] JpegDctBlockMajorAccessRank Rank(
	    uint32_t semantic_slot_id, uint32_t block_x, uint32_t block_y, uint32_t local_image_index) const;
	[[nodiscard]] uint32_t RankBefore(
	    uint32_t semantic_slot_id, uint32_t block_x, uint32_t block_y, uint32_t local_image_exclusive) const;
	// Resolve both endpoints against an already-resolved rank cell.  This avoids
	// repeating coordinate/topology lookup for every group in a compact batch
	// plan and scans sparse/missing payloads at most once for the interval.
	[[nodiscard]] JpegDctBlockMajorAccessRankInterval RankCellInterval(
	    uint32_t cell_id, uint32_t local_image_begin, uint32_t local_image_end) const;
	[[nodiscard]] std::optional<uint32_t> SelectCell(uint32_t cell_id, uint32_t present_rank) const;
	[[nodiscard]] std::optional<uint32_t>
	Select(uint32_t semantic_slot_id, uint32_t block_x, uint32_t block_y, uint32_t present_rank) const;
	[[nodiscard]] std::optional<JpegDctBlockMajorAccessGroup>
	FindGroup(uint32_t semantic_slot_id, uint32_t block_x, uint32_t block_y) const;
	// Resolve a coordinate batch in bounded sequential topology scans. Results
	// preserve input order; row_count==0 denotes an absent/out-of-range group.
	[[nodiscard]] std::vector<JpegDctBlockMajorAccessGroup> FindGroups(
	    uint32_t semantic_slot_id, const std::vector<std::array<uint32_t, 2>>& block_coordinates) const;

	void ValidateSource(const std::filesystem::path& manifest_path,
	                    const JpegDctShardManifestEntry& manifest_entry,
	                    const std::filesystem::path& metadata_path,
	                    const std::filesystem::path& fls_path) const;
	[[nodiscard]] JpegDctBlockMajorAccessValidationReport
	ValidateAgainstMetadata(const JpegDctDatasetMetadata& metadata, bool exhaustive_rank_validation) const;

private:
	struct Impl;
	explicit JpegDctBlockMajorAccessDescriptor(std::unique_ptr<Impl> impl) noexcept;
	std::unique_ptr<Impl> impl_;
};

JpegDctBlockMajorAccessShardReport build_jpeg_dct_block_major_access_descriptor(
    const std::filesystem::path& manifest_path,
    const JpegDctShardManifest& manifest,
    const JpegDctShardManifestEntry& shard,
    const std::filesystem::path& output_path,
    const JpegDctBlockMajorAccessBuildOptions& options = {});

JpegDctBlockMajorAccessDatasetReport build_jpeg_dct_block_major_access_dataset(
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& output_directory = {},
    const JpegDctBlockMajorAccessBuildOptions& options = {});

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_BLOCK_MAJOR_ACCESS_HPP
