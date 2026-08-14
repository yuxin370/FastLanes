#ifndef GALP_FORMAT_COMPACT_DESCRIPTOR_V3_HPP
#define GALP_FORMAT_COMPACT_DESCRIPTOR_V3_HPP

#include "fls/footer/rowgroup_descriptor_generated.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace galp::format {

inline constexpr uint32_t kCompactDescriptorV3Version = 3U;

struct CompactV3ComponentInput {
	uint32_t semantic_slot_id        = 0U;
	uint32_t width_in_blocks         = 0U;
	uint32_t height_in_blocks        = 0U;
	uint32_t padded_width_in_blocks  = 0U;
	uint32_t padded_height_in_blocks = 0U;
	uint32_t row_offset              = 0U;
	uint32_t component_index         = 0U;
};

struct CompactV3ImageInput {
	uint32_t                             first_rowgroup     = 0U;
	uint32_t                             rowgroup_count     = 0U;
	uint32_t                             real_row_count     = 0U;
	uint64_t                             first_physical_row = 0U;
	std::vector<CompactV3ComponentInput> components;
};

struct CompactV3BuildOptions {
	uint32_t                         vector_size   = 1024U;
	uint32_t                         spatial_order = 0U;
	std::vector<CompactV3ImageInput> images;
};

struct CompactV3RowgroupRecord {
	uint64_t payload_offset           = 0U;
	uint32_t payload_size             = 0U;
	uint32_t real_row_count           = 0U;
	uint32_t local_image_index        = 0U;
	uint32_t image_local_vector_index = 0U;
	uint64_t payload_crc64            = 0U;
};

struct CompactV3ImageRecord {
	uint32_t first_rowgroup     = 0U;
	uint32_t rowgroup_count     = 0U;
	uint32_t real_row_count     = 0U;
	uint32_t first_component    = 0U;
	uint32_t component_count    = 0U;
	uint32_t spatial_order      = 0U;
	uint64_t first_physical_row = 0U;
};

struct CompactV3CoefficientRange {
	uint32_t offset = 0U;
	uint32_t size   = 0U;
};

// Canonical v3 files store one vector per rowgroup and encode only the
// rowgroup-varying segment geometry in their scalar page.  The ordinary
// expansion API below remains useful for compatibility and diagnostics, but
// rebuilding 64 native ColumnDescriptor objects and another FlatBuffer for
// every vector rowgroup is needlessly expensive in the training hot path.
// These lightweight views retain the normalized schema pointers from the
// descriptor mmap and materialize only the dynamic bytes and segment ranges.
struct CompactV3DirectSegment {
	uint64_t                  entrypoint_offset = 0U;
	uint64_t                  entrypoint_size   = 0U;
	uint64_t                  data_offset       = 0U;
	uint64_t                  data_size         = 0U;
	fastlanes::EntryPointType entry_point_type  = fastlanes::EntryPointType::UINT8;
};

struct CompactV3DirectColumn {
	const fastlanes::ColumnDescriptor* schema          = nullptr;
	uint32_t                           segment_begin   = 0U;
	uint32_t                           segment_count   = 0U;
	uint32_t                           maximum_offset  = 0U;
	uint32_t                           maximum_size    = 0U;
};

struct CompactV3DirectRowgroup {
	CompactV3RowgroupRecord             record;
	std::vector<CompactV3DirectColumn>  columns;
	std::vector<CompactV3DirectSegment> segments;
	std::vector<uint8_t>                maximum_bytes;
};

struct CompactV3Report {
	uint64_t source_file_bytes        = 0U;
	uint64_t output_file_bytes        = 0U;
	uint64_t payload_bytes            = 0U;
	uint64_t source_descriptor_bytes  = 0U;
	uint64_t compact_descriptor_bytes = 0U;
	uint64_t payload_crc64            = 0U;
	uint64_t rowgroup_count           = 0U;
	uint64_t schema_count             = 0U;
	double   descriptor_reduction     = 0.0;
};

struct CompactV3PayloadAudit {
	uint64_t              expected_payload_crc64 = 0U;
	uint64_t              actual_payload_crc64   = 0U;
	uint64_t              payload_bytes          = 0U;
	uint64_t              rowgroup_count         = 0U;
	std::vector<uint64_t> actual_rowgroup_crc64;
	std::vector<uint32_t> rowgroup_crc_mismatches;

	[[nodiscard]] bool exact() const noexcept {
		return expected_payload_crc64 == actual_payload_crc64 && rowgroup_crc_mismatches.empty();
	}
};

struct CompactDescriptorV3MappingStats {
	size_t current_mapping_count = 0U;
	size_t peak_mapping_count    = 0U;
	size_t map_count             = 0U;
	size_t unmap_count           = 0U;
	size_t current_mapped_bytes  = 0U;
	size_t peak_mapped_bytes     = 0U;
};

[[nodiscard]] CompactDescriptorV3MappingStats compact_descriptor_v3_mapping_stats() noexcept;

class CompactDescriptorV3 {
public:
	static CompactDescriptorV3 Open(const std::filesystem::path& shard_path);

	CompactDescriptorV3(CompactDescriptorV3&&) noexcept;
	CompactDescriptorV3& operator=(CompactDescriptorV3&&) noexcept;
	~CompactDescriptorV3();

	CompactDescriptorV3(const CompactDescriptorV3&)            = delete;
	CompactDescriptorV3& operator=(const CompactDescriptorV3&) = delete;

	[[nodiscard]] uint64_t payload_bytes() const noexcept;
	[[nodiscard]] uint64_t payload_crc64() const noexcept;
	[[nodiscard]] uint32_t vector_size() const noexcept;
	[[nodiscard]] uint32_t spatial_order() const noexcept;
	[[nodiscard]] size_t   column_count() const noexcept;
	[[nodiscard]] size_t   rowgroup_count() const noexcept;
	[[nodiscard]] size_t   image_count() const noexcept;
	[[nodiscard]] size_t   schema_count() const noexcept;
	[[nodiscard]] size_t   descriptor_bytes() const noexcept;
	// Drop resident file-backed descriptor pages after validation/prewarm while
	// preserving the immutable mapping and all pointer identities. Pages needed
	// by a later rowgroup fault back in from the verified file.
	void release_resident_pages() const noexcept;

	[[nodiscard]] CompactV3RowgroupRecord                rowgroup(size_t rowgroup_index) const;
	[[nodiscard]] CompactV3ImageRecord                   image(size_t image_index) const;
	[[nodiscard]] CompactV3ComponentInput                component(size_t component_index) const;
	[[nodiscard]] std::vector<CompactV3CoefficientRange> coefficient_ranges(size_t rowgroup_index) const;
	[[nodiscard]] CompactV3CoefficientRange coefficient_range(size_t rowgroup_index, size_t coefficient_index) const;
	[[nodiscard]] bool supports_direct_rowgroup_geometry() const noexcept;
	[[nodiscard]] CompactV3DirectRowgroup decode_direct_rowgroup(size_t rowgroup_index) const;
	[[nodiscard]] std::unique_ptr<fastlanes::RowgroupDescriptorT> unpack_rowgroup(size_t rowgroup_index) const;

private:
	struct Impl;
	explicit CompactDescriptorV3(std::unique_ptr<Impl> impl) noexcept;
	std::unique_ptr<Impl> impl_;
};

[[nodiscard]] bool is_compact_v3_fls(const std::filesystem::path& shard_path);

CompactV3Report compact_standard_fls_to_v3(const std::filesystem::path& input_path,
                                           const std::filesystem::path& output_path,
                                           const CompactV3BuildOptions& options = {});

CompactV3Report expand_compact_fls_v3(const std::filesystem::path& input_path,
                                      const std::filesystem::path& output_path);

CompactV3PayloadAudit verify_compact_v3_payload(const std::filesystem::path& input_path);

} // namespace galp::format

#endif // GALP_FORMAT_COMPACT_DESCRIPTOR_V3_HPP
