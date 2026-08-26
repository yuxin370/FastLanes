#include "jpeg/jpeg_dct_metadata.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace galp::jpeg {
namespace {

constexpr uint16_t kMetadataLayoutZigzagColumns       = 1U << 0U;
constexpr uint16_t kMetadataLayoutLegacyZOrder        = 1U << 1U;
constexpr uint16_t kMetadataLayoutSpatialOrderPresent = 1U << 2U;
constexpr uint16_t kMetadataLayoutSpatialOrderShift   = 3U;
constexpr uint16_t kMetadataLayoutSpatialOrderMask    = 3U << kMetadataLayoutSpatialOrderShift;
constexpr uint16_t kMetadataLayoutKnownMask           = kMetadataLayoutZigzagColumns | kMetadataLayoutLegacyZOrder |
                                              kMetadataLayoutSpatialOrderPresent | kMetadataLayoutSpatialOrderMask;
constexpr uint16_t kRaggedValidationModeOnDiskId = 2;

uint16_t spatial_order_id(const JpegDctSpatialOrder order) {
	switch (order) {
	case JpegDctSpatialOrder::kRaster:
		return 0;
	case JpegDctSpatialOrder::kTiledRaster32:
		return 1;
	case JpegDctSpatialOrder::kZOrder:
		return 2;
	case JpegDctSpatialOrder::kTiledZ32:
		return 3;
	}
	throw std::runtime_error("unknown JPEG DCT spatial order");
}

JpegDctSpatialOrder spatial_order_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegDctSpatialOrder::kRaster;
	case 1:
		return JpegDctSpatialOrder::kTiledRaster32;
	case 2:
		return JpegDctSpatialOrder::kZOrder;
	case 3:
		return JpegDctSpatialOrder::kTiledZ32;
	default:
		throw std::runtime_error("unknown JPEG DCT spatial order id in metadata");
	}
}

uint16_t metadata_layout_flags(const JpegDctDatasetMetadata& metadata, const bool sectioned) {
	uint16_t flags = 0;
	if (metadata.zigzag_columns) {
		flags |= kMetadataLayoutZigzagColumns;
	}
	if (metadata.z_curve_block_order) {
		flags |= kMetadataLayoutLegacyZOrder;
	}
	if (sectioned && metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
		flags |= kMetadataLayoutSpatialOrderPresent;
		flags |= static_cast<uint16_t>(spatial_order_id(metadata.image_major_spatial_order)
		                               << kMetadataLayoutSpatialOrderShift);
	}
	return flags;
}

int row_ordering_id(const JpegDctRowOrdering ordering) {
	switch (ordering) {
	case JpegDctRowOrdering::kSingleImageComponentMajorBlockMajor:
		return 0;
	case JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor:
		return 1;
	case JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor:
		return 2;
	}
	return -1;
}

JpegDctRowOrdering row_ordering_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegDctRowOrdering::kSingleImageComponentMajorBlockMajor;
	case 1:
		return JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	case 2:
		return JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor;
	default:
		throw std::runtime_error("unknown JPEG DCT row ordering id in metadata");
	}
}

uint16_t metadata_profile_id(const JpegMetadataProfile profile) {
	switch (profile) {
	case JpegMetadataProfile::kDctDatasetOnly:
		return 0;
	case JpegMetadataProfile::kReconstructableJpeg:
		return 1;
	case JpegMetadataProfile::kPreserveOriginalMarkers:
		return 2;
	}
	return 0;
}

JpegCompressionPartitionPolicy partition_policy_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegCompressionPartitionPolicy::kBySemanticSlot;
	case 1:
		return JpegCompressionPartitionPolicy::kByEncodingProfile;
	default:
		throw std::runtime_error("unknown JPEG DCT compression partition policy id in metadata");
	}
}

JpegDctCoefficientEncoding coefficient_encoding_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegDctCoefficientEncoding::kDense64FastLanes;
	case 1:
		return JpegDctCoefficientEncoding::kExpCrossRleI16;
	case 2:
		return JpegDctCoefficientEncoding::kDcDeltaDense64FastLanes;
	case 3:
		return JpegDctCoefficientEncoding::kDcDeltaAcSparseRle;
	case 4:
		return JpegDctCoefficientEncoding::kJpegLikeRunLength;
	default:
		throw std::runtime_error("unknown JPEG DCT coefficient encoding id in metadata");
	}
}

class BinaryWriter {
public:
	explicit BinaryWriter(std::ostream& output)
	    : output_(output) {
	}

	void u8(const uint8_t value) {
		output_.put(static_cast<char>(value));
	}
	void u16(const uint16_t value) {
		u8(static_cast<uint8_t>(value & 0xffU));
		u8(static_cast<uint8_t>((value >> 8U) & 0xffU));
	}
	void u32(const uint32_t value) {
		for (unsigned shift = 0; shift < 32; shift += 8) {
			u8(static_cast<uint8_t>((value >> shift) & 0xffU));
		}
	}
	void u64(const uint64_t value) {
		for (unsigned shift = 0; shift < 64; shift += 8) {
			u8(static_cast<uint8_t>((value >> shift) & 0xffU));
		}
	}
	void i32(const int32_t value) {
		u32(static_cast<uint32_t>(value));
	}
	void bytes(const uint8_t* data, const size_t size) {
		output_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
	}
	void string(const std::string& value) {
		u32(static_cast<uint32_t>(value.size()));
		output_.write(value.data(), static_cast<std::streamsize>(value.size()));
	}
	[[nodiscard]] std::ostream& output() noexcept {
		return output_;
	}

private:
	std::ostream& output_;
};

class BinaryReader {
public:
	explicit BinaryReader(std::vector<uint8_t> input)
	    : data_(std::move(input)) {
	}

	[[nodiscard]] bool eof() const noexcept {
		return position_ == data_.size();
	}
	[[nodiscard]] size_t remaining() const noexcept {
		return data_.size() - position_;
	}
	uint8_t u8() {
		require(1);
		return data_[position_++];
	}
	uint16_t u16() {
		uint16_t value = 0;
		for (unsigned shift = 0; shift < 16; shift += 8) {
			value |= static_cast<uint16_t>(u8()) << shift;
		}
		return value;
	}
	uint32_t u32() {
		uint32_t value = 0;
		for (unsigned shift = 0; shift < 32; shift += 8) {
			value |= static_cast<uint32_t>(u8()) << shift;
		}
		return value;
	}
	uint64_t u64() {
		uint64_t value = 0;
		for (unsigned shift = 0; shift < 64; shift += 8) {
			value |= static_cast<uint64_t>(u8()) << shift;
		}
		return value;
	}
	int32_t i32() {
		return static_cast<int32_t>(u32());
	}
	std::vector<uint8_t> bytes(const size_t size) {
		require(size);
		std::vector<uint8_t> result(data_.begin() + static_cast<std::ptrdiff_t>(position_),
		                            data_.begin() + static_cast<std::ptrdiff_t>(position_ + size));
		position_ += size;
		return result;
	}
	std::string string() {
		const auto size = u32();
		require(size);
		std::string result(reinterpret_cast<const char*>(data_.data() + position_), size);
		position_ += size;
		return result;
	}

private:
	void require(const size_t size) const {
		if (size > remaining()) {
			throw std::runtime_error("truncated JPEG DCT binary metadata");
		}
	}

	std::vector<uint8_t> data_;
	size_t               position_ = 0;
};

enum class MetadataSection : uint16_t {
	kComponentGrid            = 1,
	kPerImageGrid             = 2,
	kReconstructableImageInfo = 3,
	kOriginalMarkers          = 4,
	kEncodingProfiles         = 5,
	kBlockGroupIndex          = 6,
	kImageGroupIndex          = 7,
};

template <typename Function>
std::vector<uint8_t> build_section(Function&& function) {
	std::ostringstream payload(std::ios::binary);
	BinaryWriter       writer(payload);
	function(writer);
	const auto string = payload.str();
	return {string.begin(), string.end()};
}

void write_section(BinaryWriter& writer, const MetadataSection id, const std::vector<uint8_t>& payload) {
	writer.u16(static_cast<uint16_t>(id));
	writer.u64(static_cast<uint64_t>(payload.size()));
	writer.bytes(payload.data(), payload.size());
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
	std::ifstream input(path, std::ios::binary);
	if (!input) {
		throw std::runtime_error("failed to open binary file: " + path.string());
	}
	return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void expect_magic(BinaryReader& reader, const std::array<uint8_t, 8>& expected, const char* label) {
	for (const auto byte : expected) {
		if (reader.u8() != byte) {
			throw std::runtime_error(std::string("invalid ") + label + " magic");
		}
	}
}

const std::vector<JpegComponentMetadata>* legacy_metadata_components(const JpegDctDatasetMetadata& metadata) {
	if (!metadata.images.empty()) {
		return &metadata.images.front().components;
	}
	return metadata.semantic_components.empty() ? nullptr : &metadata.semantic_components;
}

} // namespace

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata& metadata, const std::filesystem::path& output_path) {
	std::ofstream output(output_path, std::ios::binary);
	if (!output) {
		throw std::runtime_error("failed to open JPEG DCT metadata output: " + output_path.string());
	}
	const auto*   components      = legacy_metadata_components(metadata);
	const auto    component_count = components == nullptr ? 0 : components->size();
	BinaryWriter  writer(output);
	const uint8_t magic[8] {'G', 'J', 'D', 'C', 'T', 'M', 'D', '1'};
	writer.bytes(magic, sizeof(magic));
	writer.u16(1);
	writer.u16(static_cast<uint16_t>(row_ordering_id(metadata.row_ordering)));
	writer.u16(kRaggedValidationModeOnDiskId);
	writer.u16(metadata_layout_flags(metadata, false));
	writer.u64(static_cast<uint64_t>(metadata.image_count));
	writer.u32(static_cast<uint32_t>(component_count));
	for (size_t index = 0; index < component_count; ++index) {
		const auto& component = (*components)[index];
		writer.u32(static_cast<uint32_t>(component.component_index));
		writer.i32(component.component_id);
		writer.u32(component.width_in_blocks);
		writer.u32(component.height_in_blocks);
		writer.u32(component.padded_width_in_blocks);
		writer.u32(component.padded_height_in_blocks);
		writer.i32(component.h_samp_factor);
		writer.i32(component.v_samp_factor);
	}
	const bool has_image_records =
	    metadata.row_ordering == JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor ||
	    metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor;
	writer.u8(has_image_records ? 1U : 0U);
	if (has_image_records) {
		for (const auto& image : metadata.images) {
			for (const auto& component : image.components) {
				writer.u8(component.present ? 1U : 0U);
				writer.u32(component.width_in_blocks);
				writer.u32(component.height_in_blocks);
			}
		}
	}
}

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata&       metadata,
                             const std::filesystem::path&        output_path,
                             const JpegDctMetadataWriterOptions& options) {
	std::ofstream output(output_path, std::ios::binary);
	if (!output) {
		throw std::runtime_error("failed to open JPEG DCT metadata output: " + output_path.string());
	}
	const auto* components = metadata.semantic_components.empty() ? nullptr : &metadata.semantic_components;
	if (components == nullptr && !metadata.images.empty()) {
		components = &metadata.images.front().components;
	}
	const auto    component_count = components == nullptr ? 0 : components->size();
	BinaryWriter  writer(output);
	const uint8_t magic[8] {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'};
	writer.bytes(magic, sizeof(magic));
	writer.u16(3);
	writer.u16(metadata_profile_id(options.profile));
	writer.u16(static_cast<uint16_t>(row_ordering_id(metadata.row_ordering)));
	writer.u16(kRaggedValidationModeOnDiskId);
	writer.u16(metadata_layout_flags(metadata, true));
	writer.u64(static_cast<uint64_t>(metadata.image_count));
	writer.u32(static_cast<uint32_t>(component_count));
	writer.u32(static_cast<uint32_t>(metadata.encoding_profiles.size()));
	writer.u16(static_cast<uint16_t>(metadata.compression_partition_policy));
	writer.u16(static_cast<uint16_t>(metadata.coefficient_encoding));

	write_section(writer, MetadataSection::kComponentGrid, build_section([&](BinaryWriter& section) {
		              for (size_t index = 0; index < component_count; ++index) {
			              const auto& component = (*components)[index];
			              section.u32(component.semantic_slot_id);
			              section.u32(static_cast<uint32_t>(component.component_index));
			              section.i32(component.component_id);
			              section.u32(component.width_in_blocks);
			              section.u32(component.height_in_blocks);
			              section.u32(component.padded_width_in_blocks);
			              section.u32(component.padded_height_in_blocks);
		              }
	              }));
	write_section(writer, MetadataSection::kEncodingProfiles, build_section([&](BinaryWriter& section) {
		              for (const auto& profile : metadata.encoding_profiles) {
			              section.u32(profile.profile_id);
			              section.i32(profile.h_samp_factor);
			              section.i32(profile.v_samp_factor);
			              section.i32(profile.quant_tbl_no);
			              section.u64(profile.quant_table_fingerprint);
			              for (const auto value : profile.quant_table_values) {
				              section.u16(value);
			              }
		              }
	              }));
	write_section(writer, MetadataSection::kBlockGroupIndex, build_section([&](BinaryWriter& section) {
		              for (const auto& group : metadata.block_group_index) {
			              section.u32(group.semantic_slot_id);
			              section.u32(group.z_order_index);
			              section.u32(group.block_x);
			              section.u32(group.block_y);
			              section.u64(group.row_start);
			              section.u32(group.row_count);
			              section.u32(group.fls_rowgroup_index);
			              section.u32(group.row_start_in_rowgroup);
		              }
	              }));
	write_section(writer, MetadataSection::kImageGroupIndex, build_section([&](BinaryWriter& section) {
		              for (const auto& group : metadata.image_group_index) {
			              section.u32(group.local_image_index);
			              section.u64(group.row_start);
			              section.u32(group.row_count);
			              section.u32(group.fls_rowgroup_index);
			              section.u32(group.row_start_in_rowgroup);
		              }
	              }));

	if (!metadata.images.empty()) {
		write_section(writer, MetadataSection::kPerImageGrid, build_section([&](BinaryWriter& section) {
			              for (const auto& image : metadata.images) {
				              section.u32(image.image_width);
				              section.u32(image.image_height);
				              section.u8(image.data_precision);
				              section.i32(image.jpeg_color_space);
				              section.u32(image.warning_count);
				              for (const auto& component : image.components) {
					              section.u8(component.present ? 1U : 0U);
					              section.u32(component.semantic_slot_id);
					              section.u32(static_cast<uint32_t>(component.local_component_index));
					              section.i32(component.component_id);
					              section.u32(component.width_in_blocks);
					              section.u32(component.height_in_blocks);
					              section.u32(component.encoding_profile_id);
				              }
			              }
		              }));
	}

	if (options.profile == JpegMetadataProfile::kReconstructableJpeg ||
	    options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		write_section(writer, MetadataSection::kReconstructableImageInfo, build_section([&](BinaryWriter& section) {
			              section.u64(static_cast<uint64_t>(metadata.images.size()));
			              for (const auto& image : metadata.images) {
				              section.u32(image.image_width);
				              section.u32(image.image_height);
				              section.u8(image.data_precision);
				              section.i32(image.jpeg_color_space);
				              section.u32(static_cast<uint32_t>(image.quant_tables.size()));
				              for (const auto& table : image.quant_tables) {
					              section.u8(table.table_id);
					              for (const auto value : table.values) {
						              section.u16(value);
					              }
				              }
			              }
		              }));
	}
	if (options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		write_section(writer, MetadataSection::kOriginalMarkers, build_section([&](BinaryWriter& section) {
			              section.u64(static_cast<uint64_t>(metadata.images.size()));
			              for (const auto& image : metadata.images) {
				              section.u32(static_cast<uint32_t>(image.markers.size()));
				              for (const auto& marker : image.markers) {
					              section.u8(marker.marker);
					              section.u64(static_cast<uint64_t>(marker.payload.size()));
					              section.bytes(marker.payload.data(), marker.payload.size());
				              }
			              }
		              }));
	}
}

void write_jpeg_dct_shard_manifest(const JpegDctShardManifest& manifest, const std::filesystem::path& output_path) {
	std::ofstream output(output_path, std::ios::binary);
	if (!output) {
		throw std::runtime_error("failed to open JPEG DCT shard manifest output: " + output_path.string());
	}
	BinaryWriter  writer(output);
	const uint8_t magic[8] {'G', 'J', 'D', 'C', 'T', 'S', 'H', '1'};
	writer.bytes(magic, sizeof(magic));
	writer.u32(manifest.version);
	writer.u16(kRaggedValidationModeOnDiskId);
	writer.u32(manifest.rowgroup_vectors);
	writer.u32(manifest.rowgroups_per_shard);
	writer.u64(manifest.image_count);
	writer.u32(static_cast<uint32_t>(manifest.shards.size()));
	for (const auto& shard : manifest.shards) {
		writer.u32(shard.shard_id);
		writer.u64(shard.first_global_image_index);
		writer.u32(shard.image_count);
		writer.u64(shard.real_row_count);
		writer.u64(shard.padding_row_count);
		writer.u64(shard.physical_row_count);
		writer.u32(shard.rowgroup_count);
		writer.u32(shard.block_group_count);
		writer.u64(shard.fls_file_size);
		writer.u64(shard.metadata_file_size);
		writer.string(shard.fls_file_name);
		writer.string(shard.metadata_file_name);
	}
	if (manifest.version == 3U && !manifest.descriptor_kind.empty()) {
		const uint8_t extension_magic[8] {'G', 'J', 'D', 'C', 'C', 'V', '3', '1'};
		writer.bytes(extension_magic, sizeof(extension_magic));
		writer.string(manifest.physical_layout);
		writer.string(manifest.descriptor_kind);
		writer.u32(manifest.vector_size);
		writer.string(manifest.spatial_order_name);
		writer.u16(spatial_order_id(manifest.spatial_order));
		writer.u32(static_cast<uint32_t>(manifest.shards.size()));
		for (const auto& shard : manifest.shards) {
			writer.u32(shard.shard_id);
			writer.u64(shard.payload_size);
			writer.u64(shard.payload_crc64);
			writer.u64(shard.compact_descriptor_size);
			writer.u64(shard.source_descriptor_size);
		}
	}
}

namespace detail {

JpegDctShardManifest read_jpeg_dct_shard_manifest_file(const std::filesystem::path& path) {
	BinaryReader reader(read_binary_file(path));
	expect_magic(reader, {'G', 'J', 'D', 'C', 'T', 'S', 'H', '1'}, "JPEG DCT shard manifest");
	JpegDctShardManifest manifest;
	manifest.version = reader.u32();
	static_cast<void>(reader.u16());
	manifest.rowgroup_vectors    = reader.u32();
	manifest.rowgroups_per_shard = reader.u32();
	manifest.image_count         = reader.u64();
	const auto shard_count       = reader.u32();
	manifest.shards.reserve(shard_count);
	for (uint32_t index = 0; index < shard_count; ++index) {
		JpegDctShardManifestEntry entry;
		entry.shard_id                 = reader.u32();
		entry.first_global_image_index = reader.u64();
		entry.image_count              = reader.u32();
		entry.real_row_count           = reader.u64();
		entry.padding_row_count        = reader.u64();
		entry.physical_row_count       = reader.u64();
		entry.rowgroup_count           = reader.u32();
		entry.block_group_count        = reader.u32();
		entry.fls_file_size            = reader.u64();
		entry.metadata_file_size       = reader.u64();
		entry.fls_file_name            = reader.string();
		entry.metadata_file_name       = reader.string();
		manifest.shards.push_back(std::move(entry));
	}
	if (!reader.eof()) {
		expect_magic(reader, {'G', 'J', 'D', 'C', 'C', 'V', '3', '1'}, "JPEG DCT Compact v3 extension");
		manifest.physical_layout = reader.string();
		manifest.descriptor_kind = reader.string();
		manifest.vector_size     = reader.u32();
		manifest.spatial_order_name = reader.string();
		manifest.spatial_order   = spatial_order_from_id(reader.u16());
		const auto extension_shard_count = reader.u32();
		if (extension_shard_count != manifest.shards.size()) {
			throw std::runtime_error("JPEG DCT Compact v3 extension shard count mismatch");
		}
		for (uint32_t index = 0U; index < extension_shard_count; ++index) {
			const auto shard_id = reader.u32();
			if (shard_id != manifest.shards[index].shard_id) {
				throw std::runtime_error("JPEG DCT Compact v3 extension shard id mismatch");
			}
			auto& shard                     = manifest.shards[index];
			shard.payload_size              = reader.u64();
			shard.payload_crc64             = reader.u64();
			shard.compact_descriptor_size   = reader.u64();
			shard.source_descriptor_size    = reader.u64();
		}
		if (!reader.eof()) {
			throw std::runtime_error("JPEG DCT Compact v3 extension has trailing bytes");
		}
	}
	// Manifest v1/v2 predate the explicit layout string carried by the v3
	// extension. Normalize their version-defined layouts at the reader boundary
	// so consumers cannot accidentally treat an empty legacy field as an
	// unknown or different physical layout.
	if (manifest.version == 1U && manifest.physical_layout.empty()) {
		manifest.physical_layout = "dct-major/spatial-major-image-minor";
	} else if (manifest.version == 2U && manifest.physical_layout.empty()) {
		manifest.physical_layout = "image-major";
	}
	return manifest;
}

JpegDctDatasetMetadata read_jpeg_dct_metadata_file(const std::filesystem::path& path) {
	BinaryReader reader(read_binary_file(path));
	expect_magic(reader, {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'}, "JPEG DCT metadata");
	JpegDctDatasetMetadata metadata;
	if (reader.u16() != 3U) {
		throw std::runtime_error("unsupported JPEG DCT metadata version");
	}
	static_cast<void>(reader.u16());
	metadata.row_ordering = row_ordering_from_id(reader.u16());
	static_cast<void>(reader.u16());
	const auto layout_flags = reader.u16();
	if ((layout_flags & ~kMetadataLayoutKnownMask) != 0) {
		throw std::runtime_error("JPEG DCT metadata contains unknown layout flags");
	}
	metadata.zigzag_columns      = (layout_flags & kMetadataLayoutZigzagColumns) != 0;
	metadata.z_curve_block_order = (layout_flags & kMetadataLayoutLegacyZOrder) != 0;
	if ((layout_flags & kMetadataLayoutSpatialOrderPresent) != 0) {
		if (metadata.row_ordering != JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
			throw std::runtime_error("JPEG DCT spatial order flag is only valid for image-major metadata");
		}
		metadata.image_major_spatial_order = spatial_order_from_id(static_cast<uint16_t>(
		    (layout_flags & kMetadataLayoutSpatialOrderMask) >> kMetadataLayoutSpatialOrderShift));
	} else if (metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
		metadata.image_major_spatial_order = JpegDctSpatialOrder::kRaster;
	}
	metadata.image_count                  = reader.u64();
	const auto component_count            = reader.u32();
	const auto profile_count              = reader.u32();
	metadata.compression_partition_policy = partition_policy_from_id(reader.u16());
	metadata.coefficient_encoding         = coefficient_encoding_from_id(reader.u16());
	metadata.encoding_profiles.reserve(profile_count);

	while (!reader.eof()) {
		const auto section_id   = static_cast<MetadataSection>(reader.u16());
		const auto payload_size = reader.u64();
		if (payload_size > std::numeric_limits<size_t>::max()) {
			throw std::runtime_error("JPEG DCT metadata section is too large");
		}
		BinaryReader section(reader.bytes(static_cast<size_t>(payload_size)));
		switch (section_id) {
		case MetadataSection::kComponentGrid:
			while (!section.eof()) {
				JpegComponentMetadata component;
				component.semantic_slot_id        = section.u32();
				component.component_index         = section.u32();
				component.component_id            = section.i32();
				component.width_in_blocks         = section.u32();
				component.height_in_blocks        = section.u32();
				component.padded_width_in_blocks  = section.u32();
				component.padded_height_in_blocks = section.u32();
				metadata.semantic_components.push_back(component);
			}
			break;
		case MetadataSection::kPerImageGrid:
			metadata.images.reserve(metadata.image_count);
			for (uint64_t image_index = 0; image_index < metadata.image_count; ++image_index) {
				JpegImageMetadata image;
				image.image_width      = section.u32();
				image.image_height     = section.u32();
				image.data_precision   = section.u8();
				image.jpeg_color_space = section.i32();
				image.warning_count    = section.u32();
				image.components.reserve(component_count);
				for (uint32_t component_index = 0; component_index < component_count; ++component_index) {
					JpegComponentMetadata component;
					component.present               = section.u8() != 0;
					component.semantic_slot_id      = section.u32();
					component.local_component_index = section.u32();
					component.component_id          = section.i32();
					component.width_in_blocks       = section.u32();
					component.height_in_blocks      = section.u32();
					component.encoding_profile_id   = section.u32();
					image.components.push_back(component);
				}
				metadata.images.push_back(std::move(image));
			}
			break;
		case MetadataSection::kBlockGroupIndex:
			while (!section.eof()) {
				JpegDctBlockGroupIndex group;
				group.semantic_slot_id = section.u32();
				group.z_order_index    = section.u32();
				group.block_x          = section.u32();
				group.block_y          = section.u32();
				group.row_start        = section.u64();
				group.row_count        = section.u32();
				if (section.remaining() >= 8) {
					group.fls_rowgroup_index    = section.u32();
					group.row_start_in_rowgroup = section.u32();
				}
				metadata.block_group_index.push_back(group);
			}
			break;
		case MetadataSection::kImageGroupIndex:
			while (!section.eof()) {
				JpegDctImageGroupIndex group;
				group.local_image_index     = section.u32();
				group.row_start             = section.u64();
				group.row_count             = section.u32();
				group.fls_rowgroup_index    = section.u32();
				group.row_start_in_rowgroup = section.u32();
				metadata.image_group_index.push_back(group);
			}
			break;
		case MetadataSection::kEncodingProfiles:
			while (!section.eof()) {
				JpegEncodingProfileMetadata profile;
				profile.profile_id              = section.u32();
				profile.h_samp_factor           = section.i32();
				profile.v_samp_factor           = section.i32();
				profile.quant_tbl_no            = section.i32();
				profile.quant_table_fingerprint = section.u64();
				for (auto& value : profile.quant_table_values) {
					value = section.u16();
				}
				metadata.encoding_profiles.push_back(profile);
			}
			break;
		case MetadataSection::kReconstructableImageInfo: {
			const auto image_count = section.u64();
			if (image_count != metadata.image_count) {
				throw std::runtime_error("JPEG DCT reconstructable metadata image count mismatch");
			}
			if (metadata.images.empty()) {
				metadata.images.resize(static_cast<size_t>(metadata.image_count));
			}
			for (uint64_t image_index = 0; image_index < image_count; ++image_index) {
				auto& image                  = metadata.images.at(static_cast<size_t>(image_index));
				image.image_width            = section.u32();
				image.image_height           = section.u32();
				image.data_precision         = section.u8();
				image.jpeg_color_space       = section.i32();
				const auto quant_table_count = section.u32();
				image.quant_tables.clear();
				image.quant_tables.reserve(quant_table_count);
				for (uint32_t table_index = 0; table_index < quant_table_count; ++table_index) {
					JpegQuantTableMetadata table;
					table.table_id = section.u8();
					for (auto& value : table.values) {
						value = section.u16();
					}
					image.quant_tables.push_back(table);
				}
			}
			break;
		}
		case MetadataSection::kOriginalMarkers:
			break;
		}
	}

	for (auto& image : metadata.images) {
		for (auto& component : image.components) {
			if (component.encoding_profile_id == std::numeric_limits<uint32_t>::max()) {
				continue;
			}
			const auto profile = std::find_if(metadata.encoding_profiles.begin(),
			                                  metadata.encoding_profiles.end(),
			                                  [&](const JpegEncodingProfileMetadata& candidate) {
				                                  return candidate.profile_id == component.encoding_profile_id;
			                                  });
			if (profile == metadata.encoding_profiles.end()) {
				continue;
			}
			component.h_samp_factor           = profile->h_samp_factor;
			component.v_samp_factor           = profile->v_samp_factor;
			component.quant_tbl_no            = profile->quant_tbl_no;
			component.quant_table_fingerprint = profile->quant_table_fingerprint;
			if (profile->quant_tbl_no < 0 || profile->quant_tbl_no > std::numeric_limits<uint8_t>::max()) {
				continue;
			}
			const auto table_id = static_cast<uint8_t>(profile->quant_tbl_no);
			const auto table    = std::find_if(image.quant_tables.begin(),
                                            image.quant_tables.end(),
                                            [&](const auto& value) { return value.table_id == table_id; });
			if (table == image.quant_tables.end()) {
				JpegQuantTableMetadata new_table;
				new_table.table_id = table_id;
				new_table.values   = profile->quant_table_values;
				image.quant_tables.push_back(new_table);
			}
		}
	}
	return metadata;
}

} // namespace detail

JpegDctShardManifest read_jpeg_dct_shard_manifest(const std::filesystem::path& path) {
	return detail::read_jpeg_dct_shard_manifest_file(path);
}

} // namespace galp::jpeg
