#include "galp/jpeg_dct_block_major_access.hpp"

#if GALP_WITH_JPEG_DCT

#include "jpeg/jpeg_dct_metadata.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include "fls/cfg/cfg.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

namespace galp::jpeg {
namespace {

constexpr std::array<uint8_t, 8> kMagic {{'G', 'A', 'L', 'P', 'B', 'M', 'A', '\0'}};
constexpr std::array<uint8_t, 8> kIndexMagic {{'G', 'A', 'L', 'P', 'B', 'M', 'I', '\0'}};
constexpr uint16_t kHeaderBytes             = 256U;
constexpr uint32_t kFlagLittleEndian        = 1U << 0U;
constexpr uint32_t kFlagCrc64Ecma           = 1U << 1U;
constexpr uint32_t kFlagSharedThresholdRank = 1U << 2U;
constexpr uint32_t kFlagTopologyCheckpoints = 1U << 3U;
constexpr uint32_t kKnownFlags =
    kFlagLittleEndian | kFlagCrc64Ecma | kFlagSharedThresholdRank | kFlagTopologyCheckpoints;
constexpr uint32_t kRequiredFlags = kKnownFlags;
constexpr uint64_t kCrc64Polynomial = UINT64_C(0x42f0e1eba9ea3693);

constexpr std::array<uint64_t, 256> make_crc64_table() {
	std::array<uint64_t, 256> table {};
	for (uint64_t byte = 0U; byte < table.size(); ++byte) {
		uint64_t crc = byte << 56U;
		for (unsigned bit = 0U; bit < 8U; ++bit) {
			crc = (crc & (UINT64_C(1) << 63U)) != 0U ? (crc << 1U) ^ kCrc64Polynomial : crc << 1U;
		}
		table[byte] = crc;
	}
	return table;
}

constexpr auto kCrc64Table = make_crc64_table();

constexpr size_t kDescriptorChecksumByte = 24U;
constexpr size_t kSlotRecordBytes         = 64U;
constexpr size_t kCellRecordBytes         = 8U;
constexpr size_t kTopologyRecordBytes     = 24U;
constexpr size_t kImageRecordBytes        = 24U;
constexpr size_t kComponentRecordBytes    = 24U;
constexpr size_t kQuantRecordBytes        = 136U;

constexpr size_t kOffsetSlots      = 144U;
constexpr size_t kOffsetThresholds = 152U;
constexpr size_t kOffsetCells      = 160U;
constexpr size_t kOffsetPayload    = 168U;
constexpr size_t kOffsetTopology   = 176U;
constexpr size_t kOffsetImages     = 184U;
constexpr size_t kOffsetComponents = 192U;
constexpr size_t kOffsetQuants     = 200U;

[[noreturn]] void fail(const std::string& message) {
	throw std::runtime_error("JpegDctBlockMajorAccess: " + message);
}

template <typename T>
void append_le(std::vector<uint8_t>& output, const T value) {
	static_assert(std::is_integral_v<T>);
	using U = std::make_unsigned_t<T>;
	const auto unsigned_value = static_cast<U>(value);
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		output.push_back(static_cast<uint8_t>((unsigned_value >> (byte * 8U)) & static_cast<U>(0xffU)));
	}
}

template <typename T>
void put_le(std::vector<uint8_t>& output, const size_t offset, const T value) {
	static_assert(std::is_integral_v<T>);
	if (offset > output.size() || sizeof(T) > output.size() - offset) {
		fail("internal serializer exceeded its output buffer");
	}
	using U = std::make_unsigned_t<T>;
	const auto unsigned_value = static_cast<U>(value);
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		output[offset + byte] =
		    static_cast<uint8_t>((unsigned_value >> (byte * 8U)) & static_cast<U>(0xffU));
	}
}

template <typename T>
T read_le(const uint8_t* data, const size_t size, const size_t offset, const std::string_view label) {
	static_assert(std::is_integral_v<T>);
	if (offset > size || sizeof(T) > size - offset) {
		fail("truncated " + std::string(label));
	}
	using U = std::make_unsigned_t<T>;
	U value = 0U;
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		value |= static_cast<U>(data[offset + byte]) << (byte * 8U);
	}
	return static_cast<T>(value);
}

void append_uleb128(std::vector<uint8_t>& output, uint32_t value) {
	do {
		uint8_t byte = static_cast<uint8_t>(value & 0x7fU);
		value >>= 7U;
		if (value != 0U) {
			byte |= 0x80U;
		}
		output.push_back(byte);
	} while (value != 0U);
}

uint32_t consume_uleb128(const uint8_t* data, const size_t size, size_t& cursor, const std::string_view label) {
	uint32_t value = 0U;
	for (unsigned shift = 0U; shift < 32U; shift += 7U) {
		if (cursor >= size) {
			fail("truncated ULEB128 " + std::string(label));
		}
		const auto byte = data[cursor++];
		if (shift == 28U && (byte & 0xf0U) != 0U) {
			fail("ULEB128 overflow in " + std::string(label));
		}
		value |= static_cast<uint32_t>(byte & 0x7fU) << shift;
		if ((byte & 0x80U) == 0U) {
			return value;
		}
	}
	fail("overlong ULEB128 in " + std::string(label));
}

uint64_t crc64_update(uint64_t crc, const uint8_t* data, const size_t size) {
	for (size_t index = 0U; index < size; ++index) {
		const auto table_index = static_cast<uint8_t>((crc >> 56U) ^ data[index]);
		crc = kCrc64Table[table_index] ^ (crc << 8U);
	}
	return crc;
}

uint64_t crc64_with_zeroed_field(const uint8_t* data, const size_t size, const size_t checksum_byte) {
	if (size < checksum_byte + sizeof(uint64_t)) {
		fail("descriptor is shorter than its checksum field");
	}
	uint64_t crc = crc64_update(0U, data, checksum_byte);
	const std::array<uint8_t, sizeof(uint64_t)> zeros {};
	crc = crc64_update(crc, zeros.data(), zeros.size());
	return crc64_update(crc,
	                    data + checksum_byte + sizeof(uint64_t),
	                    size - checksum_byte - sizeof(uint64_t));
}

uint64_t compute_descriptor_crc64(const uint8_t* data, const size_t size) {
	return crc64_with_zeroed_field(data, size, kDescriptorChecksumByte);
}

uint64_t crc64_file(const std::filesystem::path& path) {
	std::ifstream input(path, std::ios::binary);
	if (!input) {
		fail("cannot open source for CRC64: " + path.string());
	}
	std::array<uint8_t, 4U * 1024U * 1024U> buffer {};
	uint64_t crc = 0U;
	while (input) {
		input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
		const auto count = input.gcount();
		if (count > 0) {
			crc = crc64_update(crc, buffer.data(), static_cast<size_t>(count));
		}
	}
	if (!input.eof()) {
		fail("failed while reading source for CRC64: " + path.string());
	}
	return crc;
}

uint64_t checked_file_size(const std::filesystem::path& path) {
	std::error_code error;
	const auto size = std::filesystem::file_size(path, error);
	if (error) {
		fail("cannot inspect source file size '" + path.string() + "': " + error.message());
	}
	return size;
}

void align_to(std::vector<uint8_t>& output, const size_t alignment) {
	while (output.size() % alignment != 0U) {
		output.push_back(0U);
	}
}

void write_staged_file(const std::filesystem::path& output_path, const std::vector<uint8_t>& bytes) {
	const auto staged_path = output_path.string() + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()));
	{
		std::ofstream output(staged_path, std::ios::binary | std::ios::trunc);
		if (!output) {
			fail("cannot open staged descriptor output: " + staged_path);
		}
		output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		output.flush();
		if (!output) {
			fail("failed to write staged descriptor output: " + staged_path);
		}
	}
	std::error_code error;
	std::filesystem::rename(staged_path, output_path, error);
	if (error) {
		std::filesystem::remove(staged_path);
		fail("cannot publish descriptor '" + output_path.string() + "': " + error.message());
	}
}

uint64_t rectangle_intersection_count(const uint64_t width,
	                                  const uint64_t height,
	                                  const uint64_t origin_x,
	                                  const uint64_t origin_y,
	                                  const uint64_t size) {
	if (origin_x >= width || origin_y >= height) {
		return 0U;
	}
	return std::min(size, width - origin_x) * std::min(size, height - origin_y);
}

detail::MortonBlockCoord coordinate_at_rank(const uint32_t width,
	                                        const uint32_t height,
	                                        uint64_t       rank,
	                                        const bool     z_order) {
	if (width == 0U || height == 0U || rank >= static_cast<uint64_t>(width) * height) {
		fail("coordinate rank is outside its slot rectangle");
	}
	if (!z_order) {
		return {static_cast<uint32_t>(rank % width), static_cast<uint32_t>(rank / width), 0U};
	}
	uint64_t size = 1U;
	while (size < std::max<uint64_t>(width, height)) {
		size <<= 1U;
	}
	uint64_t origin_x = 0U;
	uint64_t origin_y = 0U;
	while (size > 1U) {
		const auto half = size >> 1U;
		bool found = false;
		for (uint32_t quadrant = 0U; quadrant < 4U; ++quadrant) {
			const auto child_x = origin_x + ((quadrant & 1U) != 0U ? half : 0U);
			const auto child_y = origin_y + ((quadrant & 2U) != 0U ? half : 0U);
			const auto count = rectangle_intersection_count(width, height, child_x, child_y, half);
			if (rank < count) {
				origin_x = child_x;
				origin_y = child_y;
				found = true;
				break;
			}
			rank -= count;
		}
		if (!found) {
			fail("failed to invert Morton coordinate rank");
		}
		size = half;
	}
	if (origin_x >= width || origin_y >= height) {
		fail("inverted Morton coordinate is outside its slot rectangle");
	}
	return {static_cast<uint32_t>(origin_x), static_cast<uint32_t>(origin_y), 0U};
}

struct CellBuild {
	uint16_t present_count = 0U;
	JpegDctBlockMajorPresenceEncoding encoding = JpegDctBlockMajorPresenceEncoding::kEmpty;
	uint32_t payload_offset = 0U;
};

struct TopologyBuild {
	uint32_t positive_before = 0U;
	uint64_t row_start       = 0U;
	uint32_t rowgroup        = 0U;
	uint32_t row_in_rowgroup = 0U;
};

struct SlotBuild {
	uint32_t semantic_slot_id = 0U;
	uint32_t max_width        = 0U;
	uint32_t max_height       = 0U;
	std::vector<uint16_t> widths;
	std::vector<uint16_t> heights;
	uint32_t first_threshold_width  = 0U;
	uint32_t first_threshold_height = 0U;
	uint32_t first_cell             = 0U;
	uint32_t coordinate_count       = 0U;
	uint32_t first_group            = 0U;
	uint32_t group_count            = 0U;
	uint32_t first_topology         = 0U;
	uint32_t topology_count         = 0U;
	bool     z_order                = false;
};

struct Shape {
	uint16_t width  = 0U;
	uint16_t height = 0U;
};

const JpegComponentMetadata* find_component(const JpegImageMetadata& image, const uint32_t semantic_slot_id) {
	const auto found = std::find_if(image.components.begin(), image.components.end(), [&](const auto& component) {
		return component.semantic_slot_id == semantic_slot_id;
	});
	return found == image.components.end() ? nullptr : &*found;
}

uint32_t threshold_cell_index(const SlotBuild& slot, const uint32_t x, const uint32_t y) {
	if (x >= slot.max_width || y >= slot.max_height) {
		fail("block coordinate is outside its semantic slot");
	}
	const auto width = std::lower_bound(slot.widths.begin(), slot.widths.end(), static_cast<uint32_t>(x + 1U));
	const auto height = std::lower_bound(slot.heights.begin(), slot.heights.end(), static_cast<uint32_t>(y + 1U));
	if (width == slot.widths.end() || height == slot.heights.end()) {
		fail("block coordinate cannot be mapped to a threshold cell");
	}
	return slot.first_cell + static_cast<uint32_t>(width - slot.widths.begin()) *
	                             static_cast<uint32_t>(slot.heights.size()) +
	       static_cast<uint32_t>(height - slot.heights.begin());
}

std::vector<uint8_t> encode_delta_ids(const std::vector<uint32_t>& ids) {
	std::vector<uint8_t> output;
	output.reserve(ids.size());
	uint32_t previous = 0U;
	for (size_t index = 0U; index < ids.size(); ++index) {
		const auto id = ids[index];
		if (index != 0U && id <= previous) {
			fail("internal presence list is not strictly increasing");
		}
		append_uleb128(output, index == 0U ? id : id - previous);
		previous = id;
	}
	return output;
}

std::vector<uint8_t> encode_bitmap(const std::vector<uint8_t>& bits,
	                               const uint32_t image_count,
	                               const uint16_t checkpoint_images) {
	std::vector<uint8_t> output = bits;
	uint32_t rank = 0U;
	for (uint32_t begin = 0U; begin < image_count; begin += checkpoint_images) {
		if (rank > std::numeric_limits<uint16_t>::max()) {
			fail("bitmap rank checkpoint exceeds uint16_t");
		}
		append_le<uint16_t>(output, static_cast<uint16_t>(rank));
		const auto end = std::min<uint32_t>(image_count, begin + checkpoint_images);
		for (uint32_t image = begin; image < end; ++image) {
			rank += (bits[image / 8U] >> (image % 8U)) & 1U;
		}
	}
	return output;
}

std::filesystem::path sidecar_name(const std::filesystem::path& directory, const uint32_t shard_id) {
	std::array<char, 64> name {};
	const auto count = std::snprintf(name.data(), name.size(), "shard_%06u.block_major_access.bin", shard_id);
	if (count < 0 || static_cast<size_t>(count) >= name.size()) {
		fail("failed to format block-major sidecar name");
	}
	return directory / name.data();
}

} // namespace

struct JpegDctBlockMajorAccessDescriptor::Impl {
	int      fd      = -1;
	uint8_t* mapping = nullptr;
	size_t   size    = 0U;

	uint64_t checksum                 = 0U;
	uint64_t manifest_crc             = 0U;
	uint64_t manifest_size            = 0U;
	uint64_t metadata_crc             = 0U;
	uint64_t metadata_size            = 0U;
	uint64_t fls_payload_crc           = 0U;
	uint64_t fls_size                 = 0U;
	uint64_t first_global_image_index = 0U;
	uint32_t shard_id                 = 0U;
	uint32_t image_count              = 0U;
	uint32_t rowgroup_vectors         = 0U;
	uint32_t vector_size              = 0U;
	uint32_t rowgroup_count           = 0U;
	uint32_t slot_count               = 0U;
	uint32_t component_count          = 0U;
	uint32_t quant_count              = 0U;
	uint32_t group_count              = 0U;
	uint32_t coordinate_count         = 0U;
	uint32_t cell_count               = 0U;
	uint32_t topology_count           = 0U;
	uint16_t rank_checkpoint_images   = 0U;
	uint16_t topology_stride          = 0U;

	uint64_t slots_offset      = 0U;
	uint64_t thresholds_offset = 0U;
	uint64_t cells_offset      = 0U;
	uint64_t payload_offset    = 0U;
	uint64_t topology_offset   = 0U;
	uint64_t images_offset     = 0U;
	uint64_t components_offset = 0U;
	uint64_t quants_offset     = 0U;

	~Impl() {
		if (mapping != nullptr) {
			::munmap(mapping, size);
		}
		if (fd >= 0) {
			::close(fd);
		}
	}

	struct SlotView {
		uint32_t semantic_slot_id = 0U;
		uint32_t max_width        = 0U;
		uint32_t max_height       = 0U;
		uint16_t width_count      = 0U;
		uint16_t height_count     = 0U;
		uint32_t first_width      = 0U;
		uint32_t first_height     = 0U;
		uint32_t first_cell       = 0U;
		uint32_t cell_count       = 0U;
		uint32_t coordinate_count = 0U;
		uint32_t first_group      = 0U;
		uint32_t group_count      = 0U;
		uint32_t first_topology   = 0U;
		uint32_t topology_count   = 0U;
		bool     z_order          = false;
	};

	struct CellView {
		uint16_t present_count = 0U;
		JpegDctBlockMajorPresenceEncoding encoding = JpegDctBlockMajorPresenceEncoding::kEmpty;
		uint32_t payload_relative_offset = 0U;
	};

	[[nodiscard]] SlotView slot(const uint32_t index) const {
		if (index >= slot_count) {
			fail("semantic slot index is outside the descriptor");
		}
		const auto offset = slots_offset + static_cast<uint64_t>(index) * kSlotRecordBytes;
		SlotView result;
		result.semantic_slot_id = read_le<uint32_t>(mapping, size, offset, "slot semantic id");
		result.max_width        = read_le<uint32_t>(mapping, size, offset + 4U, "slot max width");
		result.max_height       = read_le<uint32_t>(mapping, size, offset + 8U, "slot max height");
		result.width_count      = read_le<uint16_t>(mapping, size, offset + 12U, "slot width count");
		result.height_count     = read_le<uint16_t>(mapping, size, offset + 14U, "slot height count");
		result.first_width      = read_le<uint32_t>(mapping, size, offset + 16U, "slot first width");
		result.first_height     = read_le<uint32_t>(mapping, size, offset + 20U, "slot first height");
		result.first_cell       = read_le<uint32_t>(mapping, size, offset + 24U, "slot first cell");
		result.cell_count       = read_le<uint32_t>(mapping, size, offset + 28U, "slot cell count");
		result.coordinate_count = read_le<uint32_t>(mapping, size, offset + 32U, "slot coordinate count");
		result.first_group      = read_le<uint32_t>(mapping, size, offset + 36U, "slot first group");
		result.group_count      = read_le<uint32_t>(mapping, size, offset + 40U, "slot group count");
		result.first_topology   = read_le<uint32_t>(mapping, size, offset + 44U, "slot first topology checkpoint");
		result.topology_count   = read_le<uint32_t>(mapping, size, offset + 48U, "slot topology checkpoint count");
		result.z_order          = (read_le<uint32_t>(mapping, size, offset + 52U, "slot flags") & 1U) != 0U;
		return result;
	}

	[[nodiscard]] std::optional<SlotView> find_slot(const uint32_t semantic_slot_id) const {
		for (uint32_t index = 0U; index < slot_count; ++index) {
			auto value = slot(index);
			if (value.semantic_slot_id == semantic_slot_id) {
				return value;
			}
		}
		return std::nullopt;
	}

	[[nodiscard]] uint16_t threshold(const uint32_t index) const {
		const auto threshold_count = (cells_offset - thresholds_offset) / sizeof(uint16_t);
		if (index >= threshold_count) {
			fail("threshold index is outside the descriptor");
		}
		return read_le<uint16_t>(mapping,
		                         size,
		                         thresholds_offset + static_cast<uint64_t>(index) * sizeof(uint16_t),
		                         "threshold value");
	}

	[[nodiscard]] uint32_t lower_bound_threshold(const uint32_t first,
	                                             const uint16_t count,
	                                             const uint32_t value) const {
		uint32_t begin = 0U;
		uint32_t end   = count;
		while (begin < end) {
			const auto middle = begin + (end - begin) / 2U;
			if (threshold(first + middle) < value) {
				begin = middle + 1U;
			} else {
				end = middle;
			}
		}
		return begin;
	}

	[[nodiscard]] uint32_t cell_id(const SlotView& slot_value, const uint32_t x, const uint32_t y) const {
		if (x >= slot_value.max_width || y >= slot_value.max_height) {
			fail("block coordinate is outside its semantic slot");
		}
		const auto width_index  = lower_bound_threshold(slot_value.first_width, slot_value.width_count, x + 1U);
		const auto height_index = lower_bound_threshold(slot_value.first_height, slot_value.height_count, y + 1U);
		if (width_index >= slot_value.width_count || height_index >= slot_value.height_count) {
			fail("block coordinate cannot be mapped to a threshold cell");
		}
		return slot_value.first_cell + width_index * slot_value.height_count + height_index;
	}

	[[nodiscard]] CellView cell(const uint32_t id) const {
		if (id >= cell_count) {
			fail("rank cell is outside the descriptor");
		}
		const auto offset = cells_offset + static_cast<uint64_t>(id) * kCellRecordBytes;
		CellView result;
		result.present_count = read_le<uint16_t>(mapping, size, offset, "cell present count");
		result.encoding = static_cast<JpegDctBlockMajorPresenceEncoding>(
		    read_le<uint8_t>(mapping, size, offset + 2U, "cell encoding"));
		result.payload_relative_offset = read_le<uint32_t>(mapping, size, offset + 4U, "cell payload offset");
		return result;
	}

	[[nodiscard]] const uint8_t* payload_data(const CellView& value) const {
		const auto payload_size = topology_offset - payload_offset;
		if (value.payload_relative_offset > payload_size) {
			fail("rank cell payload offset is outside the payload section");
		}
		return mapping + payload_offset + value.payload_relative_offset;
	}

	[[nodiscard]] size_t payload_remaining(const CellView& value) const {
		return static_cast<size_t>(topology_offset - payload_offset - value.payload_relative_offset);
	}
};

JpegDctBlockMajorAccessDescriptor::JpegDctBlockMajorAccessDescriptor(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

JpegDctBlockMajorAccessDescriptor::JpegDctBlockMajorAccessDescriptor(
    JpegDctBlockMajorAccessDescriptor&&) noexcept = default;
JpegDctBlockMajorAccessDescriptor& JpegDctBlockMajorAccessDescriptor::operator=(
    JpegDctBlockMajorAccessDescriptor&&) noexcept = default;
JpegDctBlockMajorAccessDescriptor::~JpegDctBlockMajorAccessDescriptor() = default;

JpegDctBlockMajorAccessIndex read_jpeg_dct_block_major_access_index(
	const std::filesystem::path& index_path,
	const std::filesystem::path& manifest_path,
	const JpegDctShardManifest& manifest) {
	constexpr size_t kIndexHeaderBytes = 64U;
	constexpr size_t kIndexRecordBytes = 32U;
	const auto file_bytes = checked_file_size(index_path);
	if (file_bytes > std::numeric_limits<size_t>::max()) {
		fail("companion index exceeds addressable memory");
	}
	std::vector<uint8_t> encoded(static_cast<size_t>(file_bytes));
	std::ifstream input(index_path, std::ios::binary);
	if (!input || (!encoded.empty() &&
	               !input.read(reinterpret_cast<char*>(encoded.data()),
	                           static_cast<std::streamsize>(encoded.size())))) {
		fail("cannot read companion index: " + index_path.string());
	}
	if (encoded.size() < kIndexHeaderBytes ||
	    !std::equal(kIndexMagic.begin(), kIndexMagic.end(), encoded.begin())) {
		fail("companion index has invalid magic or a truncated header");
	}
	if (read_le<uint16_t>(encoded.data(), encoded.size(), 8U, "companion version") !=
	        kJpegDctBlockMajorAccessVersion ||
	    read_le<uint16_t>(encoded.data(), encoded.size(), 10U, "companion header bytes") !=
	        kIndexHeaderBytes ||
	    read_le<uint32_t>(encoded.data(), encoded.size(), 12U, "companion flags") !=
	        (kFlagLittleEndian | kFlagCrc64Ecma)) {
		fail("companion index version, header, or flags are unsupported");
	}
	const auto shard_count = read_le<uint32_t>(encoded.data(), encoded.size(), 32U, "companion shard count");
	if (shard_count != manifest.shards.size()) {
		fail("companion index shard count does not match the manifest");
	}
	const auto expected_bytes = static_cast<uint64_t>(kIndexHeaderBytes) +
	                            static_cast<uint64_t>(shard_count) * static_cast<uint64_t>(kIndexRecordBytes);
	const auto declared_bytes = read_le<uint64_t>(encoded.data(), encoded.size(), 40U, "companion index bytes");
	if (static_cast<uint64_t>(encoded.size()) != expected_bytes || declared_bytes != expected_bytes) {
		fail("companion index length does not match its shard records");
	}
	const auto stored_checksum = read_le<uint64_t>(encoded.data(), encoded.size(), 48U, "companion checksum");
	if (stored_checksum != crc64_with_zeroed_field(encoded.data(), encoded.size(), 48U)) {
		fail("companion index checksum mismatch");
	}

	JpegDctBlockMajorAccessIndex result;
	result.manifest_crc64 = read_le<uint64_t>(encoded.data(), encoded.size(), 16U, "companion manifest CRC");
	result.manifest_bytes = read_le<uint64_t>(encoded.data(), encoded.size(), 24U, "companion manifest bytes");
	result.index_bytes = declared_bytes;
	if (result.manifest_bytes != checked_file_size(manifest_path) ||
	    result.manifest_crc64 != crc64_file(manifest_path)) {
		fail("companion index belongs to a different manifest");
	}
	result.shards.reserve(shard_count);
	for (uint32_t shard_index = 0U; shard_index < shard_count; ++shard_index) {
		const auto offset = kIndexHeaderBytes + static_cast<size_t>(shard_index) * kIndexRecordBytes;
		JpegDctBlockMajorAccessIndexRecord record;
		record.shard_id = read_le<uint32_t>(encoded.data(), encoded.size(), offset, "companion shard id");
		record.image_count =
		    read_le<uint32_t>(encoded.data(), encoded.size(), offset + 4U, "companion image count");
		record.first_global_image_index =
		    read_le<uint64_t>(encoded.data(), encoded.size(), offset + 8U, "companion first image");
		record.descriptor_bytes =
		    read_le<uint64_t>(encoded.data(), encoded.size(), offset + 16U, "companion descriptor bytes");
		record.descriptor_crc64 =
		    read_le<uint64_t>(encoded.data(), encoded.size(), offset + 24U, "companion descriptor CRC");
		const auto& entry = manifest.shards[shard_index];
		if (record.shard_id != entry.shard_id || record.image_count != entry.image_count ||
		    record.first_global_image_index != entry.first_global_image_index || record.descriptor_bytes == 0U) {
			fail("companion shard record does not match the manifest");
		}
		result.shards.push_back(record);
	}
	return result;
}

JpegDctBlockMajorAccessDescriptor JpegDctBlockMajorAccessDescriptor::Open(
    const std::filesystem::path& descriptor_path) {
	auto impl = std::make_unique<Impl>();
	impl->fd = ::open(descriptor_path.c_str(), O_RDONLY | O_CLOEXEC);
	if (impl->fd < 0) {
		fail("cannot open descriptor for mmap: " + descriptor_path.string() + ": " + std::strerror(errno));
	}
	struct stat status {};
	if (::fstat(impl->fd, &status) != 0) {
		fail("cannot stat descriptor: " + descriptor_path.string());
	}
	if (status.st_size < static_cast<off_t>(kHeaderBytes) ||
	    static_cast<uint64_t>(status.st_size) > std::numeric_limits<size_t>::max()) {
		fail("descriptor file size is invalid");
	}
	impl->size = static_cast<size_t>(status.st_size);
	void* mapping = ::mmap(nullptr, impl->size, PROT_READ, MAP_SHARED, impl->fd, 0);
	if (mapping == MAP_FAILED) {
		fail("mmap failed for descriptor: " + descriptor_path.string());
	}
	impl->mapping = static_cast<uint8_t*>(mapping);
	if (!std::equal(kMagic.begin(), kMagic.end(), impl->mapping)) {
		fail("invalid descriptor magic");
	}
	if (read_le<uint16_t>(impl->mapping, impl->size, 8U, "format version") !=
	    kJpegDctBlockMajorAccessVersion) {
		fail("unsupported descriptor version");
	}
	if (read_le<uint16_t>(impl->mapping, impl->size, 10U, "header size") != kHeaderBytes) {
		fail("unsupported descriptor header size");
	}
	const auto flags = read_le<uint32_t>(impl->mapping, impl->size, 12U, "flags");
	if ((flags & kRequiredFlags) != kRequiredFlags || (flags & ~kKnownFlags) != 0U) {
		fail("descriptor flags are incompatible");
	}
	if (read_le<uint64_t>(impl->mapping, impl->size, 16U, "descriptor bytes") != impl->size) {
		fail("descriptor byte count disagrees with the mapped file");
	}
	impl->checksum = read_le<uint64_t>(impl->mapping, impl->size, 24U, "descriptor CRC64");
	if (compute_descriptor_crc64(impl->mapping, impl->size) != impl->checksum) {
		fail("descriptor CRC64 mismatch");
	}
	impl->manifest_crc             = read_le<uint64_t>(impl->mapping, impl->size, 32U, "manifest CRC64");
	impl->manifest_size            = read_le<uint64_t>(impl->mapping, impl->size, 40U, "manifest bytes");
	impl->metadata_crc             = read_le<uint64_t>(impl->mapping, impl->size, 48U, "metadata CRC64");
	impl->metadata_size            = read_le<uint64_t>(impl->mapping, impl->size, 56U, "metadata bytes");
	impl->fls_payload_crc           = read_le<uint64_t>(impl->mapping, impl->size, 64U, "FLS payload CRC64");
	impl->fls_size                 = read_le<uint64_t>(impl->mapping, impl->size, 72U, "FLS bytes");
	impl->first_global_image_index = read_le<uint64_t>(impl->mapping, impl->size, 80U, "first global image");
	impl->shard_id                 = read_le<uint32_t>(impl->mapping, impl->size, 88U, "shard id");
	impl->image_count              = read_le<uint32_t>(impl->mapping, impl->size, 92U, "image count");
	impl->rowgroup_vectors         = read_le<uint32_t>(impl->mapping, impl->size, 96U, "rowgroup vectors");
	impl->vector_size              = read_le<uint32_t>(impl->mapping, impl->size, 100U, "vector size");
	impl->rowgroup_count           = read_le<uint32_t>(impl->mapping, impl->size, 104U, "rowgroup count");
	impl->slot_count               = read_le<uint32_t>(impl->mapping, impl->size, 108U, "slot count");
	impl->component_count          = read_le<uint32_t>(impl->mapping, impl->size, 112U, "component count");
	impl->quant_count              = read_le<uint32_t>(impl->mapping, impl->size, 116U, "quant count");
	impl->group_count              = read_le<uint32_t>(impl->mapping, impl->size, 120U, "group count");
	impl->coordinate_count         = read_le<uint32_t>(impl->mapping, impl->size, 124U, "coordinate count");
	impl->cell_count               = read_le<uint32_t>(impl->mapping, impl->size, 128U, "rank cell count");
	impl->topology_count           = read_le<uint32_t>(impl->mapping, impl->size, 132U, "topology count");
	impl->rank_checkpoint_images   = read_le<uint16_t>(impl->mapping, impl->size, 136U, "rank checkpoint stride");
	impl->topology_stride          = read_le<uint16_t>(impl->mapping, impl->size, 138U, "topology stride");
	if (impl->image_count == 0U || impl->image_count > std::numeric_limits<uint16_t>::max() ||
	    impl->rank_checkpoint_images == 0U || impl->topology_stride == 0U ||
	    impl->vector_size != fastlanes::CFG::VEC_SZ) {
		fail("descriptor geometry is unsupported");
	}
	impl->slots_offset      = read_le<uint64_t>(impl->mapping, impl->size, kOffsetSlots, "slot offset");
	impl->thresholds_offset = read_le<uint64_t>(impl->mapping, impl->size, kOffsetThresholds, "threshold offset");
	impl->cells_offset      = read_le<uint64_t>(impl->mapping, impl->size, kOffsetCells, "cell offset");
	impl->payload_offset    = read_le<uint64_t>(impl->mapping, impl->size, kOffsetPayload, "payload offset");
	impl->topology_offset   = read_le<uint64_t>(impl->mapping, impl->size, kOffsetTopology, "topology offset");
	impl->images_offset     = read_le<uint64_t>(impl->mapping, impl->size, kOffsetImages, "image offset");
	impl->components_offset = read_le<uint64_t>(impl->mapping, impl->size, kOffsetComponents, "component offset");
	impl->quants_offset     = read_le<uint64_t>(impl->mapping, impl->size, kOffsetQuants, "quant offset");
	const std::array<uint64_t, 9> offsets {impl->slots_offset,
	                                       impl->thresholds_offset,
	                                       impl->cells_offset,
	                                       impl->payload_offset,
	                                       impl->topology_offset,
	                                       impl->images_offset,
	                                       impl->components_offset,
	                                       impl->quants_offset,
	                                       impl->size};
	if (offsets.front() < kHeaderBytes || !std::is_sorted(offsets.begin(), offsets.end())) {
		fail("descriptor section offsets are not monotonic");
	}
	const auto exact_section = [&](const uint64_t begin, const uint64_t end, const uint64_t count,
	                               const uint64_t record_bytes, const char* label) {
		if (count > std::numeric_limits<uint64_t>::max() / record_bytes || end - begin != count * record_bytes) {
			fail(std::string(label) + " section size disagrees with its count");
		}
	};
	exact_section(impl->slots_offset, impl->thresholds_offset, impl->slot_count, kSlotRecordBytes, "slot");
	exact_section(impl->cells_offset, impl->payload_offset, impl->cell_count, kCellRecordBytes, "cell");
	exact_section(
	    impl->topology_offset, impl->images_offset, impl->topology_count, kTopologyRecordBytes, "topology");
	exact_section(impl->images_offset, impl->components_offset, impl->image_count, kImageRecordBytes, "image");
	exact_section(
	    impl->components_offset, impl->quants_offset, impl->component_count, kComponentRecordBytes, "component");
	exact_section(impl->quants_offset, impl->size, impl->quant_count, kQuantRecordBytes, "quant");
	for (uint32_t slot_index = 0U; slot_index < impl->slot_count; ++slot_index) {
		const auto slot = impl->slot(slot_index);
		if (slot.width_count == 0U || slot.height_count == 0U ||
		    slot.cell_count != static_cast<uint32_t>(slot.width_count) * slot.height_count ||
		    slot.first_cell > impl->cell_count || slot.cell_count > impl->cell_count - slot.first_cell ||
		    slot.first_group > impl->group_count || slot.group_count > impl->group_count - slot.first_group ||
		    slot.first_topology > impl->topology_count ||
		    slot.topology_count > impl->topology_count - slot.first_topology) {
			fail("semantic slot range is outside its descriptor section");
		}
		for (uint32_t index = 1U; index < slot.width_count; ++index) {
			if (impl->threshold(slot.first_width + index - 1U) >= impl->threshold(slot.first_width + index)) {
				fail("slot width thresholds are not strictly increasing");
			}
		}
		for (uint32_t index = 1U; index < slot.height_count; ++index) {
			if (impl->threshold(slot.first_height + index - 1U) >= impl->threshold(slot.first_height + index)) {
				fail("slot height thresholds are not strictly increasing");
			}
		}
	}
	for (uint32_t cell_id = 0U; cell_id < impl->cell_count; ++cell_id) {
		const auto cell = impl->cell(cell_id);
		if (cell.present_count > impl->image_count ||
		    static_cast<uint8_t>(cell.encoding) >
		        static_cast<uint8_t>(JpegDctBlockMajorPresenceEncoding::kBitmapRank)) {
			fail("rank cell has an invalid encoding or present count");
		}
		if ((cell.present_count == 0U) != (cell.encoding == JpegDctBlockMajorPresenceEncoding::kEmpty) ||
		    (cell.present_count == impl->image_count) !=
		        (cell.encoding == JpegDctBlockMajorPresenceEncoding::kAllPresent)) {
			fail("rank cell canonical encoding disagrees with its present count");
		}
		static_cast<void>(impl->payload_data(cell));
	}
	return JpegDctBlockMajorAccessDescriptor(std::move(impl));
}

uint32_t JpegDctBlockMajorAccessDescriptor::shard_id() const noexcept {
	return impl_->shard_id;
}

uint64_t JpegDctBlockMajorAccessDescriptor::first_global_image_index() const noexcept {
	return impl_->first_global_image_index;
}

uint32_t JpegDctBlockMajorAccessDescriptor::image_count() const noexcept {
	return impl_->image_count;
}

uint32_t JpegDctBlockMajorAccessDescriptor::group_count() const noexcept {
	return impl_->group_count;
}

uint32_t JpegDctBlockMajorAccessDescriptor::rank_cell_count() const noexcept {
	return impl_->cell_count;
}

uint32_t JpegDctBlockMajorAccessDescriptor::semantic_slot_count() const noexcept {
	return impl_->slot_count;
}

size_t JpegDctBlockMajorAccessDescriptor::descriptor_bytes() const noexcept {
	return impl_->size;
}

uint64_t JpegDctBlockMajorAccessDescriptor::descriptor_crc64() const noexcept {
	return impl_->checksum;
}

JpegDctBlockMajorAccessImageRecord JpegDctBlockMajorAccessDescriptor::image(const uint32_t local_image_index) const {
	if (local_image_index >= impl_->image_count) {
		fail("image index is outside the descriptor");
	}
	const auto offset = impl_->images_offset + static_cast<uint64_t>(local_image_index) * kImageRecordBytes;
	JpegDctBlockMajorAccessImageRecord result;
	result.image_width       = read_le<uint32_t>(impl_->mapping, impl_->size, offset, "image width");
	result.image_height      = read_le<uint32_t>(impl_->mapping, impl_->size, offset + 4U, "image height");
	result.first_component   = read_le<uint32_t>(impl_->mapping, impl_->size, offset + 8U, "image first component");
	result.component_count   = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 12U, "image component count");
	result.present_slot_mask = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 14U, "image slot mask");
	result.data_precision    = read_le<uint8_t>(impl_->mapping, impl_->size, offset + 16U, "image precision");
	result.jpeg_color_space  = read_le<int16_t>(impl_->mapping, impl_->size, offset + 18U, "image color space");
	if (result.first_component > impl_->component_count ||
	    result.component_count > impl_->component_count - result.first_component) {
		fail("image component range is outside the descriptor");
	}
	return result;
}

JpegDctBlockMajorAccessComponentRecord
JpegDctBlockMajorAccessDescriptor::component(const uint32_t component_index) const {
	if (component_index >= impl_->component_count) {
		fail("component index is outside the descriptor");
	}
	const auto offset = impl_->components_offset + static_cast<uint64_t>(component_index) * kComponentRecordBytes;
	JpegDctBlockMajorAccessComponentRecord result;
	result.semantic_slot_id        = read_le<uint16_t>(impl_->mapping, impl_->size, offset, "component slot");
	result.local_component_index   = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 2U, "local component");
	result.width_in_blocks         = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 4U, "component width");
	result.height_in_blocks        = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 6U, "component height");
	result.padded_width_in_blocks  = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 8U, "padded width");
	result.padded_height_in_blocks = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 10U, "padded height");
	result.h_samp_factor           = read_le<uint8_t>(impl_->mapping, impl_->size, offset + 12U, "h sampling");
	result.v_samp_factor           = read_le<uint8_t>(impl_->mapping, impl_->size, offset + 13U, "v sampling");
	result.quant_dictionary_id     = read_le<uint16_t>(impl_->mapping, impl_->size, offset + 14U, "quant id");
	result.encoding_profile_id     = read_le<uint32_t>(impl_->mapping, impl_->size, offset + 16U, "profile id");
	result.component_id            = read_le<int16_t>(impl_->mapping, impl_->size, offset + 20U, "component id");
	result.quant_table_number      = read_le<int16_t>(impl_->mapping, impl_->size, offset + 22U, "quant table number");
	if (result.quant_dictionary_id != std::numeric_limits<uint16_t>::max() &&
	    result.quant_dictionary_id >= impl_->quant_count) {
		fail("component quant dictionary id is outside the descriptor");
	}
	return result;
}

JpegDctBlockMajorAccessRankCellRecord
JpegDctBlockMajorAccessDescriptor::rank_cell(const uint32_t cell_id) const {
	const auto cell = impl_->cell(cell_id);
	JpegDctBlockMajorAccessRankCellRecord result;
	result.cell_id                = cell_id;
	result.image_count            = impl_->image_count;
	result.present_count          = cell.present_count;
	result.rank_checkpoint_images = impl_->rank_checkpoint_images;
	result.encoding               = cell.encoding;
	size_t payload_bytes = 0U;
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList ||
	    cell.encoding == JpegDctBlockMajorPresenceEncoding::kMissingList) {
		const auto listed_count = cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList
		                              ? cell.present_count
		                              : impl_->image_count - cell.present_count;
		const auto* data = impl_->payload_data(cell);
		const auto size  = impl_->payload_remaining(cell);
		for (uint32_t index = 0U; index < listed_count; ++index) {
			(void)consume_uleb128(data, size, payload_bytes, "exported rank-cell id");
		}
	} else if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kBitmapRank) {
		const auto bit_bytes = (impl_->image_count + 7U) / 8U;
		const auto checkpoint_count =
		    (impl_->image_count + impl_->rank_checkpoint_images - 1U) / impl_->rank_checkpoint_images;
		const auto required = static_cast<uint64_t>(bit_bytes) +
		                      static_cast<uint64_t>(checkpoint_count) * sizeof(uint16_t);
		if (required > impl_->payload_remaining(cell) || required > std::numeric_limits<size_t>::max()) {
			fail("exported bitmap-rank payload is truncated");
		}
		payload_bytes = static_cast<size_t>(required);
	}
	if (payload_bytes != 0U) {
		const auto* data = impl_->payload_data(cell);
		result.payload.assign(data, data + payload_bytes);
	}
	return result;
}

JpegDctBlockMajorAccessQuantTableRecord
JpegDctBlockMajorAccessDescriptor::quant_table(const uint16_t dictionary_id) const {
	if (dictionary_id >= impl_->quant_count) {
		fail("quant dictionary id is outside the descriptor");
	}
	const auto offset = impl_->quants_offset + static_cast<uint64_t>(dictionary_id) * kQuantRecordBytes;
	JpegDctBlockMajorAccessQuantTableRecord result;
	result.dictionary_id = dictionary_id;
	result.fingerprint = read_le<uint64_t>(impl_->mapping, impl_->size, offset, "quant fingerprint");
	for (size_t coefficient = 0U; coefficient < result.values.size(); ++coefficient) {
		result.values[coefficient] = read_le<uint16_t>(impl_->mapping,
		                                                   impl_->size,
		                                                   offset + sizeof(uint64_t) +
		                                                       coefficient * sizeof(uint16_t),
		                                                   "quant coefficient");
	}
	return result;
}

JpegDctBlockMajorAccessRank JpegDctBlockMajorAccessDescriptor::Rank(const uint32_t semantic_slot_id,
	                                                                const uint32_t block_x,
	                                                                const uint32_t block_y,
	                                                                const uint32_t local_image_index) const {
	if (local_image_index >= impl_->image_count) {
		fail("rank image index is outside the descriptor");
	}
	const auto slot = impl_->find_slot(semantic_slot_id);
	if (!slot || block_x >= slot->max_width || block_y >= slot->max_height) {
		return {};
	}
	JpegDctBlockMajorAccessRank result;
	result.cell_id = impl_->cell_id(*slot, block_x, block_y);
	const auto cell = impl_->cell(result.cell_id);
	switch (cell.encoding) {
	case JpegDctBlockMajorPresenceEncoding::kEmpty:
		return result;
	case JpegDctBlockMajorPresenceEncoding::kAllPresent:
		result.rank    = local_image_index;
		result.present = true;
		return result;
	case JpegDctBlockMajorPresenceEncoding::kSparseList:
	case JpegDctBlockMajorPresenceEncoding::kMissingList: {
		const auto listed_count = cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList
		                              ? cell.present_count
		                              : impl_->image_count - cell.present_count;
		const auto* data = impl_->payload_data(cell);
		const auto size  = impl_->payload_remaining(cell);
		size_t cursor = 0U;
		uint32_t value = 0U;
		uint32_t listed_before = 0U;
		bool listed = false;
		for (uint32_t index = 0U; index < listed_count; ++index) {
			const auto delta = consume_uleb128(data, size, cursor, "presence id");
			if (index == 0U) {
				value = delta;
			} else {
				if (delta == 0U || value > std::numeric_limits<uint32_t>::max() - delta) {
					fail("presence list is not strictly increasing");
				}
				value += delta;
			}
			if (value >= impl_->image_count) {
				fail("presence list image id is outside the shard");
			}
			if (value < local_image_index) {
				++listed_before;
			} else {
				listed = value == local_image_index;
				break;
			}
		}
		if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList) {
			result.rank    = listed_before;
			result.present = listed;
		} else {
			result.rank    = local_image_index - listed_before;
			result.present = !listed;
		}
		return result;
	}
	case JpegDctBlockMajorPresenceEncoding::kBitmapRank: {
		const auto bit_bytes = (impl_->image_count + 7U) / 8U;
		const auto checkpoint = local_image_index / impl_->rank_checkpoint_images;
		const auto* data = impl_->payload_data(cell);
		const auto required = static_cast<uint64_t>(bit_bytes) +
		                      static_cast<uint64_t>((impl_->image_count + impl_->rank_checkpoint_images - 1U) /
		                                            impl_->rank_checkpoint_images) *
		                          sizeof(uint16_t);
		if (required > impl_->payload_remaining(cell)) {
			fail("bitmap-rank payload is truncated");
		}
		result.rank = read_le<uint16_t>(data,
		                                static_cast<size_t>(required),
		                                bit_bytes + checkpoint * sizeof(uint16_t),
		                                "bitmap rank checkpoint");
		const auto begin = checkpoint * impl_->rank_checkpoint_images;
		for (uint32_t image = begin; image < local_image_index; ++image) {
			result.rank += (data[image / 8U] >> (image % 8U)) & 1U;
		}
		result.present = ((data[local_image_index / 8U] >> (local_image_index % 8U)) & 1U) != 0U;
		return result;
	}
	}
	fail("unknown presence encoding");
}

uint32_t JpegDctBlockMajorAccessDescriptor::RankBefore(const uint32_t semantic_slot_id,
	                                                    const uint32_t block_x,
	                                                    const uint32_t block_y,
	                                                    const uint32_t local_image_exclusive) const {
	if (local_image_exclusive > impl_->image_count) {
		fail("rank-exclusive image index is outside the descriptor");
	}
	const auto slot = impl_->find_slot(semantic_slot_id);
	if (!slot || block_x >= slot->max_width || block_y >= slot->max_height) {
		return 0U;
	}
	const auto cell_id = impl_->cell_id(*slot, block_x, block_y);
	return RankCellInterval(cell_id, 0U, local_image_exclusive).rank_end;
}

JpegDctBlockMajorAccessRankInterval JpegDctBlockMajorAccessDescriptor::RankCellInterval(
	const uint32_t cell_id, const uint32_t local_image_begin, const uint32_t local_image_end) const {
	if (local_image_begin > local_image_end || local_image_end > impl_->image_count) {
		fail("rank-cell interval is outside the descriptor");
	}
	const auto cell = impl_->cell(cell_id);
	JpegDctBlockMajorAccessRankInterval result;
	switch (cell.encoding) {
	case JpegDctBlockMajorPresenceEncoding::kEmpty:
		return result;
	case JpegDctBlockMajorPresenceEncoding::kAllPresent:
		return {local_image_begin, local_image_end};
	case JpegDctBlockMajorPresenceEncoding::kSparseList:
	case JpegDctBlockMajorPresenceEncoding::kMissingList: {
		const auto listed_count = cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList
		                              ? cell.present_count
		                              : impl_->image_count - cell.present_count;
		const auto* data = impl_->payload_data(cell);
		const auto size  = impl_->payload_remaining(cell);
		size_t cursor = 0U;
		uint32_t value = 0U;
		uint32_t listed_before_begin = 0U;
		uint32_t listed_before_end   = 0U;
		for (uint32_t index = 0U; index < listed_count; ++index) {
			const auto delta = consume_uleb128(data, size, cursor, "rank-cell interval presence id");
			if (index == 0U) {
				value = delta;
			} else {
				if (delta == 0U || value > std::numeric_limits<uint32_t>::max() - delta) {
					fail("rank-cell interval presence list is not strictly increasing");
				}
				value += delta;
			}
			if (value >= impl_->image_count) {
				fail("rank-cell interval presence id is outside the shard");
			}
			listed_before_begin += value < local_image_begin ? 1U : 0U;
			listed_before_end += value < local_image_end ? 1U : 0U;
			if (value >= local_image_end) {
				break;
			}
		}
		if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList) {
			return {listed_before_begin, listed_before_end};
		}
		return {local_image_begin - listed_before_begin, local_image_end - listed_before_end};
	}
	case JpegDctBlockMajorPresenceEncoding::kBitmapRank: {
		const auto bit_bytes = (impl_->image_count + 7U) / 8U;
		const auto checkpoint_count =
		    (impl_->image_count + impl_->rank_checkpoint_images - 1U) / impl_->rank_checkpoint_images;
		const auto required = static_cast<uint64_t>(bit_bytes) +
		                      static_cast<uint64_t>(checkpoint_count) * sizeof(uint16_t);
		if (required > impl_->payload_remaining(cell)) {
			fail("rank-cell interval bitmap payload is truncated");
		}
		const auto* data = impl_->payload_data(cell);
		const auto rank_before = [&](const uint32_t exclusive) {
			if (exclusive == impl_->image_count) {
				return static_cast<uint32_t>(cell.present_count);
			}
			const auto checkpoint = exclusive / impl_->rank_checkpoint_images;
			uint32_t rank = read_le<uint16_t>(data,
			                                       static_cast<size_t>(required),
			                                       bit_bytes + checkpoint * sizeof(uint16_t),
			                                       "rank-cell interval bitmap checkpoint");
			uint32_t image = checkpoint * impl_->rank_checkpoint_images;
			while (image < exclusive && (image & 7U) != 0U) {
				rank += (data[image / 8U] >> (image % 8U)) & 1U;
				++image;
			}
			while (image + 8U <= exclusive) {
				rank += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(data[image / 8U])));
				image += 8U;
			}
			while (image < exclusive) {
				rank += (data[image / 8U] >> (image % 8U)) & 1U;
				++image;
			}
			return rank;
		};
		return {rank_before(local_image_begin), rank_before(local_image_end)};
	}
	}
	fail("unknown presence encoding");
}

std::optional<uint32_t> JpegDctBlockMajorAccessDescriptor::Select(const uint32_t semantic_slot_id,
	                                                               const uint32_t block_x,
	                                                               const uint32_t block_y,
	                                                               const uint32_t present_rank) const {
	const auto slot = impl_->find_slot(semantic_slot_id);
	if (!slot || block_x >= slot->max_width || block_y >= slot->max_height) {
		return std::nullopt;
	}
	return SelectCell(impl_->cell_id(*slot, block_x, block_y), present_rank);
}

std::optional<uint32_t> JpegDctBlockMajorAccessDescriptor::SelectCell(
	const uint32_t cell_id, const uint32_t present_rank) const {
	const auto cell = impl_->cell(cell_id);
	if (present_rank >= cell.present_count) {
		return std::nullopt;
	}
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kAllPresent) {
		return present_rank;
	}
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList) {
		const auto* data = impl_->payload_data(cell);
		const auto size  = impl_->payload_remaining(cell);
		size_t cursor = 0U;
		uint32_t value = 0U;
		for (uint32_t index = 0U; index <= present_rank; ++index) {
			const auto delta = consume_uleb128(data, size, cursor, "sparse select id");
			value = index == 0U ? delta : value + delta;
		}
		return value;
	}
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kMissingList) {
		const auto missing_count = impl_->image_count - cell.present_count;
		const auto* data = impl_->payload_data(cell);
		const auto size  = impl_->payload_remaining(cell);
		size_t cursor = 0U;
		uint32_t candidate = present_rank;
		uint32_t missing_value = 0U;
		for (uint32_t missing_index = 0U; missing_index < missing_count; ++missing_index) {
			const auto delta = consume_uleb128(data, size, cursor, "missing select id");
			if (missing_index == 0U) {
				missing_value = delta;
			} else {
				if (delta == 0U || missing_value > std::numeric_limits<uint32_t>::max() - delta) {
					fail("missing list is not strictly increasing");
				}
				missing_value += delta;
			}
			if (missing_value > candidate) {
				break;
			}
			++candidate;
		}
		if (candidate >= impl_->image_count) {
			fail("missing-list select did not find a present image");
		}
		return candidate;
	}
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kBitmapRank) {
		const auto* data = impl_->payload_data(cell);
		const auto bit_bytes = (impl_->image_count + 7U) / 8U;
		const auto checkpoint_count =
		    (impl_->image_count + impl_->rank_checkpoint_images - 1U) / impl_->rank_checkpoint_images;
		const auto required = static_cast<uint64_t>(bit_bytes) +
		                      static_cast<uint64_t>(checkpoint_count) * sizeof(uint16_t);
		if (required > impl_->payload_remaining(cell)) {
			fail("bitmap select payload is truncated");
		}
		uint32_t checkpoint = 0U;
		while (checkpoint + 1U < checkpoint_count &&
		       read_le<uint16_t>(data,
		                         static_cast<size_t>(required),
		                         bit_bytes + (checkpoint + 1U) * sizeof(uint16_t),
		                         "bitmap select checkpoint") <= present_rank) {
			++checkpoint;
		}
		uint32_t rank = read_le<uint16_t>(data,
		                                  static_cast<size_t>(required),
		                                  bit_bytes + checkpoint * sizeof(uint16_t),
		                                  "bitmap select checkpoint");
		uint32_t image = checkpoint * impl_->rank_checkpoint_images;
		while (image < impl_->image_count && (image & 7U) != 0U) {
			if (((data[image / 8U] >> (image % 8U)) & 1U) != 0U && rank++ == present_rank) {
				return image;
			}
			++image;
		}
		while (image + 8U <= impl_->image_count) {
			const auto byte = data[image / 8U];
			const auto count = static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(byte)));
			if (rank + count > present_rank) {
				for (uint32_t bit = 0U; bit < 8U; ++bit) {
					if (((byte >> bit) & 1U) != 0U && rank++ == present_rank) {
						return image + bit;
					}
				}
				fail("bitmap select byte rank is inconsistent");
			}
			rank += count;
			image += 8U;
		}
		while (image < impl_->image_count) {
			if (((data[image / 8U] >> (image % 8U)) & 1U) != 0U && rank++ == present_rank) {
				return image;
			}
			++image;
		}
		fail("bitmap select did not find a present image");
	}
	return std::nullopt;
}

std::optional<JpegDctBlockMajorAccessGroup>
JpegDctBlockMajorAccessDescriptor::FindGroup(const uint32_t semantic_slot_id,
	                                          const uint32_t block_x,
	                                          const uint32_t block_y) const {
	const auto slot = impl_->find_slot(semantic_slot_id);
	if (!slot || block_x >= slot->max_width || block_y >= slot->max_height) {
		return std::nullopt;
	}
	const auto position = static_cast<uint32_t>(detail::block_order_rank(slot->max_width,
	                                                                  slot->max_height,
	                                                                  block_x,
	                                                                  block_y,
	                                                                  slot->z_order
	                                                                      ? JpegDctSpatialOrder::kZOrder
	                                                                      : JpegDctSpatialOrder::kRaster));
	const auto checkpoint_relative = position / impl_->topology_stride;
	if (checkpoint_relative >= slot->topology_count) {
		fail("topology checkpoint is outside its slot");
	}
	const auto checkpoint_offset = impl_->topology_offset +
	                               static_cast<uint64_t>(slot->first_topology + checkpoint_relative) *
	                                   kTopologyRecordBytes;
	uint32_t positive_before =
	    read_le<uint32_t>(impl_->mapping, impl_->size, checkpoint_offset, "checkpoint positive rank");
	uint64_t row_start = read_le<uint64_t>(impl_->mapping, impl_->size, checkpoint_offset + 8U, "checkpoint row");
	uint32_t rowgroup =
	    read_le<uint32_t>(impl_->mapping, impl_->size, checkpoint_offset + 16U, "checkpoint rowgroup");
	uint32_t row_in_rowgroup =
	    read_le<uint32_t>(impl_->mapping, impl_->size, checkpoint_offset + 20U, "checkpoint row in rowgroup");
	const uint64_t rowgroup_capacity = static_cast<uint64_t>(impl_->rowgroup_vectors) * impl_->vector_size;
	const auto begin = checkpoint_relative * impl_->topology_stride;
	for (uint32_t current = begin; current <= position; ++current) {
		const auto coordinate = coordinate_at_rank(slot->max_width, slot->max_height, current, slot->z_order);
		const auto rank_cell_id = impl_->cell_id(*slot, coordinate.x, coordinate.y);
		const auto row_count = impl_->cell(rank_cell_id).present_count;
		if (row_count == 0U) {
			if (current == position) {
				return std::nullopt;
			}
			continue;
		}
		if (row_in_rowgroup != 0U && static_cast<uint64_t>(row_in_rowgroup) + row_count > rowgroup_capacity) {
			++rowgroup;
			row_in_rowgroup = 0U;
		}
		if (current == position) {
			return JpegDctBlockMajorAccessGroup {semantic_slot_id,
			                                          block_x,
			                                          block_y,
			                                          slot->first_group + positive_before,
			                                          rank_cell_id,
			                                          row_start,
			                                          row_count,
			                                          rowgroup,
			                                          row_in_rowgroup};
		}
		++positive_before;
		row_start += row_count;
		row_in_rowgroup += row_count;
	}
	fail("topology scan did not resolve the requested group");
}

std::vector<JpegDctBlockMajorAccessGroup> JpegDctBlockMajorAccessDescriptor::FindGroups(
	const uint32_t semantic_slot_id, const std::vector<std::array<uint32_t, 2>>& block_coordinates) const {
	std::vector<JpegDctBlockMajorAccessGroup> results(block_coordinates.size());
	const auto slot = impl_->find_slot(semantic_slot_id);
	if (!slot || block_coordinates.empty()) {
		return results;
	}
	struct Query {
		uint32_t position    = 0U;
		uint32_t input_index = 0U;
		uint32_t x           = 0U;
		uint32_t y           = 0U;
	};
	std::vector<Query> queries;
	queries.reserve(block_coordinates.size());
	for (uint32_t input_index = 0U; input_index < block_coordinates.size(); ++input_index) {
		const auto x = block_coordinates[input_index][0];
		const auto y = block_coordinates[input_index][1];
		if (x >= slot->max_width || y >= slot->max_height) {
			continue;
		}
		const auto position = detail::block_order_rank(slot->max_width,
		                                               slot->max_height,
		                                               x,
		                                               y,
		                                               slot->z_order ? JpegDctSpatialOrder::kZOrder
		                                                             : JpegDctSpatialOrder::kRaster);
		if (position > std::numeric_limits<uint32_t>::max()) {
			fail("coordinate position exceeds descriptor-v1 limits");
		}
		queries.push_back({static_cast<uint32_t>(position), input_index, x, y});
	}
	std::sort(queries.begin(), queries.end(), [](const auto& lhs, const auto& rhs) {
		return std::tie(lhs.position, lhs.input_index) < std::tie(rhs.position, rhs.input_index);
	});
	const uint64_t rowgroup_capacity = static_cast<uint64_t>(impl_->rowgroup_vectors) * impl_->vector_size;
	uint32_t current_position = 0U;
	uint32_t positive_before = 0U;
	uint64_t row_start = 0U;
	uint32_t rowgroup = 0U;
	uint32_t row_in_rowgroup = 0U;
	bool state_valid = false;
	size_t query_index = 0U;
	while (query_index < queries.size()) {
		const auto target = queries[query_index].position;
		if (!state_valid || target < current_position || target - current_position > impl_->topology_stride) {
			const auto checkpoint_relative = target / impl_->topology_stride;
			if (checkpoint_relative >= slot->topology_count) {
				fail("batch topology checkpoint is outside its slot");
			}
			const auto checkpoint_offset = impl_->topology_offset +
			                               static_cast<uint64_t>(slot->first_topology + checkpoint_relative) *
			                                   kTopologyRecordBytes;
			positive_before =
			    read_le<uint32_t>(impl_->mapping, impl_->size, checkpoint_offset, "batch checkpoint positive rank");
			row_start =
			    read_le<uint64_t>(impl_->mapping, impl_->size, checkpoint_offset + 8U, "batch checkpoint row");
			rowgroup = read_le<uint32_t>(
			    impl_->mapping, impl_->size, checkpoint_offset + 16U, "batch checkpoint rowgroup");
			row_in_rowgroup = read_le<uint32_t>(
			    impl_->mapping, impl_->size, checkpoint_offset + 20U, "batch checkpoint row in rowgroup");
			current_position = checkpoint_relative * impl_->topology_stride;
			state_valid = true;
		}
		while (current_position <= target) {
			const auto coordinate =
			    coordinate_at_rank(slot->max_width, slot->max_height, current_position, slot->z_order);
			const auto rank_cell_id = impl_->cell_id(*slot, coordinate.x, coordinate.y);
			const auto row_count = impl_->cell(rank_cell_id).present_count;
			if (row_count != 0U && row_in_rowgroup != 0U &&
			    static_cast<uint64_t>(row_in_rowgroup) + row_count > rowgroup_capacity) {
				++rowgroup;
				row_in_rowgroup = 0U;
			}
			if (current_position == target) {
				while (query_index < queries.size() && queries[query_index].position == target) {
					const auto& query = queries[query_index++];
					if (row_count != 0U) {
						results[query.input_index] = {semantic_slot_id,
						                              query.x,
						                              query.y,
						                              slot->first_group + positive_before,
						                              rank_cell_id,
						                              row_start,
						                              row_count,
						                              rowgroup,
						                              row_in_rowgroup};
					}
				}
			}
			if (row_count != 0U) {
				++positive_before;
				row_start += row_count;
				row_in_rowgroup += row_count;
			}
			++current_position;
		}
	}
	return results;
}

void JpegDctBlockMajorAccessDescriptor::ValidateSource(const std::filesystem::path& manifest_path,
	                                                    const JpegDctShardManifestEntry& manifest_entry,
	                                                    const std::filesystem::path& metadata_path,
	                                                    const std::filesystem::path& fls_path) const {
	if (manifest_entry.shard_id != impl_->shard_id ||
	    manifest_entry.first_global_image_index != impl_->first_global_image_index ||
	    manifest_entry.image_count != impl_->image_count || manifest_entry.rowgroup_count != impl_->rowgroup_count ||
	    manifest_entry.block_group_count != impl_->group_count) {
		fail("descriptor shard identity disagrees with the manifest entry");
	}
	if (checked_file_size(manifest_path) != impl_->manifest_size || crc64_file(manifest_path) != impl_->manifest_crc) {
		fail("descriptor source manifest identity mismatch");
	}
	if (checked_file_size(metadata_path) != impl_->metadata_size ||
	    crc64_file(metadata_path) != impl_->metadata_crc) {
		fail("descriptor source metadata identity mismatch");
	}
	if (checked_file_size(fls_path) != impl_->fls_size ||
	    (impl_->fls_payload_crc != 0U && manifest_entry.payload_crc64 != impl_->fls_payload_crc)) {
		fail("descriptor source FLS identity mismatch");
	}
}

JpegDctBlockMajorAccessValidationReport JpegDctBlockMajorAccessDescriptor::ValidateAgainstMetadata(
	const JpegDctDatasetMetadata& metadata, const bool exhaustive_rank_validation) const {
	if (metadata.row_ordering != JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor ||
	    metadata.images.size() != impl_->image_count || metadata.block_group_index.size() != impl_->group_count) {
		fail("descriptor is incompatible with the supplied block-major metadata");
	}
	JpegDctBlockMajorAccessValidationReport report;
	uint32_t component_cursor = 0U;
	for (uint32_t image_index = 0U; image_index < impl_->image_count; ++image_index) {
		const auto& expected = metadata.images[image_index];
		const auto actual = image(image_index);
		uint16_t expected_mask = 0U;
		uint16_t expected_components = 0U;
		for (const auto& expected_component : expected.components) {
			if (!expected_component.present) {
				continue;
			}
			if (expected_component.semantic_slot_id >= 16U) {
				fail("metadata semantic slot cannot be represented in the image presence mask");
			}
			expected_mask |= static_cast<uint16_t>(1U << expected_component.semantic_slot_id);
			++expected_components;
			const auto actual_component = component(component_cursor++);
			if (actual_component.semantic_slot_id != expected_component.semantic_slot_id ||
			    actual_component.local_component_index != expected_component.local_component_index ||
			    actual_component.width_in_blocks != expected_component.width_in_blocks ||
			    actual_component.height_in_blocks != expected_component.height_in_blocks ||
			    actual_component.h_samp_factor != expected_component.h_samp_factor ||
			    actual_component.v_samp_factor != expected_component.v_samp_factor ||
			    actual_component.encoding_profile_id != expected_component.encoding_profile_id) {
				fail("descriptor component record disagrees with metadata");
			}
			++report.components_checked;
		}
		if (actual.image_width != expected.image_width || actual.image_height != expected.image_height ||
		    actual.first_component + actual.component_count != component_cursor ||
		    actual.component_count != expected_components || actual.present_slot_mask != expected_mask ||
		    actual.data_precision != expected.data_precision) {
			fail("descriptor image record disagrees with metadata");
		}
		++report.images_checked;
	}
	if (component_cursor != impl_->component_count) {
		fail("descriptor contains unreferenced component records");
	}
	uint32_t group_cursor = 0U;
	uint64_t row_cursor = 0U;
	uint32_t rowgroup_cursor = 0U;
	uint32_t row_in_rowgroup = 0U;
	const uint64_t rowgroup_capacity = static_cast<uint64_t>(impl_->rowgroup_vectors) * impl_->vector_size;
	for (uint32_t slot_index = 0U; slot_index < impl_->slot_count; ++slot_index) {
		const auto slot = impl_->slot(slot_index);
		uint32_t slot_positive = 0U;
		for (uint32_t position = 0U; position < slot.coordinate_count; ++position) {
			if (position % impl_->topology_stride == 0U) {
				const auto checkpoint_relative = position / impl_->topology_stride;
				if (checkpoint_relative >= slot.topology_count) {
					fail("descriptor topology checkpoint count is too small");
				}
				const auto offset = impl_->topology_offset +
				                    static_cast<uint64_t>(slot.first_topology + checkpoint_relative) *
				                        kTopologyRecordBytes;
				if (read_le<uint32_t>(impl_->mapping, impl_->size, offset, "checkpoint positive rank") !=
				        slot_positive ||
				    read_le<uint64_t>(impl_->mapping, impl_->size, offset + 8U, "checkpoint row") != row_cursor ||
				    read_le<uint32_t>(impl_->mapping, impl_->size, offset + 16U, "checkpoint rowgroup") !=
				        rowgroup_cursor ||
				    read_le<uint32_t>(impl_->mapping, impl_->size, offset + 20U, "checkpoint row in rowgroup") !=
				        row_in_rowgroup) {
					fail("descriptor topology checkpoint disagrees with sequential reconstruction");
				}
			}
			const auto coordinate = coordinate_at_rank(slot.max_width, slot.max_height, position, slot.z_order);
			const auto row_count = impl_->cell(impl_->cell_id(slot, coordinate.x, coordinate.y)).present_count;
			if (row_count == 0U) {
				continue;
			}
			if (row_in_rowgroup != 0U && static_cast<uint64_t>(row_in_rowgroup) + row_count > rowgroup_capacity) {
				++rowgroup_cursor;
				row_in_rowgroup = 0U;
			}
			if (group_cursor >= metadata.block_group_index.size()) {
				fail("descriptor topology has more groups than legacy metadata");
			}
			const auto& expected = metadata.block_group_index[group_cursor];
			if (expected.semantic_slot_id != slot.semantic_slot_id ||
			    expected.block_x != coordinate.x || expected.block_y != coordinate.y ||
			    expected.row_start != row_cursor || expected.row_count != row_count ||
			    expected.fls_rowgroup_index != rowgroup_cursor ||
			    expected.row_start_in_rowgroup != row_in_rowgroup) {
				fail("descriptor topology record disagrees with legacy block-group metadata");
			}
			++group_cursor;
			++slot_positive;
			row_cursor += row_count;
			row_in_rowgroup += row_count;
			++report.groups_checked;
		}
		if (slot_positive != slot.group_count ||
		    slot.topology_count != (slot.coordinate_count + impl_->topology_stride - 1U) / impl_->topology_stride) {
			fail("descriptor semantic-slot topology totals are inconsistent");
		}
	}
	if (group_cursor != impl_->group_count) {
		fail("descriptor topology has fewer groups than legacy metadata");
	}
	if (exhaustive_rank_validation) {
		for (uint32_t slot_index = 0U; slot_index < impl_->slot_count; ++slot_index) {
			const auto slot = impl_->slot(slot_index);
			std::vector<Shape> shapes(impl_->image_count);
			for (uint32_t image_index = 0U; image_index < impl_->image_count; ++image_index) {
				const auto* expected_component = find_component(metadata.images[image_index], slot.semantic_slot_id);
				if (expected_component != nullptr && expected_component->present) {
					if (expected_component->width_in_blocks > std::numeric_limits<uint16_t>::max() ||
					    expected_component->height_in_blocks > std::numeric_limits<uint16_t>::max()) {
						fail("metadata component grid exceeds descriptor-v1 validation limits");
					}
					shapes[image_index] = {static_cast<uint16_t>(expected_component->width_in_blocks),
					                       static_cast<uint16_t>(expected_component->height_in_blocks)};
				}
			}
			std::vector<uint8_t> actual_presence(impl_->image_count, 0U);
			for (uint32_t width_index = 0U; width_index < slot.width_count; ++width_index) {
				for (uint32_t height_index = 0U; height_index < slot.height_count; ++height_index) {
					const auto x = static_cast<uint32_t>(impl_->threshold(slot.first_width + width_index) - 1U);
					const auto y = static_cast<uint32_t>(impl_->threshold(slot.first_height + height_index) - 1U);
					const auto cell = impl_->cell(slot.first_cell + width_index * slot.height_count + height_index);
					std::fill(actual_presence.begin(), actual_presence.end(), 0U);
					if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kAllPresent) {
						std::fill(actual_presence.begin(), actual_presence.end(), 1U);
					} else if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList ||
					           cell.encoding == JpegDctBlockMajorPresenceEncoding::kMissingList) {
						if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kMissingList) {
							std::fill(actual_presence.begin(), actual_presence.end(), 1U);
						}
						const auto listed_count = cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList
						                              ? cell.present_count
						                              : impl_->image_count - cell.present_count;
						const auto* data = impl_->payload_data(cell);
						const auto size = impl_->payload_remaining(cell);
						size_t cursor = 0U;
						uint32_t value = 0U;
						for (uint32_t listed_index = 0U; listed_index < listed_count; ++listed_index) {
							const auto delta = consume_uleb128(data, size, cursor, "validation presence id");
							if (listed_index == 0U) {
								value = delta;
							} else {
								if (delta == 0U || value > std::numeric_limits<uint32_t>::max() - delta) {
									fail("validation presence list is not strictly increasing");
								}
								value += delta;
							}
							if (value >= impl_->image_count) {
								fail("validation presence id is outside the shard");
							}
							actual_presence[value] =
							    cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList ? 1U : 0U;
						}
					} else if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kBitmapRank) {
						const auto* data = impl_->payload_data(cell);
						const auto bit_bytes = (impl_->image_count + 7U) / 8U;
						const auto checkpoint_count =
						    (impl_->image_count + impl_->rank_checkpoint_images - 1U) /
						    impl_->rank_checkpoint_images;
						const uint64_t required =
						    static_cast<uint64_t>(bit_bytes) + static_cast<uint64_t>(checkpoint_count) * sizeof(uint16_t);
						if (required > impl_->payload_remaining(cell)) {
							fail("validation bitmap-rank payload is truncated");
						}
						uint32_t actual_rank = 0U;
						for (uint32_t checkpoint = 0U; checkpoint < checkpoint_count; ++checkpoint) {
							if (read_le<uint16_t>(data,
							                      static_cast<size_t>(required),
							                      bit_bytes + checkpoint * sizeof(uint16_t),
							                      "validation bitmap checkpoint") != actual_rank) {
								fail("bitmap rank checkpoint disagrees with its bitmap");
							}
							const auto begin = checkpoint * impl_->rank_checkpoint_images;
							const auto end = std::min<uint32_t>(impl_->image_count, begin + impl_->rank_checkpoint_images);
							for (uint32_t image_index = begin; image_index < end; ++image_index) {
								actual_presence[image_index] =
								    static_cast<uint8_t>((data[image_index / 8U] >> (image_index % 8U)) & 1U);
								actual_rank += actual_presence[image_index];
							}
						}
					}
					uint32_t expected_rank = 0U;
					for (uint32_t image_index = 0U; image_index < impl_->image_count; ++image_index) {
						const bool expected_present = x < shapes[image_index].width && y < shapes[image_index].height;
						if ((actual_presence[image_index] != 0U) != expected_present) {
							fail("descriptor rank disagrees with legacy image-minor presence");
						}
						if (expected_present) {
							++expected_rank;
							++report.select_queries_checked;
						}
						++report.rank_queries_checked;
					}
					if (expected_rank != cell.present_count) {
						fail("descriptor rank cell count disagrees with decoded presence");
					}
					++report.rank_cells_checked;
				}
			}
		}
	}
	return report;
}

JpegDctBlockMajorAccessShardReport build_jpeg_dct_block_major_access_descriptor(
	const std::filesystem::path& manifest_path,
	const JpegDctShardManifest& manifest,
	const JpegDctShardManifestEntry& shard,
	const std::filesystem::path& output_path,
	const JpegDctBlockMajorAccessBuildOptions& options) {
	if (manifest.version != 1U || (!manifest.physical_layout.empty() &&
	                               manifest.physical_layout != "dct-major/spatial-major-image-minor")) {
		fail("builder requires a manifest-v1 spatial-major/image-minor dataset");
	}
	if (options.rank_checkpoint_images == 0U || options.topology_checkpoint_coordinates == 0U) {
		fail("builder checkpoint strides must be positive");
	}
	if (shard.image_count == 0U || shard.image_count > std::numeric_limits<uint16_t>::max()) {
		fail("builder requires 1..65535 images per shard");
	}
	const auto source_directory = manifest_path.parent_path();
	const auto metadata_path = source_directory / shard.metadata_file_name;
	const auto fls_path      = source_directory / shard.fls_file_name;
	const auto metadata = detail::read_jpeg_dct_metadata_file(metadata_path);
	if (metadata.row_ordering != JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor ||
	    metadata.images.size() != shard.image_count || metadata.block_group_index.size() != shard.block_group_count) {
		fail("source shard is not a compatible ragged block-major shard");
	}
	if (checked_file_size(metadata_path) != shard.metadata_file_size || checked_file_size(fls_path) != shard.fls_file_size) {
		fail("source shard sizes disagree with the manifest");
	}

	std::vector<uint32_t> slot_ids;
	for (const auto& image : metadata.images) {
		for (const auto& component : image.components) {
			if (component.present) {
				slot_ids.push_back(component.semantic_slot_id);
			}
		}
	}
	std::sort(slot_ids.begin(), slot_ids.end());
	slot_ids.erase(std::unique(slot_ids.begin(), slot_ids.end()), slot_ids.end());
	if (slot_ids.empty() || slot_ids.size() > 16U) {
		fail("builder requires between one and sixteen semantic slots");
	}

	std::vector<SlotBuild> slots;
	std::vector<CellBuild> cells;
	std::vector<uint8_t> presence_payload;
	std::vector<uint16_t> thresholds;
	JpegDctBlockMajorAccessShardReport report;
	report.shard_id      = shard.shard_id;
	report.image_count   = shard.image_count;
	report.group_count   = shard.block_group_count;
	report.descriptor_path = output_path;

	for (const auto semantic_slot_id : slot_ids) {
		SlotBuild slot;
		slot.semantic_slot_id = semantic_slot_id;
		slot.z_order = metadata.z_curve_block_order;
		std::vector<Shape> shapes(metadata.images.size());
		for (size_t image_index = 0U; image_index < metadata.images.size(); ++image_index) {
			const auto* component = find_component(metadata.images[image_index], semantic_slot_id);
			if (component == nullptr || !component->present) {
				continue;
			}
			if (component->width_in_blocks == 0U || component->height_in_blocks == 0U ||
			    component->width_in_blocks > std::numeric_limits<uint16_t>::max() ||
			    component->height_in_blocks > std::numeric_limits<uint16_t>::max()) {
				fail("component block grid cannot be represented by descriptor-v1");
			}
			shapes[image_index] = {static_cast<uint16_t>(component->width_in_blocks),
			                       static_cast<uint16_t>(component->height_in_blocks)};
			slot.widths.push_back(shapes[image_index].width);
			slot.heights.push_back(shapes[image_index].height);
		}
		std::sort(slot.widths.begin(), slot.widths.end());
		slot.widths.erase(std::unique(slot.widths.begin(), slot.widths.end()), slot.widths.end());
		std::sort(slot.heights.begin(), slot.heights.end());
		slot.heights.erase(std::unique(slot.heights.begin(), slot.heights.end()), slot.heights.end());
		if (slot.widths.empty() || slot.heights.empty()) {
			fail("semantic slot has no present component grids");
		}
		slot.max_width  = slot.widths.back();
		slot.max_height = slot.heights.back();
		const uint64_t coordinate_count = static_cast<uint64_t>(slot.max_width) * slot.max_height;
		const uint64_t cell_count = static_cast<uint64_t>(slot.widths.size()) * slot.heights.size();
		if (coordinate_count > std::numeric_limits<uint32_t>::max() ||
		    cell_count > std::numeric_limits<uint32_t>::max() ||
		    cells.size() > std::numeric_limits<uint32_t>::max() - cell_count) {
			fail("semantic slot geometry exceeds descriptor-v1 limits");
		}
		slot.coordinate_count = static_cast<uint32_t>(coordinate_count);
		slot.first_threshold_width = static_cast<uint32_t>(thresholds.size());
		thresholds.insert(thresholds.end(), slot.widths.begin(), slot.widths.end());
		slot.first_threshold_height = static_cast<uint32_t>(thresholds.size());
		thresholds.insert(thresholds.end(), slot.heights.begin(), slot.heights.end());
		slot.first_cell = static_cast<uint32_t>(cells.size());

		std::map<uint16_t, uint32_t> width_indices;
		std::map<uint16_t, uint32_t> height_indices;
		for (uint32_t index = 0U; index < slot.widths.size(); ++index) {
			width_indices.emplace(slot.widths[index], index);
		}
		for (uint32_t index = 0U; index < slot.heights.size(); ++index) {
			height_indices.emplace(slot.heights[index], index);
		}
		std::vector<uint32_t> suffix(static_cast<size_t>(cell_count), 0U);
		for (const auto shape : shapes) {
			if (shape.width != 0U) {
				++suffix[width_indices.at(shape.width) * slot.heights.size() + height_indices.at(shape.height)];
			}
		}
		for (size_t wi = slot.widths.size(); wi-- > 0U;) {
			for (size_t hi = slot.heights.size(); hi-- > 0U;) {
				auto& value = suffix[wi * slot.heights.size() + hi];
				if (wi + 1U < slot.widths.size()) {
					value += suffix[(wi + 1U) * slot.heights.size() + hi];
				}
				if (hi + 1U < slot.heights.size()) {
					value += suffix[wi * slot.heights.size() + hi + 1U];
				}
				if (wi + 1U < slot.widths.size() && hi + 1U < slot.heights.size()) {
					value -= suffix[(wi + 1U) * slot.heights.size() + hi + 1U];
				}
			}
		}
		for (uint32_t wi = 0U; wi < slot.widths.size(); ++wi) {
			for (uint32_t hi = 0U; hi < slot.heights.size(); ++hi) {
				const auto present_count = suffix[wi * slot.heights.size() + hi];
				CellBuild cell;
				cell.present_count = static_cast<uint16_t>(present_count);
				cell.payload_offset = static_cast<uint32_t>(presence_payload.size());
				if (present_count == 0U) {
					cell.encoding = JpegDctBlockMajorPresenceEncoding::kEmpty;
					++report.empty_cells;
					cells.push_back(cell);
					continue;
				}
				if (present_count == shard.image_count) {
					cell.encoding = JpegDctBlockMajorPresenceEncoding::kAllPresent;
					++report.all_present_cells;
					cells.push_back(cell);
					continue;
				}
				std::vector<uint32_t> present_ids;
				std::vector<uint32_t> missing_ids;
				present_ids.reserve(present_count);
				missing_ids.reserve(shard.image_count - present_count);
				std::vector<uint8_t> bits((shard.image_count + 7U) / 8U, 0U);
				for (uint32_t image_index = 0U; image_index < shard.image_count; ++image_index) {
					const bool present = shapes[image_index].width >= slot.widths[wi] &&
					                     shapes[image_index].height >= slot.heights[hi];
					if (present) {
						present_ids.push_back(image_index);
						bits[image_index / 8U] |= static_cast<uint8_t>(1U << (image_index % 8U));
					} else {
						missing_ids.push_back(image_index);
					}
				}
				if (present_ids.size() != present_count) {
					fail("internal threshold-cell suffix count mismatch");
				}
				auto sparse_payload = encode_delta_ids(present_ids);
				auto missing_payload = encode_delta_ids(missing_ids);
				auto bitmap_payload = encode_bitmap(bits, shard.image_count, options.rank_checkpoint_images);
				const std::vector<uint8_t>* selected = &sparse_payload;
				cell.encoding = JpegDctBlockMajorPresenceEncoding::kSparseList;
				if (missing_payload.size() < selected->size()) {
					selected = &missing_payload;
					cell.encoding = JpegDctBlockMajorPresenceEncoding::kMissingList;
				}
				if (bitmap_payload.size() < selected->size()) {
					selected = &bitmap_payload;
					cell.encoding = JpegDctBlockMajorPresenceEncoding::kBitmapRank;
				}
				if (presence_payload.size() > std::numeric_limits<uint32_t>::max() - selected->size()) {
					fail("presence payload exceeds descriptor-v1 limits");
				}
				presence_payload.insert(presence_payload.end(), selected->begin(), selected->end());
				switch (cell.encoding) {
				case JpegDctBlockMajorPresenceEncoding::kSparseList: ++report.sparse_cells; break;
				case JpegDctBlockMajorPresenceEncoding::kMissingList: ++report.missing_cells; break;
				case JpegDctBlockMajorPresenceEncoding::kBitmapRank: ++report.bitmap_cells; break;
				default: break;
				}
				cells.push_back(cell);
			}
		}
		slots.push_back(std::move(slot));
	}
	report.rank_cell_count        = static_cast<uint32_t>(cells.size());
	report.presence_payload_bytes = presence_payload.size();

	std::vector<TopologyBuild> topology;
	uint32_t global_group_cursor = 0U;
	uint64_t global_row_cursor   = 0U;
	uint32_t rowgroup_cursor     = 0U;
	uint32_t row_in_rowgroup     = 0U;
	const uint64_t rowgroup_capacity = static_cast<uint64_t>(manifest.rowgroup_vectors) * fastlanes::CFG::VEC_SZ;
	for (auto& slot : slots) {
		slot.first_group    = global_group_cursor;
		slot.first_topology = static_cast<uint32_t>(topology.size());
		for (uint32_t position = 0U; position < slot.coordinate_count; ++position) {
			if (position % options.topology_checkpoint_coordinates == 0U) {
				topology.push_back({global_group_cursor - slot.first_group,
				                    global_row_cursor,
				                    rowgroup_cursor,
				                    row_in_rowgroup});
			}
			const auto coordinate = coordinate_at_rank(slot.max_width, slot.max_height, position, slot.z_order);
			const auto cell_id = threshold_cell_index(slot, coordinate.x, coordinate.y);
			const auto row_count = cells.at(cell_id).present_count;
			if (row_count == 0U) {
				continue;
			}
			if (row_in_rowgroup != 0U && static_cast<uint64_t>(row_in_rowgroup) + row_count > rowgroup_capacity) {
				++rowgroup_cursor;
				row_in_rowgroup = 0U;
			}
			if (global_group_cursor >= metadata.block_group_index.size()) {
				fail("reconstructed topology contains more groups than legacy metadata");
			}
			const auto& expected = metadata.block_group_index[global_group_cursor];
			if (expected.semantic_slot_id != slot.semantic_slot_id ||
			    expected.block_x != coordinate.x || expected.block_y != coordinate.y ||
			    expected.row_start != global_row_cursor || expected.row_count != row_count ||
			    expected.fls_rowgroup_index != rowgroup_cursor || expected.row_start_in_rowgroup != row_in_rowgroup) {
				fail("reconstructed topology disagrees with legacy block-group metadata");
			}
			++global_group_cursor;
			global_row_cursor += row_count;
			row_in_rowgroup += row_count;
		}
		slot.group_count    = global_group_cursor - slot.first_group;
		slot.topology_count = static_cast<uint32_t>(topology.size()) - slot.first_topology;
	}
	if (global_group_cursor != shard.block_group_count || global_row_cursor != shard.real_row_count ||
	    (shard.rowgroup_count != 0U && rowgroup_cursor + 1U != shard.rowgroup_count)) {
		fail("reconstructed topology totals disagree with the manifest");
	}

	std::map<std::array<uint16_t, 64>, uint16_t> quant_ids;
	std::vector<JpegEncodingProfileMetadata> unique_quants;
	std::map<uint32_t, uint16_t> profile_quant_ids;
	for (const auto& profile : metadata.encoding_profiles) {
		auto found = quant_ids.find(profile.quant_table_values);
		if (found == quant_ids.end()) {
			if (quant_ids.size() >= std::numeric_limits<uint16_t>::max()) {
				fail("quant dictionary exceeds descriptor-v1 limits");
			}
			const auto id = static_cast<uint16_t>(quant_ids.size());
			quant_ids.emplace(profile.quant_table_values, id);
			unique_quants.push_back(profile);
			profile_quant_ids.emplace(profile.profile_id, id);
		} else {
			profile_quant_ids.emplace(profile.profile_id, found->second);
		}
	}
	uint32_t component_count = 0U;
	for (const auto& image_metadata : metadata.images) {
		for (const auto& component_metadata : image_metadata.components) {
			component_count += component_metadata.present ? 1U : 0U;
		}
	}

	std::vector<uint8_t> descriptor(kHeaderBytes, 0U);
	align_to(descriptor, 8U);
	put_le<uint64_t>(descriptor, kOffsetSlots, descriptor.size());
	for (const auto& slot : slots) {
		const auto begin = descriptor.size();
		append_le<uint32_t>(descriptor, slot.semantic_slot_id);
		append_le<uint32_t>(descriptor, slot.max_width);
		append_le<uint32_t>(descriptor, slot.max_height);
		append_le<uint16_t>(descriptor, static_cast<uint16_t>(slot.widths.size()));
		append_le<uint16_t>(descriptor, static_cast<uint16_t>(slot.heights.size()));
		append_le<uint32_t>(descriptor, slot.first_threshold_width);
		append_le<uint32_t>(descriptor, slot.first_threshold_height);
		append_le<uint32_t>(descriptor, slot.first_cell);
		append_le<uint32_t>(descriptor, static_cast<uint32_t>(slot.widths.size() * slot.heights.size()));
		append_le<uint32_t>(descriptor, slot.coordinate_count);
		append_le<uint32_t>(descriptor, slot.first_group);
		append_le<uint32_t>(descriptor, slot.group_count);
		append_le<uint32_t>(descriptor, slot.first_topology);
		append_le<uint32_t>(descriptor, slot.topology_count);
		append_le<uint32_t>(descriptor, slot.z_order ? 1U : 0U);
		descriptor.resize(begin + kSlotRecordBytes, 0U);
	}
	put_le<uint64_t>(descriptor, kOffsetThresholds, descriptor.size());
	for (const auto value : thresholds) {
		append_le<uint16_t>(descriptor, value);
	}
	align_to(descriptor, 8U);
	put_le<uint64_t>(descriptor, kOffsetCells, descriptor.size());
	for (const auto& cell : cells) {
		append_le<uint16_t>(descriptor, cell.present_count);
		append_le<uint8_t>(descriptor, static_cast<uint8_t>(cell.encoding));
		append_le<uint8_t>(descriptor, 0U);
		append_le<uint32_t>(descriptor, cell.payload_offset);
	}
	put_le<uint64_t>(descriptor, kOffsetPayload, descriptor.size());
	descriptor.insert(descriptor.end(), presence_payload.begin(), presence_payload.end());
	align_to(descriptor, 8U);
	put_le<uint64_t>(descriptor, kOffsetTopology, descriptor.size());
	for (const auto& checkpoint : topology) {
		append_le<uint32_t>(descriptor, checkpoint.positive_before);
		append_le<uint32_t>(descriptor, 0U);
		append_le<uint64_t>(descriptor, checkpoint.row_start);
		append_le<uint32_t>(descriptor, checkpoint.rowgroup);
		append_le<uint32_t>(descriptor, checkpoint.row_in_rowgroup);
	}
	put_le<uint64_t>(descriptor, kOffsetImages, descriptor.size());
	uint32_t first_component = 0U;
	for (const auto& image_metadata : metadata.images) {
		uint16_t present_mask = 0U;
		uint16_t present_components = 0U;
		for (const auto& component_metadata : image_metadata.components) {
			if (!component_metadata.present) {
				continue;
			}
			if (component_metadata.semantic_slot_id >= 16U) {
				fail("semantic slot cannot be represented in descriptor-v1 image mask");
			}
			present_mask |= static_cast<uint16_t>(1U << component_metadata.semantic_slot_id);
			++present_components;
		}
		append_le<uint32_t>(descriptor, image_metadata.image_width);
		append_le<uint32_t>(descriptor, image_metadata.image_height);
		append_le<uint32_t>(descriptor, first_component);
		append_le<uint16_t>(descriptor, present_components);
		append_le<uint16_t>(descriptor, present_mask);
		append_le<uint8_t>(descriptor, image_metadata.data_precision);
		append_le<uint8_t>(descriptor, 0U);
		if (image_metadata.jpeg_color_space < std::numeric_limits<int16_t>::min() ||
		    image_metadata.jpeg_color_space > std::numeric_limits<int16_t>::max()) {
			fail("JPEG color space cannot be represented in descriptor-v1");
		}
		append_le<int16_t>(descriptor, static_cast<int16_t>(image_metadata.jpeg_color_space));
		append_le<uint32_t>(descriptor, 0U);
		first_component += present_components;
	}
	put_le<uint64_t>(descriptor, kOffsetComponents, descriptor.size());
	for (const auto& image_metadata : metadata.images) {
		for (const auto& value : image_metadata.components) {
			if (!value.present) {
				continue;
			}
			const auto checked_u16 = [](const uint64_t number, const char* label) {
				if (number > std::numeric_limits<uint16_t>::max()) {
					fail(std::string(label) + " cannot be represented in descriptor-v1");
				}
				return static_cast<uint16_t>(number);
			};
			append_le<uint16_t>(descriptor, checked_u16(value.semantic_slot_id, "semantic slot"));
			append_le<uint16_t>(descriptor, checked_u16(value.local_component_index, "local component"));
			append_le<uint16_t>(descriptor, checked_u16(value.width_in_blocks, "component width"));
			append_le<uint16_t>(descriptor, checked_u16(value.height_in_blocks, "component height"));
			append_le<uint16_t>(descriptor, checked_u16(value.padded_width_in_blocks, "padded width"));
			append_le<uint16_t>(descriptor, checked_u16(value.padded_height_in_blocks, "padded height"));
			append_le<uint8_t>(descriptor, static_cast<uint8_t>(checked_u16(value.h_samp_factor, "h sampling")));
			append_le<uint8_t>(descriptor, static_cast<uint8_t>(checked_u16(value.v_samp_factor, "v sampling")));
			const auto quant = profile_quant_ids.find(value.encoding_profile_id);
			append_le<uint16_t>(descriptor,
			                    quant == profile_quant_ids.end() ? std::numeric_limits<uint16_t>::max() : quant->second);
			append_le<uint32_t>(descriptor, value.encoding_profile_id);
			if (value.component_id < std::numeric_limits<int16_t>::min() ||
			    value.component_id > std::numeric_limits<int16_t>::max() ||
			    value.quant_tbl_no < std::numeric_limits<int16_t>::min() ||
			    value.quant_tbl_no > std::numeric_limits<int16_t>::max()) {
				fail("component metadata cannot be represented in descriptor-v1");
			}
			append_le<int16_t>(descriptor, static_cast<int16_t>(value.component_id));
			append_le<int16_t>(descriptor, static_cast<int16_t>(value.quant_tbl_no));
		}
	}
	put_le<uint64_t>(descriptor, kOffsetQuants, descriptor.size());
	for (const auto& quant : unique_quants) {
		append_le<uint64_t>(descriptor, quant.quant_table_fingerprint);
		for (const auto value : quant.quant_table_values) {
			append_le<uint16_t>(descriptor, value);
		}
	}

	if (descriptor.size() > std::numeric_limits<uint64_t>::max()) {
		fail("descriptor exceeds format size limits");
	}
	std::copy(kMagic.begin(), kMagic.end(), descriptor.begin());
	put_le<uint16_t>(descriptor, 8U, kJpegDctBlockMajorAccessVersion);
	put_le<uint16_t>(descriptor, 10U, kHeaderBytes);
	put_le<uint32_t>(descriptor, 12U, kKnownFlags);
	put_le<uint64_t>(descriptor, 16U, descriptor.size());
	put_le<uint64_t>(descriptor, 32U, crc64_file(manifest_path));
	put_le<uint64_t>(descriptor, 40U, checked_file_size(manifest_path));
	put_le<uint64_t>(descriptor, 48U, crc64_file(metadata_path));
	put_le<uint64_t>(descriptor, 56U, checked_file_size(metadata_path));
	put_le<uint64_t>(descriptor, 64U, shard.payload_crc64);
	put_le<uint64_t>(descriptor, 72U, checked_file_size(fls_path));
	put_le<uint64_t>(descriptor, 80U, shard.first_global_image_index);
	put_le<uint32_t>(descriptor, 88U, shard.shard_id);
	put_le<uint32_t>(descriptor, 92U, shard.image_count);
	put_le<uint32_t>(descriptor, 96U, manifest.rowgroup_vectors);
	put_le<uint32_t>(descriptor, 100U, fastlanes::CFG::VEC_SZ);
	put_le<uint32_t>(descriptor, 104U, shard.rowgroup_count);
	put_le<uint32_t>(descriptor, 108U, static_cast<uint32_t>(slots.size()));
	put_le<uint32_t>(descriptor, 112U, component_count);
	put_le<uint32_t>(descriptor, 116U, static_cast<uint32_t>(unique_quants.size()));
	put_le<uint32_t>(descriptor, 120U, shard.block_group_count);
	uint64_t coordinate_total = 0U;
	for (const auto& slot : slots) {
		coordinate_total += slot.coordinate_count;
	}
	put_le<uint32_t>(descriptor, 124U, static_cast<uint32_t>(coordinate_total));
	put_le<uint32_t>(descriptor, 128U, static_cast<uint32_t>(cells.size()));
	put_le<uint32_t>(descriptor, 132U, static_cast<uint32_t>(topology.size()));
	put_le<uint16_t>(descriptor, 136U, options.rank_checkpoint_images);
	put_le<uint16_t>(descriptor, 138U, options.topology_checkpoint_coordinates);
	put_le<uint64_t>(descriptor,
	                 kDescriptorChecksumByte,
	                 compute_descriptor_crc64(descriptor.data(), descriptor.size()));

	std::error_code directory_error;
	std::filesystem::create_directories(output_path.parent_path(), directory_error);
	if (directory_error) {
		fail("cannot create descriptor output directory: " + directory_error.message());
	}
	write_staged_file(output_path, descriptor);
	report.descriptor_bytes = descriptor.size();
	if (options.validate_after_write) {
		auto loaded = JpegDctBlockMajorAccessDescriptor::Open(output_path);
		loaded.ValidateSource(manifest_path, shard, metadata_path, fls_path);
		report.validation = loaded.ValidateAgainstMetadata(metadata, options.exhaustive_rank_validation);
	}
	return report;
}

JpegDctBlockMajorAccessDatasetReport build_jpeg_dct_block_major_access_dataset(
	const std::filesystem::path& manifest_path,
	const std::filesystem::path& output_directory,
	const JpegDctBlockMajorAccessBuildOptions& options) {
	const auto manifest = detail::read_jpeg_dct_shard_manifest_file(manifest_path);
	if (manifest.version != 1U) {
		fail("dataset builder requires manifest version 1");
	}
	const auto directory = output_directory.empty() ? manifest_path.parent_path() / "block_major_access_v1"
	                                                : output_directory;
	std::error_code error;
	std::filesystem::create_directories(directory, error);
	if (error) {
		fail("cannot create dataset descriptor directory: " + error.message());
	}
	JpegDctBlockMajorAccessDatasetReport report;
	for (const auto& shard : manifest.shards) {
		report.source_dataset_bytes += shard.fls_file_size + shard.metadata_file_size;
		auto shard_report = build_jpeg_dct_block_major_access_descriptor(
		    manifest_path, manifest, shard, sidecar_name(directory, shard.shard_id), options);
		report.descriptor_bytes += shard_report.descriptor_bytes;
		report.shards.push_back(std::move(shard_report));
	}

	std::vector<uint8_t> index(64U, 0U);
	std::copy(kIndexMagic.begin(), kIndexMagic.end(), index.begin());
	put_le<uint16_t>(index, 8U, kJpegDctBlockMajorAccessVersion);
	put_le<uint16_t>(index, 10U, 64U);
	put_le<uint32_t>(index, 12U, kFlagLittleEndian | kFlagCrc64Ecma);
	put_le<uint64_t>(index, 16U, crc64_file(manifest_path));
	put_le<uint64_t>(index, 24U, checked_file_size(manifest_path));
	put_le<uint32_t>(index, 32U, static_cast<uint32_t>(manifest.shards.size()));
	for (size_t shard_index = 0U; shard_index < manifest.shards.size(); ++shard_index) {
		const auto& shard = manifest.shards[shard_index];
		const auto& shard_report = report.shards[shard_index];
		append_le<uint32_t>(index, shard.shard_id);
		append_le<uint32_t>(index, shard.image_count);
		append_le<uint64_t>(index, shard.first_global_image_index);
		append_le<uint64_t>(index, shard_report.descriptor_bytes);
		auto descriptor = JpegDctBlockMajorAccessDescriptor::Open(shard_report.descriptor_path);
		append_le<uint64_t>(index, descriptor.descriptor_crc64());
	}
	put_le<uint64_t>(index, 40U, index.size());
	put_le<uint64_t>(index, 48U, crc64_with_zeroed_field(index.data(), index.size(), 48U));
	report.index_path = directory / "manifest.block_major_access.bin";
	write_staged_file(report.index_path, index);
	report.index_bytes = index.size();
	const auto added_bytes = report.descriptor_bytes + report.index_bytes;
	report.storage_growth_ratio = report.source_dataset_bytes == 0U
	                                  ? 0.0
	                                  : static_cast<double>(added_bytes) /
	                                        static_cast<double>(report.source_dataset_bytes);
	report.passes_one_percent  = added_bytes <= report.source_dataset_bytes / 100U;
	report.passes_half_percent = added_bytes <= report.source_dataset_bytes / 200U;
	if (!report.passes_one_percent) {
		fail("serialized block-major access descriptors exceed the hard one-percent storage limit");
	}
	return report;
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
