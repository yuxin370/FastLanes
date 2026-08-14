#include "jpeg/jpeg_dct_canonical_plan.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace galp::jpeg::detail {
namespace {

constexpr std::array<std::byte, 8U> kMagic {
	std::byte {'G'}, std::byte {'A'}, std::byte {'L'}, std::byte {'P'},
	std::byte {'C'}, std::byte {'P'}, std::byte {'0'}, std::byte {'2'}};
constexpr uint32_t kVersion          = 2U;
constexpr uint32_t kEndianMarker     = UINT32_C(0x01020304);
constexpr uint32_t kHeaderBytes      = 256U;
constexpr uint32_t kPlannerAbi       = 2U;
constexpr size_t   kChecksumOffset   = 120U;
constexpr size_t   kCountOffset      = 144U;
constexpr size_t   kCountCount       = 10U;
constexpr uint64_t kFileHardCapBytes = UINT64_C(16) * 1024U * 1024U;

constexpr std::array<uint64_t, 256U> make_crc64_table() {
	std::array<uint64_t, 256U> table {};
	for (uint64_t index = 0U; index < table.size(); ++index) {
		uint64_t value = index << 56U;
		for (uint32_t bit = 0U; bit < 8U; ++bit) {
			value = (value & UINT64_C(0x8000000000000000)) != 0U
			            ? (value << 1U) ^ UINT64_C(0x42F0E1EBA9EA3693)
			            : value << 1U;
		}
		table[index] = value;
	}
	return table;
}

constexpr auto kCrc64Table = make_crc64_table();

uint64_t crc64_update(uint64_t crc, const std::byte* data, const size_t size) {
	for (size_t index = 0U; index < size; ++index) {
		const auto byte = std::to_integer<uint8_t>(data[index]);
		crc = kCrc64Table[((crc >> 56U) ^ byte) & 0xFFU] ^ (crc << 8U);
	}
	return crc;
}

uint64_t crc64_with_zeroed_checksum(const std::vector<std::byte>& bytes) {
	if (bytes.size() < kChecksumOffset + sizeof(uint64_t)) {
		throw std::runtime_error("canonical plan sidecar is shorter than its checksum field");
	}
	uint64_t crc = crc64_update(0U, bytes.data(), kChecksumOffset);
	const std::array<std::byte, sizeof(uint64_t)> zeros {};
	crc = crc64_update(crc, zeros.data(), zeros.size());
	return crc64_update(
	    crc,
	    bytes.data() + kChecksumOffset + sizeof(uint64_t),
	    bytes.size() - kChecksumOffset - sizeof(uint64_t));
}

template <typename T>
void append_le(std::vector<std::byte>& output, const T value) {
	using U = std::make_unsigned_t<T>;
	const auto bits = static_cast<U>(value);
	for (size_t index = 0U; index < sizeof(T); ++index) {
		output.push_back(std::byte((bits >> (index * 8U)) & U {0xFFU}));
	}
}

template <typename T>
void put_le(std::vector<std::byte>& output, const size_t offset, const T value) {
	if (offset > output.size() || sizeof(T) > output.size() - offset) {
		throw std::runtime_error("canonical plan header write is out of bounds");
	}
	using U = std::make_unsigned_t<T>;
	const auto bits = static_cast<U>(value);
	for (size_t index = 0U; index < sizeof(T); ++index) {
		output[offset + index] = std::byte((bits >> (index * 8U)) & U {0xFFU});
	}
}

class Cursor {
public:
	Cursor(const std::byte* data, const size_t size) : data_(data), size_(size) {}

	template <typename T>
	T read(const char* label) {
		if (position_ > size_ || sizeof(T) > size_ - position_) {
			throw std::runtime_error(std::string("canonical plan is truncated at ") + label);
		}
		using U = std::make_unsigned_t<T>;
		U bits = 0U;
		for (size_t index = 0U; index < sizeof(T); ++index) {
			bits |= static_cast<U>(std::to_integer<uint8_t>(data_[position_ + index])) << (index * 8U);
		}
		position_ += sizeof(T);
		return static_cast<T>(bits);
	}

	void read_bytes(uint8_t* output, const size_t count, const char* label) {
		if (position_ > size_ || count > size_ - position_) {
			throw std::runtime_error(std::string("canonical plan is truncated at ") + label);
		}
		std::memcpy(output, data_ + position_, count);
		position_ += count;
	}

	[[nodiscard]] size_t position() const noexcept { return position_; }

private:
	const std::byte* data_ = nullptr;
	size_t size_           = 0U;
	size_t position_       = 0U;
};

template <typename T>
T header_value(const std::vector<std::byte>& bytes, const size_t offset, const char* label) {
	Cursor cursor(bytes.data() + offset, bytes.size() - offset);
	return cursor.read<T>(label);
}

void append_bool(std::vector<std::byte>& output, const bool value) {
	append_le<uint8_t>(output, value ? 1U : 0U);
}

bool read_bool(Cursor& cursor, const char* label) {
	const auto value = cursor.read<uint8_t>(label);
	if (value > 1U) {
		throw std::runtime_error(std::string("canonical plan has an invalid boolean at ") + label);
	}
	return value != 0U;
}

void append_var_u32(std::vector<std::byte>& output, uint32_t value) {
	do {
		auto byte = static_cast<uint8_t>(value & 0x7FU);
		value >>= 7U;
		if (value != 0U) {
			byte |= 0x80U;
		}
		output.push_back(std::byte {byte});
	} while (value != 0U);
}

uint32_t read_var_u32(Cursor& cursor, const char* label) {
	uint32_t value = 0U;
	for (uint32_t index = 0U; index < 5U; ++index) {
		const auto byte = cursor.read<uint8_t>(label);
		const auto payload = static_cast<uint32_t>(byte & 0x7FU);
		if (index == 4U && payload > 0x0FU) {
			throw std::runtime_error(std::string("canonical plan varint overflows at ") + label);
		}
		value |= payload << (index * 7U);
		if ((byte & 0x80U) == 0U) {
			if (index != 0U && payload == 0U) {
				throw std::runtime_error(std::string("canonical plan varint is non-canonical at ") + label);
			}
			return value;
		}
	}
	throw std::runtime_error(std::string("canonical plan varint is unterminated at ") + label);
}

void append_var_i32(std::vector<std::byte>& output, const int32_t value) {
	const auto bits = std::bit_cast<uint32_t>(value);
	append_var_u32(output, (bits << 1U) ^ (0U - (bits >> 31U)));
}

int32_t read_var_i32(Cursor& cursor, const char* label) {
	const auto encoded = read_var_u32(cursor, label);
	return std::bit_cast<int32_t>((encoded >> 1U) ^ (0U - (encoded & 1U)));
}

template <typename T>
T read_var_bounded(Cursor& cursor, const char* label) {
	const auto value = read_var_u32(cursor, label);
	if (value > std::numeric_limits<T>::max()) {
		throw std::runtime_error(std::string("canonical plan compact field overflows at ") + label);
	}
	return static_cast<T>(value);
}

void append_stats(std::vector<std::byte>& output, const JpegDctBlockMajorCompactPlanStats& stats) {
	append_le<uint64_t>(output, stats.request_count);
	append_le<uint64_t>(output, stats.request_sort_items);
	append_le<uint64_t>(output, stats.unique_image_count);
	append_le<uint64_t>(output, stats.duplicate_output_count);
	append_le<uint64_t>(output, stats.shard_local_request_runs);
	append_le<uint64_t>(output, stats.support_rectangle_count);
	append_le<uint64_t>(output, stats.touched_block_groups);
	append_le<uint64_t>(output, stats.group_rank_runs);
	append_le<uint64_t>(output, stats.selected_rowgroups);
	append_le<uint64_t>(output, stats.selected_vector_runs);
	append_le<uint64_t>(output, stats.selected_vectors);
	append_le<uint64_t>(output, stats.duplicate_physical_read_count);
	append_le<uint64_t>(output, stats.rowgroup_revisit_count);
	append_le<uint64_t>(output, stats.vector_run_revisit_count);
	append_le<uint64_t>(output, stats.physical_read_order_inversions);
	append_le<uint64_t>(output, stats.touched_rank_cells);
	append_le<uint64_t>(output, stats.rank_payload_bytes);
	append_le<uint64_t>(output, stats.touched_quant_tables);
	append_le<uint64_t>(output, stats.expanded_transform_items);
	append_le<uint64_t>(output, stats.global_transform_sort_items);
	append_le<uint64_t>(output, stats.compact_plan_bytes);
	append_le<uint64_t>(output, stats.compact_plan_peak_bytes);
	append_bool(output, stats.input_was_shard_local_monotonic);
}

JpegDctBlockMajorCompactPlanStats read_stats(Cursor& cursor) {
	JpegDctBlockMajorCompactPlanStats stats;
	stats.request_count = cursor.read<uint64_t>("stats request count");
	stats.request_sort_items = cursor.read<uint64_t>("stats request sort items");
	stats.unique_image_count = cursor.read<uint64_t>("stats unique images");
	stats.duplicate_output_count = cursor.read<uint64_t>("stats duplicate outputs");
	stats.shard_local_request_runs = cursor.read<uint64_t>("stats shard request runs");
	stats.support_rectangle_count = cursor.read<uint64_t>("stats support rectangles");
	stats.touched_block_groups = cursor.read<uint64_t>("stats groups");
	stats.group_rank_runs = cursor.read<uint64_t>("stats rank runs");
	stats.selected_rowgroups = cursor.read<uint64_t>("stats selected rowgroups");
	stats.selected_vector_runs = cursor.read<uint64_t>("stats selected vector runs");
	stats.selected_vectors = cursor.read<uint64_t>("stats selected vectors");
	stats.duplicate_physical_read_count = cursor.read<uint64_t>("stats duplicate physical reads");
	stats.rowgroup_revisit_count = cursor.read<uint64_t>("stats rowgroup revisits");
	stats.vector_run_revisit_count = cursor.read<uint64_t>("stats vector run revisits");
	stats.physical_read_order_inversions = cursor.read<uint64_t>("stats physical inversions");
	stats.touched_rank_cells = cursor.read<uint64_t>("stats rank cells");
	stats.rank_payload_bytes = cursor.read<uint64_t>("stats rank payload bytes");
	stats.touched_quant_tables = cursor.read<uint64_t>("stats quant tables");
	stats.expanded_transform_items = cursor.read<uint64_t>("stats expanded transform items");
	stats.global_transform_sort_items = cursor.read<uint64_t>("stats global sort items");
	stats.compact_plan_bytes = cursor.read<uint64_t>("stats compact bytes");
	stats.compact_plan_peak_bytes = cursor.read<uint64_t>("stats compact peak bytes");
	stats.input_was_shard_local_monotonic = read_bool(cursor, "stats monotonic input");
	return stats;
}

std::vector<std::byte> encode_payload(const JpegDctBlockMajorCompactPlan& plan) {
	std::vector<std::byte> output;
	output.reserve(static_cast<size_t>(std::min<uint64_t>(
	    plan.stats.compact_plan_bytes + 4096U, std::numeric_limits<size_t>::max())));
	append_stats(output, plan.stats);
	for (const auto& request : plan.requests) {
		uint8_t present_mask = 0U;
		for (uint32_t component_index = 0U; component_index < request.components.size(); ++component_index) {
			present_mask |= request.components[component_index].present
			                    ? static_cast<uint8_t>(1U << component_index)
			                    : 0U;
		}
		append_le<uint8_t>(output, present_mask);
		for (const auto& component : request.components) {
			if (!component.present) {
				continue;
			}
			append_var_i32(output, component.x);
			append_var_i32(output, component.y);
			append_var_u32(output, component.width);
			append_var_u32(output, component.height);
			append_var_u32(output, component.source_width_in_blocks);
			append_var_u32(output, component.source_height_in_blocks);
			append_var_u32(output, component.quant_table_index);
			append_var_u32(output, component.h_samp_factor);
			append_var_u32(output, component.v_samp_factor);
		}
	}
	for (const auto& binding : plan.group_bindings) {
		append_var_u32(output, binding.semantic_slot_id);
		append_var_u32(output, binding.block_x);
		append_var_u32(output, binding.block_y);
		append_var_u32(output, binding.group_id);
		append_var_u32(output, binding.rank_cell_id);
		append_var_u32(output, binding.runtime_rank_cell_index);
		append_var_u32(output, binding.fls_rowgroup_index);
		append_var_u32(output, binding.row_start_in_rowgroup);
		append_var_u32(output, binding.first_rank_run);
		append_var_u32(output, binding.rank_run_count);
	}
	for (const auto& run : plan.rank_runs) {
		append_le<uint32_t>(output, run.group_binding_index);
		append_le<uint32_t>(output, run.rank_begin);
		append_le<uint32_t>(output, run.rank_count);
	}
	for (const auto& cell : plan.rank_cells) {
		append_le<uint32_t>(output, cell.source_cell_id);
		append_le<uint32_t>(output, cell.image_count);
		append_le<uint32_t>(output, cell.payload_offset);
		append_le<uint32_t>(output, cell.payload_size);
		append_le<uint16_t>(output, cell.present_count);
		append_le<uint16_t>(output, cell.rank_checkpoint_images);
		append_le<uint8_t>(output, static_cast<uint8_t>(cell.encoding));
	}
	for (const auto value : plan.rank_payload) {
		append_le<uint8_t>(output, value);
	}
	for (const auto& table : plan.quant_tables) {
		append_le<uint64_t>(output, table.fingerprint);
		for (const auto value : table.values) {
			append_le<uint16_t>(output, value);
		}
	}
	for (const auto& rowgroup : plan.rowgroups) {
		append_le<uint32_t>(output, rowgroup.shard_id);
		append_le<uint32_t>(output, rowgroup.rowgroup_index);
		append_le<uint32_t>(output, rowgroup.first_vector_run);
		append_le<uint32_t>(output, rowgroup.vector_run_count);
		append_le<uint32_t>(output, rowgroup.selected_vectors);
	}
	for (const auto& run : plan.vector_runs) {
		append_le<uint32_t>(output, run.shard_id);
		append_le<uint32_t>(output, run.rowgroup_index);
		append_le<uint32_t>(output, run.first_vector);
		append_le<uint32_t>(output, run.vector_count);
	}
	return output;
}

size_t checked_count(const uint64_t count, const uint64_t payload_bytes, const uint64_t minimum_record_bytes) {
	if (count > std::numeric_limits<size_t>::max() ||
	    (minimum_record_bytes != 0U && count > payload_bytes / minimum_record_bytes)) {
		throw std::runtime_error("canonical plan section count exceeds payload bounds");
	}
	return static_cast<size_t>(count);
}

JpegDctBlockMajorCompactPlan decode_payload(
	const std::byte* data,
	const size_t size,
	const std::array<uint64_t, kCountCount>& counts,
	const JpegDctCanonicalPlanIdentity& identity) {
	Cursor cursor(data, size);
	JpegDctBlockMajorCompactPlan plan;
	plan.stats = read_stats(cursor);
	if (counts[0] != identity.image_count || counts[1] != identity.image_count ||
	    counts[2] != identity.image_count) {
		throw std::runtime_error("canonical plan image section counts do not match the shard identity");
	}
	plan.requests.reserve(checked_count(counts[0], size, 1U));
	for (uint64_t index = 0U; index < counts[0]; ++index) {
		JpegDctBlockMajorCompactRequest request;
		request.global_image_index = static_cast<uint32_t>(identity.first_global_image + index);
		request.shard_id = identity.shard_id;
		request.local_image_index = static_cast<uint32_t>(index);
		request.output_slot = static_cast<uint32_t>(index);
		request.unique_image_index = static_cast<uint32_t>(index);
		const auto present_mask = cursor.read<uint8_t>("request component presence mask");
		if ((present_mask & ~UINT8_C(0x07)) != 0U) {
			throw std::runtime_error("canonical plan request has an invalid component presence mask");
		}
		for (uint32_t component_index = 0U; component_index < request.components.size(); ++component_index) {
			auto& component = request.components[component_index];
			component.present = (present_mask & (1U << component_index)) != 0U;
			if (!component.present) {
				continue;
			}
			component.semantic_slot_id = component_index;
			component.x = read_var_i32(cursor, "support x");
			component.y = read_var_i32(cursor, "support y");
			component.width = read_var_u32(cursor, "support width");
			component.height = read_var_u32(cursor, "support height");
			component.source_width_in_blocks = read_var_bounded<uint16_t>(cursor, "support source width");
			component.source_height_in_blocks = read_var_bounded<uint16_t>(cursor, "support source height");
			component.quant_table_index = read_var_bounded<uint16_t>(cursor, "support quant table");
			component.h_samp_factor = read_var_bounded<uint8_t>(cursor, "support horizontal sampling");
			component.v_samp_factor = read_var_bounded<uint8_t>(cursor, "support vertical sampling");
		}
		plan.requests.push_back(std::move(request));
	}
	plan.unique_images.reserve(checked_count(counts[1], size, 0U));
	for (uint64_t index = 0U; index < counts[1]; ++index) {
		plan.unique_images.push_back({identity.shard_id,
		                              static_cast<uint32_t>(index),
		                              static_cast<uint32_t>(index),
		                              1U});
	}
	plan.duplicate_output_slots.reserve(checked_count(counts[2], size, 0U));
	for (uint64_t index = 0U; index < counts[2]; ++index) {
		plan.duplicate_output_slots.push_back(static_cast<uint32_t>(index));
	}
	plan.group_bindings.reserve(checked_count(counts[3], size, 10U));
	for (uint64_t index = 0U; index < counts[3]; ++index) {
		plan.group_bindings.push_back({identity.shard_id,
		                               read_var_u32(cursor, "binding semantic slot"),
		                               read_var_u32(cursor, "binding block x"),
		                               read_var_u32(cursor, "binding block y"),
		                               read_var_u32(cursor, "binding group"),
		                               read_var_u32(cursor, "binding source rank cell"),
		                               read_var_u32(cursor, "binding runtime rank cell"),
		                               read_var_u32(cursor, "binding rowgroup"),
		                               read_var_u32(cursor, "binding row start"),
		                               read_var_u32(cursor, "binding first rank run"),
		                               read_var_u32(cursor, "binding rank run count")});
	}
	plan.rank_runs.reserve(checked_count(counts[4], size, 12U));
	for (uint64_t index = 0U; index < counts[4]; ++index) {
		plan.rank_runs.push_back({cursor.read<uint32_t>("rank run binding"),
		                          cursor.read<uint32_t>("rank run begin"),
		                          cursor.read<uint32_t>("rank run count")});
	}
	plan.rank_cells.reserve(checked_count(counts[5], size, 21U));
	for (uint64_t index = 0U; index < counts[5]; ++index) {
		JpegDctBlockMajorRuntimeRankCell cell;
		cell.shard_id = identity.shard_id;
		cell.source_cell_id = cursor.read<uint32_t>("rank cell source id");
		cell.image_count = cursor.read<uint32_t>("rank cell image count");
		cell.payload_offset = cursor.read<uint32_t>("rank cell payload offset");
		cell.payload_size = cursor.read<uint32_t>("rank cell payload size");
		cell.present_count = cursor.read<uint16_t>("rank cell present count");
		cell.rank_checkpoint_images = cursor.read<uint16_t>("rank cell checkpoint");
		const auto encoding = cursor.read<uint8_t>("rank cell encoding");
		if (encoding > static_cast<uint8_t>(JpegDctBlockMajorPresenceEncoding::kBitmapRank)) {
			throw std::runtime_error("canonical plan has an invalid rank-cell encoding");
		}
		cell.encoding = static_cast<JpegDctBlockMajorPresenceEncoding>(encoding);
		plan.rank_cells.push_back(cell);
	}
	plan.rank_payload.resize(checked_count(counts[6], size, 1U));
	if (!plan.rank_payload.empty()) {
		cursor.read_bytes(plan.rank_payload.data(), plan.rank_payload.size(), "rank payload");
	}
	plan.quant_tables.reserve(checked_count(counts[7], size, 136U));
	for (uint64_t index = 0U; index < counts[7]; ++index) {
		JpegDctBlockMajorRuntimeQuantTable table;
		table.fingerprint = cursor.read<uint64_t>("quant fingerprint");
		for (auto& value : table.values) {
			value = cursor.read<uint16_t>("quant value");
		}
		plan.quant_tables.push_back(table);
	}
	plan.rowgroups.reserve(checked_count(counts[8], size, 20U));
	for (uint64_t index = 0U; index < counts[8]; ++index) {
		plan.rowgroups.push_back({cursor.read<uint32_t>("selected rowgroup shard"),
		                          cursor.read<uint32_t>("selected rowgroup index"),
		                          cursor.read<uint32_t>("selected rowgroup first run"),
		                          cursor.read<uint32_t>("selected rowgroup run count"),
		                          cursor.read<uint32_t>("selected rowgroup vectors")});
	}
	plan.vector_runs.reserve(checked_count(counts[9], size, 16U));
	for (uint64_t index = 0U; index < counts[9]; ++index) {
		plan.vector_runs.push_back({cursor.read<uint32_t>("vector run shard"),
		                           cursor.read<uint32_t>("vector run rowgroup"),
		                           cursor.read<uint32_t>("vector run first"),
		                           cursor.read<uint32_t>("vector run count")});
	}
	if (cursor.position() != size) {
		throw std::runtime_error("canonical plan has trailing payload bytes");
	}
	return plan;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
	const auto size = std::filesystem::file_size(path);
	if (size < kHeaderBytes || size > kFileHardCapBytes || size > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error("canonical plan sidecar size is outside bounds");
	}
	std::vector<std::byte> bytes(static_cast<size_t>(size));
	std::ifstream input(path, std::ios::binary);
	input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	if (!input || input.peek() != std::char_traits<char>::eof()) {
		throw std::runtime_error("canonical plan sidecar is truncated or changed while reading");
	}
	return bytes;
}

void validate_plan(
	const JpegDctBlockMajorCompactPlan& plan,
	const JpegDctCanonicalPlanIdentity& identity) {
	if (plan.requests.size() != identity.image_count || plan.unique_images.size() != identity.image_count ||
	    plan.duplicate_output_slots.size() != identity.image_count || plan.stats.request_count != identity.image_count ||
	    plan.stats.unique_image_count != identity.image_count || plan.stats.duplicate_output_count != 0U ||
	    !plan.stats.input_was_shard_local_monotonic || plan.stats.selected_rowgroups != plan.rowgroups.size() ||
	    plan.stats.selected_vector_runs != plan.vector_runs.size() ||
	    plan.stats.touched_block_groups != plan.group_bindings.size() ||
	    plan.stats.group_rank_runs != plan.rank_runs.size() || plan.stats.touched_rank_cells != plan.rank_cells.size() ||
	    plan.stats.rank_payload_bytes != plan.rank_payload.size() ||
	    plan.stats.touched_quant_tables != plan.quant_tables.size()) {
		throw std::runtime_error("canonical plan structural counters are inconsistent");
	}
	uint64_t selected_vectors = 0U;
	for (uint32_t index = 0U; index < identity.image_count; ++index) {
		const auto& request = plan.requests[index];
		const auto& image = plan.unique_images[index];
		if (request.global_image_index != identity.first_global_image + index || request.shard_id != identity.shard_id ||
		    request.local_image_index != index || request.output_slot != index || request.unique_image_index != index ||
		    request.horizontal_flip || image.shard_id != identity.shard_id || image.local_image_index != index ||
		    image.first_fanout != index || image.fanout_count != 1U || plan.duplicate_output_slots[index] != index) {
			throw std::runtime_error("canonical plan image range or fanout is not canonical");
		}
		for (uint32_t component_index = 0U; component_index < request.components.size(); ++component_index) {
			const auto& component = request.components[component_index];
			if (component.present && component.semantic_slot_id != component_index) {
				throw std::runtime_error("canonical plan component slot is not canonical");
			}
		}
	}
	uint64_t previous_key = 0U;
	bool have_previous = false;
	for (const auto& rowgroup : plan.rowgroups) {
		if (rowgroup.shard_id != identity.shard_id || rowgroup.first_vector_run > plan.vector_runs.size() ||
		    rowgroup.vector_run_count > plan.vector_runs.size() - rowgroup.first_vector_run) {
			throw std::runtime_error("canonical plan rowgroup binding is invalid");
		}
		const auto key = (static_cast<uint64_t>(rowgroup.shard_id) << 32U) | rowgroup.rowgroup_index;
		if (have_previous && key <= previous_key) {
			throw std::runtime_error("canonical plan rowgroups are not strictly ordered");
		}
		previous_key = key;
		have_previous = true;
		uint64_t rowgroup_vectors = 0U;
		uint32_t previous_end = 0U;
		for (uint32_t run_index = 0U; run_index < rowgroup.vector_run_count; ++run_index) {
			const auto& run = plan.vector_runs[rowgroup.first_vector_run + run_index];
			if (run.shard_id != rowgroup.shard_id || run.rowgroup_index != rowgroup.rowgroup_index ||
			    run.vector_count == 0U || run.first_vector < previous_end ||
			    run.first_vector > identity.rowgroup_vectors ||
			    run.vector_count > identity.rowgroup_vectors - run.first_vector) {
				throw std::runtime_error("canonical plan vector run is invalid");
			}
			previous_end = run.first_vector + run.vector_count;
			rowgroup_vectors += run.vector_count;
		}
		if (rowgroup_vectors != rowgroup.selected_vectors) {
			throw std::runtime_error("canonical plan selected-vector counter is inconsistent");
		}
		selected_vectors += rowgroup_vectors;
	}
	if (selected_vectors != plan.stats.selected_vectors) {
		throw std::runtime_error("canonical plan total selected-vector counter is inconsistent");
	}
	for (const auto& binding : plan.group_bindings) {
		if (binding.shard_id != identity.shard_id || binding.runtime_rank_cell_index >= plan.rank_cells.size() ||
		    binding.first_rank_run > plan.rank_runs.size() ||
		    binding.rank_run_count > plan.rank_runs.size() - binding.first_rank_run) {
			throw std::runtime_error("canonical plan group binding is invalid");
		}
	}
	for (const auto& run : plan.rank_runs) {
		if (run.group_binding_index >= plan.group_bindings.size()) {
			throw std::runtime_error("canonical plan rank run references an invalid group");
		}
	}
	for (const auto& cell : plan.rank_cells) {
		if (cell.shard_id != identity.shard_id || cell.payload_offset > plan.rank_payload.size() ||
		    cell.payload_size > plan.rank_payload.size() - cell.payload_offset) {
			throw std::runtime_error("canonical plan rank cell payload is invalid");
		}
	}
}

std::vector<std::byte> encode_file(
	const JpegDctCanonicalPlanIdentity& identity,
	const JpegDctBlockMajorCompactPlan& plan) {
	auto payload = encode_payload(plan);
	if (payload.size() > kFileHardCapBytes - kHeaderBytes) {
		throw std::runtime_error("canonical plan payload exceeds its 16 MiB hard cap");
	}
	std::vector<std::byte> bytes(kHeaderBytes, std::byte {0});
	std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
	put_le<uint32_t>(bytes, 8U, kVersion);
	put_le<uint32_t>(bytes, 12U, kEndianMarker);
	put_le<uint32_t>(bytes, 16U, kHeaderBytes);
	put_le<uint32_t>(bytes, 20U, kPlannerAbi);
	put_le<uint64_t>(bytes, 24U, identity.manifest_size);
	put_le<uint64_t>(bytes, 32U, identity.manifest_crc64);
	put_le<uint64_t>(bytes, 40U, identity.descriptor_size);
	put_le<uint64_t>(bytes, 48U, identity.descriptor_crc64);
	put_le<uint64_t>(bytes, 56U, identity.source_size);
	put_le<uint64_t>(bytes, 64U, identity.source_stat_digest);
	put_le<uint64_t>(bytes, 72U, identity.source_payload_crc64);
	put_le<uint64_t>(bytes, 80U, identity.first_global_image);
	put_le<uint64_t>(bytes, 88U, identity.transform_digest);
	const auto audit_digest = crc64_update(0U, payload.data(), payload.size());
	put_le<uint64_t>(bytes, 96U, audit_digest);
	put_le<uint64_t>(bytes, 104U, kHeaderBytes);
	put_le<uint64_t>(bytes, 112U, payload.size());
	put_le<uint32_t>(bytes, 128U, identity.shard_id);
	put_le<uint32_t>(bytes, 132U, identity.image_count);
	put_le<uint32_t>(bytes, 136U, identity.rowgroup_vectors);
	const std::array<uint64_t, kCountCount> counts {
	    plan.requests.size(), plan.unique_images.size(), plan.duplicate_output_slots.size(),
	    plan.group_bindings.size(), plan.rank_runs.size(), plan.rank_cells.size(), plan.rank_payload.size(),
	    plan.quant_tables.size(), plan.rowgroups.size(), plan.vector_runs.size()};
	for (size_t index = 0U; index < counts.size(); ++index) {
		put_le<uint64_t>(bytes, kCountOffset + index * sizeof(uint64_t), counts[index]);
	}
	bytes.insert(bytes.end(), payload.begin(), payload.end());
	put_le<uint64_t>(bytes, kChecksumOffset, crc64_with_zeroed_checksum(bytes));
	return bytes;
}

class FileLock {
public:
	explicit FileLock(const std::filesystem::path& target) {
		const auto lock_path = target.string() + ".lock";
		fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
		if (fd_ < 0 || ::flock(fd_, LOCK_EX) != 0) {
			const auto error = errno;
			if (fd_ >= 0) {
				::close(fd_);
			}
			throw std::system_error(error, std::generic_category(), "failed to lock canonical plan sidecar");
		}
	}
	~FileLock() {
		if (fd_ >= 0) {
			::flock(fd_, LOCK_UN);
			::close(fd_);
		}
	}
	FileLock(const FileLock&) = delete;
	FileLock& operator=(const FileLock&) = delete;
private:
	int fd_ = -1;
};

void write_all(const int fd, const std::byte* data, const size_t size) {
	size_t written = 0U;
	while (written < size) {
		const auto count = ::write(fd, data + written, size - written);
		if (count < 0) {
			if (errno == EINTR) {
				continue;
			}
			throw std::system_error(errno, std::generic_category(), "failed to write canonical plan sidecar");
		}
		if (count == 0) {
			throw std::runtime_error("canonical plan sidecar write made no progress");
		}
		written += static_cast<size_t>(count);
	}
}

void write_atomic(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
	static std::atomic<uint64_t> sequence {0U};
	const auto staged = path.string() + ".tmp." + std::to_string(::getpid()) + "." +
	                    std::to_string(sequence.fetch_add(1U, std::memory_order_relaxed));
	int fd = -1;
	try {
		fd = ::open(staged.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
		if (fd < 0) {
			throw std::system_error(errno, std::generic_category(), "failed to create canonical plan staging file");
		}
		write_all(fd, bytes.data(), bytes.size());
		if (::fsync(fd) != 0) {
			throw std::system_error(errno, std::generic_category(), "failed to fsync canonical plan staging file");
		}
		if (::close(fd) != 0) {
			fd = -1;
			throw std::system_error(errno, std::generic_category(), "failed to close canonical plan staging file");
		}
		fd = -1;
		if (::rename(staged.c_str(), path.c_str()) != 0) {
			throw std::system_error(errno, std::generic_category(), "failed to publish canonical plan sidecar");
		}
		const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
		const auto directory_fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
		if (directory_fd < 0) {
			throw std::system_error(errno, std::generic_category(), "failed to open canonical plan directory");
		}
		const auto sync_result = ::fsync(directory_fd);
		const auto sync_error = errno;
		::close(directory_fd);
		if (sync_result != 0) {
			throw std::system_error(sync_error, std::generic_category(), "failed to fsync canonical plan directory");
		}
	} catch (...) {
		if (fd >= 0) {
			::close(fd);
		}
		std::error_code ignored;
		std::filesystem::remove(staged, ignored);
		throw;
	}
}

void append_transform_vector(std::vector<std::byte>& bytes, const std::vector<uint32_t>& values) {
	append_le<uint64_t>(bytes, values.size());
	for (const auto value : values) {
		append_le<uint32_t>(bytes, value);
	}
}

} // namespace

uint64_t jpeg_dct_canonical_transform_digest(const JpegDctGridTransformSpec& transform) {
	std::vector<std::byte> bytes;
	append_le<uint32_t>(bytes, transform.y_output_width_blocks);
	append_le<uint32_t>(bytes, transform.y_output_height_blocks);
	append_le<uint32_t>(bytes, transform.cbcr_output_width_blocks);
	append_le<uint32_t>(bytes, transform.cbcr_output_height_blocks);
	append_le<uint32_t>(bytes, transform.crop_reference_width_blocks);
	append_le<uint32_t>(bytes, transform.crop_reference_height_blocks);
	append_le<uint32_t>(bytes, transform.crop_origin_alignment_blocks);
	append_le<uint32_t>(bytes, transform.chroma_crop_scale_x);
	append_le<uint32_t>(bytes, transform.chroma_crop_scale_y);
	append_le<uint32_t>(bytes, std::bit_cast<uint32_t>(transform.clamp_min));
	append_le<uint32_t>(bytes, std::bit_cast<uint32_t>(transform.clamp_max));
	append_le<uint32_t>(bytes, static_cast<uint32_t>(transform.output_data_type));
	append_le<uint32_t>(bytes, std::bit_cast<uint32_t>(transform.output_add));
	append_le<uint32_t>(bytes, std::bit_cast<uint32_t>(transform.output_scale));
	append_bool(bytes, transform.dequantize);
	append_bool(bytes, transform.require_all_coefficients);
	append_bool(bytes, transform.allow_grayscale);
	append_transform_vector(bytes, transform.preferred_small_crop_width_blocks);
	append_transform_vector(bytes, transform.preferred_small_crop_height_blocks);
	append_le<uint64_t>(bytes, transform.allowed_chroma_sampling_ratios.size());
	for (const auto& ratio : transform.allowed_chroma_sampling_ratios) {
		append_le<uint16_t>(bytes, ratio.horizontal_numerator);
		append_le<uint16_t>(bytes, ratio.horizontal_denominator);
		append_le<uint16_t>(bytes, ratio.vertical_numerator);
		append_le<uint16_t>(bytes, ratio.vertical_denominator);
	}
	return crc64_update(0U, bytes.data(), bytes.size());
}

uint64_t jpeg_dct_canonical_source_stat_digest(const std::filesystem::path& source_path) {
	struct stat status {};
	if (::stat(source_path.c_str(), &status) != 0) {
		throw std::system_error(errno, std::generic_category(), "failed to stat canonical plan source");
	}
	std::vector<std::byte> bytes;
	append_le<uint64_t>(bytes, static_cast<uint64_t>(status.st_dev));
	append_le<uint64_t>(bytes, static_cast<uint64_t>(status.st_ino));
	append_le<uint64_t>(bytes, static_cast<uint64_t>(status.st_size));
	append_le<uint64_t>(bytes, static_cast<uint64_t>(status.st_mtim.tv_sec));
	append_le<uint64_t>(bytes, static_cast<uint64_t>(status.st_mtim.tv_nsec));
	append_le<uint64_t>(bytes, static_cast<uint64_t>(status.st_ctim.tv_sec));
	append_le<uint64_t>(bytes, static_cast<uint64_t>(status.st_ctim.tv_nsec));
	return crc64_update(0U, bytes.data(), bytes.size());
}

uint64_t jpeg_dct_canonical_plan_audit_digest(const JpegDctBlockMajorCompactPlan& plan) {
	const auto payload = encode_payload(plan);
	return crc64_update(0U, payload.data(), payload.size());
}

JpegDctCanonicalPlanLoadResult load_jpeg_dct_canonical_plan_template(
	const std::filesystem::path& path,
	const JpegDctCanonicalPlanIdentity& identity) {
	const auto load_started = std::chrono::steady_clock::now();
	auto bytes = read_file(path);
	const auto load_finished = std::chrono::steady_clock::now();
	if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin()) ||
	    header_value<uint32_t>(bytes, 8U, "version") != kVersion ||
	    header_value<uint32_t>(bytes, 12U, "endian marker") != kEndianMarker ||
	    header_value<uint32_t>(bytes, 16U, "header size") != kHeaderBytes ||
	    header_value<uint32_t>(bytes, 20U, "planner ABI") != kPlannerAbi ||
	    header_value<uint64_t>(bytes, 24U, "manifest size") != identity.manifest_size ||
	    header_value<uint64_t>(bytes, 32U, "manifest CRC") != identity.manifest_crc64 ||
	    header_value<uint64_t>(bytes, 40U, "descriptor size") != identity.descriptor_size ||
	    header_value<uint64_t>(bytes, 48U, "descriptor CRC") != identity.descriptor_crc64 ||
	    header_value<uint64_t>(bytes, 56U, "source size") != identity.source_size ||
	    header_value<uint64_t>(bytes, 64U, "source stat digest") != identity.source_stat_digest ||
	    header_value<uint64_t>(bytes, 72U, "source payload CRC") != identity.source_payload_crc64 ||
	    header_value<uint64_t>(bytes, 80U, "first image") != identity.first_global_image ||
	    header_value<uint64_t>(bytes, 88U, "transform digest") != identity.transform_digest ||
	    header_value<uint32_t>(bytes, 128U, "shard id") != identity.shard_id ||
	    header_value<uint32_t>(bytes, 132U, "image count") != identity.image_count ||
	    header_value<uint32_t>(bytes, 136U, "rowgroup vectors") != identity.rowgroup_vectors) {
		throw std::runtime_error("canonical plan sidecar identity or ABI mismatch");
	}
	const auto payload_offset = header_value<uint64_t>(bytes, 104U, "payload offset");
	const auto payload_size = header_value<uint64_t>(bytes, 112U, "payload size");
	if (payload_offset != kHeaderBytes || payload_size != bytes.size() - kHeaderBytes ||
	    header_value<uint64_t>(bytes, kChecksumOffset, "sidecar CRC") != crc64_with_zeroed_checksum(bytes)) {
		throw std::runtime_error("canonical plan sidecar length or checksum mismatch");
	}
	const auto audit_digest = header_value<uint64_t>(bytes, 96U, "plan audit digest");
	if (audit_digest != crc64_update(0U, bytes.data() + kHeaderBytes, static_cast<size_t>(payload_size))) {
		throw std::runtime_error("canonical plan payload audit digest mismatch");
	}
	std::array<uint64_t, kCountCount> counts {};
	for (size_t index = 0U; index < counts.size(); ++index) {
		counts[index] = header_value<uint64_t>(bytes, kCountOffset + index * sizeof(uint64_t), "section count");
	}
	auto plan = decode_payload(bytes.data() + kHeaderBytes, static_cast<size_t>(payload_size), counts, identity);
	validate_plan(plan, identity);
	const auto validation_finished = std::chrono::steady_clock::now();
	JpegDctCanonicalPlanLoadResult result;
	result.plan              = std::move(plan);
	result.sidecar_bytes     = bytes.size();
	result.plan_audit_digest = audit_digest;
	result.load_ms = std::chrono::duration<double, std::milli>(load_finished - load_started).count();
	result.validation_ms =
	    std::chrono::duration<double, std::milli>(validation_finished - load_finished).count();
	return result;
}

JpegDctCanonicalPlanEnsureResult ensure_jpeg_dct_canonical_plan_template(
	const std::filesystem::path& path,
	const JpegDctCanonicalPlanIdentity& identity,
	const JpegDctBlockMajorCompactPlan& plan) {
	static std::mutex process_mutex;
	std::lock_guard<std::mutex> process_guard(process_mutex);
	std::filesystem::create_directories(path.parent_path());
	FileLock lock(path);
	try {
		auto existing = load_jpeg_dct_canonical_plan_template(path, identity);
		if (encode_payload(existing.plan) != encode_payload(plan)) {
			throw std::runtime_error("canonical plan sidecar differs from the deterministic rebuilt plan");
		}
		return {existing.sidecar_bytes, existing.plan_audit_digest, true};
	} catch (const std::exception&) {
		// An explicit offline ensure call deterministically replaces any rejected
		// artifact. Production Plan() remains read-only and falls back in memory.
	}
	validate_plan(plan, identity);
	auto bytes = encode_file(identity, plan);
	write_atomic(path, bytes);
	auto validated = load_jpeg_dct_canonical_plan_template(path, identity);
	if (encode_payload(validated.plan) != encode_payload(plan)) {
		throw std::runtime_error("canonical plan sidecar roundtrip changed the deterministic plan");
	}
	return {validated.sidecar_bytes, validated.plan_audit_digest, false};
}

} // namespace galp::jpeg::detail
