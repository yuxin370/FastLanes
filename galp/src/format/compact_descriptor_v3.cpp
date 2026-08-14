#include "format/compact_descriptor_v3.hpp"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/footer/column_descriptor_generated.h"
#include "fls/footer/segment_descriptor.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/footer/table_descriptor_generated.h"
#include "fls/info.hpp"
#include "fls/io/file.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace galp::format {
namespace {

constexpr std::array<uint8_t, 8> kMagic {{'G', 'A', 'L', 'P', 'C', 'V', '3', '\0'}};
constexpr uint16_t               kFormatVersion              = 3U;
constexpr uint16_t               kHeaderBytes                = 256U;
constexpr uint32_t               kFlagLittleEndian           = 1U << 0U;
constexpr uint32_t               kFlagCrc64Ecma              = 1U << 1U;
constexpr uint32_t               kFlagCompressedPages        = 1U << 2U;
constexpr uint32_t               kFlagOmitExpressionSizes    = 1U << 3U;
constexpr uint32_t               kFlagDenseSegmentGeometry   = 1U << 4U;
constexpr uint32_t               kFlagDenseCoefficientRanges = 1U << 5U;
constexpr uint32_t               kRequiredFlags = kFlagLittleEndian | kFlagCrc64Ecma | kFlagCompressedPages;
constexpr uint32_t               kKnownFlags =
    kRequiredFlags | kFlagOmitExpressionSizes | kFlagDenseSegmentGeometry | kFlagDenseCoefficientRanges;
constexpr uint32_t kWriterFlags            = kKnownFlags;
constexpr uint64_t kCrc64Polynomial        = UINT64_C(0x42f0e1eba9ea3693);
constexpr size_t   kDescriptorChecksumByte = 168U;
constexpr size_t   kImageRecordBytes       = 32U;
constexpr size_t   kComponentRecordBytes   = 32U;
constexpr size_t   kRowgroupRecordBytes    = 48U;
constexpr size_t   kCoefficientRecordBytes = 8U;
constexpr size_t   kMaxColumnDepth         = 128U;
constexpr size_t   kMaxDecodedPageBytes    = 64U * 1024U * 1024U;

std::atomic<size_t> g_compact_mapping_count {0U};
std::atomic<size_t> g_compact_mapping_peak {0U};
std::atomic<size_t> g_compact_map_count {0U};
std::atomic<size_t> g_compact_unmap_count {0U};
std::atomic<size_t> g_compact_mapped_bytes {0U};
std::atomic<size_t> g_compact_mapped_bytes_peak {0U};

void update_peak(std::atomic<size_t>& peak_value, const size_t value) noexcept {
	auto peak = peak_value.load(std::memory_order_relaxed);
	while (value > peak &&
	       !peak_value.compare_exchange_weak(peak, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
	}
}

void record_compact_mapping(const size_t bytes) noexcept {
	const auto mappings = g_compact_mapping_count.fetch_add(1U, std::memory_order_relaxed) + 1U;
	const auto mapped_bytes = g_compact_mapped_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
	g_compact_map_count.fetch_add(1U, std::memory_order_relaxed);
	update_peak(g_compact_mapping_peak, mappings);
	update_peak(g_compact_mapped_bytes_peak, mapped_bytes);
}

void record_compact_unmapping(const size_t bytes) noexcept {
	g_compact_mapping_count.fetch_sub(1U, std::memory_order_relaxed);
	g_compact_mapped_bytes.fetch_sub(bytes, std::memory_order_relaxed);
	g_compact_unmap_count.fetch_add(1U, std::memory_order_relaxed);
}

struct Section {
	uint64_t offset = 0U;
	uint64_t size   = 0U;
};

[[noreturn]] void fail(const std::string& message) {
	throw std::runtime_error("CompactDescriptorV3: " + message);
}

void require_status(const fastlanes::Status status, const std::string_view context) {
	if (!status.success) {
		fail(std::string(context) + ": " + std::string(fastlanes::Status::message_for(status.code)));
	}
}

template <typename T>
void append_le(std::vector<uint8_t>& output, const T value) {
	static_assert(std::is_unsigned_v<T>);
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		output.push_back(static_cast<uint8_t>((value >> (byte * 8U)) & static_cast<T>(0xffU)));
	}
}

template <typename T>
void put_le(std::vector<uint8_t>& output, const size_t offset, const T value) {
	static_assert(std::is_unsigned_v<T>);
	if (offset > output.size() || sizeof(T) > output.size() - offset) {
		fail("internal header write exceeded the descriptor buffer");
	}
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		output[offset + byte] = static_cast<uint8_t>((value >> (byte * 8U)) & static_cast<T>(0xffU));
	}
}

template <typename T>
T read_le(const uint8_t* data, const size_t size, const size_t offset, const std::string_view label) {
	static_assert(std::is_unsigned_v<T>);
	if (offset > size || sizeof(T) > size - offset) {
		fail("truncated " + std::string(label));
	}
	T value = 0U;
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		value |= static_cast<T>(data[offset + byte]) << (byte * 8U);
	}
	return value;
}

void append_uleb128(std::vector<uint8_t>& output, uint64_t value) {
	do {
		uint8_t byte = static_cast<uint8_t>(value & 0x7fU);
		value >>= 7U;
		if (value != 0U) {
			byte |= 0x80U;
		}
		output.push_back(byte);
	} while (value != 0U);
}

uint64_t consume_uleb128(const uint8_t* const data, const size_t size, size_t& cursor, const std::string_view label) {
	uint64_t value = 0U;
	for (unsigned shift = 0U; shift < 64U; shift += 7U) {
		if (cursor >= size) {
			fail("truncated rowgroup page while reading " + std::string(label));
		}
		const uint8_t byte = data[cursor++];
		if (shift == 63U && (byte & 0xfeU) != 0U) {
			fail("ULEB128 overflow while reading " + std::string(label));
		}
		value |= static_cast<uint64_t>(byte & 0x7fU) << shift;
		if ((byte & 0x80U) == 0U) {
			return value;
		}
	}
	fail("overlong ULEB128 while reading " + std::string(label));
}

void flush_page_literals(std::vector<uint8_t>&       output,
                         const std::vector<uint8_t>& input,
                         size_t&                     literal_begin,
                         const size_t                literal_end) {
	while (literal_begin < literal_end) {
		const auto count = std::min<size_t>(128U, literal_end - literal_begin);
		output.push_back(static_cast<uint8_t>(count - 1U));
		output.insert(output.end(),
		              input.begin() + static_cast<std::ptrdiff_t>(literal_begin),
		              input.begin() + static_cast<std::ptrdiff_t>(literal_begin + count));
		literal_begin += count;
	}
}

uint32_t page_match_key(const std::vector<uint8_t>& input, const size_t offset) {
	return static_cast<uint32_t>(input[offset]) | (static_cast<uint32_t>(input[offset + 1U]) << 8U) |
	       (static_cast<uint32_t>(input[offset + 2U]) << 16U);
}

std::vector<uint8_t> encode_rowgroup_page(const std::vector<uint8_t>& input) {
	std::vector<uint8_t> raw;
	raw.reserve(input.size() + 12U);
	raw.push_back(0U);
	append_uleb128(raw, input.size());
	raw.insert(raw.end(), input.begin(), input.end());
	if (input.size() < 4U) {
		return raw;
	}

	std::vector<uint8_t> compressed;
	compressed.reserve(input.size());
	compressed.push_back(1U);
	append_uleb128(compressed, input.size());
	std::unordered_map<uint32_t, size_t> last_position;
	last_position.reserve(input.size());
	size_t cursor        = 0U;
	size_t literal_begin = 0U;
	while (cursor < input.size()) {
		size_t match_length   = 0U;
		size_t match_distance = 0U;
		if (cursor + 2U < input.size()) {
			const auto key   = page_match_key(input, cursor);
			const auto found = last_position.find(key);
			if (found != last_position.end() && cursor > found->second &&
			    cursor - found->second <= std::numeric_limits<uint16_t>::max()) {
				const auto limit = std::min<size_t>(130U, input.size() - cursor);
				while (match_length < limit && input[found->second + match_length] == input[cursor + match_length]) {
					++match_length;
				}
				if (match_length >= 4U) {
					match_distance = cursor - found->second;
				} else {
					match_length = 0U;
				}
			}
		}
		if (match_length == 0U) {
			if (cursor + 2U < input.size()) {
				last_position[page_match_key(input, cursor)] = cursor;
			}
			++cursor;
			continue;
		}
		flush_page_literals(compressed, input, literal_begin, cursor);
		compressed.push_back(static_cast<uint8_t>(0x80U | (match_length - 3U)));
		append_le<uint16_t>(compressed, static_cast<uint16_t>(match_distance));
		const auto match_end = cursor + match_length;
		for (size_t update = cursor; update < match_end; ++update) {
			if (update + 2U < input.size()) {
				last_position[page_match_key(input, update)] = update;
			}
		}
		cursor        = match_end;
		literal_begin = cursor;
	}
	flush_page_literals(compressed, input, literal_begin, input.size());
	return compressed.size() < raw.size() ? compressed : raw;
}

std::vector<uint8_t> decode_rowgroup_page(const uint8_t* const encoded, const size_t encoded_size) {
	if (encoded_size == 0U) {
		fail("rowgroup page is empty");
	}
	size_t     cursor           = 1U;
	const auto decoded_size_u64 = consume_uleb128(encoded, encoded_size, cursor, "decoded page size");
	if (decoded_size_u64 > kMaxDecodedPageBytes) {
		fail("decoded rowgroup page exceeds the safety limit");
	}
	const auto decoded_size = static_cast<size_t>(decoded_size_u64);
	if (encoded[0] == 0U) {
		if (cursor > encoded_size || decoded_size != encoded_size - cursor) {
			fail("raw rowgroup page has an invalid size");
		}
		return {encoded + cursor, encoded + encoded_size};
	}
	if (encoded[0] != 1U) {
		fail("rowgroup page uses an unknown scalar-page codec");
	}
	std::vector<uint8_t> output;
	output.reserve(decoded_size);
	while (output.size() < decoded_size) {
		if (cursor >= encoded_size) {
			fail("truncated compressed rowgroup page token stream");
		}
		const uint8_t token = encoded[cursor++];
		if ((token & 0x80U) == 0U) {
			const size_t literal_size = static_cast<size_t>(token) + 1U;
			if (literal_size > decoded_size - output.size() || cursor > encoded_size ||
			    literal_size > encoded_size - cursor) {
				fail("compressed rowgroup page literal exceeds its bounds");
			}
			output.insert(output.end(), encoded + cursor, encoded + cursor + literal_size);
			cursor += literal_size;
			continue;
		}
		const size_t match_size = static_cast<size_t>(token & 0x7fU) + 3U;
		if (cursor > encoded_size || sizeof(uint16_t) > encoded_size - cursor ||
		    match_size > decoded_size - output.size()) {
			fail("compressed rowgroup page match exceeds its bounds");
		}
		const auto distance = read_le<uint16_t>(encoded, encoded_size, cursor, "rowgroup page match distance");
		cursor += sizeof(uint16_t);
		if (distance == 0U || distance > output.size()) {
			fail("compressed rowgroup page has an invalid match distance");
		}
		for (size_t byte = 0U; byte < match_size; ++byte) {
			output.push_back(output[output.size() - distance]);
		}
	}
	if (cursor != encoded_size) {
		fail("compressed rowgroup page has trailing bytes");
	}
	return output;
}

uint64_t crc64_update(uint64_t crc, const uint8_t* data, const size_t size) {
	for (size_t index = 0U; index < size; ++index) {
		crc ^= static_cast<uint64_t>(data[index]) << 56U;
		for (unsigned bit = 0U; bit < 8U; ++bit) {
			crc = (crc & (UINT64_C(1) << 63U)) != 0U ? (crc << 1U) ^ kCrc64Polynomial : crc << 1U;
		}
	}
	return crc;
}

uint64_t descriptor_crc64(const uint8_t* const data, const size_t size) {
	if (size < kDescriptorChecksumByte + sizeof(uint64_t)) {
		fail("descriptor is shorter than its checksum field");
	}
	uint64_t                                    crc = crc64_update(0U, data, kDescriptorChecksumByte);
	const std::array<uint8_t, sizeof(uint64_t)> zeros {};
	crc = crc64_update(crc, zeros.data(), zeros.size());
	return crc64_update(
	    crc, data + kDescriptorChecksumByte + sizeof(uint64_t), size - kDescriptorChecksumByte - sizeof(uint64_t));
}

uint64_t crc64_file_range(fastlanes::File& file, uint64_t offset, uint64_t size) {
	constexpr size_t     kBufferBytes = 4U * 1024U * 1024U;
	std::vector<uint8_t> buffer(kBufferBytes);
	uint64_t             crc = 0U;
	while (size != 0U) {
		const auto chunk = static_cast<size_t>(std::min<uint64_t>(size, buffer.size()));
		file.ReadRangeUnchecked(buffer.data(), offset, chunk);
		crc = crc64_update(crc, buffer.data(), chunk);
		offset += chunk;
		size -= chunk;
	}
	return crc;
}

uint32_t checked_u32(const uint64_t value, const std::string_view label) {
	if (value > std::numeric_limits<uint32_t>::max()) {
		fail(std::string(label) + " exceeds uint32 range");
	}
	return static_cast<uint32_t>(value);
}

uint16_t checked_u16(const uint64_t value, const std::string_view label) {
	if (value > std::numeric_limits<uint16_t>::max()) {
		fail(std::string(label) + " exceeds uint16 range");
	}
	return static_cast<uint16_t>(value);
}

size_t checked_size(const uint64_t value, const std::string_view label) {
	if (value > std::numeric_limits<size_t>::max()) {
		fail(std::string(label) + " exceeds addressable memory");
	}
	return static_cast<size_t>(value);
}

uint64_t checked_product(const uint64_t left, const uint64_t right, const std::string_view label) {
	if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
		fail(std::string(label) + " overflows uint64");
	}
	return left * right;
}

void normalize_column(fastlanes::ColumnDescriptorT& column, const size_t depth) {
	if (depth > kMaxColumnDepth) {
		fail("column descriptor nesting exceeds the safety limit");
	}
	column.column_offset = 0U;
	column.total_size    = 0U;
	column.n_null        = 0U;
	if (column.max != nullptr) {
		std::fill(column.max->binary_data.begin(), column.max->binary_data.end(), uint8_t {0U});
	}
	for (auto& expression : column.expr_space) {
		if (expression == nullptr) {
			fail("column schema contains a null expression result");
		}
		expression->size = 0U;
	}
	for (auto& segment : column.segment_descriptors) {
		if (segment == nullptr) {
			fail("column schema contains a null segment descriptor");
		}
		segment->entrypoint_offset = 0U;
		segment->entrypoint_size   = 0U;
		segment->data_offset       = 0U;
		segment->data_size         = 0U;
	}
	for (auto& child : column.children) {
		if (child == nullptr) {
			fail("column schema contains a null child descriptor");
		}
		normalize_column(*child, depth + 1U);
	}
}

std::vector<uint8_t> normalized_column_schema(const fastlanes::ColumnDescriptor& column) {
	std::unique_ptr<fastlanes::ColumnDescriptorT> native(column.UnPack());
	if (native == nullptr) {
		fail("failed to unpack a column schema");
	}
	normalize_column(*native, 1U);
	flatbuffers::FlatBufferBuilder builder;
	const auto                     root = fastlanes::ColumnDescriptor::Pack(builder, native.get());
	fastlanes::FinishColumnDescriptorBuffer(builder, root);
	auto detached = builder.Release();
	return {detached.data(), detached.data() + detached.size()};
}

void append_column_variables(std::vector<uint8_t>&              output,
                             const fastlanes::ColumnDescriptor& source,
                             const size_t                       depth,
                             const uint64_t                     rowgroup_vectors,
                             const bool                         omit_expression_sizes,
                             const bool                         dense_segment_geometry) {
	if (depth > kMaxColumnDepth) {
		fail("column descriptor nesting exceeds the safety limit");
	}
	if (dense_segment_geometry && source.children() != nullptr && !source.children()->empty()) {
		fail("canonical dense segment geometry currently requires root-only columns");
	}
	if (depth != 1U) {
		append_uleb128(output, source.column_offset());
		append_uleb128(output, source.total_size());
	}
	append_uleb128(output, source.n_null());
	if (const auto* maximum = source.max(); maximum != nullptr) {
		if (const auto* bytes = maximum->binary_data(); bytes != nullptr) {
			output.insert(output.end(), bytes->begin(), bytes->end());
		}
	}
	if (const auto* expressions = source.expr_space(); expressions != nullptr) {
		for (const auto* expression : *expressions) {
			if (expression == nullptr) {
				fail("column descriptor contains a null expression result");
			}
			if (!omit_expression_sizes) {
				append_uleb128(output, expression->size());
			}
		}
	}
	if (const auto* segments = source.segment_descriptors(); segments != nullptr) {
		uint64_t dense_offset = source.column_offset();
		for (const auto* segment : *segments) {
			if (segment == nullptr) {
				fail("column descriptor contains a null segment descriptor");
			}
			if (!dense_segment_geometry) {
				append_uleb128(output, segment->entrypoint_offset());
				append_uleb128(output, segment->entrypoint_size());
				append_uleb128(output, segment->data_offset());
				append_uleb128(output, segment->data_size());
				continue;
			}
			const auto entrypoint_width = fastlanes::sizeof_entry_point_type(segment->entry_point_t());
			if (entrypoint_width == 0U) {
				fail("column segment uses an invalid entry-point type");
			}
			const auto entrypoint_size =
			    checked_product(rowgroup_vectors, entrypoint_width, "segment entry-point size");
			if (segment->entrypoint_offset() != dense_offset || segment->entrypoint_size() != entrypoint_size ||
			    entrypoint_size > std::numeric_limits<uint64_t>::max() - dense_offset ||
			    segment->data_offset() != dense_offset + entrypoint_size ||
			    segment->data_size() > std::numeric_limits<uint64_t>::max() - segment->data_offset()) {
				fail("column segments are not in the canonical dense one-vector layout");
			}
			append_uleb128(output, segment->data_size());
			dense_offset = segment->data_offset() + segment->data_size();
		}
		if (dense_segment_geometry && !segments->empty() &&
		    (source.total_size() > std::numeric_limits<uint64_t>::max() - source.column_offset() ||
		     dense_offset != source.column_offset() + source.total_size())) {
			fail("dense column segments do not exactly cover their column payload");
		}
	}
	if (const auto* children = source.children(); children != nullptr) {
		for (const auto* child : *children) {
			if (child == nullptr) {
				fail("column descriptor contains a null child descriptor");
			}
			append_column_variables(
			    output, *child, depth + 1U, rowgroup_vectors, omit_expression_sizes, dense_segment_geometry);
		}
	}
}

void apply_column_variables(fastlanes::ColumnDescriptorT& target,
                            const uint8_t* const          page,
                            const size_t                  page_size,
                            size_t&                       cursor,
                            const size_t                  depth,
                            const uint64_t                rowgroup_vectors,
                            const bool                    omit_expression_sizes,
                            const bool                    dense_segment_geometry) {
	if (depth > kMaxColumnDepth) {
		fail("column descriptor nesting exceeds the safety limit");
	}
	if (depth != 1U) {
		target.column_offset = consume_uleb128(page, page_size, cursor, "column offset");
		target.total_size    = consume_uleb128(page, page_size, cursor, "column size");
	}
	target.n_null = consume_uleb128(page, page_size, cursor, "null count");
	if (target.max != nullptr) {
		const auto byte_count = target.max->binary_data.size();
		if (cursor > page_size || byte_count > page_size - cursor) {
			fail("truncated rowgroup page while reading column maximum");
		}
		std::copy_n(page + cursor, byte_count, target.max->binary_data.begin());
		cursor += byte_count;
	}
	for (auto& expression : target.expr_space) {
		if (expression == nullptr) {
			fail("schema contains a null expression result");
		}
		if (!omit_expression_sizes) {
			expression->size = consume_uleb128(page, page_size, cursor, "expression size");
		}
	}
	uint64_t dense_offset = target.column_offset;
	for (auto& segment : target.segment_descriptors) {
		if (segment == nullptr) {
			fail("schema contains a null segment descriptor");
		}
		if (!dense_segment_geometry) {
			segment->entrypoint_offset = consume_uleb128(page, page_size, cursor, "entrypoint offset");
			segment->entrypoint_size   = consume_uleb128(page, page_size, cursor, "entrypoint size");
			segment->data_offset       = consume_uleb128(page, page_size, cursor, "segment data offset");
			segment->data_size         = consume_uleb128(page, page_size, cursor, "segment data size");
			continue;
		}
		const auto entrypoint_width = fastlanes::sizeof_entry_point_type(segment->entry_point_t);
		if (entrypoint_width == 0U) {
			fail("schema segment uses an invalid entry-point type");
		}
		const auto entrypoint_size = checked_product(rowgroup_vectors, entrypoint_width, "segment entry-point size");
		if (entrypoint_size > std::numeric_limits<uint64_t>::max() - dense_offset) {
			fail("dense segment entry-point geometry overflows uint64");
		}
		segment->entrypoint_offset = dense_offset;
		segment->entrypoint_size   = entrypoint_size;
		segment->data_offset       = dense_offset + entrypoint_size;
		segment->data_size         = consume_uleb128(page, page_size, cursor, "segment data size");
		if (segment->data_size > std::numeric_limits<uint64_t>::max() - segment->data_offset) {
			fail("dense segment data geometry overflows uint64");
		}
		dense_offset = segment->data_offset + segment->data_size;
	}
	if (dense_segment_geometry && !target.segment_descriptors.empty() &&
	    (target.total_size > std::numeric_limits<uint64_t>::max() - target.column_offset ||
	     dense_offset != target.column_offset + target.total_size)) {
		fail("reconstructed dense segments do not exactly cover their column payload");
	}
	for (auto& child : target.children) {
		if (child == nullptr) {
			fail("schema contains a null child descriptor");
		}
		apply_column_variables(*child,
		                       page,
		                       page_size,
		                       cursor,
		                       depth + 1U,
		                       rowgroup_vectors,
		                       omit_expression_sizes,
		                       dense_segment_geometry);
	}
}

void validate_metadata_only_column_geometry(const fastlanes::ColumnDescriptorT& column,
	                                         const size_t                         depth,
	                                         const size_t                         rowgroup_index,
	                                         const size_t                         column_index) {
	if (depth > kMaxColumnDepth) {
		fail("metadata-only rowgroup " + std::to_string(rowgroup_index) + " column " +
		     std::to_string(column_index) + " exceeds the column nesting safety limit");
	}
	if (column.column_offset != 0U || column.total_size != 0U || column.n_null != 0U) {
		fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
		     std::to_string(column_index) + " declares storage or null-state outside constant metadata");
	}
	for (const auto& segment : column.segment_descriptors) {
		if (segment == nullptr || segment->entrypoint_offset != 0U || segment->entrypoint_size != 0U ||
		    segment->data_offset != 0U || segment->data_size != 0U) {
			fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
			     std::to_string(column_index) + " declares physical segment storage");
		}
	}
	for (const auto& child : column.children) {
		if (child == nullptr) {
			fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
			     std::to_string(column_index) + " contains a null child descriptor");
		}
		validate_metadata_only_column_geometry(*child, depth + 1U, rowgroup_index, column_index);
	}
}

void validate_metadata_only_rowgroup(const fastlanes::RowgroupDescriptorT& rowgroup,
	                                  const size_t                          rowgroup_index) {
	if (rowgroup.m_size != 0U || rowgroup.m_n_vec == 0U || rowgroup.m_n_tuples == 0U ||
	    rowgroup.m_column_descriptors.empty()) {
		fail("invalid metadata-only rowgroup geometry at rowgroup " + std::to_string(rowgroup_index));
	}

	const size_t         column_count = rowgroup.m_column_descriptors.size();
	std::vector<uint8_t> visit_state(column_count, 0U);
	const auto validate_column = [&](auto&& self, const size_t column_index) -> void {
		if (column_index >= column_count) {
			fail("zero-payload rowgroup " + std::to_string(rowgroup_index) +
			     " references a column outside its schema");
		}
		if (visit_state[column_index] == 2U) {
			return;
		}
		if (visit_state[column_index] == 1U) {
			fail("zero-payload rowgroup " + std::to_string(rowgroup_index) +
			     " contains a metadata dependency cycle");
		}
		visit_state[column_index] = 1U;
		const auto& column = rowgroup.m_column_descriptors[column_index];
		if (column == nullptr) {
			fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " contains a null column descriptor");
		}
		validate_metadata_only_column_geometry(*column, 1U, rowgroup_index, column_index);
		if (column->encoding_rpn == nullptr || column->encoding_rpn->operator_tokens.size() != 1U) {
			fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
			     std::to_string(column_index) + " does not have one metadata-only encoding operator");
		}

		const auto token = column->encoding_rpn->operator_tokens.front();
		switch (token) {
		case fastlanes::OperatorToken::EXP_CONSTANT_I08:
			if (!column->encoding_rpn->operand_tokens.empty() || column->max == nullptr ||
			    column->max->binary_data.size() != sizeof(int8_t)) {
				fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
				     std::to_string(column_index) + " has invalid I8 constant metadata");
			}
			break;
		case fastlanes::OperatorToken::EXP_CONSTANT_I16:
			if (!column->encoding_rpn->operand_tokens.empty() || column->max == nullptr ||
			    column->max->binary_data.size() != sizeof(int16_t)) {
				fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
				     std::to_string(column_index) + " has invalid I16 constant metadata");
			}
			break;
		case fastlanes::OperatorToken::EXP_EQUAL: {
			if (column->encoding_rpn->operand_tokens.size() != 1U ||
			    column->encoding_rpn->operand_tokens.front() >= column_count) {
				fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
				     std::to_string(column_index) + " has an invalid equality dependency");
			}
			self(self, static_cast<size_t>(column->encoding_rpn->operand_tokens.front()));
			break;
		}
		default:
			fail("zero-payload rowgroup " + std::to_string(rowgroup_index) + " column " +
			     std::to_string(column_index) + " uses payload-dependent operator " +
			     fastlanes::token_to_string(token));
		}
		visit_state[column_index] = 2U;
	};

	for (size_t column_index = 0U; column_index < column_count; ++column_index) {
		validate_column(validate_column, column_index);
	}
}

Section append_section(std::vector<uint8_t>& descriptor, const std::vector<uint8_t>& section) {
	while ((descriptor.size() & 7U) != 0U) {
		descriptor.push_back(0U);
	}
	Section result {descriptor.size(), section.size()};
	descriptor.insert(descriptor.end(), section.begin(), section.end());
	return result;
}

std::vector<uint8_t> make_schema_section(const std::vector<std::vector<uint8_t>>& schemas) {
	const uint64_t       directory_bytes = checked_product(schemas.size() + 1U, sizeof(uint64_t), "schema directory");
	std::vector<uint8_t> output(checked_size(directory_bytes, "schema directory"), 0U);
	uint64_t             cursor = directory_bytes;
	for (size_t index = 0U; index < schemas.size(); ++index) {
		put_le<uint64_t>(output, index * sizeof(uint64_t), cursor);
		output.insert(output.end(), schemas[index].begin(), schemas[index].end());
		cursor += schemas[index].size();
	}
	put_le<uint64_t>(output, schemas.size() * sizeof(uint64_t), cursor);
	return output;
}

void copy_prefix(const std::filesystem::path& input_path, std::ofstream& output, uint64_t byte_count) {
	constexpr size_t kCopyBytes = 8U * 1024U * 1024U;
	std::ifstream    input(input_path, std::ios::binary);
	if (!input) {
		fail("cannot open input file for payload copy: " + input_path.string());
	}
	std::vector<char> buffer(kCopyBytes);
	while (byte_count != 0U) {
		const auto chunk = static_cast<std::streamsize>(std::min<uint64_t>(byte_count, buffer.size()));
		input.read(buffer.data(), chunk);
		if (input.gcount() != chunk) {
			fail("short read while copying the FLS header and payload");
		}
		output.write(buffer.data(), chunk);
		if (!output) {
			fail("failed to write the FLS header and payload");
		}
		byte_count -= static_cast<uint64_t>(chunk);
	}
}

std::filesystem::path staging_path_for(const std::filesystem::path& output_path) {
	const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
	return std::filesystem::path(output_path.string() + ".galp-stage-" + std::to_string(nonce));
}

class StagingCleanup {
public:
	explicit StagingCleanup(std::filesystem::path path)
	    : path_(std::move(path)) {
	}
	~StagingCleanup() {
		if (active_) {
			std::error_code ignored;
			std::filesystem::remove(path_, ignored);
		}
	}
	void release() noexcept {
		active_ = false;
	}

private:
	std::filesystem::path path_;
	bool                  active_ = true;
};

} // namespace

struct CompactDescriptorV3::Impl {
	~Impl() {
		if (mapping != MAP_FAILED) {
			::munmap(mapping, mapping_size);
			record_compact_unmapping(mapping_size);
		}
	}

	void*          mapping                  = MAP_FAILED;
	size_t         mapping_size             = 0U;
	const uint8_t* data                     = nullptr;
	size_t         size                     = 0U;
	uint64_t       payload_size             = 0U;
	uint64_t       payload_hash             = 0U;
	uint64_t       rowgroups                = 0U;
	uint32_t       columns                  = 0U;
	uint32_t       vector_rows              = 0U;
	uint32_t       order                    = 0U;
	uint32_t       schemas                  = 0U;
	uint32_t       images                   = 0U;
	uint32_t       components               = 0U;
	bool           omit_expression_sizes    = false;
	bool           dense_segment_geometry   = false;
	bool           dense_coefficient_ranges = false;
	Section        schema_section;
	Section        image_section;
	Section        component_section;
	Section        rowgroup_section;
	Section        coefficient_section;
	Section        page_section;

	[[nodiscard]] const uint8_t* section_data(const Section& section) const {
		return data + checked_size(section.offset, "section offset");
	}

	[[nodiscard]] std::pair<const uint8_t*, size_t> schema_bytes(const size_t schema_index) const {
		if (schema_index >= schemas) {
			throw std::out_of_range("CompactDescriptorV3 schema index out of range");
		}
		const auto* section = section_data(schema_section);
		const auto  begin   = read_le<uint64_t>(section,
                                             checked_size(schema_section.size, "schema section size"),
                                             schema_index * sizeof(uint64_t),
                                             "schema offset");
		const auto  end     = read_le<uint64_t>(section,
                                           checked_size(schema_section.size, "schema section size"),
                                           (schema_index + 1U) * sizeof(uint64_t),
                                           "schema end offset");
		if (begin > end || end > schema_section.size) {
			fail("schema blob exceeds its section");
		}
		return {section + checked_size(begin, "schema offset"), checked_size(end - begin, "schema size")};
	}

	[[nodiscard]] CompactV3RowgroupRecord rowgroup_record(const size_t rowgroup_index) const {
		if (rowgroup_index >= rowgroups) {
			throw std::out_of_range("CompactDescriptorV3 rowgroup index out of range");
		}
		const auto*             record = section_data(rowgroup_section) + rowgroup_index * kRowgroupRecordBytes;
		CompactV3RowgroupRecord result;
		result.payload_offset    = read_le<uint64_t>(record, kRowgroupRecordBytes, 0U, "rowgroup payload offset");
		result.payload_size      = read_le<uint32_t>(record, kRowgroupRecordBytes, 8U, "rowgroup payload size");
		result.real_row_count    = read_le<uint32_t>(record, kRowgroupRecordBytes, 12U, "rowgroup row count");
		result.local_image_index = read_le<uint32_t>(record, kRowgroupRecordBytes, 28U, "rowgroup image index");
		result.image_local_vector_index = read_le<uint32_t>(record, kRowgroupRecordBytes, 32U, "rowgroup vector index");
		result.payload_crc64 = read_le<uint64_t>(record, kRowgroupRecordBytes, 40U, "rowgroup payload CRC64");
		return result;
	}

	[[nodiscard]] std::pair<const uint8_t*, size_t> page_bytes(const size_t rowgroup_index) const {
		const auto* record = section_data(rowgroup_section) + rowgroup_index * kRowgroupRecordBytes;
		const auto  offset = read_le<uint64_t>(record, kRowgroupRecordBytes, 16U, "rowgroup page offset");
		const auto  length = read_le<uint32_t>(record, kRowgroupRecordBytes, 24U, "rowgroup page size");
		if (offset > page_section.size || length > page_section.size - offset) {
			fail("rowgroup page exceeds the page section");
		}
		return {section_data(page_section) + checked_size(offset, "rowgroup page offset"), length};
	}
};

CompactDescriptorV3::CompactDescriptorV3(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}
CompactDescriptorV3::CompactDescriptorV3(CompactDescriptorV3&&) noexcept            = default;
CompactDescriptorV3& CompactDescriptorV3::operator=(CompactDescriptorV3&&) noexcept = default;
CompactDescriptorV3::~CompactDescriptorV3()                                         = default;

CompactDescriptorV3MappingStats compact_descriptor_v3_mapping_stats() noexcept {
	return {g_compact_mapping_count.load(std::memory_order_relaxed),
	        g_compact_mapping_peak.load(std::memory_order_relaxed),
	        g_compact_map_count.load(std::memory_order_relaxed),
	        g_compact_unmap_count.load(std::memory_order_relaxed),
	        g_compact_mapped_bytes.load(std::memory_order_relaxed),
	        g_compact_mapped_bytes_peak.load(std::memory_order_relaxed)};
}

CompactDescriptorV3 CompactDescriptorV3::Open(const std::filesystem::path& shard_path) {
	fastlanes::File       file(shard_path);
	fastlanes::FileFooter footer {};
	require_status(fastlanes::FileFooter::Load(footer, file), "read FLS footer");
	const uint64_t file_size = file.Size();
	if (file_size < sizeof(fastlanes::FileFooter) ||
	    footer.table_descriptor_offset > file_size - sizeof(fastlanes::FileFooter) ||
	    footer.table_descriptor_size > file_size - sizeof(fastlanes::FileFooter) - footer.table_descriptor_offset ||
	    footer.table_descriptor_offset + footer.table_descriptor_size + sizeof(fastlanes::FileFooter) != file_size) {
		fail("compact descriptor footer range is invalid");
	}
	if (footer.table_descriptor_size < kHeaderBytes) {
		fail("descriptor is too small for a CompactDescriptorV3 header");
	}

	const int descriptor_fd = ::open(shard_path.c_str(), O_RDONLY | O_CLOEXEC);
	if (descriptor_fd < 0) {
		fail("cannot open compact shard for mmap: " + shard_path.string());
	}
	const long page_size_long = ::sysconf(_SC_PAGE_SIZE);
	if (page_size_long <= 0) {
		::close(descriptor_fd);
		fail("cannot determine the host page size");
	}
	const uint64_t page_size  = static_cast<uint64_t>(page_size_long);
	const uint64_t map_offset = footer.table_descriptor_offset - footer.table_descriptor_offset % page_size;
	const uint64_t delta      = footer.table_descriptor_offset - map_offset;
	if (footer.table_descriptor_size > std::numeric_limits<uint64_t>::max() - delta ||
	    map_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
		::close(descriptor_fd);
		fail("compact descriptor mmap geometry exceeds the host file-offset range");
	}
	const uint64_t map_bytes = delta + footer.table_descriptor_size;
	if (map_bytes > std::numeric_limits<size_t>::max()) {
		::close(descriptor_fd);
		fail("compact descriptor mmap exceeds addressable memory");
	}
	void* mapping = ::mmap(
	    nullptr, static_cast<size_t>(map_bytes), PROT_READ, MAP_PRIVATE, descriptor_fd, static_cast<off_t>(map_offset));
	::close(descriptor_fd);
	if (mapping == MAP_FAILED) {
		fail("mmap failed for compact descriptor: " + shard_path.string());
	}

	auto impl          = std::make_unique<Impl>();
	impl->mapping      = mapping;
	impl->mapping_size = static_cast<size_t>(map_bytes);
	record_compact_mapping(impl->mapping_size);
	impl->data         = static_cast<const uint8_t*>(mapping) + static_cast<size_t>(delta);
	impl->size         = checked_size(footer.table_descriptor_size, "descriptor size");
	const auto* data   = impl->data;
	if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
		fail("descriptor magic mismatch");
	}
	if (read_le<uint16_t>(data, impl->size, 8U, "format version") != kFormatVersion ||
	    read_le<uint16_t>(data, impl->size, 10U, "header size") != kHeaderBytes) {
		fail("unsupported descriptor version or header size");
	}
	const uint32_t flags = read_le<uint32_t>(data, impl->size, 12U, "feature flags");
	if ((flags & kRequiredFlags) != kRequiredFlags || (flags & ~kKnownFlags) != 0U) {
		fail("unsupported descriptor feature flags");
	}
	impl->omit_expression_sizes    = (flags & kFlagOmitExpressionSizes) != 0U;
	impl->dense_segment_geometry   = (flags & kFlagDenseSegmentGeometry) != 0U;
	impl->dense_coefficient_ranges = (flags & kFlagDenseCoefficientRanges) != 0U;
	if (read_le<uint64_t>(data, impl->size, 16U, "descriptor size") != impl->size) {
		fail("descriptor header size disagrees with the FLS footer");
	}
	impl->payload_size = read_le<uint64_t>(data, impl->size, 24U, "payload size");
	impl->payload_hash = read_le<uint64_t>(data, impl->size, 32U, "payload CRC64");
	impl->rowgroups    = read_le<uint64_t>(data, impl->size, 40U, "rowgroup count");
	impl->columns      = read_le<uint32_t>(data, impl->size, 48U, "column count");
	impl->vector_rows  = read_le<uint32_t>(data, impl->size, 52U, "vector size");
	impl->order        = read_le<uint32_t>(data, impl->size, 56U, "spatial order");
	impl->schemas      = read_le<uint32_t>(data, impl->size, 60U, "schema count");
	impl->images       = read_le<uint32_t>(data, impl->size, 64U, "image count");
	impl->components   = read_le<uint32_t>(data, impl->size, 68U, "component count");
	if (impl->rowgroups > std::numeric_limits<size_t>::max()) {
		fail("rowgroup directory exceeds addressable memory");
	}
	impl->schema_section             = {read_le<uint64_t>(data, impl->size, 72U, "schema section offset"),
	                                    read_le<uint64_t>(data, impl->size, 80U, "schema section size")};
	impl->image_section              = {read_le<uint64_t>(data, impl->size, 88U, "image section offset"),
	                                    read_le<uint64_t>(data, impl->size, 96U, "image section size")};
	impl->component_section          = {read_le<uint64_t>(data, impl->size, 104U, "component section offset"),
	                                    read_le<uint64_t>(data, impl->size, 112U, "component section size")};
	impl->rowgroup_section           = {read_le<uint64_t>(data, impl->size, 120U, "rowgroup section offset"),
	                                    read_le<uint64_t>(data, impl->size, 128U, "rowgroup section size")};
	impl->coefficient_section        = {read_le<uint64_t>(data, impl->size, 136U, "coefficient section offset"),
	                                    read_le<uint64_t>(data, impl->size, 144U, "coefficient section size")};
	impl->page_section               = {read_le<uint64_t>(data, impl->size, 152U, "page section offset"),
	                                    read_le<uint64_t>(data, impl->size, 160U, "page section size")};
	const uint64_t expected_checksum = read_le<uint64_t>(data, impl->size, kDescriptorChecksumByte, "descriptor CRC64");
	if (descriptor_crc64(data, impl->size) != expected_checksum) {
		fail("descriptor CRC64 mismatch");
	}
	if (impl->payload_size > std::numeric_limits<uint64_t>::max() - sizeof(fastlanes::FileHeader) ||
	    impl->payload_size + sizeof(fastlanes::FileHeader) != footer.table_descriptor_offset) {
		fail("payload size disagrees with the descriptor boundary");
	}
	if (impl->rowgroups == 0U || impl->columns == 0U || impl->vector_rows == 0U || impl->schemas == 0U) {
		fail("descriptor declares an empty required dimension");
	}

	const auto validate_section = [&](const Section& section, const std::string_view label) {
		if (section.offset < kHeaderBytes || section.offset > impl->size ||
		    section.size > impl->size - section.offset) {
			fail(std::string(label) + " exceeds descriptor bounds");
		}
	};
	validate_section(impl->schema_section, "schema section");
	validate_section(impl->image_section, "image section");
	validate_section(impl->component_section, "component section");
	validate_section(impl->rowgroup_section, "rowgroup section");
	validate_section(impl->coefficient_section, "coefficient section");
	validate_section(impl->page_section, "rowgroup page section");
	uint64_t canonical_section_end = kHeaderBytes;
	for (const auto& section : {impl->schema_section,
	                            impl->image_section,
	                            impl->component_section,
	                            impl->rowgroup_section,
	                            impl->coefficient_section,
	                            impl->page_section}) {
		if (canonical_section_end > std::numeric_limits<uint64_t>::max() - 7U) {
			fail("descriptor section alignment overflows uint64");
		}
		const auto aligned_end = (canonical_section_end + 7U) & ~uint64_t {7U};
		if (section.offset != aligned_end) {
			fail("descriptor sections are not in canonical dense order");
		}
		canonical_section_end = section.offset + section.size;
	}
	if (canonical_section_end != impl->size) {
		fail("descriptor has unclassified trailing bytes");
	}
	if (impl->schema_section.size <
	        checked_product(static_cast<uint64_t>(impl->schemas) + 1U, sizeof(uint64_t), "schema directory") ||
	    impl->image_section.size != checked_product(impl->images, kImageRecordBytes, "image directory") ||
	    impl->component_section.size !=
	        checked_product(impl->components, kComponentRecordBytes, "component directory") ||
	    impl->rowgroup_section.size != checked_product(impl->rowgroups, kRowgroupRecordBytes, "rowgroup directory") ||
	    impl->coefficient_section.size !=
	        checked_product(checked_product(impl->rowgroups, impl->columns, "coefficient record count"),
	                        impl->dense_coefficient_ranges ? sizeof(uint32_t) : kCoefficientRecordBytes,
	                        "coefficient directory")) {
		fail("a fixed-width directory has an invalid size");
	}

	uint64_t expected_schema_offset =
	    checked_product(static_cast<uint64_t>(impl->schemas) + 1U, sizeof(uint64_t), "schema directory");
	for (size_t schema_index = 0U; schema_index < impl->schemas; ++schema_index) {
		const auto [schema_data, schema_size] = impl->schema_bytes(schema_index);
		const auto* schema_section_data       = impl->section_data(impl->schema_section);
		const auto  schema_begin              = read_le<uint64_t>(schema_section_data,
                                                    checked_size(impl->schema_section.size, "schema section size"),
                                                    schema_index * sizeof(uint64_t),
                                                    "schema offset");
		const auto  schema_end                = read_le<uint64_t>(schema_section_data,
                                                  checked_size(impl->schema_section.size, "schema section size"),
                                                  (schema_index + 1U) * sizeof(uint64_t),
                                                  "schema end offset");
		if (schema_begin != expected_schema_offset || schema_size == 0U) {
			fail("schema dictionary is not a dense non-empty partition");
		}
		expected_schema_offset = schema_end;
		flatbuffers::Verifier verifier(schema_data, schema_size);
		if (!fastlanes::VerifyColumnDescriptorBuffer(verifier)) {
			fail("schema dictionary contains an invalid ColumnDescriptor FlatBuffer");
		}
	}
	if (expected_schema_offset != impl->schema_section.size) {
		fail("schema dictionary does not exactly cover its section");
	}
	uint64_t expected_payload_offset = sizeof(fastlanes::FileHeader);
	uint64_t expected_page_offset    = 0U;
	for (size_t rowgroup_index = 0U; rowgroup_index < impl->rowgroups; ++rowgroup_index) {
		const auto record = impl->rowgroup_record(rowgroup_index);
		if (record.real_row_count == 0U || record.real_row_count > impl->vector_rows ||
		    record.payload_offset != expected_payload_offset ||
		    record.payload_offset > footer.table_descriptor_offset ||
		    record.payload_size > footer.table_descriptor_offset - record.payload_offset ||
		    (record.payload_size == 0U && record.payload_crc64 != 0U)) {
			fail("rowgroup directory contains invalid payload geometry");
		}
		uint64_t coefficient_end = 0U;
		for (size_t coefficient = 0U; coefficient < impl->columns; ++coefficient) {
			const auto  record_bytes = impl->dense_coefficient_ranges ? sizeof(uint32_t) : kCoefficientRecordBytes;
			const auto  record_index = rowgroup_index * impl->columns + coefficient;
			const auto* range        = impl->section_data(impl->coefficient_section) + record_index * record_bytes;
			const auto  offset       = impl->dense_coefficient_ranges
			                               ? coefficient_end
			                               : read_le<uint32_t>(range, record_bytes, 0U, "coefficient offset");
			const auto  size         = read_le<uint32_t>(
                range, record_bytes, impl->dense_coefficient_ranges ? 0U : sizeof(uint32_t), "coefficient size");
			if (offset != coefficient_end || size > record.payload_size - coefficient_end) {
				fail("coefficient directory is not a dense exact rowgroup partition");
			}
			coefficient_end += size;
		}
		if (coefficient_end != record.payload_size) {
			fail("coefficient directory does not exactly cover its rowgroup payload");
		}
		expected_payload_offset += record.payload_size;
		const auto* rowgroup_record =
		    impl->section_data(impl->rowgroup_section) + rowgroup_index * kRowgroupRecordBytes;
		const auto page_offset = read_le<uint64_t>(rowgroup_record, kRowgroupRecordBytes, 16U, "rowgroup page offset");
		const auto rowgroup_page_size =
		    read_le<uint32_t>(rowgroup_record, kRowgroupRecordBytes, 24U, "rowgroup page size");
		if (page_offset != expected_page_offset || rowgroup_page_size == 0U ||
		    rowgroup_page_size > impl->page_section.size - page_offset) {
			fail("rowgroup pages are not a dense non-empty partition");
		}
		expected_page_offset += rowgroup_page_size;
	}
	if (expected_payload_offset != footer.table_descriptor_offset) {
		fail("rowgroup directory does not exactly cover the compressed payload");
	}
	if (expected_page_offset != impl->page_section.size) {
		fail("rowgroup pages do not exactly cover their section");
	}
	uint64_t expected_first_rowgroup     = 0U;
	uint64_t expected_first_component    = 0U;
	uint64_t expected_first_physical_row = 0U;
	for (size_t image_index = 0U; image_index < impl->images; ++image_index) {
		const auto* record             = impl->section_data(impl->image_section) + image_index * kImageRecordBytes;
		const auto  first_rowgroup     = read_le<uint32_t>(record, kImageRecordBytes, 0U, "image first rowgroup");
		const auto  rowgroup_count     = read_le<uint32_t>(record, kImageRecordBytes, 4U, "image rowgroup count");
		const auto  real_row_count     = read_le<uint32_t>(record, kImageRecordBytes, 8U, "image row count");
		const auto  first_component    = read_le<uint32_t>(record, kImageRecordBytes, 12U, "image first component");
		const auto  component_count    = read_le<uint16_t>(record, kImageRecordBytes, 16U, "image component count");
		const auto  spatial_order      = read_le<uint32_t>(record, kImageRecordBytes, 20U, "image spatial order");
		const auto  first_physical_row = read_le<uint64_t>(record, kImageRecordBytes, 24U, "image first physical row");
		if (first_rowgroup != expected_first_rowgroup || first_rowgroup > impl->rowgroups || rowgroup_count == 0U ||
		    real_row_count == 0U || rowgroup_count > impl->rowgroups - first_rowgroup ||
		    first_component != expected_first_component || first_component > impl->components ||
		    component_count == 0U || component_count > impl->components - first_component ||
		    spatial_order != impl->order || first_physical_row != expected_first_physical_row) {
			fail("image directory is not a dense valid partition");
		}
		expected_first_rowgroup += rowgroup_count;
		expected_first_component += component_count;
		if (real_row_count > std::numeric_limits<uint64_t>::max() - expected_first_physical_row) {
			fail("image physical-row directory overflows uint64");
		}
		expected_first_physical_row += real_row_count;
		uint64_t image_row_count = 0U;
		for (uint32_t local_vector = 0U; local_vector < rowgroup_count; ++local_vector) {
			const auto rowgroup = impl->rowgroup_record(first_rowgroup + local_vector);
			if (rowgroup.local_image_index != image_index || rowgroup.image_local_vector_index != local_vector) {
				fail("rowgroup image mapping disagrees with the image directory");
			}
			if (image_row_count > real_row_count || rowgroup.real_row_count > real_row_count - image_row_count) {
				fail("image rowgroups exceed the image row count");
			}
			image_row_count += rowgroup.real_row_count;
		}
		if (image_row_count != real_row_count) {
			fail("image row count disagrees with its vector rowgroups");
		}
		uint64_t component_row_count = 0U;
		for (uint32_t local_component = 0U; local_component < component_count; ++local_component) {
			const auto  component_index = first_component + local_component;
			const auto* component       = impl->section_data(impl->component_section) +
			                        static_cast<size_t>(component_index) * kComponentRecordBytes;
			const auto width  = read_le<uint32_t>(component, kComponentRecordBytes, 4U, "component width");
			const auto height = read_le<uint32_t>(component, kComponentRecordBytes, 8U, "component height");
			const auto padded_width =
			    read_le<uint32_t>(component, kComponentRecordBytes, 12U, "component padded width");
			const auto padded_height =
			    read_le<uint32_t>(component, kComponentRecordBytes, 16U, "component padded height");
			const auto row_offset = read_le<uint32_t>(component, kComponentRecordBytes, 20U, "component row offset");
			if (width == 0U || height == 0U || width > padded_width || height > padded_height ||
			    row_offset != component_row_count) {
				fail("component directory contains invalid block-grid geometry");
			}
			const auto component_rows = checked_product(width, height, "component block count");
			if (component_row_count > real_row_count || component_rows > real_row_count - component_row_count) {
				fail("component grids exceed the image row count");
			}
			component_row_count += component_rows;
		}
		if (component_row_count != real_row_count) {
			fail("component grids do not exactly cover their image");
		}
	}
	if (impl->images != 0U &&
	    (expected_first_rowgroup != impl->rowgroups || expected_first_component != impl->components)) {
		fail("image directory does not cover all rowgroups and components");
	}
	if (impl->images == 0U && impl->components != 0U) {
		fail("component directory requires an image directory");
	}
	CompactDescriptorV3 result(std::move(impl));
	for (size_t rowgroup_index = 0U; rowgroup_index < result.rowgroup_count(); ++rowgroup_index) {
		if (result.rowgroup(rowgroup_index).payload_size == 0U) {
			validate_metadata_only_rowgroup(*result.unpack_rowgroup(rowgroup_index), rowgroup_index);
		}
	}
	return result;
}

uint64_t CompactDescriptorV3::payload_bytes() const noexcept {
	return impl_->payload_size;
}
uint64_t CompactDescriptorV3::payload_crc64() const noexcept {
	return impl_->payload_hash;
}
uint32_t CompactDescriptorV3::vector_size() const noexcept {
	return impl_->vector_rows;
}
uint32_t CompactDescriptorV3::spatial_order() const noexcept {
	return impl_->order;
}
size_t CompactDescriptorV3::column_count() const noexcept {
	return impl_->columns;
}
size_t CompactDescriptorV3::rowgroup_count() const noexcept {
	return static_cast<size_t>(impl_->rowgroups);
}
size_t CompactDescriptorV3::image_count() const noexcept {
	return impl_->images;
}
size_t CompactDescriptorV3::schema_count() const noexcept {
	return impl_->schemas;
}
size_t CompactDescriptorV3::descriptor_bytes() const noexcept {
	return impl_->size;
}

void CompactDescriptorV3::release_resident_pages() const noexcept {
#if defined(MADV_DONTNEED)
	if (impl_ != nullptr && impl_->mapping != MAP_FAILED && impl_->mapping_size != 0U) {
		(void)::madvise(impl_->mapping, impl_->mapping_size, MADV_DONTNEED);
	}
#endif
}

CompactV3RowgroupRecord CompactDescriptorV3::rowgroup(const size_t rowgroup_index) const {
	return impl_->rowgroup_record(rowgroup_index);
}

CompactV3ImageRecord CompactDescriptorV3::image(const size_t image_index) const {
	if (image_index >= impl_->images) {
		throw std::out_of_range("CompactDescriptorV3 image index out of range");
	}
	const auto*          record = impl_->section_data(impl_->image_section) + image_index * kImageRecordBytes;
	CompactV3ImageRecord result;
	result.first_rowgroup     = read_le<uint32_t>(record, kImageRecordBytes, 0U, "image first rowgroup");
	result.rowgroup_count     = read_le<uint32_t>(record, kImageRecordBytes, 4U, "image rowgroup count");
	result.real_row_count     = read_le<uint32_t>(record, kImageRecordBytes, 8U, "image row count");
	result.first_component    = read_le<uint32_t>(record, kImageRecordBytes, 12U, "image first component");
	result.component_count    = read_le<uint16_t>(record, kImageRecordBytes, 16U, "image component count");
	result.spatial_order      = read_le<uint32_t>(record, kImageRecordBytes, 20U, "image spatial order");
	result.first_physical_row = read_le<uint64_t>(record, kImageRecordBytes, 24U, "image first physical row");
	return result;
}

CompactV3ComponentInput CompactDescriptorV3::component(const size_t component_index) const {
	if (component_index >= impl_->components) {
		throw std::out_of_range("CompactDescriptorV3 component index out of range");
	}
	const auto* record = impl_->section_data(impl_->component_section) + component_index * kComponentRecordBytes;
	CompactV3ComponentInput result;
	result.semantic_slot_id        = read_le<uint32_t>(record, kComponentRecordBytes, 0U, "component semantic slot");
	result.width_in_blocks         = read_le<uint32_t>(record, kComponentRecordBytes, 4U, "component width");
	result.height_in_blocks        = read_le<uint32_t>(record, kComponentRecordBytes, 8U, "component height");
	result.padded_width_in_blocks  = read_le<uint32_t>(record, kComponentRecordBytes, 12U, "component padded width");
	result.padded_height_in_blocks = read_le<uint32_t>(record, kComponentRecordBytes, 16U, "component padded height");
	result.row_offset              = read_le<uint32_t>(record, kComponentRecordBytes, 20U, "component row offset");
	result.component_index         = read_le<uint32_t>(record, kComponentRecordBytes, 24U, "component index");
	return result;
}

std::vector<CompactV3CoefficientRange> CompactDescriptorV3::coefficient_ranges(const size_t rowgroup_index) const {
	if (rowgroup_index >= impl_->rowgroups) {
		throw std::out_of_range("CompactDescriptorV3 coefficient rowgroup index out of range");
	}
	std::vector<CompactV3CoefficientRange> ranges;
	ranges.reserve(impl_->columns);
	uint64_t   dense_offset = 0U;
	const auto record_bytes = impl_->dense_coefficient_ranges ? sizeof(uint32_t) : kCoefficientRecordBytes;
	for (size_t coefficient_index = 0U; coefficient_index < impl_->columns; ++coefficient_index) {
		const auto  record_index = rowgroup_index * impl_->columns + coefficient_index;
		const auto* record       = impl_->section_data(impl_->coefficient_section) + record_index * record_bytes;
		const auto  offset       = impl_->dense_coefficient_ranges
		                               ? checked_u32(dense_offset, "dense coefficient offset")
		                               : read_le<uint32_t>(record, record_bytes, 0U, "coefficient offset");
		const auto  size         = read_le<uint32_t>(
            record, record_bytes, impl_->dense_coefficient_ranges ? 0U : sizeof(uint32_t), "coefficient size");
		ranges.push_back({offset, size});
		dense_offset += size;
	}
	return ranges;
}

CompactV3CoefficientRange CompactDescriptorV3::coefficient_range(const size_t rowgroup_index,
                                                                 const size_t coefficient_index) const {
	if (rowgroup_index >= impl_->rowgroups || coefficient_index >= impl_->columns) {
		throw std::out_of_range("CompactDescriptorV3 coefficient range index out of range");
	}
	if (impl_->dense_coefficient_ranges) {
		return coefficient_ranges(rowgroup_index)[coefficient_index];
	}
	const auto  record_index = rowgroup_index * impl_->columns + coefficient_index;
	const auto* record       = impl_->section_data(impl_->coefficient_section) + record_index * kCoefficientRecordBytes;
	return {read_le<uint32_t>(record, kCoefficientRecordBytes, 0U, "coefficient offset"),
	        read_le<uint32_t>(record, kCoefficientRecordBytes, 4U, "coefficient size")};
}

bool CompactDescriptorV3::supports_direct_rowgroup_geometry() const noexcept {
	return impl_->omit_expression_sizes && impl_->dense_segment_geometry && impl_->dense_coefficient_ranges;
}

CompactV3DirectRowgroup CompactDescriptorV3::decode_direct_rowgroup(const size_t rowgroup_index) const {
	if (!supports_direct_rowgroup_geometry()) {
		fail("direct rowgroup geometry requires canonical compact flags");
	}
	const auto record                            = impl_->rowgroup_record(rowgroup_index);
	const auto [encoded_page, encoded_page_size] = impl_->page_bytes(rowgroup_index);
	const auto decoded_page                      = decode_rowgroup_page(encoded_page, encoded_page_size);
	const auto* page                              = decoded_page.data();
	const auto  page_size                         = decoded_page.size();
	const auto  ranges                            = coefficient_ranges(rowgroup_index);

	CompactV3DirectRowgroup output;
	output.record = record;
	output.columns.reserve(impl_->columns);
	output.segments.reserve(impl_->columns * 4U);
	size_t cursor = 0U;
	for (size_t column_index = 0U; column_index < impl_->columns; ++column_index) {
		const auto schema_index = consume_uleb128(page, page_size, cursor, "column schema id");
		if (schema_index >= impl_->schemas) {
			fail("rowgroup page references an unknown column schema");
		}
		const auto [schema_data, schema_size] = impl_->schema_bytes(static_cast<size_t>(schema_index));
		(void)schema_size;
		const auto* schema = fastlanes::GetColumnDescriptor(schema_data);
		if (schema == nullptr) {
			fail("direct rowgroup column schema is missing");
		}
		if (schema->children() != nullptr && !schema->children()->empty()) {
			fail("canonical direct rowgroup geometry requires root-only columns");
		}

		const auto range = ranges[column_index];
		if (range.offset > record.payload_size || range.size > record.payload_size - range.offset) {
			fail("coefficient directory exceeds direct rowgroup geometry");
		}
		(void)consume_uleb128(page, page_size, cursor, "null count");

		CompactV3DirectColumn column;
		column.schema         = schema;
		column.segment_begin  = checked_u32(output.segments.size(), "direct segment begin");
		column.maximum_offset = checked_u32(output.maximum_bytes.size(), "direct maximum offset");
		if (const auto* maximum = schema->max(); maximum != nullptr) {
			if (const auto* bytes = maximum->binary_data(); bytes != nullptr) {
				const auto byte_count = static_cast<size_t>(bytes->size());
				if (cursor > page_size || byte_count > page_size - cursor) {
					fail("truncated rowgroup page while reading direct column maximum");
				}
				column.maximum_size = checked_u32(byte_count, "direct maximum size");
				output.maximum_bytes.insert(output.maximum_bytes.end(), page + cursor, page + cursor + byte_count);
				cursor += byte_count;
			}
		}

		uint64_t dense_offset = range.offset;
		if (const auto* segment_descriptors = schema->segment_descriptors(); segment_descriptors != nullptr) {
			for (const auto* segment : *segment_descriptors) {
				if (segment == nullptr) {
					fail("direct rowgroup schema contains a null segment descriptor");
				}
				const auto entrypoint_width = fastlanes::sizeof_entry_point_type(segment->entry_point_t());
				if (entrypoint_width == 0U) {
					fail("direct rowgroup segment uses an invalid entry-point type");
				}
				const uint64_t entrypoint_size = entrypoint_width; // Canonical v3 rowgroups contain one vector.
				if (entrypoint_size > std::numeric_limits<uint64_t>::max() - dense_offset) {
					fail("direct rowgroup segment entry-point geometry overflows uint64");
				}
				const uint64_t data_offset = dense_offset + entrypoint_size;
				const uint64_t data_size = consume_uleb128(page, page_size, cursor, "segment data size");
				if (data_size > std::numeric_limits<uint64_t>::max() - data_offset) {
					fail("direct rowgroup segment data geometry overflows uint64");
				}
				output.segments.push_back(CompactV3DirectSegment {
				    dense_offset, entrypoint_size, data_offset, data_size, segment->entry_point_t()});
				dense_offset = data_offset + data_size;
			}
		}
		if (dense_offset != static_cast<uint64_t>(range.offset) + range.size) {
			fail("direct rowgroup segments do not exactly cover their coefficient range");
		}
		column.segment_count =
		    checked_u32(output.segments.size() - column.segment_begin, "direct segment count");
		output.columns.push_back(column);
	}
	if (cursor != page_size) {
		fail("direct rowgroup page has trailing bytes");
	}
	return output;
}

std::unique_ptr<fastlanes::RowgroupDescriptorT>
CompactDescriptorV3::unpack_rowgroup(const size_t rowgroup_index) const {
	const auto record                            = impl_->rowgroup_record(rowgroup_index);
	const auto [encoded_page, encoded_page_size] = impl_->page_bytes(rowgroup_index);
	const auto  decoded_page                     = decode_rowgroup_page(encoded_page, encoded_page_size);
	const auto* page                             = decoded_page.data();
	const auto  page_size                        = decoded_page.size();
	auto        output                           = std::make_unique<fastlanes::RowgroupDescriptorT>();
	output->m_n_vec                              = 1U;
	output->m_size                               = record.payload_size;
	output->m_offset                             = record.payload_offset;
	output->m_n_tuples                           = record.real_row_count;
	output->m_column_descriptors.reserve(impl_->columns);
	const auto ranges = coefficient_ranges(rowgroup_index);
	size_t     cursor = 0U;
	for (size_t column_index = 0U; column_index < impl_->columns; ++column_index) {
		const auto schema_index = consume_uleb128(page, page_size, cursor, "column schema id");
		if (schema_index >= impl_->schemas) {
			fail("rowgroup page references an unknown column schema");
		}
		const auto [schema_data, schema_size]                = impl_->schema_bytes(static_cast<size_t>(schema_index));
		const auto*                                   schema = fastlanes::GetColumnDescriptor(schema_data);
		std::unique_ptr<fastlanes::ColumnDescriptorT> column(schema->UnPack());
		if (column == nullptr) {
			fail("failed to unpack a compact column schema");
		}
		const auto expected_range = ranges[column_index];
		if (expected_range.offset > record.payload_size ||
		    expected_range.size > record.payload_size - expected_range.offset) {
			fail("coefficient directory exceeds reconstructed rowgroup geometry");
		}
		column->column_offset = expected_range.offset;
		column->total_size    = expected_range.size;
		apply_column_variables(
		    *column, page, page_size, cursor, 1U, 1U, impl_->omit_expression_sizes, impl_->dense_segment_geometry);
		output->m_column_descriptors.push_back(std::move(column));
	}
	if (cursor != page_size) {
		fail("rowgroup page has trailing bytes");
	}
	return output;
}

bool is_compact_v3_fls(const std::filesystem::path& shard_path) {
	try {
		fastlanes::File       file(shard_path);
		fastlanes::FileFooter footer {};
		if (!fastlanes::FileFooter::Load(footer, file).success || footer.table_descriptor_size < kMagic.size()) {
			return false;
		}
		std::array<uint8_t, kMagic.size()> magic {};
		file.ReadRangeUnchecked(magic.data(), footer.table_descriptor_offset, magic.size());
		return magic == kMagic;
	} catch (const std::exception&) { return false; }
}

CompactV3Report compact_standard_fls_to_v3(const std::filesystem::path& input_path,
                                           const std::filesystem::path& output_path,
                                           const CompactV3BuildOptions& options) {
	if (std::filesystem::absolute(input_path) == std::filesystem::absolute(output_path)) {
		fail("compactor input and output paths must differ");
	}
	if (std::filesystem::exists(output_path)) {
		fail("compactor output already exists: " + output_path.string());
	}
	if (options.vector_size == 0U) {
		fail("vector size must be greater than zero");
	}
	fastlanes::File       input(input_path);
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	require_status(fastlanes::FileHeader::Load(header, input), "read FLS header");
	require_status(fastlanes::FileFooter::Load(footer, input), "read FLS footer");
	if (header.magic_bytes != fastlanes::Info::get_magic_bytes() ||
	    footer.magic_bytes != fastlanes::Info::get_magic_bytes()) {
		fail("input FLS header/footer magic mismatch");
	}
	if (!static_cast<bool>(header.settings.inline_footer)) {
		fail("the offline compactor currently requires an inline standard TableDescriptor");
	}
	const uint64_t input_size = input.Size();
	if (footer.table_descriptor_offset < sizeof(fastlanes::FileHeader) ||
	    footer.table_descriptor_offset > input_size - sizeof(fastlanes::FileFooter) ||
	    footer.table_descriptor_size > input_size - sizeof(fastlanes::FileFooter) - footer.table_descriptor_offset ||
	    footer.table_descriptor_offset + footer.table_descriptor_size + sizeof(fastlanes::FileFooter) != input_size) {
		fail("input TableDescriptor range is invalid");
	}
	if (is_compact_v3_fls(input_path)) {
		fail("input is already a Compact v3 shard");
	}
	const auto descriptor = fastlanes::TableDescriptorHandle::FromFileSlice(
	    input, footer.table_descriptor_offset, footer.table_descriptor_size, true);
	const auto* table = descriptor.Get();
	if (table == nullptr || table->m_rowgroup_descriptors() == nullptr || table->m_rowgroup_descriptors()->empty()) {
		fail("input TableDescriptor has no rowgroups");
	}
	if (table->m_table_binary_size() != footer.table_descriptor_offset) {
		fail("input TableDescriptor payload boundary disagrees with the FLS footer");
	}
	const auto* rowgroups = table->m_rowgroup_descriptors();
	const auto* first     = rowgroups->Get(0U);
	if (first == nullptr || first->m_column_descriptors() == nullptr || first->m_column_descriptors()->empty()) {
		fail("input first rowgroup has no columns");
	}
	const uint32_t column_count = first->m_column_descriptors()->size();

	std::vector<uint32_t> rowgroup_image(rowgroups->size(), std::numeric_limits<uint32_t>::max());
	std::vector<uint32_t> rowgroup_image_vector(rowgroups->size(), 0U);
	uint32_t              expected_first_rowgroup     = 0U;
	uint64_t              expected_first_physical_row = 0U;
	uint64_t              component_count             = 0U;
	for (size_t image_index = 0U; image_index < options.images.size(); ++image_index) {
		const auto& image = options.images[image_index];
		if (image.first_rowgroup != expected_first_rowgroup || image.rowgroup_count == 0U ||
		    image.rowgroup_count > rowgroups->size() - image.first_rowgroup || image.real_row_count == 0U ||
		    image.components.empty() || image.first_physical_row != expected_first_physical_row) {
			fail("image directory must densely and exactly partition shard rowgroups");
		}
		uint64_t component_rows = 0U;
		for (const auto& component : image.components) {
			if (component.width_in_blocks == 0U || component.height_in_blocks == 0U ||
			    component.width_in_blocks > component.padded_width_in_blocks ||
			    component.height_in_blocks > component.padded_height_in_blocks ||
			    component.row_offset != component_rows) {
				fail("image component directory has invalid block-grid geometry");
			}
			component_rows +=
			    checked_product(component.width_in_blocks, component.height_in_blocks, "component block count");
		}
		if (component_rows != image.real_row_count) {
			fail("image component grids do not exactly cover its real rows");
		}
		for (uint32_t vector = 0U; vector < image.rowgroup_count; ++vector) {
			rowgroup_image[image.first_rowgroup + vector]        = checked_u32(image_index, "local image index");
			rowgroup_image_vector[image.first_rowgroup + vector] = vector;
		}
		expected_first_rowgroup += image.rowgroup_count;
		expected_first_physical_row += image.real_row_count;
		component_count += image.components.size();
	}
	if (!options.images.empty() && expected_first_rowgroup != rowgroups->size()) {
		fail("image directory does not cover every rowgroup");
	}
	if (component_count > std::numeric_limits<uint32_t>::max()) {
		fail("component directory exceeds uint32 range");
	}

	std::unordered_map<std::string, uint32_t> schema_ids;
	std::vector<std::vector<uint8_t>>         schemas;
	std::vector<uint8_t>                      pages;
	std::vector<uint8_t>                      coefficients;
	std::vector<uint8_t>                      rowgroup_directory;
	rowgroup_directory.reserve(rowgroups->size() * kRowgroupRecordBytes);
	coefficients.reserve(rowgroups->size() * static_cast<size_t>(column_count) * sizeof(uint32_t));

	uint64_t expected_rowgroup_offset = sizeof(fastlanes::FileHeader);
	for (size_t rowgroup_index = 0U; rowgroup_index < rowgroups->size(); ++rowgroup_index) {
		const auto* rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
		if (rowgroup == nullptr || rowgroup->m_column_descriptors() == nullptr ||
		    rowgroup->m_column_descriptors()->size() != column_count) {
			fail("rowgroup schemas do not have a stable root-column count");
		}
		if (rowgroup->m_n_vec() != 1U) {
			fail("each Compact v3 rowgroup must contain exactly one FastLanes vector");
		}
			if (rowgroup->m_n_tuples() == 0U || rowgroup->m_n_tuples() > options.vector_size ||
		    rowgroup->m_offset() != expected_rowgroup_offset || rowgroup->m_offset() > footer.table_descriptor_offset ||
		    rowgroup->m_size() > footer.table_descriptor_offset - rowgroup->m_offset()) {
				fail("rowgroup payload geometry is invalid");
			}
			if (rowgroup->m_size() == 0U) {
				std::unique_ptr<fastlanes::RowgroupDescriptorT> native(rowgroup->UnPack());
				if (native == nullptr) {
					fail("failed to unpack metadata-only source rowgroup");
				}
				validate_metadata_only_rowgroup(*native, rowgroup_index);
			}
			std::vector<uint8_t> rowgroup_page;
		uint64_t             expected_column_offset = 0U;
		for (const auto* column : *rowgroup->m_column_descriptors()) {
			if (column == nullptr) {
				fail("rowgroup contains a null column descriptor");
			}
			auto        schema = normalized_column_schema(*column);
			std::string key(reinterpret_cast<const char*>(schema.data()), schema.size());
			auto [position, inserted] = schema_ids.emplace(std::move(key), checked_u32(schemas.size(), "schema id"));
			if (inserted) {
				schemas.push_back(std::move(schema));
			}
			append_uleb128(rowgroup_page, position->second);
			append_column_variables(rowgroup_page, *column, 1U, rowgroup->m_n_vec(), true, true);
			if (column->column_offset() > std::numeric_limits<uint32_t>::max() ||
			    column->total_size() > std::numeric_limits<uint32_t>::max() ||
			    column->column_offset() > rowgroup->m_size() ||
			    column->total_size() > rowgroup->m_size() - column->column_offset() ||
			    column->column_offset() != expected_column_offset) {
				fail("coefficient range exceeds its vector rowgroup");
			}
			append_le<uint32_t>(coefficients, static_cast<uint32_t>(column->total_size()));
			expected_column_offset += column->total_size();
		}
		if (expected_column_offset != rowgroup->m_size()) {
			fail("coefficient ranges do not exactly cover their vector rowgroup");
		}
		const auto     encoded_page = encode_rowgroup_page(rowgroup_page);
		const uint64_t page_offset  = pages.size();
		pages.insert(pages.end(), encoded_page.begin(), encoded_page.end());
		append_le<uint64_t>(rowgroup_directory, rowgroup->m_offset());
		append_le<uint32_t>(rowgroup_directory, checked_u32(rowgroup->m_size(), "rowgroup payload size"));
		append_le<uint32_t>(rowgroup_directory, checked_u32(rowgroup->m_n_tuples(), "rowgroup row count"));
		append_le<uint64_t>(rowgroup_directory, page_offset);
		append_le<uint32_t>(rowgroup_directory, checked_u32(encoded_page.size(), "rowgroup page size"));
		append_le<uint32_t>(rowgroup_directory, rowgroup_image[rowgroup_index]);
		append_le<uint32_t>(rowgroup_directory, rowgroup_image_vector[rowgroup_index]);
		append_le<uint32_t>(rowgroup_directory, 0U);
		append_le<uint64_t>(rowgroup_directory, crc64_file_range(input, rowgroup->m_offset(), rowgroup->m_size()));
		expected_rowgroup_offset += rowgroup->m_size();
	}
	if (expected_rowgroup_offset != footer.table_descriptor_offset) {
		fail("standard rowgroups do not exactly cover the compressed payload");
	}
	for (const auto& image : options.images) {
		uint64_t row_count = 0U;
		for (uint32_t vector = 0U; vector < image.rowgroup_count; ++vector) {
			const auto* rowgroup = rowgroups->Get(image.first_rowgroup + vector);
			row_count += rowgroup->m_n_tuples();
		}
		if (row_count != image.real_row_count) {
			fail("image real-row count disagrees with its standard rowgroups");
		}
	}

	std::vector<uint8_t> image_directory;
	std::vector<uint8_t> component_directory;
	image_directory.reserve(options.images.size() * kImageRecordBytes);
	component_directory.reserve(static_cast<size_t>(component_count) * kComponentRecordBytes);
	uint32_t first_component = 0U;
	for (const auto& image : options.images) {
		append_le<uint32_t>(image_directory, image.first_rowgroup);
		append_le<uint32_t>(image_directory, image.rowgroup_count);
		append_le<uint32_t>(image_directory, image.real_row_count);
		append_le<uint32_t>(image_directory, first_component);
		append_le<uint16_t>(image_directory, checked_u16(image.components.size(), "image component count"));
		append_le<uint16_t>(image_directory, 0U);
		append_le<uint32_t>(image_directory, options.spatial_order);
		append_le<uint64_t>(image_directory, image.first_physical_row);
		for (const auto& component : image.components) {
			append_le<uint32_t>(component_directory, component.semantic_slot_id);
			append_le<uint32_t>(component_directory, component.width_in_blocks);
			append_le<uint32_t>(component_directory, component.height_in_blocks);
			append_le<uint32_t>(component_directory, component.padded_width_in_blocks);
			append_le<uint32_t>(component_directory, component.padded_height_in_blocks);
			append_le<uint32_t>(component_directory, component.row_offset);
			append_le<uint32_t>(component_directory, component.component_index);
			append_le<uint32_t>(component_directory, 0U);
		}
		first_component += checked_u32(image.components.size(), "image component count");
	}

	const auto           schema_section_bytes = make_schema_section(schemas);
	std::vector<uint8_t> compact_descriptor(kHeaderBytes, 0U);
	const Section        schema_section      = append_section(compact_descriptor, schema_section_bytes);
	const Section        image_section       = append_section(compact_descriptor, image_directory);
	const Section        component_section   = append_section(compact_descriptor, component_directory);
	const Section        rowgroup_section    = append_section(compact_descriptor, rowgroup_directory);
	const Section        coefficient_section = append_section(compact_descriptor, coefficients);
	const Section        page_section        = append_section(compact_descriptor, pages);
	std::copy(kMagic.begin(), kMagic.end(), compact_descriptor.begin());
	put_le<uint16_t>(compact_descriptor, 8U, kFormatVersion);
	put_le<uint16_t>(compact_descriptor, 10U, kHeaderBytes);
	put_le<uint32_t>(compact_descriptor, 12U, kWriterFlags);
	put_le<uint64_t>(compact_descriptor, 16U, compact_descriptor.size());
	const uint64_t payload_size = footer.table_descriptor_offset - sizeof(fastlanes::FileHeader);
	const uint64_t payload_crc  = crc64_file_range(input, sizeof(fastlanes::FileHeader), payload_size);
	put_le<uint64_t>(compact_descriptor, 24U, payload_size);
	put_le<uint64_t>(compact_descriptor, 32U, payload_crc);
	put_le<uint64_t>(compact_descriptor, 40U, rowgroups->size());
	put_le<uint32_t>(compact_descriptor, 48U, column_count);
	put_le<uint32_t>(compact_descriptor, 52U, options.vector_size);
	put_le<uint32_t>(compact_descriptor, 56U, options.spatial_order);
	put_le<uint32_t>(compact_descriptor, 60U, checked_u32(schemas.size(), "schema count"));
	put_le<uint32_t>(compact_descriptor, 64U, checked_u32(options.images.size(), "image count"));
	put_le<uint32_t>(compact_descriptor, 68U, static_cast<uint32_t>(component_count));
	put_le<uint64_t>(compact_descriptor, 72U, schema_section.offset);
	put_le<uint64_t>(compact_descriptor, 80U, schema_section.size);
	put_le<uint64_t>(compact_descriptor, 88U, image_section.offset);
	put_le<uint64_t>(compact_descriptor, 96U, image_section.size);
	put_le<uint64_t>(compact_descriptor, 104U, component_section.offset);
	put_le<uint64_t>(compact_descriptor, 112U, component_section.size);
	put_le<uint64_t>(compact_descriptor, 120U, rowgroup_section.offset);
	put_le<uint64_t>(compact_descriptor, 128U, rowgroup_section.size);
	put_le<uint64_t>(compact_descriptor, 136U, coefficient_section.offset);
	put_le<uint64_t>(compact_descriptor, 144U, coefficient_section.size);
	put_le<uint64_t>(compact_descriptor, 152U, page_section.offset);
	put_le<uint64_t>(compact_descriptor, 160U, page_section.size);
	put_le<uint64_t>(compact_descriptor,
	                 kDescriptorChecksumByte,
	                 descriptor_crc64(compact_descriptor.data(), compact_descriptor.size()));

	if (!output_path.parent_path().empty()) {
		std::filesystem::create_directories(output_path.parent_path());
	}
	const auto     staged_path = staging_path_for(output_path);
	StagingCleanup cleanup(staged_path);
	std::ofstream  output(staged_path, std::ios::binary | std::ios::trunc);
	if (!output) {
		fail("cannot create staged compact shard: " + staged_path.string());
	}
	copy_prefix(input_path, output, footer.table_descriptor_offset);
	output.write(reinterpret_cast<const char*>(compact_descriptor.data()),
	             static_cast<std::streamsize>(compact_descriptor.size()));
	fastlanes::FileFooter compact_footer = footer;
	compact_footer.table_descriptor_size = compact_descriptor.size();
	output.write(reinterpret_cast<const char*>(&compact_footer), sizeof(compact_footer));
	output.close();
	if (!output) {
		fail("failed to finalize staged compact shard");
	}
	std::filesystem::rename(staged_path, output_path);
	cleanup.release();

	CompactV3Report report;
	report.source_file_bytes        = input_size;
	report.output_file_bytes        = std::filesystem::file_size(output_path);
	report.payload_bytes            = payload_size;
	report.source_descriptor_bytes  = footer.table_descriptor_size;
	report.compact_descriptor_bytes = compact_descriptor.size();
	report.payload_crc64            = payload_crc;
	report.rowgroup_count           = rowgroups->size();
	report.schema_count             = schemas.size();
	report.descriptor_reduction =
	    footer.table_descriptor_size == 0U
	        ? 0.0
	        : 1.0 - static_cast<double>(compact_descriptor.size()) / static_cast<double>(footer.table_descriptor_size);
	return report;
}

CompactV3Report expand_compact_fls_v3(const std::filesystem::path& input_path,
                                      const std::filesystem::path& output_path) {
	if (std::filesystem::absolute(input_path) == std::filesystem::absolute(output_path)) {
		fail("expander input and output paths must differ");
	}
	if (std::filesystem::exists(output_path)) {
		fail("expander output already exists: " + output_path.string());
	}
	fastlanes::File       input(input_path);
	fastlanes::FileFooter footer {};
	require_status(fastlanes::FileFooter::Load(footer, input), "read compact FLS footer");
	const uint64_t input_size = input.Size();
	auto           compact    = CompactDescriptorV3::Open(input_path);

	fastlanes::TableDescriptorT table;
	table.m_table_binary_size = footer.table_descriptor_offset;
	table.m_rowgroup_descriptors.reserve(compact.rowgroup_count());
	for (size_t rowgroup_index = 0U; rowgroup_index < compact.rowgroup_count(); ++rowgroup_index) {
		table.m_rowgroup_descriptors.push_back(compact.unpack_rowgroup(rowgroup_index));
	}
	const auto descriptor = fastlanes::TableDescriptorHandle::FromNative(table);
	if (!output_path.parent_path().empty()) {
		std::filesystem::create_directories(output_path.parent_path());
	}
	const auto     staged_path = staging_path_for(output_path);
	StagingCleanup cleanup(staged_path);
	std::ofstream  output(staged_path, std::ios::binary | std::ios::trunc);
	if (!output) {
		fail("cannot create staged expanded shard: " + staged_path.string());
	}
	copy_prefix(input_path, output, footer.table_descriptor_offset);
	output.write(reinterpret_cast<const char*>(descriptor.data()), static_cast<std::streamsize>(descriptor.size()));
	fastlanes::FileFooter expanded_footer = footer;
	expanded_footer.table_descriptor_size = descriptor.size();
	output.write(reinterpret_cast<const char*>(&expanded_footer), sizeof(expanded_footer));
	output.close();
	if (!output) {
		fail("failed to finalize staged expanded shard");
	}
	std::filesystem::rename(staged_path, output_path);
	cleanup.release();

	CompactV3Report report;
	report.source_file_bytes        = input_size;
	report.output_file_bytes        = std::filesystem::file_size(output_path);
	report.payload_bytes            = compact.payload_bytes();
	report.source_descriptor_bytes  = descriptor.size();
	report.compact_descriptor_bytes = footer.table_descriptor_size;
	report.payload_crc64            = compact.payload_crc64();
	report.rowgroup_count           = compact.rowgroup_count();
	report.schema_count             = compact.schema_count();
	report.descriptor_reduction     = descriptor.size() == 0U ? 0.0
	                                                          : 1.0 - static_cast<double>(footer.table_descriptor_size) /
                                                                      static_cast<double>(descriptor.size());
	return report;
}

CompactV3PayloadAudit verify_compact_v3_payload(const std::filesystem::path& input_path) {
	auto                  compact = CompactDescriptorV3::Open(input_path);
	fastlanes::File       file(input_path);
	CompactV3PayloadAudit audit;
	audit.expected_payload_crc64 = compact.payload_crc64();
	audit.actual_payload_crc64   = crc64_file_range(file, sizeof(fastlanes::FileHeader), compact.payload_bytes());
	audit.payload_bytes          = compact.payload_bytes();
	audit.rowgroup_count         = compact.rowgroup_count();
	audit.actual_rowgroup_crc64.reserve(compact.rowgroup_count());
	for (size_t rowgroup_index = 0U; rowgroup_index < compact.rowgroup_count(); ++rowgroup_index) {
		const auto rowgroup     = compact.rowgroup(rowgroup_index);
		const auto actual_crc64 = crc64_file_range(file, rowgroup.payload_offset, rowgroup.payload_size);
		audit.actual_rowgroup_crc64.push_back(actual_crc64);
		if (actual_crc64 != rowgroup.payload_crc64) {
			audit.rowgroup_crc_mismatches.push_back(checked_u32(rowgroup_index, "rowgroup index"));
		}
	}
	return audit;
}

} // namespace galp::format
