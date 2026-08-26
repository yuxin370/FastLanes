// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/format/reader.cu
// ────────────────────────────────────────────────────────
#include "engine/materialization/zero_copy_materializer.cuh"
#include "core/operator_capabilities.hpp"
#include "format/compact_descriptor_v3.hpp"
#include "format/compact_read_plan.hpp"
#include "format/reader.cuh"
#include "flatbuffers/flatbuffer_builder.h"
#include "fls/cor/lyt/buf.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/segment_descriptor.hpp"
#include "fls/io/file.hpp"
#include "galp/errors.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <flatbuffers/base.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace galp::format::detail {

struct SparseByteRange {
	size_t offset = 0;
	size_t size   = 0;
};

struct SparseRowgroupAccessIndex {
	bool                                      supported = false;
	std::string                               fallback_reason;
	std::vector<SparseByteRange>              index_ranges;
	std::vector<SparseByteRange>              shared_ranges;
	std::vector<std::vector<SparseByteRange>> vector_ranges;
	// For every logical column, identify its non-shared segment positions in
	// vector_ranges.  This lets a single sparse plan intersect spatial vectors
	// with coefficient columns without rebuilding descriptor geometry.
	std::vector<std::vector<size_t>>           column_vector_range_indices;
	std::vector<size_t>                       vector_storage_bytes;
	std::vector<std::byte>                    static_prefix;
};

struct SparseDatasetAccessIndex {
	size_t rowgroup_count = 0U;
	mutable std::mutex mutex;
	mutable std::unordered_map<size_t, std::shared_ptr<const SparseRowgroupAccessIndex>> rowgroups;
};

struct SparseReadRecipeRecord {
	uint32_t                     rowgroup_index = 0U;
	uint64_t                     rowgroup_bytes = 0U;
	uint64_t                     selected_storage_bytes = 0U;
	uint64_t                     selection_digest = 0U;
	std::vector<uint64_t>        selection_words;
	std::vector<SparseByteRange> index_ranges;
	std::vector<SparseByteRange> shared_ranges;
	std::vector<SparseByteRange> source_ranges;
	mutable std::shared_ptr<const SparseRowgroupAccessIndex> prehydrated_access;
	mutable size_t metadata_bytes       = 0U;
	mutable size_t metadata_pread_count = 0U;
	mutable double rehydrate_service_ms = 0.0;
};

struct SparseReadRecipeIndex {
	std::filesystem::path path;
	uint64_t              source_file_size    = 0U;
	uint64_t              source_fingerprint  = 0U;
	uint64_t              source_stat_digest  = 0U;
	uint64_t              descriptor_digest   = 0U;
	uint64_t              sidecar_crc64       = 0U;
	size_t                sidecar_bytes       = 0U;
	std::vector<SparseReadRecipeRecord> records;
};

constexpr std::array<std::byte, 8> kSparseReadRecipeMagic {
    static_cast<std::byte>('F'), static_cast<std::byte>('L'), static_cast<std::byte>('S'),
    static_cast<std::byte>('S'), static_cast<std::byte>('R'), static_cast<std::byte>('P'),
	static_cast<std::byte>('0'), static_cast<std::byte>('3')};
constexpr uint32_t kSparseReadRecipeVersion      = 3U;
constexpr uint32_t kSparseReadRecipeEndianMarker = UINT32_C(0x01020304);
constexpr size_t   kSparseReadRecipeHeaderSize   = 128U;
constexpr size_t   kSparseReadRecipeRecordSize   = 80U;
constexpr size_t   kSparseReadRecipeChecksumByte = 88U;

constexpr std::array<uint64_t, 256> make_sparse_recipe_crc64_table() {
	std::array<uint64_t, 256> table {};
	for (uint64_t value = 0U; value < table.size(); ++value) {
		uint64_t crc = value;
		for (uint32_t bit = 0U; bit < 8U; ++bit) {
			crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? UINT64_C(0xC96C5795D7870F42) : 0U);
		}
		table[value] = crc;
	}
	return table;
}

constexpr auto kSparseRecipeCrc64Table = make_sparse_recipe_crc64_table();

uint64_t sparse_recipe_crc64_update(uint64_t crc, const void* const data, const size_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	for (size_t index = 0U; index < size; ++index) {
		crc = kSparseRecipeCrc64Table[(crc ^ bytes[index]) & 0xFFU] ^ (crc >> 8U);
	}
	return crc;
}

uint64_t sparse_recipe_crc64_update_u64_le(uint64_t crc, const uint64_t value) {
	std::array<std::byte, sizeof(uint64_t)> encoded {};
	for (size_t byte = 0U; byte < encoded.size(); ++byte) {
		encoded[byte] = static_cast<std::byte>((value >> (byte * 8U)) & 0xFFU);
	}
	return sparse_recipe_crc64_update(crc, encoded.data(), encoded.size());
}

uint64_t sparse_recipe_source_stat_digest(const std::filesystem::path& path) {
	uint64_t digest = 0U;
#if !defined(_WIN32)
	struct stat status {};
	if (::stat(path.c_str(), &status) != 0) {
		throw std::runtime_error("failed to stat sparse read recipe source: " +
		                         std::string(std::strerror(errno)));
	}
	digest = sparse_recipe_crc64_update_u64_le(digest, static_cast<uint64_t>(status.st_dev));
	digest = sparse_recipe_crc64_update_u64_le(digest, static_cast<uint64_t>(status.st_ino));
	digest = sparse_recipe_crc64_update_u64_le(digest, static_cast<uint64_t>(status.st_size));
	digest = sparse_recipe_crc64_update_u64_le(digest, static_cast<uint64_t>(status.st_mtim.tv_sec));
	digest = sparse_recipe_crc64_update_u64_le(digest, static_cast<uint64_t>(status.st_mtim.tv_nsec));
	digest = sparse_recipe_crc64_update_u64_le(digest, static_cast<uint64_t>(status.st_ctim.tv_sec));
	digest = sparse_recipe_crc64_update_u64_le(digest, static_cast<uint64_t>(status.st_ctim.tv_nsec));
#else
	digest = sparse_recipe_crc64_update_u64_le(digest, std::filesystem::file_size(path));
	digest = sparse_recipe_crc64_update_u64_le(
	    digest,
	    static_cast<uint64_t>(std::filesystem::last_write_time(path).time_since_epoch().count()));
#endif
	return digest;
}

uint64_t sparse_recipe_crc64_file(fastlanes::File& file) {
	constexpr size_t kChunkBytes = 4U * 1024U * 1024U;
	std::vector<std::byte> buffer(kChunkBytes);
	const auto file_size = file.Size();
	uint64_t crc = 0U;
	fastlanes::n_t offset = 0U;
	while (offset < file_size) {
		const auto remaining = file_size - offset;
		const auto chunk = static_cast<size_t>(std::min<fastlanes::n_t>(remaining, kChunkBytes));
		file.ReadRangeUnchecked(buffer.data(), offset, chunk);
		crc = sparse_recipe_crc64_update(crc, buffer.data(), chunk);
		offset += chunk;
	}
	return crc;
}

template <typename T>
void put_recipe_le(std::vector<std::byte>& bytes, const size_t offset, const T value) {
	static_assert(std::is_integral_v<T>);
	if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
		throw std::runtime_error("sparse read recipe write exceeds its buffer");
	}
	using Unsigned = std::make_unsigned_t<T>;
	const auto encoded = static_cast<Unsigned>(value);
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		bytes[offset + byte] = static_cast<std::byte>((encoded >> (byte * 8U)) & 0xFFU);
	}
}

template <typename T>
T read_recipe_le(const std::byte* const bytes,
	             const size_t           size,
	             const size_t           offset,
	             const char* const      label) {
	static_assert(std::is_integral_v<T>);
	if (offset > size || sizeof(T) > size - offset) {
		throw std::runtime_error(std::string("truncated sparse read recipe ") + label);
	}
	using Unsigned = std::make_unsigned_t<T>;
	Unsigned value = 0U;
	for (size_t byte = 0U; byte < sizeof(T); ++byte) {
		value |= static_cast<Unsigned>(std::to_integer<uint8_t>(bytes[offset + byte])) << (byte * 8U);
	}
	return static_cast<T>(value);
}

uint64_t sparse_recipe_crc64_with_zeroed_checksum(const std::vector<std::byte>& bytes) {
	if (bytes.size() < kSparseReadRecipeChecksumByte + sizeof(uint64_t)) {
		throw std::runtime_error("sparse read recipe is shorter than its checksum field");
	}
	uint64_t crc = sparse_recipe_crc64_update(0U, bytes.data(), kSparseReadRecipeChecksumByte);
	const std::array<std::byte, sizeof(uint64_t)> zeros {};
	crc = sparse_recipe_crc64_update(crc, zeros.data(), zeros.size());
	return sparse_recipe_crc64_update(
	    crc,
	    bytes.data() + kSparseReadRecipeChecksumByte + sizeof(uint64_t),
	    bytes.size() - kSparseReadRecipeChecksumByte - sizeof(uint64_t));
}

constexpr std::array<std::byte, 8> kSparseVectorBundleMagic {
    static_cast<std::byte>('F'), static_cast<std::byte>('L'), static_cast<std::byte>('S'),
    static_cast<std::byte>('V'), static_cast<std::byte>('B'), static_cast<std::byte>('R'),
    static_cast<std::byte>('0'), static_cast<std::byte>('1')};
constexpr uint32_t kSparseVectorBundleVersion    = 1U;
constexpr size_t   kSparseVectorBundleHeaderSize = 64U;

struct SparseVectorBundleRowgroup {
	uint64_t              source_offset  = 0;
	uint64_t              source_size    = 0;
	uint64_t              prefix_offset  = 0;
	uint64_t              prefix_size    = 0;
	uint64_t              vectors_offset = 0;
	uint32_t              n_vecs         = 0;
	std::vector<uint64_t> vector_offsets;
};

struct SparseVectorBundleIndex {
	std::filesystem::path                       path;
	std::shared_ptr<fastlanes::File>            file;
	uint64_t                                    source_file_size = 0;
	std::vector<SparseVectorBundleRowgroup>     rowgroups;
};

void append_u32(std::vector<std::byte>& bytes, const uint32_t value) {
	const auto begin = bytes.size();
	bytes.resize(begin + sizeof(value));
	std::memcpy(bytes.data() + begin, &value, sizeof(value));
}

void append_u64(std::vector<std::byte>& bytes, const uint64_t value) {
	const auto begin = bytes.size();
	bytes.resize(begin + sizeof(value));
	std::memcpy(bytes.data() + begin, &value, sizeof(value));
}

uint32_t consume_u32(const std::vector<std::byte>& bytes, size_t& cursor, const char* const label) {
	if (cursor > bytes.size() || sizeof(uint32_t) > bytes.size() - cursor) {
		throw std::runtime_error(std::string("truncated sparse vector bundle ") + label);
	}
	uint32_t value = 0;
	std::memcpy(&value, bytes.data() + cursor, sizeof(value));
	cursor += sizeof(value);
	return value;
}

uint64_t consume_u64(const std::vector<std::byte>& bytes, size_t& cursor, const char* const label) {
	if (cursor > bytes.size() || sizeof(uint64_t) > bytes.size() - cursor) {
		throw std::runtime_error(std::string("truncated sparse vector bundle ") + label);
	}
	uint64_t value = 0;
	std::memcpy(&value, bytes.data() + cursor, sizeof(value));
	cursor += sizeof(value);
	return value;
}

void collect_segment_descriptors(const fastlanes::ColumnDescriptor&              column,
                                 std::vector<const fastlanes::SegmentDescriptor*>& out) {
	if (const auto* segments = column.segment_descriptors(); segments != nullptr) {
		for (flatbuffers::uoffset_t i = 0; i < segments->size(); ++i) {
			if (const auto* segment = segments->Get(i); segment != nullptr) {
				out.push_back(segment);
			}
		}
	}
	if (const auto* children = column.children(); children != nullptr) {
		for (flatbuffers::uoffset_t i = 0; i < children->size(); ++i) {
			if (const auto* child = children->Get(i); child != nullptr) {
				collect_segment_descriptors(*child, out);
			}
		}
	}
}

std::vector<const fastlanes::SegmentDescriptor*>
rowgroup_segment_descriptors(const fastlanes::RowgroupDescriptor& rowgroup) {
	std::vector<const fastlanes::SegmentDescriptor*> out;
	if (const auto* columns = rowgroup.m_column_descriptors(); columns != nullptr) {
		for (flatbuffers::uoffset_t i = 0; i < columns->size(); ++i) {
			if (const auto* column = columns->Get(i); column != nullptr) {
				collect_segment_descriptors(*column, out);
			}
		}
	}
	return out;
}

bool validate_sparse_column_operators(const fastlanes::ColumnDescriptor& column, std::string* const reason) {
	const auto fail = [&](const std::string& value) {
		if (reason != nullptr) {
			*reason = value;
		}
		return false;
	};
	const auto* rpn    = column.encoding_rpn();
	const auto* tokens = rpn == nullptr ? nullptr : rpn->operator_tokens();
	if (tokens == nullptr || tokens->empty()) {
		return fail("column-has-no-operator-capability");
	}
	for (flatbuffers::uoffset_t token_index = 0; token_index < tokens->size(); ++token_index) {
		const auto token = tokens->Get(token_index);
		if (!galp::expression::is_sparse_read_supported_token(token)) {
			return fail("operator-sparse-read-unsupported:" + fastlanes::token_to_string(token));
		}
	}
	if (const auto* children = column.children(); children != nullptr) {
		for (flatbuffers::uoffset_t child_index = 0; child_index < children->size(); ++child_index) {
			const auto* child = children->Get(child_index);
			if (child == nullptr) {
				return fail("null-child-column-descriptor");
			}
			if (!validate_sparse_column_operators(*child, reason)) {
				return false;
			}
		}
	}
	return true;
}

bool validate_sparse_vector_segments(const fastlanes::RowgroupDescriptor&                   rowgroup,
                                     const std::vector<const fastlanes::SegmentDescriptor*>& segments,
                                     std::string* const                                      reason) {
	const auto fail = [&](const std::string& value) {
		if (reason != nullptr) {
			*reason = value;
		}
		return false;
	};
	const size_t n_vecs   = static_cast<size_t>(rowgroup.m_n_vec());
	const size_t rg_bytes = static_cast<size_t>(rowgroup.m_size());
	if (n_vecs == 0) {
		return fail("rowgroup-has-no-vectors");
	}
	if (segments.empty()) {
		return fail("rowgroup-has-no-persistent-vector-segments");
	}
	const auto* columns = rowgroup.m_column_descriptors();
	if (columns == nullptr) {
		return fail("rowgroup-has-no-column-descriptors");
	}
	for (flatbuffers::uoffset_t column_index = 0; column_index < columns->size(); ++column_index) {
		const auto* column = columns->Get(column_index);
		if (column == nullptr) {
			return fail("null-column-descriptor");
		}
		if (!validate_sparse_column_operators(*column, reason)) {
			return false;
		}
	}
	for (const auto* segment : segments) {
		if (segment == nullptr) {
			return fail("null-segment-descriptor");
		}
		const size_t point_width = fastlanes::sizeof_entry_point_type(segment->entry_point_t());
		if (point_width == 0) {
			return fail("unsupported-entrypoint-type");
		}
		if (segment->entrypoint_size() % point_width != 0) {
			return fail("segment-entrypoint-size-is-misaligned");
		}
		const size_t entrypoint_count = static_cast<size_t>(segment->entrypoint_size()) / point_width;
		if (entrypoint_count != 1U && entrypoint_count != n_vecs) {
			return fail("segment-entrypoint-count-is-neither-shared-nor-vector-addressable");
		}
		if (segment->entrypoint_offset() > rg_bytes || segment->entrypoint_size() > rg_bytes - segment->entrypoint_offset() ||
		    segment->data_offset() > rg_bytes || segment->data_size() > rg_bytes - segment->data_offset()) {
			return fail("segment-range-exceeds-rowgroup");
		}
	}
	if (reason != nullptr) {
		reason->clear();
	}
	return true;
}

std::vector<SparseByteRange> coalesce_ranges(std::vector<SparseByteRange> ranges) {
	ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const auto& range) { return range.size == 0; }),
	             ranges.end());
	std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
		return lhs.offset < rhs.offset || (lhs.offset == rhs.offset && lhs.size < rhs.size);
	});
	std::vector<SparseByteRange> out;
	for (const auto& range : ranges) {
		if (out.empty() || range.offset > out.back().offset + out.back().size) {
			out.push_back(range);
			continue;
		}
		const size_t end = std::max(out.back().offset + out.back().size, range.offset + range.size);
		out.back().size = end - out.back().offset;
	}
	return out;
}

uint64_t entrypoint_value(const std::byte* const data,
	                      const fastlanes::EntryPointType type,
	                      const size_t index) {
	switch (type) {
	case fastlanes::EntryPointType::UINT8:
		return static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(data)[index]);
	case fastlanes::EntryPointType::UINT16: {
		uint16_t value = 0;
		std::memcpy(&value, data + index * sizeof(value), sizeof(value));
		return value;
	}
	case fastlanes::EntryPointType::UINT32: {
		uint32_t value = 0;
		std::memcpy(&value, data + index * sizeof(value), sizeof(value));
		return value;
	}
	case fastlanes::EntryPointType::UINT64: {
		uint64_t value = 0;
		std::memcpy(&value, data + index * sizeof(value), sizeof(value));
		return value;
	}
	default:
		throw std::runtime_error("unsupported sparse vector entrypoint type");
	}
}

size_t segment_entrypoint_count(const fastlanes::SegmentDescriptor& segment) {
	const size_t point_width = fastlanes::sizeof_entry_point_type(segment.entry_point_t());
	if (point_width == 0U || segment.entrypoint_size() % point_width != 0U) {
		throw std::runtime_error("invalid sparse vector segment entrypoint encoding");
	}
	return static_cast<size_t>(segment.entrypoint_size()) / point_width;
}

SparseByteRange segment_vector_range(const fastlanes::SegmentDescriptor& segment,
	                                 const std::byte* const              backing_data,
	                                 const uint32_t                       vector) {
	const auto* points = backing_data + static_cast<size_t>(segment.entrypoint_offset());
	const uint64_t begin = vector == 0U ? 0U : entrypoint_value(points, segment.entry_point_t(), vector - 1U);
	const uint64_t end   = entrypoint_value(points, segment.entry_point_t(), vector);
	if (end < begin || end > segment.data_size()) {
		throw std::runtime_error("invalid cumulative entrypoint in sparse vector bundle");
	}
	return SparseByteRange {static_cast<size_t>(segment.data_offset() + begin), static_cast<size_t>(end - begin)};
}

std::vector<SparseByteRange>
segment_index_ranges(const std::vector<const fastlanes::SegmentDescriptor*>& segments) {
	std::vector<SparseByteRange> ranges;
	ranges.reserve(segments.size());
	for (const auto* segment : segments) {
		ranges.push_back(SparseByteRange {static_cast<size_t>(segment->entrypoint_offset()),
		                                  static_cast<size_t>(segment->entrypoint_size())});
	}
	return coalesce_ranges(std::move(ranges));
}

std::vector<SparseByteRange>
segment_shared_ranges(const std::vector<const fastlanes::SegmentDescriptor*>& segments,
	                  const std::byte* const                                      backing_data) {
	std::vector<SparseByteRange> ranges;
	ranges.reserve(segments.size());
	for (const auto* segment : segments) {
		if (segment_entrypoint_count(*segment) != 1U) {
			continue;
		}
		const auto* points = backing_data + static_cast<size_t>(segment->entrypoint_offset());
		const uint64_t end = entrypoint_value(points, segment->entry_point_t(), 0U);
		if (end > segment->data_size()) {
			throw std::runtime_error("invalid shared entrypoint in sparse vector bundle");
		}
		ranges.push_back(
		    SparseByteRange {static_cast<size_t>(segment->data_offset()), static_cast<size_t>(end)});
	}
	return coalesce_ranges(std::move(ranges));
}

size_t total_range_bytes(const std::vector<SparseByteRange>& ranges) {
	size_t total = 0U;
	for (const auto& range : ranges) {
		if (range.size > std::numeric_limits<size_t>::max() - total) {
			throw std::runtime_error("sparse vector bundle byte count overflow");
		}
		total += range.size;
	}
	return total;
}

std::vector<std::byte> pack_ranges(const std::byte* const backing_data,
	                               const std::vector<SparseByteRange>& ranges) {
	std::vector<std::byte> packed(total_range_bytes(ranges));
	size_t                 cursor = 0U;
	for (const auto& range : ranges) {
		std::memcpy(packed.data() + cursor, backing_data + range.offset, range.size);
		cursor += range.size;
	}
	return packed;
}

void scatter_ranges(const std::byte* const packed,
	                const size_t           packed_size,
	                std::byte* const       backing_data,
	                const std::vector<SparseByteRange>& ranges,
	                size_t& cursor) {
	for (const auto& range : ranges) {
		if (cursor > packed_size || range.size > packed_size - cursor) {
			throw std::runtime_error("truncated sparse vector bundle rowgroup prefix");
		}
		std::memcpy(backing_data + range.offset, packed + cursor, range.size);
		cursor += range.size;
	}
}

template <typename T>
void sparse_recipe_crc64_scalar(uint64_t& crc, const T value) {
	static_assert(std::is_integral_v<T> || std::is_enum_v<T>);
	crc = sparse_recipe_crc64_update(crc, &value, sizeof(value));
}

void sparse_descriptor_column_digest(uint64_t& crc, const fastlanes::ColumnDescriptor& column) {
	const auto* rpn = column.encoding_rpn();
	const auto* operators = rpn == nullptr ? nullptr : rpn->operator_tokens();
	const uint32_t operator_count = operators == nullptr ? 0U : operators->size();
	sparse_recipe_crc64_scalar(crc, operator_count);
	for (uint32_t index = 0U; index < operator_count; ++index) {
		sparse_recipe_crc64_scalar(crc, static_cast<uint32_t>(operators->Get(index)));
	}
	const auto* segments = column.segment_descriptors();
	const uint32_t segment_count = segments == nullptr ? 0U : segments->size();
	sparse_recipe_crc64_scalar(crc, segment_count);
	for (uint32_t index = 0U; index < segment_count; ++index) {
		const auto* segment = segments->Get(index);
		const uint8_t present = segment == nullptr ? 0U : 1U;
		sparse_recipe_crc64_scalar(crc, present);
		if (segment == nullptr) {
			continue;
		}
		sparse_recipe_crc64_scalar(crc, segment->entrypoint_offset());
		sparse_recipe_crc64_scalar(crc, segment->entrypoint_size());
		sparse_recipe_crc64_scalar(crc, segment->data_offset());
		sparse_recipe_crc64_scalar(crc, segment->data_size());
		sparse_recipe_crc64_scalar(crc, static_cast<uint32_t>(segment->entry_point_t()));
	}
	const auto* children = column.children();
	const uint32_t child_count = children == nullptr ? 0U : children->size();
	sparse_recipe_crc64_scalar(crc, child_count);
	for (uint32_t index = 0U; index < child_count; ++index) {
		const auto* child = children->Get(index);
		const uint8_t present = child == nullptr ? 0U : 1U;
		sparse_recipe_crc64_scalar(crc, present);
		if (child != nullptr) {
			sparse_descriptor_column_digest(crc, *child);
		}
	}
}

uint64_t sparse_descriptor_digest(const fastlanes::TableDescriptor& table_descriptor) {
	uint64_t crc = 0U;
	const auto* rowgroups = table_descriptor.m_rowgroup_descriptors();
	const uint32_t rowgroup_count = rowgroups == nullptr ? 0U : rowgroups->size();
	sparse_recipe_crc64_scalar(crc, rowgroup_count);
	for (uint32_t rowgroup_index = 0U; rowgroup_index < rowgroup_count; ++rowgroup_index) {
		const auto* rowgroup = rowgroups->Get(rowgroup_index);
		const uint8_t present = rowgroup == nullptr ? 0U : 1U;
		sparse_recipe_crc64_scalar(crc, present);
		if (rowgroup == nullptr) {
			continue;
		}
		sparse_recipe_crc64_scalar(crc, rowgroup->m_offset());
		sparse_recipe_crc64_scalar(crc, rowgroup->m_size());
		sparse_recipe_crc64_scalar(crc, rowgroup->m_n_vec());
		const auto* columns = rowgroup->m_column_descriptors();
		const uint32_t column_count = columns == nullptr ? 0U : columns->size();
		sparse_recipe_crc64_scalar(crc, column_count);
		for (uint32_t column_index = 0U; column_index < column_count; ++column_index) {
			const auto* column = columns->Get(column_index);
			const uint8_t column_present = column == nullptr ? 0U : 1U;
			sparse_recipe_crc64_scalar(crc, column_present);
			if (column != nullptr) {
				sparse_descriptor_column_digest(crc, *column);
			}
		}
	}
	return crc;
}

std::vector<uint64_t> sparse_selection_words(const size_t vector_count,
	                                         const std::vector<uint32_t>& selected_vectors) {
	std::vector<uint64_t> words((vector_count + 63U) / 64U, 0U);
	for (const auto vector : selected_vectors) {
		if (vector >= vector_count) {
			throw std::out_of_range("sparse recipe selected vector exceeds rowgroup vector count");
		}
		words[vector / 64U] |= UINT64_C(1) << (vector % 64U);
	}
	return words;
}

uint64_t sparse_selection_digest(const std::vector<uint64_t>& words) {
	uint64_t crc = 0U;
	const uint64_t count = words.size();
	crc = sparse_recipe_crc64_update(crc, &count, sizeof(count));
	return sparse_recipe_crc64_update(crc, words.data(), words.size() * sizeof(uint64_t));
}

std::shared_ptr<const SparseRowgroupAccessIndex>
build_sparse_rowgroup_access_index(fastlanes::File& file, const fastlanes::RowgroupDescriptor& rowgroup) {
	auto entry = std::make_shared<SparseRowgroupAccessIndex>();
	const auto segments = rowgroup_segment_descriptors(rowgroup);
	if (!validate_sparse_vector_segments(rowgroup, segments, &entry->fallback_reason)) {
		return entry;
	}
	entry->index_ranges = segment_index_ranges(segments);
	const auto rowgroup_bytes = static_cast<size_t>(rowgroup.m_size());
	std::vector<std::byte> index_backing(rowgroup_bytes, std::byte {0});
	for (const auto& range : entry->index_ranges) {
		file.ReadRangeUnchecked(index_backing.data() + range.offset, rowgroup.m_offset() + range.offset, range.size);
	}
	entry->shared_ranges = segment_shared_ranges(segments, index_backing.data());
	for (const auto& range : entry->shared_ranges) {
		file.ReadRangeUnchecked(index_backing.data() + range.offset, rowgroup.m_offset() + range.offset, range.size);
	}
	entry->static_prefix = pack_ranges(index_backing.data(), entry->index_ranges);
	const auto shared_prefix = pack_ranges(index_backing.data(), entry->shared_ranges);
	entry->static_prefix.insert(entry->static_prefix.end(), shared_prefix.begin(), shared_prefix.end());
	entry->vector_ranges.resize(rowgroup.m_n_vec());
	entry->vector_storage_bytes.assign(rowgroup.m_n_vec(), 0U);
	const auto* columns = rowgroup.m_column_descriptors();
	if (columns == nullptr) {
		throw std::runtime_error("sparse rowgroup column descriptors are missing");
	}
	entry->column_vector_range_indices.resize(columns->size());
	size_t vector_range_index = 0U;
	for (flatbuffers::uoffset_t column_index = 0U; column_index < columns->size(); ++column_index) {
		const auto* column = columns->Get(column_index);
		if (column == nullptr) {
			throw std::runtime_error("sparse rowgroup column descriptor is missing");
		}
		std::vector<const fastlanes::SegmentDescriptor*> column_segments;
		collect_segment_descriptors(*column, column_segments);
		for (const auto* segment : column_segments) {
			if (segment_entrypoint_count(*segment) == 1U) {
				continue;
			}
			entry->column_vector_range_indices[column_index].push_back(vector_range_index++);
			for (uint32_t vector = 0U; vector < rowgroup.m_n_vec(); ++vector) {
				const auto range = segment_vector_range(*segment, index_backing.data(), vector);
				entry->vector_ranges[vector].push_back(range);
				if (range.size > std::numeric_limits<size_t>::max() - entry->vector_storage_bytes[vector]) {
					throw std::runtime_error("sparse rowgroup vector byte count overflow");
				}
				entry->vector_storage_bytes[vector] += range.size;
			}
		}
	}
	if (std::any_of(entry->vector_ranges.begin(), entry->vector_ranges.end(), [&](const auto& ranges) {
		    return ranges.size() != vector_range_index;
	    })) {
		throw std::logic_error("sparse rowgroup column/vector range index is inconsistent");
	}
	entry->supported = true;
	entry->fallback_reason.clear();
	return entry;
}

std::shared_ptr<const SparseDatasetAccessIndex>
build_sparse_dataset_access_index(const fastlanes::TableDescriptor& table_descriptor) {
	auto index = std::make_shared<SparseDatasetAccessIndex>();
	const auto* rowgroups = table_descriptor.m_rowgroup_descriptors();
	index->rowgroup_count = rowgroups == nullptr ? 0U : rowgroups->size();
	return index;
}

std::shared_ptr<const SparseRowgroupAccessIndex> sparse_rowgroup_access(
	fastlanes::File& file,
	const fastlanes::TableDescriptor& table_descriptor,
	const std::shared_ptr<const SparseDatasetAccessIndex>& dataset_index,
	const size_t rowgroup_index) {
	if (!dataset_index || rowgroup_index >= dataset_index->rowgroup_count) {
		throw std::out_of_range("sparse rowgroup access index is out of range");
	}
	std::lock_guard<std::mutex> guard(dataset_index->mutex);
	if (const auto found = dataset_index->rowgroups.find(rowgroup_index);
	    found != dataset_index->rowgroups.end()) {
		return found->second;
	}
	const auto* rowgroups = table_descriptor.m_rowgroup_descriptors();
	const auto* rowgroup = rowgroups == nullptr
	                           ? nullptr
	                           : rowgroups->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
	std::shared_ptr<const SparseRowgroupAccessIndex> built;
	if (rowgroup == nullptr) {
		auto unsupported = std::make_shared<SparseRowgroupAccessIndex>();
		unsupported->fallback_reason = "null-rowgroup-descriptor";
		built = std::move(unsupported);
	} else {
		built = build_sparse_rowgroup_access_index(file, *rowgroup);
	}
	dataset_index->rowgroups.emplace(rowgroup_index, built);
	return built;
}

fastlanes::TableDescriptorHandle load_table_descriptor(fastlanes::File&              file,
	                                                    const std::filesystem::path& file_path) {
	fastlanes::FileHeader file_header {};
	fastlanes::FileFooter file_footer {};

	fastlanes::FileHeader::Load(file_header, file);
	fastlanes::FileFooter::Load(file_footer, file);

	if (file_header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file, file_footer.table_descriptor_offset, file_footer.table_descriptor_size, /*verify=*/true);
	}

	const auto footer_path = file_path.parent_path() / "table_descriptor.fbb";
	return fastlanes::TableDescriptorHandle::FromFile(footer_path, /*verify=*/true);
}

fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::File file(file_path);
	return load_table_descriptor(file, file_path);
}

std::shared_ptr<const SparseVectorBundleIndex>
load_sparse_vector_bundle_index(const std::filesystem::path&                 bundle_path,
	                            const fastlanes::TableDescriptor&             table_descriptor,
	                            const uint64_t                                 source_file_size) {
	if (!std::filesystem::exists(bundle_path)) {
		return nullptr;
	}
	auto bundle_file = std::make_shared<fastlanes::File>(bundle_path);
	if (bundle_file->Size() < kSparseVectorBundleHeaderSize) {
		throw std::runtime_error("sparse vector bundle header is truncated: " + bundle_path.string());
	}
	std::vector<std::byte> header(kSparseVectorBundleHeaderSize);
	bundle_file->ReadRangeUnchecked(header.data(), 0U, header.size());
	if (!std::equal(kSparseVectorBundleMagic.begin(), kSparseVectorBundleMagic.end(), header.begin())) {
		throw std::runtime_error("sparse vector bundle magic mismatch: " + bundle_path.string());
	}
	size_t cursor = kSparseVectorBundleMagic.size();
	const uint32_t version = consume_u32(header, cursor, "version");
	if (version != kSparseVectorBundleVersion) {
		throw std::runtime_error("unsupported sparse vector bundle version: " + std::to_string(version));
	}
	const uint32_t rowgroup_count  = consume_u32(header, cursor, "rowgroup count");
	const uint64_t recorded_source = consume_u64(header, cursor, "source file size");
	const uint64_t directory_offset = consume_u64(header, cursor, "directory offset");
	const uint64_t directory_size   = consume_u64(header, cursor, "directory size");
	if (recorded_source != source_file_size) {
		throw std::runtime_error("sparse vector bundle source size does not match FLS payload");
	}
	const auto* rowgroups = table_descriptor.m_rowgroup_descriptors();
	if (rowgroups == nullptr || rowgroup_count != rowgroups->size()) {
		throw std::runtime_error("sparse vector bundle rowgroup count does not match FLS descriptor");
	}
	const uint64_t bundle_size = bundle_file->Size();
	if (directory_offset > bundle_size || directory_size > bundle_size - directory_offset ||
	    directory_size > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error("sparse vector bundle directory exceeds file bounds");
	}
	std::vector<std::byte> directory(static_cast<size_t>(directory_size));
	bundle_file->ReadRangeUnchecked(directory.data(), directory_offset, directory.size());

	auto index              = std::make_shared<SparseVectorBundleIndex>();
	index->path             = bundle_path;
	index->file             = std::move(bundle_file);
	index->source_file_size = recorded_source;
	index->rowgroups.reserve(rowgroup_count);
	cursor = 0U;
	for (uint32_t rowgroup_index = 0; rowgroup_index < rowgroup_count; ++rowgroup_index) {
		SparseVectorBundleRowgroup entry;
		entry.source_offset  = consume_u64(directory, cursor, "rowgroup source offset");
		entry.source_size    = consume_u64(directory, cursor, "rowgroup source size");
		entry.prefix_offset  = consume_u64(directory, cursor, "rowgroup prefix offset");
		entry.prefix_size    = consume_u64(directory, cursor, "rowgroup prefix size");
		entry.vectors_offset = consume_u64(directory, cursor, "rowgroup vectors offset");
		entry.n_vecs         = consume_u32(directory, cursor, "rowgroup vector count");
		static_cast<void>(consume_u32(directory, cursor, "rowgroup flags"));
		entry.vector_offsets.reserve(static_cast<size_t>(entry.n_vecs) + 1U);
		for (uint32_t vector = 0; vector <= entry.n_vecs; ++vector) {
			entry.vector_offsets.push_back(consume_u64(directory, cursor, "vector offset"));
		}
		const auto* rowgroup = rowgroups->Get(rowgroup_index);
		if (rowgroup == nullptr || entry.source_offset != rowgroup->m_offset() ||
		    entry.source_size != rowgroup->m_size() || entry.n_vecs != rowgroup->m_n_vec()) {
			throw std::runtime_error("sparse vector bundle rowgroup geometry does not match FLS descriptor");
		}
		if (entry.prefix_offset > directory_offset || entry.prefix_size > directory_offset - entry.prefix_offset ||
		    entry.vectors_offset > directory_offset || entry.vector_offsets.empty() ||
		    entry.vector_offsets.front() != 0U) {
			throw std::runtime_error("sparse vector bundle rowgroup payload is malformed");
		}
		for (size_t vector = 1; vector < entry.vector_offsets.size(); ++vector) {
			if (entry.vector_offsets[vector] < entry.vector_offsets[vector - 1U]) {
				throw std::runtime_error("sparse vector bundle vector offsets are not monotonic");
			}
		}
		if (entry.vector_offsets.back() > directory_offset - entry.vectors_offset) {
			throw std::runtime_error("sparse vector bundle vector payload exceeds file bounds");
		}
		index->rowgroups.push_back(std::move(entry));
	}
	if (cursor != directory.size()) {
		throw std::runtime_error("sparse vector bundle directory has trailing bytes");
	}
	return index;
}

std::vector<SparseByteRange> read_sparse_recipe_ranges(const std::byte* const payload,
	                                                   const size_t payload_size,
	                                                   size_t& cursor,
	                                                   const uint32_t count,
	                                                   const size_t rowgroup_bytes,
	                                                   const char* const label) {
	std::vector<SparseByteRange> ranges;
	ranges.reserve(count);
	size_t previous_end = 0U;
	const auto read_varuint32 = [&](const char* const field) {
		uint32_t value = 0U;
		for (uint32_t byte_index = 0U; byte_index < 5U; ++byte_index) {
			if (cursor >= payload_size) {
				throw std::runtime_error(std::string("truncated sparse read recipe ") + field);
			}
			const auto byte = std::to_integer<uint8_t>(payload[cursor++]);
			if (byte_index == 4U && (byte & 0xF0U) != 0U) {
				throw std::runtime_error(std::string("overflowing sparse read recipe ") + field);
			}
			value |= static_cast<uint32_t>(byte & 0x7FU) << (byte_index * 7U);
			if ((byte & 0x80U) == 0U) {
				if (byte_index != 0U && (byte & 0x7FU) == 0U) {
					throw std::runtime_error(std::string("non-canonical sparse read recipe ") + field);
				}
				return value;
			}
		}
		throw std::runtime_error(std::string("unterminated sparse read recipe ") + field);
	};
	for (uint32_t index = 0U; index < count; ++index) {
		const auto delta = read_varuint32(label);
		const auto size  = read_varuint32(label);
		if (delta > rowgroup_bytes - std::min(previous_end, rowgroup_bytes)) {
			throw std::runtime_error(std::string("sparse read recipe has overflowing ") + label);
		}
		const auto offset = previous_end + delta;
		if (size == 0U || offset > rowgroup_bytes || size > rowgroup_bytes - offset ||
		    (!ranges.empty() && delta == 0U)) {
			throw std::runtime_error(std::string("sparse read recipe has invalid/non-monotonic ") + label);
		}
		ranges.push_back({offset, size});
		previous_end = static_cast<size_t>(offset) + size;
	}
	return ranges;
}

std::shared_ptr<const SparseReadRecipeIndex> load_sparse_read_recipe_index(
	const std::filesystem::path& recipe_path,
	const std::filesystem::path& source_path,
	const fastlanes::TableDescriptor& table_descriptor,
	const uint64_t source_file_size,
	const uint64_t expected_source_fingerprint,
	double* const load_ms,
	double* const validation_ms) {
	if (recipe_path.empty() || !std::filesystem::exists(recipe_path)) {
		return nullptr;
	}
	const auto load_begin = std::chrono::steady_clock::now();
	fastlanes::File recipe_file(recipe_path);
	const auto file_size_u64 = recipe_file.Size();
	if (file_size_u64 > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error("sparse read recipe exceeds host address space");
	}
	std::vector<std::byte> encoded(static_cast<size_t>(file_size_u64));
	if (!encoded.empty()) {
		recipe_file.ReadRangeUnchecked(encoded.data(), 0U, encoded.size());
	}
	const auto load_end = std::chrono::steady_clock::now();
	if (load_ms != nullptr) {
		*load_ms = std::chrono::duration<double, std::milli>(load_end - load_begin).count();
	}
	const auto validation_begin = load_end;
	if (encoded.size() < kSparseReadRecipeHeaderSize ||
	    !std::equal(kSparseReadRecipeMagic.begin(), kSparseReadRecipeMagic.end(), encoded.begin())) {
		throw std::runtime_error("sparse read recipe magic/header mismatch: " + recipe_path.string());
	}
	const auto version = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), 8U, "version");
	const auto endian = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), 12U, "endian marker");
	const auto header_size = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), 16U, "header size");
	const auto record_size = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), 20U, "record size");
	if (version != kSparseReadRecipeVersion || endian != kSparseReadRecipeEndianMarker ||
	    header_size != kSparseReadRecipeHeaderSize || record_size != kSparseReadRecipeRecordSize) {
		throw std::runtime_error("unsupported sparse read recipe version/endian/ABI: " + recipe_path.string());
	}
	const auto recorded_source_size = read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 24U, "source size");
	const auto recorded_source_fingerprint =
	    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 32U, "source fingerprint");
	const auto recorded_descriptor_digest =
	    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 40U, "descriptor digest");
	const auto record_count = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), 48U, "record count");
	const auto directory_offset =
	    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 56U, "directory offset");
	const auto directory_size = read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 64U, "directory size");
	const auto payload_offset = read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 72U, "payload offset");
	const auto payload_size = read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 80U, "payload size");
	const auto recorded_crc = read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 88U, "checksum");
	const auto recorded_source_stat_digest =
	    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), 112U, "source stat digest");
	if (recorded_source_size != source_file_size || recorded_source_fingerprint == 0U ||
	    recorded_source_stat_digest == 0U ||
	    recorded_source_stat_digest != sparse_recipe_source_stat_digest(source_path) ||
	    (expected_source_fingerprint != 0U && recorded_source_fingerprint != expected_source_fingerprint) ||
	    recorded_descriptor_digest != sparse_descriptor_digest(table_descriptor)) {
		throw std::runtime_error("sparse read recipe source identity mismatch: " + recipe_path.string());
	}
	if (directory_offset != kSparseReadRecipeHeaderSize ||
	    directory_size != static_cast<uint64_t>(record_count) * kSparseReadRecipeRecordSize ||
	    payload_offset != directory_offset + directory_size || payload_offset > encoded.size() ||
	    payload_size != encoded.size() - payload_offset || recorded_crc != sparse_recipe_crc64_with_zeroed_checksum(encoded)) {
		throw std::runtime_error("sparse read recipe bounds/checksum mismatch: " + recipe_path.string());
	}
	const auto* rowgroups = table_descriptor.m_rowgroup_descriptors();
	if (rowgroups == nullptr) {
		throw std::runtime_error("sparse read recipe source has no rowgroups");
	}
	auto result = std::make_shared<SparseReadRecipeIndex>();
	result->path               = recipe_path;
	result->source_file_size   = recorded_source_size;
	result->source_fingerprint = recorded_source_fingerprint;
	result->source_stat_digest = recorded_source_stat_digest;
	result->descriptor_digest  = recorded_descriptor_digest;
	result->sidecar_crc64      = recorded_crc;
	result->sidecar_bytes      = encoded.size();
	result->records.reserve(record_count);
	uint32_t previous_rowgroup = 0U;
	bool have_previous = false;
	for (uint32_t record_index = 0U; record_index < record_count; ++record_index) {
		const size_t base = static_cast<size_t>(directory_offset) +
		                    static_cast<size_t>(record_index) * kSparseReadRecipeRecordSize;
		SparseReadRecipeRecord record;
		record.rowgroup_index = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), base, "rowgroup index");
		const auto selected_count =
		    read_recipe_le<uint32_t>(encoded.data(), encoded.size(), base + 4U, "selected vector count");
		record.rowgroup_bytes = read_recipe_le<uint64_t>(encoded.data(), encoded.size(), base + 8U, "rowgroup bytes");
		record.selected_storage_bytes =
		    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), base + 16U, "selected bytes");
		const auto selection_word_count =
		    read_recipe_le<uint32_t>(encoded.data(), encoded.size(), base + 24U, "selection word count");
		const auto index_count = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), base + 28U, "index range count");
		const auto shared_count = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), base + 32U, "shared range count");
		const auto source_count = read_recipe_le<uint32_t>(encoded.data(), encoded.size(), base + 36U, "source range count");
		const auto record_payload_offset =
		    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), base + 40U, "record payload offset");
		const auto record_payload_size =
		    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), base + 48U, "record payload size");
		record.selection_digest =
		    read_recipe_le<uint64_t>(encoded.data(), encoded.size(), base + 56U, "selection digest");
		const auto record_crc = read_recipe_le<uint64_t>(encoded.data(), encoded.size(), base + 64U, "record checksum");
		if (record.rowgroup_index >= rowgroups->size() || (have_previous && record.rowgroup_index <= previous_rowgroup)) {
			throw std::runtime_error("sparse read recipe rowgroups are not strictly monotonic/in range");
		}
		const auto* rowgroup = rowgroups->Get(record.rowgroup_index);
		if (rowgroup == nullptr || record.rowgroup_bytes != rowgroup->m_size() ||
		    selection_word_count != (static_cast<uint64_t>(rowgroup->m_n_vec()) + 63U) / 64U ||
		    record_payload_offset < payload_offset || record_payload_offset > encoded.size() ||
		    record_payload_size > encoded.size() - record_payload_offset) {
			throw std::runtime_error("sparse read recipe record geometry/bounds mismatch");
		}
		const auto* record_payload = encoded.data() + static_cast<size_t>(record_payload_offset);
		if (record_crc != sparse_recipe_crc64_update(0U, record_payload, static_cast<size_t>(record_payload_size))) {
			throw std::runtime_error("sparse read recipe record checksum mismatch");
		}
		size_t cursor = 0U;
		record.selection_words.reserve(selection_word_count);
		for (uint32_t word = 0U; word < selection_word_count; ++word) {
			record.selection_words.push_back(
			    read_recipe_le<uint64_t>(record_payload, record_payload_size, cursor, "selection bitmap"));
			cursor += sizeof(uint64_t);
		}
		if (sparse_selection_digest(record.selection_words) != record.selection_digest) {
			throw std::runtime_error("sparse read recipe selection digest mismatch");
		}
		size_t observed_selected = 0U;
		for (const auto word : record.selection_words) {
			observed_selected += static_cast<size_t>(std::popcount(word));
		}
		if (observed_selected != selected_count || observed_selected == 0U) {
			throw std::runtime_error("sparse read recipe selected vector count mismatch");
		}
		record.index_ranges = read_sparse_recipe_ranges(
		    record_payload, record_payload_size, cursor, index_count, record.rowgroup_bytes, "index ranges");
		record.shared_ranges = read_sparse_recipe_ranges(
		    record_payload, record_payload_size, cursor, shared_count, record.rowgroup_bytes, "shared ranges");
		record.source_ranges = read_sparse_recipe_ranges(
		    record_payload, record_payload_size, cursor, source_count, record.rowgroup_bytes, "source ranges");
		if (cursor != record_payload_size || total_range_bytes(record.source_ranges) != record.selected_storage_bytes) {
			throw std::runtime_error("sparse read recipe record has trailing bytes or incorrect selected byte count");
		}
		previous_rowgroup = record.rowgroup_index;
		have_previous = true;
		result->records.push_back(std::move(record));
	}
	const auto validation_end = std::chrono::steady_clock::now();
	if (validation_ms != nullptr) {
		*validation_ms = std::chrono::duration<double, std::milli>(validation_end - validation_begin).count();
	}
	return result;
}

const SparseReadRecipeRecord* find_sparse_read_recipe_record(
	const SparseReadRecipeIndex& recipe,
	const size_t rowgroup_index,
	const std::vector<uint64_t>& selection_words) {
	const auto found = std::lower_bound(
	    recipe.records.begin(), recipe.records.end(), rowgroup_index, [](const auto& record, const size_t index) {
		    return record.rowgroup_index < index;
	    });
	if (found == recipe.records.end() || found->rowgroup_index != rowgroup_index ||
	    found->selection_digest != sparse_selection_digest(selection_words) ||
	    found->selection_words != selection_words) {
		return nullptr;
	}
	return &*found;
}

std::shared_ptr<const SparseRowgroupAccessIndex> rehydrate_sparse_recipe_access(
	fastlanes::File& file,
	const fastlanes::RowgroupDescriptor& rowgroup,
	const SparseReadRecipeRecord& recipe,
	size_t* const metadata_bytes,
	size_t* const metadata_pread_count) {
	auto access = std::make_shared<SparseRowgroupAccessIndex>();
	access->supported     = true;
	access->index_ranges  = recipe.index_ranges;
	access->shared_ranges = recipe.shared_ranges;
	const size_t index_bytes  = total_range_bytes(access->index_ranges);
	const size_t shared_bytes = total_range_bytes(access->shared_ranges);
	if (shared_bytes > std::numeric_limits<size_t>::max() - index_bytes) {
		throw std::overflow_error("sparse read recipe static prefix byte count overflow");
	}
	access->static_prefix.resize(index_bytes + shared_bytes);
	size_t cursor = 0U;
	const auto read_ranges = [&](const std::vector<SparseByteRange>& ranges) {
		for (const auto& range : ranges) {
			file.ReadRangeUnchecked(
			    access->static_prefix.data() + cursor, rowgroup.m_offset() + range.offset, range.size);
			cursor += range.size;
			if (metadata_bytes != nullptr) {
				*metadata_bytes += range.size;
			}
			if (metadata_pread_count != nullptr) {
				++*metadata_pread_count;
			}
		}
	};
	read_ranges(access->index_ranges);
	read_ranges(access->shared_ranges);
	if (cursor != access->static_prefix.size()) {
		throw std::runtime_error("sparse read recipe static prefix rehydration mismatch");
	}
	return access;
}

struct SparseRecipePrehydrateStats {
	double wall_ms       = 0.0;
	double service_ms    = 0.0;
	size_t worker_count  = 0U;
	size_t metadata_bytes = 0U;
	size_t metadata_pread_count = 0U;
};

SparseRecipePrehydrateStats prehydrate_sparse_recipe_access(
	fastlanes::File& file,
	const fastlanes::TableDescriptor& table_descriptor,
	const SparseReadRecipeIndex& recipe,
	const size_t requested_workers) {
	SparseRecipePrehydrateStats stats;
	if (requested_workers == 0U || recipe.records.empty()) {
		return stats;
	}
	const auto* rowgroups = table_descriptor.m_rowgroup_descriptors();
	if (rowgroups == nullptr) {
		throw std::runtime_error("sparse read recipe source has no rowgroups");
	}
	// Open/cache the descriptor before workers enter ReadRangeUnchecked. pread
	// itself is offset-based and safe to issue concurrently on the shared fd.
	(void)file.Size();
	stats.worker_count = std::min(requested_workers, recipe.records.size());
	std::atomic<size_t> next_record {0U};
	std::atomic<bool>   stop {false};
	std::mutex          failure_mutex;
	std::exception_ptr  failure;
	const auto wall_begin = std::chrono::steady_clock::now();
	std::vector<std::thread> workers;
	workers.reserve(stats.worker_count);
	for (size_t worker = 0U; worker < stats.worker_count; ++worker) {
		workers.emplace_back([&]() {
			while (!stop.load(std::memory_order_acquire)) {
				const size_t index = next_record.fetch_add(1U, std::memory_order_relaxed);
				if (index >= recipe.records.size()) {
					return;
				}
				try {
					auto& record = recipe.records[index];
					const auto* rowgroup = rowgroups->Get(record.rowgroup_index);
					if (rowgroup == nullptr) {
						throw std::runtime_error("sparse read recipe rowgroup descriptor is missing");
					}
					const auto begin = std::chrono::steady_clock::now();
					record.prehydrated_access = rehydrate_sparse_recipe_access(
					    file,
					    *rowgroup,
					    record,
					    &record.metadata_bytes,
					    &record.metadata_pread_count);
					record.rehydrate_service_ms = std::chrono::duration<double, std::milli>(
					    std::chrono::steady_clock::now() - begin).count();
				} catch (...) {
					{
						std::lock_guard<std::mutex> guard(failure_mutex);
						if (!failure) {
							failure = std::current_exception();
						}
					}
					stop.store(true, std::memory_order_release);
					return;
				}
			}
		});
	}
	for (auto& worker : workers) {
		worker.join();
	}
	stats.wall_ms = std::chrono::duration<double, std::milli>(
	    std::chrono::steady_clock::now() - wall_begin).count();
	if (failure) {
		std::rethrow_exception(failure);
	}
	for (const auto& record : recipe.records) {
		stats.service_ms += record.rehydrate_service_ms;
		stats.metadata_bytes += record.metadata_bytes;
		stats.metadata_pread_count += record.metadata_pread_count;
	}
	return stats;
}

} // namespace galp::format::detail

namespace galp::format {

SparseReadBoundedCoalesceResult coalesce_sparse_read_ranges_bounded(
	const std::vector<SparseReadBoundedRowgroupInput>& inputs,
	const SparseReadBoundedCoalesceOptions&            options) {
	constexpr uint32_t kHardMaximumAmplificationPpm = 1'100'000U;
	const auto whole_ppm = options.whole_run_amplification_ppm;
	const auto shard_ppm = options.per_shard_amplification_ppm == 0U
	                           ? whole_ppm
	                           : options.per_shard_amplification_ppm;
	const auto rowgroup_ppm = options.per_rowgroup_amplification_ppm == 0U
	                              ? whole_ppm
	                              : options.per_rowgroup_amplification_ppm;
	for (const auto [value, label] :
	     {std::pair {whole_ppm, "whole-run"}, std::pair {shard_ppm, "per-shard"},
	      std::pair {rowgroup_ppm, "per-rowgroup"}}) {
		if (value < kSparseReadAmplificationScale || value > kHardMaximumAmplificationPpm) {
			throw std::invalid_argument(std::string("bounded sparse read ") + label +
			                            " amplification must be in [1.0, 1.10]");
		}
	}

	const auto checked_add = [](const size_t lhs, const size_t rhs, const char* const label) {
		if (rhs > std::numeric_limits<size_t>::max() - lhs) {
			throw std::overflow_error(std::string("bounded sparse read ") + label + " overflow");
		}
		return lhs + rhs;
	};
	const auto capped_scaled_bytes = [](const size_t exact, const uint32_t ppm, const size_t physical_limit) {
		using Wide = unsigned __int128;
		const Wide scaled = static_cast<Wide>(exact) * static_cast<Wide>(ppm) /
		                    static_cast<Wide>(kSparseReadAmplificationScale);
		return static_cast<size_t>(std::min<Wide>(scaled, physical_limit));
	};

	struct WorkingRowgroup {
		SparseReadBoundedRowgroupResult result;
		std::vector<bool>                selected_boundaries;
		std::vector<size_t>              parent;
		std::vector<size_t>              component_begin;
		std::vector<size_t>              component_end;
		size_t                           physical_bytes = 0U;
		size_t                           target_bytes   = 0U;
		size_t                           max_run_bytes  = 0U;
	};
	struct GapCandidate {
		size_t   gap_size       = 0U;
		uint32_t shard_id       = 0U;
		size_t   rowgroup_id    = 0U;
		size_t   boundary_id    = 0U;
		size_t   input_index    = 0U;
	};

	SparseReadBoundedCoalesceResult aggregate;
	aggregate.rowgroups.resize(inputs.size());
	std::vector<WorkingRowgroup> working(inputs.size());
	std::vector<GapCandidate> candidates;
	std::set<std::pair<uint32_t, size_t>> identities;
	std::map<uint32_t, size_t> shard_exact_bytes;
	std::map<uint32_t, size_t> shard_full_bytes;
	for (size_t input_index = 0U; input_index < inputs.size(); ++input_index) {
		const auto& input = inputs[input_index];
		if (!identities.emplace(input.shard_id, input.rowgroup_id).second) {
			throw std::invalid_argument("bounded sparse read contains a duplicate shard/rowgroup identity");
		}
		auto ranges = input.exact_ranges;
		ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const auto& range) {
		 return range.size == 0U;
		}), ranges.end());
		std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
			return lhs.offset < rhs.offset || (lhs.offset == rhs.offset && lhs.size < rhs.size);
		});
		std::vector<SparseReadRange> canonical;
		canonical.reserve(ranges.size());
		for (const auto& range : ranges) {
			if (range.offset > input.full_storage_bytes ||
			    range.size > input.full_storage_bytes - range.offset) {
				throw std::out_of_range("bounded sparse read exact range exceeds its rowgroup");
			}
			const size_t range_end = range.offset + range.size;
			if (canonical.empty()) {
				canonical.push_back(range);
				continue;
			}
			auto& previous = canonical.back();
			const size_t previous_end = previous.offset + previous.size;
			if (range.offset <= previous_end) {
				const size_t merged_end = std::max(previous_end, range_end);
				previous.size = merged_end - previous.offset;
			} else {
				canonical.push_back(range);
			}
		}

		auto& state = working[input_index];
		state.result.shard_id              = input.shard_id;
		state.result.rowgroup_id            = input.rowgroup_id;
		state.result.full_storage_bytes      = input.full_storage_bytes;
		state.result.exact_ranges            = std::move(canonical);
		state.result.exact_storage_bytes     = 0U;
		for (const auto& range : state.result.exact_ranges) {
			state.result.exact_storage_bytes = checked_add(
			    state.result.exact_storage_bytes, range.size, "rowgroup exact byte count");
		}
		state.physical_bytes = state.result.exact_storage_bytes;
		state.target_bytes = capped_scaled_bytes(
		    state.result.exact_storage_bytes, rowgroup_ppm, input.full_storage_bytes);
		state.max_run_bytes = options.max_physical_run_bytes == 0U
		                          ? input.full_storage_bytes
		                          : std::min(options.max_physical_run_bytes, input.full_storage_bytes);
		for (const auto& range : state.result.exact_ranges) {
			if (range.size > state.max_run_bytes) {
				throw std::invalid_argument("bounded sparse read max run is smaller than an exact extent");
			}
		}
		const size_t range_count = state.result.exact_ranges.size();
		state.selected_boundaries.assign(range_count > 0U ? range_count - 1U : 0U, false);
		state.parent.resize(range_count);
		state.component_begin.resize(range_count);
		state.component_end.resize(range_count);
		for (size_t index = 0U; index < range_count; ++index) {
			state.parent[index]          = index;
			state.component_begin[index] = state.result.exact_ranges[index].offset;
			state.component_end[index]   = state.result.exact_ranges[index].offset +
			                               state.result.exact_ranges[index].size;
			if (index + 1U < range_count) {
				const size_t next_offset = state.result.exact_ranges[index + 1U].offset;
				const size_t gap = next_offset - state.component_end[index];
				candidates.push_back({gap, input.shard_id, input.rowgroup_id, index, input_index});
			}
		}

		aggregate.exact_storage_bytes = checked_add(
		    aggregate.exact_storage_bytes, state.result.exact_storage_bytes, "whole-run exact bytes");
		aggregate.full_storage_bytes = checked_add(
		    aggregate.full_storage_bytes, input.full_storage_bytes, "whole-run full bytes");
		aggregate.exact_extent_count = checked_add(
		    aggregate.exact_extent_count, range_count, "whole-run exact extent count");
		shard_exact_bytes[input.shard_id] = checked_add(
		    shard_exact_bytes[input.shard_id], state.result.exact_storage_bytes, "shard exact bytes");
		shard_full_bytes[input.shard_id] = checked_add(
		    shard_full_bytes[input.shard_id], input.full_storage_bytes, "shard full bytes");
	}

	const size_t whole_target = capped_scaled_bytes(
	    aggregate.exact_storage_bytes, whole_ppm, aggregate.full_storage_bytes);
	size_t whole_physical = aggregate.exact_storage_bytes;
	std::map<uint32_t, size_t> shard_target;
	std::map<uint32_t, size_t> shard_physical = shard_exact_bytes;
	for (const auto& [shard_id, exact_bytes] : shard_exact_bytes) {
		shard_target[shard_id] = capped_scaled_bytes(exact_bytes, shard_ppm, shard_full_bytes.at(shard_id));
	}
	std::stable_sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
		return std::tie(lhs.gap_size, lhs.shard_id, lhs.rowgroup_id, lhs.boundary_id) <
		       std::tie(rhs.gap_size, rhs.shard_id, rhs.rowgroup_id, rhs.boundary_id);
	});
	const auto find_root = [](WorkingRowgroup& state, size_t node) {
		size_t root = node;
		while (state.parent[root] != root) {
			root = state.parent[root];
		}
		while (state.parent[node] != node) {
			const size_t next = state.parent[node];
			state.parent[node] = root;
			node = next;
		}
		return root;
	};
	for (const auto& candidate : candidates) {
		auto& state = working[candidate.input_index];
		const bool budget_ok = candidate.gap_size <= whole_target - whole_physical &&
		                       candidate.gap_size <= shard_target.at(candidate.shard_id) -
		                                                 shard_physical.at(candidate.shard_id) &&
		                       candidate.gap_size <= state.target_bytes - state.physical_bytes;
		if (!budget_ok) {
			++aggregate.budget_rejected_gap_count;
			continue;
		}
		const size_t left_root  = find_root(state, candidate.boundary_id);
		const size_t right_root = find_root(state, candidate.boundary_id + 1U);
		if (left_root == right_root) {
			throw std::logic_error("bounded sparse read boundary was selected twice");
		}
		const size_t merged_begin = std::min(state.component_begin[left_root], state.component_begin[right_root]);
		const size_t merged_end   = std::max(state.component_end[left_root], state.component_end[right_root]);
		if (merged_end - merged_begin > state.max_run_bytes) {
			++aggregate.max_run_rejected_gap_count;
			continue;
		}
		state.selected_boundaries[candidate.boundary_id] = true;
		state.parent[right_root]          = left_root;
		state.component_begin[left_root] = merged_begin;
		state.component_end[left_root]   = merged_end;
		state.physical_bytes += candidate.gap_size;
		whole_physical += candidate.gap_size;
		shard_physical[candidate.shard_id] += candidate.gap_size;
		++aggregate.selected_gap_count;
	}

	for (size_t input_index = 0U; input_index < working.size(); ++input_index) {
		auto& state = working[input_index];
		auto& output = state.result;
		if (!output.exact_ranges.empty()) {
			output.physical_ranges.push_back(output.exact_ranges.front());
			for (size_t boundary = 0U; boundary < state.selected_boundaries.size(); ++boundary) {
				const auto& next = output.exact_ranges[boundary + 1U];
				if (!state.selected_boundaries[boundary]) {
					output.physical_ranges.push_back(next);
					continue;
				}
				const auto& previous = output.exact_ranges[boundary];
				const size_t previous_end = previous.offset + previous.size;
				const size_t gap_size = next.offset - previous_end;
				output.merged_holes.push_back({previous_end, gap_size});
				auto& physical = output.physical_ranges.back();
				physical.size = next.offset + next.size - physical.offset;
			}
		}
		output.physical_storage_bytes = state.physical_bytes;
		output.merged_gap_bytes = output.physical_storage_bytes - output.exact_storage_bytes;
		size_t verified_physical_bytes = 0U;
		for (const auto& range : output.physical_ranges) {
			verified_physical_bytes = checked_add(
			    verified_physical_bytes, range.size, "verified physical byte count");
			if (range.size > state.max_run_bytes) {
				throw std::logic_error("bounded sparse read emitted an oversized physical run");
			}
		}
		if (verified_physical_bytes != output.physical_storage_bytes ||
		    output.physical_storage_bytes > state.target_bytes ||
		    output.physical_storage_bytes > output.full_storage_bytes) {
			throw std::logic_error("bounded sparse read result violates its byte budget");
		}
		aggregate.physical_storage_bytes = checked_add(
		    aggregate.physical_storage_bytes, output.physical_storage_bytes, "whole-run physical bytes");
		aggregate.merged_gap_bytes = checked_add(
		    aggregate.merged_gap_bytes, output.merged_gap_bytes, "whole-run gap bytes");
		aggregate.physical_run_count = checked_add(
		    aggregate.physical_run_count, output.physical_ranges.size(), "whole-run physical runs");
		aggregate.rowgroups[input_index] = std::move(output);
	}
	if (aggregate.physical_storage_bytes != whole_physical ||
	    aggregate.merged_gap_bytes != aggregate.physical_storage_bytes - aggregate.exact_storage_bytes ||
	    aggregate.physical_storage_bytes > whole_target) {
		throw std::logic_error("bounded sparse read whole-run accounting mismatch");
	}
	return aggregate;
}

namespace {

size_t align_compact_batch_offset(const size_t offset) {
	constexpr size_t alignment = kCompactBatchRowgroupAlignment;
	static_assert(alignment != 0U && (alignment & (alignment - 1U)) == 0U);
	if (offset > std::numeric_limits<size_t>::max() - (alignment - 1U)) {
		throw std::overflow_error("Compact v3 batch backing alignment overflow");
	}
	return (offset + alignment - 1U) & ~(alignment - 1U);
}

struct OwnedCompactRowgroupDescriptor {
	std::shared_ptr<const std::vector<uint8_t>> owner;
	const fastlanes::RowgroupDescriptor*        descriptor = nullptr;
};

OwnedCompactRowgroupDescriptor make_compact_rowgroup_descriptor(const CompactDescriptorV3& compact,
	                                                            const size_t rowgroup_index) {
	const auto native = compact.unpack_rowgroup(rowgroup_index);
	flatbuffers::FlatBufferBuilder builder;
	const auto root = fastlanes::RowgroupDescriptor::Pack(builder, native.get());
	fastlanes::FinishRowgroupDescriptorBuffer(builder, root);
	auto detached = builder.Release();
	auto bytes = std::make_shared<std::vector<uint8_t>>(detached.data(), detached.data() + detached.size());
	const auto* descriptor = fastlanes::GetRowgroupDescriptor(bytes->data());
	return {std::move(bytes), descriptor};
}

bool compact_direct_rowgroup_matches_plan(const CompactV3DirectRowgroup&          rowgroup,
	                                      const std::vector<ZeroCopyColumnPlan>& plan) {
	if (rowgroup.columns.size() != plan.size()) {
		return false;
	}
	for (size_t column_index = 0U; column_index < plan.size(); ++column_index) {
		const auto* schema = rowgroup.columns[column_index].schema;
		const auto* rpn    = schema == nullptr ? nullptr : schema->encoding_rpn();
		const auto* ops    = rpn == nullptr ? nullptr : rpn->operator_tokens();
		const auto* operands = rpn == nullptr ? nullptr : rpn->operand_tokens();
		if (ops == nullptr || ops->size() != 1U || ops->Get(0U) != plan[column_index].token) {
			return false;
		}
		const size_t operand_count = operands == nullptr ? 0U : operands->size();
		if (operand_count != plan[column_index].operand_ids.size()) {
			return false;
		}
		for (size_t operand_index = 0U; operand_index < operand_count; ++operand_index) {
			if (operands->Get(static_cast<flatbuffers::uoffset_t>(operand_index)) !=
			    plan[column_index].operand_ids[operand_index]) {
				return false;
			}
		}
	}
	return true;
}

// std::span permits an empty range, but RowgroupView and ColumnView retain a
// pointer even when a metadata-only rowgroup owns no physical bytes. Keep one
// process-lifetime, suitably aligned address for that empty range. No read or
// write is ever issued against this byte.
alignas(std::max_align_t) std::byte kMetadataOnlyBacking {};

std::byte* metadata_only_backing() noexcept {
	return &kMetadataOnlyBacking;
}

template <typename Callback>
void parallel_for_compact_views(const size_t count, const size_t requested_workers, Callback&& callback) {
	const size_t worker_count = std::min(count, std::max<size_t>(1U, requested_workers));
	if (worker_count <= 1U) {
		for (size_t index = 0U; index < count; ++index) {
			callback(index);
		}
		return;
	}
	std::atomic<size_t>        next {0U};
	std::vector<std::thread>   workers;
	std::vector<std::exception_ptr> errors(worker_count);
	workers.reserve(worker_count);
	for (size_t worker = 0U; worker < worker_count; ++worker) {
		workers.emplace_back([&, worker]() {
			try {
				while (true) {
					const auto index = next.fetch_add(1U, std::memory_order_relaxed);
					if (index >= count) {
						return;
					}
					callback(index);
				}
			} catch (...) { errors[worker] = std::current_exception(); }
		});
	}
	for (auto& worker : workers) {
		worker.join();
	}
	for (const auto& error : errors) {
		if (error) {
			std::rethrow_exception(error);
		}
	}
}

void append_recipe_bytes(std::vector<std::byte>& destination, const void* const source, const size_t size) {
	if (size > std::numeric_limits<size_t>::max() - destination.size()) {
		throw std::overflow_error("sparse read recipe size overflow");
	}
	const auto begin = destination.size();
	destination.resize(begin + size);
	if (size != 0U) {
		std::memcpy(destination.data() + begin, source, size);
	}
}

template <typename T>
void append_recipe_le(std::vector<std::byte>& destination, const T value) {
	const auto begin = destination.size();
	destination.resize(begin + sizeof(T));
	detail::put_recipe_le(destination, begin, value);
}

void append_recipe_varuint32(std::vector<std::byte>& destination, uint32_t value) {
	do {
		auto byte = static_cast<uint8_t>(value & 0x7FU);
		value >>= 7U;
		if (value != 0U) {
			byte |= 0x80U;
		}
		destination.push_back(static_cast<std::byte>(byte));
	} while (value != 0U);
}

void append_recipe_ranges(std::vector<std::byte>& destination,
	                      const std::vector<detail::SparseByteRange>& ranges,
	                      const size_t rowgroup_bytes) {
	size_t previous_end = 0U;
	for (const auto& range : ranges) {
		if (range.size == 0U || range.offset > rowgroup_bytes || range.size > rowgroup_bytes - range.offset ||
		    range.offset > std::numeric_limits<uint32_t>::max() ||
		    range.size > std::numeric_limits<uint32_t>::max() ||
		    (!ranges.empty() && range.offset < previous_end)) {
			throw std::runtime_error("sparse read recipe range is invalid or exceeds uint32 encoding");
		}
		const auto delta = range.offset - previous_end;
		if (previous_end != 0U && delta == 0U) {
			throw std::runtime_error("sparse read recipe ranges must not overlap or be adjacent");
		}
		append_recipe_varuint32(destination, static_cast<uint32_t>(delta));
		append_recipe_varuint32(destination, static_cast<uint32_t>(range.size));
		previous_end = range.offset + range.size;
	}
}

class SparseRecipeWriterLock {
public:
	explicit SparseRecipeWriterLock(const std::filesystem::path& recipe_path) {
		if (recipe_path.empty()) {
			throw std::invalid_argument("sparse read recipe output path is empty");
		}
		const auto directory = recipe_path.parent_path().empty()
		                           ? std::filesystem::current_path()
		                           : recipe_path.parent_path();
		std::filesystem::create_directories(directory);
#if !defined(_WIN32)
		fd_ = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
		if (fd_ < 0) {
			throw std::runtime_error("failed to open sparse read recipe directory for single-flight: " +
			                         std::string(std::strerror(errno)));
		}
		while (::flock(fd_, LOCK_EX) != 0) {
			if (errno == EINTR) {
				continue;
			}
			const auto message = std::string(std::strerror(errno));
			static_cast<void>(::close(fd_));
			fd_ = -1;
			throw std::runtime_error("failed to lock sparse read recipe directory: " + message);
		}
#endif
	}

	~SparseRecipeWriterLock() {
#if !defined(_WIN32)
		if (fd_ >= 0) {
			static_cast<void>(::flock(fd_, LOCK_UN));
			static_cast<void>(::close(fd_));
		}
#endif
	}

	SparseRecipeWriterLock(const SparseRecipeWriterLock&)            = delete;
	SparseRecipeWriterLock& operator=(const SparseRecipeWriterLock&) = delete;

private:
#if !defined(_WIN32)
	int fd_ = -1;
#endif
};

void write_recipe_file_atomic(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
	if (path.empty()) {
		throw std::invalid_argument("sparse read recipe output path is empty");
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
#if defined(_WIN32)
	const auto staged = std::filesystem::path(path.string() + ".tmp");
	{
		std::ofstream output(staged, std::ios::binary | std::ios::trunc);
		output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		output.close();
		if (!output) {
			throw std::runtime_error("failed to write sparse read recipe staging file");
		}
	}
	std::filesystem::rename(staged, path);
#else
	const auto staged = std::filesystem::path(path.string() + ".tmp." + std::to_string(::getpid()));
	const int fd = ::open(staged.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
	if (fd < 0) {
		throw std::runtime_error("failed to create sparse read recipe staging file: " +
		                         std::string(std::strerror(errno)));
	}
	bool committed = false;
	bool fd_open   = true;
	try {
		size_t written = 0U;
		while (written < bytes.size()) {
			const auto result = ::write(fd, bytes.data() + written, bytes.size() - written);
			if (result < 0) {
				if (errno == EINTR) {
					continue;
				}
				throw std::runtime_error("failed to write sparse read recipe staging file: " +
				                         std::string(std::strerror(errno)));
			}
			if (result == 0) {
				throw std::runtime_error("short write while writing sparse read recipe staging file");
			}
			written += static_cast<size_t>(result);
		}
		if (::fsync(fd) != 0) {
			throw std::runtime_error("failed to fsync sparse read recipe staging file: " +
			                         std::string(std::strerror(errno)));
		}
		if (::close(fd) != 0) {
			throw std::runtime_error("failed to close sparse read recipe staging file: " +
			                         std::string(std::strerror(errno)));
		}
		fd_open = false;
		if (::rename(staged.c_str(), path.c_str()) != 0) {
			throw std::runtime_error("failed to atomically install sparse read recipe: " +
			                         std::string(std::strerror(errno)));
		}
		if (!path.parent_path().empty()) {
			const int directory_fd = ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
			if (directory_fd >= 0) {
				static_cast<void>(::fsync(directory_fd));
				static_cast<void>(::close(directory_fd));
			}
		}
		committed = true;
	} catch (...) {
		if (fd_open) {
			static_cast<void>(::close(fd));
		}
		std::error_code ignored;
		std::filesystem::remove(staged, ignored);
		throw;
	}
	if (!committed) {
		throw std::runtime_error("sparse read recipe atomic commit failed");
	}
#endif
}

} // namespace

size_t FlsReader::compact_batch_backing_bytes(const std::vector<size_t>& rowgroup_indices) const {
	if (m_compact_descriptor == nullptr) {
		throw std::invalid_argument("compact batch backing sizing requires Compact v3");
	}
	size_t cursor = 0U;
	for (const size_t rowgroup_index : rowgroup_indices) {
		const auto record = m_compact_descriptor->rowgroup(rowgroup_index);
		if (record.payload_size == 0U) {
			continue;
		}
		cursor = align_compact_batch_offset(cursor);
		if (record.payload_size > std::numeric_limits<size_t>::max() - cursor) {
			throw std::overflow_error("Compact v3 batch backing size overflow");
		}
		cursor += record.payload_size;
	}
	return cursor;
}

std::vector<size_t> FlsReader::compact_largest_image_backing_bytes(const size_t limit) const {
	if (m_compact_descriptor == nullptr) {
		throw std::invalid_argument("compact image backing sizing requires Compact v3");
	}
	if (limit == 0U) {
		return {};
	}
	std::vector<size_t> largest;
	largest.reserve(std::min(limit, m_compact_descriptor->image_count()));
	for (size_t image_index = 0U; image_index < m_compact_descriptor->image_count(); ++image_index) {
		const auto image = m_compact_descriptor->image(image_index);
		size_t     bytes = 0U;
		for (size_t local = 0U; local < image.rowgroup_count; ++local) {
			const size_t rowgroup_index = static_cast<size_t>(image.first_rowgroup) + local;
			const auto   record         = m_compact_descriptor->rowgroup(rowgroup_index);
			if (record.payload_size == 0U) {
				continue;
			}
			bytes = align_compact_batch_offset(bytes);
			if (record.payload_size > std::numeric_limits<size_t>::max() - bytes) {
				throw std::overflow_error("Compact v3 image backing size overflow");
			}
			bytes += record.payload_size;
		}
		bytes = align_compact_batch_offset(bytes);
		if (largest.size() < limit) {
			largest.push_back(bytes);
			std::push_heap(largest.begin(), largest.end(), std::greater<size_t> {});
		} else if (bytes > largest.front()) {
			std::pop_heap(largest.begin(), largest.end(), std::greater<size_t> {});
			largest.back() = bytes;
			std::push_heap(largest.begin(), largest.end(), std::greater<size_t> {});
		}
	}
	std::sort(largest.begin(), largest.end(), std::greater<size_t> {});
	// Static metadata prewarm deliberately evicts descriptor pages after
	// validation. The dataset-wide contract scan faults image/rowgroup records
	// back in, so drop those pages again instead of retaining several GiB across
	// the full training manifest.
	m_compact_descriptor->release_resident_pages();
	return largest;
}

struct SparseVectorReadPlan::Impl {
	enum class Strategy {
		kFullRowgroup,
		kSourceRanges,
		kBoundedSourceRanges,
		kBundleRuns,
		kBundleEnvelope,
		kBundlePacked,
	};
	struct BundleRun {
		uint64_t                             file_offset   = 0U;
		size_t                               size          = 0U;
		size_t                               packed_offset = 0U;
		std::vector<detail::SparseByteRange> logical_ranges;
	};
	struct EnvelopeCopy {
		size_t source_offset  = 0U;
		size_t logical_offset = 0U;
		size_t size           = 0U;
	};

	std::shared_ptr<const uint8_t> owner;
	std::shared_ptr<const detail::SparseRowgroupAccessIndex> access;
	size_t                         rowgroup_index       = 0U;
	size_t                         rowgroup_bytes       = 0U;
	size_t                         full_vector_count    = 0U;
	size_t                         selected_vector_count = 0U;
	size_t                         storage_bytes        = 0U;
	size_t                         selected_storage_bytes = 0U;
	std::vector<uint8_t>           materialized_columns;
	Strategy                       strategy             = Strategy::kFullRowgroup;
	std::string                    fallback_reason;
	std::vector<detail::SparseByteRange> exact_source_ranges;
	std::vector<detail::SparseByteRange> source_ranges;
	std::vector<detail::SparseByteRange> merged_holes;
	size_t                                merged_gap_bytes = 0U;
	SubmissionBackend                      submission_backend = SubmissionBackend::kSynchronousPread;
	uint32_t                               io_uring_queue_depth = 0U;
	std::vector<BundleRun>               bundle_runs;
	uint64_t                              envelope_file_offset = 0U;
	size_t                                envelope_size        = 0U;
	std::vector<EnvelopeCopy>             envelope_copies;
	size_t                                packed_bytes = 0U;
	std::vector<galp::execution::PackedRowgroupScatterRange> packed_scatter_ranges;
	bool                                  recipe_hit              = false;
	double                                recipe_lookup_ms        = 0.0;
	double                                recipe_rehydrate_ms     = 0.0;
	size_t                                recipe_source_metadata_bytes = 0U;
	size_t                                recipe_source_metadata_pread_count = 0U;
	double                                endpoint_resolution_ms  = 0.0;
	double                                range_gather_ms          = 0.0;
	double                                range_sort_coalesce_ms   = 0.0;
};

size_t SparseVectorReadPlan::rowgroup_index() const noexcept {
	return impl_ ? impl_->rowgroup_index : 0U;
}

size_t SparseVectorReadPlan::selected_vector_count() const noexcept {
	return impl_ ? impl_->selected_vector_count : 0U;
}

size_t SparseVectorReadPlan::storage_bytes() const noexcept {
	return impl_ ? impl_->storage_bytes : 0U;
}

size_t SparseVectorReadPlan::full_storage_bytes() const noexcept {
	return impl_ ? impl_->rowgroup_bytes : 0U;
}

size_t SparseVectorReadPlan::merged_gap_bytes() const noexcept {
	return impl_ ? impl_->merged_gap_bytes : 0U;
}

size_t SparseVectorReadPlan::estimated_pread_count() const noexcept {
	if (!impl_) {
		return 0U;
	}
	switch (impl_->strategy) {
	case Impl::Strategy::kFullRowgroup:
	case Impl::Strategy::kBundleEnvelope:
		return 1U;
	case Impl::Strategy::kSourceRanges:
	case Impl::Strategy::kBoundedSourceRanges:
		return impl_->source_ranges.size();
	case Impl::Strategy::kBundleRuns:
	case Impl::Strategy::kBundlePacked:
		// Each contiguous bundle run is submitted independently.  A short
		// preadv may add another syscall, so this is a lower-bound estimate.
		return impl_->bundle_runs.size();
	}
	return 0U;
}

bool SparseVectorReadPlan::recipe_hit() const noexcept {
	return impl_ && impl_->recipe_hit;
}

double SparseVectorReadPlan::recipe_lookup_ms() const noexcept {
	return impl_ ? impl_->recipe_lookup_ms : 0.0;
}

double SparseVectorReadPlan::recipe_rehydrate_ms() const noexcept {
	return impl_ ? impl_->recipe_rehydrate_ms : 0.0;
}

size_t SparseVectorReadPlan::recipe_source_metadata_bytes() const noexcept {
	return impl_ ? impl_->recipe_source_metadata_bytes : 0U;
}

size_t SparseVectorReadPlan::recipe_source_metadata_pread_count() const noexcept {
	return impl_ ? impl_->recipe_source_metadata_pread_count : 0U;
}

double SparseVectorReadPlan::endpoint_resolution_ms() const noexcept {
	return impl_ ? impl_->endpoint_resolution_ms : 0.0;
}

double SparseVectorReadPlan::range_gather_ms() const noexcept {
	return impl_ ? impl_->range_gather_ms : 0.0;
}

double SparseVectorReadPlan::range_sort_coalesce_ms() const noexcept {
	return impl_ ? impl_->range_sort_coalesce_ms : 0.0;
}

std::vector<SparseReadRange> SparseVectorReadPlan::exact_source_ranges() const {
	std::vector<SparseReadRange> result;
	if (!impl_) {
		return result;
	}
	const auto& ranges = impl_->exact_source_ranges.empty() ? impl_->source_ranges : impl_->exact_source_ranges;
	result.reserve(ranges.size());
	for (const auto& range : ranges) {
		result.push_back({range.offset, range.size});
	}
	return result;
}

std::vector<SparseReadRange> SparseVectorReadPlan::physical_source_ranges() const {
	std::vector<SparseReadRange> result;
	if (!impl_) {
		return result;
	}
	result.reserve(impl_->source_ranges.size());
	for (const auto& range : impl_->source_ranges) {
		result.push_back({range.offset, range.size});
	}
	return result;
}

std::vector<SparseReadRange> SparseVectorReadPlan::merged_hole_ranges() const {
	std::vector<SparseReadRange> result;
	if (!impl_) {
		return result;
	}
	result.reserve(impl_->merged_holes.size());
	for (const auto& range : impl_->merged_holes) {
		result.push_back({range.offset, range.size});
	}
	return result;
}

SparseVectorReadPlan SparseVectorReadPlan::with_bounded_coalescing(
	const SparseReadBoundedRowgroupResult& result,
	const SubmissionBackend submission_backend,
	const uint32_t io_uring_queue_depth) const {
	if (!impl_) {
		throw std::invalid_argument("cannot bound an empty sparse vector read plan");
	}
	if (impl_->strategy != Impl::Strategy::kSourceRanges) {
		throw std::invalid_argument("bounded coalescing requires an exact source-range plan");
	}
	if (result.rowgroup_id != impl_->rowgroup_index || result.full_storage_bytes != impl_->rowgroup_bytes ||
	    result.exact_storage_bytes != impl_->selected_storage_bytes) {
		throw std::invalid_argument("bounded coalescing result does not match its sparse vector read plan");
	}
	if ((submission_backend == SubmissionBackend::kIoUring) != (io_uring_queue_depth != 0U)) {
		throw std::invalid_argument("io_uring bounded reads require a positive queue depth only for io_uring mode");
	}
	const auto exact = exact_source_ranges();
	if (exact.size() != result.exact_ranges.size()) {
		throw std::invalid_argument("bounded coalescing changed the exact range count");
	}
	for (size_t index = 0U; index < exact.size(); ++index) {
		if (exact[index].offset != result.exact_ranges[index].offset ||
		    exact[index].size != result.exact_ranges[index].size) {
			throw std::invalid_argument("bounded coalescing changed an exact source range");
		}
	}
	auto bounded = std::make_shared<Impl>(*impl_);
	bounded->strategy = Impl::Strategy::kBoundedSourceRanges;
	bounded->source_ranges.clear();
	bounded->source_ranges.reserve(result.physical_ranges.size());
	for (const auto& range : result.physical_ranges) {
		bounded->source_ranges.push_back({range.offset, range.size});
	}
	bounded->merged_holes.clear();
	bounded->merged_holes.reserve(result.merged_holes.size());
	for (const auto& range : result.merged_holes) {
		bounded->merged_holes.push_back({range.offset, range.size});
	}
	bounded->storage_bytes   = result.physical_storage_bytes;
	bounded->merged_gap_bytes = result.merged_gap_bytes;
	bounded->submission_backend = submission_backend;
	bounded->io_uring_queue_depth = io_uring_queue_depth;
	return SparseVectorReadPlan(std::move(bounded));
}

SparseVectorReadPlan::SubmissionBackend SparseVectorReadPlan::submission_backend() const noexcept {
	return impl_ ? impl_->submission_backend : SubmissionBackend::kSynchronousPread;
}

uint32_t SparseVectorReadPlan::io_uring_queue_depth() const noexcept {
	return impl_ ? impl_->io_uring_queue_depth : 0U;
}

SparseVectorReadPlan::Backend SparseVectorReadPlan::backend() const noexcept {
	if (!impl_) {
		return Backend::kFullRowgroup;
	}
	switch (impl_->strategy) {
	case Impl::Strategy::kFullRowgroup:
		return Backend::kFullRowgroup;
	case Impl::Strategy::kSourceRanges:
		return Backend::kSourceRanges;
	case Impl::Strategy::kBoundedSourceRanges:
		return Backend::kBoundedSourceRanges;
	case Impl::Strategy::kBundleRuns:
		return Backend::kBundleRuns;
	case Impl::Strategy::kBundleEnvelope:
		return Backend::kBundleEnvelope;
	case Impl::Strategy::kBundlePacked:
		return Backend::kBundlePacked;
	}
	return Backend::kFullRowgroup;
}

bool SparseVectorReadPlan::uses_sparse_read() const noexcept {
	return impl_ && impl_->strategy != Impl::Strategy::kFullRowgroup;
}

bool SparseVectorReadPlan::uses_packed_device_scatter() const noexcept {
	return impl_ && impl_->strategy == Impl::Strategy::kBundlePacked;
}

std::filesystem::path sparse_vector_bundle_path(const std::filesystem::path& fls_path) {
	auto path = fls_path;
	path.replace_extension(".svb");
	return path;
}

void write_sparse_vector_bundle(const std::filesystem::path& fls_path,
	                            const std::filesystem::path& bundle_path) {
	if (std::filesystem::exists(bundle_path)) {
		throw std::runtime_error("sparse vector bundle output already exists: " + bundle_path.string());
	}
	const auto staged_path = std::filesystem::path(bundle_path.string() + ".tmp");
	if (std::filesystem::exists(staged_path)) {
		throw std::runtime_error("stale sparse vector bundle staging file exists: " + staged_path.string());
	}

	fastlanes::File source(fls_path);
	auto table_handle = detail::load_table_descriptor(source, fls_path);
	const auto* table = table_handle.Get();
	if (table == nullptr || table->m_rowgroup_descriptors() == nullptr) {
		throw std::runtime_error("FLS table has no rowgroup descriptors for sparse vector bundling");
	}
	const auto* rowgroups = table->m_rowgroup_descriptors();
	std::vector<detail::SparseVectorBundleRowgroup> directory;
	directory.reserve(rowgroups->size());

	try {
		std::ofstream output(staged_path, std::ios::binary | std::ios::trunc);
		if (!output) {
			throw std::runtime_error("failed to open sparse vector bundle output: " + staged_path.string());
		}
		std::array<std::byte, detail::kSparseVectorBundleHeaderSize> empty_header {};
		output.write(reinterpret_cast<const char*>(empty_header.data()),
		             static_cast<std::streamsize>(empty_header.size()));

		for (flatbuffers::uoffset_t rowgroup_index = 0; rowgroup_index < rowgroups->size(); ++rowgroup_index) {
			const auto* rowgroup = rowgroups->Get(rowgroup_index);
			if (rowgroup == nullptr) {
				throw std::runtime_error("null rowgroup descriptor while building sparse vector bundle");
			}
			const size_t rowgroup_bytes = static_cast<size_t>(rowgroup->m_size());
			std::vector<std::byte> backing(rowgroup_bytes);
			source.ReadRangeUnchecked(backing.data(), rowgroup->m_offset(), rowgroup->m_size());
			const auto segments = detail::rowgroup_segment_descriptors(*rowgroup);
			std::string capability_reason;
			if (!detail::validate_sparse_vector_segments(*rowgroup, segments, &capability_reason)) {
				throw std::runtime_error("cannot build sparse vector bundle for rowgroup " +
				                         std::to_string(rowgroup_index) + ": " + capability_reason);
			}

			detail::SparseVectorBundleRowgroup entry;
			entry.source_offset = rowgroup->m_offset();
			entry.source_size   = rowgroup->m_size();
			entry.n_vecs        = rowgroup->m_n_vec();
			entry.prefix_offset = static_cast<uint64_t>(output.tellp());
			const auto index_ranges  = detail::segment_index_ranges(segments);
			const auto shared_ranges = detail::segment_shared_ranges(segments, backing.data());
			auto       prefix        = detail::pack_ranges(backing.data(), index_ranges);
			const auto shared        = detail::pack_ranges(backing.data(), shared_ranges);
			prefix.insert(prefix.end(), shared.begin(), shared.end());
			output.write(reinterpret_cast<const char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
			if (!output) {
				throw std::runtime_error("failed to write sparse vector bundle rowgroup prefix");
			}
			entry.prefix_size = static_cast<uint64_t>(output.tellp()) - entry.prefix_offset;
			entry.vectors_offset = static_cast<uint64_t>(output.tellp());
			entry.vector_offsets.reserve(static_cast<size_t>(entry.n_vecs) + 1U);
			entry.vector_offsets.push_back(0U);
			for (uint32_t vector = 0; vector < entry.n_vecs; ++vector) {
				std::vector<detail::SparseByteRange> vector_ranges;
				vector_ranges.reserve(segments.size());
				for (const auto* segment : segments) {
					if (detail::segment_entrypoint_count(*segment) == 1U) {
						continue;
					}
					vector_ranges.push_back(detail::segment_vector_range(*segment, backing.data(), vector));
				}
				const auto vector_payload = detail::pack_ranges(backing.data(), vector_ranges);
				output.write(reinterpret_cast<const char*>(vector_payload.data()),
				             static_cast<std::streamsize>(vector_payload.size()));
				if (!output) {
					throw std::runtime_error("failed to write sparse vector bundle vector payload");
				}
				entry.vector_offsets.push_back(static_cast<uint64_t>(output.tellp()) - entry.vectors_offset);
			}
			directory.push_back(std::move(entry));
		}

		const uint64_t directory_offset = static_cast<uint64_t>(output.tellp());
		std::vector<std::byte> directory_bytes;
		for (const auto& entry : directory) {
			detail::append_u64(directory_bytes, entry.source_offset);
			detail::append_u64(directory_bytes, entry.source_size);
			detail::append_u64(directory_bytes, entry.prefix_offset);
			detail::append_u64(directory_bytes, entry.prefix_size);
			detail::append_u64(directory_bytes, entry.vectors_offset);
			detail::append_u32(directory_bytes, entry.n_vecs);
			detail::append_u32(directory_bytes, 0U);
			for (const auto offset : entry.vector_offsets) {
				detail::append_u64(directory_bytes, offset);
			}
		}
		output.write(reinterpret_cast<const char*>(directory_bytes.data()),
		             static_cast<std::streamsize>(directory_bytes.size()));
		if (!output) {
			throw std::runtime_error("failed to write sparse vector bundle directory");
		}

		std::vector<std::byte> header;
		header.insert(header.end(), detail::kSparseVectorBundleMagic.begin(), detail::kSparseVectorBundleMagic.end());
		detail::append_u32(header, detail::kSparseVectorBundleVersion);
		detail::append_u32(header, static_cast<uint32_t>(directory.size()));
		detail::append_u64(header, source.Size());
		detail::append_u64(header, directory_offset);
		detail::append_u64(header, directory_bytes.size());
		header.resize(detail::kSparseVectorBundleHeaderSize, std::byte {0});
		output.seekp(0, std::ios::beg);
		output.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
		output.close();
		if (!output) {
			throw std::runtime_error("failed to finalize sparse vector bundle: " + staged_path.string());
		}
		std::filesystem::rename(staged_path, bundle_path);
	} catch (...) {
		std::error_code ignored;
		std::filesystem::remove(staged_path, ignored);
		throw;
	}
}

SparseReadRecipeWriteStats write_sparse_read_recipe(
	const std::filesystem::path&                   fls_path,
	const std::filesystem::path&                   recipe_path,
	const uint64_t                                 source_fingerprint,
	const std::vector<SparseReadRecipeSelection>& input_selections) {
	static std::mutex writer_mutex;
	std::lock_guard<std::mutex> writer_guard(writer_mutex);
	if (input_selections.empty()) {
		throw std::invalid_argument("sparse read recipe requires at least one rowgroup selection");
	}
	SparseRecipeWriterLock writer_lock(recipe_path);
	fastlanes::File source(fls_path);
	const auto source_stat_digest = detail::sparse_recipe_source_stat_digest(fls_path);
	auto descriptor_owner = detail::load_table_descriptor(source, fls_path);
	const auto* table = descriptor_owner.Get();
	const auto* rowgroups = table == nullptr ? nullptr : table->m_rowgroup_descriptors();
	if (rowgroups == nullptr) {
		throw std::runtime_error("sparse read recipe source has no rowgroup descriptors");
	}
	auto selections = input_selections;
	std::sort(selections.begin(), selections.end(), [](const auto& lhs, const auto& rhs) {
		return lhs.rowgroup_index < rhs.rowgroup_index;
	});
	for (size_t index = 0U; index < selections.size(); ++index) {
		auto& selection = selections[index];
		if (selection.rowgroup_index >= rowgroups->size() ||
		    (index != 0U && selection.rowgroup_index == selections[index - 1U].rowgroup_index)) {
			throw std::invalid_argument("sparse read recipe rowgroups must be unique and in range");
		}
		std::sort(selection.selected_vectors.begin(), selection.selected_vectors.end());
		selection.selected_vectors.erase(
		    std::unique(selection.selected_vectors.begin(), selection.selected_vectors.end()),
		    selection.selected_vectors.end());
		if (selection.selected_vectors.empty()) {
			throw std::invalid_argument("sparse read recipe rowgroup selection is empty");
		}
	}

	// The builder is the only writer. Under the cross-process directory lock,
	// accept an existing file only when its source identity, descriptor ABI, and
	// every canonical selection match. Invalid/truncated files are rebuilt below;
	// runtime readers remain strictly read-only.
	try {
		double ignored_load_ms       = 0.0;
		double ignored_validation_ms = 0.0;
		const auto existing = detail::load_sparse_read_recipe_index(
		    recipe_path,
		    fls_path,
		    *table,
		    source.Size(),
		    source_fingerprint,
		    &ignored_load_ms,
		    &ignored_validation_ms);
		bool selections_match = existing != nullptr &&
		                        (source_fingerprint == 0U || existing->source_fingerprint == source_fingerprint) &&
		                        existing->records.size() == selections.size();
		if (selections_match) {
			for (size_t index = 0U; index < selections.size(); ++index) {
				const auto& selection = selections[index];
				const auto* rowgroup = rowgroups->Get(
				    static_cast<flatbuffers::uoffset_t>(selection.rowgroup_index));
				if (rowgroup == nullptr || existing->records[index].rowgroup_index != selection.rowgroup_index ||
				    existing->records[index].selection_words !=
				        detail::sparse_selection_words(rowgroup->m_n_vec(), selection.selected_vectors)) {
					selections_match = false;
					break;
				}
			}
		}
		if (selections_match) {
			SparseReadRecipeWriteStats reused;
			reused.rowgroup_count       = existing->records.size();
			reused.sidecar_bytes        = existing->sidecar_bytes;
			reused.source_fingerprint   = existing->source_fingerprint;
			reused.source_stat_digest   = existing->source_stat_digest;
			reused.descriptor_digest    = existing->descriptor_digest;
			reused.sidecar_crc64        = existing->sidecar_crc64;
			reused.reused_existing      = true;
			reused.rowgroups.reserve(existing->records.size());
			for (const auto& record : existing->records) {
				for (const auto word : record.selection_words) {
					reused.selected_vector_count += static_cast<size_t>(std::popcount(word));
				}
				reused.exact_range_count += record.source_ranges.size();
				reused.exact_storage_bytes += record.selected_storage_bytes;
				SparseReadRecipeRowgroupStats rowgroup_stats;
				rowgroup_stats.rowgroup_index         = record.rowgroup_index;
				rowgroup_stats.rowgroup_storage_bytes = record.rowgroup_bytes;
				rowgroup_stats.exact_ranges.reserve(record.source_ranges.size());
				for (const auto& range : record.source_ranges) {
					rowgroup_stats.exact_ranges.push_back({range.offset, range.size});
				}
				reused.rowgroups.push_back(std::move(rowgroup_stats));
			}
			return reused;
		}
	} catch (const std::exception&) {
		// A builder call is an explicit rebuild request. The staged write below
		// replaces a rejected file atomically; runtime loading still rejects it.
	}

	auto lazy_index = detail::build_sparse_dataset_access_index(*table);
	std::vector<detail::SparseReadRecipeRecord> records;
	records.reserve(selections.size());
	SparseReadRecipeWriteStats stats;
	stats.source_fingerprint = source_fingerprint == 0U
	                               ? detail::sparse_recipe_crc64_file(source)
	                               : source_fingerprint;
	stats.source_stat_digest = source_stat_digest;
	stats.descriptor_digest  = detail::sparse_descriptor_digest(*table);
	for (const auto& selection : selections) {
		const auto* rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(selection.rowgroup_index));
		if (rowgroup == nullptr) {
			throw std::runtime_error("sparse read recipe selected rowgroup descriptor is null");
		}
		auto access = detail::sparse_rowgroup_access(source, *table, lazy_index, selection.rowgroup_index);
		if (!access->supported) {
			throw std::runtime_error("sparse read recipe source rowgroup is unsupported: " + access->fallback_reason);
		}
		if (selection.selected_vectors.back() >= rowgroup->m_n_vec()) {
			throw std::out_of_range("sparse read recipe selected vector is out of range");
		}
		detail::SparseReadRecipeRecord record;
		record.rowgroup_index   = static_cast<uint32_t>(selection.rowgroup_index);
		record.rowgroup_bytes   = rowgroup->m_size();
		record.selection_words  = detail::sparse_selection_words(rowgroup->m_n_vec(), selection.selected_vectors);
		record.selection_digest = detail::sparse_selection_digest(record.selection_words);
		record.index_ranges     = access->index_ranges;
		record.shared_ranges    = access->shared_ranges;
		std::vector<detail::SparseByteRange> source_ranges;
		for (const auto vector : selection.selected_vectors) {
			const auto& ranges = access->vector_ranges.at(vector);
			source_ranges.insert(source_ranges.end(), ranges.begin(), ranges.end());
		}
		record.source_ranges = detail::coalesce_ranges(std::move(source_ranges));
		record.selected_storage_bytes = detail::total_range_bytes(record.source_ranges);
		stats.selected_vector_count += selection.selected_vectors.size();
		stats.exact_range_count += record.source_ranges.size();
		stats.exact_storage_bytes += record.selected_storage_bytes;
		SparseReadRecipeRowgroupStats rowgroup_stats;
		rowgroup_stats.rowgroup_index         = record.rowgroup_index;
		rowgroup_stats.rowgroup_storage_bytes = record.rowgroup_bytes;
		rowgroup_stats.exact_ranges.reserve(record.source_ranges.size());
		for (const auto& range : record.source_ranges) {
			rowgroup_stats.exact_ranges.push_back({range.offset, range.size});
		}
		stats.rowgroups.push_back(std::move(rowgroup_stats));
		records.push_back(std::move(record));
	}
	stats.rowgroup_count = records.size();

	const size_t directory_bytes = records.size() * detail::kSparseReadRecipeRecordSize;
	std::vector<std::byte> encoded(detail::kSparseReadRecipeHeaderSize + directory_bytes, std::byte {0});
	std::copy(detail::kSparseReadRecipeMagic.begin(), detail::kSparseReadRecipeMagic.end(), encoded.begin());
	detail::put_recipe_le<uint32_t>(encoded, 8U, detail::kSparseReadRecipeVersion);
	detail::put_recipe_le<uint32_t>(encoded, 12U, detail::kSparseReadRecipeEndianMarker);
	detail::put_recipe_le<uint32_t>(encoded, 16U, detail::kSparseReadRecipeHeaderSize);
	detail::put_recipe_le<uint32_t>(encoded, 20U, detail::kSparseReadRecipeRecordSize);
	detail::put_recipe_le<uint64_t>(encoded, 24U, source.Size());
	detail::put_recipe_le<uint64_t>(encoded, 32U, stats.source_fingerprint);
	detail::put_recipe_le<uint64_t>(encoded, 40U, stats.descriptor_digest);
	detail::put_recipe_le<uint32_t>(encoded, 48U, static_cast<uint32_t>(records.size()));
	detail::put_recipe_le<uint64_t>(encoded, 56U, detail::kSparseReadRecipeHeaderSize);
	detail::put_recipe_le<uint64_t>(encoded, 64U, directory_bytes);
	detail::put_recipe_le<uint64_t>(encoded, 72U, encoded.size());
	detail::put_recipe_le<uint64_t>(encoded, 96U, stats.exact_storage_bytes);
	detail::put_recipe_le<uint64_t>(encoded, 104U, stats.exact_range_count);
	detail::put_recipe_le<uint64_t>(encoded, 112U, stats.source_stat_digest);
	for (size_t record_index = 0U; record_index < records.size(); ++record_index) {
		const auto& record = records[record_index];
		std::vector<std::byte> payload;
		for (const auto word : record.selection_words) {
			append_recipe_le<uint64_t>(payload, word);
		}
		append_recipe_ranges(payload, record.index_ranges, record.rowgroup_bytes);
		append_recipe_ranges(payload, record.shared_ranges, record.rowgroup_bytes);
		append_recipe_ranges(payload, record.source_ranges, record.rowgroup_bytes);
		const auto payload_offset = encoded.size();
		append_recipe_bytes(encoded, payload.data(), payload.size());
		const auto base = detail::kSparseReadRecipeHeaderSize +
		                  record_index * detail::kSparseReadRecipeRecordSize;
		detail::put_recipe_le<uint32_t>(encoded, base, record.rowgroup_index);
		size_t selected_count = 0U;
		for (const auto word : record.selection_words) {
			selected_count += static_cast<size_t>(std::popcount(word));
		}
		detail::put_recipe_le<uint32_t>(encoded, base + 4U, static_cast<uint32_t>(selected_count));
		detail::put_recipe_le<uint64_t>(encoded, base + 8U, record.rowgroup_bytes);
		detail::put_recipe_le<uint64_t>(encoded, base + 16U, record.selected_storage_bytes);
		detail::put_recipe_le<uint32_t>(encoded, base + 24U, static_cast<uint32_t>(record.selection_words.size()));
		detail::put_recipe_le<uint32_t>(encoded, base + 28U, static_cast<uint32_t>(record.index_ranges.size()));
		detail::put_recipe_le<uint32_t>(encoded, base + 32U, static_cast<uint32_t>(record.shared_ranges.size()));
		detail::put_recipe_le<uint32_t>(encoded, base + 36U, static_cast<uint32_t>(record.source_ranges.size()));
		detail::put_recipe_le<uint64_t>(encoded, base + 40U, payload_offset);
		detail::put_recipe_le<uint64_t>(encoded, base + 48U, payload.size());
		detail::put_recipe_le<uint64_t>(encoded, base + 56U, record.selection_digest);
		detail::put_recipe_le<uint64_t>(
		    encoded, base + 64U, detail::sparse_recipe_crc64_update(0U, payload.data(), payload.size()));
	}
	detail::put_recipe_le<uint64_t>(
	    encoded, 80U, encoded.size() - detail::kSparseReadRecipeHeaderSize - directory_bytes);
	stats.sidecar_crc64 = detail::sparse_recipe_crc64_with_zeroed_checksum(encoded);
	detail::put_recipe_le<uint64_t>(encoded, detail::kSparseReadRecipeChecksumByte, stats.sidecar_crc64);
	write_recipe_file_atomic(recipe_path, encoded);
	stats.sidecar_bytes = encoded.size();
	return stats;
}

std::filesystem::path sparse_read_recipe_path(
	const std::filesystem::path& directory,
	const uint32_t               shard_id) {
	std::ostringstream name;
	name << "shard_" << std::setw(6) << std::setfill('0') << shard_id << ".sparse_read_recipe.bin";
	return directory / name.str();
}

FlsReader::FlsReader(const std::filesystem::path& file_path,
                     const bool                   load_column_names,
                     const bool                   enable_sparse_vector_reads)
	: FlsReader(file_path,
	            FlsReaderOptions {.load_column_names         = load_column_names,
	                              .enable_sparse_vector_reads = enable_sparse_vector_reads}) {
}

struct FlsReaderStaticMetadata::Impl {
	std::string                                             source_path_key;
	std::shared_ptr<const fastlanes::TableDescriptorHandle> table_descriptor;
	std::shared_ptr<CompactDescriptorV3>                    compact_descriptor;
	std::shared_ptr<const detail::SparseVectorBundleIndex>  sparse_vector_bundle;
	std::shared_ptr<const detail::SparseDatasetAccessIndex> sparse_access_index;
	std::shared_ptr<const ZeroCopySchemaPlan>               zero_copy_schema_plan;
	std::shared_ptr<const uint8_t>                          sparse_plan_owner;
	std::shared_ptr<const detail::SparseReadRecipeIndex>    sparse_read_recipe;
	SparseReaderInitializationStats                         initialization_stats;
	bool                                                    load_column_names = true;
	bool                                                    sparse_vector_reads_enabled = true;
	bool                                                    build_shared_zero_copy_schema_plan = true;
	std::string                                             sparse_read_recipe_path_key;
	uint64_t                                                sparse_read_recipe_source_fingerprint = 0U;
	size_t                                                  retained_bytes = 0U;
};

FlsReaderStaticMetadata::FlsReaderStaticMetadata(std::shared_ptr<const Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

FlsReaderStaticMetadata::~FlsReaderStaticMetadata() = default;

size_t FlsReaderStaticMetadata::retained_bytes() const noexcept {
	return impl_ == nullptr ? 0U : impl_->retained_bytes;
}

FlsReader::FlsReader(const std::filesystem::path& file_path, const FlsReaderOptions& options)
    : m_file(std::make_shared<fastlanes::File>(file_path))
    , m_load_column_names(options.load_column_names) {
	const auto descriptor_open_begin = std::chrono::steady_clock::now();
	if (is_compact_v3_fls(file_path)) {
		m_compact_descriptor =
		    std::make_shared<CompactDescriptorV3>(CompactDescriptorV3::Open(file_path));
	} else {
		m_table_descriptor = std::make_shared<fastlanes::TableDescriptorHandle>(
		    detail::load_table_descriptor(*m_file, file_path));
	}
	m_sparse_initialization_stats.descriptor_open_ms = std::chrono::duration<double, std::milli>(
	    std::chrono::steady_clock::now() - descriptor_open_begin).count();
	if (options.enable_sparse_vector_reads && m_table_descriptor != nullptr) {
		m_sparse_vector_bundle = detail::load_sparse_vector_bundle_index(
		    sparse_vector_bundle_path(file_path), *table_descriptor(), m_file->Size());
		const auto sparse_index_begin = std::chrono::steady_clock::now();
		m_sparse_access_index = detail::build_sparse_dataset_access_index(*table_descriptor());
		m_sparse_initialization_stats.sparse_access_index_build_ms =
		    std::chrono::duration<double, std::milli>(
		        std::chrono::steady_clock::now() - sparse_index_begin).count();
		const auto source_validation_begin = std::chrono::steady_clock::now();
		m_sparse_read_recipe = detail::load_sparse_read_recipe_index(
		    options.sparse_read_recipe_path,
		    file_path,
		    *table_descriptor(),
		    m_file->Size(),
		    options.sparse_read_recipe_source_fingerprint,
		    &m_sparse_initialization_stats.sparse_recipe_load_ms,
		    &m_sparse_initialization_stats.sparse_recipe_validation_ms);
		m_sparse_initialization_stats.source_validation_ms = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - source_validation_begin).count();
		if (m_sparse_read_recipe) {
			m_sparse_initialization_stats.sparse_recipe_loaded        = true;
			m_sparse_initialization_stats.sparse_recipe_sidecar_bytes = m_sparse_read_recipe->sidecar_bytes;
			m_sparse_initialization_stats.sparse_recipe_record_count  = m_sparse_read_recipe->records.size();
			const auto prehydrate = detail::prehydrate_sparse_recipe_access(
			    *m_file,
			    *table_descriptor(),
			    *m_sparse_read_recipe,
			    options.sparse_read_recipe_rehydrate_workers);
			m_sparse_initialization_stats.sparse_recipe_rehydrate_ms = prehydrate.wall_ms;
			m_sparse_initialization_stats.sparse_recipe_rehydrate_service_ms = prehydrate.service_ms;
			m_sparse_initialization_stats.sparse_recipe_rehydrate_workers = prehydrate.worker_count;
			m_sparse_initialization_stats.sparse_recipe_source_metadata_bytes = prehydrate.metadata_bytes;
			m_sparse_initialization_stats.sparse_recipe_source_metadata_pread_count =
			    prehydrate.metadata_pread_count;
		}
	}
	// Compact-v3 reconstructs per-rowgroup geometry on demand.  Always retain
	// the shared expression/schema plan so the hot rowgroup-only training path
	// does not also allocate a RowgroupView and a ZeroCopyColumn vector for all
	// coefficient columns.  Each reconstructed rowgroup is still checked
	// against the plan before the fast path is used.
	const auto schema_plan_begin = std::chrono::steady_clock::now();
	if (options.build_shared_zero_copy_schema_plan || m_compact_descriptor != nullptr) {
		m_zero_copy_schema_plan = std::make_shared<ZeroCopySchemaPlan>(build_shared_zero_copy_schema_plan());
	}
	m_sparse_initialization_stats.zero_copy_schema_plan_build_ms =
	    std::chrono::duration<double, std::milli>(
	        std::chrono::steady_clock::now() - schema_plan_begin).count();
	auto metadata                         = std::make_shared<FlsReaderStaticMetadata::Impl>();
	metadata->source_path_key             = file_path.lexically_normal().string();
	metadata->table_descriptor            = m_table_descriptor;
	metadata->compact_descriptor          = m_compact_descriptor;
	metadata->sparse_vector_bundle        = m_sparse_vector_bundle;
	metadata->sparse_access_index         = m_sparse_access_index;
	metadata->zero_copy_schema_plan       = m_zero_copy_schema_plan;
	metadata->sparse_plan_owner           = m_sparse_plan_owner;
	metadata->sparse_read_recipe          = m_sparse_read_recipe;
	metadata->initialization_stats        = m_sparse_initialization_stats;
	metadata->load_column_names           = m_load_column_names;
	metadata->sparse_vector_reads_enabled = options.enable_sparse_vector_reads;
	metadata->build_shared_zero_copy_schema_plan = options.build_shared_zero_copy_schema_plan;
	metadata->sparse_read_recipe_path_key = options.sparse_read_recipe_path.lexically_normal().string();
	metadata->sparse_read_recipe_source_fingerprint = options.sparse_read_recipe_source_fingerprint;
	// Compact descriptors dominate retained static storage and are mmap-backed.
	// The small shared schema-plan vectors are accounted separately by their
	// owned capacities so the cache reports an honest lower-level byte total.
	metadata->retained_bytes = m_compact_descriptor == nullptr ? 0U : m_compact_descriptor->descriptor_bytes();
	if (m_zero_copy_schema_plan != nullptr) {
		metadata->retained_bytes += sizeof(ZeroCopySchemaPlan) +
		                            m_zero_copy_schema_plan->columns.capacity() * sizeof(ZeroCopyColumnPlan) +
		                            m_zero_copy_schema_plan->build_order.capacity() * sizeof(size_t);
		for (const auto& column : m_zero_copy_schema_plan->columns) {
			metadata->retained_bytes += column.name.capacity() +
			                            column.operand_ids.capacity() * sizeof(uint32_t);
		}
	}
	m_static_metadata = std::shared_ptr<const FlsReaderStaticMetadata>(
	    new FlsReaderStaticMetadata(std::move(metadata)));
	if (m_compact_descriptor != nullptr) {
		m_compact_descriptor->release_resident_pages();
	}
}

FlsReader::FlsReader(const std::filesystem::path&                   file_path,
                     const FlsReaderOptions&                        options,
                     std::shared_ptr<const FlsReaderStaticMetadata> static_metadata)
    : m_file(std::make_shared<fastlanes::File>(file_path))
    , m_load_column_names(options.load_column_names)
    , m_static_metadata(std::move(static_metadata)) {
	if (m_static_metadata == nullptr || m_static_metadata->impl_ == nullptr) {
		throw std::invalid_argument("FlsReader shared static metadata is empty");
	}
	const auto& metadata = *m_static_metadata->impl_;
	if (metadata.source_path_key != file_path.lexically_normal().string()) {
		throw std::invalid_argument("FlsReader shared static metadata belongs to a different source path");
	}
	if (metadata.load_column_names != options.load_column_names ||
	    metadata.sparse_vector_reads_enabled != options.enable_sparse_vector_reads ||
	    metadata.build_shared_zero_copy_schema_plan != options.build_shared_zero_copy_schema_plan ||
	    metadata.sparse_read_recipe_path_key != options.sparse_read_recipe_path.lexically_normal().string() ||
	    metadata.sparse_read_recipe_source_fingerprint != options.sparse_read_recipe_source_fingerprint) {
		throw std::invalid_argument("FlsReader shared static metadata options do not match the payload reader");
	}
	m_table_descriptor       = metadata.table_descriptor;
	m_compact_descriptor     = metadata.compact_descriptor;
	m_sparse_vector_bundle   = metadata.sparse_vector_bundle;
	m_sparse_access_index    = metadata.sparse_access_index;
	m_zero_copy_schema_plan  = metadata.zero_copy_schema_plan;
	m_sparse_plan_owner      = metadata.sparse_plan_owner;
	m_sparse_read_recipe     = metadata.sparse_read_recipe;
	// Rebinding immutable metadata performs no descriptor/index work.  Keep the
	// per-reader initialization timings at zero so telemetry cannot mistake a
	// static-cache hit for another descriptor open.
	m_sparse_initialization_stats = {};
}

const fastlanes::TableDescriptor* FlsReader::table_descriptor() const {
	if (m_compact_descriptor != nullptr) {
		throw std::runtime_error("Compact v3 reader does not load or expose a FastLanes TableDescriptor");
	}
	if (m_table_descriptor == nullptr) {
		throw std::runtime_error("TableDescriptor owner is not initialized");
	}
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	return td;
}

size_t FlsReader::rowgroup_count() const {
	if (m_compact_descriptor != nullptr) {
		return m_compact_descriptor->rowgroup_count();
	}
	const auto* td = table_descriptor();
	return static_cast<size_t>(td->m_rowgroup_descriptors()->size());
}

size_t FlsReader::rowgroup_storage_bytes(const size_t rowgroup_idx) const {
	if (m_compact_descriptor != nullptr) {
		return m_compact_descriptor->rowgroup(rowgroup_idx).payload_size;
	}
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto n_rgs = td->m_rowgroup_descriptors()->size();
	if (rowgroup_idx >= n_rgs) {
		throw std::out_of_range("rowgroup_idx out of range");
	}
	const auto* rg = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	return static_cast<size_t>(rg->m_size());
}

bool FlsReader::has_sparse_vector_bundle() const noexcept {
	return static_cast<bool>(m_sparse_vector_bundle);
}

bool FlsReader::is_compact_v3() const noexcept {
	return static_cast<bool>(m_compact_descriptor);
}

const SparseReaderInitializationStats& FlsReader::sparse_initialization_stats() const noexcept {
	return m_sparse_initialization_stats;
}

std::shared_ptr<const FlsReaderStaticMetadata> FlsReader::share_static_metadata() const noexcept {
	return m_static_metadata;
}

size_t FlsReader::static_metadata_bytes() const noexcept {
	return m_static_metadata == nullptr ? 0U : m_static_metadata->retained_bytes();
}

bool FlsReader::sparse_vector_read_supported(const size_t rowgroup_idx, std::string* const reason) const {
	if (m_compact_descriptor != nullptr) {
		if (rowgroup_idx >= m_compact_descriptor->rowgroup_count()) {
			throw std::out_of_range("rowgroup_idx out of range");
		}
		if (reason != nullptr) {
			*reason = "compact-v3-rowgroup-is-one-vector";
		}
		return false;
	}
	if (!m_sparse_access_index) {
		if (reason != nullptr) {
			*reason = "sparse-access-index-disabled";
		}
		return false;
	}
	if (rowgroup_idx >= m_sparse_access_index->rowgroup_count) {
		throw std::out_of_range("rowgroup_idx out of range");
	}
	const auto entry = detail::sparse_rowgroup_access(
	    *m_file, *table_descriptor(), m_sparse_access_index, rowgroup_idx);
	if (reason != nullptr) {
		*reason = entry->fallback_reason;
	}
	return entry->supported;
}

SparseVectorReadPlan FlsReader::compile_sparse_vector_read_plan(
	const size_t rowgroup_idx, const std::vector<uint32_t>& selected_vectors, const bool packed_device_scatter) const {
	return compile_sparse_vector_read_plan(
	    rowgroup_idx, selected_vectors, std::vector<uint8_t> {}, packed_device_scatter);
}

SparseVectorReadPlan FlsReader::compile_sparse_vector_read_plan(
	const size_t                    rowgroup_idx,
	const std::vector<uint32_t>&    selected_vectors,
	const std::vector<uint8_t>&     selected_columns,
	const bool                      packed_device_scatter) const {
	if (m_compact_descriptor != nullptr) {
		if (!selected_columns.empty()) {
			throw std::invalid_argument(
			    "combined selected-vector/selected-column plans require manifest-v1 FLS storage");
		}
		if (rowgroup_idx >= m_compact_descriptor->rowgroup_count()) {
			throw std::out_of_range("rowgroup_idx out of range");
		}
		if (selected_vectors.empty()) {
			throw std::invalid_argument("selected vector read requires at least one vector");
		}
		if (std::any_of(selected_vectors.begin(), selected_vectors.end(), [](const uint32_t vector) {
			    return vector != 0U;
		    })) {
			throw std::out_of_range("Compact v3 rowgroup contains exactly one vector");
		}
		const auto record = m_compact_descriptor->rowgroup(rowgroup_idx);
		auto plan = std::make_shared<SparseVectorReadPlan::Impl>();
		plan->owner                  = m_sparse_plan_owner;
		plan->rowgroup_index         = rowgroup_idx;
		plan->rowgroup_bytes         = record.payload_size;
		plan->full_vector_count      = 1U;
		plan->selected_vector_count  = 1U;
		plan->storage_bytes          = record.payload_size;
		plan->selected_storage_bytes = record.payload_size;
		plan->fallback_reason        = "compact-v3-rowgroup-is-one-vector";
		static_cast<void>(packed_device_scatter);
		return SparseVectorReadPlan(std::move(plan));
	}
	const auto* td = m_table_descriptor->Get();
	if (td == nullptr) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto* rowgroups = td->m_rowgroup_descriptors();
	if (rowgroups == nullptr || rowgroup_idx >= rowgroups->size()) {
		throw std::out_of_range("rowgroup_idx out of range");
	}
	if (selected_vectors.empty()) {
		throw std::invalid_argument("selected vector read requires at least one vector");
	}
	const auto* rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	if (rowgroup == nullptr) {
		throw std::runtime_error("rowgroup descriptor is missing");
	}
	std::vector<uint8_t> materialized_columns(selected_columns);
	std::sort(materialized_columns.begin(), materialized_columns.end());
	materialized_columns.erase(
	    std::unique(materialized_columns.begin(), materialized_columns.end()), materialized_columns.end());
	const auto* column_descriptors = rowgroup->m_column_descriptors();
	if (column_descriptors == nullptr) {
		throw std::runtime_error("rowgroup column descriptors are missing");
	}
	if (!materialized_columns.empty() && materialized_columns.back() >= column_descriptors->size()) {
		throw std::out_of_range("selected column exceeds rowgroup column count");
	}
	std::vector<uint8_t> physical_columns;
	if (!materialized_columns.empty()) {
		const auto column_plan = detail::build_zero_copy_column_plan(*rowgroup, /*load_column_names=*/false);
		std::vector<uint8_t> state(column_plan.size(), 0U);
		std::vector<bool>    active(column_plan.size(), false);
		const auto resolve = [&](const auto& self, const size_t column_index) -> void {
			if (column_index >= column_plan.size()) {
				throw std::out_of_range("selected column alias target exceeds rowgroup column count");
			}
			if (state[column_index] == 2U) {
				return;
			}
			if (state[column_index] == 1U) {
				throw std::runtime_error("cycle detected in selected column aliases");
			}
			state[column_index] = 1U;
			if (column_plan[column_index].alias_of.has_value()) {
				self(self, *column_plan[column_index].alias_of);
			} else {
				active[column_index] = true;
			}
			state[column_index] = 2U;
		};
		for (const auto column : materialized_columns) {
			resolve(resolve, column);
		}
		for (size_t column = 0U; column < active.size(); ++column) {
			if (active[column]) {
				physical_columns.push_back(static_cast<uint8_t>(column));
			}
		}
		if (physical_columns.empty()) {
			throw std::runtime_error("selected columns resolve to no physical payload columns");
		}
	}
	const size_t rowgroup_bytes = static_cast<size_t>(rowgroup->m_size());
	const size_t vector_count   = static_cast<size_t>(rowgroup->m_n_vec());
	std::vector<uint32_t> vectors(selected_vectors);
	std::sort(vectors.begin(), vectors.end());
	vectors.erase(std::unique(vectors.begin(), vectors.end()), vectors.end());
	if (vectors.back() >= vector_count) {
		throw std::out_of_range("selected vector exceeds rowgroup vector count");
	}
	if (!m_sparse_access_index || rowgroup_idx >= m_sparse_access_index->rowgroup_count) {
		throw std::runtime_error("sparse dataset access index is unavailable");
	}
	auto plan = std::make_shared<SparseVectorReadPlan::Impl>();
	plan->owner                 = m_sparse_plan_owner;
	plan->rowgroup_index        = rowgroup_idx;
	plan->rowgroup_bytes        = rowgroup_bytes;
	plan->full_vector_count     = vector_count;
	plan->selected_vector_count = vectors.size();
	plan->materialized_columns  = materialized_columns;
	const auto recipe_lookup_begin = std::chrono::steady_clock::now();
	const auto selection_words = detail::sparse_selection_words(vector_count, vectors);
	const detail::SparseReadRecipeRecord* recipe_record = nullptr;
	if (m_sparse_read_recipe && materialized_columns.empty()) {
		recipe_record = detail::find_sparse_read_recipe_record(*m_sparse_read_recipe, rowgroup_idx, selection_words);
	}
	plan->recipe_lookup_ms = std::chrono::duration<double, std::milli>(
	    std::chrono::steady_clock::now() - recipe_lookup_begin).count();
	if (recipe_record != nullptr && vectors.size() < vector_count && !m_sparse_vector_bundle) {
		if (recipe_record->prehydrated_access) {
			plan->access = recipe_record->prehydrated_access;
		} else {
			const auto rehydrate_begin = std::chrono::steady_clock::now();
			plan->access = detail::rehydrate_sparse_recipe_access(
			    *m_file,
			    *rowgroup,
			    *recipe_record,
			    &plan->recipe_source_metadata_bytes,
			    &plan->recipe_source_metadata_pread_count);
			plan->recipe_rehydrate_ms = std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - rehydrate_begin).count();
		}
		plan->strategy               = SparseVectorReadPlan::Impl::Strategy::kSourceRanges;
		plan->source_ranges          = recipe_record->source_ranges;
		plan->exact_source_ranges    = plan->source_ranges;
		plan->storage_bytes          = recipe_record->selected_storage_bytes;
		plan->selected_storage_bytes = recipe_record->selected_storage_bytes;
		plan->recipe_hit             = true;
		return SparseVectorReadPlan(std::move(plan));
	}

	const auto endpoint_begin = std::chrono::steady_clock::now();
	const auto access = detail::sparse_rowgroup_access(
	    *m_file, *table_descriptor(), m_sparse_access_index, rowgroup_idx);
	plan->endpoint_resolution_ms = std::chrono::duration<double, std::milli>(
	    std::chrono::steady_clock::now() - endpoint_begin).count();
	plan->access = access;
	if (!access->supported || (vectors.size() >= vector_count && materialized_columns.empty())) {
		plan->fallback_reason = access->supported ? "selected-vectors-cover-full-rowgroup" : access->fallback_reason;
		plan->storage_bytes   = rowgroup_bytes;
		plan->selected_storage_bytes = rowgroup_bytes;
		return SparseVectorReadPlan(std::move(plan));
	}

	if (!m_sparse_vector_bundle || !materialized_columns.empty()) {
		plan->strategy = SparseVectorReadPlan::Impl::Strategy::kSourceRanges;
		std::vector<detail::SparseByteRange> ranges;
		const auto gather_begin = std::chrono::steady_clock::now();
		for (const auto vector : vectors) {
			const auto& vector_ranges = access->vector_ranges.at(vector);
			if (materialized_columns.empty()) {
				ranges.insert(ranges.end(), vector_ranges.begin(), vector_ranges.end());
				continue;
			}
			for (const auto physical_column : physical_columns) {
				if (physical_column >= access->column_vector_range_indices.size()) {
					throw std::logic_error("selected physical column is absent from sparse access geometry");
				}
				for (const auto range_index : access->column_vector_range_indices[physical_column]) {
					if (range_index >= vector_ranges.size()) {
						throw std::logic_error("selected column/vector sparse range index is out of bounds");
					}
					ranges.push_back(vector_ranges[range_index]);
				}
			}
		}
		plan->range_gather_ms = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - gather_begin).count();
		const auto coalesce_begin = std::chrono::steady_clock::now();
		plan->source_ranges = detail::coalesce_ranges(std::move(ranges));
		plan->exact_source_ranges = plan->source_ranges;
		plan->range_sort_coalesce_ms = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - coalesce_begin).count();
		for (const auto& range : plan->source_ranges) {
			plan->storage_bytes += range.size;
		}
		plan->selected_storage_bytes = plan->storage_bytes;
		return SparseVectorReadPlan(std::move(plan));
	}

	if (rowgroup_idx >= m_sparse_vector_bundle->rowgroups.size()) {
		throw std::runtime_error("sparse vector bundle rowgroup index is out of range");
	}
	const auto& bundle = m_sparse_vector_bundle->rowgroups[rowgroup_idx];
	if (packed_device_scatter) {
		plan->strategy     = SparseVectorReadPlan::Impl::Strategy::kBundlePacked;
		plan->packed_bytes = access->static_prefix.size();
		size_t prefix_cursor = 0U;
		for (const auto& range : access->index_ranges) {
			plan->packed_scatter_ranges.push_back(
			    {prefix_cursor, range.offset, range.size});
			prefix_cursor += range.size;
		}
		for (const auto& range : access->shared_ranges) {
			plan->packed_scatter_ranges.push_back(
			    {prefix_cursor, range.offset, range.size});
			prefix_cursor += range.size;
		}
		if (prefix_cursor != access->static_prefix.size()) {
			throw std::runtime_error("compiled sparse bundle prefix size mismatch");
		}
	}

	const char* const bundle_policy = std::getenv("GALP_VECTOR_BUNDLE_READ_POLICY");
	const bool envelope = !packed_device_scatter && bundle_policy != nullptr &&
	                      std::strcmp(bundle_policy, "envelope") == 0;
	if (envelope) {
		plan->strategy             = SparseVectorReadPlan::Impl::Strategy::kBundleEnvelope;
		plan->envelope_file_offset = bundle.vectors_offset;
		plan->envelope_size        = static_cast<size_t>(bundle.vector_offsets.at(vectors.back() + 1U));
		plan->storage_bytes        = plan->envelope_size;
		for (const auto vector : vectors) {
			size_t source_cursor = static_cast<size_t>(bundle.vector_offsets.at(vector));
			for (const auto& range : access->vector_ranges.at(vector)) {
				plan->envelope_copies.push_back({source_cursor, range.offset, range.size});
				plan->selected_storage_bytes += range.size;
				source_cursor += range.size;
			}
			if (source_cursor != static_cast<size_t>(bundle.vector_offsets.at(vector + 1U))) {
				throw std::runtime_error("compiled sparse bundle envelope vector size mismatch");
			}
		}
		return SparseVectorReadPlan(std::move(plan));
	}

	if (!packed_device_scatter) {
		plan->strategy = SparseVectorReadPlan::Impl::Strategy::kBundleRuns;
	}
	for (size_t position = 0U; position < vectors.size();) {
		const uint32_t run_begin = vectors[position];
		uint32_t       run_end   = run_begin + 1U;
		++position;
		while (position < vectors.size() && vectors[position] == run_end) {
			++run_end;
			++position;
		}
		const uint64_t packed_begin = bundle.vector_offsets.at(run_begin);
		const uint64_t packed_end   = bundle.vector_offsets.at(run_end);
		if (packed_end < packed_begin || packed_end - packed_begin > std::numeric_limits<size_t>::max()) {
			throw std::runtime_error("compiled sparse bundle run exceeds addressable memory");
		}
		SparseVectorReadPlan::Impl::BundleRun run;
		run.file_offset   = bundle.vectors_offset + packed_begin;
		run.size          = static_cast<size_t>(packed_end - packed_begin);
		run.packed_offset = plan->packed_bytes;
		for (uint32_t vector = run_begin; vector < run_end; ++vector) {
			const auto& ranges = access->vector_ranges.at(vector);
			run.logical_ranges.insert(run.logical_ranges.end(), ranges.begin(), ranges.end());
		}
		const size_t logical_bytes = std::accumulate(
		    run.logical_ranges.begin(), run.logical_ranges.end(), size_t {0},
		    [](const size_t total, const detail::SparseByteRange& range) { return total + range.size; });
		if (logical_bytes != run.size) {
			throw std::runtime_error("compiled sparse bundle run size mismatch");
		}
		if (packed_device_scatter) {
			size_t packed_cursor = run.packed_offset;
			for (const auto& range : run.logical_ranges) {
				plan->packed_scatter_ranges.push_back({packed_cursor, range.offset, range.size});
				packed_cursor += range.size;
			}
			plan->packed_bytes += run.size;
		}
		plan->storage_bytes += run.size;
		plan->selected_storage_bytes += run.size;
		plan->bundle_runs.push_back(std::move(run));
	}
	return SparseVectorReadPlan(std::move(plan));
}

void FlsReader::read_rowgroup_bytes_into(const size_t        rowgroup_idx,
                                         std::byte* const    backing_data,
                                         const size_t        backing_capacity,
                                         ZeroCopyReadTiming* timing) {
	if (m_compact_descriptor != nullptr) {
		const auto record = m_compact_descriptor->rowgroup(rowgroup_idx);
		if ((record.payload_size != 0U && backing_data == nullptr) || backing_capacity < record.payload_size) {
			throw std::runtime_error("external rowgroup backing is null or too small");
		}
		if (timing != nullptr) {
			timing->storage_bytes              = record.payload_size;
			timing->logical_storage_bytes      = record.payload_size;
			timing->full_storage_bytes         = record.payload_size;
			timing->selected_coefficient_count = m_compact_descriptor->column_count();
			timing->full_coefficient_count     = m_compact_descriptor->column_count();
		}
		if (record.payload_size == 0U) {
			return;
		}
		const auto pread_start = std::chrono::steady_clock::now();
		m_file->ReadRangeUnchecked(backing_data, record.payload_offset, record.payload_size);
		const auto pread_end = std::chrono::steady_clock::now();
		if (timing != nullptr) {
			constexpr size_t page_size = 4096U;
			const auto first_page = record.payload_offset / page_size;
			const auto last_page = (record.payload_offset + record.payload_size - 1U) / page_size;
			timing->physical_page_bytes       = (last_page - first_page + 1U) * page_size;
			timing->full_physical_page_bytes  = timing->physical_page_bytes;
			timing->coalesced_read_run_count  = 1U;
			timing->pread_count += 1U;
			timing->pread_ms += std::chrono::duration<double, std::milli>(pread_end - pread_start).count();
			if (timing->pread_start == std::chrono::steady_clock::time_point {} || pread_start < timing->pread_start) {
				timing->pread_start = pread_start;
			}
			if (pread_end > timing->pread_end) {
				timing->pread_end = pread_end;
			}
		}
		return;
	}
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto n_rgs = td->m_rowgroup_descriptors()->size();
	if (rowgroup_idx >= n_rgs) {
		throw std::out_of_range("rowgroup_idx out of range");
	}

	const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	const size_t rg_bytes = static_cast<size_t>(rg->m_size());
	if ((rg_bytes != 0U && backing_data == nullptr) || backing_capacity < rg_bytes) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}
	if (timing != nullptr) {
		timing->storage_bytes      = rg_bytes;
		timing->full_storage_bytes = rg_bytes;
	}
	if (rg_bytes == 0U) {
		return;
	}

	const auto pread_start = std::chrono::steady_clock::now();
	m_file->ReadRangeUnchecked(backing_data, rg->m_offset(), rg->m_size());
	const auto pread_end = std::chrono::steady_clock::now();
	if (timing != nullptr) {
		timing->pread_count += 1U;
		timing->pread_ms += std::chrono::duration<double, std::milli>(pread_end - pread_start).count();
		if (timing->pread_start == std::chrono::steady_clock::time_point {} || pread_start < timing->pread_start) {
			timing->pread_start = pread_start;
		}
		if (pread_end > timing->pread_end) {
			timing->pread_end = pread_end;
		}
	}
}

void FlsReader::read_rowgroup_bytes_selected_vectors_into(const size_t                 rowgroup_idx,
	                                                      const std::vector<uint32_t>& selected_vectors,
	                                                      std::byte* const             backing_data,
	                                                      const size_t                 backing_capacity,
	                                                      ZeroCopyReadTiming*          timing) {
	if (m_compact_descriptor != nullptr) {
		if (selected_vectors.empty()) {
			throw std::invalid_argument("selected vector read requires at least one vector");
		}
		if (std::any_of(selected_vectors.begin(), selected_vectors.end(), [](const uint32_t vector) {
			    return vector != 0U;
		    })) {
			throw std::out_of_range("Compact v3 rowgroup contains exactly one vector");
		}
		if (timing != nullptr) {
			timing->sparse_fallback_reason = "compact-v3-rowgroup-is-one-vector";
		}
		read_rowgroup_bytes_into(rowgroup_idx, backing_data, backing_capacity, timing);
		return;
	}
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto n_rgs = td->m_rowgroup_descriptors()->size();
	if (rowgroup_idx >= n_rgs) {
		throw std::out_of_range("rowgroup_idx out of range");
	}
	const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	const size_t rg_bytes = static_cast<size_t>(rg->m_size());
	const size_t n_vecs   = static_cast<size_t>(rg->m_n_vec());
	if (backing_data == nullptr || backing_capacity < rg_bytes) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}
	if (selected_vectors.empty()) {
		throw std::invalid_argument("selected vector read requires at least one vector");
	}
	std::vector<uint32_t> unique_vectors = selected_vectors;
	std::sort(unique_vectors.begin(), unique_vectors.end());
	unique_vectors.erase(std::unique(unique_vectors.begin(), unique_vectors.end()), unique_vectors.end());
	if (unique_vectors.back() >= n_vecs) {
		throw std::out_of_range("selected vector exceeds rowgroup vector count");
	}

	if (!m_sparse_access_index || rowgroup_idx >= m_sparse_access_index->rowgroup_count) {
		throw std::runtime_error("sparse dataset access index is unavailable");
	}
	const auto access_owner = detail::sparse_rowgroup_access(
	    *m_file, *table_descriptor(), m_sparse_access_index, rowgroup_idx);
	const auto& access_index = *access_owner;
	const bool  supported    = access_index.supported;
	if (timing != nullptr) {
		timing->full_storage_bytes    = rg_bytes;
		timing->sparse_read_supported = supported;
	}
	if (!supported || unique_vectors.size() >= n_vecs) {
		if (timing != nullptr) {
			timing->sparse_fallback_reason =
			    supported ? "selected-vectors-cover-full-rowgroup" : access_index.fallback_reason;
		}
		read_rowgroup_bytes_into(rowgroup_idx, backing_data, backing_capacity, timing);
		return;
	}

	// Existing rowgroup materialization and decode code expects descriptors to
	// retain their original offsets. Populate the index and selected payload
	// ranges in their logical positions; selected-vector decode never observes
	// the unpopulated holes.
	if (m_sparse_vector_bundle) {
		if (rowgroup_idx >= m_sparse_vector_bundle->rowgroups.size()) {
			throw std::runtime_error("sparse vector bundle rowgroup index is out of range");
		}
		const auto& bundle_entry = m_sparse_vector_bundle->rowgroups[rowgroup_idx];
		const auto record_bundle_read = [&](const uint64_t offset, const uint64_t size, std::byte* const destination) {
			if (size == 0U) {
				return;
			}
			if (size > std::numeric_limits<size_t>::max()) {
				throw std::runtime_error("sparse vector bundle read exceeds addressable memory");
			}
			const auto start = std::chrono::steady_clock::now();
			m_sparse_vector_bundle->file->ReadRangeUnchecked(destination, offset, size);
			const auto end = std::chrono::steady_clock::now();
			if (timing != nullptr) {
				timing->storage_bytes += static_cast<size_t>(size);
				++timing->pread_count;
				timing->pread_ms += std::chrono::duration<double, std::milli>(end - start).count();
				if (timing->pread_start == std::chrono::steady_clock::time_point {} || start < timing->pread_start) {
					timing->pread_start = start;
				}
				if (end > timing->pread_end) {
					timing->pread_end = end;
				}
			}
		};

		const char* const bundle_policy = std::getenv("GALP_VECTOR_BUNDLE_READ_POLICY");
		const bool envelope_read = bundle_policy != nullptr && std::strcmp(bundle_policy, "envelope") == 0;
		if (envelope_read) {
			const uint32_t last_vector = unique_vectors.back();
			const uint64_t envelope_end = bundle_entry.vectors_offset + bundle_entry.vector_offsets.at(last_vector + 1U);
			if (envelope_end < bundle_entry.vectors_offset ||
			    envelope_end - bundle_entry.vectors_offset > std::numeric_limits<size_t>::max()) {
				throw std::runtime_error("invalid sparse vector bundle envelope");
			}
			std::vector<std::byte> envelope(static_cast<size_t>(envelope_end - bundle_entry.vectors_offset));
			record_bundle_read(bundle_entry.vectors_offset, envelope.size(), envelope.data());
			size_t prefix_cursor = 0U;
			detail::scatter_ranges(
			    access_index.static_prefix.data(),
			    access_index.static_prefix.size(),
			    backing_data,
			    access_index.index_ranges,
			    prefix_cursor);
			detail::scatter_ranges(
			    access_index.static_prefix.data(),
			    access_index.static_prefix.size(),
			    backing_data,
			    access_index.shared_ranges,
			    prefix_cursor);
			if (prefix_cursor != access_index.static_prefix.size()) {
				throw std::runtime_error("sparse vector bundle envelope prefix size mismatch");
			}
			for (const uint32_t vector : unique_vectors) {
				size_t packed_cursor = static_cast<size_t>(bundle_entry.vector_offsets.at(vector));
				const size_t packed_end = static_cast<size_t>(bundle_entry.vector_offsets.at(vector + 1U));
				for (const auto& range : access_index.vector_ranges.at(vector)) {
					if (packed_cursor > packed_end || range.size > packed_end - packed_cursor ||
					    packed_cursor > envelope.size() || range.size > envelope.size() - packed_cursor) {
						throw std::runtime_error("truncated sparse vector bundle envelope vector payload");
					}
					std::memcpy(backing_data + range.offset, envelope.data() + packed_cursor, range.size);
					packed_cursor += range.size;
				}
				if (packed_cursor != packed_end) {
					throw std::runtime_error("sparse vector bundle envelope vector payload size mismatch");
				}
			}
			if (timing != nullptr) {
				timing->used_sparse_read                 = true;
				timing->used_vector_bundle_read          = true;
				timing->used_vector_bundle_envelope_read = true;
			}
			return;
		}

		size_t prefix_cursor = 0U;
		detail::scatter_ranges(
		    access_index.static_prefix.data(),
		    access_index.static_prefix.size(),
		    backing_data,
		    access_index.index_ranges,
		    prefix_cursor);
		detail::scatter_ranges(
		    access_index.static_prefix.data(),
		    access_index.static_prefix.size(),
		    backing_data,
		    access_index.shared_ranges,
		    prefix_cursor);
		if (prefix_cursor != access_index.static_prefix.size()) {
			throw std::runtime_error("sparse vector bundle rowgroup prefix size mismatch");
		}

		for (size_t selected_pos = 0U; selected_pos < unique_vectors.size();) {
			const uint32_t run_begin = unique_vectors[selected_pos];
			uint32_t       run_end   = run_begin + 1U;
			++selected_pos;
			while (selected_pos < unique_vectors.size() && unique_vectors[selected_pos] == run_end) {
				++run_end;
				++selected_pos;
			}
			const uint64_t packed_begin = bundle_entry.vector_offsets.at(run_begin);
			const uint64_t packed_end   = bundle_entry.vector_offsets.at(run_end);
			if (packed_end < packed_begin || packed_end - packed_begin > std::numeric_limits<size_t>::max()) {
				throw std::runtime_error("invalid sparse vector bundle selected-vector run");
			}
			std::vector<fastlanes::FileScatterReadTarget> scatter_targets;
			size_t packed_size = 0U;
			for (uint32_t vector = run_begin; vector < run_end; ++vector) {
				const auto& vector_ranges = access_index.vector_ranges.at(vector);
				scatter_targets.reserve(scatter_targets.size() + vector_ranges.size());
				for (const auto& range : vector_ranges) {
					if (range.size > std::numeric_limits<size_t>::max() - packed_size) {
						throw std::runtime_error("sparse vector bundle selected-vector payload size overflow");
					}
					packed_size += range.size;
					scatter_targets.push_back(fastlanes::FileScatterReadTarget {backing_data + range.offset, range.size});
				}
			}
			if (packed_size != static_cast<size_t>(packed_end - packed_begin)) {
				throw std::runtime_error("sparse vector bundle selected-vector payload size mismatch");
			}
			const auto start = std::chrono::steady_clock::now();
			const auto read_count = m_sparse_vector_bundle->file->ReadScatterUnchecked(
			    scatter_targets, bundle_entry.vectors_offset + packed_begin);
			const auto end = std::chrono::steady_clock::now();
			if (timing != nullptr) {
				timing->storage_bytes += packed_size;
				timing->pread_count += static_cast<size_t>(read_count);
				timing->pread_ms += std::chrono::duration<double, std::milli>(end - start).count();
				if (timing->pread_start == std::chrono::steady_clock::time_point {} || start < timing->pread_start) {
					timing->pread_start = start;
				}
				if (end > timing->pread_end) {
					timing->pread_end = end;
				}
			}
		}
		if (timing != nullptr) {
			timing->used_sparse_read        = true;
			timing->used_vector_bundle_read = true;
		}
		return;
	}
	std::memset(backing_data, 0, rg_bytes);
	size_t static_prefix_cursor = 0U;
	detail::scatter_ranges(access_index.static_prefix.data(),
	                       access_index.static_prefix.size(),
	                       backing_data,
	                       access_index.index_ranges,
	                       static_prefix_cursor);
	detail::scatter_ranges(access_index.static_prefix.data(),
	                       access_index.static_prefix.size(),
	                       backing_data,
	                       access_index.shared_ranges,
	                       static_prefix_cursor);
	if (static_prefix_cursor != access_index.static_prefix.size()) {
		throw std::runtime_error("sparse rowgroup static prefix size mismatch");
	}
	auto read_ranges = [&](std::vector<detail::SparseByteRange> ranges) {
		for (const auto& range : detail::coalesce_ranges(std::move(ranges))) {
			const auto start = std::chrono::steady_clock::now();
			m_file->ReadRangeUnchecked(backing_data + range.offset, rg->m_offset() + range.offset, range.size);
			const auto end = std::chrono::steady_clock::now();
			if (timing != nullptr) {
				timing->storage_bytes += range.size;
				++timing->pread_count;
				timing->pread_ms += std::chrono::duration<double, std::milli>(end - start).count();
				if (timing->pread_start == std::chrono::steady_clock::time_point {} || start < timing->pread_start) {
					timing->pread_start = start;
				}
				if (end > timing->pread_end) {
					timing->pread_end = end;
				}
			}
		}
	};

	std::vector<detail::SparseByteRange> payload_ranges;
	for (const uint32_t vector : unique_vectors) {
		const auto& ranges = access_index.vector_ranges.at(vector);
		payload_ranges.insert(payload_ranges.end(), ranges.begin(), ranges.end());
	}
	read_ranges(std::move(payload_ranges));
	if (timing != nullptr) {
		timing->used_sparse_read = true;
	}
}

void FlsReader::read_rowgroup_bytes_selected_columns_into(const size_t                rowgroup_idx,
	                                                       const std::vector<uint8_t>& selected_columns,
	                                                       std::byte* const           backing_data,
	                                                       const size_t               backing_capacity,
	                                                       ZeroCopyReadTiming*        timing) {
	if (selected_columns.empty()) {
		throw std::invalid_argument("selected-column read requires at least one column");
	}
	if (m_compact_descriptor == nullptr) {
		if (timing != nullptr) {
			timing->sparse_fallback_reason = "coefficient-range-read-requires-compact-v3";
		}
		read_rowgroup_bytes_into(rowgroup_idx, backing_data, backing_capacity, timing);
		return;
	}

	const auto record = m_compact_descriptor->rowgroup(rowgroup_idx);
	if ((record.payload_size != 0U && backing_data == nullptr) || backing_capacity < record.payload_size) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}
	if (rowgroup_idx > std::numeric_limits<uint32_t>::max()) {
		throw std::out_of_range("Compact v3 selected rowgroup exceeds uint32 range");
	}
	const auto plan = compile_compact_read_plan(
	    *m_compact_descriptor, {static_cast<uint32_t>(rowgroup_idx)}, selected_columns);
	if (record.payload_size != 0U) {
		std::memset(backing_data, 0, record.payload_size);
	}
	for (const auto& range : plan.ranges()) {
		const auto pread_start = std::chrono::steady_clock::now();
		m_file->ReadRangeUnchecked(backing_data + range.backing_offset, range.file_offset, range.size);
		const auto pread_end = std::chrono::steady_clock::now();
		if (timing != nullptr) {
			timing->storage_bytes += range.size;
			++timing->pread_count;
			timing->pread_ms += std::chrono::duration<double, std::milli>(pread_end - pread_start).count();
			if (timing->pread_start == std::chrono::steady_clock::time_point {} ||
			    pread_start < timing->pread_start) {
				timing->pread_start = pread_start;
			}
			if (pread_end > timing->pread_end) {
				timing->pread_end = pread_end;
			}
		}
	}
	if (timing != nullptr) {
		const auto& stats = plan.stats();
		timing->logical_storage_bytes       = static_cast<size_t>(stats.logical_bytes);
		timing->full_storage_bytes          = static_cast<size_t>(stats.full_rowgroup_bytes);
		timing->physical_page_bytes         = static_cast<size_t>(stats.physical_page_bytes);
		timing->full_physical_page_bytes    = static_cast<size_t>(stats.full_physical_page_bytes);
		timing->coalesced_read_run_count    = static_cast<size_t>(stats.coalesced_run_count);
		timing->selected_coefficient_count  = static_cast<size_t>(stats.selected_coefficient_count);
		timing->full_coefficient_count      = static_cast<size_t>(stats.full_coefficient_count);
		timing->sparse_read_supported       = true;
		timing->used_sparse_read            = stats.read_bytes < stats.full_rowgroup_bytes;
		timing->used_coefficient_range_read = true;
		if (record.payload_size == 0U) {
			timing->sparse_fallback_reason = "metadata-only-rowgroup-zero-io";
		} else if (!timing->used_sparse_read) {
			timing->sparse_fallback_reason = "selected-columns-cover-full-rowgroup";
		}
	}
}

ZeroCopyRowgroup FlsReader::make_zero_copy_rowgroup_from_backing(const size_t          rowgroup_idx,
                                                                 std::shared_ptr<void> backing_owner,
                                                                 std::byte* const      backing_data,
                                                                 const size_t          backing_capacity,
                                                                 const bool            backing_is_pinned,
	                                                             ZeroCopyReadTiming*   timing,
	                                                             const bool prefer_compact_direct_geometry) {
	const auto  setup_start = std::chrono::steady_clock::now();
	OwnedCompactRowgroupDescriptor compact_rowgroup;
	std::shared_ptr<const CompactV3DirectRowgroup> compact_direct;
	bool compact_direct_matches_schema_plan = false;
	const fastlanes::RowgroupDescriptor* rg = nullptr;
	if (m_compact_descriptor != nullptr) {
		if (rowgroup_idx >= m_compact_descriptor->rowgroup_count()) {
			throw std::out_of_range("rowgroup_idx out of range");
		}
		if (prefer_compact_direct_geometry && m_compact_descriptor->supports_direct_rowgroup_geometry()) {
			compact_direct = std::make_shared<CompactV3DirectRowgroup>(
			    m_compact_descriptor->decode_direct_rowgroup(rowgroup_idx));
			compact_direct_matches_schema_plan =
			    m_zero_copy_schema_plan != nullptr && m_zero_copy_schema_plan->enabled &&
			    compact_direct_rowgroup_matches_plan(*compact_direct, m_zero_copy_schema_plan->columns);
		}
		if (compact_direct == nullptr) {
			compact_rowgroup = make_compact_rowgroup_descriptor(*m_compact_descriptor, rowgroup_idx);
			rg               = compact_rowgroup.descriptor;
		}
	} else {
		const auto* td = m_table_descriptor != nullptr ? m_table_descriptor->Get() : nullptr;
		if (td == nullptr) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		const auto n_rgs = td->m_rowgroup_descriptors()->size();
		if (rowgroup_idx >= n_rgs) {
			throw std::out_of_range("rowgroup_idx out of range");
		}
		rg = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	}
	if (rg == nullptr && compact_direct == nullptr) {
		throw std::runtime_error("rowgroup descriptor is missing");
	}
	const size_t rg_bytes = compact_direct != nullptr ? compact_direct->record.payload_size
	                                                : static_cast<size_t>(rg->m_size());
	if ((rg_bytes != 0U && backing_data == nullptr) || backing_capacity < rg_bytes) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}
	std::byte* const effective_backing_data = rg_bytes == 0U ? metadata_only_backing() : backing_data;
	const bool       effective_pinned       = rg_bytes != 0U && backing_is_pinned;
	if (rg_bytes == 0U) {
		backing_owner.reset();
	}

	const size_t n_vecs   = compact_direct != nullptr ? 1U : static_cast<size_t>(rg->m_n_vec());
	const size_t n_values = n_vecs * galp::codec::consts::VALUES_PER_VECTOR;
	const size_t n_tuples = compact_direct != nullptr ? compact_direct->record.real_row_count
	                                                : static_cast<size_t>(rg->m_n_tuples());

	auto backing_span = fastlanes::span<std::byte> {effective_backing_data, rg_bytes};
	const auto* col_descs = rg == nullptr ? nullptr : rg->m_column_descriptors();
	const bool use_schema_plan = compact_direct_matches_schema_plan ||
	    (m_zero_copy_schema_plan && m_zero_copy_schema_plan->enabled && col_descs != nullptr &&
	     m_zero_copy_schema_plan->columns.size() == col_descs->size() &&
	     (m_compact_descriptor == nullptr ||
	      detail::rowgroup_matches_zero_copy_plan(*rg, m_zero_copy_schema_plan->columns)));

	std::shared_ptr<fastlanes::RowgroupView> view;
	if (!use_schema_plan && compact_direct == nullptr) {
		if (rg == nullptr) {
			throw std::runtime_error("zero-copy rowgroup fallback descriptor is missing");
		}
		view = std::make_shared<fastlanes::RowgroupView>(backing_span, *rg);
	}

	ZeroCopyRowgroup out {};
	out.rowgroup_index         = rowgroup_idx;
	out.n_values               = n_values;
	out.n_vecs                 = n_vecs;
	out.n_tuples               = n_tuples;
	out.table_descriptor_owner = m_table_descriptor;
	out.rowgroup_descriptor_owner = std::move(compact_rowgroup.owner);
	out.compact_descriptor_owner = m_compact_descriptor;
	out.compact_direct_owner     = std::move(compact_direct);
	out.rowgroup_descriptor    = rg;
	out.backing_owner          = std::move(backing_owner);
	out.backing_span           = backing_span;
	out.transfer_backing_span  = effective_pinned ? backing_span : fastlanes::span<std::byte> {};
	out.backing_capacity_bytes = rg_bytes == 0U ? 0U : backing_capacity;
	out.backing_is_pinned      = effective_pinned;
	out.rowgroup_view          = view;
	if (timing != nullptr) {
		timing->used_pinned_backing = effective_pinned;
		timing->logical_backing_capacity_bytes = out.backing_capacity_bytes;
	}

	const auto record_timing = [&](const std::chrono::steady_clock::time_point setup_end) {
		if (timing == nullptr) {
			return;
		}
		// A sparse read intentionally leaves holes in a full-size logical
		// backing.  Do not overwrite the physical-byte count recorded by the
		// reader with that logical backing size.
		timing->full_storage_bytes = rg_bytes;
		timing->zero_copy_view_setup_ms += std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
	};

	if (use_schema_plan) {
		out.schema_plan_owner = m_zero_copy_schema_plan;
		out.schema_plan       = out.schema_plan_owner.get();
		record_timing(std::chrono::steady_clock::now());
		return out;
	}

	if (out.compact_direct_owner != nullptr) {
		out.columns.reserve(out.compact_direct_owner->columns.size());
		for (size_t col_idx = 0U; col_idx < out.compact_direct_owner->columns.size(); ++col_idx) {
			const auto& direct_column = out.compact_direct_owner->columns[col_idx];
			const auto* schema        = direct_column.schema;
			const auto* rpn           = schema == nullptr ? nullptr : schema->encoding_rpn();
			const auto* ops           = rpn == nullptr ? nullptr : rpn->operator_tokens();
			if (ops == nullptr || ops->size() != 1U) {
				std::ostringstream msg;
				msg << "only single-op expressions are supported in compact zero-copy reader; got ops=[";
				if (ops != nullptr) {
					for (size_t i = 0U; i < ops->size(); ++i) {
						if (i > 0U) {
							msg << ", ";
						}
						msg << fastlanes::token_to_string(ops->Get(static_cast<flatbuffers::uoffset_t>(i)));
					}
				}
				msg << "]";
				const std::string col_name =
				    (m_load_column_names && schema != nullptr && schema->name() != nullptr) ? schema->name()->str()
				                                                                           : std::string {};
				throw galp::UnsupportedFormatError(msg.str(), rowgroup_idx, col_idx, col_name);
			}

			ZeroCopyColumn col {};
			col.column_index      = col_idx;
			col.name              = (m_load_column_names && schema->name() != nullptr) ? schema->name()->str()
			                                                                        : std::string {};
			col.token             = ops->Get(0U);
			col.column_descriptor = schema;
			col.operand_tokens    = rpn->operand_tokens();
			col.compact_rowgroup  = out.compact_direct_owner.get();
			col.compact_column    = &direct_column;
			col.column_span       = backing_span;
			if (col.token == fastlanes::OperatorToken::EXP_EQUAL && col.operand_tokens != nullptr &&
			    col.operand_tokens->size() >= 1U) {
				col.skip_decompress = true;
				col.alias_of        = static_cast<size_t>(col.operand_tokens->Get(0U));
			}
			out.columns.push_back(std::move(col));
		}
		record_timing(std::chrono::steady_clock::now());
		return out;
	}

	if (col_descs == nullptr) {
		throw std::runtime_error("zero-copy rowgroup column descriptors are missing");
	}
	out.columns.reserve(col_descs->size());
	for (size_t col_idx = 0; col_idx < col_descs->size(); ++col_idx) {
		const auto& col_desc = *col_descs->Get(static_cast<flatbuffers::uoffset_t>(col_idx));
		const auto* rpn      = col_desc.encoding_rpn();
		if (!rpn || !rpn->operator_tokens()) {
			throw std::runtime_error("missing encoding_rpn/operator_tokens");
		}
		const auto* ops = rpn->operator_tokens();
		if (ops->size() != 1) {
			std::ostringstream msg;
			msg << "only single-op expressions are supported in zero-copy reader; got ops=[";
			for (size_t i = 0; i < ops->size(); ++i) {
				if (i > 0) {
					msg << ", ";
				}
				msg << fastlanes::token_to_string(ops->Get(static_cast<flatbuffers::uoffset_t>(i)));
			}
			msg << "]";
			const std::string col_name =
			    (m_load_column_names && col_desc.name()) ? col_desc.name()->str() : std::string {};
			throw galp::UnsupportedFormatError(msg.str(), rowgroup_idx, col_idx, col_name);
		}

		ZeroCopyColumn col {};
		col.column_index      = col_idx;
		col.name              = (m_load_column_names && col_desc.name()) ? col_desc.name()->str() : std::string {};
		col.token             = ops->Get(0);
		col.column_descriptor = &col_desc;
		col.operand_tokens    = rpn->operand_tokens();
		col.column_view       = &(*view)[static_cast<fastlanes::n_t>(col_idx)];
		col.column_span       = backing_span;
		if (col.token == fastlanes::OperatorToken::EXP_EQUAL && col.operand_tokens && col.operand_tokens->size() >= 1) {
			col.skip_decompress = true;
			col.alias_of        = static_cast<size_t>(col.operand_tokens->Get(0));
		}
		out.columns.push_back(col);
	}

	record_timing(std::chrono::steady_clock::now());
	return out;
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_into(const size_t          rowgroup_idx,
                                                         std::shared_ptr<void> backing_owner,
                                                         std::byte* const      backing_data,
                                                         const size_t          backing_capacity,
                                                         const bool            backing_is_pinned,
                                                         ZeroCopyReadTiming*   timing) {
	read_rowgroup_bytes_into(rowgroup_idx, backing_data, backing_capacity, timing);
	return make_zero_copy_rowgroup_from_backing(
	    rowgroup_idx, std::move(backing_owner), backing_data, backing_capacity, backing_is_pinned, timing);
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_selected_vectors_into(
	const size_t                 rowgroup_idx,
	const std::vector<uint32_t>& selected_vectors,
	std::shared_ptr<void>        backing_owner,
	std::byte* const             backing_data,
	const size_t                 backing_capacity,
	const bool                   backing_is_pinned,
	ZeroCopyReadTiming*          timing) {
	read_rowgroup_bytes_selected_vectors_into(
	    rowgroup_idx, selected_vectors, backing_data, backing_capacity, timing);
	return make_zero_copy_rowgroup_from_backing(
	    rowgroup_idx, std::move(backing_owner), backing_data, backing_capacity, backing_is_pinned, timing);
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_selected_columns_into(
	const size_t                rowgroup_idx,
	const std::vector<uint8_t>& selected_columns,
	std::shared_ptr<void>       backing_owner,
	std::byte* const            backing_data,
	const size_t                backing_capacity,
	const bool                  backing_is_pinned,
	ZeroCopyReadTiming*         timing) {
	read_rowgroup_bytes_selected_columns_into(
	    rowgroup_idx, selected_columns, backing_data, backing_capacity, timing);
	auto rowgroup = make_zero_copy_rowgroup_from_backing(
	    rowgroup_idx, std::move(backing_owner), backing_data, backing_capacity, backing_is_pinned, timing);
	rowgroup.materialized_column_indices = selected_columns;
	return rowgroup;
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy(const size_t rowgroup_idx, ZeroCopyReadTiming* timing) {
	const size_t storage_bytes = rowgroup_storage_bytes(rowgroup_idx);
	if (storage_bytes == 0U) {
		return read_rowgroup_zero_copy_into(
		    rowgroup_idx, {}, nullptr, 0U, /*backing_is_pinned=*/false, timing);
	}
	auto backing = std::make_shared<fastlanes::Buf>(storage_bytes);
	return read_rowgroup_zero_copy_into(rowgroup_idx,
	                                    std::static_pointer_cast<void>(backing),
	                                    reinterpret_cast<std::byte*>(backing->mutable_data()),
	                                    backing->Capacity(),
	                                    /*backing_is_pinned=*/false,
	                                    timing);
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_selected_vectors(
	const size_t rowgroup_idx, const std::vector<uint32_t>& selected_vectors, ZeroCopyReadTiming* timing) {
	const size_t storage_bytes = rowgroup_storage_bytes(rowgroup_idx);
	if (storage_bytes == 0U) {
		return read_rowgroup_zero_copy_selected_vectors_into(
		    rowgroup_idx, selected_vectors, {}, nullptr, 0U, /*backing_is_pinned=*/false, timing);
	}
	auto backing = std::make_shared<fastlanes::Buf>(storage_bytes);
	return read_rowgroup_zero_copy_selected_vectors_into(rowgroup_idx,
	                                                     selected_vectors,
	                                                     std::static_pointer_cast<void>(backing),
	                                                     reinterpret_cast<std::byte*>(backing->mutable_data()),
	                                                     backing->Capacity(),
	                                                     /*backing_is_pinned=*/false,
	                                                     timing);
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_selected_columns(
	const size_t rowgroup_idx, const std::vector<uint8_t>& selected_columns, ZeroCopyReadTiming* timing) {
	const size_t storage_bytes = rowgroup_storage_bytes(rowgroup_idx);
	if (storage_bytes == 0U) {
		return read_rowgroup_zero_copy_selected_columns_into(
		    rowgroup_idx, selected_columns, {}, nullptr, 0U, /*backing_is_pinned=*/false, timing);
	}
	auto backing = std::make_shared<fastlanes::Buf>(storage_bytes);
	return read_rowgroup_zero_copy_selected_columns_into(rowgroup_idx,
	                                                    selected_columns,
	                                                    std::static_pointer_cast<void>(backing),
	                                                    reinterpret_cast<std::byte*>(backing->mutable_data()),
	                                                    backing->Capacity(),
	                                                    /*backing_is_pinned=*/false,
	                                                    timing);
}

std::vector<ZeroCopyRowgroup>
FlsReader::read_compact_rowgroups_zero_copy_scatter(const std::vector<size_t>&             rowgroup_indices,
	                                                std::vector<ZeroCopyReadTiming>* const timings,
	                                                const size_t view_workers) {
	if (m_compact_descriptor == nullptr) {
		throw std::invalid_argument("scatter rowgroup reads require Compact v3");
	}
	const size_t storage_bytes = compact_batch_backing_bytes(rowgroup_indices);
	auto backing = storage_bytes == 0U ? std::shared_ptr<fastlanes::Buf> {}
	                                  : std::make_shared<fastlanes::Buf>(storage_bytes);
	return read_compact_rowgroups_zero_copy_scatter_into(
	    rowgroup_indices,
	    std::static_pointer_cast<void>(backing),
	    backing == nullptr ? nullptr : reinterpret_cast<std::byte*>(backing->mutable_data()),
	    backing == nullptr ? 0U : backing->Capacity(),
	    /*backing_is_pinned=*/false,
	    timings,
	    view_workers);
}

std::vector<ZeroCopyRowgroup> FlsReader::read_compact_rowgroups_zero_copy_scatter_into(
	const std::vector<size_t>&             rowgroup_indices,
	std::shared_ptr<void>                  backing_owner,
	std::byte* const                       backing_data,
	const size_t                           backing_capacity,
	const bool                             backing_is_pinned,
	std::vector<ZeroCopyReadTiming>* const timings,
	const size_t                           view_workers) {
	if (m_compact_descriptor == nullptr) {
		throw std::invalid_argument("scatter rowgroup reads require Compact v3");
	}
	if (timings != nullptr) {
		timings->assign(rowgroup_indices.size(), ZeroCopyReadTiming {});
	}
	if (rowgroup_indices.empty()) {
		return {};
	}

	struct ReadEntry {
		 size_t                          output_index   = 0U;
		 size_t                          rowgroup_index = 0U;
		 CompactV3RowgroupRecord         record {};
		 size_t                          backing_offset = 0U;
	};
	std::vector<ReadEntry> entries;
	entries.reserve(rowgroup_indices.size());
	std::vector<size_t> output_index_by_rowgroup(m_compact_descriptor->rowgroup_count(),
	                                             std::numeric_limits<size_t>::max());
	std::vector<size_t> read_order;
	read_order.reserve(rowgroup_indices.size());
	size_t backing_cursor = 0U;
	for (size_t output_index = 0U; output_index < rowgroup_indices.size(); ++output_index) {
		const size_t rowgroup_index = rowgroup_indices[output_index];
		const auto   record         = m_compact_descriptor->rowgroup(rowgroup_index);
		if (output_index_by_rowgroup[rowgroup_index] != std::numeric_limits<size_t>::max()) {
			throw std::invalid_argument("Compact v3 scatter read contains a duplicate rowgroup");
		}
		output_index_by_rowgroup[rowgroup_index] = output_index;
		if (record.payload_size > std::numeric_limits<uint64_t>::max() - record.payload_offset ||
		    record.payload_offset + record.payload_size > m_file->Size()) {
			throw std::runtime_error("Compact v3 rowgroup payload exceeds the shard bounds");
		}
		const size_t backing_offset = record.payload_size == 0U ? 0U : align_compact_batch_offset(backing_cursor);
		if (record.payload_size > std::numeric_limits<size_t>::max() - backing_offset) {
			throw std::overflow_error("Compact v3 scatter backing size overflow");
		}
		if (record.payload_size != 0U) {
			backing_cursor = backing_offset + record.payload_size;
		}
		entries.push_back({output_index, rowgroup_index, record, backing_offset});
		if (record.payload_size != 0U) {
			read_order.push_back(output_index);
		}
		if (timings != nullptr) {
			auto& timing                      = timings->at(output_index);
			timing.storage_bytes              = record.payload_size;
			timing.logical_storage_bytes      = record.payload_size;
			timing.full_storage_bytes         = record.payload_size;
			timing.selected_coefficient_count = m_compact_descriptor->column_count();
			timing.full_coefficient_count     = m_compact_descriptor->column_count();
		}
	}
	const size_t required_capacity = backing_cursor;
	if ((required_capacity != 0U && (backing_owner == nullptr || backing_data == nullptr)) ||
	    backing_capacity < required_capacity) {
		throw std::invalid_argument("Compact v3 scatter external backing is null or too small");
	}
	std::sort(read_order.begin(), read_order.end(), [&](const size_t left, const size_t right) {
		return entries[left].record.payload_offset < entries[right].record.payload_offset;
	});

	struct ScatterRun {
		size_t   begin       = 0U;
		size_t   end         = 0U;
		uint64_t file_offset = 0U;
	};
	std::vector<ScatterRun> runs;
	runs.reserve(read_order.size());
	size_t run_begin = 0U;
	while (run_begin < read_order.size()) {
		size_t   run_end         = run_begin + 1U;
		const auto& first_entry  = entries[read_order[run_begin]];
		uint64_t expected_offset = first_entry.record.payload_offset + first_entry.record.payload_size;
		while (run_end < read_order.size() &&
		       entries[read_order[run_end]].record.payload_offset == expected_offset) {
			const auto& entry = entries[read_order[run_end]];
			if (entry.record.payload_size > std::numeric_limits<uint64_t>::max() - expected_offset) {
				throw std::overflow_error("Compact v3 scatter run exceeds uint64 range");
			}
			expected_offset += entry.record.payload_size;
			++run_end;
		}
		runs.push_back(ScatterRun {run_begin, run_end, first_entry.record.payload_offset});
		run_begin = run_end;
	}

	// Preserve physical-page accounting in file order. The actual preadv calls
	// below write disjoint backing spans and can therefore run independently.
	constexpr uint64_t page_size        = 4096U;
	uint64_t           covered_page_end = 0U;
	for (const auto& run : runs) {
		for (size_t read_position = run.begin; read_position < run.end; ++read_position) {
			const auto& entry = entries[read_order[read_position]];
			if (timings == nullptr) {
				continue;
			}
			auto&          timing = timings->at(entry.output_index);
			const uint64_t first_page = entry.record.payload_offset / page_size;
			const uint64_t last_page_exclusive =
			    (entry.record.payload_offset + entry.record.payload_size - 1U) / page_size + 1U;
			const uint64_t newly_covered_begin = std::max(first_page, covered_page_end);
			if (newly_covered_begin < last_page_exclusive) {
				timing.physical_page_bytes =
				    static_cast<size_t>((last_page_exclusive - newly_covered_begin) * page_size);
			}
			timing.full_physical_page_bytes = timing.physical_page_bytes;
			covered_page_end                = std::max(covered_page_end, last_page_exclusive);
		}
	}

	parallel_for_compact_views(runs.size(), view_workers, [&](const size_t run_index) {
		const auto& run         = runs[run_index];
		const auto& first_entry = entries[read_order[run.begin]];
		std::vector<fastlanes::FileScatterReadTarget> targets;
		targets.reserve(run.end - run.begin);
		for (size_t read_position = run.begin; read_position < run.end; ++read_position) {
			auto& entry = entries[read_order[read_position]];
			targets.push_back({backing_data + entry.backing_offset, entry.record.payload_size});
		}
		const auto pread_start = std::chrono::steady_clock::now();
		const auto pread_count = m_file->ReadScatterUnchecked(targets, run.file_offset);
		const auto pread_end   = std::chrono::steady_clock::now();

		if (timings != nullptr) {
			auto& run_timing       = timings->at(first_entry.output_index);
			run_timing.pread_count = pread_count;
#if !defined(_WIN32)
			run_timing.preadv_count = pread_count;
#endif
			run_timing.coalesced_read_run_count = pread_count;
			run_timing.pread_ms    = std::chrono::duration<double, std::milli>(pread_end - pread_start).count();
			run_timing.pread_start = pread_start;
			run_timing.pread_end   = pread_end;
		}
	});

	std::vector<ZeroCopyRowgroup> rowgroups(rowgroup_indices.size());
	parallel_for_compact_views(entries.size(), view_workers, [&](const size_t entry_index) {
		auto&       entry  = entries[entry_index];
		auto* const timing = timings == nullptr ? nullptr : &timings->at(entry.output_index);
		auto* const rowgroup_backing =
		    entry.record.payload_size == 0U ? nullptr : backing_data + entry.backing_offset;
		rowgroups[entry.output_index] =
		    make_zero_copy_rowgroup_from_backing(entry.rowgroup_index,
		                                         backing_owner,
		                                         rowgroup_backing,
		                                         entry.record.payload_size,
		                                         backing_is_pinned,
		                                         timing,
		                                         /*prefer_compact_direct_geometry=*/true);
		if (backing_is_pinned && required_capacity != 0U) {
			rowgroups[entry.output_index].transfer_backing_span =
			    fastlanes::span<std::byte> {backing_data, required_capacity};
		}
	});
	return rowgroups;
}

std::vector<ZeroCopyRowgroup>
FlsReader::read_compact_rowgroups_zero_copy_selected_columns(const std::vector<size_t>&             rowgroup_indices,
                                                             const std::vector<uint8_t>&            selected_columns,
	                                                         std::vector<ZeroCopyReadTiming>* const timings,
	                                                         const size_t view_workers) {
	if (m_compact_descriptor == nullptr) {
		throw std::invalid_argument("batched selected-column reads require Compact v3");
	}
	const size_t storage_bytes = compact_batch_backing_bytes(rowgroup_indices);
	auto backing = storage_bytes == 0U ? std::shared_ptr<fastlanes::Buf> {}
	                                  : std::make_shared<fastlanes::Buf>(storage_bytes);
	return read_compact_rowgroups_zero_copy_selected_columns_into(
	    rowgroup_indices,
	    selected_columns,
	    std::static_pointer_cast<void>(backing),
	    backing == nullptr ? nullptr : reinterpret_cast<std::byte*>(backing->mutable_data()),
	    backing == nullptr ? 0U : backing->Capacity(),
	    /*backing_is_pinned=*/false,
	    timings,
	    view_workers);
}

std::vector<ZeroCopyRowgroup> FlsReader::read_compact_rowgroups_zero_copy_selected_columns_into(
	const std::vector<size_t>&             rowgroup_indices,
	const std::vector<uint8_t>&            selected_columns,
	std::shared_ptr<void>                  backing_owner,
	std::byte* const                       backing_data,
	const size_t                           backing_capacity,
	const bool                             backing_is_pinned,
	std::vector<ZeroCopyReadTiming>* const timings,
	const size_t                           view_workers) {
	if (m_compact_descriptor == nullptr) {
		throw std::invalid_argument("batched selected-column reads require Compact v3");
	}
	if (selected_columns.empty()) {
		throw std::invalid_argument("batched selected-column read requires at least one column");
	}
	if (timings != nullptr) {
		timings->assign(rowgroup_indices.size(), ZeroCopyReadTiming {});
	}
	if (rowgroup_indices.empty()) {
		return {};
	}

	struct ReadEntry {
		size_t                          rowgroup_index = 0U;
		CompactV3RowgroupRecord         record {};
		size_t                          backing_offset = 0U;
	};
	std::vector<ReadEntry> entries;
	entries.reserve(rowgroup_indices.size());
	std::vector<size_t>   output_index_by_rowgroup(m_compact_descriptor->rowgroup_count(),
                                                 std::numeric_limits<size_t>::max());
	std::vector<uint32_t> plan_rowgroups;
	plan_rowgroups.reserve(rowgroup_indices.size());
	size_t backing_cursor = 0U;
	for (size_t output_index = 0U; output_index < rowgroup_indices.size(); ++output_index) {
		const size_t rowgroup_index = rowgroup_indices[output_index];
		if (rowgroup_index > std::numeric_limits<uint32_t>::max()) {
			throw std::out_of_range("Compact v3 batched rowgroup exceeds uint32 range");
		}
		const auto record = m_compact_descriptor->rowgroup(rowgroup_index);
		if (output_index_by_rowgroup[rowgroup_index] != std::numeric_limits<size_t>::max()) {
			throw std::invalid_argument("Compact v3 selected-column batch contains a duplicate rowgroup");
		}
		output_index_by_rowgroup[rowgroup_index] = output_index;
		plan_rowgroups.push_back(static_cast<uint32_t>(rowgroup_index));
		const size_t backing_offset = record.payload_size == 0U ? 0U : align_compact_batch_offset(backing_cursor);
		if (record.payload_size > std::numeric_limits<size_t>::max() - backing_offset) {
			throw std::overflow_error("Compact v3 selected-column backing size overflow");
		}
		if (record.payload_size != 0U) {
			backing_cursor = backing_offset + record.payload_size;
		}
		entries.push_back({rowgroup_index, record, backing_offset});
	}
	const size_t required_capacity = backing_cursor;
	if ((required_capacity != 0U && (backing_owner == nullptr || backing_data == nullptr)) ||
	    backing_capacity < required_capacity) {
		throw std::invalid_argument("Compact v3 selected-column external backing is null or too small");
	}
	if (required_capacity != 0U) {
		std::memset(backing_data, 0, required_capacity);
	}

	const auto plan = compile_compact_read_plan(*m_compact_descriptor, plan_rowgroups, selected_columns);
	for (const auto& range : plan.ranges()) {
		if (range.rowgroup_index >= output_index_by_rowgroup.size()) {
			throw std::runtime_error("Compact v3 selected-column plan references an unknown rowgroup");
		}
		const size_t output_index = output_index_by_rowgroup[range.rowgroup_index];
		if (output_index == std::numeric_limits<size_t>::max()) {
			throw std::runtime_error("Compact v3 selected-column plan references an unrequested rowgroup");
		}
		auto& entry = entries[output_index];
		if (range.backing_offset > entry.record.payload_size ||
		    range.size > entry.record.payload_size - range.backing_offset) {
			throw std::runtime_error("Compact v3 selected-column range exceeds its rowgroup backing");
		}
		const auto pread_start = std::chrono::steady_clock::now();
			m_file->ReadRangeUnchecked(backing_data + entry.backing_offset + range.backing_offset,
		                           range.file_offset,
		                           range.size);
		const auto pread_end = std::chrono::steady_clock::now();
		if (timings != nullptr) {
			auto& timing = timings->at(output_index);
			timing.storage_bytes += range.size;
			++timing.pread_count;
			++timing.coalesced_read_run_count;
			timing.pread_ms += std::chrono::duration<double, std::milli>(pread_end - pread_start).count();
			if (timing.pread_start == std::chrono::steady_clock::time_point {} || pread_start < timing.pread_start) {
				timing.pread_start = pread_start;
			}
			if (pread_end > timing.pread_end) {
				timing.pread_end = pread_end;
			}
		}
	}
	if (timings != nullptr) {
		const auto& stats = plan.stats();
		if (stats.logical_bytes > std::numeric_limits<size_t>::max() ||
		    stats.physical_page_bytes > std::numeric_limits<size_t>::max() ||
		    stats.full_physical_page_bytes > std::numeric_limits<size_t>::max()) {
			throw std::overflow_error("Compact v3 selected-column metrics exceed addressable memory");
		}
		auto& aggregate_timing                    = timings->front();
		aggregate_timing.logical_storage_bytes    = static_cast<size_t>(stats.logical_bytes);
		aggregate_timing.physical_page_bytes      = static_cast<size_t>(stats.physical_page_bytes);
		aggregate_timing.full_physical_page_bytes = static_cast<size_t>(stats.full_physical_page_bytes);
		for (size_t output_index = 0U; output_index < entries.size(); ++output_index) {
			auto& timing                       = timings->at(output_index);
			timing.full_storage_bytes          = entries[output_index].record.payload_size;
			timing.selected_coefficient_count  = static_cast<size_t>(stats.selected_coefficient_count);
			timing.full_coefficient_count      = static_cast<size_t>(stats.full_coefficient_count);
			timing.sparse_read_supported       = true;
			timing.used_sparse_read            = timing.storage_bytes < timing.full_storage_bytes;
			timing.used_coefficient_range_read = true;
			if (entries[output_index].record.payload_size == 0U) {
				timing.sparse_fallback_reason = "metadata-only-rowgroup-zero-io";
			} else if (!timing.used_sparse_read) {
				timing.sparse_fallback_reason = "selected-columns-cover-full-rowgroup";
			}
		}
	}

	std::vector<ZeroCopyRowgroup> rowgroups(rowgroup_indices.size());
	parallel_for_compact_views(entries.size(), view_workers, [&](const size_t output_index) {
		auto&       entry  = entries[output_index];
		auto* const timing = timings == nullptr ? nullptr : &timings->at(output_index);
		auto* const rowgroup_backing =
		    entry.record.payload_size == 0U ? nullptr : backing_data + entry.backing_offset;
		rowgroups[output_index] =
		    make_zero_copy_rowgroup_from_backing(entry.rowgroup_index,
		                                         backing_owner,
		                                         rowgroup_backing,
		                                         entry.record.payload_size,
		                                         backing_is_pinned,
		                                         timing,
		                                         /*prefer_compact_direct_geometry=*/true);
		if (backing_is_pinned && required_capacity != 0U) {
			rowgroups[output_index].transfer_backing_span =
			    fastlanes::span<std::byte> {backing_data, required_capacity};
		}
		rowgroups[output_index].materialized_column_indices = selected_columns;
	});
	return rowgroups;
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_selected_vectors_packed(
    const size_t rowgroup_idx, const std::vector<uint32_t>& selected_vectors, ZeroCopyReadTiming* timing) {
	if (m_compact_descriptor != nullptr) {
		return read_rowgroup_zero_copy_selected_vectors(rowgroup_idx, selected_vectors, timing);
	}
	const auto* td = m_table_descriptor->Get();
	if (td == nullptr) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto* rowgroups = td->m_rowgroup_descriptors();
	if (rowgroups == nullptr || rowgroup_idx >= rowgroups->size()) {
		throw std::out_of_range("rowgroup_idx out of range");
	}
	if (selected_vectors.empty()) {
		throw std::invalid_argument("selected vector read requires at least one vector");
	}
	const auto*           rowgroup       = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	const size_t          rowgroup_bytes = static_cast<size_t>(rowgroup->m_size());
	const size_t          vector_count   = static_cast<size_t>(rowgroup->m_n_vec());
	std::vector<uint32_t> unique_vectors = selected_vectors;
	std::sort(unique_vectors.begin(), unique_vectors.end());
	unique_vectors.erase(std::unique(unique_vectors.begin(), unique_vectors.end()), unique_vectors.end());
	if (unique_vectors.back() >= vector_count) {
		throw std::out_of_range("selected vector exceeds rowgroup vector count");
	}

	const auto  segments = detail::rowgroup_segment_descriptors(*rowgroup);
	std::string capability_reason;
	const bool  supported = detail::validate_sparse_vector_segments(*rowgroup, segments, &capability_reason);
	if (!m_sparse_vector_bundle || !supported || unique_vectors.size() >= vector_count) {
		return read_rowgroup_zero_copy_selected_vectors(rowgroup_idx, selected_vectors, timing);
	}
	if (rowgroup_idx >= m_sparse_vector_bundle->rowgroups.size()) {
		throw std::runtime_error("sparse vector bundle rowgroup index is out of range");
	}
	const auto& bundle_entry = m_sparse_vector_bundle->rowgroups[rowgroup_idx];
	if (bundle_entry.prefix_size > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error("sparse vector bundle prefix exceeds addressable memory");
	}

	size_t packed_bytes = static_cast<size_t>(bundle_entry.prefix_size);
	for (const uint32_t vector : unique_vectors) {
		const uint64_t begin = bundle_entry.vector_offsets.at(vector);
		const uint64_t end   = bundle_entry.vector_offsets.at(vector + 1U);
		if (end < begin || end - begin > std::numeric_limits<size_t>::max() - packed_bytes) {
			throw std::runtime_error("sparse vector bundle packed payload size overflow");
		}
		packed_bytes += static_cast<size_t>(end - begin);
	}

	auto logical = std::make_shared<fastlanes::Buf>(rowgroup_bytes);
	auto packed  = std::make_shared<std::vector<std::byte>>(packed_bytes);
	auto* const logical_data = reinterpret_cast<std::byte*>(logical->mutable_data());
	auto* const packed_data  = packed->data();
	auto device_payload = std::make_shared<galp::execution::PackedRowgroupDevicePayload>();
	device_payload->packed_owner  = std::static_pointer_cast<void>(packed);
	device_payload->packed_data   = packed_data;
	device_payload->packed_bytes  = packed_bytes;
	device_payload->packed_capacity_bytes = packed->capacity();
	device_payload->logical_data  = logical_data;
	device_payload->logical_bytes = rowgroup_bytes;
	device_payload->ranges.reserve(
	    segments.size() * unique_vectors.size() + 2U * segments.size());

	const auto record_read = [&](const uint64_t file_offset,
	                             const size_t   size,
	                             std::byte* const destination) {
		if (size == 0U) {
			return;
		}
		const auto start = std::chrono::steady_clock::now();
		m_sparse_vector_bundle->file->ReadRangeUnchecked(destination, file_offset, size);
		const auto end = std::chrono::steady_clock::now();
		if (timing != nullptr) {
			timing->storage_bytes += size;
			++timing->pread_count;
			timing->pread_ms += std::chrono::duration<double, std::milli>(end - start).count();
			if (timing->pread_start == std::chrono::steady_clock::time_point {} || start < timing->pread_start) {
				timing->pread_start = start;
			}
			if (end > timing->pread_end) {
				timing->pread_end = end;
			}
		}
	};

	record_read(bundle_entry.prefix_offset, static_cast<size_t>(bundle_entry.prefix_size), packed_data);
	size_t packed_cursor = 0U;
	const auto scatter_prefix_ranges = [&](const std::vector<detail::SparseByteRange>& ranges) {
		for (const auto& range : ranges) {
			if (packed_cursor > packed_bytes || range.size > packed_bytes - packed_cursor) {
				throw std::runtime_error("truncated sparse vector bundle rowgroup prefix");
			}
			std::memcpy(logical_data + range.offset, packed_data + packed_cursor, range.size);
			device_payload->ranges.push_back(
			    galp::execution::PackedRowgroupScatterRange {packed_cursor, range.offset, range.size});
			packed_cursor += range.size;
		}
	};
	const auto index_ranges = detail::segment_index_ranges(segments);
	scatter_prefix_ranges(index_ranges);
	const auto shared_ranges = detail::segment_shared_ranges(segments, logical_data);
	scatter_prefix_ranges(shared_ranges);
	if (packed_cursor != bundle_entry.prefix_size) {
		throw std::runtime_error("sparse vector bundle rowgroup prefix size mismatch");
	}

	for (size_t selected_pos = 0U; selected_pos < unique_vectors.size();) {
		const size_t run_packed_begin = packed_cursor;
		const uint32_t run_begin = unique_vectors[selected_pos];
		uint32_t       run_end   = run_begin + 1U;
		++selected_pos;
		while (selected_pos < unique_vectors.size() && unique_vectors[selected_pos] == run_end) {
			++run_end;
			++selected_pos;
		}
		const uint64_t bundle_begin = bundle_entry.vector_offsets.at(run_begin);
		const uint64_t bundle_end   = bundle_entry.vector_offsets.at(run_end);
		if (bundle_end < bundle_begin || bundle_end - bundle_begin > packed_bytes - packed_cursor) {
			throw std::runtime_error("invalid sparse vector bundle selected-vector run");
		}
		record_read(bundle_entry.vectors_offset + bundle_begin,
		            static_cast<size_t>(bundle_end - bundle_begin),
		            packed_data + packed_cursor);
		for (uint32_t vector = run_begin; vector < run_end; ++vector) {
			for (const auto* segment : segments) {
				if (detail::segment_entrypoint_count(*segment) == 1U) {
					continue;
				}
				const auto range = detail::segment_vector_range(*segment, logical_data, vector);
				if (range.size > packed_bytes - packed_cursor) {
					throw std::runtime_error("truncated sparse vector bundle selected-vector payload");
				}
				device_payload->ranges.push_back(
				    galp::execution::PackedRowgroupScatterRange {packed_cursor, range.offset, range.size});
				packed_cursor += range.size;
			}
		}
		if (packed_cursor - run_packed_begin != static_cast<size_t>(bundle_end - bundle_begin)) {
			throw std::runtime_error("sparse vector bundle selected-vector payload size mismatch");
		}
	}
	if (packed_cursor != packed_bytes) {
		throw std::runtime_error("sparse vector bundle packed payload size mismatch");
	}
	if (std::getenv("GALP_VECTOR_BUNDLE_DEVICE_SCATTER_CPU_MIRROR") != nullptr) {
		const char* max_range_env = std::getenv("GALP_VECTOR_BUNDLE_DEVICE_SCATTER_CPU_MIRROR_MAX_BYTES");
		size_t      max_range_bytes = std::numeric_limits<size_t>::max();
		if (max_range_env != nullptr && *max_range_env != '\0') {
			char*              end   = nullptr;
			const unsigned long long value = std::strtoull(max_range_env, &end, 10);
			if (end == max_range_env || *end != '\0' || value > std::numeric_limits<size_t>::max()) {
				throw std::invalid_argument(
				    "GALP_VECTOR_BUNDLE_DEVICE_SCATTER_CPU_MIRROR_MAX_BYTES must be a non-negative integer");
			}
			max_range_bytes = static_cast<size_t>(value);
		}
		for (const auto& range : device_payload->ranges) {
			if (range.size <= max_range_bytes) {
				std::memcpy(logical_data + range.logical_offset, packed_data + range.packed_offset, range.size);
			}
		}
	}
	if (timing != nullptr) {
		timing->full_storage_bytes      = rowgroup_bytes;
		timing->sparse_read_supported   = true;
		timing->used_sparse_read        = true;
		timing->used_vector_bundle_read = true;
	}

	auto zero_copy = make_zero_copy_rowgroup_from_backing(rowgroup_idx,
	                                                      std::static_pointer_cast<void>(logical),
	                                                      logical_data,
	                                                      logical->Capacity(),
	                                                      /*backing_is_pinned=*/false,
	                                                      timing);
	zero_copy.packed_device_payload = std::move(device_payload);
	return zero_copy;
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_compiled(const SparseVectorReadPlan& compiled,
	                                                          ZeroCopyReadTiming* const timing) {
	if (!compiled.impl_) {
		throw std::invalid_argument("compiled sparse vector read plan is empty");
	}
	const auto& plan = *compiled.impl_;
	if (plan.owner != m_sparse_plan_owner) {
		throw std::invalid_argument("compiled sparse vector read plan belongs to a different reader");
	}
	if (plan.strategy == SparseVectorReadPlan::Impl::Strategy::kFullRowgroup) {
		if (timing != nullptr) {
			timing->sparse_fallback_reason = plan.fallback_reason;
			timing->sparse_read_supported = plan.fallback_reason == "selected-vectors-cover-full-rowgroup";
		}
		auto zero_copy = read_rowgroup_zero_copy(plan.rowgroup_index, timing);
		zero_copy.materialized_column_indices = plan.materialized_columns;
		return zero_copy;
	}
	if (!plan.access) {
		throw std::runtime_error("compiled sparse rowgroup access index is unavailable");
	}
	const auto& access = *plan.access;
	const auto* rowgroup = table_descriptor()->m_rowgroup_descriptors()->Get(
	    static_cast<flatbuffers::uoffset_t>(plan.rowgroup_index));
	if (rowgroup == nullptr || static_cast<size_t>(rowgroup->m_size()) != plan.rowgroup_bytes) {
		throw std::runtime_error("compiled sparse vector read plan no longer matches its rowgroup");
	}

	auto logical = std::make_shared<fastlanes::Buf>(plan.rowgroup_bytes);
	auto* const logical_data = reinterpret_cast<std::byte*>(logical->mutable_data());
	std::memset(logical_data, 0, plan.rowgroup_bytes);
	const auto restore_static_prefix = [&]() {
		const auto start = std::chrono::steady_clock::now();
		size_t prefix_cursor = 0U;
		detail::scatter_ranges(access.static_prefix.data(),
		                       access.static_prefix.size(),
		                       logical_data,
		                       access.index_ranges,
		                       prefix_cursor);
		detail::scatter_ranges(access.static_prefix.data(),
		                       access.static_prefix.size(),
		                       logical_data,
		                       access.shared_ranges,
		                       prefix_cursor);
		if (prefix_cursor != access.static_prefix.size()) {
			throw std::runtime_error("compiled sparse rowgroup prefix size mismatch");
		}
		if (timing != nullptr) {
			timing->static_prefix_restore_bytes += access.static_prefix.size();
			timing->static_prefix_restore_ms += std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - start).count();
		}
	};
	const bool bounded_source_ranges =
	    plan.strategy == SparseVectorReadPlan::Impl::Strategy::kBoundedSourceRanges;
	if (!bounded_source_ranges) {
		restore_static_prefix();
	}

	const auto record_read = [&](const std::chrono::steady_clock::time_point start,
	                             const std::chrono::steady_clock::time_point end,
	                             const size_t bytes,
	                             const size_t reads) {
		if (timing == nullptr) {
			return;
		}
		timing->storage_bytes += bytes;
		timing->pread_count += reads;
		timing->pread_ms += std::chrono::duration<double, std::milli>(end - start).count();
		if (timing->pread_start == std::chrono::steady_clock::time_point {} || start < timing->pread_start) {
			timing->pread_start = start;
		}
		if (end > timing->pread_end) {
			timing->pread_end = end;
		}
	};

	std::shared_ptr<galp::execution::PackedRowgroupDevicePayload> device_payload;
	if (plan.strategy == SparseVectorReadPlan::Impl::Strategy::kSourceRanges || bounded_source_ranges) {
		for (const auto& range : plan.source_ranges) {
			if (range.offset > plan.rowgroup_bytes || range.size > plan.rowgroup_bytes - range.offset) {
				throw std::runtime_error("compiled sparse source range is out of bounds");
			}
		}
		if (bounded_source_ranges &&
		    plan.submission_backend == SparseVectorReadPlan::SubmissionBackend::kIoUring) {
			std::vector<fastlanes::FileRangeReadTarget> targets;
			targets.reserve(plan.source_ranges.size());
			for (const auto& range : plan.source_ranges) {
				if (range.offset > std::numeric_limits<uint64_t>::max() - rowgroup->m_offset()) {
					throw std::overflow_error("compiled sparse io_uring file offset overflow");
				}
				targets.push_back({logical_data + range.offset, rowgroup->m_offset() + range.offset, range.size});
			}
			const auto start = std::chrono::steady_clock::now();
			const auto result = m_file->ReadRangesIoUringUnchecked(targets, plan.io_uring_queue_depth);
			const auto end = std::chrono::steady_clock::now();
			const auto initial_request_count =
			    static_cast<size_t>(std::count_if(targets.begin(), targets.end(), [](const auto& target) {
				    return target.size != 0U;
			    }));
			if (result.bytes != plan.storage_bytes || result.read_request_count < initial_request_count ||
			    result.completion_count != result.read_request_count) {
				throw std::runtime_error("compiled sparse io_uring result does not match the frozen physical plan");
			}
			if (timing != nullptr) {
				timing->storage_bytes += result.bytes;
				timing->io_uring_read_request_count += result.read_request_count;
				timing->io_uring_completion_count += result.completion_count;
				timing->io_uring_submit_syscall_count += result.submit_syscall_count;
				timing->io_uring_wait_syscall_count += result.wait_syscall_count;
				timing->io_uring_ring_mapped_bytes =
				    std::max(timing->io_uring_ring_mapped_bytes, static_cast<size_t>(result.ring_mapped_bytes));
				timing->io_uring_newly_mapped_ring_bytes += result.newly_mapped_ring_bytes;
				timing->io_uring_ms += std::chrono::duration<double, std::milli>(end - start).count();
				timing->pread_ms += std::chrono::duration<double, std::milli>(end - start).count();
				timing->pread_start = start;
				timing->pread_end = end;
				timing->used_io_uring = true;
			}
		} else {
			for (const auto& range : plan.source_ranges) {
				const auto start = std::chrono::steady_clock::now();
				m_file->ReadRangeUnchecked(
				    logical_data + range.offset, rowgroup->m_offset() + range.offset, range.size);
				const auto end = std::chrono::steady_clock::now();
				record_read(start, end, range.size, 1U);
			}
		}
		if (bounded_source_ranges) {
			const auto clear_start = std::chrono::steady_clock::now();
			size_t cleared_bytes = 0U;
			for (const auto& hole : plan.merged_holes) {
				if (hole.offset > plan.rowgroup_bytes || hole.size > plan.rowgroup_bytes - hole.offset ||
				    hole.size > std::numeric_limits<size_t>::max() - cleared_bytes) {
					throw std::runtime_error("compiled sparse merged hole is out of bounds");
				}
				std::memset(logical_data + hole.offset, 0, hole.size);
				cleared_bytes += hole.size;
			}
			if (cleared_bytes != plan.merged_gap_bytes) {
				throw std::runtime_error("compiled sparse merged-hole byte count mismatch");
			}
			if (timing != nullptr) {
				timing->merged_gap_bytes += plan.merged_gap_bytes;
				timing->hole_clear_bytes += cleared_bytes;
				timing->hole_clear_ms += std::chrono::duration<double, std::milli>(
				    std::chrono::steady_clock::now() - clear_start).count();
				timing->used_bounded_gap_read = true;
			}
			restore_static_prefix();
		}
	} else if (plan.strategy == SparseVectorReadPlan::Impl::Strategy::kBundleEnvelope) {
		auto envelope = std::make_shared<std::vector<std::byte>>(plan.envelope_size);
		const auto start = std::chrono::steady_clock::now();
		m_sparse_vector_bundle->file->ReadRangeUnchecked(
		    envelope->data(), plan.envelope_file_offset, plan.envelope_size);
		const auto end = std::chrono::steady_clock::now();
		record_read(start, end, plan.envelope_size, 1U);
		for (const auto& copy : plan.envelope_copies) {
			if (copy.source_offset > envelope->size() || copy.size > envelope->size() - copy.source_offset ||
			    copy.logical_offset > plan.rowgroup_bytes || copy.size > plan.rowgroup_bytes - copy.logical_offset) {
				throw std::runtime_error("compiled sparse bundle envelope copy is out of range");
			}
			std::memcpy(logical_data + copy.logical_offset, envelope->data() + copy.source_offset, copy.size);
		}
	} else if (plan.strategy == SparseVectorReadPlan::Impl::Strategy::kBundleRuns) {
		for (const auto& run : plan.bundle_runs) {
			std::vector<fastlanes::FileScatterReadTarget> targets;
			targets.reserve(run.logical_ranges.size());
			for (const auto& range : run.logical_ranges) {
				targets.push_back({logical_data + range.offset, range.size});
			}
			const auto start = std::chrono::steady_clock::now();
			const auto reads = m_sparse_vector_bundle->file->ReadScatterUnchecked(targets, run.file_offset);
			const auto end = std::chrono::steady_clock::now();
			record_read(start, end, run.size, static_cast<size_t>(reads));
		}
	} else if (plan.strategy == SparseVectorReadPlan::Impl::Strategy::kBundlePacked) {
		auto packed = std::make_shared<std::vector<std::byte>>(plan.packed_bytes);
		std::memcpy(packed->data(), access.static_prefix.data(), access.static_prefix.size());
		for (const auto& run : plan.bundle_runs) {
			if (run.packed_offset > packed->size() || run.size > packed->size() - run.packed_offset) {
				throw std::runtime_error("compiled sparse bundle packed run is out of range");
			}
			const auto start = std::chrono::steady_clock::now();
			m_sparse_vector_bundle->file->ReadRangeUnchecked(
			    packed->data() + run.packed_offset, run.file_offset, run.size);
			const auto end = std::chrono::steady_clock::now();
			record_read(start, end, run.size, 1U);
		}
		device_payload = std::make_shared<galp::execution::PackedRowgroupDevicePayload>();
		device_payload->packed_owner  = std::static_pointer_cast<void>(packed);
		device_payload->packed_data   = packed->data();
		device_payload->packed_bytes  = packed->size();
		device_payload->packed_capacity_bytes = packed->capacity();
		device_payload->logical_data  = logical_data;
		device_payload->logical_bytes = plan.rowgroup_bytes;
		device_payload->ranges        = plan.packed_scatter_ranges;
		if (std::getenv("GALP_VECTOR_BUNDLE_DEVICE_SCATTER_CPU_MIRROR") != nullptr) {
			for (const auto& range : device_payload->ranges) {
				std::memcpy(logical_data + range.logical_offset,
				            packed->data() + range.packed_offset,
				            range.size);
			}
		}
	}
	if (timing != nullptr) {
		timing->full_storage_bytes      = plan.rowgroup_bytes;
		timing->selected_storage_bytes  = plan.selected_storage_bytes;
		timing->sparse_read_supported   = true;
		timing->used_sparse_read        = true;
		timing->used_vector_bundle_read =
		    plan.strategy != SparseVectorReadPlan::Impl::Strategy::kSourceRanges &&
		    plan.strategy != SparseVectorReadPlan::Impl::Strategy::kBoundedSourceRanges;
		timing->used_vector_bundle_envelope_read =
		    plan.strategy == SparseVectorReadPlan::Impl::Strategy::kBundleEnvelope;
		if (!plan.materialized_columns.empty()) {
			timing->logical_storage_bytes       = plan.selected_storage_bytes;
			timing->selected_coefficient_count  = plan.materialized_columns.size();
			timing->full_coefficient_count      = rowgroup->m_column_descriptors()->size();
			timing->used_coefficient_range_read = true;
		}
	}
	auto zero_copy = make_zero_copy_rowgroup_from_backing(plan.rowgroup_index,
	                                                      std::static_pointer_cast<void>(logical),
	                                                      logical_data,
	                                                      logical->Capacity(),
	                                                      /*backing_is_pinned=*/false,
	                                                      timing);
	zero_copy.materialized_column_indices = plan.materialized_columns;
	zero_copy.packed_device_payload = std::move(device_payload);
	return zero_copy;
}

Rowgroup FlsReader::materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy) const {
	return detail::materialize_zero_copy_rowgroup(std::move(zero_copy));
}

Rowgroup FlsReader::read_rowgroup_zero_copy_materialized(const size_t rowgroup_idx) {
	return materialize_zero_copy_rowgroup(read_rowgroup_zero_copy(rowgroup_idx));
}

Rowgroup FlsReader::read_rowgroup(const size_t rowgroup_idx) {
	return detail::make_owning_rowgroup(read_rowgroup_zero_copy_materialized(rowgroup_idx));
}

std::vector<Rowgroup> FlsReader::read_table() {
	const size_t          n_rgs = rowgroup_count();
	std::vector<Rowgroup> out;
	out.reserve(n_rgs);
	for (size_t rg_idx = 0; rg_idx < n_rgs; ++rg_idx) {
		out.emplace_back(read_rowgroup(rg_idx));
	}
	return out;
}

ZeroCopySchemaPlan FlsReader::build_shared_zero_copy_schema_plan() const {
	ZeroCopySchemaPlan plan {};
	if (m_compact_descriptor != nullptr) {
		if (m_compact_descriptor->rowgroup_count() == 0U) {
			return plan;
		}
		const auto first = make_compact_rowgroup_descriptor(*m_compact_descriptor, 0U);
		if (first.descriptor == nullptr) {
			return plan;
		}
		try {
			plan.columns     = detail::build_zero_copy_column_plan(*first.descriptor, m_load_column_names);
			plan.build_order = detail::build_zero_copy_column_order(plan.columns);
			plan.enabled     = true;
		} catch (const std::exception&) {
			plan.columns.clear();
			plan.build_order.clear();
		}
		return plan;
	}
	const auto*        td = m_table_descriptor->Get();
	if (!td || !td->m_rowgroup_descriptors() || td->m_rowgroup_descriptors()->size() == 0) {
		return plan;
	}

	const auto* rowgroups      = td->m_rowgroup_descriptors();
	const auto* first_rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(0));
	if (first_rowgroup == nullptr) {
		return plan;
	}
	try {
		plan.columns     = detail::build_zero_copy_column_plan(*first_rowgroup, m_load_column_names);
		plan.build_order = detail::build_zero_copy_column_order(plan.columns);
	} catch (const std::exception&) {
		plan.columns.clear();
		plan.build_order.clear();
		return plan;
	}
	plan.enabled = true;
	for (size_t i = 1; i < rowgroups->size(); ++i) {
		const auto* rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(i));
		if (rowgroup == nullptr || !detail::rowgroup_matches_zero_copy_plan(*rowgroup, plan.columns)) {
			plan.enabled = false;
			plan.columns.clear();
			plan.build_order.clear();
			break;
		}
	}
	return plan;
}

} // namespace galp::format
