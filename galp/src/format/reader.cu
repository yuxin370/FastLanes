// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/format/reader.cu
// ────────────────────────────────────────────────────────
#include "engine/materialization/zero_copy_materializer.cuh"
#include "core/operator_capabilities.hpp"
#include "format/reader.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/segment_descriptor.hpp"
#include "fls/io/file.hpp"
#include "galp/errors.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <flatbuffers/base.h>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

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
	std::vector<size_t>                       vector_storage_bytes;
	std::vector<std::byte>                    static_prefix;
};

struct SparseDatasetAccessIndex {
	std::vector<SparseRowgroupAccessIndex> rowgroups;
};

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

std::shared_ptr<const SparseDatasetAccessIndex>
build_sparse_dataset_access_index(fastlanes::File& file, const fastlanes::TableDescriptor& table_descriptor) {
	auto index = std::make_shared<SparseDatasetAccessIndex>();
	const auto* rowgroups = table_descriptor.m_rowgroup_descriptors();
	if (rowgroups == nullptr) {
		return index;
	}
	index->rowgroups.resize(rowgroups->size());
	for (flatbuffers::uoffset_t rowgroup_index = 0; rowgroup_index < rowgroups->size(); ++rowgroup_index) {
		const auto* rowgroup = rowgroups->Get(rowgroup_index);
		auto&       entry    = index->rowgroups[rowgroup_index];
		if (rowgroup == nullptr) {
			entry.fallback_reason = "null-rowgroup-descriptor";
			continue;
		}
		const auto segments = rowgroup_segment_descriptors(*rowgroup);
		if (!validate_sparse_vector_segments(*rowgroup, segments, &entry.fallback_reason)) {
			continue;
		}
		entry.index_ranges = segment_index_ranges(segments);
		const auto rowgroup_bytes = static_cast<size_t>(rowgroup->m_size());
		std::vector<std::byte> index_backing(rowgroup_bytes, std::byte {0});
		for (const auto& range : entry.index_ranges) {
			file.ReadRangeUnchecked(
			    index_backing.data() + range.offset, rowgroup->m_offset() + range.offset, range.size);
		}
		entry.shared_ranges = segment_shared_ranges(segments, index_backing.data());
		for (const auto& range : entry.shared_ranges) {
			file.ReadRangeUnchecked(
			    index_backing.data() + range.offset, rowgroup->m_offset() + range.offset, range.size);
		}
		entry.static_prefix = pack_ranges(index_backing.data(), entry.index_ranges);
		const auto shared_prefix = pack_ranges(index_backing.data(), entry.shared_ranges);
		entry.static_prefix.insert(entry.static_prefix.end(), shared_prefix.begin(), shared_prefix.end());
		entry.vector_ranges.resize(rowgroup->m_n_vec());
		entry.vector_storage_bytes.assign(rowgroup->m_n_vec(), 0U);
		for (const auto* segment : segments) {
			if (segment_entrypoint_count(*segment) == 1U) {
				continue;
			}
			for (uint32_t vector = 0; vector < rowgroup->m_n_vec(); ++vector) {
				const auto range = segment_vector_range(*segment, index_backing.data(), vector);
				entry.vector_ranges[vector].push_back(range);
				if (range.size > std::numeric_limits<size_t>::max() - entry.vector_storage_bytes[vector]) {
					throw std::runtime_error("sparse rowgroup vector byte count overflow");
				}
				entry.vector_storage_bytes[vector] += range.size;
			}
		}
		entry.supported = true;
		entry.fallback_reason.clear();
	}
	return index;
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

} // namespace galp::format::detail

namespace galp::format {

struct SparseVectorReadPlan::Impl {
	enum class Strategy {
		kFullRowgroup,
		kSourceRanges,
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
	size_t                         rowgroup_index       = 0U;
	size_t                         rowgroup_bytes       = 0U;
	size_t                         full_vector_count    = 0U;
	size_t                         selected_vector_count = 0U;
	size_t                         storage_bytes        = 0U;
	Strategy                       strategy             = Strategy::kFullRowgroup;
	std::string                    fallback_reason;
	std::vector<detail::SparseByteRange> source_ranges;
	std::vector<BundleRun>               bundle_runs;
	uint64_t                              envelope_file_offset = 0U;
	size_t                                envelope_size        = 0U;
	std::vector<EnvelopeCopy>             envelope_copies;
	size_t                                packed_bytes = 0U;
	std::vector<galp::execution::PackedRowgroupScatterRange> packed_scatter_ranges;
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

size_t SparseVectorReadPlan::estimated_pread_count() const noexcept {
	if (!impl_) {
		return 0U;
	}
	switch (impl_->strategy) {
	case Impl::Strategy::kFullRowgroup:
	case Impl::Strategy::kBundleEnvelope:
		return 1U;
	case Impl::Strategy::kSourceRanges:
		return impl_->source_ranges.size();
	case Impl::Strategy::kBundleRuns:
	case Impl::Strategy::kBundlePacked:
		// Each contiguous bundle run is submitted independently.  A short
		// preadv may add another syscall, so this is a lower-bound estimate.
		return impl_->bundle_runs.size();
	}
	return 0U;
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

FlsReader::FlsReader(const std::filesystem::path& file_path,
                     const bool                   load_column_names,
                     const bool                   enable_sparse_vector_reads)
	: FlsReader(file_path,
	            FlsReaderOptions {.load_column_names         = load_column_names,
	                              .enable_sparse_vector_reads = enable_sparse_vector_reads}) {
}

FlsReader::FlsReader(const std::filesystem::path& file_path, const FlsReaderOptions& options)
    : m_file(std::make_shared<fastlanes::File>(file_path))
    , m_table_descriptor(
	      std::make_shared<fastlanes::TableDescriptorHandle>(
	          detail::load_table_descriptor(*m_file, file_path)))
    , m_load_column_names(options.load_column_names) {
	if (options.enable_sparse_vector_reads) {
		m_sparse_vector_bundle = detail::load_sparse_vector_bundle_index(
		    sparse_vector_bundle_path(file_path), *table_descriptor(), m_file->Size());
		m_sparse_access_index = detail::build_sparse_dataset_access_index(*m_file, *table_descriptor());
	}
	if (options.build_shared_zero_copy_schema_plan) {
		m_zero_copy_schema_plan = std::make_shared<ZeroCopySchemaPlan>(build_shared_zero_copy_schema_plan());
	}
}

const fastlanes::TableDescriptor* FlsReader::table_descriptor() const {
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	return td;
}

size_t FlsReader::rowgroup_count() const {
	const auto* td = table_descriptor();
	return static_cast<size_t>(td->m_rowgroup_descriptors()->size());
}

size_t FlsReader::rowgroup_storage_bytes(const size_t rowgroup_idx) const {
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

bool FlsReader::sparse_vector_read_supported(const size_t rowgroup_idx, std::string* const reason) const {
	if (!m_sparse_access_index) {
		if (reason != nullptr) {
			*reason = "sparse-access-index-disabled";
		}
		return false;
	}
	if (rowgroup_idx >= m_sparse_access_index->rowgroups.size()) {
		throw std::out_of_range("rowgroup_idx out of range");
	}
	const auto& entry = m_sparse_access_index->rowgroups[rowgroup_idx];
	if (reason != nullptr) {
		*reason = entry.fallback_reason;
	}
	return entry.supported;
}

SparseVectorReadPlan FlsReader::compile_sparse_vector_read_plan(
	const size_t rowgroup_idx, const std::vector<uint32_t>& selected_vectors, const bool packed_device_scatter) const {
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
	const size_t rowgroup_bytes = static_cast<size_t>(rowgroup->m_size());
	const size_t vector_count   = static_cast<size_t>(rowgroup->m_n_vec());
	std::vector<uint32_t> vectors(selected_vectors);
	std::sort(vectors.begin(), vectors.end());
	vectors.erase(std::unique(vectors.begin(), vectors.end()), vectors.end());
	if (vectors.back() >= vector_count) {
		throw std::out_of_range("selected vector exceeds rowgroup vector count");
	}
	if (!m_sparse_access_index || rowgroup_idx >= m_sparse_access_index->rowgroups.size()) {
		throw std::runtime_error("sparse dataset access index is unavailable");
	}
	const auto& access = m_sparse_access_index->rowgroups[rowgroup_idx];
	auto plan = std::make_shared<SparseVectorReadPlan::Impl>();
	plan->owner                 = m_sparse_plan_owner;
	plan->rowgroup_index        = rowgroup_idx;
	plan->rowgroup_bytes        = rowgroup_bytes;
	plan->full_vector_count     = vector_count;
	plan->selected_vector_count = vectors.size();
	if (!access.supported || vectors.size() >= vector_count) {
		plan->fallback_reason = access.supported ? "selected-vectors-cover-full-rowgroup" : access.fallback_reason;
		plan->storage_bytes   = rowgroup_bytes;
		return SparseVectorReadPlan(std::move(plan));
	}

	if (!m_sparse_vector_bundle) {
		plan->strategy = SparseVectorReadPlan::Impl::Strategy::kSourceRanges;
		std::vector<detail::SparseByteRange> ranges;
		for (const auto vector : vectors) {
			const auto& vector_ranges = access.vector_ranges.at(vector);
			ranges.insert(ranges.end(), vector_ranges.begin(), vector_ranges.end());
		}
		plan->source_ranges = detail::coalesce_ranges(std::move(ranges));
		for (const auto& range : plan->source_ranges) {
			plan->storage_bytes += range.size;
		}
		return SparseVectorReadPlan(std::move(plan));
	}

	if (rowgroup_idx >= m_sparse_vector_bundle->rowgroups.size()) {
		throw std::runtime_error("sparse vector bundle rowgroup index is out of range");
	}
	const auto& bundle = m_sparse_vector_bundle->rowgroups[rowgroup_idx];
	if (packed_device_scatter) {
		plan->strategy     = SparseVectorReadPlan::Impl::Strategy::kBundlePacked;
		plan->packed_bytes = access.static_prefix.size();
		size_t prefix_cursor = 0U;
		for (const auto& range : access.index_ranges) {
			plan->packed_scatter_ranges.push_back(
			    {prefix_cursor, range.offset, range.size});
			prefix_cursor += range.size;
		}
		for (const auto& range : access.shared_ranges) {
			plan->packed_scatter_ranges.push_back(
			    {prefix_cursor, range.offset, range.size});
			prefix_cursor += range.size;
		}
		if (prefix_cursor != access.static_prefix.size()) {
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
			for (const auto& range : access.vector_ranges.at(vector)) {
				plan->envelope_copies.push_back({source_cursor, range.offset, range.size});
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
			const auto& ranges = access.vector_ranges.at(vector);
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
		plan->bundle_runs.push_back(std::move(run));
	}
	return SparseVectorReadPlan(std::move(plan));
}

void FlsReader::read_rowgroup_bytes_into(const size_t        rowgroup_idx,
                                         std::byte* const    backing_data,
                                         const size_t        backing_capacity,
                                         ZeroCopyReadTiming* timing) {
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
	if (backing_data == nullptr || backing_capacity < rg_bytes) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}

	const auto pread_start = std::chrono::steady_clock::now();
	m_file->ReadRangeUnchecked(backing_data, rg->m_offset(), rg->m_size());
	const auto pread_end = std::chrono::steady_clock::now();
	if (timing != nullptr) {
		timing->storage_bytes = rg_bytes;
		timing->full_storage_bytes = rg_bytes;
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

	if (!m_sparse_access_index || rowgroup_idx >= m_sparse_access_index->rowgroups.size()) {
		throw std::runtime_error("sparse dataset access index is unavailable");
	}
	const auto& access_index = m_sparse_access_index->rowgroups[rowgroup_idx];
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

ZeroCopyRowgroup FlsReader::make_zero_copy_rowgroup_from_backing(const size_t          rowgroup_idx,
                                                                 std::shared_ptr<void> backing_owner,
                                                                 std::byte* const      backing_data,
                                                                 const size_t          backing_capacity,
                                                                 const bool            backing_is_pinned,
                                                                 ZeroCopyReadTiming*   timing) {
	const auto  setup_start = std::chrono::steady_clock::now();
	const auto* td          = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto n_rgs = td->m_rowgroup_descriptors()->size();
	if (rowgroup_idx >= n_rgs) {
		throw std::out_of_range("rowgroup_idx out of range");
	}

	const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	const size_t rg_bytes = static_cast<size_t>(rg->m_size());
	if (backing_data == nullptr || backing_capacity < rg_bytes) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}

	const size_t n_vecs   = static_cast<size_t>(rg->m_n_vec());
	const size_t n_values = n_vecs * galp::codec::consts::VALUES_PER_VECTOR;
	const size_t n_tuples = static_cast<size_t>(rg->m_n_tuples());

	auto        backing_span    = fastlanes::span<std::byte> {backing_data, rg_bytes};
	const auto& col_descs       = *rg->m_column_descriptors();
	const bool  use_schema_plan = m_zero_copy_schema_plan && m_zero_copy_schema_plan->enabled &&
	                             m_zero_copy_schema_plan->columns.size() == col_descs.size();

	std::shared_ptr<fastlanes::RowgroupView> view;
	if (!use_schema_plan) {
		view = std::make_shared<fastlanes::RowgroupView>(backing_span, *rg);
	}

	ZeroCopyRowgroup out {};
	out.rowgroup_index         = rowgroup_idx;
	out.n_values               = n_values;
	out.n_vecs                 = n_vecs;
	out.n_tuples               = n_tuples;
	out.table_descriptor_owner = m_table_descriptor;
	out.rowgroup_descriptor    = rg;
	out.backing_owner          = std::move(backing_owner);
	out.backing_span           = backing_span;
	out.backing_is_pinned      = backing_is_pinned;
	out.rowgroup_view          = view;
	if (timing != nullptr) {
		timing->used_pinned_backing = backing_is_pinned;
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

	out.columns.reserve(col_descs.size());
	for (size_t col_idx = 0; col_idx < col_descs.size(); ++col_idx) {
		const auto& col_desc = *col_descs.Get(static_cast<flatbuffers::uoffset_t>(col_idx));
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

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy(const size_t rowgroup_idx, ZeroCopyReadTiming* timing) {
	auto backing = std::make_shared<fastlanes::Buf>(rowgroup_storage_bytes(rowgroup_idx));
	return read_rowgroup_zero_copy_into(rowgroup_idx,
	                                    std::static_pointer_cast<void>(backing),
	                                    reinterpret_cast<std::byte*>(backing->mutable_data()),
	                                    backing->Capacity(),
	                                    /*backing_is_pinned=*/false,
	                                    timing);
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_selected_vectors(
	const size_t rowgroup_idx, const std::vector<uint32_t>& selected_vectors, ZeroCopyReadTiming* timing) {
	auto backing = std::make_shared<fastlanes::Buf>(rowgroup_storage_bytes(rowgroup_idx));
	return read_rowgroup_zero_copy_selected_vectors_into(rowgroup_idx,
	                                                     selected_vectors,
	                                                     std::static_pointer_cast<void>(backing),
	                                                     reinterpret_cast<std::byte*>(backing->mutable_data()),
	                                                     backing->Capacity(),
	                                                     /*backing_is_pinned=*/false,
	                                                     timing);
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_selected_vectors_packed(
	const size_t rowgroup_idx, const std::vector<uint32_t>& selected_vectors, ZeroCopyReadTiming* timing) {
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
	const size_t rowgroup_bytes = static_cast<size_t>(rowgroup->m_size());
	const size_t vector_count   = static_cast<size_t>(rowgroup->m_n_vec());
	std::vector<uint32_t> unique_vectors = selected_vectors;
	std::sort(unique_vectors.begin(), unique_vectors.end());
	unique_vectors.erase(std::unique(unique_vectors.begin(), unique_vectors.end()), unique_vectors.end());
	if (unique_vectors.back() >= vector_count) {
		throw std::out_of_range("selected vector exceeds rowgroup vector count");
	}

	const auto segments = detail::rowgroup_segment_descriptors(*rowgroup);
	std::string capability_reason;
	const bool supported = detail::validate_sparse_vector_segments(*rowgroup, segments, &capability_reason);
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
		return read_rowgroup_zero_copy(plan.rowgroup_index, timing);
	}
	if (!m_sparse_access_index || plan.rowgroup_index >= m_sparse_access_index->rowgroups.size()) {
		throw std::runtime_error("compiled sparse dataset access index is unavailable");
	}
	const auto& access = m_sparse_access_index->rowgroups[plan.rowgroup_index];
	const auto* rowgroup = table_descriptor()->m_rowgroup_descriptors()->Get(
	    static_cast<flatbuffers::uoffset_t>(plan.rowgroup_index));
	if (rowgroup == nullptr || static_cast<size_t>(rowgroup->m_size()) != plan.rowgroup_bytes) {
		throw std::runtime_error("compiled sparse vector read plan no longer matches its rowgroup");
	}

	auto logical = std::make_shared<fastlanes::Buf>(plan.rowgroup_bytes);
	auto* const logical_data = reinterpret_cast<std::byte*>(logical->mutable_data());
	std::memset(logical_data, 0, plan.rowgroup_bytes);
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
	if (plan.strategy == SparseVectorReadPlan::Impl::Strategy::kSourceRanges) {
		for (const auto& range : plan.source_ranges) {
			const auto start = std::chrono::steady_clock::now();
			m_file->ReadRangeUnchecked(
			    logical_data + range.offset, rowgroup->m_offset() + range.offset, range.size);
			const auto end = std::chrono::steady_clock::now();
			record_read(start, end, range.size, 1U);
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
		timing->sparse_read_supported   = true;
		timing->used_sparse_read        = true;
		timing->used_vector_bundle_read = plan.strategy != SparseVectorReadPlan::Impl::Strategy::kSourceRanges;
		timing->used_vector_bundle_envelope_read =
		    plan.strategy == SparseVectorReadPlan::Impl::Strategy::kBundleEnvelope;
	}
	auto zero_copy = make_zero_copy_rowgroup_from_backing(plan.rowgroup_index,
	                                                      std::static_pointer_cast<void>(logical),
	                                                      logical_data,
	                                                      logical->Capacity(),
	                                                      /*backing_is_pinned=*/false,
	                                                      timing);
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
