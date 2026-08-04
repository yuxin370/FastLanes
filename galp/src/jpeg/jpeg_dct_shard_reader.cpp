#include "jpeg/jpeg_dct_shard_reader.hpp"
#include "format/compact_descriptor_v3.hpp"
#include "fls/connection.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include "fls/reader/table_reader.hpp"
#include "fls/table/rowgroup.hpp"
#include "flatbuffers/flatbuffer_builder.h"
#include <algorithm>
#include <array>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace galp::jpeg::detail {
namespace {

template <typename ColT>
const ColT* typed_column_ptr(const fastlanes::col_pt& column) {
	const auto* holder = std::get_if<fastlanes::up<ColT>>(&column);
	return holder == nullptr || !*holder ? nullptr : holder->get();
}

template <typename T>
int16_t checked_dct_value(const T value) {
	return static_cast<int16_t>(value);
}

int16_t coefficient_value(const fastlanes::col_pt& column, const size_t row_index) {
	if (const auto* col = typed_column_ptr<fastlanes::col_i08>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	if (const auto* col = typed_column_ptr<fastlanes::col_i16>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	if (const auto* col = typed_column_ptr<fastlanes::col_i32>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	if (const auto* col = typed_column_ptr<fastlanes::col_i64>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u08_col_t>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u16_col_t>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u32_col_t>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u64_col_t>(column)) {
		return checked_dct_value(col->data.at(row_index));
	}
	throw std::runtime_error("JPEG DCT FLS column materialized to an unsupported type");
}

template <typename ColT>
bool append_selected_typed_column(const fastlanes::col_pt&        column,
                                  const std::span<const uint32_t> selected_vectors,
                                  const size_t                    destination_base,
                                  std::vector<int16_t>&           destination) {
	const auto* typed = typed_column_ptr<ColT>(column);
	if (typed == nullptr) {
		return false;
	}
	for (size_t selected_index = 0; selected_index < selected_vectors.size(); ++selected_index) {
		const auto source_base = static_cast<size_t>(selected_vectors[selected_index]) * fastlanes::CFG::VEC_SZ;
		const auto destination_vector_base = destination_base + selected_index * fastlanes::CFG::VEC_SZ;
		if (source_base > typed->data.size() || fastlanes::CFG::VEC_SZ > typed->data.size() - source_base) {
			throw std::runtime_error("JPEG DCT selected vector exceeds its materialized column");
		}
		std::transform(typed->data.begin() + static_cast<std::ptrdiff_t>(source_base),
		               typed->data.begin() + static_cast<std::ptrdiff_t>(source_base + fastlanes::CFG::VEC_SZ),
		               destination.begin() + static_cast<std::ptrdiff_t>(destination_vector_base),
		               [](const auto value) { return static_cast<int16_t>(value); });
	}
	return true;
}

void append_selected_column(const fastlanes::col_pt&        column,
                            const std::span<const uint32_t> selected_vectors,
                            const size_t                    destination_base,
                            std::vector<int16_t>&           destination) {
	if (append_selected_typed_column<fastlanes::col_i08>(column, selected_vectors, destination_base, destination) ||
	    append_selected_typed_column<fastlanes::col_i16>(column, selected_vectors, destination_base, destination) ||
	    append_selected_typed_column<fastlanes::col_i32>(column, selected_vectors, destination_base, destination) ||
	    append_selected_typed_column<fastlanes::col_i64>(column, selected_vectors, destination_base, destination) ||
	    append_selected_typed_column<fastlanes::u08_col_t>(column, selected_vectors, destination_base, destination) ||
	    append_selected_typed_column<fastlanes::u16_col_t>(column, selected_vectors, destination_base, destination) ||
	    append_selected_typed_column<fastlanes::u32_col_t>(column, selected_vectors, destination_base, destination) ||
	    append_selected_typed_column<fastlanes::u64_col_t>(column, selected_vectors, destination_base, destination)) {
		return;
	}
	throw std::runtime_error("JPEG DCT FLS column materialized to an unsupported type");
}

fastlanes::up<fastlanes::Rowgroup>
materialize_compact_rowgroup(const std::filesystem::path&              fls_path,
	                          const galp::format::CompactDescriptorV3& descriptor,
	                          const uint32_t                           rowgroup_index) {
	auto native = descriptor.unpack_rowgroup(rowgroup_index);
	flatbuffers::FlatBufferBuilder builder;
	const auto root = fastlanes::RowgroupDescriptor::Pack(builder, native.get());
	fastlanes::FinishRowgroupDescriptorBuffer(builder, root);
	const auto* rowgroup_descriptor = fastlanes::GetRowgroupDescriptor(builder.GetBufferPointer());
	if (rowgroup_descriptor == nullptr) {
		throw std::runtime_error("failed to reconstruct compact JPEG DCT rowgroup descriptor");
	}
	fastlanes::Connection     connection;
	fastlanes::RowgroupReader reader(fls_path, *rowgroup_descriptor, connection);
	return reader.materialize();
}

std::string normalized_path_key(const std::filesystem::path& path) {
	return std::filesystem::absolute(path).lexically_normal().string();
}

} // namespace

struct JpegDctShardCpuReader::Impl {
	struct CachedCompactDescriptor {
		std::shared_ptr<galp::format::CompactDescriptorV3> descriptor;
		std::list<std::string>::iterator                   lru_position;
	};

	std::shared_ptr<galp::format::CompactDescriptorV3>
	compact_descriptor(const std::filesystem::path& fls_path) {
		const auto key = normalized_path_key(fls_path);
		if (const auto found = compact_paths.find(key); found != compact_paths.end()) {
			compact_lru.splice(compact_lru.begin(), compact_lru, found->second.lru_position);
			return found->second.descriptor;
		}
		auto descriptor = std::make_shared<galp::format::CompactDescriptorV3>(
		    galp::format::CompactDescriptorV3::Open(fls_path));
		constexpr size_t kCompactDescriptorCacheCapacity = 8U;
		if (compact_paths.size() >= kCompactDescriptorCacheCapacity) {
			compact_paths.erase(compact_lru.back());
			compact_lru.pop_back();
		}
		compact_lru.push_front(key);
		compact_paths.emplace(key, CachedCompactDescriptor {descriptor, compact_lru.begin()});
		return descriptor;
	}

	mutable std::mutex                                                                      descriptor_mutex;
	mutable std::unordered_map<uint32_t, std::shared_ptr<fastlanes::TableDescriptorHandle>> descriptors;
	std::list<std::string>                                                                  compact_lru;
	std::unordered_map<std::string, CachedCompactDescriptor>                                compact_paths;
};

JpegDctShardCpuReader::JpegDctShardCpuReader()
    : impl_(std::make_unique<Impl>()) {
}

JpegDctShardCpuReader::~JpegDctShardCpuReader() = default;

uint64_t JpegDctShardCpuReader::RowgroupStorageBytes(const uint32_t               shard_id,
                                                     const std::filesystem::path& fls_path,
                                                     const std::vector<uint32_t>& rowgroup_indices) const {
	std::lock_guard<std::mutex> guard(impl_->descriptor_mutex);
	if (galp::format::is_compact_v3_fls(fls_path)) {
		const auto descriptor = impl_->compact_descriptor(fls_path);
		uint64_t bytes = 0U;
		for (const auto rowgroup_index : rowgroup_indices) {
			const auto rowgroup_bytes = descriptor->rowgroup(rowgroup_index).payload_size;
			if (rowgroup_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
				throw std::runtime_error("JPEG DCT rowgroup storage-byte audit overflow");
			}
			bytes += rowgroup_bytes;
		}
		return bytes;
	}
	auto&                       descriptor = impl_->descriptors[shard_id];
	if (!descriptor) {
		fastlanes::File       file(fls_path);
		fastlanes::FileHeader header {};
		fastlanes::FileFooter footer {};
		fastlanes::FileHeader::Load(header, file);
		fastlanes::FileFooter::Load(footer, file);
		if (header.settings.inline_footer) {
			descriptor =
			    std::make_shared<fastlanes::TableDescriptorHandle>(fastlanes::TableDescriptorHandle::FromFileSlice(
			        file, footer.table_descriptor_offset, footer.table_descriptor_size, true));
		} else {
			descriptor = std::make_shared<fastlanes::TableDescriptorHandle>(
			    fastlanes::TableDescriptorHandle::FromFile(fls_path.parent_path() / "table_descriptor.fbb", true));
		}
	}
	const auto* table = descriptor->Get();
	if (table == nullptr || table->m_rowgroup_descriptors() == nullptr) {
		throw std::runtime_error("JPEG DCT storage audit could not load rowgroup descriptors");
	}
	uint64_t bytes = 0U;
	for (const auto rowgroup_index : rowgroup_indices) {
		if (rowgroup_index >= table->m_rowgroup_descriptors()->size()) {
			throw std::out_of_range("JPEG DCT storage audit rowgroup index is out of range");
		}
		const auto* rowgroup       = table->m_rowgroup_descriptors()->Get(rowgroup_index);
		const auto  rowgroup_bytes = static_cast<uint64_t>(rowgroup->m_size());
		if (rowgroup_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
			throw std::runtime_error("JPEG DCT rowgroup storage-byte audit overflow");
		}
		bytes += rowgroup_bytes;
	}
	return bytes;
}

JpegDctBlockGroup JpegDctShardCpuReader::ReadBlockGroup(const std::filesystem::path&  fls_path,
                                                        const JpegDctBlockGroupIndex& group) const {
	fastlanes::up<fastlanes::Rowgroup> rowgroup;
	if (galp::format::is_compact_v3_fls(fls_path)) {
		std::shared_ptr<galp::format::CompactDescriptorV3> descriptor;
		{
			std::lock_guard<std::mutex> guard(impl_->descriptor_mutex);
			descriptor = impl_->compact_descriptor(fls_path);
		}
		rowgroup = materialize_compact_rowgroup(fls_path, *descriptor, group.fls_rowgroup_index);
	} else {
		fastlanes::Connection connection;
		auto table_reader = connection.read_fls(fls_path);
		auto rowgroup_reader = table_reader->get_rowgroup_reader(group.fls_rowgroup_index);
		rowgroup = rowgroup_reader->materialize();
	}
	if (rowgroup->internal_rowgroup.size() < 64) {
		throw std::runtime_error("JPEG DCT FLS rowgroup has fewer than 64 coefficient columns");
	}

	JpegDctBlockGroup result;
	result.index = group;
	result.rows.reserve(group.row_count);
	for (uint32_t row_offset = 0; row_offset < group.row_count; ++row_offset) {
		const auto            materialized_row = static_cast<size_t>(group.row_start_in_rowgroup) + row_offset;
		JpegDctCoefficientRow row {};
		for (size_t column = 0; column < row.size(); ++column) {
			row[column] = coefficient_value(rowgroup->internal_rowgroup[column], materialized_row);
		}
		result.rows.push_back(row);
	}
	return result;
}

MaterializedJpegDctImage
JpegDctShardCpuReader::MaterializeImage(const std::filesystem::path&                   fls_path,
                                        const uint32_t                                 global_image_index,
                                        const std::vector<JpegDctMaterializeBlockRef>& blocks) const {
	const bool compact = galp::format::is_compact_v3_fls(fls_path);
	fastlanes::Connection connection;
	auto table_reader = compact ? fastlanes::up<fastlanes::TableReader> {} : connection.read_fls(fls_path);
	std::shared_ptr<galp::format::CompactDescriptorV3> compact_descriptor;
	if (compact) {
		std::lock_guard<std::mutex> guard(impl_->descriptor_mutex);
		compact_descriptor = impl_->compact_descriptor(fls_path);
	}
	std::unordered_map<uint32_t, fastlanes::up<fastlanes::Rowgroup>> rowgroups;
	const auto materialized_rowgroup = [&](const uint32_t rowgroup_index) -> fastlanes::Rowgroup& {
		auto found = rowgroups.find(rowgroup_index);
		if (found == rowgroups.end()) {
			if (compact) {
				found = rowgroups.emplace(
				    rowgroup_index,
				    materialize_compact_rowgroup(fls_path, *compact_descriptor, rowgroup_index)).first;
			} else {
				auto reader = table_reader->get_rowgroup_reader(rowgroup_index);
				found       = rowgroups.emplace(rowgroup_index, reader->materialize()).first;
			}
		}
		if (found->second->internal_rowgroup.size() < 64) {
			throw std::runtime_error("JPEG DCT FLS rowgroup has fewer than 64 coefficient columns");
		}
		return *found->second;
	};

	MaterializedJpegDctImage image;
	image.global_image_index = global_image_index;
	image.blocks.reserve(blocks.size());
	for (const auto& block_ref : blocks) {
		if (!block_ref.row.present) {
			throw std::runtime_error("JPEG DCT image materialization lost a present block");
		}
		auto&      rowgroup = materialized_rowgroup(block_ref.row.fls_rowgroup_index);
		const auto row_index =
		    static_cast<size_t>(block_ref.row.row_start_in_rowgroup) + block_ref.row.row_offset_in_block_group;
		MaterializedJpegDctBlock block;
		block.semantic_slot_id = block_ref.semantic_slot_id;
		block.block_x          = block_ref.block_x;
		block.block_y          = block_ref.block_y;
		for (size_t column = 0; column < block.coefficients.size(); ++column) {
			block.coefficients[column] = coefficient_value(rowgroup.internal_rowgroup[column], row_index);
		}
		image.blocks.push_back(block);
	}
	return image;
}

struct JpegDctSelectedVectorProfileReader::Impl {
	fastlanes::Connection                 connection;
	fastlanes::up<fastlanes::TableReader> table_reader;
	std::filesystem::path                 fls_path;
	std::shared_ptr<galp::format::CompactDescriptorV3> compact_descriptor;

	explicit Impl(const std::filesystem::path& fls_path)
	    : fls_path(fls_path) {
		if (galp::format::is_compact_v3_fls(fls_path)) {
			compact_descriptor = std::make_shared<galp::format::CompactDescriptorV3>(
			    galp::format::CompactDescriptorV3::Open(fls_path));
		} else {
			table_reader = connection.read_fls(fls_path);
		}
	}
};

JpegDctSelectedVectorProfileReader::JpegDctSelectedVectorProfileReader(const std::filesystem::path& fls_path)
    : impl_(std::make_unique<Impl>(fls_path)) {
}

JpegDctSelectedVectorProfileReader::~JpegDctSelectedVectorProfileReader() = default;

void JpegDctSelectedVectorProfileReader::AppendRowgroupSelectedVectors(
    const uint32_t                        rowgroup_index,
    const std::span<const uint32_t>       selected_vectors,
    std::array<std::vector<int16_t>, 64>& columns) const {
	if (selected_vectors.empty()) {
		throw std::invalid_argument("JPEG DCT crop profile selected-vector list is empty");
	}
	if (!std::is_sorted(selected_vectors.begin(), selected_vectors.end()) ||
	    std::adjacent_find(selected_vectors.begin(), selected_vectors.end()) != selected_vectors.end()) {
		throw std::invalid_argument("JPEG DCT crop profile selected vectors must be strictly increasing");
	}
	fastlanes::up<fastlanes::Rowgroup> rowgroup;
	if (impl_->compact_descriptor) {
		if (selected_vectors.size() != 1U || selected_vectors.front() != 0U) {
			throw std::out_of_range("Compact v3 rowgroup contains exactly one vector");
		}
		rowgroup = materialize_compact_rowgroup(
		    impl_->fls_path, *impl_->compact_descriptor, rowgroup_index);
	} else {
		auto rowgroup_reader = impl_->table_reader->get_rowgroup_reader(rowgroup_index);
		rowgroup = rowgroup_reader->materialize();
	}
	if (rowgroup->internal_rowgroup.size() < columns.size()) {
		throw std::runtime_error("JPEG DCT FLS rowgroup has fewer than 64 coefficient columns");
	}
	if (selected_vectors.back() >= rowgroup->m_descriptor.m_n_vec) {
		throw std::out_of_range("JPEG DCT crop profile selected vector exceeds its source rowgroup");
	}
	const size_t destination_base = columns.front().size();
	if (!std::all_of(
	        columns.begin(), columns.end(), [&](const auto& column) { return column.size() == destination_base; })) {
		throw std::runtime_error("JPEG DCT crop profile columns have inconsistent lengths");
	}
	if (selected_vectors.size() > (std::numeric_limits<size_t>::max() - destination_base) / fastlanes::CFG::VEC_SZ) {
		throw std::overflow_error("JPEG DCT crop profile selected-vector output size overflow");
	}
	const size_t destination_size = destination_base + selected_vectors.size() * fastlanes::CFG::VEC_SZ;
	for (auto& column : columns) {
		column.resize(destination_size);
	}
	for (size_t column = 0; column < columns.size(); ++column) {
		append_selected_column(
		    rowgroup->internal_rowgroup[column], selected_vectors, destination_base, columns[column]);
	}
}

} // namespace galp::jpeg::detail
