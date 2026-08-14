#include "jpeg_dct_exact_verifier.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct_storage.hpp"
#include "jpeg_dct_metadata.hpp"
#include "jpeg_dct_order.hpp"
#include "jpeg_dct_shard_reader.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace galp::jpeg {

namespace {

using BlockKey = std::tuple<uint32_t, uint32_t, uint32_t>;

struct ShardTask {
	JpegDctShardManifestEntry entry;
};

auto mismatch_order_key(const JpegDctExactMismatch& mismatch) {
	return std::tuple {mismatch.global_image_index,
	                   mismatch.semantic_slot_id,
	                   mismatch.block_y,
	                   mismatch.block_x,
	                   mismatch.coefficient,
	                   static_cast<uint8_t>(mismatch.kind)};
}

void consider_mismatch(JpegDctExactVerificationResult& result, const JpegDctExactMismatch& candidate) {
	if (!candidate.present) {
		return;
	}
	if (!result.first_mismatch.present || mismatch_order_key(candidate) < mismatch_order_key(result.first_mismatch)) {
		result.first_mismatch = candidate;
	}
}

JpegDctExactMismatch make_mismatch(const uint32_t                 image_index,
                                   const BlockKey&                key,
                                   const uint32_t                 coefficient,
                                   const int16_t                  expected,
                                   const int16_t                  actual,
                                   const JpegDctExactMismatchKind kind) {
	const auto [semantic_slot_id, block_y, block_x] = key;
	return {true, image_index, semantic_slot_id, block_y, block_x, coefficient, expected, actual, kind};
}

constexpr uint64_t kMissingPhysicalRowKey = std::numeric_limits<uint64_t>::max();
constexpr size_t   kMissingMetadataGroupIndex = std::numeric_limits<size_t>::max();
constexpr size_t   kMissingSourceRow = std::numeric_limits<size_t>::max();

struct CachedSourceGroupMapping {
	uint64_t key                  = 0U;
	size_t   metadata_group_index = kMissingMetadataGroupIndex;
};

struct CachedImageMajorSourceMapping {
	std::vector<uint64_t> source_group_keys;
	std::vector<uint64_t> source_group_rows;
	std::vector<size_t>   source_rows_in_physical_order;
	std::vector<BlockKey> physical_keys;
};

uint64_t pack_physical_row_key(const uint32_t image_index, const BlockKey& key) {
	const auto [semantic_slot_id, block_y, block_x] = key;
	if (semantic_slot_id > 0xFFU || block_y > 0xFFFU || block_x > 0xFFFU) {
		throw std::runtime_error("JPEG DCT exact-verifier block key exceeds packed bounds");
	}
	return (static_cast<uint64_t>(image_index) << 32U) |
	       (static_cast<uint64_t>(semantic_slot_id) << 24U) |
	       (static_cast<uint64_t>(block_y) << 12U) | static_cast<uint64_t>(block_x);
}

uint64_t pack_group_key(const uint32_t semantic_slot_id, const uint32_t block_y, const uint32_t block_x) {
	if (semantic_slot_id > 0xFFU || block_y > 0xFFFU || block_x > 0xFFFU) {
		throw std::runtime_error("JPEG DCT exact-verifier block-group key exceeds packed bounds");
	}
	return (static_cast<uint64_t>(semantic_slot_id) << 24U) |
	       (static_cast<uint64_t>(block_y) << 12U) | static_cast<uint64_t>(block_x);
}

std::pair<uint32_t, BlockKey> unpack_physical_row_key(const uint64_t packed) {
	const auto image_index = static_cast<uint32_t>(packed >> 32U);
	const auto semantic_slot_id = static_cast<uint32_t>((packed >> 24U) & 0xFFU);
	const auto block_y = static_cast<uint32_t>((packed >> 12U) & 0xFFFU);
	const auto block_x = static_cast<uint32_t>(packed & 0xFFFU);
	return {image_index, BlockKey {semantic_slot_id, block_y, block_x}};
}

bool image_major_mapping_matches(const CachedImageMajorSourceMapping& mapping,
                                 const JpegDctTable&                   source_table) {
	const auto& source_groups = source_table.metadata.block_group_index;
	if (mapping.source_group_keys.size() != source_groups.size() ||
	    mapping.source_group_rows.size() != source_groups.size() ||
	    mapping.source_rows_in_physical_order.size() != source_table.row_count ||
	    mapping.physical_keys.size() != source_table.row_count) {
		return false;
	}
	for (size_t group_index = 0U; group_index < source_groups.size(); ++group_index) {
		const auto& group = source_groups[group_index];
		if (group.row_count != 1U ||
		    mapping.source_group_keys[group_index] !=
		        pack_group_key(group.semantic_slot_id, group.block_y, group.block_x) ||
		    mapping.source_group_rows[group_index] != group.row_start) {
			return false;
		}
	}
	return true;
}

void rebuild_image_major_mapping(CachedImageMajorSourceMapping& mapping,
                                 const JpegDctTable&             source_table,
                                 const JpegImageMetadata&        image_metadata,
                                 const JpegDctDatasetMetadata&   shard_metadata,
                                 const uint32_t                  expected_row_count) {
	struct SlotLayout {
		uint64_t offset = 0U;
		uint32_t width  = 0U;
		uint32_t height = 0U;
	};
	std::unordered_map<uint32_t, SlotLayout> slot_layouts;
	slot_layouts.reserve(shard_metadata.semantic_components.size());
	uint64_t physical_rows = 0U;
	for (const auto& slot : shard_metadata.semantic_components) {
		const auto component = std::find_if(
		    image_metadata.components.begin(), image_metadata.components.end(), [&](const auto& candidate) {
			    return candidate.present && candidate.semantic_slot_id == slot.semantic_slot_id;
		    });
		if (component == image_metadata.components.end()) {
			continue;
		}
		const auto component_rows =
		    static_cast<uint64_t>(component->width_in_blocks) * component->height_in_blocks;
		if (component_rows > std::numeric_limits<uint64_t>::max() - physical_rows) {
			throw std::runtime_error("image-major exact-verifier component-row count overflow");
		}
		if (!slot_layouts.emplace(slot.semantic_slot_id,
		                          SlotLayout {physical_rows,
		                                      component->width_in_blocks,
		                                      component->height_in_blocks})
		         .second) {
			throw std::runtime_error("image-major exact-verifier metadata contains duplicate semantic slots");
		}
		physical_rows += component_rows;
	}
	if (physical_rows != expected_row_count || physical_rows != source_table.row_count) {
		throw std::runtime_error("image-major exact-verifier source geometry does not match the image record");
	}

	mapping = {};
	const auto& source_groups = source_table.metadata.block_group_index;
	mapping.source_group_keys.reserve(source_groups.size());
	mapping.source_group_rows.reserve(source_groups.size());
	mapping.source_rows_in_physical_order.assign(source_table.row_count, kMissingSourceRow);
	mapping.physical_keys.resize(source_table.row_count);
	for (const auto& group : source_groups) {
		if (group.row_count != 1U || group.row_start >= source_table.row_count) {
			throw std::runtime_error("image-major exact-verifier source block group is not a single valid row");
		}
		const auto layout = slot_layouts.find(group.semantic_slot_id);
		if (layout == slot_layouts.end()) {
			throw std::runtime_error("image-major exact-verifier source uses an unknown semantic slot");
		}
		const auto rank = detail::block_order_rank(layout->second.width,
		                                           layout->second.height,
		                                           group.block_x,
		                                           group.block_y,
		                                           shard_metadata.image_major_spatial_order);
		const auto physical_row = layout->second.offset + rank;
		if (physical_row >= mapping.source_rows_in_physical_order.size()) {
			throw std::runtime_error("image-major exact-verifier source block rank is outside its image record");
		}
		const auto row = static_cast<size_t>(physical_row);
		if (mapping.source_rows_in_physical_order[row] != kMissingSourceRow) {
			throw std::runtime_error("image-major exact-verifier source has duplicate physical block ownership");
		}
		const auto key = pack_group_key(group.semantic_slot_id, group.block_y, group.block_x);
		mapping.source_group_keys.push_back(key);
		mapping.source_group_rows.push_back(group.row_start);
		mapping.source_rows_in_physical_order[row] = static_cast<size_t>(group.row_start);
		mapping.physical_keys[row] = BlockKey {group.semantic_slot_id, group.block_y, group.block_x};
	}
	if (std::find(mapping.source_rows_in_physical_order.begin(),
	              mapping.source_rows_in_physical_order.end(),
	              kMissingSourceRow) != mapping.source_rows_in_physical_order.end()) {
		throw std::runtime_error("image-major exact-verifier source does not cover its physical image record");
	}
}

JpegDctExactVerificationResult verify_image_major_shard(
    const std::filesystem::path&              manifest_path,
    const JpegDctShardManifestEntry&          entry,
    const std::vector<std::filesystem::path>& source_paths) {
	const auto metadata_path = manifest_path.parent_path() / entry.metadata_file_name;
	const auto metadata = detail::read_jpeg_dct_metadata_file(metadata_path);
	if (metadata.row_ordering != JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor ||
	    metadata.image_group_index.size() != entry.image_count || metadata.images.size() != entry.image_count) {
		throw std::runtime_error("image-major exact verifier received inconsistent shard metadata");
	}
	const auto fls_path = manifest_path.parent_path() / entry.fls_file_name;
	detail::JpegDctSelectedVectorProfileReader physical_reader(fls_path);
	CachedImageMajorSourceMapping cached_mapping;
	JpegDctExactVerificationResult result;
	uint32_t expected_rowgroup = 0U;
	uint64_t metadata_rows = 0U;

	for (uint32_t local_image = 0U; local_image < entry.image_count; ++local_image) {
		const auto global_image_u64 = entry.first_global_image_index + local_image;
		if (global_image_u64 > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("manifest global image index exceeds the reader index type");
		}
		const auto global_image = static_cast<uint32_t>(global_image_u64);
		const auto& image_group = metadata.image_group_index[local_image];
		if (image_group.local_image_index != local_image || image_group.row_start != metadata_rows ||
		    image_group.row_count == 0U || image_group.fls_rowgroup_index != expected_rowgroup) {
			throw std::runtime_error("image-major exact-verifier image index is not dense and physical-order aligned");
		}
		const auto next_rowgroup = local_image + 1U < entry.image_count
		                               ? metadata.image_group_index[local_image + 1U].fls_rowgroup_index
		                               : entry.rowgroup_count;
		if (next_rowgroup <= expected_rowgroup || next_rowgroup > entry.rowgroup_count) {
			throw std::runtime_error("image-major exact-verifier image has an invalid rowgroup range");
		}

		const auto source_table = read_jpeg_dct_file(source_paths.at(static_cast<size_t>(global_image)));
		if (std::any_of(source_table.columns.begin(), source_table.columns.end(), [&](const auto& column) {
			    return column.size() < source_table.row_count;
		    })) {
			throw std::runtime_error("source JPEG coefficient column is shorter than its row count");
		}
		++result.source_images;
		if (source_table.row_count > std::numeric_limits<uint64_t>::max() - result.expected_blocks ||
		    image_group.row_count > std::numeric_limits<uint64_t>::max() - result.actual_blocks) {
			throw std::runtime_error("image-major exact-verifier block count overflow");
		}
		result.expected_blocks += source_table.row_count;
		result.actual_blocks += image_group.row_count;
		if (!image_major_mapping_matches(cached_mapping, source_table)) {
			rebuild_image_major_mapping(cached_mapping,
			                            source_table,
			                            metadata.images[local_image],
			                            metadata,
			                            image_group.row_count);
		}

		std::array<std::vector<int16_t>, 64> actual_columns;
		for (uint32_t rowgroup = expected_rowgroup; rowgroup < next_rowgroup; ++rowgroup) {
			physical_reader.AppendFullRowgroup(rowgroup, actual_columns);
		}
		if (actual_columns.front().size() < image_group.row_count ||
		    std::any_of(actual_columns.begin(), actual_columns.end(), [&](const auto& column) {
			    return column.size() != actual_columns.front().size();
		    })) {
			throw std::runtime_error("image-major exact-verifier materialized an invalid image rowgroup range");
		}

		const auto shared_rows = std::min<size_t>(source_table.row_count, image_group.row_count);
		if (source_table.row_count > image_group.row_count) {
			result.missing_blocks += source_table.row_count - image_group.row_count;
			consider_mismatch(result,
			                  make_mismatch(global_image,
			                                cached_mapping.physical_keys[image_group.row_count],
			                                0U,
			                                0,
			                                0,
			                                JpegDctExactMismatchKind::kMissingBlock));
		} else if (image_group.row_count > source_table.row_count) {
			result.extra_blocks += image_group.row_count - source_table.row_count;
			consider_mismatch(result,
			                  make_mismatch(global_image,
			                                BlockKey {0U, 0U, 0U},
			                                0U,
			                                0,
			                                0,
			                                JpegDctExactMismatchKind::kExtraBlock));
		}
		for (size_t physical_row = 0U; physical_row < shared_rows; ++physical_row) {
			const auto source_row = cached_mapping.source_rows_in_physical_order[physical_row];
			for (size_t coefficient = 0U; coefficient < actual_columns.size(); ++coefficient) {
				const auto expected = source_table.columns[coefficient][source_row];
				const auto actual = actual_columns[coefficient][physical_row];
				if (expected == actual) {
					continue;
				}
				++result.coefficient_mismatches;
				result.max_abs_difference = std::max(
				    result.max_abs_difference, std::abs(static_cast<int>(expected) - static_cast<int>(actual)));
				consider_mismatch(result,
				                  make_mismatch(global_image,
				                                cached_mapping.physical_keys[physical_row],
				                                static_cast<uint32_t>(coefficient),
				                                expected,
				                                actual,
				                                JpegDctExactMismatchKind::kCoefficient));
			}
		}
		expected_rowgroup = next_rowgroup;
		metadata_rows += image_group.row_count;
	}
	if (expected_rowgroup != entry.rowgroup_count || metadata_rows != entry.real_row_count) {
		throw std::runtime_error("image-major exact-verifier metadata does not cover its physical shard");
	}
	return result;
}

JpegDctExactVerificationResult verify_spatial_major_shard(
    const std::filesystem::path&              manifest_path,
    const JpegDctShardManifestEntry&          entry,
    const std::vector<std::filesystem::path>& source_paths) {
	if (entry.physical_row_count > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error("spatial-major verification shard exceeds addressable memory");
	}
	std::vector<JpegDctCoefficientRow> expected_rows(static_cast<size_t>(entry.physical_row_count));
	std::vector<uint64_t> physical_row_keys(static_cast<size_t>(entry.physical_row_count), kMissingPhysicalRowKey);
	const auto metadata_path = manifest_path.parent_path() / entry.metadata_file_name;
	const auto metadata = detail::read_jpeg_dct_metadata_file(metadata_path);
	if (metadata.row_ordering != JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor) {
		throw std::runtime_error("spatial-major exact verifier received non-spatial shard metadata");
	}
	std::unordered_map<uint64_t, size_t> group_lookup;
	group_lookup.reserve(metadata.block_group_index.size());
	for (size_t group_index = 0U; group_index < metadata.block_group_index.size(); ++group_index) {
		const auto& group = metadata.block_group_index[group_index];
		const auto key = pack_group_key(group.semantic_slot_id, group.block_y, group.block_x);
		if (!group_lookup.emplace(key, group_index).second) {
			throw std::runtime_error("spatial-major verification metadata contains a duplicate block group");
		}
	}
	std::vector<uint32_t> group_ranks(metadata.block_group_index.size(), 0U);
	std::vector<CachedSourceGroupMapping> cached_source_mapping;
	JpegDctExactVerificationResult result;

	for (uint32_t local_image = 0U; local_image < entry.image_count; ++local_image) {
		const auto global_image_u64 = entry.first_global_image_index + local_image;
		if (global_image_u64 > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("manifest global image index exceeds the reader index type");
		}
		const auto global_image = static_cast<uint32_t>(global_image_u64);
		const auto source_table = read_jpeg_dct_file(source_paths.at(static_cast<size_t>(global_image)));
		if (std::any_of(source_table.columns.begin(), source_table.columns.end(), [&](const auto& column) {
			    return column.size() < source_table.row_count;
		    })) {
			throw std::runtime_error("source JPEG coefficient column is shorter than its row count");
		}
		++result.source_images;
		if (source_table.row_count > std::numeric_limits<uint64_t>::max() - result.expected_blocks) {
			throw std::runtime_error("spatial-major verification expected-block count overflow");
		}
		result.expected_blocks += source_table.row_count;

		const auto& source_groups = source_table.metadata.block_group_index;
		bool        mapping_matches = cached_source_mapping.size() == source_groups.size();
		if (mapping_matches) {
			for (size_t source_group_index = 0U; source_group_index < source_groups.size(); ++source_group_index) {
				const auto& source_group = source_groups[source_group_index];
				if (cached_source_mapping[source_group_index].key !=
				    pack_group_key(source_group.semantic_slot_id, source_group.block_y, source_group.block_x)) {
					mapping_matches = false;
					break;
				}
			}
		}
		if (!mapping_matches) {
			cached_source_mapping.clear();
			cached_source_mapping.reserve(source_groups.size());
			for (const auto& source_group : source_groups) {
				const auto key =
				    pack_group_key(source_group.semantic_slot_id, source_group.block_y, source_group.block_x);
				const auto found = group_lookup.find(key);
				cached_source_mapping.push_back(
				    {key, found == group_lookup.end() ? kMissingMetadataGroupIndex : found->second});
			}
		}

		for (size_t source_group_index = 0U; source_group_index < source_groups.size(); ++source_group_index) {
			const auto& source_group = source_groups[source_group_index];
			const BlockKey key {source_group.semantic_slot_id, source_group.block_y, source_group.block_x};
			const auto metadata_group_index = cached_source_mapping[source_group_index].metadata_group_index;
			for (uint32_t row_offset = 0U; row_offset < source_group.row_count; ++row_offset) {
				const auto source_row_u64 = source_group.row_start + row_offset;
				if (source_row_u64 >= source_table.row_count) {
					throw std::runtime_error("source JPEG block-group row is outside the coefficient table");
				}
				if (metadata_group_index == kMissingMetadataGroupIndex) {
					++result.missing_blocks;
					consider_mismatch(
					    result, make_mismatch(global_image, key, 0U, 0, 0, JpegDctExactMismatchKind::kMissingBlock));
					continue;
				}
				const auto& group = metadata.block_group_index[metadata_group_index];
				const auto rank = group_ranks[metadata_group_index]++;
				if (rank >= group.row_count) {
					++result.missing_blocks;
					consider_mismatch(
					    result, make_mismatch(global_image, key, 0U, 0, 0, JpegDctExactMismatchKind::kMissingBlock));
					continue;
				}
				const auto physical_row_u64 = group.row_start + rank;
				if (physical_row_u64 >= entry.physical_row_count) {
					throw std::runtime_error("spatial-major verification row reference is outside its shard");
				}
				const auto row = static_cast<size_t>(physical_row_u64);
				if (physical_row_keys[row] != kMissingPhysicalRowKey) {
					throw std::runtime_error("spatial-major verification found duplicate physical row ownership");
				}
				for (size_t coefficient = 0U; coefficient < expected_rows[row].size(); ++coefficient) {
					expected_rows[row][coefficient] =
					    source_table.columns[coefficient][static_cast<size_t>(source_row_u64)];
				}
				physical_row_keys[row] = pack_physical_row_key(global_image, key);
			}
		}
	}
	for (size_t group_index = 0U; group_index < metadata.block_group_index.size(); ++group_index) {
		const auto expected_rows_in_group = metadata.block_group_index[group_index].row_count;
		if (group_ranks[group_index] < expected_rows_in_group) {
			result.extra_blocks += expected_rows_in_group - group_ranks[group_index];
		}
	}
	result.actual_blocks = entry.real_row_count;
	if (result.extra_blocks != 0U) {
		consider_mismatch(result,
		                  make_mismatch(static_cast<uint32_t>(entry.first_global_image_index),
		                                BlockKey {0U, 0U, 0U},
		                                0U,
		                                0,
		                                0,
		                                JpegDctExactMismatchKind::kExtraBlock));
	}

	const auto fls_path = manifest_path.parent_path() / entry.fls_file_name;
	detail::JpegDctSelectedVectorProfileReader physical_reader(fls_path);
	std::vector<std::vector<size_t>> rowgroup_groups(entry.rowgroup_count);
	for (size_t group_index = 0U; group_index < metadata.block_group_index.size(); ++group_index) {
		const auto rowgroup_index = metadata.block_group_index[group_index].fls_rowgroup_index;
		if (rowgroup_index >= entry.rowgroup_count) {
			throw std::runtime_error("spatial-major verification block group references an invalid rowgroup");
		}
		rowgroup_groups[rowgroup_index].push_back(group_index);
	}
	for (uint32_t rowgroup = 0U; rowgroup < entry.rowgroup_count; ++rowgroup) {
		std::array<std::vector<int16_t>, 64> actual_columns;
		physical_reader.AppendFullRowgroup(rowgroup, actual_columns);
		const auto materialized_rows = actual_columns.front().size();
		for (const auto& column : actual_columns) {
			if (column.size() != materialized_rows) {
				throw std::runtime_error("spatial-major verification materialized inconsistent coefficient columns");
			}
		}
		for (const auto group_index : rowgroup_groups[rowgroup]) {
			const auto& group = metadata.block_group_index[group_index];
			if (group.row_start > entry.physical_row_count ||
			    group.row_count > entry.physical_row_count - group.row_start) {
				throw std::runtime_error("spatial-major verification block group exceeds physical shard rows");
			}
			if (group.row_start_in_rowgroup > materialized_rows ||
			    group.row_count > materialized_rows - group.row_start_in_rowgroup) {
				throw std::runtime_error("spatial-major verification block group exceeds its materialized rowgroup");
			}
			for (uint32_t row_offset = 0U; row_offset < group.row_count; ++row_offset) {
				const auto physical_row = static_cast<size_t>(group.row_start + row_offset);
				const auto actual_row = static_cast<size_t>(group.row_start_in_rowgroup) + row_offset;
				const auto packed_key = physical_row_keys[physical_row];
				if (packed_key == kMissingPhysicalRowKey) {
					continue;
				}
				const auto [image_index, key] = unpack_physical_row_key(packed_key);
				for (size_t coefficient = 0U; coefficient < actual_columns.size(); ++coefficient) {
					const auto expected = expected_rows[physical_row][coefficient];
					const auto actual = actual_columns[coefficient][actual_row];
					if (expected == actual) {
						continue;
					}
					++result.coefficient_mismatches;
					result.max_abs_difference = std::max(
					    result.max_abs_difference, std::abs(static_cast<int>(expected) - static_cast<int>(actual)));
					consider_mismatch(result,
					                  make_mismatch(image_index,
					                                key,
					                                static_cast<uint32_t>(coefficient),
					                                expected,
					                                actual,
					                                JpegDctExactMismatchKind::kCoefficient));
				}
			}
		}
	}
	return result;
}

void merge_result(JpegDctExactVerificationResult& total, const JpegDctExactVerificationResult& local) {
	total.source_images += local.source_images;
	total.expected_blocks += local.expected_blocks;
	total.actual_blocks += local.actual_blocks;
	total.missing_blocks += local.missing_blocks;
	total.extra_blocks += local.extra_blocks;
	total.coefficient_mismatches += local.coefficient_mismatches;
	total.max_abs_difference = std::max(total.max_abs_difference, local.max_abs_difference);
	consider_mismatch(total, local.first_mismatch);
}

std::vector<ShardTask> make_shard_tasks(const JpegDctShardManifest& manifest) {
	std::vector<ShardTask> tasks;
	tasks.reserve(manifest.shards.size());
	for (const auto& entry : manifest.shards) {
		if (entry.first_global_image_index > manifest.image_count ||
		    entry.image_count > manifest.image_count - entry.first_global_image_index) {
			throw std::runtime_error("manifest shard image range is outside the manifest image count");
		}
		tasks.push_back({entry});
	}

	std::sort(tasks.begin(), tasks.end(), [](const ShardTask& left, const ShardTask& right) {
		return std::tie(left.entry.first_global_image_index, left.entry.shard_id) <
		       std::tie(right.entry.first_global_image_index, right.entry.shard_id);
	});
	uint64_t expected_first_image = 0U;
	for (const auto& task : tasks) {
		if (task.entry.first_global_image_index != expected_first_image) {
			throw std::runtime_error("manifest shard image ranges contain a gap or overlap");
		}
		expected_first_image += task.entry.image_count;
	}
	if (expected_first_image != manifest.image_count) {
		throw std::runtime_error("manifest shard image ranges do not cover the manifest image count");
	}

	std::sort(tasks.begin(), tasks.end(), [](const ShardTask& left, const ShardTask& right) {
		return left.entry.shard_id < right.entry.shard_id;
	});
	for (size_t idx = 1U; idx < tasks.size(); ++idx) {
		if (tasks[idx - 1U].entry.shard_id == tasks[idx].entry.shard_id) {
			throw std::runtime_error("manifest contains duplicate shard ids");
		}
	}
	return tasks;
}

} // namespace

JpegDctExactVerificationResult verify_jpeg_dct_manifest_exact(const std::filesystem::path&              manifest_path,
                                                              const std::vector<std::filesystem::path>& source_paths,
                                                              const size_t verify_workers) {
	if (verify_workers == 0U || verify_workers > kMaxJpegDctExactVerificationWorkers) {
		throw std::invalid_argument("verify_workers must be in [1, 32]");
	}

	const auto manifest = detail::read_jpeg_dct_shard_manifest_file(manifest_path);
	if (source_paths.size() != manifest.image_count) {
		throw std::runtime_error("source JPEG count does not match the manifest image count");
	}
	const auto tasks = make_shard_tasks(manifest);
	if (tasks.empty()) {
		return {};
	}

	const auto                                  worker_count = std::min(verify_workers, tasks.size());
	std::vector<JpegDctExactVerificationResult> shard_results(tasks.size());
	std::atomic<size_t>                         next_task {0U};
	std::atomic<bool>                           cancelled {false};
	std::mutex                                  exception_mutex;
	std::exception_ptr                          first_exception;

	const auto worker = [&]() {
		try {
			while (!cancelled.load(std::memory_order_relaxed)) {
				const auto task_index = next_task.fetch_add(1U, std::memory_order_relaxed);
				if (task_index >= tasks.size()) {
					return;
				}
				const auto&                    entry = tasks[task_index].entry;
				JpegDctExactVerificationResult shard_result;
				if (manifest.physical_layout == "dct-major/spatial-major-image-minor") {
					shard_result = verify_spatial_major_shard(manifest_path, entry, source_paths);
				} else if (manifest.physical_layout == "image-major" ||
				           manifest.physical_layout == "image-major-vector-rowgroups") {
					shard_result = verify_image_major_shard(manifest_path, entry, source_paths);
				} else {
					throw std::runtime_error("JPEG DCT exact verifier received an unsupported physical layout");
				}
				shard_results[task_index] = std::move(shard_result);
			}
		} catch (...) {
			{
				std::lock_guard<std::mutex> lock(exception_mutex);
				if (!first_exception) {
					first_exception = std::current_exception();
				}
			}
			cancelled.store(true, std::memory_order_relaxed);
		}
	};

	std::vector<std::thread> workers;
	workers.reserve(worker_count);
	try {
		for (size_t worker_index = 0U; worker_index < worker_count; ++worker_index) {
			workers.emplace_back(worker);
		}
	} catch (...) {
		cancelled.store(true, std::memory_order_relaxed);
		for (auto& thread : workers) {
			if (thread.joinable()) {
				thread.join();
			}
		}
		throw;
	}
	for (auto& thread : workers) {
		thread.join();
	}
	if (first_exception) {
		std::rethrow_exception(first_exception);
	}

	JpegDctExactVerificationResult result;
	for (const auto& shard_result : shard_results) {
		merge_result(result, shard_result);
	}
	return result;
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
