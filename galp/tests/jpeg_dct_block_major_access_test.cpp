#include "galp/jpeg_dct_block_major_access.hpp"
#include "galp/jpeg_dct_block_major_plan.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include "galp/profiles/rgbnomore.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include "jpeg/jpeg_dct_active_output_schedule.hpp"
#include "jpeg/jpeg_dct_plan_types.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

class BlockMajorTemporaryDirectory {
public:
	BlockMajorTemporaryDirectory() {
		const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
		path_ = std::filesystem::temp_directory_path() / ("galp_block_major_access_" + std::to_string(suffix));
		std::filesystem::create_directories(path_);
	}
	~BlockMajorTemporaryDirectory() {
		std::error_code ignored;
		std::filesystem::remove_all(path_, ignored);
	}
	[[nodiscard]] const std::filesystem::path& path() const noexcept {
		return path_;
	}

private:
	std::filesystem::path path_;
};

struct SyntheticDataset {
	galp::jpeg::JpegDctDatasetMetadata metadata;
	galp::jpeg::JpegDctShardManifest   manifest;
	std::filesystem::path              manifest_path;
	std::filesystem::path              metadata_path;
	std::filesystem::path              fls_path;
};

SyntheticDataset make_synthetic_dataset(const std::filesystem::path& directory) {
	using namespace galp::jpeg;
	SyntheticDataset result;
	result.manifest_path = directory / "manifest.bin";
	result.metadata_path = directory / "shard_000000.meta.bin";
	result.fls_path      = directory / "shard_000000.fls";

	JpegEncodingProfileMetadata profile;
	profile.profile_id              = 7U;
	profile.h_samp_factor           = 1;
	profile.v_samp_factor           = 1;
	profile.quant_tbl_no            = 0;
	profile.quant_table_fingerprint = UINT64_C(0x123456789abcdef0);
	for (size_t index = 0U; index < profile.quant_table_values.size(); ++index) {
		profile.quant_table_values[index] = static_cast<uint16_t>(index + 1U);
	}
	result.metadata.encoding_profiles.push_back(profile);
	result.metadata.row_ordering = JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	result.metadata.z_curve_block_order = true;
	const std::array<std::array<uint16_t, 2>, 8> y_shapes {{{1U, 4U},
	                                                        {4U, 1U},
	                                                        {2U, 2U},
	                                                        {3U, 3U},
	                                                        {3U, 1U},
	                                                        {1U, 3U},
	                                                        {4U, 2U},
	                                                        {2U, 4U}}};
	for (uint32_t image_index = 0U; image_index < y_shapes.size(); ++image_index) {
		JpegImageMetadata image;
		image.image_width      = 16U + image_index;
		image.image_height     = 12U + image_index;
		image.data_precision   = 8U;
		image.jpeg_color_space = image_index == 7U ? 1 : 3;
		JpegComponentMetadata y;
		y.semantic_slot_id        = 0U;
		y.local_component_index   = 0U;
		y.component_id            = 1;
		y.width_in_blocks         = y_shapes[image_index][0];
		y.height_in_blocks        = y_shapes[image_index][1];
		y.padded_width_in_blocks  = y.width_in_blocks;
		y.padded_height_in_blocks = y.height_in_blocks;
		y.h_samp_factor           = 1;
		y.v_samp_factor           = 1;
		y.quant_tbl_no            = 0;
		y.encoding_profile_id     = profile.profile_id;
		image.components.push_back(y);
		JpegComponentMetadata chroma = y;
		chroma.semantic_slot_id      = 1U;
		chroma.local_component_index = 1U;
		chroma.component_id          = 2;
		chroma.present               = image_index == 0U || image_index == 3U;
		chroma.width_in_blocks       = chroma.present ? 1U : 0U;
		chroma.height_in_blocks      = chroma.present ? 1U : 0U;
		chroma.padded_width_in_blocks  = chroma.width_in_blocks;
		chroma.padded_height_in_blocks = chroma.height_in_blocks;
		image.components.push_back(chroma);
		chroma.semantic_slot_id      = 2U;
		chroma.local_component_index = 2U;
		chroma.component_id          = 3;
		image.components.push_back(chroma);
		result.metadata.images.push_back(std::move(image));
	}
	result.metadata.image_count = result.metadata.images.size();
	result.metadata.semantic_components = result.metadata.images.front().components;
	result.metadata.semantic_components[0].width_in_blocks = 4U;
	result.metadata.semantic_components[0].height_in_blocks = 4U;

	uint64_t row_cursor = 0U;
	for (uint32_t slot = 0U; slot < 3U; ++slot) {
		const uint32_t width  = slot == 0U ? 4U : 1U;
		const uint32_t height = slot == 0U ? 4U : 1U;
		const auto order = galp::jpeg::detail::make_block_order(width, height, true);
		for (uint32_t position = 0U; position < order.size(); ++position) {
			const auto coordinate = order[position];
			uint32_t row_count = 0U;
			for (const auto& image : result.metadata.images) {
				const auto& component = image.components[slot];
				row_count += component.present && coordinate.x < component.width_in_blocks &&
				                     coordinate.y < component.height_in_blocks
				                 ? 1U
				                 : 0U;
			}
			if (row_count == 0U) {
				continue;
			}
			result.metadata.block_group_index.push_back(JpegDctBlockGroupIndex {slot,
			                                                                            position,
			                                                                            coordinate.x,
			                                                                            coordinate.y,
			                                                                            row_cursor,
			                                                                            row_count,
			                                                                            0U,
			                                                                            static_cast<uint32_t>(row_cursor)});
			row_cursor += row_count;
		}
	}
	write_jpeg_dct_metadata(result.metadata,
	                        result.metadata_path,
	                        JpegDctMetadataWriterOptions {JpegMetadataProfile::kReconstructableJpeg});
	{
		std::ofstream fls(result.fls_path, std::ios::binary);
		std::vector<uint8_t> dummy_fls(1024U * 1024U, 0U);
		const std::array<uint8_t, 9> marker {{'s', 'y', 'n', 't', 'h', 'e', 't', 'i', 'c'}};
		std::copy(marker.begin(), marker.end(), dummy_fls.begin());
		fls.write(reinterpret_cast<const char*>(dummy_fls.data()), static_cast<std::streamsize>(dummy_fls.size()));
	}
	JpegDctShardManifestEntry entry;
	entry.shard_id                 = 0U;
	entry.first_global_image_index = 0U;
	entry.image_count              = static_cast<uint32_t>(result.metadata.images.size());
	entry.real_row_count           = row_cursor;
	entry.physical_row_count       = row_cursor;
	entry.rowgroup_count           = 1U;
	entry.block_group_count        = static_cast<uint32_t>(result.metadata.block_group_index.size());
	entry.fls_file_size            = std::filesystem::file_size(result.fls_path);
	entry.metadata_file_size       = std::filesystem::file_size(result.metadata_path);
	entry.fls_file_name            = result.fls_path.filename().string();
	entry.metadata_file_name       = result.metadata_path.filename().string();
	result.manifest.version          = 1U;
	result.manifest.rowgroup_vectors = 1U;
	result.manifest.image_count      = result.metadata.images.size();
	result.manifest.shards.push_back(entry);
	write_jpeg_dct_shard_manifest(result.manifest, result.manifest_path);
	return result;
}

SyntheticDataset make_two_shard_synthetic_dataset(const std::filesystem::path& directory) {
	auto result = make_synthetic_dataset(directory);
	auto second = result.manifest.shards.front();
	second.shard_id = 1U;
	second.first_global_image_index = second.image_count;
	second.fls_file_name      = "shard_000001.fls";
	second.metadata_file_name = "shard_000001.meta.bin";
	std::filesystem::copy_file(result.fls_path, directory / second.fls_file_name);
	std::filesystem::copy_file(result.metadata_path, directory / second.metadata_file_name);
	result.manifest.shards.push_back(second);
	result.manifest.image_count += second.image_count;
	galp::jpeg::write_jpeg_dct_shard_manifest(result.manifest, result.manifest_path);
	return result;
}

using SelectedPhysicalRow = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>;
using SelectedVector      = std::tuple<uint32_t, uint32_t, uint32_t>;

const galp::jpeg::JpegComponentMetadata* find_test_component(
	const galp::jpeg::JpegImageMetadata& image, const uint32_t semantic_slot_id) {
	for (const auto& component : image.components) {
		if (component.present && component.semantic_slot_id == semantic_slot_id) {
			return &component;
		}
	}
	return nullptr;
}

const galp::jpeg::JpegDctBlockGroupIndex& find_test_group(
	const galp::jpeg::JpegDctDatasetMetadata& metadata,
	const uint32_t semantic_slot_id,
	const uint32_t block_x,
	const uint32_t block_y) {
	for (const auto& group : metadata.block_group_index) {
		if (group.semantic_slot_id == semantic_slot_id && group.block_x == block_x && group.block_y == block_y) {
			return group;
		}
	}
	throw std::runtime_error("synthetic legacy group was not found");
}

std::set<SelectedPhysicalRow> expected_physical_rows(
	const SyntheticDataset& source, const galp::jpeg::JpegDctBlockMajorCompactPlan& plan) {
	std::set<SelectedPhysicalRow> result;
	for (const auto& request : plan.requests) {
		const auto& image = source.metadata.images.at(request.local_image_index);
		for (const auto& support : request.components) {
			if (!support.present) {
				continue;
			}
			const auto* component = find_test_component(image, support.semantic_slot_id);
			if (component == nullptr) {
				throw std::runtime_error("compact request selected a missing synthetic component");
			}
			const auto x0 = static_cast<uint32_t>(
			    std::clamp<int64_t>(support.x, 0, static_cast<int64_t>(component->width_in_blocks)));
			const auto y0 = static_cast<uint32_t>(
			    std::clamp<int64_t>(support.y, 0, static_cast<int64_t>(component->height_in_blocks)));
			const auto x1 = static_cast<uint32_t>(std::clamp<int64_t>(
			    static_cast<int64_t>(support.x) + support.width, 0, component->width_in_blocks));
			const auto y1 = static_cast<uint32_t>(std::clamp<int64_t>(
			    static_cast<int64_t>(support.y) + support.height, 0, component->height_in_blocks));
			for (uint32_t y = y0; y < y1; ++y) {
				for (uint32_t x = x0; x < x1; ++x) {
					uint32_t rank = 0U;
					for (uint32_t preceding = 0U; preceding < request.local_image_index; ++preceding) {
						const auto* preceding_component =
						    find_test_component(source.metadata.images[preceding], support.semantic_slot_id);
						rank += preceding_component != nullptr && x < preceding_component->width_in_blocks &&
						                y < preceding_component->height_in_blocks
						            ? 1U
						            : 0U;
					}
					result.emplace(request.shard_id, support.semantic_slot_id, x, y, rank);
				}
			}
		}
	}
	return result;
}

std::set<SelectedVector> expected_vectors(const SyntheticDataset& source,
	                                       const std::set<SelectedPhysicalRow>& physical_rows) {
	std::set<SelectedVector> result;
	for (const auto& [shard_id, semantic_slot_id, x, y, rank] : physical_rows) {
		const auto& group = find_test_group(source.metadata, semantic_slot_id, x, y);
		result.emplace(shard_id,
		               group.fls_rowgroup_index,
		               static_cast<uint32_t>((group.row_start_in_rowgroup + rank) / 1024U));
	}
	return result;
}

std::set<SelectedVector> actual_vectors(const galp::jpeg::JpegDctBlockMajorCompactPlan& plan) {
	std::set<SelectedVector> result;
	for (const auto& run : plan.vector_runs) {
		for (uint32_t vector = run.first_vector; vector < run.first_vector + run.vector_count; ++vector) {
			result.emplace(run.shard_id, run.rowgroup_index, vector);
		}
	}
	return result;
}

std::set<SelectedVector> preview_vectors(const galp::jpeg::JpegDctDeviceBatchPlanPreview& preview) {
	std::set<SelectedVector> result;
	for (const auto& rowgroup : preview.rowgroup_vector_plans) {
		for (const auto vector : rowgroup.selected_vectors) {
			result.emplace(rowgroup.rowgroup.shard_id, rowgroup.rowgroup.rowgroup_index, vector);
		}
	}
	return result;
}

galp::jpeg::JpegDctBlockMajorAccessRank decode_exported_rank_cell(
	const galp::jpeg::JpegDctBlockMajorAccessRankCellRecord& cell, const uint32_t local_image_index) {
	using galp::jpeg::JpegDctBlockMajorPresenceEncoding;
	galp::jpeg::JpegDctBlockMajorAccessRank result {cell.cell_id, 0U, false};
	if (local_image_index >= cell.image_count) {
		throw std::runtime_error("exported test rank image is out of range");
	}
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kEmpty) {
		return result;
	}
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kAllPresent) {
		result.rank    = local_image_index;
		result.present = true;
		return result;
	}
	if (cell.encoding == JpegDctBlockMajorPresenceEncoding::kBitmapRank) {
		const auto bit_bytes = (cell.image_count + 7U) / 8U;
		const auto checkpoint = local_image_index / cell.rank_checkpoint_images;
		const auto checkpoint_offset = bit_bytes + checkpoint * sizeof(uint16_t);
		if (checkpoint_offset + 1U >= cell.payload.size()) {
			throw std::runtime_error("exported test bitmap checkpoint is truncated");
		}
		result.rank = static_cast<uint32_t>(cell.payload[checkpoint_offset]) |
		              (static_cast<uint32_t>(cell.payload[checkpoint_offset + 1U]) << 8U);
		const auto begin = checkpoint * cell.rank_checkpoint_images;
		for (uint32_t image = begin; image < local_image_index; ++image) {
			result.rank += (cell.payload[image / 8U] >> (image % 8U)) & 1U;
		}
		result.present = ((cell.payload[local_image_index / 8U] >> (local_image_index % 8U)) & 1U) != 0U;
		return result;
	}
	const auto listed_count = cell.encoding == JpegDctBlockMajorPresenceEncoding::kSparseList
	                              ? cell.present_count
	                              : cell.image_count - cell.present_count;
	size_t cursor = 0U;
	uint32_t value = 0U;
	uint32_t listed_before = 0U;
	bool listed = false;
	for (uint32_t index = 0U; index < listed_count; ++index) {
		uint32_t delta = 0U;
		unsigned shift = 0U;
		while (true) {
			if (cursor >= cell.payload.size() || shift >= 32U) {
				throw std::runtime_error("exported test ULEB128 is malformed");
			}
			const auto byte = cell.payload[cursor++];
			delta |= static_cast<uint32_t>(byte & 0x7fU) << shift;
			if ((byte & 0x80U) == 0U) {
				break;
			}
			shift += 7U;
		}
		value = index == 0U ? delta : value + delta;
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

galp::jpeg::detail::JpegDctDeviceBlockMajorActiveOutputSchedule reference_active_output_schedule(
    const galp::jpeg::detail::JpegDctDeviceBlockMajorPlanlessPlan& plan,
    const std::vector<galp::jpeg::detail::JpegDctDeviceBlockMajorRowgroupWorkset>& worksets,
    const galp::jpeg::JpegDctGridTransformSpec& transform) {
	using namespace galp::jpeg::detail;
	JpegDctDeviceBlockMajorActiveOutputSchedule schedule;
	if (plan.images.empty()) {
		schedule.offsets.push_back(0U);
		return schedule;
	}
	uint32_t workset_count = 0U;
	std::unordered_map<uint64_t, uint32_t> rowgroup_workset;
	for (const auto& workset : worksets) {
		workset_count = std::max(workset_count, workset.workset_index + 1U);
		rowgroup_workset.emplace(
		    (static_cast<uint64_t>(workset.shard_id) << 32U) | workset.rowgroup_index,
		    workset.workset_index);
	}
	struct GroupKey {
		uint32_t shard;
		uint32_t slot;
		uint32_t x;
		uint32_t y;
		bool operator==(const GroupKey& other) const noexcept {
			return shard == other.shard && slot == other.slot && x == other.x && y == other.y;
		}
	};
	struct GroupKeyHash {
		size_t operator()(const GroupKey& key) const noexcept {
			uint64_t hash = static_cast<uint64_t>(key.shard) * UINT64_C(0x9e3779b185ebca87);
			for (const auto value : {key.slot, key.x, key.y}) {
				hash ^= static_cast<uint64_t>(value) + UINT64_C(0x9e3779b97f4a7c15) +
				        (hash << 6U) + (hash >> 2U);
			}
			return static_cast<size_t>(hash);
		}
	};
	std::unordered_map<GroupKey, uint32_t, GroupKeyHash> group_workset;
	group_workset.reserve(plan.groups.size());
	for (const auto& group : plan.groups) {
		const auto owner = rowgroup_workset.at(
		    (static_cast<uint64_t>(group.shard_id) << 32U) | group.rowgroup_index);
		group_workset.emplace(
		    GroupKey {group.shard_id, group.semantic_slot_id, group.block_x, group.block_y}, owner);
	}
	const auto y_blocks = static_cast<uint64_t>(transform.y_output_width_blocks) *
	                      transform.y_output_height_blocks;
	const auto cbcr_blocks = static_cast<uint64_t>(transform.cbcr_output_width_blocks) *
	                         transform.cbcr_output_height_blocks;
	const auto blocks_per_image = y_blocks + 2U * cbcr_blocks;
	schedule.logical_output_block_count = blocks_per_image * plan.images.size();
	std::vector<std::vector<uint32_t>> outputs(workset_count);
	const auto average = static_cast<size_t>(schedule.logical_output_block_count / workset_count);
	for (auto& slice : outputs) {
		slice.reserve(average);
	}
	std::vector<uint32_t> owner_generations(workset_count, 0U);
	std::vector<uint32_t> owners;
	owners.reserve(std::min<size_t>(workset_count, 16U));
	uint32_t generation = 0U;
	for (size_t image_index = 0U; image_index < plan.images.size(); ++image_index) {
		const auto& image = plan.images[image_index];
		for (uint32_t component_index = 0U; component_index < image.components.size(); ++component_index) {
			const auto& component = image.components[component_index];
			if (component.present == 0U) {
				continue;
			}
			const auto output_width = component_index == 0U ? transform.y_output_width_blocks
			                                                   : transform.cbcr_output_width_blocks;
			const auto output_height = component_index == 0U ? transform.y_output_height_blocks
			                                                    : transform.cbcr_output_height_blocks;
			const auto component_offset = component_index == 0U
			                                  ? 0U
			                                  : y_blocks + static_cast<uint64_t>(component_index - 1U) * cbcr_blocks;
			for (uint32_t output_y = 0U; output_y < output_height; ++output_y) {
				const auto source_y_begin = static_cast<uint32_t>(
				    static_cast<uint64_t>(output_y) * component.y_down_factor / component.y_up_factor);
				const auto source_y_end = static_cast<uint32_t>(
				    ((static_cast<uint64_t>(output_y + 1U) * component.y_down_factor) - 1U) /
				    component.y_up_factor);
				for (uint32_t output_x = 0U; output_x < output_width; ++output_x) {
					if (++generation == 0U) {
						std::fill(owner_generations.begin(), owner_generations.end(), 0U);
						generation = 1U;
					}
					owners.clear();
					const auto source_x_begin = static_cast<uint32_t>(
					    static_cast<uint64_t>(output_x) * component.x_down_factor / component.x_up_factor);
					const auto source_x_end = static_cast<uint32_t>(
					    ((static_cast<uint64_t>(output_x + 1U) * component.x_down_factor) - 1U) /
					    component.x_up_factor);
					for (uint32_t source_y = source_y_begin; source_y <= source_y_end; ++source_y) {
						for (uint32_t source_x = source_x_begin; source_x <= source_x_end; ++source_x) {
							const auto absolute_x = static_cast<int64_t>(component.crop_x) + source_x;
							const auto absolute_y = static_cast<int64_t>(component.crop_y) + source_y;
							if (absolute_x < 0 || absolute_y < 0 || absolute_x >= component.width_in_blocks ||
							    absolute_y >= component.height_in_blocks) {
								continue;
							}
							++schedule.source_contribution_count;
							const auto owner = group_workset.at(GroupKey {image.shard_id,
							                                                     component.semantic_slot_id,
							                                                     static_cast<uint32_t>(absolute_x),
							                                                     static_cast<uint32_t>(absolute_y)});
							if (owner_generations[owner] != generation) {
								owner_generations[owner] = generation;
								owners.push_back(owner);
							}
						}
					}
					std::sort(owners.begin(), owners.end());
					const auto linear = static_cast<uint32_t>(
					    static_cast<uint64_t>(image_index) * blocks_per_image + component_offset +
					    static_cast<uint64_t>(output_y) * output_width + output_x);
					for (const auto owner : owners) {
						outputs[owner].push_back(linear);
					}
				}
			}
		}
	}
	schedule.offsets.assign(static_cast<size_t>(workset_count) + 1U, 0U);
	for (uint32_t workset = 0U; workset < workset_count; ++workset) {
		schedule.offsets[workset + 1U] = schedule.offsets[workset] + outputs[workset].size();
	}
	schedule.active_output_blocks.reserve(static_cast<size_t>(schedule.offsets.back()));
	uint64_t output_capacity_bytes = 0U;
	for (const auto& slice : outputs) {
		schedule.active_output_blocks.insert(schedule.active_output_blocks.end(), slice.begin(), slice.end());
		output_capacity_bytes += static_cast<uint64_t>(slice.capacity()) * sizeof(uint32_t);
	}
	schedule.output_workset_ownership_count = schedule.offsets.back();
	schedule.temporary_bytes_peak = output_capacity_bytes +
	    static_cast<uint64_t>(plan.groups.size()) *
	        (sizeof(GroupKey) + sizeof(uint32_t) + 4U * sizeof(void*)) +
	    static_cast<uint64_t>(worksets.size()) *
	        (sizeof(uint64_t) + sizeof(uint32_t) + 4U * sizeof(void*));
	return schedule;
}

TEST(JpegDctBlockMajorAccess, MmapRankSelectAndTopologyMatchLegacyMetadata) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_synthetic_dataset(temporary.path());
	const auto descriptor_path = temporary.path() / "access" / "shard_000000.block_major_access.bin";
	galp::jpeg::JpegDctBlockMajorAccessBuildOptions options;
	options.rank_checkpoint_images          = 8U;
	options.topology_checkpoint_coordinates = 3U;
	options.exhaustive_rank_validation       = true;
	const auto report = galp::jpeg::build_jpeg_dct_block_major_access_descriptor(
	    source.manifest_path, source.manifest, source.manifest.shards.front(), descriptor_path, options);
	EXPECT_EQ(report.validation.images_checked, 8U);
	EXPECT_EQ(report.validation.groups_checked, source.metadata.block_group_index.size());
	EXPECT_GT(report.all_present_cells, 0U);
	EXPECT_GT(report.sparse_cells, 0U);
	EXPECT_GT(report.missing_cells, 0U);
	EXPECT_GT(report.bitmap_cells, 0U);
	EXPECT_GT(report.empty_cells, 0U);

	auto descriptor = galp::jpeg::JpegDctBlockMajorAccessDescriptor::Open(descriptor_path);
	descriptor.ValidateSource(
	    source.manifest_path, source.manifest.shards.front(), source.metadata_path, source.fls_path);
	const auto all = descriptor.Rank(0U, 0U, 0U, 6U);
	EXPECT_TRUE(all.present);
	EXPECT_EQ(all.rank, 6U);
	const auto sparse = descriptor.Rank(0U, 3U, 0U, 6U);
	EXPECT_TRUE(sparse.present);
	EXPECT_EQ(sparse.rank, 1U);
	const auto absent = descriptor.Rank(0U, 3U, 0U, 3U);
	EXPECT_FALSE(absent.present);
	EXPECT_EQ(absent.rank, 1U);
	EXPECT_EQ(descriptor.Select(0U, 3U, 0U, 0U), 1U);
	EXPECT_EQ(descriptor.Select(0U, 3U, 0U, 1U), 6U);
	EXPECT_FALSE(descriptor.FindGroup(0U, 3U, 3U).has_value());
	for (uint32_t group_index = 0U; group_index < source.metadata.block_group_index.size(); ++group_index) {
		const auto& legacy = source.metadata.block_group_index[group_index];
		const auto compact = descriptor.FindGroup(legacy.semantic_slot_id, legacy.block_x, legacy.block_y);
		ASSERT_TRUE(compact.has_value());
		EXPECT_EQ(compact->group_id, group_index);
		EXPECT_EQ(compact->row_start, legacy.row_start);
		EXPECT_EQ(compact->row_count, legacy.row_count);
		EXPECT_EQ(compact->fls_rowgroup_index, legacy.fls_rowgroup_index);
		EXPECT_EQ(compact->row_start_in_rowgroup, legacy.row_start_in_rowgroup);
		const auto exported = descriptor.rank_cell(compact->rank_cell_id);
		for (uint32_t rank = 0U; rank < compact->row_count; ++rank) {
			EXPECT_EQ(descriptor.SelectCell(compact->rank_cell_id, rank),
			          descriptor.Select(legacy.semantic_slot_id, legacy.block_x, legacy.block_y, rank));
		}
		EXPECT_FALSE(descriptor.SelectCell(compact->rank_cell_id, compact->row_count).has_value());
		for (uint32_t image = 0U; image < source.metadata.images.size(); ++image) {
			const auto expected = descriptor.Rank(legacy.semantic_slot_id, legacy.block_x, legacy.block_y, image);
			const auto actual   = decode_exported_rank_cell(exported, image);
			EXPECT_EQ(actual.cell_id, expected.cell_id);
			EXPECT_EQ(actual.rank, expected.rank);
			EXPECT_EQ(actual.present, expected.present);
		}
		for (uint32_t begin = 0U; begin <= source.metadata.images.size(); ++begin) {
			for (uint32_t end = begin; end <= source.metadata.images.size(); ++end) {
				const auto interval = descriptor.RankCellInterval(compact->rank_cell_id, begin, end);
				const auto expected_begin = begin == source.metadata.images.size()
				                                ? compact->row_count
				                                : descriptor.Rank(
				                                      legacy.semantic_slot_id, legacy.block_x, legacy.block_y, begin).rank;
				const auto expected_end = end == source.metadata.images.size()
				                              ? compact->row_count
				                              : descriptor.Rank(
				                                    legacy.semantic_slot_id, legacy.block_x, legacy.block_y, end).rank;
				EXPECT_EQ(interval.rank_begin, expected_begin);
				EXPECT_EQ(interval.rank_end, expected_end);
			}
		}
	}
	EXPECT_THROW((void)descriptor.RankCellInterval(0U, 5U, 4U), std::runtime_error);
	EXPECT_THROW((void)descriptor.RankCellInterval(0U, 0U, 9U), std::runtime_error);
	const auto quant = descriptor.quant_table(0U);
	EXPECT_EQ(quant.fingerprint, source.metadata.encoding_profiles.front().quant_table_fingerprint);
	EXPECT_EQ(quant.values, source.metadata.encoding_profiles.front().quant_table_values);
}

TEST(JpegDctBlockMajorAccess, DatasetBuilderPublishesCompanionIndexAndStorageGates) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_synthetic_dataset(temporary.path());
	galp::jpeg::JpegDctBlockMajorAccessBuildOptions options;
	options.validate_after_write = true;
	const auto report = galp::jpeg::build_jpeg_dct_block_major_access_dataset(
	    source.manifest_path, temporary.path() / "sidecars", options);
	ASSERT_EQ(report.shards.size(), 1U);
	EXPECT_TRUE(report.passes_one_percent);
	EXPECT_TRUE(report.passes_half_percent);
	EXPECT_TRUE(std::filesystem::exists(report.index_path));
	EXPECT_EQ(std::filesystem::file_size(report.index_path), report.index_bytes);
	EXPECT_TRUE(std::filesystem::exists(report.shards.front().descriptor_path));
	EXPECT_EQ(std::filesystem::file_size(report.shards.front().descriptor_path),
	          report.shards.front().descriptor_bytes);
	EXPECT_GT(report.shards.front().presence_payload_bytes, 0U);
	const auto companion = galp::jpeg::read_jpeg_dct_block_major_access_index(
	    report.index_path, source.manifest_path, source.manifest);
	ASSERT_EQ(companion.shards.size(), 1U);
	EXPECT_EQ(companion.index_bytes, report.index_bytes);
	EXPECT_EQ(companion.shards.front().descriptor_bytes, report.shards.front().descriptor_bytes);
	{
		std::fstream bytes(report.index_path, std::ios::binary | std::ios::in | std::ios::out);
		bytes.seekg(63);
		char value = 0;
		bytes.read(&value, 1);
		value ^= 0x5a;
		bytes.seekp(63);
		bytes.write(&value, 1);
	}
	EXPECT_THROW((void)galp::jpeg::read_jpeg_dct_block_major_access_index(
	                 report.index_path, source.manifest_path, source.manifest),
	             std::runtime_error);
}

TEST(JpegDctBlockMajorAccess, RejectsCorruptionAndWrongSourceIdentity) {
	BlockMajorTemporaryDirectory temporary;
	auto source = make_synthetic_dataset(temporary.path());
	const auto descriptor_path = temporary.path() / "access.bin";
	galp::jpeg::build_jpeg_dct_block_major_access_descriptor(
	    source.manifest_path, source.manifest, source.manifest.shards.front(), descriptor_path);
	{
		auto descriptor = galp::jpeg::JpegDctBlockMajorAccessDescriptor::Open(descriptor_path);
		auto wrong_manifest = source.manifest;
		++wrong_manifest.rowgroups_per_shard;
		const auto wrong_path = temporary.path() / "wrong_manifest.bin";
		galp::jpeg::write_jpeg_dct_shard_manifest(wrong_manifest, wrong_path);
		EXPECT_THROW(descriptor.ValidateSource(
		                 wrong_path, source.manifest.shards.front(), source.metadata_path, source.fls_path),
		             std::runtime_error);
	}
	{
		std::fstream bytes(descriptor_path, std::ios::binary | std::ios::in | std::ios::out);
		bytes.seekg(300);
		char value = 0;
		bytes.read(&value, 1);
		value ^= 0x5a;
		bytes.seekp(300);
		bytes.write(&value, 1);
	}
	EXPECT_THROW(galp::jpeg::JpegDctBlockMajorAccessDescriptor::Open(descriptor_path), std::runtime_error);
}

TEST(JpegDctBlockMajorPlan, RandomDuplicateCrossShardRowsAndVectorsMatchLegacyMapping) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_two_shard_synthetic_dataset(temporary.path());
	const auto sidecars = temporary.path() / "sidecars";
	galp::jpeg::build_jpeg_dct_block_major_access_dataset(source.manifest_path, sidecars);
	galp::jpeg::JpegDctBlockMajorCompactPlanner planner(source.manifest_path, sidecars);
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests {
	    {10U, {1U, 1U, 8U, 7U}, false, "ten", "a"},
	    {1U, {}, false, "one-full", "b"},
	    {1U, {2U, 2U, 5U, 4U}, true, "one-small", "c"},
	    {15U, {}, false, "fifteen", "d"},
	    {0U, {0U, 0U, 7U, 6U}, false, "zero", "e"},
	    {9U, {3U, 2U, 6U, 5U}, false, "nine", "f"},
	};
	const auto plan = planner.Plan(requests);
	EXPECT_FALSE(plan.stats.input_was_shard_local_monotonic);
	EXPECT_EQ(plan.stats.request_sort_items, requests.size());
	EXPECT_EQ(plan.stats.unique_image_count, requests.size() - 1U);
	EXPECT_EQ(plan.stats.duplicate_output_count, 1U);
	EXPECT_EQ(plan.stats.expanded_transform_items, 0U);
	EXPECT_EQ(plan.stats.global_transform_sort_items, 0U);
	EXPECT_EQ(plan.requests.size(), requests.size());
	EXPECT_LT(plan.stats.compact_plan_bytes, 1U << 20U);
	EXPECT_GE(plan.stats.compact_plan_peak_bytes, plan.stats.compact_plan_bytes);
	std::set<uint32_t> output_slots;
	for (const auto& request : plan.requests) {
		output_slots.insert(request.output_slot);
	}
	EXPECT_EQ(output_slots, (std::set<uint32_t> {0U, 1U, 2U, 3U, 4U, 5U}));
	std::array<const galp::jpeg::JpegDctBlockMajorCompactRequest*, 6> by_output {};
	for (const auto& request : plan.requests) {
		by_output.at(request.output_slot) = &request;
	}
	ASSERT_NE(by_output[0], nullptr);
	EXPECT_EQ(by_output[0]->components[0].x, 0);
	EXPECT_EQ(by_output[0]->components[0].y, 0);
	EXPECT_EQ(by_output[0]->components[0].width, 1U);
	EXPECT_EQ(by_output[0]->components[0].height, 2U);
	ASSERT_NE(by_output[2], nullptr);
	EXPECT_TRUE(by_output[2]->horizontal_flip);
	EXPECT_EQ(by_output[2]->components[0].x, 0);
	EXPECT_EQ(by_output[2]->components[0].y, 0);
	EXPECT_EQ(by_output[2]->components[0].width, 2U);
	EXPECT_EQ(by_output[2]->components[0].height, 1U);
	ASSERT_NE(by_output[3], nullptr);
	EXPECT_TRUE(by_output[3]->components[0].present);
	EXPECT_FALSE(by_output[3]->components[1].present);
	EXPECT_FALSE(by_output[3]->components[2].present);
	ASSERT_NE(by_output[4], nullptr);
	EXPECT_EQ(by_output[4]->components[0].width, 1U);
	EXPECT_EQ(by_output[4]->components[0].height, 2U);
	EXPECT_TRUE(by_output[4]->components[1].present);
	EXPECT_TRUE(by_output[4]->components[2].present);
	EXPECT_EQ(by_output[4]->components[1].width, 1U);
	EXPECT_EQ(by_output[4]->components[1].height, 1U);
	ASSERT_NE(by_output[5], nullptr);
	EXPECT_EQ(by_output[5]->components[0].width, 3U);
	EXPECT_EQ(by_output[5]->components[0].height, 1U);
	const auto expected_rows = expected_physical_rows(source, plan);
	EXPECT_TRUE(plan.rank_runs.empty());
	EXPECT_EQ(actual_vectors(plan), expected_vectors(source, expected_rows));
}

TEST(JpegDctBlockMajorPlan, DescriptorCacheLoadsOnlyTouchedShardsAndIsConcurrentLifetimeSafe) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_two_shard_synthetic_dataset(temporary.path());
	const auto sidecars = temporary.path() / "sidecars";
	galp::jpeg::build_jpeg_dct_block_major_access_dataset(source.manifest_path, sidecars);
	galp::jpeg::JpegDctBlockMajorCompactPlanner planner(source.manifest_path, sidecars);
	EXPECT_EQ(planner.loaded_descriptor_count(), 0U);
	EXPECT_EQ(planner.loaded_descriptor_bytes(), 0U);
	EXPECT_GT(planner.descriptor_cache_byte_bound(), 0U);

	std::atomic<uint32_t> completed {0U};
	std::vector<std::thread> workers;
	std::vector<std::exception_ptr> errors(8U);
	for (size_t worker = 0U; worker < errors.size(); ++worker) {
		workers.emplace_back([&, worker]() {
			try {
				const auto local = planner.Plan({{1U, {}, false, "one", "concurrent"}});
				if (!local.requests.empty()) {
					completed.fetch_add(1U, std::memory_order_relaxed);
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
	EXPECT_EQ(completed.load(std::memory_order_relaxed), errors.size());
	EXPECT_EQ(planner.loaded_descriptor_count(), 1U);
	EXPECT_GT(planner.loaded_descriptor_bytes(), 0U);
	EXPECT_LT(planner.loaded_descriptor_bytes(), planner.descriptor_cache_byte_bound());

	const auto cross_shard = planner.Plan({{1U, {}, false, "one", "a"},
	                                       {15U, {}, false, "fifteen", "b"}});
	EXPECT_EQ(cross_shard.requests.size(), 2U);
	EXPECT_EQ(planner.loaded_descriptor_count(), 2U);
	EXPECT_EQ(planner.loaded_descriptor_bytes(), planner.descriptor_cache_byte_bound());
}

TEST(JpegDctBlockMajorPlan, SequentialFixedCropUsesRequestRunsWithoutExpandedItems) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_two_shard_synthetic_dataset(temporary.path());
	const auto sidecars = temporary.path() / "sidecars";
	galp::jpeg::build_jpeg_dct_block_major_access_dataset(source.manifest_path, sidecars);
	galp::jpeg::JpegDctBlockMajorCompactPlanner planner(source.manifest_path, sidecars);
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	for (const auto image : {1U, 2U, 4U, 5U, 6U, 7U, 9U, 10U, 12U, 13U, 14U, 15U}) {
		requests.push_back({image, {}, false, std::to_string(image), "fixed"});
	}
	const auto plan = planner.Plan(requests, galp::profiles::rgbnomore_val_dct_grid_transform());
	EXPECT_TRUE(plan.stats.input_was_shard_local_monotonic);
	EXPECT_EQ(plan.stats.request_sort_items, 0U);
	EXPECT_EQ(plan.stats.shard_local_request_runs, 4U);
	EXPECT_EQ(plan.stats.expanded_transform_items, 0U);
	EXPECT_EQ(plan.stats.global_transform_sort_items, 0U);
	EXPECT_GT(plan.stats.touched_block_groups, 0U);
	EXPECT_GT(plan.stats.selected_vectors, 0U);
	EXPECT_EQ(plan.stats.duplicate_physical_read_count, 0U);
	EXPECT_EQ(plan.stats.rowgroup_revisit_count, 0U);
	EXPECT_EQ(plan.stats.vector_run_revisit_count, 0U);
	EXPECT_EQ(plan.stats.physical_read_order_inversions, 0U);
	EXPECT_TRUE(std::is_sorted(plan.rowgroups.begin(), plan.rowgroups.end(), [](const auto& left, const auto& right) {
		return std::tie(left.shard_id, left.rowgroup_index) < std::tie(right.shard_id, right.rowgroup_index);
	}));
	for (const auto& rowgroup : plan.rowgroups) {
		uint32_t previous_end = 0U;
		for (uint32_t run_index = 0U; run_index < rowgroup.vector_run_count; ++run_index) {
			const auto& run = plan.vector_runs[rowgroup.first_vector_run + run_index];
			EXPECT_EQ(run.shard_id, rowgroup.shard_id);
			EXPECT_EQ(run.rowgroup_index, rowgroup.rowgroup_index);
			EXPECT_GE(run.first_vector, previous_end);
			previous_end = run.first_vector + run.vector_count;
		}
	}
	EXPECT_EQ(plan.stats.touched_rank_cells, plan.rank_cells.size());
	EXPECT_EQ(plan.stats.rank_payload_bytes, plan.rank_payload.size());
	EXPECT_EQ(plan.stats.touched_quant_tables, plan.quant_tables.size());
	ASSERT_FALSE(plan.rank_cells.empty());
	ASSERT_FALSE(plan.quant_tables.empty());
	for (const auto& binding : plan.group_bindings) {
		ASSERT_LT(binding.runtime_rank_cell_index, plan.rank_cells.size());
		const auto& cell = plan.rank_cells[binding.runtime_rank_cell_index];
		EXPECT_EQ(cell.shard_id, binding.shard_id);
		EXPECT_EQ(cell.source_cell_id, binding.rank_cell_id);
		EXPECT_LE(static_cast<uint64_t>(cell.payload_offset) + cell.payload_size, plan.rank_payload.size());
	}
	for (const auto& request : plan.requests) {
		for (const auto& support : request.components) {
			if (support.present) {
				EXPECT_LT(support.quant_table_index, plan.quant_tables.size());
			}
		}
	}
	const auto expected_rows = expected_physical_rows(source, plan);
	EXPECT_TRUE(plan.rank_runs.empty());
	EXPECT_EQ(actual_vectors(plan), expected_vectors(source, expected_rows));
}

TEST(JpegDctBlockMajorPlan, CanonicalTemplateRoundTripsWithStableHitMissDigest) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_synthetic_dataset(temporary.path());
	const auto sidecars = temporary.path() / "sidecars";
	galp::jpeg::build_jpeg_dct_block_major_access_dataset(source.manifest_path, sidecars);
	galp::jpeg::JpegDctBlockMajorCompactPlanner planner(source.manifest_path, sidecars);
	const auto transform = galp::profiles::rgbnomore_val_dct_grid_transform();
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	for (uint32_t image = 0U; image < source.manifest.shards.front().image_count; ++image) {
		requests.push_back({image, {}, false, {}, {}});
	}
	const auto miss = planner.Plan(requests, transform);
	EXPECT_EQ(miss.stats.canonical_template_hit_count, 0U);
	EXPECT_EQ(miss.stats.canonical_template_miss_count, 1U);
	EXPECT_NE(miss.stats.canonical_template_audit_digest, 0U);

	const auto written = planner.WriteCanonicalShardPlanTemplate(
	    source.manifest.shards.front().shard_id, transform, sidecars);
	EXPECT_FALSE(written.reused_existing);
	EXPECT_GT(written.sidecar_bytes, 0U);
	EXPECT_EQ(written.plan_audit_digest, miss.stats.canonical_template_audit_digest);
	EXPECT_EQ(written.request_count, requests.size());
	EXPECT_EQ(written.rowgroup_count, miss.rowgroups.size());
	EXPECT_EQ(written.vector_run_count, miss.vector_runs.size());

	const auto hit = planner.Plan(requests, transform);
	EXPECT_EQ(hit.stats.canonical_template_hit_count, 1U);
	EXPECT_EQ(hit.stats.canonical_template_miss_count, 0U);
	EXPECT_EQ(hit.stats.canonical_template_audit_digest, miss.stats.canonical_template_audit_digest);
	EXPECT_EQ(hit.stats.canonical_template_sidecar_bytes, written.sidecar_bytes);
	EXPECT_EQ(actual_vectors(hit), actual_vectors(miss));
	EXPECT_EQ(hit.stats.selected_vectors, miss.stats.selected_vectors);
	EXPECT_EQ(hit.group_bindings.size(), miss.group_bindings.size());
	EXPECT_EQ(hit.rank_cells.size(), miss.rank_cells.size());
	EXPECT_EQ(hit.rank_payload, miss.rank_payload);
	EXPECT_EQ(hit.quant_tables.size(), miss.quant_tables.size());

	const auto reused = planner.WriteCanonicalShardPlanTemplate(
	    source.manifest.shards.front().shard_id, transform, sidecars);
	EXPECT_TRUE(reused.reused_existing);
	EXPECT_EQ(reused.sidecar_bytes, written.sidecar_bytes);
	EXPECT_EQ(reused.plan_audit_digest, written.plan_audit_digest);

	auto different_transform = transform;
	different_transform.output_add = 1.0F;
	const auto profile_miss = planner.Plan(requests, different_transform);
	EXPECT_EQ(profile_miss.stats.canonical_template_hit_count, 0U);
	EXPECT_EQ(profile_miss.stats.canonical_template_miss_count, 1U);
	// The transform identity is part of the sidecar cache key, so changing it
	// must miss even when the immutable compact-plan payload is unchanged.
	// The audit digest intentionally describes that payload and must therefore
	// remain stable for this output-only transform change.
	EXPECT_EQ(profile_miss.stats.canonical_template_audit_digest, hit.stats.canonical_template_audit_digest);
}

TEST(JpegDctBlockMajorPlan, ActiveOutputOwnershipIsOneTimeDeterministicAndWorksetMajor) {
	using namespace galp::jpeg;
	using namespace galp::jpeg::detail;
	JpegDctDeviceBlockMajorPlanlessPlan plan;
	const auto grayscale_image = [](const uint32_t request_index,
	                                const uint32_t shard_id,
	                                const uint32_t local_image_index,
	                                const uint32_t source_width,
	                                const uint16_t up,
	                                const uint16_t down) {
		JpegDctDevicePlanlessImageDescriptor image;
		image.request_index     = request_index;
		image.shard_id          = shard_id;
		image.local_image_index = local_image_index;
		auto& y                 = image.components[0];
		y.semantic_slot_id      = 0U;
		y.width_in_blocks       = source_width;
		y.height_in_blocks      = 1U;
		y.crop_width            = source_width;
		y.crop_height           = 1U;
		y.x_up_factor           = up;
		y.x_down_factor         = down;
		y.y_up_factor           = 1U;
		y.y_down_factor         = 1U;
		y.present               = 1U;
		return image;
	};
	// The first two descriptors are duplicate requests for the same source
	// image.  Their down2 outputs are owned by disjoint worksets.  The partial
	// tail descriptor uses a 2/3 rational phase: output 0 spans worksets 0 and
	// 2, while output 1 is wholly owned by workset 2.
	plan.images.push_back(grayscale_image(0U, 0U, 5U, 4U, 1U, 2U));
	plan.images.push_back(grayscale_image(1U, 0U, 5U, 4U, 1U, 2U));
	plan.images.push_back(grayscale_image(2U, 1U, 0U, 3U, 2U, 3U));
	const auto add_group = [&](const uint32_t shard,
	                           const uint32_t x,
	                           const uint32_t rowgroup) {
		plan.groups.push_back(
		    {shard, 0U, x, 0U, rowgroup, 0U, 0U, std::numeric_limits<uint32_t>::max(),
		     std::numeric_limits<uint32_t>::max()});
	};
	add_group(0U, 0U, 0U);
	add_group(0U, 1U, 0U);
	add_group(0U, 2U, 1U);
	add_group(0U, 3U, 1U);
	add_group(1U, 0U, 2U);
	add_group(1U, 1U, 3U);
	add_group(1U, 2U, 4U);
	build_block_major_coordinate_group_lookup(plan);
	for (auto& image : plan.images) {
		for (auto& component : image.components) {
			if (component.present != 0U) {
				component.block_major_coordinate_lookup_index =
				    find_block_major_coordinate_group_lookup(plan, image.shard_id, component.semantic_slot_id);
			}
		}
	}
	std::vector<JpegDctDeviceBlockMajorRowgroupWorkset> worksets {
	    {0U, 0U, 0U}, {0U, 1U, 1U}, {1U, 2U, 0U}, {1U, 3U, 2U}, {1U, 4U, 2U},
	    // Workset 3 is deliberately empty of contributing groups.
	    {1U, 99U, 3U},
	};
	JpegDctGridTransformSpec transform;
	transform.y_output_width_blocks     = 2U;
	transform.y_output_height_blocks    = 1U;
	transform.cbcr_output_width_blocks  = 1U;
	transform.cbcr_output_height_blocks = 1U;
	const auto schedule0 = build_block_major_active_output_schedule(plan, worksets, transform);
	const auto schedule1 = build_block_major_active_output_schedule(plan, worksets, transform);
	const auto reference = reference_active_output_schedule(plan, worksets, transform);
	EXPECT_EQ(schedule0.logical_output_block_count, 12U);
	EXPECT_EQ(schedule0.source_contribution_count, 12U);
	EXPECT_EQ(schedule0.source_contribution_visit_count, 24U);
	EXPECT_EQ(schedule0.offsets, (std::vector<uint64_t> {0U, 3U, 5U, 7U, 7U}));
	EXPECT_EQ(schedule0.output_workset_ownership_count, 7U);
	EXPECT_EQ(schedule0.active_output_blocks,
	          (std::vector<uint32_t> {0U, 4U, 8U, 1U, 5U, 8U, 9U}));
	EXPECT_EQ(schedule0.offsets, reference.offsets);
	EXPECT_EQ(schedule0.active_output_blocks, reference.active_output_blocks);
	EXPECT_EQ(schedule0.source_contribution_count, reference.source_contribution_count);
	EXPECT_EQ(schedule0.offsets, schedule1.offsets);
	EXPECT_EQ(schedule0.active_output_blocks, schedule1.active_output_blocks);
	for (size_t repeat = 0U; repeat < 100U; ++repeat) {
		const auto repeated = build_block_major_active_output_schedule(plan, worksets, transform);
		EXPECT_EQ(repeated.offsets, schedule0.offsets);
		EXPECT_EQ(repeated.active_output_blocks, schedule0.active_output_blocks);
		EXPECT_EQ(repeated.source_contribution_count, schedule0.source_contribution_count);
	}
	for (size_t workset = 0U; workset + 1U < schedule0.offsets.size(); ++workset) {
		const auto begin = schedule0.active_output_blocks.begin() + schedule0.offsets[workset];
		const auto end = schedule0.active_output_blocks.begin() + schedule0.offsets[workset + 1U];
		EXPECT_TRUE(std::is_sorted(begin, end));
		EXPECT_EQ(std::adjacent_find(begin, end), end);
	}
}

TEST(JpegDctBlockMajorPlan, ActiveOutputIntervalSidecarRoundTripsAndBindsEveryDecisionKey) {
	using namespace galp::jpeg;
	using namespace galp::jpeg::detail;
	BlockMajorTemporaryDirectory temporary;
	JpegDctDeviceBlockMajorActiveOutputSchedule schedule;
	schedule.offsets = {0U, 6U, 10U};
	schedule.active_output_blocks = {0U, 1U, 2U, 8U, 9U, 15U, 3U, 4U, 12U, 14U};
	schedule.logical_output_block_count = 16U;
	schedule.source_contribution_count = 20U;
	schedule.source_contribution_visit_count = 40U;
	schedule.output_workset_ownership_count = schedule.active_output_blocks.size();

	JpegDctGridTransformSpec transform;
	transform.y_output_width_blocks = 4U;
	transform.y_output_height_blocks = 2U;
	transform.cbcr_output_width_blocks = 2U;
	transform.cbcr_output_height_blocks = 2U;
	const std::vector<JpegDctActiveOutputDecisionRecord> decisions {
	    {7U, 0U, 0U, 1U, 3U, 1U, 32U, 64U, 4096U, 3U},
	    {7U, 1U, 1U, 1U, 3U, 1U, 30U, 64U, 3900U, 2U},
	};
	auto coefficient_variant = decisions;
	coefficient_variant[0].runtime_decision      = 9U;
	coefficient_variant[0].read_strategy         = 8U;
	coefficient_variant[0].submission_backend    = 7U;
	coefficient_variant[0].selected_vector_count = 16U;
	coefficient_variant[0].physical_bytes        = 2048U;
	coefficient_variant[0].physical_run_count    = 12U;
	EXPECT_EQ(jpeg_dct_active_output_decision_digest(decisions),
	          jpeg_dct_active_output_decision_digest(coefficient_variant));
	const std::vector<JpegDctActiveOutputDecisionRecord> ownership_only {
	    {7U, 0U, 0U},
	    {7U, 1U, 1U},
	};
	EXPECT_EQ(jpeg_dct_active_output_decision_digest(decisions),
	          jpeg_dct_active_output_decision_digest(ownership_only));
	auto ownership_variant = decisions;
	ownership_variant[0].workset_index = 1U;
	EXPECT_NE(jpeg_dct_active_output_decision_digest(decisions),
	          jpeg_dct_active_output_decision_digest(ownership_variant));
	JpegDctActiveOutputScheduleKey key {
	    temporary.path(),
	    7U,
	    0x123456789abcdef0ULL,
	    jpeg_dct_active_output_decision_digest(decisions),
	    jpeg_dct_active_output_transform_digest(transform),
	    512U << 20U,
	    64U,
	    1U,
	    2U,
	    kJpegDctActiveOutputSchedulePlannerAbi};

	const auto persisted = persist_jpeg_dct_active_output_schedule(key, schedule);
	ASSERT_TRUE(persisted.persisted) << persisted.rejection_reason;
	EXPECT_FALSE(persisted.rejected);
	EXPECT_GT(persisted.interval_count, 0U);
	EXPECT_TRUE(std::filesystem::is_regular_file(jpeg_dct_active_output_schedule_path(key)));

	auto loaded = load_jpeg_dct_active_output_schedule(key);
	ASSERT_TRUE(loaded.hit) << loaded.rejection_reason;
	ASSERT_TRUE(loaded.schedule.has_value());
	EXPECT_EQ(loaded.schedule->offsets, schedule.offsets);
	EXPECT_EQ(loaded.schedule->active_output_blocks, schedule.active_output_blocks);
	EXPECT_EQ(loaded.schedule->logical_output_block_count, schedule.logical_output_block_count);
	EXPECT_EQ(loaded.schedule->source_contribution_count, schedule.source_contribution_count);
	EXPECT_EQ(loaded.schedule->source_contribution_visit_count, schedule.source_contribution_visit_count);
	EXPECT_EQ(loaded.schedule->output_workset_ownership_count, schedule.output_workset_ownership_count);
	EXPECT_TRUE(loaded.schedule->sidecar_mapping);
	EXPECT_LE(loaded.mapped_bytes, kJpegDctActiveOutputScheduleMmapCapacityBytes);

	auto mismatched = key;
	mismatched.decision_digest ^= 1U;
	const auto rejected = load_jpeg_dct_active_output_schedule(mismatched);
	EXPECT_FALSE(rejected.hit);
	EXPECT_TRUE(rejected.rejected);
}

TEST(JpegDctBlockMajorPlan, CoordinateGroupLookupUsesPerShardSlotBoundingBoxesAndPreservesHoles) {
	using namespace galp::jpeg::detail;
	JpegDctDeviceBlockMajorPlanlessPlan plan;
	const auto add_group = [&](const uint32_t shard,
	                           const uint32_t slot,
	                           const uint32_t x,
	                           const uint32_t y,
	                           const uint32_t rowgroup) {
		plan.groups.push_back({shard,
		                       slot,
		                       x,
		                       y,
		                       rowgroup,
		                       0U,
		                       0U,
		                       std::numeric_limits<uint32_t>::max(),
		                       std::numeric_limits<uint32_t>::max()});
	};
	add_group(0U, 0U, 10U, 20U, 0U);
	add_group(0U, 0U, 12U, 20U, 1U);
	add_group(0U, 0U, 10U, 21U, 2U);
	add_group(0U, 0U, 12U, 21U, 3U);
	add_group(0U, 1U, 3U, 4U, 4U);
	add_group(1U, 0U, 10U, 20U, 5U);
	build_block_major_coordinate_group_lookup(plan);

	ASSERT_EQ(plan.coordinate_group_lookups.size(), 3U);
	const auto lookup0_index = find_block_major_coordinate_group_lookup(plan, 0U, 0U);
	const auto lookup1_index = find_block_major_coordinate_group_lookup(plan, 0U, 1U);
	const auto lookup2_index = find_block_major_coordinate_group_lookup(plan, 1U, 0U);
	ASSERT_EQ(lookup0_index, 0U);
	ASSERT_EQ(lookup1_index, 1U);
	ASSERT_EQ(lookup2_index, 2U);
	EXPECT_EQ(find_block_major_coordinate_group_lookup(plan, 1U, 1U),
	          std::numeric_limits<uint32_t>::max());
	const auto& lookup0 = plan.coordinate_group_lookups[lookup0_index];
	EXPECT_EQ(lookup0.origin_x, 10U);
	EXPECT_EQ(lookup0.origin_y, 20U);
	EXPECT_EQ(lookup0.width, 3U);
	EXPECT_EQ(lookup0.height, 2U);
	EXPECT_EQ(lookup0.stride, 3U);
	EXPECT_EQ(lookup0.populated_group_count, 4U);
	ASSERT_LE(lookup0.group_index_base + 6U, plan.coordinate_group_indices.size());
	const auto base = static_cast<size_t>(lookup0.group_index_base);
	EXPECT_EQ(plan.coordinate_group_indices[base + 0U], 0U);
	EXPECT_EQ(plan.coordinate_group_indices[base + 1U], kInvalidJpegDctBlockMajorGroupIndex);
	EXPECT_EQ(plan.coordinate_group_indices[base + 2U], 1U);
	EXPECT_EQ(plan.coordinate_group_indices[base + 3U], 2U);
	EXPECT_EQ(plan.coordinate_group_indices[base + 4U], kInvalidJpegDctBlockMajorGroupIndex);
	EXPECT_EQ(plan.coordinate_group_indices[base + 5U], 3U);
	EXPECT_EQ(plan.coordinate_group_indices.size(), 8U);
}

TEST(JpegDctBlockMajorPlan, CoordinateGroupLookupRejectsInvalidTopology) {
	using namespace galp::jpeg::detail;
	const auto group = [](const uint32_t x, const uint32_t y, const uint32_t rowgroup) {
		return JpegDctDeviceBlockMajorGroupBinding {
		    0U,
		    0U,
		    x,
		    y,
		    rowgroup,
		    0U,
		    0U,
		    std::numeric_limits<uint32_t>::max(),
		    std::numeric_limits<uint32_t>::max()};
	};
	JpegDctDeviceBlockMajorPlanlessPlan duplicate;
	duplicate.groups = {group(0U, 0U, 0U), group(0U, 0U, 1U)};
	EXPECT_THROW(build_block_major_coordinate_group_lookup(duplicate), std::runtime_error);

	JpegDctDeviceBlockMajorPlanlessPlan unsorted;
	unsorted.groups = {group(1U, 0U, 0U), group(0U, 0U, 1U)};
	EXPECT_THROW(build_block_major_coordinate_group_lookup(unsorted), std::runtime_error);

	JpegDctDeviceBlockMajorPlanlessPlan overflowing;
	overflowing.groups = {group(0U, 0U, 0U), group(std::numeric_limits<uint32_t>::max(), 0U, 1U)};
	EXPECT_THROW(build_block_major_coordinate_group_lookup(overflowing), std::runtime_error);
}

TEST(JpegDctBlockMajorPlan, ActiveOutputScheduleValidatesLookupBoundsHolesAndGroupIndices) {
	using namespace galp::jpeg;
	using namespace galp::jpeg::detail;
	JpegDctDeviceBlockMajorPlanlessPlan plan;
	JpegDctDevicePlanlessImageDescriptor image;
	image.shard_id = 0U;
	auto& y = image.components[0];
	y.semantic_slot_id = 0U;
	y.width_in_blocks = 2U;
	y.height_in_blocks = 1U;
	y.crop_width = 2U;
	y.crop_height = 1U;
	y.x_up_factor = 1U;
	y.y_up_factor = 1U;
	y.x_down_factor = 1U;
	y.y_down_factor = 1U;
	y.present = 1U;
	plan.images.push_back(image);
	plan.groups.push_back({0U,
	                       0U,
	                       0U,
	                       0U,
	                       0U,
	                       0U,
	                       0U,
	                       std::numeric_limits<uint32_t>::max(),
	                       std::numeric_limits<uint32_t>::max()});
	plan.groups.push_back({0U,
	                       0U,
	                       1U,
	                       0U,
	                       1U,
	                       0U,
	                       0U,
	                       std::numeric_limits<uint32_t>::max(),
	                       std::numeric_limits<uint32_t>::max()});
	build_block_major_coordinate_group_lookup(plan);
	plan.images[0].components[0].block_major_coordinate_lookup_index =
	    find_block_major_coordinate_group_lookup(plan, 0U, 0U);
	const std::vector<JpegDctDeviceBlockMajorRowgroupWorkset> worksets {{0U, 0U, 0U}, {0U, 1U, 1U}};
	JpegDctGridTransformSpec transform;
	transform.y_output_width_blocks = 2U;
	transform.y_output_height_blocks = 1U;
	ASSERT_NO_THROW(static_cast<void>(build_block_major_active_output_schedule(plan, worksets, transform)));

	auto missing_lookup = plan;
	missing_lookup.images[0].components[0].block_major_coordinate_lookup_index =
	    std::numeric_limits<uint32_t>::max();
	EXPECT_THROW(static_cast<void>(
	                 build_block_major_active_output_schedule(missing_lookup, worksets, transform)),
	             std::runtime_error);

	auto invalid_base = plan;
	invalid_base.coordinate_group_lookups[0].group_index_base =
	    invalid_base.coordinate_group_indices.size();
	EXPECT_THROW(static_cast<void>(
	                 build_block_major_active_output_schedule(invalid_base, worksets, transform)),
	             std::runtime_error);

	auto hole = plan;
	hole.coordinate_group_indices[0] = kInvalidJpegDctBlockMajorGroupIndex;
	EXPECT_THROW(static_cast<void>(build_block_major_active_output_schedule(hole, worksets, transform)),
	             std::runtime_error);

	auto out_of_range_group = plan;
	out_of_range_group.coordinate_group_indices[0] = static_cast<uint32_t>(out_of_range_group.groups.size());
	EXPECT_THROW(static_cast<void>(
	                 build_block_major_active_output_schedule(out_of_range_group, worksets, transform)),
	             std::runtime_error);

}

TEST(JpegDctBlockMajorPlan, ActiveOutputScheduleSkipsNegativeCropCoordinatesDeterministically) {
	using namespace galp::jpeg;
	using namespace galp::jpeg::detail;
	JpegDctDeviceBlockMajorPlanlessPlan plan;
	JpegDctDevicePlanlessImageDescriptor image;
	image.shard_id = 0U;
	auto& y = image.components[0];
	y.semantic_slot_id = 0U;
	y.width_in_blocks = 1U;
	y.height_in_blocks = 1U;
	y.crop_x = -1;
	y.crop_width = 2U;
	y.crop_height = 1U;
	y.x_up_factor = 1U;
	y.y_up_factor = 1U;
	y.x_down_factor = 1U;
	y.y_down_factor = 1U;
	y.present = 1U;
	plan.images.push_back(image);
	plan.groups.push_back({0U,
	                       0U,
	                       0U,
	                       0U,
	                       0U,
	                       0U,
	                       0U,
	                       std::numeric_limits<uint32_t>::max(),
	                       std::numeric_limits<uint32_t>::max()});
	build_block_major_coordinate_group_lookup(plan);
	plan.images[0].components[0].block_major_coordinate_lookup_index = 0U;
	const std::vector<JpegDctDeviceBlockMajorRowgroupWorkset> worksets {{0U, 0U, 0U}};
	JpegDctGridTransformSpec transform;
	transform.y_output_width_blocks = 2U;
	transform.y_output_height_blocks = 1U;
	const auto schedule = build_block_major_active_output_schedule(plan, worksets, transform);
	EXPECT_EQ(schedule.source_contribution_count, 1U);
	EXPECT_EQ(schedule.source_contribution_visit_count, 2U);
	EXPECT_EQ(schedule.offsets, (std::vector<uint64_t> {0U, 1U}));
	EXPECT_EQ(schedule.active_output_blocks, (std::vector<uint32_t> {1U}));
}

TEST(JpegDctBlockMajorPlan, DISABLED_RepresentativeActiveOutputScheduleMicrobenchmark) {
	using namespace galp::jpeg;
	using namespace galp::jpeg::detail;
	constexpr uint32_t image_count = 1000U;
	constexpr uint32_t workset_count = 16U;
	constexpr std::array<uint32_t, 3> widths {256U, 224U, 224U};
	constexpr std::array<uint32_t, 3> heights {256U, 224U, 224U};
	constexpr std::array<uint32_t, 3> crop_widths {56U, 28U, 28U};
	constexpr std::array<uint32_t, 3> crop_heights {56U, 28U, 28U};
	constexpr uint32_t rowgroup_count = 253U;
	constexpr uint32_t group_count =
	    widths[0] * heights[0] + widths[1] * heights[1] + widths[2] * heights[2];
	JpegDctDeviceBlockMajorPlanlessPlan plan;
	for (uint32_t slot = 0U; slot < widths.size(); ++slot) {
		for (uint32_t y = 0U; y < heights[slot]; ++y) {
			for (uint32_t x = 0U; x < widths[slot]; ++x) {
				const auto group_index = static_cast<uint32_t>(plan.groups.size());
				plan.groups.push_back({0U,
				                       slot,
				                       x,
				                       y,
				                       group_index * rowgroup_count / group_count,
				                       0U,
				                       0U,
				                       std::numeric_limits<uint32_t>::max(),
				                       std::numeric_limits<uint32_t>::max()});
			}
		}
	}
	build_block_major_coordinate_group_lookup(plan);
	ASSERT_EQ(plan.groups.size(), group_count);
	plan.images.reserve(image_count);
	for (uint32_t image_index = 0U; image_index < image_count; ++image_index) {
		JpegDctDevicePlanlessImageDescriptor image;
		image.shard_id = 0U;
		for (uint32_t slot = 0U; slot < image.components.size(); ++slot) {
			auto& component = image.components[slot];
			component.semantic_slot_id = slot;
			component.block_major_coordinate_lookup_index =
			    find_block_major_coordinate_group_lookup(plan, 0U, slot);
			component.width_in_blocks = widths[slot];
			component.height_in_blocks = heights[slot];
			component.crop_x = static_cast<int32_t>(
			    (static_cast<uint64_t>(image_index) * (17U + 6U * slot)) %
			    (widths[slot] - crop_widths[slot] + 1U));
			component.crop_y = static_cast<int32_t>(
			    (static_cast<uint64_t>(image_index) * (31U + 6U * slot)) %
			    (heights[slot] - crop_heights[slot] + 1U));
			component.crop_width = crop_widths[slot];
			component.crop_height = crop_heights[slot];
			component.x_up_factor = 1U;
			component.y_up_factor = 1U;
			component.x_down_factor = 2U;
			component.y_down_factor = 2U;
			component.present = 1U;
		}
		plan.images.push_back(image);
	}
	std::vector<JpegDctDeviceBlockMajorRowgroupWorkset> worksets;
	worksets.reserve(rowgroup_count);
	for (uint32_t rowgroup = 0U; rowgroup < rowgroup_count; ++rowgroup) {
		worksets.push_back({0U, rowgroup, rowgroup * workset_count / rowgroup_count});
	}
	JpegDctGridTransformSpec transform;
	transform.y_output_width_blocks = 28U;
	transform.y_output_height_blocks = 28U;
	transform.cbcr_output_width_blocks = 14U;
	transform.cbcr_output_height_blocks = 14U;
	const auto warmup = build_block_major_active_output_schedule(plan, worksets, transform);
	const auto reference_warmup = reference_active_output_schedule(plan, worksets, transform);
	ASSERT_EQ(warmup.logical_output_block_count, 1176000U);
	ASSERT_EQ(warmup.source_contribution_count, 4704000U);
	ASSERT_EQ(warmup.source_contribution_visit_count, 9408000U);
	ASSERT_LT(warmup.temporary_bytes_peak, 1U << 20U);
	ASSERT_EQ(warmup.offsets, reference_warmup.offsets);
	ASSERT_EQ(warmup.active_output_blocks, reference_warmup.active_output_blocks);

	std::vector<double> samples_ms;
	std::vector<double> reference_samples_ms;
	samples_ms.reserve(10U);
	reference_samples_ms.reserve(10U);
	const auto run_new = [&]() {
		const auto schedule = build_block_major_active_output_schedule(plan, worksets, transform);
		EXPECT_EQ(schedule.offsets, warmup.offsets);
		EXPECT_EQ(schedule.active_output_blocks, warmup.active_output_blocks);
		samples_ms.push_back(schedule.total_build_ms);
	};
	const auto run_reference = [&]() {
		const auto started = std::chrono::steady_clock::now();
		const auto schedule = reference_active_output_schedule(plan, worksets, transform);
		reference_samples_ms.push_back(std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - started).count());
		EXPECT_EQ(schedule.offsets, warmup.offsets);
		EXPECT_EQ(schedule.active_output_blocks, warmup.active_output_blocks);
	};
	for (size_t repeat = 0U; repeat < 10U; ++repeat) {
		if (repeat % 2U == 0U) {
			run_reference();
			run_new();
		} else {
			run_new();
			run_reference();
		}
	}
	std::sort(samples_ms.begin(), samples_ms.end());
	std::sort(reference_samples_ms.begin(), reference_samples_ms.end());
	const auto median_ms = (samples_ms[4] + samples_ms[5]) / 2.0;
	const auto p95_ms = samples_ms.back();
	const auto mean_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / samples_ms.size();
	const auto reference_median_ms = (reference_samples_ms[4] + reference_samples_ms[5]) / 2.0;
	const auto reference_p95_ms = reference_samples_ms.back();
	const auto reference_mean_ms =
	    std::accumulate(reference_samples_ms.begin(), reference_samples_ms.end(), 0.0) /
	    reference_samples_ms.size();
	double squared_error = 0.0;
	for (const auto sample : samples_ms) {
		squared_error += (sample - mean_ms) * (sample - mean_ms);
	}
	double reference_squared_error = 0.0;
	for (const auto sample : reference_samples_ms) {
		reference_squared_error += (sample - reference_mean_ms) * (sample - reference_mean_ms);
	}
	const auto cv = mean_ms == 0.0 ? 0.0 : std::sqrt(squared_error / samples_ms.size()) / mean_ms;
	const auto reference_cv = reference_mean_ms == 0.0
	                              ? 0.0
	                              : std::sqrt(reference_squared_error / reference_samples_ms.size()) /
	                                    reference_mean_ms;
	RecordProperty("source_contributions", warmup.source_contribution_count);
	RecordProperty("source_contribution_visits", warmup.source_contribution_visit_count);
	RecordProperty("ownership_count", warmup.output_workset_ownership_count);
	RecordProperty("temporary_bytes_peak", warmup.temporary_bytes_peak);
	RecordProperty("median_ms", median_ms);
	RecordProperty("p95_ms", p95_ms);
	RecordProperty("cv", cv);
	RecordProperty("reference_median_ms", reference_median_ms);
	RecordProperty("reference_p95_ms", reference_p95_ms);
	RecordProperty("reference_cv", reference_cv);
	RecordProperty("median_reduction_fraction", 1.0 - median_ms / reference_median_ms);
	RecordProperty("temporary_bytes_reduction_fraction",
	               1.0 - static_cast<double>(warmup.temporary_bytes_peak) /
	                         reference_warmup.temporary_bytes_peak);
}

TEST(JpegDctBlockMajorPlan, ProductionPreviewSelectsCompactPlanlessPath) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_two_shard_synthetic_dataset(temporary.path());
	const auto sidecars = temporary.path() / "block_major_access_v1";
	galp::jpeg::build_jpeg_dct_block_major_access_dataset(source.manifest_path, sidecars);
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	for (const auto image : {0U, 1U, 2U, 3U, 4U, 9U, 10U, 12U, 15U}) {
		requests.push_back({image, {}, false, std::to_string(image), "production"});
	}
	galp::jpeg::JpegDctBlockMajorCompactPlanner compact_planner(source.manifest_path, sidecars);
	const auto compact =
	    compact_planner.Plan(requests, galp::profiles::rgbnomore_val_dct_grid_transform());
	galp::jpeg::JpegDctShardDatasetReader reader(source.manifest_path);
	const auto cold_reader_stats = reader.InitializationStats();
	EXPECT_TRUE(cold_reader_stats.block_major_metadata_lazy);
	EXPECT_EQ(cold_reader_stats.eagerly_loaded_shard_metadata_count, 0U);
	EXPECT_EQ(cold_reader_stats.loaded_shard_metadata_count, 0U);
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform            = galp::profiles::rgbnomore_val_dct_grid_transform();
	options.enable_planless_execution = true;
	options.enable_rowgroup_prefetch  = false;
	options.decode_batch_rowgroups    = 1U;
	options.decode_workset_capacity_bytes = 1U;
	options.crop_execution_mode = galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
	const auto preview = reader.PlanDeviceDctBatch(requests, options);
	const auto planned_reader_stats = reader.InitializationStats();
	EXPECT_EQ(planned_reader_stats.loaded_shard_metadata_count, 2U);
	EXPECT_EQ(planned_reader_stats.block_major_loaded_descriptor_count, 2U);
	EXPECT_TRUE(preview.uses_planless_fixed_transform);
	EXPECT_EQ(preview.host_expanded_transform_items_created, 0U);
	EXPECT_EQ(preview.host_output_block_source_lists_created, 0U);
	EXPECT_EQ(preview.host_global_transform_sort_items, 0U);
	EXPECT_EQ(preview.compact_image_descriptor_count, requests.size());
	EXPECT_GT(preview.coordinate_group_lookup_count, 0U);
	EXPECT_GE(preview.coordinate_group_index_entries, preview.coordinate_group_index_populated);
	EXPECT_EQ(preview.coordinate_group_index_populated, compact.group_bindings.size());
	EXPECT_EQ(preview.coordinate_group_index_holes,
	          preview.coordinate_group_index_entries - preview.coordinate_group_index_populated);
	EXPECT_GT(preview.coordinate_group_index_bytes, 0U);
	EXPECT_GT(preview.coordinate_group_index_density, 0.0);
	EXPECT_LE(preview.coordinate_group_index_density, 1.0);
	EXPECT_EQ(preview.decode_workset_capacity_bytes, 1U);
	EXPECT_GT(preview.estimated_max_decode_workset_bytes, preview.decode_workset_capacity_bytes);
	EXPECT_EQ(preview.estimated_oversized_decode_rowgroups, preview.rowgroups.size());
	EXPECT_EQ(preview_vectors(preview), actual_vectors(compact));
	EXPECT_FALSE(options.grid_transform->require_all_coefficients);
	for (const auto& coefficients : std::vector<std::vector<uint8_t>> {
	         {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U},
	         {0U, 3U, 7U}}) {
		auto selected_options = options;
		selected_options.coefficient_selection.coefficients = coefficients;
		const auto selected_preview = reader.PlanDeviceDctBatch(requests, selected_options);
		EXPECT_TRUE(selected_preview.uses_planless_fixed_transform);
		EXPECT_EQ(selected_preview.selected_coefficients, coefficients);
		EXPECT_EQ(selected_preview.coefficients_per_block, coefficients.size());
	}
	auto guarded_options = options;
	guarded_options.grid_transform->require_all_coefficients = true;
	guarded_options.coefficient_selection.coefficients       = {0U, 1U, 2U};
	EXPECT_THROW((void)reader.PlanDeviceDctBatch(requests, guarded_options), std::runtime_error);
	auto legacy_options = options;
	legacy_options.enable_planless_execution = false;
	const auto legacy_preview = reader.PlanDeviceDctBatch(requests, legacy_options);
	EXPECT_FALSE(legacy_preview.uses_planless_fixed_transform);
	EXPECT_GT(legacy_preview.host_expanded_transform_items_created, 0U);
}

TEST(JpegDctBlockMajorPlan, MissingSidecarSelectsExplicitLegacyFallbackBoundary) {
	BlockMajorTemporaryDirectory temporary;
	const auto source = make_synthetic_dataset(temporary.path());
	EXPECT_THROW(galp::jpeg::JpegDctBlockMajorCompactPlanner(source.manifest_path, temporary.path() / "missing"),
	             std::runtime_error);
}

} // namespace
