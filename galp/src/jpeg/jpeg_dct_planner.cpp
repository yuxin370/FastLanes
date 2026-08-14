#include "fls/cfg/cfg.hpp"
#include "galp/jpeg_dct_block_major_plan.hpp"
#include "galp/jpeg_dct_diagnostics.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include "jpeg/jpeg_dct_device_bridge.hpp"
#include "jpeg/jpeg_dct_metadata.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include "jpeg/jpeg_dct_policy.hpp"
#include "jpeg/jpeg_dct_prepared_plan.hpp"
#include "jpeg/jpeg_dct_shard_reader.hpp"
#include "format/compact_descriptor_v3.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/io/file.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace galp::jpeg {
namespace {

// Match the float32 torch.mm conversion used by the CUDA reference-down2
// path. Tiny non-zero entries are observable at round-to-even boundaries.
constexpr float kRgbNoMoreDown2Conversion[] = {
#include "jpeg/jpeg_dct_reference_down2.inc"
};
static_assert(sizeof(kRgbNoMoreDown2Conversion) / sizeof(kRgbNoMoreDown2Conversion[0]) == 8U * 16U);

std::filesystem::path resolve_manifest_member(const std::filesystem::path& root,
	                                          const std::string&           member,
	                                          const std::string_view       label) {
	const std::filesystem::path relative(member);
	if (relative.empty() || relative.is_absolute() ||
	    std::any_of(relative.begin(), relative.end(), [](const auto& component) { return component == ".."; })) {
		throw std::runtime_error("JPEG DCT manifest contains an unsafe " + std::string(label) + " path");
	}
	const auto dataset_root   = root.empty() ? std::filesystem::current_path() : root;
	const auto canonical_root = std::filesystem::weakly_canonical(dataset_root);
	const auto candidate      = std::filesystem::weakly_canonical(dataset_root / relative);
	auto root_it = canonical_root.begin();
	auto path_it = candidate.begin();
	for (; root_it != canonical_root.end() && path_it != candidate.end(); ++root_it, ++path_it) {
		if (*root_it != *path_it) {
			throw std::runtime_error("JPEG DCT manifest " + std::string(label) + " escapes the dataset root");
		}
	}
	if (root_it != canonical_root.end()) {
		throw std::runtime_error("JPEG DCT manifest " + std::string(label) + " escapes the dataset root");
	}
	return candidate;
}

bool uses_bounded_io_uring(const JpegDctCropExecutionMode mode) noexcept {
	return mode == JpegDctCropExecutionMode::kBoundedIoUringRangeReadSelectedDecode ||
	       mode == JpegDctCropExecutionMode::kBoundedIoUringScheduledRangeReadSelectedDecode;
}

void configure_crop_execution_mode(detail::JpegDctDeviceRowgroupPlan& rowgroup,
                                   const JpegDctCropExecutionMode      mode) {
	const auto automatic = detail::choose_jpeg_dct_runtime_policy(
	    rowgroup.selected_vector_count, rowgroup.full_vector_count, rowgroup.selected_chunks_fit);
	const bool selected_decode_possible = rowgroup.selected_chunks_fit && rowgroup.full_vector_count != 0U &&
	                                      rowgroup.selected_vector_count < rowgroup.full_vector_count;
	rowgroup.sparse_storage_read                 = false;
	rowgroup.automatic_sparse_storage_candidate = false;
	switch (mode) {
	case JpegDctCropExecutionMode::kAutomatic:
		// The logical preview is decided solely from information available to the
		// planner.  Keep physical sparse-I/O feasibility separate: the compiled-I/O
		// phase may refine this decision once it knows byte and pread costs, but a
		// preview must never publish that future candidate as an already-selected
		// decode policy.
		rowgroup.runtime_policy = automatic;
		rowgroup.automatic_sparse_storage_candidate = selected_decode_possible;
		rowgroup.read_strategy = automatic.decision == detail::JpegDctRuntimePolicyDecision::kSelectedVectors
		                             ? detail::JpegDctReadStrategy::kBitmapExact
		                             : detail::JpegDctReadStrategy::kFullRowgroup;
		break;
	case JpegDctCropExecutionMode::kFullRowgroupDecode:
		rowgroup.runtime_policy = {detail::JpegDctRuntimePolicyDecision::kFullRowgroup,
		                           detail::JpegDctRuntimePolicyReason::kForcedFullRowgroup};
		rowgroup.read_strategy = detail::JpegDctReadStrategy::kFullRowgroup;
		break;
	case JpegDctCropExecutionMode::kRowgroupReadSelectedDecode:
	case JpegDctCropExecutionMode::kVectorRangeReadSelectedDecode:
	case JpegDctCropExecutionMode::kBoundedRangeReadSelectedDecode:
	case JpegDctCropExecutionMode::kBoundedIoUringRangeReadSelectedDecode:
	case JpegDctCropExecutionMode::kBoundedIoUringScheduledRangeReadSelectedDecode:
		if (!selected_decode_possible) {
			rowgroup.runtime_policy = automatic;
			rowgroup.read_strategy = automatic.decision == detail::JpegDctRuntimePolicyDecision::kFullRowgroup
			                             ? detail::JpegDctReadStrategy::kFullRowgroup
			                             : detail::JpegDctReadStrategy::kBitmapExact;
			break;
		}
		rowgroup.runtime_policy = {detail::JpegDctRuntimePolicyDecision::kSelectedVectors,
		                           detail::JpegDctRuntimePolicyReason::kForcedSelectedVectors};
		rowgroup.sparse_storage_read = mode == JpegDctCropExecutionMode::kVectorRangeReadSelectedDecode ||
		                               mode == JpegDctCropExecutionMode::kBoundedRangeReadSelectedDecode ||
		                               uses_bounded_io_uring(mode);
		rowgroup.read_strategy =
		    mode == JpegDctCropExecutionMode::kBoundedRangeReadSelectedDecode ||
		            uses_bounded_io_uring(mode)
		        ? detail::JpegDctReadStrategy::kRunIntervalBounded
		        : (rowgroup.sparse_storage_read ? detail::JpegDctReadStrategy::kRunIntervalExact
		                                        : detail::JpegDctReadStrategy::kBitmapExact);
		break;
	}
}

} // namespace

void detail::build_block_major_coordinate_group_lookup(JpegDctDeviceBlockMajorPlanlessPlan& plan) {
	plan.coordinate_group_lookups.clear();
	plan.coordinate_group_indices.clear();
	if (plan.groups.empty()) {
		return;
	}
	if (plan.groups.size() > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("block-major coordinate lookup group count exceeds uint32 range");
	}
	const auto group_less = [](const auto& lhs, const auto& rhs) {
		return std::tie(lhs.shard_id, lhs.semantic_slot_id, lhs.block_y, lhs.block_x) <
		       std::tie(rhs.shard_id, rhs.semantic_slot_id, rhs.block_y, rhs.block_x);
	};
	if (!std::is_sorted(plan.groups.begin(), plan.groups.end(), group_less)) {
		throw std::runtime_error("block-major compact group bindings are not in device lookup order");
	}

	for (size_t group_begin = 0U; group_begin < plan.groups.size();) {
		const auto shard_id         = plan.groups[group_begin].shard_id;
		const auto semantic_slot_id = plan.groups[group_begin].semantic_slot_id;
		size_t group_end = group_begin;
		uint32_t min_x = std::numeric_limits<uint32_t>::max();
		uint32_t min_y = std::numeric_limits<uint32_t>::max();
		uint32_t max_x = 0U;
		uint32_t max_y = 0U;
		while (group_end < plan.groups.size() && plan.groups[group_end].shard_id == shard_id &&
		       plan.groups[group_end].semantic_slot_id == semantic_slot_id) {
			const auto& group = plan.groups[group_end];
			min_x = std::min(min_x, group.block_x);
			min_y = std::min(min_y, group.block_y);
			max_x = std::max(max_x, group.block_x);
			max_y = std::max(max_y, group.block_y);
			++group_end;
		}
		const uint64_t width  = static_cast<uint64_t>(max_x) - min_x + 1U;
		const uint64_t height = static_cast<uint64_t>(max_y) - min_y + 1U;
		if (width > std::numeric_limits<uint32_t>::max() || height > std::numeric_limits<uint32_t>::max() ||
		    (height != 0U && width > std::numeric_limits<uint64_t>::max() / height)) {
			throw std::runtime_error("block-major coordinate lookup extent overflow");
		}
		const auto area = width * height;
		if (area > plan.coordinate_group_indices.max_size() - plan.coordinate_group_indices.size()) {
			throw std::runtime_error("block-major coordinate lookup exceeds host address range");
		}
		const auto base = plan.coordinate_group_indices.size();
		plan.coordinate_group_indices.resize(
		    base + static_cast<size_t>(area), kInvalidJpegDctBlockMajorGroupIndex);
		for (size_t group_index = group_begin; group_index < group_end; ++group_index) {
			const auto& group = plan.groups[group_index];
			const auto local = (static_cast<uint64_t>(group.block_y) - min_y) * width +
			                   (static_cast<uint64_t>(group.block_x) - min_x);
			const auto entry = base + static_cast<size_t>(local);
			if (entry >= plan.coordinate_group_indices.size() ||
			    plan.coordinate_group_indices[entry] != kInvalidJpegDctBlockMajorGroupIndex) {
				throw std::runtime_error("block-major coordinate lookup contains a duplicate coordinate");
			}
			plan.coordinate_group_indices[entry] = static_cast<uint32_t>(group_index);
		}
		const auto populated = group_end - group_begin;
		if (populated > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("block-major coordinate lookup descriptor exceeds runtime range");
		}
		plan.coordinate_group_lookups.push_back({shard_id,
		                                         semantic_slot_id,
		                                         min_x,
		                                         min_y,
		                                         static_cast<uint32_t>(width),
		                                         static_cast<uint32_t>(height),
		                                         static_cast<uint32_t>(width),
		                                         static_cast<uint64_t>(base),
		                                         static_cast<uint32_t>(populated)});
		uint32_t verified_populated = 0U;
		for (uint32_t local_y = 0U; local_y < height; ++local_y) {
			for (uint32_t local_x = 0U; local_x < width; ++local_x) {
				const auto group_index = plan.coordinate_group_indices[
				    base + static_cast<size_t>(local_y) * static_cast<size_t>(width) + local_x];
				if (group_index == kInvalidJpegDctBlockMajorGroupIndex) {
					continue;
				}
				if (group_index >= plan.groups.size()) {
					throw std::runtime_error("block-major coordinate lookup group index is out of range");
				}
				const auto& group = plan.groups[group_index];
				if (group.shard_id != shard_id || group.semantic_slot_id != semantic_slot_id ||
				    group.block_x != min_x + local_x || group.block_y != min_y + local_y) {
					throw std::runtime_error("block-major coordinate lookup resolved the wrong group");
				}
				++verified_populated;
			}
		}
		if (verified_populated != populated) {
			throw std::runtime_error("block-major coordinate lookup population is inconsistent");
		}
		group_begin = group_end;
	}
}

uint32_t detail::find_block_major_coordinate_group_lookup(
    const JpegDctDeviceBlockMajorPlanlessPlan& plan,
    const uint32_t                             shard_id,
    const uint32_t                             semantic_slot_id) {
	const auto found = std::lower_bound(
	    plan.coordinate_group_lookups.begin(),
	    plan.coordinate_group_lookups.end(),
	    std::pair {shard_id, semantic_slot_id},
	    [](const auto& lookup, const auto& key) {
		    return std::pair {lookup.shard_id, lookup.semantic_slot_id} < key;
	    });
	if (found == plan.coordinate_group_lookups.end() || found->shard_id != shard_id ||
	    found->semantic_slot_id != semantic_slot_id) {
		return std::numeric_limits<uint32_t>::max();
	}
	return static_cast<uint32_t>(found - plan.coordinate_group_lookups.begin());
}

detail::JpegDctDeviceBlockMajorActiveOutputSchedule detail::build_block_major_active_output_schedule(
    const JpegDctDeviceBlockMajorPlanlessPlan&                 plan,
    const std::vector<JpegDctDeviceBlockMajorRowgroupWorkset>& rowgroup_worksets,
    const JpegDctGridTransformSpec&                            transform) {
	using Clock = std::chrono::steady_clock;
	const auto total_start = Clock::now();
	JpegDctDeviceBlockMajorActiveOutputSchedule schedule;
	if (plan.images.empty()) {
		schedule.offsets.push_back(0U);
		schedule.total_build_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
		return schedule;
	}
	if (rowgroup_worksets.empty()) {
		throw std::runtime_error("block-major active-output schedule has images but no worksets");
	}

	const auto y_blocks = static_cast<uint64_t>(transform.y_output_width_blocks) *
	                      transform.y_output_height_blocks;
	const auto cbcr_channel_blocks = static_cast<uint64_t>(transform.cbcr_output_width_blocks) *
	                                 transform.cbcr_output_height_blocks;
	const auto blocks_per_image = y_blocks + 2U * cbcr_channel_blocks;
	if (blocks_per_image == 0U || plan.images.size() > std::numeric_limits<uint32_t>::max() / blocks_per_image) {
		throw std::runtime_error("block-major active-output logical grid exceeds uint32 range");
	}
	schedule.logical_output_block_count = blocks_per_image * plan.images.size();

	const auto group_workset_start = Clock::now();
	uint32_t workset_count = 0U;
	for (const auto& binding : rowgroup_worksets) {
		if (binding.workset_index == std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("block-major active-output workset index exceeds runtime range");
		}
		workset_count = std::max(workset_count, binding.workset_index + 1U);
	}
	std::vector<uint8_t> workset_seen(workset_count, 0U);
	std::unordered_map<uint64_t, uint32_t> rowgroup_to_workset;
	rowgroup_to_workset.reserve(rowgroup_worksets.size());
	for (const auto& binding : rowgroup_worksets) {
		const auto rowgroup_key = (static_cast<uint64_t>(binding.shard_id) << 32U) | binding.rowgroup_index;
		if (!rowgroup_to_workset.emplace(rowgroup_key, binding.workset_index).second) {
			throw std::runtime_error("block-major rowgroup belongs to multiple active-output worksets");
		}
		workset_seen[binding.workset_index] = 1U;
	}
	if (std::find(workset_seen.begin(), workset_seen.end(), 0U) != workset_seen.end()) {
		throw std::runtime_error("block-major active-output workset indices are not dense");
	}
	std::vector<uint32_t> group_workset(plan.groups.size(), std::numeric_limits<uint32_t>::max());
	for (size_t group_index = 0U; group_index < plan.groups.size(); ++group_index) {
		const auto& group = plan.groups[group_index];
		const auto rowgroup_key = (static_cast<uint64_t>(group.shard_id) << 32U) | group.rowgroup_index;
		const auto owner = rowgroup_to_workset.find(rowgroup_key);
		if (owner == rowgroup_to_workset.end()) {
			throw std::runtime_error("block-major active-output group has no resident workset");
		}
		group_workset[group_index] = owner->second;
	}
	schedule.group_workset_build_ms =
	    std::chrono::duration<double, std::milli>(Clock::now() - group_workset_start).count();

	std::vector<uint32_t> owner_generations(workset_count, 0U);
	std::vector<uint32_t> owners;
	owners.reserve(std::min<size_t>(workset_count, 16U));
	const auto enumerate = [&](auto&& emit) -> uint64_t {
		std::fill(owner_generations.begin(), owner_generations.end(), 0U);
		owners.clear();
		uint32_t generation = 0U;
		uint64_t contribution_count = 0U;
		for (size_t image_index = 0U; image_index < plan.images.size(); ++image_index) {
			const auto& image = plan.images[image_index];
			for (uint32_t component_index = 0U; component_index < image.components.size(); ++component_index) {
				const auto& component = image.components[component_index];
				if (component.present == 0U) {
					if (component.block_major_coordinate_lookup_index != std::numeric_limits<uint32_t>::max()) {
						throw std::runtime_error("absent block-major component has a coordinate lookup");
					}
					continue;
				}
				if (component.x_up_factor == 0U || component.y_up_factor == 0U ||
				    component.x_down_factor == 0U || component.y_down_factor == 0U) {
					throw std::runtime_error("block-major active-output component has an invalid axis relation");
				}
				if (component.block_major_coordinate_lookup_index >= plan.coordinate_group_lookups.size()) {
					throw std::runtime_error("block-major active-output component has no coordinate lookup");
				}
				const auto& lookup =
				    plan.coordinate_group_lookups[component.block_major_coordinate_lookup_index];
				if (lookup.shard_id != image.shard_id || lookup.semantic_slot_id != component.semantic_slot_id ||
				    lookup.width == 0U || lookup.height == 0U || lookup.stride < lookup.width) {
					throw std::runtime_error("block-major active-output component coordinate lookup is inconsistent");
				}
				const uint64_t lookup_area = static_cast<uint64_t>(lookup.stride) * lookup.height;
				if (lookup.group_index_base > plan.coordinate_group_indices.size() ||
				    lookup_area > plan.coordinate_group_indices.size() - lookup.group_index_base) {
					throw std::runtime_error("block-major active-output coordinate lookup range is invalid");
				}
				const auto* coordinate_groups =
				    plan.coordinate_group_indices.data() + static_cast<size_t>(lookup.group_index_base);
				const auto output_width = component_index == 0U ? transform.y_output_width_blocks
				                                                   : transform.cbcr_output_width_blocks;
				const auto output_height = component_index == 0U ? transform.y_output_height_blocks
				                                                    : transform.cbcr_output_height_blocks;
				const uint64_t component_offset = component_index == 0U
				                                              ? 0U
				                                              : y_blocks + static_cast<uint64_t>(component_index - 1U) *
				                                                               cbcr_channel_blocks;
				for (uint32_t output_y = 0U; output_y < output_height; ++output_y) {
					const auto source_y_begin = static_cast<uint32_t>(
					    (static_cast<uint64_t>(output_y) * component.y_down_factor) / component.y_up_factor);
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
						    (static_cast<uint64_t>(output_x) * component.x_down_factor) / component.x_up_factor);
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
								++contribution_count;
								if (absolute_x < lookup.origin_x || absolute_y < lookup.origin_y ||
								    static_cast<uint64_t>(absolute_x) - lookup.origin_x >= lookup.width ||
								    static_cast<uint64_t>(absolute_y) - lookup.origin_y >= lookup.height) {
									throw std::runtime_error(
									    "block-major active-output source coordinate is outside its lookup");
								}
								const auto local = (static_cast<uint64_t>(absolute_y) - lookup.origin_y) * lookup.stride +
								                   (static_cast<uint64_t>(absolute_x) - lookup.origin_x);
								const auto group_index = coordinate_groups[local];
								if (group_index == kInvalidJpegDctBlockMajorGroupIndex || group_index >= plan.groups.size()) {
									throw std::runtime_error(
									    "block-major active-output source coordinate is a topology hole");
								}
								const auto workset = group_workset[group_index];
								if (workset >= workset_count) {
									throw std::runtime_error("block-major source group has no active-output workset");
								}
								if (owner_generations[workset] != generation) {
									owner_generations[workset] = generation;
									owners.push_back(workset);
								}
							}
						}
						std::sort(owners.begin(), owners.end());
						const auto linear = static_cast<uint32_t>(
						    static_cast<uint64_t>(image_index) * blocks_per_image + component_offset +
						    static_cast<uint64_t>(output_y) * output_width + output_x);
						for (const auto workset : owners) {
							emit(workset, linear);
						}
					}
				}
			}
		}
		return contribution_count;
	};

	schedule.offsets.assign(static_cast<size_t>(workset_count) + 1U, 0U);
	const auto count_start = Clock::now();
	schedule.source_contribution_count = enumerate([&](const uint32_t workset, const uint32_t) {
		if (schedule.offsets[workset + 1U] == std::numeric_limits<uint64_t>::max()) {
			throw std::runtime_error("block-major active-output ownership count overflow");
		}
		++schedule.offsets[workset + 1U];
	});
	schedule.active_output_count_ms =
	    std::chrono::duration<double, std::milli>(Clock::now() - count_start).count();

	const auto prefix_start = Clock::now();
	for (uint32_t workset = 0U; workset < workset_count; ++workset) {
		if (schedule.offsets[workset + 1U] >
		    std::numeric_limits<uint64_t>::max() - schedule.offsets[workset]) {
			throw std::runtime_error("block-major active-output prefix sum overflow");
		}
		schedule.offsets[workset + 1U] += schedule.offsets[workset];
	}
	schedule.output_workset_ownership_count = schedule.offsets.back();
	if (schedule.output_workset_ownership_count >
	    schedule.logical_output_block_count * static_cast<uint64_t>(workset_count)) {
		throw std::runtime_error("block-major active-output ownership count exceeds the logical grid");
	}
	if (schedule.offsets.back() > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error("block-major active-output schedule exceeds host address range");
	}
	schedule.active_output_blocks.resize(static_cast<size_t>(schedule.offsets.back()));
	std::vector<uint64_t> cursors(schedule.offsets.begin(), schedule.offsets.end() - 1U);
	schedule.active_output_prefix_ms =
	    std::chrono::duration<double, std::milli>(Clock::now() - prefix_start).count();

	const auto fill_start = Clock::now();
	const auto fill_contribution_count = enumerate([&](const uint32_t workset, const uint32_t linear) {
		if (cursors[workset] >= schedule.offsets[workset + 1U]) {
			throw std::runtime_error("block-major active-output fill exceeded its workset slice");
		}
		schedule.active_output_blocks[static_cast<size_t>(cursors[workset]++)] = linear;
	});
	if (fill_contribution_count != schedule.source_contribution_count ||
	    schedule.source_contribution_count > std::numeric_limits<uint64_t>::max() - fill_contribution_count) {
		throw std::runtime_error("block-major active-output count/fill contribution mismatch");
	}
	schedule.source_contribution_visit_count = schedule.source_contribution_count + fill_contribution_count;
	for (uint32_t workset = 0U; workset < workset_count; ++workset) {
		if (cursors[workset] != schedule.offsets[workset + 1U]) {
			throw std::runtime_error("block-major active-output fill did not complete its workset slice");
		}
		const auto begin = schedule.active_output_blocks.begin() + static_cast<ptrdiff_t>(schedule.offsets[workset]);
		const auto end = schedule.active_output_blocks.begin() + static_cast<ptrdiff_t>(schedule.offsets[workset + 1U]);
		if (!std::is_sorted(begin, end) || std::adjacent_find(begin, end) != end ||
		    std::any_of(begin, end, [&](const uint32_t output) {
			    return output >= schedule.logical_output_block_count;
		    })) {
			throw std::runtime_error("block-major active-output workset slice is not sorted, unique, and bounded");
		}
	}
	schedule.active_output_fill_ms =
	    std::chrono::duration<double, std::milli>(Clock::now() - fill_start).count();

	// The coordinate index belongs to the plan and is reported as persistent
	// compact-plan storage. The schedule peak contains only per-build ownership
	// state; there are no node-based coordinate maps or nested output vectors.
	schedule.temporary_bytes_peak =
	    static_cast<uint64_t>(group_workset.capacity()) * sizeof(uint32_t) +
	    static_cast<uint64_t>(rowgroup_worksets.size()) *
	        (sizeof(uint64_t) + sizeof(uint32_t) + 4U * sizeof(void*)) +
	    static_cast<uint64_t>(workset_seen.capacity()) * sizeof(uint8_t) +
	    static_cast<uint64_t>(owner_generations.capacity()) * sizeof(uint32_t) +
	    static_cast<uint64_t>(owners.capacity()) * sizeof(uint32_t) +
	    static_cast<uint64_t>(cursors.capacity()) * sizeof(uint64_t);
	schedule.total_build_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
	return schedule;
}

struct JpegDctShardDatasetReader::Impl {
	struct BlockGroupKey {
		uint32_t semantic_slot_id = 0;
		uint32_t block_x          = 0;
		uint32_t block_y          = 0;

		bool operator==(const BlockGroupKey& other) const noexcept {
			return semantic_slot_id == other.semantic_slot_id && block_x == other.block_x && block_y == other.block_y;
		}
	};

	struct BlockGroupKeyHash {
		size_t operator()(const BlockGroupKey& key) const noexcept {
			uint64_t h = static_cast<uint64_t>(key.semantic_slot_id) * 0x9e3779b185ebca87ULL;
			h ^= static_cast<uint64_t>(key.block_x) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
			h ^= static_cast<uint64_t>(key.block_y) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
			return static_cast<size_t>(h);
		}
	};

	struct RankCursorKey {
		uint32_t shard_id    = 0;
		size_t   group_index = 0;

		bool operator==(const RankCursorKey& other) const noexcept {
			return shard_id == other.shard_id && group_index == other.group_index;
		}
	};

	struct RankCursorKeyHash {
		size_t operator()(const RankCursorKey& key) const noexcept {
			auto h = static_cast<uint64_t>(key.shard_id) * 0x9e3779b185ebca87ULL;
			h ^= static_cast<uint64_t>(key.group_index) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
			return static_cast<size_t>(h);
		}
	};

	struct RankCursor {
		uint32_t next_local_image    = 0;
		uint32_t present_before_next = 0;
	};

	struct ShardState {
		JpegDctShardManifestEntry                                    entry;
		std::filesystem::path                                        fls_path;
		std::filesystem::path                                        metadata_path;
		mutable JpegDctDatasetMetadata                               metadata;
		mutable std::vector<uint64_t>                                rowgroup_n_tuples;
		mutable std::unordered_map<BlockGroupKey, size_t, BlockGroupKeyHash> block_group_lookup;
		mutable std::shared_ptr<std::mutex>                           metadata_mutex = std::make_shared<std::mutex>();
		mutable bool                                                  metadata_loaded = false;
		uint32_t planless_direct_layout_index    = std::numeric_limits<uint32_t>::max();
		bool     planless_direct_image_rowgroups = false;
		bool     vector_rowgroups                = false;
	};

	struct PlanlessStaticComponentDescriptor {
		uint32_t component_row_offset = 0;
		uint32_t quant_table_index    = 0;
		uint16_t width_in_blocks      = 0;
		uint16_t height_in_blocks     = 0;
		int8_t   h_samp_factor        = 0;
		int8_t   v_samp_factor        = 0;
		uint8_t  present              = 0;
		uint8_t  reserved             = 0;

		bool operator==(const PlanlessStaticComponentDescriptor& other) const {
			return component_row_offset == other.component_row_offset && quant_table_index == other.quant_table_index &&
			       width_in_blocks == other.width_in_blocks && height_in_blocks == other.height_in_blocks &&
			       h_samp_factor == other.h_samp_factor && v_samp_factor == other.v_samp_factor &&
			       present == other.present;
		}
	};

	struct alignas(64) PlanlessStaticLayoutDescriptor {
		std::array<PlanlessStaticComponentDescriptor, 3> components {};
		uint32_t                                         full_vector_count = 0;
		uint8_t                                          zigzag_columns    = 0;
		uint8_t                                          spatial_order     = 0;
		uint8_t                                          image_major       = 0;
		uint8_t                                          vector_rowgroups  = 0;
		std::array<uint8_t, 8>                           padding {};

		bool operator==(const PlanlessStaticLayoutDescriptor& other) const {
			return components == other.components && zigzag_columns == other.zigzag_columns &&
			       spatial_order == other.spatial_order && image_major == other.image_major &&
			       vector_rowgroups == other.vector_rowgroups && full_vector_count == other.full_vector_count;
		}
	};

	struct alignas(32) PlanlessStaticImageDescriptor {
		uint32_t fls_rowgroup_index    = 0;
		uint32_t row_start_in_rowgroup = 0;
		uint32_t rowgroup_count         = 0;
		uint32_t layout_index          = 0;
		uint16_t image_width           = 0;
		uint16_t image_height          = 0;
		std::array<uint8_t, 12> padding {};
	};

	struct PlanlessStaticShardDescriptor {
		const std::filesystem::path* fls_path                 = nullptr;
		uint64_t                     first_global_image_index = 0U;
		uint32_t                     shard_id                 = 0U;
		uint32_t                     rowgroup_count           = 0U;
		uint32_t                     direct_layout_index      = std::numeric_limits<uint32_t>::max();
		bool                         direct_image_rowgroups   = false;
	};
	struct PlanlessAxisCapacityKey {
		std::array<uint32_t, 4> output_extents {};

		bool operator==(const PlanlessAxisCapacityKey& other) const noexcept {
			return output_extents == other.output_extents;
		}
	};
	struct PlanlessAxisCapacityKeyHash {
		size_t operator()(const PlanlessAxisCapacityKey& key) const noexcept {
			size_t hash = 0U;
			for (const auto extent : key.output_extents) {
				hash ^= static_cast<size_t>(extent) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
			}
			return hash;
		}
	};
	struct PlanlessAxisCapacityContract {
		size_t float_count = 0U;
		bool   complete    = false;
	};
	static_assert(sizeof(PlanlessStaticComponentDescriptor) == 16U);
	static_assert(sizeof(PlanlessStaticLayoutDescriptor) == 64U);
	static_assert(sizeof(PlanlessStaticImageDescriptor) == 32U);
	static_assert(sizeof(PlanlessStaticShardDescriptor) == 32U);
	mutable JpegDctReaderInitializationStats initialization_stats;
	mutable std::mutex initialization_stats_mutex;
	bool lazy_block_major_metadata = false;
	std::filesystem::path block_major_access_directory;

	explicit Impl(const std::filesystem::path& manifest_path)
	    : root_dir(manifest_path.parent_path()) {
		const auto total_started = std::chrono::steady_clock::now();
		const auto manifest_started = std::chrono::steady_clock::now();
		manifest = detail::read_jpeg_dct_shard_manifest_file(manifest_path);
		initialization_stats.manifest_load_ms = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - manifest_started).count();
		initialization_stats.manifest_shard_count = manifest.shards.size();
		block_major_access_directory = root_dir / "block_major_access_v1";
		if (const auto* override_directory = std::getenv("GALP_BLOCK_MAJOR_ACCESS_DIR");
		    override_directory != nullptr && *override_directory != '\0') {
			block_major_access_directory = override_directory;
		}
		lazy_block_major_metadata =
		    manifest.version == 1U &&
		    std::filesystem::is_regular_file(block_major_access_directory / "manifest.block_major_access.bin");
		initialization_stats.block_major_metadata_lazy = lazy_block_major_metadata;
		if (manifest.version == 3U && !manifest.uses_compact_descriptor()) {
			throw std::runtime_error("JPEG DCT manifest-v3 is missing the canonical compact descriptor contract");
		}
		shards.reserve(manifest.shards.size());
		for (const auto& entry : manifest.shards) {
			const auto path_validation_started = std::chrono::steady_clock::now();
			ShardState state;
			state.entry             = entry;
			state.fls_path          = resolve_manifest_member(root_dir, entry.fls_file_name, "shard");
			state.metadata_path     = resolve_manifest_member(root_dir, entry.metadata_file_name, "metadata");
			if (manifest.uses_compact_descriptor() &&
			    (!std::filesystem::is_regular_file(state.fls_path) ||
			     std::filesystem::file_size(state.fls_path) != entry.fls_file_size)) {
				throw std::runtime_error("JPEG DCT shard file size disagrees with the manifest");
			}
			if (manifest.uses_compact_descriptor() &&
			    (!std::filesystem::is_regular_file(state.metadata_path) ||
			     std::filesystem::file_size(state.metadata_path) != entry.metadata_file_size)) {
				throw std::runtime_error("JPEG DCT metadata file size disagrees with the manifest");
			}
			if (manifest.uses_compact_descriptor()) {
				if (manifest.physical_layout != "image-major-vector-rowgroups" || manifest.vector_size != 1024U ||
				    manifest.spatial_order_name != "tiled-z32" ||
				    manifest.spatial_order != JpegDctSpatialOrder::kTiledZ32 || manifest.rowgroup_vectors != 1U ||
				    !galp::format::is_compact_v3_fls(state.fls_path)) {
					throw std::runtime_error("JPEG DCT Compact v3 manifest contract is invalid");
				}
				fastlanes::File file(state.fls_path);
				fastlanes::FileFooter footer {};
				const auto file_size = file.Size();
				if (!fastlanes::FileFooter::Load(footer, file).success ||
				    footer.table_descriptor_size < 256U ||
				    footer.table_descriptor_size != entry.compact_descriptor_size ||
				    footer.table_descriptor_offset < sizeof(fastlanes::FileHeader) ||
				    footer.table_descriptor_offset - sizeof(fastlanes::FileHeader) != entry.payload_size ||
				    footer.table_descriptor_offset > file_size - sizeof(fastlanes::FileFooter) ||
				    footer.table_descriptor_size >
				        file_size - sizeof(fastlanes::FileFooter) - footer.table_descriptor_offset ||
				    footer.table_descriptor_offset + footer.table_descriptor_size !=
				        file_size - sizeof(fastlanes::FileFooter)) {
					throw std::runtime_error("JPEG DCT Compact v3 shard geometry disagrees with the manifest");
				}
				std::array<uint8_t, 72> compact_header {};
				file.ReadRangeUnchecked(compact_header.data(), footer.table_descriptor_offset, compact_header.size());
				const auto compact_u32 = [&](const size_t offset) {
					uint32_t value = 0U;
					for (size_t byte = 0U; byte < sizeof(value); ++byte) {
						value |= static_cast<uint32_t>(compact_header[offset + byte]) << (byte * 8U);
					}
					return value;
				};
				const auto compact_u64 = [&](const size_t offset) {
					uint64_t value = 0U;
					for (size_t byte = 0U; byte < sizeof(value); ++byte) {
						value |= static_cast<uint64_t>(compact_header[offset + byte]) << (byte * 8U);
					}
					return value;
				};
				if (compact_u64(16U) != entry.compact_descriptor_size || compact_u64(24U) != entry.payload_size ||
				    compact_u64(32U) != entry.payload_crc64 || compact_u64(40U) != entry.rowgroup_count ||
				    compact_u32(48U) != 64U || compact_u32(52U) != manifest.vector_size ||
				    compact_u32(56U) != static_cast<uint32_t>(manifest.spatial_order)) {
					throw std::runtime_error("JPEG DCT Compact v3 descriptor header disagrees with the manifest");
				}
			}
			initialization_stats.shard_path_validation_ms += std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - path_validation_started).count();
			state.vector_rowgroups  = manifest.uses_independent_vector_rowgroups();
			shards.push_back(std::move(state));
			if (!lazy_block_major_metadata) {
				ensure_shard_metadata(shards.back());
			}
		}
		if (manifest.version == 3U && manifest.uses_compact_descriptor()) {
			auto paths = std::make_shared<std::vector<std::filesystem::path>>();
			paths->reserve(shards.size());
			for (const auto& shard : shards) {
				paths->push_back(shard.fls_path);
			}
			image_major_static_metadata_paths = std::move(paths);
		}
		const auto profile_started = std::chrono::steady_clock::now();
		if (!lazy_block_major_metadata) {
		planless_static_images.resize(manifest.image_count);
		if (manifest.image_count > std::numeric_limits<size_t>::max() / 3U) {
			throw std::runtime_error("JPEG DCT image count exceeds the compiled access profile cache range");
		}
		compiled_fixed_access_profiles.resize(static_cast<size_t>(manifest.image_count) * 3U);
		planless_static_shards.resize(shards.size());
		planless_shard_indices_valid = !shards.empty() && shards.size() <= std::numeric_limits<uint16_t>::max();
		uint64_t expected_first_global_image_index = 0U;
		for (const auto& shard : shards) {
			const auto& entry = shard.entry;
			if (entry.first_global_image_index != expected_first_global_image_index ||
			    entry.image_count > manifest.image_count - expected_first_global_image_index) {
				planless_shard_indices_valid = false;
				break;
			}
			expected_first_global_image_index += entry.image_count;
		}
		planless_shard_indices_valid =
		    planless_shard_indices_valid && expected_first_global_image_index == manifest.image_count;
		if (planless_shard_indices_valid) {
			const auto shard_stride = shards.front().entry.image_count;
			bool       uniform      = shard_stride > 0U;
			for (size_t shard_index = 0; uniform && shard_index < shards.size(); ++shard_index) {
				const auto& entry = shards[shard_index].entry;
				uniform           = shard_index <= std::numeric_limits<uint64_t>::max() / shard_stride &&
				          entry.first_global_image_index == shard_index * shard_stride &&
				          (shard_index + 1U == shards.size() ? entry.image_count <= shard_stride
				                                             : entry.image_count == shard_stride);
			}
			if (uniform) {
				planless_uniform_shard_image_count = shard_stride;
			}
		}
		if (planless_shard_indices_valid && planless_uniform_shard_image_count == 0U) {
			planless_shard_indices.resize(manifest.image_count);
		}
		std::unordered_map<uint64_t, std::vector<uint32_t>> quant_indices_by_fingerprint;
		for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
			auto& shard = shards[shard_index];
			bool  direct_image_rowgroups =
			    shard.metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor;
			uint32_t direct_layout_index = std::numeric_limits<uint32_t>::max();
			for (size_t local_image_index = 0; local_image_index < shard.metadata.images.size(); ++local_image_index) {
				const auto global_image_index = shard.entry.first_global_image_index + local_image_index;
				if (global_image_index >= planless_static_images.size()) {
					throw std::runtime_error("JPEG DCT shard image range exceeds manifest image count");
				}
				const auto& image      = shard.metadata.images[local_image_index];
				auto&       descriptor = planless_static_images[global_image_index];
				if (image.image_width > std::numeric_limits<uint16_t>::max() ||
				    image.image_height > std::numeric_limits<uint16_t>::max()) {
					throw std::runtime_error("JPEG DCT image dimensions exceed compact planless descriptor range");
				}
				descriptor.image_width  = static_cast<uint16_t>(image.image_width);
				descriptor.image_height = static_cast<uint16_t>(image.image_height);
				if (!planless_shard_indices.empty()) {
					planless_shard_indices[global_image_index] = static_cast<uint16_t>(shard_index);
				}
				PlanlessStaticLayoutDescriptor layout;
				layout.zigzag_columns = static_cast<uint8_t>(shard.metadata.zigzag_columns ? 1U : 0U);
				layout.spatial_order  = static_cast<uint8_t>(shard.metadata.image_major_spatial_order);
				layout.image_major = static_cast<uint8_t>(
				    shard.metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor);
				layout.vector_rowgroups = static_cast<uint8_t>(shard.vector_rowgroups);
				if (layout.image_major) {
					const auto& group                = shard.metadata.image_group_index.at(local_image_index);
					descriptor.fls_rowgroup_index    = group.fls_rowgroup_index;
					descriptor.row_start_in_rowgroup = group.row_start_in_rowgroup;
					const auto full_vectors = row_count_to_vector_count(
					    shard.vector_rowgroups ? group.row_count
					                           : shard.rowgroup_n_tuples.at(group.fls_rowgroup_index));
					if (full_vectors > std::numeric_limits<uint32_t>::max()) {
						layout.image_major = 0U;
					} else {
						layout.full_vector_count = static_cast<uint32_t>(full_vectors);
						descriptor.rowgroup_count = shard.vector_rowgroups
						                                ? static_cast<uint32_t>(full_vectors)
						                                : 1U;
					}
					if (descriptor.fls_rowgroup_index > shard.entry.rowgroup_count ||
					    descriptor.rowgroup_count >
					        shard.entry.rowgroup_count - descriptor.fls_rowgroup_index) {
						throw std::runtime_error("JPEG DCT image vector-rowgroup range exceeds the shard");
					}
					direct_image_rowgroups = direct_image_rowgroups && !shard.vector_rowgroups &&
					                         group.fls_rowgroup_index == local_image_index &&
					                         group.row_start_in_rowgroup == 0U;
				}
				std::array<const JpegComponentMetadata*, 3> fixed_components {};
				std::array<uint32_t, 3>                     component_row_offsets {};
				bool                                        compact_fields_fit   = true;
				uint64_t                                    component_row_offset = 0;
				for (const auto& component : image.components) {
					const auto current_row_offset = component_row_offset;
					if (component.present) {
						component_row_offset +=
						    static_cast<uint64_t>(component.width_in_blocks) * component.height_in_blocks;
					}
					if (component.present && component.semantic_slot_id < fixed_components.size()) {
						fixed_components[component.semantic_slot_id] = &component;
						if (current_row_offset > std::numeric_limits<uint32_t>::max()) {
							compact_fields_fit = false;
						} else {
							component_row_offsets[component.semantic_slot_id] =
							    static_cast<uint32_t>(current_row_offset);
						}
					}
				}
				if (fixed_components[0] == nullptr) {
					component_row_offset = 0;
					for (const auto& component : image.components) {
						const auto current_row_offset = component_row_offset;
						if (component.present) {
							component_row_offset +=
							    static_cast<uint64_t>(component.width_in_blocks) * component.height_in_blocks;
						}
						if (component.present && component.local_component_index < fixed_components.size()) {
							if (current_row_offset > std::numeric_limits<uint32_t>::max()) {
								compact_fields_fit = false;
							} else {
								component_row_offsets[component.local_component_index] =
								    static_cast<uint32_t>(current_row_offset);
							}
							fixed_components[component.local_component_index] = &component;
						}
					}
				}
				for (size_t slot = 0; slot < fixed_components.size(); ++slot) {
					const auto* component = fixed_components[slot];
					if (component == nullptr || !component->present) {
						continue;
					}
					if (component->width_in_blocks > std::numeric_limits<uint16_t>::max() ||
					    component->height_in_blocks > std::numeric_limits<uint16_t>::max() ||
					    component->h_samp_factor < std::numeric_limits<int8_t>::min() ||
					    component->h_samp_factor > std::numeric_limits<int8_t>::max() ||
					    component->v_samp_factor < std::numeric_limits<int8_t>::min() ||
					    component->v_samp_factor > std::numeric_limits<int8_t>::max()) {
						compact_fields_fit = false;
						continue;
					}
					const auto quant_table = std::find_if(
					    image.quant_tables.begin(), image.quant_tables.end(), [&](const JpegQuantTableMetadata& table) {
						    return component->quant_tbl_no >= 0 &&
						           table.table_id == static_cast<uint8_t>(component->quant_tbl_no);
					    });
					if (quant_table == image.quant_tables.end()) {
						continue;
					}
					uint32_t dictionary_index = std::numeric_limits<uint32_t>::max();
					auto&    candidates       = quant_indices_by_fingerprint[component->quant_table_fingerprint];
					for (const auto candidate : candidates) {
						if (planless_quant_tables[candidate] == quant_table->values) {
							dictionary_index = candidate;
							break;
						}
					}
					if (dictionary_index == std::numeric_limits<uint32_t>::max()) {
						dictionary_index = static_cast<uint32_t>(planless_quant_tables.size());
						planless_quant_tables.push_back(quant_table->values);
						candidates.push_back(dictionary_index);
					}
					auto& compact_component                = layout.components[slot];
					compact_component.component_row_offset = component_row_offsets[slot];
					compact_component.width_in_blocks      = static_cast<uint16_t>(component->width_in_blocks);
					compact_component.height_in_blocks     = static_cast<uint16_t>(component->height_in_blocks);
					compact_component.h_samp_factor        = static_cast<int8_t>(component->h_samp_factor);
					compact_component.v_samp_factor        = static_cast<int8_t>(component->v_samp_factor);
					compact_component.quant_table_index    = dictionary_index;
					compact_component.present              = 1U;
				}
				if (!compact_fields_fit) {
					layout.image_major = 0U;
				}
				const auto layout_found =
				    std::find(planless_static_layouts.begin(), planless_static_layouts.end(), layout);
				if (layout_found == planless_static_layouts.end()) {
					descriptor.layout_index = static_cast<uint32_t>(planless_static_layouts.size());
					planless_static_layouts.push_back(std::move(layout));
				} else {
					descriptor.layout_index =
					    static_cast<uint32_t>(std::distance(planless_static_layouts.begin(), layout_found));
				}
				if (direct_layout_index == std::numeric_limits<uint32_t>::max()) {
					direct_layout_index = descriptor.layout_index;
				} else {
					direct_image_rowgroups = direct_image_rowgroups && direct_layout_index == descriptor.layout_index;
				}
			}
			if (direct_image_rowgroups && direct_layout_index < planless_static_layouts.size() &&
			    planless_static_layouts[direct_layout_index].image_major) {
				shard.planless_direct_layout_index    = direct_layout_index;
				shard.planless_direct_image_rowgroups = true;
			}
			auto& compact_shard                    = planless_static_shards[shard_index];
			compact_shard.fls_path                 = &shard.fls_path;
			compact_shard.first_global_image_index = shard.entry.first_global_image_index;
			compact_shard.shard_id                 = shard.entry.shard_id;
			compact_shard.rowgroup_count           = shard.entry.rowgroup_count;
			compact_shard.direct_layout_index      = shard.planless_direct_layout_index;
			compact_shard.direct_image_rowgroups   = shard.planless_direct_image_rowgroups;
		}
		}
		initialization_stats.transform_profile_construction_ms = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - profile_started).count();
		const auto companion_started = std::chrono::steady_clock::now();
		const bool block_major_v1 = lazy_block_major_metadata ||
		                            (manifest.version == 1U && !shards.empty() &&
		                            std::all_of(shards.begin(), shards.end(), [](const auto& shard) {
			                            return shard.metadata.row_ordering ==
			                                   JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
		                            }));
		if (block_major_v1) {
			if (std::filesystem::is_regular_file(block_major_access_directory / "manifest.block_major_access.bin")) {
				block_major_compact_planner =
				    std::make_unique<JpegDctBlockMajorCompactPlanner>(manifest_path, block_major_access_directory);
			}
		}
		initialization_stats.block_major_companion_index_load_ms = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - companion_started).count();
		initialization_stats.total_ms = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - total_started).count();
	}

	static std::vector<uint64_t> derive_rowgroup_n_tuples(const JpegDctDatasetMetadata& metadata,
	                                                      const uint32_t                rowgroup_count,
	                                                      const bool vector_rowgroups) {
		std::vector<uint64_t> rowgroup_n_tuples(rowgroup_count, 0);
		if (metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
			for (const auto& group : metadata.image_group_index) {
				if (group.fls_rowgroup_index >= rowgroup_n_tuples.size()) {
					throw std::runtime_error("JPEG DCT image index exceeds shard manifest rowgroup count");
				}
				if (vector_rowgroups) {
					uint64_t remaining = group.row_count;
					uint64_t rowgroup_index = group.fls_rowgroup_index;
					while (remaining != 0U) {
						if (rowgroup_index >= rowgroup_n_tuples.size()) {
							throw std::runtime_error("JPEG DCT manifest-v3 image exceeds shard rowgroup count");
						}
						const auto chunk_rows = std::min<uint64_t>(fastlanes::CFG::VEC_SZ, remaining);
						rowgroup_n_tuples[rowgroup_index++] = chunk_rows;
						remaining -= chunk_rows;
					}
					continue;
				}
				const uint64_t row_end = static_cast<uint64_t>(group.row_start_in_rowgroup) + group.row_count;
				rowgroup_n_tuples[group.fls_rowgroup_index] =
				    std::max(rowgroup_n_tuples[group.fls_rowgroup_index], row_end);
			}
			return rowgroup_n_tuples;
		}
		for (const auto& group : metadata.block_group_index) {
			if (group.fls_rowgroup_index >= rowgroup_n_tuples.size()) {
				throw std::runtime_error("JPEG DCT metadata rowgroup index exceeds shard manifest rowgroup count");
			}
			const uint64_t row_end =
			    static_cast<uint64_t>(group.row_start_in_rowgroup) + static_cast<uint64_t>(group.row_count);
			rowgroup_n_tuples[group.fls_rowgroup_index] =
			    std::max(rowgroup_n_tuples[group.fls_rowgroup_index], row_end);
		}
		return rowgroup_n_tuples;
	}

	void ensure_shard_metadata(ShardState& state) const {
		std::lock_guard<std::mutex> shard_guard(*state.metadata_mutex);
		if (state.metadata_loaded) {
			return;
		}
		const auto metadata_load_started = std::chrono::steady_clock::now();
		state.metadata = detail::read_jpeg_dct_metadata_file(state.metadata_path);
		const auto metadata_loaded = std::chrono::steady_clock::now();
		if (lazy_block_major_metadata &&
		    state.metadata.row_ordering != JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor) {
			throw std::runtime_error("block-major companion index is attached to incompatible shard metadata");
		}
		state.rowgroup_n_tuples =
		    derive_rowgroup_n_tuples(state.metadata, state.entry.rowgroup_count, state.vector_rowgroups);
		if (state.metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
			if (state.metadata.image_group_index.size() != state.entry.image_count ||
			    state.metadata.images.size() != state.entry.image_count) {
				throw std::runtime_error(
				    "JPEG DCT image-major metadata must contain one dense image index record per image");
			}
			for (size_t image_idx = 0; image_idx < state.metadata.image_group_index.size(); ++image_idx) {
				const auto& group = state.metadata.image_group_index[image_idx];
				if (group.local_image_index != image_idx || group.fls_rowgroup_index >= state.entry.rowgroup_count) {
					throw std::runtime_error("JPEG DCT image-major metadata index is malformed");
				}
			}
		}
		state.block_group_lookup.reserve(state.metadata.block_group_index.size());
		for (size_t group_idx = 0; group_idx < state.metadata.block_group_index.size(); ++group_idx) {
			const auto& group = state.metadata.block_group_index[group_idx];
			state.block_group_lookup.emplace(BlockGroupKey {group.semantic_slot_id, group.block_x, group.block_y},
			                                 group_idx);
		}
		state.metadata_loaded = true;
		const auto metadata_indexed = std::chrono::steady_clock::now();
		std::lock_guard<std::mutex> stats_guard(initialization_stats_mutex);
		initialization_stats.shard_metadata_load_ms +=
		    std::chrono::duration<double, std::milli>(metadata_loaded - metadata_load_started).count();
		initialization_stats.shard_metadata_index_ms +=
		    std::chrono::duration<double, std::milli>(metadata_indexed - metadata_loaded).count();
		++initialization_stats.loaded_shard_metadata_count;
		if (!lazy_block_major_metadata) {
			++initialization_stats.eagerly_loaded_shard_metadata_count;
		}
	}

	static size_t row_count_to_vector_count(const uint64_t row_count) {
		return static_cast<size_t>((row_count + fastlanes::CFG::VEC_SZ - 1U) / fastlanes::CFG::VEC_SZ);
	}

	const ShardState& shard_for_global_image(const uint32_t global_image_index) const {
		for (auto& shard : shards) {
			const auto first = shard.entry.first_global_image_index;
			const auto last  = first + shard.entry.image_count;
			if (global_image_index >= first && global_image_index < last) {
				ensure_shard_metadata(shard);
				return shard;
			}
		}
		throw std::runtime_error("JPEG DCT global image index is outside the shard manifest");
	}

	const ShardState& shard_by_id(const uint32_t shard_id) const {
		for (auto& shard : shards) {
			if (shard.entry.shard_id == shard_id) {
				ensure_shard_metadata(shard);
				return shard;
			}
		}
		throw std::runtime_error("JPEG DCT shard id is outside the shard manifest");
	}

	static const JpegDctBlockGroupIndex& find_group(const JpegDctDatasetMetadata& metadata,
	                                                const uint32_t                semantic_slot_id,
	                                                const uint32_t                block_x,
	                                                const uint32_t                block_y) {
		const auto* group = find_group_or_null(metadata, semantic_slot_id, block_x, block_y);
		if (group == nullptr) {
			throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
		}
		return *group;
	}

	static const JpegDctBlockGroupIndex* find_group_or_null(const JpegDctDatasetMetadata& metadata,
	                                                        const uint32_t                semantic_slot_id,
	                                                        const uint32_t                block_x,
	                                                        const uint32_t                block_y) {
		for (const auto& group : metadata.block_group_index) {
			if (group.semantic_slot_id == semantic_slot_id && group.block_x == block_x && group.block_y == block_y) {
				return &group;
			}
		}
		return nullptr;
	}

	static const JpegDctBlockGroupIndex* find_group_or_null(const ShardState& shard,
	                                                        const uint32_t    semantic_slot_id,
	                                                        const uint32_t    block_x,
	                                                        const uint32_t    block_y) {
		const auto it = shard.block_group_lookup.find(BlockGroupKey {semantic_slot_id, block_x, block_y});
		if (it == shard.block_group_lookup.end()) {
			return nullptr;
		}
		return &shard.metadata.block_group_index[it->second];
	}

	static size_t group_index_in_shard(const ShardState& shard, const JpegDctBlockGroupIndex& group) {
		return static_cast<size_t>(&group - shard.metadata.block_group_index.data());
	}

	static bool image_has_block(const JpegImageMetadata& image,
	                            const uint32_t           semantic_slot_id,
	                            const uint32_t           block_x,
	                            const uint32_t           block_y) {
		for (const auto& component : image.components) {
			if (component.semantic_slot_id == semantic_slot_id) {
				return component.present && block_x < component.width_in_blocks && block_y < component.height_in_blocks;
			}
		}
		return false;
	}

	static uint32_t ceil_mul_div_u32(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
		if (divisor == 0) {
			throw std::runtime_error("JPEG DCT crop planning encountered a zero image dimension");
		}
		const uint64_t product = static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs);
		return static_cast<uint32_t>((product + divisor - 1U) / divisor);
	}

	static uint32_t floor_mul_div_u32(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
		if (divisor == 0) {
			throw std::runtime_error("JPEG DCT crop planning encountered a zero image dimension");
		}
		return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs)) / divisor);
	}

	static JpegDctCropBox effective_crop_box(const JpegImageMetadata& image, JpegDctCropBox crop) {
		if (image.image_width == 0 || image.image_height == 0) {
			throw std::runtime_error("JPEG DCT device crop planning requires per-image dimensions");
		}
		if (crop.width == 0 || crop.height == 0) {
			crop.x      = 0;
			crop.y      = 0;
			crop.width  = image.image_width;
			crop.height = image.image_height;
			return crop;
		}
		if (crop.x >= image.image_width || crop.y >= image.image_height) {
			throw std::out_of_range("JPEG DCT crop starts outside the source image");
		}
		if (crop.width > image.image_width - crop.x) {
			crop.width = image.image_width - crop.x;
		}
		if (crop.height > image.image_height - crop.y) {
			crop.height = image.image_height - crop.y;
		}
		return crop;
	}

	static uint32_t closest_aligned_crop_extent(const uint32_t               source_extent,
	                                            const uint32_t               output_extent,
	                                            const uint32_t               reference_extent,
	                                            const std::vector<uint32_t>& preferred_small_extents) {
		if (source_extent == 0 || output_extent == 0 || reference_extent == 0) {
			return 0;
		}
		const auto target = static_cast<uint32_t>(
		    std::nearbyint((static_cast<double>(source_extent) * static_cast<double>(output_extent)) /
		                   static_cast<double>(reference_extent)));
		if (target <= output_extent && !preferred_small_extents.empty()) {
			uint32_t best      = output_extent;
			uint32_t best_diff = std::numeric_limits<uint32_t>::max();
			for (const auto choice : preferred_small_extents) {
				if (choice == 0 || choice > output_extent) {
					continue;
				}
				const auto diff = choice > target ? choice - target : target - choice;
				if (diff < best_diff) {
					best      = choice;
					best_diff = diff;
				}
			}
			return best;
		}
		auto closest = static_cast<uint32_t>(
		    std::nearbyint(static_cast<double>(target) / static_cast<double>(output_extent)) * output_extent);
		if (closest > source_extent) {
			closest = closest > output_extent ? closest - output_extent : output_extent;
		}
		return std::max<uint32_t>(1U, closest);
	}

	static void validate_grid_transform_spec(const JpegDctGridTransformSpec& spec) {
		if (spec.y_output_width_blocks == 0 || spec.y_output_height_blocks == 0 || spec.cbcr_output_width_blocks == 0 ||
		    spec.cbcr_output_height_blocks == 0 || spec.crop_reference_width_blocks == 0 ||
		    spec.crop_reference_height_blocks == 0) {
			throw std::runtime_error("transformed DCT grid requires non-zero output and crop-reference geometry");
		}
		if (spec.crop_origin_alignment_blocks == 0 || spec.chroma_crop_scale_x == 0 || spec.chroma_crop_scale_y == 0) {
			throw std::runtime_error("transformed DCT grid requires positive crop alignment and chroma scale");
		}
		if (spec.clamp_min < std::numeric_limits<int16_t>::min() ||
		    spec.clamp_max > std::numeric_limits<int16_t>::max() || spec.clamp_min > spec.clamp_max) {
			throw std::runtime_error("transformed DCT grid clamp range must be ordered and fit int16");
		}
		if (!std::isfinite(spec.output_add) || !std::isfinite(spec.output_scale)) {
			throw std::runtime_error("transformed DCT grid FP32 output affine must be finite");
		}
		if (spec.output_data_type == JpegDctGridOutputDataType::kInt16 &&
		    (spec.output_add != 0.0F || spec.output_scale != 1.0F)) {
			throw std::runtime_error("transformed DCT grid int16 output cannot apply an FP32 affine");
		}
		if (!spec.dequantize) {
			throw std::runtime_error("transformed DCT grid executor currently requires dequantize=true");
		}
		if (!spec.require_all_coefficients) {
			throw std::runtime_error("transformed DCT grid executor currently requires all 64 DCT coefficients");
		}
		for (const auto& ratio : spec.allowed_chroma_sampling_ratios) {
			if (ratio.horizontal_numerator == 0 || ratio.horizontal_denominator == 0 || ratio.vertical_numerator == 0 ||
			    ratio.vertical_denominator == 0) {
				throw std::runtime_error("transformed DCT grid sampling ratios must be positive");
			}
		}
	}

	static bool sampling_ratio_allowed(const JpegComponentMetadata&    reference,
	                                   const JpegComponentMetadata&    component,
	                                   const JpegDctGridTransformSpec& spec) {
		return std::any_of(spec.allowed_chroma_sampling_ratios.begin(),
		                   spec.allowed_chroma_sampling_ratios.end(),
		                   [&](const JpegDctSamplingRatio& ratio) {
			                   return static_cast<uint32_t>(component.h_samp_factor) * ratio.horizontal_denominator ==
			                              static_cast<uint32_t>(reference.h_samp_factor) * ratio.horizontal_numerator &&
			                          static_cast<uint32_t>(component.v_samp_factor) * ratio.vertical_denominator ==
			                              static_cast<uint32_t>(reference.v_samp_factor) * ratio.vertical_numerator;
		                   });
	}

	static int32_t floor_div_i32(const int32_t value, const int32_t divisor) {
		if (divisor <= 0) {
			throw std::invalid_argument("floor_div_i32 requires a positive divisor");
		}
		int32_t       quotient = value / divisor;
		const int32_t rem      = value % divisor;
		if (rem != 0 && ((rem < 0) != (divisor < 0))) {
			--quotient;
		}
		return quotient;
	}

	struct DctResizeAxisWeight {
		uint32_t out_block = 0;
		uint32_t out_coeff = 0;
		uint32_t in_block  = 0;
		uint32_t in_coeff  = 0;
		float    weight    = 0.0F;
	};

	struct DctResizeCacheCounters {
		size_t conversion_hits        = 0;
		size_t conversion_misses      = 0;
		size_t resize_weight_hits     = 0;
		size_t resize_weight_misses   = 0;
		double resize_weight_build_ms = 0.0;
	};

	static DctResizeCacheCounters& dct_resize_cache_counters() {
		thread_local DctResizeCacheCounters counters;
		return counters;
	}

	static DctResizeCacheCounters dct_resize_cache_counter_delta(const DctResizeCacheCounters& before,
	                                                             const DctResizeCacheCounters& after) {
		return DctResizeCacheCounters {after.conversion_hits - before.conversion_hits,
		                               after.conversion_misses - before.conversion_misses,
		                               after.resize_weight_hits - before.resize_weight_hits,
		                               after.resize_weight_misses - before.resize_weight_misses,
		                               after.resize_weight_build_ms - before.resize_weight_build_ms};
	}

	static std::vector<float> dct_conversion_matrix_uncached(const uint32_t mult) {
		const uint32_t     n = 8U * mult;
		std::vector<float> large(static_cast<size_t>(n) * n);
		std::vector<float> small(64);
		constexpr double   pi    = 3.141592653589793238462643383279502884;
		const auto         basis = [](const uint32_t rows, const uint32_t u, const uint32_t x) {
            const double scale =
                u == 0 ? std::sqrt(1.0 / static_cast<double>(rows)) : std::sqrt(2.0 / static_cast<double>(rows));
            return static_cast<float>(scale * std::cos((static_cast<double>(u) * (static_cast<double>(x) + 0.5) * pi) /
                                                       static_cast<double>(rows)));
		};
		for (uint32_t u = 0; u < n; ++u) {
			for (uint32_t x = 0; x < n; ++x) {
				large[static_cast<size_t>(u) * n + x] = basis(n, u, x);
			}
		}
		for (uint32_t u = 0; u < 8U; ++u) {
			for (uint32_t x = 0; x < 8U; ++x) {
				small[static_cast<size_t>(u) * 8U + x] = basis(8U, u, x);
			}
		}
		std::vector<float> conversion(static_cast<size_t>(n) * n, 0.0F);
		for (uint32_t out = 0; out < n; ++out) {
			for (uint32_t block = 0; block < mult; ++block) {
				for (uint32_t in = 0; in < 8U; ++in) {
					float sum = 0.0F;
					for (uint32_t x = 0; x < 8U; ++x) {
						sum += large[static_cast<size_t>(out) * n + block * 8U + x] *
						       small[static_cast<size_t>(in) * 8U + x];
					}
					conversion[static_cast<size_t>(out) * n + block * 8U + in] = sum;
				}
			}
		}
		return conversion;
	}

	static const std::vector<float>& dct_conversion_matrix(const uint32_t mult) {
		struct Entry {
			uint32_t           mult = 0;
			std::vector<float> values;
		};
		// References to two different factors are used together while composing
		// a rational resize matrix. deque preserves existing element addresses
		// when another factor is appended; a vector reallocation left the first
		// reference dangling and made cold mixed-ratio planning nondeterministic.
		thread_local std::deque<Entry> cache;
		auto&                          counters = dct_resize_cache_counters();
		for (auto& entry : cache) {
			if (entry.mult == mult) {
				++counters.conversion_hits;
				return entry.values;
			}
		}
		++counters.conversion_misses;
		cache.push_back(Entry {mult, dct_conversion_matrix_uncached(mult)});
		return cache.back().values;
	}

	static std::vector<DctResizeAxisWeight> dct_resize_axis_weights_uncached(const uint32_t source_blocks,
	                                                                         const uint32_t output_blocks) {
		std::vector<DctResizeAxisWeight> weights;
		if (source_blocks == output_blocks) {
			weights.reserve(static_cast<size_t>(output_blocks) * 8U);
			for (uint32_t block = 0; block < output_blocks; ++block) {
				for (uint32_t coeff = 0; coeff < 8U; ++coeff) {
					weights.push_back(DctResizeAxisWeight {block, coeff, block, coeff, 1.0F});
				}
			}
			return weights;
		}
		if (source_blocks > output_blocks && source_blocks % output_blocks == 0) {
			const uint32_t mult = source_blocks / output_blocks;
			const auto     conv = dct_conversion_matrix(mult);
			const float    norm = 1.0F / std::sqrt(static_cast<float>(mult));
			weights.reserve(static_cast<size_t>(output_blocks) * 8U * mult * 8U);
			for (uint32_t out_block = 0; out_block < output_blocks; ++out_block) {
				for (uint32_t out_coeff = 0; out_coeff < 8U; ++out_coeff) {
					for (uint32_t in_subblock = 0; in_subblock < mult; ++in_subblock) {
						for (uint32_t in_coeff = 0; in_coeff < 8U; ++in_coeff) {
							const auto idx = static_cast<size_t>(out_coeff) * (mult * 8U) + in_subblock * 8U + in_coeff;
							weights.push_back(DctResizeAxisWeight {
							    out_block, out_coeff, out_block * mult + in_subblock, in_coeff, conv[idx] * norm});
						}
					}
				}
			}
			return weights;
		}
		if (output_blocks > source_blocks && output_blocks % source_blocks == 0) {
			const uint32_t mult = output_blocks / source_blocks;
			const auto     conv = dct_conversion_matrix(mult);
			const float    norm = std::sqrt(static_cast<float>(mult));
			weights.reserve(static_cast<size_t>(source_blocks) * 8U * mult * 8U);
			for (uint32_t in_block = 0; in_block < source_blocks; ++in_block) {
				for (uint32_t out_subblock = 0; out_subblock < mult; ++out_subblock) {
					for (uint32_t out_coeff = 0; out_coeff < 8U; ++out_coeff) {
						for (uint32_t in_coeff = 0; in_coeff < 8U; ++in_coeff) {
							const uint32_t out_block = in_block * mult + out_subblock;
							const auto     idx =
							    static_cast<size_t>(in_coeff) * (mult * 8U) + out_subblock * 8U + out_coeff;
							weights.push_back(
							    DctResizeAxisWeight {out_block, out_coeff, in_block, in_coeff, conv[idx] * norm});
						}
					}
				}
			}
			return weights;
		}
		throw std::runtime_error("transformed DCT grid exact resize currently requires integer up/downsample factors");
	}

	static std::vector<DctResizeAxisWeight> dct_resize_axis_weights(const uint32_t source_blocks,
	                                                                const uint32_t output_blocks) {
		constexpr size_t kCapacity = 32;
		struct Entry {
			uint32_t                         source_blocks = 0;
			uint32_t                         output_blocks = 0;
			std::vector<DctResizeAxisWeight> values;
		};
		thread_local std::vector<Entry> cache;
		auto&                           counters = dct_resize_cache_counters();
		for (auto& entry : cache) {
			if (entry.source_blocks == source_blocks && entry.output_blocks == output_blocks) {
				++counters.resize_weight_hits;
				return entry.values;
			}
		}
		++counters.resize_weight_misses;
		const auto build_start = std::chrono::steady_clock::now();
		auto       values      = dct_resize_axis_weights_uncached(source_blocks, output_blocks);
		const auto build_end   = std::chrono::steady_clock::now();
		counters.resize_weight_build_ms += std::chrono::duration<double, std::milli>(build_end - build_start).count();
		if (cache.size() >= kCapacity) {
			cache.erase(cache.begin());
		}
		cache.push_back(Entry {source_blocks, output_blocks, std::move(values)});
		return cache.back().values;
	}

	static std::array<float, 64> transformed_dct_grid_axis_matrix(const uint16_t up_factor,
	                                                              const uint16_t down_factor,
	                                                              const uint16_t source_block,
	                                                              const uint16_t output_block) {
		if (up_factor == 0U || down_factor == 0U) {
			throw std::runtime_error("transformed DCT grid resize requires positive rational factors");
		}
		std::array<float, 64> matrix {};
		if (up_factor == 1U && down_factor == 2U) {
			if (source_block / 2U != output_block) {
				return matrix;
			}
			const auto subblock = static_cast<size_t>(source_block % 2U);
			for (size_t out_coeff = 0U; out_coeff < 8U; ++out_coeff) {
				for (size_t in_coeff = 0U; in_coeff < 8U; ++in_coeff) {
					matrix[out_coeff * 8U + in_coeff] =
					    kRgbNoMoreDown2Conversion[out_coeff * 16U + subblock * 8U + in_coeff] /
					    0x1.6a09e60000000p+0F;
				}
			}
			return matrix;
		}
		const auto& up_conversion   = dct_conversion_matrix(up_factor);
		const auto& down_conversion = dct_conversion_matrix(down_factor);
		const auto  axis_weight     = [](const std::vector<float>& conversion,
                                    const uint16_t            factor,
                                    const bool                upsample,
                                    const uint16_t            subblock,
                                    const uint8_t             out_coeff,
                                    const uint8_t             in_coeff) {
            if (factor == 1U) {
                return out_coeff == in_coeff ? 1.0F : 0.0F;
            }
            const auto stride = static_cast<size_t>(factor) * 8U;
            if (upsample) {
                return conversion[static_cast<size_t>(in_coeff) * stride + static_cast<size_t>(subblock) * 8U +
                                  out_coeff] *
                       std::sqrt(static_cast<float>(factor));
            }
            return conversion[static_cast<size_t>(out_coeff) * stride + static_cast<size_t>(subblock) * 8U + in_coeff] /
                   std::sqrt(static_cast<float>(factor));
		};

		const auto            up_start = static_cast<uint32_t>(source_block) * up_factor;
		for (uint8_t out_coeff = 0; out_coeff < 8U; ++out_coeff) {
			for (uint8_t in_coeff = 0; in_coeff < 8U; ++in_coeff) {
				float sum = 0.0F;
				for (uint16_t up_subblock = 0; up_subblock < up_factor; ++up_subblock) {
					const auto upsampled_block = up_start + up_subblock;
					if (upsampled_block / down_factor != output_block) {
						continue;
					}
					const auto down_subblock = static_cast<uint16_t>(upsampled_block % down_factor);
					for (uint8_t mid_coeff = 0; mid_coeff < 8U; ++mid_coeff) {
						sum += axis_weight(up_conversion, up_factor, true, up_subblock, mid_coeff, in_coeff) *
						       axis_weight(down_conversion, down_factor, false, down_subblock, out_coeff, mid_coeff);
					}
				}
				matrix[static_cast<size_t>(out_coeff) * 8U + in_coeff] = sum;
			}
		}
		return matrix;
	}

	// Large ImageNet crops legitimately produce reduced source/output factors
	// above 64 (the strict-1k acceptance workload reaches 145). Phase storage is
	// linear in the two factors; 256 keeps each retained program bounded while
	// avoiding whole-batch fallback to the expanded legacy transform.
	static constexpr uint16_t kMaxPlanlessRationalAxisFactor = 256U;
	// The ImageNet-512 28/14-block contract remains below one MiB even when
	// 4:4:4 layouts are present. Keep a generous but finite ceiling so malformed
	// or exotic layouts cannot turn a prewarm guarantee into an unbounded eager
	// allocation.
	static constexpr size_t kMaxPlanlessAxisCapacityContractBytes = 16U * 1024U * 1024U;

	static bool planless_axis_uses_reference_program(const uint16_t up_factor, const uint16_t down_factor) noexcept {
		return up_factor == 1U && down_factor >= 1U && down_factor <= 2U;
	}

	PlanlessAxisCapacityContract
	planless_axis_capacity_contract(const JpegDctGridTransformSpec& transform) const {
		const PlanlessAxisCapacityKey cache_key {{transform.y_output_width_blocks,
		                                                transform.y_output_height_blocks,
		                                                transform.cbcr_output_width_blocks,
		                                                transform.cbcr_output_height_blocks}};
		{
			const std::lock_guard<std::mutex> lock(planless_axis_capacity_cache_mutex);
			if (const auto found = planless_axis_capacity_cache.find(cache_key);
			    found != planless_axis_capacity_cache.end()) {
				return found->second;
			}
		}

		// Explicit crops can produce any positive component-block extent up to
		// the immutable layout maximum. Enumerating those extents gives a formal
		// workload bound; it does not depend on sample order or warmup coverage.
		std::array<uint32_t, 6> max_component_extents {};
		for (const auto& layout : planless_static_layouts) {
			for (size_t slot = 0U; slot < layout.components.size(); ++slot) {
				const auto& component = layout.components[slot];
				if (!component.present) {
					continue;
				}
				max_component_extents[slot * 2U] =
				    std::max<uint32_t>(max_component_extents[slot * 2U], component.width_in_blocks);
				max_component_extents[slot * 2U + 1U] =
				    std::max<uint32_t>(max_component_extents[slot * 2U + 1U], component.height_in_blocks);
			}
		}

		std::unordered_set<uint32_t> relations;
		const auto collect_relations = [&](const uint32_t max_source_extent, const uint32_t output_extent) {
			if (output_extent == 0U) {
				return;
			}
			for (uint32_t source_extent = 1U; source_extent <= max_source_extent; ++source_extent) {
				const auto divisor = std::gcd(source_extent, output_extent);
				const auto up       = output_extent / divisor;
				const auto down     = source_extent / divisor;
				if (up > kMaxPlanlessRationalAxisFactor || down > kMaxPlanlessRationalAxisFactor ||
				    planless_axis_uses_reference_program(static_cast<uint16_t>(up), static_cast<uint16_t>(down))) {
					continue;
				}
				relations.insert((up << 16U) | down);
			}
		};
		for (size_t slot = 0U; slot < 3U; ++slot) {
			const auto output_width = slot == 0U ? transform.y_output_width_blocks
			                                     : transform.cbcr_output_width_blocks;
			const auto output_height = slot == 0U ? transform.y_output_height_blocks
			                                      : transform.cbcr_output_height_blocks;
			collect_relations(max_component_extents[slot * 2U], output_width);
			collect_relations(max_component_extents[slot * 2U + 1U], output_height);
		}

		PlanlessAxisCapacityContract contract {0U, true};
		constexpr size_t max_float_count = kMaxPlanlessAxisCapacityContractBytes / sizeof(float);
		for (const auto relation : relations) {
			const size_t phase_count = static_cast<size_t>(relation >> 16U) +
			                           static_cast<size_t>(relation & 0xffffU) - 1U;
			if (phase_count > max_float_count / 64U ||
			    contract.float_count > max_float_count - phase_count * 64U) {
				contract = PlanlessAxisCapacityContract {0U, false};
				break;
			}
			contract.float_count += phase_count * 64U;
		}
		{
			const std::lock_guard<std::mutex> lock(planless_axis_capacity_cache_mutex);
			return planless_axis_capacity_cache.emplace(cache_key, contract).first->second;
		}
	}

	static std::vector<float> transformed_dct_grid_axis_phase_matrices_uncached(const uint16_t up_factor,
	                                                                            const uint16_t down_factor) {
		if (up_factor == 0U || down_factor == 0U || std::gcd(up_factor, down_factor) != 1U) {
			throw std::runtime_error("planless transformed DCT axis program requires reduced positive factors");
		}
		// The caller keeps the resulting phase dictionary in its compact runtime
		// plan.  This routine owns only the cold construction path; the bounded
		// cache below prevents repeated batches from rebuilding the much larger
		// temporary conversion matrices.
		const auto conversion_prefix = [](const uint16_t factor) {
			// Planless phase evaluation only addresses DCT rows 0..7.  Building
			// the historical full (8*factor)^2 conversion matrix computed and
			// retained 8*factor rows even though every row after the first eight
			// was dead.  Keep the identical summation order for the live prefix,
			// reducing cold random-crop construction from quadratic to linear
			// storage and work in the rational factor.
			const uint32_t n = 8U * factor;
			std::vector<float> large(static_cast<size_t>(8U) * n);
			std::vector<float> small(64U);
			constexpr double pi = 3.141592653589793238462643383279502884;
			const auto basis = [](const uint32_t rows, const uint32_t u, const uint32_t x) {
				const double scale = u == 0U ? std::sqrt(1.0 / static_cast<double>(rows))
				                               : std::sqrt(2.0 / static_cast<double>(rows));
				return static_cast<float>(
				    scale * std::cos((static_cast<double>(u) * (static_cast<double>(x) + 0.5) * pi) /
				                     static_cast<double>(rows)));
			};
			for (uint32_t u = 0U; u < 8U; ++u) {
				for (uint32_t x = 0U; x < n; ++x) {
					large[static_cast<size_t>(u) * n + x] = basis(n, u, x);
				}
				for (uint32_t x = 0U; x < 8U; ++x) {
					small[static_cast<size_t>(u) * 8U + x] = basis(8U, u, x);
				}
			}
			std::vector<float> conversion(static_cast<size_t>(8U) * n, 0.0F);
			for (uint32_t out = 0U; out < 8U; ++out) {
				for (uint32_t block = 0U; block < factor; ++block) {
					for (uint32_t in = 0U; in < 8U; ++in) {
						float sum = 0.0F;
						for (uint32_t x = 0U; x < 8U; ++x) {
							sum += large[static_cast<size_t>(out) * n + block * 8U + x] *
							       small[static_cast<size_t>(in) * 8U + x];
						}
						conversion[static_cast<size_t>(out) * n + block * 8U + in] = sum;
					}
				}
			}
			return conversion;
		};
		const auto up_conversion   = conversion_prefix(up_factor);
		const auto down_conversion = up_factor == down_factor ? up_conversion : conversion_prefix(down_factor);
		const auto axis_weight = [](const std::vector<float>& conversion,
		                            const uint16_t            factor,
		                            const bool                upsample,
		                            const uint16_t            subblock,
		                            const uint8_t             out_coeff,
		                            const uint8_t             in_coeff) {
			if (factor == 1U) {
				return out_coeff == in_coeff ? 1.0F : 0.0F;
			}
			const auto stride = static_cast<size_t>(factor) * 8U;
			if (upsample) {
				return conversion[static_cast<size_t>(in_coeff) * stride + static_cast<size_t>(subblock) * 8U +
				                  out_coeff] *
				       std::sqrt(static_cast<float>(factor));
			}
			return conversion[static_cast<size_t>(out_coeff) * stride + static_cast<size_t>(subblock) * 8U +
			                  in_coeff] /
			       std::sqrt(static_cast<float>(factor));
		};
		const auto           phase_count = static_cast<size_t>(up_factor) + down_factor - 1U;
		std::vector<float>   matrices(phase_count * 64U, 0.0F);
		std::vector<uint8_t> populated(phase_count, 0U);
		// Over one reduced period (down source blocks -> up output blocks),
		// source*up-output*down visits every contributing relative phase once.
		// The 8x8 matrix is therefore shared by every absolute block pair with
		// that phase; no per-output table is required.
		for (uint16_t source_block = 0U; source_block < down_factor; ++source_block) {
			const auto output_begin = (static_cast<uint32_t>(source_block) * up_factor) / down_factor;
			const auto output_end   = ((static_cast<uint32_t>(source_block + 1U) * up_factor) - 1U) / down_factor;
			for (uint32_t output_block = output_begin; output_block <= output_end; ++output_block) {
				const auto relative_phase =
				    static_cast<int32_t>(source_block) * up_factor - static_cast<int32_t>(output_block) * down_factor;
				const auto phase_index = relative_phase + static_cast<int32_t>(up_factor) - 1;
				if (phase_index < 0 || static_cast<size_t>(phase_index) >= phase_count) {
					throw std::runtime_error("planless transformed DCT axis phase is out of range");
				}
				std::array<float, 64> matrix {};
				const auto up_start = static_cast<uint32_t>(source_block) * up_factor;
				for (uint8_t out_coeff = 0U; out_coeff < 8U; ++out_coeff) {
					for (uint8_t in_coeff = 0U; in_coeff < 8U; ++in_coeff) {
						float sum = 0.0F;
						for (uint16_t up_subblock = 0U; up_subblock < up_factor; ++up_subblock) {
							const auto upsampled_block = up_start + up_subblock;
							if (upsampled_block / down_factor != output_block) {
								continue;
							}
							const auto down_subblock = static_cast<uint16_t>(upsampled_block % down_factor);
							for (uint8_t mid_coeff = 0U; mid_coeff < 8U; ++mid_coeff) {
								sum += axis_weight(up_conversion,
								                   up_factor,
								                   true,
								                   up_subblock,
								                   mid_coeff,
								                   in_coeff) *
								       axis_weight(down_conversion,
								                   down_factor,
								                   false,
								                   down_subblock,
								                   out_coeff,
								                   mid_coeff);
							}
						}
						matrix[static_cast<size_t>(out_coeff) * 8U + in_coeff] = sum;
					}
				}
				std::copy(matrix.begin(), matrix.end(), matrices.begin() + static_cast<size_t>(phase_index) * 64U);
				populated[static_cast<size_t>(phase_index)] = 1U;
			}
		}
		if (std::find(populated.begin(), populated.end(), 0U) != populated.end()) {
			throw std::runtime_error("planless transformed DCT axis phase dictionary is incomplete");
		}
		return matrices;
	}

	static std::vector<float> transformed_dct_grid_axis_phase_matrices(const uint16_t up_factor,
	                                                                  const uint16_t down_factor) {
		// Native prefetch handles may plan successive batches on different worker
		// threads, so this cache must be process-shared rather than thread_local.
		// A 64-image training batch can legitimately contain more than 32 distinct
		// reduced crop ratios; 1024 bounds retained phase programs while covering
		// the observed training working set. Return a value copy so FIFO eviction
		// cannot invalidate a matrix being copied into an in-flight batch plan.
		constexpr size_t kCapacity = 1024U;
		struct Entry {
			uint16_t           up_factor   = 0U;
			uint16_t           down_factor = 0U;
			std::vector<float> values;
		};
		struct Cache {
			std::mutex         mutex;
			std::vector<Entry> entries;
		};
		static Cache cache;
		auto&        counters = dct_resize_cache_counters();
		{
			std::lock_guard<std::mutex> lock(cache.mutex);
			for (const auto& entry : cache.entries) {
				if (entry.up_factor == up_factor && entry.down_factor == down_factor) {
					++counters.resize_weight_hits;
					return entry.values;
				}
			}
		}
		// Do not serialize independent cold ratios behind a large-factor build.
		// A concurrent duplicate is harmless and is resolved under the insertion
		// lock below.
		++counters.resize_weight_misses;
		const auto build_start = std::chrono::steady_clock::now();
		auto       values      = transformed_dct_grid_axis_phase_matrices_uncached(up_factor, down_factor);
		const auto build_end   = std::chrono::steady_clock::now();
		counters.resize_weight_build_ms +=
		    std::chrono::duration<double, std::milli>(build_end - build_start).count();
		std::lock_guard<std::mutex> lock(cache.mutex);
		for (const auto& entry : cache.entries) {
			if (entry.up_factor == up_factor && entry.down_factor == down_factor) {
				return entry.values;
			}
		}
		if (cache.entries.size() >= kCapacity) {
			cache.entries.erase(cache.entries.begin());
		}
		cache.entries.push_back(Entry {up_factor, down_factor, std::move(values)});
		return cache.entries.back().values;
	}

	static uint8_t natural_to_physical_coeff(const uint8_t natural_coeff, const bool zigzag_columns) {
		if (!zigzag_columns) {
			return natural_coeff;
		}
		static constexpr std::array<uint8_t, 64> kNaturalToPhysical {
		    0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42, 3,  8,  12, 17, 25, 30,
		    41, 43, 9,  11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38,
		    46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
		};
		if (natural_coeff >= kNaturalToPhysical.size()) {
			throw std::runtime_error("invalid natural DCT coefficient index");
		}
		return kNaturalToPhysical[natural_coeff];
	}

	static JpegDctRowRef locate_row_in_shard(const ShardState& shard,
	                                         const uint32_t    local_image_index,
	                                         const uint32_t    semantic_slot_id,
	                                         const uint32_t    block_x,
	                                         const uint32_t    block_y) {
		if (local_image_index >= shard.metadata.images.size()) {
			throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
		}

		JpegDctRowRef ref;
		ref.shard_id          = shard.entry.shard_id;
		ref.local_image_index = local_image_index;
		ref.semantic_slot_id  = semantic_slot_id;
		ref.block_x           = block_x;
		ref.block_y           = block_y;
		if (shard.metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
			const auto& image                = shard.metadata.images[local_image_index];
			uint64_t    component_row_offset = 0;
			for (const auto& component : image.components) {
				if (component.semantic_slot_id != semantic_slot_id) {
					if (component.present) {
						component_row_offset +=
						    static_cast<uint64_t>(component.width_in_blocks) * component.height_in_blocks;
					}
					continue;
				}
				if (!component.present || block_x >= component.width_in_blocks ||
				    block_y >= component.height_in_blocks) {
					return ref;
				}
				const auto&    image_group = shard.metadata.image_group_index.at(local_image_index);
				const uint64_t row_offset =
				    component_row_offset + detail::block_order_rank(component.width_in_blocks,
				                                                    component.height_in_blocks,
				                                                    block_x,
				                                                    block_y,
				                                                    shard.metadata.image_major_spatial_order);
				if (row_offset >= image_group.row_count || row_offset > std::numeric_limits<uint32_t>::max()) {
					throw std::runtime_error("JPEG DCT image-major row offset is outside its image record");
				}
				if (shard.vector_rowgroups) {
					ref.fls_rowgroup_index = image_group.fls_rowgroup_index +
					                           static_cast<uint32_t>(row_offset / fastlanes::CFG::VEC_SZ);
					ref.row_start_in_rowgroup     = 0U;
					ref.row_offset_in_block_group = static_cast<uint32_t>(row_offset % fastlanes::CFG::VEC_SZ);
				} else {
					ref.fls_rowgroup_index        = image_group.fls_rowgroup_index;
					ref.row_start_in_rowgroup     = image_group.row_start_in_rowgroup;
					ref.row_offset_in_block_group = static_cast<uint32_t>(row_offset);
				}
				ref.present                   = true;
				ref.physical_row_index        = image_group.row_start + row_offset;
				return ref;
			}
			return ref;
		}

		const auto has_target =
		    image_has_block(shard.metadata.images[local_image_index], semantic_slot_id, block_x, block_y);
		const auto* group = find_group_or_null(shard, semantic_slot_id, block_x, block_y);
		if (group == nullptr) {
			if (!has_target) {
				return ref;
			}
			throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
		}
		ref.fls_rowgroup_index    = group->fls_rowgroup_index;
		ref.row_start_in_rowgroup = group->row_start_in_rowgroup;

		uint32_t rank = 0;
		for (uint32_t image_idx = 0; image_idx < local_image_index; ++image_idx) {
			if (image_has_block(shard.metadata.images[image_idx], semantic_slot_id, block_x, block_y)) {
				++rank;
			}
		}
		ref.row_offset_in_block_group = rank;
		ref.present                   = has_target;
		ref.physical_row_index        = group->row_start + ref.row_offset_in_block_group;
		return ref;
	}

	static uint32_t plan_ragged_rank(const ShardState&             shard,
	                                 const JpegDctBlockGroupIndex& group,
	                                 const uint32_t                local_image_index,
	                                 std::unordered_map<RankCursorKey, RankCursor, RankCursorKeyHash>& rank_cursors) {
		const auto key    = RankCursorKey {shard.entry.shard_id, group_index_in_shard(shard, group)};
		auto&      cursor = rank_cursors[key];
		if (local_image_index < cursor.next_local_image) {
			uint32_t rank = 0;
			for (uint32_t image_idx = 0; image_idx < local_image_index; ++image_idx) {
				if (image_has_block(
				        shard.metadata.images[image_idx], group.semantic_slot_id, group.block_x, group.block_y)) {
					++rank;
				}
			}
			return rank;
		}
		while (cursor.next_local_image < local_image_index) {
			if (image_has_block(shard.metadata.images[cursor.next_local_image],
			                    group.semantic_slot_id,
			                    group.block_x,
			                    group.block_y)) {
				++cursor.present_before_next;
			}
			++cursor.next_local_image;
		}
		return cursor.present_before_next;
	}

	static JpegDctRowRef
	locate_row_in_shard_for_plan(const ShardState&                                                 shard,
	                             const uint32_t                                                    local_image_index,
	                             const uint32_t                                                    semantic_slot_id,
	                             const uint32_t                                                    block_x,
	                             const uint32_t                                                    block_y,
	                             std::unordered_map<RankCursorKey, RankCursor, RankCursorKeyHash>& rank_cursors) {
		if (local_image_index >= shard.metadata.images.size()) {
			throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
		}

		JpegDctRowRef ref;
		ref.shard_id          = shard.entry.shard_id;
		ref.local_image_index = local_image_index;
		ref.semantic_slot_id  = semantic_slot_id;
		ref.block_x           = block_x;
		ref.block_y           = block_y;
		if (shard.metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
			// The image-major mapping is O(components) and independent of all
			// preceding images; no ragged-rank cursor is required.
			return locate_row_in_shard(shard, local_image_index, semantic_slot_id, block_x, block_y);
		}

		const auto has_target =
		    image_has_block(shard.metadata.images[local_image_index], semantic_slot_id, block_x, block_y);
		const auto* group = find_group_or_null(shard, semantic_slot_id, block_x, block_y);
		if (group == nullptr) {
			if (!has_target) {
				return ref;
			}
			throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
		}
		ref.fls_rowgroup_index    = group->fls_rowgroup_index;
		ref.row_start_in_rowgroup = group->row_start_in_rowgroup;

		ref.row_offset_in_block_group = plan_ragged_rank(shard, *group, local_image_index, rank_cursors);
		ref.present                   = has_target;
		ref.physical_row_index        = group->row_start + ref.row_offset_in_block_group;
		return ref;
	}

	[[nodiscard]] size_t planless_shard_index_for(const uint32_t global_image_index) const {
		if (global_image_index >= manifest.image_count || !planless_shard_indices_valid) {
			throw std::runtime_error("JPEG DCT global image index is outside the compact shard index");
		}
		if (planless_uniform_shard_image_count != 0U) {
			const auto shard_index = static_cast<size_t>(global_image_index / planless_uniform_shard_image_count);
			if (shard_index >= planless_static_shards.size()) {
				throw std::runtime_error("JPEG DCT derived compact shard index is invalid");
			}
			return shard_index;
		}
		if (global_image_index >= planless_shard_indices.size()) {
			throw std::runtime_error("JPEG DCT global image index is outside the compact shard lookup");
		}
		return static_cast<size_t>(planless_shard_indices[global_image_index]);
	}

	struct CompiledAccessProfileLookup {
		std::shared_ptr<const std::vector<uint32_t>> selected_chunks;
		bool                                         cache_hit = false;
	};
	struct CompiledAccessProfileKey {
		uint32_t global_image_index = 0;
		uint32_t component_slot     = 0;
		uint32_t crop_x             = 0;
		uint32_t crop_y             = 0;
		uint32_t crop_width         = 0;
		uint32_t crop_height        = 0;

		bool operator==(const CompiledAccessProfileKey&) const = default;
	};
	struct CompiledAccessProfileKeyHash {
		size_t operator()(const CompiledAccessProfileKey& key) const noexcept {
			uint64_t hash = 1469598103934665603ULL;
			const auto mix = [&hash](const uint32_t value) {
				hash ^= value;
				hash *= 1099511628211ULL;
			};
			mix(key.global_image_index);
			mix(key.component_slot);
			mix(key.crop_x);
			mix(key.crop_y);
			mix(key.crop_width);
			mix(key.crop_height);
			return static_cast<size_t>(hash);
		}
	};
	struct CompiledAccessProfileSlot {
		CompiledAccessProfileKey                         key {};
		std::shared_ptr<const std::vector<uint32_t>>     selected_chunks;
	};

	CompiledAccessProfileLookup compiled_access_profile(
	    const uint32_t global_image_index,
	    const size_t component_slot,
	    const uint32_t clipped_x,
	    const uint32_t clipped_y,
	    const uint32_t clipped_width,
	    const uint32_t clipped_height,
	    const PlanlessStaticComponentDescriptor& source,
	    const PlanlessStaticLayoutDescriptor& layout,
	    const uint32_t row_start_in_rowgroup) const {
		if (component_slot > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("JPEG DCT component slot exceeds the compiled access profile key");
		}
		const CompiledAccessProfileKey key {global_image_index,
		                                    static_cast<uint32_t>(component_slot),
		                                    clipped_x,
		                                    clipped_y,
		                                    clipped_width,
		                                    clipped_height};
		const auto direct_slot_index = static_cast<size_t>(global_image_index) * 3U + component_slot;
		if (direct_slot_index >= compiled_fixed_access_profiles.size()) {
			throw std::runtime_error("JPEG DCT compiled access profile slot is outside the dataset index");
		}
		{
			std::lock_guard<std::mutex> lock(compiled_access_profile_mutex);
			const auto& direct = compiled_fixed_access_profiles[direct_slot_index];
			if (direct.selected_chunks && direct.key == key) {
				return {direct.selected_chunks, true};
			}
			const auto found = compiled_access_profiles.find(key);
			if (found != compiled_access_profiles.end()) {
				return {found->second, true};
			}
		}
		auto chunks = std::make_shared<std::vector<uint32_t>>();
		const auto rank_intervals = detail::block_order_rectangle_rank_intervals(
		    source.width_in_blocks,
		    source.height_in_blocks,
		    clipped_x,
		    clipped_y,
		    clipped_width,
		    clipped_height,
		    static_cast<JpegDctSpatialOrder>(layout.spatial_order));
		const uint64_t logical_base = static_cast<uint64_t>(row_start_in_rowgroup) + source.component_row_offset;
		for (const auto& interval : rank_intervals) {
			const auto first_vector = (logical_base + interval.begin) / fastlanes::CFG::VEC_SZ;
			const auto vector_end =
			    (logical_base + interval.end + fastlanes::CFG::VEC_SZ - 1U) / fastlanes::CFG::VEC_SZ;
			const auto vector_stride = layout.vector_rowgroups ? 1U : detail::kJpegDctDeviceUnpackNVectors;
			const auto first_chunk = (first_vector / vector_stride) * vector_stride;
			for (uint64_t vector = first_chunk; vector < vector_end; vector += vector_stride) {
				if (vector >= layout.full_vector_count || vector > std::numeric_limits<uint32_t>::max()) {
					throw std::runtime_error("JPEG DCT compiled access profile exceeds its rowgroup");
				}
				chunks->push_back(static_cast<uint32_t>(vector));
			}
		}
		std::sort(chunks->begin(), chunks->end());
		chunks->erase(std::unique(chunks->begin(), chunks->end()), chunks->end());
		{
			std::lock_guard<std::mutex> lock(compiled_access_profile_mutex);
			auto& direct = compiled_fixed_access_profiles[direct_slot_index];
			if (direct.selected_chunks && direct.key == key) {
				return {direct.selected_chunks, true};
			}
			const auto concurrent = compiled_access_profiles.find(key);
			if (concurrent != compiled_access_profiles.end()) {
				return {concurrent->second, true};
			}
			if (!direct.selected_chunks) {
				direct.key             = key;
				direct.selected_chunks = chunks;
				return {std::move(chunks), false};
			}
			while (compiled_access_profiles.size() >= kCompiledAccessProfileFallbackCapacity &&
			       !compiled_access_profile_order.empty()) {
				compiled_access_profiles.erase(compiled_access_profile_order.front());
				compiled_access_profile_order.pop_front();
			}
			compiled_access_profile_order.push_back(key);
			compiled_access_profiles.emplace(key, chunks);
		}
		return {std::move(chunks), false};
	}

	std::optional<detail::JpegDctDeviceBatchPlan>
	try_block_major_planless_device_batch(const std::vector<JpegDctImageCropRequest>& requests,
	                                      const JpegDctDeviceBatchOptions&            options) const {
		if (!block_major_compact_planner || !options.enable_planless_execution ||
		    options.layout != JpegDctDeviceLayout::kTransformedDctGrid || !options.grid_transform.has_value()) {
			return std::nullopt;
		}
		const auto& transform = *options.grid_transform;
		auto compact = block_major_compact_planner->Plan(requests, transform);
		detail::JpegDctDeviceBatchPlan plan;
		plan.compact_plan_bytes                     = compact.stats.compact_plan_bytes;
		plan.compact_plan_peak_bytes                = compact.stats.compact_plan_peak_bytes;
		plan.canonical_template_hit_count           = compact.stats.canonical_template_hit_count;
		plan.canonical_template_miss_count          = compact.stats.canonical_template_miss_count;
		plan.canonical_template_sidecar_bytes       = compact.stats.canonical_template_sidecar_bytes;
		plan.canonical_template_audit_digest        = compact.stats.canonical_template_audit_digest;
		plan.canonical_template_load_ms             = compact.stats.canonical_template_load_ms;
		plan.canonical_template_validation_ms       = compact.stats.canonical_template_validation_ms;
		plan.duplicate_physical_read_count          = compact.stats.duplicate_physical_read_count;
		plan.rowgroup_revisit_count                 = compact.stats.rowgroup_revisit_count;
		plan.vector_run_revisit_count               = compact.stats.vector_run_revisit_count;
		plan.physical_read_order_inversions         = compact.stats.physical_read_order_inversions;
		plan.layout                        = options.layout;
		plan.grid_transform                = transform;
		plan.unify_rowgroups_across_shards = true;
		plan.uses_planless_fixed_transform = true;
		plan.selected_coefficients         = detail::normalize_coefficient_selection(options.coefficient_selection);
		plan.coefficient_selection_shape   = detail::classify_coefficient_selection(plan.selected_coefficients);
		plan.coefficients_per_block        = plan.selected_coefficients.size();
		if (transform.require_all_coefficients && !detail::selects_all_coefficients(plan.selected_coefficients)) {
			throw std::runtime_error(
			    "block-major planless transformed DCT requires all 64 source coefficients");
		}
		plan.scheduling_policy           = options.scheduling_policy;
		plan.transform_blocks_per_launch = options.transform_blocks_per_launch;
		plan.transform_ctas_per_launch   = options.transform_ctas_per_launch;
		plan.use_low_priority_streams    = options.use_low_priority_streams;
		plan.async_planless_completion  = options.async_planless_completion;
		plan.transform_submission_gate   = options.transform_submission_gate;
		plan.block_major_double_buffer_policy = options.block_major_double_buffer_policy;
		plan.decode_workset_capacity_bytes = options.decode_workset_capacity_bytes == 0U
		                                         ? kDefaultJpegDctDeviceDecodeWorksetCapacityBytes
		                                         : options.decode_workset_capacity_bytes;
		plan.bounded_read_amplification_ppm       = options.bounded_read_amplification_ppm;
		plan.bounded_read_local_amplification_ppm = options.bounded_read_local_amplification_ppm;
		plan.bounded_read_max_run_bytes            = options.bounded_read_max_run_bytes;
		plan.bounded_io_uring_enabled = uses_bounded_io_uring(options.crop_execution_mode);
		plan.bounded_io_uring_queue_depth = plan.bounded_io_uring_enabled
		                                           ? kDefaultJpegDctBoundedIoUringQueueDepth
		                                           : 0U;
		plan.rowgroup_prefetch.enabled   = options.enable_rowgroup_prefetch;
		plan.rowgroup_prefetch.depth = options.rowgroup_prefetch_depth == 0U
		                                   ? kDefaultJpegDctDeviceRowgroupPrefetchDepth
		                                   : options.rowgroup_prefetch_depth;
		plan.rowgroup_prefetch.workers = options.rowgroup_prefetch_workers == 0U
		                                     ? kDefaultJpegDctDeviceRowgroupPrefetchWorkers
		                                     : options.rowgroup_prefetch_workers;
		plan.rowgroup_prefetch.min_decode_batches = options.rowgroup_prefetch_min_decode_batches == 0U
		                                                ? kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches
		                                                : options.rowgroup_prefetch_min_decode_batches;
		plan.image_layouts.resize(requests.size());
		plan.ycbcr_dct_grid_shape.y =
		    {requests.size(), 1U, transform.y_output_height_blocks, transform.y_output_width_blocks, 8U, 8U};
		plan.ycbcr_dct_grid_shape.cbcr =
		    {requests.size(), 2U, transform.cbcr_output_height_blocks, transform.cbcr_output_width_blocks, 8U, 8U};

		struct AxisRelation {
			uint16_t up_factor   = 0U;
			uint16_t down_factor = 0U;
		};
		const auto reduced_axis_relation = [](const uint32_t source_extent,
		                                      const uint32_t output_extent) -> std::optional<AxisRelation> {
			if (source_extent == 0U || output_extent == 0U) {
				return std::nullopt;
			}
			const auto divisor = std::gcd(source_extent, output_extent);
			const auto up      = output_extent / divisor;
			const auto down    = source_extent / divisor;
			if (up > kMaxPlanlessRationalAxisFactor || down > kMaxPlanlessRationalAxisFactor) {
				return std::nullopt;
			}
			return AxisRelation {static_cast<uint16_t>(up), static_cast<uint16_t>(down)};
		};
		std::unordered_map<uint32_t, uint32_t> axis_program_bases;
		const auto axis_program_base = [&](const AxisRelation relation) -> std::optional<uint32_t> {
			if (planless_axis_uses_reference_program(relation.up_factor, relation.down_factor)) {
				return std::numeric_limits<uint32_t>::max();
			}
			const auto key = (static_cast<uint32_t>(relation.up_factor) << 16U) | relation.down_factor;
			if (const auto found = axis_program_bases.find(key); found != axis_program_bases.end()) {
				return found->second;
			}
			const auto& matrices =
			    transformed_dct_grid_axis_phase_matrices(relation.up_factor, relation.down_factor);
			const auto matrix_count  = matrices.size() / 64U;
			const auto current_count = plan.fixed_resize_weight_matrices.size() / 64U;
			if (matrices.size() % 64U != 0U || current_count + matrix_count > std::numeric_limits<uint32_t>::max()) {
				return std::nullopt;
			}
			const auto base = static_cast<uint32_t>(current_count);
			plan.fixed_resize_weight_matrices.insert(
			    plan.fixed_resize_weight_matrices.end(), matrices.begin(), matrices.end());
			axis_program_bases.emplace(key, base);
			++plan.planless_axis_program_count;
			plan.planless_axis_phase_matrix_count += matrix_count;
			return base;
		};
		for (const auto& table : compact.quant_tables) {
			plan.fixed_quant_tables.insert(plan.fixed_quant_tables.end(), table.values.begin(), table.values.end());
		}
		auto device_plan = std::make_shared<detail::JpegDctDeviceBlockMajorPlanlessPlan>();
		device_plan->active_output_schedule_sidecar_enabled =
		    options.crop_execution_mode ==
		    JpegDctCropExecutionMode::kBoundedIoUringScheduledRangeReadSelectedDecode;
		if (device_plan->active_output_schedule_sidecar_enabled) {
			device_plan->active_output_schedule_directory = block_major_access_directory;
			device_plan->canonical_plan_digest = compact.stats.canonical_template_audit_digest;
			if (device_plan->canonical_plan_digest == 0U) {
				throw std::runtime_error("scheduled active-output mode requires a canonical plan audit digest");
			}
		}
		device_plan->images.reserve(compact.requests.size());
		device_plan->groups.reserve(compact.group_bindings.size());
		device_plan->rank_cells.reserve(compact.rank_cells.size());
		device_plan->rank_payload = std::move(compact.rank_payload);
		for (const auto& cell : compact.rank_cells) {
			device_plan->rank_cells.push_back({cell.image_count,
			                                   cell.payload_offset,
			                                   cell.payload_size,
			                                   cell.present_count,
			                                   cell.rank_checkpoint_images,
			                                   static_cast<uint8_t>(cell.encoding)});
		}
		for (const auto& source : compact.group_bindings) {
			device_plan->groups.push_back({source.shard_id,
			                                  source.semantic_slot_id,
			                                  source.block_x,
			                                  source.block_y,
			                                  source.fls_rowgroup_index,
			                                  source.row_start_in_rowgroup,
			                                  source.runtime_rank_cell_index});
		}
		if (!std::is_sorted(device_plan->groups.begin(), device_plan->groups.end(), [](const auto& lhs, const auto& rhs) {
			    return std::tie(lhs.shard_id, lhs.semantic_slot_id, lhs.block_y, lhs.block_x) <
			           std::tie(rhs.shard_id, rhs.semantic_slot_id, rhs.block_y, rhs.block_x);
		    })) {
			throw std::runtime_error("block-major compact group bindings are not in device lookup order");
		}
		detail::build_block_major_coordinate_group_lookup(*device_plan);
		const auto coordinate_lookup_bytes =
		    device_plan->coordinate_group_lookups.capacity() *
		        sizeof(detail::JpegDctDeviceBlockMajorCoordinateGroupLookup) +
		    device_plan->coordinate_group_indices.capacity() * sizeof(uint32_t);
		plan.coordinate_group_lookup_count    = device_plan->coordinate_group_lookups.size();
		plan.coordinate_group_index_entries   = device_plan->coordinate_group_indices.size();
		plan.coordinate_group_index_populated = device_plan->groups.size();
		if (plan.coordinate_group_index_populated > plan.coordinate_group_index_entries) {
			throw std::runtime_error("block-major coordinate lookup population exceeds its storage");
		}
		plan.coordinate_group_index_holes =
		    plan.coordinate_group_index_entries - plan.coordinate_group_index_populated;
		plan.coordinate_group_index_bytes = coordinate_lookup_bytes;
		plan.coordinate_group_index_density = plan.coordinate_group_index_entries == 0U
		                                          ? 0.0
		                                          : static_cast<double>(plan.coordinate_group_index_populated) /
		                                                plan.coordinate_group_index_entries;
		if (coordinate_lookup_bytes > std::numeric_limits<size_t>::max() - plan.compact_plan_bytes ||
		    coordinate_lookup_bytes > std::numeric_limits<size_t>::max() - plan.compact_plan_peak_bytes) {
			throw std::runtime_error("block-major coordinate lookup accounting overflow");
		}
		plan.compact_plan_bytes += coordinate_lookup_bytes;
		plan.compact_plan_peak_bytes += coordinate_lookup_bytes;
		const auto sampling_ratio_allowed = [&](const JpegDctBlockMajorSupportRectangle& reference,
		                                        const JpegDctBlockMajorSupportRectangle& component) {
			return std::any_of(transform.allowed_chroma_sampling_ratios.begin(),
			                   transform.allowed_chroma_sampling_ratios.end(),
			                   [&](const JpegDctSamplingRatio& ratio) {
				                   return static_cast<uint32_t>(component.h_samp_factor) *
				                                  ratio.horizontal_denominator ==
				                              static_cast<uint32_t>(reference.h_samp_factor) *
				                                  ratio.horizontal_numerator &&
				                          static_cast<uint32_t>(component.v_samp_factor) * ratio.vertical_denominator ==
				                              static_cast<uint32_t>(reference.v_samp_factor) * ratio.vertical_numerator;
			                   });
		};
		for (const auto& source_request : compact.requests) {
			const auto& y  = source_request.components[0];
			const auto& cb = source_request.components[1];
			const auto& cr = source_request.components[2];
			if (!y.present || cb.present != cr.present || (!cb.present && !transform.allow_grayscale) ||
			    (cb.present && (!sampling_ratio_allowed(y, cb) || cb.h_samp_factor != cr.h_samp_factor ||
			                    cb.v_samp_factor != cr.v_samp_factor))) {
				return std::nullopt;
			}
			detail::JpegDctDevicePlanlessImageDescriptor image;
			image.request_index     = source_request.output_slot;
			image.shard_id          = source_request.shard_id;
			image.local_image_index = source_request.local_image_index;
			image.horizontal_flip   = source_request.horizontal_flip ? 1U : 0U;
			image.zigzag_columns = shard_by_id(source_request.shard_id).metadata.zigzag_columns ? 1U : 0U;
			uint64_t source_blocks = 0U;
			for (size_t slot = 0U; slot < source_request.components.size(); ++slot) {
				const auto& support = source_request.components[slot];
				if (!support.present) {
					continue;
				}
				const auto output_width = slot == 0U ? transform.y_output_width_blocks
				                                     : transform.cbcr_output_width_blocks;
				const auto output_height = slot == 0U ? transform.y_output_height_blocks
				                                      : transform.cbcr_output_height_blocks;
				const auto x_relation = reduced_axis_relation(support.width, output_width);
				const auto y_relation = reduced_axis_relation(support.height, output_height);
				if (!x_relation.has_value() || !y_relation.has_value()) {
					return std::nullopt;
				}
				const auto x_program = axis_program_base(*x_relation);
				const auto y_program = axis_program_base(*y_relation);
				if (!x_program.has_value() || !y_program.has_value() ||
				    support.quant_table_index >= compact.quant_tables.size()) {
					return std::nullopt;
				}
				auto& component                   = image.components[slot];
				component.semantic_slot_id         = support.semantic_slot_id;
				component.block_major_coordinate_lookup_index =
				    detail::find_block_major_coordinate_group_lookup(
				        *device_plan, source_request.shard_id, support.semantic_slot_id);
				if (component.block_major_coordinate_lookup_index == std::numeric_limits<uint32_t>::max()) {
					throw std::runtime_error("block-major component has no coordinate group lookup");
				}
				component.width_in_blocks          = support.source_width_in_blocks;
				component.height_in_blocks         = support.source_height_in_blocks;
				component.crop_x                   = support.x;
				component.crop_y                   = support.y;
				component.crop_width               = support.width;
				component.crop_height              = support.height;
				component.x_up_factor              = x_relation->up_factor;
				component.y_up_factor              = y_relation->up_factor;
				component.x_down_factor            = x_relation->down_factor;
				component.y_down_factor            = y_relation->down_factor;
				component.x_phase_matrix_base      = *x_program;
				component.y_phase_matrix_base      = *y_program;
				component.quant_table_index        = support.quant_table_index;
				component.present                  = 1U;
				const auto x0 = static_cast<uint32_t>(
				    std::clamp<int64_t>(support.x, 0, support.source_width_in_blocks));
				const auto y0 = static_cast<uint32_t>(
				    std::clamp<int64_t>(support.y, 0, support.source_height_in_blocks));
				const auto x1 = static_cast<uint32_t>(std::clamp<int64_t>(
				    static_cast<int64_t>(support.x) + support.width, 0, support.source_width_in_blocks));
				const auto y1 = static_cast<uint32_t>(std::clamp<int64_t>(
				    static_cast<int64_t>(support.y) + support.height, 0, support.source_height_in_blocks));
				source_blocks += static_cast<uint64_t>(x1 - x0) * (y1 - y0);
				++plan.fixed_transform_component_count;
			}
			if (source_blocks > std::numeric_limits<uint32_t>::max() ||
			    source_request.output_slot >= plan.image_layouts.size()) {
				throw std::runtime_error("block-major compact image descriptor exceeds runtime limits");
			}
			plan.image_layouts[source_request.output_slot] =
			    {source_request.global_image_index, 0U, static_cast<uint32_t>(source_blocks)};
			device_plan->images.push_back(image);
		}

		plan.shards->emplace_back();
		auto& logical_shard = plan.shards->front();
		logical_shard.rowgroups.reserve(compact.rowgroups.size());
		for (const auto& source_rowgroup : compact.rowgroups) {
			const auto& source_shard = shard_by_id(source_rowgroup.shard_id);
			if (source_rowgroup.rowgroup_index >= source_shard.rowgroup_n_tuples.size()) {
				throw std::runtime_error("block-major compact rowgroup exceeds source metadata");
			}
			detail::JpegDctDeviceRowgroupPlan rowgroup;
			rowgroup.rowgroup_index       = source_rowgroup.rowgroup_index;
			rowgroup.source_shard_id      = source_rowgroup.shard_id;
			rowgroup.source_payload_crc64 = source_shard.entry.payload_crc64;
			rowgroup.source_fls_path      = &source_shard.fls_path;
			rowgroup.full_vector_count     = row_count_to_vector_count(
			    source_shard.rowgroup_n_tuples[source_rowgroup.rowgroup_index]);
			rowgroup.has_vector_plan       = true;
			rowgroup.block_major_planless  = device_plan;
			for (uint32_t run_index = 0U; run_index < source_rowgroup.vector_run_count; ++run_index) {
				const auto& run = compact.vector_runs.at(source_rowgroup.first_vector_run + run_index);
				if (run.shard_id != source_rowgroup.shard_id || run.rowgroup_index != source_rowgroup.rowgroup_index) {
					throw std::runtime_error("block-major compact vector run is bound to the wrong rowgroup");
				}
				for (uint32_t vector = run.first_vector; vector < run.first_vector + run.vector_count; ++vector) {
					rowgroup.selected_vectors.push_back(vector);
				}
			}
			if (rowgroup.selected_vectors.empty() ||
			    rowgroup.selected_vectors.back() >= rowgroup.full_vector_count) {
				throw std::runtime_error("block-major compact rowgroup has an invalid vector selection");
			}
			rowgroup.selected_vector_count = rowgroup.selected_vectors.size();
			rowgroup.selected_chunks_fit = detail::selected_decode_chunks_fit(
			    rowgroup.selected_vectors, rowgroup.full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
			configure_crop_execution_mode(rowgroup, options.crop_execution_mode);
			const auto decoded_vectors =
			    rowgroup.runtime_policy.decision == detail::JpegDctRuntimePolicyDecision::kFullRowgroup
			        ? rowgroup.full_vector_count
			        : rowgroup.selected_vector_count;
			constexpr size_t bytes_per_vector = fastlanes::CFG::VEC_SZ * detail::kJpegDctCoefficientCount * sizeof(int16_t);
			if (rowgroup.full_vector_count > std::numeric_limits<size_t>::max() / bytes_per_vector ||
			    decoded_vectors > std::numeric_limits<size_t>::max() / bytes_per_vector - rowgroup.full_vector_count) {
				throw std::runtime_error("block-major rowgroup workset byte estimate overflow");
			}
			rowgroup.estimated_workset_resident_bytes =
			    (rowgroup.full_vector_count + decoded_vectors) * bytes_per_vector;
			if (plan.decode_workset_capacity_bytes != 0U &&
			    rowgroup.estimated_workset_resident_bytes > plan.decode_workset_capacity_bytes) {
				++plan.estimated_oversized_decode_rowgroups;
			}
			plan.planned_selected_vector_count += rowgroup.selected_vector_count;
			plan.estimated_selected_vector_count +=
			    rowgroup.runtime_policy.decision == detail::JpegDctRuntimePolicyDecision::kFullRowgroup
			        ? rowgroup.full_vector_count
			        : rowgroup.selected_vector_count;
			plan.full_vector_count += rowgroup.full_vector_count;
			plan.rowgroups.push_back({source_rowgroup.shard_id, source_rowgroup.rowgroup_index});
			logical_shard.rowgroups.push_back(std::move(rowgroup));
		}
		if (!logical_shard.rowgroups.empty()) {
			logical_shard.shard_id = logical_shard.rowgroups.front().source_shard_id;
			logical_shard.fls_path = logical_shard.rowgroups.front().source_fls_path;
			logical_shard.mixed_physical_shards = std::any_of(
			    logical_shard.rowgroups.begin(), logical_shard.rowgroups.end(), [&](const auto& rowgroup) {
				    return rowgroup.source_shard_id != logical_shard.shard_id;
			    });
		}
		// Block-major worksets are deliberately bounded. Each workset uploads only
		// its resident rowgroup bindings and the output-owned kernel contributes a
		// deterministic linear partial into the zero-initialized accumulation grid.
		// Worksets execute in rowgroup order, so no atomics or per-source partial
		// buffers are required.
		plan.decode_batch_rowgroups = options.decode_batch_rowgroups == 0U
		                                  ? kDefaultJpegDctDecodeBatchRowgroups
		                                  : options.decode_batch_rowgroups;
		{
			size_t pending_bytes = 0U;
			size_t pending_rowgroups = 0U;
			for (const auto& rowgroup : logical_shard.rowgroups) {
				const auto bytes = rowgroup.estimated_workset_resident_bytes;
				const bool count_full = pending_rowgroups >= plan.decode_batch_rowgroups;
				const bool bytes_full = plan.decode_workset_capacity_bytes != 0U && pending_bytes != 0U &&
				                        (pending_bytes >= plan.decode_workset_capacity_bytes ||
				                         bytes > plan.decode_workset_capacity_bytes - pending_bytes);
				if (count_full || bytes_full) {
					plan.estimated_max_decode_workset_bytes =
					    std::max(plan.estimated_max_decode_workset_bytes, pending_bytes);
					pending_bytes     = 0U;
					pending_rowgroups = 0U;
				}
				pending_bytes += bytes;
				++pending_rowgroups;
			}
			plan.estimated_max_decode_workset_bytes =
			    std::max(plan.estimated_max_decode_workset_bytes, pending_bytes);
		}
		for (auto& image_layout : plan.image_layouts) {
			image_layout.block_offset = plan.fixed_transform_source_block_count;
			plan.fixed_transform_source_block_count += image_layout.block_count;
		}
		plan.fixed_transform_output_block_count =
		    (plan.ycbcr_dct_grid_shape.y_count() + plan.ycbcr_dct_grid_shape.cbcr_count()) / 64U;
		plan.planned_saved_vector_count   = plan.full_vector_count - plan.planned_selected_vector_count;
		plan.estimated_saved_vector_count = plan.full_vector_count - plan.estimated_selected_vector_count;
		plan.planned_selected_vector_ratio =
		    plan.full_vector_count == 0U
		        ? 0.0
		        : static_cast<double>(plan.planned_selected_vector_count) / plan.full_vector_count;
		plan.estimated_selected_vector_ratio =
		    plan.full_vector_count == 0U
		        ? 0.0
		        : static_cast<double>(plan.estimated_selected_vector_count) / plan.full_vector_count;
		return plan;
	}

	std::optional<detail::JpegDctDeviceBatchPlan>
	try_planless_device_batch(const std::vector<JpegDctImageCropRequest>& requests,
	                          const JpegDctDeviceBatchOptions&            options) const {
		using PlanningClock = std::chrono::steady_clock;
		const auto planning_elapsed_ms = [](const PlanningClock::time_point begin,
		                                    const PlanningClock::time_point end) {
			return std::chrono::duration<double, std::milli>(end - begin).count();
		};
		const bool image_major_v2 = manifest.version == 2U;
		const bool compact_v3 = manifest.version == 3U && manifest.uses_compact_descriptor() &&
		                        manifest.uses_independent_vector_rowgroups() &&
		                        manifest.physical_layout == "image-major-vector-rowgroups" &&
		                        manifest.vector_size == fastlanes::CFG::VEC_SZ && manifest.rowgroup_vectors == 1U;
		if (!options.enable_planless_execution || options.layout != JpegDctDeviceLayout::kTransformedDctGrid ||
		    !options.grid_transform.has_value() || (!image_major_v2 && !compact_v3) ||
		    !planless_shard_indices_valid) {
			return std::nullopt;
		}
		const auto&                    transform                    = *options.grid_transform;
		const auto                     resize_cache_counters_before = dct_resize_cache_counters();
		detail::JpegDctDeviceBatchPlan plan;
		plan.layout                        = options.layout;
		plan.grid_transform                = transform;
		plan.unify_rowgroups_across_shards = true;
		plan.uses_planless_fixed_transform = true;
		plan.compact_v3_storage             = compact_v3;
		const auto axis_capacity_contract   = planless_axis_capacity_contract(transform);
		plan.planless_axis_program_capacity_contract_count = axis_capacity_contract.float_count;
		plan.planless_axis_program_capacity_contract_complete = axis_capacity_contract.complete;
		if (compact_v3) {
			plan.static_metadata_prewarm_paths = image_major_static_metadata_paths;
		}
		plan.selected_coefficients         = detail::normalize_coefficient_selection(options.coefficient_selection);
		plan.coefficient_selection_shape   = detail::classify_coefficient_selection(plan.selected_coefficients);
		plan.coefficients_per_block        = plan.selected_coefficients.size();
		if (transform.require_all_coefficients && !detail::selects_all_coefficients(plan.selected_coefficients)) {
			throw std::runtime_error(
			    "transformed DCT grid requires dct_coeffs=all because DCT resize uses all 64 source coefficients");
		}
		plan.decode_batch_rowgroups =
		    options.decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : options.decode_batch_rowgroups;
		plan.decode_workset_capacity_bytes = options.decode_workset_capacity_bytes == 0U
		                                         ? kDefaultJpegDctDeviceDecodeWorksetCapacityBytes
		                                         : options.decode_workset_capacity_bytes;
		plan.bounded_read_amplification_ppm       = options.bounded_read_amplification_ppm;
		plan.bounded_read_local_amplification_ppm = options.bounded_read_local_amplification_ppm;
		plan.bounded_read_max_run_bytes            = options.bounded_read_max_run_bytes;
		plan.bounded_io_uring_enabled = uses_bounded_io_uring(options.crop_execution_mode);
		plan.bounded_io_uring_queue_depth = plan.bounded_io_uring_enabled
		                                           ? kDefaultJpegDctBoundedIoUringQueueDepth
		                                           : 0U;
		plan.scheduling_policy           = options.scheduling_policy;
		plan.transform_blocks_per_launch = options.transform_blocks_per_launch;
		plan.transform_ctas_per_launch   = options.transform_ctas_per_launch;
		plan.use_low_priority_streams    = options.use_low_priority_streams;
		plan.async_planless_completion  = options.async_planless_completion;
		plan.transform_submission_gate   = options.transform_submission_gate;
		plan.block_major_double_buffer_policy = options.block_major_double_buffer_policy;
		plan.rowgroup_prefetch.enabled   = options.enable_rowgroup_prefetch;
		plan.rowgroup_prefetch.depth = options.rowgroup_prefetch_depth == 0 ? kDefaultJpegDctDeviceRowgroupPrefetchDepth
		                                                                    : options.rowgroup_prefetch_depth;
		plan.rowgroup_prefetch.workers            = options.rowgroup_prefetch_workers == 0
		                                                ? kDefaultJpegDctDeviceRowgroupPrefetchWorkers
		                                                : options.rowgroup_prefetch_workers;
		plan.rowgroup_prefetch.min_decode_batches = options.rowgroup_prefetch_min_decode_batches == 0
		                                                ? kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches
		                                                : options.rowgroup_prefetch_min_decode_batches;
		plan.image_layouts.resize(requests.size());
		plan.ycbcr_dct_grid_shape.y = {
		    requests.size(), 1U, transform.y_output_height_blocks, transform.y_output_width_blocks, 8U, 8U};
		plan.ycbcr_dct_grid_shape.cbcr = {
		    requests.size(), 2U, transform.cbcr_output_height_blocks, transform.cbcr_output_width_blocks, 8U, 8U};

		struct AxisRelation {
			uint16_t up_factor   = 0U;
			uint16_t down_factor = 0U;
		};
		const auto reduced_axis_relation = [](const uint32_t source_extent,
		                                      const uint32_t output_extent) -> std::optional<AxisRelation> {
			if (source_extent == 0U || output_extent == 0U) {
				return std::nullopt;
			}
			const auto divisor     = std::gcd(source_extent, output_extent);
			const auto up_factor   = output_extent / divisor;
			const auto down_factor = source_extent / divisor;
			if (up_factor > kMaxPlanlessRationalAxisFactor || down_factor > kMaxPlanlessRationalAxisFactor) {
				return std::nullopt;
			}
			return AxisRelation {static_cast<uint16_t>(up_factor), static_cast<uint16_t>(down_factor)};
		};
		const auto supported_axis = [&](const uint32_t crop_extent, const uint32_t output_extent) {
			// RGB-no-more keeps the requested crop geometry and zero-pads source
			// blocks which lie outside a component.  This matters for uncommon
			// sampling layouts such as 4:1:1 and 4:4:0, whose chroma plane can be
			// smaller than the profile's fixed half-resolution crop.
			return reduced_axis_relation(crop_extent, output_extent).has_value();
		};
		const auto sampling_ratio_allowed_compact = [&](const PlanlessStaticComponentDescriptor& reference,
		                                                const PlanlessStaticComponentDescriptor& component) {
			return std::any_of(
			    transform.allowed_chroma_sampling_ratios.begin(),
			    transform.allowed_chroma_sampling_ratios.end(),
			    [&](const JpegDctSamplingRatio& ratio) {
				    return static_cast<uint32_t>(component.h_samp_factor) * ratio.horizontal_denominator ==
				               static_cast<uint32_t>(reference.h_samp_factor) * ratio.horizontal_numerator &&
				           static_cast<uint32_t>(component.v_samp_factor) * ratio.vertical_denominator ==
				               static_cast<uint32_t>(reference.v_samp_factor) * ratio.vertical_numerator;
			    });
		};

		plan.shards->reserve(1U);
		struct RowgroupPlanSlot {
			uint64_t key   = std::numeric_limits<uint64_t>::max();
			size_t   index = 0U;
		};
		size_t rowgroup_slot_target = requests.size();
		if (compact_v3) {
			rowgroup_slot_target = 0U;
			for (const auto& request : requests) {
				if (request.global_image_index >= planless_static_images.size()) {
					throw std::out_of_range("JPEG DCT compact request image index is out of range");
				}
				const auto image_rowgroups = planless_static_images[request.global_image_index].rowgroup_count;
				if (image_rowgroups > std::numeric_limits<size_t>::max() - rowgroup_slot_target) {
					throw std::runtime_error("JPEG DCT compact rowgroup table is too large");
				}
				rowgroup_slot_target += image_rowgroups;
			}
		}
		if (rowgroup_slot_target == std::numeric_limits<size_t>::max()) {
			throw std::runtime_error("JPEG DCT compact rowgroup table is too large");
		}
		size_t rowgroup_slot_count = 2U;
		// The open-addressed table needs one empty sentinel slot.  Sizing to the
		// next power of two above the formal upper bound is sufficient; doubling
		// that bound zero-filled a second, unused table-sized region every batch.
		while (rowgroup_slot_count < rowgroup_slot_target + 1U) {
			if (rowgroup_slot_count > std::numeric_limits<size_t>::max() / 2U) {
				throw std::runtime_error("JPEG DCT compact rowgroup table is too large");
			}
			rowgroup_slot_count *= 2U;
		}
		std::vector<RowgroupPlanSlot> rowgroup_plan_slots(rowgroup_slot_count);
		plan.rowgroups.reserve(rowgroup_slot_target);
		const auto                    rowgroup_plan_slot = [&](const uint64_t key) -> RowgroupPlanSlot& {
            auto hash = key;
            hash      = (hash ^ (hash >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            hash      = (hash ^ (hash >> 27U)) * 0x94d049bb133111ebULL;
            hash ^= hash >> 31U;
            auto slot = static_cast<size_t>(hash) & (rowgroup_slot_count - 1U);
            while (rowgroup_plan_slots[slot].key != std::numeric_limits<uint64_t>::max() &&
                   rowgroup_plan_slots[slot].key != key) {
                slot = (slot + 1U) & (rowgroup_slot_count - 1U);
            }
            return rowgroup_plan_slots[slot];
		};
		std::unordered_map<uint32_t, uint32_t> batch_quant_table_indices;
		batch_quant_table_indices.reserve(std::min(planless_quant_tables.size(), requests.size() * 3U));
		const auto batch_quant_table_index = [&](const uint32_t static_index) {
			const auto found = batch_quant_table_indices.find(static_index);
			if (found != batch_quant_table_indices.end()) {
				return found->second;
			}
			if (static_index >= planless_quant_tables.size()) {
				throw std::runtime_error("JPEG DCT compact quantization-table index is invalid");
			}
			const auto  batch_index = static_cast<uint32_t>(plan.fixed_quant_tables.size() / 64U);
			const auto& values      = planless_quant_tables[static_index];
			plan.fixed_quant_tables.insert(plan.fixed_quant_tables.end(), values.begin(), values.end());
			batch_quant_table_indices.emplace(static_index, batch_index);
			return batch_index;
		};
		std::unordered_map<uint32_t, uint32_t> planless_axis_program_bases;
		planless_axis_program_bases.reserve(6U);
		const auto planless_axis_program_base = [&](const AxisRelation relation) -> std::optional<uint32_t> {
			if (planless_axis_uses_reference_program(relation.up_factor, relation.down_factor)) {
				return std::numeric_limits<uint32_t>::max();
			}
			const auto key = (static_cast<uint32_t>(relation.up_factor) << 16U) | relation.down_factor;
			if (const auto found = planless_axis_program_bases.find(key); found != planless_axis_program_bases.end()) {
				return found->second;
			}
			const auto& matrices =
			    transformed_dct_grid_axis_phase_matrices(relation.up_factor, relation.down_factor);
			const auto  matrix_count  = matrices.size() / 64U;
			const auto  current_count = plan.fixed_resize_weight_matrices.size() / 64U;
			if (matrices.size() % 64U != 0U || current_count + matrix_count > std::numeric_limits<uint32_t>::max()) {
				return std::nullopt;
			}
			const auto base = static_cast<uint32_t>(current_count);
			plan.fixed_resize_weight_matrices.insert(
			    plan.fixed_resize_weight_matrices.end(), matrices.begin(), matrices.end());
			planless_axis_program_bases.emplace(key, base);
			++plan.planless_axis_program_count;
			plan.planless_axis_phase_matrix_count += matrix_count;
			return base;
		};
		plan.shards->emplace_back();
		auto& logical_shard = plan.shards->front();
		logical_shard.rowgroups.reserve(rowgroup_slot_target);
		auto image_major_vector_plan = compact_v3
		                                   ? std::make_shared<detail::JpegDctDeviceImageMajorPlanlessPlan>()
		                                   : std::shared_ptr<detail::JpegDctDeviceImageMajorPlanlessPlan> {};
		if (image_major_vector_plan) {
			image_major_vector_plan->images.reserve(requests.size());
			image_major_vector_plan->vector_sources.reserve(rowgroup_slot_target);
		}
		std::vector<uint64_t> compact_full_image_keys;
		compact_full_image_keys.reserve(requests.size());
		size_t compact_full_vector_count = 0U;
		std::vector<uint32_t> image_selected_vectors;
		image_selected_vectors.reserve(rowgroup_slot_target);
		detail::JpegDctDeviceRowgroupPlan compact_vector_policy;
		if (compact_v3) {
			compact_vector_policy.full_vector_count      = 1U;
			compact_vector_policy.selected_vector_count  = 1U;
			compact_vector_policy.selected_chunks_fit    = true;
			compact_vector_policy.has_vector_plan        = true;
			compact_vector_policy.image_major_planless   = true;
			constexpr size_t bytes_per_vector =
			    fastlanes::CFG::VEC_SZ * detail::kJpegDctCoefficientCount * sizeof(int16_t);
			compact_vector_policy.estimated_workset_resident_bytes = 2U * bytes_per_vector;
			configure_crop_execution_mode(compact_vector_policy, options.crop_execution_mode);
		}

		const auto dynamic_images_begin = PlanningClock::now();
		for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
			const auto crop_geometry_begin = PlanningClock::now();
			const auto& request     = requests[request_index];
			const auto  shard_index = planless_shard_index_for(request.global_image_index);
			if (shard_index >= planless_static_shards.size()) {
				throw std::runtime_error("JPEG DCT compact shard index is invalid");
			}
			const auto& shard = planless_static_shards[shard_index];
			if (request_index == 0U) {
				logical_shard.shard_id = shard.shard_id;
				logical_shard.fls_path = shard.fls_path;
			} else if (logical_shard.shard_id != shard.shard_id) {
				logical_shard.mixed_physical_shards = true;
			}
			const auto local_image_index = request.global_image_index - shard.first_global_image_index;
			if (local_image_index > std::numeric_limits<uint32_t>::max()) {
				return std::nullopt;
			}
			const auto& image_descriptor = planless_static_images[request.global_image_index];
			uint32_t layout_index;
			uint32_t fls_rowgroup_index;
			uint32_t row_start_in_rowgroup;
			if (shard.direct_image_rowgroups) {
				layout_index          = shard.direct_layout_index;
				fls_rowgroup_index    = static_cast<uint32_t>(local_image_index);
				row_start_in_rowgroup = 0U;
			} else {
				layout_index          = image_descriptor.layout_index;
				fls_rowgroup_index    = image_descriptor.fls_rowgroup_index;
				row_start_in_rowgroup = image_descriptor.row_start_in_rowgroup;
			}
			if (layout_index >= planless_static_layouts.size()) {
				throw std::runtime_error("JPEG DCT compact layout index is invalid");
			}
			const auto& layout = planless_static_layouts[layout_index];
			const auto& y      = layout.components[0];
			const auto& cb     = layout.components[1];
			const auto& cr     = layout.components[2];
			if (!layout.image_major || !y.present || y.width_in_blocks == 0U || y.height_in_blocks == 0U ||
			    (cb.present != cr.present) || (!cb.present && !transform.allow_grayscale) ||
			    (cb.present && (cb.h_samp_factor != cr.h_samp_factor || cb.v_samp_factor != cr.v_samp_factor ||
			                    !sampling_ratio_allowed_compact(y, cb)))) {
				return std::nullopt;
			}
			int32_t  y_crop_x      = 0;
			int32_t  y_crop_y      = 0;
			uint32_t y_crop_w      = 0;
			uint32_t y_crop_h      = 0;
			int32_t  chroma_crop_x = 0;
			int32_t  chroma_crop_y = 0;
			uint32_t chroma_crop_w = 0;
			uint32_t chroma_crop_h = 0;
			const bool explicit_crop = request.source_crop.width != 0U && request.source_crop.height != 0U;
			if (explicit_crop) {
				if (image_descriptor.image_width == 0U || image_descriptor.image_height == 0U) {
					throw std::runtime_error("JPEG DCT compact crop requires exact JPEG pixel dimensions");
				}
				auto crop = request.source_crop;
				if (crop.x >= image_descriptor.image_width || crop.y >= image_descriptor.image_height) {
					throw std::out_of_range("JPEG DCT crop starts outside the source image");
				}
				crop.width  = std::min<uint32_t>(crop.width, image_descriptor.image_width - crop.x);
				crop.height = std::min<uint32_t>(crop.height, image_descriptor.image_height - crop.y);
				struct ComponentCrop {
					int32_t  x = 0;
					int32_t  y = 0;
					uint32_t width  = 0;
					uint32_t height = 0;
				};
				const auto component_crop = [&](const PlanlessStaticComponentDescriptor& component) {
					const auto x0 = floor_mul_div_u32(crop.x, component.width_in_blocks, image_descriptor.image_width);
					const auto y0 = floor_mul_div_u32(crop.y, component.height_in_blocks, image_descriptor.image_height);
					const auto x1 = std::min<uint32_t>(
					    component.width_in_blocks,
					    ceil_mul_div_u32(crop.x + crop.width, component.width_in_blocks, image_descriptor.image_width));
					const auto y1 = std::min<uint32_t>(
					    component.height_in_blocks,
					    ceil_mul_div_u32(crop.y + crop.height, component.height_in_blocks, image_descriptor.image_height));
					return ComponentCrop {static_cast<int32_t>(x0),
					                      static_cast<int32_t>(y0),
					                      std::max<uint32_t>(1U, x1 - x0),
					                      std::max<uint32_t>(1U, y1 - y0)};
				};
				const auto y_box = component_crop(y);
				y_crop_x = y_box.x;
				y_crop_y = y_box.y;
				y_crop_w = y_box.width;
				y_crop_h = y_box.height;
				if (cb.present) {
					const auto chroma_box = component_crop(cb);
					chroma_crop_x = chroma_box.x;
					chroma_crop_y = chroma_box.y;
					chroma_crop_w = chroma_box.width;
					chroma_crop_h = chroma_box.height;
				}
			} else {
				y_crop_w = closest_aligned_crop_extent(y.width_in_blocks,
				                                              transform.y_output_width_blocks,
				                                              transform.crop_reference_width_blocks,
				                                              transform.preferred_small_crop_width_blocks);
				y_crop_h = closest_aligned_crop_extent(y.height_in_blocks,
				                                              transform.y_output_height_blocks,
				                                              transform.crop_reference_height_blocks,
				                                              transform.preferred_small_crop_height_blocks);
				y_crop_x = floor_div_i32(static_cast<int32_t>(y.width_in_blocks) - static_cast<int32_t>(y_crop_w), 2);
				y_crop_y = floor_div_i32(static_cast<int32_t>(y.height_in_blocks) - static_cast<int32_t>(y_crop_h), 2);
				const auto alignment = static_cast<int32_t>(transform.crop_origin_alignment_blocks);
				y_crop_x = floor_div_i32(y_crop_x, alignment) * alignment;
				y_crop_y = floor_div_i32(y_crop_y, alignment) * alignment;
				chroma_crop_x = floor_div_i32(y_crop_x, static_cast<int32_t>(transform.chroma_crop_scale_x));
				chroma_crop_y = floor_div_i32(y_crop_y, static_cast<int32_t>(transform.chroma_crop_scale_y));
				chroma_crop_w = std::max<uint32_t>(1U, y_crop_w / transform.chroma_crop_scale_x);
				chroma_crop_h = std::max<uint32_t>(1U, y_crop_h / transform.chroma_crop_scale_y);
			}
			if (!supported_axis(y_crop_w, transform.y_output_width_blocks) ||
			    !supported_axis(y_crop_h, transform.y_output_height_blocks) ||
			    (cb.present &&
			     (!supported_axis(chroma_crop_w, transform.cbcr_output_width_blocks) ||
			      !supported_axis(chroma_crop_h, transform.cbcr_output_height_blocks)))) {
				return std::nullopt;
			}
			plan.crop_geometry_planning_ms +=
			    planning_elapsed_ms(crop_geometry_begin, PlanningClock::now());

			detail::JpegDctDevicePlanlessImageDescriptor launch;
			launch.request_index         = static_cast<uint32_t>(request_index);
			launch.shard_id              = shard.shard_id;
			launch.local_image_index     = static_cast<uint32_t>(local_image_index);
			launch.row_start_in_rowgroup = row_start_in_rowgroup;
			launch.zigzag_columns        = layout.zigzag_columns;
			launch.spatial_order         = layout.spatial_order;
			launch.horizontal_flip       = request.horizontal_flip ? 1U : 0U;
			uint64_t source_blocks = 0;
			image_selected_vectors.clear();
			for (size_t slot = 0; slot < launch.components.size(); ++slot) {
				const auto& source = layout.components[slot];
				if (!source.present) {
					continue;
				}
				const bool luma       = slot == 0U;
				const auto crop_x     = luma ? y_crop_x : chroma_crop_x;
				const auto crop_y     = luma ? y_crop_y : chroma_crop_y;
				const auto crop_w     = luma ? y_crop_w : chroma_crop_w;
				const auto crop_h     = luma ? y_crop_h : chroma_crop_h;
				const auto output_w   = luma ? transform.y_output_width_blocks : transform.cbcr_output_width_blocks;
				const auto output_h   = luma ? transform.y_output_height_blocks : transform.cbcr_output_height_blocks;
				const auto x_relation = reduced_axis_relation(crop_w, output_w);
				const auto y_relation = reduced_axis_relation(crop_h, output_h);
				if (!x_relation.has_value() || !y_relation.has_value()) {
					return std::nullopt;
				}
				const auto axis_program_begin = PlanningClock::now();
				const auto x_program_base = planless_axis_program_base(*x_relation);
				const auto y_program_base = planless_axis_program_base(*y_relation);
				plan.axis_program_planning_ms +=
				    planning_elapsed_ms(axis_program_begin, PlanningClock::now());
				if (!x_program_base.has_value() || !y_program_base.has_value()) {
					return std::nullopt;
				}
				auto& component                = launch.components[slot];
				component.component_row_offset = source.component_row_offset;
				component.width_in_blocks      = source.width_in_blocks;
				component.height_in_blocks     = source.height_in_blocks;
				component.crop_x               = crop_x;
				component.crop_y               = crop_y;
				component.crop_width           = crop_w;
				component.crop_height          = crop_h;
				component.x_up_factor          = x_relation->up_factor;
				component.y_up_factor          = y_relation->up_factor;
				component.x_down_factor        = x_relation->down_factor;
				component.y_down_factor        = y_relation->down_factor;
				component.x_phase_matrix_base  = *x_program_base;
				component.y_phase_matrix_base  = *y_program_base;
				component.quant_table_index    = batch_quant_table_index(source.quant_table_index);
				component.present              = 1U;
				const auto clip_coordinate = [](const int64_t value, const uint32_t extent) {
					return static_cast<uint32_t>(std::clamp<int64_t>(value, 0, extent));
				};
				const auto clipped_x0 = clip_coordinate(crop_x, source.width_in_blocks);
				const auto clipped_y0 = clip_coordinate(crop_y, source.height_in_blocks);
				const auto clipped_x1 =
				    clip_coordinate(static_cast<int64_t>(crop_x) + crop_w, source.width_in_blocks);
				const auto clipped_y1 =
				    clip_coordinate(static_cast<int64_t>(crop_y) + crop_h, source.height_in_blocks);
				if (clipped_x0 < clipped_x1 && clipped_y0 < clipped_y1) {
					const auto interval_begin = PlanningClock::now();
					const auto profile = compiled_access_profile(request.global_image_index,
					                                             slot,
					                                             clipped_x0,
					                                             clipped_y0,
					                                             clipped_x1 - clipped_x0,
					                                             clipped_y1 - clipped_y0,
					                                             source,
					                                             layout,
					                                             row_start_in_rowgroup);
					plan.crop_interval_planning_ms +=
					    planning_elapsed_ms(interval_begin, PlanningClock::now());
					image_selected_vectors.insert(image_selected_vectors.end(),
					                              profile.selected_chunks->begin(),
					                              profile.selected_chunks->end());
					plan.compiled_access_profile_hits += profile.cache_hit ? 1U : 0U;
					plan.compiled_access_profile_misses += profile.cache_hit ? 0U : 1U;
					source_blocks += static_cast<uint64_t>(clipped_x1 - clipped_x0) * (clipped_y1 - clipped_y0);
				}
				++plan.fixed_transform_component_count;
			}

			const auto binding_begin = PlanningClock::now();
			std::sort(image_selected_vectors.begin(), image_selected_vectors.end());
			image_selected_vectors.erase(
			    std::unique(image_selected_vectors.begin(), image_selected_vectors.end()), image_selected_vectors.end());
			if (image_selected_vectors.empty()) {
				throw std::runtime_error("JPEG DCT compact crop selected no source vectors");
			}
			if (layout.vector_rowgroups) {
				if (!image_major_vector_plan || image_descriptor.rowgroup_count != layout.full_vector_count ||
				    image_major_vector_plan->vector_sources.size() > std::numeric_limits<uint32_t>::max() ||
				    layout.full_vector_count > std::numeric_limits<uint32_t>::max() -
				                                   image_major_vector_plan->vector_sources.size()) {
					throw std::runtime_error("JPEG DCT compact image vector binding range is invalid");
				}
				launch.row_start_in_rowgroup = 0U;
				launch.vector_binding_base =
				    static_cast<uint32_t>(image_major_vector_plan->vector_sources.size());
				launch.vector_binding_count = layout.full_vector_count;
				image_major_vector_plan->vector_sources.resize(
				    image_major_vector_plan->vector_sources.size() + layout.full_vector_count);
				const auto full_image_key = (static_cast<uint64_t>(shard_index) << 32U) | fls_rowgroup_index;
				if (std::find(compact_full_image_keys.begin(), compact_full_image_keys.end(), full_image_key) ==
				    compact_full_image_keys.end()) {
					compact_full_image_keys.push_back(full_image_key);
					compact_full_vector_count += layout.full_vector_count;
				}
				for (const auto local_vector : image_selected_vectors) {
					if (local_vector >= layout.full_vector_count) {
						throw std::runtime_error("JPEG DCT compact crop vector exceeds its image");
					}
					const auto physical_rowgroup = fls_rowgroup_index + local_vector;
					auto& source = image_major_vector_plan->vector_sources[
					    static_cast<size_t>(launch.vector_binding_base) + local_vector];
					source = {shard.shard_id, physical_rowgroup};
					const auto rowgroup_key = (static_cast<uint64_t>(shard_index) << 32U) | physical_rowgroup;
					auto&      rowgroup_slot = rowgroup_plan_slot(rowgroup_key);
					if (rowgroup_slot.key == std::numeric_limits<uint64_t>::max()) {
						rowgroup_slot.key   = rowgroup_key;
						rowgroup_slot.index = logical_shard.rowgroups.size();
						logical_shard.rowgroups.emplace_back();
						auto& rowgroup_plan = logical_shard.rowgroups.back();
						rowgroup_plan.rowgroup_index       = physical_rowgroup;
						rowgroup_plan.source_shard_id      = shard.shard_id;
						rowgroup_plan.source_fls_path      = shard.fls_path;
						rowgroup_plan.full_vector_count      = compact_vector_policy.full_vector_count;
						rowgroup_plan.selected_vector_count  = compact_vector_policy.selected_vector_count;
						rowgroup_plan.selected_chunks_fit    = compact_vector_policy.selected_chunks_fit;
						rowgroup_plan.has_vector_plan        = compact_vector_policy.has_vector_plan;
						rowgroup_plan.image_major_planless   = compact_vector_policy.image_major_planless;
						rowgroup_plan.sparse_storage_read    = compact_vector_policy.sparse_storage_read;
						rowgroup_plan.automatic_sparse_storage_candidate =
						    compact_vector_policy.automatic_sparse_storage_candidate;
						rowgroup_plan.runtime_policy = compact_vector_policy.runtime_policy;
						rowgroup_plan.read_strategy  = compact_vector_policy.read_strategy;
						rowgroup_plan.estimated_workset_resident_bytes =
						    compact_vector_policy.estimated_workset_resident_bytes;
						plan.rowgroups.push_back(
						    JpegDctDeviceRowgroupMetadata {shard.shard_id, physical_rowgroup});
					}
				}
				image_major_vector_plan->images.push_back(launch);
			} else {
				const auto rowgroup_key  = (static_cast<uint64_t>(shard_index) << 32U) | fls_rowgroup_index;
				auto&      rowgroup_slot = rowgroup_plan_slot(rowgroup_key);
				if (rowgroup_slot.key == std::numeric_limits<uint64_t>::max()) {
					rowgroup_slot.key   = rowgroup_key;
					rowgroup_slot.index = logical_shard.rowgroups.size();
					detail::JpegDctDeviceRowgroupPlan rowgroup_plan;
					rowgroup_plan.rowgroup_index       = fls_rowgroup_index;
					rowgroup_plan.source_shard_id      = shard.shard_id;
					rowgroup_plan.source_fls_path      = shard.fls_path;
					if (fls_rowgroup_index >= shard.rowgroup_count) {
						throw std::runtime_error("JPEG DCT compact rowgroup exceeds shard metadata");
					}
					rowgroup_plan.full_vector_count = layout.full_vector_count;
					rowgroup_plan.has_vector_plan   = true;
					logical_shard.rowgroups.push_back(std::move(rowgroup_plan));
					plan.rowgroups.push_back(JpegDctDeviceRowgroupMetadata {shard.shard_id, fls_rowgroup_index});
				}
				auto& rowgroup_plan = logical_shard.rowgroups[rowgroup_slot.index];
				rowgroup_plan.planless_images.push_back(launch);
				rowgroup_plan.selected_vectors.insert(rowgroup_plan.selected_vectors.end(),
				                                          image_selected_vectors.begin(),
				                                          image_selected_vectors.end());
			}
			if (source_blocks > std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT compact image source block count overflow");
			}
			plan.image_layouts[request_index] =
			    JpegDctDeviceImageLayout {request.global_image_index, 0U, static_cast<uint32_t>(source_blocks)};
			plan.rowgroup_binding_planning_ms +=
			    planning_elapsed_ms(binding_begin, PlanningClock::now());
		}
		plan.dynamic_image_planning_ms =
		    planning_elapsed_ms(dynamic_images_begin, PlanningClock::now());
		const auto finalize_begin = PlanningClock::now();
		if (!image_major_vector_plan) {
			for (auto& rowgroup_plan : logical_shard.rowgroups) {
				auto& selected_vectors = rowgroup_plan.selected_vectors;
				std::sort(selected_vectors.begin(), selected_vectors.end());
				selected_vectors.erase(
				    std::unique(selected_vectors.begin(), selected_vectors.end()), selected_vectors.end());
				if (selected_vectors.empty()) {
					throw std::runtime_error("JPEG DCT compact crop selected no source vectors");
				}
				const auto planned_selected_vector_count = detail::selected_decode_vector_count(
				    selected_vectors, rowgroup_plan.full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				rowgroup_plan.selected_vector_count = planned_selected_vector_count;
				rowgroup_plan.selected_chunks_fit = detail::selected_decode_chunks_fit(
				    selected_vectors, rowgroup_plan.full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				configure_crop_execution_mode(rowgroup_plan, options.crop_execution_mode);
				plan.planned_selected_vector_count += planned_selected_vector_count;
				plan.estimated_selected_vector_count +=
				    rowgroup_plan.runtime_policy.decision == detail::JpegDctRuntimePolicyDecision::kFullRowgroup
				        ? rowgroup_plan.full_vector_count
				        : planned_selected_vector_count;
				plan.full_vector_count += rowgroup_plan.full_vector_count;
			}
		}
		if (image_major_vector_plan) {
			logical_shard.image_major_planless_owner = image_major_vector_plan;
			plan.full_vector_count               = compact_full_vector_count;
			plan.planned_selected_vector_count   = logical_shard.rowgroups.size();
			plan.estimated_selected_vector_count = logical_shard.rowgroups.size();
			plan.decode_batch_rowgroups          =
			    std::max(plan.decode_batch_rowgroups, logical_shard.rowgroups.size());
			constexpr size_t bytes_per_vector =
			    fastlanes::CFG::VEC_SZ * detail::kJpegDctCoefficientCount * sizeof(int16_t);
			const auto workset_vectors = logical_shard.rowgroups.size();
			if (workset_vectors > std::numeric_limits<size_t>::max() / (2U * bytes_per_vector) ||
			    (plan.decode_workset_capacity_bytes != 0U &&
			     workset_vectors * 2U * bytes_per_vector > plan.decode_workset_capacity_bytes)) {
				return std::nullopt;
			}
			plan.compact_plan_bytes = image_major_vector_plan->images.size() *
			                                 sizeof(detail::JpegDctDevicePlanlessImageDescriptor) +
			                             image_major_vector_plan->vector_sources.size() *
			                                 sizeof(detail::JpegDctDevicePlanlessVectorSource);
			plan.compact_plan_peak_bytes = plan.compact_plan_bytes;
		}
		for (auto& image_layout : plan.image_layouts) {
			image_layout.block_offset = plan.fixed_transform_source_block_count;
			plan.fixed_transform_source_block_count += image_layout.block_count;
		}
		plan.fixed_transform_output_block_count =
		    (plan.ycbcr_dct_grid_shape.y_count() + plan.ycbcr_dct_grid_shape.cbcr_count()) / 64U;
		plan.planned_saved_vector_count = plan.full_vector_count - plan.planned_selected_vector_count;
		plan.estimated_saved_vector_count = plan.full_vector_count - plan.estimated_selected_vector_count;
		plan.planned_selected_vector_ratio =
		    plan.full_vector_count == 0U
		        ? 0.0
		        : static_cast<double>(plan.planned_selected_vector_count) / static_cast<double>(plan.full_vector_count);
		plan.estimated_selected_vector_ratio =
		    plan.full_vector_count == 0U
		        ? 0.0
		        : static_cast<double>(plan.estimated_selected_vector_count) /
		              static_cast<double>(plan.full_vector_count);
		const auto resize_cache_delta =
		    dct_resize_cache_counter_delta(resize_cache_counters_before, dct_resize_cache_counters());
		plan.resize_weight_build_ms             = resize_cache_delta.resize_weight_build_ms;
		plan.dct_resize_weight_cache_hits       = resize_cache_delta.resize_weight_hits;
		plan.dct_resize_weight_cache_misses     = resize_cache_delta.resize_weight_misses;
		plan.dct_conversion_matrix_cache_hits   = resize_cache_delta.conversion_hits;
		plan.dct_conversion_matrix_cache_misses = resize_cache_delta.conversion_misses;
		plan.plan_finalize_ms = planning_elapsed_ms(finalize_begin, PlanningClock::now());
		return plan;
	}

	detail::JpegDctDeviceBatchPlan plan_device_batch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                 const JpegDctDeviceBatchOptions&            options) const {
		if (options.layout != JpegDctDeviceLayout::kImageMajorComponentBlockCoeff &&
		    options.layout != JpegDctDeviceLayout::kYcbcrDctGrid &&
		    options.layout != JpegDctDeviceLayout::kTransformedDctGrid) {
			throw std::runtime_error("unsupported JPEG DCT device output layout");
		}
		const bool output_transformed_dct_grid = options.layout == JpegDctDeviceLayout::kTransformedDctGrid;
		if (output_transformed_dct_grid != options.grid_transform.has_value()) {
			throw std::runtime_error(
			    "transformed DCT grid layout and grid_transform specification must be provided together");
		}
		if (options.grid_transform.has_value()) {
			validate_grid_transform_spec(*options.grid_transform);
		}
		if (auto planless = try_block_major_planless_device_batch(requests, options); planless.has_value()) {
			return std::move(*planless);
		}
		if (auto planless = try_planless_device_batch(requests, options); planless.has_value()) {
			return std::move(*planless);
		}

		detail::JpegDctDeviceBatchPlan plan;
		const auto                     resize_cache_counters_before = dct_resize_cache_counters();
		plan.layout                                                 = options.layout;
		plan.compact_v3_storage                                     = manifest.uses_compact_descriptor();
		// Version 2 stores every image in an independently addressable rowgroup. Shards are only file containers;
		// a random training batch remains one logical decode workset even when its images span multiple files.
		// Legacy spatial-major transformed grids must also remain plan-wide: their expanded source items can
		// contribute to the same output coefficient from different rowgroups. Keeping them split falls back to
		// unordered atomic accumulation and makes the rounded DCT grid vary across identical executions.
		plan.unify_rowgroups_across_shards = manifest.version >= 2U || output_transformed_dct_grid;
		if (options.grid_transform.has_value()) {
			plan.grid_transform = *options.grid_transform;
		}
		plan.selected_coefficients       = detail::normalize_coefficient_selection(options.coefficient_selection);
		plan.coefficient_selection_shape = detail::classify_coefficient_selection(plan.selected_coefficients);
		plan.coefficients_per_block      = plan.selected_coefficients.size();
		if (output_transformed_dct_grid && options.grid_transform->require_all_coefficients &&
		    !detail::selects_all_coefficients(plan.selected_coefficients)) {
			throw std::runtime_error(
			    "transformed DCT grid requires dct_coeffs=all because DCT resize uses all 64 source coefficients");
		}
		const bool output_ycbcr_dct_grid = options.layout == JpegDctDeviceLayout::kYcbcrDctGrid ||
		                                   options.layout == JpegDctDeviceLayout::kTransformedDctGrid;
		const auto* grid_transform = options.grid_transform.has_value() ? &*options.grid_transform : nullptr;
		const bool  materialize_projection_items =
		    output_ycbcr_dct_grid ||
		    plan.coefficient_selection_shape.kind != detail::JpegDctCoefficientSelectionKind::kAll;
		plan.decode_batch_rowgroups =
		    options.decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : options.decode_batch_rowgroups;
		plan.decode_workset_capacity_bytes = options.decode_workset_capacity_bytes == 0U
		                                         ? kDefaultJpegDctDeviceDecodeWorksetCapacityBytes
		                                         : options.decode_workset_capacity_bytes;
		plan.bounded_read_amplification_ppm       = options.bounded_read_amplification_ppm;
		plan.bounded_read_local_amplification_ppm = options.bounded_read_local_amplification_ppm;
		plan.bounded_read_max_run_bytes            = options.bounded_read_max_run_bytes;
		plan.bounded_io_uring_enabled = uses_bounded_io_uring(options.crop_execution_mode);
		plan.bounded_io_uring_queue_depth = plan.bounded_io_uring_enabled
		                                           ? kDefaultJpegDctBoundedIoUringQueueDepth
		                                           : 0U;
		plan.scheduling_policy           = options.scheduling_policy;
		plan.transform_blocks_per_launch = options.transform_blocks_per_launch;
		plan.transform_ctas_per_launch   = options.transform_ctas_per_launch;
		plan.use_low_priority_streams    = options.use_low_priority_streams;
		plan.async_planless_completion  = options.async_planless_completion;
		plan.transform_submission_gate   = options.transform_submission_gate;
		plan.block_major_double_buffer_policy = options.block_major_double_buffer_policy;
		plan.rowgroup_prefetch.enabled   = options.enable_rowgroup_prefetch;
		plan.rowgroup_prefetch.depth = options.rowgroup_prefetch_depth == 0 ? kDefaultJpegDctDeviceRowgroupPrefetchDepth
		                                                                    : options.rowgroup_prefetch_depth;
		plan.rowgroup_prefetch.workers            = options.rowgroup_prefetch_workers == 0
		                                                ? kDefaultJpegDctDeviceRowgroupPrefetchWorkers
		                                                : options.rowgroup_prefetch_workers;
		plan.rowgroup_prefetch.min_decode_batches = options.rowgroup_prefetch_min_decode_batches == 0
		                                                ? kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches
		                                                : options.rowgroup_prefetch_min_decode_batches;
		plan.image_layouts.reserve(requests.size());
		std::unordered_map<RankCursorKey, RankCursor, RankCursorKeyHash> rank_cursors;
		std::unordered_map<uint32_t, size_t>                             shard_plan_indices;
		std::vector<std::unordered_map<uint32_t, size_t>>                rowgroup_plan_indices;
		uint32_t                                                         grid_y_width        = 0;
		uint32_t                                                         grid_y_height       = 0;
		uint32_t                                                         grid_cbcr_width     = 0;
		uint32_t                                                         grid_cbcr_height    = 0;
		bool                                                             grid_y_shape_set    = false;
		bool                                                             grid_cbcr_shape_set = false;

		const auto ensure_grid_component_shape =
		    [&](const uint32_t semantic_slot_id, const uint32_t width, const uint32_t height) {
			    if (!output_ycbcr_dct_grid) {
				    return;
			    }
			    if (width == 0 || height == 0) {
				    throw std::runtime_error("YCbCr DCT grid layout requires non-empty component crops");
			    }
			    auto ensure_shape = [&](bool& shape_set, uint32_t& expected_width, uint32_t& expected_height) {
				    if (!shape_set) {
					    expected_width  = width;
					    expected_height = height;
					    shape_set       = true;
					    return;
				    }
				    if (expected_width != width || expected_height != height) {
					    throw std::runtime_error(
					        "YCbCr DCT grid layout requires every image in the batch to use the same crop-derived "
					        "DCT grid shape");
				    }
			    };
			    if (semantic_slot_id == 0) {
				    ensure_shape(grid_y_shape_set, grid_y_width, grid_y_height);
			    } else if (semantic_slot_id == 1 || semantic_slot_id == 2) {
				    ensure_shape(grid_cbcr_shape_set, grid_cbcr_width, grid_cbcr_height);
			    } else {
				    throw std::runtime_error("YCbCr DCT grid layout supports only semantic slots 0, 1, and 2");
			    }
		    };

		const auto shard_plan_index_for = [&](const ShardState& shard) -> size_t {
			const auto found = shard_plan_indices.find(shard.entry.shard_id);
			if (found != shard_plan_indices.end()) {
				return found->second;
			}
			const size_t                   shard_plan_index = plan.shards->size();
			detail::JpegDctDeviceShardPlan shard_plan;
			shard_plan.shard_id = shard.entry.shard_id;
			shard_plan.fls_path = &shard.fls_path;
			plan.shards->push_back(std::move(shard_plan));
			rowgroup_plan_indices.emplace_back();
			shard_plan_indices.emplace(shard.entry.shard_id, shard_plan_index);
			return shard_plan_index;
		};

		const auto rowgroup_plan_index_for = [&](const size_t   shard_plan_index,
		                                         const uint32_t rowgroup_index) -> size_t {
			auto&      rowgroup_indices = rowgroup_plan_indices[shard_plan_index];
			const auto found            = rowgroup_indices.find(rowgroup_index);
			if (found != rowgroup_indices.end()) {
				return found->second;
			}
			auto&                             shard_plan          = (*plan.shards)[shard_plan_index];
			const size_t                      rowgroup_plan_index = shard_plan.rowgroups.size();
			detail::JpegDctDeviceRowgroupPlan rowgroup_plan;
			rowgroup_plan.rowgroup_index = rowgroup_index;
			shard_plan.rowgroups.push_back(std::move(rowgroup_plan));
			rowgroup_indices.emplace(rowgroup_index, rowgroup_plan_index);
			return rowgroup_plan_index;
		};

		std::unordered_map<uint64_t, uint32_t> fixed_axis_matrix_indices;
		const auto                             fixed_axis_matrix_index_for = [&](const uint16_t up_factor,
                                                     const uint16_t down_factor,
                                                     const uint16_t source_block,
                                                     const uint16_t output_block,
                                                     const bool     horizontal_flip) {
            const auto key = (static_cast<uint64_t>(up_factor) << 48U) | (static_cast<uint64_t>(down_factor) << 32U) |
                             (static_cast<uint64_t>(source_block) << 16U) |
                             (static_cast<uint64_t>(output_block) << 1U) | static_cast<uint64_t>(horizontal_flip);
            const auto found = fixed_axis_matrix_indices.find(key);
            if (found != fixed_axis_matrix_indices.end()) {
                return found->second;
            }
            auto matrix = transformed_dct_grid_axis_matrix(up_factor, down_factor, source_block, output_block);
            if (horizontal_flip) {
                for (uint32_t out_coefficient = 1U; out_coefficient < 8U; out_coefficient += 2U) {
                    for (uint32_t in_coefficient = 0U; in_coefficient < 8U; ++in_coefficient) {
                        matrix[out_coefficient * 8U + in_coefficient] = -matrix[out_coefficient * 8U + in_coefficient];
                    }
                }
            }
            const auto index = static_cast<uint32_t>(plan.fixed_resize_weight_matrices.size() / 64U);
            plan.fixed_resize_weight_matrices.insert(
                plan.fixed_resize_weight_matrices.end(), matrix.begin(), matrix.end());
            fixed_axis_matrix_indices.emplace(key, index);
            return index;
		};

		for (size_t request_idx = 0; request_idx < requests.size(); ++request_idx) {
			const auto& request = requests[request_idx];
			const auto& shard   = shard_for_global_image(request.global_image_index);
			const auto  local_image_index =
			    static_cast<uint32_t>(request.global_image_index - shard.entry.first_global_image_index);
			if (local_image_index >= shard.metadata.images.size()) {
				throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
			}

			const auto&                  image              = shard.metadata.images[local_image_index];
			const auto                   crop               = effective_crop_box(image, request.source_crop);
			const JpegComponentMetadata* y_component        = nullptr;
			const JpegComponentMetadata* fixed_cb_component = nullptr;
			const JpegComponentMetadata* fixed_cr_component = nullptr;
			constexpr uint32_t           kInvalidFixedSlot  = std::numeric_limits<uint32_t>::max();
			uint32_t                     fixed_y_slot       = kInvalidFixedSlot;
			uint32_t                     fixed_cb_slot      = kInvalidFixedSlot;
			uint32_t                     fixed_cr_slot      = kInvalidFixedSlot;
			for (const auto& component : image.components) {
				if (component.semantic_slot_id == 0 && component.present) {
					y_component  = &component;
					fixed_y_slot = component.semantic_slot_id;
				} else if (component.semantic_slot_id == 1 && component.present) {
					fixed_cb_slot      = component.semantic_slot_id;
					fixed_cb_component = &component;
				} else if (component.semantic_slot_id == 2 && component.present) {
					fixed_cr_slot      = component.semantic_slot_id;
					fixed_cr_component = &component;
				}
			}
			if (output_transformed_dct_grid && y_component == nullptr) {
				std::array<const JpegComponentMetadata*, 3> fallback_components {};
				for (const auto& component : image.components) {
					if (!component.present || component.width_in_blocks == 0 || component.height_in_blocks == 0) {
						continue;
					}
					if (component.local_component_index >= fallback_components.size()) {
						continue;
					}
					fallback_components[component.local_component_index] = &component;
				}
				if (fallback_components[0] != nullptr) {
					y_component  = fallback_components[0];
					fixed_y_slot = fallback_components[0]->semantic_slot_id;
				}
				if (fallback_components[1] != nullptr) {
					fixed_cb_slot      = fallback_components[1]->semantic_slot_id;
					fixed_cb_component = fallback_components[1];
				}
				if (fallback_components[2] != nullptr) {
					fixed_cr_slot      = fallback_components[2]->semantic_slot_id;
					fixed_cr_component = fallback_components[2];
				}
			}
			int32_t  fixed_y_x0      = 0;
			int32_t  fixed_y_y0      = 0;
			uint32_t fixed_y_w       = 0;
			uint32_t fixed_y_h       = 0;
			int32_t  fixed_chroma_x0 = 0;
			int32_t  fixed_chroma_y0 = 0;
			uint32_t fixed_chroma_w  = 0;
			uint32_t fixed_chroma_h  = 0;
			if (output_transformed_dct_grid) {
				if (y_component == nullptr || y_component->width_in_blocks == 0 || y_component->height_in_blocks == 0) {
					throw std::runtime_error("transformed DCT grid requires a present Y component");
				}
				if ((fixed_cb_component == nullptr) != (fixed_cr_component == nullptr)) {
					throw std::runtime_error(
					    "transformed DCT grid requires both Cb and Cr components when chroma is present");
				}
				if (fixed_cb_component != nullptr && fixed_cr_component != nullptr) {
					const auto sampling_matches = [&](const JpegComponentMetadata* chroma) {
						return chroma->h_samp_factor == fixed_cb_component->h_samp_factor &&
						       chroma->v_samp_factor == fixed_cb_component->v_samp_factor;
					};
					if (!sampling_matches(fixed_cr_component)) {
						throw std::runtime_error("transformed DCT grid requires matching Cb/Cr sampling factors");
					}
					if (!sampling_ratio_allowed(*y_component, *fixed_cb_component, *grid_transform)) {
						throw std::runtime_error(
						    "transformed DCT grid profile does not allow this chroma sampling ratio");
					}
				} else if (!grid_transform->allow_grayscale) {
					throw std::runtime_error("transformed DCT grid profile does not allow grayscale input");
				}
				fixed_y_w  = closest_aligned_crop_extent(y_component->width_in_blocks,
                                                        grid_transform->y_output_width_blocks,
                                                        grid_transform->crop_reference_width_blocks,
                                                        grid_transform->preferred_small_crop_width_blocks);
				fixed_y_h  = closest_aligned_crop_extent(y_component->height_in_blocks,
                                                        grid_transform->y_output_height_blocks,
                                                        grid_transform->crop_reference_height_blocks,
                                                        grid_transform->preferred_small_crop_height_blocks);
				fixed_y_x0 = floor_div_i32(
				    static_cast<int32_t>(y_component->width_in_blocks) - static_cast<int32_t>(fixed_y_w), 2);
				fixed_y_y0 = floor_div_i32(
				    static_cast<int32_t>(y_component->height_in_blocks) - static_cast<int32_t>(fixed_y_h), 2);
				const auto alignment = static_cast<int32_t>(grid_transform->crop_origin_alignment_blocks);
				fixed_y_x0           = floor_div_i32(fixed_y_x0, alignment) * alignment;
				fixed_y_y0           = floor_div_i32(fixed_y_y0, alignment) * alignment;
				fixed_chroma_x0 = floor_div_i32(fixed_y_x0, static_cast<int32_t>(grid_transform->chroma_crop_scale_x));
				fixed_chroma_y0 = floor_div_i32(fixed_y_y0, static_cast<int32_t>(grid_transform->chroma_crop_scale_y));
				fixed_chroma_w  = std::max<uint32_t>(1U, fixed_y_w / grid_transform->chroma_crop_scale_x);
				fixed_chroma_h  = std::max<uint32_t>(1U, fixed_y_h / grid_transform->chroma_crop_scale_y);
			}

			JpegDctDeviceImageLayout image_layout;
			image_layout.global_image_index = request.global_image_index;
			image_layout.block_offset       = static_cast<uint64_t>(plan.block_metadata.size());
			bool grid_seen_y                = false;
			bool grid_seen_cb               = false;
			bool grid_seen_cr               = false;

			for (const auto& component : image.components) {
				if (!component.present || component.width_in_blocks == 0 || component.height_in_blocks == 0) {
					continue;
				}
				const bool fixed_component_y =
				    output_transformed_dct_grid && component.semantic_slot_id == fixed_y_slot;
				const bool fixed_component_cb =
				    output_transformed_dct_grid && component.semantic_slot_id == fixed_cb_slot;
				const bool fixed_component_cr =
				    output_transformed_dct_grid && component.semantic_slot_id == fixed_cr_slot;
				if (output_transformed_dct_grid && !fixed_component_y && !fixed_component_cb && !fixed_component_cr) {
					continue;
				}
				uint32_t fixed_quant_table_index = 0;
				if (output_transformed_dct_grid) {
					const auto quant_table =
					    std::find_if(image.quant_tables.begin(),
					                 image.quant_tables.end(),
					                 [&](const JpegQuantTableMetadata& candidate) {
						                 return component.quant_tbl_no >= 0 &&
						                        candidate.table_id == static_cast<uint8_t>(component.quant_tbl_no);
					                 });
					if (quant_table == image.quant_tables.end()) {
						throw std::runtime_error(
						    "transformed DCT grid requires the JPEG quantization table for every component");
					}
					fixed_quant_table_index = static_cast<uint32_t>(plan.fixed_quant_tables.size() / 64U);
					plan.fixed_quant_tables.insert(
					    plan.fixed_quant_tables.end(), quant_table->values.begin(), quant_table->values.end());
				}

				uint32_t x0 = std::min(component.width_in_blocks,
				                       floor_mul_div_u32(crop.x, component.width_in_blocks, image.image_width));
				uint32_t y0 = std::min(component.height_in_blocks,
				                       floor_mul_div_u32(crop.y, component.height_in_blocks, image.image_height));
				uint32_t x1 =
				    std::min(component.width_in_blocks,
				             ceil_mul_div_u32(crop.x + crop.width, component.width_in_blocks, image.image_width));
				uint32_t y1 =
				    std::min(component.height_in_blocks,
				             ceil_mul_div_u32(crop.y + crop.height, component.height_in_blocks, image.image_height));
				uint32_t out_w = x1 - x0;
				uint32_t out_h = y1 - y0;
				if (output_transformed_dct_grid) {
					if (fixed_component_y) {
						out_w = grid_transform->y_output_width_blocks;
						out_h = grid_transform->y_output_height_blocks;
					} else if (fixed_component_cb || fixed_component_cr) {
						out_w = grid_transform->cbcr_output_width_blocks;
						out_h = grid_transform->cbcr_output_height_blocks;
					}
				}
				if (!output_transformed_dct_grid) {
					ensure_grid_component_shape(component.semantic_slot_id, out_w, out_h);
				}
				if (output_transformed_dct_grid) {
					if (fixed_component_y) {
						grid_seen_y = true;
					} else if (fixed_component_cb) {
						grid_seen_cb = true;
					} else if (fixed_component_cr) {
						grid_seen_cr = true;
					}
				} else if (output_ycbcr_dct_grid) {
					if (component.semantic_slot_id == 0) {
						grid_seen_y = true;
					} else if (component.semantic_slot_id == 1) {
						grid_seen_cb = true;
					} else if (component.semantic_slot_id == 2) {
						grid_seen_cr = true;
					}
				}
				if (output_transformed_dct_grid) {
					int32_t    crop_x0 = static_cast<int32_t>(x0);
					int32_t    crop_y0 = static_cast<int32_t>(y0);
					uint32_t   crop_w  = x1 - x0;
					uint32_t   crop_h  = y1 - y0;
					const bool explicit_request_crop =
					    request.source_crop.width != 0U && request.source_crop.height != 0U;
					if (fixed_component_y && !explicit_request_crop) {
						crop_x0 = fixed_y_x0;
						crop_y0 = fixed_y_y0;
						crop_w  = std::max<uint32_t>(1U, fixed_y_w);
						crop_h  = std::max<uint32_t>(1U, fixed_y_h);
					} else if ((fixed_component_cb || fixed_component_cr) && !explicit_request_crop) {
						crop_x0 = fixed_chroma_x0;
						crop_y0 = fixed_chroma_y0;
						crop_w  = fixed_chroma_w;
						crop_h  = fixed_chroma_h;
					}
					const auto fixed_relation = [](const uint32_t source, const uint32_t output) {
						struct Relation {
							uint16_t up_factor   = 0;
							uint16_t down_factor = 0;
						};
						if (source == 0U || output == 0U) {
							return Relation {};
						}
						const auto gcd         = std::gcd(source, output);
						const auto up_factor   = output / gcd;
						const auto down_factor = source / gcd;
						if (up_factor > std::numeric_limits<uint16_t>::max() ||
						    down_factor > std::numeric_limits<uint16_t>::max()) {
							return Relation {};
						}
						return Relation {static_cast<uint16_t>(up_factor), static_cast<uint16_t>(down_factor)};
					};
					const auto x_relation = fixed_relation(crop_w, out_w);
					const auto y_relation = fixed_relation(crop_h, out_h);
					const bool use_specialized_fixed_transform =
					    x_relation.up_factor != 0U && x_relation.down_factor != 0U && y_relation.up_factor != 0U &&
					    y_relation.down_factor != 0U;
					if (!use_specialized_fixed_transform) {
						throw std::runtime_error(
						    "transformed DCT grid source-to-output geometry exceeds supported factor range");
					}
					std::unordered_map<uint64_t, JpegDctRowRef> source_refs;
					if (!use_specialized_fixed_transform) {
						source_refs.reserve(static_cast<size_t>(crop_w) * static_cast<size_t>(crop_h));
					}
					for (uint32_t local_y = 0; local_y < crop_h; ++local_y) {
						const int32_t source_y_i = crop_y0 + static_cast<int32_t>(local_y);
						if (source_y_i < 0 || source_y_i >= static_cast<int32_t>(component.height_in_blocks)) {
							continue;
						}
						for (uint32_t local_x = 0; local_x < crop_w; ++local_x) {
							const int32_t source_x_i = crop_x0 + static_cast<int32_t>(local_x);
							if (source_x_i < 0 || source_x_i >= static_cast<int32_t>(component.width_in_blocks)) {
								continue;
							}
							const auto source_x = static_cast<uint32_t>(source_x_i);
							const auto source_y = static_cast<uint32_t>(source_y_i);
							auto       ref      = locate_row_in_shard_for_plan(
                                shard, local_image_index, component.semantic_slot_id, source_x, source_y, rank_cursors);
							if (!ref.present) {
								throw std::runtime_error(
								    "transformed DCT grid source block is missing from shard metadata");
							}
							if (!use_specialized_fixed_transform) {
								const auto source_key =
								    (static_cast<uint64_t>(local_y) << 32U) | static_cast<uint64_t>(local_x);
								source_refs.emplace(source_key, ref);
							}
							const auto metadata_block_index = static_cast<uint64_t>(plan.block_metadata.size());
							plan.block_metadata.push_back(
							    JpegDctDeviceBlockMetadata {static_cast<uint32_t>(request_idx),
							                                request.global_image_index,
							                                component.semantic_slot_id,
							                                source_x,
							                                source_y});
							const auto shard_plan_index = shard_plan_index_for(shard);
							const auto rowgroup_plan_index =
							    rowgroup_plan_index_for(shard_plan_index, ref.fls_rowgroup_index);
							auto& rowgroup_plan = (*plan.shards)[shard_plan_index].rowgroups[rowgroup_plan_index];
							rowgroup_plan.items.push_back(detail::JpegDctDeviceGatherItem {
							    ref.fls_rowgroup_index,
							    ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
							    metadata_block_index});
							if (use_specialized_fixed_transform) {
								const auto output_x0 =
								    (static_cast<uint64_t>(local_x) * x_relation.up_factor) / x_relation.down_factor;
								const auto output_x1 =
								    ((static_cast<uint64_t>(local_x + 1U) * x_relation.up_factor) - 1U) /
								    x_relation.down_factor;
								const auto output_y0 =
								    (static_cast<uint64_t>(local_y) * y_relation.up_factor) / y_relation.down_factor;
								const auto output_y1 =
								    ((static_cast<uint64_t>(local_y + 1U) * y_relation.up_factor) - 1U) /
								    y_relation.down_factor;
								for (uint32_t output_y = static_cast<uint32_t>(output_y0);
								     output_y <= std::min<uint32_t>(static_cast<uint32_t>(output_y1), out_h - 1U);
								     ++output_y) {
									for (uint32_t output_x = static_cast<uint32_t>(output_x0);
									     output_x <= std::min<uint32_t>(static_cast<uint32_t>(output_x1), out_w - 1U);
									     ++output_x) {
										const auto stored_output_x =
										    request.horizontal_flip ? out_w - 1U - output_x : output_x;
										uint64_t output_block_index = 0;
										if (fixed_component_y) {
											output_block_index = (static_cast<uint64_t>(request_idx) *
											                          grid_transform->y_output_height_blocks +
											                      output_y) *
											                         grid_transform->y_output_width_blocks +
											                     stored_output_x;
										} else {
											const auto channel = fixed_component_cb ? 0ULL : 1ULL;
											output_block_index = ((static_cast<uint64_t>(request_idx) * 2U + channel) *
											                          grid_transform->cbcr_output_height_blocks +
											                      output_y) *
											                         grid_transform->cbcr_output_width_blocks +
											                     stored_output_x;
										}
										const auto x_weight_matrix_index =
										    fixed_axis_matrix_index_for(x_relation.up_factor,
										                                x_relation.down_factor,
										                                static_cast<uint16_t>(local_x),
										                                static_cast<uint16_t>(output_x),
										                                request.horizontal_flip);
										const auto y_weight_matrix_index =
										    fixed_axis_matrix_index_for(y_relation.up_factor,
										                                y_relation.down_factor,
										                                static_cast<uint16_t>(local_y),
										                                static_cast<uint16_t>(output_y),
										                                false);
										rowgroup_plan.fixed_transform_items.push_back(
										    detail::JpegDctDeviceFixedTransformItem {
										        ref.fls_rowgroup_index,
										        ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
										        output_block_index,
										        static_cast<uint32_t>(request_idx),
										        static_cast<uint16_t>(local_x),
										        static_cast<uint16_t>(local_y),
										        static_cast<uint16_t>(stored_output_x),
										        static_cast<uint16_t>(output_y),
										        static_cast<uint8_t>(
										            fixed_component_y ? 0U : (fixed_component_cb ? 1U : 2U)),
										        shard.metadata.zigzag_columns,
										        request.horizontal_flip,
										        x_relation.down_factor,
										        y_relation.down_factor,
										        static_cast<uint8_t>(x_relation.up_factor == 1U &&
										                                     x_relation.down_factor == 2U
										                                 ? local_x % 2U
										                                 : 0U),
										        static_cast<uint8_t>(y_relation.up_factor == 1U &&
										                                     y_relation.down_factor == 2U
										                                 ? local_y % 2U
										                                 : 0U),
										        false,
										        false,
										        x_relation.up_factor,
										        y_relation.up_factor,
										        x_relation.down_factor,
										        y_relation.down_factor,
										        fixed_quant_table_index,
										        x_weight_matrix_index,
										        y_weight_matrix_index});
									}
								}
							}
						}
					}
					if (!use_specialized_fixed_transform) {
						const auto y_weights = dct_resize_axis_weights(crop_h, out_h);
						const auto x_weights = dct_resize_axis_weights(crop_w, out_w);
						for (const auto& wy : y_weights) {
							for (const auto& wx : x_weights) {
								const auto source_key =
								    (static_cast<uint64_t>(wy.in_block) << 32U) | static_cast<uint64_t>(wx.in_block);
								const auto found_ref = source_refs.find(source_key);
								if (found_ref == source_refs.end()) {
									continue;
								}
								const auto& ref                    = found_ref->second;
								uint64_t    projection_block_index = 0;
								uint8_t     output_grid_tensor     = 0;
								if (fixed_component_y) {
									projection_block_index =
									    (static_cast<uint64_t>(request_idx) * grid_transform->y_output_height_blocks +
									     wy.out_block) *
									        grid_transform->y_output_width_blocks +
									    wx.out_block;
									output_grid_tensor = kJpegDctYcbcrDctGridTensorY;
								} else {
									const auto channel     = fixed_component_cb ? 0ULL : 1ULL;
									projection_block_index = ((static_cast<uint64_t>(request_idx) * 2U + channel) *
									                              grid_transform->cbcr_output_height_blocks +
									                          wy.out_block) *
									                             grid_transform->cbcr_output_width_blocks +
									                         wx.out_block;
									output_grid_tensor = kJpegDctYcbcrDctGridTensorCbCr;
								}
								const auto source_natural_coeff = static_cast<uint8_t>(wy.in_coeff * 8U + wx.in_coeff);
								const auto output_natural_coeff =
								    static_cast<uint8_t>(wy.out_coeff * 8U + wx.out_coeff);
								const auto source_physical_coeff =
								    natural_to_physical_coeff(source_natural_coeff, shard.metadata.zigzag_columns);
								const auto shard_plan_index = shard_plan_index_for(shard);
								const auto rowgroup_plan_index =
								    rowgroup_plan_index_for(shard_plan_index, ref.fls_rowgroup_index);
								auto& rowgroup_plan = (*plan.shards)[shard_plan_index].rowgroups[rowgroup_plan_index];
								rowgroup_plan.projection_items.push_back(detail::JpegDctDeviceProjectionItem {
								    ref.fls_rowgroup_index,
								    ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
								    projection_block_index,
								    0U,
								    source_physical_coeff,
								    source_physical_coeff,
								    output_natural_coeff,
								    output_grid_tensor,
								    wy.weight * wx.weight});
							}
						}
					}
					continue;
				}
				for (uint32_t out_block_y = 0; out_block_y < out_h; ++out_block_y) {
					for (uint32_t out_block_x = 0; out_block_x < out_w; ++out_block_x) {
						const uint32_t source_w = std::max<uint32_t>(1U, x1 - x0);
						const uint32_t source_h = std::max<uint32_t>(1U, y1 - y0);
						const uint32_t block_x =
						    output_transformed_dct_grid
						        ? std::min<uint32_t>(component.width_in_blocks - 1U,
						                             x0 + floor_mul_div_u32(out_block_x, source_w, out_w))
						        : x0 + out_block_x;
						const uint32_t block_y =
						    output_transformed_dct_grid
						        ? std::min<uint32_t>(component.height_in_blocks - 1U,
						                             y0 + floor_mul_div_u32(out_block_y, source_h, out_h))
						        : y0 + out_block_y;
						auto ref = locate_row_in_shard_for_plan(
						    shard, local_image_index, component.semantic_slot_id, block_x, block_y, rank_cursors);
						if (!ref.present) {
							continue;
						}

						const auto metadata_block_index   = static_cast<uint64_t>(plan.block_metadata.size());
						auto       projection_block_index = metadata_block_index;
						uint8_t    output_grid_tensor     = 0;
						if (output_ycbcr_dct_grid) {
							const auto local_block_y = static_cast<uint64_t>(out_block_y);
							const auto local_block_x = static_cast<uint64_t>(out_block_x);
							if (component.semantic_slot_id == 0) {
								projection_block_index =
								    (static_cast<uint64_t>(request_idx) * grid_y_height + local_block_y) *
								        grid_y_width +
								    local_block_x;
								output_grid_tensor = kJpegDctYcbcrDctGridTensorY;
							} else {
								const auto channel = static_cast<uint64_t>(component.semantic_slot_id - 1U);
								projection_block_index =
								    ((static_cast<uint64_t>(request_idx) * 2U + channel) * grid_cbcr_height +
								     local_block_y) *
								        grid_cbcr_width +
								    local_block_x;
								output_grid_tensor = kJpegDctYcbcrDctGridTensorCbCr;
							}
						}
						plan.block_metadata.push_back(JpegDctDeviceBlockMetadata {static_cast<uint32_t>(request_idx),
						                                                          request.global_image_index,
						                                                          component.semantic_slot_id,
						                                                          block_x,
						                                                          block_y});
						const auto shard_plan_index = shard_plan_index_for(shard);
						const auto rowgroup_plan_index =
						    rowgroup_plan_index_for(shard_plan_index, ref.fls_rowgroup_index);
						auto& rowgroup_plan = (*plan.shards)[shard_plan_index].rowgroups[rowgroup_plan_index];
						rowgroup_plan.items.push_back(
						    detail::JpegDctDeviceGatherItem {ref.fls_rowgroup_index,
						                                     ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
						                                     metadata_block_index});
						if (materialize_projection_items) {
							const auto row_in_rowgroup = ref.row_start_in_rowgroup + ref.row_offset_in_block_group;
							for (size_t coeff_slot = 0; coeff_slot < plan.selected_coefficients.size(); ++coeff_slot) {
								const auto logical_coeff = plan.selected_coefficients[coeff_slot];
								const auto output_coeff  = output_ycbcr_dct_grid && shard.metadata.zigzag_columns
								                               ? detail::kZigzagColumnToNaturalIndex[logical_coeff]
								                               : logical_coeff;
								rowgroup_plan.projection_items.push_back(
								    detail::JpegDctDeviceProjectionItem {ref.fls_rowgroup_index,
								                                         row_in_rowgroup,
								                                         projection_block_index,
								                                         static_cast<uint16_t>(coeff_slot),
								                                         logical_coeff,
								                                         logical_coeff,
								                                         output_coeff,
								                                         output_grid_tensor});
							}
						}
					}
				}
			}
			if (output_ycbcr_dct_grid && !grid_seen_y) {
				throw std::runtime_error("YCbCr DCT grid layout requires a Y component per image");
			}
			if (output_ycbcr_dct_grid && !output_transformed_dct_grid && (!grid_seen_cb || !grid_seen_cr)) {
				throw std::runtime_error("YCbCr DCT grid layout requires Y, Cb, and Cr components per image");
			}

			image_layout.block_count =
			    static_cast<uint32_t>(static_cast<uint64_t>(plan.block_metadata.size()) - image_layout.block_offset);
			plan.image_layouts.push_back(image_layout);
		}
		if (output_ycbcr_dct_grid) {
			if (output_transformed_dct_grid) {
				plan.ycbcr_dct_grid_shape.y    = {requests.size(),
				                                  1U,
				                                  grid_transform->y_output_height_blocks,
				                                  grid_transform->y_output_width_blocks,
				                                  8U,
				                                  8U};
				plan.ycbcr_dct_grid_shape.cbcr = {requests.size(),
				                                  2U,
				                                  grid_transform->cbcr_output_height_blocks,
				                                  grid_transform->cbcr_output_width_blocks,
				                                  8U,
				                                  8U};
			} else {
				if (!requests.empty() && (!grid_y_shape_set || !grid_cbcr_shape_set)) {
					throw std::runtime_error("YCbCr DCT grid layout could not determine Y/CbCr grid shapes");
				}
				plan.ycbcr_dct_grid_shape.y    = {requests.size(), 1U, grid_y_height, grid_y_width, 8U, 8U};
				plan.ycbcr_dct_grid_shape.cbcr = {requests.size(), 2U, grid_cbcr_height, grid_cbcr_width, 8U, 8U};
			}
		}

		std::unordered_set<uint64_t> fixed_transform_component_keys;
		std::unordered_set<uint64_t> fixed_transform_output_block_keys;
		if (output_transformed_dct_grid) {
			fixed_transform_component_keys.reserve(requests.size() * 3U);
			fixed_transform_output_block_keys.reserve(
			    (plan.ycbcr_dct_grid_shape.y_count() + plan.ycbcr_dct_grid_shape.cbcr_count()) / 64U);
		}
		for (auto& shard_plan : *plan.shards) {
			std::sort(shard_plan.rowgroups.begin(), shard_plan.rowgroups.end(), [](const auto& lhs, const auto& rhs) {
				return lhs.rowgroup_index < rhs.rowgroup_index;
			});
			const auto& shard = shard_by_id(shard_plan.shard_id);
			for (auto& rowgroup_plan : shard_plan.rowgroups) {
				plan.rowgroups.push_back(
				    JpegDctDeviceRowgroupMetadata {shard_plan.shard_id, rowgroup_plan.rowgroup_index});
				if (rowgroup_plan.rowgroup_index >= shard.rowgroup_n_tuples.size()) {
					throw std::runtime_error("JPEG DCT planned rowgroup exceeds shard rowgroup metadata");
				}
				if (output_transformed_dct_grid) {
					std::unordered_set<uint32_t> source_rows;
					source_rows.reserve(std::min(rowgroup_plan.fixed_transform_items.size(),
					                             shard.rowgroup_n_tuples[rowgroup_plan.rowgroup_index]));
					for (const auto& item : rowgroup_plan.fixed_transform_items) {
						fixed_transform_component_keys.insert((static_cast<uint64_t>(item.image_index) << 8U) |
						                                      item.component);
						source_rows.insert(item.row_in_rowgroup);
						fixed_transform_output_block_keys.insert((static_cast<uint64_t>(item.image_index) << 40U) |
						                                         (static_cast<uint64_t>(item.component) << 32U) |
						                                         (static_cast<uint64_t>(item.output_block_y) << 16U) |
						                                         item.output_block_x);
					}
					plan.fixed_transform_source_block_count += source_rows.size();
				}
				const auto full_vector_count =
				    row_count_to_vector_count(shard.rowgroup_n_tuples[rowgroup_plan.rowgroup_index]);
				auto selected_vectors = detail::selected_decode_vectors(
				    rowgroup_plan.items, full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				const auto planned_selected_vector_count = detail::selected_decode_vector_count(
				    selected_vectors, full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				const auto selected_chunks_fit = detail::selected_decode_chunks_fit(
				    selected_vectors, full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				rowgroup_plan.selected_vector_count = planned_selected_vector_count;
				rowgroup_plan.full_vector_count     = full_vector_count;
				rowgroup_plan.selected_chunks_fit   = selected_chunks_fit;
				rowgroup_plan.has_vector_plan       = true;
				configure_crop_execution_mode(rowgroup_plan, options.crop_execution_mode);
				const auto runtime_policy = rowgroup_plan.runtime_policy;
				// Automatic physical-I/O refinement may promote a logically-full
				// preview to selected decode after real storage costs are known.  Build
				// both execution representations for that explicit candidate now; the
				// compiled-I/O phase chooses which representation becomes active.
				const bool prepare_selected_decode =
				    runtime_policy.decision == detail::JpegDctRuntimePolicyDecision::kSelectedVectors ||
				    rowgroup_plan.automatic_sparse_storage_candidate;
				if (prepare_selected_decode) {
					rowgroup_plan.selected_gather_items = detail::remap_items_to_selected_vectors(
					    rowgroup_plan.items, selected_vectors, detail::kJpegDctDeviceUnpackNVectors);
					if (materialize_projection_items) {
						rowgroup_plan.selected_projection_items = detail::remap_projection_items_to_selected_vectors(
						    rowgroup_plan.projection_items, selected_vectors, detail::kJpegDctDeviceUnpackNVectors);
					}
					if (output_transformed_dct_grid) {
						rowgroup_plan.selected_fixed_transform_items =
						    detail::remap_fixed_transform_items_to_selected_vectors(
						        rowgroup_plan.fixed_transform_items,
						        selected_vectors,
						        detail::kJpegDctDeviceUnpackNVectors);
					}
				}
				rowgroup_plan.selected_vectors = std::move(selected_vectors);
				plan.planned_selected_vector_count += planned_selected_vector_count;
				plan.estimated_selected_vector_count +=
				    runtime_policy.decision == detail::JpegDctRuntimePolicyDecision::kFullRowgroup
				        ? full_vector_count
				        : planned_selected_vector_count;
				plan.full_vector_count += full_vector_count;
			}
		}
		if (output_transformed_dct_grid) {
			struct OrderedFixedTransformItem {
				uint32_t source_index       = 0;
				uint8_t  output_tensor      = 0;
				uint64_t output_block_index = 0;
				uint16_t local_block_y      = 0;
				uint16_t local_block_x      = 0;
			};
			size_t fixed_item_count = 0;
			for (const auto& shard_plan : *plan.shards) {
				for (const auto& rowgroup_plan : shard_plan.rowgroups) {
					fixed_item_count += rowgroup_plan.fixed_transform_items.size();
				}
			}
			if (fixed_item_count > std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT fixed transform batch exceeds deterministic index range");
			}
			plan.host_expanded_transform_items_created = fixed_item_count;
			plan.host_global_transform_sort_items      = fixed_item_count;
			std::vector<OrderedFixedTransformItem> ordered_items;
			ordered_items.reserve(fixed_item_count);
			uint32_t source_index = 0;
			for (const auto& shard_plan : *plan.shards) {
				for (const auto& rowgroup_plan : shard_plan.rowgroups) {
					for (const auto& item : rowgroup_plan.fixed_transform_items) {
						ordered_items.push_back(
						    OrderedFixedTransformItem {source_index++,
						                               static_cast<uint8_t>(item.component == 0U ? 0U : 1U),
						                               item.output_block_index,
						                               item.local_block_y,
						                               item.local_block_x});
					}
				}
			}
			std::stable_sort(ordered_items.begin(), ordered_items.end(), [](const auto& lhs, const auto& rhs) {
				return std::tie(lhs.output_tensor, lhs.output_block_index, lhs.local_block_y, lhs.local_block_x) <
				       std::tie(rhs.output_tensor, rhs.output_block_index, rhs.local_block_y, rhs.local_block_x);
			});
			auto& item_order    = *plan.fixed_transform_item_order;
			auto& group_offsets = *plan.fixed_transform_group_offsets;
			item_order.resize(ordered_items.size());
			group_offsets.reserve(fixed_transform_output_block_keys.size() + 1U);
			for (size_t ordered_index = 0; ordered_index < ordered_items.size(); ++ordered_index) {
				const auto& item = ordered_items[ordered_index];
				if (ordered_index == 0U || item.output_tensor != ordered_items[ordered_index - 1U].output_tensor ||
				    item.output_block_index != ordered_items[ordered_index - 1U].output_block_index) {
					group_offsets.push_back(static_cast<uint32_t>(ordered_index));
				}
				item_order[item.source_index] = static_cast<uint32_t>(ordered_index);
			}
			if (!item_order.empty()) {
				group_offsets.push_back(static_cast<uint32_t>(item_order.size()));
			}
			plan.host_output_block_source_lists_created = group_offsets.empty() ? 0U : group_offsets.size() - 1U;
		}
		plan.fixed_transform_component_count    = fixed_transform_component_keys.size();
		plan.fixed_transform_output_block_count = fixed_transform_output_block_keys.size();
		plan.planned_saved_vector_count         = plan.full_vector_count - plan.planned_selected_vector_count;
		plan.estimated_saved_vector_count       = plan.full_vector_count - plan.estimated_selected_vector_count;
		plan.planned_selected_vector_ratio =
		    plan.full_vector_count == 0
		        ? 0.0
		        : static_cast<double>(plan.planned_selected_vector_count) / static_cast<double>(plan.full_vector_count);
		plan.estimated_selected_vector_ratio = plan.full_vector_count == 0
		                                           ? 0.0
		                                           : static_cast<double>(plan.estimated_selected_vector_count) /
		                                                 static_cast<double>(plan.full_vector_count);
		const auto resize_cache_delta =
		    dct_resize_cache_counter_delta(resize_cache_counters_before, dct_resize_cache_counters());
		plan.resize_weight_build_ms             = resize_cache_delta.resize_weight_build_ms;
		plan.dct_resize_weight_cache_hits       = resize_cache_delta.resize_weight_hits;
		plan.dct_resize_weight_cache_misses     = resize_cache_delta.resize_weight_misses;
		plan.dct_conversion_matrix_cache_hits   = resize_cache_delta.conversion_hits;
		plan.dct_conversion_matrix_cache_misses = resize_cache_delta.conversion_misses;
		return plan;
	}

	static std::string device_plan_cache_key(const std::vector<JpegDctImageCropRequest>& requests,
	                                         const JpegDctDeviceBatchOptions&            options) {
		std::ostringstream key;
		key << static_cast<int>(options.layout) << ':' << options.decode_batch_rowgroups << ':'
		    << options.decode_workset_capacity_bytes << ':'
		    << options.enable_rowgroup_prefetch << ':' << options.rowgroup_prefetch_depth << ':'
		    << options.rowgroup_prefetch_workers << ':' << options.rowgroup_prefetch_min_decode_batches << ':'
		    << options.enable_planless_execution << ':' << static_cast<int>(options.scheduling_policy) << ':'
		    << options.transform_blocks_per_launch << ':' << options.transform_ctas_per_launch << ':'
		    << options.use_low_priority_streams << ':' << static_cast<int>(options.crop_execution_mode) << ':'
		    << options.bounded_read_amplification_ppm << ':'
		    << options.bounded_read_local_amplification_ppm << ':'
		    << options.bounded_read_max_run_bytes << ':';
		if (options.grid_transform.has_value()) {
			const auto& spec = *options.grid_transform;
			key << spec.y_output_width_blocks << ',' << spec.y_output_height_blocks << ','
			    << spec.cbcr_output_width_blocks << ',' << spec.cbcr_output_height_blocks << ','
			    << spec.crop_reference_width_blocks << ',' << spec.crop_reference_height_blocks << ','
			    << spec.crop_origin_alignment_blocks << ',' << spec.chroma_crop_scale_x << ','
			    << spec.chroma_crop_scale_y << ',' << spec.clamp_min << ',' << spec.clamp_max << ','
			    << static_cast<int>(spec.output_data_type) << ',' << std::hexfloat << spec.output_add << ','
			    << spec.output_scale << std::defaultfloat << ',' << spec.dequantize << ','
			    << spec.require_all_coefficients << ',' << spec.allow_grayscale << ':';
			for (const auto value : spec.preferred_small_crop_width_blocks) {
				key << value << ',';
			}
			key << ':';
			for (const auto value : spec.preferred_small_crop_height_blocks) {
				key << value << ',';
			}
			key << ':';
			for (const auto& ratio : spec.allowed_chroma_sampling_ratios) {
				key << ratio.horizontal_numerator << '/' << ratio.horizontal_denominator << ','
				    << ratio.vertical_numerator << '/' << ratio.vertical_denominator << ';';
			}
		} else {
			key << "none";
		}
		key << ':';
		const auto coefficients = detail::normalize_coefficient_selection(options.coefficient_selection);
		for (const auto coefficient : coefficients) {
			key << static_cast<unsigned>(coefficient) << ',';
		}
		key << ';';
		for (const auto& request : requests) {
			key << request.global_image_index << ',' << request.source_crop.x << ',' << request.source_crop.y << ','
			    << request.source_crop.width << ',' << request.source_crop.height << ',' << request.horizontal_flip
			    << ';';
		}
		return key.str();
	}

	detail::JpegDctDeviceBatchPlan cached_device_batch_plan(const std::vector<JpegDctImageCropRequest>& requests,
	                                                        const JpegDctDeviceBatchOptions&            options) const {
		if (options.layout != JpegDctDeviceLayout::kTransformedDctGrid) {
			return plan_device_batch(requests, options);
		}
		if (options.grid_transform.has_value()) {
			validate_grid_transform_spec(*options.grid_transform);
		}
		// Compact plans are cheaper to rebuild than to hash, retain, and look up.
		// Bypass and retire the exact-batch cache even if a legacy caller leaves
		// a nonzero cache capacity configured.
		auto planless = try_block_major_planless_device_batch(requests, options);
		if (!planless.has_value()) {
			planless = try_planless_device_batch(requests, options);
		}
		if (planless.has_value()) {
			std::vector<detail::JpegDctDeviceBatchPlan> retired_plans;
			{
				std::lock_guard<std::mutex> guard(device_plan_cache_mutex);
				retired_plans.reserve(device_plan_cache.size());
				for (auto& entry : device_plan_cache) {
					retired_plans.push_back(std::move(entry.second));
				}
				device_plan_cache.clear();
			}
			planless->plan_cache_evictions = retired_plans.size();
			retired_plans.clear();
			return std::move(*planless);
		}
		if (options.plan_cache_capacity == 0) {
			std::vector<detail::JpegDctDeviceBatchPlan> retired_plans;
			{
				std::lock_guard<std::mutex> guard(device_plan_cache_mutex);
				retired_plans.reserve(device_plan_cache.size());
				for (auto& entry : device_plan_cache) {
					retired_plans.push_back(std::move(entry.second));
				}
				device_plan_cache.clear();
			}
			const auto evictions = retired_plans.size();
			// Cached transform graphs can be large; destroy them after releasing
			// the cache mutex so other readers are not blocked by deallocation.
			retired_plans.clear();
			auto plan                 = plan_device_batch(requests, options);
			plan.plan_cache_misses    = 1;
			plan.plan_cache_evictions = evictions;
			return plan;
		}
		const auto                                    key = device_plan_cache_key(requests, options);
		std::vector<detail::JpegDctDeviceBatchPlan>   retired_plans;
		std::optional<detail::JpegDctDeviceBatchPlan> cached_plan;
		size_t                                        hit_evictions = 0;
		{
			std::lock_guard<std::mutex> guard(device_plan_cache_mutex);
			const auto                  found = device_plan_cache.find(key);
			if (found != device_plan_cache.end()) {
				cached_plan = found->second;
				if (device_plan_cache.size() > options.plan_cache_capacity) {
					retired_plans.reserve(device_plan_cache.size() - options.plan_cache_capacity);
				}
				for (auto it = device_plan_cache.begin();
				     device_plan_cache.size() > options.plan_cache_capacity && it != device_plan_cache.end();) {
					if (it == found) {
						++it;
						continue;
					}
					retired_plans.push_back(std::move(it->second));
					it = device_plan_cache.erase(it);
					++hit_evictions;
				}
			}
		}
		retired_plans.clear();
		if (cached_plan.has_value()) {
			auto plan                               = std::move(*cached_plan);
			plan.plan_cache_hits                    = 1;
			plan.plan_cache_misses                  = 0;
			plan.plan_cache_evictions               = hit_evictions;
			plan.resize_weight_build_ms             = 0.0;
			plan.dct_resize_weight_cache_hits       = 0;
			plan.dct_resize_weight_cache_misses     = 0;
			plan.dct_conversion_matrix_cache_hits   = 0;
			plan.dct_conversion_matrix_cache_misses = 0;
			return plan;
		}
		auto plan                           = plan_device_batch(requests, options);
		plan.exact_batch_plan_cache_enabled = true;
		plan.plan_cache_hits                = 0;
		plan.plan_cache_misses              = 1;
		plan.plan_cache_evictions           = 0;
		retired_plans.clear();
		{
			std::lock_guard<std::mutex> guard(device_plan_cache_mutex);
			const auto                  concurrent = device_plan_cache.find(key);
			if (concurrent == device_plan_cache.end()) {
				while (device_plan_cache.size() >= options.plan_cache_capacity) {
					auto evict = device_plan_cache.begin();
					retired_plans.push_back(std::move(evict->second));
					device_plan_cache.erase(evict);
					++plan.plan_cache_evictions;
				}
				device_plan_cache.emplace(key, plan);
			} else {
				for (auto it = device_plan_cache.begin();
				     device_plan_cache.size() > options.plan_cache_capacity && it != device_plan_cache.end();) {
					if (it == concurrent) {
						++it;
						continue;
					}
					retired_plans.push_back(std::move(it->second));
					it = device_plan_cache.erase(it);
					++plan.plan_cache_evictions;
				}
			}
		}
		retired_plans.clear();
		return plan;
	}

	JpegDctDeviceBatchPlanEstimate estimate_device_batch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                     const JpegDctDeviceBatchOptions&            options) const {
		if (options.layout != JpegDctDeviceLayout::kImageMajorComponentBlockCoeff &&
		    options.layout != JpegDctDeviceLayout::kYcbcrDctGrid &&
		    options.layout != JpegDctDeviceLayout::kTransformedDctGrid) {
			throw std::runtime_error("unsupported JPEG DCT device output layout");
		}

		JpegDctDeviceBatchPlanEstimate estimate;
		estimate.layout = options.layout;
		std::unordered_map<uint64_t, size_t> seen_rowgroups;

		const auto add_rowgroup = [&](const ShardState& shard, const uint32_t rowgroup_index) {
			const auto key = (static_cast<uint64_t>(shard.entry.shard_id) << 32U) | rowgroup_index;
			if (!seen_rowgroups.emplace(key, estimate.rowgroups.size()).second) {
				return;
			}
			if (rowgroup_index >= shard.rowgroup_n_tuples.size()) {
				throw std::runtime_error("JPEG DCT estimated rowgroup exceeds shard rowgroup metadata");
			}
			estimate.rowgroups.push_back(JpegDctDeviceRowgroupMetadata {shard.entry.shard_id, rowgroup_index});
			estimate.full_vector_count += row_count_to_vector_count(shard.rowgroup_n_tuples[rowgroup_index]);
		};

		for (const auto& request : requests) {
			const auto& shard = shard_for_global_image(request.global_image_index);
			const auto  local_image_index =
			    static_cast<uint32_t>(request.global_image_index - shard.entry.first_global_image_index);
			if (local_image_index >= shard.metadata.images.size()) {
				throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
			}

			const auto& image = shard.metadata.images[local_image_index];
			const auto  crop  = effective_crop_box(image, request.source_crop);

			for (const auto& component : image.components) {
				if (!component.present || component.width_in_blocks == 0 || component.height_in_blocks == 0) {
					continue;
				}

				const uint32_t x0 = std::min(component.width_in_blocks,
				                             floor_mul_div_u32(crop.x, component.width_in_blocks, image.image_width));
				const uint32_t y0 = std::min(component.height_in_blocks,
				                             floor_mul_div_u32(crop.y, component.height_in_blocks, image.image_height));
				const uint32_t x1 =
				    std::min(component.width_in_blocks,
				             ceil_mul_div_u32(crop.x + crop.width, component.width_in_blocks, image.image_width));
				const uint32_t y1 =
				    std::min(component.height_in_blocks,
				             ceil_mul_div_u32(crop.y + crop.height, component.height_in_blocks, image.image_height));
				if (shard.metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
					const auto block_count = static_cast<uint64_t>(x1 - x0) * static_cast<uint64_t>(y1 - y0);
					if (block_count > std::numeric_limits<size_t>::max() - estimate.block_count) {
						throw std::runtime_error("JPEG DCT estimated block count overflow");
					}
					estimate.block_count += static_cast<size_t>(block_count);
					if (block_count != 0U && !shard.vector_rowgroups) {
						const auto& image_group = shard.metadata.image_group_index.at(local_image_index);
						add_rowgroup(shard, image_group.fls_rowgroup_index);
					} else if (block_count != 0U) {
						for (uint32_t block_y = y0; block_y < y1; ++block_y) {
							for (uint32_t block_x = x0; block_x < x1; ++block_x) {
								const auto ref = locate_row_in_shard(shard,
								                                     local_image_index,
								                                     component.semantic_slot_id,
								                                     block_x,
								                                     block_y);
								if (!ref.present) {
									throw std::runtime_error("JPEG DCT manifest-v3 crop block is missing");
								}
								add_rowgroup(shard, ref.fls_rowgroup_index);
							}
						}
					}
					continue;
				}
				for (uint32_t block_y = y0; block_y < y1; ++block_y) {
					for (uint32_t block_x = x0; block_x < x1; ++block_x) {
						const auto* group = find_group_or_null(shard, component.semantic_slot_id, block_x, block_y);
						if (group == nullptr) {
							throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
						}

						++estimate.block_count;
						add_rowgroup(shard, group->fls_rowgroup_index);
					}
				}
			}
		}

		std::sort(estimate.rowgroups.begin(), estimate.rowgroups.end(), [](const auto& lhs, const auto& rhs) {
			if (lhs.shard_id != rhs.shard_id) {
				return lhs.shard_id < rhs.shard_id;
			}
			return lhs.rowgroup_index < rhs.rowgroup_index;
		});
		return estimate;
	}

	std::filesystem::path                       root_dir;
	JpegDctShardManifest                        manifest;
	mutable std::vector<ShardState>             shards;
	std::vector<PlanlessStaticImageDescriptor>  planless_static_images;
	std::vector<PlanlessStaticShardDescriptor>  planless_static_shards;
	std::vector<PlanlessStaticLayoutDescriptor> planless_static_layouts;
	std::vector<std::array<uint16_t, 64>>       planless_quant_tables;
	std::vector<uint16_t>                       planless_shard_indices;
	std::shared_ptr<const std::vector<std::filesystem::path>> image_major_static_metadata_paths;
	uint64_t                                    planless_uniform_shard_image_count = 0U;
	bool                                        planless_shard_indices_valid       = false;
	std::unique_ptr<JpegDctBlockMajorCompactPlanner> block_major_compact_planner;
	std::shared_ptr<const uint8_t>              plan_owner_token                   = std::make_shared<uint8_t>(0);
	detail::JpegDctShardCpuReader               cpu_reader;
	detail::JpegDctDeviceHostBridge             device_bridge;
	mutable std::mutex                          device_plan_cache_mutex;
	mutable std::unordered_map<std::string, detail::JpegDctDeviceBatchPlan> device_plan_cache;
	static constexpr size_t kCompiledAccessProfileFallbackCapacity = 8192U;
	mutable std::mutex compiled_access_profile_mutex;
	mutable std::vector<CompiledAccessProfileSlot> compiled_fixed_access_profiles;
	mutable std::unordered_map<CompiledAccessProfileKey,
	                           std::shared_ptr<const std::vector<uint32_t>>,
	                           CompiledAccessProfileKeyHash>
	    compiled_access_profiles;
	mutable std::deque<CompiledAccessProfileKey> compiled_access_profile_order;
	mutable std::mutex planless_axis_capacity_cache_mutex;
	mutable std::unordered_map<PlanlessAxisCapacityKey,
	                           PlanlessAxisCapacityContract,
	                           PlanlessAxisCapacityKeyHash>
	    planless_axis_capacity_cache;
};

JpegDctShardDatasetReader::JpegDctShardDatasetReader(const std::filesystem::path& manifest_path)
    : impl_(std::make_unique<Impl>(manifest_path)) {
}

JpegDctShardDatasetReader::~JpegDctShardDatasetReader() = default;

JpegDctShardDatasetReader::JpegDctShardDatasetReader(JpegDctShardDatasetReader&&) noexcept = default;

JpegDctShardDatasetReader& JpegDctShardDatasetReader::operator=(JpegDctShardDatasetReader&&) noexcept = default;

uint64_t JpegDctShardDatasetReader::image_count() const noexcept {
	return impl_ == nullptr ? 0 : impl_->manifest.image_count;
}

JpegDctReaderInitializationStats JpegDctShardDatasetReader::InitializationStats() const noexcept {
	if (impl_ == nullptr) {
		return {};
	}
	JpegDctReaderInitializationStats stats;
	{
		std::lock_guard<std::mutex> guard(impl_->initialization_stats_mutex);
		stats = impl_->initialization_stats;
	}
	if (impl_->block_major_compact_planner) {
		stats.block_major_loaded_descriptor_count =
		    impl_->block_major_compact_planner->loaded_descriptor_count();
		stats.block_major_loaded_descriptor_bytes =
		    impl_->block_major_compact_planner->loaded_descriptor_bytes();
		stats.block_major_descriptor_cache_byte_bound =
		    impl_->block_major_compact_planner->descriptor_cache_byte_bound();
		stats.block_major_descriptor_open_ms =
		    impl_->block_major_compact_planner->descriptor_open_ms();
		stats.block_major_descriptor_validation_ms =
		    impl_->block_major_compact_planner->descriptor_validation_ms();
	}
	return stats;
}

JpegImageMetadata JpegDctShardDatasetReader::ImageMetadata(const uint32_t global_image_index) const {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto& shard             = impl_->shard_for_global_image(global_image_index);
	const auto  local_image_index = static_cast<uint32_t>(global_image_index - shard.entry.first_global_image_index);
	if (local_image_index >= shard.metadata.images.size()) {
		throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
	}
	return shard.metadata.images[local_image_index];
}

uint64_t JpegDctShardDatasetReader::RowgroupStorageBytes(const uint32_t               shard_id,
                                                         const std::vector<uint32_t>& rowgroup_indices) const {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto& shard = impl_->shard_by_id(shard_id);
	return impl_->cpu_reader.RowgroupStorageBytes(shard_id, shard.fls_path, rowgroup_indices);
}

JpegDctDeviceBatchPlanPreview
JpegDctShardDatasetReader::PlanDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                              const JpegDctDeviceBatchOptions&            options) const {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto plan_start = std::chrono::steady_clock::now();
	auto       plan       = impl_->plan_device_batch(requests, options);
	const auto plan_end   = std::chrono::steady_clock::now();

	JpegDctDeviceBatchPlanPreview preview;
	for (const auto& shard_plan : *plan.shards) {
		for (const auto& rowgroup_plan : shard_plan.rowgroups) {
			const auto source_shard_id = rowgroup_plan.source_shard_id == std::numeric_limits<uint32_t>::max()
			                                 ? shard_plan.shard_id
			                                 : rowgroup_plan.source_shard_id;
			preview.rowgroup_vector_plans.push_back(JpegDctDeviceRowgroupVectorPlan {
			    JpegDctDeviceRowgroupMetadata {source_shard_id, rowgroup_plan.rowgroup_index},
			    rowgroup_plan.full_vector_count,
			    rowgroup_plan.selected_vectors});
		}
	}
	preview.layout                             = plan.layout;
	preview.image_layouts                      = std::move(plan.image_layouts);
	preview.block_metadata                     = std::move(plan.block_metadata);
	preview.rowgroups                          = std::move(plan.rowgroups);
	preview.planned_selected_vector_count      = plan.planned_selected_vector_count;
	preview.estimated_selected_vector_count    = plan.estimated_selected_vector_count;
	preview.full_vector_count                  = plan.full_vector_count;
	preview.planned_saved_vector_count         = plan.planned_saved_vector_count;
	preview.estimated_saved_vector_count       = plan.estimated_saved_vector_count;
	preview.selected_coefficients              = std::move(plan.selected_coefficients);
	preview.coefficients_per_block             = plan.coefficients_per_block;
	preview.planned_selected_vector_ratio      = plan.planned_selected_vector_ratio;
	preview.estimated_selected_vector_ratio    = plan.estimated_selected_vector_ratio;
	preview.ycbcr_dct_grid_shape               = plan.ycbcr_dct_grid_shape;
	preview.uses_planless_fixed_transform      = plan.uses_planless_fixed_transform;
	preview.compact_image_descriptor_count     = plan.uses_planless_fixed_transform ? preview.image_layouts.size() : 0U;
	preview.fixed_transform_component_count    = plan.fixed_transform_component_count;
	preview.fixed_transform_source_block_count = plan.fixed_transform_source_block_count;
	preview.fixed_transform_output_block_count = plan.fixed_transform_output_block_count;
	preview.host_expanded_transform_items_created  = plan.host_expanded_transform_items_created;
	preview.host_output_block_source_lists_created = plan.host_output_block_source_lists_created;
	preview.host_global_transform_sort_items       = plan.host_global_transform_sort_items;
	preview.planless_axis_program_count            = plan.planless_axis_program_count;
	preview.planless_axis_phase_matrix_count       = plan.planless_axis_phase_matrix_count;
	preview.compiled_access_profile_hits           = plan.compiled_access_profile_hits;
	preview.compiled_access_profile_misses         = plan.compiled_access_profile_misses;
	preview.planless_axis_program_bytes            = plan.fixed_resize_weight_matrices.size() * sizeof(float);
	preview.planless_axis_program_capacity_contract_bytes =
	    plan.planless_axis_program_capacity_contract_count * sizeof(float);
	preview.planless_axis_program_capacity_contract_complete =
	    plan.planless_axis_program_capacity_contract_complete;
	preview.compact_plan_bytes                     = plan.compact_plan_bytes;
	preview.compact_plan_peak_bytes                = plan.compact_plan_peak_bytes;
	preview.coordinate_group_lookup_count          = plan.coordinate_group_lookup_count;
	preview.coordinate_group_index_entries         = plan.coordinate_group_index_entries;
	preview.coordinate_group_index_populated       = plan.coordinate_group_index_populated;
	preview.coordinate_group_index_holes           = plan.coordinate_group_index_holes;
	preview.coordinate_group_index_bytes           = plan.coordinate_group_index_bytes;
	preview.coordinate_group_index_density         = plan.coordinate_group_index_density;
	preview.exact_batch_plan_cache_enabled         = plan.exact_batch_plan_cache_enabled;
	preview.compact_reader_image_locator_bytes =
	    impl_->planless_static_images.capacity() * sizeof(Impl::PlanlessStaticImageDescriptor);
	preview.compact_reader_shard_index_bytes = impl_->planless_shard_indices.capacity() * sizeof(uint16_t);
	preview.compact_reader_layout_dictionary_bytes =
	    impl_->planless_static_layouts.capacity() * sizeof(Impl::PlanlessStaticLayoutDescriptor);
	preview.compact_reader_quant_table_dictionary_bytes =
	    impl_->planless_quant_tables.capacity() * sizeof(std::array<uint16_t, 64>);
	preview.compact_reader_shard_descriptor_bytes =
	    impl_->planless_static_shards.capacity() * sizeof(Impl::PlanlessStaticShardDescriptor);
	preview.compact_reader_shard_index_derived = impl_->planless_uniform_shard_image_count != 0U;
	preview.decode_workset_capacity_bytes = plan.decode_workset_capacity_bytes;
	preview.estimated_max_decode_workset_bytes = plan.estimated_max_decode_workset_bytes;
	preview.estimated_oversized_decode_rowgroups = plan.estimated_oversized_decode_rowgroups;
	preview.compact_reader_total_bytes =
	    preview.compact_reader_image_locator_bytes + preview.compact_reader_shard_index_bytes +
	    preview.compact_reader_layout_dictionary_bytes + preview.compact_reader_quant_table_dictionary_bytes +
	    preview.compact_reader_shard_descriptor_bytes;
	preview.planning_ms                      = std::chrono::duration<double, std::milli>(plan_end - plan_start).count();
	preview.resize_weight_build_ms           = plan.resize_weight_build_ms;
	preview.dct_resize_weight_cache_hits     = plan.dct_resize_weight_cache_hits;
	preview.dct_resize_weight_cache_misses   = plan.dct_resize_weight_cache_misses;
	preview.dct_conversion_matrix_cache_hits = plan.dct_conversion_matrix_cache_hits;
	preview.dct_conversion_matrix_cache_misses = plan.dct_conversion_matrix_cache_misses;
	return preview;
}

JpegDctDeviceBatchPlanEstimate
JpegDctShardDatasetReader::EstimateDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                  const JpegDctDeviceBatchOptions&            options) const {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto estimate_start = std::chrono::steady_clock::now();
	auto       estimate       = impl_->estimate_device_batch(requests, options);
	const auto estimate_end   = std::chrono::steady_clock::now();
	estimate.planning_ms      = std::chrono::duration<double, std::milli>(estimate_end - estimate_start).count();
	return estimate;
}

JpegDctDeviceBatchPreparedPlan
JpegDctShardDatasetReader::PrepareDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                 const JpegDctDeviceBatchOptions&            options) {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto plan_start = std::chrono::steady_clock::now();
	auto       plan       = impl_->cached_device_batch_plan(requests, options);
	const auto compile_start = std::chrono::steady_clock::now();
	plan.plan_device_batch_ms =
	    std::chrono::duration<double, std::milli>(compile_start - plan_start).count();
	impl_->device_bridge.CompileIoPlan(plan);
	const auto plan_end   = std::chrono::steady_clock::now();
	plan.compile_io_plan_ms =
	    std::chrono::duration<double, std::milli>(plan_end - compile_start).count();
	plan.planning_ms      = std::chrono::duration<double, std::milli>(plan_end - plan_start).count();
	return JpegDctDeviceBatchPreparedPlan(
	    std::make_unique<JpegDctDeviceBatchPreparedPlan::Impl>(std::move(plan), options, impl_->plan_owner_token));
}

void JpegDctShardDatasetReader::StagePreparedDeviceDctBatchIo(JpegDctDeviceBatchPreparedPlan& plan) {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	if (plan.impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT prepared device batch plan is empty");
	}
	if (plan.impl_->owner_token != impl_->plan_owner_token) {
		throw std::runtime_error("JPEG DCT prepared device batch plan belongs to a different reader");
	}
	impl_->device_bridge.StageIo(plan.impl_->plan);
}

JpegDctDeviceBatch JpegDctShardDatasetReader::ReadPreparedDeviceDctBatch(JpegDctDeviceBatchPreparedPlan plan) {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	if (plan.impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT prepared device batch plan is empty");
	}
	if (plan.impl_->owner_token != impl_->plan_owner_token) {
		throw std::runtime_error("JPEG DCT prepared device batch plan belongs to a different reader");
	}
	StagePreparedDeviceDctBatchIo(plan);
	auto       detail_plan         = std::move(plan.impl_->plan);
	const auto requested_selection = detail::normalize_coefficient_selection(plan.impl_->options.coefficient_selection);
	if (requested_selection != detail_plan.selected_coefficients) {
		throw std::runtime_error("JPEG DCT prepared device batch coefficient selection was modified");
	}
	auto options = plan.impl_->options;
	plan.impl_.reset();
	auto batch = impl_->device_bridge.Execute(std::move(detail_plan), options);
	return batch;
}

JpegDctDeviceBatch JpegDctShardDatasetReader::ReadDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                                 const JpegDctDeviceBatchOptions&            options) {
	return ReadPreparedDeviceDctBatch(PrepareDeviceDctBatch(requests, options));
}

JpegDctRowRef JpegDctShardDatasetReader::LocateRow(const uint32_t global_image_index,
                                                   const uint32_t semantic_slot_id,
                                                   const uint32_t block_x,
                                                   const uint32_t block_y) {
	const auto& shard             = impl_->shard_for_global_image(global_image_index);
	const auto  local_image_index = static_cast<uint32_t>(global_image_index - shard.entry.first_global_image_index);
	return Impl::locate_row_in_shard(shard, local_image_index, semantic_slot_id, block_x, block_y);
}

JpegDctBlockGroup JpegDctShardDatasetReader::ReadBlockGroup(const uint32_t shard_id,
                                                            const uint32_t semantic_slot_id,
                                                            const uint32_t block_x,
                                                            const uint32_t block_y) {
	const auto& shard = impl_->shard_by_id(shard_id);
	const auto& group = Impl::find_group(shard.metadata, semantic_slot_id, block_x, block_y);
	return impl_->cpu_reader.ReadBlockGroup(shard.fls_path, group);
}

MaterializedJpegDctImage JpegDctShardDatasetReader::MaterializeImageDct(const uint32_t global_image_index) {
	const auto&                                     shard = impl_->shard_for_global_image(global_image_index);
	std::vector<detail::JpegDctMaterializeBlockRef> blocks;
	if (shard.metadata.row_ordering == JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor) {
		const auto local_image_index = static_cast<uint32_t>(global_image_index - shard.entry.first_global_image_index);
		const auto& image_metadata   = shard.metadata.images.at(local_image_index);
		for (const auto& component : image_metadata.components) {
			if (!component.present) {
				continue;
			}
			for (uint32_t block_y = 0; block_y < component.height_in_blocks; ++block_y) {
				for (uint32_t block_x = 0; block_x < component.width_in_blocks; ++block_x) {
					blocks.push_back(detail::JpegDctMaterializeBlockRef {
					    component.semantic_slot_id,
					    block_x,
					    block_y,
					    LocateRow(global_image_index, component.semantic_slot_id, block_x, block_y)});
				}
			}
		}
		return impl_->cpu_reader.MaterializeImage(shard.fls_path, global_image_index, blocks);
	}
	for (const auto& group : shard.metadata.block_group_index) {
		auto ref = LocateRow(global_image_index, group.semantic_slot_id, group.block_x, group.block_y);
		if (!ref.present || ref.row_offset_in_block_group >= group.row_count) {
			continue;
		}
		blocks.push_back(
		    detail::JpegDctMaterializeBlockRef {group.semantic_slot_id, group.block_x, group.block_y, ref});
	}
	return impl_->cpu_reader.MaterializeImage(shard.fls_path, global_image_index, blocks);
}

} // namespace galp::jpeg
