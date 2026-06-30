// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/benchmark_support/pipeline.cu
// ────────────────────────────────────────────────────────
#include "galp_tools/benchmark_support/pipeline.cuh"

#if GALP_WITH_JPEG_DCT

#include "cuda/cuda_macros.cuh"
#include "jpeg/jpeg_dct_device.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace galp::execution {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

double external_plan_overhead_ms(const double measured_plan_ms, const galp::jpeg::JpegDctDeviceBatch& executed_batch) {
	return detail::external_plan_overhead_ms(measured_plan_ms, executed_batch.execution_stats().planning_ms);
}

struct BlockKey {
	uint32_t request_index      = 0;
	uint32_t global_image_index = 0;
	uint32_t semantic_slot_id   = 0;
	uint32_t block_x            = 0;
	uint32_t block_y            = 0;

	bool operator==(const BlockKey& other) const noexcept {
		return request_index == other.request_index && global_image_index == other.global_image_index &&
		       semantic_slot_id == other.semantic_slot_id && block_x == other.block_x && block_y == other.block_y;
	}
};

struct BlockKeyHash {
	size_t operator()(const BlockKey& key) const noexcept {
		size_t h = key.request_index;
		h ^= static_cast<size_t>(key.global_image_index) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
		h ^= static_cast<size_t>(key.semantic_slot_id) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
		h ^= static_cast<size_t>(key.block_x) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
		h ^= static_cast<size_t>(key.block_y) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
		return h;
	}
};

BlockKey block_key(const galp::jpeg::JpegDctDeviceBlockMetadata& block) {
	return BlockKey {
	    block.request_index, block.global_image_index, block.semantic_slot_id, block.block_x, block.block_y};
}

std::vector<uint32_t> resolve_image_ids(galp::jpeg::JpegDctShardDatasetReader& reader,
                                        const std::vector<uint32_t>&           requested) {
	if (!requested.empty()) {
		return requested;
	}
	const auto image_count = reader.image_count();
	if (image_count > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("JPEG DCT manifest image count exceeds uint32_t image ids");
	}
	std::vector<uint32_t> image_ids;
	image_ids.reserve(static_cast<size_t>(image_count));
	for (uint64_t image_id = 0; image_id < image_count; ++image_id) {
		image_ids.push_back(static_cast<uint32_t>(image_id));
	}
	return image_ids;
}

std::vector<galp::jpeg::JpegDctImageCropRequest> make_requests(const std::vector<uint32_t>&      image_ids,
                                                               const size_t                      begin,
                                                               const size_t                      end,
                                                               const galp::jpeg::JpegDctCropBox& crop) {
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	requests.reserve(end - begin);
	for (size_t i = begin; i < end; ++i) {
		requests.push_back(galp::jpeg::JpegDctImageCropRequest {image_ids[i], crop});
	}
	return requests;
}

bool has_crop(const galp::jpeg::JpegDctCropBox& crop) {
	return crop.width != 0 && crop.height != 0;
}

uint32_t ceil_mul_div_u32(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
	if (divisor == 0) {
		throw std::runtime_error("JPEG DCT crop planning encountered a zero image dimension");
	}
	const uint64_t product = static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs);
	return static_cast<uint32_t>((product + divisor - 1U) / divisor);
}

uint32_t floor_mul_div_u32(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
	if (divisor == 0) {
		throw std::runtime_error("JPEG DCT crop planning encountered a zero image dimension");
	}
	return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs)) / divisor);
}

galp::jpeg::JpegDctCropBox effective_crop_box(const galp::jpeg::JpegImageMetadata& image,
                                              galp::jpeg::JpegDctCropBox           crop) {
	if (image.image_width == 0 || image.image_height == 0) {
		throw std::runtime_error("JPEG DCT crop planning requires per-image dimensions");
	}
	if (!has_crop(crop)) {
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

const galp::jpeg::JpegComponentMetadata* find_component(const galp::jpeg::JpegImageMetadata& image,
                                                        const uint32_t                       semantic_slot_id) {
	for (const auto& component : image.components) {
		if (component.semantic_slot_id == semantic_slot_id) {
			return &component;
		}
	}
	return nullptr;
}

std::vector<int16_t> copy_coefficients_to_host(const galp::jpeg::JpegDctDeviceBatch& batch) {
	std::vector<int16_t> host(batch.coefficient_count());
	if (!host.empty()) {
		CUDA_SAFE_CALL(cudaMemcpy(
		    host.data(), batch.device_coefficients(), host.size() * sizeof(int16_t), cudaMemcpyDeviceToHost));
	}
	return host;
}

void accumulate_cache_stats(PipelineBenchmarkStageResult& stage, const galp::jpeg::JpegDctDeviceBatch& batch) {
	const auto cache_stats = batch.cache_stats();
	stage.cache_hits += cache_stats.hits;
	stage.cache_misses += cache_stats.misses;
	stage.dense_cache_hits = stage.cache_hits;
	stage.dense_cache_misses = stage.cache_misses;
	stage.cache_inserts += cache_stats.inserts;
	stage.cache_evictions += cache_stats.evictions;
	stage.cache_resident_bytes     = cache_stats.resident_bytes;
	stage.cache_resident_rowgroups = cache_stats.resident_rowgroups;
}

void refresh_runtime_policy_summary(PipelineBenchmarkStageResult& stage) {
	if (stage.runtime_policy_selected_rowgroups == 0 && stage.runtime_policy_full_rowgroups == 0) {
		stage.runtime_policy_decision = "none";
	} else if (stage.runtime_policy_selected_rowgroups != 0 && stage.runtime_policy_full_rowgroups != 0) {
		stage.runtime_policy_decision = "mixed";
	} else if (stage.runtime_policy_selected_rowgroups != 0) {
		stage.runtime_policy_decision = "selected-vector";
	} else {
		stage.runtime_policy_decision = "full-rowgroup";
	}
	stage.runtime_policy_reason =
	    "selected=" + std::to_string(stage.runtime_policy_selected_rowgroups) +
	    ",full=" + std::to_string(stage.runtime_policy_full_rowgroups) +
	    ",tail_full=" + std::to_string(stage.runtime_policy_tail_full_rowgroups) +
	    ",ratio_full=" + std::to_string(stage.runtime_policy_ratio_full_rowgroups) +
	    ",low_saving_full=" + std::to_string(stage.runtime_policy_low_saving_full_rowgroups) +
	    ",max_selected_ratio=" + std::to_string(galp::jpeg::detail::kMaxSelectedVectorRatioForPushdown) +
	    ",min_saved_vectors=" + std::to_string(galp::jpeg::detail::kMinSavedVectorsForPushdown);
}

void accumulate_execution_stats(PipelineBenchmarkStageResult& stage, const galp::jpeg::JpegDctDeviceBatch& batch) {
	const auto stats = batch.execution_stats();
	stage.planned_selected_vector_count += stats.planned_selected_vector_count;
	stage.selected_vector_count += stats.selected_vector_count;
	stage.full_vector_count += stats.full_vector_count;
	stage.planned_saved_vector_count += stats.planned_saved_vector_count;
	stage.actual_saved_vector_count += stats.actual_saved_vector_count;
	stage.rowgroup_count += stats.rowgroup_count;
	stage.planned_selected_vector_ratio =
	    stage.full_vector_count == 0
	        ? 0.0
	        : static_cast<double>(stage.planned_selected_vector_count) / static_cast<double>(stage.full_vector_count);
	stage.selected_vector_ratio =
	    stage.full_vector_count == 0
	        ? 0.0
	        : static_cast<double>(stage.selected_vector_count) / static_cast<double>(stage.full_vector_count);
	stage.workset_count += stats.workset_count;
	stage.decode_kernel_launch_count += stats.decode_kernel_launch_count;
	stage.gather_kernel_launch_count += stats.gather_kernel_launch_count;
	stage.cached_gather_kernel_launch_count += stats.cached_gather_kernel_launch_count;
	stage.materialize_kernel_launch_count += stats.materialize_kernel_launch_count;
	stage.gather_item_count += stats.gather_item_count;
	stage.decoded_gather_item_count += stats.decoded_gather_item_count;
	stage.cached_gather_item_count += stats.cached_gather_item_count;
	stage.workset_upload_count += stats.workset_upload_count;
	stage.scratch_upload_count += stats.scratch_upload_count;
	stage.scratch_allocation_count += stats.scratch_allocation_count;
	stage.internal_sync_count += stats.internal_sync_count;
	stage.cached_gather_sync_count += stats.cached_gather_sync_count;
	stage.decoded_batch_sync_count += stats.decoded_batch_sync_count;
	stage.cached_gather_event_handoff_count += stats.cached_gather_event_handoff_count;
	stage.sparse_vector_cache_hits += stats.sparse_vector_cache_hits;
	stage.sparse_vector_cache_misses += stats.sparse_vector_cache_misses;
	stage.runtime_policy_selected_rowgroups += stats.runtime_policy_selected_rowgroups;
	stage.runtime_policy_full_rowgroups += stats.runtime_policy_full_rowgroups;
	stage.runtime_policy_tail_full_rowgroups += stats.runtime_policy_tail_full_rowgroups;
	stage.runtime_policy_ratio_full_rowgroups += stats.runtime_policy_ratio_full_rowgroups;
	stage.runtime_policy_low_saving_full_rowgroups += stats.runtime_policy_low_saving_full_rowgroups;
	refresh_runtime_policy_summary(stage);
	stage.device_planning_ms += stats.planning_ms;
	stage.plan_ms += stats.planning_ms;
	stage.workset_build_ms += stats.workset_build_ms;
	stage.workset_upload_ms += stats.workset_upload_ms;
	stage.decode_ms += stats.decode_ms;
	stage.gather_ms += stats.gather_ms;
}

void accumulate_rowgroups(PipelineBenchmarkStageResult&            stage,
                          const galp::jpeg::JpegDctDeviceBatch&    batch,
                          std::set<std::pair<uint32_t, uint32_t>>& seen_rowgroups) {
	stage.rowgroup_visits += batch.rowgroup_count();
	for (const auto& rowgroup : batch.rowgroups()) {
		if (!seen_rowgroups.insert({rowgroup.shard_id, rowgroup.rowgroup_index}).second) {
			++stage.repeated_rowgroups;
		}
	}
	stage.unique_rowgroups = seen_rowgroups.size();
}

detail::AutoPipelinePolicyDecision
choose_auto_pipeline_policy(const galp::jpeg::JpegDctDeviceBatchPreparedPlan& crop_plan,
                            const galp::jpeg::JpegDctDeviceBatchPreparedPlan& full_plan,
                            const size_t                                      decode_batch_rowgroups) {
	return detail::choose_auto_pipeline_policy_from_estimates(
	    crop_plan.block_metadata().size(),
	    full_plan.block_metadata().size(),
	    crop_plan.rowgroups().size(),
	    full_plan.rowgroups().size(),
	    crop_plan.estimated_selected_vector_count(),
	    full_plan.full_vector_count(),
	    detail::estimate_auto_worksets_for_rowgroups(crop_plan.rowgroups(), decode_batch_rowgroups),
	    detail::estimate_auto_worksets_for_rowgroups(full_plan.rowgroups(), decode_batch_rowgroups));
}

void accumulate_auto_pipeline_policy_plan(
    PipelineBenchmarkResult&                                result,
    const galp::jpeg::JpegDctDeviceBatchPreparedPlan& crop_plan,
    const galp::jpeg::JpegDctDeviceBatchPreparedPlan& full_plan,
    const detail::AutoPipelinePolicyDecision&         decision,
    std::set<std::pair<uint32_t, uint32_t>>&          pushdown_policy_seen_rowgroups,
    std::set<std::pair<uint32_t, uint32_t>>&          full_policy_seen_rowgroups) {
	result.auto_policy_selected_blocks += crop_plan.block_metadata().size();
	result.auto_policy_full_blocks += full_plan.block_metadata().size();
	result.auto_policy_selected_vectors += crop_plan.estimated_selected_vector_count();
	result.auto_policy_full_vectors += full_plan.full_vector_count();
	result.auto_policy_touched_rowgroups += crop_plan.rowgroups().size();
	result.auto_policy_full_rowgroups += full_plan.rowgroups().size();
	result.auto_policy_pushdown_reuse_candidate_rowgroups +=
	    detail::count_auto_reuse_candidate_rowgroups(crop_plan.rowgroups(), pushdown_policy_seen_rowgroups);
	result.auto_policy_full_reuse_candidate_rowgroups +=
	    detail::count_auto_reuse_candidate_rowgroups(full_plan.rowgroups(), full_policy_seen_rowgroups);
	result.auto_policy_estimated_pushdown_worksets += decision.estimated_pushdown_worksets;
	result.auto_policy_estimated_full_worksets += decision.estimated_full_worksets;
	result.auto_policy_estimated_pushdown_gather_items += decision.estimated_pushdown_gather_items;
	result.auto_policy_estimated_full_gather_items += decision.estimated_full_gather_items;
	result.auto_policy_selected_block_ratio =
	    result.auto_policy_full_blocks == 0
	        ? 0.0
	        : static_cast<double>(result.auto_policy_selected_blocks) /
	              static_cast<double>(result.auto_policy_full_blocks);
	result.auto_policy_selected_vector_ratio =
	    result.auto_policy_full_vectors == 0
	        ? 0.0
	        : static_cast<double>(result.auto_policy_selected_vectors) /
	              static_cast<double>(result.auto_policy_full_vectors);
	result.auto_policy_touched_rowgroup_ratio =
	    result.auto_policy_full_rowgroups == 0
	        ? 0.0
	        : static_cast<double>(result.auto_policy_touched_rowgroups) /
	              static_cast<double>(result.auto_policy_full_rowgroups);
	result.auto_policy_pushdown_reuse_candidate_ratio =
	    result.auto_policy_touched_rowgroups == 0
	        ? 0.0
	        : static_cast<double>(result.auto_policy_pushdown_reuse_candidate_rowgroups) /
	              static_cast<double>(result.auto_policy_touched_rowgroups);
	result.auto_policy_full_reuse_candidate_ratio =
	    result.auto_policy_full_rowgroups == 0
	        ? 0.0
	        : static_cast<double>(result.auto_policy_full_reuse_candidate_rowgroups) /
	              static_cast<double>(result.auto_policy_full_rowgroups);
	result.auto_policy_avg_full_blocks_per_rowgroup =
	    result.auto_policy_full_rowgroups == 0
	        ? 0.0
	        : static_cast<double>(result.auto_policy_full_blocks) /
	              static_cast<double>(result.auto_policy_full_rowgroups);
}

void record_auto_pipeline_policy_reason(PipelineBenchmarkResult&                         result,
                                        const detail::AutoPipelinePolicyReason reason) {
	switch (reason) {
	case detail::AutoPipelinePolicyReason::EmptyWindow:
		++result.auto_policy_empty_windows;
		break;
	case detail::AutoPipelinePolicyReason::CropCoversFullWindow:
		++result.auto_policy_crop_covers_full_windows;
		break;
	case detail::AutoPipelinePolicyReason::VerySmallCrop:
		++result.auto_policy_very_small_crop_windows;
		break;
	case detail::AutoPipelinePolicyReason::SmallWindowFixedOverhead:
		++result.auto_policy_small_window_full_windows;
		break;
	case detail::AutoPipelinePolicyReason::LargeWindowAmortizesPushdown:
		++result.auto_policy_large_window_pushdown_windows;
		break;
	case detail::AutoPipelinePolicyReason::WorksetOverheadTooHigh:
		++result.auto_policy_workset_overhead_full_windows;
		break;
	case detail::AutoPipelinePolicyReason::TinyRowgroupsFixedOverhead:
		++result.auto_policy_tiny_rowgroups_full_windows;
		break;
	case detail::AutoPipelinePolicyReason::TouchesMostRowgroups:
		++result.auto_policy_touches_most_rowgroups_windows;
		break;
	case detail::AutoPipelinePolicyReason::GatherOutputTooHigh:
		++result.auto_policy_gather_output_full_windows;
		break;
	case detail::AutoPipelinePolicyReason::CropSavesEnoughBlocks:
		++result.auto_policy_saves_enough_blocks_windows;
		break;
	case detail::AutoPipelinePolicyReason::SavingsTooSmall:
		++result.auto_policy_savings_too_small_windows;
		break;
	}
}

void accumulate_pushdown_window(PipelineBenchmarkStageResult&            stage,
                                const galp::jpeg::JpegDctDeviceBatch&    batch,
                                const double                             read_decode_ms,
                                std::set<std::pair<uint32_t, uint32_t>>& seen_rowgroups) {
	++stage.windows;
	stage.requests += batch.image_count();
	stage.input_blocks += batch.block_count();
	stage.output_blocks += batch.block_count();
	stage.output_coefficients += batch.coefficient_count();
	stage.output_bytes += batch.coefficient_bytes();
	stage.peak_window_images        = std::max(stage.peak_window_images, batch.image_count());
	stage.peak_window_input_blocks  = std::max(stage.peak_window_input_blocks, batch.block_count());
	stage.peak_window_output_blocks = std::max(stage.peak_window_output_blocks, batch.block_count());
	stage.peak_window_output_bytes  = std::max(stage.peak_window_output_bytes, batch.coefficient_bytes());
	stage.read_decode_ms += read_decode_ms;
	accumulate_cache_stats(stage, batch);
	accumulate_execution_stats(stage, batch);
	accumulate_rowgroups(stage, batch, seen_rowgroups);
}

void accumulate_full_then_crop_window(PipelineBenchmarkStageResult&            stage,
                                      const galp::jpeg::JpegDctDeviceBatch&    batch,
                                      const size_t                             output_blocks,
                                      const double                             plan_ms,
                                      const double                             read_decode_ms,
                                      const double                             transform_ms,
                                      std::set<std::pair<uint32_t, uint32_t>>& seen_rowgroups) {
	++stage.windows;
	stage.requests += batch.image_count();
	stage.input_blocks += batch.block_count();
	stage.output_blocks += output_blocks;
	stage.output_coefficients += output_blocks * 64U;
	stage.output_bytes += output_blocks * 64U * sizeof(int16_t);
	stage.peak_window_images = std::max(stage.peak_window_images, batch.image_count());
	stage.peak_window_input_blocks = std::max(stage.peak_window_input_blocks, batch.block_count());
	stage.peak_window_output_blocks = std::max(stage.peak_window_output_blocks, output_blocks);
	stage.peak_window_output_bytes =
	    std::max(stage.peak_window_output_bytes, output_blocks * 64U * sizeof(int16_t));
	stage.plan_ms += plan_ms;
	stage.read_decode_ms += read_decode_ms;
	stage.transform_ms += transform_ms;
	accumulate_cache_stats(stage, batch);
	accumulate_execution_stats(stage, batch);
	accumulate_rowgroups(stage, batch, seen_rowgroups);
}

std::vector<size_t> crop_full_batch_to_pushdown_blocks(const galp::jpeg::JpegDctDeviceBatch& pushdown,
                                                       const galp::jpeg::JpegDctDeviceBatch& full) {
	std::unordered_map<BlockKey, size_t, BlockKeyHash> full_index;
	const auto&                                        full_blocks = full.block_metadata();
	full_index.reserve(full_blocks.size());
	for (size_t block_idx = 0; block_idx < full_blocks.size(); ++block_idx) {
		full_index.emplace(block_key(full_blocks[block_idx]), block_idx);
	}

	std::vector<size_t> selected;
	selected.reserve(pushdown.block_count());
	for (const auto& block : pushdown.block_metadata()) {
		const auto it = full_index.find(block_key(block));
		if (it == full_index.end()) {
			throw std::runtime_error("full-then-crop baseline missed a pushdown-selected DCT block");
		}
		selected.push_back(it->second);
	}
	return selected;
}

bool block_intersects_crop(const galp::jpeg::JpegDctDeviceBlockMetadata& block,
                           const galp::jpeg::JpegImageMetadata&         image,
                           const galp::jpeg::JpegDctCropBox&            requested_crop) {
	const auto* component = find_component(image, block.semantic_slot_id);
	if (component == nullptr || !component->present || component->width_in_blocks == 0 ||
	    component->height_in_blocks == 0) {
		return false;
	}

	const auto crop = effective_crop_box(image, requested_crop);
	const uint32_t x0 =
	    std::min(component->width_in_blocks, floor_mul_div_u32(crop.x, component->width_in_blocks, image.image_width));
	const uint32_t y0 =
	    std::min(component->height_in_blocks, floor_mul_div_u32(crop.y, component->height_in_blocks, image.image_height));
	const uint32_t x1 =
	    std::min(component->width_in_blocks,
	             ceil_mul_div_u32(crop.x + crop.width, component->width_in_blocks, image.image_width));
	const uint32_t y1 =
	    std::min(component->height_in_blocks,
	             ceil_mul_div_u32(crop.y + crop.height, component->height_in_blocks, image.image_height));
	return block.block_x >= x0 && block.block_x < x1 && block.block_y >= y0 && block.block_y < y1;
}

std::vector<size_t> crop_full_batch_to_requested_blocks(galp::jpeg::JpegDctShardDatasetReader& reader,
                                                       const galp::jpeg::JpegDctDeviceBatch&   full,
                                                       const galp::jpeg::JpegDctCropBox&       crop) {
	std::vector<size_t> selected;
	selected.reserve(full.block_count());
	if (!has_crop(crop)) {
		selected.resize(full.block_count());
		for (size_t block_idx = 0; block_idx < selected.size(); ++block_idx) {
			selected[block_idx] = block_idx;
		}
		return selected;
	}

	std::unordered_map<uint32_t, galp::jpeg::JpegImageMetadata> image_metadata;
	for (size_t block_idx = 0; block_idx < full.block_metadata().size(); ++block_idx) {
		const auto& block = full.block_metadata()[block_idx];
		auto [it, inserted] = image_metadata.try_emplace(block.global_image_index);
		if (inserted) {
			it->second = reader.ImageMetadata(block.global_image_index);
		}
		if (block_intersects_crop(block, it->second, crop)) {
			selected.push_back(block_idx);
		}
	}
	return selected;
}

void verify_equal_blocks(const galp::jpeg::JpegDctDeviceBatch& pushdown,
                         const galp::jpeg::JpegDctDeviceBatch& full,
                         const std::vector<size_t>&            selected_full_blocks) {
	if (pushdown.block_count() != selected_full_blocks.size()) {
		throw std::runtime_error("pushdown and full-then-crop block counts differ");
	}
	const auto push_host = copy_coefficients_to_host(pushdown);
	const auto full_host = copy_coefficients_to_host(full);
	for (size_t block_idx = 0; block_idx < selected_full_blocks.size(); ++block_idx) {
		const auto full_block_idx = selected_full_blocks[block_idx];
		const auto push_offset    = block_idx * 64U;
		const auto full_offset    = full_block_idx * 64U;
		if (push_offset + 64U > push_host.size() || full_offset + 64U > full_host.size()) {
			throw std::runtime_error("DCT coefficient validation encountered an out-of-range block");
		}
		if (std::memcmp(push_host.data() + push_offset, full_host.data() + full_offset, 64U * sizeof(int16_t)) != 0) {
			throw std::runtime_error("pushdown and full-then-crop DCT coefficients differ");
		}
	}
}

} // namespace

PipelineBenchmarkResult benchmark_jpeg_dct_pipeline(const std::filesystem::path&   manifest_path,
                                                    const PipelineBenchmarkConfig& cfg) {
	if (cfg.window_images == 0) {
		throw std::invalid_argument("pipeline benchmark window_images must be greater than zero");
	}
	if (cfg.decode_batch_rowgroups == 0) {
		throw std::invalid_argument("pipeline benchmark decode_batch_rowgroups must be greater than zero");
	}

	galp::jpeg::JpegDctShardDatasetReader metadata_reader(manifest_path);
	auto                                  image_ids = resolve_image_ids(metadata_reader, cfg.image_ids);

	PipelineBenchmarkResult result;
	result.dataset_images = metadata_reader.image_count();
	result.mode           = cfg.mode;

	galp::jpeg::JpegDctDeviceBatchOptions batch_options;
	batch_options.cache_capacity_bytes = cfg.cache_capacity_bytes;
	batch_options.decode_batch_rowgroups = cfg.decode_batch_rowgroups;

	std::set<std::pair<uint32_t, uint32_t>> pushdown_rowgroups;
	std::set<std::pair<uint32_t, uint32_t>> baseline_rowgroups;
	std::set<std::pair<uint32_t, uint32_t>> auto_pushdown_policy_rowgroups;
	std::set<std::pair<uint32_t, uint32_t>> auto_full_policy_rowgroups;
	const bool run_pushdown = cfg.mode == PipelineBenchmarkMode::Pushdown || cfg.mode == PipelineBenchmarkMode::Compare;
	const bool run_baseline =
	    cfg.mode == PipelineBenchmarkMode::FullThenCrop || cfg.mode == PipelineBenchmarkMode::Compare;
	const bool run_auto = cfg.mode == PipelineBenchmarkMode::Auto;

	std::unique_ptr<galp::jpeg::JpegDctShardDatasetReader> pushdown_reader;
	std::unique_ptr<galp::jpeg::JpegDctShardDatasetReader> baseline_reader;
	if (run_pushdown || (run_auto && has_crop(cfg.crop))) {
		pushdown_reader = std::make_unique<galp::jpeg::JpegDctShardDatasetReader>(manifest_path);
	}
	if (run_baseline || run_auto) {
		baseline_reader = std::make_unique<galp::jpeg::JpegDctShardDatasetReader>(manifest_path);
	}

	for (size_t begin = 0; begin < image_ids.size(); begin += cfg.window_images) {
		const size_t end = std::min(image_ids.size(), begin + cfg.window_images);

		if (run_auto) {
			if (!has_crop(cfg.crop)) {
				const auto auto_policy_start = Clock::now();
				auto       full_requests     = make_requests(image_ids, begin, end, galp::jpeg::JpegDctCropBox {});
				auto       full_plan         = baseline_reader->PrepareDeviceDctBatch(full_requests, batch_options);
				const auto decision          = choose_auto_pipeline_policy(
				    full_plan, full_plan, batch_options.decode_batch_rowgroups);
				const auto auto_policy_end   = Clock::now();
				const auto auto_policy_ms    = elapsed_ms(auto_policy_start, auto_policy_end);
				result.auto_policy_ms += auto_policy_ms;
				result.auto_policy_reason = decision.reason;
				accumulate_auto_pipeline_policy_plan(
				    result, full_plan, full_plan, decision, auto_pushdown_policy_rowgroups, auto_full_policy_rowgroups);
				record_auto_pipeline_policy_reason(result, decision.reason_code);

				++result.auto_full_then_crop_windows;
				const auto read_start = Clock::now();
				auto       full_batch = baseline_reader->ReadPreparedDeviceDctBatch(std::move(full_plan));
				const auto read_end   = Clock::now();

				const size_t output_blocks = full_batch.block_count();

				accumulate_full_then_crop_window(result.full_then_crop,
				                                 full_batch,
				                                 output_blocks,
				                                 external_plan_overhead_ms(auto_policy_ms, full_batch),
				                                 elapsed_ms(read_start, read_end),
				                                 0.0,
				                                 baseline_rowgroups);
				continue;
			}

			const auto auto_policy_start = Clock::now();
			auto       push_requests     = make_requests(image_ids, begin, end, cfg.crop);
			auto       full_requests     = make_requests(image_ids, begin, end, galp::jpeg::JpegDctCropBox {});
			auto       crop_plan         = pushdown_reader->PrepareDeviceDctBatch(push_requests, batch_options);
			auto       full_plan         = baseline_reader->PrepareDeviceDctBatch(full_requests, batch_options);
			const auto decision          = choose_auto_pipeline_policy(
			    crop_plan, full_plan, batch_options.decode_batch_rowgroups);
			const auto auto_policy_end   = Clock::now();
			const auto auto_policy_ms    = elapsed_ms(auto_policy_start, auto_policy_end);
			result.auto_policy_ms += auto_policy_ms;
			result.auto_policy_reason = decision.reason;
			accumulate_auto_pipeline_policy_plan(
			    result, crop_plan, full_plan, decision, auto_pushdown_policy_rowgroups, auto_full_policy_rowgroups);
			record_auto_pipeline_policy_reason(result, decision.reason_code);

			if (decision.use_pushdown) {
				++result.auto_pushdown_windows;
				const auto read_start  = Clock::now();
				auto       push_batch  = pushdown_reader->ReadPreparedDeviceDctBatch(std::move(crop_plan));
				const auto read_end    = Clock::now();
				result.pushdown.plan_ms += external_plan_overhead_ms(auto_policy_ms, push_batch);
				accumulate_pushdown_window(result.pushdown,
				                           push_batch,
				                           elapsed_ms(read_start, read_end),
				                           pushdown_rowgroups);
			} else {
				++result.auto_full_then_crop_windows;
				const auto read_start = Clock::now();
				auto       full_batch = baseline_reader->ReadPreparedDeviceDctBatch(std::move(full_plan));
				const auto read_end   = Clock::now();

				const auto          transform_start = Clock::now();
				std::vector<size_t> selected_full_blocks =
				    crop_full_batch_to_requested_blocks(*baseline_reader, full_batch, cfg.crop);
				const auto transform_end = Clock::now();

				accumulate_full_then_crop_window(result.full_then_crop,
				                                 full_batch,
				                                 selected_full_blocks.size(),
				                                 external_plan_overhead_ms(auto_policy_ms, full_batch),
				                                 elapsed_ms(read_start, read_end),
				                                 elapsed_ms(transform_start, transform_end),
				                                 baseline_rowgroups);
			}
			continue;
		}

		galp::jpeg::JpegDctDeviceBatch pushdown_batch;
		if (run_pushdown) {
			const auto plan_start    = Clock::now();
			auto       push_requests = make_requests(image_ids, begin, end, cfg.crop);
			auto       push_plan     = pushdown_reader->PrepareDeviceDctBatch(push_requests, batch_options);
			const auto plan_end      = Clock::now();
			const auto read_start    = Clock::now();
			pushdown_batch           = pushdown_reader->ReadPreparedDeviceDctBatch(std::move(push_plan));
			const auto read_end      = Clock::now();
			result.pushdown.plan_ms +=
			    external_plan_overhead_ms(elapsed_ms(plan_start, plan_end), pushdown_batch);
			accumulate_pushdown_window(result.pushdown,
			                           pushdown_batch,
			                           elapsed_ms(read_start, read_end),
			                           pushdown_rowgroups);
		}

		if (run_baseline) {
			const auto plan_start    = Clock::now();
			auto       full_requests = make_requests(image_ids, begin, end, galp::jpeg::JpegDctCropBox {});
			auto       full_plan     = baseline_reader->PrepareDeviceDctBatch(full_requests, batch_options);
			const auto plan_end      = Clock::now();
			const auto read_start    = Clock::now();
			auto       full_batch    = baseline_reader->ReadPreparedDeviceDctBatch(std::move(full_plan));
			const auto read_end      = Clock::now();

			const auto          transform_start = Clock::now();
			std::vector<size_t> selected_full_blocks;
			if (run_pushdown) {
				selected_full_blocks = crop_full_batch_to_pushdown_blocks(pushdown_batch, full_batch);
			} else {
				selected_full_blocks = crop_full_batch_to_requested_blocks(*baseline_reader, full_batch, cfg.crop);
			}
			const auto transform_end = Clock::now();

			accumulate_full_then_crop_window(
			    result.full_then_crop,
			    full_batch,
			    selected_full_blocks.size(),
			    external_plan_overhead_ms(elapsed_ms(plan_start, plan_end), full_batch),
			    elapsed_ms(read_start, read_end),
			    elapsed_ms(transform_start, transform_end),
			    baseline_rowgroups);

			if (cfg.verify_outputs && run_pushdown) {
				const auto verify_start = Clock::now();
				try {
					verify_equal_blocks(pushdown_batch, full_batch, selected_full_blocks);
				} catch (const std::exception& e) {
					result.outputs_match = false;
					result.mismatch      = e.what();
				}
				const auto verify_end = Clock::now();
				result.verify_ms += elapsed_ms(verify_start, verify_end);
			}
		}
	}

	result.pushdown.total_ms = result.pushdown.plan_ms + result.pushdown.read_decode_ms + result.pushdown.transform_ms +
	                           result.pushdown.sink_ms;
	result.full_then_crop.total_ms = result.full_then_crop.plan_ms + result.full_then_crop.read_decode_ms +
	                                 result.full_then_crop.transform_ms + result.full_then_crop.sink_ms;
	if (run_auto) {
		// Each selected auto window folds its policy/prepared-plan time into the chosen stage's plan_ms.
		result.auto_total_ms = result.pushdown.total_ms + result.full_then_crop.total_ms;
	}
	return result;
}

} // namespace galp::execution

#endif // GALP_WITH_JPEG_DCT
