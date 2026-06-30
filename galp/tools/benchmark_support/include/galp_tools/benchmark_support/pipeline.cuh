// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/benchmark_support/include/galp_tools/benchmark_support/pipeline.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_SUPPORT_BENCHMARK_PIPELINE_CUH
#define GALP_SUPPORT_BENCHMARK_PIPELINE_CUH

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace galp::execution {

namespace detail {

constexpr double kAutoVerySmallCropBlockRatio      = 0.125;
constexpr double kAutoMaxLargeWindowBlockRatio     = 0.85;
constexpr double kAutoMaxSelectedBlockRatio        = 0.75;
constexpr double kAutoMaxSelectedVectorRatio       = 0.75;
constexpr double kAutoMaxTouchedRowgroupRatio      = 0.90;
constexpr size_t kAutoMinFullWindowBlocksForGeneralPushdown = 8U * 1024U;
constexpr size_t kAutoLargeFullWindowBlocks        = 64U * 1024U;
constexpr double kAutoMinAvgFullBlocksPerRowgroup  = 1024.0;
constexpr double kAutoMinSelectedVectorRatioWhenWorksetsDoNotShrink = 0.50;

enum class AutoPipelinePolicyReason {
	EmptyWindow,
	CropCoversFullWindow,
	VerySmallCrop,
	SmallWindowFixedOverhead,
	LargeWindowAmortizesPushdown,
	WorksetOverheadTooHigh,
	TinyRowgroupsFixedOverhead,
	TouchesMostRowgroups,
	GatherOutputTooHigh,
	CropSavesEnoughBlocks,
	SavingsTooSmall,
};

inline const char* auto_pipeline_policy_reason_name(const AutoPipelinePolicyReason reason) {
	switch (reason) {
	case AutoPipelinePolicyReason::EmptyWindow:
		return "empty_window";
	case AutoPipelinePolicyReason::CropCoversFullWindow:
		return "crop_covers_full_window";
	case AutoPipelinePolicyReason::VerySmallCrop:
		return "very_small_crop";
	case AutoPipelinePolicyReason::SmallWindowFixedOverhead:
		return "small_window_fixed_overhead";
	case AutoPipelinePolicyReason::LargeWindowAmortizesPushdown:
		return "large_window_amortizes_pushdown";
	case AutoPipelinePolicyReason::WorksetOverheadTooHigh:
		return "workset_overhead_too_high";
	case AutoPipelinePolicyReason::TinyRowgroupsFixedOverhead:
		return "tiny_rowgroups_fixed_overhead";
	case AutoPipelinePolicyReason::TouchesMostRowgroups:
		return "touches_most_rowgroups";
	case AutoPipelinePolicyReason::GatherOutputTooHigh:
		return "gather_output_too_high";
	case AutoPipelinePolicyReason::CropSavesEnoughBlocks:
		return "crop_saves_enough_blocks";
	case AutoPipelinePolicyReason::SavingsTooSmall:
		return "savings_too_small";
	}
	return "unknown";
}

inline size_t count_auto_reuse_candidate_rowgroups(
    const std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata>& rowgroups,
    std::set<std::pair<uint32_t, uint32_t>>&                      seen) {
	size_t repeated = 0;
	for (const auto& rowgroup : rowgroups) {
		if (!seen.insert({rowgroup.shard_id, rowgroup.rowgroup_index}).second) {
			++repeated;
		}
	}
	return repeated;
}

inline size_t estimate_auto_worksets_for_rowgroups(
    const std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata>& rowgroups,
    const size_t                                                  decode_batch_rowgroups =
        galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups) {
	const size_t effective_decode_batch_rowgroups =
	    decode_batch_rowgroups == 0 ? galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups : decode_batch_rowgroups;
	std::unordered_map<uint32_t, size_t> rowgroups_by_shard;
	rowgroups_by_shard.reserve(rowgroups.size());
	for (const auto& rowgroup : rowgroups) {
		++rowgroups_by_shard[rowgroup.shard_id];
	}
	size_t worksets = 0;
	for (const auto& [_, rowgroup_count] : rowgroups_by_shard) {
		worksets += (rowgroup_count + effective_decode_batch_rowgroups - 1U) / effective_decode_batch_rowgroups;
	}
	return worksets;
}

inline double external_plan_overhead_ms(const double measured_plan_ms, const double executed_plan_planning_ms) {
	const double extra_ms = measured_plan_ms - executed_plan_planning_ms;
	return extra_ms < 0.0 ? 0.0 : extra_ms;
}

struct AutoPipelinePolicyDecision {
	bool                     use_pushdown                 = false;
	AutoPipelinePolicyReason reason_code                  = AutoPipelinePolicyReason::EmptyWindow;
	double                   selected_block_ratio         = 0.0;
	double                   selected_vector_ratio        = 0.0;
	double                   touched_rowgroup_ratio       = 0.0;
	double                   avg_full_blocks_per_rowgroup = 0.0;
	size_t                   estimated_pushdown_worksets  = 0;
	size_t                   estimated_full_worksets      = 0;
	size_t                   estimated_pushdown_gather_items = 0;
	size_t                   estimated_full_gather_items     = 0;
	std::string              reason;
};

inline std::string format_auto_pipeline_policy_reason(const AutoPipelinePolicyDecision& decision) {
	return std::string(auto_pipeline_policy_reason_name(decision.reason_code)) +
	       ",selected_block_ratio=" + std::to_string(decision.selected_block_ratio) +
	       ",selected_vector_ratio=" + std::to_string(decision.selected_vector_ratio) +
	       ",touched_rowgroup_ratio=" + std::to_string(decision.touched_rowgroup_ratio) +
	       ",avg_full_blocks_per_rowgroup=" + std::to_string(decision.avg_full_blocks_per_rowgroup) +
	       ",estimated_pushdown_worksets=" + std::to_string(decision.estimated_pushdown_worksets) +
	       ",estimated_full_worksets=" + std::to_string(decision.estimated_full_worksets) +
	       ",estimated_pushdown_gather_items=" + std::to_string(decision.estimated_pushdown_gather_items) +
	       ",estimated_full_gather_items=" + std::to_string(decision.estimated_full_gather_items);
}

inline AutoPipelinePolicyDecision choose_auto_pipeline_policy_from_estimates(
    const size_t selected_blocks,
    const size_t full_blocks,
    const size_t touched_rowgroups,
    const size_t full_rowgroups,
    const size_t selected_vectors,
    const size_t full_vectors,
    const size_t estimated_pushdown_worksets,
    const size_t estimated_full_worksets) {
	AutoPipelinePolicyDecision decision;
	decision.selected_block_ratio =
	    full_blocks == 0 ? 0.0 : static_cast<double>(selected_blocks) / static_cast<double>(full_blocks);
	decision.selected_vector_ratio =
	    full_vectors == 0 ? decision.selected_block_ratio
	                      : static_cast<double>(selected_vectors) / static_cast<double>(full_vectors);
	decision.touched_rowgroup_ratio =
	    full_rowgroups == 0 ? 0.0 : static_cast<double>(touched_rowgroups) / static_cast<double>(full_rowgroups);
	decision.avg_full_blocks_per_rowgroup =
	    full_rowgroups == 0 ? 0.0 : static_cast<double>(full_blocks) / static_cast<double>(full_rowgroups);
	decision.estimated_pushdown_worksets = estimated_pushdown_worksets;
	decision.estimated_full_worksets     = estimated_full_worksets;
	decision.estimated_pushdown_gather_items = selected_blocks;
	decision.estimated_full_gather_items     = full_blocks;

	if (full_blocks == 0 || selected_blocks == 0) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::EmptyWindow;
	} else if (selected_blocks >= full_blocks || decision.selected_vector_ratio >= 0.95) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::CropCoversFullWindow;
	} else if (full_blocks < kAutoMinFullWindowBlocksForGeneralPushdown) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::SmallWindowFixedOverhead;
	} else if (decision.selected_block_ratio <= kAutoVerySmallCropBlockRatio &&
	           decision.selected_vector_ratio <= kAutoMaxSelectedVectorRatio) {
		decision.use_pushdown = true;
		decision.reason_code  = AutoPipelinePolicyReason::VerySmallCrop;
	} else if (full_blocks >= kAutoLargeFullWindowBlocks &&
	           decision.selected_block_ratio < kAutoMaxLargeWindowBlockRatio &&
	           decision.selected_vector_ratio < kAutoMaxLargeWindowBlockRatio) {
		decision.use_pushdown = true;
		decision.reason_code  = AutoPipelinePolicyReason::LargeWindowAmortizesPushdown;
	} else if (decision.avg_full_blocks_per_rowgroup < kAutoMinAvgFullBlocksPerRowgroup) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::TinyRowgroupsFixedOverhead;
	} else if (decision.touched_rowgroup_ratio >= kAutoMaxTouchedRowgroupRatio) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::TouchesMostRowgroups;
	} else if (decision.estimated_pushdown_worksets >= decision.estimated_full_worksets &&
	           decision.selected_vector_ratio >= kAutoMinSelectedVectorRatioWhenWorksetsDoNotShrink) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::WorksetOverheadTooHigh;
	} else if (decision.selected_block_ratio >= kAutoMaxSelectedBlockRatio) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::GatherOutputTooHigh;
	} else if (decision.selected_vector_ratio < kAutoMaxSelectedVectorRatio &&
	           decision.selected_block_ratio < kAutoMaxSelectedBlockRatio) {
		decision.use_pushdown = true;
		decision.reason_code  = AutoPipelinePolicyReason::CropSavesEnoughBlocks;
	} else {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::SavingsTooSmall;
	}

	decision.reason = format_auto_pipeline_policy_reason(decision);
	return decision;
}

inline AutoPipelinePolicyDecision choose_auto_pipeline_policy_from_counts(const size_t selected_blocks,
                                                                         const size_t full_blocks,
                                                                         const size_t touched_rowgroups,
                                                                         const size_t full_rowgroups,
                                                                         const size_t selected_vectors = 0,
                                                                         const size_t full_vectors     = 0,
                                                                         const size_t decode_batch_rowgroups =
                                                                             galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups) {
	const size_t effective_decode_batch_rowgroups =
	    decode_batch_rowgroups == 0 ? galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups : decode_batch_rowgroups;
	return choose_auto_pipeline_policy_from_estimates(
	    selected_blocks,
	    full_blocks,
	    touched_rowgroups,
	    full_rowgroups,
	    selected_vectors,
	    full_vectors,
	    (touched_rowgroups + effective_decode_batch_rowgroups - 1U) / effective_decode_batch_rowgroups,
	    (full_rowgroups + effective_decode_batch_rowgroups - 1U) / effective_decode_batch_rowgroups);
}

} // namespace detail

enum class PipelineBenchmarkMode {
	Pushdown,
	FullThenCrop,
	Compare,
	Auto,
};

struct PipelineBenchmarkConfig {
	std::vector<uint32_t>      image_ids;
	galp::jpeg::JpegDctCropBox crop {};
	PipelineBenchmarkMode      mode                 = PipelineBenchmarkMode::Compare;
	size_t                     window_images        = 256;
	size_t                     cache_capacity_bytes = 0;
	size_t                     decode_batch_rowgroups = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups;
	bool                       verify_outputs       = true;
};

struct PipelineBenchmarkStageResult {
	size_t windows                   = 0;
	size_t requests                  = 0;
	size_t input_blocks              = 0;
	size_t output_blocks             = 0;
	size_t output_coefficients       = 0;
	size_t output_bytes              = 0;
	size_t rowgroup_visits           = 0;
	size_t unique_rowgroups          = 0;
	size_t repeated_rowgroups        = 0;
	size_t cache_hits                = 0;
	size_t cache_misses              = 0;
	size_t dense_cache_hits          = 0;
	size_t dense_cache_misses        = 0;
	size_t cache_inserts             = 0;
	size_t cache_evictions           = 0;
	size_t cache_resident_bytes      = 0;
	size_t cache_resident_rowgroups  = 0;
	size_t planned_selected_vector_count = 0;
	size_t selected_vector_count     = 0;
	size_t full_vector_count         = 0;
	size_t planned_saved_vector_count = 0;
	size_t actual_saved_vector_count  = 0;
	size_t rowgroup_count             = 0;
	double planned_selected_vector_ratio = 0.0;
	double selected_vector_ratio     = 0.0;
	size_t workset_count             = 0;
	size_t decode_kernel_launch_count = 0;
	size_t gather_kernel_launch_count = 0;
	size_t cached_gather_kernel_launch_count = 0;
	size_t materialize_kernel_launch_count = 0;
	size_t gather_item_count          = 0;
	size_t decoded_gather_item_count  = 0;
	size_t cached_gather_item_count   = 0;
	size_t workset_upload_count       = 0;
	size_t scratch_upload_count       = 0;
	size_t scratch_allocation_count   = 0;
	size_t internal_sync_count        = 0;
	size_t cached_gather_sync_count   = 0;
	size_t decoded_batch_sync_count   = 0;
	size_t cached_gather_event_handoff_count = 0;
	size_t sparse_vector_cache_hits   = 0;
	size_t sparse_vector_cache_misses = 0;
	size_t runtime_policy_selected_rowgroups = 0;
	size_t runtime_policy_full_rowgroups     = 0;
	size_t runtime_policy_tail_full_rowgroups = 0;
	size_t runtime_policy_ratio_full_rowgroups = 0;
	size_t runtime_policy_low_saving_full_rowgroups = 0;
	double device_planning_ms         = 0.0;
	double workset_build_ms           = 0.0;
	double workset_upload_ms          = 0.0;
	double decode_ms                  = 0.0;
	double gather_ms                  = 0.0;
	std::string runtime_policy_decision;
	std::string runtime_policy_reason;
	size_t peak_window_images        = 0;
	size_t peak_window_input_blocks  = 0;
	size_t peak_window_output_blocks = 0;
	size_t peak_window_output_bytes  = 0;
	double plan_ms                   = 0.0;
	double read_decode_ms            = 0.0;
	double transform_ms              = 0.0;
	double sink_ms                   = 0.0;
	double total_ms                  = 0.0;
};

struct PipelineBenchmarkResult {
	uint64_t                     dataset_images = 0;
	PipelineBenchmarkMode        mode           = PipelineBenchmarkMode::Compare;
	PipelineBenchmarkStageResult pushdown;
	PipelineBenchmarkStageResult full_then_crop;
	size_t                       auto_pushdown_windows       = 0;
	size_t                       auto_full_then_crop_windows = 0;
	double                       auto_policy_ms              = 0.0;
	double                       auto_total_ms               = 0.0;
	std::string                  auto_policy_reason;
	size_t                       auto_policy_selected_blocks = 0;
	size_t                       auto_policy_full_blocks = 0;
	size_t                       auto_policy_selected_vectors = 0;
	size_t                       auto_policy_full_vectors = 0;
	size_t                       auto_policy_touched_rowgroups = 0;
	size_t                       auto_policy_full_rowgroups = 0;
	size_t                       auto_policy_estimated_pushdown_worksets = 0;
	size_t                       auto_policy_estimated_full_worksets = 0;
	size_t                       auto_policy_estimated_pushdown_gather_items = 0;
	size_t                       auto_policy_estimated_full_gather_items = 0;
	size_t                       auto_policy_pushdown_reuse_candidate_rowgroups = 0;
	size_t                       auto_policy_full_reuse_candidate_rowgroups = 0;
	double                       auto_policy_selected_block_ratio = 0.0;
	double                       auto_policy_selected_vector_ratio = 0.0;
	double                       auto_policy_touched_rowgroup_ratio = 0.0;
	double                       auto_policy_pushdown_reuse_candidate_ratio = 0.0;
	double                       auto_policy_full_reuse_candidate_ratio = 0.0;
	double                       auto_policy_avg_full_blocks_per_rowgroup = 0.0;
	size_t                       auto_policy_empty_windows = 0;
	size_t                       auto_policy_crop_covers_full_windows = 0;
	size_t                       auto_policy_very_small_crop_windows = 0;
	size_t                       auto_policy_small_window_full_windows = 0;
	size_t                       auto_policy_large_window_pushdown_windows = 0;
	size_t                       auto_policy_workset_overhead_full_windows = 0;
	size_t                       auto_policy_tiny_rowgroups_full_windows = 0;
	size_t                       auto_policy_touches_most_rowgroups_windows = 0;
	size_t                       auto_policy_gather_output_full_windows = 0;
	size_t                       auto_policy_saves_enough_blocks_windows = 0;
	size_t                       auto_policy_savings_too_small_windows = 0;
	bool                         outputs_match = true;
	double                       verify_ms     = 0.0;
	std::string                  mismatch;
};

PipelineBenchmarkResult benchmark_jpeg_dct_pipeline(const std::filesystem::path&   manifest_path,
                                                    const PipelineBenchmarkConfig& cfg = {});

} // namespace galp::execution

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_SUPPORT_BENCHMARK_PIPELINE_CUH
