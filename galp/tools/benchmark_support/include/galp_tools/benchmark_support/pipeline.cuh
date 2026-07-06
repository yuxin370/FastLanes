// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tools/benchmark_support/include/galp_tools/benchmark_support/pipeline.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_SUPPORT_BENCHMARK_PIPELINE_CUH
#define GALP_SUPPORT_BENCHMARK_PIPELINE_CUH

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "codecs/consts.cuh"
#include "galp/jpeg_dct.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace galp::execution {

namespace detail {

constexpr size_t kAutoJpegDctCoefficientCount                       = 64;
constexpr double kAutoVerySmallCropBlockRatio                       = 0.125;
constexpr double kAutoMaxLargeWindowBlockRatio                      = 0.85;
constexpr double kAutoMaxSelectedBlockRatio                         = 0.75;
constexpr double kAutoMaxSelectedVectorRatio                        = 0.75;
constexpr double kAutoMaxTouchedRowgroupRatio                       = 0.90;
constexpr size_t kAutoMinFullWindowBlocksForGeneralPushdown         = 8U * 1024U;
constexpr size_t kAutoLargeFullWindowBlocks                         = 64U * 1024U;
constexpr double kAutoMinAvgFullBlocksPerRowgroup                   = 1024.0;
constexpr double kAutoMaxTinyRowgroupPushdownBlockRatio             = 0.30;
constexpr double kAutoMaxTinyRowgroupPushdownTouchedRowgroupRatio   = 0.40;
constexpr double kAutoMinSelectedVectorRatioWhenWorksetsDoNotShrink = 0.50;
constexpr double kAutoMinMetadataFastPushdownAvgBlocksPerImage      = 64.0;
constexpr double kAutoMaxCoefficientPushdownRatio                   = 0.25;
constexpr size_t kAutoMinCoefficientPushdownBlocks                  = 1024U;
constexpr double kAutoMinCoefficientPushdownBlocksPerWorkset        = 1024.0;
constexpr size_t kAutoMinCoefficientPushdownSavedCoefficients       = 64U * 1024U;
constexpr size_t kAutoMinCoefficientPushdownSavedDecodedBytes       = 2U * 1024U * 1024U;

enum class AutoPipelinePolicyReason {
	EmptyWindow,
	CropCoversFullWindow,
	CoefficientSelectionPushdown,
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
	case AutoPipelinePolicyReason::CoefficientSelectionPushdown:
		return "coefficient_selection_pushdown";
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

inline size_t
count_auto_reuse_candidate_rowgroups(const std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata>& rowgroups,
                                     std::set<std::pair<uint32_t, uint32_t>>&                      seen) {
	size_t repeated = 0;
	for (const auto& rowgroup : rowgroups) {
		if (!seen.insert({rowgroup.shard_id, rowgroup.rowgroup_index}).second) {
			++repeated;
		}
	}
	return repeated;
}

inline size_t
estimate_auto_reuse_candidate_rowgroups(const std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata>& rowgroups,
                                        const std::set<std::pair<uint32_t, uint32_t>>&                seen) {
	size_t repeated = 0;
	for (const auto& rowgroup : rowgroups) {
		if (seen.find({rowgroup.shard_id, rowgroup.rowgroup_index}) != seen.end()) {
			++repeated;
		}
	}
	return repeated;
}

inline size_t estimate_auto_worksets_for_rowgroups(
    const std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata>& rowgroups,
    const size_t decode_batch_rowgroups = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups) {
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
	bool                     use_pushdown                    = false;
	AutoPipelinePolicyReason reason_code                     = AutoPipelinePolicyReason::EmptyWindow;
	double                   selected_block_ratio            = 0.0;
	double                   selected_vector_ratio           = 0.0;
	double                   touched_rowgroup_ratio          = 0.0;
	double                   avg_full_blocks_per_rowgroup    = 0.0;
	size_t                   estimated_pushdown_worksets     = 0;
	size_t                   estimated_full_worksets         = 0;
	size_t                   estimated_pushdown_gather_items = 0;
	size_t                   estimated_full_gather_items     = 0;
	size_t                   selected_coefficient_count      = kAutoJpegDctCoefficientCount;
	size_t                   active_physical_coefficient_count = kAutoJpegDctCoefficientCount;
	double                   selected_coefficient_ratio      = 1.0;
	size_t                   estimated_pushdown_materialization_items = 0;
	size_t                   estimated_full_materialization_items     = 0;
	size_t                   estimated_pushdown_syncs                = 0;
	size_t                   estimated_full_syncs                    = 0;
	double                   selected_blocks_per_pushdown_workset    = 0.0;
	size_t                   estimated_pushdown_reuse_candidate_rowgroups = 0;
	size_t                   estimated_full_reuse_candidate_rowgroups     = 0;
	size_t                   estimated_pushdown_decoded_bytes        = 0;
	size_t                   estimated_full_decoded_bytes            = 0;
	size_t                   estimated_output_bytes                  = 0;
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
	       ",estimated_full_gather_items=" + std::to_string(decision.estimated_full_gather_items) +
	       ",selected_coefficients=" + std::to_string(decision.selected_coefficient_count) +
	       ",active_physical_coefficients=" + std::to_string(decision.active_physical_coefficient_count) +
	       ",selected_coefficient_ratio=" + std::to_string(decision.selected_coefficient_ratio) +
	       ",estimated_pushdown_materialization_items=" +
	           std::to_string(decision.estimated_pushdown_materialization_items) +
	       ",estimated_full_materialization_items=" +
	           std::to_string(decision.estimated_full_materialization_items) +
	       ",estimated_pushdown_syncs=" + std::to_string(decision.estimated_pushdown_syncs) +
	       ",estimated_full_syncs=" + std::to_string(decision.estimated_full_syncs) +
	       ",selected_blocks_per_pushdown_workset=" +
	           std::to_string(decision.selected_blocks_per_pushdown_workset) +
	       ",estimated_pushdown_reuse_candidate_rowgroups=" +
	           std::to_string(decision.estimated_pushdown_reuse_candidate_rowgroups) +
	       ",estimated_full_reuse_candidate_rowgroups=" +
	           std::to_string(decision.estimated_full_reuse_candidate_rowgroups) +
	       ",estimated_pushdown_decoded_bytes=" + std::to_string(decision.estimated_pushdown_decoded_bytes) +
	       ",estimated_full_decoded_bytes=" + std::to_string(decision.estimated_full_decoded_bytes) +
	       ",estimated_output_bytes=" + std::to_string(decision.estimated_output_bytes);
}

inline AutoPipelinePolicyDecision choose_auto_pipeline_policy_from_estimates(const size_t selected_blocks,
                                                                             const size_t full_blocks,
                                                                             const size_t touched_rowgroups,
                                                                             const size_t full_rowgroups,
                                                                             const size_t selected_vectors,
                                                                             const size_t full_vectors,
                                                                             const size_t estimated_pushdown_worksets,
                                                                             const size_t estimated_full_worksets,
                                                                             const size_t selected_coefficients = kAutoJpegDctCoefficientCount,
                                                                             const size_t active_physical_coefficients = kAutoJpegDctCoefficientCount,
                                                                             const size_t estimated_pushdown_reuse_candidate_rowgroups = 0,
                                                                             const size_t estimated_full_reuse_candidate_rowgroups = 0) {
	AutoPipelinePolicyDecision decision;
	const size_t normalized_selected_coefficients =
	    selected_coefficients == 0 ? kAutoJpegDctCoefficientCount : selected_coefficients;
	const size_t normalized_active_physical_coefficients =
	    active_physical_coefficients == 0 ? normalized_selected_coefficients : active_physical_coefficients;
	decision.selected_block_ratio =
	    full_blocks == 0 ? 0.0 : static_cast<double>(selected_blocks) / static_cast<double>(full_blocks);
	decision.selected_vector_ratio = full_vectors == 0
	                                     ? decision.selected_block_ratio
	                                     : static_cast<double>(selected_vectors) / static_cast<double>(full_vectors);
	decision.touched_rowgroup_ratio =
	    full_rowgroups == 0 ? 0.0 : static_cast<double>(touched_rowgroups) / static_cast<double>(full_rowgroups);
	decision.avg_full_blocks_per_rowgroup =
	    full_rowgroups == 0 ? 0.0 : static_cast<double>(full_blocks) / static_cast<double>(full_rowgroups);
	decision.estimated_pushdown_worksets     = estimated_pushdown_worksets;
	decision.estimated_full_worksets         = estimated_full_worksets;
	decision.estimated_pushdown_gather_items = selected_blocks;
	decision.estimated_full_gather_items     = full_blocks;
	decision.selected_coefficient_count      = normalized_selected_coefficients;
	decision.active_physical_coefficient_count = normalized_active_physical_coefficients;
	decision.selected_coefficient_ratio =
	    static_cast<double>(normalized_selected_coefficients) / static_cast<double>(kAutoJpegDctCoefficientCount);
	decision.estimated_pushdown_materialization_items = selected_blocks * normalized_selected_coefficients;
	decision.estimated_full_materialization_items     = full_blocks * normalized_selected_coefficients;
	decision.estimated_pushdown_syncs                 = estimated_pushdown_worksets;
	decision.estimated_full_syncs                     = estimated_full_worksets;
	decision.selected_blocks_per_pushdown_workset =
	    estimated_pushdown_worksets == 0
	        ? static_cast<double>(selected_blocks)
	        : static_cast<double>(selected_blocks) / static_cast<double>(estimated_pushdown_worksets);
	decision.estimated_pushdown_reuse_candidate_rowgroups = estimated_pushdown_reuse_candidate_rowgroups;
	decision.estimated_full_reuse_candidate_rowgroups     = estimated_full_reuse_candidate_rowgroups;
	decision.estimated_pushdown_decoded_bytes =
	    selected_vectors * galp::codec::consts::VALUES_PER_VECTOR * normalized_active_physical_coefficients *
	    sizeof(int16_t);
	decision.estimated_full_decoded_bytes =
	    full_vectors * galp::codec::consts::VALUES_PER_VECTOR * kAutoJpegDctCoefficientCount * sizeof(int16_t);
	decision.estimated_output_bytes = selected_blocks * normalized_selected_coefficients * sizeof(int16_t);

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
		// 2026-07 50%+ crop sweep: tiny-imagenet still benefits from pushdown when
		// it keeps about 25% of blocks and touches about 25% of rowgroups, while
		// svhn/cifar10 remain slower at similar block ratios but about 50% rowgroup
		// touch ratios. Let strong rowgroup pruning bypass the tiny-rowgroup guard.
	} else if (decision.avg_full_blocks_per_rowgroup < kAutoMinAvgFullBlocksPerRowgroup &&
	           decision.selected_block_ratio <= kAutoMaxTinyRowgroupPushdownBlockRatio &&
	           decision.selected_vector_ratio < kAutoMaxSelectedVectorRatio &&
	           decision.touched_rowgroup_ratio <= kAutoMaxTinyRowgroupPushdownTouchedRowgroupRatio) {
		decision.use_pushdown = true;
		decision.reason_code  = AutoPipelinePolicyReason::CropSavesEnoughBlocks;
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

inline AutoPipelinePolicyDecision
choose_auto_coefficient_selection_policy_from_estimates(const size_t selected_blocks,
                                                        const size_t full_blocks,
                                                        const size_t touched_rowgroups,
                                                        const size_t full_rowgroups,
                                                        const size_t selected_vectors,
                                                        const size_t full_vectors,
                                                        const size_t estimated_pushdown_worksets,
                                                        const size_t estimated_full_worksets,
                                                        const size_t selected_coefficients,
                                                        const size_t active_physical_coefficients = 0,
                                                        const size_t estimated_pushdown_reuse_candidate_rowgroups = 0,
                                                        const size_t estimated_full_reuse_candidate_rowgroups = 0) {
	auto         decision          = choose_auto_pipeline_policy_from_estimates(selected_blocks,
                                                               full_blocks,
                                                               touched_rowgroups,
                                                               full_rowgroups,
                                                               selected_vectors,
                                                               full_vectors,
                                                               estimated_pushdown_worksets,
                                                               estimated_full_worksets,
                                                               selected_coefficients,
                                                               active_physical_coefficients == 0
                                                                   ? selected_coefficients
                                                                   : active_physical_coefficients,
                                                               estimated_pushdown_reuse_candidate_rowgroups,
                                                               estimated_full_reuse_candidate_rowgroups);
	const size_t full_coefficients = kAutoJpegDctCoefficientCount;
	const double selected_ratio =
	    full_coefficients == 0 ? 1.0
	                           : static_cast<double>(selected_coefficients) / static_cast<double>(full_coefficients);
	const size_t saved_coefficients_per_block =
	    full_coefficients > selected_coefficients ? full_coefficients - selected_coefficients : 0U;
	const bool saves_enough_coefficients =
	    saved_coefficients_per_block != 0U &&
	    selected_blocks >= (kAutoMinCoefficientPushdownSavedCoefficients + saved_coefficients_per_block - 1U) /
	                           saved_coefficients_per_block;
	const size_t decoded_bytes_saved =
	    decision.estimated_full_decoded_bytes > decision.estimated_pushdown_decoded_bytes
	        ? decision.estimated_full_decoded_bytes - decision.estimated_pushdown_decoded_bytes
	        : 0U;
	const bool saves_enough_decoded_bytes = decoded_bytes_saved >= kAutoMinCoefficientPushdownSavedDecodedBytes;
	const bool has_enough_blocks_per_workset =
	    decision.selected_blocks_per_pushdown_workset >= kAutoMinCoefficientPushdownBlocksPerWorkset;
	const bool workset_syncs_do_not_increase =
	    decision.estimated_pushdown_worksets <= decision.estimated_full_worksets &&
	    decision.estimated_pushdown_syncs <= decision.estimated_full_syncs;
	if (selected_coefficients != 0U && selected_coefficients < full_coefficients &&
	    selected_ratio <= kAutoMaxCoefficientPushdownRatio && selected_blocks >= kAutoMinCoefficientPushdownBlocks &&
	    saves_enough_coefficients && saves_enough_decoded_bytes && has_enough_blocks_per_workset &&
	    workset_syncs_do_not_increase) {
		decision.use_pushdown = true;
		decision.reason_code  = AutoPipelinePolicyReason::CoefficientSelectionPushdown;
	} else if (!decision.use_pushdown) {
		decision.use_pushdown = false;
		decision.reason_code  = AutoPipelinePolicyReason::SavingsTooSmall;
	}
	decision.reason = format_auto_pipeline_policy_reason(decision);
	return decision;
}

inline AutoPipelinePolicyDecision make_auto_pipeline_fast_policy_decision(const size_t selected_blocks,
                                                                          const size_t full_blocks,
                                                                          const size_t image_count,
                                                                          const bool   use_pushdown,
                                                                          const AutoPipelinePolicyReason reason) {
	AutoPipelinePolicyDecision decision;
	decision.use_pushdown = use_pushdown;
	decision.reason_code  = reason;
	decision.selected_block_ratio =
	    full_blocks == 0 ? 0.0 : static_cast<double>(selected_blocks) / static_cast<double>(full_blocks);
	// The fast path intentionally avoids vector/rowgroup planning. Use block ratio
	// as the fast decision proxy and mark the reason so summaries do not treat it
	// as a precise vector estimate.
	decision.selected_vector_ratio           = decision.selected_block_ratio;
	decision.touched_rowgroup_ratio          = 0.0;
	decision.avg_full_blocks_per_rowgroup    = 0.0;
	decision.estimated_pushdown_worksets     = image_count == 0 ? 0U : 1U;
	decision.estimated_full_worksets         = image_count == 0 ? 0U : 1U;
	decision.estimated_pushdown_gather_items = selected_blocks;
	decision.estimated_full_gather_items     = full_blocks;
	decision.reason                          = format_auto_pipeline_policy_reason(decision) + ",metadata_fast=1";
	return decision;
}

inline std::optional<AutoPipelinePolicyDecision> choose_auto_pipeline_policy_from_block_estimate(
    const size_t selected_blocks, const size_t full_blocks, const size_t image_count) {
	const double selected_block_ratio =
	    full_blocks == 0 ? 0.0 : static_cast<double>(selected_blocks) / static_cast<double>(full_blocks);
	const double avg_full_blocks_per_image =
	    image_count == 0 ? 0.0 : static_cast<double>(full_blocks) / static_cast<double>(image_count);

	if (full_blocks == 0 || selected_blocks == 0) {
		return make_auto_pipeline_fast_policy_decision(
		    selected_blocks, full_blocks, image_count, false, AutoPipelinePolicyReason::EmptyWindow);
	}
	if (selected_blocks >= full_blocks || selected_block_ratio >= 0.95) {
		return make_auto_pipeline_fast_policy_decision(
		    selected_blocks, full_blocks, image_count, false, AutoPipelinePolicyReason::CropCoversFullWindow);
	}
	if (full_blocks < kAutoMinFullWindowBlocksForGeneralPushdown) {
		return make_auto_pipeline_fast_policy_decision(
		    selected_blocks, full_blocks, image_count, false, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
	}
	// Pushdown decisions depend on selected-vector and workset estimates, so
	// block-only metadata gates only make fast reject decisions.
	if (selected_block_ratio <= kAutoVerySmallCropBlockRatio) {
		return std::nullopt;
	}
	if (full_blocks >= kAutoLargeFullWindowBlocks && selected_block_ratio < kAutoMaxLargeWindowBlockRatio) {
		return std::nullopt;
	}
	if (selected_block_ratio < kAutoMaxSelectedBlockRatio) {
		return std::nullopt;
	}
	if (avg_full_blocks_per_image < kAutoMinMetadataFastPushdownAvgBlocksPerImage) {
		return make_auto_pipeline_fast_policy_decision(
		    selected_blocks, full_blocks, image_count, false, AutoPipelinePolicyReason::SmallWindowFixedOverhead);
	}
	if (selected_block_ratio >= kAutoMaxSelectedBlockRatio) {
		return make_auto_pipeline_fast_policy_decision(
		    selected_blocks, full_blocks, image_count, false, AutoPipelinePolicyReason::GatherOutputTooHigh);
	}
	// CropSavesEnoughBlocks depends on touched_rowgroup_ratio for tiny rowgroups;
	// metadata-only block estimates must fall back to the exact plan there.
	return std::nullopt;
}

inline AutoPipelinePolicyDecision choose_auto_pipeline_policy_from_counts(
    const size_t selected_blocks,
    const size_t full_blocks,
    const size_t touched_rowgroups,
    const size_t full_rowgroups,
    const size_t selected_vectors       = 0,
    const size_t full_vectors           = 0,
    const size_t decode_batch_rowgroups = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups) {
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
	DctCompare,
};

struct PipelineBenchmarkConfig {
	std::vector<uint32_t>                   image_ids;
	galp::jpeg::JpegDctCropBox              crop {};
	galp::jpeg::JpegDctCoefficientSelection coefficient_selection {};
	PipelineBenchmarkMode                   mode                     = PipelineBenchmarkMode::Compare;
	size_t                                  window_images            = 256;
	size_t                                  cache_capacity_bytes     = 0;
	size_t                                  decode_batch_rowgroups   = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups;
	bool                                    enable_rowgroup_prefetch = true;
	size_t rowgroup_prefetch_depth              = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchDepth;
	size_t rowgroup_prefetch_workers            = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchWorkers;
	size_t rowgroup_prefetch_min_decode_batches = galp::jpeg::kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches;
	bool   verify_outputs                       = true;
};

struct PipelineBenchmarkStageResult {
	size_t      windows                                       = 0;
	size_t      requests                                      = 0;
	size_t      input_blocks                                  = 0;
	size_t      output_blocks                                 = 0;
	size_t      selected_coefficient_count                    = 64;
	size_t      full_coefficient_count                        = 64;
	double      selected_coefficient_ratio                    = 1.0;
	size_t      coefficients_per_block                        = 64;
	size_t      output_coefficients                           = 0;
	size_t      output_bytes                                  = 0;
	size_t      decoded_coefficients_per_block                = 64;
	size_t      decoded_coefficients                          = 0;
	size_t      decoded_bytes                                 = 0;
	size_t      rowgroup_visits                               = 0;
	size_t      unique_rowgroups                              = 0;
	size_t      repeated_rowgroups                            = 0;
	size_t      cache_hits                                    = 0;
	size_t      cache_misses                                  = 0;
	size_t      dense_cache_hits                              = 0;
	size_t      dense_cache_misses                            = 0;
	size_t      cache_inserts                                 = 0;
	size_t      cache_evictions                               = 0;
	size_t      cache_resident_bytes                          = 0;
	size_t      cache_resident_rowgroups                      = 0;
	size_t      planned_selected_vector_count                 = 0;
	size_t      selected_vector_count                         = 0;
	size_t      full_vector_count                             = 0;
	size_t      planned_saved_vector_count                    = 0;
	size_t      actual_saved_vector_count                     = 0;
	size_t      rowgroup_count                                = 0;
	double      planned_selected_vector_ratio                 = 0.0;
	double      selected_vector_ratio                         = 0.0;
	size_t      workset_count                                 = 0;
	size_t      decode_kernel_launch_count                    = 0;
	size_t      gather_kernel_launch_count                    = 0;
	size_t      prefix_gather_kernel_launch_count             = 0;
	size_t      cached_gather_kernel_launch_count             = 0;
	size_t      materialize_kernel_launch_count               = 0;
	size_t      gather_item_count                             = 0;
	size_t      decoded_gather_item_count                     = 0;
	size_t      cached_gather_item_count                      = 0;
	size_t      projection_item_count                         = 0;
	size_t      decoded_projection_item_count                 = 0;
	size_t      workset_upload_count                          = 0;
	size_t      scratch_upload_count                          = 0;
	size_t      scratch_allocation_count                      = 0;
	size_t      internal_sync_count                           = 0;
	size_t      cached_gather_sync_count                      = 0;
	size_t      decoded_batch_sync_count                      = 0;
	size_t      cached_gather_event_handoff_count             = 0;
	size_t      sparse_vector_cache_hits                      = 0;
	size_t      sparse_vector_cache_misses                    = 0;
	size_t      runtime_policy_selected_rowgroups             = 0;
	size_t      runtime_policy_full_rowgroups                 = 0;
	size_t      runtime_policy_tail_full_rowgroups            = 0;
	size_t      runtime_policy_ratio_full_rowgroups           = 0;
	size_t      runtime_policy_low_saving_full_rowgroups      = 0;
	size_t      prefetch_initial_cache_hit_rowgroup_count     = 0;
	size_t      prefetch_candidate_rowgroup_count             = 0;
	size_t      prefetch_active_shard_count                   = 0;
	size_t      prefetch_config_disabled_shard_count          = 0;
	size_t      prefetch_all_hit_shard_count                  = 0;
	size_t      prefetch_small_batch_disabled_shard_count     = 0;
	size_t      prefetch_selected_vector_disabled_shard_count = 0;
	size_t      prefetch_selected_vector_miss_rowgroup_count  = 0;
	size_t      prefetch_initial_hit_runtime_miss_count       = 0;
	size_t      prefetch_skipped_repeated_runtime_miss_count  = 0;
	size_t      prefetched_rowgroup_count                     = 0;
	size_t      prefetch_consumed_as_hit_count                = 0;
	size_t      prefetch_skipped_repeated_rowgroup_count      = 0;
	double      prefetch_consumed_as_hit_read_ms              = 0.0;
	double      prefetch_consumed_as_hit_wait_ms              = 0.0;
	double      device_planning_ms                            = 0.0;
	double      workset_build_ms                              = 0.0;
	double      workset_upload_ms                             = 0.0;
	double      decode_ms                                     = 0.0;
	double      gather_ms                                     = 0.0;
	double      decoded_gather_ms                             = 0.0;
	double      cached_gather_ms                              = 0.0;
	double      projection_ms                                 = 0.0;
	double      decoded_projection_ms                         = 0.0;
	double      prefetch_wait_ms                              = 0.0;
	double      prefetch_depth_block_ms                       = 0.0;
	double      prefetch_queue_start_ms                       = 0.0;
	double      prefetch_rowgroup_read_ms                     = 0.0;
	double      prefetch_ready_ahead_ms                       = 0.0;
	double      sync_rowgroup_read_ms                         = 0.0;
	std::string runtime_policy_decision;
	std::string runtime_policy_reason;
	size_t      peak_window_images        = 0;
	size_t      peak_window_input_blocks  = 0;
	size_t      peak_window_output_blocks = 0;
	size_t      peak_window_output_bytes  = 0;
	double      plan_ms                   = 0.0;
	double      read_decode_ms            = 0.0;
	double      transform_ms              = 0.0;
	double      sink_ms                   = 0.0;
	double      total_ms                  = 0.0;
};

struct PipelineBenchmarkResult {
	uint64_t                     dataset_images = 0;
		PipelineBenchmarkMode        mode           = PipelineBenchmarkMode::Compare;
		PipelineBenchmarkStageResult pushdown;
		PipelineBenchmarkStageResult full_then_crop;
		PipelineBenchmarkStageResult auto_no_dct_pushdown;
		PipelineBenchmarkStageResult dct_post_decode;
		size_t                       auto_pushdown_windows       = 0;
		size_t                       auto_full_then_crop_windows = 0;
		size_t                       auto_no_dct_pushdown_windows = 0;
		double                       auto_policy_ms              = 0.0;
	double                       auto_total_ms               = 0.0;
	std::string                  auto_policy_reason;
	size_t                       auto_policy_fast_gate_windows                  = 0;
	size_t                       auto_policy_estimate_windows                   = 0;
	size_t                       auto_policy_selected_blocks                    = 0;
	size_t                       auto_policy_full_blocks                        = 0;
	size_t                       auto_policy_selected_vectors                   = 0;
	size_t                       auto_policy_full_vectors                       = 0;
	size_t                       auto_policy_touched_rowgroups                  = 0;
	size_t                       auto_policy_full_rowgroups                     = 0;
	size_t                       auto_policy_estimated_pushdown_worksets        = 0;
	size_t                       auto_policy_estimated_full_worksets            = 0;
	size_t                       auto_policy_estimated_pushdown_gather_items    = 0;
	size_t                       auto_policy_estimated_full_gather_items        = 0;
	size_t                       auto_policy_pushdown_reuse_candidate_rowgroups = 0;
	size_t                       auto_policy_full_reuse_candidate_rowgroups     = 0;
	double                       auto_policy_selected_block_ratio               = 0.0;
	double                       auto_policy_selected_vector_ratio              = 0.0;
	double                       auto_policy_touched_rowgroup_ratio             = 0.0;
	double                       auto_policy_pushdown_reuse_candidate_ratio     = 0.0;
	double                       auto_policy_full_reuse_candidate_ratio         = 0.0;
	double                       auto_policy_avg_full_blocks_per_rowgroup       = 0.0;
	size_t                       auto_policy_empty_windows                      = 0;
	size_t                       auto_policy_crop_covers_full_windows           = 0;
	size_t                       auto_policy_coefficient_pushdown_windows       = 0;
	size_t                       auto_policy_very_small_crop_windows            = 0;
	size_t                       auto_policy_small_window_full_windows          = 0;
	size_t                       auto_policy_large_window_pushdown_windows      = 0;
	size_t                       auto_policy_workset_overhead_full_windows      = 0;
	size_t                       auto_policy_tiny_rowgroups_full_windows        = 0;
	size_t                       auto_policy_touches_most_rowgroups_windows     = 0;
	size_t                       auto_policy_gather_output_full_windows         = 0;
	size_t                       auto_policy_saves_enough_blocks_windows        = 0;
	size_t                       auto_policy_savings_too_small_windows          = 0;
	bool                         outputs_match                                  = true;
	double                       verify_ms                                      = 0.0;
	std::string                  mismatch;
};

PipelineBenchmarkResult benchmark_jpeg_dct_pipeline(const std::filesystem::path&   manifest_path,
                                                    const PipelineBenchmarkConfig& cfg = {});

} // namespace galp::execution

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_SUPPORT_BENCHMARK_PIPELINE_CUH
