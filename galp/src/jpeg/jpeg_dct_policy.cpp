#include "jpeg/jpeg_dct_policy.hpp"
#include "codecs/consts.cuh"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace galp::jpeg::detail {
namespace {

// Keep planner policy and remapping here: changing these host-only decisions
// must not invalidate any CUDA kernel compilation unit.

template <typename Item>
std::vector<Item> remap_to_selected_vectors(const std::vector<Item>&     items,
                                            const std::vector<uint32_t>& selected_vectors,
                                            const unsigned               unpack_n_vectors_cfg,
                                            const char*                  missing_message) {
	const auto        unpack_n_vectors = std::max(1U, unpack_n_vectors_cfg);
	std::vector<Item> remapped;
	remapped.reserve(items.size());
	for (const auto& item : items) {
		const uint32_t source_vector =
		    item.row_in_rowgroup / static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR);
		const uint32_t source_chunk = (source_vector / unpack_n_vectors) * unpack_n_vectors;
		const auto     it           = std::lower_bound(selected_vectors.begin(), selected_vectors.end(), source_chunk);
		if (it == selected_vectors.end() || *it != source_chunk) {
			throw std::runtime_error(missing_message);
		}
		auto       mapped               = item;
		const auto selected_chunk_index = static_cast<uint32_t>(std::distance(selected_vectors.begin(), it));
		const auto chunk_vector_offset  = source_vector - source_chunk;
		const auto row_offset  = item.row_in_rowgroup % static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR);
		mapped.row_in_rowgroup = (selected_chunk_index * unpack_n_vectors + chunk_vector_offset) *
		                             static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR) +
		                         row_offset;
		remapped.push_back(mapped);
	}
	return remapped;
}

} // namespace

JpegDctRuntimePolicyResult choose_jpeg_dct_runtime_policy(const size_t selected_vector_count,
                                                          const size_t full_vector_count,
                                                          const bool   selected_chunks_fit) {
	if (!selected_chunks_fit) {
		return {JpegDctRuntimePolicyDecision::kFullRowgroup, JpegDctRuntimePolicyReason::kTailChunkWouldOverrun};
	}
	if (full_vector_count == 0 || selected_vector_count >= full_vector_count) {
		return {JpegDctRuntimePolicyDecision::kFullRowgroup, JpegDctRuntimePolicyReason::kSelectedCoversMostVectors};
	}
	const double selected_ratio = static_cast<double>(selected_vector_count) / static_cast<double>(full_vector_count);
	const size_t saved_vectors  = full_vector_count - selected_vector_count;
	if (selected_ratio >= kMaxSelectedVectorRatioForPushdown) {
		return {JpegDctRuntimePolicyDecision::kFullRowgroup, JpegDctRuntimePolicyReason::kSelectedCoversMostVectors};
	}
	if (saved_vectors < kMinSavedVectorsForPushdown) {
		return {JpegDctRuntimePolicyDecision::kFullRowgroup, JpegDctRuntimePolicyReason::kSavingsTooSmall};
	}
	return {JpegDctRuntimePolicyDecision::kSelectedVectors, JpegDctRuntimePolicyReason::kCropSavesEnoughVectors};
}

JpegDctSparseStoragePolicyResult choose_jpeg_dct_sparse_storage_policy(const JpegDctSparseStorageCost& cost) {
	// These intentionally model a warm local file, where issuing many tiny
	// preads is most likely to erase byte savings.  They are conservative
	// defaults, not hardware claims; explicit vector-range mode remains
	// available for controlled experiments and tuned deployments.
	constexpr double kStorageBytesPerSecond       = 1.0 * 1024.0 * 1024.0 * 1024.0;
	constexpr double kMemoryBytesPerSecond        = 10.0 * 1024.0 * 1024.0 * 1024.0;
	constexpr double kPreadFixedCostNs            = 2000.0;
	constexpr double kMinimumPredictedSavingRatio = 0.10;
	const auto transfer_ns = [](const size_t bytes, const double bytes_per_second) {
		return static_cast<double>(bytes) * 1.0e9 / bytes_per_second;
	};

	JpegDctSparseStoragePolicyResult result;
	result.full_estimated_ns = transfer_ns(cost.full_storage_bytes, kStorageBytesPerSecond) + kPreadFixedCostNs;
	if (cost.sparse_storage_bytes >= cost.full_storage_bytes || cost.full_storage_bytes == 0U) {
		result.reason = JpegDctSparseStoragePolicyReason::kNoByteSavings;
		return result;
	}
	if (cost.sparse_pread_count == 0U) {
		result.reason = JpegDctSparseStoragePolicyReason::kNoPhysicalReads;
		return result;
	}
	result.sparse_estimated_ns = transfer_ns(cost.sparse_storage_bytes, kStorageBytesPerSecond) +
	                             static_cast<double>(cost.sparse_pread_count) * kPreadFixedCostNs;
	if (cost.requires_logical_materialization) {
		result.sparse_estimated_ns +=
		    transfer_ns(cost.full_storage_bytes + cost.sparse_storage_bytes, kMemoryBytesPerSecond);
	}
	result.use_sparse_read = result.sparse_estimated_ns <=
	                         result.full_estimated_ns * (1.0 - kMinimumPredictedSavingRatio);
	result.reason = result.use_sparse_read ? JpegDctSparseStoragePolicyReason::kPredictedFaster
	                                       : JpegDctSparseStoragePolicyReason::kFragmentationDominates;
	return result;
}

std::vector<uint8_t> normalize_coefficient_selection(const JpegDctCoefficientSelection& selection) {
	std::vector<uint8_t> coefficients;
	if (selection.coefficients.empty()) {
		coefficients.reserve(kJpegDctCoefficientCount);
		for (uint8_t coeff = 0; coeff < kJpegDctCoefficientCount; ++coeff) {
			coefficients.push_back(coeff);
		}
		return coefficients;
	}
	coefficients = selection.coefficients;
	std::array<bool, kJpegDctCoefficientCount> seen {};
	for (const auto coeff : coefficients) {
		if (coeff >= kJpegDctCoefficientCount) {
			throw std::out_of_range("JPEG DCT coefficient selection index is outside [0, 64)");
		}
		if (seen[coeff]) {
			throw std::invalid_argument("JPEG DCT coefficient selection contains duplicates");
		}
		seen[coeff] = true;
	}
	return coefficients;
}

bool selects_all_coefficients(const std::vector<uint8_t>& selected_coefficients) {
	if (selected_coefficients.size() != kJpegDctCoefficientCount) {
		return false;
	}
	for (size_t idx = 0; idx < selected_coefficients.size(); ++idx) {
		if (selected_coefficients[idx] != idx) {
			return false;
		}
	}
	return true;
}

JpegDctCoefficientSelectionShape classify_coefficient_selection(const std::vector<uint8_t>& selected_coefficients) {
	if (selects_all_coefficients(selected_coefficients)) {
		return {JpegDctCoefficientSelectionKind::kAll, kJpegDctCoefficientCount};
	}
	for (size_t idx = 0; idx < selected_coefficients.size(); ++idx) {
		if (selected_coefficients[idx] != idx) {
			return {JpegDctCoefficientSelectionKind::kList, selected_coefficients.size()};
		}
	}
	return {JpegDctCoefficientSelectionKind::kPrefix, selected_coefficients.size()};
}

std::vector<uint32_t> selected_decode_vectors(const std::vector<JpegDctDeviceGatherItem>& items,
                                              const size_t                                rowgroup_n_vecs,
                                              const unsigned                              unpack_n_vectors_cfg) {
	if (rowgroup_n_vecs > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("JPEG DCT rowgroup has too many vectors for selected-vector planning");
	}
	const auto            unpack_n_vectors = std::max(1U, unpack_n_vectors_cfg);
	std::vector<uint32_t> vectors;
	vectors.reserve(items.size());
	for (const auto& item : items) {
		const uint32_t vector_index =
		    item.row_in_rowgroup / static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR);
		if (vector_index >= rowgroup_n_vecs) {
			throw std::out_of_range("JPEG DCT crop row is outside the decoded rowgroup");
		}
		vectors.push_back((vector_index / unpack_n_vectors) * unpack_n_vectors);
	}
	std::sort(vectors.begin(), vectors.end());
	vectors.erase(std::unique(vectors.begin(), vectors.end()), vectors.end());
	return vectors;
}

std::vector<JpegDctDeviceGatherItem> remap_items_to_selected_vectors(const std::vector<JpegDctDeviceGatherItem>& items,
                                                                     const std::vector<uint32_t>& selected_vectors,
                                                                     const unsigned unpack_n_vectors_cfg) {
	return remap_to_selected_vectors(
	    items, selected_vectors, unpack_n_vectors_cfg, "JPEG DCT selected-vector remap missing source vector");
}

std::vector<JpegDctDeviceProjectionItem>
remap_projection_items_to_selected_vectors(const std::vector<JpegDctDeviceProjectionItem>& items,
                                           const std::vector<uint32_t>&                    selected_vectors,
                                           const unsigned                                  unpack_n_vectors_cfg) {
	return remap_to_selected_vectors(items,
	                                 selected_vectors,
	                                 unpack_n_vectors_cfg,
	                                 "JPEG DCT selected-vector projection remap missing source vector");
}

std::vector<JpegDctDeviceFixedTransformItem>
remap_fixed_transform_items_to_selected_vectors(const std::vector<JpegDctDeviceFixedTransformItem>& items,
                                                const std::vector<uint32_t>&                        selected_vectors,
                                                const unsigned unpack_n_vectors_cfg) {
	return remap_to_selected_vectors(items,
	                                 selected_vectors,
	                                 unpack_n_vectors_cfg,
	                                 "JPEG DCT selected-vector fixed transform remap missing source vector");
}

bool selected_decode_chunks_fit(const std::vector<uint32_t>& selected_vectors,
                                const size_t                 rowgroup_n_vecs,
                                const unsigned               unpack_n_vectors_cfg) {
	const auto unpack_n_vectors = static_cast<size_t>(std::max(1U, unpack_n_vectors_cfg));
	return std::all_of(selected_vectors.begin(), selected_vectors.end(), [&](const uint32_t vec) {
		return static_cast<size_t>(vec) + unpack_n_vectors <= rowgroup_n_vecs;
	});
}

size_t selected_decode_vector_count(const std::vector<uint32_t>& selected_vectors,
                                    const size_t                 rowgroup_n_vecs,
                                    const unsigned               unpack_n_vectors_cfg) {
	const auto unpack_n_vectors = static_cast<size_t>(std::max(1U, unpack_n_vectors_cfg));
	return std::min(rowgroup_n_vecs, selected_vectors.size() * unpack_n_vectors);
}

std::vector<uint32_t> expand_selected_decode_chunks(const std::vector<uint32_t>& selected_vectors,
                                                    const size_t                 rowgroup_n_vecs,
                                                    const unsigned               unpack_n_vectors_cfg) {
	const auto unpack_n_vectors = static_cast<size_t>(std::max(1U, unpack_n_vectors_cfg));
	std::vector<uint32_t> physical_vectors;
	physical_vectors.reserve(selected_vectors.size() * unpack_n_vectors);
	for (const uint32_t chunk_base : selected_vectors) {
		for (size_t lane = 0; lane < unpack_n_vectors; ++lane) {
			const size_t vector = static_cast<size_t>(chunk_base) + lane;
			if (vector < rowgroup_n_vecs) {
				physical_vectors.push_back(static_cast<uint32_t>(vector));
			}
		}
	}
	return physical_vectors;
}

std::vector<uint32_t> build_logical_to_compact_vector_remap(const std::vector<uint32_t>& selected_vectors,
	                                                         const size_t logical_rowgroup_n_vecs,
	                                                         const unsigned unpack_n_vectors_cfg) {
	if (logical_rowgroup_n_vecs == 0U || logical_rowgroup_n_vecs > std::numeric_limits<uint32_t>::max()) {
		throw std::invalid_argument("JPEG DCT logical vector remap requires a non-empty uint32-addressable rowgroup");
	}
	const uint32_t missing = std::numeric_limits<uint32_t>::max();
	const size_t width = std::max<size_t>(1U, unpack_n_vectors_cfg);
	if (selected_vectors.size() > static_cast<size_t>(missing) / width) {
		throw std::overflow_error("JPEG DCT compact vector remap exceeds uint32 range");
	}
	std::vector<uint32_t> remap(logical_rowgroup_n_vecs, missing);
	for (size_t selected_index = 0; selected_index < selected_vectors.size(); ++selected_index) {
		const size_t logical_base = selected_vectors[selected_index];
		for (size_t lane = 0; lane < width && logical_base + lane < logical_rowgroup_n_vecs; ++lane) {
			const size_t logical_vector = logical_base + lane;
			if (remap[logical_vector] != missing) {
				throw std::invalid_argument("JPEG DCT selected vector chunks overlap");
			}
			remap[logical_vector] = static_cast<uint32_t>(selected_index * width + lane);
		}
	}
	return remap;
}

unsigned constrain_jpeg_dct_batch_unpack_n_vectors(const unsigned                   preferred_unpack_n_vectors,
                                                   const JpegDctDeviceRowgroupPlan& rowgroup_plan) {
	const auto unpack_n_vectors = std::max(1U, preferred_unpack_n_vectors);
	if (unpack_n_vectors == 1U) {
		return 1U;
	}
	if (!rowgroup_plan.has_vector_plan ||
	    (rowgroup_plan.runtime_policy.decision == JpegDctRuntimePolicyDecision::kFullRowgroup &&
	     rowgroup_plan.full_vector_count % unpack_n_vectors != 0U)) {
		return 1U;
	}
	return unpack_n_vectors;
}

JpegDctDeviceRowgroupPrefetchPlan
plan_jpeg_dct_rowgroup_prefetch_from_hits(const JpegDctDeviceShardPlan&              shard,
                                          const std::vector<bool>&                   cache_hit_by_position,
                                          const JpegDctDeviceRowgroupPrefetchConfig& config,
                                          const size_t                               effective_decode_batch_rowgroups,
                                          const size_t                               decoded_cache_capacity_bytes) {
	JpegDctDeviceRowgroupPrefetchPlan plan;
	plan.use_prefetch_for_position.assign(shard.rowgroups.size(), false);
	plan.initial_cache_hit_for_position.assign(shard.rowgroups.size(), false);
	plan.skipped_repeated_for_position.assign(shard.rowgroups.size(), false);
	if (cache_hit_by_position.size() != shard.rowgroups.size()) {
		throw std::invalid_argument("JPEG DCT prefetch hit map size does not match shard rowgroups");
	}

	std::vector<size_t>                  candidate_positions;
	std::unordered_map<uint32_t, size_t> earlier_full_decode_miss_ordinal;
	size_t                               miss_ordinal = 0;
	for (size_t rowgroup_pos = 0; rowgroup_pos < shard.rowgroups.size(); ++rowgroup_pos) {
		const auto& rowgroup_plan = shard.rowgroups[rowgroup_pos];
		if (cache_hit_by_position[rowgroup_pos]) {
			plan.initial_cache_hit_for_position[rowgroup_pos] = true;
			++plan.initial_cache_hit_rowgroup_count;
			continue;
		}
		const bool full_rowgroup = rowgroup_plan.runtime_policy.decision == JpegDctRuntimePolicyDecision::kFullRowgroup;
		if (!full_rowgroup) {
			++plan.selected_vector_miss_rowgroup_count;
		}
		++plan.candidate_rowgroup_count;
		const auto earlier = full_rowgroup ? earlier_full_decode_miss_ordinal.find(rowgroup_plan.rowgroup_index)
		                                   : earlier_full_decode_miss_ordinal.end();
		const bool reuse   = full_rowgroup && decoded_cache_capacity_bytes != 0 &&
		                   earlier != earlier_full_decode_miss_ordinal.end() &&
		                   miss_ordinal >= earlier->second + effective_decode_batch_rowgroups;
		if (reuse) {
			plan.skipped_repeated_for_position[rowgroup_pos] = true;
			++plan.skipped_repeated_rowgroup_count;
		} else {
			candidate_positions.push_back(rowgroup_pos);
			plan.rowgroup_indices.push_back(rowgroup_plan.rowgroup_index);
			plan.selected_vectors.push_back(rowgroup_plan.sparse_storage_read ? rowgroup_plan.selected_vectors
			                                                               : std::vector<uint32_t> {});
		}
		const auto dense_bytes = rowgroup_plan.full_vector_count * galp::codec::consts::VALUES_PER_VECTOR *
		                         kJpegDctCoefficientCount * sizeof(int16_t);
		if (decoded_cache_capacity_bytes != 0 && full_rowgroup && dense_bytes <= decoded_cache_capacity_bytes) {
			earlier_full_decode_miss_ordinal.emplace(rowgroup_plan.rowgroup_index, miss_ordinal);
		}
		++miss_ordinal;
	}
	if (!config.enabled) {
		plan.disabled_by_config = !shard.rowgroups.empty();
		plan.rowgroup_indices.clear();
		plan.selected_vectors.clear();
		return plan;
	}
	if (effective_decode_batch_rowgroups == 0) {
		plan.disabled_by_small_batch_count = plan.candidate_rowgroup_count != 0;
		plan.rowgroup_indices.clear();
		plan.selected_vectors.clear();
		return plan;
	}
	if (plan.candidate_rowgroup_count == 0) {
		plan.disabled_by_all_hits =
		    !shard.rowgroups.empty() && plan.initial_cache_hit_rowgroup_count == shard.rowgroups.size();
		plan.disabled_by_selected_vector_miss =
		    !plan.disabled_by_all_hits && plan.selected_vector_miss_rowgroup_count != 0;
		return plan;
	}
	const size_t min_batches = std::max<size_t>(1, config.min_decode_batches);
	const size_t candidate_batches =
	    (plan.rowgroup_indices.size() + effective_decode_batch_rowgroups - 1U) / effective_decode_batch_rowgroups;
	if (candidate_batches < min_batches) {
		plan.disabled_by_small_batch_count = true;
		plan.rowgroup_indices.clear();
		plan.selected_vectors.clear();
		return plan;
	}
	plan.enabled = true;
	for (const auto rowgroup_pos : candidate_positions) {
		plan.use_prefetch_for_position[rowgroup_pos] = true;
	}
	return plan;
}

} // namespace galp::jpeg::detail

namespace galp::jpeg {

bool parse_jpeg_dct_coefficient_selection(const std::string_view spec, JpegDctCoefficientSelection& selection) {
	JpegDctCoefficientSelection parsed;
	if (spec == "all") {
		selection = std::move(parsed);
		return true;
	}

	const auto parse_coeff = [](const std::string_view token, uint8_t& out) -> bool {
		if (token.empty()) {
			return false;
		}
		try {
			size_t     parsed_chars = 0;
			const auto value        = std::stoull(std::string(token), &parsed_chars);
			if (parsed_chars != token.size() || value >= detail::kJpegDctCoefficientCount) {
				return false;
			}
			out = static_cast<uint8_t>(value);
			return true;
		} catch (...) { return false; }
	};

	constexpr std::string_view first_prefix = "first:";
	constexpr std::string_view list_prefix  = "list:";
	if (spec.substr(0, first_prefix.size()) == first_prefix) {
		size_t     count    = 0;
		const auto count_sv = spec.substr(first_prefix.size());
		try {
			size_t parsed_chars = 0;
			count               = std::stoull(std::string(count_sv), &parsed_chars);
			if (parsed_chars != count_sv.size()) {
				return false;
			}
		} catch (...) { return false; }
		if (count == 0 || count > detail::kJpegDctCoefficientCount) {
			return false;
		}
		parsed.coefficients.reserve(count);
		for (size_t coeff = 0; coeff < count; ++coeff) {
			parsed.coefficients.push_back(static_cast<uint8_t>(coeff));
		}
		selection = std::move(parsed);
		return true;
	}

	if (spec.substr(0, list_prefix.size()) == list_prefix) {
		auto list = spec.substr(list_prefix.size());
		if (list.empty()) {
			return false;
		}
		while (!list.empty()) {
			const auto comma = list.find(',');
			const auto token = comma == std::string_view::npos ? list : list.substr(0, comma);
			uint8_t    coeff = 0;
			if (!parse_coeff(token, coeff) ||
			    std::find(parsed.coefficients.begin(), parsed.coefficients.end(), coeff) != parsed.coefficients.end()) {
				return false;
			}
			parsed.coefficients.push_back(coeff);
			if (comma == std::string_view::npos) {
				break;
			}
			list = list.substr(comma + 1);
		}
		selection = std::move(parsed);
		return true;
	}
	return false;
}

} // namespace galp::jpeg
