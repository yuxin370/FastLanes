#ifndef GALP_JPEG_DCT_DEVICE_CUH
#define GALP_JPEG_DCT_DEVICE_CUH

#include "codecs/consts.cuh"
#include "cuda/memory/gpu_array.cuh"
#include "galp/jpeg_dct.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace galp::execution {
struct ExecutionConfig;
struct Rowgroup;
} // namespace galp::execution

namespace galp::runtime {
struct ExecutionWorkset;
} // namespace galp::runtime

namespace galp::jpeg::detail {

constexpr double   kMaxSelectedVectorRatioForPushdown = 0.75;
constexpr size_t   kMinSavedVectorsForPushdown        = 4;
constexpr unsigned kJpegDctDeviceUnpackNVectors       = 1;
constexpr size_t   kJpegDctCoefficientCount           = 64;

__host__ __device__ inline size_t
selected_dct_binding_offset(const size_t source_index, const size_t coeff_slot, const size_t coefficients_per_block) {
	return source_index * coefficients_per_block + coeff_slot;
}

__host__ __device__ inline size_t selected_dct_output_offset(const size_t output_block_index,
                                                             const size_t coeff_slot,
                                                             const size_t coefficients_per_block) {
	return output_block_index * coefficients_per_block + coeff_slot;
}

struct JpegDctDeviceScratch;
struct JpegDctCoefficientSelectionShape;

struct JpegDctDeviceScratchDeleter {
	void operator()(JpegDctDeviceScratch* scratch) const noexcept;
};

using JpegDctDeviceScratchPtr = std::unique_ptr<JpegDctDeviceScratch, JpegDctDeviceScratchDeleter>;

JpegDctDeviceScratchPtr make_jpeg_dct_device_scratch();

void append_jpeg_rowgroup_columns(galp::runtime::ExecutionWorkset&        workset,
                                  const galp::execution::Rowgroup&        rowgroup,
                                  const galp::execution::ExecutionConfig& cfg,
                                  size_t                                  expr_index_base,
                                  const std::vector<uint8_t>&             selected_coefficients,
                                  const std::vector<uint32_t>*            selected_vectors = nullptr);

void append_jpeg_rowgroup_columns(galp::runtime::ExecutionWorkset&        workset,
                                  const galp::execution::Rowgroup&        rowgroup,
                                  const galp::execution::ExecutionConfig& cfg,
                                  size_t                                  expr_index_base,
                                  const std::vector<uint8_t>&             selected_coefficients,
                                  const JpegDctCoefficientSelectionShape& selection_shape,
                                  const std::vector<uint32_t>*            selected_vectors = nullptr);

enum class JpegDctRuntimePolicyDecision {
	kSelectedVectors,
	kFullRowgroup,
};

enum class JpegDctRuntimePolicyReason {
	kCropSavesEnoughVectors,
	kTailChunkWouldOverrun,
	kSelectedCoversMostVectors,
	kSavingsTooSmall,
};

struct JpegDctRuntimePolicyResult {
	JpegDctRuntimePolicyDecision decision = JpegDctRuntimePolicyDecision::kSelectedVectors;
	JpegDctRuntimePolicyReason   reason   = JpegDctRuntimePolicyReason::kCropSavesEnoughVectors;
};

inline JpegDctRuntimePolicyResult choose_jpeg_dct_runtime_policy(const size_t selected_vector_count,
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

inline std::vector<uint8_t> normalize_coefficient_selection(const JpegDctCoefficientSelection& selection) {
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

inline bool selects_all_coefficients(const std::vector<uint8_t>& selected_coefficients) {
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

enum class JpegDctCoefficientSelectionKind {
	kAll,
	kPrefix,
	kList,
};

struct JpegDctCoefficientSelectionShape {
	JpegDctCoefficientSelectionKind kind  = JpegDctCoefficientSelectionKind::kAll;
	size_t                          count = kJpegDctCoefficientCount;

	[[nodiscard]] bool is_contiguous_prefix() const noexcept {
		return kind == JpegDctCoefficientSelectionKind::kAll || kind == JpegDctCoefficientSelectionKind::kPrefix;
	}
};

inline JpegDctCoefficientSelectionShape
classify_coefficient_selection(const std::vector<uint8_t>& selected_coefficients) {
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

struct JpegDctDeviceGatherItem {
	uint32_t rowgroup_index     = 0;
	uint32_t row_in_rowgroup    = 0;
	uint64_t output_block_index = 0;
};

inline std::vector<uint32_t> selected_decode_vectors(const std::vector<JpegDctDeviceGatherItem>& items,
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

inline std::vector<JpegDctDeviceGatherItem>
remap_items_to_selected_vectors(const std::vector<JpegDctDeviceGatherItem>& items,
                                const std::vector<uint32_t>&                selected_vectors,
                                const unsigned                              unpack_n_vectors_cfg) {
	const auto                           unpack_n_vectors = std::max(1U, unpack_n_vectors_cfg);
	std::vector<JpegDctDeviceGatherItem> remapped;
	remapped.reserve(items.size());
	for (const auto& item : items) {
		const uint32_t source_vector =
		    item.row_in_rowgroup / static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR);
		const uint32_t source_chunk = (source_vector / unpack_n_vectors) * unpack_n_vectors;
		const auto     it           = std::lower_bound(selected_vectors.begin(), selected_vectors.end(), source_chunk);
		if (it == selected_vectors.end() || *it != source_chunk) {
			throw std::runtime_error("JPEG DCT selected-vector remap missing source vector");
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

inline bool selected_decode_chunks_fit(const std::vector<uint32_t>& selected_vectors,
                                       const size_t                 rowgroup_n_vecs,
                                       const unsigned               unpack_n_vectors_cfg) {
	const auto unpack_n_vectors = static_cast<size_t>(std::max(1U, unpack_n_vectors_cfg));
	return std::all_of(selected_vectors.begin(), selected_vectors.end(), [&](const uint32_t vec) {
		return static_cast<size_t>(vec) + unpack_n_vectors <= rowgroup_n_vecs;
	});
}

inline size_t selected_decode_vector_count(const std::vector<uint32_t>& selected_vectors,
                                           const size_t                 rowgroup_n_vecs,
                                           const unsigned               unpack_n_vectors_cfg) {
	const auto unpack_n_vectors = static_cast<size_t>(std::max(1U, unpack_n_vectors_cfg));
	return std::min(rowgroup_n_vecs, selected_vectors.size() * unpack_n_vectors);
}

struct JpegDctDeviceRowgroupPlan {
	uint32_t                             rowgroup_index = 0;
	std::vector<JpegDctDeviceGatherItem> items;
	std::vector<uint32_t>                selected_vectors;
	std::vector<JpegDctDeviceGatherItem> selected_gather_items;
	size_t                               selected_vector_count = 0;
	size_t                               full_vector_count     = 0;
	bool                                 selected_chunks_fit   = true;
	bool                                 has_vector_plan       = false;
	JpegDctRuntimePolicyResult           runtime_policy {};
};

struct JpegDctDeviceShardPlan {
	uint32_t                               shard_id = 0;
	std::filesystem::path                  fls_path;
	std::vector<JpegDctDeviceRowgroupPlan> rowgroups;
};

struct JpegDctDeviceRowgroupPrefetchConfig {
	bool   enabled            = true;
	size_t depth              = kDefaultJpegDctDeviceRowgroupPrefetchDepth;
	size_t workers            = kDefaultJpegDctDeviceRowgroupPrefetchWorkers;
	size_t min_decode_batches = kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches;
};

struct JpegDctDeviceBatchPlan {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::vector<JpegDctDeviceShardPlan>        shards;
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	struct JpegDctDeviceDecodedRowgroupCache*  cache                           = nullptr;
	JpegDctDeviceScratch*                      scratch                         = nullptr;
	size_t                                     planned_selected_vector_count   = 0;
	size_t                                     estimated_selected_vector_count = 0;
	size_t                                     full_vector_count               = 0;
	size_t                                     planned_saved_vector_count      = 0;
	size_t                                     estimated_saved_vector_count    = 0;
	std::vector<uint8_t>                       selected_coefficients;
	JpegDctCoefficientSelectionShape           coefficient_selection_shape {};
	size_t                                     coefficients_per_block          = kJpegDctCoefficientCount;
	double                                     planned_selected_vector_ratio   = 0.0;
	double                                     estimated_selected_vector_ratio = 0.0;
	double                                     planning_ms                     = 0.0;
	size_t                                     decode_batch_rowgroups          = kDefaultJpegDctDecodeBatchRowgroups;
	JpegDctDeviceRowgroupPrefetchConfig        rowgroup_prefetch {};
};

struct JpegDctDeviceDecodedRowgroupCacheKey {
	uint32_t shard_id       = 0;
	uint32_t rowgroup_index = 0;

	bool operator==(const JpegDctDeviceDecodedRowgroupCacheKey& other) const noexcept {
		return shard_id == other.shard_id && rowgroup_index == other.rowgroup_index;
	}
};

struct JpegDctDeviceDecodedRowgroupCacheKeyHash {
	size_t operator()(const JpegDctDeviceDecodedRowgroupCacheKey& key) const noexcept {
		uint64_t h = static_cast<uint64_t>(key.shard_id) * 0x9e3779b185ebca87ULL;
		h ^= static_cast<uint64_t>(key.rowgroup_index) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
		return static_cast<size_t>(h);
	}
};

struct JpegDctDeviceDecodedRowgroupCacheEntry {
	uint32_t                         rows        = 0;
	size_t                           bytes       = 0;
	uint64_t                         last_access = 0;
	std::optional<GPUArray<int16_t>> blocks;
};

struct JpegDctDeviceDecodedRowgroupCache {
	void                 set_capacity(size_t bytes);
	void                 clear();
	void                 insert_ready_entry(const JpegDctDeviceDecodedRowgroupCacheKey&             key,
	                                        std::unique_ptr<JpegDctDeviceDecodedRowgroupCacheEntry> entry,
	                                        JpegDctDeviceCacheStats&                                batch_cache_stats);
	[[nodiscard]] size_t capacity_bytes() const noexcept;
	[[nodiscard]] size_t resident_bytes() const noexcept;
	[[nodiscard]] size_t resident_rowgroups() const noexcept;

	size_t   capacity = 0;
	size_t   resident = 0;
	uint64_t clock    = 0;
	std::unordered_map<JpegDctDeviceDecodedRowgroupCacheKey,
	                   std::unique_ptr<JpegDctDeviceDecodedRowgroupCacheEntry>,
	                   JpegDctDeviceDecodedRowgroupCacheKeyHash>
	    entries;
};

struct JpegDctDeviceRowgroupPrefetchPlan {
	bool                enabled                             = false;
	bool                disabled_by_config                  = false;
	bool                disabled_by_all_hits                = false;
	bool                disabled_by_small_batch_count       = false;
	bool                disabled_by_selected_vector_miss    = false;
	size_t              initial_cache_hit_rowgroup_count    = 0;
	size_t              candidate_rowgroup_count            = 0;
	size_t              selected_vector_miss_rowgroup_count = 0;
	size_t              skipped_repeated_rowgroup_count     = 0;
	std::vector<size_t> rowgroup_indices;
	std::vector<bool>   use_prefetch_for_position;
	std::vector<bool>   initial_cache_hit_for_position;
	std::vector<bool>   skipped_repeated_for_position;
};

inline bool has_decoded_cache_entry(const JpegDctDeviceDecodedRowgroupCache*    cache,
                                    const JpegDctDeviceDecodedRowgroupCacheKey& key) {
	if (cache == nullptr || cache->capacity == 0) {
		return false;
	}
	const auto it = cache->entries.find(key);
	return it != cache->entries.end() && it->second->blocks.has_value();
}

inline JpegDctDeviceRowgroupPrefetchPlan
plan_jpeg_dct_rowgroup_prefetch_from_hits(const JpegDctDeviceShardPlan&              shard,
                                          const std::vector<bool>&                   cache_hit_by_position,
                                          const JpegDctDeviceRowgroupPrefetchConfig& config,
                                          const size_t                               effective_decode_batch_rowgroups,
                                          const size_t                               decoded_cache_capacity_bytes = 0) {
	JpegDctDeviceRowgroupPrefetchPlan plan;
	plan.use_prefetch_for_position.assign(shard.rowgroups.size(), false);
	plan.initial_cache_hit_for_position.assign(shard.rowgroups.size(), false);
	plan.skipped_repeated_for_position.assign(shard.rowgroups.size(), false);
	if (cache_hit_by_position.size() != shard.rowgroups.size()) {
		throw std::invalid_argument("JPEG DCT prefetch hit map size does not match shard rowgroups");
	}

	std::vector<size_t> candidate_positions;
	candidate_positions.reserve(shard.rowgroups.size());
	plan.rowgroup_indices.reserve(shard.rowgroups.size());
	std::unordered_map<uint32_t, size_t> earlier_full_decode_miss_ordinal;
	size_t                               miss_ordinal = 0;
	for (size_t rowgroup_pos = 0; rowgroup_pos < shard.rowgroups.size(); ++rowgroup_pos) {
		const auto& rowgroup_plan = shard.rowgroups[rowgroup_pos];
		if (cache_hit_by_position[rowgroup_pos]) {
			plan.initial_cache_hit_for_position[rowgroup_pos] = true;
			++plan.initial_cache_hit_rowgroup_count;
			continue;
		}
		// Current prefetch materializes whole rowgroups and only has dense-cache reuse
		// semantics. Keep selected-vector misses synchronous until the sparse path has an
		// explicit prefetch/cache contract instead of mixing policies in this planner.
		if (rowgroup_plan.runtime_policy.decision != JpegDctRuntimePolicyDecision::kFullRowgroup) {
			++plan.selected_vector_miss_rowgroup_count;
			continue;
		}
		++plan.candidate_rowgroup_count;
		const auto earlier_full_it = earlier_full_decode_miss_ordinal.find(rowgroup_plan.rowgroup_index);
		const bool can_reuse_earlier_full_decode =
		    decoded_cache_capacity_bytes != 0 && earlier_full_it != earlier_full_decode_miss_ordinal.end() &&
		    miss_ordinal >= earlier_full_it->second + effective_decode_batch_rowgroups;
		if (!can_reuse_earlier_full_decode) {
			candidate_positions.push_back(rowgroup_pos);
			plan.rowgroup_indices.push_back(rowgroup_plan.rowgroup_index);
		} else {
			plan.skipped_repeated_for_position[rowgroup_pos] = true;
			++plan.skipped_repeated_rowgroup_count;
		}
		const auto dense_rowgroup_bytes =
		    rowgroup_plan.full_vector_count * galp::codec::consts::VALUES_PER_VECTOR * 64U * sizeof(int16_t);
		if (decoded_cache_capacity_bytes != 0 &&
		    rowgroup_plan.runtime_policy.decision == JpegDctRuntimePolicyDecision::kFullRowgroup &&
		    dense_rowgroup_bytes <= decoded_cache_capacity_bytes) {
			earlier_full_decode_miss_ordinal.emplace(rowgroup_plan.rowgroup_index, miss_ordinal);
		}
		++miss_ordinal;
	}
	if (!config.enabled) {
		plan.disabled_by_config = !shard.rowgroups.empty();
		plan.rowgroup_indices.clear();
		return plan;
	}
	if (effective_decode_batch_rowgroups == 0) {
		plan.disabled_by_small_batch_count = plan.candidate_rowgroup_count != 0;
		plan.rowgroup_indices.clear();
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
	// Prefetch only when miss candidates span enough decode batches for CPU IO/materialization
	// to run behind already-launched GPU work. This is a structural overlap guard, not a dataset threshold.
	if (candidate_batches < min_batches) {
		plan.disabled_by_small_batch_count = true;
		plan.rowgroup_indices.clear();
		return plan;
	}

	plan.enabled = true;
	for (const size_t rowgroup_pos : candidate_positions) {
		plan.use_prefetch_for_position[rowgroup_pos] = true;
	}
	return plan;
}

inline JpegDctDeviceRowgroupPrefetchPlan
plan_jpeg_dct_rowgroup_prefetch(const JpegDctDeviceShardPlan&              shard,
                                const JpegDctDeviceDecodedRowgroupCache*   cache,
                                const JpegDctDeviceRowgroupPrefetchConfig& config,
                                const size_t                               effective_decode_batch_rowgroups) {
	std::vector<bool> cache_hit_by_position(shard.rowgroups.size(), false);
	for (size_t rowgroup_pos = 0; rowgroup_pos < shard.rowgroups.size(); ++rowgroup_pos) {
		const auto& rowgroup_plan = shard.rowgroups[rowgroup_pos];
		const auto  key           = JpegDctDeviceDecodedRowgroupCacheKey {shard.shard_id, rowgroup_plan.rowgroup_index};
		cache_hit_by_position[rowgroup_pos] = has_decoded_cache_entry(cache, key);
	}
	return plan_jpeg_dct_rowgroup_prefetch_from_hits(shard,
	                                                 cache_hit_by_position,
	                                                 config,
	                                                 effective_decode_batch_rowgroups,
	                                                 cache == nullptr ? 0U : cache->capacity);
}

JpegDctDeviceBatch execute_jpeg_dct_device_batch_plan(JpegDctDeviceBatchPlan plan);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_DEVICE_CUH
