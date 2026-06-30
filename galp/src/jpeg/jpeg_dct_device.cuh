#ifndef GALP_JPEG_DCT_DEVICE_CUH
#define GALP_JPEG_DCT_DEVICE_CUH

#include "codecs/consts.cuh"
#include "cuda/memory/gpu_array.cuh"
#include "galp/jpeg_dct.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace galp::jpeg::detail {

constexpr double kMaxSelectedVectorRatioForPushdown = 0.75;
constexpr size_t kMinSavedVectorsForPushdown        = 4;
constexpr unsigned kJpegDctDeviceUnpackNVectors     = 1;

struct JpegDctDeviceScratch;

struct JpegDctDeviceScratchDeleter {
	void operator()(JpegDctDeviceScratch* scratch) const noexcept;
};

using JpegDctDeviceScratchPtr = std::unique_ptr<JpegDctDeviceScratch, JpegDctDeviceScratchDeleter>;

JpegDctDeviceScratchPtr make_jpeg_dct_device_scratch();

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
		return {JpegDctRuntimePolicyDecision::kFullRowgroup,
		        JpegDctRuntimePolicyReason::kTailChunkWouldOverrun};
	}
	if (full_vector_count == 0 || selected_vector_count >= full_vector_count) {
		return {JpegDctRuntimePolicyDecision::kFullRowgroup,
		        JpegDctRuntimePolicyReason::kSelectedCoversMostVectors};
	}
	const double selected_ratio =
	    static_cast<double>(selected_vector_count) / static_cast<double>(full_vector_count);
	const size_t saved_vectors = full_vector_count - selected_vector_count;
	if (selected_ratio >= kMaxSelectedVectorRatioForPushdown) {
		return {JpegDctRuntimePolicyDecision::kFullRowgroup,
		        JpegDctRuntimePolicyReason::kSelectedCoversMostVectors};
	}
	if (saved_vectors < kMinSavedVectorsForPushdown) {
		return {JpegDctRuntimePolicyDecision::kFullRowgroup, JpegDctRuntimePolicyReason::kSavingsTooSmall};
	}
	return {JpegDctRuntimePolicyDecision::kSelectedVectors, JpegDctRuntimePolicyReason::kCropSavesEnoughVectors};
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
		const auto row_offset =
		    item.row_in_rowgroup % static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR);
		mapped.row_in_rowgroup = (selected_chunk_index * unpack_n_vectors + chunk_vector_offset) *
		                             static_cast<uint32_t>(galp::codec::consts::VALUES_PER_VECTOR) +
		                         row_offset;
		remapped.push_back(mapped);
	}
	return remapped;
}

inline bool selected_decode_chunks_fit(const std::vector<uint32_t>&            selected_vectors,
                                       const size_t                            rowgroup_n_vecs,
                                       const unsigned                          unpack_n_vectors_cfg) {
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

struct JpegDctDeviceBatchPlan {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::vector<JpegDctDeviceShardPlan>        shards;
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	struct JpegDctDeviceDecodedRowgroupCache*  cache                         = nullptr;
	JpegDctDeviceScratch*                      scratch                       = nullptr;
	size_t                                     planned_selected_vector_count   = 0;
	size_t                                     estimated_selected_vector_count = 0;
	size_t                                     full_vector_count               = 0;
	size_t                                     planned_saved_vector_count      = 0;
	size_t                                     estimated_saved_vector_count    = 0;
	double                                     planned_selected_vector_ratio   = 0.0;
	double                                     estimated_selected_vector_ratio = 0.0;
	double                                     planning_ms                     = 0.0;
	size_t                                     decode_batch_rowgroups          = kDefaultJpegDctDecodeBatchRowgroups;
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

JpegDctDeviceBatch execute_jpeg_dct_device_batch_plan(JpegDctDeviceBatchPlan plan);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_DEVICE_CUH
