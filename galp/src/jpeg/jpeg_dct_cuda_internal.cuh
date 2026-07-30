#ifndef GALP_JPEG_DCT_CUDA_INTERNAL_CUH
#define GALP_JPEG_DCT_CUDA_INTERNAL_CUH

#include "cuda/memory/gpu_array.cuh"
#include "jpeg/jpeg_dct_device_runtime.hpp"
#include "jpeg/jpeg_dct_kernel_types.cuh"
#include "jpeg/jpeg_dct_policy.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

struct JpegDctDeviceResolvedProjection {
	std::vector<JpegDctDeviceProjectionItem> items;
	std::vector<uint8_t>                     active_physical_coefficients;
};

size_t resolve_physical_coefficient_column(const galp::execution::Rowgroup& rowgroup, size_t logical_coeff_idx);
JpegDctDeviceResolvedProjection
resolve_projection_physical_columns(const galp::execution::Rowgroup&                rowgroup,
                                    const std::vector<JpegDctDeviceProjectionItem>& projection_items);

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

bool                              has_decoded_cache_entry(const JpegDctDeviceDecodedRowgroupCache*    cache,
                                                          const JpegDctDeviceDecodedRowgroupCacheKey& key);
JpegDctDeviceRowgroupPrefetchPlan plan_jpeg_dct_rowgroup_prefetch(const JpegDctDeviceShardPlan&              shard,
                                                                  const JpegDctDeviceDecodedRowgroupCache*   cache,
                                                                  const JpegDctDeviceRowgroupPrefetchConfig& config,
                                                                  size_t effective_decode_batch_rowgroups);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_CUDA_INTERNAL_CUH
