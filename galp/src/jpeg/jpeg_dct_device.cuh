#ifndef GALP_JPEG_DCT_DEVICE_CUH
#define GALP_JPEG_DCT_DEVICE_CUH

#include "cuda/memory/gpu_array.cuh"
#include "galp/jpeg_dct.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace galp::jpeg::detail {

struct JpegDctDeviceGatherItem {
	uint32_t rowgroup_index     = 0;
	uint32_t row_in_rowgroup    = 0;
	uint64_t output_block_index = 0;
};

struct JpegDctDeviceRowgroupPlan {
	uint32_t                             rowgroup_index = 0;
	std::vector<JpegDctDeviceGatherItem> items;
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
	struct JpegDctDeviceDecodedRowgroupCache*  cache = nullptr;
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
