#pragma once

#include "galp/jpeg_dct_storage.hpp"
#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace galp::jpeg::detail {

struct JpegDctMaterializeBlockRef {
	uint32_t      semantic_slot_id = 0;
	uint32_t      block_x          = 0;
	uint32_t      block_y          = 0;
	JpegDctRowRef row;
};

// Owns the FastLanes CPU-reader state used for storage audits and random
// access. It does not own JPEG metadata and has no dependency on CUDA plans.
class JpegDctShardCpuReader {
public:
	JpegDctShardCpuReader();
	~JpegDctShardCpuReader();

	JpegDctShardCpuReader(const JpegDctShardCpuReader&)            = delete;
	JpegDctShardCpuReader& operator=(const JpegDctShardCpuReader&) = delete;

	uint64_t          RowgroupStorageBytes(uint32_t                     shard_id,
	                                       const std::filesystem::path& fls_path,
	                                       const std::vector<uint32_t>& rowgroup_indices) const;
	JpegDctBlockGroup ReadBlockGroup(const std::filesystem::path& fls_path, const JpegDctBlockGroupIndex& group) const;
	MaterializedJpegDctImage MaterializeImage(const std::filesystem::path&                   fls_path,
	                                          uint32_t                                       global_image_index,
	                                          const std::vector<JpegDctMaterializeBlockRef>& blocks) const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

// Sequential, descriptor-reusing reader for offline crop-profile generation.
// One instance owns one source FLS TableReader and appends compact selected
// vectors without rebuilding per-block row references.
class JpegDctSelectedVectorProfileReader {
public:
	explicit JpegDctSelectedVectorProfileReader(const std::filesystem::path& fls_path);
	~JpegDctSelectedVectorProfileReader();

	JpegDctSelectedVectorProfileReader(const JpegDctSelectedVectorProfileReader&)            = delete;
	JpegDctSelectedVectorProfileReader& operator=(const JpegDctSelectedVectorProfileReader&) = delete;

	void AppendRowgroupSelectedVectors(uint32_t                              rowgroup_index,
	                                   std::span<const uint32_t>             selected_vectors,
	                                   std::array<std::vector<int16_t>, 64>& columns) const;
	uint32_t AppendFullRowgroup(uint32_t                              rowgroup_index,
	                            std::array<std::vector<int16_t>, 64>& columns) const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace galp::jpeg::detail
