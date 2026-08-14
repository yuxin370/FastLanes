#ifndef GALP_SPARSE_READ_RECIPE_HPP
#define GALP_SPARSE_READ_RECIPE_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace galp::format {

struct SparseReadRange {
	size_t offset = 0U;
	size_t size   = 0U;
};

// Amplification caps are represented as integer millionths so a contract such
// as 1.02 has an exact, platform-independent byte budget.  Floating-point
// rounding at a cap boundary must never make one extra gap eligible.
inline constexpr uint32_t kSparseReadAmplificationScale = 1'000'000U;

struct SparseReadBoundedCoalesceOptions {
	uint32_t whole_run_amplification_ppm   = kSparseReadAmplificationScale;
	// Zero inherits the whole-run cap.  Non-zero values independently bound
	// each shard/rowgroup while the whole-run budget remains authoritative.
	uint32_t per_shard_amplification_ppm   = 0U;
	uint32_t per_rowgroup_amplification_ppm = 0U;
	// Zero means the containing rowgroup size.  Runs never cross rowgroups.
	size_t   max_physical_run_bytes        = 0U;
};

struct SparseReadBoundedRowgroupInput {
	uint32_t                    shard_id = 0U;
	size_t                      rowgroup_id = 0U;
	size_t                      full_storage_bytes = 0U;
	std::vector<SparseReadRange> exact_ranges;
};

struct SparseReadBoundedRowgroupResult {
	uint32_t                    shard_id = 0U;
	size_t                      rowgroup_id = 0U;
	size_t                      full_storage_bytes = 0U;
	size_t                      exact_storage_bytes = 0U;
	size_t                      physical_storage_bytes = 0U;
	size_t                      merged_gap_bytes = 0U;
	std::vector<SparseReadRange> exact_ranges;
	std::vector<SparseReadRange> physical_ranges;
	// Every selected internal gap is cleared after physical reads and before
	// the immutable index/shared prefix is restored.
	std::vector<SparseReadRange> merged_holes;
};

struct SparseReadBoundedCoalesceResult {
	std::vector<SparseReadBoundedRowgroupResult> rowgroups;
	size_t exact_storage_bytes    = 0U;
	size_t physical_storage_bytes = 0U;
	size_t merged_gap_bytes       = 0U;
	size_t full_storage_bytes     = 0U;
	size_t exact_extent_count     = 0U;
	size_t physical_run_count     = 0U;
	size_t selected_gap_count     = 0U;
	size_t max_run_rejected_gap_count = 0U;
	size_t budget_rejected_gap_count  = 0U;
};

// Deterministically select internal gaps in the stable order
// (gap_size, shard_id, rowgroup_id, boundary_id).  Exact inputs are first
// canonicalized (zero ranges removed; overlap/adjacency coalesced), checked for
// overflow/OOB, and then governed by simultaneous whole-run, per-shard, and
// per-rowgroup integer byte budgets.
SparseReadBoundedCoalesceResult coalesce_sparse_read_ranges_bounded(
	const std::vector<SparseReadBoundedRowgroupInput>& inputs,
	const SparseReadBoundedCoalesceOptions&            options);

struct SparseReadRecipeSelection {
	size_t                rowgroup_index = 0U;
	std::vector<uint32_t> selected_vectors;
};

// Offline-only details returned by the deterministic recipe builder.  The
// production reader does not retain this vector; the plan audit consumes it
// one shard at a time to replay bounded-gap budgets without touching payload
// bytes or a GPU.
struct SparseReadRecipeRowgroupStats {
	size_t                       rowgroup_index         = 0U;
	size_t                       rowgroup_storage_bytes = 0U;
	std::vector<SparseReadRange> exact_ranges;
};

struct SparseReadRecipeWriteStats {
	size_t rowgroup_count         = 0U;
	size_t selected_vector_count  = 0U;
	size_t exact_range_count      = 0U;
	size_t exact_storage_bytes    = 0U;
	size_t sidecar_bytes          = 0U;
	uint64_t source_fingerprint   = 0U;
	uint64_t source_stat_digest   = 0U;
	uint64_t descriptor_digest    = 0U;
	uint64_t sidecar_crc64        = 0U;
	bool reused_existing          = false;
	std::vector<SparseReadRecipeRowgroupStats> rowgroups;
};

SparseReadRecipeWriteStats write_sparse_read_recipe(
	const std::filesystem::path&                   fls_path,
	const std::filesystem::path&                   recipe_path,
	uint64_t                                       source_fingerprint,
	const std::vector<SparseReadRecipeSelection>& selections);

[[nodiscard]] std::filesystem::path sparse_read_recipe_path(
	const std::filesystem::path& directory,
	uint32_t                     shard_id);

} // namespace galp::format

#endif // GALP_SPARSE_READ_RECIPE_HPP
