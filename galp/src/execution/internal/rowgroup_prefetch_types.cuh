// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/execution/internal/rowgroup_prefetch_types.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_TYPES_CUH
#define ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_TYPES_CUH

#include "execution/internal/pinned_rowgroup_pool.cuh"
#include "storage/reader.cuh"
#include <chrono>
#include <cstddef>

namespace galp::runtime {

struct RowgroupReadTimeline {
	std::chrono::steady_clock::time_point read_submit {};
	std::chrono::steady_clock::time_point read_start {};
	std::chrono::steady_clock::time_point file_read_start {};
	std::chrono::steady_clock::time_point pread_start {};
	std::chrono::steady_clock::time_point pread_end {};
	std::chrono::steady_clock::time_point file_read_end {};
	std::chrono::steady_clock::time_point raw_ready_push {};
	std::chrono::steady_clock::time_point raw_ready_pop {};
	std::chrono::steady_clock::time_point build_start {};
	std::chrono::steady_clock::time_point build_end {};
	std::chrono::steady_clock::time_point ready_push {};
	std::chrono::steady_clock::time_point consumer_wait_start {};
	std::chrono::steady_clock::time_point consumer_wait_end {};
	std::chrono::steady_clock::time_point consumer_pop {};
	std::chrono::steady_clock::time_point upload_start {};
	std::chrono::steady_clock::time_point upload_end {};
	std::chrono::steady_clock::time_point read_end {};

	bool has_read_span() const {
		const std::chrono::steady_clock::time_point unset {};
		return read_start != unset && file_read_start != unset && file_read_end != unset && read_end != unset;
	}

	bool has_pread_span() const {
		const std::chrono::steady_clock::time_point unset {};
		return pread_start != unset && pread_end != unset;
	}
};

struct RowgroupReadTiming {
	double               read_ms                 = 0.0;
	double               file_read_ms            = 0.0;
	double               rowgroup_build_ms       = 0.0;
	double               pinned_acquire_ms       = 0.0;
	double               pread_ms                = 0.0;
	double               zero_copy_view_setup_ms = 0.0;
	RowgroupReadTimeline timeline {};
};

struct RowgroupPrefetchTiming {
	double depth_block_ms          = 0.0;
	double byte_block_ms           = 0.0;
	size_t worker_id               = PinnedRowgroupBufferPool::kNoOwner;
	bool   pool_slot_owner_reused  = false;
	bool   pool_slot_owner_migrated = false;
	bool   pool_slot_allocated     = false;
};

struct RowgroupReadResult {
	size_t                 rowgroup_index = 0;
	galp::format::Rowgroup       rowgroup {};
	size_t                 storage_bytes = 0;
	RowgroupReadTiming     timing {};
	RowgroupPrefetchTiming prefetch {};
};

} // namespace galp::runtime

#endif // ENGINE_EXECUTION_INTERNAL_ROWGROUP_PREFETCH_TYPES_CUH
