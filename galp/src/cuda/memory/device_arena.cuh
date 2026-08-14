// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/device_arena.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_DEVICE_ARENA_CUH
#define GALP_MEMORY_DEVICE_ARENA_CUH

#include "cuda/cuda_macros.cuh"
#include "cuda/memory/device_pool.cuh"
#include "cuda/memory/upload_metrics.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace galp::memory {

class DeviceArena {
	struct Entry {
		size_t      device_offset = 0;
		size_t      alloc_bytes   = 0;
		size_t      copy_bytes    = 0;
		const void* host_src      = nullptr;
		int         region_idx    = -1;
		size_t      region_offset = 0;
		size_t      staged_offset = 0;
	};

	struct BackingRegion {
		const std::byte* base          = nullptr;
		size_t           bytes         = 0;
		size_t           slab_bytes    = 0;
		size_t           device_offset = 0;
		bool             upload        = true;
	};

	struct ResolverTarget {
		void** dst       = nullptr;
		size_t entry_idx = 0;
	};

	static size_t         checked_add(size_t a, size_t b, const char* field);
	static size_t         checked_mul(size_t a, size_t b, const char* field);
	static std::uintptr_t checked_ptr_end(std::uintptr_t begin, size_t bytes, const char* field);
	static size_t         align_staged_offset(size_t value, size_t alignment = 256U);

public:
	explicit DeviceArena(cudaStream_t stream);
	~DeviceArena();

	DeviceArena(const DeviceArena&)            = delete;
	DeviceArena& operator=(const DeviceArena&) = delete;

	void register_backing(const void* base, size_t bytes, bool upload = true);
	void coalesce_backing_regions();

	template <typename T>
	size_t add(size_t count, const T* host_src, size_t buffer_elements = 0) {
		const size_t copy_bytes   = checked_mul(count, sizeof(T), "DeviceArena copy size overflow");
		const size_t buffer_bytes = checked_mul(buffer_elements, sizeof(T), "DeviceArena padding size overflow");
		const size_t alloc_bytes  = checked_add(copy_bytes, buffer_bytes, "DeviceArena allocation size overflow");
		const size_t idx          = entries_.size();

		if (host_src == nullptr && copy_bytes > 0) {
			throw std::invalid_argument("DeviceArena::add requires a non-null host source when copy_bytes > 0");
		}

		Entry e {};
		e.alloc_bytes = alloc_bytes;
		e.copy_bytes  = copy_bytes;
		e.host_src    = host_src;
		e.region_idx  = find_region(host_src, copy_bytes);

		if (e.region_idx < 0) {
			// The aggregate staged slab retains a 256-byte-aligned device
			// base, but individual values only require their C++ type
			// alignment. Padding every scalar/operand to 256 bytes inflated
			// JPEG workset H2D traffic by several megabytes per batch.
			staged_bytes_   = align_staged_offset(staged_bytes_, alignof(T));
			e.staged_offset = staged_bytes_;
			staged_bytes_   = checked_add(staged_bytes_, alloc_bytes, "DeviceArena staged area overflow");
			staged_entry_indices_.push_back(idx);
		} else {
			auto&       region      = regions_[e.region_idx];
			const auto* src         = reinterpret_cast<const std::byte*>(host_src);
			e.region_offset         = static_cast<size_t>(src - region.base);
			const size_t end_offset = checked_add(e.region_offset, alloc_bytes, "DeviceArena region slab overflow");
			if (end_offset > region.slab_bytes) {
				region.slab_bytes = end_offset;
			}
		}
		entries_.push_back(e);
		return idx;
	}

	void               add_resolver(std::function<void()> fn);
	void               resolve_to(void** dst, size_t entry_idx);
	void               defer_free(std::function<void()> fn);
	void               reset(bool preserve_capacity = false);
	void               set_minimum_capacity_bytes(size_t bytes);
	ArenaUploadMetrics upload(bool resolve_before_pack = false, bool backing_regions_coalesced = false);

	template <typename T>
	T* get(size_t idx) const {
		return reinterpret_cast<T*>(device_base_ + entries_[idx].device_offset);
	}

	size_t total_bytes() const;
	size_t entry_count() const;
	size_t capacity_bytes() const noexcept;
	size_t pinned_capacity_bytes() const noexcept;

private:
	struct DmaIssueStats {
		size_t bytes = 0;
		size_t count = 0;
	};

	struct LayoutPlan {
		size_t staged_device_base = 0;
		size_t total_bytes        = 0;
	};

	LayoutPlan    plan_layout();
	void          pack_staged_area();
	DmaIssueStats issue_dma(size_t staged_device_base);
	void          run_resolvers();
	void          index_new_region(size_t region_index);
	void          rebuild_region_index();
	int find_region(const void* host_src, const size_t bytes) const {
		if (host_src == nullptr || bytes == 0 || regions_.empty()) {
			return -1;
		}
		const auto src = reinterpret_cast<std::uintptr_t>(host_src);
		if (bytes > std::numeric_limits<std::uintptr_t>::max() - src) {
			throw std::overflow_error("DeviceArena source range overflow");
		}
		const auto contains = [&](const BackingRegion& region) {
			const auto base = reinterpret_cast<std::uintptr_t>(region.base);
			if (src < base) {
				return false;
			}
			const auto offset = static_cast<size_t>(src - base);
			return offset <= region.bytes && bytes <= region.bytes - offset;
		};
		if (last_region_index_ < regions_.size() && contains(regions_[last_region_index_])) {
			return static_cast<int>(last_region_index_);
		}
		if (regions_disjoint_) {
			const auto position = std::upper_bound(region_order_.begin(),
			                                       region_order_.end(),
			                                       src,
			                                       [&](const std::uintptr_t address, const size_t region_index) {
				                                       return address < reinterpret_cast<std::uintptr_t>(
				                                                            regions_[region_index].base);
			                                       });
			if (position == region_order_.begin()) {
				return -1;
			}
			const size_t index = *(position - 1);
			if (contains(regions_[index])) {
				last_region_index_ = index;
				return static_cast<int>(index);
			}
			return -1;
		}
		for (size_t index = 0U; index < regions_.size(); ++index) {
			if (contains(regions_[index])) {
				last_region_index_ = index;
				return static_cast<int>(index);
			}
		}
		return -1;
	}
	bool          ensure_capacity(size_t alloc_bytes);
	bool          ensure_pinned_capacity(size_t alloc_bytes);
	void          release_device_base();
	void          release_pinned_base();
	void          run_deferred_frees();
	static size_t round_up_pow2(size_t bytes, size_t min_bucket);
	static bool   should_measure_h2d();

	cudaStream_t                       stream_                = nullptr;
	char*                              device_base_           = nullptr;
	char*                              pinned_base_           = nullptr;
	size_t                             capacity_bytes_        = 0;
	size_t                             pinned_capacity_bytes_ = 0;
	size_t                             minimum_capacity_bytes_ = 0;
	size_t                             staged_bytes_          = 0;
	std::vector<Entry>                 entries_;
	std::vector<size_t>                staged_entry_indices_;
	std::vector<BackingRegion>         regions_;
	// Stable region indices sorted by host address. Compact-v3 contributes
	// disjoint pinned shard arenas, while small codec operands often live
	// outside every arena; indexed negative lookups avoid a full shard scan.
	std::vector<size_t>                region_order_;
	bool                               regions_disjoint_ = true;
	// Workset construction registers one backing and then appends many codec
	// chunks from that same region. Cache its index so the common containment
	// lookup is O(1); mutations that can reorder or clear regions invalidate it.
	mutable size_t                     last_region_index_ = static_cast<size_t>(-1);
	std::vector<ResolverTarget>        resolver_targets_;
	std::vector<std::function<void()>> resolvers_;
	std::vector<std::function<void()>> deferred_frees_;
};

} // namespace galp::memory

#endif // GALP_MEMORY_DEVICE_ARENA_CUH
