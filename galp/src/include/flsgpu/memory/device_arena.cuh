// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/memory/device_arena.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_MEMORY_DEVICE_ARENA_CUH
#define FLSGPU_MEMORY_DEVICE_ARENA_CUH

#include "flsgpu/memory/cuda_macros.cuh"
#include "flsgpu/memory/device_pool.cuh"
#include "flsgpu/memory/upload_metrics.cuh"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <utility>
#include <vector>

namespace flsgpu { namespace memory {

// ── DeviceArena: aggregated device allocation + staged H2D ──────────
// A workset can append many column sub-arrays into one arena, then perform one
// upload() to allocate/copy the aggregated payload and resolve device pointers.
//
// Host data must remain valid until upload().
// For temporary host columns, use defer_free() to extend their lifetime.
//
// Usage:
//   DeviceArena arena(stream);
//   auto i0 = arena.add<T>(count, host_ptr);
//   arena.upload();
//   T* d0 = arena.get<T>(i0);
class DeviceArena {
	struct Entry {
		size_t      device_offset = 0;       // resolved in upload()
		size_t      alloc_bytes   = 0;
		size_t      copy_bytes    = 0;
		const void* host_src      = nullptr;
		int         region_idx    = -1;      // -1 = staged via pinned_base_
		size_t      region_offset = 0;       // within-region offset if region_idx >= 0
		size_t      staged_offset = 0;       // within staged area if region_idx == -1
	};

	struct BackingRegion {
		const std::byte* base          = nullptr;
		size_t           bytes         = 0;  // bytes to DMA from host (the real data)
		size_t           slab_bytes    = 0;  // device slab size, >= bytes; covers tail
		                                     // padding for entries with buffer_elements.
		size_t           device_offset = 0;  // assigned in upload() layout pass
	};

	struct ResolverTarget {
		void** dst       = nullptr;
		size_t entry_idx = 0;
	};

public:
	explicit DeviceArena(cudaStream_t stream) : stream_(stream) {}
	~DeviceArena() {
		try {
			release_device_base();
			release_pinned_base();
			run_deferred_frees();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "DeviceArena destructor: %s\n", e.what());
		}
	}

	DeviceArena(const DeviceArena&)            = delete;
	DeviceArena& operator=(const DeviceArena&) = delete;

	/// Register a slot-owned pinned backing range. Any subsequent add<T>() whose
	/// host_src falls inside this range will be uploaded directly from the slot
	/// (one DMA per region), bypassing the staged pinned copy.
	/// Must be called before add<T>() for entries belonging to this region.
	void register_backing(const void* base, size_t bytes) {
		if (base == nullptr || bytes == 0) {
			return;
		}
		const auto* b = reinterpret_cast<const std::byte*>(base);
		for (const auto& r : regions_) {
			if (r.base == b && r.bytes == bytes) {
				return;
			}
		}
		regions_.push_back(BackingRegion {b, bytes, bytes, 0});
	}

	template <typename T>
	size_t add(size_t count, const T* host_src, size_t buffer_elements = 0) {
		const size_t copy_bytes  = count * sizeof(T);
		const size_t alloc_bytes = copy_bytes + buffer_elements * sizeof(T);
		const size_t idx         = entries_.size();

		Entry e {};
		e.alloc_bytes = alloc_bytes;
		e.copy_bytes  = copy_bytes;
		e.host_src    = host_src;
		e.region_idx  = find_region(host_src, copy_bytes);

		if (e.region_idx < 0) {
			staged_bytes_    = (staged_bytes_ + 255U) & ~size_t(255U);
			e.staged_offset  = staged_bytes_;
			staged_bytes_   += alloc_bytes;
		} else {
			auto&       region = regions_[e.region_idx];
			const auto* src    = reinterpret_cast<const std::byte*>(host_src);
			e.region_offset    = static_cast<size_t>(src - region.base);
			// Extend the region's device slab to cover this entry's tail padding.
			// The H2D copy still transfers only region.bytes (the real data); the
			// extra slab tail is uninitialized device memory reserved so padded
			// SIMD tail reads stay within the region's allocation instead of
			// spilling into the next region or staged area.
			const size_t end_offset = e.region_offset + alloc_bytes;
			if (end_offset > region.slab_bytes) {
				region.slab_bytes = end_offset;
			}
		}
		entries_.push_back(e);
		return idx;
	}

	/// Register a callback to be invoked after upload() resolves entry offsets.
	/// Use this to populate device column pointers after the aggregated upload.
	void add_resolver(std::function<void()> fn) {
		resolvers_.push_back(std::move(fn));
	}

	/// POD resolver: after upload() resolves entry offsets, write the resolved
	/// device address to *dst. Much faster than add_resolver() (no std::function
	/// heap allocation, no captured-reference indirection, tight inner loop).
	/// dst must point to storage that outlives upload() (e.g. into a pre-reserved
	/// Batch::device_exprs slot or into workset.device_batches).
	void resolve_to(void** dst, size_t entry_idx) {
		resolver_targets_.push_back(ResolverTarget {dst, entry_idx});
	}

	/// Defer a cleanup action until after upload() completes.
	/// Use this to extend the lifetime of temporary host columns whose data
	/// must remain valid until upload() packs it into the pinned buffer.
	void defer_free(std::function<void()> fn) {
		deferred_frees_.push_back(std::move(fn));
	}

	void reset() {
		release_device_base();
		run_deferred_frees();
		staged_bytes_ = 0;
		entries_.clear();
		regions_.clear();
		resolvers_.clear();
		resolver_targets_.clear();
	}

	ArenaUploadMetrics upload(bool resolve_before_pack = false) {
		using clock   = std::chrono::steady_clock;
		const auto ms = [](auto a, auto b) {
			return std::chrono::duration<double, std::milli>(b - a).count();
		};
		ArenaUploadMetrics metrics {};

		const auto finalize = [&]() {
			run_deferred_frees();
			regions_.clear();
		};

		if (entries_.empty()) {
			run_resolvers();
			finalize();
			return metrics;
		}

		const auto t0   = clock::now();
		const auto plan = plan_layout();
		if (plan.total_bytes == 0) {
			run_resolvers();
			finalize();
			return metrics;
		}
		const auto t1     = clock::now();
		metrics.layout_ms = ms(t0, t1);

		// Round up to power-of-2 buckets (min 64KB) to improve DevicePool cache hits.
		ensure_capacity(round_up_pow2(plan.total_bytes, 65536U));
		if (staged_bytes_ > 0) {
			ensure_pinned_capacity(round_up_pow2(staged_bytes_, 65536U));
		}
		const auto t2    = clock::now();
		metrics.alloc_ms = ms(t1, t2);

		if (resolve_before_pack) {
			run_resolvers();
		}
		const auto t3      = clock::now();
		metrics.resolve_ms = ms(t2, t3);

		pack_staged_area();
		const auto t4   = clock::now();
		metrics.pack_ms = ms(t3, t4);

		issue_dma(plan.staged_device_base);
		const auto t5        = clock::now();
		metrics.dma_issue_ms = ms(t4, t5);

		if (!resolve_before_pack) {
			run_resolvers();
		}
		finalize();
		return metrics;
	}

	template <typename T>
	T* get(size_t idx) const {
		return reinterpret_cast<T*>(device_base_ + entries_[idx].device_offset);
	}

	size_t total_bytes() const {
		size_t cursor = 0;
		for (const auto& r : regions_) {
			cursor = (cursor + 255U) & ~size_t(255U);
			cursor += r.slab_bytes;
		}
		cursor = (cursor + 255U) & ~size_t(255U);
		return cursor + staged_bytes_;
	}

	size_t entry_count() const {
		return entries_.size();
	}

private:
	struct LayoutPlan {
		size_t staged_device_base = 0;
		size_t total_bytes        = 0;
	};

	// Layout pass: each backing region gets a 256B-aligned device slab in
	// registration order; the staged area tails them. Mutates region/entry
	// device_offsets so later passes can use them directly.
	LayoutPlan plan_layout() {
		size_t cursor = 0;
		for (auto& region : regions_) {
			cursor               = (cursor + 255U) & ~size_t(255U);
			region.device_offset = cursor;
			cursor              += region.slab_bytes;
		}
		cursor                          = (cursor + 255U) & ~size_t(255U);
		const size_t staged_device_base = cursor;
		cursor                         += staged_bytes_;

		for (auto& e : entries_) {
			e.device_offset = (e.region_idx >= 0)
			                      ? regions_[e.region_idx].device_offset + e.region_offset
			                      : staged_device_base + e.staged_offset;
		}
		return LayoutPlan {staged_device_base, cursor};
	}

	// Pack staged entries (metadata + fallback scratch) into the pinned
	// buffer — one host pass feeds one aggregated DMA.
	void pack_staged_area() {
		if (staged_bytes_ == 0) {
			return;
		}
		for (const auto& e : entries_) {
			if (e.region_idx < 0 && e.host_src != nullptr && e.copy_bytes > 0) {
				std::memcpy(pinned_base_ + e.staged_offset, e.host_src, e.copy_bytes);
			}
		}
	}

	// Unified DMA issue: one cudaMemcpyAsync per backing region (direct from
	// slot-owned pinned memory, zero host pack) plus one for the staged area.
	// A single tracker event covers all DMAs since they serialize on stream_.
	void issue_dma(size_t staged_device_base) {
		for (const auto& region : regions_) {
			if (region.bytes == 0) {
				continue;
			}
			CUDA_SAFE_CALL(cudaMemcpyAsync(device_base_ + region.device_offset,
			                               region.base,
			                               region.bytes,
			                               cudaMemcpyHostToDevice,
			                               stream_));
		}
		if (staged_bytes_ > 0) {
			CUDA_SAFE_CALL(cudaMemcpyAsync(device_base_ + staged_device_base,
			                               pinned_base_,
			                               staged_bytes_,
			                               cudaMemcpyHostToDevice,
			                               stream_));
		}
		DevicePool::instance().register_external_h2d(stream_);
	}

	// Resolve flat pointer targets without std::function dispatch, then fire
	// any std::function resolvers the caller registered.
	void run_resolvers() {
		for (const auto& t : resolver_targets_) {
			*t.dst = device_base_ + entries_[t.entry_idx].device_offset;
		}
		resolver_targets_.clear();
		for (auto& fn : resolvers_) {
			fn();
		}
		resolvers_.clear();
	}

	int find_region(const void* host_src, size_t bytes) const {
		if (host_src == nullptr || bytes == 0 || regions_.empty()) {
			return -1;
		}
		const auto* src = reinterpret_cast<const std::byte*>(host_src);
		for (size_t i = 0; i < regions_.size(); ++i) {
			const auto& r = regions_[i];
			if (src >= r.base && src + bytes <= r.base + r.bytes) {
				return static_cast<int>(i);
			}
		}
		return -1;
	}

	void ensure_capacity(const size_t alloc_bytes) {
		if (device_base_ != nullptr && capacity_bytes_ >= alloc_bytes) {
			return;
		}
		release_device_base();
		auto& pool  = DevicePool::instance();
		device_base_ = reinterpret_cast<char*>(pool.alloc_on_stream(alloc_bytes, stream_));
		capacity_bytes_ = alloc_bytes;
	}

	void ensure_pinned_capacity(const size_t alloc_bytes) {
		if (pinned_base_ != nullptr && pinned_capacity_bytes_ >= alloc_bytes) {
			return;
		}
		release_pinned_base();
		auto& pool             = DevicePool::instance();
		pinned_base_           = reinterpret_cast<char*>(pool.alloc_pinned(alloc_bytes));
		pinned_capacity_bytes_ = alloc_bytes;
	}

	void release_device_base() {
		if (device_base_ == nullptr) {
			return;
		}
		auto& pool = DevicePool::instance();
		pool.release_arena_ptr(device_base_);
		device_base_    = nullptr;
		capacity_bytes_ = 0;
	}

	void release_pinned_base() {
		if (pinned_base_ == nullptr) {
			return;
		}
		auto& pool = DevicePool::instance();
		pool.release_pinned(pinned_base_);
		pinned_base_           = nullptr;
		pinned_capacity_bytes_ = 0;
	}

	void run_deferred_frees() {
		for (auto& fn : deferred_frees_) {
			fn();
		}
		deferred_frees_.clear();
	}

	/// Round up to the next power of 2 that is >= min_bucket.
	static size_t round_up_pow2(size_t bytes, size_t min_bucket) {
		if (bytes <= min_bucket) {
			return min_bucket;
		}
		// Next power of 2 >= bytes
		size_t v = bytes - 1;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v |= v >> 32;
		return v + 1;
	}

	cudaStream_t                       stream_      = nullptr;
	char*                              device_base_ = nullptr;
	char*                              pinned_base_ = nullptr;
	size_t                             capacity_bytes_ = 0;
	size_t                             pinned_capacity_bytes_ = 0;
	size_t                             staged_bytes_ = 0;
	std::vector<Entry>                 entries_;
	std::vector<BackingRegion>         regions_;
	std::vector<ResolverTarget>        resolver_targets_;
	std::vector<std::function<void()>> resolvers_;
	std::vector<std::function<void()>> deferred_frees_;
};

}} // namespace flsgpu::memory

#endif // FLSGPU_MEMORY_DEVICE_ARENA_CUH
