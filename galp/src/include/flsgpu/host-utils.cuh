// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/host-utils.cuh
// ────────────────────────────────────────────────────────
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#define CUDA_SAFE_CALL(call)                                                                                           \
	do {                                                                                                               \
		cudaError_t err = call;                                                                                        \
		if (cudaSuccess != err) {                                                                                      \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err));  \
			exit(EXIT_FAILURE);                                                                                        \
		}                                                                                                              \
	} while (0)

#define CUDA_SAFE_CALL_TRACED(call)                                                                                    \
	do {                                                                                                               \
		fprintf(stderr, "Start CUDA_CALL ['%s': line %i\n", __FILE__, __LINE__);                                       \
		cudaError_t err = call;                                                                                        \
		fprintf(stderr, "End CUDA_CALL ['%s': line %i\n", __FILE__, __LINE__);                                         \
		if (cudaSuccess != err) {                                                                                      \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err));  \
			exit(EXIT_FAILURE);                                                                                        \
		}                                                                                                              \
	} while (0)

namespace flsgpu { namespace memory {

struct DeviceAllocInfo {
	size_t       size         = 0;
	bool         async_alloc  = false;
	cudaStream_t alloc_stream = nullptr;
	bool         sub_alloc    = false; // true for arena sub-pointers (no-op on free)
};

struct PendingCopyEvent {
	cudaEvent_t event  = nullptr;
	void*       pinned = nullptr;
};

class DevicePool {
public:
	static DevicePool& instance() {
		static DevicePool pool;
		return pool;
	}

	void* alloc(size_t bytes) {
		return alloc_on_stream(bytes, nullptr);
	}

	void* alloc_on_stream(size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return nullptr;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		if (enabled_) {
			if (use_async_) {
				auto& free_list = free_async_by_stream_size_[stream_key(stream)][bytes];
				if (!free_list.empty()) {
					void* ptr = free_list.back();
					free_list.pop_back();
					in_use_[ptr] = DeviceAllocInfo {bytes, true, stream};
					return ptr;
				}
			} else {
				auto& free_list = free_sync_by_size_[bytes];
				if (!free_list.empty()) {
					void* ptr = free_list.back();
					free_list.pop_back();
					in_use_[ptr] = DeviceAllocInfo {bytes, false, nullptr};
					return ptr;
				}
			}
		}

		void* ptr         = nullptr;
		bool  async_alloc = false;
		if (use_async_) {
			auto status = cudaMallocAsync(&ptr, bytes, stream);
			if (status == cudaSuccess) {
				async_alloc = true;
			} else {
				ptr = nullptr;
			}
		}
		if (!ptr) {
			CUDA_SAFE_CALL(cudaMalloc(&ptr, bytes));
			async_alloc = false;
		}
		in_use_[ptr] = DeviceAllocInfo {bytes, async_alloc, async_alloc ? stream : nullptr};
		return ptr;
	}

	void register_sub_allocation(void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = in_use_.find(ptr);
		if (it != in_use_.end()) {
			// Preserve the real arena-base allocation record when a zero-sized entry
			// aliases offset 0. Re-registering an existing sub-allocation is harmless.
			if (!it->second.sub_alloc) {
				return;
			}
		}
		in_use_[ptr] = DeviceAllocInfo {0, false, nullptr, true};
	}


	// free() calls cudaFree if ptr is not tracked (legacy non-pool pointer fallback).
	// release_arena_ptr() silently no-ops instead — used by DeviceArena destructor
	// so that sub-pointers and device_base_ already cleaned up by free_device_expr()
	// don't crash via cudaFree on an interior address (cudaErrorInvalidValue) or
	// double-free device_base_ out of the pool free-list.
	void free(void* ptr)              { do_free(ptr, /*fallback_cudafree=*/true);  }
	void release_arena_ptr(void* ptr) { do_free(ptr, /*fallback_cudafree=*/false); }

	void* alloc_pinned(size_t bytes) {
		if (bytes == 0) {
			return nullptr;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		return alloc_pinned_locked(bytes);
	}

	void release_pinned(void* ptr) {
		std::lock_guard<std::mutex> lock(mutex_);
		release_pinned_locked(ptr);
	}

	// Register a stream the caller has already issued async H2D work on so
	// sync_h2d() (no-arg) and sync_h2d(stream) drain it before reset/destroy.
	// Use when bypassing queue_staged_h2d/copy_h2d_on_stream for aggregate DMA
	// issue (DeviceArena::upload). pinned ownership stays with the caller.
	void register_external_h2d(cudaStream_t stream) {
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(stream)].push_back(PendingCopyEvent {event, nullptr});
	}

	// Queue a single H2D transfer from an already-pinned buffer.
	// When batch is active, queued as one entry; pinned is freed after batch event.
	// When batch is not active, issued directly with its own event.
	void queue_staged_h2d(void* dst, void* pinned_src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream));
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(stream)].push_back(PendingCopyEvent {event, pinned_src});
	}

	void copy_h2d(void* dst, const void* src, size_t bytes) {
		copy_h2d_on_stream(dst, src, bytes, nullptr);
	}

	void copy_h2d_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
		if (bytes == 0) {
			return;
		}
		void* pinned = nullptr;
		if (use_pinned_ && bytes > small_copy_threshold_) {
			pinned = alloc_pinned(bytes);
			std::memcpy(pinned, src, bytes);
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice, stream));
		} else {
			CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
		}
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_copy_events_[stream_key(stream)].push_back(PendingCopyEvent {event, pinned});
	}

	void sync_h2d() {
		// Collect streams to synchronize while holding the lock, then release
		// the lock before calling cudaStreamSynchronize (which may block).
		std::vector<cudaStream_t> streams_to_sync;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			for (const auto& [stream_id, events] : pending_copy_events_) {
				(void)events;
				streams_to_sync.push_back(stream_from_key(stream_id));
			}
		}

		for (auto* s : streams_to_sync) {
			CUDA_SAFE_CALL(cudaStreamSynchronize(s));
		}

		// Now reclaim events and pinned memory under the lock.
		std::lock_guard<std::mutex> lock(mutex_);
		for (auto& [stream_id, events] : pending_copy_events_) {
			(void)stream_id;
			for (auto& pending : events) {
				if (pending.event != nullptr) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
				}
				release_pinned_locked(pending.pinned);
			}
		}
		pending_copy_events_.clear();
	}

	void sync_h2d(cudaStream_t source_stream) {
		CUDA_SAFE_CALL(cudaStreamSynchronize(source_stream));

		// Erase all pending records keyed by this stream before the caller destroys it.
		// This prevents stale-key reuse if CUDA later recycles the same stream handle value.
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = pending_copy_events_.find(stream_key(source_stream));
		if (it != pending_copy_events_.end()) {
			for (auto& pending : it->second) {
				if (pending.event != nullptr) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
				}
				release_pinned_locked(pending.pinned);
			}
			pending_copy_events_.erase(it);
		}
	}

	void set_enabled(bool enabled) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_enabled");
		enabled_ = enabled;
	}
	void set_use_async(bool use_async) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_use_async");
		use_async_ = use_async;
	}
	void set_use_pinned(bool use_pinned) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_use_pinned");
		use_pinned_ = use_pinned;
	}
	void set_small_copy_threshold(size_t bytes) {
		std::lock_guard<std::mutex> lock(mutex_);
		assert_idle_for_reconfiguration_locked("set_small_copy_threshold");
		small_copy_threshold_ = bytes;
	}

	~DevicePool() {
		sync_h2d();

		{
			size_t real_leaks = 0;
			for (auto& [ptr, info] : in_use_) {
				(void)ptr;
				if (!info.sub_alloc) ++real_leaks;
			}
			if (real_leaks > 0) {
				fprintf(stderr, "DevicePool warning: %zu in-use device allocations at shutdown; forcing free.\n",
				        real_leaks);
			}
		}
		for (auto& [ptr, info] : in_use_) {
			if (info.sub_alloc) continue; // interior arena pointer — no standalone cudaFree
			cudaFree(ptr);
		}
		in_use_.clear();

		for (auto& [size, list] : free_sync_by_size_) {
			(void)size;
			for (void* ptr : list) {
				cudaFree(ptr);
			}
		}
		free_sync_by_size_.clear();

		for (auto& [stream_id, buckets] : free_async_by_stream_size_) {
			(void)stream_id;
			for (auto& [size, list] : buckets) {
				(void)size;
				for (void* ptr : list) {
					cudaFree(ptr);
				}
			}
		}
		free_async_by_stream_size_.clear();

		if (!pinned_in_use_.empty()) {
			fprintf(stderr, "DevicePool warning: %zu in-use pinned allocations at shutdown; forcing free.\n",
			        pinned_in_use_.size());
		}
		for (auto& [ptr, bytes] : pinned_in_use_) {
			(void)bytes;
			cudaFreeHost(ptr);
		}
		pinned_in_use_.clear();

		for (auto& [size, list] : pinned_free_by_size_) {
			(void)size;
			for (void* ptr : list) {
				cudaFreeHost(ptr);
			}
		}
		pinned_free_by_size_.clear();
	}

private:
	using StreamKey = uintptr_t;

	static StreamKey stream_key(cudaStream_t stream) {
		return reinterpret_cast<StreamKey>(stream);
	}

	static cudaStream_t stream_from_key(StreamKey key) {
		return reinterpret_cast<cudaStream_t>(key);
	}

	void do_free(void* ptr, bool fallback_cudafree) {
		if (ptr == nullptr) {
			return;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = in_use_.find(ptr);
		if (it == in_use_.end()) {
			if (fallback_cudafree) {
				CUDA_SAFE_CALL(cudaFree(ptr));
			}
			return;
		}
		if (it->second.sub_alloc) {
			in_use_.erase(it);
			return; // arena sub-pointer: no actual GPU free
		}
		const auto info = it->second;
		in_use_.erase(it);
		if (enabled_) {
			if (info.async_alloc) {
				free_async_by_stream_size_[stream_key(info.alloc_stream)][info.size].push_back(ptr);
			} else {
				free_sync_by_size_[info.size].push_back(ptr);
			}
			return;
		}
		if (info.async_alloc) {
			CUDA_SAFE_CALL(cudaFreeAsync(ptr, info.alloc_stream));
			return;
		}
		CUDA_SAFE_CALL(cudaFree(ptr));
	}

	void release_pinned_locked(void*& pinned) {
		if (pinned == nullptr) {
			return;
		}
		auto pinned_it = pinned_in_use_.find(pinned);
		if (pinned_it != pinned_in_use_.end()) {
			const size_t bytes = pinned_it->second;
			pinned_in_use_.erase(pinned_it);
			if (use_pinned_) {
				pinned_free_by_size_[bytes].push_back(pinned);
			} else {
				CUDA_SAFE_CALL(cudaFreeHost(pinned));
			}
		} else {
			CUDA_SAFE_CALL(cudaFreeHost(pinned));
		}
		pinned = nullptr;
	}

	void* alloc_pinned_locked(size_t bytes) {
		if (use_pinned_) {
			// Best-fit: reuse the smallest free buffer >= bytes. Exact-size
			// matching forced fresh cudaMallocHost when staged_bytes variance
			// pushed a chunk into a nearby size bucket.
			for (auto it = pinned_free_by_size_.lower_bound(bytes); it != pinned_free_by_size_.end(); ++it) {
				if (!it->second.empty()) {
					const size_t bucket_size = it->first;
					void*        ptr         = it->second.back();
					it->second.pop_back();
					pinned_in_use_[ptr] = bucket_size;
					return ptr;
				}
			}
		}

		void* ptr = nullptr;
		CUDA_SAFE_CALL(cudaMallocHost(&ptr, bytes));
		pinned_in_use_[ptr] = bytes;
		return ptr;
	}

	void assert_idle_for_reconfiguration_locked(const char* api_name) {
		reclaim_finished_locked();
		// sub_alloc entries are phantom bookkeeping entries for arena interior pointers
		// and do not represent actual live GPU allocations — exclude them from the check.
		bool has_real_allocs = false;
		for (auto& [ptr, info] : in_use_) {
			(void)ptr;
			if (!info.sub_alloc) { has_real_allocs = true; break; }
		}
		if (!pending_copy_events_.empty() || has_real_allocs || !pinned_in_use_.empty()) {
			throw std::runtime_error(std::string("DevicePool::") + api_name +
			                         " requires idle pool (no in-flight copies or live allocations)");
		}
	}

	void reclaim_finished_locked() {
		for (auto it = pending_copy_events_.begin(); it != pending_copy_events_.end();) {
			auto& events = it->second;
			for (auto event_it = events.begin(); event_it != events.end();) {
				auto& pending = *event_it;
				if (pending.event == nullptr) {
					event_it = events.erase(event_it);
					continue;
				}
				auto status = cudaEventQuery(pending.event);
				if (status == cudaSuccess) {
					CUDA_SAFE_CALL(cudaEventDestroy(pending.event));
					pending.event = nullptr;
					release_pinned_locked(pending.pinned);
					event_it = events.erase(event_it);
				} else if (status == cudaErrorNotReady) {
					++event_it;
				} else {
					CUDA_SAFE_CALL(status);
				}
			}
			if (events.empty()) {
				it = pending_copy_events_.erase(it);
			} else {
				++it;
			}
		}
	}

	DevicePool() = default;

	std::mutex   mutex_;
	bool         enabled_              = true;
	bool         use_async_            = true;
	bool         use_pinned_           = true;
	size_t       small_copy_threshold_ = 256 * 1024;

	std::unordered_map<size_t, std::vector<void*>>                                      free_sync_by_size_;
	std::unordered_map<StreamKey, std::unordered_map<size_t, std::vector<void*>>> free_async_by_stream_size_;
	std::unordered_map<void*, DeviceAllocInfo>     in_use_;

	std::map<size_t, std::vector<void*>>           pinned_free_by_size_;
	std::unordered_map<void*, size_t>              pinned_in_use_;
	std::unordered_map<StreamKey, std::vector<PendingCopyEvent>> pending_copy_events_;
};

inline void* device_malloc(size_t bytes) {
	return DevicePool::instance().alloc(bytes);
}

inline void* device_malloc_on_stream(size_t bytes, cudaStream_t stream) {
	return DevicePool::instance().alloc_on_stream(bytes, stream);
}

inline void device_free(void* ptr) {
	DevicePool::instance().free(ptr);
}

inline void device_memcpy_h2d(void* dst, const void* src, size_t bytes) {
	DevicePool::instance().copy_h2d(dst, src, bytes);
}

inline void sync_h2d() {
	DevicePool::instance().sync_h2d();
}

inline void sync_h2d(cudaStream_t source_stream) {
	DevicePool::instance().sync_h2d(source_stream);
}

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
		release_device_base();
		release_pinned_base();
		run_deferred_frees();
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

	struct UploadPhaseMs {
		double layout_ms    = 0.0;
		double alloc_ms     = 0.0; // ensure_capacity + ensure_pinned_capacity
		double resolve_ms   = 0.0;
		double pack_ms      = 0.0; // std::memcpy into pinned
		double dma_issue_ms = 0.0; // cudaMemcpyAsync for regions + staged
	};
	UploadPhaseMs last_upload_phase_ms {};

	void upload(bool resolve_before_pack = false) {
		using clock   = std::chrono::steady_clock;
		const auto ms = [](auto a, auto b) {
			return std::chrono::duration<double, std::milli>(b - a).count();
		};
		last_upload_phase_ms = {};

		// POD-target resolution walks a flat descriptor array and writes device
		// addresses directly. Avoids std::function heap-alloc + vtable dispatch
		// that dominated host overhead when per-column resolvers were callbacks.
		const auto run_resolvers = [&]() {
			for (const auto& t : resolver_targets_) {
				*t.dst = device_base_ + entries_[t.entry_idx].device_offset;
			}
			resolver_targets_.clear();
			for (auto& fn : resolvers_) {
				fn();
			}
			resolvers_.clear();
		};
		const auto finalize = [&]() {
			run_deferred_frees();
			regions_.clear();
		};

		if (entries_.empty()) {
			run_resolvers();
			finalize();
			return;
		}

		const auto t0 = clock::now();
		// Layout pass: each backing region gets a 256B-aligned device slab in
		// registration order; the staged area tails them. Then resolve every
		// entry to its final device offset against this layout.
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

		const size_t total_bytes = cursor;
		if (total_bytes == 0) {
			run_resolvers();
			finalize();
			return;
		}
		const auto t1 = clock::now();
		last_upload_phase_ms.layout_ms = ms(t0, t1);

		// Round up to power-of-2 buckets (min 64KB) to improve DevicePool cache hits.
		ensure_capacity(round_up_pow2(total_bytes, 65536U));
		if (staged_bytes_ > 0) {
			ensure_pinned_capacity(round_up_pow2(staged_bytes_, 65536U));
		}
		const auto t2 = clock::now();
		last_upload_phase_ms.alloc_ms = ms(t1, t2);

		if (resolve_before_pack) {
			run_resolvers();
		}
		const auto t3 = clock::now();
		last_upload_phase_ms.resolve_ms = ms(t2, t3);

		// Pack staged entries (metadata + fallback scratch) into the pinned
		// buffer — one host pass feeds one aggregated DMA.
		if (staged_bytes_ > 0) {
			for (const auto& e : entries_) {
				if (e.region_idx < 0 && e.host_src != nullptr && e.copy_bytes > 0) {
					std::memcpy(pinned_base_ + e.staged_offset, e.host_src, e.copy_bytes);
				}
			}
		}
		const auto t4 = clock::now();
		last_upload_phase_ms.pack_ms = ms(t3, t4);

		// Unified DMA issue: one cudaMemcpyAsync per backing region (direct
		// from slot-owned pinned memory, zero host pack) plus one for the
		// staged area. Each DMA targets the device slab reserved above.
		for (const auto& region : regions_) {
			if (region.bytes == 0) {
				continue;
			}
			CUDA_SAFE_CALL(cudaMemcpyAsync(
			    device_base_ + region.device_offset, region.base, region.bytes,
			    cudaMemcpyHostToDevice, stream_));
		}
		if (staged_bytes_ > 0) {
			CUDA_SAFE_CALL(cudaMemcpyAsync(
			    device_base_ + staged_device_base, pinned_base_, staged_bytes_,
			    cudaMemcpyHostToDevice, stream_));
		}
		// Register a single stream event with DevicePool so sync_h2d() drains
		// these raw copies before the arena's pinned_base_/device_base_ are
		// freed or reused. One event covers all DMAs issued above because they
		// serialize on stream_.
		DevicePool::instance().register_external_h2d(stream_);
		const auto t5 = clock::now();
		last_upload_phase_ms.dma_issue_ms = ms(t4, t5);

		if (!resolve_before_pack) {
			run_resolvers();
		}
		finalize();
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

inline void device_memcpy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
	DevicePool::instance().copy_h2d_on_stream(dst, src, bytes, stream);
}

}} // namespace flsgpu::memory

template <typename T>
void free_device_pointer(T*& device_ptr) {
	if (device_ptr != nullptr) {
		flsgpu::memory::device_free(device_ptr);
	}
	device_ptr = nullptr;
}

template <typename T>
class GPUArray {
private:
	size_t allocation_size;
	size_t memory_size;
	T*     device_ptr = nullptr;

	void allocate() {
		device_ptr = reinterpret_cast<T*>(flsgpu::memory::device_malloc(allocation_size));
	}
	void allocate(cudaStream_t stream) {
		device_ptr = reinterpret_cast<T*>(flsgpu::memory::device_malloc_on_stream(allocation_size, stream));
	}

public:
	GPUArray(const size_t count) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate();
	}

	GPUArray(const size_t count, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		if (stream != nullptr) {
			allocate(stream);
		} else {
			allocate();
		}
	}

	GPUArray(const size_t count, const T* host_p) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate();
		flsgpu::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
	}

	GPUArray(const size_t count, const T* host_p, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size;
		allocate(stream);
		flsgpu::memory::device_memcpy_h2d_async(device_ptr, host_p, memory_size, stream);
	}

	GPUArray(const size_t count, const size_t buffer, const T* host_p) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size + buffer * sizeof(T);
		allocate();
		flsgpu::memory::device_memcpy_h2d(device_ptr, host_p, memory_size);
	}

	GPUArray(const size_t count, const size_t buffer, const T* host_p, cudaStream_t stream) {
		memory_size     = count * sizeof(T);
		allocation_size = memory_size + buffer * sizeof(T);
		allocate(stream);
		flsgpu::memory::device_memcpy_h2d_async(device_ptr, host_p, memory_size, stream);
	}

	// Copy constructor
	GPUArray(const GPUArray&) = delete;
	// Assignment operator deleted
	GPUArray& operator=(const GPUArray&) = delete;

	// Move constructor
	GPUArray(GPUArray&& other) noexcept
	    : allocation_size(other.allocation_size)
	    , memory_size(other.memory_size)
	    , device_ptr(other.device_ptr) {
		other.allocation_size = 0;
		other.memory_size     = 0;
		other.device_ptr      = nullptr;
	}

	// Assignment operator
	GPUArray& operator=(GPUArray&& other) noexcept {
		if (this != &other) {
			free_device_pointer(device_ptr);
			allocation_size       = other.allocation_size;
			memory_size           = other.memory_size;
			device_ptr            = other.device_ptr;
			other.allocation_size = 0;
			other.memory_size     = 0;
			other.device_ptr      = nullptr;
		}
		return *this;
	}

	~GPUArray() {
		free_device_pointer(device_ptr);
	}

	void copy_to_host(T* host_p) {
		CUDA_SAFE_CALL(cudaMemcpy(host_p, device_ptr, memory_size, cudaMemcpyDeviceToHost));
	}

	T* get() {
		return device_ptr;
	}

	T* release() {
		auto temp  = device_ptr;
		device_ptr = nullptr;
		return temp;
	}
};

#endif // GPU_UTILS_H
