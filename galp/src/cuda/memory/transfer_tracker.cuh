// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/transfer_tracker.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_TRANSFER_TRACKER_CUH
#define GALP_MEMORY_TRANSFER_TRACKER_CUH

#include "cuda/memory/cuda_macros.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace galp::memory {

// Tracks in-flight H2D transfers per stream so a caller can drain them before
// tearing down streams or pinned buffers. The tracker does not own pinned
// memory — a release callback is handed in at drain time so the pinned pool
// can decide how to reclaim each buffer (return to free-list vs. cudaFreeHost).
class TransferTracker {
public:
	using ReleasePinnedFn = std::function<void(void*)>;

	// Record an event on `stream` without pinned ownership — used when the
	// caller has already issued its own cudaMemcpyAsync (e.g. DeviceArena's
	// aggregate DMA) and just needs `sync_*` to drain it.
	void register_external(cudaStream_t stream) {
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_[key(stream)].push_back(PendingEntry {event, nullptr});
	}

	// Record an event on `stream` and remember `pinned_to_release`. When the
	// tracker drains this stream it calls the supplied release callback on the
	// pointer. `pinned_to_release` may be null — the entry is still tracked so
	// sync has a synchronisation point.
	void register_transfer(cudaStream_t stream, void* pinned_to_release) {
		cudaEvent_t event {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(event, stream));
		std::lock_guard<std::mutex> lock(mutex_);
		pending_[key(stream)].push_back(PendingEntry {event, pinned_to_release});
	}

	// Block until every tracked stream is idle, then destroy pending events and
	// invoke the release callback on every pinned pointer we held.
	void sync_all(const ReleasePinnedFn& release_pinned) {
		std::unordered_map<StreamKey, std::vector<PendingEntry>> drained;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			drained.swap(pending_);
		}

		for (const auto& [stream_id, entries] : drained) {
			(void)entries;
			CUDA_SAFE_CALL(cudaStreamSynchronize(stream_from_key(stream_id)));
		}

		for (auto& [stream_id, entries] : drained) {
			(void)stream_id;
			drain_entries(entries, release_pinned);
		}
	}

	// Drain a single stream's entries. Must be called before the caller
	// destroys the stream handle so stale keys cannot be reused if CUDA
	// recycles the handle.
	void sync_stream(cudaStream_t stream, const ReleasePinnedFn& release_pinned) {
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

		std::vector<PendingEntry> entries;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			auto it = pending_.find(key(stream));
			if (it == pending_.end()) {
				return;
			}
			entries = std::move(it->second);
			pending_.erase(it);
		}
		drain_entries(entries, release_pinned);
	}

	// Non-blocking sweep: for entries whose event has completed, invoke the
	// release callback and drop them from tracking. Used by the pool's idle
	// check so a reconfiguration doesn't reject on stale-but-done entries.
	void reclaim_finished(const ReleasePinnedFn& release_pinned) {
		std::vector<PendingEntry> finished;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			for (auto it = pending_.begin(); it != pending_.end();) {
				auto& entries = it->second;
				for (auto entry_it = entries.begin(); entry_it != entries.end();) {
					if (entry_it->event == nullptr) {
						entry_it = entries.erase(entry_it);
						continue;
					}
					auto status = cudaEventQuery(entry_it->event);
					if (status == cudaSuccess) {
						finished.push_back(*entry_it);
						entry_it = entries.erase(entry_it);
					} else if (status == cudaErrorNotReady) {
						++entry_it;
					} else {
						CUDA_SAFE_CALL(status);
					}
				}
				if (entries.empty()) {
					it = pending_.erase(it);
				} else {
					++it;
				}
			}
		}
		drain_entries(finished, release_pinned);
	}

	bool empty() {
		std::lock_guard<std::mutex> lock(mutex_);
		return pending_.empty();
	}

private:
	using StreamKey = uintptr_t;

	struct PendingEntry {
		cudaEvent_t event  = nullptr;
		void*       pinned = nullptr;
	};

	static StreamKey key(cudaStream_t s) {
		return reinterpret_cast<StreamKey>(s);
	}
	static cudaStream_t stream_from_key(StreamKey k) {
		return reinterpret_cast<cudaStream_t>(k);
	}

	static void drain_entries(std::vector<PendingEntry>& entries, const ReleasePinnedFn& release_pinned) {
		for (auto& entry : entries) {
			if (entry.event != nullptr) {
				CUDA_SAFE_CALL(cudaEventDestroy(entry.event));
				entry.event = nullptr;
			}
			if (entry.pinned != nullptr && release_pinned) {
				release_pinned(entry.pinned);
			}
		}
	}

	std::mutex                                               mutex_;
	std::unordered_map<StreamKey, std::vector<PendingEntry>> pending_;
};

} // namespace galp::memory

#endif // GALP_MEMORY_TRANSFER_TRACKER_CUH
