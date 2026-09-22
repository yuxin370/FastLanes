// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/transfer_tracker.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_MEMORY_TRANSFER_TRACKER_CUH
#define GALP_MEMORY_TRANSFER_TRACKER_CUH

#include "cuda/cuda_macros.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace galp::memory {

// Tracks in-flight H2D transfers per stream so a caller can drain them before
// tearing down streams or pinned buffers. Entries retain pinned ownership until
// completion is proven; the pool supplies the release operation at drain time.
class TransferTracker {
public:
	using ReleasePinnedFn = std::function<void(void*)>;

	// Establish ownership before submitting any DMA. The callback may attach a
	// pinned allocation directly to the entry. A failed submit/record leaves an
	// unrecorded entry that can only be reclaimed by synchronizing its stream.
	template <typename Submit>
	void submit(cudaStream_t stream, Submit&& issue) {
		const auto                  stream_id = key(stream);
		std::lock_guard<std::mutex> lock(mutex_);
		auto&                       entries = pending_[stream_id];
		entries.emplace_back();
		auto& entry = entries.back();
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&entry.event, cudaEventDisableTiming));
		issue(entry.pinned);
		CUDA_SAFE_CALL(cudaEventRecord(entry.event, stream));
		entry.recorded = true;
	}

	// Block until every tracked stream is idle, then destroy pending events and
	// invoke the release callback on every pinned pointer we held.
	void sync_all(const ReleasePinnedFn& release_pinned) {
		std::lock_guard<std::mutex> lock(mutex_);
		cudaError_t                 cleanup_error = cudaSuccess;
		for (auto it = pending_.begin(); it != pending_.end();) {
			ScopedDevice device(it->first.first);
			CUDA_SAFE_CALL(cudaStreamSynchronize(stream_from_key(it->first)));
			drain_entries(it->second, release_pinned, cleanup_error);
			if (it->second.empty()) {
				it = pending_.erase(it);
			} else {
				++it;
			}
		}
		CUDA_SAFE_CALL(cleanup_error);
	}

	// Drain a single stream's entries. Must be called before the caller
	// destroys the stream handle so stale keys cannot be reused if CUDA
	// recycles the handle.
	void sync_stream(cudaStream_t stream, const ReleasePinnedFn& release_pinned, bool only_pending = false) {
		const auto                  stream_id = key(stream);
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = pending_.find(stream_id);
		if (only_pending && it == pending_.end()) {
			return;
		}
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
		if (it != pending_.end()) {
			cudaError_t cleanup_error = cudaSuccess;
			drain_entries(it->second, release_pinned, cleanup_error);
			if (it->second.empty()) {
				pending_.erase(it);
			}
			CUDA_SAFE_CALL(cleanup_error);
		}
	}

	// Drop a stream's tracked entries without synchronizing the stream. Callers
	// must only use this after a later dependency has proven that every tracked
	// event on this stream has completed.
	void complete_stream(cudaStream_t stream, const ReleasePinnedFn& release_pinned) {
		const auto                  stream_id = key(stream);
		std::lock_guard<std::mutex> lock(mutex_);
		auto                        it = pending_.find(stream_id);
		if (it == pending_.end()) {
			return;
		}
		cudaError_t cleanup_error = cudaSuccess;
		drain_entries(it->second, release_pinned, cleanup_error);
		if (it->second.empty()) {
			pending_.erase(it);
		}
		CUDA_SAFE_CALL(cleanup_error);
	}

	// Non-blocking sweep: for entries whose event has completed, invoke the
	// release callback and drop them from tracking. Used by the pool's idle
	// check so a reconfiguration doesn't reject on stale-but-done entries.
	void reclaim_finished(const ReleasePinnedFn& release_pinned) {
		std::lock_guard<std::mutex> lock(mutex_);
		cudaError_t                 cleanup_error = cudaSuccess;
		for (auto it = pending_.begin(); it != pending_.end();) {
			ScopedDevice device(it->first.first);
			auto&        entries = it->second;
			for (auto entry_it = entries.begin(); entry_it != entries.end();) {
				auto status = entry_it->complete   ? cudaSuccess
				              : entry_it->recorded ? cudaEventQuery(entry_it->event)
				                                   : cudaErrorNotReady;
				if (status == cudaSuccess) {
					if (drain_entry(*entry_it, release_pinned, cleanup_error)) {
						entry_it = entries.erase(entry_it);
					} else {
						++entry_it;
					}
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
		CUDA_SAFE_CALL(cleanup_error);
	}

	bool empty() {
		std::lock_guard<std::mutex> lock(mutex_);
		return pending_.empty();
	}

private:
	using StreamKey = std::pair<int, uintptr_t>;

	class ScopedDevice {
	public:
		explicit ScopedDevice(int device) {
			CUDA_SAFE_CALL(cudaGetDevice(&previous_));
			if (previous_ != device) {
				CUDA_SAFE_CALL(cudaSetDevice(device));
				restore_ = true;
			}
		}
		~ScopedDevice() {
			if (restore_) {
				CUDA_LOG_CALL(cudaSetDevice(previous_));
			}
		}

		ScopedDevice(const ScopedDevice&)            = delete;
		ScopedDevice& operator=(const ScopedDevice&) = delete;

	private:
		int  previous_ = -1;
		bool restore_  = false;
	};

	struct PendingEntry {
		cudaEvent_t event    = nullptr;
		void*       pinned   = nullptr;
		bool        recorded = false;
		bool        complete = false;
	};

	static StreamKey key(cudaStream_t s) {
		int device = 0;
		CUDA_SAFE_CALL(cudaGetDevice(&device));
		return {device, reinterpret_cast<uintptr_t>(s)};
	}
	static cudaStream_t stream_from_key(StreamKey k) {
		return reinterpret_cast<cudaStream_t>(k.second);
	}

	static bool drain_entry(PendingEntry& entry, const ReleasePinnedFn& release_pinned, cudaError_t& first_error) {
		entry.complete = true;
		if (entry.event != nullptr) {
			const auto status = cudaEventDestroy(entry.event);
			if (status == cudaSuccess) {
				entry.event = nullptr;
			} else if (first_error == cudaSuccess) {
				first_error = status;
			}
		}
		if (entry.pinned != nullptr && release_pinned) {
			release_pinned(entry.pinned);
			entry.pinned = nullptr;
		}
		return entry.event == nullptr && entry.pinned == nullptr;
	}

	static void
	drain_entries(std::vector<PendingEntry>& entries, const ReleasePinnedFn& release_pinned, cudaError_t& first_error) {
		for (auto it = entries.begin(); it != entries.end();) {
			if (drain_entry(*it, release_pinned, first_error)) {
				it = entries.erase(it);
			} else {
				++it;
			}
		}
	}

	std::mutex                                     mutex_;
	std::map<StreamKey, std::vector<PendingEntry>> pending_;
};

} // namespace galp::memory

#endif // GALP_MEMORY_TRANSFER_TRACKER_CUH
