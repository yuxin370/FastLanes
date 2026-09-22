// Internal transfer ownership: entries stay in pending_ through submission,
// draining and failures. CUDA and pinned-pool operations never hold mutex_.
#ifndef GALP_MEMORY_TRANSFER_TRACKER_CUH
#define GALP_MEMORY_TRANSFER_TRACKER_CUH

#include "cuda/cuda_macros.cuh"
#include "cuda/memory/memory_diagnostics.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace galp::memory {
class TransferTracker {
public:
	using ReleasePinnedFn = std::function<void(void*)>;

	template <typename Submit>
	void submit(cudaStream_t stream, Submit&& issue) {
		submit(stream, [](void*&) {}, std::forward<Submit>(issue), {});
	}

	// prepare must not submit DMA. Once issue starts, an exception conservatively
	// requires stream synchronization, including partial DMA and record failures.
	template <typename Prepare, typename Submit>
	void submit(cudaStream_t stream, Prepare&& prepare, Submit&& issue, const ReleasePinnedFn& release_pinned) {
		[[maybe_unused]] diagnostics::SubmitTimer timer;
		const auto stream_id = key(stream);
		auto entry = std::make_shared<PendingEntry>();
		{
			std::lock_guard lock(mutex_);
			pending_[stream_id].push_back(entry);
		}
		bool dma_possible = false;
		try {
			CUDA_SAFE_CALL(cudaEventCreateWithFlags(&entry->event, cudaEventDisableTiming));
			prepare(entry->pinned);
			dma_possible = true;
			issue(entry->pinned);
			CUDA_SAFE_CALL(cudaEventRecord(entry->event, stream));
			entry->recorded = true;
		} catch (...) {
			const auto error = std::current_exception();
			if (!dma_possible) {
				// No DMA: cleanup can run immediately. If cleanup itself fails,
				// the entry remains owned and retryable, with the original error.
				cudaError_t cleanup_error = cudaSuccess;
				try { drain_entry(*entry, release_pinned, cleanup_error); } catch (...) {}
			}
			finish_submit(stream_id, entry);
			std::rethrow_exception(error);
		}
		finish_submit(stream_id, entry);
	}

	// Snapshot semantics: drains transfers reserved at entry, including any
	// still being submitted. Later submissions retain their own pending entries.
	void sync_all(const ReleasePinnedFn& release_pinned) {
		std::map<StreamKey, Entries> snapshot;
		{
			std::lock_guard lock(mutex_);
			snapshot = pending_;
		}
		cudaError_t cleanup_error = cudaSuccess;
		for (auto& [stream_id, entries] : snapshot) {
			drain(stream_id, entries, release_pinned, Mode::Sync, cleanup_error);
		}
		CUDA_SAFE_CALL(cleanup_error);
	}

	void sync_stream(cudaStream_t stream, const ReleasePinnedFn& release_pinned, bool only_pending = false) {
		const auto stream_id = key(stream);
		auto entries = snapshot(stream_id);
		if (only_pending && entries.empty()) return;
		cudaError_t cleanup_error = cudaSuccess;
		drain(stream_id, entries, release_pinned, Mode::Sync, cleanup_error);
		CUDA_SAFE_CALL(cleanup_error);
	}

	// Caller must have proven completion of the tracked work by a later stream
	// dependency. Do not race this proof with new submissions on the same stream.
	void complete_stream(cudaStream_t stream, const ReleasePinnedFn& release_pinned) {
		const auto stream_id = key(stream);
		auto entries = snapshot(stream_id);
		cudaError_t cleanup_error = cudaSuccess;
		drain(stream_id, entries, release_pinned, Mode::Complete, cleanup_error);
		CUDA_SAFE_CALL(cleanup_error);
	}

	void reclaim_finished(const ReleasePinnedFn& release_pinned) {
		std::map<StreamKey, Entries> snapshot;
		{
			std::lock_guard lock(mutex_);
			snapshot = pending_;
		}
		cudaError_t cleanup_error = cudaSuccess;
		for (auto& [stream_id, entries] : snapshot) {
			drain(stream_id, entries, release_pinned, Mode::Query, cleanup_error);
		}
		CUDA_SAFE_CALL(cleanup_error);
	}

	bool empty() {
		std::lock_guard lock(mutex_);
		return pending_.empty();
	}

private:
	using StreamKey = std::pair<int, uintptr_t>;
	struct PendingEntry {
		cudaEvent_t event = nullptr;
		void* pinned = nullptr;
		bool recorded = false;
		bool complete = false;
		// These two flags are protected by mutex_. The exclusive submitter or
		// drainer accesses all other fields without holding that mutex.
		bool submitting = true;
		bool draining = false;
	};
	using Entries = std::vector<std::shared_ptr<PendingEntry>>;
	enum class Mode { Sync, Complete, Query };

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
			if (restore_) CUDA_LOG_CALL(cudaSetDevice(previous_));
		}
		ScopedDevice(const ScopedDevice&) = delete;
		ScopedDevice& operator=(const ScopedDevice&) = delete;
	private:
		int previous_ = -1;
		bool restore_ = false;
	};

	static StreamKey key(cudaStream_t stream) {
		int device = 0;
		CUDA_SAFE_CALL(cudaGetDevice(&device));
		return {device, reinterpret_cast<uintptr_t>(stream)};
	}

	Entries snapshot(StreamKey stream_id) {
		std::lock_guard lock(mutex_);
		const auto it = pending_.find(stream_id);
		return it == pending_.end() ? Entries {} : it->second;
	}

	void erase_released(StreamKey stream_id) {
		auto it = pending_.find(stream_id);
		if (it == pending_.end()) return;
		std::erase_if(it->second, [](const auto& entry) {
			return !entry->submitting && !entry->draining && entry->complete &&
			       entry->event == nullptr && entry->pinned == nullptr;
		});
		if (it->second.empty()) pending_.erase(it);
	}

	void finish_submit(StreamKey stream_id, const std::shared_ptr<PendingEntry>& entry) {
		std::lock_guard lock(mutex_);
		entry->submitting = false;
		erase_released(stream_id);
		changed_.notify_all();
	}

	void finish_drain(StreamKey stream_id, const Entries& entries) {
		std::lock_guard lock(mutex_);
		for (auto& entry : entries) entry->draining = false;
		erase_released(stream_id);
		changed_.notify_all();
	}

	void drain(StreamKey stream_id, Entries& entries, const ReleasePinnedFn& release_pinned,
	           Mode mode, cudaError_t& cleanup_error) {
		{
			std::unique_lock lock(mutex_);
			if (mode == Mode::Query) {
				std::erase_if(entries, [](const auto& entry) { return entry->submitting || entry->draining; });
			} else {
				changed_.wait(lock, [&] {
					return std::none_of(entries.begin(), entries.end(),
					                    [](const auto& entry) { return entry->submitting || entry->draining; });
				});
			}
			for (auto& entry : entries) entry->draining = true;
		}
		try {
			ScopedDevice device(stream_id.first);
			if (mode == Mode::Sync) {
				[[maybe_unused]] diagnostics::SyncTimer timer;
				CUDA_SAFE_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_id.second)));
			}
			for (auto& entry : entries) {
				if (mode == Mode::Query && !entry->complete) {
					if (!entry->recorded) continue;
					const auto status = cudaEventQuery(entry->event);
					if (status == cudaErrorNotReady) continue;
					CUDA_SAFE_CALL(status);
				}
				drain_entry(*entry, release_pinned, cleanup_error);
			}
		} catch (...) {
			finish_drain(stream_id, entries);
			throw;
		}
		finish_drain(stream_id, entries);
	}

	static void drain_entry(PendingEntry& entry, const ReleasePinnedFn& release_pinned, cudaError_t& first_error) {
		entry.complete = true;
		if (entry.event != nullptr) {
			const auto status = cudaEventDestroy(entry.event);
			if (status == cudaSuccess) entry.event = nullptr;
			else if (first_error == cudaSuccess) first_error = status;
		}
		if (entry.pinned != nullptr && release_pinned) {
			release_pinned(entry.pinned);
			entry.pinned = nullptr;
		}
	}

	diagnostics::Mutex<diagnostics::Tracker> mutex_;
	std::condition_variable_any changed_;
	std::map<StreamKey, Entries> pending_;
};
} // namespace galp::memory
#endif
