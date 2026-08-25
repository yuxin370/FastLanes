#ifndef GALP_DIRECT_DCT_NATIVE_PIPELINE_STATE_HPP
#define GALP_DIRECT_DCT_NATIVE_PIPELINE_STATE_HPP

#include "direct_dct/logical_types.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace galp::direct_dct {

// One atomic stage plus the legacy started flag. This replaces the legacy
// telemetry integer; it does not add an atomic, mutex, future, or wait.
class NativePipelineRequestState final {
public:
	[[nodiscard]] bool try_start() noexcept {
		started_.store(true, std::memory_order_release);
		auto expected = PipelineTraceEvent::Stage::kRequestAccepted;
		return stage_.compare_exchange_strong(
		    expected, PipelineTraceEvent::Stage::kPreparing, std::memory_order_acq_rel, std::memory_order_acquire);
	}

	[[nodiscard]] bool try_cancel() noexcept {
		auto expected = PipelineTraceEvent::Stage::kRequestAccepted;
		return stage_.compare_exchange_strong(
		    expected, PipelineTraceEvent::Stage::kCancelled, std::memory_order_acq_rel, std::memory_order_acquire);
	}

	void set_stage(const PipelineTraceEvent::Stage stage) noexcept {
		stage_.store(stage, std::memory_order_release);
	}

	[[nodiscard]] PipelineTraceEvent::Stage stage() const noexcept {
		return stage_.load(std::memory_order_acquire);
	}

	[[nodiscard]] bool started() const noexcept {
		return started_.load(std::memory_order_acquire);
	}

private:
	std::atomic<bool>                      started_ {false};
	std::atomic<PipelineTraceEvent::Stage> stage_ {PipelineTraceEvent::Stage::kRequestAccepted};
};

// Consumer-thread-owned snapshot of the native pipeline state. The live
// implementation uses only the two mutexes and futures already present in the
// legacy pipeline; this value does not add synchronization to execution.
struct NativePipelineState final {
	enum class Lifecycle : uint8_t {
		kIdle,
		kRunning,
		kDrained,
		kFailed,
		kClosed,
	};

	Lifecycle lifecycle               = Lifecycle::kIdle;
	size_t    request_count           = 0U;
	size_t    next_request            = 0U;
	size_t    pending_count           = 0U;
	size_t    max_pending_count       = 0U;
	size_t    completed_request_count = 0U;
	size_t    cancelled_request_count = 0U;
	bool      closed                  = true;
};

// Snapshot of the timing fields already produced by the legacy Torch
// scheduler. Native owns their measurement with the producer lifecycle while
// the Torch adapter remains the stable metrics projector.
struct NativePipelinePrefetchMetrics final {
	int64_t producer_active_nanoseconds     = 0;
	int64_t planning_nanoseconds            = 0;
	int64_t io_staging_nanoseconds          = 0;
	int64_t ordered_submission_nanoseconds  = 0;
	int64_t submit_to_ready_nanoseconds     = 0;
};

// Explicitly enabled test/shadow observation buffer. A null buffer pointer is
// the disabled mode: no trace lock, allocation, callback, CUDA event, or sync.
class NativePipelineTraceBuffer final {
public:
	void reset(size_t request_count);
	void emit(PipelineTraceEvent event);

	[[nodiscard]] std::vector<PipelineTraceEvent> snapshot() const;
	[[nodiscard]] size_t                          size() const;

private:
	mutable std::mutex              mutex_;
	std::vector<PipelineTraceEvent> events_;
	uint64_t                        next_prepare_ordinal_    = 1U;
	uint64_t                        next_stage_ordinal_      = 1U;
	uint64_t                        next_read_ordinal_       = 1U;
	uint64_t                        next_submission_ordinal_ = 1U;
	uint64_t                        next_completion_ordinal_ = 1U;
};

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_NATIVE_PIPELINE_STATE_HPP
