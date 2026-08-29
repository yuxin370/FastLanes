#ifndef GALP_DIRECT_DCT_NATIVE_LOGICAL_BATCH_PIPELINE_DETAIL_HPP
#define GALP_DIRECT_DCT_NATIVE_LOGICAL_BATCH_PIPELINE_DETAIL_HPP

#include "direct_dct/logical_types.hpp"
#include "direct_dct/native_batch_lifetime.hpp"
#include "direct_dct/native_pipeline_state.hpp"
#include "galp/jpeg_dct_device.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace galp::direct_dct::detail {

struct NativePlanTraceIdentity final {
	uint64_t plan_identity_hash = 0U;
	uint64_t io_identity_hash   = 0U;
};

// A narrow admission controller for full materialized outputs. A permit is
// attached to the produced DirectDctBatch and therefore remains live through
// the existing NativeBatchLease lifetime. This bounds producer admission
// without introducing a second allocator or completion system.
class MaterializedOutputSlotPool final
    : public std::enable_shared_from_this<MaterializedOutputSlotPool> {
public:
	struct Snapshot final {
		size_t capacity                  = 0U;
		size_t live_slots                = 0U;
		size_t peak_live_slots           = 0U;
		size_t waiters                   = 0U;
		size_t live_bytes                = 0U;
		size_t peak_bytes                = 0U;
		size_t maximum_slot_bytes        = 0U;
	};

	class Permit final {
	public:
		~Permit() {
			if (pool_) {
				pool_->release(bytes_);
			}
		}

		Permit(const Permit&)            = delete;
		Permit& operator=(const Permit&) = delete;

		void set_materialized_bytes(const size_t bytes) {
			if (bytes_ != 0U) {
				throw std::logic_error("materialized output slot bytes were already recorded");
			}
			bytes_ = bytes;
			pool_->add_bytes(bytes);
		}

	private:
		friend class MaterializedOutputSlotPool;
		explicit Permit(std::shared_ptr<MaterializedOutputSlotPool> pool)
		    : pool_(std::move(pool)) {
		}

		std::shared_ptr<MaterializedOutputSlotPool> pool_;
		size_t                                      bytes_ = 0U;
	};

	explicit MaterializedOutputSlotPool(const size_t capacity)
	    : capacity_(capacity) {
		if (capacity_ == 0U || capacity_ > 2U) {
			throw std::invalid_argument("materialized output slot capacity must be 1 or 2");
		}
	}

	[[nodiscard]] std::shared_ptr<Permit> acquire(const std::atomic<bool>& cancelled) {
		std::unique_lock lock(mutex_);
		++waiters_;
		while (live_slots_ >= capacity_ && !cancelled.load(std::memory_order_acquire)) {
			// Consumer completion is queried by the existing native lifetime
			// authority.  Progress it only when output admission is actually
			// blocked; otherwise an event that becomes ready just after the
			// final Tensor deleter ran would have no caller left to return the
			// permit.  The pool mutex must not be held while reclaim destroys a
			// permit and calls release().
			lock.unlock();
			static_cast<void>(NativeBatchLease::reclaim_finished());
			lock.lock();
			condition_.wait_for(lock, std::chrono::microseconds {250}, [&] {
				return live_slots_ < capacity_ || cancelled.load(std::memory_order_acquire);
			});
		}
		--waiters_;
		if (cancelled.load(std::memory_order_acquire)) {
			return {};
		}
		++live_slots_;
		peak_live_slots_ = std::max(peak_live_slots_, live_slots_);
		return std::shared_ptr<Permit>(new Permit(shared_from_this()));
	}

	void notify_waiters() noexcept {
		condition_.notify_all();
	}

	[[nodiscard]] Snapshot snapshot() const noexcept {
		std::lock_guard lock(mutex_);
		return {
		    capacity_,
		    live_slots_,
		    peak_live_slots_,
		    waiters_,
		    live_bytes_,
		    peak_bytes_,
		    maximum_slot_bytes_,
		};
	}

private:
	void add_bytes(const size_t bytes) {
		std::lock_guard lock(mutex_);
		live_bytes_ += bytes;
		peak_bytes_ = std::max(peak_bytes_, live_bytes_);
		maximum_slot_bytes_ = std::max(maximum_slot_bytes_, bytes);
	}

	void release(const size_t bytes) noexcept {
		{
			std::lock_guard lock(mutex_);
			if (live_slots_ == 0U || bytes > live_bytes_) {
				std::terminate();
			}
			--live_slots_;
			live_bytes_ -= bytes;
		}
		condition_.notify_one();
	}

	const size_t            capacity_ = 0U;
	mutable std::mutex      mutex_;
	std::condition_variable condition_;
	size_t                  live_slots_         = 0U;
	size_t                  peak_live_slots_    = 0U;
	size_t                  waiters_            = 0U;
	size_t                  live_bytes_         = 0U;
	size_t                  peak_bytes_         = 0U;
	size_t                  maximum_slot_bytes_ = 0U;
};

template <typename Runtime>
class NativeLogicalBatchPipelineCore final {
public:
	using Batch = typename Runtime::Batch;

	NativeLogicalBatchPipelineCore(Runtime                         runtime,
	                               std::string                     semantic_profile_id,
	                               jpeg::JpegDctDeviceBatchOptions options,
	                               NativePipelineTraceBuffer*      trace = nullptr,
	                               size_t materialized_output_slot_capacity = 2U)
	    : shared_state_(std::make_shared<SharedState>(
	          std::move(runtime), materialized_output_slot_capacity))
	    , semantic_profile_id_(std::move(semantic_profile_id))
	    , options_(std::move(options))
	    , trace_(trace) {
		if (semantic_profile_id_.empty()) {
			throw std::invalid_argument("NativeLogicalBatchPipeline semantic profile id must not be empty");
		}
	}

	~NativeLogicalBatchPipelineCore() {
		close();
	}

	NativeLogicalBatchPipelineCore(const NativeLogicalBatchPipelineCore&)            = delete;
	NativeLogicalBatchPipelineCore& operator=(const NativeLogicalBatchPipelineCore&) = delete;
	NativeLogicalBatchPipelineCore(NativeLogicalBatchPipelineCore&&)                 = delete;
	NativeLogicalBatchPipelineCore& operator=(NativeLogicalBatchPipelineCore&&)      = delete;

	void reset(std::vector<LogicalBatchRequest> logical_requests) {
		std::vector<QueuedRequest> lowered;
		lowered.reserve(logical_requests.size());
		for (size_t index = 0U; index < logical_requests.size(); ++index) {
			lowered.push_back(lower_request(std::move(logical_requests[index]), index));
		}

		close();
		if (trace_ != nullptr) {
			trace_->reset(lowered.size());
		}
		requests_ = std::move(lowered);
		last_prefetch_.reset();
		last_prefetch_metrics_ = {};
		pipeline_state_ = {};
		pipeline_state_.lifecycle =
		    requests_.empty() ? NativePipelineState::Lifecycle::kDrained : NativePipelineState::Lifecycle::kRunning;
		pipeline_state_.request_count = requests_.size();
		pipeline_state_.closed        = false;

		for (const auto& request : requests_) {
			emit_trace(request, PipelineTraceEvent::Stage::kRequestAccepted);
		}
		fill_pending();
	}

	Batch next() {
		if (pipeline_state_.closed || pending_.empty()) {
			throw std::out_of_range("NativeLogicalBatchPipeline is closed or exhausted");
		}
		auto current = pending_.front();
		try {
			auto batch              = current->read();
			last_output_slot_owner_ = current->take_output_slot_owner();
			last_prefetch_metrics_ = current->metrics();
			last_prefetch_ = std::move(current);
			pending_.pop_front();
			++pipeline_state_.completed_request_count;
			fill_pending();
			pipeline_state_.pending_count = pending_.size();
			if (pending_.empty() && pipeline_state_.next_request == requests_.size()) {
				pipeline_state_.lifecycle = NativePipelineState::Lifecycle::kDrained;
			}
			return batch;
		} catch (...) {
			close_impl(NativePipelineState::Lifecycle::kFailed);
			throw;
		}
	}

	[[nodiscard]] bool ready() const {
		return !pending_.empty() && pending_.front()->ready();
	}

	[[nodiscard]] bool started() const noexcept {
		return !pending_.empty() && pending_.front()->started();
	}

	[[nodiscard]] size_t prefetched_batch_count() const noexcept {
		return pipeline_state_.next_request;
	}

	[[nodiscard]] NativePipelineState state() const noexcept {
		auto state = pipeline_state_;
		const auto slots = shared_state_->output_slots->snapshot();
		state.output_slot_capacity      = slots.capacity;
		state.live_output_slots         = slots.live_slots;
		state.peak_live_output_slots    = slots.peak_live_slots;
		state.output_slot_waiters       = slots.waiters;
		state.live_output_bytes         = slots.live_bytes;
		state.peak_output_bytes         = slots.peak_bytes;
		state.maximum_output_slot_bytes = slots.maximum_slot_bytes;
		return state;
	}

	// Transfers the just-delivered permit to the actual DirectDctBatch. The
	// batch then carries it through NativeBatchLease until backing reclaim.
	[[nodiscard]] std::shared_ptr<void> take_delivered_output_slot_owner() noexcept {
		return std::move(last_output_slot_owner_);
	}

	[[nodiscard]] NativePipelinePrefetchMetrics prefetch_metrics() const noexcept {
		if (last_prefetch_) {
			return last_prefetch_->metrics();
		}
		if (!pending_.empty()) {
			return pending_.front()->metrics();
		}
		return last_prefetch_metrics_;
	}

	size_t close() noexcept {
		return close_impl(NativePipelineState::Lifecycle::kClosed);
	}

private:
	static constexpr size_t kNativePrefetchDepth = 2U;

	struct SharedState final {
		explicit SharedState(Runtime runtime_in, const size_t output_slot_capacity)
		    : runtime(std::move(runtime_in))
		    , output_slots(std::make_shared<MaterializedOutputSlotPool>(output_slot_capacity)) {
		}

		Runtime                  runtime;
		std::shared_ptr<MaterializedOutputSlotPool> output_slots;
		std::mutex               runtime_mutex;
		std::mutex               prefetch_mutex;
		std::shared_future<void> prefetch_tail;
	};

	struct QueuedRequest final {
		uint64_t                                   request_identity = 0U;
		uint64_t                                   request_ordinal  = 0U;
		uint64_t                                   batch_ordinal    = 0U;
		std::vector<jpeg::JpegDctImageCropRequest> requests;
	};

	struct PendingTelemetry final {
		NativePipelineRequestState execution_state;
		std::chrono::steady_clock::time_point submitted_at = std::chrono::steady_clock::now();
		std::atomic<bool>    cancel_requested {false};
		std::atomic<int64_t> producer_active_nanoseconds {0};
		std::atomic<int64_t> planning_nanoseconds {0};
		std::atomic<int64_t> io_staging_nanoseconds {0};
		std::atomic<int64_t> ordered_submission_nanoseconds {0};
		std::atomic<int64_t> submit_to_ready_nanoseconds {0};
		uint64_t                   request_identity = 0U;
		uint64_t                   request_ordinal  = 0U;
		uint64_t                   batch_ordinal    = 0U;
	};

	class Pending final {
	public:
		Pending(std::future<Batch>                                          future,
		        std::shared_ptr<PendingTelemetry>                           telemetry,
		        std::shared_ptr<MaterializedOutputSlotPool>                  output_slots,
		        std::shared_ptr<std::shared_ptr<MaterializedOutputSlotPool::Permit>> output_slot_owner,
		        NativePipelineTraceBuffer*                                  trace)
		    : future_(std::move(future))
		    , telemetry_(std::move(telemetry))
		    , output_slots_(std::move(output_slots))
		    , output_slot_owner_(std::move(output_slot_owner))
		    , trace_(trace) {
		}

		[[nodiscard]] bool ready() const {
			return future_.valid() && future_.wait_for(std::chrono::milliseconds {0}) == std::future_status::ready;
		}

		[[nodiscard]] bool started() const noexcept {
			return telemetry_->execution_state.started();
		}

		[[nodiscard]] NativePipelinePrefetchMetrics metrics() const noexcept {
			return {
			    telemetry_->producer_active_nanoseconds.load(std::memory_order_acquire),
			    telemetry_->planning_nanoseconds.load(std::memory_order_acquire),
			    telemetry_->io_staging_nanoseconds.load(std::memory_order_acquire),
			    telemetry_->ordered_submission_nanoseconds.load(std::memory_order_acquire),
			    telemetry_->submit_to_ready_nanoseconds.load(std::memory_order_acquire),
			};
		}

		bool cancel() noexcept {
			telemetry_->cancel_requested.store(true, std::memory_order_release);
			output_slots_->notify_waiters();
			const bool cancelled = telemetry_->execution_state.try_cancel();
			if (cancelled) {
				emit(PipelineTraceEvent::Stage::kCancelled);
			}
			return cancelled;
		}

		Batch read() {
			if (!future_.valid()) {
				throw std::runtime_error("NativeLogicalBatchPipeline batch has already been consumed");
			}
			return future_.get();
		}

		[[nodiscard]] std::shared_ptr<void> take_output_slot_owner() noexcept {
			return std::move(*output_slot_owner_);
		}

	private:
		void emit(const PipelineTraceEvent::Stage stage) const noexcept {
			if (trace_ == nullptr) {
				return;
			}
			try {
				PipelineTraceEvent event;
				event.request_identity = telemetry_->request_identity;
				event.request_ordinal  = telemetry_->request_ordinal;
				event.batch_ordinal    = telemetry_->batch_ordinal;
				event.stage            = stage;
				trace_->emit(event);
			} catch (...) {
				// Test diagnostics must never change cancellation or gate release.
			}
		}

		std::future<Batch>                                          future_;
		std::shared_ptr<PendingTelemetry>                           telemetry_;
		std::shared_ptr<MaterializedOutputSlotPool>                  output_slots_;
		std::shared_ptr<std::shared_ptr<MaterializedOutputSlotPool::Permit>> output_slot_owner_;
		NativePipelineTraceBuffer*                                  trace_ = nullptr;
	};

	QueuedRequest lower_request(LogicalBatchRequest logical, const size_t request_ordinal) const {
		validate_logical_batch_request(logical);
		if (logical.semantic_profile_id != semantic_profile_id_) {
			throw std::invalid_argument("NativeLogicalBatchPipeline request semantic profile does not match pipeline");
		}

		QueuedRequest lowered;
		lowered.request_identity = logical.request_identity;
		lowered.request_ordinal  = static_cast<uint64_t>(request_ordinal);
		lowered.batch_ordinal    = logical.batch_ordinal;
		lowered.requests.reserve(logical.samples.size());
		for (auto& sample : logical.samples) {
			jpeg::JpegDctImageCropRequest request;
			request.global_image_index = sample.image_id;
			if (sample.transform.source_crop.has_value()) {
				request.source_crop = *sample.transform.source_crop;
			}
			request.horizontal_flip   = sample.transform.horizontal_flip;
			request.logical_sample_id = std::move(sample.transform.logical_sample_id);
			request.augmentation_key  = std::move(sample.transform.augmentation_key);
			lowered.requests.push_back(std::move(request));
		}
		return lowered;
	}

	void fill_pending() {
		while (pending_.size() < kNativePrefetchDepth && pipeline_state_.next_request < requests_.size()) {
			pending_.push_back(prefetch(std::move(requests_[pipeline_state_.next_request])));
			++pipeline_state_.next_request;
			pipeline_state_.max_pending_count = std::max(pipeline_state_.max_pending_count, pending_.size());
		}
		pipeline_state_.pending_count = pending_.size();
	}

	std::shared_ptr<Pending> prefetch(QueuedRequest request) {
		auto state_copy                   = shared_state_;
		auto completion                   = std::make_shared<std::promise<void>>();
		auto telemetry                    = std::make_shared<PendingTelemetry>();
		telemetry->request_identity       = request.request_identity;
		telemetry->request_ordinal        = request.request_ordinal;
		telemetry->batch_ordinal          = request.batch_ordinal;
		auto options                      = options_;
		// Native admission is output-slot driven. The legacy/manual prefetch API
		// may still use transform_submission_gate, but the production native
		// pipeline must not bind submission to the next consumer iteration.
		options.transform_submission_gate.reset();
		auto output_slot_owner = std::make_shared<
		    std::shared_ptr<MaterializedOutputSlotPool::Permit>>();
		const int device_index            = state_copy->runtime.capture_device();

		std::shared_future<void> predecessor;
		{
			std::lock_guard lock(state_copy->prefetch_mutex);
			predecessor               = state_copy->prefetch_tail;
			state_copy->prefetch_tail = completion->get_future().share();
		}

		std::future<Batch> future;
		try {
			future = std::async(
			    std::launch::async,
			    [state_copy,
			     request = std::move(request),
			     options = std::move(options),
			     device_index,
			     predecessor = std::move(predecessor),
			     completion,
			     telemetry,
			     output_slot_owner,
			     trace = trace_]() mutable -> Batch {
				    try {
					    if (!telemetry->execution_state.try_start()) {
						    throw std::runtime_error(
						        "NativeLogicalBatchPipeline request was cancelled before execution");
					    }
					    const auto active_begin   = std::chrono::steady_clock::now();
					    const auto planning_begin = active_begin;
					    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kPreparing);

					    NativePlanTraceIdentity identity;
					    if (trace != nullptr) {
						    identity = state_copy->runtime.trace_identity(request.requests, options);
					    }
					    auto prepared = state_copy->runtime.prepare(request.requests, options);
					    const auto planning_end = std::chrono::steady_clock::now();
					    telemetry->planning_nanoseconds.store(
					        std::chrono::duration_cast<std::chrono::nanoseconds>(planning_end - planning_begin).count(),
					        std::memory_order_release);
					    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kPlanReady);
					    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kPlanReady, identity);

					    const auto io_begin = std::chrono::steady_clock::now();
					    state_copy->runtime.stage(prepared);
					    const auto io_end = std::chrono::steady_clock::now();
					    telemetry->io_staging_nanoseconds.store(
					        std::chrono::duration_cast<std::chrono::nanoseconds>(io_end - io_begin).count(),
					        std::memory_order_release);
					    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kStaged);
					    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kStaged, identity);

					    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kAwaitingPredecessor);
					    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kAwaitingPredecessor, identity);
					    if (predecessor.valid()) {
						    predecessor.wait();
					    }

					    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kAwaitingOutputSlot);
					    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kAwaitingOutputSlot, identity);
					    auto permit = state_copy->output_slots->acquire(telemetry->cancel_requested);
					    if (!permit) {
						    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kCancelled);
						    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kCancelled, identity);
						    throw std::runtime_error(
						        "NativeLogicalBatchPipeline request was cancelled while awaiting an output slot");
					    }
					    *output_slot_owner = permit;
					    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kOutputSlotAcquired);
					    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kOutputSlotAcquired, identity);

					    const auto submission_begin = std::chrono::steady_clock::now();
					    state_copy->runtime.activate_device(device_index);
					    Batch batch;
					    {
						    std::lock_guard lock(state_copy->runtime_mutex);
						    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kReadStarted);
						    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kReadStarted, identity);
						    batch = state_copy->runtime.read(std::move(prepared));
						    permit->set_materialized_bytes(
						        state_copy->runtime.materialized_output_bytes(batch));
						    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kSubmitted);
						    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kSubmitted, identity);
					    }
					    const auto submission_end = std::chrono::steady_clock::now();
					    telemetry->ordered_submission_nanoseconds.store(
					        std::chrono::duration_cast<std::chrono::nanoseconds>(submission_end - submission_begin).count(),
					        std::memory_order_release);
					    const auto active_end = std::chrono::steady_clock::now();
					    telemetry->producer_active_nanoseconds.store(
					        std::chrono::duration_cast<std::chrono::nanoseconds>(active_end - active_begin).count(),
					        std::memory_order_release);
					    telemetry->submit_to_ready_nanoseconds.store(
					        std::chrono::duration_cast<std::chrono::nanoseconds>(active_end - telemetry->submitted_at).count(),
					        std::memory_order_release);
					    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kCompleted);
					    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kCompleted, identity);
					    completion->set_value();
					    return batch;
				    } catch (...) {
					    if (telemetry->execution_state.stage() != PipelineTraceEvent::Stage::kCancelled) {
						    telemetry->execution_state.set_stage(PipelineTraceEvent::Stage::kFailed);
						    emit_trace(trace, *telemetry, PipelineTraceEvent::Stage::kFailed);
					    }
					    try {
						    completion->set_value();
					    } catch (const std::future_error&) {}
					    throw;
				    }
			    });
		} catch (...) {
			completion->set_value();
			throw;
		}

		return std::make_shared<Pending>(
		    std::move(future),
		    std::move(telemetry),
		    state_copy->output_slots,
		    std::move(output_slot_owner),
		    trace_);
	}

	static void emit_trace(NativePipelineTraceBuffer*      trace,
	                       const PendingTelemetry&         telemetry,
	                       const PipelineTraceEvent::Stage stage,
	                       const NativePlanTraceIdentity   identity = {}) noexcept {
		if (trace == nullptr) {
			return;
		}
		try {
			PipelineTraceEvent event;
			event.request_identity   = telemetry.request_identity;
			event.request_ordinal    = telemetry.request_ordinal;
			event.batch_ordinal      = telemetry.batch_ordinal;
			event.stage              = stage;
			event.plan_identity_hash = identity.plan_identity_hash;
			event.io_identity_hash   = identity.io_identity_hash;
			trace->emit(event);
		} catch (...) {
			// Explicit diagnostics are non-authoritative and must not affect execution.
		}
	}

	void emit_trace(const QueuedRequest& request, const PipelineTraceEvent::Stage stage) const noexcept {
		if (trace_ == nullptr) {
			return;
		}
		try {
			PipelineTraceEvent event;
			event.request_identity = request.request_identity;
			event.request_ordinal  = request.request_ordinal;
			event.batch_ordinal    = request.batch_ordinal;
			event.stage            = stage;
			trace_->emit(event);
		} catch (...) {}
	}

	size_t close_impl(const NativePipelineState::Lifecycle terminal) noexcept {
		if (pipeline_state_.closed && pending_.empty() && requests_.empty() && !last_prefetch_) {
			return 0U;
		}
		size_t cancelled = 0U;
		for (auto& pending : pending_) {
			cancelled += pending && pending->cancel() ? 1U : 0U;
		}
		pending_.clear();
		requests_.clear();
		last_prefetch_.reset();
		last_output_slot_owner_.reset();
		pipeline_state_.request_count           = 0U;
		pipeline_state_.next_request            = 0U;
		pipeline_state_.pending_count           = 0U;
		pipeline_state_.cancelled_request_count = cancelled;
		pipeline_state_.closed                  = true;
		pipeline_state_.lifecycle               = terminal;
		if (trace_ != nullptr) {
			try {
				PipelineTraceEvent event;
				event.stage = PipelineTraceEvent::Stage::kClosed;
				trace_->emit(event);
			} catch (...) {}
		}
		return cancelled;
	}

	std::shared_ptr<SharedState>         shared_state_;
	std::string                          semantic_profile_id_;
	jpeg::JpegDctDeviceBatchOptions      options_;
	NativePipelineTraceBuffer*           trace_ = nullptr;
	std::vector<QueuedRequest>           requests_;
	std::deque<std::shared_ptr<Pending>> pending_;
	std::shared_ptr<Pending>             last_prefetch_;
	std::shared_ptr<void>                last_output_slot_owner_;
	NativePipelinePrefetchMetrics        last_prefetch_metrics_;
	NativePipelineState                  pipeline_state_;
};

} // namespace galp::direct_dct::detail

#endif // GALP_DIRECT_DCT_NATIVE_LOGICAL_BATCH_PIPELINE_DETAIL_HPP
