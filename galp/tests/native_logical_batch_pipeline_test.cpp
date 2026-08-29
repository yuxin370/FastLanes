#include "direct_dct/native_logical_batch_pipeline_detail.hpp"
#include "direct_dct/profile_registry.hpp"
#include "direct_dct/resolved_execution_policy.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using galp::direct_dct::LogicalBatchRequest;
using galp::direct_dct::NativePipelineRequestState;
using galp::direct_dct::NativePipelineState;
using galp::direct_dct::NativePipelineTraceBuffer;
using galp::direct_dct::PipelineTraceEvent;

constexpr std::string_view kProfileId           = "rgbnomore-validation-v1";
constexpr size_t           kLegacyPipelineDepth = 2U;

struct FakeRecord final {
	std::string operation;
	uint32_t    image_id = 0U;
};

struct FakeRuntimeControl final {
	void record(std::string operation, const uint32_t image_id) {
		std::lock_guard lock(mutex);
		records.push_back({std::move(operation), image_id});
	}

	void on_prepare(const uint32_t image_id) {
		std::unique_lock lock(mutex);
		records.push_back({"prepare", image_id});
		++prepare_entries;
		condition.notify_all();
		condition.wait(lock, [this] { return !hold_prepare || release_prepare; });
		if (failing_image_id.has_value() && *failing_image_id == image_id) {
			throw std::runtime_error("injected legacy producer failure");
		}
	}

	[[nodiscard]] bool wait_for_prepare_entries(const size_t expected) {
		std::unique_lock lock(mutex);
		return condition.wait_for(
		    lock, std::chrono::seconds {5}, [this, expected] { return prepare_entries >= expected; });
	}

	void unblock_prepare() {
		std::lock_guard lock(mutex);
		release_prepare = true;
		condition.notify_all();
	}

	[[nodiscard]] std::vector<uint32_t> values(const std::string_view operation) const {
		std::lock_guard       lock(mutex);
		std::vector<uint32_t> result;
		for (const auto& record : records) {
			if (record.operation == operation) {
				result.push_back(record.image_id);
			}
		}
		return result;
	}

	[[nodiscard]] size_t count(const std::string_view operation, const uint32_t image_id) const {
		std::lock_guard lock(mutex);
		return static_cast<size_t>(std::count_if(records.begin(), records.end(), [&](const auto& record) {
			return record.operation == operation && record.image_id == image_id;
		}));
	}

	mutable std::mutex      mutex;
	std::condition_variable condition;
	std::vector<FakeRecord> records;
	std::optional<uint32_t> failing_image_id;
	size_t                  prepare_entries = 0U;
	bool                    hold_prepare    = false;
	bool                    release_prepare = true;
};

struct FakePreparedBatch final {
	uint32_t                                         image_id = 0U;
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	galp::jpeg::JpegDctDeviceBatchOptions            options;
};

struct FakeBatch final {
	uint32_t              first_image_id = 0U;
	std::vector<uint32_t> image_ids;
};

class FakeRuntime final {
public:
	using PreparedBatch = FakePreparedBatch;
	using Batch         = FakeBatch;

	explicit FakeRuntime(std::shared_ptr<FakeRuntimeControl> control)
	    : control_(std::move(control)) {
	}

	[[nodiscard]] int capture_device() const noexcept {
		return 0;
	}

	void activate_device(int) noexcept {
	}

	[[nodiscard]] galp::direct_dct::detail::NativePlanTraceIdentity
	trace_identity(const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests,
	               const galp::jpeg::JpegDctDeviceBatchOptions&) {
		const auto image_id = first_image(requests);
		control_->record("plan", image_id);
		return {
		    0x100000000ULL + image_id,
		    0x200000000ULL + image_id,
		};
	}

	[[nodiscard]] PreparedBatch prepare(const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests,
	                                    const galp::jpeg::JpegDctDeviceBatchOptions&            options) {
		const auto image_id = first_image(requests);
		control_->on_prepare(image_id);
		return {image_id, requests, options};
	}

	void stage(PreparedBatch& prepared) {
		control_->record("stage", prepared.image_id);
	}

	[[nodiscard]] Batch read(PreparedBatch prepared) {
		// DirectDctRuntime::ReadPreparedBatch stages a second time in legacy.
		stage(prepared);
		if (prepared.options.transform_submission_gate) {
			prepared.options.transform_submission_gate->wait();
		}
		control_->record("read", prepared.image_id);
		Batch batch;
		batch.first_image_id = prepared.image_id;
		for (const auto& request : prepared.requests) {
			batch.image_ids.push_back(request.global_image_index);
		}
		return batch;
	}

	[[nodiscard]] size_t materialized_output_bytes(const Batch& batch) const noexcept {
		return batch.image_ids.size() * sizeof(uint32_t);
	}

private:
	static uint32_t first_image(const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests) {
		if (requests.empty()) {
			throw std::invalid_argument("fake runtime received an empty request");
		}
		return requests.front().global_image_index;
	}

	std::shared_ptr<FakeRuntimeControl> control_;
};

using FakePipeline = galp::direct_dct::detail::NativeLogicalBatchPipelineCore<FakeRuntime>;

galp::jpeg::JpegDctDeviceBatchOptions shadow_options() {
	const auto semantic = galp::direct_dct::SemanticProfileRegistry::resolve(kProfileId);
	const auto policy   = galp::direct_dct::resolve_execution_policy(kProfileId);
	return galp::direct_dct::materialize_shadow_options(semantic, policy);
}

LogicalBatchRequest make_request(const uint32_t first_image_id,
                                 const uint64_t request_ordinal,
                                 const size_t   sample_count       = 1U,
                                 const size_t   logical_batch_size = 1U) {
	LogicalBatchRequest request;
	request.request_identity    = 1000U + request_ordinal;
	request.batch_ordinal       = 50U + request_ordinal;
	request.semantic_profile_id = std::string(kProfileId);
	request.logical_batch_size  = logical_batch_size;
	request.partial_tail        = sample_count < logical_batch_size;
	for (size_t index = 0U; index < sample_count; ++index) {
		LogicalBatchRequest::Sample sample;
		sample.image_id = first_image_id + static_cast<uint32_t>(index);
		request.samples.push_back(std::move(sample));
	}
	return request;
}

std::vector<PipelineTraceEvent> events_for(const std::vector<PipelineTraceEvent>& events,
                                           const PipelineTraceEvent::Stage        stage) {
	std::vector<PipelineTraceEvent> selected;
	for (const auto& event : events) {
		if (event.stage == stage) {
			selected.push_back(event);
		}
	}
	return selected;
}

size_t event_index(const std::vector<PipelineTraceEvent>& events,
                   const uint64_t                         request_ordinal,
                   const PipelineTraceEvent::Stage        stage) {
	for (size_t index = 0U; index < events.size(); ++index) {
		if (events[index].request_ordinal == request_ordinal && events[index].stage == stage) {
			return index;
		}
	}
	throw std::runtime_error("expected trace event was not emitted");
}

template <typename Predicate>
bool wait_until(Predicate&& predicate) {
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds {5};
	while (!predicate()) {
		if (std::chrono::steady_clock::now() >= deadline) {
			return false;
		}
		std::this_thread::yield();
	}
	return true;
}

TEST(NativePipelineRequestStateContract, QueuedCancellationAndStartedTransitionMatchLegacyAtomicState) {
	NativePipelineRequestState queued;
	EXPECT_FALSE(queued.started());
	EXPECT_TRUE(queued.try_cancel());
	EXPECT_FALSE(queued.try_cancel());
	EXPECT_EQ(queued.stage(), PipelineTraceEvent::Stage::kCancelled);
	EXPECT_FALSE(queued.try_start());
	EXPECT_TRUE(queued.started());

	NativePipelineRequestState active;
	EXPECT_TRUE(active.try_start());
	EXPECT_TRUE(active.started());
	EXPECT_EQ(active.stage(), PipelineTraceEvent::Stage::kPreparing);
	EXPECT_FALSE(active.try_cancel());
	active.set_stage(PipelineTraceEvent::Stage::kStaged);
	EXPECT_EQ(active.stage(), PipelineTraceEvent::Stage::kStaged);
}

TEST(NativeLogicalBatchPipelineState, InitialStartOneBatchDrainAndCloseMatchLegacy) {
	auto         control = std::make_shared<FakeRuntimeControl>();
	FakePipeline pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options());
	const auto   initial = pipeline.state();
	EXPECT_EQ(initial.lifecycle, NativePipelineState::Lifecycle::kIdle);
	EXPECT_TRUE(initial.closed);

	pipeline.reset({make_request(100U, 0U)});
	const auto started = pipeline.state();
	EXPECT_EQ(started.lifecycle, NativePipelineState::Lifecycle::kRunning);
	EXPECT_EQ(started.request_count, 1U);
	EXPECT_EQ(started.pending_count, 1U);
	EXPECT_EQ(started.max_pending_count, 1U);
	EXPECT_EQ(pipeline.prefetched_batch_count(), 1U);

	const auto batch = pipeline.next();
	EXPECT_EQ(batch.image_ids, std::vector<uint32_t>({100U}));
	const auto drained = pipeline.state();
	EXPECT_EQ(drained.lifecycle, NativePipelineState::Lifecycle::kDrained);
	EXPECT_EQ(drained.completed_request_count, 1U);
	EXPECT_EQ(drained.pending_count, 0U);
	EXPECT_THROW(static_cast<void>(pipeline.next()), std::out_of_range);
	EXPECT_EQ(pipeline.close(), 0U);
	EXPECT_EQ(pipeline.close(), 0U);
	EXPECT_EQ(pipeline.state().lifecycle, NativePipelineState::Lifecycle::kClosed);
}

TEST(NativeLogicalBatchPipelineState, TwoBatchesFillExactlyTheFrozenLegacyDepth) {
	auto         control = std::make_shared<FakeRuntimeControl>();
	FakePipeline pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options());
	pipeline.reset({make_request(100U, 0U), make_request(101U, 1U)});
	EXPECT_EQ(pipeline.state().pending_count, kLegacyPipelineDepth);
	EXPECT_EQ(pipeline.state().max_pending_count, kLegacyPipelineDepth);
	EXPECT_EQ(pipeline.prefetched_batch_count(), 2U);
	EXPECT_EQ(pipeline.next().first_image_id, 100U);
	EXPECT_EQ(pipeline.next().first_image_id, 101U);
	EXPECT_EQ(control->values("read"), std::vector<uint32_t>({100U, 101U}));
}

TEST(NativeLogicalBatchPipelineTrace, MoreThanDepthPreservesLegacyBackpressureAndSubmissionOrder) {
	auto                      control = std::make_shared<FakeRuntimeControl>();
	NativePipelineTraceBuffer trace;
	FakePipeline              pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options(), &trace);
	pipeline.reset({make_request(100U, 0U), make_request(101U, 1U), make_request(102U, 2U)});
	EXPECT_EQ(pipeline.state().pending_count, kLegacyPipelineDepth);
	EXPECT_EQ(pipeline.prefetched_batch_count(), kLegacyPipelineDepth);

	EXPECT_EQ(pipeline.next().first_image_id, 100U);
	EXPECT_EQ(pipeline.state().pending_count, kLegacyPipelineDepth);
	EXPECT_EQ(pipeline.prefetched_batch_count(), 3U);
	EXPECT_EQ(pipeline.next().first_image_id, 101U);
	EXPECT_EQ(pipeline.next().first_image_id, 102U);
	EXPECT_EQ(pipeline.state().max_pending_count, kLegacyPipelineDepth);
	EXPECT_EQ(control->values("read"), std::vector<uint32_t>({100U, 101U, 102U}));

	const auto events   = trace.snapshot();
	const auto accepted = events_for(events, PipelineTraceEvent::Stage::kRequestAccepted);
	ASSERT_EQ(accepted.size(), 3U);
	for (size_t index = 0U; index < accepted.size(); ++index) {
		EXPECT_EQ(accepted[index].request_ordinal, index);
		EXPECT_EQ(accepted[index].request_identity, 1000U + index);
		EXPECT_EQ(accepted[index].batch_ordinal, 50U + index);
	}

	for (uint64_t request = 0U; request < 3U; ++request) {
		EXPECT_LT(event_index(events, request, PipelineTraceEvent::Stage::kPreparing),
		          event_index(events, request, PipelineTraceEvent::Stage::kPlanReady));
		EXPECT_LT(event_index(events, request, PipelineTraceEvent::Stage::kPlanReady),
		          event_index(events, request, PipelineTraceEvent::Stage::kStaged));
		EXPECT_LT(event_index(events, request, PipelineTraceEvent::Stage::kStaged),
		          event_index(events, request, PipelineTraceEvent::Stage::kAwaitingPredecessor));
		EXPECT_LT(event_index(events, request, PipelineTraceEvent::Stage::kAwaitingPredecessor),
		          event_index(events, request, PipelineTraceEvent::Stage::kAwaitingOutputSlot));
		EXPECT_LT(event_index(events, request, PipelineTraceEvent::Stage::kAwaitingOutputSlot),
		          event_index(events, request, PipelineTraceEvent::Stage::kOutputSlotAcquired));
		EXPECT_LT(event_index(events, request, PipelineTraceEvent::Stage::kOutputSlotAcquired),
		          event_index(events, request, PipelineTraceEvent::Stage::kReadStarted));
		EXPECT_LT(event_index(events, request, PipelineTraceEvent::Stage::kSubmitted),
		          event_index(events, request, PipelineTraceEvent::Stage::kCompleted));
	}

	const auto submitted = events_for(events, PipelineTraceEvent::Stage::kSubmitted);
	const auto completed = events_for(events, PipelineTraceEvent::Stage::kCompleted);
	ASSERT_EQ(submitted.size(), 3U);
	ASSERT_EQ(completed.size(), 3U);
	for (size_t index = 0U; index < 3U; ++index) {
		EXPECT_EQ(submitted[index].request_ordinal, index);
		EXPECT_EQ(submitted[index].submission_ordinal, index + 1U);
		EXPECT_EQ(completed[index].request_ordinal, index);
		EXPECT_EQ(completed[index].completion_ordinal, index + 1U);
		EXPECT_EQ(submitted[index].plan_identity_hash, 0x100000000ULL + 100U + index);
		EXPECT_EQ(submitted[index].io_identity_hash, 0x200000000ULL + 100U + index);
		EXPECT_EQ(control->count("stage", static_cast<uint32_t>(100U + index)), 2U);
	}
}

TEST(NativeLogicalBatchPipelineOutputSlots, TwoSlotsOverlapAndThirdWaitsForSafeReuse) {
	auto                      control = std::make_shared<FakeRuntimeControl>();
	NativePipelineTraceBuffer trace;
	FakePipeline pipeline(
	    FakeRuntime(control), std::string(kProfileId), shadow_options(), &trace, 2U);
	pipeline.reset({make_request(100U, 0U), make_request(101U, 1U), make_request(102U, 2U)});

	const auto first = pipeline.next();
	auto first_slot = pipeline.take_delivered_output_slot_owner();
	ASSERT_TRUE(first_slot);
	// Holding first_slot models an incomplete Batch-N consumer. Batch N+1 must
	// nevertheless submit using the independent second slot.
	ASSERT_TRUE(wait_until([&] { return control->count("read", 101U) == 1U; }));
	EXPECT_EQ(first.first_image_id, 100U);
	EXPECT_EQ(pipeline.state().live_output_slots, 2U);
	EXPECT_EQ(pipeline.state().peak_live_output_slots, 2U);

	const auto second = pipeline.next();
	auto second_slot = pipeline.take_delivered_output_slot_owner();
	ASSERT_TRUE(second_slot);
	ASSERT_TRUE(wait_until([&] { return pipeline.state().output_slot_waiters == 1U; }));
	EXPECT_EQ(second.first_image_id, 101U);
	EXPECT_EQ(control->count("read", 102U), 0U);
	EXPECT_EQ(pipeline.state().live_output_slots, 2U);

	first_slot.reset();
	ASSERT_TRUE(wait_until([&] { return control->count("read", 102U) == 1U; }));
	const auto third = pipeline.next();
	auto third_slot = pipeline.take_delivered_output_slot_owner();
	ASSERT_TRUE(third_slot);
	EXPECT_EQ(third.first_image_id, 102U);
	EXPECT_LE(pipeline.state().peak_live_output_slots, 2U);
	EXPECT_EQ(pipeline.state().maximum_output_slot_bytes, sizeof(uint32_t));
	EXPECT_EQ(pipeline.state().peak_output_bytes, 2U * sizeof(uint32_t));
}

TEST(NativeLogicalBatchPipelineState, PartialTailAndSemanticTransformsLowerWithoutReinterpretation) {
	auto         control = std::make_shared<FakeRuntimeControl>();
	FakePipeline pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options());
	auto         request                           = make_request(200U, 0U, 2U, 4U);
	request.samples[0].transform.source_crop       = galp::jpeg::JpegDctCropBox {8U, 16U, 224U, 224U};
	request.samples[0].transform.horizontal_flip   = true;
	request.samples[0].transform.logical_sample_id = "sample-200";
	request.samples[0].transform.augmentation_key  = "crop-flip";
	pipeline.reset({std::move(request)});
	const auto batch = pipeline.next();
	EXPECT_EQ(batch.image_ids, std::vector<uint32_t>({200U, 201U}));
	EXPECT_EQ(control->values("read"), std::vector<uint32_t>({200U}));
}

TEST(NativeLogicalBatchPipelineState, EmptyInvalidAndMismatchedRequestsDoNotReplaceActiveLegacyState) {
	auto         control = std::make_shared<FakeRuntimeControl>();
	FakePipeline pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options());
	EXPECT_THROW(static_cast<void>(pipeline.next()), std::out_of_range);

	pipeline.reset({make_request(100U, 0U)});
	LogicalBatchRequest empty;
	empty.semantic_profile_id = std::string(kProfileId);
	empty.logical_batch_size  = 1U;
	EXPECT_THROW(pipeline.reset({empty}), std::invalid_argument);
	EXPECT_EQ(pipeline.state().pending_count, 1U);
	EXPECT_EQ(pipeline.next().first_image_id, 100U);

	auto mismatched                = make_request(101U, 1U);
	mismatched.semantic_profile_id = "rgbnomore-validation-center-crop-512-v1";
	EXPECT_THROW(pipeline.reset({std::move(mismatched)}), std::invalid_argument);

	pipeline.reset({});
	EXPECT_EQ(pipeline.state().lifecycle, NativePipelineState::Lifecycle::kDrained);
	EXPECT_FALSE(pipeline.state().closed);
	EXPECT_THROW(static_cast<void>(pipeline.next()), std::out_of_range);
}

TEST(NativeLogicalBatchPipelineState, ResetAfterDrainStartsASeparateLegacySequence) {
	auto         control = std::make_shared<FakeRuntimeControl>();
	FakePipeline pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options());
	pipeline.reset({make_request(100U, 0U)});
	EXPECT_EQ(pipeline.next().first_image_id, 100U);
	pipeline.reset({make_request(300U, 0U), make_request(301U, 1U)});
	EXPECT_EQ(pipeline.next().first_image_id, 300U);
	EXPECT_EQ(pipeline.next().first_image_id, 301U);
	EXPECT_EQ(control->values("read"), std::vector<uint32_t>({100U, 300U, 301U}));
}

TEST(NativeLogicalBatchPipelineState, CloseWhileActiveReleasesGatesWaitsAndIsIdempotent) {
	auto control             = std::make_shared<FakeRuntimeControl>();
	control->hold_prepare    = true;
	control->release_prepare = false;
	NativePipelineTraceBuffer trace;
	FakePipeline              pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options(), &trace);
	pipeline.reset({make_request(100U, 0U), make_request(101U, 1U)});
	ASSERT_TRUE(control->wait_for_prepare_entries(2U));
	auto close_future = std::async(std::launch::async, [&pipeline] { return pipeline.close(); });
	control->unblock_prepare();
	EXPECT_EQ(close_future.get(), 0U);
	EXPECT_EQ(pipeline.close(), 0U);
	EXPECT_TRUE(pipeline.state().closed);
	EXPECT_EQ(pipeline.state().lifecycle, NativePipelineState::Lifecycle::kClosed);
	const auto events = trace.snapshot();
	ASSERT_FALSE(events.empty());
	EXPECT_EQ(events.back().stage, PipelineTraceEvent::Stage::kClosed);
}

TEST(NativeLogicalBatchPipelineState, ProducerExceptionPropagatesAndClosesLikeLegacy) {
	auto control              = std::make_shared<FakeRuntimeControl>();
	control->failing_image_id = 101U;
	NativePipelineTraceBuffer trace;
	FakePipeline              pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options(), &trace);
	pipeline.reset({make_request(100U, 0U), make_request(101U, 1U)});
	EXPECT_EQ(pipeline.next().first_image_id, 100U);
	EXPECT_THROW(static_cast<void>(pipeline.next()), std::runtime_error);
	EXPECT_TRUE(pipeline.state().closed);
	EXPECT_EQ(pipeline.state().lifecycle, NativePipelineState::Lifecycle::kFailed);
	const auto failed = events_for(trace.snapshot(), PipelineTraceEvent::Stage::kFailed);
	ASSERT_EQ(failed.size(), 1U);
	EXPECT_EQ(failed.front().request_ordinal, 1U);
}

TEST(NativeLogicalBatchPipelineState, ConsumerStopsEarlyDestructorUsesLegacyCloseSemantics) {
	auto                      control = std::make_shared<FakeRuntimeControl>();
	NativePipelineTraceBuffer trace;
	{
		FakePipeline pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options(), &trace);
		pipeline.reset({make_request(100U, 0U), make_request(101U, 1U), make_request(102U, 2U)});
		EXPECT_EQ(pipeline.next().first_image_id, 100U);
	}
	const auto events = trace.snapshot();
	ASSERT_FALSE(events.empty());
	EXPECT_EQ(events.back().stage, PipelineTraceEvent::Stage::kClosed);
	EXPECT_EQ(control->values("read").front(), 100U);
}

TEST(NativeLogicalBatchPipelineTrace, DisabledTracePerformsNoPlanPreviewOrTraceWork) {
	auto         control = std::make_shared<FakeRuntimeControl>();
	FakePipeline pipeline(FakeRuntime(control), std::string(kProfileId), shadow_options(), nullptr);
	pipeline.reset({make_request(100U, 0U), make_request(101U, 1U)});
	EXPECT_EQ(pipeline.next().first_image_id, 100U);
	EXPECT_EQ(pipeline.next().first_image_id, 101U);
	EXPECT_TRUE(control->values("plan").empty());
}

} // namespace
