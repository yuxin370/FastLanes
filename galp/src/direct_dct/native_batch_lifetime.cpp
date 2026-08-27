#include "direct_dct/native_batch_lifetime.hpp"
#include "galp/direct_dct.hpp"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <deque>
#include <stdexcept>
#include <utility>

namespace galp::direct_dct {
namespace {

void check_cuda(const cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
	}
}

class ScopedCudaDevice final {
public:
	explicit ScopedCudaDevice(const int cuda_device) {
		if (cuda_device < 0) {
			return;
		}
		check_cuda(cudaGetDevice(&previous_), "cudaGetDevice(native lifetime)");
		if (previous_ != cuda_device) {
			check_cuda(cudaSetDevice(cuda_device), "cudaSetDevice(native lifetime)");
			restore_ = true;
		}
	}

	~ScopedCudaDevice() {
		if (restore_) {
			static_cast<void>(cudaSetDevice(previous_));
		}

	}

	ScopedCudaDevice(const ScopedCudaDevice&) = delete;
	ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

private:
	int  previous_ = -1;
	bool restore_ = false;
};

} // namespace

struct NativeBatchCompletion::CudaState final {
	struct ConsumerEvent final {
		int         cuda_device = -1;
		uintptr_t   stream_identity = 0U;
		cudaEvent_t event = nullptr;
	};

	~CudaState() {
		for (auto& consumer : consumer_events) {
			if (consumer.event != nullptr) {
				int previous = -1;
				const bool restore = cudaGetDevice(&previous) == cudaSuccess &&
				                     previous != consumer.cuda_device;
				static_cast<void>(cudaSetDevice(consumer.cuda_device));
				static_cast<void>(cudaEventDestroy(consumer.event));
				if (restore) {
					static_cast<void>(cudaSetDevice(previous));
				}
			}
		}
	}

	std::vector<ConsumerEvent> consumer_events;
	bool                       events_recorded = false;
};

class NativeBatchReclaimQueue final {
public:
	static NativeBatchReclaimQueue& instance() {
		static NativeBatchReclaimQueue queue;
		return queue;
	}

	void enqueue(std::shared_ptr<NativeBatchLease> lease) {
		if (!lease) {
			return;
		}
		std::lock_guard lock(mutex_);
		pending_.push_back(std::move(lease));
		++enqueued_batch_count_;
		pending_reclaim_peak_ = std::max(pending_reclaim_peak_, pending_.size());
		live_batch_peak_ = std::max(live_batch_peak_, pending_.size());
	}

	void note_consumer_events(const size_t count) noexcept {
		std::lock_guard lock(mutex_);
		consumer_event_count_ += count;
	}

	size_t reclaim_finished() noexcept {
		size_t reclaimed = 0U;
		try {
			std::lock_guard lock(mutex_);
			for (auto it = pending_.begin(); it != pending_.end();) {
				if ((*it)->try_reclaim()) {
					it = pending_.erase(it);
					++reclaimed;
					++reclaimed_batch_count_;
				} else {
					++it;
				}
			}
		} catch (const std::exception& error) {
			std::fprintf(stderr, "GALP native batch reclaim query failed: %s\n", error.what());
		} catch (...) {
			std::fprintf(stderr, "GALP native batch reclaim query failed.\n");
		}
		return reclaimed;
	}

	NativeBatchReclaimQueueStats stats() noexcept {
		std::lock_guard lock(mutex_);
		NativeBatchReclaimQueueStats out;
		out.pending_reclaim_count = pending_.size();
		out.pending_reclaim_peak = pending_reclaim_peak_;
		out.live_batch_count = pending_.size();
		out.live_batch_peak = live_batch_peak_;
		out.enqueued_batch_count = enqueued_batch_count_;
		out.reclaimed_batch_count = reclaimed_batch_count_;
		out.consumer_event_count = consumer_event_count_;
		return out;
	}

	~NativeBatchReclaimQueue() {
		std::deque<std::shared_ptr<NativeBatchLease>> pending;
		{
			std::lock_guard lock(mutex_);
			pending.swap(pending_);
		}
		for (auto& lease : pending) {
			lease->wait_and_reclaim_noexcept();
		}
	}

private:
	std::mutex                                  mutex_;
	std::deque<std::shared_ptr<NativeBatchLease>> pending_;
	size_t pending_reclaim_peak_ = 0U;
	size_t live_batch_peak_ = 0U;
	size_t enqueued_batch_count_ = 0U;
	size_t reclaimed_batch_count_ = 0U;
	size_t consumer_event_count_ = 0U;
};

NativeBatchCompletion::NativeBatchCompletion(
    const uint64_t batch_identity,
    const uintptr_t producer_completion_event_identity,
    const int producer_cuda_device)
    : batch_identity_(batch_identity)
    , producer_completion_event_identity_(producer_completion_event_identity)
	, producer_cuda_device_(producer_cuda_device)
    , cuda_state_(std::make_unique<CudaState>()) {
}

NativeBatchCompletion::~NativeBatchCompletion() = default;

bool NativeBatchCompletion::register_consumer(ConsumerDependency dependency) {
	if (dependency.cuda_device < 0) {
		throw std::invalid_argument("consumer dependency requires a non-negative CUDA device");
	}
	std::lock_guard lock(mutex_);
	if (release_requested_) {
		throw std::logic_error("consumer dependency cannot be registered after release was requested");
	}
	const auto existing = std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& value) {
		return value.cuda_device == dependency.cuda_device &&
		       value.stream_identity == dependency.stream_identity;
	});
	if (existing != consumers_.end()) {
		if (dependency.source == ConsumerDependency::Source::kExplicitConsumer) {
			existing->source = ConsumerDependency::Source::kExplicitConsumer;
		}
		return false;
	}
	dependency.registration_ordinal = next_registration_ordinal_++;
	consumers_.push_back(dependency);
	return true;
}

void NativeBatchCompletion::mark_producer_complete() noexcept {
	std::lock_guard lock(mutex_);
	producer_complete_ = true;
}

void NativeBatchCompletion::mark_consumer_complete(
    const int cuda_device,
    const uintptr_t stream_identity) {
	std::lock_guard lock(mutex_);
	const auto consumer = std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& value) {
		return value.cuda_device == cuda_device && value.stream_identity == stream_identity;
	});
	if (consumer == consumers_.end()) {
		throw std::invalid_argument("cannot complete an unregistered consumer dependency");
	}
	consumer->complete = true;
}

void NativeBatchCompletion::request_release() noexcept {
	std::lock_guard lock(mutex_);
	release_requested_ = true;
}

void NativeBatchCompletion::record_consumer_completion_events() {
	std::lock_guard lock(mutex_);
	if (cuda_state_->events_recorded) {
		return;
	}
	std::vector<CudaState::ConsumerEvent> recorded;
	recorded.reserve(consumers_.size());
	try {
		for (const auto& consumer : consumers_) {
			if (consumer.complete) {
				continue;
			}
			ScopedCudaDevice device_guard(consumer.cuda_device);
			cudaEvent_t event = nullptr;
			check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
			           "cudaEventCreate(native consumer event)");
			try {
				check_cuda(cudaEventRecord(event, reinterpret_cast<cudaStream_t>(consumer.stream_identity)),
				           "cudaEventRecord(native consumer event)");
			} catch (...) {
				static_cast<void>(cudaEventDestroy(event));
				throw;
			}
			recorded.push_back(CudaState::ConsumerEvent {
			    consumer.cuda_device, consumer.stream_identity, event});
		}
	} catch (...) {
		for (auto& consumer : recorded) {
			ScopedCudaDevice device_guard(consumer.cuda_device);
			static_cast<void>(cudaEventDestroy(consumer.event));
		}
		throw;
	}
	cuda_state_->consumer_events = std::move(recorded);
	cuda_state_->events_recorded = true;
}

void NativeBatchCompletion::poll_cuda_dependencies() {
	std::lock_guard lock(mutex_);
	if (!producer_complete_) {
		if (producer_completion_event_identity_ == 0U) {
			producer_complete_ = true;
		} else {
			ScopedCudaDevice device_guard(producer_cuda_device_);
			const auto status = cudaEventQuery(
			    reinterpret_cast<cudaEvent_t>(producer_completion_event_identity_));
			if (status == cudaSuccess) {
				producer_complete_ = true;
			} else if (status != cudaErrorNotReady) {
				check_cuda(status, "cudaEventQuery(native producer event)");
			}
		}
	}
	for (const auto& event : cuda_state_->consumer_events) {
		const auto consumer = std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& value) {
			return value.cuda_device == event.cuda_device && value.stream_identity == event.stream_identity;
		});
		if (consumer == consumers_.end() || consumer->complete) {
			continue;
		}
		ScopedCudaDevice device_guard(event.cuda_device);
		const auto status = cudaEventQuery(event.event);
		if (status == cudaSuccess) {
			consumer->complete = true;
		} else if (status != cudaErrorNotReady) {
			check_cuda(status, "cudaEventQuery(native consumer event)");
		}
	}
}

void NativeBatchCompletion::wait_cuda_dependencies() {
	std::lock_guard lock(mutex_);
	if (!producer_complete_ && producer_completion_event_identity_ != 0U) {
		ScopedCudaDevice device_guard(producer_cuda_device_);
		check_cuda(cudaEventSynchronize(
		               reinterpret_cast<cudaEvent_t>(producer_completion_event_identity_)),
		           "cudaEventSynchronize(native producer shutdown)");
		producer_complete_ = true;
	} else if (producer_completion_event_identity_ == 0U) {
		producer_complete_ = true;
	}
	for (const auto& event : cuda_state_->consumer_events) {
		ScopedCudaDevice device_guard(event.cuda_device);
		check_cuda(cudaEventSynchronize(event.event), "cudaEventSynchronize(native consumer shutdown)");
		const auto consumer = std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& value) {
			return value.cuda_device == event.cuda_device && value.stream_identity == event.stream_identity;
		});
		if (consumer != consumers_.end()) {
			consumer->complete = true;
		}
	}
}

NativeBatchCompletionSnapshot NativeBatchCompletion::snapshot() const {
	std::lock_guard lock(mutex_);
	NativeBatchCompletionSnapshot out;
	out.batch_identity                     = batch_identity_;
	out.producer_completion_event_identity = producer_completion_event_identity_;
	out.producer_complete                  = producer_complete_;
	out.release_requested                  = release_requested_;
	out.consumer_dependency_count          = consumers_.size();
	for (const auto& consumer : consumers_) {
		out.explicit_consumer_dependency_count +=
		    consumer.source == ConsumerDependency::Source::kExplicitConsumer ? 1U : 0U;
		out.pending_consumer_dependency_count += consumer.complete ? 0U : 1U;
	}
	out.consumer_completion_event_count = cuda_state_->consumer_events.size();
	out.consumer_completion_events_recorded = cuda_state_->events_recorded;
	out.reclaim_eligible = release_requested_ && producer_complete_ &&
	                       out.pending_consumer_dependency_count == 0U;
	return out;
}

std::vector<ConsumerDependency> NativeBatchCompletion::consumer_dependencies() const {
	std::lock_guard lock(mutex_);
	return consumers_;
}

NativeBatchLease::NativeBatchLease(std::shared_ptr<NativeBatchCompletion> completion)
    : completion_(std::move(completion)) {
	if (!completion_) {
		throw std::invalid_argument("NativeBatchLease requires completion state");
	}
}

NativeBatchLease::NativeBatchLease(
    std::shared_ptr<NativeBatchCompletion> completion,
    std::shared_ptr<jpeg::DirectDctBatch> backing_batch)
    : completion_(std::move(completion))
	, backing_owner_(backing_batch)
	, backing_batch_(backing_batch.get()) {
	if (!completion_ || !backing_owner_) {
		throw std::invalid_argument("authoritative NativeBatchLease requires completion and backing batch");
	}
}

NativeBatchLease::NativeBatchLease(
    std::shared_ptr<NativeBatchCompletion> completion,
    std::shared_ptr<void> backing_owner)
    : completion_(std::move(completion))
	, backing_owner_(std::move(backing_owner)) {
	if (!completion_ || !backing_owner_) {
		throw std::invalid_argument("authoritative NativeBatchLease requires completion and backing owner");
	}
}

bool NativeBatchLease::register_consumer(ConsumerDependency dependency) {
	return completion_->register_consumer(std::move(dependency));
}

void NativeBatchLease::mark_producer_complete() noexcept {
	completion_->mark_producer_complete();
}

void NativeBatchLease::mark_consumer_complete(
    const int cuda_device,
    const uintptr_t stream_identity) {
	completion_->mark_consumer_complete(cuda_device, stream_identity);
}

void NativeBatchLease::retain_storage_reference() {
	std::lock_guard lock(mutex_);
	if (release_requested_) {
		throw std::logic_error("storage reference cannot be retained after release was requested");
	}
	++storage_reference_count_;
}

bool NativeBatchLease::release_storage_reference() noexcept {
	bool final_storage_reference = false;
	{
		std::lock_guard lock(mutex_);
		if (released_storage_reference_count_ >= storage_reference_count_) {
			return false;
		}
		++released_storage_reference_count_;
		final_storage_reference = released_storage_reference_count_ == storage_reference_count_;
	}
	request_release_if_final();
	return final_storage_reference;
}

void NativeBatchLease::retain_owner_reference() {
	std::lock_guard lock(mutex_);
	if (release_requested_) {
		throw std::logic_error("owner reference cannot be retained after release was requested");
	}
	++owner_reference_count_;
}

bool NativeBatchLease::release_owner_reference() noexcept {
	bool final_owner_reference = false;
	{
		std::lock_guard lock(mutex_);
		if (released_owner_reference_count_ >= owner_reference_count_) {
			return false;
		}
		++released_owner_reference_count_;
		final_owner_reference = released_owner_reference_count_ == owner_reference_count_;
	}
	request_release_if_final();
	return final_owner_reference;
}

bool NativeBatchLease::all_references_released_locked() const noexcept {
	return released_storage_reference_count_ == storage_reference_count_ &&
	       released_owner_reference_count_ == owner_reference_count_;
}

void NativeBatchLease::request_release_if_final() noexcept {
	bool request = false;
	{
		std::lock_guard lock(mutex_);
		if (!release_requested_ && all_references_released_locked()) {
			release_requested_ = true;
			request = true;
		}
	}
	if (request) {
		request_release();
	}
}

void NativeBatchLease::request_release() noexcept {
	bool enqueue = false;
	{
		std::lock_guard lock(mutex_);
		release_requested_ = true;
		if (backing_owner_ && !reclaim_enqueued_) {
			reclaim_enqueued_ = true;
			enqueue = true;
		}
	}
	if (!enqueue) {
		completion_->request_release();
		return;
	}
	auto self = shared_from_this();
	NativeBatchReclaimQueue::instance().enqueue(self);
	try {
		completion_->record_consumer_completion_events();
		NativeBatchReclaimQueue::instance().note_consumer_events(
		    completion_->snapshot().consumer_completion_event_count);
		completion_->request_release();
		static_cast<void>(NativeBatchReclaimQueue::instance().reclaim_finished());
	} catch (const std::exception& error) {
		std::fprintf(stderr,
		             "GALP native batch release setup failed; retaining backing storage: %s\n",
		             error.what());
	} catch (...) {
		std::fprintf(stderr,
		             "GALP native batch release setup failed; retaining backing storage.\n");
	}
}

void NativeBatchLease::observe_legacy_reclaim_eligibility(const bool eligible) noexcept {
	std::lock_guard lock(mutex_);
	legacy_reclaim_eligible_ = eligible;
}

NativeBatchLeaseSnapshot NativeBatchLease::snapshot() const {
	NativeBatchLeaseSnapshot out;
	out.completion = completion_->snapshot();
	{
		std::lock_guard lock(mutex_);
		out.storage_reference_count          = storage_reference_count_;
		out.released_storage_reference_count = released_storage_reference_count_;
		out.all_storage_references_released =
		    released_storage_reference_count_ == storage_reference_count_;
		out.legacy_reclaim_eligible = legacy_reclaim_eligible_;
		out.owner_reference_count = owner_reference_count_;
		out.released_owner_reference_count = released_owner_reference_count_;
		out.authoritative = static_cast<bool>(backing_owner_) || reclaim_enqueued_ || reclaim_executed_;
		out.backing_storage_present = static_cast<bool>(backing_owner_);
		out.reclaim_executed = reclaim_executed_;
	}
	if (out.legacy_reclaim_eligible && out.completion.reclaim_eligible) {
		out.differential = NativeEligibilityDifferential::kEquivalentEligible;
	} else if (out.legacy_reclaim_eligible) {
		out.differential = NativeEligibilityDifferential::kNativeSafer;
	} else if (out.completion.reclaim_eligible) {
		out.differential = NativeEligibilityDifferential::kNativeEarlierUnsafe;
	} else {
		out.differential = NativeEligibilityDifferential::kBothBlocked;
	}
	return out;
}

jpeg::DirectDctBatch* NativeBatchLease::backing_batch() const noexcept {
	std::lock_guard lock(mutex_);
	return backing_owner_ ? backing_batch_ : nullptr;
}

bool NativeBatchLease::authoritative() const noexcept {
	std::lock_guard lock(mutex_);
	return static_cast<bool>(backing_owner_) || reclaim_enqueued_ || reclaim_executed_;
}

bool NativeBatchLease::try_reclaim() {
	completion_->poll_cuda_dependencies();
	if (!completion_->snapshot().reclaim_eligible) {
		return false;
	}
	std::shared_ptr<void> release;
	{
		std::lock_guard lock(mutex_);
		if (!backing_owner_) {
			return reclaim_executed_;
		}
		release = std::move(backing_owner_);
		backing_batch_ = nullptr;
		reclaim_executed_ = true;
	}
	release.reset();
	return true;
}

void NativeBatchLease::wait_and_reclaim_noexcept() noexcept {
	try {
		completion_->wait_cuda_dependencies();
		completion_->request_release();
		static_cast<void>(try_reclaim());
	} catch (const std::exception& error) {
		std::fprintf(stderr, "GALP native batch shutdown reclaim failed: %s\n", error.what());
	} catch (...) {
		std::fprintf(stderr, "GALP native batch shutdown reclaim failed.\n");
	}
}

size_t NativeBatchLease::reclaim_finished() noexcept {
	return NativeBatchReclaimQueue::instance().reclaim_finished();
}

NativeBatchReclaimQueueStats NativeBatchLease::reclaim_queue_stats() noexcept {
	return NativeBatchReclaimQueue::instance().stats();
}

} // namespace galp::direct_dct
