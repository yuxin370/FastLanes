#ifndef GALP_DIRECT_DCT_NATIVE_BATCH_LIFETIME_HPP
#define GALP_DIRECT_DCT_NATIVE_BATCH_LIFETIME_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace galp::jpeg {
class DirectDctBatch;
}

namespace galp::direct_dct {

// ATen-free identity forwarded by an adapter that observes a real consumer
// stream. A zero stream identity is valid and denotes CUDA's default stream.
struct ConsumerDependency final {
	enum class Source : uint8_t {
		kGetterCompatibility,
		kExplicitConsumer,
	};

	int       cuda_device    = -1;
	uintptr_t stream_identity = 0U;
	Source    source           = Source::kGetterCompatibility;
	uint64_t  registration_ordinal = 0U;
	bool      complete             = false;
};

struct NativeBatchCompletionSnapshot final {
	uint64_t  batch_identity                    = 0U;
	uintptr_t producer_completion_event_identity = 0U;
	bool      producer_complete                 = false;
	bool      release_requested                 = false;
	size_t    consumer_dependency_count         = 0U;
	size_t    explicit_consumer_dependency_count = 0U;
	size_t    pending_consumer_dependency_count = 0U;
	size_t    consumer_completion_event_count   = 0U;
	bool      consumer_completion_events_recorded = false;
	bool      reclaim_eligible                  = false;
};

// Canonical native completion state. In authoritative mode it reuses the
// producer event owned by DirectDctBatch and owns only the consumer events
// needed to defer reclaim. Shadow users may still drive the same state by
// explicitly marking dependencies complete.
class NativeBatchCompletion final {
public:
	explicit NativeBatchCompletion(uint64_t batch_identity,
	                               uintptr_t producer_completion_event_identity = 0U,
	                               int producer_cuda_device = -1);
	~NativeBatchCompletion();

	// Returns true for a new stream dependency. Re-registering the same stream
	// is allocation-free; explicit registration upgrades getter compatibility.
	bool register_consumer(ConsumerDependency dependency);
	void mark_producer_complete() noexcept;
	void mark_consumer_complete(int cuda_device, uintptr_t stream_identity);
	void request_release() noexcept;
	void record_consumer_completion_events();
	void poll_cuda_dependencies();
	void wait_cuda_dependencies();

	[[nodiscard]] NativeBatchCompletionSnapshot snapshot() const;
	[[nodiscard]] std::vector<ConsumerDependency> consumer_dependencies() const;

private:
	struct CudaState;

	mutable std::mutex              mutex_;
	uint64_t                        batch_identity_ = 0U;
	uintptr_t                       producer_completion_event_identity_ = 0U;
	int                             producer_cuda_device_ = -1;
	std::vector<ConsumerDependency> consumers_;
	uint64_t                        next_registration_ordinal_ = 1U;
	bool                            producer_complete_ = false;
	bool                            release_requested_ = false;
	std::unique_ptr<CudaState>      cuda_state_;
};

enum class NativeEligibilityDifferential : uint8_t {
	kBothBlocked,
	kEquivalentEligible,
	kNativeSafer,
	kNativeEarlierUnsafe,
};

struct NativeBatchLeaseSnapshot final {
	NativeBatchCompletionSnapshot completion;
	size_t                        storage_reference_count = 0U;
	size_t                        released_storage_reference_count = 0U;
	bool                          all_storage_references_released = true;
	bool                          legacy_reclaim_eligible = false;
	size_t                        owner_reference_count = 0U;
	size_t                        released_owner_reference_count = 0U;
	bool                          authoritative = false;
	bool                          backing_storage_present = false;
	bool                          reclaim_executed = false;
	NativeEligibilityDifferential differential = NativeEligibilityDifferential::kBothBlocked;
};

struct NativeBatchReclaimQueueStats final {
	size_t pending_reclaim_count = 0U;
	size_t pending_reclaim_peak = 0U;
	size_t live_batch_count = 0U;
	size_t live_batch_peak = 0U;
	size_t enqueued_batch_count = 0U;
	size_t reclaimed_batch_count = 0U;
	size_t consumer_event_count = 0U;
};

// Canonical native lifetime owner. The one-argument constructor preserves the
// non-authoritative shadow/test form; the two-argument constructor transfers
// actual DirectDctBatch ownership to the lease.
class NativeBatchLease final : public std::enable_shared_from_this<NativeBatchLease> {
public:
	explicit NativeBatchLease(std::shared_ptr<NativeBatchCompletion> completion);
	NativeBatchLease(std::shared_ptr<NativeBatchCompletion> completion,
	                 std::shared_ptr<jpeg::DirectDctBatch> backing_batch);

	bool register_consumer(ConsumerDependency dependency);
	void mark_producer_complete() noexcept;
	void mark_consumer_complete(int cuda_device, uintptr_t stream_identity);

	// A reference represents one independent Tensor/Storage deleter that owns
	// the native batch. The shadow requests final release only after every
	// retained storage reference has executed its deleter.
	void retain_storage_reference();
	[[nodiscard]] bool release_storage_reference() noexcept;
	void retain_owner_reference();
	[[nodiscard]] bool release_owner_reference() noexcept;
	void request_release() noexcept;
	void observe_legacy_reclaim_eligibility(bool eligible) noexcept;

	[[nodiscard]] NativeBatchLeaseSnapshot snapshot() const;
	[[nodiscard]] jpeg::DirectDctBatch* backing_batch() const noexcept;
	[[nodiscard]] bool authoritative() const noexcept;

	static size_t reclaim_finished() noexcept;
	static NativeBatchReclaimQueueStats reclaim_queue_stats() noexcept;

private:
	friend class NativeBatchReclaimQueue;
	[[nodiscard]] bool all_references_released_locked() const noexcept;
	void request_release_if_final() noexcept;
	[[nodiscard]] bool try_reclaim();
	void wait_and_reclaim_noexcept() noexcept;

	std::shared_ptr<NativeBatchCompletion> completion_;
	mutable std::mutex                    mutex_;
	std::shared_ptr<jpeg::DirectDctBatch> backing_batch_;
	size_t                                storage_reference_count_ = 0U;
	size_t                                released_storage_reference_count_ = 0U;
	size_t                                owner_reference_count_ = 0U;
	size_t                                released_owner_reference_count_ = 0U;
	bool                                  release_requested_ = false;
	bool                                  legacy_reclaim_eligible_ = false;
	bool                                  reclaim_enqueued_ = false;
	bool                                  reclaim_executed_ = false;
};

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_NATIVE_BATCH_LIFETIME_HPP
