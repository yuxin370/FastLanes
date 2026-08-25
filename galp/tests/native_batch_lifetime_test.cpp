#include "direct_dct/native_batch_lifetime.hpp"
#include "galp/direct_dct.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

namespace {

using galp::direct_dct::ConsumerDependency;
using galp::direct_dct::NativeBatchCompletion;
using galp::direct_dct::NativeBatchLease;
using galp::direct_dct::NativeEligibilityDifferential;

ConsumerDependency consumer(
    const uintptr_t stream,
    const ConsumerDependency::Source source = ConsumerDependency::Source::kExplicitConsumer) {
	ConsumerDependency dependency;
	dependency.cuda_device     = 0;
	dependency.stream_identity = stream;
	dependency.source          = source;
	return dependency;
}

TEST(NativeBatchCompletionShadow, ProducerAndConsumerCompletionAreBothRequired) {
	auto completion = std::make_shared<NativeBatchCompletion>(17U, 0xCAFEU);
	NativeBatchLease lease(completion);
	EXPECT_TRUE(completion->register_consumer(consumer(101U)));
	lease.request_release();
	lease.observe_legacy_reclaim_eligibility(false);

	auto snapshot = lease.snapshot();
	EXPECT_FALSE(snapshot.completion.reclaim_eligible);
	EXPECT_EQ(snapshot.completion.pending_consumer_dependency_count, 1U);
	EXPECT_EQ(snapshot.differential, NativeEligibilityDifferential::kBothBlocked);

	completion->mark_producer_complete();
	EXPECT_FALSE(lease.snapshot().completion.reclaim_eligible);
	completion->mark_consumer_complete(0, 101U);
	EXPECT_TRUE(lease.snapshot().completion.reclaim_eligible);
}

TEST(NativeBatchCompletionShadow, DuplicateStreamRegistrationIsDeduplicatedAndUpgraded) {
	NativeBatchCompletion completion(18U);
	EXPECT_TRUE(completion.register_consumer(
	    consumer(0U, ConsumerDependency::Source::kGetterCompatibility)));
	EXPECT_FALSE(completion.register_consumer(consumer(0U)));
	const auto dependencies = completion.consumer_dependencies();
	ASSERT_EQ(dependencies.size(), 1U);
	EXPECT_EQ(dependencies.front().source, ConsumerDependency::Source::kExplicitConsumer);
	EXPECT_EQ(dependencies.front().registration_ordinal, 1U);
	EXPECT_EQ(completion.snapshot().explicit_consumer_dependency_count, 1U);
}

TEST(NativeBatchCompletionShadow, MultipleConsumersMustAllComplete) {
	auto completion = std::make_shared<NativeBatchCompletion>(19U);
	NativeBatchLease lease(completion);
	EXPECT_TRUE(completion->register_consumer(consumer(101U)));
	EXPECT_TRUE(completion->register_consumer(consumer(202U)));
	completion->mark_producer_complete();
	lease.request_release();
	completion->mark_consumer_complete(0, 101U);
	EXPECT_FALSE(lease.snapshot().completion.reclaim_eligible);
	completion->mark_consumer_complete(0, 202U);
	EXPECT_TRUE(lease.snapshot().completion.reclaim_eligible);
}

TEST(NativeBatchCompletionShadow, EveryTensorStorageReferenceMustRelease) {
	auto completion = std::make_shared<NativeBatchCompletion>(23U);
	NativeBatchLease lease(completion);
	EXPECT_TRUE(lease.register_consumer(consumer(101U)));
	lease.retain_storage_reference();
	lease.retain_storage_reference();
	lease.mark_producer_complete();
	lease.mark_consumer_complete(0, 101U);

	EXPECT_FALSE(lease.release_storage_reference());
	auto snapshot = lease.snapshot();
	EXPECT_EQ(snapshot.storage_reference_count, 2U);
	EXPECT_EQ(snapshot.released_storage_reference_count, 1U);
	EXPECT_FALSE(snapshot.all_storage_references_released);
	EXPECT_FALSE(snapshot.completion.release_requested);
	EXPECT_FALSE(snapshot.completion.reclaim_eligible);

	EXPECT_TRUE(lease.release_storage_reference());
	snapshot = lease.snapshot();
	EXPECT_TRUE(snapshot.all_storage_references_released);
	EXPECT_TRUE(snapshot.completion.release_requested);
	EXPECT_TRUE(snapshot.completion.reclaim_eligible);
	EXPECT_FALSE(lease.release_storage_reference());
}

TEST(NativeBatchCompletionShadow, F001ExplicitConsumerMakesNativeEligibilitySafer) {
	auto completion = std::make_shared<NativeBatchCompletion>(20U);
	NativeBatchLease lease(completion);
	EXPECT_TRUE(completion->register_consumer(
	    consumer(0U, ConsumerDependency::Source::kGetterCompatibility)));
	EXPECT_TRUE(completion->register_consumer(consumer(303U)));
	completion->mark_producer_complete();
	completion->mark_consumer_complete(0, 0U);
	lease.request_release();
	lease.observe_legacy_reclaim_eligibility(true);

	auto snapshot = lease.snapshot();
	EXPECT_FALSE(snapshot.completion.reclaim_eligible);
	EXPECT_EQ(snapshot.completion.pending_consumer_dependency_count, 1U);
	EXPECT_EQ(snapshot.differential, NativeEligibilityDifferential::kNativeSafer);

	completion->mark_consumer_complete(0, 303U);
	snapshot = lease.snapshot();
	EXPECT_TRUE(snapshot.completion.reclaim_eligible);
	EXPECT_EQ(snapshot.differential, NativeEligibilityDifferential::kEquivalentEligible);
}

TEST(NativeBatchCompletionShadow, NativeEarlierEligibilityIsARejectedDifferential) {
	auto completion = std::make_shared<NativeBatchCompletion>(21U);
	NativeBatchLease lease(completion);
	completion->mark_producer_complete();
	lease.request_release();
	lease.observe_legacy_reclaim_eligibility(false);
	EXPECT_EQ(lease.snapshot().differential, NativeEligibilityDifferential::kNativeEarlierUnsafe);
}

TEST(NativeBatchCompletionShadow, InvalidTransitionsAreRejected) {
	NativeBatchCompletion completion(22U);
	auto invalid_device = consumer(1U);
	invalid_device.cuda_device = -1;
	EXPECT_THROW(completion.register_consumer(invalid_device), std::invalid_argument);
	EXPECT_THROW(completion.mark_consumer_complete(0, 404U), std::invalid_argument);
	EXPECT_TRUE(completion.register_consumer(consumer(404U)));
	completion.request_release();
	EXPECT_THROW(completion.register_consumer(consumer(505U)), std::logic_error);
}

TEST(NativeBatchCompletionShadow, AuthoritativeLeaseReclaimsBackingAfterFinalReference) {
	auto completion = std::make_shared<NativeBatchCompletion>(24U);
	auto backing = std::make_shared<galp::jpeg::DirectDctBatch>();
	std::weak_ptr<galp::jpeg::DirectDctBatch> backing_observer = backing;
	auto lease = std::make_shared<NativeBatchLease>(completion, std::move(backing));
	lease->retain_owner_reference();

	EXPECT_TRUE(lease->authoritative());
	EXPECT_TRUE(lease->release_owner_reference());
	static_cast<void>(NativeBatchLease::reclaim_finished());

	const auto snapshot = lease->snapshot();
	EXPECT_TRUE(snapshot.authoritative);
	EXPECT_TRUE(snapshot.completion.producer_complete);
	EXPECT_TRUE(snapshot.completion.reclaim_eligible);
	EXPECT_TRUE(snapshot.reclaim_executed);
	EXPECT_FALSE(snapshot.backing_storage_present);
	EXPECT_TRUE(backing_observer.expired());
}

} // namespace
