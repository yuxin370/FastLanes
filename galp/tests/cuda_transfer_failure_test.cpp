// Link-time CUDA fault injection, private to this CPU test executable. The
// production pool/tracker/arena are compiled unchanged, without linking cudart.
#include "../src/cuda/memory/device_arena.cu"
#include "cuda/memory/gpu_array.cuh"
#include <cstdlib>
#include <gtest/gtest.h>
#include <map>

struct CUevent_st {
	bool complete = false;
};

namespace {
int                                 current_device           = 0;
int                                 synchronize_calls        = 0;
int                                 device_synchronize_calls = 0;
int                                 copies                   = 0;
int                                 fail_copy_number         = 0;
int                                 queries                  = 0;
int                                 fail_query_number        = 0;
bool                                fail_sync                = false;
bool                                fail_create              = false;
bool                                fail_record              = false;
bool                                fail_destroy             = false;
cudaStream_t                        fail_stream              = nullptr;
std::map<cudaEvent_t, cudaStream_t> events;
std::map<void*, size_t>             allocations;
struct PendingCopy {
	void*        dst;
	const void*  src;
	size_t       bytes;
	cudaStream_t stream;
};
std::vector<PendingCopy> pending_copies;

cudaStream_t stream(size_t id) {
	return reinterpret_cast<cudaStream_t>(id);
}
} // namespace

extern "C" {
const char* CUDARTAPI cudaGetErrorString(cudaError_t) {
	return "injected CUDA failure";
}
cudaError_t CUDARTAPI cudaGetDevice(int* device) {
	*device = current_device;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaSetDevice(int device) {
	current_device = device;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaMalloc(void** ptr, size_t size) {
	*ptr              = std::malloc(size);
	allocations[*ptr] = size;
	return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
cudaError_t CUDARTAPI cudaMallocHost(void** ptr, size_t size) {
	return cudaMalloc(ptr, size);
}
cudaError_t CUDARTAPI cudaMallocAsync(void** ptr, size_t size, cudaStream_t) {
	return cudaMalloc(ptr, size);
}
cudaError_t CUDARTAPI cudaFree(void* ptr) {
	const auto begin = reinterpret_cast<uintptr_t>(ptr);
	const auto end   = begin + allocations.at(ptr);
	for (const auto& copy : pending_copies) {
		const auto dst = reinterpret_cast<uintptr_t>(copy.dst);
		const auto src = reinterpret_cast<uintptr_t>(copy.src);
		EXPECT_FALSE((dst >= begin && dst < end) || (src >= begin && src < end))
		    << "freed backing before DMA completed";
	}
	allocations.erase(ptr);
	std::free(ptr);
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaFreeHost(void* ptr) {
	return cudaFree(ptr);
}
cudaError_t CUDARTAPI cudaFreeAsync(void* ptr, cudaStream_t) {
	return cudaFree(ptr);
}
cudaError_t CUDARTAPI cudaMemcpyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind, cudaStream_t value) {
	++copies;
	if (copies == fail_copy_number) {
		return cudaErrorInvalidValue;
	}
	pending_copies.push_back({dst, src, size, value});
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
	if (fail_create) {
		return cudaErrorMemoryAllocation;
	}
	*event         = new CUevent_st;
	events[*event] = nullptr;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event) {
	return cudaEventCreateWithFlags(event, 0);
}
cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t value) {
	if (fail_record) {
		return cudaErrorInvalidValue;
	}
	events[event] = value;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
	if (++queries == fail_query_number) {
		return cudaErrorInvalidValue;
	}
	return event->complete ? cudaSuccess : cudaErrorNotReady;
}
cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
	if (fail_destroy) {
		return cudaErrorInvalidValue;
	}
	events.erase(event);
	delete event;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t value) {
	++synchronize_calls;
	if (fail_sync && (fail_stream == nullptr || value == fail_stream)) {
		return cudaErrorUnknown;
	}
	for (auto it = pending_copies.begin(); it != pending_copies.end();) {
		if (it->stream == value) {
			std::memcpy(it->dst, it->src, it->bytes);
			it = pending_copies.erase(it);
		} else {
			++it;
		}
	}
	for (auto [event, event_stream] : events) {
		if (event_stream == value) {
			event->complete = true;
		}
	}
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaDeviceSynchronize() {
	++device_synchronize_calls;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
	event->complete = true;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventElapsedTime(float* elapsed, cudaEvent_t, cudaEvent_t) {
	*elapsed = 0.0F;
	return cudaSuccess;
}
}

namespace {
using galp::memory::CudaError;
using galp::memory::DeviceArena;
using galp::memory::DevicePool;
using galp::memory::TransferTracker;

class CudaTransferFailure : public ::testing::Test {
	void SetUp() override {
		copies = queries = synchronize_calls = device_synchronize_calls = 0;
		fail_copy_number = fail_query_number = 0;
		fail_sync = fail_create = fail_record = fail_destroy = false;
		fail_stream                                          = nullptr;
		DevicePool::instance().set_use_async(false);
		DevicePool::instance().set_small_copy_threshold(0);
	}
	void TearDown() override {
		fail_sync = fail_create = fail_record = fail_destroy = false;
		auto& pool                                           = DevicePool::instance();
		pool.release_cached();
		EXPECT_EQ(pool.stats().in_use_bytes, 0U);
		EXPECT_EQ(pool.pinned_stats().in_use_bytes, 0U);
		EXPECT_TRUE(events.empty());
		EXPECT_TRUE(allocations.empty());
		EXPECT_TRUE(pending_copies.empty());
		EXPECT_EQ(device_synchronize_calls, 0);
	}
};

TEST_F(CudaTransferFailure, SyncAllKeepsFailedAndUnvisitedEntries) {
	TransferTracker    tracker;
	int                pinned[3] {};
	std::vector<void*> released;
	for (size_t index = 0; index < 3; ++index) {
		tracker.submit(stream(index + 1), [&](void*& owned) { owned = &pinned[index]; });
	}
	const auto release = [&](void* ptr) {
		released.push_back(ptr);
	};
	fail_sync   = true;
	fail_stream = stream(2);
	EXPECT_THROW(tracker.sync_all(release), CudaError);
	ASSERT_EQ(released.size(), 1U);
	EXPECT_FALSE(tracker.empty());
	EXPECT_EQ(events.size(), 2U);
	fail_sync = false;
	tracker.sync_all(release);
	EXPECT_EQ(released, (std::vector<void*> {&pinned[0], &pinned[1], &pinned[2]}));
	EXPECT_TRUE(tracker.empty());
}

TEST_F(CudaTransferFailure, QueryFailureDoesNotLoseEarlierCompletedEntries) {
	TransferTracker tracker;
	int             pinned[3] {};
	int             released = 0;
	for (auto& value : pinned) {
		tracker.submit(stream(1), [&](void*& owned) { owned = &value; });
	}
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream(1)));
	fail_query_number  = 2;
	const auto release = [&](void*) {
		++released;
	};
	EXPECT_THROW(tracker.reclaim_finished(release), CudaError);
	EXPECT_EQ(released, 1);
	EXPECT_EQ(events.size(), 2U);
	tracker.reclaim_finished(release);
	EXPECT_EQ(released, 3);
	EXPECT_TRUE(tracker.empty());
}

TEST_F(CudaTransferFailure, DestroyFailureStillReleasesAllSafePinnedAllocations) {
	TransferTracker tracker;
	int             pinned[3] {};
	int             released = 0;
	for (auto& value : pinned) {
		tracker.submit(stream(1), [&](void*& owned) { owned = &value; });
	}
	fail_destroy       = true;
	const auto release = [&](void*) {
		++released;
	};
	EXPECT_THROW(tracker.sync_all(release), CudaError);
	EXPECT_EQ(released, 3);
	EXPECT_EQ(events.size(), 3U);
	fail_destroy = false;
	tracker.reclaim_finished(release);
	EXPECT_EQ(released, 3);
	EXPECT_TRUE(tracker.empty());
}

TEST_F(CudaTransferFailure, EventCreationFailurePrecedesDmaAndPinnedAllocation) {
	auto& pool  = DevicePool::instance();
	int   input = 42;
	void* dst   = pool.alloc(sizeof(input));
	fail_create = true;
	EXPECT_THROW(pool.copy_h2d_on_stream(dst, &input, sizeof(input), stream(1)), CudaError);
	EXPECT_EQ(copies, 0);
	EXPECT_EQ(pool.pinned_stats().in_use_bytes, 0U);
	fail_create = false;
	pool.free(dst);
}

TEST_F(CudaTransferFailure, RecordFailureRetainsDeviceAndPinnedUntilStreamCompletes) {
	auto& pool  = DevicePool::instance();
	int   input = 42;
	void* dst   = pool.alloc(sizeof(input));
	fail_record = fail_sync = true;
	EXPECT_THROW(pool.copy_h2d_on_stream(dst, &input, sizeof(input), stream(1)), CudaError);
	EXPECT_THROW(pool.free(dst), CudaError);
	EXPECT_EQ(pending_copies.size(), 1U);
	EXPECT_EQ(pool.stats().in_use_bytes, sizeof(input));
	EXPECT_EQ(pool.pinned_stats().in_use_bytes, sizeof(input));
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
	EXPECT_EQ(pool.pinned_stats().cached_bytes, 0U);
	fail_record = fail_sync = false;
	pool.free(dst);
	EXPECT_EQ(pool.stats().in_use_bytes, 0U);
	EXPECT_EQ(pool.pinned_stats().in_use_bytes, 0U);
}

TEST_F(CudaTransferFailure, ArenaPartialUploadKeepsBackingOnFailedSync) {
	auto&       pool = DevicePool::instance();
	int         source[2] {17, 42};
	DeviceArena arena(stream(1));
	arena.register_backing(&source[0], sizeof(int));
	arena.add(1, &source[0]);
	arena.add(1, &source[1]); // a second DMA from the staged pinned slab
	fail_copy_number = 2;
	fail_sync        = true;
	EXPECT_THROW(arena.upload(), CudaError);
	const auto device_bytes = pool.stats().in_use_bytes;
	const auto pinned_bytes = pool.pinned_stats().in_use_bytes;
	EXPECT_GT(device_bytes, 0U);
	EXPECT_GT(pinned_bytes, 0U);
	EXPECT_THROW(arena.reset(), CudaError);
	EXPECT_EQ(pending_copies.size(), 1U);
	EXPECT_EQ(pool.stats().in_use_bytes, device_bytes);
	EXPECT_EQ(pool.pinned_stats().in_use_bytes, pinned_bytes);
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
	EXPECT_EQ(pool.pinned_stats().cached_bytes, 0U);
	fail_sync = false;
	arena.reset();
}

TEST_F(CudaTransferFailure, ArenaRegistrationFailureWaitsBeforeReleasingBacking) {
	auto&       pool   = DevicePool::instance();
	int         source = 42;
	DeviceArena arena(stream(1));
	arena.add(1, &source);
	fail_record = fail_sync = true;
	EXPECT_THROW(arena.upload(), CudaError);
	EXPECT_THROW(arena.reset(true), CudaError);
	EXPECT_GT(pool.stats().in_use_bytes, 0U);
	EXPECT_GT(pool.pinned_stats().in_use_bytes, 0U);
	fail_record = fail_sync = false;
	arena.reset();
}

TEST_F(CudaTransferFailure, GpuArrayFailedConstructorUsesOneCleanupOwner) {
	int source  = 42;
	fail_record = true;
	EXPECT_THROW((GPUArray<int>(1, &source, stream(1))), CudaError);
	EXPECT_EQ(synchronize_calls, 1);
	EXPECT_EQ(DevicePool::instance().stats().in_use_bytes, 0U);
	EXPECT_EQ(DevicePool::instance().pinned_stats().in_use_bytes, 0U);
}

TEST_F(CudaTransferFailure, SuccessfulUploadStaysAsynchronous) {
	int         source = 42;
	DeviceArena arena(stream(1));
	arena.add(1, &source);
	EXPECT_NO_THROW(arena.upload());
	EXPECT_EQ(copies, 1);
	EXPECT_EQ(synchronize_calls, 0);
	EXPECT_EQ(device_synchronize_calls, 0);
	arena.reset();
	EXPECT_EQ(synchronize_calls, 1);
}
} // namespace
