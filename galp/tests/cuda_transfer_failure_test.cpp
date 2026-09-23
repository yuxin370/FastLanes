// Link-time CUDA fault injection, private to this CPU test executable. The
// production pool/tracker/arena are compiled unchanged, without linking cudart.
#include "../src/cuda/memory/device_arena.cu"
#include "cuda/memory/gpu_array.cuh"
#include <barrier>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <thread>

struct CUevent_st {
	bool complete = false;
	int  device   = 0;
};

namespace {
thread_local int                    current_device           = 0;
int                                 synchronize_calls        = 0;
int                                 device_synchronize_calls = 0;
int                                 copies                   = 0;
int                                 fail_copy_number         = 0;
int                                 queries                  = 0;
int                                 fail_query_number        = 0;
bool                                fail_sync                = false;
bool                                fail_create              = false;
bool                                fail_pinned_allocation   = false;
bool                                fail_record              = false;
bool                                fail_destroy             = false;
cudaStream_t                        fail_stream              = nullptr;
std::map<cudaEvent_t, cudaStream_t> events;
std::map<void*, size_t>             allocations;
std::map<void*, int>                allocation_devices;
std::set<void*>                     freed_addresses;
int                                 free_calls = 0, free_host_calls = 0;
int                                 fail_free_number = 0, fail_free_host_number = 0;
int                                 duplicate_frees    = 0;
constexpr size_t                    device_cache_limit = 4096;
struct PendingCopy {
	void*        dst;
	const void*  src;
	size_t       bytes;
	cudaStream_t stream;
	int          device;
};
std::vector<PendingCopy>          pending_copies;
std::mutex                        mock_mutex;
std::function<void(cudaStream_t)> before_copy, before_sync;
std::function<void(void*)>        before_free;

cudaStream_t stream(size_t id) {
	return reinterpret_cast<cudaStream_t>(id);
}

cudaError_t free_allocation(void* ptr, bool pinned) {
	if (before_free)
		before_free(ptr);
	std::lock_guard lock(mock_mutex);
	int&            calls = pinned ? free_host_calls : free_calls;
	if (++calls == (pinned ? fail_free_host_number : fail_free_number))
		return cudaErrorUnknown;
	const auto it = allocations.find(ptr);
	if (it == allocations.end()) {
		++duplicate_frees;
		ADD_FAILURE() << "duplicate/unknown CUDA free: " << ptr;
		return cudaErrorInvalidValue;
	}
	if (!pinned)
		EXPECT_EQ(allocation_devices.at(ptr), current_device) << "free on wrong CUDA device";
	const auto begin = reinterpret_cast<uintptr_t>(ptr);
	const auto end   = begin + it->second;
	for (const auto& copy : pending_copies) {
		const auto dst = reinterpret_cast<uintptr_t>(copy.dst);
		const auto src = reinterpret_cast<uintptr_t>(copy.src);
		EXPECT_FALSE((dst >= begin && dst < end) || (src >= begin && src < end))
		    << "freed backing before DMA completed";
	}
	freed_addresses.insert(ptr);
	allocation_devices.erase(ptr);
	allocations.erase(it);
	std::free(ptr);
	return cudaSuccess;
}

void expect_live(void* ptr) {
	std::lock_guard lock(mock_mutex);
	EXPECT_TRUE(allocations.contains(ptr)) << "cache returned released address " << ptr;
	EXPECT_FALSE(freed_addresses.contains(ptr));
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
	std::lock_guard lock(mock_mutex);
	*ptr                     = std::malloc(size);
	allocations[*ptr]        = size;
	allocation_devices[*ptr] = current_device;
	freed_addresses.erase(*ptr); // malloc may legitimately reuse an address.
	return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
cudaError_t CUDARTAPI cudaMallocHost(void** ptr, size_t size) {
	if (fail_pinned_allocation)
		return cudaErrorMemoryAllocation;
	return cudaMalloc(ptr, size);
}
cudaError_t CUDARTAPI cudaMallocAsync(void** ptr, size_t size, cudaStream_t) {
	return cudaMalloc(ptr, size);
}
cudaError_t CUDARTAPI cudaFree(void* ptr) {
	return free_allocation(ptr, false);
}
cudaError_t CUDARTAPI cudaFreeHost(void* ptr) {
	return free_allocation(ptr, true);
}
cudaError_t CUDARTAPI cudaFreeAsync(void* ptr, cudaStream_t) {
	return cudaFree(ptr);
}
cudaError_t CUDARTAPI cudaMemcpyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind, cudaStream_t value) {
	if (before_copy)
		before_copy(value);
	std::lock_guard lock(mock_mutex);
	++copies;
	if (copies == fail_copy_number) {
		return cudaErrorInvalidValue;
	}
	pending_copies.push_back({dst, src, size, value, current_device});
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
	std::lock_guard lock(mock_mutex);
	if (fail_create) {
		return cudaErrorMemoryAllocation;
	}
	*event           = new CUevent_st;
	(*event)->device = current_device;
	events[*event]   = nullptr;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event) {
	return cudaEventCreateWithFlags(event, 0);
}
cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t value) {
	std::lock_guard lock(mock_mutex);
	if (fail_record) {
		return cudaErrorInvalidValue;
	}
	events[event] = value;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
	std::lock_guard lock(mock_mutex);
	if (++queries == fail_query_number) {
		return cudaErrorInvalidValue;
	}
	return event->complete ? cudaSuccess : cudaErrorNotReady;
}
cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
	std::lock_guard lock(mock_mutex);
	if (fail_destroy) {
		return cudaErrorInvalidValue;
	}
	events.erase(event);
	delete event;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t value) {
	if (before_sync)
		before_sync(value);
	std::lock_guard lock(mock_mutex);
	++synchronize_calls;
	if (fail_sync && (fail_stream == nullptr || value == fail_stream)) {
		return cudaErrorUnknown;
	}
	for (auto it = pending_copies.begin(); it != pending_copies.end();) {
		if (it->stream == value && it->device == current_device) {
			std::memcpy(it->dst, it->src, it->bytes);
			it = pending_copies.erase(it);
		} else {
			++it;
		}
	}
	for (auto [event, event_stream] : events) {
		if (event_stream == value && event->device == current_device) {
			event->complete = true;
		}
	}
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaDeviceSynchronize() {
	std::lock_guard lock(mock_mutex);
	++device_synchronize_calls;
	return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
	std::lock_guard lock(mock_mutex);
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
using galp::memory::PinnedHostPool;
using galp::memory::TransferTracker;

class CudaTransferFailure : public ::testing::Test {
	void SetUp() override {
		// Set before the first singleton access; keep eviction tests small.
		setenv("GALP_DEVICE_POOL_CACHE_LIMIT_BYTES", "4096", 1);
		current_device = 0;
		before_copy = before_sync = {};
		before_free               = {};
		copies = queries = synchronize_calls = device_synchronize_calls = 0;
		fail_copy_number = fail_query_number = 0;
		free_calls = free_host_calls = duplicate_frees = 0;
		fail_free_number = fail_free_host_number = 0;
		freed_addresses.clear();
		fail_sync = fail_create = fail_record = fail_destroy = fail_pinned_allocation = false;
		fail_stream                                                                   = nullptr;
		DevicePool::instance().set_use_async(false);
		DevicePool::instance().set_small_copy_threshold(0);
	}
	void TearDown() override {
		fail_free_number = fail_free_host_number = 0;
		before_copy = before_sync = {};
		before_free               = {};
		fail_sync = fail_create = fail_record = fail_destroy = fail_pinned_allocation = false;
		auto& pool                                                                    = DevicePool::instance();
		pool.release_cached();
		EXPECT_EQ(pool.stats().in_use_bytes, 0U);
		EXPECT_EQ(pool.pinned_stats().in_use_bytes, 0U);
		EXPECT_TRUE(events.empty());
		EXPECT_TRUE(allocations.empty());
		EXPECT_TRUE(pending_copies.empty());
		EXPECT_EQ(device_synchronize_calls, 0);
		EXPECT_EQ(duplicate_frees, 0);
	}
};

TEST_F(CudaTransferFailure, PinnedReleaseEvictionFailureKeepsOwnership) {
	PinnedHostPool pool(64);
	void*          old      = pool.alloc(64);
	void*          incoming = pool.alloc(32);
	pool.release(old);
	fail_free_host_number = 1;
	EXPECT_THROW(pool.release(incoming), CudaError);
	EXPECT_EQ(pool.stats().in_use_bytes, 32U);
	EXPECT_EQ(pool.stats().cached_bytes, 64U);
	void* other = pool.alloc(64);
	EXPECT_NE(other, old); // failed-free blocks are not reusable
	EXPECT_NE(other, incoming);
	pool.release(incoming);
	pool.release(other);
	pool.release_cached();
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
}

TEST_F(CudaTransferFailure, PinnedReleaseRetryDoesNotDoubleFree) {
	PinnedHostPool pool(64);
	void*          old      = pool.alloc(64);
	void*          incoming = pool.alloc(32);
	pool.release(old);
	fail_free_host_number = 1;
	EXPECT_THROW(pool.release(incoming), CudaError);
	EXPECT_NO_THROW(pool.release(incoming));
	EXPECT_NO_THROW(pool.release_cached());
	EXPECT_EQ(duplicate_frees, 0);
	EXPECT_EQ(pool.stats().in_use_bytes, 0U);
}

TEST_F(CudaTransferFailure, PinnedReleaseCachedPartialFailureIsRetryable) {
	PinnedHostPool pool;
	void*          first  = pool.alloc(16);
	void*          second = pool.alloc(32);
	void*          third  = pool.alloc(64);
	pool.release(first);
	pool.release(second);
	pool.release(third);
	fail_free_host_number = 2;
	EXPECT_THROW(pool.release_cached(), CudaError);
	EXPECT_TRUE(freed_addresses.contains(first));
	expect_live(second);
	expect_live(third);
	EXPECT_EQ(pool.stats().cached_bytes, 96U);
	void* reused = pool.alloc(32);
	EXPECT_NE(reused, second);
	expect_live(reused);
	pool.release(reused);
	EXPECT_NO_THROW(pool.release_cached());
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
	EXPECT_TRUE(allocations.empty());
}

TEST_F(CudaTransferFailure, PinnedReleasedPointerIsNeverReturnedFromCacheAfterFree) {
	PinnedHostPool pool(64);
	void*          old      = pool.alloc(64);
	void*          incoming = pool.alloc(32);
	pool.release(old);
	fail_free_host_number = 1;
	EXPECT_THROW(pool.release(incoming), CudaError);
	pool.release(incoming);
	void* reused = pool.alloc(32);
	expect_live(reused);
	pool.release(reused);
	EXPECT_NO_THROW(pool.release_cached());
}

TEST_F(CudaTransferFailure, DeviceReleaseCachedPartialFailureIsRetryable) {
	auto& pool   = DevicePool::instance();
	void* first  = pool.alloc(16);
	void* second = pool.alloc(32);
	void* third  = pool.alloc(64);
	pool.free(first);
	pool.free(second);
	pool.free(third);
	fail_free_number = 2;
	EXPECT_THROW(pool.release_cached(), CudaError);
	EXPECT_TRUE(freed_addresses.contains(first));
	expect_live(second);
	expect_live(third);
	EXPECT_EQ(pool.stats().cached_bytes, 96U);
	void* reused = pool.alloc(32);
	EXPECT_NE(reused, second);
	expect_live(reused);
	pool.free(reused);
	EXPECT_NO_THROW(pool.release_cached());
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
}

TEST_F(CudaTransferFailure, DeviceEvictionFailureKeepsBlockOwned) {
	auto& pool     = DevicePool::instance();
	void* old      = pool.alloc(device_cache_limit);
	void* incoming = pool.alloc(32);
	pool.free(old);
	fail_free_number = 1;
	EXPECT_THROW(pool.free(incoming), CudaError);
	EXPECT_EQ(pool.stats().in_use_bytes, 32U);
	EXPECT_EQ(pool.stats().cached_bytes, device_cache_limit);
	void* other = pool.alloc(device_cache_limit);
	EXPECT_NE(other, old);
	expect_live(other);
	EXPECT_NO_THROW(pool.free(incoming));
	pool.free(other);
	pool.release_cached();
}

TEST_F(CudaTransferFailure, DeviceRetryUsesOriginalCudaDevice) {
	auto& pool     = DevicePool::instance();
	void* first    = pool.alloc(16);
	current_device = 1;
	void* second   = pool.alloc(32);
	pool.free(first);
	pool.free(second);
	current_device   = 2;
	fail_free_number = 2;
	EXPECT_THROW(pool.release_cached(), CudaError);
	EXPECT_EQ(current_device, 2);
	EXPECT_NO_THROW(pool.release_cached());
	EXPECT_EQ(current_device, 2);
	EXPECT_TRUE(allocations.empty());
}

TEST_F(CudaTransferFailure, TrackerPinnedEvictionFailureCanRetryWithoutDanglingCache) {
	PinnedHostPool  pool(64);
	TransferTracker tracker;
	void*           old      = pool.alloc(64);
	void*           incoming = pool.alloc(32);
	pool.release(old);
	char destination[32] {};
	tracker.submit(stream(1), [&](void*& pinned) {
		pinned = incoming;
		CUDA_SAFE_CALL(cudaMemcpyAsync(destination, pinned, 32, cudaMemcpyHostToDevice, stream(1)));
	});
	const auto release = [&](void* ptr) {
		pool.release(ptr);
	};
	fail_free_host_number = 1;
	EXPECT_THROW(tracker.sync_all(release), CudaError);
	EXPECT_FALSE(tracker.empty());
	EXPECT_EQ(pool.stats().in_use_bytes, 32U);
	tracker.sync_all(release);
	EXPECT_TRUE(tracker.empty());
	void* reused = pool.alloc(32);
	expect_live(reused);
	pool.release(reused);
	EXPECT_NO_THROW(pool.release_cached());
	EXPECT_FALSE(pool.has_in_use());
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
}

TEST_F(CudaTransferFailure, PinnedCacheLimitFailureIsRetryable) {
	PinnedHostPool pool;
	void*          ptr = pool.alloc(64);
	pool.release(ptr);
	fail_free_host_number = 1;
	EXPECT_THROW(pool.set_cache_limit_bytes(0), CudaError);
	EXPECT_EQ(pool.stats().cached_bytes, 64U);
	void* other = pool.alloc(64);
	EXPECT_NE(other, ptr);
	pool.release(other);
	pool.release_cached();
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
}

TEST_F(CudaTransferFailure, UncachedFreeFailureLeavesAllocationLiveForRetry) {
	PinnedHostPool pinned(0);
	void*          host   = pinned.alloc(64);
	fail_free_host_number = 1;
	EXPECT_THROW(pinned.release(host), CudaError);
	EXPECT_EQ(pinned.stats().in_use_bytes, 64U);
	pinned.release(host);
	EXPECT_EQ(pinned.stats().in_use_bytes, 0U);
	auto& pool       = DevicePool::instance();
	void* device     = pool.alloc(device_cache_limit + 1);
	fail_free_number = 1;
	EXPECT_THROW(pool.free(device), CudaError);
	EXPECT_EQ(pool.stats().in_use_bytes, device_cache_limit + 1);
	pool.free(device);
}

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

TEST_F(CudaTransferFailure, PinnedAllocationFailureCancelsBeforeDma) {
	auto& pool             = DevicePool::instance();
	int   source           = 42;
	void* dst              = pool.alloc(sizeof(source));
	fail_pinned_allocation = true;
	EXPECT_THROW(pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(1)), CudaError);
	fail_pinned_allocation = false;
	EXPECT_EQ(copies, 0);
	EXPECT_TRUE(events.empty());
	pool.free(dst);
	EXPECT_EQ(synchronize_calls, 0);
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

// A deterministic pause inside an injected CUDA operation, not a timing race.
class Gate {
public:
	void pause() {
		std::unique_lock lock(mutex_);
		entered_ = true;
		changed_.notify_all();
		changed_.wait(lock, [&] { return open_; });
	}
	void wait() {
		std::unique_lock lock(mutex_);
		changed_.wait(lock, [&] { return entered_; });
	}
	void open() {
		std::lock_guard lock(mutex_);
		open_ = true;
		changed_.notify_all();
	}

private:
	std::mutex              mutex_;
	std::condition_variable changed_;
	bool                    entered_ = false, open_ = false;
};

TEST_F(CudaTransferFailure, PinnedReleaseRejectsConcurrentRelease) {
	for (const bool eviction : {false, true}) {
		PinnedHostPool pool(eviction ? 64 : 0);
		void*          old = pool.alloc(64);
		void*          ptr = pool.alloc(32);
		pool.release(old);
		Gate gate;
		before_free = [&](void*) {
			gate.pause();
		};
		auto releasing = std::async(std::launch::async, [&] { pool.release(ptr); });
		gate.wait();
		EXPECT_THROW(pool.release(ptr), std::logic_error);
		EXPECT_EQ(pool.stats().in_use_bytes, 32U);
		gate.open();
		EXPECT_NO_THROW(releasing.get());
		before_free = {};
		EXPECT_THROW(pool.release(ptr), std::invalid_argument);
		pool.release_cached();
	}
}

TEST_F(CudaTransferFailure, PinnedFailedReleaseClearsBusyForRetry) {
	for (const bool eviction : {false, true}) {
		PinnedHostPool pool(eviction ? 64 : 0);
		void*          old = pool.alloc(64);
		void*          ptr = pool.alloc(32);
		pool.release(old);
		fail_free_host_number = free_host_calls + 1;
		Gate gate;
		before_free = [&](void*) {
			gate.pause();
		};
		auto releasing = std::async(std::launch::async, [&] { pool.release(ptr); });
		gate.wait();
		EXPECT_THROW(pool.release(ptr), std::logic_error);
		gate.open();
		EXPECT_THROW(releasing.get(), CudaError);
		before_free = {};
		EXPECT_EQ(pool.stats().in_use_bytes, 32U);
		expect_live(ptr);
		EXPECT_NO_THROW(pool.release(ptr));
		pool.release_cached();
		EXPECT_EQ(pool.stats().cached_bytes, 0U);
		EXPECT_FALSE(pool.has_in_use());
	}
}

TEST_F(CudaTransferFailure, PinnedDifferentAllocationsReleaseConcurrently) {
	PinnedHostPool pool(0);
	void*          first  = pool.alloc(64);
	void*          second = pool.alloc(64);
	Gate           gate;
	before_free = [&](void* ptr) {
		if (ptr == first)
			gate.pause();
	};
	auto releasing = std::async(std::launch::async, [&] { pool.release(first); });
	gate.wait();
	auto independent = std::async(std::launch::async, [&] { pool.release(second); });
	EXPECT_EQ(independent.wait_for(std::chrono::seconds(1)), std::future_status::ready);
	gate.open();
	EXPECT_NO_THROW(releasing.get());
	EXPECT_NO_THROW(independent.get());
	before_free = {};
	EXPECT_FALSE(pool.has_in_use());
}

TEST_F(CudaTransferFailure, ConcurrentCacheDrainersKeepPendingBlocksExclusive) {
	// The same regression covers both pools: a paused CUDA free must not block
	// ordinary allocation, and another drainer must not free the pending block.
	for (bool pinned : {false, true}) {
		PinnedHostPool host;
		auto&          device = DevicePool::instance();
		const auto     alloc  = [&] {
            return pinned ? host.alloc(64) : device.alloc(64);
		};
		const auto release = [&](void* ptr) {
			pinned ? host.release(ptr) : device.free(ptr);
		};
		const auto drain = [&] {
			pinned ? host.release_cached() : device.release_cached();
		};
		void* old = alloc();
		release(old);
		Gate gate;
		before_free = [&](void* ptr) {
			if (ptr == old)
				gate.pause();
		};
		auto first = std::async(std::launch::async, drain);
		gate.wait();
		auto second     = std::async(std::launch::async, drain);
		auto allocating = std::async(std::launch::async, alloc);
		EXPECT_EQ(allocating.wait_for(std::chrono::seconds(1)), std::future_status::ready);
		gate.open();
		void* other = allocating.get();
		EXPECT_NE(other, old);
		expect_live(other);
		EXPECT_NO_THROW(first.get());
		EXPECT_NO_THROW(second.get());
		before_free = {};
		release(other);
		drain();
	}
}

TEST_F(CudaTransferFailure, DifferentAllocationsSubmitConcurrentlyAndReturnIdle) {
	std::barrier                   start(8);
	std::vector<std::future<void>> jobs;
	for (int index = 0; index < 8; ++index) {
		jobs.push_back(std::async(std::launch::async, [&, index] {
			CUDA_SAFE_CALL(cudaSetDevice(index % 2));
			auto&      pool   = DevicePool::instance();
			const auto value  = stream(static_cast<size_t>(index + 1));
			int        source = index + 17;
			void*      dst    = pool.alloc(sizeof(source));
			start.arrive_and_wait();
			for (int repeat = 0; repeat < 50; ++repeat) {
				pool.copy_h2d_on_stream(dst, &source, sizeof(source), value);
				pool.sync_h2d(value);
				EXPECT_EQ(*static_cast<int*>(dst), source);
			}
			pool.free(dst);
		}));
	}
	for (auto& job : jobs)
		EXPECT_NO_THROW(job.get());
}

TEST_F(CudaTransferFailure, CopyRejectsOverlappingCopyAndFree) {
	auto& pool   = DevicePool::instance();
	int   source = 42;
	void* dst    = pool.alloc(sizeof(source));
	Gate  gate;
	before_copy = [&](cudaStream_t) {
		gate.pause();
	};
	auto copy =
	    std::async(std::launch::async, [&] { pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(1)); });
	gate.wait();
	EXPECT_THROW(pool.free(dst), std::logic_error);
	EXPECT_THROW(pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(2)), std::logic_error);
	EXPECT_EQ(pool.stats().in_use_bytes, sizeof(source));
	gate.open();
	EXPECT_NO_THROW(copy.get());
	pool.free(dst);
	EXPECT_THROW(pool.free(dst), std::invalid_argument); // not cudaFree(cached_ptr)
}

TEST_F(CudaTransferFailure, FreeRejectsOverlappingFreeAndCopyWhileWaiting) {
	auto& pool   = DevicePool::instance();
	int   source = 42;
	void* dst    = pool.alloc(sizeof(source));
	pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(1));
	Gate gate;
	before_sync = [&](cudaStream_t) {
		gate.pause();
	};
	auto freeing = std::async(std::launch::async, [&] { pool.free(dst); });
	gate.wait();
	EXPECT_THROW(pool.free(dst), std::logic_error);
	EXPECT_THROW(pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(2)), std::logic_error);
	gate.open();
	EXPECT_NO_THROW(freeing.get());
}

TEST_F(CudaTransferFailure, StreamSwitchKeepsAllocationExclusiveWhileUnlocked) {
	auto& pool   = DevicePool::instance();
	int   source = 42;
	void* dst    = pool.alloc(sizeof(source));
	pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(1));
	Gate gate;
	before_sync = [&](cudaStream_t value) {
		if (value == stream(1))
			gate.pause();
	};
	auto copy =
	    std::async(std::launch::async, [&] { pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(2)); });
	gate.wait();
	EXPECT_THROW(pool.free(dst), std::logic_error);
	gate.open();
	EXPECT_NO_THROW(copy.get());
	pool.free(dst);
}

TEST_F(CudaTransferFailure, WaitingStreamDoesNotBlockIndependentSubmission) {
	for (const bool fail : {false, true}) {
		TransferTracker tracker;
		int             pinned = 42;
		tracker.submit(stream(1), [&](void*& owner) { owner = &pinned; });
		Gate gate;
		fail_sync   = fail;
		fail_stream = stream(1);
		before_sync = [&](cudaStream_t value) {
			if (value == stream(1))
				gate.pause();
		};
		int  released = 0;
		auto sync = std::async(std::launch::async, [&] { tracker.sync_stream(stream(1), [&](void*) { ++released; }); });
		gate.wait();
		auto submit = std::async(std::launch::async, [&] { tracker.submit(stream(2), [](void*&) {}); });
		EXPECT_EQ(submit.wait_for(std::chrono::seconds(1)), std::future_status::ready);
		EXPECT_FALSE(tracker.empty());
		gate.open();
		submit.get();
		if (fail) {
			EXPECT_THROW(sync.get(), CudaError);
			EXPECT_EQ(released, 0);
		} else {
			EXPECT_NO_THROW(sync.get());
			EXPECT_EQ(released, 1);
		}
		before_sync = {};
		fail_sync   = false;
		tracker.sync_all([&](void*) { ++released; });
		EXPECT_EQ(released, 1);
		EXPECT_TRUE(tracker.empty());
	}
}

TEST_F(CudaTransferFailure, SyncSnapshotDoesNotConsumeLaterSubmission) {
	TransferTracker tracker;
	tracker.submit(stream(1), [](void*&) {});
	Gate gate;
	before_sync = [&](cudaStream_t) {
		gate.pause();
	};
	auto sync = std::async(std::launch::async, [&] { tracker.sync_all({}); });
	gate.wait();
	auto submit = std::async(std::launch::async, [&] { tracker.submit(stream(1), [](void*&) {}); });
	EXPECT_EQ(submit.wait_for(std::chrono::seconds(1)), std::future_status::ready);
	gate.open();
	submit.get();
	sync.get();
	EXPECT_FALSE(tracker.empty());
	EXPECT_EQ(events.size(), 1U);
	tracker.sync_all({});
	EXPECT_TRUE(tracker.empty());
}

TEST_F(CudaTransferFailure, SubmittingEntrySurvivesConcurrentSyncAndQuery) {
	TransferTracker tracker;
	Gate            gate;
	int             pinned = 42, released = 0;
	auto            submit = std::async(std::launch::async, [&] {
        tracker.submit(stream(1), [&](void*& owner) {
            owner = &pinned;
            gate.pause();
        });
    });
	gate.wait();
	tracker.reclaim_finished([&](void*) { ++released; });
	EXPECT_EQ(released, 0);
	auto sync = std::async(std::launch::async, [&] { tracker.sync_stream(stream(1), [&](void*) { ++released; }); });
	EXPECT_EQ(sync.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
	gate.open();
	submit.get();
	sync.get();
	EXPECT_EQ(released, 1);
	EXPECT_TRUE(tracker.empty());
}

TEST_F(CudaTransferFailure, PreparationFailureHasNoDmaAndReleasesReservation) {
	TransferTracker tracker;
	int             pinned = 42, released = 0;
	EXPECT_THROW(tracker.submit(
	                 stream(1),
	                 [&](void*& owner) {
		                 owner = &pinned;
		                 throw std::runtime_error("preparation failed");
	                 },
	                 [](void*&) { FAIL() << "DMA must not start"; },
	                 [&](void*) { ++released; }),
	             std::runtime_error);
	EXPECT_EQ(released, 1);
	EXPECT_TRUE(tracker.empty());
	EXPECT_EQ(synchronize_calls, 0);
}

TEST_F(CudaTransferFailure, CrossThreadFreeUsesAllocationDeviceAndWrongDeviceCopyFails) {
	auto& pool   = DevicePool::instance();
	int   source = 42;
	void* dst    = pool.alloc(sizeof(source));
	pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(1));
	auto other_device = std::async(std::launch::async, [&] {
		CUDA_SAFE_CALL(cudaSetDevice(1));
		EXPECT_THROW(pool.copy_h2d_on_stream(dst, &source, sizeof(source), stream(1)), std::invalid_argument);
		pool.free(dst);
		EXPECT_EQ(current_device, 1);
	});
	other_device.get();
}
} // namespace
