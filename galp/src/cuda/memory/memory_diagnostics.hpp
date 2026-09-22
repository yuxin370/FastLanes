#ifndef GALP_MEMORY_DIAGNOSTICS_HPP
#define GALP_MEMORY_DIAGNOSTICS_HPP

#include <mutex>

namespace galp::memory::diagnostics {
enum Lock { Tracker, Device, Pinned, LockCount };

// Standalone diagnostics only: never mix instrumented and uninstrumented
// definitions in one executable. Production has exactly std::mutex overhead.
#ifdef GALP_MEMORY_DIAGNOSTICS
} // namespace galp::memory::diagnostics
#include <array>
#include <chrono>
#include <cstdint>
namespace galp::memory::diagnostics {
struct Counters {
	std::array<uint64_t, LockCount> wait_ns {};
	uint64_t                        submit_count   = 0;
	uint64_t                        submit_host_ns = 0;
	uint64_t                        sync_count     = 0;
	uint64_t                        sync_wait_ns   = 0;
};
inline thread_local Counters counters;
using Clock = std::chrono::steady_clock;
inline uint64_t elapsed(Clock::time_point start) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
template <Lock Kind>
class Mutex {
public:
	void lock() {
		const auto start = Clock::now();
		mutex_.lock();
		counters.wait_ns[Kind] += elapsed(start);
	}
	void unlock() {
		mutex_.unlock();
	}

private:
	std::mutex mutex_;
};
class SubmitTimer {
public:
	~SubmitTimer() {
		++counters.submit_count;
		counters.submit_host_ns += elapsed(start_);
	}

private:
	Clock::time_point start_ = Clock::now();
};
class SyncTimer {
public:
	~SyncTimer() {
		++counters.sync_count;
		counters.sync_wait_ns += elapsed(start_);
	}

private:
	Clock::time_point start_ = Clock::now();
};
#else
template <Lock>
using Mutex = std::mutex;
struct SubmitTimer {};
struct SyncTimer {};
#endif
} // namespace galp::memory::diagnostics
#endif
