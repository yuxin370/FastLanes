// Benchmark application-read completion control; the author reader is unchanged
// except for reserve/finish calls around its unbuffered ReadNBytes operation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>

namespace {
std::mutex mutex;
bool       active = false;
double     rate = 0, deadline = 0, first = 0, last = 0;
uint64_t   bytes = 0, calls = 0;

double now() {
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

extern "C" int pcr_boundary_begin(double bytes_per_second) {
	std::lock_guard<std::mutex> lock(mutex);
	active   = true;
	rate     = bytes_per_second;
	deadline = first = last = 0;
	bytes = calls = 0;
	return 0;
}

extern "C" double pcr_boundary_reserve(uint64_t size) {
	std::lock_guard<std::mutex> lock(mutex);
	if (!active) {
		return 0;
	}
	const double started = now();
	if (!calls) {
		first = started;
	}
	bytes += size;
	++calls;
	if (rate > 0) {
		deadline = std::max(started, deadline) + size / rate;
		return deadline;
	}
	return 0;
}

extern "C" void pcr_boundary_finish(double until) {
	if (until > 0) {
		std::this_thread::sleep_until(std::chrono::steady_clock::time_point(
		    std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(until))));
	}
	std::lock_guard<std::mutex> lock(mutex);
	if (active) {
		last = now();
	}
}

extern "C" int pcr_boundary_stats(uint64_t* read_bytes, uint64_t* read_calls, double* span) {
	std::lock_guard<std::mutex> lock(mutex);
	*read_bytes = bytes;
	*read_calls = calls;
	*span       = calls ? last - first : 0;
	return 0;
}
