// Experimental controls for the existing native binary, loaded before importing it.
#include "fls/io/file.hpp"
#include "format/reader.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
std::mutex        rate_mutex;
bool              active           = false;
double            bytes_per_second = 0;
uint64_t          bytes_read = 0, read_calls = 0, compact_calls = 0;
Clock::time_point next_completion, first_request, last_completion;

void* native_symbol(const char* name) {
	static void* library = [] {
		auto* handle = dlopen(std::getenv("FASTLENS_NATIVE_LIBRARY"), RTLD_NOW | RTLD_NOLOAD);
		if (!handle)
			throw std::runtime_error(dlerror());
		return handle;
	}();
	auto* symbol = dlsym(library, name);
	if (!symbol)
		throw std::runtime_error(dlerror());
	return symbol;
}

Clock::time_point reserve(const uint64_t bytes) {
	std::lock_guard<std::mutex> lock(rate_mutex);
	if (!active || bytes == 0)
		return {};
	const auto now = Clock::now();
	if (read_calls++ == 0)
		first_request = now;
	bytes_read += bytes;
	if (bytes_per_second > 0) {
		next_completion = std::max(next_completion, now) +
		                  std::chrono::duration_cast<Clock::duration>(
		                      std::chrono::duration<double>(static_cast<double>(bytes) / bytes_per_second));
	} else {
		next_completion = now;
	}
	return next_completion;
}

void finish(const Clock::time_point deadline) {
	if (deadline == Clock::time_point {})
		return;
	std::this_thread::sleep_until(deadline);
	std::lock_guard<std::mutex> lock(rate_mutex);
	last_completion = std::max(last_completion, Clock::now());
}

uint64_t size_of(std::span<const fastlanes::FileRangeReadTarget> targets) {
	uint64_t bytes = 0;
	for (const auto& target : targets)
		bytes += target.size;
	return bytes;
}
} // namespace

extern "C" void input_boundary_begin(double mb_per_second) {
	std::lock_guard<std::mutex> lock(rate_mutex);
	active           = true;
	bytes_per_second = mb_per_second * 1000000.0;
	bytes_read = read_calls = compact_calls = 0;
	next_completion = first_request = last_completion = {};
}

extern "C" void input_boundary_stats(uint64_t* bytes, uint64_t* calls, uint64_t* compact, double* span) {
	std::lock_guard<std::mutex> lock(rate_mutex);
	*bytes   = bytes_read;
	*calls   = read_calls;
	*compact = compact_calls;
	*span    = read_calls ? std::chrono::duration<double>(last_completion - first_request).count() : 0;
	active   = false;
}

// CPU-only calibration exercises the same shared limiter with concurrent clients.
extern "C" void input_boundary_calibrate(uint64_t bytes, unsigned clients) {
	std::vector<std::thread> threads;
	for (unsigned i = 0; i < clients; ++i) {
		threads.emplace_back([bytes] { finish(reserve(bytes)); });
	}
	for (auto& thread : threads)
		thread.join();
}

namespace fastlanes {
void File::ReadRangeUnchecked(void* dst, n_t offset, n_t size) {
	using Function       = void (*)(File*, void*, n_t, n_t);
	static auto original = reinterpret_cast<Function>(native_symbol("_ZN9fastlanes4File18ReadRangeUncheckedEPvmm"));
	const auto  deadline = m_path.extension() == ".fls" ? reserve(size) : Clock::time_point {};
	original(this, dst, offset, size);
	finish(deadline);
}

void File::ReadRangesUnchecked(std::span<const FileRangeReadTarget> targets) {
	using Function       = void (*)(File*, std::span<const FileRangeReadTarget>);
	static auto original = reinterpret_cast<Function>(native_symbol(
	    "_ZN9fastlanes4File19ReadRangesUncheckedESt4spanIKNS_19FileRangeReadTargetELm18446744073709551615EE"));
	const auto  deadline = m_path.extension() == ".fls" ? reserve(size_of(targets)) : Clock::time_point {};
	original(this, targets);
	finish(deadline);
}

FileRangeBatchReadResult File::ReadRangesIoUringUnchecked(std::span<const FileRangeReadTarget> targets,
                                                          uint32_t                             queue_depth) {
	using Function       = FileRangeBatchReadResult (*)(File*, std::span<const FileRangeReadTarget>, uint32_t);
	static auto original = reinterpret_cast<Function>(native_symbol(
	    "_ZN9fastlanes4File26ReadRangesIoUringUncheckedESt4spanIKNS_19FileRangeReadTargetELm18446744073709551615EEj"));
	const auto  deadline = m_path.extension() == ".fls" ? reserve(size_of(targets)) : Clock::time_point {};
	auto        result   = original(this, targets, queue_depth);
	finish(deadline);
	return result;
}
} // namespace fastlanes

namespace fastlens::format {
std::vector<ZeroCopyRowgroup>
FlsReader::read_rowgroups_zero_copy_compact(const std::vector<const SparseVectorReadPlan*>& plans,
                                            std::vector<ZeroCopyReadTiming>&                timings) {
	using Function = std::vector<ZeroCopyRowgroup> (*)(
	    FlsReader*, const std::vector<const SparseVectorReadPlan*>&, std::vector<ZeroCopyReadTiming>&);
	static auto compact = reinterpret_cast<Function>(
	    native_symbol("_ZN8fastlens6format9FlsReader32read_rowgroups_zero_copy_compactERKSt6vectorIPKNS0_"
	                  "20SparseVectorReadPlanESaIS5_EERS2_INS0_18ZeroCopyReadTimingESaISA_EE"));
	static auto source = reinterpret_cast<Function>(
	    native_symbol("_ZN8fastlens6format9FlsReader33read_rowgroups_zero_copy_compiledERKSt6vectorIPKNS0_"
	                  "20SparseVectorReadPlanESaIS5_EERS2_INS0_18ZeroCopyReadTimingESaISA_EE"));
	{
		std::lock_guard<std::mutex> lock(rate_mutex);
		++compact_calls;
	}
	return (std::getenv("FASTLENS_SOURCE_LAYOUT_CONTROL") ? source : compact)(this, plans, timings);
}
} // namespace fastlens::format
