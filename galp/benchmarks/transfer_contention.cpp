// Host submission benchmark. No decode kernels or production policy changes.
#include "cuda/memory/device_pool.cuh"
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

using namespace galp::memory;
using Clock             = std::chrono::steady_clock;
constexpr size_t WINDOW = 8;

struct Result {
	std::vector<double>   latency_us;
	diagnostics::Counters counters;
};

void measure(int threads, int devices, bool shared_stream, bool pinned, size_t bytes, int rounds, int repeat) {
	auto& pool = DevicePool::instance();
	pool.set_use_async(false);
	pool.set_use_pinned(pinned);
	pool.set_small_copy_threshold(0);
	std::vector<cudaStream_t> streams(static_cast<size_t>(shared_stream ? devices : threads));
	for (size_t i = 0; i < streams.size(); ++i) {
		CUDA_SAFE_CALL(cudaSetDevice(static_cast<int>(i) % devices));
		CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
	}
	std::barrier                     gate(threads + 1);
	std::vector<std::future<Result>> jobs;
	for (int index = 0; index < threads; ++index) {
		jobs.push_back(std::async(std::launch::async, [&, index] {
			CUDA_SAFE_CALL(cudaSetDevice(index % devices));
			const auto                 stream = streams[static_cast<size_t>(shared_stream ? index % devices : index)];
			std::vector<unsigned char> source(bytes, static_cast<unsigned char>(index + 1));
			std::array<void*, WINDOW>  targets;
			for (auto& target : targets)
				target = pool.alloc(bytes);
			// Warm context, event and staging allocation paths before measurement.
			for (auto target : targets)
				pool.copy_h2d_on_stream(target, source.data(), bytes, stream);
			pool.sync_h2d(stream);
			Result result;
			result.latency_us.reserve(static_cast<size_t>(rounds) * WINDOW);
			diagnostics::counters = {};
			gate.arrive_and_wait();
			for (int round = 0; round < rounds; ++round) {
				for (auto target : targets) {
					const auto begin = Clock::now();
					pool.copy_h2d_on_stream(target, source.data(), bytes, stream);
					result.latency_us.push_back(static_cast<double>(diagnostics::elapsed(begin)) / 1000.0);
				}
				pool.sync_h2d(stream);
			}
			result.counters = diagnostics::counters;
			gate.arrive_and_wait();
			std::vector<unsigned char> actual(bytes);
			for (auto target : targets) {
				CUDA_SAFE_CALL(cudaMemcpy(actual.data(), target, bytes, cudaMemcpyDeviceToHost));
				if (actual != source)
					throw std::runtime_error("H2D roundtrip mismatch");
				pool.free(target);
			}
			return result;
		}));
	}
	gate.arrive_and_wait();
	const auto begin = Clock::now();
	gate.arrive_and_wait();
	const double          seconds = static_cast<double>(diagnostics::elapsed(begin)) / 1e9;
	std::vector<double>   samples;
	diagnostics::Counters total;
	for (auto& job : jobs) {
		auto result = job.get();
		samples.insert(samples.end(), result.latency_us.begin(), result.latency_us.end());
		for (size_t i = 0; i < diagnostics::LockCount; ++i)
			total.wait_ns[i] += result.counters.wait_ns[i];
		total.sync_wait_ns += result.counters.sync_wait_ns;
	}
	std::sort(samples.begin(), samples.end());
	const double count = static_cast<double>(samples.size());
	std::printf("submit,%d,%d,%d,%d,%d,%zu,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
	            repeat,
	            threads,
	            devices,
	            shared_stream,
	            pinned,
	            bytes,
	            samples[samples.size() / 2],
	            samples[samples.size() * 95 / 100],
	            count / seconds,
	            static_cast<double>(total.wait_ns[diagnostics::Tracker]) / count / 1000,
	            static_cast<double>(total.wait_ns[diagnostics::Device]) / count / 1000,
	            static_cast<double>(total.wait_ns[diagnostics::Pinned]) / count / 1000,
	            static_cast<double>(total.sync_wait_ns) / count / 1000,
	            count * static_cast<double>(bytes) / seconds / 1e9);
	for (size_t i = 0; i < streams.size(); ++i) {
		CUDA_SAFE_CALL(cudaSetDevice(static_cast<int>(i) % devices));
		CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]));
	}
	if (pool.stats().in_use_bytes || pool.pinned_stats().in_use_bytes)
		throw std::runtime_error("pool not idle");
	pool.release_cached();
}

// CUDA-event calibration of actual H2D, without tracker or host memcpy.
void h2d_calibration(size_t bytes) {
	CUDA_SAFE_CALL(cudaSetDevice(0));
	void *host = nullptr, *device = nullptr;
	CUDA_SAFE_CALL(cudaMallocHost(&host, bytes));
	CUDA_SAFE_CALL(cudaMalloc(&device, bytes));
	std::memset(host, 7, bytes);
	cudaEvent_t begin, end;
	CUDA_SAFE_CALL(cudaEventCreate(&begin));
	CUDA_SAFE_CALL(cudaEventCreate(&end));
	CUDA_SAFE_CALL(cudaEventRecord(begin));
	for (int i = 0; i < 100; ++i)
		CUDA_SAFE_CALL(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaEventRecord(end));
	CUDA_SAFE_CALL(cudaEventSynchronize(end));
	float ms = 0;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, begin, end));
	std::printf(
	    "raw_h2d,bytes=%zu,event_us=%.3f,GBps=%.3f\n", bytes, ms * 10, static_cast<double>(bytes) * 100 / ms / 1e6);
	CUDA_SAFE_CALL(cudaEventDestroy(begin));
	CUDA_SAFE_CALL(cudaEventDestroy(end));
	CUDA_SAFE_CALL(cudaFreeHost(host));
	CUDA_SAFE_CALL(cudaFree(device));
}

void sync_interference() {
	CUDA_SAFE_CALL(cudaSetDevice(0));
	TransferTracker tracker;
	cudaStream_t    a, b;
	CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
	CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
	std::atomic<bool> callback_started = false;
	CUDA_SAFE_CALL(cudaLaunchHostFunc(
	    a,
	    [](void* pointer) {
		    static_cast<std::atomic<bool>*>(pointer)->store(true);
		    std::this_thread::sleep_for(std::chrono::milliseconds(100));
	    },
	    &callback_started));
	tracker.submit(a, [](void*&) {});
	std::atomic<bool> sync_started = false;
	auto              waiter       = std::async(std::launch::async, [&] {
        CUDA_SAFE_CALL(cudaSetDevice(0));
        sync_started = true;
        tracker.sync_stream(a, {});
    });
	while (!callback_started || !sync_started)
		std::this_thread::yield();
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	diagnostics::counters = {};
	const auto begin      = Clock::now();
	tracker.submit(b, [](void*&) {});
	const auto submit_us = static_cast<double>(diagnostics::elapsed(begin)) / 1000.0;
	std::printf("sync_interference,submit_us=%.3f,tracker_wait_us=%.3f\n",
	            submit_us,
	            static_cast<double>(diagnostics::counters.wait_ns[diagnostics::Tracker]) / 1000.0);
	waiter.get();
	tracker.sync_all({});
	CUDA_SAFE_CALL(cudaStreamDestroy(a));
	CUDA_SAFE_CALL(cudaStreamDestroy(b));
}

int main(int argc, char** argv) {
	try {
		const size_t bytes  = argc > 1 ? std::stoull(argv[1]) : 1024 * 1024;
		const int    rounds = argc > 2 ? std::stoi(argv[2]) : 16;
		if (!bytes || rounds <= 0)
			throw std::invalid_argument("positive bytes and rounds required");
		int devices = 0, runtime = 0, driver = 0;
		CUDA_SAFE_CALL(cudaGetDeviceCount(&devices));
		CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtime));
		CUDA_SAFE_CALL(cudaDriverGetVersion(&driver));
		if (!devices)
			throw std::runtime_error("CUDA device required");
		std::printf("environment,devices=%d,runtime=%d,driver=%d\n", devices, runtime, driver);
		h2d_calibration(bytes);
		sync_interference();
		std::puts("kind,repeat,threads,devices,shared_stream,pinned,bytes,p50_us,p95_us,ops_s,tracker_wait_us_op,"
		          "device_wait_us_op,pinned_wait_us_op,sync_us_op,wall_GBps");
		for (int repeat = 0; repeat < 3; ++repeat)
			for (int gpus : {1, 2}) {
				if (devices < gpus)
					continue;
				for (bool shared : {true, false})
					for (bool pinned : {false, true})
						for (int threads : {1, 2, 4, 8}) {
							if (threads < gpus)
								continue;
							measure(threads, gpus, shared, pinned, bytes, rounds, repeat);
						}
			}
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s\n", error.what());
		return 1;
	}
}
