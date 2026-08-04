// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/memory/device_arena.cu
// ────────────────────────────────────────────────────────
#include "cuda/memory/device_arena.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace galp::memory {

size_t DeviceArena::checked_add(const size_t a, const size_t b, const char* field) {
	if (b > std::numeric_limits<size_t>::max() - a) {
		throw std::overflow_error(field);
	}
	return a + b;
}

size_t DeviceArena::checked_mul(const size_t a, const size_t b, const char* field) {
	if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
		throw std::overflow_error(field);
	}
	return a * b;
}

std::uintptr_t DeviceArena::checked_ptr_end(const std::uintptr_t begin, const size_t bytes, const char* field) {
	if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
		throw std::overflow_error(field);
	}
	return begin + bytes;
}

size_t DeviceArena::align_staged_offset(const size_t value) {
	return checked_add(value, 255U, "DeviceArena staged offset overflow") & ~size_t(255U);
}

DeviceArena::DeviceArena(cudaStream_t stream)
    : stream_(stream) {
}

DeviceArena::~DeviceArena() {
	try {
		release_device_base();
		release_pinned_base();
		run_deferred_frees();
	} catch (const std::exception& e) { std::fprintf(stderr, "DeviceArena destructor: %s\n", e.what()); }
}

void DeviceArena::register_backing(const void* base, size_t bytes, const bool upload) {
	if (base == nullptr || bytes == 0) {
		return;
	}
	(void)checked_ptr_end(reinterpret_cast<std::uintptr_t>(base), bytes, "DeviceArena backing range overflow");
	const auto* b = reinterpret_cast<const std::byte*>(base);
	for (const auto& r : regions_) {
		if (r.base == b && r.bytes == bytes) {
			if (r.upload != upload) {
				throw std::invalid_argument("DeviceArena backing registered with conflicting upload policies");
			}
			return;
		}
	}
	regions_.push_back(BackingRegion {b, bytes, bytes, 0, upload});
}

void DeviceArena::coalesce_backing_regions() {
	if (regions_.size() <= 1) {
		return;
	}

	std::sort(regions_.begin(), regions_.end(), [](const BackingRegion& a, const BackingRegion& b) {
		return reinterpret_cast<std::uintptr_t>(a.base) < reinterpret_cast<std::uintptr_t>(b.base);
	});

	std::vector<BackingRegion> merged;
	merged.reserve(regions_.size());
	for (const auto& region : regions_) {
		if (merged.empty()) {
			merged.push_back(BackingRegion {region.base, region.bytes, region.bytes, 0, region.upload});
			continue;
		}

		auto&      cur          = merged.back();
		const auto cur_begin    = reinterpret_cast<std::uintptr_t>(cur.base);
		const auto cur_end      = checked_ptr_end(cur_begin, cur.bytes, "DeviceArena coalesced range overflow");
		const auto region_begin = reinterpret_cast<std::uintptr_t>(region.base);
		const auto region_end   = checked_ptr_end(region_begin, region.bytes, "DeviceArena backing range overflow");
		if (region_begin <= cur_end && region.upload == cur.upload) {
			if (region_end > cur_end) {
				cur.bytes = static_cast<size_t>(region_end - cur_begin);
			}
			cur.slab_bytes = cur.bytes;
			continue;
		}
		merged.push_back(BackingRegion {region.base, region.bytes, region.bytes, 0, region.upload});
	}

	regions_ = std::move(merged);
	for (auto& e : entries_) {
		if (e.region_idx < 0) {
			continue;
		}
		e.region_idx = find_region(e.host_src, e.copy_bytes);
		if (e.region_idx < 0) {
			continue;
		}
		auto&       region      = regions_[e.region_idx];
		const auto* src         = reinterpret_cast<const std::byte*>(e.host_src);
		e.region_offset         = static_cast<size_t>(src - region.base);
		const size_t end_offset = checked_add(e.region_offset, e.alloc_bytes, "DeviceArena region slab overflow");
		if (end_offset > region.slab_bytes) {
			region.slab_bytes = end_offset;
		}
	}
}

void DeviceArena::add_resolver(std::function<void()> fn) {
	resolvers_.push_back(std::move(fn));
}

void DeviceArena::resolve_to(void** dst, size_t entry_idx) {
	resolver_targets_.push_back(ResolverTarget {dst, entry_idx});
}

void DeviceArena::defer_free(std::function<void()> fn) {
	deferred_frees_.push_back(std::move(fn));
}

void DeviceArena::reset(const bool preserve_capacity) {
	if (!preserve_capacity) {
		release_device_base();
		release_pinned_base();
		minimum_capacity_bytes_ = 0U;
	}
	run_deferred_frees();
	staged_bytes_ = 0;
	entries_.clear();
	staged_entry_indices_.clear();
	regions_.clear();
	resolvers_.clear();
	resolver_targets_.clear();
}

ArenaUploadMetrics DeviceArena::upload(bool resolve_before_pack, bool backing_regions_coalesced) {
	using clock   = std::chrono::steady_clock;
	const auto ms = [](auto a, auto b) {
		return std::chrono::duration<double, std::milli>(b - a).count();
	};
	ArenaUploadMetrics metrics {};
	metrics.device_capacity.minimum_capacity_bytes = minimum_capacity_bytes_;
	metrics.device_capacity.capacity_before_bytes  = capacity_bytes_;
	metrics.device_capacity.capacity_bytes         = capacity_bytes_;
	metrics.pinned_capacity.capacity_before_bytes  = pinned_capacity_bytes_;
	metrics.pinned_capacity.capacity_bytes         = pinned_capacity_bytes_;

	const auto finalize = [&]() {
		run_deferred_frees();
		regions_.clear();
	};

	if (entries_.empty()) {
		run_resolvers();
		finalize();
		return metrics;
	}

	const auto t0 = clock::now();
	if (!backing_regions_coalesced) {
		coalesce_backing_regions();
	}
	const auto plan = plan_layout();
	metrics.device_capacity.requested_bytes = plan.total_bytes;
	if (plan.total_bytes == 0) {
		run_resolvers();
		finalize();
		return metrics;
	}
	const auto t1     = clock::now();
	metrics.layout_ms = ms(t0, t1);

	const size_t device_alloc_bytes =
	    round_up_pow2(std::max(plan.total_bytes, minimum_capacity_bytes_), 65536U);
	if (ensure_capacity(device_alloc_bytes)) {
		metrics.device_capacity.growth_count = 1U;
		metrics.device_capacity.growth_bytes =
		    device_alloc_bytes - std::min(device_alloc_bytes, metrics.device_capacity.capacity_before_bytes);
	}
	metrics.device_capacity.capacity_bytes = capacity_bytes_;
	if (staged_bytes_ > 0) {
		const size_t staged_floor = staged_bytes_ >= 192U * 1024U ? 512U * 1024U : 65536U;
		const size_t pinned_alloc_bytes = round_up_pow2(staged_bytes_, staged_floor);
		metrics.pinned_capacity.requested_bytes = staged_bytes_;
		if (ensure_pinned_capacity(pinned_alloc_bytes)) {
			metrics.pinned_capacity.growth_count = 1U;
			metrics.pinned_capacity.growth_bytes =
			    pinned_alloc_bytes - std::min(pinned_alloc_bytes, metrics.pinned_capacity.capacity_before_bytes);
		}
		metrics.pinned_capacity.capacity_bytes = pinned_capacity_bytes_;
	}
	const auto t2    = clock::now();
	metrics.alloc_ms = ms(t1, t2);

	if (resolve_before_pack) {
		run_resolvers();
	}
	const auto t3      = clock::now();
	metrics.resolve_ms = ms(t2, t3);

	pack_staged_area();
	const auto t4   = clock::now();
	metrics.pack_ms = ms(t3, t4);

	cudaEvent_t dma_start {};
	cudaEvent_t dma_stop {};
	const bool  measure_h2d = should_measure_h2d();
	if (measure_h2d) {
		CUDA_SAFE_CALL(cudaEventCreate(&dma_start));
		CUDA_SAFE_CALL(cudaEventCreate(&dma_stop));
		CUDA_SAFE_CALL(cudaEventRecord(dma_start, stream_));
	}
	const auto dma_stats = issue_dma(plan.staged_device_base);
	if (measure_h2d) {
		CUDA_SAFE_CALL(cudaEventRecord(dma_stop, stream_));
		CUDA_SAFE_CALL(cudaEventSynchronize(dma_stop));
		float gpu_ms = 0.0f;
		CUDA_SAFE_CALL(cudaEventElapsedTime(&gpu_ms, dma_start, dma_stop));
		metrics.dma_gpu_ms = static_cast<double>(gpu_ms);
		CUDA_SAFE_CALL(cudaEventDestroy(dma_start));
		CUDA_SAFE_CALL(cudaEventDestroy(dma_stop));
	}
	const auto t5        = clock::now();
	metrics.dma_issue_ms = ms(t4, t5);
	metrics.dma_bytes    = dma_stats.bytes;
	metrics.dma_count    = dma_stats.count;

	if (!resolve_before_pack) {
		run_resolvers();
	}
	finalize();
	return metrics;
}

size_t DeviceArena::total_bytes() const {
	size_t cursor = 0;
	for (const auto& r : regions_) {
		cursor = (cursor + 255U) & ~size_t(255U);
		cursor += r.slab_bytes;
	}
	cursor = (cursor + 255U) & ~size_t(255U);
	return cursor + staged_bytes_;
}

size_t DeviceArena::entry_count() const {
	return entries_.size();
}

size_t DeviceArena::capacity_bytes() const noexcept {
	return capacity_bytes_;
}

size_t DeviceArena::pinned_capacity_bytes() const noexcept {
	return pinned_capacity_bytes_;
}

void DeviceArena::set_minimum_capacity_bytes(const size_t bytes) {
	minimum_capacity_bytes_ = std::max(minimum_capacity_bytes_, bytes);
}

DeviceArena::LayoutPlan DeviceArena::plan_layout() {
	size_t cursor = 0;
	for (auto& region : regions_) {
		cursor               = align_staged_offset(cursor);
		region.device_offset = cursor;
		cursor               = checked_add(cursor, region.slab_bytes, "DeviceArena layout overflow");
	}
	cursor                          = align_staged_offset(cursor);
	const size_t staged_device_base = cursor;
	cursor                          = checked_add(cursor, staged_bytes_, "DeviceArena layout overflow");

	for (auto& e : entries_) {
		e.device_offset =
		    (e.region_idx >= 0)
		        ? checked_add(regions_[e.region_idx].device_offset,
		                      e.region_offset,
		                      "DeviceArena resolved region offset overflow")
		        : checked_add(staged_device_base, e.staged_offset, "DeviceArena resolved staged offset overflow");
	}
	return LayoutPlan {staged_device_base, cursor};
}

void DeviceArena::pack_staged_area() {
	if (staged_bytes_ == 0) {
		return;
	}
	for (const size_t entry_idx : staged_entry_indices_) {
		if (entry_idx >= entries_.size()) {
			continue;
		}
		const auto& e = entries_[entry_idx];
		if (e.host_src != nullptr && e.copy_bytes > 0) {
			std::memcpy(pinned_base_ + e.staged_offset, e.host_src, e.copy_bytes);
		}
	}
}

DeviceArena::DmaIssueStats DeviceArena::issue_dma(size_t staged_device_base) {
	DmaIssueStats stats {};
	for (const auto& region : regions_) {
		if (region.bytes == 0 || !region.upload) {
			continue;
		}
		stats.bytes += region.bytes;
		++stats.count;
		CUDA_SAFE_CALL(cudaMemcpyAsync(
		    device_base_ + region.device_offset, region.base, region.bytes, cudaMemcpyHostToDevice, stream_));
	}
	if (staged_bytes_ > 0) {
		stats.bytes += staged_bytes_;
		++stats.count;
		CUDA_SAFE_CALL(cudaMemcpyAsync(
		    device_base_ + staged_device_base, pinned_base_, staged_bytes_, cudaMemcpyHostToDevice, stream_));
	}
	DevicePool::instance().register_external_h2d(stream_);
	return stats;
}

void DeviceArena::run_resolvers() {
	for (const auto& t : resolver_targets_) {
		*t.dst = device_base_ + entries_[t.entry_idx].device_offset;
	}
	resolver_targets_.clear();
	for (auto& fn : resolvers_) {
		fn();
	}
	resolvers_.clear();
}

int DeviceArena::find_region(const void* host_src, size_t bytes) const {
	if (host_src == nullptr || bytes == 0 || regions_.empty()) {
		return -1;
	}
	const auto src     = reinterpret_cast<std::uintptr_t>(host_src);
	const auto src_end = checked_ptr_end(src, bytes, "DeviceArena source range overflow");
	for (size_t i = 0; i < regions_.size(); ++i) {
		const auto& r        = regions_[i];
		const auto  base     = reinterpret_cast<std::uintptr_t>(r.base);
		const auto  base_end = checked_ptr_end(base, r.bytes, "DeviceArena backing range overflow");
		if (src >= base && src_end <= base_end) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

bool DeviceArena::ensure_capacity(const size_t alloc_bytes) {
	if (device_base_ != nullptr && capacity_bytes_ >= alloc_bytes) {
		return false;
	}
	release_device_base();
	auto& pool      = DevicePool::instance();
	device_base_    = reinterpret_cast<char*>(pool.alloc_on_stream(alloc_bytes, stream_));
	capacity_bytes_ = alloc_bytes;
	return true;
}

bool DeviceArena::ensure_pinned_capacity(const size_t alloc_bytes) {
	if (pinned_base_ != nullptr && pinned_capacity_bytes_ >= alloc_bytes) {
		return false;
	}
	release_pinned_base();
	auto& pool             = DevicePool::instance();
	pinned_base_           = reinterpret_cast<char*>(pool.alloc_pinned(alloc_bytes));
	pinned_capacity_bytes_ = alloc_bytes;
	return true;
}

void DeviceArena::release_device_base() {
	if (device_base_ == nullptr) {
		return;
	}
	auto& pool = DevicePool::instance();
	pool.release_arena_ptr(device_base_);
	device_base_    = nullptr;
	capacity_bytes_ = 0;
}

void DeviceArena::release_pinned_base() {
	if (pinned_base_ == nullptr) {
		return;
	}
	auto& pool = DevicePool::instance();
	pool.release_pinned(pinned_base_);
	pinned_base_           = nullptr;
	pinned_capacity_bytes_ = 0;
}

void DeviceArena::run_deferred_frees() {
	for (auto& fn : deferred_frees_) {
		fn();
	}
	deferred_frees_.clear();
}

size_t DeviceArena::round_up_pow2(size_t bytes, size_t min_bucket) {
	if (bytes <= min_bucket) {
		return min_bucket;
	}
	size_t v = bytes - 1;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	return v + 1;
}

bool DeviceArena::should_measure_h2d() {
	static const bool enabled = (std::getenv("GALP_MEASURE_H2D") != nullptr);
	return enabled;
}

} // namespace galp::memory
