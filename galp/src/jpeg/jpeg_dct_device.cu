#include "cuda/cuda_macros.cuh"
#include "cuda/launch/launch.cuh"
#include "cuda/memory/gpu_array.cuh"
#include "engine/materialization/metadata.cuh"
#include "engine/operators/rowgroup.cuh"
#include "engine/workset/append.cuh"
#include "engine/workset/upload.cuh"
#include "format/reader.cuh"
#include "jpeg/jpeg_dct_device.cuh"
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace galp::jpeg {

struct JpegDctDeviceBatch::Impl {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::optional<GPUArray<int16_t>>           coefficients;
	size_t                                     coefficient_count = 0;
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	JpegDctDeviceCacheStats                    cache_stats;
};

JpegDctDeviceBatch::JpegDctDeviceBatch() noexcept = default;

JpegDctDeviceBatch::JpegDctDeviceBatch(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

JpegDctDeviceBatch::~JpegDctDeviceBatch() = default;

JpegDctDeviceBatch::JpegDctDeviceBatch(JpegDctDeviceBatch&&) noexcept = default;

JpegDctDeviceBatch& JpegDctDeviceBatch::operator=(JpegDctDeviceBatch&&) noexcept = default;

const int16_t* JpegDctDeviceBatch::device_coefficients() const noexcept {
	return impl_ && impl_->coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->coefficients).get()
	                                                : nullptr;
}

size_t JpegDctDeviceBatch::coefficient_count() const noexcept {
	return impl_ ? impl_->coefficient_count : 0;
}

size_t JpegDctDeviceBatch::coefficient_bytes() const noexcept {
	return coefficient_count() * sizeof(int16_t);
}

size_t JpegDctDeviceBatch::block_count() const noexcept {
	return impl_ ? impl_->block_metadata.size() : 0;
}

size_t JpegDctDeviceBatch::image_count() const noexcept {
	return impl_ ? impl_->image_layouts.size() : 0;
}

size_t JpegDctDeviceBatch::rowgroup_count() const noexcept {
	return impl_ ? impl_->rowgroups.size() : 0;
}

JpegDctDeviceCacheStats JpegDctDeviceBatch::cache_stats() const noexcept {
	return impl_ ? impl_->cache_stats : JpegDctDeviceCacheStats {};
}

JpegDctDeviceLayout JpegDctDeviceBatch::layout() const noexcept {
	return impl_ ? impl_->layout : JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
}

const std::vector<JpegDctDeviceImageLayout>& JpegDctDeviceBatch::image_layouts() const noexcept {
	static const std::vector<JpegDctDeviceImageLayout> empty;
	return impl_ ? impl_->image_layouts : empty;
}

const std::vector<JpegDctDeviceBlockMetadata>& JpegDctDeviceBatch::block_metadata() const noexcept {
	static const std::vector<JpegDctDeviceBlockMetadata> empty;
	return impl_ ? impl_->block_metadata : empty;
}

const std::vector<JpegDctDeviceRowgroupMetadata>& JpegDctDeviceBatch::rowgroups() const noexcept {
	static const std::vector<JpegDctDeviceRowgroupMetadata> empty;
	return impl_ ? impl_->rowgroups : empty;
}

} // namespace galp::jpeg

namespace galp::jpeg::detail {
namespace {

enum class DeviceCoeffSource : uint8_t {
	kMissing = 0,
	kI8      = 1,
	kI16     = 2,
};

struct BoundCoeffColumns {
	std::array<const int8_t*, 64>     columns_i8 {};
	std::array<const int16_t*, 64>    columns_i16 {};
	std::array<DeviceCoeffSource, 64> column_sources {};
};

__global__ void gather_dct_blocks_kernel(const int8_t* const* __restrict columns_i8,
                                         const int16_t* const* __restrict columns_i16,
                                         const DeviceCoeffSource* __restrict column_sources,
                                         const JpegDctDeviceGatherItem* __restrict items,
                                         const size_t item_count,
                                         int16_t* __restrict out) {
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t total  = item_count * 64U;
	if (linear >= total) {
		return;
	}
	const size_t item_idx  = linear / 64U;
	const size_t coeff_idx = linear % 64U;
	const auto   item      = items[item_idx];
	const auto   source    = column_sources[coeff_idx];
	int16_t      value     = 0;
	if (source == DeviceCoeffSource::kI16) {
		value = columns_i16[coeff_idx][item.row_in_rowgroup];
	} else if (source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(columns_i8[coeff_idx][item.row_in_rowgroup]);
	}
	out[item.output_block_index * 64U + coeff_idx] = value;
}

__global__ void materialize_dense_dct_rowgroup_kernel(const int8_t* const* __restrict columns_i8,
                                                      const int16_t* const* __restrict columns_i16,
                                                      const DeviceCoeffSource* __restrict column_sources,
                                                      const uint32_t row_count,
                                                      int16_t* __restrict dense) {
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t total  = static_cast<size_t>(row_count) * 64U;
	if (linear >= total) {
		return;
	}
	const size_t row_idx   = linear / 64U;
	const size_t coeff_idx = linear % 64U;
	const auto   source    = column_sources[coeff_idx];
	int16_t      value     = 0;
	if (source == DeviceCoeffSource::kI16) {
		value = columns_i16[coeff_idx][row_idx];
	} else if (source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(columns_i8[coeff_idx][row_idx]);
	}
	dense[row_idx * 64U + coeff_idx] = value;
}

__global__ void gather_cached_dct_blocks_kernel(const int16_t* __restrict dense,
                                                const JpegDctDeviceGatherItem* __restrict items,
                                                const size_t item_count,
                                                int16_t* __restrict out) {
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t total  = item_count * 64U;
	if (linear >= total) {
		return;
	}
	const size_t item_idx                          = linear / 64U;
	const size_t coeff_idx                         = linear % 64U;
	const auto   item                              = items[item_idx];
	out[item.output_block_index * 64U + coeff_idx] = dense[static_cast<size_t>(item.row_in_rowgroup) * 64U + coeff_idx];
}

BoundCoeffColumns bind_coeff_columns(const galp::execution::Rowgroup&       rowgroup,
                                     const galp::runtime::ExecutionWorkset& workset) {
	if (rowgroup.columns.size() < 64) {
		throw std::runtime_error("JPEG DCT device batch requires at least 64 logical coefficient columns");
	}

	BoundCoeffColumns bound;
	{
		const auto& batch = workset.buffers.host_batches.template get<int8_t>();
		for (size_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto logical_idx = batch.expr_indices[expr_idx];
			if (logical_idx >= bound.columns_i16.size()) {
				continue;
			}
			bound.columns_i8[logical_idx]     = batch.device_exprs[expr_idx].out;
			bound.column_sources[logical_idx] = DeviceCoeffSource::kI8;
		}
	}
	{
		const auto& batch = workset.buffers.host_batches.template get<int16_t>();
		for (size_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto logical_idx = batch.expr_indices[expr_idx];
			if (logical_idx >= bound.columns_i16.size()) {
				continue;
			}
			bound.columns_i16[logical_idx]    = batch.device_exprs[expr_idx].out;
			bound.column_sources[logical_idx] = DeviceCoeffSource::kI16;
		}
	}

	const auto resolve_column = [&](const auto& self, const size_t coeff_idx) -> DeviceCoeffSource {
		if (coeff_idx >= bound.column_sources.size()) {
			throw std::runtime_error("JPEG DCT device batch coefficient alias index is outside 64-column DCT block");
		}
		if (bound.column_sources[coeff_idx] != DeviceCoeffSource::kMissing) {
			return bound.column_sources[coeff_idx];
		}
		const auto& column = rowgroup.columns[coeff_idx];
		if (column.alias_of.has_value()) {
			const auto source               = self(self, *column.alias_of);
			bound.column_sources[coeff_idx] = source;
			if (source == DeviceCoeffSource::kI8) {
				bound.columns_i8[coeff_idx] = bound.columns_i8[*column.alias_of];
			} else if (source == DeviceCoeffSource::kI16) {
				bound.columns_i16[coeff_idx] = bound.columns_i16[*column.alias_of];
			}
			return source;
		}
		throw std::runtime_error("JPEG DCT device batch could not bind a coefficient column to device output");
	};
	for (size_t coeff_idx = 0; coeff_idx < bound.column_sources.size(); ++coeff_idx) {
		(void)resolve_column(resolve_column, coeff_idx);
	}
	return bound;
}

void gather_rowgroup(const std::vector<JpegDctDeviceGatherItem>& items,
                     const galp::execution::Rowgroup&            rowgroup,
                     const galp::runtime::ExecutionWorkset&      workset,
                     int16_t*                                    output) {
	if (items.empty()) {
		return;
	}

	const auto                        bound = bind_coeff_columns(rowgroup, workset);
	GPUArray<const int8_t*>           d_columns_i8(bound.columns_i8.size(), bound.columns_i8.data());
	GPUArray<const int16_t*>          d_columns_i16(bound.columns_i16.size(), bound.columns_i16.data());
	GPUArray<DeviceCoeffSource>       d_column_sources(bound.column_sources.size(), bound.column_sources.data());
	GPUArray<JpegDctDeviceGatherItem> d_items(items.size(), items.data());
	constexpr unsigned                kThreads = 256;
	const size_t                      total    = items.size() * 64U;
	const dim3                        block(kThreads);
	const dim3                        grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
	gather_dct_blocks_kernel<<<grid, block>>>(
	    d_columns_i8.get(), d_columns_i16.get(), d_column_sources.get(), d_items.get(), items.size(), output);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void gather_cached_rowgroup(const std::vector<JpegDctDeviceGatherItem>& items, const int16_t* dense, int16_t* output) {
	if (items.empty()) {
		return;
	}

	GPUArray<JpegDctDeviceGatherItem> d_items(items.size(), items.data());
	constexpr unsigned                kThreads = 256;
	const size_t                      total    = items.size() * 64U;
	const dim3                        block(kThreads);
	const dim3                        grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
	gather_cached_dct_blocks_kernel<<<grid, block>>>(dense, d_items.get(), items.size(), output);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void materialize_dense_rowgroup(const galp::execution::Rowgroup&       rowgroup,
                                const galp::runtime::ExecutionWorkset& workset,
                                int16_t*                               dense) {
	if (rowgroup.n_tuples > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("JPEG DCT rowgroup is too large for device cache materialization");
	}
	const auto row_count = static_cast<uint32_t>(rowgroup.n_tuples);
	if (row_count == 0) {
		return;
	}
	const auto                  bound = bind_coeff_columns(rowgroup, workset);
	GPUArray<const int8_t*>     d_columns_i8(bound.columns_i8.size(), bound.columns_i8.data());
	GPUArray<const int16_t*>    d_columns_i16(bound.columns_i16.size(), bound.columns_i16.data());
	GPUArray<DeviceCoeffSource> d_column_sources(bound.column_sources.size(), bound.column_sources.data());
	constexpr unsigned          kThreads = 256;
	const size_t                total    = static_cast<size_t>(row_count) * 64U;
	const dim3                  block(kThreads);
	const dim3                  grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
	materialize_dense_dct_rowgroup_kernel<<<grid, block>>>(
	    d_columns_i8.get(), d_columns_i16.get(), d_column_sources.get(), row_count, dense);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

size_t decoded_rowgroup_bytes(const size_t rows) {
	return rows * 64U * sizeof(int16_t);
}

void execute_rowgroup_plan(galp::format::FlsReader&           rdr,
                           const uint32_t                     shard_id,
                           const JpegDctDeviceRowgroupPlan&   rowgroup_plan,
                           int16_t*                           output,
                           JpegDctDeviceDecodedRowgroupCache* cache,
                           JpegDctDeviceCacheStats&           batch_cache_stats) {
	const auto key = JpegDctDeviceDecodedRowgroupCacheKey {shard_id, rowgroup_plan.rowgroup_index};
	if (cache != nullptr && cache->capacity > 0) {
		auto it = cache->entries.find(key);
		if (it != cache->entries.end() && it->second->blocks.has_value()) {
			it->second->last_access = ++cache->clock;
			++batch_cache_stats.hits;
			gather_cached_rowgroup(rowgroup_plan.items, it->second->blocks->get(), output);
			return;
		}
		++batch_cache_stats.misses;
	}

	auto zero_copy = rdr.read_rowgroup_zero_copy(rowgroup_plan.rowgroup_index);
	auto rowgroup  = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
	if (rowgroup.columns.size() < 64) {
		galp::execution::free_rowgroup(rowgroup);
		throw std::runtime_error("JPEG DCT FLS rowgroup has fewer than 64 coefficient columns");
	}

	galp::runtime::ExecutionWorkset      workset;
	galp::runtime::ExecutionWorksetGuard guard(workset);
	try {
		galp::execution::ExecutionConfig cfg;
		cfg.write_out = true;
		galp::runtime::append_rowgroup_columns(workset, rowgroup, cfg);
		(void)galp::runtime::upload_workset(workset, cfg);
		galp::runtime::run_workset(workset, 1, cfg);
		if (cache != nullptr && cache->capacity > 0) {
			const auto rowgroup_bytes = decoded_rowgroup_bytes(rowgroup.n_tuples);
			if (rowgroup_bytes <= cache->capacity && rowgroup.n_tuples <= std::numeric_limits<uint32_t>::max()) {
				auto entry         = std::make_unique<JpegDctDeviceDecodedRowgroupCacheEntry>();
				entry->rows        = static_cast<uint32_t>(rowgroup.n_tuples);
				entry->bytes       = rowgroup_bytes;
				entry->last_access = ++cache->clock;
				entry->blocks.emplace(rowgroup.n_tuples * 64U);
				materialize_dense_rowgroup(rowgroup, workset, entry->blocks->get());
				gather_cached_rowgroup(rowgroup_plan.items, entry->blocks->get(), output);
				cache->resident += entry->bytes;
				cache->entries[key] = std::move(entry);
				++batch_cache_stats.inserts;
				while (cache->resident > cache->capacity && !cache->entries.empty()) {
					auto evict_it          = cache->entries.end();
					auto evict_last_access = std::numeric_limits<uint64_t>::max();
					for (auto it = cache->entries.begin(); it != cache->entries.end(); ++it) {
						if (it->first == key) {
							continue;
						}
						if (it->second->last_access < evict_last_access) {
							evict_it          = it;
							evict_last_access = it->second->last_access;
						}
					}
					if (evict_it == cache->entries.end()) {
						break;
					}
					cache->resident -= evict_it->second->bytes;
					cache->entries.erase(evict_it);
					++batch_cache_stats.evictions;
				}
			} else {
				gather_rowgroup(rowgroup_plan.items, rowgroup, workset, output);
			}
		} else {
			gather_rowgroup(rowgroup_plan.items, rowgroup, workset, output);
		}
	} catch (...) {
		galp::execution::free_rowgroup(rowgroup);
		throw;
	}
	galp::execution::free_rowgroup(rowgroup);
}

} // namespace

void JpegDctDeviceDecodedRowgroupCache::set_capacity(const size_t bytes) {
	capacity = bytes;
	while (resident > capacity && !entries.empty()) {
		auto evict_it          = entries.end();
		auto evict_last_access = std::numeric_limits<uint64_t>::max();
		for (auto it = entries.begin(); it != entries.end(); ++it) {
			if (it->second->last_access < evict_last_access) {
				evict_it          = it;
				evict_last_access = it->second->last_access;
			}
		}
		if (evict_it == entries.end()) {
			break;
		}
		resident -= evict_it->second->bytes;
		entries.erase(evict_it);
	}
	if (capacity == 0) {
		clear();
	}
}

void JpegDctDeviceDecodedRowgroupCache::clear() {
	entries.clear();
	resident = 0;
	clock    = 0;
}

size_t JpegDctDeviceDecodedRowgroupCache::capacity_bytes() const noexcept {
	return capacity;
}

size_t JpegDctDeviceDecodedRowgroupCache::resident_bytes() const noexcept {
	return resident;
}

size_t JpegDctDeviceDecodedRowgroupCache::resident_rowgroups() const noexcept {
	return entries.size();
}

JpegDctDeviceBatch execute_jpeg_dct_device_batch_plan(JpegDctDeviceBatchPlan plan) {
	auto impl               = std::make_unique<JpegDctDeviceBatch::Impl>();
	impl->layout            = plan.layout;
	impl->image_layouts     = std::move(plan.image_layouts);
	impl->block_metadata    = std::move(plan.block_metadata);
	impl->rowgroups         = std::move(plan.rowgroups);
	impl->coefficient_count = impl->block_metadata.size() * 64U;
	if (impl->coefficient_count != 0) {
		impl->coefficients.emplace(impl->coefficient_count);
	}

	int16_t* output = impl->coefficients.has_value() ? impl->coefficients->get() : nullptr;
	for (const auto& shard : plan.shards) {
		galp::format::FlsReader rdr(shard.fls_path, /*load_column_names=*/false);
		for (const auto& rowgroup : shard.rowgroups) {
			execute_rowgroup_plan(rdr, shard.shard_id, rowgroup, output, plan.cache, impl->cache_stats);
		}
	}
	if (plan.cache != nullptr) {
		impl->cache_stats.capacity_bytes     = plan.cache->capacity_bytes();
		impl->cache_stats.resident_bytes     = plan.cache->resident_bytes();
		impl->cache_stats.resident_rowgroups = plan.cache->resident_rowgroups();
	}
	return JpegDctDeviceBatch(std::move(impl));
}

} // namespace galp::jpeg::detail
