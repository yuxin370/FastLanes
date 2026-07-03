#include "codecs/consts.cuh"
#include "cuda/cuda_macros.cuh"
#include "cuda/launch/launch.cuh"
#include "cuda/memory/cuda_raii.cuh"
#include "cuda/memory/gpu_array.cuh"
#include "engine/materialization/metadata.cuh"
#include "engine/operators/rowgroup.cuh"
#include "engine/pipeline/rowgroup_prefetch_queue.cuh"
#include "engine/workset/append.cuh"
#include "engine/workset/upload.cuh"
#include "format/reader.cuh"
#include "jpeg/jpeg_dct_device.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace galp::jpeg {

struct JpegDctDeviceBatch::Impl {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::optional<GPUArray<int16_t>>           coefficients;
	size_t                                     coefficient_count = 0;
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	JpegDctDeviceCacheStats                    cache_stats;
	JpegDctDeviceExecutionStats                execution_stats;
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

JpegDctDeviceExecutionStats JpegDctDeviceBatch::execution_stats() const noexcept {
	return impl_ ? impl_->execution_stats : JpegDctDeviceExecutionStats {};
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

using Clock                                       = std::chrono::steady_clock;
constexpr size_t kMinJpegDctScratchBufferCapacity = 1024;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

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

struct DeviceCoeffBinding {
	const int8_t*     column_i8  = nullptr;
	const int16_t*    column_i16 = nullptr;
	DeviceCoeffSource source     = DeviceCoeffSource::kMissing;
};

struct JpegDctDeviceGatherBatchItem {
	uint32_t source_index       = 0;
	uint32_t row_in_rowgroup    = 0;
	uint64_t output_block_index = 0;
};

struct JpegDctDeviceMaterializeBatchItem {
	uint32_t source_index = 0;
	uint32_t row_count    = 0;
	int16_t* dense        = nullptr;
};

struct JpegDctDeviceCachedGatherBatchItem {
	const int16_t* dense              = nullptr;
	uint32_t       row_in_rowgroup    = 0;
	uint64_t       output_block_index = 0;
};

struct DecodedRowgroupWork {
	uint32_t                                                shard_id       = 0;
	uint32_t                                                rowgroup_index = 0;
	galp::execution::Rowgroup                               rowgroup {};
	std::vector<uint32_t>                                   owned_selected_vectors;
	const std::vector<uint32_t>*                            selected_vectors = nullptr;
	std::vector<JpegDctDeviceGatherItem>                    owned_gather_items;
	const std::vector<JpegDctDeviceGatherItem>*             gather_items            = nullptr;
	size_t                                                  expr_index_base         = 0;
	uint32_t                                                decode_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	bool                                                    decodes_full_rowgroup   = false;
	bool                                                    owns_rowgroup           = false;
	JpegDctDeviceDecodedRowgroupCacheKey                    cache_key {};
	std::unique_ptr<JpegDctDeviceDecodedRowgroupCacheEntry> cache_entry;

	DecodedRowgroupWork()                                      = default;
	DecodedRowgroupWork(const DecodedRowgroupWork&)            = delete;
	DecodedRowgroupWork& operator=(const DecodedRowgroupWork&) = delete;

	DecodedRowgroupWork(DecodedRowgroupWork&& other) noexcept
	    : shard_id(other.shard_id)
	    , rowgroup_index(other.rowgroup_index)
	    , rowgroup(std::move(other.rowgroup))
	    , owned_selected_vectors(std::move(other.owned_selected_vectors))
	    , selected_vectors(other.selected_vectors == &other.owned_selected_vectors ? &owned_selected_vectors
	                                                                               : other.selected_vectors)
	    , owned_gather_items(std::move(other.owned_gather_items))
	    , gather_items(other.gather_items == &other.owned_gather_items ? &owned_gather_items : other.gather_items)
	    , expr_index_base(other.expr_index_base)
	    , decode_unpack_n_vectors(other.decode_unpack_n_vectors)
	    , decodes_full_rowgroup(other.decodes_full_rowgroup)
	    , owns_rowgroup(std::exchange(other.owns_rowgroup, false))
	    , cache_key(other.cache_key)
	    , cache_entry(std::move(other.cache_entry)) {
	}

	DecodedRowgroupWork& operator=(DecodedRowgroupWork&& other) noexcept = delete;

	~DecodedRowgroupWork() {
		reset();
	}

	void reset() {
		if (owns_rowgroup) {
			galp::execution::free_rowgroup(rowgroup);
			owns_rowgroup = false;
		}
	}
};

template <typename T>
struct JpegDctDeviceScratchBuffer {
	T*     data     = nullptr;
	size_t capacity = 0;

	JpegDctDeviceScratchBuffer()                                             = default;
	JpegDctDeviceScratchBuffer(const JpegDctDeviceScratchBuffer&)            = delete;
	JpegDctDeviceScratchBuffer& operator=(const JpegDctDeviceScratchBuffer&) = delete;

	~JpegDctDeviceScratchBuffer() {
		release();
	}

	void release() noexcept {
		try {
			if (data != nullptr) {
				galp::memory::device_free(data);
				data     = nullptr;
				capacity = 0;
			}
		} catch (const std::exception& e) {
			std::fprintf(stderr, "JpegDctDeviceScratchBuffer destructor: %s\n", e.what());
		}
	}

	static size_t growth_capacity(const size_t count) {
		size_t next = kMinJpegDctScratchBufferCapacity;
		while (next < count) {
			if (next > std::numeric_limits<size_t>::max() / 2U) {
				return count;
			}
			next *= 2U;
		}
		return next;
	}

	[[nodiscard]] bool needs_reallocation(const size_t count) const noexcept {
		return count > capacity;
	}

	void upload(const T* host, const size_t count, cudaStream_t stream, JpegDctDeviceExecutionStats& stats) {
		if (count == 0) {
			return;
		}
		if (count > capacity) {
			if (data != nullptr) {
				galp::memory::device_free(data);
			}
			const size_t alloc_count = growth_capacity(count);
			if (alloc_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
				throw std::overflow_error("JPEG DCT scratch buffer allocation size overflow");
			}
			data     = stream != nullptr
			               ? reinterpret_cast<T*>(galp::memory::device_malloc_on_stream(alloc_count * sizeof(T), stream))
			               : reinterpret_cast<T*>(galp::memory::device_malloc(alloc_count * sizeof(T)));
			capacity = alloc_count;
			++stats.scratch_allocation_count;
		}
		galp::memory::device_memcpy_h2d_async(data, host, count * sizeof(T), stream);
		++stats.scratch_upload_count;
	}
};

struct JpegDctDeviceScratch {
	galp::memory::CudaStream                                       cache_hit_stream;
	galp::memory::CudaEvent                                        cached_gather_start;
	galp::memory::CudaEvent                                        cached_gather_done;
	galp::memory::CudaEvent                                        decoded_batch_gather_done;
	galp::runtime::ExecutionWorkset                                decode_workset;
	JpegDctDeviceScratchBuffer<DeviceCoeffBinding>                 column_bindings;
	JpegDctDeviceScratchBuffer<JpegDctDeviceGatherBatchItem>       batch_gather_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceMaterializeBatchItem>  batch_materialize_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceCachedGatherBatchItem> cached_gather_items;
	std::vector<DeviceCoeffBinding>                                host_column_bindings;
	std::vector<BoundCoeffColumns>                                 host_bound_sources;
	std::vector<JpegDctDeviceGatherBatchItem>                      host_gather_items;
	std::vector<JpegDctDeviceMaterializeBatchItem>                 host_materialize_items;
	std::vector<JpegDctDeviceCachedGatherBatchItem>                host_cached_gather_items;
	std::deque<std::vector<JpegDctDeviceCachedGatherBatchItem>>    host_cached_gather_uploads;
	std::vector<DecodedRowgroupWork>                               host_pending_works;
	bool                                                           cached_gather_in_flight = false;

	~JpegDctDeviceScratch() {
		try {
			if (cached_gather_in_flight) {
				cached_gather_done.synchronize();
				if (cache_hit_stream) {
					galp::memory::complete_h2d(cache_hit_stream.get());
				}
				cached_gather_in_flight = false;
			}
			// Free device scratch buffers while their allocation streams are still alive.
			// The workset release below may destroy the stream these were allocated on;
			// freeing after that would pass a stale stream to cudaFreeAsync.
			column_bindings.release();
			batch_gather_items.release();
			batch_materialize_items.release();
			cached_gather_items.release();
			galp::runtime::release_workset(
			    decode_workset, /*preserve_resources=*/false, /*h2d_already_complete=*/false);
		} catch (const std::exception& e) { std::fprintf(stderr, "JpegDctDeviceScratch cleanup: %s\n", e.what()); }
	}

	cudaStream_t stream_for_cache_hit() {
		if (!cache_hit_stream) {
			cache_hit_stream.create(cudaStreamNonBlocking);
		}
		return cache_hit_stream.get();
	}

	void ensure_cached_gather_events() {
		cached_gather_start.create();
		cached_gather_done.create();
	}

	void ensure_decoded_batch_events() {
		decoded_batch_gather_done.create();
	}
};

void JpegDctDeviceScratchDeleter::operator()(JpegDctDeviceScratch* scratch) const noexcept {
	delete scratch;
}

JpegDctDeviceScratchPtr make_jpeg_dct_device_scratch() {
	return JpegDctDeviceScratchPtr(new JpegDctDeviceScratch());
}

namespace {

__global__ void gather_dct_blocks_batch_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                               const JpegDctDeviceGatherBatchItem* __restrict items,
                                               const size_t item_count,
                                               int16_t* __restrict out) {
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t total  = item_count * 64U;
	if (linear >= total) {
		return;
	}
	const size_t item_idx     = linear / 64U;
	const size_t coeff_idx    = linear % 64U;
	const auto   item         = items[item_idx];
	const size_t source_coeff = static_cast<size_t>(item.source_index) * 64U + coeff_idx;
	const auto   binding      = column_bindings[source_coeff];
	int16_t      value        = 0;
	if (binding.source == DeviceCoeffSource::kI16) {
		value = binding.column_i16[item.row_in_rowgroup];
	} else if (binding.source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(binding.column_i8[item.row_in_rowgroup]);
	}
	out[item.output_block_index * 64U + coeff_idx] = value;
}

__global__ void materialize_dense_dct_rowgroup_batch_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                                            const JpegDctDeviceMaterializeBatchItem* __restrict items,
                                                            const size_t item_count) {
	const size_t item_idx = static_cast<size_t>(blockIdx.y);
	if (item_idx >= item_count) {
		return;
	}
	const auto   item   = items[item_idx];
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t total  = static_cast<size_t>(item.row_count) * 64U;
	if (linear >= total) {
		return;
	}
	const size_t local_row    = linear / 64U;
	const size_t coeff_idx    = linear % 64U;
	const size_t source_coeff = static_cast<size_t>(item.source_index) * 64U + coeff_idx;
	const auto   binding      = column_bindings[source_coeff];
	int16_t      value        = 0;
	if (binding.source == DeviceCoeffSource::kI16) {
		value = binding.column_i16[local_row];
	} else if (binding.source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(binding.column_i8[local_row]);
	}
	item.dense[local_row * 64U + coeff_idx] = value;
}

__global__ void gather_cached_dct_blocks_batch_kernel(const JpegDctDeviceCachedGatherBatchItem* __restrict items,
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
	out[item.output_block_index * 64U + coeff_idx] =
	    item.dense[static_cast<size_t>(item.row_in_rowgroup) * 64U + coeff_idx];
}

BoundCoeffColumns bind_coeff_columns(const galp::execution::Rowgroup&       rowgroup,
                                     const galp::runtime::ExecutionWorkset& workset,
                                     const size_t                           expr_index_base = 0) {
	if (rowgroup.columns.size() < 64) {
		throw std::runtime_error("JPEG DCT device batch requires at least 64 logical coefficient columns");
	}

	BoundCoeffColumns bound;
	{
		const auto& batch = workset.buffers.host_batches.template get<int8_t>();
		for (size_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto materialize_idx = batch.expr_indices[expr_idx];
			if (materialize_idx < expr_index_base || materialize_idx >= expr_index_base + bound.columns_i16.size()) {
				continue;
			}
			const auto logical_idx            = materialize_idx - expr_index_base;
			bound.columns_i8[logical_idx]     = batch.device_exprs[expr_idx].out;
			bound.column_sources[logical_idx] = DeviceCoeffSource::kI8;
		}
	}
	{
		const auto& batch = workset.buffers.host_batches.template get<int16_t>();
		for (size_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto materialize_idx = batch.expr_indices[expr_idx];
			if (materialize_idx < expr_index_base || materialize_idx >= expr_index_base + bound.columns_i16.size()) {
				continue;
			}
			const auto logical_idx            = materialize_idx - expr_index_base;
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

void gather_cached_rowgroups_batch(const std::vector<JpegDctDeviceCachedGatherBatchItem>& items,
                                   int16_t*                                               output,
                                   JpegDctDeviceScratch&                                  scratch,
                                   JpegDctDeviceExecutionStats&                           stats,
                                   cudaStream_t                                           stream) {
	if (items.empty()) {
		return;
	}

	scratch.cached_gather_items.upload(items.data(), items.size(), stream, stats);
	constexpr unsigned kThreads = 256;
	const size_t       total    = items.size() * 64U;
	const dim3         block(kThreads);
	const dim3         grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
	gather_cached_dct_blocks_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.cached_gather_items.data, items.size(), output);
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.gather_kernel_launch_count;
	++stats.cached_gather_kernel_launch_count;
}

void gather_decoded_rowgroup_batch(const std::vector<BoundCoeffColumns>&            sources,
                                   const std::vector<JpegDctDeviceGatherBatchItem>& items,
                                   int16_t*                                         output,
                                   JpegDctDeviceScratch&                            scratch,
                                   JpegDctDeviceExecutionStats&                     stats,
                                   cudaStream_t                                     stream) {
	if (sources.empty() || items.empty()) {
		return;
	}

	auto& column_bindings = scratch.host_column_bindings;
	column_bindings.clear();
	column_bindings.reserve(sources.size() * 64U);
	for (const auto& source : sources) {
		for (size_t coeff_idx = 0; coeff_idx < source.column_sources.size(); ++coeff_idx) {
			column_bindings.push_back(DeviceCoeffBinding {
			    source.columns_i8[coeff_idx], source.columns_i16[coeff_idx], source.column_sources[coeff_idx]});
		}
	}

	constexpr unsigned kThreads = 256;
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.batch_gather_items.upload(items.data(), items.size(), stream, stats);
	const size_t total = items.size() * 64U;
	const dim3   block(kThreads);
	const dim3   grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
	gather_dct_blocks_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.column_bindings.data, scratch.batch_gather_items.data, items.size(), output);
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.gather_kernel_launch_count;
}

void materialize_dense_rowgroup_batch(const std::vector<JpegDctDeviceMaterializeBatchItem>& items,
                                      JpegDctDeviceScratch&                                 scratch,
                                      JpegDctDeviceExecutionStats&                          stats,
                                      cudaStream_t                                          stream) {
	if (items.empty()) {
		return;
	}
	// Reuses the column pointer/source scratch uploaded for the preceding batch gather on this stream.
	constexpr unsigned kThreads = 256;
	scratch.batch_materialize_items.upload(items.data(), items.size(), stream, stats);
	uint32_t max_rows = 0;
	for (const auto& item : items) {
		max_rows = std::max(max_rows, item.row_count);
	}
	if (max_rows == 0) {
		return;
	}
	const size_t total = static_cast<size_t>(max_rows) * 64U;
	const dim3   block(kThreads);
	const dim3   grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads), static_cast<unsigned>(items.size()));
	materialize_dense_dct_rowgroup_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.column_bindings.data, scratch.batch_materialize_items.data, items.size());
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
}

size_t decoded_rowgroup_bytes(const size_t rows) {
	return rows * 64U * sizeof(int16_t);
}

void refresh_runtime_policy_summary(JpegDctDeviceExecutionStats& stats) {
	if (stats.runtime_policy_selected_rowgroups == 0 && stats.runtime_policy_full_rowgroups == 0) {
		stats.runtime_policy_decision = "none";
	} else if (stats.runtime_policy_selected_rowgroups != 0 && stats.runtime_policy_full_rowgroups != 0) {
		stats.runtime_policy_decision = "mixed";
	} else if (stats.runtime_policy_selected_rowgroups != 0) {
		stats.runtime_policy_decision = "selected-vector";
	} else {
		stats.runtime_policy_decision = "full-rowgroup";
	}
	stats.runtime_policy_reason = "selected=" + std::to_string(stats.runtime_policy_selected_rowgroups) +
	                              ",full=" + std::to_string(stats.runtime_policy_full_rowgroups) +
	                              ",tail_full=" + std::to_string(stats.runtime_policy_tail_full_rowgroups) +
	                              ",ratio_full=" + std::to_string(stats.runtime_policy_ratio_full_rowgroups) +
	                              ",low_saving_full=" + std::to_string(stats.runtime_policy_low_saving_full_rowgroups) +
	                              ",max_selected_ratio=" + std::to_string(kMaxSelectedVectorRatioForPushdown) +
	                              ",min_saved_vectors=" + std::to_string(kMinSavedVectorsForPushdown);
}

void record_runtime_policy(const JpegDctRuntimePolicyResult policy, JpegDctDeviceExecutionStats& stats) {
	if (policy.decision == JpegDctRuntimePolicyDecision::kSelectedVectors) {
		++stats.runtime_policy_selected_rowgroups;
	} else {
		++stats.runtime_policy_full_rowgroups;
	}
	switch (policy.reason) {
	case JpegDctRuntimePolicyReason::kCropSavesEnoughVectors:
		break;
	case JpegDctRuntimePolicyReason::kTailChunkWouldOverrun:
		++stats.runtime_policy_tail_full_rowgroups;
		break;
	case JpegDctRuntimePolicyReason::kSelectedCoversMostVectors:
		++stats.runtime_policy_ratio_full_rowgroups;
		break;
	case JpegDctRuntimePolicyReason::kSavingsTooSmall:
		++stats.runtime_policy_low_saving_full_rowgroups;
		break;
	}
}

void append_jpeg_rowgroup_columns(galp::runtime::ExecutionWorkset&        workset,
                                  const galp::execution::Rowgroup&        rowgroup,
                                  const galp::execution::ExecutionConfig& cfg,
                                  const size_t                            expr_index_base,
                                  const std::vector<uint32_t>*            selected_vectors) {
	workset.outputs.required = workset.outputs.required || cfg.write_out;
	galp::runtime::begin_workset_chunk_arena(workset, rowgroup.columns.size());
	auto* active_chunk_arena = workset.buffers.chunk_arena.get();

	const void* last_backing_base  = nullptr;
	size_t      last_backing_bytes = 0;
	for (size_t coeff_idx = 0; coeff_idx < rowgroup.columns.size(); ++coeff_idx) {
		const auto& column = rowgroup.columns[coeff_idx];
		if (column.skip_decompress) {
			continue;
		}
		if (galp::runtime::has_pinned_backing(column) &&
		    (column.backing_base != last_backing_base || column.backing_bytes != last_backing_bytes)) {
			active_chunk_arena->register_backing(column.backing_base, column.backing_bytes);
			last_backing_base  = column.backing_base;
			last_backing_bytes = column.backing_bytes;
		}
		const uint32_t selected_vector_width = std::max(1U, cfg.unpack_n_vectors);
		galp::runtime::append_column_to_workset(workset,
		                                        column,
		                                        cfg,
		                                        expr_index_base + coeff_idx,
		                                        *active_chunk_arena,
		                                        nullptr,
		                                        /*emit_typed_work_items=*/true,
		                                        /*register_backing=*/false,
		                                        selected_vectors,
		                                        selected_vector_width);
	}
}

DecodedRowgroupWork prepare_decoded_rowgroup_work_from_materialized(galp::execution::Rowgroup          rowgroup,
                                                                    uint32_t                           shard_id,
                                                                    const JpegDctDeviceRowgroupPlan&   rowgroup_plan,
                                                                    JpegDctDeviceDecodedRowgroupCache* cache,
                                                                    JpegDctDeviceExecutionStats&       execution_stats);

DecodedRowgroupWork prepare_decoded_rowgroup_work(galp::format::FlsReader&           rdr,
                                                  const uint32_t                     shard_id,
                                                  const JpegDctDeviceRowgroupPlan&   rowgroup_plan,
                                                  JpegDctDeviceDecodedRowgroupCache* cache,
                                                  JpegDctDeviceExecutionStats&       execution_stats) {
	const auto sync_read_start = Clock::now();
	auto       zero_copy       = rdr.read_rowgroup_zero_copy(rowgroup_plan.rowgroup_index);
	auto       rowgroup        = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
	const auto sync_read_end   = Clock::now();
	execution_stats.sync_rowgroup_read_ms += elapsed_ms(sync_read_start, sync_read_end);

	return prepare_decoded_rowgroup_work_from_materialized(
	    std::move(rowgroup), shard_id, rowgroup_plan, cache, execution_stats);
}

DecodedRowgroupWork prepare_decoded_rowgroup_work_from_materialized(galp::execution::Rowgroup          rowgroup,
                                                                    const uint32_t                     shard_id,
                                                                    const JpegDctDeviceRowgroupPlan&   rowgroup_plan,
                                                                    JpegDctDeviceDecodedRowgroupCache* cache,
                                                                    JpegDctDeviceExecutionStats& execution_stats) {
	DecodedRowgroupWork work;
	work.shard_id       = shard_id;
	work.rowgroup_index = rowgroup_plan.rowgroup_index;
	work.cache_key      = JpegDctDeviceDecodedRowgroupCacheKey {shard_id, rowgroup_plan.rowgroup_index};
	work.rowgroup       = std::move(rowgroup);
	work.owns_rowgroup  = true;
	if (work.rowgroup.columns.size() < 64) {
		throw std::runtime_error("JPEG DCT FLS rowgroup has fewer than 64 coefficient columns");
	}

	galp::execution::ExecutionConfig cfg;
	cfg.write_out                                    = true;
	cfg.unpack_n_vectors                             = kJpegDctDeviceUnpackNVectors;
	size_t                     selected_vector_count = 0;
	JpegDctRuntimePolicyResult policy {};
	if (rowgroup_plan.has_vector_plan && rowgroup_plan.full_vector_count == work.rowgroup.n_vecs) {
		work.selected_vectors = &rowgroup_plan.selected_vectors;
		selected_vector_count = rowgroup_plan.selected_vector_count;
		policy                = rowgroup_plan.runtime_policy;
	} else {
		work.owned_selected_vectors =
		    selected_decode_vectors(rowgroup_plan.items, work.rowgroup.n_vecs, cfg.unpack_n_vectors);
		work.selected_vectors = &work.owned_selected_vectors;
		const bool selected_chunks_fit =
		    selected_decode_chunks_fit(*work.selected_vectors, work.rowgroup.n_vecs, cfg.unpack_n_vectors);
		selected_vector_count =
		    selected_decode_vector_count(*work.selected_vectors, work.rowgroup.n_vecs, cfg.unpack_n_vectors);
		policy = choose_jpeg_dct_runtime_policy(selected_vector_count, work.rowgroup.n_vecs, selected_chunks_fit);
	}
	record_runtime_policy(policy, execution_stats);
	work.decodes_full_rowgroup = policy.decision == JpegDctRuntimePolicyDecision::kFullRowgroup;
	if (work.decodes_full_rowgroup && work.rowgroup.n_vecs % kJpegDctDeviceUnpackNVectors != 0) {
		work.decode_unpack_n_vectors = 1;
	}
	if (work.decodes_full_rowgroup) {
		work.gather_items = &rowgroup_plan.items;
	} else if (rowgroup_plan.has_vector_plan && !rowgroup_plan.selected_gather_items.empty()) {
		work.gather_items = &rowgroup_plan.selected_gather_items;
	} else {
		work.owned_gather_items =
		    remap_items_to_selected_vectors(rowgroup_plan.items, *work.selected_vectors, cfg.unpack_n_vectors);
		work.gather_items = &work.owned_gather_items;
	}
	const size_t actual_selected_vector_count =
	    work.decodes_full_rowgroup ? work.rowgroup.n_vecs : selected_vector_count;
	execution_stats.selected_vector_count += actual_selected_vector_count;

	if (cache != nullptr && cache->capacity > 0 && work.decodes_full_rowgroup) {
		const auto rowgroup_bytes = decoded_rowgroup_bytes(work.rowgroup.n_tuples);
		if (rowgroup_bytes <= cache->capacity && work.rowgroup.n_tuples <= std::numeric_limits<uint32_t>::max()) {
			work.cache_entry              = std::make_unique<JpegDctDeviceDecodedRowgroupCacheEntry>();
			work.cache_entry->rows        = static_cast<uint32_t>(work.rowgroup.n_tuples);
			work.cache_entry->bytes       = rowgroup_bytes;
			work.cache_entry->last_access = ++cache->clock;
			work.cache_entry->blocks.emplace(work.rowgroup.n_tuples * 64U);
		}
	}
	return work;
}

void finish_cached_gather_after_wait(JpegDctDeviceScratch& scratch, JpegDctDeviceExecutionStats& execution_stats) {
	if (!scratch.cached_gather_in_flight) {
		return;
	}
	const auto elapsed_ms = static_cast<double>(scratch.cached_gather_done.elapsed_since(scratch.cached_gather_start));
	execution_stats.cached_gather_ms += elapsed_ms;
	execution_stats.gather_ms += elapsed_ms;
	if (scratch.cache_hit_stream) {
		galp::memory::complete_h2d(scratch.cache_hit_stream.get());
	}
	scratch.host_cached_gather_uploads.clear();
	scratch.cached_gather_in_flight = false;
}

void drain_cached_gather(JpegDctDeviceScratch& scratch, JpegDctDeviceExecutionStats& execution_stats) {
	if (!scratch.cached_gather_in_flight) {
		return;
	}
	// Drain only when cached-gather scratch must be reused or a legacy default-stream caller needs host ordering.
	scratch.cached_gather_done.synchronize();
	++execution_stats.internal_sync_count;
	++execution_stats.cached_gather_sync_count;
	finish_cached_gather_after_wait(scratch, execution_stats);
}

void make_stream_wait_for_cached_gather(cudaStream_t                 stream,
                                        JpegDctDeviceScratch&        scratch,
                                        JpegDctDeviceExecutionStats& execution_stats) {
	if (!scratch.cached_gather_in_flight) {
		return;
	}
	if (stream == nullptr) {
		drain_cached_gather(scratch, execution_stats);
		return;
	}
	CUDA_SAFE_CALL(cudaStreamWaitEvent(stream, scratch.cached_gather_done.get(), 0));
	++execution_stats.cached_gather_event_handoff_count;
}

void finish_workset_run_after_stream_tail_wait(galp::runtime::AsyncWorksetRun& run) {
	if (!run.active) {
		return;
	}
	if (run.queued != nullptr && run.start != nullptr) {
		run.pre_kernel_event_ms = static_cast<double>(run.start->elapsed_since(*run.queued));
	}
	if (run.start != nullptr && run.stop != nullptr) {
		run.elapsed_ms = static_cast<double>(run.stop->elapsed_since(*run.start));
	}
	run.queued = nullptr;
	run.start  = nullptr;
	run.stop   = nullptr;
	run.active = false;
}

void release_completed_workset(galp::runtime::ExecutionWorkset&      workset,
                               galp::runtime::ExecutionWorksetGuard& guard,
                               const bool                            preserve_resources) {
	galp::runtime::release_workset(workset, preserve_resources, true);
	guard.dismiss();
}

void execute_decoded_rowgroup_batch(std::vector<DecodedRowgroupWork>&  works,
                                    int16_t*                           output,
                                    JpegDctDeviceDecodedRowgroupCache* cache,
                                    JpegDctDeviceCacheStats&           batch_cache_stats,
                                    JpegDctDeviceExecutionStats&       execution_stats,
                                    JpegDctDeviceScratch&              scratch) {
	if (works.empty()) {
		return;
	}

	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = works.front().decode_unpack_n_vectors;
	for (const auto& work : works) {
		if (work.decode_unpack_n_vectors != cfg.unpack_n_vectors) {
			throw std::runtime_error("JPEG DCT decode batch mixed incompatible unpack widths");
		}
	}
	auto&                                workset = scratch.decode_workset;
	galp::runtime::ExecutionWorksetGuard guard(workset);
	const auto                           build_start = Clock::now();
	galp::runtime::reserve_batch_expr_storage(workset, works.size() * 64U);
	for (size_t idx = 0; idx < works.size(); ++idx) {
		auto& work           = works[idx];
		work.expr_index_base = idx * 64U;
		append_jpeg_rowgroup_columns(workset,
		                             work.rowgroup,
		                             cfg,
		                             work.expr_index_base,
		                             work.decodes_full_rowgroup ? nullptr : work.selected_vectors);
	}
	const auto build_end = Clock::now();
	execution_stats.workset_build_ms += elapsed_ms(build_start, build_end);
	const auto upload_start = Clock::now();
	(void)galp::runtime::upload_workset(workset, cfg);
	const auto upload_end = Clock::now();
	execution_stats.workset_upload_ms += elapsed_ms(upload_start, upload_end);
	++execution_stats.workset_count;
	++execution_stats.workset_upload_count;
	size_t launches = 0;
	auto   run      = galp::runtime::run_workset_async(workset, 1, cfg, nullptr, &launches);
	execution_stats.decode_kernel_launch_count += launches;
	const cudaStream_t stream = run.stream;
	make_stream_wait_for_cached_gather(stream, scratch, execution_stats);

	auto& sources     = scratch.host_bound_sources;
	auto& batch_items = scratch.host_gather_items;
	sources.clear();
	batch_items.clear();
	sources.reserve(works.size());
	size_t total_gather_items = 0;
	for (const auto& work : works) {
		total_gather_items += work.gather_items == nullptr ? 0U : work.gather_items->size();
	}
	batch_items.reserve(total_gather_items);
	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& work = works[source_idx];
		sources.push_back(bind_coeff_columns(work.rowgroup, workset, work.expr_index_base));
		if (work.gather_items == nullptr) {
			continue;
		}
		for (const auto& item : *work.gather_items) {
			batch_items.push_back(JpegDctDeviceGatherBatchItem {
			    static_cast<uint32_t>(source_idx), item.row_in_rowgroup, item.output_block_index});
		}
	}
	execution_stats.gather_item_count += batch_items.size();
	execution_stats.decoded_gather_item_count += batch_items.size();
	gather_decoded_rowgroup_batch(sources, batch_items, output, scratch, execution_stats, stream);
	if (cache != nullptr && cache->capacity > 0) {
		auto& materialize_items = scratch.host_materialize_items;
		materialize_items.clear();
		materialize_items.reserve(works.size());
		for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
			auto& work = works[source_idx];
			if (work.cache_entry) {
				if (work.rowgroup.n_tuples > std::numeric_limits<uint32_t>::max()) {
					throw std::runtime_error("JPEG DCT rowgroup is too large for device cache materialization");
				}
				const auto row_count = static_cast<uint32_t>(work.rowgroup.n_tuples);
				materialize_items.push_back(JpegDctDeviceMaterializeBatchItem {
				    static_cast<uint32_t>(source_idx), row_count, work.cache_entry->blocks->get()});
			}
		}
		materialize_dense_rowgroup_batch(materialize_items, scratch, execution_stats, stream);
	}
	scratch.ensure_decoded_batch_events();
	scratch.decoded_batch_gather_done.record(stream);

	// Rowgroup metadata, workset output arena, and scratch are reused after this batch.
	// Wait only for the gather completion event; batch-level workset ownership can remove this later.
	scratch.decoded_batch_gather_done.synchronize();
	++execution_stats.internal_sync_count;
	++execution_stats.decoded_batch_sync_count;
	if (stream != nullptr) {
		galp::memory::complete_h2d(stream);
	}
	finish_cached_gather_after_wait(scratch, execution_stats);
	if (run.stop != nullptr) {
		const auto elapsed_ms = static_cast<double>(scratch.decoded_batch_gather_done.elapsed_since(*run.stop));
		execution_stats.decoded_gather_ms += elapsed_ms;
		execution_stats.gather_ms += elapsed_ms;
	}
	finish_workset_run_after_stream_tail_wait(run);
	execution_stats.decode_ms += run.elapsed_ms;

	if (cache != nullptr && cache->capacity > 0) {
		for (auto& work : works) {
			cache->insert_ready_entry(work.cache_key, std::move(work.cache_entry), batch_cache_stats);
		}
	}
	works.clear();
	release_completed_workset(workset, guard, /*preserve_resources=*/true);
}

void execute_cached_rowgroup_hits(const std::vector<JpegDctDeviceCachedGatherBatchItem>& items,
                                  int16_t*                                               output,
                                  JpegDctDeviceExecutionStats&                           execution_stats,
                                  JpegDctDeviceScratch&                                  scratch) {
	if (items.empty()) {
		return;
	}
	// Cached gathers use one stream. If the item scratch already has capacity, the next upload is
	// stream-ordered after the previous gather and does not need a host wait. Growing the buffer may
	// free the old device pointer, so that path still drains first.
	if (scratch.cached_gather_in_flight && scratch.cached_gather_items.needs_reallocation(items.size())) {
		drain_cached_gather(scratch, execution_stats);
	}
	auto& upload_items = scratch.host_cached_gather_uploads.emplace_back(items.begin(), items.end());
	execution_stats.gather_item_count += items.size();
	execution_stats.cached_gather_item_count += items.size();
	const cudaStream_t stream = scratch.stream_for_cache_hit();
	scratch.ensure_cached_gather_events();
	if (!scratch.cached_gather_in_flight) {
		scratch.cached_gather_start.record(stream);
	}
	gather_cached_rowgroups_batch(upload_items, output, scratch, execution_stats, stream);
	scratch.cached_gather_done.record(stream);
	scratch.cached_gather_in_flight = true;
}

void execute_shard_plan(const std::shared_ptr<galp::format::FlsReader>&  rdr,
                        const JpegDctDeviceShardPlan&                    shard,
                        int16_t*                                         output,
                        JpegDctDeviceDecodedRowgroupCache*               cache,
                        JpegDctDeviceCacheStats&                         batch_cache_stats,
                        JpegDctDeviceExecutionStats&                     execution_stats,
                        JpegDctDeviceScratch&                            scratch,
                        const size_t                                     decode_batch_rowgroups,
                        const JpegDctDeviceRowgroupPrefetchConfig&       rowgroup_prefetch,
                        std::vector<JpegDctDeviceCachedGatherBatchItem>& cached_pending) {
	if (!rdr) {
		throw std::runtime_error("execute_shard_plan: reader is null");
	}
	auto& pending = scratch.host_pending_works;
	pending.clear();
	const size_t effective_decode_batch_rowgroups =
	    decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : decode_batch_rowgroups;
	pending.reserve(std::min(effective_decode_batch_rowgroups, shard.rowgroups.size()));
	bool pending_may_insert_cache = false;
	auto prefetch_plan =
	    plan_jpeg_dct_rowgroup_prefetch(shard, cache, rowgroup_prefetch, effective_decode_batch_rowgroups);
	std::unique_ptr<galp::runtime::RowgroupPrefetchQueue> prefetch_queue;
	if (prefetch_plan.enabled) {
		const size_t scheduled_rowgroups  = prefetch_plan.rowgroup_indices.size();
		const auto   prefetch_queue_start = Clock::now();
		prefetch_queue                    = std::make_unique<galp::runtime::RowgroupPrefetchQueue>(
            rdr, std::move(prefetch_plan.rowgroup_indices), rowgroup_prefetch.depth, rowgroup_prefetch.workers);
		const auto prefetch_queue_end = Clock::now();
		execution_stats.prefetch_queue_start_ms += elapsed_ms(prefetch_queue_start, prefetch_queue_end);
		execution_stats.prefetched_rowgroup_count += scheduled_rowgroups;
	}
	execution_stats.prefetch_initial_cache_hit_rowgroup_count += prefetch_plan.initial_cache_hit_rowgroup_count;
	execution_stats.prefetch_candidate_rowgroup_count += prefetch_plan.candidate_rowgroup_count;
	execution_stats.prefetch_active_shard_count += prefetch_plan.enabled ? 1U : 0U;
	execution_stats.prefetch_config_disabled_shard_count += prefetch_plan.disabled_by_config ? 1U : 0U;
	execution_stats.prefetch_all_hit_shard_count += prefetch_plan.disabled_by_all_hits ? 1U : 0U;
	execution_stats.prefetch_small_batch_disabled_shard_count += prefetch_plan.disabled_by_small_batch_count ? 1U : 0U;
	execution_stats.prefetch_selected_vector_disabled_shard_count +=
	    prefetch_plan.disabled_by_selected_vector_miss ? 1U : 0U;
	execution_stats.prefetch_selected_vector_miss_rowgroup_count += prefetch_plan.selected_vector_miss_rowgroup_count;
	execution_stats.prefetch_skipped_repeated_rowgroup_count += prefetch_plan.skipped_repeated_rowgroup_count;

	const auto flush_cached = [&]() {
		execute_cached_rowgroup_hits(cached_pending, output, execution_stats, scratch);
		cached_pending.clear();
	};
	const auto flush_pending = [&]() {
		// Pending dense materialization can evict cached dense buffers referenced by cached_pending.
		// Launch cached gathers first; the decoded stream will wait on their completion event.
		if (pending_may_insert_cache) {
			flush_cached();
		}
		execute_decoded_rowgroup_batch(pending, output, cache, batch_cache_stats, execution_stats, scratch);
		pending_may_insert_cache = false;
	};
	const auto pop_prefetched_rowgroup = [&](const JpegDctDeviceRowgroupPlan& rowgroup_plan,
	                                         const bool                       consumed_as_hit = false) {
		auto result = prefetch_queue->pop();
		if (result.rowgroup_index != rowgroup_plan.rowgroup_index) {
			galp::execution::free_rowgroup(result.rowgroup);
			throw std::runtime_error("JPEG DCT rowgroup prefetch queue returned an unexpected rowgroup");
		}
		execution_stats.prefetch_rowgroup_read_ms += result.timing.read_ms;
		execution_stats.prefetch_depth_block_ms += result.prefetch.depth_block_ms;
		const auto ready_push = result.timing.timeline.ready_push;
		const auto wait_start = result.timing.timeline.consumer_wait_start;
		const auto wait_end   = result.timing.timeline.consumer_wait_end;
		double     wait_ms    = 0.0;
		if (wait_start != Clock::time_point {} && wait_end != Clock::time_point {} && wait_start <= wait_end) {
			wait_ms = elapsed_ms(wait_start, wait_end);
		}
		if (consumed_as_hit) {
			execution_stats.prefetch_consumed_as_hit_read_ms += result.timing.read_ms;
			execution_stats.prefetch_consumed_as_hit_wait_ms += wait_ms;
		}
		if (ready_push != Clock::time_point {} && wait_start != Clock::time_point {} && ready_push < wait_start) {
			execution_stats.prefetch_ready_ahead_ms += elapsed_ms(ready_push, wait_start);
		}
		return std::move(result.rowgroup);
	};

	for (size_t rowgroup_pos = 0; rowgroup_pos < shard.rowgroups.size(); ++rowgroup_pos) {
		const auto& rowgroup_plan = shard.rowgroups[rowgroup_pos];
		const auto  key           = JpegDctDeviceDecodedRowgroupCacheKey {shard.shard_id, rowgroup_plan.rowgroup_index};
		if (cache != nullptr && cache->capacity > 0) {
			auto it = cache->entries.find(key);
			if (it != cache->entries.end() && it->second->blocks.has_value()) {
				// Pending selected-vector decodes do not materialize dense cache entries, so they cannot
				// evict this hit. Keep accumulating them across cache hits to avoid tiny worksets.
				if (pending_may_insert_cache) {
					flush_pending();
				}
				it = cache->entries.find(key);
				if (it != cache->entries.end() && it->second->blocks.has_value()) {
					it->second->last_access = ++cache->clock;
					++batch_cache_stats.hits;
					const auto* dense = it->second->blocks->get();
					cached_pending.reserve(cached_pending.size() + rowgroup_plan.items.size());
					for (const auto& item : rowgroup_plan.items) {
						cached_pending.push_back(
						    JpegDctDeviceCachedGatherBatchItem {dense, item.row_in_rowgroup, item.output_block_index});
					}
					if (prefetch_plan.use_prefetch_for_position[rowgroup_pos]) {
						auto rowgroup = pop_prefetched_rowgroup(rowgroup_plan, /*consumed_as_hit=*/true);
						galp::execution::free_rowgroup(rowgroup);
						++execution_stats.prefetch_consumed_as_hit_count;
					}
					continue;
				}
			}
			++batch_cache_stats.misses;
			execution_stats.prefetch_initial_hit_runtime_miss_count +=
			    prefetch_plan.initial_cache_hit_for_position[rowgroup_pos] ? 1U : 0U;
			execution_stats.prefetch_skipped_repeated_runtime_miss_count +=
			    prefetch_plan.skipped_repeated_for_position[rowgroup_pos] ? 1U : 0U;
		}

		auto work =
		    prefetch_plan.use_prefetch_for_position[rowgroup_pos]
		        ? prepare_decoded_rowgroup_work_from_materialized(
		              pop_prefetched_rowgroup(rowgroup_plan), shard.shard_id, rowgroup_plan, cache, execution_stats)
		        : prepare_decoded_rowgroup_work(*rdr, shard.shard_id, rowgroup_plan, cache, execution_stats);
		if (!pending.empty() && pending.front().decode_unpack_n_vectors != work.decode_unpack_n_vectors) {
			flush_pending();
		}
		pending_may_insert_cache = pending_may_insert_cache || static_cast<bool>(work.cache_entry);
		pending.push_back(std::move(work));
		if (pending.size() >= effective_decode_batch_rowgroups) {
			flush_pending();
		}
	}
	flush_pending();
	if (prefetch_queue) {
		execution_stats.prefetch_wait_ms += prefetch_queue->wait_ms();
	}
}

} // namespace

void JpegDctDeviceDecodedRowgroupCache::insert_ready_entry(
    const JpegDctDeviceDecodedRowgroupCacheKey&             key,
    std::unique_ptr<JpegDctDeviceDecodedRowgroupCacheEntry> entry,
    JpegDctDeviceCacheStats&                                batch_cache_stats) {
	if (!entry || capacity == 0) {
		return;
	}
	const auto existing = entries.find(key);
	if (existing != entries.end()) {
		resident = existing->second->bytes <= resident ? resident - existing->second->bytes : 0;
	}
	resident += entry->bytes;
	entries[key] = std::move(entry);
	++batch_cache_stats.inserts;
	while (resident > capacity && !entries.empty()) {
		auto evict_it          = entries.end();
		auto evict_last_access = std::numeric_limits<uint64_t>::max();
		for (auto it = entries.begin(); it != entries.end(); ++it) {
			if (it->first == key) {
				continue;
			}
			if (it->second->last_access < evict_last_access) {
				evict_it          = it;
				evict_last_access = it->second->last_access;
			}
		}
		if (evict_it == entries.end()) {
			break;
		}
		resident = evict_it->second->bytes <= resident ? resident - evict_it->second->bytes : 0;
		entries.erase(evict_it);
		++batch_cache_stats.evictions;
	}
}

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
	auto impl                                           = std::make_unique<JpegDctDeviceBatch::Impl>();
	impl->layout                                        = plan.layout;
	impl->image_layouts                                 = std::move(plan.image_layouts);
	impl->block_metadata                                = std::move(plan.block_metadata);
	impl->rowgroups                                     = std::move(plan.rowgroups);
	impl->execution_stats.planning_ms                   = plan.planning_ms;
	impl->execution_stats.rowgroup_count                = impl->rowgroups.size();
	impl->execution_stats.planned_selected_vector_count = plan.planned_selected_vector_count;
	impl->execution_stats.full_vector_count             = plan.full_vector_count;
	impl->execution_stats.planned_saved_vector_count    = plan.planned_saved_vector_count;
	impl->coefficient_count                             = impl->block_metadata.size() * 64U;
	if (impl->coefficient_count != 0) {
		impl->coefficients.emplace(impl->coefficient_count);
	}

	int16_t*             output = impl->coefficients.has_value() ? impl->coefficients->get() : nullptr;
	JpegDctDeviceScratch local_scratch;
	auto&                scratch        = plan.scratch != nullptr ? *plan.scratch : local_scratch;
	auto&                cached_pending = scratch.host_cached_gather_items;
	cached_pending.clear();
	for (const auto& shard : plan.shards) {
		auto rdr = std::make_shared<galp::format::FlsReader>(shard.fls_path, /*load_column_names=*/false);
		execute_shard_plan(rdr,
		                   shard,
		                   output,
		                   plan.cache,
		                   impl->cache_stats,
		                   impl->execution_stats,
		                   scratch,
		                   plan.decode_batch_rowgroups,
		                   plan.rowgroup_prefetch,
		                   cached_pending);
	}
	execute_cached_rowgroup_hits(cached_pending, output, impl->execution_stats, scratch);
	drain_cached_gather(scratch, impl->execution_stats);
	cached_pending.clear();
	if (plan.cache != nullptr) {
		impl->cache_stats.capacity_bytes     = plan.cache->capacity_bytes();
		impl->cache_stats.resident_bytes     = plan.cache->resident_bytes();
		impl->cache_stats.resident_rowgroups = plan.cache->resident_rowgroups();
	}
	refresh_runtime_policy_summary(impl->execution_stats);
	impl->execution_stats.actual_saved_vector_count =
	    impl->execution_stats.full_vector_count >= impl->execution_stats.selected_vector_count
	        ? impl->execution_stats.full_vector_count - impl->execution_stats.selected_vector_count
	        : 0;
	return JpegDctDeviceBatch(std::move(impl));
}

} // namespace galp::jpeg::detail
