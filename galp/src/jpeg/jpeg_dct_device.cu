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
#include <cmath>
#include <cstdio>
#include <deque>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace galp::jpeg {

struct JpegDctDeviceBatch::Impl {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	int                                        cuda_device = -1;
	std::optional<GPUArray<int16_t>>           coefficients;
	std::optional<GPUArray<int16_t>>           y_coefficients;
	std::optional<GPUArray<int16_t>>           cbcr_coefficients;
	std::optional<GPUArray<float>>             y_accum;
	std::optional<GPUArray<float>>             cbcr_accum;
	std::optional<GPUArray<uint16_t>>          fixed_quant_tables;
	std::optional<GPUArray<float>>             fixed_resize_weight_matrices;
	size_t                                     coefficient_count      = 0;
	size_t                                     y_coefficient_count    = 0;
	size_t                                     cbcr_coefficient_count = 0;
	size_t                                     coefficients_per_block = 64;
	JpegDctGridOutputDataType                  grid_output_data_type  = JpegDctGridOutputDataType::kInt16;
	JpegDctYcbcrDctGridShape            ycbcr_dct_grid_shape {};
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	std::vector<uint8_t>                       selected_coefficients;
	JpegDctDeviceCacheStats                    cache_stats;
	JpegDctDeviceExecutionStats                execution_stats;
	galp::memory::CudaEvent                    fixed_grid_round_start_event;
	galp::memory::CudaEvent                    completion_event;
	bool                                       completion_synchronized           = false;
	bool                                       fixed_grid_round_timing_finalized = false;

	void finalize_fixed_grid_round_timing() {
		if (fixed_grid_round_timing_finalized || !fixed_grid_round_start_event || !completion_event) {
			return;
		}
		execution_stats.fixed_grid_round_ms +=
		    static_cast<double>(completion_event.elapsed_since(fixed_grid_round_start_event));
		fixed_grid_round_timing_finalized = true;
	}

	void synchronize_completion() {
		if (completion_event && !completion_synchronized) {
			completion_event.synchronize();
			completion_synchronized = true;
		}
		finalize_fixed_grid_round_timing();
	}

	void synchronize_completion_noexcept() noexcept {
		try {
			synchronize_completion();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "JpegDctDeviceBatch completion wait failed: %s\n", e.what());
		}
	}
};

JpegDctDeviceBatch::JpegDctDeviceBatch() noexcept = default;

JpegDctDeviceBatch::JpegDctDeviceBatch(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

JpegDctDeviceBatch::~JpegDctDeviceBatch() {
	if (impl_) {
		impl_->synchronize_completion_noexcept();
	}
}

JpegDctDeviceBatch::JpegDctDeviceBatch(JpegDctDeviceBatch&&) noexcept = default;

JpegDctDeviceBatch& JpegDctDeviceBatch::operator=(JpegDctDeviceBatch&& other) noexcept {
	if (this != &other) {
		if (impl_) {
			impl_->synchronize_completion_noexcept();
		}
		impl_ = std::move(other.impl_);
	}
	return *this;
}

const int16_t* JpegDctDeviceBatch::device_coefficients() const noexcept {
	if (impl_) {
		impl_->synchronize_completion_noexcept();
	}
	return impl_ && impl_->coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->coefficients).get()
	                                                : nullptr;
}

const int16_t* JpegDctDeviceBatch::y_coefficients() const noexcept {
	if (impl_) {
		impl_->synchronize_completion_noexcept();
	}
	return impl_ && impl_->y_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->y_coefficients).get()
	                                                  : nullptr;
}

const int16_t* JpegDctDeviceBatch::cbcr_coefficients() const noexcept {
	if (impl_) {
		impl_->synchronize_completion_noexcept();
	}
	return impl_ && impl_->cbcr_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->cbcr_coefficients).get()
	                                                     : nullptr;
}

const int16_t* JpegDctDeviceBatch::device_coefficients_async() const noexcept {
	return impl_ && impl_->coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->coefficients).get()
	                                                : nullptr;
}

const int16_t* JpegDctDeviceBatch::y_coefficients_async() const noexcept {
	return impl_ && impl_->y_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->y_coefficients).get()
	                                                  : nullptr;
}

const int16_t* JpegDctDeviceBatch::cbcr_coefficients_async() const noexcept {
	return impl_ && impl_->cbcr_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->cbcr_coefficients).get()
	                                                     : nullptr;
}

const float* JpegDctDeviceBatch::y_float_coefficients() const noexcept {
	if (impl_) {
		impl_->synchronize_completion_noexcept();
	}
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->y_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->y_accum).get()
	           : nullptr;
}

const float* JpegDctDeviceBatch::cbcr_float_coefficients() const noexcept {
	if (impl_) {
		impl_->synchronize_completion_noexcept();
	}
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->cbcr_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->cbcr_accum).get()
	           : nullptr;
}

const float* JpegDctDeviceBatch::y_float_coefficients_async() const noexcept {
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->y_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->y_accum).get()
	           : nullptr;
}

const float* JpegDctDeviceBatch::cbcr_float_coefficients_async() const noexcept {
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->cbcr_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->cbcr_accum).get()
	           : nullptr;
}

JpegDctGridOutputDataType JpegDctDeviceBatch::grid_output_data_type() const noexcept {
	return impl_ ? impl_->grid_output_data_type : JpegDctGridOutputDataType::kInt16;
}

void JpegDctDeviceBatch::synchronize() const {
	if (impl_) {
		impl_->synchronize_completion();
	}
}

size_t JpegDctDeviceBatch::coefficient_count() const noexcept {
	return impl_ ? impl_->coefficient_count : 0;
}

size_t JpegDctDeviceBatch::coefficient_bytes() const noexcept {
	return coefficient_count() * sizeof(int16_t);
}

size_t JpegDctDeviceBatch::y_coefficient_count() const noexcept {
	return impl_ ? impl_->y_coefficient_count : 0;
}

size_t JpegDctDeviceBatch::cbcr_coefficient_count() const noexcept {
	return impl_ ? impl_->cbcr_coefficient_count : 0;
}

size_t JpegDctDeviceBatch::coefficients_per_block() const noexcept {
	return impl_ ? impl_->coefficients_per_block : 64;
}

size_t JpegDctDeviceBatch::block_count() const noexcept {
	if (!impl_) {
		return 0;
	}
	if (!impl_->block_metadata.empty() || impl_->image_layouts.empty()) {
		return impl_->block_metadata.size();
	}
	const auto& last = impl_->image_layouts.back();
	return static_cast<size_t>(last.block_offset) + last.block_count;
}

size_t JpegDctDeviceBatch::image_count() const noexcept {
	return impl_ ? impl_->image_layouts.size() : 0;
}

size_t JpegDctDeviceBatch::rowgroup_count() const noexcept {
	return impl_ ? impl_->rowgroups.size() : 0;
}

int JpegDctDeviceBatch::cuda_device() const noexcept {
	return impl_ ? impl_->cuda_device : -1;
}

JpegDctDeviceCacheStats JpegDctDeviceBatch::cache_stats() const noexcept {
	return cache_stats_ref();
}

JpegDctDeviceExecutionStats JpegDctDeviceBatch::execution_stats() const noexcept {
	if (impl_) {
		// GPU stage durations are only available once their timing events complete.
		// Keep submission asynchronous and pay this wait only when callers request
		// the complete statistics snapshot.
		impl_->synchronize_completion_noexcept();
	}
	return execution_stats_ref();
}

const JpegDctDeviceCacheStats& JpegDctDeviceBatch::cache_stats_ref() const noexcept {
	static const JpegDctDeviceCacheStats empty;
	return impl_ ? impl_->cache_stats : empty;
}

const JpegDctDeviceExecutionStats& JpegDctDeviceBatch::execution_stats_ref() const noexcept {
	static const JpegDctDeviceExecutionStats empty;
	return impl_ ? impl_->execution_stats : empty;
}

JpegDctDeviceLayout JpegDctDeviceBatch::layout() const noexcept {
	return impl_ ? impl_->layout : JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
}

void* JpegDctDeviceBatch::cuda_completion_event() const noexcept {
	return impl_ && impl_->completion_event ? impl_->completion_event.get() : nullptr;
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

const std::vector<uint8_t>& JpegDctDeviceBatch::selected_coefficients() const noexcept {
	static const std::vector<uint8_t> empty;
	return impl_ ? impl_->selected_coefficients : empty;
}

JpegDctYcbcrDctGridShape JpegDctDeviceBatch::ycbcr_dct_grid_shape() const noexcept {
	return impl_ ? impl_->ycbcr_dct_grid_shape : JpegDctYcbcrDctGridShape {};
}

} // namespace galp::jpeg

namespace galp::jpeg::detail {

using Clock                                       = std::chrono::steady_clock;
constexpr size_t kMinJpegDctScratchBufferCapacity = 1024;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

size_t resolve_physical_coefficient_column(const galp::execution::Rowgroup& rowgroup,
                                           const size_t                     logical_coeff_idx) {
	if (logical_coeff_idx >= kJpegDctCoefficientCount || logical_coeff_idx >= rowgroup.columns.size()) {
		throw std::out_of_range("JPEG DCT selected coefficient is outside the rowgroup column range");
	}

	std::array<bool, kJpegDctCoefficientCount> visited {};
	size_t                                     current = logical_coeff_idx;
	for (;;) {
		if (current >= kJpegDctCoefficientCount || current >= rowgroup.columns.size()) {
			throw std::out_of_range("JPEG DCT coefficient alias target is outside the rowgroup column range");
		}
		if (visited[current]) {
			throw std::runtime_error("JPEG DCT coefficient alias cycle detected");
		}
		visited[current] = true;
		const auto& column = rowgroup.columns[current];
		if (column.alias_of.has_value()) {
			current = *column.alias_of;
			continue;
		}
		if (column.skip_decompress) {
			throw std::runtime_error("JPEG DCT coefficient is marked skip_decompress without an alias source");
		}
		return current;
	}
}

JpegDctDeviceResolvedProjection
resolve_projection_physical_columns(const galp::execution::Rowgroup&               rowgroup,
                                    const std::vector<JpegDctDeviceProjectionItem>& projection_items) {
	JpegDctDeviceResolvedProjection resolved;
	resolved.items.reserve(projection_items.size());
	std::array<bool, kJpegDctCoefficientCount> active {};
	for (auto item : projection_items) {
		const auto physical = resolve_physical_coefficient_column(rowgroup, item.logical_coefficient_id);
		item.physical_coefficient_column_id = static_cast<uint8_t>(physical);
		active[physical]                    = true;
		resolved.items.push_back(item);
	}
	for (size_t coeff_idx = 0; coeff_idx < active.size(); ++coeff_idx) {
		if (active[coeff_idx]) {
			resolved.active_physical_coefficients.push_back(static_cast<uint8_t>(coeff_idx));
		}
	}
	return resolved;
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

struct JpegDctDeviceProjectionBatchItem {
	uint32_t binding_index             = 0;
	uint32_t row_in_rowgroup           = 0;
	uint64_t output_block_index        = 0;
	uint16_t selected_coefficient_slot = 0;
	uint8_t  logical_coefficient_id    = 0;
	uint8_t  physical_coefficient_id   = 0;
	uint8_t  output_coefficient_id     = 0;
	uint8_t  output_grid_tensor        = 0;
	float    weight                    = 1.0F;
};

struct JpegDctDeviceFixedTransformBatchItem {
	uint32_t binding_base     = 0;
	uint32_t row_in_rowgroup  = 0;
	uint64_t output_block_index = 0;
	uint8_t  component        = 0;
	uint8_t  zigzag_columns   = 0;
	uint16_t x_factor         = 2;
	uint16_t y_factor         = 2;
	uint8_t  x_subblock       = 0;
	uint8_t  y_subblock       = 0;
	uint8_t  x_upsample       = 0;
	uint8_t  y_upsample       = 0;
	uint16_t x_up_factor      = 1;
	uint16_t y_up_factor      = 1;
	uint16_t x_down_factor    = 1;
	uint16_t y_down_factor    = 1;
	uint32_t quant_table_index = 0;
	uint32_t x_weight_matrix_index = 0;
	uint32_t y_weight_matrix_index = 0;
};

// RGB-no-more constructs its factor-2 conversion matrix in float32 with
// torch.mm.  The tiny, non-zero terms that result from that construction are
// part of the preprocessing's numerical contract: regenerating the
// mathematically equivalent matrix with double-precision std::cos changes
// values near round-to-even boundaries.  Only the first eight rows are needed
// when a 16x16 composed block is cropped back to one 8x8 DCT block.  Hex float
// literals preserve the reference float32 values bit-for-bit.
__device__ __constant__ float kRgbNoMoreDown2Conversion[8U * 16U] = {
    0x1.6a09e40000000p-1F, -0x1.8275a00000000p-26F, 0x1.44df280000000p-25F, -0x1.5926600000000p-30F, 0x1.257d860000000p-26F, -0x1.ee38c40000000p-24F, 0x1.0f04360000000p-26F, -0x1.a7073e0000000p-26F, 0x1.6a09e40000000p-1F, -0x1.8275a00000000p-26F, 0x1.44df280000000p-25F, -0x1.5926600000000p-30F, 0x1.257d860000000p-26F, -0x1.ee38c40000000p-24F, 0x1.0f04360000000p-26F, -0x1.a7073e0000000p-26F,
    0x1.4679360000000p-1F, 0x1.31ce1c0000000p-2F, -0x1.df2c160000000p-5F, 0x1.8aa7d80000000p-6F, -0x1.9957a00000000p-7F, 0x1.ceb2c40000000p-8F, -0x1.0175300000000p-8F, 0x1.d142040000000p-10F, -0x1.46793a0000000p-1F, 0x1.31ce200000000p-2F, 0x1.df2bf00000000p-5F, 0x1.8aa7e00000000p-6F, 0x1.9958020000000p-7F, 0x1.ceb5c80000000p-8F, 0x1.0173980000000p-8F, 0x1.d141dc0000000p-10F,
    -0x1.374a980000000p-25F, 0x1.6a09e60000000p-1F, -0x1.a67cdc0000000p-25F, 0x1.22314e0000000p-24F, 0x1.6ff0240000000p-25F, 0x1.52cd560000000p-24F, -0x1.01c3ec0000000p-24F, 0x1.b2c7680000000p-24F, 0x1.5ba54c0000000p-24F, -0x1.6a09e60000000p-1F, 0x1.133e6e0000000p-24F, -0x1.22314e0000000p-24F, -0x1.17f8120000000p-24F, -0x1.1966ac0000000p-23F, 0x1.e1c3ec0000000p-24F, -0x1.658ed00000000p-25F,
    -0x1.b8f24a0000000p-3F, 0x1.16da3a0000000p-1F, 0x1.865e1e0000000p-2F, -0x1.856b020000000p-4F, 0x1.6580580000000p-5F, -0x1.80b5d00000000p-6F, 0x1.a23c320000000p-7F, -0x1.75c5ce0000000p-8F, 0x1.b8f2440000000p-3F, 0x1.16da3c0000000p-1F, -0x1.865e1c0000000p-2F, -0x1.856aec0000000p-4F, -0x1.6580480000000p-5F, -0x1.80b5c00000000p-6F, -0x1.a23c0a0000000p-7F, -0x1.75c3000000000p-8F,
    0x1.f715080000000p-25F, -0x1.12bf920000000p-24F, 0x1.6a09e80000000p-1F, -0x1.be4b100000000p-28F, -0x1.b54c3e0000000p-25F, -0x1.c3e49a0000000p-25F, 0x1.75639e0000000p-24F, -0x1.a4625a0000000p-23F, -0x1.91d5f00000000p-26F, -0x1.095fca0000000p-23F, 0x1.6a09e60000000p-1F, 0x1.48369e0000000p-25F, 0x1.cab3c20000000p-25F, -0x1.c1f24c0000000p-24F, 0x1.c5639e0000000p-24F, -0x1.32312c0000000p-22F,
    0x1.0f88900000000p-3F, -0x1.c677a60000000p-3F, 0x1.041f9a0000000p-1F, 0x1.9a63a80000000p-2F, -0x1.b26d9c0000000p-4F, 0x1.9438ea0000000p-5F, -0x1.9dc9fa0000000p-6F, 0x1.6814be0000000p-7F, -0x1.0f88880000000p-3F, -0x1.c677940000000p-3F, -0x1.041f9e0000000p-1F, 0x1.9a63aa0000000p-2F, 0x1.b26d8c0000000p-4F, 0x1.94392e0000000p-5F, 0x1.9dca020000000p-6F, 0x1.6815fc0000000p-7F,
    -0x1.2fa9760000000p-27F, 0x1.0330ea0000000p-24F, 0x1.c75c800000000p-33F, 0x1.6a09e60000000p-1F, -0x1.d72bea0000000p-25F, 0x1.46d4d00000000p-24F, -0x1.6c12680000000p-25F, 0x1.d4b4a00000000p-23F, 0x1.02fa980000000p-23F, -0x1.9874e00000000p-31F, 0x1.9f8e280000000p-23F, -0x1.6a09e60000000p-1F, -0x1.146a0c0000000p-24F, 0x1.192b300000000p-24F, -0x1.cfb6640000000p-27F, -0x1.ecb4a00000000p-23F,
    -0x1.9388f00000000p-4F, 0x1.3517200000000p-3F, -0x1.9e6c920000000p-3F, 0x1.fcfe940000000p-2F, 0x1.a03ac20000000p-2F, -0x1.b9b2840000000p-4F, 0x1.85af280000000p-5F, -0x1.4059020000000p-6F, 0x1.9388dc0000000p-4F, 0x1.3517400000000p-3F, 0x1.9e6c740000000p-3F, 0x1.fcfeb00000000p-2F, -0x1.a03aae0000000p-2F, -0x1.b9b28c0000000p-4F, -0x1.85aede0000000p-5F, -0x1.4057cc0000000p-6F,
};

struct JpegDctDeviceCachedFixedTransformBatchItem {
	const int16_t*                        dense = nullptr;
	JpegDctDeviceFixedTransformBatchItem transform {};
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

struct JpegDctDeviceDecodedGatherBatchItem {
	uint32_t source_index       = 0;
	uint32_t row_in_rowgroup    = 0;
	uint64_t output_block_index = 0;
};

struct DecodedRowgroupWork {
	uint32_t                                                 shard_id       = 0;
	uint32_t                                                 rowgroup_index = 0;
	galp::execution::Rowgroup                                rowgroup {};
	std::vector<uint32_t>                                    owned_selected_vectors;
	const std::vector<uint32_t>*                             selected_vectors = nullptr;
	std::vector<JpegDctDeviceGatherItem>                     owned_gather_items;
	const std::vector<JpegDctDeviceGatherItem>*              gather_items = nullptr;
	std::vector<JpegDctDeviceProjectionItem>                 owned_projection_items;
	const std::vector<JpegDctDeviceProjectionItem>*          projection_items = nullptr;
	std::vector<JpegDctDeviceFixedTransformItem>             owned_fixed_transform_items;
	const std::vector<JpegDctDeviceFixedTransformItem>*      fixed_transform_items = nullptr;
	const std::vector<JpegDctDevicePlanlessImageDescriptor>* planless_images       = nullptr;
	std::vector<uint8_t>                                     active_physical_coefficients;
	size_t                                                   expr_index_base         = 0;
	uint32_t                                                 decode_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	bool                                                     decodes_full_rowgroup   = false;
	bool                                                     owns_rowgroup           = false;
	JpegDctDeviceDecodedRowgroupCacheKey                     cache_key {};
	std::unique_ptr<JpegDctDeviceDecodedRowgroupCacheEntry>  cache_entry;

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
	    , owned_projection_items(std::move(other.owned_projection_items))
	    , projection_items(other.projection_items == &other.owned_projection_items ? &owned_projection_items
	                                                                               : other.projection_items)
	    , owned_fixed_transform_items(std::move(other.owned_fixed_transform_items))
	    , fixed_transform_items(other.fixed_transform_items == &other.owned_fixed_transform_items
	                                ? &owned_fixed_transform_items
	                                : other.fixed_transform_items)
	    , planless_images(other.planless_images)
	    , active_physical_coefficients(std::move(other.active_physical_coefficients))
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
	galp::memory::CudaStream                                               cache_hit_stream;
	galp::memory::CudaEvent                                                cached_gather_start;
	galp::memory::CudaEvent                                                cached_gather_done;
	galp::memory::CudaEvent                                                decoded_batch_gather_done;
	galp::memory::CudaStream                                               fixed_grid_round_stream;
	galp::memory::CudaStream                                               transform_stream;
	galp::memory::CudaEvent                                                decode_to_transform_event;
	galp::runtime::ExecutionWorkset                                        decode_workset;
	JpegDctDeviceScratchBuffer<DeviceCoeffBinding>                         column_bindings;
	JpegDctDeviceScratchBuffer<JpegDctDeviceProjectionBatchItem>           batch_projection_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceFixedTransformBatchItem>       batch_fixed_transform_items;
	JpegDctDeviceScratchBuffer<JpegDctDevicePlanlessImageDescriptor>       planless_image_descriptors;
	JpegDctDeviceScratchBuffer<uint32_t>                                   fixed_transform_group_offsets;
	JpegDctDeviceScratchBuffer<JpegDctDeviceMaterializeBatchItem>          batch_materialize_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceCachedGatherBatchItem>         cached_gather_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceCachedFixedTransformBatchItem> cached_fixed_transform_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceDecodedGatherBatchItem>        decoded_gather_items;
	std::vector<DeviceCoeffBinding>                                        host_column_bindings;
	std::vector<BoundCoeffColumns>                                         host_bound_sources;
	std::vector<JpegDctDeviceProjectionBatchItem>                          host_projection_items;
	std::vector<JpegDctDeviceFixedTransformBatchItem>                      host_fixed_transform_items;
	std::vector<JpegDctDeviceFixedTransformBatchItem>                      host_ordered_fixed_transform_items;
	std::vector<JpegDctDevicePlanlessImageDescriptor>                      host_planless_image_descriptors;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_item_order;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_group_offsets;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_item_groups;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_group_cursors;
	std::unordered_map<size_t, uint32_t>                                   host_workset_fixed_transform_group_lookup;
	std::vector<JpegDctDeviceMaterializeBatchItem>                         host_materialize_items;
	std::vector<JpegDctDeviceDecodedGatherBatchItem>                       host_decoded_gather_items;
	std::vector<JpegDctDeviceCachedGatherBatchItem>                        host_cached_gather_items;
	std::vector<JpegDctDeviceCachedFixedTransformBatchItem>                host_cached_fixed_transform_items;
	std::deque<std::vector<JpegDctDeviceCachedGatherBatchItem>>            host_cached_gather_uploads;
	std::deque<std::vector<JpegDctDeviceCachedFixedTransformBatchItem>>    host_cached_fixed_transform_uploads;
	std::vector<DecodedRowgroupWork>                                       host_pending_works;
	std::shared_ptr<galp::runtime::PinnedRowgroupBufferPool>               rowgroup_prefetch_pinned_pool;
	size_t                                                                 rowgroup_prefetch_pinned_pool_slots = 0;
	bool                                                                   cached_gather_in_flight             = false;
	bool                                                                   cached_fixed_transform_in_flight    = false;
	int                                                                    direct_dct_stream_priority          = 0;
	int                                                                    cuda_least_stream_priority          = 0;
	int                                                                    cuda_greatest_stream_priority       = 0;
	bool                                                                   direct_dct_low_priority_streams     = false;
	size_t                                                                 transform_blocks_per_launch         = 0;
	size_t                                                                 transform_ctas_per_launch           = 0;
	bool                                                                   planless_transform_resources_ready  = false;
	size_t                                                                 planless_transform_registers_per_thread = 0;
	size_t                                                                 planless_transform_static_shared_bytes_per_cta = 0;
	size_t                                                                 planless_transform_local_bytes_per_thread = 0;
	size_t                                                                 planless_transform_max_active_ctas_per_sm = 0;
	size_t                                                                 cuda_max_threads_per_sm             = 0;
	size_t                                                                 cuda_warp_size                      = 0;
	std::list<std::string>                                                 fls_reader_lru;
	struct CachedFlsReaderEntry {
		std::shared_ptr<galp::format::FlsReader> reader;
		std::list<std::string>::iterator         lru_position;
	};
	std::unordered_map<std::string, CachedFlsReaderEntry> fls_readers;

	std::shared_ptr<galp::format::FlsReader> fls_reader(const std::filesystem::path& path) {
		const auto key   = path.lexically_normal().string();
		const auto found = fls_readers.find(key);
		if (found != fls_readers.end()) {
			fls_reader_lru.splice(fls_reader_lru.begin(), fls_reader_lru, found->second.lru_position);
			return found->second.reader;
		}

		constexpr size_t kFlsReaderCacheCapacity = 64;
		if (fls_readers.size() >= kFlsReaderCacheCapacity) {
			fls_readers.erase(fls_reader_lru.back());
			fls_reader_lru.pop_back();
		}
		auto reader = std::make_shared<galp::format::FlsReader>(path, /*load_column_names=*/false);
		fls_reader_lru.push_front(key);
		fls_readers.emplace(key, CachedFlsReaderEntry {reader, fls_reader_lru.begin()});
		return reader;
	}

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
			batch_projection_items.release();
			batch_fixed_transform_items.release();
			planless_image_descriptors.release();
			fixed_transform_group_offsets.release();
			batch_materialize_items.release();
			cached_gather_items.release();
			cached_fixed_transform_items.release();
			decoded_gather_items.release();
			galp::runtime::release_workset(
			    decode_workset, /*preserve_resources=*/false, /*h2d_already_complete=*/false);
		} catch (const std::exception& e) { std::fprintf(stderr, "JpegDctDeviceScratch cleanup: %s\n", e.what()); }
	}

	cudaStream_t stream_for_cache_hit() {
		if (!cache_hit_stream) {
			cache_hit_stream.create_with_priority(cudaStreamNonBlocking, direct_dct_stream_priority);
		}
		return cache_hit_stream.get();
	}

	cudaStream_t stream_for_fixed_grid_rounding() {
		if (!fixed_grid_round_stream) {
			fixed_grid_round_stream.create_with_priority(cudaStreamNonBlocking, direct_dct_stream_priority);
		}
		return fixed_grid_round_stream.get();
	}

	cudaStream_t stream_for_transform() {
		if (!transform_stream) {
			transform_stream.create_with_priority(cudaStreamNonBlocking, direct_dct_stream_priority);
		}
		return transform_stream.get();
	}

	void configure_scheduling(const bool   use_low_priority,
	                          const size_t blocks_per_launch,
	                          const size_t ctas_per_launch) {
		CUDA_SAFE_CALL(cudaDeviceGetStreamPriorityRange(
		    &cuda_least_stream_priority, &cuda_greatest_stream_priority));
		direct_dct_low_priority_streams       = use_low_priority;
		direct_dct_stream_priority            = use_low_priority ? cuda_least_stream_priority : 0;
		transform_blocks_per_launch           = blocks_per_launch;
		transform_ctas_per_launch             = ctas_per_launch;
		decode_workset.transfer.stream_priority = direct_dct_stream_priority;
	}

	int actual_stream_priority(const cudaStream_t stream) const {
		if (stream == nullptr) {
			return std::numeric_limits<int>::max();
		}
		int actual_priority = 0;
		CUDA_SAFE_CALL(cudaStreamGetPriority(stream, &actual_priority));
		return actual_priority;
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

constexpr unsigned kPlanlessTransformThreadsPerCta        = 64U;
constexpr unsigned kLimitedPlanlessTransformCtasPerLaunch = 64U;

struct FixedTransformPlanView {
	const std::vector<uint32_t>* item_order    = nullptr;
	const std::vector<uint32_t>* group_offsets = nullptr;
};

size_t fixed_transform_item_count(const std::vector<DecodedRowgroupWork>& works) {
	size_t count = 0;
	for (const auto& work : works) {
		if (work.fixed_transform_items != nullptr) {
			count += work.fixed_transform_items->size();
		}
	}
	return count;
}

FixedTransformPlanView fixed_transform_plan_for_workset(
    const std::vector<uint32_t>* fixed_transform_item_order,
    const std::vector<uint32_t>* fixed_transform_group_offsets,
    const size_t                 source_item_offset,
    const size_t                 source_item_count,
    JpegDctDeviceScratch&        scratch) {
	if (fixed_transform_item_order == nullptr && fixed_transform_group_offsets == nullptr) {
		return {};
	}
	if (fixed_transform_item_order == nullptr || fixed_transform_group_offsets == nullptr) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan is incomplete");
	}
	const auto& plan_order   = *fixed_transform_item_order;
	const auto& plan_offsets = *fixed_transform_group_offsets;
	if (plan_order.empty()) {
		if (!plan_offsets.empty() || source_item_offset != 0U || source_item_count != 0U) {
			throw std::runtime_error("JPEG DCT deterministic fixed-transform plan is inconsistent");
		}
		return {};
	}
	if (plan_offsets.size() < 2U || plan_offsets.front() != 0U || plan_offsets.back() != plan_order.size() ||
	    !std::is_sorted(plan_offsets.begin(), plan_offsets.end())) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform group offsets are invalid");
	}
	if (source_item_offset > plan_order.size() || source_item_count > plan_order.size() - source_item_offset) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform workset exceeds the plan");
	}
	if (source_item_count == 0U) {
		return {};
	}
	if (source_item_offset == 0U && source_item_count == plan_order.size()) {
		return {fixed_transform_item_order, fixed_transform_group_offsets};
	}
	if (source_item_count > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("JPEG DCT fixed-transform workset exceeds deterministic index range");
	}

	// The prepared permutation maps plan-wide source order to plan-wide groups.
	// A workset is a contiguous source slice, but its global grouped indices need
	// not be contiguous when request order differs from shard/rowgroup order.
	// Repack represented groups in first-source order while retaining source order
	// within each group (the global stable sort uses that same within-group order).
	auto& workset_order         = scratch.host_workset_fixed_transform_item_order;
	auto& workset_offsets       = scratch.host_workset_fixed_transform_group_offsets;
	auto& workset_item_groups   = scratch.host_workset_fixed_transform_item_groups;
	auto& workset_group_cursors = scratch.host_workset_fixed_transform_group_cursors;
	auto& workset_group_lookup  = scratch.host_workset_fixed_transform_group_lookup;
	workset_order.resize(source_item_count);
	workset_item_groups.resize(source_item_count);
	workset_offsets.clear();
	workset_offsets.push_back(0U);
	workset_group_lookup.clear();
	workset_group_lookup.reserve(std::min(source_item_count, plan_offsets.size() - 1U));
	for (size_t source_index = 0; source_index < source_item_count; ++source_index) {
		const auto global_ordered_index = plan_order[source_item_offset + source_index];
		if (global_ordered_index >= plan_order.size()) {
			throw std::runtime_error("JPEG DCT deterministic fixed-transform permutation is invalid");
		}
		const auto upper = std::upper_bound(plan_offsets.begin(), plan_offsets.end(), global_ordered_index);
		if (upper == plan_offsets.begin() || upper == plan_offsets.end()) {
			throw std::runtime_error("JPEG DCT deterministic fixed-transform group lookup failed");
		}
		const auto global_group = static_cast<size_t>(std::distance(plan_offsets.begin(), upper) - 1);
		auto [group_it, inserted] =
		    workset_group_lookup.emplace(global_group, static_cast<uint32_t>(workset_group_lookup.size()));
		if (inserted) {
			workset_offsets.push_back(0U);
		}
		const auto local_group            = group_it->second;
		workset_item_groups[source_index] = local_group;
		++workset_offsets[static_cast<size_t>(local_group) + 1U];
	}
	for (size_t group = 1; group < workset_offsets.size(); ++group) {
		workset_offsets[group] += workset_offsets[group - 1U];
	}
	workset_group_cursors.assign(workset_offsets.begin(), workset_offsets.end() - 1);
	for (size_t source_index = 0; source_index < source_item_count; ++source_index) {
		workset_order[source_index] = workset_group_cursors[workset_item_groups[source_index]]++;
	}
	return {&workset_order, &workset_offsets};
}

__device__ __forceinline__ uint8_t natural_to_physical_coeff_device(const uint8_t natural,
                                                                    const uint8_t zigzag_columns) {
	if (zigzag_columns == 0) {
		return natural;
	}
	constexpr uint8_t natural_to_zigzag[64] {
	    0,  1,  5,  6,  14, 15, 27, 28,
	    2,  4,  7,  13, 16, 26, 29, 42,
	    3,  8,  12, 17, 25, 30, 41, 43,
	    9,  11, 18, 24, 31, 40, 44, 53,
	    10, 19, 23, 32, 39, 45, 52, 54,
	    20, 22, 33, 38, 46, 51, 55, 60,
	    21, 34, 37, 47, 50, 56, 59, 61,
	    35, 36, 48, 49, 57, 58, 62, 63,
	};
	return natural_to_zigzag[natural];
}

__global__ void project_dct_coefficients_batch_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                                      const JpegDctDeviceProjectionBatchItem* __restrict items,
                                                      const size_t item_count,
                                                      const size_t coefficients_per_block,
                                                      int16_t* __restrict out) {
	const size_t item_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (item_idx >= item_count) {
		return;
	}
	const auto item    = items[item_idx];
	const auto binding = column_bindings[item.binding_index];
	int16_t    value   = 0;
	if (binding.source == DeviceCoeffSource::kI16) {
		value = binding.column_i16[item.row_in_rowgroup];
	} else if (binding.source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(binding.column_i8[item.row_in_rowgroup]);
	}
	out[selected_dct_output_offset(
	    item.output_block_index, item.selected_coefficient_slot, coefficients_per_block)] = value;
}

__global__ void project_dct_ycbcr_grid_batch_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                                    const JpegDctDeviceProjectionBatchItem* __restrict items,
                                                    const size_t item_count,
                                                    int16_t* __restrict y_out,
                                                    int16_t* __restrict cbcr_out,
                                                    float* __restrict y_accum,
                                                    float* __restrict cbcr_accum) {
	const size_t item_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (item_idx >= item_count) {
		return;
	}
	const auto item    = items[item_idx];
	const auto binding = column_bindings[item.binding_index];
	int16_t    value   = 0;
	if (binding.source == DeviceCoeffSource::kI16) {
		value = binding.column_i16[item.row_in_rowgroup];
	} else if (binding.source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(binding.column_i8[item.row_in_rowgroup]);
	}
	const auto coeff_idx = static_cast<size_t>(item.output_coefficient_id);
	if (item.output_grid_tensor == kJpegDctYcbcrDctGridTensorY && y_out != nullptr) {
		const auto offset = item.output_block_index * kJpegDctCoefficientCount + coeff_idx;
		if (y_accum != nullptr) {
			atomicAdd(y_accum + offset, static_cast<float>(value) * item.weight);
		} else {
			y_out[offset] = value;
		}
	} else if (item.output_grid_tensor == kJpegDctYcbcrDctGridTensorCbCr && cbcr_out != nullptr) {
		const auto offset = item.output_block_index * kJpegDctCoefficientCount + coeff_idx;
		if (cbcr_accum != nullptr) {
			atomicAdd(cbcr_accum + offset, static_cast<float>(value) * item.weight);
		} else {
			cbcr_out[offset] = value;
		}
	}
}

__device__ __forceinline__ float
finalize_dct_grid_float(const float value, const float output_add, const float output_scale) {
	float rounded = nearbyintf(value);
	rounded       = fminf(32767.0F, fmaxf(-32768.0F, rounded));
	rounded       = __fadd_rn(rounded, output_add);
	return __fmul_rn(rounded, output_scale);
}

__device__ __forceinline__ void store_planless_dct_grid_value(const JpegDctDevicePlanlessImageDescriptor& image,
                                                              const uint32_t                              component,
                                                              const uint32_t                              output_x,
                                                              const uint32_t                              output_y,
                                                              const uint32_t y_output_width,
                                                              const uint32_t y_output_height,
                                                              const uint32_t cbcr_output_width,
                                                              const uint32_t cbcr_output_height,
                                                              const uint32_t lane,
                                                              const float    value,
                                                              float* __restrict y_accum,
                                                              float* __restrict cbcr_accum) {
	if (lane >= 64U) {
		return;
	}
	if (component == 0U && y_accum != nullptr) {
		const auto output_block_index =
		    (static_cast<uint64_t>(image.request_index) * y_output_height + output_y) * y_output_width + output_x;
		y_accum[output_block_index * 64U + lane] = value;
	} else if (component != 0U && cbcr_accum != nullptr) {
		const auto output_block_index =
		    ((static_cast<uint64_t>(image.request_index) * 2U + component - 1U) * cbcr_output_height + output_y) *
		        cbcr_output_width +
		    output_x;
		cbcr_accum[output_block_index * 64U + lane] = value;
	}
}

__global__ void round_dct_grid_accum_pair_kernel(const float* __restrict y_in,
                                                 const size_t y_count,
                                                 const float* __restrict cbcr_in,
                                                 const size_t cbcr_count,
                                                 int16_t* __restrict y_out,
                                                 int16_t* __restrict cbcr_out) {
	const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= y_count + cbcr_count) {
		return;
	}
	const bool   is_y      = idx < y_count;
	const size_t local_idx = is_y ? idx : idx - y_count;
	const float* input     = is_y ? y_in : cbcr_in;
	int16_t*     output    = is_y ? y_out : cbcr_out;
	float        value     = nearbyintf(input[local_idx]);
	value                  = fminf(32767.0F, fmaxf(-32768.0F, value));
	output[local_idx]      = static_cast<int16_t>(value);
}

__global__ void round_affine_dct_grid_accum_pair_kernel(float* __restrict y_in_out,
                                                        const size_t y_count,
                                                        float* __restrict cbcr_in_out,
                                                        const size_t cbcr_count,
                                                        const float  output_add,
                                                        const float  output_scale) {
	const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= y_count + cbcr_count) {
		return;
	}
	const bool   is_y      = idx < y_count;
	const size_t local_idx = is_y ? idx : idx - y_count;
	float*       in_out    = is_y ? y_in_out : cbcr_in_out;
	// Match two separate FP32 eager elementwise operations. Explicit rounding
	// prevents contraction into an FMA, which would change boundary values.
	in_out[local_idx] = finalize_dct_grid_float(in_out[local_idx], output_add, output_scale);
}

__device__ uint64_t planless_rectangle_intersection_count(const uint64_t width,
                                                          const uint64_t height,
                                                          const uint64_t origin_x,
                                                          const uint64_t origin_y,
                                                          const uint64_t size) {
	if (origin_x >= width || origin_y >= height) {
		return 0U;
	}
	return min(size, width - origin_x) * min(size, height - origin_y);
}

__device__ uint64_t planless_morton_rank_in_rectangle(const uint32_t width,
                                                      const uint32_t height,
                                                      const uint32_t x,
                                                      const uint32_t y) {
	uint64_t size = 1U;
	while (size < max(static_cast<uint64_t>(width), static_cast<uint64_t>(height))) {
		size <<= 1U;
	}
	uint64_t rank     = 0U;
	uint64_t origin_x = 0U;
	uint64_t origin_y = 0U;
	while (size > 1U) {
		const auto half            = size >> 1U;
		const auto qx              = static_cast<uint32_t>(x >= origin_x + half);
		const auto qy              = static_cast<uint32_t>(y >= origin_y + half);
		const auto target_quadrant = qx | (qy << 1U);
		for (uint32_t quadrant = 0; quadrant < target_quadrant; ++quadrant) {
			const auto child_x = origin_x + ((quadrant & 1U) != 0U ? half : 0U);
			const auto child_y = origin_y + ((quadrant & 2U) != 0U ? half : 0U);
			rank += planless_rectangle_intersection_count(width, height, child_x, child_y, half);
		}
		origin_x += qx != 0U ? half : 0U;
		origin_y += qy != 0U ? half : 0U;
		size = half;
	}
	return rank;
}

__device__ uint64_t planless_block_order_rank(
    const uint32_t width, const uint32_t height, const uint32_t x, const uint32_t y, const uint8_t spatial_order) {
	constexpr uint32_t tile_blocks = 32U;
	if (spatial_order == static_cast<uint8_t>(JpegDctSpatialOrder::kRaster)) {
		return static_cast<uint64_t>(y) * width + x;
	}
	if (spatial_order == static_cast<uint8_t>(JpegDctSpatialOrder::kZOrder)) {
		return planless_morton_rank_in_rectangle(width, height, x, y);
	}
	const auto tile_x          = (x / tile_blocks) * tile_blocks;
	const auto tile_y          = (y / tile_blocks) * tile_blocks;
	const auto tile_width      = min(tile_blocks, width - tile_x);
	const auto tile_height     = min(tile_blocks, height - tile_y);
	const auto before_tile_row = static_cast<uint64_t>(tile_y) * width;
	const auto before_tile     = static_cast<uint64_t>(tile_height) * tile_x;
	const auto local_x         = x - tile_x;
	const auto local_y         = y - tile_y;
	const auto within_tile     = spatial_order == static_cast<uint8_t>(JpegDctSpatialOrder::kTiledRaster32)
	                                 ? static_cast<uint64_t>(local_y) * tile_width + local_x
	                                 : planless_morton_rank_in_rectangle(tile_width, tile_height, local_x, local_y);
	return before_tile_row + before_tile + within_tile;
}

__device__ float planless_axis_phase_weight(const float* __restrict phase_matrices,
                                            const uint32_t matrix_base,
                                            const uint16_t up_factor,
                                            const uint16_t down_factor,
                                            const uint32_t source_block,
                                            const uint32_t output_block,
                                            const uint32_t out_coeff,
                                            const uint32_t in_coeff) {
	if (matrix_base != std::numeric_limits<uint32_t>::max()) {
		const auto relative_phase =
		    static_cast<int64_t>(source_block) * up_factor - static_cast<int64_t>(output_block) * down_factor;
		const auto phase_index = static_cast<int64_t>(matrix_base) + relative_phase + up_factor - 1;
		return phase_matrices[static_cast<size_t>(phase_index) * 64U + out_coeff * 8U + in_coeff];
	}
	if (up_factor == 1U && down_factor == 1U) {
		return out_coeff == in_coeff ? 1.0F : 0.0F;
	}
	if (up_factor == 1U && down_factor == 2U) {
		const auto subblock = source_block - output_block * 2U;
		return kRgbNoMoreDown2Conversion[out_coeff * 16U + subblock * 8U + in_coeff] / 0x1.6a09e60000000p+0F;
	}
	return 0.0F;
}

__global__ void transformed_dct_grid_planless_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                                     const JpegDctDevicePlanlessImageDescriptor* __restrict images,
                                                     const size_t   image_count,
                                                     const uint64_t output_block_offset,
                                                     const uint64_t output_block_count,
                                                     const uint16_t* __restrict quant_tables,
                                                     const float* __restrict phase_matrices,
                                                     const uint32_t y_output_width,
                                                     const uint32_t y_output_height,
                                                     const uint32_t cbcr_output_width,
                                                     const uint32_t cbcr_output_height,
                                                     const int32_t  clamp_min,
                                                     const int32_t  clamp_max,
                                                     float* __restrict y_accum,
                                                     float* __restrict cbcr_accum) {
	const auto       lane                = static_cast<uint32_t>(threadIdx.x);
	const uint64_t   y_blocks            = static_cast<uint64_t>(y_output_width) * y_output_height;
	const uint64_t   cbcr_channel_blocks = static_cast<uint64_t>(cbcr_output_width) * cbcr_output_height;
	const uint64_t   blocks_per_image    = y_blocks + 2U * cbcr_channel_blocks;
	__shared__ float composed[16U * 16U];
	__shared__ float vertical[8U * 16U];
	__shared__ float source[64U];
	__shared__ float horizontal[64U];
	for (uint64_t launch_block = blockIdx.x; launch_block < output_block_count; launch_block += gridDim.x) {
		const uint64_t linear_block = output_block_offset + launch_block;
		if (blocks_per_image == 0U || linear_block >= image_count * blocks_per_image || quant_tables == nullptr) {
			continue;
		}
		const auto image_index = static_cast<size_t>(linear_block / blocks_per_image);
		const auto local_block = linear_block % blocks_per_image;
		uint32_t   component   = 0U;
		uint32_t   output_x    = 0U;
		uint32_t   output_y    = 0U;
		if (local_block < y_blocks) {
			output_y = static_cast<uint32_t>(local_block / y_output_width);
			output_x = static_cast<uint32_t>(local_block % y_output_width);
		} else {
			const auto chroma_local  = local_block - y_blocks;
			component                = 1U + static_cast<uint32_t>(chroma_local / cbcr_channel_blocks);
			const auto channel_local = chroma_local % cbcr_channel_blocks;
			output_y                 = static_cast<uint32_t>(channel_local / cbcr_output_width);
			output_x                 = static_cast<uint32_t>(channel_local % cbcr_output_width);
		}
		const auto image      = images[image_index];
		const auto descriptor = image.components[component];
		if (descriptor.present == 0U || descriptor.x_up_factor == 0U || descriptor.y_up_factor == 0U ||
		    descriptor.x_down_factor == 0U || descriptor.y_down_factor == 0U) {
			continue;
		}
		const auto x_down                   = static_cast<uint32_t>(descriptor.x_down_factor);
		const auto y_down                   = static_cast<uint32_t>(descriptor.y_down_factor);
		const bool use_reference_down2_axes = descriptor.x_up_factor == 1U && descriptor.y_up_factor == 1U &&
		                                      x_down >= 1U && x_down <= 2U && y_down >= 1U && y_down <= 2U;
		if (use_reference_down2_axes) {
			const auto source_width  = x_down * 8U;
			const auto source_height = y_down * 8U;
			const auto source_count  = source_width * source_height;
			// Keep the block at two warps: the transform produces 64 coefficients, and
			// the largest canonical down2 source contains only four coefficients per
			// lane.  A 256-thread block left six warps idle after the source load and
			// needlessly limited residency across the tens of thousands of output
			// blocks in an ImageNet batch.
			for (uint32_t source_linear = lane; source_linear < source_count; source_linear += blockDim.x) {
				const auto source_block_slot = source_linear / 64U;
				const auto coeff             = static_cast<uint8_t>(source_linear % 64U);
				const auto subblock_x        = source_block_slot % x_down;
				const auto subblock_y        = source_block_slot / x_down;
				const auto source_x = static_cast<uint32_t>(descriptor.crop_x) + output_x * x_down + subblock_x;
				const auto source_y = static_cast<uint32_t>(descriptor.crop_y) + output_y * y_down + subblock_y;
				const auto rank     = planless_block_order_rank(
                    descriptor.width_in_blocks, descriptor.height_in_blocks, source_x, source_y, image.spatial_order);
				const auto row =
				    static_cast<uint64_t>(image.row_start_in_rowgroup) + descriptor.component_row_offset + rank;
				const auto physical = natural_to_physical_coeff_device(coeff, image.zigzag_columns != 0U);
				const auto binding  = column_bindings[image.binding_base + physical];
				int16_t    value    = 0;
				if (binding.source == DeviceCoeffSource::kI16) {
					value = binding.column_i16[row];
				} else if (binding.source == DeviceCoeffSource::kI8) {
					value = static_cast<int16_t>(binding.column_i8[row]);
				}
				const auto quant =
				    static_cast<int32_t>(quant_tables[static_cast<size_t>(descriptor.quant_table_index) * 64U + coeff]);
				const auto composed_y = subblock_y * 8U + coeff / 8U;
				const auto composed_x = subblock_x * 8U + coeff % 8U;
				composed[composed_y * source_width + composed_x] =
				    static_cast<float>(min(clamp_max, max(clamp_min, static_cast<int32_t>(value) * quant)));
			}
			__syncthreads();
			for (uint32_t vertical_linear = lane; vertical_linear < 8U * source_width; vertical_linear += blockDim.x) {
				const auto out_y    = vertical_linear / source_width;
				const auto source_x = vertical_linear % source_width;
				float      sum      = 0.0F;
				if (y_down == 1U) {
					sum = composed[out_y * source_width + source_x];
				} else {
#pragma unroll
					for (uint32_t source_y = 0; source_y < 16U; ++source_y) {
						sum = fmaf(kRgbNoMoreDown2Conversion[out_y * 16U + source_y],
						           composed[source_y * source_width + source_x],
						           sum);
					}
				}
				vertical[out_y * source_width + source_x] = sum;
			}
			__syncthreads();
			if (lane < 64U) {
				const auto out_y = lane / 8U;
				const auto out_x = lane % 8U;
				float      sum   = 0.0F;
				if (x_down == 1U) {
					sum = vertical[out_y * source_width + out_x];
				} else {
#pragma unroll
					for (uint32_t source_x = 0; source_x < 16U; ++source_x) {
						sum = fmaf(vertical[out_y * source_width + source_x],
						           kRgbNoMoreDown2Conversion[out_x * 16U + source_x],
						           sum);
					}
				}
				const auto factor_product = x_down * y_down;
				const auto value          = factor_product == 4U   ? sum * 0.5F
				                            : factor_product == 2U ? sum / 0x1.6a09e60000000p+0F
				                                                   : sum;
				store_planless_dct_grid_value(image,
				                              component,
				                              output_x,
				                              output_y,
				                              y_output_width,
				                              y_output_height,
				                              cbcr_output_width,
				                              cbcr_output_height,
				                              lane,
				                              value,
				                              y_accum,
				                              cbcr_accum);
			}
			__syncthreads();
			continue;
		}

		const bool needs_x_program = descriptor.x_phase_matrix_base != std::numeric_limits<uint32_t>::max();
		const bool needs_y_program = descriptor.y_phase_matrix_base != std::numeric_limits<uint32_t>::max();
		if ((needs_x_program || needs_y_program) && phase_matrices == nullptr) {
			continue;
		}
		const auto source_x_begin = static_cast<uint32_t>((static_cast<uint64_t>(output_x) * descriptor.x_down_factor) /
		                                                  descriptor.x_up_factor);
		const auto source_x_end   = static_cast<uint32_t>(
            ((static_cast<uint64_t>(output_x + 1U) * descriptor.x_down_factor) - 1U) / descriptor.x_up_factor);
		const auto source_y_begin = static_cast<uint32_t>((static_cast<uint64_t>(output_y) * descriptor.y_down_factor) /
		                                                  descriptor.y_up_factor);
		const auto source_y_end   = static_cast<uint32_t>(
            ((static_cast<uint64_t>(output_y + 1U) * descriptor.y_down_factor) - 1U) / descriptor.y_up_factor);
		float output_sum = 0.0F;
		for (uint32_t source_y_block = source_y_begin; source_y_block <= source_y_end; ++source_y_block) {
			for (uint32_t source_x_block = source_x_begin; source_x_block <= source_x_end; ++source_x_block) {
				if (lane < 64U) {
					const auto coeff    = static_cast<uint8_t>(lane);
					const auto source_x = static_cast<uint32_t>(descriptor.crop_x) + source_x_block;
					const auto source_y = static_cast<uint32_t>(descriptor.crop_y) + source_y_block;
					const auto rank     = planless_block_order_rank(descriptor.width_in_blocks,
                                                                descriptor.height_in_blocks,
                                                                source_x,
                                                                source_y,
                                                                image.spatial_order);
					const auto row =
					    static_cast<uint64_t>(image.row_start_in_rowgroup) + descriptor.component_row_offset + rank;
					const auto physical = natural_to_physical_coeff_device(coeff, image.zigzag_columns != 0U);
					const auto binding  = column_bindings[image.binding_base + physical];
					int16_t    value    = 0;
					if (binding.source == DeviceCoeffSource::kI16) {
						value = binding.column_i16[row];
					} else if (binding.source == DeviceCoeffSource::kI8) {
						value = static_cast<int16_t>(binding.column_i8[row]);
					}
					const auto quant = static_cast<int32_t>(
					    quant_tables[static_cast<size_t>(descriptor.quant_table_index) * 64U + coeff]);
					source[lane] =
					    static_cast<float>(min(clamp_max, max(clamp_min, static_cast<int32_t>(value) * quant)));
				}
				__syncthreads();
				if (lane < 64U) {
					const auto source_y_coeff = lane / 8U;
					const auto out_x_coeff    = lane % 8U;
					float      x_sum          = 0.0F;
					for (uint32_t in_x_coeff = 0U; in_x_coeff < 8U; ++in_x_coeff) {
						const auto wx = planless_axis_phase_weight(phase_matrices,
						                                           descriptor.x_phase_matrix_base,
						                                           descriptor.x_up_factor,
						                                           descriptor.x_down_factor,
						                                           source_x_block,
						                                           output_x,
						                                           out_x_coeff,
						                                           in_x_coeff);
						x_sum += source[source_y_coeff * 8U + in_x_coeff] * wx;
					}
					horizontal[lane] = x_sum;
				}
				__syncthreads();
				if (lane < 64U) {
					const auto out_x_coeff = lane % 8U;
					const auto out_y_coeff = lane / 8U;
					float      weighted    = 0.0F;
					for (uint32_t in_y_coeff = 0U; in_y_coeff < 8U; ++in_y_coeff) {
						const auto wy = planless_axis_phase_weight(phase_matrices,
						                                           descriptor.y_phase_matrix_base,
						                                           descriptor.y_up_factor,
						                                           descriptor.y_down_factor,
						                                           source_y_block,
						                                           output_y,
						                                           out_y_coeff,
						                                           in_y_coeff);
						weighted += horizontal[in_y_coeff * 8U + out_x_coeff] * wy;
					}
					output_sum += weighted;
				}
				__syncthreads();
			}
		}
		if (lane < 64U) {
			store_planless_dct_grid_value(image,
			                              component,
			                              output_x,
			                              output_y,
			                              y_output_width,
			                              y_output_height,
			                              cbcr_output_width,
			                              cbcr_output_height,
			                              lane,
			                              output_sum,
			                              y_accum,
			                              cbcr_accum);
		}
		__syncthreads();
	}
}

void record_planless_transform_resources(JpegDctDeviceScratch& scratch, JpegDctDeviceExecutionStats& stats) {
	if (!scratch.planless_transform_resources_ready) {
		cudaFuncAttributes attributes {};
		CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, transformed_dct_grid_planless_kernel));
		int max_active_ctas_per_sm = 0;
		CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		    &max_active_ctas_per_sm, transformed_dct_grid_planless_kernel, kPlanlessTransformThreadsPerCta, 0));
		int device = 0;
		int max_threads_per_sm = 0;
		int warp_size = 0;
		CUDA_SAFE_CALL(cudaGetDevice(&device));
		CUDA_SAFE_CALL(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
		CUDA_SAFE_CALL(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
		if (attributes.numRegs <= 0 || max_active_ctas_per_sm <= 0 || max_threads_per_sm <= 0 || warp_size <= 0) {
			throw std::runtime_error("invalid CUDA planless transform kernel resource attributes");
		}
		scratch.planless_transform_registers_per_thread = static_cast<size_t>(attributes.numRegs);
		scratch.planless_transform_static_shared_bytes_per_cta = attributes.sharedSizeBytes;
		scratch.planless_transform_local_bytes_per_thread = attributes.localSizeBytes;
		scratch.planless_transform_max_active_ctas_per_sm = static_cast<size_t>(max_active_ctas_per_sm);
		scratch.cuda_max_threads_per_sm = static_cast<size_t>(max_threads_per_sm);
		scratch.cuda_warp_size = static_cast<size_t>(warp_size);
		scratch.planless_transform_resources_ready = true;
	}
	stats.planless_transform_registers_per_thread = scratch.planless_transform_registers_per_thread;
	stats.planless_transform_static_shared_bytes_per_cta =
	    scratch.planless_transform_static_shared_bytes_per_cta;
	stats.planless_transform_local_bytes_per_thread = scratch.planless_transform_local_bytes_per_thread;
	stats.planless_transform_threads_per_cta = kPlanlessTransformThreadsPerCta;
	stats.planless_transform_max_active_ctas_per_sm = scratch.planless_transform_max_active_ctas_per_sm;
	stats.cuda_max_threads_per_sm = scratch.cuda_max_threads_per_sm;
	stats.cuda_warp_size = scratch.cuda_warp_size;
}

__global__ void transformed_dct_grid_sources_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                                    const JpegDctDeviceFixedTransformBatchItem* __restrict items,
                                                    const size_t item_count,
                                                    const uint16_t* __restrict quant_tables,
                                                    const float* __restrict resize_weight_matrices,
                                                    const int32_t clamp_min,
                                                    const int32_t clamp_max,
                                                    float* __restrict y_accum,
                                                    float* __restrict cbcr_accum) {
	const size_t item_idx = static_cast<size_t>(blockIdx.x);
	const auto   lane     = static_cast<uint8_t>(threadIdx.x);
	if (item_idx >= item_count || lane >= 64U || quant_tables == nullptr || resize_weight_matrices == nullptr) {
		return;
	}
	const auto       item = items[item_idx];
	__shared__ float source[64];
	__shared__ float horizontal[64];
	const auto       physical = natural_to_physical_coeff_device(lane, item.zigzag_columns);
	const auto       binding  = column_bindings[item.binding_base + physical];
	int16_t          value    = 0;
	if (binding.source == DeviceCoeffSource::kI16) {
		value = binding.column_i16[item.row_in_rowgroup];
	} else if (binding.source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(binding.column_i8[item.row_in_rowgroup]);
	}
	const auto quant = static_cast<int32_t>(quant_tables[static_cast<size_t>(item.quant_table_index) * 64U + lane]);
	source[lane]     = static_cast<float>(min(clamp_max, max(clamp_min, static_cast<int32_t>(value) * quant)));
	__syncthreads();
	const auto source_y    = static_cast<uint8_t>(lane / 8U);
	const auto out_x_coeff = static_cast<uint8_t>(lane % 8U);
	float      x_sum       = 0.0F;
	for (uint8_t in_x_coeff = 0; in_x_coeff < 8U; ++in_x_coeff) {
		const float wx = resize_weight_matrices[static_cast<size_t>(item.x_weight_matrix_index) * 64U +
		                                        out_x_coeff * 8U + in_x_coeff];
		x_sum += source[source_y * 8U + in_x_coeff] * wx;
	}
	horizontal[lane] = x_sum;
	__syncthreads();
	const auto out_y_coeff = static_cast<uint8_t>(lane / 8U);
	float      weighted    = 0.0F;
	for (uint8_t in_y_coeff = 0; in_y_coeff < 8U; ++in_y_coeff) {
		const float wy = resize_weight_matrices[static_cast<size_t>(item.y_weight_matrix_index) * 64U +
		                                        out_y_coeff * 8U + in_y_coeff];
		weighted += horizontal[in_y_coeff * 8U + out_x_coeff] * wy;
	}
	if (weighted == 0.0F) {
		return;
	}
	if (item.component == 0 && y_accum != nullptr) {
		atomicAdd(y_accum + item.output_block_index * 64U + lane, weighted);
	} else if (item.component != 0 && cbcr_accum != nullptr) {
		atomicAdd(cbcr_accum + item.output_block_index * 64U + lane, weighted);
	}
}

__global__ void transformed_dct_grid_grouped_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                                    const JpegDctDeviceFixedTransformBatchItem* __restrict items,
                                                    const uint32_t* __restrict group_offsets,
                                                    const size_t group_count,
                                                    const uint16_t* __restrict quant_tables,
                                                    const float* __restrict resize_weight_matrices,
                                                    const int32_t clamp_min,
                                                    const int32_t clamp_max,
                                                    float* __restrict y_accum,
                                                    float* __restrict cbcr_accum) {
	const size_t group_idx = static_cast<size_t>(blockIdx.x);
	const auto   lane      = static_cast<uint32_t>(threadIdx.x);
	if (group_idx >= group_count || quant_tables == nullptr || resize_weight_matrices == nullptr) {
		return;
	}
	const uint32_t begin = group_offsets[group_idx];
	const uint32_t end   = group_offsets[group_idx + 1U];
	if (begin >= end) {
		return;
	}

	// The validation profile's dominant resizes downsample either or both axes by
	// two.  Preserve the reference operation graph instead of distributing the
	// transform into algebraically equivalent per-source matrices: compose the
	// source blocks, apply C_y @ source and intermediate @ C_x.T, then let the
	// common round-to-even kernel quantize the result.  For the 2x2 case this also
	// performs 3,072 rather than 4,096 multiply-adds per output block.
	const auto first_item               = items[begin];
	const auto x_down_factor            = static_cast<uint32_t>(first_item.x_down_factor);
	const auto y_down_factor            = static_cast<uint32_t>(first_item.y_down_factor);
	bool       use_reference_down2_axes = first_item.x_up_factor == 1U && first_item.y_up_factor == 1U &&
	                                x_down_factor >= 1U && x_down_factor <= 2U && y_down_factor >= 1U &&
	                                y_down_factor <= 2U && end - begin == x_down_factor * y_down_factor;
	uint8_t subblock_mask = 0U;
	for (uint32_t item_idx = begin; use_reference_down2_axes && item_idx < end; ++item_idx) {
		const auto item          = items[item_idx];
		use_reference_down2_axes = item.x_up_factor == 1U && item.y_up_factor == 1U &&
		                           item.x_down_factor == x_down_factor && item.y_down_factor == y_down_factor &&
		                           item.x_subblock < x_down_factor && item.y_subblock < y_down_factor;
		if (use_reference_down2_axes) {
			const auto slot = static_cast<uint32_t>(item.y_subblock) * x_down_factor + item.x_subblock;
			subblock_mask |= static_cast<uint8_t>(1U << slot);
		}
	}
	const auto expected_subblock_mask =
	    use_reference_down2_axes ? static_cast<uint8_t>((1U << (x_down_factor * y_down_factor)) - 1U) : 0U;
	use_reference_down2_axes = use_reference_down2_axes && subblock_mask == expected_subblock_mask;

	__shared__ float composed[16U * 16U];
	__shared__ float vertical[8U * 16U];
	if (use_reference_down2_axes) {
		const auto source_width  = x_down_factor * 8U;
		const auto source_height = y_down_factor * 8U;
		const auto source_count  = source_width * source_height;
		if (lane < source_count) {
			const auto item     = items[begin + lane / 64U];
			const auto coeff    = static_cast<uint8_t>(lane % 64U);
			const auto physical = natural_to_physical_coeff_device(coeff, item.zigzag_columns);
			const auto binding  = column_bindings[item.binding_base + physical];
			int16_t    value    = 0;
			if (binding.source == DeviceCoeffSource::kI16) {
				value = binding.column_i16[item.row_in_rowgroup];
			} else if (binding.source == DeviceCoeffSource::kI8) {
				value = static_cast<int16_t>(binding.column_i8[item.row_in_rowgroup]);
			}
			const auto quant =
			    static_cast<int32_t>(quant_tables[static_cast<size_t>(item.quant_table_index) * 64U + coeff]);
			const auto source_y = static_cast<uint32_t>(item.y_subblock) * 8U + coeff / 8U;
			const auto source_x = static_cast<uint32_t>(item.x_subblock) * 8U + coeff % 8U;
			composed[source_y * source_width + source_x] =
			    static_cast<float>(min(clamp_max, max(clamp_min, static_cast<int32_t>(value) * quant)));
		}
		__syncthreads();
		if (lane < 8U * source_width) {
			const auto out_y    = lane / source_width;
			const auto source_x = lane % source_width;
			float      sum      = 0.0F;
			if (y_down_factor == 1U) {
				sum = composed[out_y * source_width + source_x];
			} else {
#pragma unroll
				for (uint32_t source_y = 0; source_y < 16U; ++source_y) {
					sum = fmaf(kRgbNoMoreDown2Conversion[out_y * 16U + source_y],
					           composed[source_y * source_width + source_x],
					           sum);
				}
			}
			vertical[out_y * source_width + source_x] = sum;
		}
		__syncthreads();
		if (lane < 64U) {
			const auto out_y = lane / 8U;
			const auto out_x = lane % 8U;
			float      sum   = 0.0F;
			if (x_down_factor == 1U) {
				sum = vertical[out_y * source_width + out_x];
			} else {
#pragma unroll
				for (uint32_t source_x = 0; source_x < 16U; ++source_x) {
					sum = fmaf(vertical[out_y * source_width + source_x],
					           kRgbNoMoreDown2Conversion[out_x * 16U + source_x],
					           sum);
				}
			}
			const auto target         = items[begin];
			const auto factor_product = x_down_factor * y_down_factor;
			const auto value          = factor_product == 4U   ? sum * 0.5F
			                            : factor_product == 2U ? sum / 0x1.6a09e60000000p+0F
			                                                   : sum;
			if (target.component == 0U && y_accum != nullptr) {
				y_accum[target.output_block_index * 64U + lane] = value;
			} else if (target.component != 0U && cbcr_accum != nullptr) {
				cbcr_accum[target.output_block_index * 64U + lane] = value;
			}
		}
		return;
	}

	__shared__ float source[64];
	__shared__ float horizontal[64];
	float            sum = 0.0F;
	for (uint32_t item_idx = begin; item_idx < end; ++item_idx) {
		const auto item = items[item_idx];
		if (lane < 64U) {
			const auto coeff    = static_cast<uint8_t>(lane);
			const auto physical = natural_to_physical_coeff_device(coeff, item.zigzag_columns);
			const auto binding  = column_bindings[item.binding_base + physical];
			int16_t    value    = 0;
			if (binding.source == DeviceCoeffSource::kI16) {
				value = binding.column_i16[item.row_in_rowgroup];
			} else if (binding.source == DeviceCoeffSource::kI8) {
				value = static_cast<int16_t>(binding.column_i8[item.row_in_rowgroup]);
			}
			const auto quant =
			    static_cast<int32_t>(quant_tables[static_cast<size_t>(item.quant_table_index) * 64U + coeff]);
			source[lane] = static_cast<float>(min(clamp_max, max(clamp_min, static_cast<int32_t>(value) * quant)));
		}
		__syncthreads();
		if (lane < 64U) {
			const auto source_y    = static_cast<uint8_t>(lane / 8U);
			const auto out_x_coeff = static_cast<uint8_t>(lane % 8U);
			float      x_sum       = 0.0F;
			for (uint8_t in_x_coeff = 0; in_x_coeff < 8U; ++in_x_coeff) {
				const float wx = resize_weight_matrices[static_cast<size_t>(item.x_weight_matrix_index) * 64U +
				                                        out_x_coeff * 8U + in_x_coeff];
				x_sum += source[source_y * 8U + in_x_coeff] * wx;
			}
			horizontal[lane] = x_sum;
		}
		__syncthreads();
		if (lane < 64U) {
			const auto out_x_coeff = static_cast<uint8_t>(lane % 8U);
			const auto out_y_coeff = static_cast<uint8_t>(lane / 8U);
			float      weighted    = 0.0F;
			for (uint8_t in_y_coeff = 0; in_y_coeff < 8U; ++in_y_coeff) {
				const float wy = resize_weight_matrices[static_cast<size_t>(item.y_weight_matrix_index) * 64U +
				                                        out_y_coeff * 8U + in_y_coeff];
				weighted += horizontal[in_y_coeff * 8U + out_x_coeff] * wy;
			}
			sum += weighted;
		}
	}
	if (lane < 64U) {
		const auto target = items[begin];
		if (target.component == 0U && y_accum != nullptr) {
			y_accum[target.output_block_index * 64U + lane] = sum;
		} else if (target.component != 0U && cbcr_accum != nullptr) {
			cbcr_accum[target.output_block_index * 64U + lane] = sum;
		}
	}
}

__global__ void transformed_dct_grid_cached_kernel(
    const JpegDctDeviceCachedFixedTransformBatchItem* __restrict items,
    const size_t item_count,
    const uint16_t* __restrict quant_tables,
    const float* __restrict resize_weight_matrices,
    const int32_t clamp_min,
    const int32_t clamp_max,
    float* __restrict y_accum,
    float* __restrict cbcr_accum) {
	const size_t item_idx = static_cast<size_t>(blockIdx.x);
	const auto lane = static_cast<uint8_t>(threadIdx.x);
	if (item_idx >= item_count || lane >= 64U || quant_tables == nullptr || resize_weight_matrices == nullptr) {
		return;
	}
	const auto cached = items[item_idx];
	const auto item = cached.transform;
	__shared__ float source[64];
	__shared__ float horizontal[64];
	const auto physical = natural_to_physical_coeff_device(lane, item.zigzag_columns);
	const auto value = cached.dense[item.row_in_rowgroup * 64U + physical];
	const auto quant = static_cast<int32_t>(
	    quant_tables[static_cast<size_t>(item.quant_table_index) * 64U + lane]);
	source[lane] = static_cast<float>(min(clamp_max, max(clamp_min, static_cast<int32_t>(value) * quant)));
	__syncthreads();
	const auto source_y = static_cast<uint8_t>(lane / 8U);
	const auto out_x_coeff = static_cast<uint8_t>(lane % 8U);
	float x_sum = 0.0F;
	for (uint8_t in_x_coeff = 0; in_x_coeff < 8U; ++in_x_coeff) {
		const float wx = resize_weight_matrices[
		    static_cast<size_t>(item.x_weight_matrix_index) * 64U + out_x_coeff * 8U + in_x_coeff];
		x_sum += source[source_y * 8U + in_x_coeff] * wx;
	}
	horizontal[lane] = x_sum;
	__syncthreads();
	const auto out_y_coeff = static_cast<uint8_t>(lane / 8U);
	float weighted = 0.0F;
	for (uint8_t in_y_coeff = 0; in_y_coeff < 8U; ++in_y_coeff) {
		const float wy = resize_weight_matrices[
		    static_cast<size_t>(item.y_weight_matrix_index) * 64U + out_y_coeff * 8U + in_y_coeff];
		weighted += horizontal[in_y_coeff * 8U + out_x_coeff] * wy;
	}
	if (weighted == 0.0F) {
		return;
	}
	if (item.component == 0 && y_accum != nullptr) {
		atomicAdd(y_accum + item.output_block_index * 64U + lane, weighted);
	} else if (item.component != 0 && cbcr_accum != nullptr) {
		atomicAdd(cbcr_accum + item.output_block_index * 64U + lane, weighted);
	}
}

__global__ void gather_decoded_dct_blocks_batch_kernel(const DeviceCoeffBinding* __restrict column_bindings,
                                                       const JpegDctDeviceDecodedGatherBatchItem* __restrict items,
                                                       const size_t item_count,
                                                       int16_t* __restrict out) {
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t total  = item_count * kJpegDctCoefficientCount;
	if (linear >= total) {
		return;
	}
	const size_t item_idx  = linear / kJpegDctCoefficientCount;
	const size_t coeff_idx = linear % kJpegDctCoefficientCount;
	const auto   item      = items[item_idx];
	const auto   binding   = column_bindings[item.source_index * kJpegDctCoefficientCount + coeff_idx];
	int16_t      value     = 0;
	if (binding.source == DeviceCoeffSource::kI16) {
		value = binding.column_i16[item.row_in_rowgroup];
	} else if (binding.source == DeviceCoeffSource::kI8) {
		value = static_cast<int16_t>(binding.column_i8[item.row_in_rowgroup]);
	}
	out[item.output_block_index * kJpegDctCoefficientCount + coeff_idx] = value;
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

template <typename Fn>
void for_each_selected_coefficient(const std::vector<uint8_t>&             selected_coefficients,
                                   const JpegDctCoefficientSelectionShape& selection_shape,
                                   Fn&&                                    fn) {
	if (selection_shape.is_contiguous_prefix()) {
		for (size_t coeff_idx = 0; coeff_idx < selection_shape.count; ++coeff_idx) {
			fn(coeff_idx);
		}
		return;
	}
	for (const auto coeff_idx : selected_coefficients) {
		fn(static_cast<size_t>(coeff_idx));
	}
}

BoundCoeffColumns bind_coeff_columns(const galp::execution::Rowgroup&        rowgroup,
                                     const galp::runtime::ExecutionWorkset&  workset,
                                     const std::vector<uint8_t>&             coefficients_to_bind,
                                     const JpegDctCoefficientSelectionShape& binding_selection_shape,
                                     const size_t                            expr_index_base = 0) {
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
	for_each_selected_coefficient(coefficients_to_bind, binding_selection_shape, [&](const size_t coeff_idx) {
		(void)resolve_column(resolve_column, coeff_idx);
	});
	return bound;
}

void append_compact_column_bindings(const BoundCoeffColumns&       source,
                                    const std::vector<uint8_t>&    active_physical_coefficients,
                                    std::vector<DeviceCoeffBinding>& column_bindings,
                                    std::array<uint32_t, kJpegDctCoefficientCount>& binding_index_by_physical) {
	binding_index_by_physical.fill(std::numeric_limits<uint32_t>::max());
	for (const auto physical_coeff : active_physical_coefficients) {
		const auto coeff_idx = static_cast<size_t>(physical_coeff);
		if (coeff_idx >= source.column_sources.size()) {
			throw std::runtime_error("JPEG DCT physical coefficient column is outside binding range");
		}
		if (source.column_sources[coeff_idx] == DeviceCoeffSource::kMissing) {
			throw std::runtime_error("JPEG DCT projection references an unbound physical coefficient column");
		}
		const auto binding_index                 = static_cast<uint32_t>(column_bindings.size());
		binding_index_by_physical[coeff_idx]     = binding_index;
		column_bindings.push_back(DeviceCoeffBinding {
		    source.columns_i8[coeff_idx], source.columns_i16[coeff_idx], source.column_sources[coeff_idx]});
	}
}

void append_dense_column_bindings(const BoundCoeffColumns&                        source,
                                  std::vector<DeviceCoeffBinding>&                column_bindings,
                                  std::array<uint32_t, kJpegDctCoefficientCount>& binding_index_by_physical) {
	binding_index_by_physical.fill(std::numeric_limits<uint32_t>::max());
	const auto binding_base = static_cast<uint32_t>(column_bindings.size());
	for (size_t coeff_idx = 0; coeff_idx < kJpegDctCoefficientCount; ++coeff_idx) {
		binding_index_by_physical[coeff_idx] = binding_base + static_cast<uint32_t>(coeff_idx);
		column_bindings.push_back(DeviceCoeffBinding {
		    source.columns_i8[coeff_idx], source.columns_i16[coeff_idx], source.column_sources[coeff_idx]});
	}
}

void project_planless_transformed_dct_grid_batch(const std::vector<BoundCoeffColumns>&   sources,
                                                 const std::vector<DecodedRowgroupWork>& works,
                                                 const uint16_t*                         quant_tables,
                                                 const float*                            phase_matrices,
                                                 const JpegDctGridTransformSpec&         transform,
                                                 float*                                  y_accum,
                                                 float*                                  cbcr_accum,
                                                 JpegDctDeviceScratch&                   scratch,
                                                 JpegDctDeviceExecutionStats&            stats,
                                                 const size_t                            transform_blocks_per_launch,
                                                 const size_t                            transform_ctas_per_launch,
                                                 cudaStream_t                            stream) {
	if (sources.empty() || works.empty()) {
		return;
	}
	if (quant_tables == nullptr || sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT planless transform has invalid sources or quantization tables");
	}
	auto& column_bindings = scratch.host_column_bindings;
	auto& images          = scratch.host_planless_image_descriptors;
	column_bindings.clear();
	images.clear();
	column_bindings.reserve(works.size() * kJpegDctCoefficientCount);
	size_t image_count = 0;
	for (const auto& work : works) {
		if (work.planless_images != nullptr) {
			image_count += work.planless_images->size();
		}
	}
	images.reserve(image_count);
	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto                                     binding_base = static_cast<uint32_t>(column_bindings.size());
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		append_dense_column_bindings(sources[source_idx], column_bindings, binding_index_by_physical);
		const auto* source_images = works[source_idx].planless_images;
		if (source_images == nullptr) {
			continue;
		}
		for (auto image : *source_images) {
			image.binding_base = binding_base;
			images.push_back(image);
		}
	}
	if (images.empty()) {
		return;
	}
	record_planless_transform_resources(scratch, stats);
	const uint64_t blocks_per_image =
	    static_cast<uint64_t>(transform.y_output_width_blocks) * transform.y_output_height_blocks +
	    2U * static_cast<uint64_t>(transform.cbcr_output_width_blocks) * transform.cbcr_output_height_blocks;
	const uint64_t output_blocks = blocks_per_image * images.size();
	if (output_blocks > std::numeric_limits<unsigned>::max()) {
		throw std::runtime_error("JPEG DCT planless transform grid exceeds CUDA launch range");
	}
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.planless_image_descriptors.upload(images.data(), images.size(), stream, stats);
	const uint64_t launch_output_limit = transform_blocks_per_launch == 0
	                                         ? output_blocks
	                                         : std::min<uint64_t>(output_blocks, transform_blocks_per_launch);
	for (uint64_t offset = 0; offset < output_blocks; offset += launch_output_limit) {
		const auto launch_output_blocks = std::min<uint64_t>(launch_output_limit, output_blocks - offset);
		const auto limited_cta_limit    = transform_ctas_per_launch == 0
		                                      ? static_cast<uint64_t>(kLimitedPlanlessTransformCtasPerLaunch)
		                                      : static_cast<uint64_t>(transform_ctas_per_launch);
		const auto launch_ctas          = static_cast<unsigned>(
            transform_blocks_per_launch == 0 ? launch_output_blocks
                                             : std::min<uint64_t>(launch_output_blocks, limited_cta_limit));
		transformed_dct_grid_planless_kernel<<<dim3(launch_ctas), dim3(kPlanlessTransformThreadsPerCta), 0, stream>>>(
		    scratch.column_bindings.data,
		    scratch.planless_image_descriptors.data,
		    images.size(),
		    offset,
		    launch_output_blocks,
		    quant_tables,
		    phase_matrices,
		    transform.y_output_width_blocks,
		    transform.y_output_height_blocks,
		    transform.cbcr_output_width_blocks,
		    transform.cbcr_output_height_blocks,
		    transform.clamp_min,
		    transform.clamp_max,
		    y_accum,
		    cbcr_accum);
		CUDA_SAFE_CALL(cudaGetLastError());
		++stats.materialize_kernel_launch_count;
		++stats.planless_transform_kernel_launch_count;
		stats.planless_transform_max_blocks_per_launch =
		    std::max(stats.planless_transform_max_blocks_per_launch, static_cast<size_t>(launch_ctas));
		stats.planless_transform_max_output_blocks_per_launch =
		    std::max(stats.planless_transform_max_output_blocks_per_launch, static_cast<size_t>(launch_output_blocks));
	}
	stats.planless_image_descriptor_count += images.size();
	stats.planless_transform_output_block_count += output_blocks;
	stats.device_mapping_fused = true;
}

void project_transformed_dct_grid_batch(const std::vector<BoundCoeffColumns>&   sources,
                                        const std::vector<DecodedRowgroupWork>& works,
                                        const std::vector<uint32_t>*            fixed_transform_item_order,
                                        const std::vector<uint32_t>*            fixed_transform_group_offsets,
                                        const uint16_t*                         quant_tables,
                                        const float*                            resize_weight_matrices,
                                        const JpegDctGridTransformSpec&         transform,
                                        float*                                  y_accum,
                                        float*                                  cbcr_accum,
                                        JpegDctDeviceScratch&                   scratch,
                                        JpegDctDeviceExecutionStats&            stats,
                                        cudaStream_t                            stream) {
	if (sources.empty() || works.empty()) {
		return;
	}
	if (quant_tables == nullptr || resize_weight_matrices == nullptr) {
		throw std::runtime_error("JPEG DCT fixed transform requires uploaded quantization tables and resize weights");
	}
	if (sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT fixed transform source/work count mismatch");
	}
	auto& column_bindings       = scratch.host_column_bindings;
	auto& source_order_items    = scratch.host_fixed_transform_items;
	auto& ordered_items         = scratch.host_ordered_fixed_transform_items;
	column_bindings.clear();
	source_order_items.clear();
	ordered_items.clear();

	size_t total_items = 0;
	for (const auto& work : works) {
		if (work.fixed_transform_items != nullptr) {
			total_items += work.fixed_transform_items->size();
		}
	}
	const bool deterministic = fixed_transform_item_order != nullptr && fixed_transform_group_offsets != nullptr &&
	                           !fixed_transform_item_order->empty();
	if (deterministic &&
	    (fixed_transform_item_order->size() != total_items || fixed_transform_group_offsets->size() < 2U ||
	     fixed_transform_group_offsets->front() != 0U || fixed_transform_group_offsets->back() != total_items)) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan does not match decoded items");
	}
	auto& transform_items = deterministic ? ordered_items : source_order_items;
	column_bindings.reserve(works.size() * kJpegDctCoefficientCount);
	if (deterministic) {
		transform_items.resize(total_items);
	} else {
		transform_items.reserve(total_items);
	}
	size_t flat_item_index = 0;
	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& source = sources[source_idx];
		const auto& work   = works[source_idx];
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		const auto binding_base = static_cast<uint32_t>(column_bindings.size());
		append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
		if (work.fixed_transform_items == nullptr) {
			continue;
		}
		for (const auto& item : *work.fixed_transform_items) {
			const auto device_item =
			    JpegDctDeviceFixedTransformBatchItem {binding_base,
			                                          item.row_in_rowgroup,
			                                          item.output_block_index,
			                                          item.component,
			                                          static_cast<uint8_t>(item.zigzag_columns ? 1U : 0U),
			                                          item.x_factor,
			                                          item.y_factor,
			                                          item.x_subblock,
			                                          item.y_subblock,
			                                          static_cast<uint8_t>(item.x_upsample ? 1U : 0U),
			                                          static_cast<uint8_t>(item.y_upsample ? 1U : 0U),
			                                          item.x_up_factor,
			                                          item.y_up_factor,
			                                          item.x_down_factor,
			                                          item.y_down_factor,
			                                          item.quant_table_index,
			                                          item.x_weight_matrix_index,
			                                          item.y_weight_matrix_index};
			if (deterministic) {
				const auto ordered_index = (*fixed_transform_item_order)[flat_item_index];
				if (ordered_index >= transform_items.size()) {
					throw std::runtime_error("JPEG DCT deterministic fixed-transform permutation is out of range");
				}
				transform_items[ordered_index] = device_item;
			} else {
				transform_items.push_back(device_item);
			}
			++flat_item_index;
		}
	}
	if (transform_items.empty()) {
		return;
	}
	// The deterministic grouped kernel uses one thread per coefficient in a
	// composed factor-2 16x16 source block.  Its generic fallback keeps only the
	// first 64 lanes active while all lanes participate in barriers.
	constexpr unsigned kGroupedThreads = 256;
	constexpr unsigned kSourceThreads  = 64;
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.batch_fixed_transform_items.upload(transform_items.data(), transform_items.size(), stream, stats);
	const dim3 block(deterministic ? kGroupedThreads : kSourceThreads);
	if (deterministic) {
		scratch.fixed_transform_group_offsets.upload(
		    fixed_transform_group_offsets->data(), fixed_transform_group_offsets->size(), stream, stats);
		const size_t group_count = fixed_transform_group_offsets->size() - 1U;
		const dim3   grid(static_cast<unsigned>(group_count));
		transformed_dct_grid_grouped_kernel<<<grid, block, 0, stream>>>(scratch.column_bindings.data,
		                                                                scratch.batch_fixed_transform_items.data,
		                                                                scratch.fixed_transform_group_offsets.data,
		                                                                group_count,
		                                                                quant_tables,
		                                                                resize_weight_matrices,
		                                                                transform.clamp_min,
		                                                                transform.clamp_max,
		                                                                y_accum,
		                                                                cbcr_accum);
	} else {
		const dim3 grid(static_cast<unsigned>(transform_items.size()));
		transformed_dct_grid_sources_kernel<<<grid, block, 0, stream>>>(scratch.column_bindings.data,
		                                                                scratch.batch_fixed_transform_items.data,
		                                                                transform_items.size(),
		                                                                quant_tables,
		                                                                resize_weight_matrices,
		                                                                transform.clamp_min,
		                                                                transform.clamp_max,
		                                                                y_accum,
		                                                                cbcr_accum);
	}
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	stats.fixed_transform_item_count += transform_items.size();
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

	void gather_decoded_rowgroup_batch(const std::vector<BoundCoeffColumns>& sources,
	                                   const std::vector<DecodedRowgroupWork>& works,
	                                   int16_t* output,
	                                   JpegDctDeviceScratch& scratch,
	                                   JpegDctDeviceExecutionStats& stats,
	                                   cudaStream_t stream) {
		if (sources.empty() || works.empty()) {
			return;
		}
		if (sources.size() != works.size()) {
			throw std::runtime_error("JPEG DCT decoded gather source/work count mismatch");
		}

		auto& column_bindings = scratch.host_column_bindings;
		auto& gather_items    = scratch.host_decoded_gather_items;
		column_bindings.clear();
		gather_items.clear();

		size_t total_gather_items = 0;
		for (const auto& work : works) {
			if (work.gather_items != nullptr) {
				total_gather_items += work.gather_items->size();
			}
		}
		column_bindings.reserve(works.size() * kJpegDctCoefficientCount);
		gather_items.reserve(total_gather_items);

		for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
			const auto& source = sources[source_idx];
			const auto& work   = works[source_idx];
			std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
			append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
			if (work.gather_items == nullptr) {
				continue;
			}
			if (source_idx > std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT decoded gather source index overflow");
			}
			for (const auto& item : *work.gather_items) {
				gather_items.push_back(JpegDctDeviceDecodedGatherBatchItem {
				    static_cast<uint32_t>(source_idx), item.row_in_rowgroup, item.output_block_index});
			}
		}
		if (gather_items.empty()) {
			return;
		}

		constexpr unsigned kThreads = 256;
		scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
		scratch.decoded_gather_items.upload(gather_items.data(), gather_items.size(), stream, stats);
		const size_t total = gather_items.size() * kJpegDctCoefficientCount;
		const dim3   block(kThreads);
		const dim3   grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
		gather_decoded_dct_blocks_batch_kernel<<<grid, block, 0, stream>>>(
		    scratch.column_bindings.data, scratch.decoded_gather_items.data, gather_items.size(), output);
		CUDA_SAFE_CALL(cudaGetLastError());
		++stats.gather_kernel_launch_count;
		stats.gather_item_count += gather_items.size();
		stats.decoded_gather_item_count += gather_items.size();
	}

	void project_decoded_rowgroup_batch(const std::vector<BoundCoeffColumns>& sources,
	                                    const std::vector<DecodedRowgroupWork>& works,
                                    const size_t coefficients_per_block,
                                    const bool use_dense_bindings,
                                    int16_t* output,
                                    JpegDctDeviceScratch& scratch,
                                    JpegDctDeviceExecutionStats& stats,
                                    cudaStream_t stream) {
	if (sources.empty() || works.empty() || coefficients_per_block == 0) {
		return;
	}
	if (sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT projection source/work count mismatch");
	}

	auto& column_bindings  = scratch.host_column_bindings;
	auto& projection_items = scratch.host_projection_items;
	column_bindings.clear();
	projection_items.clear();
	const auto build_start = Clock::now();

	size_t total_projection_items = 0;
	size_t total_active_columns   = 0;
	for (const auto& work : works) {
		if (work.projection_items != nullptr) {
			total_projection_items += work.projection_items->size();
		}
		total_active_columns += use_dense_bindings ? kJpegDctCoefficientCount : work.active_physical_coefficients.size();
	}
	column_bindings.reserve(total_active_columns);
	projection_items.reserve(total_projection_items);

	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& source = sources[source_idx];
		const auto& work   = works[source_idx];
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		if (use_dense_bindings) {
			append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
		} else {
			append_compact_column_bindings(
			    source, work.active_physical_coefficients, column_bindings, binding_index_by_physical);
		}
		if (work.projection_items == nullptr) {
			continue;
		}
		for (const auto& item : *work.projection_items) {
			const auto physical_coeff = static_cast<size_t>(item.physical_coefficient_column_id);
			if (physical_coeff >= binding_index_by_physical.size() ||
			    binding_index_by_physical[physical_coeff] == std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT projection references a physical column that was not bound");
			}
			projection_items.push_back(JpegDctDeviceProjectionBatchItem {
			    binding_index_by_physical[physical_coeff],
			    item.row_in_rowgroup,
			    item.output_block_index,
			    item.selected_coefficient_slot,
			    item.logical_coefficient_id,
			    item.physical_coefficient_column_id,
			    item.output_coefficient_id,
			    item.output_grid_tensor});
		}
	}
	if (projection_items.empty()) {
		stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());
		return;
	}
	stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());

	constexpr unsigned kThreads = 256;
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.batch_projection_items.upload(projection_items.data(), projection_items.size(), stream, stats);
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>((projection_items.size() + kThreads - 1U) / kThreads));
	project_dct_coefficients_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.column_bindings.data,
	    scratch.batch_projection_items.data,
	    projection_items.size(),
	    coefficients_per_block,
	    output);
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	stats.projection_item_count += projection_items.size();
	stats.decoded_projection_item_count += projection_items.size();
	stats.jpeg_dct_projection_items_materialized += projection_items.size();
}

void project_decoded_ycbcr_grid_batch(const std::vector<BoundCoeffColumns>&   sources,
                                      const std::vector<DecodedRowgroupWork>& works,
                                      const bool                              use_dense_bindings,
                                      int16_t*                                y_output,
                                      int16_t*                                cbcr_output,
                                      float*                                  y_accum,
                                      float*                                  cbcr_accum,
                                      JpegDctDeviceScratch&                   scratch,
                                      JpegDctDeviceExecutionStats&            stats,
                                      cudaStream_t                            stream) {
	if (sources.empty() || works.empty()) {
		return;
	}
	if (sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT YCbCr-grid projection source/work count mismatch");
	}

	auto& column_bindings  = scratch.host_column_bindings;
	auto& projection_items = scratch.host_projection_items;
	column_bindings.clear();
	projection_items.clear();
	const auto build_start = Clock::now();

	size_t total_projection_items = 0;
	size_t total_active_columns   = 0;
	for (const auto& work : works) {
		if (work.projection_items != nullptr) {
			total_projection_items += work.projection_items->size();
		}
		total_active_columns += use_dense_bindings ? kJpegDctCoefficientCount : work.active_physical_coefficients.size();
	}
	column_bindings.reserve(total_active_columns);
	projection_items.reserve(total_projection_items);

	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& source = sources[source_idx];
		const auto& work   = works[source_idx];
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		if (use_dense_bindings) {
			append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
		} else {
			append_compact_column_bindings(
			    source, work.active_physical_coefficients, column_bindings, binding_index_by_physical);
		}
		if (work.projection_items == nullptr) {
			continue;
		}
		for (const auto& item : *work.projection_items) {
			const auto physical_coeff = static_cast<size_t>(item.physical_coefficient_column_id);
			if (physical_coeff >= binding_index_by_physical.size() ||
			    binding_index_by_physical[physical_coeff] == std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT YCbCr-grid projection references an unbound physical column");
			}
			projection_items.push_back(JpegDctDeviceProjectionBatchItem {
			    binding_index_by_physical[physical_coeff],
			    item.row_in_rowgroup,
			    item.output_block_index,
			    item.selected_coefficient_slot,
			    item.logical_coefficient_id,
			    item.physical_coefficient_column_id,
			    item.output_coefficient_id,
			    item.output_grid_tensor,
			    item.weight});
		}
	}
	if (projection_items.empty()) {
		stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());
		return;
	}
	stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());

	constexpr unsigned kThreads = 256;
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.batch_projection_items.upload(projection_items.data(), projection_items.size(), stream, stats);
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>((projection_items.size() + kThreads - 1U) / kThreads));
	project_dct_ycbcr_grid_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.column_bindings.data,
	    scratch.batch_projection_items.data,
	    projection_items.size(),
	    y_output,
	    cbcr_output,
	    y_accum,
	    cbcr_accum);
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	++stats.project_decoded_ycbcr_grid_launch_count;
	stats.projection_item_count += projection_items.size();
	stats.decoded_projection_item_count += projection_items.size();
	stats.jpeg_dct_projection_items_materialized += projection_items.size();
}

void materialize_dense_rowgroup_batch(const std::vector<JpegDctDeviceMaterializeBatchItem>& items,
                                      JpegDctDeviceScratch&                                 scratch,
                                      JpegDctDeviceExecutionStats&                          stats,
                                      cudaStream_t                                          stream) {
	if (items.empty()) {
		return;
	}
	// Reuses the column pointer/source scratch uploaded for the preceding batch projection on this stream.
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

void round_fixed_ycbcr_grid_outputs(float*                          y_accum,
                                    float*                          cbcr_accum,
                                    int16_t*                        y_output,
                                    int16_t*                        cbcr_output,
                                    const size_t                    y_count,
                                    const size_t                    cbcr_count,
                                    JpegDctDeviceScratch&           scratch,
                                    JpegDctDeviceExecutionStats&    stats,
                                    galp::memory::CudaEvent&        timing_start_event,
                                    galp::memory::CudaEvent&        completion_event,
                                    const JpegDctGridTransformSpec& transform) {
	constexpr unsigned kThreads     = 256;
	const bool         float_output = transform.output_data_type == JpegDctGridOutputDataType::kFloat32;
	const bool         launch_y     = y_accum != nullptr && y_count != 0 && (float_output || y_output != nullptr);
	const bool launch_cbcr = cbcr_accum != nullptr && cbcr_count != 0 && (float_output || cbcr_output != nullptr);
	if (!launch_y && !launch_cbcr) {
		return;
	}
	const size_t       active_y_count    = launch_y ? y_count : 0U;
	const size_t       active_cbcr_count = launch_cbcr ? cbcr_count : 0U;
	const size_t       total_count       = active_y_count + active_cbcr_count;
	const cudaStream_t stream            = scratch.stream_for_fixed_grid_rounding();
	timing_start_event.create();
	completion_event.create();
	timing_start_event.record(stream);
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>((total_count + kThreads - 1U) / kThreads));
	if (float_output) {
		round_affine_dct_grid_accum_pair_kernel<<<grid, block, 0, stream>>>(
		    y_accum, active_y_count, cbcr_accum, active_cbcr_count, transform.output_add, transform.output_scale);
	} else {
		round_dct_grid_accum_pair_kernel<<<grid, block, 0, stream>>>(
		    y_accum, active_y_count, cbcr_accum, active_cbcr_count, y_output, cbcr_output);
	}
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	++stats.fixed_grid_finalize_kernel_launch_count;
	completion_event.record(stream);
	++stats.fixed_grid_round_event_handoff_count;
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

DecodedRowgroupWork
prepare_decoded_rowgroup_work_from_materialized(galp::execution::Rowgroup               rowgroup,
                                                uint32_t                                shard_id,
                                                const JpegDctDeviceRowgroupPlan&        rowgroup_plan,
                                                const std::vector<uint8_t>&             selected_coefficients,
                                                const JpegDctCoefficientSelectionShape& selection_shape,
                                                bool                                    force_projection,
                                                unsigned                                decode_unpack_n_vectors,
                                                JpegDctDeviceDecodedRowgroupCache*      cache,
                                                JpegDctDeviceExecutionStats&            execution_stats);

DecodedRowgroupWork prepare_decoded_rowgroup_work(galp::format::FlsReader&                rdr,
                                                  const uint32_t                          shard_id,
                                                  const JpegDctDeviceRowgroupPlan&        rowgroup_plan,
                                                  const std::vector<uint8_t>&             selected_coefficients,
                                                  const JpegDctCoefficientSelectionShape& selection_shape,
                                                  const bool                              force_projection,
                                                  const unsigned                          decode_unpack_n_vectors,
                                                  JpegDctDeviceDecodedRowgroupCache*      cache,
                                                  JpegDctDeviceExecutionStats&            execution_stats) {
	const auto                       sync_read_start = Clock::now();
	galp::format::ZeroCopyReadTiming io_timing {};
	auto                             zero_copy = rdr.read_rowgroup_zero_copy(rowgroup_plan.rowgroup_index, &io_timing);
	auto                             rowgroup  = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
	const auto                       sync_read_end = Clock::now();
	execution_stats.sync_rowgroup_read_ms += elapsed_ms(sync_read_start, sync_read_end);
	execution_stats.rowgroup_storage_bytes_read += io_timing.storage_bytes;

	return prepare_decoded_rowgroup_work_from_materialized(std::move(rowgroup),
	                                                       shard_id,
	                                                       rowgroup_plan,
	                                                       selected_coefficients,
	                                                       selection_shape,
	                                                       force_projection,
	                                                       decode_unpack_n_vectors,
	                                                       cache,
	                                                       execution_stats);
}

DecodedRowgroupWork
prepare_decoded_rowgroup_work_from_materialized(galp::execution::Rowgroup               rowgroup,
                                                const uint32_t                          shard_id,
                                                const JpegDctDeviceRowgroupPlan&        rowgroup_plan,
                                                const std::vector<uint8_t>&             selected_coefficients,
                                                const JpegDctCoefficientSelectionShape& selection_shape,
                                                const bool                              force_projection,
                                                const unsigned                          decode_unpack_n_vectors,
                                                JpegDctDeviceDecodedRowgroupCache*      cache,
                                                JpegDctDeviceExecutionStats&            execution_stats) {
	DecodedRowgroupWork work;
	work.shard_id       = shard_id;
	work.rowgroup_index = rowgroup_plan.rowgroup_index;
	work.cache_key      = JpegDctDeviceDecodedRowgroupCacheKey {shard_id, rowgroup_plan.rowgroup_index};
	work.rowgroup       = std::move(rowgroup);
	work.owns_rowgroup  = true;
	// JPEG's projection/selected-vector path appends columns directly instead
	// of using append_expressions(), so resolve external dictionaries here.
	// This keeps DICTREF strictly host-side and guarantees that cache misses,
	// selected-vector decodes, and full-rowgroup decodes all upload local plans.
	{
		auto expressions = galp::expression::assemble(work.rowgroup);
		galp::execution::resolve_dict_refs(expressions);
	}
	if (work.rowgroup.columns.size() < 64) {
		throw std::runtime_error("JPEG DCT FLS rowgroup has fewer than 64 coefficient columns");
	}

	galp::execution::ExecutionConfig cfg;
	cfg.write_out                                    = true;
	cfg.unpack_n_vectors                             = std::max(1U, decode_unpack_n_vectors);
	work.decode_unpack_n_vectors                     = cfg.unpack_n_vectors;
	size_t                     selected_vector_count = 0;
	JpegDctRuntimePolicyResult policy {};
	const bool                 can_reuse_vector_plan = rowgroup_plan.has_vector_plan &&
	                                   rowgroup_plan.full_vector_count == work.rowgroup.n_vecs &&
	                                   cfg.unpack_n_vectors == kJpegDctDeviceUnpackNVectors;
	if (can_reuse_vector_plan) {
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
	if (work.decodes_full_rowgroup && work.rowgroup.n_vecs % cfg.unpack_n_vectors != 0U) {
		throw std::runtime_error("JPEG DCT batch unpack width is incompatible with a full rowgroup");
	}
	if (work.decodes_full_rowgroup) {
		work.gather_items = &rowgroup_plan.items;
	} else if (can_reuse_vector_plan && !rowgroup_plan.selected_gather_items.empty()) {
		work.gather_items = &rowgroup_plan.selected_gather_items;
	} else {
		work.owned_gather_items =
		    remap_items_to_selected_vectors(rowgroup_plan.items, *work.selected_vectors, cfg.unpack_n_vectors);
		work.gather_items = &work.owned_gather_items;
	}
	if (!rowgroup_plan.planless_images.empty()) {
		work.active_physical_coefficients = selected_coefficients;
		work.planless_images              = &rowgroup_plan.planless_images;
	} else if (!rowgroup_plan.fixed_transform_items.empty()) {
		work.active_physical_coefficients = selected_coefficients;
		if (work.decodes_full_rowgroup) {
			work.fixed_transform_items = &rowgroup_plan.fixed_transform_items;
		} else if (can_reuse_vector_plan && !rowgroup_plan.selected_fixed_transform_items.empty()) {
			work.fixed_transform_items = &rowgroup_plan.selected_fixed_transform_items;
		} else {
			work.owned_fixed_transform_items = remap_fixed_transform_items_to_selected_vectors(
			    rowgroup_plan.fixed_transform_items, *work.selected_vectors, cfg.unpack_n_vectors);
			work.fixed_transform_items = &work.owned_fixed_transform_items;
		}
	} else if (selection_shape.kind == JpegDctCoefficientSelectionKind::kAll && !force_projection) {
		work.active_physical_coefficients = selected_coefficients;
	} else {
		const std::vector<JpegDctDeviceProjectionItem>* logical_projection_items = nullptr;
		if (work.decodes_full_rowgroup) {
			logical_projection_items = &rowgroup_plan.projection_items;
		} else if (can_reuse_vector_plan && !rowgroup_plan.selected_projection_items.empty()) {
			logical_projection_items = &rowgroup_plan.selected_projection_items;
		} else {
			work.owned_projection_items = remap_projection_items_to_selected_vectors(
			    rowgroup_plan.projection_items, *work.selected_vectors, cfg.unpack_n_vectors);
			logical_projection_items = &work.owned_projection_items;
		}
		if (logical_projection_items == nullptr || logical_projection_items->empty()) {
			throw std::runtime_error("JPEG DCT rowgroup projection plan is missing");
		}
		auto resolved_projection    = resolve_projection_physical_columns(work.rowgroup, *logical_projection_items);
		work.owned_projection_items = std::move(resolved_projection.items);
		work.projection_items       = &work.owned_projection_items;
		work.active_physical_coefficients = std::move(resolved_projection.active_physical_coefficients);
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
	if (scratch.cached_fixed_transform_in_flight) {
		execution_stats.fixed_transform_ms += elapsed_ms;
	} else {
		execution_stats.cached_gather_ms += elapsed_ms;
		execution_stats.gather_ms += elapsed_ms;
	}
	if (scratch.cache_hit_stream) {
		galp::memory::complete_h2d(scratch.cache_hit_stream.get());
	}
	scratch.host_cached_gather_uploads.clear();
	scratch.host_cached_fixed_transform_uploads.clear();
	scratch.cached_gather_in_flight          = false;
	scratch.cached_fixed_transform_in_flight = false;
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

void execute_decoded_rowgroup_batch(std::vector<DecodedRowgroupWork>&       works,
                                    const std::vector<uint32_t>*            fixed_transform_item_order,
                                    const std::vector<uint32_t>*            fixed_transform_group_offsets,
                                    int16_t*                                output,
                                    int16_t*                                y_output,
                                    int16_t*                                cbcr_output,
                                    float*                                  y_accum,
                                    float*                                  cbcr_accum,
                                    const uint16_t*                         fixed_quant_tables,
                                    const float*                            fixed_resize_weight_matrices,
                                    const JpegDctGridTransformSpec&         grid_transform,
                                    const std::vector<uint8_t>&             selected_coefficients,
                                    const JpegDctCoefficientSelectionShape& selection_shape,
                                    const bool                              output_ycbcr_dct_grid,
                                    const bool                              output_transformed_dct_grid,
                                    JpegDctDeviceDecodedRowgroupCache*      cache,
                                    JpegDctDeviceCacheStats&                batch_cache_stats,
                                    JpegDctDeviceExecutionStats&            execution_stats,
                                    JpegDctDeviceScratch&                   scratch) {
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
	galp::runtime::reserve_batch_expr_storage(workset, works.size() * kJpegDctCoefficientCount);
	for (size_t idx = 0; idx < works.size(); ++idx) {
		auto& work                          = works[idx];
		work.expr_index_base                = idx * 64U;
		const auto physical_selection_shape = classify_coefficient_selection(work.active_physical_coefficients);
		append_jpeg_rowgroup_columns(workset,
		                             work.rowgroup,
		                             cfg,
		                             work.expr_index_base,
		                             work.active_physical_coefficients,
		                             physical_selection_shape,
		                             work.decodes_full_rowgroup ? nullptr : work.selected_vectors);
	}
	const auto build_end = Clock::now();
	execution_stats.workset_build_ms += elapsed_ms(build_start, build_end);
	const auto upload_start     = Clock::now();
	const auto upload_breakdown = galp::runtime::upload_workset(workset, cfg);
	const auto upload_end       = Clock::now();
	execution_stats.workset_upload_ms += elapsed_ms(upload_start, upload_end);
	execution_stats.workset_upload_prep_ms += upload_breakdown.prep_ms;
	execution_stats.workset_upload_arena_ms += upload_breakdown.arena_upload_ms;
	execution_stats.workset_upload_arena_pack_ms += upload_breakdown.arena_pack_ms;
	execution_stats.workset_upload_arena_layout_ms += upload_breakdown.arena.layout_ms;
	execution_stats.workset_upload_arena_alloc_ms += upload_breakdown.arena.alloc_ms;
	execution_stats.workset_upload_arena_resolve_ms += upload_breakdown.arena.resolve_ms;
	execution_stats.workset_upload_dma_issue_ms += upload_breakdown.arena.dma_issue_ms;
	execution_stats.workset_upload_event_record_ms += upload_breakdown.event_record_ms;
	execution_stats.workset_upload_dma_bytes += upload_breakdown.arena.dma_bytes;
	execution_stats.workset_upload_dma_count += upload_breakdown.arena.dma_count;
	++execution_stats.workset_count;
	++execution_stats.workset_upload_count;
	size_t launches = 0;
	auto   run      = galp::runtime::run_workset_async(workset, 1, cfg, nullptr, &launches);
	execution_stats.decode_kernel_launch_count += launches;
	if (galp::runtime::use_async_h2d() && workset.transfer.h2d_stream && workset.transfer.h2d_ready_event) {
		++execution_stats.copy_to_decode_event_handoff_count;
	}
	const cudaStream_t decode_stream = run.stream;

	auto& sources = scratch.host_bound_sources;
	sources.clear();
	sources.reserve(works.size());
	const bool materializes_dense_cache =
	    cache != nullptr && cache->capacity > 0 && selects_all_coefficients(selected_coefficients);
	const bool uses_decoded_gather =
	    selection_shape.kind == JpegDctCoefficientSelectionKind::kAll && !output_ycbcr_dct_grid;
	const bool batch_has_expanded_fixed_transform =
	    output_transformed_dct_grid && std::any_of(works.begin(), works.end(), [](const auto& work) {
		    return work.fixed_transform_items != nullptr && !work.fixed_transform_items->empty();
	    });
	const bool batch_has_planless_fixed_transform =
	    output_transformed_dct_grid && std::any_of(works.begin(), works.end(), [](const auto& work) {
		    return work.planless_images != nullptr && !work.planless_images->empty();
	    });
	if (batch_has_expanded_fixed_transform && batch_has_planless_fixed_transform) {
		throw std::runtime_error("JPEG DCT workset mixed expanded and planless fixed transforms");
	}
	const bool batch_has_fixed_transform = batch_has_expanded_fixed_transform || batch_has_planless_fixed_transform;
	cudaStream_t materialize_stream = decode_stream;
	if (batch_has_fixed_transform) {
		scratch.decode_to_transform_event.create_with_flags(cudaEventDisableTiming);
		scratch.decode_to_transform_event.record(decode_stream);
		materialize_stream = scratch.stream_for_transform();
		CUDA_SAFE_CALL(cudaStreamWaitEvent(materialize_stream, scratch.decode_to_transform_event.get(), 0));
		++execution_stats.decode_to_transform_event_handoff_count;
	}
	make_stream_wait_for_cached_gather(materialize_stream, scratch, execution_stats);
	const bool use_dense_bindings        = materializes_dense_cache || uses_decoded_gather || batch_has_fixed_transform;
	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& work = works[source_idx];
		if (use_dense_bindings) {
			sources.push_back(bind_coeff_columns(
			    work.rowgroup, workset, selected_coefficients, selection_shape, work.expr_index_base));
		} else {
			const auto physical_selection_shape = classify_coefficient_selection(work.active_physical_coefficients);
			sources.push_back(bind_coeff_columns(work.rowgroup,
			                                     workset,
			                                     work.active_physical_coefficients,
			                                     physical_selection_shape,
			                                     work.expr_index_base));
		}
	}
	if (batch_has_planless_fixed_transform) {
		project_planless_transformed_dct_grid_batch(sources,
		                                            works,
		                                            fixed_quant_tables,
		                                            fixed_resize_weight_matrices,
		                                            grid_transform,
		                                            y_accum,
		                                            cbcr_accum,
		                                            scratch,
		                                            execution_stats,
		                                            scratch.transform_blocks_per_launch,
		                                            scratch.transform_ctas_per_launch,
		                                            materialize_stream);
	} else if (batch_has_expanded_fixed_transform) {
		project_transformed_dct_grid_batch(sources,
		                                   works,
		                                   fixed_transform_item_order,
		                                   fixed_transform_group_offsets,
		                                   fixed_quant_tables,
		                                   fixed_resize_weight_matrices,
		                                   grid_transform,
		                                   y_accum,
		                                   cbcr_accum,
		                                   scratch,
		                                   execution_stats,
		                                   materialize_stream);
	} else if (uses_decoded_gather) {
		gather_decoded_rowgroup_batch(sources, works, output, scratch, execution_stats, materialize_stream);
	} else if (output_ycbcr_dct_grid) {
		project_decoded_ycbcr_grid_batch(sources,
		                                 works,
		                                 materializes_dense_cache,
		                                 y_output,
		                                 cbcr_output,
		                                 y_accum,
		                                 cbcr_accum,
		                                 scratch,
		                                 execution_stats,
		                                 materialize_stream);
	} else {
		project_decoded_rowgroup_batch(sources,
		                               works,
		                               selected_coefficients.size(),
		                               materializes_dense_cache,
		                               output,
		                               scratch,
		                               execution_stats,
		                               materialize_stream);
	}
	if (materializes_dense_cache) {
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
		materialize_dense_rowgroup_batch(materialize_items, scratch, execution_stats, materialize_stream);
	}
	scratch.ensure_decoded_batch_events();
	scratch.decoded_batch_gather_done.record(materialize_stream);

	// Rowgroup metadata, workset output arena, and scratch are reused after this batch.
	// Wait only for the projection completion event; batch-level workset ownership can remove this later.
	scratch.decoded_batch_gather_done.synchronize();
	++execution_stats.internal_sync_count;
	++execution_stats.decoded_batch_sync_count;
	if (decode_stream != nullptr) {
		galp::memory::complete_h2d(decode_stream);
	}
	if (materialize_stream != nullptr && materialize_stream != decode_stream) {
		galp::memory::complete_h2d(materialize_stream);
	}
	finish_cached_gather_after_wait(scratch, execution_stats);
	if (run.stop != nullptr) {
		const auto elapsed_ms = static_cast<double>(scratch.decoded_batch_gather_done.elapsed_since(*run.stop));
		if (uses_decoded_gather) {
			execution_stats.decoded_gather_ms += elapsed_ms;
			execution_stats.gather_ms += elapsed_ms;
		} else if (batch_has_fixed_transform) {
			execution_stats.fixed_transform_ms += elapsed_ms;
		} else {
			execution_stats.decoded_projection_ms += elapsed_ms;
			execution_stats.projection_ms += elapsed_ms;
		}
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
	if (scratch.cached_gather_in_flight && scratch.cached_fixed_transform_in_flight) {
		drain_cached_gather(scratch, execution_stats);
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
		scratch.cached_fixed_transform_in_flight = false;
	}
	gather_cached_rowgroups_batch(upload_items, output, scratch, execution_stats, stream);
	scratch.cached_gather_done.record(stream);
	scratch.cached_gather_in_flight = true;
}

void execute_cached_fixed_transform_hits(
    const std::vector<JpegDctDeviceCachedFixedTransformBatchItem>& items,
    const uint16_t*                                                quant_tables,
    const float*                                                   resize_weight_matrices,
    const JpegDctGridTransformSpec&                                transform,
    float*                                                         y_accum,
    float*                                                         cbcr_accum,
    JpegDctDeviceExecutionStats&                                   execution_stats,
    JpegDctDeviceScratch&                                          scratch) {
	if (items.empty()) {
		return;
	}
	if (scratch.cached_gather_in_flight && !scratch.cached_fixed_transform_in_flight) {
		drain_cached_gather(scratch, execution_stats);
	}
	if (scratch.cached_gather_in_flight &&
	    scratch.cached_fixed_transform_items.needs_reallocation(items.size())) {
		drain_cached_gather(scratch, execution_stats);
	}
	auto& upload_items = scratch.host_cached_fixed_transform_uploads.emplace_back(items.begin(), items.end());
	const auto stream = scratch.stream_for_cache_hit();
	scratch.ensure_cached_gather_events();
	if (!scratch.cached_gather_in_flight) {
		scratch.cached_gather_start.record(stream);
		scratch.cached_fixed_transform_in_flight = true;
	}
	scratch.cached_fixed_transform_items.upload(
	    upload_items.data(), upload_items.size(), stream, execution_stats);
	constexpr unsigned kThreads = 64;
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>(upload_items.size()));
	transformed_dct_grid_cached_kernel<<<grid, block, 0, stream>>>(
	    scratch.cached_fixed_transform_items.data,
	    upload_items.size(),
	    quant_tables,
	    resize_weight_matrices,
	    transform.clamp_min,
	    transform.clamp_max,
	    y_accum,
	    cbcr_accum);
	CUDA_SAFE_CALL(cudaGetLastError());
	++execution_stats.materialize_kernel_launch_count;
	execution_stats.fixed_transform_item_count += upload_items.size();
	execution_stats.cached_gather_item_count += upload_items.size();
	scratch.cached_gather_done.record(stream);
	scratch.cached_gather_in_flight = true;
}

void execute_unified_image_major_plan(
    const std::vector<JpegDctDeviceShardPlan>&      shards,
	const std::vector<uint32_t>*                     fixed_transform_item_order,
	const std::vector<uint32_t>*                     fixed_transform_group_offsets,
    int16_t*                                        output,
    int16_t*                                        y_output,
    int16_t*                                        cbcr_output,
    float*                                          y_accum,
    float*                                          cbcr_accum,
    const uint16_t*                                 fixed_quant_tables,
    const float*                                    fixed_resize_weight_matrices,
    const JpegDctGridTransformSpec&                 grid_transform,
    const bool                                      output_ycbcr_dct_grid,
    const bool                                      output_transformed_dct_grid,
    const std::vector<uint8_t>&                     selected_coefficients,
    const JpegDctCoefficientSelectionShape&         selection_shape,
	JpegDctDeviceDecodedRowgroupCache*               cache,
    JpegDctDeviceCacheStats&                        batch_cache_stats,
    JpegDctDeviceExecutionStats&                    execution_stats,
	JpegDctDeviceScratch&                           scratch,
	const size_t                                    decode_batch_rowgroups,
	std::vector<JpegDctDeviceCachedGatherBatchItem>& cached_pending,
	std::vector<JpegDctDeviceCachedFixedTransformBatchItem>& cached_fixed_pending) {
	auto& pending = scratch.host_pending_works;
	pending.clear();
	size_t rowgroup_count = 0;
	for (const auto& shard : shards) {
		rowgroup_count += shard.rowgroups.size();
	}
	pending.reserve(rowgroup_count);
	const size_t effective_decode_batch_rowgroups =
	    decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : decode_batch_rowgroups;

	struct MissPlanRef {
		const JpegDctDeviceRowgroupPlan* rowgroup = nullptr;
		uint32_t                         shard_id = std::numeric_limits<uint32_t>::max();
		const std::filesystem::path*     fls_path = nullptr;
	};
	std::vector<MissPlanRef> misses;
	misses.reserve(rowgroup_count);

	// Classify the entire cross-shard batch before any miss can insert into and
	// evict from the cache. Hit buffers are gathered first; all remaining misses
	// are then safe to aggregate into worksets independent of physical shard.
	for (const auto& shard : shards) {
		for (const auto& rowgroup_plan : shard.rowgroups) {
			const auto source_shard_id = rowgroup_plan.source_shard_id == std::numeric_limits<uint32_t>::max()
			                                 ? shard.shard_id
			                                 : rowgroup_plan.source_shard_id;
			const auto* source_fls_path =
			    rowgroup_plan.source_fls_path == nullptr ? shard.fls_path : rowgroup_plan.source_fls_path;
			const auto key = JpegDctDeviceDecodedRowgroupCacheKey {source_shard_id, rowgroup_plan.rowgroup_index};
			if (cache != nullptr && cache->capacity > 0) {
				auto it = cache->entries.find(key);
				if (it != cache->entries.end() && it->second->blocks.has_value()) {
					it->second->last_access = ++cache->clock;
					++batch_cache_stats.hits;
					const auto* dense = it->second->blocks->get();
					if (output_transformed_dct_grid) {
						cached_fixed_pending.reserve(
						    cached_fixed_pending.size() + rowgroup_plan.fixed_transform_items.size());
						for (const auto& item : rowgroup_plan.fixed_transform_items) {
							cached_fixed_pending.push_back(JpegDctDeviceCachedFixedTransformBatchItem {
							    dense,
							    JpegDctDeviceFixedTransformBatchItem {0U,
							                                          item.row_in_rowgroup,
							                                          item.output_block_index,
							                                          item.component,
							                                          static_cast<uint8_t>(item.zigzag_columns ? 1U : 0U),
							                                          item.x_factor,
							                                          item.y_factor,
							                                          item.x_subblock,
							                                          item.y_subblock,
							                                          static_cast<uint8_t>(item.x_upsample ? 1U : 0U),
							                                          static_cast<uint8_t>(item.y_upsample ? 1U : 0U),
							                                          item.x_up_factor,
							                                          item.y_up_factor,
							                                          item.x_down_factor,
							                                          item.y_down_factor,
							                                          item.quant_table_index,
							                                          item.x_weight_matrix_index,
							                                          item.y_weight_matrix_index}});
						}
					} else {
						cached_pending.reserve(cached_pending.size() + rowgroup_plan.items.size());
						for (const auto& item : rowgroup_plan.items) {
							cached_pending.push_back(
							    JpegDctDeviceCachedGatherBatchItem {dense, item.row_in_rowgroup, item.output_block_index});
						}
					}
					continue;
				}
				++batch_cache_stats.misses;
			}
			misses.push_back(MissPlanRef {&rowgroup_plan, source_shard_id, source_fls_path});
		}
	}

	execute_cached_rowgroup_hits(cached_pending, output, execution_stats, scratch);
	execute_cached_fixed_transform_hits(cached_fixed_pending,
	                                    fixed_quant_tables,
	                                    fixed_resize_weight_matrices,
	                                    grid_transform,
	                                    y_accum,
	                                    cbcr_accum,
	                                    execution_stats,
	                                    scratch);
	cached_pending.clear();
	cached_fixed_pending.clear();
	unsigned batch_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	for (const auto miss : misses) {
		if (miss.rowgroup == nullptr) {
			throw std::runtime_error("unified JPEG DCT cache miss plan is invalid");
		}
		batch_unpack_n_vectors =
		    constrain_jpeg_dct_batch_unpack_n_vectors(batch_unpack_n_vectors, *miss.rowgroup);
	}

	size_t fixed_transform_source_item_offset = 0;
	const auto flush_pending = [&]() {
		if (pending.empty()) {
			return;
		}
		const auto workset_item_count = fixed_transform_item_count(pending);
		const auto workset_fixed_transform_plan =
		    fixed_transform_plan_for_workset(fixed_transform_item_order,
		                                     fixed_transform_group_offsets,
		                                     fixed_transform_source_item_offset,
		                                     workset_item_count,
		                                     scratch);
		execute_decoded_rowgroup_batch(pending,
		                               workset_fixed_transform_plan.item_order,
		                               workset_fixed_transform_plan.group_offsets,
		                               output,
		                               y_output,
		                               cbcr_output,
		                               y_accum,
		                               cbcr_accum,
		                               fixed_quant_tables,
		                               fixed_resize_weight_matrices,
		                               grid_transform,
		                               selected_coefficients,
		                               selection_shape,
		                               output_ycbcr_dct_grid,
		                               output_transformed_dct_grid,
		                               cache,
		                               batch_cache_stats,
		                               execution_stats,
		                               scratch);
		fixed_transform_source_item_offset += workset_item_count;
	};

	uint32_t current_shard_id = std::numeric_limits<uint32_t>::max();
	std::shared_ptr<galp::format::FlsReader> rdr;
	for (const auto miss : misses) {
		if (miss.rowgroup == nullptr || miss.fls_path == nullptr ||
		    miss.shard_id == std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("unified JPEG DCT cache miss plan is invalid");
		}
		if (!rdr || current_shard_id != miss.shard_id) {
			current_shard_id = miss.shard_id;
			rdr              = scratch.fls_reader(*miss.fls_path);
		}
		auto work = prepare_decoded_rowgroup_work(*rdr,
			                                          miss.shard_id,
			                                          *miss.rowgroup,
			                                          selected_coefficients,
			                                          selection_shape,
			                                          output_ycbcr_dct_grid,
			                                          batch_unpack_n_vectors,
			                                          cache,
			                                          execution_stats);
		pending.push_back(std::move(work));
		if (pending.size() >= effective_decode_batch_rowgroups) {
			flush_pending();
		}
	}
	flush_pending();
	if (fixed_transform_item_order != nullptr &&
	    fixed_transform_source_item_offset != fixed_transform_item_order->size()) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan was not fully consumed");
	}
}

void execute_shard_plan(const std::shared_ptr<galp::format::FlsReader>&  rdr,
                        const JpegDctDeviceShardPlan&                    shard,
	                    const std::vector<uint32_t>*                     fixed_transform_item_order,
	                    const std::vector<uint32_t>*                     fixed_transform_group_offsets,
                        int16_t*                                         output,
                        int16_t*                                         y_output,
                        int16_t*                                         cbcr_output,
                        float*                                           y_accum,
                        float*                                           cbcr_accum,
                        const uint16_t*                                  fixed_quant_tables,
                        const float*                                     fixed_resize_weight_matrices,
                        const JpegDctGridTransformSpec&                   grid_transform,
                        const bool                                       output_ycbcr_dct_grid,
                        const bool                                       output_transformed_dct_grid,
                        const std::vector<uint8_t>&                      selected_coefficients,
                        const JpegDctCoefficientSelectionShape&          selection_shape,
                        JpegDctDeviceDecodedRowgroupCache*               cache,
                        JpegDctDeviceCacheStats&                         batch_cache_stats,
                        JpegDctDeviceExecutionStats&                     execution_stats,
                        JpegDctDeviceScratch&                            scratch,
                        const size_t                                     decode_batch_rowgroups,
	                        const JpegDctDeviceRowgroupPrefetchConfig&       rowgroup_prefetch,
	                        std::vector<JpegDctDeviceCachedGatherBatchItem>& cached_pending,
	                        std::vector<JpegDctDeviceCachedFixedTransformBatchItem>& cached_fixed_pending) {
	if (!rdr) {
		throw std::runtime_error("execute_shard_plan: reader is null");
	}
	auto& pending = scratch.host_pending_works;
	pending.clear();
	const size_t effective_decode_batch_rowgroups =
	    decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : decode_batch_rowgroups;
	pending.reserve(std::min(effective_decode_batch_rowgroups, shard.rowgroups.size()));
	unsigned batch_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	for (const auto& rowgroup_plan : shard.rowgroups) {
		batch_unpack_n_vectors =
		    constrain_jpeg_dct_batch_unpack_n_vectors(batch_unpack_n_vectors, rowgroup_plan);
	}
	bool pending_may_insert_cache = false;
	auto prefetch_plan =
	    plan_jpeg_dct_rowgroup_prefetch(shard, cache, rowgroup_prefetch, effective_decode_batch_rowgroups);
	std::unique_ptr<galp::runtime::RowgroupPrefetchQueue> prefetch_queue;
	if (prefetch_plan.enabled) {
		const size_t scheduled_rowgroups  = prefetch_plan.rowgroup_indices.size();
		const auto   prefetch_queue_start = Clock::now();
		// Popped rowgroups remain live until the current decode workset completes, while
		// the producer may keep `depth` more rowgroups queued. Size the shared pool for
		// both sets so workers cannot deadlock waiting for leases held by `pending`.
		const size_t pinned_pool_slots = effective_decode_batch_rowgroups + rowgroup_prefetch.depth +
		                                 std::max<size_t>(1U, rowgroup_prefetch.workers);
		if (!scratch.rowgroup_prefetch_pinned_pool ||
		    scratch.rowgroup_prefetch_pinned_pool_slots < pinned_pool_slots) {
			scratch.rowgroup_prefetch_pinned_pool =
			    galp::runtime::PinnedRowgroupBufferPool::create(pinned_pool_slots);
			scratch.rowgroup_prefetch_pinned_pool_slots = pinned_pool_slots;
		}
		prefetch_queue = std::make_unique<galp::runtime::RowgroupPrefetchQueue>(
		    rdr,
		    std::move(prefetch_plan.rowgroup_indices),
		    rowgroup_prefetch.depth,
		    rowgroup_prefetch.workers,
		    scratch.rowgroup_prefetch_pinned_pool);
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
		execute_cached_fixed_transform_hits(cached_fixed_pending,
		                                    fixed_quant_tables,
		                                    fixed_resize_weight_matrices,
		                                    grid_transform,
		                                    y_accum,
		                                    cbcr_accum,
		                                    execution_stats,
		                                    scratch);
		cached_pending.clear();
		cached_fixed_pending.clear();
	};
	size_t fixed_transform_source_item_offset = 0;
	const auto flush_pending                      = [&]() {
		// Pending dense materialization can evict cached dense buffers referenced by cached_pending.
		// Launch cached gathers first; the decoded stream will wait on their completion event.
		if (pending_may_insert_cache) {
			flush_cached();
		}
		const auto workset_item_count = fixed_transform_item_count(pending);
		const auto workset_fixed_transform_plan =
		    fixed_transform_plan_for_workset(fixed_transform_item_order,
		                                     fixed_transform_group_offsets,
		                                     fixed_transform_source_item_offset,
		                                     workset_item_count,
		                                     scratch);
        execute_decoded_rowgroup_batch(pending,
                                       workset_fixed_transform_plan.item_order,
                                       workset_fixed_transform_plan.group_offsets,
                                       output,
                                       y_output,
                                       cbcr_output,
                                       y_accum,
                                       cbcr_accum,
                                       fixed_quant_tables,
                                       fixed_resize_weight_matrices,
                                       grid_transform,
                                       selected_coefficients,
                                       selection_shape,
                                       output_ycbcr_dct_grid,
                                       output_transformed_dct_grid,
                                       cache,
                                       batch_cache_stats,
                                       execution_stats,
                                       scratch);
        fixed_transform_source_item_offset += workset_item_count;
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
		execution_stats.rowgroup_storage_bytes_read += result.storage_bytes;
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
					if (output_transformed_dct_grid) {
						cached_fixed_pending.reserve(cached_fixed_pending.size() +
						                             rowgroup_plan.fixed_transform_items.size());
						for (const auto& item : rowgroup_plan.fixed_transform_items) {
							cached_fixed_pending.push_back(JpegDctDeviceCachedFixedTransformBatchItem {
							    dense,
							    JpegDctDeviceFixedTransformBatchItem {
							        0U,
							        item.row_in_rowgroup,
							        item.output_block_index,
							        item.component,
							        static_cast<uint8_t>(item.zigzag_columns ? 1U : 0U),
							        item.x_factor,
							        item.y_factor,
							        item.x_subblock,
							        item.y_subblock,
							        static_cast<uint8_t>(item.x_upsample ? 1U : 0U),
							        static_cast<uint8_t>(item.y_upsample ? 1U : 0U),
							        item.x_up_factor,
							        item.y_up_factor,
							        item.x_down_factor,
							        item.y_down_factor,
							        item.quant_table_index,
							        item.x_weight_matrix_index,
							        item.y_weight_matrix_index}});
						}
					} else {
						cached_pending.reserve(cached_pending.size() + rowgroup_plan.items.size());
						for (const auto& item : rowgroup_plan.items) {
							cached_pending.push_back(JpegDctDeviceCachedGatherBatchItem {
							    dense, item.row_in_rowgroup, item.output_block_index});
						}
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

		auto work                = prefetch_plan.use_prefetch_for_position[rowgroup_pos]
		                               ? prepare_decoded_rowgroup_work_from_materialized(pop_prefetched_rowgroup(rowgroup_plan),
                                                                          shard.shard_id,
                                                                          rowgroup_plan,
                                                                          selected_coefficients,
                                                                          selection_shape,
                                                                          output_ycbcr_dct_grid,
                                                                          batch_unpack_n_vectors,
                                                                          cache,
                                                                          execution_stats)
		                               : prepare_decoded_rowgroup_work(*rdr,
                                                        shard.shard_id,
                                                        rowgroup_plan,
                                                        selected_coefficients,
                                                        selection_shape,
                                                        output_ycbcr_dct_grid,
                                                        batch_unpack_n_vectors,
                                                        cache,
                                                        execution_stats);
		pending_may_insert_cache = pending_may_insert_cache || static_cast<bool>(work.cache_entry);
		pending.push_back(std::move(work));
		if (pending.size() >= effective_decode_batch_rowgroups) {
			flush_pending();
		}
	}
	flush_pending();
	if (fixed_transform_item_order != nullptr &&
	    fixed_transform_source_item_offset != fixed_transform_item_order->size()) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan was not fully consumed");
	}
	if (prefetch_queue) {
		execution_stats.prefetch_wait_ms += prefetch_queue->wait_ms();
	}
}

} // namespace

void append_jpeg_rowgroup_columns(galp::runtime::ExecutionWorkset&        workset,
                                  const galp::execution::Rowgroup&        rowgroup,
                                  const galp::execution::ExecutionConfig& cfg,
                                  const size_t                            expr_index_base,
                                  const std::vector<uint8_t>&             selected_coefficients,
                                  const std::vector<uint32_t>*            selected_vectors) {
	append_jpeg_rowgroup_columns(workset,
	                             rowgroup,
	                             cfg,
	                             expr_index_base,
	                             selected_coefficients,
	                             classify_coefficient_selection(selected_coefficients),
	                             selected_vectors);
}

void append_jpeg_rowgroup_columns(galp::runtime::ExecutionWorkset&        workset,
                                  const galp::execution::Rowgroup&        rowgroup,
                                  const galp::execution::ExecutionConfig& cfg,
                                  const size_t                            expr_index_base,
                                  const std::vector<uint8_t>&             selected_coefficients,
                                  const JpegDctCoefficientSelectionShape& selection_shape,
                                  const std::vector<uint32_t>*            selected_vectors) {
	workset.outputs.required = workset.outputs.required || cfg.write_out;
	galp::runtime::begin_workset_chunk_arena(workset, rowgroup.columns.size());
	auto* active_chunk_arena = workset.buffers.chunk_arena.get();

	std::array<bool, kJpegDctCoefficientCount> decode_coefficients {};
	const auto mark_decode_coefficient = [&](const auto& self, const size_t coeff_idx) -> void {
		if (coeff_idx >= kJpegDctCoefficientCount || coeff_idx >= rowgroup.columns.size()) {
			throw std::out_of_range("JPEG DCT selected coefficient is outside the rowgroup column range");
		}
		if (decode_coefficients[coeff_idx]) {
			return;
		}
		decode_coefficients[coeff_idx] = true;
		const auto& column             = rowgroup.columns[coeff_idx];
		if (column.alias_of.has_value()) {
			self(self, *column.alias_of);
		}
	};
	if (selection_shape.kind == JpegDctCoefficientSelectionKind::kAll) {
		for (size_t coeff_idx = 0; coeff_idx < kJpegDctCoefficientCount && coeff_idx < rowgroup.columns.size();
		     ++coeff_idx) {
			decode_coefficients[coeff_idx] = true;
		}
	} else {
		for_each_selected_coefficient(selected_coefficients, selection_shape, [&](const size_t coeff_idx) {
			mark_decode_coefficient(mark_decode_coefficient, coeff_idx);
		});
	}

	const void* last_backing_base  = nullptr;
	size_t      last_backing_bytes = 0;
	for (size_t coeff_idx = 0; coeff_idx < kJpegDctCoefficientCount && coeff_idx < rowgroup.columns.size();
	     ++coeff_idx) {
		if (!decode_coefficients[coeff_idx]) {
			continue;
		}
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
	auto impl = std::make_unique<JpegDctDeviceBatch::Impl>();
	CUDA_SAFE_CALL(cudaGetDevice(&impl->cuda_device));
	impl->layout                                             = plan.layout;
	impl->image_layouts                                      = std::move(plan.image_layouts);
	impl->block_metadata                                     = std::move(plan.block_metadata);
	impl->rowgroups                                          = std::move(plan.rowgroups);
	impl->selected_coefficients                              = std::move(plan.selected_coefficients);
	impl->coefficients_per_block                             = plan.coefficients_per_block;
	impl->execution_stats.planning_ms                        = plan.planning_ms;
	impl->execution_stats.plan_cache_hits                    = plan.plan_cache_hits;
	impl->execution_stats.plan_cache_misses                  = plan.plan_cache_misses;
	impl->execution_stats.plan_cache_evictions               = plan.plan_cache_evictions;
	impl->execution_stats.exact_batch_plan_cache_enabled     = plan.exact_batch_plan_cache_enabled;
	impl->execution_stats.resize_weight_build_ms             = plan.resize_weight_build_ms;
	impl->execution_stats.dct_resize_weight_cache_hits       = plan.dct_resize_weight_cache_hits;
	impl->execution_stats.dct_resize_weight_cache_misses     = plan.dct_resize_weight_cache_misses;
	impl->execution_stats.dct_conversion_matrix_cache_hits   = plan.dct_conversion_matrix_cache_hits;
	impl->execution_stats.dct_conversion_matrix_cache_misses = plan.dct_conversion_matrix_cache_misses;
	impl->execution_stats.rowgroup_count                     = impl->rowgroups.size();
	impl->execution_stats.planned_selected_vector_count      = plan.planned_selected_vector_count;
	impl->execution_stats.full_vector_count                  = plan.full_vector_count;
	impl->execution_stats.planned_saved_vector_count         = plan.planned_saved_vector_count;
	impl->execution_stats.cache_enabled                      = plan.cache_enabled;
	impl->execution_stats.fixed_transform_image_count =
	    plan.layout == JpegDctDeviceLayout::kTransformedDctGrid ? impl->image_layouts.size() : 0;
	impl->execution_stats.fixed_transform_component_count        = plan.fixed_transform_component_count;
	impl->execution_stats.fixed_transform_source_block_count     = plan.fixed_transform_source_block_count;
	impl->execution_stats.fixed_transform_output_block_count     = plan.fixed_transform_output_block_count;
	impl->execution_stats.host_expanded_transform_items_created  = plan.host_expanded_transform_items_created;
	impl->execution_stats.host_output_block_source_lists_created = plan.host_output_block_source_lists_created;
	impl->execution_stats.host_global_transform_sort_items       = plan.host_global_transform_sort_items;
	impl->execution_stats.planless_axis_program_count            = plan.planless_axis_program_count;
	impl->execution_stats.planless_axis_phase_matrix_count       = plan.planless_axis_phase_matrix_count;
	impl->execution_stats.planless_axis_program_bytes = plan.fixed_resize_weight_matrices.size() * sizeof(float);
	impl->ycbcr_dct_grid_shape                        = plan.ycbcr_dct_grid_shape;

	const bool output_ycbcr_dct_grid =
	    plan.layout == JpegDctDeviceLayout::kYcbcrDctGrid || plan.layout == JpegDctDeviceLayout::kTransformedDctGrid;
	const bool output_weighted_grid = plan.layout == JpegDctDeviceLayout::kTransformedDctGrid;
	impl->grid_output_data_type =
	    output_weighted_grid ? plan.grid_transform.output_data_type : JpegDctGridOutputDataType::kInt16;
	impl->execution_stats.fixed_grid_output_float32 =
	    impl->grid_output_data_type == JpegDctGridOutputDataType::kFloat32;
	impl->execution_stats.fixed_grid_output_affine_applied = impl->execution_stats.fixed_grid_output_float32;
	impl->execution_stats.fixed_grid_output_add   = plan.grid_transform.output_add;
	impl->execution_stats.fixed_grid_output_scale = plan.grid_transform.output_scale;
	if (output_ycbcr_dct_grid) {
		impl->coefficient_count      = 0;
		impl->y_coefficient_count    = impl->ycbcr_dct_grid_shape.y_count();
		impl->cbcr_coefficient_count = impl->ycbcr_dct_grid_shape.cbcr_count();
		if (impl->y_coefficient_count != 0) {
			if (impl->grid_output_data_type == JpegDctGridOutputDataType::kInt16) {
				impl->y_coefficients.emplace(impl->y_coefficient_count);
				CUDA_SAFE_CALL(cudaMemset(impl->y_coefficients->get(), 0, impl->y_coefficient_count * sizeof(int16_t)));
			}
			if (output_weighted_grid) {
				impl->y_accum.emplace(impl->y_coefficient_count);
				CUDA_SAFE_CALL(cudaMemset(impl->y_accum->get(), 0, impl->y_coefficient_count * sizeof(float)));
			}
		}
		if (impl->cbcr_coefficient_count != 0) {
			if (impl->grid_output_data_type == JpegDctGridOutputDataType::kInt16) {
				impl->cbcr_coefficients.emplace(impl->cbcr_coefficient_count);
				CUDA_SAFE_CALL(
				    cudaMemset(impl->cbcr_coefficients->get(), 0, impl->cbcr_coefficient_count * sizeof(int16_t)));
			}
			if (output_weighted_grid) {
				impl->cbcr_accum.emplace(impl->cbcr_coefficient_count);
				CUDA_SAFE_CALL(cudaMemset(impl->cbcr_accum->get(), 0, impl->cbcr_coefficient_count * sizeof(float)));
			}
		}
	} else {
		impl->coefficient_count = impl->block_metadata.size() * impl->coefficients_per_block;
	}
	if (impl->coefficient_count != 0) {
		impl->coefficients.emplace(impl->coefficient_count);
	}

	int16_t*   output      = impl->coefficients.has_value() ? impl->coefficients->get() : nullptr;
	int16_t*   y_output    = impl->y_coefficients.has_value() ? impl->y_coefficients->get() : nullptr;
	int16_t*   cbcr_output = impl->cbcr_coefficients.has_value() ? impl->cbcr_coefficients->get() : nullptr;
	float*     y_accum     = impl->y_accum.has_value() ? impl->y_accum->get() : nullptr;
	float*     cbcr_accum  = impl->cbcr_accum.has_value() ? impl->cbcr_accum->get() : nullptr;
	const bool has_weighted_grid_output =
	    output_weighted_grid && (impl->y_coefficient_count != 0 || impl->cbcr_coefficient_count != 0);
	if (has_weighted_grid_output) {
		if (plan.fixed_quant_tables.empty() || plan.fixed_quant_tables.size() % 64U != 0U) {
			throw std::runtime_error("JPEG DCT fixed transform plan has invalid quantization tables");
		}
		if (!plan.uses_planless_fixed_transform &&
		    (plan.fixed_resize_weight_matrices.empty() || plan.fixed_resize_weight_matrices.size() % 64U != 0U)) {
			throw std::runtime_error("JPEG DCT fixed transform plan has invalid resize weight matrices");
		}
		impl->fixed_quant_tables.emplace(plan.fixed_quant_tables.size(), plan.fixed_quant_tables.data());
		if (!plan.fixed_resize_weight_matrices.empty()) {
			impl->fixed_resize_weight_matrices.emplace(plan.fixed_resize_weight_matrices.size(),
			                                           plan.fixed_resize_weight_matrices.data());
		}
	}
	const uint16_t* fixed_quant_tables =
	    impl->fixed_quant_tables.has_value() ? impl->fixed_quant_tables->get() : nullptr;
	const float* fixed_resize_weight_matrices =
	    impl->fixed_resize_weight_matrices.has_value() ? impl->fixed_resize_weight_matrices->get() : nullptr;
	JpegDctDeviceScratch local_scratch;
	auto&                scratch              = plan.scratch != nullptr ? *plan.scratch : local_scratch;
	scratch.configure_scheduling(
	    plan.use_low_priority_streams, plan.transform_blocks_per_launch, plan.transform_ctas_per_launch);
	impl->execution_stats.direct_dct_stream_priority      = scratch.direct_dct_stream_priority;
	impl->execution_stats.cuda_least_stream_priority      = scratch.cuda_least_stream_priority;
	impl->execution_stats.cuda_greatest_stream_priority   = scratch.cuda_greatest_stream_priority;
	impl->execution_stats.direct_dct_low_priority_streams = scratch.direct_dct_low_priority_streams;
	switch (plan.scheduling_policy) {
	case JpegDctSchedulingPolicy::kFullyOverlapped:
		impl->execution_stats.scheduling_policy = "fully-overlapped";
		break;
	case JpegDctSchedulingPolicy::kLimitedOverlap:
		impl->execution_stats.scheduling_policy = "limited-overlap";
		break;
	case JpegDctSchedulingPolicy::kSerial:
		impl->execution_stats.scheduling_policy = "serial";
		break;
	}
	auto& cached_pending       = scratch.host_cached_gather_items;
	auto& cached_fixed_pending = scratch.host_cached_fixed_transform_items;
	cached_pending.clear();
	cached_fixed_pending.clear();
	auto*      decode_cache = output_ycbcr_dct_grid && !output_weighted_grid ? nullptr : plan.cache;
	const bool use_deterministic_fixed_transform =
	    plan.unify_rowgroups_across_shards && decode_cache == nullptr && !plan.fixed_transform_item_order->empty();
	const auto* fixed_transform_item_order =
	    use_deterministic_fixed_transform ? plan.fixed_transform_item_order.get() : nullptr;
	const auto* fixed_transform_group_offsets =
	    use_deterministic_fixed_transform ? plan.fixed_transform_group_offsets.get() : nullptr;
	const bool mixed_physical_shards =
	    plan.shards->size() > 1U || (!plan.shards->empty() && plan.shards->front().mixed_physical_shards);
	if (plan.unify_rowgroups_across_shards && mixed_physical_shards) {
		execute_unified_image_major_plan(*plan.shards,
		                                 fixed_transform_item_order,
		                                 fixed_transform_group_offsets,
		                                 output,
		                                 y_output,
		                                 cbcr_output,
		                                 y_accum,
		                                 cbcr_accum,
		                                 fixed_quant_tables,
		                                 fixed_resize_weight_matrices,
		                                 plan.grid_transform,
		                                 output_ycbcr_dct_grid,
		                                 output_weighted_grid,
		                                 impl->selected_coefficients,
		                                 plan.coefficient_selection_shape,
		                                 decode_cache,
		                                 impl->cache_stats,
		                                 impl->execution_stats,
		                                 scratch,
		                                 plan.decode_batch_rowgroups,
		                                 cached_pending,
		                                 cached_fixed_pending);
	} else {
		for (const auto& shard : *plan.shards) {
			if (shard.fls_path == nullptr) {
				throw std::runtime_error("JPEG DCT shard plan has no FLS path");
			}
			auto rdr = scratch.fls_reader(*shard.fls_path);
			execute_shard_plan(rdr,
			                   shard,
			                   fixed_transform_item_order,
			                   fixed_transform_group_offsets,
			                   output,
			                   y_output,
			                   cbcr_output,
			                   y_accum,
			                   cbcr_accum,
			                   fixed_quant_tables,
			                   fixed_resize_weight_matrices,
			                   plan.grid_transform,
			                   output_ycbcr_dct_grid,
			                   output_weighted_grid,
			                   impl->selected_coefficients,
			                   plan.coefficient_selection_shape,
			                   decode_cache,
			                   impl->cache_stats,
			                   impl->execution_stats,
			                   scratch,
			                   use_deterministic_fixed_transform
			                       ? std::max(plan.decode_batch_rowgroups, shard.rowgroups.size())
			                       : plan.decode_batch_rowgroups,
			                   plan.rowgroup_prefetch,
			                   cached_pending,
			                   cached_fixed_pending);
		}
	}
	if (!output_ycbcr_dct_grid) {
		execute_cached_rowgroup_hits(cached_pending, output, impl->execution_stats, scratch);
	} else if (output_weighted_grid) {
		execute_cached_fixed_transform_hits(cached_fixed_pending,
		                                    fixed_quant_tables,
		                                    fixed_resize_weight_matrices,
		                                    plan.grid_transform,
		                                    y_accum,
		                                    cbcr_accum,
		                                    impl->execution_stats,
		                                    scratch);
	}
	drain_cached_gather(scratch, impl->execution_stats);
	cached_pending.clear();
	cached_fixed_pending.clear();
	if (output_weighted_grid) {
		round_fixed_ycbcr_grid_outputs(y_accum,
		                               cbcr_accum,
		                               y_output,
		                               cbcr_output,
		                               impl->y_coefficient_count,
		                               impl->cbcr_coefficient_count,
		                               scratch,
		                               impl->execution_stats,
		                               impl->fixed_grid_round_start_event,
		                               impl->completion_event,
		                               plan.grid_transform);
	}
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
	const auto native_device                                       = galp::memory::device_pool_stats();
	impl->execution_stats.direct_dct_h2d_stream_priority = scratch.actual_stream_priority(
	    scratch.decode_workset.transfer.h2d_stream ? scratch.decode_workset.transfer.h2d_stream.get() : nullptr);
	impl->execution_stats.direct_dct_decode_stream_priority = scratch.actual_stream_priority(
	    scratch.decode_workset.transfer.compute_stream ? scratch.decode_workset.transfer.compute_stream.get()
	                                                   : nullptr);
	impl->execution_stats.direct_dct_transform_stream_priority =
	    scratch.actual_stream_priority(scratch.transform_stream ? scratch.transform_stream.get() : nullptr);
	impl->execution_stats.direct_dct_round_stream_priority = scratch.actual_stream_priority(
	    scratch.fixed_grid_round_stream ? scratch.fixed_grid_round_stream.get() : nullptr);
	impl->execution_stats.direct_dct_stream_priority      = impl->execution_stats.direct_dct_transform_stream_priority;
	impl->execution_stats.galp_native_device_in_use_bytes = native_device.in_use_bytes;
	impl->execution_stats.galp_native_device_peak_in_use_bytes     = native_device.peak_in_use_bytes;
	impl->execution_stats.galp_native_device_cached_bytes          = native_device.cached_bytes;
	impl->execution_stats.galp_native_device_allocation_requests   = native_device.allocation_requests;
	impl->execution_stats.galp_native_device_cuda_allocation_count = native_device.cuda_allocation_count;
	impl->execution_stats.galp_native_device_cuda_allocation_bytes = native_device.cuda_allocation_bytes;
	return JpegDctDeviceBatch(std::move(impl));
}

} // namespace galp::jpeg::detail
