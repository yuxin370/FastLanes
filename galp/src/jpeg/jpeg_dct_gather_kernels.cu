#include "jpeg/jpeg_dct_kernel_launch.cuh"

namespace galp::jpeg::detail {

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
	out[selected_dct_output_offset(item.output_block_index, item.selected_coefficient_slot, coefficients_per_block)] =
	    value;
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

} // namespace galp::jpeg::detail
