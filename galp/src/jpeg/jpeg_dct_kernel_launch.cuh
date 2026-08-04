#pragma once

#include "jpeg/jpeg_dct_kernel_types.cuh"
#include <cstddef>
#include <cstdint>

namespace galp::jpeg::detail {

__global__ void project_dct_coefficients_batch_kernel(const DeviceCoeffBinding*               column_bindings,
                                                      const JpegDctDeviceProjectionBatchItem* items,
                                                      size_t                                  item_count,
                                                      size_t                                  coefficients_per_block,
                                                      int16_t*                                out);
__global__ void project_dct_ycbcr_grid_batch_kernel(const DeviceCoeffBinding*               column_bindings,
                                                    const JpegDctDeviceProjectionBatchItem* items,
                                                    size_t                                  item_count,
                                                    int16_t*                                y_out,
                                                    int16_t*                                cbcr_out,
                                                    float*                                  y_accum,
                                                    float*                                  cbcr_accum);
__global__ void gather_decoded_dct_blocks_batch_kernel(const DeviceCoeffBinding*                  column_bindings,
                                                       const JpegDctDeviceDecodedGatherBatchItem* items,
                                                       size_t                                     item_count,
                                                       int16_t*                                   out);
__global__ void materialize_dense_dct_rowgroup_batch_kernel(const DeviceCoeffBinding*                column_bindings,
                                                            const JpegDctDeviceMaterializeBatchItem* items,
                                                            size_t                                   item_count);
__global__ void
gather_cached_dct_blocks_batch_kernel(const JpegDctDeviceCachedGatherBatchItem* items, size_t item_count, int16_t* out);

__global__ void round_dct_grid_accum_pair_kernel(
    const float* y_in, size_t y_count, const float* cbcr_in, size_t cbcr_count, int16_t* y_out, int16_t* cbcr_out);
__global__ void round_affine_dct_grid_accum_pair_kernel(
    float* y_in_out, size_t y_count, float* cbcr_in_out, size_t cbcr_count, float output_add, float output_scale);
__global__ void transformed_dct_grid_planless_kernel(const DeviceCoeffBinding*                   column_bindings,
                                                     const JpegDctDevicePlanlessImageDescriptor* images,
	                                                     const uint32_t*                            logical_to_compact_vectors,
	                                                     const uint32_t*                            image_vector_bindings,
	                                                     const uint32_t*                            active_output_blocks,
	                                                     const JpegDctDeviceBlockMajorGroupBinding* block_major_groups,
	                                                     size_t                                      block_major_group_count,
	                                                     const JpegDctDeviceBlockMajorRankCell*     block_major_rank_cells,
	                                                     size_t                                      block_major_rank_cell_count,
	                                                     const uint8_t*                             block_major_rank_payload,
	                                                     size_t                                      block_major_rank_payload_size,
                                                     size_t                                      image_count,
                                                     uint64_t                                    output_block_offset,
                                                     uint64_t                                    output_block_count,
                                                     const uint16_t*                             quant_tables,
                                                     const float*                                phase_matrices,
                                                     uint32_t                                    y_output_width,
                                                     uint32_t                                    y_output_height,
                                                     uint32_t                                    cbcr_output_width,
                                                     uint32_t                                    cbcr_output_height,
                                                     int32_t                                     clamp_min,
                                                     int32_t                                     clamp_max,
                                                     float*                                      y_accum,
                                                     float*                                      cbcr_accum);
__global__ void transformed_dct_grid_sources_kernel(const DeviceCoeffBinding*                   column_bindings,
                                                    const JpegDctDeviceFixedTransformBatchItem* items,
                                                    size_t                                      item_count,
                                                    const uint16_t*                             quant_tables,
                                                    const float*                                resize_weight_matrices,
                                                    int32_t                                     clamp_min,
                                                    int32_t                                     clamp_max,
                                                    float*                                      y_accum,
                                                    float*                                      cbcr_accum);
__global__ void transformed_dct_grid_grouped_kernel(const DeviceCoeffBinding*                   column_bindings,
                                                    const JpegDctDeviceFixedTransformBatchItem* items,
                                                    const uint32_t*                             group_offsets,
                                                    size_t                                      group_count,
                                                    const uint16_t*                             quant_tables,
                                                    const float*                                resize_weight_matrices,
                                                    int32_t                                     clamp_min,
                                                    int32_t                                     clamp_max,
                                                    float*                                      y_accum,
                                                    float*                                      cbcr_accum);
__global__ void transformed_dct_grid_cached_kernel(const JpegDctDeviceCachedFixedTransformBatchItem* items,
                                                   size_t                                            item_count,
                                                   const uint16_t*                                   quant_tables,
                                                   const float* resize_weight_matrices,
                                                   int32_t      clamp_min,
                                                   int32_t      clamp_max,
                                                   float*       y_accum,
                                                   float*       cbcr_accum);

} // namespace galp::jpeg::detail
