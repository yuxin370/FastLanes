#include "galp/jpeg_dct_format.hpp"
#include "codecs/consts.cuh"
#include "jpeg/jpeg_dct_kernel_launch.cuh"

namespace galp::jpeg::detail {

// Match the float32 torch.mm conversion used by RGB-no-more exactly. The
// tiny non-zero entries are observable at round-to-even boundaries.
__device__ __constant__ float kRgbNoMoreDown2Conversion[8U * 16U] = {
#include "jpeg/jpeg_dct_reference_down2.inc"
};

__device__ __forceinline__ uint8_t natural_to_physical_coeff_device(const uint8_t natural,
                                                                    const uint8_t zigzag_columns) {
	if (zigzag_columns == 0) {
		return natural;
	}
	constexpr uint8_t natural_to_zigzag[64] {
	    0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42, 3,  8,  12, 17, 25, 30,
	    41, 43, 9,  11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38,
	    46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
	};
	return natural_to_zigzag[natural];
}

__device__ __forceinline__ float
finalize_dct_grid_float(const float value, const float output_add, const float output_scale) {
	float rounded = nearbyintf(value);
	rounded       = fminf(32767.0F, fmaxf(-32768.0F, rounded));
	rounded       = __fadd_rn(rounded, output_add);
	return __fmul_rn(rounded, output_scale);
}

__device__ __forceinline__ float dct_grid_madd_rn(const float lhs, const float rhs, const float accum) {
	return __fmaf_rn(lhs, rhs, accum);
}

__device__ __forceinline__ float dct_grid_add_rn(const float lhs, const float rhs) {
	return __fadd_rn(lhs, rhs);
}

__device__ __forceinline__ float normalize_reference_down2_value(const float value,
	                                                               const uint32_t factor_product) {
	return factor_product == 4U   ? __fmul_rn(value, 0.5F)
	       : factor_product == 2U ? __fdiv_rn(value, 0x1.6a09e60000000p+0F)
	                              : value;
}

__device__ __forceinline__ uint64_t remap_planless_row_base(
	const uint32_t remap_base,
	const uint32_t* __restrict logical_to_compact_vectors,
	const uint64_t logical_row) {
	if (remap_base == std::numeric_limits<uint32_t>::max()) {
		return logical_row;
	}
	if (logical_to_compact_vectors == nullptr) {
		return std::numeric_limits<uint64_t>::max();
	}
	const uint64_t logical_vector = logical_row / galp::codec::consts::VALUES_PER_VECTOR;
	const uint64_t row_in_vector  = logical_row % galp::codec::consts::VALUES_PER_VECTOR;
	const uint32_t compact_vector = logical_to_compact_vectors[remap_base + logical_vector];
	if (compact_vector == std::numeric_limits<uint32_t>::max()) {
		return std::numeric_limits<uint64_t>::max();
	}
	return static_cast<uint64_t>(compact_vector) * galp::codec::consts::VALUES_PER_VECTOR + row_in_vector;
}

__device__ __forceinline__ uint64_t remap_planless_row(
	const JpegDctDevicePlanlessImageDescriptor& image,
	const uint32_t* __restrict logical_to_compact_vectors,
	const uint64_t logical_row) {
	return remap_planless_row_base(image.vector_remap_base, logical_to_compact_vectors, logical_row);
}

struct PlanlessLocatedRow {
	uint64_t row          = std::numeric_limits<uint64_t>::max();
	uint32_t binding_base = std::numeric_limits<uint32_t>::max();
};

__device__ __forceinline__ bool block_major_group_less(const JpegDctDeviceBlockMajorGroupBinding& group,
	                                                    const uint32_t shard_id,
	                                                    const uint32_t semantic_slot_id,
	                                                    const uint32_t block_y,
	                                                    const uint32_t block_x) {
	if (group.shard_id != shard_id) {
		return group.shard_id < shard_id;
	}
	if (group.semantic_slot_id != semantic_slot_id) {
		return group.semantic_slot_id < semantic_slot_id;
	}
	if (group.block_y != block_y) {
		return group.block_y < block_y;
	}
	return group.block_x < block_x;
}

__device__ __forceinline__ const JpegDctDeviceBlockMajorGroupBinding* find_block_major_group(
	const JpegDctDeviceBlockMajorGroupBinding* __restrict groups,
	const size_t group_count,
	const uint32_t shard_id,
	const uint32_t semantic_slot_id,
	const uint32_t block_x,
	const uint32_t block_y) {
	size_t begin = 0U;
	size_t end   = group_count;
	while (begin < end) {
		const auto middle = begin + (end - begin) / 2U;
		if (block_major_group_less(groups[middle], shard_id, semantic_slot_id, block_y, block_x)) {
			begin = middle + 1U;
		} else {
			end = middle;
		}
	}
	if (begin >= group_count) {
		return nullptr;
	}
	const auto& group = groups[begin];
	return group.shard_id == shard_id && group.semantic_slot_id == semantic_slot_id &&
	               group.block_x == block_x && group.block_y == block_y
	           ? &group
	           : nullptr;
}

__device__ __forceinline__ bool consume_block_major_uleb128(const uint8_t* __restrict payload,
	                                                         const uint32_t payload_size,
	                                                         uint32_t& cursor,
	                                                         uint32_t& value) {
	value = 0U;
	for (uint32_t shift = 0U; shift < 32U; shift += 7U) {
		if (cursor >= payload_size) {
			return false;
		}
		const auto byte = payload[cursor++];
		if (shift == 28U && (byte & 0xf0U) != 0U) {
			return false;
		}
		value |= static_cast<uint32_t>(byte & 0x7fU) << shift;
		if ((byte & 0x80U) == 0U) {
			return true;
		}
	}
	return false;
}

__device__ __forceinline__ bool block_major_rank(const JpegDctDeviceBlockMajorRankCell& cell,
	                                              const uint8_t* __restrict rank_payload,
	                                              const size_t rank_payload_size,
	                                              const uint32_t local_image_index,
	                                              uint32_t& rank) {
	constexpr uint8_t empty_encoding       = 0U;
	constexpr uint8_t all_present_encoding = 1U;
	constexpr uint8_t sparse_encoding      = 2U;
	constexpr uint8_t missing_encoding     = 3U;
	constexpr uint8_t bitmap_encoding      = 4U;
	rank = 0U;
	if (local_image_index >= cell.image_count || cell.encoding == empty_encoding) {
		return false;
	}
	if (cell.encoding == all_present_encoding) {
		rank = local_image_index;
		return true;
	}
	if (cell.payload_offset > rank_payload_size || cell.payload_size > rank_payload_size - cell.payload_offset ||
	    rank_payload == nullptr) {
		return false;
	}
	const auto* payload = rank_payload + cell.payload_offset;
	if (cell.encoding == bitmap_encoding) {
		if (cell.rank_checkpoint_images == 0U) {
			return false;
		}
		const auto bit_bytes = (cell.image_count + 7U) / 8U;
		const auto checkpoint = local_image_index / cell.rank_checkpoint_images;
		const auto checkpoint_offset = bit_bytes + checkpoint * 2U;
		if (checkpoint_offset + 1U >= cell.payload_size || local_image_index / 8U >= bit_bytes) {
			return false;
		}
		rank = static_cast<uint32_t>(payload[checkpoint_offset]) |
		       (static_cast<uint32_t>(payload[checkpoint_offset + 1U]) << 8U);
		const auto begin = checkpoint * cell.rank_checkpoint_images;
		for (uint32_t image = begin; image < local_image_index; ++image) {
			rank += (payload[image / 8U] >> (image % 8U)) & 1U;
		}
		return ((payload[local_image_index / 8U] >> (local_image_index % 8U)) & 1U) != 0U;
	}
	if (cell.encoding != sparse_encoding && cell.encoding != missing_encoding) {
		return false;
	}
	const auto listed_count =
	    cell.encoding == sparse_encoding ? static_cast<uint32_t>(cell.present_count)
	                                    : cell.image_count - static_cast<uint32_t>(cell.present_count);
	uint32_t cursor = 0U;
	uint32_t value = 0U;
	uint32_t listed_before = 0U;
	bool listed = false;
	for (uint32_t index = 0U; index < listed_count; ++index) {
		uint32_t delta = 0U;
		if (!consume_block_major_uleb128(payload, cell.payload_size, cursor, delta)) {
			return false;
		}
		value = index == 0U ? delta : value + delta;
		if (value < local_image_index) {
			++listed_before;
		} else {
			listed = value == local_image_index;
			break;
		}
	}
	if (cell.encoding == sparse_encoding) {
		rank = listed_before;
		return listed;
	}
	rank = local_image_index - listed_before;
	return !listed;
}

__device__ uint64_t planless_block_order_rank(
    uint32_t width, uint32_t height, uint32_t x, uint32_t y, uint8_t spatial_order);

__device__ __forceinline__ PlanlessLocatedRow locate_planless_row(
	const JpegDctDevicePlanlessImageDescriptor& image,
	const JpegDctDevicePlanlessComponentDescriptor& component,
	const uint32_t block_x,
	const uint32_t block_y,
	const uint32_t* __restrict logical_to_compact_vectors,
	const uint32_t* __restrict image_vector_bindings,
	const JpegDctDeviceBlockMajorGroupBinding* __restrict block_major_groups,
	const size_t block_major_group_count,
	const JpegDctDeviceBlockMajorRankCell* __restrict block_major_rank_cells,
	const size_t block_major_rank_cell_count,
	const uint8_t* __restrict block_major_rank_payload,
	const size_t block_major_rank_payload_size) {
	if (block_major_groups == nullptr) {
		const auto rank = planless_block_order_rank(
		    component.width_in_blocks, component.height_in_blocks, block_x, block_y, image.spatial_order);
		const auto logical_row =
		    static_cast<uint64_t>(image.row_start_in_rowgroup) + component.component_row_offset + rank;
		if (image.vector_binding_count != 0U) {
			const auto logical_vector = logical_row / galp::codec::consts::VALUES_PER_VECTOR;
			if (image_vector_bindings == nullptr || logical_vector >= image.vector_binding_count) {
				return {};
			}
			const auto binding_base = image_vector_bindings[image.vector_binding_base + logical_vector];
			if (binding_base == std::numeric_limits<uint32_t>::max()) {
				return {};
			}
			return {logical_row % galp::codec::consts::VALUES_PER_VECTOR, binding_base};
		}
		return {remap_planless_row(image, logical_to_compact_vectors, logical_row), image.binding_base};
	}
	const auto* group = find_block_major_group(block_major_groups,
	                                           block_major_group_count,
	                                           image.shard_id,
	                                           component.semantic_slot_id,
	                                           block_x,
	                                           block_y);
	if (group == nullptr || group->rank_cell_index >= block_major_rank_cell_count ||
	    group->coefficient_binding_base == std::numeric_limits<uint32_t>::max()) {
		return {};
	}
	uint32_t rank = 0U;
	if (!block_major_rank(block_major_rank_cells[group->rank_cell_index],
	                     block_major_rank_payload,
	                     block_major_rank_payload_size,
	                     image.local_image_index,
	                     rank)) {
		return {};
	}
	const auto logical_row = static_cast<uint64_t>(group->row_start_in_rowgroup) + rank;
	return {remap_planless_row_base(group->vector_remap_base, logical_to_compact_vectors, logical_row),
	        group->coefficient_binding_base};
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
	                                                          const bool     accumulate,
                                                              float* __restrict y_accum,
                                                              float* __restrict cbcr_accum) {
	if (lane >= 64U) {
		return;
	}
	const auto stored_x     = image.horizontal_flip != 0U
	                              ? ((component == 0U ? y_output_width : cbcr_output_width) - 1U - output_x)
	                              : output_x;
	const auto stored_value = image.horizontal_flip != 0U && (lane % 8U) % 2U != 0U ? -value : value;
	if (component == 0U && y_accum != nullptr) {
		const auto output_block_index =
		    (static_cast<uint64_t>(image.request_index) * y_output_height + output_y) * y_output_width + stored_x;
		if (accumulate) {
			y_accum[output_block_index * 64U + lane] += stored_value;
		} else {
			y_accum[output_block_index * 64U + lane] = stored_value;
		}
	} else if (component != 0U && cbcr_accum != nullptr) {
		const auto output_block_index =
		    ((static_cast<uint64_t>(image.request_index) * 2U + component - 1U) * cbcr_output_height + output_y) *
		        cbcr_output_width +
		    stored_x;
		if (accumulate) {
			cbcr_accum[output_block_index * 64U + lane] += stored_value;
		} else {
			cbcr_accum[output_block_index * 64U + lane] = stored_value;
		}
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
                                                     const uint32_t* __restrict logical_to_compact_vectors,
	                                                     const uint32_t* __restrict image_vector_bindings,
	                                                     const uint32_t* __restrict active_output_blocks,
	                                                     const JpegDctDeviceBlockMajorGroupBinding* __restrict block_major_groups,
	                                                     const size_t block_major_group_count,
	                                                     const JpegDctDeviceBlockMajorRankCell* __restrict block_major_rank_cells,
	                                                     const size_t block_major_rank_cell_count,
	                                                     const uint8_t* __restrict block_major_rank_payload,
	                                                     const size_t block_major_rank_payload_size,
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
	const bool       accumulates_block_major_partials = block_major_groups != nullptr;
	__shared__ float composed[16U * 16U];
	__shared__ float vertical[8U * 16U];
	__shared__ float source[64U];
	__shared__ float horizontal[64U];
	__shared__ uint64_t located_rows[4];
	__shared__ uint32_t located_binding_bases[4];
	for (uint64_t launch_block = blockIdx.x; launch_block < output_block_count; launch_block += gridDim.x) {
		const uint64_t output_index = output_block_offset + launch_block;
		const uint64_t linear_block =
		    active_output_blocks == nullptr ? output_index : active_output_blocks[output_index];
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
			const auto source_width       = x_down * 8U;
			const auto source_block_count = x_down * y_down;
			// Keep the block at two warps: the transform produces 64 coefficients, and
			// the largest canonical down2 source contains only four coefficients per
			// lane.  A 256-thread block left six warps idle after the source load and
			// needlessly limited residency across the tens of thousands of output
			// blocks in an ImageNet batch.
			// Locate the at-most-four down2 source blocks in parallel, then let every
			// coefficient lane load all of its source values. The previous lane-0
			// loop placed a block-wide barrier on both sides of every source load.
			if (lane < source_block_count) {
				const uint32_t source_block_slot = lane;
				const auto subblock_x = source_block_slot % x_down;
				const auto subblock_y = source_block_slot / x_down;
				const auto source_x_i = static_cast<int64_t>(descriptor.crop_x) + output_x * x_down + subblock_x;
				const auto source_y_i = static_cast<int64_t>(descriptor.crop_y) + output_y * y_down + subblock_y;
				const bool source_in_bounds = source_x_i >= 0 && source_y_i >= 0 &&
				                              source_x_i < descriptor.width_in_blocks &&
				                              source_y_i < descriptor.height_in_blocks;
				const auto located = source_in_bounds
				                         ? locate_planless_row(image,
				                                               descriptor,
				                                               static_cast<uint32_t>(source_x_i),
				                                               static_cast<uint32_t>(source_y_i),
				                                               logical_to_compact_vectors,
				                                               image_vector_bindings,
				                                               block_major_groups,
				                                               block_major_group_count,
				                                               block_major_rank_cells,
				                                               block_major_rank_cell_count,
				                                               block_major_rank_payload,
				                                               block_major_rank_payload_size)
				                         : PlanlessLocatedRow {};
				located_rows[source_block_slot]          = located.row;
				located_binding_bases[source_block_slot] = located.binding_base;
			}
			__syncthreads();
			const auto coeff    = static_cast<uint8_t>(lane);
			const auto physical = natural_to_physical_coeff_device(coeff, image.zigzag_columns != 0U);
			const auto quant =
			    static_cast<int32_t>(quant_tables[static_cast<size_t>(descriptor.quant_table_index) * 64U + coeff]);
			for (uint32_t source_block_slot = 0U; source_block_slot < source_block_count; ++source_block_slot) {
				const auto subblock_x = source_block_slot % x_down;
				const auto subblock_y = source_block_slot / x_down;
				int16_t    value    = 0;
				if (located_rows[source_block_slot] != std::numeric_limits<uint64_t>::max() &&
				    located_binding_bases[source_block_slot] != std::numeric_limits<uint32_t>::max()) {
					const auto binding = column_bindings[located_binding_bases[source_block_slot] + physical];
					if (binding.source == DeviceCoeffSource::kI16) {
						value = binding.column_i16[located_rows[source_block_slot]];
					} else if (binding.source == DeviceCoeffSource::kI8) {
						value = static_cast<int16_t>(binding.column_i8[located_rows[source_block_slot]]);
					}
				}
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
						sum = dct_grid_madd_rn(kRgbNoMoreDown2Conversion[out_y * 16U + source_y],
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
						sum = dct_grid_madd_rn(vertical[out_y * source_width + source_x],
						                       kRgbNoMoreDown2Conversion[out_x * 16U + source_x],
						                       sum);
					}
				}
				const auto factor_product = x_down * y_down;
				const auto value          = normalize_reference_down2_value(sum, factor_product);
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
				                              accumulates_block_major_partials,
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
				const auto source_x_i = static_cast<int64_t>(descriptor.crop_x) + source_x_block;
				const auto source_y_i = static_cast<int64_t>(descriptor.crop_y) + source_y_block;
				const bool source_in_bounds = source_x_i >= 0 && source_y_i >= 0 &&
				                              source_x_i < descriptor.width_in_blocks &&
				                              source_y_i < descriptor.height_in_blocks;
				if (lane == 0U) {
					const auto located = source_in_bounds
					                         ? locate_planless_row(image,
					                                               descriptor,
					                                               static_cast<uint32_t>(source_x_i),
						                                               static_cast<uint32_t>(source_y_i),
						                                               logical_to_compact_vectors,
						                                               image_vector_bindings,
						                                               block_major_groups,
					                                               block_major_group_count,
					                                               block_major_rank_cells,
					                                               block_major_rank_cell_count,
					                                               block_major_rank_payload,
					                                               block_major_rank_payload_size)
					                         : PlanlessLocatedRow {};
					located_rows[0]         = located.row;
					located_binding_bases[0] = located.binding_base;
				}
				__syncthreads();
				if (lane < 64U) {
					const auto coeff = static_cast<uint8_t>(lane);
					const auto physical = natural_to_physical_coeff_device(coeff, image.zigzag_columns != 0U);
					int16_t    value    = 0;
					if (located_rows[0] != std::numeric_limits<uint64_t>::max() &&
					    located_binding_bases[0] != std::numeric_limits<uint32_t>::max()) {
						const auto binding = column_bindings[located_binding_bases[0] + physical];
						if (binding.source == DeviceCoeffSource::kI16) {
							value = binding.column_i16[located_rows[0]];
						} else if (binding.source == DeviceCoeffSource::kI8) {
							value = static_cast<int16_t>(binding.column_i8[located_rows[0]]);
						}
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
						x_sum = dct_grid_madd_rn(source[source_y_coeff * 8U + in_x_coeff], wx, x_sum);
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
						weighted = dct_grid_madd_rn(horizontal[in_y_coeff * 8U + out_x_coeff], wy, weighted);
					}
					output_sum = dct_grid_add_rn(output_sum, weighted);
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
			                              accumulates_block_major_partials,
			                              y_accum,
			                              cbcr_accum);
		}
		__syncthreads();
	}
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
		                           item.horizontal_flip == first_item.horizontal_flip &&
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
						sum = dct_grid_madd_rn(kRgbNoMoreDown2Conversion[out_y * 16U + source_y],
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
						sum = dct_grid_madd_rn(vertical[out_y * source_width + source_x],
						                       kRgbNoMoreDown2Conversion[out_x * 16U + source_x],
						                       sum);
				}
			}
			const auto target         = items[begin];
			const auto factor_product = x_down_factor * y_down_factor;
			const auto value          = normalize_reference_down2_value(sum, factor_product);
			const auto stored_value   = target.horizontal_flip != 0U && (lane % 8U) % 2U != 0U ? -value : value;
			if (target.component == 0U && y_accum != nullptr) {
				// Grouped launches are ordered on one transform stream. A legacy spatial-major output group can
				// span decode worksets, so retain the deterministic partial sum from every earlier launch.
				y_accum[target.output_block_index * 64U + lane] += stored_value;
			} else if (target.component != 0U && cbcr_accum != nullptr) {
				cbcr_accum[target.output_block_index * 64U + lane] += stored_value;
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
				x_sum = dct_grid_madd_rn(source[source_y * 8U + in_x_coeff], wx, x_sum);
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
				weighted = dct_grid_madd_rn(horizontal[in_y_coeff * 8U + out_x_coeff], wy, weighted);
			}
			sum = dct_grid_add_rn(sum, weighted);
		}
	}
	if (lane < 64U) {
		const auto target = items[begin];
		if (target.component == 0U && y_accum != nullptr) {
			y_accum[target.output_block_index * 64U + lane] += sum;
		} else if (target.component != 0U && cbcr_accum != nullptr) {
			cbcr_accum[target.output_block_index * 64U + lane] += sum;
		}
	}
}

__global__ void transformed_dct_grid_cached_kernel(const JpegDctDeviceCachedFixedTransformBatchItem* __restrict items,
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
	const auto       cached = items[item_idx];
	const auto       item   = cached.transform;
	__shared__ float source[64];
	__shared__ float horizontal[64];
	const auto       physical = natural_to_physical_coeff_device(lane, item.zigzag_columns);
	const auto       value    = cached.dense[item.row_in_rowgroup * 64U + physical];
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

} // namespace galp::jpeg::detail
