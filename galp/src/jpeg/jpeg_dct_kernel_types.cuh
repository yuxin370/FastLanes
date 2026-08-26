#ifndef GALP_JPEG_DCT_KERNEL_TYPES_CUH
#define GALP_JPEG_DCT_KERNEL_TYPES_CUH

#include "jpeg/jpeg_dct_plan_types.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace galp::jpeg::detail {

enum class DeviceCoeffSource : uint8_t {
	kMissing = 0,
	kI8      = 1,
	kI16     = 2,
};

struct DeviceCoeffBinding {
	const int8_t*     column_i8  = nullptr;
	const int16_t*    column_i16 = nullptr;
	DeviceCoeffSource source     = DeviceCoeffSource::kMissing;
};

// Request-scoped sparse frequency plan. The transform output remains a dense
// 8x8 grid; these compact lists remove zero raw coefficients from the two
// separable mixing passes without materializing per-rowgroup remap tables.
struct JpegDctDeviceSparseTransformPlan {
	static constexpr uint8_t kMissingBinding = std::numeric_limits<uint8_t>::max();

	uint8_t selected_coefficient_count = 0U;
	std::array<uint8_t, 64> natural_to_compact_binding {};
	std::array<uint8_t, 8>  selected_y_count_by_x {};
	std::array<uint8_t, 64> selected_y_by_x {};
	std::array<uint8_t, 8>  selected_x_count_by_y {};
	std::array<uint8_t, 64> selected_x_by_y {};
	uint8_t                 active_x_count = 0U;
	std::array<uint8_t, 8>  active_x {};
	uint8_t                 active_y_count = 0U;
	std::array<uint8_t, 8>  active_y {};
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
	uint32_t binding_base          = 0;
	uint32_t row_in_rowgroup       = 0;
	uint64_t output_block_index    = 0;
	uint8_t  component             = 0;
	uint8_t  zigzag_columns        = 0;
	uint8_t  horizontal_flip       = 0;
	uint16_t x_factor              = 2;
	uint16_t y_factor              = 2;
	uint8_t  x_subblock            = 0;
	uint8_t  y_subblock            = 0;
	uint8_t  x_upsample            = 0;
	uint8_t  y_upsample            = 0;
	uint16_t x_up_factor           = 1;
	uint16_t y_up_factor           = 1;
	uint16_t x_down_factor         = 1;
	uint16_t y_down_factor         = 1;
	uint32_t quant_table_index     = 0;
	uint32_t x_weight_matrix_index = 0;
	uint32_t y_weight_matrix_index = 0;
};

struct JpegDctDeviceCachedFixedTransformBatchItem {
	const int16_t*                       dense = nullptr;
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

__host__ __device__ inline size_t
selected_dct_binding_offset(const size_t source_index, const size_t coeff_slot, const size_t coefficients_per_block) {
	return source_index * coefficients_per_block + coeff_slot;
}

__host__ __device__ inline size_t selected_dct_output_offset(const size_t output_block_index,
                                                             const size_t coeff_slot,
                                                             const size_t coefficients_per_block) {
	return output_block_index * coefficients_per_block + coeff_slot;
}

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_KERNEL_TYPES_CUH
