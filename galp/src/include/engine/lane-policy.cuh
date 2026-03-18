#ifndef ENGINE_LANE_POLICY_CUH
#define ENGINE_LANE_POLICY_CUH

#include "engine/expression.cuh"
#include "flsgpu/utils.cuh"

namespace dispatch {

__host__ __device__ __forceinline__ constexpr uint32_t lane_count_for_type(const TypeTag type) {
	switch (type) {
	case TypeTag::I8:
		return static_cast<uint32_t>(utils::get_n_lanes<int8_t>());
	case TypeTag::I16:
		return static_cast<uint32_t>(utils::get_n_lanes<int16_t>());
	default:
		return 0;
	}
}

__host__ __device__ __forceinline__ constexpr TypeTag semantic_lane_type(const TypeTag value_type,
                                                                         const PlanKind plan) {
	switch (plan) {
	case PlanKind::DICT_FFOR_U8:
	case PlanKind::DICT_FFOR_SLPATCH_U8:
	case PlanKind::RLE_U8:
		return TypeTag::I8;
	case PlanKind::DICT_FFOR_U16:
	case PlanKind::DICT_FFOR_SLPATCH_U16:
	case PlanKind::RLE_U16:
		return TypeTag::I16;
	default:
		return value_type;
	}
}

__host__ __device__ __forceinline__ constexpr uint32_t semantic_lane_count(const TypeTag value_type,
                                                                           const PlanKind plan) {
	return lane_count_for_type(semantic_lane_type(value_type, plan));
}

template <typename T>
__host__ __device__ __forceinline__ constexpr uint32_t semantic_lane_count(const DeviceExpression<T>& expr) {
	return semantic_lane_count(type_tag_for<T>(), expr.plan);
}

} // namespace dispatch

#endif // ENGINE_LANE_POLICY_CUH
