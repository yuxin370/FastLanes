// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/expression.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXPRESSION_CUH
#define ENGINE_EXPRESSION_CUH

#include "codecs/encodings/all.cuh"
#include "core/data/model.cuh"
#include "fls/footer/operator_token_generated.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace galp::expression {

struct Expression {
	galp::execution::Column* column; // non-owning
};

inline bool is_supported_token(const fastlanes::OperatorToken token) {
	using enum fastlanes::OperatorToken;
	switch (token) {
	case EXP_UNCOMPRESSED_I08:
	case EXP_CONSTANT_I08:
	case EXP_FFOR_I08:
	case EXP_FFOR_I16:
	case EXP_FFOR_SLPATCH_I08:
	case EXP_FFOR_SLPATCH_I16:
	case EXP_FREQUENCY_I08:
	case EXP_FREQUENCY_I16:
	case EXP_CROSS_RLE_I08:
	case EXP_CROSS_RLE_I16:
	case EXP_RLE_I08_U16:
	case EXP_RLE_I16_U16:
	case EXP_RLE_I08_SLPATCH_U16:
	case EXP_RLE_I16_SLPATCH_U16:
	case EXP_EQUAL:
	case EXP_DICT_I08_FFOR_SLPATCH_U08:
	case EXP_DICT_I16_FFOR_SLPATCH_U08:
	case EXP_DICT_I16_FFOR_SLPATCH_U16:
	case EXP_DICT_I08_FFOR_U08:
	case EXP_DICT_I16_FFOR_U16:
	case EXP_DICT_I16_FFOR_U08:
	case EXP_DICT_I08_U08:
		return true;
	default:
		return false;
	}
}

std::vector<Expression> assemble(galp::execution::Rowgroup& rowgroup);

} // namespace galp::expression

namespace galp::execution {

enum class PlanKind : uint8_t {
	UNCOMPRESSED,
	CONSTANT,
	UNFFOR,
	UNFFOR_SLPATCH,
	RLE_U8,
	RLE_U16,
	RLE_SLPATCH_U16,
	FREQUENCY,
	CROSS_RLE,
	DICT_FFOR_U8,
	DICT_FFOR_U16,
	DICT_FFOR_SLPATCH_U8,
	DICT_FFOR_SLPATCH_U16,
};

enum class TypeTag : uint8_t {
	I8,
	I16,
};

struct WorkItemAny {
	uint32_t expr_index;
	uint32_t vector_index;
	TypeTag  type;
	uint32_t output_vector_index;
};

inline constexpr uint32_t kInvalidExprIndex = std::numeric_limits<uint32_t>::max();

struct MixedWorkSlot {
	WorkItemAny first;
	WorkItemAny second;
};

__host__ __device__ constexpr inline WorkItemAny invalid_work_item() {
	return WorkItemAny {kInvalidExprIndex, 0, TypeTag::I8, 0};
}

__host__ __device__ constexpr inline bool is_valid_work_item(const WorkItemAny& work) {
	return work.expr_index != kInvalidExprIndex;
}

template <typename T>
constexpr TypeTag type_tag_for();

template <>
__host__ __device__ constexpr TypeTag type_tag_for<int8_t>() {
	return TypeTag::I8;
}

template <>
__host__ __device__ constexpr TypeTag type_tag_for<int16_t>() {
	return TypeTag::I16;
}

template <typename T>
struct DeviceExpression {
	PlanKind plan;
	size_t   n_values;
	size_t   output_n_values   = 0;
	bool     freq_use_extended = false;
	T*       out;
	union {
		galp::codec::device::BPColumn<T>                    bp;
		galp::codec::device::CONSTANTColumn<T>              constant;
		galp::codec::device::FFORColumn<T>                  ffor;
		galp::codec::device::SLPATCHColumn<T>               slpatch;
		galp::codec::device::DICTFFORColumn<T, uint16_t>    dictffor_u16;
		galp::codec::device::DICTFFORColumn<T, uint8_t>     dictffor_u8;
		galp::codec::device::DICTSLPATCHColumn<T, uint16_t> dictslpatch_u16;
		galp::codec::device::DICTSLPATCHColumn<T, uint8_t>  dictslpatch_u8;
		galp::codec::device::FREQColumn<T>                  freq;
		galp::codec::device::FREQExtendedColumn<T>          freq_extended;
		galp::codec::device::CROSSRLEColumn<T>              crossrle;
		galp::codec::device::RLEColumn<T, uint8_t>          rle_u8;
		galp::codec::device::RLEColumn<T, uint16_t>         rle_u16;
		galp::codec::device::RLESLPATCHColumn<T, uint16_t>  rle_slpatch_u16;
	} col;
};

} // namespace galp::execution

#endif // ENGINE_EXPRESSION_CUH
