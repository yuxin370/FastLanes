// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/expression.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXPRESSION_CUH
#define ENGINE_EXPRESSION_CUH

#include "fls/footer/operator_token_generated.h"
#include "flsgpu/structs.cuh"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace reader {
struct Column;
struct Rowgroup;
} // namespace reader

namespace expr {

enum class OperatorKind {
	UNCOMPRESSED,
	UNFFOR,
	SLPATCH,
	RSUM,
	RLE,
	FREQUENCY,
	CROSS_RLE,
	DICT,
	CONSTANT,
	EQUAL,
};

struct Expression {
	reader::Column*           column; // non-owning
	std::vector<OperatorKind> ops;    // logical operator chain
};

inline std::vector<OperatorKind> ops_for_token(const fastlanes::OperatorToken token) {
	using enum fastlanes::OperatorToken;
	switch (token) {
	case EXP_UNCOMPRESSED_I08:
		return {OperatorKind::UNCOMPRESSED};
	case EXP_CONSTANT_I08:
		return {OperatorKind::CONSTANT};
	case EXP_FFOR_I08:
	case EXP_FFOR_I16:
		return {OperatorKind::UNFFOR};
	case EXP_FFOR_SLPATCH_I08:
	case EXP_FFOR_SLPATCH_I16:
		return {OperatorKind::UNFFOR, OperatorKind::SLPATCH};
	case EXP_FREQUENCY_I08:
	case EXP_FREQUENCY_I16:
		return {OperatorKind::FREQUENCY};
	case EXP_CROSS_RLE_I08:
		return {OperatorKind::CROSS_RLE};
	case EXP_RLE_I08_U16:
	case EXP_RLE_I16_U16:
		return {OperatorKind::UNFFOR, OperatorKind::RSUM, OperatorKind::RLE};
	case EXP_EQUAL:
		return {OperatorKind::EQUAL};
	case EXP_DICT_I08_FFOR_SLPATCH_U08:
	case EXP_DICT_I16_FFOR_SLPATCH_U08:
	case EXP_DICT_I16_FFOR_SLPATCH_U16:
		return {OperatorKind::UNFFOR, OperatorKind::SLPATCH, OperatorKind::DICT};
	case EXP_DICT_I08_FFOR_U08:
	case EXP_DICT_I16_FFOR_U16:
	case EXP_DICT_I16_FFOR_U08:
		return {OperatorKind::UNFFOR, OperatorKind::DICT};
	case EXP_DICT_I08_U08:
		return {OperatorKind::UNFFOR, OperatorKind::DICT};
	default:
		return {};
	}
}

std::vector<Expression> assemble(reader::Rowgroup& rowgroup);

} // namespace expr

namespace dispatch {

enum class PlanKind : uint8_t {
	UNCOMPRESSED,
	CONSTANT,
	UNFFOR,
	UNFFOR_SLPATCH,
	RLE,
	FREQUENCY,
	CROSS_RLE,
	DICT_FFOR,
	DICT_FFOR_SLPATCH,
};

enum class TypeTag : uint8_t {
	I8,
	I16,
};

struct WorkItemAny {
	uint32_t expr_index;
	uint32_t vector_index;
	TypeTag  type;
};

template <typename T>
constexpr TypeTag type_tag_for();

template <>
constexpr TypeTag type_tag_for<int8_t>() {
	return TypeTag::I8;
}

template <>
constexpr TypeTag type_tag_for<int16_t>() {
	return TypeTag::I16;
}

template <typename T>
struct DeviceExpression {
	PlanKind plan;
	size_t   n_values;
	uint8_t  dict_index_bits = 0;
	T*       out;
	union {
		flsgpu::device::BPColumn<T>                   bp;
		flsgpu::device::CONSTANTColumn<T>             constant;
		flsgpu::device::FFORColumn<T>                 ffor;
		flsgpu::device::SLPATCHColumn<T>              slpatch;
		flsgpu::device::DICTFFORColumn<T>             dictffor;
		flsgpu::device::DICTFFORColumn<T, uint8_t>    dictffor_u8;
		flsgpu::device::DICTSLPATCHColumn<T>          dictslpatch;
		flsgpu::device::DICTSLPATCHColumn<T, uint8_t> dictslpatch_u8;
		flsgpu::device::FREQColumn<T>                 freq;
		flsgpu::device::CROSSRLEColumn<T>             crossrle;
		flsgpu::device::RLEColumn<T, uint16_t>        rle;
	} col;
};

inline bool ops_match(const std::vector<expr::OperatorKind>& ops, std::initializer_list<expr::OperatorKind> expected) {
	if (ops.size() != expected.size()) {
		return false;
	}
	size_t idx = 0;
	for (auto kind : expected) {
		if (ops[idx++] != kind) {
			return false;
		}
	}
	return true;
}

inline PlanKind plan_for_ops(const std::vector<expr::OperatorKind>& ops) {
	using enum expr::OperatorKind;
	if (ops_match(ops, {UNCOMPRESSED})) {
		return PlanKind::UNCOMPRESSED;
	}
	if (ops_match(ops, {CONSTANT})) {
		return PlanKind::CONSTANT;
	}
	if (ops_match(ops, {UNFFOR})) {
		return PlanKind::UNFFOR;
	}
	if (ops_match(ops, {UNFFOR, SLPATCH})) {
		return PlanKind::UNFFOR_SLPATCH;
	}
	if (ops_match(ops, {FREQUENCY})) {
		return PlanKind::FREQUENCY;
	}
	if (ops_match(ops, {CROSS_RLE})) {
		return PlanKind::CROSS_RLE;
	}
	if (ops_match(ops, {UNFFOR, RSUM, RLE})) {
		return PlanKind::RLE;
	}
	if (ops_match(ops, {UNFFOR, DICT})) {
		return PlanKind::DICT_FFOR;
	}
	if (ops_match(ops, {UNFFOR, SLPATCH, DICT})) {
		return PlanKind::DICT_FFOR_SLPATCH;
	}
	throw std::runtime_error("unsupported operator chain in dispatch");
}

} // namespace dispatch

#endif // ENGINE_EXPRESSION_CUH
