// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
#ifndef GALP_CORE_OPERATOR_CAPABILITIES_HPP
#define GALP_CORE_OPERATOR_CAPABILITIES_HPP

#include "fls/footer/operator_token_generated.h"
#include <array>
#include <cstdint>

namespace galp::execution {

enum class PlanKind : uint8_t {
	UNCOMPRESSED,
	CONSTANT,
	DELTA,
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

} // namespace galp::execution

namespace galp::expression {

// Single source of truth for every I8/I16 token that the FastLanes wizard can
// emit for GALP columns. This header deliberately has no CUDA dependencies so
// writer validation and audit tools consume the exact same table as dispatch.
struct OperatorCapability {
	fastlanes::OperatorToken token;
	bool                     gpu_supported;
	bool                     wizard_candidate;
	bool                     has_static_type;
	galp::execution::TypeTag type;
	bool                     has_static_plan;
	galp::execution::PlanKind plan;
	// Sparse range reads use the same per-vector segment entrypoints as GPU
	// materialization.  Keep the capability explicit so a future operator can
	// remain GPU-decodable while requiring a full-rowgroup storage fallback.
	bool sparse_read_supported = gpu_supported;
};

inline constexpr std::array<OperatorCapability, 29> kOperatorCapabilities {{
    {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::UNCOMPRESSED},
    {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::UNCOMPRESSED},
    {fastlanes::OperatorToken::EXP_CONSTANT_I08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::CONSTANT},
    {fastlanes::OperatorToken::EXP_CONSTANT_I16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::CONSTANT},
    {fastlanes::OperatorToken::EXP_FFOR_I08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::UNFFOR},
    {fastlanes::OperatorToken::EXP_FFOR_I16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::UNFFOR},
    {fastlanes::OperatorToken::EXP_FFOR_SLPATCH_I08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::UNFFOR_SLPATCH},
    {fastlanes::OperatorToken::EXP_FFOR_SLPATCH_I16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::UNFFOR_SLPATCH},
    {fastlanes::OperatorToken::EXP_DELTA_I08, true, true, true, execution::TypeTag::I8, true,
	 execution::PlanKind::DELTA},
    {fastlanes::OperatorToken::EXP_DELTA_I16, true, true, true, execution::TypeTag::I16, true,
	 execution::PlanKind::DELTA},
    {fastlanes::OperatorToken::EXP_FREQUENCY_I08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::FREQUENCY},
    {fastlanes::OperatorToken::EXP_FREQUENCY_I16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::FREQUENCY},
    {fastlanes::OperatorToken::EXP_CROSS_RLE_I08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::CROSS_RLE},
    {fastlanes::OperatorToken::EXP_CROSS_RLE_I16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::CROSS_RLE},
    {fastlanes::OperatorToken::EXP_RLE_I08_U16, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::RLE_U16},
    {fastlanes::OperatorToken::EXP_RLE_I16_U16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::RLE_U16},
    {fastlanes::OperatorToken::EXP_RLE_I08_SLPATCH_U16, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::RLE_SLPATCH_U16},
    {fastlanes::OperatorToken::EXP_RLE_I16_SLPATCH_U16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::RLE_SLPATCH_U16},
    // Equality aliases an earlier physical column, so its concrete type and
    // plan are inherited from that operand during materialization.
    {fastlanes::OperatorToken::EXP_EQUAL, true, true, false, execution::TypeTag::I8, false,
     execution::PlanKind::UNCOMPRESSED},
    // Nullable I16 columns can select this before expression competition.
    // JPEG-DCT columns are non-null, but it is still part of the I16 wizard
    // outcome space and must be represented explicitly in the capability map.
    {fastlanes::OperatorToken::EXP_NULL_I16, false, true, true, execution::TypeTag::I16, false,
     execution::PlanKind::UNCOMPRESSED},
    {fastlanes::OperatorToken::EXP_DICT_I08_FFOR_SLPATCH_U08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::DICT_FFOR_SLPATCH_U8},
    {fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U08, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::DICT_FFOR_SLPATCH_U8},
    {fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::DICT_FFOR_SLPATCH_U16},
    {fastlanes::OperatorToken::EXP_DICT_I08_FFOR_U08, true, true, true, execution::TypeTag::I8, true,
     execution::PlanKind::DICT_FFOR_U8},
    {fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::DICT_FFOR_U8},
    {fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U16, true, true, true, execution::TypeTag::I16, true,
     execution::PlanKind::DICT_FFOR_U16},
	// External dictionary references are resolved to an existing local
	// DICT_FFOR/DICT_SLPATCH plan before GPU dispatch.
	{fastlanes::OperatorToken::EXP_DICT_I16_U08, true, true, true, execution::TypeTag::I16, false,
	 execution::PlanKind::DICT_FFOR_U8},
	{fastlanes::OperatorToken::EXP_DICT_I16_U16, true, true, true, execution::TypeTag::I16, false,
	 execution::PlanKind::DICT_FFOR_U16},
    {fastlanes::OperatorToken::EXP_DICT_I08_U08, true, true, true, execution::TypeTag::I8, false,
     execution::PlanKind::DICT_FFOR_U8},
}};

constexpr const OperatorCapability* capability_for_token(const fastlanes::OperatorToken token) {
	for (const auto& capability : kOperatorCapabilities) {
		if (capability.token == token) {
			return &capability;
		}
	}
	return nullptr;
}

constexpr bool is_supported_token(const fastlanes::OperatorToken token) {
	const auto* capability = capability_for_token(token);
	return capability != nullptr && capability->gpu_supported;
}

constexpr bool is_sparse_read_supported_token(const fastlanes::OperatorToken token) {
	const auto* capability = capability_for_token(token);
	return capability != nullptr && capability->sparse_read_supported;
}

} // namespace galp::expression

#endif // GALP_CORE_OPERATOR_CAPABILITIES_HPP
