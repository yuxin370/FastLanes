#ifndef GALP_DIRECT_DCT_PROFILE_REGISTRY_HPP
#define GALP_DIRECT_DCT_PROFILE_REGISTRY_HPP

#include "galp/jpeg_dct_device.hpp"
#include <array>
#include <optional>
#include <string_view>

namespace galp::direct_dct {

// Shadow-only canonical candidate. This registry owns semantic output
// contracts; execution tuning is resolved separately.
class SemanticProfileRegistry final {
public:
	struct SemanticProfile final {
		std::string_view                              id;
		jpeg::JpegDctDeviceLayout                     output_layout;
		std::optional<jpeg::JpegDctGridTransformSpec> grid_transform;
		jpeg::JpegDctCoefficientSelection             coefficient_selection;
	};

	inline static constexpr std::array<std::string_view, 3> kProfileIds = {
	    "rgbnomore-validation-v1",
	    "rgbnomore-validation-center-crop-512-v1",
	    "rgbnomore-training-pls-v1",
	};

	[[nodiscard]] static constexpr const auto& available_profile_ids() noexcept {
		return kProfileIds;
	}

	[[nodiscard]] static SemanticProfile resolve(std::string_view profile_id);
};

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_PROFILE_REGISTRY_HPP
