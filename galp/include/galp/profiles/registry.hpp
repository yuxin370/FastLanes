#ifndef GALP_PROFILES_REGISTRY_HPP
#define GALP_PROFILES_REGISTRY_HPP

#include "galp/profiles/rgbnomore.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace galp::profiles {

inline std::vector<std::string> available_direct_dct_profile_ids() {
	return {
	    std::string(kRgbNoMoreValidationProfileId),
	    std::string(kRgbNoMoreValidationCenterCrop512ProfileId),
	};
}

inline RegisteredDirectDctProfile resolve_direct_dct_profile(const std::string_view profile_id) {
	if (profile_id == kRgbNoMoreValidationProfileId) {
		return rgbnomore_validation_profile();
	}
	if (profile_id == kRgbNoMoreValidationCenterCrop512ProfileId) {
		return rgbnomore_validation_center_crop_512_profile();
	}
	throw std::invalid_argument(
	    "unknown Direct-DCT profile '" + std::string(profile_id) +
	    "'; call available_direct_dct_profiles() for the registered profile ids");
}

} // namespace galp::profiles

#endif // GALP_PROFILES_REGISTRY_HPP
