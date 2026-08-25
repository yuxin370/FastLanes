#include "direct_dct/profile_registry.hpp"
#include <stdexcept>
#include <string>
#include <utility>

namespace galp::direct_dct {
namespace {

jpeg::JpegDctGridTransformSpec rgbnomore_transform() {
	jpeg::JpegDctGridTransformSpec spec;
	spec.y_output_width_blocks              = 28U;
	spec.y_output_height_blocks             = 28U;
	spec.cbcr_output_width_blocks           = 14U;
	spec.cbcr_output_height_blocks          = 14U;
	spec.crop_reference_width_blocks        = 32U;
	spec.crop_reference_height_blocks       = 32U;
	spec.crop_origin_alignment_blocks       = 2U;
	spec.chroma_crop_scale_x                = 2U;
	spec.chroma_crop_scale_y                = 2U;
	spec.clamp_min                          = -1024;
	spec.clamp_max                          = 1016;
	spec.dequantize                         = true;
	spec.require_all_coefficients           = false;
	spec.allow_grayscale                    = true;
	spec.preferred_small_crop_width_blocks  = {2U, 4U, 14U, 28U};
	spec.preferred_small_crop_height_blocks = {2U, 4U, 14U, 28U};
	spec.allowed_chroma_sampling_ratios     = {
        jpeg::JpegDctSamplingRatio {1U, 1U, 1U, 1U},
        jpeg::JpegDctSamplingRatio {1U, 2U, 1U, 2U},
        jpeg::JpegDctSamplingRatio {1U, 2U, 1U, 1U},
        jpeg::JpegDctSamplingRatio {1U, 1U, 1U, 2U},
        jpeg::JpegDctSamplingRatio {1U, 4U, 1U, 1U},
    };
	return spec;
}

jpeg::JpegDctGridTransformSpec rgbnomore_validation_transform() {
	auto spec             = rgbnomore_transform();
	spec.output_data_type = jpeg::JpegDctGridOutputDataType::kFloat32;
	spec.output_add       = 4.0F;
	spec.output_scale     = 1.0F / 1020.0F;
	return spec;
}

} // namespace

SemanticProfileRegistry::SemanticProfile SemanticProfileRegistry::resolve(const std::string_view profile_id) {
	if (profile_id == kProfileIds[0]) {
		return SemanticProfile {
		    kProfileIds[0],
		    jpeg::JpegDctDeviceLayout::kTransformedDctGrid,
		    rgbnomore_validation_transform(),
		    {},
		};
	}
	if (profile_id == kProfileIds[1]) {
		return SemanticProfile {
		    kProfileIds[1],
		    jpeg::JpegDctDeviceLayout::kTransformedDctGrid,
		    rgbnomore_validation_transform(),
		    {},
		};
	}
	if (profile_id == kProfileIds[2]) {
		return SemanticProfile {
		    kProfileIds[2],
		    jpeg::JpegDctDeviceLayout::kTransformedDctGrid,
		    rgbnomore_transform(),
		    {},
		};
	}
	throw std::invalid_argument("unknown shadow Direct-DCT semantic profile '" + std::string(profile_id) + "'");
}

} // namespace galp::direct_dct
