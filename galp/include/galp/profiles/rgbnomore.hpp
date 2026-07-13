#ifndef GALP_PROFILES_RGBNOMORE_HPP
#define GALP_PROFILES_RGBNOMORE_HPP

#include "galp/jpeg_dct.hpp"

namespace galp::profiles {

// Optional application profile. No RGB-no-more name or geometry is required by
// the generic JPEG-DCT API or executor when this header is not included.
inline jpeg::JpegDctGridTransformSpec rgbnomore_val_dct_grid_transform() {
	jpeg::JpegDctGridTransformSpec spec;
	spec.y_output_width_blocks     = 28;
	spec.y_output_height_blocks    = 28;
	spec.cbcr_output_width_blocks  = 14;
	spec.cbcr_output_height_blocks = 14;
	spec.crop_reference_width_blocks  = 32;
	spec.crop_reference_height_blocks = 32;
	spec.crop_origin_alignment_blocks = 2;
	spec.chroma_crop_scale_x           = 2;
	spec.chroma_crop_scale_y           = 2;
	spec.clamp_min                      = -1024;
	spec.clamp_max                      = 1016;
	spec.dequantize                     = true;
	spec.require_all_coefficients       = true;
	spec.allow_grayscale                = true;
	spec.preferred_small_crop_width_blocks  = {2, 4, 14, 28};
	spec.preferred_small_crop_height_blocks = {2, 4, 14, 28};
	spec.allowed_chroma_sampling_ratios = {
	    jpeg::JpegDctSamplingRatio {1, 1, 1, 1},
	    jpeg::JpegDctSamplingRatio {1, 2, 1, 2},
	};
	return spec;
}

} // namespace galp::profiles

#endif // GALP_PROFILES_RGBNOMORE_HPP
