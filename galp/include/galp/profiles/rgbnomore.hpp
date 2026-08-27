#ifndef GALP_PROFILES_RGBNOMORE_HPP
#define GALP_PROFILES_RGBNOMORE_HPP

#include "galp/profiles/direct_dct.hpp"

#include <utility>

namespace galp::profiles {

inline constexpr std::string_view kRgbNoMoreValidationProfileId = "rgbnomore-validation-v1";
inline constexpr std::string_view kRgbNoMoreValidationCenterCrop512ProfileId =
    "rgbnomore-validation-center-crop-512-v1";
inline constexpr std::string_view kRgbNoMoreTrainingPlsProfileId = "rgbnomore-training-pls-v1";

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
	// Missing raw source coefficients are defined as zero before
	// dequantization and frequency mixing.  The model-facing transformed grid
	// remains dense, so callers may select any valid JPEG coefficient subset
	// without changing the RGB-no-more tensor contract.
	spec.require_all_coefficients       = false;
	spec.allow_grayscale                = true;
	spec.preferred_small_crop_width_blocks  = {2, 4, 14, 28};
	spec.preferred_small_crop_height_blocks = {2, 4, 14, 28};
	spec.allowed_chroma_sampling_ratios = {
	    jpeg::JpegDctSamplingRatio {1, 1, 1, 1},
	    jpeg::JpegDctSamplingRatio {1, 2, 1, 2},
	    jpeg::JpegDctSamplingRatio {1, 2, 1, 1},
	    jpeg::JpegDctSamplingRatio {1, 1, 1, 2},
	    jpeg::JpegDctSamplingRatio {1, 4, 1, 1},
	};
	return spec;
}

// Canonical model-ready output used by the production PyTorch pipeline.  Keep
// the affine conversion beside the geometry so Python never has to reproduce
// the executor's block/chroma/numeric contract.
inline jpeg::JpegDctGridTransformSpec rgbnomore_val_dct_grid_transform_fp32() {
	auto spec             = rgbnomore_val_dct_grid_transform();
	spec.output_data_type = jpeg::JpegDctGridOutputDataType::kFloat32;
	spec.output_add       = 4.0F;
	spec.output_scale     = 1.0F / 1020.0F;
	return spec;
}

inline DirectDctOutputProfile rgbnomore_validation_output_profile() {
	DirectDctOutputProfile output;
	output.id             = kRgbNoMoreValidationProfileId;
	output.layout         = jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	output.grid_transform = rgbnomore_val_dct_grid_transform_fp32();
	return output;
}

inline RegisteredDirectDctProfile rgbnomore_validation_profile() {
	return RegisteredDirectDctProfile {
	    kRgbNoMoreValidationProfileId,
	    rgbnomore_validation_output_profile(),
	    compact_v3_runtime_policy(),
	};
}

// The fixed-512 variant changes the runtime policy, not RGB-no-more's model
// transform.  ResizedCenterCrop_DCT(32, 28) uses a 32-block pseudo-resize
// reference: a 64-block luma source is centre-cropped to 56 blocks and then
// downsampled to 28.  Replacing the reference with the 64-block source extent
// would incorrectly turn this into an identity resize of a 28-block crop.
inline RegisteredDirectDctProfile rgbnomore_validation_center_crop_512_profile() {
	auto output = rgbnomore_validation_output_profile();
	output.id   = kRgbNoMoreValidationCenterCrop512ProfileId;
	return RegisteredDirectDctProfile {
	    kRgbNoMoreValidationCenterCrop512ProfileId,
	    std::move(output),
	    block_major_scheduled_bounded_runtime_policy(),
	};
}

// Physical-PLS training uses arbitrary keyed source crops, but the same 28/14
// model grid. The block-major scheduled runtime performs physical-order reads
// and writes transformed images directly into the native scheduler's shuffled
// output slots. M/G are epoch-scheduler fields, not output-profile fields.
inline RegisteredDirectDctProfile rgbnomore_training_pls_profile() {
	DirectDctOutputProfile output;
	output.id             = kRgbNoMoreTrainingPlsProfileId;
	output.layout         = jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	// RandAugment operates in the published dequantized int16 domain; the
	// native PLS CUDA postprocessor performs normalization and Mixup afterward.
	output.grid_transform = rgbnomore_val_dct_grid_transform();
	return RegisteredDirectDctProfile {
	    kRgbNoMoreTrainingPlsProfileId,
	    std::move(output),
	    block_major_dynamic_crop_bounded_runtime_policy(),
	};
}

} // namespace galp::profiles

#endif // GALP_PROFILES_RGBNOMORE_HPP
