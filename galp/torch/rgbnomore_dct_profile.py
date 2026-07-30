"""Optional RGB-no-more parameters for GALP's generic transformed DCT grid."""

from __future__ import annotations

from typing import Any


RGBNOMORE_VAL_DCT_GRID_TRANSFORM: dict[str, Any] = {
    "y_output_width_blocks": 28,
    "y_output_height_blocks": 28,
    "cbcr_output_width_blocks": 14,
    "cbcr_output_height_blocks": 14,
    "crop_reference_width_blocks": 32,
    "crop_reference_height_blocks": 32,
    "crop_origin_alignment_blocks": 2,
    "chroma_crop_scale_x": 2,
    "chroma_crop_scale_y": 2,
    "clamp_min": -1024,
    "clamp_max": 1016,
    "dequantize": True,
    "require_all_coefficients": True,
    "allow_grayscale": True,
    "preferred_small_crop_width_blocks": [2, 4, 14, 28],
    "preferred_small_crop_height_blocks": [2, 4, 14, 28],
    # Full ImageNet-val contains 4:4:4, 4:2:0, 4:2:2, vertical 4:4:0,
    # 4:1:1, grayscale, and one four-component JPEG. RGB-no-more's own DCT
    # loader applies the same fixed chroma crop rule to all color layouts.
    "allowed_chroma_sampling_ratios": [
        (1, 1, 1, 1),
        (1, 2, 1, 2),
        (1, 2, 1, 1),
        (1, 1, 1, 2),
        (1, 4, 1, 1),
    ],
}

# Production pushdown output: the generic executor preserves the legacy
# nearbyint/clamp policy, then applies the same two FP32 operations previously
# issued by the PyTorch adapter. Keep the int16 profile above for compatibility
# and transform-reference diagnostics.
RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32: dict[str, Any] = {
    **RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
    "output_dtype": "float32",
    "output_add": 4.0,
    "output_scale": 1.0 / 1020.0,
}
