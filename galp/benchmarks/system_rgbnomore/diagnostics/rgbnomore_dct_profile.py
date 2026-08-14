"""Experimental raw transform specs for RGB-no-more A/B diagnostics.

Production code must use ``galp.profiles.rgbnomore``.  These dictionaries are
retained only for diagnostics that intentionally exercise the private,
low-level binding with non-production variants.
"""

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
    "allowed_chroma_sampling_ratios": [
        (1, 1, 1, 1),
        (1, 2, 1, 2),
        (1, 2, 1, 1),
        (1, 1, 1, 2),
        (1, 4, 1, 1),
    ],
}

RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32: dict[str, Any] = {
    **RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
    "output_dtype": "float32",
    "output_add": 4.0,
    "output_scale": 1.0 / 1020.0,
}
