"""Full-source 512px JPEG contract and online DCT geometry, shared by all CNNs."""
from pathlib import Path

import torch

from galp.benchmarks.dct_models.dct_geometry import load_upsample_dct


SOURCE_PROFILE = dict(input_geometry="source512", source_pixels=[512, 512],
                      component_blocks=[[64, 64], [32, 32], [32, 32]],
                      coefficients_per_component=64,
                      offline_geometry=False, offline_frequency_selection=False)
CROP = [32, 32, 448, 448]  # x, y, width, height in source pixels


def read_source_jpeg(path):
    """Read original JPEG coefficients for verification, independent of GALP layout."""
    import numpy as np
    from jpeg2dct.numpy import load
    from PIL import Image, JpegImagePlugin
    with Image.open(path) as image:
        if image.size != (512, 512) or JpegImagePlugin.get_sampling(image) != 2:
            raise ValueError("source reference requires 512px 4:2:0 JPEG")
        tables = [np.asarray(image.quantization[component[3]], dtype=np.uint16)
                  for component in image.layer]
    return load(str(path), normalized=False), tables


def source_options(options, execution_mode):
    """Keep all source frequencies; model channel projection happens after resize."""
    options["dct_coeffs"] = "all"
    options["crop_execution_mode"] = execution_mode
    options["grid_transform"].update(
        crop_reference_width_blocks=64, crop_reference_height_blocks=64,
        crop_origin_alignment_blocks=2, chroma_crop_scale_x=2, chroma_crop_scale_y=2,
        allowed_chroma_sampling_ratios=[[1, 2, 1, 2]], require_all_coefficients=True)
    return options


class SourceReference:
    """Independent CPU crop/resize before native final rounding and saturation."""
    def __init__(self, upstream: Path, grid: int):
        self.resize = load_upsample_dct(upstream, resize=True)
        self.grid = grid

    def __call__(self, coefficients, tables):
        result = []
        for c, (q, table) in enumerate(zip(coefficients, tables)):
            size, border = (64, 4) if c == 0 else (32, 2)
            if tuple(q.shape) != (size, size, 64):
                raise ValueError(f"expected full source component {(size, size, 64)}, got {q.shape}")
            raw = torch.as_tensor(q).float() * torch.as_tensor(table).reshape(64)
            raw = raw[border:-border, border:-border].reshape(1, size-2*border, size-2*border, 8, 8)
            result.append(self.resize(raw, self.grid, dtype_out=torch.float32, conv_mxs={})
                          .reshape(self.grid, self.grid, 64))
        return result
