"""Stored 512px 4:2:0 GALP coefficients to the DCTNet component geometry.

Inputs are quantized integers in natural (row-major frequency) order; outputs
are dequantized float32, before model channel selection and normalization.
Chroma interpolation is RGB-no-more's spectral zero-padding operation, with
neither clipping nor rounding. It is not pixel resizing followed by JPEG.
"""

import ast
import math
from pathlib import Path

import einops
import torch


def load_upsample_dct(rgbnomore_root: Path, *, resize=False):
    """Load upstream upsampling (or general resizing) without its global utils.

    Both upstream repositories use a top-level ``utils`` package. Loading only
    these self-contained function definitions avoids a module-name collision.
    """
    source = Path(rgbnomore_root) / "utils/dct_ops.py"
    names = {
        "generate_basis_matrix",
        "expand_basis_matrix_blockwise",
        "generate_conversion_matrix",
        "upsample_dct",
    }
    if resize:
        names.update(("downsample_dct", "resize_dct"))
    parsed = ast.parse(source.read_text(), filename=str(source))
    definitions = [
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise ValueError(f"Missing required DCT resampling functions in {source}")
    namespace = {"torch": torch, "einops": einops, "math": math}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["resize_dct" if resize else "upsample_dct"]


class StoredDctAdapter:
    def __init__(self, rgbnomore_root: Path, target_grid=56):
        if target_grid not in (56, 112):
            raise ValueError("Supported target block grids are 56 and 112")
        self.target_grid = target_grid
        self.upsample_dct = load_upsample_dct(rgbnomore_root)

    def __call__(self, coefficients, quantization_tables):
        """Adapt Y/Cb/Cr tensors [H,W,64], with matching natural-order QTs.

        The decoder/descriptor must supply the actual component shapes. Only
        the specified 64x64 / 32x32 / 32x32 geometry is supported.
        """
        if len(coefficients) != 3 or len(quantization_tables) != 3:
            raise ValueError("Expected three components and their three quantization tables")
        components = []
        for name, coefficient, table, size, border in zip(
            ("Y", "Cb", "Cr"), coefficients, quantization_tables,
            (64, 32, 32), (4, 2, 2),
        ):
            coefficient = torch.as_tensor(coefficient)
            table = torch.as_tensor(table, device=coefficient.device)
            if tuple(coefficient.shape) != (size, size, 64):
                raise ValueError(f"{name}: expected {(size, size, 64)}, got {tuple(coefficient.shape)}")
            if coefficient.is_floating_point() or coefficient.is_complex():
                raise TypeError(f"{name}: expected quantized integer coefficients")
            if tuple(table.shape) not in ((64,), (8, 8)):
                raise ValueError(f"{name}: expected natural-order 64-entry quantization table")
            dequantized = coefficient.to(torch.float32) * table.reshape(64).to(torch.float32)
            components.append(dequantized[border:-border, border:-border].contiguous())
        chroma = torch.stack(components[1:]).reshape(2, 28, 28, 8, 8)
        upsampled, _, _ = self.upsample_dct(chroma, L=2, M=2, dtype=torch.float32)
        if self.target_grid == 112:
            # Preserve the same 448px source crop. Increase all three 56x56
            # grids by two for the larger target, using the same DCT math.
            planes = torch.cat((components[0].reshape(1, 56, 56, 8, 8), upsampled))
            expanded, _, _ = self.upsample_dct(planes, L=2, M=2, dtype=torch.float32)
            return tuple(c.reshape(112, 112, 64) for c in expanded)
        return components[0], upsampled[0].reshape(56, 56, 64), upsampled[1].reshape(56, 56, 64)
