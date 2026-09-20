"""Official base eFUN model and its raw DCT input for the shared CNN benchmarks."""
from __future__ import annotations

import collections.abc
import importlib
import io
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "galp/benchmarks/system_rgbnomore"))
from inference.run import (DEFAULT_CHECKPOINT_DIR, DEFAULT_DATA_ROOT,
                           DEFAULT_E2E_V3_ROOT, DEFAULT_GALP_MANIFEST,
                           DEFAULT_INDEX_CSV, DEFAULT_RGBNOMORE_ROOT)

UPSTREAM = Path(os.environ.get("EFUN_ROOT", HERE / "FUN")).resolve()
PROFILE, MODEL_NAME, RUN_NAME = "efun", "eFUN", "efun"
SUBSET, GRID = "192", 28
CHECKPOINT = Path(os.environ.get("EFUN_CHECKPOINT", DEFAULT_CHECKPOINT_DIR / "efun.pth"))
INDICES = [list(range(64)) for _ in range(3)]
CHANNELS = list(range(192))
MEAN, STD = torch.zeros(192, 1, 1), torch.ones(192, 1, 1)
VALIDATION_DATA = DEFAULT_E2E_V3_ROOT / "dct_major_efun"


def upstream_timm():
    """Import the author's fork; its removed torch._six dependency only uses ABCs."""
    if not (UPSTREAM / "timm/models/eFUN.py").is_file():
        raise FileNotFoundError(f"Clone https://github.com/kfirgoldberg/FUN.git into {UPSTREAM}")
    if "timm" in sys.modules and Path(sys.modules["timm"].__file__).resolve().parent != UPSTREAM / "timm":
        raise RuntimeError("eFUN requires the author's timm fork; run efun.py in a separate process")
    sys.path.insert(0, str(UPSTREAM))
    six = types.ModuleType("torch._six")
    six.container_abcs = collections.abc
    sys.modules["torch._six"] = six
    return importlib.import_module("timm")


def model(checkpoint=CHECKPOINT):
    net = upstream_timm().create_model("efun", pretrained=False)
    # The published training checkpoint includes optimizer/argument metadata.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return net.eval().requires_grad_(False)


def organize(components):
    if [tuple(c.shape) for c in components] != [(GRID, GRID, 64)] * 3:
        raise ValueError("eFUN requires three 28x28 grids in natural frequency order")
    return torch.cat([torch.as_tensor(c).permute(2, 0, 1).float() for c in components]).contiguous()


class Reference:
    def __init__(self):
        upstream_timm()
        from timm.data import create_transform
        self.transform = create_transform((3, 224, 224), use_prefetcher=False,
                                          dct=True, interpolation="bicubic", crop_pct=0.875)

    def coefficients(self, path, verify=False):
        from jpeg2dct.numpy import loads
        with Image.open(path) as source:
            image = source.convert("RGB")
        # Run the exact official resize/crop/ToTensor prefix and its JPEG roundtrip.
        for transform in self.transform.transforms[:-1]:
            image = transform(image)
        output = io.BytesIO()
        T.ToPILImage()(image).save(output, format="jpeg", quality=100, subsampling=0)
        blob = output.getvalue()
        dct = self.transform.transforms[-1]
        value = dct._upsample_and_concat(*loads(blob, normalized=False))
        coefficients = tuple(c.permute(1, 2, 0).numpy().astype(np.int16)
                             for c in value.split(64))
        # Store the author's raw model values with identity scaling. These are
        # not the dequantized coefficients of the JPEG before jpeg2dct transcodes it.
        tables = np.ones((3, 64), dtype=np.uint16)
        if verify:
            torch.testing.assert_close(organize(coefficients), dct(image), rtol=0, atol=0)
        return np.stack(coefficients), tables, coefficients

    def __call__(self, path):
        with Image.open(path) as source:
            return self.transform(source.convert("RGB"))


def profile():
    return dict(model=MODEL_NAME, checkpoint=str(CHECKPOINT), shape=[192, GRID, GRID],
                component_order=["Y", "Cb", "Cr"], frequency_order="natural row-major u*8+v",
                indices=INDICES, mean=MEAN.flatten().tolist(), std=STD.flatten().tolist(),
                stored_dtype="int16 raw model coefficients", stored_component_shapes=[[GRID, GRID, 64]] * 3,
                dequantization="identity tables for stored raw values, not source JPEG quantization tables",
                jpeg_quality=100, jpeg_subsample=0, source_color="RGB",
                geometry="official Resize(256)/CenterCrop(224), Q100 JPEG 4:4:4, jpeg2dct raw decode and author chroma upsampling")
