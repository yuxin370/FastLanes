"""Official DCTNet input contracts; upstream files remain unchanged."""
from __future__ import annotations

import collections
import collections.abc
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "galp/benchmarks/system_rgbnomore"))
from inference.run import (DEFAULT_CHECKPOINT_DIR, DEFAULT_DATA_ROOT,
                           DEFAULT_E2E_V3_ROOT, DEFAULT_GALP_MANIFEST,
                           DEFAULT_INDEX_CSV, DEFAULT_RGBNOMORE_ROOT)

UPSTREAM = HERE / "DCTNet/classification"
PROFILE = os.environ.get("DCTNET_PROFILE", "resnet64")
if PROFILE in ("resnet24", "resnet64"):
    SUBSET, GRID, RESIZE, CROP = PROFILE.removeprefix("resnet"), 56, 512, 448
    MODEL_NAME = "ResNetDCT_Upscaled_Static"
    RUN_NAME = f"dctnet_static{SUBSET}"
    CHECKPOINT = DEFAULT_CHECKPOINT_DIR / f"resnet50dct_upscaled_static_{SUBSET}/model_best.pth.tar"
elif PROFILE in ("mobilenet24", "mobilenet32"):
    SUBSET, GRID, RESIZE, CROP = PROFILE.removeprefix("mobilenet"), 112, 1024, 896
    MODEL_NAME = "MobileNetV2DCT_Subset_woinp"
    RUN_NAME = f"dctnet_mobilenet{SUBSET}"
    CHECKPOINT = DEFAULT_CHECKPOINT_DIR / f"mobilenetv2dct_upscaled_static_{SUBSET}/model_best.pth.tar"
else:
    raise ValueError(f"unknown DCTNET_PROFILE: {PROFILE}")
sys.path.insert(0, str(UPSTREAM))
# Python 3.10 moved this ABC; no numerical behavior changes.
collections.Iterable = collections.abc.Iterable
from datasets import train_upscaled_static_mean, train_upscaled_static_std
from main import subset_channel_index

INDICES = subset_channel_index[SUBSET]
CHANNELS = [c + 64 * component for component, indices in enumerate(INDICES) for c in indices]
MEAN = torch.tensor([train_upscaled_static_mean[c] for c in CHANNELS])[:, None, None]
STD = torch.tensor([train_upscaled_static_std[c] for c in CHANNELS])[:, None, None]


def model(checkpoint=CHECKPOINT):
    if PROFILE.startswith("resnet"):
        from models.imagenet.resnet import ResNetDCT_Upscaled_Static
        net = ResNetDCT_Upscaled_Static(channels=int(SUBSET), pretrained=False)
    else:
        from unittest.mock import patch
        from models.imagenet import mobilenetv2 as mobile
        # The official constructor unconditionally loads RGB initialization.
        # Bypass only that load: the complete DCT state is strictly restored below.
        with patch.object(mobile, "mobilenetv2",
                          side_effect=lambda pretrained, **kw: mobile.MobileNetV2(**kw)):
            net = mobile.MobileNetV2DCT_Subset_woinp(channels=int(SUBSET))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval().requires_grad_(False)
    return net


def organize(components):
    if [tuple(c.shape) for c in components] != [(GRID, GRID, 64)] * 3:
        raise ValueError(f"DCTNet requires three {GRID}x{GRID} grids with 64 natural-order frequencies")
    planes = [torch.as_tensor(c).permute(2, 0, 1).float()[indices]
              for c, indices in zip(components, INDICES)]
    return (torch.cat(planes) - MEAN) / STD


class Reference:
    def __init__(self):
        import cv2
        from turbojpeg import TurboJPEG
        from datasets import cvtransforms as T
        cv2.setNumThreads(1)
        self.encoder = TurboJPEG(str(Path(sys.prefix) / "lib/libturbojpeg.so"))
        self.geometry = T.Compose([T.Resize(RESIZE), T.CenterCrop(CROP), T.Upscale(2)])
        self.dct = T.TransformUpscaledDCT.__new__(T.TransformUpscaledDCT)
        self.dct.jpeg_encoder = self.encoder
        self.tensor = T.Compose([T.ToTensorDCT(), T.SubsetDCT(SUBSET), T.Aggregate(),
                                T.NormalizeDCT(train_upscaled_static_mean,
                                               train_upscaled_static_std, channels=SUBSET)])

    def coefficients(self, path, verify=False):
        import cv2
        from jpeg2dct.numpy import loads
        image = cv2.imread(str(path))  # official loader is BGR
        if image is None:
            raise ValueError(f"cannot decode source image: {path}")
        branches = self.geometry(image)
        encoded = [self.encoder.encode(b, quality=100, jpeg_subsample=2) for b in branches]
        quantized = [loads(b, normalized=False) for b in encoded]
        tables = []
        for blob in encoded:
            with Image.open(io.BytesIO(blob)) as jpeg:
                tables.append(jpeg.quantization)
        q = [quantized[0][0], quantized[1][1], quantized[1][2]]
        qt = np.array([tables[0][0], tables[1][1], tables[1][1]], dtype=np.uint16)
        dequantized = [coeff.astype(np.int32) * table for coeff, table in zip(q, qt)]
        if verify:
            normalized = [loads(b, normalized=True) for b in encoded]
            expected = [normalized[0][0], normalized[1][1], normalized[1][2]]
            for actual, target in zip(dequantized, expected):
                np.testing.assert_array_equal(actual, target)
        return np.stack(q), qt, dequantized

    def __call__(self, path):
        import cv2
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot decode source image: {path}")
        components = self.dct(self.geometry(image))
        result, _, _ = self.tensor(tuple(components))
        return result


def profile():
    return dict(model=MODEL_NAME, checkpoint=str(CHECKPOINT),
                shape=[int(SUBSET), GRID, GRID], component_order=["Y", "Cb", "Cr"],
                frequency_order="natural row-major u*8+v", indices=INDICES,
                mean=MEAN.flatten().tolist(), std=STD.flatten().tolist(),
                stored_dtype="int16 quantized", stored_component_shapes=[[GRID, GRID, 64]] * 3,
                dequantization="multiply component quantization table exactly once",
                jpeg_quality=100, jpeg_subsample=2, source_color="BGR",
                geometry=f"official Resize({RESIZE}), CenterCrop({CROP}), Upscale(2) bilinear; Y from {CROP}, Cb/Cr from {CROP*2}")
