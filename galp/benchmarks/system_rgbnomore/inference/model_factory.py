#!/usr/bin/env python3
"""Model construction shared by the canonical end-to-end benchmark."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def _plainvit(rgbnomore_root: Path):
    root = str(rgbnomore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("models.plainvit")


def build_rgb_model(rgbnomore_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = _plainvit(rgbnomore_root).ViT(
        in_channels=3,
        patch_size=16,
        emb_size=192,
        depth=12,
        n_classes=1000,
        drop_p=0.0,
        device=device,
        dtype=torch.float32,
        num_heads=3,
        head_size=64,
        pixel_space="RGB",
    )
    checkpoint_object = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_object.get("model_state_dict", checkpoint_object))
    model.eval()
    return model


def build_dct_model(rgbnomore_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = _plainvit(rgbnomore_root).ViT(
        in_channels=3,
        patch_size=16,
        emb_size=192,
        depth=12,
        n_classes=1000,
        drop_p=0.0,
        device=device,
        dtype=torch.float32,
        num_heads=3,
        head_size=64,
        pixel_space="DCT",
        ver=1,
        use_subblock=True,
    )
    checkpoint_object = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_object.get("model_state_dict", checkpoint_object))
    model.eval()
    return model
