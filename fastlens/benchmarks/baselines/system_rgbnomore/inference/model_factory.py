#!/usr/bin/env python3
"""Registered model construction for the canonical end-to-end benchmark."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


VITTI_MODEL_ID = "rgbnomore-vitti-224-v1"
SWINV2_T_MODEL_ID = "rgbnomore-swinv2-t-256-window8-v1"
DEFAULT_MODEL_ID = VITTI_MODEL_ID


@dataclass(frozen=True)
class InferenceModelSpec:
    model_id: str
    recipe_id: str
    rgb_architecture: str
    dct_architecture: str
    rgb_checkpoint_name: str
    dct_checkpoint_name: str
    rgb_dataset: str
    dct_dataset: str
    rgb_size: int
    dct_transform: str
    dct_profile_id: str
    y_shape: tuple[int, ...]
    cbcr_shape: tuple[int, ...]
    builder: Callable[[Path, str, torch.device], torch.nn.Module]
    performance_gate_profile_id: str
    e2e_performance_targets: tuple[tuple[str, float | bool], ...]

    @property
    def dct_blocks(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.y_shape[1:3], self.cbcr_shape[1:3])


def _rgbnomore_module(root: Path, qualified_name: str) -> Any:
    resolved = root.resolve()
    root_text = str(resolved)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(qualified_name)
    imported = Path(module.__file__).resolve()
    if resolved not in imported.parents:
        raise RuntimeError(
            f"imported {qualified_name} from {imported}, expected below {resolved}"
        )
    return module


def _build_vitti(root: Path, domain: str, device: torch.device) -> torch.nn.Module:
    model = _rgbnomore_module(root, "models.plainvit").ViT(
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
        pixel_space=domain.upper(),
        **({"ver": 1, "use_subblock": True} if domain == "dct" else {}),
    )
    return model.to(device)


def _build_swinv2_t(root: Path, domain: str, device: torch.device) -> torch.nn.Module:
    model = _rgbnomore_module(root, "models.swinv2").SwinTransformerV2(
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=torch.nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        device=device,
        pixel_space=domain,
    )
    return model.to(device)


_SPECS = {
    VITTI_MODEL_ID: InferenceModelSpec(
        model_id=VITTI_MODEL_ID,
        recipe_id="rgbnomore_imagenet_vitti_300ep_recipe_family",
        rgb_architecture="RGB-no-more ViT-Ti RGB",
        dct_architecture="RGB-no-more JPEG-Ti ViT-Ti DCT",
        rgb_checkpoint_name="imgnetRGBViTTi_ep300_74.1.pth",
        dct_checkpoint_name="imgnetDCTViTTi_ep300_75.1.pth",
        rgb_dataset="imagenet",
        dct_dataset="imagenet_dct",
        rgb_size=224,
        dct_transform="ResizedCenterCrop_DCT(32,28)",
        dct_profile_id="rgbnomore-validation-v1",
        y_shape=(1, 28, 28, 8, 8),
        cbcr_shape=(2, 14, 14, 8, 8),
        builder=_build_vitti,
        performance_gate_profile_id="vitti-4090-e2e-v1",
        e2e_performance_targets=(
            ("minimum_hot_median_to_dali_hot_median_ratio", 1.10),
            ("require_hot_min_above_dali_hot_median", True),
            ("maximum_hot_throughput_cv", 0.05),
            ("planning_median_ms_max", 2.0),
            ("planning_p95_ms_max", 3.0),
            ("device_mapping_median_ms_max", 1.0),
            ("device_mapping_plus_fixed_transform_median_ms_max", 6.5),
        ),
    ),
    SWINV2_T_MODEL_ID: InferenceModelSpec(
        model_id=SWINV2_T_MODEL_ID,
        recipe_id="rgbnomore_imagenet_swinv2_t_256_window8_300ep_recipe_family",
        rgb_architecture="RGB-no-more SwinV2-T RGB (256, window=8)",
        dct_architecture="RGB-no-more SwinV2-T DCT (256, window=8)",
        rgb_checkpoint_name="imgnetSwinRGB_ep300_79.0.pth",
        dct_checkpoint_name="imgnetSwinDCT_ep300_79.4.pth",
        rgb_dataset="imagenet_swin",
        dct_dataset="imagenet_dct_swin",
        rgb_size=256,
        dct_transform="Resize_DCT(32)",
        dct_profile_id="rgbnomore-swinv2-validation-v1",
        y_shape=(1, 32, 32, 8, 8),
        cbcr_shape=(2, 16, 16, 8, 8),
        builder=_build_swinv2_t,
        performance_gate_profile_id="swinv2-t-256-window8-report-only-v1",
        e2e_performance_targets=(),
    ),
}

MODEL_IDS = tuple(_SPECS)


def resolve_model(model_id: str = DEFAULT_MODEL_ID) -> InferenceModelSpec:
    try:
        return _SPECS[model_id]
    except KeyError as error:
        raise ValueError(f"unknown inference model {model_id!r}; expected one of {MODEL_IDS}") from error


def _build_model(
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
    *,
    model_id: str,
    domain: str,
) -> torch.nn.Module:
    model = resolve_model(model_id).builder(rgbnomore_root, domain, device)
    checkpoint_object = torch.load(checkpoint, map_location=device)
    state = checkpoint_object.get("model_state_dict", checkpoint_object)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_rgb_model(
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
    model_id: str = DEFAULT_MODEL_ID,
) -> torch.nn.Module:
    return _build_model(
        rgbnomore_root, checkpoint, device, model_id=model_id, domain="rgb"
    )


def build_dct_model(
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
    model_id: str = DEFAULT_MODEL_ID,
) -> torch.nn.Module:
    return _build_model(
        rgbnomore_root, checkpoint, device, model_id=model_id, domain="dct"
    )


__all__ = [
    "DEFAULT_MODEL_ID",
    "MODEL_IDS",
    "SWINV2_T_MODEL_ID",
    "VITTI_MODEL_ID",
    "InferenceModelSpec",
    "build_dct_model",
    "build_rgb_model",
    "resolve_model",
]
