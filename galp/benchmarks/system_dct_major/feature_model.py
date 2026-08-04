#!/usr/bin/env python3
"""RGB-no-more-compatible evaluation and feature-extraction models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch

from common import REPO_ROOT


MODEL_FACTORY = REPO_ROOT / "galp/benchmarks/system_rgbnomore/inference/model_factory.py"


def _load_model_factory() -> Any:
    spec = importlib.util.spec_from_file_location("dct_major_rgbnomore_model_factory", MODEL_FACTORY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RGB-no-more model factory: {MODEL_FACTORY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeatureExtractor(torch.nn.Module):
    """Expose a stable ViT feature tap without changing checkpoint weights."""

    def __init__(self, model: torch.nn.Module, *, domain: str, stage: str = "penultimate") -> None:
        super().__init__()
        if domain not in {"rgb", "dct"}:
            raise ValueError(f"unsupported model domain: {domain}")
        if stage not in {"pooled", "penultimate"}:
            raise ValueError(f"unsupported feature stage: {stage}")
        self.model = model
        self.domain = domain
        self.stage = stage

    def forward(self, first: torch.Tensor, second: torch.Tensor | None = None) -> torch.Tensor:
        if self.domain == "rgb":
            hidden = self.model.patchembed(first)
        else:
            if second is None:
                raise ValueError("DCT feature extraction requires Y and CbCr tensors")
            hidden = self.model.patchembed(first, second)
        hidden = self.model.encoder(hidden)
        for name, layer in self.model.classhead.named_children():
            hidden = layer(hidden)
            if self.stage == "pooled" and name == "ch_gap":
                break
            if self.stage == "penultimate" and name == "ch_tanh":
                break
        if hidden.ndim != 2 or hidden.shape[1] != 192:
            raise RuntimeError(f"expected [B,192] features at {self.stage}, got {tuple(hidden.shape)}")
        return hidden


def build_workload_model(
    *,
    domain: str,
    workload: str,
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
    feature_stage: str = "penultimate",
) -> torch.nn.Module:
    factory = _load_model_factory()
    if domain == "rgb":
        model = factory.build_rgb_model(rgbnomore_root, checkpoint, device)
    elif domain == "dct":
        model = factory.build_dct_model(rgbnomore_root, checkpoint, device)
    else:
        raise ValueError(f"unsupported model domain: {domain}")
    if workload == "evaluation":
        return model.eval()
    if workload == "feature-extraction":
        return FeatureExtractor(model, domain=domain, stage=feature_stage).eval()
    raise ValueError(f"unsupported workload: {workload}")


def expected_output_width(workload: str) -> int:
    if workload == "evaluation":
        return 1000
    if workload == "feature-extraction":
        return 192
    raise ValueError(f"unsupported workload: {workload}")

