#!/usr/bin/env python3
"""Model registry for Direct-DCT PLS training workloads.

Every registered model consumes the same model-facing ``(y, cbcr)`` contract.
The registry owns architecture construction and example inputs; storage,
physical PLS scheduling, and the training loop remain model-independent.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch


VITTI_MODEL_ID = "rgbnomore-vitti-dct-224-v1"
SWINV2_T_MODEL_ID = "rgbnomore-swinv2-t-dct-224-v1"
DEFAULT_MODEL_ID = VITTI_MODEL_ID


@dataclass(frozen=True)
class DctInputContract:
    profile_id: str
    y_shape: tuple[int, ...]
    cbcr_shape: tuple[int, ...]
    classes: int = 1000

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "ycbcr-dct-grid-v1",
            "profile_id": self.profile_id,
            "y_shape_without_batch": list(self.y_shape),
            "cbcr_shape_without_batch": list(self.cbcr_shape),
            "classes": self.classes,
        }


@dataclass(frozen=True)
class DctModelSpec:
    model_id: str
    architecture: str
    recipe_name: str
    input_contract: DctInputContract
    configuration: Mapping[str, Any]
    expected_trainable_parameters: int
    source_paths: tuple[str, ...]
    supported_conditions: tuple[str, ...]
    builder: Callable[[Path, torch.device], torch.nn.Module]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "domain": "dct",
            **dict(self.configuration),
            "input_contract": self.input_contract.as_dict(),
            "expected_trainable_parameters": self.expected_trainable_parameters,
            "supported_conditions": list(self.supported_conditions),
        }


_DCT_224 = DctInputContract(
    profile_id="rgbnomore-training-pls-v1",
    y_shape=(1, 28, 28, 8, 8),
    cbcr_shape=(2, 14, 14, 8, 8),
)


def _rgbnomore_module(root: Path, qualified_name: str) -> Any:
    resolved = root.resolve()
    root_text = str(resolved)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        imported = Path(existing.__file__).resolve()
        if resolved not in imported.parents:
            raise RuntimeError(
                f"{qualified_name} already imported from {imported}, expected below {resolved}"
            )
        return existing
    module = importlib.import_module(qualified_name)
    imported = Path(module.__file__).resolve()
    if resolved not in imported.parents:
        raise RuntimeError(
            f"imported {qualified_name} from {imported}, expected below {resolved}"
        )
    return module


def _build_vitti(root: Path, device: torch.device) -> torch.nn.Module:
    # Reuse the established factory so the historical ViT parameter and source
    # checks remain authoritative.
    from training.model_factory import build_model

    return build_model(root, "dct", device)


def _build_swinv2_t(root: Path, device: torch.device) -> torch.nn.Module:
    module = _rgbnomore_module(root, "models.swinv2")
    model = module.SwinTransformerV2(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
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
        # RGB-no-more keeps the DCT conversion matrices as plain tensors rather
        # than registered buffers, so they must be created on the final device.
        device=device,
        pixel_space="dct",
    ).to(device)
    model.train()
    return model


_SPECS = {
    VITTI_MODEL_ID: DctModelSpec(
        model_id=VITTI_MODEL_ID,
        architecture="rgbnomore-vitti-v1",
        recipe_name="rgbnomore-vitti-dct-published-v1",
        input_contract=_DCT_224,
        configuration={
            "pixel_space": "DCT",
            "patch_size": 16,
            "embedding_dimension": 192,
            "layers": 12,
            "attention_heads": 3,
            "head_size": 64,
            "classes": 1000,
            "dropout": 0.0,
            "dct_version": 1,
            "dct_use_subblock": True,
        },
        expected_trainable_parameters=5_642_728,
        source_paths=("models/plainvit.py", "utils/dct_ops.py"),
        supported_conditions=("A0", "A1", "B2", "B6", "N2", "N6"),
        builder=_build_vitti,
    ),
    SWINV2_T_MODEL_ID: DctModelSpec(
        model_id=SWINV2_T_MODEL_ID,
        architecture="rgbnomore-swinv2-t-v1",
        recipe_name="rgbnomore-swinv2-t-dct-224-v1",
        input_contract=_DCT_224,
        configuration={
            "pixel_space": "DCT",
            "image_size": 224,
            "patch_size": 4,
            "embedding_dimension": 96,
            "depths": [2, 2, 6, 2],
            "attention_heads": [3, 6, 12, 24],
            "window_size": 7,
            "mlp_ratio": 4.0,
            "classes": 1000,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "drop_path": 0.2,
            "patch_normalization": True,
            "dct_stem": "grouped-subblock-ycbcr-v1",
        },
        expected_trainable_parameters=28_344_850,
        source_paths=(
            "models/swinv2.py",
            "models/plainvit.py",
            "utils/dct_ops.py",
        ),
        supported_conditions=("B6",),
        builder=_build_swinv2_t,
    ),
}

MODEL_IDS = tuple(_SPECS)


def resolve_model(model_id: str = DEFAULT_MODEL_ID) -> DctModelSpec:
    try:
        return _SPECS[str(model_id)]
    except KeyError as error:
        raise ValueError(
            f"unknown DCT model {model_id!r}; expected one of {MODEL_IDS}"
        ) from error


def model_configuration(model_id: str = DEFAULT_MODEL_ID) -> dict[str, Any]:
    return resolve_model(model_id).as_dict()


def build_model(
    rgbnomore_root: Path,
    model_id: str,
    device: torch.device,
) -> torch.nn.Module:
    spec = resolve_model(model_id)
    model = spec.builder(rgbnomore_root, device)
    actual = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if actual != spec.expected_trainable_parameters:
        raise RuntimeError(
            f"{model_id} trainable parameter regression: expected "
            f"{spec.expected_trainable_parameters}, got {actual}"
        )
    return model


def example_inputs(
    model_id: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("example input batch size must be positive")
    contract = resolve_model(model_id).input_contract
    return (
        torch.zeros((batch_size, *contract.y_shape), dtype=torch.float32, device=device),
        torch.zeros(
            (batch_size, *contract.cbcr_shape), dtype=torch.float32, device=device
        ),
    )


def recipe_for_model(model_id: str) -> str:
    return resolve_model(model_id).recipe_name


def source_provenance(rgbnomore_root: Path, model_id: str) -> list[dict[str, str]]:
    """Hash external model sources so a frozen run cannot silently change them."""

    root = rgbnomore_root.resolve()
    rows: list[dict[str, str]] = []
    for relative_path in resolve_model(model_id).source_paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({"path": relative_path, "sha256": digest.hexdigest()})
    return rows


__all__ = [
    "DEFAULT_MODEL_ID",
    "MODEL_IDS",
    "SWINV2_T_MODEL_ID",
    "VITTI_MODEL_ID",
    "DctInputContract",
    "DctModelSpec",
    "build_model",
    "example_inputs",
    "model_configuration",
    "recipe_for_model",
    "resolve_model",
    "source_provenance",
]
