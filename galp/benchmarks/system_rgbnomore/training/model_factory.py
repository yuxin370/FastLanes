#!/usr/bin/env python3
"""Formal RGB-no-more ViT-Ti model construction and reset state handling."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.artifacts import sha256_file, tensor_state_sha256


MODEL_ARCHITECTURE = "rgbnomore-vitti-v1"
EXPECTED_PARAMETER_COUNTS = {"rgb": 5_716_456, "dct": 5_642_728}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = []
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_state_artifact(state: dict[str, Any]) -> dict[str, Any]:
    encoded = base64.b64encode(pickle.dumps(state, protocol=5)).decode("ascii")
    return {
        "encoding": "pickle-v5-base64",
        "sha256": rng_state_sha256(state),
        "data": encoded,
    }


def rng_state_sha256(state: dict[str, Any]) -> str:
    """Hash RNG values without pickle tensor-storage identity metadata."""

    digest = hashlib.sha256()
    digest.update(pickle.dumps(state["python"], protocol=5))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode("ascii"))
    numpy_values = np.asarray(numpy_state[1]).copy(order="C")
    digest.update(str(numpy_values.dtype).encode("ascii"))
    digest.update(numpy_values.tobytes())
    digest.update(pickle.dumps(tuple(numpy_state[2:]), protocol=5))
    torch_cpu = state["torch_cpu"].detach().cpu().contiguous()
    digest.update(torch_cpu.numpy().tobytes())
    cuda_states = state.get("torch_cuda", [])
    digest.update(len(cuda_states).to_bytes(4, "little"))
    for value in cuda_states:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state_from_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    raw = base64.b64decode(payload["data"])
    state = pickle.loads(raw)
    if rng_state_sha256(state) != payload["sha256"]:
        raise ValueError("RNG state artifact hash mismatch")
    return state


def _plainvit(rgbnomore_root: Path):
    root = rgbnomore_root.resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    existing = sys.modules.get("models.plainvit")
    if existing is not None:
        imported = Path(existing.__file__).resolve()
        if root not in imported.parents:
            raise RuntimeError(f"models.plainvit already imported from {imported}, expected below {root}")
        return existing
    module = importlib.import_module("models.plainvit")
    imported = Path(module.__file__).resolve()
    if root not in imported.parents:
        raise RuntimeError(f"imported models.plainvit from {imported}, expected below {root}")
    return module


def model_configuration(domain: str) -> dict[str, Any]:
    if domain not in ("rgb", "dct"):
        raise ValueError(f"invalid model domain: {domain}")
    return {
        "architecture": MODEL_ARCHITECTURE,
        "domain": domain,
        "pixel_space": domain.upper(),
        "patch_size": 16,
        "embedding_dimension": 192,
        "layers": 12,
        "attention_heads": 3,
        "head_size": 64,
        "classes": 1000,
        "dropout": 0.0,
        "precision": "fp32",
        "dct_ver": 1 if domain == "dct" else None,
        "dct_use_subblock": True if domain == "dct" else None,
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNTS[domain],
    }


def build_model(rgbnomore_root: Path, domain: str, device: torch.device) -> torch.nn.Module:
    config = model_configuration(domain)
    arguments: dict[str, Any] = {
        "in_channels": 3,
        "patch_size": config["patch_size"],
        "emb_size": config["embedding_dimension"],
        "depth": config["layers"],
        "n_classes": config["classes"],
        "drop_p": config["dropout"],
        "device": device,
        "dtype": torch.float32,
        "num_heads": config["attention_heads"],
        "head_size": config["head_size"],
        "pixel_space": "RGB" if domain == "rgb" else "DCT",
    }
    if domain == "dct":
        arguments.update(ver=1, use_subblock=True)
    model = _plainvit(rgbnomore_root).ViT(**arguments)
    model.train()
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected = EXPECTED_PARAMETER_COUNTS[domain]
    if actual != expected:
        raise RuntimeError(f"{domain} trainable parameter regression: expected {expected}, got {actual}")
    return model


def clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in state.items()
    }


def _model_state_from_checkpoint(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a mapping")
    state = payload.get("model_state_dict", payload.get("model"))
    if state is None and payload and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dict")
    return state


def initialize_model(
    model: torch.nn.Module,
    *,
    init_mode: str,
    checkpoint: Path | None,
    domain: str,
) -> dict[str, Any]:
    mode = init_mode.lower()
    provenance: dict[str, Any] = {"mode": mode, "domain": domain, "checkpoint": None}
    if mode == "random":
        if checkpoint is not None:
            raise ValueError("random init mode does not accept an initialization checkpoint")
        provenance["convergence_classification"] = "from_scratch_short_convergence"
    elif mode in ("weights", "full-checkpoint"):
        if checkpoint is None:
            raise ValueError(f"{mode} init mode requires a {domain} initialization checkpoint")
        checkpoint = checkpoint.resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            checkpoint_architecture = payload.get("model_architecture")
            checkpoint_domain = payload.get("model_domain")
            if checkpoint_architecture is not None and checkpoint_architecture != MODEL_ARCHITECTURE:
                raise ValueError(
                    f"checkpoint architecture is {checkpoint_architecture!r}, expected {MODEL_ARCHITECTURE!r}"
                )
            if checkpoint_domain is not None and checkpoint_domain != domain:
                raise ValueError(
                    f"checkpoint domain is {checkpoint_domain!r}, expected {domain!r}"
                )
        if mode == "full-checkpoint":
            required = {
                "model_architecture",
                "model_domain",
                "model_configuration",
                "optimizer_configuration",
                "scheduler_configuration",
                "scheduler_total_steps",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "global_step",
                "epoch",
                "rng_state",
                "augmentation_state",
                "sample_order_cursor",
            }
            missing = sorted(required - set(payload)) if isinstance(payload, dict) else sorted(required)
            if missing:
                raise ValueError(f"full checkpoint is missing required fields: {missing}")
            if payload["model_architecture"] != MODEL_ARCHITECTURE:
                raise ValueError("full checkpoint architecture does not match the formal model")
            if payload["model_domain"] != domain:
                raise ValueError("full checkpoint domain does not match the requested model domain")
            if payload["model_configuration"] != model_configuration(domain):
                raise ValueError("full checkpoint model configuration does not match exactly")
        state = _model_state_from_checkpoint(payload)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"strict state load unexpectedly returned missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        provenance["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "model_architecture": MODEL_ARCHITECTURE,
            "architecture_verification": "strict_parameter_name_shape_dtype_signature",
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        provenance["convergence_classification"] = "fine_tuning" if mode == "weights" else "resumed_training"
        if mode == "full-checkpoint":
            # The runner consumes this private, tensor-bearing field before
            # serializing provenance.  Keeping it here avoids loading a large
            # checkpoint twice while ensuring it can never enter contract JSON.
            provenance["_full_checkpoint_payload"] = payload
    else:
        raise ValueError(f"unsupported init mode: {init_mode}")
    state = clone_state_dict(model.state_dict())
    provenance["initial_state_sha256"] = tensor_state_sha256(state)
    provenance["actual_trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return provenance


def reset_training_state(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    initial: dict[str, Any],
) -> None:
    model.load_state_dict(initial["model"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(initial["scheduler"]))
    if scaler is not None and initial.get("scaler") is not None:
        scaler.load_state_dict(copy.deepcopy(initial["scaler"]))
    restore_rng_state(initial["rng"])


def capture_training_state(model: Any, optimizer: Any, scheduler: Any, scaler: Any = None) -> dict[str, Any]:
    return {
        "model": clone_state_dict(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        "rng": capture_rng_state(),
    }
