#!/usr/bin/env python3
"""Reproducible optimizer, parameter-group, and scheduler construction."""

from __future__ import annotations

from typing import Any


def parameter_groups(model: Any, *, learning_rate: float, weight_decay: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decay: list[Any] = []
    no_decay: list[Any] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1]
        excluded = parameter.ndim <= 1 or leaf == "bias"
        if excluded:
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    groups = [
        {"params": decay, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
    ]
    records = [
        {
            "stable_id": "decay",
            "parameter_names": decay_names,
            "parameter_count": sum(value.numel() for value in decay),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "excludes_bias": True,
            "excludes_normalization": True,
            "independent_weight_decayer": False,
        },
        {
            "stable_id": "no_decay",
            "parameter_names": no_decay_names,
            "parameter_count": sum(value.numel() for value in no_decay),
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "contains_bias_or_normalization": True,
            "independent_weight_decayer": False,
        },
    ]
    return groups, records


def build_optimizer(model: Any, config: dict[str, Any]):
    import torch

    groups, records = parameter_groups(
        model,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    optimizer_name = str(config["type"]).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            groups,
            betas=tuple(float(value) for value in config["betas"]),
            eps=float(config["epsilon"]),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            groups,
            momentum=float(config["momentum"]),
            nesterov=bool(config["nesterov"]),
        )
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    return optimizer, records


def build_scheduler(optimizer: Any, config: dict[str, Any], *, total_steps: int):
    import torch

    scheduler_name = str(config["type"]).lower()
    warmup = int(config["warmup_steps"])

    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if scheduler_name == "constant":
            return 1.0
        if scheduler_name == "cosine":
            import math

            progress = (step - warmup) / max(1, total_steps - warmup)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unsupported scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def resolved_optimizer_config(args: Any) -> dict[str, Any]:
    return {
        "type": str(args.optimizer).lower(),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "momentum": float(args.momentum),
        "nesterov": bool(args.nesterov),
        "betas": [float(value) for value in args.betas],
        "epsilon": float(args.eps),
        "weight_decay_exclusion": "bias_and_parameters_with_ndim_le_1",
        "gradient_clipping_norm": None if args.gradient_clipping is None else float(args.gradient_clipping),
        "loss_function": "cross_entropy",
        "label_smoothing": float(args.label_smoothing),
    }


def resolved_scheduler_config(args: Any) -> dict[str, Any]:
    return {
        "type": str(args.scheduler).lower(),
        "warmup_steps": int(args.scheduler_warmup_steps),
        "step_unit": "optimizer_step",
        "minimum_learning_rate_ratio": 0.0 if str(args.scheduler).lower() == "cosine" else 1.0,
    }
