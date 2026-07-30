#!/usr/bin/env python3
"""Numerical correctness and train-step measurement helpers."""

from __future__ import annotations

import math
import os
import statistics
from typing import Any, Iterable, Sequence


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / mean


def tensor_is_finite(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value).all().item())


def nested_tensors_finite(value: Any) -> bool:
    import torch

    if torch.is_tensor(value):
        return tensor_is_finite(value)
    if isinstance(value, dict):
        return all(nested_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_tensors_finite(item) for item in value)
    return True


def gradient_summary(model: Any, *, include_per_parameter: bool = True) -> dict[str, Any]:
    import torch

    finite = True
    nonzero = False
    squared_norm = 0.0
    tensors = 0
    elements = 0
    maximum = 0.0
    per_parameter: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensors += 1
        elements += gradient.numel()
        is_finite = bool(torch.isfinite(gradient).all().item())
        finite = finite and is_finite
        norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        max_abs = float(gradient.detach().abs().max().item()) if gradient.numel() else 0.0
        nonzero = nonzero or norm > 0.0
        squared_norm += norm * norm
        maximum = max(maximum, max_abs)
        if include_per_parameter:
            per_parameter[name] = {"finite": is_finite, "l2_norm": norm, "max_abs": max_abs}
    return {
        "finite": finite,
        "nonzero": nonzero,
        "tensor_count": tensors,
        "element_count": elements,
        "global_l2_norm": math.sqrt(squared_norm),
        "max_abs": maximum,
        "per_parameter": per_parameter,
    }


def parameter_update_summary(before: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch

    changed = 0
    squared_norm = 0.0
    maximum = 0.0
    per_parameter: dict[str, dict[str, float | bool]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter.detach().cpu() - before[name].detach().cpu()
        norm = float(torch.linalg.vector_norm(delta.float()).item())
        max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
        did_change = bool(torch.count_nonzero(delta).item())
        changed += int(did_change)
        squared_norm += norm * norm
        maximum = max(maximum, max_abs)
        per_parameter[name] = {"changed": did_change, "l2_norm": norm, "max_abs": max_abs}
    return {
        "changed": changed > 0,
        "changed_parameter_tensors": changed,
        "global_l2_norm": math.sqrt(squared_norm),
        "max_abs": maximum,
        "per_parameter": per_parameter,
    }


def process_memory() -> dict[str, int | None]:
    rss = None
    peak = None
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"rss_bytes": rss, "peak_rss_bytes": peak}


def topk_accuracy(logits: Any, labels: Any, topk: Iterable[int] = (1, 5)) -> dict[str, float]:
    maximum = max(topk)
    _, predicted = logits.topk(maximum, dim=1, largest=True, sorted=True)
    correct = predicted.eq(labels.view(-1, 1))
    return {f"top{value}": float(correct[:, :value].any(dim=1).float().mean().item()) for value in topk}
