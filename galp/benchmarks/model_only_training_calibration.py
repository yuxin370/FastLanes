#!/usr/bin/env python3
"""Shared fixed-input model-only training calibration.

Inputs and labels must already reside on the target device.  The measured
scope is forward, loss, backward, clipping, optimizer, independent weight
decay, scheduler, and the selected shared audit policy.  It intentionally
contains no reader, decode, augmentation, or H2D work.
"""

from __future__ import annotations

from contextlib import nullcontext
import time
from typing import Any, Mapping, Sequence

import torch

from galp.benchmarks.training_audit_policy import (
    decision_for_next_update,
    validate_audit_policy,
)


MODEL_ONLY_SCHEMA = "galp-model-only-training-calibration-v1"


def run_model_only_calibration(
    *,
    domain: str,
    execution_model: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    inputs: Sequence[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    audit_policy: Mapping[str, Any],
    microbatch_images: int,
    gradient_accumulation: int,
    gradient_clipping_norm: float,
    warmup_updates: int,
    measured_updates: int,
    precision: str = "fp32",
) -> dict[str, Any]:
    policy = validate_audit_policy(audit_policy)
    if domain not in {"rgb", "dct"}:
        raise ValueError("model-only domain must be 'rgb' or 'dct'")
    if int(labels.shape[0]) != int(microbatch_images):
        raise ValueError("model-only labels do not match the fixed microbatch")
    if any(int(value.shape[0]) != int(microbatch_images) for value in inputs):
        raise ValueError("model-only inputs do not match the fixed microbatch")
    if warmup_updates < 0 or measured_updates <= 0 or gradient_accumulation <= 0:
        raise ValueError("model-only update counts/accumulation are invalid")
    if precision not in {"fp32", "bf16-autocast"}:
        raise ValueError("model-only precision must be 'fp32' or 'bf16-autocast'")

    def autocast_context() -> Any:
        if precision == "fp32":
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    deferred_finite = torch.ones((), dtype=torch.bool, device=device)
    audit_seconds = 0.0
    measured_seconds = 0.0
    measured_microbatches = 0
    completed_updates = 0

    def one_update(*, measured: bool) -> None:
        nonlocal audit_seconds, measured_microbatches, completed_updates
        optimizer.zero_grad(set_to_none=True)
        learning_rate = scheduler.prepare_next_update()
        decision = decision_for_next_update(policy, completed_updates)
        for _microbatch in range(gradient_accumulation):
            with autocast_context():
                logits = execution_model(*inputs)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            audit_started = time.perf_counter()
            if decision.synchronous_loss_logits:
                if not bool(torch.isfinite(loss).item()) or not bool(
                    torch.isfinite(logits).all().item()
                ):
                    raise FloatingPointError("non-finite model-only loss/logits")
                # Preserve strict mode's historical scalar materialization.
                float(loss.detach().item())
            else:
                nonlocal_deferred[0] = (
                    nonlocal_deferred[0]
                    & torch.isfinite(loss)
                    & torch.isfinite(logits).all()
                )
            audit_seconds += time.perf_counter() - audit_started
            (loss / gradient_accumulation).backward()
            if measured:
                measured_microbatches += 1
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise FloatingPointError("model-only calibration produced no gradients")
        audit_started = time.perf_counter()
        if decision.synchronous_gradients:
            if not all(
                bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            ):
                raise FloatingPointError("non-finite model-only gradients")
        else:
            for gradient in gradients:
                nonlocal_deferred[0] = (
                    nonlocal_deferred[0] & torch.isfinite(gradient).all()
                )
        audit_seconds += time.perf_counter() - audit_started
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(gradient_clipping_norm)
        )
        optimizer.step()
        weight_decayer.step(learning_rate)
        scheduler.complete_update()
        completed_updates += 1
        if decision.synchronous_parameters:
            audit_started = time.perf_counter()
            if not all(
                bool(torch.isfinite(parameter).all().item())
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite model-only parameters")
            audit_seconds += time.perf_counter() - audit_started

    # A list gives the nested function a mutable cell without exposing a
    # controller object in the hot loop.
    nonlocal_deferred = [deferred_finite]
    for _ in range(warmup_updates):
        one_update(measured=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_started = time.perf_counter()
    for _ in range(measured_updates):
        one_update(measured=True)
    if (
        policy["audit_mode"] == "runtime-first-100"
        and completed_updates > int(policy["strict_update_count"])
    ):
        audit_started = time.perf_counter()
        deferred = nonlocal_deferred[0]
        for parameter in model.parameters():
            deferred = deferred & torch.isfinite(parameter).all()
        if not bool(deferred.item()):
            raise FloatingPointError("deferred model-only finite audit failed")
        audit_seconds += time.perf_counter() - audit_started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_seconds = time.perf_counter() - measured_started

    measured_images = measured_microbatches * int(microbatch_images)
    return {
        "schema_version": MODEL_ONLY_SCHEMA,
        "domain": domain,
        "input_residency": "fixed-pre-resident-device-tensors",
        "excluded_work": ["data-read", "decode", "augmentation", "H2D"],
        "included_work": [
            "forward",
            "cross-entropy-loss",
            "backward",
            "gradient-clipping",
            "optimizer",
            "independent-weight-decay",
            "scheduler",
            "audit-policy",
        ],
        "precision": precision,
        "microbatch_images": int(microbatch_images),
        "gradient_accumulation": int(gradient_accumulation),
        "warmup_optimizer_updates": int(warmup_updates),
        "measured_optimizer_updates": int(measured_updates),
        "measured_microbatches": measured_microbatches,
        "measured_images": measured_images,
        "seconds": measured_seconds,
        "images_per_second": measured_images / measured_seconds,
        "milliseconds_per_microbatch": (
            measured_seconds * 1000.0 / measured_microbatches
        ),
        "audit_seconds_including_warmup": audit_seconds,
        "audit_policy_hash": policy["audit_policy_hash"],
        "interpretation": (
            "explanatory model calibration only; do not subtract it from E2E time "
            "because model and pipeline work can overlap"
        ),
    }
