#!/usr/bin/env python3
"""Profile a bounded native physical-PLS training window with fine NVTX ranges.

This is a profiling-only wrapper.  It leaves the registered training runtime
sources unchanged, restores an epoch-boundary checkpoint through the normal
runner, replaces only the epoch loop with an NVTX-instrumented equivalent, and
exits immediately after the requested closed-pool capture.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from galp.benchmarks.training_audit_policy import (
    TrainingAuditPolicy,
    TrainingAuditState,
)
from galp.benchmarks.system_dct_major.training_pls import train
class ProfileCaptureComplete(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__("native PLS profiling capture completed")
        self.payload = dict(payload)


@contextlib.contextmanager
def _range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _profiling_contract_validation(
    path: Path,
    *,
    condition_id: str,
    seed: int,
    recipe_hash: str,
    layout_hash: str,
    execution_backend: str,
    physical_galp_manifest: Path | None,
    premixed_mapping_csv: Path | None,
    expected_mapping_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate immutable workload identity while allowing profiling wrappers."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "condition_id": condition_id,
        "training_seed": seed,
        "recipe_hash": recipe_hash,
        "layout_hash": layout_hash,
        "execution_mode": "native_physical_pls",
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected or execution_backend != train.NATIVE_PHYSICAL_BACKEND:
        raise ValueError(
            f"profiling workload identity differs: expected={expected}, observed={observed}"
        )
    if physical_galp_manifest is None or premixed_mapping_csv is None:
        raise ValueError("profiling physical execution arguments are incomplete")
    physical = payload.get("physical_execution", {})
    expected_physical = {
        "physical_galp_manifest": str(physical_galp_manifest.resolve()),
        "physical_galp_manifest_sha256": train.sha256_file(
            physical_galp_manifest.resolve()
        ),
        "premixed_mapping_csv": str(premixed_mapping_csv.resolve()),
        "premixed_mapping_sha256": expected_mapping_sha256,
    }
    observed_physical = {
        key: physical.get(key) for key in expected_physical
    }
    if observed_physical != expected_physical:
        raise ValueError(
            "profiling physical-layout identity differs: "
            f"expected={expected_physical}, observed={observed_physical}"
        )
    return payload, {
        "schema_version": "galp-pls-profile-contract-validation-v1",
        "scientific_result": False,
        "workload_identity_matches": True,
        "runtime_source_hash_gate": "profiling-only wrapper override",
    }


def _make_profiled_epoch(
    *,
    warmup_pools: int,
    capture_pools: int,
    capture_json: Path,
):
    capture_end_pool = warmup_pools + capture_pools

    def profiled_epoch(
        *,
        pipeline: Any,
        execution_model: torch.nn.Module,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        weight_decayer: Any,
        scheduler: Any,
        device: torch.device,
        recipe: Mapping[str, Any],
        metrics: Any,
        loader_totals: dict[str, float],
        integration_checks: dict[str, Any],
        condition_id: str,
        seed: int,
        epoch: int,
        expected_sample_count: int,
        global_update: int,
        processed_images: int,
        loss_since_log: float,
        samples_since_log: int,
        last_logged_update: int,
        integration_check_first_100: bool,
        audit_policy: TrainingAuditPolicy,
    ) -> dict[str, Any]:
        del loader_totals, integration_checks, integration_check_first_100
        epoch_loss_sum = 0.0
        epoch_samples = 0
        epoch_microbatches = 0
        epoch_updates = 0
        pool_count = 0
        coverage = np.zeros(expected_sample_count, dtype=np.bool_)
        order_hash = hashlib.sha256()
        microbatch_images = int(recipe["training"]["physical_microbatch"])
        accumulation = int(recipe["training"]["gradient_accumulation"])
        audit = TrainingAuditState(
            audit_policy,
            completed_updates=global_update,
            device=device,
        )
        pipeline.start_epoch(epoch)
        profiling_active = False
        capture_started_wall: float | None = None
        captured_images = 0
        captured_microbatches = 0
        captured_updates = 0
        capture_prefetch_start: dict[str, Any] | None = None
        capture_audit_start: dict[str, int] | None = None
        capture_execution_totals: dict[str, int | float] = {}

        while pipeline.has_next_pool:
            if pool_count == warmup_pools:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                capture_prefetch_start = pipeline.prefetch_stats
                capture_audit_start = audit.counters.as_dict()
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(
                    f"profile-galp-native-b6-pools_{warmup_pools}_{capture_end_pool - 1}"
                )
                profiling_active = True
                capture_started_wall = time.perf_counter()

            pool_context = (
                _range(f"galp.pool.{pool_count}")
                if profiling_active
                else contextlib.nullcontext()
            )
            with pool_context:
                load_context = (
                    _range("galp.pool.load")
                    if profiling_active
                    else contextlib.nullcontext()
                )
                with load_context:
                    pool = pipeline.next_pool()
                pool_microbatches = int(pool.microbatch_count)
                pool_images = int(pool.image_count)
                pool_seen_images = 0

                for window_begin in range(0, pool_microbatches, accumulation):
                    window_microbatches = min(
                        accumulation, pool_microbatches - window_begin
                    )
                    window_sample_count = min(
                        window_microbatches * microbatch_images,
                        pool_images - window_begin * microbatch_images,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    learning_rate = scheduler.prepare_next_update()
                    update_loss_sum = 0.0

                    for _local_index in range(window_microbatches):
                        loader_context = (
                            _range("training.loader.next_batch")
                            if profiling_active
                            else contextlib.nullcontext()
                        )
                        with loader_context:
                            native_batch = next(pool)
                        handoff_context = (
                            _range("training.input_handoff")
                            if profiling_active
                            else contextlib.nullcontext()
                        )
                        with handoff_context:
                            y, cbcr, targets = native_batch.tensors
                            image_ids = native_batch.global_image_ids
                            labels = native_batch.labels
                            batch_size = len(image_ids)
                        audit_context = (
                            _range("training.batch.audit")
                            if profiling_active
                            else contextlib.nullcontext()
                        )
                        with audit_context:
                            if (
                                batch_size <= 0
                                or batch_size != int(y.shape[0])
                                or batch_size != int(cbcr.shape[0])
                                or batch_size != int(targets.shape[0])
                                or batch_size != len(labels)
                            ):
                                raise RuntimeError(
                                    "native PLS tensor/identity cardinalities differ"
                                )
                            for image_id in image_ids:
                                if image_id < 0 or image_id >= expected_sample_count:
                                    raise RuntimeError(
                                        f"native PLS emitted invalid image ID {image_id}"
                                    )
                                if coverage[image_id]:
                                    raise RuntimeError(
                                        f"native PLS emitted duplicate image ID {image_id}"
                                    )
                                coverage[image_id] = True
                                order_hash.update(int(image_id).to_bytes(8, "little"))
                        forward_context = (
                            _range("training.model.forward")
                            if profiling_active
                            else contextlib.nullcontext()
                        )
                        with forward_context:
                            logits = execution_model(y, cbcr)
                        loss_context = (
                            _range("training.loss")
                            if profiling_active
                            else contextlib.nullcontext()
                        )
                        with loss_context:
                            loss = torch.nn.functional.cross_entropy(logits, targets)
                            try:
                                audited_loss_sum = audit.observe(
                                    loss, logits, batch_size
                                )
                            except FloatingPointError as error:
                                raise FloatingPointError(
                                    "non-finite native PLS loss/logits"
                                ) from error
                        backward_context = (
                            _range("training.model.backward")
                            if profiling_active
                            else contextlib.nullcontext()
                        )
                        with backward_context:
                            (loss * (batch_size / window_sample_count)).backward()
                        update_loss_sum += audited_loss_sum
                        pool_seen_images += batch_size
                        epoch_microbatches += 1
                        if profiling_active:
                            captured_images += batch_size
                            captured_microbatches += 1
                        del native_batch, y, cbcr, targets, logits, loss

                    optimizer_context = (
                        _range("training.optimizer")
                        if profiling_active
                        else contextlib.nullcontext()
                    )
                    with optimizer_context:
                        with (
                            _range("training.optimizer.clip_grad")
                            if profiling_active
                            else contextlib.nullcontext()
                        ):
                            try:
                                audit.check_gradients_and_clip(
                                    model.parameters(),
                                    max_norm=float(
                                        recipe["optimizer"]["gradient_clipping_norm"]
                                    ),
                                )
                            except FloatingPointError as error:
                                raise FloatingPointError(
                                    "non-finite gradients"
                                ) from error
                        with (
                            _range("training.optimizer.adamw")
                            if profiling_active
                            else contextlib.nullcontext()
                        ):
                            optimizer.step()
                        with (
                            _range("training.optimizer.weight_decay")
                            if profiling_active
                            else contextlib.nullcontext()
                        ):
                            weight_decayer.step(learning_rate)
                        scheduler.complete_update()
                    global_update += 1
                    audit.complete_update()
                    epoch_updates += 1
                    processed_images += window_sample_count
                    epoch_samples += window_sample_count
                    epoch_loss_sum += update_loss_sum
                    loss_since_log += update_loss_sum
                    samples_since_log += window_sample_count
                    if profiling_active:
                        captured_updates += 1
                    if global_update % int(
                        recipe["logging"]["train_loss_every_optimizer_updates"]
                    ) == 0:
                        deferred_loss, _deferred_samples = audit.read_deferred_loss()
                        epoch_loss_sum += deferred_loss
                        loss_since_log += deferred_loss
                        metrics.append(
                            {
                                "record_type": "train",
                                "scope": "optimizer-window",
                                "condition": condition_id,
                                "seed": seed,
                                "epoch": epoch + 1,
                                "optimizer_update": global_update,
                                "processed_images": processed_images,
                                "train_loss": loss_since_log / samples_since_log,
                                "learning_rate": learning_rate,
                                "window_optimizer_updates": global_update
                                - last_logged_update,
                                "window_samples": samples_since_log,
                                "execution_backend": train.NATIVE_PHYSICAL_BACKEND,
                            }
                        )
                        loss_since_log = 0.0
                        samples_since_log = 0
                        last_logged_update = global_update

                if pool_seen_images != pool_images:
                    raise RuntimeError(
                        f"native PLS pool consumed {pool_seen_images}/{pool_images} images"
                    )
                boundary_context = (
                    _range("galp.pool.retire_stats_reclaim")
                    if profiling_active
                    else contextlib.nullcontext()
                )
                with boundary_context:
                    stats_context = (
                        _range("galp.pool.stats")
                        if profiling_active
                        else contextlib.nullcontext()
                    )
                    with stats_context:
                        _pool_stats = pool.execution_stats
                        if profiling_active:
                            for key, value in _pool_stats.items():
                                if isinstance(value, (int, float)) and not isinstance(value, bool):
                                    capture_execution_totals[key] = (
                                        capture_execution_totals.get(key, 0) + value
                                    )
                    retire_context = (
                        _range("galp.pool.retire")
                        if profiling_active
                        else contextlib.nullcontext()
                    )
                    with retire_context:
                        pool.retire()
                        del pool
                    reclaim_context = (
                        _range("galp.pool.reclaim")
                        if profiling_active
                        else contextlib.nullcontext()
                    )
                    with reclaim_context:
                        pipeline.reclaim_finished_pools()

            pool_count += 1
            if profiling_active and pool_count == capture_end_pool:
                torch.cuda.synchronize(device)
                capture_seconds = time.perf_counter() - float(capture_started_wall)
                capture_prefetch_end = pipeline.prefetch_stats
                resource_probe = getattr(
                    getattr(pipeline, "_module", None),
                    "_lifetime_reclaim_stats_for_test",
                    None,
                )
                native_resources = (
                    dict(resource_probe()) if resource_probe is not None else None
                )
                capture_prefetch = {
                    key: (
                        float(value)
                        - float((capture_prefetch_start or {}).get(key, 0.0))
                        if isinstance(value, (int, float))
                        and key
                        not in {
                            "context_capacity",
                            "live_context_count",
                            "peak_live_context_count",
                            "context_waiter_count",
                            "peak_context_waiter_count",
                        }
                        else value
                    )
                    for key, value in capture_prefetch_end.items()
                }
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
                pipeline.close()
                payload = {
                    "schema_version": "galp-native-pls-nsys-capture-v1",
                    "state": "capture-complete",
                    "scientific_result": False,
                    "epoch": epoch + 1,
                    "warmup_pools": warmup_pools,
                    "captured_pools": capture_pools,
                    "captured_microbatches": captured_microbatches,
                    "captured_optimizer_updates": captured_updates,
                    "captured_images": captured_images,
                    "capture_wall_seconds_with_instrumentation": capture_seconds,
                    "native_pool_prefetch_start": capture_prefetch_start,
                    "native_pool_prefetch_end": capture_prefetch_end,
                    "native_pool_prefetch": capture_prefetch,
                    "native_execution_totals": capture_execution_totals,
                    "torch_peak_memory": {
                        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    },
                    "native_resource_stats": native_resources,
                    "training_audit": {
                        "policy": audit.policy.as_contract(),
                        "counters": {
                            key: int(value)
                            - int((capture_audit_start or {}).get(key, 0))
                            for key, value in audit.counters.as_dict().items()
                        },
                    },
                    "outer_nvtx": (
                        f"profile-galp-native-b6-pools_{warmup_pools}_"
                        f"{capture_end_pool - 1}"
                    ),
                }
                capture_json.parent.mkdir(parents=True, exist_ok=True)
                capture_json.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise ProfileCaptureComplete(payload)

        raise RuntimeError(
            f"native PLS epoch ended before capture pools {warmup_pools}:"
            f"{capture_end_pool - 1}"
        )

    return profiled_epoch


def _parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-warmup-pools", type=int, default=8)
    parser.add_argument("--profile-pools", type=int, default=4)
    parser.add_argument("--capture-json", type=Path, required=True)
    known, remaining = parser.parse_known_args(argv)
    if known.profile_warmup_pools < 0 or known.profile_pools <= 0:
        raise ValueError("profile pool bounds must be positive")
    return known, remaining


def main(argv: Sequence[str] | None = None) -> int:
    profile, train_argv = _parse_args(argv)
    args = train._parse_args(train_argv)
    train._validate_contract = _profiling_contract_validation
    train._train_native_physical_epoch = _make_profiled_epoch(
        warmup_pools=profile.profile_warmup_pools,
        capture_pools=profile.profile_pools,
        capture_json=profile.capture_json.resolve(),
    )
    try:
        train.run(args)
    except ProfileCaptureComplete as complete:
        print(json.dumps(complete.payload, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
