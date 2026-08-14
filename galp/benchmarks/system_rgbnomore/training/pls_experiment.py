#!/usr/bin/env python3
"""Bridge the PLS scientific schedule into the existing training runner.

This module implements the statistical-emulation phase.  It requests optimizer
mini-batches in the exact pre-registered PLS order and derives one crop per
source physical shard/epoch, while leaving the existing pipeline adapters
unchanged.  Consequently it measures model effects without claiming that the
adapter has already materialized a complete GPU pool; that later runtime phase
has a separate contract.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.augmentation import (
    AugmentationDecision,
    derive_augmentation,
    derive_shard_shared_crop_augmentation,
)
from training.pipeline import TrainingSample
from training.sample_order import SampleIdentity


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from galp.benchmarks.system_dct_major.training_pls.schedule import (  # noqa: E402
    PlsScheduleConfig,
    SampleRecord,
    ScheduleResult,
    ScheduledSample,
    build_schedule,
)
from galp.benchmarks.system_dct_major.training_pls.matrix import (  # noqa: E402
    resolve_condition,
)


@dataclass
class PlsExecutionPlan:
    result: ScheduleResult
    scheduled_batches: list[list[ScheduledSample]]
    batches: list[list[SampleIdentity]]
    dropped_per_epoch: dict[int, int]

    @property
    def summary(self) -> dict[str, Any]:
        return self.result.summary

    def selected_summary(self) -> dict[str, Any]:
        """Compact evidence for the exact optimizer batches used by a run."""

        selected = [value for batch in self.scheduled_batches for value in batch]
        epoch_segments = sorted({(value.epoch, value.segment_id) for value in selected})
        segment_digest = hashlib.sha256(
            "\n".join(f"{epoch}:{segment}" for epoch, segment in epoch_segments).encode(
                "utf-8"
            )
        ).hexdigest()
        return {
            "schedule_summary": self.result.summary,
            "selected_optimizer_batches": len(self.scheduled_batches),
            "selected_samples": len(selected),
            "selected_epochs": sorted({value.epoch for value in selected}),
            "selected_epoch_segment_count": len(epoch_segments),
            "selected_epoch_segments_sha256": segment_digest,
            "selected_first_identity": (
                None if not selected else self.batches[0][0].as_dict()
            ),
            "selected_last_identity": (
                None if not selected else self.batches[-1][-1].as_dict()
            ),
            "implementation_scope": {
                "statistical_schedule_emulation": True,
                "complete_gpu_pool_materialization": False,
                "claim": (
                    "this path measures training effects of the exact closed-wave order "
                    "and shard-shared crops; it does not claim GPU-pool runtime locality"
                ),
            },
            "storage_order_matches_galp_image_ids": self.storage_order_matches_galp_image_ids(),
        }

    def selected_pool_batches(self) -> list[dict[str, Any]]:
        """Return exact closed-pool boundaries for the selected batch window."""

        groups: list[dict[str, Any]] = []
        for batch in self.scheduled_batches:
            keys = {(value.epoch, value.pool_index) for value in batch}
            if len(keys) != 1:
                raise RuntimeError("an optimizer batch crosses a closed PLS pool boundary")
            epoch, pool_index = next(iter(keys))
            if groups and (groups[-1]["epoch"], groups[-1]["pool_index"]) == (
                epoch,
                pool_index,
            ):
                groups[-1]["optimizer_batches"] += 1
                groups[-1]["sample_count"] += len(batch)
            else:
                groups.append(
                    {
                        "epoch": epoch,
                        "pool_index": pool_index,
                        "optimizer_batches": 1,
                        "sample_count": len(batch),
                    }
                )
        return groups

    def is_pool_batch_boundary(self, batch_count: int) -> bool:
        if batch_count <= 0:
            return False
        cumulative = 0
        for group in self.selected_pool_batches():
            cumulative += int(group["optimizer_batches"])
            if cumulative == batch_count:
                return True
            if cumulative > batch_count:
                return False
        return False

    def storage_order_matches_galp_image_ids(self) -> bool:
        ordered_ids = [
            sample.galp_image_id
            for segment in self.result.segments
            for sample in segment.samples
        ]
        return ordered_ids == list(range(len(ordered_ids)))


def condition_config(condition_id: str) -> dict[str, Any]:
    """Translate the core condition names for the legacy step-based runner.

    The formal 300-epoch experiment uses ``training_pls.train``.  Keeping this
    translation prevents the older diagnostic runner from receiving the new
    ``per-pls``/``closed-pool`` spellings or a null pool size, neither of which
    its historical scheduler accepts.
    """

    resolved = resolve_condition(condition_id)
    return {
        **resolved,
        "core_crop_policy": resolved["crop_policy"],
        "core_order_policy": resolved["order_policy"],
        "organization": "current",
        "crop_policy": (
            "per-shard" if resolved["crop_policy"] == "per-pls" else "per-sample"
        ),
        "order_policy": (
            "pls-wave" if resolved["order_policy"] == "closed-pool" else "global"
        ),
        "segments_per_pool": int(resolved.get("segments_per_pool") or 4),
    }


def _sample_records(samples: Sequence[TrainingSample]) -> list[SampleRecord]:
    result: list[SampleRecord] = []
    for index, sample in enumerate(samples):
        if sample.galp_image_id is None:
            raise ValueError(
                "PLS experiment requires galp_image_id for every training sample"
            )
        result.append(
            SampleRecord(
                logical_sample_id=sample.logical_sample_id,
                label=sample.label,
                galp_image_id=int(sample.galp_image_id),
                width=sample.width,
                height=sample.height,
                manifest_index=index,
            )
        )
    return result


def build_execution_plan(
    samples: Sequence[TrainingSample],
    *,
    seed: int,
    batch_count: int,
    batch_size: int,
    drop_last: bool,
    start_cursor: Mapping[str, Any] | None,
    organization: str,
    crop_policy: str,
    order_policy: str,
    segment_images: int,
    segments_per_pool: int,
    distributed_rank: int,
    distributed_world_size: int,
    complete_final_pool: bool = False,
    organization_seed: int = 20260810,
) -> PlsExecutionPlan:
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")
    records = _sample_records(samples)
    approximate_batches_per_epoch = max(1, len(records) // max(1, batch_size))
    start_epoch = int((start_cursor or {}).get("epoch", 0))
    epochs = max(1, start_epoch + math.ceil((batch_count + 1) / approximate_batches_per_epoch) + 1)
    while True:
        config = PlsScheduleConfig(
            segment_images=segment_images,
            segments_per_pool=segments_per_pool,
            optimizer_batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            organization_seed=organization_seed,
            organization=organization,
            crop_policy=crop_policy,
            order_policy=order_policy,
            drop_last=drop_last,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
        result = build_schedule(records, config)
        scheduled = result.optimizer_batches
        cursor = start_cursor or {"epoch": 0, "position": -1}
        last_epoch = int(cursor.get("epoch", 0))
        last_position = int(cursor.get("position", -1))
        cursor_id = cursor.get("logical_sample_id")
        start_index = 0
        if last_position >= 0:
            found = None
            for index, batch in enumerate(scheduled):
                tail = batch[-1]
                if tail.epoch == last_epoch and tail.epoch_position == last_position:
                    if cursor_id is not None and tail.sample.logical_sample_id != cursor_id:
                        raise ValueError("PLS resume cursor logical sample does not match schedule")
                    found = index + 1
                    break
            if found is None:
                if epochs <= last_epoch + 1:
                    epochs *= 2
                    continue
                raise ValueError("PLS resume cursor is not on a closed optimizer-batch boundary")
            start_index = found
        selected = scheduled[start_index : start_index + batch_count]
        if len(selected) == batch_count:
            if complete_final_pool and selected:
                first_key = (
                    selected[0][0].epoch,
                    selected[0][0].pool_index,
                )
                if start_index > 0:
                    previous = scheduled[start_index - 1][0]
                    if (previous.epoch, previous.pool_index) == first_key:
                        raise ValueError(
                            "PLS GPU-pool resume cursor must be at a closed-pool boundary"
                        )
                last_key = (selected[-1][0].epoch, selected[-1][0].pool_index)
                end_index = start_index + len(selected)
                while end_index < len(scheduled):
                    following = scheduled[end_index][0]
                    if (following.epoch, following.pool_index) != last_key:
                        break
                    selected.append(scheduled[end_index])
                    end_index += 1
            identities = [
                [
                    SampleIdentity(value.epoch, value.epoch_position, value.sample.logical_sample_id)
                    for value in batch
                ]
                for batch in selected
            ]
            return PlsExecutionPlan(
                result=result,
                scheduled_batches=[list(batch) for batch in selected],
                batches=identities,
                dropped_per_epoch=dict(result.dropped_per_epoch),
            )
        epochs *= 2


def augmentation_batches(
    plan: PlsExecutionPlan,
    samples: Mapping[str, TrainingSample],
    *,
    seed: int,
    domain: str,
) -> list[list[AugmentationDecision]]:
    result: list[list[AugmentationDecision]] = []
    for batch in plan.scheduled_batches:
        decisions: list[AugmentationDecision] = []
        for value in batch:
            sample = samples[value.sample.logical_sample_id]
            if plan.result.config.crop_policy == "per-shard":
                decision = derive_shard_shared_crop_augmentation(
                    seed=seed,
                    epoch=value.epoch,
                    physical_shard_id=value.source_shard_id,
                    logical_sample_id=value.sample.logical_sample_id,
                    source_width=sample.width,
                    source_height=sample.height,
                    domain=domain,
                )
            else:
                decision = derive_augmentation(
                    seed=seed,
                    epoch=value.epoch,
                    logical_sample_id=value.sample.logical_sample_id,
                    source_width=sample.width,
                    source_height=sample.height,
                    domain=domain,
                )
            decisions.append(decision)
        result.append(decisions)
    return result
