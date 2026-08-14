#!/usr/bin/env python3
"""Epoch-local streaming schedules for the four core PLS conditions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Sequence

import numpy as np

from .matrix import resolve_condition


SCHEDULE_SCHEMA = "galp-pls-core-epoch-schedule-v2"


class ScheduleLayout(Protocol):
    sample_count: int
    logical_sample_ids: Sequence[str]
    virtual_pls_ids: np.ndarray
    positions_by_pls: Sequence[np.ndarray]


def stable_digest(namespace: str, *values: object) -> str:
    payload = json_key(namespace, *values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(namespace: str, *values: object) -> int:
    return int.from_bytes(
        hashlib.sha256(json_key(namespace, *values).encode("utf-8")).digest()[:8],
        "little",
    )


def json_key(namespace: str, *values: object) -> str:
    return ":".join((namespace, *(str(value) for value in values)))


def crop_key(
    *,
    training_seed: int,
    epoch: int,
    logical_sample_id: str,
    virtual_pls_id: int,
    crop_policy: str,
) -> str:
    if crop_policy == "per-sample":
        return stable_digest(
            "crop-per-sample", training_seed, epoch, logical_sample_id
        )
    if crop_policy == "per-pls":
        return stable_digest("crop-per-pls", training_seed, epoch, virtual_pls_id)
    raise ValueError(f"unknown crop policy: {crop_policy}")


def flip_key(*, training_seed: int, epoch: int, logical_sample_id: str) -> str:
    return stable_digest(
        "horizontal-flip", training_seed, epoch, logical_sample_id
    )


def randaugment_key(
    *, training_seed: int, epoch: int, logical_sample_id: str, operation_index: int
) -> str:
    return stable_digest(
        "dct-randaugment",
        training_seed,
        epoch,
        logical_sample_id,
        operation_index,
    )


def mixup_key(*, training_seed: int, epoch: int, microbatch_index: int) -> str:
    return stable_digest("dct-mixup", training_seed, epoch, microbatch_index)


@dataclass(frozen=True)
class ScheduledItem:
    planned_position: int
    virtual_pls_id: int
    epoch_position: int
    pool_index: int


@dataclass(frozen=True)
class Microbatch:
    epoch: int
    pool_index: int
    microbatch_index: int
    pool_microbatch_index: int
    items: tuple[ScheduledItem, ...]

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class ClosedPool:
    epoch: int
    pool_index: int
    virtual_pls_ids: tuple[int, ...]
    ordered_positions: tuple[int, ...]
    microbatches: tuple[Microbatch, ...]

    @property
    def sample_count(self) -> int:
        return len(self.ordered_positions)


@dataclass(frozen=True)
class EpochSummary:
    schema_version: str
    condition_id: str
    seed: int
    epoch: int
    sample_count: int
    pool_count: int
    microbatch_count: int
    optimizer_update_count: int
    sample_order_digest: str
    pool_membership_digest: str


def _digest_positions(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def epoch_position_pools(
    layout: ScheduleLayout,
    *,
    condition_id: str,
    seed: int,
    epoch: int,
) -> Iterator[tuple[int, tuple[int, ...], list[int]]]:
    """Yield one epoch's pools without constructing any other epoch."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    condition = resolve_condition(condition_id)
    if condition["order_policy"] == "global":
        positions = list(range(layout.sample_count))
        random.Random(stable_seed("global-sample-order", seed, epoch)).shuffle(positions)
        yield 0, tuple(range(len(layout.positions_by_pls))), positions
        return

    pls_order = list(range(len(layout.positions_by_pls)))
    random.Random(stable_seed("closed-pool-pls-order", seed, epoch)).shuffle(pls_order)
    segments_per_pool = int(condition["segments_per_pool"])
    for pool_index, begin in enumerate(range(0, len(pls_order), segments_per_pool)):
        pool_pls = tuple(pls_order[begin : begin + segments_per_pool])
        positions = [
            int(position)
            for pls_id in pool_pls
            for position in layout.positions_by_pls[pls_id]
        ]
        random.Random(
            stable_seed("closed-pool-sample-order", seed, epoch, pool_index)
        ).shuffle(positions)
        yield pool_index, pool_pls, positions


def iter_epoch_pools(
    layout: ScheduleLayout,
    *,
    condition_id: str,
    seed: int,
    epoch: int,
    microbatch_size: int = 64,
) -> Iterator[ClosedPool]:
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    epoch_position = 0
    microbatch_index = 0
    for pool_index, pls_ids, positions in epoch_position_pools(
        layout, condition_id=condition_id, seed=seed, epoch=epoch
    ):
        microbatches: list[Microbatch] = []
        for pool_microbatch_index, begin in enumerate(
            range(0, len(positions), microbatch_size)
        ):
            items: list[ScheduledItem] = []
            for position in positions[begin : begin + microbatch_size]:
                items.append(
                    ScheduledItem(
                        planned_position=position,
                        virtual_pls_id=int(layout.virtual_pls_ids[position]),
                        epoch_position=epoch_position,
                        pool_index=pool_index,
                    )
                )
                epoch_position += 1
            microbatches.append(
                Microbatch(
                    epoch=epoch,
                    pool_index=pool_index,
                    microbatch_index=microbatch_index,
                    pool_microbatch_index=pool_microbatch_index,
                    items=tuple(items),
                )
            )
            microbatch_index += 1
        yield ClosedPool(
            epoch=epoch,
            pool_index=pool_index,
            virtual_pls_ids=pls_ids,
            ordered_positions=tuple(positions),
            microbatches=tuple(microbatches),
        )
    if epoch_position != layout.sample_count:
        raise RuntimeError(
            f"epoch schedule emitted {epoch_position} samples; expected {layout.sample_count}"
        )


def update_windows(
    pool: ClosedPool, *, gradient_accumulation: int = 16
) -> Iterator[tuple[Microbatch, ...]]:
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    for begin in range(0, len(pool.microbatches), gradient_accumulation):
        yield pool.microbatches[begin : begin + gradient_accumulation]


def summarize_epoch(
    layout: ScheduleLayout,
    *,
    condition_id: str,
    seed: int,
    epoch: int,
    microbatch_size: int = 64,
    gradient_accumulation: int = 16,
) -> EpochSummary:
    positions: list[int] = []
    pool_membership: list[int] = []
    pools = 0
    microbatches = 0
    updates = 0
    for pool in iter_epoch_pools(
        layout,
        condition_id=condition_id,
        seed=seed,
        epoch=epoch,
        microbatch_size=microbatch_size,
    ):
        pools += 1
        positions.extend(pool.ordered_positions)
        pool_membership.extend(pool.virtual_pls_ids)
        pool_membership.append(2**63 - 1)
        microbatches += len(pool.microbatches)
        updates += sum(1 for _ in update_windows(pool, gradient_accumulation=gradient_accumulation))
    if len(positions) != layout.sample_count:
        raise RuntimeError("schedule summary did not cover the layout")
    if len(set(positions)) != layout.sample_count:
        raise RuntimeError("schedule summary contains missing or duplicate positions")
    return EpochSummary(
        schema_version=SCHEDULE_SCHEMA,
        condition_id=condition_id,
        seed=int(seed),
        epoch=int(epoch),
        sample_count=len(positions),
        pool_count=pools,
        microbatch_count=microbatches,
        optimizer_update_count=updates,
        sample_order_digest=_digest_positions(positions),
        pool_membership_digest=_digest_positions(pool_membership),
    )


def total_optimizer_updates(
    layout: ScheduleLayout,
    *,
    condition_id: str,
    seed: int,
    epochs: int,
    microbatch_size: int = 64,
    gradient_accumulation: int = 16,
) -> int:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    # Pool sizes, not their order, determine the update count.  Evaluate one
    # epoch using the actual frozen mapping and multiply only after verifying
    # the registered G=1024/M=4 layout yields an epoch-invariant count.
    first = summarize_epoch(
        layout,
        condition_id=condition_id,
        seed=seed,
        epoch=0,
        microbatch_size=microbatch_size,
        gradient_accumulation=gradient_accumulation,
    ).optimizer_update_count
    if epochs > 1:
        second = summarize_epoch(
            layout,
            condition_id=condition_id,
            seed=seed,
            epoch=1,
            microbatch_size=microbatch_size,
            gradient_accumulation=gradient_accumulation,
        ).optimizer_update_count
        if first != second:
            return sum(
                summarize_epoch(
                    layout,
                    condition_id=condition_id,
                    seed=seed,
                    epoch=epoch,
                    microbatch_size=microbatch_size,
                    gradient_accumulation=gradient_accumulation,
                ).optimizer_update_count
                for epoch in range(epochs)
            )
    return first * epochs


def policy_digest(
    *, layout_hash: str, seed: int, condition_id: str, kind: str
) -> str:
    condition = resolve_condition(condition_id)
    if kind == "sample_order":
        policy = condition["order_policy"]
    elif kind == "pool_membership":
        policy = condition["order_policy"]
    elif kind == "crop_key":
        policy = condition["crop_policy"]
    else:
        raise ValueError(f"unknown policy digest kind: {kind}")
    return stable_digest(f"{kind}-policy", layout_hash, seed, policy)
