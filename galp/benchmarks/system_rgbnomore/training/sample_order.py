#!/usr/bin/env python3
"""Canonical sample ordering and four-stage sample identity accounting."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from training.schema import TRAINING_SAMPLE_ORDER_SCHEMA


@dataclass(frozen=True, order=True)
class SampleIdentity:
    epoch: int
    position: int
    logical_sample_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "logical_sample_id": self.logical_sample_id,
        }

    @classmethod
    def from_value(cls, value: "SampleIdentity | dict[str, Any]") -> "SampleIdentity":
        if isinstance(value, cls):
            return value
        return cls(int(value["epoch"]), int(value["position"]), str(value["logical_sample_id"]))


def _epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"galp-training-order-v1:{seed}:{epoch}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def canonical_epoch_order(logical_sample_ids: Sequence[str], seed: int, epoch: int) -> list[SampleIdentity]:
    if len(set(logical_sample_ids)) != len(logical_sample_ids):
        raise ValueError("canonical train manifest contains duplicate logical_sample_id values")
    shuffled = list(map(str, logical_sample_ids))
    random.Random(_epoch_seed(seed, epoch)).shuffle(shuffled)
    return [SampleIdentity(epoch, position, sample_id) for position, sample_id in enumerate(shuffled)]


def canonical_stream(
    logical_sample_ids: Sequence[str], seed: int, *, start_epoch: int = 0
) -> Iterator[SampleIdentity]:
    epoch = start_epoch
    while True:
        yield from canonical_epoch_order(logical_sample_ids, seed, epoch)
        epoch += 1


def batch_stream(
    logical_sample_ids: Sequence[str],
    seed: int,
    batch_size: int,
    *,
    drop_last: bool,
    start_epoch: int = 0,
) -> Iterator[tuple[list[SampleIdentity], int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    epoch = start_epoch
    while True:
        values = canonical_epoch_order(logical_sample_ids, seed, epoch)
        complete = len(values) // batch_size
        limit = complete * batch_size if drop_last else len(values)
        for begin in range(0, limit, batch_size):
            yield values[begin : min(begin + batch_size, limit)], 0
        dropped = len(values) - limit
        if dropped and drop_last:
            yield [], dropped
        epoch += 1


class SampleOrderLedger:
    """Record requested/read/emitted/consumed identities without conflating prefetch."""

    def __init__(self, pipeline: str) -> None:
        self.pipeline = pipeline
        self.requested: list[SampleIdentity] = []
        self.prefetched: list[SampleIdentity] = []
        self.emitted: list[SampleIdentity] = []
        self.consumed: list[SampleIdentity] = []
        self.dropped_per_epoch: dict[int, int] = {}

    def _extend(self, field: str, values: Iterable[SampleIdentity | dict[str, Any]]) -> None:
        target = getattr(self, field)
        target.extend(SampleIdentity.from_value(value) for value in values)

    def record_requested(self, values: Iterable[SampleIdentity | dict[str, Any]]) -> None:
        self._extend("requested", values)

    def record_prefetched(self, values: Iterable[SampleIdentity | dict[str, Any]]) -> None:
        self._extend("prefetched", values)

    def record_emitted(self, values: Iterable[SampleIdentity | dict[str, Any]]) -> None:
        self._extend("emitted", values)

    def record_consumed(self, values: Iterable[SampleIdentity | dict[str, Any]]) -> None:
        self._extend("consumed", values)

    def record_dropped(self, epoch: int, count: int) -> None:
        if count < 0:
            raise ValueError("dropped sample count cannot be negative")
        self.dropped_per_epoch[epoch] = self.dropped_per_epoch.get(epoch, 0) + count

    @staticmethod
    def _dicts(values: Sequence[SampleIdentity]) -> list[dict[str, Any]]:
        return [value.as_dict() for value in values]

    def validate(self, expected_consumed: Sequence[SampleIdentity]) -> dict[str, Any]:
        duplicate_consumed = len(self.consumed) != len(set(self.consumed))
        consumed_matches = self.consumed == list(expected_consumed)
        emitted_prefix = self.emitted[: len(self.consumed)] == self.consumed
        requested_set = set(self.requested)
        emitted_were_requested = all(value in requested_set for value in self.emitted)
        prefetched_set = set(self.prefetched)
        emitted_were_read = all(value in prefetched_set for value in self.emitted)
        overrun = max(0, len(self.prefetched) - len(self.consumed))
        failures: list[str] = []
        if duplicate_consumed:
            failures.append("duplicate optimizer-consumed sample identity")
        if not consumed_matches:
            failures.append("optimizer-consumed order differs from canonical order")
        if not emitted_prefix:
            failures.append("emitted sample prefix differs from optimizer-consumed order")
        if not emitted_were_requested:
            failures.append("pipeline emitted an identity that was never requested")
        if not emitted_were_read:
            failures.append("pipeline emitted an identity that was never prefetched/read")
        return {
            "ok": not failures,
            "failures": failures,
            "duplicate_consumed": duplicate_consumed,
            "consumed_matches_canonical": consumed_matches,
            "emitted_prefix_matches_consumed": emitted_prefix,
            "prefetch_overrun": overrun,
        }

    def as_dict(self, expected_consumed: Sequence[SampleIdentity] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": TRAINING_SAMPLE_ORDER_SCHEMA,
            "pipeline": self.pipeline,
            "identity_definition": ["epoch", "position", "logical_sample_id"],
            "requested_ids": self._dicts(self.requested),
            "prefetched_read_ids": self._dicts(self.prefetched),
            "emitted_ids": self._dicts(self.emitted),
            "optimizer_consumed_ids": self._dicts(self.consumed),
            "prefetch_overrun": max(0, len(self.prefetched) - len(self.consumed)),
            "dropped_samples_per_epoch": {str(key): value for key, value in sorted(self.dropped_per_epoch.items())},
        }
        if expected_consumed is not None:
            payload["validation"] = self.validate(expected_consumed)
        return payload
