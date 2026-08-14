#!/usr/bin/env python3
"""Deterministic physical-load-segment schedules and audit statistics.

One complete physical load segment (PLS) is the experimental load unit.  A
shuffle wave materializes ``segments_per_pool`` complete PLSs, uniformly
permutes every sample in that closed pool, consumes the complete permutation,
and only then advances to the next pool.  This is intentionally not a
streaming/reservoir shuffle.

For ``per-shard`` crop policy, every source physical shard has exactly one crop
key per epoch.  Runtime-balanced logical segments may touch several source
shards, but they do not merge their crop identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEDULE_SCHEMA = "galp-pls-schedule-v1"
TRAINING_MANIFEST_FORMAT = "galp-rgbnomore-training-manifest-v1"

ORGANIZATIONS = (
    "current",
    "storage-hash",
    "storage-stratified",
    "runtime-balanced",
)
CROP_POLICIES = ("per-sample", "per-shard")
ORDER_POLICIES = ("global", "pls-wave")


def _digest_int(namespace: str, *values: object) -> int:
    payload = ":".join((namespace, *(str(value) for value in values)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "little")


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class SampleRecord:
    logical_sample_id: str
    label: int
    galp_image_id: int
    width: int
    height: int
    manifest_index: int


@dataclass(frozen=True)
class ScheduledSample:
    sample: SampleRecord
    source_shard_id: int
    segment_id: int
    crop_key: str
    epoch: int
    pool_index: int
    epoch_position: int


@dataclass(frozen=True)
class PhysicalLoadSegment:
    segment_id: int
    samples: tuple[SampleRecord, ...]
    source_shard_ids: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class PoolWave:
    epoch: int
    pool_index: int
    segment_ids: tuple[int, ...]
    source_shard_ids: tuple[int, ...]
    samples_before_shuffle: tuple[str, ...]
    samples_after_shuffle: tuple[str, ...]
    crop_keys: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.samples_after_shuffle)


@dataclass(frozen=True)
class PlsScheduleConfig:
    segment_images: int = 1024
    segments_per_pool: int = 4
    optimizer_batch_size: int = 64
    epochs: int = 1
    seed: int = 11997733
    organization_seed: int = 20260810
    organization: str = "current"
    crop_policy: str = "per-shard"
    order_policy: str = "pls-wave"
    drop_last: bool = True
    distributed_rank: int = 0
    distributed_world_size: int = 1

    def validate(self) -> None:
        if self.segment_images <= 0:
            raise ValueError("segment_images must be positive")
        if self.segments_per_pool <= 0:
            raise ValueError("segments_per_pool must be positive")
        if self.optimizer_batch_size <= 0:
            raise ValueError("optimizer_batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.organization not in ORGANIZATIONS:
            raise ValueError(f"unknown organization {self.organization!r}; expected {ORGANIZATIONS}")
        if self.crop_policy not in CROP_POLICIES:
            raise ValueError(f"unknown crop policy {self.crop_policy!r}; expected {CROP_POLICIES}")
        if self.order_policy not in ORDER_POLICIES:
            raise ValueError(f"unknown order policy {self.order_policy!r}; expected {ORDER_POLICIES}")
        if self.distributed_world_size <= 0:
            raise ValueError("distributed_world_size must be positive")
        if not 0 <= self.distributed_rank < self.distributed_world_size:
            raise ValueError("distributed_rank must be within distributed_world_size")

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_images": self.segment_images,
            "segments_per_pool": self.segments_per_pool,
            "optimizer_batch_size": self.optimizer_batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "organization_seed": self.organization_seed,
            "organization": self.organization,
            "crop_policy": self.crop_policy,
            "order_policy": self.order_policy,
            "drop_last": self.drop_last,
            "distributed_rank": self.distributed_rank,
            "distributed_world_size": self.distributed_world_size,
            "pool_semantics": (
                "materialize complete segments, uniformly permute the closed pool, "
                "consume it completely, then load the next pool"
            ),
            "pool_size_definition": "sum(actual sample count of every complete PLS in the pool)",
            "crop_semantics": (
                "one crop key per source physical shard per epoch"
                if self.crop_policy == "per-shard"
                else "one crop key per logical sample per epoch"
            ),
        }


@dataclass
class ScheduleResult:
    config: PlsScheduleConfig
    segments: list[PhysicalLoadSegment]
    epochs: list[list[ScheduledSample]]
    pools: list[PoolWave]
    optimizer_batches: list[list[ScheduledSample]]
    dropped_per_epoch: dict[int, int]
    summary: dict[str, Any]


def load_training_samples(path: Path) -> list[SampleRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared_format = payload.get("format")
    if declared_format not in (None, TRAINING_MANIFEST_FORMAT):
        raise ValueError(f"unexpected training manifest format {declared_format!r}: {path}")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"training manifest contains no samples: {path}")
    result: list[SampleRecord] = []
    seen_ids: set[str] = set()
    seen_image_ids: set[int] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise ValueError(f"sample {index} is not an object")
        logical_id = str(raw.get("logical_sample_id", ""))
        if not logical_id or logical_id in seen_ids:
            raise ValueError(f"sample {index} has a missing or duplicate logical_sample_id")
        image_id = int(raw.get("galp_image_id", index))
        if image_id in seen_image_ids:
            raise ValueError(f"sample {index} duplicates galp_image_id {image_id}")
        label = int(raw["label"])
        if label < 0:
            raise ValueError(f"sample {index} has a negative label")
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"sample {index} lacks positive source dimensions")
        seen_ids.add(logical_id)
        seen_image_ids.add(image_id)
        result.append(SampleRecord(logical_id, label, image_id, width, height, index))
    return result


def _current_order(samples: Sequence[SampleRecord]) -> list[SampleRecord]:
    return sorted(samples, key=lambda sample: (sample.galp_image_id, sample.logical_sample_id))


def _hash_order(samples: Sequence[SampleRecord], seed: int) -> list[SampleRecord]:
    return sorted(
        samples,
        key=lambda sample: (
            _digest_int("galp-pls-storage-hash-v1", seed, sample.logical_sample_id),
            sample.logical_sample_id,
        ),
    )


def _stratified_order(samples: Sequence[SampleRecord], seed: int) -> list[SampleRecord]:
    buckets: dict[int, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        buckets[sample.label].append(sample)
    for label, bucket in buckets.items():
        random.Random(_digest_int("galp-pls-class-bucket-v1", seed, label)).shuffle(bucket)
    result: list[SampleRecord] = []
    round_index = 0
    while buckets:
        active_labels = sorted(buckets)
        random.Random(_digest_int("galp-pls-class-round-v1", seed, round_index)).shuffle(active_labels)
        empty: list[int] = []
        for label in active_labels:
            bucket = buckets[label]
            result.append(bucket.pop())
            if not bucket:
                empty.append(label)
        for label in empty:
            del buckets[label]
        round_index += 1
    return result


def _chunks(values: Sequence[SampleRecord], size: int) -> Iterator[list[SampleRecord]]:
    for begin in range(0, len(values), size):
        yield list(values[begin : begin + size])


def build_segments(
    samples: Sequence[SampleRecord], config: PlsScheduleConfig
) -> tuple[list[PhysicalLoadSegment], dict[str, int]]:
    """Build proposed PLS membership and source-shard identities.

    ``runtime-balanced`` changes logical load membership without rewriting the
    current physical source.  Every other organization models a storage layout
    and therefore assigns one new source-shard id to each resulting PLS.
    """

    config.validate()
    current = _current_order(samples)
    current_source_shard = {
        sample.logical_sample_id: index // config.segment_images for index, sample in enumerate(current)
    }
    if config.organization == "current":
        ordered = current
    elif config.organization == "storage-hash":
        ordered = _hash_order(samples, config.organization_seed)
    else:
        ordered = _stratified_order(samples, config.organization_seed)

    segments: list[PhysicalLoadSegment] = []
    source_by_sample: dict[str, int] = {}
    for segment_id, chunk in enumerate(_chunks(ordered, config.segment_images)):
        if config.organization == "runtime-balanced":
            source_ids = tuple(
                sorted({current_source_shard[sample.logical_sample_id] for sample in chunk})
            )
            for sample in chunk:
                source_by_sample[sample.logical_sample_id] = current_source_shard[sample.logical_sample_id]
        else:
            source_ids = (segment_id,)
            for sample in chunk:
                source_by_sample[sample.logical_sample_id] = segment_id
        segments.append(PhysicalLoadSegment(segment_id, tuple(chunk), source_ids))
    if len(source_by_sample) != len(samples):
        raise RuntimeError("PLS construction did not assign every sample exactly once")
    return segments, source_by_sample


def crop_key(
    config: PlsScheduleConfig,
    *,
    epoch: int,
    sample: SampleRecord,
    source_shard_id: int,
) -> str:
    if config.crop_policy == "per-shard":
        identity = f"shard:{source_shard_id}"
    else:
        identity = f"sample:{sample.logical_sample_id}"
    return hashlib.sha256(
        f"galp-pls-crop-v1:{config.seed}:{epoch}:{identity}".encode("utf-8")
    ).hexdigest()


def _schedule_global_epoch(
    samples: Sequence[SampleRecord],
    source_by_sample: Mapping[str, int],
    segment_by_sample: Mapping[str, int],
    config: PlsScheduleConfig,
    epoch: int,
) -> tuple[list[ScheduledSample], list[PoolWave]]:
    shuffled = list(samples)
    random.Random(_digest_int("galp-pls-global-order-v1", config.seed, epoch)).shuffle(shuffled)
    scheduled = [
        ScheduledSample(
            sample=sample,
            source_shard_id=source_by_sample[sample.logical_sample_id],
            segment_id=segment_by_sample[sample.logical_sample_id],
            crop_key=crop_key(
                config,
                epoch=epoch,
                sample=sample,
                source_shard_id=source_by_sample[sample.logical_sample_id],
            ),
            epoch=epoch,
            pool_index=0,
            epoch_position=position,
        )
        for position, sample in enumerate(shuffled)
    ]
    pool = PoolWave(
        epoch=epoch,
        pool_index=0,
        segment_ids=tuple(sorted(set(segment_by_sample.values()))),
        source_shard_ids=tuple(sorted(set(source_by_sample.values()))),
        samples_before_shuffle=tuple(sample.logical_sample_id for sample in samples),
        samples_after_shuffle=tuple(sample.logical_sample_id for sample in shuffled),
        crop_keys=tuple(value.crop_key for value in scheduled),
    )
    return scheduled, [pool]


def _schedule_wave_epoch(
    segments: Sequence[PhysicalLoadSegment],
    source_by_sample: Mapping[str, int],
    config: PlsScheduleConfig,
    epoch: int,
) -> tuple[list[ScheduledSample], list[PoolWave]]:
    segment_order = list(segments)
    random.Random(_digest_int("galp-pls-segment-order-v1", config.seed, epoch)).shuffle(segment_order)
    scheduled: list[ScheduledSample] = []
    pools: list[PoolWave] = []
    for pool_index, begin in enumerate(range(0, len(segment_order), config.segments_per_pool)):
        selected = segment_order[begin : begin + config.segments_per_pool]
        before = [sample for segment in selected for sample in segment.samples]
        after = list(before)
        random.Random(_digest_int("galp-pls-pool-order-v1", config.seed, epoch, pool_index)).shuffle(after)
        pool_scheduled: list[ScheduledSample] = []
        segment_by_sample = {
            sample.logical_sample_id: segment.segment_id
            for segment in selected
            for sample in segment.samples
        }
        for sample in after:
            source_shard = source_by_sample[sample.logical_sample_id]
            value = ScheduledSample(
                sample=sample,
                source_shard_id=source_shard,
                segment_id=segment_by_sample[sample.logical_sample_id],
                crop_key=crop_key(
                    config,
                    epoch=epoch,
                    sample=sample,
                    source_shard_id=source_shard,
                ),
                epoch=epoch,
                pool_index=pool_index,
                epoch_position=len(scheduled) + len(pool_scheduled),
            )
            pool_scheduled.append(value)
        scheduled.extend(pool_scheduled)
        pools.append(
            PoolWave(
                epoch=epoch,
                pool_index=pool_index,
                segment_ids=tuple(segment.segment_id for segment in selected),
                source_shard_ids=tuple(
                    sorted({source for segment in selected for source in segment.source_shard_ids})
                ),
                samples_before_shuffle=tuple(sample.logical_sample_id for sample in before),
                samples_after_shuffle=tuple(sample.logical_sample_id for sample in after),
                crop_keys=tuple(value.crop_key for value in pool_scheduled),
            )
        )
    return scheduled, pools


def _batch_statistics(batch: Sequence[ScheduledSample]) -> dict[str, Any]:
    counts = Counter(value.sample.label for value in batch)
    total = len(batch)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    source_counts = Counter(value.source_shard_id for value in batch)
    crop_counts = Counter(value.crop_key for value in batch)
    return {
        "samples": total,
        "unique_classes": len(counts),
        "label_entropy_bits": entropy,
        "single_class": len(counts) == 1,
        "unique_source_shards": len(source_counts),
        "unique_crop_keys": len(crop_counts),
        "largest_source_shard_fraction": max(source_counts.values()) / total,
        "largest_crop_fraction": max(crop_counts.values()) / total,
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _summarize(
    samples: Sequence[SampleRecord],
    segments: Sequence[PhysicalLoadSegment],
    epochs: Sequence[Sequence[ScheduledSample]],
    pools: Sequence[PoolWave],
    optimizer_batches: Sequence[Sequence[ScheduledSample]],
    dropped_per_epoch: Mapping[int, int],
    config: PlsScheduleConfig,
) -> dict[str, Any]:
    expected_ids = {sample.logical_sample_id for sample in samples}
    epoch_coverage: list[dict[str, Any]] = []
    for epoch_index, values in enumerate(epochs):
        ids = [value.sample.logical_sample_id for value in values]
        observed = set(ids)
        epoch_coverage.append(
            {
                "epoch": epoch_index,
                "scheduled_samples": len(ids),
                "unique_samples": len(observed),
                "duplicate_samples": len(ids) - len(observed),
                "missing_samples": len(expected_ids - observed),
                "extra_samples": len(observed - expected_ids),
                "complete": observed == expected_ids and len(ids) == len(expected_ids),
                "order_sha256": _sha256_lines(ids),
                "crop_trace_sha256": _sha256_lines(value.crop_key for value in values),
            }
        )
    batch_stats = [_batch_statistics(batch) for batch in optimizer_batches]
    return {
        "schema_version": SCHEDULE_SCHEMA,
        "purpose": (
            "measure model convergence and final-accuracy effects; schedule mixing and runtime metrics "
            "are explanatory observations, not optimization acceptance gates"
        ),
        "sample_count": len(samples),
        "class_count": len({sample.label for sample in samples}),
        "segment_count": len(segments),
        "segment_size": _distribution([segment.size for segment in segments]),
        "source_shards_per_segment": _distribution(
            [len(segment.source_shard_ids) for segment in segments]
        ),
        "pool_count": len(pools),
        "pool_size": _distribution([pool.size for pool in pools]),
        "source_shards_per_pool": _distribution([len(pool.source_shard_ids) for pool in pools]),
        "optimizer_batch_count": len(optimizer_batches),
        "optimizer_batch_size": _distribution([len(batch) for batch in optimizer_batches]),
        "unique_classes_per_batch": _distribution(
            [stats["unique_classes"] for stats in batch_stats]
        ),
        "label_entropy_bits_per_batch": _distribution(
            [stats["label_entropy_bits"] for stats in batch_stats]
        ),
        "unique_source_shards_per_batch": _distribution(
            [stats["unique_source_shards"] for stats in batch_stats]
        ),
        "unique_crop_keys_per_batch": _distribution(
            [stats["unique_crop_keys"] for stats in batch_stats]
        ),
        "largest_source_shard_fraction_per_batch": _distribution(
            [stats["largest_source_shard_fraction"] for stats in batch_stats]
        ),
        "largest_crop_fraction_per_batch": _distribution(
            [stats["largest_crop_fraction"] for stats in batch_stats]
        ),
        "single_class_batch_count": sum(int(stats["single_class"]) for stats in batch_stats),
        "dropped_per_epoch": {str(key): value for key, value in sorted(dropped_per_epoch.items())},
        "epoch_coverage": epoch_coverage,
        "valid": all(item["complete"] for item in epoch_coverage),
        "config": config.as_dict(),
    }


def build_schedule(samples: Sequence[SampleRecord], config: PlsScheduleConfig) -> ScheduleResult:
    config.validate()
    if not samples:
        raise ValueError("samples must not be empty")
    if len({sample.logical_sample_id for sample in samples}) != len(samples):
        raise ValueError("samples contain duplicate logical_sample_id values")
    segments, source_by_sample = build_segments(samples, config)
    segment_by_sample = {
        sample.logical_sample_id: segment.segment_id
        for segment in segments
        for sample in segment.samples
    }
    all_epochs: list[list[ScheduledSample]] = []
    all_pools: list[PoolWave] = []
    optimizer_batches: list[list[ScheduledSample]] = []
    dropped: dict[int, int] = {}
    for epoch in range(config.epochs):
        if config.order_policy == "global":
            scheduled, pools = _schedule_global_epoch(
                samples, source_by_sample, segment_by_sample, config, epoch
            )
        else:
            scheduled, pools = _schedule_wave_epoch(segments, source_by_sample, config, epoch)

        if config.distributed_world_size > 1:
            if config.order_policy == "pls-wave":
                rank_pool_keys = {
                    (pool.epoch, pool.pool_index)
                    for pool in pools
                    if pool.pool_index % config.distributed_world_size == config.distributed_rank
                }
                rank_ids = {
                    sample_id
                    for pool in pools
                    if (pool.epoch, pool.pool_index) in rank_pool_keys
                    for sample_id in pool.samples_after_shuffle
                }
                scheduled = [value for value in scheduled if value.sample.logical_sample_id in rank_ids]
                pools = [pool for pool in pools if (pool.epoch, pool.pool_index) in rank_pool_keys]
            else:
                scheduled = scheduled[config.distributed_rank :: config.distributed_world_size]
            scheduled = [
                ScheduledSample(
                    sample=value.sample,
                    source_shard_id=value.source_shard_id,
                    segment_id=value.segment_id,
                    crop_key=value.crop_key,
                    epoch=value.epoch,
                    pool_index=value.pool_index,
                    epoch_position=position,
                )
                for position, value in enumerate(scheduled)
            ]

        all_epochs.append(scheduled)
        all_pools.extend(pools)
        # Preserve the closed-wave lifetime contract: optimizer batches never
        # combine a tail from one pool with samples from its successor.  Full
        # production PLS sizes are expected to be batch-aligned; an unaligned
        # canary remains honest by either emitting a partial batch or recording
        # a per-pool dropped tail.
        pool_sizes = [pool.size for pool in pools]
        if config.distributed_world_size > 1 and config.order_policy == "global":
            pool_sizes = [len(scheduled)]
        cursor = 0
        for pool_size in pool_sizes:
            pool_values = scheduled[cursor : cursor + pool_size]
            cursor += pool_size
            usable = len(pool_values)
            if config.drop_last:
                usable = usable // config.optimizer_batch_size * config.optimizer_batch_size
                dropped[epoch] = dropped.get(epoch, 0) + len(pool_values) - usable
            for begin in range(0, usable, config.optimizer_batch_size):
                batch = pool_values[begin : min(begin + config.optimizer_batch_size, usable)]
                if batch:
                    optimizer_batches.append(batch)
        if cursor != len(scheduled):
            raise RuntimeError("closed-pool batching did not consume the scheduled epoch exactly")
    summary = _summarize(
        samples,
        segments,
        all_epochs,
        all_pools,
        optimizer_batches,
        dropped,
        config,
    )
    return ScheduleResult(
        config=config,
        segments=segments,
        epochs=all_epochs,
        pools=all_pools,
        optimizer_batches=optimizer_batches,
        dropped_per_epoch=dropped,
        summary=summary,
    )


def batch_records(result: ScheduleResult) -> Iterator[dict[str, Any]]:
    for index, batch in enumerate(result.optimizer_batches):
        record = _batch_statistics(batch)
        record.update(
            batch_index=index,
            epoch=batch[0].epoch,
            pool_index=batch[0].pool_index,
            first_epoch_position=batch[0].epoch_position,
            last_epoch_position=batch[-1].epoch_position,
            sample_order_sha256=_sha256_lines(value.sample.logical_sample_id for value in batch),
            crop_trace_sha256=_sha256_lines(value.crop_key for value in batch),
        )
        yield record
