#!/usr/bin/env python3
"""Pure selection and access-pattern helpers for the image-order benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_SAMPLING = frozenset(("4:2:0", "4:4:4"))


@dataclass(frozen=True)
class ShardRange:
    shard_id: int
    first_image_id: int
    image_count: int

    @property
    def end_image_id(self) -> int:
        return self.first_image_id + self.image_count


@dataclass(frozen=True)
class GalpLayout:
    version: int
    rowgroup_vectors: int
    rowgroups_per_shard: int
    image_count: int
    shards: tuple[ShardRange, ...]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_galp_layout(path: Path) -> GalpLayout:
    """Read only the geometry/ranges needed to define rowgroup-aligned batches."""

    data = path.read_bytes()
    offset = 0

    def take(fmt: str) -> tuple[int, ...]:
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(data):
            raise ValueError(f"truncated GALP shard manifest: {path}")
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values

    def take_string() -> str:
        nonlocal offset
        (size,) = take("<I")
        if offset + size > len(data):
            raise ValueError(f"truncated GALP shard manifest string: {path}")
        value = data[offset : offset + size].decode("utf-8")
        offset += size
        return value

    if data[:8] != b"GJDCTSH1":
        raise ValueError(f"unexpected GALP shard manifest format: {path}")
    offset = 8
    version, _reserved, rowgroup_vectors, rowgroups_per_shard, image_count = take("<IHIIQ")
    (shard_count,) = take("<I")
    shards: list[ShardRange] = []
    for _ in range(shard_count):
        (
            shard_id,
            first_image_id,
            shard_image_count,
            _real_rows,
            _padding_rows,
            _physical_rows,
            _rowgroup_count,
            _block_group_count,
        ) = take("<IQIQQQII")
        take("<QQ")
        take_string()
        take_string()
        shards.append(ShardRange(int(shard_id), int(first_image_id), int(shard_image_count)))
    if offset != len(data):
        raise ValueError(f"GALP shard manifest has trailing bytes: {path}")
    if not rowgroup_vectors or not shards:
        raise ValueError(f"GALP shard manifest has invalid geometry: {path}")
    return GalpLayout(
        version=int(version),
        rowgroup_vectors=int(rowgroup_vectors),
        rowgroups_per_shard=int(rowgroups_per_shard),
        image_count=int(image_count),
        shards=tuple(shards),
    )


def _chunks(values: Sequence[int], size: int) -> list[list[int]]:
    if size <= 0 or len(values) % size:
        raise ValueError(f"cannot split {len(values)} values into batches of {size}")
    return [list(values[offset : offset + size]) for offset in range(0, len(values), size)]


def _flatten(batches: Iterable[Sequence[int]]) -> list[int]:
    return [image_id for batch in batches for image_id in batch]


def aligned_supported_batches(
    layout: GalpLayout,
    eligible_ids: set[int],
    batch_size: int,
) -> list[list[int]]:
    """Return batches aligned to the physical image-vector rowgroup geometry."""

    if batch_size != layout.rowgroup_vectors:
        raise ValueError(
            "the formal locality comparison requires batch_size == manifest rowgroup_vectors "
            f"({batch_size} != {layout.rowgroup_vectors})"
        )
    candidates: list[list[int]] = []
    for shard in layout.shards:
        for start in range(shard.first_image_id, shard.end_image_id - batch_size + 1, layout.rowgroup_vectors):
            batch = list(range(start, start + batch_size))
            if all(image_id in eligible_ids for image_id in batch):
                candidates.append(batch)
    return candidates


def build_condition_orders(
    *,
    layout: GalpLayout,
    eligible_ids: Sequence[int],
    labels: Sequence[int],
    batch_size: int,
    warmup_batches: int,
    measurement_batches: int,
    seed: int,
) -> dict[str, Any]:
    """Build operational and paired orders without mixing warmup/measured cohorts."""

    if len(labels) != layout.image_count:
        raise ValueError(f"label count {len(labels)} does not match image count {layout.image_count}")
    eligible = sorted(set(int(image_id) for image_id in eligible_ids))
    if not eligible or eligible[0] < 0 or eligible[-1] >= layout.image_count:
        raise ValueError("eligible image IDs are empty or outside the manifest")
    total_batches = warmup_batches + measurement_batches
    total_samples = total_batches * batch_size
    eligible_set = set(eligible)
    candidates = aligned_supported_batches(layout, eligible_set, batch_size)
    if len(candidates) < total_batches:
        raise ValueError(
            f"need {total_batches} fully supported aligned batches, found {len(candidates)}"
        )

    # The operational condition exactly mirrors benchmarks/system_rgbnomore/manifest.py:
    # random.Random(seed).shuffle(full eligible permutation), then take its prefix.
    current_random = list(eligible)
    random.Random(seed).shuffle(current_random)
    current_random = current_random[:total_samples]

    selected_candidates = list(candidates)
    random.Random(seed).shuffle(selected_candidates)
    selected_candidates = selected_candidates[:total_batches]
    warmup_contiguous = selected_candidates[:warmup_batches]
    measured_contiguous = selected_candidates[warmup_batches:]

    warmup_scattered = _flatten(warmup_contiguous)
    measured_scattered = _flatten(measured_contiguous)
    random.Random(seed ^ 0x51A7E).shuffle(warmup_scattered)
    random.Random(seed ^ 0xC0171).shuffle(measured_scattered)

    paired_scattered = _chunks(warmup_scattered, batch_size) + _chunks(measured_scattered, batch_size)
    contiguous = [list(batch) for batch in selected_candidates]
    current_batches = _chunks(current_random, batch_size)

    paired_warmup_ids = set(_flatten(warmup_contiguous))
    paired_measured_ids = set(_flatten(measured_contiguous))
    if paired_warmup_ids & paired_measured_ids:
        raise AssertionError("paired warmup and measurement cohorts overlap")
    if set(_flatten(paired_scattered[:warmup_batches])) != paired_warmup_ids:
        raise AssertionError("paired scattered warmup cohort changed")
    if set(_flatten(paired_scattered[warmup_batches:])) != paired_measured_ids:
        raise AssertionError("paired scattered measured cohort changed")

    # A representative, non-overlapping evaluation set for the training-order
    # probe. It is not part of the timed performance comparison.
    excluded = paired_warmup_ids | paired_measured_ids
    training_eval_ids = [image_id for image_id in current_random if image_id not in excluded]
    if len(training_eval_ids) < measurement_batches * batch_size:
        remaining = [image_id for image_id in eligible if image_id not in excluded and image_id not in training_eval_ids]
        random.Random(seed ^ 0xE7A1).shuffle(remaining)
        training_eval_ids.extend(remaining)
    training_eval_ids = training_eval_ids[: measurement_batches * batch_size]

    conditions = {
        "current_random": current_batches,
        "paired_scattered": paired_scattered,
        "contiguous": contiguous,
    }
    return {
        "schema_version": "galp_image_order_selection_v1",
        "seed": int(seed),
        "batch_size": int(batch_size),
        "warmup_batches": int(warmup_batches),
        "measurement_batches": int(measurement_batches),
        "eligible_image_count": len(eligible),
        "aligned_candidate_batch_count": len(candidates),
        "layout": {
            "version": layout.version,
            "rowgroup_vectors": layout.rowgroup_vectors,
            "rowgroups_per_shard": layout.rowgroups_per_shard,
            "image_count": layout.image_count,
            "shards": [asdict(shard) for shard in layout.shards],
        },
        "conditions": {
            name: {
                "batches": batches,
                "flat_image_ids": _flatten(batches),
                "access": access_pattern_summary(batches, labels, layout),
            }
            for name, batches in conditions.items()
        },
        "paired_invariants": {
            "warmup_image_ids_equal": True,
            "measured_image_ids_equal": True,
            "measured_image_id_set_sha256": canonical_sha256(sorted(paired_measured_ids)),
        },
        "training_probe": {
            "train_image_ids": _flatten(measured_contiguous),
            "train_contiguous_batches": measured_contiguous,
            "evaluation_image_ids": training_eval_ids,
            "train_evaluation_overlap": len(set(training_eval_ids) & paired_measured_ids),
        },
    }


def image_rowgroup(layout: GalpLayout, image_id: int) -> tuple[int, int]:
    for shard in layout.shards:
        if shard.first_image_id <= image_id < shard.end_image_id:
            return shard.shard_id, (image_id - shard.first_image_id) // layout.rowgroup_vectors
    raise ValueError(f"image ID {image_id} is outside the GALP layout")


def _entropy_bits(values: Sequence[int]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "max": ordered[-1],
    }


def access_pattern_summary(
    batches: Sequence[Sequence[int]],
    labels: Sequence[int],
    layout: GalpLayout,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("access pattern contains no batches")
    rowgroups_per_batch: list[float] = []
    shards_per_batch: list[float] = []
    unique_classes_per_batch: list[float] = []
    label_entropy_bits_per_batch: list[float] = []
    adjacent_pairs = 0
    total_pairs = 0
    absolute_deltas: list[float] = []
    for batch in batches:
        if not batch:
            raise ValueError("access pattern contains an empty batch")
        rowgroups = {image_rowgroup(layout, int(image_id)) for image_id in batch}
        rowgroups_per_batch.append(float(len(rowgroups)))
        shards_per_batch.append(float(len({shard_id for shard_id, _ in rowgroups})))
        batch_labels = [int(labels[int(image_id)]) for image_id in batch]
        unique_classes_per_batch.append(float(len(set(batch_labels))))
        label_entropy_bits_per_batch.append(_entropy_bits(batch_labels))
        for left, right in zip(batch, batch[1:]):
            delta = abs(int(right) - int(left))
            adjacent_pairs += int(int(right) == int(left) + 1)
            total_pairs += 1
            absolute_deltas.append(float(delta))
    return {
        "batch_count": len(batches),
        "rowgroups_per_batch": _numeric_summary(rowgroups_per_batch),
        "shards_per_batch": _numeric_summary(shards_per_batch),
        "unique_classes_per_batch": _numeric_summary(unique_classes_per_batch),
        "label_entropy_bits_per_batch": _numeric_summary(label_entropy_bits_per_batch),
        "forward_adjacent_pair_fraction": adjacent_pairs / total_pairs if total_pairs else 0.0,
        "absolute_image_id_delta": _numeric_summary(absolute_deltas) if absolute_deltas else None,
    }
