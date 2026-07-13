#!/usr/bin/env python3
"""Quantify label mixing for path-contiguous versus shuffled training batches."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_label_stats(labels: Sequence[int], order: Sequence[int], batch_size: int) -> dict[str, Any]:
    usable = len(order) // batch_size * batch_size
    unique_classes: list[int] = []
    entropy_bits: list[float] = []
    single_class_batches = 0
    for offset in range(0, usable, batch_size):
        batch_labels = [int(labels[index]) for index in order[offset : offset + batch_size]]
        counts = Counter(batch_labels)
        unique_classes.append(len(counts))
        single_class_batches += int(len(counts) == 1)
        entropy_bits.append(
            -sum((count / batch_size) * math.log2(count / batch_size) for count in counts.values())
        )
    return {
        "batches": len(unique_classes),
        "unique_classes_mean": statistics.fmean(unique_classes),
        "unique_classes_p50": statistics.median(unique_classes),
        "unique_classes_min": min(unique_classes),
        "unique_classes_max": max(unique_classes),
        "entropy_bits_mean": statistics.fmean(entropy_bits),
        "entropy_bits_p50": statistics.median(entropy_bits),
        "single_class_batch_fraction": single_class_batches / len(unique_classes),
    }


def analyze_training_index(index_csv: Path, batch_size: int, seed: int) -> dict[str, Any]:
    rows: list[tuple[str, int]] = []
    with index_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Filepath", "Label"}.issubset(reader.fieldnames):
            raise ValueError(f"{index_csv} must contain Filepath and Label columns")
        for row_number, row in enumerate(reader, start=2):
            path = str(row["Filepath"]).replace("\\", "/")
            while path.startswith("./"):
                path = path[2:]
            label = int(row["Label"])
            if not 0 <= label < 1000:
                raise ValueError(f"{index_csv}:{row_number} has invalid label {label}")
            rows.append((path, label))
    if not rows:
        raise ValueError(f"{index_csv} contains no samples")
    rows.sort(key=lambda item: item[0])
    labels = [label for _, label in rows]
    contiguous_order = list(range(len(labels)))
    random_order = list(contiguous_order)
    random.Random(seed).shuffle(random_order)
    usable = len(labels) // batch_size * batch_size
    return {
        "schema_version": "galp_training_order_stats_v1",
        "source_index_csv": str(index_csv.resolve()),
        "source_index_sha256": _sha256_file(index_csv),
        "image_count": len(labels),
        "batch_size": batch_size,
        "seed": seed,
        "dropped_tail": len(labels) - usable,
        "ordering": "paths sorted lexicographically, matching GALP image-ID construction",
        "contiguous": _batch_label_stats(labels, contiguous_order, batch_size),
        "random": _batch_label_stats(labels, random_order, batch_size),
    }
