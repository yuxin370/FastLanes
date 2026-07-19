#!/usr/bin/env python3
"""Shared, dependency-free helpers for the four-pipeline system benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence


CONTRACT_SCHEMA = "galp_system_benchmark_contract_v2"
MANIFEST_SCHEMA = "galp_system_benchmark_manifest_v2"
RESULT_SCHEMA = "galp_system_benchmark_result_v2"
PIPELINES = ("galp", "rgbnomore", "dali", "pytorch")
CONTRACT_PIPELINES = PIPELINES + ("galp_legacy",)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def fingerprint_file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    for _ in range(3):
        before = file_identity(path)
        digest = sha256_file(path)
        after = file_identity(path)
        if before == after:
            return {
                "path": str(path),
                "size_bytes": after["size_bytes"],
                "sha256": digest,
                "file_identity": after,
            }
    raise RuntimeError(f"file changed repeatedly while hashing: {path}")


def verify_file_fingerprint(path: Path, expected: dict[str, Any], label: str) -> None:
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    current_identity = file_identity(path)
    require(
        current_identity["size_bytes"] == expected.get("size_bytes"),
        f"{label} size changed after contract creation: {path}",
    )
    if current_identity == expected.get("file_identity"):
        return
    current = fingerprint_file(path)
    require(
        current["sha256"] == expected.get("sha256"),
        f"{label} SHA-256 changed after contract creation: {path}",
    )


def galp_manifest_payloads(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    data = manifest_path.read_bytes()
    offset = 0

    def take(fmt: str) -> tuple[int, ...]:
        nonlocal offset
        size = struct.calcsize(fmt)
        require(offset + size <= len(data), f"truncated GALP shard manifest: {manifest_path}")
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values

    def take_string() -> str:
        nonlocal offset
        (size,) = take("<I")
        require(offset + size <= len(data), f"truncated GALP shard manifest string: {manifest_path}")
        raw = data[offset : offset + size]
        offset += size
        return raw.decode("utf-8")

    require(data[:8] == b"GJDCTSH1", f"unexpected GALP shard manifest format: {manifest_path}")
    offset = 8
    take("<IHIIQ")  # version, reserved, rowgroup geometry, image count
    (shard_count,) = take("<I")
    root = manifest_path.parent
    payloads: list[dict[str, Any]] = []
    for _ in range(shard_count):
        take("<IQIQQQII")
        fls_size, metadata_size = take("<QQ")
        fls_name = take_string()
        metadata_name = take_string()
        for kind, name, expected_size in (
            ("fls", fls_name, fls_size),
            ("metadata", metadata_name, metadata_size),
        ):
            path = (root / name).resolve()
            require(path.is_relative_to(root), f"GALP {kind} payload escapes manifest directory: {name}")
            payloads.append(
                {
                    "kind": kind,
                    "relative_path": path.relative_to(root).as_posix(),
                    "path": path,
                    "expected_size": expected_size,
                }
            )
    require(offset == len(data), f"GALP shard manifest has trailing bytes: {manifest_path}")
    return payloads


def cached_file_fingerprints(
    files: Sequence[dict[str, Any]],
    cache_path: Path,
    *,
    cache_format: str,
    allow_hash_misses: bool = True,
) -> list[dict[str, Any]]:
    cached_by_path: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        cached = read_json(cache_path)
        if isinstance(cached, dict) and cached.get("format") == cache_format:
            cached_by_path = {
                str(item.get("path")): item
                for item in cached.get("files", [])
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            }

    fingerprints: list[dict[str, Any]] = []
    cache_changed = False
    for spec in files:
        path = Path(spec["path"]).resolve()
        identity = file_identity(path)
        expected_size = int(spec.get("expected_size", identity["size_bytes"]))
        require(identity["size_bytes"] == expected_size, f"GALP payload size disagrees with manifest: {path}")
        cached = cached_by_path.get(str(path))
        if (
            cached is not None
            and cached.get("file_identity") == identity
            and isinstance(cached.get("sha256"), str)
            and len(cached["sha256"]) == 64
        ):
            fingerprint = dict(cached)
        else:
            require(
                allow_hash_misses,
                f"payload fingerprint cache is missing or stale for {path}; "
                "regenerate it during dataset preparation or request an explicit refresh",
            )
            fingerprint = fingerprint_file(path)
            cache_changed = True
        fingerprint["kind"] = spec.get("kind", "payload")
        fingerprint["relative_path"] = spec.get("relative_path", path.name)
        fingerprints.append(fingerprint)

    if cache_changed or not cache_path.is_file():
        write_json(cache_path, {"format": cache_format, "files": fingerprints})
    return fingerprints


def _require_int(mapping: dict[str, Any], key: str, minimum: int) -> int:
    value = mapping.get(key)
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, f"{key} must be >= {minimum}")
    return int(value)


def load_contract(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    require(isinstance(payload, dict), "contract must be a JSON object")
    require(payload.get("schema_version") == CONTRACT_SCHEMA, f"contract schema must be {CONTRACT_SCHEMA}")
    for section in ("dataset", "execution", "preprocess", "models", "pipelines", "timing", "semantic_validation"):
        require(isinstance(payload.get(section), dict), f"contract.{section} must be an object")

    execution = payload["execution"]
    _require_int(execution, "batch_size", 1)
    _require_int(execution, "workers", 0)
    _require_int(execution, "warmup_batches", 0)
    _require_int(execution, "measurement_batches", 1)
    _require_int(execution, "repeats", 1)
    _require_int(execution, "seed", 0)
    require(execution.get("precision") in ("fp32", "amp_fp16", "amp_bf16"), "unsupported execution.precision")
    require(isinstance(execution.get("device"), str) and execution["device"], "execution.device must be set")
    require(execution.get("drop_last") is True, "the v1 contract requires execution.drop_last=true")

    manifest_path = Path(payload["dataset"].get("sample_manifest", ""))
    require(str(manifest_path), "dataset.sample_manifest must be set")
    require(isinstance(payload["dataset"].get("manifest_sha256"), str), "dataset.manifest_sha256 must be set")

    configured = payload["pipelines"].get("enabled")
    require(isinstance(configured, list) and configured, "pipelines.enabled must be a non-empty list")
    require(len(set(configured)) == len(configured), "pipelines.enabled contains duplicates")
    require(all(item in CONTRACT_PIPELINES for item in configured), f"pipelines.enabled must be a subset of {CONTRACT_PIPELINES}")
    if "galp" in configured:
        galp = payload["pipelines"].get("galp")
        require(isinstance(galp, dict), "pipelines.galp must be an object")
        require(isinstance(galp.get("manifest_fingerprint"), dict), "GALP manifest fingerprint is missing")
        fingerprints = galp.get("payload_fingerprints")
        require(isinstance(fingerprints, list) and fingerprints, "GALP payload fingerprints are missing")
        for index, fingerprint in enumerate(fingerprints):
            require(isinstance(fingerprint, dict), f"GALP payload fingerprint {index} must be an object")
            require(
                isinstance(fingerprint.get("sha256"), str) and len(fingerprint["sha256"]) == 64,
                f"GALP payload fingerprint {index} has bad SHA-256",
            )

    for domain in ("rgb", "dct"):
        model = payload["models"].get(domain)
        require(isinstance(model, dict), f"models.{domain} must be an object")
        for key in ("architecture", "checkpoint", "checkpoint_sha256", "input_domain", "recipe_id"):
            require(isinstance(model.get(key), str) and model[key], f"models.{domain}.{key} must be set")

    require(
        payload["timing"].get("boundary") == "steady_state_batch_request_to_metrics_complete",
        "unsupported timing boundary",
    )
    require(payload["timing"].get("cuda_sync_per_batch") is True, "v2 requires per-batch CUDA synchronization")
    require(
        payload["timing"].get("cuda_sync_scope") == "model_stream_only",
        "v2 requires model-stream completion without draining next-batch preprocessing",
    )
    return payload


def load_sample_manifest(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path)
    require(isinstance(payload, dict), "sample manifest must be a JSON object")
    require(payload.get("schema_version") == MANIFEST_SCHEMA, f"sample manifest schema must be {MANIFEST_SCHEMA}")
    samples = payload.get("samples")
    require(isinstance(samples, list) and samples, "sample manifest must contain samples")
    actual_hash = sha256_json(payload)
    if expected_sha256 is not None:
        require(actual_hash == expected_sha256, f"sample manifest SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}")
    for ordinal, sample in enumerate(samples):
        require(isinstance(sample, dict), f"sample {ordinal} must be an object")
        require(sample.get("ordinal") == ordinal, f"sample ordinal mismatch at {ordinal}")
        require(isinstance(sample.get("sample_id"), str) and sample["sample_id"], f"sample {ordinal} has no sample_id")
        require(isinstance(sample.get("path"), str) and sample["path"], f"sample {ordinal} has no path")
        require(isinstance(sample.get("label"), int) and 0 <= sample["label"] < 1000, f"sample {ordinal} has bad label")
        require(isinstance(sample.get("galp_image_id"), int) and sample["galp_image_id"] >= 0, f"sample {ordinal} has bad GALP image id")
        require(isinstance(sample.get("size_bytes"), int) and sample["size_bytes"] > 0, f"sample {ordinal} has bad size")
        require(sample.get("jpeg_sampling") in ("4:2:0", "4:4:4"), f"sample {ordinal} is outside the shared JPEG sampling domain")
        require(isinstance(sample.get("sha256"), str) and len(sample["sha256"]) == 64, f"sample {ordinal} has bad SHA-256")
        identity = sample.get("file_identity")
        require(isinstance(identity, dict), f"sample {ordinal} has no file identity snapshot")
        for key in ("device", "inode", "size_bytes", "mtime_ns", "ctime_ns"):
            require(isinstance(identity.get(key), int), f"sample {ordinal} has bad file identity field {key}")
        verify_file_fingerprint(Path(sample["path"]), sample, f"sample {ordinal}")
    return payload, samples


def measured_samples(samples: Sequence[dict[str, Any]], batch_size: int, warmup_batches: int, measurement_batches: int) -> list[dict[str, Any]]:
    begin = batch_size * warmup_batches
    end = begin + batch_size * measurement_batches
    require(len(samples) >= end, f"manifest has {len(samples)} samples but contract needs {end}")
    return list(samples[begin:end])


def batches(samples: Sequence[dict[str, Any]], batch_size: int, count: int, *, offset_batches: int = 0) -> Iterable[list[dict[str, Any]]]:
    begin = offset_batches * batch_size
    for batch_index in range(count):
        start = begin + batch_index * batch_size
        batch = list(samples[start : start + batch_size])
        require(len(batch) == batch_size, f"incomplete batch {batch_index}: expected {batch_size}, got {len(batch)}")
        yield batch


def sample_trace(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {"ordinal": item["ordinal"], "sample_id": item["sample_id"], "label": item["label"]}
        for item in samples
    ]
    return {
        "count": len(rows),
        "sha256": sha256_json(rows),
        "first": rows[:8],
        "last": rows[-8:] if len(rows) > 8 else rows,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    require(values, "cannot compute a percentile of an empty sequence")
    require(0.0 <= quantile <= 1.0, "quantile must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, float]:
    require(values, "cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    mean = statistics.fmean(numeric)
    return {
        "count": len(numeric),
        "mean": mean,
        "cv_population": statistics.pstdev(numeric) / abs(mean) if mean else 0.0,
        "min": min(numeric),
        "p50": percentile(numeric, 0.50),
        "p90": percentile(numeric, 0.90),
        "p95": percentile(numeric, 0.95),
        "p99": percentile(numeric, 0.99),
        "max": max(numeric),
    }


def normalize_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"checkpoint does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def source_tree_metadata(root: Path, relative_files: Sequence[str]) -> dict[str, Any]:
    root = root.resolve()
    unique_relative_files = list(dict.fromkeys(relative_files))

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()

    files: dict[str, str | None] = {}
    for relative in unique_relative_files:
        path = root / relative
        files[relative] = sha256_file(path) if path.is_file() else None
    status = git("status", "--porcelain")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    runtime_status = git("status", "--porcelain", "--", *unique_relative_files)
    return {
        "root": str(root),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "git_tracked_dirty": bool(tracked_status),
        "git_tracked_status": tracked_status.splitlines(),
        "benchmark_source_clean": not bool(runtime_status),
        "runtime_git_status": runtime_status.splitlines(),
        "tracked_diff_sha256": sha256_json(git("diff", "--binary")),
        "runtime_diff_sha256": sha256_json(git("diff", "HEAD", "--binary", "--", *unique_relative_files)),
        "runtime_file_sha256": files,
    }
