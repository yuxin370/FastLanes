#!/usr/bin/env python3
"""Canonical artifact IO, hashing, and source provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": stat.st_size,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and process.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {process.stderr.strip()}")
    return process.stdout


def repository_provenance(root: Path, runtime_files: Iterable[Path] = ()) -> dict[str, Any]:
    root = root.resolve()
    revision = _git(root, "rev-parse", "HEAD").strip()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = _git(root, "diff", "--binary", "HEAD", "--")
    runtime_paths = [path.resolve() for path in runtime_files]
    runtime_records: list[dict[str, Any]] = []
    for resolved in runtime_paths:
        if resolved.is_file():
            runtime_records.append(file_record(resolved))
    untracked_records: list[dict[str, Any]] = []
    untracked_relative = {line[3:] for line in status.splitlines() if line.startswith("?? ")}
    for candidate in runtime_paths:
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative in untracked_relative and candidate.is_file():
            untracked_records.append(file_record(candidate))
    return {
        "root": str(root),
        "head_revision": revision,
        "dirty": bool(status.strip()),
        "git_status": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(tracked_diff.encode("utf-8")),
        "runtime_files": sorted(runtime_records, key=lambda item: item["path"]),
        "untracked_runtime_files": sorted(untracked_records, key=lambda item: item["path"]),
    }


def runtime_metadata() -> dict[str, Any]:
    def safe_environment() -> dict[str, str]:
        redacted_markers = (
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "PASSWD",
            "API_KEY",
            "PRIVATE_KEY",
            "CREDENTIAL",
        )
        return {
            name: ("<redacted>" if any(marker in name.upper() for marker in redacted_markers) else value)
            for name, value in sorted(os.environ.items())
        }

    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "environment": safe_environment(),
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            result["cuda_device"] = {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": [properties.major, properties.minor],
            }
    except Exception as error:  # pragma: no cover - used for metadata, not correctness
        result["torch_error"] = f"{type(error).__name__}: {error}"
    try:
        import nvidia.dali

        result["dali"] = nvidia.dali.__version__
    except Exception as error:  # pragma: no cover - optional runtime
        result["dali_error"] = f"{type(error).__name__}: {error}"
    return result


def tensor_state_sha256(state: dict[str, Any]) -> str:
    """Hash tensor state by names, dtypes, shapes, and exact CPU bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json_bytes(list(tensor.shape)))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def nested_state_sha256(value: Any) -> str:
    """Hash nested optimizer/scheduler state without relying on pickle bytes."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None  # type: ignore[assignment]
        if torch is not None and torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json_bytes(list(tensor.shape)))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda value: str(value)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"list" if isinstance(item, list) else b"tuple")
            for child in item:
                visit(child)
        elif isinstance(item, bytes):
            digest.update(b"bytes")
            digest.update(item)
        else:
            digest.update(canonical_json_bytes(item))

    visit(value)
    return digest.hexdigest()


def write_artifact_hashes(output_dir: Path, excluded: Iterable[str] = ("artifact_hashes.json",)) -> dict[str, Any]:
    excluded_set = set(excluded)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in excluded_set:
            continue
        records[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    payload = {"algorithm": "sha256", "files": records}
    write_json(output_dir / "artifact_hashes.json", payload)
    return payload


def verify_artifact_hashes(output_dir: Path, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, expected in payload.get("files", {}).items():
        path = output_dir / name
        if not path.is_file():
            errors.append(f"artifact missing: {name}")
            continue
        actual = sha256_file(path)
        if actual != expected.get("sha256"):
            errors.append(f"artifact hash mismatch: {name}")
        if path.stat().st_size != expected.get("size_bytes"):
            errors.append(f"artifact size mismatch: {name}")
    return errors
