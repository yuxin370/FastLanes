#!/usr/bin/env python3
"""Prepare an auditable Epoch-2 profiling replay from a formal Epoch-1 checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REGISTERED_PIPELINES = ("d2", "d3", "pytorch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="completed formal equal-image benchmark root",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="planned profiling benchmark root with its own contract.json",
    )
    parser.add_argument(
        "--pipeline", choices=REGISTERED_PIPELINES, default="d2"
    )
    parser.add_argument("--checkpoint-epoch", type=int, default=1)
    return parser.parse_args(argv)


def _normalized_workload_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields that a bounded profiling replay is allowed to change."""

    normalized = copy.deepcopy(dict(contract))
    normalized.pop("contract_hash", None)
    profiling = normalized.pop("profiling", {})
    validation = normalized.get("validation")
    if isinstance(validation, dict):
        expected_validation_epochs = [0, 1, 2]
        if (
            bool(profiling.get("enabled", False))
            and bool(profiling.get("skip_profiled_epoch_validation", False))
            and int(profiling.get("epoch", -1)) in expected_validation_epochs
        ):
            expected_validation_epochs.remove(int(profiling["epoch"]))
        if validation.get("epochs") != expected_validation_epochs:
            raise ValueError(
                "validation epochs do not match the contract's profiling controls"
            )
        validation["epochs"] = [0, 1, 2]
    return normalized


def _validate_replay_contracts(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    pipeline: str,
) -> None:
    for label, contract in (("source", source), ("target", target)):
        pipelines = contract.get("pipelines")
        if not isinstance(pipelines, list) or pipeline not in pipelines:
            raise ValueError(
                f"{pipeline!r} is not registered in the {label} benchmark contract"
            )
    if _normalized_workload_contract(source) != _normalized_workload_contract(target):
        raise ValueError(
            "source and target workload contracts differ beyond profiling controls"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    target_contract_path = target_root / "contract.json"
    source_contract_path = source_root / "contract.json"
    if not source_contract_path.is_file():
        raise FileNotFoundError(source_contract_path)
    if not target_contract_path.is_file():
        raise FileNotFoundError(
            f"generate the profiling contract before preparing replay: {target_contract_path}"
        )
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    target_contract = json.loads(target_contract_path.read_text(encoding="utf-8"))
    profiling = target_contract.get("profiling", {})
    if not profiling.get("enabled") or int(profiling.get("epoch", -1)) != 2:
        raise ValueError("target contract must enable profiling for Epoch 2")
    _validate_replay_contracts(
        source_contract, target_contract, pipeline=args.pipeline
    )

    source_run = source_root / "runs" / args.pipeline
    target_run = target_root / "runs" / args.pipeline
    source_checkpoint = source_run / f"checkpoint_epoch_{args.checkpoint_epoch:03d}.pt"
    source_metrics = source_run / "metrics.jsonl"
    source_initial = source_root / "initial_rgb_state.pt"
    for path in (source_checkpoint, source_metrics, source_initial):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("completed_epoch", -1)) != args.checkpoint_epoch:
        raise ValueError("source checkpoint is not at the requested epoch boundary")
    if payload.get("contract_hash") != source_contract.get("contract_hash"):
        raise ValueError("source checkpoint does not belong to the source contract")
    if payload.get("pending_validation_epoch") is not None:
        raise ValueError("source checkpoint still has pending validation")
    if int(payload.get("gradient_accumulation_state", {}).get("samples", -1)) != 0:
        raise ValueError("source checkpoint is not at an accumulation boundary")

    target_run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_initial, target_root / "initial_rgb_state.pt")
    shutil.copy2(source_metrics, target_run / "metrics.jsonl")
    source_contract_hash = str(source_contract["contract_hash"])
    payload["contract_hash"] = str(target_contract["contract_hash"])
    temporary_checkpoint = target_run / ".latest.pt.tmp"
    torch.save(payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, target_run / "latest.pt")
    evidence = {
        "schema_version": "galp-equal-image-profile-replay-v1",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "pipeline": args.pipeline,
        "checkpoint_epoch": args.checkpoint_epoch,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "target_checkpoint_sha256": _sha256(target_run / "latest.pt"),
        "state_translation": "contract_hash only after normalized workload equality",
        "source_contract_hash": source_contract_hash,
        "target_contract_hash": target_contract["contract_hash"],
        "optimizer_update": int(payload["global_optimizer_update"]),
        "processed_images": int(payload["processed_image_count"]),
        "metrics_cursor": dict(payload["metrics_cursor"]),
        "scientific_result": False,
        "purpose": "bounded attribution trace; unprofiled full epoch remains authoritative",
    }
    _atomic_json(target_root / "profile_replay_origin.json", evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
