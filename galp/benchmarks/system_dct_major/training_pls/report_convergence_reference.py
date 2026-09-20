#!/usr/bin/env python3
"""Compare a Native B6 convergence prefix with a standard DCT reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    NATIVE_PHYSICAL_BACKEND,
    STANDARD_DCT_BACKEND,
    blocking_code_identity,
    condition_identity_hash,
)
from .recipe import sha256_json
from .report import _line_chart, normalized_auc


REPORT_SCHEMA = "galp-dct-convergence-reference-report-v1"
PAIRED_IDENTITY_FIELDS = (
    "condition_id",
    "training_seed",
    "train_manifest_hash",
    "validation_manifest_hash",
    "model_id",
    "model_configuration",
    "model_input_contract",
    "model_source_provenance",
    "model_execution",
    "initial_model_hash",
    "recipe_hash",
    "optimizer_configuration",
    "weight_decay_configuration",
    "scheduler",
    "epochs",
    "total_optimizer_updates",
    "microbatch_size",
    "gradient_accumulation",
    "effective_update_batch",
    "mixup",
    "randaugment",
    "flip",
    "validation_preprocessing",
    "precision",
    "audit_policy",
    "checkpoint_schedule",
    "execution_device",
    "required_gpu_name_substring",
    "reference_topology",
    "execution_topology",
    "layout_hash",
    "physical_layout_plan_hash",
    "crop_policy",
    "crop_key_scope",
    "order_policy",
    "segments_per_pool",
    "pool_membership_digest",
    "sample_order_digest",
    "crop_key_digest",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid metrics JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"metrics row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _validate_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    expected_run_hash = sha256_json(
        {key: value for key, value in contract.items() if key != "run_manifest_hash"}
    )
    if contract.get("run_manifest_hash") != expected_run_hash:
        raise ValueError(f"run-manifest hash mismatch: {path}")
    if contract.get("condition_hash") != condition_identity_hash(contract):
        raise ValueError(f"condition identity hash mismatch: {path}")
    return contract


def _execution_backend(contract: Mapping[str, Any]) -> str:
    mode = str(contract.get("execution_mode", ""))
    if mode == "native_physical_pls":
        return NATIVE_PHYSICAL_BACKEND
    if mode == "standard_rgbnomore_dct":
        return STANDARD_DCT_BACKEND
    return mode


def _validation_by_epoch(
    metrics: Sequence[Mapping[str, Any]], max_epoch: int
) -> dict[int, dict[str, Any]]:
    result = {
        int(row["epoch"]): dict(row)
        for row in metrics
        if row.get("record_type") == "validation"
        and int(row.get("epoch", -1)) <= max_epoch
    }
    if not result:
        raise ValueError("run has no validation records in the requested prefix")
    return result


def _train_epoch_by_epoch(
    metrics: Sequence[Mapping[str, Any]], max_epoch: int
) -> dict[int, dict[str, Any]]:
    return {
        int(row["epoch"]): dict(row)
        for row in metrics
        if row.get("record_type") == "train"
        and row.get("scope") == "epoch"
        and int(row.get("epoch", -1)) <= max_epoch
    }


def build_report(
    *, native_run: Path, reference_run: Path, max_epoch: int = 50
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if max_epoch <= 0:
        raise ValueError("max_epoch must be positive")
    native_run = native_run.resolve()
    reference_run = reference_run.resolve()
    native_contract = _validate_contract(native_run / "run_manifest.json")
    reference_contract = _validate_contract(reference_run / "run_manifest.json")
    if _execution_backend(native_contract) != NATIVE_PHYSICAL_BACKEND:
        raise ValueError("--native-run is not a native physical B6 run")
    if _execution_backend(reference_contract) != STANDARD_DCT_BACKEND:
        raise ValueError("--reference-run is not a standard RGB-no-more DCT run")
    identity_differences = {
        field: {
            "native": native_contract.get(field),
            "reference": reference_contract.get(field),
        }
        for field in PAIRED_IDENTITY_FIELDS
        if native_contract.get(field) != reference_contract.get(field)
    }
    if identity_differences:
        raise ValueError(
            "native/reference paired training identity differs: "
            f"{sorted(identity_differences)}"
        )
    native_mapping_hash = str(
        native_contract.get("physical_execution", {}).get(
            "premixed_mapping_sha256", ""
        )
    )
    reference_mapping_hash = str(
        reference_contract.get("standard_dct_reference", {}).get(
            "premixed_mapping_sha256", ""
        )
    )
    if not native_mapping_hash or native_mapping_hash != reference_mapping_hash:
        raise ValueError(
            "native/reference premixed mapping SHA-256 differs: "
            f"native={native_mapping_hash!r}, reference={reference_mapping_hash!r}"
        )
    native_status = _read_json(native_run / "run_status.json")
    reference_status = _read_json(reference_run / "run_status.json")
    for label, status in (("native", native_status), ("reference", reference_status)):
        if int(status.get("completed_epoch", -1)) < max_epoch:
            raise ValueError(
                f"{label} run completed only epoch {status.get('completed_epoch')}; "
                f"epoch {max_epoch} is required"
            )
    native_metrics = _read_metrics(native_run / "metrics.jsonl")
    reference_metrics = _read_metrics(reference_run / "metrics.jsonl")
    native_validation = _validation_by_epoch(native_metrics, max_epoch)
    reference_validation = _validation_by_epoch(reference_metrics, max_epoch)
    common_epochs = sorted(set(native_validation) & set(reference_validation))
    if not common_epochs or common_epochs[-1] != max_epoch:
        raise ValueError(
            f"native/reference have no common validation record at epoch {max_epoch}"
        )
    rows: list[dict[str, Any]] = []
    for epoch in common_epochs:
        native = native_validation[epoch]
        reference = reference_validation[epoch]
        for field in ("optimizer_update", "processed_images"):
            if int(native[field]) != int(reference[field]):
                raise ValueError(f"epoch {epoch} {field} differs between paired runs")
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update": int(native["optimizer_update"]),
                "processed_images": int(native["processed_images"]),
                "native_top1": float(native["validation_top1"]),
                "reference_top1": float(reference["validation_top1"]),
                "top1_native_minus_reference": float(native["validation_top1"])
                - float(reference["validation_top1"]),
                "native_top5": float(native["validation_top5"]),
                "reference_top5": float(reference["validation_top5"]),
                "top5_native_minus_reference": float(native["validation_top5"])
                - float(reference["validation_top5"]),
                "native_validation_loss": float(native["validation_loss"]),
                "reference_validation_loss": float(reference["validation_loss"]),
                "validation_loss_native_minus_reference": float(
                    native["validation_loss"]
                )
                - float(reference["validation_loss"]),
            }
        )
    total_images = int(native_validation[max_epoch]["processed_images"])
    native_auc = normalized_auc(
        [native_validation[epoch] for epoch in common_epochs], total_images
    )
    reference_auc = normalized_auc(
        [reference_validation[epoch] for epoch in common_epochs], total_images
    )
    native_train = _train_epoch_by_epoch(native_metrics, max_epoch)
    reference_train = _train_epoch_by_epoch(reference_metrics, max_epoch)
    common_train_epochs = sorted(set(native_train) & set(reference_train))
    expected_train_epochs = list(range(1, max_epoch + 1))
    if common_train_epochs != expected_train_epochs:
        raise ValueError(
            "native/reference do not contain every paired training epoch in the "
            f"1..{max_epoch} prefix"
        )
    exact_digest_fields = ("sample_order_digest", "pool_membership_digest")
    schedule_digest_checks: dict[str, dict[str, Any]] = {
        field: {
            "status": "match"
            if all(
                native_train[epoch].get(field)
                == reference_train[epoch].get(field)
                for epoch in common_train_epochs
            )
            else "mismatch",
            "epochs_checked": len(common_train_epochs),
        }
        for field in exact_digest_fields
    }
    failed_digests = [
        field
        for field, observation in schedule_digest_checks.items()
        if observation["status"] != "match"
    ]
    if failed_digests:
        raise ValueError(
            "native/reference logical schedule digests differ: "
            f"{failed_digests}"
        )
    for field in (
        "crop_key_digest",
        "flip_key_digest",
        "randaugment_digest",
        "mixup_digest",
    ):
        schedule_digest_checks[field] = {
            "status": "not-comparable-native-owned",
            "native_marker": native_train[common_train_epochs[-1]].get(field),
            "reference_digest": reference_train[common_train_epochs[-1]].get(field),
            "contract_identity_locked": True,
        }
    native_code = blocking_code_identity(native_contract["code_version"])
    reference_code = blocking_code_identity(reference_contract["code_version"])
    report = {
        "schema_version": REPORT_SCHEMA,
        "comparison": "native-b6-vs-standard-rgbnomore-dct",
        "model_id": native_contract["model_id"],
        "seed": int(native_contract["training_seed"]),
        "max_epoch": max_epoch,
        "common_validation_epochs": common_epochs,
        "paired_identity_valid": True,
        "premixed_mapping_identity": {
            "sha256": native_mapping_hash,
            "status": "match",
        },
        "paired_identity_fields": list(PAIRED_IDENTITY_FIELDS),
        "runtime_source_identity_match": native_code == reference_code,
        "runtime_source_identities": {
            "native": native_code,
            "reference": reference_code,
        },
        "schedule_digest_checks": schedule_digest_checks,
        "normalized_top1_auc": {
            "native_b6": native_auc,
            "standard_dct": reference_auc,
            "native_minus_standard": native_auc - reference_auc,
        },
        "epoch_comparisons": rows,
        "thresholds_enforced": False,
        "strategy_selection_performed": False,
        "claim_boundary": (
            "Single-seed prefix convergence comparison. It can reveal divergence or "
            "curve displacement, but it does not estimate across-seed variance or "
            "establish statistical equivalence. No ViT-derived threshold is applied."
        ),
        "inputs": {
            "native_run": str(native_run),
            "reference_run": str(reference_run),
        },
    }
    return report, rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_report(
    output_dir: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("convergence_reference.json", "convergence_reference.csv"):
        if (output_dir / name).exists():
            raise ValueError(f"report output already exists: {output_dir / name}")
    _atomic_json(output_dir / "convergence_reference.json", report)
    fields = tuple(rows[0])
    csv_path = output_dir / "convergence_reference.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _line_chart(
        output_dir / "top1_convergence_reference.png",
        title="SwinV2 DCT convergence: Native B6 vs standard DCT",
        xlabel="Processed images",
        ylabel="Validation top-1 (%)",
        series=(
            {
                "label": "Native B6",
                "color": (31, 119, 180),
                "width": 4,
                "markers": True,
                "points": [
                    (row["processed_images"], row["native_top1"]) for row in rows
                ],
            },
            {
                "label": "Standard DCT",
                "color": (214, 39, 40),
                "width": 4,
                "markers": True,
                "points": [
                    (row["processed_images"], row["reference_top1"])
                    for row in rows
                ],
            },
        ),
    )
    _line_chart(
        output_dir / "validation_loss_reference.png",
        title="SwinV2 DCT validation loss: Native B6 vs standard DCT",
        xlabel="Processed images",
        ylabel="Validation loss",
        series=(
            {
                "label": "Native B6",
                "color": (31, 119, 180),
                "width": 4,
                "markers": True,
                "points": [
                    (row["processed_images"], row["native_validation_loss"])
                    for row in rows
                ],
            },
            {
                "label": "Standard DCT",
                "color": (214, 39, 40),
                "width": 4,
                "markers": True,
                "points": [
                    (row["processed_images"], row["reference_validation_loss"])
                    for row in rows
                ],
            },
        ),
    )
    auc = report["normalized_top1_auc"]
    final = rows[-1]
    lines = [
        "# SwinV2 standard-DCT convergence reference",
        "",
        (
            f"Model: `{report['model_id']}`; seed: `{report['seed']}`; "
            f"prefix: epoch `{report['max_epoch']}`."
        ),
        "",
        (
            "No ViT-derived pass/fail threshold is enforced. This is a "
            "single-seed diagnostic comparison."
        ),
        "",
        "| Metric at common endpoint | Native B6 | Standard DCT | Native - Standard |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Top-1 (%) | {final['native_top1']:.4f} | "
            f"{final['reference_top1']:.4f} | "
            f"{final['top1_native_minus_reference']:.4f} |"
        ),
        (
            f"| Top-5 (%) | {final['native_top5']:.4f} | "
            f"{final['reference_top5']:.4f} | "
            f"{final['top5_native_minus_reference']:.4f} |"
        ),
        (
            f"| Validation loss | {final['native_validation_loss']:.6f} | "
            f"{final['reference_validation_loss']:.6f} | "
            f"{final['validation_loss_native_minus_reference']:.6f} |"
        ),
        (
            f"| Normalized Top-1 AUC | {auc['native_b6']:.6f} | "
            f"{auc['standard_dct']:.6f} | "
            f"{auc['native_minus_standard']:.6f} |"
        ),
        "",
        f"Runtime source identity match: `{report['runtime_source_identity_match']}`.",
        "",
        report["claim_boundary"],
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epoch", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report, rows = build_report(
        native_run=args.native_run,
        reference_run=args.reference_run,
        max_epoch=args.max_epoch,
    )
    write_report(args.output_dir.resolve(), report, rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "max_epoch": report["max_epoch"],
                "runtime_source_identity_match": report[
                    "runtime_source_identity_match"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
