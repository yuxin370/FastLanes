#!/usr/bin/env python3
"""Create a strict single-seed fixed-budget report for Premixed PLS runs.

Unlike the live plotting command, this report requires all four conditions to
have the exact requested validation record and permanent checkpoint.  It is a
single-seed diagnostic report: effects are reported as individual paired
contrasts and never receive a confidence interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .matrix import CORE_CONDITION_IDS
from .plot_premixed_progress import (
    CONDITION_LABELS,
    _atomic_line_chart,
    default_run_paths,
    parse_run_overrides,
    read_validation_metrics,
)
from .report import COLORS


SCHEMA_VERSION = "galp-pls-premixed-fixed-budget-report-v1"
CONTRACT_EQUAL_FIELDS = (
    "recipe_hash",
    "layout_hash",
    "initial_model_hash",
    "backend_implementation",
    "execution_mode",
    "epochs",
    "microbatch_size",
    "gradient_accumulation",
    "effective_update_batch",
)
EXPECTED_POLICIES = {
    "A0": ("per-sample", "global", None),
    "A1": ("per-pls", "global", None),
    "B2": ("per-sample", "closed-pool", 4),
    "B6": ("per-pls", "closed-pool", 4),
}
EFFECTS = (
    ("crop_global", "A1 - A0", {"A1": 1.0, "A0": -1.0}),
    ("crop_closed_pool", "B6 - B2", {"B6": 1.0, "B2": -1.0}),
    ("shuffle_per_sample", "B2 - A0", {"B2": 1.0, "A0": -1.0}),
    ("shuffle_per_pls", "B6 - A1", {"B6": 1.0, "A1": -1.0}),
    (
        "crop_main",
        "0.5 * [(A1 - A0) + (B6 - B2)]",
        {"A1": 0.5, "A0": -0.5, "B6": 0.5, "B2": -0.5},
    ),
    (
        "shuffle_main",
        "0.5 * [(B2 - A0) + (B6 - A1)]",
        {"B2": 0.5, "A0": -0.5, "B6": 0.5, "A1": -0.5},
    ),
    (
        "crop_x_shuffle",
        "B6 - B2 - A1 + A0",
        {"B6": 1.0, "B2": -1.0, "A1": -1.0, "A0": 1.0},
    ),
    ("combined_vs_standard", "B6 - A0", {"B6": 1.0, "A0": -1.0}),
)
ENDPOINTS = ("validation_top1", "validation_top5", "validation_loss")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def validate_contracts(run_paths: Mapping[str, Path], *, seed: int) -> dict[str, Any]:
    contracts: dict[str, dict[str, Any]] = {}
    for condition in CORE_CONDITION_IDS:
        path = run_paths[condition] / "condition_contract.json"
        contract = _read_json(path)
        if not contract:
            raise ValueError(f"missing or invalid condition contract: {path}")
        if contract.get("condition_id") != condition or int(contract.get("training_seed", -1)) != seed:
            raise ValueError(f"condition contract identity mismatch: {path}")
        expected_crop, expected_order, expected_m = EXPECTED_POLICIES[condition]
        observed = (
            contract.get("crop_policy"),
            contract.get("order_policy"),
            contract.get("segments_per_pool"),
        )
        if observed != (expected_crop, expected_order, expected_m):
            raise ValueError(f"condition policy mismatch for {condition}: {observed}")
        contracts[condition] = contract

    reference = contracts["A0"]
    for field in CONTRACT_EQUAL_FIELDS:
        values = {condition: contract.get(field) for condition, contract in contracts.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"non-whitelisted contract mismatch for {field}: {values}")
    return {
        "seed": seed,
        "equal_fields": {field: reference.get(field) for field in CONTRACT_EQUAL_FIELDS},
        "contracts": {
            condition: {
                "path": str((run_paths[condition] / "condition_contract.json").resolve()),
                "condition_hash": contract.get("condition_hash"),
                "crop_policy": contract.get("crop_policy"),
                "order_policy": contract.get("order_policy"),
                "segments_per_pool": contract.get("segments_per_pool"),
            }
            for condition, contract in contracts.items()
        },
    }


def collect_fixed_budget(
    run_paths: Mapping[str, Path],
    *,
    seed: int,
    epoch: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    if epoch <= 0:
        raise ValueError("fixed-budget epoch must be positive")
    contracts = validate_contracts(run_paths, seed=seed)
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    fixed: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for condition in CORE_CONDITION_IDS:
        run_dir = run_paths[condition].resolve()
        records, metadata = read_validation_metrics(run_dir, condition=condition, seed=seed)
        through = [record for record in records if int(record["epoch"]) <= epoch]
        exact = [record for record in records if int(record["epoch"]) == epoch]
        checkpoint = run_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        status = _read_json(run_dir / "run_status.json")
        completed = int(status.get("completed_epoch") or 0)
        if len(exact) != 1:
            raise ValueError(f"{condition}/seed {seed} lacks exact epoch-{epoch} validation")
        if not checkpoint.is_file():
            raise ValueError(f"{condition}/seed {seed} lacks {checkpoint.name}")
        if completed < epoch:
            raise ValueError(f"{condition}/seed {seed} completed epoch {completed}, below {epoch}")
        if not through or int(through[0]["epoch"]) != 0:
            raise ValueError(f"{condition}/seed {seed} lacks epoch-0 validation for AUC")
        records_by_condition[condition] = through
        fixed.append(dict(exact[0]))
        sources[condition] = {
            **metadata,
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "completed_epoch_at_report": completed,
        }
    return records_by_condition, fixed, {"contract_validation": contracts, "sources": sources}


def effect_rows(fixed: Sequence[Mapping[str, Any]], *, epoch: int, seed: int) -> list[dict[str, Any]]:
    by_condition = {str(row["condition"]): row for row in fixed}
    if set(by_condition) != set(CORE_CONDITION_IDS):
        raise ValueError(f"effects require exactly {CORE_CONDITION_IDS}")
    rows: list[dict[str, Any]] = []
    for effect_id, formula, weights in EFFECTS:
        for endpoint in ENDPOINTS:
            value = sum(float(by_condition[condition][endpoint]) * weight for condition, weight in weights.items())
            rows.append({
                "effect_id": effect_id,
                "formula": formula,
                "endpoint": endpoint,
                "seed": seed,
                "epoch": epoch,
                "single_seed_difference": value,
                "paired_seed_count": 1,
                "confidence_interval_available": False,
            })
    return rows


def partial_normalized_auc(records: Sequence[Mapping[str, Any]], endpoint: str = "validation_top1") -> float:
    if len(records) < 2:
        raise ValueError("partial AUC requires at least two validation points")
    ordered = sorted(records, key=lambda row: int(row["processed_images"]))
    horizon = float(ordered[-1]["processed_images"])
    if horizon <= 0:
        raise ValueError("partial AUC horizon must be positive")
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        x0 = float(left["processed_images"]) / horizon
        x1 = float(right["processed_images"]) / horizon
        y0 = float(left[endpoint])
        y1 = float(right[endpoint])
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def auc_rows(
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    epoch: int,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "seed": seed,
            "through_epoch": epoch,
            "metric": f"partial normalized AUC through epoch {epoch}",
            "validation_top1_auc": partial_normalized_auc(records_by_condition[condition]),
            "validation_point_count": len(records_by_condition[condition]),
        }
        for condition in CORE_CONDITION_IDS
    ]


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _plot_curves(
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    *,
    epoch: int,
    seed: int,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    specs = (
        ("validation_top1", "top1", "Validation top-1 (%)"),
        ("validation_top5", "top5", "Validation top-5 (%)"),
        ("validation_loss", "validation_loss", "Validation loss"),
    )
    for endpoint, label, ylabel in specs:
        path = output_dir / f"{label}_convergence_to_epoch_{epoch:03d}.png"
        _atomic_line_chart(
            path,
            title=f"Premixed fixed-budget {ylabel} through E{epoch} (seed {seed})",
            xlabel="Processed images (millions)",
            ylabel=ylabel,
            series=[{
                "label": CONDITION_LABELS[condition],
                "color": COLORS[condition],
                "width": 4,
                "markers": True,
                "points": [
                    (float(row["processed_images"]) / 1_000_000.0, float(row[endpoint]))
                    for row in records_by_condition[condition]
                ],
            } for condition in CORE_CONDITION_IDS],
        )
        outputs[label] = str(path)
    return outputs


def _plot_points(
    output_dir: Path,
    *,
    epoch: int,
    seed: int,
    fixed: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    aucs: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    by_condition = {str(row["condition"]): row for row in fixed}
    points_path = output_dir / f"fixed_budget_top1_epoch_{epoch:03d}.png"
    _atomic_line_chart(
        points_path,
        title=f"Premixed Top-1 at E{epoch} (single seed {seed})",
        xlabel="Condition index (A0, A1, B2, B6)",
        ylabel="Validation top-1 (%)",
        series=[{
            "label": "fixed-budget top-1",
            "color": (31, 119, 180),
            "width": 3,
            "markers": True,
            "points": [(float(index), float(by_condition[condition]["validation_top1"])) for index, condition in enumerate(CORE_CONDITION_IDS)],
        }],
    )
    top1_effects = [row for row in effects if row["endpoint"] == "validation_top1"]
    effects_path = output_dir / f"single_seed_effects_epoch_{epoch:03d}.png"
    _atomic_line_chart(
        effects_path,
        title=f"Premixed Top-1 effects at E{epoch} (single seed; no CI)",
        xlabel="Effect index (see CSV)",
        ylabel="Top-1 difference (percentage points)",
        series=[{
            "label": "single-seed difference",
            "color": (214, 39, 40),
            "width": 3,
            "markers": True,
            "points": [(float(index), float(row["single_seed_difference"])) for index, row in enumerate(top1_effects)],
        }],
    )
    auc_path = output_dir / f"partial_auc_epoch_{epoch:03d}.png"
    _atomic_line_chart(
        auc_path,
        title=f"Partial normalized Top-1 AUC through E{epoch}",
        xlabel="Condition index (A0, A1, B2, B6)",
        ylabel="Partial normalized Top-1 AUC",
        series=[{
            "label": "partial AUC",
            "color": (44, 160, 44),
            "width": 3,
            "markers": True,
            "points": [(float(index), float(row["validation_top1_auc"])) for index, row in enumerate(aucs)],
        }],
    )
    return {"fixed_top1": str(points_path), "effects": str(effects_path), "partial_auc": str(auc_path)}


def render_markdown(result: Mapping[str, Any]) -> str:
    epoch = int(result["epoch"])
    lines = [
        f"# Premixed PLS fixed-budget report: epoch {epoch}",
        "",
        f"Seed: `{result['seed']}`. This is a **single-seed milestone**, not final accuracy.",
        "",
        "## Fixed-budget metrics",
        "",
        "| Condition | Top-1 | Top-5 | Validation loss |",
        "|---|---:|---:|---:|",
    ]
    for row in result["fixed_budget_metrics"]:
        lines.append(
            f"| {row['condition']} | {float(row['validation_top1']):.3f} | "
            f"{float(row['validation_top5']):.3f} | {float(row['validation_loss']):.4f} |"
        )
    lines.extend([
        "",
        "## Top-1 single-seed effects",
        "",
        "| Effect | Formula | Difference (pp) |",
        "|---|---|---:|",
    ])
    for row in result["effects"]:
        if row["endpoint"] == "validation_top1":
            lines.append(
                f"| {row['effect_id']} | `{row['formula']}` | "
                f"{float(row['single_seed_difference']):+.3f} |"
            )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "- All values use the exact same epoch and seed across A0/A1/B2/B6.",
        "- No confidence interval is computed for one seed.",
        "- Partial AUC covers only the validation grid through this epoch.",
        "- These semantic-emulation results do not demonstrate physical byte reduction.",
        "",
    ])
    return "\n".join(lines)


def generate_report(
    run_paths: Mapping[str, Path],
    output_dir: Path,
    *,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, fixed, evidence = collect_fixed_budget(run_paths, seed=seed, epoch=epoch)
    effects = effect_rows(fixed, epoch=epoch, seed=seed)
    aucs = auc_rows(records, epoch=epoch, seed=seed)
    convergence = [row for condition in CORE_CONDITION_IDS for row in records[condition]]
    suffix = f"epoch_{epoch:03d}"
    _atomic_csv(output_dir / f"fixed_budget_metrics_{suffix}.csv", fixed)
    _atomic_csv(output_dir / f"convergence_curves_to_{suffix}.csv", convergence)
    _atomic_csv(output_dir / f"single_seed_effects_{suffix}.csv", effects)
    _atomic_csv(output_dir / f"partial_auc_{suffix}.csv", aucs)
    plots = {
        **_plot_curves(records, output_dir, epoch=epoch, seed=seed),
        **_plot_points(output_dir, epoch=epoch, seed=seed, fixed=fixed, effects=effects, aucs=aucs),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "seed": seed,
        "epoch": epoch,
        "condition_count": len(fixed),
        "complete": len(fixed) == len(CORE_CONDITION_IDS),
        "fixed_budget_metrics": fixed,
        "effects": effects,
        "partial_auc": aucs,
        "evidence": evidence,
        "plots": plots,
        "claim_boundary": "Single-seed fixed-budget milestone; no CI and not final 300-epoch accuracy.",
    }
    _atomic_json(output_dir / f"milestone_results_{suffix}.json", result)
    _atomic_bytes(output_dir / f"milestone_report_{suffix}.md", render_markdown(result).encode("utf-8"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args(argv)
    run_paths = default_run_paths(args.experiment_root.resolve(), args.seed)
    run_paths.update(parse_run_overrides(args.run))
    result = generate_report(run_paths, args.output_dir, seed=args.seed, epoch=args.epoch)
    print(json.dumps({
        "complete": result["complete"],
        "epoch": result["epoch"],
        "seed": result["seed"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
