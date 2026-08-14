#!/usr/bin/env python3
"""Plot live validation progress for the four premixed PLS conditions.

The script reads the current ``metrics.jsonl`` files on every invocation.  It
does not require ``final_result.json`` and therefore works for paused, failed,
and still-running jobs.  A partially written JSONL tail is ignored so the
script can safely run while training is appending metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .matrix import CORE_CONDITION_IDS
from .report import COLORS, _line_chart


SCHEMA_VERSION = "galp-pls-premixed-progress-plot-v1"
DEFAULT_EXPERIMENT_ROOT = Path(
    os.environ.get(
        "PLS_EXPERIMENT_ROOT",
        "/mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811",
    )
)
CONDITION_LABELS = {
    "A0": "A0: sample/global",
    "A1": "A1: PLS/global",
    "B2": "B2: sample/closed-M4",
    "B6": "B6: PLS/closed-M4",
}
CSV_FIELDS = (
    "condition",
    "seed",
    "epoch",
    "optimizer_update",
    "processed_images",
    "validation_top1",
    "validation_top5",
    "validation_loss",
    "validation_latency_seconds",
    "validation_samples",
    "source_metrics",
)


def default_run_paths(experiment_root: Path, seed: int) -> dict[str, Path]:
    """Return the registered primary premixed run paths for one seed."""

    seed_name = f"seed_{seed}"
    seed_root = experiment_root / "seed_first" / seed_name
    common = seed_root / "premixed_4090" / "runs"
    return {
        "A0": common / "A0" / seed_name,
        "A1": seed_root / "premixed_pro6000_a1" / "runs" / "A1" / seed_name,
        "B2": common / "B2" / seed_name,
        "B6": common / "B6" / seed_name,
    }


def parse_run_overrides(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``--run CONDITION=PATH`` overrides."""

    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --run {value!r}; expected CONDITION=PATH")
        condition, raw_path = value.split("=", 1)
        condition = condition.strip().upper()
        if condition not in CORE_CONDITION_IDS:
            raise ValueError(
                f"invalid --run condition {condition!r}; expected {CORE_CONDITION_IDS}"
            )
        if condition in result:
            raise ValueError(f"duplicate --run override for {condition}")
        if not raw_path.strip():
            raise ValueError(f"empty path in --run {value!r}")
        result[condition] = Path(raw_path).expanduser()
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_validation_metrics(
    run_dir: Path,
    *,
    condition: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and deduplicate the currently available validation records."""

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return [], {
            "condition": condition,
            "seed": seed,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "metrics_exists": False,
            "ignored_json_lines": 0,
            "duplicate_validation_epochs": 0,
            "run_status": _read_json(run_dir / "run_status.json"),
        }

    by_epoch: dict[int, dict[str, Any]] = {}
    ignored = 0
    duplicates = 0
    with metrics_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                ignored += 1
                continue
            if value.get("record_type") != "validation":
                continue
            try:
                epoch = int(value["epoch"])
                observed_seed = int(value["seed"])
                observed_condition = str(value["condition"])
                record = {
                    "condition": observed_condition,
                    "seed": observed_seed,
                    "epoch": epoch,
                    "optimizer_update": int(value["optimizer_update"]),
                    "processed_images": int(value["processed_images"]),
                    "validation_top1": float(value["validation_top1"]),
                    "validation_top5": float(value["validation_top5"]),
                    "validation_loss": float(value["validation_loss"]),
                    "validation_latency_seconds": float(
                        value.get("validation_latency_seconds", 0.0)
                    ),
                    "validation_samples": int(value.get("validation_samples", 0)),
                    "source_metrics": str(metrics_path),
                    "source_line": line_number,
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid validation record at {metrics_path}:{line_number}: {error}"
                ) from error
            if observed_condition != condition or observed_seed != seed:
                raise ValueError(
                    f"validation identity mismatch at {metrics_path}:{line_number}: "
                    f"expected {condition}/seed {seed}, observed "
                    f"{observed_condition}/seed {observed_seed}"
                )
            if epoch in by_epoch:
                duplicates += 1
            # Resume can legitimately rewrite a later record.  The final valid
            # occurrence is the authoritative live value for that epoch.
            by_epoch[epoch] = record

    records = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    stat = metrics_path.stat()
    status = _read_json(run_dir / "run_status.json")
    metadata = {
        "condition": condition,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_exists": True,
        "metrics_size_bytes": stat.st_size,
        "metrics_mtime_unix": stat.st_mtime,
        "ignored_json_lines": ignored,
        "duplicate_validation_epochs": duplicates,
        "validation_point_count": len(records),
        "latest_epoch": records[-1]["epoch"] if records else None,
        "latest_validation": records[-1] if records else None,
        "run_status": status,
    }
    return records, metadata


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _x_axis(record: Mapping[str, Any], choice: str) -> float:
    if choice == "epoch":
        return float(record["epoch"])
    if choice == "processed-images":
        return float(record["processed_images"]) / 1_000_000.0
    if choice == "optimizer-update":
        return float(record["optimizer_update"])
    raise ValueError(f"unknown x-axis {choice!r}")


def _x_label(choice: str) -> str:
    return {
        "epoch": "Epoch",
        "processed-images": "Processed images (millions)",
        "optimizer-update": "Optimizer update",
    }[choice]


def _atomic_line_chart(path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        _line_chart(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _combine_charts(paths: Sequence[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        combined = Image.new("RGB", (width, height), "white")
        offset = 0
        for image in images:
            combined.paste(image, (0, offset))
            offset += image.height
        temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
        try:
            combined.save(temporary)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    finally:
        for image in images:
            image.close()


def generate_plots(
    run_paths: Mapping[str, Path],
    output_dir: Path,
    *,
    seed: int,
    x_axis: str = "epoch",
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Read current records, write audit tables, and atomically refresh plots."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for condition in CORE_CONDITION_IDS:
        records, metadata = read_validation_metrics(
            run_paths[condition], condition=condition, seed=seed
        )
        records_by_condition[condition] = records
        sources[condition] = metadata

    missing = [condition for condition, rows in records_by_condition.items() if not rows]
    if missing and not allow_missing:
        raise ValueError(
            f"no validation points for {missing}; pass --allow-missing to plot partial coverage"
        )
    if len(missing) == len(CORE_CONDITION_IDS):
        raise ValueError("none of the four premixed runs has a validation point")

    all_rows = [
        row
        for condition in CORE_CONDITION_IDS
        for row in records_by_condition[condition]
    ]
    _atomic_csv(output_dir / "premixed_validation_progress.csv", all_rows)

    chart_specs = (
        (
            "validation_top1",
            "premixed_top1.png",
            "Premixed PLS validation top-1 progress",
            "Validation top-1 (%)",
        ),
        (
            "validation_top5",
            "premixed_top5.png",
            "Premixed PLS validation top-5 progress",
            "Validation top-5 (%)",
        ),
        (
            "validation_loss",
            "premixed_validation_loss.png",
            "Premixed PLS validation loss progress",
            "Validation loss",
        ),
    )
    chart_paths: list[Path] = []
    for endpoint, filename, title, ylabel in chart_specs:
        path = output_dir / filename
        series = []
        for condition in CORE_CONDITION_IDS:
            rows = records_by_condition[condition]
            if not rows:
                continue
            series.append(
                {
                    "label": CONDITION_LABELS[condition],
                    "color": COLORS[condition],
                    "width": 4,
                    "markers": True,
                    "points": [
                        (_x_axis(row, x_axis), float(row[endpoint])) for row in rows
                    ],
                }
            )
        _atomic_line_chart(
            path,
            title=f"{title} (seed {seed}; live milestone data)",
            xlabel=_x_label(x_axis),
            ylabel=ylabel,
            series=series,
        )
        chart_paths.append(path)

    combined_path = output_dir / "premixed_validation_curves.png"
    _combine_charts(chart_paths, combined_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "seed": seed,
        "x_axis": x_axis,
        "conditions": list(CORE_CONDITION_IDS),
        "missing_conditions": missing,
        "sources": sources,
        "outputs": {
            "combined_plot": str(combined_path),
            "top1_plot": str(chart_paths[0]),
            "top5_plot": str(chart_paths[1]),
            "validation_loss_plot": str(chart_paths[2]),
            "validation_csv": str(output_dir / "premixed_validation_progress.csv"),
        },
        "claim_boundary": (
            "Live single-seed milestone curves; differing last epochs are retained. "
            "These are not final 300-epoch or multi-seed results."
        ),
    }
    _atomic_json(output_dir / "premixed_progress_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="PLS experiment root (default: PLS_EXPERIMENT_ROOT or current NVMe root)",
    )
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="override a condition run directory; repeat for multiple conditions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: EXPERIMENT_ROOT/plots/premixed_progress/seed_SEED",
    )
    parser.add_argument(
        "--x-axis",
        choices=("epoch", "processed-images", "optimizer-update"),
        default="epoch",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="plot available conditions when one or more metrics files are absent",
    )
    args = parser.parse_args(argv)

    experiment_root = args.experiment_root.expanduser().resolve()
    run_paths = default_run_paths(experiment_root, args.seed)
    run_paths.update(parse_run_overrides(args.run))
    output_dir = args.output_dir or (
        experiment_root / "plots" / "premixed_progress" / f"seed_{args.seed}"
    )
    result = generate_plots(
        run_paths,
        output_dir,
        seed=args.seed,
        x_axis=args.x_axis,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
