#!/usr/bin/env python3
"""Build the final retokenization operating-point table from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from galp.experiments.dct_retokenization.compute import mac_table


def read_rows(directory: Path) -> list[dict[str, str]]:
    path = directory.resolve() / "accuracy_table.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["configuration_id"]: row for row in rows}


def percent(row: dict[str, str] | None, name: str = "top1_percent") -> float | None:
    return None if row is None else float(row[name])


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def optional_rows(directory: Path | None) -> dict[str, dict[str, str]]:
    if directory is None:
        return {}
    return keyed(read_rows(directory))


def benchmark_rows(directory: Path | None) -> dict[tuple[str, int], dict[str, str]]:
    if directory is None:
        return {}
    with (directory.resolve() / "benchmark.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["benchmark"], int(row["tokens"])): row for row in rows}


def token_penalties(
    rows: dict[str, dict[str, str]],
    *,
    k64_id: str,
    k32_id: str,
    baseline: float,
    k32_n196: float,
) -> dict[str, float | None]:
    k64 = percent(rows.get(k64_id))
    k32 = percent(rows.get(k32_id))
    d64 = None if k64 is None else k64 - baseline
    d32 = None if k32 is None else k32 - k32_n196
    return {
        "d64_token_penalty_pp": d64,
        "d32_token_penalty_pp": d32,
        "exploratory_interaction_pp": None if d64 is None or d32 is None else d32 - d64,
    }


def run(args: argparse.Namespace) -> int:
    frozen = keyed(read_rows(args.frozen_dir))
    adapter = optional_rows(args.adapter_eval_dir)
    short_ft = optional_rows(args.short_ft_eval_dir)
    calibration = optional_rows(args.calibration_eval_dir)
    benchmarks = benchmark_rows(args.benchmark_dir)
    macs = {int(row["tokens"]): row for row in mac_table()}

    baseline = percent(frozen["k64_n196"])
    k32_n196 = percent(frozen["k32_n196"])
    assert baseline is not None and k32_n196 is not None

    configurations = [
        (64, 196, "k64_n196", "k64_n196", "k64_n196"),
        (32, 196, "k32_n196", "k32_n196", "k32_n196"),
        (64, 98, "k64_n98_width", "adapter_k64_n98_width", "short_ft_k64_n98_width"),
        (32, 98, "k32_n98_width", "adapter_k32_n98_width", "short_ft_k32_n98_width"),
        (32, 49, "k32_n49", "adapter_k32_n49", "short_ft_k32_n49"),
    ]
    final_rows: list[dict[str, Any]] = []
    for k, n, frozen_id, adapter_id, ft_id in configurations:
        frozen_top1 = percent(frozen.get(frozen_id))
        adapter_top1 = percent(adapter.get(adapter_id))
        short_top1 = percent(short_ft.get(ft_id))
        selected = short_top1 if short_top1 is not None else adapter_top1
        same_k = baseline if k == 64 else k32_n196
        benchmark = benchmarks.get(("prototype_full_model_forward", n))
        final_rows.append(
            {
                "k": k,
                "n": n,
                "frozen_top1_percent": frozen_top1,
                "adapter_top1_percent": adapter_top1,
                "calibration_top1_percent": percent(calibration.get(f"calibration_k{k}_n{n}_width")),
                "short_ft_top1_percent": short_top1,
                "delta_coefficient_pp": same_k - baseline,
                "delta_token_after_short_ft_pp": None if selected is None else selected - same_k,
                "prototype_total_gmac": float(macs[n]["prototype_total_gmac"]),
                "mac_relative_to_n196": float(macs[n]["relative_to_n196"]),
                "prototype_median_images_per_second": (
                    None if benchmark is None else float(benchmark["median_images_per_second"])
                ),
            }
        )

    stage_penalties = {
        "frozen_width": token_penalties(
            frozen,
            k64_id="k64_n98_width",
            k32_id="k32_n98_width",
            baseline=baseline,
            k32_n196=k32_n196,
        ),
        "adapter": token_penalties(
            adapter,
            k64_id="adapter_k64_n98_width",
            k32_id="adapter_k32_n98_width",
            baseline=baseline,
            k32_n196=k32_n196,
        ),
        "short_ft": token_penalties(
            short_ft,
            k64_id="short_ft_k64_n98_width",
            k32_id="short_ft_k32_n98_width",
            baseline=baseline,
            k32_n196=k32_n196,
        ),
    }
    k32_ft = percent(short_ft.get("short_ft_k32_n98_width"))
    final_penalties = stage_penalties["short_ft"]
    if k32_ft is None:
        verdict = "pending"
    else:
        loss = baseline - k32_ft
        verdict = "Strong GO" if loss <= 0.3 else "Promising" if loss <= 0.6 else "Weak" if loss > 1.0 else "intermediate"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "final_operating_points.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader(); writer.writerows(final_rows)
    summary = {
        "baseline_k64_n196_top1_percent": baseline,
        "k32_n196_top1_percent": k32_n196,
        "stage_penalties": stage_penalties,
        "verdict": verdict,
        "interaction_warning": "A single seed is exploratory and does not establish a frequency-redundancy claim.",
    }
    (output_dir / "penalties_and_verdict.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Retokenization result summary",
        "",
        "| K | N | Frozen | Adapter | Short FT | Δcoeff | Δtoken | MAC rel. | GPU img/s |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['k']} | {row['n']} | {fmt(row['frozen_top1_percent'])} | "
            f"{fmt(row['adapter_top1_percent'])} | {fmt(row['short_ft_top1_percent'])} | "
            f"{fmt(row['delta_coefficient_pp'])} pp | {fmt(row['delta_token_after_short_ft_pp'])} pp | "
            f"{fmt(row['mac_relative_to_n196'], 4)}x | "
            f"{fmt(row['prototype_median_images_per_second'], 1)} |"
        )
    lines.extend(
        [
            "",
            f"- final D64: {fmt(final_penalties['d64_token_penalty_pp'])} pp",
            f"- final D32: {fmt(final_penalties['d32_token_penalty_pp'])} pp",
            "- final exploratory I = D32 - D64: "
            f"{fmt(final_penalties['exploratory_interaction_pp'])} pp",
            f"- decision: {verdict}",
            "",
            "The interaction is descriptive for one matched seed and is not a causal redundancy claim.",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--adapter-eval-dir", type=Path)
    parser.add_argument("--short-ft-eval-dir", type=Path)
    parser.add_argument("--calibration-eval-dir", type=Path)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
