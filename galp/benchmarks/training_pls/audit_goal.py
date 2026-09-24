#!/usr/bin/env python3
"""Audit implementation and runtime evidence for the core PLS experiment.

The audit is deliberately read-only with respect to training runs.  It reads
the registered matrix plan instead of inventing run identities, tolerates a
partially written metrics tail, and distinguishes a recorded ``running`` state
from a process that is actually present on the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "galp-pls-goal-audit-v1"
MILESTONES = (10, 30, 50, 100, 300)
CORE_ARTIFACTS = (
    "matrix_execution_plan.json",
    "condition_execution_order.json",
    "recipe_contract.json",
    "analysis_contract.json",
    "condition_contract_diff.json",
    "environment.json",
    "run_status.csv",
    "failures.jsonl",
)
FINAL_REPORT_ARTIFACTS = (
    "convergence_curves.csv",
    "final_metrics.csv",
    "paired_effects.csv",
    "factorial_effects.csv",
    "model_results.json",
    "report.md",
    "top1_convergence.png",
    "top5_convergence.png",
    "train_loss_curves.png",
    "final_top1_by_seed.png",
    "paired_differences.png",
    "factorial_effects.png",
    "normalized_auc.png",
)
PHYSICAL_CANARY_ARTIFACTS = (
    "fls_tensor_equivalence.json",
    "fls_native_counters.csv",
    "crop_pushdown_evidence.json",
    "physical_io_evidence.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, hash_bytes: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if result["exists"]:
        stat = path.stat()
        result.update(size_bytes=stat.st_size, mtime_unix=stat.st_mtime)
        if hash_bytes:
            result["sha256"] = _sha256(path)
    return result


def _checkpoint_epochs(run_dir: Path) -> list[int]:
    values: list[int] = []
    for path in run_dir.glob("checkpoint_epoch_*.pt"):
        try:
            values.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(values))


def _validation_records(path: Path) -> tuple[dict[int, dict[str, Any]], int]:
    by_epoch: dict[int, dict[str, Any]] = {}
    malformed = 0
    try:
        source = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return by_epoch, malformed
    with source:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if value.get("record_type") != "validation":
                continue
            try:
                by_epoch[int(value["epoch"])] = value
            except (KeyError, TypeError, ValueError):
                malformed += 1
    return by_epoch, malformed


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    return next(
        (parsed for value in values if (parsed := _optional_int(value)) is not None),
        None,
    )


def active_output_processes() -> dict[str, list[int]]:
    """Return output directories named by active Python command lines."""

    result: dict[str, list[int]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return result
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().split(b"\0")
            argv = [part.decode("utf-8", errors="replace") for part in parts if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        try:
            raw_output = argv[argv.index("--output-dir") + 1]
        except (ValueError, IndexError):
            continue
        path = str(Path(raw_output).expanduser().resolve())
        result.setdefault(path, []).append(int(entry.name))
    return {key: sorted(value) for key, value in sorted(result.items())}


def process_scan_metadata() -> dict[str, Any]:
    """Describe whether this process can see the host PID namespace."""

    proc = Path("/proc")
    visible = sum(1 for entry in proc.iterdir() if entry.name.isdigit()) if proc.is_dir() else 0
    try:
        init_argv = (proc / "1/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, PermissionError):
        init_argv = ""
    limited = visible < 10 or "bwrap" in init_argv or "codex-linux-sandbox" in init_argv
    return {
        "visible_pid_count": visible,
        "pid_1_command": init_argv[:500],
        "scope": "limited-pid-namespace" if limited else "host-pid-namespace",
        "authoritative_for_host_activity": not limited,
    }


def scan_run(
    run_dir: Path,
    *,
    condition: str,
    seed: int,
    active: Mapping[str, Sequence[int]],
    process_scan_authoritative: bool = True,
) -> dict[str, Any]:
    resolved = run_dir.expanduser().resolve()
    status = _read_json(resolved / "run_status.json")
    pause = _read_json(resolved / "pause_result.json")
    final = _read_json(resolved / "final_result.json")
    final_checkpoint = (
        final.get("checkpoint")
        if isinstance(final.get("checkpoint"), Mapping)
        else {}
    )
    checkpoints = _checkpoint_epochs(resolved) if resolved.is_dir() else []
    validations, malformed = _validation_records(resolved / "metrics.jsonl")
    completed_candidates = (
        status.get("completed_epoch"),
        pause.get("completed_epoch"),
        final.get("completed_epoch"),
        final.get("final_epoch"),
        final.get("epoch"),
        final_checkpoint.get("final_epoch"),
        checkpoints[-1] if checkpoints else None,
    )
    completed = max(
        (
            parsed
            for value in completed_candidates
            if (parsed := _optional_int(value)) is not None
        ),
        default=0,
    )
    final_epoch = _first_int(
        final_checkpoint.get("final_epoch"),
        final.get("completed_epoch"),
        final.get("final_epoch"),
        final.get("epoch"),
    )
    final_condition = final.get("condition")
    final_seed = _optional_int(final.get("seed"))
    final_identity_matches = (
        final_condition == condition and final_seed == seed
    )
    final_result_valid = (
        bool(final)
        and final.get("state") == "completed"
        and final_identity_matches
        and final_epoch is not None
    )
    pids = list(active.get(str(resolved), ()))
    recorded_running_is_active: bool | None
    if status.get("state") != "running":
        recorded_running_is_active = None
    elif pids:
        recorded_running_is_active = True
    elif process_scan_authoritative:
        recorded_running_is_active = False
    else:
        recorded_running_is_active = None
    latest = validations[max(validations)] if validations else None
    return {
        "condition": condition,
        "seed": seed,
        "run_dir": str(resolved),
        "run_dir_exists": resolved.is_dir(),
        "recorded_state": status.get("state"),
        "recorded_running_is_active": recorded_running_is_active,
        "activity_evidence": (
            "active-process-observed"
            if pids
            else "not-observed"
            if process_scan_authoritative
            else "unavailable-limited-pid-namespace"
        ),
        "active_pids": pids,
        "completed_epoch": completed,
        "optimizer_update": status.get("optimizer_update"),
        "processed_images": status.get("processed_images"),
        "latest_checkpoint_exists": (resolved / "latest.pt").is_file(),
        "checkpoint_epochs": checkpoints,
        "validation_epochs": sorted(validations),
        "malformed_metrics_lines": malformed,
        "latest_validation": latest,
        "final_result_exists": bool(final),
        "final_result_state": final.get("state"),
        "final_result_condition": final_condition,
        "final_result_seed": final_seed,
        "final_result_identity_matches": final_identity_matches,
        "final_result_valid": final_result_valid,
        "final_epoch": final_epoch,
        "epoch_300_complete": final_result_valid and final_epoch == 300,
    }


def parse_run_values(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --comparison-run {value!r}; use CONDITION=PATH")
        condition, raw_path = value.split("=", 1)
        condition = condition.strip().upper()
        if not condition or condition in result or not raw_path.strip():
            raise ValueError(f"invalid or duplicate --comparison-run {value!r}")
        result[condition] = Path(raw_path).expanduser()
    return result


def _run_from_command(
    command: Mapping[str, Any],
    active: Mapping[str, Sequence[int]],
    *,
    process_scan_authoritative: bool,
) -> dict[str, Any]:
    return scan_run(
        Path(str(command["output_dir"])),
        condition=str(command["condition"]),
        seed=int(command["seed"]),
        active=active,
        process_scan_authoritative=process_scan_authoritative,
    )


def _milestone_coverage(runs: Sequence[Mapping[str, Any]], target: int) -> dict[str, Any]:
    reached = []
    for run in runs:
        validation_ok = target in run["validation_epochs"]
        checkpoint_ok = target in run["checkpoint_epochs"]
        final_ok = target != 300 or run["epoch_300_complete"]
        if validation_ok and checkpoint_ok and final_ok:
            reached.append(f"{run['condition']}/seed_{run['seed']}")
    return {
        "target_epoch": target,
        "required_runs": len(runs),
        "complete_runs": len(reached),
        "complete": len(reached) == len(runs) and bool(runs),
        "reached": reached,
    }


def build_audit(
    experiment_root: Path,
    layout_plan_path: Path,
    formal_report_dir: Path,
    *,
    comparison_id: str | None = None,
    comparison_seed: int | None = None,
    comparison_runs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    experiment_root = experiment_root.expanduser().resolve()
    layout_plan_path = layout_plan_path.expanduser().resolve()
    formal_report_dir = formal_report_dir.expanduser().resolve()
    plan_path = experiment_root / "matrix_execution_plan.json"
    matrix = _read_json(plan_path)
    commands = matrix.get("commands") if isinstance(matrix.get("commands"), list) else []
    process_scan = process_scan_metadata()
    active = active_output_processes()
    formal_runs = [
        _run_from_command(
            command,
            active,
            process_scan_authoritative=bool(process_scan["authoritative_for_host_activity"]),
        )
        for command in commands
    ]
    expected = {
        (str(condition), int(seed))
        for condition in matrix.get("conditions", [])
        for seed in matrix.get("seeds", [])
    }
    registered = {(run["condition"], run["seed"]) for run in formal_runs}

    layout = _read_json(layout_plan_path)
    mapping_name = str(layout.get("sample_mapping_file") or "physical_layout_samples.parquet")
    mapping_path = layout_plan_path.parent / mapping_name
    core_artifacts = {
        name: _artifact(experiment_root / name, hash_bytes=name.endswith(".json"))
        for name in CORE_ARTIFACTS
    }
    report_artifacts = {
        name: _artifact(formal_report_dir / name) for name in FINAL_REPORT_ARTIFACTS
    }
    physical_artifacts = {
        name: _artifact(formal_report_dir / name) for name in PHYSICAL_CANARY_ARTIFACTS
    }

    comparison: dict[str, Any] | None = None
    if comparison_runs:
        if comparison_seed is None:
            raise ValueError("--comparison-seed is required with --comparison-run")
        scanned = [
            scan_run(
                path,
                condition=condition,
                seed=comparison_seed,
                active=active,
                process_scan_authoritative=bool(process_scan["authoritative_for_host_activity"]),
            )
            for condition, path in sorted(comparison_runs.items())
        ]
        comparison = {
            "comparison_id": comparison_id or "comparison",
            "seed": comparison_seed,
            "run_count": len(scanned),
            "runs": scanned,
            "common_validation_epochs": sorted(
                set.intersection(
                    *(set(run["validation_epochs"]) for run in scanned)
                ) if scanned else set()
            ),
            "all_epoch_100_complete": bool(scanned)
            and all(100 in run["validation_epochs"] and 100 in run["checkpoint_epochs"] for run in scanned),
            "all_epoch_300_complete": bool(scanned)
            and all(run["epoch_300_complete"] for run in scanned),
            "claim_boundary": (
                "Exploratory comparison only; a single seed cannot provide the preregistered "
                "four-seed paired confidence intervals or final formal matrix result."
            ),
        }

    all_final = bool(formal_runs) and all(run["epoch_300_complete"] for run in formal_runs)
    all_report = all(item["exists"] for item in report_artifacts.values())
    matrix_exact = bool(expected) and expected == registered and len(commands) == len(expected)
    layout_link_ok = bool(layout.get("layout_hash")) and layout.get("layout_hash") == matrix.get("layout_hash")
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "experiment_root": str(experiment_root),
        "formal_report_dir": str(formal_report_dir),
        "process_scan": process_scan,
        "layout": {
            "plan": _artifact(layout_plan_path, hash_bytes=True),
            "mapping": _artifact(mapping_path),
            "layout_hash": layout.get("layout_hash"),
            "matrix_layout_hash": matrix.get("layout_hash"),
            "matrix_layout_hash_matches": layout_link_ok,
            "sample_count": layout.get("sample_count"),
            "target_pls_size": layout.get("target_pls_size"),
            "virtual_pls_count": layout.get("virtual_pls_count"),
        },
        "core_artifacts": core_artifacts,
        "formal_matrix": {
            "matrix_plan": _artifact(plan_path, hash_bytes=True),
            "expected_identity_count": len(expected),
            "registered_command_count": len(commands),
            "registered_matrix_exact": matrix_exact,
            "run_directory_count": sum(run["run_dir_exists"] for run in formal_runs),
            "epoch_300_complete_count": sum(run["epoch_300_complete"] for run in formal_runs),
            "all_epoch_300_complete": all_final,
            "milestones": [_milestone_coverage(formal_runs, target) for target in MILESTONES],
            "runs": formal_runs,
        },
        "comparison": comparison,
        "final_report_artifacts": report_artifacts,
        "physical_canary_artifacts": physical_artifacts,
        "goal_complete": (
            matrix_exact
            and layout_link_ok
            and all(item["exists"] for item in core_artifacts.values())
            and all_final
            and all_report
        ),
        "claim_boundaries": [
            "Recorded run_status.state=running is not accepted as an active process without a matching host PID.",
            "A final result is credited only when its condition and seed match the registered matrix identity.",
            "Milestone or single-seed metrics are not final 300-epoch four-seed results.",
            "Semantic-emulation results do not prove physical compressed-byte reduction.",
            "Physical FLS evidence is claimed only when the separate canary artifacts exist.",
        ],
    }
    return result


def _status(value: bool) -> str:
    return "complete" if value else "incomplete"


def render_markdown(audit: Mapping[str, Any]) -> str:
    matrix = audit["formal_matrix"]
    layout = audit["layout"]
    lines = [
        "# PLS core experiment goal audit",
        "",
        f"Overall status: **{_status(bool(audit['goal_complete']))}**.",
        "",
        "## Frozen layout and contracts",
        "",
        f"- Layout plan: {_status(bool(layout['plan']['exists']))}",
        f"- Layout sample mapping: {_status(bool(layout['mapping']['exists']))}",
        f"- Matrix/layout hash linkage: {_status(bool(layout['matrix_layout_hash_matches']))}",
        f"- Registered matrix identities: {matrix['registered_command_count']}/{matrix['expected_identity_count']}",
        "",
        "## Formal current-layout matrix",
        "",
        "| Condition | Seed | Recorded state | Active PID(s) | Epoch | Latest validation | Final identity | Final E300 |",
        "|---|---:|---|---|---:|---|---|---|",
    ]
    for run in matrix["runs"]:
        validation = run["latest_validation"]
        if validation:
            latest = f"E{validation['epoch']} top1={float(validation['validation_top1']):.3f}"
        else:
            latest = "none"
        pids = ",".join(str(pid) for pid in run["active_pids"]) or "none"
        final_identity = (
            "matched"
            if run["final_result_identity_matches"]
            else "mismatch"
            if run["final_result_exists"]
            else "missing"
        )
        lines.append(
            f"| {run['condition']} | {run['seed']} | {run['recorded_state'] or 'missing'} | "
            f"{pids} | {run['completed_epoch']} | {latest} | {final_identity} | "
            f"{'yes' if run['epoch_300_complete'] else 'no'} |"
        )
    lines.extend(["", "## Formal milestone coverage", "", "| Epoch | Complete runs | Required runs | Complete |", "|---:|---:|---:|---|"])
    for milestone in matrix["milestones"]:
        lines.append(
            f"| {milestone['target_epoch']} | {milestone['complete_runs']} | "
            f"{milestone['required_runs']} | {'yes' if milestone['complete'] else 'no'} |"
        )

    comparison = audit.get("comparison")
    if comparison:
        lines.extend([
            "",
            f"## {comparison['comparison_id']}",
            "",
            "| Condition | Seed | State | Active PID(s) | Epoch | Latest validation | E100 |",
            "|---|---:|---|---|---:|---|---|",
        ])
        for run in comparison["runs"]:
            validation = run["latest_validation"]
            latest = (
                f"E{validation['epoch']} top1={float(validation['validation_top1']):.3f} "
                f"loss={float(validation['validation_loss']):.4f}"
                if validation else "none"
            )
            pids = ",".join(str(pid) for pid in run["active_pids"]) or "none"
            e100 = 100 in run["validation_epochs"] and 100 in run["checkpoint_epochs"]
            lines.append(
                f"| {run['condition']} | {run['seed']} | {run['recorded_state'] or 'missing'} | "
                f"{pids} | {run['completed_epoch']} | {latest} | {'yes' if e100 else 'no'} |"
            )
        lines.extend(["", f"> {comparison['claim_boundary']}"])

    missing_reports = [name for name, value in audit["final_report_artifacts"].items() if not value["exists"]]
    lines.extend([
        "",
        "## Remaining required evidence",
        "",
        f"- Formal E300 runs: {matrix['epoch_300_complete_count']}/{matrix['expected_identity_count']}",
        f"- Missing final report artifacts: {len(missing_reports)}/{len(FINAL_REPORT_ARTIFACTS)}",
        f"- Physical canary artifacts present: {sum(value['exists'] for value in audit['physical_canary_artifacts'].values())}/{len(PHYSICAL_CANARY_ARTIFACTS)}",
        "",
        "## Claim boundaries",
        "",
    ])
    lines.extend(f"- {value}" for value in audit["claim_boundaries"])
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, data: bytes) -> None:
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


def write_audit(audit: Mapping[str, Any], output_dir: Path) -> None:
    encoded = json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    _atomic_write(output_dir / "goal_completion_audit.json", encoded)
    _atomic_write(output_dir / "goal_completion_audit.md", render_markdown(audit).encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--layout-plan", type=Path, required=True)
    parser.add_argument("--formal-report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-id")
    parser.add_argument("--comparison-seed", type=int)
    parser.add_argument("--comparison-run", action="append", default=[])
    args = parser.parse_args(argv)
    comparison_runs = parse_run_values(args.comparison_run)
    audit = build_audit(
        args.experiment_root,
        args.layout_plan,
        args.formal_report_dir,
        comparison_id=args.comparison_id,
        comparison_seed=args.comparison_seed,
        comparison_runs=comparison_runs,
    )
    write_audit(audit, args.output_dir.expanduser().resolve())
    print(json.dumps({
        "goal_complete": audit["goal_complete"],
        "formal_epoch_300_complete": audit["formal_matrix"]["epoch_300_complete_count"],
        "formal_expected": audit["formal_matrix"]["expected_identity_count"],
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
