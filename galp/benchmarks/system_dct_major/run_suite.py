#!/usr/bin/env python3
"""Run the complete, no-shuffle DCT-major publication benchmark suite.

The suite first measures DCT-major segment locality, selects the segment with
the best end-to-end median throughput, and then uses that segment for the crop
ABBA and formal feature/evaluation runs.  Long GPU work is never started by a
dry run.  Successful phases are resumable; an incomplete non-empty phase is
left untouched for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DIAGNOSTICS = HERE / "diagnostics"
DEFAULT_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_DCT_MAJOR_MANIFEST = REPO_ROOT / "galp/data/imagedataset_dct/ImageNet-val/manifest.bin"
DEFAULT_DCT_MAJOR_LABELS = DEFAULT_DCT_MAJOR_MANIFEST.with_name("labels.json")
DEFAULT_IMAGE_MAJOR_MANIFEST = REPO_ROOT / "galp/data/system_rgbnomore/e2e_v2/dct/manifest.bin"
DEFAULT_IMAGE_MAJOR_LABELS = DEFAULT_IMAGE_MAJOR_MANIFEST.with_name("labels.json")
DEFAULT_BINDING_DIR = REPO_ROOT / "build/galp/torch"
DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_PLAN_AUDIT_TOOL = REPO_ROOT / "build/galp/tools/jpeg_dct/galp_block_major_plan_audit"

PIPELINES_WITHOUT_FULL = (
    "dct_major_legacy_pushdown",
    "dct_major_pushdown",
    "image_major_pushdown",
    "rgbnomore",
    "dali",
    "pytorch",
)
ALL_PIPELINES = ("dct_major_full", *PIPELINES_WITHOUT_FULL)
SUITE_SCHEMA = "galp_dct_major_complete_suite_v1"


@dataclass(frozen=True)
class Phase:
    name: str
    command: tuple[str, ...]
    target: Path | None
    gpu: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "target": str(self.target) if self.target is not None else None,
            "gpu": self.gpu,
        }


def _parse_segment_sizes(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("segment sizes must be unique positive comma-separated integers")
    return values


def _common_run_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--data-root",
        str(args.data_root.resolve()),
        "--dct-major-manifest",
        str(args.dct_major_manifest.resolve()),
        "--dct-major-label-map",
        str(args.dct_major_label_map.resolve()),
        "--image-major-manifest",
        str(args.image_major_manifest.resolve()),
        "--image-major-label-map",
        str(args.image_major_label_map.resolve()),
        "--image-major-manifest-version",
        str(args.image_major_manifest_version),
        "--rgbnomore-root",
        str(args.rgbnomore_root.resolve()),
        "--torch-binding-dir",
        str(args.torch_binding_dir.resolve()),
        "--python",
        str(args.python.resolve()),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--decode-workset-capacity-mib",
        str(args.decode_workset_capacity_mib),
        "--block-major-double-buffer",
        args.block_major_double_buffer,
    ]
    if args.rgb_checkpoint is not None:
        result.extend(("--rgb-checkpoint", str(args.rgb_checkpoint.resolve())))
    if args.dct_checkpoint is not None:
        result.extend(("--dct-checkpoint", str(args.dct_checkpoint.resolve())))
    block_major_access_dir = getattr(args, "block_major_access_dir", None)
    if block_major_access_dir is not None:
        result.extend(("--block-major-access-dir", str(block_major_access_dir.resolve())))
    if args.hash_samples:
        result.append("--hash-samples")
    if args.hash_payloads:
        result.append("--hash-payloads")
    return result


def _run_phase(
    args: argparse.Namespace,
    name: str,
    output_dir: Path,
    *,
    workload: str,
    pipelines: Sequence[str],
    sample_count: int | None = None,
    repeats: int | None = None,
    segment_size: int | None = None,
    dry_contract: bool = False,
) -> Phase:
    target = output_dir / name
    command = [
        str(args.python.resolve()),
        str(HERE / "run.py"),
        "--output-dir",
        str(target),
        "--preset",
        "e2e" if sample_count is not None else "smoke",
        "--workload",
        workload,
        "--pipelines",
        *pipelines,
        *_common_run_args(args),
    ]
    if sample_count is not None:
        command.extend(
            (
                "--batch-size",
                str(args.batch_size),
                "--warmup-batches",
                "0",
                "--sample-count",
                str(sample_count),
            )
        )
    if repeats is not None:
        command.extend(("--repeats", str(repeats)))
    if segment_size is not None:
        command.extend(("--dct-major-segment-size", str(segment_size)))
    if dry_contract:
        command.append("--dry-run")
    return Phase(name=name, command=tuple(command), target=target, gpu=not dry_contract)


def _preflight_phases(args: argparse.Namespace, output_dir: Path) -> list[Phase]:
    planning_output = output_dir / "00_segment_planning.json"
    phases: list[Phase] = [
        Phase(
            name="00_segment_planning",
            command=(
                str(args.python.resolve()),
                str(DIAGNOSTICS / "segment_sweep.py"),
                str(args.dct_major_manifest.resolve()),
                "--torch-binding-dir",
                str(args.torch_binding_dir.resolve()),
                "--block-major-access-dir",
                str(args.block_major_access_dir.resolve()),
                "--segment-sizes",
                *(str(item) for item in args.segment_sizes),
                "--output-json",
                str(planning_output),
            ),
            target=planning_output,
            gpu=False,
        ),
    ]
    gate_specs = (
        ("sequential", ("--count", "2")),
        ("random", ("--count", "8", "--pattern", "random", "--seed", "20260731")),
        (
            "duplicates_explicit",
            ("--image-ids", "0,0,0,1,1,1,0,0", "--explicit-crops"),
        ),
        (
            "cross_shard",
            ("--image-ids", "1527,1528,1529,1530", "--require-cross-shard"),
        ),
        ("grayscale", ("--image-ids", "239", "--require-grayscale")),
    )
    for gate_name, gate_arguments in gate_specs:
        gpu_gate_output = output_dir / f"00_planless_gpu_gate_{gate_name}.json"
        phases.append(
            Phase(
                name=f"00_planless_gpu_gate_{gate_name}",
                command=(
                    str(args.plan_audit_tool.resolve()),
                    str(args.dct_major_manifest.resolve()),
                    "--descriptor-dir",
                    str(args.block_major_access_dir.resolve()),
                    *gate_arguments,
                    "--compare-legacy",
                    "--execute",
                    "--decode-batch-rowgroups",
                    "64",
                    "--prefetch-workers",
                    "2",
                    "--workset-capacity-mib",
                    str(args.decode_workset_capacity_mib),
                    "--output-json",
                    str(gpu_gate_output),
                ),
                target=gpu_gate_output,
                gpu=True,
            )
        )
    return phases


def _initial_phases(args: argparse.Namespace, output_dir: Path) -> list[Phase]:
    phases = _preflight_phases(args, output_dir)
    phases.extend(
        (
            _run_phase(
                args,
                "01_feature_smoke",
                output_dir,
                workload="feature-extraction",
                pipelines=ALL_PIPELINES,
            ),
            _run_phase(
                args,
                "02_evaluation_smoke",
                output_dir,
                workload="evaluation",
                pipelines=ALL_PIPELINES,
            ),
        )
    )
    for segment_size in args.segment_sizes:
        phases.append(
            _run_phase(
                args,
                f"03_locality_segment_{segment_size:04d}",
                output_dir,
                workload="feature-extraction",
                pipelines=("dct_major_pushdown",),
                sample_count=args.locality_samples,
                repeats=args.locality_repeats,
                segment_size=segment_size,
            )
        )
    return phases


def _selected_phases(args: argparse.Namespace, output_dir: Path, segment_size: int) -> list[Phase]:
    contract_phase = _run_phase(
        args,
        "04_crop_abba_contract",
        output_dir,
        workload="feature-extraction",
        pipelines=("dct_major_legacy_pushdown", "dct_major_pushdown"),
        sample_count=args.crop_samples,
        repeats=1,
        segment_size=segment_size,
        dry_contract=True,
    )
    abba_target = output_dir / "05_crop_abba"
    phases = [
        contract_phase,
        Phase(
            name="05_crop_abba",
            command=(
                str(args.python.resolve()),
                str(DIAGNOSTICS / "run_crop_abba.py"),
                "--contract",
                str(contract_phase.target / "contract.json"),
                "--output-dir",
                str(abba_target),
                "--python",
                str(args.python.resolve()),
            ),
            target=abba_target,
            gpu=True,
        ),
    ]
    formal_pipelines = ALL_PIPELINES if args.include_full_in_formal else PIPELINES_WITHOUT_FULL
    for prefix, workload in (("06", "feature-extraction"), ("07", "evaluation")):
        phases.append(
            _run_phase(
                args,
                f"{prefix}_formal_{workload.replace('-', '_')}",
                output_dir,
                workload=workload,
                pipelines=formal_pipelines,
                sample_count=args.formal_samples,
                repeats=args.formal_repeats,
                segment_size=segment_size,
            )
        )

    rgb_checkpoint = args.rgb_checkpoint or args.rgbnomore_root / "checkpoints/imgnetRGBViTTi_ep300_74.1.pth"
    dct_checkpoint = args.dct_checkpoint or args.rgbnomore_root / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
    for index, (domain, workload, checkpoint) in enumerate(
        (
            ("dct", "feature-extraction", dct_checkpoint),
            ("rgb", "feature-extraction", rgb_checkpoint),
            ("dct", "evaluation", dct_checkpoint),
            ("rgb", "evaluation", rgb_checkpoint),
        )
    ):
        phases.append(
            Phase(
                name=f"08_model_ceiling_{index}_{domain}_{workload.replace('-', '_')}",
                command=(
                    str(args.python.resolve()),
                    str(DIAGNOSTICS / "model_ceiling.py"),
                    "--domain",
                    domain,
                    "--workload",
                    workload,
                    "--rgbnomore-root",
                    str(args.rgbnomore_root.resolve()),
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                    "--warmup",
                    str(args.ceiling_warmup),
                    "--steps",
                    str(args.ceiling_steps),
                ),
                target=None,
                gpu=True,
            )
        )
    return phases


def _phase_marker(output_dir: Path, phase: Phase) -> Path:
    return output_dir / "phase_status" / f"{phase.name}.json"


def _target_is_occupied(target: Path | None) -> bool:
    if target is None or not target.exists():
        return False
    return target.is_file() or any(target.iterdir())


def _execute_phase(
    phase: Phase,
    output_dir: Path,
    *,
    environment: dict[str, str],
    resume: bool,
) -> None:
    marker = _phase_marker(output_dir, phase)
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("exit_code") != 0:
            raise RuntimeError(
                f"phase has a recorded failure; inspect or move its target before resuming: {phase.name}"
            )
        if recorded.get("command") != list(phase.command) or not resume:
            raise FileExistsError(f"phase already completed; use --resume with the same command: {phase.name}")
        print(f"SKIP completed phase {phase.name}", flush=True)
        return
    if _target_is_occupied(phase.target):
        raise FileExistsError(
            f"refusing to overwrite incomplete phase target {phase.target}; inspect or move it before resuming"
        )

    print("COMMAND " + shlex.join(phase.command), flush=True)
    log = output_dir / "logs" / f"{phase.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND " + shlex.join(phase.command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            phase.command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        exit_code = int(process.wait())
    status_path = marker if exit_code == 0 else marker.with_name(f"{phase.name}.failed.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {"schema_version": SUITE_SCHEMA, "phase": phase.name, "command": list(phase.command), "exit_code": exit_code},
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if exit_code != 0:
        raise RuntimeError(f"phase failed with exit code {exit_code}: {phase.name}")


def _select_best_segment(
    output_dir: Path,
    segment_sizes: Sequence[int],
) -> tuple[int, list[dict[str, float | int]]]:
    records: list[dict[str, float | int]] = []
    for segment_size in segment_sizes:
        path = output_dir / f"03_locality_segment_{segment_size:04d}" / "results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result.get("ok"):
            raise RuntimeError(f"locality candidate failed validation: segment={segment_size}")
        aggregate = next(
            item for item in result["aggregates"] if item["pipeline"] == "dct_major_pushdown"
        )
        cold = aggregate.get("cold_start") or {
            "throughput_images_per_s": float(aggregate["throughput_images_per_s"]["p50"]),
            "time_to_first_batch_ms": float(aggregate["time_to_first_batch_ms"]["p50"]),
        }
        records.append(
            {
                "segment_size": segment_size,
                "cold_end_to_end_images_per_s": float(
                    cold["throughput_images_per_s"]
                ),
                "cold_time_to_first_batch_ms": float(
                    cold["time_to_first_batch_ms"]
                ),
                "end_to_end_p50_images_per_s": float(aggregate["throughput_images_per_s"]["p50"]),
                "steady_p50_images_per_s": float(aggregate["steady_throughput_images_per_s"]["p50"]),
                "time_to_first_batch_p50_ms": float(aggregate["time_to_first_batch_ms"]["p50"]),
            }
        )
    best = max(
        records,
        key=lambda item: (
            item["cold_end_to_end_images_per_s"],
            -item["cold_time_to_first_batch_ms"],
            -item["segment_size"],
        ),
    )
    return int(best["segment_size"]), records


def _volume(args: argparse.Namespace) -> dict[str, int]:
    smoke = 2 * len(ALL_PIPELINES) * 6
    locality = len(args.segment_sizes) * args.locality_samples * args.locality_repeats
    crop_abba = 4 * args.crop_samples
    formal_pipeline_count = len(ALL_PIPELINES) if args.include_full_in_formal else len(PIPELINES_WITHOUT_FULL)
    formal = 2 * formal_pipeline_count * args.formal_samples * args.formal_repeats
    model_ceiling = 4 * args.batch_size * args.ceiling_steps
    return {
        "gpu_gate_requests": 23,
        "smoke_pipeline_images": smoke,
        "locality_pipeline_images": locality,
        "crop_abba_pipeline_images": crop_abba,
        "formal_pipeline_images": formal,
        "synthetic_model_ceiling_images": model_ceiling,
        "total_model_invocations": smoke + locality + crop_abba + formal + model_ceiling,
    }


def _gates_only_volume() -> dict[str, int]:
    return {
        "gpu_gate_requests": 23,
        "smoke_pipeline_images": 0,
        "locality_pipeline_images": 0,
        "crop_abba_pipeline_images": 0,
        "formal_pipeline_images": 0,
        "synthetic_model_ceiling_images": 0,
        "total_model_invocations": 0,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _prepare_output_dir(output_dir: Path, *, resume: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if any(output_dir.iterdir()) and not resume:
            raise FileExistsError(f"refusing to overwrite non-empty suite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    _prepare_output_dir(output_dir, resume=args.resume)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not args.plan_audit_tool.is_file():
        raise FileNotFoundError(args.plan_audit_tool)
    if not args.block_major_access_dir.is_dir():
        raise NotADirectoryError(args.block_major_access_dir)

    initial = (
        _preflight_phases(args, output_dir)
        if args.gates_only
        else _initial_phases(args, output_dir)
    )
    dry_selected = (
        []
        if args.gates_only
        else _selected_phases(args, output_dir, args.formal_segment_fallback)
    )
    volume = _gates_only_volume() if args.gates_only else _volume(args)
    plan = {
        "schema_version": SUITE_SCHEMA,
        "dry_run": bool(args.dry_run),
        "gates_only": bool(args.gates_only),
        "selection_rule": "maximum locality end-to-end p50; smaller segment wins an exact tie",
        "formal_segment_fallback_for_dry_run": args.formal_segment_fallback,
        "segment_candidates": list(args.segment_sizes),
        "volume": volume,
        "initial_phases": [phase.as_json() for phase in initial],
        "selected_phase_template": [phase.as_json() for phase in dry_selected],
    }
    plan_path = output_dir / "suite_plan.json"
    if args.resume and plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        comparable_existing = {
            key: value
            for key, value in existing.items()
            if key not in {"selected_segment_size", "selected_phases"}
        }
        if comparable_existing != plan:
            raise ValueError("resume arguments differ from the existing suite plan")
    else:
        _write_json(plan_path, plan)

    for phase in (*initial, *dry_selected):
        print(("GPU " if phase.gpu else "CPU ") + "COMMAND " + shlex.join(phase.command), flush=True)
    if args.dry_run:
        print("RESULT_JSON " + json.dumps({"dry_run": True, "plan": str(plan_path)}, sort_keys=True))
        return 0

    environment = os.environ.copy()
    pythonpath = os.pathsep.join((str(args.torch_binding_dir.resolve()), str((REPO_ROOT / "galp/torch").resolve())))
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath

    for phase in initial:
        _execute_phase(phase, output_dir, environment=environment, resume=args.resume)
    if args.gates_only:
        gate_outputs = [
            str(phase.target.resolve())
            for phase in initial
            if phase.name.startswith("00_planless_gpu_gate_") and phase.target is not None
        ]
        result = {
            "schema_version": SUITE_SCHEMA,
            "ok": True,
            "gates_only": True,
            "segment_planning": str((output_dir / "00_segment_planning.json").resolve()),
            "gpu_gate_outputs": gate_outputs,
            "volume": volume,
        }
        _write_json(output_dir / "suite_results.json", result)
        print("RESULT_JSON " + json.dumps(result, sort_keys=True))
        return 0
    selected_segment, locality_records = _select_best_segment(output_dir, args.segment_sizes)
    selected = _selected_phases(args, output_dir, selected_segment)
    plan["selected_segment_size"] = selected_segment
    plan["selected_phases"] = [phase.as_json() for phase in selected]
    _write_json(plan_path, plan)
    for phase in selected:
        _execute_phase(phase, output_dir, environment=environment, resume=args.resume)

    result = {
        "schema_version": SUITE_SCHEMA,
        "ok": True,
        "selected_segment_size": selected_segment,
        "selection_metric": "cold_end_to_end_images_per_s_then_cold_ttft",
        "locality_records": locality_records,
        "volume": volume,
        "crop_abba": str((output_dir / "05_crop_abba/abba_results.json").resolve()),
        "feature_results": str((output_dir / "06_formal_feature_extraction/results.json").resolve()),
        "evaluation_results": str((output_dir / "07_formal_evaluation/results.json").resolve()),
    }
    _write_json(output_dir / "suite_results.json", result)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dct-major-manifest", type=Path, default=DEFAULT_DCT_MAJOR_MANIFEST)
    parser.add_argument("--block-major-access-dir", type=Path, required=True)
    parser.add_argument("--dct-major-label-map", type=Path, default=DEFAULT_DCT_MAJOR_LABELS)
    parser.add_argument("--image-major-manifest", type=Path, default=DEFAULT_IMAGE_MAJOR_MANIFEST)
    parser.add_argument("--image-major-label-map", type=Path, default=DEFAULT_IMAGE_MAJOR_LABELS)
    parser.add_argument("--image-major-manifest-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--rgb-checkpoint", type=Path)
    parser.add_argument("--dct-checkpoint", type=Path)
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_BINDING_DIR)
    parser.add_argument("--plan-audit-tool", type=Path, default=DEFAULT_PLAN_AUDIT_TOOL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--decode-workset-capacity-mib", type=int, default=512)
    parser.add_argument(
        "--block-major-double-buffer",
        choices=("auto", "on", "off"),
        default="auto",
        help="bounded block-major decode double-buffer policy",
    )
    parser.add_argument("--segment-sizes", type=_parse_segment_sizes, default=(50, 250, 500, 1000, 1024))
    parser.add_argument("--locality-samples", type=int, default=5000)
    parser.add_argument("--locality-repeats", type=int, default=3)
    parser.add_argument("--crop-samples", type=int, default=1000)
    parser.add_argument("--formal-samples", type=int, default=50000)
    parser.add_argument("--formal-repeats", type=int, default=5)
    parser.add_argument("--formal-segment-fallback", type=int, default=1000)
    parser.add_argument("--ceiling-warmup", type=int, default=20)
    parser.add_argument("--ceiling-steps", type=int, default=300)
    parser.add_argument("--include-full-in-formal", action="store_true")
    parser.add_argument("--hash-samples", action="store_true")
    parser.add_argument("--hash-payloads", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gates-only",
        action="store_true",
        help="run segment planning and the five bounded planless GPU gates, then stop",
    )
    args = parser.parse_args(argv)
    for name in (
        "workers",
        "batch_size",
        "decode_workset_capacity_mib",
        "locality_samples",
        "locality_repeats",
        "crop_samples",
        "formal_samples",
        "formal_repeats",
        "formal_segment_fallback",
        "ceiling_steps",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.ceiling_warmup < 0:
        parser.error("--ceiling-warmup must be non-negative")
    return args


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
