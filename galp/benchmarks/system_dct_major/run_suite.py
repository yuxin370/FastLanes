#!/usr/bin/env python3
"""Run the no-shuffle DCT-major production-profile benchmark suite."""

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

from common import parse_manifest, resolve_coefficient_selection


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DIAGNOSTICS = HERE / "diagnostics"
DEFAULT_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_DCT_MAJOR_MANIFEST = Path("/tmp/galp-blockmajor-512-s1024-rg128/manifest.bin")
DEFAULT_DCT_MAJOR_LABELS = (
    REPO_ROOT
    / "galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/labels.json"
)
DEFAULT_BINDING_DIR = REPO_ROOT / "build/galp/torch"
DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_RAW_MASK_ORACLE_DIR = (
    REPO_ROOT
    / "galp/experiments/coefficient_mask_evaluator/runs/imagenet_val_k1_64_20260816_h100"
)

ALL_PIPELINES = (
    "dct_major_pushdown",
    "dct_major_coefficient_pushdown",
    "rgbnomore",
    "dali",
    "pytorch",
)
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


def _common_run_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--data-root",
        str(args.data_root.resolve()),
        "--dct-major-manifest",
        str(args.dct_major_manifest.resolve()),
        "--dct-major-label-map",
        str(args.dct_major_label_map.resolve()),
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
        "--dct-coeffs",
        args.dct_coeffs,
    ]
    if args.rgb_checkpoint is not None:
        result.extend(("--rgb-checkpoint", str(args.rgb_checkpoint.resolve())))
    if args.dct_checkpoint is not None:
        result.extend(("--dct-checkpoint", str(args.dct_checkpoint.resolve())))
    if args.raw_mask_oracle_dir is not None:
        result.extend(("--raw-mask-oracle-dir", str(args.raw_mask_oracle_dir.resolve())))
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
    return Phase(name=name, command=tuple(command), target=target, gpu=True)


def _initial_phases(args: argparse.Namespace, output_dir: Path) -> list[Phase]:
    first_shard_samples = int(parse_manifest(args.dct_major_manifest)["shards"][0]["image_count"])
    return [
        _run_phase(
            args,
            "02_feature_smoke",
            output_dir,
            workload="feature-extraction",
            pipelines=ALL_PIPELINES,
            sample_count=first_shard_samples,
            repeats=1,
        ),
        _run_phase(
            args,
            "03_evaluation_smoke",
            output_dir,
            workload="evaluation",
            pipelines=ALL_PIPELINES,
            sample_count=first_shard_samples,
            repeats=1,
        ),
    ]


def _contract_phase(args: argparse.Namespace, output_dir: Path) -> Phase:
    first_shard_samples = int(parse_manifest(args.dct_major_manifest)["shards"][0]["image_count"])
    phase = _run_phase(
        args,
        "00_semantic_contract",
        output_dir,
        workload="evaluation",
        pipelines=ALL_PIPELINES,
        sample_count=first_shard_samples,
        repeats=1,
    )
    return Phase(
        name=phase.name,
        command=(*phase.command, "--dry-run"),
        target=phase.target,
        gpu=False,
    )


def _semantic_phase(args: argparse.Namespace, output_dir: Path) -> Phase:
    target = output_dir / "01_coefficient_semantics.json"
    return Phase(
        name="01_coefficient_semantics",
        command=(
            str(args.python.resolve()),
            str(HERE / "verify_coefficient_semantics.py"),
            "--contract",
            str((output_dir / "00_semantic_contract/contract.json").resolve()),
            "--output",
            str(target.resolve()),
            "--sample-count",
            str(args.semantic_samples),
        ),
        target=target,
        gpu=True,
    )


def _formal_phases(args: argparse.Namespace, output_dir: Path) -> list[Phase]:
    phases: list[Phase] = []
    for prefix, workload in (("06", "feature-extraction"), ("07", "evaluation")):
        phases.append(
            _run_phase(
                args,
                f"{prefix}_formal_{workload.replace('-', '_')}",
                output_dir,
                workload=workload,
                pipelines=ALL_PIPELINES,
                sample_count=args.formal_samples,
                repeats=args.formal_repeats,
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
    failed_marker = marker.with_name(f"{phase.name}.failed.json")
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
    status_path = marker if exit_code == 0 else failed_marker
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
    failed_marker.unlink(missing_ok=True)


def _volume(args: argparse.Namespace) -> dict[str, int]:
    first_shard_samples = int(parse_manifest(args.dct_major_manifest)["shards"][0]["image_count"])
    smoke = 2 * len(ALL_PIPELINES) * first_shard_samples
    semantic = 8 * args.semantic_samples
    formal = 2 * len(ALL_PIPELINES) * args.formal_samples * args.formal_repeats
    model_ceiling = 4 * args.batch_size * args.ceiling_steps
    return {
        "smoke_pipeline_images": smoke,
        "semantic_model_invocations": semantic,
        "formal_pipeline_images": formal,
        "synthetic_model_ceiling_images": model_ceiling,
        "total_model_invocations": smoke + semantic + formal + model_ceiling,
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
    if not args.block_major_access_dir.is_dir():
        raise NotADirectoryError(args.block_major_access_dir)

    phases = [
        _contract_phase(args, output_dir),
        _semantic_phase(args, output_dir),
        *_initial_phases(args, output_dir),
        *_formal_phases(args, output_dir),
    ]
    volume = _volume(args)
    plan = {
        "schema_version": SUITE_SCHEMA,
        "dry_run": bool(args.dry_run),
        "runtime_policy": "native block-major production profile",
        "dct_coeffs": args.dct_coeffs,
        "volume": volume,
        "phases": [phase.as_json() for phase in phases],
    }
    plan_path = output_dir / "suite_plan.json"
    if args.resume and plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError("resume arguments differ from the existing suite plan")
    else:
        _write_json(plan_path, plan)

    for phase in phases:
        print(("GPU " if phase.gpu else "CPU ") + "COMMAND " + shlex.join(phase.command), flush=True)
    if args.dry_run:
        print("RESULT_JSON " + json.dumps({"dry_run": True, "plan": str(plan_path)}, sort_keys=True))
        return 0

    environment = os.environ.copy()
    pythonpath = os.pathsep.join((str(args.torch_binding_dir.resolve()), str((REPO_ROOT / "galp/torch").resolve())))
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath

    for phase in phases:
        _execute_phase(phase, output_dir, environment=environment, resume=args.resume)

    result = {
        "schema_version": SUITE_SCHEMA,
        "ok": True,
        "volume": volume,
        "feature_results": str((output_dir / "06_formal_feature_extraction/results.json").resolve()),
        "evaluation_results": str((output_dir / "07_formal_evaluation/results.json").resolve()),
        "coefficient_semantics": str((output_dir / "01_coefficient_semantics.json").resolve()),
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
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--rgb-checkpoint", type=Path)
    parser.add_argument("--dct-checkpoint", type=Path)
    parser.add_argument(
        "--raw-mask-oracle-dir",
        type=Path,
        default=(DEFAULT_RAW_MASK_ORACLE_DIR if DEFAULT_RAW_MASK_ORACLE_DIR.is_dir() else None),
    )
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_BINDING_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dct-coeffs", default="first:32")
    parser.add_argument("--formal-samples", type=int, default=50000)
    parser.add_argument("--formal-repeats", type=int, default=5)
    parser.add_argument("--semantic-samples", type=int, default=32)
    parser.add_argument("--ceiling-warmup", type=int, default=20)
    parser.add_argument("--ceiling-steps", type=int, default=300)
    parser.add_argument("--hash-samples", action="store_true")
    parser.add_argument("--hash-payloads", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "workers",
        "batch_size",
        "formal_samples",
        "formal_repeats",
        "semantic_samples",
        "ceiling_steps",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.ceiling_warmup < 0:
        parser.error("--ceiling-warmup must be non-negative")
    resolve_coefficient_selection(args.dct_coeffs)
    return args


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
