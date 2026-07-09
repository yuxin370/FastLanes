#!/usr/bin/env python3
"""Run the comparable GALP/RGB-no-more benchmark set.

This wrapper keeps the individual benchmark scripts as the source of truth, but
assembles a reproducible command set for:

1. GALP Direct-DCT -> RGB-no-more JPEG-Ti
2. RGB-no-more native DCT JPEG-Ti
3. RGB-no-more RGB ViT-Ti

It writes per-backend JSON files and a combined summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_OUTPUT_DIR = Path("/tmp/rgbnomore_comparison")
DEFAULT_TORCH_BINDING_DIR = REPO_ROOT / "build" / "galp" / "torch"


def _script(name: str) -> Path:
    return REPO_ROOT / "galp" / "examples" / name


def _default_dct_index_file(rgbnomore_root: Path, split: str) -> Path:
    filename = "indexbase_train.csv" if split == "train" else "indexbase_val.csv"
    return rgbnomore_root / "assets" / filename


def _default_manifest(split: str) -> Path:
    if split == "train":
        directory = "ImageNet-train-multi"
    elif split == "inference":
        directory = "ImageNet-val-rg8"
    else:
        directory = "ImageNet-val-rg8"
    return REPO_ROOT / "galp" / "data" / "imagedataset_dct" / directory / "manifest.bin"


def _default_label_map(manifest: Path) -> Path:
    return manifest.parent / "labels.json"


def _default_rgb_split_dir(split: str) -> str:
    return "val" if split == "inference" else split


def _default_rgb_checkpoint(rgbnomore_root: Path) -> Path:
    return rgbnomore_root / "checkpoints" / "imgnetRGBViTTi_ep300_74.1.pth"


def _default_dct_checkpoint(rgbnomore_root: Path) -> Path:
    return rgbnomore_root / "checkpoints" / "imgnetDCTViTTi_ep300_75.1.pth"


def _count_rgbnomore_index_images(index_file: Path) -> int:
    with index_file.open("r", encoding="utf-8") as stream:
        line_count = sum(1 for _line in stream)
    image_count = line_count - 1
    if image_count < 0:
        raise ValueError(f"{index_file} is empty")
    return image_count


def _validate_dali_device_args(device: str, device_id: int) -> None:
    if device == "cuda":
        return
    if not device.startswith("cuda:"):
        raise ValueError("--include-dali-rgb requires --device cuda or cuda:N")
    try:
        parsed_id = int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"invalid CUDA device string for DALI: {device}") from exc
    if parsed_id != device_id:
        raise ValueError(f"--dali-device-id {device_id} does not match --device {device}")


def _add_common_phase_args(command: list[str], args: argparse.Namespace, *, include_workers: bool) -> None:
    command.extend(
        [
            "--phase",
            args.phase,
            "--batch-size",
            str(args.batch_size),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--train-lr",
            str(args.train_lr),
        ]
    )
    if include_workers:
        command.extend(["--workers", str(args.workers)])


def _galp_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    binding_path = str(args.torch_binding_dir)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = binding_path if not current else binding_path + os.pathsep + current
    return env


def _run(command: list[str], *, env: dict[str, str] | None, dry_run: bool, log_path: Path | None = None) -> int:
    print("COMMAND " + " ".join(command), flush=True)
    if dry_run:
        return 0
    log_stream = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("w", encoding="utf-8")
        log_stream.write("COMMAND " + " ".join(command) + "\n")
        log_stream.flush()
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_stream is not None:
                log_stream.write(line)
                log_stream.flush()
        return int(process.wait())
    finally:
        if log_stream is not None:
            log_stream.close()


def _build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = args.output_dir
    galp_json = output_dir / "galp_direct_dct_rgbnomore.json"
    galp_verify_json = output_dir / "galp_manifest_verify.json"
    dct_json = output_dir / "rgbnomore_native_dct.json"
    rgb_json = output_dir / "rgbnomore_rgb.json"
    dali_rgb_json = output_dir / "dali_rgb.json"
    summary_path = output_dir / ("summary.md" if args.summary_format == "markdown" else "summary.csv")
    validation_json = output_dir / "validation.json"

    commands: list[dict[str, Any]] = []
    if not args.skip_galp:
        if args.phase == "train" and args.prepare_galp_label_map and not args.galp_label_map_json.exists():
            command = [
                sys.executable,
                str(_script("prepare_rgbnomore_direct_dct_manifest.py")),
                "--split",
                args.split,
                "--data-root",
                str(args.data_root),
                "--input-dir",
                str(args.data_dir),
                "--index-file",
                str(args.dct_index_file),
                "--label-map-json",
                str(args.galp_label_map_json),
                "--write-label-map-only",
                "--output-json",
                str(output_dir / "galp_label_map.json"),
            ]
            if args.expected_galp_images is not None:
                command.extend(["--expected-image-count", str(args.expected_galp_images)])
            commands.append(
                {
                    "name": "galp_label_map",
                    "command": command,
                    "env": None,
                    "output": args.galp_label_map_json,
                    "summary_input": False,
                }
            )
        if not args.skip_galp_verify:
            command = [
                sys.executable,
                str(_script("prepare_rgbnomore_direct_dct_manifest.py")),
                "--verify-only",
                "--manifest",
                str(args.manifest),
                "--torch-binding-dir",
                str(args.torch_binding_dir),
                "--validate-sample-images",
                str(args.verify_sample_images),
                "--output-json",
                str(galp_verify_json),
            ]
            if args.expected_galp_images is not None:
                command.extend(["--expected-image-count", str(args.expected_galp_images)])
            commands.append(
                {
                    "name": "galp_manifest_verify",
                    "command": command,
                    "env": "galp",
                    "output": galp_verify_json,
                    "summary_input": False,
                }
            )
        command = [
            sys.executable,
            str(_script("direct_dct_rgbnomore_benchmark.py")),
            str(args.manifest),
            "--rgbnomore-root",
            str(args.rgbnomore_root),
            "--checkpoint",
            str(args.dct_checkpoint),
            "--phase",
            args.phase,
            "--batch-size",
            str(args.batch_size),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--train-lr",
            str(args.train_lr),
            "--preprocess",
            args.galp_preprocess,
            "--cache-capacity-mib",
            str(args.galp_cache_capacity_mib),
            "--output-json",
            str(galp_json),
        ]
        if args.phase == "train":
            command.extend(["--label-map-json", str(args.galp_label_map_json)])
        commands.append(
            {
                "name": "galp_direct_dct_rgbnomore",
                "command": command,
                "env": "galp",
                "output": galp_json,
                "summary_input": True,
            }
        )

    if not args.skip_dct_baseline:
        command = [
            sys.executable,
            str(_script("rgbnomore_dct_baseline_benchmark.py")),
            "--rgbnomore-root",
            str(args.rgbnomore_root),
            "--checkpoint",
            str(args.dct_checkpoint),
            "--data-root",
            str(args.data_root),
            "--split",
            args.split,
            "--device",
            args.device,
            "--output-json",
            str(dct_json),
        ]
        _add_common_phase_args(command, args, include_workers=True)
        if args.eval_transform:
            command.append("--eval-transform")
        command.extend(["--index-file", str(args.dct_index_file)])
        commands.append(
            {"name": "rgbnomore_native_dct", "command": command, "env": None, "output": dct_json, "summary_input": True}
        )

    if not args.skip_rgb_baseline:
        command = [
            sys.executable,
            str(_script("rgbnomore_rgb_baseline_benchmark.py")),
            "--rgbnomore-root",
            str(args.rgbnomore_root),
            "--checkpoint",
            str(args.rgb_checkpoint),
            "--data-dir",
            str(args.data_dir),
            "--split-label",
            args.split,
            "--device",
            args.device,
            "--output-json",
            str(rgb_json),
        ]
        _add_common_phase_args(command, args, include_workers=True)
        commands.append({"name": "rgbnomore_rgb", "command": command, "env": None, "output": rgb_json, "summary_input": True})

    if args.include_dali_rgb:
        command = [
            sys.executable,
            str(_script("rgbnomore_dali_rgb_baseline_benchmark.py")),
            "--rgbnomore-root",
            str(args.rgbnomore_root),
            "--checkpoint",
            str(args.rgb_checkpoint),
            "--data-dir",
            str(args.data_dir),
            "--split-label",
            args.split,
            "--device",
            args.device,
            "--device-id",
            str(args.dali_device_id),
            "--prefetch-queue-depth",
            str(args.dali_prefetch_queue_depth),
            "--output-json",
            str(dali_rgb_json),
        ]
        _add_common_phase_args(command, args, include_workers=True)
        commands.append({"name": "dali_rgb", "command": command, "env": None, "output": dali_rgb_json, "summary_input": True})

    summary_inputs = [str(item["output"]) for item in commands if item.get("summary_input", False)]
    if summary_inputs:
        command = [
            sys.executable,
            str(_script("summarize_rgbnomore_benchmarks.py")),
            *summary_inputs,
            "--format",
            args.summary_format,
            "--output",
            str(summary_path),
        ]
        commands.append({"name": "summary", "command": command, "env": None, "output": summary_path, "summary_input": False})
    if not args.skip_validation:
        command = [
            sys.executable,
            str(_script("validate_rgbnomore_comparison.py")),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(validation_json),
            "--expected-batch-size",
            str(args.batch_size),
            "--expected-steps",
            str(args.steps),
            "--expected-phase",
            args.phase,
            "--expected-device",
            args.device,
            "--expected-split",
            args.split,
            "--expected-data-root",
            str(args.data_root),
            "--expected-rgb-data-dir",
            str(args.data_dir),
            "--expected-dct-index-file",
            str(args.dct_index_file),
            "--expected-dct-eval-transform",
            "true" if args.eval_transform else "false",
            "--expected-rgb-checkpoint",
            str(args.rgb_checkpoint),
            "--expected-dct-checkpoint",
            str(args.dct_checkpoint),
        ]
        if not args.skip_galp:
            command.extend(
                [
                    "--expected-galp-manifest",
                    str(args.manifest),
                    "--expected-galp-preprocess",
                    args.galp_preprocess,
                ]
            )
            if args.phase == "train":
                command.extend(["--expected-galp-label-map-json", str(args.galp_label_map_json)])
        if args.include_dali_rgb:
            command.extend(
                [
                    "--dali-json",
                    str(dali_rgb_json),
                    "--require-dali-rgb",
                    "--expected-dali-prefetch-queue-depth",
                    str(args.dali_prefetch_queue_depth),
                ]
            )
        if args.expected_split_images is not None:
            command.extend(["--expected-split-images", str(args.expected_split_images)])
        if args.expected_galp_images is not None:
            command.extend(["--expected-galp-images", str(args.expected_galp_images)])
        if args.skip_galp:
            command.append("--allow-missing-galp")
        commands.append(
            {
                "name": "validation",
                "command": command,
                "env": None,
                "output": validation_json,
                "summary_input": False,
            }
        )
    return commands


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparable GALP/RGB-no-more benchmark wrappers")
    parser.add_argument("--manifest", type=Path, help="GALP DCT manifest. Defaults to the selected split's current ImageNet DCT manifest.")
    parser.add_argument("--galp-label-map-json", type=Path, help="GALP image_id -> RGB-no-more label sidecar. Defaults to MANIFEST_DIR/labels.json.")
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--data-dir", type=Path, help="RGB ImageFolder split dir. Defaults to DATA_ROOT/SPLIT, with inference mapped to val.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dct-index-file", type=Path)
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_TORCH_BINDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phase", choices=("loader", "forward", "end-to-end", "train", "both"), default="both")
    parser.add_argument("--split", choices=("val", "train", "inference"), default="val")
    parser.add_argument("--eval-transform", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-lr", type=float, default=0.0, help="SGD learning rate used only with --phase train.")
    parser.add_argument("--galp-preprocess", choices=("rgbnomore-val", "rgbnomore-val-pushdown", "direct-crop"), default="rgbnomore-val-pushdown")
    parser.add_argument("--galp-cache-capacity-mib", type=int, default=1024)
    parser.add_argument("--verify-sample-images", type=int, default=8)
    parser.add_argument("--expected-galp-images", type=int)
    parser.add_argument("--expected-split-images", type=int)
    parser.add_argument("--summary-format", choices=("csv", "markdown"), default="markdown")
    parser.add_argument("--skip-galp", action="store_true")
    parser.add_argument(
        "--no-prepare-galp-label-map",
        action="store_false",
        dest="prepare_galp_label_map",
        help="Do not auto-generate MANIFEST_DIR/labels.json when running GALP --phase train.",
    )
    parser.add_argument("--skip-galp-verify", action="store_true")
    parser.add_argument("--skip-dct-baseline", action="store_true")
    parser.add_argument("--skip-rgb-baseline", action="store_true")
    parser.add_argument("--include-dali-rgb", action="store_true", help="Also run the optional NVIDIA DALI RGB baseline.")
    parser.add_argument("--dali-device-id", type=int, default=0)
    parser.add_argument("--dali-prefetch-queue-depth", type=int, default=2)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(prepare_galp_label_map=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("--batch-size and --steps must be positive")
    if args.warmup < 0 or args.workers < 0:
        raise ValueError("--warmup and --workers must be non-negative")
    if args.train_lr < 0.0:
        raise ValueError("--train-lr must be non-negative")
    if args.verify_sample_images <= 0:
        raise ValueError("--verify-sample-images must be positive")
    if args.expected_galp_images is not None and args.expected_galp_images < 0:
        raise ValueError("--expected-galp-images must be non-negative")
    if args.expected_split_images is not None and args.expected_split_images < 0:
        raise ValueError("--expected-split-images must be non-negative")
    if args.dali_device_id < 0:
        raise ValueError("--dali-device-id must be non-negative")
    if args.dali_prefetch_queue_depth <= 0:
        raise ValueError("--dali-prefetch-queue-depth must be positive")
    if args.include_dali_rgb:
        if args.workers <= 0:
            raise ValueError("--include-dali-rgb requires --workers > 0")
        _validate_dali_device_args(args.device, args.dali_device_id)
    if args.manifest is None:
        args.manifest = _default_manifest(args.split)
    if args.galp_label_map_json is None:
        args.galp_label_map_json = _default_label_map(args.manifest)
    if args.data_dir is None:
        args.data_dir = args.data_root / _default_rgb_split_dir(args.split)
    if args.dct_index_file is None:
        args.dct_index_file = _default_dct_index_file(args.rgbnomore_root, args.split)
    if args.split == "train" and not args.skip_galp and args.galp_preprocess in ("rgbnomore-val", "rgbnomore-val-pushdown") and not args.eval_transform:
        print(
            "INFO forcing --eval-transform for train split because GALP rgbnomore-val preprocessing "
            "is a deterministic eval-style DCT transform.",
            flush=True,
        )
        args.eval_transform = True
    args.rgb_checkpoint = _default_rgb_checkpoint(args.rgbnomore_root)
    args.dct_checkpoint = _default_dct_checkpoint(args.rgbnomore_root)
    if args.expected_split_images is None and args.dct_index_file.exists():
        args.expected_split_images = _count_rgbnomore_index_images(args.dct_index_file)
    if args.expected_galp_images is None and not args.skip_galp:
        args.expected_galp_images = args.expected_split_images
    if not args.skip_galp and not args.dry_run and not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    if (
        not args.skip_galp
        and args.phase == "train"
        and not args.prepare_galp_label_map
        and not args.dry_run
        and not args.galp_label_map_json.exists()
    ):
        raise FileNotFoundError(args.galp_label_map_json)
    if not args.skip_galp and not args.dry_run and not args.torch_binding_dir.exists():
        raise FileNotFoundError(args.torch_binding_dir)
    if not args.skip_dct_baseline and not args.dry_run and not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    if not args.skip_dct_baseline and not args.dry_run and not args.dct_index_file.exists():
        raise FileNotFoundError(args.dct_index_file)
    if not args.skip_rgb_baseline and not args.dry_run and not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if not args.skip_rgb_baseline and not args.dry_run and not args.rgb_checkpoint.exists():
        raise FileNotFoundError(args.rgb_checkpoint)
    if args.include_dali_rgb and not args.dry_run and not args.rgb_checkpoint.exists():
        raise FileNotFoundError(args.rgb_checkpoint)
    if not args.skip_dct_baseline and not args.dry_run and not args.dct_checkpoint.exists():
        raise FileNotFoundError(args.dct_checkpoint)
    if not args.skip_galp and not args.dry_run and not args.dct_checkpoint.exists():
        raise FileNotFoundError(args.dct_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = _build_commands(args)
    if not commands:
        raise RuntimeError("all benchmark paths were skipped")
    (args.output_dir / "commands.json").write_text(json.dumps(commands, indent=2, default=str), encoding="utf-8")

    failures: list[tuple[str, int]] = []
    for item in commands:
        env = _galp_env(args) if item["env"] == "galp" else None
        code = _run(
            item["command"],
            env=env,
            dry_run=args.dry_run,
            log_path=args.output_dir / f"{item['name']}.log",
        )
        if code != 0:
            failures.append((item["name"], code))
            if not args.continue_on_error:
                break
    if failures:
        detail = ", ".join(f"{name}={code}" for name, code in failures)
        raise SystemExit(f"benchmark command(s) failed: {detail}")
    print("RESULT_JSON " + json.dumps({"output_dir": str(args.output_dir), "commands": len(commands)}, sort_keys=True))


if __name__ == "__main__":
    main()
