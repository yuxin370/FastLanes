#!/usr/bin/env python3
"""Build a contract and run GALP, RGB-no-more, DALI, and PyTorch end to end."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    CONTRACT_SCHEMA,
    PIPELINES,
    cached_file_fingerprints,
    checkpoint_metadata,
    fingerprint_file,
    galp_manifest_payloads,
    normalize_device,
    sha256_file,
    sha256_json,
    source_tree_metadata,
    write_json,
)
from manifest import build_manifest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_GALP_MANIFEST = REPO_ROOT / "galp/data/imagedataset_dct/ImageNet-val/manifest.bin"
DEFAULT_GALP_LABEL_MAP = REPO_ROOT / "galp/data/imagedataset_dct/ImageNet-val/labels.json"
DEFAULT_BINDING_DIR = REPO_ROOT / "build/galp/torch"
DEFAULT_BENCHMARK_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")


PRESETS = {
    "smoke": {"batch_size": 2, "warmup_batches": 1, "measurement_batches": 2, "repeats": 1, "workers": 1, "semantic_samples": 2},
    "e2e": {"batch_size": 64, "warmup_batches": 5, "measurement_batches": 20, "repeats": 5, "workers": 8, "semantic_samples": 8},
}


def _value(args: argparse.Namespace, key: str) -> int:
    explicit = getattr(args, key)
    return int(PRESETS[args.preset][key] if explicit is None else explicit)


def _write_canonical_index(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Filepath", "Label"))
        writer.writeheader()
        for sample in samples:
            writer.writerow({"Filepath": sample["path"], "Label": sample["label"]})


def _git_metadata() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(command("status", "--porcelain")),
        "diff_sha256": sha256_json(command("diff", "--binary")),
    }


def _build_contract(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], Path]:
    rgbnomore_root = args.rgbnomore_root.resolve()
    data_root = args.data_root.resolve()
    index_csv = (args.index_csv or (rgbnomore_root / "assets/indexbase_val.csv")).resolve()
    rgb_checkpoint = (args.rgb_checkpoint or (rgbnomore_root / "checkpoints/imgnetRGBViTTi_ep300_74.1.pth")).resolve()
    dct_checkpoint = (args.dct_checkpoint or (rgbnomore_root / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth")).resolve()
    galp_manifest = args.galp_manifest.resolve()
    galp_label_map = args.galp_label_map_json.resolve()

    for path in (data_root, index_csv, rgb_checkpoint, dct_checkpoint, galp_manifest, galp_label_map, rgbnomore_root):
        if not path.exists():
            raise FileNotFoundError(path)
    batch_size = _value(args, "batch_size")
    warmup_batches = _value(args, "warmup_batches")
    measurement_batches = _value(args, "measurement_batches")
    repeats = _value(args, "repeats")
    workers = _value(args, "workers")
    semantic_samples = _value(args, "semantic_samples")
    if batch_size <= 0 or measurement_batches <= 0 or repeats <= 0 or workers < 0 or warmup_batches < 0:
        raise ValueError("invalid execution dimensions")
    if (
        args.galp_cache_capacity_mib < 0
        or args.galp_decode_batch_rowgroups <= 0
        or args.galp_rowgroup_prefetch_depth <= 0
        or args.galp_rowgroup_prefetch_workers <= 0
    ):
        raise ValueError("invalid GALP cache/decode dimensions")
    if "dali" in args.pipelines and workers <= 0:
        raise ValueError("DALI requires workers/num_threads > 0")
    if semantic_samples <= 0 or semantic_samples > batch_size * measurement_batches:
        raise ValueError("semantic_samples must be within the measured subset")

    sample_count = batch_size * (warmup_batches + measurement_batches)
    sample_manifest_path = output_dir / "sample_manifest.json"
    sample_manifest, sample_manifest_hash = build_manifest(
        data_root=data_root,
        split=args.split,
        index_csv=index_csv,
        galp_label_map_json=galp_label_map,
        sample_count=sample_count,
        seed=args.seed,
        output=sample_manifest_path,
    )
    canonical_index_csv = output_dir / "canonical_rgbnomore_index.csv"
    _write_canonical_index(canonical_index_csv, sample_manifest["samples"])
    rgb_meta = checkpoint_metadata(rgb_checkpoint)
    dct_meta = checkpoint_metadata(dct_checkpoint)
    device = normalize_device(args.device)
    device_id = int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0
    galp_manifest_fingerprint = fingerprint_file(galp_manifest)
    galp_payload_fingerprints: list[dict[str, Any]] = []
    galp_payload_cache: Path | None = None
    if "galp" in args.pipelines:
        galp_payload_cache = galp_manifest.with_name(galp_manifest.name + ".payload_fingerprints.json")
        galp_payload_fingerprints = cached_file_fingerprints(
            galp_manifest_payloads(galp_manifest),
            galp_payload_cache,
            cache_format="galp_shard_payload_fingerprints_v1",
            allow_hash_misses=args.refresh_galp_payload_fingerprints,
        )

    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "benchmark_id": args.benchmark_id or output_dir.name,
        "preset": args.preset,
        "dataset": {
            "name": "ImageNet-1K",
            "root": str(data_root),
            "split": args.split,
            "full_dataset_size": sample_manifest["full_dataset_size"],
            "eligible_dataset_size": sample_manifest["eligible_dataset_size"],
            "source_index_csv": str(index_csv),
            "source_index_sha256": sample_manifest["source_index_sha256"],
            "sample_manifest": str(sample_manifest_path.resolve()),
            "manifest_sha256": sample_manifest_hash,
            "canonical_index_csv": str(canonical_index_csv.resolve()),
            "canonical_index_sha256": sha256_file(canonical_index_csv),
            "sample_order": "sample_manifest.ordinal",
            "label_mapping": "RGB-no-more ImageNet-1K index CSV; never ImageFolder class ordinals",
        },
        "execution": {
            "batch_size": batch_size,
            "workers": workers,
            "warmup_batches": warmup_batches,
            "measurement_batches": measurement_batches,
            "repeats": repeats,
            "seed": args.seed,
            "device": device,
            "precision": args.precision,
            "drop_last": True,
            "aggregate_exclude_first_repeat": args.preset == "e2e" and repeats > 1,
        },
        "preprocess": {
            "rgb": {
                "name": "rgbnomore_imagenet_eval_rgb_v1",
                "decode": "JPEG_to_RGB",
                "resize_shorter": 256,
                "resize_interpolation": "bilinear_antialias",
                "crop": "center",
                "crop_size": [224, 224],
                "range": [-1.0, 1.0],
                "layout": "NCHW",
            },
            "dct": {
                "name": "rgbnomore_imagenet_eval_dct_v1",
                "decode": "JPEG_quantized_DCT",
                "dequantize": True,
                "transform": "ResizedCenterCrop_DCT(32,28)",
                "range": [-1.0, 1.0],
                "y_shape": [1, 28, 28, 8, 8],
                "cbcr_shape": [2, 14, 14, 8, 8],
            },
        },
        "models": {
            "rgb": {
                "architecture": "RGB-no-more ViT-Ti RGB",
                "input_domain": "RGB",
                "recipe_id": "rgbnomore_imagenet_vitti_300ep_recipe_family",
                "checkpoint": rgb_meta["path"],
                "checkpoint_sha256": rgb_meta["sha256"],
                "checkpoint_size_bytes": rgb_meta["size_bytes"],
            },
            "dct": {
                "architecture": "RGB-no-more JPEG-Ti ViT-Ti DCT",
                "input_domain": "JPEG_DCT",
                "recipe_id": "rgbnomore_imagenet_vitti_300ep_recipe_family",
                "checkpoint": dct_meta["path"],
                "checkpoint_sha256": dct_meta["sha256"],
                "checkpoint_size_bytes": dct_meta["size_bytes"],
            },
            "cross_domain_checkpoint_policy": "RGB and DCT models use recipe-matched domain-specific checkpoints; they are not claimed to be one checkpoint.",
        },
        "pipelines": {
            "enabled": list(args.pipelines),
            "galp": {
                "manifest": str(galp_manifest),
                "manifest_sha256": galp_manifest_fingerprint["sha256"],
                "manifest_fingerprint": galp_manifest_fingerprint,
                "payload_fingerprints": galp_payload_fingerprints,
                "payload_fingerprint_cache": str(galp_payload_cache) if galp_payload_cache is not None else None,
                "label_map_json": str(galp_label_map),
                "label_map_sha256": sha256_file(galp_label_map),
                "torch_binding_dir": str(args.torch_binding_dir.resolve()),
                "preprocess": args.galp_preprocess,
                "cache_capacity_mib": args.galp_cache_capacity_mib,
                "decode_batch_rowgroups": args.galp_decode_batch_rowgroups,
                "rowgroup_prefetch_depth": args.galp_rowgroup_prefetch_depth,
                "rowgroup_prefetch_workers": args.galp_rowgroup_prefetch_workers,
            },
            "rgbnomore": {"root": str(rgbnomore_root), "adapter_policy": "reuse_external_model_dataset_and_transform_code"},
            "dali": {
                "device_id": device_id,
                "prefetch_queue_depth": args.prefetch_factor,
                "reader": "explicit_files_and_manifest_ordinals",
            },
            "pytorch": {"prefetch_factor": args.prefetch_factor, "reader": "canonical_manifest_dataset"},
        },
        "timing": {
            "boundary": "steady_state_batch_request_to_metrics_complete",
            "includes": ["read", "decode", "preprocess", "host_to_device", "model_forward", "top1_top5_accounting"],
            "excludes": ["manifest_creation", "model_load", "pipeline_build", "warmup"],
            "cuda_sync_per_batch": True,
            "next_batch_prefetch_overlap": True,
            "latency_unit": "milliseconds_per_batch",
            "throughput_unit": "images_per_second",
            "os_page_cache_policy": "uncontrolled; e2e aggregate excludes repeat 0 and reports every repeat",
        },
        "semantic_validation": {
            "sample_count": semantic_samples,
            "identity_checks": ["sample_id", "ordinal", "label", "measured_trace_sha256"],
            "comparison_groups": [
                {
                    "pipelines": ["galp", "rgbnomore"],
                    "domain": "dct",
                    "enforcement": "strict",
                    "thresholds": {
                        "input_max_abs": 0.001,
                        "input_mean_abs": 0.0001,
                        "logit_cosine_min": 0.999,
                        "logit_top1_agreement_min": 1.0,
                    },
                },
                {
                    "pipelines": ["dali", "pytorch"],
                    "domain": "rgb",
                    "enforcement": "diagnostic",
                    "thresholds": {"input_max_abs": 0.25, "input_mean_abs": 0.008, "logit_max_abs": 1.0, "logit_cosine_min": 0.995},
                },
            ],
            "cross_domain_boundary": "No elementwise tensor/logit comparison between DCT and RGB models.",
        },
        "source_revisions": {
            "fastlanes": source_tree_metadata(
                REPO_ROOT,
                [
                    "galp/benchmarks/system_rgbnomore/common.py",
                    "galp/benchmarks/system_rgbnomore/manifest.py",
                    "galp/benchmarks/system_rgbnomore/model_factory.py",
                    "galp/benchmarks/system_rgbnomore/pipeline.py",
                    "galp/benchmarks/system_rgbnomore/validate.py",
                    "galp/benchmarks/system_rgbnomore/run.py",
                    "galp/benchmarks/system_rgbnomore/diagnostics/direct_dct.py",
                    "galp/torch/rgbnomore_dct_profile.py",
                ],
            ),
            "rgbnomore": source_tree_metadata(
                rgbnomore_root,
                ["datasets.py", "models/plainvit.py", "utils/custom_transforms.py"],
            ),
        },
    }
    contract_path = output_dir / "contract.json"
    write_json(contract_path, contract)
    return contract, contract_path


def _run_streamed(command: list[str], env: dict[str, str], log_path: Path, dry_run: bool) -> int:
    print("COMMAND " + shlex.join(command), flush=True)
    if dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, contract_path = _build_contract(args, output_dir)
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    env = os.environ.copy()
    commands: list[dict[str, Any]] = []
    for pipeline in contract["pipelines"]["enabled"]:
        command = [
            str(python),
            str(HERE / "pipeline.py"),
            "--pipeline",
            pipeline,
            "--contract",
            str(contract_path),
            "--output",
            str(output_dir / f"pipeline_{pipeline}.json"),
        ]
        pipeline_env = dict(env)
        if pipeline == "galp":
            binding = contract["pipelines"]["galp"]["torch_binding_dir"]
            pipeline_env["PYTHONPATH"] = binding + (os.pathsep + pipeline_env["PYTHONPATH"] if pipeline_env.get("PYTHONPATH") else "")
        commands.append({"name": pipeline, "command": command, "env_overrides": {"PYTHONPATH": pipeline_env.get("PYTHONPATH")} if pipeline == "galp" else {}})
        code = _run_streamed(command, pipeline_env, output_dir / f"pipeline_{pipeline}.log", args.dry_run)
        if code != 0:
            write_json(output_dir / "failed.json", {"pipeline": pipeline, "exit_code": code, "command": command})
            write_json(output_dir / "commands.json", commands)
            return code

    validation_command = [str(python), str(HERE / "validate.py"), "--contract", str(contract_path), "--output-dir", str(output_dir)]
    commands.append({"name": "validate", "command": validation_command, "env_overrides": {}})
    code = _run_streamed(validation_command, env, output_dir / "validate.log", args.dry_run)
    write_json(output_dir / "commands.json", commands)
    write_json(
        output_dir / "run_metadata.json",
        {
            "argv": sys.argv,
            "python_orchestrator": sys.executable,
            "python_benchmark": str(python),
            "host": platform.node(),
            "platform": platform.platform(),
            "git": _git_metadata(),
            "contract_sha256": sha256_json(contract),
            "dry_run": args.dry_run,
        },
    )
    return code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-id")
    parser.add_argument("--python", type=Path, default=DEFAULT_BENCHMARK_PYTHON if DEFAULT_BENCHMARK_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--index-csv", type=Path)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--rgb-checkpoint", type=Path)
    parser.add_argument("--dct-checkpoint", type=Path)
    parser.add_argument("--galp-manifest", type=Path, default=DEFAULT_GALP_MANIFEST)
    parser.add_argument("--galp-label-map-json", type=Path, default=DEFAULT_GALP_LABEL_MAP)
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_BINDING_DIR)
    parser.add_argument("--galp-cache-capacity-mib", type=int, default=1024)
    parser.add_argument("--galp-decode-batch-rowgroups", type=int, default=2)
    parser.add_argument("--galp-rowgroup-prefetch-depth", type=int, default=16)
    parser.add_argument("--galp-rowgroup-prefetch-workers", type=int, default=4)
    parser.add_argument(
        "--refresh-galp-payload-fingerprints",
        action="store_true",
        help="Explicitly hash missing/stale GALP payloads once; normal benchmark runs never scan them implicitly.",
    )
    parser.add_argument(
        "--galp-preprocess",
        choices=("rgbnomore-val-pushdown", "rgbnomore-val"),
        default="rgbnomore-val-pushdown",
        help="GALP DCT implementation; pushdown is the canonical end-to-end path, rgbnomore-val is a diagnostic reference.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-batches", type=int)
    parser.add_argument("--measurement-batches", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--semantic-samples", type=int)
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "amp_fp16", "amp_bf16"), default="fp32")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    code = run(args)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
