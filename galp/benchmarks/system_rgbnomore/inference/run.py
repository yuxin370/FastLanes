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
import time
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from shared.common import (
    CONTRACT_SCHEMA,
    CONTRACT_PIPELINES,
    GALP_PIPELINES,
    INFERENCE_PIPELINES,
    cached_file_fingerprints,
    canonical_pipeline_name,
    checkpoint_metadata,
    fingerprint_file,
    galp_manifest_payloads,
    normalize_device,
    sha256_file,
    sha256_json,
    source_tree_metadata,
    write_json,
)
from dataset.manifest import build_manifest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_E2E_DATA_ROOT = REPO_ROOT / "galp/data/system_rgbnomore/e2e_v2"
DEFAULT_E2E_V3_ROOT = REPO_ROOT / "galp/data/system_rgbnomore/e2e_v3"
DEFAULT_DATA_ROOT = DEFAULT_E2E_V3_ROOT / "imagenet_512"
DEFAULT_INDEX_CSV = DEFAULT_E2E_DATA_ROOT / "indexbase_val.csv"
DEFAULT_RGB_CHECKPOINT = DEFAULT_E2E_DATA_ROOT / "checkpoints/imgnetRGBViTTi_ep300_74.1.pth"
DEFAULT_DCT_CHECKPOINT = DEFAULT_E2E_DATA_ROOT / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
DEFAULT_GALP_MANIFEST = DEFAULT_E2E_V3_ROOT / "compact_v3_tiled_z32_rgbnomore512/manifest.bin"
DEFAULT_GALP_LABEL_MAP = DEFAULT_E2E_V3_ROOT / "compact_v3_tiled_z32_rgbnomore512/labels.json"
DEFAULT_BINDING_DIR = REPO_ROOT / "build/galp/torch"
DEFAULT_BENCHMARK_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")


PRESETS = {
    "smoke": {"batch_size": 2, "warmup_batches": 1, "measurement_batches": 2, "repeats": 1, "workers": 1, "semantic_samples": 2},
    "e2e": {"batch_size": 50, "warmup_batches": 0, "measurement_batches": 1000, "repeats": 5, "workers": 8, "semantic_samples": 8},
}
GALP_E2E_MIN_DALI_HOT_MEDIAN_RATIO = 1.10
E2E_MAX_HOT_THROUGHPUT_CV = 0.05
E2E_PIPELINES = ("galp_planless", "pytorch", "rgbnomore", "dali")


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


def _system_state_snapshot(*, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"scope": "dry_run_not_sampled"}

    def read_text(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError as error:
            return f"unavailable: {error}"

    cpu_fields = read_text("/proc/stat").splitlines()[0].split()
    cpu_ticks = [int(value) for value in cpu_fields[1:]] if cpu_fields and cpu_fields[0] == "cpu" else []
    diskstats: dict[str, dict[str, int]] = {}
    for line in read_text("/proc/diskstats").splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        name = fields[2]
        if not name.startswith(("nvme", "sd", "vd")):
            continue
        diskstats[name] = {
            "reads_completed": int(fields[3]),
            "sectors_read": int(fields[5]),
            "writes_completed": int(fields[7]),
            "sectors_written": int(fields[9]),
            "io_ms": int(fields[12]),
        }
    meminfo = {}
    for line in read_text("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "Cached", "Dirty"}:
            meminfo[key] = value.strip()
    gpu_query = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,driver_version,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = subprocess.run(gpu_query, check=False, capture_output=True, text=True, timeout=10)
        gpu_state: dict[str, Any] = {
            "command": gpu_query,
            "exit_code": gpu.returncode,
            "rows": [row.strip() for row in gpu.stdout.splitlines() if row.strip()],
            "stderr": gpu.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_state = {"command": gpu_query, "error": str(error)}
    return {
        "captured_at_unix_ns": time.time_ns(),
        "cpu_ticks": cpu_ticks,
        "loadavg": read_text("/proc/loadavg"),
        "meminfo": meminfo,
        "diskstats": diskstats,
        "gpu": gpu_state,
    }


def _system_state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_ticks = before.get("cpu_ticks", [])
    after_ticks = after.get("cpu_ticks", [])
    cpu_utilization: float | None = None
    if len(before_ticks) == len(after_ticks) and len(before_ticks) >= 5:
        deltas = [max(0, int(end) - int(start)) for start, end in zip(before_ticks, after_ticks)]
        total = sum(deltas)
        idle = deltas[3] + deltas[4]
        cpu_utilization = (total - idle) / total if total else None
    disk_delta: dict[str, dict[str, int]] = {}
    for name in sorted(set(before.get("diskstats", {})) & set(after.get("diskstats", {}))):
        disk_delta[name] = {
            key: max(0, int(after["diskstats"][name][key]) - int(before["diskstats"][name][key]))
            for key in before["diskstats"][name]
        }
    return {
        "elapsed_seconds": (
            (int(after["captured_at_unix_ns"]) - int(before["captured_at_unix_ns"])) / 1e9
            if "captured_at_unix_ns" in before and "captured_at_unix_ns" in after
            else None
        ),
        "cpu_utilization_ratio": cpu_utilization,
        "diskstats_delta": disk_delta,
    }


def _source_revision_policy(source_revisions: dict[str, Any]) -> dict[str, Any]:
    dirty_source_details = {
        name: revision.get("runtime_git_status", [])
        for name, revision in source_revisions.items()
        if not bool(revision.get("benchmark_source_clean", False))
    }
    return {
        "cleanliness_enforcement": "warning",
        "dirty_sources_at_contract_creation": dirty_source_details,
        "runtime_file_hashes_recorded": True,
        "runtime_file_changes_during_benchmark_are_errors": True,
    }


def _build_contract(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], Path]:
    rgbnomore_root = args.rgbnomore_root.resolve()
    data_root = args.data_root.resolve()
    index_csv = (args.index_csv or DEFAULT_INDEX_CSV).resolve()
    rgb_checkpoint = (args.rgb_checkpoint or DEFAULT_RGB_CHECKPOINT).resolve()
    dct_checkpoint = (args.dct_checkpoint or DEFAULT_DCT_CHECKPOINT).resolve()
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
        or args.galp_plan_cache_capacity < 0
        or args.galp_decode_batch_rowgroups <= 0
        or args.galp_batch_prefetch_depth <= 0
        or args.galp_rowgroup_prefetch_depth <= 0
        or args.galp_rowgroup_prefetch_workers <= 0
        or args.galp_rowgroup_prefetch_min_decode_batches <= 0
        or args.galp_transform_blocks_per_launch < 0
        or args.galp_transform_ctas_per_launch < 0
    ):
        raise ValueError("invalid GALP cache/decode dimensions")
    if args.galp_scheduling_policy == "limited-overlap" and args.galp_transform_blocks_per_launch <= 0:
        raise ValueError("limited-overlap requires --galp-transform-blocks-per-launch > 0")
    if args.galp_scheduling_policy == "limited-overlap" and args.galp_transform_ctas_per_launch <= 0:
        raise ValueError("limited-overlap requires --galp-transform-ctas-per-launch > 0")
    if "dali" in args.pipelines and workers <= 0:
        raise ValueError("DALI requires workers/num_threads > 0")
    if args.preset == "e2e" and not set(E2E_PIPELINES).issubset(args.pipelines):
        raise ValueError(
            "e2e requires same-round galp_planless, pytorch, rgbnomore, and dali pipelines"
        )
    if args.preset == "e2e" and (
        args.galp_cache_capacity_mib != 0 or args.galp_plan_cache_capacity != 0
    ):
        raise ValueError("e2e requires decoded-rowgroup cache=0 and exact-batch plan cache=0")
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
        expected_image_size=args.dct_source_image_size or None,
    )
    canonical_index_csv = output_dir / "canonical_rgbnomore_index.csv"
    _write_canonical_index(canonical_index_csv, sample_manifest["samples"])
    rgb_meta = checkpoint_metadata(rgb_checkpoint)
    dct_meta = checkpoint_metadata(dct_checkpoint)
    device = normalize_device(args.device)
    device_id = int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0
    galp_manifest_fingerprint = fingerprint_file(galp_manifest)
    manifest_header = galp_manifest.read_bytes()[:12]
    if len(manifest_header) != 12 or manifest_header[:8] != b"GJDCTSH1":
        raise ValueError(f"unexpected GALP shard manifest format: {galp_manifest}")
    galp_manifest_version = int.from_bytes(manifest_header[8:12], "little")
    galp_payload_fingerprints: list[dict[str, Any]] = []
    galp_vector_bundle_fingerprints: list[dict[str, Any]] = []
    galp_payload_cache: Path | None = None
    galp_native_binary: dict[str, Any] | None = None
    if set(GALP_PIPELINES).intersection(args.pipelines):
        galp_payload_cache = galp_manifest.with_name(galp_manifest.name + ".payload_fingerprints.json")
        galp_payload_fingerprints = cached_file_fingerprints(
            galp_manifest_payloads(galp_manifest),
            galp_payload_cache,
            cache_format="galp_shard_payload_fingerprints_v1",
            allow_hash_misses=args.refresh_galp_payload_fingerprints,
        )
        galp_vector_bundle_fingerprints = [
            item for item in galp_payload_fingerprints if item.get("kind") == "vector_bundle"
        ]
        binding_candidates = sorted(args.torch_binding_dir.glob("_galp_direct_dct*.so"))
        if len(binding_candidates) != 1:
            raise FileNotFoundError(
                f"expected one _galp_direct_dct shared library in {args.torch_binding_dir}, got {binding_candidates}"
            )
        galp_native_binary = fingerprint_file(binding_candidates[0])

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
            "source_image_geometry": sample_manifest["source_geometry"],
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
            "model_stream_priority": "greatest",
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
            "galp_planless": {
                "manifest": str(galp_manifest),
                "manifest_version": galp_manifest_version,
                "manifest_sha256": galp_manifest_fingerprint["sha256"],
                "manifest_fingerprint": galp_manifest_fingerprint,
                "payload_fingerprints": galp_payload_fingerprints,
                "payload_fingerprint_cache": str(galp_payload_cache) if galp_payload_cache is not None else None,
                "vector_bundle_payload_count": len(galp_vector_bundle_fingerprints),
                "vector_bundle_payload_bytes": sum(
                    int(item["size_bytes"]) for item in galp_vector_bundle_fingerprints
                ),
                "vector_bundle_read_supported": bool(galp_vector_bundle_fingerprints)
                and len(galp_vector_bundle_fingerprints)
                == sum(item.get("kind") == "fls" for item in galp_payload_fingerprints),
                "native_binary_fingerprint": galp_native_binary,
                "label_map_json": str(galp_label_map),
                "label_map_sha256": sha256_file(galp_label_map),
                "torch_binding_dir": str(args.torch_binding_dir.resolve()),
                "preprocess": args.galp_preprocess,
                "cache_capacity_mib": args.galp_cache_capacity_mib,
                "plan_cache_capacity": args.galp_plan_cache_capacity,
                "decode_batch_rowgroups": args.galp_decode_batch_rowgroups,
                "batch_prefetch_depth": args.galp_batch_prefetch_depth,
                "async_planless_completion": args.galp_async_planless_completion,
                "rowgroup_prefetch_depth": args.galp_rowgroup_prefetch_depth,
                "rowgroup_prefetch_workers": args.galp_rowgroup_prefetch_workers,
                "rowgroup_prefetch_min_decode_batches": args.galp_rowgroup_prefetch_min_decode_batches,
                "transform_execution_mode": "require-planless",
                "crop_execution_mode": args.galp_crop_execution_mode,
                "scheduling_policy": args.galp_scheduling_policy,
                "transform_blocks_per_launch": (
                    args.galp_transform_blocks_per_launch
                    if args.galp_scheduling_policy == "limited-overlap"
                    else 0
                ),
                "transform_ctas_per_launch": (
                    args.galp_transform_ctas_per_launch
                    if args.galp_scheduling_policy == "limited-overlap"
                    else 0
                ),
                "use_low_priority_streams": True,
            },
            "galp_fixed_items": {
                "manifest": str(galp_manifest),
                "manifest_version": galp_manifest_version,
                "manifest_sha256": galp_manifest_fingerprint["sha256"],
                "manifest_fingerprint": galp_manifest_fingerprint,
                "payload_fingerprints": galp_payload_fingerprints,
                "payload_fingerprint_cache": str(galp_payload_cache) if galp_payload_cache is not None else None,
                "vector_bundle_payload_count": len(galp_vector_bundle_fingerprints),
                "vector_bundle_payload_bytes": sum(
                    int(item["size_bytes"]) for item in galp_vector_bundle_fingerprints
                ),
                "vector_bundle_read_supported": bool(galp_vector_bundle_fingerprints)
                and len(galp_vector_bundle_fingerprints)
                == sum(item.get("kind") == "fls" for item in galp_payload_fingerprints),
                "native_binary_fingerprint": galp_native_binary,
                "label_map_json": str(galp_label_map),
                "label_map_sha256": sha256_file(galp_label_map),
                "torch_binding_dir": str(args.torch_binding_dir.resolve()),
                "preprocess": args.galp_preprocess,
                "cache_capacity_mib": args.galp_cache_capacity_mib,
                "plan_cache_capacity": 0,
                "decode_batch_rowgroups": args.galp_decode_batch_rowgroups,
                "batch_prefetch_depth": args.galp_batch_prefetch_depth,
                "rowgroup_prefetch_depth": args.galp_rowgroup_prefetch_depth,
                "rowgroup_prefetch_workers": args.galp_rowgroup_prefetch_workers,
                "rowgroup_prefetch_min_decode_batches": args.galp_rowgroup_prefetch_min_decode_batches,
                "transform_execution_mode": "require-fixed-items",
                "crop_execution_mode": args.galp_crop_execution_mode,
                "scheduling_policy": args.galp_scheduling_policy,
                "transform_blocks_per_launch": 0,
                "transform_ctas_per_launch": 0,
                "use_low_priority_streams": True,
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
            "cuda_sync_scope": "model_stream_only",
            "cuda_device_sync_per_batch": False,
            "next_batch_prefetch_overlap": args.galp_scheduling_policy != "serial",
            "galp_batch_prefetch_depth": args.galp_batch_prefetch_depth,
            "latency_unit": "milliseconds_per_batch",
            "throughput_unit": "images_per_second",
            "os_page_cache_policy": "uncontrolled; e2e aggregate excludes repeat 0 and reports every repeat",
        },
        "performance_gates": {
            "galp_planless": {
                "minimum_median_throughput_images_per_s": None,
                "minimum_hot_median_to_dali_hot_median_ratio": (
                    GALP_E2E_MIN_DALI_HOT_MEDIAN_RATIO if args.preset == "e2e" else None
                ),
                "require_hot_min_above_dali_hot_median": args.preset == "e2e",
                "maximum_hot_throughput_cv": E2E_MAX_HOT_THROUGHPUT_CV if args.preset == "e2e" else None,
                "planning_median_ms_max": 2.0 if args.preset == "e2e" else None,
                "planning_p95_ms_max": 3.0 if args.preset == "e2e" else None,
                "device_mapping_median_ms_max": 1.0 if args.preset == "e2e" else None,
                "device_mapping_plus_fixed_transform_median_ms_max": 6.5 if args.preset == "e2e" else None,
                "image_major_manifest_minimum_version": 2,
                "manifest_v2_rowgroups_per_image": 1,
                "worksets_per_batch": 1,
                "internal_syncs_per_batch": 0 if args.galp_async_planless_completion else 1,
                "async_planless_completion_batches_per_batch": (
                    1 if args.galp_async_planless_completion else 0
                ),
                "decode_kernels_per_batch": 1,
            }
        },
        "semantic_validation": {
            "sample_count": semantic_samples,
            "prediction_agreement_sample_count": batch_size * measurement_batches,
            "identity_checks": ["sample_id", "ordinal", "label", "measured_trace_sha256"],
            "comparison_groups": [
                {
                    "pipelines": ["galp_planless", "rgbnomore"],
                    "domain": "dct",
                    "enforcement": "strict",
                    "thresholds": {
                        "input_max_abs": 0.001,
                        "input_mean_abs": 0.0001,
                        "logit_cosine_min": 0.999,
                        "logit_top1_agreement_min": 1.0,
                        "full_prediction_top1_agreement_min": 1.0,
                        "full_prediction_sample_count": batch_size * measurement_batches,
                    },
                },
                {
                    "pipelines": ["galp_planless", "galp_fixed_items"],
                    "domain": "dct",
                    "enforcement": "strict",
                    "thresholds": {
                        "input_max_abs": 0.0,
                        "input_mean_abs": 0.0,
                        "logit_max_abs": 0.0,
                        "logit_cosine_min": 0.999999999,
                        "logit_top1_agreement_min": 1.0,
                        "full_prediction_top1_agreement_min": 1.0,
                        "full_prediction_sample_count": batch_size * measurement_batches,
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
                    "galp/benchmarks/system_rgbnomore/shared/common.py",
                    "galp/benchmarks/system_rgbnomore/dataset/manifest.py",
                    "galp/benchmarks/system_rgbnomore/inference/model_factory.py",
                    "galp/benchmarks/system_rgbnomore/inference/crop_io_ab.py",
                    "galp/benchmarks/system_rgbnomore/inference/pipeline.py",
                    "galp/benchmarks/system_rgbnomore/inference/validate.py",
                    "galp/benchmarks/system_rgbnomore/inference/run.py",
                    "galp/benchmarks/system_rgbnomore/docs/PLANLESS_DIRECT_DCT_RFC.md",
                    "galp/benchmarks/system_rgbnomore/diagnostics/audit_planless_storage_io.py",
                    "galp/benchmarks/system_rgbnomore/diagnostics/benchmark_planless_planning.py",
                    "galp/benchmarks/system_rgbnomore/diagnostics/direct_dct.py",
                    "galp/benchmarks/system_rgbnomore/diagnostics/scheduler_matrix.py",
                    "galp/benchmarks/system_rgbnomore/diagnostics/validate_pushdown.py",
                    "galp/include/galp/direct_dct.hpp",
                    "galp/include/galp/jpeg_dct_device.hpp",
                    "galp/include/galp/jpeg_dct_diagnostics.hpp",
                    "galp/include/galp/jpeg_dct_format.hpp",
                    "galp/include/galp/jpeg_dct_storage.hpp",
                    "galp/include/galp/sparse_vector_bundle.hpp",
                    "galp/include/galp/jpeg_dct.hpp",
                    "galp/src/api/direct_dct.cpp",
                    "galp/src/cuda/memory/device_arena.cu",
                    "galp/src/cuda/memory/device_arena.cuh",
                    "galp/src/cuda/memory/device_pool.cuh",
                    "galp/src/cuda/memory/upload_metrics.cuh",
                    "galp/src/engine/materialization/metadata.cu",
                    "galp/src/engine/operators/batch.cuh",
                    "galp/src/format/reader.cu",
                    "galp/src/format/reader.cuh",
                    "galp/src/format/rowgroup_io.cuh",
                    "galp/src/engine/pipeline/rowgroup_prefetch_queue.cuh",
                    "galp/src/engine/pipeline/rowgroup_prefetch_types.cuh",
                    "galp/src/engine/workset/append.cuh",
                    "galp/src/engine/workset/model.cuh",
                    "galp/src/engine/workset/upload.cu",
                    "galp/src/jpeg/jpeg_dct_planner.cpp",
                    "galp/src/jpeg/jpeg_dct_shard_reader.cpp",
                    "galp/src/jpeg/jpeg_dct_shard_writer.cpp",
                    "galp/src/jpeg/jpeg_dct_metadata.cpp",
                    "galp/src/jpeg/jpeg_dct_decode.cpp",
                    "galp/src/jpeg/jpeg_dct_device_bridge.cpp",
                    "galp/src/jpeg/jpeg_dct_device.cu",
                    "galp/src/jpeg/jpeg_dct_gather_kernels.cu",
                    "galp/src/jpeg/jpeg_dct_transform_kernels.cu",
                    "galp/src/jpeg/jpeg_dct_plan_types.hpp",
                    "galp/src/jpeg/jpeg_dct_policy.hpp",
                    "galp/src/jpeg/jpeg_dct_policy.cpp",
                    "galp/src/jpeg/jpeg_dct_device_runtime.hpp",
                    "galp/src/jpeg/jpeg_dct_cuda_internal.cuh",
                    "galp/tests/jpeg_dct_test.cpp",
                    "galp/tests/test_system_benchmark.py",
                    "galp/torch/direct_dct_torch.cpp",
                    "galp/torch/rgbnomore_dct_profile.py",
                ],
            ),
            "rgbnomore": source_tree_metadata(
                rgbnomore_root,
                ["datasets.py", "models/plainvit.py", "utils/custom_transforms.py"],
            ),
        },
    }
    contract["source_revision_policy"] = _source_revision_policy(contract["source_revisions"])
    dirty_source_details = contract["source_revision_policy"]["dirty_sources_at_contract_creation"]
    if args.preset == "e2e" and dirty_source_details:
        print(
            "WARNING: e2e contract includes uncommitted benchmark runtime sources; "
            "results remain traceable through recorded status and per-file SHA-256 values: "
            f"{dirty_source_details}",
            file=sys.stderr,
            flush=True,
        )
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
        if pipeline in GALP_PIPELINES:
            binding = contract["pipelines"][pipeline]["torch_binding_dir"]
            pipeline_env["PYTHONPATH"] = binding + (os.pathsep + pipeline_env["PYTHONPATH"] if pipeline_env.get("PYTHONPATH") else "")
        command_record = {
            "name": pipeline,
            "command": command,
            "env_overrides": {"PYTHONPATH": pipeline_env.get("PYTHONPATH")} if pipeline in GALP_PIPELINES else {},
            "system_state_before": _system_state_snapshot(dry_run=args.dry_run),
        }
        commands.append(command_record)
        code = _run_streamed(command, pipeline_env, output_dir / f"pipeline_{pipeline}.log", args.dry_run)
        command_record["system_state_after"] = _system_state_snapshot(dry_run=args.dry_run)
        command_record["system_state_delta"] = _system_state_delta(
            command_record["system_state_before"], command_record["system_state_after"]
        )
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
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=CONTRACT_PIPELINES,
        default=None,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--index-csv", type=Path)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--rgb-checkpoint", type=Path)
    parser.add_argument("--dct-checkpoint", type=Path)
    parser.add_argument("--galp-manifest", type=Path, default=DEFAULT_GALP_MANIFEST)
    parser.add_argument("--galp-label-map-json", type=Path, default=DEFAULT_GALP_LABEL_MAP)
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_BINDING_DIR)
    parser.add_argument(
        "--galp-cache-capacity-mib",
        type=int,
        default=0,
        help="Decoded-rowgroup cache size; random-access image-major training defaults to zero-copy streaming.",
    )
    parser.add_argument(
        "--galp-plan-cache-capacity",
        type=int,
        default=0,
        help="Number of transformed batch plans retained; canonical runs disable the batch-plan cache.",
    )
    parser.add_argument(
        "--galp-decode-batch-rowgroups",
        type=int,
        default=64,
        help="Legacy-layout compatibility limit; the image-major production path submits one logical batch.",
    )
    parser.add_argument(
        "--galp-batch-prefetch-depth",
        type=int,
        default=2,
        help="Ordered GALP batch lookahead; depth 2 overlaps two native reads with the current model forward.",
    )
    parser.add_argument(
        "--galp-async-planless-completion",
        action="store_true",
        help="Return planless batches after event-owned submission and serialize their transform after the prior model.",
    )
    parser.add_argument(
        "--galp-scheduling-policy",
        choices=("fully-overlapped", "limited-overlap", "serial"),
        default="limited-overlap",
        help="Direct-DCT/model overlap policy; production defaults to the measured 512-output/512-CTA limited point.",
    )
    parser.add_argument(
        "--galp-crop-execution-mode",
        choices=(
            "auto",
            "full-rowgroup-decode",
            "rowgroup-read-selected-decode",
            "vector-range-read-selected-decode",
        ),
        default="auto",
        help="Select a storage/decode granularity for same-contract crop A/B measurements.",
    )
    parser.add_argument(
        "--galp-transform-blocks-per-launch",
        type=int,
        default=512,
        help="Maximum planless output blocks per limited-overlap launch; CTA concurrency is configured separately.",
    )
    parser.add_argument(
        "--galp-transform-ctas-per-launch",
        type=int,
        default=512,
        help="Maximum planless CUDA CTAs per limited-overlap launch (measured production default: 512).",
    )
    parser.add_argument("--galp-rowgroup-prefetch-depth", type=int, default=16)
    parser.add_argument("--galp-rowgroup-prefetch-workers", type=int, default=4)
    parser.add_argument(
        "--galp-rowgroup-prefetch-min-decode-batches",
        type=int,
        default=1,
        help="Enable parallel rowgroup reads for a single image-major decode workset.",
    )
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
    parser.add_argument(
        "--dct-source-image-size",
        type=int,
        default=512,
        help=(
            "Required square JPEG source size before DCT extraction. The published RGB-no-more "
            "DCT checkpoints require 512; use 0 only for a custom checkpoint with a different recipe."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "amp_fp16", "amp_bf16"), default="fp32")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pipelines is None:
        args.pipelines = list(E2E_PIPELINES if args.preset == "e2e" else INFERENCE_PIPELINES)
    else:
        args.pipelines = [canonical_pipeline_name(name) for name in args.pipelines]
        if len(set(args.pipelines)) != len(args.pipelines):
            parser.error("--pipelines contains duplicate canonical names after alias normalization")
    if args.dct_source_image_size < 0:
        parser.error("--dct-source-image-size must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    code = run(args)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
