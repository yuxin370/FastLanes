#!/usr/bin/env python3
"""Run the auditable four-pipeline RGB-no-more training benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from shared.common import GALP_RUNTIME_PROFILE, cached_file_fingerprints

from training.artifacts import (
    nested_state_sha256,
    repository_provenance,
    runtime_metadata,
    sha256_file,
    sha256_json,
    tensor_state_sha256,
    verify_artifact_hashes,
    write_artifact_hashes,
    write_json,
)
from training.augmentation import AugmentationDecision, augmentation_contract, derive_augmentation
from training.direct_dct_reader import (
    NativeExecutionStatsAccumulator,
    merge_native_counter_snapshot,
    native_allocation_stability,
)
from training.manifest_preflight import SUPPORTED_LAYOUTS, ManifestPreflight, preflight_manifest
from training.metrics import (
    coefficient_of_variation,
    distribution,
    gradient_summary,
    nested_tensors_finite,
    parameter_update_summary,
    process_memory,
    tensor_is_finite,
    topk_accuracy,
)
from training.model_factory import (
    MODEL_ARCHITECTURE,
    build_model,
    capture_rng_state,
    capture_training_state,
    initialize_model,
    model_configuration,
    reset_training_state,
    restore_rng_state,
    rng_state_artifact,
    seed_everything,
)
from training.optimizer import (
    build_optimizer,
    build_scheduler,
    resolved_optimizer_config,
    resolved_scheduler_config,
)
from training.pipeline import (
    TrainingBatch,
    TrainingPipelineAdapter,
    TrainingSample,
    build_training_adapter,
    load_training_manifest,
    validate_dataset_separation,
)
from training.pls_experiment import (
    PlsExecutionPlan,
    augmentation_batches as pls_augmentation_batches,
    build_execution_plan as build_pls_execution_plan,
    condition_config as pls_condition_config,
)
from training.sample_order import (
    SampleIdentity,
    SampleOrderLedger,
    batch_stream,
    canonical_epoch_order,
)
from training.schema import (
    COMPARISON_GROUPS,
    DOMAINS,
    PIPELINES,
    TRAINING_CONTRACT_SCHEMA,
    TRAINING_PIPELINE_SCHEMA,
    TRAINING_RESULT_SCHEMA,
    empty_status,
    validate_comparison_groups,
    validate_pipeline_names,
    validate_required_group_coverage,
)


HERE = Path(__file__).resolve().parent
FASTLANES_ROOT = HERE.parents[3]
PHASES = ("smoke", "step", "convergence")
DEFAULT_PERFORMANCE_REPEATS = 5
DEFAULT_THROUGHPUT_CV_LIMIT = 0.10


def _csv_values(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _parse_seeds(values: str | None, seed: int) -> list[int]:
    if not values:
        return [seed, seed + 1, seed + 2]
    parsed = [int(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("--seeds must contain at least one integer")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", action="append", choices=PIPELINES)
    parser.add_argument("--enabled-pipelines", action="append")
    parser.add_argument("--required-comparison-groups", action="append", default=[])
    parser.add_argument("--phase", choices=("smoke", "step", "convergence", "all"), default="smoke")
    parser.add_argument(
        "--execution-mode",
        choices=("audit", "runtime"),
        default="audit",
        help="audit preserves exhaustive per-step checks; runtime minimizes measured-path synchronization",
    )

    parser.add_argument("--init-mode", choices=("random", "weights", "full-checkpoint"), default="random")
    parser.add_argument("--rgb-init-checkpoint", type=Path)
    parser.add_argument("--dct-init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-run", type=Path)

    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--val-root", type=Path)
    parser.add_argument("--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more"))
    parser.add_argument(
        "--galp-manifest",
        type=Path,
        help="GALP training manifest; retained as the stable train-side option name.",
    )
    parser.add_argument(
        "--galp-validation-manifest",
        type=Path,
        help=(
            "Independent GALP validation manifest. If omitted, validation reuses "
            "--galp-manifest for legacy held-out/canary runs."
        ),
    )
    parser.add_argument("--galp-torch-module-path", type=Path, default=FASTLANES_ROOT / "build/galp/torch")
    parser.add_argument("--refresh-galp-payload-fingerprints", action="store_true")
    parser.add_argument(
        "--allow-galp-layout-manifest-rebinding",
        action="store_true",
        help=(
            "Allow a training JSON bound to one GALP manifest to be used with an "
            "explicitly supplied layout-equivalent manifest. Intended for audited "
            "v2/v3 physical-layout comparisons; image-ID bounds are still checked."
        ),
    )
    parser.add_argument("--expected-manifest-version", type=int, choices=tuple(sorted(SUPPORTED_LAYOUTS)))
    parser.add_argument(
        "--expected-physical-layout",
        choices=tuple(SUPPORTED_LAYOUTS[version] for version in sorted(SUPPORTED_LAYOUTS)),
    )
    parser.add_argument("--expected-spatial-order")
    parser.add_argument("--expected-image-count", type=int)
    parser.add_argument("--expected-validation-image-count", type=int)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--distributed-rank", type=int, default=0)
    parser.add_argument("--distributed-world-size", type=int, default=1)
    parser.add_argument(
        "--pls-experiment",
        action="store_true",
        help=(
            "enable the PLS training-effect experiment schedule; this runner path "
            "emulates closed-wave order but does not claim complete GPU-pool materialization"
        ),
    )
    parser.add_argument(
        "--pls-condition-id",
        help="pre-registered condition ID (A0-A3, B0-B7, C0-C2, or D0-D3)",
    )
    parser.add_argument(
        "--pls-organization",
        choices=("current", "storage-hash", "storage-stratified", "runtime-balanced"),
        default="current",
    )
    parser.add_argument(
        "--pls-crop-policy", choices=("per-sample", "per-shard"), default="per-shard"
    )
    parser.add_argument(
        "--pls-order-policy", choices=("global", "pls-wave"), default="pls-wave"
    )
    parser.add_argument("--pls-segment-images", type=int, default=1024)
    parser.add_argument("--pls-segments-per-pool", type=int, default=4)
    parser.add_argument("--pls-organization-seed", type=int, default=20260810)
    parser.add_argument(
        "--pls-gpu-pool",
        action="store_true",
        help=(
            "materialize each complete scheduled pool in one GALP Direct-DCT request; "
            "requires a single galp pipeline"
        ),
    )

    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--seeds")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--measured-steps", type=int)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--repeats", type=int)

    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--scheduler", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--scheduler-warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clipping", type=float)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--randaugment", type=int, default=0)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--throughput-cv-limit", type=float, default=DEFAULT_THROUGHPUT_CV_LIMIT)
    parser.add_argument(
        "--dct-semantic-atol",
        type=float,
        default=1.0 / 1020.0 + 1e-7,
        help="absolute DCT gate; default permits one normalized integer-coefficient step",
    )
    parser.add_argument("--dct-semantic-rtol", type=float, default=1e-4)
    parser.add_argument("--rgb-semantic-warning-atol", type=float, default=0.10)
    parser.add_argument("--rgb-semantic-failure-atol", type=float, default=0.50)
    parser.add_argument("--semantic-gradient-cosine-dct", type=float, default=0.999)
    parser.add_argument("--semantic-gradient-cosine-rgb", type=float, default=0.99)
    parser.add_argument("--dry-run-contract", action="store_true")
    parsed = parser.parse_args(argv)
    tokens = list(sys.argv[1:] if argv is None else argv)
    parsed.execution_mode_explicit = any(
        token == "--execution-mode" or token.startswith("--execution-mode=")
        for token in tokens
    )
    return parsed


def _validate_args(args: argparse.Namespace) -> None:
    enabled_values = list(args.pipeline or []) + _csv_values(args.enabled_pipelines)
    args.enabled = validate_pipeline_names(enabled_values or PIPELINES)
    args.required_groups = validate_comparison_groups(_csv_values(args.required_comparison_groups))
    validate_required_group_coverage(args.enabled, args.required_groups)
    if args.resume_run is None and args.output_dir is None:
        raise ValueError("--output-dir is required unless --resume-run is used")
    if args.resume_run is None and (args.train_manifest is None or args.val_manifest is None):
        raise ValueError("--train-manifest and --val-manifest are required")
    if args.batch_size <= 0 or args.workers < 0 or args.prefetch_depth < 0:
        raise ValueError("batch-size must be positive; workers/prefetch-depth must be non-negative")
    if args.distributed_world_size <= 0:
        raise ValueError("--distributed-world-size must be positive")
    if args.distributed_rank < 0 or args.distributed_rank >= args.distributed_world_size:
        raise ValueError("--distributed-rank must be in [0, --distributed-world-size)")
    if args.pls_condition_id is not None:
        if not args.pls_experiment:
            raise ValueError("--pls-condition-id requires --pls-experiment")
        condition = pls_condition_config(args.pls_condition_id)
        args.pls_organization = condition["organization"]
        args.pls_crop_policy = condition["crop_policy"]
        args.pls_order_policy = condition["order_policy"]
        args.pls_segment_images = int(condition["segment_images"])
        args.pls_segments_per_pool = int(condition["segments_per_pool"])
        args.pls_resolved_condition = condition
    else:
        args.pls_resolved_condition = None
    if args.pls_segment_images <= 0 or args.pls_segments_per_pool <= 0:
        raise ValueError("PLS segment-images and segments-per-pool must be positive")
    if args.pls_experiment and args.batch_size != 64:
        raise ValueError("the pre-registered PLS experiment fixes --batch-size=64")
    if args.pls_experiment and args.distributed_world_size != 1:
        raise ValueError(
            "PLS distributed closed-wave ownership is not yet part of the "
            "pre-registered experiment; use --distributed-world-size=1"
        )
    if args.pls_gpu_pool and not args.pls_experiment:
        raise ValueError("--pls-gpu-pool requires --pls-experiment")
    if args.pls_gpu_pool and args.enabled != ["galp"]:
        raise ValueError("--pls-gpu-pool requires exactly --pipeline galp")
    if args.pls_gpu_pool and args.drop_last:
        raise ValueError(
            "--pls-gpu-pool requires --no-drop-last so every materialized pool "
            "sample is consumed before release"
        )
    if args.pls_gpu_pool and args.phase != "convergence":
        raise ValueError("--pls-gpu-pool currently requires --phase convergence")
    if args.pls_gpu_pool and args.pls_order_policy != "pls-wave":
        raise ValueError("--pls-gpu-pool requires bounded --pls-order-policy=pls-wave")
    for name in ("expected_image_count", "expected_validation_image_count"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "throughput_cv_limit",
        "dct_semantic_atol",
        "dct_semantic_rtol",
        "rgb_semantic_warning_atol",
        "rgb_semantic_failure_atol",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.rgb_semantic_failure_atol < args.rgb_semantic_warning_atol:
        raise ValueError(
            "--rgb-semantic-failure-atol must be at least --rgb-semantic-warning-atol"
        )
    for name in ("semantic_gradient_cosine_dct", "semantic_gradient_cosine_rgb"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and in [0,1]")
    for name in ("warmup_steps", "measured_steps", "train_steps", "eval_interval", "repeats"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.mixup != 0.0 or args.cutmix != 0.0 or args.randaugment != 0:
        raise ValueError("v1 default recipe requires mixup=0, cutmix=0, and randaugment=0")
    if "galp" in args.enabled:
        if args.galp_manifest is None or not args.galp_manifest.is_file():
            raise ValueError("the GALP pipeline requires --galp-manifest")
        if args.galp_validation_manifest is None:
            args.galp_validation_manifest = args.galp_manifest
        if not args.galp_validation_manifest.is_file():
            raise ValueError(
                "the GALP pipeline requires a valid --galp-validation-manifest"
            )
        args.galp_manifest_preflight = preflight_manifest(
            args.galp_manifest,
            expected_manifest_version=args.expected_manifest_version,
            expected_physical_layout=args.expected_physical_layout,
            expected_spatial_order=args.expected_spatial_order,
            expected_image_count=args.expected_image_count,
        )
        args.galp_validation_manifest_preflight = preflight_manifest(
            args.galp_validation_manifest,
            expected_manifest_version=args.expected_manifest_version,
            expected_physical_layout=args.expected_physical_layout,
            expected_spatial_order=args.expected_spatial_order,
            expected_image_count=(
                args.expected_validation_image_count
                if args.expected_validation_image_count is not None
                else (
                    args.expected_image_count
                    if args.galp_validation_manifest.resolve()
                    == args.galp_manifest.resolve()
                    else None
                )
            ),
        )
    else:
        args.galp_manifest_preflight = None
        args.galp_validation_manifest_preflight = None
    if args.init_mode == "random" and any((args.rgb_init_checkpoint, args.dct_init_checkpoint, args.resume_checkpoint)):
        raise ValueError("random init mode cannot be combined with initialization/resume checkpoints")
    for name in ("rgb_init_checkpoint", "dct_init_checkpoint", "resume_checkpoint"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.init_mode == "weights" and args.resume_checkpoint is not None:
        raise ValueError("--resume-checkpoint requires --init-mode full-checkpoint")
    domains = {DOMAINS[pipeline] for pipeline in args.enabled}
    if args.init_mode == "weights":
        if "rgb" in domains and args.rgb_init_checkpoint is None:
            raise ValueError("weights mode requires --rgb-init-checkpoint for an enabled RGB pipeline")
        if "dct" in domains and args.dct_init_checkpoint is None:
            raise ValueError("weights mode requires --dct-init-checkpoint for an enabled DCT pipeline")
    if args.init_mode == "full-checkpoint":
        if args.resume_checkpoint is not None and len(domains) > 1:
            raise ValueError(
                "--resume-checkpoint restores one domain; cross-domain runs require domain-specific checkpoints"
            )
        if args.resume_checkpoint is None and len(domains) > 1 and (
            args.rgb_init_checkpoint is None or args.dct_init_checkpoint is None
        ):
            raise ValueError(
                "cross-domain full-checkpoint runs require both domain-specific init checkpoints"
            )
    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.dry_run_contract:
        raise RuntimeError(
            "CUDA was requested but is unavailable; this is an environment skip, not a training correctness failure"
        )


def _phase_config(args: argparse.Namespace) -> dict[str, Any]:
    smoke_warmup = args.warmup_steps if args.warmup_steps is not None else 10
    step_warmup = args.warmup_steps if args.warmup_steps is not None else 10
    return {
        "smoke": {
            "enabled": args.phase in ("smoke", "all"),
            "repeats": args.repeats if args.repeats is not None else 1,
            "warmup_steps": smoke_warmup,
            "measured_steps": args.measured_steps if args.measured_steps is not None else 100,
        },
        "step": {
            "enabled": args.phase in ("step", "all"),
            "repeats": args.repeats if args.repeats is not None else DEFAULT_PERFORMANCE_REPEATS,
            "warmup_steps": step_warmup,
            "measured_steps": args.measured_steps if args.measured_steps is not None else 500,
            "aggregate_repeats": [1, 2, 3, 4],
        },
        "convergence": {
            "enabled": args.phase in ("convergence", "all"),
            "seeds": _parse_seeds(args.seeds, args.seed),
            "train_steps": args.train_steps,
            "eval_interval": args.eval_interval,
        },
    }


def _runtime_files(rgbnomore_root: Path) -> tuple[list[Path], list[Path]]:
    local = [
        HERE / name
        for name in (
            "run.py",
            "pipeline.py",
            "validate.py",
            "model_factory.py",
            "optimizer.py",
            "augmentation.py",
            "sample_order.py",
            "pls_experiment.py",
            "manifest_preflight.py",
            "direct_dct_reader.py",
            "metrics.py",
            "artifacts.py",
            "schema.py",
        )
        if (HERE / name).is_file()
    ]
    local.extend(
        path
        for path in (
            FASTLANES_ROOT / "galp/benchmarks/system_dct_major/training_pls/schedule.py",
            FASTLANES_ROOT / "galp/benchmarks/system_dct_major/training_pls/matrix.py",
            FASTLANES_ROOT / "galp/benchmarks/system_rgbnomore/shared/common.py",
            FASTLANES_ROOT / "galp/benchmarks/system_rgbnomore/shared/manifest_contract.py",
            FASTLANES_ROOT / "galp/torch/direct_dct_torch.cpp",
            FASTLANES_ROOT / "galp/torch/direct_dct.py",
            FASTLANES_ROOT / "galp/torch/diagnostics.py",
            FASTLANES_ROOT / "galp/profiles/_base.py",
            FASTLANES_ROOT / "galp/profiles/rgbnomore.py",
            FASTLANES_ROOT / "galp/include/galp/direct_dct.hpp",
            FASTLANES_ROOT / "galp/include/galp/profiles/direct_dct.hpp",
            FASTLANES_ROOT / "galp/include/galp/profiles/registry.hpp",
            FASTLANES_ROOT / "galp/include/galp/profiles/rgbnomore.hpp",
            FASTLANES_ROOT / "galp/include/galp/jpeg_dct.hpp",
            FASTLANES_ROOT / "galp/src/api/direct_dct.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_planner.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_shard_reader.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_shard_writer.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_metadata.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_decode.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_device_bridge.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_device.cu",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_gather_kernels.cu",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_transform_kernels.cu",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_plan_types.hpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_policy.hpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_policy.cpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_device_runtime.hpp",
            FASTLANES_ROOT / "galp/src/jpeg/jpeg_dct_cuda_internal.cuh",
        )
        if path.is_file()
    )
    external = [
        rgbnomore_root / name
        for name in (
            "models/plainvit.py",
            "datasets.py",
            "utils/dct_ops.py",
            "utils/custom_transforms.py",
            "utils/custom_optims.py",
            "dct_manip/dct_manip.cpp",
        )
        if (rgbnomore_root / name).is_file()
    ]
    return local, external


def _build_contract(
    args: argparse.Namespace,
    train_meta: dict[str, Any],
    val_meta: dict[str, Any],
    separation: dict[str, Any],
) -> dict[str, Any]:
    rgbnomore_root = args.rgbnomore_root.resolve()
    local_files, external_files = _runtime_files(rgbnomore_root)
    phases = _phase_config(args)
    optimizer = resolved_optimizer_config(args)
    scheduler = resolved_scheduler_config(args)
    scheduler["total_steps_by_phase"] = {
        "smoke": phases["smoke"]["warmup_steps"] + phases["smoke"]["measured_steps"],
        "step": phases["step"]["warmup_steps"] + phases["step"]["measured_steps"],
        "convergence": phases["convergence"]["train_steps"],
    }
    galp_manifest_sha256 = None
    galp_payload_cache = None
    galp_payload_fingerprints: list[dict[str, Any]] = []
    galp_native_binary = None
    galp_manifest_preflight = None
    galp_validation_manifest_sha256 = None
    galp_validation_payload_cache = None
    galp_validation_payload_fingerprints: list[dict[str, Any]] = []
    galp_validation_manifest_preflight = None
    if "galp" in args.enabled:
        preflight: ManifestPreflight = args.galp_manifest_preflight
        galp_manifest = args.galp_manifest.resolve()
        galp_manifest_sha256 = sha256_file(galp_manifest)
        galp_manifest_preflight = preflight.as_dict()
        cache_path = galp_manifest.with_name(
            galp_manifest.name + ".payload_fingerprints.json"
        )
        galp_payload_fingerprints = cached_file_fingerprints(
            preflight.fingerprint_inputs(),
            cache_path,
            cache_format="galp_shard_payload_fingerprints_v1",
            allow_hash_misses=args.refresh_galp_payload_fingerprints,
        )
        galp_payload_cache = {
            "path": str(cache_path.resolve()),
            "sha256": sha256_file(cache_path),
        }
        validation_preflight: ManifestPreflight = (
            args.galp_validation_manifest_preflight
        )
        validation_manifest = args.galp_validation_manifest.resolve()
        if validation_manifest == galp_manifest:
            galp_validation_manifest_sha256 = galp_manifest_sha256
            galp_validation_manifest_preflight = galp_manifest_preflight
            galp_validation_payload_cache = galp_payload_cache
            galp_validation_payload_fingerprints = galp_payload_fingerprints
        else:
            galp_validation_manifest_sha256 = sha256_file(validation_manifest)
            galp_validation_manifest_preflight = validation_preflight.as_dict()
            validation_cache_path = validation_manifest.with_name(
                validation_manifest.name + ".payload_fingerprints.json"
            )
            galp_validation_payload_fingerprints = cached_file_fingerprints(
                validation_preflight.fingerprint_inputs(),
                validation_cache_path,
                cache_format="galp_shard_payload_fingerprints_v1",
                allow_hash_misses=args.refresh_galp_payload_fingerprints,
            )
            galp_validation_payload_cache = {
                "path": str(validation_cache_path.resolve()),
                "sha256": sha256_file(validation_cache_path),
            }
        binding_candidates = sorted(args.galp_torch_module_path.glob("_galp_direct_dct*.so"))
        if len(binding_candidates) != 1:
            raise FileNotFoundError(
                f"expected one _galp_direct_dct shared library in {args.galp_torch_module_path}, "
                f"got {binding_candidates}"
            )
        binding = binding_candidates[0].resolve()
        galp_native_binary = {
            "path": str(binding),
            "sha256": sha256_file(binding),
            "size_bytes": binding.stat().st_size,
        }
    checkpoint_fingerprints = {
        name: (
            None
            if path is None
            else {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
                "size_bytes": path.resolve().stat().st_size,
            }
        )
        for name, path in (
            ("rgb", args.rgb_init_checkpoint),
            ("dct", args.dct_init_checkpoint),
            ("resume", args.resume_checkpoint),
        )
    }
    pls_contract = {
        "enabled": bool(args.pls_experiment),
        "condition": args.pls_resolved_condition,
        "terminology": "physical load segment (PLS)",
        "organization": args.pls_organization,
        "organization_seed": args.pls_organization_seed,
        "crop_policy": args.pls_crop_policy,
        "order_policy": args.pls_order_policy,
        "segment_images": args.pls_segment_images,
        "segments_per_pool": args.pls_segments_per_pool,
        "pool_size_definition": (
            "sum of actual sample counts in every complete PLS loaded into the closed wave; "
            "equals G*M only when all selected PLSs contain G samples"
        ),
        "closed_wave_lifetime": (
            "materialize complete PLSs, uniformly permute all resident samples, consume the "
            "pool completely, then release it and advance"
        ),
        "crop_scope": (
            "one crop configuration per source physical shard per epoch"
            if args.pls_crop_policy == "per-shard"
            else "one crop configuration per logical sample per epoch"
        ),
        "implementation_scope": {
            "statistical_schedule_emulation": bool(args.pls_experiment),
            "complete_gpu_pool_materialization_configured": bool(args.pls_gpu_pool),
            "complete_gpu_pool_materialization_measured": False,
        },
        "interpretation_policy": {
            "primary": "complete convergence curves and final top-1/top-5",
            "mixing": "explanatory observation with no pass/fail threshold",
            "runtime": "engineering context with no training-strategy pass/fail threshold",
        },
    }
    contract: dict[str, Any] = {
        "schema_version": TRAINING_CONTRACT_SCHEMA,
        "enabled_pipelines": args.enabled,
        "required_comparison_groups": args.required_groups,
        "comparison_groups": {
            name: {"pipelines": list(pipelines), "domain": name, "cross_domain_equivalence": False}
            for name, pipelines in COMPARISON_GROUPS.items()
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "domains": {domain: model_configuration(domain) for domain in ("rgb", "dct")},
            "execution": "pytorch-eager",
            "device": args.device,
            "precision": "fp32",
            "distributed": False,
            "gradient_accumulation": 1,
        },
        "initialization": {
            "mode": args.init_mode,
            "default_convergence_mode": "random",
            "rgb_checkpoint": None if args.rgb_init_checkpoint is None else str(args.rgb_init_checkpoint.resolve()),
            "dct_checkpoint": None if args.dct_init_checkpoint is None else str(args.dct_init_checkpoint.resolve()),
            "resume_checkpoint": None if args.resume_checkpoint is None else str(args.resume_checkpoint.resolve()),
            "checkpoint_fingerprints": checkpoint_fingerprints,
            "strict_state_dict_load": True,
        },
        "optimizer": optimizer,
        "scheduler": scheduler,
        "augmentation": {
            **augmentation_contract(),
            "physical_load_segment_crop": {
                "enabled": bool(args.pls_experiment),
                "policy": args.pls_crop_policy,
                "shared_key": (
                    "sha256(seed, epoch, source_physical_shard_id)"
                    if args.pls_crop_policy == "per-shard"
                    else "sha256(seed, epoch, logical_sample_id)"
                ),
                "horizontal_flip": "independently keyed per sample",
            },
        },
        "validation_augmentation": {
            "recipe": "deterministic-centered-square-resize-range-v1",
            "horizontal_flip": False,
            "dataset_semantics": val_meta.get("validation_semantics"),
            "independent_imagenet_validation": (
                val_meta.get("validation_semantics")
                == "official-imagenet-validation"
            ),
        },
        "sample_order": {
            "algorithm": (
                (
                    "galp-pls-closed-wave-v1"
                    if args.pls_order_policy == "pls-wave"
                    else "galp-pls-global-permutation-v1"
                )
                if args.pls_experiment
                else "sha256-derived-python-random-full-permutation-v1"
            ),
            "identity": ["epoch", "position", "logical_sample_id"],
            "seed": args.seed,
            "drop_last": args.drop_last,
            "prefetch_depth_batches": args.prefetch_depth,
            "validation_basis": "optimizer_consumed_ids",
            "repeat_cursor_reset": True,
            "distributed_rank": args.distributed_rank,
            "distributed_world_size": args.distributed_world_size,
            "distributed_partition": "global permutation followed by rank-strided logical-ID partition",
            "physical_load_segment": pls_contract,
        },
        "datasets": {"train": train_meta, "validation": val_meta, "separation": separation},
        "execution": {
            "mode": args.execution_mode,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "measurement_policy": {
                "audit": "per-stage synchronization and exhaustive per-step numerical checks",
                "runtime": "first-step probe outside timing; one synchronization at each measured-region boundary; deferred statistics",
                "selected": args.execution_mode,
            },
            "worker_semantics": {
                "galp": "native rowgroup-prefetch workers within one ordered asynchronous batch producer",
                "rgbnomore": "PyTorch DataLoader processes",
                "dali": "DALI pipeline threads",
                "pytorch": "PyTorch DataLoader processes",
            },
            "phases": phases,
            "global_step_definition": "one completed optimizer.step",
            "optimizer_step_definition": "zero_grad + forward + cross_entropy + backward + optional clip + optimizer.step + scheduler.step",
            "warmup_updates_model": True,
            "stage_timing_note": "stage timings may overlap and must not be summed to derive end-to-end step latency",
        },
        "pipelines": {
            "rgbnomore_root": str(rgbnomore_root),
            "galp_manifest": None if args.galp_manifest is None else str(args.galp_manifest.resolve()),
            "galp_manifest_sha256": galp_manifest_sha256,
            "galp_manifest_preflight": galp_manifest_preflight,
            "galp_manifest_expectations": {
                "version": args.expected_manifest_version,
                "physical_layout": args.expected_physical_layout,
                "spatial_order": args.expected_spatial_order,
                "image_count": args.expected_image_count,
            },
            "galp_payload_fingerprint_cache": galp_payload_cache,
            "galp_payload_fingerprints": galp_payload_fingerprints,
            "galp_validation_manifest": (
                None
                if args.galp_validation_manifest is None
                else str(args.galp_validation_manifest.resolve())
            ),
            "galp_validation_manifest_sha256": galp_validation_manifest_sha256,
            "galp_validation_manifest_preflight": galp_validation_manifest_preflight,
            "galp_validation_manifest_expectations": {
                "version": args.expected_manifest_version,
                "physical_layout": args.expected_physical_layout,
                "spatial_order": args.expected_spatial_order,
                "image_count": (
                    args.expected_validation_image_count
                    if args.expected_validation_image_count is not None
                    else (
                        args.expected_image_count
                        if args.galp_validation_manifest is not None
                        and args.galp_manifest is not None
                        and args.galp_validation_manifest.resolve()
                        == args.galp_manifest.resolve()
                        else None
                    )
                ),
            },
            "galp_validation_payload_fingerprint_cache": galp_validation_payload_cache,
            "galp_validation_payload_fingerprints": galp_validation_payload_fingerprints,
            "galp_native_binary": galp_native_binary,
            "galp_torch_module_path": str(args.galp_torch_module_path.resolve()),
            "runtime_profile": GALP_RUNTIME_PROFILE,
        },
        "gates": {
            "throughput_cv_limit": args.throughput_cv_limit,
            "semantic": {
                "dct_input_atol": args.dct_semantic_atol,
                "dct_rtol": args.dct_semantic_rtol,
                "rgb_input_warning_atol": args.rgb_semantic_warning_atol,
                "rgb_input_failure_atol": args.rgb_semantic_failure_atol,
                "gradient_cosine_dct": args.semantic_gradient_cosine_dct,
                "gradient_cosine_rgb": args.semantic_gradient_cosine_rgb,
            },
        },
        "provenance": {
            "fastlanes": repository_provenance(FASTLANES_ROOT, local_files),
            "rgbnomore": repository_provenance(rgbnomore_root, external_files),
            "actual_imported_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in local_files + external_files
            ],
            "runtime": runtime_metadata(),
        },
        "claims": {
            "cross_dct_rgb_tensor_equivalence": False,
            "cross_dct_rgb_weight_equivalence": False,
            "short_convergence_is_final_accuracy": False,
            "pls_training_effect_measured_when_enabled": bool(args.pls_experiment),
            "complete_gpu_pool_materialization": False,
            "complete_gpu_pool_materialization_configured": bool(args.pls_gpu_pool),
            "mixing_threshold_required": False,
            "runtime_improvement_required": False,
        },
    }
    contract["contract_sha256"] = sha256_json(contract)
    return contract


def _checkpoint_for_domain(args: argparse.Namespace, domain: str) -> Path | None:
    if args.resume_checkpoint is not None:
        return args.resume_checkpoint
    return args.rgb_init_checkpoint if domain == "rgb" else args.dct_init_checkpoint


def _domain_seed(base: int, domain: str) -> int:
    return base ^ (0x524742 if domain == "rgb" else 0x444354)


def _augmentation_state(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "stateless_keyed": True,
        "physical_load_segment": copy.deepcopy(
            contract["sample_order"].get("physical_load_segment", {"enabled": False})
        ),
    }


def _prepare_initial_state(
    args: argparse.Namespace,
    contract: dict[str, Any],
    domain: str,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(args.device)
    seed_everything(_domain_seed(seed, domain))
    model = build_model(args.rgbnomore_root, domain, device)
    provenance = initialize_model(
        model,
        init_mode=args.init_mode,
        checkpoint=_checkpoint_for_domain(args, domain),
        domain=domain,
    )
    full_checkpoint = provenance.pop("_full_checkpoint_payload", None)
    default_total_steps = max(
        contract["execution"]["phases"]["convergence"]["train_steps"],
        contract["execution"]["phases"]["step"]["warmup_steps"]
        + contract["execution"]["phases"]["step"]["measured_steps"],
    )
    if full_checkpoint is not None:
        expected_optimizer = {
            key: value
            for key, value in contract["optimizer"].items()
            if key != "parameter_groups_by_domain"
        }
        expected_scheduler = {
            key: value
            for key, value in contract["scheduler"].items()
            if key != "total_steps_by_phase"
        }
        if full_checkpoint["optimizer_configuration"] != expected_optimizer:
            raise ValueError("full checkpoint optimizer configuration does not match the run contract")
        if full_checkpoint["scheduler_configuration"] != expected_scheduler:
            raise ValueError("full checkpoint scheduler configuration does not match the run contract")
        total_steps = int(full_checkpoint["scheduler_total_steps"])
        if total_steps <= 0:
            raise ValueError("full checkpoint scheduler_total_steps must be positive")
    else:
        total_steps = default_total_steps
    optimizer, parameter_groups = build_optimizer(model, contract["optimizer"])
    scheduler = build_scheduler(optimizer, contract["scheduler"], total_steps=total_steps)
    cursor = {"epoch": 0, "position": -1, "logical_sample_id": None}
    global_step = 0
    checkpoint_epoch = 0
    if full_checkpoint is not None:
        optimizer.load_state_dict(full_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(full_checkpoint["scheduler_state_dict"])
        restore_rng_state(full_checkpoint["rng_state"])
        cursor = dict(full_checkpoint["sample_order_cursor"])
        global_step = int(full_checkpoint["global_step"])
        checkpoint_epoch = int(full_checkpoint["epoch"])
    initial = capture_training_state(model, optimizer, scheduler)
    initial["sample_order_cursor"] = cursor
    initial["global_step"] = global_step
    initial["checkpoint_epoch"] = checkpoint_epoch
    initial["scheduler_total_steps"] = total_steps
    initial["full_checkpoint_restored"] = full_checkpoint is not None
    expected_augmentation_state = _augmentation_state(contract)
    if (
        full_checkpoint is not None
        and args.pls_experiment
        and full_checkpoint["augmentation_state"] != expected_augmentation_state
    ):
        raise ValueError(
            "full checkpoint PLS augmentation/schedule state does not match the run contract"
        )
    initial["augmentation_state"] = (
        copy.deepcopy(full_checkpoint["augmentation_state"])
        if full_checkpoint is not None
        else expected_augmentation_state
    )
    artifact_path = output_dir / f"initial_state_{domain}_seed{seed}.pt"
    torch.save(initial, artifact_path)
    record = {
        **provenance,
        "seed": seed,
        "domain_seed": _domain_seed(seed, domain),
        "model_state_sha256": tensor_state_sha256(initial["model"]),
        "optimizer_state_sha256": nested_state_sha256(initial["optimizer"]),
        "scheduler_state_sha256": nested_state_sha256(initial["scheduler"]),
        "rng_state": rng_state_artifact(initial["rng"]),
        "parameter_groups": parameter_groups,
        "global_step": global_step,
        "sample_order_cursor": cursor,
        "scheduler_total_steps": total_steps,
        "artifact": {"path": str(artifact_path.resolve()), "sha256": sha256_file(artifact_path)},
    }
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return initial, record


def _collect_batches(
    samples: Sequence[TrainingSample],
    *,
    seed: int,
    batch_size: int,
    batch_count: int,
    drop_last: bool,
    start_cursor: dict[str, Any] | None = None,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> tuple[list[list[SampleIdentity]], dict[int, int]]:
    if not samples:
        raise ValueError("training dataset is empty")
    rank_sample_count = len(
        canonical_epoch_order(
            [sample.logical_sample_id for sample in samples],
            seed,
            0,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
    )
    if rank_sample_count == 0:
        raise ValueError(
            f"distributed rank {distributed_rank} receives no samples from a "
            f"{len(samples)}-sample dataset at world size {distributed_world_size}"
        )
    if drop_last and rank_sample_count < batch_size:
        raise ValueError(
            f"drop_last=True cannot produce a batch: rank {distributed_rank} has "
            f"{rank_sample_count} samples, "
            f"batch_size is {batch_size}"
        )
    cursor = start_cursor or {"epoch": 0, "position": -1}
    start_epoch = int(cursor.get("epoch", 0))
    last_position = int(cursor.get("position", -1))
    if start_epoch < 0 or last_position < -1:
        raise ValueError(f"invalid sample-order cursor: {cursor}")
    if last_position >= 0:
        epoch_order = canonical_epoch_order(
            [sample.logical_sample_id for sample in samples],
            seed,
            start_epoch,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
        if last_position >= len(epoch_order):
            raise ValueError(f"sample-order cursor position is outside the epoch: {cursor}")
        cursor_sample = cursor.get("logical_sample_id")
        if cursor_sample is not None and epoch_order[last_position].logical_sample_id != cursor_sample:
            raise ValueError("sample-order cursor identity does not match the canonical permutation")
        end_of_partial_batch = not drop_last and last_position == len(epoch_order) - 1
        if (last_position + 1) % batch_size != 0 and not end_of_partial_batch:
            raise ValueError("sample-order cursor is not on a completed batch boundary")
    stream = batch_stream(
        [sample.logical_sample_id for sample in samples],
        seed,
        batch_size,
        drop_last=drop_last,
        start_epoch=start_epoch,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )
    batches: list[list[SampleIdentity]] = []
    dropped: dict[int, int] = {}
    current_epoch = start_epoch
    while len(batches) < batch_count:
        batch, count = next(stream)
        if batch and batch[0].epoch == start_epoch and batch[-1].position <= last_position:
            continue
        if batch and batch[0].epoch == start_epoch and batch[0].position <= last_position:
            raise ValueError(
                "sample-order cursor is not on a completed batch boundary; exact resume is impossible"
            )
        if count:
            dropped[current_epoch] = dropped.get(current_epoch, 0) + count
            current_epoch += 1
            continue
        if batch:
            current_epoch = batch[0].epoch
            batches.append(batch)
    return batches, dropped


def _augmentation_batches(
    batches: Sequence[Sequence[SampleIdentity]],
    samples: dict[str, TrainingSample],
    *,
    seed: int,
    domain: str,
) -> list[list[AugmentationDecision]]:
    return [
        [
            derive_augmentation(
                seed=seed,
                epoch=identity.epoch,
                logical_sample_id=identity.logical_sample_id,
                source_width=samples[identity.logical_sample_id].width,
                source_height=samples[identity.logical_sample_id].height,
                domain=domain,
            )
            for identity in batch
        ]
        for batch in batches
    ]


def _collect_training_batches(
    args: argparse.Namespace,
    samples: Sequence[TrainingSample],
    *,
    seed: int,
    batch_count: int,
    start_cursor: dict[str, Any] | None,
) -> tuple[
    list[list[SampleIdentity]],
    dict[int, int],
    PlsExecutionPlan | None,
]:
    """Select either the canonical control or the PLS experimental schedule."""

    if not args.pls_experiment:
        batches, dropped = _collect_batches(
            samples,
            seed=seed,
            batch_size=args.batch_size,
            batch_count=batch_count,
            drop_last=args.drop_last,
            start_cursor=start_cursor,
            distributed_rank=args.distributed_rank,
            distributed_world_size=args.distributed_world_size,
        )
        return batches, dropped, None
    plan = build_pls_execution_plan(
        samples,
        seed=seed,
        batch_count=batch_count,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        start_cursor=start_cursor,
        organization=args.pls_organization,
        crop_policy=args.pls_crop_policy,
        order_policy=args.pls_order_policy,
        segment_images=args.pls_segment_images,
        segments_per_pool=args.pls_segments_per_pool,
        distributed_rank=args.distributed_rank,
        distributed_world_size=args.distributed_world_size,
        complete_final_pool=args.pls_gpu_pool,
        organization_seed=args.pls_organization_seed,
    )
    return plan.batches, plan.dropped_per_epoch, plan


def _training_augmentation_batches(
    plan: PlsExecutionPlan | None,
    batches: Sequence[Sequence[SampleIdentity]],
    samples: dict[str, TrainingSample],
    *,
    seed: int,
    domain: str,
) -> list[list[AugmentationDecision]]:
    if plan is not None:
        return pls_augmentation_batches(plan, samples, seed=seed, domain=domain)
    return _augmentation_batches(batches, samples, seed=seed, domain=domain)


def _flatten(values: Sequence[Sequence[Any]]) -> list[Any]:
    return [item for batch in values for item in batch]


def _adapter_pipeline_config(
    contract: dict[str, Any],
    args: argparse.Namespace,
    *,
    split: str,
    pls_plan: PlsExecutionPlan | None = None,
    enable_pls_gpu_pool: bool = True,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown adapter dataset split {split!r}")
    config = {
        **contract["pipelines"],
        "prefetch_depth": args.prefetch_depth,
        "execution_mode": args.execution_mode,
    }
    if split == "validation":
        config["galp_manifest"] = contract["pipelines"].get(
            "galp_validation_manifest", contract["pipelines"].get("galp_manifest")
        )
    elif args.pls_gpu_pool and enable_pls_gpu_pool:
        if pls_plan is None:
            raise ValueError("PLS GPU pool execution requires an exact schedule plan")
        if (
            args.pls_organization in ("storage-hash", "storage-stratified")
            and not pls_plan.storage_order_matches_galp_image_ids()
        ):
            raise ValueError(
                f"{args.pls_organization} GPU execution requires a physically rewritten "
                "GALP manifest and a training manifest whose galp_image_id sequence "
                "matches the pre-registered storage order"
            )
        config["pls_gpu_pool"] = {
            "enabled": True,
            "closed_pool_batches": pls_plan.selected_pool_batches(),
            "lifetime": (
                "materialize one complete pool, consume every optimizer batch, "
                "release it, then load the next pool"
            ),
        }
    elif args.pls_gpu_pool:
        config["pls_gpu_pool"] = {
            "enabled": False,
            "reason": "semantic first-step probe uses the ordinary mini-batch path",
        }
    return config


def _validate_galp_dataset_binding(
    samples: Sequence[TrainingSample],
    metadata: dict[str, Any],
    manifest: Path,
    preflight: ManifestPreflight,
    *,
    split: str,
    allow_layout_manifest_rebinding: bool = False,
) -> None:
    declared = metadata.get("declared_galp_manifest")
    if (
        declared is not None
        and Path(str(declared)).resolve() != manifest.resolve()
        and not allow_layout_manifest_rebinding
    ):
        raise ValueError(
            f"{split} training JSON was generated for GALP manifest {declared}, "
            f"but the runner received {manifest.resolve()}"
        )
    image_ids = [sample.galp_image_id for sample in samples]
    if any(image_id is None for image_id in image_ids):
        raise ValueError(f"{split} GALP samples must all declare galp_image_id")
    concrete_ids = [int(image_id) for image_id in image_ids if image_id is not None]
    if len(set(concrete_ids)) != len(concrete_ids):
        raise ValueError(f"{split} GALP samples contain duplicate galp_image_id values")
    outside = [
        image_id
        for image_id in concrete_ids
        if image_id < 0 or image_id >= preflight.image_count
    ]
    if outside:
        raise ValueError(
            f"{split} GALP image IDs fall outside manifest population "
            f"[0,{preflight.image_count}): {outside[:8]}"
        )


def _reproducibility_hashes(
    batches: Sequence[Sequence[SampleIdentity]],
    decisions: Sequence[Sequence[AugmentationDecision]],
) -> dict[str, str]:
    return {
        "shuffle_order_sha256": sha256_json(
            [identity.as_dict() for identity in _flatten(batches)]
        ),
        "transform_descriptor_sha256": sha256_json(
            [decision.as_dict() for decision in _flatten(decisions)]
        ),
    }


class _SyncLedger:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def record(self, reason: str) -> None:
        self.reasons.append(reason)

    def as_dict(self) -> dict[str, Any]:
        counts = {reason: self.reasons.count(reason) for reason in sorted(set(self.reasons))}
        return {"count": len(self.reasons), "reasons": counts}


def _sync(
    device: torch.device,
    ledger: _SyncLedger | None = None,
    reason: str = "unspecified",
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        if ledger is not None:
            ledger.record(reason)


def _move_batch(
    batch: TrainingBatch,
    device: torch.device,
    *,
    synchronize: bool = True,
    sync_ledger: _SyncLedger | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, float]:
    if batch.on_device:
        return batch.inputs, batch.labels, 0.0
    begin = time.perf_counter()
    inputs = tuple(value.to(device, non_blocking=device.type == "cuda") for value in batch.inputs)
    labels = batch.labels.to(device, non_blocking=device.type == "cuda")
    if synchronize:
        _sync(device, sync_ledger, "host_to_device_completion")
    return inputs, labels, time.perf_counter() - begin


def _validate_batch(
    batch: TrainingBatch,
    inputs: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    expected: Sequence[SampleIdentity],
    domain: str,
    *,
    validate_label_values: bool = True,
) -> list[str]:
    failures: list[str] = []
    if batch.identities != list(expected):
        failures.append("emitted identities differ from requested batch")
    if labels.ndim != 1 or labels.shape[0] != len(expected):
        failures.append(f"label shape is {tuple(labels.shape)}, expected [{len(expected)}]")
    if (
        validate_label_values
        and labels.numel()
        and (int(labels.min()) < 0 or int(labels.max()) >= 1000)
    ):
        failures.append("label outside ImageNet-1K range [0,1000)")
    expected_input_count = 1 if domain == "rgb" else 2
    if len(inputs) != expected_input_count:
        failures.append(f"{domain} model expected {expected_input_count} input tensors, got {len(inputs)}")
    for value in inputs:
        if value.shape[0] != len(expected):
            failures.append("input batch dimension differs from identity count")
        if value.dtype != torch.float32:
            failures.append(f"input dtype must be float32, got {value.dtype}")
    if domain == "rgb" and inputs and tuple(inputs[0].shape[1:]) != (3, 224, 224):
        failures.append(f"RGB input shape must be [B,3,224,224], got {tuple(inputs[0].shape)}")
    if domain == "dct" and len(inputs) == 2:
        if tuple(inputs[0].shape[1:]) != (1, 28, 28, 8, 8):
            failures.append(f"DCT Y input shape mismatch: {tuple(inputs[0].shape)}")
        if tuple(inputs[1].shape[1:]) != (2, 14, 14, 8, 8):
            failures.append(f"DCT CbCr input shape mismatch: {tuple(inputs[1].shape)}")
    return failures


def _train_one_step(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    adapter: TrainingPipelineAdapter,
    expected: Sequence[SampleIdentity],
    domain: str,
    device: torch.device,
    label_smoothing: float,
    gradient_clipping: float | None,
    collect_numerics: bool,
    sync_ledger: _SyncLedger | None = None,
) -> tuple[dict[str, Any], TrainingBatch, tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    _sync(device, sync_ledger, "audit_step_start")
    end_to_end_begin = time.perf_counter()
    batch = adapter.next_batch()
    inputs, labels, h2d = _move_batch(batch, device, sync_ledger=sync_ledger)
    batch_failures = _validate_batch(batch, inputs, labels, expected, domain)

    stage: dict[str, float] = dict(batch.stage_seconds)
    begin = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    stage["gradient_processing"] = time.perf_counter() - begin

    begin = time.perf_counter()
    logits = model(*inputs)
    _sync(device, sync_ledger, "forward_completion")
    stage["forward"] = time.perf_counter() - begin
    if logits.ndim != 2 or logits.shape != (len(expected), 1000):
        batch_failures.append(f"model logits shape mismatch: {tuple(logits.shape)}")

    begin = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    _sync(device, sync_ledger, "loss_completion")
    stage["loss"] = time.perf_counter() - begin

    before = None
    if collect_numerics:
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    begin = time.perf_counter()
    loss.backward()
    _sync(device, sync_ledger, "backward_completion")
    stage["backward"] = time.perf_counter() - begin
    gradients = gradient_summary(model, include_per_parameter=collect_numerics)

    begin = time.perf_counter()
    clipped_norm = None
    if gradient_clipping is not None:
        clipped_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping).item())
    _sync(device, sync_ledger, "gradient_processing_completion")
    stage["gradient_processing"] += time.perf_counter() - begin

    begin = time.perf_counter()
    optimizer.step()
    _sync(device, sync_ledger, "optimizer_completion")
    stage["optimizer"] = time.perf_counter() - begin
    update = parameter_update_summary(before, model) if before is not None else None

    begin = time.perf_counter()
    scheduler.step()
    _sync(device, sync_ledger, "scheduler_completion")
    stage["scheduler"] = time.perf_counter() - begin
    end_to_end = time.perf_counter() - end_to_end_begin
    stage["h2d"] = h2d
    record = {
        "loss": float(loss.detach().item()),
        "loss_finite": tensor_is_finite(loss.detach()),
        "gradients": gradients,
        "parameter_update": update,
        "clipped_gradient_norm": clipped_norm,
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "end_to_end_seconds": end_to_end,
        "stage_seconds": stage,
        "batch_failures": batch_failures,
        "optimizer_step_executed": True,
    }
    return record, batch, inputs, labels, logits


def _train_one_step_runtime(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    adapter: TrainingPipelineAdapter,
    expected: Sequence[SampleIdentity],
    domain: str,
    device: torch.device,
    label_smoothing: float,
    gradient_clipping: float | None,
) -> tuple[dict[str, Any], TrainingBatch]:
    """Submit one real train step without measured-path host materialization."""

    host_step_begin = time.perf_counter()
    batch = adapter.next_batch()
    cuda_events: dict[str, tuple[Any, Any]] = {}

    def event_begin(name: str) -> float:
        if device.type == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            cuda_events[name] = (begin, end)
        return time.perf_counter()

    def event_end(name: str, begin: float, stage: dict[str, float]) -> None:
        if device.type == "cuda":
            cuda_events[name][1].record()
        else:
            stage[name] = time.perf_counter() - begin

    stage = dict(batch.stage_seconds)
    begin = event_begin("h2d")
    inputs, labels, _h2d_host_submit = _move_batch(
        batch, device, synchronize=False
    )
    event_end("h2d", begin, stage)
    batch_failures = _validate_batch(
        batch,
        inputs,
        labels,
        expected,
        domain,
        validate_label_values=False,
    )

    optimizer.zero_grad(set_to_none=True)
    begin = event_begin("forward")
    logits = model(*inputs)
    event_end("forward", begin, stage)
    if logits.ndim != 2 or logits.shape != (len(expected), 1000):
        batch_failures.append(f"model logits shape mismatch: {tuple(logits.shape)}")

    begin = event_begin("loss")
    loss = torch.nn.functional.cross_entropy(
        logits, labels, label_smoothing=label_smoothing
    )
    event_end("loss", begin, stage)

    begin = event_begin("backward")
    loss.backward()
    event_end("backward", begin, stage)

    begin = event_begin("gradient_processing")
    if gradient_clipping is not None:
        # Deliberately retain the device scalar; converting it with .item()
        # would synchronize every measured step.
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
    event_end("gradient_processing", begin, stage)

    begin = event_begin("optimizer")
    optimizer.step()
    event_end("optimizer", begin, stage)
    scheduler.step()
    return (
        {
            "loss_tensor": loss.detach(),
            "stage_seconds": stage,
            "cuda_events": cuda_events,
            "host_submit_seconds": time.perf_counter() - host_step_begin,
            "batch_failures": batch_failures,
            "optimizer_step_executed": True,
            "deep_parameter_scans": 0,
            "host_scalar_materializations_in_step": 0,
        },
        batch,
    )


def _resolve_runtime_record(record: dict[str, Any]) -> None:
    for name, (begin, end) in record.pop("cuda_events").items():
        record["stage_seconds"][name] = float(begin.elapsed_time(end)) / 1000.0


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "finite": tensor_is_finite(cpu),
        "min": float(cpu.min().item()) if cpu.numel() else None,
        "max": float(cpu.max().item()) if cpu.numel() else None,
        "mean": float(cpu.float().mean().item()) if cpu.numel() else None,
        "sha256": tensor_state_sha256({"tensor": cpu}),
    }


def _first_step_probe_policy(execution_mode: str) -> str:
    if execution_mode not in ("audit", "runtime"):
        raise ValueError(f"invalid execution mode {execution_mode!r}")
    return "audit"


def _first_step_probe(
    args: argparse.Namespace,
    contract: dict[str, Any],
    pipeline: str,
    train_samples: Sequence[TrainingSample],
    initial: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    domain = DOMAINS[pipeline]
    device = torch.device(args.device)
    model = build_model(args.rgbnomore_root, domain, device)
    optimizer, _groups = build_optimizer(model, contract["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        contract["scheduler"],
        total_steps=(
            int(initial["scheduler_total_steps"])
            if initial.get("full_checkpoint_restored")
            else max(1, contract["execution"]["phases"]["convergence"]["train_steps"])
        ),
    )
    reset_training_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, initial=initial)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batches, _, pls_plan = _collect_training_batches(
        args,
        train_samples,
        seed=args.seed,
        batch_count=1 + args.prefetch_depth,
        start_cursor=initial["sample_order_cursor"],
    )
    decisions = _training_augmentation_batches(
        pls_plan,
        batches,
        {sample.logical_sample_id: sample for sample in train_samples},
        seed=args.seed,
        domain=domain,
    )
    reproducibility = _reproducibility_hashes(batches[:1], decisions[:1])
    adapter = build_training_adapter(
        pipeline,
        train_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        config=_adapter_pipeline_config(
            contract,
            args,
            split="train",
            pls_plan=pls_plan,
            enable_pls_gpu_pool=False,
        ),
    )
    adapter.begin(_flatten(batches), _flatten(decisions), [len(batch) for batch in batches])
    before_model_hash = tensor_state_sha256(model.state_dict())
    before_optimizer_hash = nested_state_sha256(optimizer.state_dict())
    before_scheduler_hash = nested_state_sha256(scheduler.state_dict())
    record, batch, inputs, labels, logits = _train_one_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        adapter=adapter,
        expected=batches[0],
        domain=domain,
        device=device,
        label_smoothing=contract["optimizer"]["label_smoothing"],
        gradient_clipping=contract["optimizer"]["gradient_clipping_norm"],
        collect_numerics=True,
    )
    read = adapter.prefetched_read_identities()
    adapter.end()
    gradient_tensors = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    update_tensors = {
        name: model.state_dict()[name].detach().cpu() - initial["model"][name].detach().cpu()
        for name in model.state_dict()
        if torch.is_tensor(model.state_dict()[name])
    }
    artifact = {
        "fresh_clone": True,
        "before_warmup": True,
        "formal_repeat_polluted": False,
        "probe_policy": _first_step_probe_policy(args.execution_mode),
        "execution_mode_independent": True,
        "initial_model_state_sha256": before_model_hash,
        "initial_optimizer_state_sha256": before_optimizer_hash,
        "initial_scheduler_state_sha256": before_scheduler_hash,
        "model_configuration": model_configuration(domain),
        "optimizer_configuration_sha256": sha256_json(contract["optimizer"]),
        "sample_ids": [identity.as_dict() for identity in batches[0]],
        "reproducibility": {
            "seed": args.seed,
            "distributed_rank": args.distributed_rank,
            "distributed_world_size": args.distributed_world_size,
            **reproducibility,
        },
        "prefetched_read_ids": [identity.as_dict() for identity in read],
        "augmentation_decisions": batch.augmentations,
        "physical_load_segment": (
            None if pls_plan is None else pls_plan.selected_summary()
        ),
        "labels": labels.detach().cpu().tolist(),
        "inputs": [_tensor_record(value) for value in inputs],
        "initial_logits": _tensor_record(logits),
        "first_step_loss": record["loss"],
        "gradient_summary": record["gradients"],
        "first_optimizer_update_summary": record["parameter_update"],
        "failures": list(record["batch_failures"]),
    }
    if not record["loss_finite"]:
        artifact["failures"].append("first-step loss is not finite")
    if not record["gradients"]["finite"]:
        artifact["failures"].append("first-step gradients are not finite")
    if not record["gradients"]["nonzero"]:
        artifact["failures"].append("first-step gradients are all zero")
    if not record["parameter_update"] or not record["parameter_update"]["changed"]:
        artifact["failures"].append("first optimizer step did not update a trainable parameter")
    raw = {
        "inputs": [value.detach().cpu() for value in inputs],
        "labels": labels.detach().cpu(),
        "logits": logits.detach().cpu(),
        "loss": record["loss"],
        "gradients": gradient_tensors,
        "updates": update_tensors,
        "sample_ids": batches[0],
        "augmentation_decisions": batch.augmentations,
        "initial_hash": before_model_hash,
        "initial_optimizer_hash": before_optimizer_hash,
        "initial_scheduler_hash": before_scheduler_hash,
        "model_configuration": model_configuration(domain),
        "optimizer_configuration_sha256": sha256_json(contract["optimizer"]),
    }
    adapter.close()
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return artifact, raw


def _run_repeat(
    args: argparse.Namespace,
    contract: dict[str, Any],
    pipeline: str,
    train_samples: Sequence[TrainingSample],
    initial: dict[str, Any],
    *,
    repeat: int,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    domain = DOMAINS[pipeline]
    device = torch.device(args.device)
    model = build_model(args.rgbnomore_root, domain, device)
    optimizer, _groups = build_optimizer(model, contract["optimizer"])
    scheduler_total_steps = (
        int(initial["scheduler_total_steps"])
        if initial.get("full_checkpoint_restored")
        else warmup_steps + measured_steps
    )
    scheduler = build_scheduler(
        optimizer, contract["scheduler"], total_steps=scheduler_total_steps
    )
    reset_training_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, initial=initial)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reset_rng = rng_state_artifact(capture_rng_state())
    initial_rng = rng_state_artifact(initial["rng"])
    reset_evidence = {
        "model_state_sha256": tensor_state_sha256(model.state_dict()),
        "optimizer_state_sha256": nested_state_sha256(optimizer.state_dict()),
        "scheduler_state_sha256": nested_state_sha256(scheduler.state_dict()),
        "matches_initial_model": tensor_state_sha256(model.state_dict()) == tensor_state_sha256(initial["model"]),
        "matches_initial_optimizer": nested_state_sha256(optimizer.state_dict()) == nested_state_sha256(initial["optimizer"]),
        "matches_initial_scheduler": nested_state_sha256(scheduler.state_dict()) == nested_state_sha256(initial["scheduler"]),
        "rng_state": reset_rng,
        "matches_initial_rng": reset_rng["sha256"] == initial_rng["sha256"],
        "sample_order_cursor": copy.deepcopy(initial["sample_order_cursor"]),
        "augmentation_state": copy.deepcopy(initial["augmentation_state"]),
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }
    total_consumed_batches = warmup_steps + measured_steps
    all_batches, dropped, pls_plan = _collect_training_batches(
        args,
        train_samples,
        seed=args.seed,
        batch_count=total_consumed_batches + args.prefetch_depth,
        start_cursor=initial["sample_order_cursor"],
    )
    decisions = _training_augmentation_batches(
        pls_plan,
        all_batches,
        {sample.logical_sample_id: sample for sample in train_samples},
        seed=args.seed,
        domain=domain,
    )
    reproducibility = _reproducibility_hashes(
        all_batches[:total_consumed_batches], decisions[:total_consumed_batches]
    )
    adapter = build_training_adapter(
        pipeline,
        train_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        config=_adapter_pipeline_config(
            contract, args, split="train", pls_plan=pls_plan
        ),
    )
    adapter.begin(_flatten(all_batches), _flatten(decisions), [len(batch) for batch in all_batches])
    ledger = SampleOrderLedger(pipeline)
    ledger.record_requested(_flatten(all_batches))
    for epoch, count in dropped.items():
        ledger.record_dropped(epoch, count)
    failures: list[str] = []
    warmup_losses: list[float] = []
    warmup_native_stats: list[dict[str, Any]] = []
    measured_native_stats: list[dict[str, Any]] = []
    warmup_native_counters: list[dict[str, int | float]] = []
    measured_native_counters: list[dict[str, int | float]] = []
    measurement_syncs = _SyncLedger()
    loader_before_measurement: dict[str, Any] = {}
    loader_after_measurement: dict[str, Any] = {}

    def record_batch(
        batch: TrainingBatch,
        *,
        phase: str,
        runtime_record: dict[str, Any] | None = None,
    ) -> None:
        ledger.record_emitted(batch.identities)
        ledger.record_consumed(batch.identities)
        if args.execution_mode == "runtime":
            adapter.snapshot_batch_metrics(batch)
        else:
            adapter.finalize_batch_metrics(batch)
        if runtime_record is not None:
            runtime_record["stage_seconds"].update(batch.stage_seconds)
        stats = dict(batch.native_execution_stats)
        counters = dict(batch.native_counters)
        if phase == "warmup":
            warmup_native_stats.append(stats)
            warmup_native_counters.append(counters)
        elif phase == "measured":
            measured_native_stats.append(stats)
            measured_native_counters.append(counters)
        else:
            raise ValueError(f"unknown training phase {phase!r}")

    step_records: list[dict[str, Any]] = []
    if args.execution_mode == "runtime":
        warmup_records: list[dict[str, Any]] = []
        for step in range(warmup_steps):
            record, batch = _train_one_step_runtime(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=adapter,
                expected=all_batches[step],
                domain=domain,
                device=device,
                label_smoothing=contract["optimizer"]["label_smoothing"],
                gradient_clipping=contract["optimizer"]["gradient_clipping_norm"],
            )
            failures.extend(
                f"warmup step {step}: {value}" for value in record["batch_failures"]
            )
            record_batch(batch, phase="warmup", runtime_record=record)
            warmup_records.append(record)
            del batch

        loader_before_measurement = adapter.loader_metrics()
        _sync(device, measurement_syncs, "measured_region_start")
        measured_begin = time.perf_counter()
        for measured in range(measured_steps):
            index = warmup_steps + measured
            record, batch = _train_one_step_runtime(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=adapter,
                expected=all_batches[index],
                domain=domain,
                device=device,
                label_smoothing=contract["optimizer"]["label_smoothing"],
                gradient_clipping=contract["optimizer"]["gradient_clipping_norm"],
            )
            failures.extend(
                f"measured step {measured}: {value}"
                for value in record["batch_failures"]
            )
            record_batch(batch, phase="measured", runtime_record=record)
            step_records.append(record)
            del batch
        _sync(device, measurement_syncs, "measured_region_end")
        measured_elapsed = time.perf_counter() - measured_begin
        loader_after_measurement = adapter.loader_metrics()

        all_runtime_records = warmup_records + step_records
        loss_values = (
            torch.stack([record["loss_tensor"] for record in all_runtime_records])
            .detach()
            .cpu()
            .tolist()
        )
        for index, (record, loss_value) in enumerate(
            zip(all_runtime_records, loss_values)
        ):
            record["loss"] = float(loss_value)
            del record["loss_tensor"]
            _resolve_runtime_record(record)
            if not math.isfinite(record["loss"]):
                region = "warmup" if index < warmup_steps else "measured"
                failures.append(f"{region} step {index}: non-finite loss")
        warmup_losses = [record["loss"] for record in warmup_records]
    else:
        for step in range(warmup_steps):
            record, batch, _inputs, _labels, _logits = _train_one_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=adapter,
                expected=all_batches[step],
                domain=domain,
                device=device,
                label_smoothing=contract["optimizer"]["label_smoothing"],
                gradient_clipping=contract["optimizer"]["gradient_clipping_norm"],
                collect_numerics=step == 0,
            )
            record_batch(batch, phase="warmup")
            warmup_losses.append(record["loss"])
            failures.extend(
                f"warmup step {step}: {value}" for value in record["batch_failures"]
            )
            if not record["loss_finite"]:
                failures.append(f"warmup step {step}: non-finite loss")
            if not record["gradients"]["finite"]:
                failures.append(f"warmup step {step}: non-finite gradient")
            if not record["gradients"]["nonzero"]:
                failures.append(f"warmup step {step}: zero gradients")
            del batch, _inputs, _labels, _logits

        loader_before_measurement = adapter.loader_metrics()
        for measured in range(measured_steps):
            index = warmup_steps + measured
            record, batch, _inputs, _labels, _logits = _train_one_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=adapter,
                expected=all_batches[index],
                domain=domain,
                device=device,
                label_smoothing=contract["optimizer"]["label_smoothing"],
                gradient_clipping=contract["optimizer"]["gradient_clipping_norm"],
                collect_numerics=measured == 0,
                sync_ledger=measurement_syncs,
            )
            record_batch(batch, phase="measured")
            failures.extend(
                f"measured step {measured}: {value}"
                for value in record["batch_failures"]
            )
            if not record["loss_finite"]:
                failures.append(f"measured step {measured}: non-finite loss")
            if not record["gradients"]["finite"]:
                failures.append(f"measured step {measured}: non-finite gradient")
            if not record["gradients"]["nonzero"]:
                failures.append(f"measured step {measured}: zero gradients")
            if (
                record["parameter_update"] is not None
                and not record["parameter_update"]["changed"]
            ):
                failures.append(
                    f"measured step {measured}: optimizer did not update parameters"
                )
            step_records.append(record)
            del batch, _inputs, _labels, _logits
        measured_elapsed = sum(
            record["end_to_end_seconds"] for record in step_records
        )
        loader_after_measurement = adapter.loader_metrics()

    native_execution_stats = NativeExecutionStatsAccumulator()
    warmup_native_execution_stats = NativeExecutionStatsAccumulator()
    measured_native_execution_stats = NativeExecutionStatsAccumulator()
    warmup_native_counter_totals: dict[str, float] = {}
    measured_native_counter_totals: dict[str, float] = {}
    for stats, counters in zip(warmup_native_stats, warmup_native_counters):
        native_execution_stats.observe(stats)
        warmup_native_execution_stats.observe(stats)
        merge_native_counter_snapshot(warmup_native_counter_totals, counters)
    for stats, counters in zip(measured_native_stats, measured_native_counters):
        native_execution_stats.observe(stats)
        measured_native_execution_stats.observe(stats)
        merge_native_counter_snapshot(measured_native_counter_totals, counters)
    native_counter_totals = measured_native_counter_totals
    allocation_stability = native_allocation_stability(
        warmup_native_stats,
        measured_native_stats,
    )

    ledger.record_prefetched(adapter.prefetched_read_identities())
    adapter.end()
    loader_metrics = adapter.loader_metrics()
    loader_measured_metrics: dict[str, Any] = {}
    for name in (
        "submitted_batches",
        "consumed_batches",
        "queue_hit_batches",
        "queue_miss_batches",
        "producer_submit_seconds",
        "producer_active_seconds",
        "producer_planning_seconds",
        "producer_io_staging_seconds",
        "producer_ordered_submission_seconds",
        "consumer_wait_seconds",
    ):
        if name in loader_after_measurement:
            loader_measured_metrics[name] = (
                float(loader_after_measurement[name])
                - float(loader_before_measurement.get(name, 0))
            )
    expected_consumed = _flatten(all_batches[:total_consumed_batches])
    order_validation = ledger.validate(expected_consumed)
    failures.extend(order_validation["failures"])
    model_finite = nested_tensors_finite(model.state_dict())
    optimizer_finite = nested_tensors_finite(optimizer.state_dict())
    if not model_finite:
        failures.append("model parameters contain NaN or Inf")
    if not optimizer_finite:
        failures.append("optimizer state contains NaN or Inf")
    expected_scheduler_last_epoch = (
        int(reset_evidence["scheduler_last_epoch"]) + warmup_steps + measured_steps
    )
    scheduler_progress_correct = int(scheduler.last_epoch) == expected_scheduler_last_epoch
    if not scheduler_progress_correct:
        failures.append(
            f"scheduler last_epoch is {scheduler.last_epoch}, expected {expected_scheduler_last_epoch}"
        )
    elapsed = measured_elapsed
    if args.execution_mode == "runtime":
        latencies = [elapsed / measured_steps * 1000.0] * measured_steps
        step_latency_method = "measured_region_wall_clock_average_no_per_step_sync"
    else:
        latencies = [record["end_to_end_seconds"] * 1000.0 for record in step_records]
        step_latency_method = "per_step_synchronized_wall_clock"
    stage_names = sorted({name for record in step_records for name in record["stage_seconds"]})
    stage_metrics = {
        name: distribution([record["stage_seconds"].get(name, 0.0) * 1000.0 for record in step_records])
        for name in stage_names
    }
    processed = sum(len(batch) for batch in all_batches[warmup_steps:total_consumed_batches])
    compute_stage_names = (
        "forward",
        "loss",
        "backward",
        "gradient_processing",
        "optimizer",
    )
    compute_seconds = sum(
        float(record["stage_seconds"].get(name, 0.0))
        for record in step_records
        for name in compute_stage_names
    )
    compute_upper_bound = processed / compute_seconds if compute_seconds else None
    measured_wait = float(loader_measured_metrics.get("consumer_wait_seconds", 0.0))
    measured_queue_consumed = int(loader_measured_metrics.get("consumed_batches", 0))
    measured_queue_hits = int(loader_measured_metrics.get("queue_hit_batches", 0))
    native_sync_reasons = {
        name: int(native_counter_totals.get(name, 0))
        for name in (
            "cached_gather_sync_count",
            "decoded_batch_sync_count",
        )
        if int(native_counter_totals.get(name, 0))
    }
    accounted_native_syncs = int(native_counter_totals.get("internal_sync_count", 0))
    memory = process_memory()
    sample_order_artifact = ledger.as_dict(expected_consumed)
    sample_order_artifact.update(
        {
            "seed": args.seed,
            "distributed_rank": args.distributed_rank,
            "distributed_world_size": args.distributed_world_size,
            **reproducibility,
        }
    )
    repeat_result = {
        "repeat": repeat,
        "execution_mode": args.execution_mode,
        "reset": reset_evidence,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "optimizer_steps": warmup_steps + measured_steps,
        "warmup_losses": warmup_losses,
        "measured_losses": [record["loss"] for record in step_records],
        "processed_images": processed,
        "throughput_images_per_s": processed / elapsed if elapsed else None,
        "compute_only_diagnostic": {
            "method": "sum of CUDA-event train-stage durations; loader and H2D excluded",
            "not_a_pipeline_result": True,
            "elapsed_seconds": compute_seconds,
            "upper_bound_images_per_s": compute_upper_bound,
            "achieved_fraction": (
                (processed / elapsed) / compute_upper_bound
                if elapsed and compute_upper_bound
                else None
            ),
        },
        "measured_region_wall_clock_seconds": elapsed,
        "step_latency_ms": distribution(latencies),
        "step_latency_method": step_latency_method,
        "stage_latency_ms": stage_metrics,
        "stage_timing_note": contract["execution"]["stage_timing_note"],
        "sample_order": sample_order_artifact,
        "physical_load_segment": (
            None if pls_plan is None else pls_plan.selected_summary()
        ),
        "prefetch_overrun": order_validation["prefetch_overrun"],
        "loader_metrics": loader_metrics,
        "loader_measured_metrics": {
            **loader_measured_metrics,
            "queue_hit_rate": (
                measured_queue_hits / measured_queue_consumed
                if measured_queue_consumed
                else None
            ),
            "consumer_wait_fraction_of_measured_wall": (
                measured_wait / elapsed if elapsed else None
            ),
        },
        "synchronization_accounting": {
            "explicit_host_device": measurement_syncs.as_dict(),
            "galp_native_internal_count": accounted_native_syncs,
            "galp_native_internal_reasons": native_sync_reasons,
            "scope": "measured train region for explicit syncs and native counters",
        },
        "measurement_instrumentation": {
            "deep_gradient_scans_in_measured_path": (
                0 if args.execution_mode == "runtime" else measured_steps
            ),
            "full_parameter_snapshots_in_measured_path": (
                0 if args.execution_mode == "runtime" else min(1, measured_steps)
            ),
            "host_scalar_materializations_in_runtime_step": (
                0 if args.execution_mode == "runtime" else None
            ),
            "deferred_loss_materialization_batches": (
                measured_steps if args.execution_mode == "runtime" else 0
            ),
            "first_step_probe_in_timing": False,
        },
        "host_memory": memory,
        "host_memory_scope": (
            "process_lifetime_rss_peak_plus_end_of_repeat_virtual_memory_fd_thread_and_mapping_snapshot"
        ),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "native_allocator_metrics": {
            name: int(value) if value.is_integer() else value
            for name, value in sorted(native_counter_totals.items())
        },
        "native_allocator_scope": (
            "measured batches: per-batch counters are summed; galp_native process-global snapshots use the maximum"
            if pipeline == "galp"
            else "not_applicable"
        ),
        "native_warmup_allocator_metrics": {
            name: int(value) if value.is_integer() else value
            for name, value in sorted(warmup_native_counter_totals.items())
        },
        "native_allocation_stability": allocation_stability if pipeline == "galp" else None,
        "native_execution_stats": (
            native_execution_stats.as_dict() if pipeline == "galp" else None
        ),
        "native_execution_stats_by_phase": (
            {
                "warmup": warmup_native_execution_stats.as_dict(),
                "measured": measured_native_execution_stats.as_dict(),
            }
            if pipeline == "galp"
            else None
        ),
        "common_statistics": {
            "samples": processed,
            "batches": measured_steps,
            "elapsed_seconds": elapsed,
            "samples_per_second": processed / elapsed if elapsed else None,
            "reader_wait_seconds": measured_wait,
            "training_step_seconds": compute_seconds,
            "peak_host_queue_depth": loader_metrics.get("max_queue_depth_batches"),
            "peak_device_memory": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
            "loss_summary": distribution([record["loss"] for record in step_records]),
        },
        "model_parameters_finite": model_finite,
        "optimizer_state_finite": optimizer_finite,
        "scheduler_last_epoch": scheduler.last_epoch,
        "scheduler_expected_last_epoch": expected_scheduler_last_epoch,
        "scheduler_progress_correct": scheduler_progress_correct,
        "failures": failures,
        "ok": not failures,
    }
    adapter.close()
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return repeat_result


def _validation_decision(sample: TrainingSample, domain: str) -> AugmentationDecision:
    side = min(sample.width, sample.height)
    if domain == "dct":
        side = max(16, (side // 16) * 16)
    return AugmentationDecision(
        seed=0,
        epoch=0,
        logical_sample_id=sample.logical_sample_id,
        source_width=sample.width,
        source_height=sample.height,
        crop_x=((sample.width - side) // 2 // (16 if domain == "dct" else 1))
        * (16 if domain == "dct" else 1),
        crop_y=((sample.height - side) // 2 // (16 if domain == "dct" else 1))
        * (16 if domain == "dct" else 1),
        crop_width=side,
        crop_height=side,
        resize_width=224,
        resize_height=224,
        horizontal_flip=False,
        interpolation="bilinear",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        domain=domain,
        dct_crop_alignment_pixels=16 if domain == "dct" else None,
    )


def _evaluate_validation(
    args: argparse.Namespace,
    contract: dict[str, Any],
    pipeline: str,
    model: torch.nn.Module,
    val_samples: Sequence[TrainingSample],
) -> dict[str, Any]:
    domain = DOMAINS[pipeline]
    device = torch.device(args.device)
    identities = [SampleIdentity(0, index, sample.logical_sample_id) for index, sample in enumerate(val_samples)]
    decisions = [_validation_decision(sample, domain) for sample in val_samples]
    batch_lengths = [
        min(args.batch_size, len(val_samples) - begin) for begin in range(0, len(val_samples), args.batch_size)
    ]
    adapter = build_training_adapter(
        pipeline,
        val_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        config=_adapter_pipeline_config(contract, args, split="validation"),
    )
    adapter.begin(identities, decisions, batch_lengths)
    model.eval()
    total = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    failures: list[str] = []
    begin = time.perf_counter()
    with torch.no_grad():
        cursor = 0
        for length in batch_lengths:
            batch = adapter.next_batch()
            inputs, labels, _h2d = _move_batch(batch, device)
            failures.extend(_validate_batch(batch, inputs, labels, identities[cursor : cursor + length], domain))
            logits = model(*inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not tensor_is_finite(logits) or not tensor_is_finite(loss):
                failures.append("validation logits or loss contain NaN/Inf")
            accuracy = topk_accuracy(logits, labels)
            loss_sum += float(loss.item()) * length
            top1_sum += accuracy["top1"] * length
            top5_sum += accuracy["top5"] * length
            total += length
            cursor += length
    _sync(device)
    seconds = time.perf_counter() - begin
    adapter.close()
    model.train()
    return {
        "samples": total,
        "loss": loss_sum / total,
        "top1": top1_sum / total,
        "top5": top5_sum / total,
        "wall_clock_seconds": seconds,
        "failures": failures,
    }


def _save_training_checkpoint(
    path: Path,
    *,
    domain: str,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    epoch: int,
    sample_order_cursor: dict[str, Any],
    optimizer_configuration: dict[str, Any],
    scheduler_configuration: dict[str, Any],
    scheduler_total_steps: int,
    augmentation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optimizer_recipe = {
        key: value
        for key, value in optimizer_configuration.items()
        if key != "parameter_groups_by_domain"
    }
    scheduler_recipe = {
        key: value
        for key, value in scheduler_configuration.items()
        if key != "total_steps_by_phase"
    }
    payload = {
        "format": "galp-rgbnomore-full-training-checkpoint-v1",
        "model_architecture": MODEL_ARCHITECTURE,
        "model_domain": domain,
        "model_configuration": model_configuration(domain),
        "optimizer_configuration": optimizer_recipe,
        "scheduler_configuration": scheduler_recipe,
        "scheduler_total_steps": scheduler_total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None,
        "global_step": global_step,
        "epoch": epoch,
        "rng_state": capture_training_state(model, optimizer, scheduler)["rng"],
        "augmentation_state": copy.deepcopy(
            augmentation_state
            if augmentation_state is not None
            else {"stateless_keyed": True}
        ),
        "sample_order_cursor": sample_order_cursor,
    }
    torch.save(payload, path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _run_convergence_seed(
    args: argparse.Namespace,
    contract: dict[str, Any],
    pipeline: str,
    train_samples: Sequence[TrainingSample],
    val_samples: Sequence[TrainingSample],
    initial: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    domain = DOMAINS[pipeline]
    device = torch.device(args.device)
    model = build_model(args.rgbnomore_root, domain, device)
    optimizer, _groups = build_optimizer(model, contract["optimizer"])
    scheduler_total_steps = (
        int(initial["scheduler_total_steps"])
        if initial.get("full_checkpoint_restored")
        else args.train_steps
    )
    scheduler = build_scheduler(
        optimizer, contract["scheduler"], total_steps=scheduler_total_steps
    )
    reset_training_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, initial=initial)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    starting_scheduler_last_epoch = int(scheduler.last_epoch)
    batches, dropped, pls_plan = _collect_training_batches(
        args,
        train_samples,
        seed=seed,
        batch_count=args.train_steps + args.prefetch_depth,
        start_cursor=initial["sample_order_cursor"],
    )
    decisions = _training_augmentation_batches(
        pls_plan,
        batches,
        {sample.logical_sample_id: sample for sample in train_samples},
        seed=seed,
        domain=domain,
    )
    reproducibility = _reproducibility_hashes(
        batches[: args.train_steps], decisions[: args.train_steps]
    )
    if args.pls_gpu_pool and (
        pls_plan is None or not pls_plan.is_pool_batch_boundary(args.train_steps)
    ):
        raise ValueError(
            "--train-steps must end at an exact closed-pool boundary when "
            "--pls-gpu-pool is enabled"
        )
    if args.pls_gpu_pool and pls_plan is not None:
        nonboundary_evaluations = [
            step
            for step in range(args.eval_interval, args.train_steps, args.eval_interval)
            if not pls_plan.is_pool_batch_boundary(step)
        ]
        if nonboundary_evaluations:
            raise ValueError(
                "--eval-interval must place every GPU-pool checkpoint at a closed-pool "
                f"boundary; first invalid step is {nonboundary_evaluations[0]}"
            )
    adapter = build_training_adapter(
        pipeline,
        train_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        config=_adapter_pipeline_config(
            contract, args, split="train", pls_plan=pls_plan
        ),
    )
    adapter.begin(_flatten(batches), _flatten(decisions), [len(batch) for batch in batches])
    ledger = SampleOrderLedger(pipeline)
    ledger.record_requested(_flatten(batches))
    for epoch, count in dropped.items():
        ledger.record_dropped(epoch, count)
    curve: list[dict[str, Any]] = []
    failures: list[str] = []
    best_top1 = -math.inf
    best_checkpoint = None
    validation_events: list[dict[str, Any]] = []
    starting_global_step = int(initial.get("global_step", 0))
    start = time.perf_counter()
    for step in range(args.train_steps):
        record, batch, _inputs, _labels, _logits = _train_one_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            adapter=adapter,
            expected=batches[step],
            domain=domain,
            device=device,
            label_smoothing=contract["optimizer"]["label_smoothing"],
            gradient_clipping=contract["optimizer"]["gradient_clipping_norm"],
            collect_numerics=step == 0,
        )
        ledger.record_emitted(batch.identities)
        ledger.record_consumed(batch.identities)
        failures.extend(f"step {step}: {value}" for value in record["batch_failures"])
        if not record["loss_finite"] or not record["gradients"]["finite"] or not record["gradients"]["nonzero"]:
            failures.append(f"step {step}: numerical correctness gate failed")
        curve.append(
            {
                "optimizer_step": step + 1,
                "global_step": starting_global_step + step + 1,
                "processed_images": sum(len(value) for value in batches[: step + 1]),
                "train_loss": record["loss"],
                "wall_clock_seconds": time.perf_counter() - start,
            }
        )
        del batch, _inputs, _labels, _logits
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.train_steps:
            evaluation = _evaluate_validation(args, contract, pipeline, model, val_samples)
            validation_wall_clock_seconds = float(evaluation["wall_clock_seconds"])
            evaluation.update(
                optimizer_step=step + 1,
                processed_images=curve[-1]["processed_images"],
                wall_clock_seconds=time.perf_counter() - start,
                validation_wall_clock_seconds=validation_wall_clock_seconds,
            )
            validation_events.append(evaluation)
            failures.extend(f"validation step {step + 1}: {value}" for value in evaluation["failures"])
            if evaluation["top1"] > best_top1:
                best_top1 = evaluation["top1"]
                best_path = output_dir / f"checkpoint_{pipeline}_seed{seed}_best.pt"
                best_checkpoint = _save_training_checkpoint(
                    best_path,
                    domain=domain,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=starting_global_step + step + 1,
                    epoch=batches[step][0].epoch,
                    sample_order_cursor=batches[step][-1].as_dict(),
                    optimizer_configuration=contract["optimizer"],
                    scheduler_configuration=contract["scheduler"],
                    scheduler_total_steps=scheduler_total_steps,
                    augmentation_state=_augmentation_state(contract),
                )
    ledger.record_prefetched(adapter.prefetched_read_identities())
    loader_metrics = adapter.loader_metrics()
    adapter.close()
    expected = _flatten(batches[: args.train_steps])
    order_validation = ledger.validate(expected)
    failures.extend(order_validation["failures"])
    expected_scheduler_last_epoch = starting_scheduler_last_epoch + args.train_steps
    scheduler_progress_correct = int(scheduler.last_epoch) == expected_scheduler_last_epoch
    if not scheduler_progress_correct:
        failures.append(
            f"scheduler last_epoch is {scheduler.last_epoch}, expected {expected_scheduler_last_epoch}"
        )
    model_finite = nested_tensors_finite(model.state_dict())
    optimizer_finite = nested_tensors_finite(optimizer.state_dict())
    if not model_finite:
        failures.append("final model parameters contain NaN or Inf")
    if not optimizer_finite:
        failures.append("final optimizer state contains NaN or Inf")
    final_path = output_dir / f"checkpoint_{pipeline}_seed{seed}_final.pt"
    final_checkpoint = _save_training_checkpoint(
        final_path,
        domain=domain,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=starting_global_step + args.train_steps,
        epoch=batches[args.train_steps - 1][0].epoch,
        sample_order_cursor=batches[args.train_steps - 1][-1].as_dict(),
        optimizer_configuration=contract["optimizer"],
        scheduler_configuration=contract["scheduler"],
        scheduler_total_steps=scheduler_total_steps,
        augmentation_state=_augmentation_state(contract),
    )
    thresholds = (0.10, 0.20, 0.30, 0.40, 0.50)
    time_to_accuracy = {
        str(threshold): next(
            (event["wall_clock_seconds"] for event in validation_events if event["top1"] >= threshold), None
        )
        for threshold in thresholds
    }
    sample_order_artifact = ledger.as_dict(expected)
    sample_order_artifact.update(
        {
            "seed": seed,
            "distributed_rank": args.distributed_rank,
            "distributed_world_size": args.distributed_world_size,
            **reproducibility,
        }
    )
    pool_metrics = loader_metrics.get("physical_load_segment_gpu_pool", {})
    materialized_pools = int(pool_metrics.get("materialized_pool_count", 0))
    fully_emitted_pools = int(
        pool_metrics.get("fully_emitted_release_eligible_pool_count", 0)
    )
    gpu_pool_evidence = {
        "configured": bool(args.pls_gpu_pool),
        "observed": bool(
            args.pls_gpu_pool
            and materialized_pools > 0
            and fully_emitted_pools == materialized_pools
        ),
        "native_request_granularity": "one complete closed pool",
        "metrics": pool_metrics,
    }
    device_memory = {
        "device": str(device),
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        device_memory.update(
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    result = {
        "seed": seed,
        "classification": "from_scratch_short_convergence" if args.init_mode == "random" else ("fine_tuning" if args.init_mode == "weights" else "resumed_training"),
        "starting_global_step": starting_global_step,
        "ending_global_step": starting_global_step + args.train_steps,
        "optimizer_steps": args.train_steps,
        "processed_images": sum(len(batch) for batch in batches[: args.train_steps]),
        "wall_clock_seconds": time.perf_counter() - start,
        "train_curve": curve,
        "validation": validation_events,
        "time_to_accuracy": time_to_accuracy,
        "sample_order": sample_order_artifact,
        "physical_load_segment": (
            None if pls_plan is None else pls_plan.selected_summary()
        ),
        "gpu_pool_materialization": gpu_pool_evidence,
        "loader_metrics": loader_metrics,
        "device_memory": device_memory,
        "model_parameters_finite": model_finite,
        "optimizer_state_finite": optimizer_finite,
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "scheduler_expected_last_epoch": expected_scheduler_last_epoch,
        "scheduler_progress_correct": scheduler_progress_correct,
        "final_model_state_sha256": tensor_state_sha256(model.state_dict()),
        "final_optimizer_state_sha256": nested_state_sha256(optimizer.state_dict()),
        "final_checkpoint": final_checkpoint,
        "best_checkpoint": best_checkpoint,
        "failures": failures,
        "ok": not failures,
    }
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate_step_repeats(repeats: Sequence[dict[str, Any]], cv_limit: float) -> dict[str, Any]:
    selected = [repeat for repeat in repeats if repeat["repeat"] in (1, 2, 3, 4)]
    throughputs = [float(repeat["throughput_images_per_s"]) for repeat in selected]
    cv = coefficient_of_variation(throughputs)
    wait_fractions = [
        float(repeat["loader_measured_metrics"]["consumer_wait_fraction_of_measured_wall"])
        for repeat in selected
        if repeat.get("loader_measured_metrics", {}).get(
            "consumer_wait_fraction_of_measured_wall"
        )
        is not None
    ]
    compute_ratios = [
        float(repeat["compute_only_diagnostic"]["achieved_fraction"])
        for repeat in selected
        if repeat.get("compute_only_diagnostic", {}).get("achieved_fraction") is not None
    ]
    hot_mean = statistics.fmean(throughputs) if throughputs else None
    cold = float(repeats[0]["throughput_images_per_s"]) if repeats else None

    def selected_values(factory: Any) -> list[float]:
        values: list[float] = []
        for repeat in selected:
            value = factory(repeat)
            if value is not None:
                values.append(float(value))
        return values

    table_summary = {
        "throughput_images_per_s": distribution(throughputs),
        "median_step_ms": distribution(
            selected_values(
                lambda repeat: repeat.get("step_latency_ms", {}).get("p50")
            )
        ),
        "p95_step_ms": distribution(
            selected_values(
                lambda repeat: repeat.get("step_latency_ms", {}).get("p95")
            )
        ),
        "loader_wait_ms": distribution(
            selected_values(
                lambda repeat: repeat.get("stage_latency_ms", {})
                .get("loader_data_wait", {})
                .get("p50")
            )
        ),
        "forward_ms": distribution(
            selected_values(
                lambda repeat: repeat.get("stage_latency_ms", {})
                .get("forward", {})
                .get("p50")
            )
        ),
        "backward_plus_optimizer_ms": distribution(
            selected_values(
                lambda repeat: (
                    float(
                        repeat.get("stage_latency_ms", {})
                        .get("backward", {})
                        .get("p50", 0.0)
                    )
                    + float(
                        repeat.get("stage_latency_ms", {})
                        .get("optimizer", {})
                        .get("p50", 0.0)
                    )
                )
            )
        ),
        "host_peak_rss_bytes": distribution(
            selected_values(
                lambda repeat: repeat.get("host_memory", {}).get("peak_rss_bytes")
            )
        ),
        "gpu_peak_allocated_bytes": distribution(
            selected_values(lambda repeat: repeat.get("gpu_peak_allocated_bytes"))
        ),
        "gpu_peak_reserved_bytes": distribution(
            selected_values(lambda repeat: repeat.get("gpu_peak_reserved_bytes"))
        ),
        "throughput_cv": cv,
    }
    return {
        "included_repeats": [repeat["repeat"] for repeat in selected],
        "excluded_repeats": [0],
        "throughput_images_per_s": distribution(throughputs),
        "throughput_cv": cv,
        "throughput_cv_limit": cv_limit,
        "performance_status": "passed" if cv is not None and cv <= cv_limit else "failed",
        "cold_repeat_images_per_s": cold,
        "hot_repeat_mean_images_per_s": hot_mean,
        "cold_to_hot_ratio": cold / hot_mean if cold is not None and hot_mean else None,
        "loader_wait_fraction": distribution(wait_fractions),
        "compute_upper_bound_achieved_fraction": distribution(compute_ratios),
        "performance_targets": {
            "historical_baseline_images_per_s": 7.83,
            "ten_x_threshold_images_per_s": 78.3,
            "ten_x_achieved": hot_mean is not None and hot_mean >= 78.3,
            "seventy_percent_compute_upper_achieved": (
                bool(compute_ratios) and statistics.fmean(compute_ratios) >= 0.70
            ),
            "loader_wait_below_twenty_percent": (
                bool(wait_fractions) and statistics.fmean(wait_fractions) < 0.20
            ),
            "throughput_cv_at_most_ten_percent": cv is not None and cv <= 0.10,
        },
        "report_table_summary": table_summary,
    }


def _aggregate_convergence(seeds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    final_top1 = [float(result["validation"][-1]["top1"]) for result in seeds]
    final_top5 = [float(result["validation"][-1]["top5"]) for result in seeds]
    final_loss = [float(result["validation"][-1]["loss"]) for result in seeds]
    wall_clock = [float(result["wall_clock_seconds"]) for result in seeds]
    processed = [float(result["processed_images"]) for result in seeds]
    time_to_accuracy: dict[str, Any] = {}
    for threshold in seeds[0]["time_to_accuracy"]:
        values = [
            float(result["time_to_accuracy"][threshold])
            for result in seeds
            if result["time_to_accuracy"][threshold] is not None
        ]
        time_to_accuracy[threshold] = {
            "reached_seed_count": len(values),
            "mean_seconds": statistics.fmean(values) if values else None,
            "std_seconds": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        }
    return {
        "seed_count": len(seeds),
        "final_validation_top1_mean": statistics.fmean(final_top1),
        "final_validation_top1_std": statistics.stdev(final_top1) if len(final_top1) > 1 else 0.0,
        "final_validation_top5_mean": statistics.fmean(final_top5),
        "final_validation_top5_std": statistics.stdev(final_top5) if len(final_top5) > 1 else 0.0,
        "final_validation_loss_mean": statistics.fmean(final_loss),
        "final_validation_loss_std": statistics.stdev(final_loss) if len(final_loss) > 1 else 0.0,
        "wall_clock_seconds_mean": statistics.fmean(wall_clock),
        "wall_clock_seconds_std": statistics.stdev(wall_clock) if len(wall_clock) > 1 else 0.0,
        "processed_images_mean": statistics.fmean(processed),
        "processed_images_std": statistics.stdev(processed) if len(processed) > 1 else 0.0,
        "time_to_accuracy": time_to_accuracy,
        "claim_boundary": "short convergence only; not final ImageNet accuracy",
    }


def _cosine(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    names = sorted(set(left) & set(right))
    if not names:
        return float("nan")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for name in names:
        a = left[name].double().reshape(-1)
        b = right[name].double().reshape(-1)
        dot += float(torch.dot(a, b).item())
        left_norm += float(torch.dot(a, a).item())
        right_norm += float(torch.dot(b, b).item())
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _tensor_drift(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    left_finite = bool(torch.isfinite(left).all().item())
    right_finite = bool(torch.isfinite(right).all().item())
    delta = (left.double() - right.double()).abs()
    return {
        "shape_match": True,
        "left_finite": left_finite,
        "right_finite": right_finite,
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_reference_abs": max(
            float(left.double().abs().max().item()) if left.numel() else 0.0,
            float(right.double().abs().max().item()) if right.numel() else 0.0,
        ),
        "top1_agreement": float(
            left.argmax(dim=1).eq(right.argmax(dim=1)).float().mean().item()
        )
        if left.ndim == 2
        else None,
    }


def _semantic_compare(group: str, left_name: str, left: dict[str, Any], right_name: str, right: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if left["initial_hash"] != right["initial_hash"]:
        failures.append("same-domain initial model state hashes differ")
    if left["initial_optimizer_hash"] != right["initial_optimizer_hash"]:
        failures.append("same-domain initial optimizer state hashes differ")
    if left["initial_scheduler_hash"] != right["initial_scheduler_hash"]:
        failures.append("same-domain initial scheduler state hashes differ")
    if left["model_configuration"] != right["model_configuration"]:
        failures.append("same-domain model configurations differ")
    if left["optimizer_configuration_sha256"] != right["optimizer_configuration_sha256"]:
        failures.append("same-domain optimizer configurations differ")
    if left["sample_ids"] != right["sample_ids"]:
        failures.append("same-domain first-step sample identities differ")
    if left["augmentation_decisions"] != right["augmentation_decisions"]:
        failures.append("same-domain augmentation decisions differ")
    if not torch.equal(left["labels"], right["labels"]):
        failures.append("same-domain first-step labels differ")
    if len(left["inputs"]) != len(right["inputs"]):
        failures.append("same-domain first-step input tensor counts differ")
    input_drift = [_tensor_drift(a, b) for a, b in zip(left["inputs"], right["inputs"])]
    max_input = max((item.get("max_abs", math.inf) for item in input_drift), default=math.inf)
    if group == "dct":
        for index, drift in enumerate(input_drift):
            limit = gates["dct_input_atol"] + gates["dct_rtol"] * drift.get(
                "max_reference_abs", math.inf
            )
            if (
                not drift.get("shape_match")
                or not drift.get("left_finite")
                or not drift.get("right_finite")
                or not math.isfinite(float(drift.get("max_abs", math.inf)))
                or drift.get("max_abs", math.inf) > limit
            ):
                failures.append(
                    f"DCT first-step input {index} drift {drift.get('max_abs')} exceeds {limit}"
                )
    else:
        if any(
            not drift.get("left_finite") or not drift.get("right_finite")
            for drift in input_drift
        ):
            failures.append("RGB first-step inputs contain NaN or Inf")
        if max_input > gates["rgb_input_failure_atol"]:
            failures.append(f"RGB first-step input drift {max_input} exceeds failure threshold")
        elif max_input > gates["rgb_input_warning_atol"]:
            warnings.append(f"RGB decoder/interpolation input drift {max_input} exceeds warning threshold")
    logits = _tensor_drift(left["logits"], right["logits"])
    loss_abs = abs(float(left["loss"]) - float(right["loss"]))
    if not logits.get("left_finite") or not logits.get("right_finite"):
        failures.append("same-domain initial logits contain NaN or Inf")
    if not math.isfinite(float(left["loss"])) or not math.isfinite(float(right["loss"])):
        failures.append("same-domain first-step loss contains NaN or Inf")
    if group == "dct":
        logit_limit = gates["dct_input_atol"] + gates["dct_rtol"] * max(
            float(left["logits"].abs().max().item()),
            float(right["logits"].abs().max().item()),
        )
        loss_limit = gates["dct_input_atol"] + gates["dct_rtol"] * max(
            abs(float(left["loss"])), abs(float(right["loss"])),
        )
        if logits.get("max_abs", math.inf) > logit_limit:
            failures.append(
                f"DCT initial-logit drift {logits.get('max_abs')} exceeds {logit_limit}"
            )
        if loss_abs > loss_limit:
            failures.append(f"DCT first-step loss drift {loss_abs} exceeds {loss_limit}")
    else:
        if logits.get("max_abs", math.inf) > gates["rgb_input_failure_atol"]:
            failures.append("RGB initial-logit drift exceeds the failure threshold")
        elif logits.get("max_abs", 0.0) > gates["rgb_input_warning_atol"]:
            warnings.append("RGB initial-logit drift exceeds the warning threshold")
        if loss_abs > gates["rgb_input_failure_atol"]:
            failures.append("RGB first-step loss drift exceeds the failure threshold")
        elif loss_abs > gates["rgb_input_warning_atol"]:
            warnings.append("RGB first-step loss drift exceeds the warning threshold")
    gradient_cosine = _cosine(left["gradients"], right["gradients"])
    update_cosine = _cosine(left["updates"], right["updates"])
    if set(left["gradients"]) != set(right["gradients"]):
        failures.append("same-domain gradient parameter sets differ")
    if set(left["updates"]) != set(right["updates"]):
        failures.append("same-domain first-update parameter sets differ")
    cosine_gate = gates["gradient_cosine_dct"] if group == "dct" else gates["gradient_cosine_rgb"]
    if not math.isfinite(gradient_cosine) or gradient_cosine < cosine_gate:
        failures.append(f"gradient cosine {gradient_cosine} is below {cosine_gate}")
    if not math.isfinite(update_cosine) or update_cosine < cosine_gate:
        failures.append(f"first-update cosine {update_cosine} is below {cosine_gate}")
    status = "failed" if failures else ("warning" if warnings else "passed")
    return {
        "group": group,
        "pipelines": [left_name, right_name],
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "model_config_match": left["model_configuration"] == right["model_configuration"],
        "optimizer_config_match": left["optimizer_configuration_sha256"]
        == right["optimizer_configuration_sha256"],
        "initial_state_hash_match": left["initial_hash"] == right["initial_hash"],
        "initial_optimizer_state_hash_match": left["initial_optimizer_hash"]
        == right["initial_optimizer_hash"],
        "initial_scheduler_state_hash_match": left["initial_scheduler_hash"]
        == right["initial_scheduler_hash"],
        "sample_ids_match": left["sample_ids"] == right["sample_ids"],
        "augmentation_decisions_match": left["augmentation_decisions"] == right["augmentation_decisions"],
        "labels_match": torch.equal(left["labels"], right["labels"]),
        "input_drift": input_drift,
        "initial_logits_drift": logits,
        "first_step_loss_abs_diff": loss_abs,
        "gradient_cosine_similarity": gradient_cosine,
        "first_update_cosine_similarity": update_cosine,
        "tensor_equivalent_claim": False if group == "rgb" else status == "passed",
    }


def _overall_status(status: dict[str, str]) -> str:
    values = [value for key, value in status.items() if key != "overall"]
    if "failed" in values:
        return "failed"
    if "warning" in values:
        return "warning"
    if "passed" in values:
        return "passed"
    if values and all(value == "not_run" for value in values):
        return "not_run"
    return "not_applicable"


def _pipeline_progress_path(output_dir: Path, pipeline: str) -> Path:
    return output_dir / f"pipeline_progress_{pipeline}.json"


def _load_pipeline_progress(
    args: argparse.Namespace, contract: dict[str, Any], pipeline: str, output_dir: Path
) -> dict[str, Any]:
    path = _pipeline_progress_path(output_dir, pipeline)
    if args.resume_run is not None and path.is_file():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("pipeline") != pipeline:
            raise ValueError(f"pipeline progress identity mismatch: {path}")
        if progress.get("contract_sha256") != contract["contract_sha256"]:
            raise ValueError(f"pipeline progress contract mismatch: {path}")
        if progress.get("execution_mode", "audit") != contract["execution"].get(
            "mode", "audit"
        ):
            raise ValueError(f"pipeline progress execution-mode mismatch: {path}")
        return progress
    return {
        "schema_version": "galp-rgbnomore-training-progress-v1",
        "pipeline": pipeline,
        "contract_sha256": contract["contract_sha256"],
        "execution_mode": contract["execution"].get("mode", "audit"),
        "status": "in_progress",
        "smoke_repeats": {},
        "step_repeats": {},
        "convergence_seeds": {},
    }


def _write_pipeline_progress(output_dir: Path, pipeline: str, progress: dict[str, Any]) -> None:
    write_json(_pipeline_progress_path(output_dir, pipeline), progress)


def _run_pipeline(
    args: argparse.Namespace,
    contract: dict[str, Any],
    pipeline: str,
    train_samples: Sequence[TrainingSample],
    val_samples: Sequence[TrainingSample],
    initial_states: dict[tuple[str, int], dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    domain = DOMAINS[pipeline]
    artifact: dict[str, Any] = {
        "schema_version": TRAINING_PIPELINE_SCHEMA,
        "pipeline": pipeline,
        "domain": domain,
        "model_architecture": MODEL_ARCHITECTURE,
        "execution_mode": args.execution_mode,
        "measurement_policy": contract["execution"]["measurement_policy"],
        "phase_results": {},
        "sample_order": {},
        "status": empty_status(),
        "failures": [],
        "warnings": [],
    }
    progress = _load_pipeline_progress(args, contract, pipeline, output_dir)
    initial = initial_states[(domain, args.seed)]
    probe, raw_probe = _first_step_probe(args, contract, pipeline, train_samples, initial)
    raw_probe_path = output_dir / f"first_step_probe_{pipeline}.pt"
    torch.save(raw_probe, raw_probe_path)
    artifact["first_step_semantic_probe"] = probe
    artifact["first_step_raw_artifact"] = {
        "path": str(raw_probe_path.resolve()),
        "sha256": sha256_file(raw_probe_path),
    }
    artifact["failures"].extend(probe["failures"])

    phases = contract["execution"]["phases"]
    if phases["smoke"]["enabled"]:
        repeats = []
        for repeat in range(phases["smoke"]["repeats"]):
            result = progress["smoke_repeats"].get(str(repeat))
            if result is None:
                result = _run_repeat(
                    args,
                    contract,
                    pipeline,
                    train_samples,
                    initial,
                    repeat=repeat,
                    warmup_steps=phases["smoke"]["warmup_steps"],
                    measured_steps=phases["smoke"]["measured_steps"],
                )
                progress["smoke_repeats"][str(repeat)] = result
                _write_pipeline_progress(output_dir, pipeline, progress)
            repeats.append(result)
        artifact["phase_results"]["smoke"] = {"repeats": repeats}
        artifact["sample_order"]["smoke"] = [result["sample_order"] for result in repeats]
        failures = [failure for result in repeats for failure in result["failures"]]
        artifact["failures"].extend(failures)
        artifact["status"]["correctness"] = "passed" if not failures else "failed"
    if phases["step"]["enabled"]:
        repeats = []
        for repeat in range(phases["step"]["repeats"]):
            result = progress["step_repeats"].get(str(repeat))
            if result is None:
                result = _run_repeat(
                    args,
                    contract,
                    pipeline,
                    train_samples,
                    initial,
                    repeat=repeat,
                    warmup_steps=phases["step"]["warmup_steps"],
                    measured_steps=phases["step"]["measured_steps"],
                )
                progress["step_repeats"][str(repeat)] = result
                _write_pipeline_progress(output_dir, pipeline, progress)
            repeats.append(result)
        aggregate = _aggregate_step_repeats(repeats, contract["gates"]["throughput_cv_limit"])
        artifact["phase_results"]["step"] = {"repeats": repeats, "aggregate": aggregate}
        artifact["sample_order"]["step"] = [result["sample_order"] for result in repeats]
        failures = [failure for result in repeats for failure in result["failures"]]
        artifact["failures"].extend(failures)
        artifact["status"]["correctness"] = "passed" if not failures else "failed"
        artifact["status"]["performance"] = aggregate["performance_status"]
        if aggregate["performance_status"] == "failed":
            artifact["warnings"].append("throughput CV exceeds the performance gate; correctness is unchanged")
    if phases["convergence"]["enabled"]:
        seed_results = []
        for seed in phases["convergence"]["seeds"]:
            result = progress["convergence_seeds"].get(str(seed))
            if result is None:
                result = _run_convergence_seed(
                    args,
                    contract,
                    pipeline,
                    train_samples,
                    val_samples,
                    initial_states[(domain, seed)],
                    seed=seed,
                    output_dir=output_dir,
                )
                progress["convergence_seeds"][str(seed)] = result
                _write_pipeline_progress(output_dir, pipeline, progress)
            seed_results.append(result)
        artifact["phase_results"]["convergence"] = {
            "seeds": seed_results,
            "aggregate": _aggregate_convergence(seed_results),
        }
        artifact["sample_order"]["convergence"] = [result["sample_order"] for result in seed_results]
        failures = [failure for result in seed_results for failure in result["failures"]]
        artifact["failures"].extend(failures)
        artifact["status"]["convergence"] = "passed" if not failures else "failed"
        if artifact["status"]["correctness"] == "not_applicable":
            artifact["status"]["correctness"] = "passed" if not failures else "failed"
    artifact["status"]["correctness"] = "failed" if artifact["failures"] else "passed"
    artifact["status"]["artifact"] = "passed"
    artifact["status"]["overall"] = _overall_status(artifact["status"])
    progress["status"] = "complete"
    _write_pipeline_progress(output_dir, pipeline, progress)
    return artifact, raw_probe


def _load_resume(args: argparse.Namespace) -> tuple[Path, dict[str, Any], set[str]]:
    output_dir = args.resume_run.resolve()
    contract = json.loads((output_dir / "contract.json").read_text(encoding="utf-8"))
    contract_mode = str(contract.get("execution", {}).get("mode", "audit"))
    if args.execution_mode_explicit and args.execution_mode != contract_mode:
        raise ValueError(
            f"resume execution mode {args.execution_mode!r} does not match immutable contract mode {contract_mode!r}"
        )
    hashes_path = output_dir / "artifact_hashes.json"
    if hashes_path.is_file():
        hash_payload = json.loads(hashes_path.read_text(encoding="utf-8"))
        hash_failures = verify_artifact_hashes(output_dir, hash_payload)
        if hash_failures:
            raise ValueError(f"resume artifact integrity failure: {hash_failures}")
    args.output_dir = output_dir
    args.enabled = list(contract["enabled_pipelines"])
    args.required_groups = list(contract["required_comparison_groups"])
    completed = {
        pipeline for pipeline in args.enabled if (output_dir / f"pipeline_{pipeline}.json").is_file()
    }
    return output_dir, contract, completed


def _path_or_none(value: str | None) -> Path | None:
    return None if value is None else Path(value)


def _hydrate_resume_args(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    """Make every runtime-affecting argument come from the immutable run contract."""

    args.pipeline = list(contract["enabled_pipelines"])
    args.enabled_pipelines = None
    args.required_comparison_groups = list(contract["required_comparison_groups"])
    args.init_mode = contract["initialization"]["mode"]
    args.rgb_init_checkpoint = _path_or_none(contract["initialization"].get("rgb_checkpoint"))
    args.dct_init_checkpoint = _path_or_none(contract["initialization"].get("dct_checkpoint"))
    args.resume_checkpoint = _path_or_none(contract["initialization"].get("resume_checkpoint"))
    args.device = contract["model"]["device"]
    args.seed = int(contract["sample_order"]["seed"])
    args.train_manifest = Path(contract["datasets"]["train"]["path"])
    args.val_manifest = Path(contract["datasets"]["validation"]["path"])
    args.train_root = None
    args.val_root = None
    args.rgbnomore_root = Path(contract["pipelines"]["rgbnomore_root"])
    args.galp_manifest = _path_or_none(contract["pipelines"].get("galp_manifest"))
    args.galp_validation_manifest = _path_or_none(
        contract["pipelines"].get("galp_validation_manifest")
    ) or args.galp_manifest
    args.galp_torch_module_path = Path(contract["pipelines"]["galp_torch_module_path"])
    args.refresh_galp_payload_fingerprints = False
    expectations = contract["pipelines"].get("galp_manifest_expectations", {})
    args.expected_manifest_version = expectations.get("version")
    args.expected_physical_layout = expectations.get("physical_layout")
    args.expected_spatial_order = expectations.get("spatial_order")
    args.expected_image_count = expectations.get("image_count")
    validation_expectations = contract["pipelines"].get(
        "galp_validation_manifest_expectations", {}
    )
    args.expected_validation_image_count = validation_expectations.get("image_count")
    args.batch_size = int(contract["execution"]["batch_size"])
    args.workers = int(contract["execution"]["workers"])
    args.execution_mode = str(contract["execution"].get("mode", "audit"))
    args.drop_last = bool(contract["sample_order"]["drop_last"])
    args.prefetch_depth = int(contract["sample_order"]["prefetch_depth_batches"])
    args.distributed_rank = int(contract["sample_order"].get("distributed_rank", 0))
    args.distributed_world_size = int(
        contract["sample_order"].get("distributed_world_size", 1)
    )
    pls = contract["sample_order"].get("physical_load_segment", {})
    args.pls_experiment = bool(pls.get("enabled", False))
    condition = pls.get("condition")
    args.pls_condition_id = (
        None
        if not condition
        else str(condition.get("requested_condition_id", condition["condition_id"]))
    )
    args.pls_organization = str(pls.get("organization", "current"))
    args.pls_organization_seed = int(pls.get("organization_seed", 20260810))
    args.pls_crop_policy = str(pls.get("crop_policy", "per-shard"))
    args.pls_order_policy = str(pls.get("order_policy", "pls-wave"))
    args.pls_segment_images = int(pls.get("segment_images", 1024))
    args.pls_segments_per_pool = int(pls.get("segments_per_pool", 4))
    args.pls_gpu_pool = bool(
        pls.get("implementation_scope", {}).get(
            "complete_gpu_pool_materialization_configured", False
        )
    )
    enabled_phases = [
        name
        for name in PHASES
        if bool(contract["execution"]["phases"][name].get("enabled", False))
    ]
    args.phase = enabled_phases[0] if len(enabled_phases) == 1 else "all"
    convergence = contract["execution"]["phases"]["convergence"]
    args.train_steps = int(convergence["train_steps"])
    args.eval_interval = int(convergence["eval_interval"])


def _load_audited_initial_states(
    output_dir: Path,
    contract: dict[str, Any],
    required_domains: set[str],
    required_seeds: set[int],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    records = list(contract.get("initial_states", []))
    by_key = {(str(record["domain"]), int(record["seed"])): record for record in records}
    states: dict[tuple[str, int], dict[str, Any]] = {}
    for domain in sorted(required_domains):
        for seed in sorted(required_seeds):
            key = (domain, seed)
            if key not in by_key:
                raise ValueError(f"resume contract lacks initial state for {domain} seed {seed}")
            record = by_key[key]
            artifact = record["artifact"]
            path = Path(artifact["path"])
            if not path.is_file():
                fallback = output_dir / path.name
                path = fallback if fallback.is_file() else path
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"resume initial-state artifact hash mismatch: {path}")
            initial = torch.load(path, map_location="cpu", weights_only=False)
            required_fields = {
                "model",
                "optimizer",
                "scheduler",
                "rng",
                "sample_order_cursor",
                "augmentation_state",
                "global_step",
                "scheduler_total_steps",
                "full_checkpoint_restored",
            }
            missing = sorted(required_fields - set(initial))
            if missing:
                raise ValueError(f"resume initial state {path} is missing fields: {missing}")
            if tensor_state_sha256(initial["model"]) != record["model_state_sha256"]:
                raise ValueError(f"resume model-state hash mismatch: {path}")
            if nested_state_sha256(initial["optimizer"]) != record["optimizer_state_sha256"]:
                raise ValueError(f"resume optimizer-state hash mismatch: {path}")
            if nested_state_sha256(initial["scheduler"]) != record["scheduler_state_sha256"]:
                raise ValueError(f"resume scheduler-state hash mismatch: {path}")
            states[key] = initial
    return states, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    completed: set[str] = set()
    if args.resume_run is not None:
        output_dir, contract, completed = _load_resume(args)
        _hydrate_resume_args(args, contract)
    _validate_args(args)
    if args.resume_run is None:
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(
                f"output directory is not empty: {output_dir}; use --resume-run to continue it"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

    if "galp" in args.enabled:
        preflights = (
            (
                "train",
                args.galp_manifest_preflight.as_dict(),
                "galp_manifest_preflight",
                output_dir / "manifest_preflight.json",
            ),
            (
                "validation",
                args.galp_validation_manifest_preflight.as_dict(),
                "galp_validation_manifest_preflight",
                output_dir / "manifest_preflight_validation.json",
            ),
        )
        for split, preflight_payload, contract_key, preflight_path in preflights:
            if args.resume_run is None:
                write_json(preflight_path, preflight_payload)
                continue
            contracted = contract["pipelines"].get(contract_key)
            if contracted is None and split == "validation":
                contracted = contract["pipelines"].get("galp_manifest_preflight")
            if contracted != preflight_payload:
                raise ValueError(
                    f"resume GALP {split} manifest preflight differs from the immutable contract"
                )
            if not preflight_path.is_file():
                raise ValueError(
                    f"resume run is missing {preflight_path.name}"
                )
            persisted = json.loads(preflight_path.read_text(encoding="utf-8"))
            if persisted != preflight_payload:
                raise ValueError(
                    f"resume {preflight_path.name} differs from current preflight"
                )

    train_samples, train_meta = load_training_manifest(
        args.train_manifest, root=args.train_root, expected_split="train"
    )
    val_samples, val_meta = load_training_manifest(args.val_manifest, root=args.val_root, expected_split="val")
    if "galp" in args.enabled:
        _validate_galp_dataset_binding(
            train_samples,
            train_meta,
            args.galp_manifest,
            args.galp_manifest_preflight,
            split="train",
            allow_layout_manifest_rebinding=args.allow_galp_layout_manifest_rebinding,
        )
        _validate_galp_dataset_binding(
            val_samples,
            val_meta,
            args.galp_validation_manifest,
            args.galp_validation_manifest_preflight,
            split="validation",
            allow_layout_manifest_rebinding=args.allow_galp_layout_manifest_rebinding,
        )
    separation = validate_dataset_separation(train_samples, val_samples)
    if not separation["ok"]:
        raise ValueError(f"train/validation dataset contamination: {separation}")
    if args.resume_run is None:
        contract = _build_contract(args, train_meta, val_meta, separation)
        write_json(output_dir / "contract.json", contract)
    elif contract["contract_sha256"] != sha256_json({key: value for key, value in contract.items() if key != "contract_sha256"}):
        raise ValueError("resume contract hash mismatch")
    if args.dry_run_contract:
        return contract

    phases = contract["execution"]["phases"]
    required_seeds = {args.seed}
    if phases["convergence"]["enabled"]:
        required_seeds.update(phases["convergence"]["seeds"])
    required_domains = {DOMAINS[pipeline] for pipeline in args.enabled}
    initial_states: dict[tuple[str, int], dict[str, Any]] = {}
    initial_records: list[dict[str, Any]] = []
    if args.resume_run is not None:
        initial_states, initial_records = _load_audited_initial_states(
            output_dir, contract, required_domains, required_seeds
        )
    else:
        for domain in sorted(required_domains):
            for seed in sorted(required_seeds):
                initial, record = _prepare_initial_state(args, contract, domain, seed, output_dir)
                initial_states[(domain, seed)] = initial
                initial_records.append(record)
        write_json(
            output_dir / "initial_states.json",
            {
                "architecture": MODEL_ARCHITECTURE,
                "states": initial_records,
                "same_domain_shared_across_pipelines": True,
                "cross_domain_weight_equivalence_claim": False,
            },
        )
        contract["initial_states"] = initial_records
        contract["optimizer"]["parameter_groups_by_domain"] = {
            domain: next(
                record["parameter_groups"]
                for record in initial_records
                if record["domain"] == domain and int(record["seed"]) == args.seed
            )
            for domain in sorted(required_domains)
        }
        contract["contract_sha256"] = sha256_json(
            {key: value for key, value in contract.items() if key != "contract_sha256"}
        )
        write_json(output_dir / "contract.json", contract)

    pipeline_artifacts: dict[str, dict[str, Any]] = {}
    raw_probes: dict[str, dict[str, Any]] = {}
    for pipeline in args.enabled:
        if pipeline in completed:
            pipeline_artifacts[pipeline] = json.loads(
                (output_dir / f"pipeline_{pipeline}.json").read_text(encoding="utf-8")
            )
            raw_artifact = pipeline_artifacts[pipeline].get("first_step_raw_artifact", {})
            raw_path = Path(raw_artifact.get("path", ""))
            if not raw_path.is_file():
                fallback = output_dir / raw_path.name
                raw_path = fallback if fallback.is_file() else raw_path
            if raw_path.is_file() and sha256_file(raw_path) == raw_artifact.get("sha256"):
                raw_probes[pipeline] = torch.load(
                    raw_path, map_location="cpu", weights_only=False
                )
            continue
        artifact, raw_probe = _run_pipeline(
            args,
            contract,
            pipeline,
            train_samples,
            val_samples,
            initial_states,
            output_dir,
        )
        pipeline_artifacts[pipeline] = artifact
        raw_probes[pipeline] = raw_probe
        write_json(output_dir / f"pipeline_{pipeline}.json", artifact)

    existing_semantic = {}
    semantic_path = output_dir / "semantic_comparison.json"
    if args.resume_run is not None and semantic_path.is_file():
        existing_semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic: dict[str, Any] = {}
    group_status: dict[str, dict[str, str]] = {}
    for group, pair in COMPARISON_GROUPS.items():
        if all(pipeline in args.enabled for pipeline in pair):
            if not all(pipeline in raw_probes for pipeline in pair):
                prior = existing_semantic.get(group, {})
                if prior.get("status") in ("passed", "warning", "failed"):
                    semantic[group] = prior
                else:
                    semantic[group] = {
                        "status": "warning",
                        "warnings": [
                            "semantic raw tensors unavailable after resume; existing pipeline probes remain auditable"
                        ],
                        "pipelines": list(pair),
                    }
            else:
                semantic[group] = _semantic_compare(
                    group,
                    pair[0],
                    raw_probes[pair[0]],
                    pair[1],
                    raw_probes[pair[1]],
                    contract["gates"]["semantic"],
                )
            status = semantic[group]["status"]
            group_status[group] = empty_status()
            group_status[group]["semantic"] = status
            group_status[group]["correctness"] = (
                "failed"
                if any(pipeline_artifacts[name]["status"]["correctness"] == "failed" for name in pair)
                else "passed"
            )
            group_status[group]["artifact"] = "passed"
            group_status[group]["overall"] = _overall_status(group_status[group])
            for name in pair:
                pipeline_artifacts[name]["status"]["semantic"] = status
                pipeline_artifacts[name]["status"]["overall"] = _overall_status(pipeline_artifacts[name]["status"])
                write_json(output_dir / f"pipeline_{name}.json", pipeline_artifacts[name])
        else:
            semantic[group] = {"status": "not_run", "pipelines": list(pair)}
            group_status[group] = empty_status("not_run")
    write_json(output_dir / "semantic_comparison.json", semantic)

    pipeline_status = {
        name: pipeline_artifacts[name]["status"] if name in pipeline_artifacts else empty_status("not_run")
        for name in PIPELINES
    }
    sample_order = {
        "basis": "optimizer_consumed_ids",
        "pipelines": {name: artifact["sample_order"] for name, artifact in pipeline_artifacts.items()},
    }
    write_json(output_dir / "sample_order.json", sample_order)
    write_json(output_dir / "augmentation_contract.json", contract["augmentation"])
    write_json(
        output_dir / "repeat_resets.json",
        {
            name: {
                phase: [repeat["reset"] for repeat in result.get("repeats", [])]
                for phase, result in artifact["phase_results"].items()
            }
            for name, artifact in pipeline_artifacts.items()
        },
    )
    curves = {
        name: artifact["phase_results"].get("convergence", {"seeds": []})
        for name, artifact in pipeline_artifacts.items()
    }
    write_json(output_dir / "training_curves.json", curves)
    commands = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "resolved_contract_command": " ".join(map(str, sys.argv)),
        "execution_mode": args.execution_mode,
    }
    write_json(output_dir / "commands.json", commands)
    write_json(output_dir / "run_metadata.json", runtime_metadata())

    results = {
        "schema_version": TRAINING_RESULT_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "execution_mode": args.execution_mode,
        "enabled_pipelines": args.enabled,
        "required_comparison_groups": args.required_groups,
        "pipeline_status": pipeline_status,
        "comparison_group_status": group_status,
        "pipelines": {
            name: {
                "artifact": f"pipeline_{name}.json",
                "status": pipeline_status[name],
                "input_domain": DOMAINS[name],
                "first_step_semantic": (
                    "passed"
                    if not artifact.get("first_step_semantic_probe", {}).get("failures")
                    else "failed"
                ),
                "step_aggregate": artifact["phase_results"].get("step", {}).get("aggregate"),
                "report_table_summary": artifact["phase_results"]
                .get("step", {})
                .get("aggregate", {})
                .get("report_table_summary"),
                "convergence_aggregate": artifact["phase_results"].get("convergence", {}).get("aggregate"),
            }
            for name, artifact in pipeline_artifacts.items()
        },
        "claim_boundary": "No final accuracy claim is made without full recipe-matched multi-seed ImageNet training.",
    }
    write_json(output_dir / "results.json", results)
    write_artifact_hashes(output_dir, excluded=("artifact_hashes.json", "validation.json"))

    from training.validate import validate_output

    validation = validate_output(output_dir, write_result=False)
    write_json(output_dir / "validation.json", validation)
    write_artifact_hashes(output_dir, excluded=("artifact_hashes.json", "validation.json"))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        result = run(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
