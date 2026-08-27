#!/usr/bin/env python3
"""Build an immutable no-shuffle contract and run the DCT-major benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from common import (
    BLOCK_MAJOR_RUNTIME_PROFILE,
    CONTRACT_SCHEMA,
    DEFAULT_PIPELINES,
    GALP_PIPELINES,
    HERE,
    PIPELINES,
    REPO_ROOT,
    RGBNOMORE_BENCHMARK_ROOT,
    collect_sequential_samples,
    fingerprint_file,
    load_sample_manifest,
    manifest_snapshot,
    parse_manifest,
    resolve_coefficient_selection,
    sha256_file,
    source_fingerprints,
    write_canonical_index,
    write_json,
    write_sample_manifest,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_DCT_MAJOR_MANIFEST = Path("/tmp/galp-blockmajor-512-s1024-rg128/manifest.bin")
DEFAULT_DCT_MAJOR_LABELS = (
    REPO_ROOT
    / "galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/labels.json"
)
DEFAULT_BINDING_DIR = REPO_ROOT / "build/galp/torch"
DEFAULT_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")
DEFAULT_RAW_MASK_ORACLE_DIR = (
    REPO_ROOT
    / "galp/experiments/coefficient_mask_evaluator/runs/imagenet_val_k1_64_20260816_h100"
)

PRESETS = {
    "smoke": {"batch_size": 2, "warmup_batches": 0, "measurement_batches": 2, "repeats": 1, "workers": 2},
    "e2e": {"batch_size": 50, "warmup_batches": 0, "measurement_batches": 1000, "repeats": 5, "workers": 8},
}


def _setting(args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    return int(PRESETS[args.preset][name] if value is None else value)


def _sample_plan(args: argparse.Namespace) -> tuple[int, int, int, int]:
    """Resolve batches so an explicit sample count is consumed in full."""

    batch_size = _setting(args, "batch_size")
    warmup_batches = _setting(args, "warmup_batches")
    measurement_batches = _setting(args, "measurement_batches")
    sample_count = batch_size * (warmup_batches + measurement_batches)
    if args.sample_count is None:
        return batch_size, warmup_batches, measurement_batches, sample_count

    sample_count = int(args.sample_count)
    warmup_images = batch_size * warmup_batches
    if sample_count <= warmup_images:
        raise ValueError(
            f"sample_count must exceed the {warmup_images} images consumed by warmup"
        )
    measured_images = sample_count - warmup_images
    derived_measurement_batches = (measured_images + batch_size - 1) // batch_size
    if args.measurement_batches is not None and measurement_batches != derived_measurement_batches:
        raise ValueError(
            "explicit measurement_batches does not consume sample_count exactly: "
            f"expected {derived_measurement_batches}, got {measurement_batches}"
        )
    return batch_size, warmup_batches, derived_measurement_batches, sample_count


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        if any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty benchmark output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _resolve_torch_binding_artifact(directory: Path) -> Path:
    directory = directory.resolve()
    candidates = sorted(
        path.resolve()
        for path in directory.glob("_galp_direct_dct*.so")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one _galp_direct_dct extension in "
            f"{directory}, found {len(candidates)}"
        )
    return candidates[0]


def _manifest_contract(path: Path, *, expected_version: int, hash_payloads: bool) -> dict[str, Any]:
    parsed = parse_manifest(path)
    if int(parsed["version"]) != expected_version:
        raise ValueError(
            f"manifest layout mismatch: {path} has version={parsed['version']} "
            f"({parsed['physical_layout']}), expected version={expected_version}"
        )
    return manifest_snapshot(path, hash_payloads=hash_payloads)


def _validate_fixed_source_geometry(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = (512, 512)

    def dimensions(sample: dict[str, Any]) -> tuple[int, int]:
        with Image.open(sample["path"]) as image:
            return tuple(int(item) for item in image.size)

    with ThreadPoolExecutor(max_workers=16) as executor:
        observed = list(executor.map(dimensions, samples))
    mismatch = next(
        (
            (index, shape, samples[index]["path"])
            for index, shape in enumerate(observed)
            if shape != expected
        ),
        None,
    )
    if mismatch is not None:
        index, shape, path = mismatch
        raise ValueError(
            f"fixed-center-224-from-512 requires every source to be 512x512; "
            f"sample {index} is {shape[0]}x{shape[1]}: {path}"
        )
    return {
        "validated_image_count": len(observed),
        "source_width": expected[0],
        "source_height": expected[1],
        "crop_width": 224,
        "crop_height": 224,
        "crop_origin_x": 144,
        "crop_origin_y": 144,
    }


def _validate_fixed_manifest_geometry(
    reader: Any,
    expected_images: int,
    *,
    metadata_getter: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove that the native store implements the fixed-512 DCT profile.

    The JPEG paths and the compressed manifest are independent inputs.  A
    matching image count/order is insufficient: two ImageNet views can have
    the same filenames while containing different JPEG encodings.  Validate
    the geometry and sampling recorded in native reconstruct metadata before
    admitting a manifest into the benchmark contract.
    """

    observed_images = int(reader.image_count)
    if observed_images != expected_images:
        raise ValueError(
            f"DCT-major manifest has {observed_images} images, expected {expected_images}"
        )

    read_metadata = metadata_getter or reader.image_metadata
    expected_components = {
        0: (64, 64, 2, 2),
        1: (32, 32, 1, 1),
        2: (32, 32, 1, 1),
    }
    for image_id in range(expected_images):
        metadata = read_metadata(image_id)
        width = int(metadata.get("image_width", -1))
        height = int(metadata.get("image_height", -1))
        if (width, height) != (512, 512):
            raise ValueError(
                "fixed-center-224-from-512 requires the DCT-major manifest to "
                f"contain the same 512x512 JPEG view; image {image_id} metadata "
                f"is {width}x{height}"
            )

        components = {
            int(component.get("semantic_slot_id", -1)): component
            for component in metadata.get("components", [])
            if bool(component.get("present", False))
        }
        if set(components) != set(expected_components):
            raise ValueError(
                "fixed-center-224-from-512 requires exactly Y/Cb/Cr components; "
                f"manifest image {image_id} has semantic slots {sorted(components)}"
            )
        for slot, expected in expected_components.items():
            component = components[slot]
            observed = (
                int(component.get("width_in_blocks", -1)),
                int(component.get("height_in_blocks", -1)),
                int(component.get("h_samp_factor", -1)),
                int(component.get("v_samp_factor", -1)),
            )
            if observed != expected:
                raise ValueError(
                    "fixed-center-224-from-512 requires 4:2:0 native DCT geometry; "
                    f"manifest image {image_id} component slot {slot} has "
                    f"blocks/sampling={observed}, expected {expected}"
                )

    return {
        "validated_image_count": expected_images,
        "source_width": 512,
        "source_height": 512,
        "jpeg_sampling": "4:2:0",
        "y_blocks": [64, 64],
        "cbcr_blocks": [32, 32],
        "semantic_slots": [0, 1, 2],
    }


def _block_major_access_contract(
    directory: Path,
    dct_major_storage: dict[str, Any],
    shard_ids: Sequence[int],
    *,
    required_active_output_schedule_shard_ids: Sequence[int] = (),
) -> dict[str, Any]:
    directory = directory.resolve()
    companion_index = directory / "manifest.block_major_access.bin"
    if not companion_index.is_file():
        raise FileNotFoundError(
            f"block-major access companion index is missing: {companion_index}"
        )
    expected = [
        directory / f"shard_{int(shard_id):06d}.block_major_access.bin"
        for shard_id in shard_ids
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"block-major shard descriptor is missing: {missing[0]}")
    observed = sorted(directory.glob("shard_*.block_major_access.bin"))
    if {path.resolve() for path in observed} != {path.resolve() for path in expected}:
        raise ValueError("block-major descriptor directory has unexpected shard files")

    companion = fingerprint_file(companion_index)
    descriptors = [
        {
            **fingerprint_file(path),
            "shard_id": int(shard_id),
        }
        for shard_id, path in zip(shard_ids, expected, strict=True)
    ]
    descriptor_bytes = int(companion["size_bytes"]) + sum(
        int(item["size_bytes"]) for item in descriptors
    )
    required_schedule_shard_ids = tuple(
        int(shard_id) for shard_id in required_active_output_schedule_shard_ids
    )
    required_schedule_paths = [
        directory / f"shard_{shard_id:06d}.active_output_schedule.bin"
        for shard_id in required_schedule_shard_ids
    ]
    missing_schedules = [path for path in required_schedule_paths if not path.is_file()]
    if missing_schedules:
        raise FileNotFoundError(
            "scheduled-range benchmark requires a pre-materialized active-output "
            f"schedule before contract snapshot: {missing_schedules[0]}"
        )
    active_output_schedules = [
        fingerprint_file(path)
        for path in sorted(directory.glob("shard_*.active_output_schedule.bin"))
    ]
    active_output_schedule_bytes = sum(
        int(item["size_bytes"]) for item in active_output_schedules
    )
    base_storage_bytes = int(dct_major_storage["persistent_bytes"]) + int(
        dct_major_storage["manifest"]["size_bytes"]
    )
    if base_storage_bytes <= 0:
        raise ValueError("DCT-major base storage is empty")
    storage_increase_ratio = descriptor_bytes / base_storage_bytes
    all_sidecar_bytes = descriptor_bytes + active_output_schedule_bytes
    all_sidecar_storage_increase_ratio = all_sidecar_bytes / base_storage_bytes
    if all_sidecar_storage_increase_ratio > 0.01:
        raise ValueError(
            "block-major sidecar storage exceeds the 1% storage gate: "
            f"{100.0 * all_sidecar_storage_increase_ratio:.6f}%"
        )
    return {
        "directory": str(directory),
        "companion_index": companion,
        "shards": descriptors,
        "shard_count": len(descriptors),
        "descriptor_bytes": descriptor_bytes,
        "required_active_output_schedule_shard_ids": list(required_schedule_shard_ids),
        "active_output_schedules": active_output_schedules,
        "active_output_schedule_bytes": active_output_schedule_bytes,
        "all_sidecar_bytes": all_sidecar_bytes,
        "base_storage_bytes": base_storage_bytes,
        "total_storage_bytes_with_descriptor": base_storage_bytes + descriptor_bytes,
        "total_storage_bytes_with_sidecars": base_storage_bytes + all_sidecar_bytes,
        "storage_increase_ratio": storage_increase_ratio,
        "storage_increase_percent": 100.0 * storage_increase_ratio,
        "all_sidecar_storage_increase_ratio": all_sidecar_storage_increase_ratio,
        "all_sidecar_storage_increase_percent": 100.0 * all_sidecar_storage_increase_ratio,
        "passes_one_percent": True,
        "passes_half_percent": all_sidecar_storage_increase_ratio <= 0.005,
    }


def build_contract(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], Path]:
    batch_size, warmup_batches, measurement_batches, sample_count = _sample_plan(args)
    repeats = _setting(args, "repeats")
    workers = _setting(args, "workers")
    if warmup_batches != 0:
        raise ValueError(
            "block-major production profile requires warmup_batches=0 so a boundary "
            "cannot split and reactivate a physical shard"
        )
    if any(name in args.pipelines for name in GALP_PIPELINES) and args.block_major_access_dir is None:
        raise ValueError("GALP DCT-major pipelines require --block-major-access-dir")

    dct_major = _manifest_contract(
        args.dct_major_manifest,
        expected_version=1,
        hash_payloads=args.hash_payloads,
    )
    image_count = int(dct_major["header"]["image_count"])
    block_major_access_dir = getattr(args, "block_major_access_dir", None)
    block_major_access: dict[str, Any] | None = None
    if block_major_access_dir is not None:
        parsed_dct_major = parse_manifest(args.dct_major_manifest)
        required_schedule_shard_ids: list[int] = []
        if any(name in args.pipelines for name in GALP_PIPELINES):
            selected_end = sample_count
            consumed = 0
            selected_shard_ids: list[int] = []
            for shard in parsed_dct_major["shards"]:
                first = int(shard["first_global_image_index"])
                count = int(shard["image_count"])
                end = first + count
                if first >= selected_end:
                    break
                if first != consumed or end > selected_end:
                    raise ValueError(
                        "scheduled manifest-shard sample_count must cover a contiguous "
                        "prefix of complete physical shards"
                    )
                selected_shard_ids.append(int(shard["shard_id"]))
                consumed = end
            if consumed != selected_end:
                raise ValueError(
                    "scheduled manifest-shard sample_count is not covered by complete physical shards"
                )
            if selected_shard_ids:
                if selected_shard_ids[0] != 0:
                    raise ValueError("scheduled active-output sidecar requires canonical shard 0")
                # The native P4 policy persists only the critical first-shard
                # schedule; later shard schedules are built under overlap.
                required_schedule_shard_ids = [0]
        block_major_access = _block_major_access_contract(
            block_major_access_dir,
            dct_major,
            [int(item["shard_id"]) for item in parsed_dct_major["shards"]],
            required_active_output_schedule_shard_ids=required_schedule_shard_ids,
        )
        block_major_access_dir = Path(block_major_access["directory"])
    samples, sample_provenance = collect_sequential_samples(
        data_root=args.data_root,
        split=args.split,
        label_map_json=args.dct_major_label_map,
        expected_images=image_count,
        sample_count=sample_count,
        hash_samples=args.hash_samples,
    )
    fixed_geometry = _validate_fixed_source_geometry(samples)
    try:
        from galp.diagnostics.direct_dct import image_metadata
        from galp.torch import DirectDctReader
    except ImportError as error:
        raise RuntimeError(
            "native Torch binding is required to validate DCT-major manifest geometry"
        ) from error
    manifest_reader = DirectDctReader(
        args.dct_major_manifest,
        module_path=args.torch_binding_dir,
    )
    manifest_geometry = _validate_fixed_manifest_geometry(
        manifest_reader,
        image_count,
        metadata_getter=lambda image_id: image_metadata(manifest_reader, image_id),
    )
    sample_manifest_path = output_dir / "sample_manifest.json"
    sample_manifest_sha256 = write_sample_manifest(sample_manifest_path, samples, sample_provenance)
    canonical_index = output_dir / "canonical_index.csv"
    write_canonical_index(canonical_index, samples)

    rgb_checkpoint = args.rgb_checkpoint or args.rgbnomore_root / "checkpoints/imgnetRGBViTTi_ep300_74.1.pth"
    dct_checkpoint = args.dct_checkpoint or args.rgbnomore_root / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
    for path in (rgb_checkpoint, dct_checkpoint, args.torch_binding_dir, args.rgbnomore_root):
        if not path.exists():
            raise FileNotFoundError(path)
    rgb_checkpoint_sha256 = sha256_file(rgb_checkpoint)
    dct_checkpoint_sha256 = sha256_file(dct_checkpoint)
    torch_binding_artifact_path = _resolve_torch_binding_artifact(args.torch_binding_dir)
    torch_binding_artifact = fingerprint_file(torch_binding_artifact_path)

    galp_config: dict[str, Any] = {
        "torch_binding_dir": str(args.torch_binding_dir.resolve()),
        "torch_binding_artifact": torch_binding_artifact,
        "runtime_profile": BLOCK_MAJOR_RUNTIME_PROFILE,
        "manifest": str(args.dct_major_manifest.resolve()),
        "manifest_version": 1,
        "physical_layout": "dct-major/spatial-major-image-minor",
        "preprocess": "rgbnomore-val-pushdown",
        "model_key": "dct",
        "checkpoint_sha256": dct_checkpoint_sha256,
        "block_major_access_dir": (
            str(block_major_access_dir) if block_major_access_dir is not None else None
        ),
        "role": "native scheduled bounded-I/O RGB-no-more profile per manifest shard",
    }
    coefficient_selection = resolve_coefficient_selection(args.dct_coeffs)
    pipeline_configs: dict[str, Any] = {
        "enabled": list(args.pipelines),
        "dct_major_pushdown": {
            **galp_config,
            **resolve_coefficient_selection("all"),
            "display_name": "GALP",
        },
        "dct_major_coefficient_pushdown": {
            **galp_config,
            **coefficient_selection,
            "display_name": "GALP-DCT-pushdown",
            "role": "native crop and configurable raw-coefficient pushdown per manifest shard",
        },
        "rgbnomore": {
            "root": str(args.rgbnomore_root.resolve()),
            "prefetch_factor": int(args.prefetch_factor),
            "shuffle": False,
        },
        "dali": {
            "device_id": int(args.device.split(":", 1)[1]) if ":" in args.device else 0,
            "prefetch_queue_depth": int(args.dali_prefetch_depth),
            "random_shuffle": False,
        },
        "pytorch": {
            "prefetch_factor": int(args.prefetch_factor),
            "shuffle": False,
        },
    }

    runtime_files = [
        HERE / "common.py",
        HERE / "feature_model.py",
        HERE / "pipeline.py",
        HERE / "run.py",
        HERE / "validate.py",
        HERE / "verify_coefficient_semantics.py",
        REPO_ROOT / "galp/torch/direct_dct.py",
        REPO_ROOT / "galp/diagnostics/direct_dct.py",
        REPO_ROOT / "galp/profiles/_base.py",
        REPO_ROOT / "galp/profiles/rgbnomore.py",
        REPO_ROOT / "galp/include/galp/profiles/direct_dct.hpp",
        REPO_ROOT / "galp/include/galp/profiles/registry.hpp",
        REPO_ROOT / "galp/include/galp/profiles/rgbnomore.hpp",
        RGBNOMORE_BENCHMARK_ROOT / "shared/manifest_contract.py",
        RGBNOMORE_BENCHMARK_ROOT / "inference/model_factory.py",
        torch_binding_artifact_path,
        args.rgbnomore_root / "models/plainvit.py",
        args.rgbnomore_root / "datasets.py",
    ]
    raw_mask_oracle: dict[str, Any] | None = None
    if args.raw_mask_oracle_dir is not None and coefficient_selection["coefficient_selection_kind"] == "prefix":
        oracle_dir = args.raw_mask_oracle_dir.resolve(strict=True)
        oracle_predictions = oracle_dir / "per_sample_top1.csv.gz"
        oracle_curve = oracle_dir / "prefix_accuracy_curve.csv"
        oracle_metadata = oracle_dir / "run_metadata.json"
        if not all(
            path.is_file()
            for path in (oracle_predictions, oracle_curve, oracle_metadata)
        ):
            raise FileNotFoundError(
                "raw-mask oracle directory must contain per_sample_top1.csv.gz, "
                "prefix_accuracy_curve.csv, and run_metadata.json"
            )
        raw_mask_oracle = {
            "schema": "coefficient-mask-evaluator-result-v1",
            "condition_id": f"prefix_k{coefficient_selection['coefficient_count']:02d}",
            "coefficient_spec": coefficient_selection["coefficient_spec"],
            "per_sample_top1": fingerprint_file(oracle_predictions),
            "prefix_accuracy_curve": fingerprint_file(oracle_curve),
            "run_metadata": fingerprint_file(oracle_metadata),
        }

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "benchmark_id": args.benchmark_id or f"dct-major-{args.workload}-{args.preset}",
        "dataset": {
            "name": "ImageNet-1K",
            "split": args.split,
            "data_root": str(args.data_root.resolve()),
            "sample_manifest": str(sample_manifest_path.resolve()),
            "sample_manifest_sha256": sample_manifest_sha256,
            "canonical_index_csv": str(canonical_index.resolve()),
            "canonical_index_sha256": sha256_file(canonical_index),
            "sample_count": sample_count,
            "full_image_count": image_count,
            "source_image_shape_hwc": [512, 512, 3],
            "source_geometry_validation": fixed_geometry,
            "manifest_geometry_validation": manifest_geometry,
            "jpeg_selected_bytes": sum(int(sample["size_bytes"]) for sample in samples),
            "jpeg_full_dataset_bytes": (
                sum(int(sample["size_bytes"]) for sample in samples)
                if sample_count == image_count
                else None
            ),
            "raw_rgb_selected_bytes": sample_count * 512 * 512 * 3,
            "sample_order": "galp_image_id_ascending",
            "shuffle": False,
            "dct_major_storage": dct_major,
            "block_major_access": block_major_access,
        },
        "execution": {
            "batch_size": batch_size,
            "workers": workers,
            "warmup_batches": warmup_batches,
            "measurement_batches": measurement_batches,
            "repeats": repeats,
            "seed": int(args.seed),
            "device": args.device,
            "precision": "fp32",
            "shuffle": False,
            "drop_last": False,
            "aggregate_exclude_first_repeat": repeats > 1,
            "model_stream_priority": "greatest",
            "cold_start_model_prime": True,
            "cold_file_cache_eviction": bool(args.evict_pipeline_file_cache),
            "cold_protocol": args.cold_protocol,
        },
        "workload": {
            "kind": args.workload,
            "feature_stage": args.feature_stage,
            "materialize_features": bool(args.materialize_features),
            "feature_output": "[N,192] after classhead.ch_tanh" if args.feature_stage == "penultimate" else "[N,192] after LayerNorm+mean pool",
        },
        "preprocess": {
            "profile": "fixed-center-224-from-512",
            "rgb": {
                "resize_shorter": None,
                "crop": "center",
                "crop_size": [224, 224],
                "crop_origin": [144, 144],
                "range": [-1.0, 1.0],
            },
            "dct": {
                "profile": "RGB-no-more CenterCrop_DCT(28) from fixed 64x64 Y-block source",
                "source_grid_size_blocks": [64, 64],
                "crop_reference_size_blocks": [32, 32],
                "center_crop_size_blocks": [56, 56],
                "y_shape": [1, 28, 28, 8, 8],
                "cbcr_shape": [2, 14, 14, 8, 8],
                "stored_coefficients": "all-64",
                "model_ready_frequency_grid": "dense-64",
                "selection_stage": "raw quantized coefficients before dequantization and frequency mixing",
                "range": [-1.0, 1.0],
            },
        },
        "models": {
            "rgbnomore_root": str(args.rgbnomore_root.resolve()),
            "rgb": {
                "architecture": "RGB-no-more ViT-Ti RGB",
                "checkpoint": str(rgb_checkpoint.resolve()),
                "checkpoint_sha256": rgb_checkpoint_sha256,
                "input_domain": "RGB",
            },
            "dct": {
                "architecture": "RGB-no-more JPEG-Ti ViT-Ti DCT",
                "checkpoint": str(dct_checkpoint.resolve()),
                "checkpoint_sha256": dct_checkpoint_sha256,
                "input_domain": "JPEG_DCT",
            },
        },
        "pipelines": pipeline_configs,
        "semantic_validation": {
            "sample_count": int(args.semantic_samples),
            "dct_quantized_level_tolerance": 1,
            "normalized_dct_level": 1.0 / 1020.0,
            "input_max_abs": 1.0e-3,
            "input_mean_abs": 1.0e-4,
            "feature_cosine_min": 0.999,
            "logit_cosine_min": 0.999,
            "semantic_top1_agreement_min": 1.0,
            "full_prediction_top1_agreement_min": 0.999,
            "native_oracle_accuracy_delta_max": 0.0005,
            "raw_mask_oracle": raw_mask_oracle,
        },
        "stability_gates": {
            "maximum_hot_cv": 0.05,
            "maximum_endpoint_drift": 0.05,
            "minimum_speedup": None,
        },
        "source_fingerprints": source_fingerprints(runtime_files),
    }
    contract_path = output_dir / "contract.json"
    write_json(contract_path, contract)
    return contract, contract_path


def _run_streamed(
    command: list[str], *, env: dict[str, str], log: Path, dry_run: bool
) -> tuple[int, float, float | None]:
    print("COMMAND " + shlex.join(command), flush=True)
    if dry_run:
        return 0, 0.0, None
    started = time.perf_counter()
    first_output_seconds: float | None = None
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND " + shlex.join(command) + "\n")
        stream.flush()
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
        with process.stdout:
            for line in process.stdout:
                if first_output_seconds is None and line.startswith("GALP_FIRST_OUTPUT_READY "):
                    first_output_seconds = time.perf_counter() - started
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
        return int(process.wait()), time.perf_counter() - started, first_output_seconds


def _pipeline_cache_paths(contract: dict[str, Any], pipeline: str) -> list[Path]:
    """Return immutable inputs whose file-backed pages define this pipeline's cold state."""

    config = contract["pipelines"][pipeline]
    paths = {
        Path(contract["dataset"]["sample_manifest"]),
        Path(contract["dataset"]["canonical_index_csv"]),
    }
    model_key = "rgb" if pipeline in {"dali", "pytorch"} else "dct"
    paths.add(Path(contract["models"][model_key]["checkpoint"]))
    if pipeline in GALP_PIPELINES:
        snapshot = contract["dataset"]["dct_major_storage"]
        paths.add(Path(snapshot["manifest"]["path"]))
        paths.update(Path(payload["path"]) for payload in snapshot["payloads"])
        if config.get("block_major_access_dir") is not None:
            access = contract["dataset"].get("block_major_access") or {}
            fingerprints = [access.get("companion_index"), *access.get("shards", [])]
            fingerprints.extend(access.get("active_output_schedules", []))
            paths.update(
                Path(item["path"])
                for item in fingerprints
                if isinstance(item, dict) and item.get("path")
            )
    else:
        samples = load_sample_manifest(
            Path(contract["dataset"]["sample_manifest"]),
            contract["dataset"]["sample_manifest_sha256"],
        )
        paths.update(Path(sample["path"]) for sample in samples)
    return sorted((path.resolve() for path in paths), key=str)


def _evict_pipeline_file_cache(contract: dict[str, Any], pipeline: str) -> dict[str, Any]:
    """Evict only this pipeline's immutable file pages; never use global drop_caches."""

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("per-file cold-cache protocol requires os.posix_fadvise(POSIX_FADV_DONTNEED)")
    paths = _pipeline_cache_paths(contract, pipeline)
    started = time.perf_counter()
    total_bytes = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"cold-cache input disappeared: {path}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
        total_bytes += path.stat().st_size
    return {
        "method": "posix_fadvise(POSIX_FADV_DONTNEED)",
        "scope": "pipeline immutable data, manifests, descriptors, and checkpoint; no global drop_caches",
        "file_count": len(paths),
        "bytes_addressed": total_bytes,
        "seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    _prepare_output_dir(output_dir)
    contract, contract_path = build_contract(args, output_dir)
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    env = os.environ.copy()
    pythonpath = str(args.torch_binding_dir.resolve())
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
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
        commands.append({"name": pipeline, "command": command, "env": {"PYTHONPATH": pythonpath}})
        if args.evict_pipeline_file_cache and not args.dry_run:
            commands[-1]["file_cache_eviction"] = _evict_pipeline_file_cache(contract, pipeline)
        code, wall_seconds, first_output_seconds = _run_streamed(
            command,
            env=env,
            log=output_dir / f"pipeline_{pipeline}.log",
            dry_run=args.dry_run,
        )
        commands[-1]["subprocess_wall_seconds"] = wall_seconds
        commands[-1]["spawn_to_first_output_seconds"] = first_output_seconds
        if code != 0:
            write_json(output_dir / "commands.json", commands)
            write_json(output_dir / "failed.json", {"pipeline": pipeline, "exit_code": code})
            return code
    validate_command = [
        str(python),
        str(HERE / "validate.py"),
        "--contract",
        str(contract_path),
        "--output-dir",
        str(output_dir),
    ]
    commands.append({"name": "validate", "command": validate_command})
    code, validation_wall_seconds, _ = _run_streamed(
        validate_command,
        env=env,
        log=output_dir / "validate.log",
        dry_run=args.dry_run,
    )
    commands[-1]["subprocess_wall_seconds"] = validation_wall_seconds
    write_json(output_dir / "commands.json", commands)
    write_json(
        output_dir / "run_metadata.json",
        {
            "argv": sys.argv,
            "python": str(python),
            "dry_run": args.dry_run,
            "directory_write_boundary": str(HERE),
            "source_tree_modified_by_runner": False,
        },
    )
    return code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--workload", choices=("feature-extraction", "evaluation"), default="feature-extraction")
    parser.add_argument("--feature-stage", choices=("pooled", "penultimate"), default="penultimate")
    parser.add_argument("--materialize-features", action="store_true")
    parser.add_argument("--pipelines", choices=PIPELINES, nargs="+", default=list(DEFAULT_PIPELINES))
    parser.add_argument("--benchmark-id")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dct-major-manifest", type=Path, default=DEFAULT_DCT_MAJOR_MANIFEST)
    parser.add_argument(
        "--block-major-access-dir",
        type=Path,
        help="validated BLOCK_MAJOR_ACCESS_V1 sidecar directory required by the GALP production profile",
    )
    parser.add_argument("--dct-major-label-map", type=Path, default=DEFAULT_DCT_MAJOR_LABELS)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--rgb-checkpoint", type=Path)
    parser.add_argument("--dct-checkpoint", type=Path)
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_BINDING_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-batches", type=int)
    parser.add_argument("--measurement-batches", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--semantic-samples", type=int, default=32)
    parser.add_argument(
        "--dct-coeffs",
        default="first:32",
        help="coefficient selection for dct_major_coefficient_pushdown: all, first:N, list:..., or random:K:SEED",
    )
    parser.add_argument(
        "--raw-mask-oracle-dir",
        type=Path,
        default=(DEFAULT_RAW_MASK_ORACLE_DIR if DEFAULT_RAW_MASK_ORACLE_DIR.is_dir() else None),
        help="completed coefficient-mask evaluator run used for full per-sample prefix validation",
    )
    parser.add_argument(
        "--cold-protocol",
        choices=("application-overlapped", "controlled-io"),
        default="application-overlapped",
        help="overlap first data prefetch with startup, or begin cold I/O only after model prime",
    )
    parser.add_argument(
        "--evict-pipeline-file-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="evict only each pipeline's immutable input pages before its independent process",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--dali-prefetch-depth", type=int, default=2)
    parser.add_argument("--hash-samples", action="store_true")
    parser.add_argument("--hash-payloads", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    resolve_coefficient_selection(args.dct_coeffs)
    return args


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
