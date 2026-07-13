#!/usr/bin/env python3
"""Validate GALP fixed-grid RGB-no-more pushdown against the Python DCT adapter."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TORCH_BINDING_DIR = REPO_ROOT / "build/galp/torch"
if DEFAULT_TORCH_BINDING_DIR.is_dir() and str(DEFAULT_TORCH_BINDING_DIR) not in sys.path:
    # Keep an explicitly configured PYTHONPATH ahead of the in-tree default.
    sys.path.append(str(DEFAULT_TORCH_BINDING_DIR))

import torch

import _galp_direct_dct as galp_dct

EXAMPLES_DIR = REPO_ROOT / "galp/examples"
TORCH_SOURCE_DIR = REPO_ROOT / "galp/torch"
for source_dir in (EXAMPLES_DIR, TORCH_SOURCE_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from direct_dct_torch_end_to_end_demo import (
    _build_rgbnomore_val_crop_transform,
    _make_image_ids,
    _read_batch,
)
from direct_dct import _component_quant_tables
from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM


DEFAULT_JPEG_TOOL = REPO_ROOT / "build/galp/tools/jpeg_dct/galp_jpeg_dct_tool"
DEFAULT_SYNTHETIC_FIXTURE_DIR = Path("/tmp/galp_rgbnomore_pushdown_fixtures")


def _read_dequantized_rgbnomore_reference(
    reader: Any,
    image_ids: list[int],
    cache_capacity_mib: int,
    transform: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RGB-no-more's dequantize/clamp-before-resize ordering."""
    y_items: list[torch.Tensor] = []
    cbcr_items: list[torch.Tensor] = []
    for image_id in image_ids:
        source = _read_batch(
            reader,
            [int(image_id)],
            None,
            "all",
            cache_capacity_mib,
            "ycbcr_dct_grid",
            None,
        )
        if source.y is None or source.cbcr is None:
            raise RuntimeError("RGB-no-more reference requires Y/CbCr grid tensors")
        y_quant, cbcr_quant = _component_quant_tables(reader, [int(image_id)], source.y.device)
        y = torch.clamp(
            source.y.to(torch.float32) * y_quant[:, None, None, None, :, :],
            min=-1024.0,
            max=1016.0,
        )
        cbcr = torch.clamp(
            source.cbcr.to(torch.float32) * cbcr_quant[:, :, None, None, :, :],
            min=-1024.0,
            max=1016.0,
        )
        transformed_y, transformed_cbcr = transform((y[0], cbcr[0]))
        y_items.append(torch.round(transformed_y).to(torch.int16))
        cbcr_items.append(torch.round(transformed_cbcr).to(torch.int16))
    return torch.stack(y_items, dim=0), torch.stack(cbcr_items, dim=0)


def _compare_tensor(name: str, expected: torch.Tensor, actual: torch.Tensor, tolerance: int) -> dict[str, Any]:
    if tuple(expected.shape) != tuple(actual.shape):
        raise RuntimeError(f"{name} shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}")
    diff = (expected.to(torch.int32) - actual.to(torch.int32)).abs()
    mismatch = diff > tolerance
    return {
        "name": name,
        "comparison_kind": "reference_compare",
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "max_abs": int(diff.max().item()) if diff.numel() else 0,
        "mean_abs": float(diff.to(torch.float32).mean().item()) if diff.numel() else 0.0,
        "mismatch_count": int(mismatch.sum().item()) if mismatch.numel() else 0,
        "element_count": int(diff.numel()),
        "tolerance": int(tolerance),
        "nonzero_ratio": float((actual != 0).sum().item() / actual.numel()) if actual.numel() else 0.0,
    }


def _nonzero_tensor_check(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    abs_tensor = tensor.to(torch.int32).abs()
    nonzero = abs_tensor != 0
    nonzero_count = int(nonzero.sum().item()) if nonzero.numel() else 0
    if nonzero_count == 0 and abs_tensor.numel() != 0:
        raise RuntimeError(f"{name} is unexpectedly all zero")
    return {
        "name": name,
        "comparison_kind": "nonzero_check",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "max_abs": int(abs_tensor.max().item()) if abs_tensor.numel() else 0,
        "mean_abs": float(abs_tensor.to(torch.float32).mean().item()) if abs_tensor.numel() else 0.0,
        "mismatch_count": 0,
        "element_count": int(abs_tensor.numel()),
        "tolerance": 0,
        "expected": "nonzero",
        "nonzero_ratio": float(nonzero_count / abs_tensor.numel()) if abs_tensor.numel() else 0.0,
    }


def _component_mode(reader: Any, image_id: int) -> str:
    components = reader.image_metadata(int(image_id)).get("components", [])
    present_slots = {int(component.get("semantic_slot_id")) for component in components if component.get("present")}
    if 0 in present_slots and 1 in present_slots and 2 in present_slots:
        return "semantic_color"
    present_local = {
        int(component.get("local_component_index"))
        for component in components
        if component.get("present") and int(component.get("local_component_index", -1)) in (0, 1, 2)
    }
    if 0 in present_local and 1 in present_local and 2 in present_local:
        return "fallback_color"
    return "grayscale"


def _sampling_mode(reader: Any, image_id: int) -> str:
    components = reader.image_metadata(int(image_id)).get("components", [])
    by_slot = {
        int(component.get("semantic_slot_id")): component
        for component in components
        if component.get("present") and int(component.get("semantic_slot_id", -1)) in (0, 1, 2)
    }
    if 0 not in by_slot:
        by_local = {
            int(component.get("local_component_index")): component
            for component in components
            if component.get("present") and int(component.get("local_component_index", -1)) in (0, 1, 2)
        }
        by_slot = by_local
    if 0 in by_slot and 1 not in by_slot and 2 not in by_slot:
        return "grayscale"
    if not {0, 1, 2}.issubset(by_slot):
        return "unknown"
    y = by_slot[0]
    cb = by_slot[1]
    cr = by_slot[2]
    if (
        int(cb.get("h_samp_factor", 0)) != int(cr.get("h_samp_factor", 0))
        or int(cb.get("v_samp_factor", 0)) != int(cr.get("v_samp_factor", 0))
    ):
        return "unsupported_mismatched_chroma"
    if (
        int(cb.get("h_samp_factor", 0)) == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) == int(y.get("v_samp_factor", 0))
    ):
        return "4:4:4"
    if (
        int(cb.get("h_samp_factor", 0)) * 2 == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) * 2 == int(y.get("v_samp_factor", 0))
    ):
        return "4:2:0"
    if (
        int(cb.get("h_samp_factor", 0)) * 2 == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) == int(y.get("v_samp_factor", 0))
    ):
        return "4:2:2"
    return "unsupported"


def _sampling_summary(reader: Any, image_ids: list[int]) -> str:
    modes = sorted({_sampling_mode(reader, image_id) for image_id in image_ids})
    return modes[0] if len(modes) == 1 else "mixed:" + ",".join(modes)


def _zero_tensor_check(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    abs_tensor = tensor.to(torch.int32).abs()
    nonzero = abs_tensor != 0
    return {
        "name": name,
        "comparison_kind": "zero_check",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "max_abs": int(abs_tensor.max().item()) if abs_tensor.numel() else 0,
        "mean_abs": float(abs_tensor.to(torch.float32).mean().item()) if abs_tensor.numel() else 0.0,
        "mismatch_count": int(nonzero.sum().item()) if nonzero.numel() else 0,
        "element_count": int(abs_tensor.numel()),
        "tolerance": 0,
        "expected": "all_zero",
        "nonzero_ratio": float(nonzero.sum().item() / abs_tensor.numel()) if abs_tensor.numel() else 0.0,
    }


def _shape_check(name: str, tensor: torch.Tensor, expected_shape_tail: tuple[int, ...]) -> dict[str, Any]:
    if tuple(tensor.shape[1:]) != expected_shape_tail:
        raise RuntimeError(f"{name} shape mismatch: expected tail {expected_shape_tail}, got {tuple(tensor.shape)}")
    return {
        "name": name,
        "comparison_kind": "shape_check",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "max_abs": 0,
        "mean_abs": 0.0,
        "mismatch_count": 0,
        "element_count": int(tensor.numel()),
        "tolerance": 0,
        "expected": "shape_only",
        "nonzero_ratio": float((tensor != 0).sum().item() / tensor.numel()) if tensor.numel() else 0.0,
    }


def _projection_distribution(checks: list[dict[str, Any]]) -> dict[str, Any]:
    if not checks:
        return {
            "sample_count": 0,
            "projection_items_per_image_min": 0.0,
            "projection_items_per_image_max": 0.0,
            "projection_items_per_image_mean": 0.0,
            "materialized_projection_items_per_image_min": 0.0,
            "materialized_projection_items_per_image_max": 0.0,
            "materialized_projection_items_per_image_mean": 0.0,
            "samples": [],
        }
    projection_values = [float(check["projection_items_per_image"]) for check in checks]
    materialized_values = [
        float(check["jpeg_dct_projection_items_materialized_per_image"]) for check in checks
    ]
    return {
        "sample_count": len(checks),
        "projection_items_per_image_min": min(projection_values),
        "projection_items_per_image_max": max(projection_values),
        "projection_items_per_image_mean": sum(projection_values) / len(projection_values),
        "materialized_projection_items_per_image_min": min(materialized_values),
        "materialized_projection_items_per_image_max": max(materialized_values),
        "materialized_projection_items_per_image_mean": sum(materialized_values) / len(materialized_values),
        "samples": [
            {
                "name": check["name"],
                "sampling": check["sampling"],
                "projection_items_per_image": check["projection_items_per_image"],
                "jpeg_dct_projection_items_materialized_per_image": check[
                    "jpeg_dct_projection_items_materialized_per_image"
                ],
            }
            for check in checks
        ],
    }


def _validate_pushdown_batch(name: str, batch: Any, expected_batch_size: int, sampling: str) -> dict[str, Any]:
    if batch.output_layout != "transformed_dct_grid":
        raise RuntimeError(f"{name} unexpected output layout: {batch.output_layout}")
    if batch.y is None or batch.cbcr is None:
        raise RuntimeError(f"{name} expected fixed-grid pushdown to return Y/CbCr tensors")
    if tuple(batch.y.shape) != (expected_batch_size, 1, 28, 28, 8, 8):
        raise RuntimeError(f"{name} unexpected Y shape: {tuple(batch.y.shape)}")
    if tuple(batch.cbcr.shape) != (expected_batch_size, 2, 14, 14, 8, 8):
        raise RuntimeError(f"{name} unexpected CbCr shape: {tuple(batch.cbcr.shape)}")
    if batch.y.dtype != torch.int16 or batch.cbcr.dtype != torch.int16:
        raise RuntimeError(f"{name} expected int16 tensors, got y={batch.y.dtype} cbcr={batch.cbcr.dtype}")
    if not batch.y.is_cuda or not batch.cbcr.is_cuda:
        raise RuntimeError(f"{name} expected CUDA tensors")
    selected_coefficients = list(getattr(batch, "selected_coefficients", []))
    if selected_coefficients != list(range(64)):
        raise RuntimeError(f"{name} expected all 64 source coefficients, got {selected_coefficients}")
    stats = batch.stats
    projection_items = int(stats.get("projection_item_count", 0))
    decoded_projection_items = int(stats.get("decoded_projection_item_count", 0))
    fixed_transform_items = int(stats.get("fixed_transform_item_count", 0))
    fixed_transform_images = int(stats.get("fixed_transform_image_count", 0))
    project_decoded_launches = int(stats.get("project_decoded_ycbcr_grid_launch_count", 0))
    projection_items_materialized = int(stats.get("jpeg_dct_projection_items_materialized", projection_items))
    generic_projection_used = (
        projection_items != 0
        or decoded_projection_items != 0
        or project_decoded_launches != 0
        or projection_items_materialized != 0
    )
    fixed_specialized_path_used = fixed_transform_items > 0 and fixed_transform_images == expected_batch_size
    fallback_reason = ""
    if generic_projection_used:
        fallback_reason = "fixed-grid path used generic projection"
    elif not fixed_specialized_path_used:
        fallback_reason = "fixed-grid transform descriptors missing"
    if generic_projection_used:
        raise RuntimeError(
            f"{name} expected fixed-grid specialized path to bypass generic projection, "
            f"got projection_item_count={projection_items} "
            f"decoded_projection_item_count={decoded_projection_items} "
            f"project_decoded_ycbcr_grid_launch_count={project_decoded_launches} "
            f"jpeg_dct_projection_items_materialized={projection_items_materialized}"
        )
    if fixed_transform_items <= 0 or fixed_transform_images != expected_batch_size:
        raise RuntimeError(
            f"{name} expected fixed-grid transform descriptors, "
            f"got fixed_transform_item_count={fixed_transform_items} fixed_transform_image_count={fixed_transform_images}"
        )
    return {
        "name": name,
        "sampling": sampling,
        "y_shape": list(batch.y.shape),
        "cbcr_shape": list(batch.cbcr.shape),
        "dtype": str(batch.y.dtype),
        "device": str(batch.y.device),
        "y_nonzero_ratio": float((batch.y != 0).sum().item() / batch.y.numel()) if batch.y.numel() else 0.0,
        "cbcr_nonzero_ratio": float((batch.cbcr != 0).sum().item() / batch.cbcr.numel()) if batch.cbcr.numel() else 0.0,
        "flattened_coefficients_per_image": int(batch.coefficients_per_block),
        "selected_coefficients": selected_coefficients,
        "projection_items": projection_items,
        "projection_items_per_image": float(projection_items / expected_batch_size) if expected_batch_size else 0.0,
        "projection_item_count": projection_items,
        "decoded_projection_item_count": decoded_projection_items,
        "generic_projection_used": generic_projection_used,
        "fixed_specialized_path_used": fixed_specialized_path_used,
        "fallback_reason": fallback_reason,
        "fixed_transform_item_count": fixed_transform_items,
        "fixed_transform_image_count": fixed_transform_images,
        "fixed_transform_component_count": int(stats.get("fixed_transform_component_count", 0)),
        "fixed_transform_source_block_count": int(stats.get("fixed_transform_source_block_count", 0)),
        "fixed_transform_output_block_count": int(stats.get("fixed_transform_output_block_count", 0)),
        "projection_item_build_seconds": float(stats.get("projection_item_build_ms", 0.0)) / 1000.0,
        "resize_weight_build_seconds": float(stats.get("resize_weight_build_ms", 0.0)) / 1000.0,
        "dct_resize_weight_cache_hits": int(stats.get("dct_resize_weight_cache_hits", 0)),
        "dct_resize_weight_cache_misses": int(stats.get("dct_resize_weight_cache_misses", 0)),
        "dct_conversion_matrix_cache_hits": int(stats.get("dct_conversion_matrix_cache_hits", 0)),
        "dct_conversion_matrix_cache_misses": int(stats.get("dct_conversion_matrix_cache_misses", 0)),
        "fixed_transform_kernel_seconds": float(stats.get("fixed_transform_ms", 0.0)) / 1000.0,
        "round_kernel_seconds": float(stats.get("fixed_grid_round_ms", 0.0)) / 1000.0,
        "project_decoded_ycbcr_grid_launches": project_decoded_launches,
        "jpeg_dct_projection_items_materialized": projection_items_materialized,
        "jpeg_dct_projection_items_materialized_per_image": (
            float(projection_items_materialized / expected_batch_size) if expected_batch_size else 0.0
        ),
        "selected_vector_count": int(stats.get("selected_vector_count", 0)),
        "full_vector_count": int(stats.get("full_vector_count", 0)),
        "decode_kernel_launch_count": int(stats.get("decode_kernel_launch_count", 0)),
        "cache_capacity_mib": int(stats.get("cache_enabled", False) and batch.cache_stats.get("capacity_bytes", 0) // (1024 * 1024)),
        "cache_active": bool(batch.cache_stats.get("capacity_bytes", 0)),
    }


def _write_synthetic_jpeg(path: Path, mode: str) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("synthetic fixture generation requires Pillow") from exc

    size = 224
    if mode in ("rgb_420", "rgb_422", "rgb_444"):
        image = Image.new("RGB", (size, size))
        pixels = image.load()
        for y in range(size):
            for x in range(size):
                pixels[x, y] = ((x * 3 + y) % 256, (x + y * 2) % 256, (x * 5 + y * 7) % 256)
        subsampling = 0 if mode == "rgb_444" else (1 if mode == "rgb_422" else 2)
        image.save(path, format="JPEG", quality=95, subsampling=subsampling)
        return
    if mode == "grayscale":
        image = Image.new("L", (size, size))
        pixels = image.load()
        for y in range(size):
            for x in range(size):
                pixels[x, y] = (x * 5 + y * 3) % 256
        image.save(path, format="JPEG", quality=95)
        return
    raise ValueError(f"unknown synthetic JPEG mode: {mode}")


def _build_synthetic_manifest(args: argparse.Namespace, fixture_name: str, image_mode: str) -> Path:
    if not args.jpeg_tool.exists():
        raise FileNotFoundError(
            f"{args.jpeg_tool} does not exist; build galp_jpeg_dct_tool or pass --jpeg-tool"
        )
    fixture_root = args.synthetic_fixture_dir / fixture_name
    image_dir = fixture_root / "images"
    out_dir = fixture_root / "dct"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    image_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_jpeg(image_dir / f"{fixture_name}.jpg", image_mode)
    cmd = [
        str(args.jpeg_tool),
        "--shard",
        "--out-dir",
        str(out_dir),
        "--metadata-profile",
        "reconstruct",
        "--shard-images",
        "16",
        "--rowgroup-vectors",
        "16",
        "--rowgroups-per-shard",
        "8",
        "--threads",
        "1",
        str(image_dir),
    ]
    subprocess.run(cmd, check=True)
    manifest = out_dir / "manifest.bin"
    if not manifest.exists():
        raise RuntimeError(f"synthetic fixture manifest was not written: {manifest}")
    return manifest


def _validate_fixture_manifest_sampling(manifest: Path, expected_sampling: str) -> None:
    reader = galp_dct.DirectDctReader(str(manifest))
    if int(reader.image_count) != 1:
        raise RuntimeError(f"{manifest} expected one synthetic image, got {reader.image_count}")
    sampling = _sampling_mode(reader, 0)
    if sampling != expected_sampling:
        raise RuntimeError(f"{manifest} expected sampling {expected_sampling}, got {sampling}")


def _run_synthetic_fixture_checks(args: argparse.Namespace, transform: Any) -> list[dict[str, Any]]:
    fixture_specs = [
        ("synthetic_420", "rgb_420", "4:2:0"),
        ("synthetic_444", "rgb_444", "4:4:4"),
        ("synthetic_grayscale", "grayscale", "grayscale"),
    ]
    fixture_results: list[dict[str, Any]] = []
    for fixture_name, image_mode, expected_sampling in fixture_specs:
        manifest = _build_synthetic_manifest(args, fixture_name, image_mode)
        _validate_fixture_manifest_sampling(manifest, expected_sampling)
        reader = galp_dct.DirectDctReader(str(manifest))
        image_ids = [0]
        tensor_results: list[dict[str, Any]] = []
        pushdown_checks: list[dict[str, Any]] = []
        pushdown = _read_batch(
            reader,
            image_ids,
            None,
            "all",
            args.cache_capacity_mib,
            "transformed_dct_grid",
            RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
        )
        pushdown_checks.append(_validate_pushdown_batch(f"{fixture_name}_pushdown", pushdown, 1, expected_sampling))
        torch.cuda.synchronize()
        if expected_sampling == "grayscale":
            tensor_results.append(_nonzero_tensor_check(f"{fixture_name}_Y", pushdown.y))
            tensor_results.append(_zero_tensor_check(f"{fixture_name}_CbCr", pushdown.cbcr))
        else:
            reference_y, reference_cbcr = _read_dequantized_rgbnomore_reference(
                reader, image_ids, args.cache_capacity_mib, transform
            )
            tensor_results.append(_compare_tensor(f"{fixture_name}_Y", reference_y, pushdown.y, args.tolerance))
            tensor_results.append(
                _compare_tensor(f"{fixture_name}_CbCr", reference_cbcr, pushdown.cbcr, args.tolerance)
            )
        fixture_results.append(
            {
                "step": fixture_name,
                "manifest": str(manifest),
                "image_ids": image_ids,
                "color_image_ids": image_ids if expected_sampling != "grayscale" else [],
                "fallback_color_image_ids": [],
                "grayscale_image_ids": image_ids if expected_sampling == "grayscale" else [],
                "sampling": expected_sampling,
                "reference_layout": "ycbcr_dct_grid" if expected_sampling != "grayscale" else "shape_zero_checks",
                "pushdown_layout": "transformed_dct_grid",
                "pushdown_checks": pushdown_checks,
                "pushdown_stats": pushdown.stats,
                "tensors": tensor_results,
            }
        )
        summary = " ".join(
            f"{result['name']} max_abs={result['max_abs']} mismatches={result['mismatch_count']}"
            for result in tensor_results
        )
        print(f"fixture={fixture_name} sampling={expected_sampling} {summary}")
    unsupported_manifest = _build_synthetic_manifest(args, "synthetic_422", "rgb_422")
    _validate_fixture_manifest_sampling(unsupported_manifest, "4:2:2")
    reader = galp_dct.DirectDctReader(str(unsupported_manifest))
    try:
        reader.plan_batch(
            [0],
            crop=None,
            dct_coeffs="all",
            cache_capacity_mib=args.cache_capacity_mib,
            layout="transformed_dct_grid",
            grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "does not allow this chroma sampling ratio" not in message:
            raise RuntimeError(f"synthetic_422 failed with an unexpected error: {message}") from exc
        fixture_results.append(
            {
                "step": "synthetic_422",
                "manifest": str(unsupported_manifest),
                "image_ids": [0],
                "color_image_ids": [0],
                "fallback_color_image_ids": [],
                "grayscale_image_ids": [],
                "sampling": "4:2:2",
                "reference_layout": "unsupported_sampling",
                "pushdown_layout": "transformed_dct_grid",
                "pushdown_checks": [],
                "pushdown_stats": None,
                "tensors": [],
                "expected_error": message,
            }
        )
        print(f"fixture=synthetic_422 sampling=4:2:2 expected_error={message}")
    else:
        raise RuntimeError("synthetic_422 expected transformed_dct_grid to reject 4:2:2 sampling")
    return fixture_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare transformed_dct_grid/rgbnomore_val runtime output with the Python RGB-no-more DCT crop adapter."
    )
    parser.add_argument("manifest")
    parser.add_argument("--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cache-capacity-mib", type=int, default=1024)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--jpeg-tool", type=Path, default=DEFAULT_JPEG_TOOL)
    parser.add_argument("--synthetic-fixture-dir", type=Path, default=DEFAULT_SYNTHETIC_FIXTURE_DIR)
    parser.add_argument("--skip-synthetic-fixtures", action="store_true")
    parser.add_argument("--output-json", default="/tmp/galp_rgbnomore_pushdown_validation.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("pushdown validation requires torch.cuda.is_available()")

    reader = galp_dct.DirectDctReader(args.manifest)
    transform = _build_rgbnomore_val_crop_transform(args.rgbnomore_root)
    results: list[dict[str, Any]] = []

    for step in range(args.steps):
        image_ids = _make_image_ids(step, args.batch_size, int(reader.image_count))
        modes = {image_id: _component_mode(reader, image_id) for image_id in image_ids}
        sampling_modes = {image_id: _sampling_mode(reader, image_id) for image_id in image_ids}
        supported_color_ids = [
            image_id
            for image_id in image_ids
            if modes[image_id] == "semantic_color" and sampling_modes[image_id] in ("4:2:0", "4:4:4")
        ]
        unsupported_color_ids = [
            image_id
            for image_id in image_ids
            if modes[image_id] == "semantic_color" and sampling_modes[image_id] not in ("4:2:0", "4:4:4")
        ]
        fallback_color_ids = [image_id for image_id in image_ids if modes[image_id] == "fallback_color"]
        grayscale_ids = [image_id for image_id in image_ids if modes[image_id] == "grayscale"]
        tensor_results: list[dict[str, Any]] = []
        pushdown_checks: list[dict[str, Any]] = []
        unsupported_checks: list[dict[str, Any]] = []
        pushdown_stats: dict[str, Any] | None = None
        if supported_color_ids:
            reference_y, reference_cbcr = _read_dequantized_rgbnomore_reference(
                reader, supported_color_ids, args.cache_capacity_mib, transform
            )
            pushdown = _read_batch(
                reader,
                supported_color_ids,
                None,
                "all",
                args.cache_capacity_mib,
                "transformed_dct_grid",
                RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
            )
            pushdown_checks.append(
                _validate_pushdown_batch(
                    "color_pushdown",
                    pushdown,
                    len(supported_color_ids),
                    _sampling_summary(reader, supported_color_ids),
                )
            )
            torch.cuda.synchronize()
            tensor_results.append(_compare_tensor("color_Y", reference_y, pushdown.y, args.tolerance))
            tensor_results.append(_compare_tensor("color_CbCr", reference_cbcr, pushdown.cbcr, args.tolerance))
            pushdown_stats = pushdown.stats
        for image_id in unsupported_color_ids:
            try:
                reader.plan_batch(
                    [image_id],
                    crop=None,
                    dct_coeffs="all",
                    cache_capacity_mib=args.cache_capacity_mib,
                    layout="transformed_dct_grid",
                    grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
                )
            except RuntimeError as exc:
                message = str(exc)
                if "does not allow this chroma sampling ratio" not in message:
                    raise RuntimeError(f"unsupported real image {image_id} failed with an unexpected error: {message}") from exc
                unsupported_checks.append(
                    {
                        "image_id": image_id,
                        "sampling": sampling_modes[image_id],
                        "expected_error": message,
                    }
                )
            else:
                raise RuntimeError(f"unsupported real image {image_id} unexpectedly planned fixed-grid pushdown")
        if grayscale_ids:
            pushdown = _read_batch(
                reader,
                grayscale_ids,
                None,
                "all",
                args.cache_capacity_mib,
                "transformed_dct_grid",
                RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
            )
            pushdown_checks.append(
                _validate_pushdown_batch("grayscale_pushdown", pushdown, len(grayscale_ids), "grayscale")
            )
            torch.cuda.synchronize()
            tensor_results.append(_nonzero_tensor_check("grayscale_Y", pushdown.y))
            tensor_results.append(_zero_tensor_check("grayscale_CbCr", pushdown.cbcr))
            pushdown_stats = pushdown.stats
        if fallback_color_ids:
            pushdown = _read_batch(
                reader,
                fallback_color_ids,
                None,
                "all",
                args.cache_capacity_mib,
                "transformed_dct_grid",
                RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
            )
            pushdown_checks.append(
                _validate_pushdown_batch(
                    "fallback_color_pushdown",
                    pushdown,
                    len(fallback_color_ids),
                    _sampling_summary(reader, fallback_color_ids),
                )
            )
            torch.cuda.synchronize()
            tensor_results.append(_shape_check("fallback_color_Y", pushdown.y, (1, 28, 28, 8, 8)))
            tensor_results.append(_shape_check("fallback_color_CbCr", pushdown.cbcr, (2, 14, 14, 8, 8)))
            pushdown_stats = pushdown.stats
        step_result = {
            "step": step,
            "image_ids": image_ids,
            "color_image_ids": supported_color_ids,
            "unsupported_color_image_ids": unsupported_color_ids,
            "fallback_color_image_ids": fallback_color_ids,
            "grayscale_image_ids": grayscale_ids,
            "sampling_by_image_id": {str(image_id): sampling_modes[image_id] for image_id in image_ids},
            "reference_layout": "ycbcr_dct_grid",
            "pushdown_layout": "transformed_dct_grid",
            "pushdown_checks": pushdown_checks,
            "unsupported_checks": unsupported_checks,
            "pushdown_stats": pushdown_stats,
            "tensors": tensor_results,
        }
        results.append(step_result)
        summary = " ".join(
            f"{result['name']} max_abs={result['max_abs']} mismatches={result['mismatch_count']}"
            for result in tensor_results
        )
        print(
            f"step={step} images={len(image_ids)} color={len(supported_color_ids)} "
            f"unsupported_color={len(unsupported_color_ids)} "
            f"fallback_color={len(fallback_color_ids)} grayscale={len(grayscale_ids)} {summary}"
        )

    if not args.skip_synthetic_fixtures:
        results.extend(_run_synthetic_fixture_checks(args, transform))

    reference_tensors = [
        tensor
        for step in results
        for tensor in step["tensors"]
        if tensor.get("comparison_kind") == "reference_compare"
    ]
    check_tensors = [
        tensor
        for step in results
        for tensor in step["tensors"]
        if tensor.get("comparison_kind") != "reference_compare"
    ]
    total_mismatches = sum(int(tensor["mismatch_count"]) for tensor in reference_tensors)
    max_abs = max((int(tensor["max_abs"]) for tensor in reference_tensors), default=0)
    check_mismatches = sum(int(tensor["mismatch_count"]) for tensor in check_tensors)
    check_max_abs = max((int(tensor["max_abs"]) for tensor in check_tensors), default=0)
    all_checks = [check for step in results for check in step["pushdown_checks"]]
    total_projection_items = sum(int(check["projection_items"]) for check in all_checks)
    total_project_decoded_launches = sum(int(check["project_decoded_ycbcr_grid_launches"]) for check in all_checks)
    generic_projection_used = any(bool(check["generic_projection_used"]) for check in all_checks)
    fixed_specialized_path_used = bool(all_checks) and all(
        bool(check["fixed_specialized_path_used"]) for check in all_checks
    )
    fallback_reasons = sorted({check["fallback_reason"] for check in all_checks if check["fallback_reason"]})
    payload = {
        "manifest": args.manifest,
        "rgbnomore_root": str(args.rgbnomore_root),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "tolerance": args.tolerance,
        "synthetic_fixtures": not args.skip_synthetic_fixtures,
        "synthetic_fixture_dir": str(args.synthetic_fixture_dir),
        "total_mismatch_count": total_mismatches,
        "max_abs": max_abs,
        "check_mismatch_count": check_mismatches,
        "check_max_abs": check_max_abs,
        "projection_items": total_projection_items,
        "projection_items_per_image_distribution": _projection_distribution(all_checks),
        "generic_projection_used": generic_projection_used,
        "fixed_specialized_path_used": fixed_specialized_path_used,
        "fallback_reason": "; ".join(fallback_reasons),
        "project_decoded_ycbcr_grid_launches": total_project_decoded_launches,
        "passed": (
            total_mismatches == 0
            and check_mismatches == 0
            and total_projection_items == 0
            and not generic_projection_used
            and fixed_specialized_path_used
        ),
        "steps_detail": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output_json}")
    if not payload["passed"]:
        reason = payload["fallback_reason"] or "fixed-grid specialized path was not exercised"
        if total_mismatches != 0:
            reason = f"{total_mismatches} values exceeded tolerance {args.tolerance}"
        elif check_mismatches != 0:
            reason = f"{check_mismatches} validation check values failed"
        raise RuntimeError(f"pushdown validation failed: {reason}")


if __name__ == "__main__":
    main()
