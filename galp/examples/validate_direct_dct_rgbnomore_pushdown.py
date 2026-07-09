#!/usr/bin/env python3
"""Validate GALP fixed-grid RGB-no-more pushdown against the Python DCT adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import _galp_direct_dct as galp_dct
from direct_dct_torch_end_to_end_demo import (
    _build_rgbnomore_val_crop_transform,
    _make_image_ids,
    _read_batch,
    _read_rgbnomore_val_crop_batch,
)


def _compare_tensor(name: str, expected: torch.Tensor, actual: torch.Tensor, tolerance: int) -> dict[str, Any]:
    if tuple(expected.shape) != tuple(actual.shape):
        raise RuntimeError(f"{name} shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}")
    diff = (expected.to(torch.int32) - actual.to(torch.int32)).abs()
    mismatch = diff > tolerance
    return {
        "name": name,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "max_abs": int(diff.max().item()) if diff.numel() else 0,
        "mean_abs": float(diff.to(torch.float32).mean().item()) if diff.numel() else 0.0,
        "mismatch_count": int(mismatch.sum().item()) if mismatch.numel() else 0,
        "element_count": int(diff.numel()),
        "tolerance": int(tolerance),
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


def _zero_tensor_check(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    abs_tensor = tensor.to(torch.int32).abs()
    nonzero = abs_tensor != 0
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "max_abs": int(abs_tensor.max().item()) if abs_tensor.numel() else 0,
        "mean_abs": float(abs_tensor.to(torch.float32).mean().item()) if abs_tensor.numel() else 0.0,
        "mismatch_count": int(nonzero.sum().item()) if nonzero.numel() else 0,
        "element_count": int(abs_tensor.numel()),
        "tolerance": 0,
        "expected": "all_zero",
    }


def _shape_check(name: str, tensor: torch.Tensor, expected_shape_tail: tuple[int, ...]) -> dict[str, Any]:
    if tuple(tensor.shape[1:]) != expected_shape_tail:
        raise RuntimeError(f"{name} shape mismatch: expected tail {expected_shape_tail}, got {tuple(tensor.shape)}")
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "max_abs": 0,
        "mean_abs": 0.0,
        "mismatch_count": 0,
        "element_count": int(tensor.numel()),
        "tolerance": 0,
        "expected": "shape_only",
    }


def _validate_pushdown_batch(name: str, batch: Any, expected_batch_size: int) -> dict[str, Any]:
    if batch.output_layout != "ycbcr_dct_grid_fixed":
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
    if projection_items != 0 or decoded_projection_items != 0:
        raise RuntimeError(
            f"{name} expected fixed-grid specialized path to bypass projection items, "
            f"got projection_item_count={projection_items} decoded_projection_item_count={decoded_projection_items}"
        )
    if fixed_transform_items <= 0 or fixed_transform_images != expected_batch_size:
        raise RuntimeError(
            f"{name} expected fixed-grid transform descriptors, "
            f"got fixed_transform_item_count={fixed_transform_items} fixed_transform_image_count={fixed_transform_images}"
        )
    return {
        "name": name,
        "y_shape": list(batch.y.shape),
        "cbcr_shape": list(batch.cbcr.shape),
        "dtype": str(batch.y.dtype),
        "device": str(batch.y.device),
        "flattened_coefficients_per_image": int(batch.coefficients_per_block),
        "selected_coefficients": selected_coefficients,
        "projection_item_count": projection_items,
        "decoded_projection_item_count": decoded_projection_items,
        "fixed_transform_item_count": fixed_transform_items,
        "fixed_transform_image_count": fixed_transform_images,
        "selected_vector_count": int(stats.get("selected_vector_count", 0)),
        "full_vector_count": int(stats.get("full_vector_count", 0)),
        "decode_kernel_launch_count": int(stats.get("decode_kernel_launch_count", 0)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ycbcr_dct_grid_fixed/rgbnomore_val runtime output with the Python RGB-no-more DCT crop adapter."
    )
    parser.add_argument("manifest")
    parser.add_argument("--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cache-capacity-mib", type=int, default=1024)
    parser.add_argument("--tolerance", type=int, default=1)
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
        color_ids = [image_id for image_id in image_ids if modes[image_id] == "semantic_color"]
        fallback_color_ids = [image_id for image_id in image_ids if modes[image_id] == "fallback_color"]
        grayscale_ids = [image_id for image_id in image_ids if modes[image_id] == "grayscale"]
        tensor_results: list[dict[str, Any]] = []
        pushdown_checks: list[dict[str, Any]] = []
        pushdown_stats: dict[str, Any] | None = None
        if color_ids:
            reference = _read_rgbnomore_val_crop_batch(reader, color_ids, args.cache_capacity_mib, transform)
            pushdown = _read_batch(
                reader,
                color_ids,
                None,
                "all",
                args.cache_capacity_mib,
                "ycbcr_dct_grid_fixed",
                "rgbnomore_val",
            )
            if reference.y is None or reference.cbcr is None:
                raise RuntimeError("expected reference validation path to produce Y/CbCr tensors")
            pushdown_checks.append(_validate_pushdown_batch("color_pushdown", pushdown, len(color_ids)))
            torch.cuda.synchronize()
            tensor_results.append(_compare_tensor("color_Y", reference.y, pushdown.y, args.tolerance))
            tensor_results.append(_compare_tensor("color_CbCr", reference.cbcr, pushdown.cbcr, args.tolerance))
            pushdown_stats = pushdown.stats
        if grayscale_ids:
            pushdown = _read_batch(
                reader,
                grayscale_ids,
                None,
                "all",
                args.cache_capacity_mib,
                "ycbcr_dct_grid_fixed",
                "rgbnomore_val",
            )
            pushdown_checks.append(_validate_pushdown_batch("grayscale_pushdown", pushdown, len(grayscale_ids)))
            torch.cuda.synchronize()
            tensor_results.append(_zero_tensor_check("grayscale_CbCr", pushdown.cbcr))
            pushdown_stats = pushdown.stats
        if fallback_color_ids:
            pushdown = _read_batch(
                reader,
                fallback_color_ids,
                None,
                "all",
                args.cache_capacity_mib,
                "ycbcr_dct_grid_fixed",
                "rgbnomore_val",
            )
            pushdown_checks.append(_validate_pushdown_batch("fallback_color_pushdown", pushdown, len(fallback_color_ids)))
            torch.cuda.synchronize()
            tensor_results.append(_shape_check("fallback_color_Y", pushdown.y, (1, 28, 28, 8, 8)))
            tensor_results.append(_shape_check("fallback_color_CbCr", pushdown.cbcr, (2, 14, 14, 8, 8)))
            pushdown_stats = pushdown.stats
        step_result = {
            "step": step,
            "image_ids": image_ids,
            "color_image_ids": color_ids,
            "fallback_color_image_ids": fallback_color_ids,
            "grayscale_image_ids": grayscale_ids,
            "reference_layout": "ycbcr_dct_grid",
            "pushdown_layout": "ycbcr_dct_grid_fixed",
            "pushdown_checks": pushdown_checks,
            "pushdown_stats": pushdown_stats,
            "tensors": tensor_results,
        }
        results.append(step_result)
        summary = " ".join(
            f"{result['name']} max_abs={result['max_abs']} mismatches={result['mismatch_count']}"
            for result in tensor_results
        )
        print(
            f"step={step} images={len(image_ids)} color={len(color_ids)} "
            f"fallback_color={len(fallback_color_ids)} grayscale={len(grayscale_ids)} {summary}"
        )

    total_mismatches = sum(int(tensor["mismatch_count"]) for step in results for tensor in step["tensors"])
    max_abs = max((int(tensor["max_abs"]) for step in results for tensor in step["tensors"]), default=0)
    payload = {
        "manifest": args.manifest,
        "rgbnomore_root": str(args.rgbnomore_root),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "tolerance": args.tolerance,
        "total_mismatch_count": total_mismatches,
        "max_abs": max_abs,
        "passed": total_mismatches == 0,
        "steps_detail": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output_json}")
    if total_mismatches != 0:
        raise RuntimeError(f"pushdown validation failed: {total_mismatches} values exceeded tolerance {args.tolerance}")


if __name__ == "__main__":
    main()
