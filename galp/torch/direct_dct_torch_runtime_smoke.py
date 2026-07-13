#!/usr/bin/env python3
"""CTest smoke for the optional stay-on-GPU direct-DCT PyTorch runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

import _galp_direct_dct as galp_dct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM


SKIP_RETURN_CODE = 77


def _parse_crop(value: str | None) -> tuple[int, int, int, int]:
    if not value:
        return (0, 0, 64, 64)
    fields = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(fields) != 4:
        raise ValueError("GALP_DIRECT_DCT_TEST_CROP must be formatted as x,y,width,height")
    return tuple(fields)  # type: ignore[return-value]


def main() -> int:
    manifest = os.environ.get("GALP_DIRECT_DCT_TEST_MANIFEST")
    if not manifest:
        print("skipping: GALP_DIRECT_DCT_TEST_MANIFEST is not set")
        return SKIP_RETURN_CODE

    batch_size = int(os.environ.get("GALP_DIRECT_DCT_TEST_BATCH_SIZE", "32"))
    crop = _parse_crop(os.environ.get("GALP_DIRECT_DCT_TEST_CROP"))
    dct_coeffs = os.environ.get("GALP_DIRECT_DCT_TEST_COEFFS", "first:8")

    reader = galp_dct.DirectDctReader(manifest)
    image_ids = list(range(min(batch_size, reader.image_count)))
    fixed_plan = reader.plan_batch(
        image_ids,
        crop=None,
        dct_coeffs="all",
        layout="transformed_dct_grid",
        grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
    )
    if fixed_plan["layout"] != "transformed_dct_grid":
        raise RuntimeError(f"unexpected transformed-grid plan layout: {fixed_plan['layout']}")
    if fixed_plan["image_count"] != len(image_ids):
        raise RuntimeError("transformed-grid plan image count does not match request count")
    if tuple(fixed_plan["y_shape"]) != (len(image_ids), 1, 28, 28, 8, 8):
        raise RuntimeError(f"unexpected transformed-grid Y shape: {tuple(fixed_plan['y_shape'])}")
    if tuple(fixed_plan["cbcr_shape"]) != (len(image_ids), 2, 14, 14, 8, 8):
        raise RuntimeError(f"unexpected transformed-grid CbCr shape: {tuple(fixed_plan['cbcr_shape'])}")
    if len(fixed_plan["selected_coefficients"]) != 64:
        raise RuntimeError("transformed-grid RGB-no-more profile must request all 64 coefficients")
    try:
        reader.plan_batch(
            image_ids,
            crop=None,
            dct_coeffs="first:8",
            layout="transformed_dct_grid",
            grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
        )
    except RuntimeError as exc:
        if "requires dct_coeffs=all" not in str(exc):
            raise
    else:
        raise RuntimeError("transformed-grid RGB-no-more profile unexpectedly accepted sparse coefficients")

    if not torch.cuda.is_available():
        print(
            "fixed_plan=ok "
            f"images={fixed_plan['image_count']} blocks={fixed_plan['block_count']} "
            f"rowgroups={fixed_plan['rowgroup_count']}; skipping CUDA read: torch.cuda.is_available() is false"
        )
        return SKIP_RETURN_CODE

    batch = reader.read_batch(image_ids, crop=crop, dct_coeffs=dct_coeffs)

    coefficients = batch.coefficients
    if not coefficients.is_cuda:
        raise RuntimeError("expected CUDA tensor from direct-DCT runtime")
    if coefficients.dtype != torch.int16:
        raise RuntimeError(f"expected torch.int16 coefficients, got {coefficients.dtype}")
    if tuple(coefficients.shape) != (batch.block_count, batch.coefficients_per_block):
        raise RuntimeError(
            "unexpected tensor shape: "
            f"shape={tuple(coefficients.shape)} expected={(batch.block_count, batch.coefficients_per_block)}"
        )
    if tuple(coefficients.stride()) != (batch.coefficients_per_block, 1):
        raise RuntimeError(f"unexpected tensor stride: {tuple(coefficients.stride())}")
    if coefficients.data_ptr() != batch.device_data_ptr:
        raise RuntimeError(
            "tensor does not wrap GALP CUDA buffer: "
            f"tensor=0x{coefficients.data_ptr():x} galp=0x{batch.device_data_ptr:x}"
        )
    if batch.coefficient_count != coefficients.numel():
        raise RuntimeError("coefficient count does not match tensor numel")
    image_offsets = batch.image_offsets_tensor
    image_counts = batch.image_counts_tensor
    block_to_image = batch.block_to_image_tensor
    for name, metadata_tensor in (
        ("image_offsets_tensor", image_offsets),
        ("image_counts_tensor", image_counts),
        ("block_to_image_tensor", block_to_image),
    ):
        if not metadata_tensor.is_cuda:
            raise RuntimeError(f"{name} is not a CUDA tensor")
        if metadata_tensor.dtype != torch.int64:
            raise RuntimeError(f"{name} must be torch.int64, got {metadata_tensor.dtype}")

    if len(batch.global_image_ids) != len(image_ids) or image_offsets.numel() != len(image_ids):
        raise RuntimeError("image metadata count does not match request count")
    if image_counts.numel() != len(image_ids):
        raise RuntimeError("image count tensor length does not match request count")
    if block_to_image.numel() != batch.block_count:
        raise RuntimeError("block metadata count does not match tensor rows")
    if int(image_counts.sum().item()) != batch.block_count:
        raise RuntimeError("image block counts do not sum to tensor rows")
    if len(batch.selected_coefficients) != batch.coefficients_per_block:
        raise RuntimeError("selected coefficient count does not match tensor columns")

    block_count = batch.block_count
    coefficients_per_block = batch.coefficients_per_block
    tensor_shape = tuple(coefficients.shape)
    device = coefficients.device
    data_ptr = coefficients.data_ptr()
    galp_ptr = batch.device_data_ptr
    stats = batch.execution_stats
    del batch

    features = coefficients.to(torch.float32).mean(dim=1)
    if not features.is_cuda:
        raise RuntimeError("placeholder direct-DCT model input left CUDA")
    del coefficients

    # Exercise the stream-aware external-storage deleter before the queued
    # consumer finishes; the next read must not reuse the pending GALP buffer.
    next_batch = reader.read_batch(image_ids, crop=crop, dct_coeffs=dct_coeffs)
    next_coefficients = next_batch.coefficients
    if not next_coefficients.is_cuda:
        raise RuntimeError("second direct-DCT tensor left CUDA")
    del next_coefficients
    del next_batch

    torch.cuda.synchronize(device)

    print(
        f"images={len(image_ids)} blocks={block_count} "
        f"coefficients_per_block={coefficients_per_block} "
        f"tensor_shape={tensor_shape} feature_shape={tuple(features.shape)} "
        f"device={device} data_ptr=0x{data_ptr:x} galp_ptr=0x{galp_ptr:x} "
        f"selected_vectors={stats['selected_vector_count']} "
        f"full_vectors={stats['full_vector_count']} "
        f"decode_kernels={stats['decode_kernel_launch_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
