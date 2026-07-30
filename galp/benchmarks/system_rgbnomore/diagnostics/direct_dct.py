#!/usr/bin/env python3
"""Diagnostic GALP Direct-DCT phase benchmark for RGB-no-more JPEG-Ti.

Published end-to-end comparisons must use ``../inference/run.py``. This script is
retained for loader/kernel/overlap diagnosis and is also imported by the
canonical benchmark's GALP adapter. It requests the
GALP Y/CbCr DCT grid layout, adapts it to RGB-no-more's JPEG-Ti input contract,
optionally dequantizes with JPEG quantization tables from GALP metadata, and
feeds the tensors into RGB-no-more's ViT-Ti DCT model.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TORCH_BINDING_DIR = REPO_ROOT / "build/galp/torch"
if DEFAULT_TORCH_BINDING_DIR.is_dir() and str(DEFAULT_TORCH_BINDING_DIR) not in sys.path:
    # Keep an explicitly configured PYTHONPATH ahead of the in-tree default.
    sys.path.append(str(DEFAULT_TORCH_BINDING_DIR))

import torch

import _galp_direct_dct as galp_dct

TORCH_SOURCE_DIR = REPO_ROOT / "galp/torch"
if str(TORCH_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_SOURCE_DIR))

from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32


DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_DCT_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints" / "imgnetDCTViTTi_ep300_75.1.pth"
SUPPORTED_SAMPLING_MODES = ("4:4:4", "4:2:0", "4:2:2", "4:4:0", "4:1:1", "grayscale", "components:4")


def _make_image_ids(step: int, batch_size: int, image_count: int) -> list[int]:
    if image_count <= 0:
        raise RuntimeError("manifest has no images")
    count = min(batch_size, image_count)
    start = (step * count) % image_count
    return [int((start + index) % image_count) for index in range(count)]


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
    if (
        int(cb.get("h_samp_factor", 0)) == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) * 2 == int(y.get("v_samp_factor", 0))
    ):
        return "4:4:0"
    if (
        int(cb.get("h_samp_factor", 0)) * 4 == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) == int(y.get("v_samp_factor", 0))
    ):
        return "4:1:1"
    return "unsupported"


def _make_benchmark_image_ids(
    reader: Any,
    args: argparse.Namespace,
    step: int,
    *,
    warmup: bool = False,
) -> tuple[list[int], list[dict[str, Any]]]:
    image_count = int(reader.image_count)
    supported_only = args.sampling_policy == "supported-only" or (
        args.sampling_policy == "preprocess-default" and args.preprocess == "rgbnomore-val-pushdown"
    )
    if getattr(args, "image_order", "sequential") == "shuffled":
        population = getattr(args, "_ordered_image_population", None)
        if population is None:
            population = [
                image_id
                for image_id in range(image_count)
                if not supported_only or _sampling_mode(reader, image_id) in SUPPORTED_SAMPLING_MODES
            ]
            random.Random(int(getattr(args, "shuffle_seed", 20260718))).shuffle(population)
            setattr(args, "_ordered_image_population", population)
        if not population:
            raise RuntimeError("shuffled benchmark population is empty")
        count = min(args.batch_size, len(population))
        if warmup:
            start = (step * count) % len(population)
            return [int(population[(start + offset) % len(population)]) for offset in range(count)], []
        start = step * args.batch_size
        if getattr(args, "no_wrap_image_ids", False):
            image_ids = [int(image_id) for image_id in population[start : start + args.batch_size]]
            if not image_ids:
                raise RuntimeError(
                    f"benchmark step {step} exceeds the shuffled image population of {len(population)}"
                )
            return image_ids, []
        return [int(population[(start + offset) % len(population)]) for offset in range(count)], []
    if getattr(args, "no_wrap_image_ids", False):
        if image_count <= 0:
            raise RuntimeError("manifest has no images")
        if supported_only:
            supported_ids = getattr(args, "_supported_image_ids", None)
            if supported_ids is None:
                supported_ids = [
                    image_id
                    for image_id in range(image_count)
                    if _sampling_mode(reader, image_id) in SUPPORTED_SAMPLING_MODES
                ]
                setattr(args, "_supported_image_ids", supported_ids)
            population = supported_ids
        else:
            population = range(image_count)
        if warmup:
            population_size = len(population)
            if population_size == 0:
                raise RuntimeError("non-wrapping benchmark population has no supported images")
            count = min(args.batch_size, population_size)
            start = (step * count) % population_size
            image_ids = [int(population[(start + offset) % population_size]) for offset in range(count)]
            return image_ids, []
        start = step * args.batch_size
        image_ids = [int(image_id) for image_id in population[start : start + args.batch_size]]
        if not image_ids:
            raise RuntimeError(
                f"benchmark step {step} exceeds the non-wrapping image population of {len(population)}"
            )
        return image_ids, []
    if not supported_only:
        return _make_image_ids(step, args.batch_size, image_count), []
    if image_count <= 0:
        raise RuntimeError("manifest has no images")
    count = min(args.batch_size, image_count)
    start = (step * count) % image_count
    image_ids: list[int] = []
    skipped: list[dict[str, Any]] = []
    visited = 0
    offset = 0
    while len(image_ids) < count and visited < image_count:
        image_id = int((start + offset) % image_count)
        offset += 1
        visited += 1
        sampling = _sampling_mode(reader, image_id)
        if sampling in SUPPORTED_SAMPLING_MODES:
            image_ids.append(image_id)
        else:
            skipped.append({"image_id": image_id, "sampling": sampling})
    if len(image_ids) != count:
        raise RuntimeError(
            "benchmark could not assemble a full supported-sampling batch; "
            f"requested={count} supported={len(image_ids)} skipped={skipped[:8]}"
        )
    return image_ids, skipped


def _record_unsupported_sampling_skips(
    records: list[dict[str, Any]],
    step: int,
    skipped: list[dict[str, Any]],
) -> None:
    for item in skipped:
        records.append({"step": int(step), **item})


def _sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_label_index(index_file: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with index_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "Filepath" not in reader.fieldnames or "Label" not in reader.fieldnames:
            raise RuntimeError(f"{index_file} must contain Filepath and Label columns")
        for row in reader:
            filepath = str(row["Filepath"]).replace("\\", "/")
            labels[filepath] = int(row["Label"])
    if not labels:
        raise RuntimeError(f"{index_file} did not contain any image labels")
    return labels


def _load_label_map_json(label_map_json: Path, expected_image_count: int) -> list[int]:
    payload = json.loads(label_map_json.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list) or not all(isinstance(label, int) for label in labels):
        raise RuntimeError(f"{label_map_json} must contain an integer labels list")
    if len(labels) != expected_image_count:
        raise RuntimeError(
            f"{label_map_json} label count {len(labels)} does not match manifest image_count {expected_image_count}"
        )
    return labels


def _imagenet_index_key_from_source(source_path: str) -> str:
    normalized = source_path.replace("\\", "/")
    for marker in ("/train/", "/val/"):
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return normalized[marker_index + 1 :]
    if normalized.startswith("train/") or normalized.startswith("val/"):
        return normalized
    raise RuntimeError(f"cannot derive RGB-no-more index key from GALP source_path: {source_path}")


def _labels_for_image_ids(
    image_ids: list[int],
    labels_by_image_id: list[int],
    device: torch.device,
) -> torch.Tensor:
    labels = []
    for image_id in image_ids:
        if image_id < 0 or image_id >= len(labels_by_image_id):
            raise RuntimeError(f"image_id {image_id} is outside label map length {len(labels_by_image_id)}")
        labels.append(labels_by_image_id[image_id])
    return torch.tensor(labels, dtype=torch.long, device=device)


def _labels_for_image_ids_from_source_path(
    reader: Any,
    image_ids: list[int],
    label_index: dict[str, int] | None,
    device: torch.device,
) -> torch.Tensor:
    if label_index is None:
        raise RuntimeError("source_path label fallback requires a loaded RGB-no-more index")
    labels = []
    for image_id in image_ids:
        metadata = reader.image_metadata(int(image_id))
        source_path = str(metadata.get("source_path", ""))
        key = _imagenet_index_key_from_source(source_path)
        if key not in label_index:
            raise RuntimeError(f"GALP source_path {source_path} maps to {key}, which is missing from the RGB-no-more index")
        labels.append(label_index[key])
    return torch.tensor(labels, dtype=torch.long, device=device)


def _read_grid_batch(
    reader: Any,
    image_ids: list[int],
    crop: tuple[int, int, int, int] | None,
    cache_capacity_mib: int,
    layout: str = "ycbcr_dct_grid",
    grid_transform: dict[str, Any] | None = None,
    decode_batch_rowgroups: int = 2,
    rowgroup_prefetch_depth: int = 16,
    rowgroup_prefetch_workers: int = 4,
    rowgroup_prefetch_min_decode_batches: int = 2,
    plan_cache_capacity: int = 128,
    enable_planless_execution: bool = True,
    scheduling_policy: str = "fully-overlapped",
    transform_blocks_per_launch: int = 0,
    transform_ctas_per_launch: int = 0,
    use_low_priority_streams: bool = False,
    crop_execution_mode: str = "auto",
) -> Any:
    return reader.read_batch(
        image_ids,
        crop=crop,
        dct_coeffs="all",
        cache_capacity_mib=cache_capacity_mib,
        decode_batch_rowgroups=decode_batch_rowgroups,
        rowgroup_prefetch_depth=rowgroup_prefetch_depth,
        rowgroup_prefetch_workers=rowgroup_prefetch_workers,
        rowgroup_prefetch_min_decode_batches=rowgroup_prefetch_min_decode_batches,
        plan_cache_capacity=plan_cache_capacity,
        enable_planless_execution=enable_planless_execution,
        scheduling_policy=scheduling_policy,
        transform_blocks_per_launch=transform_blocks_per_launch,
        transform_ctas_per_launch=transform_ctas_per_launch,
        use_low_priority_streams=use_low_priority_streams,
        crop_execution_mode=crop_execution_mode,
        layout=layout,
        grid_transform=grid_transform,
    )


def _read_compact_batch(
    reader: Any,
    image_ids: list[int],
    crop: tuple[int, int, int, int] | None,
    cache_capacity_mib: int,
) -> Any:
    return reader.read_batch(
        image_ids,
        crop=crop,
        dct_coeffs="all",
        cache_capacity_mib=cache_capacity_mib,
        layout="compact",
    )


def _selected_coefficients_are_all(selected: list[int]) -> bool:
    return len(selected) == 64 and list(selected) == list(range(64))


def _quant_table_by_id(metadata: dict[str, Any]) -> dict[int, torch.Tensor]:
    tables: dict[int, torch.Tensor] = {}
    for table in metadata.get("quant_tables", []):
        values = table.get("values", [])
        if len(values) != 64:
            continue
        tables[int(table["table_id"])] = torch.tensor(values, dtype=torch.float32).reshape(8, 8)
    return tables


def _component_quant_tables(
    reader: Any,
    image_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_tables = []
    cbcr_tables = []
    missing: list[str] = []
    for batch_index, image_id in enumerate(image_ids):
        metadata = reader.image_metadata(int(image_id))
        tables = _quant_table_by_id(metadata)
        component_tables: dict[int, torch.Tensor] = {}
        present_slots: set[int] = set()
        fallback_component_tables: dict[int, torch.Tensor] = {}
        fallback_present_slots: set[int] = set()
        for component in metadata.get("components", []):
            slot = int(component.get("semantic_slot_id", -1))
            local_index = int(component.get("local_component_index", -1))
            if bool(component.get("present")):
                present_slots.add(slot)
                if 0 <= local_index <= 2:
                    fallback_present_slots.add(local_index)
            quant_tbl_no = int(component.get("quant_tbl_no", -1))
            if slot in (0, 1, 2) and quant_tbl_no in tables:
                component_tables[slot] = tables[quant_tbl_no]
            if 0 <= local_index <= 2 and quant_tbl_no in tables:
                fallback_component_tables[local_index] = tables[quant_tbl_no]
        if 0 not in present_slots and 0 in fallback_present_slots:
            present_slots = fallback_present_slots
            component_tables = fallback_component_tables
        if 0 not in component_tables:
            missing.append(f"image_index={image_id} batch_index={batch_index} semantic_slot_id=0")
            continue
        for slot in (1, 2):
            if slot in present_slots and slot not in component_tables:
                missing.append(f"image_index={image_id} batch_index={batch_index} semantic_slot_id={slot}")
        if any(slot in present_slots and slot not in component_tables for slot in (1, 2)):
            continue
        y_tables.append(component_tables[0])
        cb_table = component_tables.get(1, torch.ones((8, 8), dtype=torch.float32))
        cr_table = component_tables.get(2, torch.ones((8, 8), dtype=torch.float32))
        cbcr_tables.append(torch.stack((cb_table, cr_table), dim=0))
    if missing:
        raise RuntimeError(
            "GALP image metadata is missing quantization tables needed to match RGB-no-more dequantized DCT inputs: "
            + "; ".join(missing[:8])
        )
    return torch.stack(y_tables, dim=0).to(device=device), torch.stack(cbcr_tables, dim=0).to(device=device)


def _scale_to_rgbnomore_dct_range(tensor: torch.Tensor) -> torch.Tensor:
    # ToRange(-1024, 1016 -> -1, 1) simplifies to (x + 4) / 1020.
    # `tensor` is a fresh FP32 conversion in the Direct-DCT adapter, so doing
    # the two affine operations in place avoids four full-sized temporaries
    # per Y/CbCr pair while preserving the reference FP32 mapping.
    return tensor.add_(4.0).mul_(1.0 / 1020.0)


def build_rgbnomore_dct_val_transform(rgbnomore_root: Path) -> torch.nn.Module:
    root = str(rgbnomore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    ctrans = importlib.import_module("utils.custom_transforms")
    return torch.nn.Sequential(
        ctrans.ResizedCenterCrop_DCT(32, 28),
        ctrans.ToRange(val_min=-1, val_max=1, orig_min=-1024, orig_max=1016, dtype=torch.float32),
    )


def _validate_rgbnomore_shapes(input_y: torch.Tensor, input_cbcr: torch.Tensor, image_ids: list[int]) -> None:
    if input_y.ndim != 6 or input_cbcr.ndim != 6:
        raise RuntimeError(f"expected rank-6 DCT grids, got y={input_y.ndim} cbcr={input_cbcr.ndim}")
    if tuple(input_y.shape[1:]) != (1, 28, 28, 8, 8):
        raise RuntimeError(f"expected RGB-no-more Y shape tail (1,28,28,8,8), got {tuple(input_y.shape)}")
    if tuple(input_cbcr.shape[1:]) != (2, 14, 14, 8, 8):
        raise RuntimeError(f"expected RGB-no-more CbCr shape tail (2,14,14,8,8), got {tuple(input_cbcr.shape)}")
    if input_y.shape[0] != len(image_ids) or input_cbcr.shape[0] != len(image_ids):
        raise RuntimeError("GALP grid batch size does not match requested image ids")


def adapt_galp_batch_to_rgbnomore(
    reader: Any,
    batch: Any,
    image_ids: list[int],
    *,
    dequantize: bool,
    scale: bool,
    preprocess: str,
    rgbnomore_dct_val_transform: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.layout not in ("ycbcr_dct_grid", "transformed_dct_grid"):
        raise RuntimeError(f"expected GALP Y/CbCr DCT grid layout, got {batch.layout!r}")
    if not _selected_coefficients_are_all(list(batch.selected_coefficients)):
        raise RuntimeError("RGB-no-more adapter requires dct_coeffs='all'; sparse first:N/list:N cannot restore full 8x8 grids")

    input_y = batch.y
    input_cbcr = batch.cbcr
    if input_y.dtype != input_cbcr.dtype or input_y.dtype not in (torch.int16, torch.float32):
        raise RuntimeError(
            "expected matching int16 or float32 GALP DCT grids, "
            f"got y={input_y.dtype} cbcr={input_cbcr.dtype}"
        )

    if input_y.dtype == torch.float32:
        if preprocess != "rgbnomore-val-pushdown" or not dequantize or not scale:
            raise RuntimeError(
                "native float32 GALP grids are already dequantized, rounded, clamped, and range-mapped; "
                "they are only valid for scaled rgbnomore-val-pushdown"
            )
        _validate_rgbnomore_shapes(input_y, input_cbcr, image_ids)
        return input_y, input_cbcr

    y_float = input_y.to(torch.float32)
    cbcr_float = input_cbcr.to(torch.float32)
    if not dequantize and preprocess == "direct-crop":
        _validate_rgbnomore_shapes(y_float, cbcr_float, image_ids)
        return y_float, cbcr_float

    if preprocess == "rgbnomore-val-pushdown":
        if not dequantize:
            raise RuntimeError(
                "rgbnomore-val-pushdown produces dequantized/clamped transformed coefficients; "
                "raw quantized output is not available from this layout"
            )
        # The configured GPU transform dequantizes and clamps every source
        # coefficient before DCT resize, matching RGB-no-more. Applying the
        # source quant table here would dequantize a second time and, more
        # importantly, cannot reproduce dequantize-before-resize semantics.
        if scale:
            y_float = _scale_to_rgbnomore_dct_range(y_float)
            cbcr_float = _scale_to_rgbnomore_dct_range(cbcr_float)
        _validate_rgbnomore_shapes(y_float, cbcr_float, image_ids)
        return y_float, cbcr_float

    if dequantize:
        y_quant, cbcr_quant = _component_quant_tables(reader, image_ids, y_float.device)
        y_float = torch.clamp(y_float * y_quant[:, None, None, None, :, :], min=-1024.0, max=1016.0)
        cbcr_float = torch.clamp(cbcr_float * cbcr_quant[:, :, None, None, :, :], min=-1024.0, max=1016.0)

    if preprocess == "rgbnomore-val":
        if not dequantize:
            raise RuntimeError("rgbnomore-val preprocessing requires dequantized DCT coefficients")
        if rgbnomore_dct_val_transform is None:
            raise RuntimeError("rgbnomore-val preprocessing requires an RGB-no-more DCT validation transform")
        y_items = []
        cbcr_items = []
        for batch_index in range(y_float.shape[0]):
            transformed_y, transformed_cbcr = rgbnomore_dct_val_transform(
                (y_float[batch_index], cbcr_float[batch_index])
            )
            y_items.append(transformed_y)
            cbcr_items.append(transformed_cbcr)
        out_y = torch.stack(y_items, dim=0)
        out_cbcr = torch.stack(cbcr_items, dim=0)
        _validate_rgbnomore_shapes(out_y, out_cbcr, image_ids)
        return out_y, out_cbcr

    if preprocess != "direct-crop":
        raise RuntimeError(f"unknown GALP RGB-no-more DCT preprocessing mode: {preprocess}")
    if scale:
        y_float = _scale_to_rgbnomore_dct_range(y_float)
        cbcr_float = _scale_to_rgbnomore_dct_range(cbcr_float)
    _validate_rgbnomore_shapes(y_float, cbcr_float, image_ids)
    return y_float, cbcr_float


def _import_rgbnomore_model(rgbnomore_root: Path) -> Any:
    root = str(rgbnomore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("models.plainvit")


def build_rgbnomore_jpeg_ti(
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    plainvit = _import_rgbnomore_model(rgbnomore_root)
    model = plainvit.ViT(
        in_channels=3,
        patch_size=16,
        emb_size=192,
        depth=12,
        n_classes=1000,
        drop_p=0.0,
        device=device,
        dtype=torch.float32,
        num_heads=3,
        head_size=64,
        pixel_space="DCT",
        ver=1,
        use_subblock=True,
    )
    checkpoint_obj = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_obj.get("model_state_dict", checkpoint_obj)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def inspect_batch(reader: Any, compact_batch: Any, grid_batch: Any, manifest: str) -> dict[str, Any]:
    image_ids = list(compact_batch.global_image_ids)
    metadata_sample = list(compact_batch.block_metadata[: min(8, len(compact_batch.block_metadata))])
    image_metadata = reader.image_metadata(int(image_ids[0])) if image_ids else {}
    first_components = image_metadata.get("components", [])
    first_quant_tables = image_metadata.get("quant_tables", [])
    coefficients = compact_batch.coefficients
    input_y = grid_batch.y
    input_cbcr = grid_batch.cbcr
    result = {
        "backend": "galp_direct_dct",
        "phase": "inspect",
        "manifest": manifest,
        "dataset_size": int(reader.image_count),
        "compact_layout": compact_batch.layout,
        "grid_layout": grid_batch.layout,
        "image_ids": image_ids,
        "coefficients_shape": list(coefficients.shape),
        "coefficients_dtype": str(coefficients.dtype),
        "coefficients_device": str(coefficients.device),
        "y_shape": list(input_y.shape),
        "cbcr_shape": list(input_cbcr.shape),
        "selected_coefficients": list(compact_batch.selected_coefficients),
        "block_metadata_sample": metadata_sample,
        "metadata_fields_present": {
            "image_index": all("global_image_index" in item and "request_index" in item for item in metadata_sample),
            "component_channel": all("semantic_slot_id" in item for item in metadata_sample),
            "block_row_col": all("block_y" in item and "block_x" in item for item in metadata_sample),
            "coefficient_u_v": bool(_selected_coefficients_are_all(list(compact_batch.selected_coefficients))),
            "quantization_table": bool(first_quant_tables),
            "dequantized": False,
        },
        "rgbnomore_shape_compatible": {
            "input_y": tuple(input_y.shape[1:]) == (1, 28, 28, 8, 8),
            "input_cbcr": tuple(input_cbcr.shape[1:]) == (2, 14, 14, 8, 8),
        },
        "first_image_component_sample": first_components[:3],
        "first_image_quant_table_ids": [int(table["table_id"]) for table in first_quant_tables],
        "compact_execution_stats": dict(compact_batch.execution_stats),
        "grid_execution_stats": dict(grid_batch.execution_stats),
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def inspect_metadata_only(reader: Any, args: argparse.Namespace) -> dict[str, Any]:
    image_count = min(args.batch_size, int(reader.image_count))
    image_ids = list(range(image_count))
    image_summaries = []
    has_all_quant_tables = True
    for image_id in image_ids:
        metadata = reader.image_metadata(int(image_id))
        quant_tables = metadata.get("quant_tables", [])
        components = metadata.get("components", [])
        component_summary = [
            {
                "semantic_slot_id": int(component.get("semantic_slot_id", -1)),
                "width_in_blocks": int(component.get("width_in_blocks", 0)),
                "height_in_blocks": int(component.get("height_in_blocks", 0)),
                "h_samp_factor": int(component.get("h_samp_factor", 0)),
                "v_samp_factor": int(component.get("v_samp_factor", 0)),
                "quant_tbl_no": int(component.get("quant_tbl_no", -1)),
            }
            for component in components
        ]
        table_ids = {int(table["table_id"]) for table in quant_tables}
        component_quant_ok = all(
            item["semantic_slot_id"] not in (0, 1, 2)
            or (item["quant_tbl_no"] >= 0 and item["quant_tbl_no"] in table_ids)
            for item in component_summary
        )
        has_all_quant_tables = has_all_quant_tables and component_quant_ok
        image_summaries.append(
            {
                "image_id": image_id,
                "image_width": int(metadata.get("image_width", 0)),
                "image_height": int(metadata.get("image_height", 0)),
                "components": component_summary,
                "quant_table_ids": sorted(table_ids),
                "component_quant_tables_available": component_quant_ok,
            }
        )
    result = {
        "backend": "galp_direct_dct",
        "phase": "metadata",
        "manifest": str(args.manifest),
        "dataset_size": int(reader.image_count),
        "manifest_image_count": int(reader.image_count),
        "inspected_image_ids": image_ids,
        "component_quant_tables_available": has_all_quant_tables,
        "rgbnomore_expected_shapes": {
            "input_y": [image_count, 1, 28, 28, 8, 8],
            "input_cbcr": [image_count, 2, 14, 14, 8, 8],
        },
        "images": image_summaries,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def _uses_planless_fixed_transform(stats: dict[str, Any]) -> bool:
    return (
        int(stats.get("planless_image_descriptor_count", 0)) > 0
        and int(stats.get("fixed_transform_item_count", 0)) == 0
        and int(stats.get("host_expanded_transform_items_created", 0)) == 0
        and int(stats.get("host_output_block_source_lists_created", 0)) == 0
        and int(stats.get("host_global_transform_sort_items", 0)) == 0
        and bool(stats.get("device_mapping_fused", False))
    )


def _uses_fixed_transform(stats: dict[str, Any]) -> bool:
    return _uses_planless_fixed_transform(stats) or int(stats.get("fixed_transform_item_count", 0)) > 0


def _accumulate_stats(totals: dict[str, int | float | str], batch: Any) -> None:
    stats = batch.execution_stats
    totals["selected_vectors"] += int(stats["selected_vector_count"])
    totals["full_vectors"] += int(stats["full_vector_count"])
    totals["planned_vector_count"] += int(
        stats.get("planned_vector_count", stats.get("planned_selected_vector_count", 0))
    )
    totals["actual_vector_count"] += int(
        stats.get("actual_vector_count", stats.get("selected_vector_count", 0))
    )
    totals["full_vector_count"] += int(stats.get("full_vector_count", 0))
    totals["decode_kernels"] += int(stats["decode_kernel_launch_count"])
    totals["rowgroups"] += int(stats["rowgroup_count"])
    totals["worksets"] += int(stats["workset_count"])
    totals["projection_items"] += int(stats["projection_item_count"])
    totals["decoded_projection_items"] += int(stats.get("decoded_projection_item_count", 0))
    totals["fixed_transform_items"] += int(stats.get("fixed_transform_item_count", 0))
    totals["fixed_grid_finalize_kernel_launches"] += int(
        stats.get("fixed_grid_finalize_kernel_launch_count", 0)
    )
    totals["fixed_grid_float32_output_batches"] += int(
        bool(stats.get("fixed_grid_output_float32", False))
    )
    totals["fixed_grid_affine_applied_batches"] += int(
        bool(stats.get("fixed_grid_output_affine_applied", False))
    )
    # Planless execution submits one compact image descriptor and creates no
    # host-side transform items, source lists, or sort entries.
    totals["planless_image_descriptors"] += int(stats.get("planless_image_descriptor_count", 0))
    totals["planless_transform_output_blocks"] += int(stats.get("planless_transform_output_block_count", 0))
    totals["planless_transform_kernel_launches"] += int(
        stats.get("planless_transform_kernel_launch_count", 0)
    )
    totals["planless_transform_max_blocks_per_launch"] = max(
        int(totals["planless_transform_max_blocks_per_launch"]),
        int(stats.get("planless_transform_max_blocks_per_launch", 0)),
    )
    totals["planless_transform_max_output_blocks_per_launch"] = max(
        int(totals["planless_transform_max_output_blocks_per_launch"]),
        int(stats.get("planless_transform_max_output_blocks_per_launch", 0)),
    )
    for key in (
        "planless_transform_registers_per_thread",
        "planless_transform_static_shared_bytes_per_cta",
        "planless_transform_local_bytes_per_thread",
        "planless_transform_threads_per_cta",
        "planless_transform_max_active_ctas_per_sm",
        "cuda_max_threads_per_sm",
        "cuda_warp_size",
    ):
        totals[key] = max(int(totals[key]), int(stats.get(key, 0)))
    totals["decode_to_transform_event_handoffs"] += int(
        stats.get("decode_to_transform_event_handoff_count", 0)
    )
    totals["copy_to_decode_event_handoffs"] += int(
        stats.get("copy_to_decode_event_handoff_count", 0)
    )
    totals["direct_dct_low_priority_batches"] += int(
        bool(stats.get("direct_dct_low_priority_streams", False))
    )
    for key in (
        "direct_dct_stream_priority",
        "direct_dct_h2d_stream_priority",
        "direct_dct_decode_stream_priority",
        "direct_dct_transform_stream_priority",
        "direct_dct_round_stream_priority",
        "cuda_least_stream_priority",
        "cuda_greatest_stream_priority",
    ):
        totals[key] = int(stats.get(key, 0))
    for key in (
        "planless_axis_program_count",
        "planless_axis_phase_matrix_count",
        "planless_axis_program_bytes",
    ):
        totals[key] = max(int(totals[key]), int(stats.get(key, 0)))
    totals["rowgroup_storage_bytes_read"] += int(stats.get("rowgroup_storage_bytes_read", 0))
    totals["compressed_payload_bytes_read"] += int(
        stats.get("compressed_payload_bytes_read", stats.get("rowgroup_storage_bytes_read", 0))
    )
    totals["full_compressed_payload_bytes"] += int(
        stats.get("full_compressed_payload_bytes", stats.get("rowgroup_storage_bytes_read", 0))
    )
    totals["pread_count"] += int(stats.get("pread_count", 0))
    totals["vector_bundle_rowgroup_count"] += int(stats.get("vector_bundle_rowgroup_count", 0))
    totals["vector_bundle_envelope_rowgroup_count"] += int(
        stats.get("vector_bundle_envelope_rowgroup_count", 0)
    )
    totals["vector_bundle_pread_count"] += int(stats.get("vector_bundle_pread_count", 0))
    totals["pinned_rowgroup_read_count"] += int(stats.get("pinned_rowgroup_read_count", 0))
    totals["pinned_rowgroup_read_bytes"] += int(stats.get("pinned_rowgroup_read_bytes", 0))
    totals["requested_source_block_count"] += int(
        stats.get("requested_source_block_count", stats.get("fixed_transform_source_block_count", 0))
    )
    totals["source_blocks_transformed"] += int(
        stats.get("source_blocks_transformed", stats.get("fixed_transform_source_block_count", 0))
    )
    totals["sparse_read_supported_batches"] += int(bool(stats.get("sparse_read_supported", False)))
    totals["sparse_read_fallback_rowgroup_count"] += int(
        stats.get("sparse_read_fallback_rowgroup_count", 0)
    )
    for key in (
        "automatic_sparse_storage_candidate_rowgroup_count",
        "automatic_sparse_storage_selected_rowgroup_count",
        "automatic_sparse_storage_rejected_rowgroup_count",
        "automatic_sparse_storage_full_bytes",
        "automatic_sparse_storage_candidate_bytes",
        "automatic_sparse_storage_candidate_pread_count",
    ):
        totals[key] += int(stats.get(key, 0))
    for key in (
        "automatic_sparse_storage_full_estimated_ns",
        "automatic_sparse_storage_candidate_estimated_ns",
    ):
        totals[key] += float(stats.get(key, 0.0))
    for key, default in (
        ("storage_read_granularity", "rowgroup"),
        ("decode_granularity", "rowgroup"),
        ("sparse_read_fallback_reason", ""),
    ):
        value = str(stats.get(key, default))
        previous = str(totals.get(key, ""))
        if not previous:
            totals[key] = value
        elif value and value != previous:
            totals[key] = "mixed" if key != "sparse_read_fallback_reason" else "; ".join(
                sorted(set(filter(None, previous.split("; ") + [value])))
            )
    for key in (
        "galp_native_device_in_use_bytes",
        "galp_native_device_peak_in_use_bytes",
        "galp_native_device_cached_bytes",
        "galp_native_device_allocation_requests",
        "galp_native_device_cuda_allocation_count",
        "galp_native_device_cuda_allocation_bytes",
        "galp_native_pinned_in_use_bytes",
        "galp_native_pinned_peak_in_use_bytes",
        "galp_native_pinned_cached_bytes",
        "galp_native_pinned_allocation_requests",
        "galp_native_pinned_cuda_allocation_count",
        "galp_native_pinned_cuda_allocation_bytes",
    ):
        totals[key] = max(int(totals[key]), int(stats.get(key, 0)))
    totals["host_expanded_transform_items_created"] += int(
        stats.get("host_expanded_transform_items_created", 0)
    )
    totals["host_output_block_source_lists_created"] += int(
        stats.get("host_output_block_source_lists_created", 0)
    )
    totals["host_global_transform_sort_items"] += int(stats.get("host_global_transform_sort_items", 0))
    totals["device_mapping_fused_batches"] += int(bool(stats.get("device_mapping_fused", False)))
    totals["fixed_transform_components"] += int(stats.get("fixed_transform_component_count", 0))
    totals["fixed_transform_source_blocks"] += int(stats.get("fixed_transform_source_block_count", 0))
    totals["fixed_transform_output_blocks"] += int(stats.get("fixed_transform_output_block_count", 0))
    totals["dct_resize_weight_cache_hits"] += int(stats.get("dct_resize_weight_cache_hits", 0))
    totals["dct_resize_weight_cache_misses"] += int(stats.get("dct_resize_weight_cache_misses", 0))
    totals["dct_conversion_matrix_cache_hits"] += int(stats.get("dct_conversion_matrix_cache_hits", 0))
    totals["dct_conversion_matrix_cache_misses"] += int(stats.get("dct_conversion_matrix_cache_misses", 0))
    totals["plan_cache_hits"] += int(stats.get("plan_cache_hits", 0))
    totals["plan_cache_misses"] += int(stats.get("plan_cache_misses", 0))
    totals["plan_cache_evictions"] += int(stats.get("plan_cache_evictions", 0))
    totals["exact_batch_plan_cache_enabled_batches"] += int(
        bool(stats.get("exact_batch_plan_cache_enabled", False))
    )
    totals["decoded_rowgroup_cache_enabled_batches"] += int(bool(stats.get("cache_enabled", False)))
    totals["project_decoded_ycbcr_grid_launches"] += int(stats.get("project_decoded_ycbcr_grid_launch_count", 0))
    totals["jpeg_dct_projection_items_materialized"] += int(
        stats.get("jpeg_dct_projection_items_materialized", stats["projection_item_count"])
    )
    totals["internal_syncs"] += int(stats["internal_sync_count"])
    totals["planning_seconds"] += float(stats.get("planning_ms", 0.0)) / 1000.0
    totals["workset_build_seconds"] += float(stats.get("workset_build_ms", 0.0)) / 1000.0
    totals["workset_upload_seconds"] += float(stats.get("workset_upload_ms", 0.0)) / 1000.0
    totals["workset_upload_arena_pack_seconds"] += (
        float(stats.get("workset_upload_arena_pack_ms", 0.0)) / 1000.0
    )
    totals["workset_upload_dma_bytes"] += int(stats.get("workset_upload_dma_bytes", 0))
    totals["workset_upload_dma_count"] += int(stats.get("workset_upload_dma_count", 0))
    totals["decode_seconds"] += float(stats.get("decode_ms", 0.0)) / 1000.0
    totals["gather_seconds"] += float(stats.get("gather_ms", 0.0)) / 1000.0
    totals["decoded_gather_seconds"] += float(stats.get("decoded_gather_ms", 0.0)) / 1000.0
    totals["prefetch_wait_seconds"] += float(stats.get("prefetch_wait_ms", 0.0)) / 1000.0
    totals["prefetch_rowgroup_read_seconds"] += float(stats.get("prefetch_rowgroup_read_ms", 0.0)) / 1000.0
    totals["sync_rowgroup_read_seconds"] += float(stats.get("sync_rowgroup_read_ms", 0.0)) / 1000.0
    projection_item_build_seconds = float(stats.get("projection_item_build_ms", 0.0)) / 1000.0
    totals["projection_build_seconds"] += projection_item_build_seconds
    totals["projection_item_build_seconds"] += projection_item_build_seconds
    totals["resize_weight_build_seconds"] += float(stats.get("resize_weight_build_ms", 0.0)) / 1000.0
    totals["gpu_projection_seconds"] += float(stats.get("projection_ms", 0.0)) / 1000.0
    totals["decoded_projection_seconds"] += float(stats.get("decoded_projection_ms", 0.0)) / 1000.0
    totals["fixed_transform_kernel_seconds"] += float(stats.get("fixed_transform_ms", 0.0)) / 1000.0
    totals["device_mapping_seconds"] += float(stats.get("device_mapping_ms", 0.0)) / 1000.0
    totals["round_kernel_seconds"] += float(stats.get("fixed_grid_round_ms", 0.0)) / 1000.0


def _empty_totals() -> dict[str, int | float | str]:
    return {
        "selected_vectors": 0,
        "full_vectors": 0,
        "planned_vector_count": 0,
        "actual_vector_count": 0,
        "full_vector_count": 0,
        "decode_kernels": 0,
        "rowgroups": 0,
        "worksets": 0,
        "projection_items": 0,
        "decoded_projection_items": 0,
        "fixed_transform_items": 0,
        "fixed_grid_finalize_kernel_launches": 0,
        "fixed_grid_float32_output_batches": 0,
        "fixed_grid_affine_applied_batches": 0,
        "planless_image_descriptors": 0,
        "planless_transform_output_blocks": 0,
        "planless_transform_kernel_launches": 0,
        "planless_transform_max_blocks_per_launch": 0,
        "planless_transform_max_output_blocks_per_launch": 0,
        "planless_transform_registers_per_thread": 0,
        "planless_transform_static_shared_bytes_per_cta": 0,
        "planless_transform_local_bytes_per_thread": 0,
        "planless_transform_threads_per_cta": 0,
        "planless_transform_max_active_ctas_per_sm": 0,
        "cuda_max_threads_per_sm": 0,
        "cuda_warp_size": 0,
        "decode_to_transform_event_handoffs": 0,
        "copy_to_decode_event_handoffs": 0,
        "direct_dct_low_priority_batches": 0,
        "direct_dct_stream_priority": 0,
        "direct_dct_h2d_stream_priority": 0,
        "direct_dct_decode_stream_priority": 0,
        "direct_dct_transform_stream_priority": 0,
        "direct_dct_round_stream_priority": 0,
        "cuda_least_stream_priority": 0,
        "cuda_greatest_stream_priority": 0,
        "planless_axis_program_count": 0,
        "planless_axis_phase_matrix_count": 0,
        "planless_axis_program_bytes": 0,
        "rowgroup_storage_bytes_read": 0,
        "compressed_payload_bytes_read": 0,
        "full_compressed_payload_bytes": 0,
        "pread_count": 0,
        "vector_bundle_rowgroup_count": 0,
        "vector_bundle_envelope_rowgroup_count": 0,
        "vector_bundle_pread_count": 0,
        "pinned_rowgroup_read_count": 0,
        "pinned_rowgroup_read_bytes": 0,
        "requested_source_block_count": 0,
        "source_blocks_transformed": 0,
        "sparse_read_supported_batches": 0,
        "sparse_read_fallback_rowgroup_count": 0,
        "automatic_sparse_storage_candidate_rowgroup_count": 0,
        "automatic_sparse_storage_selected_rowgroup_count": 0,
        "automatic_sparse_storage_rejected_rowgroup_count": 0,
        "automatic_sparse_storage_full_bytes": 0,
        "automatic_sparse_storage_candidate_bytes": 0,
        "automatic_sparse_storage_candidate_pread_count": 0,
        "automatic_sparse_storage_full_estimated_ns": 0.0,
        "automatic_sparse_storage_candidate_estimated_ns": 0.0,
        "storage_read_granularity": "",
        "decode_granularity": "",
        "sparse_read_fallback_reason": "",
        "galp_native_device_in_use_bytes": 0,
        "galp_native_device_peak_in_use_bytes": 0,
        "galp_native_device_cached_bytes": 0,
        "galp_native_device_allocation_requests": 0,
        "galp_native_device_cuda_allocation_count": 0,
        "galp_native_device_cuda_allocation_bytes": 0,
        "galp_native_pinned_in_use_bytes": 0,
        "galp_native_pinned_peak_in_use_bytes": 0,
        "galp_native_pinned_cached_bytes": 0,
        "galp_native_pinned_allocation_requests": 0,
        "galp_native_pinned_cuda_allocation_count": 0,
        "galp_native_pinned_cuda_allocation_bytes": 0,
        "host_expanded_transform_items_created": 0,
        "host_output_block_source_lists_created": 0,
        "host_global_transform_sort_items": 0,
        "device_mapping_fused_batches": 0,
        "fixed_transform_components": 0,
        "fixed_transform_source_blocks": 0,
        "fixed_transform_output_blocks": 0,
        "dct_resize_weight_cache_hits": 0,
        "dct_resize_weight_cache_misses": 0,
        "dct_conversion_matrix_cache_hits": 0,
        "dct_conversion_matrix_cache_misses": 0,
        "plan_cache_hits": 0,
        "plan_cache_misses": 0,
        "plan_cache_evictions": 0,
        "exact_batch_plan_cache_enabled_batches": 0,
        "decoded_rowgroup_cache_enabled_batches": 0,
        "project_decoded_ycbcr_grid_launches": 0,
        "jpeg_dct_projection_items_materialized": 0,
        "internal_syncs": 0,
        "planning_seconds": 0.0,
        "workset_build_seconds": 0.0,
        "workset_upload_seconds": 0.0,
        "workset_upload_arena_pack_seconds": 0.0,
        "workset_upload_dma_bytes": 0,
        "workset_upload_dma_count": 0,
        "decode_seconds": 0.0,
        "gather_seconds": 0.0,
        "decoded_gather_seconds": 0.0,
        "prefetch_wait_seconds": 0.0,
        "prefetch_rowgroup_read_seconds": 0.0,
        "sync_rowgroup_read_seconds": 0.0,
        "projection_build_seconds": 0.0,
        "projection_item_build_seconds": 0.0,
        "resize_weight_build_seconds": 0.0,
        "gpu_projection_seconds": 0.0,
        "decoded_projection_seconds": 0.0,
        "fixed_transform_kernel_seconds": 0.0,
        "device_mapping_seconds": 0.0,
        "round_kernel_seconds": 0.0,
    }


def validate_crop_pushdown_accounting(stats: dict[str, Any]) -> list[str]:
    """Reject crop-pushdown claims that are not supported by physical counters."""
    failures: list[str] = []
    planned = int(stats.get("planned_vector_count", stats.get("planned_selected_vector_count", 0)))
    actual = int(stats.get("actual_vector_count", stats.get("selected_vector_count", 0)))
    full = int(stats.get("full_vector_count", stats.get("full_vectors", 0)))
    physical_bytes = int(
        stats.get("compressed_payload_bytes_read", stats.get("rowgroup_storage_bytes_read", 0))
    )
    full_bytes = int(stats.get("full_compressed_payload_bytes", physical_bytes))
    decode_granularity = str(stats.get("decode_granularity", "rowgroup"))
    storage_granularity = str(stats.get("storage_read_granularity", "rowgroup"))
    claims_vector_pushdown = decode_granularity in {"selected-vector", "vector"}
    claims_lower_io = storage_granularity not in {"", "rowgroup", "none"}

    if claims_vector_pushdown and full > 0 and (planned >= full or actual >= full):
        failures.append(
            "vector pushdown claimed but selected/decoded vectors cover the full rowgroup"
        )
    if claims_lower_io and full_bytes > 0 and physical_bytes >= full_bytes:
        failures.append(
            "lower physical I/O claimed but compressed payload bytes equal the full-rowgroup baseline"
        )
    if (
        claims_lower_io
        and int(stats.get("source_blocks_transformed", 0)) > 0
        and full_bytes > 0
        and physical_bytes >= full_bytes
    ):
        failures.append(
            "transform block reduction was reported as storage read reduction without fewer physical bytes"
        )
    return failures


def _annotate_fixed_path_result(result: dict[str, Any], preprocess: str) -> None:
    result["planned_vector_count"] = int(
        result.get("planned_vector_count", result.get("planned_selected_vector_count", 0))
    )
    result["actual_vector_count"] = int(
        result.get("actual_vector_count", result.get("selected_vectors", 0))
    )
    result["full_vector_count"] = int(result.get("full_vector_count", result.get("full_vectors", 0)))
    result["compressed_payload_bytes_read"] = int(
        result.get("compressed_payload_bytes_read", result.get("rowgroup_storage_bytes_read", 0))
    )
    result["full_compressed_payload_bytes"] = int(
        result.get("full_compressed_payload_bytes", result["compressed_payload_bytes_read"])
    )
    result["pread_count"] = int(result.get("pread_count", 0))
    result["vector_bundle_rowgroup_count"] = int(result.get("vector_bundle_rowgroup_count", 0))
    result["vector_bundle_envelope_rowgroup_count"] = int(
        result.get("vector_bundle_envelope_rowgroup_count", 0)
    )
    result["vector_bundle_pread_count"] = int(result.get("vector_bundle_pread_count", 0))
    result["storage_read_granularity"] = str(result.get("storage_read_granularity") or "rowgroup")
    result["decode_granularity"] = str(result.get("decode_granularity") or "rowgroup")
    result["source_blocks_transformed"] = int(
        result.get("source_blocks_transformed", result.get("fixed_transform_source_blocks", 0))
    )
    full_bytes = result["full_compressed_payload_bytes"]
    result["read_amplification"] = (
        result["compressed_payload_bytes_read"] / full_bytes if full_bytes > 0 else 0.0
    )
    result["sparse_read_supported"] = bool(result.get("sparse_read_supported_batches", 0))
    accounting_failures = validate_crop_pushdown_accounting(result)
    result["crop_pushdown_accounting"] = {
        "ok": not accounting_failures,
        "failures": accounting_failures,
        "stage_chain": [
            "requested_source_blocks",
            "planned_selected_vectors",
            "actual_decoded_vectors",
            "physical_compressed_bytes_read",
        ],
    }
    if accounting_failures:
        raise RuntimeError("invalid crop-pushdown accounting: " + "; ".join(accounting_failures))
    result["generic_projection_used"] = (
        int(result.get("projection_items", 0)) != 0
        or int(result.get("decoded_projection_items", 0)) != 0
        or int(result.get("project_decoded_ycbcr_grid_launches", 0)) != 0
        or int(result.get("jpeg_dct_projection_items_materialized", 0)) != 0
    )
    result["planless_path_used"] = (
        preprocess == "rgbnomore-val-pushdown"
        and int(result.get("planless_image_descriptors", 0)) > 0
        and int(result.get("fixed_transform_items", 0)) == 0
        and int(result.get("host_expanded_transform_items_created", 0)) == 0
        and int(result.get("host_output_block_source_lists_created", 0)) == 0
        and int(result.get("host_global_transform_sort_items", 0)) == 0
        and int(result.get("exact_batch_plan_cache_enabled_batches", 0)) == 0
        and int(result.get("decoded_rowgroup_cache_enabled_batches", 0)) == 0
    )
    result["fixed_specialized_path_used"] = result["planless_path_used"] or (
        preprocess == "rgbnomore-val-pushdown" and int(result.get("fixed_transform_items", 0)) > 0
    )
    result["fallback_reason"] = "fixed-grid path used generic projection" if result["generic_projection_used"] else ""
    cache_records = [
        cache
        for batch in result.get("execution_path_batches", [])
        for cache in batch.get("cache_stats", [])
    ]
    result["cache_active"] = any(int(cache.get("capacity_bytes", 0)) > 0 for cache in cache_records)
    result["cache_disabled_reason"] = "" if result["cache_active"] else "cache_capacity_mib=0"
    if preprocess == "rgbnomore-val-pushdown" and result["generic_projection_used"]:
        raise RuntimeError(
            "RGBNoMore fixed-grid default path used generic projection: "
            f"projection_items={result.get('projection_items', 0)} "
            f"project_decoded_ycbcr_grid_launches={result.get('project_decoded_ycbcr_grid_launches', 0)}"
        )


def _accumulate_many_stats(totals: dict[str, int | float | str], batches: list[Any]) -> None:
    for batch in batches:
        _accumulate_stats(totals, batch)


def _projection_counts_for_batches(batches: list[Any]) -> tuple[int, int]:
    projection_items = 0
    materialized_items = 0
    for batch in batches:
        stats = batch.execution_stats
        projection_items += int(stats.get("projection_item_count", 0))
        materialized_items += int(
            stats.get("jpeg_dct_projection_items_materialized", stats.get("projection_item_count", 0))
        )
    return projection_items, materialized_items


def _record_projection_distribution(samples: list[dict[str, float | int]], batches: list[Any], image_count: int) -> None:
    if image_count <= 0:
        return
    projection_items, materialized_items = _projection_counts_for_batches(batches)
    samples.append(
        {
            "images": int(image_count),
            "projection_items": int(projection_items),
            "projection_items_per_image": float(projection_items / image_count),
            "jpeg_dct_projection_items_materialized": int(materialized_items),
            "jpeg_dct_projection_items_materialized_per_image": float(materialized_items / image_count),
        }
    )


def _summarize_projection_distribution(samples: list[dict[str, float | int]]) -> dict[str, Any]:
    if not samples:
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
    projection_values = [float(sample["projection_items_per_image"]) for sample in samples]
    materialized_values = [
        float(sample["jpeg_dct_projection_items_materialized_per_image"]) for sample in samples
    ]
    return {
        "sample_count": len(samples),
        "projection_items_per_image_min": min(projection_values),
        "projection_items_per_image_max": max(projection_values),
        "projection_items_per_image_mean": sum(projection_values) / len(projection_values),
        "materialized_projection_items_per_image_min": min(materialized_values),
        "materialized_projection_items_per_image_max": max(materialized_values),
        "materialized_projection_items_per_image_mean": sum(materialized_values) / len(materialized_values),
        "samples": samples,
    }


def read_and_adapt_batch(
    reader: Any,
    args: argparse.Namespace,
    image_ids: list[int],
    crop: tuple[int, int, int, int] | None,
    rgbnomore_dct_val_transform: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    if args.preprocess == "rgbnomore-val":
        y_items = []
        cbcr_items = []
        batches = []
        for image_id in image_ids:
            batch = _read_grid_batch(
                reader,
                [image_id],
                None,
                args.cache_capacity_mib,
                decode_batch_rowgroups=int(getattr(args, "decode_batch_rowgroups", 2)),
            )
            input_y, input_cbcr = adapt_galp_batch_to_rgbnomore(
                reader,
                batch,
                [image_id],
                dequantize=not args.no_dequantize,
                scale=not args.no_scale,
                preprocess=args.preprocess,
                rgbnomore_dct_val_transform=rgbnomore_dct_val_transform,
            )
            y_items.append(input_y[0])
            cbcr_items.append(input_cbcr[0])
            batches.append(batch)
        return torch.stack(y_items, dim=0), torch.stack(cbcr_items, dim=0), batches

    if args.preprocess == "rgbnomore-val-pushdown":
        batch = _read_grid_batch(
            reader,
            image_ids,
            None,
            args.cache_capacity_mib,
            layout="transformed_dct_grid",
            grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32,
            decode_batch_rowgroups=int(getattr(args, "decode_batch_rowgroups", 2)),
            rowgroup_prefetch_depth=int(getattr(args, "rowgroup_prefetch_depth", 16)),
            rowgroup_prefetch_workers=int(getattr(args, "rowgroup_prefetch_workers", 4)),
            rowgroup_prefetch_min_decode_batches=int(
                getattr(args, "rowgroup_prefetch_min_decode_batches", 2)
            ),
            plan_cache_capacity=int(getattr(args, "plan_cache_capacity", 128)),
            enable_planless_execution=bool(getattr(args, "enable_planless_execution", True)),
            crop_execution_mode=str(getattr(args, "crop_execution_mode", "auto")),
        )
        input_y, input_cbcr = adapt_galp_batch_to_rgbnomore(
            reader,
            batch,
            image_ids,
            dequantize=not args.no_dequantize,
            scale=not args.no_scale,
            preprocess=args.preprocess,
            rgbnomore_dct_val_transform=None,
        )
        return input_y, input_cbcr, [batch]

    batch = _read_grid_batch(
        reader,
        image_ids,
        crop,
        args.cache_capacity_mib,
        decode_batch_rowgroups=int(getattr(args, "decode_batch_rowgroups", 2)),
        crop_execution_mode=str(getattr(args, "crop_execution_mode", "auto")),
    )
    input_y, input_cbcr = adapt_galp_batch_to_rgbnomore(
        reader,
        batch,
        image_ids,
        dequantize=not args.no_dequantize,
        scale=not args.no_scale,
        preprocess=args.preprocess,
        rgbnomore_dct_val_transform=rgbnomore_dct_val_transform,
    )
    return input_y, input_cbcr, [batch]


def _prefetch_pushdown_batch(reader: Any, args: argparse.Namespace, image_ids: list[int]) -> Any:
    return reader.prefetch_batch(
        image_ids,
        crop=None,
        dct_coeffs="all",
        cache_capacity_mib=args.cache_capacity_mib,
        decode_batch_rowgroups=int(getattr(args, "decode_batch_rowgroups", 2)),
        rowgroup_prefetch_depth=int(getattr(args, "rowgroup_prefetch_depth", 16)),
        rowgroup_prefetch_workers=int(getattr(args, "rowgroup_prefetch_workers", 4)),
        rowgroup_prefetch_min_decode_batches=int(
            getattr(args, "rowgroup_prefetch_min_decode_batches", 2)
        ),
        plan_cache_capacity=int(getattr(args, "plan_cache_capacity", 128)),
        enable_planless_execution=bool(getattr(args, "enable_planless_execution", True)),
        scheduling_policy=str(getattr(args, "scheduling_policy", "fully-overlapped")),
        transform_blocks_per_launch=int(getattr(args, "transform_blocks_per_launch", 0)),
        transform_ctas_per_launch=int(getattr(args, "transform_ctas_per_launch", 0)),
        use_low_priority_streams=bool(getattr(args, "use_low_priority_streams", False)),
        crop_execution_mode=str(getattr(args, "crop_execution_mode", "auto")),
        layout="transformed_dct_grid",
        grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32,
    )


def _adapt_prefetched_pushdown_batch(
    reader: Any,
    args: argparse.Namespace,
    image_ids: list[int],
    pending: Any,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    batch = pending.read()
    input_y, input_cbcr = adapt_galp_batch_to_rgbnomore(
        reader,
        batch,
        image_ids,
        dequantize=not args.no_dequantize,
        scale=not args.no_scale,
        preprocess=args.preprocess,
        rgbnomore_dct_val_transform=None,
    )
    return input_y, input_cbcr, [batch]


def _read_crop_for_preprocess(args: argparse.Namespace) -> tuple[int, int, int, int] | None:
    if args.preprocess == "rgbnomore-val":
        return None
    if args.preprocess == "rgbnomore-val-pushdown":
        return None
    return tuple(int(value) for value in args.crop)


def _output_layout_for_preprocess(preprocess: str) -> str:
    return "transformed_dct_grid" if preprocess == "rgbnomore-val-pushdown" else "ycbcr_dct_grid"


def run_loader_phase(
    reader: Any,
    args: argparse.Namespace,
    crop: tuple[int, int, int, int] | None,
    device: torch.device,
    rgbnomore_dct_val_transform: torch.nn.Module | None,
) -> dict[str, Any]:
    unsupported_sampling_skips: list[dict[str, Any]] = []
    execution_path_batches: list[dict[str, Any]] = []
    for warmup_step in range(args.warmup):
        image_ids, skipped = _make_benchmark_image_ids(reader, args, warmup_step, warmup=True)
        _record_unsupported_sampling_skips(unsupported_sampling_skips, warmup_step, skipped)
        read_and_adapt_batch(reader, args, image_ids, crop, rgbnomore_dct_val_transform)
        _sync(device)

    _sync(device)
    totals = _empty_totals()
    projection_distribution_samples: list[dict[str, float | int]] = []
    total_images = 0
    input_y_shape: list[int] | None = None
    input_cbcr_shape: list[int] | None = None
    async_prefetch = args.preprocess == "rgbnomore-val-pushdown" and bool(
        getattr(args, "async_prefetch", True)
    )
    started = time.perf_counter()
    scheduled: list[tuple[list[int], list[dict[str, Any]]]] = []
    if async_prefetch:
        for step in range(args.steps):
            scheduled.append(_make_benchmark_image_ids(reader, args, step))
        pending = _prefetch_pushdown_batch(reader, args, scheduled[0][0]) if scheduled else None
    for step in range(args.steps):
        image_ids, skipped = scheduled[step] if async_prefetch else _make_benchmark_image_ids(reader, args, step)
        _record_unsupported_sampling_skips(unsupported_sampling_skips, step, skipped)
        if async_prefetch:
            if pending is None:
                raise RuntimeError("pushdown async prefetch state was not initialized")
            input_y, input_cbcr, batches = _adapt_prefetched_pushdown_batch(reader, args, image_ids, pending)
            pending = (
                _prefetch_pushdown_batch(reader, args, scheduled[step + 1][0])
                if step + 1 < args.steps
                else None
            )
        else:
            input_y, input_cbcr, batches = read_and_adapt_batch(
                reader, args, image_ids, crop, rgbnomore_dct_val_transform
            )
        if input_y_shape is None:
            input_y_shape = list(input_y.shape)
            input_cbcr_shape = list(input_cbcr.shape)
        total_images += len(image_ids)
        _accumulate_many_stats(totals, batches)
        _record_projection_distribution(projection_distribution_samples, batches, len(image_ids))
        if args.preprocess == "rgbnomore-val-pushdown" and len(batches) != 1:
            raise RuntimeError(
                f"pushdown batch must use exactly one batched read, got {len(batches)} read_batch calls"
            )
        batch_stats = [dict(batch.execution_stats) for batch in batches]
        execution_path_batches.append(
            {
                "step": step,
                "preprocess": args.preprocess,
                "output_layout": _output_layout_for_preprocess(args.preprocess),
                "image_ids": image_ids,
                "image_count": len(image_ids),
                "read_batch_call_count": len(batches),
                "fixed_specialized_path_used": all(_uses_fixed_transform(stats) for stats in batch_stats),
                "planless_path_used": all(_uses_planless_fixed_transform(stats) for stats in batch_stats),
                "generic_projection_used": any(
                    int(stats.get("projection_item_count", 0)) != 0
                    or int(stats.get("decoded_projection_item_count", 0)) != 0
                    or int(stats.get("project_decoded_ycbcr_grid_launch_count", 0)) != 0
                    for stats in batch_stats
                ),
                "fallback_reason": "",
                "stats": batch_stats,
                "cache_stats": [dict(batch.cache_stats) for batch in batches],
            }
        )
        if step == 0:
            print(f"phase=loader_to_device step=0 y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)}")
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "galp_direct_dct_rgbnomore",
        "phase": "loader_to_device",
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "manifest": str(args.manifest),
        "dataset_size": int(reader.image_count),
        "crop": None if crop is None else list(crop),
        "preprocess": args.preprocess,
        "dct_coeffs": "all",
        "output_layout": _output_layout_for_preprocess(args.preprocess),
        "dequantize": not args.no_dequantize,
        "scale_to_rgbnomore_range": not args.no_scale,
        "input_y_shape": input_y_shape,
        "input_cbcr_shape": input_cbcr_shape,
        "device": str(device),
        "cache_capacity_mib": args.cache_capacity_mib,
        "async_prefetch": async_prefetch,
        "decode_batch_rowgroups": args.decode_batch_rowgroups,
        "rowgroup_prefetch_depth": args.rowgroup_prefetch_depth,
        "rowgroup_prefetch_workers": args.rowgroup_prefetch_workers,
        "rowgroup_prefetch_min_decode_batches": args.rowgroup_prefetch_min_decode_batches,
        "sampling_policy": args.sampling_policy,
        "image_order": getattr(args, "image_order", "sequential"),
        "shuffle_seed": int(getattr(args, "shuffle_seed", 20260718)),
        "image_ids_wrapped": not args.no_wrap_image_ids,
        "unsupported_sampling_skipped_count": len(unsupported_sampling_skips),
        "unsupported_sampling_skipped_samples": unsupported_sampling_skips[:16],
        "projection_items_per_image_distribution": _summarize_projection_distribution(
            projection_distribution_samples
        ),
        "execution_path_batches": execution_path_batches,
        **totals,
    }
    _annotate_fixed_path_result(result, args.preprocess)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_forward_phase(
    reader: Any,
    args: argparse.Namespace,
    crop: tuple[int, int, int, int] | None,
    model: torch.nn.Module,
    device: torch.device,
    rgbnomore_dct_val_transform: torch.nn.Module | None,
) -> dict[str, Any]:
    unsupported_sampling_skips: list[dict[str, Any]] = []
    image_ids, skipped = _make_benchmark_image_ids(reader, args, 0)
    _record_unsupported_sampling_skips(unsupported_sampling_skips, 0, skipped)
    input_y, input_cbcr, batches = read_and_adapt_batch(
        reader, args, image_ids, crop, rgbnomore_dct_val_transform
    )
    totals = _empty_totals()
    _accumulate_many_stats(totals, batches)
    projection_distribution_samples: list[dict[str, float | int]] = []
    _record_projection_distribution(projection_distribution_samples, batches, len(image_ids))
    _sync(device)
    for _ in range(args.warmup):
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if tuple(logits.shape) != (len(image_ids), 1000):
            raise RuntimeError(f"expected logits shape {(len(image_ids), 1000)}, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    total_images = 0
    logits_shape: list[int] | None = None
    input_y_shape = list(input_y.shape)
    input_cbcr_shape = list(input_cbcr.shape)
    started = time.perf_counter()
    for step in range(args.steps):
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if tuple(logits.shape) != (len(image_ids), 1000):
            raise RuntimeError(f"expected logits shape {(len(image_ids), 1000)}, got {tuple(logits.shape)}")
        logits_shape = list(logits.shape)
        total_images += len(image_ids)
        if step == 0:
            print(
                "phase=forward_step step=0 "
                f"y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)} logits_shape={tuple(logits.shape)}"
            )
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "galp_direct_dct_rgbnomore",
        "phase": "forward_step",
        "model": "rgbnomore_jpeg_ti_vitti",
        "checkpoint": str(args.checkpoint),
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "manifest": str(args.manifest),
        "dataset_size": int(reader.image_count),
        "crop": None if crop is None else list(crop),
        "preprocess": args.preprocess,
        "dct_coeffs": "all",
        "output_layout": _output_layout_for_preprocess(args.preprocess),
        "dequantize": not args.no_dequantize,
        "scale_to_rgbnomore_range": not args.no_scale,
        "input_y_shape": input_y_shape,
        "input_cbcr_shape": input_cbcr_shape,
        "device": str(device),
        "logits_shape": logits_shape,
        "cache_capacity_mib": args.cache_capacity_mib,
        "sampling_policy": args.sampling_policy,
        "image_order": getattr(args, "image_order", "sequential"),
        "shuffle_seed": int(getattr(args, "shuffle_seed", 20260718)),
        "image_ids_wrapped": not args.no_wrap_image_ids,
        "unsupported_sampling_skipped_count": len(unsupported_sampling_skips),
        "unsupported_sampling_skipped_samples": unsupported_sampling_skips[:16],
        "projection_items_per_image_distribution": _summarize_projection_distribution(
            projection_distribution_samples
        ),
        **totals,
    }
    _annotate_fixed_path_result(result, args.preprocess)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_end_to_end_phase(
    reader: Any,
    args: argparse.Namespace,
    crop: tuple[int, int, int, int] | None,
    model: torch.nn.Module,
    device: torch.device,
    rgbnomore_dct_val_transform: torch.nn.Module | None,
) -> dict[str, Any]:
    unsupported_sampling_skips: list[dict[str, Any]] = []
    for warmup_step in range(args.warmup):
        image_ids, skipped = _make_benchmark_image_ids(reader, args, warmup_step, warmup=True)
        _record_unsupported_sampling_skips(unsupported_sampling_skips, warmup_step, skipped)
        input_y, input_cbcr, _batches = read_and_adapt_batch(
            reader, args, image_ids, crop, rgbnomore_dct_val_transform
        )
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if tuple(logits.shape) != (len(image_ids), 1000):
            raise RuntimeError(f"expected logits shape {(len(image_ids), 1000)}, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    totals = _empty_totals()
    projection_distribution_samples: list[dict[str, float | int]] = []
    execution_path_batches: list[dict[str, Any]] = []
    total_images = 0
    logits_shape: list[int] | None = None
    input_y_shape: list[int] | None = None
    input_cbcr_shape: list[int] | None = None
    async_prefetch = args.preprocess == "rgbnomore-val-pushdown" and bool(
        getattr(args, "async_prefetch", True)
    )
    started = time.perf_counter()
    scheduled: list[tuple[list[int], list[dict[str, Any]]]] = []
    if async_prefetch:
        for step in range(args.steps):
            scheduled.append(_make_benchmark_image_ids(reader, args, step))
        pending = _prefetch_pushdown_batch(reader, args, scheduled[0][0]) if scheduled else None
    for step in range(args.steps):
        image_ids, skipped = scheduled[step] if async_prefetch else _make_benchmark_image_ids(reader, args, step)
        _record_unsupported_sampling_skips(unsupported_sampling_skips, step, skipped)
        if async_prefetch:
            if pending is None:
                raise RuntimeError("pushdown async prefetch state was not initialized")
            input_y, input_cbcr, batches = _adapt_prefetched_pushdown_batch(reader, args, image_ids, pending)
            pending = (
                _prefetch_pushdown_batch(reader, args, scheduled[step + 1][0])
                if step + 1 < args.steps
                else None
            )
        else:
            input_y, input_cbcr, batches = read_and_adapt_batch(
                reader, args, image_ids, crop, rgbnomore_dct_val_transform
            )
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if tuple(logits.shape) != (len(image_ids), 1000):
            raise RuntimeError(f"expected logits shape {(len(image_ids), 1000)}, got {tuple(logits.shape)}")
        logits_shape = list(logits.shape)
        if input_y_shape is None:
            input_y_shape = list(input_y.shape)
            input_cbcr_shape = list(input_cbcr.shape)
        total_images += len(image_ids)
        _accumulate_many_stats(totals, batches)
        _record_projection_distribution(projection_distribution_samples, batches, len(image_ids))
        if args.preprocess == "rgbnomore-val-pushdown":
            if len(batches) != 1:
                raise RuntimeError(
                    f"pushdown end-to-end batch must use exactly one batched read, got {len(batches)}"
                )
            stats = dict(batches[0].execution_stats)
            fixed_path = _uses_fixed_transform(stats)
            generic_projection = (
                int(stats.get("projection_item_count", 0)) != 0
                or int(stats.get("decoded_projection_item_count", 0)) != 0
                or int(stats.get("project_decoded_ycbcr_grid_launch_count", 0)) != 0
            )
            if not fixed_path or generic_projection:
                raise RuntimeError(
                    "pushdown end-to-end batch left the fixed specialized path "
                    f"(fixed={fixed_path}, generic_projection={generic_projection})"
                )
            execution_path_batches.append(
                {
                    "step": step,
                    "image_ids": image_ids,
                    "image_count": len(image_ids),
                    "read_batch_call_count": 1,
                    "fixed_specialized_path_used": True,
                    "planless_path_used": _uses_planless_fixed_transform(stats),
                    "generic_projection_used": False,
                    "fallback_reason": "",
                    "stats": [stats],
                    "cache_stats": [dict(batches[0].cache_stats)],
                }
            )
        if step == 0:
            print(
                "phase=end_to_end step=0 "
                f"y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)} logits_shape={tuple(logits.shape)}"
            )
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "galp_direct_dct_rgbnomore",
        "phase": "end_to_end",
        "model": "rgbnomore_jpeg_ti_vitti",
        "checkpoint": str(args.checkpoint),
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "manifest": str(args.manifest),
        "dataset_size": int(reader.image_count),
        "crop": None if crop is None else list(crop),
        "preprocess": args.preprocess,
        "dct_coeffs": "all",
        "output_layout": _output_layout_for_preprocess(args.preprocess),
        "dequantize": not args.no_dequantize,
        "scale_to_rgbnomore_range": not args.no_scale,
        "input_y_shape": input_y_shape,
        "input_cbcr_shape": input_cbcr_shape,
        "device": str(device),
        "logits_shape": logits_shape,
        "cache_capacity_mib": args.cache_capacity_mib,
        "async_prefetch": async_prefetch,
        "decode_batch_rowgroups": args.decode_batch_rowgroups,
        "rowgroup_prefetch_depth": args.rowgroup_prefetch_depth,
        "rowgroup_prefetch_workers": args.rowgroup_prefetch_workers,
        "rowgroup_prefetch_min_decode_batches": args.rowgroup_prefetch_min_decode_batches,
        "sampling_policy": args.sampling_policy,
        "image_order": getattr(args, "image_order", "sequential"),
        "shuffle_seed": int(getattr(args, "shuffle_seed", 20260718)),
        "image_ids_wrapped": not args.no_wrap_image_ids,
        "unsupported_sampling_skipped_count": len(unsupported_sampling_skips),
        "unsupported_sampling_skipped_samples": unsupported_sampling_skips[:16],
        "projection_items_per_image_distribution": _summarize_projection_distribution(
            projection_distribution_samples
        ),
        "execution_path_batches": execution_path_batches,
        **totals,
    }
    _annotate_fixed_path_result(result, args.preprocess)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_train_phase(
    reader: Any,
    args: argparse.Namespace,
    crop: tuple[int, int, int, int] | None,
    model: torch.nn.Module,
    device: torch.device,
    rgbnomore_dct_val_transform: torch.nn.Module | None,
    labels_by_image_id: list[int] | None,
    label_index: dict[str, int] | None,
) -> dict[str, Any]:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr)
    unsupported_sampling_skips: list[dict[str, Any]] = []

    for warmup_step in range(args.warmup):
        image_ids, skipped = _make_benchmark_image_ids(reader, args, warmup_step, warmup=True)
        _record_unsupported_sampling_skips(unsupported_sampling_skips, warmup_step, skipped)
        input_y, input_cbcr, _batches = read_and_adapt_batch(
            reader, args, image_ids, crop, rgbnomore_dct_val_transform
        )
        labels = (
            _labels_for_image_ids(image_ids, labels_by_image_id, device)
            if labels_by_image_id is not None
            else _labels_for_image_ids_from_source_path(reader, image_ids, label_index, device)
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_y, input_cbcr)
        if tuple(logits.shape) != (len(image_ids), 1000):
            raise RuntimeError(f"expected logits shape {(len(image_ids), 1000)}, got {tuple(logits.shape)}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        _sync(device)

    _sync(device)
    totals = _empty_totals()
    projection_distribution_samples: list[dict[str, float | int]] = []
    total_images = 0
    logits_shape: list[int] | None = None
    input_y_shape: list[int] | None = None
    input_cbcr_shape: list[int] | None = None
    last_loss: float | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        image_ids, skipped = _make_benchmark_image_ids(reader, args, step)
        _record_unsupported_sampling_skips(unsupported_sampling_skips, step, skipped)
        input_y, input_cbcr, batches = read_and_adapt_batch(
            reader, args, image_ids, crop, rgbnomore_dct_val_transform
        )
        labels = (
            _labels_for_image_ids(image_ids, labels_by_image_id, device)
            if labels_by_image_id is not None
            else _labels_for_image_ids_from_source_path(reader, image_ids, label_index, device)
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_y, input_cbcr)
        if tuple(logits.shape) != (len(image_ids), 1000):
            raise RuntimeError(f"expected logits shape {(len(image_ids), 1000)}, got {tuple(logits.shape)}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        logits_shape = list(logits.shape)
        if input_y_shape is None:
            input_y_shape = list(input_y.shape)
            input_cbcr_shape = list(input_cbcr.shape)
        total_images += len(image_ids)
        _accumulate_many_stats(totals, batches)
        _record_projection_distribution(projection_distribution_samples, batches, len(image_ids))
        last_loss = float(loss.detach().cpu())
        if step == 0:
            print(
                "phase=train_step step=0 "
                f"y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)} "
                f"logits_shape={tuple(logits.shape)} loss={last_loss}"
            )
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "galp_direct_dct_rgbnomore",
        "phase": "train_step",
        "model": "rgbnomore_jpeg_ti_vitti",
        "checkpoint": str(args.checkpoint),
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "manifest": str(args.manifest),
        "dataset_size": int(reader.image_count),
        "crop": None if crop is None else list(crop),
        "preprocess": args.preprocess,
        "dct_coeffs": "all",
        "output_layout": _output_layout_for_preprocess(args.preprocess),
        "dequantize": not args.no_dequantize,
        "scale_to_rgbnomore_range": not args.no_scale,
        "input_y_shape": input_y_shape,
        "input_cbcr_shape": input_cbcr_shape,
        "device": str(device),
        "logits_shape": logits_shape,
        "loss": last_loss,
        "optimizer": "sgd",
        "train_lr": args.train_lr,
        "label_source": "rgbnomore_label_map_json" if labels_by_image_id is not None else "rgbnomore_index_file_source_path",
        "label_map_json": str(args.label_map_json) if args.label_map_json is not None else None,
        "index_file": str(args.index_file) if args.index_file is not None else None,
        "cache_capacity_mib": args.cache_capacity_mib,
        "sampling_policy": args.sampling_policy,
        "image_order": getattr(args, "image_order", "sequential"),
        "shuffle_seed": int(getattr(args, "shuffle_seed", 20260718)),
        "image_ids_wrapped": not args.no_wrap_image_ids,
        "unsupported_sampling_skipped_count": len(unsupported_sampling_skips),
        "unsupported_sampling_skipped_samples": unsupported_sampling_skips[:16],
        "projection_items_per_image_distribution": _summarize_projection_distribution(
            projection_distribution_samples
        ),
        **totals,
    }
    _annotate_fixed_path_result(result, args.preprocess)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GALP Direct-DCT to RGB-no-more JPEG-Ti benchmark")
    parser.add_argument("manifest")
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DCT_CHECKPOINT)
    parser.add_argument("--label-map-json", type=Path, help="Sidecar JSON with labels indexed by GALP image id, used for --phase train.")
    parser.add_argument("--index-file", type=Path, help="Fallback RGB-no-more index CSV used only if GALP source_path metadata is available for --phase train.")
    parser.add_argument("--phase", choices=("metadata", "inspect", "loader", "forward", "end-to-end", "train", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop", type=int, nargs=4, default=(30, 40, 224, 224), metavar=("X", "Y", "W", "H"))
    parser.add_argument(
        "--preprocess",
        choices=("rgbnomore-val", "rgbnomore-val-pushdown", "direct-crop"),
        default="rgbnomore-val",
        help="Use RGB-no-more DCT val preprocessing in Python, request GALP fixed-grid pushdown, or require direct crop output to already match JPEG-Ti shapes.",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--no-wrap-image-ids",
        action="store_true",
        help="Consume the selected image population exactly once and allow a partial final batch.",
    )
    parser.add_argument(
        "--cache-capacity-mib",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--decode-batch-rowgroups",
        type=int,
        default=2,
        help="Number of materialized rowgroups combined into one decode workset.",
    )
    parser.add_argument(
        "--plan-cache-capacity",
        type=int,
        default=0,
        help="Legacy exact-batch plan entries; canonical planless execution bypasses this cache.",
    )
    parser.add_argument(
        "--disable-planless-execution",
        action="store_false",
        dest="enable_planless_execution",
        help="Diagnostic A/B mode that forces the legacy expanded transformed-grid graph.",
    )
    parser.set_defaults(enable_planless_execution=True)
    parser.add_argument("--rowgroup-prefetch-depth", type=int, default=16)
    parser.add_argument("--rowgroup-prefetch-workers", type=int, default=4)
    parser.add_argument("--rowgroup-prefetch-min-decode-batches", type=int, default=2)
    parser.add_argument(
        "--no-async-prefetch",
        action="store_false",
        dest="async_prefetch",
        help="Disable next-batch Direct-DCT prefetch/forward overlap in the end-to-end phase.",
    )
    parser.set_defaults(async_prefetch=True)
    parser.add_argument(
        "--sampling-policy",
        choices=("preprocess-default", "supported-only", "all"),
        default="preprocess-default",
        help=(
            "Image selection policy. preprocess-default and supported-only use the complete "
            "ImageNet validation JPEG domain accepted by RGB-no-more/GALP "
            "(4:4:4, 4:2:0, 4:2:2, 4:4:0, 4:1:1, grayscale, and four-component); "
            "all disables sampling filtering for diagnostic manifests."
        ),
    )
    parser.add_argument(
        "--image-order",
        choices=("sequential", "shuffled"),
        default="sequential",
        help="Deterministic image-id trace used by loader/forward/end-to-end/train phases.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260718)
    parser.add_argument("--no-dequantize", action="store_true", help="Feed raw quantized coefficients instead of RGB-no-more dequantized inputs.")
    parser.add_argument("--no-scale", action="store_true", help="Do not scale dequantized DCT coefficients from [-1024,1016] to RGB-no-more's [-1,1] range.")
    parser.add_argument("--train-lr", type=float, default=0.0, help="SGD learning rate used only for --phase train.")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.cache_capacity_mib < 0:
        raise ValueError("--cache-capacity-mib must be non-negative")
    if args.plan_cache_capacity < 0:
        raise ValueError("--plan-cache-capacity must be non-negative")
    if args.decode_batch_rowgroups <= 0:
        raise ValueError("--decode-batch-rowgroups must be positive")
    if (
        args.rowgroup_prefetch_depth <= 0
        or args.rowgroup_prefetch_workers <= 0
        or args.rowgroup_prefetch_min_decode_batches <= 0
    ):
        raise ValueError("--rowgroup-prefetch-depth/workers/min-decode-batches must be positive")
    if args.train_lr < 0.0:
        raise ValueError("--train-lr must be non-negative")
    if args.preprocess in ("rgbnomore-val", "rgbnomore-val-pushdown") and args.no_scale:
        raise ValueError("RGB-no-more val preprocessing uses RGB-no-more ToRange(-1,1); --no-scale is only valid with --preprocess direct-crop")
    if args.phase in ("forward", "end-to-end", "train", "both") and not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.phase == "train":
        if args.label_map_json is None and args.index_file is None:
            raise ValueError("--phase train requires --label-map-json, or --index-file only when GALP source_path metadata is available")
        if args.label_map_json is not None and not args.label_map_json.exists():
            raise FileNotFoundError(args.label_map_json)
        if args.index_file is not None and not args.index_file.exists():
            raise FileNotFoundError(args.index_file)

    crop = _read_crop_for_preprocess(args)
    reader = galp_dct.DirectDctReader(args.manifest)
    results: list[dict[str, Any]] = []
    if args.phase == "metadata":
        results.append(inspect_metadata_only(reader, args))
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GALP Direct-DCT read_batch phases require CUDA")
        first_ids, first_skipped = _make_benchmark_image_ids(reader, args, 0)
        if first_skipped:
            print("unsupported_sampling_skipped_first_batch=" + json.dumps(first_skipped[:16], sort_keys=True))
    if args.phase == "inspect":
        compact_batch = _read_compact_batch(reader, first_ids, crop, args.cache_capacity_mib)
        grid_batch = _read_grid_batch(reader, first_ids, crop, args.cache_capacity_mib)
        results.append(inspect_batch(reader, compact_batch, grid_batch, str(args.manifest)))
    elif args.phase != "metadata":
        rgbnomore_dct_val_transform = (
            build_rgbnomore_dct_val_transform(args.rgbnomore_root)
            if args.preprocess == "rgbnomore-val"
            else None
        )
        first_y, first_cbcr, _first_batches = read_and_adapt_batch(
            reader, args, first_ids, crop, rgbnomore_dct_val_transform
        )
        device = first_y.device
        print(
            "adapter=galp_ycbcr_dct_grid_to_rgbnomore "
            f"crop={crop} preprocess={args.preprocess} dct_coeffs=all dequantize={not args.no_dequantize} "
            f"scale_to_rgbnomore_range={not args.no_scale} "
            f"y_shape={tuple(first_y.shape)} cbcr_shape={tuple(first_cbcr.shape)}"
        )
        if args.phase in ("loader", "both"):
            results.append(run_loader_phase(reader, args, crop, device, rgbnomore_dct_val_transform))
        if args.phase in ("forward", "both"):
            model = build_rgbnomore_jpeg_ti(args.rgbnomore_root, args.checkpoint, device)
            results.append(run_forward_phase(reader, args, crop, model, device, rgbnomore_dct_val_transform))
        if args.phase in ("end-to-end", "both"):
            model = build_rgbnomore_jpeg_ti(args.rgbnomore_root, args.checkpoint, device)
            results.append(run_end_to_end_phase(reader, args, crop, model, device, rgbnomore_dct_val_transform))
        if args.phase == "train":
            model = build_rgbnomore_jpeg_ti(args.rgbnomore_root, args.checkpoint, device)
            labels_by_image_id = (
                _load_label_map_json(args.label_map_json, int(reader.image_count))
                if args.label_map_json is not None
                else None
            )
            label_index = _load_label_index(args.index_file) if labels_by_image_id is None else None
            results.append(
                run_train_phase(
                    reader,
                    args,
                    crop,
                    model,
                    device,
                    rgbnomore_dct_val_transform,
                    labels_by_image_id,
                    label_index,
                )
            )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
