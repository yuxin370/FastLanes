#!/usr/bin/env python3
"""Benchmark GALP Direct-DCT batches against RGB-no-more JPEG-Ti.

This script keeps the existing tiny Direct-DCT demo separate. It requests the
GALP Y/CbCr DCT grid layout, adapts it to RGB-no-more's JPEG-Ti input contract,
optionally dequantizes with JPEG quantization tables from GALP metadata, and
feeds the tensors into RGB-no-more's ViT-Ti DCT model.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

import _galp_direct_dct as galp_dct


DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_DCT_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints" / "imgnetDCTViTTi_ep300_75.1.pth"


def _make_image_ids(step: int, batch_size: int, image_count: int) -> list[int]:
    if image_count <= 0:
        raise RuntimeError("manifest has no images")
    count = min(batch_size, image_count)
    start = (step * count) % image_count
    return [int((start + index) % image_count) for index in range(count)]


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
    preprocess: str = "none",
) -> Any:
    return reader.read_batch(
        image_ids,
        crop=crop,
        dct_coeffs="all",
        cache_capacity_mib=cache_capacity_mib,
        layout=layout,
        preprocess=preprocess,
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
    return (tensor + 1024.0) / 2040.0 * 2.0 - 1.0


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
    if batch.layout not in ("ycbcr_dct_grid", "ycbcr_dct_grid_fixed"):
        raise RuntimeError(f"expected GALP Y/CbCr DCT grid layout, got {batch.layout!r}")
    if not _selected_coefficients_are_all(list(batch.selected_coefficients)):
        raise RuntimeError("RGB-no-more adapter requires dct_coeffs='all'; sparse first:N/list:N cannot restore full 8x8 grids")

    input_y = batch.y
    input_cbcr = batch.cbcr
    if input_y.dtype != torch.int16 or input_cbcr.dtype != torch.int16:
        raise RuntimeError(f"expected int16 GALP DCT grids, got y={input_y.dtype} cbcr={input_cbcr.dtype}")

    y_float = input_y.to(torch.float32)
    cbcr_float = input_cbcr.to(torch.float32)
    if not dequantize and preprocess == "direct-crop":
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

    if preprocess == "rgbnomore-val-pushdown":
        if not dequantize:
            raise RuntimeError("rgbnomore-val-pushdown preprocessing requires dequantized DCT coefficients")
        if scale:
            y_float = _scale_to_rgbnomore_dct_range(y_float)
            cbcr_float = _scale_to_rgbnomore_dct_range(cbcr_float)
        _validate_rgbnomore_shapes(y_float, cbcr_float, image_ids)
        return y_float, cbcr_float

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


def _accumulate_stats(totals: dict[str, int], batch: Any) -> None:
    stats = batch.execution_stats
    totals["selected_vectors"] += int(stats["selected_vector_count"])
    totals["full_vectors"] += int(stats["full_vector_count"])
    totals["decode_kernels"] += int(stats["decode_kernel_launch_count"])
    totals["rowgroups"] += int(stats["rowgroup_count"])
    totals["worksets"] += int(stats["workset_count"])
    totals["projection_items"] += int(stats["projection_item_count"])
    totals["internal_syncs"] += int(stats["internal_sync_count"])


def _empty_totals() -> dict[str, int]:
    return {
        "selected_vectors": 0,
        "full_vectors": 0,
        "decode_kernels": 0,
        "rowgroups": 0,
        "worksets": 0,
        "projection_items": 0,
        "internal_syncs": 0,
    }


def _accumulate_many_stats(totals: dict[str, int], batches: list[Any]) -> None:
    for batch in batches:
        _accumulate_stats(totals, batch)


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
            batch = _read_grid_batch(reader, [image_id], None, args.cache_capacity_mib)
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
            layout="ycbcr_dct_grid_fixed",
            preprocess="rgbnomore_val",
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

    batch = _read_grid_batch(reader, image_ids, crop, args.cache_capacity_mib)
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


def _read_crop_for_preprocess(args: argparse.Namespace) -> tuple[int, int, int, int] | None:
    if args.preprocess == "rgbnomore-val":
        return None
    if args.preprocess == "rgbnomore-val-pushdown":
        return None
    return tuple(int(value) for value in args.crop)


def _output_layout_for_preprocess(preprocess: str) -> str:
    return "ycbcr_dct_grid_fixed" if preprocess == "rgbnomore-val-pushdown" else "ycbcr_dct_grid"


def run_loader_phase(
    reader: Any,
    args: argparse.Namespace,
    crop: tuple[int, int, int, int] | None,
    device: torch.device,
    rgbnomore_dct_val_transform: torch.nn.Module | None,
) -> dict[str, Any]:
    for warmup_step in range(args.warmup):
        image_ids = _make_image_ids(warmup_step, args.batch_size, reader.image_count)
        read_and_adapt_batch(reader, args, image_ids, crop, rgbnomore_dct_val_transform)
        _sync(device)

    _sync(device)
    totals = _empty_totals()
    total_images = 0
    input_y_shape: list[int] | None = None
    input_cbcr_shape: list[int] | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        image_ids = _make_image_ids(step, args.batch_size, reader.image_count)
        input_y, input_cbcr, batches = read_and_adapt_batch(
            reader, args, image_ids, crop, rgbnomore_dct_val_transform
        )
        if input_y_shape is None:
            input_y_shape = list(input_y.shape)
            input_cbcr_shape = list(input_cbcr.shape)
        total_images += len(image_ids)
        _accumulate_many_stats(totals, batches)
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
        **totals,
    }
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
    image_ids = _make_image_ids(0, args.batch_size, reader.image_count)
    input_y, input_cbcr, _batches = read_and_adapt_batch(
        reader, args, image_ids, crop, rgbnomore_dct_val_transform
    )
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
    }
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
    for warmup_step in range(args.warmup):
        image_ids = _make_image_ids(warmup_step, args.batch_size, reader.image_count)
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
    total_images = 0
    logits_shape: list[int] | None = None
    input_y_shape: list[int] | None = None
    input_cbcr_shape: list[int] | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        image_ids = _make_image_ids(step, args.batch_size, reader.image_count)
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
        **totals,
    }
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

    for warmup_step in range(args.warmup):
        image_ids = _make_image_ids(warmup_step, args.batch_size, reader.image_count)
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
    total_images = 0
    logits_shape: list[int] | None = None
    input_y_shape: list[int] | None = None
    input_cbcr_shape: list[int] | None = None
    last_loss: float | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        image_ids = _make_image_ids(step, args.batch_size, reader.image_count)
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
        **totals,
    }
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
        "--cache-capacity-mib",
        type=int,
        default=int(getattr(galp_dct, "DEFAULT_CACHE_CAPACITY_MIB", 1024)),
    )
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
        first_ids = _make_image_ids(0, args.batch_size, reader.image_count)
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
