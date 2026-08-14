#!/usr/bin/env python3
"""End-to-end stay-on-GPU GALP Direct-DCT PyTorch demo.

The demo intentionally stays in the DCT domain: it reads compact DCT
coefficient tensors from GALP, converts them to float on CUDA only, aggregates
block tokens into per-image logits, and optionally runs a short optimizer smoke.
It can also request the YCbCr DCT grid layout and feed the
GPU-resident grid tensors into the same smoke model as per-image DCT features.
Only crop pushdown is used here; flip/rotation augmentation is left out until a
correct DCT-domain implementation is available.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

import _galp_direct_dct as galp_dct

RGBNOMORE_DIAGNOSTICS_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/system_rgbnomore/diagnostics"
)
if str(RGBNOMORE_DIAGNOSTICS_DIR) not in sys.path:
    sys.path.insert(0, str(RGBNOMORE_DIAGNOSTICS_DIR))

from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM


@dataclass(frozen=True)
class ImageBlockLayout:
    offsets: torch.Tensor
    counts: torch.Tensor
    block_to_image: torch.Tensor


@dataclass
class DirectDctBatchView:
    coefficients: torch.Tensor
    layout: ImageBlockLayout
    block_count: int
    coefficients_per_block: int
    data_ptr: int
    galp_ptr: int
    tensor_is_galp_backed: bool
    stats: dict[str, Any]
    cache_stats: dict[str, Any]
    output_layout: str
    selected_coefficients: list[int]
    y: torch.Tensor | None = None
    cbcr: torch.Tensor | None = None
    y_ptr: int = 0
    cbcr_ptr: int = 0

    @property
    def image_count(self) -> int:
        return int(self.layout.counts.numel())


def _make_image_ids(step: int, batch_size: int, image_count: int) -> list[int]:
    if image_count <= 0:
        raise RuntimeError("manifest has no images")
    count = min(batch_size, image_count)
    start = (step * count) % image_count
    return [int((start + index) % image_count) for index in range(count)]


def _read_batch(
    reader: Any,
    image_ids: list[int],
    crop: tuple[int, int, int, int] | None,
    dct_coeffs: str,
    cache_capacity_mib: int,
    output_layout: str,
    grid_transform_spec: dict[str, Any] | None = None,
    prefetch: Any | None = None,
) -> DirectDctBatchView:
    if prefetch is None:
        batch = reader.read_batch(
            image_ids,
            crop=crop,
            dct_coeffs=dct_coeffs,
            cache_capacity_mib=cache_capacity_mib,
            layout=output_layout,
            grid_transform=grid_transform_spec,
        )
    else:
        batch = reader.read_prefetched(prefetch)
    if batch.layout != output_layout:
        raise RuntimeError(f"unexpected GALP output layout: got {batch.layout}, expected {output_layout}")

    if output_layout in ("ycbcr_dct_grid", "transformed_dct_grid"):
        y = batch.y
        cbcr = batch.cbcr
        if not y.is_cuda or not cbcr.is_cuda:
            raise RuntimeError("expected CUDA Y/CbCr grid tensors from GALP Direct-DCT runtime")
        if y.dtype != torch.int16 or cbcr.dtype != torch.int16:
            raise RuntimeError(f"expected torch.int16 Y/CbCr tensors, got y={y.dtype} cbcr={cbcr.dtype}")
        if y.ndim != 6 or cbcr.ndim != 6:
            raise RuntimeError(f"expected rank-6 Y/CbCr tensors, got y={y.ndim} cbcr={cbcr.ndim}")
        if y.shape[0] != len(image_ids) or y.shape[1] != 1 or tuple(y.shape[-2:]) != (8, 8):
            raise RuntimeError(f"unexpected Y grid shape: {tuple(y.shape)}")
        if cbcr.shape[0] != len(image_ids) or cbcr.shape[1] != 2 or tuple(cbcr.shape[-2:]) != (8, 8):
            raise RuntimeError(f"unexpected CbCr grid shape: {tuple(cbcr.shape)}")
        if y.data_ptr() != batch.y_device_data_ptr:
            raise RuntimeError(
                "expected Y tensor to wrap GALP CUDA buffer: "
                f"tensor=0x{y.data_ptr():x} galp=0x{batch.y_device_data_ptr:x}"
            )
        if cbcr.data_ptr() != batch.cbcr_device_data_ptr:
            raise RuntimeError(
                "expected CbCr tensor to wrap GALP CUDA buffer: "
                f"tensor=0x{cbcr.data_ptr():x} galp=0x{batch.cbcr_device_data_ptr:x}"
            )
        if batch.y_coefficient_count != y.numel() or batch.cbcr_coefficient_count != cbcr.numel():
            raise RuntimeError(
                "Y/CbCr coefficient count mismatch: "
                f"batch=({batch.y_coefficient_count}, {batch.cbcr_coefficient_count}) "
                f"tensor=({y.numel()}, {cbcr.numel()})"
            )
        coefficients = torch.cat((y.reshape(y.shape[0], -1), cbcr.reshape(cbcr.shape[0], -1)), dim=1)
        image_indices = torch.arange(len(image_ids), device=coefficients.device, dtype=torch.long)
        layout = ImageBlockLayout(
            offsets=image_indices,
            counts=torch.ones((len(image_ids),), device=coefficients.device, dtype=torch.long),
            block_to_image=image_indices,
        )
        view = DirectDctBatchView(
            coefficients=coefficients,
            layout=layout,
            block_count=int(coefficients.shape[0]),
            coefficients_per_block=int(coefficients.shape[1]),
            data_ptr=coefficients.data_ptr(),
            galp_ptr=batch.y_device_data_ptr,
            tensor_is_galp_backed=False,
            stats=batch.execution_stats,
            cache_stats=batch.cache_stats,
            output_layout=output_layout,
            selected_coefficients=list(batch.selected_coefficients),
            y=y,
            cbcr=cbcr,
            y_ptr=batch.y_device_data_ptr,
            cbcr_ptr=batch.cbcr_device_data_ptr,
        )
        del batch
        return view

    if output_layout != "compact":
        raise ValueError(f"unsupported output layout in demo: {output_layout}")

    coefficients = batch.coefficients
    if not coefficients.is_cuda:
        raise RuntimeError("expected CUDA tensor from GALP Direct-DCT runtime")
    if coefficients.dtype != torch.int16:
        raise RuntimeError(f"expected torch.int16 coefficients, got {coefficients.dtype}")
    if coefficients.ndim != 2:
        raise RuntimeError(f"expected rank-2 [block, coeff] tensor, got ndim={coefficients.ndim}")
    if tuple(coefficients.shape) != (batch.block_count, batch.coefficients_per_block):
        raise RuntimeError(
            "unexpected tensor shape: "
            f"shape={tuple(coefficients.shape)} expected={(batch.block_count, batch.coefficients_per_block)}"
        )
    if tuple(coefficients.stride()) != (batch.coefficients_per_block, 1):
        raise RuntimeError(f"expected compact [block, coeff] strides, got {tuple(coefficients.stride())}")
    if coefficients.data_ptr() != batch.device_data_ptr:
        raise RuntimeError(
            "expected torch tensor to wrap GALP CUDA buffer: "
            f"tensor=0x{coefficients.data_ptr():x} galp=0x{batch.device_data_ptr:x}"
        )
    if batch.coefficient_count != coefficients.numel():
        raise RuntimeError(f"coefficient count mismatch: batch={batch.coefficient_count} tensor={coefficients.numel()}")

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
        raise RuntimeError("image metadata count does not match requested image ids")
    if image_counts.numel() != len(image_ids):
        raise RuntimeError("image count tensor length does not match requested image ids")
    if block_to_image.numel() != batch.block_count:
        raise RuntimeError("block metadata count does not match tensor rows")
    if len(batch.selected_coefficients) != batch.coefficients_per_block:
        raise RuntimeError("selected coefficient count does not match tensor columns")

    layout = ImageBlockLayout(offsets=image_offsets, counts=image_counts, block_to_image=block_to_image)
    if int(layout.counts.sum().item()) != batch.block_count:
        raise RuntimeError(
            f"layout block counts sum to {int(layout.counts.sum().item())}, expected {batch.block_count}"
        )
    invalid_layout = torch.any((layout.offsets < 0) | (layout.counts < 0) | (layout.offsets + layout.counts > batch.block_count))
    if bool(invalid_layout.item()):
        raise RuntimeError(f"invalid image layout for blocks={batch.block_count}")

    view = DirectDctBatchView(
        coefficients=coefficients,
        layout=layout,
        block_count=batch.block_count,
        coefficients_per_block=batch.coefficients_per_block,
        data_ptr=coefficients.data_ptr(),
        galp_ptr=batch.device_data_ptr,
        tensor_is_galp_backed=True,
        stats=batch.execution_stats,
        cache_stats=batch.cache_stats,
        output_layout=output_layout,
        selected_coefficients=list(batch.selected_coefficients),
    )
    del batch
    return view


def _prefetch_batch(
    reader: Any,
    image_ids: list[int],
    crop: tuple[int, int, int, int] | None,
    dct_coeffs: str,
    cache_capacity_mib: int,
    output_layout: str,
    grid_transform_spec: dict[str, Any] | None = None,
) -> Any:
    return reader.prefetch_batch(
        image_ids,
        crop=crop,
        dct_coeffs=dct_coeffs,
        cache_capacity_mib=cache_capacity_mib,
        layout=output_layout,
        grid_transform=grid_transform_spec,
    )


def _build_rgbnomore_val_crop_transform(rgbnomore_root: Path) -> nn.Module:
    root = str(rgbnomore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    ctrans = importlib.import_module("utils.custom_transforms")
    return ctrans.ResizedCenterCrop_DCT(32, 28)


def _empty_execution_stats() -> dict[str, int]:
    return {
        "selected_vector_count": 0,
        "full_vector_count": 0,
        "decode_kernel_launch_count": 0,
        "rowgroup_count": 0,
        "workset_count": 0,
        "projection_item_count": 0,
        "internal_sync_count": 0,
    }


def _sum_execution_stats(batches: list[DirectDctBatchView]) -> dict[str, int]:
    totals = _empty_execution_stats()
    for batch in batches:
        for key in totals:
            totals[key] += int(batch.stats[key])
    return totals


def _read_rgbnomore_val_crop_batch(
    reader: Any,
    image_ids: list[int],
    cache_capacity_mib: int,
    transform: nn.Module,
) -> DirectDctBatchView:
    y_items = []
    cbcr_items = []
    source_batches = []
    for image_id in image_ids:
        source = _read_batch(
            reader=reader,
            image_ids=[int(image_id)],
            crop=None,
            dct_coeffs="all",
            cache_capacity_mib=cache_capacity_mib,
            output_layout="ycbcr_dct_grid",
            grid_transform_spec=None,
        )
        if source.y is None or source.cbcr is None:
            raise RuntimeError("RGB-no-more DCT val crop requires Y/CbCr grid tensors")
        transformed_y, transformed_cbcr = transform((source.y[0], source.cbcr[0]))
        y_items.append(transformed_y)
        cbcr_items.append(transformed_cbcr)
        source_batches.append(source)

    y = torch.stack(y_items, dim=0).contiguous()
    cbcr = torch.stack(cbcr_items, dim=0).contiguous()
    if tuple(y.shape[1:]) != (1, 28, 28, 8, 8):
        raise RuntimeError(f"expected RGB-no-more Y grid shape tail (1,28,28,8,8), got {tuple(y.shape)}")
    if tuple(cbcr.shape[1:]) != (2, 14, 14, 8, 8):
        raise RuntimeError(f"expected RGB-no-more CbCr grid shape tail (2,14,14,8,8), got {tuple(cbcr.shape)}")

    coefficients = torch.cat((y.reshape(y.shape[0], -1), cbcr.reshape(cbcr.shape[0], -1)), dim=1)
    image_indices = torch.arange(len(image_ids), device=coefficients.device, dtype=torch.long)
    layout = ImageBlockLayout(
        offsets=image_indices,
        counts=torch.ones((len(image_ids),), device=coefficients.device, dtype=torch.long),
        block_to_image=image_indices,
    )
    return DirectDctBatchView(
        coefficients=coefficients,
        layout=layout,
        block_count=int(coefficients.shape[0]),
        coefficients_per_block=int(coefficients.shape[1]),
        data_ptr=coefficients.data_ptr(),
        galp_ptr=source_batches[0].y_ptr if source_batches else 0,
        tensor_is_galp_backed=False,
        stats=_sum_execution_stats(source_batches),
        cache_stats={},
        output_layout="ycbcr_dct_grid",
        selected_coefficients=list(range(64)),
        y=y,
        cbcr=cbcr,
        y_ptr=y.data_ptr(),
        cbcr_ptr=cbcr.data_ptr(),
    )


def _read_demo_batch(
    reader: Any,
    image_ids: list[int],
    crop: tuple[int, int, int, int],
    args: argparse.Namespace,
    grid_transform: nn.Module | None,
) -> DirectDctBatchView:
    if args.grid_preprocess == "rgbnomore-val-crop":
        if grid_transform is None:
            raise RuntimeError("RGB-no-more val crop transform was not initialized")
        return _read_rgbnomore_val_crop_batch(reader, image_ids, args.cache_capacity_mib, grid_transform)
    transform_spec = RGBNOMORE_VAL_DCT_GRID_TRANSFORM if args.grid_preprocess == "rgbnomore-val-pushdown" else None
    return _read_batch(
        reader, image_ids, crop, args.dct_coeffs, args.cache_capacity_mib, args.output_layout, transform_spec
    )


def _mean_pool_by_layout(block_embeddings: torch.Tensor, layout: ImageBlockLayout) -> torch.Tensor:
    image_count = int(layout.counts.numel())
    sums = block_embeddings.new_zeros((image_count, block_embeddings.shape[-1]))
    if block_embeddings.shape[0] != 0:
        block_to_image = layout.block_to_image.to(device=block_embeddings.device, dtype=torch.long)
        index = block_to_image.unsqueeze(1).expand(-1, block_embeddings.shape[-1])
        sums.scatter_add_(0, index, block_embeddings)
    counts = layout.counts.to(device=block_embeddings.device, dtype=block_embeddings.dtype).clamp_min(1).unsqueeze(1)
    return sums / counts


class TinyDctMlp(nn.Module):
    def __init__(self, coefficients_per_block: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.block_projection = nn.Sequential(
            nn.Linear(coefficients_per_block, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, coefficients: torch.Tensor, layout: ImageBlockLayout) -> torch.Tensor:
        block_embeddings = self.block_projection(coefficients.to(torch.float32))
        image_features = _mean_pool_by_layout(block_embeddings, layout)
        return self.classifier(image_features)


def _sinusoidal_position_embedding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    angles = positions * frequencies.unsqueeze(0)
    embedding = torch.empty((length, dim), device=device, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(angles[:, : embedding[:, 0::2].shape[1]])
    embedding[:, 1::2] = torch.cos(angles[:, : embedding[:, 1::2].shape[1]])
    return embedding.to(dtype=dtype).unsqueeze(0)


class TinyDctVit(nn.Module):
    def __init__(self, coefficients_per_block: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        num_heads = 4 if hidden_dim % 4 == 0 else 1
        self.block_projection = nn.Linear(coefficients_per_block, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, coefficients: torch.Tensor, layout: ImageBlockLayout) -> torch.Tensor:
        block_embeddings = self.block_projection(coefficients.to(torch.float32))
        counts = layout.counts.to(device=block_embeddings.device, dtype=torch.long)
        offsets = layout.offsets.to(device=block_embeddings.device, dtype=torch.long)
        block_to_image = layout.block_to_image.to(device=block_embeddings.device, dtype=torch.long)
        batch_size = int(counts.numel())
        max_blocks = int(counts.max().item()) if batch_size > 0 else 0

        tokens = block_embeddings.new_zeros((batch_size, max_blocks, block_embeddings.shape[-1]))
        padding_mask = torch.ones((batch_size, max_blocks + 1), dtype=torch.bool, device=block_embeddings.device)
        padding_mask[:, 0] = False
        if block_embeddings.shape[0] != 0 and max_blocks > 0:
            repeated_offsets = torch.repeat_interleave(offsets, counts)
            block_positions = torch.arange(block_embeddings.shape[0], device=block_embeddings.device) - repeated_offsets
            tokens[block_to_image, block_positions] = block_embeddings
            positions = torch.arange(max_blocks, device=block_embeddings.device).unsqueeze(0)
            padding_mask[:, 1:] = positions >= counts.unsqueeze(1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        encoded = torch.cat((cls_tokens, tokens), dim=1)
        encoded = encoded + _sinusoidal_position_embedding(
            encoded.shape[1], encoded.shape[2], encoded.device, encoded.dtype
        )
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        return self.classifier(self.norm(encoded[:, 0]))


def _build_model(args: argparse.Namespace, first_batch: DirectDctBatchView) -> nn.Module:
    hidden_dim = 96 if args.model == "tiny-dct-vit" else 128
    if args.model == "tiny-dct-vit":
        return TinyDctVit(
            coefficients_per_block=first_batch.coefficients_per_block,
            hidden_dim=hidden_dim,
            num_classes=args.num_classes,
        )
    if args.model == "tiny-dct-mlp":
        return TinyDctMlp(
            coefficients_per_block=first_batch.coefficients_per_block,
            hidden_dim=hidden_dim,
            num_classes=args.num_classes,
        )
    raise ValueError(f"unknown model: {args.model}")


def _run_step(
    model: nn.Module,
    batch: DirectDctBatchView,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = (torch.arange(batch.image_count, dtype=torch.long, device=batch.coefficients.device) + step) % num_classes
    if optimizer is None:
        with torch.no_grad():
            logits = model(batch.coefficients, batch.layout)
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.coefficients, batch.layout)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return logits, loss.detach()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _accumulate_stats(totals: dict[str, int | float], batch: DirectDctBatchView) -> None:
    stats = batch.stats
    cache_stats = batch.cache_stats
    totals["selected_vectors"] += int(stats["selected_vector_count"])
    totals["full_vectors"] += int(stats["full_vector_count"])
    totals["decode_kernels"] += int(stats["decode_kernel_launch_count"])
    totals["rowgroups"] += int(stats["rowgroup_count"])
    totals["worksets"] += int(stats["workset_count"])
    totals["projection_items"] += int(stats["projection_item_count"])
    totals["internal_syncs"] += int(stats["internal_sync_count"])
    totals["cache_hits"] += int(cache_stats.get("hits", 0))
    totals["cache_misses"] += int(cache_stats.get("misses", 0))
    totals["cache_inserts"] += int(cache_stats.get("inserts", 0))
    totals["cache_evictions"] += int(cache_stats.get("evictions", 0))
    totals["cache_resident_rowgroups"] = max(
        totals["cache_resident_rowgroups"], int(cache_stats.get("resident_rowgroups", 0))
    )
    projection_item_build_seconds = float(stats.get("projection_item_build_ms", 0.0)) / 1000.0
    totals["planning_seconds"] += float(stats.get("planning_ms", 0.0)) / 1000.0
    totals["projection_build_seconds"] += projection_item_build_seconds
    totals["projection_item_build_seconds"] += projection_item_build_seconds
    totals["resize_weight_build_seconds"] += float(stats.get("resize_weight_build_ms", 0.0)) / 1000.0
    totals["gpu_projection_seconds"] += float(stats.get("projection_ms", 0.0)) / 1000.0
    totals["decoded_projection_seconds"] += float(stats.get("decoded_projection_ms", 0.0)) / 1000.0
    totals["fixed_transform_kernel_seconds"] += float(stats.get("fixed_transform_ms", 0.0)) / 1000.0
    totals["round_kernel_seconds"] += float(stats.get("fixed_grid_round_ms", 0.0)) / 1000.0


def _empty_totals() -> dict[str, int | float]:
    return {
        "selected_vectors": 0,
        "full_vectors": 0,
        "decode_kernels": 0,
        "rowgroups": 0,
        "worksets": 0,
        "projection_items": 0,
        "internal_syncs": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_inserts": 0,
        "cache_evictions": 0,
        "cache_resident_rowgroups": 0,
        "planning_seconds": 0.0,
        "projection_build_seconds": 0.0,
        "projection_item_build_seconds": 0.0,
        "resize_weight_build_seconds": 0.0,
        "gpu_projection_seconds": 0.0,
        "decoded_projection_seconds": 0.0,
        "fixed_transform_kernel_seconds": 0.0,
        "round_kernel_seconds": 0.0,
    }


def _cache_visibility(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_capacity_mib <= 0:
        return {"cache_active": False, "cache_disabled_reason": "cache_capacity_mib=0"}
    if args.output_layout == "ycbcr_dct_grid":
        return {"cache_active": False, "cache_disabled_reason": "dense cache disabled for Y/CbCr grid layouts"}
    if args.dct_coeffs != "all":
        return {"cache_active": False, "cache_disabled_reason": "dct_coeffs subset disables dense rowgroup cache"}
    return {"cache_active": True, "cache_disabled_reason": ""}


def _print_batch_line(phase: str, step: int, batch: DirectDctBatchView, logits: torch.Tensor | None = None,
                      loss: torch.Tensor | None = None) -> None:
    fields = [
        f"phase={phase}",
        f"step={step}",
        f"output_layout={batch.output_layout}",
        f"images={batch.image_count}",
        f"blocks={batch.block_count}",
        f"coefficients_per_block={batch.coefficients_per_block}",
        f"tensor_shape={tuple(batch.coefficients.shape)}",
        f"tensor_device={batch.coefficients.device}",
        f"tensor_dtype={batch.coefficients.dtype}",
        f"data_ptr=0x{batch.data_ptr:x}",
        f"galp_ptr=0x{batch.galp_ptr:x}",
        f"tensor_is_galp_backed={batch.tensor_is_galp_backed}",
    ]
    if batch.y is not None and batch.cbcr is not None:
        fields.extend([
            f"y_shape={tuple(batch.y.shape)}",
            f"cbcr_shape={tuple(batch.cbcr.shape)}",
            f"y_ptr=0x{batch.y_ptr:x}",
            f"cbcr_ptr=0x{batch.cbcr_ptr:x}",
            f"y_ptr_equal={batch.y.data_ptr() == batch.y_ptr}",
            f"cbcr_ptr_equal={batch.cbcr.data_ptr() == batch.cbcr_ptr}",
        ])
    if logits is not None:
        fields.extend([f"logits_shape={tuple(logits.shape)}", f"logits_device={logits.device}"])
    if loss is not None:
        fields.append(f"loss={float(loss):.6f}")
    fields.extend([
        f"selected_vectors={batch.stats['selected_vector_count']}",
        f"full_vectors={batch.stats['full_vector_count']}",
        f"decode_kernels={batch.stats['decode_kernel_launch_count']}",
    ])
    print(" ".join(fields))


def _run_loader_phase(reader: Any, args: argparse.Namespace, crop: tuple[int, int, int, int],
                      device: torch.device, grid_transform: nn.Module | None) -> dict[str, Any]:
    for warmup_step in range(args.warmup):
        image_ids = _make_image_ids(warmup_step, args.batch_size, reader.image_count)
        batch = _read_demo_batch(reader, image_ids, crop, args, grid_transform)
        _sync(batch.coefficients.device)
        del batch

    _sync(device)
    total_images = 0
    totals = _empty_totals()
    started = time.perf_counter()
    pending_ids: list[int] | None = None
    pending: Any | None = None
    if args.async_prefetch:
        pending_ids = _make_image_ids(0, args.batch_size, reader.image_count)
        pending = _prefetch_batch(
            reader,
            pending_ids,
            crop,
            args.dct_coeffs,
            args.cache_capacity_mib,
            args.output_layout,
            RGBNOMORE_VAL_DCT_GRID_TRANSFORM if args.grid_preprocess == "rgbnomore-val-pushdown" else None,
        )
    for step in range(args.steps):
        if args.async_prefetch:
            if pending_ids is None or pending is None:
                raise RuntimeError("async prefetch state was not initialized")
            image_ids = pending_ids
            batch = _read_batch(
                reader,
                image_ids,
                crop,
                args.dct_coeffs,
                args.cache_capacity_mib,
                args.output_layout,
                prefetch=pending,
            )
            if step + 1 < args.steps:
                pending_ids = _make_image_ids(step + 1, args.batch_size, reader.image_count)
                pending = _prefetch_batch(
                    reader,
                    pending_ids,
                    crop,
                    args.dct_coeffs,
                    args.cache_capacity_mib,
                    args.output_layout,
                    RGBNOMORE_VAL_DCT_GRID_TRANSFORM if args.grid_preprocess == "rgbnomore-val-pushdown" else None,
                )
            else:
                pending_ids = None
                pending = None
        else:
            image_ids = _make_image_ids(step, args.batch_size, reader.image_count)
            batch = _read_demo_batch(reader, image_ids, crop, args, grid_transform)
        total_images += batch.image_count
        _accumulate_stats(totals, batch)
        if step == 0 or (args.log_every > 0 and (step + 1) % args.log_every == 0):
            _print_batch_line("loader_to_device", step, batch)
        del batch
    _sync(device)
    elapsed = time.perf_counter() - started

    result = {
        "backend": "galp_direct_dct",
        "phase": "loader_to_device",
        "model": None,
        "images": total_images,
        "seconds": elapsed,
        "images_per_s": total_images / elapsed if elapsed > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "crop": list(crop),
        "dct_coeffs": args.dct_coeffs,
        "cache_capacity_mib": args.cache_capacity_mib,
        "output_layout": args.output_layout,
        "device": str(device),
        "async_prefetch": bool(args.async_prefetch),
        "grid_preprocess": args.grid_preprocess,
        **_cache_visibility(args),
        **totals,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def _run_train_phase(reader: Any, args: argparse.Namespace, crop: tuple[int, int, int, int],
                     model: nn.Module, optimizer: torch.optim.Optimizer | None,
                     device: torch.device, grid_transform: nn.Module | None) -> dict[str, Any]:
    phase = "train_step" if optimizer is not None else "forward_step"
    for warmup_step in range(args.warmup):
        image_ids = _make_image_ids(warmup_step, args.batch_size, reader.image_count)
        batch = _read_demo_batch(reader, image_ids, crop, args, grid_transform)
        logits, loss = _run_step(model, batch, optimizer, warmup_step, args.num_classes)
        if not logits.is_cuda:
            raise RuntimeError("logits left CUDA")
        _sync(batch.coefficients.device)
        del logits
        del loss
        del batch

    _sync(device)
    total_images = 0
    final_loss = 0.0
    totals = _empty_totals()
    started = time.perf_counter()
    pending_ids: list[int] | None = None
    pending: Any | None = None
    if args.async_prefetch:
        pending_ids = _make_image_ids(0, args.batch_size, reader.image_count)
        pending = _prefetch_batch(
            reader,
            pending_ids,
            crop,
            args.dct_coeffs,
            args.cache_capacity_mib,
            args.output_layout,
            RGBNOMORE_VAL_DCT_GRID_TRANSFORM if args.grid_preprocess == "rgbnomore-val-pushdown" else None,
        )
    for step in range(args.steps):
        if args.async_prefetch:
            if pending_ids is None or pending is None:
                raise RuntimeError("async prefetch state was not initialized")
            image_ids = pending_ids
            batch = _read_batch(
                reader,
                image_ids,
                crop,
                args.dct_coeffs,
                args.cache_capacity_mib,
                args.output_layout,
                prefetch=pending,
            )
            if step + 1 < args.steps:
                pending_ids = _make_image_ids(step + 1, args.batch_size, reader.image_count)
                pending = _prefetch_batch(
                    reader,
                    pending_ids,
                    crop,
                    args.dct_coeffs,
                    args.cache_capacity_mib,
                    args.output_layout,
                    RGBNOMORE_VAL_DCT_GRID_TRANSFORM if args.grid_preprocess == "rgbnomore-val-pushdown" else None,
                )
            else:
                pending_ids = None
                pending = None
        else:
            image_ids = _make_image_ids(step, args.batch_size, reader.image_count)
            batch = _read_demo_batch(reader, image_ids, crop, args, grid_transform)
        logits, loss = _run_step(model, batch, optimizer, step, args.num_classes)
        if not logits.is_cuda:
            raise RuntimeError("logits left CUDA")
        total_images += batch.image_count
        final_loss = float(loss)
        _accumulate_stats(totals, batch)
        if step == 0 or (args.log_every > 0 and (step + 1) % args.log_every == 0):
            _print_batch_line(phase, step, batch, logits=logits, loss=loss)
        del logits
        del loss
        del batch
    _sync(device)
    elapsed = time.perf_counter() - started

    result = {
        "backend": "galp_direct_dct",
        "phase": phase,
        "model": args.model,
        "images": total_images,
        "seconds": elapsed,
        "images_per_s": total_images / elapsed if elapsed > 0.0 else float("inf"),
        "final_loss": final_loss,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "crop": list(crop),
        "dct_coeffs": args.dct_coeffs,
        "cache_capacity_mib": args.cache_capacity_mib,
        "output_layout": args.output_layout,
        "device": str(device),
        "train_smoke": bool(args.train_smoke),
        "async_prefetch": bool(args.async_prefetch),
        "grid_preprocess": args.grid_preprocess,
        **_cache_visibility(args),
        **totals,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end GALP stay-on-GPU Direct-DCT PyTorch demo")
    parser.add_argument("manifest")
    parser.add_argument("--phase", choices=("loader", "train", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop", type=int, nargs=4, default=(0, 0, 64, 64), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--dct-coeffs", default="first:8")
    parser.add_argument(
        "--cache-capacity-mib",
        type=int,
        default=int(getattr(galp_dct, "DEFAULT_CACHE_CAPACITY_MIB", 1024)),
    )
    parser.add_argument("--output-layout", choices=("compact", "ycbcr_dct_grid", "transformed_dct_grid"), default="compact")
    parser.add_argument(
        "--grid-preprocess",
        choices=("none", "rgbnomore-val-crop", "rgbnomore-val-pushdown"),
        default="none",
        help="Optional Y/CbCr grid postprocess. rgbnomore-val-crop applies RGB-no-more DCT val center-crop+resize.",
    )
    parser.add_argument("--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more"))
    parser.add_argument("--model", choices=("tiny-dct-vit", "tiny-dct-mlp"), default="tiny-dct-vit")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--train-smoke", action="store_true")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=0,
                        help="Print a sample batch line every N measured steps; 0 prints only step 0.")
    parser.add_argument("--no-async-prefetch", action="store_false", dest="async_prefetch")
    parser.set_defaults(async_prefetch=True)
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
    if args.num_classes <= 1:
        raise ValueError("--num-classes must be greater than 1")
    if args.grid_preprocess != "none":
        if args.output_layout not in ("ycbcr_dct_grid", "transformed_dct_grid"):
            raise ValueError("--grid-preprocess requires a Y/CbCr grid output layout")
        if not args.rgbnomore_root.exists():
            raise FileNotFoundError(args.rgbnomore_root)
        args.dct_coeffs = "all"
        if args.grid_preprocess == "rgbnomore-val-pushdown":
            args.output_layout = "transformed_dct_grid"
        if args.async_prefetch and args.grid_preprocess != "rgbnomore-val-pushdown":
            print("grid_preprocess disables async_prefetch because it reads full-image grids per image before stacking")
            args.async_prefetch = False
    if not torch.cuda.is_available():
        raise RuntimeError("this Direct-DCT end-to-end demo requires torch.cuda.is_available()")

    crop = tuple(int(value) for value in args.crop)
    setup_reader = galp_dct.DirectDctReader(args.manifest)
    grid_transform = (
        _build_rgbnomore_val_crop_transform(args.rgbnomore_root)
        if args.grid_preprocess == "rgbnomore-val-crop"
        else None
    )
    first_ids = _make_image_ids(0, args.batch_size, setup_reader.image_count)
    first_batch = _read_demo_batch(setup_reader, first_ids, crop, args, grid_transform)
    device = first_batch.coefficients.device

    model = _build_model(args, first_batch).to(first_batch.coefficients.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if args.train_smoke else None
    if optimizer is None:
        model.eval()
    else:
        model.train()

    print(
        f"model={args.model} train_smoke={args.train_smoke} crop={crop} dct_coeffs={args.dct_coeffs} "
        f"output_layout={args.output_layout} "
        f"grid_preprocess={args.grid_preprocess} "
        f"cache_capacity_mib={args.cache_capacity_mib} "
        f"async_prefetch={args.async_prefetch} "
        f"phase={args.phase} warmup={args.warmup} augmentation=none-crop-only"
    )

    del first_batch
    del setup_reader
    _sync(device)

    results: list[dict[str, Any]] = []
    if args.phase in ("loader", "both"):
        loader_reader = galp_dct.DirectDctReader(args.manifest)
        results.append(_run_loader_phase(loader_reader, args, crop, device, grid_transform))
        del loader_reader
        _sync(device)
    if args.phase in ("train", "both"):
        train_reader = galp_dct.DirectDctReader(args.manifest)
        results.append(_run_train_phase(train_reader, args, crop, model, optimizer, device, grid_transform))
        del train_reader
        _sync(device)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
