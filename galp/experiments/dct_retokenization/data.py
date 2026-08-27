#!/usr/bin/env python3
"""Raw-JPEG DCT input path with masking before every frequency-mixing transform."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


from galp.experiments.coefficient_mask_evaluator.masks import ZIGZAG_NATURAL_INDICES


@dataclass(frozen=True)
class ImageNetSample:
    ordinal: int
    logical_sample_id: str
    path: str
    label: int
    width: int
    height: int
    jpeg_sampling: str


def load_manifest(path: Path, *, max_samples: int | None = None) -> tuple[dict[str, Any], list[ImageNetSample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"manifest contains no samples: {path}")
    if max_samples is not None:
        raw = raw[: int(max_samples)]
    samples: list[ImageNetSample] = []
    for ordinal, value in enumerate(raw):
        sample = ImageNetSample(
            ordinal=ordinal,
            logical_sample_id=str(value["logical_sample_id"]),
            path=str(value["path"]),
            label=int(value["label"]),
            width=int(value["width"]),
            height=int(value["height"]),
            jpeg_sampling=str(value["jpeg_sampling"]),
        )
        if (sample.width, sample.height, sample.jpeg_sampling) != (512, 512, "4:2:0"):
            raise ValueError(f"unsupported sample geometry: {sample}")
        if not 0 <= sample.label < 1000:
            raise ValueError(f"invalid ImageNet label: {sample}")
        samples.append(sample)
    return payload, samples


def zigzag_mask(k: int) -> torch.Tensor:
    if not 1 <= int(k) <= 64:
        raise ValueError("K must be in [1,64]")
    mask = torch.zeros((8, 8), dtype=torch.bool)
    mask.reshape(-1)[list(ZIGZAG_NATURAL_INDICES[: int(k)])] = True
    return mask


_DCT_MANIP: Any | None = None
_DCT_OPS: Any | None = None


def load_rgbnomore_modules(root: Path) -> tuple[Any, Any]:
    global _DCT_MANIP, _DCT_OPS
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    if _DCT_MANIP is None:
        _DCT_MANIP = importlib.import_module("dct_manip")
    if _DCT_OPS is None:
        _DCT_OPS = importlib.import_module("utils.dct_ops")
    return _DCT_MANIP, _DCT_OPS


class RawDctDataset(torch.utils.data.Dataset):
    """Decode only; return untouched quantized coefficient tensors."""

    def __init__(
        self,
        samples: Sequence[ImageNetSample],
        rgbnomore_root: Path,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self.samples = samples
        self.rgbnomore_root = rgbnomore_root
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples) if self.indices is None else int(self.indices.size)

    def __getitem__(self, index: int):
        ordinal = int(index) if self.indices is None else int(self.indices[index])
        sample = self.samples[ordinal]
        dct_manip, _ = load_rgbnomore_modules(self.rgbnomore_root)
        dimensions, quantization, y, cbcr = dct_manip.read_coefficients(sample.path)
        if tuple(y.shape) != (1, 64, 64, 8, 8):
            raise RuntimeError(f"unexpected Y shape {tuple(y.shape)} for {sample.path}")
        if cbcr is None or tuple(cbcr.shape) != (2, 32, 32, 8, 8):
            observed = None if cbcr is None else tuple(cbcr.shape)
            raise RuntimeError(f"unexpected CbCr shape {observed} for {sample.path}")
        if tuple(quantization.shape) != (3, 8, 8):
            raise RuntimeError(f"unexpected quantization shape {tuple(quantization.shape)}")
        expected = ((512, 512), (256, 256), (256, 256))
        observed_dimensions = tuple(tuple(int(item) for item in row) for row in dimensions.tolist())
        if observed_dimensions != expected:
            raise RuntimeError(f"unexpected component dimensions {observed_dimensions}")
        return y, cbcr, quantization, sample.label, ordinal


def worker_init(_: int) -> None:
    torch.set_num_threads(1)


def make_loader(
    samples: Sequence[ImageNetSample],
    rgbnomore_root: Path,
    *,
    indices: Sequence[int] | np.ndarray | None,
    batch_size: int,
    workers: int,
) -> torch.utils.data.DataLoader:
    dataset = RawDctDataset(samples, rgbnomore_root, indices)
    arguments: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(workers),
        "pin_memory": True,
        "worker_init_fn": worker_init,
    }
    if workers > 0:
        arguments.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(dataset, **arguments)


def epoch_permutation(sample_count: int, seed: int, epoch: int) -> np.ndarray:
    sequence = np.random.SeedSequence((int(seed), int(epoch), 0xD07C0EFF))
    return np.random.default_rng(sequence).permutation(sample_count).astype(np.int64, copy=False)


class MaskedDctPreprocessor:
    """Mask raw quantized coefficients, then execute DCT crop/resize on device."""

    def __init__(
        self,
        rgbnomore_root: Path,
        device: torch.device,
        k: int,
    ) -> None:
        _, self.dct_ops = load_rgbnomore_modules(rgbnomore_root)
        self.device = device
        self.k = int(k)
        self.mask = zigzag_mask(k).to(device)
        self.conversion_matrices: dict[int, torch.Tensor] = {}

    def _mask_and_dequantize(
        self,
        y_quantized: torch.Tensor,
        cbcr_quantized: torch.Tensor,
        quantization: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_quantized = y_quantized.to(self.device, non_blocking=True)
        cbcr_quantized = cbcr_quantized.to(self.device, non_blocking=True)
        quantization = quantization.to(self.device, non_blocking=True)
        mask = self.mask.reshape(1, 1, 1, 1, 8, 8)
        # Experimental contract: this multiplication is before dequantization,
        # crop/resize frequency mixing, position, and patch projection.
        y_quantized = y_quantized * mask
        cbcr_quantized = cbcr_quantized * mask
        y = torch.clamp(
            y_quantized * quantization[:, 0].reshape(-1, 1, 1, 1, 8, 8),
            min=-1024,
            max=1016,
        )
        cbcr = torch.clamp(
            cbcr_quantized * quantization[:, 1:3].reshape(-1, 2, 1, 1, 8, 8),
            min=-1024,
            max=1016,
        )
        return y, cbcr

    def validation(
        self,
        y_quantized: torch.Tensor,
        cbcr_quantized: torch.Tensor,
        quantization: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, cbcr = self._mask_and_dequantize(y_quantized, cbcr_quantized, quantization)
        batch = int(y.shape[0])
        y = y[:, :, 4:60, 4:60].reshape(batch, 56, 56, 8, 8)
        cbcr = cbcr[:, :, 2:30, 2:30].reshape(batch * 2, 28, 28, 8, 8)
        y = self.dct_ops.resize_dct(
            y, 28, dtype=torch.float32, conv_mxs=self.conversion_matrices
        ).reshape(batch, 1, 28, 28, 8, 8)
        cbcr = self.dct_ops.resize_dct(
            cbcr, 14, dtype=torch.float32, conv_mxs=self.conversion_matrices
        ).reshape(batch, 2, 14, 14, 8, 8)
        return (y.float() + 4.0) / 1020.0, (cbcr.float() + 4.0) / 1020.0

    def training(
        self,
        y_quantized: torch.Tensor,
        cbcr_quantized: torch.Tensor,
        quantization: torch.Tensor,
        decisions: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, cbcr = self._mask_and_dequantize(y_quantized, cbcr_quantized, quantization)
        batch = int(y.shape[0])
        if len(decisions) != batch:
            raise ValueError("augmentation decisions do not match batch cardinality")
        output_y = torch.empty((batch, 1, 28, 28, 8, 8), dtype=y.dtype, device=self.device)
        output_cbcr = torch.empty(
            (batch, 2, 14, 14, 8, 8), dtype=cbcr.dtype, device=self.device
        )
        groups: dict[tuple[int, int], list[int]] = {}
        for index, decision in enumerate(decisions):
            height = int(decision.crop_height) // 8
            width = int(decision.crop_width) // 8
            groups.setdefault((height, width), []).append(index)
        for (height, width), indices in groups.items():
            y_crops = []
            cbcr_crops = []
            for index in indices:
                decision = decisions[index]
                top = int(decision.crop_y) // 8
                left = int(decision.crop_x) // 8
                y_crops.append(y[index, :, top : top + height, left : left + width])
                cb_top, cb_left = top // 2, left // 2
                cb_height, cb_width = height // 2, width // 2
                cbcr_crops.append(
                    cbcr[
                        index,
                        :,
                        cb_top : cb_top + cb_height,
                        cb_left : cb_left + cb_width,
                    ]
                )
            selected_y = torch.cat(y_crops, dim=0)
            selected_cbcr = torch.cat(cbcr_crops, dim=0)
            resized_y = self.dct_ops.resize_dct(
                selected_y,
                28,
                dtype=torch.float32,
                conv_mxs=self.conversion_matrices,
            )
            resized_cbcr = self.dct_ops.resize_dct(
                selected_cbcr,
                14,
                dtype=torch.float32,
                conv_mxs=self.conversion_matrices,
            ).reshape(len(indices), 2, 14, 14, 8, 8)
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            output_y[index_tensor, 0] = resized_y
            output_cbcr[index_tensor] = resized_cbcr

        flip_indices = [index for index, decision in enumerate(decisions) if decision.horizontal_flip]
        if flip_indices:
            index_tensor = torch.as_tensor(flip_indices, dtype=torch.long, device=self.device)
            selected_y = output_y.index_select(0, index_tensor).flip(dims=(3,))
            selected_cbcr = output_cbcr.index_select(0, index_tensor).flip(dims=(3,))
            selected_y[..., 1::2] *= -1
            selected_cbcr[..., 1::2] *= -1
            output_y.index_copy_(0, index_tensor, selected_y)
            output_cbcr.index_copy_(0, index_tensor, selected_cbcr)
        return (output_y.float() + 4.0) / 1020.0, (output_cbcr.float() + 4.0) / 1020.0

    def audit_record(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "mask_application_stage": "raw_quantized_coefficients_immediately_after_read_coefficients",
            "stages_after_mask": [
                "quant_table_dequantization",
                "clamp",
                "crop_resize_frequency_mixing",
                "augmentation",
                "normalization",
                "patch_projection",
            ],
            "retained_natural_indices": list(ZIGZAG_NATURAL_INDICES[: self.k]),
        }
