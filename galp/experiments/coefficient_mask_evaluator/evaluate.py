#!/usr/bin/env python3
"""Evaluate raw 8x8 DCT coefficient masks with an unchanged K=64 model.

The mask is multiplied into the quantized tensors returned by
``dct_manip.read_coefficients``.  Only after that do we dequantize, clamp,
perform RGB-no-more's DCT-domain resized center crop, normalize, and call the
unmodified model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from galp.experiments.coefficient_mask_evaluator.masks import (
    MaskCondition,
    build_conditions,
    masks_numpy,
    select_conditions,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
)

_DCT_MANIP: Any | None = None
_DCT_OPS: Any | None = None
_RGBNOMORE_DATASETS: Any | None = None


def _load_rgbnomore_modules(root: Path) -> tuple[Any, Any, Any]:
    global _DCT_MANIP, _DCT_OPS, _RGBNOMORE_DATASETS
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    # torch must be imported before the extension so libc10 is already loaded.
    if _DCT_MANIP is None:
        _DCT_MANIP = importlib.import_module("dct_manip")
    if _DCT_OPS is None:
        _DCT_OPS = importlib.import_module("utils.dct_ops")
    if _RGBNOMORE_DATASETS is None:
        _RGBNOMORE_DATASETS = importlib.import_module("datasets")
    return _DCT_MANIP, _DCT_OPS, _RGBNOMORE_DATASETS


def sha256_file(path: Path, chunk_bytes: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_fingerprints(
    rgbnomore_root: Path, dct_manip_runtime: Path
) -> dict[str, dict[str, str]]:
    root = rgbnomore_root.resolve()
    sources = {
        "evaluator/evaluate.py": Path(__file__).resolve(),
        "evaluator/masks.py": (HERE / "masks.py").resolve(),
        "rgbnomore/datasets.py": (root / "datasets.py").resolve(),
        "rgbnomore/models/plainvit.py": (root / "models/plainvit.py").resolve(),
        "rgbnomore/utils/custom_transforms.py": (
            root / "utils/custom_transforms.py"
        ).resolve(),
        "rgbnomore/utils/dct_ops.py": (root / "utils/dct_ops.py").resolve(),
        "rgbnomore/utils/dct_torch_utils.py": (
            root / "utils/dct_torch_utils.py"
        ).resolve(),
        "rgbnomore/dct_manip/dct_manip.cpp": (
            root / "dct_manip/dct_manip.cpp"
        ).resolve(),
        "runtime/dct_manip": dct_manip_runtime.resolve(),
    }
    fingerprints: dict[str, dict[str, str]] = {}
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"output-affecting source is missing: {path}")
        fingerprints[name] = {"path": str(path), "sha256": sha256_file(path)}
    return fingerprints


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def command_output(arguments: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(arguments), check=False, capture_output=True, text=True, timeout=20
        )
        return {
            "command": list(arguments),
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(arguments), "error": str(error)}


def hardware_snapshot() -> dict[str, Any]:
    meminfo: dict[str, str] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "Cached", "Dirty"}:
                meminfo[key] = value.strip()
    except OSError as error:
        meminfo["error"] = str(error)
    return {
        "captured_at_unix_ns": time.time_ns(),
        "hostname": platform.node(),
        "loadavg": os.getloadavg(),
        "meminfo": meminfo,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_gpus": command_output(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            )
        ),
        "nvidia_smi_processes": command_output(
            (
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        ),
    }


def load_samples(manifest: Path, max_samples: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with manifest.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"manifest has no samples: {manifest}")
    samples = list(raw_samples if max_samples is None else raw_samples[:max_samples])
    for ordinal, sample in enumerate(samples):
        required = ("path", "label", "logical_sample_id", "width", "height", "jpeg_sampling")
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"sample {ordinal} misses fields {missing}")
        if int(sample["width"]) != 512 or int(sample["height"]) != 512:
            raise ValueError(f"sample {ordinal} is not 512x512: {sample['path']}")
        if sample["jpeg_sampling"] != "4:2:0":
            raise ValueError(f"sample {ordinal} is not 4:2:0: {sample['path']}")
        label = int(sample["label"])
        if label < 0 or label >= 1000:
            raise ValueError(f"sample {ordinal} has invalid ImageNet label {label}")
    return payload, samples


class RawDctDataset(torch.utils.data.Dataset):
    """Return untouched quantized DCT coefficients and their quant tables."""

    def __init__(self, samples: Sequence[dict[str, Any]], rgbnomore_root: Path, start: int = 0) -> None:
        self.samples = samples
        self.rgbnomore_root = rgbnomore_root
        self.start = int(start)

    def __len__(self) -> int:
        return len(self.samples) - self.start

    def __getitem__(self, relative_index: int):
        ordinal = self.start + int(relative_index)
        sample = self.samples[ordinal]
        dct_manip, _, _ = _load_rgbnomore_modules(self.rgbnomore_root)
        dimensions, quant, y_quantized, cbcr_quantized = dct_manip.read_coefficients(sample["path"])
        if tuple(y_quantized.shape) != (1, 64, 64, 8, 8):
            raise RuntimeError(
                f"unexpected Y shape {tuple(y_quantized.shape)} for sample {ordinal}: {sample['path']}"
            )
        if cbcr_quantized is None or tuple(cbcr_quantized.shape) != (2, 32, 32, 8, 8):
            observed = None if cbcr_quantized is None else tuple(cbcr_quantized.shape)
            raise RuntimeError(
                f"unexpected CbCr shape {observed} for sample {ordinal}: {sample['path']}"
            )
        if tuple(quant.shape) != (3, 8, 8):
            raise RuntimeError(f"unexpected quant-table shape {tuple(quant.shape)}")
        expected_dimensions = ((512, 512), (256, 256), (256, 256))
        if tuple(tuple(int(value) for value in row) for row in dimensions.tolist()) != expected_dimensions:
            raise RuntimeError(f"unexpected component dimensions for sample {ordinal}: {dimensions.tolist()}")
        return y_quantized, cbcr_quantized, quant, int(sample["label"]), ordinal


def worker_init(_: int) -> None:
    torch.set_num_threads(1)


@dataclass
class PreprocessAudit:
    mask_application_stage: str = "immediately_after_dct_manip.read_coefficients"
    input_domain: str = "raw_quantized_jpeg_8x8_coefficients"
    stages_after_mask: tuple[str, ...] = (
        "jpeg_quant_table_dequantization",
        "clamp_to_minus1024_plus1016",
        "dct_resized_center_crop_frequency_mixing",
        "range_normalization",
        "unchanged_model_forward",
    )
    y_raw_shape: tuple[int, ...] = (1, 64, 64, 8, 8)
    cbcr_raw_shape: tuple[int, ...] = (2, 32, 32, 8, 8)
    y_crop: tuple[int, int, int, int] = (4, 4, 56, 56)
    cbcr_crop: tuple[int, int, int, int] = (2, 2, 28, 28)
    y_output_blocks: tuple[int, int] = (28, 28)
    cbcr_output_blocks: tuple[int, int] = (14, 14)


class BatchedDctPreprocessor:
    """Vectorize the unchanged RGB-no-more validation transform over masks."""

    def __init__(self, rgbnomore_root: Path, device: torch.device) -> None:
        _, self.dct_ops, _ = _load_rgbnomore_modules(rgbnomore_root)
        self.device = device
        # RGB-no-more's cache is keyed only by scale.  Keep one cache per device.
        self.conversion_matrices: dict[int, torch.Tensor] = {}

    def __call__(
        self,
        y_quantized: torch.Tensor,
        cbcr_quantized: torch.Tensor,
        quant: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_quantized = y_quantized.to(self.device, non_blocking=True)
        cbcr_quantized = cbcr_quantized.to(self.device, non_blocking=True)
        quant = quant.to(self.device, non_blocking=True)
        masks = masks.to(self.device, dtype=torch.bool, non_blocking=True)

        condition_count = int(masks.shape[0])
        batch_size = int(y_quantized.shape[0])
        mask_view = masks.reshape(condition_count, 1, 1, 1, 1, 8, 8)

        # This is the defining placement of the experiment: masking happens to
        # read_coefficients() output, before dequantization or frequency mixing.
        masked_y_quantized = y_quantized.unsqueeze(0) * mask_view
        masked_cbcr_quantized = cbcr_quantized.unsqueeze(0) * mask_view

        y_quant_view = quant[:, 0].reshape(1, batch_size, 1, 1, 1, 8, 8)
        cbcr_quant_view = quant[:, 1:3].reshape(1, batch_size, 2, 1, 1, 8, 8)
        y = torch.clamp(masked_y_quantized * y_quant_view, min=-1024, max=1016)
        cbcr = torch.clamp(masked_cbcr_quantized * cbcr_quant_view, min=-1024, max=1016)

        y = y[:, :, :, 4:60, 4:60, :, :].reshape(
            condition_count * batch_size, 56, 56, 8, 8
        )
        cbcr = cbcr[:, :, :, 2:30, 2:30, :, :].reshape(
            condition_count * batch_size * 2, 28, 28, 8, 8
        )

        y = self.dct_ops.resize_dct(
            y, 28, dtype=torch.float32, conv_mxs=self.conversion_matrices
        )
        cbcr = self.dct_ops.resize_dct(
            cbcr, 14, dtype=torch.float32, conv_mxs=self.conversion_matrices
        )
        y = y.reshape(condition_count, batch_size, 1, 28, 28, 8, 8)
        cbcr = cbcr.reshape(condition_count, batch_size, 2, 14, 14, 8, 8)

        # Exact operation order used by RGB-no-more ToRange(-1, 1, -1024, 1016).
        y = -1.0 + ((y.to(torch.float32) - (-1024.0)) / 2040.0) * 2.0
        cbcr = -1.0 + ((cbcr.to(torch.float32) - (-1024.0)) / 2040.0) * 2.0
        return y, cbcr


def build_model(rgbnomore_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    _load_rgbnomore_modules(rgbnomore_root)
    plainvit = importlib.import_module("models.plainvit")
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
    try:
        checkpoint_object = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint_object = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_object.get("model_state_dict", checkpoint_object)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def official_cpu_transform(
    rgbnomore_root: Path,
    y_quantized: torch.Tensor,
    cbcr_quantized: torch.Tensor,
    quant: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, datasets = _load_rgbnomore_modules(rgbnomore_root)
    y = torch.clamp(y_quantized * quant[0], min=-1024, max=1016)
    cbcr = torch.clamp(
        cbcr_quantized * quant[1:3].unsqueeze(1).unsqueeze(1), min=-1024, max=1016
    )
    transform = datasets.get_transform(
        dataset="imagenet_dct", type="test", dtype=torch.float32, dtype_resize=torch.float32
    )
    return transform((y, cbcr))


def self_check(
    sample: dict[str, Any],
    rgbnomore_root: Path,
    preprocess_device: torch.device,
) -> dict[str, Any]:
    dct_manip, _, _ = _load_rgbnomore_modules(rgbnomore_root)
    _, quant, y_quantized, cbcr_quantized = dct_manip.read_coefficients(sample["path"])
    if cbcr_quantized is None:
        raise RuntimeError("self-check sample unexpectedly has no chroma")

    identity = next(
        condition for condition in build_conditions() if condition.condition_id == "prefix_k64"
    )
    dc_only = next(
        condition for condition in build_conditions() if condition.condition_id == "prefix_k01"
    )
    identity_mask = torch.from_numpy(masks_numpy([identity]))
    dc_mask = torch.from_numpy(masks_numpy([dc_only]))

    # Explicitly audit that discarded raw quantized coefficients are zero before
    # any later operation is called.
    raw_dc_y = y_quantized.unsqueeze(0) * dc_mask.reshape(1, 1, 1, 1, 8, 8)
    raw_dc_c = cbcr_quantized.unsqueeze(0) * dc_mask.reshape(1, 1, 1, 1, 8, 8)
    outside = (~dc_mask).reshape(1, 1, 1, 1, 8, 8)
    if torch.count_nonzero(raw_dc_y * outside).item() != 0:
        raise AssertionError("Y mask was not applied in the raw coefficient domain")
    if torch.count_nonzero(raw_dc_c * outside).item() != 0:
        raise AssertionError("CbCr mask was not applied in the raw coefficient domain")

    reference_y, reference_cbcr = official_cpu_transform(
        rgbnomore_root, y_quantized, cbcr_quantized, quant
    )
    cpu_preprocessor = BatchedDctPreprocessor(rgbnomore_root, torch.device("cpu"))
    cpu_y, cpu_cbcr = cpu_preprocessor(
        y_quantized.unsqueeze(0),
        cbcr_quantized.unsqueeze(0),
        quant.unsqueeze(0),
        identity_mask,
    )
    cpu_y = cpu_y[0, 0]
    cpu_cbcr = cpu_cbcr[0, 0]
    cpu_y_max_abs = float((cpu_y - reference_y).abs().max().item())
    cpu_cbcr_max_abs = float((cpu_cbcr - reference_cbcr).abs().max().item())
    if not torch.equal(cpu_y, reference_y) or not torch.equal(cpu_cbcr, reference_cbcr):
        raise AssertionError(
            "K=64 batched CPU preprocessing does not exactly reproduce the official transform: "
            f"Y max_abs={cpu_y_max_abs}, CbCr max_abs={cpu_cbcr_max_abs}"
        )

    device_y_max_abs = 0.0
    device_cbcr_max_abs = 0.0
    if preprocess_device.type != "cpu":
        device_preprocessor = BatchedDctPreprocessor(rgbnomore_root, preprocess_device)
        device_y, device_cbcr = device_preprocessor(
            y_quantized.unsqueeze(0),
            cbcr_quantized.unsqueeze(0),
            quant.unsqueeze(0),
            identity_mask,
        )
        device_y_max_abs = float((device_y[0, 0].cpu() - reference_y).abs().max().item())
        device_cbcr_max_abs = float(
            (device_cbcr[0, 0].cpu() - reference_cbcr).abs().max().item()
        )
        one_integer_step = 2.0 / 2040.0
        # CPU and CUDA float32 einsum may land on opposite sides of an integer
        # rounding tie.  Permit one coefficient level plus a small relative
        # floating-point margin, while still rejecting a two-level difference.
        tolerance = one_integer_step * 1.001 + 1e-7
        if device_y_max_abs > tolerance or device_cbcr_max_abs > tolerance:
            raise AssertionError(
                "device preprocessing differs from official CPU transform by more than one "
                f"integer coefficient: Y={device_y_max_abs}, CbCr={device_cbcr_max_abs}"
            )

    return {
        "sample_path": sample["path"],
        "raw_mask_before_dequantization": True,
        "prefix_k01_raw_discarded_y_nonzeros": 0,
        "prefix_k01_raw_discarded_cbcr_nonzeros": 0,
        "prefix_k64_cpu_vs_official_y_max_abs": cpu_y_max_abs,
        "prefix_k64_cpu_vs_official_cbcr_max_abs": cpu_cbcr_max_abs,
        "prefix_k64_device_vs_official_y_max_abs": device_y_max_abs,
        "prefix_k64_device_vs_official_cbcr_max_abs": device_cbcr_max_abs,
    }


def create_or_open_memmaps(
    output_dir: Path,
    condition_count: int,
    sample_count: int,
    resume: bool,
) -> tuple[np.memmap, np.memmap]:
    classes_path = output_dir / "top5_classes.npy"
    probabilities_path = output_dir / "top5_probabilities.npy"
    expected_classes_shape = (condition_count, sample_count, 5)
    expected_probabilities_shape = (condition_count, sample_count, 5)
    if resume:
        if not classes_path.is_file() or not probabilities_path.is_file():
            raise FileNotFoundError("resume requested but prediction memmaps are missing")
        classes = np.lib.format.open_memmap(classes_path, mode="r+")
        probabilities = np.lib.format.open_memmap(probabilities_path, mode="r+")
        if classes.shape != expected_classes_shape or probabilities.shape != expected_probabilities_shape:
            raise ValueError(
                f"resume shape mismatch: classes={classes.shape}, probabilities={probabilities.shape}"
            )
    else:
        classes = np.lib.format.open_memmap(
            classes_path, mode="w+", dtype=np.int16, shape=expected_classes_shape
        )
        probabilities = np.lib.format.open_memmap(
            probabilities_path, mode="w+", dtype=np.float32, shape=expected_probabilities_shape
        )
        classes.fill(-1)
        probabilities.fill(np.nan)
        classes.flush()
        probabilities.flush()
    return classes, probabilities


def write_samples_csv(path: Path, samples: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_index", "logical_sample_id", "path", "label", "jpeg_sampling"))
        for index, sample in enumerate(samples):
            writer.writerow(
                (
                    index,
                    sample["logical_sample_id"],
                    sample["path"],
                    int(sample["label"]),
                    sample["jpeg_sampling"],
                )
            )


def accuracy_rows(
    conditions: Sequence[MaskCondition],
    top5_classes: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, condition in enumerate(conditions):
        predictions = top5_classes[row_index]
        correct1 = int(np.count_nonzero(predictions[:, 0] == labels))
        correct5 = int(np.count_nonzero(np.any(predictions == labels[:, None], axis=1)))
        sample_count = int(labels.size)
        rows.append(
            {
                "condition_index": row_index,
                "condition_id": condition.condition_id,
                "family": condition.family,
                "k": condition.k,
                "sample_count": sample_count,
                "correct_top1": correct1,
                "correct_top5": correct5,
                "accuracy_top1": correct1 / sample_count,
                "accuracy_top5": correct5 / sample_count,
                "accuracy_top1_percent": 100.0 * correct1 / sample_count,
                "accuracy_top5_percent": 100.0 * correct5 / sample_count,
                "zigzag_ranks": json.dumps(condition.zigzag_ranks, separators=(",", ":")),
                "natural_indices": json.dumps(condition.natural_indices, separators=(",", ":")),
                "random_seed": "" if condition.random_seed is None else condition.random_seed,
            }
        )
    return rows


def write_accuracy_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_wide_predictions(
    path: Path,
    samples: Sequence[dict[str, Any]],
    conditions: Sequence[MaskCondition],
    top5_classes: np.ndarray,
    top5_probabilities: np.ndarray,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=3) as stream:
        writer = csv.writer(stream)
        header = ["sample_index", "logical_sample_id", "path", "label"]
        for condition in conditions:
            header.extend(
                (
                    f"{condition.condition_id}__top1_class",
                    f"{condition.condition_id}__top1_probability",
                    f"{condition.condition_id}__correct_top1",
                )
            )
        writer.writerow(header)
        for sample_index, sample in enumerate(samples):
            label = int(sample["label"])
            row: list[Any] = [
                sample_index,
                sample["logical_sample_id"],
                sample["path"],
                label,
            ]
            for condition_index in range(len(conditions)):
                prediction = int(top5_classes[condition_index, sample_index, 0])
                probability = float(top5_probabilities[condition_index, sample_index, 0])
                row.extend((prediction, f"{probability:.8g}", int(prediction == label)))
            writer.writerow(row)


def write_curve_svg(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    prefix = [row for row in rows if row["family"] == "zigzag_prefix"]
    if not prefix:
        return
    width, height = 1100, 680
    left, right, top, bottom = 90, 40, 55, 85
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_percentages = [float(row["accuracy_top1_percent"]) for row in rows]
    all_percentages.extend(float(row["accuracy_top5_percent"]) for row in prefix)
    y_min = max(0.0, float(np.floor(min(all_percentages) / 5.0) * 5.0))
    y_max = min(100.0, float(np.ceil(max(all_percentages) / 5.0) * 5.0))
    if y_max <= y_min:
        y_max = min(100.0, y_min + 5.0)

    def x_coord(k: int) -> float:
        return left + (k - 1) * plot_width / 63.0

    def y_coord(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    def polyline(values: Sequence[tuple[int, float]], color: str, width_px: int = 3) -> str:
        points = " ".join(f"{x_coord(k):.2f},{y_coord(value):.2f}" for k, value in values)
        return (
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{width_px}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.5}.grid{stroke:#ddd;stroke-width:1}</style>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="22">ImageNet accuracy vs. retained raw DCT coefficients</text>',
    ]
    tick_start = int(np.ceil(y_min / 5.0) * 5)
    for value in range(tick_start, int(y_max) + 1, 5):
        y = y_coord(float(value))
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        svg.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="14">{value}%</text>')
    for k in (1, 4, 8, 16, 24, 32, 40, 48, 56, 64):
        x = x_coord(k)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        svg.append(f'<text x="{x:.2f}" y="{height-bottom+25}" text-anchor="middle" font-size="14">{k}</text>')
    svg.extend(
        (
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<text x="{left+plot_width/2:.1f}" y="{height-28}" text-anchor="middle" font-size="17">Retained coefficients K (JPEG zigzag prefix)</text>',
            f'<text x="25" y="{top+plot_height/2:.1f}" text-anchor="middle" font-size="17" transform="rotate(-90 25 {top+plot_height/2:.1f})">Accuracy</text>',
        )
    )
    svg.append(
        polyline(
            [(int(row["k"]), float(row["accuracy_top1_percent"])) for row in prefix],
            "#1769aa",
        )
    )
    svg.append(
        polyline(
            [(int(row["k"]), float(row["accuracy_top5_percent"])) for row in prefix],
            "#2e7d32",
        )
    )
    control_colors = {"high": "#c62828", "mid": "#ef6c00", "random": "#7b1fa2"}
    for row in rows:
        family = str(row["family"])
        if family not in control_colors:
            continue
        x = x_coord(int(row["k"]))
        y = y_coord(float(row["accuracy_top1_percent"]))
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{control_colors[family]}" stroke="white" stroke-width="1"/>'
        )
    legend = (
        ("Prefix top-1", "#1769aa", "line"),
        ("Prefix top-5", "#2e7d32", "line"),
        ("High control top-1", "#c62828", "point"),
        ("Mid control top-1", "#ef6c00", "point"),
        ("Random control top-1", "#7b1fa2", "point"),
    )
    legend_x, legend_y = left + 18, top + 18
    svg.append(
        f'<rect x="{legend_x-10}" y="{legend_y-19}" width="230" height="{len(legend)*26+13}" fill="white" fill-opacity="0.9" stroke="#bbb"/>'
    )
    for offset, (label, color, kind) in enumerate(legend):
        y = legend_y + offset * 26
        if kind == "line":
            svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        else:
            svg.append(f'<circle cx="{legend_x+14}" cy="{y}" r="5" fill="{color}"/>')
        svg.append(f'<text x="{legend_x+38}" y="{y+5}" font-size="14">{html.escape(label)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def make_run_signature(
    args: argparse.Namespace,
    samples: Sequence[dict[str, Any]],
    conditions: Sequence[MaskCondition],
    manifest_sha256: str,
    checkpoint_sha256: str,
    source_files: dict[str, dict[str, str]],
    observed_device_names: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": "coefficient-mask-evaluator-v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "rgbnomore_root": str(args.rgbnomore_root.resolve()),
        "sample_count": len(samples),
        "first_logical_sample_id": samples[0]["logical_sample_id"],
        "last_logical_sample_id": samples[-1]["logical_sample_id"],
        "conditions": [condition.to_dict() for condition in conditions],
        "random_seed": args.random_seed,
        "precision": "fp32",
        "tf32": False,
        "mask_semantics": "same_8x8_binary_mask_for_Y_Cb_Cr_and_every_spatial_block",
        "mask_application_stage": "raw_quantized_coefficients_before_dequantization_and_frequency_mixing",
        "source_files": source_files,
        "execution_fingerprint": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "dependency_versions": {
                package: importlib.metadata.version(package)
                for package in ("einops", "scipy", "torchvision")
            },
            "device": args.device,
            "preprocess_device": args.preprocess_device or args.device,
            "expected_device_name": args.expected_device_name,
            "observed_device_names": observed_device_names,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": args.batch_size,
            "condition_chunk_size": args.condition_chunk_size,
            "torch_cpu_threads": args.torch_cpu_threads,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "no_wide_csv": args.no_wide_csv,
        },
    }


@dataclass(frozen=True)
class EvaluatorProgress:
    completed_samples: int
    inference_seconds: float


def load_progress(output_dir: Path, resume: bool) -> EvaluatorProgress:
    progress_path = output_dir / "progress.json"
    if not resume:
        return EvaluatorProgress(0, 0.0)
    if not progress_path.is_file():
        raise FileNotFoundError("resume requested but progress.json is missing")
    with progress_path.open("r", encoding="utf-8") as stream:
        progress = json.load(stream)
    completed = int(progress.get("completed_samples", -1))
    if completed < 0:
        raise ValueError("invalid completed_samples in progress.json")
    inference_seconds = float(
        progress.get(
            "inference_seconds",
            progress.get("elapsed_seconds_this_process", 0.0),
        )
    )
    if not math.isfinite(inference_seconds) or inference_seconds < 0.0:
        raise ValueError("invalid inference_seconds in progress.json")
    return EvaluatorProgress(completed, inference_seconds)


def run(args: argparse.Namespace) -> int:
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.rgbnomore_root = args.rgbnomore_root.resolve()
    args.output_dir = args.output_dir.resolve()
    for required in (args.manifest, args.checkpoint, args.rgbnomore_root):
        if not required.exists():
            raise FileNotFoundError(required)

    device = torch.device(args.device)
    preprocess_device = torch.device(args.preprocess_device or args.device)
    observed_device_names: dict[str, str] = {}
    if not args.dry_run:
        if device.type != "cuda":
            raise ValueError("the full evaluator requires a CUDA model device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
        observed_device_name = torch.cuda.get_device_name(device)
        observed_device_names["model"] = observed_device_name
        if preprocess_device.type == "cuda":
            observed_device_names["preprocess"] = torch.cuda.get_device_name(
                preprocess_device
            )
        if (
            args.expected_device_name is not None
            and observed_device_name != args.expected_device_name
        ):
            raise RuntimeError(
                "CUDA device identity mismatch: "
                f"expected {args.expected_device_name!r}, observed {observed_device_name!r}. "
                "Use a stable GPU UUID in CUDA_VISIBLE_DEVICES on hosts whose CUDA ordinal "
                "order differs from nvidia-smi indices."
            )

    manifest_payload, samples = load_samples(args.manifest, args.max_samples)
    conditions = select_conditions(build_conditions(args.random_seed), args.conditions)
    manifest_sha256 = sha256_file(args.manifest)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    dct_manip, _, _ = _load_rgbnomore_modules(args.rgbnomore_root)
    dct_manip_file = getattr(dct_manip, "__file__", None)
    if not dct_manip_file:
        raise RuntimeError("cannot fingerprint the loaded dct_manip runtime")
    source_files = source_file_fingerprints(
        args.rgbnomore_root, Path(dct_manip_file)
    )
    signature = make_run_signature(
        args,
        samples,
        conditions,
        manifest_sha256,
        checkpoint_sha256,
        source_files,
        observed_device_names,
    )

    if args.dry_run:
        print(json.dumps(signature, indent=2, sort_keys=True))
        return 0

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory without --resume: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = args.output_dir / "run_signature.json"
    if args.resume:
        with signature_path.open("r", encoding="utf-8") as stream:
            previous_signature = json.load(stream)
        if previous_signature != signature:
            raise ValueError("resume signature differs from the existing run")
    else:
        write_json(signature_path, signature)
        write_json(
            args.output_dir / "conditions.json",
            [condition.to_dict() for condition in conditions],
        )
        write_samples_csv(args.output_dir / "samples.csv", samples)

    resume_progress = load_progress(args.output_dir, args.resume)
    completed_samples = resume_progress.completed_samples
    prior_inference_seconds = resume_progress.inference_seconds
    if completed_samples > len(samples):
        raise ValueError("progress exceeds requested sample count")
    classes, probabilities = create_or_open_memmaps(
        args.output_dir, len(conditions), len(samples), args.resume
    )

    torch.set_num_threads(args.torch_cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    observed_device_name = observed_device_names["model"]

    before_hardware = hardware_snapshot()
    write_json(args.output_dir / "hardware_before.json", before_hardware)
    audit = self_check(samples[0], args.rgbnomore_root, preprocess_device)
    write_json(
        args.output_dir / "preprocess_audit.json",
        {"contract": PreprocessAudit().__dict__, "checks": audit},
    )
    print(
        "self-check passed: raw mask precedes dequantization; "
        f"K=64 CPU exact; device max_abs Y={audit['prefix_k64_device_vs_official_y_max_abs']:.9g}, "
        f"CbCr={audit['prefix_k64_device_vs_official_cbcr_max_abs']:.9g}",
        flush=True,
    )

    model = build_model(args.rgbnomore_root, args.checkpoint, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    preprocessor = BatchedDctPreprocessor(args.rgbnomore_root, preprocess_device)
    mask_tensor = torch.from_numpy(masks_numpy(list(conditions)))
    torch.cuda.reset_peak_memory_stats(device)

    dataset = RawDctDataset(samples, args.rgbnomore_root, start=completed_samples)
    loader_arguments: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        "worker_init_fn": worker_init,
    }
    if args.workers > 0:
        loader_arguments.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = torch.utils.data.DataLoader(dataset, **loader_arguments)

    run_started = time.perf_counter()
    batch_count = 0
    condition_count = len(conditions)
    write_json(
        args.output_dir / "progress.json",
        {
            "status": "running",
            "completed_samples": completed_samples,
            "sample_count": len(samples),
            "condition_count": condition_count,
            "inference_seconds": prior_inference_seconds,
            "started_at_unix_ns": time.time_ns(),
        },
    )

    try:
        for y_quantized, cbcr_quantized, quant, labels, ordinals in loader:
            batch_count += 1
            ordinals_list = [int(value) for value in ordinals.tolist()]
            expected = list(range(completed_samples, completed_samples + len(ordinals_list)))
            if ordinals_list != expected:
                raise RuntimeError(
                    f"sample order mismatch: expected {expected[:3]}..., got {ordinals_list[:3]}..."
                )
            batch_size = len(ordinals_list)

            for condition_begin in range(0, condition_count, args.condition_chunk_size):
                condition_end = min(condition_count, condition_begin + args.condition_chunk_size)
                y, cbcr = preprocessor(
                    y_quantized,
                    cbcr_quantized,
                    quant,
                    mask_tensor[condition_begin:condition_end],
                )
                chunk_size = condition_end - condition_begin
                y = y.reshape(chunk_size * batch_size, 1, 28, 28, 8, 8)
                cbcr = cbcr.reshape(chunk_size * batch_size, 2, 14, 14, 8, 8)
                if y.device != device:
                    y = y.to(device, non_blocking=True)
                    cbcr = cbcr.to(device, non_blocking=True)
                with torch.inference_mode():
                    logits = model(y, cbcr)
                    top_probabilities, top_classes = torch.topk(
                        torch.softmax(logits.float(), dim=1), k=5, dim=1
                    )
                top_classes_np = (
                    top_classes.reshape(chunk_size, batch_size, 5)
                    .to(device="cpu", dtype=torch.int16)
                    .numpy()
                )
                top_probabilities_np = (
                    top_probabilities.reshape(chunk_size, batch_size, 5).cpu().numpy()
                )
                classes[
                    condition_begin:condition_end,
                    completed_samples : completed_samples + batch_size,
                    :,
                ] = top_classes_np
                probabilities[
                    condition_begin:condition_end,
                    completed_samples : completed_samples + batch_size,
                    :,
                ] = top_probabilities_np
                del y, cbcr, logits, top_classes, top_probabilities

            completed_samples += batch_size
            elapsed = time.perf_counter() - run_started
            if batch_count % args.log_every_batches == 0 or completed_samples == len(samples):
                rate = (completed_samples - dataset.start) / elapsed if elapsed > 0 else 0.0
                print(
                    f"progress {completed_samples}/{len(samples)} source images "
                    f"({100.0*completed_samples/len(samples):.2f}%), "
                    f"{condition_count} conditions, source_rate={rate:.2f} images/s, "
                    f"effective_rate={rate*condition_count:.2f} predictions/s",
                    flush=True,
                )
            if batch_count % args.flush_every_batches == 0 or completed_samples == len(samples):
                classes.flush()
                probabilities.flush()
                write_json(
                    args.output_dir / "progress.json",
                    {
                        "status": "running",
                        "completed_samples": completed_samples,
                        "sample_count": len(samples),
                        "condition_count": condition_count,
                        "elapsed_seconds_this_process": elapsed,
                        "inference_seconds": prior_inference_seconds + elapsed,
                        "updated_at_unix_ns": time.time_ns(),
                    },
                )
    except BaseException as error:
        elapsed_this_process = time.perf_counter() - run_started
        classes.flush()
        probabilities.flush()
        write_json(
            args.output_dir / "progress.json",
            {
                "status": "failed",
                "completed_samples": completed_samples,
                "sample_count": len(samples),
                "condition_count": condition_count,
                "elapsed_seconds_this_process": elapsed_this_process,
                "inference_seconds": prior_inference_seconds + elapsed_this_process,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "updated_at_unix_ns": time.time_ns(),
            },
        )
        raise

    torch.cuda.synchronize(device)
    processed_samples_this_process = completed_samples - dataset.start
    inference_seconds_this_process = (
        time.perf_counter() - run_started
        if processed_samples_this_process > 0
        else 0.0
    )
    inference_seconds = prior_inference_seconds + inference_seconds_this_process
    if inference_seconds <= 0.0:
        raise RuntimeError("completed evaluation has no measured inference time")
    peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_cuda_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    labels_np = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int16)
    rows = accuracy_rows(conditions, classes, labels_np)
    write_accuracy_csv(args.output_dir / "accuracy_curve.csv", rows)
    prefix_rows = [row for row in rows if row["family"] == "zigzag_prefix"]
    if prefix_rows:
        write_accuracy_csv(args.output_dir / "prefix_accuracy_curve.csv", prefix_rows)
    control_rows = [row for row in rows if row["family"] != "zigzag_prefix"]
    if control_rows:
        write_accuracy_csv(args.output_dir / "matched_budget_controls.csv", control_rows)
    write_curve_svg(args.output_dir / "accuracy_curve.svg", rows)

    export_started = time.perf_counter()
    if not args.no_wide_csv:
        print("writing per_sample_top1.csv.gz", flush=True)
        write_wide_predictions(
            args.output_dir / "per_sample_top1.csv.gz",
            samples,
            conditions,
            classes,
            probabilities,
        )
    export_seconds = time.perf_counter() - export_started

    after_hardware = hardware_snapshot()
    write_json(args.output_dir / "hardware_after.json", after_hardware)
    k64 = next((row for row in rows if row["condition_id"] == "prefix_k64"), None)
    metadata = {
        "schema": "coefficient-mask-evaluator-result-v1",
        "status": "complete",
        "run_signature": signature,
        "manifest_format": manifest_payload.get("format"),
        "preprocess_audit": {"contract": PreprocessAudit().__dict__, "checks": audit},
        "model_parameter_count": parameter_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": observed_device_name,
        "preprocess_device": str(preprocess_device),
        "batch_size": args.batch_size,
        "condition_chunk_size": args.condition_chunk_size,
        "workers": args.workers,
        "inference_seconds": inference_seconds,
        "inference_seconds_this_process": inference_seconds_this_process,
        "resumed_inference_seconds": prior_inference_seconds,
        "wide_csv_export_seconds": export_seconds,
        "effective_prediction_count": len(samples) * len(conditions),
        "effective_predictions_per_second": (
            len(samples) * len(conditions) / inference_seconds
        ),
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "prefix_k64": k64,
        "hardware_before": before_hardware,
        "hardware_after": after_hardware,
        "source_files": signature["source_files"],
        "execution_fingerprint": signature["execution_fingerprint"],
        "completed_at_unix_ns": time.time_ns(),
    }
    write_json(args.output_dir / "run_metadata.json", metadata)
    write_json(
        args.output_dir / "progress.json",
        {
            "status": "complete",
            "completed_samples": len(samples),
            "sample_count": len(samples),
            "condition_count": len(conditions),
            "inference_seconds": inference_seconds,
            "inference_seconds_this_process": inference_seconds_this_process,
            "updated_at_unix_ns": time.time_ns(),
        },
    )
    print(
        f"complete: {len(samples)} images x {len(conditions)} conditions in "
        f"{inference_seconds:.1f}s; outputs={args.output_dir}",
        flush=True,
    )
    if k64 is not None:
        print(
            f"K=64 accuracy: top1={k64['accuracy_top1_percent']:.4f}% "
            f"top5={k64['accuracy_top5_percent']:.4f}%",
            flush=True,
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--expected-device-name",
        help="Fail before evaluation unless torch reports this exact CUDA device name.",
    )
    parser.add_argument(
        "--preprocess-device",
        default=None,
        help="Device for masking and DCT resize; defaults to --device.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--condition-chunk-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260816)
    parser.add_argument(
        "--conditions",
        default="all",
        help="all, or comma-separated condition IDs (useful for smoke tests).",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--flush-every-batches", type=int, default=10)
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wide-csv", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "batch_size",
        "condition_chunk_size",
        "prefetch_factor",
        "torch_cpu_threads",
        "flush_every_batches",
        "log_every_batches",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
