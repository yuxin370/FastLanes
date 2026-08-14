#!/usr/bin/env python3
"""Published RGB-no-more DCT augmentation with keyed random streams."""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .core_schedule import (
    crop_key,
    flip_key,
    mixup_key,
    randaugment_key,
    stable_seed,
)
from .recipe import DCT_RANDAUGMENT_OPERATIONS


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "system_rgbnomore"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from training.augmentation import AugmentationDecision  # noqa: E402


@dataclass(frozen=True)
class PublishedAugmentation:
    decision: AugmentationDecision
    crop_key: str
    flip_key: str
    crop_seed: int
    flip_seed: int

    def as_dict(self) -> dict[str, Any]:
        payload = self.decision.as_dict()
        payload.update(
            crop_key=self.crop_key,
            flip_key=self.flip_key,
            crop_seed=self.crop_seed,
            flip_seed=self.flip_seed,
            algorithm="RGB-no-more RandomResizedCrop_DCT keyed-v1",
        )
        return payload


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def _closest_crop_size(value: int, choices: Sequence[int], maximum: int) -> int:
    if value <= choices[-1]:
        return min(choices, key=lambda choice: (abs(choice - value), choice))
    closest = round(value / choices[-1]) * choices[-1]
    if closest > maximum:
        closest -= choices[-1]
    return int(closest)


def _published_dct_crop_blocks(
    *, seed: int, width_blocks: int, height_blocks: int
) -> tuple[int, int, int, int]:
    """Return (top, left, height, width) from RandomResizedCrop_DCT(28)."""

    if width_blocks < 2 or height_blocks < 2:
        raise ValueError("published DCT crop requires at least a 2x2 luma block grid")
    generator = _torch_generator(seed)
    choices = (2, 4, 14, 28)
    area = width_blocks * height_blocks
    for _ in range(10):
        draw = float(torch.rand((), generator=generator).item())
        target_area = area * (0.05 + draw * (1.0 - 0.05))
        width = int(round(math.sqrt(target_area)))
        width = _closest_crop_size(width, choices, width_blocks)
        height = width
        width = max(2, width)
        height = max(2, height)
        if width <= width_blocks and height <= height_blocks:
            top = int(
                torch.randint(
                    0, height_blocks - height + 1, (1,), generator=generator
                ).item()
            )
            left = int(
                torch.randint(
                    0, width_blocks - width + 1, (1,), generator=generator
                ).item()
            )
            return (top // 2 * 2, left // 2 * 2, height, width)
    width = _closest_crop_size(width_blocks, choices, width_blocks)
    height = _closest_crop_size(height_blocks, choices, height_blocks)
    top = ((height_blocks - height) // 2) // 2 * 2
    left = ((width_blocks - width) // 2) // 2 * 2
    return top, left, max(1, height), max(1, width)


def published_training_augmentation(
    *,
    training_seed: int,
    epoch: int,
    logical_sample_id: str,
    virtual_pls_id: int,
    crop_policy: str,
    source_width: int,
    source_height: int,
) -> PublishedAugmentation:
    crop_identity = (
        logical_sample_id if crop_policy == "per-sample" else int(virtual_pls_id)
    )
    crop_namespace = "crop-per-sample" if crop_policy == "per-sample" else "crop-per-pls"
    if crop_policy not in ("per-sample", "per-pls"):
        raise ValueError(f"unknown crop policy: {crop_policy}")
    crop_seed = stable_seed(crop_namespace, training_seed, epoch, crop_identity)
    flip_seed_value = stable_seed(
        "horizontal-flip", training_seed, epoch, logical_sample_id
    )
    width_blocks = (int(source_width) + 7) // 8
    height_blocks = (int(source_height) + 7) // 8
    top, left, crop_height, crop_width = _published_dct_crop_blocks(
        seed=crop_seed,
        width_blocks=width_blocks,
        height_blocks=height_blocks,
    )
    flip_generator = _torch_generator(flip_seed_value)
    horizontal_flip = float(torch.rand((), generator=flip_generator).item()) < 0.5
    decision = AugmentationDecision(
        seed=int(training_seed),
        epoch=int(epoch),
        logical_sample_id=str(logical_sample_id),
        source_width=int(source_width),
        source_height=int(source_height),
        crop_x=left * 8,
        crop_y=top * 8,
        crop_width=crop_width * 8,
        crop_height=crop_height * 8,
        resize_width=224,
        resize_height=224,
        horizontal_flip=horizontal_flip,
        interpolation="bilinear",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        domain="dct",
        dct_crop_alignment_pixels=16,
    )
    return PublishedAugmentation(
        decision=decision,
        crop_key=crop_key(
            training_seed=training_seed,
            epoch=epoch,
            logical_sample_id=logical_sample_id,
            virtual_pls_id=virtual_pls_id,
            crop_policy=crop_policy,
        ),
        flip_key=flip_key(
            training_seed=training_seed,
            epoch=epoch,
            logical_sample_id=logical_sample_id,
        ),
        crop_seed=crop_seed,
        flip_seed=flip_seed_value,
    )


def published_validation_augmentation(
    *, logical_sample_id: str, source_width: int, source_height: int, epoch: int = 0
) -> AugmentationDecision:
    """Map ResizedCenterCrop_DCT(32,28) to the native crop descriptor."""

    width_blocks = (int(source_width) + 7) // 8
    height_blocks = (int(source_height) + 7) // 8
    ratio = 28.0 / 32.0

    def closest(value: int, maximum: int) -> int:
        choices = (2, 4, 14, 28)
        return _closest_crop_size(value, choices, maximum)

    crop_width = closest(round(ratio * width_blocks), width_blocks)
    crop_height = closest(round(ratio * height_blocks), height_blocks)
    top = ((height_blocks - crop_height) // 2) // 2 * 2
    left = ((width_blocks - crop_width) // 2) // 2 * 2
    return AugmentationDecision(
        seed=0,
        epoch=int(epoch),
        logical_sample_id=str(logical_sample_id),
        source_width=int(source_width),
        source_height=int(source_height),
        crop_x=left * 8,
        crop_y=top * 8,
        crop_width=crop_width * 8,
        crop_height=crop_height * 8,
        resize_width=224,
        resize_height=224,
        horizontal_flip=False,
        interpolation="bilinear",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        domain="dct",
        dct_crop_alignment_pixels=16,
    )


def _dct_ops(rgbnomore_root: Path):
    root = str(rgbnomore_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils import dct_ops

    return dct_ops


def _magnitude(operation: str) -> tuple[float, bool]:
    # Exact magnitude-bin index 3 of 11 from RandAugment_dct.
    values: dict[str, tuple[float, bool]] = {
        "AutoContrast": (0.0, False),
        "Posterize": (2.0, False),
        "SolarizeAdd": (264.9, False),
        "Color": (0.27, True),
        "Contrast": (0.27, True),
        "Brightness": (0.27, True),
        "MidfreqAug": (0.27, True),
        "Cutout": (1.8, False),
        "TranslateX": (3.75, True),
        "TranslateY": (3.75, True),
        "Rotate90": (1.0, True),
        "AutoSaturation": (0.0, False),
        "Grayscale": (0.0, False),
        "ChromaDrop": (0.0, False),
    }
    return values[operation]


def _midfreqaug_device_safe(dops: Any, value: torch.Tensor, intensity: float) -> torch.Tensor:
    import scipy.signal

    result = value.clone()
    original_dtype = result.dtype
    kh, kw = int(result.shape[-2]), int(result.shape[-1])
    result = dops.blockshift(result, dim=(-2, -1))
    h_intensity = kh // 2 - (kh // 8 * 2.2) * abs(intensity)
    w_intensity = kw // 2 - (kw // 8 * 2.2) * abs(intensity)
    filter_h = torch.as_tensor(
        scipy.signal.windows.gaussian(kh, h_intensity),
        dtype=torch.float32,
        device=result.device,
    ).unsqueeze(1)
    filter_w = torch.as_tensor(
        scipy.signal.windows.gaussian(kw, w_intensity),
        dtype=torch.float32,
        device=result.device,
    ).unsqueeze(0)
    matrix = filter_h.mm(filter_w)
    if intensity >= 0:
        matrix = 1.0 / matrix
    result = result * matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    result = result.clamp(min=-1024, max=1016)
    result = dops.iblockshift(result, dim=(-2, -1))
    if not original_dtype.is_floating_point:
        result = torch.round(result)
    return result.to(original_dtype)


def _posterize_device_safe(value: torch.Tensor, bitoffset: int) -> torch.Tensor:
    """Device-aware equivalent of RGB-no-more ``posterize_dct``."""

    minimum, maximum = -1024, 1016
    result = value.clone()
    original_dtype = result.dtype
    dc = result[:, :, :, 0, 0].to(torch.float32) - minimum
    if bool((dc < 0).any().item()):
        raise ValueError("posterize received a coefficient below the published clamp")
    indices = torch.round(dc / (2**bitoffset)).to(torch.int64)
    table = torch.linspace(
        minimum,
        maximum,
        round((maximum - minimum) / (2**bitoffset)) + 1,
        dtype=torch.float32,
        device=result.device,
    )
    dc = table[indices]
    if not original_dtype.is_floating_point:
        dc = torch.round(dc)
    result[:, :, :, 0, 0] = dc.to(original_dtype)
    return result


def normalized_to_published_int16(value: torch.Tensor) -> torch.Tensor:
    """Invert ToRange and apply RandAugment_dct's mandatory entry clamp."""

    return torch.round(value * 1020.0 - 4.0).clamp(-1024, 1016).to(torch.int16)


def _apply_operation(
    dops: Any,
    y: torch.Tensor,
    cbcr: torch.Tensor,
    operation: str,
    magnitude: float,
    *,
    internal_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if operation == "AutoContrast":
        y = dops.autocontrast_dct(y)
    elif operation == "Posterize":
        y = _posterize_device_safe(y, bitoffset=int(magnitude))
        cbcr = _posterize_device_safe(cbcr, bitoffset=int(magnitude))
    elif operation == "SolarizeAdd":
        y, _ = dops.solarize_add_dct(y, int(magnitude), threshold=0)
    elif operation == "Color":
        cbcr = dops.contrast_dct(cbcr, 1.0 + magnitude)
    elif operation == "Contrast":
        y = dops.contrast_dct(y, 1.0 + magnitude)
    elif operation == "Brightness":
        y = dops.brightness_dct(y, 1.0 + magnitude)
    elif operation == "MidfreqAug":
        y = _midfreqaug_device_safe(dops, y, magnitude)
    elif operation == "Cutout":
        pad = round(magnitude)
        pad = int(pad - (pad % 2))
        generator = _torch_generator(internal_seed)
        center_h = int(torch.randint(0, y.shape[1], (1,), generator=generator).item()) // 2 * 2
        center_w = int(torch.randint(0, y.shape[2], (1,), generator=generator).item()) // 2 * 2
        y, _, _ = dops.cutout_dct(y, pad, 0, center_h, center_w)
        cbcr, _, _ = dops.cutout_dct(cbcr, pad // 2, 0, center_h // 2, center_w // 2)
    elif operation in ("TranslateX", "TranslateY"):
        blocks = int(magnitude - (magnitude % 2))
        direction = "W" if operation == "TranslateX" else "H"
        y = dops.translate_dct(y, blocks, direction=direction)
        cbcr = dops.translate_dct(cbcr, blocks // 2, direction=direction)
    elif operation == "Rotate90":
        y = dops.rotate_dct_90deg(y, rotate=magnitude)
        cbcr = dops.rotate_dct_90deg(cbcr, rotate=magnitude)
    elif operation == "AutoSaturation":
        cbcr = dops.autocontrast_dct(cbcr)
    elif operation == "Grayscale":
        cbcr = cbcr * 0
    elif operation == "ChromaDrop":
        cbcr = cbcr.clone()
        channel = stable_seed("chroma-drop", internal_seed) % 2
        cbcr[int(channel)] *= 0
    else:
        raise ValueError(f"unrecognized published DCT RandAugment operation {operation}")
    return (
        y.clamp(min=-1024, max=1016).contiguous(),
        cbcr.clamp(min=-1024, max=1016).contiguous(),
    )


def apply_published_randaugment_scalar_reference(
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    training_seed: int,
    epoch: int,
    logical_sample_ids: Sequence[str],
    rgbnomore_root: Path,
) -> tuple[tuple[torch.Tensor, torch.Tensor], list[list[dict[str, Any]]]]:
    """Literal per-sample reference for keyed published DCT RandAugment."""

    y_batch, cbcr_batch = inputs
    if len(logical_sample_ids) != y_batch.shape[0] or y_batch.shape[0] != cbcr_batch.shape[0]:
        raise ValueError("RandAugment identities do not match DCT batch cardinality")
    dops = _dct_ops(rgbnomore_root)
    output_y: list[torch.Tensor] = []
    output_cbcr: list[torch.Tensor] = []
    records: list[list[dict[str, Any]]] = []
    for index, logical_id in enumerate(logical_sample_ids):
        # Native Direct-DCT normalization is exactly (raw + 4) / 1020.
        y = normalized_to_published_int16(y_batch[index])
        cbcr = normalized_to_published_int16(cbcr_batch[index])
        available = list(DCT_RANDAUGMENT_OPERATIONS)
        sample_records: list[dict[str, Any]] = []
        chroma_operations = {"Grayscale", "Color", "AutoSaturation", "ChromaDrop"}
        for operation_index in range(2):
            key = randaugment_key(
                training_seed=training_seed,
                epoch=epoch,
                logical_sample_id=logical_id,
                operation_index=operation_index,
            )
            choice_seed = stable_seed(
                "dct-randaugment-choice",
                training_seed,
                epoch,
                logical_id,
                operation_index,
            )
            operation = available[choice_seed % len(available)]
            if operation in chroma_operations:
                if operation == "Grayscale":
                    available = [name for name in available if name not in chroma_operations]
                else:
                    available = [name for name in available if name != "Grayscale"]
            magnitude, signed = _magnitude(operation)
            if signed and stable_seed("dct-randaugment-sign", key) % 2:
                magnitude *= -1.0
            internal_seed = stable_seed("dct-randaugment-internal", key)
            y, cbcr = _apply_operation(
                dops,
                y,
                cbcr,
                operation,
                magnitude,
                internal_seed=internal_seed,
            )
            sample_records.append(
                {
                    "operation_index": operation_index,
                    "key": key,
                    "operation": operation,
                    "magnitude": magnitude,
                    "internal_seed": internal_seed,
                }
            )
        output_y.append((y.float() + 4.0) / 1020.0)
        output_cbcr.append((cbcr.float() + 4.0) / 1020.0)
        records.append(sample_records)
    return (torch.stack(output_y), torch.stack(output_cbcr)), records


def _posterize_batch(value: torch.Tensor, bitoffset: int) -> torch.Tensor:
    """Published posterize with one leading microbatch dimension."""

    minimum, maximum = -1024, 1016
    result = value.clone()
    dc = result[..., 0, 0].to(torch.float32) - minimum
    if bool((dc < 0).any().item()):
        raise ValueError("posterize received a coefficient below the published clamp")
    indices = torch.round(dc / (2**bitoffset)).to(torch.int64)
    table = torch.linspace(
        minimum,
        maximum,
        round((maximum - minimum) / (2**bitoffset)) + 1,
        dtype=torch.float32,
        device=result.device,
    )
    result[..., 0, 0] = torch.round(table[indices]).to(result.dtype)
    return result


def _autocontrast_batch(value: torch.Tensor) -> torch.Tensor:
    """Published autocontrast independently over each selected sample."""

    result = value.clone()
    dc = result[..., 0, 0].to(torch.float32)
    reduce_dims = tuple(range(1, dc.ndim))
    minimum = dc.amin(dim=reduce_dims, keepdim=True)
    maximum = dc.amax(dim=reduce_dims, keepdim=True)
    zero_constant = (minimum == maximum) & (maximum == 0)
    divisor = torch.where(
        zero_constant, torch.ones_like(maximum), maximum - minimum
    )
    scaled = -1024.0 + ((dc - minimum) / divisor) * 2040.0
    scaled = torch.where(zero_constant, dc, scaled)
    result[..., 0, 0] = torch.round(scaled).to(result.dtype)
    return result


def _contrast_batch(value: torch.Tensor, factor: float) -> torch.Tensor:
    result = value.clone()
    dc = result[..., 0, 0].to(torch.float32)
    result[..., 0, 0] = torch.round(dc * factor).to(result.dtype)
    return result


def _brightness_batch(value: torch.Tensor, factor: float) -> torch.Tensor:
    result = value.clone()
    dc = result[..., 0, 0].to(torch.float32)
    reduce_dims = tuple(range(1, dc.ndim))
    offset = dc.abs().mean(dim=reduce_dims, keepdim=True) * (factor - 1.0)
    result[..., 0, 0] = torch.round(dc + offset).to(result.dtype)
    return result


def _cutout_batch(
    value: torch.Tensor,
    pad: int,
    centers_h: Sequence[int],
    centers_w: Sequence[int],
) -> torch.Tensor:
    """Vector form of RGB-no-more's deliberately asymmetric pad construction."""

    result = value.clone()
    _, _, height, width, _, _ = result.shape
    rows = torch.arange(height, device=result.device).view(1, height, 1)
    columns = torch.arange(width, device=result.device).view(1, 1, width)
    center_h = torch.as_tensor(centers_h, device=result.device).view(-1, 1, 1)
    center_w = torch.as_tensor(centers_w, device=result.device).view(-1, 1, 1)
    lower = torch.clamp(center_h - pad, min=0)
    upper = torch.clamp(height - center_h - pad, min=0)
    left = torch.clamp(center_w - pad, min=0)
    right = torch.clamp(width - center_w - pad, min=0)
    inside = (
        (rows >= upper)
        & (rows < height - lower)
        & (columns >= left)
        & (columns < width - right)
    )
    return result.masked_fill(inside[:, None, :, :, None, None], 0)


def _translate_batch(
    value: torch.Tensor, blocks: int, direction: str
) -> torch.Tensor:
    result = value.clone()
    dimension = 3 if direction == "W" else 2
    result = torch.roll(result, blocks, dims=(dimension,))
    selection = [slice(None)] * result.ndim
    selection[dimension] = slice(None, blocks) if blocks >= 0 else slice(blocks, None)
    result[tuple(selection)] = 0
    return result


def _flip_fixed_batch(value: torch.Tensor, direction: str) -> torch.Tensor:
    result = value.clone()
    if direction == "horizontal":
        result[..., 1::2] *= -1
    else:
        result[..., 1::2, :] *= -1
    return result


def _rotate_batch(value: torch.Tensor, rotate: float) -> torch.Tensor:
    result = value.clone()
    if rotate == -1:
        result = torch.rot90(result, k=-1, dims=(2, 3)).transpose(-2, -1)
        return _flip_fixed_batch(result, "horizontal")
    if rotate == 1:
        result = torch.rot90(result, k=1, dims=(2, 3)).transpose(-2, -1)
        return _flip_fixed_batch(result, "vertical")
    raise ValueError(f"unsupported published Rotate90 magnitude {rotate}")


def _apply_operation_batch(
    dops: Any,
    y: torch.Tensor,
    cbcr: torch.Tensor,
    operation: str,
    magnitude: float,
    *,
    internal_seeds: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one common operation/magnitude to selected microbatch rows."""

    if operation == "AutoContrast":
        y = _autocontrast_batch(y)
    elif operation == "Posterize":
        y = _posterize_batch(y, int(magnitude))
        cbcr = _posterize_batch(cbcr, int(magnitude))
    elif operation == "SolarizeAdd":
        y = y.clone()
        dc = y[..., 0, 0]
        dc[dc < 0] += int(magnitude)
        y[..., 0, 0] = dc
    elif operation == "Color":
        cbcr = _contrast_batch(cbcr, 1.0 + magnitude)
    elif operation == "Contrast":
        y = _contrast_batch(y, 1.0 + magnitude)
    elif operation == "Brightness":
        y = _brightness_batch(y, 1.0 + magnitude)
    elif operation == "MidfreqAug":
        y = _midfreqaug_device_safe(dops, y, magnitude)
    elif operation == "Cutout":
        pad = int(round(magnitude))
        pad -= pad % 2
        centers_h: list[int] = []
        centers_w: list[int] = []
        for seed in internal_seeds:
            generator = _torch_generator(seed)
            centers_h.append(
                int(torch.randint(0, y.shape[2], (1,), generator=generator).item())
                // 2
                * 2
            )
            centers_w.append(
                int(torch.randint(0, y.shape[3], (1,), generator=generator).item())
                // 2
                * 2
            )
        y = _cutout_batch(y, pad, centers_h, centers_w)
        cbcr = _cutout_batch(
            cbcr,
            pad // 2,
            [center // 2 for center in centers_h],
            [center // 2 for center in centers_w],
        )
    elif operation in ("TranslateX", "TranslateY"):
        blocks = int(magnitude - (magnitude % 2))
        direction = "W" if operation == "TranslateX" else "H"
        y = _translate_batch(y, blocks, direction)
        cbcr = _translate_batch(cbcr, blocks // 2, direction)
    elif operation == "Rotate90":
        y = _rotate_batch(y, magnitude)
        cbcr = _rotate_batch(cbcr, magnitude)
    elif operation == "AutoSaturation":
        cbcr = _autocontrast_batch(cbcr)
    elif operation == "Grayscale":
        cbcr = cbcr * 0
    elif operation == "ChromaDrop":
        cbcr = cbcr.clone()
        channels = torch.as_tensor(
            [stable_seed("chroma-drop", seed) % 2 for seed in internal_seeds],
            dtype=torch.long,
            device=cbcr.device,
        )
        cbcr[torch.arange(cbcr.shape[0], device=cbcr.device), channels] = 0
    else:
        raise ValueError(f"unrecognized published DCT RandAugment operation {operation}")
    return (
        y.clamp(min=-1024, max=1016).contiguous(),
        cbcr.clamp(min=-1024, max=1016).contiguous(),
    )


def apply_published_randaugment(
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    training_seed: int,
    epoch: int,
    logical_sample_ids: Sequence[str],
    rgbnomore_root: Path,
) -> tuple[tuple[torch.Tensor, torch.Tensor], list[list[dict[str, Any]]]]:
    """Apply exact keyed semantics, grouped by operation to avoid tiny GPU calls."""

    y, cbcr = (normalized_to_published_int16(value) for value in inputs)
    if len(logical_sample_ids) != y.shape[0] or y.shape[0] != cbcr.shape[0]:
        raise ValueError("RandAugment identities do not match DCT batch cardinality")
    dops = _dct_ops(rgbnomore_root)
    records: list[list[dict[str, Any]]] = [[] for _ in logical_sample_ids]
    available = [list(DCT_RANDAUGMENT_OPERATIONS) for _ in logical_sample_ids]
    chroma_operations = {"Grayscale", "Color", "AutoSaturation", "ChromaDrop"}
    for operation_index in range(2):
        groups: dict[tuple[str, float], list[tuple[int, int]]] = defaultdict(list)
        for index, logical_id in enumerate(logical_sample_ids):
            key = randaugment_key(
                training_seed=training_seed,
                epoch=epoch,
                logical_sample_id=logical_id,
                operation_index=operation_index,
            )
            choice_seed = stable_seed(
                "dct-randaugment-choice",
                training_seed,
                epoch,
                logical_id,
                operation_index,
            )
            operation = available[index][choice_seed % len(available[index])]
            if operation in chroma_operations:
                if operation == "Grayscale":
                    available[index] = [
                        name for name in available[index] if name not in chroma_operations
                    ]
                else:
                    available[index] = [
                        name for name in available[index] if name != "Grayscale"
                    ]
            magnitude, signed = _magnitude(operation)
            if signed and stable_seed("dct-randaugment-sign", key) % 2:
                magnitude *= -1.0
            internal_seed = stable_seed("dct-randaugment-internal", key)
            groups[(operation, magnitude)].append((index, internal_seed))
            records[index].append(
                {
                    "operation_index": operation_index,
                    "key": key,
                    "operation": operation,
                    "magnitude": magnitude,
                    "internal_seed": internal_seed,
                }
            )
        for (operation, magnitude), members in groups.items():
            indices = torch.as_tensor(
                [index for index, _seed in members], device=y.device
            )
            group_y, group_cbcr = _apply_operation_batch(
                dops,
                y.index_select(0, indices),
                cbcr.index_select(0, indices),
                operation,
                magnitude,
                internal_seeds=[seed for _index, seed in members],
            )
            y.index_copy_(0, indices, group_y)
            cbcr.index_copy_(0, indices, group_cbcr)
    return ((y.float() + 4.0) / 1020.0, (cbcr.float() + 4.0) / 1020.0), records


def apply_published_mixup(
    inputs: tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    *,
    training_seed: int,
    epoch: int,
    microbatch_index: int,
    alpha: float = 0.2,
    classes: int = 1000,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, dict[str, Any]]:
    key = mixup_key(
        training_seed=training_seed,
        epoch=epoch,
        microbatch_index=microbatch_index,
    )
    generator_seed = stable_seed("dct-mixup-dirichlet", key)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(generator_seed % (2**63 - 1))
        components = torch._sample_dirichlet(
            torch.tensor([alpha, alpha], dtype=torch.float32)
        ).sort(descending=True).values
    original = float(components[0].item())
    rolled = float(components[1].item())
    mixed_inputs = tuple(
        value * original + value.roll(1, dims=0) * rolled for value in inputs
    )
    one_hot = torch.nn.functional.one_hot(labels, num_classes=classes).to(
        dtype=inputs[0].dtype
    )
    mixed_labels = one_hot * original + one_hot.roll(1, dims=0) * rolled
    return mixed_inputs, mixed_labels, {
        "key": key,
        "seed": generator_seed,
        "lambda_original": original,
        "lambda_rolled": rolled,
        "pairing": "roll-by-one",
    }
