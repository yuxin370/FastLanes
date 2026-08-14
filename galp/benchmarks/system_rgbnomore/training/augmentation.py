#!/usr/bin/env python3
"""Worker-independent deterministic v1 training augmentation."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from training.schema import TRAINING_AUGMENTATION_SCHEMA


@dataclass(frozen=True)
class AugmentationDecision:
    seed: int
    epoch: int
    logical_sample_id: str
    source_width: int
    source_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    resize_width: int
    resize_height: int
    horizontal_flip: bool
    interpolation: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    domain: str
    dct_crop_alignment_pixels: int | None = None

    @property
    def augmentation_key(self) -> str:
        return hashlib.sha256(
            f"galp-training-augmentation-v1:{self.seed}:{self.epoch}:{self.logical_sample_id}".encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["normalization_mean"] = list(self.normalization_mean)
        result["normalization_std"] = list(self.normalization_std)
        result["augmentation_key"] = self.augmentation_key
        return result

    def native_dct_descriptor(self) -> dict[str, Any]:
        if self.domain != "dct":
            raise ValueError("native DCT descriptor requested for RGB augmentation")
        return {
            "logical_sample_id": self.logical_sample_id,
            "augmentation_key": self.augmentation_key,
            "crop": {
                "x": self.crop_x,
                "y": self.crop_y,
                "width": self.crop_width,
                "height": self.crop_height,
                "unit": "source_pixels",
            },
            "horizontal_flip": self.horizontal_flip,
            "resize": [self.resize_height, self.resize_width],
            "interpolation": self.interpolation,
        }


def _decision_rng(seed: int, epoch: int, logical_sample_id: str) -> random.Random:
    digest = hashlib.sha256(
        f"galp-training-augmentation-v1:{seed}:{epoch}:{logical_sample_id}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "little"))


def _shared_crop_rng(seed: int, epoch: int, physical_shard_id: int) -> random.Random:
    digest = hashlib.sha256(
        f"galp-pls-shared-crop-v1:{seed}:{epoch}:{physical_shard_id}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "little"))


def _random_resized_crop(
    rng: random.Random,
    width: int,
    height: int,
    *,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> tuple[int, int, int, int]:
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(*log_ratio))
        crop_width = int(round(math.sqrt(target_area * aspect)))
        crop_height = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            x = rng.randint(0, width - crop_width)
            y = rng.randint(0, height - crop_height)
            return x, y, crop_width, crop_height
    input_ratio = width / height
    if input_ratio < ratio[0]:
        crop_width = width
        crop_height = int(round(crop_width / ratio[0]))
    elif input_ratio > ratio[1]:
        crop_height = height
        crop_width = int(round(crop_height * ratio[1]))
    else:
        crop_width, crop_height = width, height
    return (width - crop_width) // 2, (height - crop_height) // 2, crop_width, crop_height


def _align_dct_crop_within_source(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    source_width: int,
    source_height: int,
    alignment: int,
) -> tuple[int, int, int, int]:
    if alignment <= 0:
        raise ValueError("DCT crop alignment must be positive")

    def align_axis(origin: int, extent: int, source_extent: int) -> tuple[int, int]:
        max_extent = (source_extent // alignment) * alignment
        aligned_extent = min(
            max_extent,
            max(alignment, (extent // alignment) * alignment),
        )
        max_origin = ((source_extent - aligned_extent) // alignment) * alignment
        aligned_origin = min((origin // alignment) * alignment, max_origin)
        return aligned_origin, aligned_extent

    x, width = align_axis(x, width, source_width)
    y, height = align_axis(y, height, source_height)
    return x, y, width, height


def derive_augmentation(
    *,
    seed: int,
    epoch: int,
    logical_sample_id: str,
    source_width: int,
    source_height: int,
    domain: str,
    output_size: int = 224,
    scale: tuple[float, float] = (0.05, 1.0),
    rgb_ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    dct_alignment_pixels: int = 16,
) -> AugmentationDecision:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    if domain not in ("rgb", "dct"):
        raise ValueError(f"unknown augmentation domain: {domain}")
    if domain == "dct" and dct_alignment_pixels <= 0:
        raise ValueError("DCT crop alignment must be positive")
    if domain == "dct" and (
        source_width < dct_alignment_pixels or source_height < dct_alignment_pixels
    ):
        raise ValueError(
            "DCT source dimensions must be at least one alignment unit "
            f"({dct_alignment_pixels}x{dct_alignment_pixels}); got "
            f"{source_width}x{source_height}"
        )
    rng = _decision_rng(seed, epoch, logical_sample_id)
    ratio = rgb_ratio if domain == "rgb" else (1.0, 1.0)
    x, y, width, height = _random_resized_crop(
        rng, source_width, source_height, scale=scale, ratio=ratio
    )
    alignment: int | None = None
    if domain == "dct":
        alignment = dct_alignment_pixels
        x, y, width, height = _align_dct_crop_within_source(
            x,
            y,
            width,
            height,
            source_width=source_width,
            source_height=source_height,
            alignment=alignment,
        )
    return AugmentationDecision(
        seed=seed,
        epoch=epoch,
        logical_sample_id=str(logical_sample_id),
        source_width=source_width,
        source_height=source_height,
        crop_x=x,
        crop_y=y,
        crop_width=width,
        crop_height=height,
        resize_width=output_size,
        resize_height=output_size,
        horizontal_flip=rng.random() < 0.5,
        interpolation="bilinear",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        domain=domain,
        dct_crop_alignment_pixels=alignment,
    )


def derive_shard_shared_crop_augmentation(
    *,
    seed: int,
    epoch: int,
    physical_shard_id: int,
    logical_sample_id: str,
    source_width: int,
    source_height: int,
    domain: str,
    output_size: int = 224,
    scale: tuple[float, float] = (0.05, 1.0),
    rgb_ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    dct_alignment_pixels: int = 16,
) -> AugmentationDecision:
    """Derive one crop per physical shard/epoch and a per-sample flip.

    Resetting the crop RNG from ``(seed, epoch, physical_shard_id)`` makes
    equal-sized images in a shard receive exactly the same pixel crop.  For a
    future ragged source shard, the same stochastic configuration is mapped
    through each image's dimensions.  Horizontal flip remains keyed by logical
    sample because it does not change the selected DCT block region.
    """

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    if physical_shard_id < 0:
        raise ValueError("physical_shard_id must be non-negative")
    if domain not in ("rgb", "dct"):
        raise ValueError(f"unknown augmentation domain: {domain}")
    if domain == "dct" and dct_alignment_pixels <= 0:
        raise ValueError("DCT crop alignment must be positive")
    if domain == "dct" and (
        source_width < dct_alignment_pixels or source_height < dct_alignment_pixels
    ):
        raise ValueError(
            "DCT source dimensions must be at least one alignment unit "
            f"({dct_alignment_pixels}x{dct_alignment_pixels}); got "
            f"{source_width}x{source_height}"
        )
    crop_rng = _shared_crop_rng(seed, epoch, physical_shard_id)
    ratio = rgb_ratio if domain == "rgb" else (1.0, 1.0)
    x, y, width, height = _random_resized_crop(
        crop_rng,
        source_width,
        source_height,
        scale=scale,
        ratio=ratio,
    )
    alignment: int | None = None
    if domain == "dct":
        alignment = dct_alignment_pixels
        x, y, width, height = _align_dct_crop_within_source(
            x,
            y,
            width,
            height,
            source_width=source_width,
            source_height=source_height,
            alignment=alignment,
        )
    flip_rng = _decision_rng(seed, epoch, logical_sample_id)
    return AugmentationDecision(
        seed=seed,
        epoch=epoch,
        logical_sample_id=str(logical_sample_id),
        source_width=source_width,
        source_height=source_height,
        crop_x=x,
        crop_y=y,
        crop_width=width,
        crop_height=height,
        resize_width=output_size,
        resize_height=output_size,
        horizontal_flip=flip_rng.random() < 0.5,
        interpolation="bilinear",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        domain=domain,
        dct_crop_alignment_pixels=alignment,
    )


def apply_rgb_augmentation_staged(
    image: Any, decision: AugmentationDecision
) -> tuple[Any, dict[str, float]]:
    if decision.domain != "rgb":
        raise ValueError("RGB transform requires an RGB decision")
    import time

    import torchvision.transforms.functional as function
    from torchvision.transforms import InterpolationMode

    begin = time.perf_counter()
    tensor = function.resized_crop(
        image,
        decision.crop_y,
        decision.crop_x,
        decision.crop_height,
        decision.crop_width,
        [decision.resize_height, decision.resize_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    if decision.horizontal_flip:
        tensor = function.hflip(tensor)
    augmentation_seconds = time.perf_counter() - begin

    begin = time.perf_counter()
    tensor = function.to_tensor(tensor)
    tensor = function.normalize(tensor, decision.normalization_mean, decision.normalization_std)
    return tensor, {
        "augmentation": augmentation_seconds,
        "preprocess": time.perf_counter() - begin,
    }


def apply_rgb_augmentation(image: Any, decision: AugmentationDecision):
    tensor, _stages = apply_rgb_augmentation_staged(image, decision)
    return tensor


def horizontal_flip_dct(y: Any, cbcr: Any) -> tuple[Any, Any]:
    """Flip DCT grids using JPEG's block reversal/odd-u sign rule."""

    def flip(value: Any):
        result = value.flip(dims=(-3,))
        result[..., 1::2] *= -1
        return result

    return flip(y), flip(cbcr)


def augmentation_contract() -> dict[str, Any]:
    return {
        "schema_version": TRAINING_AUGMENTATION_SCHEMA,
        "recipe": "deterministic-rrc-hflip-range-v1",
        "key": ["seed", "epoch", "logical_sample_id"],
        "random_resized_crop": {"scale": [0.05, 1.0], "rgb_ratio": [0.75, 4.0 / 3.0]},
        "dct_semantics": {
            "ratio": [1.0, 1.0],
            "crop_alignment_pixels": 16,
            "horizontal_flip": "reverse block columns and negate odd horizontal DCT frequencies",
            "cross_domain_tensor_equivalence": False,
        },
        "horizontal_flip_probability": 0.5,
        "resize": [224, 224],
        "interpolation": "bilinear",
        "range_normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "mixup": False,
        "cutmix": False,
        "randaugment": False,
        "worker_rng_used": False,
    }
