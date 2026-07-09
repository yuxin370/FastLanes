#!/usr/bin/env python3
"""DALI RGB baseline using RGB-no-more ViT-Ti.

This is the optional DALI counterpart to rgbnomore_rgb_baseline_benchmark.py.
It uses the same RGB-no-more RGB ViT-Ti checkpoint and emits the same
loader_to_device / forward_step / train_step JSON records, but the ImageNet
decode and resize/crop/normalize path is handled by NVIDIA DALI.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_RGB_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints" / "imgnetRGBViTTi_ep300_74.1.pth"
DEFAULT_VAL_DIR = Path("/tmp/rgbnomore_imagenet/val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
RGB_PREPROCESS = "resize256_centercrop224_to_minus_one_one"


def _sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _import_dali() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from nvidia.dali import fn, types
        from nvidia.dali.pipeline import pipeline_def
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    except ImportError as exc:
        raise RuntimeError(
            "nvidia.dali is not installed; install NVIDIA DALI to run the DALI RGB baseline"
        ) from exc
    return fn, types, pipeline_def, DALIClassificationIterator, LastBatchPolicy


def _import_rgbnomore_model(rgbnomore_root: Path) -> Any:
    root = str(rgbnomore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("models.plainvit")


def _count_image_files(data_dir: Path) -> int:
    return sum(1 for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _resolve_cuda_device(device_arg: str, device_id: int) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise RuntimeError("DALI RGB baseline currently requires --device cuda")
    if device.index is not None and device.index != device_id:
        raise RuntimeError(f"--device-id {device_id} does not match --device {device}")
    torch.cuda.set_device(device_id if device.index is None else device.index)
    return device


def build_rgbnomore_rgb_ti(rgbnomore_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
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
        pixel_space="RGB",
    )
    checkpoint_obj = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_obj.get("model_state_dict", checkpoint_obj)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_dali_loader(args: argparse.Namespace, device_id: int) -> Any:
    fn, types, pipeline_def, DALIClassificationIterator, LastBatchPolicy = _import_dali()

    @pipeline_def
    def imagenet_rgb_pipeline() -> tuple[Any, Any]:
        images, labels = fn.readers.file(file_root=str(args.data_dir), random_shuffle=False, name="Reader")
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_shorter=256)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        )
        return images, labels

    pipeline = imagenet_rgb_pipeline(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=device_id,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    return DALIClassificationIterator(
        pipeline,
        reader_name="Reader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )


def _next_batch(loader: Any) -> tuple[torch.Tensor, torch.Tensor]:
    batch = next(loader)[0]
    images = batch["data"]
    labels = batch["label"].squeeze(-1).long()
    return images, labels


def run_loader_phase(loader: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    for _ in range(args.warmup):
        images, labels = _next_batch(loader)
        labels = labels.to(device, non_blocking=True)
        _sync(device)

    _sync(device)
    total_images = 0
    first_shape: list[int] | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        images, labels = _next_batch(loader)
        labels = labels.to(device, non_blocking=True)
        if first_shape is None:
            first_shape = list(images.shape)
            print(f"phase=loader_to_device step={step} input_shape={tuple(images.shape)} labels_shape={tuple(labels.shape)}")
        total_images += int(images.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "dali_rgb_rgbnomore",
        "phase": "loader_to_device",
        "model": None,
        "dataset_size": args.dataset_size,
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "workers": args.workers,
        "data_dir": str(args.data_dir),
        "split": args.split_label,
        "rgb_preprocess": RGB_PREPROCESS,
        "input_shape": first_shape,
        "device": str(device),
        "prefetch_queue_depth": args.prefetch_queue_depth,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_forward_phase(loader: Any, model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    images, _labels = _next_batch(loader)
    _sync(device)
    for _ in range(args.warmup):
        with torch.no_grad():
            logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    total_images = 0
    input_shape = list(images.shape)
    logits_shape: list[int] | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        with torch.no_grad():
            logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        if logits_shape is None:
            logits_shape = list(logits.shape)
            print(f"phase=forward_step step={step} input_shape={tuple(images.shape)} logits_shape={tuple(logits.shape)}")
        total_images += int(images.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "dali_rgb_rgbnomore",
        "phase": "forward_step",
        "model": "rgbnomore_rgb_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": args.dataset_size,
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "workers": args.workers,
        "data_dir": str(args.data_dir),
        "split": args.split_label,
        "rgb_preprocess": RGB_PREPROCESS,
        "input_shape": input_shape,
        "logits_shape": logits_shape,
        "device": str(device),
        "prefetch_queue_depth": args.prefetch_queue_depth,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_end_to_end_phase(loader: Any, model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    for _ in range(args.warmup):
        images, _labels = _next_batch(loader)
        with torch.no_grad():
            logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    total_images = 0
    input_shape: list[int] | None = None
    logits_shape: list[int] | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        images, _labels = _next_batch(loader)
        with torch.no_grad():
            logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        if logits_shape is None:
            input_shape = list(images.shape)
            logits_shape = list(logits.shape)
            print(f"phase=end_to_end step={step} input_shape={tuple(images.shape)} logits_shape={tuple(logits.shape)}")
        total_images += int(images.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "dali_rgb_rgbnomore",
        "phase": "end_to_end",
        "model": "rgbnomore_rgb_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": args.dataset_size,
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "workers": args.workers,
        "data_dir": str(args.data_dir),
        "split": args.split_label,
        "rgb_preprocess": RGB_PREPROCESS,
        "input_shape": input_shape,
        "logits_shape": logits_shape,
        "device": str(device),
        "prefetch_queue_depth": args.prefetch_queue_depth,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_train_phase(loader: Any, model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr)

    for _ in range(args.warmup):
        images, labels = _next_batch(loader)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        _sync(device)

    _sync(device)
    total_images = 0
    input_shape: list[int] | None = None
    logits_shape: list[int] | None = None
    last_loss: float | None = None
    started = time.perf_counter()
    for step in range(args.steps):
        images, labels = _next_batch(loader)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if logits_shape is None:
            input_shape = list(images.shape)
            logits_shape = list(logits.shape)
            print(
                "phase=train_step step=0 "
                f"input_shape={tuple(images.shape)} logits_shape={tuple(logits.shape)} loss={float(loss.detach().cpu())}"
            )
        last_loss = float(loss.detach().cpu())
        total_images += int(images.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "dali_rgb_rgbnomore",
        "phase": "train_step",
        "model": "rgbnomore_rgb_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": args.dataset_size,
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "workers": args.workers,
        "data_dir": str(args.data_dir),
        "split": args.split_label,
        "rgb_preprocess": RGB_PREPROCESS,
        "input_shape": input_shape,
        "logits_shape": logits_shape,
        "loss": last_loss,
        "optimizer": "sgd",
        "train_lr": args.train_lr,
        "device": str(device),
        "prefetch_queue_depth": args.prefetch_queue_depth,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-no-more ViT-Ti DALI RGB baseline benchmark")
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RGB_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--split-label", choices=("val", "train", "inference"), default="val")
    parser.add_argument("--phase", choices=("loader", "forward", "end-to-end", "train", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prefetch-queue-depth", type=int, default=2)
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
    if args.workers <= 0:
        raise ValueError("--workers must be positive for DALI")
    if args.prefetch_queue_depth <= 0:
        raise ValueError("--prefetch-queue-depth must be positive")
    if args.train_lr < 0.0:
        raise ValueError("--train-lr must be non-negative")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    args.dataset_size = _count_image_files(args.data_dir)
    if args.dataset_size <= 0:
        raise RuntimeError(f"no image files found under {args.data_dir}")

    _import_dali()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = _resolve_cuda_device(args.device, args.device_id)

    loader = make_dali_loader(args, args.device_id)
    results: list[dict[str, Any]] = []
    if args.phase in ("loader", "both"):
        results.append(run_loader_phase(loader, args, device))
    if args.phase in ("forward", "both"):
        model = build_rgbnomore_rgb_ti(args.rgbnomore_root, args.checkpoint, device)
        results.append(run_forward_phase(loader, model, args, device))
    if args.phase in ("end-to-end", "both"):
        model = build_rgbnomore_rgb_ti(args.rgbnomore_root, args.checkpoint, device)
        results.append(run_end_to_end_phase(loader, model, args, device))
    if args.phase == "train":
        model = build_rgbnomore_rgb_ti(args.rgbnomore_root, args.checkpoint, device)
        results.append(run_train_phase(loader, model, args, device))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
