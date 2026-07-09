#!/usr/bin/env python3
"""PyTorch RGB baseline using RGB-no-more ViT-Ti.

The GALP Direct-DCT benchmark needs a comparable RGB baseline with the same
1000-class ImageNet head and RGB-no-more checkpoint. This script measures
loader_to_device, forward_step, and optional train_step phases with
RGB-no-more's ViT-Ti RGB model.
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
from torchvision import datasets, transforms


DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_RGB_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints" / "imgnetRGBViTTi_ep300_74.1.pth"
DEFAULT_VAL_DIR = Path("/tmp/rgbnomore_imagenet/val")
RGB_PREPROCESS = "resize256_centercrop224_to_minus_one_one"


class ToMinusOneOne:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.mul(2.0).sub(1.0)


def _sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _import_rgbnomore_model(rgbnomore_root: Path) -> Any:
    root = str(rgbnomore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("models.plainvit")


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


def make_imagenet_val_loader(data_dir: Path, batch_size: int, workers: int) -> torch.utils.data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ToMinusOneOne(),
        ]
    )
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def _iter_steps(loader: torch.utils.data.DataLoader, steps: int):
    iterator = iter(loader)
    for _ in range(steps):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


def run_loader_phase(loader: torch.utils.data.DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    for images, labels in _iter_steps(loader, args.warmup):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _sync(device)

    _sync(device)
    total_images = 0
    started = time.perf_counter()
    first_shape: list[int] | None = None
    for images, labels in _iter_steps(loader, args.steps):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if first_shape is None:
            first_shape = list(images.shape)
            print(f"phase=loader_to_device step=0 input_shape={tuple(images.shape)} labels_shape={tuple(labels.shape)}")
        total_images += int(images.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "pytorch_rgb_rgbnomore",
        "phase": "loader_to_device",
        "model": None,
        "dataset_size": len(loader.dataset),
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
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_forward_phase(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    images, _labels = next(_iter_steps(loader, 1))
    images = images.to(device, non_blocking=True)
    _sync(device)
    for _ in range(args.warmup):
        with torch.no_grad():
            logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    total_images = 0
    logits_shape: list[int] | None = None
    input_shape = list(images.shape)
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
        "backend": "pytorch_rgb_rgbnomore",
        "phase": "forward_step",
        "model": "rgbnomore_rgb_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": len(loader.dataset),
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
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_end_to_end_phase(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    for images, _labels in _iter_steps(loader, args.warmup):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    total_images = 0
    logits_shape: list[int] | None = None
    input_shape: list[int] | None = None
    started = time.perf_counter()
    for images, _labels in _iter_steps(loader, args.steps):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        if logits_shape is None:
            logits_shape = list(logits.shape)
            input_shape = list(images.shape)
            print(f"phase=end_to_end step=0 input_shape={tuple(images.shape)} logits_shape={tuple(logits.shape)}")
        total_images += int(images.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "pytorch_rgb_rgbnomore",
        "phase": "end_to_end",
        "model": "rgbnomore_rgb_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": len(loader.dataset),
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
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def run_train_phase(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr)

    for images, labels in _iter_steps(loader, args.warmup):
        images = images.to(device, non_blocking=True)
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
    for images, labels in _iter_steps(loader, args.steps):
        images = images.to(device, non_blocking=True)
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
        "backend": "pytorch_rgb_rgbnomore",
        "phase": "train_step",
        "model": "rgbnomore_rgb_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": len(loader.dataset),
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
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-no-more ViT-Ti PyTorch RGB baseline benchmark")
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
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.train_lr < 0.0:
        raise ValueError("--train-lr must be non-negative")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    loader = make_imagenet_val_loader(args.data_dir, args.batch_size, args.workers)
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
