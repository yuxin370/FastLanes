#!/usr/bin/env python3
"""RGB-no-more native DCT baseline benchmark.

This is the baseline that GALP Direct-DCT should be compared against for the
JPEG-Ti path. It reuses RGB-no-more's ImageNet DCT dataset code, including
dct_manip.read_coefficients, quant-table dequantization, ResizedCenterCrop_DCT,
and ToRange preprocessing.
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
DEFAULT_DCT_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints" / "imgnetDCTViTTi_ep300_75.1.pth"
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_VAL_INDEX = DEFAULT_RGBNOMORE_ROOT / "assets" / "indexbase_val.csv"
DEFAULT_TRAIN_INDEX = DEFAULT_RGBNOMORE_ROOT / "assets" / "indexbase_train.csv"
RGBNOMORE_DCT_EVAL_PREPROCESS = "resized_center_crop_dct_32_to_28_to_minus_one_one"
RGBNOMORE_DCT_TRAIN_PREPROCESS = "random_resized_crop_dct_28_flip_randaugment_to_minus_one_one"


def _sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _import_from_rgbnomore(rgbnomore_root: Path, module: str) -> Any:
    root = str(rgbnomore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module)


def _dataset_type(args: argparse.Namespace) -> str:
    return "train" if args.split == "train" and not args.eval_transform else "test"


def _dct_preprocess_name(args: argparse.Namespace) -> str:
    if _dataset_type(args) == "train":
        return RGBNOMORE_DCT_TRAIN_PREPROCESS
    return RGBNOMORE_DCT_EVAL_PREPROCESS


def build_rgbnomore_jpeg_ti(
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    plainvit = _import_from_rgbnomore(rgbnomore_root, "models.plainvit")
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


def make_rgbnomore_dct_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    rgbnomore_datasets = _import_from_rgbnomore(args.rgbnomore_root, "datasets")
    dataset_type = _dataset_type(args)
    dataset = rgbnomore_datasets.imagenet_dataset_indexing(
        indexfile=str(args.index_file),
        type=dataset_type,
        basepath=str(args.data_root),
        load_mode="DCT",
        dtype=torch.float32,
    )
    transform = rgbnomore_datasets.get_transform(
        dataset="imagenet_dct",
        type=dataset_type,
        dtype=torch.float32,
    )
    subset = rgbnomore_datasets.SubsetWithTransform(dataset, dataset="imagenet_dct", transform=transform)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "shuffle": args.shuffle,
        "num_workers": args.workers,
        "collate_fn": None,
        "generator": generator,
    }
    if args.workers > 0:
        dataloader_kwargs["prefetch_factor"] = 8
        dataloader_kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(subset, **dataloader_kwargs)


def make_rgbnomore_dct_loader_via_selector(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    rgbnomore_datasets = _import_from_rgbnomore(args.rgbnomore_root, "datasets")
    dataset_type = _dataset_type(args)
    return rgbnomore_datasets.dataset_selector(
        dataset="imagenet_dct",
        type=dataset_type,
        indexpath=str(args.index_file),
        basepath=str(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=args.shuffle,
        trainval_split=-1,
        distributed=False,
        seed=args.seed,
        dtype=torch.float32,
    )


def _iter_steps(loader: torch.utils.data.DataLoader, steps: int):
    iterator = iter(loader)
    for _ in range(steps):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


def _move_dct_batch_to_device(
    batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (input_y, input_cbcr), labels = batch
    input_y = input_y.to(device, non_blocking=True)
    input_cbcr = input_cbcr.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return input_y, input_cbcr, labels


def _validate_dct_shapes(input_y: torch.Tensor, input_cbcr: torch.Tensor) -> None:
    if tuple(input_y.shape[1:]) != (1, 28, 28, 8, 8):
        raise RuntimeError(f"expected RGB-no-more Y shape tail (1,28,28,8,8), got {tuple(input_y.shape)}")
    if tuple(input_cbcr.shape[1:]) != (2, 14, 14, 8, 8):
        raise RuntimeError(f"expected RGB-no-more CbCr shape tail (2,14,14,8,8), got {tuple(input_cbcr.shape)}")


def run_loader_phase(
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    for batch in _iter_steps(loader, args.warmup):
        input_y, input_cbcr, _labels = _move_dct_batch_to_device(batch, device)
        _validate_dct_shapes(input_y, input_cbcr)
        _sync(device)

    _sync(device)
    total_images = 0
    first_y_shape: list[int] | None = None
    first_cbcr_shape: list[int] | None = None
    started = time.perf_counter()
    for batch in _iter_steps(loader, args.steps):
        input_y, input_cbcr, labels = _move_dct_batch_to_device(batch, device)
        _validate_dct_shapes(input_y, input_cbcr)
        if first_y_shape is None:
            first_y_shape = list(input_y.shape)
            first_cbcr_shape = list(input_cbcr.shape)
            print(
                "phase=loader_to_device step=0 "
                f"y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)} "
                f"labels_shape={tuple(labels.shape)}"
            )
        total_images += int(input_y.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "rgbnomore_native_dct",
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
        "data_root": str(args.data_root),
        "index_file": str(args.index_file),
        "split": args.split,
        "eval_transform": args.eval_transform,
        "dct_preprocess": _dct_preprocess_name(args),
        "input_y_shape": first_y_shape,
        "input_cbcr_shape": first_cbcr_shape,
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
    batch = next(_iter_steps(loader, 1))
    input_y, input_cbcr, _labels = _move_dct_batch_to_device(batch, device)
    _validate_dct_shapes(input_y, input_cbcr)
    _sync(device)
    for _ in range(args.warmup):
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    total_images = 0
    logits_shape: list[int] | None = None
    first_y_shape = list(input_y.shape)
    first_cbcr_shape = list(input_cbcr.shape)
    started = time.perf_counter()
    for step in range(args.steps):
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        if logits_shape is None:
            logits_shape = list(logits.shape)
            print(
                f"phase=forward_step step={step} "
                f"y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)} "
                f"logits_shape={tuple(logits.shape)}"
            )
        total_images += int(input_y.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "rgbnomore_native_dct",
        "phase": "forward_step",
        "model": "rgbnomore_jpeg_ti_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": len(loader.dataset),
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "workers": args.workers,
        "data_root": str(args.data_root),
        "index_file": str(args.index_file),
        "split": args.split,
        "eval_transform": args.eval_transform,
        "dct_preprocess": _dct_preprocess_name(args),
        "input_y_shape": first_y_shape,
        "input_cbcr_shape": first_cbcr_shape,
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
    for batch in _iter_steps(loader, args.warmup):
        input_y, input_cbcr, _labels = _move_dct_batch_to_device(batch, device)
        _validate_dct_shapes(input_y, input_cbcr)
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        _sync(device)

    _sync(device)
    total_images = 0
    logits_shape: list[int] | None = None
    first_y_shape: list[int] | None = None
    first_cbcr_shape: list[int] | None = None
    started = time.perf_counter()
    for batch in _iter_steps(loader, args.steps):
        input_y, input_cbcr, _labels = _move_dct_batch_to_device(batch, device)
        _validate_dct_shapes(input_y, input_cbcr)
        with torch.no_grad():
            logits = model(input_y, input_cbcr)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        if logits_shape is None:
            logits_shape = list(logits.shape)
            first_y_shape = list(input_y.shape)
            first_cbcr_shape = list(input_cbcr.shape)
            print(
                "phase=end_to_end step=0 "
                f"y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)} "
                f"logits_shape={tuple(logits.shape)}"
            )
        total_images += int(input_y.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "rgbnomore_native_dct",
        "phase": "end_to_end",
        "model": "rgbnomore_jpeg_ti_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": len(loader.dataset),
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "workers": args.workers,
        "data_root": str(args.data_root),
        "index_file": str(args.index_file),
        "split": args.split,
        "eval_transform": args.eval_transform,
        "dct_preprocess": _dct_preprocess_name(args),
        "input_y_shape": first_y_shape,
        "input_cbcr_shape": first_cbcr_shape,
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

    for batch in _iter_steps(loader, args.warmup):
        input_y, input_cbcr, labels = _move_dct_batch_to_device(batch, device)
        _validate_dct_shapes(input_y, input_cbcr)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_y, input_cbcr)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        _sync(device)

    _sync(device)
    total_images = 0
    logits_shape: list[int] | None = None
    first_y_shape: list[int] | None = None
    first_cbcr_shape: list[int] | None = None
    last_loss: float | None = None
    started = time.perf_counter()
    for batch in _iter_steps(loader, args.steps):
        input_y, input_cbcr, labels = _move_dct_batch_to_device(batch, device)
        _validate_dct_shapes(input_y, input_cbcr)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_y, input_cbcr)
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"expected 1000-class logits, got {tuple(logits.shape)}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if logits_shape is None:
            logits_shape = list(logits.shape)
            first_y_shape = list(input_y.shape)
            first_cbcr_shape = list(input_cbcr.shape)
            print(
                "phase=train_step step=0 "
                f"y_shape={tuple(input_y.shape)} cbcr_shape={tuple(input_cbcr.shape)} "
                f"logits_shape={tuple(logits.shape)} loss={float(loss.detach().cpu())}"
            )
        last_loss = float(loss.detach().cpu())
        total_images += int(input_y.shape[0])
    _sync(device)
    seconds = time.perf_counter() - started
    result = {
        "backend": "rgbnomore_native_dct",
        "phase": "train_step",
        "model": "rgbnomore_jpeg_ti_vitti",
        "checkpoint": str(args.checkpoint),
        "dataset_size": len(loader.dataset),
        "images": total_images,
        "seconds": seconds,
        "images_per_s": total_images / seconds if seconds > 0.0 else float("inf"),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "workers": args.workers,
        "data_root": str(args.data_root),
        "index_file": str(args.index_file),
        "split": args.split,
        "eval_transform": args.eval_transform,
        "dct_preprocess": _dct_preprocess_name(args),
        "input_y_shape": first_y_shape,
        "input_cbcr_shape": first_cbcr_shape,
        "logits_shape": logits_shape,
        "loss": last_loss,
        "optimizer": "sgd",
        "train_lr": args.train_lr,
        "device": str(device),
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-no-more native JPEG-Ti DCT benchmark")
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DCT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index-file", type=Path)
    parser.add_argument("--split", choices=("val", "train", "inference"), default="val")
    parser.add_argument(
        "--eval-transform",
        action="store_true",
        help="Use the validation/test DCT transform even when --split train.",
    )
    parser.add_argument("--phase", choices=("loader", "forward", "end-to-end", "train", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-lr", type=float, default=0.0, help="SGD learning rate used only for --phase train.")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.index_file is None:
        args.index_file = DEFAULT_TRAIN_INDEX if args.split == "train" else DEFAULT_VAL_INDEX
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
    if not args.rgbnomore_root.exists():
        raise FileNotFoundError(args.rgbnomore_root)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    if not args.index_file.exists():
        raise FileNotFoundError(args.index_file)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    loader = make_rgbnomore_dct_loader(args)
    results: list[dict[str, Any]] = []
    if args.phase in ("loader", "both"):
        results.append(run_loader_phase(loader, args, device))
    if args.phase in ("forward", "both"):
        model = build_rgbnomore_jpeg_ti(args.rgbnomore_root, args.checkpoint, device)
        results.append(run_forward_phase(loader, model, args, device))
    if args.phase in ("end-to-end", "both"):
        model = build_rgbnomore_jpeg_ti(args.rgbnomore_root, args.checkpoint, device)
        results.append(run_end_to_end_phase(loader, model, args, device))
    if args.phase == "train":
        model = build_rgbnomore_jpeg_ti(args.rgbnomore_root, args.checkpoint, device)
        results.append(run_train_phase(loader, model, args, device))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
