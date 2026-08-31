#!/usr/bin/env python3
"""Run one real M-PLS pool through ViT-Ti forward/backward/optimizer."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import torch

from galp.torch.experimental import DirectDctPlsPipeline

from .published_optimizer import build_published_optimizer
from .recipe import RECIPE_NAME, recipe_contract
from .train import build_paired_model


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--galp-torch-module-path", type=Path, required=True)
    parser.add_argument(
        "--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--epoch", type=int, default=7)
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.manifest, args.mapping, args.galp_torch_module_path):
        if not path.exists():
            raise FileNotFoundError(path)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native physical training smoke requires CUDA")
    torch.cuda.set_device(device)
    recipe = recipe_contract(RECIPE_NAME)
    model, initial_hash = build_paired_model(
        args.rgbnomore_root, seed=args.seed, device=device
    )
    execution_model = model
    if args.compile_model:
        from .train import compile_published_model

        execution_model = compile_published_model(model, recipe)
    total_updates = 375600
    optimizer, weight_decayer, scheduler = build_published_optimizer(
        model,
        learning_rate=float(recipe["optimizer"]["learning_rate"]),
        weight_decay=float(recipe["optimizer"]["weight_decay"]["coefficient"]),
        warmup_updates=int(recipe["scheduler"]["warmup_optimizer_updates"]),
        total_updates=total_updates,
    )
    pipeline = DirectDctPlsPipeline(
        args.manifest,
        args.mapping,
        training_seed=args.seed,
        expected_mapping_sha256=args.mapping_sha256,
        crop_policy="per-pls",
        order_policy="closed-pool",
        segments_per_pool=4,
        microbatch_images=64,
        segment_images=1024,
        model_classes=1000,
        module_path=args.galp_torch_module_path,
    ).start_epoch(args.epoch)
    started = time.perf_counter()
    pool = pipeline.next_pool()
    losses: list[float] = []
    updates = 0
    images = 0
    for window_begin in range(0, pool.microbatch_count, 16):
        window_microbatches = min(16, pool.microbatch_count - window_begin)
        window_images = min(
            window_microbatches * 64, pool.image_count - window_begin * 64
        )
        optimizer.zero_grad(set_to_none=True)
        learning_rate = scheduler.prepare_next_update()
        for _ in range(window_microbatches):
            batch = next(pool)
            y, cbcr, targets = batch.tensors
            logits = execution_model(y, cbcr)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("native physical training smoke loss is non-finite")
            batch_images = int(y.shape[0])
            (loss * (batch_images / window_images)).backward()
            losses.append(float(loss.detach().item()))
            images += batch_images
            del batch, y, cbcr, targets, logits, loss
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(recipe["optimizer"]["gradient_clipping_norm"]),
        )
        optimizer.step()
        weight_decayer.step(learning_rate)
        scheduler.complete_update()
        updates += 1
    torch.cuda.synchronize(device)
    stats = pool.execution_stats
    elapsed = time.perf_counter() - started
    if images != pool.image_count or updates != math.ceil(pool.microbatch_count / 16):
        raise RuntimeError("native physical training smoke did not consume one full pool")
    pool.retire()
    del pool
    pipeline.reclaim_finished_pools()
    pipeline.close()
    result = {
        "schema_version": "galp-native-physical-pls-training-smoke-v1",
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "epoch": args.epoch,
        "pool_images": images,
        "optimizer_updates": updates,
        "mean_loss": sum(losses) / len(losses),
        "elapsed_seconds": elapsed,
        "images_per_second": images / elapsed,
        "initial_model_hash": initial_hash,
        "model_compile": bool(args.compile_model),
        "native_execution_stats": stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
