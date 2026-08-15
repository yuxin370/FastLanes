#!/usr/bin/env python3
"""Model-facing Direct-DCT example using only the stable GALP PyTorch API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from galp.profiles.rgbnomore import VALIDATION, VALIDATION_CENTER_CROP_512
from galp.torch import DirectDctMetrics, DirectDctReader


PROFILES = {
    "validation": VALIDATION,
    "validation-center-crop-512": VALIDATION_CENTER_CROP_512,
}


class TinyDctClassifier(nn.Module):
    """Pool the model-ready Y/CbCr grids without observing storage internals."""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3 * 8 * 8, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        y_features = y.to(torch.float32).mean(dim=(2, 3)).flatten(1)
        cbcr_features = cbcr.to(torch.float32).mean(dim=(2, 3)).flatten(1)
        return self.layers(torch.cat((y_features, cbcr_features), dim=1))


def _batch_schedule(image_count: int, batch_size: int, count: int) -> list[list[int]]:
    if image_count <= 0:
        raise RuntimeError("manifest contains no images")
    actual_batch_size = min(batch_size, image_count)
    return [
        [int((step * actual_batch_size + offset) % image_count) for offset in range(actual_batch_size)]
        for step in range(count)
    ]


def _metrics_dict(metrics: DirectDctMetrics) -> dict[str, float | int | bool]:
    return {
        "complete": metrics.complete,
        "consumer_wait_ms": metrics.consumer_wait_ms,
        "submit_to_ready_ms": metrics.submit_to_ready_ms,
        "producer_ms": metrics.producer_ms,
        "planning_ms": metrics.planning_ms,
        "io_ms": metrics.io_ms,
        "decode_ms": metrics.decode_ms,
        "transform_ms": metrics.transform_ms,
        "logical_bytes": metrics.logical_bytes,
        "physical_bytes": metrics.physical_bytes,
        "peak_transient_bytes": metrics.peak_transient_bytes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny CUDA model through GALP's stable Direct-DCT pipeline API"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--module-path", type=Path, default=Path("build/galp/torch"))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--train-smoke", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.steps <= 0 or args.hidden_dim <= 0:
        raise ValueError("batch size, steps, and hidden dimension must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.num_classes <= 1:
        raise ValueError("num classes must be greater than one")
    if not torch.cuda.is_available():
        raise RuntimeError("this Direct-DCT example requires CUDA")

    profile = PROFILES[args.profile]
    reader = DirectDctReader(args.manifest, module_path=args.module_path)
    schedule = _batch_schedule(
        reader.image_count, args.batch_size, args.warmup + args.steps
    )
    pipeline = reader.pipeline(profile)
    model = TinyDctClassifier(args.hidden_dim, args.num_classes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if args.train_smoke else None
    model.train(args.train_smoke)

    def run_model_step(batch: Any) -> torch.Tensor:
        labels = torch.arange(len(batch.global_image_ids), device="cuda") % args.num_classes
        if optimizer is None:
            with torch.no_grad():
                logits = model(batch.y, batch.cbcr)
                return F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.y, batch.cbcr)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        return loss

    if args.warmup:
        pipeline.start(schedule[: args.warmup])
        for batch in pipeline:
            run_model_step(batch)
        torch.cuda.synchronize()

    # Resetting at the synchronized boundary gives the measurement interval its
    # own native aggregate. No per-batch metric read or hot-path sync is needed.
    pipeline.start(schedule[args.warmup :])
    measured_images = 0
    final_loss_tensor: torch.Tensor | None = None
    started = time.perf_counter()
    for batch in pipeline:
        loss = run_model_step(batch)
        measured_images += len(batch.global_image_ids)
        final_loss_tensor = loss.detach()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if final_loss_tensor is None:
        raise RuntimeError("the measured Direct-DCT schedule produced no model steps")
    final_loss = float(final_loss_tensor)
    metrics = pipeline.metrics
    if not metrics.complete:
        raise RuntimeError("native metrics remained incomplete after CUDA synchronization")
    pipeline.close()
    result: dict[str, Any] = {
        "api": "galp.torch",
        "profile_id": profile.id,
        "runtime_policy_id": reader.profile_info(profile)["runtime_policy_id"],
        "train_smoke": bool(args.train_smoke),
        "images": measured_images,
        "steps": args.steps,
        "seconds": elapsed,
        "images_per_s": measured_images / elapsed,
        "final_loss": final_loss,
        "metrics": _metrics_dict(metrics),
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
