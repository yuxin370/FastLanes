#!/usr/bin/env python3
"""Measure the model-only ceiling for the exact feature/evaluation graph."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from feature_model import build_workload_model, expected_output_width  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("dct", "rgb"), required=True)
    parser.add_argument("--workload", choices=("feature-extraction", "evaluation"), required=True)
    parser.add_argument("--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = build_workload_model(
        domain=args.domain,
        workload=args.workload,
        rgbnomore_root=args.rgbnomore_root,
        checkpoint=args.checkpoint,
        device=device,
    )
    if args.domain == "rgb":
        inputs = (torch.randn(args.batch_size, 3, 224, 224, device=device),)
    else:
        inputs = (
            torch.randn(args.batch_size, 1, 28, 28, 8, 8, device=device),
            torch.randn(args.batch_size, 2, 14, 14, 8, 8, device=device),
        )
    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.steps):
            output = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
    if output.shape != (args.batch_size, expected_output_width(args.workload)):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "domain": args.domain,
                "workload": args.workload,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "seconds": seconds,
                "throughput_images_per_s": args.batch_size * args.steps / seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
