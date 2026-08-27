#!/usr/bin/env python3
"""GPU-resident model-compute benchmarks; input and preprocessing are untimed."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from galp.experiments.dct_retokenization.compute import mac_table
from galp.experiments.dct_retokenization.data import MaskedDctPreprocessor, load_manifest, make_loader
from galp.experiments.dct_retokenization.model_wrapper import (
    DctRetokenizationWrapper,
    build_pretrained_dct_model,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = Path(os.environ.get("RGBNOMORE_ROOT", "RGB-no-more")).expanduser()
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hardware_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "captured_at_unix_ns": time.time_ns(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": result.stdout.strip(),
        "nvidia_smi_stderr": result.stderr.strip(),
        "loadavg": os.getloadavg(),
    }


def benchmark(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(device)
        timings: list[float] = []
        for _ in range(repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            function()
            end.record()
            end.synchronize()
            timings.append(float(begin.elapsed_time(end)))
    median = statistics.median(timings)
    mean = statistics.fmean(timings)
    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "median_latency_ms": median,
        "mean_latency_ms": mean,
        "stddev_latency_ms": statistics.pstdev(timings),
        "median_images_per_second": batch_size * 1000.0 / median,
        "mean_images_per_second": batch_size * 1000.0 / mean,
        "minimum_latency_ms": min(timings),
        "maximum_latency_ms": max(timings),
    }


def resident_dct_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    _, samples = load_manifest(args.manifest.resolve(), max_samples=max(args.batch_size, 64))
    loader = make_loader(
        samples,
        args.rgbnomore_root.resolve(),
        indices=None,
        batch_size=min(args.batch_size, len(samples)),
        workers=args.workers,
    )
    yq, cq, quant, _labels, _ordinals = next(iter(loader))
    preprocessor = MaskedDctPreprocessor(args.rgbnomore_root.resolve(), device, args.k)
    y, cbcr = preprocessor.validation(yq, cq, quant)
    if y.shape[0] < args.batch_size:
        repeats = (args.batch_size + y.shape[0] - 1) // y.shape[0]
        y = y.repeat((repeats, 1, 1, 1, 1, 1))[: args.batch_size]
        cbcr = cbcr.repeat((repeats, 1, 1, 1, 1, 1))[: args.batch_size]
    return y.contiguous(), cbcr.contiguous()


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    torch.cuda.set_device(device)
    observed_name = torch.cuda.get_device_name(device)
    if args.expected_device_name and observed_name != args.expected_device_name:
        raise RuntimeError(f"expected {args.expected_device_name!r}, observed {observed_name!r}")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    write_json(output_dir / "hardware_before.json", hardware_snapshot())
    torch.cuda.reset_peak_memory_stats(device)

    base = build_pretrained_dct_model(
        args.rgbnomore_root.resolve(), args.checkpoint.resolve(), device
    ).eval()
    wrappers = {
        196: DctRetokenizationWrapper(base, token_count=196, merge_type="fixed").eval(),
        98: DctRetokenizationWrapper(
            base, token_count=98, merge_axis="width", merge_type="learned"
        ).eval(),
        49: DctRetokenizationWrapper(
            base, token_count=49, merge_axis="height", merge_type="learned"
        ).eval(),
    }
    y, cbcr = resident_dct_batch(args, device)
    rows: list[dict[str, Any]] = []
    full_baseline_latency: float | None = None
    for tokens in (196, 98, 49):
        result = benchmark(
            lambda wrapper=wrappers[tokens]: wrapper(y, cbcr),
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        if full_baseline_latency is None:
            full_baseline_latency = float(result["median_latency_ms"])
        rows.append(
            {
                "benchmark": "prototype_full_model_forward",
                "tokens": tokens,
                **result,
                "latency_speedup_vs_n196": full_baseline_latency
                / float(result["median_latency_ms"]),
                "input_residency": "gpu_resident_preprocessed_dct",
                "includes_patch_projection": True,
                "includes_retokenization": tokens != 196,
                "includes_io_or_preprocessing": False,
            }
        )

    transformer_baseline_latency: float | None = None
    for tokens in (196, 98, 49):
        token_input = torch.randn(
            (args.batch_size, tokens, 192), dtype=torch.float32, device=device
        )
        result = benchmark(
            lambda value=token_input: base.encoder(value),
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        if transformer_baseline_latency is None:
            transformer_baseline_latency = float(result["median_latency_ms"])
        rows.append(
            {
                "benchmark": "transformer_encoder_only",
                "tokens": tokens,
                **result,
                "latency_speedup_vs_n196": transformer_baseline_latency
                / float(result["median_latency_ms"]),
                "input_residency": "gpu_resident_tokens",
                "includes_patch_projection": False,
                "includes_retokenization": False,
                "includes_io_or_preprocessing": False,
            }
        )

    with (output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    macs = mac_table()
    with (output_dir / "mac_accounting.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macs[0]))
        writer.writeheader(); writer.writerows(macs)
    write_json(
        output_dir / "run_metadata.json",
        {
            "status": "complete",
            "device_name": observed_name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "precision": "fp32",
            "tf32": False,
            "resident_dct_coefficient_count": args.k,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timing": "CUDA events with per-iteration end-event synchronization",
            "prototype_caveat": "all variants first materialize the existing 196 patch embeddings",
            "flop_convention": "1 MAC = 2 FLOPs",
            "rows": rows,
            "mac_accounting": macs,
            "completed_at_unix_ns": time.time_ns(),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    )
    write_json(output_dir / "hardware_after.json", hardware_snapshot())
    print(json.dumps(rows, indent=2), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--k", type=int, choices=(64, 32), default=32)
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.warmup, args.repeats) <= 0 or args.workers < 0:
        parser.error("invalid benchmark counts")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
