#!/usr/bin/env python3
"""Resident-tensor inference ceiling with the artifact's exact model factories."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", choices=("rgb", "dct"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(1 if args.model in ("vitti", "swinv2") else 8)
    torch.manual_seed(11997733)
    if args.model not in ("vitti", "swinv2"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    checkpoints = ROOT / "fastlens/data/system_rgbnomore/e2e_v2/checkpoints"
    if args.model in ("vitti", "swinv2"):
        from fastlens.benchmarks.baselines.system_rgbnomore.inference import model_factory as factory
        identifier = factory.VITTI_MODEL_ID if args.model == "vitti" else factory.SWINV2_T_MODEL_ID
        spec = factory.resolve_model(identifier)
        checkpoint = checkpoints / getattr(spec, args.domain + "_checkpoint_name")
        model = getattr(factory, "build_" + args.domain + "_model")(
            Path.home() / "RGB-no-more", checkpoint, device, model_id=identifier)
        shapes = [(3, spec.rgb_size, spec.rgb_size)] if args.domain == "rgb" else [spec.y_shape, spec.cbcr_shape]
    else:
        if args.model == "efun":
            import fastlens.benchmarks.dct_models as package
            from fastlens.benchmarks.dct_models import efun_backend as backend
            sys.modules["fastlens.benchmarks.dct_models.backend"] = backend
            package.backend = backend
        else:
            os.environ["DCTNET_PROFILE"] = args.model
            from fastlens.benchmarks.dct_models import backend
        if args.domain == "rgb":
            from fastlens.benchmarks.dct_models import rgb
            model, checkpoint, shapes = rgb.model().to(device), rgb.CHECKPOINT, [(3, 224, 224)]
        else:
            model, checkpoint = backend.model().to(device), backend.CHECKPOINT
            shapes = [backend.profile()["shape"]]
    inputs = tuple(torch.randn(args.batch_size, *shape, device=device) for shape in shapes)
    model.eval()
    trials = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model(*inputs)
        torch.cuda.synchronize()
        if output.shape != (args.batch_size, 1000) or not torch.isfinite(output).all():
            raise ValueError("resident model correctness gate failed")
        while len(trials) < 3 or (len(trials) < 5 and statistics.stdev([x["images_per_s"] for x in trials]) / statistics.mean([x["images_per_s"] for x in trials]) > .02):
            # Once the first three require more trials, always collect both extras.
            target = 3 if len(trials) < 3 else 5
            for _ in range(len(trials), target):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for _ in range(args.steps):
                    output = model(*inputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                trials.append(dict(trial=len(trials) + 1, samples=args.steps * args.batch_size,
                                   elapsed_s=elapsed, images_per_s=args.steps * args.batch_size / elapsed,
                                   torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   torch_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    result = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  gpu=torch.cuda.get_device_name(), model=args.model, domain=args.domain,
                  checkpoint=str(checkpoint), tensor_shapes=[list(x.shape) for x in inputs],
                  precision="fp32", matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_tf32=torch.backends.cudnn.allow_tf32, batch_size=args.batch_size,
                  warmup=args.warmup, steps=args.steps, trials=trials,
                  scope="GPU-resident model forward only; host wall time with boundary synchronization; no profiler",
                  correctness="finite [batch,1000] output; strict checkpoint loading")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
