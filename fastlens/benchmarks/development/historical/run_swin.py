#!/usr/bin/env python3
"""Evaluate the registered SwinV2 256/window-8 inference configuration."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_vitti import execute
from fastlens.benchmarks.common import gpu_uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("4090", "H100"), required=True)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(args.gpu),
               TMPDIR=str(Path.home() / "tmp"), PYTHONPATH=f"{ROOT}:{ROOT}/build/fastlens/torch",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    base = ROOT / "results/raw" / args.gpu / "swinv2"
    common = [sys.executable, "-m", "fastlens.benchmarks.baselines.system_rgbnomore.inference.run",
              "--model", "rgbnomore-swinv2-t-256-window8-v1", "--pipelines", "galp", "rgbnomore", "pytorch", "dali",
              "--batch-size", "50", "--warmup-batches", "0", "--workers", "8", "--precision", "fp32"]
    execute([*common, "--preset", "smoke", "--measurement-batches", "2", "--repeats", "1",
             "--output-dir", str(base / "correctness")], base / "correctness.log", env)
    execute([*common, "--preset", "e2e", "--measurement-batches", "1000", "--repeats", "4",
             "--output-dir", str(base / "inference")], base / "inference.log", env)
    extra = []
    for name in ("galp", "rgbnomore", "pytorch", "dali"):
        result = json.loads((base / f"inference/pipeline_{name}.json").read_text())
        values = [r["images"] / r["repeat_scope_seconds"] for r in result["repeats"][1:]]
        if statistics.stdev(values) / statistics.mean(values) > .02:
            extra.append(name)
    if extra:
        # This runner's e2e preset requires all four pipelines. Use the same
        # explicit full-pass settings with its smoke preset for the extra arms.
        command = [sys.executable, "-m", "fastlens.benchmarks.baselines.system_rgbnomore.inference.run",
                   "--model", "rgbnomore-swinv2-t-256-window8-v1", "--preset", "smoke",
                   "--pipelines", *extra, "--batch-size", "50", "--warmup-batches", "0",
                   "--measurement-batches", "1000", "--workers", "8", "--precision", "fp32",
                   "--repeats", "3", "--output-dir", str(base / "inference_extra")]
        execute(command, base / "inference_extra.log", env)
    for domain in ("rgb", "dct"):
        execute([sys.executable, "fastlens/benchmarks/development/diagnostics/model_only.py", "--model", "swinv2", "--domain", domain,
                 "--batch-size", "50", "--output", str(base / f"model_only_{domain}.json")],
                base / f"model_only_{domain}.log", env)
    subprocess.run([sys.executable, "fastlens/benchmarks/development/reports/summarize.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
