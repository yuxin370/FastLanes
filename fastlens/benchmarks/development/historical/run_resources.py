#!/usr/bin/env python3
"""Separate profiled transfer diagnostics and unprofiled memory observations."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_vitti import PIPELINES
from fastlens.benchmarks.development.historical.run_vitti import execute
from fastlens.benchmarks.common import gpu_uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("4090", "H100"), required=True)
    parser.add_argument("--mode", choices=("profile", "memory"), required=True)
    parser.add_argument("--pipelines", choices=PIPELINES, nargs="+", default=PIPELINES)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(args.gpu),
               TMPDIR=str(Path.home() / "tmp"), PYTHONPATH=f"{ROOT}:{ROOT}/build/fastlens/torch",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    root = ROOT / "results" / ("profiles" if args.mode == "profile" else "memory") / args.gpu
    for pipeline in args.pipelines:
        output = root / pipeline
        output.mkdir(parents=True, exist_ok=True)
        python = str(Path.home() / "tmp/coordl-build-cu128/coordl-python") if pipeline == "coordl" else sys.executable
        command = [python, "fastlens/benchmarks/development/diagnostics/profile_pipeline.py", "--pipeline", pipeline,
                   "--source-contract", str(ROOT / "results/raw" / args.gpu / "vitti/inference/contract.json"),
                   "--output-dir", str(output), "--gpu-samples", str(root / (pipeline + "_monitor/gpu_processes.csv"))]
        if args.mode == "memory":
            command += ["--memory-only"]
        else:
            command = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                       "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--output", str(output / "capture"), *command]
        execute(command, root / (pipeline + ".log"), env)
        if args.mode == "profile":
            subprocess.run([sys.executable, "fastlens/benchmarks/execution/run_logged.py", "--log", str(output / "export.log"), "--",
                            "nsys", "export", "--type", "sqlite", "--output", str(output / "capture.sqlite"),
                            str(output / "capture.nsys-rep")], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
