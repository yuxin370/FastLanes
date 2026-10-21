#!/usr/bin/env python3
"""Complete the registered six-pipeline ViT inference matrix, serially by GPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
PIPELINES = ["dct_major_pushdown", "rgbnomore", "pytorch", "dali", "ffcv", "coordl"]


from fastlens.benchmarks.common import gpu_uuid


def execute(command, log, env):
    metadata = log.with_suffix(".command.json")
    if metadata.exists():
        prior = json.loads(metadata.read_text())
        if prior.get("returncode") != 0:
            raise RuntimeError(f"prior run incomplete or failed: {metadata}; inspect before continuing")
        return
    from fastlens.benchmarks.dct_models.online_crop_suite import run

    run([sys.executable, "fastlens/benchmarks/execution/run_logged.py", "--log", str(log), "--", *command],
        log.parent / (log.stem + "_monitor"), env, env["CUDA_VISIBLE_DEVICES"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("4090", "H100"), nargs="+", default=["4090", "H100"])
    args = parser.parse_args()
    for gpu in args.gpu:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), TMPDIR=str(Path.home() / "tmp"),
                   PYTHONPATH=f"{ROOT}:{ROOT}/build/fastlens/torch", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        base = ROOT / "results/raw" / gpu / "vitti"
        common = [sys.executable, "-m", "fastlens.benchmarks.system_dct_major.run", "--preset", "e2e",
                  "--workload", "evaluation", "--block-major-access-dir",
                  "fastlens/data/compressed/imagenet512_val_block_major/access", "--dct-coeffs", "all",
                  "--sample-count", "50000", "--batch-size", "50", "--measurement-batches", "1000",
                  "--warmup-batches", "0", "--workers", "8", "--coordl-cache-size", "25000",
                  "--coordl-python", str(Path.home() / "tmp/coordl-build-cu128/coordl-python"),
                  "--ffcv-beton", str(Path.home() / "tmp/dct-major-coordl-full-20260924/ffcv.beton")]
        execute([*common, "--output-dir", str(base / "inference"), "--repeats", "4", "--pipelines", *PIPELINES],
                base / "inference.log", env)
        initial = json.loads((base / "inference/results.json").read_text())
        if not initial["ok"]:
            raise RuntimeError(f"full correctness gate failed: {base}")
        additional = []
        for name in PIPELINES:
            result = json.loads((base / f"inference/pipeline_{name}.json").read_text())
            values = [r["images"] / r["repeat_scope_seconds"] for r in result["repeats"][1:]]
            if statistics.stdev(values) / statistics.mean(values) > .02:
                additional.append(name)
        if additional:
            # Two additional measured full passes after a fresh warmup pass.
            execute([*common, "--output-dir", str(base / "inference_extra"), "--repeats", "3",
                     "--pipelines", *additional], base / "inference_extra.log", env)
        for domain in ("rgb", "dct"):
            execute([sys.executable, "fastlens/benchmarks/development/diagnostics/model_only.py", "--model", "vitti", "--domain", domain,
                     "--batch-size", "50", "--output", str(base / f"model_only_{domain}.json")],
                    base / f"model_only_{domain}.log", env)
        subprocess.run([sys.executable, "fastlens/benchmarks/development/reports/summarize.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
