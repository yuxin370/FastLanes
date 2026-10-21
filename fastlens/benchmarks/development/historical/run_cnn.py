#!/usr/bin/env python3
"""Full online-source CNN inference, reusing the artifact's model/readers."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.dct_models.online_crop_suite import PROFILES
from fastlens.benchmarks.development.historical.run_vitti import execute
from fastlens.benchmarks.common import gpu_uuid


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("4090", "H100"), required=True)
    parser.add_argument("--profiles", choices=PROFILES, nargs="+", default=PROFILES)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source = ROOT / "fastlens/data/compressed/imagenet512_val_block_major"
    for profile in args.profiles:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(args.gpu),
                   DCTNET_PROFILE=profile, TMPDIR=str(Path.home() / "tmp"),
                   PYTHONPATH=f"{ROOT}:{ROOT}/build/fastlens/torch", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        base = ROOT / "results/raw" / args.gpu / profile

        def driver(task):
            return ([sys.executable, "-m", "fastlens.benchmarks.dct_models.efun", task] if profile == "efun"
                    else [sys.executable, "-m", f"fastlens.benchmarks.dct_models.{task}"])

        native = [*driver("evaluate_shards"), "--new-data", str(source), "--input-geometry", "source512",
                  "--pushdown", "off", "--crop-execution-mode", "vector-range-read-selected-decode",
                  "--output-layout", "projected", "--native-runtime", "b6", "--shards-per-activation", "1",
                  "--physical-prefix", "--batch-size", "64", "--baseline-dir", str(base / "unused_baseline")]
        reference = [*driver("evaluate"), "--route", "J", "--device", "cuda", "--physical-order",
                     "--batch-size", "64", "--workers", "8"]
        for name, command, extra in (("native", native, ["--verify"]), ("cpu_dct", reference, [])):
            out = base / "correctness" / name
            execute([*command, "--count", "8", *extra, "--output-dir", str(out)], out.with_suffix(".log"), env)
        actual = load(base / "correctness/native/N_8.json")
        expected = load(base / "correctness/cpu_dct/J_8.json")
        if actual["sample_ids"] != expected["sample_ids"] or actual["predictions"] != expected["predictions"]:
            raise RuntimeError(f"{profile}: small matched CNN prediction/identity gate failed")
        if not actual["checks"] or not all(row["native_grid_and_full_source_input_exact"] for row in actual["checks"]):
            raise RuntimeError(f"{profile}: native tensor equivalence gate missing")
        if args.verify_only:
            continue
        routes = [("imagelanes", native, "N"), ("cpu_dct", reference, "J")]
        # Each RGB checkpoint is evaluated once, rather than once per DCT channel budget.
        if profile in ("mobilenet24", "resnet24", "efun"):
            for route in ("pytorch", "dali"):
                routes.append((f"rgb_{route}", [*driver("evaluate_rgb"), "--route", route, "--physical-order",
                                               "--batch-size", "64", "--workers", "8"], f"RGB_{route}"))
        for name, command, prefix in routes:
            measurements = []
            count = 3
            trial = 0
            while trial <= count:
                out = base / "inference" / name / f"trial_{trial}"
                execute([*command, "--count", "50000", "--output-dir", str(out)], out.with_suffix(".log"), env)
                result = load(out / f"{prefix}_50000.json")
                if result["samples"] != 50000 or len(set(result["sample_ids"])) != 50000:
                    raise RuntimeError(f"incomplete sample coverage: {out}")
                if trial:
                    measurements.append(result)
                if trial == 3:
                    throughput = [50000 / r["e2e_seconds"] for r in measurements]
                    if statistics.stdev(throughput) / statistics.mean(throughput) > .02:
                        count = 5
                trial += 1
            first = measurements[0]
            if any(r["sample_ids"] != first["sample_ids"] or r["predictions"] != first["predictions"] for r in measurements):
                raise RuntimeError(f"prediction/order changed across trials: {base}/{name}")
        for domain in (("rgb", "dct") if profile in ("mobilenet24", "resnet24", "efun") else ("dct",)):
            execute([sys.executable, "fastlens/benchmarks/development/diagnostics/model_only.py", "--model", profile, "--domain", domain,
                     "--batch-size", "64", "--output", str(base / f"model_only_{domain}.json")],
                    base / f"model_only_{domain}.log", env)
        native_result = load(base / "inference/fastlens/trial_1/N_50000.json")
        cpu_result = load(base / "inference/cpu_dct/trial_1/J_50000.json")
        if native_result["sample_ids"] != cpu_result["sample_ids"]:
            raise RuntimeError(f"{profile}: full matched sample order gate failed")
        agreement = sum(a == b for a, b in zip(native_result["predictions"], cpu_result["predictions"])) / 50000
        (base / "inference/correctness.json").write_text(json.dumps(dict(
            sample_order_equal=True, prediction_agreement=agreement,
            numerical_control_pass=agreement >= .999,
            note="CPU/GPU DCT resize can differ at rounding boundaries; small tensor gate permits at most 0.505 before rounding",
            native_raw=str(base / "inference/fastlens/trial_1/N_50000.json"),
            cpu_raw=str(base / "inference/cpu_dct/trial_1/J_50000.json")), indent=2) + "\n")
        if agreement < .999:
            raise RuntimeError(f"{profile}: full matched prediction gate failed: {agreement}")


if __name__ == "__main__":
    main()
