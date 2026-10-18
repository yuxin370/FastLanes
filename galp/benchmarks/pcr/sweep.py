"""Run an interleaved PCR feasibility sweep using the existing inference monitor."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.inference_load import run
from experiments.run_vitti import gpu_uuid


def summarize(output):
    groups = {}
    for path in sorted(output.glob("*/trial_*/result.json")):
        result = json.loads(path.read_text())
        key = result["route"], result["scans"], result["read_mbps"]
        groups.setdefault(key, []).append(result)
    rows = []
    for (route, scans, rate), results in groups.items():
        first = results[0]
        row = dict(route=route, scans=scans, read_mbps=rate, trials=len(results), samples=first["samples"],
                   images_per_s=statistics.median(r["images_per_s"] for r in results),
                   min_images_per_s=min(r["images_per_s"] for r in results),
                   max_images_per_s=max(r["images_per_s"] for r in results), top1=first["top1"], top5=first["top5"],
                   issued_read_bytes=first["issued_read_bytes"], storage_bytes=first["storage_bytes"],
                   prediction_agreement=first.get("prediction_agreement", 1.),
                   top1_delta_pp=first.get("top1_delta_pp", 0.))
        rows.append(row)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=("4090", "H100"), default="4090")
    parser.add_argument("--model", choices=("mobilenet24", "resnet24"), default="mobilenet24")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--read-mbps", type=float, nargs="+", default=[0, 125, 500, 2000])
    parser.add_argument("--scans", type=int, nargs="+", default=[10, 5, 2])
    parser.add_argument("--reference", type=Path, help="Accepted full-validation RGB predictions")
    parser.add_argument("--cpus", help="Explicit CPU affinity; otherwise use NUMA node 0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(args.gpu), DCTNET_PROFILE=args.model,
               PYTHONPATH=str(ROOT), TMPDIR=str(Path.home() / "tmp"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    reference = args.reference or args.output / f"jpeg_files_s0_mbps{float(args.read_mbps[0])}/trial_0/result.json"
    configurations = [
        (route, scans, rate) for rate in args.read_mbps
        for route, scans in [("jpeg_files", 0), ("jpeg_record", 0), *[("pcr", n) for n in args.scans]]]
    for trial in range(args.trials):
        # Rotate complete configurations; measurements never overlap each other.
        ordered = configurations if trial == 0 else configurations[trial:] + configurations[:trial]
        for route, scans, rate in ordered:
            name = f"{route}_s{scans}_mbps{float(rate)}"
            output = args.output / name / f"trial_{trial}"
            output.mkdir(parents=True)
            binding = f"--physcpubind={args.cpus}" if args.cpus else "--cpunodebind=0"
            command = ["numactl", binding, "--membind=0", sys.executable, "-B", "-m",
                       "galp.benchmarks.pcr.run", "--data", str(args.data), "--model", args.model,
                       "--route", route, "--scans", str(scans), "--read-mbps", str(rate),
                       "--output", str(output / "result.json")]
            if reference.exists():
                command += ["--reference", str(reference)]
            run(command, output, env, None)  # The measured command warms/queries its exact input files.
            result = json.loads((output / "result.json").read_text())
            if route != "pcr" or scans == 10:
                if result.get("prediction_agreement", 1.) != 1. or result.get("top5_prediction_agreement", 1.) != 1.:
                    raise RuntimeError("Full-fidelity predictions differ from original JPEG")
            summarize(args.output)
            print(json.dumps(dict(trial=trial, configuration=name, images_per_s=result["images_per_s"],
                                  top1=result["top1"], issued_read_bytes=result["issued_read_bytes"])), flush=True)


if __name__ == "__main__":
    main()
