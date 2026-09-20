"""Same-source online crop and output-frequency projection ablation, with idle-GPU enforcement."""
import argparse
import csv
import datetime
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
PROFILES = ("efun", "mobilenet24", "mobilenet32", "resnet24", "resnet64")
MODES = {"crop_off": "full-source-decode", "crop_on": "vector-range-read-selected-decode"}
CANDIDATES = (("legacy", 1), ("b6", 1), ("b6", 2), ("b6", 4))


def commands(source, count=50000):
    return [(f"{crop}/{layout}", ["--new-data", str(source), "--input-geometry", "source512",
             "--pushdown", "off", "--crop-execution-mode", mode, "--output-layout", layout,
             "--count", str(count), "--batch-size", "64"])
            for crop, mode in MODES.items() for layout in ("grid", "projected")]


def gpu_processes(gpu):
    output = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"], text=True)
    return {int(x.strip()) for x in output.splitlines() if x.strip().isdigit()}


def run(command, directory, env, gpu):
    active = gpu_processes(gpu)
    if active:
        raise RuntimeError(f"GPU is occupied before test: {active}; no test started")
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "run.log").open("w") as log, (directory / "gpu_processes.csv").open("w") as samples:
        monitor = subprocess.Popen(["nvidia-smi", "-i", gpu,
            "--query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits", "-lms", "100"], stdout=samples)
        process = subprocess.Popen([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        owned = {process.pid}
        seen = set()
        offset = 0
        try:
            while True:
                # Keep all observed descendants: nsys/Python children can exit
                # before the next sample is examined.
                if process.poll() is None:
                    try:
                        owned.update(p.pid for p in psutil.Process(process.pid).children(recursive=True))
                    except psutil.NoSuchProcess:
                        pass
                with (directory / "gpu_processes.csv").open() as captured:
                    captured.seek(offset)
                    for line in captured:
                        if not line.endswith("\n"):
                            break
                        offset += len(line)
                        row = next(csv.reader([line]))
                        if len(row) == 4 and row[2].strip().isdigit():
                            seen.add(int(row[2]))
                foreign = seen - owned
                if foreign:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    raise RuntimeError(f"External GPU processes {foreign}; this run is invalid, suite stopped")
                if process.poll() is not None:
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, command)
                    break
                time.sleep(0.1)
        finally:
            monitor.terminate()
            monitor.wait()
    print(datetime.datetime.now().isoformat(), "COMPLETE", directory, flush=True)


def calibrate(source, root, driver, env, gpu, profile):
    """Measure B6-derived native settings; freeze one setting for all ablations.

    All candidates use the same 8192 physical images and projected output.
    Memory feasibility includes the grid control and two resident activations.
    """
    free_mib = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free",
                                           "--format=csv,noheader,nounits"], text=True).strip())
    grid = 28 if profile == "efun" else 112 if profile.startswith("mobilenet") else 56
    channels = 192 if profile == "efun" else int(profile.removeprefix("mobilenet").removeprefix("resnet"))
    candidates, excluded = [], []
    for runtime, size in CANDIDATES:
        # Two full grids plus channel gathering/concatenation. Reserve 4 GiB
        # for reader worksets, model, CUDA context and forward workspace.
        required = 2 * 1024 * size * grid * grid * (192 + channels) * 4 + 4 * 1024**3
        item = dict(runtime=runtime, shards_per_activation=size, estimated_grid_peak_bytes=required)
        if required > free_mib * 1024**2:
            excluded.append(item)
        else:
            candidates.append(item)
    if not candidates:
        raise RuntimeError("insufficient GPU memory for the grid control")
    scores = {f'{c["runtime"]}_m{c["shards_per_activation"]}': [] for c in candidates}
    reference_predictions = None
    for repeat in range(3):
        order = candidates[repeat % len(candidates):] + candidates[:repeat % len(candidates)]
        for candidate in order:
            runtime, size = candidate["runtime"], candidate["shards_per_activation"]
            name = f"{runtime}_m{size}"
            out = root / "calibration" / name / f"repeat_{repeat}"
            command = ["--new-data", source, "--input-geometry", "source512", "--pushdown", "off",
                       "--crop-execution-mode", MODES["crop_on"], "--output-layout", "projected",
                       "--physical-prefix", "--count", "8192", "--batch-size", "64",
                       "--native-runtime", runtime, "--shards-per-activation", str(size)]
            run([*driver, *command, "--output-dir", out], out, env, gpu)
            result = json.loads((out / "N_8192.json").read_text())
            if result["samples"] != 8192 or result["decoded_images"] != 8192:
                raise AssertionError("calibration must consume the same complete 8192-image prefix")
            if reference_predictions is None:
                reference_predictions = result["predictions"]
            elif result["predictions"] != reference_predictions:
                raise AssertionError("native runtime configuration changed predictions")
            scores[name].append(result["e2e_seconds"])
    medians = {name: statistics.median(values) for name, values in scores.items()}
    winner = min(candidates, key=lambda c: medians[f'{c["runtime"]}_m{c["shards_per_activation"]}'])
    result = dict(selected=winner, median_e2e_seconds=medians, trials=scores, memory_excluded=excluded,
                  scope="fastest measured candidate on 8192 identical source images, three trials; not a global optimum",
                  ablation_policy="same selected runtime and activation size for crop off/on and grid/projected")
    (root / "calibration/selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SELECTED", profile, winner, medians, flush=True)
    return ["--native-runtime", winner["runtime"], "--shards-per-activation", str(winner["shards_per_activation"])]


def summarize(root, profiles, repeats):
    rows = []
    for profile in profiles:
        selection = json.loads((root / profile / "calibration/selection.json").read_text())
        predictions = None
        for name, _ in commands(Path("unused")):
            values = []
            for repeat in range(repeats):
                result = json.loads((root / profile / f"repeat_{repeat}" / name / "N_50000.json").read_text())
                assert result["samples"] == len(result["predictions"]) == 50000
                if predictions is None:
                    predictions = result["predictions"]
                elif result["predictions"] != predictions:
                    raise AssertionError(f"{profile}: crop/projection/repeat predictions differ")
                record = {k: result[k] for k in ("e2e_seconds", "model_seconds", "top1", "top5")}
                for key in ("source_rowgroups", "rowgroup_count", "actual_vector_count", "full_vector_count",
                            "coefficient_logical_bytes_requested", "coefficient_range_bytes_read",
                            "source_blocks_transformed"):
                    record[key] = sum(s[key] for s in result["native_shards"])
                values.append(record)
            rows.append(dict(profile=profile, route=name,
                runtime_selection=selection["selected"],
                median={k: statistics.median(v[k] for v in values) for k in values[0]}, repeats=values))
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Full-source online crop and output projection", "",
        "All 50,000 ImageNet-512 validation images; FP32, batch 64, TF32 off. Median unprofiled times.", "",
        "| Model | Route | E2E s | Model scope s | Top-1 % | Top-5 % | Source range GB |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["median"]
        lines.append(f'| {row["profile"]} | {row["route"]} | {m["e2e_seconds"]:.3f} | '
                     f'{m["model_seconds"]:.3f} | {m["top1"]:.3f} | {m["top5"]:.3f} | '
                     f'{m["coefficient_range_bytes_read"]/1e9:.3f} |')
    lines += ["", "- crop_off reads/decodes all source rowgroups; crop_on uses online crop support selection.",
        "- Both perform crop/resize online; offline data contains the complete original JPEG coefficients.",
        "- grid materializes all resized coefficients then selects model channels; projected fuses output selection.",
        "- Projection is output-frequency pushdown, not source-frequency pruning. Resize retains all 64 source frequencies.",
        "- All routes/repeats have identical predictions within each model; model input shape and MACs are unchanged.",
        "- DCT-domain resize differs from the author's pixel-domain resize/JPEG recipe; accuracy is remeasured here.",
        "- Existing source images are 512px JPEGs. This is not original-resolution ImageNet or cold-disk throughput.",
        "- GPU process sampling must contain only this test's processes; an external process stops the suite."]
    lines += ["- Runtime/activation size is selected by three 8192-image trials, then fixed across all four ablations.",
              "- Candidates include the old reader and B6 bounded io_uring/async/limited-overlap settings with M=1/2/4; oversized grid controls are excluded.",
              "- Selected settings are the fastest among the measured candidates, not a claim of a global optimum."]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-profile", action="store_true")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    if gpu_processes(gpu):
        raise RuntimeError("Selected GPU is occupied")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = args.source_data.resolve()
    for profile in args.profiles:
        env = dict(os.environ, DCTNET_PROFILE=profile, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        driver = [sys.executable, HERE / "efun.py", "evaluate_shards"] if profile == "efun" else [sys.executable, HERE / "evaluate_shards.py"]
        runtime_args = calibrate(source, root / profile, driver, env, gpu, profile)
        # Check all four modes against full-source GPU and independent CPU geometry.
        for name, command in commands(source, count=8):
            out = root / profile / "verify" / name
            run([*driver, *command, *runtime_args, "--verify", "--output-dir", out], out, env, gpu)
        rows = commands(source)
        for repeat in range(args.repeats):
            order = rows[repeat % len(rows):] + rows[:repeat % len(rows)]
            for name, command in order:
                out = root / profile / f"repeat_{repeat}" / name
                run([*driver, *command, *runtime_args, "--output-dir", out], out, env, gpu)
        if not args.skip_profile:
            for name, command in rows:
                out = root / profile / "profile" / name
                stem = out / ("galp_native" if name.endswith("grid") else "galp_projected")
                run(["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                     "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--output", stem,
                     *driver, *command, *runtime_args, "--profile", "--output-dir", out / "run"], out, env, gpu)
                subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(stem)+".sqlite",
                                str(stem)+".nsys-rep"], check=True)
                subprocess.run([sys.executable, HERE / "analyze_profiles.py", out], check=True, env=env)
    summarize(root, args.profiles, args.repeats)
    print("Complete:", root / "report.md", flush=True)


if __name__ == "__main__":
    main()
