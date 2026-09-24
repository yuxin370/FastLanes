"""Plan or run the seven block-major B6 models with CUDA Graph on RTX 4090."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
GPU = "GPU-40c637bd-acf5-ea1a-0df8-617138228467"
MODELS = ("mobilenet24", "mobilenet32", "resnet24", "resnet64", "efun", "vitti", "swinv2")
DATA = ROOT / "galp/data/compressed/imagenet512_train_block_major_premixed"
MANIFESTS = ROOT / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3"


def prepare_layout(output):
    # Keep the frozen layout/membership; only its relocated manifest metadata changes.
    from galp.benchmarks.training_pls.layout import load_layout_plan
    from galp.benchmarks.training_pls.recipe import sha256_json

    source = Path("/mnt/nvme2/home/tangyuxin/pls-experiments/pls-layout-20260811/physical_layout_plan.json")
    layout = load_layout_plan(source)
    raw = (MANIFESTS / "train.json").read_bytes()
    old_field = ('"galp_manifest": ' + json.dumps(layout["galp_manifest"])).encode()
    current_manifest = str(ROOT / "galp/data/compressed/imagenet512_train_compact_v3/dct/manifest.bin")
    new_field = ('"galp_manifest": ' + json.dumps(current_manifest)).encode()
    if raw.count(new_field) != 1 or hashlib.sha256(raw.replace(new_field, old_field, 1)).hexdigest() != layout["dataset_manifest_hash"]:
        raise ValueError("training manifest changed beyond the known galp_manifest relocation")
    layout["dataset_manifest_hash"] = hashlib.sha256(raw).hexdigest()
    layout["galp_manifest"] = current_manifest
    for key in ("sample_mapping_file", "runtime_mapping_file"):
        layout[key] = str(source.parent / layout[key])
    layout.pop("layout_hash")
    layout["layout_hash"] = sha256_json(layout)
    destination = output / "physical_layout_plan.json"
    if destination.exists() and json.loads(destination.read_text()) != layout:
        raise ValueError(f"existing layout differs: {destination}")
    destination.write_text(json.dumps(layout, indent=2) + "\n")
    return destination


def command_for(model, output, materialization, layout):
    if model in ("vitti", "swinv2"):
        model_id = "rgbnomore-vitti-dct-224-v1" if model == "vitti" else "rgbnomore-swinv2-t-dct-224-v1"
        return [sys.executable, "-u", "-m", "galp.benchmarks.training_pls.run_matrix",
                "--output-dir", str(output), "--train-manifest", str(MANIFESTS / "train.json"),
                "--val-manifest", str(MANIFESTS / "val.json"), "--layout-plan", str(layout),
                "--conditions", "B6", "--seeds", "11997733", "--model", model_id,
                "--epochs", "300", "--stop-after-epoch", "2", "--compile-mode", "reduce-overhead",
                "--execution-backend", "native-physical-pls",
                "--physical-galp-manifest", str(DATA / "dct/manifest.bin"),
                "--premixed-mapping-csv", materialization["ordered_mapping"],
                "--expected-mapping-sha256", materialization["ordered_mapping_sha256"],
                "--workers", "4", "--prefetch-depth", "2", "--required-gpu-name-substring", "RTX 4090"]
    driver = ["galp.benchmarks.dct_models.efun", "training_pls"] if model == "efun" else ["galp.benchmarks.dct_models.training_pls"]
    command = [sys.executable, "-u", "-m", *driver,
               "--manifest", str(DATA / "dct/manifest.bin"), "--mapping", materialization["ordered_mapping"],
               "--mapping-sha256", materialization["ordered_mapping_sha256"],
               "--condition", "B6", "--input-backend", "native", "--workers", "4",
               "--segments-per-pool", "4", "--torch-threads", "8", "--epochs", "2",
               "--validation-count", "50000", "--compile-mode", "reduce-overhead",
               "--transform-blocks-per-launch", "0" if model == "efun" else "32768",
               "--output-dir", str(output)]
    if model == "efun":
        command.append("--initial-validation")
    return command


def completed(model, output):
    if model in ("vitti", "swinv2"):
        run = output / "runs/B6/seed_11997733"
        path = run / "pause_result.json"
        if not path.exists():
            return False
        result = json.loads(path.read_text())
        return (result["completed_epoch"] == 2 and result["optimizer_update"] == 2504
                and result["processed_images"] == 2562334)
    path = output / "training.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    return (result["compile_mode"] == "reduce-overhead" and result["full_training_epochs_completed"] == 2
            and all(e["samples"] == e["unique_samples"] == 1281167 and e["validation"]["samples"] == 50000
                    for e in result["epochs"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results/block_major_graph_training_20260923")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    materialization = json.loads((DATA / "materialization_contract.json").read_text())
    layout = prepare_layout(output) if any(m in ("vitti", "swinv2") for m in args.models) else None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(Path.home() / "tmp/dctnet"),
               MPLCONFIGDIR=str(Path.home() / "tmp/matplotlib"))
    for model in args.models:
        destination = output / model
        if completed(model, destination):
            print("BLOCK-MAJOR GRAPH SKIP completed", model, flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        command = command_for(model, destination, materialization, layout)
        case_env = dict(env, DCTNET_PROFILE="resnet64" if model == "efun" else model)
        if not args.execute:
            if model in ("vitti", "swinv2"):
                with (destination / "plan.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, env=case_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                # Import the real migrated entry point and validate its CLI without touching CUDA.
                subprocess.run(command + ["--help"], cwd=ROOT, env=case_env, stdout=subprocess.DEVNULL, check=True)
            (destination / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            print("BLOCK-MAJOR GRAPH PLANNED", model, flush=True)
            continue
        occupied = subprocess.check_output(["nvidia-smi", "-i", GPU, "--query-compute-apps=pid,process_name",
                                             "--format=csv,noheader,nounits"], text=True)
        busy = [line for line in occupied.splitlines()
                if not line.split(",", 1)[1].strip().startswith(
                    "/home/zengletian/GPU-HASH-JOIN/GPU-Hash-Join/PHJ_GDS/PHJ_GDS_")]
        if busy:
            raise RuntimeError("GPU occupied: " + "\n".join(busy))
        if model in ("vitti", "swinv2"):
            command.append("--execute")
        (destination / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print("BLOCK-MAJOR GRAPH START", model, flush=True)
        with (destination / "run.log").open("a") as log, (destination / "gpu_4090_processes.csv").open("a") as monitor_log:
            monitor = subprocess.Popen(["nvidia-smi", "-i", GPU,
                                        "--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory,process_name",
                                        "--format=csv,noheader,nounits", "-lms", "100"], stdout=monitor_log)
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=case_env, stdout=log, stderr=subprocess.STDOUT)
                print("PID", process.pid, flush=True)
                code = process.wait()
            finally:
                monitor.terminate()
                monitor.wait()
        if code or not completed(model, destination):
            raise RuntimeError(f"{model} incomplete (exit {code}); see {destination / 'run.log'}")
        print("BLOCK-MAJOR GRAPH COMPLETE", model, flush=True)


if __name__ == "__main__":
    main()
