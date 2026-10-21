#!/usr/bin/env python3
"""Run existing full training recipes; measure epoch 2 of independent trials."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_vitti import execute
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.dct_models.online_crop_suite import PROFILES
from fastlens.benchmarks.training_pls.model_registry import MODEL_IDS


def load(path):
    return json.loads(path.read_text())


def warm_epoch(output, model, pipeline):
    if model in PROFILES:
        result = load(output / "training.json")
        first, warm = result["epochs"]
        if not warm["full_epoch"] or warm["unique_samples"] != warm["samples"]:
            raise ValueError(f"incomplete CNN training coverage: {output}")
        return dict(samples=warm["samples"], elapsed_s=warm["seconds"], loss=warm["ce"],
                    precision=result["precision"], microbatch=result["microbatch"],
                    effective_batch_size=result["microbatch"] * result["accumulation"],
                    initial_model_hash=result["initial_model_hash"],
                    updates=warm["optimizer_updates"] - first["optimizer_updates"],
                    raw_result=str(output / "training.json") + "#/epochs/1")
    if pipeline in ("imagelanes", "cpu_dct"):
        path = output / "runs/B6/seed_11997733/metrics.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        warm, = [r for r in records if r.get("scope") == "epoch" and r["epoch"] == 2]
        loss = warm["train_loss"]
        final = load(path.parent / "pause_result.json")
        initial_model_hash = load(path.parent / "model_only_dct.json")["initial_model_hash"]
        recipe = load(output / "recipe_contract.json")
        microbatch = recipe["training"]["physical_microbatch"]
        accumulation = recipe["training"]["gradient_accumulation"]
    else:
        name = "pytorch" if pipeline == "pytorch_rgb" else "d2"
        path = output / f"runs/{name}/final_result.json"
        result = load(path)
        warm = result["primary_warm_epoch"]
        loss = warm["train_loss"]
        final = result
        initial_model_hash = final["initial_model_hash"]
        microbatch, accumulation = warm["microbatch_images"], warm["gradient_accumulation"]
    return dict(samples=warm["epoch_samples"], elapsed_s=warm["epoch_seconds"], loss=loss,
                precision=final["precision"], microbatch=microbatch,
                effective_batch_size=microbatch * accumulation,
                initial_model_hash=initial_model_hash,
                updates=warm["epoch_optimizer_updates"], raw_result=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("4090", "H100"), required=True)
    parser.add_argument("--models", nargs="+", choices=(*MODEL_IDS, *PROFILES), required=True)
    parser.add_argument("--pipelines", nargs="+", choices=("imagelanes", "cpu_dct", "pytorch_rgb", "dali"),
                        default=["imagelanes", "cpu_dct", "pytorch_rgb", "dali"])
    args = parser.parse_args()
    manifests = ROOT / "fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3"
    physical = ROOT / "fastlens/data/compressed/imagenet512_train_block_major_premixed"
    materialization = load(physical / "materialization_contract.json")
    layout = ROOT / "results/preparation/training_layout"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(args.gpu),
               TMPDIR=str(Path.home() / "tmp"), PYTHONPATH=f"{ROOT}:{ROOT}/build/fastlens/torch",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if any(model in MODEL_IDS for model in args.models):
        execute([sys.executable, "-m", "fastlens.benchmarks.training_pls.plan_layout", "--train-manifest",
                 str(manifests / "train.json"), "--output-dir", str(layout), "--segment-images", "1024",
                 "--organization-seed", "20260810"], layout.with_suffix(".log"), env)
    for model in args.models:
        model_env = dict(env, DCTNET_PROFILE=model) if model in PROFILES else env
        for pipeline in args.pipelines:
            # The RGB MobileNet/ResNet are shared across their DCT channel variants.
            if model in ("mobilenet32", "resnet64") and pipeline in ("pytorch_rgb", "dali"):
                continue
            base = ROOT / "results/raw" / args.gpu / model / "training" / pipeline
            trials = []
            target = 3
            trial = 1
            while trial <= target:
                output = base / f"trial_{trial}"
                if model in PROFILES:
                    driver = ([sys.executable, "-m", "fastlens.benchmarks.dct_models.efun", "training_pls"] if model == "efun"
                              else [sys.executable, "-m", "fastlens.benchmarks.dct_models.training_pls"])
                    backend, condition = {"imagelanes": ("native", "B6"), "cpu_dct": ("jpeg", "A0"),
                                          "pytorch_rgb": ("rgb_pytorch", "RGB"), "dali": ("rgb_d2", "RGB")}[pipeline]
                    command = [*driver, "--manifest", str(physical / "dct/manifest.bin"),
                               "--mapping", materialization["ordered_mapping"],
                               "--mapping-sha256", materialization["ordered_mapping_sha256"],
                               "--condition", condition, "--input-backend", backend, "--workers", "8",
                               "--segments-per-pool", "4", "--epochs", "2", "--validation-count", "50000",
                               "--transform-blocks-per-launch", "0" if model == "efun" else "32768",
                               "--output-dir", str(output)]
                elif pipeline in ("imagelanes", "cpu_dct"):
                    command = [sys.executable, "-m", "fastlens.benchmarks.training_pls.run_matrix",
                               "--train-manifest", str(manifests / "train.json"), "--val-manifest", str(manifests / "val.json"),
                               "--layout-plan", str(layout / "physical_layout_plan.json"),
                               "--conditions", "B6", "--seeds", "11997733", "--model", model,
                               "--epochs", "300", "--stop-after-epoch", "2", "--execution-backend",
                               "native-physical-pls" if pipeline == "imagelanes" else "standard-rgbnomore-dct",
                               "--premixed-mapping-csv", materialization["ordered_mapping"],
                               "--expected-mapping-sha256", materialization["ordered_mapping_sha256"],
                               "--workers", "8", "--output-dir", str(output), "--device", "cuda:0",
                               "--required-gpu-name-substring", args.gpu, "--execute"]
                    if pipeline == "imagelanes":
                        command += ["--physical-fastlens-manifest", str(physical / "dct/manifest.bin")]
                else:
                    command = [sys.executable, "-m", "fastlens.benchmarks.baselines.system_rgbnomore.training.equal_image_epoch_benchmark",
                               "--train-manifest", str(manifests / "train.json"), "--val-manifest", str(manifests / "val.json"),
                               "--model", model, "--pipelines", "pytorch" if pipeline == "pytorch_rgb" else "d2",
                               "--epochs", "2", "--seed", "11997733", "--workers", "8", "--dali-num-threads", "8",
                               "--output-dir", str(output), "--device", "cuda:0", "--required-gpu-name-substring", args.gpu,
                               "--execute"]
                execute(command, output.with_suffix(".log"), model_env)
                warm = warm_epoch(output, model, pipeline)
                if warm["samples"] != materialization["images"] or warm["updates"] != math.ceil(warm["samples"] / 1024) or not math.isfinite(warm["loss"]):
                    raise ValueError(f"warm epoch coverage/update/finite-loss gate failed: {output}")
                trials.append(warm)
                if warm["initial_model_hash"] != trials[0]["initial_model_hash"]:
                    raise ValueError(f"initialization changed across trials: {output}")
                if pipeline == "cpu_dct":
                    native = base.parent / "fastlens/warm_epochs.json"
                    if native.exists() and warm["initial_model_hash"] != load(native)["trials"][0]["initial_model_hash"]:
                        raise ValueError(f"native/CPU-DCT initialization differs: {output}")
                if trial == 3:
                    values = [r["samples"] / r["elapsed_s"] for r in trials]
                    if statistics.stdev(values) / statistics.mean(values) > .02:
                        target = 5
                trial += 1
            # Raw epochs remain authoritative; this file links the measured scope.
            (base / "warm_epochs.json").write_text(json.dumps(dict(model=model, pipeline=pipeline, trials=trials,
                comparison="CNN B6 versus A0 and all RGB/DCT comparisons are deployment comparisons; no convergence claim"), indent=2) + "\n")


if __name__ == "__main__":
    main()
