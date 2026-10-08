"""Run L3 and JPEG/DALI with the paper's existing inference and training commands."""
import argparse
import copy
import csv
import json
import os
import statistics
import sys
from pathlib import Path

from experiments.inference_load import run
from galp.benchmarks.system_dct_major.common import file_identity, sha256_file, sha256_json
from .prepare import ROOT, encoded_path, source_samples

MODELS = {"vitti": "ViT-Ti", "swinv2": "SwinV2-T", "mobilenet24": "MobileNetV2",
          "resnet24": "ResNet-50", "efun": "eFUN"}
GPUS = {"4090": "GPU-40c637bd-acf5-ea1a-0df8-617138228467",
        "H100": "GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76"}


def read(path):
    return json.loads(Path(path).read_text())


def set_arg(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def environment(metadata, gpu, model):
    env = dict(os.environ)
    env.update({k: v for k, v in metadata["environment"].items() if v is not None})
    for key in ("IMAGELANES_EVALUATION_SNAPSHOT", "IMAGELANES_ALLOWED_RESIDENT_PIDS"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model, TMPDIR=str(Path.home() / "tmp"),
               PYTHONPATH=str(ROOT) + ":" + str(ROOT / "build/galp/torch"), PYTHONDONTWRITEBYTECODE="1")
    return env


def inference_contract(command, destination, pipeline, args):
    contract = read(command[command.index("--contract") + 1])
    manifest = read(contract["dataset"]["sample_manifest"])
    for sample in manifest["samples"]:
        if pipeline == "l3":
            sample["path"] = str(encoded_path(args.data, sample["sample_id"]))
            sample["size_bytes"] = Path(sample["path"]).stat().st_size
            if "sha256" in sample:
                sample["sha256"] = sha256_file(Path(sample["path"]))
        current = file_identity(Path(sample["path"]))
        if pipeline == "dali" and "file_identity" in sample:
            previous = sample["file_identity"]
            if any(current[k] != previous[k] for k in ("size_bytes", "mtime_ns")):
                raise ValueError("Source JPEG changed since the paper run")
        if "file_identity" in sample:
            sample["file_identity"] = current
    sample_path = destination / "samples.json"
    sample_path.write_text(json.dumps(manifest))
    contract["dataset"]["sample_manifest"] = str(sample_path)
    key = "sample_manifest_sha256" if "sample_manifest_sha256" in contract["dataset"] else "manifest_sha256"
    contract["dataset"][key] = sha256_json(manifest)
    if pipeline == "l3":
        contract["pipelines"]["dali"]["l3"] = dict(root=str(args.data), library=str(args.library))
    path = destination / "contract.json"
    path.write_text(json.dumps(contract))
    set_arg(command, "--contract", path)
    return [Path(s["path"]) for s in manifest["samples"]]


def summarize(output):
    observations = []
    for record in sorted(output.glob("*/*/*/*/trial_*/observation.json")):
        observations.append(read(record))
    groups = {}
    for row in observations:
        groups.setdefault((row["phase"], row["gpu"], row["model"], row["pipeline"]), []).append(row)
    rows = []
    for (phase, gpu, model, pipeline), trials in groups.items():
        rates = [r["images_per_s"] for r in trials]
        rows.append(dict(phase=phase, gpu=gpu, model=model, pipeline=pipeline, trials=len(trials),
                         images=trials[0]["images"], median_images_per_s=statistics.median(rates),
                         min_images_per_s=min(rates), max_images_per_s=max(rates),
                         top1=trials[0].get("top1"),
                         raw_results=";".join(r["raw_result"] for r in trials)))
    if rows:
        with (output / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=GPUS, required=True)
    parser.add_argument("--phase", choices=["inference", "training"], required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    args.library, args.data, args.output = args.library.resolve(), args.data.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    split = "val" if args.phase == "inference" else "train"
    conversion = read(args.data / (split + "_conversion.json"))
    assert conversion["images"] == (50000 if split == "val" else 98560)
    assert conversion["patch_size"] == 32, "Reconvert the initial 64x64-patch dataset"
    source = "deployment_inference_summary.csv" if args.phase == "inference" else "training_window_summary.csv"
    with (ROOT / "results/p0" / source).open() as stream:
        rows = list(csv.DictReader(stream))
    gpu_label = "RTX 4090" if args.gpu == "4090" else "H100"
    training_samples = source_samples("train") if split == "train" else None
    for model in args.models:
        family = MODELS[model]
        if args.phase == "training" and model == "efun":
            family = "EfficientNet-B0 reference"
        field = "architecture_family" if args.phase == "inference" else "model"
        template = next(r for r in rows if r["gpu"] == gpu_label and r[field] == family
                        and r["pipeline"] == "DALI" and r["domain"] == "RGB")
        original = ROOT / template["raw_results"].split(";")[0].split("#")[0]
        metadata = read(original.parent / "run.command.json")
        env = environment(metadata, args.gpu, model)
        if args.phase == "training":
            check = args.output / "checks" / args.gpu / model / "l3"
            if not (check / "result.json").exists():
                check.mkdir(parents=True, exist_ok=True)
                command = copy.copy(metadata["command"])
                pos = command.index("--correctness-report")
                del command[pos:pos + 2]
                set_arg(command, "--pipeline", "l3")
                set_arg(command, "--output", check)
                command += ["--check", "--l3-root", str(args.data), "--l3-library", str(args.library)]
                run(command, check, env, None)
        for trial in range(3):
            for pipeline in (["dali", "l3"] if trial % 2 == 0 else ["l3", "dali"]):
                destination = args.output / args.phase / args.gpu / model / pipeline / f"trial_{trial}"
                if (destination / "observation.json").exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                command = copy.copy(metadata["command"])
                if args.phase == "inference":
                    if "--contract" in command:
                        paths = inference_contract(command, destination, pipeline, args)
                        result_path = destination / "result.json"
                        set_arg(command, "--output", result_path)
                    else:
                        entries = sorted(source_samples("val"), key=lambda s: s["galp_image_id"])
                        paths = [encoded_path(args.data, e["logical_sample_id"]) if pipeline == "l3"
                                 else Path(e["path"]) for e in entries]
                        set_arg(command, "--route", pipeline)
                        set_arg(command, "--output-dir", destination)
                        if pipeline == "l3":
                            command += ["--l3-root", str(args.data), "--l3-library", str(args.library)]
                        result_path = destination / f"RGB_{pipeline}_50000.json"
                else:
                    set_arg(command, "--pipeline", pipeline)
                    set_arg(command, "--trial", trial)
                    set_arg(command, "--output", destination)
                    if pipeline == "l3":
                        set_arg(command, "--correctness-report", check / "result.json")
                        command += ["--l3-root", str(args.data), "--l3-library", str(args.library)]
                    paths = None  # The existing training runner warms the selected representation.
                    result_path = destination / "result.json"
                print("RUN", args.phase, args.gpu, model, pipeline, trial, flush=True)
                run(command, destination, env, paths)
                result = read(result_path)
                measured = result["repeats"][0] if "repeats" in result else result
                reference = read(original)
                if args.phase == "inference":
                    count = measured["images"] if "repeats" in result else result["samples"]
                    seconds = measured["repeat_scope_seconds"] if "repeats" in result else result["e2e_seconds"]
                    top1 = 100 * measured["accuracy_top1"] if "repeats" in result else result["top1"]
                    assert count == 50000
                    if "repeats" in result:
                        assert measured["sample_trace"] == reference["repeats"][0]["sample_trace"]
                        assert result["model"]["checkpoint_sha256"] == reference["model"]["checkpoint_sha256"]
                    else:
                        assert result["sample_ids"] == reference["sample_ids"]
                        assert result["checkpoint"] == reference["checkpoint"]
                else:
                    count, seconds, top1 = result["images"], result["seconds"], None
                    assert count == 65536 and result["updates"] == 64 and result["correctness_passed"]
                    assert read(result["sample_ids_file"]) == [s["logical_sample_id"] for s in training_samples[32768:98304]]
                    assert result["initial_model_hash"] == reference["initial_model_hash"]
                observation = dict(phase=args.phase, gpu=args.gpu, model=model, pipeline=pipeline, trial=trial,
                                   images=count, seconds=seconds, images_per_s=count/seconds, top1=top1,
                                   raw_result=str(result_path))
                (destination / "observation.json").write_text(json.dumps(observation, indent=2))
                summarize(args.output)
                print("RESULT", json.dumps(observation), flush=True)


if __name__ == "__main__":
    main()
