"""Real source512 numerical fixture: batch64, pool4096, accum16, group256.

Uses validation images only as a numerical fixture; exports no trained model
for accuracy evaluation. This is not the formal multiworker resource gate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image, __version__ as pillow_version, features

from .contracts import artifact, load_artifact, write_json
from .training import freeze_trace, update_windows
from .unified import microbatches

REPO = Path(__file__).resolve().parents[4]
SELECTION = REPO/"fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
ORIGINAL = REPO/"fastlens/data/system_rgbnomore/e2e_v2/imagenet"
MANIFEST = REPO/"fastlens/data/compressed/imagenet512_val_block_major/manifest.bin"
COUNT = 8194


def verify_producer(entries, original_root):
    """Prove byte-equivalent bilinear/default-JPEG production for selected images."""
    digest = []
    for index, row in enumerate(entries):
        path = Path(row["path"])
        original = Path(original_root)/row["logical_sample_id"]
        actual = path.read_bytes()
        with Image.open(original) as source:
            resized = source.convert("RGB").resize((512,512), Image.Resampling.BILINEAR)
            default, explicit = io.BytesIO(), io.BytesIO()
            resized.save(default, format="JPEG")
            resized.save(explicit, format="JPEG", quality=75, subsampling=2)
        if actual != default.getvalue() or actual != explicit.getvalue():
            raise ValueError(f"source512 is not byte-equivalent to the declared producer: {path}")
        with Image.open(io.BytesIO(actual)) as jpeg:
            if jpeg.size != (512,512) or jpeg.layer != [(1,2,2,0),(2,1,1,1),(3,1,1,1)]:
                raise ValueError(f"unexpected source512 geometry/sampling: {path}")
        digest.append(dict(image_id=row["logical_sample_id"], original=artifact(original),
                           jpeg=dict(path=str(path.resolve()), sha256=hashlib.sha256(actual).hexdigest())))
        if (index+1) % 256 == 0:
            print(f"producer bytes checked: {index+1}/{len(entries)}", flush=True)
    return dict(resize=dict(size=[512,512], mode="stretch without crop", resample="PIL.Image.Resampling.BILINEAR", color="RGB"),
                encoder=dict(name="Pillow JPEG", pillow=pillow_version, jpeg_codec=features.version_codec("jpg"),
                             options="defaults; verified byte-equivalent to explicit quality=75, subsampling=2"),
                quality=75, subsampling="4:2:0 (Pillow code 2)", checked_images=len(entries),
                scope="byte-equivalent reconstruction for selected fixture images; historical validation command not recovered",
                inputs=digest)


def production_traces(images, profile, labels, seed):
    """Two disjoint populations, each four whole source segments plus one tail."""
    if len(images) != COUNT or [r["galp_image_id"] for r in images] != list(range(COUNT)):
        raise ValueError("production fixture requires the first 8194 source positions in manifest order")
    traces = {}
    for name, groups in (("calibration_trace", (images[:4096], images[8192:8193])),
                         ("trace", (images[4096:8192], images[8193:8194]))):
        trace, offset = [], 0
        for pool, group in enumerate(groups):
            rows = freeze_trace(group, profile, seed=seed, epochs=1, batch_size=64, pool_size=4096,
                                labels=labels, partition=f"{name}:pool{pool}", crop_policy="per-pls")
            for row in rows:
                row["logical_slot"] += len(trace)
                row["microbatch_id"] += offset
                row["closed_pool_id"] = f"0:{pool}"
            trace.extend(rows)
            offset += len(microbatches(rows))
        windows = update_windows(trace, 16)
        if [len(w) for w in windows] != [1024,1024,1024,1024,1]:
            raise AssertionError("production accumulation or tail was lost")
        traces[name] = trace
    return traces


def command(output, name, *arguments):
    argv = [sys.executable, "-B", "-m", "fastlens.benchmarks.baselines.dct_baselines", *map(str, arguments)]
    print("Running", name, flush=True)
    with (output/f"{name}.log").open("x") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)


def prepare(output, *, model, selection=SELECTION, original_root=ORIGINAL):
    output = Path(output).resolve()
    selection = Path(selection).resolve()
    entries = json.loads(selection.read_text())["samples"][:COUNT]
    if len(entries) != COUNT:
        raise ValueError("need 8194 real source images for disjoint full pools and tails")
    output.mkdir(parents=True, exist_ok=False)
    recipe = verify_producer(entries, original_root)
    recipe.update(selection=artifact(selection),
                  training_producer=artifact(REPO/"fastlens/benchmarks/baselines/system_rgbnomore/training/prepare_imagenet512_v3_train.py"),
                  training_generation_log=artifact(REPO/"fastlens/data/system_rgbnomore/e2e_v3/prepare_imagenet512_v3_train.full.log"))
    write_json(output/"source_recipe.json", recipe)
    contract_dir = output/"contract"
    command(output, "prepare", "prepare", "--selection", selection, "--source-recipe", output/"source_recipe.json",
            "--model", model, "--upstream", Path.home()/"RGB-no-more", "--count", COUNT,
            "--batch-size", 64, "--pool-size", 4096, "--workers", 4, "--torch-threads", 8,
            "--seed", 11997733, "--training", "--epochs", 1, "--calibration-count", 4097,
            "--accumulation", 16, "--total-updates", 375600, "--output", contract_dir)
    config_path = contract_dir/"matched.json"
    config = json.loads(config_path.read_text())
    source, profile = load_artifact(config["source_manifest"]), load_artifact(config["model_profile"])
    labels = load_artifact(config["label_mapping"])["labels"]
    traces = production_traces(source["images"], profile, labels, 11997733)
    for name, rows in traces.items():
        path = contract_dir/f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows))
        config["calibration_trace" if name == "calibration_trace" else "request_trace"] = artifact(path)
    config["training"].update(crop_policy="per-pls", numerical_fixture_only=True,
                              schedule_population=1281167, schedule_updates_per_epoch=1252)
    write_json(config_path, config)
    from .runner import inputs
    inputs(config_path)
    # Re-read the actual frozen JPEG hashes after reconstruction; avoid a gap
    # between producer verification and the source used to encode the stores.
    checked = {r["image_id"]: r["jpeg"]["sha256"] for r in recipe["inputs"]}
    if any(r["jpeg_sha256"] != checked[r["image_id"]] for r in source["images"]):
        raise ValueError("source changed between producer verification and coefficient extraction")
    for method in ("ic_matched_shared", "ic_matched_batched", "fs_matched_b8"):
        command(output, f"build_{method}", "build", "--config", config_path, "--method", method,
                "--group-size", 256, "--output", output/method)
    return config_path


def run(output, *, model, manifest=MANIFEST, prepared=None):
    output = Path(output).resolve()
    os.environ["FASTLENS_BLOCK_MAJOR_ACCESS_DIR"] = str(Path(manifest).parent/"access")
    if prepared is None:
        config = prepare(output, model=model)
        prepared = output
        snapshot = output/"initial.pt"
        command(output, "snapshot", "snapshot", "--config", config, "--output", snapshot)
    else:
        prepared = Path(prepared).resolve()
        config, snapshot = prepared/"contract/matched.json", prepared/"initial.pt"
        from .runner import inputs
        _, _, profile, _, _ = inputs(config)
        if profile["model"] != model:
            raise ValueError("prepared model differs from --model")
        output.mkdir(parents=True, exist_ok=False)
    for name, backend, root in (("cpu_dct", "cpu_dct", None), ("fastlens", "fastlens", manifest),
                               ("ic_shared", "ic", prepared/"ic_matched_shared"),
                               ("ic_batched", "ic", prepared/"ic_matched_batched"),
                               ("fs", "fs", prepared/"fs_matched_b8")):
        common = ["--config", config, "--snapshot", snapshot, "--backend", backend,
                  "--device", "cuda:0", "--updates", 5]
        if root is not None:
            common += ["--root", root]
        if backend == "fastlens":
            common += ["--module-path", REPO/"build/fastlens/torch", "--source-workers", 48]
        calibration = output/f"{name}.calibration.json"
        command(output, f"{name}_calibrate", "calibrate", *common, "--input-atol", 0,
                "--model-atol", 0, "--state-atol", 0, "--rtol", 0, "--output", calibration)
        command(output, f"{name}_verify", "verify-training", *common, "--calibration", calibration,
                "--output", output/f"{name}.verified.json")
    print(f"Passed: {model}, 5 backends, 5 updates each, full accum16 windows and one-sample tail. Evidence: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for fixture and raw results")
    parser.add_argument("--model", choices=("vitti", "resnet24"), default="vitti")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", type=Path, help="reuse an existing production fixture and initial.pt; write fresh evidence to --output")
    args = parser.parse_args()
    if args.prepare_only and args.prepared is not None:
        parser.error("--prepared cannot be combined with --prepare-only")
    if args.prepare_only:
        print(prepare(args.output, model=args.model))
    else:
        run(args.output, model=args.model, prepared=args.prepared)


if __name__ == "__main__":
    main()
