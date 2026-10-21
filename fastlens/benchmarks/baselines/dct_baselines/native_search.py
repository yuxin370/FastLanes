"""Freeze the round-3 IC-native search domain and isolated data splits (N1)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .contracts import artifact, write_json
from .native import NativeSpec
from .store import fingerprint


DOMAIN = "ic_native_subset96_v1"
DOMAIN_SEED = 20261009


def subset96():
    """Project sample; the paper's unpublished exact 96 candidates are unknown."""
    blocks = (8, 16, 32, 64, 128, 256)
    endpoints = {(b, k) for b in blocks for k in (1, b)}
    remaining = [(b, k) for b in blocks for k in range(1, b+1) if (b, k) not in endpoints]
    rng = np.random.Generator(np.random.PCG64(DOMAIN_SEED))
    chosen = endpoints | {remaining[int(i)] for i in rng.choice(len(remaining), 84, replace=False)}
    return [dict(candidate_id=f"B{b:03d}_k{k:03d}_s1x1_q050", spec=NativeSpec(b, k).contract())
            for b, k in sorted(chosen)]


def isolated_splits(train, final, *, search_val_count, seed):
    """Use only training population for fit/selection; preserve final evaluation."""
    def population(value):
        rows = value["samples"]
        ids = [r["logical_sample_id"] for r in rows]
        paths = [str(Path(r["path"]).resolve()) for r in rows]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("source population contains duplicate IDs or paths")
        if any(type(r["label"]) is not int or not 0 <= r["label"] < 1000 for r in rows):
            raise ValueError("source labels must be integer class IDs in [0,999]")
        return ids, paths
    train_ids, train_paths = population(train)
    final_ids, final_paths = population(final)
    if set(train_ids) & set(final_ids) or set(train_paths) & set(final_paths):
        raise ValueError("training and final evaluation populations overlap")
    if type(search_val_count) is not int or not 0 < search_val_count < len(train_ids) or not final_ids:
        raise ValueError("require nonempty fit, search validation and final evaluation splits")
    ordered = sorted(train_ids)
    rng = np.random.Generator(np.random.PCG64(seed))
    permutation = rng.permutation(len(ordered))
    selected = {ordered[int(i)] for i in permutation[:search_val_count]}
    return dict(seed=seed, algorithm="PCG64 permutation of sorted training IDs; global split, not stratified",
                train_fit=[i for i in ordered if i not in selected], search_val=sorted(selected),
                final_evaluation=sorted(final_ids),
                usage=dict(weights_and_statistics="train_fit", selection_and_tuning="search_val",
                           final_evaluation="only after representation/checkpoint frozen; never used for selection"))


def validate_task(task):
    # These are inputs to the requested frozen task, not guessed training defaults.
    for key in ("model", "rgb_preprocessing", "hardware", "training", "final_training", "runtime_candidates"):
        if not isinstance(task[key], dict) or not task[key]:
            raise ValueError(f"{key} must explicitly describe the search experiment")
    required = dict(model=("family", "classes", "stem", "small_input_policy"),
                    rgb_preprocessing=("source_recipe", "train", "evaluation"),
                    hardware=("gpu_uuid", "gpu_name", "cpu_threads", "host_memory_bytes", "cuda_memory_bytes"),
                    training=("scratch_epochs", "transfer_epochs", "seed", "optimizer", "transfer_state"),
                    final_training=("epochs", "confirmation_seeds"),
                    runtime_candidates=("batch_sizes", "data_workers", "decode_workers", "cache_bytes"))
    for group, keys in required.items():
        missing = set(keys)-task[group].keys()
        if missing:
            raise ValueError(f"{group} missing fields: {sorted(missing)}")
    for key in required["runtime_candidates"]:
        values = task["runtime_candidates"][key]
        minimum = 0 if key in ("data_workers", "cache_bytes") else 1
        if not isinstance(values, list) or not values or any(type(v) is not int or v < minimum for v in values):
            raise ValueError(f"runtime_candidates.{key} must list legal integer values")
    objective = task["objective"]
    if objective["metric"] not in ("min_e2e_latency_ms", "max_accuracy"):
        raise ValueError("objective metric must be min_e2e_latency_ms or max_accuracy")
    if objective["accuracy_unit"] != "percentage_points":
        raise ValueError("accuracy budgets use percentage points")
    if objective["metric"] == "min_e2e_latency_ms":
        if not objective["reference_candidate_id"]:
            raise ValueError("accuracy constraint requires a frozen reference candidate")
        delta = objective["accuracy_delta_pp"]
        if not math.isfinite(delta) or not 0 <= delta <= 100:
            raise ValueError("accuracy delta must be in [0,100] percentage points")
    for key in ("search_gpu_hours", "latency_limit_ms", "storage_limit_bytes"):
        value = task["budgets"][key]
        if value is None and key != "search_gpu_hours":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive (optional limits may be null)")
    if objective["metric"] == "max_accuracy" and task["budgets"]["latency_limit_ms"] is None:
        raise ValueError("max_accuracy search requires a latency budget")
    if type(task["split_seed"]) is not int or task["split_seed"] < 0:
        raise ValueError("split_seed must be a nonnegative integer")


def prepare(task_path, output):
    task_path, output = Path(task_path).resolve(), Path(output)
    task = json.loads(task_path.read_text())
    validate_task(task)
    def source(key):
        path = Path(task[key])
        path = path if path.is_absolute() else task_path.parent/path
        return path, json.loads(path.read_text())
    train_path, train = source("train_manifest")
    final_path, final = source("final_manifest")
    splits = isolated_splits(train, final, search_val_count=task["search_val_count"], seed=task["split_seed"])
    candidates = subset96()
    reference = task["objective"].get("reference_candidate_id")
    if reference is not None and reference not in {r["candidate_id"] for r in candidates}:
        raise ValueError("reference candidate must belong to the frozen domain")
    domain = dict(domain=DOMAIN, seed=DOMAIN_SEED, rng="NumPy PCG64", numpy_version=np.__version__,
                  sampling="no sampling", scalar_quantization=50, candidates=candidates,
                  provenance="project deterministic subset; does not reconstruct the unpublished paper sample",
                  bucket_plan=dict(count=16, size=6, key_indices=[0,3,6,9,12,15],
                                   sorting="measured NPZ bytes/image descending (least compression first), candidate_id ascending on ties",
                                   train="every candidate in each key bucket; interpolation is bucket upper envelope, not candidate accuracy"))
    # No final images/labels are decoded for task preparation.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"candidate_manifest.json", domain)
    write_json(output/"splits.json", splits)
    contract = dict(version=1, stage="prepared", domain=DOMAIN, task=task,
                    task_input=artifact(task_path), train_source=artifact(train_path), final_source=artifact(final_path),
                    candidates=artifact(output/"candidate_manifest.json"), splits=artifact(output/"splits.json"),
                    source_identity_scope="manifest bytes; actual image-content binding belongs to encode/fit",
                    representation="native scalar-quantized mosaic; float normalization, no dequantization/IDCT or mosaic resize",
                    status="N1 only: no candidate trained/profiled/selected", headline_eligible=False)
    contract["search_contract_sha256"] = fingerprint(contract)
    write_json(output/"search_contract.json", contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.task, args.output)
    print(json.dumps(dict(contract=str(args.output/"search_contract.json"), domain=result["domain"],
                          status=result["status"], headline_eligible=False), indent=2))


if __name__ == "__main__":
    main()
