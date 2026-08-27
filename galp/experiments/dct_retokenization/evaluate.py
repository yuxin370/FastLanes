#!/usr/bin/env python3
"""Evaluate frozen or trained DCT retokenization configurations on ImageNet val."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from galp.experiments.dct_retokenization.data import MaskedDctPreprocessor, load_manifest, make_loader
from galp.experiments.dct_retokenization.delayed_wrapper import (
    DelayedRetokenizationWrapper,
    load_delayed_checkpoint,
)
from galp.experiments.dct_retokenization.model_wrapper import (
    DctRetokenizationWrapper,
    build_pretrained_dct_model,
    load_experiment_checkpoint,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RGBNOMORE_ROOT = Path(os.environ.get("RGBNOMORE_ROOT", "RGB-no-more")).expanduser()
DEFAULT_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
DEFAULT_VAL_MANIFEST = (
    REPO_ROOT
    / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
)
DEFAULT_PRIOR_RESULTS = HERE.parent / "coefficient_mask_evaluator/runs/imagenet_val_k1_64_20260816_h100"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "captured_at_unix_ns": time.time_ns(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": result.stdout.strip(),
        "nvidia_smi_stderr": result.stderr.strip(),
        "loadavg": os.getloadavg(),
    }


def load_config(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configurations = payload.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("evaluation config has no configurations")
    ids: set[str] = set()
    for config in configurations:
        required = {"id", "k", "n", "merge_axis", "merge_type"}
        if missing := required - set(config):
            raise ValueError(f"configuration misses {sorted(missing)}: {config}")
        if config["id"] in ids:
            raise ValueError(f"duplicate configuration ID {config['id']}")
        ids.add(str(config["id"]))
    return payload, configurations


def build_wrappers(
    configurations: Sequence[dict[str, Any]],
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[list[torch.nn.Module], torch.nn.Module]:
    frozen_base = build_pretrained_dct_model(rgbnomore_root, checkpoint, device).eval()
    wrappers: list[torch.nn.Module] = []
    for config in configurations:
        checkpoint_value = config.get("experiment_checkpoint")
        base = (
            frozen_base
            if checkpoint_value is None
            else build_pretrained_dct_model(rgbnomore_root, checkpoint, device).eval()
        )
        architecture = str(config.get("architecture", "post_projection"))
        if architecture == "delayed_retokenization":
            wrapper = DelayedRetokenizationWrapper(
                base,
                merge_after_block=int(config["merge_after_block"]),
                merge_axis=str(config["merge_axis"]),
                initialization=str(config.get("initialization", "keep_first")),
            ).eval()
        else:
            wrapper = DctRetokenizationWrapper(
                base,
                token_count=int(config["n"]),
                merge_axis=str(config["merge_axis"]),
                merge_type=str(config["merge_type"]),
                architecture=architecture,
                initialization=str(config.get("initialization", "average")),
                position_mode=str(config.get("position_mode", "pooled_existing")),
            ).eval()
        if checkpoint_value is not None:
            if isinstance(wrapper, DelayedRetokenizationWrapper):
                load_delayed_checkpoint(wrapper, Path(str(checkpoint_value)).resolve())
            else:
                load_experiment_checkpoint(wrapper, Path(str(checkpoint_value)).resolve())
            wrapper.eval()
        wrappers.append(wrapper)
    return wrappers, frozen_base


def prior_predictions(prior_root: Path, k: int) -> np.ndarray | None:
    if not prior_root.is_dir():
        return None
    conditions = json.loads((prior_root / "conditions.json").read_text(encoding="utf-8"))
    condition_id = f"prefix_k{k:02d}"
    matches = [index for index, value in enumerate(conditions) if value["condition_id"] == condition_id]
    if not matches:
        return None
    return np.load(prior_root / "top5_classes.npy", mmap_mode="r")[matches[0]]


def create_arrays(output_dir: Path, count: int, samples: int) -> tuple[np.memmap, np.memmap, np.memmap]:
    classes = np.lib.format.open_memmap(
        output_dir / "top5_classes.npy", mode="w+", dtype=np.int16, shape=(count, samples, 5)
    )
    probabilities = np.lib.format.open_memmap(
        output_dir / "top5_probabilities.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, samples, 5),
    )
    losses = np.lib.format.open_memmap(
        output_dir / "cross_entropy.npy", mode="w+", dtype=np.float32, shape=(count, samples)
    )
    classes.fill(-1)
    probabilities.fill(np.nan)
    losses.fill(np.nan)
    return classes, probabilities, losses


def write_per_sample_csv(
    path: Path,
    samples: Sequence[Any],
    configurations: Sequence[dict[str, Any]],
    classes: np.ndarray,
    losses: np.ndarray,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=3) as stream:
        writer = csv.writer(stream)
        header = ["sample_index", "logical_sample_id", "path", "label"]
        for config in configurations:
            header.extend((f"{config['id']}__top1", f"{config['id']}__correct", f"{config['id']}__ce"))
        writer.writerow(header)
        for sample_index, sample in enumerate(samples):
            row: list[Any] = [sample_index, sample.logical_sample_id, sample.path, sample.label]
            for config_index in range(len(configurations)):
                prediction = int(classes[config_index, sample_index, 0])
                row.extend(
                    (
                        prediction,
                        int(prediction == sample.label),
                        f"{float(losses[config_index, sample_index]):.8g}",
                    )
                )
            writer.writerow(row)


def metrics_rows(
    configurations: Sequence[dict[str, Any]],
    classes: np.ndarray,
    losses: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for index, config in enumerate(configurations):
        correct1 = int(np.count_nonzero(classes[index, :, 0] == labels))
        correct5 = int(np.count_nonzero(np.any(classes[index] == labels[:, None], axis=1)))
        raw.append(
            {
                "configuration_index": index,
                "configuration_id": config["id"],
                "k": int(config["k"]),
                "n": int(config["n"]),
                "merge_axis": config["merge_axis"],
                "merge_type": config["merge_type"],
                "architecture": config.get("architecture", "post_projection"),
                "initialization": config.get("initialization", "average"),
                "position_mode": config.get("position_mode", "pooled_existing"),
                "merge_after_block": config.get("merge_after_block", ""),
                "experiment_checkpoint": config.get("experiment_checkpoint", ""),
                "sample_count": int(labels.size),
                "correct_top1": correct1,
                "correct_top5": correct5,
                "top1_percent": 100.0 * correct1 / labels.size,
                "top5_percent": 100.0 * correct5 / labels.size,
                "cross_entropy": float(np.mean(losses[index], dtype=np.float64)),
            }
        )
    by_k_n196 = {row["k"]: row for row in raw if row["n"] == 196}
    baseline = next(row for row in raw if row["k"] == 64 and row["n"] == 196)
    for row in raw:
        same_k = by_k_n196.get(row["k"])
        row["relative_to_k64_n196_pp"] = row["top1_percent"] - baseline["top1_percent"]
        row["delta_coefficient_pp"] = (
            "" if same_k is None else same_k["top1_percent"] - baseline["top1_percent"]
        )
        row["delta_token_pp"] = "" if same_k is None else row["top1_percent"] - same_k["top1_percent"]
    return raw


def run(args: argparse.Namespace) -> int:
    config_payload, configurations = load_config(args.config.resolve())
    _, samples = load_manifest(args.manifest.resolve(), max_samples=args.max_samples)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    torch.cuda.set_device(device)
    observed_name = torch.cuda.get_device_name(device)
    if args.expected_device_name and observed_name != args.expected_device_name:
        raise RuntimeError(f"expected {args.expected_device_name!r}, observed {observed_name!r}")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(args.torch_cpu_threads)

    signature = {
        "schema": "dct-retokenization-evaluation-run-v1",
        "config": config_payload,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "sample_count": len(samples),
        "device_name": observed_name,
        "mask_stage": "raw_quantized_coefficients_before_dequantization_and_frequency_mixing",
        "position_modes": sorted(
            {str(config.get("position_mode", "pooled_existing")) for config in configurations}
        ),
    }
    write_json(output_dir / "run_signature.json", signature)
    write_json(output_dir / "hardware_before.json", gpu_snapshot())
    write_json(output_dir / "configurations.json", configurations)

    wrappers, original_model = build_wrappers(
        configurations, args.rgbnomore_root.resolve(), args.checkpoint.resolve(), device
    )
    preprocessors = {
        k: MaskedDctPreprocessor(args.rgbnomore_root.resolve(), device, k)
        for k in sorted({int(config["k"]) for config in configurations})
    }
    by_k: dict[int, list[int]] = {}
    for index, config in enumerate(configurations):
        by_k.setdefault(int(config["k"]), []).append(index)
    classes, probabilities, losses = create_arrays(output_dir, len(configurations), len(samples))
    logit_count = min(args.logit_samples, len(samples))
    logits_capture = np.lib.format.open_memmap(
        output_dir / "fixed_sample_logits.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(configurations), logit_count, 1000),
    )
    logits_capture.fill(np.nan)

    loader = make_loader(
        samples,
        args.rgbnomore_root.resolve(),
        indices=None,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    labels_np = np.asarray([sample.label for sample in samples], dtype=np.int16)
    completed = 0
    started = time.perf_counter()
    gate: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_index, (yq, cq, quant, labels, ordinals) in enumerate(loader):
            batch = int(labels.shape[0])
            observed_ordinals = [int(value) for value in ordinals.tolist()]
            expected_ordinals = list(range(completed, completed + batch))
            if observed_ordinals != expected_ordinals:
                raise RuntimeError("validation sample order changed")
            labels_device = labels.to(device, non_blocking=True)
            for k, config_indices in by_k.items():
                y, cbcr = preprocessors[k].validation(yq, cq, quant)
                for config_index in config_indices:
                    logits = wrappers[config_index](y, cbcr)
                    sample_losses = torch.nn.functional.cross_entropy(
                        logits, labels_device, reduction="none"
                    )
                    top_prob, top_classes = torch.topk(torch.softmax(logits.float(), dim=1), 5, dim=1)
                    selection = slice(completed, completed + batch)
                    classes[config_index, selection] = top_classes.to(torch.int16).cpu().numpy()
                    probabilities[config_index, selection] = top_prob.cpu().numpy()
                    losses[config_index, selection] = sample_losses.float().cpu().numpy()
                    capture_end = min(completed + batch, logit_count)
                    if completed < capture_end:
                        logits_capture[config_index, completed:capture_end] = (
                            logits[: capture_end - completed].float().cpu().numpy()
                        )

                    config = configurations[config_index]
                    if batch_index == 0 and int(config["k"]) == 64 and int(config["n"]) == 196:
                        reference = original_model(y, cbcr)
                        difference = float((reference - logits).abs().max().item())
                        gate["k64_n196_logit_max_abs"] = difference
                        gate["k64_n196_top1_equal"] = bool(
                            torch.equal(reference.argmax(1), logits.argmax(1))
                        )
                        gate["k64_n196_top5_equal"] = bool(
                            torch.equal(reference.topk(5, 1).indices, logits.topk(5, 1).indices)
                        )
                        if difference > args.logit_atol or not gate["k64_n196_top5_equal"]:
                            raise AssertionError(f"K64/N196 equivalence gate failed: {gate}")
            completed += batch
            if (batch_index + 1) % args.flush_every_batches == 0 or completed == len(samples):
                classes.flush(); probabilities.flush(); losses.flush(); logits_capture.flush()
                write_json(
                    output_dir / "progress.json",
                    {
                        "status": "running",
                        "completed_samples": completed,
                        "sample_count": len(samples),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
            if (batch_index + 1) % args.log_every_batches == 0 or completed == len(samples):
                rate = completed / (time.perf_counter() - started)
                print(
                    f"progress {completed}/{len(samples)} ({100*completed/len(samples):.2f}%), "
                    f"source_rate={rate:.2f}/s effective={rate*len(configurations):.2f} predictions/s",
                    flush=True,
                )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    rows = metrics_rows(configurations, classes, losses, labels_np)
    with (output_dir / "accuracy_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_per_sample_csv(
        output_dir / "per_sample_predictions.csv.gz", samples, configurations, classes, losses
    )

    agreements: dict[str, Any] = dict(gate)
    for k in (64, 32, 24, 16):
        prior = prior_predictions(args.prior_results.resolve(), k)
        candidates = [
            (index, config)
            for index, config in enumerate(configurations)
            if int(config["k"]) == k and int(config["n"]) == 196
        ]
        if prior is not None and candidates and prior.shape[0] >= len(samples):
            index, config = candidates[0]
            observed = np.asarray(classes[index])
            agreements[f"{config['id']}_prior_top1_agreement"] = float(
                np.mean(observed[:, 0] == prior[: len(samples), 0])
            )
            agreements[f"{config['id']}_prior_top5_exact_agreement"] = float(
                np.mean(np.all(observed == prior[: len(samples)], axis=1))
            )
    if agreements.get("k32_n196_prior_top1_agreement", 1.0) != 1.0:
        raise AssertionError("K32/N196 did not reproduce the coefficient-mask evaluator")
    write_json(output_dir / "equivalence.json", agreements)
    write_json(output_dir / "hardware_after.json", gpu_snapshot())
    metadata = {
        "status": "complete",
        "sample_count": len(samples),
        "configuration_count": len(configurations),
        "elapsed_seconds": elapsed,
        "effective_predictions_per_second": len(samples) * len(configurations) / elapsed,
        "device_name": observed_name,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "equivalence": agreements,
        "preprocessor_audits": {str(k): value.audit_record() for k, value in preprocessors.items()},
    }
    write_json(output_dir / "run_metadata.json", metadata)
    write_json(
        output_dir / "progress.json",
        {"status": "complete", "completed_samples": len(samples), "sample_count": len(samples)},
    )
    print(f"complete in {elapsed:.2f}s: {output_dir}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/frozen_matrix.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--prior-results", type=Path, default=DEFAULT_PRIOR_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--logit-samples", type=int, default=64)
    parser.add_argument("--logit-atol", type=float, default=1e-6)
    parser.add_argument("--flush-every-batches", type=int, default=20)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    for name in ("batch_size", "logit_samples", "flush_every_batches", "log_every_batches", "torch_cpu_threads"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_','-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
