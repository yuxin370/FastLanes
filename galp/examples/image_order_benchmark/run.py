#!/usr/bin/env python3
"""Run a controlled GALP random-vs-contiguous image-ID experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from analysis import distribution, summarize, write_summary_artifacts
from selection import SUPPORTED_SAMPLING, build_condition_orders, canonical_sha256, parse_galp_layout
from training_order import analyze_training_index


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SYSTEM_BENCHMARK_DIR = REPO_ROOT / "galp/benchmarks/system_rgbnomore"
DEFAULT_MANIFEST = REPO_ROOT / "galp/data/imagedataset_dct/ImageNet-val/manifest.bin"
DEFAULT_LABELS = REPO_ROOT / "galp/data/imagedataset_dct/ImageNet-val/labels.json"
DEFAULT_RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_INDEX_CSV = DEFAULT_RGBNOMORE_ROOT / "assets/indexbase_val.csv"
DEFAULT_TRAIN_INDEX_CSV = DEFAULT_RGBNOMORE_ROOT / "assets/indexbase_train.csv"
DEFAULT_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
DEFAULT_BINDING_DIR = REPO_ROOT / "build/galp/torch"
DEFAULT_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")
CONDITIONS = ("current_random", "paired_scattered", "contiguous")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampling_mode(metadata: dict[str, Any]) -> str:
    components = metadata.get("components", [])
    by_slot = {
        int(component.get("semantic_slot_id")): component
        for component in components
        if component.get("present") and int(component.get("semantic_slot_id", -1)) in (0, 1, 2)
    }
    if 0 not in by_slot:
        by_slot = {
            int(component.get("local_component_index")): component
            for component in components
            if component.get("present") and int(component.get("local_component_index", -1)) in (0, 1, 2)
        }
    if 0 in by_slot and 1 not in by_slot and 2 not in by_slot:
        return "grayscale"
    if not {0, 1, 2}.issubset(by_slot):
        return "unknown"
    y, cb, cr = by_slot[0], by_slot[1], by_slot[2]
    cb_factors = (int(cb.get("h_samp_factor", 0)), int(cb.get("v_samp_factor", 0)))
    cr_factors = (int(cr.get("h_samp_factor", 0)), int(cr.get("v_samp_factor", 0)))
    y_factors = (int(y.get("h_samp_factor", 0)), int(y.get("v_samp_factor", 0)))
    if cb_factors != cr_factors:
        return "mismatched_chroma"
    if y_factors == cb_factors:
        return "4:4:4"
    if y_factors == (cb_factors[0] * 2, cb_factors[1] * 2):
        return "4:2:0"
    if y_factors == (cb_factors[0] * 2, cb_factors[1]):
        return "4:2:2"
    return "unsupported"


def _load_labels(path: Path, image_count: int) -> tuple[list[int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if (
        not isinstance(labels, list)
        or len(labels) != image_count
        or not all(isinstance(label, int) and 0 <= label < 1000 for label in labels)
    ):
        raise ValueError(f"{path} must contain {image_count} ImageNet-1K labels indexed by GALP image ID")
    if payload.get("format") != "galp_rgbnomore_label_map_v1":
        raise ValueError(f"unexpected label-map format: {path}")
    return labels, payload


def _load_runtime_modules(binding_dir: Path):
    for path in (binding_dir, SYSTEM_BENCHMARK_DIR):
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    import _galp_direct_dct as galp_dct
    from dataset import manifest as system_manifest
    from inference import pipeline as system_pipeline

    return galp_dct, system_manifest, system_pipeline


def _samples(image_ids: Sequence[int], labels: Sequence[int]) -> list[dict[str, int]]:
    return [
        {"ordinal": ordinal, "galp_image_id": int(image_id), "label": int(labels[int(image_id)])}
        for ordinal, image_id in enumerate(image_ids)
    ]


def _batches(samples: Sequence[dict[str, int]], batch_size: int) -> list[list[dict[str, int]]]:
    if len(samples) % batch_size:
        raise ValueError(f"sample count {len(samples)} is not divisible by batch size {batch_size}")
    return [list(samples[offset : offset + batch_size]) for offset in range(0, len(samples), batch_size)]


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pipelines": {
            "galp": {
                "manifest": str(args.galp_manifest.resolve()),
                "preprocess": "rgbnomore-val-pushdown",
                "cache_capacity_mib": int(args.cache_capacity_mib),
                "decode_batch_rowgroups": int(args.decode_batch_rowgroups),
                "rowgroup_prefetch_depth": int(args.rowgroup_prefetch_depth),
                "rowgroup_prefetch_workers": int(args.rowgroup_prefetch_workers),
            },
            "rgbnomore": {"root": str(args.rgbnomore_root.resolve())},
        },
        "models": {"dct": {"checkpoint": str(args.checkpoint.resolve())}},
        "execution": {"precision": args.precision},
    }


def _native_accumulate(target: dict[str, float | int], source: dict[str, float | int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _run_condition_once(
    *,
    condition: str,
    repeat: int,
    execution_order: int,
    adapter: Any,
    model: torch.nn.Module,
    samples: Sequence[dict[str, int]],
    batch_size: int,
    warmup_batches: int,
    measurement_batches: int,
    contract: dict[str, Any],
    device: torch.device,
    system_pipeline: Any,
    capture_semantic: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    sample_batches = _batches(samples, batch_size)
    total_batches = warmup_batches + measurement_batches
    if len(sample_batches) != total_batches:
        raise ValueError(f"{condition} has {len(sample_batches)} batches, expected {total_batches}")

    def following(index: int) -> list[dict[str, int]] | None:
        return sample_batches[index + 1] if index + 1 < total_batches else None

    adapter.begin_repeat()
    for batch_index in range(warmup_batches):
        expected = sample_batches[batch_index]
        batch = adapter.load(expected, following(batch_index))
        system_pipeline._validate_batch_identity(batch, expected)
        system_pipeline._forward(model, batch.inputs, contract, device)
        torch.cuda.synchronize(device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    latency_ms: list[float] = []
    loader_ms: list[float] = []
    forward_ms: list[float] = []
    correct1 = 0
    correct5 = 0
    native_totals: dict[str, float | int] = {}
    semantic_ids: list[np.ndarray] = []
    semantic_labels: list[np.ndarray] = []
    semantic_logits: list[np.ndarray] = []

    for measured_index in range(measurement_batches):
        batch_index = warmup_batches + measured_index
        expected = sample_batches[batch_index]
        wall_started = time.perf_counter_ns()
        load_started = wall_started
        batch = adapter.load(expected, following(batch_index))
        load_ended = time.perf_counter_ns()
        system_pipeline._validate_batch_identity(batch, expected)
        forward_started = time.perf_counter_ns()
        logits = system_pipeline._forward(model, batch.inputs, contract, device)
        batch_correct1, batch_correct5 = system_pipeline._accuracy_counts(logits, batch.labels)
        torch.cuda.synchronize(device)
        wall_ended = time.perf_counter_ns()
        if capture_semantic:
            semantic_ids.append(np.asarray([sample["galp_image_id"] for sample in expected], dtype=np.int64))
            semantic_labels.append(np.asarray([sample["label"] for sample in expected], dtype=np.int64))
            semantic_logits.append(logits.detach().float().cpu().numpy())
        latency_ms.append((wall_ended - wall_started) / 1e6)
        loader_ms.append((load_ended - load_started) / 1e6)
        forward_ms.append((wall_ended - forward_started) / 1e6)
        correct1 += batch_correct1
        correct5 += batch_correct5
        _native_accumulate(native_totals, batch.native_stage_seconds)
        _native_accumulate(native_totals, batch.native_counters)

    adapter.end_repeat()
    images = measurement_batches * batch_size
    seconds = sum(latency_ms) / 1000.0
    native_stage_seconds = {
        key: float(value) for key, value in native_totals.items() if key.endswith("_seconds")
    }
    native_counters = {
        key: int(value) for key, value in native_totals.items() if not key.endswith("_seconds")
    }
    record = {
        "condition": condition,
        "repeat": repeat,
        "execution_order": execution_order,
        "images": images,
        "seconds": seconds,
        "throughput_images_per_s": images / seconds,
        "latency_ms": distribution(latency_ms),
        "loader_and_preprocess_ms": distribution(loader_ms),
        "forward_and_metrics_ms": distribution(forward_ms),
        "accuracy_top1": correct1 / images,
        "accuracy_top5": correct5 / images,
        "correct_top1": correct1,
        "correct_top5": correct5,
        "native_stage_seconds": native_stage_seconds,
        "native_counters": native_counters,
        "peak_torch_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_torch_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    semantic = None
    if capture_semantic:
        semantic = {
            "image_ids": np.concatenate(semantic_ids),
            "labels": np.concatenate(semantic_labels),
            "logits": np.concatenate(semantic_logits),
        }
    return record, semantic


def _run_performance(
    *,
    args: argparse.Namespace,
    selection: dict[str, Any],
    labels: Sequence[int],
    output_dir: Path,
    galp_dct: Any,
    system_pipeline: Any,
) -> tuple[dict[str, Any], dict[str, Path]]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the formal GALP image-order benchmark requires CUDA")
    torch.cuda.set_device(device)
    contract = _contract(args)
    model = system_pipeline._build_dct_model(contract, device)
    model.eval()
    samples_by_condition: dict[str, list[dict[str, int]]] = {}
    adapters: dict[str, Any] = {}
    for condition in CONDITIONS:
        image_ids = selection["conditions"][condition]["flat_image_ids"]
        samples_by_condition[condition] = _samples(image_ids, labels)
        adapters[condition] = system_pipeline.GalpAdapter(contract, samples_by_condition[condition], device)

    payload: dict[str, Any] = {
        "schema_version": "galp_image_order_performance_v1",
        "selection_sha256": canonical_sha256(selection),
        "exclude_first_repeat": args.repeats > 1,
        "timing_boundary": (
            "steady-state batch request through Direct-DCT read/decode/preprocess, host/device-ready inputs, "
            "JPEG-Ti forward, top1/top5 accounting, and per-batch CUDA synchronization"
        ),
        "interleaving": "rotating condition order by repeat",
        "records": {condition: [] for condition in CONDITIONS},
    }
    semantic_paths: dict[str, Path] = {}
    for repeat in range(args.repeats):
        rotation = repeat % len(CONDITIONS)
        execution_conditions = CONDITIONS[rotation:] + CONDITIONS[:rotation]
        for order_index, condition in enumerate(execution_conditions):
            print(f"PERFORMANCE repeat={repeat} order={order_index} condition={condition}", flush=True)
            record, semantic = _run_condition_once(
                condition=condition,
                repeat=repeat,
                execution_order=order_index,
                adapter=adapters[condition],
                model=model,
                samples=samples_by_condition[condition],
                batch_size=args.batch_size,
                warmup_batches=args.warmup_batches,
                measurement_batches=args.measurement_batches,
                contract=contract,
                device=device,
                system_pipeline=system_pipeline,
                capture_semantic=repeat == 0,
            )
            payload["records"][condition].append(record)
            if semantic is not None:
                path = output_dir / f"semantic_{condition}.npz"
                np.savez_compressed(path, **semantic)
                semantic_paths[condition] = path
            _write_json(output_dir / "performance.json", payload)
            print(
                f"RESULT condition={condition} repeat={repeat} "
                f"throughput={record['throughput_images_per_s']:.3f} img/s "
                f"top1={record['accuracy_top1']:.6f}",
                flush=True,
            )
    del model
    del adapters
    torch.cuda.empty_cache()
    return payload, semantic_paths


def _load_cached_inputs(
    *,
    name: str,
    image_batches: Sequence[Sequence[int]],
    labels: Sequence[int],
    contract: dict[str, Any],
    device: torch.device,
    system_pipeline: Any,
) -> dict[str, Any]:
    image_ids = [int(image_id) for batch in image_batches for image_id in batch]
    samples = _samples(image_ids, labels)
    sample_batches = _batches(samples, len(image_batches[0]))
    adapter = system_pipeline.GalpAdapter(contract, samples, device)
    input_parts: list[list[torch.Tensor]] = [[], []]
    label_parts: list[torch.Tensor] = []
    id_parts: list[torch.Tensor] = []
    adapter.begin_repeat()
    for index, expected in enumerate(sample_batches):
        following = sample_batches[index + 1] if index + 1 < len(sample_batches) else None
        batch = adapter.load(expected, following)
        system_pipeline._validate_batch_identity(batch, expected)
        torch.cuda.synchronize(device)
        for input_index, tensor in enumerate(batch.inputs):
            input_parts[input_index].append(tensor.detach().clone())
        label_parts.append(batch.labels.detach().clone())
        id_parts.append(torch.tensor([sample["galp_image_id"] for sample in expected], device=device))
        print(f"CACHE name={name} batch={index + 1}/{len(sample_batches)}", flush=True)
    adapter.end_repeat()
    cached = {
        "image_ids": torch.cat(id_parts),
        "labels": torch.cat(label_parts),
        "inputs": tuple(torch.cat(parts) for parts in input_parts),
    }
    del adapter
    torch.cuda.empty_cache()
    return cached


def _evaluate_cached(
    model: torch.nn.Module,
    cached: dict[str, Any],
    batch_size: int,
) -> dict[str, float | int]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    correct1 = 0
    correct5 = 0
    loss_sum = 0.0
    count = int(cached["labels"].shape[0])
    with torch.inference_mode():
        for offset in range(0, count, batch_size):
            inputs = tuple(tensor[offset : offset + batch_size] for tensor in cached["inputs"])
            labels = cached["labels"][offset : offset + batch_size]
            logits = model(*inputs)
            loss_sum += float(criterion(logits, labels).item())
            top5 = logits.topk(5, dim=1).indices
            matches = top5.eq(labels.reshape(-1, 1))
            correct1 += int(matches[:, :1].sum().item())
            correct5 += int(matches.sum().item())
    return {
        "samples": count,
        "loss": loss_sum / count,
        "accuracy_top1": correct1 / count,
        "accuracy_top5": correct5 / count,
        "correct_top1": correct1,
        "correct_top5": correct5,
    }


def _training_batches(
    condition: str,
    train_ids: Sequence[int],
    contiguous_batches: Sequence[Sequence[int]],
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    epoch_seed = int(seed + epoch * 1_000_003)
    if condition == "contiguous":
        batches = [list(batch) for batch in contiguous_batches]
        random.Random(epoch_seed).shuffle(batches)
        return batches
    if condition == "paired_scattered":
        order = list(train_ids)
        random.Random(epoch_seed).shuffle(order)
        return [order[offset : offset + batch_size] for offset in range(0, len(order), batch_size)]
    raise ValueError(condition)


def _run_training_probe(
    *,
    args: argparse.Namespace,
    selection: dict[str, Any],
    labels: Sequence[int],
    output_dir: Path,
    system_pipeline: Any,
) -> dict[str, Any]:
    device = torch.device(args.device)
    contract = _contract(args)
    train_batches = selection["training_probe"]["train_contiguous_batches"]
    train_ids = selection["training_probe"]["train_image_ids"]
    evaluation_ids = selection["training_probe"]["evaluation_image_ids"]
    evaluation_batches = [
        evaluation_ids[offset : offset + args.batch_size]
        for offset in range(0, len(evaluation_ids), args.batch_size)
    ]
    print("TRAINING_PROBE caching transformed inputs", flush=True)
    train_cache = _load_cached_inputs(
        name="train_contiguous_cohort",
        image_batches=train_batches,
        labels=labels,
        contract=contract,
        device=device,
        system_pipeline=system_pipeline,
    )
    evaluation_cache = _load_cached_inputs(
        name="representative_evaluation",
        image_batches=evaluation_batches,
        labels=labels,
        contract=contract,
        device=device,
        system_pipeline=system_pipeline,
    )
    id_to_row = {
        int(image_id): index for index, image_id in enumerate(train_cache["image_ids"].detach().cpu().tolist())
    }
    payload: dict[str, Any] = {
        "schema_version": "galp_image_order_training_probe_v1",
        "seed": args.seed,
        "seeds": [args.seed + offset for offset in range(args.training_seeds)],
        "epochs": args.training_epochs,
        "optimizer": "AdamW",
        "learning_rate": args.training_lr,
        "weight_decay": args.training_weight_decay,
        "gradient_clip_norm": 1.0,
        "train_samples": len(train_ids),
        "evaluation_samples": len(evaluation_ids),
        "practical_significance_threshold_percentage_points": args.practical_significance_pp,
        "scope": (
            "short checkpoint fine-tuning order-sensitivity probe on fixed preprocessed ImageNet samples; "
            "no augmentation or mixup; not a full 300-epoch ImageNet retraining claim"
        ),
        "runs": [],
    }
    baseline_model = system_pipeline._build_dct_model(contract, device)
    payload["baseline_evaluation"] = _evaluate_cached(baseline_model, evaluation_cache, args.batch_size)
    del baseline_model
    torch.cuda.empty_cache()
    _write_json(output_dir / "training_probe.json", payload)

    for seed in payload["seeds"]:
        for condition in ("paired_scattered", "contiguous"):
            print(f"TRAINING_PROBE seed={seed} condition={condition}", flush=True)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = system_pipeline._build_dct_model(contract, device)
            model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.training_lr, weight_decay=args.training_weight_decay, eps=1e-8
            )
            criterion = torch.nn.CrossEntropyLoss()
            epochs: list[dict[str, Any]] = []
            for epoch in range(args.training_epochs):
                model.train()
                batch_orders = _training_batches(
                    condition,
                    train_ids,
                    train_batches,
                    args.batch_size,
                    seed,
                    epoch,
                )
                loss_sum = 0.0
                correct = 0
                seen = 0
                for batch_ids in batch_orders:
                    rows = torch.tensor([id_to_row[int(image_id)] for image_id in batch_ids], device=device)
                    inputs = tuple(tensor.index_select(0, rows) for tensor in train_cache["inputs"])
                    batch_labels = train_cache["labels"].index_select(0, rows)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(*inputs)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    loss_sum += float(loss.detach().item()) * len(batch_ids)
                    correct += int((logits.detach().argmax(dim=1) == batch_labels).sum().item())
                    seen += len(batch_ids)
                torch.cuda.synchronize(device)
                evaluation = _evaluate_cached(model, evaluation_cache, args.batch_size)
                epoch_record = {
                    "epoch": epoch,
                    "train_loss": loss_sum / seen,
                    "train_online_accuracy_top1": correct / seen,
                    "evaluation": evaluation,
                }
                epochs.append(epoch_record)
                print(
                    f"TRAINING_RESULT seed={seed} condition={condition} epoch={epoch} "
                    f"eval_top1={evaluation['accuracy_top1']:.6f} eval_loss={evaluation['loss']:.6f}",
                    flush=True,
                )
            run = {
                "seed": seed,
                "condition": condition,
                "epochs": epochs,
                "final_evaluation": epochs[-1]["evaluation"],
            }
            payload["runs"].append(run)
            _write_json(output_dir / "training_probe.json", payload)
            del optimizer
            del model
            torch.cuda.empty_cache()
    del train_cache
    del evaluation_cache
    torch.cuda.empty_cache()
    return payload


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "diff_sha256": canonical_sha256(git("diff", "--binary")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        args.galp_manifest,
        args.label_map_json,
        args.data_root,
        args.index_csv,
        args.train_index_csv,
        args.rgbnomore_root,
        args.checkpoint,
        args.torch_binding_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.repeats <= 0 or args.training_seeds <= 0 or args.training_epochs <= 0:
        raise ValueError("repeat/training dimensions must be positive")
    galp_dct, system_manifest, system_pipeline = _load_runtime_modules(args.torch_binding_dir)
    layout = parse_galp_layout(args.galp_manifest)
    labels, label_payload = _load_labels(args.label_map_json, layout.image_count)
    source_entries = system_manifest.collect_dataset(args.data_root, args.split, args.index_csv)
    if len(source_entries) != layout.image_count:
        raise RuntimeError(
            f"source dataset count {len(source_entries)} does not match GALP image count {layout.image_count}"
        )
    source_eligible_ids = [
        int(entry["galp_image_id"])
        for entry in source_entries
        if entry["jpeg_sampling"] in SUPPORTED_SAMPLING
    ]
    source_sampling_counts: dict[str, int] = {}
    for entry in source_entries:
        mode = str(entry["jpeg_sampling"])
        source_sampling_counts[mode] = source_sampling_counts.get(mode, 0) + 1
        image_id = int(entry["galp_image_id"])
        if int(entry["label"]) != labels[image_id]:
            raise RuntimeError(f"source/GALP label mismatch at image ID {image_id}")
    reader = galp_dct.DirectDctReader(str(args.galp_manifest.resolve()))
    galp_sampling_counts: dict[str, int] = {}
    sampling_mismatches: list[dict[str, Any]] = []
    for image_id in range(layout.image_count):
        mode = _sampling_mode(reader.image_metadata(image_id))
        galp_sampling_counts[mode] = galp_sampling_counts.get(mode, 0) + 1
        source_mode = str(source_entries[image_id]["jpeg_sampling"])
        if mode != source_mode:
            sampling_mismatches.append(
                {
                    "galp_image_id": image_id,
                    "sample_id": source_entries[image_id]["sample_id"],
                    "source_jpeg_sampling": source_mode,
                    "galp_metadata_sampling": mode,
                }
            )
    del reader
    if len(source_eligible_ids) != args.expected_eligible_images:
        raise RuntimeError(
            f"eligible image count changed: expected {args.expected_eligible_images}, observed {len(source_eligible_ids)}"
        )
    selection = build_condition_orders(
        layout=layout,
        eligible_ids=source_eligible_ids,
        labels=labels,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        measurement_batches=args.measurement_batches,
        seed=args.seed,
    )
    selection["source_sampling_counts"] = source_sampling_counts
    selection["galp_metadata_sampling_counts"] = galp_sampling_counts
    selection["source_galp_sampling_mismatches"] = sampling_mismatches
    selection["galp_manifest"] = str(args.galp_manifest.resolve())
    selection["label_map_json"] = str(args.label_map_json.resolve())
    _write_json(output_dir / "selection.json", selection)
    training_order = analyze_training_index(args.train_index_csv, args.batch_size, args.seed)
    _write_json(output_dir / "training_order_stats.json", training_order)
    contract = {
        "schema_version": "galp_image_order_contract_v1",
        "created_unix_seconds": time.time(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_metadata": (
                {
                    "name": torch.cuda.get_device_name(torch.device(args.device)),
                    "capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
                    "total_memory_bytes": torch.cuda.get_device_properties(torch.device(args.device)).total_memory,
                }
                if torch.cuda.is_available() and torch.device(args.device).type == "cuda"
                else None
            ),
        },
        "files": {
            "galp_manifest": {"path": str(args.galp_manifest.resolve()), "sha256": _sha256_file(args.galp_manifest)},
            "label_map": {"path": str(args.label_map_json.resolve()), "sha256": _sha256_file(args.label_map_json)},
            "source_index_csv": {"path": str(args.index_csv.resolve()), "sha256": _sha256_file(args.index_csv)},
            "training_index_csv": {
                "path": str(args.train_index_csv.resolve()),
                "sha256": training_order["source_index_sha256"],
            },
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha256_file(args.checkpoint)},
        },
        "label_map_format": label_payload["format"],
        "selection_sha256": canonical_sha256(selection),
        "execution": {
            key: getattr(args, key)
            for key in (
                "batch_size",
                "warmup_batches",
                "measurement_batches",
                "repeats",
                "seed",
                "precision",
                "cache_capacity_mib",
                "decode_batch_rowgroups",
                "rowgroup_prefetch_depth",
                "rowgroup_prefetch_workers",
                "training_seeds",
                "training_epochs",
                "training_lr",
                "training_weight_decay",
            )
        },
        "git": _git_metadata(),
    }
    _write_json(output_dir / "contract.json", contract)
    print(
        f"SELECTION eligible={len(source_eligible_ids)} aligned_batches={selection['aligned_candidate_batch_count']} "
        f"selection_sha256={contract['selection_sha256']}",
        flush=True,
    )
    if args.selection_only:
        return {"contract": contract, "selection": selection}

    performance, semantic_paths = _run_performance(
        args=args,
        selection=selection,
        labels=labels,
        output_dir=output_dir,
        galp_dct=galp_dct,
        system_pipeline=system_pipeline,
    )
    training = None
    if not args.skip_training:
        training = _run_training_probe(
            args=args,
            selection=selection,
            labels=labels,
            output_dir=output_dir,
            system_pipeline=system_pipeline,
        )
    summary = summarize(
        selection=selection,
        performance=performance,
        semantic_paths=semantic_paths,
        training=training,
        training_order=training_order,
    )
    write_summary_artifacts(output_dir, summary, performance)
    print(f"SUMMARY {output_dir / 'report.md'}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--galp-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--label-map-json", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--train-index-csv", type=Path, default=DEFAULT_TRAIN_INDEX_CSV)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_BINDING_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "amp_fp16", "amp_bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measurement-batches", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--expected-eligible-images", type=int, default=48615)
    parser.add_argument("--cache-capacity-mib", type=int, default=1024)
    parser.add_argument("--decode-batch-rowgroups", type=int, default=2)
    parser.add_argument("--rowgroup-prefetch-depth", type=int, default=16)
    parser.add_argument("--rowgroup-prefetch-workers", type=int, default=4)
    parser.add_argument("--training-seeds", type=int, default=3)
    parser.add_argument("--training-epochs", type=int, default=3)
    parser.add_argument("--training-lr", type=float, default=1e-5)
    parser.add_argument("--training-weight-decay", type=float, default=1e-4)
    parser.add_argument("--practical-significance-pp", type=float, default=0.5)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
