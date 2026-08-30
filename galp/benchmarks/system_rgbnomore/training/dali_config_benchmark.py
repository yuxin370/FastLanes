#!/usr/bin/env python3
"""Screen DALI training configurations with the published RGB model workload.

This is a bounded engineering benchmark, not the registered full-epoch result.
It uses a prefix of the canonical full-dataset epoch order, preserves microbatch
64 and accumulation 16, and measures only after model and input-pipeline warmup.
The winning configurations must still be run with
``equal_image_epoch_benchmark.py`` for the scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
FASTLANES_ROOT = HERE.parents[3]

from galp.benchmarks.system_dct_major.training_pls.published_optimizer import (
    build_published_optimizer,
)
from galp.benchmarks.system_dct_major.training_pls.recipe import (
    RECIPE_NAME,
    recipe_contract,
)
from galp.benchmarks.system_dct_major.training_pls.train import (
    compile_published_model,
)
from galp.benchmarks.system_rgbnomore.training.augmentation import (
    derive_augmentation,
)
from galp.benchmarks.system_rgbnomore.training.model_factory import (
    build_model,
    seed_everything,
)
from galp.benchmarks.system_rgbnomore.training.pipeline import (
    DALI_VARIANTS,
    DaliTrainingAdapter,
    TrainingSample,
    resolve_dali_variant,
)
from galp.benchmarks.system_rgbnomore.training.sample_order import (
    SampleIdentity,
    canonical_epoch_order,
)


MICROBATCH_IMAGES = 64
GRADIENT_ACCUMULATION = 16


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _sample_id(raw: Mapping[str, Any]) -> str:
    value = str(raw.get("logical_sample_id", raw.get("sample_id", "")))
    if not value:
        raise ValueError("training manifest sample lacks logical_sample_id")
    return value


def _load_canonical_subset(
    manifest: Path,
    *,
    sample_count: int,
    seed: int,
    epoch: int,
) -> tuple[list[TrainingSample], list[SampleIdentity], str, int]:
    payload = json.loads(manifest.resolve().read_text(encoding="utf-8"))
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < sample_count:
        raise ValueError("training manifest is smaller than the requested workset")
    raw_by_id = {_sample_id(raw): raw for raw in raw_samples}
    if len(raw_by_id) != len(raw_samples):
        raise ValueError("training manifest contains duplicate logical IDs")
    full_order = canonical_epoch_order(list(raw_by_id), seed, epoch)
    identities = full_order[:sample_count]
    samples: list[TrainingSample] = []
    source_bytes = 0
    for identity in identities:
        raw = raw_by_id[identity.logical_sample_id]
        raw_path = Path(str(raw["path"]))
        path = (
            raw_path
            if raw_path.is_absolute()
            else manifest.resolve().parent / raw_path
        ).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_bytes += path.stat().st_size
        samples.append(
            TrainingSample(
                logical_sample_id=identity.logical_sample_id,
                path=path,
                label=int(raw["label"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
            )
        )
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(
            f"{identity.epoch}:{identity.position}:{identity.logical_sample_id}\n".encode(
                "utf-8"
            )
        )
    return samples, identities, digest.hexdigest(), source_bytes


def _metric_delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "encoded_source_bytes",
        "source_samples_read",
        "consumed_samples",
        "reader_overread_samples",
        "read_work_seconds",
        "pipeline_run_wait_seconds",
        "torch_handoff_host_seconds",
    ):
        result[name] = float(after.get(name, 0)) - float(before.get(name, 0))
        if name.endswith("bytes") or name.endswith("samples"):
            result[name] = int(result[name])
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name-substring", default="4090")
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--epoch-index", type=int, default=1)
    parser.add_argument("--warmup-microbatches", type=int, default=128)
    parser.add_argument("--measurement-microbatches", type=int, default=512)
    parser.add_argument(
        "--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more")
    )
    parser.add_argument("--dali-variant", choices=DALI_VARIANTS, default="d2")
    parser.add_argument("--dali-num-threads", type=int, default=4)
    parser.add_argument("--dali-prefetch-depth", type=int, default=2)
    parser.add_argument(
        "--dali-hybrid-huffman-threshold", type=int, default=1_000_000
    )
    parser.add_argument("--dali-hw-decoder-load", type=float, default=0.65)
    parser.add_argument("--dali-reader-initial-fill", type=int, default=1024)
    parser.add_argument(
        "--dali-reader-dont-use-mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dali-reader-read-ahead",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    for name in (
        "warmup_microbatches",
        "measurement_microbatches",
        "dali_num_threads",
        "dali_prefetch_depth",
        "dali_reader_initial_fill",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_microbatches % GRADIENT_ACCUMULATION:
        raise ValueError("warmup microbatches must align to gradient accumulation")
    if args.measurement_microbatches % GRADIENT_ACCUMULATION:
        raise ValueError("measurement microbatches must align to gradient accumulation")
    if args.dali_hybrid_huffman_threshold < 0:
        raise ValueError("DALI Huffman threshold must be non-negative")
    if not 0.0 <= args.dali_hw_decoder_load <= 1.0:
        raise ValueError("DALI hardware decoder load must be in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (
        args.required_gpu_name_substring
        and args.required_gpu_name_substring.lower() not in properties.name.lower()
    ):
        raise RuntimeError(
            f"selected GPU is {properties.name!r}, expected "
            f"{args.required_gpu_name_substring!r}"
        )

    total_microbatches = args.warmup_microbatches + args.measurement_microbatches
    sample_count = total_microbatches * MICROBATCH_IMAGES
    load_started = time.perf_counter()
    samples, identities, order_hash, selected_source_bytes = _load_canonical_subset(
        args.train_manifest,
        sample_count=sample_count,
        seed=args.seed,
        epoch=args.epoch_index,
    )
    manifest_plan_seconds = time.perf_counter() - load_started
    decisions_started = time.perf_counter()
    variant = resolve_dali_variant(args.dali_variant)
    decisions = (
        []
        if variant["augmentation_mode"] == "native"
        else [
            derive_augmentation(
                seed=args.seed,
                epoch=args.epoch_index,
                logical_sample_id=sample.logical_sample_id,
                source_width=sample.width,
                source_height=sample.height,
                domain="rgb",
            )
            for sample in samples
        ]
    )
    augmentation_plan_seconds = time.perf_counter() - decisions_started

    dali_config = {
        **variant,
        "num_threads": args.dali_num_threads,
        "prefetch_queue_depth": args.dali_prefetch_depth,
        "hybrid_huffman_threshold": args.dali_hybrid_huffman_threshold,
        "hw_decoder_load": args.dali_hw_decoder_load,
        "reader_initial_fill": args.dali_reader_initial_fill,
        "reader_dont_use_mmap": args.dali_reader_dont_use_mmap,
        "reader_read_ahead": args.dali_reader_read_ahead,
        "seed": args.seed,
    }
    adapter_started = time.perf_counter()
    adapter = DaliTrainingAdapter(
        samples,
        batch_size=MICROBATCH_IMAGES,
        workers=args.dali_num_threads,
        device=device,
        config={"phase": "train", "dali": dali_config},
    )
    adapter.begin(
        identities,
        decisions,
        [MICROBATCH_IMAGES] * total_microbatches,
    )
    adapter_setup_seconds = time.perf_counter() - adapter_started

    seed_everything(args.seed)
    model_started = time.perf_counter()
    model = build_model(args.rgbnomore_root, "rgb", device)
    published = recipe_contract(RECIPE_NAME)
    optimizer, weight_decayer, scheduler = build_published_optimizer(
        model,
        learning_rate=float(published["optimizer"]["learning_rate"]),
        weight_decay=float(published["optimizer"]["weight_decay"]["coefficient"]),
        warmup_updates=int(published["scheduler"]["warmup_optimizer_updates"]),
        total_updates=375_600,
    )
    execution_model = compile_published_model(model, published)
    model_setup_seconds = time.perf_counter() - model_started

    emitted = bytearray(sample_count)
    measured_stage_seconds: dict[str, float] = {}
    measured_loss: torch.Tensor | None = None
    measurement_started = 0.0
    loader_before: dict[str, Any] = {}
    try:
        for microbatch in range(total_microbatches):
            if microbatch % GRADIENT_ACCUMULATION == 0:
                optimizer.zero_grad(set_to_none=True)
                learning_rate = scheduler.prepare_next_update()
            if microbatch == args.warmup_microbatches:
                torch.cuda.synchronize(device)
                loader_before = adapter.loader_metrics()
                measurement_started = time.perf_counter()
                torch.cuda.nvtx.range_push(f"dali-{args.dali_variant}-measured")
            batch = adapter.next_batch()
            for identity in batch.identities:
                if not 0 <= identity.position < sample_count:
                    raise RuntimeError("DALI emitted an out-of-range sample position")
                if emitted[identity.position]:
                    raise RuntimeError("DALI emitted a duplicate sample position")
                emitted[identity.position] = 1
            logits = execution_model(*batch.inputs)
            loss = torch.nn.functional.cross_entropy(logits, batch.labels)
            (loss / GRADIENT_ACCUMULATION).backward()
            measured_loss = loss.detach()
            if microbatch >= args.warmup_microbatches:
                for name, value in batch.stage_seconds.items():
                    measured_stage_seconds[name] = (
                        measured_stage_seconds.get(name, 0.0) + float(value)
                    )
            if (microbatch + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(
                        published["optimizer"]["gradient_clipping_norm"]
                    ),
                )
                optimizer.step()
                weight_decayer.step(learning_rate)
                scheduler.complete_update()
            del batch, logits, loss
        torch.cuda.synchronize(device)
        measured_seconds = time.perf_counter() - measurement_started
        torch.cuda.nvtx.range_pop()
        loader_after = adapter.loader_metrics()
    finally:
        adapter.close()

    if emitted.count(1) != sample_count:
        raise RuntimeError(
            f"DALI covered {emitted.count(1)}/{sample_count} unique sample positions"
        )
    measured_images = args.measurement_microbatches * MICROBATCH_IMAGES
    result = {
        "schema_version": "galp-dali-training-config-screen-v2",
        "variant": args.dali_variant,
        "interpretation": (
            "bounded steady-state engineering screen; full epoch runner remains authoritative"
        ),
        "environment": {
            "hostname": platform.node(),
            "gpu_name": properties.name,
            "gpu_uuid": (
                None
                if getattr(properties, "uuid", None) is None
                else str(properties.uuid)
            ),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
        },
        "config": dali_config,
        "workload": {
            "seed": args.seed,
            "epoch_index": args.epoch_index,
            "microbatch_images": MICROBATCH_IMAGES,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "warmup_microbatches": args.warmup_microbatches,
            "measurement_microbatches": args.measurement_microbatches,
            "measurement_images": measured_images,
            "selected_samples": sample_count,
            "unique_positions": emitted.count(1),
            "canonical_prefix_sha256": order_hash,
            "selected_source_bytes": selected_source_bytes,
            "preserves_canonical_order": adapter.preserves_canonical_order(),
        },
        "setup_seconds": {
            "manifest_and_order": manifest_plan_seconds,
            "augmentation_plan": augmentation_plan_seconds,
            "adapter_and_pipeline": adapter_setup_seconds,
            "model_optimizer_compile_wrapper": model_setup_seconds,
        },
        "measurement": {
            "seconds": measured_seconds,
            "images_per_second": measured_images / measured_seconds,
            "loader_stage_work_seconds": measured_stage_seconds,
            "loader_metrics_delta": _metric_delta(loader_after, loader_before),
            "last_loss": (
                None if measured_loss is None else float(measured_loss.cpu().item())
            ),
        },
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
