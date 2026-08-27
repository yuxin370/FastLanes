#!/usr/bin/env python3
"""Matched adapter-only and short fine-tuning for DCT retokenization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from galp.benchmarks.system_dct_major.training_pls.published_augmentation import (
    apply_published_mixup,
    apply_published_randaugment,
    published_training_augmentation,
)
from galp.benchmarks.system_dct_major.training_pls.published_optimizer import (
    IndependentWeightDecay,
    PublishedUpdateScheduler,
)
from galp.experiments.dct_retokenization.data import (
    MaskedDctPreprocessor,
    epoch_permutation,
    load_manifest,
    make_loader,
)
from galp.experiments.dct_retokenization.model_wrapper import (
    DctRetokenizationWrapper,
    build_pretrained_dct_model,
    configure_trainable_parameters,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RGBNOMORE_ROOT = Path(os.environ.get("RGBNOMORE_ROOT", "RGB-no-more")).expanduser()
DEFAULT_CHECKPOINT = DEFAULT_RGBNOMORE_ROOT / "checkpoints/imgnetDCTViTTi_ep300_75.1.pth"
DEFAULT_TRAIN_MANIFEST = (
    REPO_ROOT
    / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json"
)


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


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def hardware_snapshot() -> dict[str, Any]:
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


def total_updates(sample_count: int, batch_size: int, accumulation: int, epochs: int) -> int:
    microbatches = math.ceil(sample_count / batch_size)
    return math.ceil(microbatches / accumulation) * epochs


def build_optimizer_and_schedule(
    wrapper: DctRetokenizationWrapper,
    config: dict[str, Any],
    updates: int,
    *,
    smoke_adjust_warmup: bool,
) -> tuple[torch.optim.AdamW, IndependentWeightDecay, PublishedUpdateScheduler, int]:
    trainable = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=0.0,
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["epsilon"]),
    )
    decayer = IndependentWeightDecay(
        wrapper,
        base_learning_rate=float(config["learning_rate"]),
        coefficient=float(config["weight_decay"]),
    )
    warmup = int(config["warmup_updates"])
    if smoke_adjust_warmup and warmup >= updates:
        warmup = max(1, updates // 10)
    scheduler = PublishedUpdateScheduler(
        optimizer=optimizer,
        base_learning_rate=float(config["learning_rate"]),
        warmup_updates=warmup,
        total_updates=updates,
    )
    return optimizer, decayer, scheduler, warmup


def checkpoint_payload(
    *,
    wrapper: DctRetokenizationWrapper,
    optimizer: torch.optim.Optimizer,
    decayer: IndependentWeightDecay,
    scheduler: PublishedUpdateScheduler,
    completed_epoch: int,
    resume_epoch: int,
    next_microbatch_index: int,
    epoch_loss_sum: float,
    epoch_samples: int,
    epoch_elapsed_seconds: float,
    processed_images: int,
    signature: dict[str, Any],
    trainable_names: Sequence[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "format": "dct-retokenization-training-checkpoint-v1",
        "experiment_state": wrapper.experiment_state(),
        "training_signature": signature,
        "trainable_parameter_names": list(trainable_names),
        "model_state_dict": wrapper.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "weight_decay_state_dict": decayer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "completed_epoch": int(completed_epoch),
        "resume_epoch": int(resume_epoch),
        "next_microbatch_index": int(next_microbatch_index),
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_samples": int(epoch_samples),
        "epoch_elapsed_seconds": float(epoch_elapsed_seconds),
        "processed_images": int(processed_images),
        "elapsed_seconds": float(elapsed_seconds),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_config(path: Path, expected_stage: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "dct-retokenization-training-config-v1":
        raise ValueError("unexpected training config schema")
    if payload.get("stage") != expected_stage:
        raise ValueError(f"config stage {payload.get('stage')!r} does not match {expected_stage!r}")
    if int(payload["physical_batch_size"]) * int(payload["gradient_accumulation"]) != int(
        payload["effective_batch_size"]
    ):
        raise ValueError("physical batch and accumulation do not match effective batch")
    return payload


def run(args: argparse.Namespace, expected_stage: str) -> int:
    config = parse_config(args.config.resolve(), expected_stage)
    if args.epochs is not None:
        config = dict(config)
        config["epochs"] = int(args.epochs)
    _, samples = load_manifest(args.train_manifest.resolve(), max_samples=args.max_train_samples)
    output_dir = args.output_dir.resolve()
    latest = output_dir / "latest.pt"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite non-empty training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if args.expected_device_name and device_name != args.expected_device_name:
        raise RuntimeError(f"expected {args.expected_device_name!r}, observed {device_name!r}")
    torch.set_num_threads(args.torch_cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.cuda.manual_seed_all(seed)

    base = build_pretrained_dct_model(
        args.rgbnomore_root.resolve(), args.checkpoint.resolve(), device
    )
    wrapper = DctRetokenizationWrapper(
        base,
        token_count=int(config["token_count"]),
        merge_axis=str(config["merge_axis"]),
        merge_type=str(config["merge_type"]),
    )
    trainable_names = configure_trainable_parameters(wrapper, expected_stage)
    trainable_parameters = sum(
        parameter.numel() for parameter in wrapper.parameters() if parameter.requires_grad
    )
    epochs = int(config["epochs"])
    batch_size = int(config["physical_batch_size"])
    accumulation = int(config["gradient_accumulation"])
    updates = total_updates(len(samples), batch_size, accumulation, epochs)
    optimizer, decayer, scheduler, resolved_warmup = build_optimizer_and_schedule(
        wrapper,
        config,
        updates,
        smoke_adjust_warmup=args.max_train_samples is not None,
    )
    preprocessor = MaskedDctPreprocessor(args.rgbnomore_root.resolve(), device, args.k)

    signature = {
        "schema": "dct-retokenization-training-run-v1",
        "stage": expected_stage,
        "k": int(args.k),
        "n": int(config["token_count"]),
        "merge_axis": config["merge_axis"],
        "merge_type": config["merge_type"],
        "config": config,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(args.train_manifest.resolve()),
        "sample_count": len(samples),
        "epochs": epochs,
        "total_updates": updates,
        "resolved_warmup_updates": resolved_warmup,
        "trainable_parameter_count": trainable_parameters,
        "trainable_parameter_names": trainable_names,
        "device_name": device_name,
        "mask_audit": preprocessor.audit_record(),
    }
    signature_path = output_dir / "run_signature.json"
    completed_epoch = 0
    resume_epoch = 0
    resume_microbatch_index = 0
    resume_epoch_loss = 0.0
    resume_epoch_samples = 0
    resume_epoch_elapsed = 0.0
    processed_images = 0
    elapsed_before = 0.0
    if args.resume:
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise ValueError("resume signature differs")
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        wrapper.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        decayer.load_state_dict(payload["weight_decay_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        completed_epoch = int(payload["completed_epoch"])
        resume_epoch = int(payload.get("resume_epoch", completed_epoch))
        resume_microbatch_index = int(payload.get("next_microbatch_index", 0))
        resume_epoch_loss = float(payload.get("epoch_loss_sum", 0.0))
        resume_epoch_samples = int(payload.get("epoch_samples", 0))
        resume_epoch_elapsed = float(payload.get("epoch_elapsed_seconds", 0.0))
        processed_images = int(payload["processed_images"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
    else:
        write_json(signature_path, signature)
        write_json(output_dir / "hardware_before.json", hardware_snapshot())

    wrapper.train()
    execution_model: torch.nn.Module = wrapper
    if not args.no_compile:
        execution_model = torch.compile(
            wrapper, backend="inductor", mode="default", fullgraph=False, dynamic=False
        )
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    running_loss = 0.0
    running_samples = 0
    last_log_update = scheduler.completed_updates

    for epoch in range(resume_epoch, epochs):
        epoch_started = time.perf_counter()
        permutation = epoch_permutation(len(samples), seed, epoch)
        full_microbatch_count = math.ceil(len(samples) / batch_size)
        first_microbatch = resume_microbatch_index if epoch == resume_epoch else 0
        if not 0 <= first_microbatch <= full_microbatch_count:
            raise ValueError(
                f"invalid resume microbatch {first_microbatch}/{full_microbatch_count}"
            )
        first_ordinal = first_microbatch * batch_size
        loader = make_loader(
            samples,
            args.rgbnomore_root.resolve(),
            indices=permutation[first_ordinal:],
            batch_size=batch_size,
            workers=args.workers,
        )
        epoch_loss = resume_epoch_loss if epoch == resume_epoch else 0.0
        epoch_samples = resume_epoch_samples if epoch == resume_epoch else 0
        epoch_elapsed_before = resume_epoch_elapsed if epoch == resume_epoch else 0.0
        optimizer.zero_grad(set_to_none=True)
        for local_microbatch_index, (yq, cq, quant, labels, ordinals) in enumerate(loader):
            microbatch_index = first_microbatch + local_microbatch_index
            batch = int(labels.shape[0])
            window_start = (microbatch_index // accumulation) * accumulation
            window_end = min(window_start + accumulation, full_microbatch_count)
            first_sample = window_start * batch_size
            final_sample = min(window_end * batch_size, len(samples))
            window_samples = final_sample - first_sample
            logical_ids = [samples[int(index)].logical_sample_id for index in ordinals.tolist()]
            decisions = [
                published_training_augmentation(
                    training_seed=seed,
                    epoch=epoch,
                    logical_sample_id=samples[int(index)].logical_sample_id,
                    virtual_pls_id=int(index) // 1024,
                    crop_policy="per-sample",
                    source_width=512,
                    source_height=512,
                ).decision
                for index in ordinals.tolist()
            ]
            inputs = preprocessor.training(yq, cq, quant, decisions)
            inputs, _records = apply_published_randaugment(
                inputs,
                training_seed=seed,
                epoch=epoch,
                logical_sample_ids=logical_ids,
                rgbnomore_root=args.rgbnomore_root.resolve(),
            )
            labels_device = labels.to(device, non_blocking=True)
            inputs, mixed_labels, _mixup = apply_published_mixup(
                inputs,
                labels_device,
                training_seed=seed,
                epoch=epoch,
                microbatch_index=microbatch_index,
                alpha=float(config["mixup_alpha"]),
                classes=1000,
            )
            logits = execution_model(*inputs)
            loss = torch.nn.functional.cross_entropy(logits, mixed_labels)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {microbatch_index}")
            (loss * (batch / window_samples)).backward()
            detached = float(loss.detach().item())
            epoch_loss += detached * batch
            epoch_samples += batch
            running_loss += detached * batch
            running_samples += batch
            processed_images += batch

            end_window = (
                (microbatch_index + 1) % accumulation == 0
                or microbatch_index + 1 == full_microbatch_count
            )
            if end_window:
                learning_rate = scheduler.prepare_next_update()
                gradients = [
                    parameter.grad
                    for parameter in wrapper.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients or not all(bool(torch.isfinite(value).all().item()) for value in gradients):
                    raise FloatingPointError("missing or non-finite trainable gradients")
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
                        float(config["gradient_clip_norm"]),
                    ).item()
                )
                optimizer.step()
                decayer.step(learning_rate)
                scheduler.complete_update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler.completed_updates % args.checkpoint_every_updates == 0:
                    elapsed = elapsed_before + time.perf_counter() - started
                    epoch_elapsed = epoch_elapsed_before + time.perf_counter() - epoch_started
                    payload = checkpoint_payload(
                        wrapper=wrapper,
                        optimizer=optimizer,
                        decayer=decayer,
                        scheduler=scheduler,
                        completed_epoch=epoch,
                        resume_epoch=epoch,
                        next_microbatch_index=microbatch_index + 1,
                        epoch_loss_sum=epoch_loss,
                        epoch_samples=epoch_samples,
                        epoch_elapsed_seconds=epoch_elapsed,
                        processed_images=processed_images,
                        signature=signature,
                        trainable_names=trainable_names,
                        elapsed_seconds=elapsed,
                    )
                    save_checkpoint(latest, payload)
                if scheduler.completed_updates % args.log_every_updates == 0:
                    elapsed = elapsed_before + time.perf_counter() - started
                    record = {
                        "record_type": "train_window",
                        "stage": expected_stage,
                        "k": int(args.k),
                        "epoch": epoch + 1,
                        "optimizer_update": scheduler.completed_updates,
                        "processed_images": processed_images,
                        "loss": running_loss / running_samples,
                        "learning_rate": learning_rate,
                        "gradient_norm_before_clip": gradient_norm,
                        "elapsed_seconds": elapsed,
                        "images_per_second": (processed_images / elapsed) if elapsed > 0 else 0.0,
                    }
                    append_jsonl(metrics_path, record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    running_loss = 0.0
                    running_samples = 0
                    last_log_update = scheduler.completed_updates
                    write_json(
                        output_dir / "progress.json",
                        {
                            "status": "running",
                            "completed_epoch": epoch,
                            "optimizer_update": scheduler.completed_updates,
                            "processed_images": processed_images,
                            "elapsed_seconds": elapsed,
                        },
                    )
            del inputs, labels_device, mixed_labels, logits, loss
        if epoch_samples != len(samples):
            raise RuntimeError(f"epoch consumed {epoch_samples}, expected {len(samples)}")
        epoch_seconds = epoch_elapsed_before + time.perf_counter() - epoch_started
        append_jsonl(
            metrics_path,
            {
                "record_type": "epoch",
                "stage": expected_stage,
                "k": int(args.k),
                "epoch": epoch + 1,
                "optimizer_update": scheduler.completed_updates,
                "processed_images": processed_images,
                "train_loss": epoch_loss / epoch_samples,
                "epoch_seconds": epoch_seconds,
                "images_per_second": epoch_samples / epoch_seconds,
            },
        )
        elapsed = elapsed_before + time.perf_counter() - started
        payload = checkpoint_payload(
            wrapper=wrapper,
            optimizer=optimizer,
            decayer=decayer,
            scheduler=scheduler,
            completed_epoch=epoch + 1,
            resume_epoch=epoch + 1,
            next_microbatch_index=0,
            epoch_loss_sum=0.0,
            epoch_samples=0,
            epoch_elapsed_seconds=0.0,
            processed_images=processed_images,
            signature=signature,
            trainable_names=trainable_names,
            elapsed_seconds=elapsed,
        )
        save_checkpoint(latest, payload)
        if epoch + 1 in {int(value) for value in config["checkpoint_epochs"]} or epoch + 1 == epochs:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt", payload)
        print(
            f"epoch {epoch+1}/{epochs} complete: loss={epoch_loss/epoch_samples:.6f}, "
            f"images/s={epoch_samples/epoch_seconds:.2f}",
            flush=True,
        )
        resume_microbatch_index = 0
        resume_epoch_loss = 0.0
        resume_epoch_samples = 0
        resume_epoch_elapsed = 0.0

    torch.cuda.synchronize(device)
    elapsed = elapsed_before + time.perf_counter() - started
    metadata = {
        "status": "complete",
        "stage": expected_stage,
        "k": int(args.k),
        "completed_epochs": epochs,
        "optimizer_updates": scheduler.completed_updates,
        "processed_images": processed_images,
        "elapsed_seconds": elapsed,
        "trainable_parameter_count": trainable_parameters,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_name": device_name,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    write_json(output_dir / "hardware_after.json", hardware_snapshot())
    write_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "completed_epoch": epochs,
            "optimizer_update": scheduler.completed_updates,
            "processed_images": processed_images,
        },
    )
    print(f"training complete in {elapsed:.1f}s: {output_dir}", flush=True)
    return 0


def parse_args(expected_stage: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_config = HERE / (
        "configs/adapter_only.json" if expected_stage == "adapter" else "configs/short_finetune.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--k", type=int, choices=(64, 32), required=True)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-name")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    parser.add_argument("--log-every-updates", type=int, default=100)
    parser.add_argument("--checkpoint-every-updates", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    if (
        args.workers < 0
        or args.torch_cpu_threads <= 0
        or args.log_every_updates <= 0
        or args.checkpoint_every_updates <= 0
    ):
        parser.error("worker/thread/log values are invalid")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    return args


def main(expected_stage: str, argv: Sequence[str] | None = None) -> int:
    return run(parse_args(expected_stage, argv), expected_stage)
