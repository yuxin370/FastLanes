#!/usr/bin/env python3
"""Conservative staged adaptation for input-level and delayed K32/N98 models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from galp.experiments.dct_retokenization.data import (
    MaskedDctPreprocessor,
    epoch_permutation,
    load_manifest,
    make_loader,
)
from galp.experiments.dct_retokenization.delayed_wrapper import (
    DelayedRetokenizationWrapper,
    load_delayed_checkpoint,
)
from galp.experiments.dct_retokenization.model_wrapper import (
    DctRetokenizationWrapper,
    build_pretrained_dct_model,
    load_experiment_checkpoint,
)
from galp.experiments.dct_retokenization.train_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_RGBNOMORE_ROOT,
    DEFAULT_TRAIN_MANIFEST,
    REPO_ROOT,
    append_jsonl,
    apply_published_mixup,
    apply_published_randaugment,
    hardware_snapshot,
    published_training_augmentation,
    save_checkpoint,
    sha256_file,
    total_updates,
    write_json,
)


HERE = Path(__file__).resolve().parent
DEFAULT_VAL_MANIFEST = (
    REPO_ROOT / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
)


class DifferentialWarmupCosine:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_learning_rates: Sequence[float],
        warmup_updates: int,
        total_update_count: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_learning_rates = [float(value) for value in base_learning_rates]
        self.warmup_updates = int(warmup_updates)
        self.total_update_count = int(total_update_count)
        self.completed_updates = 0
        if len(self.base_learning_rates) != len(self.optimizer.param_groups):
            raise ValueError("one base LR is required for every optimizer group")
        if not 0 <= self.warmup_updates < self.total_update_count:
            raise ValueError("warmup must be non-negative and shorter than the run")

    def multiplier(self, update_index: int) -> float:
        if self.warmup_updates and update_index <= self.warmup_updates:
            return update_index / self.warmup_updates
        progress = (update_index - self.warmup_updates) / max(
            1, self.total_update_count - self.warmup_updates
        )
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def prepare_next_update(self) -> list[float]:
        factor = self.multiplier(self.completed_updates + 1)
        values = [base * factor for base in self.base_learning_rates]
        for group, value in zip(self.optimizer.param_groups, values, strict=True):
            group["lr"] = value
        return values

    def complete_update(self) -> None:
        self.completed_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_learning_rates": self.base_learning_rates,
            "warmup_updates": self.warmup_updates,
            "total_update_count": self.total_update_count,
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        expected = {
            "base_learning_rates": self.base_learning_rates,
            "warmup_updates": self.warmup_updates,
            "total_update_count": self.total_update_count,
        }
        observed = {key: payload[key] for key in expected}
        if observed != expected:
            raise ValueError(f"scheduler contract changed: {observed} != {expected}")
        self.completed_updates = int(payload["completed_updates"])


def logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled batch-mean KL(teacher || student)."""

    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher logits must have identical shapes: "
            f"{tuple(student_logits.shape)} != {tuple(teacher_logits.shape)}"
        )
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
        torch.nn.functional.softmax(teacher_logits.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)


def configure_stage(
    wrapper: DctRetokenizationWrapper | DelayedRetokenizationWrapper,
    stage: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    retokenizer_group_name = (
        "delayed_retokenizer"
        if isinstance(wrapper, DelayedRetokenizationWrapper)
        else "superpatch_projection"
    )
    specifications: list[tuple[str, torch.nn.Module, float, float]] = [
        (
            retokenizer_group_name,
            wrapper.retokenizer,
            float(config["learning_rates"]["tokenizer"]),
            float(config["weight_decay"]["tokenizer"]),
        ),
        (
            "head",
            wrapper.base_model.classhead,
            float(config["learning_rates"]["head"]),
            float(config["weight_decay"]["head"]),
        ),
    ]
    if stage in {"stage_b", "stage_kd"}:
        block_count = int(config.get("unfrozen_last_blocks", 4))
        blocks = list(wrapper.base_model.encoder.children())
        if block_count != 4 or len(blocks) < block_count:
            raise ValueError("Stage B is intentionally restricted to the final four blocks")
        specifications.append(
            (
                "last_four_blocks",
                torch.nn.Sequential(*blocks[-block_count:]),
                float(config["learning_rates"]["last_blocks"]),
                float(config["weight_decay"]["last_blocks"]),
            )
        )
    elif stage == "delayed_stage_b":
        if not isinstance(wrapper, DelayedRetokenizationWrapper):
            raise TypeError("delayed_stage_b requires DelayedRetokenizationWrapper")
        blocks = wrapper.encoder_blocks()
        downstream = blocks[wrapper.merge_after_block :]
        expected_count = 12 - wrapper.merge_after_block
        if len(downstream) != expected_count:
            raise RuntimeError("unexpected delayed downstream block count")
        specifications.append(
            (
                "post_merge_blocks",
                torch.nn.Sequential(*downstream),
                float(config["learning_rates"]["post_merge_blocks"]),
                float(config["weight_decay"]["post_merge_blocks"]),
            )
        )
    elif stage not in {"stage_a", "delayed_stage_a"}:
        raise ValueError(
            "stage must be stage_a, stage_b, stage_kd, delayed_stage_a, "
            "or delayed_stage_b"
        )

    groups: list[dict[str, Any]] = []
    selected: set[int] = set()
    for group_name, module, learning_rate, weight_decay in specifications:
        parameters = list(module.parameters())
        for parameter in parameters:
            if id(parameter) in selected:
                raise RuntimeError(f"parameter appears in multiple groups: {group_name}")
            selected.add(id(parameter))
            parameter.requires_grad_(True)
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": group_name,
                "base_lr": learning_rate,
            }
        )
    names = [name for name, parameter in wrapper.named_parameters() if parameter.requires_grad]
    if sum(len(group["params"]) for group in groups) != len(
        [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    ):
        raise RuntimeError("optimizer grouping does not cover trainable parameters exactly")
    return groups, names


@torch.inference_mode()
def validate(
    wrapper: DctRetokenizationWrapper | DelayedRetokenizationWrapper,
    preprocessor: MaskedDctPreprocessor,
    samples: Sequence[Any],
    rgbnomore_root: Path,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    wrapper.eval()
    loader = make_loader(
        samples,
        rgbnomore_root,
        indices=None,
        batch_size=batch_size,
        workers=workers,
    )
    loss_sum = 0.0
    correct_top1 = 0
    correct_top5 = 0
    completed = 0
    started = time.perf_counter()
    for yq, cq, quant, labels, ordinals in loader:
        expected = list(range(completed, completed + int(labels.shape[0])))
        if [int(value) for value in ordinals.tolist()] != expected:
            raise RuntimeError("validation sample order changed")
        y, cbcr = preprocessor.validation(yq, cq, quant)
        labels_device = labels.to(device, non_blocking=True)
        logits = wrapper(y, cbcr)
        loss_sum += float(
            torch.nn.functional.cross_entropy(logits, labels_device, reduction="sum").item()
        )
        top5 = logits.topk(5, dim=1).indices
        correct_top1 += int((top5[:, 0] == labels_device).sum().item())
        correct_top5 += int((top5 == labels_device[:, None]).any(dim=1).sum().item())
        completed += int(labels.shape[0])
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    wrapper.train()
    return {
        "sample_count": completed,
        "top1_percent": 100.0 * correct_top1 / completed,
        "top5_percent": 100.0 * correct_top5 / completed,
        "cross_entropy": loss_sum / completed,
        "validation_seconds": elapsed,
        "validation_images_per_second": completed / elapsed,
    }


def checkpoint_payload(
    wrapper: DctRetokenizationWrapper | DelayedRetokenizationWrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: DifferentialWarmupCosine,
    signature: dict[str, Any],
    trainable_names: Sequence[str],
    *,
    completed_epoch: int,
    resume_epoch: int,
    next_microbatch_index: int,
    epoch_loss_sum: float,
    epoch_ce_loss_sum: float,
    epoch_kd_loss_sum: float,
    epoch_samples: int,
    epoch_elapsed_seconds: float,
    processed_images: int,
    elapsed_seconds: float,
    best_epoch: int,
    best_top1: float,
    pending_validation_epoch: int | None,
) -> dict[str, Any]:
    experiment_state = wrapper.experiment_state()
    return {
        "format": (
            "dct-delayed-retokenization-training-checkpoint-v1"
            if experiment_state.get("architecture") == "delayed_retokenization"
            else "dct-superpatch-training-checkpoint-v1"
        ),
        "experiment_state": experiment_state,
        "training_signature": signature,
        "trainable_parameter_names": list(trainable_names),
        "model_state_dict": wrapper.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "completed_epoch": int(completed_epoch),
        "resume_epoch": int(resume_epoch),
        "next_microbatch_index": int(next_microbatch_index),
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_ce_loss_sum": float(epoch_ce_loss_sum),
        "epoch_kd_loss_sum": float(epoch_kd_loss_sum),
        "epoch_samples": int(epoch_samples),
        "epoch_elapsed_seconds": float(epoch_elapsed_seconds),
        "processed_images": int(processed_images),
        "elapsed_seconds": float(elapsed_seconds),
        "best_epoch": int(best_epoch),
        "best_top1": float(best_top1),
        "pending_validation_epoch": (
            None
            if pending_validation_epoch is None
            else int(pending_validation_epoch)
        ),
    }


def checkpoint_pending_validation_epoch(
    payload: dict[str, Any], train_sample_count: int
) -> int | None:
    raw_epoch = payload.get("pending_validation_epoch")
    if raw_epoch is None:
        return None
    epoch = int(raw_epoch)
    if epoch <= 0 or epoch != int(payload["completed_epoch"]):
        raise ValueError("checkpoint pending validation epoch is inconsistent")
    if int(payload["resume_epoch"]) != epoch:
        raise ValueError("pending validation checkpoint must resume after that epoch")
    if int(payload["next_microbatch_index"]) != 0:
        raise ValueError("pending validation checkpoint cannot contain a partial epoch")
    if int(payload["epoch_samples"]) != int(train_sample_count):
        raise ValueError("pending validation checkpoint has incomplete epoch statistics")
    if float(payload.get("epoch_elapsed_seconds", -1.0)) < 0.0:
        raise ValueError("pending validation checkpoint has invalid epoch timing")
    return epoch


def append_epoch_metric_once(path: Path, row: dict[str, Any]) -> None:
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                existing = json.loads(line)
                if (
                    existing.get("record_type") == "epoch"
                    and int(existing.get("epoch", -1)) == int(row["epoch"])
                ):
                    return
    append_jsonl(path, {"record_type": "epoch", **row})


def write_history(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    stages_by_schema = {
        "dct-superpatch-training-config-v1": {"stage_a", "stage_b", "stage_kd"},
        "dct-delayed-retokenization-training-config-v1": {
            "delayed_stage_a",
            "delayed_stage_b",
        },
    }
    if schema not in stages_by_schema:
        raise ValueError("unexpected staged retokenization training config schema")
    if payload.get("stage") not in stages_by_schema[schema]:
        raise ValueError(f"config stage is incompatible with schema {schema!r}")
    if int(payload.get("k", -1)) != 32 or int(payload.get("token_count", -1)) != 98:
        raise ValueError("this stage is intentionally restricted to K32/N98")
    if int(payload["physical_batch_size"]) * int(payload["gradient_accumulation"]) != int(
        payload["effective_batch_size"]
    ):
        raise ValueError("physical and effective batch sizes disagree")
    if payload["stage"] == "stage_kd":
        distillation = payload.get("distillation") or {}
        if distillation.get("teacher") != "K32/N196 pretrained checkpoint":
            raise ValueError("Stage KD teacher must be K32/N196 pretrained checkpoint")
        if float(distillation.get("temperature", 0.0)) <= 0:
            raise ValueError("Stage KD temperature must be positive")
        if float(distillation.get("lambda", -1.0)) < 0:
            raise ValueError("Stage KD lambda must be non-negative")
        if bool(payload.get("randaugment", False)) or float(payload.get("mixup_alpha", 0.0)) != 0:
            raise ValueError("first-pass Stage KD intentionally excludes RandAugment and mixup")
    if schema == "dct-delayed-retokenization-training-config-v1":
        if int(payload.get("merge_after_block", -1)) not in {2, 4}:
            raise ValueError("delayed merge_after_block must be 2 or 4")
        if payload.get("merge_axis") != "width":
            raise ValueError("the bounded delayed study uses canonical width merge")
        if payload.get("initialization") not in {
            "average",
            "keep_first",
            "keep_second",
            "standard",
        }:
            raise ValueError("unsupported delayed retokenizer initialization")
    return payload


def signatures_differ_only_by_device(
    previous: dict[str, Any], current: dict[str, Any]
) -> bool:
    previous_without_device = dict(previous)
    current_without_device = dict(current)
    previous_device = previous_without_device.pop("device_name", None)
    current_device = current_without_device.pop("device_name", None)
    return (
        previous_device is not None
        and current_device is not None
        and previous_device != current_device
        and previous_without_device == current_without_device
    )


def record_device_migration(
    path: Path,
    *,
    previous_device: str,
    current_device: str,
    optimizer_update: int,
) -> None:
    if path.is_file():
        segments = json.loads(path.read_text(encoding="utf-8"))
    else:
        segments = [
            {
                "device_name": previous_device,
                "start_optimizer_update": 0,
                "end_optimizer_update": int(optimizer_update),
                "inferred_from_original_run_signature": True,
            }
        ]
    if segments and segments[-1].get("end_optimizer_update") is None:
        segments[-1]["end_optimizer_update"] = int(optimizer_update)
    segments.append(
        {
            "device_name": current_device,
            "start_optimizer_update": int(optimizer_update),
            "end_optimizer_update": None,
            "migration_recorded_at_unix_ns": time.time_ns(),
        }
    )
    write_json(path, segments)


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config.resolve())
    stage = str(config["stage"])
    delayed = config["schema"] == "dct-delayed-retokenization-training-config-v1"
    _, train_samples = load_manifest(
        args.train_manifest.resolve(), max_samples=args.max_train_samples
    )
    _, val_samples = load_manifest(args.val_manifest.resolve(), max_samples=args.max_val_samples)
    output_dir = args.output_dir.resolve()
    latest_path = output_dir / "latest.pt"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("super-patch training requires CUDA")
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
    if delayed:
        wrapper: DctRetokenizationWrapper | DelayedRetokenizationWrapper
        wrapper = DelayedRetokenizationWrapper(
            base,
            merge_after_block=int(config["merge_after_block"]),
            merge_axis=str(config["merge_axis"]),
            initialization=str(config["initialization"]),
        )
    else:
        wrapper = DctRetokenizationWrapper(
            base,
            token_count=98,
            merge_axis=str(config["merge_axis"]),
            merge_type="learned",
            architecture="superpatch",
            initialization=str(config["initialization"]),
            position_mode=str(config["position_mode"]),
        )
    initial_checkpoint = args.initial_checkpoint.resolve() if args.initial_checkpoint else None
    if stage in {"stage_b", "stage_kd", "delayed_stage_b"} and initial_checkpoint is None:
        source = "Stage A" if stage == "stage_b" else "Stage B"
        if stage == "delayed_stage_b":
            source = "Delayed Stage A"
        raise ValueError(f"{stage} must start from the {source} best checkpoint")
    if initial_checkpoint is not None:
        if delayed:
            load_delayed_checkpoint(wrapper, initial_checkpoint)
        else:
            load_experiment_checkpoint(wrapper, initial_checkpoint)

    teacher_checkpoint = (
        args.teacher_checkpoint.resolve()
        if args.teacher_checkpoint is not None
        else args.checkpoint.resolve()
    )
    teacher: torch.nn.Module | None = None
    if stage == "stage_kd":
        # The preprocessor below applies K=32 once.  The frozen original model
        # then consumes those exact inputs with N=196, so teacher/student differ
        # in token budget but not coefficient information or augmentation.
        teacher = build_pretrained_dct_model(
            args.rgbnomore_root.resolve(), teacher_checkpoint, device
        )
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    optimizer_groups, trainable_names = configure_stage(wrapper, stage, config)
    base_learning_rates = [float(group["base_lr"]) for group in optimizer_groups]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["epsilon"]),
    )
    epochs = int(config["epochs"])
    batch_size = int(config["physical_batch_size"])
    accumulation = int(config["gradient_accumulation"])
    update_count = total_updates(len(train_samples), batch_size, accumulation, epochs)
    warmup_updates = int(config["warmup_updates"])
    if args.max_train_samples is not None and warmup_updates >= update_count:
        warmup_updates = max(0, update_count // 10)
    scheduler = DifferentialWarmupCosine(
        optimizer, base_learning_rates, warmup_updates, update_count
    )
    preprocessor = MaskedDctPreprocessor(args.rgbnomore_root.resolve(), device, 32)

    signature = {
        "schema": (
            "dct-delayed-retokenization-training-run-v1"
            if delayed
            else "dct-superpatch-training-run-v1"
        ),
        "stage": stage,
        "config": config,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "initial_checkpoint": "" if initial_checkpoint is None else str(initial_checkpoint),
        "initial_checkpoint_sha256": "" if initial_checkpoint is None else sha256_file(initial_checkpoint),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "val_manifest_sha256": sha256_file(args.val_manifest.resolve()),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "total_updates": update_count,
        "resolved_warmup_updates": warmup_updates,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in wrapper.parameters() if parameter.requires_grad
        ),
        "trainable_parameter_names": trainable_names,
        "optimizer_groups": [
            {
                "group_name": group["group_name"],
                "base_lr": group["base_lr"],
                "weight_decay": group["weight_decay"],
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer_groups
        ],
        "device_name": device_name,
        "mask_audit": preprocessor.audit_record(),
    }
    if teacher is not None:
        signature["teacher"] = {
            "k": 32,
            "token_count": 196,
            "checkpoint": str(teacher_checkpoint),
            "checkpoint_sha256": sha256_file(teacher_checkpoint),
            "frozen": True,
        }
    signature_path = output_dir / "run_signature.json"
    active_signature_path = output_dir / "active_resume_signature.json"

    completed_epoch = 0
    resume_epoch = 0
    resume_microbatch = 0
    resume_epoch_loss = 0.0
    resume_epoch_ce_loss = 0.0
    resume_epoch_kd_loss = 0.0
    resume_epoch_samples = 0
    resume_epoch_elapsed = 0.0
    processed_images = 0
    elapsed_before = 0.0
    best_epoch = 0
    best_top1 = float("-inf")
    pending_validation_epoch: int | None = None
    history: list[dict[str, Any]] = []
    if args.resume:
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        comparison_path = (
            active_signature_path if active_signature_path.is_file() else signature_path
        )
        previous = json.loads(comparison_path.read_text(encoding="utf-8"))
        device_migration = previous != signature
        if device_migration and not (
            args.allow_device_migration
            and signatures_differ_only_by_device(previous, signature)
        ):
            raise ValueError(
                "resume signature differs; --allow-device-migration permits only device_name"
            )
        if device_migration:
            record_device_migration(
                output_dir / "hardware_segments.json",
                previous_device=str(previous["device_name"]),
                current_device=device_name,
                optimizer_update=int(payload["scheduler_state_dict"]["completed_updates"]),
            )
            write_json(active_signature_path, signature)
        wrapper.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        completed_epoch = int(payload["completed_epoch"])
        resume_epoch = int(payload["resume_epoch"])
        resume_microbatch = int(payload["next_microbatch_index"])
        resume_epoch_loss = float(payload["epoch_loss_sum"])
        resume_epoch_ce_loss = float(
            payload.get("epoch_ce_loss_sum", payload["epoch_loss_sum"])
        )
        resume_epoch_kd_loss = float(payload.get("epoch_kd_loss_sum", 0.0))
        resume_epoch_samples = int(payload["epoch_samples"])
        resume_epoch_elapsed = float(payload.get("epoch_elapsed_seconds", 0.0))
        processed_images = int(payload["processed_images"])
        elapsed_before = float(payload["elapsed_seconds"])
        best_epoch = int(payload["best_epoch"])
        best_top1 = float(payload["best_top1"])
        pending_validation_epoch = checkpoint_pending_validation_epoch(
            payload, len(train_samples)
        )
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        write_json(signature_path, signature)
        write_json(output_dir / "hardware_before.json", hardware_snapshot())

    wrapper.train()
    execution_model: torch.nn.Module = wrapper
    if not args.no_compile:
        execution_model = torch.compile(
            wrapper, backend="inductor", mode="default", fullgraph=False, dynamic=False
        )
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    def complete_pending_validation(
        epoch_number: int,
        *,
        train_loss_sum: float,
        train_ce_loss_sum: float,
        train_kd_loss_sum: float,
        train_samples_seen: int,
        train_seconds: float,
    ) -> None:
        nonlocal best_epoch, best_top1, completed_epoch, history
        if train_samples_seen != len(train_samples):
            raise ValueError("pending validation does not contain a complete training epoch")
        validation = validate(
            wrapper,
            preprocessor,
            val_samples,
            args.rgbnomore_root.resolve(),
            device,
            batch_size=args.val_batch_size,
            workers=args.val_workers,
        )
        row = {
            "epoch": epoch_number,
            "train_loss": train_loss_sum / train_samples_seen,
            "train_seconds": train_seconds,
            "optimizer_update": scheduler.completed_updates,
            **validation,
        }
        if teacher is not None:
            row["train_ce_loss"] = train_ce_loss_sum / train_samples_seen
            row["train_kd_loss"] = train_kd_loss_sum / train_samples_seen

        history = [
            existing
            for existing in history
            if int(existing.get("epoch", -1)) != epoch_number
        ]
        history.append(row)
        history.sort(key=lambda existing: int(existing["epoch"]))
        if validation["top1_percent"] > best_top1:
            best_top1 = float(validation["top1_percent"])
            best_epoch = epoch_number

        elapsed = elapsed_before + time.perf_counter() - started
        committed_payload = checkpoint_payload(
            wrapper,
            optimizer,
            scheduler,
            signature,
            trainable_names,
            completed_epoch=epoch_number,
            resume_epoch=epoch_number,
            next_microbatch_index=0,
            epoch_loss_sum=0.0,
            epoch_ce_loss_sum=0.0,
            epoch_kd_loss_sum=0.0,
            epoch_samples=0,
            epoch_elapsed_seconds=0.0,
            processed_images=processed_images,
            elapsed_seconds=elapsed,
            best_epoch=best_epoch,
            best_top1=best_top1,
            pending_validation_epoch=None,
        )
        epoch_checkpoint = output_dir / f"checkpoint_epoch_{epoch_number:03d}.pt"

        # Every side artifact is written idempotently before latest.pt clears
        # the pending marker. A failure at any earlier point therefore causes
        # resume to rerun validation instead of silently advancing training.
        save_checkpoint(epoch_checkpoint, committed_payload)
        if best_epoch == epoch_number:
            save_checkpoint(output_dir / "best.pt", committed_payload)
        write_json(output_dir / "history.json", history)
        write_history(output_dir / "history.csv", history)
        append_epoch_metric_once(output_dir / "metrics.jsonl", row)
        write_json(
            output_dir / "progress.json",
            {
                "status": "validated",
                "stage": stage,
                "completed_epoch": epoch_number,
                "best_epoch": best_epoch,
                "best_top1_percent": best_top1,
                "latest_validation": validation,
            },
        )
        save_checkpoint(latest_path, committed_payload)
        completed_epoch = epoch_number
        print(json.dumps({"record_type": "epoch", **row}, sort_keys=True), flush=True)

    if pending_validation_epoch is not None:
        complete_pending_validation(
            pending_validation_epoch,
            train_loss_sum=resume_epoch_loss,
            train_ce_loss_sum=resume_epoch_ce_loss,
            train_kd_loss_sum=resume_epoch_kd_loss,
            train_samples_seen=resume_epoch_samples,
            train_seconds=resume_epoch_elapsed,
        )
        resume_microbatch = 0
        resume_epoch_loss = 0.0
        resume_epoch_ce_loss = 0.0
        resume_epoch_kd_loss = 0.0
        resume_epoch_samples = 0
        resume_epoch_elapsed = 0.0

    for epoch in range(resume_epoch, epochs):
        epoch_started = time.perf_counter()
        permutation = epoch_permutation(len(train_samples), seed, epoch)
        full_microbatches = math.ceil(len(train_samples) / batch_size)
        first_microbatch = resume_microbatch if epoch == resume_epoch else 0
        loader = make_loader(
            train_samples,
            args.rgbnomore_root.resolve(),
            indices=permutation[first_microbatch * batch_size :],
            batch_size=batch_size,
            workers=args.train_workers,
        )
        epoch_loss = resume_epoch_loss if epoch == resume_epoch else 0.0
        epoch_ce_loss = resume_epoch_ce_loss if epoch == resume_epoch else 0.0
        epoch_kd_loss = resume_epoch_kd_loss if epoch == resume_epoch else 0.0
        epoch_samples = resume_epoch_samples if epoch == resume_epoch else 0
        epoch_elapsed_before = resume_epoch_elapsed if epoch == resume_epoch else 0.0

        for local_index, (yq, cq, quant, labels, ordinals) in enumerate(loader):
            microbatch_index = first_microbatch + local_index
            batch = int(labels.shape[0])
            window_start = (microbatch_index // accumulation) * accumulation
            window_end = min(window_start + accumulation, full_microbatches)
            window_samples = min(window_end * batch_size, len(train_samples)) - window_start * batch_size
            logical_ids = [train_samples[int(index)].logical_sample_id for index in ordinals.tolist()]
            decisions = [
                published_training_augmentation(
                    training_seed=seed,
                    epoch=epoch,
                    logical_sample_id=train_samples[int(index)].logical_sample_id,
                    virtual_pls_id=int(index) // 1024,
                    crop_policy="per-sample",
                    source_width=512,
                    source_height=512,
                ).decision
                for index in ordinals.tolist()
            ]
            inputs = preprocessor.training(yq, cq, quant, decisions)
            if bool(config.get("randaugment", False)):
                inputs, _ = apply_published_randaugment(
                    inputs,
                    training_seed=seed,
                    epoch=epoch,
                    logical_sample_ids=logical_ids,
                    rgbnomore_root=args.rgbnomore_root.resolve(),
                )
            labels_device = labels.to(device, non_blocking=True)
            mixup_alpha = float(config.get("mixup_alpha", 0.0))
            if mixup_alpha > 0:
                inputs, targets, _ = apply_published_mixup(
                    inputs,
                    labels_device,
                    training_seed=seed,
                    epoch=epoch,
                    microbatch_index=microbatch_index,
                    alpha=mixup_alpha,
                    classes=1000,
                )
            else:
                targets = labels_device
            teacher_logits: torch.Tensor | None = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(*inputs)
            logits = execution_model(*inputs)
            ce_loss = torch.nn.functional.cross_entropy(logits, targets)
            kd_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if teacher_logits is not None:
                kd_loss = logit_distillation_loss(
                    logits,
                    teacher_logits,
                    float(config["distillation"]["temperature"]),
                )
            loss = ce_loss + float(config.get("distillation", {}).get("lambda", 0.0)) * kd_loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite loss at epoch {epoch+1}")
            (loss * (batch / window_samples)).backward()
            epoch_loss += float(loss.detach().item()) * batch
            epoch_ce_loss += float(ce_loss.detach().item()) * batch
            epoch_kd_loss += float(kd_loss.detach().item()) * batch
            epoch_samples += batch
            processed_images += batch

            end_window = (
                (microbatch_index + 1) % accumulation == 0
                or microbatch_index + 1 == full_microbatches
            )
            if end_window:
                learning_rates = scheduler.prepare_next_update()
                gradients = [
                    parameter.grad
                    for parameter in wrapper.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients or not all(bool(value.isfinite().all().item()) for value in gradients):
                    raise FloatingPointError("missing or non-finite trainable gradients")
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
                        float(config["gradient_clip_norm"]),
                    ).item()
                )
                optimizer.step()
                scheduler.complete_update()
                optimizer.zero_grad(set_to_none=True)
                elapsed = elapsed_before + time.perf_counter() - started
                if scheduler.completed_updates % args.checkpoint_every_updates == 0:
                    payload = checkpoint_payload(
                        wrapper,
                        optimizer,
                        scheduler,
                        signature,
                        trainable_names,
                        completed_epoch=epoch,
                        resume_epoch=epoch,
                        next_microbatch_index=microbatch_index + 1,
                        epoch_loss_sum=epoch_loss,
                        epoch_ce_loss_sum=epoch_ce_loss,
                        epoch_kd_loss_sum=epoch_kd_loss,
                        epoch_samples=epoch_samples,
                        epoch_elapsed_seconds=(
                            epoch_elapsed_before + time.perf_counter() - epoch_started
                        ),
                        processed_images=processed_images,
                        elapsed_seconds=elapsed,
                        best_epoch=best_epoch,
                        best_top1=best_top1,
                        pending_validation_epoch=None,
                    )
                    save_checkpoint(latest_path, payload)
                if scheduler.completed_updates % args.log_every_updates == 0:
                    train_record = {
                        "record_type": "train_window",
                        "stage": stage,
                        "epoch": epoch + 1,
                        "optimizer_update": scheduler.completed_updates,
                        "loss": epoch_loss / epoch_samples,
                        "gradient_norm_before_clip": gradient_norm,
                        "learning_rates": learning_rates,
                        "processed_images": processed_images,
                        "elapsed_seconds": elapsed,
                    }
                    if teacher is not None:
                        train_record.update(
                            {
                                "ce_loss": epoch_ce_loss / epoch_samples,
                                "kd_loss": epoch_kd_loss / epoch_samples,
                            }
                        )
                    append_jsonl(output_dir / "metrics.jsonl", train_record)
                    write_json(
                        output_dir / "progress.json",
                        {
                            "status": "training",
                            "stage": stage,
                            "completed_epoch": epoch,
                            "current_epoch": epoch + 1,
                            "optimizer_update": scheduler.completed_updates,
                        },
                    )
            del inputs, labels_device, targets, teacher_logits, logits, ce_loss, kd_loss, loss

        if epoch_samples != len(train_samples):
            raise RuntimeError(f"epoch consumed {epoch_samples}, expected {len(train_samples)}")
        train_seconds = epoch_elapsed_before + time.perf_counter() - epoch_started
        elapsed = elapsed_before + time.perf_counter() - started
        prevalidation_payload = checkpoint_payload(
            wrapper,
            optimizer,
            scheduler,
            signature,
            trainable_names,
            completed_epoch=epoch + 1,
            resume_epoch=epoch + 1,
            next_microbatch_index=0,
            epoch_loss_sum=epoch_loss,
            epoch_ce_loss_sum=epoch_ce_loss,
            epoch_kd_loss_sum=epoch_kd_loss,
            epoch_samples=epoch_samples,
            epoch_elapsed_seconds=train_seconds,
            processed_images=processed_images,
            elapsed_seconds=elapsed,
            best_epoch=best_epoch,
            best_top1=best_top1,
            pending_validation_epoch=epoch + 1,
        )
        epoch_checkpoint = output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        save_checkpoint(epoch_checkpoint, prevalidation_payload)
        save_checkpoint(latest_path, prevalidation_payload)
        complete_pending_validation(
            epoch + 1,
            train_loss_sum=epoch_loss,
            train_ce_loss_sum=epoch_ce_loss,
            train_kd_loss_sum=epoch_kd_loss,
            train_samples_seen=epoch_samples,
            train_seconds=train_seconds,
        )
        resume_microbatch = 0
        resume_epoch_loss = 0.0
        resume_epoch_ce_loss = 0.0
        resume_epoch_kd_loss = 0.0
        resume_epoch_samples = 0
        resume_epoch_elapsed = 0.0

    write_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "stage": stage,
            "completed_epoch": epochs,
            "best_epoch": best_epoch,
            "best_top1_percent": best_top1,
        },
    )
    write_json(output_dir / "hardware_after.json", hardware_snapshot())
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--rgbnomore-root", type=Path, default=DEFAULT_RGBNOMORE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="K32/N196 teacher weights; defaults to the original pretrained checkpoint",
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-name")
    parser.add_argument("--train-workers", type=int, default=6)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--torch-cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every-updates", type=int, default=250)
    parser.add_argument("--log-every-updates", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-device-migration",
        action="store_true",
        help="allow resume when device_name is the only signature difference",
    )
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args(argv)
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        parser.error("--max-train-samples must be positive")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        parser.error("--max-val-samples must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
