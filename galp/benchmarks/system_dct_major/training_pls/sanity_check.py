#!/usr/bin/env python3
"""One 1024-sample A0 update against the RGB-no-more JPEG-DCT reference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .core_schedule import epoch_position_pools
from .layout import load_layout_mapping, validate_manifest_against_layout
from .published_augmentation import (
    apply_published_mixup,
    apply_published_randaugment,
    apply_published_randaugment_scalar_reference,
    published_training_augmentation,
)
from .published_optimizer import build_published_optimizer
from .recipe import RECIPE_NAME, recipe_contract
from .train import (
    REPO_ROOT,
    _adapter_config,
    _atomic_json,
    _load_manifest_fast,
    _move_batch,
    build_paired_model,
    compile_published_model,
)


RGB_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "system_rgbnomore"
if str(RGB_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(RGB_BENCHMARK_ROOT))

from training.direct_dct_reader import DirectDctTrainingReader  # noqa: E402
from training.pipeline import build_training_adapter  # noqa: E402
from training.sample_order import SampleIdentity  # noqa: E402


def _tensor_difference(
    left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    difference = (left.detach() - right.detach()).abs()
    return {
        "shape_left": list(left.shape),
        "shape_right": list(right.shape),
        "dtype_left": str(left.dtype),
        "dtype_right": str(right.dtype),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.float().mean().item()) if difference.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
    }


def _state_difference(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if set(left) != set(right):
        return {"allclose": False, "name_mismatch": True}
    worst_name = None
    worst = 0.0
    failures: list[str] = []
    for name in sorted(left):
        difference = (left[name].detach() - right[name].detach()).abs()
        maximum = float(difference.max().item()) if difference.numel() else 0.0
        if maximum > worst:
            worst = maximum
            worst_name = name
        if not torch.allclose(left[name], right[name], atol=atol, rtol=rtol):
            failures.append(name)
    return {
        "allclose": not failures,
        "atol": atol,
        "rtol": rtol,
        "max_abs": worst,
        "max_abs_parameter": worst_name,
        "failed_parameter_count": len(failures),
        "first_failed_parameters": failures[:20],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    recipe = recipe_contract(RECIPE_NAME)
    mapping = load_layout_mapping(args.layout_plan)
    train_samples, train_meta = _load_manifest_fast(
        args.train_manifest, expected_split="train"
    )
    validate_manifest_against_layout(
        mapping, manifest_path=args.train_manifest, samples=train_samples
    )
    positions = next(
        epoch_position_pools(
            mapping,
            condition_id="A0",
            seed=args.seed,
            epoch=0,
        )
    )[2][:1024]
    if len(positions) != 1024:
        raise ValueError("A0 reference sanity requires at least 1024 training samples")
    selected_samples = [
        train_samples[int(mapping.manifest_indices[position])] for position in positions
    ]
    identities = [
        SampleIdentity(0, index, sample.logical_sample_id)
        for index, sample in enumerate(selected_samples)
    ]
    augmentations = [
        published_training_augmentation(
            training_seed=args.seed,
            epoch=0,
            logical_sample_id=sample.logical_sample_id,
            virtual_pls_id=int(mapping.virtual_pls_ids[position]),
            crop_policy="per-sample",
            source_width=sample.width,
            source_height=sample.height,
        )
        for position, sample in zip(positions, selected_samples)
    ]
    rgbnomore_path = str(args.rgbnomore_root.resolve())
    if rgbnomore_path not in sys.path:
        sys.path.insert(0, rgbnomore_path)
    from utils.custom_transforms import RandomResizedCrop_DCT

    reference_crop = RandomResizedCrop_DCT(
        28, scale=(0.05, 1.0), ratio=(1, 1), dtype_resize=torch.float32
    )
    crop_reference_mismatches: list[dict[str, Any]] = []
    flip_reference_mismatches: list[str] = []
    for sample, augmentation in zip(selected_samples, augmentations):
        coefficient_shape = (
            1,
            math.ceil(sample.height / 8),
            math.ceil(sample.width / 8),
            1,
            1,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(augmentation.crop_seed % (2**63 - 1))
            reference_box = RandomResizedCrop_DCT.get_params(
                torch.empty(coefficient_shape),
                reference_crop.scale,
                reference_crop.ratio,
                reference_crop.even_size_choices,
                reference_crop.chroma_scale,
            )
        observed_box = (
            augmentation.decision.crop_y // 8,
            augmentation.decision.crop_x // 8,
            augmentation.decision.crop_height // 8,
            augmentation.decision.crop_width // 8,
        )
        if tuple(reference_box) != observed_box:
            crop_reference_mismatches.append(
                {
                    "logical_sample_id": sample.logical_sample_id,
                    "reference": list(reference_box),
                    "observed": list(observed_box),
                }
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(augmentation.flip_seed % (2**63 - 1))
            reference_flip = float(torch.rand(()).item()) < 0.5
        if reference_flip != augmentation.decision.horizontal_flip:
            flip_reference_mismatches.append(sample.logical_sample_id)
    crop_descriptor_digest = hashlib.sha256(
        json.dumps(
            [value.as_dict() for value in augmentations],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A0 reference sanity requires CUDA")
    torch.cuda.set_device(device)
    galp_manifest = Path(train_meta["galp_manifest"])
    reader = DirectDctTrainingReader(
        galp_manifest, module_path=args.galp_torch_module_path
    )
    current = build_training_adapter(
        "galp",
        selected_samples,
        batch_size=64,
        workers=args.workers,
        device=device,
        config=_adapter_config(
            reader=reader,
            galp_manifest=galp_manifest,
            module_path=args.galp_torch_module_path,
            prefetch_depth=2,
        ),
    )
    reference = build_training_adapter(
        "rgbnomore",
        selected_samples,
        batch_size=64,
        workers=args.workers,
        device=device,
        config={
            "rgbnomore_root": str(args.rgbnomore_root.resolve()),
            "prefetch_depth": 2,
        },
    )
    decisions = [value.decision for value in augmentations]
    current.begin(identities, decisions, [64] * 16)
    reference.begin(identities, decisions, [64] * 16)

    current_model, _current_hash = build_paired_model(
        args.rgbnomore_root, seed=args.seed, device=device
    )
    reference_model, _reference_hash = build_paired_model(
        args.rgbnomore_root, seed=args.seed, device=device
    )
    initial_hash_current = _current_hash
    initial_hash_reference = _reference_hash
    current_optimizer, current_decay, current_scheduler = build_published_optimizer(
        current_model, total_updates=375_600
    )
    reference_optimizer, reference_decay, reference_scheduler = build_published_optimizer(
        reference_model, total_updates=375_600
    )
    current_execution_model = compile_published_model(current_model, recipe)
    current_optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    current_lr = current_scheduler.prepare_next_update()
    reference_lr = reference_scheduler.prepare_next_update()
    dct_tolerance = 1.0 / 1020.0 + 1e-7
    native_differences: list[dict[str, Any]] = []
    augmented_differences: list[dict[str, Any]] = []
    logits_differences: list[dict[str, Any]] = []
    loss_differences: list[float] = []
    randaugment_digest = hashlib.sha256()
    mixup_digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        for microbatch_index in range(16):
            current_batch = current.next_batch()
            reference_batch = reference.next_batch()
            current_inputs, current_labels = _move_batch(current_batch, device)
            reference_inputs, reference_labels = _move_batch(reference_batch, device)
            if not torch.equal(current_labels, reference_labels):
                raise RuntimeError("current/reference labels differ")
            if [value.logical_sample_id for value in current_batch.identities] != [
                value.logical_sample_id for value in reference_batch.identities
            ]:
                raise RuntimeError("current/reference logical order differs")
            native_differences.append(
                {
                    "microbatch_index": microbatch_index,
                    "y": _tensor_difference(
                        current_inputs[0],
                        reference_inputs[0],
                        atol=dct_tolerance,
                        rtol=1e-4,
                    ),
                    "cbcr": _tensor_difference(
                        current_inputs[1],
                        reference_inputs[1],
                        atol=dct_tolerance,
                        rtol=1e-4,
                    ),
                }
            )
            logical_ids = [value.logical_sample_id for value in current_batch.identities]
            current_augmented, current_ra = apply_published_randaugment(
                (current_inputs[0], current_inputs[1]),
                training_seed=args.seed,
                epoch=0,
                logical_sample_ids=logical_ids,
                rgbnomore_root=args.rgbnomore_root,
            )
            reference_augmented, reference_ra = apply_published_randaugment_scalar_reference(
                (reference_inputs[0], reference_inputs[1]),
                training_seed=args.seed,
                epoch=0,
                logical_sample_ids=logical_ids,
                rgbnomore_root=args.rgbnomore_root,
            )
            if current_ra != reference_ra:
                raise RuntimeError("keyed RandAugment decisions differ")
            randaugment_digest.update(
                json.dumps(current_ra, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            augmented_differences.append(
                {
                    "microbatch_index": microbatch_index,
                    "y": _tensor_difference(
                        current_augmented[0],
                        reference_augmented[0],
                        atol=dct_tolerance,
                        rtol=1e-4,
                    ),
                    "cbcr": _tensor_difference(
                        current_augmented[1],
                        reference_augmented[1],
                        atol=dct_tolerance,
                        rtol=1e-4,
                    ),
                }
            )
            current_mixed, current_targets, current_mixup = apply_published_mixup(
                current_augmented,
                current_labels,
                training_seed=args.seed,
                epoch=0,
                microbatch_index=microbatch_index,
            )
            reference_mixed, reference_targets, reference_mixup = apply_published_mixup(
                reference_augmented,
                reference_labels,
                training_seed=args.seed,
                epoch=0,
                microbatch_index=microbatch_index,
            )
            if current_mixup != reference_mixup or not torch.equal(
                current_targets, reference_targets
            ):
                raise RuntimeError("keyed Mixup decisions differ")
            mixup_digest.update(
                json.dumps(
                    current_mixup, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            current_logits = current_execution_model(*current_mixed)
            reference_logits = reference_model(*reference_mixed)
            logits_differences.append(
                _tensor_difference(
                    current_logits,
                    reference_logits,
                    atol=args.logits_atol,
                    rtol=args.logits_rtol,
                )
            )
            current_loss = torch.nn.functional.cross_entropy(
                current_logits, current_targets
            )
            reference_loss = torch.nn.functional.cross_entropy(
                reference_logits, reference_targets
            )
            loss_differences.append(
                abs(float(current_loss.item()) - float(reference_loss.item()))
            )
            (current_loss / 16.0).backward()
            (reference_loss / 16.0).backward()
            current.snapshot_batch_metrics(current_batch)
            reference.snapshot_batch_metrics(reference_batch)
    finally:
        current.close()
        reference.close()

    current_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in current_model.named_parameters()
        if parameter.grad is not None
    }
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference_model.named_parameters()
        if parameter.grad is not None
    }
    gradient_difference = _state_difference(
        current_gradients,
        reference_gradients,
        atol=args.gradient_atol,
        rtol=args.gradient_rtol,
    )
    torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
    current_optimizer.step()
    reference_optimizer.step()
    current_decay.step(current_lr)
    reference_decay.step(reference_lr)
    current_scheduler.complete_update()
    reference_scheduler.complete_update()
    parameter_difference = _state_difference(
        current_model.state_dict(),
        reference_model.state_dict(),
        atol=args.parameter_atol,
        rtol=args.parameter_rtol,
    )
    native_ok = all(
        record[channel]["allclose"]
        for record in native_differences
        for channel in ("y", "cbcr")
    )
    augmented_ok = all(
        record[channel]["allclose"]
        for record in augmented_differences
        for channel in ("y", "cbcr")
    )
    logits_ok = all(record["allclose"] for record in logits_differences)
    loss_ok = max(loss_differences, default=0.0) <= args.loss_atol
    checks = {
        "initial_model_hash_equal": initial_hash_current == initial_hash_reference,
        "crop_descriptors_present": len(augmentations) == 1024,
        "crop_descriptors_match_reference_algorithm": not crop_reference_mismatches,
        "horizontal_flip_matches_keyed_reference_draw": not flip_reference_mismatches,
        "randaugment_decisions_equal": True,
        "mixup_decisions_equal": True,
        "dct_native_tensors_within_tolerance": native_ok,
        "dct_augmented_tensors_within_tolerance": augmented_ok,
        "logits_within_tolerance": logits_ok,
        "loss_within_tolerance": loss_ok,
        "gradients_within_tolerance": gradient_difference["allclose"],
        "one_step_parameters_within_tolerance": parameter_difference["allclose"],
    }
    result = {
        "schema_version": "galp-pls-a0-reference-sanity-v2",
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": 1024,
        "microbatch_size": 64,
        "gradient_accumulation": 16,
        "optimizer_updates": 1,
        "seed": args.seed,
        "condition": "A0",
        "layout_hash": mapping.layout_hash,
        "recipe_hash": recipe["recipe_hash"],
        "crop_descriptor_digest": crop_descriptor_digest,
        "crop_reference_mismatch_count": len(crop_reference_mismatches),
        "crop_reference_first_mismatches": crop_reference_mismatches[:20],
        "flip_reference_mismatch_count": len(flip_reference_mismatches),
        "flip_reference_first_mismatches": flip_reference_mismatches[:20],
        "randaugment_decision_digest": randaugment_digest.hexdigest(),
        "mixup_decision_digest": mixup_digest.hexdigest(),
        "initial_model_hash": initial_hash_current,
        "dct_tolerance": dct_tolerance,
        "native_differences": native_differences,
        "augmented_differences": augmented_differences,
        "logits_max_abs": max(
            (record["max_abs"] for record in logits_differences), default=0.0
        ),
        "loss_max_abs": max(loss_differences, default=0.0),
        "gradient_difference": gradient_difference,
        "parameter_difference": parameter_difference,
        "wall_clock_seconds": time.perf_counter() - started,
        "reference_path": "RGB-no-more JPEG coefficient reader + explicit published transforms",
        "current_path": "GALP Direct-DCT reader + keyed published RandAugment/Mixup",
        "claim_boundary": "one optimizer update sanity check; not convergence evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--layout-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--galp-torch-module-path", type=Path, default=REPO_ROOT / "build/galp/torch"
    )
    parser.add_argument(
        "--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more")
    )
    parser.add_argument("--logits-atol", type=float, default=2e-4)
    parser.add_argument("--logits-rtol", type=float, default=2e-4)
    parser.add_argument("--loss-atol", type=float, default=2e-5)
    parser.add_argument("--gradient-atol", type=float, default=2e-5)
    parser.add_argument("--gradient-rtol", type=float, default=2e-4)
    parser.add_argument("--parameter-atol", type=float, default=2e-6)
    parser.add_argument("--parameter-rtol", type=float, default=2e-5)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
