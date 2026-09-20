"""DCTNet model-only training calibration using the existing Transformer harness.

Real training-split inputs are prepared once with the reference preprocessing.
This measures resident-input training, not augmentation or pipeline throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import backend as B
from evaluate import Inputs

sys.path.insert(0, str(B.REPO))
sys.path.insert(0, str(B.REPO / "galp/benchmarks/system_dct_major"))
from galp.benchmarks.model_only_training_calibration import run_model_only_calibration
from galp.benchmarks.training_audit_policy import build_audit_policy
from training_pls.published_optimizer import build_published_optimizer
from training_pls.recipe import SWINV2_RECIPE_NAME, recipe_contract


def worker_init(_):
    torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["dct"] if B.PROFILE == "efun" else ["rgb", "dct"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measured-updates", type=int, default=120)
    parser.add_argument("--precision", choices=["fp32", "bf16-autocast"], default="fp32")
    parser.add_argument("--required-gpu-name", default="RTX 4090")
    args = parser.parse_args()
    if args.domain == "rgb":
        import rgb
    torch.set_num_threads(8)
    torch.manual_seed(11997733)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    if args.required_gpu_name not in torch.cuda.get_device_name(device):
        raise RuntimeError(f"this comparison requires {args.required_gpu_name}")
    recipe = recipe_contract(SWINV2_RECIPE_NAME)
    microbatch = recipe["training"]["physical_microbatch"]
    accumulation = recipe["training"]["gradient_accumulation"]
    manifest_path = B.DEFAULT_E2E_V3_ROOT / "training_manifests_official_v3/train.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "train":
        raise ValueError("model calibration requires training-split source images")
    population = manifest["samples"]
    classes = sorted({s["logical_sample_id"].split("/")[1] for s in population})
    labels = {name: index for index, name in enumerate(classes)}
    selected = random.Random(11997733).sample(population, microbatch)
    entries = [dict(s, ordinal=i, model_label=labels[s["logical_sample_id"].split("/")[1]])
               for i, s in enumerate(selected)]
    dataset = rgb.Inputs(entries) if args.domain == "rgb" else Inputs(entries, route="R")
    loader = DataLoader(dataset, batch_size=microbatch, num_workers=8,
                        worker_init_fn=worker_init)
    batch = next(iter(loader))
    inputs, targets = batch[0].to(device), batch[1].to(device)
    net = (rgb.model() if args.domain == "rgb" else B.model()).to(device)
    net.requires_grad_(True).train()
    optimizer, decay, scheduler = build_published_optimizer(
        net, total_updates=math.ceil(len(population) / (microbatch * accumulation)) * 300)
    execution_model = torch.compile(net, **{
        k: v for k, v in recipe["execution"]["model_compile"].items() if k != "enabled"})
    first_parameter = next(net.parameters())
    before = first_parameter.detach().clone()
    print(f"{B.PROFILE} {args.domain}: training calibration starts", flush=True)
    audit_policy = build_audit_policy()
    result = run_model_only_calibration(
        domain=args.domain, execution_model=execution_model, model=net,
        optimizer=optimizer, weight_decayer=decay, scheduler=scheduler,
        inputs=(inputs,), labels=targets, device=device,
        audit_policy=audit_policy, microbatch_images=microbatch,
        gradient_accumulation=accumulation, gradient_clipping_norm=1.0,
        warmup_updates=args.warmup_updates, measured_updates=args.measured_updates,
        precision=args.precision)
    changed = not torch.equal(before, first_parameter.detach())
    if not changed:
        raise RuntimeError("training calibration did not update the first parameter")
    strict_measured_updates = max(0, min(args.measured_updates,
                                        audit_policy["strict_update_count"] - args.warmup_updates))
    result.update(
        profile=B.PROFILE, device_name=torch.cuda.get_device_name(device),
        checkpoint=str(rgb.CHECKPOINT if args.domain == "rgb" else B.CHECKPOINT),
        initialization="official pretrained checkpoint; calibration updates discarded",
        training_manifest=str(manifest_path), distinct_source_images=len(entries),
        logical_sample_ids=[e["logical_sample_id"] for e in entries],
        input_shape=list(inputs.shape), parameter_update_verified=changed,
        model_compile=recipe["execution"]["model_compile"],
        preprocessing="reference center crop, resident-input calibration only",
        recipe_reference=SWINV2_RECIPE_NAME,
        measured_strict_audit_updates=strict_measured_updates,
        measured_deferred_audit_updates=args.measured_updates - strict_measured_updates,
        full_training_epochs_completed=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.domain}_model_only.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
