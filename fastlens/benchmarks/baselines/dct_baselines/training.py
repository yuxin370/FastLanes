"""Frozen published augmentation and the existing accumulated training recipe."""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import einops
import torch

from fastlens.benchmarks.training_pls import published_augmentation as published
from fastlens.benchmarks.training_pls.core_schedule import stable_seed
from fastlens.benchmarks.training_pls.published_optimizer import build_published_optimizer
from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import capture_rng_state, restore_rng_state

from .materialize import organize


def upstream_ops(root):
    """Load upstream functions without colliding with DCTNet's utils package."""
    path = Path(root) / "utils/dct_ops.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(torch=torch, einops=einops, math=math)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def freeze_operations(seed, epoch, image_id, profile):
    operations = published.published_randaugment_records(
        training_seed=seed, epoch=epoch, logical_sample_ids=[image_id])[0]
    grids = profile["output_grids"]
    cnn = profile["transform"] == "cnn_source512"
    for op in operations:
        internal = op.pop("internal_seed")
        op.pop("key")
        if op["operation"] == "Cutout":
            generator = published._torch_generator(internal)
            # The native published sampler uses the canonical 28-block grid.
            centers = [int(torch.randint(28, (1,), generator=generator))//2*2 for _ in range(2)]
            op["centers"] = [[v*grid//28 for v in centers] for grid in grids]
            op["pads"] = [2*grid//28 for grid in grids]
        elif op["operation"] in ("TranslateX", "TranslateY"):
            blocks = math.floor(op["magnitude"]/2)*2
            op["blocks"] = [blocks*g//28 for g in grids] if cnn else [blocks, blocks//2, blocks//2]
        elif op["operation"] == "ChromaDrop":
            op["channel"] = stable_seed("chroma-drop", internal) % 2
    return operations


def freeze_trace(images, profile, *, seed, epochs, batch_size, pool_size, labels, partition, crop_policy="per-sample"):
    rows = []
    microbatch = 0
    for epoch in range(epochs):
        order = list(images)
        import random
        random.Random(stable_seed("matched-order", seed, epoch, partition)).shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start:start+batch_size]
            mixup = published.published_mixup_record(training_seed=seed, epoch=epoch, microbatch_index=start//batch_size)
            ids = [f"{partition}:{epoch}:{start+i}:0" for i in range(len(batch))]
            for i, image in enumerate(batch):
                aug = published.published_training_augmentation(
                    training_seed=seed, epoch=epoch, logical_sample_id=image["image_id"],
                    virtual_pls_id=image["galp_image_id"]//1024 if crop_policy == "per-pls" else start//1024,
                    crop_policy=crop_policy, source_width=512, source_height=512).decision
                rows.append(dict(
                    occurrence_id=ids[i], image_id=image["image_id"], epoch=epoch, view_id=0,
                    logical_slot=len(rows), microbatch_id=microbatch, closed_pool_id=f"{epoch}:{start//pool_size}",
                    crop_box=[aug.crop_x, aug.crop_y, aug.crop_width, aug.crop_height], flip=aug.horizontal_flip,
                    resize_geometry=profile["output_grids"], profile_id=profile["id"],
                    augmentation_parameters=freeze_operations(seed, epoch, image["image_id"], profile),
                    mixup_partner_occurrence=ids[(i-1) % len(batch)], mixup_lambda=mixup["lambda_original"],
                    target_parameters=dict(classes=1000, lambda_rolled=mixup["lambda_rolled"]),
                    target_label=labels[image["image_id"]]))
            microbatch += 1
    return rows


def augment(raw, operations, dops, *, oracle=False):
    y, cb, cr = [value.round().clamp(-1024, 1016).to(torch.int16)[None] for value in raw]
    cbcr = torch.cat((cb, cr))
    for op in operations:
        name, magnitude = op["operation"], op["magnitude"]
        if name == "Cutout":
            values = [y, cbcr[:1], cbcr[1:]]
            result = []
            for value, center, pad in zip(values, op["centers"], op["pads"]):
                if oracle:
                    value = dops.cutout_dct(value, pad, cutout_center_height=center[0], cutout_center_width=center[1])[0]
                else:
                    value = published._cutout_batch(value[None], pad, [center[0]], [center[1]])[0]
                result.append(value)
            y, cbcr = result[0], torch.cat(result[1:])
        elif name in ("TranslateX", "TranslateY"):
            direction = "W" if name == "TranslateX" else "H"
            result = []
            for value, blocks in zip((y, cbcr[:1], cbcr[1:]), op["blocks"]):
                value = (dops.translate_dct(value, blocks, direction=direction) if oracle else
                         published._translate_batch(value[None], blocks, direction)[0])
                result.append(value)
            y, cbcr = result[0], torch.cat(result[1:])
        elif name == "ChromaDrop":
            cbcr = cbcr.clone()
            cbcr[op["channel"]] = 0
        elif oracle:
            y, cbcr = published._apply_operation(dops, y, cbcr, name, magnitude, internal_seed=0)
        else:
            yb, cb = published._apply_operation_batch(dops, y[None], cbcr[None], name, magnitude, internal_seeds=[0])
            y, cbcr = yb[0], cb[0]
        y, cbcr = y.clamp(-1024, 1016), cbcr.clamp(-1024, 1016)
    return y[0], cbcr[0], cbcr[1]


def finish_batch(raw_images, rows, profile, dops, *, oracle=False):
    if oracle:
        return oracle_finish_batch(raw_images, rows, profile, dops)
    values = [organize(profile, augment(raw, row["augmentation_parameters"], dops, oracle=oracle))
              if profile.get("training") else organize(profile, raw) for raw, row in zip(raw_images, rows)]
    inputs = tuple(torch.stack([v[i] for v in values]) for i in range(len(values[0])))
    labels = torch.tensor([r["target_label"] for r in rows], device=inputs[0].device)
    if not profile.get("training"):
        return inputs, labels
    positions = {r["occurrence_id"]: i for i, r in enumerate(rows)}
    partners = torch.tensor([positions[r["mixup_partner_occurrence"]] for r in rows], device=labels.device)
    original = inputs[0].new_tensor([r["mixup_lambda"] for r in rows])
    rolled = inputs[0].new_tensor([r["target_parameters"]["lambda_rolled"] for r in rows])
    def mix(value):
        shape = (len(rows),)+(1,)*(value.ndim-1)
        return value*original.reshape(shape) + value.index_select(0, partners)*rolled.reshape(shape)
    return tuple(mix(value) for value in inputs), mix(torch.nn.functional.one_hot(labels, 1000).float())


def oracle_finish_batch(raw_images, rows, profile, dops):
    """Scalar normalization/Mixup/targets independent of organize and finish_batch."""
    samples = []
    for raw, row in zip(raw_images, rows):
        values = augment(raw, row["augmentation_parameters"], dops, oracle=True) if profile.get("training") else raw
        if profile["transform"] == "cnn_source512":
            projected = []
            for (c, f), mean, std in zip(profile["output_channels"], profile["mean"], profile["std"]):
                value = values[c][..., f//8, f % 8].float()
                # CUDA division by a Python scalar uses a rounded reciprocal.
                # Device FP32 parameters preserve the declared tensor division.
                projected.append((value-value.new_tensor(mean))/value.new_tensor(std))
            samples.append((torch.stack(projected),))
        else:
            normalized = [(v.float()+4)/1020 if profile.get("training") else
                          -1+((v.float()+1024)/2040)*2 for v in values]
            samples.append((normalized[0][None], torch.stack(normalized[1:])))
    positions = {r["occurrence_id"]: i for i, r in enumerate(rows)}
    if not profile.get("training"):
        return tuple(torch.stack([s[c] for s in samples]) for c in range(len(samples[0]))), torch.tensor(
            [r["target_label"] for r in rows], device=samples[0][0].device)
    mixed = [[] for _ in samples[0]]
    targets = torch.zeros((len(rows), 1000), dtype=torch.float32, device=samples[0][0].device)
    for i, row in enumerate(rows):
        partner = positions[row["mixup_partner_occurrence"]]
        a, b = row["mixup_lambda"], row["target_parameters"]["lambda_rolled"]
        for c in range(len(mixed)):
            mixed[c].append(samples[i][c]*a+samples[partner][c]*b)
        targets[i, row["target_label"]] = a
        targets[i, rows[partner]["target_label"]] += b
    return tuple(torch.stack(values) for values in mixed), targets


def update_windows(trace, accumulation):
    """No accumulation across an epoch or a closed preparation pool."""
    windows, window, batches = [], [], 0
    previous = None
    for row in trace:
        identity = (row["epoch"], row["closed_pool_id"])
        new_batch = previous is None or row["microbatch_id"] != previous["microbatch_id"]
        if window and (identity != (previous["epoch"], previous["closed_pool_id"]) or (new_batch and batches == accumulation)):
            windows.append(window)
            window, batches = [], 0
        if new_batch:
            batches += 1
        window.append(row)
        previous = row
    if window:
        windows.append(window)
    return windows


def updates_per_epoch(population, batch_size, pool_size, accumulation):
    """Count the same retained tails and pool-closed windows without expanding a trace."""
    full_pools, tail = divmod(population, pool_size)
    def windows(samples):
        microbatches = (samples+batch_size-1)//batch_size
        return (microbatches+accumulation-1)//accumulation
    return full_pools*windows(pool_size)+windows(tail)


def make_optimizer(model, recipe):
    return build_published_optimizer(model, total_updates=recipe["total_updates"],
                                     warmup_updates=recipe["warmup_updates"])


def snapshot(model, recipe):
    optimizer, decay, scheduler = make_optimizer(model, recipe)
    return dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                decay=decay.state_dict(), scheduler=scheduler.state_dict(), rng=capture_rng_state())


def train_window(model, optimizer, decay, scheduler, batches, recipe, observe):
    model.train().requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    lr = scheduler.prepare_next_update()
    total = sum(len(target) for _, target in batches)
    losses = []
    for inputs, target in batches:
        device = inputs[0].device.type
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=recipe["precision"] == "bf16-autocast"):
            logits = model(*inputs)
            loss = torch.nn.functional.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite published-recipe loss")
        observe("logits", [logits.detach()])
        observe("loss", [loss.detach()])
        (loss*(len(target)/total)).backward()
        losses.append(float(loss.detach()))
    observe("gradients", [p.grad for p in model.parameters() if p.grad is not None])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    decay.step(lr)
    scheduler.complete_update()
    observe("parameters", list(model.state_dict().values()))
    observe("optimizer", [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)])
    return dict(samples=total, learning_rate=lr, losses=losses, scheduler=scheduler.state_dict(), decay=decay.state_dict())
