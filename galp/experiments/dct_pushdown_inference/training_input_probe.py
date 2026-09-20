"""Small real-training-data check of native DCTNet crop/resize and backward.

This probes the existing generic reader, not the complete B6 PLS training loop.
The extra reference reads are correctness checks, never throughput measurements.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import torch

import backend as B
from evaluate_shards import native_options, organize_batch
from old_adapter import load_upsample_dct
from storage import Reader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--required-gpu-name", default="RTX 4090")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(11997733)
    if args.required_gpu_name not in torch.cuda.get_device_name():
        raise RuntimeError(f"training input probe requires {args.required_gpu_name}")
    sys.path.insert(0, str(B.REPO / "build/galp/torch"))
    import _galp_direct_dct as native

    with args.mapping.open(newline="") as f:
        entries = list(itertools.islice(csv.DictReader(f), 8))
    train = json.loads((B.DEFAULT_E2E_V3_ROOT / "training_manifests_official_v3/train.json").read_text())
    classes = sorted({s["logical_sample_id"].split("/")[1] for s in train["samples"]})
    labels = {name: i for i, name in enumerate(classes)}
    for s in entries:
        if not s["logical_sample_id"].startswith("train/"):
            raise ValueError("probe must not train on validation images")
    ids = [int(s["planned_physical_position"]) for s in entries]
    extents = [56, 56, 28, 28, 14, 14, 4, 4]
    transforms = [dict(crop=[16, 16, extent * 8, extent * 8],
                       horizontal_flip=bool(i % 2), logical_sample_id=s["logical_sample_id"])
                  for i, (s, extent) in enumerate(zip(entries, extents))]
    resize = load_upsample_dct(B.DEFAULT_RGBNOMORE_ROOT, resize=True)
    expected = []
    expected_coefficients = []
    with Reader(train["galp_manifest"]) as source:
        for s, extent, descriptor in zip(entries, extents, transforms):
            q, tables = source.read(int(s["galp_image_id"]))
            components = []
            for c, (coef, table) in enumerate(zip(q, tables)):
                size, ratio = (64, 1) if c == 0 else (32, 2)
                if coef.shape != (size, size, 64):
                    raise ValueError(f"unexpected source grid {coef.shape}")
                raw = torch.as_tensor(coef).float() * torch.as_tensor(table).reshape(64)
                origin, count = 2 // ratio, extent // ratio
                raw = raw[origin:origin + count, origin:origin + count].reshape(1, count, count, 8, 8)
                if descriptor["horizontal_flip"]:
                    raw = raw.flip(2).clone()
                    raw[..., 1::2] *= -1
                raw = resize(raw, B.GRID, dtype_out=torch.float32, conv_mxs={})
                components.append(raw.reshape(B.GRID, B.GRID, 64))
            expected.append(B.organize(components))
            expected_coefficients.append(torch.cat([
                c.permute(2, 0, 1)[indices] for c, indices in zip(components, B.INDICES)]))
    expected = torch.stack(expected).cuda()
    expected_coefficients = torch.stack(expected_coefficients).cuda()
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(args.manifest.parent / "block_major_access_v1")
    reader = native.DirectDctReader(str(args.manifest))
    if not reader.initialization_stats["block_major_metadata_lazy"]:
        raise RuntimeError("probe requires existing block-major access sidecars")

    def options(projected):
        result = native_options(pushdown=False, projected=projected)
        result["grid_transform"].update(
            crop_reference_width_blocks=64, crop_reference_height_blocks=64,
            crop_origin_alignment_blocks=2, chroma_crop_scale_x=2, chroma_crop_scale_y=2,
            allowed_chroma_sampling_ratios=[[1, 2, 1, 2]])
        return result

    checks = []
    for projected in (False, True):
        batch = reader.read_batch(ids, transforms=transforms, **options(projected))
        x = batch.projected if projected else organize_batch(batch.y, batch.cbcr)
        torch.cuda.synchronize()
        # Existing float32 grids still round to nearest integer at finalization.
        # Report that difference explicitly instead of claiming float-reference
        # equivalence. Allow 0.005 coefficient units at FP32 half-integer ties.
        actual_coefficients = x * B.STD.cuda() + B.MEAN.cuda()
        torch.testing.assert_close(actual_coefficients, actual_coefficients.round(), atol=0.005, rtol=0)
        coefficient_error = float((actual_coefficients - expected_coefficients).abs().max())
        if coefficient_error > 0.505:
            raise AssertionError(f"native/reference difference exceeds final rounding: {coefficient_error}")
        if projected:
            torch.testing.assert_close(x, grid_input, atol=0, rtol=0)
        else:
            grid_input = x.clone()
        stats = dict(batch.execution_stats)
        checks.append(dict(output_layout="projected" if projected else "grid",
                           max_difference_vs_unrounded_input=float((x - expected).abs().max()),
                           max_difference_vs_unrounded_coefficients=coefficient_error,
                           rounded_reference_disagreements=int((actual_coefficients.round() != expected_coefficients.round()).sum()),
                           grid_projected_max_abs=float((x - grid_input).abs().max()),
                           native_stats=stats))
        if not stats["uses_planless_fixed_transform"]:
            raise RuntimeError("probe did not use the planless transform")
        if not projected:
            del x, batch
    # A real backward must retain the native input owner until it completes.
    model = B.model().cuda().requires_grad_(True).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    target = torch.tensor([labels[s["logical_sample_id"].split("/")[1]] for s in entries], device="cuda")
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("non-finite native-input training step")
    optimizer.step()
    torch.cuda.synchronize()
    if torch.equal(before, parameter.detach()):
        raise RuntimeError("native-input step did not update parameters")
    result = dict(profile=B.PROFILE, samples=len(entries), source_split="train",
                  physical_manifest=str(args.manifest), mapping=str(args.mapping),
                  physical_image_ids=ids, transforms=transforms, native_options=options(True),
                  checks=checks, loss=float(loss.detach()), parameter_update_verified=True,
                  output_shape=list(x.shape), coefficients_read="all; spatial transform mixes frequencies",
                  numerical_contract="native final round-to-nearest-even and int16 saturation, then DCTNet normalization in float32",
                  unrounded_reference_equivalent=False,
                  interpretation="crop/flip/resize correctness and one backward only; not B6 E2E or convergence",
                  full_training_epochs_completed=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_input_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
