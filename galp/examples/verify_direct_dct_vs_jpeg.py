#!/usr/bin/env python3
"""Spot-check GALP Direct-DCT decompression against original JPEG DCT coefficients.

GALP stores the quantized DCT coefficients that were entropy-decoded from the
source JPEGs at build time and then FastLanes-compressed. Decompressing a shard
should therefore reproduce the *exact* quantized DCT coefficient blocks of the
original JPEG (lossless with respect to the JPEG's coefficients), so the diff is
expected to be bit-exact (tolerance 0).

Because the dataset has ~1.28M images we sample a subset (``--sample``) rather
than diffing everything. Image ordering is ``sorted(root.rglob(<jpeg>))`` — the
same order the manifest generator used to assign ``image_id`` — because the
manifest does not persist ``source_path``.

Reference decoder: RGB-no-more's ``dct_manip.read_coefficients`` (a libjpeg
wrapper independent of GALP). ``import torch`` must happen first so that its
``libc10.so`` is on the loader path.

Example:
    PYTHONPATH=build/galp/torch \
    /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
      galp/examples/verify_direct_dct_vs_jpeg.py \
      galp/data/imagedataset_dct/ImageNet-train-multi/manifest.bin \
      --jpeg-root /tmp/rgbnomore_imagenet/train \
      --jpeg-list /tmp/train_jpegs_sorted.txt \
      --sample 32 --seed 0 \
      --output-json /tmp/galp_train_multi_vs_jpeg.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch  # noqa: F401  (must precede dct_manip so libc10.so resolves)

import dct_manip as dm
import _galp_direct_dct as galp_dct

JPEG_EXT = {".jpg", ".jpeg", ".jpe"}


def _load_jpeg_list(jpeg_root: Path, jpeg_list: Path | None) -> list[Path]:
    if jpeg_list is not None and jpeg_list.exists():
        with jpeg_list.open("r", encoding="utf-8") as handle:
            paths = [Path(line.strip()) for line in handle if line.strip()]
        if not paths:
            raise RuntimeError(f"{jpeg_list} is empty")
        return paths
    paths = sorted(p for p in jpeg_root.rglob("*") if p.is_file() and p.suffix.lower() in JPEG_EXT)
    if not paths:
        raise RuntimeError(f"no JPEG files found under {jpeg_root}")
    if jpeg_list is not None:
        with jpeg_list.open("w", encoding="utf-8") as handle:
            for path in paths:
                handle.write(str(path) + "\n")
    return paths


def _is_color(reader: Any, image_id: int) -> bool:
    """True when Y/Cb/Cr are all present (ycbcr_dct_grid only supports 3-component images)."""
    components = reader.image_metadata(int(image_id)).get("components", [])
    slots = {int(c.get("semantic_slot_id")) for c in components if c.get("present")}
    if {0, 1, 2} <= slots:
        return True
    local = {int(c.get("local_component_index", -1)) for c in components if c.get("present")}
    return {0, 1, 2} <= local


def _galp_grids(reader: Any, image_id: int, cache_capacity_mib: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    batch = reader.read_batch(
        [int(image_id)],
        crop=None,
        dct_coeffs="all",
        cache_capacity_mib=cache_capacity_mib,
        layout="ycbcr_dct_grid",
    )
    y = batch.y.detach().to("cpu", torch.int32) if batch.y is not None else None
    cbcr = batch.cbcr.detach().to("cpu", torch.int32) if batch.cbcr is not None else None
    return y, cbcr


def _compare(name: str, galp: torch.Tensor | None, ref: torch.Tensor | None, tolerance: int) -> dict[str, Any]:
    if galp is None or ref is None:
        return {"name": name, "status": "missing", "galp_present": galp is not None, "ref_present": ref is not None}
    if tuple(galp.shape) != tuple(ref.shape):
        return {
            "name": name,
            "status": "shape_mismatch",
            "galp_shape": list(galp.shape),
            "ref_shape": list(ref.shape),
        }
    diff = (galp - ref).abs()
    mismatch = int((diff > tolerance).sum().item())
    return {
        "name": name,
        "status": "ok" if mismatch == 0 else "value_mismatch",
        "shape": list(galp.shape),
        "max_abs": int(diff.max().item()) if diff.numel() else 0,
        "mismatch_count": mismatch,
        "element_count": int(diff.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--jpeg-root", type=Path, default=Path("/tmp/rgbnomore_imagenet/train"))
    parser.add_argument("--jpeg-list", type=Path, default=Path("/tmp/train_jpegs_sorted.txt"),
                        help="cached sorted jpeg list; created from --jpeg-root if absent")
    parser.add_argument("--sample", type=int, default=32, help="number of random images to spot-check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-ids", type=int, nargs="*", help="explicit image ids (overrides --sample/--seed)")
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--cache-capacity-mib", type=int, default=1024)
    parser.add_argument("--output-json", default="/tmp/galp_direct_dct_vs_jpeg.json")
    args = parser.parse_args()

    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")

    t0 = time.time()
    jpegs = _load_jpeg_list(args.jpeg_root, args.jpeg_list)
    print(f"jpeg list: {len(jpegs)} entries ({time.time() - t0:.1f}s)", flush=True)

    reader = galp_dct.DirectDctReader(args.manifest)
    image_count = int(reader.image_count)
    print(f"reader image_count: {image_count}", flush=True)
    if len(jpegs) != image_count:
        raise RuntimeError(
            f"alignment guard failed: jpeg list has {len(jpegs)} files but manifest has {image_count} images. "
            "The sorted jpeg order must match manifest image_id assignment."
        )

    if args.image_ids:
        image_ids = [int(i) for i in args.image_ids]
    else:
        rng = random.Random(args.seed)
        image_ids = sorted(rng.sample(range(image_count), min(args.sample, image_count)))

    records: list[dict[str, Any]] = []
    worst_max_abs = 0
    total_mismatch = 0
    failures = 0
    skipped_grayscale = 0
    for image_id in image_ids:
        if not _is_color(reader, image_id):
            skipped_grayscale += 1
            print(f".. image {image_id} {jpegs[image_id].relative_to(args.jpeg_root)}: skipped (grayscale/non-3-component)",
                  flush=True)
            records.append({
                "image_id": image_id,
                "path": str(jpegs[image_id].relative_to(args.jpeg_root)),
                "ok": None,
                "status": "skipped_grayscale",
            })
            continue
        gy, gc = _galp_grids(reader, image_id, args.cache_capacity_mib)
        ref = dm.read_coefficients(str(jpegs[image_id]))
        ry = ref[2].to(torch.int32).unsqueeze(0)  # (1,1,H,W,8,8)
        rc = ref[3].to(torch.int32).unsqueeze(0)  # (1,2,H,W,8,8)

        tensors = [_compare("Y", gy, ry, args.tolerance), _compare("CbCr", gc, rc, args.tolerance)]
        image_ok = all(t.get("status") == "ok" for t in tensors)
        for t in tensors:
            worst_max_abs = max(worst_max_abs, int(t.get("max_abs", 0)))
            total_mismatch += int(t.get("mismatch_count", 0))
        if not image_ok:
            failures += 1
        records.append({
            "image_id": image_id,
            "path": str(jpegs[image_id].relative_to(args.jpeg_root)),
            "ok": image_ok,
            "tensors": tensors,
        })
        flag = "OK " if image_ok else "!! "
        print(f"{flag}image {image_id} {records[-1]['path']}: "
              + ", ".join(f"{t['name']}={t.get('status')}(max_abs={t.get('max_abs','-')})" for t in tensors),
              flush=True)

    payload = {
        "manifest": args.manifest,
        "jpeg_root": str(args.jpeg_root),
        "image_count": image_count,
        "sampled": len(image_ids),
        "tolerance": args.tolerance,
        "failures": failures,
        "skipped_grayscale": skipped_grayscale,
        "worst_max_abs": worst_max_abs,
        "total_mismatch": total_mismatch,
        "records": records,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsampled={len(image_ids)} compared={len(image_ids) - skipped_grayscale} "
          f"skipped_grayscale={skipped_grayscale} failures={failures} "
          f"worst_max_abs={worst_max_abs} total_mismatch={total_mismatch}", flush=True)
    print(f"wrote {args.output_json}", flush=True)

    if failures:
        print("VERIFICATION FAILED: GALP decompression does not match original JPEG DCT coefficients", flush=True)
        sys.exit(1)
    print("VERIFICATION PASSED: GALP decompression is bit-exact vs original JPEG DCT coefficients", flush=True)


if __name__ == "__main__":
    main()
