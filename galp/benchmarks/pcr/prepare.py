"""Losslessly convert a validation subset without modifying its source JPEGs."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .storage import EOI, UPSTREAM_COMMIT, split_scans, write_record


def convert(entry, jpegtran):
    source = Path(entry["path"]).read_bytes()
    progressive = subprocess.run(
        [str(jpegtran), "-copy", "none", "-optimize", "-progressive"],
        input=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout
    with Image.open(io.BytesIO(source)) as before, Image.open(io.BytesIO(progressive)) as after:
        np.testing.assert_array_equal(np.asarray(before.convert("RGB")), np.asarray(after.convert("RGB")))
    if not source.endswith(EOI):
        raise ValueError(f"Source JPEG has no terminal EOI: {entry['path']}")
    return source, split_scans(progressive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--physical-order", action="store_true",
                        help="Use the full-validation deployment sample order")
    parser.add_argument("--record-images", type=int, default=256)
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--jpegtran", type=Path, default=shutil.which("jpegtran"))
    args = parser.parse_args()
    if not 1 <= args.count <= 50000 or args.record_images < 1 or args.workers < 1:
        parser.error("Use 1..50000 samples and positive record size/workers")
    if args.jpegtran is None:
        parser.error("Specify --jpegtran or put it on PATH")
    from galp.benchmarks.dct_models.evaluate import samples

    entries = samples(args.count, physical_order=args.physical_order)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records, jpeg_bytes = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for first in range(0, len(entries), args.record_images):
            selected = entries[first:first + args.record_images]
            converted = list(pool.map(lambda entry: convert(entry, args.jpegtran), selected))
            labels = [entry["model_label"] for entry in selected]
            record = dict(first=first, count=len(selected))
            record["pcr"] = write_record(args.output / f"PCR_{first:05d}.pcr",
                                         [item[1] for item in converted], labels)
            # Same record granularity and reader, with original sequential JPEGs.
            record["jpeg_record"] = write_record(args.output / f"JPEG_{first:05d}.pcr",
                                                 [[item[0][:-2]] for item in converted], labels)
            jpeg_bytes += sum(len(item[0]) for item in converted)
            records.append(record)
            print(f"Converted and pixel-verified {first + len(selected)}/{len(entries)}", flush=True)
    result = dict(upstream_commit=UPSTREAM_COMMIT, source="ImageNet-512 validation",
                  selection="physical galp_image_id order" if args.physical_order else "samples(count), seed 11997733",
                  physical_order=args.physical_order, entries=entries, records=records,
                  jpeg_source_bytes=jpeg_bytes, record_images=args.record_images,
                  conversion_seconds=time.perf_counter() - started,
                  full_scan_pixel_identical_images=len(entries), jpegtran=str(args.jpegtran),
                  format="upstream raw PCR payload; JSON external index; JPEG records are a packing control")
    (args.output / "index.json").write_text(json.dumps(result) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("entries", "records")}))


if __name__ == "__main__":
    main()
