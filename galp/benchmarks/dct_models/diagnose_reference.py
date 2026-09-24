"""Single-worker diagnostic of the official DCT preprocessing stages."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from jpeg2dct.numpy import loads

import galp.benchmarks.dct_models.backend as B
from galp.benchmarks.dct_models.evaluate import samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    ref = B.Reference()
    totals = dict(read=0., jpeg_decode=0., geometry=0., jpeg_encode=0.,
                  dct_extract=0., channel_normalize=0.)
    checks, size = [], 0
    entries = samples(128)
    for i, entry in enumerate(entries):
        t = time.perf_counter()
        blob = Path(entry["path"]).read_bytes()
        totals["read"] += time.perf_counter()-t
        size += len(blob)
        t = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        totals["jpeg_decode"] += time.perf_counter()-t
        t = time.perf_counter()
        branches = ref.geometry(image)
        totals["geometry"] += time.perf_counter()-t
        t = time.perf_counter()
        encoded = [ref.encoder.encode(b, quality=100, jpeg_subsample=2) for b in branches]
        totals["jpeg_encode"] += time.perf_counter()-t
        t = time.perf_counter()
        extracted = [loads(b) for b in encoded]
        totals["dct_extract"] += time.perf_counter()-t
        t = time.perf_counter()
        x, _, _ = ref.tensor((extracted[0][0], extracted[1][1], extracted[1][2]))
        totals["channel_normalize"] += time.perf_counter()-t
        if i < 4:
            error = float((x-ref(entry["path"])).abs().max())
            assert error == 0
            checks.append(error)
    result = dict(samples=len(entries), stage_seconds=totals,
                  stage_ms_per_image={k:1000*v/len(entries) for k,v in totals.items()},
                  source_bytes=size, input_max_abs_errors=checks, cpu_threads=1,
                  scope="sequential single-worker stage work; not parallel E2E or GPU timing; verification excluded")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
