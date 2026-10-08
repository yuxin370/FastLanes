"""Measure full-training L3 file sizes without retaining the encoded files."""
import argparse
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .codec import Encoder, PATCH, decode_batch
from .prepare import ROOT, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = ROOT / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json"
    samples = json.loads(manifest.read_text())["samples"]
    encoder = Encoder(args.library)
    header_bytes = struct.calcsize("<4siiB") + 3 * (512 // PATCH) ** 2 * 2
    started = time.perf_counter()
    jpeg_bytes = l3_bytes = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for begin in range(0, len(samples), 64):
            batch = samples[begin:begin + 64]
            rgb = list(pool.map(pixels, batch))
            blobs = [encoder.encode(image) for image in rgb]
            decoded = decode_batch(blobs, args.library)
            for sample, expected, actual, blob in zip(batch, rgb, decoded, blobs):
                np.testing.assert_array_equal(actual, expected)
                jpeg_bytes += Path(sample["path"]).stat().st_size
                l3_bytes += len(blob)
            done = begin + len(batch)
            if begin % 4096 == 0 or done == len(samples):
                elapsed = time.perf_counter() - started
                print(json.dumps(dict(images=done, total_images=len(samples),
                                      seconds=elapsed, l3_bytes=l3_bytes,
                                      estimated_remaining_seconds=elapsed * (len(samples) - done) / done)),
                      flush=True)
    metadata_bytes = len(samples) * header_bytes
    result = dict(split="train_full", images=len(samples), patch_size=PATCH,
                  jpeg_bytes=jpeg_bytes, l3_bytes=l3_bytes,
                  payload_bytes=l3_bytes - metadata_bytes, metadata_bytes=metadata_bytes,
                  index_bytes=0, decoded_rgb_bytes=len(samples) * 512 * 512 * 3,
                  seconds=time.perf_counter() - started,
                  measurement="Every image encoded with the benchmark encoder; serialized bytes counted in memory, files not retained",
                  metadata_scope="File headers and patch-length directories; row control bits and row padding remain in payload",
                  index_scope="No separate persistent access index; shared benchmark manifests excluded, as for JPEG",
                  correctness="Every image is pixel-identical to Pillow RGB decode of the source JPEG",
                  source_manifest=str(manifest), library=str(args.library))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
