"""Convert the exact paper validation set and RGB training window to L3."""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .codec import Encoder, PATCH, decode_batch

ROOT = Path(__file__).resolve().parents[3]
SEED = 11997733


def source_samples(split):
    path = ROOT / "galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3" / (split + ".json")
    samples = json.loads(path.read_text())["samples"]
    return random.Random(SEED).sample(samples, (32 + 64) * 1024 + 256) if split == "train" else samples


def encoded_path(root, sample_id):
    # Preserve split/class/name and the original suffix to avoid filename collisions.
    return Path(root) / (sample_id + ".l3")


def pixels(sample):
    with Image.open(sample["path"]) as image:
        return np.asarray(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    args = parser.parse_args()
    samples = source_samples(args.split)
    encoder = Encoder(args.library)
    started = time.perf_counter()
    original_bytes = stored_bytes = 0
    # Bounded batches avoid retaining the complete decoded dataset in host memory.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for begin in range(0, len(samples), 64):
            batch = samples[begin:begin + 64]
            rgb = list(pool.map(pixels, batch))
            blobs = [encoder.encode(x) for x in rgb]
            # Verify every converted image, not just a sampled codec smoke test.
            decoded = decode_batch(blobs, args.library)
            for sample, expected, actual, blob in zip(batch, rgb, decoded, blobs):
                np.testing.assert_array_equal(actual, expected)
                path = encoded_path(args.output, sample["logical_sample_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
                original_bytes += Path(sample["path"]).stat().st_size
                stored_bytes += len(blob)
            if begin % 1024 == 0:
                print(f"{args.split}: {begin + len(batch)}/{len(samples)}, {time.perf_counter()-started:.1f}s", flush=True)
    result = dict(split=args.split, images=len(samples), seconds=time.perf_counter()-started,
                  jpeg_bytes=original_bytes, l3_bytes=stored_bytes, decoded_rgb_bytes=len(samples)*512*512*3,
                  correctness="Every image is pixel-identical to Pillow RGB decode of the source JPEG",
                  source="ImageNet-512", patch_size=PATCH,
                  upstream_commit="3b8e52c226ac8749025e14858d5c5f260b562085")
    (args.output / (args.split + "_conversion.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
