"""Tiny deterministic PLS input, encoded with the production JPEG/sidecar tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import random
import subprocess


def create_fixture(root: Path, jpeg_tool: str, access_tool: str) -> tuple[Path, Path, str]:
    from PIL import Image

    source = root / "jpeg"
    source.mkdir(parents=True)
    rng = random.Random(11997733)
    for index in range(2):
        Image.frombytes("RGB", (512, 512), rng.randbytes(512 * 512 * 3)).save(
            source / f"{index}.jpg", quality=95, subsampling=2
        )
    dataset = root / "dct"
    subprocess.run([
        jpeg_tool, "--shard", "--out-dir", str(dataset), "--preset", "balanced",
        "--physical-layout", "spatial-major",
        "--shard-images", "1024", "--metadata-profile", "reconstruct", str(source),
    ], check=True)
    manifest = dataset / "manifest.bin"
    subprocess.run([
        access_tool, str(manifest), "--output-dir", str(dataset / "block_major_access_v1"),
    ], check=True)
    # This is the public CSV input contract, not hand-written binary metadata.
    mapping = root / "mapping.csv"
    with mapping.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "planned_physical_position", "virtual_pls_id", "position_in_pls",
            "galp_image_id", "logical_sample_id", "label",
        ])
        for index, label in enumerate((7, 23)):
            writer.writerow((index, 0, index, index, f"synthetic-{index}", label))
    return manifest, mapping, hashlib.sha256(mapping.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jpeg-tool", required=True)
    parser.add_argument("--access-tool", required=True)
    args = parser.parse_args()
    print(create_fixture(args.output_dir, args.jpeg_tool, args.access_tool))
