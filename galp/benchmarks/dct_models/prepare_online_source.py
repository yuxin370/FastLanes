"""Store all original source JPEG coefficients; never construct model geometry."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image, JpegImagePlugin

import galp.benchmarks.dct_models.backend as B
from galp.benchmarks.dct_models.evaluate import samples
from galp.benchmarks.dct_models.online_crop import SOURCE_PROFILE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=min(256, len(os.sched_getaffinity(0))),
                   help="total CPU thread budget, defaults to all available cores (native limit 256)")
    args = p.parse_args()
    if not 1 <= args.threads <= 256:
        p.error("threads must be in [1, 256]")
    shard_workers = min(16, args.threads)
    workers_per_shard = max(1, args.threads // shard_workers)
    verification_workers = min(16, args.threads)  # Native verifier hard limit.
    root = args.output_dir.resolve()
    # A fresh destination makes the source identity unambiguous.
    root.mkdir(parents=True, exist_ok=False)
    entries = sorted(samples(50000), key=lambda s: s["galp_image_id"])
    if len(entries) != 50000:
        raise ValueError("expected the complete 50,000-image validation split")
    started = time.perf_counter()
    for entry in entries:
        with Image.open(entry["path"]) as image:
            if image.size != (512, 512) or JpegImagePlugin.get_sampling(image) != 2:
                raise ValueError(f"source512 requires 512px 4:2:0 JPEG: {entry['path']}")
    source_validation_seconds = time.perf_counter() - started
    (root / "samples.json").write_text(json.dumps(entries))
    paths = root / "source_paths.txt"
    paths.write_text("".join(str(Path(s["path"]).resolve()) + "\n" for s in entries))
    tool = B.REPO / "build/galp/tools/jpeg_dct/galp_jpeg_dct_tool"
    timings = dict(source_validation_seconds=source_validation_seconds)

    def run(name, command):
        start = time.perf_counter()
        with (root / f"{name}.log").open("w") as log:
            subprocess.run([str(x) for x in command], check=True, stdout=log, stderr=subprocess.STDOUT)
        timings[name + "_seconds"] = time.perf_counter() - start

    run("compression", [tool, "--shard", "--out-dir", root, "--physical-layout", "spatial-major",
                        "--shard-images", 1024, "--rowgroup-vectors", 128, "--rowgroups-per-shard", 64,
                        "--layout-threads", args.threads,
                        "--shard-workers", shard_workers,
                        "--shard-decode-threads", workers_per_shard,
                        "--encoding-workers-per-shard", workers_per_shard, "--input-list", paths])
    run("verification", [tool, "--verify-manifest", root / "manifest.bin", "--verify-workers",
                         verification_workers, "--input-list", paths])
    run("access_index", [B.REPO / "build/galp/tools/jpeg_dct/galp_block_major_access_tool",
                         root / "manifest.bin", "--output-dir", root / "access",
                         "--output-json", root / "access/build.json"])
    # Publish only after exact coefficient verification and index construction.
    (root / "profile.json").write_text(json.dumps(SOURCE_PROFILE, indent=2))
    result = dict(samples=50000, offline_work="JPEG entropy decode, lossless columnar compression, access index",
                  parallelism=dict(layout_threads=args.threads, shard_workers=shard_workers,
                                   decode_workers_per_shard=workers_per_shard,
                                   encoding_workers_per_shard=workers_per_shard,
                                   verification_workers=verification_workers),
                  target_geometry_generated=False, frequency_selection=False,
                  source_jpeg_bytes=sum(Path(s["path"]).stat().st_size for s in entries),
                  stored_bytes=sum(f.stat().st_size for f in root.rglob("*") if f.is_file()), **timings)
    (root / "generation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
