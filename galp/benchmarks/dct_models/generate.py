"""Bounded multiprocess target generation and native threaded GALP encoding."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import torch

import galp.benchmarks.dct_models.backend as B
from galp.benchmarks.dct_models.evaluate import samples
from galp.benchmarks.dct_models.storage import BRIDGE, pack_image

REFERENCE = None


def init_worker():
    global REFERENCE
    torch.set_num_threads(1)
    REFERENCE = B.Reference()


def produce(entries):
    start = time.perf_counter()
    parts = []
    source_bytes = 0
    for entry in entries:
        q, qt, _ = REFERENCE.coefficients(entry["path"])
        parts.append(pack_image(entry["path"], q, qt))
        source_bytes += Path(entry["path"]).stat().st_size
    return b"".join(parts), time.perf_counter() - start, source_bytes


def available_cpus():
    cores = len(os.sched_getaffinity(0))
    group = Path("/sys/fs/cgroup") / Path(Path("/proc/self/cgroup").read_text().split("::", 1)[1].strip()).relative_to("/")
    for parent in [group, *group.parents]:
        limit = parent / "cpu.max"
        if limit.exists():
            quota, period = limit.read_text().split()
            if quota != "max":
                cores = min(cores, max(1, int(quota) // int(period)))
        if parent == Path("/sys/fs/cgroup"):
            break
    return cores


def encode_shard(pool, entries, base, workers, threads, layout):
    start = time.perf_counter()
    process = subprocess.Popen([str(BRIDGE), "encode-block" if layout == "block-major" else "encode", str(base), str(len(entries)), str(threads), str(B.GRID)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    chunks = [entries[i:i+4] for i in range(0, len(entries), 4)]
    pending = {}
    submitted = 0
    generation_seconds = source_bytes = 0
    try:
        for index in range(len(chunks)):
            while submitted < min(len(chunks), index + 2):
                pending[submitted] = pool.submit(produce, chunks[submitted])
                submitted += 1
            data, elapsed, size = pending.pop(index).result()
            process.stdin.write(data)
            generation_seconds += elapsed
            source_bytes += size
        process.stdin.close()
        output = process.stdout.read()
        if process.wait() != 0:
            raise RuntimeError(f"GALP encoding failed: {base}")
    except BaseException as error:
        process.terminate()
        process.wait()
        base.with_suffix(".failed.json").write_text(json.dumps(dict(
            error=str(error), image_ids=[s["galp_image_id"] for s in entries]), indent=2))
        raise
    stats = json.loads(output)
    stats.update(wall_seconds=time.perf_counter()-start, generation_worker_seconds=generation_seconds,
                 source_bytes=source_bytes, workers=workers, encoding_threads=threads,
                 images_per_second=len(entries)/(time.perf_counter()-start))
    stats["metadata_bytes"] = base.with_suffix(".meta.bin").stat().st_size
    base.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    base.with_suffix(".failed.json").unlink(missing_ok=True)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--encoding-threads", type=int, default=8)
    parser.add_argument("--shard-workers", type=int, default=1)
    parser.add_argument("--shard-images", type=int, default=512)
    parser.add_argument("--layout", choices=["compact", "block-major"], default="compact")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    workers = args.workers or max(1, available_cpus()-args.encoding_threads*args.shard_workers)
    if workers + args.encoding_threads*args.shard_workers > available_cpus():
        raise ValueError("worker plus encoder allocation exceeds available CPU quota")
    # Global IDs retain original sorted-image order for the full dataset.
    entries = samples(args.count)
    if args.count == 50000:
        entries.sort(key=lambda s:s["galp_image_id"])
        for ordinal, entry in enumerate(entries):
            entry["ordinal"] = ordinal
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < args.count * (3 * B.GRID * B.GRID * 64 * 2 + 100_000):
        raise RuntimeError("insufficient free space for conservative uncompressed upper bound")
    profile = B.profile()
    existing = root / "profile.json"
    if existing.exists() and json.loads(existing.read_text()) != profile:
        raise ValueError("existing output profile differs")
    existing.write_text(json.dumps(profile, indent=2))
    if (root / "samples.json").exists() and json.loads((root / "samples.json").read_text()) != entries:
        raise ValueError("resume sample IDs/order differ from existing data version")
    (root / "samples.json").write_text(json.dumps(entries, indent=2))
    start = time.perf_counter()
    stats, bases = [], []
    reused = 0
    init_worker()
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        # Start producer processes before launching dispatch threads (safe fork boundary).
        list(pool.map(produce, [[] for _ in range(workers)]))
        with cf.ThreadPoolExecutor(max_workers=args.shard_workers) as dispatch:
            pending = {}
            shards = list(range(0, len(entries), args.shard_images))
            submitted = 0
            for shard, first in enumerate(shards):
                while submitted < min(len(shards), shard + args.shard_workers):
                    base = root / f"shard_{submitted:06d}"
                    begin = shards[submitted]
                    group = entries[begin:begin+args.shard_images]
                    if base.with_suffix(".json").exists():
                        result = json.loads(base.with_suffix(".json").read_text())
                        if result["images"] != len(group):
                            raise ValueError("resume shard sample count differs")
                        if (result.get("physical_layout") == "dct-major/spatial-major-image-minor") != (args.layout == "block-major"):
                            raise ValueError("resume physical layout differs")
                        if not base.with_suffix(".fls").is_file() or not base.with_suffix(".meta.bin").is_file():
                            raise ValueError("completed shard files are missing")
                        reused += len(group)
                        done = cf.Future()
                        done.set_result(result)
                        pending[submitted] = done
                    else:
                        pending[submitted] = dispatch.submit(encode_shard, pool, group, base, workers, args.encoding_threads, args.layout)
                    submitted += 1
                result = pending.pop(shard).result()
                stats.append(result)
                bases.append(f"shard_{shard:06d}")
                # Commit only the completed contiguous prefix; completion order is not image order.
                subprocess.run([str(BRIDGE), "merge-block" if args.layout == "block-major" else "merge", str(root), *bases], check=True)
                print(json.dumps(dict(shard=shard, completed_images=sum(s["images"] for s in stats), **result)), flush=True)
    labels = dict(format="galp_rgbnomore_label_map_v1", image_count=len(entries),
                  index_file=str(B.DEFAULT_INDEX_CSV), input_dir=str(B.DEFAULT_DATA_ROOT / "val"),
                  labels=[s["label"] for s in entries], sample_ids=[s["logical_sample_id"] for s in entries],
                  manifest=str(root / "manifest.bin"), split="val",
                  samples=[dict(image_id=i, label=s["label"], path=s["logical_sample_id"])
                           for i,s in enumerate(entries)])
    (root / "labels.json").write_text(json.dumps(labels))
    wall = time.perf_counter()-start
    if reused == len(entries) and (root / "generation.json").exists():
        print(json.dumps(dict(reused_samples=reused, resume_wall_seconds=wall,
                              generation_statistics="preserved from original completed run")))
        return
    result = dict(samples=len(entries), completed_shards=len(stats), failed_shards=0,
                  invocation_wall_seconds=wall, images_per_second=(len(entries)-reused)/wall,
                  newly_encoded_samples=len(entries)-reused, reused_samples=reused,
                  available_cpus=available_cpus(), workers=workers, encoding_threads=args.encoding_threads,
                  shard_workers=args.shard_workers,
                  parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  total_disk_bytes=sum(p.stat().st_size for p in root.iterdir() if p.is_file()),
                  shards=stats)
    (root / "generation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
