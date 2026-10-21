#!/usr/bin/env python3
"""Measure representation files with stat; keep historical conversion logs explicit."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def measured_files(paths):
    records = []
    for path in sorted(set(paths)):
        stat = path.stat()
        records.append(dict(path=str(path), logical_bytes=stat.st_size, allocated_bytes=stat.st_blocks * 512))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced/section_6_3/storage')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'fastlens/data')
    parser.add_argument('--ffcv-beton', type=Path,
                        default=ROOT / 'fastlens/data/paper/ffcv/val.beton')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compressed = args.data_root / "compressed"
    data = args.data_root / "system_rgbnomore"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    raw = dict(timestamp=timestamp, git_commit=commit, command=sys.argv, representations=[])

    def add(name, view, split, payload, metadata=(), indexes=(), **extra):
        raw["representations"].append(dict(name=name, view=view, split=split,
            payload=measured_files(payload), metadata=measured_files(metadata), indexes=measured_files(indexes), **extra))

    for view, split, path in (("original", "val", data / "e2e_v2/imagenet/val"),
                              ("imagenet512", "val", data / "e2e_v3/imagenet_512/val"),
                              ("imagenet512", "train", data / "e2e_v3/imagenet_512/train")):
        files = [p for p in path.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")]
        if not files:
            raise FileNotFoundError(f"no JPEGs found: {path}")
        add("jpeg", view, split, files, sample_count=len(files), conversion_scope="source dataset; conversion not measured")
    for name, split, path in (
        ("imagelanes_source", "val", compressed / "imagenet512_val_block_major/manifest.bin"),
        ("cpu_stored_dct_optional", "val", compressed / "imagenet512_val_compact_v3/manifest.bin"),
        ("imagelanes_source", "train", compressed / "imagenet512_train_block_major_premixed/dct/manifest.bin")):
        manifest = parse_manifest(path)
        payload = [Path(shard["fls_path"]) for shard in manifest["shards"]]
        metadata = [path, *[Path(shard["metadata_path"]) for shard in manifest["shards"]]]
        metadata += [path.parent / name for name in ("samples.json", "profile.json", "labels.json")
                     if (path.parent / name).is_file()]
        add(name, "imagenet512", split, payload, metadata, sample_count=manifest["image_count"])
        if name == "imagelanes_source":
            if split == "val":
                indexes = list((path.parent / "access").glob("*.bin"))
                indexes += [compressed / "imagenet512_val_compact_v3/labels.json"]
                conversion_path = path.parent / "generation.json"
            else:
                indexes = list((path.parent / "block_major_access_v1").glob("*.bin"))
                indexes += [path.parent.parent / "ordered_mapping.csv"]
                conversion_path = path.parent.parent / "materialization_result.json"
            conversion = json.loads(conversion_path.read_text())
            # Historical evidence is kept verbatim, not silently treated as a new timed run.
            add("imagelanes_prepared", "imagenet512", split, payload, metadata, indexes,
                sample_count=manifest["image_count"], conversion_raw=str(conversion_path),
                conversion_log=conversion, conversion_scope="historical executed preparation; timing conditions differ from current evaluation" +
                ("; train access-index construction duration MISSING (conversion_s covers compression only)" if split == "train" else ""))
    beton = args.ffcv_beton
    from ffcv.reader import Reader
    reader = Reader(str(beton))
    embedded = dict(payload_bytes=int(reader.alloc_table["size"].sum()),
                    metadata_bytes=reader.header.nbytes + reader.field_descriptors.nbytes + reader.metadata.nbytes,
                    index_bytes=reader.alloc_table.nbytes)
    embedded["padding_bytes"] = beton.stat().st_size - sum(embedded.values())
    assert embedded["padding_bytes"] >= 0
    add("ffcv_beton", "imagenet512", "val", [beton], sample_count=int(reader.num_samples), embedded=embedded,
        conversion_scope="conversion duration MISSING; payload, embedded metadata/index and file padding measured with FFCV Reader")
    raw_path = output / "storage_raw.json"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    jpeg = {(r["view"], r["split"]): sum(f["logical_bytes"] for f in r["payload"])
            for r in raw["representations"] if r["name"] == "jpeg"}
    rows = []
    for index, representation in enumerate(raw["representations"]):
        payload, metadata, indexes = [sum(f["logical_bytes"] for f in representation[k]) for k in ("payload", "metadata", "indexes")]
        total = payload + metadata + indexes
        padding = None
        if "embedded" in representation:
            embedded = representation["embedded"]
            payload, metadata, indexes, padding = [embedded[k] for k in ("payload_bytes", "metadata_bytes", "index_bytes", "padding_bytes")]
        conversion = representation.get("conversion_log", {})
        conversion_seconds = conversion.get("compression_seconds")
        if conversion_seconds is not None and "access_index_seconds" in conversion:
            conversion_seconds += conversion["access_index_seconds"]
        rows.append(dict(timestamp=timestamp, git_commit=commit, representation=representation["name"],
            dataset_view=representation["view"], split=representation["split"], samples=representation["sample_count"],
            payload_bytes=payload, metadata_bytes=metadata, index_bytes=indexes, padding_bytes=padding, representation_bytes=total,
            allocated_bytes=sum(f["allocated_bytes"] for k in ("payload", "metadata", "indexes") for f in representation[k]),
            jpeg_bytes=jpeg[representation["view"], representation["split"]],
            storage_amplification=total / jpeg[representation["view"], representation["split"]],
            conversion_s=conversion_seconds, verification_s=conversion.get("verification_seconds"),
            temporary_peak_bytes=None, notes=representation.get("conversion_scope", "manifest-referenced files only; logs and unused copies excluded"),
            raw_result=f"{raw_path}#/representations/{index}"))
    fields = list(rows[0])
    with (output / "storage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
