"""Prepare the authors' ScanGroup layout from raw PCR or the training selection."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import shutil
import time

from fastlens.benchmarks.baselines.pcr.storage import MetadataRecord


def load_schema(build_dir):
    path = build_dir / "adapted/tensorflow/core/example/progressive_compressed_record_pb2.py"
    spec = importlib.util.spec_from_file_location("pcr_schema", path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    return schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument("--training-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--record-images", type=int, default=256)
    parser.add_argument("--jpegtran", type=Path, default=shutil.which("jpegtran"))
    args = parser.parse_args()
    if args.training_manifest and args.jpegtran is None:
        parser.error("Training preparation requires --jpegtran")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    schema = load_schema(args.build_dir)

    if args.training_manifest:
        from fastlens.benchmarks.baselines.pcr.prepare import convert as convert_jpeg
        entries = json.loads(args.training_manifest.read_text())
        entries = [dict(entry, model_label=entry["label"], ordinal=i) for i, entry in enumerate(entries)]
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for first in range(0, len(entries), args.record_images):
                selected = entries[first:first + args.record_images]
                converted = list(pool.map(lambda e: convert_jpeg(e, args.jpegtran), selected))
                scans = [item[1] for item in converted]
                levels = len(scans[0])
                if any(len(image) != levels for image in scans):
                    raise ValueError("Training record has inconsistent JPEG scan counts")
                name = f"PCR_{first:05d}.pcr"
                header = schema.MetadataRecord(labels=[e["model_label"] for e in selected],
                                               progressive_levels=levels, version=1).SerializeToString()
                sizes = [len(header)]
                with (args.output / name).open("xb") as target:
                    target.write(header)
                    for level in range(levels):
                        payload = schema.ScanGroup(image_bytes=[image[level] for image in scans]).SerializeToString()
                        target.write(payload)
                        sizes.append(len(payload))
                records.append(dict(path=name, sizes=sizes, first=first, count=len(selected)))
                print(f"Converted and pixel-verified {first + len(selected)}/{len(entries)}", flush=True)
        source_info = dict(training_manifest=str(args.training_manifest.resolve()),
                           full_scan_pixel_identical_images=len(entries), jpegtran=str(args.jpegtran))
    else:
        index = json.loads((args.source / "index.json").read_text())
        entries = index["entries"]
        source_info = dict(source_raw=str(args.source.resolve()))

    def convert(record):
        raw = record["pcr"]
        blob = (args.source / raw["path"]).read_bytes()
        metadata = MetadataRecord.FromString(blob[:raw["metadata_bytes"]])
        header = schema.MetadataRecord(
            labels=list(metadata.labels), progressive_levels=metadata.progressive_levels, version=1,
        ).SerializeToString()
        offset = raw["metadata_bytes"]
        sizes = [len(header)]
        with (args.output / raw["path"]).open("xb") as target:
            target.write(header)
            for lengths in raw["scan_lengths"]:
                group = schema.ScanGroup()
                for length in lengths:
                    group.image_bytes.append(blob[offset:offset + length])
                    offset += length
                payload = group.SerializeToString()
                target.write(payload)
                sizes.append(len(payload))
        if offset != len(blob):
            raise ValueError(f"Unconsumed raw bytes in {raw['path']}")
        return dict(path=raw["path"], sizes=sizes, first=record["first"], count=record["count"])

    if args.source:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(convert, index["records"]))
    offsets = schema.DatasetOffsetsIndex()
    for record in records:
        offsets.records.add(name=record["path"], offsets=record["sizes"])
    (args.output / "PCR_index.pb").write_bytes(offsets.SerializeToString())
    (args.output / "index.json").write_text(json.dumps(dict(
        **source_info, entries=entries, records=records,
        layout="authors ScanGroup protobuf; version 1", conversion_seconds=time.perf_counter() - started,
        record_images=records[0]["count"],
    )))
    print(json.dumps(dict(samples=len(entries), records=len(records),
                         payload_bytes=sum(sum(record["sizes"]) for record in records))))


if __name__ == "__main__":
    main()
