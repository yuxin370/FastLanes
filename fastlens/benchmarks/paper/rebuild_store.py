"""Rebuild the paper's old block-major stores without re-encoding their payloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

from fastlens import annotations, cli
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def copy_exact(source, destination):
    shutil.copyfile(source, destination)
    if annotations.file_sha256(source) != annotations.file_sha256(destination):
        raise ValueError(f"copy differs from the original: {source}")


def records_from_paper(source, split, paths, manifest):
    positions = ((shard["shard_id"], offset) for shard in manifest["shards"]
                 for offset in range(shard["image_count"]))
    records = []
    if split == "train":
        stream = (source / "ordered_mapping.csv").open(encoding="utf-8", newline="")
    else:
        stream = (source / "samples.json").open(encoding="utf-8")
    with stream:
        rows = csv.DictReader(stream) if split == "train" else json.load(stream)
        for index, (path, row, position) in enumerate(zip(paths, rows, positions, strict=True)):
            if split == "train":
                physical_id = int(row["planned_physical_position"])
                source_path = Path(row["source_path"])
                if (int(row["virtual_pls_id"]), int(row["position_in_pls"])) != position:
                    raise ValueError(f"PLS mapping disagrees with physical partitions at image {index}")
            else:
                physical_id, source_path = row["galp_image_id"], Path(row["path"])
            if physical_id != index or source_path != path:
                raise ValueError(f"paper sample mapping disagrees with source order at image {index}")
            records.append(annotations.InputRecord(path, row["logical_sample_id"], int(row["label"])))
    return records


def rebuild(source, output, *, split, workers, build_dir=None):
    source, output = source.resolve(), output.absolute()
    cli.check_destination(output)
    tools = cli.find_tools(build_dir)
    old_store = source / "dct" if split == "train" else source
    old_manifest = parse_manifest(old_store / "manifest.bin")
    if old_manifest["version"] != 1:
        raise ValueError("paper rebuild requires a block-major manifest-v1 dataset")
    listing = source / ("ordered_inputs.txt" if split == "train" else "source_paths.txt")
    contract = None
    if split == "train":
        contract = json.loads((source / "materialization_contract.json").read_text())
        for key, path in (("ordered_input_list", listing), ("ordered_mapping", source / "ordered_mapping.csv")):
            if annotations.file_sha256(path) != contract[key + "_sha256"]:
                raise ValueError(f"{key} disagrees with the original materialization contract")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.rebuild-", dir=output.parent))
    try:
        store = stage / "dct" if split == "train" else stage
        (store / "conversion").mkdir(parents=True)
        (store / "geometry").mkdir()
        copy_exact(listing, store / "conversion/sources.txt")
        (store / annotations.SOURCE_ORDER).write_text(
            annotations.file_sha256(store / "conversion/sources.txt") + "\n", encoding="ascii")
        paths = annotations.physical_paths(store, old_manifest["image_count"])
        records = records_from_paper(source, split, paths, old_manifest)

        pairs = []
        for shard in old_manifest["shards"]:
            for key in ("fls_file_name", "metadata_file_name"):
                destination = store / shard[key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                pairs.append((old_store / shard[key], destination))
        print(f"Copying and comparing {len(pairs)} payload/metadata files with {workers} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda pair: copy_exact(*pair), pairs))
        copy_exact(old_store / "manifest.bin", store / "manifest.bin")
        log = store / "conversion/native.log"
        access, jpeg = str(tools[cli.TOOL_NAMES[1]]), str(tools[cli.TOOL_NAMES[0]])
        cli.run_native([access, "--finalize-store", str(store / "manifest.bin")], log, "Store layout")
        cli.run_native([access, str(store / "manifest.bin"), "--output-dir", str(store / "geometry")],
                       log, "Geometry index")

        # Only member paths may change. All partition and rowgroup fields must survive.
        rebuilt = parse_manifest(store / "manifest.bin")
        for key in ("version", "rowgroup_vectors", "rowgroups_per_shard", "image_count", "shard_count"):
            if rebuilt[key] != old_manifest[key]:
                raise ValueError(f"rebuilt manifest changed {key}")
        for old, new in zip(old_manifest["shards"], rebuilt["shards"], strict=True):
            for key in old.keys() - {"fls_file_name", "metadata_file_name", "fls_path", "metadata_path"}:
                if old[key] != new[key]:
                    raise ValueError(f"rebuilt partition {old['shard_id']} changed {key}")
        cli.run_native([jpeg, "--verify-manifest", str(store / "manifest.bin"),
                        "--verify-workers", str(workers), "--input-list", str(store / "conversion/sources.txt")],
                       log, "Exact DCT verification")
        info = cli.inspect_standard_store(store, tools[cli.TOOL_NAMES[1]])
        if info["jpeg_warning_image_ids"]:
            raise ValueError("paper store contains JPEG recovery warnings")
        annotations.publish_annotations(store, records, None,
                                        label_source="ordered_mapping.csv" if split == "train" else "samples.json")
        if split == "train":
            copy_exact(source / "ordered_mapping.csv", stage / "ordered_mapping.csv")
            copy_exact(source / "materialization_contract.json", store / "conversion/original_materialization_contract.json")
            # The mapping and its hashes are unchanged; writer fields remain historical.
            contract["ordered_mapping"] = str(output / "ordered_mapping.csv")
            contract["ordered_input_list"] = str(output / "dct/conversion/sources.txt")
            (stage / "materialization_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        else:
            for name in ("samples.json", "profile.json"):
                copy_exact(source / name, stage / name)
        report = dict(schema_version=1, operation="paper-layout-rebuild", source_store=str(old_store),
                      image_count=info["image_count"], shard_count=info["shard_count"],
                      original_jpeg_bytes=sum(path.stat().st_size for path in paths),
                      rowgroup_vectors=rebuilt["rowgroup_vectors"], rowgroups_per_shard=rebuilt["rowgroups_per_shard"],
                      payload_and_metadata_bytes_unchanged=True, verification="full",
                      core_storage_bytes=info["core_storage_bytes"], source_order_sha256=info["source_order_sha256"])
        (store / cli.REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
        cli.check_destination(output)
        if output.exists():
            output.rmdir()
        stage.rename(output)
        print(f"Published {output}: {info['image_count']} images, {info['shard_count']} unchanged partitions.", flush=True)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="old validation root or premixed training root")
    parser.add_argument("--output", type=Path, required=True, help="new or empty destination; old data stays intact")
    parser.add_argument("--split", choices=("val", "train"), required=True)
    parser.add_argument("--workers", type=int, default=min(32, len(os.sched_getaffinity(0))),
                        help="parallel copies and exact verifiers, 1..32 (native verifier limit)")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in [1, 32]")
    rebuild(args.source, args.output, split=args.split, workers=args.workers, build_dir=args.build_dir)


if __name__ == "__main__":
    main()
