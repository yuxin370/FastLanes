"""Convert JPEG datasets using the native, coefficient-preserving pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from . import annotations


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = ("fastlens_jpeg_dct_tool", "fastlens_block_major_access_tool")
REPORT_NAME = "conversion/report.json"
PHYSICAL_LAYOUT = "spatial-major-image-minor"
JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe"}


class ConversionError(RuntimeError):
    """An actionable input, native pipeline, or publication failure."""


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def worker_count(value: str) -> str | int:
    return "auto" if value == "auto" else positive_int(value)


def annotation_options(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--image-root", type=Path, help="root for relative sample IDs and CSV paths; no JPEG reading")
    labels = parser.add_mutually_exclusive_group(required=required)
    labels.add_argument("--labels-from-folders", action="store_true", help="derive classes from ROOT/CLASS/... paths")
    labels.add_argument("--labels-jsonl", type=Path, help="complete path,label[,sample_id] JSONL; paths relative to this file")
    labels.add_argument("--labels-csv", type=Path, help="complete Filepath,Label CSV, with optional sample_id")
    parser.add_argument("--class-map", type=Path, help="fixed class_to_idx.json for folder labels; preserve original IDs")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastlens", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="convert original JPEG DCT coefficients")
    convert.add_argument("input_dir", nargs="?", type=Path, help="recursively discover JPEG files")
    convert.add_argument("--input-list", type=Path, help="ordered UTF-8 paths, one per line; relative to the list")
    convert.add_argument("--shuffle-seed", type=int, help="premix before partitioning using fastlens-shuffle-v1; unsigned 64-bit seed")
    annotation_options(convert, required=False)
    convert.add_argument("--output", type=Path, required=True, help="new or empty destination directory")
    convert.add_argument("--layout", choices=["block-major"], default="block-major")
    convert.add_argument("--partition-images", type=positive_int, default=1024, help="maximum images per partition (default: 1024)")
    convert.add_argument("--preset", choices=["balanced", "throughput", "crop-latency"], default="balanced")
    convert.add_argument("--workers", type=worker_count, default="auto", help="CPU budget for defaults: auto or 1..256")
    for option, description in (
        ("layout-threads", "dataset-wide JPEG header scan"),
        ("shard-decode-threads", "JPEG coefficient decoders per active shard"),
        ("shard-workers", "concurrent shards; multiplies resident shard memory"),
        ("encoding-workers-per-shard", "FastLanes rowgroup encoders per active shard"),
        ("verify-workers", "exact verification workers, 1..32; each holds a shard"),
    ):
        convert.add_argument(f"--{option}", type=positive_int, help=description)
    convert.add_argument("--verify", choices=["full", "none"], default="full", help="exact native DCT comparison (default: full)")
    convert.add_argument("--build-dir", type=Path, help="native CMake build directory; overrides FASTLENS_BUILD_DIR")
    convert.add_argument("--dry-run", action="store_true", help="check paths/tools and print plan; do not decode JPEGs or write files")
    convert.add_argument("--report", type=Path, help="also save the report at a new external path")
    inspect = commands.add_parser(
        "inspect", help="read metadata and check index compatibility (not full payload integrity)",
        description="Check dataset structure and index compatibility without decoding or checksumming the full FLS payload.",
    )
    inspect.add_argument("store", type=Path)
    inspect.add_argument("--build-dir", type=Path)
    inspect.add_argument("--json", action="store_true", help="emit machine-readable inspection results")
    annotation = commands.add_parser("annotations", help="build or validate task annotations without recompression")
    actions = annotation.add_subparsers(dest="annotation_command", required=True)
    build = actions.add_parser("build", help="add annotations to an existing store, preserving physical order")
    build.add_argument("store", type=Path)
    build.add_argument("--build-dir", type=Path)
    build.add_argument("--dry-run", action="store_true", help="validate all annotations without writing files")
    annotation_options(build, required=True)
    check = actions.add_parser("inspect", help="validate annotation bindings, counts and physical order")
    check.add_argument("store", type=Path)
    check.add_argument("--build-dir", type=Path)
    check.add_argument("--json", action="store_true")
    return parser


def parallel_config(args: argparse.Namespace) -> dict[str, int]:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    budget = min(32, max(1, available // 2)) if args.workers == "auto" else args.workers
    if budget > 256:
        raise ConversionError("--workers must be auto or in [1, 256]")
    shards = args.shard_workers or min(4, max(1, budget // 4))
    inner = min(4, max(1, budget // shards))
    config = {
        "layout_threads": args.layout_threads or budget,
        "shard_decode_threads": args.shard_decode_threads or inner,
        "shard_workers": shards,
        "encoding_workers_per_shard": args.encoding_workers_per_shard or inner,
        "verify_workers": args.verify_workers or min(4, budget),
    }
    inner_threads = max(config["shard_decode_threads"], config["encoding_workers_per_shard"])
    if config["layout_threads"] > 256 or shards * inner_threads > 256:
        raise ConversionError(
            "native safety bound: layout-threads <= 256 and shard-workers * "
            "max(shard-decode-threads, encoding-workers-per-shard) <= 256"
        )
    if config["verify_workers"] > 32:
        raise ConversionError("--verify-workers must be in [1, 32]")
    return config


def discover_inputs(input_dir: Path | None, input_list: Path | None) -> tuple[list[Path], int, str]:
    if (input_dir is None) == (input_list is None):
        raise ConversionError("supply exactly one input directory or --input-list PATH")
    if input_list is not None:
        # Do not strip whitespace: spaces are valid filename characters.
        with input_list.open(encoding="utf-8") as source:
            paths = [input_list.resolve().parent / line.rstrip("\r\n") for line in source if line.rstrip("\r\n")]
        policy = "input-list order; relative paths resolved against list directory; blank lines ignored"
    else:
        if not input_dir.is_dir():
            raise ConversionError(f"input directory does not exist: {input_dir}")
        paths = []

        def walk_error(error: OSError) -> None:
            raise error

        for directory, _, filenames in os.walk(input_dir, onerror=walk_error, followlinks=False):
            paths.extend(Path(directory) / name for name in filenames if Path(name).suffix.lower() in JPEG_EXTENSIONS)
        paths.sort(key=lambda path: path.relative_to(input_dir).as_posix())
        policy = "recursive relative POSIX paths sorted lexicographically by Unicode code point; case-sensitive; no directory symlink traversal"
    if not paths:
        raise ConversionError("no JPEG inputs found; expected .jpg, .jpeg, or .jpe files")
    seen: set[Path] = set()
    ordered = []
    source_bytes = 0
    for path in paths:
        resolved = path.resolve()
        if any(character in str(resolved) for character in "\r\n\0"):
            raise ConversionError(f"input path cannot be represented in a line-based list: {str(path)!r}")
        if resolved in seen:
            raise ConversionError(f"duplicate input path (including symlink aliases): {path}")
        if not path.is_file():
            raise ConversionError(f"input is missing or is not a regular file: {path}")
        if resolved.suffix.lower() not in JPEG_EXTENSIONS:
            raise ConversionError(f"input must have a JPEG extension (.jpg, .jpeg, .jpe): {path}")
        source_bytes += resolved.stat().st_size
        seen.add(resolved)
        ordered.append(resolved)
    return ordered, source_bytes, policy


def find_tools(build_dir: Path | None, names: tuple[str, ...] = TOOL_NAMES) -> dict[str, Path]:
    selected = build_dir or os.environ.get("FASTLENS_BUILD_DIR")
    roots = [Path(selected)] if selected else [REPO_ROOT / "build", REPO_ROOT / "build-native", *sorted(REPO_ROOT.glob("cmake-build-*"))]
    for root in roots:
        for directory in (root / "fastlens/tools/jpeg_dct", root / "bin", root):
            tools = {name: directory / name for name in names}
            if all(path.is_file() and os.access(path, os.X_OK) for path in tools.values()):
                return {name: path.resolve() for name, path in tools.items()}
    raise ConversionError(
        f"required native tools unavailable: {', '.join(names)}; searched {', '.join(str(root) for root in roots)}. "
        "Build with cmake --build BUILD_DIR --target fastlens_jpeg_dct_tool fastlens_block_major_access_tool --parallel, "
        "then set --build-dir BUILD_DIR or FASTLENS_BUILD_DIR."
    )


def check_destination(output: Path) -> None:
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ConversionError(f"refusing to replace nonempty output or symlink: {output}; choose a new or empty directory")


def run_native(command: list[str], log: Path, phase: str) -> None:
    print(f"{phase}...", flush=True)
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            # Periodic progress without streaming the encoder's per-image logs.
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"{phase}: still running...", flush=True)
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 8192))
            detail = stream.read().decode("utf-8", errors="replace")
        raise ConversionError(f"{phase} failed (exit {code}):\n{detail}")


def native_inspect(store: Path, tool: Path) -> dict:
    result = subprocess.run([str(tool), "--inspect", str(store / "manifest.bin")], capture_output=True, text=True)
    if result.returncode:
        raise ConversionError(f"native inspection failed: {result.stderr.strip()}\nRebuild fastlens_block_major_access_tool if it lacks --inspect.")
    return json.loads(result.stdout)


def annotation_source(args: argparse.Namespace) -> str | None:
    return "folders" if args.labels_from_folders else "jsonl" if args.labels_jsonl else "csv" if args.labels_csv else None


def annotations_command(args: argparse.Namespace) -> int:
    tools = find_tools(args.build_dir, (TOOL_NAMES[1],))
    result = subprocess.run([str(tools[TOOL_NAMES[1]]), "--inspect-manifest", str(args.store / "manifest.bin")],
                            capture_output=True, text=True)
    if result.returncode:
        raise ConversionError(f"native manifest inspection failed: {result.stderr.strip()}; rebuild {TOOL_NAMES[1]}")
    count = json.loads(result.stdout)["image_count"]
    if args.annotation_command == "build":
        destination = args.store / annotations.DIRECTORY
        if destination.exists() or destination.is_symlink():
            raise ConversionError(f"refusing to replace existing annotations: {destination}")
        paths = annotations.physical_paths(args.store, count)
        records, classes = annotations.prepare_records(
            paths, image_root=args.image_root, labels_from_folders=args.labels_from_folders,
            labels_jsonl=args.labels_jsonl, labels_csv=args.labels_csv, class_map=args.class_map,
        )
        if args.dry_run:
            print(f"Validated {len(records)} annotations; dry-run wrote no files.")
            return 0
        annotations.publish_annotations(args.store, records, classes, label_source=annotation_source(args))
        print(f"Published {len(records)} annotations in {destination}; existing image data was not rewritten.")
    else:
        _, metadata = annotations.load_annotations(args.store)
        if metadata["image_count"] != count:
            raise ConversionError("annotation image count disagrees with native manifest")
        print(json.dumps(metadata, indent=2) if args.json else
              f"Compatible {metadata['schema_version']}: {count} images; manifest, sample order and annotations verified.")
    return 0


def stored_bytes(store: Path) -> int:
    return sum(path.stat().st_size for path in store.rglob("*") if path.is_file())


def build_info(tools: dict[str, Path]) -> dict:
    git = shutil.which("git")
    revision = None
    dirty = None
    if git:
        result = subprocess.run([git, "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            revision = result.stdout.strip()
            result = subprocess.run([git, "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True)
            if result.returncode == 0:
                dirty = bool(result.stdout)
    return {
        "cli_source_revision": revision, "cli_source_dirty": dirty,
        "python": platform.python_version(), "platform": platform.platform(),
        "native_tools": {name: {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in tools.items()},
    }


def write_report(path: Path, report: dict, other_bytes: int) -> str:
    # Include the report itself in the measured file-byte total. Only the JSON
    # length can change here; no dataset files are rewritten or duplicated.
    report["total_output_bytes"] = other_bytes
    while True:
        report["source_to_output_ratio"] = report["original_jpeg_bytes"] / report["total_output_bytes"]
        content = json.dumps(report, indent=2, sort_keys=True) + "\n"
        total = other_bytes + len(content.encode("utf-8"))
        if total <= report["total_output_bytes"]:
            # Number formatting can shorten the JSON on a later pass. Retain
            # that space as legal JSON whitespace, so the size converges.
            content += " " * (report["total_output_bytes"] - total)
            path.write_text(content, encoding="utf-8")
            return content
        report["total_output_bytes"] = total


def convert(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = parallel_config(args)
    if args.partition_images > 65535:
        raise ConversionError("--partition-images must be in [1, 65535] for block-major access v1")
    output = args.output.absolute()
    check_destination(output)
    inputs, source_bytes, ordering = discover_inputs(args.input_dir, args.input_list)
    label_source = annotation_source(args)
    if args.class_map is not None and not args.labels_from_folders:
        raise ConversionError("--class-map requires --labels-from-folders")
    records, classes, shuffle = None, None, None
    if label_source is not None or args.shuffle_seed is not None:
        records, classes = annotations.prepare_records(
            inputs, image_root=args.image_root or args.input_dir, labels_from_folders=args.labels_from_folders,
            labels_jsonl=args.labels_jsonl, labels_csv=args.labels_csv, class_map=args.class_map,
        )
        if args.shuffle_seed is not None:
            annotations.shuffle_records(records, args.shuffle_seed)
            shuffle = {"algorithm": annotations.SHUFFLE_ALGORITHM, "seed": args.shuffle_seed}
            ordering += f"; then {annotations.SHUFFLE_ALGORITHM}, seed={args.shuffle_seed}, before partitioning"
        inputs = [record.source_path for record in records]
    tools = find_tools(args.build_dir)
    extra_report = args.report.absolute() if args.report else None
    if extra_report == output / REPORT_NAME:
        extra_report = None
    if extra_report is not None:
        if extra_report.resolve().is_relative_to(output.resolve()):
            raise ConversionError(f"--report inside the output must be {output / REPORT_NAME}")
        if extra_report.exists() or extra_report.is_symlink():
            raise ConversionError(f"refusing to replace existing --report path: {extra_report}")
        if not extra_report.parent.is_dir():
            raise ConversionError(f"--report parent directory does not exist: {extra_report.parent}")
    print(f"Discovered {len(inputs)} JPEGs ({source_bytes:,} bytes); image IDs follow {ordering}.", flush=True)
    print("Parallel configuration: " + ", ".join(f"{key}={value}" for key, value in config.items()), flush=True)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "output": str(output), "image_count": len(inputs),
                          "physical_layout": PHYSICAL_LAYOUT, "partition_images": args.partition_images,
                          "preset": args.preset, "verify": args.verify, "parallelism": config,
                          "annotations": records is not None, "shuffle": shuffle,
                          "tools": {name: str(path) for name, path in tools.items()}}, indent=2))
        return 0
    times = {"discovery_preflight_seconds": time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    # This must be a sibling, not a system temporary directory: publication is
    # one same-filesystem directory rename, including the native generation.
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.fastlens-staging-", dir=output.parent))
    exported = False
    try:
        (stage / "conversion").mkdir()
        ordered_list = stage / "conversion/sources.txt"
        ordered_list.write_text("".join(f"{path}\n" for path in inputs), encoding="utf-8")
        log = stage / ".native.log"
        jpeg = str(tools[TOOL_NAMES[0]])
        command = [jpeg, "--shard", "--out-dir", str(stage), "--physical-layout", "spatial-major",
                   "--metadata-profile", "reconstruct", "--preset", args.preset,
                   "--shard-images", str(args.partition_images), "--input-list", str(ordered_list)]
        for key, value in config.items():
            if key != "verify_workers":
                command.extend(["--" + key.replace("_", "-"), str(value)])
        phase = time.perf_counter()
        run_native(command, log, "Compression")
        times["compression_seconds"] = time.perf_counter() - phase
        phase = time.perf_counter()
        run_native([str(tools[TOOL_NAMES[1]]), "--finalize-store", str(stage / "manifest.bin")], log, "Store layout")
        times["layout_finalization_seconds"] = time.perf_counter() - phase
        phase = time.perf_counter()
        run_native([str(tools[TOOL_NAMES[1]]), str(stage / "manifest.bin"), "--output-dir", str(stage / "geometry")], log, "Geometry index")
        info = native_inspect(stage, tools[TOOL_NAMES[1]])
        if (info["manifest_version"] != 1 or info["physical_layout"] != PHYSICAL_LAYOUT
                or not info["index"]["compatible"] or info["image_count"] != len(inputs)):
            raise ConversionError("native output is not the requested complete block-major dataset")
        if info["jpeg_warning_image_ids"]:
            bad = inputs[info["jpeg_warning_image_ids"][0]]
            raise ConversionError(f"invalid JPEG: libjpeg reported corruption/recovery warnings for {bad}; repair or remove the source and retry")
        times["index_seconds"] = time.perf_counter() - phase
        times["verification_seconds"] = 0.0
        if args.verify == "full":
            phase = time.perf_counter()
            run_native([jpeg, "--verify-manifest", str(stage / "manifest.bin"), "--verify-workers", str(config["verify_workers"]),
                        "--input-list", str(ordered_list)], log, "Exact DCT verification")
            times["verification_seconds"] = time.perf_counter() - phase
        log.unlink()
        # Preserve the native input order independently of annotations and the
        # removable conversion records, including stores labelled later.
        source_order_sha256 = annotations.file_sha256(ordered_list)
        (stage / annotations.SOURCE_ORDER).write_text(source_order_sha256 + "\n", encoding="ascii")
        if records is not None:
            phase = time.perf_counter()
            annotations.publish_annotations(stage, records, classes, label_source=label_source, shuffle=shuffle)
            times["annotation_seconds"] = time.perf_counter() - phase
        report = {
            "schema_version": 1, "input_image_count": len(inputs), "original_jpeg_bytes": source_bytes,
            "compressed_payload_bytes": info["coefficient_bytes"], "metadata_bytes": info["metadata_bytes"],
            "index_bytes": info["index_bytes"], "manifest_bytes": info["manifest_bytes"],
            "source_paths_bytes": ordered_list.stat().st_size,
            "source_paths_sha256": source_order_sha256, "shard_count": info["shard_count"],
            "manifest_version": info["manifest_version"], "index_version": info["index"]["version"],
            "physical_layout": info["physical_layout"], "metadata_profile": "reconstruct",
            "partition_images": args.partition_images, "preset": args.preset,
            "rowgroup_vectors": info["rowgroup_vectors"], "rowgroups_per_shard": info["rowgroups_per_shard"],
            "workers": args.workers, "parallelism": config, "wall_times": times,
            "verification": "passed" if args.verify == "full" else "not_requested",
            "source_ordering_policy": ordering, "build": build_info(tools),
        }
        if records is not None:
            report["annotations"] = annotations.DIRECTORY
        times["total_seconds"] = time.perf_counter() - started
        report["wall_time_scope"] = "through completion of pipeline and inspection, before report serialization and atomic publication"
        content = write_report(stage / REPORT_NAME, report, stored_bytes(stage))
        if extra_report is not None:
            with extra_report.open("x", encoding="utf-8") as stream:
                exported = True
                stream.write(content)
        check_destination(output)
        stage.rename(output)  # POSIX rename refuses to replace a nonempty directory.
    except BaseException:
        if exported:
            extra_report.unlink()
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Published {output}: {len(inputs)} images, {report['shard_count']} partitions, "
          f"{report['total_output_bytes']:,} bytes, source/output={report['source_to_output_ratio']:.3f}, "
          f"verification={report['verification']}, {times['total_seconds']:.2f}s.")
    return 0


def inspect_store(args: argparse.Namespace) -> int:
    tools = find_tools(args.build_dir, (TOOL_NAMES[1],))
    info = native_inspect(args.store, tools[TOOL_NAMES[1]])
    info["total_stored_bytes"] = stored_bytes(args.store)
    info["core_storage_bytes"] = info["manifest_bytes"] + stored_bytes(args.store / "partitions") + stored_bytes(args.store / "geometry")
    info["annotation_bytes"] = stored_bytes(args.store / annotations.DIRECTORY)
    info["conversion_record_bytes"] = stored_bytes(args.store / "conversion")
    report_path = args.store / REPORT_NAME
    source_bytes = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise ConversionError(f"unsupported conversion report schema: {report_path}")
        source_bytes = report.get("original_jpeg_bytes")
        if type(source_bytes) is not int or source_bytes < 0:
            raise ConversionError(f"invalid original_jpeg_bytes in {report_path}")
    info["original_jpeg_bytes"] = source_bytes
    info["source_to_output_ratio"] = source_bytes / info["total_stored_bytes"] if source_bytes is not None else None
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print(f"Manifest v{info['manifest_version']}; layout={info['physical_layout']}")
        print(f"Images: {info['image_count']}; partitions: {info['shard_count']}")
        print(f"Coefficients: {info['coefficient_file_count']} files, {info['coefficient_bytes']:,} bytes; "
              f"metadata: {info['metadata_file_count']} files, {info['metadata_bytes']:,} bytes")
        print(f"Geometry index: {'compatible v1' if info['index']['compatible'] else 'absent'}; "
              f"{info['index_file_count']} files, {info['index_bytes']:,} bytes")
        print(f"Core storage: {info['core_storage_bytes']:,} bytes; annotations: {info['annotation_bytes']:,} bytes; "
              f"conversion records: {info['conversion_record_bytes']:,} bytes")
        print("Check scope: metadata and index compatibility; full FLS payload integrity is not checked.")
        print(f"Total stored: {info['total_stored_bytes']:,} bytes; "
              + (f"source JPEGs: {source_bytes:,} bytes; source/output={info['source_to_output_ratio']:.3f}" if source_bytes is not None else "source JPEG bytes: unknown"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "annotations":
            return annotations_command(args)
        return convert(args) if args.command == "convert" else inspect_store(args)
    except (ConversionError, OSError, ValueError) as error:
        print(f"fastlens: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("fastlens: interrupted; staging data removed", file=sys.stderr)
        return 130
