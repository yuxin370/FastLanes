#!/usr/bin/env python3
"""Build the RGB-no-more 512x512 ImageNet-train compact-v3 dataset.

The published RGB-no-more JPEG checkpoints use this source-domain recipe:

    JPEG decode -> RGB -> 512x512 bilinear resize -> JPEG re-encode -> DCT

The source available on the benchmark host is one uncompressed tar archive per
ImageNet class.  This program converts those archives directly into the
canonical ``train/<wnid>/<image>.JPEG`` tree, so it does not need a second copy
of the native-size extracted dataset.  JPEG conversion is resumable.  Compact
v3 output is deliberately never overwritten or resumed in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Sequence


EXPECTED_TRAIN_IMAGES = 1_281_167
EXPECTED_CLASSES = 1_000
JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe"}
DEFAULT_LAYOUT_THREADS = 32
DEFAULT_SHARD_DECODE_THREADS = 4


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pillow_runtime() -> dict[str, str | None]:
    from PIL import __version__ as pillow_version
    from PIL import features

    return {
        "pillow_version": str(pillow_version),
        "jpeg_codec_version": features.version_codec("jpg"),
        "jpeg_encoder_options": "Pillow defaults",
        "resize_resample": "PIL.Image.Resampling.BILINEAR",
    }


def _torch_binding_path(directory: Path) -> Path:
    candidates = sorted(directory.glob("_galp_direct_dct*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one _galp_direct_dct binding in {directory}, "
            f"found {candidates}"
        )
    return candidates[0]


def _assert_runtime_unchanged(args: argparse.Namespace, plan: dict[str, object]) -> None:
    current_tool = _sha256(args.tool)
    current_binding = _sha256(_torch_binding_path(args.torch_binding_dir))
    if current_tool != plan["tool_sha256"]:
        raise RuntimeError(
            "galp_jpeg_dct_tool changed during preparation; refusing to mix "
            f"producer/verifier binaries ({plan['tool_sha256']} -> {current_tool})"
        )
    if current_binding != plan["torch_binding_sha256"]:
        raise RuntimeError(
            "GALP Torch binding changed during preparation; refusing to mix "
            f"reader binaries ({plan['torch_binding_sha256']} -> {current_binding})"
        )


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def _resolve_parallelism(args: argparse.Namespace) -> None:
    legacy = args.compress_threads
    if legacy is not None:
        if args.layout_threads is not None and args.layout_threads != legacy:
            raise ValueError("--compress-threads conflicts with --layout-threads")
        if (
            args.shard_decode_threads is not None
            and args.shard_decode_threads != legacy
        ):
            raise ValueError(
                "--compress-threads conflicts with --shard-decode-threads"
            )
        args.layout_threads = legacy
        args.shard_decode_threads = legacy
    else:
        if args.layout_threads is None:
            args.layout_threads = DEFAULT_LAYOUT_THREADS
        if args.shard_decode_threads is None:
            args.shard_decode_threads = DEFAULT_SHARD_DECODE_THREADS
    args.legacy_compress_threads = legacy


def _compress_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.tool),
        "--shard",
        "--out-dir",
        str(args.dct_root),
        "--policy",
        "ragged",
        "--preset",
        "random-access",
        "--metadata-profile",
        "reconstruct",
        "--physical-layout",
        "image-major-vector-rowgroups",
        "--spatial-order",
        "tiled-z-32",
        "--shard-images",
        str(args.shard_images),
        "--rowgroups-per-shard",
        str(args.rowgroups_per_shard),
        "--layout-threads",
        str(args.layout_threads),
        "--shard-decode-threads",
        str(args.shard_decode_threads),
        "--shard-workers",
        str(args.shard_workers),
        "--encoding-workers-per-shard",
        str(args.encoding_workers_per_shard),
        str(args.jpeg_train_root),
    ]


def _coefficient_verify_command(
    args: argparse.Namespace, manifest: Path
) -> list[str]:
    return [
        str(args.tool),
        "--verify-manifest",
        str(manifest),
        "--verify-workers",
        str(args.verify_workers),
        str(args.jpeg_train_root),
    ]


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    command = [str(item) for item in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"COMMAND {_command_text(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND {_command_text(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}; see {log_path}"
        )


def _run_to_file(command: Sequence[str], output_path: Path) -> None:
    command = [str(item) for item in command]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"COMMAND {_command_text(command)} > {output_path}", flush=True)
    with output_path.open("w", encoding="utf-8") as output:
        subprocess.run(command, check=True, stdout=output)


def _parse_index(
    index_csv: Path, expected_image_count: int
) -> dict[str, list[str]]:
    by_wnid: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    with index_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Filepath", "Label"}.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"{index_csv} must contain Filepath and Label columns; "
                f"got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            relative = PurePosixPath(str(row["Filepath"]).strip())
            if len(relative.parts) != 3 or relative.parts[0] != "train":
                raise ValueError(
                    f"{index_csv}:{row_number} is not train/<wnid>/<JPEG>: {relative}"
                )
            _, wnid, filename = relative.parts
            if not wnid.startswith("n") or Path(filename).suffix.lower() not in JPEG_SUFFIXES:
                raise ValueError(
                    f"{index_csv}:{row_number} has an invalid ImageNet path: {relative}"
                )
            normalized = relative.as_posix()
            if normalized in seen:
                raise ValueError(f"{index_csv}:{row_number} duplicates {normalized}")
            seen.add(normalized)
            label = int(row["Label"])
            if not 0 <= label < EXPECTED_CLASSES:
                raise ValueError(
                    f"{index_csv}:{row_number} has label outside [0,999]: {label}"
                )
            by_wnid[wnid].append(filename)

    if len(seen) != expected_image_count:
        raise RuntimeError(
            f"index population mismatch: expected {expected_image_count}, got {len(seen)}"
        )
    if len(by_wnid) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"class population mismatch: expected {EXPECTED_CLASSES}, got {len(by_wnid)}"
        )
    return dict(by_wnid)


def _is_complete_512_jpeg(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "JPEG" or image.size != (512, 512):
                return False
            image.verify()
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def _process_archive(
    archive_text: str,
    filenames: Sequence[str],
    output_dir_text: str,
    create_missing: bool,
) -> dict[str, int | str]:
    """Validate/resume one class. Kept top-level for multiprocessing."""

    from PIL import Image

    archive = Path(archive_text)
    output_dir = Path(output_dir_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    reused = 0

    tar: tarfile.TarFile | None = None
    try:
        for filename in filenames:
            destination = output_dir / filename
            if _is_complete_512_jpeg(destination):
                reused += 1
                continue
            if not create_missing:
                raise RuntimeError(f"missing or invalid 512x512 JPEG: {destination}")
            if tar is None:
                tar = tarfile.open(archive, mode="r:")
            try:
                member = tar.getmember(filename)
            except KeyError as error:
                raise RuntimeError(f"{archive} does not contain {filename}") from error
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read {filename} from {archive}")
            temporary = destination.with_name(
                f".{destination.name}.tmp-{os.getpid()}"
            )
            try:
                with source, Image.open(source) as image:
                    rgb = image.convert("RGB")
                    resized = rgb.resize(
                        (512, 512), resample=Image.Resampling.BILINEAR
                    )
                    # Explicit format is necessary because the atomic temporary
                    # name no longer ends in .JPEG.  All encoder options remain
                    # Pillow defaults, matching RGB-no-more ImageResizer.save().
                    resized.save(temporary, format="JPEG")
                    resized.close()
                    rgb.close()
                if not _is_complete_512_jpeg(temporary):
                    raise RuntimeError(f"invalid JPEG produced for {destination}")
                os.replace(temporary, destination)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            converted += 1
    finally:
        if tar is not None:
            tar.close()
    return {
        "class": archive.stem,
        "images": len(filenames),
        "converted": converted,
        "reused": reused,
    }


def _prepare_jpegs(
    *,
    class_tar_root: Path,
    jpeg_train_root: Path,
    by_wnid: dict[str, list[str]],
    workers: int,
    create_missing: bool,
) -> dict[str, int]:
    tasks: list[tuple[Path, list[str], Path]] = []
    for wnid, filenames in sorted(by_wnid.items()):
        archive = class_tar_root / f"{wnid}.tar"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        tasks.append((archive, filenames, jpeg_train_root / wnid))

    jpeg_train_root.mkdir(parents=True, exist_ok=True)
    totals = {"images": 0, "converted": 0, "reused": 0}
    action = "resize/resume" if create_missing else "validate"
    print(
        f"Starting {action}: classes={len(tasks)} images="
        f"{sum(len(item[1]) for item in tasks)} workers={workers}",
        flush=True,
    )
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_archive,
                str(archive),
                filenames,
                str(output_dir),
                create_missing,
            ): archive.stem
            for archive, filenames, output_dir in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            for key in totals:
                totals[key] += int(result[key])
            if completed % 25 == 0 or completed == len(futures):
                elapsed = time.monotonic() - started
                print(
                    f"classes={completed}/{len(futures)} images={totals['images']} "
                    f"converted={totals['converted']} reused={totals['reused']} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    actual_jpegs = sum(
        1
        for directory, _, filenames in os.walk(jpeg_train_root)
        for filename in filenames
        if Path(directory, filename).suffix.lower() in JPEG_SUFFIXES
    )
    if actual_jpegs != totals["images"]:
        raise RuntimeError(
            f"output JPEG population mismatch: expected {totals['images']}, "
            f"found {actual_jpegs}; remove unexpected files before continuing"
        )
    return totals


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _compress(args: argparse.Namespace) -> Path:
    if args.dct_root.exists():
        raise FileExistsError(
            f"refusing to overwrite/resume compact-v3 output: {args.dct_root}. "
            "Use a new --dct-root, or inspect and remove the failed directory manually."
        )
    # Keep both properties explicit.  The strict preflight below independently
    # proves that argument normalization preserved manifest-v3 and tiled-z32.
    command = _compress_command(args)
    _run_logged(command, args.artifact_dir / "compress.log")
    manifest = args.dct_root / "manifest.bin"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return manifest


def _write_label_map(args: argparse.Namespace, manifest: Path) -> None:
    command = [
        sys.executable,
        "-B",
        str(args.prepare_dataset),
        "--split",
        "train",
        "--input-dir",
        str(args.jpeg_train_root),
        "--out-dir",
        str(args.dct_root),
        "--manifest",
        str(manifest),
        "--expected-image-size",
        "512",
        "--expected-image-count",
        str(args.expected_image_count),
        "--index-file",
        str(args.index_csv),
        "--selected-index-csv",
        str(args.dct_root / "index.csv"),
        "--label-map-json",
        str(args.dct_root / "labels.json"),
        "--write-label-map-only",
        "--output-json",
        str(args.artifact_dir / "source_and_labels.json"),
    ]
    _run_logged(command, args.artifact_dir / "source_and_labels.log")


def _verify(args: argparse.Namespace, manifest: Path) -> None:
    preflight = [
        sys.executable,
        "-B",
        str(args.manifest_preflight),
        str(manifest),
        "--expected-manifest-version",
        "3",
        "--expected-physical-layout",
        "image-major-vector-rowgroups",
        "--expected-spatial-order",
        "tiled-z32",
        "--expected-image-count",
        str(args.expected_image_count),
    ]
    _run_to_file(preflight, args.artifact_dir / "manifest_preflight.json")

    if not args.skip_coefficient_verify:
        verify_log = args.artifact_dir / "verify_manifest.log"
        _run_logged(_coefficient_verify_command(args, manifest), verify_log)
        lines = {
            line.strip()
            for line in verify_log.read_text(encoding="utf-8").splitlines()
        }
        if "exact: true" not in lines:
            raise RuntimeError(
                f"coefficient-exact verification did not report exact: true; see {verify_log}"
            )

    reader_verify = [
        sys.executable,
        "-B",
        str(args.prepare_dataset),
        "--verify-only",
        "--split",
        "train",
        "--input-dir",
        str(args.jpeg_train_root),
        "--out-dir",
        str(args.dct_root),
        "--manifest",
        str(manifest),
        "--expected-image-size",
        "512",
        "--expected-image-count",
        str(args.expected_image_count),
        "--validate-sample-images",
        str(args.validate_sample_images),
        "--torch-binding-dir",
        str(args.torch_binding_dir),
        "--output-json",
        str(args.artifact_dir / "reader_verify.json"),
    ]
    _run_logged(reader_verify, args.artifact_dir / "reader_verify.log")


def _generate_training_manifests(args: argparse.Namespace) -> None:
    train_manifest = args.dct_root / "manifest.bin"
    validation_manifest = args.validation_dct_root / "manifest.bin"
    for path in (train_manifest, validation_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for split, manifest, image_count, output_name in (
        (
            "train",
            train_manifest,
            args.expected_image_count,
            "manifest_preflight.json",
        ),
        (
            "validation",
            validation_manifest,
            args.expected_validation_image_count,
            "manifest_preflight_validation.json",
        ),
    ):
        preflight = [
            sys.executable,
            "-B",
            str(args.manifest_preflight),
            str(manifest),
            "--expected-manifest-version",
            "3",
            "--expected-physical-layout",
            "image-major-vector-rowgroups",
            "--expected-spatial-order",
            "tiled-z32",
            "--expected-image-count",
            str(image_count),
        ]
        print(f"Preflighting official {split} manifest", flush=True)
        _run_to_file(preflight, args.artifact_dir / output_name)
    command = [
        sys.executable,
        "-B",
        str(args.manifest_generator),
        "--jpeg-root",
        str(args.jpeg_train_root),
        "--index-csv",
        str(args.index_csv),
        "--galp-manifest",
        str(train_manifest),
        "--validation-jpeg-root",
        str(args.validation_jpeg_root),
        "--validation-index-csv",
        str(args.validation_index_csv),
        "--galp-validation-manifest",
        str(validation_manifest),
        "--output-dir",
        str(args.training_manifest_dir),
        "--train-count",
        "0",
        "--val-count",
        "0",
    ]
    if args.overwrite_training_manifests:
        command.append("--overwrite")
    _run_logged(command, args.artifact_dir / "training_manifests.log")


def _validate_arguments(args: argparse.Namespace) -> None:
    for path in (
        args.class_tar_root,
        args.index_csv,
        args.tool,
        args.prepare_dataset,
        args.manifest_preflight,
        args.manifest_generator,
        args.torch_binding_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not os.access(args.tool, os.X_OK):
        raise PermissionError(f"tool is not executable: {args.tool}")
    for name in (
        "resize_workers",
        "layout_threads",
        "shard_decode_threads",
        "shard_workers",
        "encoding_workers_per_shard",
        "verify_workers",
        "shard_images",
        "rowgroups_per_shard",
        "expected_image_count",
        "expected_validation_image_count",
        "validate_sample_images",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.verify_workers > 16:
        raise ValueError("--verify-workers must be at most 16")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    e2e = repo / "galp/data/system_rgbnomore/e2e_v3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "resize", "compress", "verify", "manifests"),
        default="all",
    )
    parser.add_argument(
        "--class-tar-root",
        type=Path,
        default=repo / "galp/data/imagedataset/ILSVRC2012_img_train",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/home/tangyuxin/RGB-no-more/assets/indexbase_train.csv"),
    )
    parser.add_argument(
        "--jpeg-train-root", type=Path, default=e2e / "imagenet_512/train"
    )
    parser.add_argument(
        "--dct-root",
        type=Path,
        default=e2e / "compact_v3_tiled_z32_rgbnomore512_train",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=e2e / "compact_v3_tiled_z32_rgbnomore512_train_prepare",
    )
    parser.add_argument(
        "--tool",
        type=Path,
        default=repo / "build/galp/tools/jpeg_dct/galp_jpeg_dct_tool",
    )
    parser.add_argument(
        "--torch-binding-dir", type=Path, default=repo / "build/galp/torch"
    )
    parser.add_argument(
        "--prepare-dataset",
        type=Path,
        default=repo
        / "galp/benchmarks/system_rgbnomore/dataset/prepare_dataset.py",
    )
    parser.add_argument(
        "--manifest-preflight",
        type=Path,
        default=repo
        / "galp/benchmarks/system_rgbnomore/training/manifest_preflight.py",
    )
    parser.add_argument(
        "--manifest-generator",
        type=Path,
        default=repo
        / "galp/benchmarks/system_rgbnomore/training/generate_imagenet_manifests.py",
    )
    parser.add_argument(
        "--validation-jpeg-root",
        type=Path,
        default=e2e / "imagenet_512/val",
    )
    parser.add_argument(
        "--validation-index-csv",
        type=Path,
        default=Path("/home/tangyuxin/RGB-no-more/assets/indexbase_val.csv"),
    )
    parser.add_argument(
        "--validation-dct-root",
        type=Path,
        default=e2e / "compact_v3_tiled_z32_rgbnomore512",
    )
    parser.add_argument(
        "--training-manifest-dir",
        type=Path,
        default=e2e / "training_manifests_official_v3",
    )
    parser.add_argument("--expected-image-count", type=int, default=EXPECTED_TRAIN_IMAGES)
    parser.add_argument("--expected-validation-image-count", type=int, default=50_000)
    parser.add_argument("--resize-workers", type=int, default=32)
    parser.add_argument(
        "--compress-threads",
        type=int,
        default=None,
        help=(
            "Legacy layout/decode control. Maps to both new controls and fails "
            "if an explicitly supplied new value conflicts."
        ),
    )
    parser.add_argument("--layout-threads", type=int, default=None)
    parser.add_argument("--shard-decode-threads", type=int, default=None)
    parser.add_argument("--shard-workers", type=int, default=4)
    parser.add_argument("--encoding-workers-per-shard", type=int, default=1)
    parser.add_argument("--verify-workers", type=int, default=16)
    parser.add_argument("--shard-images", type=int, default=8192)
    parser.add_argument("--rowgroups-per-shard", type=int, default=8192)
    parser.add_argument("--validate-sample-images", type=int, default=32)
    parser.add_argument(
        "--skip-coefficient-verify",
        action="store_true",
        help="Skip the full source-vs-manifest coefficient check (not recommended).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/index and print the resolved plan without writing data.",
    )
    parser.add_argument("--overwrite-training-manifests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _resolve_parallelism(args)
    _validate_arguments(args)
    by_wnid = _parse_index(args.index_csv, args.expected_image_count)
    torch_binding = _torch_binding_path(args.torch_binding_dir)
    free_bytes = shutil.disk_usage(args.jpeg_train_root.parent).free
    plan = {
        "schema": "galp-rgbnomore-imagenet512-train-prepare-v1",
        "phase": args.phase,
        "semantic_recipe": (
            "jpeg_decode->rgb->bilinear_resize_512x512->"
            "pillow_default_jpeg_reencode->jpeg_dct->compact_v3"
        ),
        "image_runtime": _pillow_runtime(),
        "class_tar_root": str(args.class_tar_root.resolve()),
        "index_csv": str(args.index_csv.resolve()),
        "index_sha256": _sha256(args.index_csv),
        "expected_image_count": args.expected_image_count,
        "jpeg_train_root": str(args.jpeg_train_root.resolve()),
        "dct_root": str(args.dct_root.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "validation_jpeg_root": str(args.validation_jpeg_root.resolve()),
        "validation_index_csv": str(args.validation_index_csv.resolve()),
        "validation_dct_root": str(args.validation_dct_root.resolve()),
        "training_manifest_dir": str(args.training_manifest_dir.resolve()),
        "free_bytes_before": free_bytes,
        "free_gib_before": round(free_bytes / (1024**3), 2),
        "resize_workers": args.resize_workers,
        "legacy_compress_threads": args.legacy_compress_threads,
        "layout_threads": args.layout_threads,
        "shard_decode_threads": args.shard_decode_threads,
        "shard_workers": args.shard_workers,
        "encoding_workers_per_shard": args.encoding_workers_per_shard,
        "verify_workers": args.verify_workers,
        "shard_images": args.shard_images,
        "rowgroups_per_shard": args.rowgroups_per_shard,
        "manifest_version": 3,
        "physical_layout": "image-major-vector-rowgroups",
        "spatial_order": "tiled-z32",
        "tool": str(args.tool.resolve()),
        "tool_sha256": _sha256(args.tool),
        "torch_binding": str(torch_binding.resolve()),
        "torch_binding_sha256": _sha256(torch_binding),
        "preparation_script": str(Path(__file__).resolve()),
        "preparation_script_sha256": _sha256(Path(__file__).resolve()),
        "manifest_preflight_sha256": _sha256(args.manifest_preflight),
        "manifest_generator_sha256": _sha256(args.manifest_generator),
        "started_at": _utc_now(),
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "manifests":
        _generate_training_manifests(args)
        return 0
    plan_name = "verify_plan.json" if args.phase == "verify" else "plan.json"
    _write_json(args.artifact_dir / plan_name, plan)
    planned_manifest = args.dct_root / "manifest.bin"
    _write_json(
        args.artifact_dir / "commands.json",
        {
            "schema": "galp-rgbnomore-imagenet512-commands-v1",
            "effective_values": {
                "layout_threads": args.layout_threads,
                "shard_decode_threads": args.shard_decode_threads,
                "shard_workers": args.shard_workers,
                "encoding_workers_per_shard": args.encoding_workers_per_shard,
                "verify_workers": args.verify_workers,
            },
            "commands": {
                "compress": {
                    "argv": _compress_command(args),
                    "shell": _command_text(_compress_command(args)),
                },
                "coefficient_verify": {
                    "argv": _coefficient_verify_command(args, planned_manifest),
                    "shell": _command_text(
                        _coefficient_verify_command(args, planned_manifest)
                    ),
                },
            },
        },
    )

    if args.phase in {"all", "resize"}:
        resize_summary = _prepare_jpegs(
            class_tar_root=args.class_tar_root,
            jpeg_train_root=args.jpeg_train_root,
            by_wnid=by_wnid,
            workers=args.resize_workers,
            create_missing=True,
        )
        _write_json(
            args.artifact_dir / "resize_complete.json",
            {**resize_summary, "completed_at": _utc_now(), "size": [512, 512]},
        )
        if args.phase == "resize":
            return 0
    else:
        _prepare_jpegs(
            class_tar_root=args.class_tar_root,
            jpeg_train_root=args.jpeg_train_root,
            by_wnid=by_wnid,
            workers=args.resize_workers,
            create_missing=False,
        )

    if args.phase in {"all", "compress"}:
        manifest = _compress(args)
        _assert_runtime_unchanged(args, plan)
        _write_label_map(args, manifest)
        _verify(args, manifest)
    else:
        manifest = args.dct_root / "manifest.bin"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        _write_label_map(args, manifest)
        _verify(args, manifest)

    _assert_runtime_unchanged(args, plan)

    completion = {
        **plan,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "labels_json": str((args.dct_root / "labels.json").resolve()),
        "selected_index_csv": str((args.dct_root / "index.csv").resolve()),
        "coefficient_exact_verification": not args.skip_coefficient_verify,
        "completed_at": _utc_now(),
    }
    completion_name = "VERIFY_COMPLETE.json" if args.phase == "verify" else "COMPLETE.json"
    _write_json(args.artifact_dir / completion_name, completion)
    print(f"COMPLETE {args.artifact_dir / completion_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
