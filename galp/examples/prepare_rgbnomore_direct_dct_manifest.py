#!/usr/bin/env python3
"""Prepare a GALP Direct-DCT manifest compatible with RGB-no-more JPEG-Ti.

The RGB-no-more DCT model expects dequantized JPEG DCT coefficients, so the
GALP sharded manifest must be generated with reconstructable metadata. This
wrapper invokes galp_jpeg_dct_tool with --metadata-profile reconstruct and then
performs a CPU-only metadata validation through the Torch binding.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("/tmp/rgbnomore_imagenet/val")
DEFAULT_OUT_DIR = Path("galp/data/imagedataset_dct/ImageNet-val-rg8")
DEFAULT_DATA_ROOT = Path("/tmp/rgbnomore_imagenet")
DEFAULT_TOOL = Path("build/galp/tools/jpeg_dct/galp_jpeg_dct_tool")
DEFAULT_TORCH_BINDING_DIR = Path("build/galp/torch")
DEFAULT_SPLIT_IMAGE_COUNTS = {
    "val": 50000,
    "inference": 50000,
    "train": 1281167,
}


def _split_dir_name(split: str) -> str:
    return "val" if split == "inference" else split


def _default_input_dir(data_root: Path, split: str) -> Path:
    return data_root / _split_dir_name(split)


def _default_out_dir(split: str) -> Path:
    if split == "val":
        return DEFAULT_OUT_DIR
    if split == "train":
        return Path("galp/data/imagedataset_dct/ImageNet-train-multi")
    return Path("galp/data/imagedataset_dct/ImageNet-val-rg8")


def _default_expected_count(split: str, limit: int | None) -> int:
    if limit is not None:
        return limit
    return DEFAULT_SPLIT_IMAGE_COUNTS[split]


def _is_jpeg(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".jpe"}


def _collect_jpegs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and _is_jpeg(path))


def _load_label_index(index_file: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with index_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "Filepath" not in reader.fieldnames or "Label" not in reader.fieldnames:
            raise RuntimeError(f"{index_file} must contain Filepath and Label columns")
        for row in reader:
            labels[str(row["Filepath"]).replace("\\", "/")] = int(row["Label"])
    if not labels:
        raise RuntimeError(f"{index_file} did not contain any image labels")
    return labels


def _imagenet_index_key_from_path(path: Path) -> str:
    normalized = path.as_posix()
    for marker in ("/train/", "/val/"):
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return normalized[marker_index + 1 :]
    if normalized.startswith("train/") or normalized.startswith("val/"):
        return normalized
    raise RuntimeError(f"cannot derive RGB-no-more index key from image path: {path}")


def _write_label_map(
    paths: list[Path],
    index_file: Path,
    output_json: Path,
    *,
    split: str,
    input_dir: Path,
    manifest: Path | None,
) -> dict[str, Any]:
    label_index = _load_label_index(index_file)
    labels: list[int] = []
    samples: list[dict[str, Any]] = []
    missing: list[str] = []
    for image_id, path in enumerate(paths):
        key = _imagenet_index_key_from_path(path)
        label = label_index.get(key)
        if label is None:
            missing.append(key)
            continue
        labels.append(int(label))
        if len(samples) < 8:
            samples.append({"image_id": image_id, "path": key, "label": int(label)})
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{len(missing)} input paths are missing from {index_file}: {preview}")
    payload = {
        "format": "galp_rgbnomore_label_map_v1",
        "split": split,
        "input_dir": str(input_dir),
        "index_file": str(index_file),
        "manifest": str(manifest) if manifest is not None else None,
        "image_count": len(labels),
        "labels": labels,
        "samples": samples,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "label_map_json": str(output_json),
        "label_count": len(labels),
        "index_file": str(index_file),
        "samples": samples,
    }


def _validation_summary(
    manifest: Path,
    torch_binding_dir: Path,
    sample_images: int,
    expected_image_count: int | None,
) -> dict[str, Any]:
    binding_path = str(torch_binding_dir.resolve())
    if binding_path not in sys.path:
        sys.path.insert(0, binding_path)
    import _galp_direct_dct as galp_dct  # pylint: disable=import-error,import-outside-toplevel

    reader = galp_dct.DirectDctReader(str(manifest))
    image_count = int(reader.image_count)
    inspected = min(sample_images, image_count)
    images = []
    component_quant_tables_available = True
    saw_referenced_quant_table = False
    saw_persisted_quant_table = False
    for image_id in range(inspected):
        metadata = reader.image_metadata(image_id)
        table_ids = {int(table["table_id"]) for table in metadata.get("quant_tables", [])}
        saw_persisted_quant_table = saw_persisted_quant_table or bool(table_ids)
        components = []
        image_ok = True
        for component in metadata.get("components", []):
            slot = int(component.get("semantic_slot_id", -1))
            quant_tbl_no = int(component.get("quant_tbl_no", -1))
            if slot in (0, 1, 2) and quant_tbl_no >= 0:
                saw_referenced_quant_table = True
            if slot in (0, 1, 2) and quant_tbl_no not in table_ids:
                image_ok = False
            components.append(
                {
                    "semantic_slot_id": slot,
                    "width_in_blocks": int(component.get("width_in_blocks", 0)),
                    "height_in_blocks": int(component.get("height_in_blocks", 0)),
                    "h_samp_factor": int(component.get("h_samp_factor", 0)),
                    "v_samp_factor": int(component.get("v_samp_factor", 0)),
                    "quant_tbl_no": quant_tbl_no,
                }
            )
        component_quant_tables_available = component_quant_tables_available and image_ok
        images.append(
            {
                "image_id": image_id,
                "image_width": int(metadata.get("image_width", 0)),
                "image_height": int(metadata.get("image_height", 0)),
                "quant_table_ids": sorted(table_ids),
                "component_quant_tables_available": image_ok,
                "components": components,
            }
        )

    summary = {
        "manifest": str(manifest),
        "image_count": image_count,
        "expected_image_count": expected_image_count,
        "image_count_matches_expected": expected_image_count is None or image_count == expected_image_count,
        "inspected_images": inspected,
        "component_quant_tables_available": component_quant_tables_available,
        "images": images,
    }
    if not component_quant_tables_available:
        summary["likely_metadata_profile"] = "dct" if saw_referenced_quant_table and not saw_persisted_quant_table else "unknown"
        summary["regenerate_hint"] = (
            "Regenerate the GALP manifest with --metadata-profile reconstruct; "
            "RGB-no-more DCT input requires persisted JPEG quantization table values."
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a reconstructable GALP Direct-DCT ImageNet manifest")
    parser.add_argument("--split", choices=("val", "train", "inference"), default="val")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-dir", type=Path, help="ImageFolder split directory. Defaults to DATA_ROOT/SPLIT, with inference mapped to val.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for reconstructable GALP shards.")
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_TORCH_BINDING_DIR)
    parser.add_argument("--preset", choices=("crop-latency", "balanced", "throughput"), default="balanced")
    parser.add_argument("--policy", choices=("ragged",), default="ragged")
    parser.add_argument("--shard-images", type=int, default=8192)
    parser.add_argument("--rowgroup-vectors", type=int, default=128)
    parser.add_argument("--rowgroups-per-shard", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Use the first N JPEGs after deterministic path sorting.")
    parser.add_argument("--validate-sample-images", type=int, default=8)
    parser.add_argument("--manifest", type=Path, help="Existing manifest.bin to validate with --verify-only.")
    parser.add_argument("--expected-image-count", type=int, help="Assert the manifest image_count matches this value.")
    parser.add_argument("--index-file", type=Path, help="RGB-no-more index CSV used to generate a GALP image_id -> label sidecar.")
    parser.add_argument("--label-map-json", type=Path, help="Output path for the GALP image_id -> label sidecar. Defaults to OUT_DIR/labels.json when --index-file is set.")
    parser.add_argument("--write-label-map-only", action="store_true", help="Only write the label sidecar from the selected input paths and index file; do not generate or verify shards.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory before writing.")
    parser.add_argument("--verify-only", action="store_true", help="Only validate an existing manifest; do not generate shards.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _emit_summary(summary: dict[str, Any], output_json: Path | None) -> None:
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    explicit_manifest = args.manifest is not None
    if args.input_dir is None:
        args.input_dir = _default_input_dir(args.data_root, args.split)
    if args.out_dir is None:
        args.out_dir = _default_out_dir(args.split)
    if args.label_map_json is None and args.index_file is not None:
        args.label_map_json = args.out_dir / "labels.json"
    if args.expected_image_count is None and not (args.verify_only and explicit_manifest and args.limit is None):
        args.expected_image_count = _default_expected_count(args.split, args.limit)
    if args.write_label_map_only:
        if args.index_file is None:
            raise ValueError("--write-label-map-only requires --index-file")
        if not args.input_dir.exists():
            raise FileNotFoundError(args.input_dir)
        if not args.index_file.exists():
            raise FileNotFoundError(args.index_file)
        paths = _collect_jpegs(args.input_dir)
        if args.limit is not None:
            if len(paths) < args.limit:
                raise RuntimeError(f"requested --limit {args.limit}, but only found {len(paths)} JPEG files")
            paths = paths[: args.limit]
        if args.expected_image_count is not None and len(paths) != args.expected_image_count:
            raise RuntimeError(f"label path count {len(paths)} does not match expected {args.expected_image_count}")
        summary = {
            "ok": True,
            "write_label_map_only": True,
            "split": args.split,
            "input_dir": str(args.input_dir),
            "expected_image_count": args.expected_image_count,
            **_write_label_map(
                paths,
                args.index_file,
                args.label_map_json,
                split=args.split,
                input_dir=args.input_dir,
                manifest=args.manifest or (args.out_dir / "manifest.bin"),
            ),
        }
        _emit_summary(summary, args.output_json)
        return
    if args.verify_only:
        manifest = args.manifest or (args.out_dir / "manifest.bin")
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        if args.validate_sample_images <= 0:
            raise ValueError("--validate-sample-images must be positive with --verify-only")
        if args.expected_image_count is not None and args.expected_image_count < 0:
            raise ValueError("--expected-image-count must be non-negative")
        summary = {
            "verify_only": True,
            "metadata_validation": _validation_summary(
                manifest,
                args.torch_binding_dir,
                args.validate_sample_images,
                args.expected_image_count,
            ),
        }
        validation = summary["metadata_validation"]
        failures = []
        if not validation["image_count_matches_expected"]:
            failures.append(
                f"manifest image_count {validation['image_count']} does not match expected "
                f"{validation['expected_image_count']}"
            )
        if not validation["component_quant_tables_available"]:
            failures.append("manifest is missing component quantization tables")
        summary["ok"] = not failures
        summary["failures"] = failures
        _emit_summary(summary, args.output_json)
        if failures:
            raise SystemExit(1)
        return

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)
    if not args.tool.exists():
        raise FileNotFoundError(args.tool)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.validate_sample_images < 0:
        raise ValueError("--validate-sample-images must be non-negative")
    for name in ("shard_images", "rowgroup_vectors", "rowgroups_per_shard"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out_dir} already exists; pass --overwrite to replace it")
        if not args.dry_run:
            shutil.rmtree(args.out_dir)

    input_args: list[str]
    input_count: int | None = None
    selected_paths_for_label_map: list[Path] | None = None
    if args.limit is None:
        input_args = [str(args.input_dir)]
        if args.index_file is not None:
            selected_paths_for_label_map = _collect_jpegs(args.input_dir)
            input_count = len(selected_paths_for_label_map)
    else:
        paths = _collect_jpegs(args.input_dir)
        if len(paths) < args.limit:
            raise RuntimeError(f"requested --limit {args.limit}, but only found {len(paths)} JPEG files")
        selected = paths[: args.limit]
        input_count = len(selected)
        selected_paths_for_label_map = selected
        input_args = [str(path) for path in selected]

    command = [
        str(args.tool),
        "--shard",
        "--out-dir",
        str(args.out_dir),
        "--policy",
        args.policy,
        "--preset",
        args.preset,
        "--metadata-profile",
        "reconstruct",
        "--shard-images",
        str(args.shard_images),
        "--rowgroup-vectors",
        str(args.rowgroup_vectors),
        "--rowgroups-per-shard",
        str(args.rowgroups_per_shard),
        *input_args,
    ]
    summary: dict[str, Any] = {
        "command": command,
        "input_dir": str(args.input_dir),
        "out_dir": str(args.out_dir),
        "split": args.split,
        "data_root": str(args.data_root),
        "expected_image_count": args.expected_image_count,
        "limit": args.limit,
        "input_count": input_count,
        "metadata_profile": "reconstruct",
        "dry_run": args.dry_run,
    }
    if args.index_file is not None:
        if not args.index_file.exists():
            raise FileNotFoundError(args.index_file)
        if selected_paths_for_label_map is None:
            selected_paths_for_label_map = _collect_jpegs(args.input_dir)
        if args.expected_image_count is not None and len(selected_paths_for_label_map) != args.expected_image_count:
            raise RuntimeError(
                f"label path count {len(selected_paths_for_label_map)} does not match expected {args.expected_image_count}"
            )
    print("COMMAND " + " ".join(command))
    if not args.dry_run:
        started = time.perf_counter()
        subprocess.run(command, check=True)
        summary["seconds"] = time.perf_counter() - started
        manifest = args.out_dir / "manifest.bin"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        if args.validate_sample_images > 0:
            summary["metadata_validation"] = _validation_summary(
                manifest,
                args.torch_binding_dir,
                args.validate_sample_images,
                args.expected_image_count,
            )
            if not summary["metadata_validation"]["image_count_matches_expected"]:
                validation = summary["metadata_validation"]
                raise RuntimeError(
                    f"generated manifest image_count {validation['image_count']} does not match expected "
                    f"{validation['expected_image_count']}"
                )
            if not summary["metadata_validation"]["component_quant_tables_available"]:
                raise RuntimeError("generated manifest is missing component quantization tables")
        if args.index_file is not None:
            summary["label_map"] = _write_label_map(
                selected_paths_for_label_map,
                args.index_file,
                args.label_map_json,
                split=args.split,
                input_dir=args.input_dir,
                manifest=manifest,
            )

    _emit_summary(summary, args.output_json)


if __name__ == "__main__":
    main()
