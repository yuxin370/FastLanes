#!/usr/bin/env python3
"""Scan GALP Direct-DCT manifests for RGB-no-more compatibility.

This is a cheap metadata-only helper. It does not run CUDA or decode batches;
it checks whether manifests exist, their image_count, and whether sampled
images expose component quantization tables needed by the RGB-no-more adapter.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TORCH_BINDING_DIR = REPO_ROOT / "build/galp/torch"
DEFAULT_VAL_IMAGES = 50000
DEFAULT_TRAIN_IMAGES = 1281167


def _load_binding(torch_binding_dir: Path) -> Any:
    binding_path = str(torch_binding_dir.resolve())
    if binding_path not in sys.path:
        sys.path.insert(0, binding_path)
    import _galp_direct_dct as galp_dct  # pylint: disable=import-error,import-outside-toplevel

    return galp_dct


def _candidate_manifests(paths: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        if path.is_file():
            manifests.append(path)
        elif path.is_dir():
            manifest = path / "manifest.bin"
            if manifest.exists():
                manifests.append(manifest)
            else:
                nested = sorted(path.glob("*/manifest.bin"))
                if nested:
                    manifests.extend(nested)
                else:
                    manifests.append(manifest)
        else:
            manifests.append(path if path.suffix == ".bin" else path / "manifest.bin")
    return manifests


def _directory_status(manifest: Path) -> dict[str, Any]:
    directory = manifest.parent
    shard_files = sorted(directory.glob("shard_*.fls")) if directory.exists() else []
    meta_files = sorted(directory.glob("shard_*.meta.bin")) if directory.exists() else []
    latest: Path | None = None
    latest_mtime = 0.0
    for path in [*shard_files, *meta_files, manifest]:
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if latest is None or mtime > latest_mtime:
            latest = path
            latest_mtime = mtime
    return {
        "directory": str(directory),
        "shard_count": len(shard_files),
        "meta_shard_count": len(meta_files),
        "latest_path": str(latest) if latest is not None else None,
        "latest_mtime": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds") if latest is not None else None,
    }


def _expected_image_count_for_manifest(
    manifest: Path,
    expected_images: int | None,
    expected_val_images: int | None,
    expected_train_images: int | None,
) -> int | None:
    if expected_images is not None:
        return expected_images
    name = manifest.parent.name.lower()
    if "val" in name or "inference" in name:
        return expected_val_images
    if "train" in name:
        return expected_train_images
    return None


def _inspect_label_map(manifest: Path, expected_image_count: int | None) -> dict[str, Any]:
    label_map = manifest.parent / "labels.json"
    result: dict[str, Any] = {
        "label_map_json": str(label_map),
        "label_map_exists": label_map.exists(),
        "label_map_ok": False,
    }
    if not label_map.exists():
        result["label_map_failures"] = ["label map does not exist"]
        return result
    try:
        payload = json.loads(label_map.read_text(encoding="utf-8"))
        labels = payload.get("labels") if isinstance(payload, dict) else None
        failures = []
        if not isinstance(payload, dict) or payload.get("format") != "galp_rgbnomore_label_map_v1":
            failures.append("unexpected label map format")
        if not isinstance(labels, list) or not all(isinstance(label, int) and 0 <= label < 1000 for label in labels):
            failures.append("labels must be ImageNet-1K integer labels")
        label_count = len(labels) if isinstance(labels, list) else None
        if expected_image_count is not None:
            if int(payload.get("image_count", -1)) != expected_image_count:
                failures.append(f"label map image_count does not match expected {expected_image_count}")
            if label_count != expected_image_count:
                failures.append(f"label count {label_count} does not match expected {expected_image_count}")
        result.update(
            {
                "label_map_format": payload.get("format") if isinstance(payload, dict) else None,
                "label_map_image_count": payload.get("image_count") if isinstance(payload, dict) else None,
                "label_count": label_count,
                "label_map_failures": failures,
                "label_map_ok": not failures,
            }
        )
    except Exception as exc:  # noqa: BLE001 - scanner should preserve other diagnostics.
        result["label_map_failures"] = [str(exc)]
    return result


def _inspect_manifest(
    galp_dct: Any,
    manifest: Path,
    sample_images: int,
    expected_images: int | None,
    expected_val_images: int | None,
    expected_train_images: int | None,
    check_label_map: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "manifest": str(manifest),
        "exists": manifest.exists(),
        "ok": False,
        **_directory_status(manifest),
    }
    expected_image_count = _expected_image_count_for_manifest(
        manifest,
        expected_images,
        expected_val_images,
        expected_train_images,
    )
    if expected_image_count is not None:
        result["expected_image_count"] = expected_image_count
    if check_label_map:
        result.update(_inspect_label_map(manifest, expected_image_count))
    if not manifest.exists():
        result["failures"] = ["manifest does not exist"]
        if check_label_map and not result.get("label_map_ok"):
            result["failures"].extend(result.get("label_map_failures", []))
        return result
    try:
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
                if slot in (0, 1, 2):
                    components.append(
                        {
                            "semantic_slot_id": slot,
                            "width_in_blocks": int(component.get("width_in_blocks", 0)),
                            "height_in_blocks": int(component.get("height_in_blocks", 0)),
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
        failures = []
        if image_count <= 0:
            failures.append("image_count is zero")
        image_count_matches_expected = expected_image_count is None or image_count == expected_image_count
        if not image_count_matches_expected:
            failures.append(f"image_count {image_count} does not match expected {expected_image_count}")
        if not component_quant_tables_available:
            failures.append("sampled images are missing component quantization tables")
        if check_label_map and not result.get("label_map_ok"):
            failures.extend(result.get("label_map_failures", []))
        likely_metadata_profile = None
        regenerate_hint = None
        if not component_quant_tables_available:
            likely_metadata_profile = "dct" if saw_referenced_quant_table and not saw_persisted_quant_table else "unknown"
            regenerate_hint = (
                "Regenerate the GALP manifest with --metadata-profile reconstruct; "
                "RGB-no-more DCT input requires persisted JPEG quantization table values."
            )
        result.update(
            {
                "image_count": image_count,
                "expected_image_count": expected_image_count,
                "image_count_matches_expected": image_count_matches_expected,
                "inspected_images": inspected,
                "component_quant_tables_available": component_quant_tables_available,
                "likely_metadata_profile": likely_metadata_profile,
                "regenerate_hint": regenerate_hint,
                "images": images,
                "failures": failures,
                "ok": not failures,
            }
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics should preserve scanner progress.
        result["failures"] = [str(exc)]
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan GALP manifests for RGB-no-more Direct-DCT compatibility")
    parser.add_argument("paths", type=Path, nargs="+", help="Manifest files or directories containing manifest.bin files.")
    parser.add_argument("--torch-binding-dir", type=Path, default=DEFAULT_TORCH_BINDING_DIR)
    parser.add_argument("--sample-images", type=int, default=4)
    parser.add_argument("--expected-images", type=int, help="Expected image count for every scanned manifest; overrides split-name inference.")
    parser.add_argument("--expected-val-images", type=int, default=DEFAULT_VAL_IMAGES)
    parser.add_argument("--expected-train-images", type=int, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--check-label-map", action="store_true", help="Also require MANIFEST_DIR/labels.json to match the manifest split image count.")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_images <= 0:
        raise ValueError("--sample-images must be positive")
    if args.expected_images is not None and args.expected_images < 0:
        raise ValueError("--expected-images must be non-negative")
    if args.expected_val_images is not None and args.expected_val_images < 0:
        raise ValueError("--expected-val-images must be non-negative")
    if args.expected_train_images is not None and args.expected_train_images < 0:
        raise ValueError("--expected-train-images must be non-negative")
    galp_dct = _load_binding(args.torch_binding_dir)
    manifests = _candidate_manifests(args.paths)
    results = [
        _inspect_manifest(
            galp_dct,
            manifest,
            args.sample_images,
            args.expected_images,
            args.expected_val_images,
            args.expected_train_images,
            args.check_label_map,
        )
        for manifest in manifests
    ]
    summary = {
        "ok": all(result.get("ok") for result in results),
        "manifest_count": len(results),
        "compatible_count": sum(1 for result in results if result.get("ok")),
        "results": results,
    }
    for result in results:
        status = "OK" if result.get("ok") else "FAIL"
        image_count = result.get("image_count", "?")
        expected = result.get("expected_image_count", "?")
        quant = result.get("component_quant_tables_available", False)
        label_map = result.get("label_map_ok", None)
        shards = result.get("shard_count", 0)
        latest = result.get("latest_mtime") or "?"
        label_text = "" if label_map is None else f"\tlabels={label_map}"
        print(
            f"{status}\timages={image_count}\texpected={expected}\tquant_tables={quant}\t"
            f"shards={shards}\tlatest={latest}{label_text}\t{result['manifest']}"
        )
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
