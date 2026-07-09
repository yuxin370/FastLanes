#!/usr/bin/env python3
"""Validate comparable GALP/RGB-no-more benchmark outputs.

This is a correctness gate for the JSON files produced by
run_rgbnomore_comparison.py. It checks that required backends/phases are present
and that DCT/RGB input shapes plus 1000-class logits match the comparison
contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_RGB_SHAPE_TAIL = [3, 224, 224]
EXPECTED_DCT_Y_SHAPE_TAIL = [1, 28, 28, 8, 8]
EXPECTED_DCT_CBCR_SHAPE_TAIL = [2, 14, 14, 8, 8]
EXPECTED_RGB_PREPROCESS = "resize256_centercrop224_to_minus_one_one"
EXPECTED_DCT_EVAL_PREPROCESS = "resized_center_crop_dct_32_to_28_to_minus_one_one"
EXPECTED_DCT_TRAIN_PREPROCESS = "random_resized_crop_dct_28_flip_randaugment_to_minus_one_one"
GALP_NONNEGATIVE_COUNTERS = [
    "selected_vectors",
    "internal_syncs",
]
GALP_POSITIVE_COUNTERS = [
    "full_vectors",
    "decode_kernels",
    "rowgroups",
    "worksets",
    "projection_items",
]


def _expected_phases(args: argparse.Namespace) -> tuple[str, ...]:
    if args.expected_phase == "loader":
        return ("loader_to_device",)
    if args.expected_phase == "forward":
        return ("forward_step",)
    if args.expected_phase == "end-to-end":
        return ("end_to_end",)
    if args.expected_phase == "train":
        return ("train_step",)
    return ("loader_to_device", "forward_step", "end_to_end")


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise RuntimeError(f"{path} does not contain a JSON object or list")


def _record_by_phase(records: list[dict[str, Any]], backend: str, phase: str) -> dict[str, Any] | None:
    for record in records:
        if record.get("backend") == backend and record.get("phase") == phase:
            return record
    return None


def _require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def _tail(shape: Any) -> list[int] | None:
    if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
        return None
    return shape[1:]


def _batch(shape: Any) -> int | None:
    if not isinstance(shape, list) or not shape or not isinstance(shape[0], int):
        return None
    return shape[0]


def _validate_common_perf(record: dict[str, Any], failures: list[str], label: str, args: argparse.Namespace) -> None:
    _require(int(record.get("images", 0)) > 0, failures, f"{label}: images must be positive")
    _require(float(record.get("seconds", 0.0)) > 0.0, failures, f"{label}: seconds must be positive")
    _require(float(record.get("images_per_s", 0.0)) > 0.0, failures, f"{label}: images_per_s must be positive")
    if args.expected_batch_size is not None:
        _require(
            int(record.get("batch_size", -1)) == args.expected_batch_size,
            failures,
            f"{label}: expected batch_size {args.expected_batch_size}, got {record.get('batch_size')}",
        )
    if args.expected_steps is not None:
        _require(
            int(record.get("steps", -1)) == args.expected_steps,
            failures,
            f"{label}: expected steps {args.expected_steps}, got {record.get('steps')}",
        )
    if args.expected_device is not None:
        _require(
            _same_device(record.get("device"), args.expected_device),
            failures,
            f"{label}: expected device {args.expected_device}, got {record.get('device')}",
        )


def _validate_logits(record: dict[str, Any], failures: list[str], label: str) -> None:
    logits_shape = record.get("logits_shape")
    _require(
        isinstance(logits_shape, list) and len(logits_shape) == 2 and logits_shape[1] == 1000,
        failures,
        f"{label}: logits_shape must be [B,1000], got {logits_shape}",
    )


def _validate_train_step(record: dict[str, Any], failures: list[str], label: str) -> None:
    loss = record.get("loss")
    _require(isinstance(loss, (int, float)) and math.isfinite(float(loss)), failures, f"{label}: loss must be finite, got {loss}")
    _require(record.get("optimizer") == "sgd", failures, f"{label}: expected optimizer sgd, got {record.get('optimizer')}")
    train_lr = record.get("train_lr")
    _require(
        isinstance(train_lr, (int, float)) and float(train_lr) >= 0.0,
        failures,
        f"{label}: train_lr must be a non-negative number, got {train_lr}",
    )


def _validate_dataset_size(record: dict[str, Any], failures: list[str], label: str, args: argparse.Namespace) -> None:
    if args.expected_split_images is None:
        return
    _require(
        int(record.get("dataset_size", -1)) == args.expected_split_images,
        failures,
        f"{label}: expected dataset_size {args.expected_split_images}, got {record.get('dataset_size')}",
    )


def _expected_native_dct_preprocess(args: argparse.Namespace) -> str | None:
    if args.expected_split != "train":
        return EXPECTED_DCT_EVAL_PREPROCESS
    if args.expected_dct_eval_transform == "false":
        return EXPECTED_DCT_TRAIN_PREPROCESS
    return EXPECTED_DCT_EVAL_PREPROCESS


def _validate_galp_dataset_size(record: dict[str, Any], failures: list[str], label: str, args: argparse.Namespace) -> None:
    expected_images = args.expected_galp_images if args.expected_galp_images is not None else args.expected_split_images
    if expected_images is None:
        return
    _require(
        int(record.get("dataset_size", -1)) == expected_images,
        failures,
        f"{label}: expected dataset_size {expected_images}, got {record.get('dataset_size')}",
    )


def _validate_galp_label_map(path: Path | None, expected_images: int | None, failures: list[str], label: str) -> None:
    if path is None:
        failures.append(f"{label}: expected GALP label map path is not set")
        return
    if not path.exists():
        failures.append(f"{label}: GALP label map does not exist: {path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict) and payload.get("format") == "galp_rgbnomore_label_map_v1",
        failures,
        f"{label}: GALP label map has unexpected format {payload.get('format') if isinstance(payload, dict) else type(payload)}",
    )
    if not isinstance(payload, dict):
        return
    labels = payload.get("labels")
    _require(
        isinstance(labels, list) and all(isinstance(item, int) and 0 <= item < 1000 for item in labels),
        failures,
        f"{label}: GALP label map labels must be integer ImageNet-1K labels",
    )
    if expected_images is not None:
        _require(
            int(payload.get("image_count", -1)) == expected_images,
            failures,
            f"{label}: expected label map image_count {expected_images}, got {payload.get('image_count')}",
        )
        _require(
            isinstance(labels, list) and len(labels) == expected_images,
            failures,
            f"{label}: expected label map labels length {expected_images}, got {len(labels) if isinstance(labels, list) else None}",
        )


def _validate_forward_model(
    record: dict[str, Any],
    failures: list[str],
    label: str,
    *,
    expected_model: str,
    expected_checkpoint: Path | None,
) -> None:
    _require(
        record.get("model") == expected_model,
        failures,
        f"{label}: expected model {expected_model}, got {record.get('model')}",
    )
    _require(
        _same_path(record.get("checkpoint"), expected_checkpoint),
        failures,
        f"{label}: expected checkpoint {expected_checkpoint}, got {record.get('checkpoint')}",
    )


def _validate_galp_adapter_contract(
    record: dict[str, Any],
    failures: list[str],
    label: str,
    args: argparse.Namespace,
    *,
    require_runtime_counters: bool,
) -> None:
    if args.expected_galp_preprocess is not None:
        _require(
            record.get("preprocess") == args.expected_galp_preprocess,
            failures,
            f"{label}: expected preprocess {args.expected_galp_preprocess}, got {record.get('preprocess')}",
        )
    _require(record.get("dct_coeffs") == "all", failures, f"{label}: expected dct_coeffs all, got {record.get('dct_coeffs')}")
    expected_layouts = {"ycbcr_dct_grid", "ycbcr_dct_grid_fixed"}
    _require(
        record.get("output_layout") in expected_layouts,
        failures,
        f"{label}: expected output_layout in {sorted(expected_layouts)}, got {record.get('output_layout')}",
    )
    _require(bool(record.get("dequantize")), failures, f"{label}: dequantize must be true")
    _require(
        bool(record.get("scale_to_rgbnomore_range")),
        failures,
        f"{label}: scale_to_rgbnomore_range must be true",
    )
    if require_runtime_counters:
        for key in GALP_NONNEGATIVE_COUNTERS:
            value = record.get(key)
            _require(isinstance(value, int) and value >= 0, failures, f"{label}: {key} must be a non-negative integer, got {value}")
        for key in GALP_POSITIVE_COUNTERS:
            value = record.get(key)
            _require(isinstance(value, int) and value > 0, failures, f"{label}: {key} must be a positive integer, got {value}")


def _same_path(actual: Any, expected: Path | None) -> bool:
    if expected is None:
        return True
    if not isinstance(actual, str) or not actual:
        return False
    return Path(actual).expanduser().resolve() == expected.expanduser().resolve()


def _same_device(actual: Any, expected: str) -> bool:
    actual_text = str(actual)
    if actual_text == expected:
        return True
    if expected == "cuda" and actual_text.startswith("cuda:"):
        return actual_text.split(":", 1)[1].isdigit()
    return False


def _validate_rgb_backend(
    records: list[dict[str, Any]],
    backend: str,
    label: str,
    failures: list[str],
    args: argparse.Namespace,
    *,
    required: bool,
) -> None:
    for phase in _expected_phases(args):
        record = _record_by_phase(records, backend, phase)
        if record is None and not required:
            continue
        _require(record is not None, failures, f"{label} baseline missing phase {phase}")
        if record is None:
            continue
        _validate_common_perf(record, failures, f"{label} {phase}", args)
        _validate_dataset_size(record, failures, f"{label} {phase}", args)
        input_shape = record.get("input_shape")
        _require(_tail(input_shape) == EXPECTED_RGB_SHAPE_TAIL, failures, f"{label} {phase}: bad input_shape {input_shape}")
        _require(
            record.get("rgb_preprocess") == EXPECTED_RGB_PREPROCESS,
            failures,
            f"{label} {phase}: expected rgb_preprocess {EXPECTED_RGB_PREPROCESS}, got {record.get('rgb_preprocess')}",
        )
        _require(
            _same_path(record.get("data_dir"), args.expected_rgb_data_dir),
            failures,
            f"{label} {phase}: expected data_dir {args.expected_rgb_data_dir}, got {record.get('data_dir')}",
        )
        if args.expected_split is not None:
            _require(
                str(record.get("split", "")) == args.expected_split,
                failures,
                f"{label} {phase}: expected split {args.expected_split}, got {record.get('split')}",
            )
        if backend == "dali_rgb_rgbnomore" and args.expected_dali_prefetch_queue_depth is not None:
            _require(
                int(record.get("prefetch_queue_depth", -1)) == args.expected_dali_prefetch_queue_depth,
                failures,
                f"{label} {phase}: expected prefetch_queue_depth {args.expected_dali_prefetch_queue_depth}, "
                f"got {record.get('prefetch_queue_depth')}",
            )
        if phase in ("forward_step", "end_to_end", "train_step"):
            _validate_logits(record, failures, f"{label} {phase}")
            _validate_forward_model(
                record,
                failures,
                f"{label} {phase}",
                expected_model="rgbnomore_rgb_vitti",
                expected_checkpoint=args.expected_rgb_checkpoint,
            )
            if phase == "train_step":
                _validate_train_step(record, failures, f"{label} {phase}")


def _validate_dct(records: list[dict[str, Any]], backend: str, label: str, failures: list[str], args: argparse.Namespace) -> None:
    for phase in _expected_phases(args):
        record = _record_by_phase(records, backend, phase)
        _require(record is not None, failures, f"{label} missing phase {phase}")
        if record is None:
            continue
        _validate_common_perf(record, failures, f"{label} {phase}", args)
        if backend == "rgbnomore_native_dct":
            _validate_dataset_size(record, failures, f"{label} {phase}", args)
            if args.expected_split is not None:
                _require(
                    str(record.get("split", "")) == args.expected_split,
                    failures,
                    f"{label} {phase}: expected split {args.expected_split}, got {record.get('split')}",
                )
            _require(
                _same_path(record.get("data_root"), args.expected_data_root),
                failures,
                f"{label} {phase}: expected data_root {args.expected_data_root}, got {record.get('data_root')}",
            )
            _require(
                _same_path(record.get("index_file"), args.expected_dct_index_file),
                failures,
                f"{label} {phase}: expected index_file {args.expected_dct_index_file}, got {record.get('index_file')}",
            )
            if args.expected_dct_eval_transform is not None:
                expected_eval = args.expected_dct_eval_transform == "true"
                _require(
                    bool(record.get("eval_transform")) == expected_eval,
                    failures,
                    f"{label} {phase}: expected eval_transform {expected_eval}, got {record.get('eval_transform')}",
                )
            expected_preprocess = _expected_native_dct_preprocess(args)
            if expected_preprocess is not None:
                _require(
                    record.get("dct_preprocess") == expected_preprocess,
                    failures,
                    f"{label} {phase}: expected dct_preprocess {expected_preprocess}, got {record.get('dct_preprocess')}",
                )
        if backend == "galp_direct_dct_rgbnomore":
            _require(
                _same_path(record.get("manifest"), args.expected_galp_manifest),
                failures,
                f"{label} {phase}: expected manifest {args.expected_galp_manifest}, got {record.get('manifest')}",
            )
            _validate_galp_dataset_size(record, failures, f"{label} {phase}", args)
            _validate_galp_adapter_contract(
                record,
                failures,
                f"{label} {phase}",
                args,
                require_runtime_counters=phase in ("loader_to_device", "end_to_end", "train_step"),
            )
        input_y_shape = record.get("input_y_shape")
        input_cbcr_shape = record.get("input_cbcr_shape")
        _require(_tail(input_y_shape) == EXPECTED_DCT_Y_SHAPE_TAIL, failures, f"{label} {phase}: bad input_y_shape {input_y_shape}")
        _require(
            _tail(input_cbcr_shape) == EXPECTED_DCT_CBCR_SHAPE_TAIL,
            failures,
            f"{label} {phase}: bad input_cbcr_shape {input_cbcr_shape}",
        )
        y_batch = _batch(input_y_shape)
        cbcr_batch = _batch(input_cbcr_shape)
        if y_batch is not None and cbcr_batch is not None:
            _require(y_batch == cbcr_batch, failures, f"{label} {phase}: Y/CbCr batch mismatch")
        if phase in ("forward_step", "end_to_end", "train_step"):
            _validate_logits(record, failures, f"{label} {phase}")
            _validate_forward_model(
                record,
                failures,
                f"{label} {phase}",
                expected_model="rgbnomore_jpeg_ti_vitti",
                expected_checkpoint=args.expected_dct_checkpoint,
            )
            logits_batch = _batch(record.get("logits_shape"))
            if y_batch is not None and logits_batch is not None:
                _require(y_batch == logits_batch, failures, f"{label} {phase}: logits batch mismatch")
            if phase == "train_step":
                _validate_train_step(record, failures, f"{label} {phase}")
                if backend == "galp_direct_dct_rgbnomore":
                    expected_galp_images = (
                        args.expected_galp_images if args.expected_galp_images is not None else args.expected_split_images
                    )
                    _require(
                        record.get("label_source") == "rgbnomore_label_map_json",
                        failures,
                        f"{label} {phase}: expected label_source rgbnomore_label_map_json, got {record.get('label_source')}",
                    )
                    _require(
                        _same_path(record.get("label_map_json"), args.expected_galp_label_map_json),
                        failures,
                        f"{label} {phase}: expected label_map_json {args.expected_galp_label_map_json}, got {record.get('label_map_json')}",
                    )
                    _validate_galp_label_map(
                        args.expected_galp_label_map_json,
                        expected_galp_images,
                        failures,
                        f"{label} {phase}",
                    )


def _validate_manifest_verify(
    path: Path | None,
    expected_images: int | None,
    expected_manifest: Path | None,
    failures: list[str],
) -> None:
    if path is None:
        return
    records = _load_records(path)
    record = records[0]
    validation = record.get("metadata_validation", {})
    _require(
        _same_path(validation.get("manifest"), expected_manifest),
        failures,
        f"{path}: expected verified manifest {expected_manifest}, got {validation.get('manifest')}",
    )
    _require(
        bool(validation.get("component_quant_tables_available")),
        failures,
        f"{path}: component quant tables are not available",
    )
    _require(bool(validation.get("image_count_matches_expected")), failures, f"{path}: image_count did not match expected")
    if expected_images is not None:
        _require(
            int(validation.get("image_count", -1)) == expected_images,
            failures,
            f"{path}: expected image_count {expected_images}, got {validation.get('image_count')}",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GALP/RGB-no-more comparison JSON outputs")
    parser.add_argument("--output-dir", type=Path, help="Directory produced by run_rgbnomore_comparison.py")
    parser.add_argument("--galp-json", type=Path)
    parser.add_argument("--dct-json", type=Path)
    parser.add_argument("--rgb-json", type=Path)
    parser.add_argument("--dali-json", type=Path)
    parser.add_argument("--manifest-verify-json", type=Path)
    parser.add_argument("--expected-galp-images", type=int)
    parser.add_argument("--expected-split-images", type=int)
    parser.add_argument("--expected-batch-size", type=int)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-phase", choices=("loader", "forward", "end-to-end", "train", "both"), default="both")
    parser.add_argument("--expected-device")
    parser.add_argument("--expected-split")
    parser.add_argument("--expected-data-root", type=Path)
    parser.add_argument("--expected-rgb-data-dir", type=Path)
    parser.add_argument("--expected-dct-index-file", type=Path)
    parser.add_argument("--expected-dct-eval-transform", choices=("true", "false"))
    parser.add_argument("--expected-galp-manifest", type=Path)
    parser.add_argument("--expected-galp-label-map-json", type=Path)
    parser.add_argument("--expected-galp-preprocess")
    parser.add_argument("--expected-rgb-checkpoint", type=Path)
    parser.add_argument("--expected-dct-checkpoint", type=Path)
    parser.add_argument("--expected-dali-prefetch-queue-depth", type=int)
    parser.add_argument("--allow-missing-galp", action="store_true")
    parser.add_argument("--require-dali-rgb", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is not None:
        args.galp_json = args.galp_json or (args.output_dir / "galp_direct_dct_rgbnomore.json")
        args.dct_json = args.dct_json or (args.output_dir / "rgbnomore_native_dct.json")
        args.rgb_json = args.rgb_json or (args.output_dir / "rgbnomore_rgb.json")
        args.dali_json = args.dali_json or (args.output_dir / "dali_rgb.json")
        args.manifest_verify_json = args.manifest_verify_json or (args.output_dir / "galp_manifest_verify.json")

    failures: list[str] = []
    records: list[dict[str, Any]] = []
    for label, path, required in (
        ("GALP", args.galp_json, not args.allow_missing_galp),
        ("RGB-no-more DCT", args.dct_json, True),
        ("RGB", args.rgb_json, True),
        ("DALI RGB", args.dali_json, args.require_dali_rgb),
    ):
        if path is None or not path.exists():
            _require(not required, failures, f"{label} JSON is missing: {path}")
            continue
        records.extend(_load_records(path))

    if args.manifest_verify_json is not None and args.manifest_verify_json.exists():
        _validate_manifest_verify(args.manifest_verify_json, args.expected_galp_images, args.expected_galp_manifest, failures)
    elif not args.allow_missing_galp:
        failures.append(f"manifest verify JSON is missing: {args.manifest_verify_json}")

    _validate_rgb_backend(records, "pytorch_rgb_rgbnomore", "RGB", failures, args, required=True)
    dali_json_available = args.dali_json is not None and args.dali_json.exists()
    if dali_json_available:
        _validate_rgb_backend(records, "dali_rgb_rgbnomore", "DALI RGB", failures, args, required=args.require_dali_rgb)
    _validate_dct(records, "rgbnomore_native_dct", "RGB-no-more DCT", failures, args)
    if not args.allow_missing_galp:
        _validate_dct(records, "galp_direct_dct_rgbnomore", "GALP Direct-DCT", failures, args)

    summary = {
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "record_count": len(records),
        "allow_missing_galp": args.allow_missing_galp,
        "require_dali_rgb": args.require_dali_rgb,
        "expected_batch_size": args.expected_batch_size,
        "expected_split_images": args.expected_split_images,
        "expected_steps": args.expected_steps,
        "expected_phase": args.expected_phase,
        "expected_device": args.expected_device,
        "expected_split": args.expected_split,
        "expected_data_root": str(args.expected_data_root) if args.expected_data_root is not None else None,
        "expected_rgb_data_dir": str(args.expected_rgb_data_dir) if args.expected_rgb_data_dir is not None else None,
        "expected_dct_index_file": str(args.expected_dct_index_file) if args.expected_dct_index_file is not None else None,
        "expected_dct_eval_transform": args.expected_dct_eval_transform,
        "expected_galp_manifest": str(args.expected_galp_manifest) if args.expected_galp_manifest is not None else None,
        "expected_galp_label_map_json": str(args.expected_galp_label_map_json) if args.expected_galp_label_map_json is not None else None,
        "expected_galp_preprocess": args.expected_galp_preprocess,
        "expected_rgb_checkpoint": str(args.expected_rgb_checkpoint) if args.expected_rgb_checkpoint is not None else None,
        "expected_dct_checkpoint": str(args.expected_dct_checkpoint) if args.expected_dct_checkpoint is not None else None,
        "expected_dali_prefetch_queue_depth": args.expected_dali_prefetch_queue_depth,
    }
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
