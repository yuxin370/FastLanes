"""Compare the failing calibration batch under different decode partitions.

This is a diagnostic, not a training gate or a performance result. The larger
workset variant deliberately changes resource settings to isolate arithmetic
that depends on partitioning; it cannot qualify the production pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from fastlens.diagnostics.direct_dct import execution_stats
from fastlens.profiles import rgbnomore
from fastlens.torch.direct_dct import DirectDctPipeline, DirectDctReader
from fastlens.benchmarks.baselines.system_rgbnomore.diagnostics.rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM
from .contracts import artifact, load_artifact, load_trace, validate_batches, write_json
from .gates import input_failure_details, numerical_runtime
from .runner import inputs
from .training import finish_batch, upstream_ops
from .unified import UnifiedReplay, cpu_raw, microbatches


def partition_options(*, larger_workset=False):
    # Match the registered training profile's semantics and execution settings.
    # The second configuration changes only the workset partition controls.
    return dict(layout="transformed_dct_grid", grid_transform=dict(RGBNOMORE_VAL_DCT_GRID_TRANSFORM),
                dct_coeffs="all", cache_capacity_mib=0, plan_cache_capacity=0,
                decode_batch_rowgroups=1024 if larger_workset else 64,
                decode_workset_capacity_mib=2048 if larger_workset else 512,
                block_major_double_buffer="disabled" if larger_workset else "auto",
                enable_planless_execution=True, enable_rowgroup_prefetch=True,
                rowgroup_prefetch_depth=16, rowgroup_prefetch_workers=8, rowgroup_prefetch_min_decode_batches=1,
                scheduling_policy="limited-overlap", transform_blocks_per_launch=512,
                transform_ctas_per_launch=512, use_low_priority_streams=True, async_planless_completion=True,
                crop_execution_mode="bounded-io-uring-range-read-selected-decode",
                bounded_read_amplification_cap=1.1, bounded_read_local_amplification_cap=0., bounded_read_max_run_bytes=0)


def capture(pipeline, ids, rows, device):
    transforms = [dict(crop=dict(zip(("x", "y", "width", "height"), row["crop_box"])),
                       horizontal_flip=row["flip"], logical_sample_id=row["image_id"]) for row in rows]
    values, actual_ids, stats = [], [], []
    pipeline.start([ids], transforms_by_batch=[transforms])
    for batch in pipeline:
        batch.record_stream(torch.cuda.current_stream(device))
        actual_ids.extend(batch.global_image_ids)
        y, cbcr = batch.y.float(), batch.cbcr.float()
        values.extend((y[i, 0].clone(), cbcr[i, 0].clone(), cbcr[i, 1].clone()) for i in range(len(y)))
        raw = execution_stats(batch)
        keys = ("uses_planless_fixed_transform", "decode_batch_count", "decode_workset_capacity_bytes",
                "max_estimated_decode_workset_bytes", "planless_transform_active_output_workset_count",
                "planless_transform_output_workset_ownership_count", "fixed_grid_output_dtype")
        stats.append({key: raw[key] for key in keys if key in raw})
    if actual_ids != ids:
        raise AssertionError("diagnostic output occurrence order differs from the frozen batch")
    return values, stats


def run(config_path, root, module_path, output):
    config, source, profile, records, _ = inputs(config_path)
    if profile["model"] != "vitti" or not profile.get("training"):
        raise ValueError("this diagnostic targets the production ViT-Ti training fixture")
    labels = load_artifact(config["label_mapping"])["labels"]
    trace = load_trace(config["calibration_trace"]["path"], profile, records, labels)
    if artifact(config["calibration_trace"]["path"]) != config["calibration_trace"]:
        raise ValueError("calibration trace changed")
    validate_batches(trace, config, training=True)
    rows = microbatches(trace)[0]
    ids = [records[row["image_id"]]["galp_image_id"] for row in rows]
    selected = dict(source, images=[records[image_id] for image_id in dict.fromkeys(row["image_id"] for row in rows)])
    device = torch.device("cuda:0")
    runtime = numerical_runtime(device)
    torch.set_num_threads(config["torch_threads"])
    os.environ["FASTLENS_BLOCK_MAJOR_ACCESS_DIR"] = str(Path(root).resolve().parent/"access")
    print(f"Checking source q/Q for the {len(selected['images'])} images in microbatch {rows[0]['microbatch_id']}", flush=True)
    report = dict(runtime=runtime, microbatch_id=rows[0]["microbatch_id"], samples=len(rows),
                  diagnostic_only=True, headline_eligible=False, variants={})
    dops = upstream_ops(profile["upstream"])
    expected = [tuple(value.to(device) for value in cpu_raw(records[row["image_id"]], row, profile, dops)) for row in rows]
    oracle_inputs, oracle_targets = finish_batch(expected, rows, profile, dops, oracle=True)
    with UnifiedReplay("fastlens", root, config, selected, profile, device=device, module_path=module_path) as replay:
        replay.verify_source()
        report["source_q_and_tables_exact"] = True
        reader = DirectDctReader(root, module_path=module_path)
        for name in ("production", "configured_same_partition", "larger_workset"):
            print(f"Running {name}", flush=True)
            if name == "production":
                pipeline = replay.pipeline
            else:
                native = reader._native.pipeline_batch_options(**partition_options(larger_workset=name == "larger_workset"))
                pipeline = DirectDctPipeline._from_native(native, rgbnomore.TRAINING_PLS.id, reader.image_count)
            with pipeline:
                actual, stats = capture(pipeline, ids, rows, device)
                actual_inputs, targets = finish_batch(actual, rows, profile, dops)
                torch.testing.assert_close(targets, oracle_targets, atol=0, rtol=0)
                details = input_failure_details(replay, rows, actual, expected)
                details["execution_stats"] = stats
                details["inputs_mismatches"] = [int((a != b).sum()) for a, b in zip(actual_inputs, oracle_inputs)]
                details["inputs_max_abs"] = [float((a.double()-b.double()).abs().max()) for a, b in zip(actual_inputs, oracle_inputs)]
                report["variants"][name] = details
                print(json.dumps(dict(variant=name, raw_mismatches=details["raw_mismatches_by_component"],
                                      inputs_mismatches=details["inputs_mismatches"], execution_stats=stats)), flush=True)
    write_json(output, report)
    print(f"Diagnosis saved: {output}; production gate remains unqualified", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, default=Path(__file__).resolve().parents[4]/"build/fastlens/torch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.root, args.module_path, args.output)


if __name__ == "__main__":
    main()
