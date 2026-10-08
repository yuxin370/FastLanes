"""GPU A/B of existing entry points and the model-facing Python facade.

Uses the same installed native binary for both arms. Correctness captures are
separate from alternating, warmed input-pipeline timings; no model speedup is
inferred. Run with python -m galp.torch.api_facade_ab --help.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import _galp_direct_dct as native

from galp.profiles import DctModelProfile
from galp.profiles.rgbnomore import TRAINING_PLS, VALIDATION
from galp.torch import DirectDctPipeline, DirectDctReader, TrainingPolicy
from galp.torch.direct_dct import _cnn_pipeline_options
from galp.torch.experimental import DirectDctPlsPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=(
        "cnn-inference", "transformer-inference", "cnn-training", "transformer-training",
    ))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--reference-json", type=Path, help="Existing CNN N_*.json with native_options")
    parser.add_argument("--baseline-python", type=Path, help="Pre-change direct_dct.py for Transformer inference")
    parser.add_argument("--images", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("a working CUDA device is required for this A/B")
    training = args.case.endswith("training")
    cnn = args.case.startswith("cnn")
    baseline_reader = DirectDctReader
    if args.baseline_python is not None:
        spec = importlib.util.spec_from_file_location("galp_api_baseline", args.baseline_python)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        baseline_reader = baseline.DirectDctReader
    if cnn:
        if args.reference_json is None:
            parser.error("CNN cases require --reference-json")
        reference = json.loads(args.reference_json.read_text())["native_options"]
        grid = reference["grid_transform"]
        channels = grid["output_channels"]
        profile = DctModelProfile(
            grid["y_output_width_blocks"],
            tuple((c, f) for c, f, _, _ in channels),
            tuple((subtract, divide) for _, _, subtract, divide in channels),
        )
    else:
        profile = TRAINING_PLS if training else VALIDATION
    if cnn and not training and _cnn_pipeline_options(profile) != reference:
        raise ValueError("reference options differ from the existing source512 CNN policy")
    if training:
        if args.mapping is None:
            parser.error("training requires --mapping")
        with args.mapping.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        policy = TrainingPolicy(args.mapping, seed=11997733, expected_mapping_sha256=digest, segments_per_pool=1)
    else:
        reader = native.DirectDctReader(str(args.manifest.resolve()))
        count = min(args.images, reader.image_count)
        preparation_size = 1024 if cnn else args.batch_size
        batches = [list(range(first, min(first + preparation_size, count)))
                   for first in range(0, count, preparation_size)]
        transforms = [[dict(crop=[32, 32, 448, 448], horizontal_flip=False) for _ in ids]
                      for ids in batches] if cnn else None

    def create(arm):
        if arm == "facade":
            return DirectDctPipeline(
                args.manifest, profile=profile, batch_size=args.batch_size,
                training=policy if training else None,
            )
        if training:
            return DirectDctPlsPipeline(
                args.manifest, args.mapping, training_seed=policy.seed,
                expected_mapping_sha256=digest, segments_per_pool=1,
                microbatch_images=args.batch_size,
                output_grid_size=profile.output_grid_size if cnn else 0,
                output_channels=channels if cnn else None,
            )
        if cnn:
            # Match the facade's reader lifetime. Reusing only the old reader
            # would give that arm warmed native metadata across fresh trials.
            existing_reader = native.DirectDctReader(str(args.manifest.resolve()))
            return existing_reader.pipeline_batch_options(**reference, output_batch_images=args.batch_size)
        return baseline_reader(args.manifest).pipeline(profile)

    def consume(pipeline, arm):
        for batch in pipeline:
            if cnn and not training and arm == "existing":
                ids = batch.global_image_ids
                for first in range(0, len(ids), args.batch_size):
                    count = min(args.batch_size, len(ids) - first)
                    yield ids[first:first + count], (batch.projected_range(first, count),), batch
            else:
                tensors = (batch.projected,) if cnn else (batch.y, batch.cbcr)
                if training:
                    tensors += (batch.targets,)
                yield batch.global_image_ids, tensors, batch._native

    def execution_counts(batch):
        stats = batch.execution_stats
        return {key: stats[key] for key in (
            "compressed_payload_bytes_read", "workset_upload_dma_bytes",
            "decode_kernel_launch_count", "planless_transform_kernel_launch_count",
        )}

    def run(arm, capture=False):
        digest = hashlib.sha256()
        ids_seen = []
        shapes = []
        execution = []
        previous_owner = None
        with closing(create(arm)) as pipeline:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            if training:
                pipeline.start_epoch(0)
            else:
                if arm == "existing" and not cnn:
                    pipeline.start(batches, transforms_by_batch=transforms)
                else:
                    pipeline.reset(batches, transforms_by_batch=transforms)
            iterator = consume(pipeline, arm)
            count = 0
            for ids, tensors, owner in iterator:
                count += len(ids)
                if capture:
                    if not training and owner is not previous_owner:
                        if previous_owner is not None:
                            execution.append(execution_counts(previous_owner))
                        previous_owner = owner
                    ids_seen.extend(ids)
                    for tensor in tensors:
                        shapes.append((list(tensor.shape), list(tensor.stride()), str(tensor.dtype)))
                        digest.update(tensor.detach().cpu().numpy().tobytes())
                else:
                    # Small GPU consumer in both arms; no per-batch host sync.
                    for tensor in tensors:
                        result = tensor.square().mean()
                if training and count >= args.images:
                    break
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            peak = torch.cuda.max_memory_allocated()
            if capture and not training:
                execution.append(execution_counts(previous_owner))
            if capture:
                retained = tensors[0]
                expected_retained = retained.cpu()
            iterator.close()
            del tensors, owner, previous_owner
            if not capture:
                del result
        native.manual_reclaim()
        if capture:
            # Exported tensors, not Python pipeline objects, own the backing.
            torch.testing.assert_close(retained.cpu(), expected_retained, rtol=0, atol=0)
        return dict(seconds=elapsed, images=count, torch_peak_bytes=peak,
                    ids=ids_seen, shapes=shapes, execution=execution,
                    tensor_sha256=digest.hexdigest() if capture else None)

    captures = [run(arm, capture=True) for arm in ("existing", "facade")]
    for key in ("images", "ids", "shapes", "tensor_sha256", "execution"):
        if captures[0][key] != captures[1][key]:
            raise AssertionError(f"new and existing entry points differ: {key}")
    print(json.dumps(dict(case=args.case, device=torch.cuda.get_device_name(),
                         correctness="exact inputs, targets (training), IDs, shapes and strides",
                         execution_counters_equal=None if training else True,
                         checked_images=captures[0]["images"])), flush=True)
    # Both paths have now executed once, outside performance measurement.
    measurements = {arm: [] for arm in ("existing", "facade")}
    for trial in range(args.trials):
        order = ("existing", "facade") if trial % 2 == 0 else ("facade", "existing")
        for arm in order:
            result = run(arm)
            measurements[arm].append({k: result[k] for k in ("seconds", "images", "torch_peak_bytes")})
            print(json.dumps(dict(trial=trial, arm=arm, **measurements[arm][-1])), flush=True)
    medians = {arm: statistics.median(r["seconds"] for r in rows) for arm, rows in measurements.items()}
    print(json.dumps(dict(case=args.case, median_seconds=medians,
                         facade_time_change_percent=100 * (medians["facade"] / medians["existing"] - 1))), flush=True)


if __name__ == "__main__":
    main()
