"""Inspect IC-batched's first inference pool; never emit a correctness certificate."""
import argparse
import json
from pathlib import Path
import threading

import numpy as np
import torch

from fastlens.benchmarks.baselines.dct_baselines.gates import numerical_runtime
from fastlens.benchmarks.baselines.dct_baselines.materialize import build_model, cnn_backend
from fastlens.benchmarks.baselines.dct_baselines.runner import Replay, inputs
from fastlens.benchmarks.baselines.dct_baselines.source import source_images
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader
from fastlens.benchmarks.dct_models.online_crop import SourceReference


def run(config_path, root, *, device="cuda:0", repeats=3):
    config, _, profile, records, trace = inputs(config_path)
    if profile["transform"] != "cnn_source512" or profile.get("training"):
        raise ValueError("this diagnostic targets CNN checkpoint inference")
    numerical_runtime(device)
    torch.set_num_threads(config["torch_threads"])
    # Match the failing gate's initialization, including its CUDA context.
    model = build_model(profile, torch.device(device)).eval().requires_grad_(False)
    rows = trace[:config["preparation_pool_size"]]
    watched = rows[:config["model_microbatch_size"]]
    watched_ids = {r["image_id"] for r in watched}
    backend = cnn_backend(profile["model"])
    print(f"Diagnostic: pool={len(rows)}, inspected={len(watched)}, workers={config['workers']}, "
          f"torch_threads={torch.get_num_threads()}, device={device}", flush=True)
    failures = 0
    with StoreReader(root, workers=config["workers"], decode_budget_bytes=config["decode_budget_bytes"],
                     memory_budget_bytes=config["reader_memory_budget_bytes"]) as reader:
        for repeat in range(repeats):
            replay = Replay(reader, profile, config)
            materialize, observed, local = replay.materialize, {}, threading.local()
            resize = materialize.resize

            def inspect_resize(*args, **kwargs):
                result = resize(*args, **kwargs)
                if local.image_id in watched_ids:
                    c = local.component
                    frequencies = [f for component, f in profile["output_channels"] if component == c]
                    grid = profile["output_grids"][c]
                    observed[local.image_id]["unrounded"].append(
                        result.detach().reshape(grid, grid, 64)[..., frequencies].clone())
                local.component += 1
                return result

            def inspect_materialize(q, tables, row, *, cropped):
                local.image_id, local.component = row["image_id"], 0
                if local.image_id in watched_ids:
                    observed[local.image_id] = dict(q=q, tables=tables, unrounded=[])
                return materialize(q, tables, row, cropped=cropped)

            materialize.resize = inspect_resize
            replay.materialize = inspect_materialize
            with torch.inference_mode():
                batches, _ = replay.pool(rows)
                cpu_inputs = batches[0][0][0]
                actual = cpu_inputs.to(device)
                expected, reference_raw = [], []
                for image, row in zip(source_images(dict(images=[records[r["image_id"]] for r in watched])), watched):
                    capture = observed[row["image_id"]]
                    x, y, width, height = row["crop_box"]
                    for c, (q, table) in enumerate(zip(image.coefficients, image.tables)):
                        scale = 8 if c == 0 else 16
                        np.testing.assert_array_equal(capture["q"][c], q[y//scale:(y+height)//scale, x//scale:(x+width)//scale])
                        np.testing.assert_array_equal(capture["tables"][c], table)
                    raw = SourceReference(Path(profile["upstream"]), backend.GRID)(
                        tuple(q.reshape(*q.shape[:2], 64) for q in image.coefficients), image.tables)
                    reference_raw.append(raw)
                    expected.append(backend.organize([v.round().clamp(-32768, 32767) for v in raw]))
                expected = torch.stack(expected).to(device)
                tolerance = config["tolerance"]
                bad = (~torch.isclose(actual, expected, atol=tolerance["atol"], rtol=tolerance["rtol"])).nonzero()
                examples = []
                for sample, channel, y, x in bad[:8].tolist():
                    row = watched[sample]
                    component, frequency = profile["output_channels"][channel]
                    frequencies = [f for c, f in profile["output_channels"] if c == component]
                    worker_raw = observed[row["image_id"]]["unrounded"][component][y, x, frequencies.index(frequency)]
                    oracle_raw = reference_raw[sample][component][y, x, frequency]
                    examples.append(dict(image_id=row["image_id"], index=[sample, channel, y, x],
                        component=component, frequency=frequency, actual=float(actual[sample, channel, y, x]),
                        expected=float(expected[sample, channel, y, x]), worker_unrounded=float(worker_raw),
                        oracle_unrounded=float(oracle_raw), worker_rounded=float(worker_raw.round()),
                        oracle_rounded=float(oracle_raw.round())))
                result = dict(repeat=repeat, source_crop_q_and_tables_exact=True,
                    host_to_device_exact=bool(torch.equal(actual.cpu(), cpu_inputs)),
                    mismatches=int(bad.shape[0]), max_abs_error=float((actual-expected).abs().max()), examples=examples)
                failures += int(bad.shape[0])
                if not bad.numel():
                    result["max_abs_logits_error"] = float((model(actual)-model(expected)).abs().max())
                print(json.dumps(result, sort_keys=True), flush=True)
    print(f"Diagnostic finished: {failures} out-of-tolerance values across {repeats} repeats. "
          "Full inference verification is still required.", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    run(args.config, args.store, device=args.device, repeats=args.repeats)


if __name__ == "__main__":
    main()
