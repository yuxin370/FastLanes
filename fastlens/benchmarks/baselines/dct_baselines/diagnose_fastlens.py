"""Check one production microbatch before rerunning the complete numerical gate."""
import argparse
import json
import os
from pathlib import Path

import torch

from .contracts import artifact, load_artifact, load_trace
from .gates import input_failure_details, numerical_runtime
from .production_gate import MANIFEST, REPO
from .runner import inputs
from .training import finish_batch
from .unified import UnifiedReplay, cpu_raw, microbatches


def check_inference(replay, rows):
    from .materialize import build_model, source_reference
    from .source import source_images
    values = [source_reference(replay.profile, image, row) for image, row in
              zip(source_images(dict(images=[replay.records[r['image_id']] for r in rows])), rows)]
    expected = tuple(torch.stack([v[i] for v in values]).to(replay.device) for i in range(len(values[0])))
    model = build_model(replay.profile, replay.device).eval().requires_grad_(False)
    with torch.inference_mode():
        batches = replay.inference_batches(rows)
        try:
            _, actual = next(batches)
            errors = [float((a-b).abs().max()) for a, b in zip(actual, expected)]
            print('Inference input max absolute errors: ' + json.dumps(errors), flush=True)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            logits, reference = model(*actual), model(*expected)
            print(f'Inference logits max absolute error: {float((logits-reference).abs().max())}', flush=True)
            torch.testing.assert_close(logits, reference, atol=0, rtol=0)
        finally:
            batches.close()
    print(f'Passed: {len(rows)} images, checkpoint inference inputs and logits exact. '
          'Diagnostic only; full inference verification still required.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=0)
    args = parser.parse_args()
    config, source, profile, records, trace = inputs(args.prepared/"contract/matched.json")
    if profile.get('training'):
        if artifact(config["calibration_trace"]["path"]) != config["calibration_trace"]:
            raise ValueError("calibration trace changed")
        labels = load_artifact(config["label_mapping"])["labels"]
        trace = load_trace(config["calibration_trace"]["path"], profile, records, labels)
    batches = microbatches(trace)
    if not 0 <= args.microbatch < len(batches):
        parser.error("microbatch index is outside the trace")
    rows = batches[args.microbatch]
    selected = {r["image_id"] for r in rows}
    source = dict(source, images=[r for r in source["images"] if r["image_id"] in selected])
    os.environ["FASTLENS_BLOCK_MAJOR_ACCESS_DIR"] = str(MANIFEST.parent/"access")
    numerical_runtime("cuda:0")
    torch.set_num_threads(config["torch_threads"])
    with UnifiedReplay("fastlens", MANIFEST, config, source, profile, device="cuda:0",
                       module_path=REPO/"build/fastlens/torch") as replay:
        replay.verify_source(workers=min(config["workers"], len(selected)))
        if not profile.get('training'):
            check_inference(replay, rows)
            return
        actual = replay.raw(rows)
        expected = [tuple(v.to(replay.device) for v in cpu_raw(records[r["image_id"]], r, profile, replay.dops))
                    for r in rows]
        details = input_failure_details(replay, rows, actual, expected)
        print("Input diagnosis: " + json.dumps(details, sort_keys=True), flush=True)
        for a, b in zip(actual, expected):
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        a = finish_batch(actual, rows, profile, replay.dops)
        b = finish_batch(expected, rows, profile, replay.dops, oracle=True)
        for left, right in zip((*a[0], a[1]), (*b[0], b[1])):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    print(f"Passed: {len(rows)} images, raw coefficients, augmented/Mixup inputs and targets exact. "
          "Input-only diagnostic; full calibration/verification still required.", flush=True)


if __name__ == "__main__":
    main()
