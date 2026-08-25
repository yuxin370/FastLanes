#!/usr/bin/env python3
"""Targeted Phase-5 proof for non-blocking native metrics completion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    from galp.profiles.rgbnomore import VALIDATION
    from galp.torch import DirectDctReader

    reader = DirectDctReader(args.manifest, module_path=args.module_path)
    pipeline = reader.pipeline(VALIDATION)
    pipeline.start([[0, 1, 2, 3]])
    batch = next(pipeline)

    before = dict(batch._native._metrics_completion)
    sentinel = batch.y.reshape(-1)[0].float() + batch.cbcr.reshape(-1)[0].float()
    natural_boundary = torch.cuda.Event()
    natural_boundary.record(torch.cuda.current_stream())
    natural_boundary.synchronize()
    after_batch = dict(batch._native._metrics_completion)
    after_pipeline = dict(pipeline._native._metrics_completion)
    metrics = pipeline.metrics
    pipeline.close()

    if not after_batch["host_snapshot_taken"]:
        raise AssertionError("host metrics snapshot was not observed")
    if not after_batch["gpu_timings_finalized"]:
        raise AssertionError("batch GPU timings did not finalize after the natural boundary")
    if not after_pipeline["gpu_timings_finalized"] or not metrics.complete:
        raise AssertionError("pipeline GPU timings did not finalize after the natural boundary")

    payload = {
        "schema": "galp-phase5-metrics-completion-v1",
        "before": before,
        "after_batch": after_batch,
        "after_pipeline": after_pipeline,
        "metrics_schema": "galp-direct-dct-metrics-v2",
        "sentinel": float(sentinel.item()),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "result": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE 5 METRICS COMPLETION SMOKE PASS")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
