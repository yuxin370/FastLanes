#!/usr/bin/env python3
"""Small deterministic Phase-4B lifetime-shadow wiring smoke.

This test deliberately completes CUDA consumer work before dropping the
native-backed tensors. It validates shadow wiring and multi-Storage accounting
without relying on a timing-sensitive outstanding-consumer window.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path


PROFILE = "rgbnomore-validation-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING")
    if launch_blocking not in (None, "", "0"):
        raise RuntimeError("lifetime shadow smoke requires asynchronous CUDA submission")

    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    import torch
    import _galp_direct_dct as native
    from galp.torch import DirectDctReader

    reader = DirectDctReader(args.manifest)
    pipeline = reader.pipeline(PROFILE).start([[0, 1, 2, 3]])
    batch = next(pipeline)
    shadow = batch._native._enable_lifetime_shadow_for_test()
    y = batch.y
    cbcr = batch.cbcr
    side = torch.cuda.Stream()
    batch.record_stream(side)
    with torch.cuda.stream(side):
        value = y.square().mean() + cbcr.square().mean()
    side.synchronize()

    before_release = dict(shadow.snapshot)
    del y, cbcr, batch
    gc.collect()
    native.manual_reclaim()
    after_release = dict(shadow.snapshot)
    pipeline.close()

    expected_before = {
        "consumer_dependency_count": 2,
        "explicit_consumer_dependency_count": 1,
        "storage_reference_count": 2,
        "released_storage_reference_count": 0,
        "release_requested": False,
    }
    for key, expected in expected_before.items():
        if before_release[key] != expected:
            raise AssertionError(f"before-release {key}: {before_release[key]!r} != {expected!r}")
    expected_after = {
        "released_storage_reference_count": 2,
        "all_storage_references_released": True,
        "release_requested": True,
        "pending_consumer_dependency_count": 0,
        "legacy_reclaim_eligible": True,
        "native_reclaim_eligible": True,
        "differential": "equivalent_eligible",
    }
    for key, expected in expected_after.items():
        if after_release[key] != expected:
            raise AssertionError(f"after-release {key}: {after_release[key]!r} != {expected!r}")
    if not bool(torch.isfinite(value).item()):
        raise AssertionError("consumer produced a non-finite value")

    payload = {
        "schema": "galp-phase4-lifetime-shadow-smoke-v1",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "native_extension": str(Path(native.__file__).resolve()),
        "before_release": before_release,
        "after_release": after_release,
        "result": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE 4B LIFETIME SHADOW SMOKE PASS")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
