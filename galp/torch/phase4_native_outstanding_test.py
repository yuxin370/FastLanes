#!/usr/bin/env python3
"""Deterministic Phase-4A native outstanding-consumer proof."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


PROFILE = "rgbnomore-validation-v1"


def _wait_for_shadow(native, shadow, *, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    snapshot = dict(shadow.snapshot)
    while time.monotonic() < deadline:
        native.manual_reclaim()
        snapshot = dict(shadow.snapshot)
        if snapshot["native_reclaim_eligible"]:
            return snapshot
        time.sleep(0.001)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError("native outstanding-consumer test requires asynchronous CUDA submission")
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    import torch
    import _galp_direct_dct as native
    import _galp_phase4_outstanding_test as harness
    from galp.torch import DirectDctReader

    reader = DirectDctReader(args.manifest)
    pipeline = reader.pipeline(PROFILE).start([[0, 1, 2, 3]])
    batch = next(pipeline)
    shadow = batch._native._enable_lifetime_shadow_for_test()
    y = batch.y
    if y.dtype != torch.float32:
        raise AssertionError(f"native sentinel consumer requires FP32 Y, got {y.dtype}")
    batch._native._wait_for_producer_completion_for_test()
    producer_ready = dict(shadow.snapshot)
    if not producer_ready["producer_complete"]:
        raise AssertionError("producer completion was not established before consumer arm")
    if producer_ready["release_requested"]:
        raise AssertionError("producer readiness check unexpectedly requested release")
    zero_index = tuple(0 for _ in y.shape)
    expected_checksum = float(y[zero_index].item())
    side = torch.cuda.Stream()
    batch.record_stream(side)

    handle = harness.arm_outstanding_consumer(
        int(y.data_ptr()), int(side.cuda_stream), int(side.device_index), 5000
    )
    try:
        if handle.consumer_complete:
            raise AssertionError("native consumer completed before gate release")
        del y, batch
        native.manual_reclaim()
        before_release = dict(shadow.snapshot)
        if handle.consumer_complete:
            raise AssertionError("native consumer completed during pre-release observation")
        if before_release["native_reclaim_eligible"]:
            raise AssertionError("native shadow became eligible while consumer was outstanding")
        if not before_release["producer_complete"]:
            raise AssertionError("producer completion regressed during consumer-only observation")
        if before_release["pending_consumer_dependency_count"] < 1:
            raise AssertionError("native shadow lost the registered side-stream dependency")

        handle.release_gate()
        if not handle.wait_consumer(7000):
            raise AssertionError("native consumer did not complete after gate release")
        if handle.gate_timed_out:
            raise AssertionError("device watchdog, rather than host release, opened the test gate")
        checksum = float(handle.consumer_checksum)
        after_release = _wait_for_shadow(native, shadow, timeout_seconds=1.0)
        if not after_release["native_reclaim_eligible"]:
            raise AssertionError("native shadow remained ineligible after consumer completion")
        if not after_release["producer_complete"]:
            raise AssertionError("producer completion regressed after consumer completion")
        if after_release["pending_consumer_dependency_count"] != 0:
            raise AssertionError("native shadow retained a completed consumer dependency")
        if math.isnan(expected_checksum):
            if not math.isnan(checksum):
                raise AssertionError("native sentinel checksum changed NaN payload semantics")
        elif checksum != expected_checksum:
            raise AssertionError(
                f"native sentinel checksum mismatch: {checksum!r} != {expected_checksum!r}"
            )
        handle.cleanup(7000)
        pipeline.close()
    except BaseException:
        handle.release_gate()
        handle.cleanup(7000)
        pipeline.close()
        raise

    payload = {
        "schema": "galp-phase4-native-outstanding-v1",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "native_extension": str(Path(native.__file__).resolve()),
        "harness_extension": str(Path(harness.__file__).resolve()),
        "gate": "single-block-system-atomic-with-device-watchdog",
        "consumer": "native-first-float-sentinel-read",
        "expected_checksum": expected_checksum,
        "consumer_checksum": checksum,
        "producer_ready_before_consumer_arm": producer_ready,
        "before_release": before_release,
        "after_release": after_release,
        "result": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE 4A NATIVE OUTSTANDING-CONSUMER PROOF PASS")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
