#!/usr/bin/env python3
"""Deterministic GPU proof for bounded Native Direct-DCT output overlap."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


PROFILE = "rgbnomore-validation-v1"


def _wait_until(
    predicate: Callable[[], bool],
    *,
    native: Any | None = None,
    timeout_seconds: float = 7.0,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if native is not None:
            native.manual_reclaim()
        if predicate():
            return True
        time.sleep(0.001)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-timeout-ms", type=int, default=5000)
    parser.add_argument("--memcheck-mode", action="store_true")
    args = parser.parse_args()

    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError("output-overlap proof requires asynchronous CUDA submission")
    if os.environ.get("GALP_DIRECT_DCT_OUTPUT_SLOT_CAPACITY", "2") != "2":
        raise RuntimeError("deterministic overlap proof requires output slot capacity 2")

    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    import torch
    import _galp_direct_dct as native
    import _galp_phase4_outstanding_test as harness
    from galp.torch import DirectDctReader

    batches = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]
    reader = DirectDctReader(args.manifest)
    pipeline = reader.pipeline(PROFILE).start(batches)
    first = next(pipeline)
    if first.global_image_ids != batches[0]:
        raise AssertionError("first Native batch order mismatch")
    first_y = first.y
    first._native._wait_for_producer_completion_for_test()
    expected = float(first_y[tuple(0 for _ in first_y.shape)].item())
    side = torch.cuda.Stream()
    first.record_stream(side)
    handle = harness.arm_outstanding_consumer(
        int(first_y.data_ptr()),
        int(side.cuda_stream),
        int(side.device_index),
        args.gate_timeout_ms,
    )

    try:
        if not _wait_until(lambda: bool(pipeline._native.ready)):
            raise AssertionError("Batch N+1 was not submitted while Batch N consumer was gated")
        before_second = dict(pipeline._native._native_state_for_test)
        if before_second["live_output_slots"] != 2:
            raise AssertionError(f"second output slot was not live: {before_second}")
        if not args.memcheck_mode and handle.consumer_complete:
            raise AssertionError("Batch N consumer completed before the overlap observation")

        second = next(pipeline)
        if second.global_image_ids != batches[1]:
            raise AssertionError("second Native batch order mismatch")
        second_y = second.y
        if not _wait_until(
            lambda: int(pipeline._native._native_state_for_test["output_slot_waiters"])
            >= 1
        ):
            raise AssertionError("Batch N+2 did not wait behind the two-slot bound")
        while_two_live = dict(pipeline._native._native_state_for_test)
        if (
            while_two_live["live_output_slots"] != 2
            or while_two_live["peak_live_output_slots"] != 2
            or bool(pipeline._native.ready)
        ):
            raise AssertionError(f"two-slot bound was not enforced: {while_two_live}")

        del first_y, first
        gc.collect()
        native.manual_reclaim()
        before_release = dict(pipeline._native._native_state_for_test)
        if not args.memcheck_mode:
            if handle.consumer_complete:
                raise AssertionError("gated Batch N consumer completed before host release")
            if before_release["live_output_slots"] != 2 or bool(pipeline._native.ready):
                raise AssertionError(
                    "Batch N slot was reused before its registered consumer completed"
                )

        handle.release_gate()
        if not handle.wait_consumer(args.gate_timeout_ms + 7000):
            raise AssertionError("Batch N side consumer did not complete after gate release")
        if not args.memcheck_mode and handle.gate_timed_out:
            raise AssertionError("device watchdog, rather than host release, opened the gate")
        checksum = float(handle.consumer_checksum)
        if math.isnan(expected):
            if not math.isnan(checksum):
                raise AssertionError("overlap sentinel changed NaN semantics")
        elif checksum != expected:
            raise AssertionError(f"overlap sentinel mismatch: {checksum!r} != {expected!r}")

        if not _wait_until(lambda: bool(pipeline._native.ready), native=native):
            raise AssertionError("Batch N+2 did not proceed after Batch N became reclaimable")
        third = next(pipeline)
        if third.global_image_ids != batches[2]:
            raise AssertionError("third Native batch order mismatch")
        third_y = third.y
        after_reuse = dict(pipeline._native._native_state_for_test)
        if after_reuse["peak_live_output_slots"] != 2:
            raise AssertionError(f"output-slot peak exceeded the capacity: {after_reuse}")

        result = {
            "schema": "galp-native-output-overlap-v1",
            "result": "PASS",
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "uuid": str(torch.cuda.get_device_properties(0).uuid),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            "native_extension": str(Path(native.__file__).resolve()),
            "consumer_checksum": checksum,
            "expected_checksum": expected,
            "before_second": before_second,
            "while_two_live": while_two_live,
            "before_consumer_release": before_release,
            "after_reuse": after_reuse,
            "ordering": "N+1_SUBMITTED_BEFORE_N_CONSUMER_COMPLETE",
            "bounded_slots": "PEAK_2",
        }
        del third_y, third, second_y, second
        pipeline.close()
        handle.cleanup(args.gate_timeout_ms + 7000)
    except BaseException:
        handle.release_gate()
        handle.cleanup(args.gate_timeout_ms + 7000)
        pipeline.close()
        raise

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("GALP NATIVE OUTPUT OVERLAP PASS")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
