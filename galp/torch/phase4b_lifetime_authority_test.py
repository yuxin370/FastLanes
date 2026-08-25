#!/usr/bin/env python3
"""Targeted Phase-4B production lifetime-authority checks."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


PROFILE = "rgbnomore-validation-v1"


def _load(args: Any) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    import torch
    import _galp_direct_dct as native
    import _galp_phase4_outstanding_test as harness
    from galp.torch import DirectDctReader

    return torch, native, harness, DirectDctReader


def _reclaim_until(native: Any, predicate: Any, timeout_seconds: float = 2.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    reclaimed = 0
    while time.monotonic() < deadline:
        reclaimed += int(native.manual_reclaim())
        if predicate():
            return reclaimed
        time.sleep(0.001)
    return reclaimed


def _same_stream(torch: Any, native: Any, DirectDctReader: Any, manifest: Path) -> dict[str, Any]:
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE).start([[0, 1, 2, 3]])
    if pipeline._native._lifetime_backend_for_test != "native":
        raise AssertionError("same-stream case did not select native lifetime")
    batch = next(pipeline)
    shadow = batch._native._enable_lifetime_shadow_for_test()
    y = batch.y
    value = y.square().mean()
    expected_finite = bool(torch.isfinite(value).item())
    _ = pipeline.metrics
    del y, batch
    pipeline.close()
    del pipeline
    gc.collect()
    reclaimed = _reclaim_until(native, lambda: bool(shadow.snapshot["reclaim_executed"]))
    snapshot = dict(shadow.snapshot)
    if not expected_finite or not snapshot["reclaim_executed"] or snapshot["backing_storage_present"]:
        raise AssertionError(f"same-stream native reclaim failed: {snapshot}")
    return {"reclaimed": reclaimed, "snapshot": snapshot, "result": "PASS"}


def _explicit_side_stream(
    torch: Any,
    native: Any,
    harness: Any,
    DirectDctReader: Any,
    manifest: Path,
    gate_timeout_ms: int,
    memcheck_mode: bool,
) -> dict[str, Any]:
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE).start([[0, 1, 2, 3]])
    if pipeline._native._lifetime_backend_for_test != "native":
        raise AssertionError("explicit-stream case did not select native lifetime")
    batch = next(pipeline)
    shadow = batch._native._enable_lifetime_shadow_for_test()
    y = batch.y
    batch._native._wait_for_producer_completion_for_test()
    _ = pipeline.metrics
    expected = float(y[tuple(0 for _ in y.shape)].item())
    pointer = int(y.data_ptr())
    backing_bytes = int(y.numel() * y.element_size())
    side = torch.cuda.Stream()
    batch.record_stream(side)
    handle = harness.arm_outstanding_consumer(
        pointer, int(side.cuda_stream), int(side.device_index), gate_timeout_ms
    )
    try:
        if memcheck_mode:
            # Compute Sanitizer may make cudaEventQuery wait for instrumented
            # work.  Do not use that altered timing as an eligibility oracle;
            # the normal run above owns the deterministic before/after proof.
            # Releasing all references here still exercises the dangerous
            # path under memcheck, and the device watchdog keeps it finite.
            del y, batch
            pipeline.close()
            del pipeline
            gc.collect()
            native.manual_reclaim()
            if not handle.wait_consumer(gate_timeout_ms + 7000):
                raise AssertionError("instrumented side consumer did not complete")
            checksum = float(handle.consumer_checksum)
            _reclaim_until(native, lambda: bool(shadow.snapshot["reclaim_executed"]))
            after = dict(shadow.snapshot)
            if not after["reclaim_executed"] or after["backing_storage_present"]:
                raise AssertionError(f"instrumented native backing did not reclaim safely: {after}")
            if math.isnan(expected):
                if not math.isnan(checksum):
                    raise AssertionError("instrumented sentinel changed NaN semantics")
            elif checksum != expected:
                raise AssertionError(f"instrumented sentinel mismatch: {checksum!r} != {expected!r}")
            gate_timed_out = bool(handle.gate_timed_out)
            handle.cleanup(gate_timeout_ms + 7000)
            return {
                "mode": "memcheck",
                "gate_timed_out": gate_timed_out,
                "after_consumer_complete": after,
                "expected_checksum": expected,
                "consumer_checksum": checksum,
                "result": "PASS",
            }
        if handle.consumer_complete:
            raise AssertionError("side consumer completed before gate release")
        del y, batch
        pipeline.close()
        del pipeline
        gc.collect()
        native.manual_reclaim()
        before = dict(shadow.snapshot)
        resources_before = dict(native._lifetime_reclaim_stats_for_test())
        if (
            not before["producer_complete"]
            or before["native_reclaim_eligible"]
            or before["reclaim_executed"]
            or not before["backing_storage_present"]
            or handle.consumer_complete
        ):
            raise AssertionError(f"native backing was not protected before consumer completion: {before}")

        early_probe_pointers = [
            int(value) for value in native._device_pool_reuse_probe_for_test(backing_bytes, 3)
        ]
        if pointer in early_probe_pointers:
            raise AssertionError("backing region was reused while the registered consumer was outstanding")

        handle.release_gate()
        if not handle.wait_consumer(7000) or handle.gate_timed_out:
            raise AssertionError("side consumer did not complete through host gate release")
        checksum = float(handle.consumer_checksum)
        _reclaim_until(native, lambda: bool(shadow.snapshot["reclaim_executed"]))
        after = dict(shadow.snapshot)
        resources_after = dict(native._lifetime_reclaim_stats_for_test())
        if not after["reclaim_executed"] or after["backing_storage_present"]:
            raise AssertionError(f"native backing was not reclaimed after consumer completion: {after}")
        if after["pending_consumer_dependency_count"] != 0:
            raise AssertionError("completed side-stream dependency remained pending")
        if math.isnan(expected):
            if not math.isnan(checksum):
                raise AssertionError("sentinel checksum changed NaN semantics")
        elif checksum != expected:
            raise AssertionError(f"sentinel checksum mismatch: {checksum!r} != {expected!r}")
        reuse_probe_pointers = [
            int(value) for value in native._device_pool_reuse_probe_for_test(backing_bytes, 5)
        ]
        if pointer not in reuse_probe_pointers:
            raise AssertionError(
                "released backing region was not observed in five deterministic same-shape reuse attempts"
            )
        handle.cleanup(7000)
    except BaseException:
        handle.release_gate()
        handle.cleanup(7000)
        raise
    return {
        "backing_pointer": pointer,
        "early_probe_pointers": early_probe_pointers,
        "reuse_probe_pointers": reuse_probe_pointers,
        "before_consumer_complete": before,
        "after_consumer_complete": after,
        "resources_before": resources_before,
        "resources_after": resources_after,
        "expected_checksum": expected,
        "consumer_checksum": checksum,
        "result": "PASS",
    }


def _legacy_rollback(torch: Any, native: Any, DirectDctReader: Any, manifest: Path) -> dict[str, Any]:
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE).start([[0, 1, 2, 3]])
    if pipeline._native._lifetime_backend_for_test != "legacy":
        raise AssertionError("rollback case did not select legacy lifetime")
    batch = next(pipeline)
    y = batch.y
    value = y.square().mean()
    torch.cuda.current_stream().synchronize()
    finite = bool(torch.isfinite(value).item())
    del y, batch
    pipeline.close()
    del pipeline
    gc.collect()
    reclaimed = _reclaim_until(native, lambda: True)
    if not finite:
        raise AssertionError("legacy rollback produced a non-finite value")
    return {"reclaimed": reclaimed, "result": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("same", "explicit", "rollback"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate-timeout-ms",
        type=int,
        default=5000,
        help="test-only device watchdog; increase only under instrumentation",
    )
    parser.add_argument(
        "--memcheck-mode",
        action="store_true",
        help="avoid timing assertions invalidated by Compute Sanitizer instrumentation",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError("Phase-4B lifetime tests require asynchronous CUDA submission")
    torch, native, harness, DirectDctReader = _load(args)
    if args.case == "rollback":
        result = _legacy_rollback(torch, native, DirectDctReader, args.manifest)
    elif args.case == "same":
        result = _same_stream(torch, native, DirectDctReader, args.manifest)
    else:
        result = _explicit_side_stream(
            torch,
            native,
            harness,
            DirectDctReader,
            args.manifest,
            args.gate_timeout_ms,
            args.memcheck_mode,
        )
    payload = {
        "schema": "galp-phase4b-lifetime-authority-v1",
        "case": args.case,
        "lifetime_env": os.environ.get("GALP_PHASE4_NATIVE_LIFETIME", "<unset>"),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "native_extension": str(Path(native.__file__).resolve()),
        "result": result,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"PHASE 4B {args.case.upper()} PASS")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
