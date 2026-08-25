#!/usr/bin/env python3
"""Deterministic, guarded F-001 lifetime baseline for Phase 4A.

The known-bad case keeps a second tensor-storage reference alive.  This lets
the test prove that the legacy queue declares the getter stream eligible while
the real side-stream consumer is pending without allowing the backing region
to be freed and read after free during baseline collection.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROFILE = "rgbnomore-validation-v1"
CONSUMER_DELAY_CYCLES = 5_000_000_000
LEGACY_RELEASE_DELAY_CYCLES = 250_000_000


class HarnessInvalid(RuntimeError):
    """The test failed to establish its required observation window."""


def _reclaim_until(native: Any, *, want_positive: bool, attempts: int = 20) -> int:
    total = 0
    for _ in range(attempts):
        total += int(native.manual_reclaim())
        if (total > 0) == want_positive:
            break
        time.sleep(0.01)
    return total


def _batch(DirectDctReader: Any, manifest: Path) -> tuple[Any, Any]:
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE).start([[0, 1, 2, 3]])
    return pipeline, next(pipeline)


def _finish_producer_and_metrics(pipeline: Any, torch: Any) -> Any:
    """Remove producer/metrics ownership before probing tensor-storage lifetime.

    The synchronization is test setup only.  It happens before the delayed
    consumer is queued, so it cannot make an unsafe consumer lifetime pass.
    """

    torch.cuda.synchronize()
    metrics = pipeline.metrics
    if not metrics.complete:
        raise HarnessInvalid("producer metrics remained incomplete after test setup synchronization")
    return metrics


def _require_outstanding(done: Any, label: str) -> None:
    if done.query():
        raise HarnessInvalid(f"{label} completed before the lifetime observation point")


def _same_stream_correctness_case(
    DirectDctReader: Any, torch: Any, native: Any, manifest: Path
) -> dict[str, Any]:
    """Validate the ordinary fast path without requiring a pending queue entry."""

    pipeline, batch = _batch(DirectDctReader, manifest)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = batch.y
        value = y.square().mean()
    stream.synchronize()
    metrics = pipeline.metrics
    if not metrics.complete:
        raise HarnessInvalid("same-stream producer metrics remained incomplete")
    del y, batch
    gc.collect()
    externally_reclaimed = int(native.manual_reclaim())
    pipeline.close()
    if not bool(torch.isfinite(value).item()):
        raise AssertionError("supported same-stream consumer produced a non-finite value")
    return {
        "consumer_stream": int(stream.cuda_stream),
        "consumer_completed": True,
        "external_manual_reclaim_count": externally_reclaimed,
        "release_observation": (
            "manual_reclaim_observed" if externally_reclaimed else "eligible_for_immediate_internal_reclaim"
        ),
        "result": "PASS",
    }


def _outstanding_consumer_defers_reclaim_case(
    DirectDctReader: Any, torch: Any, native: Any, manifest: Path
) -> dict[str, Any]:
    """Force the supported consumer stream to remain outstanding at release."""

    pipeline, batch = _batch(DirectDctReader, manifest)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = batch.y
    _finish_producer_and_metrics(pipeline, torch)

    done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(CONSUMER_DELAY_CYCLES)
        value = y.square().mean()
        done.record()
        del y, batch
    _require_outstanding(done, "same-stream delayed consumer")
    gc.collect()
    early = _reclaim_until(native, want_positive=False, attempts=1)
    _require_outstanding(done, "same-stream delayed consumer")
    stream.synchronize()
    after = _reclaim_until(native, want_positive=True)
    pipeline.close()
    if early != 0 or after == 0 or not bool(torch.isfinite(value).item()):
        raise AssertionError(
            f"outstanding same-stream deferral failed: early={early} after={after}"
        )
    return {
        "consumer_stream": int(stream.cuda_stream),
        "consumer_was_outstanding": True,
        "reclaimed_while_outstanding": early,
        "reclaimed_after_completion": after,
        "result": "PASS",
    }


def _f001_guarded_case(DirectDctReader: Any, torch: Any, native: Any, manifest: Path) -> dict[str, Any]:
    pipeline, batch = _batch(DirectDctReader, manifest)
    # Both external storages are created on the default stream. cbcr_guard
    # deliberately keeps the native allocation alive after the y deleter's
    # legacy eligibility decision, preventing real UAF in the baseline run.
    y = batch.y
    cbcr_guard = batch.cbcr
    pointer = int(batch._native.y_device_data_ptr)
    _finish_producer_and_metrics(pipeline, torch)

    side = torch.cuda.Stream()
    done = torch.cuda.Event()
    with torch.cuda.stream(side):
        torch.cuda._sleep(CONSUMER_DELAY_CYCLES)
        value = y.square().mean()
        done.record()

    # Make the getter-observed default stream's release event deterministic:
    # it is initially pending, then becomes eligible while the much longer
    # real side-stream consumer is still blocked.  cbcr_guard keeps the actual
    # backing allocation alive, so baseline collection cannot trigger UAF.
    default_stream = torch.cuda.default_stream()
    with torch.cuda.stream(default_stream):
        torch.cuda._sleep(LEGACY_RELEASE_DELAY_CYCLES)
    del y, batch
    gc.collect()
    _require_outstanding(done, "F-001 side-stream consumer")
    before_getter_stream_completion = _reclaim_until(native, want_positive=False, attempts=1)
    default_stream.synchronize()
    _require_outstanding(done, "F-001 side-stream consumer")
    eligible_while_consumer_pending = _reclaim_until(native, want_positive=True)
    side.synchronize()
    del cbcr_guard
    gc.collect()
    after_guard_release = int(native.manual_reclaim())
    pipeline.close()
    if before_getter_stream_completion != 0 or eligible_while_consumer_pending == 0:
        raise AssertionError(
            "F-001 eligibility mismatch: "
            f"before={before_getter_stream_completion} while_pending={eligible_while_consumer_pending}"
        )
    if not bool(torch.isfinite(value).item()):
        raise AssertionError("guarded F-001 side-stream consumer produced a non-finite value")
    return {
        "backing_pointer": pointer,
        "getter_stream": int(default_stream.cuda_stream),
        "actual_consumer_stream": int(side.cuda_stream),
        "consumer_pending_when_legacy_became_eligible": True,
        "reclaimed_before_getter_stream_completion": before_getter_stream_completion,
        "legacy_reference_reclaimed_while_actual_consumer_pending": eligible_while_consumer_pending,
        "external_reclaim_after_guard_release": after_guard_release,
        "guard_prevented_actual_free": True,
        "result": "CURRENTLY_UNSAFE_REPRODUCED",
    }


def _git_head(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _cmake_source_root(module_path: Path) -> Path:
    cache = module_path.parents[1] / "CMakeCache.txt"
    if not cache.is_file():
        raise HarnessInvalid(f"missing CMake cache for module path: {cache}")
    prefix = "CMAKE_HOME_DIRECTORY:INTERNAL="
    for line in cache.read_text().splitlines():
        if line.startswith(prefix):
            return Path(line[len(prefix) :]).resolve()
    raise HarnessInvalid(f"CMAKE_HOME_DIRECTORY is missing from {cache}")


def _provenance(args: Any, galp: Any, galp_torch: Any, native: Any) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    module_path = args.module_path.resolve()
    galp_file = Path(galp.__file__).resolve()
    galp_torch_file = Path(galp_torch.__file__).resolve()
    native_file = Path(native.__file__).resolve()
    cmake_source_root = _cmake_source_root(module_path)
    expected_package_root = source_root / "galp"
    if not galp_file.is_relative_to(expected_package_root):
        raise HarnessInvalid(f"galp imported from {galp_file}, outside {expected_package_root}")
    if not galp_torch_file.is_relative_to(expected_package_root / "torch"):
        raise HarnessInvalid(
            f"galp.torch imported from {galp_torch_file}, outside {expected_package_root / 'torch'}"
        )
    if native_file.parent != module_path:
        raise HarnessInvalid(f"_galp_direct_dct imported from {native_file}, expected {module_path}")
    if cmake_source_root != source_root:
        raise HarnessInvalid(
            f"extension build source is {cmake_source_root}, but --source-root is {source_root}"
        )
    binding_source = source_root / "galp" / "torch" / "direct_dct_torch.cpp"
    if native_file.stat().st_mtime < binding_source.stat().st_mtime:
        raise HarnessInvalid(
            f"extension {native_file} is older than binding source {binding_source}; rebuild before baseline"
        )
    return {
        "source_root": str(source_root),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "<unset>"),
        "cmake_source_root": str(cmake_source_root),
        "git_head": _git_head(source_root),
        "galp_file": str(galp_file),
        "galp_torch_file": str(galp_torch_file),
        "native_extension_file": str(native_file),
        "native_extension_mtime_ns": native_file.stat().st_mtime_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING")
    if launch_blocking not in (None, "", "0"):
        raise HarnessInvalid(
            "Phase-4 stream-lifetime tests require asynchronous CUDA submission; "
            f"CUDA_LAUNCH_BLOCKING={launch_blocking!r} invalidates the observation window"
        )

    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    os.environ.pop("GALP_PHASE3_NATIVE_DELEGATE", None)
    import torch
    import _galp_direct_dct as native
    import galp
    import galp.torch as galp_torch
    from galp.torch import DirectDctReader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    provenance = _provenance(args, galp, galp_torch, native)
    print("PHASE 4A PROVENANCE")
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)

    payload: dict[str, Any] = {
        "schema": "galp-phase4-f001-baseline-v4",
        "provenance": provenance,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    payload.update(
        {
            "supported_same_stream": _same_stream_correctness_case(
                DirectDctReader, torch, native, args.manifest
            ),
            "outstanding_consumer_defers_reclaim": _outstanding_consumer_defers_reclaim_case(
                DirectDctReader, torch, native, args.manifest
            ),
            "known_f001_guarded": _f001_guarded_case(
                DirectDctReader, torch, native, args.manifest
            ),
        }
    )
    payload["result"] = "BASELINE_RECORDED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE 4A F-001 BASELINE RECORDED")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
