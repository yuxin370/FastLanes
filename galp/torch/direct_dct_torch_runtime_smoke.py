#!/usr/bin/env python3
"""CUDA smoke for the stable GALP Direct-DCT PyTorch pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import _galp_direct_dct as native_diagnostics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader


SKIP_RETURN_CODE = 77


def _validate_batch(batch: object, image_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = batch.y
    cbcr = batch.cbcr
    if not y.is_cuda or not cbcr.is_cuda:
        raise RuntimeError("Direct-DCT profile outputs must remain on CUDA")
    if y.dtype != torch.float32 or cbcr.dtype != torch.float32:
        raise RuntimeError(f"expected FP32 profile output, got y={y.dtype} cbcr={cbcr.dtype}")
    if tuple(y.shape) != (image_count, 1, 28, 28, 8, 8):
        raise RuntimeError(f"unexpected Y shape: {tuple(y.shape)}")
    if tuple(cbcr.shape) != (image_count, 2, 14, 14, 8, 8):
        raise RuntimeError(f"unexpected CbCr shape: {tuple(cbcr.shape)}")
    if batch.layout != "transformed_dct_grid":
        raise RuntimeError(f"unexpected Direct-DCT layout: {batch.layout}")
    return y, cbcr


def main() -> int:
    manifest = os.environ.get("GALP_DIRECT_DCT_TEST_MANIFEST")
    if not manifest:
        print("skipping: GALP_DIRECT_DCT_TEST_MANIFEST is not set")
        return SKIP_RETURN_CODE
    if not torch.cuda.is_available():
        print("skipping: torch.cuda.is_available() is false")
        return SKIP_RETURN_CODE

    batch_size = int(os.environ.get("GALP_DIRECT_DCT_TEST_BATCH_SIZE", "8"))
    reader = DirectDctReader(manifest)
    image_ids = list(range(min(batch_size, reader.image_count)))
    if not image_ids:
        raise RuntimeError("Direct-DCT smoke manifest contains no images")

    profile_info = reader.profile_info(VALIDATION)
    pipeline = reader.pipeline(VALIDATION).start([image_ids])
    consumer_stream = torch.cuda.Stream()

    # Access the external tensors on the real consumer stream, then finish only
    # the producer work and release the pipeline metric owner's reference. This
    # isolates the tensor-storage deleter as the remaining lifetime mechanism.
    with torch.cuda.stream(consumer_stream):
        first = next(pipeline)
        first_y, first_cbcr = _validate_batch(first, len(image_ids))
    consumer_stream.synchronize()
    metrics = pipeline.metrics
    if not metrics.complete:
        raise RuntimeError("Direct-DCT producer metrics remained incomplete after synchronization")

    # Keep the consumer observably pending while the Python/native batch and
    # every external input tensor reference are destroyed. Reclaim must remain
    # impossible until work on this non-default stream reaches the recorded
    # storage-deleter events.
    consumer_done = torch.cuda.Event()
    with torch.cuda.stream(consumer_stream):
        # Five billion SM clock cycles leave a multi-second observation window
        # on the supported datacenter GPUs. This is test-only work used to keep
        # the registered consumer stream pending while reclaim is inspected.
        torch.cuda._sleep(5_000_000_000)
        first_features = first_y.square().mean() + first_cbcr.square().mean()
        consumer_done.record()
        del first, first_y, first_cbcr
    if consumer_done.query():
        raise RuntimeError("consumer-stream delay completed before lifetime verification")
    reclaimed_early = int(native_diagnostics.manual_reclaim())
    if reclaimed_early != 0:
        state = "completed during reclaim" if consumer_done.query() else "still pending"
        raise RuntimeError(f"reclaimed {reclaimed_early} Direct-DCT owners while consumer was {state}")
    consumer_stream.synchronize()
    reclaimed_after_completion = int(native_diagnostics.manual_reclaim())
    if reclaimed_after_completion == 0:
        raise RuntimeError("Direct-DCT owners were not reclaimable after the consumer stream completed")
    if not bool(torch.isfinite(first_features).item()):
        raise RuntimeError("non-default-stream Direct-DCT consumer produced a non-finite value")
    if metrics.physical_bytes <= 0 or metrics.logical_bytes <= 0:
        raise RuntimeError("Direct-DCT pipeline reported empty storage metrics")
    pipeline.close()

    print(
        f"profile={VALIDATION.id} runtime_policy={profile_info['runtime_policy_id']} "
        f"images={len(image_ids)} device={torch.cuda.get_device_name(0)} "
        f"consumer_stream={consumer_stream.cuda_stream} "
        f"reclaimed_after_completion={reclaimed_after_completion} "
        f"decode_ms={metrics.decode_ms:.3f} transform_ms={metrics.transform_ms:.3f} "
        f"logical_bytes={metrics.logical_bytes} physical_bytes={metrics.physical_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
