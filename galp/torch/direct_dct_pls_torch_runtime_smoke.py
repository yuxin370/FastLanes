#!/usr/bin/env python3
"""CUDA smoke for the experimental physical-PLS Direct-DCT adapter."""

from __future__ import annotations

import gc
import os
import torch
import _galp_direct_dct as native_backend

from galp.torch.experimental import DirectDctPlsPipeline


SKIP_RETURN_CODE = 77


def main() -> int:
    manifest = os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MANIFEST")
    mapping = os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MAPPING")
    mapping_sha256 = os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MAPPING_SHA256")
    if not manifest or not mapping or not mapping_sha256:
        print(
            "skipping: GALP_DIRECT_DCT_PLS_TEST_MANIFEST and "
            "GALP_DIRECT_DCT_PLS_TEST_MAPPING/MAPPING_SHA256 are not set"
        )
        return SKIP_RETURN_CODE
    if not torch.cuda.is_available():
        print("skipping: torch.cuda.is_available() is false")
        return SKIP_RETURN_CODE

    microbatch_images = int(os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MICROBATCH", "64"))
    pipeline = DirectDctPlsPipeline(
        manifest,
        mapping,
        training_seed=11997733,
        expected_mapping_sha256=mapping_sha256,
        segments_per_pool=4,
        microbatch_images=microbatch_images,
    ).start_epoch(7)
    pool = pipeline.next_pool()
    if pool.image_count > 4096 or pool.microbatch_count <= 0:
        raise RuntimeError("unexpected native PLS pool cardinality")
    microbatch = next(pool)
    y, cbcr, targets = microbatch.tensors
    torch.cuda.synchronize()

    if tuple(y.shape[1:]) != (1, 28, 28, 8, 8):
        raise RuntimeError(f"unexpected PLS Y shape: {tuple(y.shape)}")
    if tuple(cbcr.shape[1:]) != (2, 14, 14, 8, 8):
        raise RuntimeError(f"unexpected PLS CbCr shape: {tuple(cbcr.shape)}")
    if targets.shape != (y.shape[0], 1000):
        raise RuntimeError(f"unexpected PLS target shape: {tuple(targets.shape)}")
    if not bool(torch.isfinite(y).all() and torch.isfinite(cbcr).all() and torch.isfinite(targets).all()):
        raise RuntimeError("PLS pipeline emitted non-finite model tensors")
    if not torch.allclose(targets.sum(dim=1), torch.ones(y.shape[0], device=y.device)):
        raise RuntimeError("PLS Mixup targets do not sum to one")
    if len(microbatch.global_image_ids) != y.shape[0] or len(microbatch.labels) != y.shape[0]:
        raise RuntimeError("PLS model tensor and identity cardinalities differ")
    execution_stats = pool.execution_stats
    if not bool(execution_stats.get("uses_planless_fixed_transform")):
        raise RuntimeError("PLS runtime did not report planless fixed-transform execution")

    # Exercise the ownership boundary on a stream that was not current when
    # the native-backed tensors were created. Their deleter must not release
    # native backing storage until this consumer has finished.
    consumer_stream = torch.cuda.Stream(device=y.device)
    microbatch.record_stream(consumer_stream)
    with torch.cuda.stream(consumer_stream):
        cross_stream_checksum = y.square().sum() + cbcr.square().sum() + targets.sum()

    summary = (
        f"images={y.shape[0]} epoch={microbatch.epoch} pool={microbatch.pool_index} "
        f"pool_images={pool.image_count} "
        f"device={torch.cuda.get_device_name(0)} y={tuple(y.shape)} cbcr={tuple(cbcr.shape)} "
        f"targets={tuple(targets.shape)} selected_vectors={execution_stats['selected_vector_count']} "
        f"full_vectors={execution_stats['full_vector_count']} "
        f"physical_bytes={execution_stats['compressed_payload_bytes_read']}"
    )
    del y, cbcr, targets, microbatch
    del pool
    pipeline.close()
    del pipeline
    gc.collect()
    consumer_stream.synchronize()
    if not bool(torch.isfinite(cross_stream_checksum).item()):
        raise RuntimeError("cross-stream PLS consumer produced a non-finite checksum")
    del cross_stream_checksum, consumer_stream
    torch.cuda.synchronize()
    native_backend.reclaim_direct_dct_pls_pools()
    gc.collect()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
