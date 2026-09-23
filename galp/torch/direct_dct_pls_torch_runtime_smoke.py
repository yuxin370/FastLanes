#!/usr/bin/env python3
"""CUDA smoke for the experimental physical-PLS Direct-DCT adapter."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import tempfile
import torch
import _galp_direct_dct as native_backend

from galp.torch.experimental import DirectDctPlsPipeline
from galp.profiles.rgbnomore import TRAINING_PLS

from create_pls_smoke_fixture import create_fixture


SKIP_RETURN_CODE = 77


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpeg-tool")
    parser.add_argument("--access-tool")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        if os.environ.get("GALP_REQUIRE_CUDA_SMOKE") == "1":
            raise RuntimeError("CUDA-enabled PyTorch and a working GPU are required")
        print("skipping: torch.cuda.is_available() is false")
        return SKIP_RETURN_CODE
    manifest = os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MANIFEST")
    mapping = os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MAPPING")
    mapping_sha256 = os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MAPPING_SHA256")
    if any((manifest, mapping, mapping_sha256)) and not all((manifest, mapping, mapping_sha256)):
        raise RuntimeError("external PLS fixture requires manifest, mapping and mapping SHA-256")
    if manifest:
        return run_smoke(manifest, mapping, mapping_sha256)
    if not args.jpeg_tool or not args.access_tool:
        if os.environ.get("GALP_REQUIRE_CUDA_SMOKE") == "1":
            raise RuntimeError("PLS smoke requires fixture encoder tools or an external fixture")
        print("skipping: build GALP tools for the synthetic PLS fixture, or supply an external fixture")
        return SKIP_RETURN_CODE
    with tempfile.TemporaryDirectory(prefix="galp-pls-smoke-") as directory:
        manifest, mapping, mapping_sha256 = create_fixture(
            Path(directory), args.jpeg_tool, args.access_tool
        )
        return run_smoke(manifest, mapping, mapping_sha256, synthetic=True)


def run_smoke(manifest, mapping, mapping_sha256, *, synthetic=False) -> int:
    microbatch_images = 2 if synthetic else int(os.environ.get("GALP_DIRECT_DCT_PLS_TEST_MICROBATCH", "64"))
    baseline = native_backend._lifetime_reclaim_stats_for_test()
    pipeline = DirectDctPlsPipeline(
        manifest,
        mapping,
        training_seed=11997733,
        expected_mapping_sha256=mapping_sha256,
        segments_per_pool=4,
        microbatch_images=microbatch_images,
        profile=TRAINING_PLS,
    ).start_epoch(7)
    pool = pipeline.next_pool()
    if pool.image_count > 4096 or pool.microbatch_count <= 0:
        raise RuntimeError("unexpected native PLS pool cardinality")
    microbatch = next(pool)
    y, cbcr, targets = microbatch.tensors
    torch.cuda.synchronize()

    if any(tensor.dtype != torch.float32 or not tensor.is_cuda for tensor in (y, cbcr, targets)):
        raise RuntimeError("PLS must return CUDA float32 tensors")
    if any(tensor.device != y.device for tensor in (cbcr, targets)):
        raise RuntimeError("PLS tensor CUDA devices disagree")
    if synthetic:
        if sorted(zip(microbatch.global_image_ids, microbatch.labels)) != [(0, 7), (1, 23)]:
            raise RuntimeError("PLS sample identities/labels differ from the synthetic input")
        if y.shape[0] != 2 or pool.image_count != 2 or pool.microbatch_count != 1:
            raise RuntimeError("synthetic PLS smoke must consume exactly one two-image batch")

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
    expected_checksum = (y.square().sum() + cbcr.square().sum() + targets.sum()).item()
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
    if cross_stream_checksum.item() != expected_checksum:
        raise RuntimeError("PLS tensor contents changed across the ownership/lifetime boundary")
    del cross_stream_checksum, consumer_stream
    torch.cuda.synchronize()
    native_backend.reclaim_direct_dct_pls_pools()
    gc.collect()
    final = native_backend._lifetime_reclaim_stats_for_test()
    if final["native"]["pending_reclaim_count"] != baseline["native"]["pending_reclaim_count"]:
        raise RuntimeError("PLS smoke left native owners pending after consumer completion")
    if final["native"]["reclaimed_batch_count"] <= baseline["native"]["reclaimed_batch_count"]:
        raise RuntimeError("PLS smoke did not reclaim native tensor backing")
    for pool_name in ("device_pool", "pinned_pool"):
        if final[pool_name]["in_use_bytes"] != baseline[pool_name]["in_use_bytes"]:
            raise RuntimeError(f"PLS smoke left allocations in {pool_name}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
