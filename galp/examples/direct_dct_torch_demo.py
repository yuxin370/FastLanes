#!/usr/bin/env python3
"""Minimal stay-on-GPU direct-DCT runtime demo.

Build with:
  cmake -S . -B build-galp-torch -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
    -DGALP_BUILD_TORCH=ON \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  cmake --build build-galp-torch --target _galp_direct_dct -j

Run with PYTHONPATH pointed at the extension output directory:
  PYTHONPATH=build-galp-torch/galp/torch \
    python3 galp/examples/direct_dct_torch_demo.py /path/to/manifest.bin
"""

from __future__ import annotations

import argparse

import torch

import _galp_direct_dct as galp_dct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--dct-coeffs", default="first:8")
    args = parser.parse_args()

    reader = galp_dct.DirectDctReader(args.manifest)
    image_ids = list(range(min(args.batch_size, reader.image_count)))
    batch = reader.read_batch(image_ids, crop=args.crop, dct_coeffs=args.dct_coeffs)

    coefficients = batch.coefficients
    if not coefficients.is_cuda:
        raise RuntimeError("expected a CUDA tensor backed by the GALP DCT runtime")
    if coefficients.dtype != torch.int16:
        raise RuntimeError(f"expected torch.int16 coefficients, got {coefficients.dtype}")
    block_count = batch.block_count
    coefficients_per_block = batch.coefficients_per_block
    if coefficients.ndim != 2 or tuple(coefficients.shape) != (block_count, coefficients_per_block):
        raise RuntimeError(
            "unexpected DCT tensor shape: "
            f"shape={tuple(coefficients.shape)} expected={(block_count, coefficients_per_block)}"
        )
    if tuple(coefficients.stride()) != (coefficients_per_block, 1):
        raise RuntimeError(f"expected compact [block, coeff] strides, got {tuple(coefficients.stride())}")
    if coefficients.data_ptr() != batch.device_data_ptr:
        raise RuntimeError(
            "expected torch tensor to wrap the GALP CUDA buffer: "
            f"tensor=0x{coefficients.data_ptr():x} galp=0x{batch.device_data_ptr:x}"
        )
    if batch.coefficient_count != coefficients.numel():
        raise RuntimeError(f"coefficient count mismatch: batch={batch.coefficient_count} tensor={coefficients.numel()}")
    if len(batch.global_image_ids) != len(image_ids) or len(batch.image_layouts) != len(image_ids):
        raise RuntimeError("image metadata count does not match requested image ids")
    if len(batch.block_metadata) != block_count:
        raise RuntimeError("block metadata count does not match tensor rows")
    if len(batch.selected_coefficients) != coefficients_per_block:
        raise RuntimeError("selected coefficient count does not match tensor columns")
    device_data_ptr = batch.device_data_ptr
    tensor_data_ptr = coefficients.data_ptr()
    tensor_shape = tuple(coefficients.shape)
    device = coefficients.device
    stats = batch.execution_stats
    del batch

    # Placeholder direct-DCT model input path: consume the DCT tensor on GPU after
    # releasing the Python batch wrapper and original tensor reference.
    features = coefficients.to(torch.float32).mean(dim=1)
    del coefficients
    print(
        f"images={len(image_ids)} blocks={block_count} "
        f"coefficients_per_block={coefficients_per_block} "
        f"tensor_shape={tensor_shape} feature_shape={tuple(features.shape)} "
        f"device={device} data_ptr=0x{tensor_data_ptr:x} "
        f"galp_ptr=0x{device_data_ptr:x} "
        f"selected_vectors={stats['selected_vector_count']} "
        f"full_vectors={stats['full_vector_count']} "
        f"decode_kernels={stats['decode_kernel_launch_count']}"
    )


if __name__ == "__main__":
    main()
