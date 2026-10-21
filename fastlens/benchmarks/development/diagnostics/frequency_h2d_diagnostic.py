#!/usr/bin/env python3
"""Separate CUPTI capture of compressed payload and projection-descriptor transfers."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'build/fastlens/torch')]
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.system_dct_major.common import parse_manifest
from fastlens.benchmarks.profiling.input_motivation import cache_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('4090', 'H100'), default='4090')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid(args.gpu)
    import torch
    import _fastlens_direct_dct as native
    torch.set_num_threads(8)
    source = ROOT / 'fastlens/data/compressed/imagenet512_val_block_major'
    manifest = parse_manifest(source / 'manifest.bin')
    shards = [manifest['shards'][i] for i in (0, 16, 32, 47)]
    ids = [list(range(s['first_global_image_index'], s['first_global_image_index'] + s['image_count']))
           for s in shards]
    assert cache_pages([Path(s['fls_path']) for s in shards], warm=True)['resident_fraction'] == 1
    reader = native.DirectDctReader(str(source / 'manifest.bin'))
    options = dict(layout='compact', cache_capacity_mib=0, plan_cache_capacity=0,
                   decode_batch_rowgroups=64, rowgroup_prefetch_workers=8)
    # Cover every measured source shard and every frequency budget outside capture.
    checked = [batch_ids[:8] for batch_ids in ids]
    for k in (8, 16, 24, 32, 48, 64):
        for batch_ids in checked:
            a = reader.read_batch(batch_ids, dct_coeffs=f'first:{k}', **options)
            b = reader.read_batch(batch_ids, dct_coeffs='all', **options)
            torch.testing.assert_close(a.coefficients, b.coefficients[:, :k], atol=0, rtol=0)
            del a, b
    torch.cuda.synchronize()
    results = {}
    torch.cuda.cudart().cudaProfilerStart()
    for arm in ('selected', 'late'):
        stats = []
        torch.cuda.nvtx.range_push(f'frequency.{arm}')
        for batch_ids in ids:
            batch = reader.read_batch(batch_ids, dct_coeffs='first:8' if arm == 'selected' else 'all', **options)
            tensor = batch.coefficients
            output = tensor if arm == 'selected' else tensor[:, :8].contiguous()
            torch.cuda.synchronize()
            stats.append(batch.execution_stats)
            del output, tensor, batch
        torch.cuda.nvtx.range_pop()
        results[arm] = stats
    torch.cuda.cudart().cudaProfilerStop()
    args.output.write_text(json.dumps(dict(k=8, native_batches=results, correctness_checked_image_ids=checked,
                                          checked_budgets=[8, 16, 24, 32, 48, 64], coefficients_exact=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
