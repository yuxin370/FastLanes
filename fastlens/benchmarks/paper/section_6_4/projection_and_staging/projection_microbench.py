#!/usr/bin/env python3
"""Raw coefficient projection: early column reads versus late GPU selection.

Measures storage-to-contiguous-int16-tensor preparation, without a model or resize.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'build/fastlens/torch')]
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.system_dct_major.common import parse_manifest
from fastlens.benchmarks.profiling.input_motivation import cache_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('4090', 'H100'), default='4090')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coefficients', type=int, nargs='+', choices=(8, 16, 24, 32, 48, 64),
                        default=(8, 16, 24, 32, 48, 64))
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid(args.gpu)
    import torch
    import _fastlens_direct_dct as native
    torch.set_num_threads(8)
    source = ROOT / 'fastlens/data/compressed/imagenet512_val_block_major'
    manifest = parse_manifest(source / 'manifest.bin')
    shards = [manifest['shards'][i] for i in (0, 16, 32, 47)]
    requests = [list(range(s['first_global_image_index'], s['first_global_image_index'] + s['image_count']))
                for s in shards]
    paths = [Path(s['fls_path']) for s in shards]
    args.output.mkdir(parents=True, exist_ok=True)
    residency = cache_pages(paths, warm=True)
    assert residency['resident_fraction'] == 1
    reader = native.DirectDctReader(str(source / 'manifest.bin'))
    options = dict(layout='compact', cache_capacity_mib=0, plan_cache_capacity=0,
                   decode_batch_rowgroups=64, rowgroup_prefetch_workers=8)
    rows = []
    counters = ('requested_source_block_count', 'selected_vector_count', 'full_vector_count',
                'selected_coefficient_count', 'full_coefficient_count',
                'selected_compressed_payload_bytes', 'rowgroup_storage_bytes_read',
                'workset_upload_dma_bytes', 'decoded_coefficient_bytes', 'planning_ms', 'decode_ms',
                'gather_ms', 'coalesced_read_run_count', 'decoded_gather_item_count',
                'decoded_projection_item_count', 'projection_item_build_ms')
    for k in args.coefficients:
        # Check actual coefficients, not just downstream predictions, outside timing.
        for ids in requests:
            a = reader.read_batch(ids[:8], dct_coeffs=f'first:{k}', **options)
            b = reader.read_batch(ids[:8], dct_coeffs='all', **options)
            torch.testing.assert_close(a.coefficients, b.coefficients[:, :k], atol=0, rtol=0)
            del a, b
        for trial in range(4):
            for arm in (('selected', 'late') if trial % 2 else ('late', 'selected')):
                target = args.output / f'k{k}_{arm}_trial{trial}.json'
                stats = []
                elapsed = 0.
                for ids in requests:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    batch = reader.read_batch(ids, dct_coeffs=f'first:{k}' if arm == 'selected' else 'all', **options)
                    tensor = batch.coefficients
                    output = tensor if arm == 'selected' else tensor[:, :k].contiguous()
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    assert output.shape == (len(ids) * 6144, k)
                    stats.append(batch.execution_stats)
                    assert stats[-1]['decoded_projection_item_count'] == 0
                    assert stats[-1]['decoded_gather_item_count'] == len(ids) * 6144
                    del output, tensor, batch
                if arm == 'late':
                    assert all(s['selected_coefficient_count'] == s['full_coefficient_count'] for s in stats)
                result = dict(k=k, arm=arm, trial=trial, samples=sum(map(len, requests)),
                              elapsed_s=elapsed, native_batches=stats, image_ids=requests,
                              gpu=torch.cuda.get_device_name(), cache=residency,
                              coefficients_exact=True, correctness_checked_image_ids=[ids[:8] for ids in requests],
                              timing_scope='read, decode, compact gather, late selection; excludes reader startup')
                target.write_text(json.dumps(result, indent=2) + '\n')
                if trial:
                    row = {key: result[key] for key in ('k', 'arm', 'trial', 'samples', 'elapsed_s')}
                    row['images_per_s'] = result['samples'] / elapsed
                    row.update({name: sum(s[name] for s in stats) for name in counters})
                    rows.append(row)
                print(json.dumps({key: result[key] for key in ('k', 'arm', 'trial', 'samples', 'elapsed_s')}), flush=True)
        with (args.output / 'frequency.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
