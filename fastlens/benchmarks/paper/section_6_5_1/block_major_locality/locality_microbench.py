#!/usr/bin/env python3
"""Replay shared, correlated and independent crops through two physical layouts."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'build/fastlens/torch')]
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.system_dct_major.common import parse_manifest, resolve_coefficient_selection
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
    roots = {name: ROOT / 'fastlens/data/compressed' / directory for name, directory in (
        ('block_major', 'imagenet512_val_block_major'), ('image_major_tiled', 'imagenet512_val_compact_v3'))}
    samples = json.loads((roots['block_major'] / 'samples.json').read_text())
    with (roots['image_major_tiled'] / 'index.csv').open() as stream:
        image_major_samples = list(csv.DictReader(stream))
    assert all(a['logical_sample_id'] == b['Filepath'] for a, b in zip(samples, image_major_samples))
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['FASTLENS_BLOCK_MAJOR_ACCESS_DIR'] = str(roots['block_major'] / 'access')
    readers = {name: native.DirectDctReader(str(root / 'manifest.bin')) for name, root in roots.items()}
    # Four separate source segments, with the same IDs and logical request order for both layouts.
    cohorts = [list(range(i * 1024, (i + 1) * 1024)) for i in (0, 16, 32, 47)]
    cache = {}
    for name, root in roots.items():
        m = parse_manifest(root / 'manifest.bin')
        paths = [Path(s['fls_path']) for s in m['shards'] if any(
            s['first_global_image_index'] <= i < s['first_global_image_index'] + s['image_count']
            for ids in cohorts for i in ids)]
        cache[name] = cache_pages(paths, warm=True)
        assert cache[name]['resident_fraction'] == 1
    frequencies = resolve_coefficient_selection('first:8')['resolved_natural_indices']
    transform = dict(y_output_width_blocks=32, y_output_height_blocks=32,
                     cbcr_output_width_blocks=32, cbcr_output_height_blocks=32,
                     crop_reference_width_blocks=64, crop_reference_height_blocks=64,
                     crop_origin_alignment_blocks=2, chroma_crop_scale_x=2, chroma_crop_scale_y=2,
                     allowed_chroma_sampling_ratios=[[1, 2, 1, 2]], require_all_coefficients=True,
                     output_dtype='float32', dequantize=True, clamp_min=-32768, clamp_max=32767,
                     output_add=0., output_scale=1.,
                     output_channels=[[c, f, 0., 1.] for c in range(3) for f in frequencies])
    options = dict(layout='transformed_dct_grid', grid_transform=transform, dct_coeffs='all',
                   cache_capacity_mib=0, plan_cache_capacity=0, decode_batch_rowgroups=64,
                   rowgroup_prefetch_workers=8, crop_execution_mode='vector-range-read-selected-decode',
                   decode_workset_capacity_mib=2048, transform_ctas_per_launch=512)
    counters = ('requested_source_block_count', 'selected_vector_count', 'full_vector_count',
                'rowgroup_storage_bytes_read', 'selected_compressed_payload_bytes',
                'workset_upload_dma_bytes', 'planning_ms', 'decode_ms', 'planless_transform_gpu_kernel_ms',
                'coalesced_read_run_count')
    rows = []
    for policy in ('shared', 'correlated', 'independent'):
        rng = random.Random(11997733)
        transforms = []
        for ids in cohorts:
            crops = []
            for _ in ids:
                x, y = ((128, 128) if policy == 'shared' else
                        (rng.randrange(6, 11) * 16, rng.randrange(6, 11) * 16) if policy == 'correlated' else
                        (rng.randrange(17) * 16, rng.randrange(17) * 16))
                crops.append(dict(crop=[x, y, 256, 256], horizontal_flip=False))
            transforms.append(crops)
        for batch_size in (64, 1024):
            requests = [(ids[start:start + batch_size], crops[start:start + batch_size])
                        for ids, crops in zip(cohorts, transforms) for start in range(0, len(ids), batch_size)]
            references = []
            for trial in range(4):
                names = ('block_major', 'image_major_tiled') if trial % 2 == 0 else ('image_major_tiled', 'block_major')
                for name in names:
                    stats, elapsed = [], 0.
                    for index, (ids, crops) in enumerate(requests):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        batch = readers[name].read_batch(ids, transforms=crops, **options)
                        tensor = batch.projected
                        torch.cuda.synchronize()
                        elapsed += time.perf_counter() - start
                        stats.append(batch.execution_stats)
                        assert stats[-1]['uses_planless_fixed_transform']
                        if trial == 0:
                            if name == 'block_major':
                                references.append(tensor.cpu())
                            else:
                                torch.testing.assert_close(tensor.cpu(), references[index], atol=0, rtol=0)
                        del tensor, batch
                    result = dict(policy=policy, batch_size=batch_size, layout=name, trial=trial,
                                  samples=sum(map(len, cohorts)), elapsed_s=elapsed, native_batches=stats,
                                  cache=cache[name], decode_workset_capacity_mib=2048,
                                  timing_scope='synchronous input preparation, no model; reader startup excluded')
                    (args.output / f'{policy}_b{batch_size}_{name}_{trial}.json').write_text(json.dumps(result, indent=2) + '\n')
                    if trial:
                        row = {k: result[k] for k in ('policy', 'batch_size', 'layout', 'trial', 'samples', 'elapsed_s')}
                        row.update({k: sum(s[k] for s in stats) for k in counters})
                        rows.append(row)
                    print(json.dumps({k: result[k] for k in ('policy', 'batch_size', 'layout', 'trial', 'elapsed_s')}), flush=True)
            with (args.output / 'locality.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == '__main__':
    main()
