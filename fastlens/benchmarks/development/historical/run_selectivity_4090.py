#!/usr/bin/env python3
"""Paired source-projection sweeps using the existing CNN control."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.system_dct_major.common import parse_manifest
from fastlens.benchmarks.profiling.input_motivation import cache_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('4090', 'H100'), default='4090')
    parser.add_argument('--spatial-crops', type=int, nargs='+', choices=(160, 256, 352, 448, 512),
                        default=(160, 256, 352, 448, 512), help='crop sizes for the spatial sweep')
    parser.add_argument('--experiment', choices=('spatial', 'joint', 'channels', 'coalescing', 'cold'), required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/selectivity_4090_20261001')
    parser.add_argument('--count', type=int, default=16384)
    args = parser.parse_args()
    source = ROOT / 'fastlens/data/compressed/imagenet512_val_block_major'
    manifest = parse_manifest(source / 'manifest.bin')
    paths = [Path(s['fls_path']) for s in manifest['shards']
             if s['first_global_image_index'] < args.count]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(args.gpu), DCTNET_PROFILE='mobilenet24',
               PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', TMPDIR=str(Path.home() / 'tmp'),
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    configurations = {
        'spatial': [(crop, 64) for crop in args.spatial_crops],
        'joint': [(crop, k) for crop in (256, 512) for k in (8, 64)],
        'channels': [(512, k) for k in (8, 16, 24, 32, 48, 64)],
        'cold': [(crop, 64) for crop in (160, 512)],
        'coalescing': [(crop, 64) for crop in (160, 352)],
    }[args.experiment]
    counters = ('requested_source_block_count', 'selected_vector_count', 'full_vector_count',
                'selected_coefficient_count', 'full_coefficient_count', 'decoded_coefficient_bytes',
                'selected_compressed_payload_bytes', 'rowgroup_storage_bytes_read',
                'workset_upload_dma_bytes', 'workset_upload_dma_count', 'planning_ms', 'decode_ms',
                'planless_transform_gpu_kernel_ms', 'coalesced_read_run_count', 'merged_gap_bytes',
                'bounded_exact_storage_bytes', 'bounded_physical_storage_bytes',
                'bounded_exact_extent_count', 'bounded_physical_run_count', 'bounded_coalesce_ms')
    rows = []
    pair = (('selected',) if args.experiment == 'channels' else
            ('exact', 'bounded') if args.experiment == 'coalescing' else ('selected', 'full'))
    for crop, k in configurations:
        outputs = {}
        for trial in range(-1, 4):
            # Correctness, one untimed warmup, then three alternating paired trials.
            arms = pair if trial % 2 else pair[::-1]
            for arm in arms:
                mode = 'full-source-decode' if arm == 'full' else 'vector-range-read-selected-decode'
                stage = 'correctness' if trial == -1 else f'trial_{trial}'
                directory = args.output / ('cold_raw' if args.experiment == 'cold' else 'raw') / f'g{crop}_k{k}_{arm}' / stage
                directory.mkdir(parents=True, exist_ok=True)
                count = 8 if trial == -1 else args.count
                target = directory / f'N_{count}.json'
                if not target.exists():
                    command = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable,
                               'fastlens/benchmarks/paper/shared/cnn_control.py', '--crop-size', str(crop),
                               '--source-coefficients', str(k), '--new-data', str(source),
                               '--input-geometry', 'source512', '--pushdown', 'off',
                               '--crop-execution-mode', mode, '--output-layout', 'projected',
                               '--native-runtime', 'b6', '--shards-per-activation', '1',
                               '--physical-prefix', '--batch-size', '64', '--count', str(count),
                               '--model-threads', '8', '--warmup-batches', '20',
                               '--baseline-dir', str(args.output / 'unused'), '--output-dir', str(directory)]
                    if trial == -1:
                        command.append('--verify')
                    if args.experiment == 'coalescing':
                        command += ['--read-amplification-cap', '1.0' if arm == 'exact' else '1.1']
                    if args.experiment == 'cold':
                        for path in paths:
                            with path.open('rb') as stream:
                                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        cache = cache_pages(paths)
                        (directory / 'cold_cache.json').write_text(json.dumps(cache, indent=2) + '\n')
                        if cache['resident_pages']:
                            raise RuntimeError('Payload pages remain resident after file-specific eviction')
                    run(command, directory, env, None if args.experiment == 'cold' else paths)
                raw = json.loads(target.read_text())
                assert raw['samples'] == count == len(set(raw['sample_ids']))
                assert raw['control']['source_coefficients'] == k
                assert raw['online_crop'][2:] == [crop, crop]
                if trial == -1:
                    assert raw['checks'] and all(c['native_grid_and_full_source_input_exact'] for c in raw['checks'])
                stats = raw['native_shards']
                if k == 64:
                    assert all(s['selected_coefficient_count'] == s['full_coefficient_count'] for s in stats)
                else:
                    assert sum(s['selected_coefficient_count'] for s in stats) < sum(s['full_coefficient_count'] for s in stats)
                if trial > 0:
                    outputs[arm, trial] = raw
                    row = dict(experiment=args.experiment, crop_pixels=crop,
                               retained_area=(crop / 512) ** 2, source_coefficients=k,
                               arm=arm, trial=trial, samples=count,
                               elapsed_s=raw['e2e_seconds'], images_per_s=count / raw['e2e_seconds'],
                               model_seconds=raw['model_seconds'], input_wait_seconds=raw['input_wait_seconds'],
                               top1=raw['top1'], raw_result=str(target.relative_to(ROOT)))
                    row.update({name: sum(s[name] for s in stats) for name in counters})
                    io = raw['control']
                    row['process_storage_read_bytes'] = io['process_io_end']['read_bytes'] - io['process_io_start']['read_bytes']
                    rows.append(row)
            if trial > 0 and len(pair) == 2:
                a, b = outputs[pair[0], trial], outputs[pair[1], trial]
                assert a['sample_ids'] == b['sample_ids'] and a['predictions'] == b['predictions']
                assert a['checkpoint'] == b['checkpoint']
                if args.experiment == 'coalescing':
                    for key in ('selected_vector_count', 'selected_compressed_payload_bytes'):
                        assert sum(s[key] for s in a['native_shards']) == sum(s[key] for s in b['native_shards'])
        destination = args.output / f'{args.experiment}.csv'
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        medians = {arm: statistics.median(outputs[arm, t]['e2e_seconds'] for t in (1, 2, 3))
                   for arm in pair}
        summary = dict(crop=crop, k=k, median_seconds=medians)
        if len(pair) == 2:
            summary['time_ratio'] = medians[pair[1]] / medians[pair[0]]
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
