#!/usr/bin/env python3
"""Matched P2 controls on a verified coefficient-identical shard subset."""
import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_vitti import execute
from fastlens.benchmarks.common import gpu_uuid


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('4090', 'H100'), required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    preparation = read(args.prepared / 'preparation.json')
    count = preparation['samples']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(args.gpu), DCTNET_PROFILE='efun',
               TMPDIR=str(Path.home() / 'tmp'), PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    root = ROOT / 'results/ablation/raw' / args.gpu
    cases = [(f'crop_{crop}_{arm}', crop, 64, arm, 'projected', 'compressed')
             for crop in (512, 448, 352, 256) for arm in ('on', 'off')]
    cases += [(f'frequency_32_{arm}', 512, 32, arm, 'projected', 'compressed') for arm in ('on', 'off')]
    cases += [('construction_grid', 448, 64, 'on', 'grid', 'compressed'),
              ('compression_raw', 448, 64, 'on', 'projected', 'raw_int16')]
    results = {}
    for name, crop, k, arm, layout, representation in cases:
        base = root / name
        common = [sys.executable, 'fastlens/benchmarks/paper/shared/cnn_control.py', '--crop-size', str(crop),
                  '--source-coefficients', str(k), '--new-data', preparation[representation],
                  '--input-geometry', 'source512', '--pushdown', 'off', '--crop-execution-mode',
                  'full-source-decode' if arm == 'off' else 'vector-range-read-selected-decode',
                  '--output-layout', layout, '--native-runtime', 'b6', '--shards-per-activation', '1',
                  '--physical-prefix', '--batch-size', '64', '--baseline-dir', str(base / 'unused')]
        check = base / 'correctness'
        execute([*common, '--count', '8', '--verify', '--output-dir', str(check)], check.with_suffix('.log'), env)
        gate = read(check / 'N_8.json')
        if not gate['checks'] or not all(c['native_grid_and_full_source_input_exact'] for c in gate['checks']):
            raise ValueError(f'tensor correctness gate failed: {check}')
        observed = []
        trial, target = 0, 3
        while trial <= target:
            output = base / f'trial_{trial}'
            execute([*common, '--count', str(count), '--output-dir', str(output)], output.with_suffix('.log'), env)
            raw = read(output / f'N_{count}.json')
            if raw['samples'] != count or len(set(raw['sample_ids'])) != count:
                raise ValueError(f'sample coverage failed: {output}')
            if trial:
                observed.append(raw)
            if trial == 3:
                values = [count / r['e2e_seconds'] for r in observed]
                if statistics.stdev(values) / statistics.mean(values) > .02:
                    target = 5
            trial += 1
        first = observed[0]
        if any(r['sample_ids'] != first['sample_ids'] or r['predictions'] != first['predictions'] for r in observed):
            raise ValueError(f'trial order/prediction mismatch: {base}')
        # One independent diagnostic; its elapsed time never enters performance rows.
        diagnostic = base / 'diagnostic'
        diagnostic.mkdir(exist_ok=True)
        execute(['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                 '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--output', str(diagnostic / 'capture'),
                 *common, '--count', str(count), '--diagnostic-window', str(diagnostic / 'window.json'),
                 '--output-dir', str(diagnostic)], diagnostic.with_suffix('.log'), env)
        export = diagnostic / 'export.log'
        if not export.with_suffix('.command.json').exists():
            subprocess.run([sys.executable, 'fastlens/benchmarks/execution/run_logged.py', '--log', str(export), '--',
                            'nsys', 'export', '--type', 'sqlite', '--output', str(diagnostic / 'capture.sqlite'),
                            str(diagnostic / 'capture.nsys-rep')], env=env, check=True)
        profiled = read(diagnostic / f'N_{count}.json')
        if profiled['sample_ids'] != first['sample_ids'] or profiled['predictions'] != first['predictions']:
            raise ValueError(f'diagnostic correctness differs: {diagnostic}')
        with sqlite3.connect(diagnostic / 'capture.sqlite') as connection:
            (start, end), = connection.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value)='control.measured_pass'").fetchall()
            copies = connection.execute('SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1 AND start<? AND end>?', (end, start)).fetchall()
            if any(a < start or b > end for a, b, _ in copies):
                raise ValueError('H2D crosses diagnostic window')
        rows = []
        for trial, raw in enumerate(observed, 1):
            shards = raw['native_shards']
            total = lambda key: sum(s[key] for s in shards)
            metadata = read((base / f'trial_{trial}').with_suffix('.command.json'))
            rows.append(dict(timestamp=metadata['timestamp'], git_commit=metadata['git_commit'],
                gpu=raw['device_name'], model='eFUN-DCT', domain='dct', pipeline='imagelanes',
                precision='fp32', batch_size=64, trial=trial, samples=count, case=name,
                retained_fraction=(crop/512)**2, source_coefficients=k, pushdown=arm,
                output_layout=layout, representation=representation,
                storage_bytes=sum(p.stat().st_size for p in Path(preparation[representation]).glob('*.fls')),
                elapsed_s=raw['e2e_seconds'], images_per_s=count/raw['e2e_seconds'],
                requested_source_blocks=total('requested_source_block_count'),
                selected_vectors=total('selected_vector_count'),
                dependency_payload_bytes=total('selected_compressed_payload_bytes'),
                issued_read_bytes=total('rowgroup_storage_bytes_read'),
                h2d_bytes=sum(size for _, _, size in copies), h2d_copy_count=len(copies),
                h2d_memcpy_s=sum(b-a for a,b,_ in copies)/1e9,
                preprocessing_kernel_ms=total('planless_transform_gpu_kernel_ms'),
                native_decode_ms=total('decode_ms'),
                torch_peak_allocated_bytes=raw['peak_cuda_allocated_bytes'],
                accuracy=raw['top1']/100, correctness_status='PASS',
                raw_result=str(base / f'trial_{trial}' / f'N_{count}.json'),
                diagnostic_raw=str(diagnostic / 'capture.sqlite'),
                notes='Representative shard subset, warm cache; H2D from separate diagnostic; dependency payload excludes static metadata; no physical NVMe measurement'))
        results[name] = (first, rows)
    pairs = [(f'crop_{crop}_on', f'crop_{crop}_off') for crop in (512,448,352,256)]
    pairs += [('frequency_32_on','frequency_32_off'), ('crop_448_on','construction_grid'),
              ('crop_448_on','compression_raw')]
    for left, right in pairs:
        a, b = results[left][0], results[right][0]
        if a['sample_ids'] != b['sample_ids'] or a['predictions'] != b['predictions']:
            raise ValueError(f'matched prediction/order gate failed: {left}, {right}')
        if right == 'construction_grid' and sum(s['rowgroup_storage_bytes_read'] for s in a['native_shards']) != sum(s['rowgroup_storage_bytes_read'] for s in b['native_shards']):
            raise ValueError('direct/grid construction changed selected input bytes')
    groups = dict(crop_pushdown=[name for name in results if name.startswith('crop_')],
                  frequency_pushdown=['frequency_32_on','frequency_32_off'],
                  tensor_construction=['crop_448_on','construction_grid'],
                  compression=['crop_448_on','compression_raw'])
    for name, members in groups.items():
        rows = [row for member in members for row in results[member][1]]
        write_csv(ROOT / 'results/ablation' / f'{name}_{args.gpu}.csv', rows)
        combined = []
        for source in sorted((ROOT / 'results/ablation').glob(f'{name}_*.csv')):
            with source.open() as stream:
                combined.extend(csv.DictReader(stream))
        write_csv(ROOT / 'results/ablation' / f'{name}.csv', combined)
    table = []
    for name, members in groups.items():
        for member in members:
            rows = results[member][1]
            values = [r['images_per_s'] for r in rows]
            table.append(dict(experiment=name, gpu=args.gpu, case=member, model='eFUN-DCT',
                samples=count, trials=len(rows), median_images_per_s=statistics.median(values),
                min_images_per_s=min(values), max_images_per_s=max(values),
                cv=statistics.stdev(values)/statistics.mean(values),
                measurement_status='PROVISIONAL' if statistics.stdev(values)/statistics.mean(values) > .02 else 'VALIDATED',
                issued_read_bytes=rows[0]['issued_read_bytes'], h2d_bytes=rows[0]['h2d_bytes'],
                correctness_status='PASS', raw_results=';'.join(r['raw_result'] for r in rows)))
    write_csv(ROOT / 'results' / f'ablation_table_{args.gpu}.csv', table)
    combined = []
    for source in sorted((ROOT / 'results').glob('ablation_table_*.csv')):
        with source.open() as stream:
            combined.extend(csv.DictReader(stream))
    write_csv(ROOT / 'results/ablation_table.csv', combined)
    (root / 'correctness.json').write_text(json.dumps(dict(matched_pairs=pairs, passed=True,
        sample_count=count, preparation=str(args.prepared / 'preparation.json')),indent=2)+'\n')


if __name__ == '__main__':
    main()
