"""Matched pipeline controls and preparation sweeps, serially on both paper GPUs."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.common import write_csv
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def read(path):
    return json.loads(path.read_text())


def profile_summary(directory, label):
    database = directory / 'capture.sqlite'
    subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(database),
                    str(directory / 'capture.nsys-rep')], check=True)
    with sqlite3.connect(database) as connection:
        (start, end), = connection.execute(
            "SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE COALESCE(n.text,s.value)=?", (label,)).fetchall()
        copies = connection.execute(
            'SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY '
            'WHERE copyKind=1 AND start<? AND end>?', (end, start)).fetchall()
        assert copies and all(start <= a <= b <= end for a, b, _ in copies)
        kernels = connection.execute(
            'SELECT s.value,SUM(k.end-k.start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k '
            'JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? '
            'GROUP BY s.value', (start, end)).fetchall()
    write_csv(directory / 'kernels.csv',
              [dict(kernel=n, cumulative_ns=t, launches=c) for n, t, c in kernels],
              ['kernel', 'cumulative_ns', 'launches'])
    return dict(total_h2d_bytes=sum(size for _, _, size in copies),
                h2d_copy_count=len(copies), h2d_cumulative_s=sum(b-a for a, b, _ in copies)/1e9,
                decode_kernel_s=sum(t for n, t, _ in kernels if 'decompress_dispatch' in n)/1e9,
                diagnostic_raw=str(database))


def core_controls(output, env, timing_only=False):
    source = ROOT / 'fastlens/data/compressed/imagenet512_val_block_major'
    preparation = read(ROOT / 'results/preparation/raw_int16_phase2_16384/preparation.json')
    assert preparation['samples'] == 16384
    cases = [('direct', source, 'projected'), ('grid', source, 'grid'),
             ('compressed', Path(preparation['compressed']), 'projected'),
             ('raw_int16', Path(preparation['raw_int16']), 'projected')]
    references, rows = {}, []
    stages = [('warmup', 16384)] if timing_only else [('correctness', 8), ('warmup', 16384)]
    stages += [(f'trial_{t}', 16384) for t in (1, 2, 3)]
    if not timing_only:
        stages += [('diagnostic', 16384), ('memory', 16384)]
    for stage_index, (stage, count) in enumerate(stages):
        ordered = cases if stage_index % 2 == 0 else cases[::-1]
        for name, source, layout in ordered:
            directory = output / name / stage
            directory.mkdir(parents=True, exist_ok=False)
            paths = [Path(s['fls_path']) for s in parse_manifest(source / 'manifest.bin')['shards']
                     if s['first_global_image_index'] < count]
            command = [sys.executable, 'fastlens/benchmarks/paper/shared/cnn_control.py', '--crop-size', '448',
                       '--source-coefficients', '64', '--new-data', str(source),
                       '--input-geometry', 'source512', '--pushdown', 'off',
                       '--crop-execution-mode', 'vector-range-read-selected-decode',
                       '--output-layout', layout, '--native-runtime', 'b6',
                       '--shards-per-activation', '1', '--physical-prefix', '--batch-size', '64',
                       '--model-threads', '8', '--warmup-batches', '20', '--count', str(count),
                       '--baseline-dir', str(output / 'unused'), '--output-dir', str(directory)]
            if stage == 'correctness':
                command.append('--verify')
            if stage == 'diagnostic':
                command += ['--diagnostic-window', str(directory / 'window.json')]
                command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                           '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                           '--output', str(directory / 'capture'), *command]
            if stage == 'memory':
                with (directory / 'gpu_process_memory.csv').open('x') as telemetry:
                    monitor = subprocess.Popen(['nvidia-smi',
                        '--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory',
                        '--format=csv,noheader,nounits', '-lms', '100'], stdout=telemetry)
                    try:
                        run(command, directory, env, paths, bind_to_gpu=True)
                    finally:
                        monitor.terminate()
                        monitor.wait()
            else:
                run(command, directory, env, paths, bind_to_gpu=True)
            raw = read(directory / f'N_{count}.json')
            assert raw['samples'] == len(set(raw['sample_ids'])) == count
            if stage == 'correctness':
                assert raw['checks'] and all(c['native_grid_and_full_source_input_exact'] for c in raw['checks'])
            if count in references:
                for key in ('sample_ids', 'predictions', 'checkpoint', 'precision', 'batch_size'):
                    assert raw[key] == references[count][key], (name, stage, key)
            else:
                references[count] = raw
            if stage.startswith('trial_'):
                stats = raw['native_shards']
                row = dict(gpu=raw['device_name'], case=name, trial=int(stage[-1]), samples=count,
                           elapsed_s=raw['e2e_seconds'], images_per_s=count/raw['e2e_seconds'],
                           raw_result=str(directory / f'N_{count}.json'))
                row.update({key: sum(s[key] for s in stats) for key in (
                    'rowgroup_storage_bytes_read', 'workset_upload_dma_bytes', 'decoded_coefficient_bytes',
                    'selected_vector_count', 'planning_ms', 'decode_ms')})
                rows.append(row)
                write_csv(output / 'timings.csv', rows, list(rows[0]))
            elif stage == 'diagnostic':
                (directory / 'summary.json').write_text(json.dumps(
                    profile_summary(directory, 'control.measured_pass'), indent=2) + '\n')
            elif stage == 'memory':
                window, residency = raw['control'], []
                with (directory / 'gpu_process_memory.csv').open() as stream:
                    for stamp, pid, uuid, mib in csv.reader(stream):
                        ns = int(datetime.datetime.strptime(stamp.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()*1e9)
                        if int(pid) == window['pid'] and window['start_unix_ns'] <= ns <= window['end_unix_ns']:
                            residency.append(int(mib)*1024*1024)
                assert residency, 'No process-memory observations inside measured window'
                (directory / 'summary.json').write_text(json.dumps(dict(
                    process_peak_bytes=max(residency), torch_peak_allocated_bytes=window['torch_peak_allocated_bytes'],
                    torch_peak_reserved_bytes=window['torch_peak_reserved_bytes']), indent=2) + '\n')
    summary = []
    for name, _, _ in cases:
        selected = [r for r in rows if r['case'] == name]
        row = dict(case=name, median_s=statistics.median(r['elapsed_s'] for r in selected),
                   min_s=min(r['elapsed_s'] for r in selected), max_s=max(r['elapsed_s'] for r in selected))
        if not timing_only:
            row.update(read(output / name / 'diagnostic/summary.json'))
            row.update(read(output / name / 'memory/summary.json'))
        summary.append(row)
    write_csv(output / 'summary.csv', summary, list(summary[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('4090', 'H100'), nargs='+', default=('4090', 'H100'))
    parser.add_argument('--experiments', choices=('core', 'locality', 'frequency'), nargs='+',
                        default=('core', 'locality', 'frequency'))
    parser.add_argument('--timing-only', action='store_true',
                        help='repeat core warmup and paired timing; retain the existing correctness and resource runs')
    parser.add_argument('--projection-coefficients', type=int, nargs='+', choices=(8, 16, 24, 32, 48, 64),
                        default=(8, 16, 24, 32, 48, 64))
    args = parser.parse_args()
    for gpu in args.gpu:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), DCTNET_PROFILE='mobilenet24',
                   PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', TMPDIR=str(Path.home() / 'tmp'),
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                   PYTHONDONTWRITEBYTECODE='1', FASTLENS_NSYS_FINE='0')
        for experiment in args.experiments:
            output = args.output.resolve() / gpu / experiment
            print('BEGIN', gpu, experiment, flush=True)
            if experiment == 'core':
                core_controls(output, env, timing_only=args.timing_only)
            else:
                script = 'fastlens/benchmarks/paper/section_6_5_1/block_major_locality/locality_microbench.py' if experiment == 'locality' else 'fastlens/benchmarks/paper/section_6_4/projection_and_staging/projection_microbench.py'
                output.mkdir(parents=True, exist_ok=False)
                extra = ['--coefficients', *map(str, args.projection_coefficients)] if experiment == 'frequency' else []
                run([sys.executable, script, '--gpu', gpu, '--output', str(output), *extra],
                    output, env, None, bind_to_gpu=True)
            print('COMPLETE', gpu, experiment, flush=True)


if __name__ == '__main__':
    main()
