"""Retry failed full-Beton cells with missing-range prewarming after existing work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import OUT
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import DATA
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import PYTHON
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import read
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import save
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import report
from fastlens.benchmarks.development.historical.resume_full_rgb_epochs import accepted
from fastlens.benchmarks.development.historical.retry_full_rgb_contention import wait_for_process
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot


def main():
    cache_node = int(os.environ.get('FASTLENS_FFCV_CACHE_MEMORY_NODE', '0'))
    for pid, script in ((439000, 'fastlens/benchmarks/development/historical/resume_full_rgb_frozen_20261006.py'),
                        (503108, 'fastlens/benchmarks/development/historical/run_prepared_frequency_20261006.py')):
        wait_for_process(pid, script)
    state = read(OUT / 'status.json')
    if state['status'] not in ('COMPLETE', 'FINISHED_WITH_FAILURES'):
        raise RuntimeError('Previous full-epoch queue has not finished normally')
    jobs = []
    for job in state['jobs']:
        if job['gpu'] != 'H100' or job['pipeline'] != 'ffcv':
            continue
        if job['status'] == 'DONE':
            accepted(Path(job['output']))
        elif job['status'] == 'FAILED' and 'Source page cache is not fully warm' in (Path(job['output']) / 'run.log').read_text():
            metadata = read(Path(job['output']) / 'run.command.json')
            assert metadata['returncode'] == 1 and metadata['code_unchanged_during_run']
            assert not (Path(job['output']) / 'measurement_active').exists()
            jobs.append(job)
    if not jobs:
        print('No full-FFCV cache failures remain; no replacement run', flush=True)
        return
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(OUT / 'runtime_snapshot_frozen_20261006.json')
    verify_snapshot()
    files = dict(read(OUT / 'runtime_snapshot_frozen_20261006.json')['files'])
    for path in (Path(__file__).resolve(), ROOT / 'fastlens/benchmarks/development/historical/full_ffcv_missing_cache.py'):
        files[str(path)] = file_sha(path)
    snapshot = OUT / f'runtime_snapshot_ffcv_missing_node{cache_node}_chunked_query_20261006.json'
    save(snapshot, dict(files=files, fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()))
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(snapshot)
    device, = [d['selection'] for d in read(ROOT / 'results/inventory/device_mapping.json') if d['requested'] == 'H100']
    for job in jobs:
        previous = Path(job['output'])
        old_command = read(previous / 'run.command.json')['command']
        gate = Path(old_command[old_command.index('--correctness-report') + 1])
        check, metadata = read(gate), read(gate.parent / 'run.command.json')
        assert check['correctness_passed'] and metadata['returncode'] == 0 and metadata['code_unchanged_during_run']
        reference = accepted(Path(next(j['output'] for j in state['jobs'] if j['status'] == 'DONE'
            and j['gpu'] == '4090' and j['pipeline'] == 'ffcv' and j['model'] == job['model'])))
        for key in ('initial_model_hash', 'precision', 'batch', 'accumulation', 'seed',
                    'augmentation_configuration', 'compile', 'tf32', 'effective_batch_size'):
            assert check[key] == reference[key], key
        destination = previous.parent / f'ffcv_missing_cache_node{cache_node}_chunked_query_20261006'
        destination.mkdir()
        job.setdefault('previous_attempts', []).append(dict(output=str(previous), status=job['status'], error=job['error']))
        job.update(output=str(destination), status='RUNNING')
        job.pop('error')
        state.update(status='RUNNING', pid=os.getpid())
        report(state)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, DCTNET_PROFILE=job['model'],
                   TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                   PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        try:
            run(['numactl', '--cpunodebind=0', '--membind=0', PYTHON, 'fastlens/benchmarks/development/historical/full_ffcv_missing_cache.py',
                 '--gpu', 'H100', '--model', job['model'], '--pipeline', 'ffcv', '--trial', '0', '--full-epoch',
                 '--ffcv-root', str(DATA), '--correctness-report', str(gate), '--output', str(destination)],
                destination, env, None)
            result = accepted(destination)
            assert result['sample_order_sha256'] == reference['sample_order_sha256']
            save(destination / 'accepted.json', dict(correctness=True, comparison_type='deployment',
                 raw_result=str(destination / 'result.json'), runtime_snapshot=str(snapshot),
                 cache_placement=str(destination / 'cache_placement.json'),
                 notes=f'One replacement for observed premeasurement cache failure. Excluded helper reads only missing page ranges on memory node {cache_node}; training CPU/memory remain node 0. Exact archived training entry retained. Not isolated GPU scaling.'))
            job.update(status='DONE', acceptance_notes=f' Excluded missing-range prewarming on memory node {cache_node}; original entry SHA recorded; not isolated GPU scaling.')
        except (RuntimeError, ValueError, AssertionError, subprocess.CalledProcessError, FileNotFoundError) as error:
            job.update(status='FAILED', error=str(error))
            save(destination / 'failure.json', dict(error=str(error)))
            if 'Evaluation code changed' in str(error) or isinstance(error, subprocess.CalledProcessError) and error.returncode == 86:
                state['status'] = 'BLOCKED_CODE_CHANGE'
                report(state)
                raise
        report(state)
    state['status'] = 'COMPLETE' if all(j['status'] == 'DONE' for j in state['jobs']) else 'FINISHED_WITH_FAILURES'
    report(state)


if __name__ == '__main__':
    main()
