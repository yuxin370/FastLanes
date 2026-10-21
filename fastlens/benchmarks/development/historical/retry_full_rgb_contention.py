"""Retry only full RGB cells interrupted by observed target-GPU contention."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import OUT
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import DATA
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import PYTHON
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import read
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import save
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import report
from fastlens.benchmarks.development.historical.resume_full_rgb_epochs import accepted
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot


def wait_for_process(pid, expected_script):
    import psutil
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return
    except psutil.NoSuchProcess:
        return
    if expected_script not in ' '.join(process.cmdline()):
        raise RuntimeError(f'PID {pid} is no longer the expected process')
    print(f'Waiting for {pid} ({expected_script}); no process is signalled', flush=True)
    while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
        time.sleep(5)
    print(f'Finished waiting for {pid}', flush=True)


def main():
    # Both are observed recurring competitors, not hypothetical reservations.
    wait_for_process(2625766, 'fastlens/benchmarks/development/historical/resume_full_rgb_epochs.py')
    state = read(OUT / 'status.json')
    if state['status'] not in ('COMPLETE', 'FINISHED_WITH_FAILURES'):
        raise RuntimeError('Previous queue did not finish normally')
    jobs = [j for j in state['jobs'] if j['status'] == 'TODO' or
            (j['status'] == 'FAILED' and j.get('error', '').startswith('Target GPU contention:'))]
    if not jobs:
        print('No contention failures to retry', flush=True)
        return
    snapshot = read(OUT / 'runtime_snapshot_numa1.json')
    if any(file_sha(p) != sha for p, sha in snapshot['files'].items()):
        raise RuntimeError('Measured source changed; inspect before resuming')
    files = dict(snapshot['files'])
    files[str(Path(__file__).resolve())] = file_sha(__file__)
    snapshot = dict(files=files, fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    save(OUT / 'runtime_snapshot_contention_retry.json', snapshot)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(OUT / 'runtime_snapshot_contention_retry.json')
    references = {}
    for job in state['jobs']:
        if job['status'] == 'DONE':
            r = accepted(Path(job['output']))
            references.setdefault(job['model'], r)
    state.update(status='RUNNING', pid=os.getpid())
    devices = {x['requested']: x['selection'] for x in read(ROOT / 'results/inventory/device_mapping.json')}
    competitors = {'4090': (2680604, 'fastlens/benchmarks/development/diagnostics/resnet_stride_probe.py'),
                   'H100': (2603424, 'run_scale_phj_gds17_260930.py')}
    waited = set()
    for job in jobs:
        if job['gpu'] not in waited:
            report(state)
            wait_for_process(*competitors[job['gpu']])
            waited.add(job['gpu'])
        previous = Path(job['output'])
        destination = previous.parent / (previous.name + '_contention_retry')
        original_gate = OUT / 'checks' / job['gpu'] / job['model'] / job['pipeline']
        reuse_gate = ((original_gate / 'result.json').exists()
                      and read(original_gate / 'result.json')['correctness_passed']
                      and read(original_gate / 'run.command.json').get('returncode') == 0
                      and read(original_gate / 'run.command.json')['code_unchanged_during_run'])
        gate = original_gate if reuse_gate else original_gate.parent / (job['pipeline'] + '_contention_retry')
        destination.mkdir()
        if not reuse_gate:
            gate.mkdir()
        job.setdefault('previous_attempts', []).append(dict(output=str(previous), status=job['status'], error=job.get('error', 'Not started')))
        job.update(output=str(destination), status='RUNNING'); job.pop('error', None)
        report(state)
        interpreter = str(Path.home() / 'tmp/coordl-build-cu128/coordl-python') if job['pipeline'] == 'coordl' else PYTHON
        base = ['numactl', '--cpunodebind=0', '--membind=0', interpreter]
        options = ['--gpu', job['gpu'], '--model', job['model'], '--pipeline', job['pipeline'], '--trial', '0']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[job['gpu']], DCTNET_PROFILE=job['model'],
                   TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   OPENBLAS_NUM_THREADS='8', PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
                   PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        try:
            verify_snapshot()
            if not reuse_gate:
                run([*base, 'fastlens/benchmarks/paper/shared/training_window_rgb.py', *options, '--check', '--output', str(gate)], gate, env, None)
            check = read(gate / 'result.json'); reference = references[job['model']]
            if not check['correctness_passed']:
                raise ValueError('Short RGB correctness check failed')
            for key in ('initial_model_hash', 'precision', 'batch', 'accumulation'):
                if check[key] != reference[key]:
                    raise ValueError(f'RGB configuration mismatch: {key}')
            driver = 'fastlens/benchmarks/development/historical/full_ffcv_remote_cache.py' if job['pipeline'] == 'ffcv' else 'fastlens/benchmarks/paper/shared/training_window_rgb.py'
            run([*base, driver, *options, '--full-epoch', '--ffcv-root', str(DATA),
                 '--correctness-report', str(gate / 'result.json'), '--output', str(destination)], destination, env, None)
            r = accepted(destination)
            if r['sample_order_sha256'] != reference['sample_order_sha256']:
                raise ValueError('Full sample order differs from accepted RGB runs')
            save(destination / 'accepted.json', dict(correctness=True, comparison_type='deployment', raw_result=str(destination / 'result.json')))
            job['status'] = 'DONE'
        except (RuntimeError, ValueError, subprocess.CalledProcessError, FileNotFoundError) as error:
            job.update(status='FAILED', error=str(error))
            save(destination / 'failure.json', dict(error=str(error)))
            if 'Evaluation code changed' in str(error):
                raise
        finally:
            if job['pipeline'] == 'coordl':
                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
                cleanup_coordl_cache(destination); cleanup_coordl_cache(gate)
            report(state)
    state['status'] = 'COMPLETE' if all(j['status'] == 'DONE' for j in state['jobs']) else 'FINISHED_WITH_FAILURES'
    report(state)


if __name__ == '__main__':
    main()
