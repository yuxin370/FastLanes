"""Finish only missing authorized H100 RGB epochs after the FFCV window queue."""
import datetime
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
from fastlens.benchmarks.execution.revision_results import verify_snapshot


def main():
    wait_for_process(4187868, 'resume_20261006.sh')
    for task, gpu in [('training_raw', '4090'), ('training', 'H100'), ('training_raw', 'H100')]:
        rows = read(ROOT / 'results/ffcv_jpeg_20261005' / task / gpu / 'summary.json')
        if len(rows) != 5 or any(r['trials'] != 3 for r in rows):
            raise RuntimeError('Preceding FFCV window matrix is incomplete; epochs not started')
    state = read(OUT / 'status.json')
    if state['status'] != 'FINISHED_WITH_FAILURES':
        raise RuntimeError('Prior full-epoch campaign state changed; inspect before resuming')
    jobs = [j for j in state['jobs'] if j['status'] != 'DONE']
    if len(jobs) != 8 or any(j['gpu'] != 'H100' for j in jobs):
        raise RuntimeError('Expected precisely eight missing H100 epochs')
    snapshot = OUT / 'runtime_snapshot_20261006.json'
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(snapshot)
    verify_snapshot()
    references = {}
    for job in state['jobs']:
        if job['status'] == 'DONE':
            result = accepted(Path(job['output']))
            references.setdefault(job['model'], result)
    save(OUT / 'status_before_resume_20261006.json', state)
    state.update(status='RUNNING', pid=os.getpid(),
                 runtime_revision='20261006: fresh input/update/tail gates; historical valid epochs retained',
                 resumed_at=datetime.datetime.now().astimezone().isoformat())
    devices = {x['requested']: x['selection'] for x in read(ROOT / 'results/inventory/device_mapping.json')}
    for job in jobs:
        previous = job['output']
        destination = Path(previous).parent / (job['pipeline'] + '_revision_20261006')
        gate = OUT / 'checks/H100' / job['model'] / (job['pipeline'] + '_revision_20261006')
        destination.mkdir()
        gate.mkdir()
        job.setdefault('previous_attempts', []).append(dict(output=previous, status=job['status'], error=job.get('error', '')))
        job.update(output=str(destination), status='RUNNING')
        job.pop('error', None)
        report(state)
        interpreter = str(Path.home() / 'tmp/coordl-build-cu128/coordl-python') if job['pipeline'] == 'coordl' else PYTHON
        base = ['numactl', '--cpunodebind=0', '--membind=0', interpreter]
        options = ['--gpu', 'H100', '--model', job['model'], '--pipeline', job['pipeline'], '--trial', '0']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices['H100'], DCTNET_PROFILE=job['model'],
                   TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   OPENBLAS_NUM_THREADS='8', PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
                   PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        tail = OUT / 'tail_checks/revision_20261006' / job['pipeline']
        try:
            verify_snapshot()
            if not (tail / 'result.json').exists():
                tail.mkdir(parents=True, exist_ok=True)
                run([*base, 'fastlens/benchmarks/development/diagnostics/check_rgb_epoch_tail.py', '--pipeline', job['pipeline'],
                     '--output', str(tail)], tail, env, None)
            tail_result = read(tail / 'result.json')
            if not (tail_result['passed'] and tail_result['samples'] == 143 and tail_result['batch_sizes'] == [64, 64, 15]):
                raise ValueError('Current reader tail check failed')
            run([*base, 'fastlens/benchmarks/paper/shared/training_window_rgb.py', *options, '--check', '--output', str(gate)], gate, env, None)
            check = read(gate / 'result.json')
            reference = references[job['model']]
            if not check['correctness_passed']:
                raise ValueError('Current input/update check failed')
            for key in ['initial_model_hash', 'precision', 'batch', 'accumulation',
                        'augmentation_configuration', 'seed', 'compile', 'tf32', 'effective_batch_size']:
                if check[key] != reference[key]:
                    raise ValueError(f'Current training contract differs from accepted RGB epoch: {key}')
            driver = 'fastlens/benchmarks/development/historical/full_ffcv_remote_cache.py' if job['pipeline'] == 'ffcv' else 'fastlens/benchmarks/paper/shared/training_window_rgb.py'
            run([*base, driver, *options, '--full-epoch', '--ffcv-root', str(DATA),
                 '--correctness-report', str(gate / 'result.json'), '--output', str(destination)], destination, env, None)
            result = accepted(destination)
            if result['sample_order_sha256'] != reference['sample_order_sha256']:
                raise ValueError('Full sample order differs from accepted RGB epochs')
            save(destination / 'accepted.json', dict(correctness=True, comparison_type='deployment',
                 raw_result=str(destination / 'result.json'), runtime_snapshot=str(snapshot),
                 notes='Fresh revision; matching training contract and full sample sequence. Historical 4090 implementation retained, not an isolated GPU scaling control.'))
            job['status'] = 'DONE'
        except (RuntimeError, ValueError, subprocess.CalledProcessError, FileNotFoundError) as error:
            job.update(status='FAILED', error=str(error))
            save(destination / 'failure.json', dict(error=str(error)))
            if 'Evaluation code changed' in str(error):
                state['status'] = 'BLOCKED_CODE_CHANGE'
                report(state)
                raise
        finally:
            if job['pipeline'] == 'coordl':
                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
                for path in [destination, gate, tail]:
                    cleanup_coordl_cache(path)
            report(state)
    state['status'] = 'COMPLETE' if all(j['status'] == 'DONE' for j in state['jobs']) else 'FINISHED_WITH_FAILURES'
    state['finished'] = datetime.datetime.now().astimezone().isoformat()
    report(state)


if __name__ == '__main__':
    main()
