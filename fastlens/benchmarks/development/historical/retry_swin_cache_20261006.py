"""One replacement epoch for the observed H100 Swin/PyTorch cache failure."""
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
    wait_for_process(4191943, 'fastlens/benchmarks/development/historical/resume_full_rgb_20261006.py')
    state = read(OUT / 'status.json')
    job, = [j for j in state['jobs'] if (j['gpu'], j['model'], j['pipeline']) == ('H100', 'swinv2', 'pytorch')]
    if job['status'] == 'DONE':
        accepted(Path(job['output']))
        print('Equivalent accepted epoch already exists; no replacement run', flush=True)
        return
    if state['status'] != 'FINISHED_WITH_FAILURES' or job['status'] != 'FAILED':
        raise RuntimeError('Previous queue is not at the expected completed boundary')
    previous = Path(job['output'])
    if 'Source page cache is not fully warm' not in (previous / 'run.log').read_text():
        raise RuntimeError('Failure is not the observed premeasurement cache failure')
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(OUT / 'runtime_snapshot_20261006.json')
    verify_snapshot()
    files = dict(read(OUT / 'runtime_snapshot_20261006.json')['files'])
    files[str(Path(__file__).resolve())] = file_sha(__file__)
    snapshot = OUT / 'runtime_snapshot_cache_retry_20261006.json'
    save(snapshot, dict(files=files, fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()))
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(snapshot)
    gate = OUT / 'checks/H100/swinv2/pytorch_revision_20261006'
    check, command = read(gate / 'result.json'), read(gate / 'run.command.json')
    assert check['correctness_passed'] and command['returncode'] == 0 and command['code_unchanged_during_run']
    reference = accepted(Path(next(j['output'] for j in state['jobs'] if j['status'] == 'DONE' and j['model'] == 'swinv2')))
    for key in ('initial_model_hash', 'precision', 'batch', 'accumulation', 'seed',
                'augmentation_configuration', 'compile', 'tf32', 'effective_batch_size'):
        assert check[key] == reference[key], key
    destination = previous.parent / 'pytorch_cache_retry_20261006'
    destination.mkdir()
    job.setdefault('previous_attempts', []).append(dict(output=str(previous), status=job['status'], error=job['error']))
    job.update(output=str(destination), status='RUNNING')
    job.pop('error')
    state.update(status='RUNNING', pid=os.getpid())
    report(state)
    device, = [d['selection'] for d in read(ROOT / 'results/inventory/device_mapping.json') if d['requested'] == 'H100']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, DCTNET_PROFILE='swinv2',
               TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        run(['numactl', '--cpunodebind=0', '--membind=0', PYTHON, 'fastlens/benchmarks/paper/shared/training_window_rgb.py',
             '--gpu', 'H100', '--model', 'swinv2', '--pipeline', 'pytorch', '--trial', '0', '--full-epoch',
             '--ffcv-root', str(DATA), '--correctness-report', str(gate / 'result.json'), '--output', str(destination)],
            destination, env, None)
        result = accepted(destination)
        if result['sample_order_sha256'] != reference['sample_order_sha256']:
            raise ValueError('Full sample order differs from accepted RGB epochs')
        save(destination / 'accepted.json', dict(correctness=True, comparison_type='deployment',
             raw_result=str(destination / 'result.json'), runtime_snapshot=str(snapshot),
             notes='One replacement for premeasurement cache-residency failure; same protocol and gates.'))
        job['status'] = 'DONE'
    except (RuntimeError, ValueError, subprocess.CalledProcessError, FileNotFoundError) as error:
        job.update(status='FAILED', error=str(error))
        save(destination / 'failure.json', dict(error=str(error)))
    state['status'] = 'COMPLETE' if all(j['status'] == 'DONE' for j in state['jobs']) else 'FINISHED_WITH_FAILURES'
    report(state)


if __name__ == '__main__':
    main()
