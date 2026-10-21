"""Finish five missing RGB epochs using the original, hash-verified entry point."""
import datetime
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
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot


def main():
    state = read(OUT / 'status.json')
    if state['status'] != 'BLOCKED_CODE_CHANGE':
        raise RuntimeError('Expected the reviewed code-change boundary')
    reviewed = OUT / 'epochs/H100/mobilenet24/pytorch_revision_20261006'
    review = read(reviewed / 'code_change_review.json')
    assert review['accepted'] and file_sha(reviewed / 'result.json') == review['raw_result_sha256']
    assert file_sha(reviewed / 'run.command.json') == review['command_record_sha256']
    jobs = [j for j in state['jobs'] if j['status'] != 'DONE']
    assert len(jobs) == 5 and all(j['gpu'] == 'H100' for j in jobs)
    files = dict(read(OUT / 'runtime_snapshot_20261006.json')['files'])
    # Neither mutable CLI is executed: the exact old training entry is loaded
    # from its verified archive; the RGB inference CLI is not a training dependency.
    for path in ('fastlens/benchmarks/paper/shared/training_window_rgb.py', 'fastlens/benchmarks/dct_models/evaluate_rgb.py'):
        del files[str(ROOT / path)]
    files[str(ROOT / 'fastlens/benchmarks/development/historical/run_full_rgb_epochs.py')] = file_sha(ROOT / 'fastlens/benchmarks/development/historical/run_full_rgb_epochs.py')
    for path, expected in files.items():
        if file_sha(path) != expected:
            raise RuntimeError('Other original dependency changed: ' + path)
    for path in (Path(__file__).resolve(), ROOT / 'fastlens/benchmarks/development/historical/frozen_rgb_epoch_20261006.py',
                 ROOT / 'fastlens/benchmarks/development/historical/full_ffcv_interleaved_cache.py',
                 OUT / 'source_before_stream_priority/training_window_rgb.py'):
        files[str(path)] = file_sha(path)
    snapshot = OUT / 'runtime_snapshot_frozen_20261006.json'
    save(snapshot, dict(files=files, fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()))
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(snapshot)
    references = {model: accepted(Path(next(j['output'] for j in state['jobs'] if
        j['gpu'] == '4090' and j['model'] == model and j['pipeline'] == 'pytorch'))) for model in ('swinv2', 'mobilenet24')}
    device, = [x['selection'] for x in read(ROOT / 'results/inventory/device_mapping.json') if x['requested'] == 'H100']
    state.update(status='RUNNING', pid=os.getpid(), runtime_revision='Original entry archived by start hash; five missing H100 epochs only',
                 resumed_at=datetime.datetime.now().astimezone().isoformat())
    report(state)
    for job in jobs:
        previous = Path(job['output'])
        destination = previous.parent / (job['pipeline'] + '_frozen_20261006')
        gate = OUT / 'checks/H100' / job['model'] / (job['pipeline'] + '_frozen_20261006')
        destination.mkdir()
        gate.mkdir()
        job.setdefault('previous_attempts', []).append(dict(output=str(previous), status=job['status'], error=job.get('error', '')))
        job.update(output=str(destination), status='RUNNING')
        job.pop('error', None)
        report(state)
        interpreter = str(Path.home() / 'tmp/coordl-build-cu128/coordl-python') if job['pipeline'] == 'coordl' else PYTHON
        base = ['numactl', '--cpunodebind=0', '--membind=0', interpreter, 'fastlens/benchmarks/development/historical/frozen_rgb_epoch_20261006.py']
        options = ['--gpu', 'H100', '--model', job['model'], '--pipeline', job['pipeline'], '--trial', '0']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, DCTNET_PROFILE=job['model'],
                   TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                   PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        try:
            verify_snapshot()
            tail = OUT / 'tail_checks/revision_20261006' / job['pipeline']
            t, tm = read(tail / 'result.json'), read(tail / 'run.command.json')
            assert t['passed'] and t['samples'] == 143 and t['batch_sizes'] == [64, 64, 15]
            assert tm['returncode'] == 0 and tm['code_unchanged_during_run']
            run([*base, *options, '--check', '--output', str(gate)], gate, env, None)
            check = read(gate / 'result.json')
            assert check['correctness_passed']
            reference = references[job['model']]
            for key in ('initial_model_hash', 'precision', 'batch', 'accumulation', 'augmentation_configuration',
                        'seed', 'compile', 'tf32', 'effective_batch_size'):
                assert check[key] == reference[key], key
            run([*base, *options, '--full-epoch', '--ffcv-root', str(DATA), '--correctness-report', str(gate / 'result.json'),
                 '--output', str(destination)], destination, env, None)
            result = accepted(destination)
            assert result['sample_order_sha256'] == reference['sample_order_sha256']
            notes = 'Original entry SHA preserved in executed_entry.json; complete epoch and identical RGB contract/order.'
            if job['pipeline'] == 'ffcv':
                notes += ' Excluded cache helper permits allocations on both NUMA nodes; training remains CPU/memory node 0. See cache_placement.json; not isolated GPU scaling.'
            save(destination / 'accepted.json', dict(correctness=True, comparison_type='deployment',
                 raw_result=str(destination / 'result.json'), runtime_snapshot=str(snapshot), notes=notes))
            job.update(status='DONE', acceptance_notes=' ' + notes)
        except (RuntimeError, ValueError, AssertionError, subprocess.CalledProcessError, FileNotFoundError) as error:
            job.update(status='FAILED', error=str(error))
            save(destination / 'failure.json', dict(error=str(error)))
            if 'Evaluation code changed' in str(error) or isinstance(error, subprocess.CalledProcessError) and error.returncode == 86:
                state['status'] = 'BLOCKED_CODE_CHANGE'
                report(state)
                raise
        finally:
            if job['pipeline'] == 'coordl':
                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
                cleanup_coordl_cache(destination)
                cleanup_coordl_cache(gate)
            report(state)
    state['status'] = 'COMPLETE' if all(j['status'] == 'DONE' for j in state['jobs']) else 'FINISHED_WITH_FAILURES'
    state['finished'] = datetime.datetime.now().astimezone().isoformat()
    report(state)


if __name__ == '__main__':
    main()
