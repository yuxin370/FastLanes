"""Interleave 48 Swin inference passes after the corrected training study."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.development.reports.report_resident_study import write_csv
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.paper.shared.swin_unified_inference import PATHS


from fastlens.benchmarks.paper.shared.references import expected_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/swin_inference_20261011_fixed')
    parser.add_argument('--after-training', type=Path, required=True)
    parser.add_argument('--native-dir', type=Path, default=ROOT/'build/fastlens/torch')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    native_dir = out/'native_runtime'
    native_dir.mkdir()
    native = next(args.native_dir.glob('_fastlens_direct_dct.*.so'))
    shutil.copy2(native, native_dir/native.name)
    files = [Path(__file__), ROOT/'fastlens/benchmarks/paper/shared/swin_unified_inference.py',
             ROOT/'fastlens/benchmarks/paper/shared/remaining_baseline_inference.py', ROOT/'fastlens/benchmarks/baselines/system_rgbnomore/shared/paper_adapters.py',
             ROOT/'fastlens/benchmarks/execution/inference_load.py', ROOT/'fastlens/benchmarks/execution/run_logged.py',
             ROOT/'fastlens/benchmarks/execution/resource_guard.py', native_dir/native.name]
    for directory in ('fastlens/benchmarks/system_dct_major', 'fastlens/benchmarks/baselines/system_rgbnomore/inference',
                      'fastlens/benchmarks/baselines/l3', 'fastlens/torch'):
        files.extend((ROOT/directory).rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): file_sha(p) for p in files}
    save(out/'runtime_snapshot.json', dict(files=hashes,
        fingerprint=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()))
    save(out/'plan.json', dict(paths=PATHS, trials=3, gpus=list(GPUS), formal_runs=48,
        images=50000, batch=50, precision='fp32', host_threads=8, workers=8,
        CUBLAS_WORKSPACE_CONFIG=None, model_stream_priority='greatest',
        timing='Identical wall-clock loop, including reader initialization and metrics.',
        order='Rotate paths by 2*trial; retain all trials. No historical maxima.'))
    state = dict(status='WAITING_FOR_TRAINING', pid=os.getpid(), completed_runs=0, planned_runs=48, active=None)
    save(out/'status.json', state)
    env = dict(os.environ, TMPDIR=str(Path.home()/'tmp'), PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=f'{ROOT}:{native_dir}', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        FASTLENS_EVALUATION_SNAPSHOT=str(out/'runtime_snapshot.json'))
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    records = []
    try:
        references = {(gpu, path): expected_predictions(gpu, path) for gpu in GPUS for path in PATHS}
        while True:
            training = read(args.after_training/'status.json')
            if training['status'] == 'COMPLETE':
                break
            if training['status'] == 'FAILED':
                raise RuntimeError('Training study stopped; inference has not started.')
            time.sleep(10)
        state['status'] = 'RUNNING'
        for gpu in GPUS:
            env['CUDA_VISIBLE_DEVICES'] = GPUS[gpu]
            for trial in (-1, 0, 1, 2):
                rotate = 0 if trial == -1 else 2*trial
                order = PATHS[rotate:]+PATHS[:rotate]
                for path in order:
                    dest = out/gpu/path/('check' if trial == -1 else f'trial_{trial}')
                    dest.mkdir(parents=True)
                    state['active'] = str(dest.relative_to(out))
                    save(out/'status.json', state)
                    interpreter = str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if path == 'coordl' else sys.executable
                    command = ['numactl', '--cpunodebind=0', '--membind=0', interpreter, '-B',
                               'fastlens/benchmarks/paper/shared/swin_unified_inference.py', '--path', path, '--output', str(dest)]
                    if trial == -1:
                        command.append('--check')
                    while True:
                        try:
                            run(command, dest, env, None)
                            break
                        except RuntimeError as error:
                            if not str(error).startswith('Target GPU contention:'):
                                raise
                            if path == 'coordl':
                                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
                                cleanup_coordl_cache(dest)
                            dest.rename(dest.with_name(dest.name+'_interrupted_'+str(time.time_ns())))
                            dest.mkdir()
                            print('GPU contention: waiting to retry', dest, flush=True)
                    metadata = read(dest/'run.command.json')
                    assert metadata['returncode'] == 0 and metadata['code_unchanged_during_run']
                    result = read(dest/'result.json')
                    assert result['images'] == (100 if trial == -1 else 50000)
                    assert result['host_threads'] == 8 and result['CUBLAS_WORKSPACE_CONFIG'] is None
                    expected = references[gpu, path]
                    assert [p[0] for p in result['predictions']] == [expected[s] for s in result['sample_ids']], 'Predictions differ from accepted baseline; inspect before accepting throughput'
                    if trial == -1:
                        continue
                    gate = read(out/gpu/path/'check/result.json')
                    assert result['sample_ids'][:100] == gate['sample_ids']
                    assert result['predictions'][:100] == gate['predictions']
                    if trial:
                        previous = read(out/gpu/path/'trial_0/result.json')
                        assert result['predictions_sha256'] == previous['predictions_sha256']
                        assert result['sample_ids'] == previous['sample_ids']
                    records.append(dict(gpu=gpu, path=path, trial=trial, images=50000,
                        seconds=result['seconds'], images_per_s=result['images_per_s'], top1=result['top1'],
                        raw_result=str((dest/'result.json').relative_to(ROOT))))
                    write_csv(out/'trials.csv', records)
                    summary = []
                    for g, p in dict.fromkeys((r['gpu'], r['path']) for r in records):
                        same = [r for r in records if (r['gpu'], r['path']) == (g, p)]
                        rates = [r['images_per_s'] for r in same]
                        summary.append(dict(gpu=g, path=p, trials=len(same),
                            median_images_per_s=statistics.median(rates), min_images_per_s=min(rates),
                            max_images_per_s=max(rates), top1=same[0]['top1']))
                    write_csv(out/'summary.csv', summary)
                    state['completed_runs'] = len(records)
                    save(out/'status.json', state)
        state.update(status='COMPLETE', active=None)
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(out/'status.json', state)


if __name__ == '__main__':
    main()
