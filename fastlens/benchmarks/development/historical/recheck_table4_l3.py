"""Recheck the three Table 4 deployments where FastLens trails L3."""
import argparse
import csv
import datetime
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot

CONFIGS = [('4090', 'vitti'), ('H100', 'vitti'), ('H100', 'swinv2')]
GPUS = {'4090': 'GPU-40c637bd-acf5-ea1a-0df8-617138228467',
        'H100': 'GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76'}
HISTORY = {
    ('4090', 'vitti'): 'results/training_expansion_20260930/windows/4090/vitti/imagelanes/trial_0/result.json',
    ('H100', 'vitti'): 'results/h100_reserved_windows_20261002/vitti/trial_0/result.json',
    ('H100', 'swinv2'): 'results/p0/raw/H100/swinv2/imagelanes/trial_0_refresh_20260929/result.json',
}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def window_command(gpu, model, pipeline):
    command = ['fastlens/benchmarks/paper/shared/training_window_rgb.py', '--gpu', gpu, '--model', model, '--pipeline', pipeline]
    if pipeline == 'l3':
        command += ['--l3-root', str(Path.home() / 'tmp/l3-baseline/data-patch32'),
                    '--l3-library', str(Path.home() / 'tmp/l3-baseline/build/libl3-patch32.so')]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/table4_l3_recheck_20261007')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    native = next((ROOT / 'build/fastlens/torch').glob('_fastlens_direct_dct.*.so'))
    save(out / 'handoff.json', dict(native_path=str(native)))
    snapshot(out)
    state = dict(status='RUNNING', pid=os.getpid(), started=datetime.datetime.now().astimezone().isoformat(),
                 active=None, completed_runs=0, planned_runs=36, configurations=CONFIGS,
                 protocol='Three alternating repeats for full windows and resident-model diagnostics; '
                          'retain every valid trial; no performance-based retries.')
    save(out / 'status.json', state)
    env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp'), PYTHONPATH=f'{ROOT}:{native.parent}',
               FASTLENS_TRAINING_NATIVE_DIR=str(native.parent),
               FASTLENS_EVALUATION_SNAPSHOT=str(out / 'runtime_snapshot.json'),
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', FASTLENS_NSYS_FINE='0')
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(out / 'runtime_snapshot.json')
    base = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable]
    records = []

    def launch(command, destination):
        state['active'] = str(destination.relative_to(out))
        save(out / 'status.json', state)
        destination.mkdir(parents=True)
        run([*base, *command], destination, env, None)
        meta = read(destination / 'run.command.json')
        assert meta['returncode'] == 0 and meta['code_unchanged_during_run']
        return read(destination / 'result.json')

    for gpu, model in CONFIGS:
        env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model)
        target = out / gpu / model
        matched = launch(['fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py', '--model', model, '--epoch', '0',
                          '--output', str(target / 'matched/result.json')], target / 'matched')
        assert matched['passed']
        for pipeline in ('imagelanes', 'l3'):
            dest = target / pipeline / 'check'
            check = launch([*window_command(gpu, model, pipeline), '--trial', '0', '--check',
                            '--output', str(dest)], dest)
            assert check['correctness_passed']
        for phase in ('window', 'model_only'):
            for trial in range(3):
                for pipeline in (('imagelanes', 'l3') if trial % 2 == 0 else ('l3', 'imagelanes')):
                    dest = target / pipeline / phase / f'trial_{trial}'
                    extra = (['--model-only'] if phase == 'model_only' else
                             ['--correctness-report', str(target / pipeline / 'check/result.json')])
                    result = launch([*window_command(gpu, model, pipeline), '--trial', str(trial),
                                     *extra, '--output', str(dest)], dest)
                    assert result['images'] == 65536 and result['updates'] == 64 and result['warmup_updates'] == 32
                    assert result['correctness_passed'] and result['loss_finite'] and result['parameter_update_verified']
                    assert not result['profile'] and not result['memory_diagnostic']
                    old = (read(ROOT / HISTORY[(gpu, model)]) if pipeline == 'imagelanes' else
                           read(ROOT / f'results/l3_patch32_20261006/training/{gpu}/{model}/l3/trial_0/result.json'))
                    for key in ('initial_model_hash', 'precision', 'batch', 'accumulation', 'warmup_updates'):
                        assert result[key] == old[key], (key, result[key], old[key])
                    if phase == 'window':
                        assert result['sample_order_sha256'] == old['sample_order_sha256']
                        assert result['cache_before']['resident_fraction'] == 1
                        ids = read(result['sample_ids_file'])
                        assert len(ids) == len(set(ids)) == 65536
                    records.append(dict(gpu=gpu, model=model, pipeline=pipeline, phase=phase, trial=trial,
                                        images=result['images'], seconds=result['seconds'],
                                        images_per_s=result['images'] / result['seconds'],
                                        raw_result=str((dest / 'result.json').relative_to(ROOT))))
                    write_csv(out / 'trials.csv', records)
                    state['completed_runs'] += 1
                    save(out / 'status.json', state)
                    print('COMPLETED', records[-1], flush=True)
    summary = []
    for gpu, model in CONFIGS:
        for phase in ('window', 'model_only'):
            for pipeline in ('imagelanes', 'l3'):
                group = [r for r in records if (r['gpu'], r['model'], r['phase'], r['pipeline']) ==
                         (gpu, model, phase, pipeline)]
                values = [r['images_per_s'] for r in group]
                summary.append(dict(gpu=gpu, model=model, phase=phase, pipeline=pipeline, trials=len(values),
                                    median_images_per_s=statistics.median(values), min_images_per_s=min(values),
                                    max_images_per_s=max(values), raw_results=';'.join(r['raw_result'] for r in group)))
    write_csv(out / 'summary.csv', summary)
    state.update(status='COMPLETE', active=None, finished=datetime.datetime.now().astimezone().isoformat())
    save(out / 'status.json', state)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
