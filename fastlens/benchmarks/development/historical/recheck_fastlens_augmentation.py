"""Paired FastLens training windows with full or basic augmentation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.development.historical.recheck_table4_l3 import window_command
from fastlens.benchmarks.development.historical.recheck_table4_l3 import write_csv
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot

MODELS = ('vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'efun')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/imagelanes_augmentation_20261010')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    (out / 'native_runtime').mkdir()
    source = next(args.native_dir.glob('_fastlens_direct_dct.*.so'))
    native = out / 'native_runtime' / source.name
    shutil.copy2(source, native)
    save(out / 'handoff.json', dict(native_path=str(native)))
    frozen = snapshot(out)
    # This single-GPU window never imports the separate distributed-training CLI.
    # Concurrent edits to that unused CLI previously stopped the campaign.
    frozen['files'].pop('fastlens/benchmarks/training_pls/distributed.py')
    frozen['fingerprint'] = hashlib.sha256(json.dumps(frozen['files'], sort_keys=True).encode()).hexdigest()
    frozen['excluded_unused_entrypoints'] = ['fastlens/benchmarks/training_pls/distributed.py']
    save(out / 'runtime_snapshot.json', frozen)
    save(out / 'plan.json', dict(models=MODELS, gpus=['4090', 'H100'], trials=3,
        arms=['full', 'basic'], warmup_updates=32, measured_updates=64, measured_images=65536,
        policy='Serial paired full/basic, basic/full, full/basic runs; keep all three trials. '
               'Wait for occupied target GPU. Annotate shared host load. No inference or full epochs.'))
    state = dict(status='RUNNING', pid=os.getpid(), started=datetime.datetime.now().astimezone().isoformat(),
                 active=None, completed_runs=0, planned_runs=60)
    save(out / 'status.json', state)
    env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp'), PYTHONPATH=f'{ROOT}:{native.parent}',
               FASTLENS_TRAINING_NATIVE_DIR=str(native.parent),
               FASTLENS_EVALUATION_SNAPSHOT=str(out / 'runtime_snapshot.json'),
               PYTHONDONTWRITEBYTECODE='1', FASTLENS_NSYS_FINE='0', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    base = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable, '-B']
    records = []

    def launch(command, dest):
        state['active'] = str(dest.relative_to(out))
        save(out / 'status.json', state)
        dest.mkdir(parents=True)
        run([*base, *command], dest, env, None)
        metadata = read(dest / 'run.command.json')
        assert metadata['returncode'] == 0 and metadata['code_unchanged_during_run']

    def summarize():
        rows = []
        for gpu in ('4090', 'H100'):
            for model in MODELS:
                groups = {arm: [r for r in records if (r['gpu'], r['model'], r['augmentation']) ==
                               (gpu, model, arm)] for arm in ('full', 'basic')}
                if not all(len(v) == 3 for v in groups.values()):
                    continue
                full, basic = (statistics.median(r['images_per_s'] for r in groups[a]) for a in ('full', 'basic'))
                rows.append(dict(gpu=gpu, model=model, full_median_images_per_s=full,
                    basic_median_images_per_s=basic, basic_speedup=basic/full,
                    full_min_images_per_s=min(r['images_per_s'] for r in groups['full']),
                    full_max_images_per_s=max(r['images_per_s'] for r in groups['full']),
                    basic_min_images_per_s=min(r['images_per_s'] for r in groups['basic']),
                    basic_max_images_per_s=max(r['images_per_s'] for r in groups['basic']),
                    full_results=';'.join(r['raw_result'] for r in groups['full']),
                    basic_results=';'.join(r['raw_result'] for r in groups['basic'])))
        if rows:
            write_csv(out / 'summary.csv', rows)

    try:
        for gpu in ('4090', 'H100'):
            env['CUDA_VISIBLE_DEVICES'] = GPUS[gpu]
            for model in MODELS:
                env['DCTNET_PROFILE'] = model
                target = out / gpu / model
                for arm in ('full', 'basic'):
                    matched = target / arm / 'matched'
                    command = (['fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py', '--model', model]
                               if model in ('vitti', 'swinv2') else ['fastlens/benchmarks/development/diagnostics/check_mobile_training.py'])
                    launch([*command, '--augmentation', arm, '--output', str(matched / 'result.json')], matched)
                    assert read(matched / 'result.json')['passed']
                    check = target / arm / 'check'
                    launch([*window_command(gpu, model, 'imagelanes'), '--fastlens-augmentation', arm,
                            '--trial', '0', '--check', '--output', str(check)], check)
                    assert read(check / 'result.json')['correctness_passed']
                contract = None
                for trial in range(3):
                    for arm in (('full', 'basic') if trial % 2 == 0 else ('basic', 'full')):
                        dest = target / arm / f'trial_{trial}'
                        launch([*window_command(gpu, model, 'imagelanes'), '--fastlens-augmentation', arm,
                                '--trial', str(trial), '--correctness-report', str(target / arm / 'check/result.json'),
                                '--output', str(dest)], dest)
                        result = read(dest / 'result.json')
                        assert result['images'] == 65536 and result['updates'] == 64
                        assert result['correctness_passed'] and result['loss_finite'] and result['parameter_update_verified']
                        assert result['cache_before']['resident_fraction'] == 1
                        current = {k: result[k] for k in ('initial_model_hash', 'sample_order_sha256',
                                   'precision', 'batch', 'accumulation', 'warmup_updates')}
                        if contract is None:
                            contract = current
                        assert current == contract, 'Paired training configuration or sample order changed'
                        records.append(dict(gpu=gpu, model=model, augmentation=arm, trial=trial,
                            images=result['images'], seconds=result['seconds'],
                            images_per_s=result['images']/result['seconds'],
                            raw_result=str((dest / 'result.json').relative_to(ROOT))))
                        write_csv(out / 'trials.csv', records)
                        summarize()
                        state['completed_runs'] += 1
                        save(out / 'status.json', state)
                        print('COMPLETED', records[-1], flush=True)
        state.update(status='COMPLETE', active=None, finished=datetime.datetime.now().astimezone().isoformat())
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(out / 'status.json', state)


if __name__ == '__main__':
    main()
