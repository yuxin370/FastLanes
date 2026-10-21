"""Extend the current matched training/inference cohort to ResNet-DCT24/64."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.paper.shared.remaining_baseline_inference import reference_predictions
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.paper.shared.references import inference_reference
from fastlens.benchmarks.development.historical.run_remaining_baselines import summarize
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save

TRAIN_NATIVE = ROOT/'results/resident_study_20261011_fixed/native_runtime'
INFER_NATIVE = ROOT/'results/swin_inference_20261011_fixed/native_runtime'
PATHS = dict(training=('cpu_dct', 'imagelanes_basic', 'dct_resident'),
             inference=('cpu_dct', 'imagelanes'))
MODELS = ('resnet24', 'resnet64')
GPU_ORDER = ('H100', '4090')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/resnet_matched_20261013')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=True)
    if not (out/'plan.json').exists():
        for gpu in GPU_ORDER:
            for model in MODELS:
                assert reference_predictions(inference_reference(gpu, model, 'cpu_dct')) == reference_predictions(
                    inference_reference(gpu, model, 'imagelanes'))
        files = dict(read(ROOT/'results/remaining_baselines_20261012/runtime_snapshot.json')['files'])
        for name in ('fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py', 'fastlens/benchmarks/paper/shared/remaining_baseline_training.py', 'fastlens/benchmarks/paper/shared/remaining_baseline_inference.py', 'fastlens/benchmarks/development/historical/run_resnet_matched.py'):
            path = ROOT / name
            files[str(path.relative_to(ROOT))] = file_sha(path)
        save(out/'runtime_snapshot.json', dict(files=files,
            fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()))
        save(out/'plan.json', dict(models=MODELS, gpu_order=GPU_ORDER, paths=PATHS, trials=3,
            training_runs=36, inference_runs=24, training_images=65536, inference_images=50000,
            training_protocol='restored-state-real-resident-8-v2', batch=64, accumulation=16,
            warmup_updates=32, resident_batches=8, resident_unique_samples=512,
            CUBLAS_WORKSPACE_CONFIG=None, randaugment=False, mixup=False,
            training_precision='bf16', inference_precision='fp32', host_threads=8,
            order='Per GPU/model/phase rotate paths by 2*trial; retain all three trials.',
            scope='Matched DCT comparison; original ResNet geometry; windows, not full epochs.',
            training_native=str(TRAIN_NATIVE), inference_native=str(INFER_NATIVE)))
        save(out/'status.json', dict(status='QUEUED', pid=None, completed_runs=0, planned_runs=60, active=None))
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(out/'runtime_snapshot.json')
    verify_snapshot()
    if args.prepare:
        print('Prepared 36 training windows and 24 full-validation measurements.')
        return
    state = read(out/'status.json')
    if state['status'] == 'COMPLETE':
        return
    state.update(status='RUNNING', pid=os.getpid())
    state.pop('error', None)
    records = list(csv.DictReader((out/'trials.csv').open())) if (out/'trials.csv').exists() else []
    env = dict(os.environ, TMPDIR=str(Path.home()/'tmp'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', FASTLENS_NSYS_FINE='0')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)

    def launch(command, dest):
        while True:
            state['active'] = str(dest.relative_to(out))
            save(out/'status.json', state)
            if dest.exists():
                dest.rename(dest.with_name(dest.name+'_failed_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            dest.mkdir(parents=True)
            try:
                run([*base, *command], dest, env, None)
            except RuntimeError as error:
                if not str(error).startswith('Target GPU contention:'):
                    raise
                print('GPU contention: wait and repeat this interrupted attempt.', flush=True)
                continue
            metadata = read(dest/'run.command.json')
            assert metadata['returncode'] == 0 and metadata['code_unchanged_during_run']
            return read(dest/'result.json')

    try:
        for gpu in GPU_ORDER:
            node = 0 if gpu == 'H100' else 1
            base = ['numactl', f'--cpunodebind={node}', f'--membind={node}', sys.executable, '-B']
            while GPUS[gpu] not in subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines():
                state.update(status='WAITING_FOR_GPU', active=gpu)
                save(out/'status.json', state)
                print(f'Waiting for {gpu} ({GPUS[gpu]}) to become visible.', flush=True)
                time.sleep(30)
            state.update(status='RUNNING', active=None)
            save(out/'status.json', state)
            for model in MODELS:
                env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model,
                    PYTHONPATH=f'{ROOT}:{TRAIN_NATIVE}', FASTLENS_TRAINING_NATIVE_DIR=str(TRAIN_NATIVE))
                matched = out/'checks'/gpu/model/'matched_dct_basic'
                if not (matched/'result.json').exists():
                    launch(['fastlens/benchmarks/development/diagnostics/check_mobile_training.py', '--augmentation', 'basic',
                            '--output', str(matched/'result.json')], matched)
                assert read(matched/'result.json')['passed']
                for phase, paths in PATHS.items():
                    native = TRAIN_NATIVE if phase == 'training' else INFER_NATIVE
                    env.update(PYTHONPATH=f'{ROOT}:{native}', FASTLENS_TRAINING_NATIVE_DIR=str(native))
                    commands, checks = {}, {}
                    for path in paths:
                        driver = ('fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py' if path in ('imagelanes_basic', 'dct_resident')
                                  else f'remaining_baseline_{phase}')
                        command = [driver, '--model', model, '--path', path]
                        if phase == 'training':
                            command += ['--gpu', gpu]
                        else:
                            command += ['--reference', str(inference_reference(gpu, model, 'cpu_dct'))]
                        commands[path] = command
                        dest = out/phase/gpu/model/path/'check'
                        gate = read(dest/'result.json') if (dest/'result.json').exists() else launch(
                            [*command, '--check', '--output', str(dest)], dest)
                        assert gate['correctness_passed'] and gate['CUBLAS_WORKSPACE_CONFIG'] is None
                        checks[path] = gate
                    if phase == 'training':
                        reference = checks['imagelanes_basic']
                        for gate in checks.values():
                            assert gate['restored_state'] == reference['restored_state']
                            assert gate['prefix_512_sha256'] == reference['prefix_512_sha256']
                            for key in ('shape', 'dtype'):
                                assert [t[key] for t in gate['tensor_contract']] == [t[key] for t in reference['tensor_contract']]
                    for trial in range(3):
                        rotate = 2*trial % len(paths)
                        for path in paths[rotate:]+paths[:rotate]:
                            if any((r['phase'], r['gpu'], r['model'], r['path'], int(r['trial'])) ==
                                   (phase, gpu, model, path, trial) for r in records):
                                continue
                            dest = out/phase/gpu/model/path/f'trial_{trial}'
                            command = [*commands[path], '--output', str(dest)]
                            if phase == 'training':
                                command += ['--trial', str(trial), '--correctness-report', str(dest.parent/'check/result.json')]
                            result = launch(command, dest)
                            assert result['images'] == (65536 if phase == 'training' else 50000)
                            assert result['correctness_passed'] and result['host_threads'] == 8
                            assert result['CUBLAS_WORKSPACE_CONFIG'] is None
                            if phase == 'training':
                                assert result['restored_state'] == checks[path]['restored_state']
                                assert result['prefix_512_sha256'] == checks[path]['prefix_512_sha256']
                                assert result['updates'] == 64 and result['warmup_updates'] == 32
                                previous = [r for r in records if (r['phase'], r['gpu'], r['model']) ==
                                            (phase, gpu, model) and (r['path'] == 'dct_resident') == (path == 'dct_resident')]
                                if previous:
                                    assert result['sample_order_sha256'] == read(ROOT/previous[0]['raw_result'])['sample_order_sha256']
                            elif trial:
                                assert result['predictions_sha256'] == read(dest.parent/'trial_0/result.json')['predictions_sha256']
                            records.append(dict(phase=phase, gpu=gpu, model=model, path=path, trial=trial,
                                images=result['images'], seconds=result['seconds'], images_per_s=result['images_per_s'],
                                raw_result=str((dest/'result.json').relative_to(ROOT))))
                            summarize(out, records)
                            state['completed_runs'] = len(records)
                            save(out/'status.json', state)
                            print('COMPLETED', records[-1], flush=True)
        state.update(status='COMPLETE', active=None)
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(out/'status.json', state)


if __name__ == '__main__':
    main()
