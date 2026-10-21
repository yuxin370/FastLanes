"""132 interleaved resident-input and end-to-end training measurements."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot
from fastlens.benchmarks.development.reports.report_resident_study import summarize
from fastlens.benchmarks.development.reports.report_resident_study import write_csv


def groups():
    result = []
    for gpu in ('4090', 'H100'):
        for model in ('vitti', 'swinv2'):
            result.append(dict(stage=1 if gpu == '4090' else 2, gpu=gpu, name=model,
                paths=[(model, p) for p in ('ffcv_raw', 'imagelanes_basic', 'rgb_resident', 'ffcv_jpeg', 'dct_resident')]))
    for gpu in ('4090', 'H100'):
        result.append(dict(stage=3, gpu=gpu, name='mobilenet', paths=[
            ('mobilenet24', 'ffcv_raw'), ('mobilenet24', 'imagelanes_basic'),
            ('mobilenet24', 'rgb_resident'), ('mobilenet32', 'dct_resident'),
            ('mobilenet24', 'ffcv_jpeg'), ('mobilenet24', 'dct_resident'), ('mobilenet32', 'imagelanes_basic')]))
        result.append(dict(stage=3, gpu=gpu, name='efun', paths=[('efun', p) for p in
            ('ffcv_raw', 'imagelanes_basic', 'rgb_resident', 'ffcv_jpeg', 'dct_resident')]))
    assert sum(len(g['paths'])*3 for g in result) == 132
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/resident_study_20261011_fixed')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if not args.resume:
        out.mkdir()
        (out / 'native_runtime').mkdir()
        source = next((ROOT / 'results/imagelanes_augmentation_20261010_restart_1440/native_runtime').glob('_fastlens_direct_dct.*.so'))
        native = out / 'native_runtime' / source.name
        shutil.copy2(source, native)
        save(out / 'handoff.json', dict(native_path=str(native), native_origin=str(source)))
        frozen = snapshot(out)
        frozen['files'].pop('fastlens/benchmarks/training_pls/distributed.py')
        for name in ('fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py', 'fastlens/benchmarks/development/historical/run_resident_study.py', 'fastlens/benchmarks/development/reports/report_resident_study.py', 'fastlens/benchmarks/execution/inference_load.py', 'fastlens/benchmarks/execution/revision_results.py', 'fastlens/benchmarks/development/diagnostics/profile_resident_study.py', 'fastlens/benchmarks/development/diagnostics/analyze_resident_trace.py'):
            path = ROOT / name
            frozen['files'][str(path.relative_to(ROOT))] = file_sha(path)
        frozen['fingerprint'] = hashlib.sha256(json.dumps(frozen['files'], sort_keys=True).encode()).hexdigest()
        frozen['excluded_unused_entrypoints'] = ['fastlens/benchmarks/training_pls/distributed.py']
        save(out / 'runtime_snapshot.json', frozen)
        save(out / 'plan.json', dict(groups=groups(), trials=3, measured_runs=132,
            CUBLAS_WORKSPACE_CONFIG=None, model_stream_priority='greatest for both RGB and DCT',
            order='Within each trial rotate the declared path order left by 2*trial positions.',
            window=dict(warmup_updates=32, measured_images=65536, batch=64, accumulation=16,
                        restore='model, optimizer, decay, scheduler, RNG; restart input sequence at position 0'),
            resident=dict(real_batches=8, unique_samples=512, cycle='batch index modulo 8', RGB_source='FFCV raw'),
            profiling=dict(after_formal_runs=60, selection='largest and smallest absolute median paired g_DCT among four Transformer configurations',
                           ffcv='JPEG if raw/JPEG medians differ by <=2%; otherwise the faster median',
                           images=16384, paths=5, controls='one same-length uninstrumented run per profiled path'),
            optional_210_run_extension=False, selection_policy='All three trials retained; no historical maxima or performance-based reruns.'))
        state = dict(status='RUNNING', pid=os.getpid(), started=datetime.datetime.now().astimezone().isoformat(),
                     stage=1, completed_runs=0, planned_runs=132, active=None)
        records = []
    else:
        state = read(out / 'status.json')
        assert state['status'] == 'FAILED'
        state.update(status='RUNNING', pid=os.getpid())
        state.pop('error', None)
        records = list(csv.DictReader((out / 'trials.csv').open())) if (out / 'trials.csv').exists() else []
    save(out / 'status.json', state)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(out / 'runtime_snapshot.json')
    verify_snapshot()
    native_dir = out / 'native_runtime'
    env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp'), PYTHONPATH=f'{ROOT}:{native_dir}',
        FASTLENS_TRAINING_NATIVE_DIR=str(native_dir), PYTHONDONTWRITEBYTECODE='1', FASTLENS_NSYS_FINE='0',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    base = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable, '-B']
    completed = {(r['gpu'], r['model'], r['path'], int(r['trial'])) for r in records}

    def launch(command, dest):
        state['active'] = str(dest.relative_to(out))
        save(out / 'status.json', state)
        if dest.exists():
            dest.rename(dest.with_name(dest.name + '_failed_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        dest.mkdir(parents=True)
        run([*base, *command], dest, env, None)
        meta = read(dest / 'run.command.json')
        assert meta['returncode'] == 0 and meta['code_unchanged_during_run']

    try:
        for group in groups():
            state['stage'] = group['stage']
            gpu = group['gpu']
            env['CUDA_VISIBLE_DEVICES'] = GPUS[gpu]
            paths = group['paths']
            for model in dict.fromkeys(m for m, path in paths):
                env['DCTNET_PROFILE'] = model
                dest = out / 'checks' / gpu / model / 'matched_dct_basic'
                if not (dest / 'result.json').exists():
                    command = (['fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py', '--model', model]
                               if model in ('vitti', 'swinv2') else ['fastlens/benchmarks/development/diagnostics/check_mobile_training.py'])
                    launch([*command, '--augmentation', 'basic', '--output', str(dest / 'result.json')], dest)
                assert read(dest / 'result.json')['passed']
            check_results = {}
            for model, path in paths:
                env['DCTNET_PROFILE'] = model
                dest = out / 'checks' / gpu / model / path
                command = ['fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py', '--gpu', gpu, '--model', model, '--path', path]
                if not (dest / 'result.json').exists():
                    launch([*command, '--check', '--output', str(dest)], dest)
                d = read(dest / 'result.json')
                assert d['correctness_passed'] and d['restoration_verified']
                check_results[model, path] = d
            for model in dict.fromkeys(m for m, path in paths):
                members = [v for (m, p), v in check_results.items() if m == model]
                for domain in ('rgb', 'dct'):
                    same = [v for v in members if v['domain'] == domain]
                    assert all(v['restored_state'] == same[0]['restored_state'] for v in same)
                    assert all(v['prefix_512_sha256'] == same[0]['prefix_512_sha256'] for v in same)
                    assert all(v['tensor_contract'] == same[0]['tensor_contract'] for v in same)
            for trial in range(3):
                rotate = (2*trial) % len(paths)
                for model, path in paths[rotate:] + paths[:rotate]:
                    if (gpu, model, path, trial) in completed:
                        continue
                    env['DCTNET_PROFILE'] = model
                    dest = out / 'formal' / gpu / model / path / f'trial_{trial}'
                    gate = out / 'checks' / gpu / model / path / 'result.json'
                    launch(['fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py', '--gpu', gpu, '--model', model,
                            '--path', path, '--trial', str(trial), '--correctness-report', str(gate),
                            '--output', str(dest)], dest)
                    d = read(dest / 'result.json')
                    assert d['images'] == 65536 and d['updates'] == 64 and not d['profile']
                    assert d['correctness_passed'] and d['restoration_verified']
                    assert d['prefix_512_sha256'] == check_results[model, path]['prefix_512_sha256']
                    previous = [r for r in records if (r['gpu'], r['model'], r['path']) == (gpu, model, path)]
                    if previous:
                        assert d['sample_order_sha256'] == read(ROOT / previous[0]['raw_result'])['sample_order_sha256']
                    records.append(dict(stage=group['stage'], gpu=gpu, model=model, path=path, trial=trial,
                        images=d['images'], unique_samples=d['unique_samples'], seconds=d['seconds'],
                        images_per_s=d['images_per_s'], raw_result=str((dest / 'result.json').relative_to(ROOT))))
                    write_csv(out / 'trials.csv', records)
                    summarize(out)
                    state['completed_runs'] = len(records)
                    save(out / 'status.json', state)
                    print('COMPLETED', records[-1], flush=True)
            if (group['stage'] == 2 and group['name'] == 'swinv2'
                    and not ((out / 'profiles/status.json').exists()
                             and read(out / 'profiles/status.json')['status'] == 'COMPLETE')):
                state.update(active='representative profiling', stage='profiling')
                save(out / 'status.json', state)
                subprocess.run([sys.executable, '-B', '-u', 'fastlens/benchmarks/development/diagnostics/profile_resident_study.py',
                                '--output', str(out)], env=env, check=True)
        state.update(status='COMPLETE', active=None, finished=datetime.datetime.now().astimezone().isoformat())
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(out / 'status.json', state)


if __name__ == '__main__':
    main()
