"""Refresh remaining Table 4 windows and all FastLens full-validation inference."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.development.historical.recheck_table4_l3 import window_command
from fastlens.benchmarks.development.historical.recheck_table4_l3 import write_csv
from fastlens.benchmarks.paper.shared.references import PROFILES
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.development.historical.run_phase2_repeats import predictions
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot
from fastlens.benchmarks.system_dct_major.common import file_identity, parse_manifest, selected_batches, sample_trace, sha256_json

REMAINING = [('4090', 'swinv2'), ('4090', 'mobilenet24'), ('4090', 'mobilenet32'),
             ('4090', 'efun'), ('H100', 'mobilenet24'), ('H100', 'mobilenet32'), ('H100', 'efun')]


def source_jobs():
    training = []
    for row in csv.DictReader((ROOT / 'results/paper/evaluation_records/training_window_selected.csv').open()):
        if row['pipeline'] != 'FastLens':
            continue
        gpu = '4090' if row['gpu'] == 'RTX 4090' else 'H100'
        name = row['paper_workload'].replace(' (reference)', '')
        model = 'swinv2' if name == 'SwinV2-T' else PROFILES[name]
        if (gpu, model) in REMAINING:
            training.append(dict(gpu=gpu, model=model, reference=row['raw_results'].split(';')[0]))
    assert len(training) == 7
    inference = []
    for row in csv.DictReader((ROOT / 'results/p0/deployment_inference_summary.csv').open()):
        if row['pipeline'] != 'FastLens':
            continue
        source = ROOT / row['raw_results'].split(';')[0].split('#')[0]
        metadata = read(source.with_name('run.command.json'))
        inference.append(dict(gpu='4090' if row['gpu'] == 'RTX 4090' else 'H100',
                              model=PROFILES[row['model']], reference=str(source),
                              command=metadata['command'], environment=metadata['environment']))
    assert len(inference) == 14
    return training, inference


def check_inference(data, original, model):
    if model == 'vitti':
        measured, previous = data['repeats'][0], original['repeats'][0]
        assert measured['images'] == 50000
        for key in ('sample_trace', 'correct_top1'):
            assert measured[key] == previous[key], key
        assert data['model']['checkpoint_sha256'] == original['model']['checkpoint_sha256']
        with np.load(data['semantic_artifact']) as actual, np.load(original['semantic_artifact']) as expected:
            for name in ('input_0', 'input_1'):
                delta = np.abs(actual[name] - expected[name])
                assert delta.max() <= .001 and delta.mean() <= .0001, 'Input tensor mismatch'
        elapsed = measured['repeat_scope_seconds']
        top1 = 100 * measured['correct_top1'] / 50000
    else:
        assert data['samples'] == 50000
        ids = 'image_ids' if model == 'swinv2' else 'sample_ids'
        for key in (ids, 'top1', 'checkpoint'):
            assert data[key] == original[key], key
        elapsed = data['elapsed_s'] if model == 'swinv2' else data['e2e_seconds']
        top1 = data['top1']
    assert predictions(data) == predictions(original), 'Full prediction mismatch'
    return elapsed, top1


def refresh_file_identity(entry):
    path = Path(entry['path'])
    current = file_identity(path)
    for key in ('size_bytes', 'mtime_ns'):
        assert current[key] == entry['file_identity'][key], str(path)
    if entry.get('sha256'):
        assert file_sha(path) == entry['sha256'], str(path)
    entry['file_identity'] = current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/imagelanes_remaining_20261008')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    training, inference = source_jobs()
    out = args.output.resolve()
    out.mkdir(exist_ok=args.resume)
    native = next((ROOT / 'build/fastlens/torch').glob('_fastlens_direct_dct.*.so'))
    if args.resume:
        os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(out / 'runtime_snapshot.json')
        verify_snapshot()
        frozen = read(out / 'runtime_snapshot.json')
        plan = read(out / 'plan.json')
        training, inference = plan['training'], plan['inference']
        state = read(out / 'status.json')
        assert state['status'] == 'FAILED'
        state.update(status='RUNNING', pid=os.getpid())
        state.pop('error', None)
    else:
        save(out / 'handoff.json', dict(native_path=str(native)))
        frozen = snapshot(out)
        previous = read(ROOT / 'results/table4_l3_recheck_20261007/runtime_snapshot.json')
        assert all(frozen['files'][name] == sha for name, sha in previous['files'].items()), 'Implementation changed since the three-group recheck'
        for folder in ('fastlens/benchmarks/system_dct_major', 'fastlens/benchmarks/baselines/system_rgbnomore/inference',
                       'fastlens/benchmarks/dct_models'):
            for path in (ROOT / folder).rglob('*.py'):
                frozen['files'][str(path.relative_to(ROOT))] = file_sha(path)
        for name in ('fastlens/benchmarks/development/historical/vit_inference_once.py', 'fastlens/benchmarks/paper/section_6_3/h2d_traffic/swin_block_major.py'):
            frozen['files'][name] = file_sha(ROOT / name)
        frozen['fingerprint'] = hashlib.sha256(json.dumps(frozen['files'], sort_keys=True).encode()).hexdigest()
        save(out / 'runtime_snapshot.json', frozen)
        save(out / 'plan.json', dict(training=training, inference=inference, trials=3,
                                     training_images=65536, inference_images=50000,
                                     policy='Serial runs, all three repeats retained, no performance-based retries; no full epochs or RGB baselines.'))
        state = dict(status='RUNNING', pid=os.getpid(), started=datetime.datetime.now().astimezone().isoformat(),
                     active=None, completed_runs=0, planned_runs=63)
    save(out / 'status.json', state)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(out / 'runtime_snapshot.json')
    env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp'), PYTHONPATH=f'{ROOT}:{native.parent}',
               FASTLENS_TRAINING_NATIVE_DIR=str(native.parent),
               FASTLENS_EVALUATION_SNAPSHOT=str(out / 'runtime_snapshot.json'),
               PYTHONDONTWRITEBYTECODE='1', FASTLENS_NSYS_FINE='0')
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    base = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable]
    records = list(csv.DictReader((out / 'trials.csv').open())) if args.resume else []
    for row in records:
        row['trial'] = int(row['trial'])
        row['images_per_s'] = float(row['images_per_s'])
    completed = {(r['phase'], r['gpu'], r['model'], r['trial']) for r in records}

    def launch(command, destination, paths=None):
        state['active'] = str(destination.relative_to(out))
        save(out / 'status.json', state)
        destination.mkdir(parents=True, exist_ok=True)
        run(command, destination, env, paths)
        meta = read(destination / 'run.command.json')
        assert meta['returncode'] == 0 and meta['code_unchanged_during_run']

    def record(phase, gpu, model, trial, images, seconds, raw, top1=None):
        records.append(dict(phase=phase, gpu=gpu, model=model, trial=trial, images=images,
                            seconds=seconds, images_per_s=images / seconds, top1=top1,
                            raw_result=str(raw.relative_to(ROOT))))
        write_csv(out / 'trials.csv', records)
        state['completed_runs'] += 1
        save(out / 'status.json', state)
        print('COMPLETED', records[-1], flush=True)

    try:
        for job in training:
            gpu, model = job['gpu'], job['model']
            if all(('training', gpu, model, trial) in completed for trial in range(3)):
                continue
            target = out / 'training' / gpu / model
            env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model,
                       OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
            matched = target / 'matched'
            check_command = (['fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py', '--model', model, '--epoch', '0']
                             if model == 'swinv2' else ['fastlens/benchmarks/development/diagnostics/check_mobile_training.py'])
            launch([*base, *check_command, '--output', str(matched / 'result.json')], matched)
            assert read(matched / 'result.json')['passed']
            check = target / 'check'
            command = window_command(gpu, model, 'imagelanes')
            launch([*base, *command, '--trial', '0', '--check', '--output', str(check)], check)
            assert read(check / 'result.json')['correctness_passed']
            old = read(ROOT / job['reference'])
            for trial in range(3):
                if ('training', gpu, model, trial) in completed:
                    continue
                dest = target / f'trial_{trial}'
                launch([*base, *command, '--trial', str(trial), '--correctness-report', str(check / 'result.json'),
                        '--output', str(dest)], dest)
                result = read(dest / 'result.json')
                assert result['images'] == 65536 and result['updates'] == 64
                assert result['correctness_passed'] and result['loss_finite'] and result['parameter_update_verified']
                assert not result['profile'] and not result['memory_diagnostic']
                assert result['cache_before']['resident_fraction'] == 1
                for key in ('initial_model_hash', 'sample_order_sha256', 'precision', 'batch', 'accumulation', 'warmup_updates'):
                    assert result[key] == old[key], key
                record('training', gpu, model, trial, result['images'], result['seconds'], dest / 'result.json')
        for job in inference:
            gpu, model = job['gpu'], job['model']
            source = Path(job['reference'])
            original = read(source)
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env[name] = job['environment'][name]
            env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model)
            env.pop('CUBLAS_WORKSPACE_CONFIG', None)
            for trial in range(3):
                if ('inference', gpu, model, trial) in completed:
                    continue
                dest = out / 'inference' / gpu / model / f'trial_{trial}'
                if dest.exists():
                    dest.rename(dest.with_name(dest.name + '_failed_' + datetime.datetime.now().strftime('%H%M%S')))
                dest.mkdir(parents=True)
                command = list(job['command'])
                option = '--output' if '--output' in command else '--output-dir'
                raw = dest / source.name
                command[command.index(option) + 1] = str(raw if option == '--output' else dest)
                if '--contract' in command:
                    contract = read(command[command.index('--contract') + 1])
                    artifact = dict(path=str(native), sha256=frozen['native_sha256'],
                                    size_bytes=native.stat().st_size, file_identity=file_identity(native))
                    for config in contract['pipelines'].values():
                        if 'torch_binding_dir' in config:
                            config['torch_binding_dir'] = str(native.parent)
                        for field in ('torch_binding_artifact', 'native_binary_fingerprint'):
                            if field in config:
                                config[field] = artifact
                    storage = contract['dataset']['dct_major_storage']
                    access = contract['dataset']['block_major_access']
                    for entry in [storage['manifest'], *storage['payloads'], access['companion_index'],
                                  *access['shards'], *access['active_output_schedules']]:
                        refresh_file_identity(entry)
                    sample_path = Path(contract['dataset']['sample_manifest'])
                    samples = read(sample_path)
                    _, measured = selected_batches(contract, samples['samples'])
                    assert sample_trace(measured) == original['repeats'][0]['sample_trace']
                    for entry in samples['samples']:
                        refresh_file_identity(entry)
                    save(dest / 'samples.json', samples)
                    contract['dataset']['sample_manifest'] = str(dest / 'samples.json')
                    contract['dataset']['sample_manifest_sha256'] = sha256_json(samples)
                    save(dest / 'contract.json', contract)
                    command[command.index('--contract') + 1] = str(dest / 'contract.json')
                    manifest = Path(contract['pipelines'][command[command.index('--pipeline') + 1]]['manifest'])
                else:
                    manifest = (ROOT / 'fastlens/data/compressed/imagenet512_val_block_major/manifest.bin'
                                if model == 'swinv2' else Path(command[command.index('--new-data') + 1]) / 'manifest.bin')
                paths = [Path(shard['fls_path']) for shard in parse_manifest(manifest)['shards']]
                launch(command, dest, paths)
                data = read(raw)
                seconds, top1 = check_inference(data, original, model)
                record('inference', gpu, model, trial, 50000, seconds, raw, top1)
        summaries = []
        for key in dict.fromkeys((r['phase'], r['gpu'], r['model']) for r in records):
            group = [r for r in records if (r['phase'], r['gpu'], r['model']) == key]
            rates = [r['images_per_s'] for r in group]
            assert len(rates) == 3
            summaries.append(dict(phase=key[0], gpu=key[1], model=key[2], trials=3,
                                  median_images_per_s=statistics.median(rates), min_images_per_s=min(rates),
                                  max_images_per_s=max(rates), top1=group[0]['top1'],
                                  raw_results=';'.join(r['raw_result'] for r in group)))
        write_csv(out / 'summary.csv', summaries)
    except (AssertionError, RuntimeError, subprocess.CalledProcessError) as error:
        state.update(status='FAILED', error=str(error))
        save(out / 'status.json', state)
        raise
    state.update(status='COMPLETE', active=None, finished=datetime.datetime.now().astimezone().isoformat())
    save(out / 'status.json', state)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
