"""Interleave Raw/JPEG FFCV rechecks for observed throughput or repeatability anomalies."""
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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.development.historical.run_compressed_ffcv import GPUS
from fastlens.benchmarks.development.historical.run_compressed_ffcv import MODELS
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot
from fastlens.benchmarks.system_dct_major.common import file_identity, sha256_json

TRAINING = [('H100', 'mobilenet24'), ('H100', 'swinv2'), ('H100', 'vitti'), ('H100', 'resnet24'),
            ('4090', 'vitti'), ('4090', 'swinv2'), ('4090', 'mobilenet24'), ('4090', 'efun')]
OUT = ROOT / 'results/ffcv_anomaly_recheck_20261008'
FIELDS = ('initial_model_hash', 'sample_order_sha256', 'precision', 'images', 'updates',
          'warmup_updates', 'batch', 'accumulation', 'seed', 'augmentation_configuration',
          'compile', 'tf32', 'cpu_affinity', 'pytorch_workers', 'host_threads')


def write_csv(path, data):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def jobs():
    with (ROOT / 'results/ffcv_jpeg_20261005/comparison.csv').open() as stream:
        pairs = list(csv.DictReader(stream))
    requested = [('inference', 'H100', 'mobilenet24')]
    requested += [('training', gpu, model) for gpu, model in TRAINING]
    result = []
    for task, gpu, model in requested:
        source = next(r for r in pairs if (r['task'], r['gpu'].replace('RTX ', ''), r['model'])
                      == (task, gpu, MODELS[model]))
        result.append(dict(task=task, gpu=gpu, model=model, source=source))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=args.resume)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(OUT / 'runtime_snapshot.json')
    if not args.resume:
        native = next((ROOT / 'build/fastlens/torch').glob('_fastlens_direct_dct.*.so'))
        save(OUT / 'handoff.json', dict(native_path=str(native)))
        frozen = snapshot(OUT)
        for folder in ('fastlens/benchmarks/system_dct_major', 'fastlens/benchmarks/baselines/system_rgbnomore/inference'):
            for path in (ROOT / folder).rglob('*.py'):
                frozen['files'][str(path.relative_to(ROOT))] = file_sha(path)
        for name in ('fastlens/benchmarks/development/historical/training_ffcv_crop.py', 'fastlens/benchmarks/development/historical/paired_rgb_once.py', 'fastlens/benchmarks/execution/inference_load.py', 'fastlens/benchmarks/development/historical/recheck_ffcv_anomalies.py'):
            path = ROOT / name
            frozen['files'][str(path.relative_to(ROOT))] = file_sha(path)
        frozen['fingerprint'] = hashlib.sha256(json.dumps(frozen['files'], sort_keys=True).encode()).hexdigest()
        save(OUT / 'runtime_snapshot.json', frozen)
        save(OUT / 'plan.json', dict(jobs=jobs(), repeats=3, measured_runs=54,
            selection='Absolute old/new or JPEG/raw throughput change >=5%, or training trial sample CV >5%.',
            policy='Serial GPUs; alternate raw/jpeg, jpeg/raw, raw/jpeg; retain all three trials; no performance-based retries; no paper edits.',
            CUBLAS_WORKSPACE_CONFIG=None,
            historical_environment_limit='Old launch metadata did not record CUBLAS_WORKSPACE_CONFIG; this experiment explicitly unsets it.'))
        save(OUT / 'status.json', dict(status='PREPARED', completed_runs=0, planned_runs=54, active=None))
    verify_snapshot()
    if args.prepare_only:
        print('Prepared 9 pairs / 54 measured runs plus 16 training checks.', flush=True)
        return
    state = read(OUT / 'status.json')
    assert state['status'] in ('PREPARED', 'FAILED'), state
    state.update(status='RUNNING', pid=os.getpid(), started=datetime.datetime.now().astimezone().isoformat())
    state.pop('error', None)
    save(OUT / 'status.json', state)
    records = list(csv.DictReader((OUT / 'trials.csv').open())) if (OUT / 'trials.csv').exists() else []
    for r in records:
        r['trial'], r['images_per_s'] = int(r['trial']), float(r['images_per_s'])
    completed = {(r['task'], r['gpu'], r['model'], r['encoding'], r['trial']) for r in records}
    env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp'), PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               PYTHONDONTWRITEBYTECODE='1', FASTLENS_NSYS_FINE='0')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)

    def execute(template, base, check=None, trial=None):
        # Retain interrupted attempts; only GPU contention is automatically retried.
        while True:
            dest = base
            attempt = 0
            while dest.exists():
                attempt += 1
                dest = base.with_name(base.name + f'_attempt_{attempt}')
            dest.mkdir(parents=True)
            command = list(template['command'])
            command[command.index('--output') + 1] = str(dest / 'pipeline_ffcv.json' if job['task'] == 'inference' else dest)
            if job['task'] == 'inference':
                contract = read(command[command.index('--contract') + 1])
                samples = read(contract['dataset']['sample_manifest'])
                for sample in samples['samples']:
                    current = file_identity(Path(sample['path']))
                    assert current['size_bytes'] == sample['size_bytes']
                    sample['file_identity'] = current
                save(dest / 'samples.json', samples)
                contract['dataset'].update(sample_manifest=str(dest / 'samples.json'), sample_manifest_sha256=sha256_json(samples))
                save(dest / 'contract.json', contract)
                command[command.index('--contract') + 1] = str(dest / 'contract.json')
                paths = [Path(contract['pipelines']['ffcv']['beton'])]
                result = dest / 'pipeline_ffcv.json'
            else:
                option = command.index('--correctness-report')
                del command[option:option + 2]
                command[command.index('--trial') + 1] = str(trial or 0)
                command += ['--correctness-report', str(check)] if check else ['--check']
                paths = None
                result = dest / 'result.json'
            state['active'] = str(dest.relative_to(OUT))
            save(OUT / 'status.json', state)
            telemetry = subprocess.check_output(['nvidia-smi', '-i', env['CUDA_VISIBLE_DEVICES'],
                '--query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu,utilization.gpu,memory.used',
                '--format=csv'], text=True)
            (dest / 'gpu_before.csv').write_text(telemetry)
            try:
                run(command, dest, env, paths)
            except RuntimeError as error:
                if not str(error).startswith('Target GPU contention:'):
                    raise
                print(str(error) + '; retained attempt; waiting for target GPU', flush=True)
                continue
            meta = read(dest / 'run.command.json')
            assert meta['returncode'] == 0 and meta['code_unchanged_during_run']
            return result

    try:
        for job in read(OUT / 'plan.json')['jobs']:
            task, gpu, model = job['task'], job['gpu'], job['model']
            env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model)
            templates, references, checks = {}, {}, {}
            target = OUT / task / gpu / model
            for encoding in ('raw', 'jpeg'):
                ref = ROOT / job['source'][encoding + '_results'].split(';')[0]
                references[encoding] = read(ref)
                templates[encoding] = read(ref.parent / 'run.command.json')
                if task == 'training' and not all((task, gpu, model, encoding, i) in completed for i in range(3)):
                    checks[encoding] = execute(templates[encoding], target / encoding / 'check')
                    assert read(checks[encoding])['correctness_passed']
            for trial in range(3):
                for encoding in (('raw', 'jpeg') if trial % 2 == 0 else ('jpeg', 'raw')):
                    if (task, gpu, model, encoding, trial) in completed:
                        continue
                    result = execute(templates[encoding], target / encoding / f'trial_{trial}', checks.get(encoding), trial)
                    data, reference = read(result), references[encoding]
                    top1 = None
                    if task == 'training':
                        assert data['correctness_passed'] and data['loss_finite'] and data['parameter_update_verified']
                        assert not data['profile'] and not data['memory_diagnostic']
                        assert data['cache_before']['resident_fraction'] == 1
                        for field in FIELDS:
                            assert data[field] == reference[field], (field, data[field], reference[field])
                        images, seconds = data['images'], data['seconds']
                    else:
                        measured, previous = data['repeats'][0], reference['repeats'][0]
                        assert data['model'] == reference['model']
                        for field in ('sample_trace', 'top1_predictions_sha256', 'correct_top1'):
                            assert measured[field] == previous[field], field
                        images, seconds = measured['images'], measured['repeat_scope_seconds']
                        assert images == 50000
                        top1 = measured['accuracy_top1'] * 100
                    record = dict(task=task, gpu=gpu, model=model, encoding=encoding, trial=trial,
                                  images=images, seconds=seconds, images_per_s=images / seconds,
                                  top1=top1, raw_result=str(result.relative_to(ROOT)))
                    records.append(record)
                    write_csv(OUT / 'trials.csv', records)
                    state['completed_runs'] = len(records)
                    save(OUT / 'status.json', state)
                    print('COMPLETED', record, flush=True)
        summaries = []
        for task, gpu, model, encoding in dict.fromkeys((r['task'], r['gpu'], r['model'], r['encoding']) for r in records):
            group = [r for r in records if (r['task'], r['gpu'], r['model'], r['encoding']) == (task, gpu, model, encoding)]
            rates = [r['images_per_s'] for r in group]
            assert len(rates) == 3
            summaries.append(dict(task=task, gpu=gpu, model=model, encoding=encoding, trials=3,
                median_images_per_s=statistics.median(rates), min_images_per_s=min(rates), max_images_per_s=max(rates),
                cv_percent=100 * statistics.stdev(rates) / statistics.mean(rates), top1=group[0]['top1'],
                raw_results=';'.join(r['raw_result'] for r in group)))
        write_csv(OUT / 'summary.csv', summaries)
    except (AssertionError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        state.update(status='FAILED', error=str(error))
        save(OUT / 'status.json', state)
        raise
    state.update(status='COMPLETE', active=None, finished=datetime.datetime.now().astimezone().isoformat())
    save(OUT / 'status.json', state)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
