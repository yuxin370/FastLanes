"""Retest four authorized DCT windows in ascending frozen deployment-ratio order."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot

OUT = ROOT / 'results/remaining_4090_windows_20261002'
WAITING_CONTROLLER = 3018342
PROFILES = {'ViT-Ti DCT': 'vitti', 'MobileNetV2-DCT32': 'mobilenet32',
            'SwinV2-T DCT': 'swinv2', 'eFUN-DCT': 'efun', 'MobileNetV2-DCT24': 'mobilenet24'}


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=['4090', 'H100'], default='4090')
    parser.add_argument('--deadline', help='End of the authorized exclusive window, ISO timestamp')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--waiting-controller', type=int, default=WAITING_CONTROLLER,
                        help='Idle evaluation controller to suspend; 0 when the old controller has exited')
    parser.add_argument('--models', nargs='+', choices=sorted(PROFILES.values()))
    parser.add_argument('--reuse-checks', type=Path, help='Previously completed checks under the same runtime snapshot')
    parser.add_argument('--reuse-trials', type=Path, nargs='+', help='Reuse accepted trials from interrupted three-trial cohorts')
    args = parser.parse_args()
    label = 'RTX 4090' if args.gpu == '4090' else 'H100'
    if args.gpu == 'H100':
        OUT = ROOT / 'results/h100_exclusive_windows_20261002'
    if args.output:
        OUT = args.output.resolve()
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None
    controller = psutil.Process(args.waiting_controller) if args.waiting_controller else None
    mobile_queue = False
    if controller is not None:
        mobile_queue = 'fastlens/benchmarks/development/historical/recheck_mobile_window_priority.py' in controller.cmdline()
        full_queue = 'fastlens/benchmarks/development/historical/retry_full_rgb_contention.py' in controller.cmdline()
        if not (mobile_queue or full_queue):
            raise RuntimeError('Waiting controller identity changed')
        current = read(ROOT / ('results/mobilenet24_window_recheck_20261002/status.json' if mobile_queue
                               else 'results/full_rgb_epochs_20261002/status.json'))
        idle = current['status'] == 'WAITING_H100' if mobile_queue else not any(j['status'] == 'RUNNING' for j in current['jobs'])
        if not idle or controller.children(recursive=True):
            raise RuntimeError('H100 queue is no longer at an idle boundary; leave it untouched')
    OUT.mkdir()
    ratios = list(csv.DictReader((ROOT / 'results/paper/training_deployment_ratios.csv').open()))
    ordered = sorted([r for r in ratios if r['gpu'] == label and r['dct_model'] in PROFILES
                      and (not args.models or PROFILES[r['dct_model']] in args.models)
                      and (args.gpu == 'H100' or r['dct_model'] != 'MobileNetV2-DCT24')],
                     key=lambda r: float(r['deployment_ratio']))
    if args.gpu == 'H100':
        ordered.sort(key=lambda r: r['dct_model'] != 'MobileNetV2-DCT24')
    assert len(ordered) == (len(args.models) if args.models else 5 if args.gpu == 'H100' else 4)
    summaries = list(csv.DictReader((ROOT / 'results/p0/training_window_summary.csv').open()))
    jobs = []
    for ratio in ordered:
        historical = next(r for r in summaries if r['gpu'] == label and r['domain'] == 'DCT'
                          and r['pipeline'] == 'FastLens' and r['raw_results'] == ratio['dct_raw'])
        jobs.append(dict(model=PROFILES[ratio['dct_model']], display_model=ratio['dct_model'],
                         historical_summary=historical, previous_deployment_ratio=ratio['deployment_ratio'],
                         rgb_reference=ratio, status='TODO'))
    save(OUT / 'plan.json', dict(jobs=jobs, trials_per_model=3, measured_images=65536,
         scope='FastLens DCT window recheck only; RGB references reused; no complete epochs.',
         deadline=args.deadline,
         policy='H100 DCT24 first, then ascending prior deployment ratio; three trials, median/min/max; keep all old/new results; report regressions.'))
    handoff = read(ROOT / 'results/training_expansion_20260930/handoff.json')
    save(OUT / 'handoff.json', handoff)
    frozen = snapshot(OUT)
    if args.reuse_checks and read(args.reuse_checks / 'runtime_snapshot.json')['fingerprint'] != frozen['fingerprint']:
        raise RuntimeError('Cannot reuse correctness checks from a different runtime')
    for previous_cohort in args.reuse_trials or []:
        if read(previous_cohort / 'runtime_snapshot.json')['fingerprint'] != frozen['fingerprint']:
            raise RuntimeError('Cannot reuse trials from a different runtime')
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(OUT / 'runtime_snapshot.json')
    state = dict(status='RUNNING', pid=os.getpid(), active=None, jobs=jobs,
                 suspended_idle_controller=args.waiting_controller, completed_trials=0)
    save(OUT / 'status.json', state)
    native_dir = str(Path(handoff['native_path']).parent)
    gpu = next(r['selection'] for r in read(ROOT / 'results/inventory/device_mapping.json') if r['requested'] == args.gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TMPDIR=str(Path.home() / 'tmp'),
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               PYTHONPATH=f'{ROOT}:{native_dir}', FASTLENS_TRAINING_NATIVE_DIR=native_dir,
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', FASTLENS_NSYS_FINE='0')
    base = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable]
    raw = []
    results = []

    def launch(command, destination):
        destination.mkdir(parents=True)
        run(command, destination, env, None)

    def terminate(signum, frame):
        raise SystemExit(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    if controller is not None:
        controller.suspend()
    try:
        if controller is not None and controller.children(recursive=True):
            raise RuntimeError('H100 work started during handoff; resume and leave it untouched')
        for job in jobs:
            model = job['model']
            if deadline:
                prior_commands = [read((ROOT / p).with_name('run.command.json'))
                                  for p in job['historical_summary']['raw_results'].split(';')]
                predicted = 90 + sum(m['process_wall_s'] for m in prior_commands)
                if time.time() + predicted > deadline:
                    job.update(status='TODO', notes='Deferred: estimated three-trial duration exceeds exclusive window remainder')
                    save(OUT / 'status.json', state)
                    continue
            env['DCTNET_PROFILE'] = model
            job['status'] = 'RUNNING'
            state['active'] = f'{model}:correctness'
            save(OUT / 'status.json', state)
            try:
                check_root = args.reuse_checks.resolve() if args.reuse_checks else OUT
                matched = check_root / model / 'matched'
                command = ['fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py', '--model', model, '--epoch', '0'] if model in ('vitti', 'swinv2') else ['fastlens/benchmarks/development/diagnostics/check_mobile_training.py']
                if not args.reuse_checks:
                    launch([*base, *command, '--output', str(matched / 'result.json')], matched)
                else:
                    meta = read(matched / 'run.command.json')
                    if meta['returncode'] != 0 or not meta['code_unchanged_during_run']:
                        raise ValueError('Reused matched check did not complete cleanly')
                if not read(matched / 'result.json')['passed']:
                    raise ValueError('Matched tensor/update check failed')
                options = ['fastlens/benchmarks/paper/shared/training_window_rgb.py', '--gpu', args.gpu, '--model', model, '--pipeline', 'imagelanes']
                gate = check_root / model / 'check'
                if not args.reuse_checks:
                    launch([*base, *options, '--trial', '0', '--check', '--output', str(gate)], gate)
                else:
                    meta = read(gate / 'run.command.json')
                    if meta['returncode'] != 0 or not meta['code_unchanged_during_run']:
                        raise ValueError('Reused window check did not complete cleanly')
                if not read(gate / 'result.json')['correctness_passed']:
                    raise ValueError('Window correctness check failed')
                reference = read(ROOT / job['historical_summary']['raw_results'].split(';')[0])
                trials = []
                for trial in range(3):
                    destination = OUT / model / f'trial_{trial}'
                    state['active'] = f'{model}:trial_{trial}'
                    save(OUT / 'status.json', state)
                    previous = next((p.resolve() / model / f'trial_{trial}' for p in args.reuse_trials or []
                                     if (p / model / f'trial_{trial}' / 'accepted.json').exists()), None)
                    reused = previous is not None
                    if reused:
                        destination = previous
                        print(f'REUSE {destination}', flush=True)
                    else:
                        launch([*base, *options, '--trial', str(trial), '--correctness-report', str(gate / 'result.json'),
                                '--output', str(destination)], destination)
                    d = read(destination / 'result.json')
                    m = read(destination / 'run.command.json')
                    for key in ('initial_model_hash', 'sample_order_sha256', 'precision', 'batch', 'accumulation', 'warmup_updates'):
                        if d[key] != reference[key]:
                            raise ValueError(f'Historical contract changed: {key}')
                    ids = read(Path(d['sample_ids_file']))
                    if not (d['images'] == 65536 and d['updates'] == 64 and d['mode'] == 'training'
                            and d['correctness_passed'] and d['loss_finite'] and d['parameter_update_verified']
                            and not d['profile'] and not d['memory_diagnostic'] and d['cache_before']['resident_fraction'] == 1
                            and len(ids) == len(set(ids)) == 65536
                            and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == d['sample_order_sha256']
                            and m['returncode'] == 0 and m['code_unchanged_during_run']):
                        raise ValueError('Window acceptance failed')
                    if not reused:
                        save(destination / 'accepted.json', dict(correctness_passed=True, code_fingerprint=frozen['fingerprint'],
                             raw_result=str(destination / 'result.json'), correctness_report=str(gate / 'result.json')))
                    row = dict(gpu=label, model=model, trial=trial, samples=d['images'], elapsed_s=d['seconds'],
                               images_per_s=d['images'] / d['seconds'], raw_result=str(destination / 'result.json'))
                    raw.append(row)
                    trials.append(row)
                    state['completed_trials'] += 1
                    write_csv(OUT / 'window_raw.csv', raw)
                    save(OUT / 'status.json', state)
                rates = [r['images_per_s'] for r in trials]
                median = statistics.median(rates)
                result = dict(model=job['display_model'], median_images_per_s=median, min_images_per_s=min(rates),
                     max_images_per_s=max(rates), trials=3, cv_percent=100 * statistics.stdev(rates) / statistics.mean(rates),
                     old_median_images_per_s=float(job['historical_summary']['median_images_per_s']),
                     change_percent=100 * (median / float(job['historical_summary']['median_images_per_s']) - 1),
                     deployment_ratio=median / float(job['rgb_reference']['rgb_images_per_s']),
                     raw_results=';'.join(r['raw_result'] for r in trials))
                results.append(result)
                write_csv(OUT / 'window_summary.csv', results)
                job['status'] = 'DONE'
                print(json.dumps(result), flush=True)
            except (ValueError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as error:
                job.update(status='FAILED', error=str(error))
                save(OUT / model / 'failure.json', dict(error=str(error)))
                if 'Evaluation code changed' in str(error):
                    raise
            finally:
                save(OUT / 'status.json', state)
        state.update(status='COMPLETE' if all(j['status'] == 'DONE' for j in jobs) else 'FINISHED_WITH_FAILURES', active=None)
    except BaseException as error:
        state.update(status='FAILED', error=str(error))
        raise
    finally:
        if args.gpu == 'H100' and mobile_queue:
            # This idle queue's pending DCT24 job is superseded by this recheck.
            # Its existing SIGTERM handler resumes the full-epoch controller.
            controller.terminate()
        if controller is not None:
            controller.resume()
        state['h100_queue_resumed'] = controller is not None
        save(OUT / 'status.json', state)


if __name__ == '__main__':
    main()
