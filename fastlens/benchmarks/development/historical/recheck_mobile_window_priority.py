"""Six fixed MobileNetV2-DCT24 window trials before the waiting H100 epoch queue."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot

OUT = ROOT / 'results/mobilenet24_window_recheck_20261002'
FULL_CONTROLLER = 2834770
H100_COMPETITOR = 2603424


def main():
    controller = psutil.Process(FULL_CONTROLLER)
    if 'fastlens/benchmarks/development/historical/retry_full_rgb_contention.py' not in controller.cmdline():
        raise RuntimeError('Full-epoch controller identity changed')
    if controller.children(recursive=True):
        raise RuntimeError('Full-epoch controller has active work; leave it untouched')
    full = read(ROOT / 'results/full_rgb_epochs_20261002/status.json')
    if any(j['status'] == 'RUNNING' for j in full['jobs']):
        raise RuntimeError('An epoch is still active; leave it untouched')
    OUT.mkdir()
    handoff = read(ROOT / 'results/training_expansion_20260930/handoff.json')
    save(OUT / 'handoff.json', handoff)
    frozen = snapshot(OUT)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(OUT / 'runtime_snapshot.json')
    state = dict(status='RUNNING', pid=os.getpid(), active=None, completed=[],
                 policy='Three fixed trials per GPU; all retained, no best-of selection; 4090 first while H100 recurring workload is active.',
                 suspended_idle_controller=FULL_CONTROLLER, native_sha256=handoff['native_sha256'])
    save(OUT / 'status.json', state)
    (OUT / 'PROTOCOL.md').write_text(
        'MobileNetV2-DCT24 FastLens window recheck: 3 trials per GPU, 65,536 measured images, '
        '64 updates, 32 excluded warmup updates, BF16, microbatch 64, accumulation 16, seed 11997733. '
        'Use the pinned training native runtime already used by the expansion campaign. '
        'NUMA0, 8 host threads, B6 augmentation. Pass matched tensor/update and short-window gates first. '
        'Require historical initialization/sample-order hashes and warm source cache. '
        'Retain all old and new observations; report median/min/max and regression against the old cohort. '
        'Other-GPU activity is annotated; target-GPU activity blocks. No active epoch is suspended.\n')
    old_summary = list(csv.DictReader((ROOT / 'results/p0/training_window_summary.csv').open()))
    save(OUT / 'previous_window_summary.json', old_summary)
    devices = {r['requested']: r['selection'] for r in read(ROOT / 'results/inventory/device_mapping.json')}
    accepted = []

    def launch(command, destination, env):
        destination.mkdir(parents=True)
        run(command, destination, env, None)

    def terminate(signum, frame):
        raise SystemExit(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    controller.suspend()
    try:
        for gpu in ('4090', 'H100'):
            if gpu == 'H100' and psutil.pid_exists(H100_COMPETITOR):
                competitor = psutil.Process(H100_COMPETITOR)
                if 'run_scale_phj_gds17_260930.py' not in ' '.join(competitor.cmdline()):
                    raise RuntimeError('H100 competitor PID identity changed')
                state.update(status='WAITING_H100', active=None)
                save(OUT / 'status.json', state)
                print('Waiting for recurring H100 workload', H100_COMPETITOR, flush=True)
                while competitor.is_running() and competitor.status() != psutil.STATUS_ZOMBIE:
                    time.sleep(5)
            native_dir = str(Path(handoff['native_path']).parent)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[gpu], DCTNET_PROFILE='mobilenet24',
                       TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                       OPENBLAS_NUM_THREADS='8', PYTHONPATH=f'{ROOT}:{native_dir}',
                       FASTLENS_TRAINING_NATIVE_DIR=native_dir, PYTHONDONTWRITEBYTECODE='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
            base = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable]
            matched = OUT / gpu / 'matched'
            state.update(status='RUNNING', active=f'{gpu}:matched_correctness')
            save(OUT / 'status.json', state)
            launch([*base, 'fastlens/benchmarks/development/diagnostics/check_mobile_training.py', '--output', str(matched / 'result.json')], matched, env)
            if not read(matched / 'result.json')['passed']:
                raise ValueError('Matched tensor/update gate failed')
            options = ['fastlens/benchmarks/paper/shared/training_window_rgb.py', '--gpu', gpu, '--model', 'mobilenet', '--pipeline', 'imagelanes']
            gate = OUT / gpu / 'check'
            launch([*base, *options, '--trial', '0', '--check', '--output', str(gate)], gate, env)
            if not read(gate / 'result.json')['correctness_passed']:
                raise ValueError('Window correctness gate failed')
            reference = read(ROOT / f'results/p0/raw/{gpu}/mobilenet/fastlens/trial_0_refresh_20260929/result.json')
            for trial in range(3):
                destination = OUT / gpu / f'trial_{trial}'
                state['active'] = f'{gpu}:trial_{trial}'
                save(OUT / 'status.json', state)
                launch([*base, *options, '--trial', str(trial), '--correctness-report', str(gate / 'result.json'),
                        '--output', str(destination)], destination, env)
                result = read(destination / 'result.json')
                metadata = read(destination / 'run.command.json')
                for key in ('initial_model_hash', 'sample_order_sha256', 'precision', 'batch', 'accumulation', 'warmup_updates'):
                    if result[key] != reference[key]:
                        raise ValueError(f'Historical window contract changed: {key}')
                ids = read(Path(result['sample_ids_file']))
                if not (result['correctness_passed'] and result['loss_finite'] and result['parameter_update_verified']
                        and result['images'] == 65536 and result['updates'] == 64 and result['mode'] == 'training'
                        and not result['profile'] and not result['memory_diagnostic']
                        and result['cache_before']['resident_fraction'] == 1
                        and len(ids) == len(set(ids)) == 65536
                        and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == result['sample_order_sha256']
                        and metadata['returncode'] == 0 and metadata['code_unchanged_during_run']):
                    raise ValueError('Window acceptance failed')
                save(destination / 'accepted.json', dict(correctness_passed=True, raw_result=str(destination / 'result.json'),
                     code_fingerprint=frozen['fingerprint'], correctness_report=str(gate / 'result.json')))
                accepted.append(dict(gpu=gpu, trial=trial, samples=result['images'], elapsed_s=result['seconds'],
                                     images_per_s=result['images'] / result['seconds'], raw_result=str(destination / 'result.json')))
                state['completed'].append(str(destination))
                save(OUT / 'status.json', state)
            rates = [r['images_per_s'] for r in accepted if r['gpu'] == gpu]
            previous = next(r for r in old_summary if r['gpu'] == ('RTX 4090' if gpu == '4090' else gpu)
                            and r['model'] == 'MobileNetV2' and r['domain'] == 'DCT' and r['pipeline'] == 'FastLens')
            save(OUT / gpu / 'summary.json', dict(trials=3, median=statistics.median(rates), min=min(rates), max=max(rates),
                 cv_percent=100 * statistics.stdev(rates) / statistics.mean(rates),
                 previous_median=float(previous['median_images_per_s']),
                 change_percent=100 * (statistics.median(rates) / float(previous['median_images_per_s']) - 1),
                 raw_results=[r['raw_result'] for r in accepted if r['gpu'] == gpu]))
            with (OUT / 'window_raw.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(accepted[0]))
                writer.writeheader(); writer.writerows(accepted)
        state.update(status='COMPLETE', active=None)
    except BaseException as error:
        state.update(status='FAILED', error=str(error))
        raise
    finally:
        controller.resume()
        state['full_epoch_controller_resumed'] = True
        save(OUT / 'status.json', state)


if __name__ == '__main__':
    main()
