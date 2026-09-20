"""Run the remaining CNN experiments sequentially, independently of the chat session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from report_cnn_suite import CONFIGS, DATA, HERE, ROOT

GPU = 'GPU-40c637bd-acf5-ea1a-0df8-617138228467'
TRAIN_RUN = 'rtx4090_cnn_training_baselines_20260914'
MEMORY_RUN = 'rtx4090_cnn_memory_20260914'
SUMMARY = DATA/'runs/dctnet_mobilenet24/rtx4090_cnn_complete_20260914'
PROFILES = ['mobilenet24', 'mobilenet32', 'resnet24', 'resnet64']
BASELINES = [(p, b) for p in PROFILES for b in ('native', 'jpeg')]
BASELINES += [(p, b) for p in ('mobilenet24', 'resnet24') for b in ('rgb_pytorch', 'rgb_d2', 'rgb_d3')]


def directory(profile):
    return dict(zip(PROFILES, [c[1] for c in CONFIGS]))[profile]


def read(path):
    return json.loads(path.read_text())


def validate_training(folder, full):
    result = read(folder/'training.json')
    if full:
        assert result['full_training_epochs_completed'] == 2, folder
        assert result['nvml_process_peak_mib'] is not None, folder
        for epoch in range(2):
            row = read(folder/f'epoch_{epoch}.json')
            assert row['full_epoch'] and row['samples'] == row['unique_samples'] == 1281167, folder
            assert row['validation']['samples'] == 50000, folder
    else:
        assert result['data_only'], folder
        row = result['epochs'][0]
        assert row['samples'] == row['unique_samples'] == 20480, folder


def jobs(memory_run=MEMORY_RUN):
    # The existing matrix skips completed configurations and resumes completed epochs.
    yield ('B6_full', ['bash', str(HERE/'native_training_matrix.sh'), 'full',
        'rtx4090_native_training_optimized_20260914', 'B6', '4', '32768'], None, 'b6', None)
    for mode in ('full', 'data', 'profile'):
        for profile, backend in BASELINES:
            if mode == 'full' and backend == 'native':
                continue
            out = DATA/'runs'/directory(profile)/TRAIN_RUN/f'{backend}_{mode}'
            expected = out/('breakdown.json' if mode == 'profile' else 'training.json')
            yield (f'{profile}_{backend}_{mode}', ['bash', str(HERE/'cnn_training_baselines.sh'),
                mode, TRAIN_RUN, f'{profile}:{backend}'], expected, mode, out)
    for profile in PROFILES:
        run = DATA/'runs'/directory(profile)/memory_run
        mother = 'dctnet_mobilenet32' if profile.startswith('mobilenet') else 'dctnet_static64'
        yield (f'{profile}_inference_memory', ['bash', str(HERE/'complete_dct24.sh'), GPU, memory_run,
            f'{profile}:{directory(profile)}:{mother}'], None, 'inference', run/'full')
        yield (f'{profile}_model_only', [sys.executable, str(HERE/'diagnose_models.py'),
            '--output', str(run/'models.json')], run/'models.json', 'json', run)
        # Same 4096-image capture after 16384 warmup images, for every inference path.
        for route in ('rgb_pytorch', 'rgb_dali', 'R', 'O', 'grid_off', 'grid_on', 'projected_off', 'projected_on'):
            out = run/'profiles'/route
            stem = 'rgb_dali' if route == 'rgb_dali' else ('galp_native' if route.startswith('grid') else
                'galp_projected' if route.startswith('projected') else route)
            if route.startswith('rgb_'):
                command = [sys.executable, str(HERE/'evaluate_rgb.py'), '--route', route[4:], '--workers', '64']
            elif route in ('R', 'O'):
                command = [sys.executable, str(HERE/'evaluate.py'), '--route', route, '--device', 'cuda', '--workers', '64']
            else:
                layout, pushdown = route.split('_')
                command = [sys.executable, str(HERE/'evaluate_shards.py'), '--new-data',
                    str(DATA/f'dct_major_{mother}'), '--output-layout', layout, '--pushdown', pushdown,
                    '--baseline-dir', str(run/'full')]
            command += ['--count', '50000', '--batch-size', '64', '--profile', '--output-dir', str(out)]
            yield (f'{profile}_inference_trace_{route}', command, out/'breakdown.json', 'inference_profile', (out, stem))


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    return {int(pid.strip()) for line in output.splitlines() if line.strip()
            for uuid, pid in [line.split(',')] if uuid.strip() == GPU}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--check', action='store_true', help='list commands without starting GPU work')
    parser.add_argument('--inference-only', action='store_true')
    parser.add_argument('--memory-run', default=MEMORY_RUN)
    parser.add_argument('--wait-for-gpu', action='store_true', help='wait for other compute processes and reject contended measurements')
    args = parser.parse_args()
    work = [job for job in jobs(args.memory_run) if not args.inference_only or job[3] in ('inference', 'inference_profile', 'json')]
    if args.check:
        print(json.dumps([dict(name=n, command=c) for n, c, *_ in work], indent=2))
        return
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR='/home/tangyuxin/tmp/dctnet',
        MPLCONFIGDIR='/home/tangyuxin/tmp/matplotlib', MEASURE_PROCESS_MEMORY='1')
    control = (DATA/'runs/dctnet_mobilenet24'/args.memory_run/'queue') if args.inference_only else SUMMARY/'queue'
    control.mkdir(parents=True, exist_ok=True)
    state = dict(pid=os.getpid(), state='starting', started=time.time(), total_jobs=len(work), jobs=[], memory_run=args.memory_run)

    def save():
        state['updated'] = time.time()
        temporary = control/'status.json.new'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(control/'status.json')

    save()
    if args.wait_pid and psutil.pid_exists(args.wait_pid):
        process = psutil.Process(args.wait_pid)
        assert any(a.endswith('/native_training_matrix.sh') for a in process.cmdline()), process.cmdline()
        state.update(state='waiting_for_existing_B6', waiting_pid=args.wait_pid)
        save()
        print(f'Waiting for existing B6 matrix PID {args.wait_pid}', flush=True)
        process.wait()

    def run(command, log):
        print('Running: '+ ' '.join(command), flush=True)
        gpu_job = args.wait_for_gpu and (command[0] == 'bash' or command[:2] == ['nsys', 'profile'] or
            any(Path(a).name == 'diagnose_models.py' for a in command))
        if gpu_job:
            while foreign := gpu_processes():
                state.update(state='waiting_for_gpu', competing_pids=sorted(foreign))
                save()
                time.sleep(10)
            state.update(state='running', competing_pids=[])
            save()
        with log.open('a') as output:
            if not gpu_job:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
                return
            known = set()
            conflicts = set()
            with subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT) as child:
                root = psutil.Process(child.pid)
                known.add(child.pid)
                while child.poll() is None:
                    try:
                        known.update(p.pid for p in root.children(recursive=True))
                    except psutil.NoSuchProcess:
                        pass
                    current = gpu_processes()
                    # Include children that started while nvidia-smi was querying the driver.
                    try:
                        known.update(p.pid for p in root.children(recursive=True))
                    except psutil.NoSuchProcess:
                        pass
                    conflicts.update(current - known)
                    time.sleep(1)
                code = child.wait()
            measurement = dict(command=command, returncode=code, competing_pids=sorted(conflicts),
                sampling_seconds=1, gpu_uuid=GPU)
            log.with_suffix('.gpu_occupancy.json').write_text(json.dumps(measurement, indent=2)+'\n')
            if conflicts:
                raise subprocess.CalledProcessError(75, command, output=f'competing GPU processes: {sorted(conflicts)}')
            if code:
                raise subprocess.CalledProcessError(code, command)

    for name, command, expected, kind, out in work:
        state.update(state='running', current_job=name, command=command)
        save()
        os.environ['DCTNET_PROFILE'] = name.split('_')[0] if name != 'B6_full' else 'mobilenet24'
        log = control/f'{name}.log'
        try:
            if kind == 'inference_profile':
                folder, stem = out
                folder.mkdir(parents=True, exist_ok=True)
                if not (folder/f'{stem}.nsys-rep').exists():
                    run(['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                        '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--output', str(folder/stem), *command], log)
                if not (folder/f'{stem}.sqlite').exists():
                    run(['nsys', 'export', '--type', 'sqlite', '--output', str(folder/f'{stem}.sqlite'),
                        str(folder/f'{stem}.nsys-rep')], log)
                if not expected.exists():
                    run([sys.executable, str(HERE/'analyze_profiles.py'), str(folder)], log)
                assert read(expected)[stem]['captured_images'] == 4096
            else:
                if expected is None or not expected.exists():
                    run(command, log)
                if kind == 'b6':
                    for _, folder, _ in CONFIGS:
                        validate_training(DATA/'runs'/folder/'rtx4090_native_training_optimized_20260914/B6_full_m4_inplace', True)
                elif kind in ('full', 'data'):
                    validate_training(out, kind == 'full')
                elif kind == 'inference':
                    from report_cnn_suite import PATHS
                    for _, relative in PATHS:
                        assert read(out/relative)['samples'] == 50000, out/relative
                        metric = relative.split('/')[0] if '/' in relative else relative.removesuffix('_50000.json')
                        measurement = read(out/f'memory_{metric}.json')
                        assert measurement['returncode'] == 0 and measurement['nvml_process_peak_mib'] is not None, metric
                elif kind == 'profile':
                    assert read(expected)['captured_images'] == 16384, expected
                else:
                    read(expected)
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError, AssertionError) as error:
            # Each job owns its subprocesses. A failed job does not invalidate independent experiments.
            state['jobs'].append(dict(name=name, state='failed', error=str(error), detail=getattr(error, 'output', None), log=str(log)))
            print(f'FAILED {name}: {error}', flush=True)
        else:
            state['jobs'].append(dict(name=name, state='complete', log=str(log)))
        save()
    for script in ('report_cnn_suite.py', 'plot_cnn_suite.py'):
        try:
            command = [sys.executable, str(HERE/script)]
            if script == 'report_cnn_suite.py':
                command += ['--inference-run', args.memory_run]
            run(command, control/f'{script}.log')
        except subprocess.CalledProcessError as error:
            state['jobs'].append(dict(name=script, state='failed', error=str(error)))
    state.update(state='failed' if any(j['state'] == 'failed' for j in state['jobs']) else 'complete', finished=time.time())
    save()
    print(json.dumps(dict(state=state['state'], status=str(control/'status.json'), report=str(SUMMARY/'CNN_E2E_RESULTS.md'))), flush=True)
    raise SystemExit(1 if state['state'] == 'failed' else 0)


if __name__ == '__main__':
    main()
