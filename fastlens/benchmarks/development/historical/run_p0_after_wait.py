#!/usr/bin/env python3
"""Wait for the existing suite; collect missing H100 PyTorch/DALI P0 windows serially."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.execution.resource_guard import ResourceGuard

STATE = ROOT/'results/p0/queue_state.json'


def compute_pids():
    out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in out.splitlines() if line.strip()}


def external_pending(pid, created):
    try:
        process = psutil.Process(pid)
        return process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-created', type=float, required=True)
    parser.add_argument('--retry-native-tail',action='store_true',help='Use a new attempt directory after the audited incomplete-tail correction')
    parser.add_argument('--gpu', nargs='+', choices=['H100','4090'], default=['H100','4090'])
    parser.add_argument('--pipelines', nargs='+', choices=['pytorch','dali','imagelanes','coordl','ffcv'], default=['pytorch','dali','imagelanes','coordl','ffcv'])
    args = parser.parse_args()
    state = dict(controller_pid=os.getpid(), wait_pid=args.wait_pid, wait_created=args.wait_created,
                 external_finished=False, scope='All missing P0 five-system windows, H100 then RTX4090; retain equivalent accepted results')
    def record(phase, **extra):
        state.update(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state,indent=2)+'\n')
        print(state['updated_at'], phase, extra, flush=True)
    def refresh():
        subprocess.run([sys.executable,'fastlens/benchmarks/development/historical/replan.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],cwd=ROOT,check=True)

    record('WAITING_FOR_EXTERNAL_TASK')
    while external_pending(args.wait_pid, args.wait_created):
        time.sleep(30)
    state['external_finished'] = True
    record('WAITING_FOR_IDLE_HOST')
    refresh()
    with (ROOT/'results/p0/training_window_raw.csv').open() as stream:
        historical={(r['gpu'],r['model'],r['pipeline'],int(r['trial'])) for r in csv.DictReader(stream)
                    if r['valid']=='True' and int(r['samples'])==65536}
    pipeline_names={'pytorch':'PyTorch','dali':'DALI','imagelanes':'FastLens','coordl':'CoorDL','ffcv':'FFCV'}
    failures=[]
    for gpu in args.gpu:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), TMPDIR=str(Path.home()/'tmp'),
                   PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   OPENBLAS_NUM_THREADS='8', FASTLENS_NSYS_FINE='0')
        guard = ResourceGuard(env['CUDA_VISIBLE_DEVICES'])
        for model in ('swinv2','mobilenet'):
            for pipeline in args.pipelines:
                for trial in range(3):
                    output = ROOT/'results/p0/raw'/gpu/model/pipeline/f'trial_{trial}'
                    key = ('RTX 4090' if gpu=='4090' else gpu, 'SwinV2-T' if model=='swinv2' else 'MobileNetV2', pipeline_names[pipeline], trial)
                    if key in historical:
                        continue
                    if args.retry_native_tail and pipeline=='imagelanes':
                        output=output.with_name(output.name+'_tail_fixed')
                    accepted = output/'accepted.json'
                    if accepted.exists():
                        continue
                    if (output/'run.log').exists() or (output/'result.json').exists():
                        failures.append(str(output) + ': existing incomplete attempt requires audit; not overwritten')
                        break
                    while guard.blockers():
                        record('WAITING_FOR_IDLE_HOST', active=None, blockers=guard.observations)
                        time.sleep(30)
                    if pipeline == 'ffcv' and not (ROOT/'results/preparation/p0_ffcv/conversion.json').exists():
                        if (ROOT/'results/preparation/p0_ffcv_conversion.log').exists():
                            failures.append('Prior FFCV conversion failed; raw log retained for audit')
                            output.mkdir(parents=True,exist_ok=True)
                            (output/'failure.json').write_text(json.dumps(dict(reason=failures[-1],
                                source_log='results/preparation/p0_ffcv_conversion.log'),indent=2)+'\n')
                            break
                        record('PREPARING_FFCV', active=None)
                        prep = subprocess.Popen(['numactl','--cpunodebind=0','--membind=0',sys.executable,
                            'fastlens/benchmarks/execution/run_logged.py','--log','results/preparation/p0_ffcv_conversion.log','--',
                            sys.executable,'fastlens/benchmarks/baselines/ffcv/prepare_p0_ffcv.py'],cwd=ROOT,env=env,start_new_session=True)
                        try:
                            prep.wait(timeout=1800)
                        except subprocess.TimeoutExpired:
                            os.killpg(prep.pid,signal.SIGTERM)
                            prep.wait()
                        if prep.returncode:
                            failures.append('FFCV preparation failed; see results/preparation/p0_ffcv_conversion.log')
                            output.mkdir(parents=True,exist_ok=True)
                            (output/'failure.json').write_text(json.dumps(dict(reason=failures[-1],returncode=prep.returncode,
                                source_log='results/preparation/p0_ffcv_conversion.log'),indent=2)+'\n')
                            break
                    output.mkdir(parents=True, exist_ok=True)
                    interpreter = str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if pipeline=='coordl' else sys.executable
                    command = ['numactl','--cpunodebind=0','--membind=0',sys.executable,'fastlens/benchmarks/execution/run_logged.py',
                               '--log',str(output/'run.log'),'--',interpreter,'fastlens/benchmarks/paper/shared/training_window_rgb.py',
                               '--gpu',gpu,'--model',model,'--pipeline',pipeline,'--trial',str(trial),'--output',str(output)]
                    record('RUNNING', active=str(output.relative_to(ROOT)), command=command)
                    started = time.monotonic()
                    budget_elapsed = 0.0
                    budget_checked = started
                    with (output/'launcher.log').open('x') as stream, (output/'gpu_process_memory.csv').open('x') as telemetry, (output/'concurrent_load.jsonl').open('x') as load_log:
                        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        monitor = subprocess.Popen(['nvidia-smi','--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory',
                                                    '--format=csv,noheader,nounits','-lms','100'],stdout=telemetry)
                        reason = None
                        try:
                            while child.poll() is None:
                                owned = {child.pid}
                                try:
                                    owned.update(p.pid for p in psutil.Process(child.pid).children(recursive=True))
                                except psutil.NoSuchProcess:
                                    break
                                foreign = guard.blockers(owned)
                                load_log.write(json.dumps(dict(timestamp=time.time(),observations=guard.observations,load_average=os.getloadavg()))+'\n')
                                load_log.flush()
                                measuring = (output/'measurement_active').exists()
                                now = time.monotonic()
                                # Resource waiting is not baseline setup/debugging time.
                                if not foreign or measuring:
                                    budget_elapsed += now - budget_checked
                                budget_checked = now
                                if foreign and measuring:
                                    reason = f'Competing GPU/host resources during measurement: {guard.observations}; owned trial rejected'
                                elif budget_elapsed > 1800:
                                    reason = '30-minute baseline setup/execution budget reached'
                                if reason:
                                    if child.poll() is None:
                                        os.killpg(child.pid,signal.SIGTERM)
                                    break
                                time.sleep(2)
                            rc = child.wait()
                        finally:
                            monitor.terminate()
                            monitor.wait()
                    if reason or rc:
                        failure = dict(returncode=rc, reason=reason or 'Benchmark failed; inspect run.log', command=command)
                        (output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                        failures.append(str(output) + ': ' + failure['reason'])
                        record('BASELINE_BLOCKED', failure=failure)
                        refresh()
                        break
                    result = json.loads((output/'result.json').read_text())
                    if not result['correctness_passed'] or result['images'] != 65536 or result['updates'] != 64:
                        raise RuntimeError(f'P0 result failed correctness: {output}')
                    accepted.write_text(json.dumps(dict(raw_result=str(output/'result.json'),
                                                        correctness_passed=True, shared_host_guard_passed=True, resource_policy='same GPU exclusive; other GPU/host loads annotated in concurrent_load.jsonl; no isolation inferred'),indent=2)+'\n')
                    record('TRIAL_DONE', active=None)
                    refresh()
    record('P0_QUEUE_FINISHED', active=None, failures=failures,
           next_action='Audit blocked cases; only after P0 is complete proceed to strict full-epoch Swin')
    refresh()


if __name__ == '__main__':
    main()
