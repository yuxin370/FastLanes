#!/usr/bin/env python3
"""Run only the four short-warmup, single-full-epoch Swin comparisons after P0."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.resource_guard import ResourceGuard
from fastlens.benchmarks.common import gpu_uuid


def read(path):
    return json.loads(path.read_text())


def validate_epoch(output):
    run = output/'runs/B6/seed_11997733'
    result = read(run/'pause_result.json')
    records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    epoch, = [row for row in records if row.get('scope') == 'epoch']
    checks = result['integration_checks']
    if (epoch['epoch'] != 1 or epoch['epoch_samples'] != 1281167
            or epoch['epoch_optimizer_updates'] != 1252 or not math.isfinite(epoch['train_loss'])
            or not all(checks[key] for key in ('coverage_counters_valid', 'gradients_finite',
                                             'loss_finite', 'parameters_finite'))):
        raise ValueError(f'Full epoch correctness failed: {output}')
    warm = read(run.parent/'short_warmup.json')
    contract = read(run/'run_manifest.json')
    if not warm['initial_state_restored'] or warm['initial_model_hash'] != contract['initial_model_hash']:
        raise ValueError(f'Warmup changed initialization: {output}')
    return dict(epoch=epoch, initial_model_hash=contract['initial_model_hash'],
                recipe_hash=contract['recipe_hash'], precision=result['precision'],
                warmup=str((run.parent/'short_warmup.json').relative_to(ROOT)),
                raw_result=str((run/'metrics.jsonl').relative_to(ROOT)),
                correctness_passed=True, resource_guard_passed=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--retry-contended',action='store_true',help='Retry the interrupted RTX4090 CPU-DCT epoch in a new directory')
    args=parser.parse_args()
    subprocess.run([sys.executable, 'fastlens/benchmarks/development/historical/replan.py'], cwd=ROOT, check=True)
    with (ROOT/'results/experiment_manifest.csv').open() as stream:
        pending = [r for r in csv.DictReader(stream) if r['experiment_id'].startswith('p0:')
                   and r['task'] == 'training_window' and r['status'] not in ('DONE', 'BLOCKED')]
    subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],cwd=ROOT,check=True)
    if pending:
        raise RuntimeError('P0 still has runnable missing cells; P1 has not been launched')
    if not read(ROOT/'results/checks/swin_b6_native_cpu_tensors.json')['passed']:
        raise ValueError('Matched tensor gate has not passed')
    manifests = ROOT/'fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3'
    physical = ROOT/'fastlens/data/compressed/imagenet512_train_block_major_premixed'
    materialization = read(physical/'materialization_contract.json')
    for gpu in ('H100', '4090'):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), TMPDIR=str(Path.home()/'tmp'),
                   PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', OMP_NUM_THREADS='8',
                   MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', FASTLENS_NSYS_FINE='0')
        guard = ResourceGuard(env['CUDA_VISIBLE_DEVICES'])
        gate=ROOT/f'results/checks/swin_b6_epoch0_{gpu}.json'
        if not gate.exists():
            while guard.blockers():
                time.sleep(5)
            subprocess.run(['numactl','--cpunodebind=0','--membind=0',sys.executable,
                'fastlens/benchmarks/execution/run_logged.py','--log',str(gate.with_suffix('.log')),'--',sys.executable,
                'fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py','--epoch','0','--output',str(gate)],cwd=ROOT,env=env,check=True)
        if not read(gate)['passed']:
            raise ValueError('GPU-specific epoch-0 matched input gate failed')
        for pipeline in ('imagelanes', 'cpu_dct'):
            output = ROOT/'results/p1/short_warmup_v2'/gpu/pipeline
            if (output/'accepted.json').exists():
                continue
            if args.retry_contended and gpu=='4090' and pipeline=='cpu_dct':
                output=ROOT/'results/p1/short_warmup_retry'/gpu/pipeline
                if (output/'accepted.json').exists(): continue
            if (output/'run.log').exists():
                raise RuntimeError(f'Prior attempt requires audit, not automatic rerun: {output}')
            while guard.blockers():
                print('Waiting for P1 resources:', guard.observations, flush=True)
                time.sleep(30)
            output.mkdir(parents=True, exist_ok=True)
            plan = output/'matrix_execution_plan.json'
            if not plan.exists():
                planning = [sys.executable, '-m', 'fastlens.benchmarks.training_pls.run_matrix',
                    '--train-manifest', str(manifests/'train.json'), '--val-manifest', str(manifests/'val.json'),
                    '--layout-plan', str(ROOT/'results/preparation/training_layout/physical_layout_plan.json'),
                    '--conditions', 'B6', '--seeds', '11997733', '--model', 'rgbnomore-swinv2-t-dct-224-v1',
                    '--epochs', '300', '--stop-after-epoch', '1', '--execution-backend',
                    'native-physical-pls' if pipeline == 'imagelanes' else 'standard-rgbnomore-dct',
                    '--premixed-mapping-csv', materialization['ordered_mapping'],
                    '--expected-mapping-sha256', materialization['ordered_mapping_sha256'],
                    '--workers', '8', '--output-dir', str(output), '--device', 'cuda:0',
                    '--required-gpu-name-substring', gpu]
                if pipeline == 'imagelanes':
                    planning += ['--physical-fastlens-manifest', str(physical/'dct/manifest.bin')]
                subprocess.run(['numactl', '--cpunodebind=0', '--membind=0', sys.executable,
                    'fastlens/benchmarks/execution/run_logged.py', '--log', str(output/'plan.log'), '--', *planning],
                    cwd=ROOT, env=env, check=True)
            planned, = read(plan)['commands']
            train_args = [value for value in planned['argv'][3:] if value != '--resume']
            command = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable,
                'fastlens/benchmarks/execution/run_logged.py', '--log', str(output/'run.log'), '--', sys.executable,
                'fastlens/benchmarks/development/historical/full_epoch_short_warmup.py', *train_args]
            marker = Path(planned['output_dir']).parent/'measurement_active'
            with (output/'launcher.log').open('x') as log, (output/'gpu_process_memory.csv').open('x') as telemetry, (output/'concurrent_load.jsonl').open('x') as load_log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                monitor = subprocess.Popen(['nvidia-smi', '--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory',
                                            '--format=csv,noheader,nounits', '-lms', '100'], stdout=telemetry)
                reason = None
                try:
                    while child.poll() is None:
                        try:
                            owned = {child.pid, *(p.pid for p in psutil.Process(child.pid).children(recursive=True))}
                        except psutil.NoSuchProcess:
                            break
                        blocked=guard.blockers(owned)
                        load_log.write(json.dumps(dict(timestamp=time.time(),observations=guard.observations,load_average=os.getloadavg()))+'\n')
                        load_log.flush()
                        if blocked and marker.exists():
                            reason = str(guard.observations)
                            os.killpg(child.pid, signal.SIGTERM)
                            break
                        time.sleep(2)
                    rc = child.wait()
                finally:
                    monitor.terminate()
                    monitor.wait()
            if rc or reason:
                (output/'failure.json').write_text(json.dumps(dict(returncode=rc, reason=reason), indent=2)+'\n')
                raise RuntimeError(f'P1 attempt failed; retained for audit: {output}')
            accepted = validate_epoch(output)
            if pipeline == 'cpu_dct':
                native = read(ROOT/'results/p1/short_warmup_v2'/gpu/'fastlens/accepted.json')
                if any(accepted[key] != native[key] for key in ('initial_model_hash', 'recipe_hash', 'precision')):
                    raise ValueError('Paired model/recipe differs')
                if any(accepted['epoch'][key] != native['epoch'][key]
                       for key in ('sample_order_digest', 'pool_membership_digest')):
                    raise ValueError('Paired sample sequence or pool membership differs')
            (output/'accepted.json').write_text(json.dumps(accepted, indent=2)+'\n')
            subprocess.run([sys.executable, 'fastlens/benchmarks/development/historical/replan.py'], cwd=ROOT, check=True)
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
