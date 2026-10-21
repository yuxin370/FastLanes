#!/usr/bin/env python3
"""Separate 16,384-image RTX4090 resource windows after the primary phases."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.execution.resource_guard import ResourceGuard
from fastlens.benchmarks.common import gpu_uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipelines',nargs='+',choices=['pytorch','dali','coordl','ffcv','imagelanes'],required=True)
    args=parser.parse_args()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid('4090'),PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
             OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TMPDIR=str(Path.home()/'tmp'),FASTLENS_NSYS_FINE='0')
    guard=ResourceGuard(env['CUDA_VISIBLE_DEVICES'])
    for model in ['swinv2','mobilenet']:
        for pipeline in args.pipelines:
            for mode in ['memory','profile']:
                output=ROOT/'results/resources/raw/4090'/model/pipeline/mode
                if (output/'accepted.json').exists():
                    continue
                if (output/'run.log').exists():
                    print(f'BLOCKED: retained incomplete attempt {output}',flush=True)
                    continue
                while guard.blockers():
                    print('Waiting for resources',guard.observations,flush=True)
                    time.sleep(30)
                output.mkdir(parents=True,exist_ok=True)
                python=str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if pipeline=='coordl' else sys.executable
                command=[python,'fastlens/benchmarks/paper/shared/training_window_rgb.py','--gpu','4090','--model',model,'--pipeline',pipeline,
                         '--trial','0','--output',str(output),'--measured-images','16384','--'+mode]
                if mode=='profile':
                    command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                        '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(output/'capture'),*command]
                command=['numactl','--cpunodebind=0','--membind=0',sys.executable,'fastlens/benchmarks/execution/run_logged.py',
                         '--log',str(output/'run.log'),'--',*command]
                with (output/'launcher.log').open('x') as log,(output/'gpu_process_memory.csv').open('x') as telemetry, (output/'concurrent_load.jsonl').open('x') as load:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    monitor=subprocess.Popen(['nvidia-smi','--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory',
                                              '--format=csv,noheader,nounits','-lms','100'],stdout=telemetry)
                    reason=None
                    try:
                        while child.poll() is None:
                            try:
                                owned={child.pid,*(p.pid for p in psutil.Process(child.pid).children(recursive=True))}
                            except psutil.NoSuchProcess:
                                break
                            blocked=guard.blockers(owned)
                            load.write(json.dumps(dict(timestamp=time.time(),observations=guard.observations,
                                load_average=os.getloadavg()))+'\n')
                            load.flush()
                            if blocked and (output/'measurement_active').exists():
                                reason=str(guard.observations)
                                os.killpg(child.pid,signal.SIGTERM)
                                break
                            time.sleep(2)
                        rc=child.wait()
                    finally:
                        monitor.terminate()
                        monitor.wait()
                if rc or reason:
                    (output/'failure.json').write_text(json.dumps(dict(returncode=rc,reason=reason),indent=2)+'\n')
                    continue
                result=json.loads((output/'result.json').read_text())
                if result['images']!=16384 or result['updates']!=16 or not result['correctness_passed']:
                    raise ValueError(f'Resource correctness gate failed: {output}')
                if mode=='profile':
                    subprocess.run([sys.executable,'fastlens/benchmarks/execution/run_logged.py','--log',str(output/'export.log'),'--',
                        'nsys','export','--type','sqlite','--output',str(output/'capture.sqlite'),str(output/'capture.nsys-rep')],
                        cwd=ROOT,env=env,check=True)
                (output/'accepted.json').write_text(json.dumps(dict(correctness_passed=True,resource_guard_passed=True,
                    raw_result=str(output/'result.json'),scope='resource diagnostic only'),indent=2)+'\n')
                subprocess.run([sys.executable,'fastlens/benchmarks/development/diagnostics/analyze_resource_windows.py'],cwd=ROOT,check=True)
                subprocess.run([sys.executable,'fastlens/benchmarks/development/historical/replan.py'],cwd=ROOT,check=True)
                subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
