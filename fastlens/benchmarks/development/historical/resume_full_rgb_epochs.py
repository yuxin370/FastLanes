"""Continue the 16 authorized epochs after a clean boundary; fix only FFCV cache placement."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import OUT
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import DATA
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import PYTHON
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import read
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import save
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import report
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot


def accepted(path):
    r=read(path/'result.json');m=read(path/'run.command.json')
    if not (r['correctness_passed'] and r['full_epoch'] and r['images']==1281167 and r['updates']==1252
            and r['loss_finite'] and r['warmup_initial_state_restored'] and r['final_update_samples']==143
            and r['parameter_update_verified'] and m['returncode']==0 and m['code_unchanged_during_run']):
        raise ValueError('Full epoch acceptance failed: '+str(path))
    return r


def handoff():
    import psutil
    previous=read(OUT/'status.json')
    controller=psutil.Process(previous['pid'])
    if 'fastlens/benchmarks/development/historical/run_full_rgb_epochs.py' not in controller.cmdline():
        raise ValueError('Unexpected controller identity')
    active=[j for j in previous['jobs'] if j['status']=='RUNNING']
    if len(active)!=1:
        raise ValueError('Expected one active epoch at handoff scheduling')
    target=Path(active[0]['output'])/'run.command.json'
    save(OUT/'handoff_numa1.json',dict(status='WAITING_CURRENT_EPOCH',old_pid=controller.pid,
         new_pid=os.getpid(),target=str(target),started=datetime.datetime.now().astimezone().isoformat()))
    print('Waiting for current epoch, without interrupting it:',target,flush=True)
    while not target.exists() or 'returncode' not in read(target):
        if not controller.is_running() or controller.status()==psutil.STATUS_ZOMBIE:
            raise RuntimeError('Old controller exited before the selected epoch completed')
        time.sleep(.2)
    # Only the orchestrator is briefly stopped; the measured process has exited.
    controller.send_signal(signal.SIGSTOP)
    if list((OUT/'epochs').rglob('measurement_active')):
        controller.send_signal(signal.SIGCONT)
        raise RuntimeError('Another measured epoch already began; original queue left running')
    # A newly-started short correctness check may finish normally. No measured
    # process or third-party process is signalled or terminated.
    while True:
        live=[]
        for child in controller.children(recursive=True):
            try:
                if child.is_running() and child.status()!=psutil.STATUS_ZOMBIE:
                    live.append(child.pid)
            except psutil.NoSuchProcess:
                pass
        if not live:break
        time.sleep(.2)
    controller.send_signal(signal.SIGTERM)
    controller.send_signal(signal.SIGCONT)
    controller.wait()
    save(OUT/'handoff_numa1.json',dict(status='OLD_CONTROLLER_EXITED_AT_BOUNDARY',old_pid=controller.pid,
         new_pid=os.getpid(),target=str(target),timestamp=datetime.datetime.now().astimezone().isoformat()))


def main():
    handoff()
    state=read(OUT/'status.json')
    # Reuse results which finished before the old controller could publish them.
    initials={};order=None
    for job in state['jobs']:
        path=Path(job['output'])
        if (path/'result.json').exists():
            r=accepted(path);job['status']='DONE'
            config=(r['initial_model_hash'],r['precision'],r['batch'],r['accumulation'])
            if initials.setdefault(job['model'],config)!=config:raise ValueError('RGB initialization mismatch')
            if order is not None and order!=r['sample_order_sha256']:raise ValueError('RGB sample order mismatch')
            order=r['sample_order_sha256']
            if not (path/'accepted.json').exists():
                save(path/'accepted.json',dict(correctness=True,comparison_type='deployment',raw_result=str(path/'result.json')))
        elif job['status']=='FAILED' and job['pipeline']=='ffcv':
            job['previous_attempts']=[dict(output=job['output'],error=job.get('error',''))]
            job['output']=str(path.parent/'ffcv_numa1_retry')
            job.pop('error',None);job['status']='TODO'
        elif job['status']=='RUNNING':
            job['status']='TODO'
    snapshot=read(OUT/'runtime_snapshot.json')
    changed=[p for p,sha in snapshot['files'].items() if file_sha(p)!=sha]
    if changed:raise RuntimeError('Original runtime changed: '+', '.join(changed))
    paths=[str(Path(__file__).resolve()),str(ROOT/'fastlens/benchmarks/development/historical/full_ffcv_remote_cache.py'),
           str(ROOT/'fastlens/benchmarks/profiling/input_motivation.py')]
    files=dict(snapshot['files']);files.update({p:file_sha(p) for p in paths})
    snapshot=dict(files=files,fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    save(OUT/'runtime_snapshot_numa1.json',snapshot)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT']=str(OUT/'runtime_snapshot_numa1.json')
    state.update(status='RUNNING',pid=os.getpid(),cache_placement_revision='FFCV excluded warm/query on node 1; timed process unchanged on node 0')
    report(state)
    devices={x['requested']:x['selection'] for x in read(ROOT/'results/inventory/device_mapping.json')}
    for job in state['jobs']:
        if job['status']=='DONE':continue
        destination=Path(job['output']);destination.mkdir(parents=True,exist_ok=True)
        gate=OUT/'checks'/job['gpu']/job['model']/job['pipeline'];gate.mkdir(parents=True,exist_ok=True)
        interpreter=str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if job['pipeline']=='coordl' else PYTHON
        options=['--gpu',job['gpu'],'--model',job['model'],'--pipeline',job['pipeline'],'--trial','0']
        base=['numactl','--cpunodebind=0','--membind=0',interpreter]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[job['gpu']],DCTNET_PROFILE=job['model'],
            TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
            PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        job['status']='RUNNING';report(state)
        try:
            verify_snapshot()
            if not (gate/'result.json').exists():
                run([*base,'fastlens/benchmarks/paper/shared/training_window_rgb.py',*options,'--check','--output',str(gate)],gate,env,None)
            check=read(gate/'result.json')
            if not check['correctness_passed']:raise ValueError('Small input/update gate failed')
            config=(check['initial_model_hash'],check['precision'],check['batch'],check['accumulation'])
            if initials.setdefault(job['model'],config)!=config:raise ValueError('RGB configuration differs')
            driver='fastlens/benchmarks/development/historical/full_ffcv_remote_cache.py' if job['pipeline']=='ffcv' else 'fastlens/benchmarks/paper/shared/training_window_rgb.py'
            run([*base,driver,*options,'--full-epoch','--ffcv-root',str(DATA),'--correctness-report',str(gate/'result.json'),
                 '--output',str(destination)],destination,env,None)
            r=accepted(destination)
            if order!=r['sample_order_sha256']:raise ValueError('RGB full sample order differs')
            save(destination/'accepted.json',dict(correctness=True,comparison_type='deployment',raw_result=str(destination/'result.json')))
            job['status']='DONE'
        except (RuntimeError,ValueError,subprocess.CalledProcessError,FileNotFoundError) as error:
            job.update(status='FAILED',error=str(error));save(destination/'failure.json',dict(error=str(error)))
            if 'Evaluation code changed' in str(error):
                state['status']='BLOCKED_CODE_CHANGE';report(state);break
        finally:
            if job['pipeline']=='coordl':
                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
                cleanup_coordl_cache(destination);cleanup_coordl_cache(gate)
            report(state)
    if state['status']=='RUNNING':
        state['status']='COMPLETE' if all(j['status']=='DONE' for j in state['jobs']) else 'FINISHED_WITH_FAILURES'
    state['finished']=datetime.datetime.now().astimezone().isoformat();report(state)


if __name__=='__main__':main()
