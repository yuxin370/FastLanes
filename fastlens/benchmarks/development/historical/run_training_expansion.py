"""Execute the audited missing training windows serially, retaining failed attempts."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.development.historical.training_expansion_plan import plan
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.execution.inference_load import run


def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')


def snapshot(out):
    paths=set()
    for directory in ('fastlens/torch','fastlens/benchmarks/training_pls','fastlens/benchmarks/baselines/system_rgbnomore/training',
                      'fastlens/benchmarks/dct_models/DCTNet','fastlens/benchmarks/dct_models/FUN'):
        paths.update(p for p in (ROOT/directory).rglob('*.py') if 'tests' not in p.parts)
    paths.update(ROOT/'fastlens/benchmarks/dct_models'/f'{name}.py' for name in
                 ('backend','efun_backend','rgb','training_pls','training_jpeg','dct_geometry','capture','efun'))
    paths.update(Path('/home/tangyuxin/RGB-no-more')/name for name in
                 ('models/plainvit.py','models/swinv2.py','utils/dct_ops.py'))
    paths.update(ROOT / name for name in ('fastlens/benchmarks/paper/shared/training_window_rgb.py', 'fastlens/benchmarks/development/historical/p0_adapters.py', 'fastlens/benchmarks/development/historical/training_ffcv_crop.py', 'fastlens/benchmarks/development/diagnostics/check_training_update.py', 'fastlens/benchmarks/development/diagnostics/check_mobile_training.py', 'fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py', 'fastlens/benchmarks/development/historical/full_epoch_short_warmup.py', 'fastlens/benchmarks/execution/run_logged.py', 'fastlens/benchmarks/execution/resource_guard.py'))
    native=Path(read(out/'handoff.json')['native_path']);paths.add(native)
    files={str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):file_sha(p) for p in sorted(paths)}
    result=dict(files=files,native_path=str(native.relative_to(ROOT)),native_sha256=file_sha(native),
        fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    save(out/'runtime_snapshot.json',result)
    (out/'source_changes.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',choices=['4090','H100'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-resident-pid',type=int,action='append',default=[],
                   help='Explicitly authorized idle GPU resident; active use still blocks')
    args=p.parse_args();out=args.output.resolve()
    os.environ['FASTLENS_ALLOWED_RESIDENT_PIDS']=','.join(map(str,args.allow_resident_pid))
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handoff=read(ROOT/'results/remaining_inference_recheck_20260929/status.json')
    if handoff['status']!='COMPLETE' or handoff.get('paper_updated') is not True:
        raise RuntimeError('Original inference/paper handoff incomplete')
    import psutil
    others=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        argv=proc.info['cmdline'] or []
        if any(Path(a).name in ('run_training_expansion.py','recheck_remaining_inference.py',
            'recheck_latest_inference.py','training_window_rgb.py') for a in argv[1:]):
            others.append(proc.pid)
    if others:raise RuntimeError(f'Another benchmark/controller is running: {others}')
    queue=read(out/'plan.json') if (out/'plan.json').exists() else plan(ROOT,out)
    # The earlier ABI-failed diagnostic remains at its original path.
    fixed=out/'checks/4090/vitti/pytorch_abi_fixed/result.json'
    if fixed.exists():
        for job in queue['jobs']:
            if job['phase']=='window' and job['gpu']=='4090' and job['model']=='vitti' and job['pipeline']=='pytorch':
                job['check_command'][job['check_command'].index('--output')+1]=str(fixed.parent)
                job['command'][job['command'].index('--correctness-report')+1]=str(fixed)
    if not (out/'plan.json').exists():save(out/'plan.json',queue)
    else:queue=read(out/'plan.json')
    if not (out/'runtime_snapshot.json').exists():snapshot(out)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT']=str(out/'runtime_snapshot.json')
    fingerprint=verify_snapshot()
    state=dict(status='RUNNING',pid=os.getpid(),gpu=args.gpu,started=datetime.datetime.now().astimezone().isoformat(),
        active=None,completed=[],failures=[],window_total=sum(j['gpu']==args.gpu and j['phase']=='window' for j in queue['jobs']),
        fingerprint=fingerprint,allowed_resident_pids=args.allow_resident_pid,policy='Windows first across both GPUs; no automatic repeats; no extra epochs.')
    save(out/'status.json',state)
    blocked=set();initials={}
    def launch(command,destination,env):
        destination.mkdir(parents=True,exist_ok=True)
        if (destination/'run.log').exists():raise RuntimeError(f'Existing unfinished attempt retained: {destination}')
        try:
            run(command,destination,env,None)
        finally:
            if '--pipeline' in command and command[command.index('--pipeline')+1]=='coordl':
                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
                cleanup_coordl_cache(destination)
    for job in queue['jobs']:
        if job['gpu']!=args.gpu or job['phase']!='window':continue
        key=(job['model'],job['pipeline'])
        destination=Path(job['output'])
        state['active']=dict(model=job['model'],pipeline=job['pipeline'],trial=job['trial'])
        save(out/'status.json',state)
        if (destination/'accepted.json').exists():state['completed'].append(str(destination));continue
        if key in blocked:continue
        env=dict(os.environ,**job['environment'],FASTLENS_TRAINING_NATIVE_DIR=str(out/'native_runtime'),
                 PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env['PYTHONPATH']=f'{ROOT}:{out}/native_runtime'
        try:
            verify_snapshot()
            if job['pipeline']=='imagelanes':
                matched=next(j for j in queue['jobs'] if j['phase']=='epoch' and j['gpu']==args.gpu and j['model']==job['model'])
                gate=Path(matched['check_command'][-1])
                if not gate.exists():launch(matched['check_command'],gate.parent/'matched_check',env)
                if not read(gate)['passed']:raise ValueError('Matched tensor/update gate failed')
            gate=Path(job['check_command'][job['check_command'].index('--output')+1])/'result.json'
            if not gate.exists():launch(job['check_command'],gate.parent,env)
            check=read(gate)
            if not check['correctness_passed']:raise ValueError('Window correctness gate failed')
            if job['pipeline']!='imagelanes':
                value=(check['initial_model_hash'],check['precision'],check['batch'],check['accumulation'])
                if job.get('reference_correctness_report'):
                    reference=read(job['reference_correctness_report'])
                    if value!=tuple(reference[k] for k in ('initial_model_hash','precision','batch','accumulation')):
                        raise ValueError('RGB configuration differs from the retained PyTorch baseline')
                    if check['sample_order_sha256']!=reference['sample_order_sha256']:
                        raise ValueError('RGB sample order differs from the retained PyTorch baseline')
                previous=initials.setdefault(job['model'],value)
                if previous!=value:raise ValueError('RGB family initialization/precision/batch differs between systems')
            launch(job['command'],destination,env)
            result=read(destination/'result.json')
            if not (result['correctness_passed'] and result['images']==65536 and result['updates']==64
                    and result['parameter_update_verified'] and result['loss_finite'] and result['mode']=='training'):
                raise ValueError('Window acceptance failed')
            meta=read(destination/'run.command.json')
            if not meta['code_unchanged_during_run']:raise ValueError('Runtime changed during measurement')
            save(destination/'accepted.json',dict(correctness_passed=True,comparison_type='deployment',
                raw_result=str((destination/'result.json').relative_to(ROOT)),code_fingerprint=fingerprint,
                correctness_report=str(gate),native_sha256=read(out/'handoff.json')['native_sha256']))
            state['completed'].append(str(destination))
        except (subprocess.CalledProcessError,RuntimeError,ValueError,FileNotFoundError) as error:
            destination.mkdir(parents=True,exist_ok=True)
            failure=dict(status='BLOCKED',model=job['model'],pipeline=job['pipeline'],trial=job['trial'],error=repr(error))
            save(destination/'failure.json',failure);state['failures'].append(failure);blocked.add(key)
            # Code changes invalidate the runtime gate for every subsequent job.
            if 'Evaluation code changed' in str(error):state['status']='BLOCKED_CODE_CHANGE';break
        finally:
            save(out/'status.json',state)
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/report_training_expansion.py','--output',str(out)],cwd=ROOT,check=True)
    if state['status']=='RUNNING':state['status']='WINDOWS_FINISHED_WITH_BLOCKED_CELLS' if state['failures'] else 'WINDOWS_COMPLETE_WAITING_OTHER_GPU'
    state['active']=None;state['finished']=datetime.datetime.now().astimezone().isoformat();save(out/'status.json',state)
    subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/report_training_expansion.py','--output',str(out)],cwd=ROOT,check=True)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
