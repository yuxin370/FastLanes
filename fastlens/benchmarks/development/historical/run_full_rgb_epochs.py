"""Sixteen authorized full RGB epochs, sequentially; retain all old results."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot

OUT = ROOT/'results/full_rgb_epochs_20261002'
DATA = Path('/mnt/nvme2/home/tangyuxin/fastlens_ffcv_full_train_20261002')
PYTHON = '/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python'
NAMES = {'pytorch':'PyTorch', 'dali':'DALI', 'coordl':'CoorDL', 'ffcv':'FFCV'}
MODELS = {'swinv2':'SwinV2-T', 'mobilenet24':'MobileNetV2'}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def write_csv(path, rows, fields):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def report(state):
    rows=[]
    for job in state['jobs']:
        if job['status'] != 'DONE':
            continue
        path=Path(job['output'])/'result.json'
        r=read(path)
        rows.append(dict(timestamp=r['timestamp'],git_commit=r['git_commit'],gpu=r['gpu'],
            architecture_family=MODELS[job['model']], model=MODELS[job['model']],domain='RGB',
            pipeline=NAMES[job['pipeline']],comparison_type='deployment',task='training_epoch',
            samples=r['images'],optimizer_updates=r['updates'],elapsed_s=r['seconds'],
            images_per_s=r['images']/r['seconds'],batch_size=r['batch'],effective_batch_size=1024,
            precision=r['precision'],trial=0,trials=1,final_loss=r['final_loss'],
            initial_model_hash=r['initial_model_hash'],sample_order_sha256=r['sample_order_sha256'],
            worker_count=16,cpu_affinity=json.dumps(r['cpu_affinity']),numa_binding=r['numa_policy'],
            correctness='PASS',raw_result=str(path.relative_to(ROOT)),
            notes='One complete epoch; excluded short warmup with restored model/optimizer; final update 143 images; no convergence claim.' + job.get('acceptance_notes', '')))
    if rows:
        write_csv(OUT/'full_epoch_rgb.csv',rows,list(rows[0]))
        write_csv(ROOT/'results/p1/full_epoch_rgb_deployment.csv',rows,list(rows[0]))
    manifest=ROOT/'results/experiment_manifest.csv'
    with manifest.open() as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; existing=list(reader)
    prefix='full_rgb_epochs_20261002:'
    existing=[r for r in existing if not r['experiment_id'].startswith(prefix)]
    for job in state['jobs']:
        row=dict.fromkeys(fields,'')
        row.update(experiment_id=prefix+':'.join(job[k] for k in ('gpu','model','pipeline')),
            phase='P4-RGB-full',task='training_epoch',gpu=job['gpu'],architecture=MODELS[job['model']],
            rgb_model=MODELS[job['model']],model_variant=job['model'],domain='RGB',pipeline=NAMES[job['pipeline']],
            sample_count=1281167,precision='bf16-autocast',batch_size=64,status=job['status'],
            result_file=str(Path(job['output']).relative_to(ROOT)/'result.json'),
            correctness='PASS' if job['status']=='DONE' else '',notes=job.get('error',job.get('acceptance_notes','One measured complete epoch; effective batch 1024; final update 143.')))
        existing.append(row)
    write_csv(manifest,existing,fields)
    lines=['# Full RGB deployment epochs','',f"Status: {state['status']}",'',
           'One measured epoch per configuration; 1,281,167 samples; 1,252 updates. CPU/memory node 0; GPUs sequential.', '',
           '| GPU | Model | PyTorch | DALI | CoorDL | FFCV |', '|---|---|---:|---:|---:|---:|']
    for gpu in ('4090','H100'):
        for model in MODELS:
            cells=[]
            for pipeline in NAMES:
                job=next(j for j in state['jobs'] if (j['gpu'],j['model'],j['pipeline'])==(gpu,model,pipeline))
                if job['status']=='DONE':
                    r=read(Path(job['output'])/'result.json'); cells.append(f"{r['images']/r['seconds']:.1f}")
                else:
                    cells.append(job['status'])
            lines.append('| '+' | '.join([gpu,MODELS[model],*cells])+' |')
    lines+=['','Units: images/s. Unfinished cells contain no estimated throughput.',
            'Historical FastLens full epochs remain in results/p1/full_epoch_training.csv; no historical result is replaced.']
    (OUT/'STATUS.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'status.json',state)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'status.json').exists():
        raise FileExistsError('Existing campaign retained; inspect its state before resuming')
    for pipeline in NAMES:
        tail=read(OUT/'tail_checks'/pipeline/'result.json')
        if not (tail['passed'] and tail['samples']==143 and tail['batch_sizes']==[64,64,15]):
            raise ValueError('Reader tail correctness check failed: '+pipeline)
    paths=[ROOT / name for name in ('fastlens/benchmarks/development/historical/run_full_rgb_epochs.py', 'fastlens/benchmarks/paper/shared/training_window_rgb.py', 'fastlens/benchmarks/development/historical/p0_adapters.py', 'fastlens/benchmarks/development/historical/training_ffcv_crop.py', 'fastlens/benchmarks/execution/resource_guard.py', 'fastlens/benchmarks/execution/run_logged.py', 'fastlens/benchmarks/development/diagnostics/check_rgb_epoch_tail.py')]
    for directory in ('fastlens/benchmarks/baselines/system_rgbnomore/training','fastlens/benchmarks/training_pls','fastlens/benchmarks/dct_models'):
        paths += list((ROOT/directory).glob('*.py'))
    paths += [Path('/home/tangyuxin/RGB-no-more/models')/f'{name}.py' for name in ('plainvit','swinv2')]
    files={str(p):file_sha(p) for p in sorted(set(paths))}
    save(OUT/'runtime_snapshot.json',dict(files=files,fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()))
    os.environ['FASTLENS_EVALUATION_SNAPSHOT']=str(OUT/'runtime_snapshot.json')
    state=dict(status='PREPARING_FFCV',pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),jobs=[])
    devices={x['requested']:x['selection'] for x in read(ROOT/'results/inventory/device_mapping.json')}
    for gpu in ('4090','H100'):
        for model in MODELS:
            for pipeline in NAMES:
                destination=OUT/'epochs'/gpu/model/pipeline
                state['jobs'].append(dict(gpu=gpu,model=model,pipeline=pipeline,output=str(destination),status='TODO'))
    report(state)
    # Conversion precedes performance measurements to avoid shared-storage contention.
    meta=OUT/'preparation/conversion.command.json'
    while not meta.exists() or 'returncode' not in read(meta):
        time.sleep(10)
    if read(meta)['returncode'] != 0:
        state['status']='BLOCKED_CONVERSION';report(state)
        raise RuntimeError('FFCV conversion failed; see preparation/conversion.log')
    subprocess.run([PYTHON,'fastlens/benchmarks/execution/run_logged.py','--log',str(OUT/'preparation/verification.log'),'--',
        PYTHON,'fastlens/benchmarks/baselines/ffcv/verify_p0_ffcv.py','--output',str(DATA)],check=True,cwd=ROOT)
    if not read(DATA/'verification.json')['passed']:
        raise ValueError('FFCV verification failed')
    # The full raw representation slightly exceeds one NUMA node. Populate its
    # page cache without a strict memory-node restriction; benchmark processes
    # themselves retain the same node-0 policy as the existing measurements.
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    save(OUT/'preparation/warm_cache.json',cache_pages([DATA/'train_window.beton'],warm=True))
    state['status']='RUNNING';report(state)
    initials={}; order=None
    for job in state['jobs']:
        destination=Path(job['output']); destination.mkdir(parents=True)
        gate=OUT/'checks'/job['gpu']/job['model']/job['pipeline'];gate.mkdir(parents=True)
        interpreter=str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if job['pipeline']=='coordl' else PYTHON
        base=['numactl','--cpunodebind=0','--membind=0',interpreter,'fastlens/benchmarks/paper/shared/training_window_rgb.py',
            '--gpu',job['gpu'],'--model',job['model'],'--pipeline',job['pipeline'],'--trial','0']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[job['gpu']],DCTNET_PROFILE=job['model'],
            TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
            PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        job['status']='RUNNING';report(state)
        try:
            verify_snapshot()
            run([*base,'--check','--output',str(gate)],gate,env,None)
            check=read(gate/'result.json')
            if not check['correctness_passed']:
                raise ValueError('Small input/update check failed')
            key=job['model']
            config=(check['initial_model_hash'],check['precision'],check['batch'],check['accumulation'])
            if initials.setdefault(key,config)!=config:
                raise ValueError('RGB configuration mismatch across systems/GPUs')
            command=[*base,'--full-epoch','--ffcv-root',str(DATA),'--correctness-report',str(gate/'result.json'),'--output',str(destination)]
            run(command,destination,env,None)
            r=read(destination/'result.json')
            if not (r['correctness_passed'] and r['full_epoch'] and r['images']==1281167 and r['updates']==1252
                    and r['loss_finite'] and r['warmup_initial_state_restored'] and r['final_update_samples']==143
                    and r['parameter_update_verified'] and read(destination/'run.command.json')['code_unchanged_during_run']):
                raise ValueError('Full epoch acceptance failed')
            if order is not None and order!=r['sample_order_sha256']:
                raise ValueError('Sample order differs from earlier RGB full epochs')
            order=r['sample_order_sha256']
            save(destination/'accepted.json',dict(correctness=True,comparison_type='deployment',raw_result=str(destination/'result.json')))
            job['status']='DONE'
        except (RuntimeError, ValueError, subprocess.CalledProcessError, FileNotFoundError) as error:
            job.update(status='FAILED',error=str(error))
            save(destination/'failure.json',dict(error=str(error)))
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


if __name__=='__main__':
    main()
