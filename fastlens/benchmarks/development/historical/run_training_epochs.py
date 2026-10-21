"""Run only the twenty missing single matched epochs after both window queues finish."""
import argparse
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.development.historical.run_training_expansion import snapshot
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.execution.inference_load import run


def epoch_result(job,population):
    out=Path(job['output'])
    if job['model'] in ('vitti','swinv2'):
        directory=out/'runs/B6/seed_11997733'
        raw=directory/'metrics.jsonl'
        epoch,=[json.loads(l) for l in raw.read_text().splitlines() if json.loads(l).get('scope')=='epoch']
        final=read(directory/'pause_result.json');warm=read(directory.parent/'short_warmup.json')
        calibration=read(directory/'model_only_dct.json')
        initial=calibration['initial_model_hash']
        assert calibration['microbatch_images']==64 and calibration['gradient_accumulation']==16
        assert all(final['integration_checks'][k] for k in ('coverage_counters_valid','loss_finite','gradients_finite','parameters_finite'))
        assert epoch['epoch_samples']==population and epoch['epoch_optimizer_updates']==math.ceil(population/1024)
        result=dict(samples=epoch['epoch_samples'],optimizer_updates=epoch['epoch_optimizer_updates'],elapsed_s=epoch['epoch_seconds'],
            final_loss=epoch['final_loss'],training_loss=epoch['train_loss'],initial_model_hash=initial,precision=final['precision'],
            sample_order_digest=epoch['sample_order_digest'],matching_digests={k:v for k,v in epoch.items() if k.endswith('_digest')})
    else:
        raw=out/'training.json';final=read(raw);epoch,=final['epochs'];warm=final['warmup'];initial=final['initial_model_hash']
        assert final['full_training_epochs_completed']==1 and epoch['full_epoch']
        assert epoch['samples']==epoch['unique_samples']==population and epoch['optimizer_updates']==math.ceil(population/1024)
        assert final['microbatch']==64 and final['accumulation']==16 and not final['data_only']
        result=dict(samples=epoch['samples'],optimizer_updates=epoch['optimizer_updates'],elapsed_s=epoch['seconds'],
            final_loss=epoch['final_loss'],training_loss=epoch['ce'],initial_model_hash=initial,precision=final['precision'],
            sample_order_digest=epoch['sample_order_digest'],matching_digests=dict(sample_order_digest=epoch['sample_order_digest'],pool_images=[p['images'] for p in epoch['pools']]))
    assert warm['initial_state_restored'] and warm['initial_model_hash']==initial
    assert warm['cache_before']['resident_pages']==warm['cache_before']['pages']
    assert all(math.isfinite(result[k]) for k in ('elapsed_s','final_loss','training_loss')) and result['elapsed_s']>0
    return dict(result,raw_result=str(raw.relative_to(ROOT)),images_per_s=population/result['elapsed_s'],loss_finite=True,
                warmup_file=str((out/'short_warmup.json') if job['model'] not in ('vitti','swinv2') else directory.parent/'short_warmup.json'))


def publish_pair(jobs,out):
    values=[read(Path(j['output'])/'accepted.json') for j in jobs]
    for field in ('initial_model_hash','precision','sample_order_digest','samples','optimizer_updates','code_fingerprint'):
        if values[0][field]!=values[1][field]:raise ValueError(f'Matched epoch mismatch: {field}')
    pair_report = None
    if jobs[0]['model'] in ('vitti', 'swinv2'):
        from fastlens.benchmarks.development.diagnostics.check_epoch_pair import verify_transformer_pair
        evidence = verify_transformer_pair(jobs, values)
        pair_report = Path(jobs[0]['output']).parent/'pair_verification.json'
        save(pair_report, evidence)
    elif values[0]['matching_digests'] != values[1]['matching_digests']:
        raise ValueError('Matched epoch mismatch: matching_digests')
    records=[]
    names={'vitti':'ViT-Ti DCT','mobilenet32':'MobileNetV2-DCT32','resnet24':'ResNet-50-DCT24','resnet64':'ResNet-50-DCT64','efun':'eFUN-DCT'}
    for job,value in zip(jobs,values):
        meta=read(Path(job['output'])/'run.command.json')
        record=dict(timestamp=meta['timestamp'],git_commit=meta['git_commit'],code_fingerprint=meta['code_fingerprint'],
            gpu='RTX 4090' if job['gpu']=='4090' else 'H100',model=names[job['model']],model_variant=job['model'],
            domain='DCT',pipeline='FastLens' if job['pipeline']=='imagelanes' else 'CPU-DCT',task='training_epoch',
            comparison_type='matched_dct',precision=value['precision'],batch_size=64,effective_batch_size=1024,trial=0,trials=1,
            samples=value['samples'],consumed_samples=value['samples'],optimizer_updates=value['optimizer_updates'],
            elapsed_s=value['elapsed_s'],images_per_s=value['images_per_s'],final_loss=value['final_loss'],training_loss=value['training_loss'],
            loss_finite=True,initial_model_hash=value['initial_model_hash'],sample_order_digest=value['sample_order_digest'],
            raw_result=value['raw_result'],source_log=str((Path(job['output'])/'run.log').relative_to(ROOT)),command=meta['shell_command'],
            worker_count=8 if job['pipeline']=='cpu_dct' else '',cpu_affinity='',numa_binding='numactl --cpunodebind=0 --membind=0 (command)',
            initialization='Existing seeded scratch training recipe; restored after excluded short warmup',
            augmentation_configuration='B6 closed-pool DCT crop, RandAugment and Mixup; matched recipe',
            correctness_status=('PASS: observed order/pools, matched contract and logical replay; independent sampled tensor/update check'
                if pair_report else 'PASS: full coverage, updates, finite values, initialization and pair digests; independent tensor/update check'),
            correctness_report=str(pair_report) if pair_report else job['correctness_report'],strict_matched=True,valid=True,implementation_cohort=out.name,measurement_cohort=out.name,
            concurrent_load_notes=str((Path(job['output'])/'concurrent_load.jsonl').relative_to(ROOT)),
            notes='One measured complete epoch per configuration, separate short warmup; no convergence claim; shared host/GPU conditions in raw logs.')
        records.append(record)
    for path in [ROOT/'results/p1/full_epoch_training.csv',ROOT/'results/p1/full_epoch_training_all_versions.csv',out/'epoch_raw.csv']:
        if path.exists():
            with path.open() as f:existing=list(csv.DictReader(f))
        else:existing=[]
        known={r['raw_result'] for r in existing};new_records=[r for r in records if r['raw_result'] not in known]
        if not new_records:continue
        existing.extend(new_records)
        fields=list(dict.fromkeys(k for r in existing for k in r))
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(existing)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publish-existing', action='store_true', help='Verify and publish accepted pairs only; never launch training')
    args=parser.parse_args();out=args.output.resolve()
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.publish_existing:
        jobs=read(out/'plan.json')['jobs']
        for gpu,model in dict.fromkeys((j['gpu'],j['model']) for j in jobs):
            pair=[j for j in jobs if (j['gpu'],j['model'])==(gpu,model)]
            if len(pair)==2 and all((Path(j['output'])/'accepted.json').exists() for j in pair):
                for job in pair:
                    accepted=read(Path(job['output'])/'accepted.json')
                    raw=epoch_result(job,accepted['samples'])
                    if any(accepted[k]!=v for k,v in raw.items()):
                        raise ValueError(f'Accepted epoch differs from raw result: {job["output"]}')
                    meta=read(Path(job['output'])/'run.command.json')
                    if meta['returncode']!=0 or not meta['code_unchanged_during_run'] or meta['code_fingerprint']!=accepted['code_fingerprint']:
                        raise ValueError(f'Invalid saved execution metadata: {job["output"]}')
                publish_pair(pair,out)
                print(f'Published existing pair: {gpu}/{model}',flush=True)
        return
    import psutil
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid==os.getpid():continue
        if any(Path(a).name in ('run_training_epochs.py','run_training_expansion.py','training_window_rgb.py','full_epoch_short_warmup.py') for a in (p.info['cmdline'] or [])[1:]):
            raise RuntimeError(f'Another training process is active: {p.pid}')
    jobs=read(out/'plan.json')['jobs'];population=read(ROOT/'fastlens/data/compressed/imagenet512_train_block_major_premixed/materialization_contract.json')['images']
    if not (out/'runtime_snapshot.json').exists():snapshot(out)
    os.environ['FASTLENS_EVALUATION_SNAPSHOT']=str(out/'runtime_snapshot.json');fingerprint=verify_snapshot()
    state=dict(status='RUNNING',pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),active=None,completed=[],failures=[],epoch_total=len(jobs),fingerprint=fingerprint)
    save(out/'status.json',state)
    def stop(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    for job in jobs:
        destination=Path(job['output']);state['active']={k:job[k] for k in ('gpu','model','pipeline')};save(out/'status.json',state)
        if (destination/'accepted.json').exists():raise FileExistsError('Do not repeat a measured epoch')
        destination.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,**job['environment'],FASTLENS_TRAINING_NATIVE_DIR=str(out/'native_runtime'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env['PYTHONPATH']=f'{ROOT}:{out}/native_runtime'
        os.environ['FASTLENS_ALLOWED_RESIDENT_PIDS']=job['environment'].get('FASTLENS_ALLOWED_RESIDENT_PIDS','')
        env['FASTLENS_ALLOWED_RESIDENT_PIDS']=os.environ['FASTLENS_ALLOWED_RESIDENT_PIDS']
        try:
            verify_snapshot();assert read(job['correctness_report'])['passed']
            if job['gpu']=='H100' and not (out/'h100_idle_ready.json').exists():
                since=None
                with (out/'h100_idle_samples.jsonl').open('x') as stream:
                    while True:
                        raw=subprocess.check_output(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],'--query-gpu=utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],text=True)
                        values=list(map(int,raw.strip().split(',')));now=time.monotonic()
                        since=(now if since is None else since) if values==[0,0] else None
                        row=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),utilization=values,idle_s=0 if since is None else now-since)
                        stream.write(json.dumps(row)+'\n');stream.flush()
                        if row['idle_s']>=300:save(out/'h100_idle_ready.json',row);break
                        time.sleep(5)
            command=list(job['command'])
            if job['command_role']!='measure':
                assert '--execute' not in command and command[command.index('--stop-after-epoch')+1]=='1'
                command+=['--fastlens-torch-module-path',str(out/'native_runtime')]
                prep=destination/'prepare';prep.mkdir();run(command,prep,env,None)
                generated,=read(destination/'matrix_execution_plan.json')['commands']
                argv=generated['argv'];assert argv[1:3]==['-m','fastlens.benchmarks.training_pls.train']
                # Matrix preparation creates run metadata and adds --resume; no checkpoint exists here.
                argv.remove('--resume')
                command=['numactl','--cpunodebind=0','--membind=0',argv[0],'fastlens/benchmarks/development/historical/full_epoch_short_warmup.py',*argv[3:]]
            else:
                assert command[command.index('--epochs')+1]=='1' and '--short-warmup' in command
            run(command,destination,env,None)
            record=epoch_result(job,population);meta=read(destination/'run.command.json');assert meta['code_unchanged_during_run']
            save(destination/'accepted.json',dict(record,code_fingerprint=fingerprint,correctness_report=job['correctness_report']))
            state['completed'].append(str(destination))
            pair=[j for j in jobs if (j['gpu'],j['model'])==(job['gpu'],job['model'])]
            if len(pair)==2 and all((Path(j['output'])/'accepted.json').exists() for j in pair):publish_pair(pair,out)
        except (AssertionError,RuntimeError,ValueError,FileNotFoundError,subprocess.CalledProcessError) as error:
            failure=dict(**state['active'],error=repr(error));save(destination/'failure.json',failure);state['failures'].append(failure)
            if 'Evaluation code changed' in str(error):state['status']='BLOCKED_CODE_CHANGE';break
        except BaseException as error:
            # Preserve the interrupted job and re-raise after the monitor cleans up.
            state['status']='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED_CONTROLLER'
            state['failures'].append(dict(**state['active'],error=repr(error)))
            state['interrupted_active']=state['active'];state['active']=None
            state['finished']=datetime.datetime.now().astimezone().isoformat()
            raise
        finally:save(out/'status.json',state)
    if state['status']=='RUNNING':state['status']='COMPLETE' if not state['failures'] else 'FINISHED_WITH_BLOCKED_CELLS'
    state['active']=None;state['finished']=datetime.datetime.now().astimezone().isoformat();save(out/'status.json',state)


if __name__=='__main__':main()
