"""Interleaved ViT repeats and two Swin epoch rechecks; retain prior tables."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import subprocess
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.refresh_evaluation import read
from fastlens.benchmarks.development.historical.refresh_evaluation import rows
from fastlens.benchmarks.development.historical.refresh_evaluation import save
from fastlens.benchmarks.development.historical.refresh_evaluation import write_csv
from fastlens.benchmarks.development.historical.refresh_evaluation import original_data
from fastlens.benchmarks.development.historical.refresh_evaluation import full_swin
from fastlens.benchmarks.execution.revision_results import SNAPSHOT
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.run_phase2_repeats import predictions
from fastlens.benchmarks.system_dct_major.common import parse_manifest

OUT = ROOT/'results/recheck_20260929'
SYSTEMS = ['PyTorch', 'DALI', 'FastLens', 'CoorDL', 'FFCV']


def summarize():
    observations = [read(p) for p in sorted((OUT/'inference').glob('*/trial_*/accepted.json'))]
    if observations:
        write_csv(OUT/'inference_raw.csv', observations, list(observations[0]))
    summaries = []
    previous = rows(OUT/'previous_inference_summary.csv')
    for system in SYSTEMS:
        selected = [r for r in observations if r['pipeline'] == system]
        if not selected:
            continue
        rates = [r['images_per_s'] for r in selected]
        old = next(r for r in previous if r['gpu']=='H100' and r['architecture_family']=='ViT-Ti' and r['pipeline']==system)
        median = statistics.median(rates)
        summaries.append(dict(gpu='H100', model=selected[0]['model'], pipeline=system,
            trials=len(rates), samples_per_trial=50000, batch_size=selected[0]['batch_size'],
            precision=selected[0]['precision'], domain=selected[0]['domain'],
            median_images_per_s=median, min_images_per_s=min(rates), max_images_per_s=max(rates),
            cv_percent=100*statistics.stdev(rates)/statistics.mean(rates) if len(rates)>1 else '',
            previous_median_images_per_s=old['median_images_per_s'],
            change_percent=100*(median/float(old['median_images_per_s'])-1),
            raw_results=';'.join(r['raw_result'] for r in selected),
            notes='Separate interleaved recheck cohort; concurrent host load recorded, not isolated.'))
        if (OUT/'loaded_inference_summary.csv').exists():
            loaded=next(r for r in rows(OUT/'loaded_inference_summary.csv') if r['pipeline']==system)
            summaries[-1].update(loaded_median_images_per_s=loaded['median_images_per_s'],
                change_vs_loaded_percent=100*(median/float(loaded['median_images_per_s'])-1),
                loaded_raw_results=loaded['raw_results'])
    if summaries:
        write_csv(OUT/'inference_summary.csv', summaries, list(summaries[0]))
    epochs = []
    for gpu in ('H100', '4090'):
        p=OUT/'full_epoch'/gpu/'accepted.json'
        if not p.exists():
            continue
        observed=read(p)
        old=read(ROOT/f'results/p1/short_warmup_refresh_20260929/{gpu}/fastlens/accepted.json')
        epoch, prior=observed['epoch'],old['epoch']
        epochs.append(dict(gpu=gpu, model='SwinV2-T DCT', pipeline='FastLens',
            samples=epoch['epoch_samples'], optimizer_updates=epoch['epoch_optimizer_updates'],
            elapsed_s=epoch['epoch_seconds'], images_per_s=epoch['images_per_second'],
            previous_images_per_s=prior['images_per_second'],
            change_percent=100*(epoch['images_per_second']/prior['images_per_second']-1),
            raw_result=observed['raw_result'], previous_raw_result=old['raw_result'],
            notes='Same-code native epoch recheck; CPU control not rerun. Host load recorded.'))
        if (OUT/'loaded_full_epoch_summary.csv').exists():
            loaded=next(r for r in rows(OUT/'loaded_full_epoch_summary.csv') if r['gpu']==gpu)
            epochs[-1].update(loaded_images_per_s=loaded['images_per_s'],
                change_vs_loaded_percent=100*(epoch['images_per_second']/float(loaded['images_per_s'])-1),
                loaded_raw_result=loaded['raw_result'])
    if epochs:
        write_csv(OUT/'full_epoch_summary.csv', epochs, list(epochs[0]))


def inference(template, system, trial):
    output=OUT/'inference'/system/f'trial_{trial}'
    if (output/'accepted.json').exists():
        return 'REUSED'
    output.mkdir(parents=True, exist_ok=False)
    reference=read(template)
    original, expected=original_data(reference)
    metadata=read(template.parent/'run.command.json')
    command=list(metadata['command'])
    contract=read(command[command.index('--contract')+1])
    assert contract['execution']['repeats']==1
    assert contract['execution']['measurement_batches']*contract['execution']['batch_size']==50000
    raw=output/Path(command[command.index('--output')+1]).name
    command[command.index('--output')+1]=str(raw)
    contract_path=output/'contract.json'
    command[command.index('--contract')+1]=str(contract_path)
    if 'coordl' in contract['pipelines']:
        contract['pipelines']['coordl']['file_list']=str(output/'coordl_file_list.txt')
    save(contract_path,contract)
    env=dict(os.environ,**{k:v for k,v in metadata['environment'].items() if v is not None})
    env['FASTLENS_EVALUATION_SNAPSHOT']=str(SNAPSHOT)
    if system=='FFCV':
        paths=[Path(contract['pipelines']['ffcv']['beton'])]
    elif system=='FastLens':
        pipeline=command[command.index('--pipeline')+1]
        paths=[Path(s['fls_path']) for s in parse_manifest(Path(contract['pipelines'][pipeline]['manifest']))['shards']]
    else:
        paths=[Path(s['path']) for s in read(contract['dataset']['sample_manifest'])['samples']]
    run(command,output,env,paths)
    data=read(raw)
    measured=data['repeats'][0]
    assert measured['images']==50000
    assert measured['sample_trace']==expected['sample_trace']
    assert measured['correct_top1']==expected['correct_top1']
    assert predictions(data)==predictions(original)
    assert data['model']['checkpoint_sha256']==original['model']['checkpoint_sha256']
    if system=='FastLens':
        actual=np.load(data['semantic_artifact'])
        previous=np.load(original['semantic_artifact'])
        for name in ('input_0','input_1'):
            delta=np.abs(actual[name]-previous[name])
            assert delta.max()<=.001 and delta.mean()<=.0001
    meta=read(output/'run.command.json')
    assert meta['code_unchanged_during_run']
    elapsed=measured['repeat_scope_seconds']
    save(output/'accepted.json',dict(reference,timestamp=meta['timestamp'],trial=trial,
        elapsed_s=elapsed,images_per_s=50000/elapsed,top1=measured['accuracy_top1']*100,
        top5=measured['accuracy_top5']*100,code_fingerprint=meta['code_fingerprint'],
        raw_result=str(raw.relative_to(ROOT))+'#/repeats/0',
        correctness='PASS: full 50000 IDs/predictions, checkpoint and correct count equal reference; DCT tensor gate; unchanged code.',
        notes='Interleaved H100 ViT recheck. Prior observations retained separately; shared-host load is not isolation.'))
    return 'DONE'


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--epoch-gpus',nargs='+',choices=['H100','4090'],default=['H100','4090'])
    parser.add_argument('--wait-for-pid',type=int)
    args=parser.parse_args()
    OUT=args.output.resolve()
    if args.wait_for_pid and psutil.pid_exists(args.wait_for_pid):
        predecessor=psutil.Process(args.wait_for_pid)
        save(OUT/'status.json',dict(status='WAITING',pid=os.getpid(),waiting_for_pid=predecessor.pid,
            reason='Keep performance runs serial; waiting for current 4090 epoch/controller to exit.'))
        while predecessor.is_running() and predecessor.status()!=psutil.STATUS_ZOMBIE:
            time.sleep(10)
    verify_snapshot()
    summaries=rows(OUT/'previous_inference_summary.csv')
    templates={}
    for system in SYSTEMS:
        summary=next(r for r in summaries if r['gpu']=='H100' and r['architecture_family']=='ViT-Ti' and r['pipeline']==system)
        paths=[ROOT/Path(p.split('#')[0]).parent/'accepted.json' for p in summary['raw_results'].split(';')]
        templates[system]=max((p for p in paths if p.exists()),key=lambda p:read(p)['timestamp'])
    jobs=[]
    for trial in range(1,4):
        order=SYSTEMS[2*(trial-1):]+SYSTEMS[:2*(trial-1)]
        jobs.extend(('inference',system,trial) for system in order)
    jobs.extend(('full_epoch',gpu,1) for gpu in args.epoch_gpus)
    state=dict(status='RUNNING',pid=os.getpid(),total=len(jobs),active=None,completed=[],
        policy='Serial GPUs; interleaved system order; identical saved contracts; concurrent host load recorded; separate cohort, no prior table replacement.')
    for kind,target,trial in jobs:
        state['active']=[kind,target,trial]
        save(OUT/'status.json',state)
        print('START',state['active'],flush=True)
        verify_snapshot()
        try:
            if kind=='inference':
                status=inference(templates[target],target,trial)
            else:
                full_swin(target,OUT/'full_epoch'/target)
                status='DONE'
        except (subprocess.CalledProcessError,AssertionError,RuntimeError) as error:
            status='FAILED: '+repr(error)
        state['completed'].append(dict(experiment=state['active'],status=status))
        save(OUT/'status.json',state)
        summarize()
        print(status,state['active'],flush=True)
    state.update(active=None,status='COMPLETE' if all(r['status'] in ('DONE','REUSED') for r in state['completed']) else 'BLOCKED_CELLS')
    save(OUT/'status.json',state)


if __name__=='__main__':
    main()
