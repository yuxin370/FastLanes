"""Read-only reuse audit and concrete queue for the authorized training expansion.

No measurements, cache warming, data conversion, table updates or paper writes.
Run from the isolated checkout with --source pointing at the active repository.
"""
import argparse
import csv
import json
import math
from pathlib import Path

PROFILES=('vitti','swinv2','mobilenet24','mobilenet32','resnet24','resnet64','efun')
RGB_PROFILES=('vitti','swinv2','mobilenet24','resnet24','efun')
PYTHON='/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python'


def read(path):return json.loads(path.read_text())
def rows(path):return list(csv.DictReader(path.open()))


def plan(source, output):
    windows=rows(source/'results/p0/training_window_raw.csv')
    epochs=rows(source/'results/p1/full_epoch_training.csv')
    reused_windows=[]
    for row in windows:
        if row['pipeline']!='FastLens':continue
        raw=read(source/row['raw_result'])
        if not (raw['correctness_passed'] and raw['images']==65536 and raw['updates']==64
                and raw['batch']*raw['accumulation']==1024 and math.isfinite(raw['training_loss'])):
            raise ValueError('Invalid existing native window: '+row['raw_result'])
        accepted=read((source/row['raw_result']).parent/'accepted.json')
        if not accepted['correctness_passed']:raise ValueError(row['raw_result'])
        reused_windows.append(row)
    reused_epochs=[]
    digests=set()
    for row in epochs:
        path=source/row['raw_result']
        if path.suffix=='.json':
            raw=read(path);epoch,=raw['epochs']
            valid=(epoch['full_epoch'] and epoch['samples']==epoch['unique_samples']==1281167
                   and epoch['optimizer_updates']==1252 and math.isfinite(epoch['final_loss'])
                   and raw['warmup']['initial_state_restored']
                   and raw['initial_model_hash']==raw['warmup']['initial_model_hash'])
        else:
            epoch,=[json.loads(line) for line in path.read_text().splitlines() if json.loads(line).get('scope')=='epoch']
            valid=(epoch['epoch_samples']==1281167 and epoch['epoch_optimizer_updates']==1252
                   and math.isfinite(epoch['train_loss']))
            pause=read(path.parent/'pause_result.json')
            valid &= all(pause['integration_checks'][k] for k in ('coverage_counters_valid','loss_finite','gradients_finite','parameters_finite'))
        if not valid:raise ValueError('Invalid existing epoch: '+str(path))
        digests.add(epoch['sample_order_digest']);reused_epochs.append(row)
    if len(digests)!=1:raise ValueError('Existing paired epoch sample order differs')
    jobs=[]
    prefix=['numactl','--cpunodebind=0','--membind=0']
    for gpu in ('4090','H100'):
        for profile in RGB_PROFILES:
            for pipeline in ('pytorch','dali','coordl','ffcv'):
                interpreter=str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if pipeline=='coordl' else PYTHON
                gate=output/'checks'/gpu/profile/pipeline
                check=[*prefix,interpreter,'fastlens/benchmarks/paper/shared/training_window_rgb.py','--gpu',gpu,'--model',profile,
                       '--pipeline',pipeline,'--trial','0','--output',str(gate),'--check']
                for trial in range(3):
                    dest=output/'windows'/gpu/profile/pipeline/f'trial_{trial}'
                    command=[*prefix,interpreter,'fastlens/benchmarks/paper/shared/training_window_rgb.py','--gpu',gpu,'--model',profile,
                        '--pipeline',pipeline,'--trial',str(trial),'--output',str(dest),
                        '--correctness-report',str(gate/'result.json')]
                    jobs.append(dict(phase='window',gpu=gpu,model=profile,pipeline=pipeline,trial=trial,
                        check_command=check,command=command,output=str(dest)))
        for profile in PROFILES:
            if profile in ('swinv2','mobilenet24'):continue
            gate=output/'checks'/gpu/profile/'imagelanes'
            check=[*prefix,PYTHON,'fastlens/benchmarks/paper/shared/training_window_rgb.py','--gpu',gpu,'--model',profile,
                   '--pipeline','imagelanes','--trial','0','--output',str(gate),'--check']
            for trial in range(3):
                dest=output/'windows'/gpu/profile/'imagelanes'/f'trial_{trial}'
                jobs.append(dict(phase='window',gpu=gpu,model=profile,pipeline='imagelanes',trial=trial,
                    check_command=check,command=[*prefix,PYTHON,'fastlens/benchmarks/paper/shared/training_window_rgb.py',
                    '--gpu',gpu,'--model',profile,'--pipeline','imagelanes','--trial',str(trial),
                    '--output',str(dest),'--correctness-report',str(gate/'result.json')],output=str(dest)))
    physical=source/'fastlens/data/compressed/imagenet512_train_block_major_premixed'
    material=read(physical/'materialization_contract.json')
    manifests=source/'fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3'
    for gpu in ('4090','H100'):
        for profile in PROFILES:
            if profile in ('swinv2','mobilenet24'):continue
            gate=output/'checks'/gpu/profile/'matched.json'
            check=[*prefix,PYTHON,'fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py','--model','vitti','--epoch','0'] if profile=='vitti' else [*prefix,PYTHON,'fastlens/benchmarks/development/diagnostics/check_mobile_training.py']
            check+=['--output',str(gate)]
            for pipeline in ('imagelanes','cpu_dct'):
                dest=output/'epochs'/gpu/profile/pipeline
                if profile=='vitti':
                    command=[*prefix,PYTHON,'-m','fastlens.benchmarks.training_pls.run_matrix',
                        '--train-manifest',str(manifests/'train.json'),'--val-manifest',str(manifests/'val.json'),
                        '--layout-plan',str(source/'results/preparation/training_layout/physical_layout_plan.json'),
                        '--conditions','B6','--seeds','11997733','--model','rgbnomore-vitti-dct-224-v1',
                        '--epochs','300','--stop-after-epoch','1','--execution-backend',
                        'native-physical-pls' if pipeline=='imagelanes' else 'standard-rgbnomore-dct',
                        '--premixed-mapping-csv',material['ordered_mapping'],
                        '--expected-mapping-sha256',material['ordered_mapping_sha256'],
                        '--workers','8','--output-dir',str(dest),'--device','cuda:0','--required-gpu-name-substring',gpu]
                    if pipeline=='imagelanes':command+=['--physical-fastlens-manifest',str(physical/'dct/manifest.bin')]
                else:
                    module=['fastlens.benchmarks.dct_models.efun','training_pls'] if profile=='efun' else ['fastlens.benchmarks.dct_models.training_pls']
                    command=[*prefix,PYTHON,'-m',*module,'--manifest',str(physical/'dct/manifest.bin'),
                        '--mapping',material['ordered_mapping'],'--mapping-sha256',material['ordered_mapping_sha256'],
                        '--condition','B6','--input-backend','native' if pipeline=='imagelanes' else 'jpeg',
                        '--epochs','1','--short-warmup','--validation-count','0','--workers','8',
                        '--transform-blocks-per-launch','32768','--transform-ctas-per-launch','512',
                        '--compile-mode','default','--output-dir',str(dest)]
                jobs.append(dict(phase='epoch',gpu=gpu,model=profile,pipeline=pipeline,trial=0,
                    check_command=check,command=command,output=str(dest),
                    command_role='prepare_only_then_full_epoch_short_warmup_from_generated_argv' if profile=='vitti' else 'measure'))
    devices={d['requested']:d['selection'] for d in read(source/'results/inventory/device_mapping.json')}
    for job in jobs:
        job['environment']={
            'CUDA_VISIBLE_DEVICES':devices[job['gpu']], 'DCTNET_PROFILE':job['model'],
            'TMPDIR':str(Path.home()/'tmp'), 'PYTHONPATH':f'{source}:{source}/build/fastlens/torch',
            'OMP_NUM_THREADS':'8', 'MKL_NUM_THREADS':'8', 'OPENBLAS_NUM_THREADS':'8', 'FASTLENS_NSYS_FINE':'0'}
    inventory=read(source/'results/inventory/models.json')
    pairs=rows(source/'results/inventory/workload_pairs.csv')
    return dict(status='ADAPTED_GPU_VALIDATION_PENDING',inference_handoff=read(source/'results/remaining_inference_recheck_20260929/status.json'),
        source=str(source),output=str(output),discovered_model_ids=[m['id'] for m in inventory['models']],
        unsupported_variants=[p for p in pairs if p['relationship']=='unpaired_training_constructor'],
        reused_windows=reused_windows,reused_epochs=reused_epochs,
        replaced_rgb_windows=[r for r in windows if r['pipeline']!='FastLens'],
        replacement_reason='Unified keyed sample/crop/flip order and bilinear resize; old DALI shuffled, FFCV AREA and .08 crop scale, CoorDL native .08 crop scale, PyTorch PIL antialias. Old RGB runs retained, not pooled.',
        jobs=jobs,window_trials=sum(j['phase']=='window' for j in jobs),full_epochs=sum(j['phase']=='epoch' for j in jobs),
        policy='GPU correctness first, then all missing windows, then missing single full epochs; GPUs serial; no automatic repeats; existing native cohorts explicitly historical; no convergence claims')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args()
    result=plan(args.source,args.output)
    with args.plan.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','window_trials','full_epochs')}))


if __name__=='__main__':main()
