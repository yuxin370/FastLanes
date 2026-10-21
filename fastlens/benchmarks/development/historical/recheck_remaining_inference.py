"""Serial interleaved inference remeasurement; retain every prior observation."""
import argparse
from collections import defaultdict
import datetime
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.refresh_evaluation import read
from fastlens.benchmarks.development.historical.refresh_evaluation import rows
from fastlens.benchmarks.development.historical.refresh_evaluation import save
from fastlens.benchmarks.development.historical.refresh_evaluation import write_csv
from fastlens.benchmarks.development.historical.refresh_evaluation import original_data
from fastlens.benchmarks.development.historical.refresh_evaluation import environment
from fastlens.benchmarks.execution.revision_results import select_current
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.run_phase2_repeats import predictions
from fastlens.benchmarks.system_dct_major.common import parse_manifest

OUT = ROOT/'results/remaining_inference_recheck_20260929'
KEY = ('gpu', 'model', 'pipeline')
from fastlens.benchmarks.paper.shared.references import PROFILES


def key(row):
    return tuple(row[k] for k in KEY)


def prepare():
    verify_snapshot()
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():
        return read(OUT/'plan.json')
    groups = defaultdict(list)
    for row in select_current(rows(ROOT/'results/p0/inference_observations_raw.csv')):
        groups[key(row)].append(row)
    jobs = []
    for group in groups.values():
        ref = max(group, key=lambda r: r['timestamp'])
        if ref['gpu']=='H100' and ref['architecture_family']=='ViT-Ti' and ref['pipeline']!='CPU-DCT':
            continue  # Complete, stable three-trial lower-load campaign already exists.
        source = ROOT/ref['raw_result'].split('#')[0]
        candidates = [source.parent/'run.command.json', source.parent.with_suffix('.command.json')]
        meta_path = next((p for p in candidates if p.exists()), None)
        profile = PROFILES[ref['model']]
        if meta_path is None:
            assert profile in ('vitti','swinv2') and source.parent.name=='inference'
            meta_path = source.parent.with_suffix('.command.json')
            metadata = read(meta_path)
            driver = 'fastlens/benchmarks/development/historical/vit_cpu_inference_once.py' if profile=='vitti' else 'fastlens/benchmarks/development/historical/swin_inference_once.py'
            pipeline = 'rgbnomore' if ref['pipeline']=='CPU-DCT' else ref['pipeline'].lower()
            command = [sys.executable, driver]
            if profile!='vitti':
                command += ['--pipeline', pipeline]
            command += ['--contract', str(source.parent/'contract.json'), '--output', str(source)]
        else:
            metadata = read(meta_path)
            command = metadata['command']
            # Legacy transformer metadata launches the entire multi-pipeline suite.
            if 'fastlens.benchmarks.system_dct_major.run' in command or 'fastlens.benchmarks.baselines.system_rgbnomore.inference.run' in command:
                pipeline = 'rgbnomore' if ref['pipeline']=='CPU-DCT' else ref['pipeline'].lower()
                command = [sys.executable, ('fastlens/benchmarks/development/historical/vit_cpu_inference_once.py' if profile=='vitti' else 'fastlens/benchmarks/development/historical/swin_inference_once.py')]
                if profile!='vitti':
                    command += ['--pipeline', pipeline]
                command += ['--contract', str(source.parent/'contract.json'), '--output', str(source)]
        assert '--output' in command or '--output-dir' in command, command
        if command[0]!='numactl':
            command = ['numactl','--cpunodebind=0','--membind=0',*command]
        contract = read(command[command.index('--contract')+1]) if '--contract' in command else None
        if contract:
            contract['execution'].update(repeats=1, aggregate_exclude_first_repeat=False, warmup_batches=0)
        gpu = '4090' if ref['gpu']=='RTX 4090' else 'H100'
        jobs.append(dict(id=f'{gpu}_{profile}_{ref["pipeline"].lower()}', reference=ref,
            command=command, contract=contract, metadata_source=str(meta_path.relative_to(ROOT)),
            environment={k:v for k,v in metadata['environment'].items() if v is not None},
            profile=profile, gpu=gpu,
            previous_median=statistics.median(int(r['samples'])/float(r['elapsed_s']) for r in group)))
    assert len(jobs)==63 and len({j['id'] for j in jobs})==63
    for relative in ('results/p0/deployment_inference_summary.csv','results/p0/deployment_ratios.csv',
                     'results/p0/matched_dct_inference.csv','results/PAPER_STATUS.md',
                     'fastlens/docs/paper/sections/07_evaluation.tex','fastlens/docs/paper/outputs/main.pdf'):
        target=OUT/'before'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    plan=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_fingerprint=verify_snapshot(), driver_sha=file_sha(Path(__file__)),
        vit_cpu_driver_sha=file_sha(ROOT/'fastlens/benchmarks/development/historical/vit_cpu_inference_once.py'), jobs=jobs,
        policy='50000 images; 3 trials, 2 more if CV>2%; no outlier removal; GPUs serial; rotate systems; existing H100 ViT deployment reused; host load annotated.')
    save(OUT/'plan.json',plan)
    return plan


def execute(job, trial):
    output=OUT/'inference'/job['id']/f'trial_{trial}'
    if (output/'failure.json').exists():
        output=output.with_name(output.name+'_retry')
    if (output/'accepted.json').exists():
        return
    output.mkdir(parents=True,exist_ok=False)
    reference=job['reference']
    original, expected=original_data(reference)
    command=list(job['command'])
    raw=output/Path(reference['raw_result'].split('#')[0]).name
    option='--output' if '--output' in command else '--output-dir'
    command[command.index(option)+1]=str(raw if option=='--output' else output)
    if job['contract']:
        contract=json.loads(json.dumps(job['contract']))
        if 'coordl' in contract['pipelines']:
            contract['pipelines']['coordl']['file_list']=str(output/'coordl_file_list.txt')
        save(output/'contract.json',contract)
        command[command.index('--contract')+1]=str(output/'contract.json')
        if reference['pipeline']=='FFCV':
            paths=[Path(contract['pipelines']['ffcv']['beton'])]
        elif reference['pipeline']=='FastLens':
            pipeline=command[command.index('--pipeline')+1]
            paths=[Path(s['fls_path']) for s in parse_manifest(Path(contract['pipelines'][pipeline]['manifest']))['shards']]
        else:
            paths=[Path(s['path']) for s in read(contract['dataset']['sample_manifest'])['samples']]
    elif reference['pipeline']=='FastLens':
        source=Path(command[command.index('--new-data')+1])
        paths=[Path(s['fls_path']) for s in parse_manifest(source/'manifest.bin')['shards']]
    else:
        paths=[Path(s['path']) for s in read(ROOT/'results/raw/H100/mobilenet24/inference/cpu_dct/trial_1/samples_50000.json')]
    env=dict(os.environ,**job['environment'])
    env.update(environment(job['gpu'],job['profile']))
    env['FASTLENS_EVALUATION_SNAPSHOT']=os.environ.get('FASTLENS_EVALUATION_SNAPSHOT',env['FASTLENS_EVALUATION_SNAPSHOT'])
    env['PYTHONPATH']=str(OUT/'native_runtime')+':'+env['PYTHONPATH']
    run(command,output,env,paths)
    data=read(raw)
    repeated='repeats' in data
    measured=data['repeats'][0] if repeated else data
    keys=('images','sample_trace','correct_top1') if repeated else ('samples','sample_ids','top1')
    assert measured[keys[0]]==50000 and all(measured[k]==expected[k] for k in keys), 'Sample IDs/count/accuracy mismatch'
    assert predictions(data)==predictions(original), 'Full prediction mismatch'
    checkpoint=data['model']['checkpoint_sha256'] if repeated else data['checkpoint']
    old_checkpoint=original['model']['checkpoint_sha256'] if repeated else original['checkpoint']
    assert checkpoint==old_checkpoint, 'Checkpoint mismatch'
    if repeated and reference['pipeline']=='FastLens':
        actual=np.load(data['semantic_artifact'])
        previous=np.load(original['semantic_artifact'])
        for name in ('input_0','input_1'):
            delta=np.abs(actual[name]-previous[name])
            assert delta.max()<=.001 and delta.mean()<=.0001, 'Tensor mismatch'
    meta=read(output/'run.command.json')
    assert meta['code_unchanged_during_run']
    elapsed=measured['repeat_scope_seconds'] if repeated else measured['e2e_seconds']
    save(output/'accepted.json',dict(reference,timestamp=meta['timestamp'],git_commit=meta['git_commit'],
        trial=trial,elapsed_s=elapsed,images_per_s=50000/elapsed,
        code_fingerprint=meta['code_fingerprint'],measurement_cohort=OUT.name,
        raw_result=str(raw.relative_to(ROOT))+('#/repeats/0' if repeated else ''),
        correctness='PASS: full 50000 IDs/predictions, checkpoint and correct count equal accepted reference; unchanged implementation and established tensor contract.',
        notes='Serial interleaved remeasurement; NUMA 0; unchanged saved configuration; warm cache; full host-load trace; no profiling. Prior results retained.'))


def summarize(plan, finished):
    observations=[read(p) for p in sorted((OUT/'inference').glob('*/trial_*/accepted.json'))]
    if observations:
        write_csv(OUT/'inference_raw.csv',observations,list(observations[0]))
    summary=[]
    for job in plan['jobs']:
        selected=[r for r in observations if key(r)==key(job['reference'])]
        if not selected:
            continue
        rates=[r['images_per_s'] for r in selected]
        median=statistics.median(rates)
        cv=100*statistics.stdev(rates)/statistics.mean(rates) if len(rates)>1 else None
        summary.append(dict(zip(KEY,key(job['reference'])),experiment_id=job['id'],trials=len(rates),
            median_images_per_s=median,min_images_per_s=min(rates),max_images_per_s=max(rates),
            cv_percent=cv,previous_median_images_per_s=job['previous_median'],
            change_percent=100*(median/job['previous_median']-1),
            status=('STABLE' if cv<=2 else 'VARIABLE') if job['id'] in finished else 'INCOMPLETE',
            raw_results=';'.join(r['raw_result'] for r in selected)))
    if summary:
        write_csv(OUT/'inference_summary.csv',summary,list(summary[0]))
    return {r['experiment_id']:r for r in summary}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    plan=prepare()
    if args.prepare:
        print('Prepared',len(plan['jobs']),'configurations;',3*len(plan['jobs']),'initial full passes')
        return
    assert file_sha(Path(__file__))==plan['driver_sha']
    assert file_sha(ROOT/'fastlens/benchmarks/development/historical/vit_cpu_inference_once.py')==plan['vit_cpu_driver_sha']
    state=read(OUT/'status.json') if args.resume else dict(completed=[],finished_groups=[],failures=[])
    state.update(status='RUNNING',pid=os.getpid(),active=None)
    families=['MobileNetV2','ResNet-50','eFUN','SwinV2-T','ViT-Ti']
    for family in families:
        for gpu in ('H100','4090'):
            jobs=sorted([j for j in plan['jobs'] if j['gpu']==gpu and j['reference']['architecture_family']==family],key=lambda j:j['id'])
            if all(j['id'] in state['finished_groups'] for j in jobs):
                continue
            failed=set()
            for trial in range(1,6):
                summary=summarize(plan,state['finished_groups'])
                if trial==4:
                    extra={j['id'] for j in jobs if j['id'] not in failed and summary[j['id']]['cv_percent']>2}
                ordered=jobs[(trial-1)%len(jobs):]+jobs[:(trial-1)%len(jobs)]
                for job in ordered:
                    if [job['id'],trial] in state['completed']:
                        continue
                    if job['id'] in failed or (trial>3 and job['id'] not in extra):
                        continue
                    state['active']=[job['id'],trial]
                    save(OUT/'status.json',state)
                    print('START',state['active'],flush=True)
                    try:
                        verify_snapshot()
                    except RuntimeError as error:
                        state.update(status='BLOCKED',reason=str(error))
                        save(OUT/'status.json',state)
                        raise
                    try:
                        execute(job,trial)
                    except (subprocess.CalledProcessError,AssertionError,RuntimeError) as error:
                        failed.add(job['id'])
                        failure=dict(experiment=state['active'],error=repr(error))
                        state['failures'].append(failure)
                        save(OUT/'inference'/job['id']/f'trial_{trial}'/'failure.json',failure)
                        print('FAILED',failure,flush=True)
                    else:
                        state['completed'].append(state['active'])
                        print('ACCEPTED',state['active'],flush=True)
                    summarize(plan,state['finished_groups'])
            state['finished_groups'].extend(j['id'] for j in jobs if j['id'] not in failed)
            summarize(plan,state['finished_groups'])
            save(OUT/'status.json',state)
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)
    state.update(active=None,status='COMPLETE' if not state['failures'] else 'BLOCKED_CELLS')
    save(OUT/'status.json',state)
    for script in ('fastlens/benchmarks/development/reports/deployment_report.py','fastlens/benchmarks/development/reports/export_milestone_a.py','fastlens/benchmarks/development/reports/populate_paper.py'):
        subprocess.run([sys.executable,script],check=True)
    subprocess.run(['bash','fastlens/docs/paper/scripts/build_paper.sh',str(Path.home()/'tmp/fastlens-remaining-inference-paper')],check=True)
    state['paper_updated']=True
    save(OUT/'status.json',state)


if __name__=='__main__':
    main()
