"""Append accepted expansion observations; preserve historical trials and summaries."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[4]
FAMILIES={'vitti':'ViT-Ti','swinv2':'SwinV2-T','mobilenet24':'MobileNetV2','mobilenet32':'MobileNetV2',
          'resnet24':'ResNet-50','resnet64':'ResNet-50','efun':'eFUN'}
SYSTEMS={'pytorch':'PyTorch','dali':'DALI','coordl':'CoorDL','ffcv':'FFCV','imagelanes':'FastLens'}


def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def write(p,records,fields):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
def union_fields(base,records):return list(dict.fromkeys([*base,*(k for r in records for k in r)]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    state=read(out/'status.json');plan=read(out/'plan.json')
    for name in ('results/p0/training_window_raw.csv','results/p0/training_window_summary.csv',
                 'results/p1/full_epoch_training.csv','results/experiment_manifest.csv','results/PAPER_STATUS.md','benchmark_results/archive/experiments/REPORT.md'):
        backup=out/'before'/name
        if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,backup)
    accepted=[]
    for job in plan['jobs']:
        if job['phase']!='window':continue
        directory=Path(job['output'])
        if not (directory/'accepted.json').exists():continue
        d=read(directory/'result.json');a=read(directory/'accepted.json');m=read(directory/'run.command.json')
        family=FAMILIES[job['model']]
        model=family if d['domain']=='rgb' else {'vitti':'ViT-Ti DCT','swinv2':'SwinV2-T DCT','efun':'eFUN-DCT'}.get(job['model'],family+'-DCT'+job['model'][-2:])
        if d['domain']=='rgb' and job['model']=='efun':model='EfficientNet-B0 reference'
        r=dict(timestamp=d['timestamp'],git_commit=d['git_commit'],code_fingerprint=a['code_fingerprint'],
            gpu='RTX 4090' if job['gpu']=='4090' else 'H100',cpu=d['cpu'],architecture_family=family,model=model,
            model_variant=job['model'],domain=d['domain'].upper(),initialization=d['initialization'],
            pipeline=SYSTEMS[job['pipeline']],comparison_type='deployment',task='training_window',
            precision=d['precision'],batch_size=d['batch'],effective_batch_size=d['effective_batch_size'],trial=d['trial'],
            samples=d['images'],consumed_samples=d['images'],optimizer_updates=d['updates'],elapsed_s=d['seconds'],
            images_per_s=d['images']/d['seconds'],final_loss=d['final_loss'],loss_finite=d['loss_finite'],training_loss=d['training_loss'],
            initial_model_hash=d['initial_model_hash'],worker_count=d['worker_count'],decoder_threads=d['decoder_threads'],
            prefetch=d['prefetch'],cache_mode=d['cache_mode'],cpu_affinity=json.dumps(d['cpu_affinity']),numa_binding=d['numa_policy'],
            augmentation_configuration=d['augmentation_configuration'],sample_order_sha256=d['sample_order_sha256'],
            crop_verification=d['crop_verification'],rgb_input_reference=d.get('rgb_input_reference',''),
            correctness_status='PASS',correctness_report=a['correctness_report'],
            raw_result=a['raw_result'],source_log=str((directory/'run.log').relative_to(ROOT)),command=shlex.join(m['command']),
            concurrent_load_notes=str((directory/'concurrent_load.jsonl').relative_to(ROOT)),valid=True,strict_matched=False,
            implementation_cohort=out.name,measurement_cohort=out.name,notes=d['notes'],native_sha256=a['native_sha256'] if d['domain']=='dct' else '')
        accepted.append(r)
    raw_path=ROOT/'results/p0/training_window_raw.csv';raw=rows(raw_path);known={r['raw_result'] for r in raw}
    raw.extend(r for r in accepted if r['raw_result'] not in known)
    write(raw_path,raw,union_fields(list(raw[0]),accepted))
    write(out/'window_raw.csv',accepted,union_fields(list(raw[0]),accepted))
    groups={}
    for r in accepted:groups.setdefault((r['gpu'],r['model'],r['pipeline']),[]).append(r)
    summary=[];changes=[]
    historical=rows(out/'before/results/p0/training_window_summary.csv')
    for key,values in groups.items():
        if len(values)!=3:continue
        for field in ('initial_model_hash','sample_order_sha256','precision','optimizer_updates','samples'):
            if len({r[field] for r in values})!=1:raise ValueError(f'Trial contract changed: {key}, {field}')
        rates=[r['images_per_s'] for r in values];median=statistics.median(rates);cv=statistics.stdev(rates)/statistics.mean(rates)
        s=dict(gpu=key[0],model=key[1],pipeline=key[2],domain=values[0]['domain'],task='training_window',
            precision=values[0]['precision'],batch_size=values[0]['batch_size'],
            effective_batch_size=values[0]['effective_batch_size'],samples=65536,trials=3,
            median_elapsed_s=statistics.median(r['elapsed_s'] for r in values),
            median_images_per_s=median,min_images_per_s=min(rates),
            max_images_per_s=max(rates),cv_percent=100*cv,coefficient_of_variation=cv,
            raw_results=';'.join(r['raw_result'] for r in values),implementation_cohort=out.name,
            notes='All three trials retained; VARIABLE' if cv>.05 else 'All three trials retained')
        summary.append(s)
        previous=next((r for r in historical if (r['gpu'],r['model'],r['pipeline'])==key),None)
        if previous:
            before=float(previous['median_images_per_s']);changes.append(dict(gpu=key[0],model=key[1],pipeline=key[2],
                old_images_per_s=before,new_images_per_s=median,change_percent=100*(median/before-1),
                old_raw=previous['raw_results'],new_raw=s['raw_results'],
                notes='Configuration/version/shared-host change; not isolated code causality. Previous observations retained.'))
    path=ROOT/'results/p0/training_window_summary.csv';old=rows(path);keys={(r['gpu'],r['model'],r['pipeline']) for r in summary}
    combined=[r for r in old if (r['gpu'],r['model'],r['pipeline']) not in keys]+summary
    write(path,combined,union_fields(list(old[0]),summary))
    if summary:write(out/'window_summary.csv',summary,list(summary[0]))
    if changes:write(out/'performance_changes.csv',changes,list(changes[0]))
    path=ROOT/'results/experiment_manifest.csv';manifest=rows(path);prefix=out.name+':'
    manifest=[r for r in manifest if not r['experiment_id'].startswith(prefix)]
    blocked={(r['model'],r['pipeline']) for r in state['failures']}
    for job in plan['jobs']:
        directory=Path(job['output']);done=(directory/'accepted.json').exists()
        status='DONE' if done else 'BLOCKED' if job['gpu']==state['gpu'] and (job['model'],job['pipeline']) in blocked else 'PENDING'
        manifest.append(dict(experiment_id=prefix+':'.join(str(job[k]) for k in ('phase','gpu','model','pipeline','trial')),
            phase='P0-D' if job['phase']=='window' else 'P1',task='training_window' if job['phase']=='window' else 'training_epoch',
            gpu='RTX 4090' if job['gpu']=='4090' else 'H100',architecture=FAMILIES[job['model']],model_variant=job['model'],
            domain='DCT' if job['pipeline'] in ('imagelanes','cpu_dct') else 'RGB',pipeline=SYSTEMS.get(job['pipeline'],'CPU-DCT'),
            sample_count=65536 if job['phase']=='window' else 1281167,precision='fp32' if job['model']=='vitti' else 'bf16-autocast',
            batch_size=64,status=status,result_file=str(directory.relative_to(ROOT)),correctness='PASS' if done else '',
            notes='Original inference/paper complete; 4090 authorized first. All missing windows precede complete epochs.'))
    write(path,manifest,list(manifest[0]))
    section=(f'\n## {out.name}\n\n'+f"Status: {state['status']}; GPU {state['gpu']}; accepted windows {len(accepted)}/{plan['window_trials']}; active {state['active']}; blocked cells {len(state['failures'])}.\n\n"+
        plan.get('report_scope','75 windows per GPU; 12 historical native windows and 8 matched epochs reused. 48 old RGB windows retained but replaced under unified crop/order/resize/initialization semantics. 20 missing single epochs follow all missing windows. No convergence claim. Shared host, not isolated; H100 VLLM activity recorded.')+'\n\n'+
        f"Plan/raw logs/state: `{out.relative_to(ROOT)}`. Native pinned: `{read(out/'handoff.json')['native_sha256']}`. Old tables preserved under `before/`. Correctness failures and all observations retained; no automatic timing retries.\n")
    begin=f'<!-- {out.name} begin -->'
    end=f'<!-- {out.name} end -->'
    section=begin+section+end+'\n'
    for path in (ROOT/'results/PAPER_STATUS.md',ROOT/'benchmark_results/archive/experiments/REPORT.md'):
        text=path.read_text()
        if begin in text:
            start=text.index(begin);finish=text.index(end,start)+len(end)
            text=text[:start]+section.rstrip()+text[finish:]
        else:text=text.rstrip()+'\n\n'+section
        path.write_text(text)
    print(json.dumps(dict(accepted_windows=len(accepted),complete_groups=len(summary),status=state['status'])))


if __name__=='__main__':main()
