"""Paired spatial-pushdown ablation of the actual Section 4.2 workloads."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import jobs
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import inference_inputs
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import training_inputs
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import verify
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import metrics
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import set_arg


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def compare(a, b, training):
    if training:
        for key in ('initial_model_hash', 'sample_order_sha256', 'images', 'updates'):
            assert a[key] == b[key], (key, a[key], b[key])
        assert math.isclose(a['training_loss'], b['training_loss'], rel_tol=1e-3, abs_tol=1e-4)
    else:
        aa = a['repeats'][0] if 'repeats' in a else a
        bb = b['repeats'][0] if 'repeats' in b else b
        for key in ('top1_predictions_sha256', 'sample_ids', 'predictions', 'top1', 'accuracy_top1'):
            if key in aa:
                assert aa[key] == bb[key], key


def profile_metrics(destination, images):
    """Assign whole asynchronous copies by their GPU start time."""
    with sqlite3.connect(destination/'capture.sqlite') as connection:
        (start, end), = connection.execute(
            "SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE COALESCE(n.text,s.value)='fastlens.measured_pass'").fetchall()
        overlapping = connection.execute(
            'SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY '
            'WHERE copyKind=1 AND start<? AND end>?', (end, start)).fetchall()
    copies = [(a,b,n) for a,b,n in overlapping if a >= start]
    total = sum(n for a,b,n in copies)
    return dict(h2d_bytes=total, h2d_bytes_per_image=total/images, h2d_copy_count=len(copies),
                h2d_scope='Whole H2D copies whose GPU start time is within the measured NVTX window',
                h2d_trailing_boundary_bytes=sum(n for a,b,n in copies if b > end),
                h2d_leading_boundary_bytes_excluded=sum(n for a,b,n in overlapping if a < start))


def summarize(output):
    rows = [read(p) for p in sorted(output.glob('*/*/*/*/measurement.json'))]
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (output/'measurements.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = []
    for phase in ('inference', 'training'):
        for model in sorted({r['model'] for r in rows if r['phase'] == phase}):
            for mode in ('warm', 'cold'):
                cohorts = {}
                for trial in range(3):
                    pair = [next((r for r in rows if r['phase']==phase and r['model']==model
                                  and r['stage']==f'{mode}_{trial}' and r['arm']==arm), None)
                            for arm in ('selected', 'full')]
                    if all(pair):
                        a, b = pair
                        if a['native_sha256'] != b['native_sha256']:
                            print('EXCLUDED cross-version pair', phase, model, mode, trial, flush=True)
                            continue
                        cohorts.setdefault(a['native_sha256'], []).append(pair)
                for native_sha, pairs in cohorts.items():
                    item = dict(phase=phase, model=model, cache=mode, native_sha256=native_sha, pairs=len(pairs),
                        selected_median_s=statistics.median(a['seconds'] for a,b in pairs),
                        full_median_s=statistics.median(b['seconds'] for a,b in pairs),
                        paired_speedup_median=statistics.median(b['seconds']/a['seconds'] for a,b in pairs),
                        paired_speedup_min=min(b['seconds']/a['seconds'] for a,b in pairs),
                        paired_speedup_max=max(b['seconds']/a['seconds'] for a,b in pairs))
                    if mode=='cold':
                        for key in ('total_storage_read_bytes','measured_window_storage_read_bytes'):
                            a=statistics.median(x[key] for x,y in pairs)
                            b=statistics.median(y[key] for x,y in pairs)
                            item.update({f'selected_{key}':a, f'full_{key}':b,
                                         f'saved_{key}':b-a, f'reduction_{key}':1-a/b})
                    summary.append(item)
    save(output/'summary.json', summary)
    profiles = [r for r in rows if r['stage'] == 'profile' and r['phase'] == 'training']
    resources = []
    for model in sorted({r['model'] for r in profiles}):
        pair = {r['arm']: r for r in profiles if r['model'] == model}
        if set(pair) != {'full', 'selected'}:
            continue
        a, b = pair['selected'], pair['full']
        assert a['native_sha256'] == b['native_sha256']
        item = dict(model=model, native_sha256=a['native_sha256'])
        for key in ('h2d_bytes', 'planned_vectors', 'actual_vectors'):
            if key in a and key in b:
                item.update({f'selected_{key}': a[key], f'full_{key}': b[key],
                             f'saved_{key}': b[key]-a[key], f'reduction_{key}': 1-a[key]/b[key]})
        resources.append(item)
    save(output/'resource_summary.json', resources)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/crop_workloads_20261010')
    parser.add_argument('--library',type=Path,default=Path.home()/'tmp/crop_workload_20261010/libcrop_control.so')
    parser.add_argument('--native-dir',type=Path,default=ROOT/'build/fastlens/torch')
    parser.add_argument('--training-resources',action='store_true',
                        help='All five RTX 4090 deployment training workloads: checks, two cold pairs, one H2D pair')
    parser.add_argument('--stages',nargs='+',choices=['check','cold_0','cold_1','profile'],
                        help='Run selected training-resource stages when shared input activity prevents cold measurements')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    native, = args.native_dir.glob('_fastlens_direct_dct.*.so')
    native_stamp=(native.stat().st_size,native.stat().st_mtime_ns)
    native_sha=hashlib.sha256(native.read_bytes()).hexdigest()
    for path in output.glob('*/*/*/*/measurement.json'):
        assert read(path)['native_sha256'] == native_sha, f'Reused result has a different native binary: {path}'
    selected=[j for j in jobs() if j['gpu']=='4090' and j['pipeline']=='FastLens'
              and ((args.training_resources and j['phase']=='training') or
                   (not args.training_resources and j['model'] in ('vitti','mobilenet24')))]
    selected.sort(key=lambda j:(j['phase'],j['model']))
    stages=['check']+[f'warm_{i}' for i in range(3)]+[f'cold_{i}' for i in range(3)]+['memory','profile']
    if args.training_resources:
        assert {j['model'] for j in selected} == {'vitti','swinv2','mobilenet24','mobilenet32','efun'}
        stages=['check','cold_0','cold_1','profile']
    required_stages = list(stages)
    if args.stages:
        assert args.training_resources, '--stages is for the training-resource campaign'
        stages=args.stages
    for stage in stages:
        for job in selected:
            training=job['phase']=='training'
            arms=('selected','full') if stage in ('check','memory','profile') or stage.endswith(('0','2')) else ('full','selected')
            pair={}
            for arm in arms:
                dest=output/job['phase']/job['model']/stage/arm
                if (dest/'measurement.json').exists():
                    pair[arm]=read(dest/'measurement.json')
                    continue
                assert (native.stat().st_size,native.stat().st_mtime_ns)==native_stamp, 'Native binary changed during campaign'
                dest.mkdir(parents=True,exist_ok=True)
                assert not (dest/'run.log').exists(), f'Inspect incomplete attempt: {dest}'
                command=list(job['metadata']['command'])
                env=dict(os.environ, **{k:v for k,v in job['metadata']['environment'].items() if v is not None})
                for key in ('FASTLENS_EVALUATION_SNAPSHOT','FASTLENS_ALLOWED_RESIDENT_PIDS'):
                    env.pop(key,None)
                env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(Path.home()/'tmp'),
                           PYTHONPATH=f'{ROOT}:{args.native_dir.resolve()}',
                           FASTLENS_TRAINING_NATIVE_DIR=str(args.native_dir.resolve()))
                option='--output' if '--output' in command else '--output-dir'
                raw=dest/Path(job['source']).name
                set_arg(command,option,dest if training or option=='--output-dir' else raw)
                if training:
                    if '--correctness-report' in command:
                        i=command.index('--correctness-report');del command[i:i+2]
                    if stage=='check':command.append('--check')
                    elif stage=='profile':command.append('--profile')
                    elif stage=='memory' or stage.startswith('cold'):command.append('--memory')
                    else:command+=['--correctness-report',str(output/job['phase']/job['model']/'check'/arm/'result.json')]
                    paths=training_inputs(job,command) if stage.startswith('cold') else None
                else:
                    paths=inference_inputs(job,command,dest)
                index=next(i for i,x in enumerate(command) if Path(x).name in ('python','python3'))
                workload=dest/'workload_command.json'
                save(workload,command[index+1:])
                command=command[:index+1]+['fastlens/benchmarks/development/diagnostics/crop_workload_control.py','--library',str(args.library),
                    '--native-dir',str(args.native_dir.resolve()),
                    '--arm',arm,'--command',str(workload),'--output',str(dest)]
                if args.training_resources:
                    command.append('--training-vectors')
                if stage=='check':command.append('--check-inputs')
                mode='cold' if stage.startswith('cold') else stage
                if mode in ('memory','profile','cold'):
                    command+=['--resource',mode]
                if mode=='cold':
                    save(dest/'input_paths.json',[str(p) for p in paths])
                    command+=['--input-paths',str(dest/'input_paths.json')]
                    paths=None
                if mode=='profile':
                    command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(dest/'capture'),*command]
                save(output/'state.json',dict(phase='running',job=job['phase'],model=job['model'],stage=stage,arm=arm,pid=os.getpid()))
                print('RUN',job['phase'],job['model'],stage,arm,flush=True)
                monitors=[]
                streams=[]
                if mode in ('memory','cold'):
                    for filename,query in [('gpu_process_memory.csv',['--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory']),
                                           ('gpu_utilization.csv',['-i',env['CUDA_VISIBLE_DEVICES'],'--query-gpu=timestamp,utilization.gpu'])]:
                        stream=(dest/filename).open('x');streams.append(stream)
                        monitors.append(subprocess.Popen(['nvidia-smi',*query,'--format=csv,noheader,nounits','-lms','100'],stdout=stream))
                try:
                    run(command,dest,env,paths)
                finally:
                    for m in monitors:m.terminate();m.wait()
                    for f in streams:f.close()
                result=read(raw)
                if stage=='check' and training:
                    assert result['correctness_passed'] and result['images']==1024
                    count=1024
                else:count=verify(job,result,cold_start=mode=='cold')
                actual=result['repeats'][0] if 'repeats' in result else result
                seconds=actual['repeat_scope_seconds'] if 'repeat_scope_seconds' in actual else actual.get('e2e_seconds',actual.get('seconds'))
                record=dict(phase=job['phase'],model=job['model'],stage=stage,arm=arm,images=count,
                            seconds=seconds,raw_result=str(raw),native_sha256=native_sha)
                if mode=='profile':
                    subprocess.run(['nsys','export','--type','sqlite','--output',str(dest/'capture.sqlite'),str(dest/'capture.nsys-rep')],check=True,stdout=subprocess.DEVNULL)
                if mode=='profile':
                    record.update(profile_metrics(dest,count))
                elif mode in ('memory','cold'):
                    record.update(metrics(dest,mode,count))
                if mode=='cold':
                    record.update({k:v for k,v in read(dest/'cold_start_io.json').items() if k.endswith('storage_read_bytes')})
                if args.training_resources:
                    vectors = read(dest/'training_vectors.json')
                    assert vectors['measured_images'] == count
                    record.update({k: vectors[k] for k in ('planned_vectors','actual_vectors',
                                                          'measured_pool_count','boundary_pool_count')})
                save(dest/'measurement.json',record)
                pair[arm]=record
                summarize(output)
                print('DONE',job['phase'],job['model'],stage,arm,seconds,flush=True)
            compare(read(Path(pair['selected']['raw_result'])),read(Path(pair['full']['raw_result'])),training)
            if stage=='check':
                a=read(output/job['phase']/job['model']/stage/'selected/crop_control.json')
                b=read(output/job['phase']/job['model']/stage/'full/crop_control.json')
                assert a['model_input_hashes'] and len(a['model_input_hashes'])==len(b['model_input_hashes'])
                if a['model_input_hashes'] != b['model_input_hashes'] or training:
                    assert training, 'Inference model inputs differ'
                    import torch
                    base=output/job['phase']/job['model']/stage
                    differences=[]
                    for i in range(len(a['model_input_hashes'])):
                        aa=torch.load(base/'selected'/f'input_{i:02d}.pt',weights_only=True)
                        bb=torch.load(base/'full'/f'input_{i:02d}.pt',weights_only=True)
                        for j,(x,y) in enumerate(zip(aa,bb)):
                            target=j==len(aa)-1
                            # Existing Swin/ViT input and matched-update tolerances.
                            torch.testing.assert_close(x,y,atol=1e-7 if target else 1/1020+1e-7,
                                                       rtol=1e-6 if target else 1e-4)
                            differences.append(float((x-y).abs().max()))
                    aa=torch.load(base/'selected/final_state.pt',weights_only=True)
                    bb=torch.load(base/'full/final_state.pt',weights_only=True)
                    state_difference=0.
                    for k,x in aa.items():
                        torch.testing.assert_close(x,bb[k],rtol=1e-3,atol=1e-4)
                        state_difference=max(state_difference,float((x-bb[k]).abs().max()))
                    save(base/'numerical_comparison.json',dict(input_max_abs=max(differences),
                        state_max_abs=state_difference,input_atol=1/1020+1e-7,input_rtol=1e-4,
                        update_atol=1e-4,update_rtol=1e-3))
    complete = all((output/j['phase']/j['model']/stage/arm/'measurement.json').exists()
                   for j in selected for stage in required_stages for arm in ('selected','full'))
    save(output/'state.json',dict(phase='complete' if complete else 'partial',pid=os.getpid()))
    print('CROP WORKLOAD QUEUE COMPLETE' if complete else 'SELECTED STAGES COMPLETE; OTHER MEASUREMENTS PENDING',flush=True)


if __name__=='__main__':
    main()
