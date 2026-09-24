"""Analyze bounded inference captures using existing Nsight interval helpers."""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

from galp.benchmarks.profiling.analyze_training_nsys import (
    _clip, _duration, _gpu_events, _named_nvtx, _union_ns,
)
from galp.benchmarks.profiling.build_actual_overlap_comparison import _intersection_ns


def analyze(path):
    c = sqlite3.connect(path)
    windows = c.execute(
        "SELECT n.start,n.end,n.globalTid FROM NVTX_EVENTS n LEFT JOIN StringIds s "
        "ON n.textId=s.id WHERE COALESCE(n.text,s.value)='dctnet.capture'").fetchall()
    assert len(windows) == 1 and windows[0][1] is not None
    start, end, main_tid = windows[0]
    stages = defaultdict(list)
    for name, a, b, tid in _named_nvtx(c, start, end):
        if tid == main_tid and (name.startswith('input.') or name in ('model.forward', 'metrics')):
            stages[name].append((a, b))
    assert len(stages['model.forward']) == 64
    ranges = sorted((a, b, name) for name, values in stages.items() for a,b in values)
    starts = [a for a,_,_ in ranges]
    launches = {}
    for corr, a, tid in c.execute("SELECT correlationId,start,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        assert corr not in launches
        launches[corr] = (a, tid)
    groups = defaultdict(list)
    top = defaultdict(lambda: [0, 0])
    for a,b,corr,name in c.execute(
        "SELECT k.start,k.end,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName=s.id WHERE k.start < ? AND k.end > ?", (end,start)):
        category = 'unclassified'
        if corr in launches:
            launch, tid = launches[corr]
            if tid == main_tid:
                i = bisect.bisect_right(starts, launch)-1
                if i >= 0 and ranges[i][0] <= launch < ranges[i][1]:
                    scope = ranges[i][2]
                    category = 'model' if scope == 'model.forward' else ('input' if scope.startswith('input.') else 'metrics')
                else:
                    category = 'other_main'
            elif path.stem in ('rgb_dali', 'galp_native', 'galp_projected'):
                category = 'input'
        interval = _clip(a,b,start,end)
        groups[category].append(interval)
        top[(category,name)][0] += interval[1]-interval[0]
        top[(category,name)][1] += 1
    _, copies = _gpu_events(c,start,end,kind='galp')
    copy_groups = defaultdict(list)
    copy_bytes = defaultdict(int)
    for kind,a,b,size in copies:
        copy_groups[kind].append((a,b))
        copy_bytes[kind] += size
    memsets = []
    # Nsight omits the table when the captured path has no cudaMemset calls.
    if c.execute("SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone():
        memsets = [interval for a,b in c.execute(
            'SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start < ? AND end > ?', (end,start))
            if (interval := _clip(a,b,start,end))]
    all_kernels = [v for values in groups.values() for v in values]
    all_copies = [(a,b) for _,a,b,_ in copies]
    busy = _union_ns(all_kernels+all_copies+memsets)
    gpu = {name:dict(kernel_count=len(values), union_seconds=_union_ns(values)/1e9,
                    summed_kernel_seconds=sum(b-a for a,b in values)/1e9)
           for name,values in groups.items()}
    idle = (end-start-busy)/1e9
    result = dict(trace=str(path.with_suffix('.nsys-rep').resolve()), captured_images=4096,
                  window_seconds=(end-start)/1e9,
                  main_stages={k:_duration([b-a for a,b in v]) for k,v in stages.items()},
                  gpu=gpu, gpu_busy_seconds=busy/1e9, gpu_idle_seconds=idle,
                  gpu_idle_fraction=idle/((end-start)/1e9),
                  model_input_overlap_seconds=_intersection_ns(groups['model'],groups['input'])/1e9,
                  model_copy_overlap_seconds=_intersection_ns(groups['model'],all_copies)/1e9,
                  copies={k:dict(union_seconds=_union_ns(v)/1e9,bytes=copy_bytes[k]) for k,v in copy_groups.items()},
                  top_kernels=[dict(category=k[0],name=k[1],summed_seconds=v[0]/1e9,count=v[1])
                               for k,v in sorted(top.items(),key=lambda kv:kv[1][0],reverse=True)[:20]],
                  classification='CUDA runtime correlation to main-thread NVTX launch scope; DALI/native background GPU launches are input; unmatched launches retained separately',
                  scope='profiling window only, not unprofiled throughput; kernel/memcpy interval union is activity, not SM utilization')
    c.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args = p.parse_args()
    results = {path.stem:analyze(path) for path in sorted(args.directory.glob('*.sqlite'))}
    (args.directory/'breakdown.json').write_text(json.dumps(results,indent=2))
    for name,r in results.items():
        print(name, json.dumps({k:v for k,v in r.items() if k not in ('top_kernels','trace','classification','scope')}))
    plot(results, args.directory/'breakdown.svg')


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    stages, device = [], []
    for name in names:
        r = results[name]
        stage = {k:v['sum_seconds'] for k,v in r['main_stages'].items()}
        row = [stage.get('input.wait',0), sum(stage.get(k,0) for k in
               ('input.handoff','input.organize','input.prefetch_submit')),
               stage['model.forward'], stage.get('metrics',0)]
        stages.append(row+[r['window_seconds']-sum(row)])
        m = r['gpu']['model']['union_seconds']
        i = r['gpu'].get('input',{}).get('union_seconds',0)
        overlap = r['model_input_overlap_seconds']
        device.append([m-overlap,overlap,i-overlap,
                       r['gpu_busy_seconds']-m-i+overlap,r['gpu_idle_seconds']])
    specs = [(stages,['Input wait','Input handoff/organize','Forward scope (includes waits)',
                      'Metrics','Other host time'],['#e69f00','#56b4e9','#009e73','#999999','#dddddd']),
             (device,['Model kernels only','Model + input overlap','Input kernels only',
                      'Other GPU activity','No GPU activity'],['#009e73','#cc79a7','#e69f00','#56b4e9','#dddddd'])]
    for ax,(rows,labels,colors),title in zip(axes,specs,
            ['Main-thread scopes','GPU activity interval union (not SM utilization)']):
        left = np.zeros(len(names))
        for values,label,color in zip(np.array(rows).T,labels,colors):
            ax.barh(names,values,left=left,label=label,color=color)
            left += values
        ax.invert_yaxis()
        ax.set_xlabel('Seconds per captured 4096 images; profiler enabled')
        ax.set_title(title,loc='left')
        ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=3,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(output)
    fig.savefig(output.with_suffix('.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
