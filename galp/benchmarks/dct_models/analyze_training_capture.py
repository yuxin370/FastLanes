"""Attribute training CUDA work by launch thread and model NVTX ranges."""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

from galp.benchmarks.profiling.analyze_training_nsys import _clip, _merge, _named_nvtx, _union_ns
from galp.benchmarks.profiling.build_actual_overlap_comparison import _intersection_ns


def gap_breakdown(c, start, end, main_tid):
    events = []
    for table in ('KERNEL', 'MEMCPY', 'MEMSET'):
        table = 'CUPTI_ACTIVITY_KIND_' + table
        if c.execute('SELECT 1 FROM sqlite_master WHERE name=?', (table,)).fetchone():
            events.extend((max(a, start), min(b, end), corr) for a, b, corr in c.execute(
                f'SELECT start,end,correlationId FROM {table} WHERE start<? AND end>?', (end, start)))
    events.sort()
    gaps, cursor = [], start
    for a, b in _merge([(a, b) for a, b, _ in events]):
        if cursor < a:
            gaps.append((cursor, a))
        cursor = b
    if cursor < end:
        gaps.append((cursor, end))
    buckets = defaultdict(lambda: dict(count=0, seconds=0.0))
    for a, b in gaps:
        duration = b-a
        key = 'under_10us' if duration < 10_000 else '10_to_100us' if duration < 100_000 else '100us_to_1ms' if duration < 1_000_000 else 'at_least_1ms'
        buckets[key]['count'] += 1
        buckets[key]['seconds'] += duration/1e9
    launches = {corr: (a, b) for corr, a, b in c.execute(
        'SELECT correlationId,start,end FROM CUPTI_ACTIVITY_KIND_RUNTIME')}
    starts = [a for a, _, _ in events]
    relative = defaultdict(float)
    for a, b in gaps:
        i = bisect.bisect_left(starts, b)
        call = launches.get(events[i][2]) if i < len(events) else None
        if call is None:
            relative['unknown_seconds'] += (b-a)/1e9
            continue
        x, y = call
        relative['before_launch_seconds'] += max(0, min(b, x)-a)/1e9
        relative['inside_launch_seconds'] += max(0, min(b, y)-max(a, x))/1e9
        relative['after_launch_seconds'] += max(0, b-max(a, y))/1e9
    scopes = defaultdict(list)
    for name, a, b, tid in _named_nvtx(c, start, end):
        if tid == main_tid and name.startswith(('detail.', 'model.', 'input.', 'training.')):
            scopes[name].append((a, b))
    return dict(count=len(gaps), duration_buckets=dict(buckets),
        relative_to_next_activity_launch=dict(relative),
        host_scopes={name: dict(host_seconds=_union_ns(v)/1e9,
            gpu_idle_seconds=_intersection_ns(gaps, v)/1e9) for name, v in scopes.items()},
        note='Idle means no traced kernel/copy/memset activity. Nested host scopes overlap. '
             'Relation to the next activity launch does not establish that all GPU queues were empty.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeline', type=Path, help='export a 360 ms GPU timeline (PNG and PDF)')
    p.add_argument('--label', default='CNN training')
    p.add_argument('--window-name', default='dctnet.capture')
    args = p.parse_args()
    c = sqlite3.connect(args.sqlite)
    start, end, main_tid = c.execute(
        'SELECT start,end,globalTid FROM NVTX_EVENTS WHERE text=?', (args.window_name,)).fetchone()
    model_prefixes = ('model.', 'training.model.', 'training.loss', 'training.optimizer', 'training.batch.audit')
    ranges = sorted((a, b, name) for name, a, b, tid in _named_nvtx(c, start, end)
                    if tid == main_tid and name.startswith(model_prefixes))
    starts = [a for a, _, _ in ranges]
    input_ranges = sorted((a, b) for name, a, b, tid in _named_nvtx(c, start, end)
                          if tid == main_tid and name.startswith(('input.', 'training.loader.', 'training.input_handoff')))
    input_starts = [a for a, _ in input_ranges]
    detail_ranges = sorted((a, b, name) for name, a, b, tid in _named_nvtx(c, start, end)
                           if tid == main_tid and (name.startswith(('detail.', *model_prefixes[1:]))
                                                  or name == 'model.optimizer_audit'))
    detail_starts = [a for a, _, _ in detail_ranges]
    launch = {corr: (t, tid) for corr, t, tid in c.execute('SELECT correlationId,start,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME')}
    # PyTorch backward kernels are launched by autograd worker threads as well
    # as the Python thread. Identify the native producer from its own kernels.
    native_tids = {launch[corr][1] for corr, name, full in c.execute(
        'SELECT k.correlationId,s.value,f.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id JOIN StringIds f ON k.demangledName=f.id')
        if corr in launch and launch[corr][1] != main_tid and ('dali::' in full or 'nvjpeg::' in full or 'transformed_dct_grid' in name
            or name in ('projected_stats', 'projected_to_int16', 'projected_normalize_mixup'))}
    groups, names = defaultdict(list), defaultdict(lambda: [0, 0])
    model_phases = defaultdict(list)
    for a, b, corr, name in c.execute('SELECT k.start,k.end,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id WHERE k.start<? AND k.end>?', (end, start)):
        category = 'unattributed'
        if corr in launch:
            t, tid = launch[corr]
            i = bisect.bisect_right(starts, t) - 1
            if tid in native_tids:
                category = 'input'
            elif tid != main_tid or (i >= 0 and ranges[i][0] <= t < ranges[i][1]):
                category = 'model'
            else:
                j = bisect.bisect_right(input_starts, t)-1
                category = 'input' if j >= 0 and input_ranges[j][0] <= t < input_ranges[j][1] else 'main_auxiliary'
        a, b = _clip(a, b, start, end)
        groups[category].append((a, b))
        if category == 'model' and corr in launch:
            i = bisect.bisect_right(detail_starts, launch[corr][0])-1
            phase = detail_ranges[i][2] if i >= 0 and launch[corr][0] < detail_ranges[i][1] else 'outside_detail_scope'
            model_phases[phase].append((a, b))
        names[(category, name)][0] += 1
        names[(category, name)][1] += b-a
    copy_bytes = defaultdict(int)
    for a,b,kind,size in c.execute('SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?', (end,start)):
        groups[f'copy_kind_{kind}'].append(_clip(a,b,start,end))
        copy_bytes[kind] += size
    if c.execute("SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone():
        groups['memset'] = [_clip(a,b,start,end) for a,b in c.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start<? AND end>?', (end,start))]
    busy = _union_ns([v for intervals in groups.values() for v in intervals])
    sync_tails = []
    for name, a, b, tid in _named_nvtx(c, start, end):
        if tid != main_tid or name != 'detail.pool_sync':
            continue
        last_model = max((min(y, b) for x, y in groups['model'] if x < b and y > a), default=a)
        sync_tails.append(dict(host_seconds=(b-a)/1e9, after_last_model_seconds=(b-last_model)/1e9,
            input_busy_after_last_model_seconds=_intersection_ns(groups['input'], [(last_model, b)])/1e9))
    result = dict(trace=str(args.sqlite), captured_images=16384, window_seconds=(end-start)/1e9,
        groups={key: dict(events=len(v), union_seconds=_union_ns(v)/1e9, summed_seconds=sum(b-a for a,b in v)/1e9) for key,v in groups.items()},
        gpu_busy_seconds=busy/1e9, gpu_idle_seconds=(end-start-busy)/1e9,
        input_model_overlap_seconds=_intersection_ns(groups['input'], groups['model'])/1e9,
        gaps=gap_breakdown(c, start, end, main_tid),
        model_phases={key: dict(events=len(v), union_seconds=_union_ns(v)/1e9)
                      for key, v in model_phases.items()},
        pool_sync=dict(calls=len(sync_tails),
            totals={key: sum(v[key] for v in sync_tails) for key in (
                'host_seconds', 'after_last_model_seconds', 'input_busy_after_last_model_seconds')},
            note='Input activity after the last model kernel inside the host pool sync scope; '
                 'not a prediction of the speedup from removing device-wide synchronization.'),
        largest_h2d_sizes=[dict(bytes_per_copy=size, calls=count, total_bytes=total, seconds=seconds)
            for size, count, total, seconds in c.execute(
                'SELECT bytes,count(*),sum(bytes),sum(min(end,?)-max(start,?))/1e9 '
                'FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>? AND copyKind=1 '
                'GROUP BY bytes ORDER BY sum(bytes) DESC LIMIT 5', (end, start, end, start))],
        transform_resources=[dict(grid_x=g, block_x=b, registers_per_thread=r,
            shared_bytes_per_block=s, local_bytes_per_thread=l, calls=n)
            for g, b, r, s, l, n in c.execute(
                'SELECT k.gridX,k.blockX,k.registersPerThread,k.staticSharedMemory,k.localMemoryPerThread,count(*) '
                'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id '
                "WHERE k.start<? AND k.end>? AND s.value='transformed_dct_grid_planless_kernel' "
                'GROUP BY 1,2,3,4,5', (end, start))],
        copy_bytes=dict(copy_bytes), top_kernels=[dict(category=key[0],name=key[1],calls=v[0],seconds=v[1]/1e9)
            for key,v in sorted(names.items(), key=lambda item:item[1][1], reverse=True)[:25]],
        note='profiled 16384-image window; first compile excluded; audit phase follows global update count including resume; overlapping durations are not additive')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.timeline:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        lanes = [('GPU input', groups['input']), ('Model GPU', groups['model']),
                 ('Copies', [v for key, values in groups.items() if key.startswith('copy_kind_') for v in values])]
        activity = sorted(groups['input'] or groups['copy_kind_1'] or groups['model'])
        regions = []
        for a,b in activity:
            if regions and a-regions[-1][1] <= 1_000_000:
                regions[-1] = (regions[-1][0], max(b,regions[-1][1]))
            else:
                regions.append((a,b))
        first = max(start, next((a for a,b in regions if b-a >= 1_000_000), activity[0][0])-50_000_000)
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), layout='constrained')
        for ax, begin, finish, scale, unit in [(axes[0],start,end,1e9,'s'), (axes[1],first,first+360_000_000,1e6,'ms')]:
            for i, (label, intervals) in enumerate(lanes):
                # Coalesce sub-10us gaps for the full-window rendering only; statistics above use exact intervals.
                visible=[]
                for a,b in sorted(intervals):
                    if a >= finish or b <= begin:
                        continue
                    a,b=max(a,begin),min(b,finish)
                    if scale == 1e9 and visible and a-visible[-1][1] <= 10_000:
                        visible[-1]=(visible[-1][0],max(b,visible[-1][1]))
                    else:
                        visible.append((a,b))
                ax.broken_barh([((a-begin)/scale,(b-a)/scale) for a,b in visible], (i-.3,.6),
                              facecolors=['#3274a1','#e1812c','#3a923a'][i])
            ax.set(yticks=list(range(len(lanes))), yticklabels=[v[0] for v in lanes], xlim=(0,(finish-begin)/scale),
                   xlabel=f'Time ({unit}), offset {(begin-start)/1e9:.3f} s from capture start')
            ax.grid(axis='x', alpha=.2)
        axes[0].set_title(args.label + ': full capture (gaps <10 us merged for display)')
        axes[1].set_title('360 ms detail around the first sustained input / copy activity')
        fig.savefig(args.timeline, dpi=180)
        fig.savefig(args.timeline.with_suffix('.pdf'))
        plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='top_kernels'},indent=2))

if __name__ == '__main__':
    main()
