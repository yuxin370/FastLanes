"""Attribute Nsight activities without treating API duration as GPU wait time."""
import bisect
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

from fastlens.benchmarks.development.reports.report_resident_study import write_csv


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def subtract(intervals, removed):
    result = []
    removed = union(removed)
    for start, end in union(intervals):
        for left, right in removed:
            if right <= start:
                continue
            if left >= end:
                break
            if left > start:
                result.append([start, left])
            start = max(start, right)
        if start < end:
            result.append([start, end])
    return result


def ids(text):
    return {k: int(v) for k, v in re.findall(r'\b(u|m|b)=(\d+)', text)}


def category(name, scope):
    lower = name.lower()
    if any(s in lower for s in ('randaugment', 'compute_stats', 'projected_stats')):
        return 'RandAugment'
    if 'normalize_mixup' in lower:
        return 'Fused normalization / Mixup (full or basic)'
    if 'mixup_targets' in lower:
        return 'Mixup targets'
    if any(s in lower for s in ('convert_to_int16', 'projected_to_int16')):
        return 'DCT conversion'
    if 'decompress' in lower:
        return 'Decompression'
    if any(s in lower for s in ('transformed_dct', 'round_dct', 'project_dct', 'gather_dct', 'materialize')):
        return 'DCT crop / resize / projection'
    if scope.startswith('model.forward'):
        return 'Model forward'
    if scope.startswith('model.backward'):
        return 'Model backward'
    if scope.startswith('optimizer.'):
        return 'Optimizer / gradient clipping'
    if scope.startswith('input.'):
        return 'Other input GPU operations'
    return 'Unattributed GPU operations'


def analyze(sqlite_path, output):
    output = Path(output)
    c = sqlite3.connect(sqlite_path)
    c.row_factory = sqlite3.Row
    strings = dict(c.execute('SELECT id,value FROM StringIds'))
    nvtx = [dict(r) for r in c.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS')]
    for r in nvtx:
        r['text'] = r['text'] or strings.get(r['textId'], '')
    window, = [r for r in nvtx if r['text'] == 'resident_study.measured_window']
    lo, hi = window['start'], window['end']
    scopes = defaultdict(list)
    for r in nvtx:
        if r['end'] and r['text'].startswith(('input.', 'model.', 'optimizer.')):
            scopes[r['globalTid']].append(r)
    for rows in scopes.values():
        rows.sort(key=lambda r: r['start'])
    starts = {tid: [r['start'] for r in rows] for tid, rows in scopes.items()}

    def enclosing(api):
        if api is None or api['globalTid'] not in scopes:
            return ''
        tid = api['globalTid']
        i = bisect.bisect_right(starts[tid], api['start']) - 1
        if i >= 0 and api['end'] <= scopes[tid][i]['end']:
            return scopes[tid][i]['text']
        return ''

    runtime = {r['correlationId']: dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')}
    raw_kernels = [dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    model_streams = {r['streamId'] for r in raw_kernels
                     if enclosing(runtime.get(r['correlationId'])).startswith('model.forward')}
    # Autograd launches on its worker thread, inside the main thread's backward range.
    backward = sorted((r for r in nvtx if r['end'] and r['text'].startswith('model.backward')),
                      key=lambda r: r['start'])
    backward_starts = [r['start'] for r in backward]
    kernels, operations, grouped, lanes = [], [], defaultdict(list), defaultdict(list)
    per_batch = defaultdict(list)
    for r in raw_kernels:
        if r['end'] <= lo or r['start'] >= hi:
            continue
        api = runtime.get(r['correlationId'])
        scope = enclosing(api)
        if not scope and api and r['streamId'] in model_streams:
            i = bisect.bisect_right(backward_starts, api['start']) - 1
            if i >= 0 and api['end'] <= backward[i]['end']:
                scope = backward[i]['text']
        name = strings[r['shortName']]
        kind = category(name, scope)
        r.update(name=name, category=kind, scope=scope,
                 api_start=api['start'] if api else None, api_end=api['end'] if api else None)
        kernels.append(r)
        operations.append(r)
        grouped[kind, name].append([max(lo, r['start']), min(hi, r['end'])])
        lane = ('Model / optimizer' if kind.startswith(('Model ', 'Optimizer ')) else
                'Other GPU work' if kind == 'Unattributed GPU operations' else 'GPU input processing')
        lanes[lane].append([r['start'], r['end']])
        if lane == 'GPU input processing' and api:
            lanes['Input producer CUDA submission'].append([api['start'], api['end']])
        if scope.startswith('model.forward'):
            per_batch[ids(scope)['b']].append(r)
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # Resident captures legitimately have no transfers.
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for row in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            r = dict(row)
            if r['end'] <= lo or r['start'] >= hi:
                continue
            api = runtime.get(r['correlationId'])
            r.update(api_start=api['start'] if api else None, api_end=api['end'] if api else None)
            operations.append(r)
            lanes['H2D' if r['copyKind'] == 1 else 'Other transfers'].append([r['start'], r['end']])
    for r in nvtx:
        if r['end'] and r['text'].startswith('input.request'):
            lanes['CPU input request / retrieval'].append([r['start'], r['end']])
    events = {}
    if 'CUPTI_ACTIVITY_KIND_CUDA_EVENT' in tables:
        for row in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
            r = dict(row)
            events[r['globalPid'], r['contextId'], r['eventId'], r['eventSyncId']] = r
    syncs = [dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')]
    stream_ops = defaultdict(list)
    for r in operations:
        if r['api_start'] is not None:
            stream_ops[r['streamId']].append(r)
    for rows in stream_ops.values():
        rows.sort(key=lambda r: r['api_start'])
    dependencies, confirmed = [], []
    for s in syncs:
        api = runtime.get(s['correlationId'])
        scope = enclosing(api)
        if not scope.startswith(('input.request', 'input.handoff')):
            continue
        if s['syncType'] == 4:
            # NativeWindow has an existing device barrier at pool boundaries.
            # Attribute only its trailing wait for known input kernels, after
            # all preceding non-input work has finished; do not count API cost.
            preceding = [r for r in kernels if r['api_start'] is not None and r['api_start'] < api['start']]
            data = [r['end'] for r in preceding if not r['category'].startswith(
                ('Model ', 'Optimizer ', 'Unattributed '))]
            other = [r['end'] for r in preceding if r['category'].startswith(
                ('Model ', 'Optimizer ', 'Unattributed '))]
            transfers = [r['end'] for r in operations if 'copyKind' in r and r['api_start'] is not None
                         and r['api_start'] < api['start']]
            start = max([lo, api['start'], *other, *transfers])
            end = min(api['end'], max(data, default=lo))
            if start < end:
                confirmed.append([start, end])
                batch = ids(scope)
                dependencies.append(dict(batch=batch['b'], microbatch=batch['m'], update=batch['u'],
                    consumer_stream=s['streamId'], event_id=None, event_sync_id=None,
                    wait_api_start_ns=api['start'], wait_api_end_ns=api['end'], producer_stream=None,
                    event_ready_ns=end, confirmed_start_ns=start, confirmed_end_ns=end,
                    reason='Input-request device barrier: trailing known input GPU work after non-input work completed'))
            continue
        if s['syncType'] != 2:
            continue
        batch = ids(scope)
        event = events.get((s['globalPid'], s['contextId'], s['eventId'], s['eventSyncId']))
        following = [r for r in per_batch[batch['b']] if r['streamId'] == s['streamId'] and r['api_start'] >= api['end']]
        row = dict(batch=batch['b'], microbatch=batch['m'], update=batch['u'],
            consumer_stream=s['streamId'], event_id=s['eventId'], event_sync_id=s['eventSyncId'],
            wait_api_start_ns=api['start'], wait_api_end_ns=api['end'],
            producer_stream=event['streamId'] if event else None,
            event_ready_ns=event['timestamp'] if event else None,
            confirmed_start_ns=None, confirmed_end_ns=None, reason='')
        if not event or not event['timestamp']:
            row['reason'] = 'Event completion unavailable (often recorded before capture)'
        elif not following:
            row['reason'] = 'No directly correlated model consumer on this stream'
        else:
            consumer = min(following, key=lambda r: r['api_start'])
            preceding = [r['end'] for r in stream_ops[s['streamId']] if r['api_start'] < api['start']]
            start = max([lo, api['end'], consumer['api_end'], *preceding])
            end = event['timestamp']
            # Other unresolved stream dependencies prevent exclusive attribution.
            competing = []
            for other in syncs:
                other_api = runtime.get(other['correlationId'])
                if (other['syncType'] == 2 and other['streamId'] == s['streamId'] and other_api
                        and other_api['start'] < consumer['api_start']
                        and other['eventSyncId'] != s['eventSyncId']):
                    ready = events.get((other['globalPid'], other['contextId'], other['eventId'], other['eventSyncId']))
                    if ready and ready['timestamp'] > start:
                        competing.append(ready['timestamp'])
                    elif ready is None and other_api['start'] >= max(preceding, default=lo):
                        competing.append(end)
            start = max([start, *competing])
            if start < end <= consumer['start']:
                row.update(confirmed_start_ns=start, confirmed_end_ns=end,
                           reason='Consumer queued; preceding stream work complete; input event pending')
                confirmed.append([start, end])
            else:
                row['reason'] = 'No positive input-only interval established'
        dependencies.append(row)
    confirmed = union(confirmed)
    lanes['Confirmed input dependency wait'] = confirmed
    model = union(lanes['Model / optimizer'])
    gaps = subtract([[model[0][0], model[-1][1]]], model) if model else []
    lanes['Other model gaps (not input-attributed)'] = subtract(gaps, confirmed)
    rows = [dict(category=k[0], kernel=k[1], calls=len(v), cumulative_gpu_ms=sum(b-a for a,b in v)/1e6,
                 interval_union_ms=sum(b-a for a,b in union(v))/1e6) for k,v in sorted(grouped.items())]
    write_csv(output / 'kernel_times.csv', rows)
    write_csv(output / 'input_dependencies.csv', dependencies)
    totals = []
    for kind in sorted({r['category'] for r in rows}):
        matching = [r for r in rows if r['category'] == kind]
        intervals = [p for (cat, name), values in grouped.items() if cat == kind for p in values]
        totals.append(dict(category=kind, calls=sum(r['calls'] for r in matching),
            cumulative_gpu_ms=sum(r['cumulative_gpu_ms'] for r in matching),
            interval_union_ms=sum(b-a for a,b in union(intervals))/1e6))
    write_csv(output / 'kernel_categories.csv', totals)
    first_fourth = [r['start'] for r in kernels if r['scope'].startswith('model.forward u=3 m=0 ')]
    summary = dict(window_start_ns=lo, window_end_ns=hi, profiled_window_ms=(hi-lo)/1e6,
        confirmed_input_wait_ms=sum(b-a for a,b in confirmed)/1e6,
        f_input_wait=sum(b-a for a,b in confirmed)/(hi-lo),
        input_dependencies=len(dependencies), unavailable_event_completions=sum(r['event_ready_ns'] is None for r in dependencies),
        other_model_gaps_ms=sum(b-a for a,b in lanes['Other model gaps (not input-attributed)'])/1e6,
        timeline_anchor_ns=min(first_fourth) if first_fourth else None,
        scope='Profiled training window only. Confirmed wait is a conservative attributable subset; unavailable dependencies are not zero waits.',
        accounting='Kernel cumulative times can overlap. Other model gaps include host scheduling, checks, data work and unattributed time; they are not all input stalls.',
        cpu_production='Input-producer CUDA submission is visible; CPU preparation before CUDA API submission is not attributed.',
        prefetch='Kernel accounting includes concurrent prefetch within the capture, possibly for samples consumed after its end.',
        categories=totals)
    (output / 'trace_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output / 'timeline_intervals.json').write_text(json.dumps({k:union(v) for k,v in lanes.items()})+'\n')
    c.close()
    return summary


def render_timelines(destination, paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lanes = ['CPU input request / retrieval', 'Input producer CUDA submission', 'H2D',
             'GPU input processing', 'Model / optimizer', 'Confirmed input dependency wait',
             'Other model gaps (not input-attributed)']
    colors = ['#777777', '#a6a075', '#4b87ad', '#53a78b', '#5663a1', '#cf654e', '#c5c5c5']
    fig, axes = plt.subplots(len(paths), 1, figsize=(13, 2.5*len(paths)), sharex=True, constrained_layout=True)
    for ax, (path, directory) in zip(axes, paths):
        summary = json.loads((directory/'trace_summary.json').read_text())
        intervals = json.loads((directory/'timeline_intervals.json').read_text())
        anchor = summary['timeline_anchor_ns']
        assert anchor is not None, 'Production timeline requires at least four updates'
        for i, (lane, color) in enumerate(zip(lanes, colors)):
            data = [(max(0,(a-anchor)/1e6), min(200,(b-anchor)/1e6)-max(0,(a-anchor)/1e6))
                    for a,b in intervals.get(lane,[]) if b>anchor and a<anchor+200_000_000]
            ax.broken_barh(data, (i-.35,.7), facecolors=color)
        ax.set_yticks(range(len(lanes)), lanes, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(path, loc='left', fontsize=10)
        ax.set_xlim(0,200)
        ax.grid(axis='x', alpha=.2)
    axes[-1].set_xlabel('Milliseconds from first forward kernel of update 4 (same scale for all paths)')
    fig.savefig(destination.with_suffix('.pdf'))
    fig.savefig(destination.with_suffix('.png'), dpi=170)
    plt.close(fig)
