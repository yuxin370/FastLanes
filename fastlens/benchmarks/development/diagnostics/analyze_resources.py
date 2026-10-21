#!/usr/bin/env python3
"""Extract observed transfer/memory costs; profiler times never enter throughput tables."""
import bisect
import csv
import datetime
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.common import write_csv
from fastlens.benchmarks.profiling.analyze_training_nsys import _merge, _named_nvtx, _union_ns


def verified_repeat(directory):
    observed = json.loads((directory / 'pipeline.json').read_text())['repeats'][1]
    primary = ROOT / 'results/raw' / directory.parent.name / 'vitti/inference' / ('pipeline_' + directory.name + '.json')
    reference = json.loads(primary.read_text())['repeats'][1]
    if observed['images'] != 50000 or observed['sample_trace'] != reference['sample_trace']:
        raise ValueError(f'resource run sample coverage differs: {directory}')
    if observed['top1_predictions_sha256'] != reference['top1_predictions_sha256']:
        raise ValueError(f'resource run predictions differ: {directory}')
    return observed


def intersection(a, b):
    a, b = _merge(a), _merge(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def transfers(path, window_result=None):
    if window_result is None:
        verified_repeat(path.parent)
    elif not window_result['correctness_passed'] or window_result['images'] != 16384 or not window_result['profile']:
        raise ValueError('Resource window failed the sample-count/profile/correctness gate')
    connection = sqlite3.connect(path)
    windows = connection.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s "
        "ON n.textId=s.id WHERE COALESCE(n.text,s.value)='fastlens.measured_pass'").fetchall()
    (start, end), = windows
    stages = _named_nvtx(connection, start, end)
    forward = {}
    for name, a, b, tid in stages:
        if name == "model.forward":
            forward.setdefault(tid, []).append((a, b))
    for ranges in forward.values():
        ranges.sort()
    runtime = {correlation: (a, tid) for correlation, a, tid in connection.execute(
        "SELECT correlationId,start,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    model, other = [], []
    for a, b, correlation in connection.execute("SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", (start, end)):
        launch, tid = runtime.get(correlation, (-1, -1))
        ranges = forward.get(tid, [])
        index = bisect.bisect_right(ranges, (launch, float('inf'))) - 1
        is_model = index >= 0 and ranges[index][0] <= launch < ranges[index][1]
        (model if is_model else other).append((a, b))
    copies = connection.execute("SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?", (end, start)).fetchall()
    if any(a < start or b > end for a, b, _, _ in copies):
        raise ValueError(f"copy crosses capture window; cannot apportion bytes: {path}")
    h2d = [(a, b) for a, b, _, kind in copies if kind == 1]
    byte_count = sum(size for _, _, size, kind in copies if kind == 1)
    duration = sum(b - a for a, b in h2d) / 1e9
    cpu_input = [(a, b) for name, a, b, _ in stages if name in ("input.prepare", "input.handoff")]
    activity = model + other + [(a, b) for a, b, _, _ in copies]
    activity += connection.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>=? AND end<=?", (start, end)).fetchall()
    if window_result is None:
        memory = json.loads((path.parent / "memory_samples.json").read_text())
        metadata = json.loads(path.parent.with_suffix('.command.json').read_text())
        if "H100" not in memory['gpu'] and path.parts[-3] == 'H100':
            raise ValueError(f"wrong GPU: {path}")
        model_name = 'ViT-Ti'
    else:
        memory = dict(gpu=window_result['gpu'], pipeline=window_result['arm'], samples=window_result['images'])
        metadata = window_result
        model_name = window_result['model']
    row = dict(timestamp=metadata['timestamp'], git_commit=metadata['git_commit'],
        gpu=memory['gpu'], model=model_name, pipeline=memory['pipeline'], samples=memory['samples'],
        h2d_bytes=byte_count, h2d_bytes_per_image=byte_count / memory['samples'], h2d_copy_count=len(h2d),
        memcpy_cumulative_s=duration, effective_transfer_bytes_per_s=byte_count / duration if duration else None,
        cpu_input_submission_s=_union_ns(cpu_input) / 1e9,
        h2d_model_overlap_s=intersection(h2d, model) / 1e9,
        other_kernel_model_overlap_s=intersection(other, model) / 1e9,
        gpu_no_activity_s=((end - start) - _union_ns(activity)) / 1e9,
        exposed_model_wait_s=None, raw_result=str(path.relative_to(ROOT)),
        notes="Separate profiled warm pass/window; CPU scope is load/handoff only; model range includes forward/backward for training, excludes optimizer; unclassified kernels are not assumed to be preparation; no-work includes all causes; exposed input wait unmeasured")
    connection.close()
    return row, (start, end, cpu_input, h2d, model, other)


def memory_cost(path):
    source = json.loads(path.read_text())
    metadata = json.loads(path.parent.with_suffix('.command.json').read_text())
    if source['profiler_enabled']:
        raise ValueError('primary memory observations must be unprofiled')
    result = verified_repeat(path.parent)
    samples = source['measurements']
    residency = []
    with Path(source['gpu_samples']).open() as stream:
        for row in csv.reader(stream):
            if len(row) != 4 or row[2].strip() != str(source['pid']):
                continue
            stamp = int(datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp() * 1e9)
            if source['window']['start_unix_ns'] <= stamp <= source['window']['end_unix_ns']:
                residency.append(int(row[3].strip()) * 1024 * 1024)
    pinned = [r['torch_pinned_allocated_bytes'] for r in samples if r['torch_pinned_allocated_bytes'] is not None]
    native = result.get('native_totals', {})
    return dict(timestamp=metadata['timestamp'], git_commit=metadata['git_commit'],
        gpu=source['gpu'], model='ViT-Ti', pipeline=source['pipeline'], samples=source['samples'],
        gpu_peak_process_bytes=max(residency) if residency else None, gpu_residency_samples=len(residency),
        torch_peak_allocated_bytes=result['peak_torch_gpu_allocated_bytes'],
        torch_peak_reserved_bytes=result['peak_torch_gpu_reserved_bytes'],
        native_peak_bytes=native.get('actual_transient_total_allocated_high_water_bytes'),
        native_counter_scope='transient allocated workspace high-water; excludes other native allocations',
        host_rss_peak_bytes=max(r['host_rss_bytes'] for r in samples),
        process_tree_rss_peak_bytes=max(r['tree_rss_bytes'] for r in samples),
        pinned_peak_bytes=max(pinned) if pinned else None,
        native_pinned_peak_in_use_bytes=native.get('galp_native_pinned_peak_in_use_bytes'),
        raw_result=str(path.relative_to(ROOT)), gpu_samples=source['gpu_samples'],
        notes='Unprofiled second full pass; process residency and RSS sampled at 100ms; tree RSS includes shared pages repeatedly; pinned_peak_bytes is PyTorch allocator only; native pinned counter is separate; no sum of independent peaks')


def timeline(captures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [(p, data) for p, data in captures if p in ('pytorch', 'dali', 'ffcv', 'dct_major_pushdown')]
    if not selected:
        return
    figure, axes = plt.subplots(len(selected), 1, figsize=(12, 2.1 * len(selected)), squeeze=False)
    for ax, (pipeline, (start, end, cpu, h2d, model, other)) in zip(axes[:, 0], selected):
        left = start + (end - start) // 2
        right = min(left + 360_000_000, end)
        for y, events, color in [(0, cpu, '#e69f00'), (1, h2d, '#56b4e9'), (2, model, '#009e73'), (3, other, '#777777')]:
            intervals = [(max(a, left) - left, min(b, right) - max(a, left)) for a, b in _merge(events) if a < right and b > left]
            ax.broken_barh([(a / 1e6, b / 1e6) for a, b in intervals], (y - .35, .7), facecolors=color)
        ax.set_yticks(range(4), ['CPU load/handoff', 'H2D', 'Model kernels', 'Other kernels'])
        ax.set_xlim(0, (right - left) / 1e6)
        ax.set_title(pipeline, loc='left')
        ax.set_xlabel('Milliseconds from capture midpoint')
    figure.suptitle('ViT-Ti / H100: separate profiled warm passes (RGB/DCT deployment comparison)')
    figure.tight_layout()
    output = ROOT / 'results/figures'
    output.mkdir(exist_ok=True)
    figure.savefig(output / 'h2d_timeline.pdf')
    figure.savefig(output / 'h2d_timeline.png', dpi=180)
    plt.close(figure)


def main():
    h2d, captures = [], []
    for path in sorted((ROOT / 'results/profiles').glob('*/*/capture.sqlite')):
        row, intervals = transfers(path)
        h2d.append(row)
        if path.parts[-3] == 'H100':
            captures.append((row['pipeline'], intervals))
    if h2d:
        write_csv(ROOT / 'results/cost/h2d.csv', h2d, list(h2d[0]))
    memory = [memory_cost(path) for path in sorted((ROOT / 'results/memory').glob('*/*/memory_samples.json'))]
    if memory:
        write_csv(ROOT / 'results/cost/memory.csv', memory, list(memory[0]))
    timeline(captures)
    print(f'Exported {len(h2d)} transfer captures and {len(memory)} memory observations')


if __name__ == '__main__':
    main()
