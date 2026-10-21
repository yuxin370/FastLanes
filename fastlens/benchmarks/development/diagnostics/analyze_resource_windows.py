#!/usr/bin/env python3
"""Export the executed 16,384-image resource diagnostics, never primary throughput."""
import csv
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.diagnostics.analyze_resources import transfers
from fastlens.benchmarks.common import write_csv


def memory_row(directory, result):
    if result['profile'] or not result['memory_diagnostic'] or result['images'] != 16384 or not result['correctness_passed']:
        raise ValueError('Unprofiled memory window failed correctness/scope checks')
    source = json.loads((directory/'memory_samples.json').read_text())
    residency = []
    with (directory/'gpu_process_memory.csv').open() as stream:
        for row in csv.reader(stream):
            stamp, pid, _, mib = [value.strip() for value in row]
            if int(pid) != source['pid']:
                continue
            timestamp = int(datetime.datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').timestamp()*1e9)
            if source['start_unix_ns'] <= timestamp <= source['end_unix_ns']:
                residency.append(int(mib)*1024*1024)
    if not residency:
        raise ValueError(f'No process GPU observations inside measured interval: {directory}')
    rows = source['measurements']
    pinned = [r['pinned_allocated_bytes'] for r in rows if r['pinned_allocated_bytes'] is not None]
    native=[r['native_pools'] for r in rows if 'native_pools' in r]
    # Sum only mutually exclusive current counters from the same pool snapshot.
    native_resident=[r['device_pool']['in_use_bytes']+r['device_pool']['cached_bytes'] for r in native]
    native_pinned=[r['pinned_pool']['in_use_bytes']+r['pinned_pool']['cached_bytes'] for r in native]
    return dict(timestamp=result['timestamp'], git_commit=result['git_commit'], gpu=result['gpu'],
        model=result['model'], domain=result['domain'], pipeline=result['arm'], samples=result['images'],
        gpu_peak_process_bytes=max(residency), gpu_residency_samples=len(residency),
        torch_peak_allocated_bytes=source['torch_peak_allocated_bytes'],
        torch_peak_reserved_bytes=source['torch_peak_reserved_bytes'],
        native_peak_bytes=max(native_resident) if native_resident else None,
        native_pinned_peak_bytes=max(native_pinned) if native_pinned else None,
        host_rss_peak_bytes=max(r['host_rss_bytes'] for r in rows),
        process_tree_rss_peak_bytes=max(r['tree_rss_bytes'] for r in rows),
        pinned_peak_bytes=max(pinned) if pinned else None,
        raw_result=str((directory/'memory_samples.json').relative_to(ROOT)),
        gpu_samples=str((directory/'gpu_process_memory.csv').relative_to(ROOT)),
        notes='Unprofiled 16384-image window; 100ms sampling; native DevicePool current in-use+cached sampled together (not all native allocations); PyTorch/native pinned pools separate; tree RSS double-counts shared pages; no sum of independent peaks')


def main():
    h2d, memory = [], []
    for accepted in sorted((ROOT/'results/resources/raw/4090').glob('*/*/*/accepted.json')):
        directory = accepted.parent
        result = json.loads((directory/'result.json').read_text())
        if result['profile']:
            row, _ = transfers(directory/'capture.sqlite', result)
            row['domain'] = result['domain']
            h2d.append(row)
        else:
            memory.append(memory_row(directory,result))
    for name, rows in [('h2d',h2d),('memory',memory)]:
        if rows:
            write_csv(ROOT/f'results/resources/{name}.csv',rows,list(rows[0]))
    print(json.dumps(dict(h2d_captures=len(h2d), memory_windows=len(memory))))


if __name__ == '__main__':
    main()
