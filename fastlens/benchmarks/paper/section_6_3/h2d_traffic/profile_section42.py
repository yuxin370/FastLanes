"""Instrument the existing inference timer boundaries without changing workload code.

Only the listed evaluators are instrumented. Their command, warmup, batches,
model, predictions and timed loop are otherwise unchanged. Diagnostic timings
must not replace the paper's unprofiled throughput measurements.
"""
import argparse
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

# Exact, unique anchors at the existing complete-pass timer boundaries.
BOUNDARIES = {
    'fastlens/benchmarks/paper/shared/remaining_baseline_inference.py': (
        '        started = time.perf_counter()',
        '        seconds = time.perf_counter()-started'),
    'fastlens/benchmarks/paper/shared/swin_unified_inference.py': (
        '        started = time.perf_counter()',
        '        seconds = time.perf_counter() - started'),
    'fastlens/benchmarks/paper/shared/remaining_baseline_training.py': (
        '    started = time.perf_counter()',
        '    seconds = time.perf_counter() - started'),
    'fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py': (
        '    started = time.perf_counter()',
        '    seconds = time.perf_counter() - started'),
    'fastlens/benchmarks/paper/section_6_3/h2d_traffic/swin_block_major.py': (
        '        start = time.perf_counter()',
        '        elapsed = time.perf_counter() - start'),
    'fastlens/benchmarks/dct_models/evaluate.py': (
        '    start = time.perf_counter()',
        '    wall = time.perf_counter() - start'),
    'fastlens/benchmarks/dct_models/evaluate_rgb.py': (
        '        start = time.perf_counter()',
        '        wall = time.perf_counter()-start'),
    'fastlens/benchmarks/dct_models/evaluate_shards.py': (
        '    started = time.perf_counter()',
        '    wall = time.perf_counter() - started'),
    'fastlens/benchmarks/system_dct_major/pipeline.py': (
        '        repeat_started_ns = time.perf_counter_ns()',
        '        adapter.end_repeat()'),
    'fastlens/benchmarks/baselines/system_rgbnomore/inference/pipeline.py': (
        '        repeat_started = time.perf_counter()',
        '        repeat_records.append(record)'),
}


def io_snapshot():
    process = psutil.Process()
    rows = []
    for member in [process, *process.children(recursive=True)]:
        try:
            rows.append(dict(pid=member.pid, read_bytes=member.io_counters().read_bytes))
        except psutil.NoSuchProcess:
            pass
    return rows


def instrument(source, filename):
    relative = str(Path(filename).resolve().relative_to(ROOT))
    if relative == 'fastlens/benchmarks/paper/shared/training_window_rgb.py' and mode == 'cold':
        # A cold-start diagnostic must not force a fully resident input cache.
        first = source.index('    cache_before = None if args.model_only or args.check else cache_pages(paths)')
        last = source.index('    updates = math.ceil(args.measured_images / 1024)', first)
        source = source[:first] + '    cache_before = cache_pages(paths)\n    cache_rewarm = None\n' + source[last:]
        source = source.replace("cache_mode='warm'", "cache_mode='cold-start'")
        source = source.replace('Warming compressed source page cache', 'Checking compressed input page residency')
        source = source.replace('Warming input representation page cache', 'Checking input page residency')
        return source
    if relative not in BOUNDARIES:
        return source
    start, end = BOUNDARIES[relative]
    lines = source.splitlines()
    assert lines.count(start) == lines.count(end) == 1, relative
    result = []
    for line in lines:
        indent = line[:len(line)-len(line.lstrip())]
        if line == start:
            result.append(indent + "__import__('fastlens.benchmarks.paper.section_6_3.h2d_traffic.profile_section42', fromlist=['begin']).begin()")
        result.append(line)
        if line == end:
            result.append(indent + "__import__('fastlens.benchmarks.paper.section_6_3.h2d_traffic.profile_section42', fromlist=['end']).end()")
    return '\n'.join(result) + '\n'


def begin():
    global memory, started
    import torch
    assert not started, 'Expected exactly one full inference pass'
    started = True
    torch.cuda.synchronize()
    if mode == 'profile':
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('fastlens.measured_pass')
    else:
        from fastlens.benchmarks.paper.shared.window_memory import WindowMemory
        memory = WindowMemory(output)
        memory.begin()


def end():
    global finished
    import torch
    assert started and not finished
    torch.cuda.synchronize()
    if mode == 'profile':
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    else:
        memory.end()
    finished = True


def main():
    global mode, output, started, finished
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', type=Path, required=True)
    parser.add_argument('--mode', choices=['memory', 'profile', 'cold'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-paths', type=Path)
    args = parser.parse_args()
    mode, output, started, finished = args.mode, args.output, False, False
    sys.modules['fastlens.benchmarks.paper.section_6_3.h2d_traffic.profile_section42'] = sys.modules[__name__]
    command = json.loads(args.command.read_text())
    if mode == 'cold':
        from fastlens.benchmarks.profiling import input_motivation
        paths = [Path(p) for p in json.loads(args.input_paths.read_text())]
        # DONTNEED is advisory: two observed runs retained pages after one pass.
        # Retry eviction before the counters/timer start, never accept a warm page.
        for attempt in range(3):
            for path in paths:
                with path.open('rb') as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            cache = input_motivation.cache_pages(paths)
            cache['eviction_attempts'] = attempt + 1
            if cache['resident_pages'] == 0:
                break
            if attempt < 2:
                time.sleep(1)
        (output / 'cold_cache.json').write_text(json.dumps(cache, indent=2)+'\n')
        assert cache['resident_pages'] == 0, 'Input pages remain resident; not a cold start'
        # Keep the actual adapter/cache-fill and model warmup, but omit the
        # artificial file-cache preloading used by the throughput experiment.
        original_cache_pages = input_motivation.cache_pages
        input_motivation.cache_pages = lambda paths, warm=False: original_cache_pages(paths, warm=False)
        cold_before = io_snapshot()
    original = importlib.machinery.SourceFileLoader.get_code

    def get_code(loader, fullname):
        path = Path(loader.path).resolve()
        if path.is_relative_to(ROOT) and str(path.relative_to(ROOT)) in BOUNDARIES:
            return compile(instrument(path.read_text(), path), str(path), 'exec')
        return original(loader, fullname)

    importlib.machinery.SourceFileLoader.get_code = get_code
    if command[0] == '-m':
        sys.argv = command[1:]
        runpy.run_module(command[1], run_name='__main__')
    elif command[0] == '-c':
        sys.argv = ['-c', *command[2:]]
        exec(compile(command[1], '<original command>', 'exec'), {'__name__': '__main__'})
    else:
        script = ROOT / command[0]
        sys.argv = command
        exec(compile(instrument(script.read_text(), script), str(script), 'exec'),
             {'__name__': '__main__', '__file__': str(script)})
    training = command[0] == 'fastlens/benchmarks/paper/shared/training_window_rgb.py'
    if mode == 'cold':
        cold_after = io_snapshot()
        window = json.loads((output / 'memory_samples.json').read_text())
        total = lambda rows: sum(row['read_bytes'] for row in rows)
        record = dict(io_before=cold_before, io_after=cold_after,
            startup_storage_read_bytes=total(window['io_before'])-total(cold_before),
            measured_window_storage_read_bytes=window['process_tree_storage_read_bytes'],
            after_window_storage_read_bytes=total(cold_after)-total(window['io_after']),
            total_storage_read_bytes=total(cold_after)-total(cold_before),
            scope='Cold input payload pages verified before workload launch, after diagnostic imports; process-tree storage reads include model setup, actual adapter cache fill and warmup, measured pass/window and cleanup; not dataset-file-only IO or a full training epoch')
        (output / 'cold_start_io.json').write_text(json.dumps(record, indent=2)+'\n')
    if not training:
        assert started and finished, 'The workload did not execute the expected resource window'


if __name__ == '__main__':
    main()
