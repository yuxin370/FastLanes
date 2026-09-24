"""Measure a managed benchmark process, including native/DALI GPU allocations."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command:
        p.error('a benchmark command is required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gpu_path = args.output.with_suffix('.gpu.csv')
    child = subprocess.Popen(command, start_new_session=True)
    known_pids, io, peaks = {child.pid}, {}, dict(main_rss_bytes=0, tree_rss_sum_bytes=0)
    unavailable_io = set()
    started = time.perf_counter()
    root = psutil.Process(child.pid)
    with gpu_path.open('w') as gpu:
        monitor = subprocess.Popen(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits', '-lms', '100'], stdout=gpu)
        try:
            while child.poll() is None:
                try:
                    processes = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    processes = []
                rss = 0
                for process in processes:
                    try:
                        resident = process.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                    known_pids.add(process.pid)
                    rss += resident
                    if process.pid not in unavailable_io:
                        try:
                            counters = process.io_counters()
                            io[process.pid] = dict(read_bytes=counters.read_bytes, write_bytes=counters.write_bytes,
                                                  read_chars=counters.read_chars)
                        except psutil.NoSuchProcess:
                            pass
                        except psutil.AccessDenied:
                            # Some driver utilities protect /proc/PID/io even from their parent.
                            unavailable_io.add(process.pid)
                    if process.pid == child.pid:
                        peaks['main_rss_bytes'] = max(peaks['main_rss_bytes'], resident)
                peaks['tree_rss_sum_bytes'] = max(peaks['tree_rss_sum_bytes'], rss)
                time.sleep(.1)
            returncode = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            monitor.terminate()
            monitor.wait()
    gpu_peak = max((int(line.split(',')[1]) for line in gpu_path.read_text().splitlines()
                    if line.split(',')[0].strip().isdigit() and int(line.split(',')[0]) in known_pids
                    and line.split(',')[1].strip().isdigit()), default=None)
    result = dict(command=command, returncode=returncode, process_wall_seconds=time.perf_counter()-started,
        nvml_process_peak_mib=gpu_peak, cpu_peaks=peaks,
        sampled_tree_io={k:sum(v[k] for v in io.values()) for k in ('read_bytes','write_bytes','read_chars')},
        unavailable_io_pids=sorted(unavailable_io),
        sampling_interval_seconds=.1, benchmark_pid=child.pid,
        scope='process lifetime including imports, initialization, compilation and validation; benchmark E2E remains in its own JSON',
        memory_semantics='NVML process peak includes non-Torch allocations; summed tree RSS double-counts shared pages, not PSS',
        io_semantics='sampled OS process counters, last samples can miss shutdown I/O; read_bytes is storage I/O, read_chars includes cache/IPC')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
