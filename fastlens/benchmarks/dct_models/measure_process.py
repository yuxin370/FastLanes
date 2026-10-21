"""Measure a managed benchmark process, including native/DALI GPU allocations."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil

from fastlens.benchmarks.dct_models.gpu_occupancy import GpuCheckError, gpu_occupancy


class ProcessWindow:
    """Sample this process tree only while the benchmark window is open."""

    def __init__(self, interval=.1):
        self.root = psutil.Process()
        self.interval = interval
        self.initial, self.latest = {}, {}
        self.rss_peak = 0
        self.threads_peak = 0
        self.unavailable_io = set()
        self.finished = threading.Event()

    def sample(self):
        rss = threads = 0
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                cpu = process.cpu_times()
                resident = process.memory_info().rss
                threads += process.num_threads()
                values = dict(cpu_core_seconds=cpu.user+cpu.system)
                try:
                    io = process.io_counters()
                    values['physical_read_bytes'] = io.read_bytes
                except psutil.AccessDenied:
                    self.unavailable_io.add(process.pid)
                self.latest[process.pid] = values
                rss += resident
            except psutil.NoSuchProcess:
                # DataLoader children may exit between enumeration and sampling.
                continue
        self.rss_peak = max(self.rss_peak, rss)
        self.threads_peak = max(self.threads_peak, threads)

    def start(self):
        self.sample()
        self.initial = {pid: values.copy() for pid, values in self.latest.items()}
        self.thread = threading.Thread(target=self._sample_until_finished, daemon=True)
        self.thread.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.result = self.stop()

    def _sample_until_finished(self):
        while not self.finished.wait(self.interval):
            self.sample()

    def stop(self):
        self.finished.set()
        self.thread.join()
        self.sample()
        totals = {key: sum(values.get(key, 0)-self.initial.get(pid, {}).get(key, 0)
                           for pid, values in self.latest.items())
                  for key in ('cpu_core_seconds', 'physical_read_bytes')}
        if self.unavailable_io:
            totals['physical_read_bytes'] = None
        return dict(totals, tree_rss_peak_bytes=self.rss_peak, tree_threads_peak=self.threads_peak,
                    sampling_interval_s=self.interval,
                    scope='measured window; process tree CPU and storage reads; sampled RSS sum double-counts shared pages',
                    unavailable_io_pids=sorted(self.unavailable_io))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', help='check other-user MPS clients and non-MPS processes outside this benchmark')
    p.add_argument('--rss-limit-mib', type=int, help='invalidate a run exceeding this sampled process-tree RSS sum')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.rss_limit_mib is not None and args.rss_limit_mib < 1:
        p.error('rss-limit-mib must be positive')
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
    mps_servers, own_mps_clients, mps_foreign = set(), set(), set()
    extra_gpu_processes = {}
    gpu_check_error = None
    pending_gpu_text = ""
    started = time.perf_counter()
    root = psutil.Process(child.pid)
    with gpu_path.open('w') as gpu, gpu_path.open() as gpu_samples:
        fields = 'pid,used_gpu_memory'+(',gpu_uuid' if args.gpu_uuid else '')
        monitor = subprocess.Popen(['nvidia-smi', '--query-compute-apps='+fields,
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
                if args.gpu_uuid and gpu_check_error is None:
                    if monitor.poll() is not None:
                        gpu_check_error = 'nvidia-smi process monitoring stopped unexpectedly'
                        print('GPU occupancy check failed; this run is invalid: '+gpu_check_error, flush=True)
                    lines = (pending_gpu_text + gpu_samples.read()).split('\n')
                    pending_gpu_text = lines.pop()
                    sampled = [int(parts[0]) for line in lines if len(parts := line.split(',')) == 3
                               and parts[0].strip().isdigit() and parts[2].strip() == args.gpu_uuid]
                    for pid in set(sampled) - known_pids - extra_gpu_processes.keys():
                        try:
                            process = psutil.Process(pid)
                            extra_gpu_processes[pid] = dict(status='observed', uid=process.uids().real,
                                ppid=process.ppid(), name=process.name(), cmdline=process.cmdline())
                        except psutil.NoSuchProcess:
                            extra_gpu_processes[pid] = dict(status='gone')
                        except psutil.AccessDenied:
                            extra_gpu_processes[pid] = dict(status='access_denied')
                    if sampled:
                        try:
                            occupancy = gpu_occupancy(args.gpu_uuid, processes=sampled, own_pids=known_pids)
                            mps_servers.update(occupancy['mps_servers'])
                            own_mps_clients.update(occupancy['own_mps_client_pids'])
                            mps_foreign.update(occupancy['blocking_pids'])
                        except GpuCheckError as error:
                            gpu_check_error = str(error)
                            print('GPU occupancy check failed; this run is invalid: '+gpu_check_error, flush=True)
                time.sleep(.1)
            returncode = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            monitor.terminate()
            monitor.wait()
    gpu_lines = gpu_path.read_text().splitlines()
    gpu_peak = max((int(line.split(',')[1]) for line in gpu_lines
                    if line.split(',')[0].strip().isdigit() and int(line.split(',')[0]) in known_pids
                    and line.split(',')[1].strip().isdigit()), default=None)
    foreign = sorted(({int(parts[0]) for line in gpu_lines if len(parts := line.split(',')) == 3
                       and parts[0].strip().isdigit() and parts[2].strip() == args.gpu_uuid} | mps_foreign)
                     - (known_pids | mps_servers | own_mps_clients)) if args.gpu_uuid else []
    over_memory = args.rss_limit_mib is not None and peaks['tree_rss_sum_bytes'] > args.rss_limit_mib*1024**2
    valid = returncode == 0 and not foreign and not over_memory and gpu_check_error is None
    result = dict(command=command, returncode=returncode, valid=valid,
        foreign_gpu_pids=foreign, rss_limit_mib=args.rss_limit_mib, memory_limit_exceeded=over_memory,
        gpu_uuid=args.gpu_uuid, process_wall_seconds=time.perf_counter()-started,
        gpu_check_error=gpu_check_error, mps_server_pids=sorted(mps_servers),
        extra_gpu_processes=extra_gpu_processes,
        same_uid_mps_client_pids=sorted(own_mps_clients),
        gpu_sharing_policy='MPS clients of other UIDs block; same-UID MPS clients allowed; non-MPS processes outside the benchmark tree block',
        nvml_process_peak_mib=gpu_peak, cpu_peaks=peaks,
        sampled_tree_io={k:sum(v[k] for v in io.values()) for k in ('read_bytes','write_bytes','read_chars')},
        unavailable_io_pids=sorted(unavailable_io),
        sampling_interval_seconds=.1, benchmark_pid=child.pid,
        scope='process lifetime including imports, initialization, compilation and validation; benchmark E2E remains in its own JSON',
        memory_semantics='NVML process peak includes non-Torch allocations when attributable to the benchmark; shared MPS-server memory is not attributed; summed tree RSS double-counts shared pages, not PSS',
        io_semantics='sampled OS process counters, last samples can miss shutdown I/O; read_bytes is storage I/O, read_chars includes cache/IPC')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if not valid:
        print(f'Invalid run: returncode={returncode}, foreign_gpu_pids={foreign}, memory_limit_exceeded={over_memory}, gpu_check_error={gpu_check_error}', flush=True)
    raise SystemExit(returncode if returncode else (0 if valid else 1))


if __name__ == '__main__':
    main()
