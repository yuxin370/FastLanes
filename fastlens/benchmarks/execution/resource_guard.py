"""Block occupied target GPUs; annotate other GPU and host activity."""
import csv
import os
from pathlib import Path
import subprocess
import time

import psutil

def gpu_numa_nodes():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pci.bus_id',
                                      '--format=csv,noheader'], text=True)
    nodes = {}
    for uuid, address in csv.reader(output.splitlines()):
        domain, bus, slot = address.strip().lower().split(':')
        device = Path('/sys/bus/pci/devices') / f'{int(domain, 16):04x}:{bus}:{slot}'
        nodes[uuid.strip()] = int((device / 'numa_node').read_text())
    return nodes


def resident_model_window(process):
    """Recognize the inspected artifact's resident-tensor measured loop.

    Remote host placement alone cannot establish that another GPU on the same
    NUMA node is not issuing competing H2D copies.
    """
    command = process.cmdline()
    module = ['-m', 'fastlens.benchmarks.profiling.input_motivation']
    if not any(command[i:i+2] == module for i in range(len(command)-1)):
        return False
    if '--arm' not in command or command[command.index('--arm')+1] != 'model-only':
        return False
    if '--output' not in command:
        return False
    output = Path(command[command.index('--output')+1])
    if not output.is_absolute():
        output = Path(process.cwd())/output
    log = output/'run.log'
    return log.exists() and ': measuring ' in log.read_text()


class ResourceGuard:
    def __init__(self, gpu_uuid):
        self.gpu_uuid = gpu_uuid
        self.gpu_nodes = gpu_numa_nodes()
        self.numa_node = self.gpu_nodes[gpu_uuid]
        cpulist = Path(f'/sys/devices/system/node/node{self.numa_node}/cpulist').read_text().strip()
        self.local_cpus = set()
        for part in cpulist.split(','):
            bounds = part.split('-')
            self.local_cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        self.previous_io = {}
        self.previous_cpu = {}
        self.previous_cpu_time = None
        self.observations = []
        self.allowed_resident_pids = {int(p) for p in os.environ.get(
            'FASTLENS_ALLOWED_RESIDENT_PIDS', '').split(',') if p}

    def host_blockers(self, owned):
        """Detect CPU-only bulk jobs as well as GPU input workers.

        Two core-equivalents flags activity for later interpretation. Per user
        policy, unproven shared-resource contention does not block execution.
        """
        now = time.monotonic()
        previous = self.previous_cpu
        elapsed = None if self.previous_cpu_time is None else now-self.previous_cpu_time
        current = {}
        blocked = set()
        for process in psutil.process_iter(['pid','create_time','cpu_times','cpu_affinity'], ad_value=None):
            info = process.info
            if info['pid'] in owned or info['cpu_times'] is None:
                continue
            identity = (info['pid'],info['create_time'])
            cpu = info['cpu_times'].user+info['cpu_times'].system
            current[identity] = cpu
            if elapsed and identity in previous:
                cores = (cpu-previous[identity])/elapsed
                affinity = info['cpu_affinity']
                if cores > 2 and (affinity is None or self.local_cpus.intersection(affinity)):
                    self.observations.append(dict(pid=info['pid'],cpu_core_equivalents=cores,
                        cpu_affinity=affinity,reason='Potential shared-NUMA CPU load; annotated, not proof of contention'))
        self.previous_cpu,self.previous_cpu_time = current,now
        if elapsed is None:
            self.observations.append(dict(reason='Collecting initial host CPU activity interval'))
        return blocked

    def blockers(self, owned=()):
        out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                       '--format=csv,noheader,nounits'], text=True)
        blocked = set()
        self.observations = []
        activity = {}
        if self.allowed_resident_pids:
            monitor = subprocess.check_output(['nvidia-smi', 'pmon', '-s', 'um', '-c', '1'], text=True)
            for line in monitor.splitlines():
                fields = line.split()
                if len(fields) >= 5 and fields[0].isdigit() and fields[1].isdigit():
                    activity[int(fields[1])] = dict(sm_percent=fields[3], memory_percent=fields[4])
        for pid_text, gpu_text in csv.reader(out.splitlines()):
            pid, gpu = int(pid_text), gpu_text.strip()
            if pid in owned:
                continue
            observation = dict(pid=pid, gpu=gpu, timestamp=time.time())
            if gpu == self.gpu_uuid:
                # A loader/compiler child can start after the caller sampled owned PIDs.
                try:
                    process = psutil.Process(pid)
                    parents = [parent.pid for parent in process.parents()]
                    observation.update(command=process.cmdline(), parent_pids=parents,
                                       process_created=process.create_time())
                    if set(parents).intersection(owned):
                        observation.update(owned_descendant=True, blocking=False,
                                           reason='new child of this experiment')
                        self.observations.append(observation)
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                    observation['process_identity_unavailable'] = type(error).__name__
                if pid in self.allowed_resident_pids:
                    utilization = activity.get(pid, {})
                    active = any(value != '-' and float(value) > 0 for value in utilization.values())
                    observation.update(utilization=utilization, allowed_resident=True,
                        reason='authorized resident process has observed GPU activity' if active else
                        'user-authorized same-GPU resident; missing counters do not prove inactivity')
                    if active:
                        blocked.add(pid)
                else:
                    observation['reason'] = 'target GPU already in use'
                    blocked.add(pid)
            else:
                try:
                    process = psutil.Process(pid)
                    processes = [process, *process.children(recursive=True)]
                    affinity, snapshots, local_anon = set(), [], 0
                    for member in processes:
                        affinity.update(member.cpu_affinity())
                        for thread in member.threads():
                            affinity.update(os.sched_getaffinity(thread.id))
                        io = member.io_counters()
                        snapshots.append((member.pid, member.create_time(), io.read_bytes, io.write_bytes, io.read_chars))
                        local_anon += sum(int(field.split('=')[1]) for line in Path(f'/proc/{member.pid}/numa_maps').read_text().splitlines()
                                          if 'anon=' in line for field in line.split()
                                          if field.startswith(f'N{self.numa_node}='))
                    stamp = tuple(sorted(snapshots))
                    prior = self.previous_io.get(pid)
                    self.previous_io[pid] = stamp
                    isolated = not (affinity & self.local_cpus) and local_anon == 0 and prior == stamp
                    resident = self.gpu_nodes[gpu] != self.numa_node or resident_model_window(process)
                    isolated = isolated and resident
                    observation.update(cpu_affinity=sorted(affinity), target_numa_node=self.numa_node,
                                       target_numa_anon_pages=local_anon,
                                       io_snapshot=stamp, process_count=len(processes),
                                       independent_pcie_or_resident_window=resident,
                                       reason='remote NUMA; no observed input I/O' if isolated else 'shared or unverified host resources')
                except psutil.NoSuchProcess:
                    if not psutil.pid_exists(pid):
                        continue
                    observation['reason'] = 'worker changed during isolation check; recheck next interval'
                except (psutil.AccessDenied, PermissionError):
                    observation['reason'] = 'cannot verify remote process resource isolation'
                except (FileNotFoundError, ProcessLookupError):
                    observation['reason'] = 'process/thread changed during isolation check; recheck next interval'
            observation['blocking'] = pid in blocked
            self.observations.append(observation)
        blocked.update(self.host_blockers(owned))
        return blocked


def owned_pids():
    return {os.getpid(), *(p.pid for p in psutil.Process().children(recursive=True))}


def wait_for_input_resources(gpu_uuid):
    guard = ResourceGuard(gpu_uuid)
    while guard.blockers(owned_pids()):
        print('Waiting before measurement:', guard.observations, flush=True)
        time.sleep(2)
    return guard.observations
