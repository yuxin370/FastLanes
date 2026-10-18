"""GPU ownership checks shared by the experiment queue and its monitor."""
import os
from pathlib import Path
import subprocess

import psutil


class GpuCheckError(RuntimeError):
    """The active clients could not be established reliably."""


def _query(command, *, input=None):
    try:
        result = subprocess.run(command, input=input, text=True, capture_output=True,
                                check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise GpuCheckError(f"GPU occupancy check failed: {error}") from error
    if result.stderr.strip():
        raise GpuCheckError(f"GPU occupancy check failed: {result.stderr.strip()}")
    return result.stdout


def _control_pids(command):
    output = _query(["nvidia-cuda-mps-control"], input=command+"\n")
    values = output.split()
    if any(not value.isdecimal() or int(value) < 1 for value in values):
        raise GpuCheckError(f"Unexpected MPS reply to {command!r}: {output!r}")
    return sorted(set(map(int, values)))


def gpu_occupancy(gpu_uuid, *, processes=None, own_pids=()):
    """Allow same-UID MPS clients; exclude only our own non-MPS process tree.

    The monitor can supply PIDs from its existing NVML stream rather than
    launching another nvidia-smi process at every sample.
    """
    if processes is None:
        output = _query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                         "--format=csv,noheader,nounits"])
        processes = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or not fields[1].isdecimal():
                raise GpuCheckError(f"Unexpected GPU process reply: {line!r}")
            if fields[0] == gpu_uuid:
                processes.append(int(fields[1]))
    servers, non_mps, clients = [], set(), []
    for pid in sorted(set(processes)-set(own_pids)):
        try:
            command = psutil.Process(pid).cmdline()
        except psutil.NoSuchProcess:
            # Keep a short-lived unclassified compute process as interference.
            non_mps.add(pid)
            continue
        except psutil.AccessDenied as error:
            raise GpuCheckError(f"Cannot identify GPU process {pid}: {error}") from error
        if command and Path(command[0]).name == "nvidia-cuda-mps-server":
            servers.append(pid)
        else:
            non_mps.add(pid)
    if servers:
        registered = _control_pids("get_server_list")
        if not set(servers).issubset(registered):
            raise GpuCheckError("Visible MPS server is absent from get_server_list; check CUDA_MPS_PIPE_DIRECTORY/permissions")
        for server in servers:
            command = f"get_client_list {server}"
            for pid in _control_pids(command):
                try:
                    uid = psutil.Process(pid).uids().real
                except psutil.NoSuchProcess as error:
                    # A client can exit between the control reply and /proc.
                    # Confirm its departure; an unresolved namespace/PID fails.
                    if pid not in _control_pids(command):
                        continue
                    raise GpuCheckError(f"Cannot resolve active MPS client PID {pid}") from error
                except psutil.AccessDenied as error:
                    raise GpuCheckError(f"Cannot read UID of MPS client {pid}: {error}") from error
                clients.append(dict(server_pid=server, pid=pid, uid=uid))
    own_clients = {client["pid"] for client in clients if client["uid"] == os.getuid()}
    foreign_clients = {client["pid"] for client in clients if client["uid"] != os.getuid()}
    return dict(blocking_pids=sorted((non_mps-own_clients) | foreign_clients),
                mps_servers=servers, mps_clients=clients,
                own_mps_client_pids=sorted(own_clients))
