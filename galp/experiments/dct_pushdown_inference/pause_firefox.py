"""Temporarily pause the explicitly authorized browser tree; always resume it.

Run with sudo when the browser belongs to another account. The experiment
controller creates OUTPUT/resume to release the pause; Ctrl-C also resumes it.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firefox", type=int, required=True)
    parser.add_argument("--xvfb", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "firefox" not in Path(f"/proc/{args.firefox}/comm").read_text().lower():
        raise ValueError("The authorized Firefox PID no longer identifies Firefox")
    if Path(f"/proc/{args.xvfb}/comm").read_text().strip() != "Xvfb":
        raise ValueError("The authorized Xvfb PID no longer identifies Xvfb")
    if (args.output / "resume").exists():
        raise ValueError("Use a fresh output directory for this pause")
    rows = [tuple(map(int, line.split())) for line in
            subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()]
    pids = {args.firefox, args.xvfb}
    while True:
        children = {pid for pid, parent in rows if parent in pids}
        if children <= pids:
            break
        pids |= children
    stopped = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    started = time.time()
    try:
        for pid in sorted(pids):
            try:
                token = identity(pid)
                os.kill(pid, signal.SIGSTOP)
                stopped[pid] = token
            except ProcessLookupError:
                continue
        state = dict(state="paused", pids=list(stopped), started=started)
        (args.output / "browser_state.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        # Bound the interruption even if the experiment controller disappears.
        deadline = time.monotonic() + 1800
        while not (args.output / "resume").exists() and time.monotonic() < deadline:
            time.sleep(0.5)
    finally:
        for pid, token in stopped.items():
            try:
                if identity(pid) == token:
                    os.kill(pid, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError):
                continue
        state = dict(state="resumed", pids=list(stopped), started=started, ended=time.time())
        (args.output / "browser_state.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
