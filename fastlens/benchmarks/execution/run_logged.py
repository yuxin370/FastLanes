#!/usr/bin/env python3
"""Run one experiment and preserve its command, environment and raw output."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[0] == "--" else args.command
    args.log.parent.mkdir(parents=True, exist_ok=True)
    revision = os.environ.get('FASTLENS_EVALUATION_SNAPSHOT')
    fingerprint = None
    if revision:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from fastlens.benchmarks.execution.revision_results import verify_snapshot
        fingerprint = verify_snapshot()
    record = dict(command=command, shell_command=shlex.join(command), cwd=os.getcwd(),
                  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host_load_average_start=os.getloadavg(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  code_fingerprint=fingerprint,
                  environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "TMPDIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DCTNET_PROFILE", "FASTLENS_EVALUATION_SNAPSHOT", "FASTLENS_ALLOWED_RESIDENT_PIDS")})
    metadata = args.log.with_suffix(".command.json")
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    start = time.monotonic()
    with args.log.open("x") as stream:
        with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT) as result:
            # Reap the measurement before the caller cleans its private caches.
            # The monitor sends SIGTERM to this wrapper and its process group.
            previous = signal.signal(signal.SIGTERM, lambda *_: result.terminate())
            try:
                result.wait()
            finally:
                signal.signal(signal.SIGTERM, previous)
    record.update(returncode=result.returncode, process_wall_s=time.monotonic() - start,
                  host_load_average_end=os.getloadavg())
    if revision:
        try:
            verify_snapshot()
            record['code_unchanged_during_run'] = True
        except RuntimeError as error:
            record.update(code_unchanged_during_run=False, code_change_error=str(error), returncode=86)
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    raise SystemExit(record['returncode'])


if __name__ == "__main__":
    main()
