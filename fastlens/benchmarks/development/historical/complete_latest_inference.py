"""Monitor idle H100 windows and finish only five outstanding inference checks."""
import datetime
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical import recheck_latest_inference as latest
from fastlens.benchmarks.development.historical import recheck_remaining_inference as replay

OUT = latest.OUT
replay.OUT = OUT
plan = latest.read(OUT / 'plan.json')
jobs = [j for j in plan['jobs'] if j['gpu'] == 'H100' and j['profile'] in
        ('mobilenet24', 'resnet64', 'efun', 'swinv2', 'vitti')]
results_path = OUT / 'monitored_results.csv'
observations = latest.rows(results_path) if results_path.exists() else []
jobs = [j for j in jobs if j['reference']['model'] not in {r['model'] for r in observations}]
failures = []


def status(active):
    completed = sum(r['model'] != 'MobileNetV2-DCT24' for r in observations)
    latest.save(OUT / 'status.json', dict(updated=datetime.datetime.now().astimezone().isoformat(),
        pid=os.getpid(), controller_running=bool(active), status='RUNNING_OR_WAITING' if active else ('COMPLETE_WITH_CAVEATS' if len(observations) == 5 else 'INCOMPLETE'),
        accepted=10 + completed, total=14, active=active, followup_failed=failures,
        anomalous_first_pass_retained=True,
        anomaly_rechecked=any(r['model'] == 'MobileNetV2-DCT24' for r in observations),
        notes='First H100 MobileNet24 pass remains in the raw table; all H100 passes have same-GPU VLLM residency caveat.'))


while jobs:
    job = jobs.pop(0)
    previous = [int(re.match(r'trial_(\d+)', p.name).group(1))
                for p in (OUT / 'inference' / job['id']).glob('trial_*')]
    trial = max(previous) + 1
    directory = OUT / 'inference' / job['id'] / f'trial_{trial}'
    status(job['id'] + ':waiting_for_120s_idle')
    # VLLM resumed between short idle observations. Require a longer idle interval,
    # while retaining the normal per-process activity check during each run.
    quiet_since = None
    with (OUT / 'monitor_idle_trace.jsonl').open('a') as trace:
        while True:
            utilization = int(subprocess.check_output(['nvidia-smi', '-i', '1',
                '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
            now = time.monotonic()
            trace.write(json.dumps(dict(timestamp=time.time(), gpu_utilization_percent=utilization, job=job['id'])) + '\n')
            trace.flush()
            quiet_since = (quiet_since if quiet_since is not None else now) if utilization == 0 else None
            status(job['id'] + ':waiting_for_120s_idle')
            if quiet_since is not None and now - quiet_since >= 120:
                break
            time.sleep(10)
    status(job['id'])
    try:
        replay.execute(job, trial)
        row = latest.read(directory / 'accepted.json')
        row['implementation_cohort'] = 'native_' + plan['native']['sha256'][:16]
        row['notes'] = ('Monitored follow-up after a contended or anomalous pass; historical evidence retained. '
                        'VLLM PID 4037300 remains resident on H100; unavailable pmon counters do not prove inactivity. '
                        'Two minutes of observed GPU idle before launch; not an exclusive-GPU result.')
        for field in ('model_only_images_per_s', 'storage_bytes', 'metadata_bytes', 'issued_read_bytes',
                      'h2d_bytes', 'h2d_copy_count', 'gpu_peak_process_bytes', 'torch_peak_allocated_bytes',
                      'torch_peak_reserved_bytes', 'native_peak_bytes', 'host_rss_peak_bytes', 'pinned_peak_bytes'):
            row[field] = ''
        latest.save(directory / 'accepted.json', row)
        observations.append(dict(gpu=row['gpu'], model=row['model'], samples=50000,
            images_per_s=row['images_per_s'], elapsed_s=row['elapsed_s'],
            previous_median_images_per_s=job['previous_median'],
            change_percent=100 * (row['images_per_s'] / job['previous_median'] - 1),
            raw_result=row['raw_result'], correctness=row['correctness'], notes=row['notes']))
        latest.write_csv(results_path, observations, list(observations[0]))
    except (AssertionError, RuntimeError, subprocess.CalledProcessError) as error:
        directory.mkdir(parents=True, exist_ok=True)
        latest.save(directory / 'failure.json', dict(error=str(error), type=type(error).__name__))
        failures.append(job['id'])
        print('FOLLOWUP FAILED', job['id'], repr(error), flush=True)
        if isinstance(error, RuntimeError) and str(error).startswith('Target GPU contention:'):
            jobs.append(job)  # Observed transient competition: keep monitoring instead of exiting.
        else:
            status(None)
            raise  # Correctness or implementation failures require investigation.
status(None)
