"""One version-pinned FastLens pass per existing model/GPU; retain old results."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical import recheck_remaining_inference as replay
from fastlens.benchmarks.execution.revision_results import capture
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.system_dct_major.common import file_identity

OUT = ROOT / 'results/latest_inference_20260930'
replay.OUT = OUT
read, save, rows, write_csv = replay.read, replay.save, replay.rows, replay.write_csv


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'plan.json').exists():
        return read(OUT / 'plan.json')
    source = capture()
    native = ROOT / source['native_path']
    pinned = OUT / 'native_runtime' / native.name
    pinned.parent.mkdir(exist_ok=True)
    shutil.copy2(native, pinned)
    assert file_sha(pinned) == source['native_sha256'] == file_sha(native)
    save(OUT / 'source_snapshot.json', source)
    (OUT / 'source_changes.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', 'galp'], cwd=ROOT))
    artifact = dict(path=str(pinned), sha256=file_sha(pinned), size_bytes=pinned.stat().st_size,
                    file_identity=file_identity(pinned))
    files = {p: sha for p, sha in source['files'].items()
             if p.endswith('.py') and '/tests/' not in p and '/training/' not in p
             and 'training_window' not in p}
    for path in [pinned, Path(__file__), ROOT / 'fastlens/benchmarks/development/historical/recheck_remaining_inference.py',
                 ROOT / 'fastlens/benchmarks/execution/inference_load.py', ROOT / 'fastlens/benchmarks/execution/resource_guard.py']:
        files[str(path.relative_to(ROOT))] = file_sha(path)
    snapshot = dict(files=files, native_sha256=artifact['sha256'], native_path=str(pinned.relative_to(ROOT)),
                    fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    save(OUT / 'runtime_snapshot.json', snapshot)
    jobs = []
    for summary in rows(ROOT / 'results/p0/deployment_inference_summary.csv'):
        if summary['pipeline'] != 'FastLens':
            continue
        source_dir = (ROOT / summary['raw_results'].split(';')[0].split('#')[0]).parent
        ref = read(source_dir / 'accepted.json')
        metadata = read(source_dir / 'run.command.json')
        command = metadata['command']
        contract = read(command[command.index('--contract') + 1]) if '--contract' in command else None
        if contract:
            for config in contract['pipelines'].values():
                if not isinstance(config, dict):
                    continue
                if 'torch_binding_dir' in config:
                    config['torch_binding_dir'] = str(pinned.parent)
                for field in ('torch_binding_artifact', 'native_binary_fingerprint'):
                    if field in config:
                        config[field] = artifact
        profile = replay.PROFILES[ref['model']]
        gpu = '4090' if ref['gpu'] == 'RTX 4090' else 'H100'
        jobs.append(dict(id=f'{gpu}_{profile}_fastlens', gpu=gpu, profile=profile,
                         reference=ref, command=command, contract=contract,
                         environment={k: v for k, v in metadata['environment'].items() if v is not None},
                         previous_median=float(summary['median_images_per_s']),
                         previous_raw=summary['raw_results']))
    order = ['mobilenet24', 'mobilenet32', 'resnet24', 'resnet64', 'efun', 'swinv2', 'vitti']
    jobs.sort(key=lambda j: (j['gpu'] != '4090', order.index(j['profile'])))
    assert len(jobs) == 14 and len({j['id'] for j in jobs}) == 14
    plan = dict(created=datetime.datetime.now().astimezone().isoformat(), native=artifact, jobs=jobs,
                policy='One complete warm 50000-image ImageNet-512 pass; original configurations; '
                       'sequential GPUs; other-GPU load annotated; same-GPU occupancy waits; '
                       'full prediction and sample identity checks; no profiling or automatic repeats.')
    save(OUT / 'plan.json', plan)
    return plan


def report(plan, active=None):
    records, failures = [], []
    for job in plan['jobs']:
        directory = OUT / 'inference' / job['id'] / 'trial_1'
        if (directory / 'failure.json').exists():
            failures.append(job['id'])
        if not (directory / 'accepted.json').exists():
            continue
        row = read(directory / 'accepted.json')
        change = 100 * (row['images_per_s'] / job['previous_median'] - 1)
        records.append(dict(gpu=row['gpu'], model=row['model'], samples=50000, trial=1,
                            elapsed_s=row['elapsed_s'], images_per_s=row['images_per_s'],
                            previous_median_images_per_s=job['previous_median'], change_percent=change,
                            status='OBSERVED_DECREASE' if change < 0 else 'OBSERVED_INCREASE',
                            correctness=row['correctness'], raw_result=row['raw_result'],
                            previous_raw=job['previous_raw'], native_sha256=plan['native']['sha256'],
                            notes='One new pass versus historical three-trial median; host load may differ. ' + row['notes']))
    if records:
        write_csv(OUT / 'performance_changes.csv', records, list(records[0]))
    save(OUT / 'status.json', dict(updated=datetime.datetime.now().astimezone().isoformat(),
         pid=os.getpid(), status='COMPLETE' if len(records) == 14 else ('RUNNING_OR_WAITING' if active else 'INCOMPLETE'),
         accepted=len(records), total=14, failed=failures, active=active))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.prepare:
        print('Prepared', len(plan['jobs']), 'passes;', plan['native']['sha256'])
        return
    os.environ.setdefault('FASTLENS_EVALUATION_SNAPSHOT', str(OUT / 'runtime_snapshot.json'))
    verify_snapshot()
    gate = OUT / 'streamed_tensor_gate'
    if not (gate / 'PASS.json').exists():
        gate.mkdir(exist_ok=True)
        env = replay.environment('4090', 'mobilenet24')
        env['FASTLENS_EVALUATION_SNAPSHOT'] = os.environ['FASTLENS_EVALUATION_SNAPSHOT']
        env['PYTHONPATH'] = str(OUT / 'native_runtime') + ':' + env['PYTHONPATH']
        code = ('import torch, _fastlens_direct_dct, unittest; '
                'from fastlens.benchmarks.dct_models.tests.test_online_crop import OnlineCropTest; '
                'r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([OnlineCropTest('
                '"test_bounded_pipeline_delivers_projected_ranges_on_consumer_stream")])); '
                'assert r.wasSuccessful() and not r.skipped')
        replay.run(['numactl', '--cpunodebind=0', '--membind=0', sys.executable, '-c', code], gate, env, None)
        save(gate / 'PASS.json', dict(native_sha256=plan['native']['sha256'], correctness='Exact streamed vs full-grid tensor equality; consumer-stream ownership and tail checks'))
    for job in plan['jobs']:
        output = OUT / 'inference' / job['id'] / 'trial_1'
        if (output / 'accepted.json').exists():
            continue
        if (output / 'failure.json').exists():
            continue  # Failed evidence is retained; no unbounded retries.
        report(plan, active=job['id'])
        try:
            replay.execute(job, 1)
            accepted = read(output / 'accepted.json')
            accepted['implementation_cohort'] = 'native_' + plan['native']['sha256'][:16]
            accepted['notes'] = 'One-pass latest-version remeasurement; warm ImageNet-512; no profiling; NUMA0; full load trace. Historical trials retained separately.'
            residents = os.environ.get('FASTLENS_ALLOWED_RESIDENT_PIDS')
            if residents and job['gpu'] == 'H100':
                accepted['notes'] += (' User-authorized same-GPU resident PID(s): ' + residents +
                    '; pmon activity sampled, unavailable counters do not establish absence of compute contention; not an exclusive-GPU measurement.')
            # Historical resource counters inherited by the replay helper are not new measurements.
            for field in ('model_only_images_per_s', 'storage_bytes', 'metadata_bytes', 'issued_read_bytes',
                          'h2d_bytes', 'h2d_copy_count', 'gpu_peak_process_bytes', 'torch_peak_allocated_bytes',
                          'torch_peak_reserved_bytes', 'native_peak_bytes', 'host_rss_peak_bytes', 'pinned_peak_bytes'):
                accepted[field] = ''
            save(output / 'accepted.json', accepted)
        except (AssertionError, RuntimeError, subprocess.CalledProcessError) as error:
            output.mkdir(parents=True, exist_ok=True)
            save(output / 'failure.json', dict(error=str(error), type=type(error).__name__))
            print('FAILED', job['id'], repr(error), flush=True)
        report(plan)


if __name__ == '__main__':
    main()
