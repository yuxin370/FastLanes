"""Eight fixed training windows isolating CPU/memory NUMA binding on the 4090."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.development.reports.report_resident_study import write_csv
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save

COHORT = ROOT/'results/resnet_matched_20261013'
OUT = ROOT/'results/resnet_numa_probe_20261013'
NATIVE = ROOT/'results/resident_study_20261011_fixed/native_runtime'
# Reverse both path and node ordering in round two, fixed before measurement.
ORDER = [(0, 'cpu_dct', 0), (0, 'cpu_dct', 1),
         (0, 'imagelanes_basic', 1), (0, 'imagelanes_basic', 0),
         (1, 'imagelanes_basic', 0), (1, 'imagelanes_basic', 1),
         (1, 'cpu_dct', 1), (1, 'cpu_dct', 0)]


def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'plan.json').exists():
        files = dict(read(COHORT/'runtime_snapshot.json')['files'])
        files[str(Path(__file__).resolve().relative_to(ROOT))] = file_sha(__file__)
        save(OUT/'runtime_snapshot.json', dict(files=files,
            fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()))
        save(OUT/'plan.json', dict(order=ORDER, model='resnet24', gpu='4090',
            measured_images=65536, warmup_updates=32, batch=64, accumulation=16,
            precision='bf16-autocast', basic_augmentation=True, practical_threshold_percent=3,
            decision='Continue on node 1 only if every paired throughput change is within +/-3%; otherwise inspect.',
            cache='Existing warm-cache protocol; no page-cache eviction or migration.',
            scope='Current topology, change CPU and memory binding only; two repetitions per node/path.'))
        (OUT/'gpu_topology.csv').write_text(subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,pcie.link.gen.max,pcie.link.width.max',
            '--format=csv'], text=True))
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(OUT/'runtime_snapshot.json')
    verify_snapshot()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS['4090'], DCTNET_PROFILE='resnet24',
        TMPDIR=str(Path.home()/'tmp'), PYTHONPATH=f'{ROOT}:{NATIVE}',
        FASTLENS_TRAINING_NATIVE_DIR=str(NATIVE), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', FASTLENS_NSYS_FINE='0')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    records = list(csv.DictReader((OUT/'trials.csv').open())) if (OUT/'trials.csv').exists() else []
    state = dict(status='RUNNING', pid=os.getpid(), completed_runs=len(records), planned_runs=8, active=None)
    save(OUT/'status.json', state)
    try:
        for trial, path, node in ORDER:
            if any((int(r['trial']), r['path'], int(r['numa_node'])) == (trial, path, node) for r in records):
                continue
            dest = OUT/path/f'node_{node}'/f'trial_{trial}'
            state['active'] = str(dest.relative_to(OUT))
            save(OUT/'status.json', state)
            if dest.exists():
                dest.rename(dest.with_name(dest.name+'_failed_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            dest.mkdir(parents=True)
            driver = 'fastlens/benchmarks/paper/shared/remaining_baseline_training.py' if path == 'cpu_dct' else 'fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py'
            reference = COHORT/'training/4090/resnet24'/path
            command = ['numactl', f'--cpunodebind={node}', f'--membind={node}', sys.executable, '-B',
                driver, '--gpu', '4090', '--model', 'resnet24', '--path', path,
                '--trial', str(trial), '--correctness-report', str(reference/'check/result.json'),
                '--output', str(dest)]
            run(command, dest, env, None)
            meta, result = read(dest/'run.command.json'), read(dest/'result.json')
            assert meta['returncode'] == 0 and meta['code_unchanged_during_run']
            previous = read(reference/'trial_0/result.json')
            for key in ('restored_state', 'sample_order_sha256', 'tensor_contract', 'precision',
                        'images', 'warmup_updates', 'batch', 'accumulation', 'compile', 'tf32',
                        'cuda_graphs', 'host_threads', 'model_stream_priority', 'CUBLAS_WORKSPACE_CONFIG'):
                assert result[key] == previous[key], key
            assert result['correctness_passed'] and result['restoration_verified']
            cpus = set()
            for span in Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip().split(','):
                bounds = span.split('-')
                cpus.update(range(int(bounds[0]), int(bounds[-1])+1))
            assert set(result['cpu_affinity']) == cpus
            records.append(dict(path=path, numa_node=node, trial=trial, images=result['images'],
                seconds=result['seconds'], images_per_s=result['images_per_s'],
                process_wall_s=meta['process_wall_s'], raw_result=str((dest/'result.json').relative_to(ROOT))))
            write_csv(OUT/'trials.csv', records)
            state['completed_runs'] = len(records)
            save(OUT/'status.json', state)
            print('COMPLETED', records[-1], flush=True)
        comparisons = []
        for path in ('cpu_dct', 'imagelanes_basic'):
            rates = {(int(r['numa_node']), int(r['trial'])): float(r['images_per_s']) for r in records if r['path'] == path}
            changes = [(rates[1, t]/rates[0, t]-1)*100 for t in range(2)]
            comparisons.append(dict(path=path,
                node0_median_images_per_s=statistics.median(rates[0, t] for t in range(2)),
                node1_median_images_per_s=statistics.median(rates[1, t] for t in range(2)),
                round0_change_percent=changes[0], round1_change_percent=changes[1],
                median_change_percent=statistics.median(changes),
                within_threshold=all(abs(x) <= 3 for x in changes)))
        write_csv(OUT/'comparison.csv', comparisons)
        state.update(status='COMPLETE', active=None,
                     within_threshold=all(r['within_threshold'] for r in comparisons))
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(OUT/'status.json', state)


if __name__ == '__main__':
    main()
