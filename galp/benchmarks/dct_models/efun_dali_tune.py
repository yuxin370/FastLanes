"""Bounded DALI tuning using the existing eFUN evaluation/training drivers."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='dali_tuning_20260916')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[3])
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(Path.home()/'tmp/dctnet'),
                      MPLCONFIGDIR=str(Path.home()/'tmp/matplotlib'))
    exp = Path(__file__).resolve().parent
    base = Path('galp/data/system_rgbnomore/e2e_v3/runs/efun')
    root = base/args.run
    root.mkdir(exist_ok=False)
    driver = [sys.executable, str(exp/'efun.py')]
    physical = Path('/home/tangyuxin/gfastlanes/FastLanes/galp/data/compressed/imagenet512_train_block_major_premixed')
    contract = json.loads((physical/'materialization_contract.json').read_text())
    train = [*driver, 'training_pls', '--manifest', str(physical/'dct/manifest.bin'),
             '--mapping', contract['ordered_mapping'], '--mapping-sha256', contract['ordered_mapping_sha256'],
             '--condition', 'RGB', '--input-backend', 'rgb_d2', '--segments-per-pool', '4', '--epochs', '2']
    infer = [*driver, 'evaluate_rgb', '--route', 'dali', '--count', '50000', '--batch-size', '64']
    checkpoint = base/'training_v1/rgb_d2_full/epoch_0.pth'
    records = []

    def idle_gpu():
        active = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError(f'Selected GPU has compute processes: {active}')

    def execute(command, directory):
        directory.mkdir(parents=True, exist_ok=True)
        print(datetime.datetime.now().isoformat(), 'START', str(directory.relative_to(root)), flush=True)
        with (directory/'run.log').open('w') as log:
            subprocess.run([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT, check=True)

    def measure(mode, config, repeat):
        idle_gpu()
        workers, depth = config
        directory = root/mode/f'w{workers}_q{depth}_r{repeat}'
        command = infer if mode == 'inference' else [*train, '--resume', str(checkpoint),
                                                    '--max-pools', '13', '--validation-count', '1000']
        offset = samples.tell()
        execute([*command, '--workers', workers, '--dali-prefetch-depth', depth,
                 '--output-dir', directory], directory)
        with samples_path.open() as observed:
            observed.seek(offset)
            pids = {row[2].strip() for row in csv.reader(observed) if len(row) == 4 and row[2].strip().isdigit()}
        if len(pids) != 1:
            raise RuntimeError(f'Expected one GPU process for {directory}, observed {pids}')
        if mode == 'inference':
            result = json.loads((directory/'RGB_dali_50000.json').read_text())
            assert result['samples'] == 50000
            seconds = result['e2e_seconds']
        else:
            result = json.loads((directory/'training.json').read_text())['epochs'][0]
            assert result['samples'] == result['unique_samples'] == 13*4096
            assert result['optimizer_updates'] == 1252+13*4
            # Exclude epoch setup and the first pool (compilation/warmup).
            seconds = result['pools'][-1]['completed_epoch_seconds']-result['pools'][0]['completed_epoch_seconds']
        record = dict(mode=mode, workers=workers, prefetch_depth=depth, repeat=repeat,
                      seconds=seconds, images=50000 if mode == 'inference' else 12*4096,
                      directory=str(directory), gpu_pid=next(iter(pids)))
        records.append(record)
        (root/'measurements.json').write_text(json.dumps(records, indent=2)+'\n')
        print('RESULT', json.dumps(record), flush=True)

    def median(mode, config):
        return statistics.median(r['seconds'] for r in records
                                 if r['mode'] == mode and (r['workers'], r['prefetch_depth']) == config)

    def profile(mode, config, command):
        idle_gpu()
        directory = root/'profiles'/mode
        stem = directory/('rgb_dali' if mode == 'inference' else 'capture')
        workers, depth = config
        execute(['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                 '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--output', stem,
                 *command, '--workers', workers, '--dali-prefetch-depth', depth,
                 '--profile', '--output-dir', directory/'run'], directory)
        with (directory/'export.log').open('w') as log:
            subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(stem)+'.sqlite',
                            str(stem)+'.nsys-rep'], stdout=log, stderr=subprocess.STDOUT, check=True)
        if mode == 'inference':
            analyze = [sys.executable, str(exp/'analyze_profiles.py'), str(directory)]
        else:
            analyze = [sys.executable, str(exp/'analyze_training_capture.py'), str(stem)+'.sqlite',
                       '--output', str(directory/'breakdown.json'), '--timeline', str(directory/'timeline.png'),
                       '--label', f'EfficientNet-B0 DALI w{workers} q{depth} / epoch 2']
        with (directory/'analysis.log').open('w') as log:
            subprocess.run(analyze, stdout=log, stderr=subprocess.STDOUT, check=True)

    idle_gpu()
    samples_path = root/'gpu_processes.csv'
    with samples_path.open('w') as samples:
        monitor = subprocess.Popen(['nvidia-smi', '-i', gpu,
                                    '--query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits', '-lms', '100'], stdout=samples)
        try:
            configs = [(w, q) for w in (4, 8, 16) for q in (2, 4)]
            rng = random.Random(11997733)
            for repeat in range(3):
                order = configs.copy()
                rng.shuffle(order)
                for config in order:
                    measure('inference', config, repeat)
            inference_best = min(configs, key=lambda c: median('inference', c))
            order = configs.copy()
            rng.shuffle(order)
            for config in order:
                measure('training', config, 0)
            baseline = (4, 2)
            challenger = min((c for c in configs if c != baseline), key=lambda c: median('training', c))
            for repeat, order in [(1, [challenger, baseline]), (2, [baseline, challenger])]:
                for config in order:
                    measure('training', config, repeat)
            training_best = min([baseline, challenger], key=lambda c: median('training', c))
            selected = dict(inference=dict(workers=inference_best[0], prefetch_depth=inference_best[1],
                                           median_seconds=median('inference', inference_best)),
                            training=dict(workers=training_best[0], prefetch_depth=training_best[1],
                                          median_seconds=median('training', training_best)))
            inference_runs = [r for r in records if r['mode'] == 'inference' and
                              (r['workers'], r['prefetch_depth']) == inference_best]
            selected['inference']['representative_run'] = sorted(inference_runs, key=lambda r:r['seconds'])[1]['directory']
            (root/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
            print('SELECTED', json.dumps(selected), flush=True)
            idle_gpu()
            full = root/'rgb_d2_full'
            execute([*train, '--workers', training_best[0], '--dali-prefetch-depth', training_best[1],
                     '--initial-validation', '--validation-count', '50000', '--output-dir', full], full)
            profile('inference', inference_best, infer)
            profile('training', training_best, [*train, '--resume', str(full/'epoch_0.pth')])
        finally:
            monitor.terminate()
            monitor.wait()
    print('DALI tuning, full training and profiling completed', datetime.datetime.now().isoformat(), flush=True)


if __name__ == '__main__':
    main()
