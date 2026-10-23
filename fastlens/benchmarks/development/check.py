"""Parallel development checks; no paper performance measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def cpu_tests(jobs, env):
    commands = [
        ['discover', '-s', f'fastlens/benchmarks/{suite}/tests', '-t', '.']
        for suite in ('system_dct_major', 'training_pls', 'dct_retokenization',
                      'coefficient_mask_evaluator', 'profiling', 'baselines/dct_baselines')
    ]
    commands += [
        [f'fastlens.benchmarks.baselines.system_rgbnomore.tests.{name}' for name in (
            'test_system_benchmark', 'test_training_benchmark', 'test_equal_image_epoch_benchmark',
            'test_prepare_imagenet512_v3_train', 'test_rgbnomore_fingerprint_cache')],
        ['fastlens.benchmarks.dct_models.tests.test_evaluation'],
        ['fastlens.benchmarks.paper.tests.test_reproduction_inputs'],
    ]
    env = dict(env, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def check(arguments):
        command = [sys.executable, '-B', '-m', 'unittest', *arguments]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        return command, result

    failed = False
    with ThreadPoolExecutor(max_workers=min(jobs, len(commands))) as pool:
        for command, result in pool.map(check, commands):
            print(f'{"FAIL" if result.returncode else "OK"}: {shlex.join(command[4:])}')
            # Retain unittest's counts, skip counts and failure details.
            if result.returncode:
                print(result.stdout + result.stderr)
                failed = True
            else:
                print('\n'.join(result.stderr.strip().splitlines()[-4:]))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    cpu_tests(args.jobs, dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
