#!/usr/bin/env python3
"""Serial H100 continuation: remaining P0, separate P1 captures, then P2 controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.common import gpu_uuid


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid('H100'),
               TMPDIR=str(Path.home() / 'tmp'), PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch')
    # Wait for this session's already-running CNN controller, never another user's job.
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if 'fastlens/benchmarks/development/historical/run_cnn.py' in command and 'H100' in command:
            print(f'Waiting for existing CNN controller {process.pid}', flush=True)
            process.wait()
    stages = [(['fastlens/benchmarks/development/historical/run_cnn.py', '--gpu', 'H100'], 'cnn_complete')]
    for model in ('rgbnomore-swinv2-t-dct-224-v1', 'mobilenet24', 'efun',
                  'rgbnomore-vitti-dct-224-v1', 'resnet24', 'mobilenet32', 'resnet64'):
        stages.append((['fastlens/benchmarks/development/historical/run_training.py', '--gpu', 'H100', '--models', model], 'training_' + model))
    # The cleanup stage refuses to act until the approved condition is satisfied.
    stages.append((['fastlens/benchmarks/development/historical/finish_coordl_h100.py'], 'coordl_after_other_jobs'))
    pipelines = ['pytorch', 'dali', 'ffcv', 'dct_major_pushdown', 'rgbnomore', 'coordl']
    for mode in ('profile', 'memory'):
        stages.append((['fastlens/benchmarks/development/historical/run_resources.py', '--gpu', 'H100', '--mode', mode,
                        '--pipelines', *pipelines], 'resources_' + mode))
    prepared = ROOT / 'results/preparation/compression_control'
    stages += [(['fastlens/benchmarks/paper/section_6_4/compressed_transfer_and_tensor/prepare_compression_control.py', '--output-root', str(prepared)], 'compression_preparation'),
               (['fastlens/benchmarks/development/historical/run_core_ablations.py', '--gpu', 'H100', '--prepared', str(prepared)], 'core_ablations'),
               (['fastlens/benchmarks/development/diagnostics/analyze_resources.py'], 'resource_analysis'),
               (['fastlens/benchmarks/development/reports/summarize.py'], 'summary'),
               (['fastlens/benchmarks/development/reports/derive_costs.py'], 'costs'),
               (['fastlens/benchmarks/development/reports/plot_results.py'], 'figures')]
    for command, name in stages:
        log = ROOT / 'results/checks/h100_continuation' / (name + '.log')
        metadata = log.with_suffix('.command.json')
        if metadata.exists():
            if json.loads(metadata.read_text()).get('returncode') != 0:
                raise RuntimeError(f'Prior stage failed or was interrupted; inspect {metadata}')
            continue
        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'START', name, flush=True)
        subprocess.run([sys.executable, 'fastlens/benchmarks/execution/run_logged.py', '--log', str(log), '--',
                        sys.executable, *command], cwd=ROOT, env=env, check=True)
        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'COMPLETE', name, flush=True)


if __name__ == '__main__':
    main()
