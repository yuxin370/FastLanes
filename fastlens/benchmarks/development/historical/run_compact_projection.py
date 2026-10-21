"""Run the compact frequency sweep and a separate H2D capture on RTX 4090."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid


def main():
    output = ROOT / 'results/compact_projection_20261001'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid('4090'),
               TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sweep = output / 'frequency'
    sweep.mkdir(parents=True, exist_ok=True)
    run(['numactl', '--cpunodebind=0', '--membind=0', sys.executable,
         'fastlens/benchmarks/paper/section_6_4/projection_and_staging/projection_microbench.py', '--output', str(sweep)],
        sweep, env, None)
    capture = output / 'diagnostic'
    capture.mkdir(parents=True, exist_ok=True)
    run(['numactl', '--cpunodebind=0', '--membind=0',
         'nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
         '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
         '--output', str(capture / 'capture'), sys.executable,
         'fastlens/benchmarks/development/diagnostics/frequency_h2d_diagnostic.py', '--output', str(capture / 'native.json')],
        capture, env, None)
    subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output',
                    str(capture / 'capture.sqlite'), str(capture / 'capture.nsys-rep')],
                   env=env, check=True)


if __name__ == '__main__':
    main()
