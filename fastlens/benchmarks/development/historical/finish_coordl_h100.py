#!/usr/bin/env python3
"""Finish H100 CoorDL after the user's conditional cache-cleanup approval applies."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_vitti import execute
from fastlens.benchmarks.common import gpu_uuid


def main():
    import psutil
    # This is the other evaluation suite observed when cleanup was approved conditionally.
    if psutil.pid_exists(3321918):
        raise RuntimeError('Conditional cleanup remains pending: other suite PID 3321918 still exists')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    if active.strip():
        raise RuntimeError(f'Cache cleanup deferred while other GPU jobs are active: {active}')
    base = ROOT / 'results/raw/H100/vitti'
    directory = base / 'inference'
    contract_path = directory / 'contract.json'
    contract = json.loads(contract_path.read_text())
    manifest = json.loads(Path(contract['dataset']['sample_manifest']).read_text())
    cache_root = Path('/dev/shm/cache/home/tangyuxin/gfastlanes/FastLanes/fastlens/data/system_rgbnomore/e2e_v3/imagenet_512/val')
    removed_files = removed_bytes = 0
    for sample in manifest['samples']:
        for suffix in ('', '-tmp'):
            target = Path('/dev/shm/cache') / (str(Path(sample['path']).relative_to('/')) + suffix)
            if not target.is_relative_to(cache_root):
                raise ValueError(f'Cache entry outside approved subtree: {target}')
            if target.exists():
                removed_bytes += target.stat().st_size
                target.unlink()
                removed_files += 1
    print(json.dumps(dict(action='conditionally authorized CoorDL cache cleanup',
                          cache_root=str(cache_root), removed_files=removed_files, removed_bytes=removed_bytes,
                          source_jpegs_modified=False)), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid('H100'), TMPDIR=str(Path.home()/'tmp'),
               PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    python = str(Path.home() / 'tmp/coordl-build-cu128/coordl-python')
    command = [python, '-m', 'fastlens.benchmarks.system_dct_major.pipeline', '--pipeline', 'coordl',
               '--contract', str(contract_path), '--output', str(directory / 'pipeline_coordl.json')]
    execute(command, directory / 'pipeline_coordl_resumed.log', env)
    observed = json.loads((directory / 'pipeline_coordl.json').read_text())
    values = [r['images']/r['repeat_scope_seconds'] for r in observed['repeats'][1:]]
    if statistics.stdev(values)/statistics.mean(values) > .02:
        extra = base / 'inference_extra'
        extra.mkdir(exist_ok=True)
        additional = json.loads(contract_path.read_text())
        additional['execution']['repeats'] = 3
        additional['pipelines']['enabled'] = ['coordl']
        (extra / 'contract.json').write_text(json.dumps(additional,indent=2)+'\n')
        execute([python, '-m', 'fastlens.benchmarks.system_dct_major.pipeline', '--pipeline', 'coordl',
                 '--contract', str(extra / 'contract.json'), '--output', str(extra / 'pipeline_coordl.json')],
                extra / 'pipeline_coordl_resumed.log', env)
        subprocess.run([sys.executable, '-m', 'fastlens.benchmarks.system_dct_major.validate', '--contract',
                        str(extra/'contract.json'), '--output-dir', str(extra)], env=env, check=True)
    subprocess.run([sys.executable, '-m', 'fastlens.benchmarks.system_dct_major.validate', '--contract',
                    str(contract_path), '--output-dir', str(directory)], env=env, check=True)


if __name__ == '__main__':
    main()
