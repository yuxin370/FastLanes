"""Use both NUMA nodes for excluded full-Beton prewarming; train on NUMA 0."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def cache_pages(paths, warm=False):
    if len(paths) != 1 or Path(paths[0]).suffix != '.beton':
        raise ValueError('Only the single full-training Beton is supported')
    command = ['numactl', '--cpunodebind=1', '--interleave=0,1', sys.executable,
               str(ROOT / 'fastlens/benchmarks/development/historical/full_ffcv_remote_cache.py'), '--cache-helper', str(paths[0])]
    if warm:
        command.append('--warm')
    return json.loads(subprocess.check_output(command, text=True, cwd=ROOT))


def main():
    if '--full-epoch' not in sys.argv or sys.argv[sys.argv.index('--pipeline') + 1] != 'ffcv':
        raise ValueError('Only full RGB FFCV epochs use this cache policy')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cache_placement.json').write_text(json.dumps(dict(
        warmup_and_residency_query_cpu_node=1, cache_helper_memory_policy='interleave=0,1',
        training_cpu_node=0, training_memory_node=0, profiler=False,
        reason='Observed full-cache gate failure after two NUMA-1 reads of the 1.075-TB Beton. Allow new cache allocations on both nodes; retain the complete residency gate.',
        notes='Allocation policy only: existing cache pages are not migrated, and equal physical placement is not assumed. Different from earlier NUMA-1-only warmup; not an isolated GPU-scaling control.'), indent=2) + '\n')
    from fastlens.benchmarks.profiling import input_motivation
    input_motivation.cache_pages = cache_pages
    from fastlens.benchmarks.paper.shared.training_window_rgb import main as train
    train()


if __name__ == '__main__':
    main()
