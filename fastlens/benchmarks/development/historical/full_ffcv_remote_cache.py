"""Warm/query full Beton from NUMA 1; leave training on its original NUMA 0 policy."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def remote_cache_pages(paths, warm=False):
    if len(paths)!=1 or Path(paths[0]).suffix!='.beton':
        raise ValueError('This wrapper is only for the single full-training Beton')
    command=['numactl','--cpunodebind=1','--membind=1',sys.executable,str(Path(__file__).resolve()),
             '--cache-helper',str(paths[0])]
    if warm:
        command.append('--warm')
    return json.loads(subprocess.check_output(command,text=True,cwd=ROOT))


def main():
    from fastlens.benchmarks.profiling import input_motivation
    if sys.argv[1]=='--cache-helper':
        print(json.dumps(input_motivation.cache_pages([Path(sys.argv[2])],warm='--warm' in sys.argv)))
        return
    if '--full-epoch' not in sys.argv or sys.argv[sys.argv.index('--pipeline')+1]!='ffcv':
        raise ValueError('Only full RGB FFCV epochs use this cache placement wrapper')
    output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    (output/'cache_placement.json').write_text(json.dumps(dict(
        warmup_and_residency_query_numa_node=1,training_cpu_node=0,training_memory_node=0,
        reason='Full raw Beton approaches one NUMA node capacity; warm evicted pages and allocate the mincore vector on node 1, retaining the benchmark binding and exact residency gate.',
        profiler=False),indent=2)+'\n')
    input_motivation.cache_pages=remote_cache_pages
    from fastlens.benchmarks.paper.shared.training_window_rgb import main as train
    train()


if __name__=='__main__':main()
