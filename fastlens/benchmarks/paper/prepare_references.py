"""Generate full inference references from the configured public inputs."""
import argparse
import json
from pathlib import Path

from .common import ROOT, PYTHON, execute, gpu_environment
from .shared.references import DEFAULT_REFERENCES, reference_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), default=['4090', 'H100'])
    parser.add_argument('--models', nargs='+', help='Default: all models in the configuration')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    records = json.loads(args.references.read_text())
    commands = []
    for gpu in args.gpus:
        for model in args.models or records[gpu]:
            for pipeline in records[gpu][model]:
                inputs = reference_inputs(args.references, gpu, model, pipeline, preparing=True)
                if inputs['reference'].name != 'result.json':
                    parser.error('Generated references must name a result.json output; existing evaluator outputs can be used directly by chapter runners')
                if inputs['reference'].exists():
                    raise FileExistsError(inputs['reference'])
                if model in ('vitti', 'swinv2') or pipeline in ('coordl', 'ffcv_raw', 'ffcv_jpeg', 'l3'):
                    if 'contract' not in inputs:
                        parser.error(f'{gpu}/{model}/{pipeline} requires a pipeline contract')
                script = ('swin_unified_inference.py' if model == 'swinv2'
                          else 'remaining_baseline_inference.py')
                command = [*PYTHON, ROOT / 'fastlens/benchmarks/paper/shared' / script,
                           '--path', pipeline, '--output', inputs['reference'].parent]
                if model != 'swinv2':
                    command += ['--model', model, '--prepare-reference']
                if 'contract' in inputs:
                    command += ['--contract', inputs['contract']]
                commands.append((gpu, command))
    for gpu, command in commands:
        env = gpu_environment(gpu, args.dry_run, OMP_NUM_THREADS=8, MKL_NUM_THREADS=8,
                              OPENBLAS_NUM_THREADS=8, FASTLENS_NSYS_FINE=0)
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
        execute(command, args, env)


if __name__ == '__main__':
    main()
