"""Shared measured protocol for Sections 6.2.1, 6.2.3 and 6.2.4."""
import json
from pathlib import Path

from .common import ROOT, PYTHON, execute, gpu_environment, parser, plot
from .shared.references import DEFAULT_REFERENCES, reference_inputs, expected_predictions

MATCHED_MODELS = ('vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'resnet24', 'resnet64', 'efun')
DEPLOYMENT_MODELS = ('vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'efun')
RGB = ('pytorch', 'dali', 'coordl', 'ffcv_raw', 'ffcv_jpeg', 'l3')
RESIDENT_PATHS = ('ffcv_raw', 'ffcv_jpeg', 'rgb_resident', 'imagelanes_basic', 'dct_resident')


def main(kind, experiment, figure):
    p = parser(__doc__, experiment)
    p.add_argument('--references', type=Path, default=DEFAULT_REFERENCES,
                   help='JSON mapping GPU/model/pipeline to fresh reference and contract files')
    p.add_argument('--l3-root', type=Path, default=ROOT / 'fastlens/data/baselines/l3')
    p.add_argument('--l3-library', type=Path, default=ROOT / 'build/baselines/l3/libl3-patch32.so')
    p.add_argument('--ffcv-jpeg-root', type=Path, default=ROOT / 'results/preparation/ffcv_jpeg/train')
    p.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), default=['4090', 'H100'])
    p.add_argument('--models', nargs='+', choices=MATCHED_MODELS,
                   default=list(MATCHED_MODELS if kind == 'matched' else DEPLOYMENT_MODELS))
    p.add_argument('--phases', nargs='+', choices=('training', 'inference'),
                   default=['training'] if kind == 'resident' else ['training', 'inference'])
    p.add_argument('--repeats', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--numa-node', type=int, help='default: 1 for ResNet on 4090; 0 otherwise')
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, figure)
        return
    if kind == 'resident' and 'inference' in args.phases:
        p.error('Section 6.2.4 measures training only')
    references = {}
    if 'inference' in args.phases and not args.dry_run:
        for gpu in args.gpus:
            for model in args.models:
                paths = ('cpu_dct', 'imagelanes') if kind == 'matched' else (*RGB, 'imagelanes')
                for path in paths:
                    if model == 'mobilenet32' and path in RGB:
                        continue
                    references[gpu, model, path] = reference_inputs(args.references, gpu, model, path)
    for gpu in args.gpus:
        env = gpu_environment(gpu, args.dry_run, OMP_NUM_THREADS=8, MKL_NUM_THREADS=8,
                              OPENBLAS_NUM_THREADS=8, FASTLENS_NSYS_FINE=0)
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
        for model in args.models:
            node = args.numa_node if args.numa_node is not None else int(gpu == '4090' and model.startswith('resnet'))
            base = ['numactl', f'--cpunodebind={node}', f'--membind={node}', *PYTHON]
            for phase in args.phases:
                if kind == 'matched':
                    paths = ('cpu_dct', 'imagelanes_basic' if phase == 'training' else 'imagelanes')
                elif kind == 'resident':
                    paths = (*RGB, 'imagelanes_basic', 'dct_resident', 'rgb_resident')
                else:
                    paths = (*RGB, 'imagelanes_basic' if phase == 'training' else 'imagelanes')
                # The two DCT MobileNet profiles share one RGB model/reference.
                if model == 'mobilenet32':
                    paths = tuple(path for path in paths if path not in (*RGB, 'rgb_resident'))
                # Check all paths before measuring; rotate their order in each paired trial.
                for trial in [-1, *range(args.repeats)]:
                    rotate = 0 if trial == -1 else 2 * trial % len(paths)
                    for path in (*paths[rotate:], *paths[:rotate]):
                        cli_path = path.replace('imagelanes', 'fastlens')
                        destination = args.output.resolve() / phase / gpu / model / cli_path
                        output = destination / ('check' if trial == -1 else f'trial_{trial}')
                        if phase == 'training':
                            script = ('fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py' if path in RESIDENT_PATHS
                                      else 'fastlens/benchmarks/paper/shared/remaining_baseline_training.py')
                            command = [*base, ROOT / script,
                                       '--gpu', gpu, '--model', model, '--path', cli_path]
                            if path == 'l3':
                                command += ['--l3-root', args.l3_root, '--l3-library', args.l3_library]
                            elif path == 'ffcv_jpeg':
                                command += ['--ffcv-root', args.ffcv_jpeg_root]
                            command += (['--check'] if trial == -1 else
                                        ['--trial', trial, '--correctness-report', destination / 'check/result.json'])
                        else:
                            inputs = (dict(reference=f'<{args.references}:{gpu}/{model}/{cli_path}:reference>',
                                           contract=f'<{args.references}:{gpu}/{model}/{cli_path}:contract>')
                                      if args.dry_run else references[gpu, model, path])
                            if model == 'swinv2':
                                command = [*base, ROOT / 'fastlens/benchmarks/paper/shared/swin_unified_inference.py',
                                           '--path', cli_path, '--contract', inputs['contract']]
                            else:
                                command = [*base, ROOT / 'fastlens/benchmarks/paper/shared/remaining_baseline_inference.py',
                                           '--model', model, '--path', cli_path, '--reference', inputs['reference']]
                                if 'contract' in inputs and not args.dry_run:
                                    command += ['--contract', inputs['contract']]
                            if trial == -1:
                                command.append('--check')
                        command = [*command, '--output', output]
                        if not args.dry_run:
                            output.mkdir(parents=True, exist_ok=True)
                            metadata = dict(command=list(map(str, command)), phase=phase, gpu=gpu,
                                            model=model, pipeline=cli_path,
                                            environment={key: env[key] for key in (
                                                'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                'OPENBLAS_NUM_THREADS', 'RGBNOMORE_ROOT', 'PYTHONPATH',
                                                'FASTLENS_NSYS_FINE') if key in env})
                            (output / 'run.command.json').write_text(json.dumps(metadata, indent=2) + '\n')
                        execute(command, args, env)
                        if phase == 'inference' and model == 'swinv2' and not args.dry_run:
                            expected = expected_predictions(inputs['reference'], inputs['contract'])
                            result = json.loads((output / 'result.json').read_text())
                            actual = dict(zip(result['sample_ids'], (row[0] for row in result['predictions'])))
                            if any(prediction != expected[sample] for sample, prediction in actual.items()):
                                raise ValueError(f'Swin predictions differ from the accepted reference: {output}')
