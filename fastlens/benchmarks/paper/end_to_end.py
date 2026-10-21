"""Shared measured protocol for Sections 6.2.1, 6.2.3 and 6.2.4."""
import json

from .common import ROOT, PYTHON, execute, gpu_environment, parser, plot

MATCHED_MODELS = ('vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'resnet24', 'resnet64', 'efun')
DEPLOYMENT_MODELS = ('vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'efun')
RGB = ('pytorch', 'dali', 'coordl', 'ffcv_raw', 'ffcv_jpeg', 'l3')
RESIDENT_PATHS = ('ffcv_raw', 'ffcv_jpeg', 'rgb_resident', 'imagelanes_basic', 'dct_resident')


def main(kind, experiment, figure):
    p = parser(__doc__, experiment)
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
                            command += (['--check'] if trial == -1 else
                                        ['--trial', trial, '--correctness-report', destination / 'check/result.json'])
                        elif model == 'swinv2':
                            command = [*base, ROOT / 'fastlens/benchmarks/paper/shared/swin_unified_inference.py', '--path', cli_path]
                            if trial == -1:
                                command.append('--check')
                        else:
                            # Full-population prediction references are external inputs, not plotted CSVs.
                            if args.dry_run:
                                reference = f'<accepted-reference:{gpu}/{model}/{path}>'
                            else:
                                from fastlens.benchmarks.paper.shared.references import inference_reference
                                reference = inference_reference(gpu, model, path)
                            command = [*base, ROOT / 'fastlens/benchmarks/paper/shared/remaining_baseline_inference.py',
                                       '--model', model, '--path', cli_path, '--reference', reference]
                            if trial == -1:
                                command.append('--check')
                        execute([*command, '--output', output], args, env)
                        if phase == 'inference' and model == 'swinv2' and not args.dry_run:
                            from fastlens.benchmarks.paper.shared.references import expected_predictions
                            expected = expected_predictions(gpu, path)
                            result = json.loads((output / 'result.json').read_text())
                            actual = dict(zip(result['sample_ids'], (row[0] for row in result['predictions'])))
                            if any(prediction != expected[sample] for sample, prediction in actual.items()):
                                raise ValueError(f'Swin predictions differ from the accepted reference: {output}')
