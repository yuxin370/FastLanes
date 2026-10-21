"""Figure 10: JPEG-B8 compact training (D3/D15), with the measured per-point repeat counts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastlens.benchmarks.paper.common import DATA, ROOT, PYTHON, environment, execute, gpu_environment, parser, plot

METHODS = ('cpu_dct', 'ic_prepared_shared', 'ic_prepared_batched', 'fs_prefetched', 'fastlens')


def measured_repeats(gpu, model, diagonals, method):
    if gpu == '4090' and diagonals == 3:
        if model == 'resnet50' or model == 'vitti' and method.startswith('ic_prepared'):
            return 3
    return 1


def main():
    p = parser(__doc__, 'section_6_2_2/compact_input', ('prepare', 'run', 'plot'))
    p.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), default=['4090', 'H100'])
    p.add_argument('--models', nargs='+', choices=('resnet50', 'mobilenet_v2', 'vitti'),
                   default=['resnet50', 'mobilenet_v2', 'vitti'])
    p.add_argument('--diagonals', nargs='+', type=int, choices=(3, 15), default=[3, 15])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--source-manifest', type=Path, default=DATA / 'paper/train8194/source.json')
    p.add_argument('--fs-store', type=Path, default=DATA / 'paper/train8194/fs_matched_b8')
    p.add_argument('--prepared', type=Path, default=DATA / 'paper/compact')
    p.add_argument('--native-manifest', type=Path,
                   default=DATA / 'compressed/imagenet512_train_block_major_premixed/dct/manifest.bin')
    p.add_argument('--module-path', type=Path, default=ROOT / 'build/fastlens/torch')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--pool-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--build-workers', type=int, default=32)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--cpus', default='0-7', help='eight measured CPU cores; adjust CPU IDs for the host')
    p.add_argument('--warmup-seconds', type=float, default=60.)
    p.add_argument('--min-seconds', type=float, default=180.)
    p.add_argument('--repeats', type=int, help='override published counts: 53 single points and 7 medians of 3')
    p.add_argument('--seed', type=int, default=20261014)
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, 'plot_compact_training.py')
        return
    if args.action == 'prepare':
        def prepare(diagonals):
            profile = args.prepared / f'd{diagonals}.json'
            execute([*PYTHON, '-m', 'fastlens.benchmarks.baselines.dct_baselines.b8_compact', 'fit',
                     '--source-manifest', args.source_manifest, '--source-diagonals', diagonals,
                     '--upstream', environment()['RGBNOMORE_ROOT'], '--workers', args.build_workers,
                     '--output', profile], args)
            execute([*PYTHON, '-m', 'fastlens.benchmarks.baselines.dct_baselines.prepared_compact',
                     '--profile', profile, '--source-store', args.fs_store,
                     '--workers', args.build_workers, '--group-size', 256,
                     '--output', args.prepared / f'prepared-d{diagonals}'], args)
        with ThreadPoolExecutor(max_workers=len(args.diagonals)) as pool:
            list(pool.map(prepare, args.diagonals))
        return
    # Timed GPU workloads run serially so data preparation and other measurements do not compete.
    for gpu in args.gpus:
        env = gpu_environment(gpu, args.dry_run, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1)
        for model in args.models:
            for diagonals in args.diagonals:
                for method in args.methods:
                    repeats = args.repeats if args.repeats is not None else measured_repeats(gpu, model, diagonals, method)
                    command = [*PYTHON, '-m', 'fastlens.benchmarks.baselines.dct_baselines.compact_benchmark',
                               '--profile', args.prepared / f'd{diagonals}.json', '--model', model,
                               '--method', method, '--device', 'cuda:0', '--module-path', args.module_path,
                               '--batch-size', args.batch_size, '--pool-size', args.pool_size,
                               '--workers', args.workers, '--torch-threads', args.torch_threads,
                               '--warmup-seconds', args.warmup_seconds, '--min-seconds', args.min_seconds,
                               '--seed', args.seed, '--repeats', repeats,
                               '--output', args.output / gpu / model / f'D{diagonals}' / f'{method}.json']
                    if method != 'cpu_dct':
                        root = (args.native_manifest if method == 'fastlens' else args.fs_store if method == 'fs_prefetched'
                                else args.prepared / f'prepared-d{diagonals}' / method)
                        command += ['--root', root]
                    execute(['taskset', '-c', args.cpus, *command], args, env)


if __name__ == '__main__':
    main()
