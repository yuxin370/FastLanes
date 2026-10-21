"""Table 2: full 50K H100 inference, source D4/D6/D8, shape-preserving zero fill."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from fastlens.benchmarks.paper.common import DATA, ROOT, PYTHON, environment, execute, gpu_environment, parser

METHODS = ('cpu_dct', 'fastlens', 'ic_static_shared', 'ic_static_batched', 'fs')
CACHE_MIB = dict(cpu_dct=0, fastlens=0, ic_static_shared=1024, ic_static_batched=1024, fs=512)
CLI = [*PYTHON, '-m', 'fastlens.benchmarks.baselines.dct_baselines']


def main():
    p = parser(__doc__, 'section_6_2_2/shape_preserving_selection', ('prepare', 'run', 'table'))
    p.add_argument('--gpu', choices=('4090', 'H100'), default='H100')
    p.add_argument('--models', nargs='+', choices=('mobilenet24', 'resnet24', 'vitti'),
                   default=['mobilenet24', 'resnet24', 'vitti'])
    p.add_argument('--diagonals', nargs='+', type=int, choices=(4, 6, 8), default=[4, 6, 8])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--source-manifest', type=Path, default=DATA / 'paper/val50000/source.json')
    p.add_argument('--prepared', type=Path, default=DATA / 'paper/selected')
    p.add_argument('--fs-store', type=Path, default=DATA / 'paper/val50000/fs_matched_b8')
    p.add_argument('--native-manifest', type=Path,
                   default=DATA / 'compressed/imagenet512_val_block_major/manifest.bin')
    p.add_argument('--module-path', type=Path, default=ROOT / 'build/fastlens/torch')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 1) // 8))
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--pool-size', type=int, default=256)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--cpus', default='0-7', help='eight measured CPU cores; adjust CPU IDs for the host')
    p.add_argument('--seed', type=int, default=11997733)
    p.add_argument('--warmup-seconds', type=float, default=60.)
    p.add_argument('--min-seconds', type=float, default=180.)
    p.add_argument('--repeats', type=int, default=1)
    args = p.parse_args()
    if args.action == 'table':
        path = args.artifact_root / 'fastlens/benchmarks/paper/results/section_6_2_2/selected_inference.csv'
        print(path.read_text(), end='')
        return

    def config(model, diagonals, method):
        return args.prepared / 'contracts' / model / f'D{diagonals}' / method / 'matched.json'

    def prepare(item):
        model, diagonals, method = item
        execute([*CLI, 'prepare', '--source-manifest', args.source_manifest,
                 '--model', model, '--upstream', environment()['RGBNOMORE_ROOT'],
                 '--source-diagonals', diagonals, '--batch-size', args.batch_size,
                 '--pool-size', args.pool_size, '--workers', args.workers, '--cpu-loader', 'dataloader',
                 '--prefetch-factor', 2, '--torch-threads', args.torch_threads, '--seed', args.seed,
                 '--request-order', 'source', '--atol', '1e-5', '--rtol', '1e-5',
                 '--decode-budget-mib', 1024, '--reader-budget-mib', 2048,
                 '--host-input-budget-mib', 32768, '--decoded-cache-mib', CACHE_MIB[method],
                 '--output', config(model, diagonals, method).parent], args)

    cases = [(m, d, method) for m in args.models for d in args.diagonals for method in args.methods]
    if args.action == 'prepare':
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(cases))) as pool:
            list(pool.map(prepare, cases))
            commands = []
            for diagonals in args.diagonals:
                for method, group in (('ic_static_shared', 1), ('ic_static_batched', 256)):
                    if method in args.methods:
                        commands.append([*CLI, 'build', '--config', config(args.models[0], diagonals, method),
                                         '--method', 'ic_matched_fixedk', '--group-size', group,
                                         '--workers', args.workers, '--dtype-policy', 'paper',
                                         '--output', args.prepared / f'D{diagonals}' / method])
            list(pool.map(lambda command: execute(command, args), commands))
        return
    env = gpu_environment(args.gpu, args.dry_run, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
                          FASTLENS_BLOCK_MAJOR_ACCESS_DIR=args.native_manifest.parent / 'access')
    for model, diagonals, method in cases:
        backend = 'ic' if method.startswith('ic_') else method
        root = (args.native_manifest if method == 'fastlens' else args.fs_store if method == 'fs'
                else args.prepared / f'D{diagonals}' / method)
        destination = args.output / args.gpu / model / f'D{diagonals}' / method
        if not args.dry_run:
            destination.mkdir(parents=True, exist_ok=True)
        inputs = ['--config', config(model, diagonals, method), '--backend', backend,
                  '--device', 'cuda:0', '--module-path', args.module_path, '--source-workers', args.workers]
        if method != 'cpu_dct':
            inputs += ['--root', root]
        execute(['taskset', '-c', args.cpus, *CLI, 'verify-inference', *inputs,
                 '--output', destination / 'verification.json'], args, env)
        execute(['taskset', '-c', args.cpus, *CLI, 'bench', *inputs, '--evidence', destination / 'verification.json',
                 '--run-kind', 'inference', '--warmup-seconds', args.warmup_seconds,
                 '--min-seconds', args.min_seconds, '--observe-cache', '--repeats', args.repeats,
                 '--output', destination / 'results.jsonl'], args, env)


if __name__ == '__main__':
    main()
