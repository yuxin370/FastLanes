"""Build paper dependencies or redraw fastlens.pdf's measured figures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import statistics

ROOT = Path(__file__).resolve().parents[3]
PLOTS = ROOT / 'fastlens/docs/paper/scripts'


def run(command, env):
    print('+', shlex.join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def figures(output, artifacts, jobs, env):
    env = dict(env, FASTLENS_FIGURE_OUTPUT=str(output.resolve()),
               FASTLENS_ARTIFACT_ROOT=str(artifacts.resolve()))
    commands = [
        ['plot_matched_dct.py'], ['plot_compact_training.py'],
        ['plot_deployment.py'],
        ['plot_training_resident_attainment.py'],
        ['plot_training_convergence.py'], ['plot_compact_staging.py'],
        ['plot_h2d_traffic.py'], ['plot_core_ablations.py'], ['plot_locality.py'],
        ['plot_read_bandwidth.py'],
    ]
    with ThreadPoolExecutor(max_workers=min(jobs, len(commands))) as pool:
        list(pool.map(lambda args: run([sys.executable, '-B', PLOTS / args[0], *args[1:]], env), commands))
    print(f'Figures written to {output.resolve() / "figures"}. No benchmarks or LaTeX compilation performed.')


def published_statistics(artifacts):
    """Use the plotting functions' selected records and per-configuration ratios."""
    os.environ['FASTLENS_ARTIFACT_ROOT'] = str(artifacts.resolve())
    sys.path.insert(0, str(PLOTS))
    from plot_matched_dct import matched_data, MODELS
    from plot_training_resident_attainment import resident_data
    from plot_deployment_training import training_data, MODELS as DEPLOYMENT_MODELS, SYSTEMS
    from plot_deployment_inference import inference_data
    from chart_style import RESULTS, read

    _, ratios = matched_data()
    matched = {task: {gpu: statistics.mean(float(ratios[task][gpu, model][0]) for model, _ in MODELS)
                      for gpu in ('4090', 'H100')} for task in ratios}
    resident = resident_data()
    attainment = {}
    for domain in ('RGB', 'DCT'):
        values = [float(value[0]) for (_, _, kind), value in resident.items() if kind == domain]
        attainment[domain] = dict(arithmetic_mean_percent=statistics.mean(values),
                                  geometric_mean_percent=statistics.geometric_mean(values))
    train, inference = training_data(), inference_data()
    configurations = [(gpu, model) for gpu in ('4090', 'H100') for model, _ in DEPLOYMENT_MODELS]
    inference_ratios = [float(inference[gpu, model, 'FastLens'][0]) /
                        max(float(inference[gpu, model, name][0]) for name in SYSTEMS[:6])
                        for gpu, model in configurations]
    wins = sum(train[gpu, model, 'FastLens'][0] > max(train[gpu, model, name][0] for name in SYSTEMS[:6])
               for gpu, model in configurations)
    rows = read(RESULTS / 'section_6_2_2/compact_training.csv')
    rates = {(r['gpu'], r['model'], r['diagonals'], r['method']): float(r['images_s']) for r in rows}
    compact = [(gpu, model, diagonals) for gpu, model, diagonals, method in rates if method == 'fastlens']
    speedups = {}
    for baseline in ('cpu_dct', 'best_ic_or_fs'):
        values = [rates[gpu, model, diagonals, 'fastlens'] /
                  (rates[gpu, model, diagonals, 'cpu_dct'] if baseline == 'cpu_dct' else
                   max(rates[gpu, model, diagonals, method]
                       for method in ('ic_prepared_shared', 'ic_prepared_batched', 'fs_prefetched')))
                  for gpu, model, diagonals in compact]
        speedups[baseline] = statistics.geometric_mean(values)
    print(json.dumps(dict(figure9_mean_paired_speedups=matched, figure10_geometric_speedups=speedups,
                          figure11a=dict(training_wins=int(wins), configurations=len(configurations),
                                         inference_mean_percent=100 * statistics.mean(inference_ratios)),
                          figure11b_configuration_medians=attainment), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='configure and build the native benchmark targets')
    build.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    build.add_argument('--nvcomp', action='store_true', help='also build compressor_bench; requires nvCOMP')
    plot = commands.add_parser('figures', help='redraw final-PDF statistics using the six RGB baselines')
    plot.add_argument('--output', type=Path, default=ROOT / 'results/reproduced/paper')
    plot.add_argument('--artifact-root', type=Path, default=ROOT,
                      help='repository containing fastlens/benchmarks/paper/results/')
    stats = commands.add_parser('statistics', help='print published aggregate numbers using the plotting data')
    stats.add_argument('--artifact-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    scratch = Path.home() / 'tmp'
    scratch.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('TMPDIR', str(scratch))
    env.setdefault('MPLCONFIGDIR', str(scratch / 'matplotlib'))
    if args.command == 'statistics':
        os.environ.setdefault('MPLCONFIGDIR', str(scratch / 'matplotlib'))
        published_statistics(args.artifact_root)
    elif args.command == 'figures':
        figures(args.output, args.artifact_root, args.jobs, env)
    else:
        build_dir = args.build_dir.resolve()
        run(['cmake', '-S', ROOT, '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release',
             '-DFASTLENS_BUILD_BENCHMARKS=ON',
             '-DFASTLENS_BUILD_TESTS=ON', '-DFASTLENS_BUILD_TOOLS=ON', '-DFASTLENS_BUILD_TORCH=ON', '-DFASTLENS_WITH_JPEG_DCT=ON',
             f'-DFASTLENS_WITH_NVCOMP={"ON" if args.nvcomp else "OFF"}',
             f'-DPython3_EXECUTABLE={sys.executable}'], env)
        targets = ['fastlens_dct_storage', 'fastlens_raw_dct_control', 'fastlens_block_major_access_tool',
                   '_fastlens_direct_dct', 'fastlens_input_boundary_controls']
        if args.nvcomp:
            targets.append('compressor_bench')
        run(['cmake', '--build', build_dir, '--target', *targets, '--parallel', str(args.jobs)], env)


if __name__ == '__main__':
    main()
