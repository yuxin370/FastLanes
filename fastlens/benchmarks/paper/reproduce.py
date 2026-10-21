"""Build paper dependencies or redraw fastlens.pdf's measured figures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import subprocess
import sys

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
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    scratch = Path.home() / 'tmp'
    scratch.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('TMPDIR', str(scratch))
    env.setdefault('MPLCONFIGDIR', str(scratch / 'matplotlib'))
    if args.command == 'figures':
        figures(args.output, args.artifact_root, args.jobs, env)
    else:
        build_dir = args.build_dir.resolve()
        run(['cmake', '-S', ROOT, '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release',
             '-DFASTLENS_BUILD_BENCHMARKS=ON',
             '-DFASTLENS_BUILD_TESTS=ON', '-DFASTLENS_BUILD_TOOLS=ON', '-DFASTLENS_BUILD_TORCH=ON', '-DFASTLENS_WITH_JPEG_DCT=ON',
             f'-DFASTLENS_WITH_NVCOMP={"ON" if args.nvcomp else "OFF"}',
             f'-DPython3_EXECUTABLE={sys.executable}'], env)
        targets = ['fastlens_dct_storage', '_fastlens_direct_dct', 'fastlens_input_boundary_controls']
        if args.nvcomp:
            targets.append('compressor_bench')
        run(['cmake', '--build', build_dir, '--target', *targets, '--parallel', str(args.jobs)], env)


if __name__ == '__main__':
    main()
