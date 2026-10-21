"""Small command helpers shared by the final-paper entry points."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'fastlens/data'
OUTPUT = ROOT / 'results/reproduced'
PLOTS = ROOT / 'fastlens/docs/paper/scripts'
PYTHON = [sys.executable, '-B']


def parser(description, experiment, actions=('run', 'plot')):
    result = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('action', choices=actions, nargs='?', default='run')
    result.add_argument('--output', type=Path, default=OUTPUT / experiment,
                        help='new measurements or regenerated figures')
    result.add_argument('--artifact-root', type=Path, default=ROOT,
                        help='repository-shaped tree containing the published results, for plot')
    result.add_argument('--dry-run', action='store_true', help='print commands without executing')
    return result


def environment(**overrides):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('TMPDIR', str(Path.home() / 'tmp'))
    env.setdefault('MPLCONFIGDIR', str(Path.home() / 'tmp/matplotlib'))
    env.setdefault('RGBNOMORE_ROOT', str(Path.home() / 'RGB-no-more'))
    env['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT), str(ROOT / 'build/fastlens/torch'), env.get('PYTHONPATH', '')])
    env.update({key: str(value) for key, value in overrides.items()})
    return env


def execute(command, args, env=None):
    print('+', shlex.join(map(str, command)), flush=True)
    if not args.dry_run:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env or environment(), check=True)


def plot(args, *scripts):
    env = environment(FASTLENS_FIGURE_OUTPUT=args.output.resolve(),
                      FASTLENS_ARTIFACT_ROOT=args.artifact_root.resolve())
    for script in scripts:
        name, *options = shlex.split(script)
        execute([*PYTHON, PLOTS / name, *options], args, env)


def gpu_environment(gpu, dry_run=False, **overrides):
    # Resolve the installed card by model name; do not bake the authors' UUIDs into a release.
    from fastlens.benchmarks.common import gpu_uuid
    return environment(**overrides, CUDA_VISIBLE_DEVICES=gpu if dry_run else gpu_uuid(gpu))
