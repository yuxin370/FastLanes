"""Figure 13 and projection controls: separate input-only and end-to-end measurements."""
from pathlib import Path

from fastlens.benchmarks.paper.common import ROOT, PYTHON, execute, parser, plot


def main():
    p = parser(__doc__, 'section_6_4/projection_and_staging')
    p.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), help='override the published GPUs: 4090 for projection, H100 for staging')
    p.add_argument('--experiments', nargs='+', choices=('projection', 'staging'), default=['projection', 'staging'])
    p.add_argument('--libraries', type=Path, default=ROOT / 'build/fastlens/benchmarks')
    p.add_argument('--crops', type=int, nargs='+', choices=(448, 160), default=[448, 160])
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, 'plot_compact_staging.py')
        return
    if 'projection' in args.experiments:
        execute([*PYTHON, ROOT / 'fastlens/benchmarks/paper/shared/run_pipeline_rerun.py', '--experiments', 'frequency',
                 '--gpu', *(args.gpus or ['4090']), '--projection-coefficients', '8', '--output', args.output], args)
    if 'staging' in args.experiments:
        execute([*PYTHON, ROOT / 'fastlens/benchmarks/paper/shared/run_remaining_reruns.py', '--experiments', 'staging',
                 '--gpu', *(args.gpus or ['H100']), '--libraries', args.libraries, '--boundary-crops', *args.crops,
                 '--boundary-trials', args.repeats, '--output', args.output], args)


if __name__ == '__main__':
    main()
