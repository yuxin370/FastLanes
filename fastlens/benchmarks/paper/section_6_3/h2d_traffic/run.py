"""Figure 14: separate CUPTI H2D diagnostics, outside the throughput trials."""
from pathlib import Path
from fastlens.benchmarks.paper.common import ROOT, PYTHON, execute, parser, plot


def main():
    p = parser(__doc__, 'section_6_3/h2d_traffic')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'results/reproduced',
                   help='Completed chapter trials to profile')
    p.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), default=['4090'])
    p.add_argument('--models', nargs='+', choices=('vitti', 'mobilenet24'), default=['vitti', 'mobilenet24'])
    p.add_argument('--phases', nargs='+', choices=('training', 'inference'), default=['training', 'inference'])
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, 'plot_h2d_traffic.py')
        return
    execute([*PYTHON, ROOT / 'fastlens/benchmarks/paper/section_6_3/h2d_traffic/run_section42_resources.py',
             '--run-dir', args.run_dir, '--gpus', *args.gpus, '--models', *args.models, '--phases', *args.phases,
             '--output', args.output], args)


if __name__ == '__main__':
    main()
