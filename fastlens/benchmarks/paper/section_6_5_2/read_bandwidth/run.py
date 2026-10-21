"""Section 6.5.2: shared read-completion limits of 125/500/2,000 MB/s."""
from pathlib import Path

from fastlens.benchmarks.paper.common import ROOT, PYTHON, execute, parser, plot


def main():
    p = parser(__doc__, 'section_6_5_2/read_bandwidth')
    p.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), default=['4090', 'H100'])
    p.add_argument('--libraries', type=Path, default=ROOT / 'build/fastlens/benchmarks')
    p.add_argument('--crops', type=int, nargs='+', choices=(448, 160), default=[448, 160])
    p.add_argument('--rates', type=int, nargs='+', choices=(125, 500, 2000), default=[125, 500, 2000])
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, 'plot_read_bandwidth.py')
        return
    execute([*PYTHON, ROOT / 'fastlens/benchmarks/paper/shared/run_remaining_reruns.py', '--experiments', 'bandwidth',
             '--gpu', *args.gpus, '--libraries', args.libraries, '--boundary-crops', *args.crops,
             '--read-bandwidths', *args.rates, '--boundary-trials', args.repeats,
             '--output', args.output], args)


if __name__ == '__main__':
    main()
