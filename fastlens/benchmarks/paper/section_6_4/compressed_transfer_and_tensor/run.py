"""Figure 15: raw/compressed transfer and grid/direct-tensor controls."""
from fastlens.benchmarks.paper.common import ROOT, PYTHON, execute, parser, plot


def main():
    p = parser(__doc__, 'section_6_4/compressed_transfer_and_tensor')
    p.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), default=['4090', 'H100'])
    p.add_argument('--timing-only', action='store_true', help='omit separate Nsight and memory diagnostic passes')
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, 'plot_core_ablations.py')
        return
    command = [*PYTHON, ROOT / 'fastlens/benchmarks/paper/shared/run_pipeline_rerun.py', '--experiments', 'core',
               '--gpu', *args.gpus, '--output', args.output]
    if args.timing_only:
        command.append('--timing-only')
    execute(command, args)


if __name__ == '__main__':
    main()
