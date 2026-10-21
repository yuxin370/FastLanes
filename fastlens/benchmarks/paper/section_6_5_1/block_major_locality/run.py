"""Figure 16: 4,096 images, shared/correlated/independent crops, pools 64/1,024."""
from fastlens.benchmarks.paper.common import ROOT, PYTHON, execute, parser, plot


def main():
    p = parser(__doc__, 'section_6_5_1/block_major_locality')
    p.add_argument('--gpus', nargs='+', choices=('4090', 'H100'), default=['4090', 'H100'])
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, 'plot_locality.py')
        return
    execute([*PYTHON, ROOT / 'fastlens/benchmarks/paper/shared/run_pipeline_rerun.py', '--experiments', 'locality',
             '--gpu', *args.gpus, '--output', args.output], args)


if __name__ == '__main__':
    main()
