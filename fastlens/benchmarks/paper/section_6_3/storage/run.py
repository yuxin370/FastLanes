"""Table 3 storage: source JPEG, FastLens files/indexes and the FFCV store."""
from pathlib import Path

from fastlens.benchmarks.paper.common import DATA, ROOT, PYTHON, execute, parser


def main():
    p = parser(__doc__, 'section_6_3/storage', ('run', 'table'))
    p.add_argument('--data-root', type=Path, default=DATA)
    p.add_argument('--ffcv-beton', type=Path,
                   default=ROOT / 'fastlens/data/paper/ffcv/val.beton')
    args = p.parse_args()
    if args.action == 'table':
        print((args.artifact_root / 'fastlens/benchmarks/paper/results/section_6_3/storage_table3.csv').read_text(), end='')
        return
    execute([*PYTHON, ROOT / 'fastlens/benchmarks/paper/section_6_3/storage/storage.py', '--data-root', args.data_root,
             '--ffcv-beton', args.ffcv_beton, '--output', args.output], args)


if __name__ == '__main__':
    main()
