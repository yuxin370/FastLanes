"""Figure 12: 300-epoch shared-PLS and per-sample crop training."""
import hashlib
from pathlib import Path

from fastlens.benchmarks.paper.common import DATA, PYTHON, execute, parser, plot


def main():
    p = parser(__doc__, 'section_6_2_5/training_convergence')
    p.add_argument('--models', nargs='+', choices=('swinv2', 'mobilenet24'), default=['swinv2', 'mobilenet24'])
    p.add_argument('--crop-policies', nargs='+', choices=('per-pls', 'per-sample'), default=['per-pls', 'per-sample'])
    p.add_argument('--nproc-per-node', type=int, default=8)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--seed', type=int, default=11997733)
    p.add_argument('--microbatch', type=int, default=64)
    p.add_argument('--global-batch', type=int, default=1024)
    p.add_argument('--manifest', type=Path,
                   default=DATA / 'compressed/imagenet512_train_block_major_premixed/dct/manifest.bin')
    p.add_argument('--mapping', type=Path,
                   default=DATA / 'compressed/imagenet512_train_block_major_premixed/ordered_mapping.csv')
    p.add_argument('--validation-root', type=Path, default=DATA / 'compressed')
    args = p.parse_args()
    if args.action == 'plot':
        plot(args, 'plot_training_convergence.py')
        return
    mapping_hash = '<SHA256 of mapping>' if args.dry_run else hashlib.sha256(args.mapping.read_bytes()).hexdigest()
    for model in args.models:
        validation = ('imagenet512_val_block_major' if model == 'swinv2'
                      else 'backup/offline_model_input/imagenet512_val_mobilenet112_block_major')
        for policy in args.crop_policies:
            execute([*PYTHON, '-m', 'torch.distributed.run', '--standalone',
                     '--nproc-per-node', args.nproc_per_node, '-m', 'fastlens.benchmarks.training_pls.distributed',
                     '--model', model, '--manifest', args.manifest, '--mapping', args.mapping,
                     '--mapping-sha256', mapping_hash, '--validation-data', args.validation_root / validation,
                     '--epochs', args.epochs, '--seed', args.seed, '--microbatch', args.microbatch,
                     '--global-batch', args.global_batch, '--segments-per-pool', 4,
                     '--crop-policy', policy, '--precision', 'bf16', '--io-backend', 'pread',
                     '--torch-threads', 4, '--validation-count', 50000,
                     '--output-dir', args.output / model / policy], args)


if __name__ == '__main__':
    main()
