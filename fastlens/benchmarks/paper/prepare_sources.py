"""Freeze the paper's JPEG-B8 populations and build their full-frequency FS stores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from itertools import islice
import json
import os
from pathlib import Path

from .common import DATA


def prepare(split, args, workers):
    from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact, write_json
    from fastlens.benchmarks.baselines.dct_baselines.source import freeze_source
    from fastlens.benchmarks.baselines.dct_baselines.store import build_store

    count = args.train_count if split == 'train' else args.val_count
    destination = args.output / ('train8194' if split == 'train' else 'val50000')
    destination.mkdir(parents=True, exist_ok=False)
    if split == 'train':
        with args.mapping.open() as stream:
            samples = [dict(logical_sample_id=row['logical_sample_id'], label=int(row['label']),
                            path=str(args.jpeg_root / row['logical_sample_id']),
                            galp_image_id=int(row['planned_physical_position']))
                       for row in islice(csv.DictReader(stream), count)]
    else:
        population = json.loads(args.val_selection.read_text())['samples']
        samples = [dict(row, path=str(args.jpeg_root / row['logical_sample_id'])) for row in population[:count]]
    if len(samples) != count:
        raise ValueError(f'{split}: expected {count} source images, found {len(samples)}')
    selection = destination / 'selection.json'
    write_json(selection, dict(split=split, samples=samples))
    # Existing ImageNet-512 source recipe; this command extracts q/Q without re-encoding JPEGs.
    recipe = dict(resize='Pillow BILINEAR 512x512', encoder='Pillow JPEG',
                  quality=75, subsampling='4:2:0')
    source_path = destination / 'source.json'
    source = freeze_source(selection, source_path, recipe, workers=workers)
    build_store(source, destination / 'fs_matched_b8', 'fs_matched_b8',
                group_size=256, dtype_policy='paper', source_artifact=artifact(source_path), workers=workers)
    print(f'{split}: {len(samples)} images; {destination}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--splits', nargs='+', choices=('train', 'val'), default=['train', 'val'])
    p.add_argument('--jpeg-root', type=Path, default=DATA / 'system_rgbnomore/e2e_v3/imagenet_512')
    p.add_argument('--mapping', type=Path,
                   default=DATA / 'compressed/imagenet512_train_block_major_premixed/ordered_mapping.csv')
    p.add_argument('--val-selection', type=Path,
                   default=DATA / 'system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json')
    p.add_argument('--output', type=Path, default=DATA / 'paper')
    p.add_argument('--train-count', type=int, default=8194, help='smaller values are development smoke tests only')
    p.add_argument('--val-count', type=int, default=50000, help='smaller values do not reproduce Table 2')
    p.add_argument('--workers', type=int, help='workers per split; default divides available CPU cores across splits')
    args = p.parse_args()
    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 1) // len(args.splits))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    with ThreadPoolExecutor(max_workers=len(args.splits)) as pool:
        list(pool.map(lambda split: prepare(split, args, workers), args.splits))


if __name__ == '__main__':
    main()
