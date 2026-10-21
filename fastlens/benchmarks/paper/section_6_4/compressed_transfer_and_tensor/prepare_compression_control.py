#!/usr/bin/env python3
"""Prepare identical compressed/raw-int16 shard subsets for causal ablations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'fastlens/data/compressed/imagenet512_val_block_major')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shard-ids', type=int, nargs='+', default=[0, 16, 32, 48])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = parse_manifest(args.source_root / 'manifest.bin')
    all_samples = json.loads((args.source_root / 'samples.json').read_text())
    selected = [manifest['shards'][index] for index in args.shard_ids]
    samples = [sample for shard in selected for sample in all_samples[
        shard['first_global_image_index']:shard['first_global_image_index'] + shard['image_count']]]
    names = [f'shard_{index:06d}' for index in range(len(selected))]
    compressed, raw = (args.output_root / name for name in ('compressed', 'raw_int16'))
    for directory in (compressed, raw):
        directory.mkdir()
        (directory / 'samples.json').write_text(json.dumps(samples) + '\n')
        shutil.copyfile(args.source_root / 'profile.json', directory / 'profile.json')

    def run(command, log):
        subprocess.run([sys.executable, str(ROOT / 'fastlens/benchmarks/execution/run_logged.py'), '--log', str(log), '--',
                        *map(str, command)], check=True)

    for shard, name in zip(selected, names):
        source = Path(shard['fls_path'])
        os.link(source, compressed / (name + '.fls'))
        for directory in (compressed, raw):
            shutil.copyfile(shard['metadata_path'], directory / (name + '.meta.bin'))
        prior=ROOT/'results/preparation/raw_int16/shard_000000_schema_retry.log'
        verified=json.loads(prior.read_text().strip().splitlines()[-1]) if prior.exists() else None
        if verified and Path(verified['source']).resolve()==source.resolve() and verified['coefficients_and_rowgroups_exact']:
            os.link(ROOT/verified['destination'],raw/(name+'.fls'))
            (raw/(name+'.reuse.json')).write_text(json.dumps(dict(raw_log=str(prior),verification=verified),indent=2)+'\n')
        else:
            run([ROOT / 'build/fastlens/benchmarks/fastlens_raw_dct_control', source, raw / (name + '.fls')],
                raw / (name + '.log'))
    for directory in (compressed, raw):
        run([ROOT / 'build/fastlens/benchmarks/fastlens_dct_storage', 'merge-block', directory, *names],
            directory / 'manifest.log')
        run([ROOT / 'build/fastlens/tools/jpeg_dct/fastlens_block_major_access_tool', directory / 'manifest.bin',
             '--output-dir', directory / 'access', '--output-json', directory / 'access/build.json'],
            directory / 'index.log')
    (args.output_root / 'preparation.json').write_text(json.dumps(dict(
        source_manifest=str(args.source_root / 'manifest.bin'), source_shard_ids=args.shard_ids,
        samples=len(samples), compressed=str(compressed), raw_int16=str(raw),
        coefficient_and_rowgroup_verification='each raw shard command must succeed before this file is written',
        metadata='byte-identical copies; the same subset manifest builder and indexer for both representations',
        scope='representative shard subset; not full ImageNet accuracy or full-pass throughput'), indent=2) + '\n')


if __name__ == '__main__':
    main()
