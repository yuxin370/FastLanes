#!/usr/bin/env python3
"""Prepare the same contiguous physical image range for all Figure 15 controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.system_dct_major.common import parse_manifest
from fastlens.cli import positive_int


def select_shards(manifest, image_count, start_image=0, shard_ids=None):
    """Keep whole, consecutive source partitions so rowgroup geometry is unchanged."""
    shards = manifest['shards']
    if shard_ids is None:
        selected = [shard for shard in shards
                    if start_image <= shard['first_global_image_index'] < start_image + image_count]
    else:
        by_id = {shard['shard_id']: shard for shard in shards}
        if any(index not in by_id for index in shard_ids):
            raise ValueError('selected partition is absent from the source manifest')
        selected = [by_id[index] for index in shard_ids]
        start_image = selected[0]['first_global_image_index']
    position = start_image
    for shard in selected:
        if shard['first_global_image_index'] != position:
            raise ValueError('select a contiguous physical image range in manifest order')
        position += shard['image_count']
    if not selected or position != start_image + image_count:
        raise ValueError('image range must contain exactly --image-count images in whole source partitions')
    return selected


def prepared_controls(path):
    """Validate the paired experiment inputs before selecting a GPU or measuring."""
    preparation = json.loads(Path(path).read_text())
    compressed, raw = (Path(preparation[key]) for key in ('compressed', 'raw_int16'))
    samples = json.loads((compressed / 'samples.json').read_text())
    if len(samples) != preparation['samples'] or samples != json.loads((raw / 'samples.json').read_text()):
        raise ValueError('control datasets have different sample counts or physical sample order')
    if (compressed / 'profile.json').read_bytes() != (raw / 'profile.json').read_bytes():
        raise ValueError('control datasets have different profiles')
    layouts = []
    for root in (compressed, raw):
        manifest = parse_manifest(root / 'manifest.bin')
        if manifest['image_count'] != len(samples):
            raise ValueError('control manifest count differs from prepared sample order')
        layouts.append([(s['first_global_image_index'], s['image_count'], s['rowgroup_count'],
                         s['real_row_count'], s['physical_row_count']) for s in manifest['shards']])
    if layouts[0] != layouts[1]:
        raise ValueError('compressed/raw control partition or rowgroup geometry differs')
    return preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='native Block-major dataset with samples.json and profile.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--image-count', type=positive_int, default=16384)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--start-image', type=int, default=0, help='first physical image, on a partition boundary')
    selection.add_argument('--shard-ids', type=int, nargs='+', help='explicit consecutive source partition IDs')
    parser.add_argument('--workers', type=positive_int, default=1,
                        help='parallel partitions; each raw rewrite materializes a complete partition')
    args = parser.parse_args()
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    manifest = parse_manifest(source_root / 'manifest.bin')
    if manifest['version'] != 1:
        parser.error('source must be a native Block-major manifest')
    all_samples = json.loads((source_root / 'samples.json').read_text())
    if not isinstance(all_samples, list) or len(all_samples) != manifest['image_count']:
        parser.error('samples.json must be the complete physical-order sample array')
    json.loads((source_root / 'profile.json').read_text())
    selected = select_shards(manifest, args.image_count, args.start_image, args.shard_ids)
    first = selected[0]['first_global_image_index']
    samples = all_samples[first:first + args.image_count]
    build = args.build_dir.resolve()
    raw_tool = build / 'fastlens/benchmarks/fastlens_raw_dct_control'
    merge_tool = build / 'fastlens/benchmarks/fastlens_dct_storage'
    index_tool = build / 'fastlens/tools/jpeg_dct/fastlens_block_major_access_tool'
    for tool in (raw_tool, merge_tool, index_tool):
        if not tool.is_file() or not os.access(tool, os.X_OK):
            parser.error(f'build required native target first: {tool}')
    output_root.mkdir(parents=True, exist_ok=False)
    compressed, raw = (output_root / name for name in ('compressed', 'raw_int16'))
    for directory in (compressed, raw):
        directory.mkdir()
        (directory / 'samples.json').write_text(json.dumps(samples) + '\n')
        shutil.copyfile(source_root / 'profile.json', directory / 'profile.json')

    def run(command, log):
        subprocess.run([sys.executable, str(ROOT / 'fastlens/benchmarks/execution/run_logged.py'),
                        '--log', str(log), '--', *map(str, command)], check=True)

    names = [f'shard_{index:06d}' for index in range(len(selected))]

    def prepare_partition(item):
        shard, name = item
        source = Path(shard['fls_path'])
        os.link(source, compressed / (name + '.fls'))
        for directory in (compressed, raw):
            shutil.copyfile(shard['metadata_path'], directory / (name + '.meta.bin'))
        run([raw_tool, source, raw / (name + '.fls')], raw / (name + '.log'))

    with ThreadPoolExecutor(max_workers=min(args.workers, len(selected))) as pool:
        list(pool.map(prepare_partition, zip(selected, names)))

    def finalize(directory):
        run([merge_tool, 'merge-block', directory, *names], directory / 'manifest.log')
        run([index_tool, directory / 'manifest.bin', '--output-dir', directory / 'geometry',
             '--output-json', directory / 'geometry/build.json'], directory / 'index.log')

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(finalize, (compressed, raw)))
    result = output_root / 'preparation.json'
    result.write_text(json.dumps(dict(
        source_manifest=str(source_root / 'manifest.bin'),
        source_shard_ids=[shard['shard_id'] for shard in selected], source_image_start=first,
        samples=len(samples), compressed=str(compressed), raw_int16=str(raw),
        coefficient_and_rowgroup_verification='raw writer verifies all real coefficients and rowgroup boundaries',
        metadata='byte-identical copies; same subset manifest builder and geometry indexer',
        scope='contiguous source subset; all four controls use the prepared sample order'), indent=2) + '\n')
    prepared_controls(result)
    print(result)


if __name__ == '__main__':
    main()
