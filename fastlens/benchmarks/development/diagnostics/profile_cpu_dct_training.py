"""Complete the two CPU-DCT training H2D entries in the paper resource table."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import metrics
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import read
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['vitti', 'mobilenet24'], default=['vitti', 'mobilenet24'])
    args = parser.parse_args()
    output = ROOT/'results/section42_resources_20261008'
    for model in args.models:
        source = output/'training/4090'/model/'fastlens/profile'
        destination = output/'training/4090'/model/'cpu-dct/profile'
        if (destination/'measurement.json').exists():
            print('Already measured:', model, flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        if (destination/'run.log').exists():
            raise FileExistsError(f'Inspect the incomplete run before restarting: {destination}')
        metadata = read(source/'run.command.json')
        command = list(metadata['command'])
        # Retain Nsight capture options, NUMA placement, and the training window.
        for index, argument in enumerate(command[:-1]):
            if argument == '--output':
                command[index+1] = str(destination/'capture' if command[index+1].endswith('/capture') else destination)
            elif argument == '--pipeline':
                command[index+1] = 'cpu-dct'
        env = dict(os.environ)
        env.update({key: value for key, value in metadata['environment'].items() if value is not None})
        env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(Path.home()/'tmp'))
        run(command, destination, env, None)
        actual, expected = read(destination/'result.json'), read(source/'result.json')
        assert actual['profile'] and actual['correctness_passed'] and actual['arm'] == 'cpu-dct'
        for key in ('images', 'updates', 'warmup_updates', 'batch', 'accumulation', 'precision',
                    'initial_model_hash', 'sample_order_sha256', 'augmentation_configuration'):
            assert actual[key] == expected[key], (model, key)
        assert actual['images'] == 65536 and actual['worker_count'] == 8
        subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(destination/'capture.sqlite'),
                        str(destination/'capture.nsys-rep')], check=True)
        record = dict(phase='training', gpu='4090', model=model, pipeline='CPU-DCT',
            source=str(destination/'result.json'), mode='profile', images=actual['images'],
            **metrics(destination, 'profile', actual['images']),
            scope='CPU-DCT training H2D diagnostic: existing JPEG/DCT adapters; same model, B6 sample order, '
                  'crop/augmentation recipe, precision and 65536-image window as the FastLens resource run; '
                  '32 excluded warmup updates; not a throughput measurement')
        save(destination/'measurement.json', record)
        records = [read(path) for path in sorted(output.glob('*/*/*/*/*/measurement.json'))]
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output/'measurements.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print('H2D', model, record['h2d_bytes'], 'bytes;', record['h2d_bytes_per_image']/1e6, 'MB/image', flush=True)


if __name__ == '__main__':
    main()
