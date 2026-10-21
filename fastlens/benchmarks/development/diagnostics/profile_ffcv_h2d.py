"""Measure JPEG-backed FFCV H2D traffic for the four RTX 4090 Table 4 entries."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import inference_inputs
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import metrics
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import read
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import save
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import set_arg
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ffcv_h2d_20261014')
    args = parser.parse_args()
    output = args.output.resolve()
    records = []
    for phase in ('training', 'inference'):
        for model in ('vitti', 'mobilenet24'):
            destination = output / phase / model
            destination.mkdir(parents=True, exist_ok=True)
            measurement = destination / 'measurement.json'
            if measurement.exists():
                records.append(read(measurement))
                continue
            assert not (destination / 'run.log').exists(), f'Incomplete run retained: {destination}'
            summary = read(ROOT / 'results/ffcv_jpeg_20261005' / phase / '4090/summary.json')
            model_name = 'ViT-Ti RGB' if model == 'vitti' else 'MobileNetV2 RGB'
            selected, = [row for row in summary if row['model'] == model_name]
            source = ROOT / selected['raw_results'][0]
            filename = source.name
            metadata = read(source.parent / 'run.command.json')
            job = dict(phase=phase, gpu='4090', model=model, pipeline='FFCV', source=str(source))
            command = list(metadata['command'])
            env = dict(os.environ)
            env.update({k: v for k, v in metadata['environment'].items() if v is not None})
            for key in ('FASTLENS_EVALUATION_SNAPSHOT', 'FASTLENS_ALLOWED_RESIDENT_PIDS'):
                env.pop(key, None)
            env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(Path.home() / 'tmp'),
                       PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', DCTNET_PROFILE=model)
            result = destination / filename
            set_arg(command, '--output', destination if phase == 'training' else result)
            paths = None
            if phase == 'training':
                index = command.index('--correctness-report')
                del command[index:index + 2]
                command.append('--profile')
            else:
                paths = inference_inputs(job, command, destination)
                index = next(i for i, value in enumerate(command) if Path(value).name == 'python')
                save(destination / 'workload_command.json', command[index + 1:])
                command = command[:index + 1] + [
                    'fastlens/benchmarks/paper/section_6_3/h2d_traffic/profile_section42.py', '--command', str(destination / 'workload_command.json'),
                    '--mode', 'profile', '--output', str(destination)]
            command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                       '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                       '--output', str(destination / 'capture'), *command]
            print('PROFILE FFCV JPEG', phase, model, flush=True)
            run(command, destination, env, paths)
            images = verify(job, read(result))
            subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(destination / 'capture.sqlite'),
                            str(destination / 'capture.nsys-rep')], check=True)
            record = dict(job, pipeline='FFCV JPEG', mode='profile', images=images,
                          **metrics(destination, 'profile', images),
                          scope='Total CUPTI H2D inside fastlens.measured_pass; identical window boundaries '
                                'to Table 4 raw FFCV; excludes model warmup; diagnostic timings are not throughput results')
            save(measurement, record)
            records.append(record)
            print('H2D', phase, model, record['h2d_bytes_per_image'] / 1e6, 'MB/image', flush=True)
    with (output / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
