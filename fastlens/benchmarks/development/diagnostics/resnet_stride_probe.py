"""Compare existing ResNet checkpoints at layer2 strides 1/2 on an idle H100.

Also fill the missing RTX 4090 model-only references for the deployment table.
The model change is local to each worker; upstream constructors are unchanged.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def worker(args):
    os.environ['DCTNET_PROFILE'] = args.profile
    from fastlens.benchmarks.dct_models import backend

    original = backend.model

    def model(*positional, **keywords):
        net = original(*positional, **keywords)
        block = net.model[1][0]
        block.conv2.stride = (args.stride, args.stride)
        block.downsample[0].stride = (args.stride, args.stride)
        block.stride = args.stride
        return net

    backend.model = model
    if args.worker == 'validation':
        from fastlens.benchmarks.dct_models import evaluate_shards
        sys.argv = [__file__, '--new-data', str(ROOT / 'fastlens/data/compressed/imagenet512_val_block_major'),
                    '--count', '50000', '--batch-size', '64', '--input-geometry', 'source512',
                    '--pushdown', 'off', '--crop-execution-mode', 'vector-range-read-selected-decode',
                    '--output-layout', 'projected', '--native-runtime', 'b6', '--physical-prefix',
                    '--baseline-dir', str(args.output / 'unused'), '--output-dir', str(args.output)]
        evaluate_shards.main()
        result = args.output / 'N_50000.json'
    else:
        from fastlens.benchmarks.development.diagnostics import model_only
        result = args.output / 'result.json'
        sys.argv = [__file__, '--model', args.profile, '--domain', 'dct', '--batch-size', '64',
                    '--output', str(result)]
        model_only.main()
    record = json.loads(result.read_text())
    record.update(layer2_stride=args.stride, weights='original checkpoint; no fine-tuning')
    result.write_text(json.dumps(record, indent=2) + '\n')


def publish_references(output):
    from fastlens.benchmarks.development.reports.populate_paper import deployment_inference_table

    path = ROOT / 'results/paper/deployment_model_only.csv'
    with path.open() as stream:
        references = list(csv.DictReader(stream))
    models = {'eFUN-DCT': 'efun', 'MobileNetV2-DCT32': 'mobilenet32',
              'ResNet-50-DCT24': 'resnet24', 'ResNet-50-DCT64': 'resnet64'}
    for row in references:
        for domain in ('rgb', 'dct'):
            if row[f'{domain}_images_per_s']:
                continue
            assert row['gpu'] == 'RTX 4090'
            model = models[row['dct_model']]
            if domain == 'rgb' and model == 'resnet64':
                model = 'resnet24'
            directory = output / '4090' / model / f'model_only_{domain}'
            candidates = [p for p in directory.rglob('result.json')
                          if json.loads((p.parent / 'run.command.json').read_text()).get('returncode') == 0]
            result, = candidates
            observations = [json.loads(line) for line in
                            (result.parent / 'concurrent_load.jsonl').read_text().splitlines()]
            assert not any(o.get('gpu') == 'GPU-40c637bd-acf5-ea1a-0df8-617138228467'
                           and o.get('blocking') for sample in observations for o in sample.get('observations', []))
            data = json.loads(result.read_text())
            assert data['precision'] == 'fp32' and data['batch_size'] == 64
            row[f'{domain}_images_per_s'] = statistics.median(t['images_per_s'] for t in data['trials'])
            row[f'{domain}_raw_result'] = str(result.relative_to(ROOT))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=references[0])
        writer.writeheader()
        writer.writerows(references)
    with (ROOT / 'results/paper/deployment_inference.csv').open() as stream:
        deployment = list(csv.DictReader(stream))
    deployment = [r for gpu in ('RTX 4090', 'H100') for r in deployment
                  if r['gpu'] == gpu and r['FastLens']]
    with (ROOT / 'results/paper/matched_dct_inference.csv').open() as stream:
        matched = list(csv.DictReader(stream))
    manuscript = ROOT / 'fastlens/docs/paper/sections/07_evaluation.tex'
    text = manuscript.read_text()
    start = text.rfind(r'\begin{table*}', 0, text.index(r'\label{tab:deployment-inference}'))
    end = text.index(r'\end{table*}', start) + len(r'\end{table*}')
    manuscript.write_text(text[:start] + deployment_inference_table(deployment, matched) + text[end:])
    subprocess.run(['bash', str(ROOT / 'fastlens/docs/paper/scripts/build_paper.sh'),
                    str(Path.home() / 'tmp/resnet-stride-paper-build')], cwd=ROOT, check=True)
    print('COMPLETE: deployment model-only references, table, and PDF updated', flush=True)


def run_suite(output):
    from fastlens.benchmarks.execution import inference_load
    from fastlens.benchmarks.system_dct_major.common import parse_manifest

    class TargetGpuGuard(inference_load.ResourceGuard):
        def blockers(self, owned=()):
            blocked = super().blockers(owned)
            # This experiment explicitly permits activity on other GPUs.
            target_pids = {o['pid'] for o in self.observations if o.get('gpu') == self.gpu_uuid}
            return blocked & target_pids

    inference_load.ResourceGuard = TargetGpuGuard
    run = inference_load.run
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'fastlens/data/compressed/imagenet512_val_block_major/manifest.bin'
    paths = [Path(shard['fls_path']) for shard in parse_manifest(source)['shards']]
    devices = {'H100': 'GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76',
               '4090': 'GPU-40c637bd-acf5-ea1a-0df8-617138228467'}
    prefix = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable]
    env = dict(os.environ, PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
               TMPDIR=str(Path.home() / 'tmp'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')

    def measure(destination, command, environment, input_paths=None):
        # Other users' recurring jobs interrupted this probe. Preserve each
        # attempt and reuse only a finished, uncontended process result.
        attempt = destination
        number = 0
        while True:
            metadata = attempt / 'run.command.json'
            if metadata.exists():
                if json.loads(metadata.read_text()).get('returncode') == 0:
                    return
            else:
                attempt.mkdir(parents=True, exist_ok=True)
                actual = [s.replace(str(destination), str(attempt)) for s in command]
                try:
                    run(actual, attempt, environment, input_paths)
                    return
                except RuntimeError as error:
                    if not str(error).startswith('Target GPU contention:'):
                        raise
                    print(f'INTERRUPTED {attempt}: {error}; waiting before a fresh attempt', flush=True)
                    time.sleep(30)
            number += 1
            attempt = destination / f'attempt_{number}'

    for profile in ('resnet24', 'resnet64'):
        for stride in (1, 2):
            for mode in ('validation', 'model-only'):
                destination = output / 'H100' / profile / f'stride{stride}' / mode
                command = [*prefix, str(Path(__file__).resolve()), '--worker', mode,
                           '--profile', profile, '--stride', str(stride), '--output', str(destination)]
                measure(destination, command, dict(env, CUDA_VISIBLE_DEVICES=devices['H100']),
                    paths if mode == 'validation' else None)
                print(f'COMPLETE H100 {profile} stride={stride} {mode}', flush=True)
    # RGB models are shared by the 24/32 and 24/64 channel variants.
    references = [('efun', 'rgb'), ('efun', 'dct'), ('mobilenet32', 'dct'),
                  ('resnet24', 'rgb'), ('resnet24', 'dct'), ('resnet64', 'dct')]
    for profile, domain in references:
        destination = output / '4090' / profile / f'model_only_{domain}'
        command = [*prefix, str(ROOT / 'fastlens/benchmarks/development/diagnostics/model_only.py'), '--model', profile,
                   '--domain', domain, '--batch-size', '50' if profile == 'swinv2' else '64',
                   '--output', str(destination / 'result.json')]
        measure(destination, command, dict(env, CUDA_VISIBLE_DEVICES=devices['4090']))
        print(f'COMPLETE 4090 {profile} {domain} model-only', flush=True)
    publish_references(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('validation', 'model-only'))
    parser.add_argument('--profile', choices=('resnet24', 'resnet64'))
    parser.add_argument('--stride', type=int, choices=(1, 2))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        if args.profile is None or args.stride is None:
            parser.error('worker requires --profile and --stride')
        worker(args)
    else:
        run_suite(args.output)
