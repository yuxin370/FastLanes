"""Measure the JPEG-backed FFCV baseline with the existing paper runners."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution import inference_load
from fastlens.benchmarks.system_dct_major.common import file_identity, sha256_json

GPUS = {'4090': 'GPU-40c637bd-acf5-ea1a-0df8-617138228467',
        'H100': 'GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76'}
MODELS = {'vitti': 'ViT-Ti RGB', 'swinv2': 'SwinV2-T RGB',
          'mobilenet24': 'MobileNetV2 RGB', 'efun': 'EfficientNet-B0 reference',
          'resnet24': 'ResNet-50 RGB'}


class TargetGpuGuard(inference_load.ResourceGuard):
    def blockers(self, owned=()):
        blocked = super().blockers(owned)
        target = {o['pid'] for o in self.observations if o.get('gpu') == self.gpu_uuid}
        return blocked & target


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=GPUS, required=True)
    parser.add_argument('--task', choices=['inference', 'training'], required=True)
    parser.add_argument('--storage', choices=['jpeg', 'raw'], default='jpeg')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--output', type=Path, default=ROOT/'results/ffcv_jpeg_20261005')
    args = parser.parse_args()
    inference_load.ResourceGuard = TargetGpuGuard
    output = args.output.resolve()
    task_directory = args.task if args.storage == 'jpeg' else args.task+'_raw'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[args.gpu],
               PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TMPDIR=str(Path.home()/'tmp'))
    summaries = []
    gpu = 'RTX 4090' if args.gpu == '4090' else 'H100'
    sources = read_csv(ROOT/'results/p0/deployment_inference_summary.csv')
    for profile in args.models:
        env['DCTNET_PROFILE'] = profile
        model_output = output/task_directory/args.gpu/profile
        model_output.mkdir(parents=True, exist_ok=True)
        rates, accuracies, results = [], [], []
        if args.task == 'inference':
            source, = [s for s in sources if (s['gpu'],s['model'],s['pipeline']) == (gpu,MODELS[profile],'FFCV')]
            reference = ROOT/source['raw_results'].split(';')[0].split('#')[0]
            contract = json.loads((reference.parent/'contract.json').read_text())
            old = json.loads(reference.read_text())['repeats'][0]
            manifest = json.loads(Path(contract['dataset']['sample_manifest']).read_text())
            entries = manifest['samples']
            stored = json.loads((output/'preparation/val/samples.json').read_text())
            positions = {s['path']:i for i,s in enumerate(stored)}
            assert len(positions) == len(entries) == 50000
            indices = [positions[s['path']] for s in entries]
            remap = [0]*len(stored)
            for sample,index in zip(entries,indices):
                remap[stored[index]['ordinal']] = sample['ordinal']
            # Source files have moved since the historical runs; record current
            # filesystem identities in this run's manifest, retaining sample order.
            for sample in entries:
                identity = file_identity(Path(sample['path']))
                assert identity['size_bytes'] == sample['size_bytes']
                sample['file_identity'] = identity
            manifest_path = model_output/'samples.json'
            manifest_path.write_text(json.dumps(manifest)+'\n')
            contract['dataset'].update(sample_manifest=str(manifest_path),
                                       sample_manifest_sha256=sha256_json(manifest))
            beton = Path(contract['pipelines']['ffcv']['beton'])
            if args.storage == 'jpeg':
                beton = output/'preparation/val/val.beton'
                contract['pipelines']['ffcv'].update(beton=str(beton), indices=indices, ordinal_remap=remap)
            original = json.loads((reference.parent/'run.command.json').read_text())['command']
        else:
            beton = (output/'preparation/train/train_window.beton' if args.storage == 'jpeg'
                     else ROOT/'results/preparation/p0_ffcv/train_window.beton')
            source = None
        for trial in (range(1,4) if args.task == 'inference' else range(-1,3)):
            base = model_output/('check' if trial == -1 else f'trial_{trial}')
            directory, attempt = base, 0
            while (directory/'concurrent_load.jsonl').exists():
                command_file = directory/'run.command.json'
                if command_file.exists() and json.loads(command_file.read_text()).get('returncode') == 0:
                    break
                attempt += 1
                directory = base/f'attempt_{attempt}'
            directory.mkdir(parents=True, exist_ok=True)
            result = directory/('pipeline_ffcv.json' if args.task == 'inference' else 'result.json')
            if args.task == 'inference':
                contract_path = directory/'contract.json'
                contract_path.write_text(json.dumps(contract, indent=2)+'\n')
                command = list(original)
                command[command.index('--contract')+1] = str(contract_path)
                command[command.index('--output')+1] = str(result)
            else:
                command = ['numactl','--cpunodebind=0','--membind=0',sys.executable,
                           'fastlens/benchmarks/paper/shared/training_window_rgb.py','--gpu',args.gpu,'--model',profile,
                           '--pipeline','ffcv','--trial',str(max(0,trial)),'--output',str(directory),
                           '--ffcv-root',str(beton.parent)]
                command += ['--check'] if trial == -1 else ['--correctness-report',str(check_result)]
            if not result.exists():
                inference_load.run(command, directory, env, [beton] if args.task == 'inference' else None)
            metadata = json.loads((directory/'run.command.json').read_text())
            assert metadata['returncode'] == 0
            data = json.loads(result.read_text())
            if args.task == 'inference':
                record, = data['repeats']
                assert record['images'] == 50000 and record['sample_trace'] == old['sample_trace']
                if rates:
                    assert record['top1_predictions_sha256'] == prediction_hash
                prediction_hash = record['top1_predictions_sha256']
                rates.append(record['images']/record['repeat_scope_seconds'])
                accuracies.append(100*record['accuracy_top1'])
            else:
                assert data['correctness_passed']
                if trial == -1:
                    check_result = result
                    continue
                assert data['images'] == 65536 and data['updates'] == 64
                rates.append(data['images']/data['seconds'])
            results.append(str(result.relative_to(ROOT)))
            print(f'COMPLETE {args.task} {gpu} {profile} trial {trial}: {rates[-1]:.1f} images/s', flush=True)
        summaries.append(dict(gpu=gpu, model=MODELS[profile], task=args.task, storage='FFCV-JPEG original bytes' if args.storage == 'jpeg' else 'FFCV raw RGB',
                              trials=3, samples=50000 if args.task == 'inference' else 65536,
                              median_images_per_s=statistics.median(rates), min_images_per_s=min(rates),
                              max_images_per_s=max(rates), top1=statistics.median(accuracies) if accuracies else None,
                              raw_reference_images_per_s=float(source['median_images_per_s']) if source else None,
                              raw_reference_top1=float(source['top1']) if source else None,
                              raw_results=results))
        path = output/task_directory/args.gpu/'summary.json'
        path.write_text(json.dumps(summaries, indent=2)+'\n')


if __name__ == '__main__':
    while True:
        try:
            main()
            break
        except RuntimeError as error:
            if not str(error).startswith('Target GPU contention:'):
                raise
            print(f'{error}; retaining interrupted attempt, waiting to resume', flush=True)
            time.sleep(30)
