"""Accepted full-population references used by paper correctness checks."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

def read(path):
    return json.loads(Path(path).read_text())


def inference_reference(gpu, model, path):
    """Fixed accepted semantic references; their historical speed is never used."""
    if path == 'imagelanes':
        name = 'pipeline_dct_major_pushdown.json' if model == 'vitti' else 'N_50000.json'
        return ROOT/f'results/imagelanes_remaining_20261008/inference/{gpu}/{model}/trial_0/{name}'
    if path == 'cpu_dct':
        name = 'pipeline_rgbnomore.json' if model == 'vitti' else 'J_50000.json'
        return ROOT/f'results/remaining_inference_recheck_20260929/inference/{gpu}_{model}_cpu-dct/trial_1/{name}'
    if path == 'l3':
        name = 'result.json' if model == 'vitti' else 'RGB_l3_50000.json'
        return ROOT/f'results/l3_patch32_20261006/inference/{gpu}/{model}/l3/trial_0/{name}'
    if path == 'ffcv_jpeg':
        folder = ROOT/f'results/ffcv_jpeg_20261005/inference/{gpu}/{model}'
        return sorted(folder.glob('trial_*/**/pipeline_ffcv.json'))[0]
    labels = {'vitti': 'ViT-Ti RGB', 'mobilenet24': 'MobileNetV2 RGB', 'efun': 'EfficientNet-B0 reference'}
    names = {'pytorch': 'PyTorch', 'dali': 'DALI', 'coordl': 'CoorDL', 'ffcv_raw': 'FFCV'}
    rows = list(csv.DictReader((ROOT/'results/p0/deployment_inference_summary.csv').open()))
    row, = [r for r in rows if r['gpu'] == ('RTX 4090' if gpu == '4090' else gpu)
            and r['model'] == labels[model] and r['pipeline'] == names[path]]
    return ROOT/row['raw_results'].split(';')[0].split('#')[0]


def expected_predictions(gpu, path):
    import numpy as np
    if path == 'imagelanes':
        data = read(ROOT/f'results/imagelanes_remaining_20261008/inference/{gpu}/swinv2/trial_0/result.json')
        contract = read(ROOT/'results/latest_inference_20260930/inference/4090_swinv2_imagelanes/trial_1/contract.json')
        samples = sorted(read(contract['dataset']['sample_manifest'])['samples'], key=lambda s: s['galp_image_id'])
        predictions = [p[0] for p in data['predictions']]
    else:
        if path == 'l3':
            source = ROOT/f'results/l3_patch32_20261006/inference/{gpu}/swinv2/l3/trial_0/result.json'
        elif path == 'ffcv_jpeg':
            trial = 2 if gpu == 'H100' else 1
            source = ROOT/f'results/ffcv_jpeg_20261005/inference/{gpu}/swinv2/trial_{trial}/pipeline_ffcv.json'
        else:
            folder = 'cpu-dct' if path == 'cpu_dct' else 'ffcv' if path == 'ffcv_raw' else path
            filename = 'rgbnomore' if path == 'cpu_dct' else folder
            source = ROOT/f'results/remaining_inference_recheck_20260929/inference/{gpu}_swinv2_{folder}/trial_1/pipeline_{filename}.json'
        data = read(source)
        contract = read(data['contract'])
        samples = read(contract['dataset']['sample_manifest'])['samples']
        predictions = np.load(data['semantic_artifact'])['top1_predictions'].reshape(-1).tolist()
    assert len(samples) == len(predictions) == 50000
    return {s['sample_id']: p for s, p in zip(samples, predictions)}


PROFILES = {'ViT-Ti DCT': 'vitti', 'ViT-Ti RGB': 'vitti',
    'SwinV2-T DCT': 'swinv2', 'SwinV2-T RGB': 'swinv2',
    'EfficientNet-B0 reference': 'efun', 'eFUN-DCT': 'efun',
    'MobileNetV2 RGB': 'mobilenet24', 'MobileNetV2-DCT24': 'mobilenet24',
    'MobileNetV2-DCT32': 'mobilenet32', 'ResNet-50 RGB': 'resnet24',
    'ResNet-50-DCT24': 'resnet24', 'ResNet-50-DCT64': 'resnet64'}
