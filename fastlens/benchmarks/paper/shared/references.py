"""Reference inputs generated from the current dataset, models and checkpoints."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REFERENCES = ROOT / 'results/rebuilt/references.json'


def reference_inputs(config, gpu, model, path, *, preparing=False):
    """Resolve a workload's files relative to the reference configuration."""
    config = Path(config).resolve()
    records = json.loads(config.read_text())
    pipeline = path.replace('imagelanes', 'fastlens')
    entry = records[gpu][model][pipeline]
    files = {key: (config.parent / entry[key]).resolve()
             for key in ('reference', 'contract') if key in entry}
    if 'reference' not in files:
        raise ValueError(f'{config}: {gpu}/{model}/{pipeline} needs a reference')
    for key, filename in files.items():
        if key == 'reference' and preparing:
            continue
        if not filename.is_file():
            raise FileNotFoundError(filename)
    if model == 'swinv2' and 'contract' not in files:
        raise ValueError(f'{config}: {gpu}/{model}/{pipeline} needs a contract')
    return files


def expected_predictions(reference, contract):
    """Load Swin's full population in the reference's own sample order."""
    import numpy as np
    data = json.loads(Path(reference).read_text())
    source = json.loads(Path(contract).read_text())
    samples = json.loads(Path(source['dataset']['sample_manifest']).read_text())['samples']
    if 'semantic_artifact' in data:
        recorded = json.loads(Path(data['contract']).read_text())
        original = json.loads(Path(recorded['dataset']['sample_manifest']).read_text())['samples']
        assert [(s['sample_id'], s['label']) for s in samples] == [
            (s['sample_id'], s['label']) for s in original]
        with np.load(data['semantic_artifact']) as artifact:
            predictions = artifact['top1_predictions'].reshape(-1).tolist()
        ids = [s['sample_id'] for s in samples]
    else:
        predictions = [p[0] if isinstance(p, list) else p for p in data['predictions']]
        if 'sample_ids' in data:
            ids = data['sample_ids']
        else:
            by_image = {s['galp_image_id']: s['sample_id'] for s in samples}
            ids = [by_image[i] for i in data['image_ids']]
    assert len(ids) == len(predictions) == len(set(ids)) == 50000
    assert set(ids) == {s['sample_id'] for s in samples}
    return dict(zip(ids, predictions))


PROFILES = {'ViT-Ti DCT': 'vitti', 'ViT-Ti RGB': 'vitti',
    'SwinV2-T DCT': 'swinv2', 'SwinV2-T RGB': 'swinv2',
    'EfficientNet-B0 reference': 'efun', 'eFUN-DCT': 'efun',
    'MobileNetV2 RGB': 'mobilenet24', 'MobileNetV2-DCT24': 'mobilenet24',
    'MobileNetV2-DCT32': 'mobilenet32', 'ResNet-50 RGB': 'resnet24',
    'ResNet-50-DCT24': 'resnet24', 'ResNet-50-DCT64': 'resnet64'}
