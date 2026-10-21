"""Sequential paired full-validation Swin runs on the two paper GPUs."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.system_dct_major.common import parse_manifest

OUT = ROOT / 'results/swin_block_major_20261001'
DATA = ROOT / 'fastlens/data/compressed'


def main():
    source_contract = json.loads((ROOT / 'results/latest_inference_20260930/inference/'
                                 '4090_swinv2_fastlens/trial_1/contract.json').read_text())
    samples = json.loads(Path(source_contract['dataset']['sample_manifest']).read_text())['samples']
    paths = {'cpu': [Path(s['path']) for s in samples]}
    for arm, folder in [('block', 'imagenet512_val_block_major'), ('compact', 'imagenet512_val_compact_v3')]:
        paths[arm] = [Path(s['fls_path']) for s in parse_manifest(DATA / folder / 'manifest.bin')['shards']]
    for gpu in ['4090', 'H100']:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), TMPDIR=str(Path.home()/'tmp'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for trial in range(4):
            for arm in (['block', 'compact', 'cpu'] if trial % 2 == 0 else ['cpu', 'compact', 'block']):
                directory = OUT / gpu / f'trial_{trial}' / arm
                directory.mkdir(parents=True, exist_ok=True)
                command = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable,
                           'fastlens/benchmarks/paper/section_6_3/h2d_traffic/swin_block_major.py', '--arm', arm, '--output', str(directory/'result.json')]
                run(command, directory, env, paths[arm])
                result = json.loads((directory/'result.json').read_text())
                assert result['samples'] == 50000
                if arm == 'block':
                    assert sum(s['planless_image_descriptor_count'] for s in result['native_batches']) == 50000
                    assert sum(s['projection_item_count'] for s in result['native_batches']) == 0
        predictions = None
        for trial in range(4):
            for arm in ['block', 'compact', 'cpu']:
                r = json.loads((OUT/gpu/f'trial_{trial}'/arm/'result.json').read_text())
                top1 = [v[0] for v in r['predictions']]
                if predictions is None:
                    predictions = top1
                assert top1 == predictions, (gpu, trial, arm, 'prediction mismatch')
        print(gpu, 'all full-validation top-1 predictions agree', flush=True)


if __name__ == '__main__':
    main()
