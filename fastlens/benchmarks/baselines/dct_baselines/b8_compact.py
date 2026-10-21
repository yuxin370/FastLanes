"""Compact training inputs adapted from existing JPEG B8 stores, without FLS re-encoding."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import VERSION, artifact, load_artifact, source_diagonal_frequencies, write_json
from .materialize import operation_order, source_diagonal_profile
from .native import fit_statistics
from .runner import Replay
from .store import fingerprint
from .unified import UnifiedReplay


def make_contract(source_path, diagonals, upstream):
    source_ref = artifact(source_path)
    source = load_artifact(source_ref)
    records = source['images']
    if not records or any(record['split'] != 'train' for record in records):
        raise ValueError('B8 compact fitting requires an explicitly frozen training population')
    if any((record['width'], record['height']) != (512, 512) or
           [c['block_grid'] for c in record['components']] != [[64, 64], [32, 32], [32, 32]] for record in records):
        raise ValueError('B8 compact adaptation requires source512 YCbCr 4:2:0')
    frequencies = source_diagonal_frequencies(diagonals)
    side, grid = min(diagonals, 8), 28
    channels = [[component, frequency] for component in range(3) for frequency in frequencies]
    profile = dict(version=VERSION, id=f'b8-compact-grid28-d{diagonals}', representation='jpeg_b8',
                   model='b8_compact', upstream=str(Path(upstream).resolve()), transform='cnn_source512',
                   source_frequencies=[list(range(64))]*3,
                   dependency_derivation='full source before explicit lossy source mask and spectral resize',
                   dependency_evidence=artifact(Path(__file__).with_name('contracts.py')),
                   resize_implementation=artifact(Path(upstream)/'utils/dct_ops.py'),
                   input_geometry='512x512 YCbCr 4:2:0', dtype='float32',
                   output_channels=channels, output_grids=[grid]*3,
                   mean=[0.]*len(channels), std=[1.]*len(channels),
                   default_crop=[32, 32, 448, 448], label_policy='source_label',
                   tensors=[dict(name='projected', shape=[len(channels), grid, grid], layout='CHW')])
    profile = source_diagonal_profile(profile, diagonals)
    profile['operation_order'] = operation_order(profile)
    return dict(version=1, split='train', representation='jpeg_b8_matched_compact_adaptation',
                source_manifest=source_ref, source_identity=source_ref['sha256'],
                source_diagonals=diagonals, output_frequencies=list(frequencies), mosaic_side=side,
                input_shape=[3, grid*side, grid*side], components=['Y', 'Cb', 'Cr'], profile=profile,
                normalization=None,
                semantics='source q mask before dequantize/crop/resize/round/clamp; output mask after resize; compact mosaic; per-plane normalization including structural zeros',
                scope='matched JPEG B8 adaptation, not original IC B32 scalar-Q50; no FLS re-encoding')


class B8CompactDataset:
    """Reuse persistent CPU readers or the native CNN producer; model training is separate."""
    def __init__(self, contract, backend, root=None, *, device='cpu', module_path=None,
                 batch_size=64, pool_size=256, workers=8, decode_budget_mib=1024,
                 reader_memory_mib=2048, normalize=True, content_evidence=None):
        self.manifest = json.loads(Path(contract).read_text()) if isinstance(contract, (str, Path)) else contract
        self.backend = backend
        self.profile = self.manifest['profile']
        self.source = load_artifact(self.manifest['source_manifest'])
        self.samples = self.source['images']
        if any(record['split'] != 'train' for record in self.samples):
            raise ValueError('compact training cannot consume a validation population')
        self.image_count = len(self.samples)
        self.sample_ids = [row['image_id'] for row in self.samples]
        self.labels = [row['label'] for row in self.samples]
        self.input_shape = tuple(self.manifest['input_shape'])
        self.pool_size, self.batch_size = pool_size, batch_size
        self.device = torch.device(device if backend == 'fastlens' else 'cpu')
        self.execution_partition = dict(read_decode='native CUDA pipeline' if backend == 'fastlens' else 'cpu',
                                        normalization=str(self.device))
        self.producer_h2d_unmeasured = backend == 'fastlens'
        config = dict(source_manifest=self.manifest['source_manifest'], workers=workers,
                      model_microbatch_size=batch_size, preparation_pool_size=pool_size,
                      decode_budget_bytes=decode_budget_mib*1024**2, reader_memory_budget_bytes=reader_memory_mib*1024**2)
        config["store_content_evidence"] = content_evidence
        self.producer = UnifiedReplay(backend, root, config, self.source, self.profile,
                                      device=self.device, module_path=module_path)
        self.executor = ThreadPoolExecutor(max_workers=workers) if backend == 'cpu_dct' else None
        if self.producer.reader is not None:
            # UnifiedReplay checked content once at construction. Timed reads retain
            # NPZ/CRC validation but do not repeat a complete member SHA256.
            self.producer.reader.verify_reads = False
        self.cpu = Replay(self.producer.reader, self.profile, config, records=self.producer.records,
                          executor=self.executor) if backend != 'fastlens' else None
        self.side = self.manifest['mosaic_side']
        frequencies = self.manifest['output_frequencies']
        self.u = torch.tensor([f//8 for f in frequencies], device=self.device)
        self.v = torch.tensor([f % 8 for f in frequencies], device=self.device)
        self.normalize = normalize
        if normalize:
            statistics = self.manifest['normalization']
            if statistics is None:
                raise ValueError('fit training-only normalization before training')
            self.mean = torch.tensor(statistics['mean'], dtype=torch.float32, device=self.device)[None, :, None, None]
            self.std = torch.tensor(statistics['std'], dtype=torch.float32, device=self.device)[None, :, None, None]

    def read(self, indices):
        indices = list(indices)
        rows = [dict(occurrence_id=f'read:{slot}:{index}', image_id=self.sample_ids[index],
                     target_label=self.labels[index], microbatch_id=slot//self.batch_size,
                     crop_box=self.profile['default_crop'], flip=False, profile_id=self.profile['id'])
                for slot, index in enumerate(indices)]
        # no_grad is intentional: output must remain a normal tensor that the
        # training model can save for backward (not an inference-mode tensor).
        with torch.no_grad():
            output = torch.zeros((len(indices), *self.input_shape), dtype=torch.float32, device=self.device)
            if self.cpu is not None:
                batches, counters = self.cpu.pool(rows)
                tensors = (batch[0][0] for batch in batches)
                observed = asdict(counters) if self.producer.reader is not None else None
            else:
                tensors = (batch[0] for _, batch in self.producer.inference_batches(rows))
                observed = None  # Native internal I/O/H2D are not inferred from final tensor size.
            offset = 0
            for projected in tensors:
                n = len(projected)
                block_view = output[offset:offset+n].reshape(n, 3, 28, self.side, 28, self.side).permute(0, 1, 2, 4, 3, 5)
                selected = projected.reshape(n, 3, len(self.u), 28, 28).permute(0, 1, 3, 4, 2)
                block_view[..., self.u, self.v] = selected
                offset += n
            if self.normalize:
                output.sub_(self.mean).div_(self.std)
        return output, torch.tensor([self.labels[index] for index in indices], dtype=torch.int64), observed

    def close(self):
        self.producer.close()
        if self.executor is not None:
            self.executor.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def fit(source_manifest, output, *, diagonals, upstream, pool_size=256, workers=8):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    contract = make_contract(source_manifest, diagonals, upstream)
    with B8CompactDataset(contract, 'cpu_dct', pool_size=pool_size, workers=workers, normalize=False) as dataset:
        def mosaics():
            for start in range(0, dataset.image_count, pool_size):
                values, _, _ = dataset.read(range(start, min(start+pool_size, dataset.image_count)))
                yield from values.numpy()
        contract['normalization'] = fit_statistics(mosaics(), split='train')
    contract['data_identity'] = fingerprint(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('fit', help='fit small-input normalization from frozen training JPEG B8 only')
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-diagonals', type=int, choices=range(1, 16), required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--pool-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('train', help='real classifier updates; no pretrained checkpoint or accuracy claim')
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--input-evidence', type=Path, help='reuse a passed compact input check and unchanged audited store files')
    p.add_argument('--backend', choices=('cpu_dct', 'ic', 'fs', 'fastlens'), required=True)
    p.add_argument('--root', type=Path)
    p.add_argument('--module-path', type=Path)
    from .compact_models import MODEL_NAMES
    p.add_argument('--model', choices=MODEL_NAMES, default='resnet50')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--pool-size', type=int, default=256)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--decode-budget-mib', type=int, default=1024)
    p.add_argument('--reader-memory-mib', type=int, default=2048)
    p.add_argument('--warmup-seconds', type=float, default=60.)
    duration = p.add_mutually_exclusive_group()
    duration.add_argument('--min-seconds', type=float, default=180.)
    duration.add_argument('--epochs', type=int)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        result = fit(args.source_manifest, args.output, diagonals=args.source_diagonals, upstream=args.upstream,
                     pool_size=args.pool_size, workers=args.workers)
        print(json.dumps(dict(profile=str(args.output), input_shape=result['input_shape'], representation=result['representation'])))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.backend != 'cpu_dct' and args.root is None:
        parser.error('this backend requires --root')
    input_evidence = entry = None
    if args.input_evidence is not None:
        input_evidence = json.loads(args.input_evidence.read_text())
        if (input_evidence['status'] != 'passed' or input_evidence['profile'] != artifact(args.profile)
                or input_evidence['device'] != args.device):
            raise ValueError('input evidence differs from the compact profile or device')
        for candidate in input_evidence['backends'].values():
            identity = candidate['backend']
            if identity['kind'] != args.backend:
                continue
            if args.backend in ('ic', 'fs') and identity['store']['manifest']['path'] != str((args.root/'store.json').resolve()):
                continue
            if args.backend == 'fastlens' and identity['manifest']['path'] != str(args.root.resolve()):
                continue
            entry = candidate
            break
        if entry is None or entry['status'] != 'passed':
            raise ValueError('input evidence does not cover this backend/store')
        if any(artifact(ref['path']) != ref for ref in entry['implementation']):
            raise ValueError('compact input implementation changed after verification')
    from .compact_training import benchmark
    torch.set_num_threads(args.torch_threads)
    with B8CompactDataset(args.profile, args.backend, args.root, device=args.device, module_path=args.module_path,
                          batch_size=args.batch_size, pool_size=args.pool_size, workers=args.workers,
                          decode_budget_mib=args.decode_budget_mib, reader_memory_mib=args.reader_memory_mib,
                          content_evidence=entry["content_verification"] if entry is not None else None) as dataset:
        if entry is not None and dataset.producer.identity != entry["backend"]:
            raise ValueError("input evidence backend identity changed")
        result = benchmark(dataset, device=args.device, batch_size=args.batch_size, pool_size=args.pool_size,
                           accumulation=args.accumulation, warmup_seconds=args.warmup_seconds, min_seconds=args.min_seconds,
                           epochs=args.epochs, repeats=args.repeats, seed=args.seed, learning_rate=args.learning_rate,
                           model_name=args.model)
        result['backend'] = dataset.producer.identity
        result['content_verification'] = dataset.producer.content_verification
        result['input_evidence'] = artifact(args.input_evidence) if args.input_evidence is not None else None
        result['training_implementation'] = artifact(Path(__file__).with_name('compact_training.py'))
    result.update(profile=artifact(args.profile), workers=args.workers,
                  reader_budgets=dict(decode_bytes=args.decode_budget_mib*1024**2, memory_bytes=args.reader_memory_mib*1024**2),
                  experiment_type='jpeg_b8_matched_compact_training_throughput')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(status='passed', output=str(args.output), median_images_s=result['median_images_s'],
                         claim='random-initialized training throughput, no validation accuracy'), indent=2))


if __name__ == '__main__':
    main()
