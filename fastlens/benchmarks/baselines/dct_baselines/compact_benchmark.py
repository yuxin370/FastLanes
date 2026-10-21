"""Reproduce the five JPEG-B8 compact-input training paths in FastLens Figure 10."""
import argparse
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path

import torch

from .b8_compact import B8CompactDataset
from .compact_models import MODEL_NAMES
from .compact_training import batch_plan, benchmark
from .contracts import artifact, write_json
from .prefetched_compact import PrefetchedCompactDataset
from .prepared_compact import PreparedCompactDataset
from .streamed_b8_compact import StreamedB8CompactDataset

METHODS = ('cpu_dct', 'ic_prepared_shared', 'ic_prepared_batched', 'fs_prefetched', 'fastlens')


@contextmanager
def reader(args):
    with ExitStack() as stack:
        if args.method == 'fastlens':
            dataset = stack.enter_context(StreamedB8CompactDataset(
                args.profile, args.root, device=args.device, module_path=args.module_path,
                batch_size=args.batch_size, pool_size=args.pool_size, workers=args.workers))
        elif args.method == 'cpu_dct':
            dataset = stack.enter_context(B8CompactDataset(
                args.profile, 'cpu_dct', batch_size=args.batch_size,
                pool_size=args.pool_size, workers=args.workers))
        else:
            constructor = (partial(B8CompactDataset, args.profile, 'fs', args.root,
                                   batch_size=args.batch_size, pool_size=args.pool_size, workers=1)
                           if args.method == 'fs_prefetched' else
                           partial(PreparedCompactDataset, args.root / 'prepared.json', workers=1))
            parent = stack.enter_context(constructor())
            if args.method.startswith('ic_prepared'):
                if parent.manifest['profile'] != artifact(args.profile) or parent.manifest['method'] != args.method:
                    raise ValueError('prepared store differs from the requested compact profile or method')
                content = parent.content_verification
            else:
                content = parent.producer.content_verification
            dataset = stack.enter_context(PrefetchedCompactDataset(
                parent, partial(constructor, content_evidence=content),
                batch_size=args.batch_size, pool_size=args.pool_size,
                workers=args.workers, pin_memory=False))
        yield dataset


def check_inputs(dataset, reference, *, batch_size, pool_size):
    """Check the actual timed reader, including streamed native ranges and tails."""
    plan = batch_plan(dataset.image_count, batch_size, pool_size, 1)
    if dataset.image_count != reference.image_count:
        raise ValueError('compact readers have different populations')
    stream = dataset.iter_training_batches(plan) if hasattr(dataset, 'iter_training_batches') else None
    checked = 0
    try:
        for start, stop, windows in plan:
            expected, labels, _ = reference.read(range(start, stop))
            if stream is None:
                actual, targets, _ = dataset.read(range(start, stop))
                torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(targets.cpu(), labels, atol=0, rtol=0)
            else:
                for window in windows:
                    for row in window:
                        actual, targets, _ = next(stream)
                        left, right = row['start'], row['stop']
                        torch.testing.assert_close(actual.cpu(), expected[left:right], atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(targets.cpu(), labels[left:right], atol=0, rtol=0)
            checked += stop - start
        if stream is not None:
            try:
                next(stream)
            except StopIteration:
                pass
            else:
                raise AssertionError('compact reader yielded extra batches')
    finally:
        if stream is not None:
            stream.close()
    return dict(samples=checked, inputs_match=True, labels_match=True, atol=1e-5, rtol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True, help='b8_compact fit output (D3 or D15)')
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--root', type=Path, help='prepared IC directory, FS store, or native manifest.bin')
    parser.add_argument('--module-path', type=Path)
    parser.add_argument('--model', choices=MODEL_NAMES, default='resnet50')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--pool-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--torch-threads', type=int, default=2)
    parser.add_argument('--warmup-seconds', type=float, default=60.)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--min-seconds', type=float, default=180.)
    duration.add_argument('--epochs', type=int, help='complete passes over the supplied population')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.method != 'cpu_dct' and args.root is None:
        parser.error('--root is required for this method')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.torch_threads)
    with reader(args) as dataset:
        with B8CompactDataset(args.profile, 'cpu_dct', workers=args.workers,
                              batch_size=args.batch_size, pool_size=args.pool_size) as reference:
            checked = check_inputs(dataset, reference, batch_size=args.batch_size, pool_size=args.pool_size)
        result = {} if args.check_only else benchmark(
            dataset, model_name=args.model, device=args.device, batch_size=args.batch_size,
            pool_size=args.pool_size, accumulation=1, warmup_seconds=args.warmup_seconds,
            min_seconds=args.min_seconds, epochs=args.epochs, repeats=args.repeats, seed=args.seed)
    result.update(method=args.method, profile=artifact(args.profile), input_check=checked,
                  root=str(args.root.resolve()) if args.root is not None else None,
                  module_path=str(args.module_path.resolve()) if args.module_path is not None else None,
                  workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f'Checked {checked["samples"]} inputs; result: {args.output}')


if __name__ == '__main__':
    main()
