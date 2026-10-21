"""CUDA regression for reordered projected ranges and split-source scratch.

Uses the existing source512 manifest; no JPEG/store generation or model training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from fastlens.profiles import DctModelProfile
from fastlens.torch.direct_dct import DirectDctBatch, DirectDctReader, _cnn_pipeline_options


def shutdown_child(manifest, module_path):
    """Retire a final batch while a consumer is outstanding, then exit without polling."""
    torch.cuda.init()
    reader = DirectDctReader(manifest, module_path=module_path)
    native = reader._module
    native.manual_reclaim()  # Streaming next() can poll before the first device allocation.
    profile = DctModelProfile(2, ((0, 0), (0, 1), (1, 0), (2, 8)),
                             ((1., 3.), (-2., 5.), (3., 7.), (-4., 11.)))
    pipeline = reader._native.pipeline_batch_options(**_cnn_pipeline_options(profile), output_batch_images=1)
    pipeline.reset([[0, 1]])
    batch = DirectDctBatch(next(pipeline), 'cnn-source512-v1')
    value = batch.projected
    expected = value.clone()
    torch.cuda.synchronize()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        batch.record_stream(consumer)
        torch.cuda._sleep(1000000000)
        copied = value.clone()
    pipeline.close()
    del pipeline, value, batch
    stats = native._lifetime_reclaim_stats_for_test()
    if sum(stats[k]['pending_reclaim_count'] for k in ('native', 'legacy')) == 0:
        raise AssertionError('shutdown regression did not leave a deferred batch')
    consumer.synchronize()
    torch.testing.assert_close(copied, expected, atol=0, rtol=0)
    del reader


def check_shutdown(manifest, module_path):
    for lifetime in ('1', '0'):
        env = dict(os.environ, FASTLENS_PHASE4_NATIVE_LIFETIME=lifetime,
                   FASTLENS_BLOCK_MAJOR_ACCESS_DIR=str(manifest.parent/'access'))
        child = subprocess.run([sys.executable, '-B', '-c',
            'from pathlib import Path; import sys; '
            'from fastlens.torch.streamed_projection_test import shutdown_child; '
            'shutdown_child(Path(sys.argv[1]), Path(sys.argv[2]))', str(manifest), str(module_path)],
            env=env, capture_output=True, text=True)
        if child.returncode or child.stderr:
            raise AssertionError(f'shutdown regression failed ({lifetime=}):\n{child.stdout}{child.stderr}')


def run(manifest, module_path):
    os.environ['FASTLENS_BLOCK_MAJOR_ACCESS_DIR'] = str(manifest.parent/'access')
    reader = DirectDctReader(manifest, module_path=module_path)
    if reader.image_count < 1027:
        raise ValueError('use the source512 fixture with at least two physical shards')
    profile = DctModelProfile(2, ((0, 0), (0, 1), (1, 0), (2, 8)),
                              ((1., 3.), (-2., 5.), (3., 7.), (-4., 11.)))
    options = _cnn_pipeline_options(profile)
    read_options = dict(options)
    read_options.pop('async_planless_completion')  # A pipeline-only option; direct reads are the oracle.
    ids = [1026, 2, 1024, 2, 0]  # Reversed physical order, duplicate source and output tail.
    consumer = torch.cuda.Stream()
    observed = []
    with torch.inference_mode():
        for mixed in (False, True):
            # Y block rows 7 and 8 belong to different physical rowgroups in
            # source512. Starting at pixel 56 makes one down2 output need both.
            # Pixel 48 paired rows 6/7 and 8/9 within their own rowgroups.
            transforms = [dict(crop=dict(x=32+16*(i % 2 if mixed else 0), y=56, width=32, height=32),
                               horizontal_flip=bool(mixed and i % 2)) for i in range(len(ids))]
            reference_batch = reader._native.read_batch(ids, transforms=transforms, **read_options)
            expected = reference_batch.projected.clone()
            torch.cuda.synchronize()
            del reference_batch
            # One physical rowgroup per workset splits the boundary pair above.
            # The tested path always keeps range events.
            streamed = dict(options, decode_batch_rowgroups=1, transform_blocks_per_launch=1)
            pipeline = reader._native.pipeline_batch_options(**streamed, output_batch_images=2)
            pipeline.reset([ids, list(reversed(ids))], transforms_by_batch=[transforms, list(reversed(transforms))])
            copies, stats = [], []
            try:
                for expected_ids in (ids, list(reversed(ids))):
                    batch = DirectDctBatch(next(pipeline), 'cnn-source512-v1')
                    if batch.global_image_ids != expected_ids:
                        raise AssertionError('streamed projection changed occurrence order')
                    pieces = []
                    with torch.cuda.stream(consumer):
                        batch.record_stream(consumer)
                        for first in range(0, len(ids), 2):
                            value = batch.projected_range(first, min(2, len(ids)-first))
                            # Read the range as soon as its event allows, before
                            # waiting on the whole preparation batch's metrics.
                            pieces.append(value.clone())
                            del value
                    copies.append(pieces)
                    stats.append(dict(batch._native.execution_stats))
                    del batch
            finally:
                pipeline.close()
            # Reuse the reader immediately; second-stream tensor owners may
            # still be in flight. The earlier clones must remain unchanged.
            reused = reader._native.read_batch(ids, transforms=transforms,
                **dict(read_options, decode_batch_rowgroups=1, transform_blocks_per_launch=1))
            repeated = reused.projected.clone()
            consumer.synchronize()
            torch.testing.assert_close(torch.cat(copies[0]), expected, atol=0, rtol=0)
            torch.testing.assert_close(torch.cat(copies[1]), expected.flip(0), atol=0, rtol=0)
            torch.testing.assert_close(repeated, expected, atol=0, rtol=0)
            for item in stats:
                used = item['actual_transient_split_source_used_peak_bytes']
                capacity = item['actual_transient_split_source_capacity_peak_bytes']
                metadata = item['actual_transient_split_metadata_capacity_peak_bytes']
                worksets = item['planless_transform_active_output_workset_count']
                if not 0 < used <= capacity or worksets < 4:
                    raise AssertionError('regression did not exercise cross-workset split sources: '
                                         f'{mixed=}, {used=}, {capacity=}, {worksets=}')
                total = item['actual_transient_total_allocated_high_water_bytes']
                if total < capacity+metadata or item['actual_transient_memory_gate_passed'] != (total <= 512*1024**2):
                    raise AssertionError('split-source allocation is missing from the public transient total/gate')
            observed.append(dict(mixed_crop_and_flip=mixed, execution_stats=stats))
            del reused, repeated, copies, expected
    return dict(status='passed', gpu=torch.cuda.get_device_name(), cases=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--module-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check_shutdown(args.manifest, args.module_path)
    result = run(args.manifest, args.module_path)
    result['deferred_shutdown_release'] = 'passed (native and legacy)'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Passed: streamed projected ranges, split sources and reader reuse. Evidence: {args.output}')


if __name__ == '__main__':
    main()
