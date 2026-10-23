"""Paired Swin input layouts with fixed model batches and physical sample order."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'build/fastlens/torch')]

BLOCK_PROFILE = 'rgbnomore-swinv2-block-major-validation-v1'
COMPACT_PROFILE = 'rgbnomore-swinv2-validation-v1'
DATA = ROOT / 'fastlens/data/compressed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--arm', choices=['block', 'compact', 'cpu'], default='block')
    parser.add_argument('--count', type=int, default=50000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    import torch
    from fastlens.torch import DirectDctReader
    from fastlens.diagnostics.direct_dct import execution_stats_snapshot, initialization_stats
    from fastlens.benchmarks.baselines.system_rgbnomore.inference import pipeline as P
    from fastlens.benchmarks.baselines.system_rgbnomore.shared.common import sha256_file

    torch.set_num_threads(8)
    os.environ['FASTLENS_BLOCK_MAJOR_ACCESS_DIR'] = str(DATA / 'imagenet512_val_block_major/access')
    os.environ['FASTLENS_PHASE6_NATIVE_PHYSICAL'] = '1' if args.arm == 'block' else '0'
    contract = json.loads(args.contract.read_text())
    samples = sorted(json.loads(Path(contract['dataset']['sample_manifest']).read_text())['samples'],
                     key=lambda x: x['galp_image_id'])[:args.count]
    for i, s in enumerate(samples):
        assert s['galp_image_id'] == i
        s['ordinal'] = i
    stored = json.loads((DATA / 'imagenet512_val_block_major/samples.json').read_text())
    with (DATA / 'imagenet512_val_compact_v3/index.csv').open() as f:
        compact_samples = list(csv.DictReader(f))
    assert all(s['sample_id'] == stored[i]['logical_sample_id'] == compact_samples[i]['Filepath']
               for i, s in enumerate(samples))
    checkpoint = contract['models']['dct']
    assert sha256_file(Path(checkpoint['checkpoint'])) == checkpoint['checkpoint_sha256']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.arm == 'cpu':
        index = args.output.parent / 'physical_order.csv'
        with index.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filepath', 'Label'])
            writer.writerows((s['path'], s['label']) for s in samples)
        contract['dataset']['canonical_index_csv'] = str(index)
    device = torch.device('cuda:0')

    def reader(arm):
        directory = 'imagenet512_val_block_major' if arm == 'block' else 'imagenet512_val_compact_v3'
        return DirectDctReader(DATA / directory / 'manifest.bin')

    if args.check:
        a, b = reader('block'), reader('compact')
        records = []
        for first in (0, 1000, 16370, 32760, 48980, 49980):
            ids = list(range(first, first + 20))
            x, y = a.read(ids, BLOCK_PROFILE), b.read(ids, COMPACT_PROFILE)
            differences = []
            for u, v in zip((x.y, x.cbcr), (y.y, y.cbcr)):
                torch.testing.assert_close(u, v, atol=0, rtol=0)
                differences.append(float((u-v).abs().max()))
            records.append(dict(ids=ids, max_absolute_difference=differences))
            del x, y
        args.output.write_text(json.dumps(dict(exact=True, images=120, records=records), indent=2)+'\n')
        print('Exact Swin input equality: 120 images, including segment boundaries.', flush=True)
        return

    model = P._build_dct_model(contract, device)
    with torch.inference_mode():
        dummy = (torch.zeros((50, 1, 32, 32, 8, 8), device=device),
                 torch.zeros((50, 2, 16, 16, 8, 8), device=device))
        for _ in range(20):
            model(*dummy)
        del dummy
        torch.cuda.synchronize()
        model_stream = torch.cuda.Stream(priority=-1)
        start = time.perf_counter()
        native = args.arm != 'cpu'
        if native:
            source = reader(args.arm)
            profile = BLOCK_PROFILE if args.arm == 'block' else COMPACT_PROFILE
            pipeline = source.pipeline(profile)
            pipeline.start([list(range(i, min(i+50, args.count))) for i in range(0, args.count, 50)])
        else:
            adapter = P.RgbNoMoreAdapter(contract, samples, device)
            adapter.begin_repeat()
        predictions, stats = [], []
        with torch.cuda.stream(model_stream):
            for i in range(0, args.count, 50):
                if native:
                    batch = next(pipeline)
                    assert list(batch.global_image_ids) == list(range(i, min(i+50, args.count)))
                    inputs = (batch.y, batch.cbcr)
                else:
                    batch = adapter.load(samples[i:i+50])
                    assert batch.ordinals == list(range(i, min(i+50, args.count)))
                    assert batch.labels.tolist() == [s['label'] for s in samples[i:i+50]]
                    inputs = tuple(t.to(device, non_blocking=True) for t in batch.inputs)
                logits = model(*inputs)
                predictions.append(logits.topk(5, dim=1).indices)
                if native:
                    snapshot = execution_stats_snapshot(batch)
                    stats.append({k: snapshot[k] for k in (
                        'uses_planless_fixed_transform', 'planless_image_descriptor_count',
                        'projection_item_count', 'jpeg_dct_projection_items_materialized',
                        'requested_source_block_count', 'selected_vector_count', 'full_vector_count',
                        'selected_compressed_payload_bytes', 'rowgroup_storage_bytes_read',
                        'workset_upload_dma_bytes', 'storage_read_granularity', 'decode_granularity',
                        'duplicate_decode_source_count', 'planning_ms', 'runtime_policy_decision',
                        'runtime_policy_reason') if k in snapshot})
                del logits, inputs, batch
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        predicted = torch.cat(predictions).cpu()
        labels = torch.tensor([s['label'] for s in samples])
        result = dict(arm=args.arm, samples=args.count, model_batch=50, elapsed_s=elapsed,
                      images_per_s=args.count/elapsed, gpu=torch.cuda.get_device_name(),
                      top1=100*float((predicted[:, 0] == labels).float().mean()),
                      top5=100*float((predicted == labels[:, None]).any(1).float().mean()),
                      sample_order='ascending physical image ID, same for all arms',
                      image_ids=[s['galp_image_id'] for s in samples], predictions=predicted.tolist(),
                      checkpoint=checkpoint, native_batches=stats, command=sys.argv,
                      timing_scope='reader/adapter startup, input preparation and model; excludes model load/warmup and final accuracy reduction')
        if native:
            result.update(profile=source.profile_info(profile), initialization=initialization_stats(source),
                          pipeline_metrics=dict(pipeline._native.metrics))
            pipeline.close()
            del pipeline, source
        else:
            adapter.close()
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: result[k] for k in ('arm','samples','elapsed_s','images_per_s','top1')}), flush=True)


if __name__ == '__main__':
    main()
