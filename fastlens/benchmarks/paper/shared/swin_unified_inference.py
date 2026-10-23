"""Swin full-validation comparison with one timing and execution protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
PATHS = ('pytorch', 'dali', 'coordl', 'ffcv_raw', 'ffcv_jpeg', 'l3', 'cpu_dct', 'imagelanes')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=lambda name: name.replace('fastlens', 'imagelanes'), metavar='PATH', choices=PATHS, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    contract = read(args.contract)
    # Resolve the JPEG ABI before RGB-no-more loads its older JPEG binding.
    import cv2
    cv2.setNumThreads(1)
    if args.path.startswith('ffcv'):
        from ffcv.loader import Loader
    import torch
    from fastlens.benchmarks.baselines.system_rgbnomore.inference import pipeline as P
    from fastlens.benchmarks.system_dct_major import pipeline as M
    from fastlens.benchmarks.system_dct_major.common import parse_manifest
    from fastlens.benchmarks.profiling.input_motivation import cache_pages

    args.output.mkdir(parents=True, exist_ok=True)
    samples = read(contract['dataset']['sample_manifest'])['samples']
    dct = args.path in ('imagelanes', 'cpu_dct')
    if dct:
        samples = sorted(samples, key=lambda s: s['galp_image_id'])
        for i, sample in enumerate(samples):
            assert sample['galp_image_id'] == i
            sample['ordinal'] = i
    assert len(samples) == 50000
    domain = 'dct' if dct else 'rgb'
    checkpoint = contract['models'][domain]
    assert P.sha256_file(Path(checkpoint['checkpoint'])) == checkpoint['checkpoint_sha256']
    if args.path != 'l3':
        contract['pipelines']['dali'].pop('l3', None)
    count = 100 if args.check else 50000
    selected = samples[:count]
    if args.path == 'cpu_dct':
        import csv
        index = args.output/'physical_order.csv'
        with index.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Filepath', 'Label'])
            writer.writerows((s['path'], s['label']) for s in selected)
        contract['dataset']['canonical_index_csv'] = str(index)
    adapter_contract = contract
    if args.path == 'coordl':
        adapter_contract['pipelines']['coordl']['file_list'] = str(args.output/'coordl_file_list.txt')
    adapter_contract['execution'].update(batch_size=50, workers=8, precision='fp32')
    if args.path == 'imagelanes':
        manifest = ROOT/'fastlens/data/compressed/imagenet512_val_block_major/manifest.bin'
        paths = [Path(s['fls_path']) for s in parse_manifest(manifest)['shards']]
    elif args.path.startswith('ffcv'):
        paths = [Path(adapter_contract['pipelines']['ffcv']['beton'])]
    elif args.path == 'l3':
        from fastlens.benchmarks.baselines.l3.prepare import encoded_path
        paths = [encoded_path(contract['pipelines']['dali']['l3']['root'], s['sample_id']) for s in selected]
    else:
        paths = [Path(s['path']) for s in selected]
    cache = cache_pages(paths, warm=True)
    assert cache['resident_fraction'] == 1
    if args.path == 'coordl':
        from fastlens.benchmarks.paper.shared.remaining_baseline_inference import coordl_samples
        reader_samples = coordl_samples(samples, args.output)
    else:
        reader_samples = samples
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    model = getattr(P, '_build_' + domain + '_model')(contract, device)
    shapes = [(3, 256, 256)] if not dct else [(1, 32, 32, 8, 8), (2, 16, 16, 8, 8)]
    with torch.inference_mode():
        dummy = tuple(torch.zeros(50, *shape, device=device) for shape in shapes)
        for _ in range(20):
            model(*dummy)
        del dummy
        torch.cuda.synchronize()
        predictions = []
        loss_sum, correct1, correct5 = 0., 0, 0
        started = time.perf_counter()
        if args.path == 'imagelanes':
            from fastlens.torch import DirectDctReader
            os.environ['FASTLENS_BLOCK_MAJOR_ACCESS_DIR'] = str(manifest.parent/'access')
            os.environ['FASTLENS_PHASE6_NATIVE_PHYSICAL'] = '1'
            reader = DirectDctReader(manifest)
            source = reader.pipeline('rgbnomore-swinv2-block-major-validation-v1')
            # The physical reader needs full-shard coverage even for a short check.
            source.start([list(range(i, i+50)) for i in range(0, len(samples), 50)])
        else:
            if args.path in ('ffcv_raw', 'ffcv_jpeg', 'coordl'):
                cls, name = (M.CoorDLAdapter, 'coordl') if args.path == 'coordl' else (M.FfcvAdapter, 'ffcv')
                # FFCV's background producer must receive ordinary mutable tensors.
                with torch.inference_mode(False):
                    source = cls(adapter_contract, reader_samples, device, name)
                    source.begin_repeat()
            else:
                cls = {'pytorch': P.PyTorchAdapter, 'dali': P.DaliAdapter,
                       'l3': P.DaliAdapter, 'cpu_dct': P.RgbNoMoreAdapter}[args.path]
                source = cls(adapter_contract, selected, device)
                source.begin_repeat()
        # FFCV sets a global thread count while constructing its loader.
        torch.set_num_threads(8)
        for first in range(0, count, 50):
            expected = selected[first:first+50]
            if args.path == 'imagelanes':
                batch = next(source)
                assert list(batch.global_image_ids) == [s['galp_image_id'] for s in expected]
                inputs = (batch.y, batch.cbcr)
                labels = torch.tensor([s['label'] for s in expected], device=device)
            else:
                batch = source.load(expected)
                assert batch.ordinals == [s['ordinal'] for s in expected]
                inputs = tuple(x.to(device, non_blocking=True) for x in batch.inputs)
                labels = batch.labels.to(device, non_blocking=True)
            logits = model(*inputs)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
            top5 = logits.topk(5, dim=1).indices
            matches = top5.eq(labels[:, None])
            correct1 += int(matches[:, :1].sum())
            correct5 += int(matches.sum())
            predictions.extend(top5.cpu().tolist())
            stream.synchronize()
            del batch, inputs, logits, labels, top5, matches
        seconds = time.perf_counter() - started
    if args.path.startswith('ffcv'):
        if args.check:
            source.iterator.terminate_event.set()
        else:
            assert next(source.iterator, None) is None
        source.iterator.join()
        torch.cuda.synchronize()
        source.iterator.close()
    else:
        source.close()
    if args.path == 'coordl':
        from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
        cleanup_coordl_cache(args.output)
    result = dict(path=args.path, images=count, seconds=seconds, images_per_s=count/seconds,
        top1=100*correct1/count, top5=100*correct5/count, ce=loss_sum/count,
        sample_ids=[s['sample_id'] for s in selected], predictions=predictions,
        checkpoint=checkpoint, batch=50, precision='fp32', tf32=False, compile=False,
        host_threads=torch.get_num_threads(), workers=8, model_stream_priority=stream.priority,
        cpu_affinity=sorted(os.sched_getaffinity(0)), CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cache=cache, gpu=torch.cuda.get_device_name(), correctness_check=args.check,
        timing_scope='One wall-clock interval: reader construction/startup through all 50000 logits, CE, top1/top5 and prediction copies; model load and 20-batch model warmup excluded.',
        predictions_sha256=hashlib.sha256(json.dumps(predictions).encode()).hexdigest())
    (args.output/'result.json').write_text(json.dumps(result)+'\n')
    print(json.dumps({k: result[k] for k in ('path', 'images', 'images_per_s', 'top1')}), flush=True)


if __name__ == '__main__':
    main()
