"""Full-validation baselines with a common reader-to-metrics wall-clock interval."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def coordl_samples(samples, output):
    """Use the existing training cache-ownership convention for interrupted inference."""
    source_root = Path(os.path.commonpath([str(Path(s['path']).resolve().parent) for s in samples]))
    namespace = Path(tempfile.mkdtemp(prefix='fastlens-coordl-', dir=Path.home()/'tmp'))
    (namespace/'source').symlink_to(source_root, target_is_directory=True)
    (output/'coordl_cache_namespace.json').write_text(json.dumps(dict(
        namespace=str(namespace), source_root=str(source_root), owner_pid=os.getpid()))+'\n')
    return [dict(s, path=str(namespace/'source'/Path(s['path']).resolve().relative_to(source_root))) for s in samples]


def reference_predictions(path):
    import numpy as np
    data = read(path)
    if 'semantic_artifact' in data:
        contract = read(data['contract'])
        samples = read(contract['dataset']['sample_manifest'])['samples']
        with np.load(data['semantic_artifact']) as artifact:
            predictions = artifact['top1_predictions'].reshape(-1).tolist()
        ids = [s['sample_id'] for s in samples]
    else:
        predictions = data['predictions']
        if isinstance(predictions[0], list):
            predictions = [p[0] for p in predictions]
        ids = data['sample_ids']
    assert len(ids) == len(predictions) == len(set(ids)) == 50000
    return dict(zip(ids, predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['vitti', 'mobilenet24', 'mobilenet32', 'efun', 'resnet24', 'resnet64'], required=True)
    parser.add_argument('--path', type=lambda name: name.replace('fastlens', 'imagelanes'), metavar='PATH', choices=['pytorch', 'dali', 'coordl', 'ffcv_raw', 'ffcv_jpeg', 'l3', 'cpu_dct', 'imagelanes'], required=True)
    reference_mode = parser.add_mutually_exclusive_group(required=True)
    reference_mode.add_argument('--reference', type=Path)
    reference_mode.add_argument('--prepare-reference', action='store_true',
                                help='Generate a full-population reference before measured chapter trials')
    parser.add_argument('--contract', type=Path, help='Pipeline contract when not embedded in the reference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.prepare_reference and args.check:
        parser.error('--prepare-reference requires the full 50000-image population')
    previous = {} if args.prepare_reference else read(args.reference)
    contract_path = args.contract or previous.get('contract')
    contract = read(contract_path) if contract_path is not None else None
    if args.contract is not None and 'contract' in previous:
        original = read(read(previous['contract'])['dataset']['sample_manifest'])['samples']
        configured = read(contract['dataset']['sample_manifest'])['samples']
        assert [(s['sample_id'], s['label']) for s in original] == [
            (s['sample_id'], s['label']) for s in configured]
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'result.json').exists()
    os.environ['DCTNET_PROFILE'] = args.model
    import cv2
    cv2.setNumThreads(1)
    if args.path.startswith('ffcv'):
        from ffcv.loader import Loader
    import torch
    from fastlens.benchmarks.system_dct_major import pipeline as M
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    from fastlens.benchmarks.system_dct_major.common import parse_manifest
    from fastlens.benchmarks.execution.revision_results import file_sha
    dct = args.path in ('cpu_dct', 'imagelanes')
    cnn = args.model != 'vitti'
    batch_size = 64
    count = 128 if args.check else 50000
    device = torch.device('cuda:0')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    if cnn:
        if args.model == 'efun':
            import fastlens.benchmarks.dct_models as package
            from fastlens.benchmarks.dct_models import efun_backend as B
            sys.modules['fastlens.benchmarks.dct_models.backend'] = B
            package.backend = B
        from fastlens.benchmarks.dct_models import backend as B, rgb
        from fastlens.benchmarks.dct_models.evaluate import samples as cnn_samples, Inputs
        entries = cnn_samples(50000, physical_order=True)
        samples = [dict(sample_id=e['logical_sample_id'], path=e['path'], label=e['model_label'],
                        ordinal=i, galp_image_id=e['galp_image_id']) for i, e in enumerate(entries)]
        model = (B.model() if dct else rgb.model()).to(device)
        checkpoint_path = B.CHECKPOINT if dct else rgb.CHECKPOINT
        if not args.prepare_reference:
            previous_checkpoint = previous['checkpoint'] if 'checkpoint' in previous else previous['model']['checkpoint']
            assert Path(previous_checkpoint).resolve() == Path(checkpoint_path).resolve()
        shapes = [(sum(map(len, B.INDICES)), B.GRID, B.GRID)] if dct else [(3, 224, 224)]
    else:
        assert contract is not None
        samples = read(contract['dataset']['sample_manifest'])['samples']
        if dct:
            samples.sort(key=lambda s: s['galp_image_id'])
            for i, sample in enumerate(samples):
                sample['ordinal'] = i
        domain = 'dct' if dct else 'rgb'
        checkpoint_path = Path(contract['models'][domain]['checkpoint'])
        assert file_sha(checkpoint_path) == contract['models'][domain]['checkpoint_sha256']
        model = M.build_workload_model(rgbnomore_root=Path(contract['models']['rgbnomore_root']),
            checkpoint=checkpoint_path, device=device, domain=domain,
            workload=contract['workload']['kind'], feature_stage=contract['workload'].get('feature_stage', 'penultimate'))
        shapes = [(1, 28, 28, 8, 8), (2, 14, 14, 8, 8)] if dct else [(3, *contract['preprocess']['rgb']['crop_size'])]
    model.eval()
    selected = samples[:count]
    expected_predictions = None if args.prepare_reference else reference_predictions(args.reference)
    if expected_predictions is not None:
        assert set(s['sample_id'] for s in samples) == set(expected_predictions)
    if contract is not None:
        if cnn:
            other = read(contract['dataset']['sample_manifest'])['samples']
            assert [(s['sample_id'], s['label']) for s in samples] == [(s['sample_id'], s['label']) for s in other]
        contract['execution'].update(batch_size=batch_size, workers=8, precision='fp32', warmup_batches=0)
        if args.path == 'coordl':
            contract['pipelines']['coordl']['file_list'] = str(args.output/'coordl_file_list.txt')
        if args.path == 'cpu_dct':
            index = args.output/'physical_order.csv'
            with index.open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(['Filepath', 'Label'])
                writer.writerows((s['path'], s['label']) for s in selected)
            contract['dataset']['canonical_index_csv'] = str(index)
    manifest = ROOT/'fastlens/data/compressed/imagenet512_val_block_major/manifest.bin'
    if args.path == 'imagelanes':
        paths = [Path(s['fls_path']) for s in parse_manifest(manifest)['shards']]
    elif args.path.startswith('ffcv'):
        paths = [Path(contract['pipelines']['ffcv']['beton'])]
    elif args.path == 'l3':
        from fastlens.benchmarks.baselines.l3.prepare import encoded_path
        l3 = contract['pipelines']['dali']['l3']
        paths = [encoded_path(l3['root'], s['sample_id']) for s in selected]
    else:
        paths = [Path(s['path']) for s in selected]
    cache = cache_pages(paths, warm=True)
    assert cache['resident_fraction'] == 1
    reader_samples = coordl_samples(samples, args.output) if args.path == 'coordl' else samples

    # Reader creation is deferred until the common timed interval below.
    def batches():
        nonlocal source
        if cnn and args.path == 'imagelanes':
            import _fastlens_direct_dct as native
            from fastlens.benchmarks.dct_models.evaluate_shards import native_options, apply_b6_runtime
            from fastlens.benchmarks.dct_models.online_crop import source_options, CROP
            os.environ['FASTLENS_BLOCK_MAJOR_ACCESS_DIR'] = str(manifest.parent/'access')
            reader = native.DirectDctReader(str(manifest))
            options = apply_b6_runtime(source_options(native_options(False, True), 'vector-range-read-selected-decode'))
            source = reader.pipeline_batch_options(**options, output_batch_images=batch_size)
            # The production physical reader requires complete shard coverage,
            # including in a check that consumes only a short prefix.
            ids = [s['galp_image_id'] for s in samples]
            groups = [ids[i:i+1024] for i in range(0, len(ids), 1024)]
            source.reset(groups, transforms_by_batch=[[dict(crop=CROP, horizontal_flip=False) for _ in group] for group in groups])
            offset = 0
            for group in groups:
                batch = next(source)
                assert list(batch.global_image_ids) == group
                for first in range(0, min(len(group), count-offset), batch_size):
                    size = min(batch_size, len(group)-first, count-offset)
                    x = batch.projected_range(first, size)
                    yield (x,), torch.tensor([s['label'] for s in selected[offset:offset+size]], device=device), [s['sample_id'] for s in selected[offset:offset+size]]
                    offset += size
                del batch
                if offset == count:
                    break
        elif cnn and args.path in ('pytorch', 'cpu_dct', 'dali', 'l3'):
            chosen = entries[:count]
            if args.path in ('pytorch', 'cpu_dct'):
                dataset = Inputs(chosen, 'J') if dct else rgb.Inputs(chosen)
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8,
                    shuffle=False, pin_memory=True, prefetch_factor=2, worker_init_fn=rgb.worker_init)
                source = iter(loader)
                for first in range(0, count, batch_size):
                    values = next(source)
                    yield (values[0],), values[1], [s['sample_id'] for s in selected[first:first+len(values[1])]]
            else:
                if args.path == 'l3':
                    chosen = [dict(e, path=str(path)) for e, path in zip(chosen, paths)]
                source = rgb.dali_loader(chosen, batch_size, 8, l3_library=Path(l3['library']) if args.path == 'l3' else None)
                for first in range(0, count, batch_size):
                    values = next(source)[0]
                    ids = values['ordinal'].reshape(-1).cpu().tolist()
                    assert ids == list(range(first, first+len(ids)))
                    yield (values['image'],), torch.tensor([selected[i]['label'] for i in ids], device=device), [selected[i]['sample_id'] for i in ids]
        else:
            name = {'ffcv_raw': 'ffcv', 'ffcv_jpeg': 'ffcv', 'cpu_dct': 'rgbnomore', 'imagelanes': 'dct_major_pushdown', 'l3': 'dali'}.get(args.path, args.path)
            cls = {'pytorch': M.PyTorchAdapter, 'dali': M.DaliAdapter, 'coordl': M.CoorDLAdapter,
                   'ffcv': M.FfcvAdapter, 'rgbnomore': M.RgbNoMoreAdapter, 'dct_major_pushdown': M.FastLensAdapter}[name]
            if args.path == 'imagelanes':
                contract['pipelines'][name]['torch_binding_dir'] = str(ROOT / 'build/fastlens/torch')
            # FFCV's producer thread writes these buffers outside InferenceMode.
            with torch.inference_mode(False):
                source = cls(contract, reader_samples if name in ('ffcv', 'coordl', 'dct_major_pushdown') else selected, device, name)
                source.begin_repeat()
            torch.set_num_threads(8)
            for first in range(0, count, batch_size):
                expected = selected[first:first+batch_size]
                batch = source.load(expected)
                assert batch.ordinals == [s['ordinal'] for s in expected]
                inputs = batch.inputs
                if cnn:
                    pixels = inputs[0].add(1).mul(.5)
                    mean, std = pixels.new_tensor(rgb.MEAN)[None, :, None, None], pixels.new_tensor(rgb.STD)[None, :, None, None]
                    inputs = ((pixels-mean)/std,)
                yield inputs, batch.labels, [s['sample_id'] for s in expected]

    source = None
    with torch.inference_mode():
        dummy = tuple(torch.zeros(batch_size, *shape, device=device) for shape in shapes)
        for _ in range(20):
            model(*dummy)
        del dummy
        torch.cuda.synchronize()
        predictions, sample_ids = [], []
        ce, correct1, correct5 = 0., 0, 0
        started = time.perf_counter()
        for inputs, labels, ids in batches():
            inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            labels = labels.to(device, non_blocking=True)
            logits = model(*inputs)
            ce += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
            top5 = logits.topk(5, dim=1).indices
            matches = top5.eq(labels[:, None])
            correct1 += int(matches[:, :1].sum())
            correct5 += int(matches.sum())
            predictions.extend(top5.cpu().tolist())
            sample_ids.extend(ids)
            stream.synchronize()
            del inputs, logits, labels, top5, matches
        seconds = time.perf_counter()-started
    if args.path.startswith('ffcv'):
        source.iterator.terminate_event.set()
        source.iterator.join()
        torch.cuda.synchronize()
        source.iterator.close()
    elif not (cnn and args.path in ('pytorch', 'cpu_dct', 'dali', 'l3')):
        if args.path == 'imagelanes' and not cnn:
            source.end_repeat()
        source.close()
    if args.path == 'coordl':
        from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
        cleanup_coordl_cache(args.output)
    assert sample_ids == [s['sample_id'] for s in selected]
    mismatches = [dict(sample_id=s, actual=p[0], expected=expected_predictions[s])
                  for s, p in zip(sample_ids, predictions) if expected_predictions is not None and p[0] != expected_predictions[s]]
    if mismatches:
        (args.output/'prediction_mismatch.json').write_text(json.dumps(dict(
            gpu=torch.cuda.get_device_name(), reference=str(args.reference) if args.reference is not None else None, mismatches=mismatches), indent=2)+'\n')
        raise AssertionError(f'{len(mismatches)}/{count} predictions differ from accepted baseline')
    result = dict(model=args.model, path=args.path, images=count, seconds=seconds, images_per_s=count/seconds,
        top1=100*correct1/count, top5=100*correct5/count, ce=ce/count, predictions=predictions, sample_ids=sample_ids,
        predictions_sha256=hashlib.sha256(json.dumps(predictions).encode()).hexdigest(),
        checkpoint=str(checkpoint_path), checkpoint_sha256=file_sha(checkpoint_path), correctness_passed=not args.prepare_reference,
        mode='reference' if args.prepare_reference else 'correctness' if args.check else 'measurement',
        reference=str(args.reference) if args.reference is not None else None, cache=cache, batch=batch_size, precision='fp32', tf32=False, compile=False,
        host_threads=torch.get_num_threads(), workers=8, model_stream_priority=stream.priority,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), gpu=torch.cuda.get_device_name(),
        timing_scope='Reader construction/startup through logits, CE, top1/top5 and prediction copies; model load and 20-batch model warmup excluded.')
    (args.output/'result.json').write_text(json.dumps(result)+'\n')
    print(json.dumps({k: result[k] for k in ('model', 'path', 'images_per_s', 'top1')}), flush=True)


if __name__ == '__main__':
    main()
