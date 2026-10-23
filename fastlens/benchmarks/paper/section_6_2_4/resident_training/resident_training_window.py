"""Training-window controls with real resident inputs and a restored initial state."""
import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
PROTOCOL = 'restored-state-real-resident-8-v2'
PATHS = ('ffcv_raw', 'ffcv_jpeg', 'rgb_resident', 'imagelanes_basic', 'dct_resident', 'imagelanes_full')


class RealResident:
    """Eight detached real microbatches, preserving tensor layout and identities."""
    def __init__(self, source):
        import torch
        from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingBatch
        self.batches = []
        for _ in range(8):
            batch = source.next_batch()
            self.batches.append(TrainingBatch(
                inputs=tuple(x.detach().clone(memory_format=torch.preserve_format) for x in batch.inputs),
                labels=batch.labels.detach().clone(), identities=list(batch.identities),
                augmentations=[], on_device=True, stage_seconds={}))
            del batch
        self.position = 0

    def next_batch(self):
        batch = self.batches[self.position % len(self.batches)]
        self.position += 1
        return batch

    def reset(self):
        self.position = 0

    def close(self):
        self.batches.clear()


class InitialExecutionState:
    def __init__(self, model, optimizer, decay, scheduler):
        from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import capture_rng_state
        self.model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.optimizer = copy.deepcopy(optimizer.state_dict())
        self.decay = copy.deepcopy(decay.state_dict())
        self.scheduler = copy.deepcopy(scheduler.state_dict())
        self.rng = capture_rng_state()

    def restore(self, model, optimizer, decay, scheduler):
        from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import restore_rng_state
        model.load_state_dict(self.model, strict=True)
        optimizer.load_state_dict(copy.deepcopy(self.optimizer))
        decay.load_state_dict(copy.deepcopy(self.decay))
        scheduler.load_state_dict(copy.deepcopy(self.scheduler))
        optimizer.zero_grad(set_to_none=False)
        restore_rng_state(self.rng)

    @staticmethod
    def hashes(model, optimizer, decay, scheduler):
        from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256, nested_state_sha256
        from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import capture_rng_state, rng_state_sha256
        return dict(model=tensor_state_sha256(model.state_dict()),
                    optimizer=nested_state_sha256(optimizer.state_dict()),
                    decay=nested_state_sha256(decay.state_dict()),
                    scheduler=nested_state_sha256(scheduler.state_dict()),
                    rng=rng_state_sha256(capture_rng_state()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', choices=['4090', 'H100'], required=True)
    p.add_argument('--model', choices=['vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'efun', 'resnet24', 'resnet64'], required=True)
    p.add_argument('--path', type=lambda name: name.replace('fastlens', 'imagelanes'), metavar='PATH', choices=PATHS, required=True)
    p.add_argument('--trial', type=int, choices=range(3), default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    p.add_argument('--correctness-report', type=Path)
    p.add_argument('--measured-images', type=int, choices=[16384, 65536], default=65536)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--ffcv-root', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise FileExistsError(args.output / 'result.json')
    resident = args.path.endswith('_resident')
    domain = 'dct' if args.path.startswith(('dct_', 'imagelanes_')) else 'rgb'
    if not args.check:
        if args.correctness_report is None:
            p.error('A passed correctness report is required')
        gate = json.loads(args.correctness_report.read_text())
        assert gate['correctness_passed'] and gate['protocol'] == PROTOCOL
        assert (gate['gpu_key'], gate['model'], gate['path']) == (args.gpu, args.model, args.path)
    os.environ['DCTNET_PROFILE'] = args.model
    import cv2
    cv2.setNumThreads(1)
    if domain == 'rgb':
        from ffcv.loader import Loader  # Resolve its JPEG ABI before loading model dependencies.
    import torch
    from fastlens.benchmarks.paper.shared.training_window_rgb import build_window_model
    from fastlens.benchmarks.paper.shared.training_window_rgb import SEED
    from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import FfcvWindow, NativeWindow, check_rgb_input
    from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingSample
    from fastlens.benchmarks.baselines.system_rgbnomore.training.augmentation import derive_augmentation
    from fastlens.benchmarks.training_pls.published_optimizer import build_published_optimizer
    from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import seed_everything
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    from fastlens.benchmarks.execution.resource_guard import wait_for_input_resources
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    assert args.gpu in torch.cuda.get_device_name()
    model_stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    model_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(model_stream)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(SEED)
    model, initial_hash, initialization = build_window_model(args.model, domain, device)
    optimizer, decay, scheduler = build_published_optimizer(model, total_updates=375600)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    initial = InitialExecutionState(model, optimizer, decay, scheduler)
    expected_state = initial.hashes(model, optimizer, decay, scheduler)
    execute = torch.compile(model, backend='inductor', options={'triton.cudagraphs': False}, dynamic=False)
    precision = 'fp32' if args.model == 'vitti' else 'bf16-autocast'
    samples, decisions = [], []
    if domain == 'rgb':
        ffcv_root = args.ffcv_root or ROOT / ('results/preparation/ffcv_jpeg/train'
                                             if args.path == 'ffcv_jpeg' else 'results/preparation/p0_ffcv')
        entries = json.loads((ffcv_root / 'samples.json').read_text())
        samples = [TrainingSample(e['logical_sample_id'], Path(e['path']), e['label'], e['width'], e['height'], e['galp_image_id']) for e in entries]
        decisions = [derive_augmentation(seed=SEED, epoch=2, logical_sample_id=s.logical_sample_id,
            source_width=s.width, source_height=s.height, domain='rgb') for s in samples]
        paths = [ffcv_root / 'train_window.beton']
        def make_source():
            return FfcvWindow(samples, device, decisions, root=ffcv_root)
    else:
        def make_source():
            source = NativeWindow(args.model, 'full' if args.path == 'imagelanes_full' else 'basic')
            source.begin()
            return source
        from fastlens.benchmarks.system_dct_major.common import parse_manifest
        manifest = ROOT / 'fastlens/data/compressed/imagenet512_train_block_major_premixed/dct/manifest.bin'
        paths = [Path(s['fls_path']) for s in parse_manifest(manifest)['shards']]
    print('Preparing warm input pages and real inputs', flush=True)
    cache_warm = None if args.check else cache_pages(paths, warm=True)
    source = make_source()
    if resident:
        adapter = RealResident(source)
        torch.cuda.synchronize()
        source.close()
        del source
    else:
        adapter = source
        del source
    torch.set_num_threads(8)
    initial.restore(model, optimizer, decay, scheduler)
    warmup_updates = 1 if args.check else 32
    measured_images = 1024 if args.check else args.measured_images
    finite = torch.ones((), device=device, dtype=torch.bool)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    ids, batch_metadata = [], []
    input_checked = False

    def scope(name):
        return torch.cuda.nvtx.range(name) if args.profile else contextlib.nullcontext()

    def update(index, measured=False):
        nonlocal finite, loss_sum, input_checked
        with scope(f'optimizer.zero u={index}'):
            optimizer.zero_grad(set_to_none=False)
            lr = scheduler.prepare_next_update()
        for micro in range(16):
            batch_id = index * 16 + micro
            tag = f'u={index} m={micro} b={batch_id}'
            with scope('input.request ' + tag):
                batch = adapter.next_batch()
            if args.profile:
                native = batch.native_stats_source if domain == 'dct' and not resident else None
                detail = f' pool={native.pool_index} pool_micro={native.microbatch_index_in_pool}' if native is not None else ''
                torch.cuda.nvtx.mark('input.acquired ' + tag + detail)
            with scope('input.handoff ' + tag):
                x, target = batch.inputs, batch.labels
                assert all(v.is_cuda and v.shape[0] == 64 for v in x) and target.is_cuda
                assert target.dtype == (torch.float32 if args.path == 'imagelanes_full' else torch.int64)
            if args.check and not input_checked:
                if domain == 'rgb':
                    check_rgb_input(x[0], target, batch.identities, samples, decisions,
                        pipeline='ffcv-jpeg' if args.path == 'ffcv_jpeg' else 'ffcv',
                        device=device, dali_config={}, output=args.output)
                input_checked = True
            if measured:
                ids.extend(i.logical_sample_id for i in batch.identities)
                if batch_id == 0:
                    batch_metadata.extend(dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype)) for v in (*x, target))
            with scope('model.forward ' + tag):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision != 'fp32'):
                    logits = execute(*x)
                    loss = torch.nn.functional.cross_entropy(logits, target)
            with scope('model.backward ' + tag):
                (loss / 16).backward()
            finite &= torch.isfinite(loss.detach())
            if args.check:
                finite &= torch.isfinite(logits.detach()).all()
            if measured:
                loss_sum += loss.detach().double() * 64
            del batch, x, target, logits, loss
        with scope(f'optimizer.update u={index}'):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite &= torch.isfinite(norm)
            optimizer.step()
            decay.step(lr)
            scheduler.complete_update()

    print(f'Warmup: {warmup_updates} updates', flush=True)
    for step in range(warmup_updates):
        update(step)
    torch.cuda.synchronize()
    if resident:
        adapter.reset()
    else:
        adapter.close()
        del adapter
        adapter = make_source()
    torch.set_num_threads(8)
    resource_conditions = wait_for_input_resources(os.environ['CUDA_VISIBLE_DEVICES'])
    cache_before = None if resident or args.check else cache_pages(paths)
    if cache_before is not None and cache_before['resident_fraction'] != 1:
        cache_pages(paths, warm=True)
        cache_before = cache_pages(paths)
        assert cache_before['resident_fraction'] == 1
    initial.restore(model, optimizer, decay, scheduler)
    restored_state = initial.hashes(model, optimizer, decay, scheduler)
    assert restored_state == expected_state, 'Warmup state was not fully restored'
    if not args.check:
        assert restored_state == gate['restored_state'], 'Measured state differs from correctness run'
    finite.fill_(True)
    loss_sum.zero_()
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    torch.cuda.synchronize()
    if args.profile:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('resident_study.measured_window')
    print(f'Measuring {measured_images} image executions after full state restoration', flush=True)
    timestamp = datetime.datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    for step in range(measured_images // 1024):
        update(step, measured=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if args.profile:
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    assert bool(finite) and not torch.equal(before, parameter.detach())
    assert scheduler.completed_updates == measured_images // 1024
    assert len(ids) == measured_images
    assert len(set(ids)) == (512 if resident else measured_images)
    if domain == 'rgb':
        expected_ids = [s.logical_sample_id for s in samples[:512 if resident else measured_images]]
        assert ids == (expected_ids * (measured_images // 512) if resident else expected_ids)
    input_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    prefix_hash = hashlib.sha256(json.dumps(ids[:512]).encode()).hexdigest()
    native_prefetch = adapter.pipeline.prefetch_stats if domain == 'dct' and not resident else None
    adapter.close()
    result = dict(protocol=PROTOCOL, timestamp=timestamp, gpu_key=args.gpu, gpu=torch.cuda.get_device_name(),
        model=args.model, domain=domain, path=args.path, trial=args.trial,
        mode='correctness' if args.check else 'profile' if args.profile else 'measurement',
        images=measured_images, unique_samples=len(set(ids)), resident_batches=8 if resident else None,
        resident_source='ffcv_raw' if args.path == 'rgb_resident' else 'imagelanes_basic' if resident else None,
        resident_cycle='batch index modulo 8' if resident else None,
        seconds=seconds, images_per_s=measured_images/seconds, updates=scheduler.completed_updates,
        batch=64, accumulation=16, warmup_updates=warmup_updates, precision=precision,
        initialization=initialization, initial_model_hash=initial_hash, restored_state=restored_state,
        restoration_verified=True, loss_finite=True, parameter_update_verified=True, correctness_passed=True,
        training_loss=float(loss_sum)/measured_images, sample_order_sha256=input_hash, prefix_512_sha256=prefix_hash,
        tensor_contract=batch_metadata, native_prefetch=native_prefetch, resource_conditions=resource_conditions,
        cache_warm=cache_warm, cache_before=cache_before, host_threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), model_stream_priority=torch.cuda.current_stream().priority,
        compile='inductor/default', cuda_graphs=False, tf32=False, profile=args.profile,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), command=sys.argv,
        scope='Resident repeats 512 real samples; end-to-end consumes distinct samples; no training-quality claim.')
    (args.output / 'sample_ids.json').write_text(json.dumps(ids) + '\n')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('path', 'model', 'images_per_s', 'restoration_verified', 'unique_samples')}), flush=True)


if __name__ == '__main__':
    main()
