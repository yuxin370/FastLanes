"""Additional baselines with the same restored-state training loop as the core study."""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
PATHS = ('pytorch', 'dali', 'coordl', 'l3', 'cpu_dct')


from fastlens.benchmarks.paper.section_6_2_4.resident_training.resident_training_window import InitialExecutionState
from fastlens.benchmarks.paper.section_6_2_4.resident_training.resident_training_window import PROTOCOL
from fastlens.benchmarks.baselines.system_rgbnomore.training.paper_inputs import training_inputs, check_cpu_input


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', choices=['4090', 'H100', 'PRO 6000'], required=True)
    p.add_argument('--model', choices=['vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'efun', 'resnet24', 'resnet64'], required=True)
    p.add_argument('--path', choices=PATHS, required=True)
    p.add_argument('--trial', type=int, choices=range(3), default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    p.add_argument('--correctness-report', type=Path)
    p.add_argument('--l3-root', type=Path)
    p.add_argument('--l3-library', type=Path)
    args = p.parse_args()
    if args.path == 'l3' and (args.l3_root is None or args.l3_library is None):
        p.error('L3 requires --l3-root and --l3-library')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise FileExistsError(args.output / 'result.json')
    if args.gpu == 'PRO 6000' and not args.check:
        p.error('PRO 6000 is for correctness checks only')
    domain = 'dct' if args.path == 'cpu_dct' else 'rgb'
    if not args.check:
        if args.correctness_report is None:
            p.error('A passed correctness report is required')
        gate = json.loads(args.correctness_report.read_text())
        assert gate['correctness_passed'] and gate['protocol'] == PROTOCOL
        assert (gate['gpu_key'], gate['model'], gate['path']) == (args.gpu, args.model, args.path)
    os.environ['DCTNET_PROFILE'] = args.model
    import cv2
    cv2.setNumThreads(1)
    import torch
    from fastlens.benchmarks.paper.shared.training_window_rgb import build_window_model
    from fastlens.benchmarks.paper.shared.training_window_rgb import SEED
    from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import check_rgb_input
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
    make_source, paths, samples, decisions, variant = training_inputs(args.model, args.path, device, args.output,
        l3_root=args.l3_root, l3_library=args.l3_library)
    print('Preparing warm input pages and real inputs', flush=True)
    cache_warm = None if args.check else cache_pages(paths, warm=True)
    source = make_source()
    adapter = source
    del source
    torch.set_num_threads(8)
    initial.restore(model, optimizer, decay, scheduler)
    warmup_updates = 1 if args.check else 32
    measured_images = 1024 if args.check else 65536
    finite = torch.ones((), device=device, dtype=torch.bool)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    ids, batch_metadata = [], []
    input_checked = False

    def scope(name):
        return contextlib.nullcontext()

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
            with scope('input.handoff ' + tag):
                x = tuple(v.to(device, non_blocking=True) for v in batch.inputs)
                target = batch.labels.to(device, non_blocking=True)
                assert all(v.is_cuda and v.shape[0] == 64 for v in x) and target.is_cuda
                assert target.dtype == torch.int64
            if args.check and not input_checked:
                if domain == 'rgb':
                    check_rgb_input(x[0], target, batch.identities, samples, decisions,
                        pipeline=args.path, device=device, dali_config=variant, output=args.output)
                else:
                    check_cpu_input(args.model, batch, args.output)
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
    adapter.close()
    del adapter
    adapter = make_source()
    torch.set_num_threads(8)
    resource_conditions = wait_for_input_resources(os.environ['CUDA_VISIBLE_DEVICES'])
    cache_before = None if args.check else cache_pages(paths)
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
    print(f'Measuring {measured_images} image executions after full state restoration', flush=True)
    timestamp = datetime.datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    for step in range(measured_images // 1024):
        update(step, measured=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    assert bool(finite) and not torch.equal(before, parameter.detach())
    assert scheduler.completed_updates == measured_images // 1024
    assert len(ids) == measured_images
    assert len(set(ids)) == measured_images
    if domain == 'rgb':
        expected_ids = [s.logical_sample_id for s in samples[:measured_images]]
        assert ids == expected_ids
    input_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    prefix_hash = hashlib.sha256(json.dumps(ids[:512]).encode()).hexdigest()
    native_prefetch = None
    coordl_cache = adapter.cache_stats if args.path == 'coordl' else None
    adapter.close()
    result = dict(coordl_cache=coordl_cache, workers=8 if domain == 'dct' else 16, dali=variant, protocol=PROTOCOL, timestamp=timestamp, gpu_key=args.gpu, gpu=torch.cuda.get_device_name(),
        model=args.model, domain=domain, path=args.path, trial=args.trial,
        mode='correctness' if args.check else 'measurement',
        images=measured_images, unique_samples=len(set(ids)), resident_batches=None,
        seconds=seconds, images_per_s=measured_images/seconds, updates=scheduler.completed_updates,
        batch=64, accumulation=16, warmup_updates=warmup_updates, precision=precision,
        initialization=initialization, initial_model_hash=initial_hash, restored_state=restored_state,
        restoration_verified=True, loss_finite=True, parameter_update_verified=True, correctness_passed=True,
        training_loss=float(loss_sum)/measured_images, sample_order_sha256=input_hash, prefix_512_sha256=prefix_hash,
        tensor_contract=batch_metadata, native_prefetch=native_prefetch, resource_conditions=resource_conditions,
        cache_warm=cache_warm, cache_before=cache_before, host_threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), model_stream_priority=torch.cuda.current_stream().priority,
        compile='inductor/default', cuda_graphs=False, tf32=False, profile=False,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), command=sys.argv,
        scope='Basic augmentation; distinct dataset samples; restored state and input sequence after warmup.')
    (args.output / 'sample_ids.json').write_text(json.dumps(ids) + '\n')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('path', 'model', 'images_per_s', 'restoration_verified', 'unique_samples')}), flush=True)


if __name__ == '__main__':
    main()
