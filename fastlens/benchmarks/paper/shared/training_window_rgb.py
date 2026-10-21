#!/usr/bin/env python3
"""One unprofiled P0 RGB window, using the existing motivation-study adapters."""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
SEED = 11997733


def update_microbatch_sizes(sample_count):
    return [min(64, sample_count-offset) for offset in range(0, sample_count, 64)]


def build_window_model(profile, domain, device):
    import torch
    from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import build_model, SWINV2_T_MODEL_ID, seed_everything
    from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
    seed_everything(SEED)
    os.environ['DCTNET_PROFILE'] = profile
    if profile in ('vitti', 'swinv2'):
        from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import VITTI_MODEL_ID
        model_id = VITTI_MODEL_ID if profile == 'vitti' else SWINV2_T_MODEL_ID
        if domain == 'dct':
            from fastlens.benchmarks.training_pls.train import build_paired_model
            model, initial_hash = build_paired_model((Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more'))),
                model_id=model_id, seed=SEED, device=device)
        else:
            model = build_model((Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more'))), 'rgb', device, model_id=model_id)
            initial_hash = tensor_state_sha256(model.state_dict())
        initialization = 'registered seeded scratch training constructor; seed=11997733'
    else:
        if profile == 'efun':
            import fastlens.benchmarks.dct_models as package
            from fastlens.benchmarks.dct_models import efun_backend as backend
            sys.modules['fastlens.benchmarks.dct_models.backend'] = backend
            package.backend = backend
        if domain == 'dct':
            from fastlens.benchmarks.dct_models import backend
            model = backend.model()
        else:
            from fastlens.benchmarks.dct_models import rgb
            model = rgb.model()
        # Match the existing full-epoch training initialization, including BN buffers.
        torch.manual_seed(SEED)
        for module in model.modules():
            if list(module.parameters(recurse=False)):
                module.reset_parameters()
        model.requires_grad_(True)
        initial_hash = tensor_state_sha256(model.state_dict())
        initialization = 'strict official checkpoint load, then seeded reset of all parameter-bearing layers; existing training_pls recipe'
    return model.to(device).train(), initial_hash, initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=['4090', 'H100', 'PRO 6000'], required=True)
    parser.add_argument('--model', choices=['vitti', 'swinv2', 'mobilenet', 'mobilenet24', 'mobilenet32', 'resnet24', 'resnet64', 'efun'], required=True)
    parser.add_argument('--pipeline', type=lambda name: name.replace('fastlens', 'imagelanes'), metavar='PATH', choices=['pytorch', 'dali', 'l3', 'ffcv', 'coordl', 'imagelanes', 'cpu-dct'], required=True)
    parser.add_argument('--l3-root', type=Path)
    parser.add_argument('--l3-library', type=Path)
    parser.add_argument('--trial', type=int, choices=range(3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Excluded small correctness run: one warmup and one measured update')
    parser.add_argument('--correctness-report', type=Path, help='Passed --check result required for a performance run')
    parser.add_argument('--model-only', action='store_true', help='Use resident tensors for this pipeline domain; retain identical model/update execution')
    parser.add_argument('--measured-images', type=int, choices=[65536,16384], default=65536)
    parser.add_argument('--full-epoch', action='store_true', help='One complete RGB training epoch; restore initialization after short warmup')
    parser.add_argument('--fastlens-augmentation', choices=['full', 'basic'], default='full')
    parser.add_argument('--ffcv-root', type=Path, default=ROOT/'results/preparation/p0_ffcv')
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument('--profile', action='store_true')
    capture.add_argument('--memory', action='store_true')
    args = parser.parse_args()
    if args.pipeline == 'l3' and (args.l3_root is None or args.l3_library is None):
        parser.error('L3 requires --l3-root and --l3-library')
    if args.full_epoch and (args.check or args.model_only or args.profile or args.memory or args.pipeline in ('imagelanes', 'cpu-dct')):
        parser.error('--full-epoch is an unprofiled RGB training measurement')
    if args.pipeline == 'cpu-dct' and (not args.profile or args.model not in ('vitti', 'mobilenet', 'mobilenet24')):
        parser.error('CPU-DCT window diagnostics require --profile and ViT-Ti or MobileNetV2-DCT24')
    if (args.output / 'result.json').exists():
        raise FileExistsError('Existing result must be audited/reused, not overwritten')
    if args.gpu == 'PRO 6000' and not args.check:
        parser.error('PRO 6000 is reserved for correctness diagnostics')
    if args.fastlens_augmentation != 'full' and (args.pipeline != 'imagelanes' or args.model_only):
        parser.error('Basic augmentation requires the FastLens input pipeline')
    jpeg_ffcv = (args.pipeline == 'ffcv' and not args.model_only and
                 json.loads((args.ffcv_root/'conversion.json').read_text()).get('write_mode', 'raw') == 'jpg')
    rgb_input_reference = ('ffcv-jpeg-full-decode-cpu-postprocess-v1' if jpeg_ffcv else
                           'native-full-decode-cpu-postprocess-v1' if args.pipeline in ('dali', 'coordl')
                           else 'pil-cpu-postprocess-v1')
    warmup_updates = 1 if args.check else 2 if args.full_epoch else 32
    if args.check:
        args.measured_images = 1024
    elif not args.profile and not args.memory and not args.model_only:
        if args.correctness_report is None:
            parser.error('Run --check first and provide --correctness-report')
        gate = json.loads(args.correctness_report.read_text())
        if args.pipeline == 'imagelanes' and gate.get('imagelanes_augmentation', 'full') != args.fastlens_augmentation:
            raise ValueError('Correctness report uses a different augmentation policy')
        if not (gate['correctness_passed'] and gate['mode']=='correctness' and gate['model']==args.model
                and gate['arm']==args.pipeline and args.gpu in gate['gpu']):
            raise ValueError('Correctness report does not match this training configuration')
        if (args.pipeline in ('dali', 'coordl') or jpeg_ffcv) and gate.get('rgb_input_reference') != rgb_input_reference:
            raise ValueError('Correctness report does not validate the standard native RGB decoder')
    profile = 'mobilenet24' if args.model == 'mobilenet' else args.model
    os.environ['DCTNET_PROFILE'] = profile
    # Resolve OpenCV's JPEG ABI before RGB-no-more imports its older JPEG binding.
    # DataLoader workers inherit these loaded libraries.
    import cv2
    cv2.setNumThreads(1)
    # Load the baseline's JPEG ABI before the artifact's native reader dependencies.
    if args.pipeline == 'ffcv' and not args.model_only:
        from ffcv.loader import Loader
    elif args.pipeline == 'coordl' and not args.model_only:
        from nvidia.dali import backend
    import torch
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    from fastlens.benchmarks.baselines.system_rgbnomore.training.augmentation import derive_augmentation
    from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import (
        TrainingSample, DaliTrainingAdapter, resolve_dali_variant)
    from fastlens.benchmarks.baselines.system_rgbnomore.training.sample_order import SampleIdentity
    from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import seed_everything
    from fastlens.benchmarks.training_pls.published_optimizer import build_published_optimizer

    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    if args.pipeline != 'imagelanes':
        model_stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
        model_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(model_stream)
    device = torch.device('cuda:0')
    if args.gpu not in torch.cuda.get_device_name(device):
        raise RuntimeError('CUDA device does not match the requested GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(SEED)
    domain = 'dct' if args.pipeline in ('imagelanes', 'cpu-dct') else 'rgb'
    precision = 'fp32' if profile=='vitti' else 'bf16-autocast'
    model, initial_hash, initialization = build_window_model(profile, domain, device)
    if not args.check and args.correctness_report is not None and gate['initial_model_hash'] != initial_hash:
        raise ValueError('Initialization differs from correctness gate')
    execute = torch.compile(model, backend='inductor', options={'triton.cudagraphs': False}, dynamic=False)
    optimizer, decay, scheduler = build_published_optimizer(model, total_updates=375600)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    variant = dict(resolve_dali_variant('d2'), num_threads=16, prefetch_queue_depth=2, seed=SEED, resize_antialias=False)
    if args.pipeline == 'l3':
        variant.update(decoder_mode='l3-full-decode-then-slice', role='l3-rgb-baseline')
    if args.model_only:
        from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import ResidentWindow
        adapter=ResidentWindow(args.model,domain,device)
        cache_warm=None
        paths=[]
    elif args.pipeline in ('imagelanes', 'cpu-dct'):
        from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import NativeWindow, CpuDctWindow
        adapter = (NativeWindow(profile,args.fastlens_augmentation) if args.pipeline == 'imagelanes'
                   else CpuDctWindow(profile, warmup_updates*1024+args.measured_images))
        print('Warming input representation page cache', flush=True)
        cache_warm = None if args.check else adapter.warm()
        paths = [] if args.check else adapter.paths
        adapter.begin()
    else:
        population = json.loads((ROOT/'fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json').read_text())['samples']
        entries = random.Random(SEED).sample(population, len(population) if args.full_epoch else (32 + 64) * 1024 + 256)
        if args.full_epoch:
            args.measured_images = len(entries)
            if args.measured_images != 1281167:
                raise ValueError('Full training manifest must contain 1281167 samples')
        del population
        samples = [TrainingSample(e['logical_sample_id'], Path(e['path']), e['label'], e['width'], e['height'], e['galp_image_id']) for e in entries]
        paths = ([args.ffcv_root/'train_window.beton'] if args.pipeline == 'ffcv' else [s.path for s in samples])
        if args.pipeline == 'l3':
            from fastlens.benchmarks.baselines.l3.prepare import encoded_path
            paths = [encoded_path(args.l3_root, e["logical_sample_id"]) for e in entries]
        print('Warming input representation page cache', flush=True)
        cache_warm = None if args.check else cache_pages(paths, warm=True)
        decisions = [derive_augmentation(seed=SEED, epoch=2, logical_sample_id=s.logical_sample_id,
            source_width=s.width, source_height=s.height, domain='rgb') for s in samples]
        if args.pipeline in ('ffcv','coordl'):
            from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import FfcvWindow, CoorDLWindow
            adapter = FfcvWindow(samples,device,decisions,root=args.ffcv_root) if args.pipeline == 'ffcv' else CoorDLWindow(samples,device,args.output,decisions)
        else:
            identities = [SampleIdentity(2, i, s.logical_sample_id) for i, s in enumerate(samples)]
            from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import PlannedRgbWindow
            cls = DaliTrainingAdapter if args.pipeline in ('dali', 'l3') else PlannedRgbWindow
            config = {'phase': 'train', 'dali': variant}
            if args.pipeline == 'l3':
                config['l3'] = dict(root=str(args.l3_root), library=str(args.l3_library))
            adapter = cls(samples, batch_size=64, workers=16, device=device, config=config)
            adapter.begin(identities, decisions)
    torch.set_num_threads(8)  # FFCV may change the host thread setting during setup.
    checked_rgb = False
    last_loss = None
    finite = torch.ones((), device=device, dtype=torch.bool)
    measured_ids = []
    consumed = 0
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    if args.full_epoch:
        initial_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        first_batch = adapter.next_batch()
    first_pending = args.full_epoch

    def span(name):
        return torch.cuda.nvtx.range(name) if args.profile else contextlib.nullcontext()

    def update(measure=False):
        nonlocal finite, consumed, loss_sum, last_loss, checked_rgb, first_pending
        optimizer.zero_grad(set_to_none=False)
        learning_rate = scheduler.prepare_next_update()
        update_samples = min(1024, args.measured_images-consumed) if args.full_epoch and measure else 1024
        for expected_batch in update_microbatch_sizes(update_samples):
            with span('input.prepare'):
                if args.full_epoch and (not measure or first_pending):
                    batch = first_batch
                    if measure:
                        first_pending = False
                else:
                    batch = adapter.next_batch()
            with span('input.handoff'):
                x = tuple(value.to(device, non_blocking=True) for value in batch.inputs)
                target = batch.labels.to(device, non_blocking=True)
            if any(value.shape[0] != expected_batch for value in x) or target.shape[0] != expected_batch or (not args.model_only and len(batch.identities) != expected_batch):
                raise RuntimeError('Microbatch differs from the expected sample count')
            if (args.check or args.full_epoch) and domain=='rgb' and not checked_rgb:
                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import check_rgb_input
                check_rgb_input(x[0],target,batch.identities,samples,decisions,
                    pipeline='ffcv-jpeg' if jpeg_ffcv else args.pipeline,
                    device=device,dali_config=variant,output=args.output)
                checked_rgb=True
            if measure:
                consumed += int(target.shape[0])
                measured_ids.extend(i.logical_sample_id for i in batch.identities)
            with span('model.forward'):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision!='fp32'):
                    logits = execute(*x)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                (loss * (expected_batch / update_samples)).backward()
            finite &= torch.isfinite(loss.detach())
            if args.check:
                finite &= torch.isfinite(logits.detach()).all()
            if measure:
                loss_sum += loss.detach().double() * target.shape[0]
                last_loss = loss.detach()
            del logits, loss, x, target, batch
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite &= torch.isfinite(norm)
        optimizer.step()
        decay.step(learning_rate)
        scheduler.complete_update()

    print(f'Compiling and executing {warmup_updates} excluded warmup updates', flush=True)
    for _ in range(warmup_updates):
        update()
    if args.full_epoch:
        # Compile the actual short final microbatch before starting the timer.
        tail = args.measured_images % 64
        if tail:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision!='fp32'):
                logits = execute(*(v[:tail].to(device) for v in first_batch.inputs))
                loss = torch.nn.functional.cross_entropy(logits, first_batch.labels[:tail].to(device))
            loss.backward()
            del logits, loss
        model.load_state_dict(initial_state)
        del initial_state
        from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
        if tensor_state_sha256(model.state_dict()) != initial_hash:
            raise ValueError('Warmup failed to restore initial model state')
        optimizer, decay, scheduler = build_published_optimizer(model, total_updates=375600)
        seed_everything(SEED)
    torch.cuda.synchronize()
    from fastlens.benchmarks.execution.resource_guard import wait_for_input_resources
    resource_conditions = wait_for_input_resources(os.environ['CUDA_VISIBLE_DEVICES'])
    cache_before = None if args.model_only or args.check else cache_pages(paths)
    cache_rewarm = None
    if cache_before is not None and cache_before['resident_pages'] != cache_before['pages']:
        print('Rewarming source pages evicted during the excluded resource wait', flush=True)
        cache_rewarm = cache_pages(paths, warm=True)
        resource_conditions = wait_for_input_resources(os.environ['CUDA_VISIBLE_DEVICES'])
        cache_before = cache_pages(paths)
    if cache_before is not None and cache_before['resident_pages'] != cache_before['pages']:
        raise RuntimeError('Source page cache is not fully warm')
    updates = math.ceil(args.measured_images / 1024)
    print(f'Measuring {updates} updates / {args.measured_images} images', flush=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'measurement_active').touch()
    memory = None
    if args.memory:
        from fastlens.benchmarks.paper.shared.window_memory import WindowMemory
        native_stats=(adapter.pipeline._module._lifetime_reclaim_stats_for_test
                      if args.pipeline=='imagelanes' and not args.model_only else None)
        memory = WindowMemory(args.output,native_stats=native_stats)
        memory.begin()
    if args.profile:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('fastlens.measured_pass')
    parameter = next(model.parameters())
    parameter_before = parameter.detach().clone()
    torch.cuda.synchronize()
    last_loss = None
    started = time.perf_counter()
    for step in range(updates):
        update(measure=True)
        if args.full_epoch and (step+1)%100 == 0:
            print(f'Progress: {step+1}/{updates} updates; {consumed} samples', flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if args.profile:
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    if memory:
        memory.end()
    (args.output/'measurement_active').unlink()
    if consumed != args.measured_images or (not args.model_only and (len(measured_ids) != consumed or len(set(measured_ids)) != consumed)):
        raise RuntimeError('Measured sample count/uniqueness gate failed')
    measured_start = 0 if args.full_epoch else warmup_updates*1024
    if not args.model_only and args.pipeline in ('pytorch','dali','l3','ffcv','coordl') and measured_ids != [s.logical_sample_id for s in samples[measured_start:measured_start+args.measured_images]]:
        raise RuntimeError('Sample sequence differs from the selected window')
    if torch.equal(parameter_before, parameter.detach()):
        raise RuntimeError('Measured optimizer updates did not change the first parameter')
    if scheduler.completed_updates != (0 if args.full_epoch else warmup_updates) + updates:
        raise RuntimeError('Optimizer update count mismatch')
    if not bool(finite.item()):
        raise FloatingPointError('Non-finite loss or gradient norm')
    native_prefetch = adapter.pipeline.prefetch_stats if args.pipeline == 'imagelanes' and not args.model_only else None
    coordl_cache=adapter.cache_stats if args.pipeline=='coordl' and not args.model_only else None
    adapter.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'sample_ids.json').write_text(json.dumps(measured_ids) + '\n')
    result = dict(timestamp=timestamp, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  gpu=torch.cuda.get_device_name(), model=args.model, model_variant=profile,
                  architecture_family=('EfficientNet-B0 reference / eFUN' if profile=='efun' else 'MobileNetV2' if profile.startswith('mobilenet') else 'ResNet-50' if profile.startswith('resnet') else profile),
                  initialization=initialization,
                  cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                  worker_count=8 if args.pipeline=='cpu-dct' else 1 if domain=='dct' else 16, decoder_threads=16 if args.pipeline in ('dali','l3','coordl') else 1 if args.pipeline in ('pytorch','cpu-dct') else None,
                  prefetch=None if args.pipeline=='imagelanes' else 2, cache_mode='warm' if not args.check else 'unchecked diagnostic',
                  comparison_type='deployment', effective_batch_size=1024,
                  consumed_samples=consumed, optimizer_updates=updates, final_loss=float(last_loss.item()),
                  loss_finite=True, parameter_update_verified=True, domain=domain, arm='model-only' if args.model_only else args.pipeline, trial=args.trial,
                  mode='correctness' if args.check else 'training_model_only' if args.model_only else 'training_full_epoch' if args.full_epoch else 'training', profile=args.profile, memory_diagnostic=args.memory, images=consumed, seconds=seconds, batch=64, accumulation=16, updates=updates,
                  full_epoch=args.full_epoch, warmup_initial_state_restored=args.full_epoch,
                  warmup_input_policy='reuse first real batch; compile final short batch; restore model and optimizer before measurement' if args.full_epoch else 'excluded prefix',
                  final_update_samples=args.measured_images%1024 or 1024,
                  warmup_updates=warmup_updates, precision=precision, initial_model_hash=initial_hash,
                  training_loss=float(loss_sum.item())/consumed, seed=SEED, compile='inductor/default', cuda_graphs=False,
                  tf32=False, pytorch_workers=0 if args.model_only else 8 if args.pipeline=='cpu-dct' else None if args.pipeline == 'imagelanes' else 16,
                  native_pls_workers=1 if args.pipeline=='imagelanes' and not args.model_only else None,
                  skipped_partial_sample_ids=adapter.skipped_partial_ids if domain=='dct' and not args.model_only else [],
                  native_prefetch=native_prefetch, coordl_cache=coordl_cache, host_threads=torch.get_num_threads(),
                  model_stream_priority=torch.cuda.current_stream().priority, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  numa_policy=subprocess.check_output(['numactl','--show'],text=True),
                  cache_warm=cache_warm, cache_rewarm=cache_rewarm, cache_before=cache_before, dali=variant if args.pipeline in ('dali','l3') else None,
                  l3=dict(root=str(args.l3_root), library=str(args.l3_library)) if args.pipeline=='l3' else None,
                  sample_order_sha256=hashlib.sha256(json.dumps(measured_ids).encode()).hexdigest(),
                  sample_ids_file=str(args.output/'sample_ids.json'), correctness_passed=True, rgb_prefix_checked=checked_rgb, resource_conditions=resource_conditions,
                  command=sys.argv, scope=f'{"Resident synthetic GPU tensors (not distinct dataset samples)" if args.model_only else "ImageNet-512"}; {consumed} image executions; {warmup_updates} excluded warmup updates; profiler={args.profile}; memory_diagnostic={args.memory}',
                  imagelanes_augmentation=args.fastlens_augmentation if domain=='dct' else None,
                  augmentation_configuration=(('B6 closed-pool crop, RandAugment, Mixup' if args.fastlens_augmentation=='full' else 'B6 closed-pool crop/flip, normalization, hard labels; no RandAugment or Mixup') if domain=='dct' else
                      'keyed per-sample RRC scale .05..1 aspect .75..4/3; flip; resize224 bilinear without antialias; normalize .5'),
                  rgb_input_reference=rgb_input_reference if domain=='rgb' else None,
                  crop_verification='shared keyed decisions; RGB prefix gate max=8/255, mean=1/255 against rgb_input_reference; native-decoder PIL differences diagnostic only; DCT separate matched reader/update gate',
                  notes='Deployment ratio only: RGB/DCT architecture, inputs and augmentation differ. EfficientNet-B0 is only an eFUN reference. No convergence claim; shared host.')

    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
