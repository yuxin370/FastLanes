"""ImageLanes-only, epoch-boundary-resumable DDP convergence training.

The native reader shards complete, already scheduled microbatches. Crop,
RandAugment, Mixup partners and Mixup seeds retain their single-rank semantics.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from galp.torch.experimental import DirectDctPlsPipeline
from .model_registry import SWINV2_T_MODEL_ID
from .published_optimizer import build_published_optimizer
from galp.benchmarks.system_rgbnomore.training.model_factory import seed_everything
from galp.benchmarks.system_rgbnomore.training.artifacts import tensor_state_sha256


def initialize():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return rank, world, torch.device('cuda', local)


def make_model(name, root, device, seed):
    seed_everything(seed)
    if name == 'swinv2':
        from .train import build_paired_model
        model, _ = build_paired_model(root, model_id=SWINV2_T_MODEL_ID, device=device, seed=seed)
    else:
        os.environ['DCTNET_PROFILE'] = 'mobilenet24'
        from galp.benchmarks.dct_models import backend
        model = backend.model().requires_grad_(True)
        torch.manual_seed(seed)
        for module in model.modules():
            if list(module.parameters(recurse=False)):
                module.reset_parameters()
        model.to(device)
    return model.train()


def pipeline_options(model):
    if model == 'swinv2':
        return {}
    from galp.benchmarks.dct_models.training_pls import channels
    return dict(output_grid_size=112, output_channels=channels())


def model_inputs(batch, name):
    return (batch.projected,) if name == 'mobilenet24' else (batch.y, batch.cbcr)


def dummy_inputs(name, device):
    shapes = [(1, 24, 112, 112)] if name == 'mobilenet24' else [
        (1, 1, 28, 28, 8, 8), (1, 2, 14, 14, 8, 8)]
    return tuple(torch.zeros(s, device=device) for s in shapes)


def label_columns(mapping):
    with mapping.open() as stream:
        classes = {}
        for row in csv.DictReader(stream):
            name = row['logical_sample_id'].split('/')[1]
            value = int(row['label'])
            if name in classes and classes[name] != value:
                raise ValueError('class labels change within the mapping')
            classes[name] = value
    if sorted(classes.values()) != list(range(1000)):
        raise ValueError('training labels must form the complete ImageNet class permutation')
    return [classes[name] for name in sorted(classes)]


def synchronize_buffers(model):
    # Standard DDP uses rank 0 running statistics for evaluation/checkpointing.
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)


def train_epoch(ddp, pipeline, optimizer, decayer, scheduler, *, model_name,
                epoch, microbatch, global_batch, precision, columns,
                max_pools=0):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = next(ddp.parameters()).device
    accumulation = global_batch // (microbatch * world)
    if accumulation < 1 or global_batch % (microbatch * world):
        raise ValueError('global batch must be divisible by microbatch * world size')
    ddp.train()
    coverage = torch.zeros(pipeline.sample_count, dtype=torch.int32, device=device)
    local_seen = np.zeros(pipeline.sample_count, dtype=np.bool_)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    dummy = dummy_inputs(model_name, device)
    order = hashlib.sha256()
    start_updates = scheduler.completed_updates
    local_count = global_count = expected_updates = pools = 0
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    pipeline.start_epoch(epoch)
    while pipeline.has_next_pool:
        pool = pipeline.next_pool()
        lengths = [None] * world
        dist.all_gather_object(lengths, (pool.pool_index, pool.image_count))
        if len({value[0] for value in lengths}) != 1:
            raise RuntimeError('ranks disagree on the global PLS pool boundary')
        sizes = [value[1] for value in lengths]
        steps = math.ceil(max(sizes) / microbatch)
        expected_updates += math.ceil(steps / accumulation)
        global_count += sum(sizes)
        for step in range(steps):
            if step % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
                lr = scheduler.prepare_next_update()
                window_samples = sum(max(0, min(n - step * microbatch,
                                                accumulation * microbatch)) for n in sizes)
            synchronize = (step + 1) % accumulation == 0 or step + 1 == steps
            count = max(0, min(microbatch, sizes[rank] - step * microbatch))
            with nullcontext() if synchronize else ddp.no_sync():
                if count:
                    batch = next(pool)
                    ids = np.asarray(batch.global_image_ids, dtype=np.int64)
                    if len(ids) != count or len(np.unique(ids)) != count or local_seen[ids].any():
                        raise RuntimeError('duplicate or incorrect local sample IDs')
                    local_seen[ids] = True
                    coverage[torch.as_tensor(ids, device=device)] += 1
                    order.update(ids.astype('<u8').tobytes())
                    x = model_inputs(batch, model_name)
                    target = batch.targets if columns is None else batch.targets[:, columns]
                else:
                    # Empty tail ranks still join DDP reduction. Eval mode prevents
                    # dummy examples changing BatchNorm statistics or dropout RNG.
                    ddp.module.eval()
                    x = dummy
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    logits = ddp(*x)
                    loss = (torch.nn.functional.cross_entropy(logits, target, reduction='sum')
                            if count else logits.sum() * 0)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('non-finite training loss')
                # DDP averages gradients across ranks; undo that average and
                # normalize by actual global samples, including uneven tails.
                (loss * (world / window_samples)).backward()
                loss_sum += loss.detach().double()
                local_count += count
                if count:
                    del target, batch
                else:
                    ddp.module.train()
                del x, logits, loss
            if synchronize:
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                decayer.step(lr)
                scheduler.complete_update()
        torch.cuda.current_stream().synchronize()
        del pool
        pipeline.reclaim_finished_pools()
        pools += 1
        if rank == 0:
            print(json.dumps(dict(stage='train', epoch=epoch, pools=pools,
                                  samples=global_count, updates=scheduler.completed_updates)), flush=True)
        if max_pools and pools == max_pools:
            break
    torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    dist.all_reduce(coverage)
    dist.all_reduce(loss_sum)
    if int(coverage.sum()) != global_count or int(coverage.max()) != 1:
        raise RuntimeError('global sample coverage/count failed')
    full = global_count == pipeline.sample_count
    if not max_pools and (not full or not bool((coverage == 1).all())):
        raise RuntimeError('incomplete global training epoch')
    updates = scheduler.completed_updates - start_updates
    if updates != expected_updates:
        raise RuntimeError('incorrect optimizer update count')
    if not all(bool(torch.isfinite(p).all()) for p in ddp.parameters()):
        raise FloatingPointError('non-finite parameters after epoch')
    synchronize_buffers(ddp.module)
    orders = [None] * world
    dist.all_gather_object(orders, dict(rank=rank, samples=local_count, sample_order_sha256=order.hexdigest()))
    hashes = [None] * world
    dist.all_gather_object(hashes, tensor_state_sha256(ddp.module.state_dict()))
    if len(set(hashes)) != 1:
        raise RuntimeError('model replicas differ after DDP updates')
    return dict(epoch=epoch, samples=global_count, full_epoch=full, optimizer_updates=updates,
                elapsed_s=elapsed.item(), images_per_s=global_count / elapsed.item(),
                loss=loss_sum.item() / global_count, coverage='PASS', finite='PASS',
                replica_state='PASS', model_sha256=hashes[0], rank_orders=orders)


@torch.inference_mode()
def validate(model, *, model_name, root, count=50000, batch_size=64):
    """Partition physical validation batches; reduce integer counts and loss sums."""
    from galp.benchmarks.training_pls.published_augmentation import published_validation_augmentation
    rank, world = dist.get_rank(), dist.get_world_size()
    device = next(model.parameters()).device
    stored = json.loads((root / 'samples.json').read_text())[:count]
    if len(stored) != count or len({s['logical_sample_id'] for s in stored}) != count:
        raise ValueError('validation population/identities differ')
    os.environ['GALP_BLOCK_MAJOR_ACCESS_DIR'] = str(root / 'access')
    if model_name == 'mobilenet24':
        import _galp_direct_dct as native
        from galp.benchmarks.dct_models import backend
        from galp.benchmarks.dct_models.evaluate_shards import native_options
        model_fields = {'model', 'checkpoint', 'shape', 'indices', 'mean', 'std'}
        profile = json.loads((root / 'profile.json').read_text())
        if ({k: v for k, v in profile.items() if k not in model_fields} !=
                {k: v for k, v in backend.profile().items() if k not in model_fields}):
            raise ValueError('validation representation differs from the model input contract')
        reader = native.DirectDctReader(str(root / 'manifest.bin'))
        options = native_options(pushdown=True, projected=True)
    else:
        from galp.torch import DirectDctReader
        from galp.profiles.rgbnomore import VALIDATION
        reader = DirectDctReader(root / 'manifest.bin')
    synchronize_buffers(model)
    model.eval()
    counts = torch.zeros(3, dtype=torch.int64, device=device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    predictions = []
    for begin in range(rank * batch_size, count, world * batch_size):
        end = min(begin + batch_size, count)
        entries = stored[begin:end]
        ids = list(range(begin, end))
        if model_name == 'mobilenet24':
            batch = reader.read_batch(ids, **options)
            inputs = (batch.projected,)
            labels = [s['model_label'] for s in entries]
        else:
            transforms = [published_validation_augmentation(
                logical_sample_id=s['logical_sample_id'], source_width=s['width'],
                source_height=s['height']).native_dct_descriptor() for s in entries]
            batch = reader.read(ids, VALIDATION, transforms=transforms)
            inputs = (batch.y, batch.cbcr)
            labels = [s['label'] for s in entries]
        if list(batch.global_image_ids) != ids:
            raise RuntimeError('validation native sample IDs changed')
        target = torch.tensor(labels, device=device)
        logits = model(*inputs)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('non-finite validation logits')
        top = logits.topk(5, dim=1).indices
        counts += torch.stack([torch.tensor(len(ids), device=device),
                               (top[:, 0] == target).sum(), (top == target[:, None]).any(1).sum()])
        loss_sum += torch.nn.functional.cross_entropy(logits, target, reduction='sum').double()
        predictions.extend(zip(ids, top[:, 0].cpu().tolist()))
        del logits, target, inputs, batch
    dist.all_reduce(counts)
    dist.all_reduce(loss_sum)
    total, top1, top5 = counts.tolist()
    if total != count:
        raise RuntimeError('distributed validation sample count mismatch')
    model.train()
    return dict(samples=total, correct_predictions=top1, correct_top5=top5,
                top1=100 * top1 / total, top5=100 * top5 / total, loss=loss_sum.item() / total), predictions


def save_checkpoint(path, model, optimizer, decayer, scheduler, *, contract, epoch):
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, dict(python=random.getstate(), numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state()))
    if dist.get_rank() == 0:
        payload = dict(contract=contract, completed_epochs=epoch, model=model.state_dict(),
                       optimizer=optimizer.state_dict(), decayer=decayer.state_dict(),
                       scheduler=scheduler.state_dict(), rank_rng=states)
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)
    dist.barrier()


def restore_checkpoint(path, model, optimizer, decayer, scheduler, contract):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['contract'] != contract:
        raise ValueError('DDP resume contract differs')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    decayer.load_state_dict(payload['decayer'])
    scheduler.load_state_dict(payload['scheduler'])
    rng = payload['rank_rng'][dist.get_rank()]
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'])
    torch.cuda.set_rng_state(rng['torch_cuda'])
    return payload['completed_epochs']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['swinv2', 'mobilenet24'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--mapping-sha256', required=True)
    p.add_argument('--validation-data', type=Path, required=True)
    p.add_argument('--rgbnomore-root', type=Path, default=Path(os.environ['RGBNOMORE_ROOT']))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--stop-after-epoch', type=int)
    p.add_argument('--microbatch', type=int, default=64)
    p.add_argument('--global-batch', type=int, default=1024)
    p.add_argument('--segments-per-pool', type=int, default=4)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--seed', type=int, default=11997733)
    p.add_argument('--torch-threads', type=int, default=4)
    p.add_argument('--io-backend', choices=['io_uring', 'pread'], default='io_uring')
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-pools', type=int, default=0, help='diagnostic only; no checkpoint')
    p.add_argument('--validation-count', type=int, default=50000)
    args = p.parse_args()
    if min(args.epochs, args.microbatch, args.global_batch, args.segments_per_pool, args.torch_threads) <= 0:
        p.error('epochs, batches, pool size and thread count must be positive')
    if args.max_pools < 0 or (args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs):
        p.error('invalid diagnostic pool limit or stop epoch')
    rank, world, device = initialize()
    torch.set_num_threads(args.torch_threads)
    if args.global_batch % (args.microbatch * world):
        p.error('global batch must be divisible by microbatch * world size')
    if args.segments_per_pool * 1024 % args.global_batch:
        p.error('global batch must divide the full PLS pool size')
    if not 0 <= args.validation_count <= 50000:
        p.error('validation count must be between 0 and 50000')
    if not args.max_pools and args.validation_count != 50000:
        p.error('convergence runs require all 50,000 validation images')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'latest.pt').exists() and args.resume is None:
        raise FileExistsError('use --resume for an existing training checkpoint')
    os.environ['GALP_BLOCK_MAJOR_ACCESS_DIR'] = str(args.manifest.parent / 'block_major_access_v1')
    model = make_model(args.model, args.rgbnomore_root, device, args.seed)
    initial_hash = tensor_state_sha256(model.state_dict())
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=True)
    pipeline = DirectDctPlsPipeline(args.manifest, args.mapping, training_seed=args.seed,
        expected_mapping_sha256=args.mapping_sha256, rank=rank, world_size=world,
        microbatch_images=args.microbatch, segments_per_pool=args.segments_per_pool,
        io_backend=args.io_backend,
        **pipeline_options(args.model))
    columns = (torch.tensor(label_columns(args.mapping), device=device) if args.model == 'mobilenet24' else None)
    updates_per_epoch = math.ceil(pipeline.sample_count / args.global_batch)
    optimizer, decayer, scheduler = build_published_optimizer(model, total_updates=updates_per_epoch * args.epochs)
    contract = dict(model=args.model, initialization='seeded scratch', initial_model_sha256=initial_hash,
        seed=args.seed, world_size=world, microbatch=args.microbatch, global_batch=args.global_batch,
        precision=args.precision, epochs=args.epochs, segments_per_pool=args.segments_per_pool,
        samples=pipeline.sample_count, mapping_sha256=args.mapping_sha256,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        io_backend=args.io_backend,
        validation_data=str(args.validation_data.resolve()),
        augmentation='B6; unchanged per-microbatch native crop/RandAugment/Mixup',
        sharding='global pool microbatch round-robin', batchnorm='local microbatch; rank0 running buffers',
        optimizer='published AdamW + independent decay; lr=.003; warmup=10000; clip=1',
        compile=False)
    seed_everything(args.seed + rank)
    first = restore_checkpoint(args.resume, model, optimizer, decayer, scheduler, contract) if args.resume else 0
    if rank == 0:
        (args.output_dir / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    history = ([json.loads(line) for line in (args.output_dir / 'metrics.jsonl').read_text().splitlines()]
               if args.resume else [])
    best_top1 = max((row['validation']['top1'] for row in history), default=-1.)
    try:
        for epoch in range(first, args.stop_after_epoch or args.epochs):
            epoch_started = time.perf_counter()
            os.environ['GALP_BLOCK_MAJOR_ACCESS_DIR'] = str(args.manifest.parent / 'block_major_access_v1')
            record = train_epoch(ddp, pipeline, optimizer, decayer, scheduler, model_name=args.model,
                epoch=epoch, microbatch=args.microbatch, global_batch=args.global_batch,
                precision=args.precision, columns=columns, max_pools=args.max_pools)
            if record['full_epoch'] and record['optimizer_updates'] != updates_per_epoch:
                raise RuntimeError('full epoch update count differs from global batch contract')
            if args.max_pools:
                pipeline.close()
            if args.validation_count:
                validation_started = time.perf_counter()
                record['validation'], _ = validate(model, model_name=args.model,
                    root=args.validation_data, count=args.validation_count)
                record['validation_elapsed_s'] = time.perf_counter() - validation_started
                record['validation_completed_s'] = time.perf_counter() - epoch_started
            if not args.max_pools:
                save_checkpoint(args.output_dir / 'latest.pt', model, optimizer, decayer, scheduler,
                                contract=contract, epoch=epoch + 1)
                if record['validation']['top1'] > best_top1:
                    best_top1 = record['validation']['top1']
                    if rank == 0:
                        shutil.copyfile(args.output_dir / 'latest.pt', args.output_dir / 'best.pt')
                    dist.barrier()
            record['epoch_wall_s'] = time.perf_counter() - epoch_started
            record['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
            record['learning_rate'] = optimizer.param_groups[0]['lr']
            record['optimizer_updates_total'] = scheduler.completed_updates
            if rank == 0:
                with (args.output_dir / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps(dict(stage='epoch_complete', **record)), flush=True)
            if args.max_pools:
                break
    finally:
        pipeline.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
