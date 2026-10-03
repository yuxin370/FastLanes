"""Run with torchrun on CUDA; exercises real native DCT inputs and DDP state.

The fixture intentionally has short pools, incomplete updates and empty ranks.
Gradient tolerances are fixed before running: rtol=1e-4, atol=2e-6 (FP32).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from galp.benchmarks.training_pls.distributed import (
    initialize, make_model, pipeline_options, model_inputs, train_epoch,
    save_checkpoint, restore_checkpoint,
)
from galp.benchmarks.training_pls.published_optimizer import build_published_optimizer
from galp.benchmarks.system_rgbnomore.training.model_factory import seed_everything
from galp.benchmarks.system_rgbnomore.training.artifacts import nested_state_sha256
from galp.torch.experimental import DirectDctPlsPipeline


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 1000)

    def forward(self, *inputs):
        first = sum(x.flatten(1).mean(1) for x in inputs)
        second = sum(x.flatten(1).square().mean(1) for x in inputs)
        return self.head(torch.stack((first, second), dim=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--model', choices=['swinv2', 'mobilenet24'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rank, world, device = initialize()
    torch.set_num_threads(2)
    os.environ['DCTNET_PROFILE'] = 'mobilenet24'
    seed = 11997733
    mapping = args.fixture / 'mapping.csv'
    common = dict(training_seed=seed, microbatch_images=2, segment_images=16,
                  io_backend='pread',
                  segments_per_pool=4, expected_mapping_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
                  **pipeline_options(args.model))

    def pipeline(r, w):
        return DirectDctPlsPipeline(args.fixture / 'dct/manifest.bin', mapping,
                                    rank=r, world_size=w, **common)

    def collect(r, w):
        result = []
        pipe = pipeline(r, w).start_epoch(0)
        while pipe.has_next_pool:
            pool = pipe.next_pool()
            batches = []
            for batch in pool:
                batches.append((list(batch.global_image_ids),
                    tuple(x.clone() for x in model_inputs(batch, args.model)), batch.targets.clone()))
                del batch
            result.append(batches)
            del pool
            pipe.reclaim_finished_pools()
        pipe.close()
        return result

    serial = collect(0, 1)
    sharded = collect(rank, world)
    assert len(serial) == len(sharded)
    ids = []
    for full, local in zip(serial, sharded):
        expected = full[rank::world]
        assert len(local) == len(expected)
        for observed, reference in zip(local, expected):
            assert observed[0] == reference[0]
            ids.extend(observed[0])
            for x, y in zip(observed[1], reference[1]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            torch.testing.assert_close(observed[2], reference[2], rtol=0, atol=0)
    all_ids = [None] * world
    dist.all_gather_object(all_ids, ids)
    assert sorted(i for part in all_ids for i in part) == list(range(67))

    # Direct mathematical control of weighted DDP accumulation, using exactly
    # the native inputs above. It includes no_sync and a 3-image final update.
    seed_everything(seed)
    probe = Probe().to(device)
    reference = Probe().to(device)
    reference.load_state_dict(probe.state_dict())
    ddp_probe = DDP(probe, device_ids=[device.index])
    opt, decay, schedule = build_published_optimizer(probe, total_updates=30000)
    ref_opt, ref_decay, ref_schedule = build_published_optimizer(reference, total_updates=30000)
    pipe = pipeline(rank, world)
    result = train_epoch(ddp_probe, pipe, opt, decay, schedule, model_name=args.model,
        epoch=0, microbatch=2, global_batch=32, precision='fp32', columns=None)
    pipe.close()
    for pool in serial:
        for start in range(0, len(pool), 16):
            window = pool[start:start + 16]
            n = sum(len(batch[0]) for batch in window)
            ref_opt.zero_grad(set_to_none=True)
            lr = ref_schedule.prepare_next_update()
            for _, x, target in window:
                (torch.nn.functional.cross_entropy(reference(*x), target, reduction='sum') / n).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
            ref_opt.step()
            ref_decay.step(lr)
            ref_schedule.complete_update()
    assert result['samples'] == 67 and result['optimizer_updates'] == 3
    for x, y in zip(probe.parameters(), reference.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-4, atol=2e-6)
    for x, y in zip(opt.state.values(), ref_opt.state.values()):
        for key in x:
            torch.testing.assert_close(x[key], y[key], rtol=1e-4, atol=2e-6)
    del ddp_probe, probe, reference, opt, ref_opt, serial, sharded

    # Actual model training and epoch-boundary resume. No tensors, optimizer
    # state, data cursor or rank-local stochastic state may drift after resume.
    model = make_model(args.model, Path(os.environ['RGBNOMORE_ROOT']), device, seed)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=True)
    optimizer, decayer, scheduler = build_published_optimizer(model, total_updates=30000)
    pipe = pipeline(rank, world)
    seed_everything(seed + rank)
    record0 = train_epoch(ddp, pipe, optimizer, decayer, scheduler, model_name=args.model,
        epoch=0, microbatch=2, global_batch=32, precision='bf16', columns=None)
    args.output.mkdir(parents=True, exist_ok=True)
    contract = dict(model=args.model, world_size=world, fixture_samples=67, precision='bf16')
    save_checkpoint(args.output / 'epoch1.pt', model, optimizer, decayer, scheduler, contract=contract, epoch=1)
    record1 = train_epoch(ddp, pipe, optimizer, decayer, scheduler, model_name=args.model,
        epoch=1, microbatch=2, global_batch=32, precision='bf16', columns=None)
    expected = nested_state_sha256(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                        decayer=decayer.state_dict(), scheduler=scheduler.state_dict()))
    save_checkpoint(args.output / 'epoch2_reference.pt', model, optimizer, decayer, scheduler,
                    contract=contract, epoch=2)
    reducer = ddp._get_ddp_logging_data()
    complete = restore_checkpoint(args.output / 'epoch1.pt', model, optimizer, decayer, scheduler, contract)
    assert complete == 1 and scheduler.completed_updates == 3
    replay = train_epoch(ddp, pipe, optimizer, decayer, scheduler, model_name=args.model,
        epoch=1, microbatch=2, global_batch=32, precision='bf16', columns=None)
    actual = nested_state_sha256(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                      decayer=decayer.state_dict(), scheduler=scheduler.state_dict()))
    assert actual == expected
    for key in ['samples', 'optimizer_updates', 'loss', 'rank_orders', 'model_sha256']:
        assert replay[key] == record1[key], key
    pipe.close()
    if rank == 0:
        output = dict(model=args.model, world_size=world, passed=True,
                      native_tensors_and_mixup='bitwise equal to serial', sample_coverage=67,
                      weighted_gradient_and_optimizer='PASS; rtol=1e-4 atol=2e-6',
                      bf16_training=[record0, record1], resume='bitwise model/optimizer/scheduler and loss equality',
                      ddp_bucket_sizes=reducer.get('bucket_sizes'),
                      ddp_rebuilt_bucket_sizes=reducer.get('rebuilt_bucket_sizes'),
                      ddp_rebuilt_parameter_indices=reducer.get('rebuilt_per_bucket_param_indices'),
                      scope='synthetic correctness, not convergence or performance')
        (args.output / 'correctness.json').write_text(json.dumps(output, indent=2) + '\n')
        print(json.dumps(output), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
