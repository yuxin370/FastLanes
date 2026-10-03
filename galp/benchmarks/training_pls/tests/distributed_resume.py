"""Fresh-process replay of checkpoints produced by distributed_smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from galp.benchmarks.training_pls.distributed import (
    initialize, make_model, pipeline_options, train_epoch, restore_checkpoint,
)
from galp.benchmarks.training_pls.published_optimizer import build_published_optimizer
from galp.torch.experimental import DirectDctPlsPipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    rank, world, device = initialize()
    torch.set_num_threads(2)
    reference = json.loads((args.reference / 'correctness.json').read_text())
    name = reference['model']
    model = make_model(name, Path(os.environ['RGBNOMORE_ROOT']), device, 11997733)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=True)
    optimizer, decayer, scheduler = build_published_optimizer(model, total_updates=30000)
    contract = dict(model=name, world_size=world, fixture_samples=67, precision='bf16')
    assert restore_checkpoint(args.reference / 'epoch1.pt', model, optimizer, decayer, scheduler, contract) == 1
    mapping = args.fixture / 'mapping.csv'
    pipeline = DirectDctPlsPipeline(args.fixture / 'dct/manifest.bin', mapping, training_seed=11997733,
        expected_mapping_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
        rank=rank, world_size=world, microbatch_images=2, segment_images=16,
        segments_per_pool=4, io_backend='pread', **pipeline_options(name))
    result = train_epoch(ddp, pipeline, optimizer, decayer, scheduler, model_name=name,
        epoch=1, microbatch=2, global_batch=32, precision='bf16', columns=None)
    for key in ['samples', 'optimizer_updates', 'loss', 'rank_orders']:
        assert result[key] == reference['bf16_training'][1][key], key
    assert scheduler.completed_updates == 6
    target = torch.load(args.reference / 'epoch2_reference.pt', map_location='cpu', weights_only=False)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # The same FP32 bounds used in the serial/DDP gradient check. In a new
    # process DDP may start with its initial (not yet rebuilt) bucket ordering.
    # Keep integer counters and the complete stochastic/sample sequence exact.
    torch.testing.assert_close(state, target['model'], rtol=1e-4, atol=2e-6)
    actual_optimizer = optimizer.state_dict()
    for value in actual_optimizer['state'].values():
        for key in value:
            if torch.is_tensor(value[key]):
                value[key] = value[key].cpu()
    torch.testing.assert_close(actual_optimizer, target['optimizer'], rtol=1e-4, atol=2e-6)
    assert decayer.state_dict() == target['decayer']
    assert scheduler.state_dict() == target['scheduler']
    torch.testing.assert_close(torch.cuda.get_rng_state(), target['rank_rng'][rank]['torch_cuda'], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), target['rank_rng'][rank]['torch_cpu'], rtol=0, atol=0)
    maximum = max(float((v.float() - target['model'][k].float()).abs().max()) for k, v in state.items())
    pipeline.close()
    if rank == 0:
        reducer = ddp._get_ddp_logging_data()
        result.update(passed=True, scope='fresh-process checkpoint replay; exact loss/order/counters/RNG; FP32 state tolerance',
                      model_max_abs_difference=maximum,
                      model_bitwise_equal=result['model_sha256'] == reference['bf16_training'][1]['model_sha256'],
                      ddp_bucket_sizes=reducer.get('bucket_sizes'),
                      ddp_rebuilt_bucket_sizes=reducer.get('rebuilt_bucket_sizes'),
                      ddp_rebuilt_parameter_indices=reducer.get('rebuilt_per_bucket_param_indices'),
                      state_rtol=1e-4, state_atol=2e-6)
        (args.reference / 'fresh_process_resume.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
