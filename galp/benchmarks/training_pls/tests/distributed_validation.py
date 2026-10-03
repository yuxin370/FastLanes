"""Compare one-rank and multi-rank full-validation predictions/aggregation."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from galp.benchmarks.training_pls.distributed import initialize, make_model, validate
from galp.benchmarks.system_rgbnomore.training.artifacts import tensor_state_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['swinv2', 'mobilenet24'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    rank, world, device = initialize()
    torch.set_num_threads(2)
    model = make_model(args.model, Path(os.environ['RGBNOMORE_ROOT']), device, 11997733)
    model_hash = tensor_state_sha256(model.state_dict())
    metrics, local = validate(model, model_name=args.model, root=args.data)
    predictions = [None] * world
    dist.all_gather_object(predictions, local)
    if rank == 0:
        ordered = sorted(pair for part in predictions for pair in part)
        assert [pair[0] for pair in ordered] == list(range(50000))
        result = dict(model=args.model, world_size=world, model_sha256=model_hash,
                      metrics=metrics, predictions=[pair[1] for pair in ordered],
                      scope='DDP validation correctness with scratch model; not checkpoint quality')
        if args.reference:
            reference = json.loads(args.reference.read_text())
            assert reference['model_sha256'] == model_hash
            assert reference['predictions'] == result['predictions']
            for field in ['samples', 'correct_predictions', 'correct_top5', 'top1', 'top5']:
                assert reference['metrics'][field] == metrics[field], field
            assert abs(reference['metrics']['loss'] - metrics['loss']) < 1e-8
            result['agreement'] = 'PASS: all 50000 predictions and integer counts match one rank'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'predictions'}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
