"""Check projected RandAugment dependencies against a dense 192-channel native pool."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import galp.benchmarks.dct_models.backend as B
sys.path.insert(0, str(B.REPO / 'build/galp/torch'))
from galp.torch.experimental import DirectDctPlsPipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--mapping-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    os.environ['GALP_BLOCK_MAJOR_ACCESS_DIR'] = str(args.manifest.parent / 'block_major_access_v1')
    # Deliberately omit transposed frequencies in the requested output. The native dependency closure must add them.
    chosen = [0, 1, 7, 10, 64, 67, 128, 144]
    all_channels = [[c // 64, c % 64, (c + 1) * .1, (c + 2) * .01] for c in range(192)]
    records, targets, positions = [], [], []
    for dense in (True, False):
        pipeline = DirectDctPlsPipeline(args.manifest, args.mapping, training_seed=11997733,
            expected_mapping_sha256=args.mapping_sha256, segments_per_pool=1,
            output_grid_size=28, output_channels=all_channels if dense else [all_channels[c] for c in chosen])
        pipeline.start_epoch(0)
        pool = pipeline.next_pool()
        maximum = 0.
        for index, batch in enumerate(pool):
            tensor = batch.projected.cpu()
            if dense:
                records.append(tensor[:, chosen].clone())
                targets.append(batch.targets.cpu())
                positions.append(batch.global_image_ids)
            else:
                torch.testing.assert_close(tensor, records[index], atol=0, rtol=0)
                torch.testing.assert_close(batch.targets.cpu(), targets[index], atol=0, rtol=0)
                assert batch.global_image_ids == positions[index]
                maximum = max(maximum, float((tensor - records[index]).abs().max()))
            del batch, tensor
        del pool
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.synchronize()
    result = dict(samples=sum(len(p) for p in positions), selected_flat_frequencies=chosen,
        max_abs=maximum, input_targets_ids_exact=True, grid=28,
        scope='same native augmentation math; dense-vs-projected checks dependency closure, output placement, normalization and Mixup; not an independent augmentation implementation')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'projection_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
