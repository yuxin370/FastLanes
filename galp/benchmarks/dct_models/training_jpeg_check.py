"""Real-JPEG check of the shared external-tensor augmentation entry point."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import galp.benchmarks.dct_models.backend as B
sys.path.insert(0, str(B.REPO / 'build/galp/torch'))
import _galp_direct_dct as native
from galp.benchmarks.dct_models.training_jpeg import JpegDctDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    records = []
    with args.mapping.open() as handle:
        for row in csv.DictReader(handle):
            records.append((row['logical_sample_id'], int(row['label']), row['source_path']))
            if len(records) == 4:
                break
    channels = [[c//64, c%64, m, s] for c, m, s in
                zip(B.CHANNELS, B.MEAN.flatten().tolist(), B.STD.flatten().tolist())]
    selected = native.ProjectedDctTrainingAugment(channels, 1000)
    dense = native.ProjectedDctTrainingAugment([[c, f, 0., 1.] for c in range(3) for f in range(64)], 1000)
    dataset = JpegDctDataset(records, list(range(4)), 0,
                            [(c, f) for c in range(3) for f in range(64)], B.GRID)
    data = [dataset[i] for i in range(4)]
    x = torch.stack([v[0] for v in data]).cuda()
    labels, ids = [v[1] for v in data], [v[3] for v in data]
    # Check JPEG source history and crop geometry against the existing GALP mother data.
    from galp.benchmarks.dct_models.evaluate_shards import native_options
    from galp.benchmarks.training_pls.published_augmentation import published_training_augmentation
    transforms = []
    for i, logical_id in enumerate(ids):
        d = published_training_augmentation(training_seed=11997733, epoch=0, logical_sample_id=logical_id,
            virtual_pls_id=i//1024, crop_policy='per-sample', source_width=512, source_height=512).decision
        transforms.append(dict(crop=[d.crop_x, d.crop_y, d.crop_width, d.crop_height],
                               horizontal_flip=d.horizontal_flip, logical_sample_id=logical_id))
    options = native_options(pushdown=False, projected=True)
    options['grid_transform'].update(crop_reference_width_blocks=64, crop_reference_height_blocks=64,
        crop_origin_alignment_blocks=2, chroma_crop_scale_x=2, chroma_crop_scale_y=2,
        allowed_chroma_sampling_ratios=[[1, 2, 1, 2]],
        output_channels=[[c, f, 0., 1.] for c, f in selected.input_channels])
    os.environ['GALP_BLOCK_MAJOR_ACCESS_DIR'] = str(args.manifest.parent/'block_major_access_v1')
    reader = native.DirectDctReader(str(args.manifest))
    batch = reader.read_batch(list(range(4)), transforms=transforms, **options)
    jpeg_raw = x[:, [c*64+f for c, f in selected.input_channels]]
    difference = (batch.projected-jpeg_raw).abs()
    # Independent FP32 summation can straddle a half-integer before rounding.
    if float(difference.max()) > 1.:
        raise AssertionError('JPEG/native source or DCT geometry differs beyond an integer rounding tie')
    raw_max, raw_differences = float(difference.max()), int((difference != 0).sum())
    # Input dependency order is distinct from final output order, including in the dense case.
    reference, rt = dense.apply(x[:, [c*64+f for c, f in dense.input_channels]].contiguous(),
                               ids, labels, 11997733, 0, 0)
    actual, at = selected.apply(x[:, [c*64+f for c, f in selected.input_channels]].contiguous(),
                               ids, labels, 11997733, 0, 0)
    expected = (reference[:, B.CHANNELS]-B.MEAN.cuda())/B.STD.cuda()
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
    torch.testing.assert_close(rt, at, atol=0, rtol=0)
    result = dict(device=torch.cuda.get_device_name(), profile=B.PROFILE, samples=4,
        max_abs=float((actual-expected).abs().max()), targets_exact=True, finite=bool(torch.isfinite(actual).all()),
        jpeg_native_raw_max_abs=raw_max, jpeg_native_integer_disagreements=raw_differences,
        scope='correctness only; shared RandAugment dense versus selected dependencies on the same real JPEG grids; no performance claim')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'projection_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
