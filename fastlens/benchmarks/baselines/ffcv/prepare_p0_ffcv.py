#!/usr/bin/env python3
"""Convert only the existing RGB window's 98,560 training JPEGs, losslessly, to raw FFCV."""
import json
import argparse
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'results/preparation/p0_ffcv'


class WindowDataset:
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        from PIL import Image
        import numpy as np
        entry = self.entries[index]
        with Image.open(entry['path']) as image:
            return np.asarray(image.convert('RGB')), int(entry['label']), index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--full-train',action='store_true')
    args=parser.parse_args()
    out=args.output
    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField
    out.mkdir(parents=True,exist_ok=True)
    if (out/'conversion.json').exists():
        print('Reusing completed conversion', flush=True)
        return
    if (out/'train_window.beton').exists():
        raise FileExistsError('Incomplete conversion exists; preserve and inspect it before another attempt')
    source = ROOT/'fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json'
    population = json.loads(source.read_text())['samples']
    entries = random.Random(11997733).sample(population,len(population) if args.full_train else 98560)
    del population
    jpeg_bytes=sum(Path(entry['path']).stat().st_size for entry in entries)
    (out/'samples.json').write_text(json.dumps(entries)+'\n')
    started = time.perf_counter()
    DatasetWriter(str(out/'train_window.beton'), {'image':RGBImageField(write_mode='raw'),
                  'label':IntField(),'ordinal':IntField()},num_workers=args.workers).from_indexed_dataset(WindowDataset(entries))
    seconds = time.perf_counter()-started
    record = dict(source=str(source), samples=len(entries), conversion_s=seconds,
                  jpeg_bytes=jpeg_bytes, dataset_scope='Complete ImageNet-512 training dataset' if args.full_train else '98560-image P0 training selection; not the complete training dataset',
                  workers=args.workers,cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),
                  representation_bytes=(out/'train_window.beton').stat().st_size,
                  metadata_bytes=(out/'samples.json').stat().st_size,
                  format='FFCV raw RGB; decoded original ImageNet-512 JPEG pixels; no new JPEG quantization')
    (out/'conversion.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
