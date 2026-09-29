"""Global-shuffle JPEG/PyTorch reference for the native DCT training contract.

Workers read the existing 512px JPEGs once and construct unnormalized DCT grids.
The shared GPU postprocessor supplies exactly the same keyed RandAugment and
Mixup as PLS. This adapter never reads or emulates a GALP shard.
"""
import csv
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import galp.benchmarks.dct_models.backend as B
from galp.benchmarks.dct_models.dct_geometry import load_upsample_dct
from galp.benchmarks.training_pls.core_schedule import stable_seed, epoch_position_pools
from galp.benchmarks.training_pls.published_augmentation import published_training_augmentation


def worker_init(_):
    torch.set_num_threads(1)


class JpegDctDataset(Dataset):
    def __init__(self, records, order, epoch, input_channels, grid, crop_policy='per-sample'):
        self.records, self.order, self.epoch = records, order, epoch
        self.input_channels, self.grid = input_channels, grid
        self.resize = None
        self.matrices = {}
        self.crop_policy = crop_policy

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        from jpeg2dct.numpy import loads
        if self.resize is None:
            self.resize = load_upsample_dct(B.DEFAULT_RGBNOMORE_ROOT, resize=True)
        physical_id = self.order[index]
        logical_id, label, path = self.records[physical_id]
        t = time.perf_counter()
        encoded = Path(path).read_bytes()
        read = time.perf_counter() - t
        t = time.perf_counter()
        components = loads(encoded, normalized=True)  # Already dequantized, natural frequency order.
        decode = time.perf_counter() - t
        if [v.shape for v in components] != [(64, 64, 64), (32, 32, 64), (32, 32, 64)]:
            raise ValueError(f'Unexpected source DCT geometry: {logical_id}')
        t = time.perf_counter()
        decision = published_training_augmentation(training_seed=11997733, epoch=self.epoch,
            logical_sample_id=logical_id, virtual_pls_id=physical_id // 1024, crop_policy=self.crop_policy,
            source_width=512, source_height=512).decision
        output = torch.empty(len(self.input_channels), self.grid, self.grid)
        for c, array in enumerate(components):
            scale = 8 if c == 0 else 16
            y, x = decision.crop_y // scale, decision.crop_x // scale
            h, w = decision.crop_height // scale, decision.crop_width // scale
            coeff = torch.from_numpy(array[y:y+h, x:x+w].copy()).float().reshape(1, h, w, 8, 8)
            value = self.resize(coeff, self.grid, dtype_out=torch.float32, conv_mxs=self.matrices)
            value = value.reshape(self.grid, self.grid, 64).round_().clamp_(-32768, 32767)
            positions = [i for i, channel in enumerate(self.input_channels) if channel[0] == c]
            frequencies = [self.input_channels[i][1] for i in positions]
            selected = value[:, :, frequencies].permute(2, 0, 1)
            if decision.horizontal_flip:
                sign = torch.tensor([(-1.) ** (f % 8) for f in frequencies])[:, None, None]
                selected = selected.flip(2) * sign
            output[positions] = selected
        return output, label, physical_id, logical_id, read, decode, time.perf_counter()-t, len(encoded)


class JpegPool:
    def __init__(self, pipeline, offset, pool_index):
        self.pipeline, self.offset = pipeline, offset
        self.pool_index = pool_index
        self.image_count = pipeline.pool_lengths[pool_index]
        self.execution_stats = dict(read_seconds=0., decode_seconds=0., transform_worker_seconds=0.,
                                    requested_jpeg_bytes=0, h2d_augmentation_seconds=0.)

    def __iter__(self):
        p = self.pipeline
        for local in range(0, self.image_count, p.microbatch):
            values = next(p.iterator)
            x, labels, ids, logical_ids, read, decode, transform, sizes = values
            t = time.perf_counter()
            with torch.cuda.nvtx.range('input.jpeg_h2d_shared_augmentation'):
                x = x.cuda(non_blocking=True)
                x, targets = p.augment.apply(x, list(logical_ids), labels.tolist(), 11997733,
                                             p.epoch, p.microbatch_offsets[self.pool_index]+local//p.microbatch)
            self.execution_stats['h2d_augmentation_seconds'] += time.perf_counter()-t
            self.execution_stats['read_seconds'] += float(read.sum())
            self.execution_stats['decode_seconds'] += float(decode.sum())
            self.execution_stats['transform_worker_seconds'] += float(transform.sum())
            self.execution_stats['requested_jpeg_bytes'] += int(sizes.sum())
            n = len(ids)
            yield SimpleNamespace(global_image_ids=ids.tolist(), projected=x, targets=targets,
                image_count=n, microbatch_index_in_pool=local//p.microbatch, pool_offset=local,
                is_pool_end=local+n == self.image_count)


class JpegTrainingPipeline:
    def __init__(self, mapping, *, output_channels, grid, workers, microbatch, pool_images, condition='A0'):
        import _galp_direct_dct as native
        self.records = []
        with mapping.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if int(row['planned_physical_position']) != len(self.records):
                    raise ValueError('Premixed mapping is not in physical position order')
                self.records.append((row['logical_sample_id'], int(row['label']), row['source_path']))
        self.sample_count = len(self.records)
        self.augment = native.ProjectedDctTrainingAugment(output_channels, 1000)
        self.grid, self.workers, self.microbatch, self.pool_images = grid, workers, microbatch, pool_images
        self.iterator = None
        self.condition = condition
        if condition not in ('A0', 'B6') or (condition=='B6' and pool_images!=4096):
            raise ValueError('JPEG reference supports A0 or registered B6 pools of four segments')

    def start_epoch(self, epoch):
        self.close()
        self.epoch, self.offset, self.pool_index = epoch, 0, 0
        if self.condition=='B6':
            layout=SimpleNamespace(sample_count=self.sample_count, positions_by_pls=[
                np.arange(i,min(i+1024,self.sample_count)) for i in range(0,self.sample_count,1024)])
            pools=[ids for _,_,ids in epoch_position_pools(layout,condition_id='B6',seed=11997733,epoch=epoch)]
            order=[i for pool in pools for i in pool]
            self.pool_lengths=[len(pool) for pool in pools]
        else:
            order=list(range(self.sample_count))
            random.Random(stable_seed('global-sample-order',11997733,epoch)).shuffle(order)
            self.pool_lengths=[min(self.pool_images,self.sample_count-i) for i in range(0,self.sample_count,self.pool_images)]
        batches=[]
        self.microbatch_offsets=[]
        offset=0
        for length in self.pool_lengths:
            self.microbatch_offsets.append(len(batches))
            batches.extend(list(range(i,min(i+self.microbatch,offset+length))) for i in range(offset,offset+length,self.microbatch))
            offset+=length
        dataset = JpegDctDataset(self.records, order, epoch, self.augment.input_channels, self.grid,
            crop_policy='per-pls' if self.condition=='B6' else 'per-sample')
        self.loader = DataLoader(dataset, batch_sampler=batches, num_workers=self.workers,
            pin_memory=True, prefetch_factor=2 if self.workers else None, worker_init_fn=worker_init)
        self.iterator = iter(self.loader)

    @property
    def has_next_pool(self):
        return self.offset < self.sample_count

    def next_pool(self):
        pool = JpegPool(self, self.offset, self.pool_index)
        self.offset += pool.image_count
        self.pool_index += 1
        return pool

    @property
    def prefetch_stats(self):
        return dict(backend='JPEG/PyTorch', workers=self.workers, prefetch_batches_per_worker=2)

    def reclaim_finished_pools(self):
        pass

    def close(self):
        self.iterator = None
