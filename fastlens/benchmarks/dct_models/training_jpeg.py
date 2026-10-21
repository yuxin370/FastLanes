"""Global-shuffle JPEG/PyTorch reference for the native DCT training contract.

Workers read the existing 512px JPEGs once and construct unnormalized DCT grids.
The shared GPU postprocessor supplies exactly the same keyed RandAugment and
Mixup as PLS. This adapter never reads or emulates a FASTLENS shard.
"""
import csv
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import fastlens.benchmarks.dct_models.backend as B
from fastlens.benchmarks.dct_models.dct_geometry import load_upsample_dct
from fastlens.benchmarks.training_pls.core_schedule import stable_seed, epoch_position_pools
from fastlens.benchmarks.training_pls.published_augmentation import published_training_augmentation


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

    def read_components(self, logical_id, path):
        from jpeg2dct.numpy import loads
        t = time.perf_counter()
        encoded = Path(path).read_bytes()
        read = time.perf_counter() - t
        t = time.perf_counter()
        components = loads(encoded, normalized=True)  # Already dequantized, natural frequency order.
        decode = time.perf_counter() - t
        return components, read, decode, len(encoded)

    def __getitem__(self, index):
        logical_id, _, path = self.records[self.order[index]]
        components, read, decode, size = self.read_components(logical_id, path)
        return self.transform(index, components, read, decode, size)

    def decision(self, index):
        physical_id = self.order[index]
        return published_training_augmentation(training_seed=11997733, epoch=self.epoch,
            logical_sample_id=self.records[physical_id][0], virtual_pls_id=physical_id // 1024, crop_policy=self.crop_policy,
            source_width=512, source_height=512).decision

    def transform(self, index, components, read, decode, size, *, cropped=False):
        if self.resize is None:
            self.resize = load_upsample_dct(B.DEFAULT_RGBNOMORE_ROOT, resize=True)
        physical_id = self.order[index]
        logical_id, label, path = self.records[physical_id]
        decision = self.decision(index)
        expected = [(decision.crop_height//s, decision.crop_width//s, 64) for s in (8, 16, 16)] if cropped else [(64,64,64), (32,32,64), (32,32,64)]
        if [v.shape for v in components] != expected:
            raise ValueError(f'Unexpected source DCT geometry: {logical_id}')
        t = time.perf_counter()
        output = torch.empty(len(self.input_channels), self.grid, self.grid)
        for c, array in enumerate(components):
            scale = 8 if c == 0 else 16
            y, x = (0, 0) if cropped else (decision.crop_y // scale, decision.crop_x // scale)
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
        return output, label, physical_id, logical_id, read, decode, time.perf_counter()-t, size


class StoreDctDataset(JpegDctDataset):
    """Change only coefficient acquisition; preserve the published JPEG training adapter."""
    def __init__(self, *args, store, **kwargs):
        super().__init__(*args, **kwargs)
        self.store, self.reader = Path(store), None

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        from fastlens.benchmarks.baselines.dct_baselines.contracts import request_for_row
        from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader
        t = time.perf_counter()
        first_read = self.reader is None
        if first_read:
            # One existing bounded reader per DataLoader worker. No decoded cache.
            self.reader = StoreReader(self.store)
        requests = []
        for index in indices:
            decision = self.decision(index)
            row = dict(image_id=self.records[self.order[index]][0],
                       crop_box=[decision.crop_x, decision.crop_y, decision.crop_width, decision.crop_height])
            requests.append(request_for_row(row, dict(source_frequencies=[list(range(64))]*3)))
        views, counters = self.reader.read(self.reader.plan(requests))
        if first_read:
            counters.add(self.reader.initialization_counters)
        decode = time.perf_counter()-t
        result = []
        from dataclasses import asdict
        for position, (index, view) in enumerate(zip(indices, views)):
            components = [(q.astype(np.int32)*table.astype(np.int32)).reshape(*q.shape[:2], 64)
                          for q, table in zip(view.dense(), view.tables)]
            item = self.transform(index, components, 0., decode if position == 0 else 0.,
                                  counters.application_read_bytes if position == 0 else 0, cropped=True)
            # Counts belong to the merged read, not to every member image.
            observed = asdict(counters) if position == 0 else {key: 0 for key in asdict(counters)}
            result.append((*item, observed))
        return result


class JpegPool:
    def __init__(self, pipeline, offset, pool_index):
        self.pipeline, self.offset = pipeline, offset
        self.pool_index = pool_index
        self.image_count = pipeline.pool_lengths[pool_index]
        self.execution_stats = (dict(read_seconds=0., decode_seconds=0., requested_jpeg_bytes=0) if pipeline.store is None else
                                dict(read_decode_seconds=0., application_read_bytes=0))
        self.execution_stats.update(transform_worker_seconds=0., h2d_augmentation_seconds=0.)

    def __iter__(self):
        p = self.pipeline
        for local in range(0, self.image_count, p.microbatch):
            values = next(p.iterator)
            x, labels, ids, logical_ids, read, decode, transform, sizes = values[:8]
            if p.store is not None:
                for key, value in values[8].items():
                    total = int(value.max()) if key.endswith('peak_bytes') else int(value.sum())
                    self.execution_stats[key] = max(self.execution_stats.get(key, 0), total) if key.endswith('peak_bytes') else self.execution_stats.get(key, 0)+total
            t = time.perf_counter()
            with torch.cuda.nvtx.range('input.jpeg_h2d_shared_augmentation'):
                x = x.cuda(non_blocking=True)
                x, targets = p.augment.apply(x, list(logical_ids), labels.tolist(), 11997733,
                                             p.epoch, p.microbatch_offsets[self.pool_index]+local//p.microbatch)
            self.execution_stats['h2d_augmentation_seconds'] += time.perf_counter()-t
            if p.store is None:
                self.execution_stats['read_seconds'] += float(read.sum())
                self.execution_stats['decode_seconds'] += float(decode.sum())
                self.execution_stats['requested_jpeg_bytes'] += int(sizes.sum())
            else:
                self.execution_stats['read_decode_seconds'] += float(read.sum()+decode.sum())
            self.execution_stats['transform_worker_seconds'] += float(transform.sum())
            n = len(ids)
            yield SimpleNamespace(global_image_ids=ids.tolist(), projected=x, targets=targets,
                image_count=n, microbatch_index_in_pool=local//p.microbatch, pool_offset=local,
                is_pool_end=local+n == self.image_count)


class JpegTrainingPipeline:
    def __init__(self, mapping, *, output_channels, grid, workers, microbatch, pool_images, condition='A0', store=None):
        import _fastlens_direct_dct as native
        self.records = []
        with mapping.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if int(row['planned_physical_position']) != len(self.records):
                    raise ValueError('Premixed mapping is not in physical position order')
                self.records.append((row['logical_sample_id'], int(row['label']), row['source_path']))
        self.sample_count = len(self.records)
        self.store = Path(store) if store is not None else None
        self.store_identity = None
        self.store_content_verification = None
        if self.store is not None:
            from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, training_store_identity
            self.store_identity = training_store_identity(self.store, self.records)
            with StoreReader(self.store) as reader:
                self.store_content_verification = reader.verify_content()
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
        dataset_type = JpegDctDataset if self.store is None else StoreDctDataset
        kwargs = {} if self.store is None else dict(store=self.store)
        dataset = dataset_type(self.records, order, epoch, self.augment.input_channels, self.grid,
            crop_policy='per-pls' if self.condition=='B6' else 'per-sample', **kwargs)
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
        return dict(backend='JPEG/PyTorch' if self.store is None else self.store_identity['method'],
                    workers=self.workers, prefetch_batches_per_worker=2)

    def reclaim_finished_pools(self):
        pass

    def close(self):
        self.iterator = None
