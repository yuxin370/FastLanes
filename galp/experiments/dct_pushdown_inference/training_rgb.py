"""CNN wrapper around the existing equal-image PyTorch and DALI adapters."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import backend as B
from galp.benchmarks.system_rgbnomore.training.augmentation import derive_augmentation
from galp.benchmarks.system_rgbnomore.training.pipeline import (
    TrainingSample, PyTorchTrainingAdapter, DaliTrainingAdapter, resolve_dali_variant)
from galp.benchmarks.system_rgbnomore.training.sample_order import canonical_epoch_order, SampleIdentity
from galp.benchmarks.system_rgbnomore.training.equal_image_epoch_benchmark import _center_validation_decision


class RgbPool:
    def __init__(self, pipeline, offset):
        self.pipeline = pipeline
        self.pool_index = offset // pipeline.pool_images
        self.image_count = min(pipeline.pool_images, pipeline.sample_count-offset)
        self.execution_stats = {}

    def __iter__(self):
        p = self.pipeline
        for local in range(0, self.image_count, p.microbatch):
            batch = p.adapter.next_batch()
            x = batch.inputs[0].to('cuda', non_blocking=True)
            labels = batch.labels.to('cuda', non_blocking=True)
            for name, value in batch.stage_seconds.items():
                self.execution_stats[name] = self.execution_stats.get(name, 0.) + value
            n = len(batch.identities)
            yield SimpleNamespace(projected=x, targets=labels, image_count=n,
                global_image_ids=[p.positions[i.logical_sample_id] for i in batch.identities],
                microbatch_index_in_pool=local//p.microbatch, pool_offset=local,
                is_pool_end=local+n == self.image_count, keepalive=batch)


class RgbTrainingPipeline:
    def __init__(self, *, variant, workers, microbatch, pool_images, dali_prefetch_depth=2):
        source = json.loads((B.DEFAULT_E2E_V3_ROOT/'training_manifests_official_v3/train.json').read_text())
        self.samples = [TrainingSample(s['logical_sample_id'], Path(s['path']), s['label'], s['width'], s['height'],
                                      s['galp_image_id']) for s in source['samples']]
        self.positions = {s.logical_sample_id: i for i, s in enumerate(self.samples)}
        self.sample_count = len(self.samples)
        self.variant, self.workers, self.microbatch, self.pool_images = variant, workers, microbatch, pool_images
        self.dali_prefetch_depth = dali_prefetch_depth
        self.adapter = None

    def start_epoch(self, epoch):
        self.close()
        self.offset = 0
        identities = canonical_epoch_order([s.logical_sample_id for s in self.samples], 11997733, epoch)
        decisions = [] if self.variant == 'd3' else [derive_augmentation(seed=11997733, epoch=epoch,
            logical_sample_id=i.logical_sample_id, source_width=self.samples[self.positions[i.logical_sample_id]].width,
            source_height=self.samples[self.positions[i.logical_sample_id]].height, domain='rgb') for i in identities]
        cls = PyTorchTrainingAdapter if self.variant == 'pytorch' else DaliTrainingAdapter
        config = dict(phase='train')
        if self.variant != 'pytorch':
            config['dali'] = dict(resolve_dali_variant(self.variant), num_threads=self.workers,
                                 prefetch_queue_depth=self.dali_prefetch_depth)
        self.adapter = cls(self.samples, batch_size=self.microbatch, workers=self.workers,
                           device=torch.device('cuda'), config=config)
        self.adapter.begin(identities, decisions)

    @property
    def has_next_pool(self):
        return self.offset < self.sample_count

    def next_pool(self):
        pool = RgbPool(self, self.offset)
        self.offset += pool.image_count
        return pool

    @property
    def prefetch_stats(self):
        return self.adapter.loader_metrics()

    def reclaim_finished_pools(self):
        pass

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None


@torch.inference_mode()
def validate_rgb(net, count, workers):
    from evaluate import samples
    entries = samples(count)
    selected = [TrainingSample(s['logical_sample_id'], Path(s['path']), s['model_label'], s['width'], s['height'],
                               s['galp_image_id']) for s in entries]
    identities = [SampleIdentity(0, i, s.logical_sample_id) for i, s in enumerate(selected)]
    adapter = PyTorchTrainingAdapter(selected, batch_size=64, workers=workers, device=torch.device('cuda'), config={})
    net.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    adapter.begin(identities, [_center_validation_decision(s) for s in selected])
    total = top1 = top5 = 0
    ce = 0.
    try:
        while total < count:
            batch = adapter.next_batch()
            target = batch.labels.cuda(non_blocking=True)
            logits = net(batch.inputs[0].cuda(non_blocking=True))
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('non-finite RGB validation logits')
            rank = logits.topk(5, dim=1).indices
            top1 += int((rank[:, 0] == target).sum())
            top5 += int((rank == target[:, None]).any(1).sum())
            ce += float(torch.nn.functional.cross_entropy(logits, target, reduction='sum'))
            total += len(target)
        torch.cuda.synchronize()
        wall = time.perf_counter()-started
    finally:
        adapter.close()
    net.train()
    assert total == count
    return dict(samples=total, top1=100*top1/total, top5=100*top5/total, ce=ce/total, seconds=wall,
                precision='fp32', input='Transformer RGB validation contract: center square, resize 224, mean/std .5')
