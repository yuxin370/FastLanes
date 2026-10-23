"""Existing baseline readers under the restored-state, basic-augmentation protocol."""
import os
import json
from pathlib import Path

import torch

from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import CpuDctWindow, NativeWindow
from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingBatch, TrainingSample
from fastlens.benchmarks.baselines.system_rgbnomore.training.sample_order import SampleIdentity

ROOT = Path(__file__).resolve().parents[5]
SEED = 11997733


class BasicCpuDctWindow(CpuDctWindow):
    def begin(self):
        if self.model in ('vitti', 'swinv2'):
            from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import RgbNoMoreTrainingAdapter
            from fastlens.benchmarks.training_pls.published_augmentation import published_training_augmentation
            samples, identities, decisions = [], [], []
            for batch in self.batches:
                for i in batch:
                    row = self.records[i]
                    samples.append(TrainingSample(row['logical_sample_id'], Path(row['source_path']),
                                                  int(row['label']), 512, 512, i))
                    identities.append(SampleIdentity(2, len(identities), row['logical_sample_id']))
                    decisions.append(published_training_augmentation(training_seed=SEED, epoch=2,
                        logical_sample_id=row['logical_sample_id'], virtual_pls_id=int(row['virtual_pls_id']),
                        crop_policy='per-pls', source_width=512, source_height=512).decision)
            self.adapter = RgbNoMoreTrainingAdapter(samples, batch_size=64, workers=8,
                device=torch.device('cuda:0'), config={'rgbnomore_root': str(Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more')))})
            self.adapter.begin(identities, decisions, [len(b) for b in self.batches])
        else:
            from fastlens.benchmarks.dct_models import backend as B
            from fastlens.benchmarks.dct_models.training_pls import channels
            from fastlens.benchmarks.dct_models.training_jpeg import JpegTrainingPipeline

            class BasicAugmentation:
                input_channels = [(int(c), int(f)) for c, f, _, _ in channels()]

                def apply(self, x, logical_ids, labels, seed, epoch, microbatch):
                    mean = B.MEAN.to(x.device).reshape(1, -1, 1, 1)
                    std = B.STD.to(x.device).reshape(1, -1, 1, 1)
                    return ((x.round().clamp(-1024, 1016) - mean) / std,
                            torch.tensor(labels, device=x.device, dtype=torch.int64))

            self.pipeline = JpegTrainingPipeline(self.mapping, output_channels=channels(), grid=B.GRID,
                workers=8, microbatch=64, pool_images=4096, condition='B6')
            self.pipeline.augment = BasicAugmentation()
            self.pipeline.start_epoch(2)
            self.pool = iter(self.pipeline.next_pool())

    def next_batch(self):
        while True:
            if self.model in ('vitti', 'swinv2'):
                batch = self.adapter.next_batch()
                identities = batch.identities
                inputs = tuple(((x.cuda(non_blocking=True)*1020.-4.).round().clamp(-1024, 1016)+4.)/1020.
                               for x in batch.inputs)
                labels = batch.labels.cuda(non_blocking=True)
            else:
                batch = next(self.pool)
                identities = [SampleIdentity(2, i, self.records[i]['logical_sample_id'])
                              for i in batch.global_image_ids]
                inputs, labels = (batch.projected,), batch.targets
                if batch.is_pool_end:
                    self.pool = iter(self.pipeline.next_pool())
            if len(identities) < 64:
                continue
            return TrainingBatch(inputs=inputs, labels=labels, identities=identities,
                                 augmentations=[], on_device=True, stage_seconds={})

    def close(self):
        if self.model in ('vitti', 'swinv2'):
            self.adapter.close()
        else:
            self.pipeline.close()


def training_inputs(model, path, device, output, *, l3_root=None, l3_library=None):
    """Return a restartable source factory and its actual source files."""
    if path == 'cpu_dct':
        def make():
            source = BasicCpuDctWindow(model, 65536)
            source.begin()
            return source
        plan = BasicCpuDctWindow(model, 65536)
        return make, plan.paths, [], [], {}
    from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import CoorDLWindow, PlannedRgbWindow
    from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import DaliTrainingAdapter, resolve_dali_variant
    from fastlens.benchmarks.baselines.system_rgbnomore.training.augmentation import derive_augmentation
    entries = json.loads((ROOT/'results/preparation/p0_ffcv/samples.json').read_text())
    samples = [TrainingSample(e['logical_sample_id'], Path(e['path']), e['label'],
                             e['width'], e['height'], e['galp_image_id']) for e in entries]
    decisions = [derive_augmentation(seed=SEED, epoch=2, logical_sample_id=s.logical_sample_id,
                 source_width=s.width, source_height=s.height, domain='rgb') for s in samples]
    identities = [SampleIdentity(2, i, s.logical_sample_id) for i, s in enumerate(samples)]
    variant = dict(resolve_dali_variant('d2'), num_threads=16, prefetch_queue_depth=2,
                   seed=SEED, resize_antialias=False)
    config = {'phase': 'train', 'dali': variant}
    paths = [s.path for s in samples]
    if path == 'l3':
        from fastlens.benchmarks.baselines.l3.prepare import encoded_path
        config['l3'] = dict(root=str(l3_root), library=str(l3_library))
        paths = [encoded_path(config['l3']['root'], s.logical_sample_id) for s in samples]

    def make():
        if path == 'coordl':
            return CoorDLWindow(samples, device, output, decisions)
        cls = PlannedRgbWindow if path == 'pytorch' else DaliTrainingAdapter
        source = cls(samples, batch_size=64, workers=16, device=device, config=config)
        source.begin(identities, decisions)
        return source
    return make, paths, samples, decisions, variant


def check_cpu_input(model, batch, output):
    native = NativeWindow(model, 'basic')
    native.begin()
    try:
        reference = native.next_batch()
        assert [i.logical_sample_id for i in batch.identities] == [i.logical_sample_id for i in reference.identities]
        torch.testing.assert_close(batch.labels, reference.labels, rtol=0, atol=0)
        atol, rtol = (1/1020+1e-7, 1e-4) if model in ('vitti', 'swinv2') else (1e-5, 1e-5)
        differences = []
        for actual, expected in zip(batch.inputs, reference.inputs, strict=True):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            differences.append(float((actual-expected).abs().max()))
        (output/'cpu_input_check.json').write_text(json.dumps(dict(passed=True, max_abs=differences,
            atol=atol, rtol=rtol, augmentation='basic', epoch=2, samples=64))+'\n')
        del reference
    finally:
        native.close()
