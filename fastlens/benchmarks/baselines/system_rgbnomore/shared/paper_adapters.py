"""Existing FastLens data plane and FFCV/CoorDL inputs for the bounded P0 runner."""
import csv
import json
import os
from pathlib import Path
import time
import shutil
import tempfile

import torch
from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingBatch
from fastlens.benchmarks.baselines.system_rgbnomore.training.sample_order import SampleIdentity

ROOT = Path(__file__).resolve().parents[5]


class ResidentWindow:
    """Same model tensor contract and target representation, without input work."""
    def __init__(self,model,domain,device):
        labels=torch.arange(64,device=device)%1000
        if domain=='rgb':
            inputs=(torch.zeros(64,3,224,224,device=device),)
        else:
            labels=torch.nn.functional.one_hot(labels,1000).float()
            if model in ('swinv2','vitti'):
                inputs=(torch.zeros(64,1,28,28,8,8,device=device),torch.zeros(64,2,14,14,8,8,device=device))
            else:
                from fastlens.benchmarks.dct_models import backend as B
                inputs=(torch.zeros(64,sum(map(len,B.INDICES)),B.GRID,B.GRID,device=device),)
        self.batch=TrainingBatch(inputs=inputs,labels=labels,identities=[],augmentations=[],on_device=True,stage_seconds={})

    def next_batch(self):
        return self.batch

    def close(self):
        pass


class FfcvWindow:
    def __init__(self, samples, device, decisions, root=None):
        from ffcv.loader import Loader, OrderOption
        from ffcv.fields.decoders import IntDecoder
        from fastlens.benchmarks.baselines.system_rgbnomore.training.ffcv_crop import PlannedCropDecoder
        from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, RandomHorizontalFlip
        self.samples = samples
        root = Path(root) if root is not None else ROOT/'results/preparation/p0_ffcv'
        stored = json.loads((root/'samples.json').read_text())
        if [s.logical_sample_id for s in samples] != [s['logical_sample_id'] for s in stored]:
            raise ValueError('FFCV converted sample sequence differs from the RGB window')
        self.loader = Loader(str(root/'train_window.beton'),batch_size=64,num_workers=16,
            order=OrderOption.SEQUENTIAL,drop_last=False,seed=11997733,
            pipelines={'image':[PlannedCropDecoder(decisions),ToTensor(),ToDevice(device),ToTorchImage()],
                       'label':[IntDecoder(),ToTensor(),ToDevice(device)],'ordinal':[IntDecoder(),ToTensor()]})
        self.iterator = iter(self.loader)

    def next_batch(self):
        images, labels, ordinals = next(self.iterator)
        ids = ordinals.reshape(-1).tolist()
        return TrainingBatch(inputs=(images.float().div_(127.5).sub_(1),),labels=labels.reshape(-1).long(),
            identities=[SampleIdentity(2,i,self.samples[i].logical_sample_id) for i in ids],
            augmentations=[],on_device=True,stage_seconds={})

    def close(self):
        self.iterator.terminate_event.set()
        self.iterator.join()
        torch.cuda.synchronize()
        self.iterator.close()
        self.iterator = None


def cleanup_coordl_cache(output):
    """Remove only the private namespace created by this attempt, also after interruption."""
    receipt = Path(output)/'coordl_cache_namespace.json'
    if not receipt.exists():
        return
    owner = json.loads(receipt.read_text())
    namespace = Path(owner['namespace'])
    if namespace.parent != Path.home()/'tmp' or not namespace.name.startswith('fastlens-coordl-'):
        raise ValueError(f'Not an owned CoorDL namespace: {namespace}')
    cached = Path('/dev/shm/cache')/namespace.relative_to('/')
    if cached.exists() or namespace.exists():
        import psutil
        if owner['owner_pid'] != os.getpid() and psutil.pid_exists(owner['owner_pid']):
            raise RuntimeError('CoorDL cache owner is still alive; cache retained')
    if cached.exists():
        shutil.rmtree(cached)
    if namespace.exists():
        (namespace/'source').unlink()
        namespace.rmdir()


class CoorDLWindow:
    def __init__(self,samples,device,output,decisions):
        from nvidia.dali import backend,ops,types
        from nvidia.dali.pipeline import Pipeline
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        if 'cache_size' not in backend.GetSchema('FileReader').GetArgumentNames():
            raise RuntimeError('CoorDL must run through the existing coordl-python environment')
        self.samples, self.device, self.output = samples, device, output
        output.mkdir(parents=True, exist_ok=True)
        source_root = Path(os.path.commonpath([str(s.path.resolve().parent) for s in samples]))
        namespace = Path(tempfile.mkdtemp(prefix='fastlens-coordl-', dir=Path.home()/'tmp'))
        (namespace/'source').symlink_to(source_root, target_is_directory=True)
        (output/'coordl_cache_namespace.json').write_text(json.dumps(dict(
            namespace=str(namespace), source_root=str(source_root), owner_pid=os.getpid()))+'\n')
        self.cache_paths = []
        lines = []
        for i,sample in enumerate(samples):
            path = str((namespace/'source'/sample.path.resolve().relative_to(source_root)).relative_to('/'))
            for suffix in ('','-tmp'):
                cache = Path('/dev/shm/cache')/(path+suffix)
                if cache.exists():
                    raise FileExistsError(f'Existing shared CoorDL cache is not owned by this run: {cache}')
                self.cache_paths.append(cache)
            lines.append(f'{path} {i}\n')
        file_list=output/'coordl_files.txt'
        file_list.parent.mkdir(parents=True,exist_ok=True)
        file_list.write_text(''.join(lines))

        import numpy as np
        anchors = [np.array([d.crop_y,d.crop_x],dtype=np.float32) for d in decisions]
        shapes = [np.array([d.crop_height,d.crop_width],dtype=np.float32) for d in decisions]
        mirrors = [np.array(int(d.horizontal_flip),dtype=np.int32) for d in decisions]

        class InputPipeline(Pipeline):
            def __init__(self):
                super().__init__(batch_size=64,num_threads=16,device_id=0,seed=11997733,prefetch_queue_depth=2)
                self.reader=ops.FileReader(file_root='/',file_list=str(file_list),cache_size=len(samples)+1,
                    num_nodes=1,node_id=0,num_shards=1,shard_id=0,random_shuffle=False,
                    shuffle_after_epoch=False,pad_last_batch=True)
                self.decode=ops.ImageDecoder(device='mixed',output_type=types.RGB)
                self.crop=ops.Slice(device='gpu',axes=[0,1],normalized_anchor=False,normalized_shape=False)
                # Old CoorDL's LINEAR is non-antialiased for both downscaling and upscaling.
                self.resize=ops.Resize(device='gpu',resize_x=224,resize_y=224,interp_type=types.INTERP_LINEAR)
                self.anchor=ops.ExternalSource()
                self.shape=ops.ExternalSource()
                self.flip=ops.ExternalSource()
                self.cursor=0
                self.normalize=ops.CropMirrorNormalize(device='gpu',output_dtype=types.FLOAT,output_layout='CHW',
                    mean=[127.5]*3,std=[127.5]*3)
            def define_graph(self):
                encoded,ordinal=self.reader(name='Reader')
                self.anchor_node,self.shape_node,self.flip_node=self.anchor(),self.shape(),self.flip()
                return self.normalize(self.resize(self.crop(self.decode(encoded),self.anchor_node,self.shape_node)),mirror=self.flip_node),ordinal
            def iter_setup(self):
                ids=[min(i,len(samples)-1) for i in range(self.cursor,self.cursor+64)]
                self.feed_input(self.anchor_node,[anchors[i] for i in ids])
                self.feed_input(self.shape_node,[shapes[i] for i in ids])
                self.feed_input(self.flip_node,[mirrors[i] for i in ids])
                self.cursor=(self.cursor+64)%((len(samples)+63)//64*64)
        cache_started=time.perf_counter()
        self.iterator=DALIGenericIterator([InputPipeline()],output_map=['image','ordinal'],size=len(samples),
            auto_reset=False,fill_last_batch=False,last_batch_padded=True)
        # Warm CoorDL's own JPEG cache using only this bounded input set, without model updates.
        for _ in self.iterator:
            pass
        cache_seconds=time.perf_counter()-cache_started
        files=[path.stat() for path in self.cache_paths if path.exists()]
        self.cache_stats=dict(samples=len(samples),files=len(files),representation_bytes=sum(s.st_size for s in files),
            allocated_bytes=sum(s.st_blocks*512 for s in files),source_jpeg_bytes=sum(s.path.stat().st_size for s in samples),
            cache_population_s=cache_seconds,filesystem='tmpfs',
            scope='Own cache files after pipeline construction and excluded complete bounded cache-fill pass; tmpfs backing is separate from process RSS')
        self.iterator.reset()

    def next_batch(self):
        values=next(self.iterator)[0]
        ids=values['ordinal'].reshape(-1).cpu().tolist()
        return TrainingBatch(inputs=(values['image'],),labels=torch.tensor([self.samples[i].label for i in ids],device=self.device),
            identities=[SampleIdentity(2,i,self.samples[i].logical_sample_id) for i in ids],
            augmentations=[],on_device=True,stage_seconds={})

    def close(self):
        self.iterator=None
        cleanup_coordl_cache(self.output)


class CpuDctWindow:
    """Existing JPEG/DCT training adapters in the same B6 window as NativeWindow."""
    def __init__(self, model, image_count):
        import numpy as np
        from types import SimpleNamespace
        from fastlens.benchmarks.training_pls.core_schedule import epoch_position_pools
        root = ROOT/'fastlens/data/compressed/imagenet512_train_block_major_premixed'
        self.mapping = root/'ordered_mapping.csv'
        with self.mapping.open() as stream:
            self.records = list(csv.DictReader(stream))
        layout = SimpleNamespace(sample_count=len(self.records), positions_by_pls=[
            np.arange(i, min(i+1024, len(self.records))) for i in range(0, len(self.records), 1024)])
        self.batches = []
        for _, _, positions in epoch_position_pools(layout, condition_id='B6', seed=11997733, epoch=2):
            self.batches.extend(positions[i:i+64] for i in range(0, len(positions), 64))
            if sum(len(batch) == 64 for batch in self.batches)*64 >= image_count+4096:
                break
        self.paths = [Path(self.records[i]['source_path']) for batch in self.batches for i in batch]
        self.model, self.microbatch = model, 0
        self.skipped_partial_ids = []

    def warm(self):
        from fastlens.benchmarks.profiling.input_motivation import cache_pages
        return cache_pages(self.paths, warm=True)

    def begin(self):
        if self.model == 'vitti':
            from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingSample, RgbNoMoreTrainingAdapter
            from fastlens.benchmarks.training_pls.published_augmentation import published_training_augmentation
            samples, identities, decisions = [], [], []
            for batch in self.batches:
                for i in batch:
                    row = self.records[i]
                    logical_id = row['logical_sample_id']
                    samples.append(TrainingSample(logical_id, Path(row['source_path']), int(row['label']), 512, 512, i))
                    identities.append(SampleIdentity(2, len(identities), logical_id))
                    decisions.append(published_training_augmentation(training_seed=11997733, epoch=2,
                        logical_sample_id=logical_id, virtual_pls_id=i//1024, crop_policy='per-pls',
                        source_width=512, source_height=512).decision)
            self.adapter = RgbNoMoreTrainingAdapter(samples, batch_size=64, workers=8,
                device=torch.device('cuda:0'), config={'phase': 'train', 'rgbnomore_root': str(Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more')))})
            self.adapter.begin(identities, decisions, [len(batch) for batch in self.batches])
        else:
            from fastlens.benchmarks.dct_models import backend as B
            from fastlens.benchmarks.dct_models.training_jpeg import JpegTrainingPipeline
            pairs = [(component, f) for component, fs in enumerate(B.INDICES) for f in fs]
            channels = [[c, f, mean, std] for (c, f), mean, std in
                        zip(pairs, B.MEAN.flatten().tolist(), B.STD.flatten().tolist())]
            self.pipeline = JpegTrainingPipeline(self.mapping, output_channels=channels, grid=B.GRID,
                workers=8, microbatch=64, pool_images=4096, condition='B6')
            self.pipeline.start_epoch(2)
            self.pool = iter(self.pipeline.next_pool())

    def next_batch(self):
        from fastlens.benchmarks.training_pls.published_augmentation import apply_published_randaugment, apply_published_mixup
        while True:
            if self.model == 'vitti':
                batch = self.adapter.next_batch()
                identities = batch.identities
                inputs = tuple(value.cuda(non_blocking=True) for value in batch.inputs)
                labels = batch.labels.cuda(non_blocking=True)
                inputs, _ = apply_published_randaugment(inputs, training_seed=11997733, epoch=2,
                    logical_sample_ids=[item.logical_sample_id for item in identities],
                    rgbnomore_root=(Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more'))))
                inputs, targets, _ = apply_published_mixup(inputs, labels, training_seed=11997733,
                    epoch=2, microbatch_index=self.microbatch)
            else:
                batch = next(self.pool)
                identities = [SampleIdentity(2, i, self.records[i]['logical_sample_id']) for i in batch.global_image_ids]
                inputs, targets = (batch.projected,), batch.targets
                if batch.is_pool_end:
                    self.pool = iter(self.pipeline.next_pool())
            self.microbatch += 1
            if len(identities) < 64:
                self.skipped_partial_ids.extend(item.logical_sample_id for item in identities)
                continue
            return TrainingBatch(inputs=inputs, labels=targets, identities=identities,
                augmentations=[], on_device=True, stage_seconds={})

    def close(self):
        if self.model == 'vitti':
            self.adapter.close()
        else:
            self.pipeline.close()


class NativeWindow:
    def __init__(self,model,augmentation="full"):
        from fastlens.torch.experimental import DirectDctPlsPipeline
        root=ROOT/'fastlens/data/compressed/imagenet512_train_block_major_premixed'
        contract=json.loads((root/'materialization_contract.json').read_text())
        self.manifest=root/'dct/manifest.bin'
        os.environ['FASTLENS_BLOCK_MAJOR_ACCESS_DIR']=str(self.manifest.parent/'block_major_access_v1')
        with (root/'ordered_mapping.csv').open() as stream:
            self.records=list(csv.DictReader(stream))
        options={}
        if model not in ('swinv2','vitti'):
            from fastlens.benchmarks.dct_models import backend as B
            pairs=[(component,f) for component,fs in enumerate(B.INDICES) for f in fs]
            channels=[[c,f,mean,std] for (c,f),mean,std in zip(pairs,B.MEAN.flatten().tolist(),B.STD.flatten().tolist())]
            options=dict(output_grid_size=B.GRID,output_channels=channels,transform_blocks_per_launch=32768)
        self.pipeline=DirectDctPlsPipeline(self.manifest,root/'ordered_mapping.csv',training_seed=11997733,
            expected_mapping_sha256=contract['ordered_mapping_sha256'],crop_policy='per-pls',order_policy='closed-pool',
            segments_per_pool=4,microbatch_images=64,
            enable_randaugment=augmentation=="full",enable_mixup=augmentation=="full",
            module_path=Path(os.environ.get('FASTLENS_TRAINING_NATIVE_DIR',ROOT/'build/fastlens/torch')),**options)
        self.model=model
        self.pool=None
        self.previous_end=False
        self.skipped_partial_ids=[]

    def warm(self):
        from fastlens.benchmarks.profiling.input_motivation import cache_pages
        from fastlens.benchmarks.system_dct_major.common import parse_manifest
        manifest=parse_manifest(self.manifest)
        self.paths=[Path(s['fls_path']) for s in manifest['shards']]
        return cache_pages(self.paths,warm=True)

    def begin(self):
        self.pipeline.start_epoch(2)

    def next_batch(self):
        # Fixed-window deployments consume full microbatches only. Record every
        # skipped pool tail; full-epoch matched readers retain their own policy.
        while True:
            if self.previous_end:
                torch.cuda.synchronize()
                self.pool.retire()
                self.pipeline.reclaim_finished_pools()
                self.pool=None
            if self.pool is None:
                self.pool=self.pipeline.next_pool()
            batch=next(self.pool)
            self.previous_end=batch.is_pool_end
            ids=batch.global_image_ids
            if len(ids)<64:
                self.skipped_partial_ids.extend(self.records[i]['logical_sample_id'] for i in ids)
                del batch
                continue
            inputs=(batch.projected,) if self.model not in ('swinv2','vitti') else (batch.y,batch.cbcr)
            return TrainingBatch(inputs=inputs,labels=batch.targets,
                identities=[SampleIdentity(batch.epoch,i,self.records[i]['logical_sample_id']) for i in ids],
                augmentations=[],on_device=True,stage_seconds={},native_stats_source=batch)

    def close(self):
        torch.cuda.synchronize()
        if self.pool is not None:
            self.pool.retire()
            self.pool=None
        self.pipeline.close()


def decode_native_rgb_reference(samples, *, pipeline, device, dali_config):
    """Independent full decode of source files, used only by the excluded input check."""
    import numpy as np
    if pipeline == 'ffcv-jpeg':
        from ffcv.libffcv import imdecode
        decoded = []
        for sample in samples:
            pixels = np.empty((sample.height, sample.width, 3), dtype=np.uint8)
            imdecode(np.fromfile(sample.path, dtype=np.uint8), pixels,
                     sample.height, sample.width, sample.height, sample.width,
                     0, 0, 1, 1, False, False)
            decoded.append(pixels)
        return decoded
    from nvidia.dali import backend, ops, types
    from nvidia.dali.pipeline import Pipeline
    if pipeline == 'coordl' and 'cache_size' not in backend.GetSchema('FileReader').GetArgumentNames():
        raise RuntimeError('CoorDL reference requires the existing coordl-python environment')
    encoded = [np.fromfile(sample.path, dtype=np.uint8) for sample in samples]
    decoder_options = (dict(hybrid_huffman_threshold=dali_config.get('hybrid_huffman_threshold', 1_000_000),
                            hw_decoder_load=dali_config.get('hw_decoder_load', 0.65))
                       if pipeline == 'dali' else {})

    class FullDecode(Pipeline):
        def __init__(self):
            super().__init__(batch_size=len(samples), num_threads=16, device_id=device.index or 0,
                             seed=11997733, prefetch_queue_depth=1)
            self.source = ops.ExternalSource()
            self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB, **decoder_options)

        def define_graph(self):
            self.encoded_node = self.source()
            return self.decode(self.encoded_node)

        def iter_setup(self):
            self.feed_input(self.encoded_node, encoded)

    reference = FullDecode()
    reference.build()
    decoded = reference.run()[0].as_cpu()
    return [np.array(decoded.at(i), copy=True) for i in range(len(samples))]


def check_rgb_input(images, labels, identities, samples, decisions, *, pipeline, device, dali_config, output):
    """Check native decode + independent CPU postprocessing; report PIL differences separately."""
    import cv2
    prefix = samples[:len(identities)]
    native = pipeline in ('dali', 'coordl', 'ffcv-jpeg')
    decoded = (decode_native_rgb_reference(prefix, pipeline=pipeline, device=device, dali_config=dali_config)
               if native else None)
    reference = PlannedRgbDataset(samples, decisions)
    actual_labels = labels.detach().cpu().tolist()
    report = dict(reference='native-full-decode + CPU postprocess' if native else 'PIL + CPU postprocess',
                  max_error_limit=8/255, mean_error_limit=1/255,
                  order_matches=[i.logical_sample_id for i in identities] == [s.logical_sample_id for s in prefix],
                  labels_match=actual_labels == [s.label for s in prefix], samples=[])
    for index, identity in enumerate(identities[:4]):
        pil_expected, label, _ = reference[index]
        expected = pil_expected
        if native:
            d = decisions[index]
            pixels = decoded[index]
            if pixels.shape != (prefix[index].height, prefix[index].width, 3):
                raise ValueError(f'Native RGB decode shape mismatch: {prefix[index].logical_sample_id}')
            pixels = cv2.resize(pixels[d.crop_y:d.crop_y+d.crop_height, d.crop_x:d.crop_x+d.crop_width],
                                (224, 224), interpolation=cv2.INTER_LINEAR)
            if d.horizontal_flip:
                pixels = pixels[:, ::-1]
            expected = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
        actual = images[index].detach().cpu()
        if actual.shape != expected.shape:
            raise ValueError(f'RGB model input shape mismatch: {identity.logical_sample_id}')
        delta, pil_delta = (actual-expected).abs(), (actual-pil_expected).abs()
        report['samples'].append(dict(logical_sample_id=identity.logical_sample_id,
            expected_sample_id=prefix[index].logical_sample_id, index=index,
            expected_label=label, actual_label=actual_labels[index],
            max_error=float(delta.max()), mean_error=float(delta.mean()),
            pil_max_error=float(pil_delta.max()), pil_mean_error=float(pil_delta.mean()),
            augmentation=decisions[index].as_dict()))
    output.mkdir(parents=True, exist_ok=True)
    (output/'rgb_input_check.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['order_matches']:
        raise ValueError('RGB sample sequence mismatch')
    if not report['labels_match']:
        raise ValueError('RGB label mismatch')
    for row in report['samples']:
        if not (row['max_error'] <= 8/255 and row['mean_error'] <= 1/255):
            raise ValueError(f'RGB keyed crop/flip/normalization gate failed: {row}')
    return report


class PlannedRgbDataset(torch.utils.data.Dataset):
    """Same keyed crop/flip as DALI/CoorDL/FFCV, bilinear without antialias."""
    def __init__(self, samples, decisions):
        self.samples, self.decisions = samples, decisions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import cv2
        import numpy as np
        from PIL import Image
        sample, d = self.samples[index], self.decisions[index]
        with Image.open(sample.path) as image:
            pixels = np.asarray(image.convert('RGB'))
        pixels = cv2.resize(pixels[d.crop_y:d.crop_y+d.crop_height, d.crop_x:d.crop_x+d.crop_width],
                            (224,224), interpolation=cv2.INTER_LINEAR)
        if d.horizontal_flip:
            pixels = pixels[:, ::-1]
        x = torch.from_numpy(pixels.copy()).permute(2,0,1).float().div_(127.5).sub_(1)
        return x, sample.label, index


def rgb_worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class PlannedRgbWindow:
    def __init__(self, samples, *, batch_size, workers, device, config):
        self.samples, self.batch_size, self.workers = samples, batch_size, workers

    def begin(self, identities, decisions):
        self.identities, self.decisions = identities, decisions
        self.loader = torch.utils.data.DataLoader(PlannedRgbDataset(self.samples, decisions),
            batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers,
            pin_memory=True, worker_init_fn=rgb_worker_init,
            prefetch_factor=2 if self.workers else None)
        self.iterator = iter(self.loader)

    def next_batch(self):
        x, labels, indices = next(self.iterator)
        ids = indices.tolist()
        return TrainingBatch(inputs=(x,), labels=labels,
            identities=[self.identities[i] for i in ids],
            augmentations=[self.decisions[i] for i in ids], on_device=False, stage_seconds={})

    def close(self):
        self.iterator = None
        self.loader = None
