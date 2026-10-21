"""IC-style offline compact B8 mosaics; online NPZ decode and normalization only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache, partial
import io
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from .contracts import ReadRequest, artifact, load_artifact, write_json
from .materialize import Materializer
from .store import Counters, StoreReader, _build_groups, fingerprint, write_npz


@lru_cache(maxsize=1)
def _worker_reader(profile_path, source_store):
    # The parent verified content before dispatch; workers read those frozen files.
    torch.set_num_threads(1)
    profile = json.loads(Path(profile_path).read_text())
    return profile, Materializer(profile['profile']), StoreReader(source_store, verify_reads=False)


def _prepare_group(task, *, profile_path, source_store, root):
    group_id, indexed_samples = task
    profile, materializer, reader = _worker_reader(profile_path, source_store)
    frequencies = tuple(map(tuple, profile['profile']['source_frequencies']))
    requests = [ReadRequest(sample['image_id'], frequencies) for _, sample in indexed_samples]
    views, _ = reader.read(reader.plan(requests))
    side = profile['mosaic_side']
    frequency = profile['output_frequencies']
    u, v = np.asarray(frequency)//8, np.asarray(frequency) % 8
    values = np.zeros((len(views), *profile['input_shape']), dtype='<i4')
    blocks = values.reshape(len(views), 3, 28, side, 28, side).transpose(0, 1, 2, 4, 3, 5)
    for index, view in enumerate(views):
        raw = materializer.raw(materializer.source_coefficients(view), view.tables,
                               dict(crop_box=profile['profile']['default_crop'], flip=False))
        for component, plane in enumerate(raw):
            blocks[index, component, ..., u, v] = plane.numpy()[..., u, v].astype('<i4').transpose(2, 0, 1)

    indices = [index for index, _ in indexed_samples]

    def shared():
        records, files = [], {}
        for index, value in zip(indices, values):
            name = f'image_{index:08d}.npz'
            files[name] = write_npz(root/'ic_prepared_shared'/name, q=value[None])
            records.append(dict(file=name, indices=[index]))
        return records, files

    def batched():
        name = f'group_{group_id:06d}.npz'
        return ([dict(file=name, indices=indices)],
                {name: write_npz(root/'ic_prepared_batched'/name, q=values)})

    with ThreadPoolExecutor(max_workers=2) as executor:
        shared_result = executor.submit(shared)
        batched_result = executor.submit(batched)
        return shared_result.result(), batched_result.result()


def prepare(profile, source_store, output, *, group_size=256, workers=32, content_evidence=None):
    """Transform once from an already verified coefficient store, not a reference dump."""
    profile, source_store, root = Path(profile).resolve(), Path(source_store).resolve(), Path(output).resolve()
    contract = json.loads(profile.read_text())
    source = load_artifact(contract['source_manifest'])
    samples = source['images']
    if (group_size <= 0 or workers <= 0 or not samples or contract['normalization'] is None
            or contract['split'] != 'train' or any(sample['split'] != 'train' for sample in samples)):
        raise ValueError('prepared compact requires normalized training data and positive group/workers')
    with StoreReader(source_store) as reader:
        if reader.manifest['source_artifact'] != contract['source_manifest']:
            raise ValueError('coefficient store belongs to a different source population')
        # The existing audit/stat mechanism protects the source without repeating q/Q checks.
        verified = reader.verify_content(content_evidence)
    root.mkdir(parents=True, exist_ok=False)
    methods = ('ic_prepared_shared', 'ic_prepared_batched')
    stores = {method: dict(groups=[], files={}) for method in methods}
    for method in methods:
        (root/method).mkdir()
    started = time.perf_counter()
    iterator = iter(enumerate(samples))
    tasks = enumerate(iter(lambda: list(itertools.islice(iterator, group_size)), []))
    build = partial(_prepare_group, profile_path=str(profile), source_store=str(source_store), root=root)
    processed = 0
    for results in _build_groups(tasks, build, workers):
        for method, (groups, files) in zip(methods, results):
            stores[method]['groups'].extend(groups)
            stores[method]['files'].update(files)
        processed += sum(len(group['indices']) for group in results[1][0])
        if processed % 1024 == 0 or processed == len(samples):
            print(f'Prepared compact images: {processed}/{len(samples)}', flush=True)
    wall = time.perf_counter()-started
    outputs = {}
    for method in methods:
        manifest = dict(version=1, method=method, split='train', representation='jpeg_b8_prepared_compact_int32',
                        profile=artifact(profile), source_manifest=contract['source_manifest'],
                        source_identity=contract['source_identity'], data_identity=contract['data_identity'],
                        input_shape=contract['input_shape'], normalization=contract['normalization'],
                        samples=[dict(image_id=row['image_id'], label=row['label']) for row in samples],
                        offline_operations=['source_q_mask', 'dequantize', 'fixed_crop', 'spectral_resize',
                                            'round_clamp', 'output_frequency_selection', 'compact_mosaic'],
                        online_operations=['npz_decode', 'normalize_fp32'],
                        source_store=verified['store'], group_size=1 if method.endswith('shared') else group_size,
                        **stores[method])
        manifest['content_fingerprint'] = fingerprint(manifest)
        manifest['construction'] = dict(workers=workers, compression_threads_per_group=2,
                                        shared_build_wall_s=wall, source_content_verification=verified)
        path = root/method/'prepared.json'
        write_json(path, manifest)
        outputs[method] = str(path)
    return outputs


class PreparedCompactDataset:
    """Read fixed compact int32 mosaics; preserve order, duplicates, and tail samples."""
    # The same content audit and cheap stat reuse used by existing coefficient stores.
    _path = StoreReader._path
    _check_content = StoreReader._check_content
    content_file_states = StoreReader.content_file_states
    verify_content = StoreReader.verify_content

    def __init__(self, manifest, *, workers=1, content_evidence=None):
        path = Path(manifest).resolve()
        self.root = path.parent
        self.manifest = json.loads(path.read_text())
        expected = fingerprint({key: value for key, value in self.manifest.items()
                                if key not in ('content_fingerprint', 'construction')})
        if (workers <= 0 or self.manifest['version'] != 1 or self.manifest['split'] != 'train'
                or self.manifest['method'] not in ('ic_prepared_shared', 'ic_prepared_batched')
                or self.manifest['content_fingerprint'] != expected):
            raise ValueError('invalid prepared compact manifest or workers')
        contract = load_artifact(self.manifest['profile'])
        if (self.manifest['input_shape'] != contract['input_shape']
                or self.manifest['normalization'] != contract['normalization']
                or self.manifest['source_manifest'] != contract['source_manifest']):
            raise ValueError('prepared compact manifest differs from its source profile')
        self.identity = dict(manifest=artifact(path), content_fingerprint=expected)
        self.content_verification = self.verify_content(content_evidence)
        self.samples = self.manifest['samples']
        self.image_count = len(self.samples)
        self.sample_ids = [row['image_id'] for row in self.samples]
        self.labels = [row['label'] for row in self.samples]
        self.input_shape = tuple(self.manifest['input_shape'])
        self.groups = self.manifest['groups']
        self.group_for_index = {index: (group_id, local) for group_id, group in enumerate(self.groups)
                                for local, index in enumerate(group['indices'])}
        self.mean = np.asarray(self.manifest['normalization']['mean'], dtype=np.float32)[None, :, None, None]
        self.std = np.asarray(self.manifest['normalization']['std'], dtype=np.float32)[None, :, None, None]
        self.workers = workers
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.execution_partition = dict(read_decode='cpu', normalization='cpu', offline='crop/resize/round/mosaic')
        self.producer_h2d_unmeasured = False

    def read(self, indices):
        indices = list(indices)
        if not indices or any(type(index) is not int or not 0 <= index < self.image_count for index in indices):
            raise ValueError('read requires nonempty valid integer indices')
        output = np.empty((len(indices), *self.input_shape), dtype=np.float32)
        tasks = {}
        for destination, index in enumerate(indices):
            group_id, local = self.group_for_index[index]
            tasks.setdefault(group_id, []).append((destination, local))

        def decode(task):
            group_id, destinations = task
            group = self.groups[group_id]
            info = self.manifest['files'][group['file']]
            encoded = self._path(group['file']).read_bytes()
            with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
                values = archive['q']
            if values.dtype != np.dtype('<i4') or values.shape != (len(group['indices']), *self.input_shape):
                raise ValueError('stored prepared compact dtype/shape differs from its manifest')
            for destination, local in destinations:
                output[destination] = values[local]
            return Counters(application_read_bytes=len(encoded), compressed_payload_read_bytes=info['q_payload_bytes'],
                            metadata_read_bytes=len(encoded)-info['q_payload_bytes'], file_opens=1, read_calls=1,
                            inflated_members=1, decoded_q_values=values.size, decoded_q_bytes=values.nbytes)

        per_image = int(np.prod(self.input_shape))
        counters = Counters(requested_q_values=len(indices)*per_image,
                            unique_requested_q_values=len(set(indices))*per_image,
                            full_grid_requested_q_values=len(indices)*per_image, output_peak_bytes=output.nbytes)
        tasks = iter(tasks.items())
        # Avoid executor.map eagerly queueing the complete dataset for large requests.
        while chunk := list(itertools.islice(tasks, self.workers)):
            active = sum(self.manifest['files'][self.groups[group_id]['file']]['decoded_bytes'] for group_id, _ in chunk)
            counters.active_decoded_peak_bytes = max(counters.active_decoded_peak_bytes, active)
            for observed in (self.executor.map(decode, chunk) if self.executor else map(decode, chunk)):
                counters.add(observed)
        np.subtract(output, self.mean, out=output)
        np.divide(output, self.std, out=output)
        counters.reader_memory_peak_bytes = output.nbytes+counters.active_decoded_peak_bytes
        return torch.from_numpy(output), torch.tensor([self.labels[index] for index in indices], dtype=torch.int64), asdict(counters)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--source-store', type=Path, required=True)
    parser.add_argument('--content-evidence', type=Path, help='existing StoreReader content_verification JSON')
    parser.add_argument('--group-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.content_evidence.read_text()) if args.content_evidence else None
    outputs = prepare(args.profile, args.source_store, args.output, group_size=args.group_size,
                      workers=args.workers, content_evidence=evidence)
    print(json.dumps(dict(status='passed', manifests=outputs), indent=2))


if __name__ == '__main__':
    main()
