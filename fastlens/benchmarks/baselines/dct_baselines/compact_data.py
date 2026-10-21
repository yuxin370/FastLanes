"""Train-only native B32/Q50 compact mosaics, sharing one FS store across budgets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
import io
import itertools
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image

from .contracts import ReadRequest, artifact, write_json
from .native import NativeSpec, fit_statistics, fs_source, mosaic, read_ic_archive, write_ic_mosaics
from .store import (Counters, DEFAULT_DECODE_BUDGET, DEFAULT_MEMORY_BUDGET,
                    StoreReader, _build_groups, build_store, fingerprint)


METHODS = ("ic_shared", "ic_batched", "fs")


def _archive_info(path, indices, values):
    with zipfile.ZipFile(path) as archive:
        payload = next(info.compress_size for info in archive.infolist() if info.filename == "mosaic.npy")
    return dict(file=str(path), indices=indices, size_bytes=path.stat().st_size,
                q_payload_bytes=payload, decoded_q_values=sum(v.size for v in values),
                decoded_q_bytes=sum(v.nbytes for v in values))


def _prepare_group(task, *, root, image_size, ks):
    group_id, records = task
    sources = []
    for index, record in records:
        path = Path(record["path"])
        with Image.open(path) as image:
            if image.size != (image_size, image_size):
                raise ValueError(f"{path}: expected {image_size}x{image_size}; resize explicitly before preparing")
            rgb = np.asarray(image.convert("RGB"))
        source = fs_source(rgb, record["logical_sample_id"], record["label"], "train")
        source.metadata.update(path=str(path), source_sha256=artifact(path)["sha256"])
        sources.append(source)
    archives = {}
    indices = [index for index, _ in records]
    for k in ks:
        spec = NativeSpec(k=k)
        values = [mosaic(np.stack(source.coefficients), k) for source in sources]
        shared = []
        for index, value in zip(indices, values):
            path = root/f"k{k}"/"ic_shared"/f"image_{index:08d}.npz"
            write_ic_mosaics(path, [value], spec)
            shared.append(_archive_info(path, [index], [value]))
        path = root/f"k{k}"/"ic_batched"/f"group_{group_id:06d}.npz"
        write_ic_mosaics(path, values, spec)
        archives[k] = dict(ic_shared=shared, ic_batched=[_archive_info(path, indices, values)])
    return sources, archives


def prepare(selection, output, *, image_size, ks=(3, 8, 16, 32), group_size=256, workers=1):
    """No online/offline implicit resize; selection must explicitly identify train samples."""
    selection, root = Path(selection).resolve(), Path(output).resolve()
    value = json.loads(selection.read_text())
    rows = value["samples"]
    if not rows or any(row.get("split", value.get("split")) != "train" for row in rows):
        raise ValueError("compact preparation requires an explicitly marked training split")
    if image_size <= 0 or image_size % 32 or group_size <= 0 or workers <= 0:
        raise ValueError("image_size must be a positive multiple of 32; group_size/workers must be positive")
    ks = tuple(ks)
    if not ks or len(set(ks)) != len(ks):
        raise ValueError("provide distinct nonempty triangle budgets")
    for k in ks:
        NativeSpec(k=k)
    samples = []
    for row in rows:
        if type(row["label"]) is not int or not 0 <= row["label"] < 1000:
            raise ValueError("labels must be integer ImageNet class IDs in [0,999]")
        path = Path(row["path"])
        path = path if path.is_absolute() else selection.parent/path
        samples.append(dict(logical_sample_id=str(row["logical_sample_id"]), label=row["label"],
                            path=str(path.resolve()), split="train"))
    if len({row["logical_sample_id"] for row in samples}) != len(samples) or len({row["path"] for row in samples}) != len(samples):
        raise ValueError("training selection must contain unique image IDs and paths")
    root.mkdir(parents=True, exist_ok=False)
    for k in ks:
        for method in METHODS[:2]:
            (root/f"k{k}"/method).mkdir(parents=True)
    started = time.perf_counter()
    stores = {k: dict(ic_shared=[], ic_batched=[]) for k in ks}
    iterator = iter(enumerate(samples))
    tasks = enumerate(iter(lambda: list(itertools.islice(iterator, group_size)), []))
    build = partial(_prepare_group, root=root, image_size=image_size, ks=ks)

    def sources():
        for images, archives in _build_groups(tasks, build, workers):
            for k in ks:
                for method in METHODS[:2]:
                    stores[k][method].extend(archives[k][method])
            yield from images

    # Both stages use bounded submission. At most workers native transform groups
    # and workers FS compression groups are in flight; all budgets share this DCT.
    source_ref = artifact(selection)
    fs_manifest = build_store(sources(), root/"fs_native", "fs_native", group_size=group_size,
                              source_artifact=source_ref, workers=workers)
    for row, fs_record in zip(samples, fs_manifest["images"]):
        row["source_sha256"] = fs_record["metadata"]["source_sha256"]
    source_identity = fingerprint(samples)
    outputs = {}
    for k in ks:
        spec = NativeSpec(k=k)
        statistics = fit_statistics((read_ic_archive(info["file"], spec)[0]
                                     for info in stores[k]["ic_shared"]), split="train")
        shape = [3, image_size//32*k, image_size//32*k]
        representation = spec.contract()
        manifest = dict(version=1, split="train", samples=samples, source_selection=source_ref,
                        source_identity=source_identity, image_size=image_size, input_shape=shape,
                        representation=representation, normalization=statistics,
                        data_identity=fingerprint(dict(source=source_identity, representation=representation,
                                                       normalization=statistics)),
                        stores=dict(**stores[k], fs=dict(root=str(root/"fs_native"),
                                                         content_fingerprint=fs_manifest["content_fingerprint"])),
                        construction=dict(group_size=group_size, transform_workers=workers,
                                          fs_compression_workers=workers, total_wall_s=time.perf_counter()-started,
                                          fs_shared_across_k=list(ks), resize="none"))
        path = root/f"k{k}"/"compact.json"
        write_json(path, manifest)
        outputs[k] = str(path)
    return outputs


class CompactDataset:
    """Bounded preparation-pool reads; no full B32 dense reconstruction or runtime hash audit."""
    def __init__(self, manifest, method, pool_size=256, *, workers=1,
                 decode_budget_bytes=DEFAULT_DECODE_BUDGET, memory_budget_bytes=DEFAULT_MEMORY_BUDGET):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest["split"] != "train" or method not in METHODS or pool_size <= 0 or workers <= 0:
            raise ValueError("invalid compact training dataset/method/pool/workers")
        self.method, self.pool_size = method, pool_size
        self.samples = self.manifest["samples"]
        self.image_count = len(self.samples)
        self.sample_ids = [row["logical_sample_id"] for row in self.samples]
        self.labels = [row["label"] for row in self.samples]
        self.input_shape = tuple(self.manifest["input_shape"])
        self.k = self.manifest["representation"]["k"]
        self.spec = NativeSpec(k=self.k)
        self.frequencies = tuple(u*32+v for u in range(self.k) for v in range(self.k-u))
        self.compact_positions = np.array([f//32*self.k+f % 32 for f in self.frequencies])
        self.mean = np.asarray(self.manifest["normalization"]["mean"], dtype=np.float32)[None, :, None, None]
        self.std = np.asarray(self.manifest["normalization"]["std"], dtype=np.float32)[None, :, None, None]
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 and method != "fs" else None
        self.memory_budget_bytes = memory_budget_bytes
        self.reader = None
        if method == "fs":
            self.reader = StoreReader(self.manifest["stores"]["fs"]["root"], workers=workers,
                                      decode_budget_bytes=decode_budget_bytes, memory_budget_bytes=memory_budget_bytes,
                                      verify_reads=False)
            if self.reader.identity["content_fingerprint"] != self.manifest["stores"]["fs"]["content_fingerprint"]:
                self.reader.close()
                raise ValueError("FS source identity differs from compact manifest")
        else:
            self.groups = self.manifest["stores"][method]
            self.group_for_index = {index: (group_id, local) for group_id, group in enumerate(self.groups)
                                    for local, index in enumerate(group["indices"])}
            self.workers = workers
            self.decode_budget_bytes = decode_budget_bytes

    def read(self, indices):
        import torch

        indices = list(indices)
        if not indices or any(type(index) is not int or not 0 <= index < self.image_count for index in indices):
            raise ValueError("read requires nonempty valid integer indices")
        output_bytes = len(indices)*int(np.prod(self.input_shape))*4
        if output_bytes >= self.memory_budget_bytes:
            raise MemoryError("compact reader budget cannot hold requested float32 outputs")
        output = np.empty((len(indices), *self.input_shape), dtype=np.float32)
        if self.method == "fs":
            # Reserve the final pool plus a single mosaic and its transposed copy.
            scratch_bytes = 2*output[0].nbytes
            self.reader.memory_budget_bytes = self.memory_budget_bytes-output_bytes-scratch_bytes
            if self.reader.memory_budget_bytes <= 0:
                raise MemoryError("compact reader budget cannot hold FS assembly scratch")
            requests = [ReadRequest(self.sample_ids[index], (self.frequencies,)*3) for index in indices]
            views, counters = self.reader.read(self.reader.plan(requests))
            grid = self.manifest["image_size"]//32
            for destination, view in enumerate(views):
                # k*k structural zeros only, never allocate a B32 full-frequency tensor.
                compact = np.zeros((3, grid, grid, self.k*self.k), dtype=np.float32)
                for component, values in enumerate(view.coefficients):
                    compact[component, ..., self.compact_positions] = np.moveaxis(values, -1, 0)
                output[destination] = compact.reshape(3, grid, grid, self.k, self.k).transpose(0, 1, 3, 2, 4).reshape(self.input_shape)
        else:
            tasks = {}
            for destination, index in enumerate(indices):
                group_id, local = self.group_for_index[index]
                tasks.setdefault(group_id, []).append((destination, local))
            counters = Counters(requested_q_values=len(indices)*3*(self.manifest["image_size"]//32)**2*len(self.frequencies))
            counters.unique_requested_q_values = len(set(indices))*counters.requested_q_values//len(indices)
            counters.full_grid_requested_q_values = counters.requested_q_values

            def decode(task):
                group_id, destinations = task
                info = self.groups[group_id]
                encoded = Path(info["file"]).read_bytes()
                with np.load(io.BytesIO(encoded), allow_pickle=False) as arrays:
                    payload, shapes, offsets = arrays["mosaic"], arrays["shapes"], arrays["offsets"]
                    contract = json.loads(arrays["contract"].tobytes())
                    if contract != json.loads(json.dumps(self.spec.contract())) or payload.dtype != np.dtype("<i4"):
                        raise ValueError("native IC representation differs from compact manifest")
                    for destination, local in destinations:
                        if tuple(shapes[local]) != self.input_shape:
                            raise ValueError("stored compact mosaic geometry differs")
                        output[destination] = payload[offsets[local]:offsets[local+1]].reshape(self.input_shape)
                return Counters(application_read_bytes=len(encoded), compressed_payload_read_bytes=info["q_payload_bytes"],
                                metadata_read_bytes=len(encoded)-info["q_payload_bytes"], file_opens=1, read_calls=1,
                                inflated_members=4, decoded_q_values=payload.size, decoded_q_bytes=payload.nbytes)

            tasks = iter(tasks.items())
            budget = min(self.decode_budget_bytes, self.memory_budget_bytes-output.nbytes)
            while True:
                chunk, size = [], 0
                for task in tasks:
                    member_bytes = self.groups[task[0]]["decoded_q_bytes"]
                    if member_bytes > budget:
                        raise MemoryError("compact reader budget cannot hold outputs and one whole NPZ member")
                    if chunk and size+member_bytes > budget:
                        tasks = itertools.chain([task], tasks)
                        break
                    chunk.append(task)
                    size += member_bytes
                    if len(chunk) == self.workers:
                        break
                if not chunk:
                    break
                counters.active_decoded_peak_bytes = max(counters.active_decoded_peak_bytes, size)
                for observed in (self.executor.map(decode, chunk) if self.executor else map(decode, chunk)):
                    counters.add(observed)
        np.subtract(output, self.mean, out=output)
        np.divide(output, self.std, out=output)
        counters.output_peak_bytes += output.nbytes
        if self.method == "fs":
            counters.output_peak_bytes += scratch_bytes
        counters.reader_memory_peak_bytes = max(counters.reader_memory_peak_bytes,
                                                counters.output_peak_bytes+counters.active_decoded_peak_bytes)
        return torch.from_numpy(output), torch.tensor([self.labels[index] for index in indices], dtype=torch.int64), asdict(counters)

    def iter_pools(self):
        for start in range(0, self.image_count, self.pool_size):
            yield self.read(list(range(start, min(start+self.pool_size, self.image_count))))

    def close(self):
        if self.reader is not None:
            self.reader.close()
        if self.executor is not None:
            self.executor.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 8, 16, 32])
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.output, image_size=args.image_size,
                             ks=args.k, group_size=args.group_size, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
