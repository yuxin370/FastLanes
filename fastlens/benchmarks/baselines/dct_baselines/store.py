"""IC image-major and FS whole-diagonal NPZ storage, planning and accounting."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from functools import lru_cache, partial
import io
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import resource
import time
import zipfile

import numpy as np

from .contracts import VERSION, ReadRequest, artifact, frequency_set, signed_cast, write_json

IC_METHODS = ("ic_matched_shared", "ic_matched_fixedk", "ic_matched_batched")
METHODS = (*IC_METHODS, "fs_matched_b8", "fs_native")
DEFAULT_DECODE_BUDGET = 512 * 1024**2
DEFAULT_MEMORY_BUDGET = 1024 * 1024**2


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def training_store_identity(root, records, backend=None):
    """Check a matched store against an established experiment population."""
    with StoreReader(root) as reader:
        method = reader.manifest["method"]
        if reader.manifest["variant"] != "adapted":
            raise ValueError("formal DCT training requires an original-JPEG matched store")
        if backend == "ic-matched-dct" and method not in IC_METHODS or backend == "fs-matched-dct" and method != "fs_matched_b8":
            raise ValueError("baseline store method differs from the execution backend")
        if set(reader.images) != {record[0] for record in records}:
            raise ValueError("baseline store and experiment contain different images")
        for logical_id, label, path in records:
            record = reader.images[logical_id]
            info = record["metadata"]
            if info["label"] != label or Path(info["path"]).resolve() != Path(path).resolve() or record["masks"] != [list(range(64))]*3:
                raise ValueError("baseline source identity/label/frequency dependencies differ from the experiment")
            if artifact(path)["sha256"] != info["jpeg_sha256"]:
                raise ValueError(f"current JPEG content differs from the baseline source: {logical_id}")
        return dict(store=reader.identity, source=reader.manifest["source_artifact"],
                    method=method, decoded_cache_bytes=0, reader_scope="bounded-read-many-per-microbatch",
                    source_verification="current JPEG SHA256 checked at startup; cache state unmeasured",
                    experiment_type="shared_partial_processing_training",
                    execution_partition=dict(read_decode_geometry="cpu", augmentation_normalization_mixup="model device",
                                             model="model device"))


def diagonal_frequencies(block_size, diagonal):
    return tuple(u*block_size + diagonal-u for u in range(block_size) if 0 <= diagonal-u < block_size)


def _indexer(indices):
    """Use a strided view for regular selections, including whole diagonals."""
    if len(indices) == 1:
        return slice(indices[0], indices[0]+1)
    step = indices[1]-indices[0]
    if step > 0 and all(right-left == step for left, right in zip(indices, indices[1:])):
        return slice(indices[0], indices[-1]+1, step)
    return list(indices)


@lru_cache(maxsize=128)
def _ic_selection(stored, requested):
    if stored == requested:
        return slice(None)
    positions = {frequency: index for index, frequency in enumerate(stored)}
    return _indexer(tuple(positions[frequency] for frequency in requested))


@lru_cache(maxsize=128)
def _fs_selections(block_size, frequencies):
    positions = {frequency: index for index, frequency in enumerate(frequencies)}
    selections = []
    for d in sorted({f//block_size+f % block_size for f in frequencies}):
        diagonal = diagonal_frequencies(block_size, d)
        pairs = [(positions[f], i) for i, f in enumerate(diagonal) if f in positions]
        selections.append((d, len(diagonal), _indexer(tuple(p[0] for p in pairs)),
                           _indexer(tuple(p[1] for p in pairs))))
    return tuple(selections)


def mask_family(frequencies, b):
    values = set(frequencies)
    for k in range(1, b+1):
        if values == {u*b+v for u in range(b) for v in range(b) if u+v < k}:
            return f"native_triangle_k{k}"
    return "matched_arbitrary_adapter"


def write_npz(path, **arrays):
    if any(np.asarray(a).dtype.kind not in "iu" for a in arrays.values()):
        raise TypeError("stores contain numeric integer arrays only")
    np.savez_compressed(path, **arrays)
    encoded = Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        payload = sum(info.compress_size for info in archive.infolist() if info.filename == "q.npy")
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), size_bytes=len(encoded),
                decoded_bytes=sum(np.asarray(a).nbytes for a in arrays.values()), q_payload_bytes=payload)


def _build_group(task, *, root, method, source_frequencies, dtype_policy, from_manifest):
    group_id, batch = task
    if from_manifest:
        from .source import source_images
        # Decode inside the worker; send only frozen metadata across processes.
        batch = list(source_images(dict(images=batch)))
    b = 32 if method == "fs_native" else 8
    records, files, table_parts = [], {}, []
    offsets = [0]
    for image in batch:
        if image.metadata["block_size"] != b:
            raise ValueError("native FS requires B32; matched adapters require JPEG B8")
        if method == "fs_native" and image.metadata.get("representation") != "native_b32_q50":
            raise ValueError("FS native needs separately quantized B32/Q50 input")
        if method != "fs_native" and image.metadata.get("representation", "jpeg_b8") != "jpeg_b8":
            raise ValueError("native coefficients cannot enter a matched store")
        masks = ([list(range(b*b)) for _ in image.coefficients] if source_frequencies is None else source_frequencies)
        if len(masks) != len(image.coefficients):
            raise ValueError("one source frequency mask is required per component")
        masks = [list(frequency_set(mask, b)) for mask in masks]
        records.append(dict(metadata=image.metadata, group=group_id,
                            component_offset=offsets[-1], masks=masks,
                            mask_family=[mask_family(mask, b) for mask in masks]))
        offsets.append(offsets[-1]+len(image.coefficients))
        table_parts.extend(image.tables)
    tables = np.stack(table_parts).astype("<u2")
    # Counts of spatial blocks in image/component order. The same prefix
    # describes all FS columns; no spatial or per-scalar seek index exists.
    blocks = [q.shape[0]*q.shape[1] for image in batch for q in image.coefficients]
    group = dict(images=[i.metadata["image_id"] for i in batch], block_offsets=np.cumsum([0, *blocks]).tolist(), units=[])
    if method in IC_METHODS:
        parts, value_offsets = [], [0]
        for image, record in zip(batch, records):
            for q, mask in zip(image.coefficients, record["masks"]):
                part = q.reshape(*q.shape[:2], b*b)[..., mask].reshape(-1)
                parts.append(part)
                value_offsets.append(value_offsets[-1]+part.size)
        payload = np.concatenate(parts)
        dtype = "<i4" if dtype_policy == "paper" else "<i2"
        if dtype_policy == "safe" and payload.min() >= -128 and payload.max() <= 127:
            dtype = "i1"
        payload = signed_cast(payload, dtype)
        filename = f"group_{group_id:06d}.npz"
        files[filename] = write_npz(root / filename, q=payload, tables=tables)
        group.update(value_offsets=value_offsets, table_file=filename)
        group["units"].append(dict(file=filename, diagonal=None, dtype=payload.dtype.str, values=payload.size))
    else:
        table_file = f"group_{group_id:06d}_tables.npz"
        files[table_file] = write_npz(root / table_file, tables=tables)
        group["table_file"] = table_file
        for d in range(2*b-1):
            diagonal = diagonal_frequencies(b, d)
            parts = [q.reshape(*q.shape[:2], b*b)[..., diagonal].reshape(-1) for image in batch for q in image.coefficients]
            payload = np.concatenate(parts)
            dtype = "i1" if dtype_policy != "int16" and payload.min() >= -128 and payload.max() <= 127 else "<i2"
            payload = signed_cast(payload, dtype)
            filename = f"group_{group_id:06d}_d{d:03d}.npz"
            files[filename] = write_npz(root / filename, q=payload)
            group["units"].append(dict(file=filename, diagonal=d, dtype=payload.dtype.str, values=payload.size))
    return group, records, files


def _build_groups(tasks, build, workers):
    if workers == 1:
        yield from map(build, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        # Keep at most workers groups in flight; Executor.map in Python 3.11
        # would eagerly consume the whole source iterator. Commit in source order.
        pending = deque(executor.submit(build, task) for task in itertools.islice(tasks, workers))
        while pending:
            yield pending.popleft().result()
            for task in itertools.islice(tasks, 1):
                pending.append(executor.submit(build, task))


def build_store(images, root, method, *, group_size=256, source_frequencies=None,
                dependency_evidence=None, dtype_policy="paper", source_artifact=None, workers=1):
    """Build from a frozen source manifest or an iterable of decoded SourceImages.

    With a manifest, workers perform JPEG extraction and NPZ compression locally.
    Both paths retain source order and write the same physical representation.
    """
    if method not in METHODS or group_size <= 0 or workers <= 0:
        raise ValueError("unknown method or invalid storage group size/worker count")
    if dtype_policy not in ("paper", "int16", "safe"):
        raise ValueError("dtype_policy must be paper, int16 or safe")
    if method == "ic_matched_fixedk" and (source_frequencies is None or not dependency_evidence):
        raise ValueError("FixedK needs explicit source dependencies and evidence")
    if source_frequencies is not None and method != "ic_matched_fixedk":
        raise ValueError("only the explicit FixedK variant prunes matched source frequencies")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    child_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    effective_group_size = group_size if method in ("ic_matched_batched", "ic_matched_fixedk", "fs_matched_b8", "fs_native") else 1
    groups, records, seen, files = [], [], set(), {}
    from_manifest = isinstance(images, dict)
    iterator = iter(images["images"] if from_manifest else images)
    batches = iter(lambda: list(itertools.islice(iterator, effective_group_size)), [])
    build = partial(_build_group, root=root, method=method, source_frequencies=source_frequencies,
                    dtype_policy=dtype_policy, from_manifest=from_manifest)
    for group, group_records, group_files in _build_groups(enumerate(batches), build, workers):
        for record in group_records:
            image_id = record["metadata"]["image_id"]
            if image_id in seen:
                raise ValueError("duplicate image ID")
            seen.add(image_id)
        groups.append(group)
        records.extend(group_records)
        files.update(group_files)
    if not records:
        raise ValueError("cannot build an empty store")
    child_finished = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = ((child_finished.ru_utime+child_finished.ru_stime) -
                 (child_started.ru_utime+child_started.ru_stime)) if workers > 1 else 0.
    manifest = dict(version=VERSION, method=method, variant="native" if method == "fs_native" else "adapted",
                    attribution="paper-guided reimplementation with a JPEG-DCT adapter" if method != "fs_native" else "paper-guided native mechanism",
                    block_size=32 if method == "fs_native" else 8, storage_group_images=effective_group_size,
                    dtype_policy=dtype_policy, codec="numpy.savez_compressed / ZIP_DEFLATED",
                    serialization_order=("image/component/raster-block/stored-mask-order" if method in IC_METHODS else
                                         "image/component/raster-block/ascending-u-within-diagonal"),
                    numpy_version=np.__version__, source_artifact=source_artifact,
                    dependency_evidence=dependency_evidence, images=records, groups=groups, files=files,
                    conversion_wall_s=time.perf_counter()-started, conversion_cpu_s=time.process_time()-cpu_started+child_cpu)
    manifest["content_fingerprint"] = fingerprint({k: v for k, v in manifest.items()
                                                 if k not in ("conversion_wall_s", "conversion_cpu_s")})
    write_json(root / "store.json", manifest)
    return manifest


@dataclass
class Counters:
    application_read_bytes: int = 0
    compressed_payload_read_bytes: int = 0
    metadata_read_bytes: int = 0
    file_opens: int = 0
    read_calls: int = 0
    decoded_q_values: int = 0
    decoded_q_bytes: int = 0
    inflated_members: int = 0
    file_read_wall_s: float = 0.
    inflate_wall_s: float = 0.
    materialize_wall_s: float = 0.
    requested_q_values: int = 0
    unique_requested_q_values: int = 0
    decoded_cache_hits: int = 0
    decoded_cache_misses: int = 0
    cache_evictions: int = 0
    decoded_cache_peak_bytes: int = 0
    active_decoded_peak_bytes: int = 0
    output_peak_bytes: int = 0
    reader_memory_peak_bytes: int = 0
    full_grid_requested_q_values: int = 0

    def add(self, other):
        for key, value in asdict(other).items():
            if key.endswith("peak_bytes"):
                setattr(self, key, max(getattr(self, key), value))
            else:
                setattr(self, key, getattr(self, key)+value)


@dataclass
class ReadPlan:
    requests: tuple[ReadRequest, ...]
    units: tuple[str, ...]
    table_files: tuple[str, ...]
    requested_q_values: int
    unique_requested_q_values: int
    full_grid_requested_q_values: int


@dataclass
class CoefficientView:
    metadata: dict
    coefficients: tuple[np.ndarray, ...]  # [selected_h, selected_w, selected_frequencies]
    tables: tuple[np.ndarray, ...]
    request: ReadRequest

    def dense(self):
        """Return complete natural-order coefficients; may share memory with this view."""
        b = self.metadata["block_size"]
        result = []
        for q, frequencies in zip(self.coefficients, self.request.frequencies):
            if tuple(frequencies) == tuple(range(b*b)):
                result.append(q.reshape(*q.shape[:2], b, b))
                continue
            if set(frequencies) != set(range(b*b)):
                raise ValueError("dense model materialization requires a complete source dependency set")
            value = np.empty((*q.shape[:2], b*b), dtype=q.dtype)
            value[..., list(frequencies)] = q
            result.append(value.reshape(*q.shape[:2], b, b))
        return tuple(result)


class StoreReader:
    def __init__(self, root, *, cache_bytes=0, workers=1, decode_budget_bytes=DEFAULT_DECODE_BUDGET,
                 memory_budget_bytes=DEFAULT_MEMORY_BUDGET, verify_reads=True, profile_stages=False):
        if cache_bytes < 0 or workers < 1 or decode_budget_bytes <= 0 or memory_budget_bytes <= cache_bytes:
            raise ValueError("invalid decoded cache budget/worker count")
        self.root = Path(root).resolve()
        encoded = (self.root / "store.json").read_bytes()
        self.manifest = json.loads(encoded)
        if self.manifest["version"] != VERSION or self.manifest["method"] not in METHODS:
            raise ValueError("unsupported store manifest")
        if "content_fingerprint" not in self.manifest:
            raise ValueError("store lacks payload content identity; rebuild the store")
        expected = fingerprint({k: v for k, v in self.manifest.items()
                                if k not in ("content_fingerprint", "conversion_wall_s", "conversion_cpu_s")})
        if expected != self.manifest["content_fingerprint"]:
            raise ValueError("store content fingerprint differs from its manifest")
        self.identity = dict(manifest=dict(path=str(self.root/"store.json"),
                                          sha256=hashlib.sha256(encoded).hexdigest()), content_fingerprint=expected)
        self.decode_budget_bytes, self.memory_budget_bytes = decode_budget_bytes, memory_budget_bytes
        self.workers = workers
        self.verify_reads = verify_reads
        self.images = {r["metadata"]["image_id"]: r for r in self.manifest["images"]}
        self.units = {u["file"]: u for g in self.manifest["groups"] for u in g["units"]}
        self.cache_bytes, self.cache_size = cache_bytes, 0
        self.profile_stages = profile_stages
        self.cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.initialization_counters = Counters(application_read_bytes=len(encoded), metadata_read_bytes=len(encoded), file_opens=1, read_calls=1)
        self.counters = Counters()

    def _path(self, name):
        path = (self.root / name).resolve()
        if self.root not in path.parents:
            raise ValueError("store file is outside the store directory")
        return path

    def _check_content(self, name, encoded):
        info = self.manifest["files"][name]
        if len(encoded) != info["size_bytes"] or hashlib.sha256(encoded).hexdigest() != info["sha256"]:
            raise ValueError(f"store payload content changed: {name}")

    def content_file_states(self):
        """Cheap change detection for the frozen files covered by a full audit."""
        result = []
        for name in sorted(self.manifest["files"]):
            state = self._path(name).stat()
            result.append([name, state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns])
        return result

    def verify_content(self, evidence=None):
        """Hash once, or reuse that audit only while identity and file states match."""
        started, cpu = time.perf_counter(), time.process_time()
        states = self.content_file_states()
        if evidence is not None:
            if evidence["store"] != self.identity or evidence["file_states"] != states:
                raise ValueError("verified store content or file state changed")
        else:
            for name in self.manifest["files"]:
                self._check_content(name, self._path(name).read_bytes())
            if self.content_file_states() != states:
                raise ValueError("store files changed during content verification")
        return dict(wall_s=time.perf_counter()-started, cpu_core_s=time.process_time()-cpu,
                    bytes=0 if evidence is not None else sum(f["size_bytes"] for f in self.manifest["files"].values()),
                    store=self.identity, file_states=states, reused=evidence is not None)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown()
        self.cache.clear()
        self.cache_size = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def plan(self, requests):
        requests = tuple(requests)
        units, tables, needed = set(), set(), {}
        logical = full = 0
        b = self.manifest["block_size"]
        for request in requests:
            record = self.images[request.image_id]
            metadata = record["metadata"]
            boxes = request.boxes(metadata)
            group = self.manifest["groups"][record["group"]]
            tables.add(group["table_file"])
            for c, (frequencies, box, mask, info) in enumerate(zip(request.frequencies, boxes, record["masks"], metadata["components"])):
                if not set(frequencies).issubset(mask):
                    raise ValueError("request exceeds stored source frequency dependencies")
                y, x, h, w = box
                logical += h*w*len(frequencies)
                full += math.prod(info["block_grid"])*len(frequencies)
                # A bounded preparation-pool union, not a dataset-wide scalar index.
                key = (request.image_id, c)
                needed.setdefault(key, []).append((box, frequencies))
            if self.manifest["method"] in IC_METHODS:
                units.add(group["units"][0]["file"])
            else:
                diagonals = {selection[0] for frequencies in request.frequencies
                             for selection in _fs_selections(b, tuple(frequencies))}
                units.update(u["file"] for u in group["units"] if u["diagonal"] in diagonals)
        unique = 0
        for (image_id, c), selections in needed.items():
            if len(selections) == 1:
                (_, _, h, w), frequencies = selections[0]
                unique += h*w*len(frequencies)
                continue
            mask = np.zeros((*self.images[image_id]["metadata"]["components"][c]["block_grid"], b*b), dtype=bool)
            for (y, x, h, w), frequencies in selections:
                mask[y:y+h, x:x+w, list(frequencies)] = True
            unique += int(np.count_nonzero(mask))
        return ReadPlan(requests, tuple(sorted(units)), tuple(sorted(tables)), logical, unique, full)

    def _decode_file(self, name):
        path = self._path(name)
        started = time.perf_counter() if self.profile_stages else 0.
        encoded = path.read_bytes()
        read_s = time.perf_counter()-started if self.profile_stages else 0.
        if self.verify_reads:
            self._check_content(name, encoded)
        payload_bytes = self.manifest["files"][name]["q_payload_bytes"]
        started = time.perf_counter() if self.profile_stages else 0.
        with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        inflate_s = time.perf_counter()-started if self.profile_stages else 0.
        if any(array.dtype.kind not in "iu" for array in arrays.values()):
            raise ValueError("noninteger storage member")
        counters = Counters(application_read_bytes=len(encoded), compressed_payload_read_bytes=payload_bytes,
                            metadata_read_bytes=len(encoded)-payload_bytes, file_opens=1, read_calls=1,
                            inflated_members=len(arrays), file_read_wall_s=read_s, inflate_wall_s=inflate_s)
        if name in self.units:
            q = arrays["q"]
            unit = self.units[name]
            if q.ndim != 1 or q.size != unit["values"] or q.dtype.str != unit["dtype"]:
                raise ValueError("coefficient member differs from its declared shape/type")
            counters.decoded_q_values, counters.decoded_q_bytes = q.size, q.nbytes
        return arrays, counters

    def read(self, plan):
        counters = Counters(requested_q_values=plan.requested_q_values,
                            unique_requested_q_values=plan.unique_requested_q_values,
                            full_grid_requested_q_values=plan.full_grid_requested_q_values)
        b, ic = self.manifest["block_size"], self.manifest["method"] in IC_METHODS
        tasks, layouts, output_bytes = {}, [], 0
        for request in plan.requests:
            record = self.images[request.image_id]
            group = self.manifest["groups"][record["group"]]
            components = []
            for c, (frequencies, box, info, mask) in enumerate(zip(
                    request.frequencies, request.boxes(record["metadata"]), record["metadata"]["components"], record["masks"])):
                index = record["component_offset"]+c
                selections = () if ic else _fs_selections(b, tuple(frequencies))
                dtype = (np.dtype(group["units"][0]["dtype"]) if ic else
                         np.result_type(*(np.dtype(group["units"][d]["dtype"]) for d, *_ in selections)))
                shape = (box[2], box[3], len(frequencies))
                output_bytes += math.prod(shape)*dtype.itemsize + b*b*2
                components.append((shape, dtype))
                target = (len(layouts), c, index, box, info["block_grid"], frequencies, mask, group)
                tasks.setdefault(group["table_file"], []).append(("tables", target, None))
                if ic:
                    selection = _ic_selection(tuple(mask), tuple(frequencies))
                    tasks.setdefault(group["units"][0]["file"], []).append(("ic", target, selection))
                else:
                    for d, width, destination, source in selections:
                        tasks.setdefault(group["units"][d]["file"], []).append(("fs", target, (width, destination, source)))
            layouts.append(components)
        # Reserve LRU and gathered outputs before admitting any decode futures.
        budget = min(self.decode_budget_bytes, self.memory_budget_bytes-self.cache_bytes-output_bytes)
        names = sorted(set(plan.units) | set(plan.table_files))
        misses = [name for name in names if name not in self.cache]
        if output_bytes+self.cache_size > self.memory_budget_bytes or (misses and (
                budget <= 0 or any(self.manifest["files"][name]["decoded_bytes"] > budget for name in misses))):
            raise MemoryError("reader budget cannot hold requested outputs/cache and one whole NPZ member")
        views = [CoefficientView(self.images[r.image_id]["metadata"],
                    tuple(np.empty(shape, dtype=dtype) for shape, dtype in layout),
                    tuple(np.empty((b, b), dtype=np.uint16) for _ in layout), r)
                 for r, layout in zip(plan.requests, layouts)]
        counters.output_peak_bytes = output_bytes
        counters.decoded_cache_peak_bytes = self.cache_size
        counters.reader_memory_peak_bytes = output_bytes+self.cache_size

        def scatter(name, arrays):
            for kind, (v, c, index, (y, x, h, w), (height, width), frequencies, mask, group), selection in tasks[name]:
                if kind == "tables":
                    views[v].tables[c][...] = arrays["tables"][index]
                elif kind == "ic":
                    offsets = group["value_offsets"]
                    raw = arrays["q"][offsets[index]:offsets[index+1]].reshape(height, width, len(mask))
                    views[v].coefficients[c][...] = raw[y:y+h, x:x+w, selection]
                else:
                    diagonal_width, destination, source = selection
                    offsets = group["block_offsets"]
                    raw = arrays["q"][offsets[index]*diagonal_width:offsets[index+1]*diagonal_width]
                    raw = raw.reshape(height, width, diagonal_width)
                    views[v].coefficients[c][..., destination] = raw[y:y+h, x:x+w, source]

        def decode_chunk(chunk, size):
            counters.active_decoded_peak_bytes = max(counters.active_decoded_peak_bytes, size)
            counters.reader_memory_peak_bytes = max(counters.reader_memory_peak_bytes, output_bytes+self.cache_size+size)
            def process(name):
                arrays, observed = self._decode_file(name)
                scatter(name, arrays)
                # Uncached members die in the worker before its future completes.
                # A completed future then retains counters, not whole groups.
                if self.manifest["files"][name]["decoded_bytes"] > self.cache_bytes:
                    return None, observed
                return arrays, observed
            futures = [self.executor.submit(process, name) for name in chunk] if self.executor else []
            for i, name in enumerate(chunk):
                arrays, observed = futures[i].result() if futures else process(name)
                if futures:
                    futures[i] = None  # A completed future must not retain the full member.
                counters.add(observed)
                if arrays is not None:
                    member_size = sum(a.nbytes for a in arrays.values())
                    while self.cache_size+member_size > self.cache_bytes:
                        _, evicted = self.cache.popitem(last=False)
                        self.cache_size -= sum(a.nbytes for a in evicted.values())
                        counters.cache_evictions += 1
                        del evicted
                    self.cache[name] = arrays
                    self.cache_size += member_size
                    counters.decoded_cache_peak_bytes = max(counters.decoded_cache_peak_bytes, self.cache_size)
                del arrays

        chunk, size = [], 0
        for name in names:
            member_size = self.manifest["files"][name]["decoded_bytes"]
            if chunk and (name in self.cache or len(chunk) == self.workers or size+member_size > budget):
                decode_chunk(chunk, size)
                chunk, size = [], 0
            if name in self.cache:
                counters.decoded_cache_hits += 1
                self.cache.move_to_end(name)
                scatter(name, self.cache[name])
                counters.reader_memory_peak_bytes = max(counters.reader_memory_peak_bytes, output_bytes+self.cache_size)
            else:
                # A preceding miss may have evicted this initially cached member.
                if member_size > budget:
                    raise MemoryError("reader budget cannot hold requested outputs/cache and one whole NPZ member")
                counters.decoded_cache_misses += 1
                chunk.append(name)
                size += member_size
        if chunk:
            decode_chunk(chunk, size)
        self.counters.add(counters)
        return views, counters


def storage_bytes(root):
    """Count archive headers, tables, labels and geometry as well as payloads."""
    root = Path(root)
    manifest = json.loads((root / "store.json").read_text())
    files = {"store.json"} | {g["table_file"] for g in manifest["groups"]} | {
        u["file"] for g in manifest["groups"] for u in g["units"]}
    logical = allocated = payload = 0
    for name in files:
        stat = (root / name).stat()
        logical += stat.st_size
        allocated += stat.st_blocks*512
        if name.endswith(".npz"):
            payload += manifest["files"][name]["q_payload_bytes"]
    return dict(coefficient_data_bytes=payload, metadata_index_label_bytes=logical-payload,
                total_unique_bytes=logical, allocated_disk_bytes=allocated, file_count=len(files))
