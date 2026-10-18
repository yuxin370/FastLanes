"""IC image-major and FS whole-diagonal NPZ storage, planning and accounting."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import io
import itertools
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .contracts import VERSION, ReadRequest, artifact, frequency_set, signed_cast, write_json

IC_METHODS = ("ic_matched_shared", "ic_matched_fixedk", "ic_matched_batched")
METHODS = (*IC_METHODS, "fs_matched_b8", "fs_native")


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
        return dict(store=artifact(Path(root)/"store.json"), source=reader.manifest["source_artifact"],
                    method=method, decoded_cache_bytes=0, reader_scope="whole-member-per-sample")


def diagonal_frequencies(block_size, diagonal):
    return tuple(u*block_size + diagonal-u for u in range(block_size) if 0 <= diagonal-u < block_size)


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


def build_store(images, root, method, *, group_size=256, source_frequencies=None,
                dependency_evidence=None, dtype_policy="paper", source_artifact=None):
    if method not in METHODS or group_size <= 0:
        raise ValueError("unknown method or invalid storage group size")
    if dtype_policy not in ("paper", "int16", "safe"):
        raise ValueError("dtype_policy must be paper, int16 or safe")
    if method == "ic_matched_fixedk" and (source_frequencies is None or not dependency_evidence):
        raise ValueError("FixedK needs explicit source dependencies and evidence")
    if source_frequencies is not None and method != "ic_matched_fixedk":
        raise ValueError("only the explicit FixedK variant prunes matched source frequencies")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    effective_group_size = group_size if method in ("ic_matched_batched", "fs_matched_b8", "fs_native") else 1
    groups, records, seen = [], [], set()
    iterator = iter(images)
    block_size = None
    while batch := list(itertools.islice(iterator, effective_group_size)):
        group_id = len(groups)
        offsets = [0]
        table_parts = []
        for image in batch:
            b = image.metadata["block_size"]
            if block_size is None:
                block_size = b
            if b != block_size or (method == "fs_native" and b != 32) or (method != "fs_native" and b != 8):
                raise ValueError("native FS requires B32; matched adapters require JPEG B8")
            if method == "fs_native" and image.metadata.get("representation") != "native_b32_q50":
                raise ValueError("FS native needs separately quantized B32/Q50 input")
            if method != "fs_native" and image.metadata.get("representation", "jpeg_b8") != "jpeg_b8":
                raise ValueError("native coefficients cannot enter a matched store")
            if image.metadata["image_id"] in seen:
                raise ValueError("duplicate image ID")
            seen.add(image.metadata["image_id"])
            masks = ([list(range(b*b)) for _ in image.coefficients] if source_frequencies is None else source_frequencies)
            if len(masks) != len(image.coefficients):
                raise ValueError("one source frequency mask is required per component")
            masks = [list(frequency_set(mask, b)) for mask in masks]
            records.append(dict(metadata=image.metadata, group=group_id,
                                component_offset=offsets[-1], masks=masks,
                                mask_family=[mask_family(mask, b) for mask in masks]))
            offsets.append(offsets[-1]+len(image.coefficients))
            table_parts.extend(image.tables)
        group_records = records[-len(batch):]
        tables = np.stack(table_parts).astype("<u2")
        # Counts of spatial blocks in image/component order. The same prefix
        # describes all FS columns; no spatial or per-scalar seek index exists.
        blocks = [q.shape[0]*q.shape[1] for image in batch for q in image.coefficients]
        group = dict(images=[i.metadata["image_id"] for i in batch], block_offsets=np.cumsum([0, *blocks]).tolist(), units=[])
        if method in IC_METHODS:
            parts, value_offsets = [], [0]
            for image, record in zip(batch, group_records):
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
            write_npz(root / filename, q=payload, tables=tables)
            group.update(value_offsets=value_offsets, table_file=filename)
            group["units"].append(dict(file=filename, diagonal=None, dtype=payload.dtype.str, values=payload.size))
        else:
            table_file = f"group_{group_id:06d}_tables.npz"
            write_npz(root / table_file, tables=tables)
            group["table_file"] = table_file
            for d in range(2*b-1):
                diagonal = diagonal_frequencies(b, d)
                parts = [q.reshape(*q.shape[:2], b*b)[..., diagonal].reshape(-1) for image in batch for q in image.coefficients]
                payload = np.concatenate(parts)
                dtype = "i1" if dtype_policy != "int16" and payload.min() >= -128 and payload.max() <= 127 else "<i2"
                payload = signed_cast(payload, dtype)
                filename = f"group_{group_id:06d}_d{d:03d}.npz"
                write_npz(root / filename, q=payload)
                group["units"].append(dict(file=filename, diagonal=d, dtype=payload.dtype.str, values=payload.size))
        groups.append(group)
    if not records:
        raise ValueError("cannot build an empty store")
    manifest = dict(version=VERSION, method=method, variant="native" if method == "fs_native" else "adapted",
                    attribution="paper-guided reimplementation with a JPEG-DCT adapter" if method != "fs_native" else "paper-guided native mechanism",
                    block_size=block_size, storage_group_images=effective_group_size,
                    dtype_policy=dtype_policy, codec="numpy.savez_compressed / ZIP_DEFLATED",
                    serialization_order="image/component/raster-block/ascending-u-within-diagonal",
                    numpy_version=np.__version__, source_artifact=source_artifact,
                    dependency_evidence=dependency_evidence, images=records, groups=groups,
                    conversion_wall_s=time.perf_counter()-started, conversion_cpu_s=time.process_time()-cpu_started)
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
    requested_q_values: int = 0
    unique_requested_q_values: int = 0
    decoded_cache_hits: int = 0
    decoded_cache_misses: int = 0
    cache_evictions: int = 0
    decoded_cache_peak_bytes: int = 0

    def add(self, other):
        for key, value in asdict(other).items():
            if key == "decoded_cache_peak_bytes":
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


@dataclass
class CoefficientView:
    metadata: dict
    coefficients: tuple[np.ndarray, ...]  # [selected_h, selected_w, selected_frequencies]
    tables: tuple[np.ndarray, ...]
    request: ReadRequest

    def dense(self):
        """Reconstruct only an explicitly complete request; omitted q is not zero."""
        b = self.metadata["block_size"]
        result = []
        for q, frequencies in zip(self.coefficients, self.request.frequencies):
            if set(frequencies) != set(range(b*b)):
                raise ValueError("dense model materialization requires a complete source dependency set")
            value = np.empty((*q.shape[:2], b*b), dtype=q.dtype)
            value[..., list(frequencies)] = q
            result.append(value.reshape(*q.shape[:2], b, b))
        return tuple(result)


class StoreReader:
    def __init__(self, root, *, cache_bytes=0, workers=1):
        if cache_bytes < 0 or workers < 1:
            raise ValueError("invalid decoded cache budget/worker count")
        self.root = Path(root).resolve()
        encoded = (self.root / "store.json").read_bytes()
        self.manifest = json.loads(encoded)
        if self.manifest["version"] != VERSION or self.manifest["method"] not in METHODS:
            raise ValueError("unsupported store manifest")
        self.images = {r["metadata"]["image_id"]: r for r in self.manifest["images"]}
        self.units = {u["file"]: u for g in self.manifest["groups"] for u in g["units"]}
        self.cache_bytes, self.cache_size = cache_bytes, 0
        self.cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.initialization_counters = Counters(application_read_bytes=len(encoded), metadata_read_bytes=len(encoded), file_opens=1, read_calls=1)
        self.counters = Counters()

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
        logical = 0
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
                # A bounded preparation-pool union, not a dataset-wide scalar index.
                key = (request.image_id, c)
                if key not in needed:
                    needed[key] = np.zeros((*info["block_grid"], b*b), dtype=bool)
                needed[key][y:y+h, x:x+w, list(frequencies)] = True
            if self.manifest["method"] in IC_METHODS:
                units.add(group["units"][0]["file"])
            else:
                diagonals = {f//b+f % b for mask in request.frequencies for f in mask}
                units.update(u["file"] for u in group["units"] if u["diagonal"] in diagonals)
        return ReadPlan(requests, tuple(sorted(units)), tuple(sorted(tables)), logical,
                        int(sum(np.count_nonzero(value) for value in needed.values())))

    def _decode_file(self, name):
        path = (self.root / name).resolve()
        if self.root not in path.parents:
            raise ValueError("store file is outside the store directory")
        encoded = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            payload_bytes = sum(info.compress_size for info in archive.infolist() if info.filename == "q.npy")
        with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if any(array.dtype.kind not in "iu" for array in arrays.values()):
            raise ValueError("noninteger storage member")
        counters = Counters(application_read_bytes=len(encoded), compressed_payload_read_bytes=payload_bytes,
                            metadata_read_bytes=len(encoded)-payload_bytes, file_opens=1, read_calls=1)
        if name in self.units:
            q = arrays["q"]
            unit = self.units[name]
            if q.ndim != 1 or q.size != unit["values"] or q.dtype.str != unit["dtype"]:
                raise ValueError("coefficient member differs from its declared shape/type")
            counters.decoded_q_values, counters.decoded_q_bytes = q.size, q.nbytes
        return arrays, counters

    def read(self, plan):
        counters = Counters(requested_q_values=plan.requested_q_values,
                            unique_requested_q_values=plan.unique_requested_q_values)
        decoded, misses = {}, []
        for name in sorted(set(plan.units) | set(plan.table_files)):
            if name in self.cache:
                decoded[name] = self.cache[name]
                self.cache.move_to_end(name)
                counters.decoded_cache_hits += 1
            else:
                misses.append(name)
                counters.decoded_cache_misses += 1
        results = map(self._decode_file, misses) if self.executor is None else self.executor.map(self._decode_file, misses)
        for name, (arrays, observed) in zip(misses, results):
            decoded[name] = arrays
            counters.add(observed)
            size = sum(a.nbytes for a in arrays.values())
            if size <= self.cache_bytes:
                while self.cache_size+size > self.cache_bytes:
                    _, evicted = self.cache.popitem(last=False)
                    self.cache_size -= sum(a.nbytes for a in evicted.values())
                    counters.cache_evictions += 1
                self.cache[name] = arrays
                self.cache_size += size
                counters.decoded_cache_peak_bytes = max(counters.decoded_cache_peak_bytes, self.cache_size)
        views = []
        b = self.manifest["block_size"]
        for request in plan.requests:
            record = self.images[request.image_id]
            metadata = record["metadata"]
            group = self.manifest["groups"][record["group"]]
            all_tables = decoded[group["table_file"]]["tables"]
            parts, tables = [], []
            for c, (frequencies, box, component, mask) in enumerate(zip(request.frequencies, request.boxes(metadata), metadata["components"], record["masks"])):
                index = record["component_offset"]+c
                height, width = component["block_grid"]
                y, x, h, w = box
                if self.manifest["method"] in IC_METHODS:
                    offsets = group["value_offsets"]
                    raw = decoded[group["units"][0]["file"]]["q"][offsets[index]:offsets[index+1]]
                    raw = raw.reshape(height, width, len(mask))
                    part = raw[y:y+h, x:x+w, [mask.index(f) for f in frequencies]].copy()
                else:
                    arrays = []
                    # At most 15/63 diagonals. All blocks/images of each touched
                    # member were already decoded; selection happens afterwards.
                    for d in sorted({f//b+f % b for f in frequencies}):
                        diag = diagonal_frequencies(b, d)
                        unit = group["units"][d]
                        offsets = group["block_offsets"]
                        raw = decoded[unit["file"]]["q"][offsets[index]*len(diag):offsets[index+1]*len(diag)]
                        raw = raw.reshape(height, width, len(diag))
                        positions = [i for i, f in enumerate(frequencies) if f in diag]
                        arrays.append((positions, raw[y:y+h, x:x+w, [diag.index(frequencies[i]) for i in positions]]))
                    dtype = np.result_type(*(array.dtype for _, array in arrays))
                    part = np.empty((h, w, len(frequencies)), dtype=dtype)
                    for positions, array in arrays:
                        part[..., positions] = array
                parts.append(part)
                tables.append(all_tables[index].copy())
            views.append(CoefficientView(metadata, tuple(parts), tuple(tables), request))
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
            with zipfile.ZipFile(root / name) as archive:
                payload += sum(info.compress_size for info in archive.infolist() if info.filename == "q.npy")
    return dict(coefficient_data_bytes=payload, metadata_index_label_bytes=logical-payload,
                total_unique_bytes=logical, allocated_disk_bytes=allocated, file_count=len(files))
