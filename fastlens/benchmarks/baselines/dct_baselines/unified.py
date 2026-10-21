"""CPU-DCT, FastLens and IC/FS readers consuming the same occurrence batches."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
import math
from pathlib import Path

import numpy as np
import torch

from fastlens.benchmarks.dct_models.online_crop import read_source_jpeg
from .contracts import ReadRequest, artifact, request_for_row
from .materialize import Materializer, PROJECTED_MODELS, cnn_backend
from .source import source_images
from .store import StoreReader
from .training import finish_batch, upstream_ops


def microbatches(rows):
    return [list(group) for _, group in groupby(rows, key=lambda row: row["microbatch_id"])]


def cpu_raw(record, row, profile, dops, *, rounded=True):
    """Independent JPEG2DCT extraction and literal upstream geometry operations.

    rounded=False exposes FP32 resize values after flip for failure diagnosis.
    """
    coefficients, tables = read_source_jpeg(record["path"])
    values = []
    cnn = profile["transform"] == "cnn_source512"
    x, y, width, height = row["crop_box"]
    diagonals = profile.get("source_diagonals", 15)
    omitted = [f for f in range(64) if f//8+f%8 >= diagonals] if diagonals < 15 else ()
    for c, (q, table, grid) in enumerate(zip(coefficients, tables, profile["output_grids"])):
        q = torch.from_numpy(q).reshape(1, *q.shape[:2], 8, 8)
        if omitted:
            q = q.clone()
            q.reshape(*q.shape[:3], 64)[..., omitted] = 0
        table = torch.as_tensor(table.copy()).reshape(8, 8)
        raw = q.float()*table.float() if cnn else (q.to(torch.int16)*table.to(torch.int16)).clamp(-1024, 1016)
        scale = 8 if c == 0 else 16
        cropped = dops.crop_dct(raw, y//scale, x//scale, height//scale, width//scale)
        resized = dops.resize_dct(cropped, grid, dtype=torch.float32, dtype_out=torch.float32, conv_mxs={})
        if rounded:
            resized = resized.round().clamp(-32768, 32767) if cnn else resized.round().to(torch.int16).float()
        if row["flip"]:
            resized = dops.flip_dct(resized, direction="horizontal")
        values.append(resized[0])
    return tuple(values)


def _compare_source_coefficients(image, coefficients, tables):
    if len(coefficients) != len(image.coefficients) or len(tables) != len(image.tables):
        raise AssertionError("component counts differ")
    for a, b, qa, qb in zip(coefficients, image.coefficients, tables, image.tables):
        np.testing.assert_array_equal(a, b, err_msg=image.metadata["image_id"])
        np.testing.assert_array_equal(qa, qb, err_msg=image.metadata["image_id"])


def _verify_native_source_chunk(root, records):
    from fastlens.benchmarks.dct_models.storage import Reader
    with Reader(root, command="read-source-manifest") as bridge:
        for image in source_images(dict(images=records)):
            q, tables = bridge.read(image.metadata["galp_image_id"])
            expected = {key: image.metadata[key] for key in ("width", "height")}
            expected["components"] = [{key: component[key] for key in
                ("component_id", "sampling", "table_id", "block_grid", "valid_pixels")}
                for component in image.metadata["components"]]
            if bridge.last_stats["metadata"] != expected:
                raise AssertionError(f"native source component metadata differs: {image.metadata['image_id']}")
            _compare_source_coefficients(image, [v.reshape(*v.shape[:2], 8, 8) for v in q],
                                         [v.reshape(8, 8) for v in tables])
    return len(records)


def verify_native_source(root, source, *, workers=1):
    """Partition exact G1 checks across independent native reader processes."""
    if workers <= 0:
        raise ValueError("source-workers must be positive")
    records = sorted(source["images"], key=lambda record: record["galp_image_id"])
    workers = min(workers, len(records))
    size = (len(records)+workers-1)//workers
    chunks = [records[start:start+size] for start in range(0, len(records), size)]
    print(f"Source q/Q verification: {len(records)} images, {len(chunks)} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_verify_native_source_chunk, root, chunk) for chunk in chunks]
        checked = 0
        for future in as_completed(futures):
            checked += future.result()
            print(f"Source q/Q checked: {checked}/{len(records)}", flush=True)


def native_source_frequencies(root, profile):
    """Resolve natural frequencies to FLS columns by reading metadata headers."""
    if profile.get("source_diagonals", 15) == 15:
        return "all"
    from fastlens.benchmarks.system_dct_major.common import parse_manifest, ZIGZAG_COLUMN_TO_NATURAL_INDEX
    orders = set()
    for shard in parse_manifest(Path(root))["shards"]:
        with Path(shard["metadata_path"]).open("rb") as stream:
            header = stream.read(18)
        layout = {b"GJDCTMD1": (1, 14), b"GJDCTMD3": (3, 16)}.get(header[:8])
        if layout is None or len(header) < layout[1]+2 or int.from_bytes(header[8:10], "little") != layout[0]:
            raise ValueError("unsupported FLS metadata header for source frequency selection")
        orders.add(bool(int.from_bytes(header[layout[1]:layout[1]+2], "little") & 1))
    if len(orders) != 1:
        raise ValueError("partial source selection requires a common coefficient order across FLS shards")
    natural = set(profile["source_frequencies"][0])
    columns = ZIGZAG_COLUMN_TO_NATURAL_INDEX if orders.pop() else range(64)
    return tuple(column for column, frequency in enumerate(columns) if frequency in natural)


class UnifiedReplay:
    def __init__(self, kind, root, config, source, profile, *, device="cpu", module_path=None):
        if kind not in ("cpu_dct", "ic", "fs", "fastlens"):
            raise ValueError("unknown matched backend")
        self.kind, self.root, self.config, self.source, self.profile = kind, root, config, source, profile
        self.records = {row["image_id"]: row for row in source["images"]}
        self.device = torch.device(device)
        self.dops = upstream_ops(profile["upstream"])
        self.materialize = Materializer(profile)
        self.reader = self.pipeline = None
        self.content_verification = None
        if kind in ("ic", "fs"):
            self.reader = StoreReader(root, workers=config["workers"], decode_budget_bytes=config["decode_budget_bytes"],
                                      cache_bytes=config.get("decoded_cache_bytes", 0),
                                      memory_budget_bytes=config["reader_memory_budget_bytes"])
            self.content_verification = self.reader.verify_content(config.get("store_content_evidence"))
            if self.reader.manifest["source_artifact"] != config["source_manifest"]:
                raise ValueError("store source identity differs from the frozen source")
            method = self.reader.manifest["method"]
            if (kind == "ic" and not method.startswith("ic_matched_")) or (kind == "fs" and method != "fs_matched_b8"):
                raise ValueError("backend/store representation mismatch")
            self.identity = dict(kind=kind, store=self.reader.identity)
        elif kind == "fastlens":
            if self.device.type != "cuda":
                raise ValueError("FastLens replay requires CUDA")
            if any("galp_image_id" not in record for record in source["images"]):
                raise ValueError("FastLens requires an explicit fastlens_image_id for every frozen source image")
            from fastlens.profiles import DctModelProfile
            from fastlens.profiles import rgbnomore
            from fastlens.torch.direct_dct import DirectDctPipeline, DirectDctReader
            source_selection = native_source_frequencies(root, profile)
            if profile["transform"] == "cnn_source512":
                native_profile = (DctModelProfile(profile["output_grids"][0], tuple((c, f) for c in range(3) for f in range(64)),
                                                  source_frequency_policy=source_selection)
                                  if profile.get("training") else
                                  DctModelProfile(profile["output_grids"][0], tuple(map(tuple, profile["output_channels"])),
                                                  tuple(zip(profile["mean"], profile["std"])),
                                                  source_frequency_policy=source_selection))
            else:
                native_profile = (rgbnomore.TRAINING_PLS if profile.get("training") else
                                  rgbnomore.SWINV2_VALIDATION if profile["model"] == "swinv2" else rgbnomore.VALIDATION)
            with torch.cuda.device(self.device):
                if source_selection != "all" and profile["transform"] != "cnn_source512":
                    reader = DirectDctReader(root, module_path=module_path)
                    self.pipeline = reader.pipeline(native_profile, coefficients=source_selection)
                else:
                    self.pipeline = DirectDctPipeline(root, profile=native_profile,
                        batch_size=config["model_microbatch_size"], module_path=module_path)
            self.identity = dict(kind=kind, manifest=artifact(root))
            # The imported extension path, rather than its module name, is the identity.
            import sys
            native = sys.modules[self.pipeline._native.__class__.__module__]
            self.identity["binding"] = artifact(native.__file__)
        else:
            import dct_manip
            self.identity = dict(kind=kind, decoder=artifact(dct_manip.__file__))

    def verify_source(self, *, workers=1):
        """G1: independently verify all selected q/Q and source block geometry."""
        if self.kind == "fastlens":
            from fastlens.benchmarks.dct_models.storage import BRIDGE
            verify_native_source(self.root, self.source, workers=workers)
            self.identity["coefficient_bridge"] = artifact(BRIDGE)
            return
        if self.reader:
            self._verify_store_source()
            return
        for image in source_images(self.source):
            q, tables = read_source_jpeg(image.metadata["path"])
            q = [v.reshape(*v.shape[:2], 8, 8) for v in q]
            tables = [v.reshape(8, 8) for v in tables]
            _compare_source_coefficients(image, q, tables)

    def _verify_store_source(self):
        """Coalesce G1 reads within storage groups; retain the reader's memory bound."""
        reader = self.reader
        checked = 0

        def check(records):
            nonlocal checked
            requests = [ReadRequest(r["image_id"], tuple(map(tuple, reader.images[r["image_id"]]["masks"])))
                        for r in records]
            views, _ = reader.read(reader.plan(requests))
            for image, view in zip(source_images(dict(images=records)), views):
                if any(view.metadata[key] != value for key, value in image.metadata.items()):
                    raise AssertionError("store source metadata differs from the frozen JPEG")
                for actual, expected, frequencies, table, expected_table in zip(
                        view.coefficients, image.coefficients, view.request.frequencies, view.tables, image.tables):
                    selected = expected.reshape(*expected.shape[:2], 64)[..., list(frequencies)]
                    np.testing.assert_array_equal(actual, selected, err_msg=image.metadata["image_id"])
                    np.testing.assert_array_equal(table, expected_table, err_msg=image.metadata["image_id"])
            checked += len(records)
            print(f"Source q/Q checked: {checked}/{len(self.source['images'])}", flush=True)

        def group_id(record):
            return reader.images[record["image_id"]]["group"]

        for index, records in groupby(sorted(self.source["images"], key=group_id), key=group_id):
            group = reader.manifest["groups"][index]
            files = {group["table_file"], *(u["file"] for u in group["units"])}
            # Leave room for at least one complete decode; read() admits parallel
            # files into the remaining budget. No cache survives into G2/training.
            largest = max(reader.manifest["files"][name]["decoded_bytes"] for name in files)
            output_budget = reader.memory_budget_bytes-reader.cache_bytes-largest
            dtype = np.result_type(*(np.dtype(u["dtype"]) for u in group["units"]))
            batch, size = [], 0
            for record in records:
                masks = reader.images[record["image_id"]]["masks"]
                needed = sum(math.prod(c["block_grid"])*len(mask)*dtype.itemsize+128
                             for c, mask in zip(record["components"], masks))
                if batch and (len(batch) == self.config["model_microbatch_size"] or size+needed > output_budget):
                    check(batch)
                    batch, size = [], 0
                batch.append(record)
                size += needed
            if batch:
                check(batch)

    def raw(self, rows):
        if self.reader:
            requests = [request_for_row(r, self.profile) for r in rows]
            views, _ = self.reader.read(self.reader.plan(requests))
            return [tuple(v.to(self.device) for v in self.materialize.raw(
                        self.materialize.source_coefficients(view), view.tables, row, cropped=True, source_selected=True))
                    for view, row in zip(views, rows)]
        if self.pipeline:
            ids = [self.records[r["image_id"]]["galp_image_id"] for r in rows]
            transforms = [dict(crop=dict(zip(("x", "y", "width", "height"), r["crop_box"])),
                               horizontal_flip=r["flip"], logical_sample_id=r["image_id"]) for r in rows]
            self.pipeline.start([ids], transforms_by_batch=[transforms])
            values, actual_ids = [], []
            for batch in self.pipeline:
                batch.record_stream(torch.cuda.current_stream(self.device))
                actual_ids.extend(batch.global_image_ids)
                if self.profile["transform"] == "cnn_source512":
                    value = batch.projected.clone()
                    n, _, h, w = value.shape
                    dense = value.reshape(n, 3, 8, 8, h, w).permute(0, 1, 4, 5, 2, 3)
                    values.extend(tuple(v[c] for c in range(3)) for v in dense)
                else:
                    y, cbcr = batch.y, batch.cbcr
                    # TRAINING_PLS exports raw int16, validation exports ToRange float32.
                    if self.profile.get("training"):
                        dense_y, dense_c = y.float(), cbcr.float()
                    else:
                        dense_y = (y*1020-4).round()
                        dense_c = (cbcr*1020-4).round()
                    values.extend((dense_y[i, 0].clone(), dense_c[i, 0].clone(), dense_c[i, 1].clone()) for i in range(len(y)))
            if actual_ids != ids:
                raise AssertionError("native output occurrence order differs from the frozen trace")
            return values
        return [tuple(v.to(self.device) for v in cpu_raw(self.records[r["image_id"]], r, self.profile, self.dops)) for r in rows]

    def inference_batches(self, trace):
        """Keep CPU baseline preprocessing and native CUDA preprocessing distinct.

        Native CNN requests retain whole preparation pools and use projected
        range events; Transformer pools are sliced only at the model boundary.
        """
        if self.profile.get("training"):
            raise ValueError("checkpoint inference requires an inference profile without training augmentation")
        size = self.config["model_microbatch_size"]
        pool_size = self.config["preparation_pool_size"]
        if self.pipeline:
            pools = [trace[start:start+pool_size] for start in range(0, len(trace), pool_size)]
            ids = [[self.records[r["image_id"]]["galp_image_id"] for r in rows] for rows in pools]
            transforms = [[dict(crop=dict(zip(("x", "y", "width", "height"), r["crop_box"])),
                                horizontal_flip=r["flip"], logical_sample_id=r["image_id"]) for r in rows] for rows in pools]
            self.pipeline.start(ids, transforms_by_batch=transforms)
            offset = 0
            for batch in self.pipeline:
                rows = trace[offset:offset+len(batch.global_image_ids)]
                if batch.global_image_ids != [self.records[r["image_id"]]["galp_image_id"] for r in rows]:
                    raise AssertionError("native inference occurrence order differs from the trace")
                batch.record_stream(torch.cuda.current_stream(self.device))
                tensors = ((batch.projected,) if self.profile["transform"] == "cnn_source512" else (batch.y, batch.cbcr))
                for start in range(0, len(rows), size):
                    yield rows[start:start+size], tuple(t[start:start+size] for t in tensors)
                offset += len(rows)
            if offset != len(trace):
                raise AssertionError("native inference dropped trace occurrences")
        else:
            from concurrent.futures import ThreadPoolExecutor
            from .runner import Replay, inference_loader
            if self.config.get("cpu_loader") == "dataloader":
                offset = 0
                loader = inference_loader(self.root if self.reader else None, self.profile, self.config, self.records, trace)
                for batches, _ in loader:
                    for tensors, labels in batches:
                        rows = trace[offset:offset+len(labels)]
                        torch.testing.assert_close(labels, torch.tensor([r["target_label"] for r in rows]), atol=0, rtol=0)
                        yield rows, tuple(t.to(self.device) for t in tensors)
                        offset += len(labels)
                return
            # Verify the same parallel read/materialization used for timing.
            with ThreadPoolExecutor(max_workers=self.config["workers"]) as executor:
                replay = Replay(self.reader, self.profile, self.config, records=self.records, executor=executor)
                for start in range(0, len(trace), pool_size):
                    rows = trace[start:start+pool_size]
                    batches, _ = replay.pool(rows)
                    for offset, (tensors, labels) in zip(range(0, len(rows), size), batches):
                        batch_rows = rows[offset:offset+size]
                        torch.testing.assert_close(labels, torch.tensor([r["target_label"] for r in batch_rows]), atol=0, rtol=0)
                        yield batch_rows, tuple(t.to(self.device) for t in tensors)

    def batch(self, rows, *, oracle=False):
        raw = ([cpu_raw(self.records[r["image_id"]], r, self.profile, self.dops) for r in rows] if oracle else self.raw(rows))
        raw = [tuple(v.to(self.device) for v in sample) for sample in raw]
        return finish_batch(raw, rows, self.profile, self.dops, oracle=oracle)

    def close(self):
        if self.reader:
            self.reader.close()
        if self.pipeline:
            self.pipeline.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def build_training_model(profile, device, seed=11997733):
    if profile["model"] in PROJECTED_MODELS:
        model = cnn_backend(profile["model"]).model(Path(profile["checkpoint"]["path"])).requires_grad_(True).train()
        torch.manual_seed(seed)
        for module in model.modules():
            if list(module.parameters(recurse=False)):
                module.reset_parameters()
        return model.to(device)
    from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import build_model, VITTI_MODEL_ID, SWINV2_T_MODEL_ID
    model_id = VITTI_MODEL_ID if profile["model"] == "vitti" else SWINV2_T_MODEL_ID
    return build_model(Path(profile["upstream"]), "dct", torch.device(device), model_id)
