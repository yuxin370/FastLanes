"""CPU-DCT, FastLens and IC/FS readers consuming the same occurrence batches."""
from __future__ import annotations

from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from galp.benchmarks.dct_models.online_crop import read_source_jpeg
from .contracts import ReadRequest, artifact
from .materialize import Materializer, PROJECTED_MODELS, cnn_backend
from .source import source_images
from .store import StoreReader
from .training import finish_batch, upstream_ops


def microbatches(rows):
    return [list(group) for _, group in groupby(rows, key=lambda row: row["microbatch_id"])]


def cpu_raw(record, row, profile, dops):
    """Independent JPEG2DCT extraction and literal upstream geometry operations."""
    coefficients, tables = read_source_jpeg(record["path"])
    values = []
    cnn = profile["transform"] == "cnn_source512"
    x, y, width, height = row["crop_box"]
    for c, (q, table, grid) in enumerate(zip(coefficients, tables, profile["output_grids"])):
        q = torch.from_numpy(q).reshape(1, *q.shape[:2], 8, 8)
        table = torch.as_tensor(table.copy()).reshape(8, 8)
        raw = q.float()*table.float() if cnn else (q.to(torch.int16)*table.to(torch.int16)).clamp(-1024, 1016)
        scale = 8 if c == 0 else 16
        cropped = dops.crop_dct(raw, y//scale, x//scale, height//scale, width//scale)
        resized = dops.resize_dct(cropped, grid, dtype=torch.float32, dtype_out=torch.float32, conv_mxs={})
        resized = resized.round().clamp(-32768, 32767) if cnn else resized.round().to(torch.int16).float()
        if row["flip"]:
            resized = dops.flip_dct(resized, direction="horizontal")
        values.append(resized[0])
    return tuple(values)


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
        if kind in ("ic", "fs"):
            self.reader = StoreReader(root, workers=config["workers"])
            if self.reader.manifest["source_artifact"] != config["source_manifest"]:
                raise ValueError("store source identity differs from the frozen source")
            method = self.reader.manifest["method"]
            if (kind == "ic" and not method.startswith("ic_matched_")) or (kind == "fs" and method != "fs_matched_b8"):
                raise ValueError("backend/store representation mismatch")
            self.identity = dict(kind=kind, store=artifact(Path(root)/"store.json"))
        elif kind == "fastlens":
            if self.device.type != "cuda":
                raise ValueError("FastLens replay requires CUDA")
            if any("galp_image_id" not in record for record in source["images"]):
                raise ValueError("FastLens requires an explicit galp_image_id for every frozen source image")
            from galp.profiles import DctModelProfile
            from galp.profiles import rgbnomore
            from galp.torch.direct_dct import DirectDctPipeline
            if profile["transform"] == "cnn_source512":
                native_profile = DctModelProfile(profile["output_grids"][0], tuple((c, f) for c in range(3) for f in range(64)))
            else:
                native_profile = (rgbnomore.TRAINING_PLS if profile.get("training") else
                                  rgbnomore.SWINV2_VALIDATION if profile["model"] == "swinv2" else rgbnomore.VALIDATION)
            with torch.cuda.device(self.device):
                self.pipeline = DirectDctPipeline(root, profile=native_profile,
                    batch_size=config["model_microbatch_size"], module_path=module_path)
            self.identity = dict(kind=kind, manifest=artifact(root))
            # The imported extension path, rather than its module name, is the identity.
            import sys
            native = sys.modules[self.pipeline._native.__class__.__module__]
            self.identity["binding"] = artifact(native.__file__)
        else:
            import jpeg2dct.numpy
            self.identity = dict(kind=kind, decoder=artifact(jpeg2dct.numpy.__file__))

    def verify_source(self):
        """G1: independently verify all selected q/Q and source block geometry."""
        bridge = None
        if self.kind == "fastlens":
            from galp.benchmarks.dct_models.storage import Reader, BRIDGE
            bridge = Reader(self.root, command="read-source-manifest")
            self.identity["coefficient_bridge"] = artifact(BRIDGE)
        try:
            for image in source_images(self.source):
                if bridge:
                    q, tables = bridge.read(image.metadata["galp_image_id"])
                    info = bridge.last_stats["metadata"]
                    expected = {key: image.metadata[key] for key in ("width", "height")}
                    expected["components"] = [{key: component[key] for key in
                                               ("component_id", "sampling", "table_id", "block_grid", "valid_pixels")}
                                              for component in image.metadata["components"]]
                    if info != expected:
                        raise AssertionError("native source component metadata differs from the frozen JPEG")
                    q = [v.reshape(*v.shape[:2], 8, 8) for v in q]
                    tables = [v.reshape(8, 8) for v in tables]
                elif self.reader:
                    record = self.reader.images[image.metadata["image_id"]]["metadata"]
                    if any(record[key] != value for key, value in image.metadata.items()):
                        raise AssertionError("store source metadata differs from the frozen JPEG")
                    req = ReadRequest(image.metadata["image_id"], (tuple(range(64)),)*3)
                    views, _ = self.reader.read(self.reader.plan([req]))
                    q, tables = views[0].dense(), views[0].tables
                else:
                    q, tables = read_source_jpeg(image.metadata["path"])
                    q = [v.reshape(*v.shape[:2], 8, 8) for v in q]
                    tables = [v.reshape(8, 8) for v in tables]
                if len(q) != len(image.coefficients) or len(tables) != len(image.tables):
                    raise AssertionError("component counts differ")
                for a, b, qa, qb in zip(q, image.coefficients, tables, image.tables):
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(qa, qb)
        finally:
            if bridge:
                bridge.close()

    def raw(self, rows):
        if self.reader:
            requests = [ReadRequest(r["image_id"], tuple(tuple(f) for f in self.profile["source_frequencies"])) for r in rows]
            views, _ = self.reader.read(self.reader.plan(requests))
            return [tuple(v.to(self.device) for v in self.materialize.raw(view.dense(), view.tables, row))
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
    from galp.benchmarks.system_rgbnomore.training.model_factory import build_model, VITTI_MODEL_ID, SWINV2_T_MODEL_ID
    model_id = VITTI_MODEL_ID if profile["model"] == "vitti" else SWINV2_T_MODEL_ID
    return build_model(Path(profile["upstream"]), "dct", torch.device(device), model_id)
