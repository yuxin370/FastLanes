"""Extract original q/Q through libjpeg's coefficient API; never decode pixels."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np

from galp.benchmarks.common import sha256_file
from .contracts import SourceImage, VERSION, artifact, write_json


def read_jpeg(path, image_id, label, split):
    import torch  # Load libc10 before the existing extension.
    import dct_manip
    from PIL import Image

    path = Path(path).resolve()
    with Image.open(path) as jpeg:
        if jpeg.format != "JPEG" or jpeg.mode not in ("L", "YCbCr", "RGB"):
            raise ValueError("only grayscale and three-component JPEG sources are supported")
        width, height = jpeg.size
        layers = jpeg.layer
        if len(layers) not in (1, 3) or (len(layers) == 3 and layers[1][1:3] != layers[2][1:3]):
            raise ValueError("coefficient API requires grayscale or matching chroma sampling")
        header_tables = tuple(np.asarray(jpeg.quantization[c[3]], dtype="<u2").reshape(8, 8) for c in layers)
    _, quant, y, chroma = dct_manip.read_coefficients(str(path))
    coefficients = (y[0].numpy().copy(),) if chroma is None else tuple(
        t.numpy().copy() for t in (y[0], chroma[0], chroma[1]))
    # libjpeg Q is unsigned, while the extension exposes its bits as int16.
    tables = tuple(t.numpy().view(np.uint16).copy() for t in quant)
    max_h, max_v = max(c[1] for c in layers), max(c[2] for c in layers)
    components = []
    for (identifier, hs, vs, table_id), q, table, expected in zip(layers, coefficients, tables, header_tables):
        np.testing.assert_array_equal(table, expected)
        valid_h, valid_w = (height*vs + max_v-1)//max_v, (width*hs + max_h-1)//max_h
        if q.shape[:2] != ((valid_h+7)//8, (valid_w+7)//8):
            raise ValueError("JPEG coefficient grid differs from header geometry")
        components.append(dict(component_id=identifier, sampling=[hs, vs], table_id=table_id,
                               block_grid=list(q.shape[:2]), valid_pixels=[valid_h, valid_w],
                               coefficient_dtype=q.dtype.str,
                               coefficient_sha256=hashlib.sha256(q.astype("<i2").tobytes()).hexdigest()))
    metadata = dict(image_id=str(image_id), label=int(label), split=split, path=str(path),
                    jpeg_sha256=sha256_file(path), width=width, height=height,
                    block_size=8, components=components,
                    padding="retain libjpeg edge blocks; no added MCU dummy blocks",
                    frequency_order="natural-u-major-v-minor-v1")
    return SourceImage(metadata, coefficients, tables)


def freeze_source(selection_path, output, recipe, count=None):
    """Import the actual source selection, without guessing its preprocessing recipe."""
    import json
    payload = json.loads(Path(selection_path).read_text())
    entries = payload["samples"]
    if count is not None:
        if not 0 < count <= len(entries):
            raise ValueError("count must be within the source selection")
        entries = entries[:count]
    if any(recipe.get(key) is None for key in ("resize", "encoder", "quality", "subsampling")):
        raise ValueError("source recipe must explicitly record resize, encoder, quality and subsampling")
    records = []
    for entry in entries:
        image = read_jpeg(entry["path"], entry["logical_sample_id"], entry["label"], payload["split"])
        record = dict(image.metadata, quantization_tables=[t.tolist() for t in image.tables])
        if "galp_image_id" in entry:
            record["galp_image_id"] = entry["galp_image_id"]
        records.append(record)
    if len({r["image_id"] for r in records}) != len(records):
        raise ValueError("source IDs must be unique")
    manifest = dict(version=VERSION, representation="original_jpeg_quantized_dct",
                    selection=artifact(selection_path), preprocessing_recipe=recipe, images=records)
    write_json(output, manifest)
    return manifest


def source_images(manifest):
    for record in manifest["images"]:
        image = read_jpeg(record["path"], record["image_id"], record["label"], record["split"])
        expected = {k: v for k, v in record.items() if k not in ("quantization_tables", "galp_image_id")}
        if image.metadata != expected:
            raise ValueError(f"source changed: {record['image_id']}")
        for table, expected_table in zip(image.tables, record["quantization_tables"]):
            np.testing.assert_array_equal(table, expected_table)
        if "galp_image_id" in record:
            image.metadata["galp_image_id"] = record["galp_image_id"]
        yield image
