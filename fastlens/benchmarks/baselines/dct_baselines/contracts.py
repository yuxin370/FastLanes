"""Versioned source, model and occurrence contracts, independent of storage."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from fastlens.benchmarks.common import sha256_file

VERSION = 1


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path))


def load_artifact(ref):
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return json.loads(path.read_text())


def signed_cast(values, dtype):
    """Range-check before every storage narrowing; never clip or wrap."""
    values = np.asarray(values)
    target = np.dtype(dtype)
    if values.dtype.kind != "i" or target.kind != "i":
        raise TypeError("coefficients require signed integers")
    limits = np.iinfo(target)
    if values.size and (values.min() < limits.min or values.max() > limits.max):
        raise OverflowError(f"coefficient outside {target} range")
    return values.astype(target)


def frequency_set(values, block_size):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(
        type(v) is not int or not 0 <= v < block_size**2 for v in values
    ):
        raise ValueError("frequencies must be unique natural-order indices within the block")
    return values


@dataclass
class SourceImage:
    metadata: dict
    coefficients: tuple[np.ndarray, ...]  # component -> [block_y, block_x, u, v]
    tables: tuple[np.ndarray, ...]

    def __post_init__(self):
        b = self.metadata["block_size"]
        components = self.metadata["components"]
        if len(self.coefficients) != len(components) or len(self.tables) != len(components):
            raise ValueError("coefficient/table/component counts differ")
        for info, q, table in zip(components, self.coefficients, self.tables):
            if q.shape != (*info["block_grid"], b, b) or q.dtype.kind != "i":
                raise ValueError("source coefficients require signed [block_y,block_x,u,v] arrays")
            if table.shape != (b, b) or table.dtype.kind not in "iu" or np.any(table <= 0) or np.any(table > 65535):
                raise ValueError("quantization tables require BxB integers in [1,65535]")


@dataclass(frozen=True)
class ReadRequest:
    image_id: str
    frequencies: tuple[tuple[int, ...], ...]
    # Per-component [y,x,height,width] in blocks. None means the complete grid.
    regions: tuple[tuple[int, int, int, int], ...] | None = None

    def boxes(self, metadata):
        if len(self.frequencies) != len(metadata["components"]):
            raise ValueError("request must specify every component")
        boxes = (tuple((0, 0, *c["block_grid"]) for c in metadata["components"])
                 if self.regions is None else self.regions)
        if len(boxes) != len(self.frequencies):
            raise ValueError("one region is required per component")
        for frequencies, box, component in zip(self.frequencies, boxes, metadata["components"]):
            frequency_set(frequencies, metadata["block_size"])
            y, x, h, w = box
            height, width = component["block_grid"]
            if min(y, x) < 0 or min(h, w) <= 0 or y+h > height or x+w > width:
                raise ValueError("block region is outside the component grid")
        return boxes


def request_for_row(row, profile):
    x, y, width, height = row["crop_box"]
    regions = tuple((y//scale, x//scale, height//scale, width//scale) for scale in (8, 16, 16))
    return ReadRequest(row["image_id"], tuple(tuple(f) for f in profile["source_frequencies"]), regions)


def source_diagonal_frequencies(diagonals):
    """Natural-order B8 coefficients in complete low-frequency diagonals."""
    if type(diagonals) is not int or not 1 <= diagonals <= 15:
        raise ValueError("source-diagonals must be an integer in [1,15]")
    return tuple(f for f in range(64) if f//8 + f%8 < diagonals)


def validate_profile(profile):
    if profile["version"] != VERSION or profile["representation"] != "jpeg_b8":
        raise ValueError("unsupported model profile")
    for frequencies in profile["source_frequencies"]:
        frequency_set(frequencies, 8)
    if not profile["dependency_derivation"]:
        raise ValueError("source frequency dependencies need explicit evidence")
    # Existing resize dependencies remain all64. Partial input is a separately
    # declared lossy source representation, never a smaller dependency closure.
    diagonals = profile.get("source_diagonals", 15)
    selected = source_diagonal_frequencies(diagonals)
    if diagonals < 15:
        if (profile.get("source_zero_fill") != "quantized_before_dequantization" or
                profile["source_frequencies"] != [list(selected)] * 3):
            raise ValueError("partial profiles require the exact diagonal prefix and explicit source q zero-fill")
    elif profile["transform"] in ("cnn_source512", "rgbnomore_source512") and (
        profile["source_frequencies"] != [list(range(64))] * 3
    ):
        raise ValueError("existing resize profiles require all 64 source frequencies")


def load_trace(path, profile, images, labels):
    validate_profile(profile)
    trace = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    occurrences = set()
    for slot, row in enumerate(trace):
        if row["occurrence_id"] in occurrences or row["logical_slot"] != slot:
            raise ValueError("trace needs unique occurrences and contiguous logical slots")
        occurrences.add(row["occurrence_id"])
        if row["image_id"] not in images or row["profile_id"] != profile["id"]:
            raise ValueError("trace image/profile does not resolve")
        if type(row["target_label"]) is not int or not 0 <= row["target_label"] < 1000 or row["target_label"] != labels[row["image_id"]]:
            raise ValueError("trace label differs from the frozen label mapping")
        if profile.get("training"):
            validate_training_row(row, profile)
        elif row["augmentation_parameters"] or row["mixup_partner_occurrence"] is not None or row["mixup_lambda"] != 1:
            raise ValueError("training augmentation requires a frozen training profile")
        x, y, w, h = row["crop_box"]
        # This is the actual supported boundary of the current source512 adapter.
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > 512 or y+h > 512 or any(v % 16 for v in (x, y, w, h)):
            raise ValueError("source512 replay requires in-bounds 16-pixel-aligned crop geometry")
        if row["resize_geometry"] != profile["output_grids"] or (not profile.get("training") and row["target_parameters"] != {}):
            raise ValueError("trace resize/target semantics differ from the frozen profile")
        if type(row["flip"]) is not bool or row["epoch"] < 0 or row["view_id"] < 0 or row["microbatch_id"] < 0:
            raise ValueError("invalid occurrence parameters")
    if not trace:
        raise ValueError("empty request trace")
    if profile.get("training"):
        from itertools import groupby
        for _, batch in groupby(trace, key=lambda r: r["microbatch_id"]):
            batch = list(batch)
            for i, row in enumerate(batch):
                if row["mixup_partner_occurrence"] != batch[(i-1) % len(batch)]["occurrence_id"]:
                    raise ValueError("Mixup partner must roll within the actual retained microbatch")
                if any(row[key] != batch[0][key] for key in ("epoch", "closed_pool_id", "mixup_lambda", "target_parameters")):
                    raise ValueError("microbatch augmentation/target contract is inconsistent")
    return trace


def validate_batches(trace, config, *, training):
    """The same global microbatch and tail contract for calibration and verification."""
    from itertools import groupby
    previous = None
    epoch_slot = 0
    for index, (_, batch) in enumerate(groupby(trace, key=lambda r: r["microbatch_id"])):
        rows = list(batch)
        if rows[0]["microbatch_id"] != index or len(rows) > config["model_microbatch_size"]:
            raise ValueError("microbatch assignment differs from the frozen model batch")
        if previous and len(previous) < config["model_microbatch_size"] and (not training or previous[0]["epoch"] == rows[0]["epoch"]):
            raise ValueError("a partial microbatch is only allowed at the epoch tail")
        if training:
            if previous and previous[0]["epoch"] != rows[0]["epoch"]:
                if rows[0]["epoch"] < previous[0]["epoch"]:
                    raise ValueError("training epochs must be ordered")
                epoch_slot = 0
            if rows[0]["closed_pool_id"] != f"{rows[0]['epoch']}:{epoch_slot//config['preparation_pool_size']}":
                raise ValueError("training pool boundaries differ from the frozen preparation pool")
            epoch_slot += len(rows)
        previous = rows


def validate_training_row(row, profile):
    from fastlens.benchmarks.training_pls.recipe import DCT_RANDAUGMENT_OPERATIONS
    operations = row["augmentation_parameters"]
    if not isinstance(operations, list) or len(operations) != 2:
        raise ValueError("published training requires two explicit RandAugment operations")
    for stage, op in enumerate(operations):
        if op["operation"] not in DCT_RANDAUGMENT_OPERATIONS or op["operation_index"] != stage or not math.isfinite(op["magnitude"]):
            raise ValueError("invalid frozen RandAugment operation")
        name = op["operation"]
        if name == "Cutout":
            if len(op["centers"]) != 3 or len(op["pads"]) != 3:
                raise ValueError("cutout needs an explicit center/pad for every component")
            for center, pad, grid in zip(op["centers"], op["pads"], profile["output_grids"]):
                if len(center) != 2 or any(type(v) is not int or not 0 <= v < grid for v in center) or type(pad) is not int or pad < 0:
                    raise ValueError("cutout geometry outside the profile")
        elif name in ("TranslateX", "TranslateY") and (len(op["blocks"]) != 3 or any(type(v) is not int for v in op["blocks"])):
            raise ValueError("translation needs explicit component block offsets")
        elif name == "ChromaDrop" and op["channel"] not in (0, 1):
            raise ValueError("invalid chroma-drop channel")
    original, rolled = row["mixup_lambda"], row["target_parameters"]["lambda_rolled"]
    if (row["target_parameters"]["classes"] != 1000 or not 0 <= row["target_label"] < 1000 or
            not math.isfinite(original) or not math.isfinite(rolled) or not 0 <= rolled <= original <= 1 or
            abs(original+rolled-1) > 1e-6):
        raise ValueError("invalid frozen Mixup targets")
