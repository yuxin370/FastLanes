"""Native IC transform and FS B32/Q50 inputs, separate from JPEG adapters."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np

from .contracts import SourceImage, signed_cast


@dataclass(frozen=True)
class NativeSpec:
    block_size: int = 32
    k: int = 32
    sampling: tuple[int, int] = (1, 1)  # row/column strides, shared by all components
    scalar_quantization: int = 50

    def __post_init__(self):
        if self.block_size not in (8, 16, 32, 64, 128, 256) or not 1 <= self.k <= self.block_size:
            raise ValueError("native IC requires a paper-domain B and 1 <= k <= B")
        if self.sampling not in ((1, 1), (2, 1), (1, 2), (2, 2)) or self.scalar_quantization not in (20, 50, 100):
            raise ValueError("sampling/quantization outside the native IC candidate domain")

    def contract(self):
        return dict(asdict(self), color="RGB[0,255] -> Y,Cr,Cb BT.601 full range; subtract 128 from all planes",
                    padding="edge replication to a multiple of B after sampling",
                    dct="FP64 orthonormal DCT-II", rounding="nearest ties-to-even",
                    output="int32 three-plane triangular k-by-k block mosaic")


def candidates():
    return [NativeSpec(b, k, sampling, q) for sampling in ((1, 1), (2, 1), (1, 2), (2, 2))
            for b in (8, 16, 32, 64, 128, 256) for k in range(1, b+1) for q in (20, 50, 100)]


def dct_basis(b):
    u = np.arange(b)[:, None]
    basis = np.sqrt(2/b) * np.cos(np.pi * (2*np.arange(b)+1) * u / (2*b))
    basis[0] /= np.sqrt(2)
    return basis


def quantize_rgb(rgb, spec):
    """Batched matrix DCT; no per-block or per-coefficient Python hot loop."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("native codec requires HWC uint8 RGB in [0,255]")
    r, g, b = np.moveaxis(rgb.astype(np.float64), -1, 0)
    y = .299*r + .587*g + .114*b
    planes = np.stack((y, (r-y)*.713+128, (b-y)*.564+128)) - 128
    sy, sx = spec.sampling
    planes = planes[:, ::sy, ::sx]
    height, width = planes.shape[1:]
    block = spec.block_size
    planes = np.pad(planes, ((0, 0), (0, (-height) % block), (0, (-width) % block)), mode="edge")
    blocks = planes.reshape(3, planes.shape[1]//block, block, planes.shape[2]//block, block).transpose(0, 1, 3, 2, 4)
    basis = dct_basis(block)
    transformed = basis @ blocks @ basis.T
    rounded = np.rint(transformed / spec.scalar_quantization)
    if np.any(rounded < np.iinfo(np.int32).min) or np.any(rounded > np.iinfo(np.int32).max):
        raise OverflowError("native coefficient exceeds int32")
    return rounded.astype("<i4")


def mosaic(coefficients, k):
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 5 or coefficients.shape[0] != 3 or coefficients.shape[-1] != coefficients.shape[-2]:
        raise ValueError("mosaic requires [3,block_y,block_x,B,B]")
    if not 1 <= k <= coefficients.shape[-1]:
        raise ValueError("triangle k outside block")
    cropped = coefficients[..., :k, :k].copy()
    u, v = np.indices((k, k))
    cropped[..., u+v >= k] = 0
    return cropped.transpose(0, 1, 3, 2, 4).reshape(3, coefficients.shape[1]*k, coefficients.shape[2]*k)


def encode_ic(rgb, spec):
    return signed_cast(mosaic(quantize_rgb(rgb, spec), spec.k), "<i4")


def write_ic_archive(path, rgbs, spec):
    """One image or one explicit IC-b batch, with a single compressed member."""
    write_ic_mosaics(path, [encode_ic(rgb, spec) for rgb in rgbs], spec)


def write_ic_mosaics(path, values, spec):
    """Write already quantized mosaics without recomputing a shared native DCT."""
    values = [signed_cast(value, "<i4") for value in values]
    if not values:
        raise ValueError("empty native IC archive")
    shapes = np.asarray([value.shape for value in values], dtype="<i4")
    offsets = np.cumsum([0, *(value.size for value in values)], dtype="<i8")
    contract = json.dumps(spec.contract(), sort_keys=True).encode()
    np.savez_compressed(path, mosaic=np.concatenate([value.reshape(-1) for value in values]),
                        shapes=shapes, offsets=offsets, contract=np.frombuffer(contract, dtype=np.uint8))


def read_ic_archive(path, spec):
    """Restore int32 mosaics without dequantization, IDCT or RGB reconstruction."""
    with np.load(path, allow_pickle=False) as arrays:
        contract = json.loads(arrays["contract"].tobytes())
        if contract != json.loads(json.dumps(spec.contract())):
            raise ValueError("native IC preprocessing contract differs")
        payload, shapes, offsets = arrays["mosaic"], arrays["shapes"], arrays["offsets"]
        if payload.dtype != np.dtype("<i4"):
            raise ValueError("native IC mosaic must be int32")
        return [payload[offsets[i]:offsets[i+1]].reshape(shape).copy() for i, shape in enumerate(shapes)]


def fit_statistics(mosaics, *, split):
    """Population statistics per plane, including structural zeros, train only."""
    if split != "train":
        raise ValueError("native normalization statistics must come from the training split")
    count = 0
    total, squared = np.zeros(3), np.zeros(3)
    for value in mosaics:
        value = np.asarray(value, dtype=np.float64).reshape(3, -1)
        count += value.shape[1]
        total += value.sum(axis=1)
        squared += (value*value).sum(axis=1)
    if count == 0:
        raise ValueError("empty native statistics population")
    mean = total/count
    std = np.sqrt(np.maximum(squared/count-mean*mean, 0))
    if np.any(std == 0):
        raise ValueError("native normalization has a zero-variance plane")
    return dict(split=split, axis="per-plane across images and mosaic spatial positions, structural zeros included",
                count_per_plane=count, mean=mean.tolist(), std=std.tolist())


def normalize_mosaic(value, statistics):
    mean, std = np.asarray(statistics["mean"], dtype=np.float32), np.asarray(statistics["std"], dtype=np.float32)
    return (np.asarray(value, dtype=np.float32)-mean[:, None, None])/std[:, None, None]


def fs_source(rgb, image_id, label=0, split="native_sanity"):
    spec = NativeSpec()
    q = quantize_rgb(rgb, spec)
    digest = hashlib.sha256(json.dumps(spec.contract(), sort_keys=True).encode()).hexdigest()
    metadata = dict(image_id=str(image_id), label=label, split=split, block_size=32,
                    representation="native_b32_q50", native_contract_sha256=digest,
                    components=[dict(component_id=c, block_grid=list(q.shape[1:3])) for c in range(3)])
    return SourceImage(metadata, tuple(q), tuple(np.full((32, 32), 50, dtype="<u2") for _ in range(3)))
