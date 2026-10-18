"""Reuse the repository's model/resize implementations with frozen semantics."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch

from .contracts import VERSION, artifact, validate_profile
from galp.benchmarks.dct_models.dct_geometry import load_upsample_dct

CNN_MODELS = ("mobilenet24", "mobilenet32", "resnet24", "resnet64")
PROJECTED_MODELS = (*CNN_MODELS, "efun")
MODELS = (*PROJECTED_MODELS, "vitti", "swinv2")


def cnn_backend(name):
    if name == "efun":
        from galp.benchmarks.dct_models import efun_backend
        return efun_backend
    os.environ["DCTNET_PROFILE"] = name
    from galp.benchmarks.dct_models import backend
    if backend.PROFILE != name:
        raise RuntimeError("run each DCTNet profile in a separate process, as in existing runners")
    return backend


def operation_order(profile):
    if profile["transform"] == "cnn_source512":
        order = ["dequantize_fp32", "crop", "spectral_resize_fp32", "round_even",
                 "saturate_int16", "flip_blocks_and_odd_v_sign"]
        normalization = ["select_channels", "normalize_fp32"]
    else:
        order = ["dequantize_int16_as_existing_reference", "clamp_-1024_1016", "crop",
                 "spectral_resize_fp32", "round_even_to_int16", "flip_blocks_and_odd_v_sign"]
        normalization = ["organize_y_cbcr", "affine_(x+4)/1020_fp32" if profile.get("training") else "affine_-1_1"]
    if profile.get("training"):
        order += ["round_even_clamp_-1024_1016_to_int16", "published_randaugment_frozen_clamp_after_each_operation"]
    order += normalization
    if profile.get("training"):
        order += ["frozen_mixup_inputs_and_soft_targets"]
    return order


def resolve_profile(name, upstream, checkpoint=None):
    upstream = Path(upstream).resolve()
    profile = dict(version=VERSION, id=name, representation="jpeg_b8", model=name,
                   upstream=str(upstream), source_frequencies=[list(range(64))]*3,
                   dependency_derivation="full-source reference: spectral resize mixes source frequencies before final output selection",
                   dependency_evidence=artifact(Path(__file__).parents[1] / "dct_models/online_crop.py"),
                   resize_implementation=artifact(upstream / "utils/dct_ops.py"),
                   input_geometry="512x512 YCbCr 4:2:0", dtype="float32")
    if name in PROJECTED_MODELS:
        b = cnn_backend(name)
        path = Path(checkpoint) if checkpoint is not None else b.CHECKPOINT
        profile.update(transform="cnn_source512", output_channels=[list(divmod(c, 64)) for c in b.CHANNELS],
                       output_grids=[b.GRID]*3, mean=b.MEAN.flatten().tolist(), std=b.STD.flatten().tolist(),
                       tensors=[dict(name="projected", shape=[len(b.CHANNELS), b.GRID, b.GRID], layout="CHW")],
                       default_crop=[32, 32, 448, 448], label_policy="sorted_class_names",
                       operation_order=["dequantize_fp32", "crop", "spectral_resize_fp32", "round_even",
                                        "saturate_int16", "flip_blocks_and_odd_v_sign", "select_channels", "normalize"],
                       repository_profile=b.profile(), model_adapter=artifact(Path(b.__file__)))
    elif name in ("vitti", "swinv2"):
        from galp.benchmarks.system_rgbnomore.inference import model_factory as factory
        from galp.benchmarks.system_rgbnomore.inference.run import DEFAULT_CHECKPOINT_DIR
        model_id = factory.VITTI_MODEL_ID if name == "vitti" else factory.SWINV2_T_MODEL_ID
        spec = factory.resolve_model(model_id)
        path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT_DIR / spec.dct_checkpoint_name
        profile.update(transform="rgbnomore_source512", model_id=model_id,
                       output_channels=[list(range(64))]*3, output_grids=[spec.y_shape[1], spec.cbcr_shape[1], spec.cbcr_shape[1]],
                       tensors=[dict(name="y", shape=list(spec.y_shape), layout="C,block_y,block_x,u,v"),
                                dict(name="cbcr", shape=list(spec.cbcr_shape), layout="C,block_y,block_x,u,v")],
                       default_crop=[32, 32, 448, 448] if name == "vitti" else [0, 0, 512, 512],
                       label_policy="source_label", model_adapter=artifact(Path(factory.__file__)),
                       reference_dependencies=[artifact(upstream / "datasets.py"), artifact(upstream / "utils/custom_transforms.py")],
                       operation_order=["dequantize_int16_as_existing_reference", "clamp_-1024_1016", "crop",
                                        "spectral_resize_fp32", "round_even_to_int16", "flip_blocks_and_odd_v_sign", "affine_-1_1"])
    else:
        raise ValueError(f"unsupported model: {name}")
    profile["checkpoint"] = artifact(path)
    profile["operation_order"] = operation_order(profile)
    validate_profile(profile)
    return profile


def build_model(profile, device):
    if profile["model"] in PROJECTED_MODELS:
        return cnn_backend(profile["model"]).model(Path(profile["checkpoint"]["path"])).to(device)
    from galp.benchmarks.system_rgbnomore.inference.model_factory import build_dct_model
    return build_dct_model(Path(profile["upstream"]), Path(profile["checkpoint"]["path"]), device, profile["model_id"])


class Materializer:
    def __init__(self, profile):
        validate_profile(profile)
        self.profile = profile
        self.resize = load_upsample_dct(Path(profile["upstream"]), resize=True)
        self.matrices = {}

    def __call__(self, coefficients, tables, row, *, cropped=False):
        output = self.raw(coefficients, tables, row, cropped=cropped)
        return organize(self.profile, output)

    def raw(self, coefficients, tables, row, *, cropped=False):
        # Storage supports ragged/grayscale; these existing model adapters have
        # a narrower, explicitly checked source geometry.
        x, y, width, height = row["crop_box"]
        shapes = [(height//s, width//s) for s in (8, 16, 16)] if cropped else [(64, 64), (32, 32), (32, 32)]
        if len(coefficients) != 3 or [q.shape[:2] for q in coefficients] != shapes:
            raise ValueError("model materialization requires source512 4:2:0")
        cnn = self.profile["transform"] == "cnn_source512"
        output = []
        for c, (q, table, grid) in enumerate(zip(coefficients, tables, self.profile["output_grids"])):
            # RGB-no-more currently multiplies int16 q/Q before clamping.
            # Retain its arithmetic here; do not silently substitute CNN math.
            raw = (torch.as_tensor(q).float() * torch.as_tensor(table).float() if cnn else
                   torch.as_tensor(q).to(torch.int16) * torch.as_tensor(table.astype(np.int16)))
            if not cnn:
                raw = raw.clamp(-1024, 1016)
            scale = 8 if c == 0 else 16
            h, w, cy, cx = height//scale, width//scale, 0 if cropped else y//scale, 0 if cropped else x//scale
            value = self.resize(raw[cy:cy+h, cx:cx+w].reshape(1, h, w, 8, 8), grid,
                                dtype=torch.float32, dtype_out=torch.float32, conv_mxs=self.matrices)
            value = value.reshape(grid, grid, 8, 8)
            if cnn:
                value = value.round().clamp(-32768, 32767)
            else:
                # Upstream resize_dct(dtype_out='keep') restores the int16
                # input type before ToRange. Preserve this rounding boundary.
                value = value.round().to(torch.int16).float()
            if row["flip"]:
                value = value.flip(1) * torch.tensor([1., -1.]*4).reshape(1, 1, 1, 8)
            output.append(value)
        return tuple(output)


def organize(profile, output):
    if profile["transform"] == "cnn_source512":
        value = torch.stack([output[c].flatten(-2)[..., f] for c, f in profile["output_channels"]]).float()
        mean = value.new_tensor(profile["mean"])[:, None, None]
        std = value.new_tensor(profile["std"])[:, None, None]
        return ((value-mean)/std,)
    if profile.get("training"):
        y_value, cb, cr = ((value.float()+4)/1020 for value in output)
    else:
        y_value, cb, cr = (-1.0 + ((value.float()+1024)/2040)*2.0 for value in output)
    return y_value[None], torch.stack((cb, cr))


def source_reference(profile, image, row):
    """Existing independent center-crop reference, never the store's planner."""
    if row["crop_box"] != profile["default_crop"] or row["flip"]:
        # Arbitrary aligned crop/flip is compared separately to a dense spatial
        # oracle in tests. Formal checkpoint gates use the existing validation.
        raise ValueError("checkpoint reference currently verifies the existing validation crop without flip")
    q, tables = image.coefficients, image.tables
    if profile["model"] in PROJECTED_MODELS:
        from galp.benchmarks.dct_models.online_crop import SourceReference
        b = cnn_backend(profile["model"])
        values = SourceReference(Path(profile["upstream"]), b.GRID)(
            tuple(v.reshape(*v.shape[:2], 64) for v in q), tables)
        return (b.organize([v.round().clamp(-32768, 32767) for v in values]),)
    # The upstream test transform is separate from this adapter's crop/resize.
    root = str(Path(profile["upstream"]).resolve())
    sys.path.insert(0, root)
    import importlib
    from galp.benchmarks.system_rgbnomore.inference.model_factory import resolve_model
    datasets = importlib.import_module("datasets")
    if Path(datasets.__file__).resolve().parent != Path(root):
        raise RuntimeError("upstream datasets module collision; use a fresh process per model")
    spec = resolve_model(profile["model_id"])
    transform = datasets.get_transform(dataset=spec.dct_dataset, type="test", dtype=torch.float32, dtype_resize=torch.float32)
    quant = torch.from_numpy(np.stack(tables).astype(np.int16))
    y = (torch.from_numpy(q[0])[None] * quant[0]).clamp(-1024, 1016)
    cbcr = (torch.from_numpy(np.stack(q[1:])) * quant[1:, None, None]).clamp(-1024, 1016)
    return transform((y, cbcr))
