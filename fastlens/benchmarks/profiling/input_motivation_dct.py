"""Existing JPEG-to-DCT model/input paths for the bounded motivation study."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset

from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import (
    RgbNoMoreTrainingAdapter, TrainingBatch, TrainingPipelineAdapter,
)

RGBNOMORE = (Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more')))


def build_dct_model(name, device):
    if name in ("vitti-dct", "swinv2-dct"):
        from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import (
            build_model, VITTI_MODEL_ID, SWINV2_T_MODEL_ID,
        )
        return build_model(RGBNOMORE, "dct", device,
                           model_id=VITTI_MODEL_ID if name == "vitti-dct" else SWINV2_T_MODEL_ID)
    if name.startswith("efun"):
        import fastlens.benchmarks.dct_models as package
        from fastlens.benchmarks.dct_models import efun_backend as backend
        # The existing eFUN driver installs this same backend for shared inputs.
        sys.modules["fastlens.benchmarks.dct_models.backend"] = backend
        package.backend = backend
        return backend.upstream_timm().create_model(name.replace("-", "_"), pretrained=False).to(device)
    os.environ["DCTNET_PROFILE"] = name
    from fastlens.benchmarks.dct_models import backend
    # Reuse the official constructor/checkpoint adapter, then reset all learned
    # layers as in the repository's from-scratch CNN training driver.
    model = backend.model()
    torch.manual_seed(11997733)
    for module in model.modules():
        if list(module.parameters(recurse=False)):
            module.reset_parameters()
    return model.requires_grad_(True).to(device)


def worker_init(_):
    torch.set_num_threads(1)


class CnnDctDataset(Dataset):
    def __init__(self, plan, mode):
        from fastlens.benchmarks.dct_models import backend
        self.plan, self.mode = plan, mode
        self.reference = None
        self.mean, self.std = backend.MEAN, backend.STD
        if mode == "training":
            from fastlens.benchmarks.dct_models.training_jpeg import JpegDctDataset
            records = [(s.logical_sample_id, s.label, str(s.path)) for s, _, _ in plan]
            channels = [(component, frequency) for component, frequencies in enumerate(backend.INDICES)
                        for frequency in frequencies]
            self.raw = JpegDctDataset(records, list(range(len(records))), 2, channels, backend.GRID)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        sample, _, _ = self.plan[index]
        if self.mode == "training":
            x = self.raw[index][0]
            # The bounded study uses crop/flip and model normalization, with
            # hard labels; it does not add the PLS RandAugment/Mixup stage.
            x = (x.clamp(-1024, 1016) - self.mean) / self.std
        else:
            if self.reference is None:
                from fastlens.benchmarks.dct_models import backend
                self.reference = backend.Reference()
            x = self.reference(str(sample.path))
        return x, sample.label, index


class CnnDctAdapter(TrainingPipelineAdapter):
    pipeline = "pytorch"
    domain = "dct"

    def begin(self, identities, decisions, batch_lengths=None):
        super().begin(identities, decisions, batch_lengths)
        loader = DataLoader(CnnDctDataset(self._planned, self.config["mode"]),
                            batch_sampler=self._batch_ranges, num_workers=self.workers,
                            pin_memory=self.device.type == "cuda", persistent_workers=self.workers > 0,
                            prefetch_factor=2 if self.workers else None, worker_init_fn=worker_init)
        self._iterator = iter(loader)

    def next_batch(self):
        x, labels, indices = next(self._iterator)
        return TrainingBatch(inputs=(x,), labels=labels.long(),
                             identities=[self._planned[index][1] for index in indices.tolist()],
                             augmentations=[], on_device=False, stage_seconds={})


def dct_adapter(name, mode, samples, device):
    cls = RgbNoMoreTrainingAdapter if name in ("vitti-dct", "swinv2-dct") else CnnDctAdapter
    return cls(samples, batch_size=64, workers=16, device=device,
               config={"rgbnomore_root": str(RGBNOMORE), "mode": mode})


def preprocessing(name, mode):
    if name in ("vitti-dct", "swinv2-dct"):
        geometry = "keyed DCT crop/flip" if mode == "training" else "center-square DCT crop"
        return (f"RGB-no-more direct JPEG coefficients, dequantization/clamp, {geometry}, "
                "Y28/CbCr14 DCT resize, (x+4)/1020 normalization; hard labels, no RandAugment/Mixup")
    if mode == "training":
        return ("existing JpegDctDataset: JPEG coefficient decode, per-sample DCT crop/resize/flip, "
                "rounding and clamp[-1024,1016], model-specific channel selection/normalization; "
                "hard labels, no RandAugment/Mixup")
    return ("existing author reference: RGB decode, model-specific resize/crop and JPEG roundtrip, "
            "JPEG-to-DCT conversion, channel selection and model-specific normalization")
