#!/usr/bin/env python3
"""Unmodified RGB-no-more ViT composed with an experimental retokenizer."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from galp.experiments.dct_retokenization.token_merge import (
    FixedSpatialProjection,
    LearnedSpatialMerge,
    LearnedSuperPatchProjection,
    MergeGeometry,
    merge_positions,
)


VALID_ARCHITECTURES = ("post_projection", "superpatch")
VALID_POSITION_MODES = ("pooled_existing", "superpatch_center")


def _plainvit(rgbnomore_root: Path):
    root = str(rgbnomore_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("models.plainvit")


def build_pretrained_dct_model(
    rgbnomore_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    plainvit = _plainvit(rgbnomore_root)
    model = plainvit.ViT(
        in_channels=3,
        patch_size=16,
        emb_size=192,
        depth=12,
        n_classes=1000,
        drop_p=0.0,
        device=device,
        dtype=torch.float32,
        num_heads=3,
        head_size=64,
        pixel_space="DCT",
        ver=1,
        use_subblock=True,
    )
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict):
        state = payload.get("model_state_dict", payload.get("model", payload))
    else:
        state = payload
    model.load_state_dict(state, strict=True)
    return model


class DctRetokenizationWrapper(nn.Module):
    """Split the existing patch embedding before position, then retokenize."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        token_count: int = 196,
        merge_axis: str = "height",
        merge_type: str = "fixed",
        architecture: str = "post_projection",
        initialization: str = "average",
        position_mode: str = "pooled_existing",
    ) -> None:
        super().__init__()
        if getattr(base_model, "pixel_space", "").lower() != "dct":
            raise ValueError("DctRetokenizationWrapper requires a DCT ViT")
        self.base_model = base_model
        self.geometry = MergeGeometry(token_count, merge_axis)
        self.merge_type = str(merge_type)
        self.architecture = str(architecture)
        self.initialization = str(initialization)
        self.position_mode = str(position_mode)
        if self.architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"architecture must be one of {VALID_ARCHITECTURES}")
        if self.position_mode not in VALID_POSITION_MODES:
            raise ValueError(f"position_mode must be one of {VALID_POSITION_MODES}")
        if self.position_mode == "superpatch_center" and token_count == 196:
            raise ValueError("superpatch-center position requires N<196")
        projection = self.base_model.patchembed.projection
        dimension = int(projection[0].out_features)
        if self.architecture == "superpatch":
            if self.merge_type != "learned":
                raise ValueError("DCT Super-Patch Projection is a learned projection")
            if token_count != 98:
                raise ValueError("the first DCT Super-Patch Projection supports N=98 only")
            self.retokenizer = LearnedSuperPatchProjection(
                token_count,
                merge_axis,
                int(projection[0].in_features),
                dimension,
                projection[0],
                initialization=self.initialization,
                device=projection[0].weight.device,
                dtype=projection[0].weight.dtype,
            )
        elif self.merge_type == "fixed":
            self.retokenizer = FixedSpatialProjection(
                token_count, merge_axis, self.initialization
            )
        elif self.merge_type == "learned":
            self.retokenizer = LearnedSpatialMerge(
                token_count,
                merge_axis,
                dimension,
                device=projection[0].weight.device,
                dtype=projection[0].weight.dtype,
                initialization=self.initialization,
            )
        else:
            raise ValueError("merge_type must be 'fixed' or 'learned'")
        self._position_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}

    @property
    def token_count(self) -> int:
        return self.geometry.token_count

    @property
    def merge_axis(self) -> str:
        return self.geometry.merge_axis

    def preprojection_features(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        """Return the natural Y/Cb/Cr feature grid before the old final Linear."""

        patch = self.base_model.patchembed
        plainvit = sys.modules[patch.__class__.__module__]
        if patch.combine_Y:
            y = patch.rearrange_Y(y)
            y = plainvit.apply_subblock(y, patch.conv_Y, combine=patch.combine_Y)
        else:
            y = plainvit.apply_subblock(y, patch.conv_Y, combine=patch.combine_Y)
            y = patch.rearrange_Y(y)
        if patch.combine_C:
            cbcr = patch.rearrange_C(cbcr)
            cbcr = plainvit.apply_subblock(cbcr, patch.conv_C, combine=patch.combine_C)
        else:
            cbcr = plainvit.apply_subblock(cbcr, patch.conv_C, combine=patch.combine_C)
            cbcr = patch.rearrange_C(cbcr)
        y = patch.collapser(y)
        cbcr = patch.collapser(cbcr)
        combined = torch.cat((y, cbcr), dim=3)
        if tuple(combined.shape[1:3]) != (14, 14):
            raise RuntimeError(f"expected a 14x14 patch grid, got {tuple(combined.shape)}")
        return combined

    def project_content(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        """Call existing patch submodules but stop before SinCosEmbedding."""

        combined = self.preprojection_features(y, cbcr)
        content = self.base_model.patchembed.projection[0](combined)
        if tuple(content.shape[1:3]) != (14, 14):
            raise RuntimeError(f"expected a 14x14 patch grid, got {tuple(content.shape)}")
        return content

    def original_position_grid(self, content: torch.Tensor) -> torch.Tensor:
        key = (content.device.type, content.device.index, content.dtype)
        cached = self._position_cache.get(key)
        if cached is None:
            seed = torch.zeros(
                (1, 14, 14, content.shape[-1]),
                dtype=content.dtype,
                device=content.device,
            )
            cached = self.base_model.patchembed.projection[1](seed).detach()
            self._position_cache[key] = cached
        return cached

    def superpatch_position_grid(self, content: torch.Tensor) -> torch.Tensor:
        """SinCos positions at centers in the original 14x14 coordinate system."""

        dimension = int(content.shape[-1])
        if dimension % 4:
            raise ValueError("embedding dimension must be divisible by four")
        if self.geometry.token_count != 98:
            raise ValueError("superpatch center positions currently support N=98")
        if self.geometry.merge_axis == "width":
            height_coordinates = torch.arange(14, dtype=content.dtype, device=content.device)
            width_coordinates = torch.arange(7, dtype=content.dtype, device=content.device) * 2 + 0.5
        else:
            height_coordinates = torch.arange(7, dtype=content.dtype, device=content.device) * 2 + 0.5
            width_coordinates = torch.arange(14, dtype=content.dtype, device=content.device)
        hgrid, wgrid = torch.meshgrid(height_coordinates, width_coordinates, indexing="ij")
        frequency = torch.log(torch.tensor(10000, dtype=torch.int32, device=content.device))
        frequency = frequency / (dimension // 4 - 1)
        frequency = torch.exp(
            -torch.arange(dimension // 4, dtype=content.dtype, device=content.device)
            * frequency
        )
        height_phase = torch.einsum("p,f->pf", hgrid.flatten(), frequency)
        width_phase = torch.einsum("p,f->pf", wgrid.flatten(), frequency)
        return torch.cat(
            (
                width_phase.sin(),
                width_phase.cos(),
                height_phase.sin(),
                height_phase.cos(),
            ),
            dim=-1,
        ).reshape(1, *self.geometry.output_grid, dimension)

    def forward_tokens(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        features = self.preprojection_features(y, cbcr)
        if self.architecture == "superpatch":
            # Direct path: never materialize the old [B,14,14,192] token grid.
            merged_content = self.retokenizer(features)
            position = self.original_position_grid(merged_content)
        else:
            content = self.base_model.patchembed.projection[0](features)
            merged_content = self.retokenizer(content)
            position = self.original_position_grid(content)
        merged_position = (
            merge_positions(position, self.geometry)
            if self.position_mode == "pooled_existing"
            else self.superpatch_position_grid(merged_content)
        )
        positioned = merged_content + merged_position
        return positioned.reshape(positioned.shape[0], -1, positioned.shape[-1])

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(y, cbcr)
        encoded = self.base_model.encoder(tokens)
        return self.base_model.classhead(encoded)

    def experiment_state(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "merge_axis": self.merge_axis,
            "merge_type": self.merge_type,
            "architecture": self.architecture,
            "initialization": self.initialization,
            "position_mode": self.position_mode,
        }


def configure_trainable_parameters(wrapper: DctRetokenizationWrapper, mode: str) -> list[str]:
    mode = str(mode)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    if mode == "adapter":
        if not isinstance(wrapper.retokenizer, LearnedSpatialMerge) or wrapper.retokenizer.linear is None:
            raise ValueError("adapter training requires a learned N<196 retokenizer")
        for parameter in wrapper.retokenizer.parameters():
            parameter.requires_grad_(True)
    elif mode == "calibration":
        if not isinstance(wrapper.retokenizer, LearnedSpatialMerge) or wrapper.retokenizer.linear is None:
            raise ValueError("calibration requires a learned N<196 retokenizer")
        for parameter in wrapper.retokenizer.parameters():
            parameter.requires_grad_(True)
        for parameter in wrapper.base_model.classhead.parameters():
            parameter.requires_grad_(True)
    elif mode == "full":
        for parameter in wrapper.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError("mode must be adapter, calibration, or full")
    names = [name for name, parameter in wrapper.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError("training mode selected no parameters")
    return names


def load_experiment_checkpoint(
    wrapper: DctRetokenizationWrapper,
    checkpoint: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = wrapper.experiment_state()
    observed = dict(payload.get("experiment_state") or {})
    # Checkpoints produced before the super-patch study used these implicit
    # defaults.  Canonicalize them rather than invalidating the completed runs.
    observed.setdefault("architecture", "post_projection")
    observed.setdefault("initialization", "average")
    observed.setdefault("position_mode", "pooled_existing")
    observed.pop("position_rule", None)
    if observed != expected:
        raise ValueError(f"retokenization checkpoint mismatch: expected {expected}, got {observed}")
    wrapper.load_state_dict(payload["model_state_dict"], strict=True)
    return payload
