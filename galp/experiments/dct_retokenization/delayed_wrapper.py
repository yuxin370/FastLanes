#!/usr/bin/env python3
"""Delayed K32/N98 retokenization composed from unmodified RGB-no-more modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from galp.experiments.dct_retokenization.token_merge import LearnedSpatialMerge


VALID_MERGE_AFTER_BLOCKS = (2, 4)


class DelayedRetokenizationWrapper(nn.Module):
    """Run early pretrained blocks at N=196, then learned spatial merge to N=98."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        merge_after_block: int,
        merge_axis: str = "width",
        initialization: str = "keep_first",
    ) -> None:
        super().__init__()
        if getattr(base_model, "pixel_space", "").lower() != "dct":
            raise ValueError("DelayedRetokenizationWrapper requires a DCT ViT")
        self.base_model = base_model
        self.merge_after_block = int(merge_after_block)
        if self.merge_after_block not in VALID_MERGE_AFTER_BLOCKS:
            raise ValueError(
                f"merge_after_block must be one of {VALID_MERGE_AFTER_BLOCKS}"
            )
        blocks = list(self.base_model.encoder.children())
        if len(blocks) != 12:
            raise ValueError(f"expected a 12-block encoder, observed {len(blocks)}")
        projection = self.base_model.patchembed.projection[0]
        dimension = int(projection.out_features)
        self.retokenizer = LearnedSpatialMerge(
            98,
            merge_axis,
            dimension,
            device=projection.weight.device,
            dtype=projection.weight.dtype,
            initialization=initialization,
        )
        self.merge_axis = str(merge_axis)
        self.initialization = str(initialization)
        self.dimension = dimension

    def encoder_blocks(self) -> list[nn.Module]:
        return list(self.base_model.encoder.children())

    def forward_pre_merge(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        tokens = self.base_model.patchembed(y, cbcr)
        if tuple(tokens.shape[1:]) != (196, self.dimension):
            raise RuntimeError(
                f"expected [B,196,{self.dimension}] patch tokens, got {tuple(tokens.shape)}"
            )
        for block in self.encoder_blocks()[: self.merge_after_block]:
            tokens = block(tokens)
        return tokens

    def merge_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (196, self.dimension):
            raise ValueError(
                f"expected [B,196,{self.dimension}] latent tokens, got {tuple(tokens.shape)}"
            )
        grid = tokens.reshape(tokens.shape[0], 14, 14, self.dimension)
        merged = self.retokenizer(grid)
        return merged.reshape(merged.shape[0], 98, self.dimension)

    def forward_tokens(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        tokens = self.merge_latents(self.forward_pre_merge(y, cbcr))
        for block in self.encoder_blocks()[self.merge_after_block :]:
            tokens = block(tokens)
        return tokens

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        return self.base_model.classhead(self.forward_tokens(y, cbcr))

    def experiment_state(self) -> dict[str, Any]:
        return {
            "architecture": "delayed_retokenization",
            "k": 32,
            "token_count": 98,
            "merge_axis": self.merge_axis,
            "merge_after_block": self.merge_after_block,
            "merge_type": "learned_spatial_concatenation",
            "initialization": self.initialization,
        }


def load_delayed_checkpoint(
    wrapper: DelayedRetokenizationWrapper,
    checkpoint: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    observed = payload.get("experiment_state")
    expected = wrapper.experiment_state()
    if observed != expected:
        raise ValueError(
            f"delayed-retokenization checkpoint mismatch: expected {expected}, got {observed}"
        )
    wrapper.load_state_dict(payload["model_state_dict"], strict=True)
    return payload
