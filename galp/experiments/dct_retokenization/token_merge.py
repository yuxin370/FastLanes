#!/usr/bin/env python3
"""Fixed and average-initialized learned spatial token merging."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


VALID_TOKEN_COUNTS = (196, 98, 49)
VALID_MERGE_AXES = ("height", "width")
VALID_INITIALIZATIONS = ("average", "keep_first", "keep_second", "standard")


@dataclass(frozen=True)
class MergeGeometry:
    token_count: int
    merge_axis: str = "height"

    def __post_init__(self) -> None:
        if self.token_count not in VALID_TOKEN_COUNTS:
            raise ValueError(f"token_count must be one of {VALID_TOKEN_COUNTS}")
        if self.merge_axis not in VALID_MERGE_AXES:
            raise ValueError(f"merge_axis must be one of {VALID_MERGE_AXES}")

    @property
    def output_grid(self) -> tuple[int, int]:
        if self.token_count == 196:
            return (14, 14)
        if self.token_count == 98:
            return (7, 14) if self.merge_axis == "height" else (14, 7)
        return (7, 7)

    @property
    def group_size(self) -> int:
        return {196: 1, 98: 2, 49: 4}[self.token_count]


def spatial_groups(tokens: torch.Tensor, geometry: MergeGeometry) -> torch.Tensor:
    """Return [B,H',W',group,D] with an explicit deterministic group order."""

    if tokens.ndim != 4 or tuple(tokens.shape[1:3]) != (14, 14):
        raise ValueError(f"expected [B,14,14,D], got {tuple(tokens.shape)}")
    batch, height, width, dimension = tokens.shape
    if geometry.token_count == 196:
        return tokens.reshape(batch, height, width, 1, dimension)
    if geometry.token_count == 98 and geometry.merge_axis == "height":
        # [z(row=0), z(row=1)] for each output coordinate.
        return (
            tokens.reshape(batch, 7, 2, width, dimension)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    if geometry.token_count == 98:
        # [z(col=0), z(col=1)] for each output coordinate.
        return tokens.reshape(batch, height, 7, 2, dimension).contiguous()
    # [z00,z01,z10,z11], with horizontal position varying fastest.
    return (
        tokens.reshape(batch, 7, 2, 7, 2, dimension)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(batch, 7, 7, 4, dimension)
    )


def fixed_average_merge(tokens: torch.Tensor, geometry: MergeGeometry) -> torch.Tensor:
    if geometry.token_count == 196:
        return tokens
    return spatial_groups(tokens, geometry).mean(dim=-2)


def fixed_spatial_projection(
    tokens: torch.Tensor,
    geometry: MergeGeometry,
    initialization: str,
) -> torch.Tensor:
    """Apply a non-trainable spatial control projection.

    ``average`` is a negative/control baseline.  ``keep_first`` and
    ``keep_second`` preserve one member of each pair and diagnose whether the
    average itself destroys the pretrained token distribution.
    """

    initialization = str(initialization)
    if initialization not in VALID_INITIALIZATIONS[:-1]:
        raise ValueError("fixed projection must be average, keep_first, or keep_second")
    if geometry.token_count == 196:
        return tokens
    groups = spatial_groups(tokens, geometry)
    if initialization == "average":
        return groups.mean(dim=-2)
    if geometry.group_size != 2:
        raise ValueError("keep-first/second controls are defined only for N=98")
    return groups[..., 0 if initialization == "keep_first" else 1, :]


class FixedSpatialMerge(nn.Module):
    def __init__(self, token_count: int, merge_axis: str = "height") -> None:
        super().__init__()
        self.geometry = MergeGeometry(token_count, merge_axis)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return fixed_average_merge(tokens, self.geometry)


class FixedSpatialProjection(nn.Module):
    def __init__(
        self,
        token_count: int,
        merge_axis: str = "height",
        initialization: str = "average",
    ) -> None:
        super().__init__()
        self.geometry = MergeGeometry(token_count, merge_axis)
        self.initialization = str(initialization)
        if self.initialization == "standard":
            raise ValueError("standard initialization requires a learned projection")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return fixed_spatial_projection(tokens, self.geometry, self.initialization)


class LearnedSpatialMerge(nn.Module):
    """Concatenate spatial neighbors and map them back to D dimensions."""

    def __init__(
        self,
        token_count: int,
        merge_axis: str = "height",
        dimension: int = 192,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        initialization: str = "average",
    ) -> None:
        super().__init__()
        self.geometry = MergeGeometry(token_count, merge_axis)
        self.dimension = int(dimension)
        self.initialization = str(initialization)
        if self.initialization not in VALID_INITIALIZATIONS:
            raise ValueError(f"initialization must be one of {VALID_INITIALIZATIONS}")
        if self.geometry.group_size == 1:
            self.linear: nn.Linear | None = None
        else:
            self.linear = nn.Linear(
                self.geometry.group_size * self.dimension,
                self.dimension,
                bias=True,
                device=device,
                dtype=dtype,
            )
            self.reset_projection()

    @torch.no_grad()
    def reset_projection(self) -> None:
        if self.linear is None:
            return
        if self.initialization == "standard":
            self.linear.reset_parameters()
            return
        self.linear.weight.zero_()
        identity = torch.eye(
            self.dimension,
            dtype=self.linear.weight.dtype,
            device=self.linear.weight.device,
        )
        if self.initialization == "average":
            active_groups = range(self.geometry.group_size)
            scale = 1.0 / self.geometry.group_size
        elif self.geometry.group_size != 2:
            raise ValueError("keep-first/second initialization is defined only for N=98")
        else:
            active_groups = (0,) if self.initialization == "keep_first" else (1,)
            scale = 1.0
        for group_index in active_groups:
            begin = group_index * self.dimension
            self.linear.weight[:, begin : begin + self.dimension].copy_(identity * scale)
        self.linear.bias.zero_()

    @torch.no_grad()
    def reset_to_average(self) -> None:
        """Backward-compatible explicit reset used by earlier experiments."""

        self.initialization = "average"
        self.reset_projection()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            return tokens
        groups = spatial_groups(tokens, self.geometry)
        flattened = groups.flatten(start_dim=-2)
        return self.linear(flattened)


class LearnedSuperPatchProjection(nn.Module):
    """Project concatenated pre-projection features directly to one token."""

    def __init__(
        self,
        token_count: int,
        merge_axis: str,
        input_dimension: int,
        output_dimension: int,
        source_projection: nn.Linear,
        *,
        initialization: str = "standard",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.geometry = MergeGeometry(token_count, merge_axis)
        if self.geometry.group_size != 2:
            raise ValueError("the first DCT Super-Patch Projection supports N=98 only")
        self.input_dimension = int(input_dimension)
        self.output_dimension = int(output_dimension)
        self.initialization = str(initialization)
        if self.initialization not in VALID_INITIALIZATIONS:
            raise ValueError(f"initialization must be one of {VALID_INITIALIZATIONS}")
        if tuple(source_projection.weight.shape) != (self.output_dimension, self.input_dimension):
            raise ValueError("source patch projection has an unexpected shape")
        self.linear = nn.Linear(
            self.geometry.group_size * self.input_dimension,
            self.output_dimension,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.reset_projection(source_projection)

    @torch.no_grad()
    def reset_projection(self, source_projection: nn.Linear) -> None:
        if self.initialization == "standard":
            self.linear.reset_parameters()
            return
        self.linear.weight.zero_()
        if self.initialization == "average":
            active_groups = range(self.geometry.group_size)
            scale = 1.0 / self.geometry.group_size
        else:
            active_groups = (0,) if self.initialization == "keep_first" else (1,)
            scale = 1.0
        for group_index in active_groups:
            begin = group_index * self.input_dimension
            self.linear.weight[
                :, begin : begin + self.input_dimension
            ].copy_(source_projection.weight * scale)
        if source_projection.bias is None:
            self.linear.bias.zero_()
        else:
            self.linear.bias.copy_(source_projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        groups = spatial_groups(features, self.geometry)
        return self.linear(groups.flatten(start_dim=-2))


def merge_positions(tokens: torch.Tensor, geometry: MergeGeometry) -> torch.Tensor:
    """Positions always use fixed spatial averaging, never the learned Linear."""

    return fixed_average_merge(tokens, geometry)
