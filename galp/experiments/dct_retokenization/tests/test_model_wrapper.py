#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch
from torch import nn

from galp.experiments.dct_retokenization.model_wrapper import DctRetokenizationWrapper


class _SinCosStub(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _CountingLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(384, 192)
        self.forward_calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(value)


class _PatchStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(_CountingLinear(), _SinCosStub())


class _BaseStub(nn.Module):
    pixel_space = "DCT"

    def __init__(self) -> None:
        super().__init__()
        self.patchembed = _PatchStub()


class ModelWrapperTest(unittest.TestCase):
    def test_superpatch_center_position_shape_and_coordinates(self) -> None:
        wrapper = DctRetokenizationWrapper(
            _BaseStub(),
            token_count=98,
            merge_axis="width",
            merge_type="learned",
            architecture="superpatch",
            initialization="standard",
            position_mode="superpatch_center",
        )
        content = torch.zeros(2, 14, 14, 192)
        position = wrapper.superpatch_position_grid(content)
        self.assertEqual(tuple(position.shape), (1, 14, 7, 192))
        # First width phase is evaluated at the original-coordinate center 0.5.
        self.assertAlmostEqual(float(position[0, 0, 0, 0]), float(torch.sin(torch.tensor(0.5))), places=6)
        # First height phase is row 0, so its sine is zero.
        self.assertEqual(float(position[0, 0, 0, 96]), 0.0)

    def test_superpatch_path_does_not_materialize_old_content_tokens(self) -> None:
        class _FeatureStubWrapper(DctRetokenizationWrapper):
            def preprojection_features(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
                return torch.zeros(y.shape[0], 14, 14, 384)

            def original_position_grid(self, content: torch.Tensor) -> torch.Tensor:
                return torch.zeros(1, 14, 14, 192)

        base = _BaseStub()
        wrapper = _FeatureStubWrapper(
            base,
            token_count=98,
            merge_axis="width",
            merge_type="learned",
            architecture="superpatch",
            initialization="average",
        )
        observed = wrapper.forward_tokens(torch.zeros(2, 1), torch.zeros(2, 1))
        self.assertEqual(tuple(observed.shape), (2, 98, 192))
        self.assertEqual(base.patchembed.projection[0].forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
