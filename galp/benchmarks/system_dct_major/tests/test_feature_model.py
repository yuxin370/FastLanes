from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from feature_model import FeatureExtractor, expected_output_width  # noqa: E402


class MeanTokens(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=1)


class DctPatch(torch.nn.Module):
    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        return y + cbcr


class FakeModel(torch.nn.Module):
    def __init__(self, dct: bool = False) -> None:
        super().__init__()
        self.patchembed = DctPatch() if dct else torch.nn.Identity()
        self.encoder = torch.nn.Identity()
        self.classhead = torch.nn.Sequential(
            OrderedDict(
                [
                    ("ch_lrnorm", torch.nn.Identity()),
                    ("ch_gap", MeanTokens()),
                    ("ch_linear1", torch.nn.Identity()),
                    ("ch_tanh", torch.nn.Tanh()),
                    ("ch_linear2", torch.nn.Linear(192, 1000)),
                ]
            )
        )


class FeatureModelTest(unittest.TestCase):
    def test_rgb_penultimate_tap_matches_head_prefix(self) -> None:
        model = FakeModel()
        inputs = torch.randn(3, 5, 192)
        expected = torch.tanh(inputs.mean(dim=1))
        actual = FeatureExtractor(model, domain="rgb", stage="penultimate")(inputs)
        torch.testing.assert_close(actual, expected)

    def test_dct_pooled_tap_accepts_two_inputs(self) -> None:
        model = FakeModel(dct=True)
        y = torch.randn(2, 4, 192)
        cbcr = torch.randn(2, 4, 192)
        actual = FeatureExtractor(model, domain="dct", stage="pooled")(y, cbcr)
        torch.testing.assert_close(actual, (y + cbcr).mean(dim=1))

    def test_output_width_contract(self) -> None:
        self.assertEqual(expected_output_width("feature-extraction"), 192)
        self.assertEqual(expected_output_width("evaluation"), 1000)


if __name__ == "__main__":
    unittest.main()
