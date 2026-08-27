#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from galp.experiments.dct_retokenization.delayed_wrapper import DelayedRetokenizationWrapper
from galp.experiments.dct_retokenization.train_superpatch import configure_stage, load_config


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class _PatchStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(384, 192))

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        del cbcr
        values = torch.arange(196 * 192, dtype=y.dtype, device=y.device)
        return values.reshape(1, 196, 192).expand(y.shape[0], -1, -1)


class _RecordingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.sequence_lengths: list[int] = []

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.sequence_lengths.append(int(tokens.shape[1]))
        return tokens + self.scale


class _HeadStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(192, 5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens.mean(dim=1))


class _BaseStub(nn.Module):
    pixel_space = "DCT"

    def __init__(self) -> None:
        super().__init__()
        self.patchembed = _PatchStub()
        self.encoder = nn.Sequential(*[_RecordingBlock() for _ in range(12)])
        self.classhead = _HeadStub()


class DelayedWrapperTest(unittest.TestCase):
    def test_merge_occurs_after_four_blocks(self) -> None:
        base = _BaseStub()
        wrapper = DelayedRetokenizationWrapper(
            base,
            merge_after_block=4,
            merge_axis="width",
            initialization="keep_first",
        )
        logits = wrapper(torch.zeros(2, 1), torch.zeros(2, 1))
        self.assertEqual(tuple(logits.shape), (2, 5))
        lengths = [block.sequence_lengths for block in base.encoder]
        self.assertEqual(lengths[:4], [[196]] * 4)
        self.assertEqual(lengths[4:], [[98]] * 8)

    def test_keep_first_initialization_uses_even_width_latents(self) -> None:
        wrapper = DelayedRetokenizationWrapper(
            _BaseStub(),
            merge_after_block=2,
            merge_axis="width",
            initialization="keep_first",
        )
        tokens = torch.arange(196 * 192, dtype=torch.float32).reshape(1, 196, 192)
        merged = wrapper.merge_latents(tokens).reshape(1, 14, 7, 192)
        source = tokens.reshape(1, 14, 14, 192)[:, :, 0::2]
        torch.testing.assert_close(merged, source, rtol=0, atol=0)

    def test_stage_a_and_b_freeze_boundaries(self) -> None:
        config_a = load_config(EXPERIMENT_ROOT / "configs" / "delayed_stage_a_block4.json")
        wrapper_a = DelayedRetokenizationWrapper(
            _BaseStub(), merge_after_block=4, initialization=config_a["initialization"]
        )
        groups_a, _ = configure_stage(wrapper_a, "delayed_stage_a", config_a)
        self.assertEqual([group["group_name"] for group in groups_a], [
            "delayed_retokenizer",
            "head",
        ])
        self.assertFalse(any(parameter.requires_grad for parameter in wrapper_a.base_model.encoder.parameters()))

        config_b = load_config(EXPERIMENT_ROOT / "configs" / "delayed_stage_b_block4.json")
        wrapper_b = DelayedRetokenizationWrapper(
            _BaseStub(), merge_after_block=4, initialization=config_b["initialization"]
        )
        groups_b, _ = configure_stage(wrapper_b, "delayed_stage_b", config_b)
        self.assertEqual(groups_b[-1]["group_name"], "post_merge_blocks")
        blocks = wrapper_b.encoder_blocks()
        self.assertFalse(any(parameter.requires_grad for block in blocks[:4] for parameter in block.parameters()))
        self.assertTrue(all(parameter.requires_grad for block in blocks[4:] for parameter in block.parameters()))

    def test_rejects_unsupported_merge_location(self) -> None:
        with self.assertRaises(ValueError):
            DelayedRetokenizationWrapper(_BaseStub(), merge_after_block=3)


if __name__ == "__main__":
    unittest.main()
