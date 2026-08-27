#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from galp.experiments.dct_retokenization.token_merge import (
    FixedSpatialProjection,
    FixedSpatialMerge,
    LearnedSpatialMerge,
    LearnedSuperPatchProjection,
    MergeGeometry,
    spatial_groups,
)


class TokenMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.tokens = torch.randn(3, 14, 14, 192)

    def test_output_shapes(self) -> None:
        expected = {
            (196, "height"): (3, 14, 14, 192),
            (98, "height"): (3, 7, 14, 192),
            (98, "width"): (3, 14, 7, 192),
            (49, "height"): (3, 7, 7, 192),
        }
        for (count, axis), shape in expected.items():
            with self.subTest(count=count, axis=axis):
                result = FixedSpatialMerge(count, axis)(self.tokens)
                self.assertEqual(tuple(result.shape), shape)

    def test_height_pair_order(self) -> None:
        values = torch.arange(14 * 14).reshape(1, 14, 14, 1).float()
        groups = spatial_groups(values, MergeGeometry(98, "height"))
        self.assertEqual(groups[0, 0, 5, :, 0].tolist(), [5.0, 19.0])

    def test_width_pair_order(self) -> None:
        values = torch.arange(14 * 14).reshape(1, 14, 14, 1).float()
        groups = spatial_groups(values, MergeGeometry(98, "width"))
        self.assertEqual(groups[0, 5, 0, :, 0].tolist(), [70.0, 71.0])

    def test_2x2_order(self) -> None:
        values = torch.arange(14 * 14).reshape(1, 14, 14, 1).float()
        groups = spatial_groups(values, MergeGeometry(49, "height"))
        self.assertEqual(groups[0, 0, 0, :, 0].tolist(), [0.0, 1.0, 14.0, 15.0])

    def test_learned_pair_initialization_equals_average(self) -> None:
        for axis in ("height", "width"):
            with self.subTest(axis=axis):
                fixed = FixedSpatialMerge(98, axis)(self.tokens)
                learned = LearnedSpatialMerge(98, axis)(self.tokens)
                torch.testing.assert_close(learned, fixed, rtol=1e-6, atol=1e-6)

    def test_learned_quad_initialization_equals_average(self) -> None:
        fixed = FixedSpatialMerge(49, "height")(self.tokens)
        learned = LearnedSpatialMerge(49, "height")(self.tokens)
        torch.testing.assert_close(learned, fixed, rtol=1e-6, atol=1e-6)

    def test_keep_controls_and_learned_initializations(self) -> None:
        groups = spatial_groups(self.tokens, MergeGeometry(98, "width"))
        for initialization, group_index in (("keep_first", 0), ("keep_second", 1)):
            with self.subTest(initialization=initialization):
                fixed = FixedSpatialProjection(98, "width", initialization)(self.tokens)
                learned = LearnedSpatialMerge(
                    98, "width", initialization=initialization
                )(self.tokens)
                torch.testing.assert_close(fixed, groups[..., group_index, :])
                torch.testing.assert_close(learned, fixed, rtol=1e-6, atol=1e-6)

    def test_folded_superpatch_matches_post_projection_controls(self) -> None:
        features = torch.randn(2, 14, 14, 384)
        source = torch.nn.Linear(384, 192)
        projected = source(features)
        for initialization in ("average", "keep_first", "keep_second"):
            with self.subTest(initialization=initialization):
                expected = FixedSpatialProjection(
                    98, "width", initialization
                )(projected)
                observed = LearnedSuperPatchProjection(
                    98,
                    "width",
                    384,
                    192,
                    source,
                    initialization=initialization,
                )(features)
                torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)

    def test_position_linearity_identity(self) -> None:
        content = torch.randn_like(self.tokens)
        position = torch.randn_like(self.tokens)
        for count, axis in ((98, "height"), (98, "width"), (49, "height")):
            merge = FixedSpatialMerge(count, axis)
            torch.testing.assert_close(
                merge(content + position),
                merge(content) + merge(position),
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
