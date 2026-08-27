#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from galp.experiments.coefficient_mask_evaluator.masks import (
    CONTROL_BUDGETS,
    ZIGZAG_NATURAL_INDICES,
    build_conditions,
    masks_numpy,
)


class MaskDefinitionTest(unittest.TestCase):
    def test_zigzag_matches_repository_contract(self) -> None:
        self.assertEqual(ZIGZAG_NATURAL_INDICES[:8], (0, 1, 8, 16, 9, 2, 3, 10))
        self.assertEqual(ZIGZAG_NATURAL_INDICES[-1], 63)
        self.assertEqual(len(ZIGZAG_NATURAL_INDICES), 64)
        self.assertEqual(set(ZIGZAG_NATURAL_INDICES), set(range(64)))

    def test_condition_count_and_budgets(self) -> None:
        conditions = build_conditions()
        masks = masks_numpy(conditions)
        self.assertEqual(len(conditions), 76)
        self.assertEqual(masks.shape, (76, 8, 8))
        for condition, mask in zip(conditions, masks, strict=True):
            self.assertEqual(int(mask.sum()), condition.k)

    def test_prefixes_are_nested_and_k64_is_identity(self) -> None:
        conditions = build_conditions()
        prefix_masks = masks_numpy(conditions[:64]).reshape(64, 64)
        for k in range(1, 64):
            self.assertTrue(np.all(prefix_masks[k - 1] <= prefix_masks[k]))
        self.assertTrue(np.all(prefix_masks[-1]))

    def test_control_definitions(self) -> None:
        by_id = {condition.condition_id: condition for condition in build_conditions()}
        for k in CONTROL_BUDGETS:
            self.assertEqual(by_id[f"high_k{k:02d}"].zigzag_ranks, tuple(range(64 - k, 64)))
            start = (64 - k) // 2
            self.assertEqual(by_id[f"mid_k{k:02d}"].zigzag_ranks, tuple(range(start, start + k)))
            self.assertEqual(len(by_id[f"random_k{k:02d}"].zigzag_ranks), k)

    def test_random_controls_are_reproducible_and_seeded(self) -> None:
        first = [condition for condition in build_conditions(1234) if condition.family == "random"]
        second = [condition for condition in build_conditions(1234) if condition.family == "random"]
        third = [condition for condition in build_conditions(5678) if condition.family == "random"]
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
