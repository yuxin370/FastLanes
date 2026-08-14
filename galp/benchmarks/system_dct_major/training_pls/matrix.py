#!/usr/bin/env python3
"""Frozen 2x2 PLS crop/shuffle model-effect matrix."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MATRIX_SCHEMA = "galp-pls-core-matrix-v2"
CORE_CONDITION_IDS = ("A0", "A1", "B2", "B6")
PAIRED_SEEDS = (11997733, 11997734, 11997735, 11997736)
BALANCED_EXECUTION_ORDER = {
    11997733: ("A0", "A1", "B2", "B6"),
    11997734: ("A1", "B2", "B6", "A0"),
    11997735: ("B2", "B6", "A0", "A1"),
    11997736: ("B6", "A0", "A1", "B2"),
}


def _condition(
    condition_id: str,
    *,
    crop_policy: str,
    crop_key_scope: str,
    order_policy: str,
    question: str,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "crop_policy": crop_policy,
        "crop_key_scope": crop_key_scope,
        "order_policy": order_policy,
        "organization": "frozen-physical-layout-plan",
        "segment_images": 1024,
        "segments_per_pool": 4 if order_policy == "closed-pool" else None,
        "execution_mode": "semantic_emulation",
        "question": question,
        "selection_role": "pre-registered scientific contrast; never performance-selected",
    }


def core_matrix() -> dict[str, Any]:
    conditions = [
        _condition(
            "A0",
            crop_policy="per-sample",
            crop_key_scope="logical_sample_id",
            order_policy="global",
            question="standard per-sample-crop/global-shuffle control",
        ),
        _condition(
            "A1",
            crop_policy="per-pls",
            crop_key_scope="virtual_pls_id",
            order_policy="global",
            question="crop-sharing effect under exactly the A0 sample order",
        ),
        _condition(
            "B2",
            crop_policy="per-sample",
            crop_key_scope="logical_sample_id",
            order_policy="closed-pool",
            question="closed-pool shuffle effect under per-sample crop",
        ),
        _condition(
            "B6",
            crop_policy="per-pls",
            crop_key_scope="virtual_pls_id",
            order_policy="closed-pool",
            question="combined shared-crop and closed-pool target strategy",
        ),
    ]
    return {
        "schema_version": MATRIX_SCHEMA,
        "design": "2x2 paired-seed factorial",
        "fixed": {
            "segment_images": 1024,
            "segments_per_closed_pool": 4,
            "organization_seed": 20260810,
            "physical_microbatch": 64,
            "gradient_accumulation": 16,
            "effective_update_batch": 1024,
            "epochs": 300,
            "execution_mode": "semantic_emulation",
        },
        "conditions": conditions,
        "paired_seeds": list(PAIRED_SEEDS),
        "balanced_execution_order": {
            str(seed): list(order) for seed, order in BALANCED_EXECUTION_ORDER.items()
        },
        "estimands": [
            {
                "effect_id": "crop_global",
                "formula": "A1 - A0",
                "kind": "simple",
            },
            {
                "effect_id": "crop_closed_pool",
                "formula": "B6 - B2",
                "kind": "simple",
            },
            {
                "effect_id": "shuffle_per_sample_crop",
                "formula": "B2 - A0",
                "kind": "simple",
            },
            {
                "effect_id": "shuffle_per_pls_crop",
                "formula": "B6 - A1",
                "kind": "simple",
            },
            {
                "effect_id": "crop_main",
                "formula": "0.5 * [(A1 - A0) + (B6 - B2)]",
                "kind": "factorial",
            },
            {
                "effect_id": "shuffle_main",
                "formula": "0.5 * [(B2 - A0) + (B6 - A1)]",
                "kind": "factorial",
            },
            {
                "effect_id": "crop_x_shuffle",
                "formula": "B6 - B2 - A1 + A0",
                "kind": "factorial",
            },
            {
                "effect_id": "target_vs_standard",
                "formula": "B6 - A0",
                "kind": "factorial",
            },
        ],
        "interpretation_policy": {
            "winner_ranking": False,
            "strategy_gates": False,
            "confidence_interval_including_zero_means_no_effect": False,
            "practical_top1_margin_percentage_points": 0.3,
            "system_metrics": "explanatory only",
        },
    }


def resolve_condition(condition_id: str) -> dict[str, Any]:
    condition_id = str(condition_id).upper()
    matches = [
        condition
        for condition in core_matrix()["conditions"]
        if condition["condition_id"] == condition_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"unknown core PLS condition {condition_id!r}; expected {CORE_CONDITION_IDS}"
        )
    return deepcopy(matches[0])


def execution_order(seeds: list[int] | tuple[int, ...]) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    for seed in seeds:
        if seed not in BALANCED_EXECUTION_ORDER:
            raise ValueError(
                f"seed {seed} has no pre-registered balanced order; expected {PAIRED_SEEDS}"
            )
        for position, condition_id in enumerate(BALANCED_EXECUTION_ORDER[seed], start=1):
            result.append((seed, position, condition_id))
    return result
