#!/usr/bin/env python3
"""Frozen 2x2 PLS matrix plus explicitly supplemental controls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MATRIX_SCHEMA = "galp-pls-core-matrix-v2"
SUPPLEMENTAL_MATRIX_SCHEMA = "galp-pls-matrix-with-supplemental-controls-v1"
CORE_CONDITION_IDS = ("A0", "A1", "B2", "B6")
SUPPLEMENTAL_CONDITION_IDS = ("N6", "N2")
REGISTERED_CONDITION_IDS = CORE_CONDITION_IDS + SUPPLEMENTAL_CONDITION_IDS
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
        "segments_per_pool": (
            4 if order_policy in {"closed-pool", "physical-order"} else None
        ),
        "execution_mode": "semantic_emulation",
        "question": question,
        "selection_role": "pre-registered scientific contrast; never performance-selected",
    }


def supplemental_conditions() -> list[dict[str, Any]]:
    """Return controls that must not alter the registered 2x2 estimands."""

    return [
        _condition(
            "N2",
            crop_policy="per-sample",
            crop_key_scope="logical_sample_id",
            order_policy="physical-order",
            question=(
                "per-sample-crop no-epoch-shuffle control: consume the frozen "
                "premixed physical order in consecutive M=4 pools"
            ),
        ),
        _condition(
            "N6",
            crop_policy="per-pls",
            crop_key_scope="virtual_pls_id",
            order_policy="physical-order",
            question=(
                "no-epoch-shuffle control: consume the frozen premixed physical "
                "order in consecutive M=4 pools"
            ),
        )
    ]


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


def experiment_matrix(condition_ids: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Describe a plan without rewriting the original four-condition design."""

    selected = {str(value).upper() for value in condition_ids}
    supplemental = [
        condition
        for condition in supplemental_conditions()
        if condition["condition_id"] in selected
    ]
    matrix = core_matrix()
    if not supplemental:
        return matrix
    matrix["schema_version"] = SUPPLEMENTAL_MATRIX_SCHEMA
    matrix["design"] = "registered 2x2 paired-seed factorial plus supplemental controls"
    matrix["conditions"].extend(supplemental)
    matrix["supplemental_estimands"] = [
        {
            "effect_id": "closed_pool_shuffle_vs_none_per_pls_crop",
            "formula": "B6 - N6",
            "kind": "supplemental paired simple effect",
        },
        {
            "effect_id": "global_shuffle_vs_none_per_pls_crop",
            "formula": "A1 - N6",
            "kind": "supplemental paired simple effect",
        },
        {
            "effect_id": "closed_pool_shuffle_vs_none_per_sample_crop",
            "formula": "B2 - N2",
            "kind": "supplemental paired simple effect",
        },
        {
            "effect_id": "global_shuffle_vs_none_per_sample_crop",
            "formula": "A0 - N2",
            "kind": "supplemental paired simple effect",
        },
        {
            "effect_id": "crop_effect_without_epoch_shuffle",
            "formula": "N6 - N2",
            "kind": "supplemental paired simple effect",
        },
        {
            "effect_id": "crop_x_closed_shuffle_vs_none",
            "formula": "B6 - B2 - N6 + N2",
            "kind": "supplemental paired interaction",
        },
    ]
    matrix["supplemental_interpretation"] = (
        "N2 and N6 are supplemental no-shuffle controls and are never folded "
        "into the original registered 2x2 main-effect or interaction formulas."
    )
    return matrix


def resolve_condition(condition_id: str) -> dict[str, Any]:
    condition_id = str(condition_id).upper()
    matches = [
        condition
        for condition in core_matrix()["conditions"] + supplemental_conditions()
        if condition["condition_id"] == condition_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"unknown PLS condition {condition_id!r}; expected {REGISTERED_CONDITION_IDS}"
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
        for offset, condition_id in enumerate(SUPPLEMENTAL_CONDITION_IDS, start=1):
            result.append((seed, len(CORE_CONDITION_IDS) + offset, condition_id))
    return result
