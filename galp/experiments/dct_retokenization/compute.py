#!/usr/bin/env python3
"""Architecture-level MAC accounting for the RGB-no-more ViT-Ti prototype."""

from __future__ import annotations


EMBEDDING_DIMENSION = 192
LAYERS = 12
PATCH_TOKENS = 196
PATCH_INPUT_DIMENSION = 384
CLASSES = 1000


def transformer_macs(tokens: int, dimension: int = EMBEDDING_DIMENSION, layers: int = LAYERS) -> int:
    # Per layer: QKV + output projections + 2 attention matmuls + two-layer 4D FFN.
    per_layer = 12 * tokens * dimension * dimension + 2 * tokens * tokens * dimension
    return layers * per_layer


def patch_projection_macs() -> int:
    return PATCH_TOKENS * PATCH_INPUT_DIMENSION * EMBEDDING_DIMENSION


def retokenizer_macs(tokens: int, dimension: int = EMBEDDING_DIMENSION) -> int:
    if tokens == 196:
        return 0
    group_size = 2 if tokens == 98 else 4 if tokens == 49 else None
    if group_size is None:
        raise ValueError("tokens must be 196, 98, or 49")
    return tokens * group_size * dimension * dimension


def classifier_macs(dimension: int = EMBEDDING_DIMENSION, classes: int = CLASSES) -> int:
    return dimension * dimension + dimension * classes


def prototype_macs(tokens: int) -> int:
    return (
        patch_projection_macs()
        + retokenizer_macs(tokens)
        + transformer_macs(tokens)
        + classifier_macs()
    )


def mac_table() -> list[dict[str, float | int]]:
    baseline = prototype_macs(196)
    rows = []
    for tokens in (196, 98, 49):
        macs = prototype_macs(tokens)
        rows.append(
            {
                "tokens": tokens,
                "patch_projection_macs": patch_projection_macs(),
                "retokenizer_macs": retokenizer_macs(tokens),
                "transformer_macs": transformer_macs(tokens),
                "classifier_macs": classifier_macs(),
                "prototype_total_macs": macs,
                "prototype_total_gmac": macs / 1e9,
                "relative_to_n196": macs / baseline,
                "theoretical_reduction_x": baseline / macs,
                "flops_convention_1mac_equals_2flops": 2 * macs,
            }
        )
    return rows
