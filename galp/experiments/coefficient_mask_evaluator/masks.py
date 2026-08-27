#!/usr/bin/env python3
"""Frequency-mask definitions for the coefficient-mask evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# This is intentionally identical to
# galp::jpeg::detail::kZigzagColumnToNaturalIndex.  Values are row-major
# natural indices into the final two dimensions of an 8x8 coefficient block.
ZIGZAG_NATURAL_INDICES: tuple[int, ...] = (
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
)

CONTROL_BUDGETS: tuple[int, ...] = (4, 8, 16, 32)


@dataclass(frozen=True)
class MaskCondition:
    condition_id: str
    family: str
    k: int
    zigzag_ranks: tuple[int, ...]
    natural_indices: tuple[int, ...]
    random_seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["zigzag_ranks"] = list(self.zigzag_ranks)
        payload["natural_indices"] = list(self.natural_indices)
        return payload


def _condition(
    condition_id: str,
    family: str,
    ranks: tuple[int, ...],
    *,
    random_seed: int | None = None,
) -> MaskCondition:
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate zigzag ranks in {condition_id}")
    if any(rank < 0 or rank >= 64 for rank in ranks):
        raise ValueError(f"zigzag rank outside [0, 63] in {condition_id}")
    natural = tuple(ZIGZAG_NATURAL_INDICES[rank] for rank in ranks)
    return MaskCondition(
        condition_id=condition_id,
        family=family,
        k=len(ranks),
        zigzag_ranks=ranks,
        natural_indices=natural,
        random_seed=random_seed,
    )


def build_conditions(random_seed: int = 20260816) -> list[MaskCondition]:
    """Return 64 prefixes and 12 deterministic matched-budget controls."""

    conditions = [
        _condition(
            condition_id=f"prefix_k{k:02d}",
            family="zigzag_prefix",
            ranks=tuple(range(k)),
        )
        for k in range(1, 65)
    ]

    for family in ("high", "mid", "random"):
        for k in CONTROL_BUDGETS:
            condition_id = f"{family}_k{k:02d}"
            if family == "high":
                ranks = tuple(range(64 - k, 64))
                seed = None
            elif family == "mid":
                start = (64 - k) // 2
                ranks = tuple(range(start, start + k))
                seed = None
            else:
                # Give each budget an independent, reproducible stream.  Sorting
                # does not alter the selected set and makes metadata easier to read.
                seed_sequence = np.random.SeedSequence((int(random_seed), int(k)))
                rng = np.random.default_rng(seed_sequence)
                ranks = tuple(sorted(int(rank) for rank in rng.choice(64, size=k, replace=False)))
                seed = int(random_seed)
            conditions.append(
                _condition(
                    condition_id=condition_id,
                    family=family,
                    ranks=ranks,
                    random_seed=seed,
                )
            )

    if len(conditions) != 76:
        raise AssertionError(f"expected 76 conditions, got {len(conditions)}")
    return conditions


def select_conditions(
    conditions: list[MaskCondition], selection: str | None
) -> list[MaskCondition]:
    """Select a comma-separated set of IDs, preserving canonical order."""

    if selection is None or selection.strip().lower() == "all":
        return conditions
    requested = {item.strip() for item in selection.split(",") if item.strip()}
    known = {condition.condition_id for condition in conditions}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown condition IDs: {sorted(unknown)}")
    selected = [condition for condition in conditions if condition.condition_id in requested]
    if not selected:
        raise ValueError("condition selection is empty")
    return selected


def masks_numpy(conditions: list[MaskCondition]) -> np.ndarray:
    masks = np.zeros((len(conditions), 8, 8), dtype=np.bool_)
    for row, condition in enumerate(conditions):
        flat = masks[row].reshape(-1)
        flat[np.asarray(condition.natural_indices, dtype=np.int64)] = True
    return masks
