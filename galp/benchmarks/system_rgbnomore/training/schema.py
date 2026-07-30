#!/usr/bin/env python3
"""Schema constants and status helpers for the training benchmark.

The training schema is deliberately independent of the inference benchmark
schema in :mod:`common`.  Keeping the names separate prevents inference
consumers from accidentally interpreting optimizer-step results as inference
results.
"""

from __future__ import annotations

from typing import Any, Iterable


TRAINING_CONTRACT_SCHEMA = "galp-rgbnomore-training-contract-v1"
TRAINING_RESULT_SCHEMA = "galp-rgbnomore-training-results-v1"
TRAINING_PIPELINE_SCHEMA = "galp-rgbnomore-training-pipeline-v1"
TRAINING_VALIDATION_SCHEMA = "galp-rgbnomore-training-validation-v1"
TRAINING_SAMPLE_ORDER_SCHEMA = "galp-rgbnomore-training-sample-order-v1"
TRAINING_AUGMENTATION_SCHEMA = "galp-rgbnomore-training-augmentation-v1"

PIPELINES = ("galp", "rgbnomore", "dali", "pytorch")
DOMAINS = {"galp": "dct", "rgbnomore": "dct", "dali": "rgb", "pytorch": "rgb"}
COMPARISON_GROUPS = {
    "dct": ("galp", "rgbnomore"),
    "rgb": ("dali", "pytorch"),
}
STATUS_VALUES = ("passed", "failed", "warning", "not_run", "not_applicable")
STATUS_DIMENSIONS = (
    "correctness",
    "semantic",
    "performance",
    "convergence",
    "artifact",
    "overall",
)


def empty_status(value: str = "not_applicable") -> dict[str, str]:
    if value not in STATUS_VALUES:
        raise ValueError(f"invalid training status: {value}")
    return {dimension: value for dimension in STATUS_DIMENSIONS}


def validate_pipeline_names(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        name = str(value).strip().lower()
        if name not in PIPELINES:
            raise ValueError(f"unknown pipeline {value!r}; expected one of {PIPELINES}")
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("enabled_pipelines must not be empty")
    return result


def validate_comparison_groups(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        name = str(value).strip().lower()
        if name not in COMPARISON_GROUPS:
            raise ValueError(
                f"unknown comparison group {value!r}; expected one of {tuple(COMPARISON_GROUPS)}"
            )
        if name not in result:
            result.append(name)
    return result


def validate_required_group_coverage(enabled: Iterable[str], required: Iterable[str]) -> None:
    enabled_set = set(enabled)
    for group in required:
        missing = sorted(set(COMPARISON_GROUPS[group]) - enabled_set)
        if missing:
            raise ValueError(f"required comparison group {group!r} is missing pipelines {missing}")


def validate_training_document(payload: dict[str, Any], schema: str) -> list[str]:
    """Return structural schema errors without needing a third-party package."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["document is not a JSON object"]
    if payload.get("schema_version") != schema:
        errors.append(f"schema_version must be {schema!r}")
    if schema == TRAINING_CONTRACT_SCHEMA:
        for field in (
            "enabled_pipelines",
            "required_comparison_groups",
            "model",
            "optimizer",
            "scheduler",
            "augmentation",
            "sample_order",
            "datasets",
            "execution",
            "provenance",
        ):
            if field not in payload:
                errors.append(f"contract missing {field}")
    elif schema == TRAINING_RESULT_SCHEMA:
        for field in (
            "contract_sha256",
            "pipeline_status",
            "comparison_group_status",
            "pipelines",
        ):
            if field not in payload:
                errors.append(f"results missing {field}")
    elif schema == TRAINING_PIPELINE_SCHEMA:
        for field in (
            "pipeline",
            "domain",
            "phase_results",
            "sample_order",
            "status",
        ):
            if field not in payload:
                errors.append(f"pipeline artifact missing {field}")
    elif schema == TRAINING_VALIDATION_SCHEMA:
        for field in ("ok", "failures", "warnings", "checks", "pipeline_status", "comparison_group_status"):
            if field not in payload:
                errors.append(f"validation missing {field}")
    return errors
