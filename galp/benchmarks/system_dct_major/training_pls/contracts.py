#!/usr/bin/env python3
"""Condition whitelist and auditable contracts for the core matrix."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core_schedule import policy_digest
from .layout import sha256_file
from .matrix import resolve_condition
from .recipe import sha256_json


CONTRACT_SCHEMA = "galp-pls-condition-contract-v2"
CONTRACT_DIFF_SCHEMA = "galp-pls-condition-contract-diff-v2"
RUN_MANIFEST_SCHEMA = "galp-pls-run-manifest-v1"
TRAINING_SOURCE_SCOPE = "galp-pls-training-runtime-v1"

_TRAINING_PLS_RUNTIME_MODULES = frozenset(
    {
        "__init__.py",
        "artifacts.py",
        "contracts.py",
        "core_schedule.py",
        "layout.py",
        "matrix.py",
        "parquet_helper.py",
        "published_augmentation.py",
        "published_optimizer.py",
        "recipe.py",
        "run_matrix.py",
        "schedule.py",
        "train.py",
    }
)

_TRAINING_RUNTIME_EXTERNAL_PATHS = frozenset(
    {
        "galp/benchmarks/system_rgbnomore/training/artifacts.py",
        "galp/benchmarks/system_rgbnomore/training/augmentation.py",
        "galp/benchmarks/system_rgbnomore/training/direct_dct_reader.py",
        "galp/benchmarks/system_rgbnomore/training/model_factory.py",
        "galp/benchmarks/system_rgbnomore/training/pipeline.py",
        "galp/benchmarks/system_rgbnomore/training/pls_experiment.py",
        "galp/benchmarks/system_rgbnomore/training/sample_order.py",
        "galp/include/galp/direct_dct_pls.hpp",
        "galp/include/galp/profiles/rgbnomore.hpp",
        "galp/src/api/direct_dct_pls.cpp",
        "galp/src/api/direct_dct_pls_postprocess.cu",
        "galp/src/api/direct_dct_pls_postprocess.hpp",
        "galp/torch/__init__.py",
        "galp/torch/direct_dct.py",
        "galp/torch/direct_dct_pls_torch.cpp",
        "galp/torch/direct_dct_pls_torch.hpp",
    }
)

ALLOWED_CONDITION_DIFFERENCES = frozenset(
    {
        "condition_id",
        "crop_policy",
        "crop_key_scope",
        "order_policy",
        "segments_per_pool",
        "pool_membership_digest",
        "sample_order_digest",
        "crop_key_digest",
    }
)


@lru_cache(maxsize=8)
def _cached_file_hash(path: str) -> str:
    return sha256_file(Path(path))


def code_version(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short")
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    core_paths = sorted(
        path
        for path in (
            repo_root / "galp/benchmarks/system_dct_major/training_pls"
        ).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".md"}
    )
    for extra in (
        repo_root / "galp/benchmarks/system_dct_major/tests/test_training_pls.py",
        *(
            repo_root / relative
            for relative in sorted(_TRAINING_RUNTIME_EXTERNAL_PATHS)
        ),
        repo_root / "docs/PHYSICAL_PLS_TRAINING_PIPELINE.md",
    ):
        if extra.is_file():
            core_paths.append(extra)
    source_files = [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
        }
        for path in sorted(set(core_paths))
    ]
    runtime_source_files = [
        row
        for row in source_files
        if _is_training_runtime_source(str(row["path"]))
    ]
    return {
        "commit": commit,
        "working_tree_clean": not bool(status),
        "working_tree_status_sha256": sha256_json(status.splitlines()),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "experiment_source_files": source_files,
        "experiment_source_tree_sha256": sha256_json(source_files),
        "training_source_scope": TRAINING_SOURCE_SCOPE,
        "training_runtime_source_files": runtime_source_files,
        "training_runtime_source_tree_sha256": sha256_json(runtime_source_files),
    }


def _is_training_runtime_source(relative_path: str) -> bool:
    prefix = "galp/benchmarks/system_dct_major/training_pls/"
    if relative_path.startswith(prefix):
        return Path(relative_path).name in _TRAINING_PLS_RUNTIME_MODULES
    return relative_path in _TRAINING_RUNTIME_EXTERNAL_PATHS


def blocking_code_identity(version: Mapping[str, Any]) -> dict[str, str]:
    """Return the code identity that is allowed to block execution/resume.

    Whole-repository Git status and diff hashes remain useful provenance, but
    unrelated edits must not invalidate a prepared training run.  Only the
    explicitly scoped runtime source tree participates in the blocking identity.
    """

    scope = version.get("training_source_scope")
    source_hash = version.get("training_runtime_source_tree_sha256")
    if scope != TRAINING_SOURCE_SCOPE or not isinstance(source_hash, str):
        raise ValueError(
            "run manifest lacks the scoped training runtime source identity; "
            "regenerate it with training_pls.run_matrix"
        )
    return {
        "training_source_scope": str(scope),
        "training_runtime_source_tree_sha256": source_hash,
    }


def code_provenance_differences(
    planned: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Report non-blocking repository provenance changes for audit output."""

    fields = (
        "commit",
        "working_tree_clean",
        "working_tree_status_sha256",
        "tracked_diff_sha256",
        "experiment_source_tree_sha256",
    )
    return {
        field: {"planned": planned.get(field), "current": current.get(field)}
        for field in fields
        if planned.get(field) != current.get(field)
    }


def condition_identity_hash(contract: Mapping[str, Any]) -> str:
    """Hash only fields that define training and checkpoint compatibility."""

    payload = {
        key: value
        for key, value in contract.items()
        if key not in {"condition_hash", "run_manifest_hash"}
    }
    code = payload.get("code_version")
    if not isinstance(code, Mapping):
        raise ValueError("run manifest lacks code_version")
    payload["code_version"] = blocking_code_identity(code)
    return sha256_json(payload)


def build_condition_contract(
    *,
    condition_id: str,
    seed: int,
    train_manifest: Path,
    val_manifest: Path,
    layout_plan: Mapping[str, Any],
    recipe: Mapping[str, Any],
    total_optimizer_updates: int,
    initial_model_hash: str,
    code: Mapping[str, Any],
    device: str,
    execution_backend: str = "semantic-emulation",
    physical_execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    condition = resolve_condition(condition_id)
    layout_hash = str(layout_plan["layout_hash"])
    if execution_backend not in {"semantic-emulation", "native-physical-pls"}:
        raise ValueError(f"unsupported execution backend: {execution_backend!r}")
    native_physical = execution_backend == "native-physical-pls"
    if native_physical and physical_execution is None:
        raise ValueError("native-physical-pls requires a physical execution contract")
    if not native_physical and physical_execution is not None:
        raise ValueError("semantic-emulation cannot carry a physical execution contract")
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "run_manifest_schema": RUN_MANIFEST_SCHEMA,
        "artifact_role": "resolved-run-manifest",
        "condition_id": condition_id,
        "training_seed": int(seed),
        "crop_policy": condition["crop_policy"],
        "crop_key_scope": condition["crop_key_scope"],
        "order_policy": condition["order_policy"],
        "segments_per_pool": condition["segments_per_pool"],
        "pool_membership_digest": policy_digest(
            layout_hash=layout_hash,
            seed=seed,
            condition_id=condition_id,
            kind="pool_membership",
        ),
        "sample_order_digest": policy_digest(
            layout_hash=layout_hash,
            seed=seed,
            condition_id=condition_id,
            kind="sample_order",
        ),
        "crop_key_digest": policy_digest(
            layout_hash=layout_hash,
            seed=seed,
            condition_id=condition_id,
            kind="crop_key",
        ),
        "physical_layout_plan_hash": layout_hash,
        "train_manifest_hash": _cached_file_hash(str(train_manifest.resolve())),
        "validation_manifest_hash": _cached_file_hash(str(val_manifest.resolve())),
        "model_configuration": recipe["model"],
        "model_execution": recipe["execution"],
        "initial_model_hash": initial_model_hash,
        "optimizer_configuration": recipe["optimizer"],
        "weight_decay_configuration": recipe["optimizer"]["weight_decay"],
        "scheduler": recipe["scheduler"],
        "epochs": recipe["training"]["epochs"],
        "total_optimizer_updates": int(total_optimizer_updates),
        "microbatch_size": recipe["training"]["physical_microbatch"],
        "gradient_accumulation": recipe["training"]["gradient_accumulation"],
        "effective_update_batch": recipe["training"]["effective_update_batch"],
        "mixup": recipe["augmentation"]["mixup"],
        "randaugment": recipe["augmentation"]["randaugment"],
        "flip": recipe["augmentation"]["horizontal_flip"],
        "validation_preprocessing": recipe["validation"],
        "precision": recipe["training"]["precision"],
        "checkpoint_schedule": {
            "rolling": "every completed epoch",
            "permanent": "every formal validation epoch",
            "resume_granularity": "completed epoch",
        },
        "backend_implementation": (
            "galp-native-direct-dct-pls-block-major-v1"
            if native_physical
            else "shared-galp-direct-dct-semantic-backend-v2"
        ),
        "recipe_hash": recipe["recipe_hash"],
        "code_version": dict(code),
        "execution_device": str(device),
        "execution_mode": (
            "native_physical_pls" if native_physical else "semantic_emulation"
        ),
        "semantic_emulation": not native_physical,
        "physical_fls_observed": native_physical,
        "physical_gpu_pool": native_physical,
        "layout_hash": layout_hash,
        "reference_topology": "8gpu_ddp",
        "execution_topology": "single_gpu_accum16",
        "distributed_bitwise_equivalence_claim": False,
    }
    if native_physical:
        contract["physical_execution"] = dict(physical_execution or {})
    contract["condition_hash"] = condition_identity_hash(contract)
    contract["run_manifest_hash"] = sha256_json(contract)
    return contract


def _without_hash(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"condition_hash", "run_manifest_hash"}
    }


def condition_differences(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    left = _without_hash(baseline)
    right = _without_hash(candidate)
    keys = sorted(set(left) | set(right))
    return {
        key: {"baseline": left.get(key), "candidate": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def validate_seed_block_contracts(
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not contracts:
        raise ValueError("seed block contains no condition contracts")
    seeds = {int(contract["training_seed"]) for contract in contracts}
    if len(seeds) != 1:
        raise ValueError(f"condition contract seed block is not paired: {sorted(seeds)}")
    by_condition = {str(contract["condition_id"]): contract for contract in contracts}
    if "A0" not in by_condition:
        raise ValueError("condition contract seed block lacks A0 baseline")
    records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for condition_id in sorted(by_condition):
        differences = condition_differences(by_condition["A0"], by_condition[condition_id])
        unexpected = sorted(set(differences) - ALLOWED_CONDITION_DIFFERENCES)
        record = {
            "condition_id": condition_id,
            "baseline": "A0",
            "different_fields": sorted(differences),
            "differences": differences,
            "unexpected_fields": unexpected,
            "valid": not unexpected,
        }
        records.append(record)
        if unexpected:
            violations.append(record)
    result = {
        "schema_version": CONTRACT_DIFF_SCHEMA,
        "training_seed": next(iter(seeds)),
        "allowed_different_fields": sorted(ALLOWED_CONDITION_DIFFERENCES),
        "comparisons": records,
        "valid": not violations,
        "violations": violations,
    }
    if violations:
        raise ValueError(
            "condition contract whitelist violation: "
            + json.dumps(violations, sort_keys=True)
        )
    return result
