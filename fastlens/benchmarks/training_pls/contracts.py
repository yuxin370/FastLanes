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
from fastlens.benchmarks.training_audit_policy import validate_audit_policy


CONTRACT_SCHEMA = "galp-pls-condition-contract-v3"
CONTRACT_DIFF_SCHEMA = "galp-pls-condition-contract-diff-v3"
RUN_MANIFEST_SCHEMA = "galp-pls-run-manifest-v2"
TRAINING_SOURCE_SCOPE = "galp-training-runtime-v2"
STANDARD_DCT_BACKEND = "standard-rgbnomore-dct"
IC_DCT_BACKEND = "ic-matched-dct"
FS_DCT_BACKEND = "fs-matched-dct"
DCT_REFERENCE_BACKENDS = (STANDARD_DCT_BACKEND, IC_DCT_BACKEND, FS_DCT_BACKEND)
SEMANTIC_BACKEND = "semantic-emulation"
NATIVE_PHYSICAL_BACKEND = "native-physical-pls"
EXECUTION_BACKENDS = (
    SEMANTIC_BACKEND,
    NATIVE_PHYSICAL_BACKEND,
    STANDARD_DCT_BACKEND,
    IC_DCT_BACKEND,
    FS_DCT_BACKEND,
)
BACKEND_IMPLEMENTATIONS = {
    SEMANTIC_BACKEND: "shared-galp-direct-dct-semantic-backend-v2",
    NATIVE_PHYSICAL_BACKEND: "galp-native-direct-dct-pls-block-major-v1",
    STANDARD_DCT_BACKEND: "rgbnomore-jpeg-dct-premixed-reference-v2",
    IC_DCT_BACKEND: "ic-original-jpeg-q-reference-v1",
    FS_DCT_BACKEND: "fs-original-jpeg-q-reference-v1",
}
EXECUTION_MODES = {
    SEMANTIC_BACKEND: "semantic_emulation",
    NATIVE_PHYSICAL_BACKEND: "native_physical_pls",
    STANDARD_DCT_BACKEND: "standard_rgbnomore_dct",
    IC_DCT_BACKEND: "ic_matched_dct",
    FS_DCT_BACKEND: "fs_matched_dct",
}

_TRAINING_PLS_RUNTIME_MODULES = frozenset(
    {
        "__init__.py",
        "artifacts.py",
        "contracts.py",
        "core_schedule.py",
        "layout.py",
        "matrix.py",
        "model_registry.py",
        "parquet_helper.py",
        "published_augmentation.py",
        "published_optimizer.py",
        "recipe.py",
        "report_convergence_reference.py",
        "report_equal_image_performance.py",
        "run_matrix.py",
        "train.py",
    }
)

_TRAINING_RUNTIME_EXTERNAL_PATHS = frozenset(
    {
        "fastlens/benchmarks/baselines/system_rgbnomore/training/artifacts.py",
        "fastlens/benchmarks/baselines/system_rgbnomore/training/augmentation.py",
        "fastlens/benchmarks/baselines/system_rgbnomore/training/direct_dct_reader.py",
        "fastlens/benchmarks/baselines/system_rgbnomore/training/model_factory.py",
        "fastlens/benchmarks/baselines/system_rgbnomore/training/pipeline.py",
        "fastlens/benchmarks/baselines/dct_baselines/store.py",
        "fastlens/benchmarks/baselines/dct_baselines/contracts.py",
        "fastlens/benchmarks/baselines/system_rgbnomore/training/sample_order.py",
        "fastlens/benchmarks/baselines/system_rgbnomore/training/equal_image_epoch_benchmark.py",
        "fastlens/benchmarks/training_pls/report_equal_image_performance.py",
        "fastlens/benchmarks/common.py",
        "fastlens/benchmarks/training_audit_policy.py",
        "fastlens/benchmarks/model_only_training_calibration.py",
        "fastlens/include/fastlens/direct_dct_pls.hpp",
        "fastlens/include/fastlens/profiles/rgbnomore.hpp",
        "fastlens/src/api/direct_dct_pls.cpp",
        "fastlens/src/api/direct_dct_pls_postprocess.cu",
        "fastlens/src/api/direct_dct_pls_postprocess.hpp",
        "fastlens/torch/__init__.py",
        "fastlens/torch/direct_dct.py",
        "fastlens/torch/direct_dct_pls_torch.cpp",
        "fastlens/torch/direct_dct_pls_torch.hpp",
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
            repo_root / "fastlens/benchmarks/training_pls"
        ).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".md"}
    )
    for extra in (
        repo_root / "fastlens/benchmarks/training_pls/tests/test_training_pls.py",
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
    prefix = "fastlens/benchmarks/training_pls/"
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
            "regenerate it with fastlens.benchmarks.training_pls.run_matrix"
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
    model_source_provenance: Sequence[Mapping[str, str]],
    audit_policy: Mapping[str, Any],
    required_gpu_name_substring: str,
    execution_backend: str = SEMANTIC_BACKEND,
    physical_execution: Mapping[str, Any] | None = None,
    standard_dct_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    condition = resolve_condition(condition_id)
    layout_hash = str(layout_plan["layout_hash"])
    if execution_backend not in EXECUTION_BACKENDS:
        raise ValueError(f"unsupported execution backend: {execution_backend!r}")
    native_physical = execution_backend == NATIVE_PHYSICAL_BACKEND
    standard_dct = execution_backend in DCT_REFERENCE_BACKENDS
    resolved_audit_policy = validate_audit_policy(audit_policy)
    if native_physical and physical_execution is None:
        raise ValueError("native-physical-pls requires a physical execution contract")
    if not native_physical and physical_execution is not None:
        raise ValueError(
            f"{execution_backend} cannot carry a physical execution contract"
        )
    if standard_dct and standard_dct_reference is None:
        raise ValueError(
            "standard-rgbnomore-dct requires a premixed reference contract"
        )
    if not standard_dct and standard_dct_reference is not None:
        raise ValueError(
            f"{execution_backend} cannot carry a standard DCT reference contract"
        )
    resolved_standard_reference = dict(standard_dct_reference or {})
    if standard_dct:
        mapping_path = str(
            resolved_standard_reference.get("premixed_mapping_csv", "")
        )
        mapping_hash = str(
            resolved_standard_reference.get("premixed_mapping_sha256", "")
        )
        if not mapping_path:
            raise ValueError("standard DCT reference lacks premixed_mapping_csv")
        if len(mapping_hash) != 64:
            raise ValueError(
                "standard DCT reference premixed_mapping_sha256 must contain "
                "64 hex characters"
            )
        try:
            int(mapping_hash, 16)
        except ValueError as error:
            raise ValueError(
                "standard DCT reference premixed_mapping_sha256 must contain "
                "64 hex characters"
            ) from error
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
        "model_id": recipe["model"]["model_id"],
        "model_input_contract": recipe["model"]["input_contract"],
        "model_source_provenance": [dict(row) for row in model_source_provenance],
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
        "audit_policy": resolved_audit_policy,
        "model_only_calibration": {
            "enabled": True,
            "domain": "dct",
            "microbatch_images": recipe["training"]["physical_microbatch"],
            "gradient_accumulation": recipe["training"]["gradient_accumulation"],
            "precision": recipe["training"]["precision"],
            "fixed_pre_resident_gpu_inputs": True,
            "warmup_optimizer_updates": 5,
            "measured_optimizer_updates": 120,
            "subtraction_from_e2e_forbidden": True,
        },
        "checkpoint_schedule": {
            "rolling": "every completed epoch",
            "permanent": "every formal validation epoch",
            "resume_granularity": "completed epoch",
        },
        "backend_implementation": BACKEND_IMPLEMENTATIONS[execution_backend],
        "recipe_hash": recipe["recipe_hash"],
        "code_version": dict(code),
        "execution_device": str(device),
        "required_gpu_name_substring": str(required_gpu_name_substring),
        "execution_mode": EXECUTION_MODES[execution_backend],
        "semantic_emulation": execution_backend == SEMANTIC_BACKEND,
        "physical_fls_observed": native_physical,
        "physical_gpu_pool": native_physical,
        "layout_hash": layout_hash,
        "reference_topology": "8gpu_ddp",
        "execution_topology": "single_gpu_accum16",
        "distributed_bitwise_equivalence_claim": False,
    }
    if native_physical:
        contract["physical_execution"] = dict(physical_execution or {})
    if standard_dct:
        contract["standard_dct_reference"] = {
            **resolved_standard_reference,
            "schema_version": "galp-standard-rgbnomore-dct-reference-v2",
            "training_input": "source JPEG decoded by RGB-no-more dct_manip",
            "logical_schedule": (
                "physical-writer premixed mapping followed by identical condition "
                "schedule and augmentation keys"
            ),
            "validation_input": "shared FASTLENS Direct-DCT validation reader",
            "comparison_scope": (
                "recipe-matched DCT reference for convergence or validated "
                "equal-image E1/E2 system performance; no physical FLS or "
                "storage-reduction claim"
            ),
        }
        if execution_backend in (IC_DCT_BACKEND, FS_DCT_BACKEND):
            if "baseline_store" not in resolved_standard_reference:
                raise ValueError("IC/FS reference lacks baseline_store identity")
            contract["execution_backend"] = execution_backend
            contract["headline_eligible"] = False
            contract["standard_dct_reference"]["training_input"] = "original JPEG q/Q from " + resolved_standard_reference["baseline_store"]["method"]
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
