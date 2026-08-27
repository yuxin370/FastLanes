#!/usr/bin/env python3
"""Resolve run manifests, then optionally execute the registered PLS matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .contracts import (
    build_condition_contract,
    code_version,
    validate_seed_block_contracts,
)
from .layout import load_layout_mapping, load_layout_plan, sha256_file
from .matrix import (
    CORE_CONDITION_IDS,
    PAIRED_SEEDS,
    core_matrix,
    experiment_matrix,
    execution_order,
    resolve_condition,
)
from .recipe import RECIPE_NAME, assert_recipe_overrides, sha256_json


REPO_ROOT = Path(__file__).resolve().parents[4]
RGB_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "system_rgbnomore"
if str(RGB_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(RGB_BENCHMARK_ROOT))

from training.artifacts import tensor_state_sha256  # noqa: E402
from training.model_factory import build_model, seed_everything  # noqa: E402


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parse_csv(raw: str, *, cast: Any = str) -> list[Any]:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("comma-separated option contains no values")
    return values


def _condition_ids(raw: str) -> list[str]:
    values = [str(value).upper() for value in _parse_csv(raw)]
    if len(set(values)) != len(values):
        raise ValueError("--conditions contains duplicate condition IDs")
    for value in values:
        resolve_condition(value)
    return values


def _seeds(raw: str) -> list[int]:
    values = _parse_csv(raw, cast=int)
    if len(set(values)) != len(values):
        raise ValueError("--seeds contains duplicates")
    unexpected = sorted(set(values) - set(PAIRED_SEEDS))
    if unexpected:
        raise ValueError(
            f"core matrix accepts only pre-registered paired seeds {PAIRED_SEEDS}; got {unexpected}"
        )
    return values


def _seed_devices(
    raw: str | None, *, seeds: Sequence[int], default_device: str
) -> dict[int, str]:
    if raw is None:
        return {int(seed): str(default_device) for seed in seeds}
    assignments: dict[int, str] = {}
    for entry in _parse_csv(raw):
        seed_text, separator, device = str(entry).partition("=")
        if not separator or not seed_text.strip() or not device.strip():
            raise ValueError(
                "--seed-devices entries must use the form training_seed=device"
            )
        seed = int(seed_text)
        if seed in assignments:
            raise ValueError(f"--seed-devices repeats seed {seed}")
        assignments[seed] = device.strip()
    expected = {int(seed) for seed in seeds}
    if set(assignments) != expected:
        raise ValueError(
            "--seed-devices must assign every selected seed exactly once; "
            f"expected {sorted(expected)}, got {sorted(assignments)}"
        )
    return assignments


def _initial_model_hashes(seeds: Sequence[int], rgbnomore_root: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    device = torch.device("cpu")
    for seed in seeds:
        seed_everything(seed)
        model = build_model(rgbnomore_root, "dct", device)
        result[seed] = tensor_state_sha256(model.state_dict())
        del model
    return result


def _updates_per_epoch(mapping: Any, condition_id: str) -> int:
    condition = resolve_condition(condition_id)
    microbatch = 64
    accumulation = 16
    if condition["order_policy"] == "global":
        return math.ceil(math.ceil(mapping.sample_count / microbatch) / accumulation)
    sizes = [len(positions) for positions in mapping.positions_by_pls]
    segments_per_pool = int(condition["segments_per_pool"])
    # The formal 1,252-PLS layout is divisible by four, so shuffling PLS IDs
    # cannot change the multiset of pool sizes: 312 full pools plus one pool
    # containing the tail PLS.  Reject layouts where this shortcut is unsafe.
    if len(sizes) % segments_per_pool:
        raise ValueError(
            "core closed-pool total-update contract requires virtual_pls_count "
            "to be divisible by segments_per_pool"
        )
    full_size = int(mapping.plan["target_pls_size"])
    non_full = [size for size in sizes if size != full_size]
    if len(non_full) > 1:
        raise ValueError("core layout may contain at most one tail PLS")
    full_pool_samples = full_size * segments_per_pool
    full_pool_updates = math.ceil(
        math.ceil(full_pool_samples / microbatch) / accumulation
    )
    pool_count = len(sizes) // segments_per_pool
    if not non_full:
        return pool_count * full_pool_updates
    tail_pool_samples = full_size * (segments_per_pool - 1) + non_full[0]
    tail_pool_updates = math.ceil(
        math.ceil(tail_pool_samples / microbatch) / accumulation
    )
    return (pool_count - 1) * full_pool_updates + tail_pool_updates


def _environment(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "requested_device": device,
        "cuda_available_in_planner_process": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        payload["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "uuid": (
                    None
                    if getattr(torch.cuda.get_device_properties(index), "uuid", None)
                    is None
                    else str(torch.cuda.get_device_properties(index).uuid)
                ),
                "total_memory": int(
                    torch.cuda.get_device_properties(index).total_memory
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    return payload


def _write_status_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "seed",
        "condition",
        "condition_position",
        "device",
        "state",
        "returncode",
        "output_dir",
        "started_at_unix",
        "ended_at_unix",
        "retry_reason",
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_failure(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    recipe = assert_recipe_overrides(recipe=RECIPE_NAME, epochs=args.epochs)
    conditions = _condition_ids(args.conditions)
    seeds = _seeds(args.seeds)
    seed_devices = _seed_devices(
        args.seed_devices, seeds=seeds, default_device=args.device
    )
    layout_plan = load_layout_plan(args.layout_plan)
    mapping = load_layout_mapping(args.layout_plan)
    if int(layout_plan["target_pls_size"]) != 1024:
        raise ValueError("core matrix requires a frozen G=1024 layout")
    if sha256_file(args.train_manifest.resolve()) != layout_plan["dataset_manifest_hash"]:
        raise ValueError("--train-manifest differs from the frozen physical layout plan")
    physical_execution: dict[str, Any] | None = None
    if args.execution_backend == "native-physical-pls":
        actual_mapping_hash = sha256_file(args.premixed_mapping_csv.resolve())
        if actual_mapping_hash != args.expected_mapping_sha256:
            raise ValueError(
                "--expected-mapping-sha256 differs from --premixed-mapping-csv"
            )
        physical_execution = {
            "schema_version": "galp-native-physical-pls-execution-v1",
            "semantic_profile": "rgbnomore-training-pls-v1",
            "physical_galp_manifest": str(args.physical_galp_manifest.resolve()),
            "physical_galp_manifest_sha256": sha256_file(
                args.physical_galp_manifest.resolve()
            ),
            "premixed_mapping_csv": str(args.premixed_mapping_csv.resolve()),
            "premixed_mapping_sha256": actual_mapping_hash,
            "segment_images": 1024,
            "segments_per_closed_pool": 4,
            "microbatch_images": 64,
            "native_crop_pushdown": True,
            "native_physical_order": True,
            "gpu_resident_closed_pool": True,
            "cuda_transform": True,
            "cuda_ordered_output_placement": True,
            "cuda_mixup": True,
            "sample_order_policy_from_condition_contract": True,
        }
    code = code_version(REPO_ROOT)
    initial_hashes = _initial_model_hashes(seeds, args.rgbnomore_root)
    updates_by_condition = {
        condition_id: _updates_per_epoch(mapping, condition_id) * args.epochs
        for condition_id in conditions
    }
    if len(set(updates_by_condition.values())) != 1:
        raise ValueError(
            f"core condition total optimizer updates differ: {updates_by_condition}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_by_seed: dict[int, list[dict[str, Any]]] = {}
    contract_paths: dict[tuple[int, str], Path] = {}
    legacy_contract_paths: dict[tuple[int, str], Path] = {}
    for seed in seeds:
        contracts: list[dict[str, Any]] = []
        for condition_id in conditions:
            contract = build_condition_contract(
                condition_id=condition_id,
                seed=seed,
                train_manifest=args.train_manifest,
                val_manifest=args.val_manifest,
                layout_plan=layout_plan,
                recipe=recipe,
                total_optimizer_updates=updates_by_condition[condition_id],
                initial_model_hash=initial_hashes[seed],
                code=code,
                device=seed_devices[seed],
                execution_backend=args.execution_backend,
                physical_execution=physical_execution,
            )
            contracts.append(contract)
            legacy_contract_path = (
                output_dir
                / "contracts"
                / f"seed_{seed}"
                / f"{condition_id}.json"
            )
            run_dir = output_dir / "runs" / condition_id / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_manifest_path = run_dir / "run_manifest.json"
            _atomic_json(run_manifest_path, contract)
            contract_paths[(seed, condition_id)] = run_manifest_path
            # Compatibility artifacts for existing reports and operational tools.
            # New commands consume run_manifest.json directly.
            _atomic_json(legacy_contract_path, contract)
            _atomic_json(run_dir / "condition_contract.json", contract)
            legacy_contract_paths[(seed, condition_id)] = legacy_contract_path
        contracts_by_seed[seed] = contracts

    diff_blocks = []
    for seed in seeds:
        if set(conditions) == set(CORE_CONDITION_IDS):
            diff_blocks.append(validate_seed_block_contracts(contracts_by_seed[seed]))
        else:
            diff_blocks.append(
                {
                    "training_seed": seed,
                    "valid": None,
                    "reason": "partial condition subset; full whitelist comparison not applicable",
                }
            )
    _atomic_json(
        output_dir / "condition_contract_diff.json",
        {
            "schema_version": "galp-pls-condition-contract-diff-collection-v2",
            "all_full_blocks_valid": all(
                block.get("valid") is not False for block in diff_blocks
            ),
            "seed_blocks": diff_blocks,
        },
    )
    _atomic_json(output_dir / "recipe_contract.json", recipe)
    execution_rows = [
        {
            "seed": seed,
            "condition_position": position,
            "condition": condition_id,
            "device": seed_devices[seed],
        }
        for seed, position, condition_id in execution_order(seeds)
        if condition_id in conditions
    ]
    _atomic_json(
        output_dir / "condition_execution_order.json",
        {
            "balanced": not any(
                int(row["condition_position"]) > len(CORE_CONDITION_IDS)
                for row in execution_rows
            ),
            "rows": execution_rows,
            "policy": (
                "fixed Latin rotation for core conditions; supplemental controls "
                "follow the registered core order"
            ),
        },
    )
    registered_matrix = experiment_matrix(conditions)
    analysis_contract = {
        "schema_version": "galp-pls-analysis-contract-v2",
        "matrix_hash": sha256_json(registered_matrix),
        "primary_endpoint": "final-checkpoint validation top-1",
        "secondary_endpoints": [
            "final top-5",
            "final validation loss",
            "top-1 convergence curve",
            "normalized top-1 AUC",
            "fixed-epoch top-1/top-5",
            "train-loss curve",
        ],
        "paired_before_aggregate": True,
        "confidence_interval": "two-sided 95% Student-t",
        "practical_top1_margin_percentage_points": 0.3,
        "confidence_band": "pointwise, not simultaneous",
        "winner_ranking": False,
        "system_metric_strategy_gate": False,
        "estimands": core_matrix()["estimands"],
    }
    if "supplemental_estimands" in registered_matrix:
        analysis_contract["supplemental_estimands"] = registered_matrix[
            "supplemental_estimands"
        ]
        analysis_contract["supplemental_interpretation"] = registered_matrix[
            "supplemental_interpretation"
        ]
    _atomic_json(output_dir / "analysis_contract.json", analysis_contract)
    environment = _environment(args.device)
    environment["requested_seed_devices"] = {
        str(seed): seed_devices[seed] for seed in seeds
    }
    _atomic_json(output_dir / "environment.json", environment)

    commands: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    first_seed = seeds[0]
    for row in execution_rows:
        seed = int(row["seed"])
        condition_id = str(row["condition"])
        run_dir = output_dir / "runs" / condition_id / f"seed_{seed}"
        argv = [
            str(Path(sys.executable).resolve()),
            "-m",
            "training_pls.train",
            "--train-manifest",
            str(args.train_manifest.resolve()),
            "--val-manifest",
            str(args.val_manifest.resolve()),
            "--layout-plan",
            str(args.layout_plan.resolve()),
            "--run-manifest",
            str(contract_paths[(seed, condition_id)].resolve()),
            "--condition",
            condition_id,
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--device",
            seed_devices[seed],
            "--output-dir",
            str(run_dir),
            "--workers",
            str(args.workers),
            "--prefetch-depth",
            str(args.prefetch_depth),
            "--galp-torch-module-path",
            str(args.galp_torch_module_path.resolve()),
            "--rgbnomore-root",
            str(args.rgbnomore_root.resolve()),
            "--resume",
        ]
        if args.execution_backend == "native-physical-pls":
            argv.extend(
                [
                    "--execution-backend",
                    "native-physical-pls",
                    "--physical-galp-manifest",
                    str(args.physical_galp_manifest.resolve()),
                    "--premixed-mapping-csv",
                    str(args.premixed_mapping_csv.resolve()),
                    "--expected-mapping-sha256",
                    args.expected_mapping_sha256,
                ]
            )
        if seed == first_seed:
            argv.append("--integration-check-first-100")
        if args.stop_after_epoch is not None:
            argv.extend(["--stop-after-epoch", str(args.stop_after_epoch)])
        commands.append(
            {
                **row,
                "output_dir": str(run_dir),
                "run_manifest": str(contract_paths[(seed, condition_id)]),
                "condition_contract": str(
                    legacy_contract_paths[(seed, condition_id)]
                ),
                "argv": argv,
                "shell": shlex.join(argv),
            }
        )
        statuses.append(
            {
                **row,
                "state": "planned",
                "returncode": "",
                "output_dir": str(run_dir),
                "started_at_unix": "",
                "ended_at_unix": "",
                "retry_reason": "",
            }
        )
    _write_status_csv(output_dir / "run_status.csv", statuses)
    failures_path = output_dir / "failures.jsonl"
    failures_path.touch(exist_ok=True)
    plan = {
        "schema_version": "galp-pls-core-execution-plan-v2",
        "execute_requested": bool(args.execute),
        "matrix": registered_matrix,
        "conditions": conditions,
        "seeds": seeds,
        "seed_devices": {str(seed): seed_devices[seed] for seed in seeds},
        "epochs": args.epochs,
        "stop_after_epoch": args.stop_after_epoch,
        "total_runs": len(commands),
        "layout_hash": layout_plan["layout_hash"],
        "recipe_hash": recipe["recipe_hash"],
        "execution_backend": args.execution_backend,
        "physical_execution": physical_execution,
        "total_optimizer_updates_per_run": next(iter(updates_by_condition.values())),
        "commands": commands,
        "result_policy": (
            "retain every run; never filter conditions using accuracy, throughput, "
            "memory, loader wait, or class-mixing metrics"
        ),
    }
    _atomic_json(output_dir / "matrix_execution_plan.json", plan)
    return {"plan": plan, "statuses": statuses}


def execute_plan(
    output_dir: Path,
    plan: Mapping[str, Any],
    statuses: list[dict[str, Any]],
    *,
    continue_on_error: bool,
) -> int:
    status_by_key = {
        (int(row["seed"]), str(row["condition"])): row for row in statuses
    }
    failures_path = output_dir / "failures.jsonl"
    overall = 0
    for command in plan["commands"]:
        key = (int(command["seed"]), str(command["condition"]))
        status = status_by_key[key]
        final_result = Path(command["output_dir"]) / "final_result.json"
        if final_result.is_file():
            status.update(state="completed", returncode=0)
            _write_status_csv(output_dir / "run_status.csv", statuses)
            continue
        requested_stop = plan.get("stop_after_epoch")
        existing_status_path = Path(command["output_dir"]) / "run_status.json"
        if requested_stop is not None and existing_status_path.is_file():
            try:
                existing_status = json.loads(
                    existing_status_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                existing_status = {}
            if (
                existing_status.get("state") == "paused-at-epoch-boundary"
                and int(existing_status.get("completed_epoch", -1))
                >= int(requested_stop)
            ):
                status.update(
                    state="paused-at-epoch-boundary",
                    returncode=0,
                    retry_reason="requested epoch boundary already completed",
                )
                _write_status_csv(output_dir / "run_status.csv", statuses)
                continue
        status.update(state="running", started_at_unix=time.time())
        _write_status_csv(output_dir / "run_status.csv", statuses)
        completed = subprocess.run(command["argv"], check=False)
        run_state = "completed" if completed.returncode == 0 else "failed"
        run_status_path = Path(command["output_dir"]) / "run_status.json"
        if completed.returncode == 0 and run_status_path.is_file():
            try:
                recorded_state = json.loads(
                    run_status_path.read_text(encoding="utf-8")
                ).get("state")
            except (OSError, json.JSONDecodeError):
                recorded_state = None
            if recorded_state == "paused-at-epoch-boundary":
                run_state = recorded_state
        status.update(
            state=run_state,
            returncode=completed.returncode,
            ended_at_unix=time.time(),
        )
        _write_status_csv(output_dir / "run_status.csv", statuses)
        if completed.returncode != 0:
            overall = completed.returncode
            failure = {
                "seed": key[0],
                "condition": key[1],
                "condition_position": command["condition_position"],
                "returncode": completed.returncode,
                "run_failure_artifact": str(
                    Path(command["output_dir"]) / "failure.json"
                ),
                "ended_at_unix": status["ended_at_unix"],
            }
            _append_failure(failures_path, failure)
            if not continue_on_error:
                return overall
    return overall


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--layout-plan", type=Path, required=True)
    parser.add_argument("--conditions", default=",".join(CORE_CONDITION_IDS))
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in PAIRED_SEEDS)
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help=(
            "operational epoch-boundary pause forwarded to every selected run; "
            "the scientific recipe remains fixed at 300 epochs"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--execution-backend",
        choices=("semantic-emulation", "native-physical-pls"),
        default="semantic-emulation",
    )
    parser.add_argument("--physical-galp-manifest", type=Path)
    parser.add_argument("--premixed-mapping-csv", type=Path)
    parser.add_argument("--expected-mapping-sha256")
    parser.add_argument(
        "--seed-devices",
        help=(
            "optional comma-separated per-seed execution devices, for example "
            "11997733=cuda:1,11997734=cuda:0"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument(
        "--galp-torch-module-path", type=Path, default=REPO_ROOT / "build/galp/torch"
    )
    parser.add_argument(
        "--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more")
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.prefetch_depth < 0:
        raise ValueError("prefetch depth must be non-negative")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch < 300:
        raise ValueError("--stop-after-epoch must be in [1, 299]")
    for path in (args.train_manifest, args.val_manifest, args.layout_plan):
        if not path.is_file():
            raise FileNotFoundError(path)
    physical_values = (
        args.physical_galp_manifest,
        args.premixed_mapping_csv,
        args.expected_mapping_sha256,
    )
    if args.execution_backend == "native-physical-pls":
        if any(value is None for value in physical_values):
            raise ValueError(
                "native-physical-pls requires --physical-galp-manifest, "
                "--premixed-mapping-csv, and --expected-mapping-sha256"
            )
        for path in (args.physical_galp_manifest, args.premixed_mapping_csv):
            if not path.is_file():
                raise FileNotFoundError(path)
        if len(args.expected_mapping_sha256) != 64:
            raise ValueError("--expected-mapping-sha256 must contain 64 hex characters")
        try:
            int(args.expected_mapping_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "--expected-mapping-sha256 must contain 64 hex characters"
            ) from error
    elif any(value is not None for value in physical_values):
        raise ValueError(
            "physical PLS paths are accepted only with "
            "--execution-backend native-physical-pls"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    built = build_plan(args)
    plan = built["plan"]
    if not args.execute:
        for command in plan["commands"]:
            print(command["shell"])
        return 0
    return execute_plan(
        args.output_dir.resolve(),
        plan,
        built["statuses"],
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
