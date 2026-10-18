"""Independent numerical calibration and published-recipe G1/G2/G3 checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from galp.benchmarks.system_rgbnomore.training.artifacts import tensor_state_sha256
from galp.benchmarks.system_rgbnomore.training.model_factory import (
    capture_rng_state, restore_rng_state, rng_state_sha256, seed_everything,
)
from .contracts import artifact, load_trace, write_json
from .runner import implementation_artifacts, inputs
from .training import make_optimizer, snapshot, train_window, update_windows
from .unified import UnifiedReplay, build_training_model, microbatches


def create_snapshot(config_path, output):
    config, _, profile, _, _ = inputs(config_path)
    if not profile.get("training"):
        raise ValueError("snapshot requires a training contract")
    original_rng = capture_rng_state()
    try:
        seed_everything(config["trace_generation_seed"])
        model = build_training_model(profile, "cpu", seed=config["trace_generation_seed"])
        state = snapshot(model, config["training"])
        state.update(profile=artifact(config["model_profile"]["path"]), recipe=config["training"],
                     initial_model_sha256=tensor_state_sha256(model.state_dict()))
        with Path(output).open("xb") as stream:
            torch.save(state, stream)
    finally:
        restore_rng_state(original_rng)
    return dict(snapshot=artifact(output), initial_model_sha256=state["initial_model_sha256"])


class Comparisons:
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.maximum = {}
        self.expected = []

    def compare(self, stage, actual, expected):
        if len(actual) != len(expected):
            raise AssertionError(f"{stage}: tensor cardinality differs")
        limits = self.tolerance["inputs" if stage == "inputs" else
                                "state" if stage in ("parameters", "optimizer") else "model"]
        error = 0.
        for a, b in zip(actual, expected):
            a, b = a.detach().cpu(), b.detach().cpu()
            torch.testing.assert_close(a, b, atol=limits["atol"], rtol=limits["rtol"], msg=lambda msg: f"{stage}: {msg}")
            if a.numel():
                error = max(error, float((a.double()-b.double()).abs().max()))
        self.maximum[stage] = max(self.maximum.get(stage, 0.), error)

    def reference(self, stage, values):
        self.expected.append((stage, [v.detach().cpu().clone() for v in values]))

    def candidate(self, stage, values):
        expected_stage, expected = self.expected.pop(0)
        if expected_stage != stage:
            raise AssertionError("training observation order differs")
        self.compare(stage, values, expected)


def paired_updates(model, actual_batches, reference_batches, windows, state, recipe, comparisons):
    """Both executions restore the complete initial model/optimizer/decay/schedule/RNG."""
    reference = copy.deepcopy(model)
    candidate = model
    optimizers = []
    for net in (reference, candidate):
        net.load_state_dict(state["model"], strict=True)
        optimizer, decay, scheduler = make_optimizer(net, recipe)
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        decay.load_state_dict(state["decay"])
        scheduler.load_state_dict(state["scheduler"])
        optimizers.append((optimizer, decay, scheduler))
    original_rng = capture_rng_state()
    results = []
    try:
        restore_rng_state(state["rng"])
        for window in windows:
            ids = [batch[0]["microbatch_id"] for batch in microbatches(window)]
            step_rng = capture_rng_state()
            expected = train_window(reference, *optimizers[0], [reference_batches[i] for i in ids], recipe, comparisons.reference)
            expected_rng = rng_state_sha256(capture_rng_state())
            restore_rng_state(step_rng)
            actual = train_window(candidate, *optimizers[1], [actual_batches[i] for i in ids], recipe, comparisons.candidate)
            if comparisons.expected or actual["scheduler"] != expected["scheduler"] or actual["decay"] != expected["decay"]:
                raise AssertionError("optimizer observation, scheduler or decay state differs")
            if rng_state_sha256(capture_rng_state()) != expected_rng:
                raise AssertionError("paired training consumed different RNG streams")
            results.append(actual)
    finally:
        restore_rng_state(original_rng)
    return results


def numerical_runtime(device):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
                matmul_precision="highest", tf32=False, cudnn_deterministic=True)


def verify_training(config_path, snapshot_path, backend, root, output, *, device="cpu", module_path=None,
                    updates=10, calibration_path=None, tolerance=None):
    config, source, profile, records, trace = inputs(config_path)
    if not profile.get("training") or updates <= 0:
        raise ValueError("training gates require a training contract and positive update count")
    runtime = numerical_runtime(device)
    torch.set_num_threads(config["torch_threads"])
    implementation = implementation_artifacts()
    snapshot_ref = artifact(snapshot_path)
    # Locally produced full training snapshots contain Python/NumPy RNG state.
    state = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if state["profile"] != config["model_profile"] or state["recipe"] != config["training"]:
        raise ValueError("initial snapshot profile/recipe differs from the contract")
    if state["initial_model_sha256"] != tensor_state_sha256(state["model"]):
        raise ValueError("initial model state identity differs from its snapshot")
    from .contracts import load_artifact, validate_batches
    labels = load_artifact(config["label_mapping"])["labels"]
    golden = load_trace(config["calibration_trace"]["path"], profile, records, labels)
    validate_batches(golden, config, training=True)
    if artifact(config["calibration_trace"]["path"]) != config["calibration_trace"]:
        raise ValueError("calibration trace changed")
    if {r["image_id"] for r in golden} & {r["image_id"] for r in trace}:
        raise ValueError("calibration and verification images must be disjoint")
    calibration = None
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text())
        expected = dict(resolved_contract=artifact(config_path), snapshot=snapshot_ref, runtime=runtime,
                        implementation=implementation, calibration_trace=config["calibration_trace"])
        if calibration["status"] != "passed" or any(calibration[key] != value for key, value in expected.items()):
            raise ValueError("stale or incompatible numerical calibration")
        tolerance = calibration["tolerance"]
        if updates != calibration["updates"]:
            raise ValueError("verification update count differs from calibration")
    else:
        trace = golden
    if tolerance is None:
        raise ValueError("calibration needs explicit predeclared input/model/state tolerances")
    comparisons = Comparisons(tolerance)
    windows = update_windows(trace, config["training"]["accumulation"])
    if len(windows) < updates:
        raise ValueError(f"trace has only {len(windows)} optimizer updates; freeze more epochs to check {updates}")
    windows = windows[:updates]
    checked_trace = [row for window in windows for row in window]
    with UnifiedReplay(backend, root, config, source, profile, device=device, module_path=module_path) as replay:
        replay.verify_source()
        if calibration and replay.identity != calibration["backend"]:
            raise ValueError("calibration backend artifacts differ")
        actual, expected = {}, {}
        for rows in microbatches(checked_trace):
            index = rows[0]["microbatch_id"]
            a, b = replay.batch(rows), replay.batch(rows, oracle=True)
            comparisons.compare("inputs", a[0], b[0])
            torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
            actual[index], expected[index] = a, b
        model = build_training_model(profile, device)
        results = paired_updates(model, actual, expected, windows, state, config["training"], comparisons)
        evidence = dict(status="passed", phase="verification" if calibration else "calibration",
                        resolved_contract=artifact(config_path), snapshot=snapshot_ref,
                        initial_model_sha256=state["initial_model_sha256"], runtime=runtime, implementation=implementation,
                        calibration_trace=config["calibration_trace"], backend=replay.identity,
                        source_q_and_tables_exact=True, target_tensors_exact=True,
                        max_abs_errors=comparisons.maximum, tolerance=tolerance, updates=updates,
                        occurrence_count=len(checked_trace), unique_image_count=len({r["image_id"] for r in checked_trace}),
                        training=results, headline_eligible=False)
        if calibration_path:
            evidence["calibration"] = artifact(calibration_path)
    write_json(output, evidence)
    return evidence
