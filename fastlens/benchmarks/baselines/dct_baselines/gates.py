"""Independent numerical calibration and published-recipe G1/G2/G3 checks."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import time

import torch

from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import (
    capture_rng_state, restore_rng_state, rng_state_sha256, seed_everything,
)
from .contracts import artifact, load_trace, write_json
from .runner import implementation_artifacts, inputs
from .training import augment, finish_batch, make_optimizer, snapshot, train_window, update_windows
from .unified import UnifiedReplay, build_training_model, cpu_raw, microbatches


def verify_inference(config_path, backend, root, output, *, device="cpu", module_path=None, source_workers=1,
                     source_evidence=None):
    """Pretrained checkpoint gate on each system's own preprocessing path."""
    from .materialize import build_model, source_reference
    from .source import source_images
    from .runner import assert_tensors, cold_input_files
    if source_workers <= 0:
        raise ValueError("source-workers must be positive")
    config, source, profile, records, trace = inputs(config_path)
    if profile.get("training"):
        raise ValueError("prepare an inference contract; training augmentation/snapshots are not checkpoint inference")
    runtime = numerical_runtime(device)
    torch.set_num_threads(config["torch_threads"])
    model = build_model(profile, torch.device(device)).eval().requires_grad_(False)
    state_hash = tensor_state_sha256(model.state_dict())
    model_code = [artifact(path) for path in sorted({inspect.getfile(type(m)) for m in model.modules()})]
    atol, rtol = config["tolerance"]["atol"], config["tolerance"]["rtol"]
    count = top1 = top5 = 0
    input_error = logit_error = loss = reference_loss = 0.
    started = time.perf_counter()
    # Bind the full numerical gate to compressed inputs as well as the manifest.
    # Later timing repeats check these bytes without decoding every source again.
    native_input_files = ([artifact(path) for path in cold_input_files(backend, root, records, None)]
                          if backend == "fastlens" else None)
    with UnifiedReplay(backend, root, config, source, profile, device=device, module_path=module_path) as replay:
        if source_evidence is None:
            replay.verify_source(workers=source_workers)
        else:
            from .source import verify_source_files
            prior = json.loads(Path(source_evidence).read_text())
            if backend == "fastlens":
                from fastlens.benchmarks.dct_models.storage import BRIDGE
                replay.identity["coefficient_bridge"] = artifact(BRIDGE)
            if (prior["status"] != "passed" or not prior["source_q_exact"] or not prior["quantization_tables_exact"]
                    or prior["source_manifest"] != config["source_manifest"]
                    or prior["implementation"] != implementation_artifacts() or prior["backend"] != replay.identity
                    or (backend == "fastlens" and (prior["native_input_files"] != native_input_files
                        or any(artifact(ref["path"]) != ref for ref in prior["native_sources"])))):
                raise ValueError("source verification evidence does not match current source/store/implementation")
            verify_source_files(source)
            print("Reusing unchanged source q/Q evidence; checking every final input and prediction again.", flush=True)
        preflight_s = time.perf_counter()-started
        with torch.inference_mode():
            for rows, actual in replay.inference_batches(trace):
                values = [source_reference(profile, image, row) for image, row in
                          zip(source_images(dict(images=[records[r["image_id"]] for r in rows])), rows)]
                expected = tuple(torch.stack([v[i] for v in values]).to(device) for i in range(len(values[0])))
                input_error = max(input_error, assert_tensors(actual, expected, atol, rtol))
                logits, reference = model(*actual), model(*expected)
                logit_error = max(logit_error, assert_tensors((logits,), (reference,), atol, rtol))
                rank, reference_rank = logits.topk(5, dim=1).indices, reference.topk(5, dim=1).indices
                torch.testing.assert_close(rank[:, 0], reference_rank[:, 0], atol=0, rtol=0)
                torch.testing.assert_close(rank.sort(1).values, reference_rank.sort(1).values, atol=0, rtol=0)
                labels = torch.tensor([r["target_label"] for r in rows], device=device)
                top1 += int((rank[:, 0] == labels).sum())
                top5 += int((rank == labels[:, None]).any(1).sum())
                actual_loss = torch.nn.functional.cross_entropy(logits, labels)
                expected_loss = torch.nn.functional.cross_entropy(reference, labels)
                assert_tensors((actual_loss,), (expected_loss,), atol, rtol)
                loss += float(actual_loss)*len(rows)
                reference_loss += float(expected_loss)*len(rows)
                count += len(rows)
                print(f"Inference samples checked: {count}/{len(trace)}", flush=True)
        if count != len(trace) or tensor_state_sha256(model.state_dict()) != state_hash:
            raise AssertionError("inference changed model state or lost trace occurrences")
        evidence = dict(status="passed", phase="pretrained_inference", experiment_type="matched_system_inference",
                        execution_partition=dict(read_decode="native CUDA pipeline" if backend == "fastlens" else "cpu",
                            geometry_normalization="native CUDA" if backend == "fastlens" else "cpu",
                            augmentation="disabled", model=str(device)),
                        resolved_contract=artifact(config_path), source_manifest=config["source_manifest"],
                        source_evidence_reused=artifact(source_evidence) if source_evidence is not None else None,
                        checkpoint=profile["checkpoint"],
                        loaded_model_state_sha256=state_hash, model_implementation=model_code,
                        implementation=implementation_artifacts(), backend=replay.identity, runtime=runtime,
                        source_q_exact=True, quantization_tables_exact=True, dependency_closure_verified=True,
                        final_tensor_match=True, logits_match=True, predictions_match=True, training_update_match=False,
                        prediction_comparison="exact top1 and top5 class sets",
                        max_abs_tensor_error=input_error, max_relative_tensor_error=0. if input_error == 0 else None,
                        max_abs_logits_error=logit_error, tolerance=config["tolerance"], device=device,
                        source_image_count=len(records), occurrence_count=count,
                        metrics=dict(top1_fraction=top1/count, top5_fraction=top5/count,
                                     cross_entropy=loss/count, reference_cross_entropy=reference_loss/count),
                        source_preflight_wall_s=preflight_s, cache_state="unmeasured/post-verification",
                        headline_eligible=False)
        if replay.reader:
            evidence["store"] = replay.reader.identity
        if backend == "fastlens":
            evidence["native_input_files"] = native_input_files
            repo = Path(__file__).resolve().parents[4]
            evidence["native_sources"] = [artifact(repo/name) for name in (
                "fastlens/include/fastlens/jpeg_dct_device.hpp", "fastlens/include/fastlens/profiles/rgbnomore.hpp",
                "fastlens/src/jpeg/jpeg_dct_planner.cpp", "fastlens/src/jpeg/jpeg_dct_plan_types.hpp",
                "fastlens/src/jpeg/jpeg_dct_device.cu", "fastlens/src/jpeg/jpeg_dct_transform_kernels.cu",
                "fastlens/src/jpeg/jpeg_dct_kernel_launch.cuh", "fastlens/torch/direct_dct_torch.cpp")]
    write_json(output, evidence)
    return evidence


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


def input_failure_details(replay, rows, actual_raw, expected_raw):
    """Describe the failed batch without changing its inputs or tolerance."""
    raw_counts, augmented_counts = [0, 0, 0], [0, 0, 0]
    raw_maximum = [0., 0., 0.]
    examples = []
    for row, actual, expected in zip(rows, actual_raw, expected_raw):
        mismatches = [(a.detach().cpu() != b.detach().cpu()).nonzero() for a, b in zip(actual, expected)]
        for c, (a, b, indices) in enumerate(zip(actual, expected, mismatches)):
            raw_counts[c] += len(indices)
            raw_maximum[c] = max(raw_maximum[c], float((a.double()-b.double()).abs().max()))
        if replay.profile.get("training"):
            a = augment(actual, row["augmentation_parameters"], replay.dops)
            b = augment(expected, row["augmentation_parameters"], replay.dops, oracle=True)
            for c, (left, right) in enumerate(zip(a, b)):
                augmented_counts[c] += int((left != right).sum())
        if any(len(indices) for indices in mismatches) and len(examples) < 8:
            unrounded = cpu_raw(replay.records[row["image_id"]], row, replay.profile, replay.dops, rounded=False)
            coefficients = []
            for c, indices in enumerate(mismatches):
                for index in indices[:4].tolist():
                    coordinate = tuple(index)
                    value = float(unrounded[c][coordinate])
                    coefficients.append(dict(component=c, index=index,
                        native=float(actual[c][coordinate]), cpu=float(expected[c][coordinate]),
                        cpu_unrounded_after_flip=value,
                        distance_to_half_integer=abs(value-(round(value-.5)+.5))))
            examples.append(dict(image_id=row["image_id"], occurrence_id=row["occurrence_id"],
                galp_image_id=replay.records[row["image_id"]].get("galp_image_id"),
                crop_box=row["crop_box"], flip=row["flip"],
                augmentation_parameters=row.get("augmentation_parameters", []), coefficients=coefficients))
    return dict(microbatch_id=rows[0]["microbatch_id"],
                raw_mismatches_by_component=raw_counts, raw_max_abs_by_component=raw_maximum,
                augmented_mismatches_by_component=augmented_counts if replay.profile.get("training") else None,
                examples=examples, native_pre_round_accumulator_exposed=False)


def verify_training(config_path, snapshot_path, backend, root, output, *, device="cpu", module_path=None,
                    updates=10, calibration_path=None, tolerance=None, source_workers=1):
    if source_workers <= 0:
        raise ValueError("source-workers must be positive")
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
        replay.verify_source(workers=source_workers)
        if calibration and replay.identity != calibration["backend"]:
            raise ValueError("calibration backend artifacts differ")
        actual, expected = {}, {}
        batches = microbatches(checked_trace)
        for rows in batches:
            index = rows[0]["microbatch_id"]
            actual_raw = replay.raw(rows)
            expected_raw = [tuple(v.to(replay.device) for v in cpu_raw(replay.records[r["image_id"]], r, profile, replay.dops))
                            for r in rows]
            a = finish_batch(actual_raw, rows, profile, replay.dops)
            b = finish_batch(expected_raw, rows, profile, replay.dops, oracle=True)
            try:
                comparisons.compare("inputs", a[0], b[0])
            except AssertionError as error:
                details = input_failure_details(replay, rows, actual_raw, expected_raw)
                raise AssertionError(f"{error}\nInput failure diagnosis: {json.dumps(details, sort_keys=True)}") from error
            torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
            actual[index], expected[index] = a, b
            print(f"Input microbatch checked: {len(actual)}/{len(batches)}", flush=True)
        model = build_training_model(profile, device)
        results = paired_updates(model, actual, expected, windows, state, config["training"], comparisons)
        evidence = dict(status="passed", phase="verification" if calibration else "calibration",
                        experiment_type="training_numerical_fixture_shared_partial_processing",
                        resolved_contract=artifact(config_path), snapshot=snapshot_ref,
                        initial_model_sha256=state["initial_model_sha256"], runtime=runtime, implementation=implementation,
                        calibration_trace=config["calibration_trace"], backend=replay.identity,
                        source_q_and_tables_exact=True, target_tensors_exact=True,
                        source_verification_workers=source_workers if backend == "fastlens" else 1,
                        max_abs_errors=comparisons.maximum, tolerance=tolerance, updates=updates,
                        occurrence_count=len(checked_trace), unique_image_count=len({r["image_id"] for r in checked_trace}),
                        training=results, headline_eligible=False)
        if calibration_path:
            evidence["calibration"] = artifact(calibration_path)
    write_json(output, evidence)
    return evidence
