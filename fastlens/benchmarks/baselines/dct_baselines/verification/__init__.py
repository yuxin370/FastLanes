"""Reusable inference checks; the measured pipeline and its identity stay unchanged."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import torch

from ..contracts import artifact, write_json
from ..gates import numerical_runtime
from ..materialize import build_model, source_reference
from ..runner import assert_tensors, implementation_artifacts, inputs
from ..source import source_images, verify_source_files
from ..unified import UnifiedReplay
from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_identities(config, profile, trace, runtime, implementation, verifier):
    # These fields describe model construction, not preprocessing. Everything
    # else remains in the input key, including normalization and dependencies.
    preprocessing = {key: value for key, value in profile.items()
                     if key not in ("id", "model", "model_id", "checkpoint", "repository_profile")}
    requests = [{key: value for key, value in row.items() if key != "profile_id"} for row in trace]
    input_identity = dict(source=config["source_manifest"], preprocessing=preprocessing,
        trace_sha256=digest(requests), batch_size=config["model_microbatch_size"],
        precision=config["precision"], torch_threads=config["torch_threads"], runtime=runtime,
        implementation=implementation, verifier=verifier)
    model_identity = dict(input_identity_sha256=digest(input_identity), profile=profile, runtime=runtime,
                          implementation=implementation, verifier=verifier)
    return input_identity, model_identity


def batch_signature(rows, tensors):
    return dict(occurrences=[row["occurrence_id"] for row in rows],
                images=[row["image_id"] for row in rows], labels=[row["target_label"] for row in rows],
                tensors=[dict(shape=list(t.shape), dtype=str(t.dtype),
                              sha256=hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                         for t in tensors])


def load_reference(path, identity, field):
    if not path.exists():
        return None
    cache = json.loads(path.read_text())
    # The completed gate binds the cache bytes. An interrupted writer, stale
    # reference, or modified fingerprint must never become a passed reference.
    evidence = json.loads(Path(cache["evidence_path"]).read_text())
    if (cache["identity"] != identity or evidence["status"] != "passed"
            or evidence["reference_cache"][field] != artifact(path)
            or evidence["occurrence_count"] != cache["occurrence_count"]):
        raise ValueError("reference cache is incomplete, changed, or belongs to another workload")
    return cache


def model_reference_from_evidence(path, config, profile, trace, runtime, implementation):
    prior = json.loads(Path(path).read_text())
    if artifact(prior["resolved_contract"]["path"]) != prior["resolved_contract"]:
        raise ValueError("model evidence contract changed")
    previous, _, previous_profile, _, previous_trace = inputs(prior["resolved_contract"]["path"])
    if (prior["status"] != "passed" or prior["phase"] != "pretrained_inference"
            or not prior["final_tensor_match"] or not prior["logits_match"] or not prior["predictions_match"]
            or prior["implementation"] != implementation or prior["runtime"] != runtime
            or prior["occurrence_count"] != len(trace) or previous_profile != profile or previous_trace != trace
            or any(previous[key] != config[key] for key in
                   ("source_manifest", "model_microbatch_size", "precision", "torch_threads"))
            or any(artifact(ref["path"]) != ref for ref in prior["model_implementation"])):
        raise ValueError("model evidence differs from current workload/checkpoint/runtime")
    return dict(model_implementation=prior["model_implementation"],
                loaded_model_state_sha256=prior["loaded_model_state_sha256"],
                metrics=dict(top1_fraction=prior["metrics"]["top1_fraction"],
                             top5_fraction=prior["metrics"]["top5_fraction"],
                             cross_entropy=prior["metrics"]["reference_cross_entropy"]))


def verify_inference(config_path, backend, root, output, *, cache_dir, device="cpu", source_evidence=None,
                     model_evidence=None):
    """Check each actual input once; share source audits, input oracles and model results."""
    if backend not in ("cpu_dct", "ic", "fs"):
        raise ValueError("reusable CPU-baseline verification supports cpu_dct, ic and fs")
    output, cache_dir = Path(output), Path(cache_dir)
    if output.exists():
        raise FileExistsError(output)
    config, source, profile, records, trace = inputs(config_path)
    if profile.get("training"):
        raise ValueError("reference reuse is for deterministic pretrained inference")
    runtime = numerical_runtime(device)
    torch.set_num_threads(config["torch_threads"])
    implementation = implementation_artifacts()
    verifier = [artifact(path) for path in sorted(Path(__file__).parent.glob("*.py"))]
    input_identity, model_identity = reference_identities(config, profile, trace, runtime, implementation, verifier)
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_path = cache_dir / ("inputs-"+digest(input_identity)+".json")
    model_path = cache_dir / ("model-"+digest(model_identity)+".json")
    input_cache = load_reference(input_path, input_identity, "inputs")
    model_cache = load_reference(model_path, model_identity, "model")
    if model_cache and not input_cache:
        raise ValueError("model reference has lost its verified input reference")
    model_evidence_reused = None
    if model_cache is None and model_evidence is not None:
        model_cache = model_reference_from_evidence(model_evidence, config, profile, trace, runtime, implementation)
        model_evidence_reused = artifact(model_evidence)
    if model_cache:
        model_code = model_cache["model_implementation"]
        if any(artifact(ref["path"]) != ref for ref in model_code):
            raise ValueError("cached model implementation changed")
        state_hash = model_cache["loaded_model_state_sha256"]
    model = None

    def get_model():
        nonlocal model, model_code, state_hash
        if model is None:
            model = build_model(profile, torch.device(device)).eval().requires_grad_(False)
            current_state = tensor_state_sha256(model.state_dict())
            current_code = [artifact(path) for path in sorted({inspect.getfile(type(m)) for m in model.modules()})]
            if model_cache and (current_state != state_hash or current_code != model_code):
                raise ValueError("loaded model differs from the cached model reference")
            state_hash, model_code = current_state, current_code
        return model

    checked = reused_inputs = reused_models = oracle_batches = model_forwards = 0
    input_error = logit_error = loss_delta = 0.
    top1 = top5 = 0
    reference_loss = 0.
    signatures = []
    atol, rtol = config["tolerance"]["atol"], config["tolerance"]["rtol"]
    started = time.perf_counter()
    with UnifiedReplay(backend, root, config, source, profile, device=device) as replay:
        source_identity = dict(source_manifest=config["source_manifest"], backend=replay.identity,
                               implementation=implementation)
        source_path = cache_dir / ("source-"+digest(source_identity)+".json")
        prior_path = Path(source_evidence) if source_evidence is not None else source_path
        source_reused = prior_path.exists()
        if source_reused:
            prior = json.loads(prior_path.read_text())
            if (prior["status"] != "passed" or not prior["source_q_exact"] or not prior["quantization_tables_exact"]
                    or any(prior[key] != value for key, value in source_identity.items())):
                raise ValueError("source evidence differs from current source/store/implementation")
            verify_source_files(source)
            print("Reusing passed source q/Q check; no coefficient comparison.", flush=True)
        else:
            if source_evidence is not None:
                raise FileNotFoundError(prior_path)
            replay.verify_source()
        # Save this independently of the model check so a later input/model
        # failure does not discard a successful, model-independent source audit.
        if not source_path.exists():
            write_json(source_path, dict(source_identity, status="passed", source_q_exact=True,
                                         quantization_tables_exact=True))
        preflight_s = time.perf_counter()-started
        with torch.inference_mode():
            for index, (rows, actual) in enumerate(replay.inference_batches(trace)):
                n = len(rows)
                if rows != trace[checked:checked+n]:
                    raise AssertionError("inference occurrence order or labels changed")
                signature = batch_signature(rows, actual)
                saved = input_cache["batches"][index] if input_cache else None
                exact = saved is not None and signature == saved
                if exact:
                    reused_inputs += 1
                    expected = actual
                else:
                    values = [source_reference(profile, image, row) for image, row in
                              zip(source_images(dict(images=[records[r["image_id"]] for r in rows])), rows)]
                    expected = tuple(torch.stack([v[i] for v in values]).to(device) for i in range(len(values[0])))
                    oracle_batches += 1
                    expected_signature = batch_signature(rows, expected)
                    if saved is not None and expected_signature != saved:
                        raise ValueError("independent input reference differs from its saved fingerprint")
                    input_error = max(input_error, assert_tensors(actual, expected, atol, rtol))
                    exact = signature == expected_signature
                    if input_cache is None:
                        signatures.append(expected_signature)
                if exact and model_cache:
                    # Bit-identical inputs, labels, batches, checkpoint and
                    # inference implementation imply the cached model result.
                    reused_models += 1
                else:
                    network = get_model()
                    reference = network(*expected)
                    model_forwards += 1
                    logits = reference if exact else network(*actual)
                    model_forwards += int(not exact)
                    logit_error = max(logit_error, assert_tensors((logits,), (reference,), atol, rtol))
                    rank, reference_rank = logits.topk(5, dim=1).indices, reference.topk(5, dim=1).indices
                    torch.testing.assert_close(rank[:, 0], reference_rank[:, 0], atol=0, rtol=0)
                    torch.testing.assert_close(rank.sort(1).values, reference_rank.sort(1).values, atol=0, rtol=0)
                    labels = torch.tensor([r["target_label"] for r in rows], device=device)
                    actual_loss = torch.nn.functional.cross_entropy(logits, labels)
                    expected_loss = torch.nn.functional.cross_entropy(reference, labels)
                    assert_tensors((actual_loss,), (expected_loss,), atol, rtol)
                    loss_delta += float(actual_loss-expected_loss)*n
                    if model_cache is None:
                        top1 += int((rank[:, 0] == labels).sum())
                        top5 += int((rank == labels[:, None]).any(1).sum())
                        reference_loss += float(expected_loss)*n
                previous = checked
                checked += n
                if checked//1024 != previous//1024 or checked == len(trace):
                    print(f"Inference samples checked: {checked}/{len(trace)}; "
                          f"reused input batches={reused_inputs}, model batches={reused_models}", flush=True)
        if checked != len(trace) or (input_cache and len(input_cache["batches"]) != index+1):
            raise AssertionError("inference lost occurrences or changed batch boundaries")
        if model is not None and tensor_state_sha256(model.state_dict()) != state_hash:
            raise AssertionError("inference changed model state")
        if model_cache:
            reference_metrics = model_cache["metrics"]
        else:
            reference_metrics = dict(top1_fraction=top1/checked, top5_fraction=top5/checked,
                                     cross_entropy=reference_loss/checked)
        if input_cache is None:
            write_json(input_path, dict(identity=input_identity, evidence_path=str(output.resolve()),
                                       occurrence_count=checked, batches=signatures))
        if not model_path.exists():
            write_json(model_path, dict(identity=model_identity, evidence_path=str(output.resolve()),
                occurrence_count=checked, metrics=reference_metrics, model_implementation=model_code,
                loaded_model_state_sha256=state_hash))
        evidence = dict(status="passed", phase="pretrained_inference", experiment_type="matched_system_inference",
            execution_partition=dict(read_decode="cpu", geometry_normalization="cpu", augmentation="disabled", model=str(device)),
            resolved_contract=artifact(config_path), source_manifest=config["source_manifest"],
            source_evidence_reused=artifact(prior_path) if source_reused else None,
            model_evidence_reused=model_evidence_reused,
            checkpoint=profile["checkpoint"], loaded_model_state_sha256=state_hash, model_implementation=model_code,
            implementation=implementation, verification_implementation=verifier, backend=replay.identity, runtime=runtime,
            source_q_exact=True, quantization_tables_exact=True, dependency_closure_verified=True,
            final_tensor_match=True, logits_match=True, predictions_match=True, training_update_match=False,
            prediction_comparison="exact inputs with shared deterministic model; differing batches compared directly",
            verification_method="reused_input_fingerprints_and_model_results",
            reuse=dict(input_batches=reused_inputs, model_batches=reused_models,
                       independent_reference_batches=oracle_batches, model_forward_calls=model_forwards),
            reference_cache=dict(inputs=artifact(input_path), model=artifact(model_path)),
            max_abs_tensor_error=input_error, max_relative_tensor_error=0. if input_error == 0 else None,
            max_abs_logits_error=logit_error, tolerance=config["tolerance"], device=str(device),
            source_image_count=len(records), occurrence_count=checked,
            metrics=dict(reference_metrics, cross_entropy=reference_metrics["cross_entropy"]+loss_delta/checked,
                         reference_cross_entropy=reference_metrics["cross_entropy"]),
            source_preflight_wall_s=preflight_s, cache_state="unmeasured/post-verification", headline_eligible=False)
        if replay.reader:
            evidence["store"] = replay.reader.identity
    write_json(output, evidence)
    return evidence
