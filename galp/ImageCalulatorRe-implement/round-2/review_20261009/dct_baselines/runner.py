"""Correctness gates and bounded closed-window CPU/GPU inference replay."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from .contracts import ReadRequest, artifact, load_artifact, load_trace, write_json
from .materialize import Materializer, build_model, source_reference
from .source import read_jpeg, source_images
from .store import StoreReader, Counters, storage_bytes

SCHEMA = Path(__file__).parents[2] / "ImageCalulatorRe-implement/fastlens_agent_pack/contracts/result.schema.json"


def implementation_artifacts():
    return [artifact(path) for path in sorted(Path(__file__).parent.glob("*.py"))]


def inputs(config_path):
    config = json.loads(Path(config_path).read_text())
    source = load_artifact(config["source_manifest"])
    profile = load_artifact(config["model_profile"])
    if config["version"] != 1 or source["version"] != 1 or source["representation"] != "original_jpeg_quantized_dct":
        raise ValueError("unsupported resolved/source contract")
    if any(not math.isfinite(config["tolerance"][key]) or config["tolerance"][key] < 0 for key in ("atol", "rtol")):
        raise ValueError("tolerances must be finite and nonnegative")
    for ref in (profile["checkpoint"], profile["resize_implementation"], profile["model_adapter"], profile["dependency_evidence"],
                *profile.get("reference_dependencies", []), *profile.get("training_dependencies", [])):
        if artifact(ref["path"]) != ref:
            raise ValueError(f"profile dependency changed: {ref['path']}")
    if artifact(config["request_trace"]["path"]) != config["request_trace"]:
        raise ValueError("request trace changed")
    records = {record["image_id"]: record for record in source["images"]}
    trace = load_trace(config["request_trace"]["path"], profile, records)
    if config["precision"] not in ("fp32", "bf16-autocast") or config["tail_policy"] != "keep":
        raise ValueError("unsupported precision/tail policy")
    if min(config["model_microbatch_size"], config["preparation_pool_size"], config["workers"], config["torch_threads"]) <= 0:
        raise ValueError("batch/pool/worker/thread counts must be positive")
    if config["preparation_pool_size"] % config["model_microbatch_size"]:
        raise ValueError("preparation pools must contain whole model microbatches")
    if profile.get("training"):
        from .unified import microbatches
        previous = None
        batches = microbatches(trace)
        for index, rows in enumerate(batches):
            if rows[0]["microbatch_id"] != index or len(rows) > config["model_microbatch_size"]:
                raise ValueError("training microbatch assignment differs from the frozen model batch")
            if previous and len(previous) < config["model_microbatch_size"] and previous[0]["epoch"] == rows[0]["epoch"]:
                raise ValueError("a partial microbatch is only allowed at the epoch tail")
            previous = rows
        if config["precision"] != config["training"]["precision"]:
            raise ValueError("training precision differs from the frozen recipe")
    else:
        for slot, row in enumerate(trace):
            if row["microbatch_id"] != slot//config["model_microbatch_size"]:
                raise ValueError("trace microbatch assignment differs from the frozen model batch")
    return config, source, profile, records, trace


def request(row, profile):
    return ReadRequest(row["image_id"], tuple(tuple(f) for f in profile["source_frequencies"]))


def assert_tensors(actual, expected, atol, rtol):
    if len(actual) != len(expected):
        raise AssertionError("model tensor counts differ")
    error = 0.
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        error = max(error, float((a-b).abs().max()))
    return error


def check_updates(model, actual_batches, expected_batches, labels, optimizer_factory, steps, atol, rtol):
    """Same optimizer factory and RNG for both complete backward/update replays."""
    from galp.benchmarks.system_rgbnomore.training.model_factory import capture_rng_state, restore_rng_state
    rng = capture_rng_state()
    actual_model, expected_model = copy.deepcopy(model), copy.deepcopy(model)
    actual_model.requires_grad_(True).train()
    expected_model.requires_grad_(True).train()
    actual_optimizer = optimizer_factory(actual_model.parameters())
    expected_optimizer = optimizer_factory(expected_model.parameters())
    try:
        for step in range(steps):
            index = step % len(actual_batches)
            step_rng = capture_rng_state()
            snapshots = []
            for net, optimizer, batch in ((expected_model, expected_optimizer, expected_batches[index]),
                                          (actual_model, actual_optimizer, actual_batches[index])):
                restore_rng_state(step_rng)
                optimizer.zero_grad(set_to_none=True)
                logits = net(*batch)
                loss = torch.nn.functional.cross_entropy(logits, labels[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite semantic training check")
                loss.backward()
                gradients = [p.grad.detach().clone() for p in net.parameters() if p.grad is not None]
                optimizer.step()
                snapshots.append((loss.detach(), gradients))
            assert_tensors((snapshots[0][0], *snapshots[0][1]), (snapshots[1][0], *snapshots[1][1]), atol, rtol)
            assert_tensors(tuple(actual_model.state_dict().values()), tuple(expected_model.state_dict().values()), atol, rtol)
    finally:
        restore_rng_state(rng)
    return dict(steps=steps, loss_gradients_and_parameters_match=True,
                optimizer="caller-supplied factory; CLI uses SGD(lr=.01,momentum=.9) semantic smoke, not a published training recipe")


def verify(config_path, root, output, *, device="cpu", check_model=False, training_steps=0):
    config, source, profile, records, trace = inputs(config_path)
    if profile.get("training"):
        raise ValueError("use calibrate/verify-training for frozen published-recipe training")
    torch.set_num_threads(config["torch_threads"])
    atol, rtol = config["tolerance"]["atol"], config["tolerance"]["rtol"]
    materialize = Materializer(profile)
    tensor_error = 0.
    with StoreReader(root, workers=config["workers"]) as reader:
        if reader.manifest["source_artifact"] != config["source_manifest"]:
            raise ValueError("store was built from a different source manifest")
        # All stored source integers and all Q tables, independently of the trace.
        for image in source_images(source):
            record = reader.images[image.metadata["image_id"]]
            masks = tuple(tuple(mask) for mask in record["masks"])
            views, _ = reader.read(reader.plan([ReadRequest(image.metadata["image_id"], masks)]))
            for actual, expected, mask, table, expected_table in zip(
                views[0].coefficients, image.coefficients, masks, views[0].tables, image.tables
            ):
                np.testing.assert_array_equal(actual, expected.reshape(*expected.shape[:2], 64)[..., list(mask)])
                np.testing.assert_array_equal(table, expected_table)
        model = build_model(profile, torch.device(device)) if check_model or training_steps else None
        actual_batches, expected_batches, batch_labels = [], [], []
        batch = config["model_microbatch_size"]
        for offset in range(0, len(trace), batch):
            rows = trace[offset:offset+batch]
            views, _ = reader.read(reader.plan([request(row, profile) for row in rows]))
            actual, expected = [], []
            for row, view in zip(rows, views):
                record = records[row["image_id"]]
                image = read_jpeg(record["path"], record["image_id"], record["label"], record["split"])
                a = materialize(view.dense(), view.tables, row)
                b = source_reference(profile, image, row)
                tensor_error = max(tensor_error, assert_tensors(a, b, atol, rtol))
                actual.append(a)
                expected.append(b)
            if model is not None:
                a = tuple(torch.stack([item[i] for item in actual]).to(device) for i in range(len(actual[0])))
                b = tuple(torch.stack([item[i] for item in expected]).to(device) for i in range(len(expected[0])))
                with torch.inference_mode():
                    assert_tensors((model(*a),), (model(*b),), atol, rtol)
                # A bounded set suffices for the short update semantic check.
                if training_steps and len(actual_batches) < training_steps:
                    actual_batches.append(a)
                    expected_batches.append(b)
                    batch_labels.append(torch.tensor([row["target_label"] for row in rows], device=device))
        training = None
        if training_steps:
            training = check_updates(model, actual_batches, expected_batches, batch_labels,
                                     lambda params: torch.optim.SGD(params, lr=.01, momentum=.9), training_steps, atol, rtol)
        evidence = dict(status="passed", resolved_contract=artifact(config_path), store=artifact(Path(root) / "store.json"),
                        implementation=implementation_artifacts(), device=device,
                        source_q_exact=True, quantization_tables_exact=True, dependency_closure_verified=True,
                        final_tensor_match=True, logits_match=model is not None, training_update_match=training is not None,
                        max_abs_tensor_error=tensor_error, max_relative_tensor_error=0. if tensor_error == 0 else None,
                        source_image_count=len(records), occurrence_count=len(trace), training=training,
                        counters=asdict(reader.counters), tolerance=config["tolerance"])
    write_json(output, evidence)
    return evidence


class Replay:
    def __init__(self, reader, profile, config):
        self.reader, self.profile, self.config = reader, profile, config
        self.materialize = Materializer(profile)

    def pool(self, rows):
        # Physical planning and gathering occur inside this timed function.
        views, counters = self.reader.read(self.reader.plan([request(row, self.profile) for row in rows]))
        def transform(pair):
            row, view = pair
            return self.materialize(view.dense(), view.tables, row)
        executor = self.reader.executor
        transformed = list(map(transform, zip(rows, views)) if executor is None else executor.map(transform, zip(rows, views)))
        batches = []
        size = self.config["model_microbatch_size"]
        for start in range(0, len(rows), size):
            values = transformed[start:start+size]
            tensors = tuple(torch.stack([v[i] for v in values]) for i in range(len(values[0])))
            labels = torch.tensor([r["target_label"] for r in rows[start:start+size]])
            batches.append((tensors, labels))
        return batches, counters


def benchmark(config_path, root, evidence_path, *, device="cpu", run_kind="input_only", cache_bytes=0):
    config, source, profile, records, trace = inputs(config_path)
    if profile.get("training"):
        raise ValueError("training correctness contracts use verify-training; inference timing requires an inference contract")
    evidence = json.loads(Path(evidence_path).read_text())
    if evidence["status"] != "passed" or evidence["resolved_contract"] != artifact(config_path) or (
        evidence["store"] != artifact(Path(root) / "store.json") or evidence["implementation"] != implementation_artifacts()
    ):
        raise ValueError("correctness evidence is stale or belongs to another contract/store")
    if evidence["device"] != device or (run_kind == "inference" and not evidence["logits_match"]):
        raise ValueError("verify the same device and inference logits before timing")
    if run_kind not in ("input_only", "inference"):
        raise ValueError("formal training replay is not integrated")
    torch.set_num_threads(config["torch_threads"])
    target_device = torch.device(device)
    model = build_model(profile, target_device) if run_kind == "inference" else None
    if target_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    count = top1 = top5 = h2d = h2d_coefficients = pinned_peak = 0
    predictions, pinned_pending = [], []
    outstanding_pools = []
    first_event = None
    first_latency = None
    cpu_started = time.process_time()
    with StoreReader(root, workers=config["workers"], cache_bytes=cache_bytes) as reader, ThreadPoolExecutor(max_workers=1) as prefetch:
        replay = Replay(reader, profile, config)
        counters = Counters()
        pool_size = config["preparation_pool_size"]
        pools = iter(trace[offset:offset+pool_size] for offset in range(0, len(trace), pool_size))
        started = time.perf_counter()
        first = next(pools)
        future = prefetch.submit(replay.pool, first)
        with torch.inference_mode():
            while future is not None:
                batches, observed = future.result()
                counters.add(observed)
                following = next(pools, None)
                future = prefetch.submit(replay.pool, following) if following is not None else None
                if target_device.type == "cuda":
                    outstanding_pools = [event for event in outstanding_pools if not event.query()]
                    # Limit GPU/pinned staging to two submitted preparation
                    # pools without synchronizing each model microbatch.
                    if len(outstanding_pools) == 2:
                        outstanding_pools.pop(0).synchronize()
                for tensors, labels in batches:
                    n = labels.numel()
                    if target_device.type == "cuda":
                        # Reclaim pinned buffers only after their asynchronous copy.
                        pinned_pending = [(event, size) for event, size in pinned_pending if not event.query()]
                        pinned = tuple(t.pin_memory() for t in (*tensors, labels))
                        size = sum(t.numel()*t.element_size() for t in pinned)
                        h2d += size
                        h2d_coefficients += sum(t.numel()*t.element_size() for t in pinned[:-1])
                        values = tuple(t.to(target_device, non_blocking=True) for t in pinned)
                        tensors, labels = values[:-1], values[-1]
                        event = torch.cuda.Event()
                        event.record()
                        pinned_pending.append((event, size))
                        pinned_peak = max(pinned_peak, sum(size for _, size in pinned_pending))
                    if model is not None:
                        logits = model(*tensors)
                        rank = logits.topk(5, dim=1).indices
                        predictions.append((torch.isfinite(logits).all(), (rank[:, 0] == labels).sum(),
                                            (rank == labels[:, None]).any(1).sum()))
                    count += n
                    if first_event is None and first_latency is None:
                        if target_device.type == "cuda":
                            first_event = torch.cuda.Event(enable_timing=True)
                            first_event.record()
                        else:
                            first_latency = time.perf_counter()-started
                if target_device.type == "cuda":
                    pool_done = torch.cuda.Event()
                    pool_done.record()
                    outstanding_pools.append(pool_done)
            # Metric collection and the last CUDA completion are in the window.
            if predictions:
                values = torch.stack([torch.stack((finite.long(), a, b)) for finite, a, b in predictions]).cpu()
                if not bool(values[:, 0].all()):
                    raise FloatingPointError("non-finite benchmark logits")
                top1, top5 = int(values[:, 1].sum()), int(values[:, 2].sum())
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
        if count != len(trace):
            raise AssertionError("trace occurrence coverage mismatch")
        store = reader.manifest
        init = asdict(reader.initialization_counters)
    result = result_record(config_path, config, profile, root, evidence_path, evidence, target_device, store,
                           count, len({r["image_id"] for r in trace}), elapsed, first_latency, counters,
                           h2d, h2d_coefficients, pinned_peak, run_kind, top1, top5, cache_bytes)
    # Keep detailed accounting separate from the draft schema's bounded fields.
    detail = dict(counters=asdict(counters), initialization_io=init, cpu_core_seconds=time.process_time()-cpu_started,
                  decode_amplification_denominator="sum of per-preparation-pool deduplicated required q values",
                  decode_amplification=(counters.decoded_q_values/counters.unique_requested_q_values
                                        if counters.unique_requested_q_values else None))
    return result, detail


def result_record(config_path, config, profile, root, evidence_path, evidence, device, store,
                  count, unique, elapsed, first_latency, counters, h2d, h2d_coefficients, pinned_peak, run_kind, top1, top5, cache_bytes):
    sizes = storage_bytes(root)
    correctness = {key: evidence[key] for key in ("status", "source_q_exact", "quantization_tables_exact",
                   "dependency_closure_verified", "final_tensor_match", "logits_match", "training_update_match",
                   "max_abs_tensor_error", "max_relative_tensor_error")}
    quality = dict(status="measured" if run_kind == "inference" else "not_measured", split="trace",
                   sample_count=count if run_kind == "inference" else None,
                   top1_fraction=top1/count if run_kind == "inference" else None,
                   top5_fraction=top5/count if run_kind == "inference" else None,
                   allowed_loss_percentage_points=None, observed_loss_percentage_points=None)
    return dict(schema_version="0.1-draft", run_id=f"{store['method']}-{time.time_ns()}", status="completed",
                method=store["method"], variant=f"adapted_cpu_transform_{store['dtype_policy']}_decoded_cache_{cache_bytes}",
                comparison_scope="source_representation_matched", run_kind=run_kind, measurement_mode="actual",
                cache_mode="decoded_host" if cache_bytes else "unverified", headline_eligible=False,
                artifacts=dict(resolved_contract=artifact(config_path), source_manifest=config["source_manifest"],
                               model_profile=config["model_profile"], request_trace=config["request_trace"],
                               checkpoint=profile["checkpoint"], correctness_evidence=artifact(evidence_path), cache_evidence=None),
                environment=dict(repository_revision=config["repository_revision"],
                                 hardware=dict(cpu=platform.processor(), device=str(device),
                                               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                                 software=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)),
                resolved_parameters=dict(model_microbatch_size=config["model_microbatch_size"],
                                         preparation_pool_size=config["preparation_pool_size"], storage_group_images=store["storage_group_images"],
                                         workers=config["workers"], prefetch=1, precision="fp32",
                                         source_frequencies={str(c): v for c, v in enumerate(profile["source_frequencies"])},
                                         output_channels={"tensors": profile["tensors"], "channels": profile["output_channels"]}),
                counts=dict(sample_occurrences=count, unique_images=unique, optimizer_updates=0,
                            requested_q_values=counters.requested_q_values, physically_decoded_q_values=counters.decoded_q_values),
                metrics=dict(elapsed_wall_s=elapsed, throughput_images_s=count/elapsed, estimated_elapsed_s=None,
                             estimated_throughput_images_s=None, first_batch_latency_s=first_latency,
                             batch_latency_s=dict(p50=None, p95=None, p99=None),
                             io=dict(compressed_payload_read_bytes=counters.compressed_payload_read_bytes,
                                     metadata_read_bytes=counters.metadata_read_bytes, application_read_bytes=counters.application_read_bytes,
                                     physical_storage_read_bytes=None, read_operations=counters.read_calls, file_open_operations=counters.file_opens),
                             h2d=dict(total_bytes=h2d, coefficient_payload_bytes=h2d_coefficients,
                                      metadata_indices_labels_padding_bytes=h2d-h2d_coefficients),
                             storage=dict(**{k: v for k, v in sizes.items() if k != "file_count"}, temporary_peak_bytes=None),
                             memory=dict(host_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                         pinned_peak_bytes=pinned_peak, gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                                         gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None)),
                correctness=correctness, quality=quality,
                offline_cost=dict(included_in_elapsed_wall=False, conversion_wall_s=store["conversion_wall_s"],
                                  cpu_core_hours=store["conversion_cpu_s"]/3600, gpu_hours=0, search_trials=0,
                                  profile_copies=1), synthetic_delivery=None,
                notes=["Diagnostic replay; G4 calibration, cache evidence and full G5 experiments are outstanding.",
                       "Application bytes include complete NPZ archives; physical disk bytes are unmeasured.",
                       "CPU transform adapter transfers final float32 tensors and labels, not compact coefficients.",
                       "H2D coefficient payload counts final float32 DCT model inputs; the other transfer is int64 labels.",
                       "Host RSS is process high-water mark, including initialization; threads share the same process.",
                       "Timing includes first preparation pool and final completion. No excluded warmup is performed."])


def report(paths, output):
    import statistics
    import jsonschema
    schema = json.loads(SCHEMA.read_text())
    groups = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            jsonschema.Draft202012Validator(schema).validate(row)
            if row["status"] != "completed":
                continue
            # Different checkpoints/traces, cache/runtime controls and variants
            # cannot silently share a summary row.
            key = json.dumps({k: row[k] for k in ("method", "variant", "run_kind", "cache_mode", "resolved_parameters",
                                                 "measurement_mode", "comparison_scope", "environment")}, sort_keys=True)
            key += json.dumps({k: row["artifacts"][k] for k in ("resolved_contract", "checkpoint", "request_trace")}, sort_keys=True)
            groups.setdefault(key, []).append(row)
    rows = []
    for group in groups.values():
        mode = group[0]["measurement_mode"]
        raw = [r["metrics"]["estimated_throughput_images_s" if mode == "modeled" else "throughput_images_s"] for r in group]
        rows.append(dict(method=group[0]["method"], variant=group[0]["variant"], run_kind=group[0]["run_kind"],
                         measurement_mode=mode, cache_mode=group[0]["cache_mode"],
                         headline_eligible=all(r["headline_eligible"] for r in group), repeats=len(raw),
                         median_images_s=statistics.median(raw), min_images_s=min(raw), max_images_s=max(raw), raw_images_s=raw))
    write_json(output, rows)
    return rows
