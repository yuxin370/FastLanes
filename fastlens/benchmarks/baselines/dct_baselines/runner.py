"""Correctness gates and bounded closed-window CPU/GPU inference replay."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
import copy
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from .contracts import ReadRequest, artifact, load_artifact, load_trace, request_for_row, validate_batches, write_json
from .materialize import Materializer, build_model, operation_order, source_reference
from .source import read_jpeg, source_images, verify_source_files
from .store import StoreReader, Counters, fingerprint, storage_bytes

SCHEMA = Path(__file__).with_name("result.schema.json")


def implementation_artifacts():
    return [artifact(path) for path in sorted(Path(__file__).parent.glob("*.py"))]


def inputs(config_path):
    config = json.loads(Path(config_path).read_text())
    source = load_artifact(config["source_manifest"])
    profile = load_artifact(config["model_profile"])
    if profile["operation_order"] != operation_order(profile):
        raise ValueError("profile operation order differs from the executed inference/training pipeline")
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
    labels = load_artifact(config["label_mapping"])
    if labels["policy"] != profile["label_policy"] or set(labels["labels"]) != set(records):
        raise ValueError("label mapping does not cover the frozen source/profile")
    if labels["policy"] == "sorted_class_names" and labels["classes"] != {name: i for i, name in enumerate(sorted(labels["classes"]))}:
        raise ValueError("checkpoint class mapping must use sorted class names")
    for image_id, label in labels["labels"].items():
        expected = labels["classes"][image_id.split("/")[1]] if labels["policy"] == "sorted_class_names" else records[image_id]["label"]
        if type(label) is not int or not 0 <= label < 1000 or label != expected:
            raise ValueError("label mapping differs from the declared checkpoint/source policy")
    trace = load_trace(config["request_trace"]["path"], profile, records, labels["labels"])
    if config["precision"] not in ("fp32", "bf16-autocast") or config["tail_policy"] != "keep":
        raise ValueError("unsupported precision/tail policy")
    if min(config["model_microbatch_size"], config["preparation_pool_size"], config["workers"], config["torch_threads"]) <= 0:
        raise ValueError("batch/pool/worker/thread counts must be positive")
    if config.get("cpu_loader") == "dataloader" and config["prefetch_factor"] <= 0:
        raise ValueError("DataLoader prefetch factor must be positive")
    if config["preparation_pool_size"] % config["model_microbatch_size"]:
        raise ValueError("preparation pools must contain whole model microbatches")
    if profile.get("training"):
        from .training import updates_per_epoch
        if config["precision"] != config["training"]["precision"]:
            raise ValueError("training precision differs from the frozen recipe")
        recipe = config["training"]
        per_epoch = updates_per_epoch(recipe["population_size"], config["model_microbatch_size"],
                                      config["preparation_pool_size"], recipe["accumulation"])
        if recipe["updates_per_epoch"] != per_epoch or (not recipe["total_updates_override"] and
                recipe["total_updates"] != per_epoch*recipe["schedule_epochs"]):
            raise ValueError("training schedule differs from the pool/tail update rules")
    elif config["precision"] != "fp32":
        raise ValueError("inference replay supports fp32 only")
    validate_batches(trace, config, training=bool(profile.get("training")))
    if min(config["decode_budget_bytes"], config["reader_memory_budget_bytes"], config["host_input_budget_bytes"]) <= 0:
        raise ValueError("reader memory/decode budgets must be positive")
    if not 0 <= config.get("decoded_cache_bytes", 0) < config["reader_memory_budget_bytes"]:
        raise ValueError("decoded cache must fit within the reader budget")
    return config, source, profile, records, trace


def request(row, profile):
    return request_for_row(row, profile)


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
    from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import capture_rng_state, restore_rng_state
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
    with StoreReader(root, workers=config["workers"], decode_budget_bytes=config["decode_budget_bytes"],
                     memory_budget_bytes=config["reader_memory_budget_bytes"]) as reader:
        reader.verify_content()
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
        from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
        loaded_model_state = tensor_state_sha256(model.state_dict()) if model is not None else None
        actual_batches, expected_batches, batch_labels = [], [], []
        batch = config["model_microbatch_size"]
        for offset in range(0, len(trace), batch):
            rows = trace[offset:offset+batch]
            views, _ = reader.read(reader.plan([request(row, profile) for row in rows]))
            actual, expected = [], []
            for row, view in zip(rows, views):
                record = records[row["image_id"]]
                image = read_jpeg(record["path"], record["image_id"], record["label"], record["split"])
                a = materialize(materialize.source_coefficients(view), view.tables, row, cropped=True, source_selected=True)
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
        evidence = dict(status="passed", resolved_contract=artifact(config_path), store=reader.identity,
                        experiment_type="matched_system_inference", checkpoint=profile["checkpoint"],
                        loaded_model_state_sha256=loaded_model_state,
                        implementation=implementation_artifacts(), device=device,
                        source_q_exact=True, quantization_tables_exact=True, dependency_closure_verified=True,
                        final_tensor_match=True, logits_match=model is not None, training_update_match=training is not None,
                        max_abs_tensor_error=tensor_error, max_relative_tensor_error=0. if tensor_error == 0 else None,
                        source_image_count=len(records), occurrence_count=len(trace), training=training,
                        counters=asdict(reader.counters), tolerance=config["tolerance"])
    write_json(output, evidence)
    return evidence


class Replay:
    def __init__(self, reader, profile, config, *, records=None, executor=None):
        self.reader, self.profile, self.config = reader, profile, config
        self.materialize = Materializer(profile)
        self.records, self.executor = records, executor

    def pool(self, rows):
        # Physical planning and gathering occur inside this timed function.
        if self.reader is None:
            import dct_manip
            # Current file hashes are checked before timing. The timed path
            # reads original q/Q without repeating hashes or oracle checks.
            def transform(row):
                _, quant, y, chroma = dct_manip.read_coefficients(self.records[row["image_id"]]["path"])
                q = (y[0].numpy(),) if chroma is None else (y[0].numpy(), chroma[0].numpy(), chroma[1].numpy())
                tables = tuple(t.numpy().view(np.uint16) for t in quant)
                return self.materialize(q, tables, row)
            transformed = list(map(transform, rows) if self.executor is None else self.executor.map(transform, rows))
            counters = Counters()
        else:
            views, counters = self.reader.read(self.reader.plan([request(row, self.profile) for row in rows]))
            def transform(pair):
                row, view = pair
                return self.materialize(self.materialize.source_coefficients(view), view.tables, row, cropped=True, source_selected=True)
            executor = self.reader.executor
            transform_started = time.perf_counter() if self.config.get("profile_stages") else 0.
            transformed = list(map(transform, zip(rows, views)) if executor is None else executor.map(transform, zip(rows, views)))
            if self.config.get("profile_stages"):
                counters.materialize_wall_s = time.perf_counter()-transform_started
        batches = []
        size = self.config["model_microbatch_size"]
        for start in range(0, len(rows), size):
            values = transformed[start:start+size]
            tensors = tuple(torch.stack([v[i] for v in values]) for i in range(len(values[0])))
            labels = torch.tensor([r["target_label"] for r in rows[start:start+size]])
            batches.append((tensors, labels))
        return batches, counters


class InferencePools(torch.utils.data.Dataset):
    """One task reads a preparation pool, with a bounded worker-local decoded cache."""
    def __init__(self, root, profile, config, records, trace, *, reader_only=False):
        self.root, self.profile, self.config = root, profile, config
        self.records, self.trace, self.replay = records, trace, None
        self.reader_only = reader_only

    def __len__(self):
        return math.ceil(len(self.trace)/self.config["preparation_pool_size"])

    def __getitem__(self, index):
        if self.replay is None:
            # Parent preflight verifies immutable source/store content once.
            # DataLoader supplies process parallelism; no nested thread pool.
            reader = (StoreReader(self.root, workers=1, verify_reads=False,
                                 profile_stages=self.config.get("profile_stages", False),
                                 cache_bytes=self.config.get("decoded_cache_bytes", 0),
                                 decode_budget_bytes=self.config["decode_budget_bytes"],
                                 memory_budget_bytes=self.config["reader_memory_budget_bytes"])
                      if self.root is not None else None)
            self.replay = Replay(reader, self.profile, self.config, records=self.records)
        size = self.config["preparation_pool_size"]
        rows = self.trace[index*size:(index+1)*size]
        if self.reader_only:
            reader = self.replay.reader
            views, counters = reader.read(reader.plan([request(row, self.profile) for row in rows]))
            return len(views), counters
        return self.replay.pool(rows)


def inference_loader(root, profile, config, records, trace, *, reader_only=False):
    return torch.utils.data.DataLoader(InferencePools(root, profile, config, records, trace, reader_only=reader_only),
        batch_size=None, num_workers=config["workers"], prefetch_factor=config["prefetch_factor"],
        persistent_workers=True, multiprocessing_context="spawn")


def benchmark_reader(config_path, root, *, repeats=3):
    """Reader-only tuning on a separate trace; no geometry, transfers or model."""
    from fastlens.benchmarks.dct_models.measure_process import ProcessWindow
    if repeats < 1:
        raise ValueError("reader repeats must be positive")
    config, source, profile, records, trace = inputs(config_path)
    if profile.get("training"):
        raise ValueError("reader tuning uses an inference request trace")
    config = dict(config, profile_stages=True)
    torch.set_num_threads(config["torch_threads"])
    multiprocess = config.get("cpu_loader") == "dataloader"
    # A reader task holds gathered q/Q until it returns counts, with no tensor
    # preparation queue. Each worker's reader budget includes that task output.
    bound = config["reader_memory_budget_bytes"]*(config["workers"] if multiprocess else 1)
    if bound > config["host_input_budget_bytes"]:
        raise MemoryError("reader workers exceed the declared host input budget")
    with StoreReader(root, workers=config["workers"], cache_bytes=config.get("decoded_cache_bytes", 0),
                     decode_budget_bytes=config["decode_budget_bytes"],
                     memory_budget_bytes=config["reader_memory_budget_bytes"],
                     profile_stages=True, verify_reads=False) as reader:
        if reader.manifest["source_artifact"] != config["source_manifest"]:
            raise ValueError("store source identity differs from the tuning trace")
        content_check = reader.verify_content()
        source_check = verify_source_files(source)
        loader = inference_loader(root, profile, config, records, trace, reader_only=True) if multiprocess else None
        def pools():
            if loader is not None:
                yield from loader
            else:
                size = config["preparation_pool_size"]
                for start in range(0, len(trace), size):
                    requests = [request(row, profile) for row in trace[start:start+size]]
                    views, counters = reader.read(reader.plan(requests))
                    yield len(views), counters
                    del views
        observations = []
        for repeat in range(-1, repeats):  # one complete excluded warmup pass
            counters, count = Counters(), 0
            with ProcessWindow() as monitor:
                started = time.perf_counter()
                for n, observed in pools():
                    count += n
                    counters.add(observed)
                elapsed = time.perf_counter()-started
            if count != len(trace):
                raise AssertionError("reader tuning lost trace occurrences")
            if repeat >= 0:
                observations.append(dict(repeat=repeat, sample_occurrences=count, elapsed_wall_s=elapsed,
                    images_s=count/elapsed, counters=asdict(counters), resources=monitor.result,
                    mean_active_inflaters=counters.inflate_wall_s/elapsed,
                    decode_amplification=counters.decoded_q_values/counters.unique_requested_q_values))
        return dict(experiment_type="reader_only_tuning", headline_eligible=False, method=reader.manifest["method"],
                    resolved_contract=artifact(config_path), implementation=implementation_artifacts(),
                    store=reader.identity, storage_group_images=reader.manifest["storage_group_images"],
                    config=config, source_verification=source_check, content_verification=content_check,
                    warmup_passes=1, cache_policy="continuous_post_warmup", runs=observations,
                    notes="No model accuracy claim; same DataLoader task/cache policy as inference. Inflate concurrency is summed inflate wall time / window wall time. Select candidates here, confirm with E2E on a separate trace.")


def cold_input_files(backend, root, records, reader):
    if reader is not None:
        return [reader.root/name for name in reader.manifest["files"]]
    if backend == "cpu_dct":
        return [Path(record["path"]) for record in records.values()]
    from fastlens.benchmarks.system_dct_major.common import parse_manifest
    manifest = parse_manifest(Path(root))
    ids = {record["galp_image_id"] for record in records.values()}
    paths = {Path(root).resolve()}
    for shard in manifest["shards"]:
        first = shard["first_global_image_index"]
        if any(first <= image_id < first+shard["image_count"] for image_id in ids):
            paths.update((Path(shard["fls_path"]), Path(shard["metadata_path"])))
    access = Path(os.environ.get("FASTLENS_BLOCK_MAJOR_ACCESS_DIR", Path(root).resolve().parent/"access"))
    paths.update(path for path in access.rglob("*") if path.is_file())
    return sorted(paths)


def evict_input_files(paths):
    """Same per-file eviction as system_dct_major; confirm residency before timing."""
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    started = time.perf_counter()
    for path in paths:
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    observed = cache_pages(paths)
    if observed["resident_pages"]:
        raise RuntimeError(f"input cache eviction incomplete: {observed['resident_pages']}/{observed['pages']} pages resident")
    return dict(observed, wall_s=time.perf_counter()-started, method="POSIX_FADV_DONTNEED + mincore",
                scope="input file pages before each complete dataset pass; model/reader initialization excluded")


def benchmark(config_path, root, evidence_path, *, device="cpu", run_kind="input_only", cache_bytes=None,
              backend=None, module_path=None, source_workers=1, min_seconds=0., cold_cache=False, warmup_seconds=0.,
              observe_cache=False, profile_stages=False):
    if any(not math.isfinite(v) or v < 0 for v in (min_seconds, warmup_seconds)):
        raise ValueError("measurement and warmup durations must be finite and nonnegative")
    config, source, profile, records, trace = inputs(config_path)
    cache_bytes = config.get("decoded_cache_bytes", 0) if cache_bytes is None else cache_bytes
    if cache_bytes < 0 or cache_bytes >= config["reader_memory_budget_bytes"]:
        raise ValueError("decoded cache must fit within the reader budget")
    if cold_cache and cache_bytes:
        raise ValueError("cold input measurements cannot retain a decoded cache")
    config = dict(config, decoded_cache_bytes=cache_bytes, profile_stages=profile_stages)
    if profile.get("training"):
        raise ValueError("training correctness contracts use verify-training; inference timing requires an inference contract")
    evidence = json.loads(Path(evidence_path).read_text())
    if evidence["status"] != "passed" or evidence["resolved_contract"] != artifact(config_path) or evidence["implementation"] != implementation_artifacts():
        raise ValueError("correctness evidence is stale or belongs to another contract/store")
    if evidence["device"] != device or (run_kind == "inference" and not evidence["logits_match"]):
        raise ValueError("verify the same device and inference logits before timing")
    if "runtime" in evidence:
        from .gates import numerical_runtime
        if evidence["runtime"] != numerical_runtime(device):
            raise ValueError("inference runtime differs from its numerical evidence")
    if run_kind not in ("input_only", "inference"):
        raise ValueError("formal training replay is not integrated")
    if backend not in (None, "cpu_dct", "fastlens", "ic", "fs"):
        raise ValueError("unknown matched backend")
    if backend in ("cpu_dct", "fastlens") and cache_bytes:
        raise ValueError("decoded cache is supported only by IC/FS readers")
    multiprocess = backend != "fastlens" and config.get("cpu_loader") == "dataloader"
    torch.set_num_threads(config["torch_threads"])
    target_device = torch.device(device)
    pool_bytes = min(config["preparation_pool_size"], len(trace))*(sum(math.prod(t["shape"])*4 for t in profile["tensors"])+8)
    # Current + next transformed pool + next pool's stacking input; at most
    # two pinned pools await DMA. Model state/inflate workspace are separate.
    host_input_bound = config["reader_memory_budget_bytes"]+pool_bytes*(5 if target_device.type == "cuda" else 3)
    if multiprocess:
        host_input_bound = (config["workers"]*config["reader_memory_budget_bytes"] +
                           pool_bytes*(config["workers"]*(config["prefetch_factor"]+2)+3))
    if backend != "fastlens" and host_input_bound > config["host_input_budget_bytes"]:
        raise MemoryError("reader, prepared pools and pinned staging exceed the declared host input budget")
    model = build_model(profile, target_device) if run_kind == "inference" else None
    if model is not None:
        from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
        if tensor_state_sha256(model.state_dict()) != evidence.get("loaded_model_state_sha256"):
            raise ValueError("loaded checkpoint state differs from the inference evidence; verify this model first")
        if "model_implementation" in evidence:
            import inspect
            if [artifact(path) for path in sorted({inspect.getfile(type(m)) for m in model.modules()})] != evidence["model_implementation"]:
                raise ValueError("model implementation differs from the inference evidence")
    if target_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    count = top1 = top5 = h2d = h2d_coefficients = pinned_peak = 0
    passes, elapsed, cpu_elapsed = 0, 0., 0.
    pass_seconds, resource_windows = [], []
    warming = warmup_seconds > 0
    warmup = dict(dataset_passes=0, elapsed_wall_s=0.)
    cache_observations, native_pass_metrics = [], []
    predictions, pinned_pending = [], []
    gpu_profile = []
    gpu_stage_ms = dict(h2d=0., model=0.)
    outstanding_pools = []
    first_event = None
    first_latency = None
    initialization_started, initialization_cpu = time.perf_counter(), time.process_time()
    with ExitStack() as stack:
        from fastlens.benchmarks.dct_models.measure_process import ProcessWindow
        monitor_scope = stack.enter_context(ExitStack())
        reader = native = None
        content_check = None
        if backend == "fastlens":
            from .unified import UnifiedReplay
            for ref in evidence["native_sources"]:
                if artifact(ref["path"]) != ref:
                    raise ValueError("native source differs from the inference evidence")
            from fastlens.benchmarks.dct_models.storage import BRIDGE
            current_inputs = [artifact(path) for path in cold_input_files(backend, root, records, None)]
            if current_inputs != evidence.get("native_input_files"):
                raise ValueError("native input files changed or evidence lacks content binding; rerun verify-inference")
            native = stack.enter_context(UnifiedReplay(backend, root, config, source, profile,
                                                       device=device, module_path=module_path))
            native.identity["coefficient_bridge"] = artifact(BRIDGE)
            if native.identity != evidence["backend"]:
                raise ValueError("native manifest or binary differs from the inference evidence")
            print("Reusing full source q/Q verification: native input content and binaries unchanged.", flush=True)
        elif backend == "cpu_dct":
            import dct_manip
            if evidence["backend"] != dict(kind=backend, decoder=artifact(dct_manip.__file__)):
                raise ValueError("JPEG decoder differs from the inference evidence")
            workers = stack.enter_context(ThreadPoolExecutor(max_workers=config["workers"]))
            replay = Replay(None, profile, config, records=records, executor=workers)
        else:
            reader = stack.enter_context(StoreReader(root, workers=config["workers"], cache_bytes=cache_bytes,
                decode_budget_bytes=config["decode_budget_bytes"], memory_budget_bytes=config["reader_memory_budget_bytes"],
                verify_reads=False, profile_stages=profile_stages))
            if evidence["store"] != reader.identity:
                raise ValueError("correctness evidence belongs to another store content identity")
            if backend is not None and evidence["backend"]["kind"] != backend:
                raise ValueError("backend differs from the inference evidence")
            # Frozen benchmark files: hash once before timing, as for CPU-DCT.
            # NPZ CRC and shape/dtype validation remain in the actual decoder.
            content_check = reader.verify_content()
            replay = Replay(reader, profile, config)
        source_check = verify_source_files(source)
        cache_paths = cold_input_files(backend, root, records, reader) if cold_cache or observe_cache else []
        loader = inference_loader(root if reader else None, profile, config, records, trace) if multiprocess else None
        counters = Counters()
        pool_size = config["preparation_pool_size"]
        prefetch = stack.enter_context(ThreadPoolExecutor(max_workers=1)) if native is None else None
        def prepared_pools():
            if loader is not None:
                yield from loader
            elif native is not None:
                # Native producer, stream dependencies and projected readiness
                # stay intact. Never synchronize its stats at a microbatch.
                for rows, tensors in native.inference_batches(trace):
                    yield [(tensors, torch.tensor([r["target_label"] for r in rows]))], Counters()
            else:
                pools = iter(trace[offset:offset+pool_size] for offset in range(0, len(trace), pool_size))
                future = prefetch.submit(replay.pool, next(pools))
                while future is not None:
                    batches, observed = future.result()
                    following = next(pools, None)
                    future = prefetch.submit(replay.pool, following) if following is not None else None
                    yield batches, observed
        initialization = dict(wall_s=time.perf_counter()-initialization_started, cpu_core_s=time.process_time()-initialization_cpu)
        while warming or passes == 0 or elapsed < min_seconds:
            if cold_cache and not warming:
                cache_observations.append(evict_input_files(cache_paths))
            elif observe_cache and not warming and passes == 0:
                from fastlens.benchmarks.profiling.input_motivation import cache_pages
                cache_observations.append(dict(cache_pages(cache_paths), method="mincore; no eviction",
                                               scope="after preflight and complete warmup passes, before measurement"))
            if not warming and (passes == 0 or cold_cache):
                monitor = monitor_scope.enter_context(ProcessWindow())
            started, cpu_started = time.perf_counter(), time.process_time()
            if passes == 0:
                measurement_started = started
            gpu_start = None
            if target_device.type == "cuda":
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
            with torch.inference_mode():
                for batches, observed in prepared_pools():
                    counters.add(observed)
                    if target_device.type == "cuda" and native is None:
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
                            pinned = tuple(t.pin_memory() for t in ((labels,) if native else (*tensors, labels)))
                            size = sum(t.numel()*t.element_size() for t in pinned)
                            h2d += size
                            h2d_coefficients += sum(t.numel()*t.element_size() for t in pinned[:-1])
                            if profile_stages:
                                begin = torch.cuda.Event(enable_timing=True)
                                begin.record()
                            values = tuple(t.to(target_device, non_blocking=True) for t in pinned)
                            if profile_stages:
                                end = torch.cuda.Event(enable_timing=True)
                                end.record()
                                gpu_profile.append(("h2d", begin, end))
                            if native is None:
                                tensors = values[:-1]
                            labels = values[-1]
                            event = torch.cuda.Event()
                            event.record()
                            pinned_pending.append((event, size))
                            pinned_peak = max(pinned_peak, sum(size for _, size in pinned_pending))
                        if model is not None:
                            if profile_stages and target_device.type == "cuda":
                                begin = torch.cuda.Event(enable_timing=True)
                                begin.record()
                            logits = model(*tensors)
                            if profile_stages and target_device.type == "cuda":
                                end = torch.cuda.Event(enable_timing=True)
                                end.record()
                                gpu_profile.append(("model", begin, end))
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
                    if target_device.type == "cuda" and native is None:
                        pool_done = torch.cuda.Event()
                        pool_done.record()
                        outstanding_pools.append(pool_done)
                # Metric collection and the last CUDA completion are in the window.
                if predictions:
                    values = torch.stack([torch.stack((finite.long(), a, b)) for finite, a, b in predictions]).cpu()
                    if not bool(values[:, 0].all()):
                        raise FloatingPointError("non-finite benchmark logits")
                    pass_top1 = int(values[:, 1].sum())
                    pass_top5 = int(values[:, 2].sum())
                    if passes == 0:
                        top1, top5 = pass_top1, pass_top5
                    elif (pass_top1, pass_top5) != (top1, top5):
                        raise AssertionError("replayed predictions differ from the first complete pass")
                    predictions.clear()
                if target_device.type == "cuda":
                    torch.cuda.synchronize()
                pass_elapsed = time.perf_counter()-started
                elapsed = elapsed+pass_elapsed if cold_cache else time.perf_counter()-measurement_started
                if not warming:
                    pass_seconds.append(pass_elapsed)
                cpu_elapsed += time.process_time()-cpu_started
                if first_event is not None and first_latency is None:
                    first_latency = gpu_start.elapsed_time(first_event)/1000
            passes += 1
            for stage, begin, end in gpu_profile:
                gpu_stage_ms[stage] += begin.elapsed_time(end)
            gpu_profile.clear()
            if cold_cache and not warming:
                monitor_scope.close()
                resource_windows.append(monitor.result)
            if native is not None:
                metrics = asdict(native.pipeline.metrics)
                if not metrics["complete"]:
                    raise AssertionError("native metrics are incomplete after final CUDA synchronization")
                native_pass_metrics.append(metrics)
            phase = "Warmup" if warming else "Measured"
            print(f"{phase} dataset passes: {passes}; samples: {count}; timed seconds: {elapsed:.3f}", flush=True)
            if warming and elapsed >= warmup_seconds:
                warmup = dict(dataset_passes=passes, elapsed_wall_s=elapsed)
                warming = False
                passes, elapsed, cpu_elapsed = 0, 0., 0.
                count = top1 = top5 = h2d = h2d_coefficients = pinned_peak = 0
                counters = Counters()
                native_pass_metrics.clear()
                gpu_stage_ms = dict(h2d=0., model=0.)
                pinned_pending.clear()
                outstanding_pools.clear()
                first_event = first_latency = None
                if target_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
        if not cold_cache:
            monitor_scope.close()
            resource_windows.append(monitor.result)
        if count != len(trace)*passes:
            raise AssertionError("trace occurrence coverage mismatch")
        if model is not None and "metrics" in evidence:
            if (top1/len(trace), top5/len(trace)) != (evidence["metrics"]["top1_fraction"], evidence["metrics"]["top5_fraction"]):
                raise AssertionError("timed predictions differ from the correctness evidence")
        store = reader.manifest if reader else None
        init = asdict(reader.initialization_counters) if reader else None
        store_identity = reader.identity if reader else None
        native_metrics = native_pass_metrics[0] if len(native_pass_metrics) == 1 else None
    result = result_record(config_path, config, profile, root, evidence_path, evidence, target_device, store,
                           count, len({r["image_id"] for r in trace}), elapsed, first_latency, counters,
                           h2d, h2d_coefficients, pinned_peak, run_kind, top1, top5, cache_bytes,
                           store_identity, cpu_elapsed, backend=backend)
    result["resolved_parameters"]["measurement_min_seconds"] = min_seconds
    result["resolved_parameters"]["warmup_seconds"] = warmup_seconds
    result["resolved_parameters"]["cpu_loader"] = config.get("cpu_loader", "threadpool") if not native else None
    result["resolved_parameters"]["decoded_cache_bytes_per_reader"] = cache_bytes
    result["resolved_parameters"]["cache_policy"] = "cold_each_pass" if cold_cache else "continuous_post_warmup"
    result["resolved_parameters"]["profile_stages"] = profile_stages
    result["counts"]["dataset_passes"] = passes
    result["metrics"]["first_pass_wall_s"] = pass_seconds[0]
    result["metrics"]["pass_wall_s"] = pass_seconds
    result["quality"]["sample_count"] = len(trace) if model is not None else None
    if model is not None:
        result["quality"].update(top1_fraction=top1/len(trace), top5_fraction=top5/len(trace))
    result["metrics"]["cpu_core_seconds"] = (sum(w["cpu_core_seconds"] for w in resource_windows)
                                              if multiprocess else cpu_elapsed)
    result["metrics"]["memory"]["host_rss_peak_bytes"] = max(w["tree_rss_peak_bytes"] for w in resource_windows)
    reads = [w["physical_read_bytes"] for w in resource_windows]
    result["metrics"]["io"]["physical_storage_read_bytes"] = None if None in reads else sum(reads)
    if multiprocess:
        result["resolved_parameters"]["prefetch"] = config["prefetch_factor"]
    result["notes"] = [n for n in result["notes"] if not n.startswith(("Host RSS", "Application bytes"))]
    result["notes"].append("Application bytes count compressed archives; process-tree read_bytes measures storage I/O separately.")
    result["notes"].append("CPU and physical reads cover the measured process tree; sampled summed RSS double-counts shared pages. Reader array peaks are worker-local; external NVML monitoring covers process lifetime.")
    if warmup_seconds:
        result["notes"] = [n for n in result["notes"] if not n.startswith("Timing includes")]
        result["notes"].append("Warmup is excluded; timed complete dataset passes include first input delivery and final CUDA completion.")
    if cold_cache:
        result["cache_mode"] = "cold_verified"
        result["notes"][0] = "Matched inference with verified cold input file pages before each complete dataset pass."
    elif observe_cache:
        result["cache_mode"] = "mixed_measured"
        result["notes"][0] = "Matched inference with observed post-warmup page residency; continuous replay without per-pass eviction."
    result["notes"].append("Content/source verification and cache preparation are outside timing; files must remain immutable during the run.")
    # Keep detailed accounting separate from the draft schema's bounded fields.
    detail = dict(counters=asdict(counters), initialization_io=init, cpu_core_seconds=result["metrics"]["cpu_core_seconds"],
                  parent_cpu_core_seconds=cpu_elapsed,
                  initialization=initialization, content_verification=content_check,
                  source_verification=source_check, dataset_passes=passes, measurement_min_seconds=min_seconds,
                  cache_preparation=cache_observations, native_pass_metrics=native_pass_metrics, warmup=warmup,
                  resource_windows=resource_windows, pass_wall_s=pass_seconds,
                  preflight_touched=dict(jpeg_images=len(records), store_files=len(reader.manifest["files"]) if reader else None),
                  stage_timing=(dict(cpu_file_read_work_s=counters.file_read_wall_s,
                                     cpu_inflate_work_s=counters.inflate_wall_s,
                                     cpu_materialize_work_s=counters.materialize_wall_s,
                                     gpu_events_ms=gpu_stage_ms,
                                     scope="profiling run; concurrent worker/stream stage durations are not additive E2E time; native H2D event covers labels only")
                                if profile_stages else None),
                  backend=evidence.get("backend"), native_metrics=native_metrics,
                  experiment_type=f"matched_system_{run_kind}",
                  loaded_model_state_sha256=evidence.get("loaded_model_state_sha256"),
                  prepared_pool_bytes=pool_bytes if native is None else None,
                  host_input_reserved_bound_bytes=host_input_bound if native is None else None,
                  memory_scope="retained decoded arrays, LRU and gathered q/Q per reader; excludes NumPy inflate workspace, model tensors and pinned staging",
                  decode_amplification_denominator="sum of per-preparation-pool deduplicated required q values",
                  decode_amplification=(counters.decoded_q_values/counters.unique_requested_q_values
                                        if counters.unique_requested_q_values else None))
    if reader is None:
        detail["counters"] = None  # These store-specific counters were not measured.
    return result, detail


def result_record(config_path, config, profile, root, evidence_path, evidence, device, store,
                  count, unique, elapsed, first_latency, counters, h2d, h2d_coefficients, pinned_peak, run_kind, top1, top5, cache_bytes,
                  store_identity, cpu_elapsed, *, backend=None):
    sizes = storage_bytes(root) if store else dict(coefficient_data_bytes=None, metadata_index_label_bytes=None,
                                                  total_unique_bytes=None, allocated_disk_bytes=None)
    method = store["method"] if store else backend
    native = backend == "fastlens"
    correctness = {key: evidence[key] for key in ("status", "source_q_exact", "quantization_tables_exact",
                   "dependency_closure_verified", "final_tensor_match", "logits_match", "training_update_match",
                   "max_abs_tensor_error", "max_relative_tensor_error")}
    quality = dict(status="measured" if run_kind == "inference" else "not_measured", split="trace",
                   sample_count=count if run_kind == "inference" else None,
                   top1_fraction=top1/count if run_kind == "inference" else None,
                   top5_fraction=top5/count if run_kind == "inference" else None,
                   allowed_loss_percentage_points=None, observed_loss_percentage_points=None)
    implementation = [{"name": Path(ref["path"]).name, "sha256": ref["sha256"]} for ref in evidence["implementation"]]
    result = dict(schema_version="0.2-diagnostic", run_id=f"{method}-{time.time_ns()}", status="completed",
                experiment_type=f"matched_system_{run_kind}",
                execution_partition=dict(read_decode_geometry_normalization="native CUDA pipeline" if native else "cpu", augmentation="disabled",
                                         model=str(device) if run_kind == "inference" else "not executed"),
                loaded_model_state_sha256=evidence.get("loaded_model_state_sha256"),
                fingerprints=dict(store=store_identity["content_fingerprint"] if store else None,
                                  implementation=fingerprint(implementation),
                                  native_binding=evidence["backend"]["binding"]["sha256"] if native else None),
                method=method, variant=(f"adapted_cpu_transform_{store['dtype_policy']}_decoded_cache_{cache_bytes}" if store
                                        else "native_cuda_pipeline" if native else "jpeg_coefficients_cpu_transform"),
                comparison_scope="source_representation_matched", run_kind=run_kind, measurement_mode="actual",
                cache_mode="decoded_host" if cache_bytes else "unverified", headline_eligible=False,
                artifacts=dict(resolved_contract=artifact(config_path), source_manifest=config["source_manifest"],
                               model_profile=config["model_profile"], request_trace=config["request_trace"],
                               label_mapping=config["label_mapping"], store=store_identity,
                               backend=evidence.get("backend"),
                               checkpoint=profile["checkpoint"], correctness_evidence=artifact(evidence_path), cache_evidence=None),
                environment=dict(repository_revision=config["repository_revision"],
                                 hardware=dict(cpu=platform.processor(), device=str(device),
                                               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                               gpu_uuid=str(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None,
                                               cpu_affinity=",".join(map(str, sorted(os.sched_getaffinity(0))))),
                                 software=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)),
                resolved_parameters=dict(model_microbatch_size=config["model_microbatch_size"],
                                         preparation_pool_size=config["preparation_pool_size"], storage_group_images=store["storage_group_images"] if store else None,
                                         workers=None if native else config["workers"], prefetch=None if native else 1,
                                         torch_threads=config["torch_threads"], precision=config["precision"],
                                         model=profile["model"], profile_id=profile["id"],
                                         decode_budget_bytes=config["decode_budget_bytes"], reader_memory_budget_bytes=config["reader_memory_budget_bytes"],
                                         host_input_budget_bytes=config["host_input_budget_bytes"],
                                         source_frequencies={str(c): v for c, v in enumerate(profile["source_frequencies"])},
                                         output_channels={"tensors": profile["tensors"], "channels": profile["output_channels"]}),
                counts=dict(sample_occurrences=count, unique_images=unique, optimizer_updates=0,
                             requested_q_values=counters.requested_q_values, physically_decoded_q_values=counters.decoded_q_values,
                             full_grid_requested_q_values=counters.full_grid_requested_q_values),
                metrics=dict(elapsed_wall_s=elapsed, cpu_core_seconds=cpu_elapsed, throughput_images_s=count/elapsed, estimated_elapsed_s=None,
                             estimated_throughput_images_s=None, first_batch_latency_s=first_latency,
                             batch_latency_s=dict(p50=None, p95=None, p99=None),
                             io=dict(compressed_payload_read_bytes=counters.compressed_payload_read_bytes,
                                     metadata_read_bytes=counters.metadata_read_bytes, application_read_bytes=counters.application_read_bytes,
                                     physical_storage_read_bytes=None, read_operations=counters.read_calls, file_open_operations=counters.file_opens),
                             h2d=dict(total_bytes=h2d, coefficient_payload_bytes=h2d_coefficients,
                                      metadata_indices_labels_padding_bytes=h2d-h2d_coefficients),
                             storage=dict(**{k: v for k, v in sizes.items() if k != "file_count"}, temporary_peak_bytes=None),
                             memory=dict(host_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                         active_decoded_peak_bytes=counters.active_decoded_peak_bytes,
                                         gathered_output_peak_bytes=counters.output_peak_bytes,
                                         reader_memory_peak_bytes=counters.reader_memory_peak_bytes,
                                         pinned_peak_bytes=pinned_peak, gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                                         gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None)),
                correctness=correctness, quality=quality,
                offline_cost=dict(included_in_elapsed_wall=False, conversion_wall_s=store["conversion_wall_s"] if store else None,
                                  cpu_core_hours=store["conversion_cpu_s"]/3600 if store else None,
                                  gpu_hours=0 if store else None, search_trials=0, profile_copies=1 if store else None), synthetic_delivery=None,
                notes=["Diagnostic inference pilot; cache state is unverified and full-scale performance is outstanding.",
                       "Application bytes include complete NPZ archives; physical disk bytes are unmeasured.",
                       "CPU transform adapter transfers final float32 tensors and labels, not compact coefficients.",
                       "H2D coefficient payload counts final float32 DCT model inputs; the other transfer is int64 labels.",
                       "Host RSS is process high-water mark, including initialization; threads share the same process.",
                       "Timing includes first preparation pool and final completion. No excluded warmup is performed."])
    if store is None:
        for key in ("requested_q_values", "physically_decoded_q_values", "full_grid_requested_q_values"):
            result["counts"][key] = None
        result["metrics"]["io"] = dict.fromkeys(result["metrics"]["io"])
        for key in ("active_decoded_peak_bytes", "gathered_output_peak_bytes", "reader_memory_peak_bytes"):
            result["metrics"]["memory"][key] = None
        result["notes"] = [note for note in result["notes"] if not note.startswith("Application bytes")]
        result["notes"].append("JPEG/native byte accounting and conversion costs are unmeasured; null is not zero.")
    if native:
        result["metrics"]["h2d"] = dict.fromkeys(result["metrics"]["h2d"])
        result["metrics"]["memory"]["pinned_peak_bytes"] = None
        result["notes"] = [note for note in result["notes"] if not note.startswith(("CPU transform", "H2D coefficient"))]
        result["notes"].extend([
            "Native producer defaults run under the same CPU affinity and GPU allocation; IC/FS reader budgets do not control native memory.",
            "Torch peaks exclude native allocations; native transient metrics and process-level NVML observations are separate.",
        ])
    return result


def report(paths, output):
    import statistics
    import jsonschema
    schema = json.loads(SCHEMA.read_text())
    groups = {}
    for path in paths:
        process_path = Path(path).with_suffix(".process.json")
        if process_path.exists():
            process = json.loads(process_path.read_text())
            if process.get("valid") is False or process.get("returncode", 0):
                raise ValueError(f"invalid resource/process observation: {process_path}")
        for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
            row = json.loads(line)
            jsonschema.Draft202012Validator(schema).validate(row)
            if row["status"] != "completed":
                continue
            # Different checkpoints/traces, cache/runtime controls and variants
            # cannot silently share a summary row.
            key = json.dumps({k: row[k] for k in ("method", "variant", "run_kind", "cache_mode", "resolved_parameters", "fingerprints",
                                                 "measurement_mode", "comparison_scope", "environment")}, sort_keys=True)
            key += json.dumps({k: row["artifacts"][k] for k in ("resolved_contract", "checkpoint", "request_trace")}, sort_keys=True)
            key += json.dumps([row.get("experiment_type", "legacy_unclassified"), row.get("execution_partition"),
                               row.get("loaded_model_state_sha256"), row["artifacts"].get("backend")], sort_keys=True)
            groups.setdefault(key, []).append((row, dict(path=str(Path(path).resolve()), line=line_number,
                                                        run_id=row["run_id"], correctness_evidence=row["artifacts"]["correctness_evidence"])))
    rows = []
    for observations in groups.values():
        group = [row for row, _ in observations]
        mode = group[0]["measurement_mode"]
        raw = [r["metrics"]["estimated_throughput_images_s" if mode == "modeled" else "throughput_images_s"] for r in group]
        identity = group[0]
        rows.append(dict(method=identity["method"], variant=identity["variant"], run_kind=identity["run_kind"],
                         experiment_type=identity.get("experiment_type", "legacy_unclassified"),
                         execution_partition=identity.get("execution_partition"),
                         loaded_model_state_sha256=identity.get("loaded_model_state_sha256"),
                         model=identity["resolved_parameters"]["model"], profile_id=identity["resolved_parameters"]["profile_id"],
                         fingerprints=identity["fingerprints"], artifacts=identity["artifacts"],
                         environment=identity["environment"], resolved_parameters=identity["resolved_parameters"],
                         run_ids=[r["run_id"] for r in group],
                         raw_runs=[ref for _, ref in observations],
                         measurement_mode=mode, cache_mode=group[0]["cache_mode"],
                         headline_eligible=all(r["headline_eligible"] for r in group), repeats=len(raw),
                         median_images_s=statistics.median(raw), min_images_s=min(raw), max_images_s=max(raw), raw_images_s=raw))
        rows[-1]["stdev_images_s"] = statistics.stdev(raw) if len(raw) > 1 else None
        first = [r["metrics"].get("first_pass_wall_s") for r in group]
        if all(value is not None for value in first):
            rows[-1].update(median_first_pass_wall_s=statistics.median(first), raw_first_pass_wall_s=first)
        rows[-1]["quality"] = identity["quality"]
    write_json(output, rows)
    return rows
