"""Isolated runner control-flow probes; no torch, CUDA, model, or full repo execution.

The benchmark function and report/result_record functions are compiled unchanged
from the uploaded source AST. External pipeline components are small stand-ins.
JSON schema validation is replaced by a required-field check solely to isolate
report grouping (this is not claimed as Draft 2020-12 validation).
"""
from __future__ import annotations

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import types
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "dct_baselines" / "runner.py"
SCHEMA = HERE.parent.parent / "output" / "fastlens_agent_pack" / "contracts" / "result.schema.json"
FIXTURE = HERE / "fixture"
FIXTURE.mkdir(exist_ok=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_artifact(ref):
    assert artifact(ref["path"]) == ref
    return json.loads(Path(ref["path"]).read_text())


def burn_cpu(seconds):
    until = time.process_time() + seconds
    while time.process_time() < until:
        pass


@dataclass
class Counters:
    application_read_bytes: int = 0
    compressed_payload_read_bytes: int = 0
    metadata_read_bytes: int = 0
    file_opens: int = 0
    read_calls: int = 0
    decoded_q_values: int = 0
    decoded_q_bytes: int = 0
    requested_q_values: int = 0
    unique_requested_q_values: int = 0
    decoded_cache_hits: int = 0
    decoded_cache_misses: int = 0
    cache_evictions: int = 0
    decoded_cache_peak_bytes: int = 0

    def add(self, other):
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


class DummyDevice:
    def __init__(self, value):
        self.value = value
        self.type = value.split(":")[0]

    def __str__(self):
        return self.value


class DummyTorch:
    __version__ = "isolated-probe-no-torch"
    autocast_calls = 0
    device = DummyDevice

    @staticmethod
    def set_num_threads(value):
        pass

    @staticmethod
    def inference_mode():
        return nullcontext()

    @classmethod
    def autocast(cls, *args, **kwargs):
        cls.autocast_calls += 1
        return nullcontext()


class DummyLabels:
    def numel(self):
        return 1


STORE = {
    "method": "ic_matched_shared", "dtype_policy": "paper",
    "storage_group_images": 1, "conversion_wall_s": 0., "conversion_cpu_s": 0.,
}


class DummyReader:
    def __init__(self, *args, **kwargs):
        # Stand-in for expensive initialization outside wall-clock window.
        burn_cpu(.025)
        self.manifest = STORE
        self.initialization_counters = Counters(metadata_read_bytes=1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyReplay:
    def __init__(self, *args):
        pass

    def pool(self, rows):
        return [((), DummyLabels()) for _ in rows], Counters()


def storage_bytes(root):
    # Stand-in for all-NPZ storage accounting after elapsed has been recorded.
    burn_cpu(.075)
    return dict(coefficient_data_bytes=1, metadata_index_label_bytes=1,
                total_unique_bytes=2, allocated_disk_bytes=4096, file_count=1)


class MinimalValidator:
    def __init__(self, schema):
        self.schema = schema

    def validate(self, row):
        # Inputs originate from the original result_record function below.
        assert set(self.schema["required"]).issubset(row)


sys.modules["jsonschema"] = types.SimpleNamespace(Draft202012Validator=MinimalValidator)
namespace = dict(
    Path=Path, json=json, math=math, time=time, torch=DummyTorch, np=np,
    platform=platform, resource=resource, artifact=artifact, load_artifact=load_artifact,
    load_trace=lambda path, profile, records: [json.loads(s) for s in Path(path).read_text().splitlines()],
    implementation_artifacts=lambda: [artifact(SOURCE)],
    StoreReader=DummyReader, Replay=DummyReplay, Counters=Counters,
    ThreadPoolExecutor=ThreadPoolExecutor, asdict=asdict, storage_bytes=storage_bytes,
    SCHEMA=SCHEMA, write_json=write_json,
)
parsed = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
            and node.name in {"inputs", "benchmark", "result_record", "report"}]
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)

# Every dependency/artifact fingerprint used by inputs is real for this fixture;
# source decoding, model building and schema validation are explicitly excluded.
dependency = FIXTURE / "dependency.txt"
dependency.write_text("immutable test dependency\n")
dep = artifact(dependency)
source = FIXTURE / "source.json"
write_json(source, dict(version=1, representation="original_jpeg_quantized_dct",
                        images=[dict(image_id="0")]))
profile = FIXTURE / "profile.json"
write_json(profile, dict(checkpoint=dep, resize_implementation=dep, model_adapter=dep,
                         dependency_evidence=dep, source_frequencies=[list(range(64))],
                         tensors=[dict(name="x", shape=[1, 1, 1])], output_channels=[[0, 0]]))
trace = FIXTURE / "trace.jsonl"
trace.write_text(json.dumps(dict(image_id="0", target_label=0, microbatch_id=0)) + "\n")
config_path = FIXTURE / "matched.json"
write_json(config_path, dict(version=1, source_manifest=artifact(source),
                            model_profile=artifact(profile), request_trace=artifact(trace),
                            repository_revision="unchanged-head", precision="bf16-autocast",
                            tail_policy="keep", tolerance=dict(atol=0., rtol=0.),
                            model_microbatch_size=1, preparation_pool_size=1,
                            workers=1, torch_threads=1))
write_json(FIXTURE / "store.json", STORE)
evidence_path = FIXTURE / "evidence.json"
write_json(evidence_path, dict(
    status="passed", resolved_contract=artifact(config_path), store=artifact(FIXTURE / "store.json"),
    implementation=[artifact(SOURCE)], device="cpu", logits_match=False,
    source_q_exact=True, quantization_tables_exact=True, dependency_closure_verified=True,
    final_tensor_match=True, training_update_match=False,
    max_abs_tensor_error=0., max_relative_tensor_error=0.,
))
row, detail = namespace["benchmark"](config_path, FIXTURE, evidence_path)
assert row["metrics"]["elapsed_wall_s"] < .05
assert detail["cpu_core_seconds"] >= .1
assert row["resolved_parameters"]["precision"] == "fp32"

first, second = copy.deepcopy(row), copy.deepcopy(row)
first["run_id"], second["run_id"] = "version-a", "version-b"
first["metrics"]["throughput_images_s"] = 100.
second["metrics"]["throughput_images_s"] = 900.
first["metrics"]["elapsed_wall_s"] = first["counts"]["sample_occurrences"] / 100.
second["metrics"]["elapsed_wall_s"] = second["counts"]["sample_occurrences"] / 900.
second_evidence = FIXTURE / "evidence-b.json"
changed_evidence = json.loads(evidence_path.read_text())
changed_evidence["implementation"][0]["sha256"] = "b" * 64
changed_evidence["store"]["sha256"] = "c" * 64
write_json(second_evidence, changed_evidence)
second["artifacts"]["correctness_evidence"] = artifact(second_evidence)
raw = FIXTURE / "changed-implementation.jsonl"
raw.write_text("\n".join(map(json.dumps, (first, second))) + "\n")
summary = namespace["report"]([raw], FIXTURE / "merged-summary.json")
assert len(summary) == 1 and summary[0]["repeats"] == 2
assert summary[0]["median_images_s"] == 500.

# Separately show that different contracts remain separate internally, but all
# their identifying contract/model/device fields disappear in the summary.
second["artifacts"]["checkpoint"] = artifact(second_evidence)
second["artifacts"]["resolved_contract"] = artifact(second_evidence)
second["environment"]["hardware"]["device"] = "different-device"
raw.write_text("\n".join(map(json.dumps, (first, second))) + "\n")
separate_summary = namespace["report"]([raw], FIXTURE / "different-contract-summary.json")
assert len(separate_summary) == 2
identity = lambda r: {k: v for k, v in r.items()
                      if k not in {"median_images_s", "min_images_s", "max_images_s", "raw_images_s"}}
assert identity(separate_summary[0]) == identity(separate_summary[1])

result = {
    "scope": "AST-isolated control flow; no real PyTorch/CUDA/full repo execution; schema required-fields check only",
    "source": artifact(SOURCE),
    "cpu_accounting": {
        "injected_outside_window_cpu_s": .1,
        "elapsed_wall_s": row["metrics"]["elapsed_wall_s"],
        "reported_cpu_core_seconds": detail["cpu_core_seconds"],
        "outside_cpu_included": True,
    },
    "precision": {
        "accepted_config_precision": load_artifact(artifact(config_path))["precision"],
        "reported_precision": row["resolved_parameters"]["precision"],
        "autocast_calls": DummyTorch.autocast_calls,
    },
    "report_merges_distinct_evidence": {
        "evidence_refs_different": first["artifacts"]["correctness_evidence"] != second["artifacts"]["correctness_evidence"],
        "group_count": len(summary), "summary": summary,
    },
    "summary_drops_group_identity": {
        "group_count": len(separate_summary), "summary": separate_summary,
    },
}
write_json(HERE / "results.json", result)
print(json.dumps(result, indent=2))
