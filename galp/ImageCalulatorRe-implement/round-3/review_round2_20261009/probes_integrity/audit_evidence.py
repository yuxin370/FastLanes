"""Read-only artifact consistency audit; does not execute Torch or GPU evidence.

Original JSON bytes and absolute provenance paths remain unchanged. Explicit
submission prefixes are mapped solely to locate bytes for hash comparison.
"""
from pathlib import Path
import ast
import collections
import hashlib
import importlib.util
import json
import sys
import types

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = ROOT / "submission"
CODE = ROOT / "reconstructed/galp/benchmarks/dct_baselines"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped(path):
    rules = (
        ("/galp/benchmarks/dct_baselines/", CODE),
        ("/round-2-response/", SUB),
        ("/dct-round2-h100-20261009-205546/", SUB / "evidence-h100"),
    )
    for marker, directory in rules:
        if marker in path:
            candidate = directory / path.split(marker, 1)[1]
            if candidate.is_relative_to(directory):
                return candidate
    return None


def references(value, key=""):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            for hash_key in ("sha256", "jpeg_sha256"):
                if hash_key in value:
                    yield key, value["path"], value[hash_key]
        for name, child in value.items():
            yield from references(child, f"{key}/{name}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from references(child, f"{key}/{i}")


def main():
    training_path = CODE / "training.py"
    namespace = {}
    tree = ast.parse(training_path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("update_windows", "updates_per_epoch")]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(training_path), "exec"), namespace)
    schedule = []
    for population, batch, pool, accumulation, expected in (
            (8, 2, 2, 4, 4), (100, 4, 12, 4, 9), (1281167, 64, 4096, 16, 1252)):
        actual = namespace["updates_per_epoch"](population, batch, pool, accumulation)
        schedule.append(dict(population=population, microbatch=batch, pool=pool,
                             accumulation=accumulation, expected=expected, actual=actual,
                             passed=actual == expected))
    (HERE / "schedule_results.json").write_text(json.dumps(schedule, indent=2) + "\n")
    traces = []
    for trace_path in sorted((SUB / "evidence-v2/models").glob("*/contract/trace.jsonl")):
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
        # All three supplied diagnostic recipes declare accumulation 16.
        windows = namespace["update_windows"](rows, 16)
        traces.append(dict(model=trace_path.parent.parent.name, declared_accumulation=16,
                           window_count=len(windows),
                           microbatches_per_window=[len({r["microbatch_id"] for r in w}) for w in windows],
                           samples_per_window=[len(w) for w in windows]))
    (HERE / "trace_window_recheck.json").write_text(json.dumps(traces, indent=2) + "\n")
    roots = sorted(p for base in (SUB / "evidence-v2/models", SUB / "evidence-h100")
                   for p in base.rglob("*.json")
                   if p.name.endswith((".verified.json", ".calibration.json")))
    queue, visited, checks, runs = list(roots), set(), [], []
    for path in roots:
        data = json.loads(path.read_text())
        runs.append(dict(file=str(path.relative_to(SUB)), phase=data["phase"],
                         model=path.parent.name, backend=data["backend"]["kind"],
                         device=data["runtime"]["device"], status=data["status"],
                         headline_eligible=data["headline_eligible"], updates=data["updates"],
                         occurrences=data["occurrence_count"], unique_images=data["unique_image_count"],
                         max_abs_errors=data["max_abs_errors"], tolerance=data["tolerance"],
                         initial_model_sha256=data["initial_model_sha256"],
                         implementation_hashes_match=all(mapped(v["path"]) is not None and
                             sha(mapped(v["path"])) == v["sha256"] for v in data["implementation"]),
                         implementation_count=len(data["implementation"])))
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        for key, original, expected in references(json.loads(path.read_text())):
            local = mapped(original)
            if local is None:
                status = "external_dependency_not_bundled"
            elif not local.is_file():
                status = "snapshot_or_checkpoint_not_bundled" if local.suffix in (".pt", ".pth") else "missing"
            else:
                status = "match" if sha(local) == expected else "mismatch"
                if local.suffix == ".json" and local not in visited:
                    queue.append(local)
            checks.append(dict(referrer=str(path.relative_to(SUB)), key=key,
                               original_path=original, local_path=str(local) if local else None,
                               expected_sha256=expected, status=status))

    common = types.ModuleType("galp.benchmarks.common")
    common.sha256_file = sha
    sys.modules[common.__name__] = common
    package = types.ModuleType("review_integrity")
    package.__path__ = [str(CODE)]
    sys.modules[package.__name__] = package
    for name in ("contracts", "store"):
        module_name = f"review_integrity.{name}"
        spec = importlib.util.spec_from_file_location(module_name, CODE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    contracts = sys.modules["review_integrity.contracts"]
    store = sys.modules["review_integrity.store"]
    source = json.loads((SUB / "evidence-v2/source/source.json").read_text())
    source_images = {v["image_id"]: v for v in source["images"]}
    stores = []
    for path in sorted((SUB / "evidence-v2/stores").glob("*/store.json")):
        data = json.loads(path.read_text())
        payloads = []
        for filename, expected in data["files"].items():
            actual = path.parent / filename
            payloads.append(dict(file=filename, sha256_match=sha(actual) == expected["sha256"],
                                 size_match=actual.stat().st_size == expected["size_bytes"]))
        with store.StoreReader(path.parent) as reader:
            preflight = reader.verify_content()
            requests = [contracts.ReadRequest(image_id, (tuple(range(64)),) * 3)
                        for image_id in source_images]
            views, counters = reader.read(reader.plan(requests))
            coefficient_checks, table_checks = [], []
            for view in views:
                expected = source_images[view.request.image_id]
                for index, q in enumerate(view.dense()):
                    coefficient_checks.append(hashlib.sha256(np.asarray(q, dtype="<i2").tobytes()).hexdigest()
                                              == expected["components"][index]["coefficient_sha256"])
                    table_checks.append(np.array_equal(view.tables[index], expected["quantization_tables"][index]))
        stores.append(dict(store=str(path.relative_to(SUB)), group_images=data["storage_group_images"],
                           fingerprint=data["content_fingerprint"], payloads=payloads,
                           preflight=preflight, source_coefficient_hash_checks=coefficient_checks,
                           source_table_checks=table_checks))
    by_model = collections.defaultdict(set)
    for run in runs:
        by_model[run["model"]].add(run["initial_model_sha256"])
    summary = dict(
        root_record_count=len(roots), verification_records=sum(v["phase"] == "verification" for v in runs),
        calibration_records=sum(v["phase"] == "calibration" for v in runs),
        record_status_counts=dict(collections.Counter(v["status"] for v in runs)),
        all_module_hashes_match=all(v["implementation_hashes_match"] for v in runs),
        module_hash_comparisons=sum(v["implementation_count"] for v in runs),
        recursive_reference_status_counts=dict(collections.Counter(v["status"] for v in checks)),
        all_available_references_match=all(v["status"] not in ("mismatch", "missing") for v in checks),
        unique_initial_model_hashes_per_model={k: len(v) for k, v in by_model.items()},
        payload_files=sum(len(v["payloads"]) for v in stores),
        all_payloads_match=all(p["sha256_match"] and p["size_match"] for v in stores for p in v["payloads"]),
        coefficient_hash_checks=sum(len(v["source_coefficient_hash_checks"]) for v in stores),
        all_stored_q_match_declared_source_q=all(all(v["source_coefficient_hash_checks"]) for v in stores),
        all_stored_Q_match_declared_source_Q=all(all(v["source_table_checks"]) for v in stores),
        schedule_cases_passed=all(v["passed"] for v in schedule),
        observed_microbatches_per_update=sorted({n for v in traces for n in v["microbatches_per_window"]}),
        scope="Artifact consistency and NumPy store read only; no model, CUDA, parent-repository build, or independent JPEG coefficient extraction executed.")
    output = dict(summary=summary, runs=runs, references=checks, stores=stores)
    (HERE / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failures = [v for v in checks if v["status"] in ("mismatch", "missing")]
    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
