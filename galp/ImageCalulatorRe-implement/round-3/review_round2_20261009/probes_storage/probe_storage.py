"""Independent NumPy-only review of the original Round-2 storage modules.

Only the unavailable galp.benchmarks.common.sha256_file helper is shimmed.
No implementation under review is edited or replaced.
"""
from pathlib import Path
import importlib.util
import hashlib
import sys
import types
import json
import weakref
import threading
import gc
from dataclasses import asdict
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "reconstructed/galp/benchmarks/dct_baselines"
OUT = HERE / "artifacts"
OUT.mkdir(exist_ok=True)
common = types.ModuleType("galp.benchmarks.common")
common.sha256_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.modules[common.__name__] = common
package = types.ModuleType("review_storage")
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
for basename in ("contracts", "store"):
    name = f"review_storage.{basename}"
    spec = importlib.util.spec_from_file_location(name, SOURCE / f"{basename}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
contracts = sys.modules["review_storage.contracts"]
store = sys.modules["review_storage.store"]


def image(i, *, ragged=False, components=3):
    qs, tables, info = [], [], []
    rng = np.random.default_rng(900 + i)
    for c in range(components):
        h, w = ((2 + (i + c) % 2, 3 + c % 2) if ragged else (2, 3))
        q = rng.integers(-127, 128, size=(h, w, 8, 8), dtype=np.int16)
        q[..., 0, 0] = -129 - i - c
        q[..., 7, 7] = 128 + i + c
        qs.append(q)
        table = rng.integers(1, 65536, size=(8, 8), dtype=np.uint16)
        tables.append(table)
        info.append(dict(block_grid=[h, w]))
    return contracts.SourceImage(dict(image_id=str(i), block_size=8, components=info), tuple(qs), tuple(tables))


def exact(views, sources):
    for view in views:
        expected = sources[int(view.request.image_id)]
        for c, box in enumerate(view.request.boxes(expected.metadata)):
            y, x, h, w = box
            f = list(view.request.frequencies[c])
            q = expected.coefficients[c].reshape(*expected.coefficients[c].shape[:2], 64)
            np.testing.assert_array_equal(view.coefficients[c], q[y:y+h, x:x+w, f])
            np.testing.assert_array_equal(view.tables[c], expected.tables[c])


def payload_rewrite():
    records = []
    for method in ("ic_matched_shared", "fs_matched_b8"):
        for key in ("q", "tables"):
            root = OUT / f"rewrite_{method}_{key}"
            manifest = store.build_store([image(0)], root, method)
            group = manifest["groups"][0]
            name = group["table_file"] if key == "tables" else group["units"][0]["file"]
            with store.StoreReader(root) as reader:
                identity = reader.identity
                reader.verify_content()
                path = root / name
                with np.load(path, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                arrays[key].flat[0] += 1
                np.savez_compressed(path, **arrays)
                rejections = {}
                for operation in ("preflight", "decode"):
                    try:
                        if operation == "preflight":
                            reader.verify_content()
                        else:
                            reader.read(reader.plan([contracts.ReadRequest("0", (tuple(range(64)),)*3)]))
                    except ValueError as error:
                        assert "payload content changed" in str(error)
                        rejections[operation] = str(error)
                    else:
                        raise AssertionError(f"{method} {key}: stale bytes accepted by {operation}")
                assert reader.identity == identity
                records.append(dict(method=method, member=key, rejected=rejections))
    return records


def lifetimes():
    sources = [image(i) for i in range(32)]
    root = OUT / "lifetimes_icb"
    manifest = store.build_store(sources, root, "ic_matched_batched", group_size=4)
    member = manifest["files"][manifest["groups"][0]["units"][0]["file"]]["decoded_bytes"]
    requests = [contracts.ReadRequest(str(i), ((63, 0, 8),)*3, ((0, 1, 1, 2),)*3) for i in range(0, 32, 4)]
    results = []
    for workers in (1, 4):
        references, peaks, lock = [], [0], threading.Lock()
        admitted = 1 if workers == 1 else 2
        barrier = threading.Barrier(admitted)
        with store.StoreReader(root, workers=workers, cache_bytes=0,
                               decode_budget_bytes=admitted*member,
                               memory_budget_bytes=4*member) as reader:
            decode = reader._decode_file
            def tracked(name):
                arrays, counts = decode(name)
                with lock:
                    references.extend(weakref.ref(a) for a in arrays.values())
                    peaks[0] = max(peaks[0], sum(a.nbytes for r in references if (a := r()) is not None))
                barrier.wait(timeout=10)
                return arrays, counts
            reader._decode_file = tracked
            views, counts = reader.read(reader.plan(requests))
            exact(views, sources)
            gc.collect()
            alive = sum(a.nbytes for r in references if (a := r()) is not None)
            assert peaks[0] <= admitted*member and alive == 0
            assert counts.file_opens == 8 and counts.active_decoded_peak_bytes <= admitted*member
            results.append(dict(workers=workers, member_bytes=member, live_decoded_peak_bytes=peaks[0],
                                alive_after_read=alive, counters=asdict(counts)))
    return results


def scatter_and_cache():
    sources = [image(i, ragged=True) for i in range(9)]
    requests = [contracts.ReadRequest(str(i), ((63, 0, 8, 27, 1, 56), (1, 8, 15), (63, 7, 56)),
                                      ((0, 1, 2, 2), (1, 0, 1, 3), (0, 0, 2, 2)))
                for i in (8, 0, 4, 1, 8, 5)]
    results = []
    for method in ("ic_matched_shared", "ic_matched_batched", "fs_matched_b8"):
        root = OUT / f"scatter_{method}"
        manifest = store.build_store(sources, root, method, group_size=4)
        cache = sum(f["decoded_bytes"] for f in manifest["files"].values())
        with store.StoreReader(root, workers=4, cache_bytes=cache,
                               decode_budget_bytes=cache, memory_budget_bytes=3*cache) as reader:
            plan = reader.plan(requests)
            first, initial = reader.read(plan)
            exact(first, sources)
            for _ in range(5):
                second, hit = reader.read(plan)
                exact(second, sources)
                assert hit.file_opens == 0 and hit.decoded_q_bytes == 0 and hit.decoded_cache_hits == initial.file_opens
            results.append(dict(method=method, initial_counters=asdict(initial), cached_counters=asdict(hit)))
    return results


def cache_admission():
    root = OUT / "cache_admission"
    source = image(0, components=1)
    manifest = store.build_store([source], root, "ic_matched_shared")
    member = next(iter(manifest["files"].values()))["decoded_bytes"]
    memory = member*2 + 256
    with store.StoreReader(root, cache_bytes=member, decode_budget_bytes=member,
                           memory_budget_bytes=memory) as reader:
        tiny = contracts.ReadRequest("0", ((0,),), ((0, 0, 1, 1),))
        reader.read(reader.plan([tiny]))
        large = contracts.ReadRequest("0", (tuple(range(64)),))
        # Resident cache + gathered complete output = 2 * member, within the budget.
        output = source.coefficients[0].size * 4 + source.tables[0].nbytes
        try:
            reader.read(reader.plan([large]))
        except MemoryError as error:
            return dict(all_data_cached=True, cached_bytes=reader.cache_size, output_bytes=output,
                        actual_retained_needed=reader.cache_size+output, budget_bytes=memory,
                        rejected_because_nonexistent_decode_reserved=str(error))
        raise AssertionError("Expected current overly conservative cache admission to reject")


results = dict(scope="NumPy-only original contracts.py/store.py; only common.sha256_file shimmed",
               implementation_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (SOURCE/"store.py", SOURCE/"contracts.py")},
               payload_rewrite=payload_rewrite(), lifetimes=lifetimes(),
               scatter_cache=scatter_and_cache(), cache_admission=cache_admission())
(HERE / "results.json").write_text(json.dumps(results, indent=2)+"\n")
print(json.dumps(results, indent=2))
