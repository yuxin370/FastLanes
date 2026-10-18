"""Isolated NumPy review; the sole shim is the absent sha256_file utility.

No torch, galp backend, JPEG extension, model, or CUDA behavior is simulated.
All dct_baselines source files are imported unmodified from the upload copy.
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import types
import weakref

import numpy as np
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dct_baselines"
sys.path.insert(0, str(ROOT))
for name in ("galp", "galp.benchmarks"):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
common = types.ModuleType("galp.benchmarks.common")
common.sha256_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.modules[common.__name__] = common

from dct_baselines.contracts import SourceImage, ReadRequest, artifact
from dct_baselines.store import StoreReader, build_store, diagonal_frequencies, mask_family, storage_bytes
from dct_baselines.native import NativeSpec, candidates, quantize_rgb, mosaic, encode_ic, fs_source, write_ic_archive, read_ic_archive


def example(i, grids=((3, 4), (2, 2), (2, 2))):
    rng = np.random.default_rng(i)
    coeffs = tuple(rng.integers(-128, 128, (*g, 8, 8), dtype=np.int16) for g in grids)
    coeffs[0].reshape(-1)[:4] = [-129, -128, 127, 128]
    info = {"image_id": str(i), "label": i, "block_size": 8,
            "components": [{"block_grid": list(g), "component_id": c} for c, g in enumerate(grids)]}
    tables = tuple(rng.integers(1, 65536, (8, 8), dtype=np.uint16) for _ in grids)
    return SourceImage(info, coeffs, tables)


results = {"scope": "isolated NumPy/SciPy core tests; only absent galp.benchmarks.common.sha256_file is shimmed", "checks": []}

with tempfile.TemporaryDirectory(prefix="storage-probe-", dir=ROOT / "probes_storage") as tmp:
    tmp = Path(tmp)
    images = [example(0), example(1, ((1, 3),)), example(2, ((2, 1), (1, 1), (1, 1)))]
    for method in ("ic_matched_shared", "ic_matched_batched", "fs_matched_b8", "ic_matched_fixedk"):
        selected = [example(i) for i in range(3)] if method.endswith("fixedk") else images
        kwargs = {"source_frequencies": [[63, 8, 0]]*3, "dependency_evidence": "review explicit synthetic request"} if method.endswith("fixedk") else {}
        store = tmp / method
        manifest = build_store(selected, store, method, group_size=2, **kwargs)
        requests = []
        for i in (2, 0, 2, 1):
            masks = tuple((63, 8, 0) if kwargs else (63, 1, 8, 0, 15) for _ in selected[i].coefficients)
            regions = tuple((0, 0, 1, 1) for _ in masks)
            requests.append(ReadRequest(str(i), masks, regions))
        with StoreReader(store, workers=2) as reader:
            views, counters = reader.read(reader.plan(requests))
            for req, view in zip(requests, views):
                source = selected[int(req.image_id)]
                for c, q in enumerate(view.coefficients):
                    expected = source.coefficients[c].reshape(*source.coefficients[c].shape[:2], 64)
                    np.testing.assert_array_equal(q, expected[:1, :1, list(req.frequencies[c])])
                    np.testing.assert_array_equal(view.tables[c], source.tables[c])
            assert counters.requested_q_values > counters.unique_requested_q_values
            assert counters.application_read_bytes == counters.compressed_payload_read_bytes + counters.metadata_read_bytes
        assert storage_bytes(store)["total_unique_bytes"] == sum(p.stat().st_size for p in store.iterdir())
        results["checks"].append({"name": method + " roundtrip", "passed": True,
                                  "coverage": "ragged and grayscale where applicable; uint16 Q >32767; out-of-order K; spatial slice; duplicate/order/tail; byte accounting"})

    for b in (8, 32):
        groups = [diagonal_frequencies(b, d) for d in range(2*b-1)]
        assert sorted(f for g in groups for f in g) == list(range(b*b))
    assert mask_family([0, 1, 8], 8) == "native_triangle_k2"
    assert mask_family(range(64), 8) == "matched_arbitrary_adapter"
    results["checks"].append({"name": "diagonal partitions and native-mask exact equality", "passed": True})

    store = tmp / "overread"
    selected = [example(i) for i in range(2)]
    manifest = build_store(selected, store, "fs_matched_b8", group_size=2)
    with StoreReader(store) as reader:
        req = ReadRequest("0", ((1,),)*3, ((0, 0, 1, 1),)*3)
        _, counters = reader.read(reader.plan([req, req]))
        expected = 2 * sum(q.shape[0]*q.shape[1] for i in selected for q in i.coefficients)
        assert counters.decoded_q_values == expected
        assert counters.requested_q_values == 6
        assert counters.unique_requested_q_values == 3
        assert counters.file_opens == 2
    results["checks"].append({"name": "FS complete-column overread with partial diagonal + crop + duplicate", "passed": True,
                              "decoded_q_values": counters.decoded_q_values, "logical_q_values": 6, "unique_q_values": 3})

    store = tmp / "cache"
    build_store([example(i, ((1, 1),)) for i in range(2)], store, "ic_matched_shared")
    with StoreReader(store, cache_bytes=384) as reader:
        def read(i):
            return reader.read(reader.plan([ReadRequest(str(i), (tuple(range(64)),))]))[1]
        first, second, third, fourth = read(0), read(0), read(1), read(0)
        assert first.decoded_q_values == 64 and second.decoded_q_values == 0
        assert second.application_read_bytes == 0 and third.cache_evictions == 1 and fourth.decoded_q_values == 64
    results["checks"].append({"name": "decoded cache hits and bounded LRU eviction", "passed": True})

    width_image = example(0, ((1, 1),))
    width_image.coefficients[0].fill(0)
    width_image.coefficients[0].flat[0] = -129
    width_manifest = build_store([width_image], tmp / "width", "fs_matched_b8")
    assert width_manifest["groups"][0]["units"][0]["dtype"] == "<i2"
    assert width_manifest["groups"][0]["units"][1]["dtype"] == "|i1"
    widened = SourceImage(width_image.metadata, (width_image.coefficients[0].astype(np.int32),), width_image.tables)
    widened.coefficients[0].flat[0] = 32768
    try:
        build_store([widened], tmp / "overflow", "fs_matched_b8")
    except OverflowError:
        pass
    else:
        raise AssertionError("FS narrowing overflow not rejected")
    results["checks"].append({"name": "FS dtype signed range and overflow", "passed": True})

    rgbs = [np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8) for i in range(3)]
    for spec in (NativeSpec(8, 3, (2, 2), 20), NativeSpec(32, 3)):
        rgb = rgbs[0]
        r, g, b = np.moveaxis(rgb.astype(np.float64), -1, 0)
        y = .299*r + .587*g + .114*b
        planes = np.stack([y, (r-y)*.713+128, (b-y)*.564+128])-128
        planes = planes[:, ::spec.sampling[0], ::spec.sampling[1]]
        block = spec.block_size
        dense = planes.reshape(3, planes.shape[1]//block, block, planes.shape[2]//block, block).transpose(0, 1, 3, 2, 4)
        expected = np.rint(dctn(dense, type=2, norm="ortho", axes=(-2, -1))/spec.scalar_quantization).astype(np.int32)
        np.testing.assert_array_equal(quantize_rgb(rgb, spec), expected)
    assert len(candidates()) == 6048
    assert np.count_nonzero(mosaic(np.ones((3, 1, 1, 8, 8), dtype=np.int32), 8)) == 108
    spec = NativeSpec(32, 3)
    expected = [encode_ic(rgb, spec) for rgb in rgbs]
    write_ic_archive(tmp / "native.npz", rgbs, spec)
    for actual, target in zip(read_ic_archive(tmp / "native.npz", spec), expected):
        np.testing.assert_array_equal(actual, target)
    store = tmp / "fs_native"
    build_store([fs_source(rgb, i) for i, rgb in enumerate(rgbs)], store, "fs_native", group_size=2)
    with StoreReader(store) as reader:
        views, _ = reader.read(reader.plan([ReadRequest(str(i), (tuple(range(1024)),)*3) for i in range(3)]))
        for view, target in zip(views, expected):
            np.testing.assert_array_equal(mosaic(np.stack(view.dense()), 3), target)
    results["checks"].append({"name": "native SciPy DCT-II oracle, 6048 candidates, B8 triangle36, IC/IC-b/FS B32 mosaic", "passed": True})

    # Actual payload mutation reproducer. No runner or model is mocked.
    store = tmp / "mutation"
    source = example(0, ((1, 1),))
    manifest = build_store([source], store, "ic_matched_shared")
    before = artifact(store / "store.json")
    file = store / manifest["groups"][0]["units"][0]["file"]
    payload_before = artifact(file)
    with np.load(file, allow_pickle=False) as loaded:
        values = {key: loaded[key] for key in loaded.files}
    old_q, old_qtable = int(values["q"][0]), int(values["tables"][0, 0, 0])
    values["q"][0] += 1
    values["tables"][0, 0, 0] = 1 if old_qtable != 1 else 2
    np.savez_compressed(file, **values)
    after = artifact(store / "store.json")
    with StoreReader(store) as reader:
        views, _ = reader.read(reader.plan([ReadRequest("0", (tuple(range(64)),))]))
        accepted_q, accepted_qtable = int(views[0].coefficients[0][0, 0, 0]), int(views[0].tables[0][0, 0])
    assert before == after and payload_before != artifact(file)
    assert accepted_q == old_q+1 and accepted_qtable != old_qtable
    results["payload_mutation_reproducer"] = {
        "store_manifest_artifact_unchanged": before == after,
        "payload_artifact_changed": payload_before != artifact(file),
        "reader_accepted_changed_q": [old_q, accepted_q],
        "reader_accepted_changed_Q": [old_qtable, accepted_qtable],
        "runner_executed": False,
        "static_gate_implication": "runner.py:168,207-210 binds evidence only to store.json, so this mutation leaves its store identity condition satisfied",
    }

    # Observe references, not RSS or performance. Zero cache does not bound
    # the complete per-plan decoded working set.
    store = tmp / "retention"
    images = [example(i, ((2, 2),)) for i in range(16)]
    build_store(images, store, "ic_matched_batched", group_size=4)
    class ObserveReader(StoreReader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.refs = []
            self.peak_live_q_arrays = 0
            self.peak_live_q_bytes = 0
        def _decode_file(self, name):
            arrays, counters = super()._decode_file(name)
            self.refs.append(weakref.ref(arrays["q"]))
            live = [r() for r in self.refs if r() is not None]
            self.peak_live_q_arrays = max(self.peak_live_q_arrays, len(live))
            self.peak_live_q_bytes = max(self.peak_live_q_bytes, sum(a.nbytes for a in live))
            return arrays, counters
    with ObserveReader(store, cache_bytes=0) as reader:
        reqs = [ReadRequest(str(i), ((0,),), ((0, 0, 1, 1),)) for i in (0, 4, 8, 12)]
        _, counters = reader.read(reader.plan(reqs))
        assert reader.peak_live_q_arrays == 4
        assert reader.cache_size == 0
        results["decoded_working_set_reproducer"] = {
            "decoded_cache_budget_bytes": 0,
            "requested_q_values": counters.unique_requested_q_values,
            "whole_groups_simultaneously_live": reader.peak_live_q_arrays,
            "whole_q_bytes_simultaneously_live": reader.peak_live_q_bytes,
            "note": "Small synthetic retention proof. This is not an RSS benchmark or observed OOM."
        }

out = ROOT / "probes_storage" / "results.json"
out.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
