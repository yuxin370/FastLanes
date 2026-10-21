from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fastlens.benchmarks.baselines.dct_baselines.contracts import ReadRequest, SourceImage, signed_cast, load_trace
from fastlens.benchmarks.baselines.dct_baselines.native import (
    NativeSpec, candidates, encode_ic, fit_statistics, fs_source, mosaic, normalize_mosaic, quantize_rgb, read_ic_archive, write_ic_archive,
)
from fastlens.benchmarks.baselines.dct_baselines.store import (
    StoreReader, build_store, diagonal_frequencies, mask_family, storage_bytes,
)
from fastlens.benchmarks.baselines.dct_baselines.runner import benchmark, check_updates, implementation_artifacts
from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact, write_json


def image(identifier, grids=((3, 4), (2, 2), (2, 2)), b=8):
    rng = np.random.default_rng(identifier)
    q = tuple(rng.integers(-128, 128, size=(*grid, b, b), dtype=np.int16) for grid in grids)
    q[0].reshape(-1)[:4] = [-129, -128, 127, 128]
    metadata = dict(image_id=str(identifier), label=identifier, split="test", block_size=b,
                    components=[dict(component_id=c, block_grid=list(grid)) for c, grid in enumerate(grids)])
    return SourceImage(metadata, q, tuple(np.full((b, b), c+1, dtype="<u2") for c in range(len(grids))))


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch, prefix="dct-baselines-test-")
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()


class ParallelBuildTests(TemporaryTest):
    def test_group_submission_is_bounded_and_results_keep_source_order(self):
        from fastlens.benchmarks.baselines.dct_baselines.store import _build_groups
        consumed = []
        def tasks():
            for value in range(9):
                consumed.append(value)
                yield value
        results = _build_groups(tasks(), abs, 2)
        try:
            for expected in range(9):
                self.assertEqual(next(results), expected)
                self.assertLessEqual(len(consumed)-(expected+1), 1)
            with self.assertRaises(StopIteration):
                next(results)
        finally:
            results.close()

    def source(self):
        from PIL import Image
        from fastlens.benchmarks.baselines.dct_baselines.source import read_jpeg
        records = []
        rng = np.random.default_rng(407)
        # Odd grids, grayscale and a one-image final group.
        for i, shape in enumerate(((17, 19, 3), (11, 13), (9, 17, 3))):
            path = self.root/f'{i}.jpg'
            Image.fromarray(rng.integers(0, 256, shape, np.uint8)).save(path, quality=75, subsampling=2)
            sample = read_jpeg(path, str(i), i, 'test')
            records.append(dict(sample.metadata, quantization_tables=[t.tolist() for t in sample.tables],
                                galp_image_id=i))
        return dict(images=records)

    def test_parallel_jpeg_extraction_and_archives_match_serial_with_tail(self):
        from fastlens.benchmarks.baselines.dct_baselines.source import source_images
        source = self.source()
        expected = list(source_images(source))
        for method in ('ic_matched_shared', 'ic_matched_batched', 'fs_matched_b8'):
            with self.subTest(method=method):
                serial_root, parallel_root = self.root/f'{method}-serial', self.root/f'{method}-parallel'
                serial = build_store(source_images(source), serial_root, method, group_size=2)
                parallel = build_store(source, parallel_root, method, group_size=2, workers=2)
                self.assertEqual(serial['content_fingerprint'], parallel['content_fingerprint'])
                for key in ('images', 'groups', 'files'):
                    self.assertEqual(serial[key], parallel[key])
                for name in serial['files']:
                    self.assertEqual((serial_root/name).read_bytes(), (parallel_root/name).read_bytes())
                self.assertEqual([len(g['images']) for g in parallel['groups']],
                                 [1, 1, 1] if method == 'ic_matched_shared' else [2, 1])
                with StoreReader(parallel_root) as reader:
                    requests = [ReadRequest(str(i), (tuple(range(64)),)*len(expected[i].coefficients))
                                for i in (2, 0, 1, 2)]
                    views, _ = reader.read(reader.plan(requests))
                    for i, view in zip((2, 0, 1, 2), views):
                        for a, b in zip(view.dense(), expected[i].coefficients):
                            np.testing.assert_array_equal(a, b)
                        for a, b in zip(view.tables, expected[i].tables):
                            np.testing.assert_array_equal(a, b)

    def test_worker_source_failure_and_cross_group_duplicates_publish_no_manifest(self):
        import copy
        source = self.source()
        changed = copy.deepcopy(source)
        changed['images'][0]['jpeg_sha256'] = '0'*64
        duplicated = dict(images=[source['images'][0], source['images'][1], source['images'][0]])
        for name, invalid, error in (('changed', changed, 'source changed'),
                                     ('duplicate', duplicated, 'duplicate image ID')):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                build_store(invalid, self.root/name, 'ic_matched_batched', group_size=2, workers=2)
            self.assertFalse((self.root/name/'store.json').exists())


class StoreTests(TemporaryTest):
    def test_matched_roundtrip_ragged_grayscale_tail_reordering_and_repeat(self):
        images = [image(0), image(1, ((1, 3),)), image(2, ((2, 1), (1, 1), (1, 1)))]
        for method in ("ic_matched_shared", "ic_matched_fixedk", "ic_matched_batched", "fs_matched_b8"):
            with self.subTest(method=method):
                path = self.root / method
                fixed = method == "ic_matched_fixedk"
                # Use a uniform component count for this FixedK case; grayscale
                # is independently covered by the other shared/batch stores.
                population = [image(0), image(1), image(2)] if fixed else images
                masks = [[0, 8, 63]]*3 if fixed else None
                built = build_store(population, path, method, group_size=2, source_frequencies=masks,
                                    dependency_evidence="test explicit frequency-only request" if fixed else None)
                with StoreReader(path, workers=2) as reader:
                    requests = [ReadRequest(str(i), tuple(tuple([0, 8, 63]) if fixed else tuple(range(64))
                                                        for _ in population[i].coefficients)) for i in (2, 0, 2, 1)]
                    actual, stats = reader.read(reader.plan(requests))
                    for request, view in zip(requests, actual):
                        expected = population[int(request.image_id)]
                        for c, (q, table) in enumerate(zip(view.coefficients, view.tables)):
                            np.testing.assert_array_equal(q, expected.coefficients[c].reshape(*q.shape[:2], 64)[..., list(request.frequencies[c])])
                            np.testing.assert_array_equal(table, expected.tables[c])
                    self.assertGreater(stats.requested_q_values, stats.unique_requested_q_values)
                    self.assertEqual(stats.application_read_bytes, stats.compressed_payload_read_bytes+stats.metadata_read_bytes)
                if method in ("ic_matched_batched", "fs_matched_b8"):
                    self.assertEqual([len(g["images"]) for g in built["groups"]], [2, 1])
                sizes = storage_bytes(path)
                self.assertEqual(sizes["total_unique_bytes"], sum(p.stat().st_size for p in path.iterdir()))
                self.assertEqual(sizes["total_unique_bytes"], sizes["coefficient_data_bytes"]+sizes["metadata_index_label_bytes"])

    def test_partial_diagonal_reads_whole_group_and_spatial_grid(self):
        images = [image(0), image(1)]
        build_store(images, self.root / "fs", "fs_matched_b8", group_size=2)
        with StoreReader(self.root / "fs") as reader:
            req = ReadRequest("0", ((1,), (1,), (1,)), ((1, 1, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1)))
            plan = reader.plan([req, req])
            self.assertEqual(len(plan.units), 1)
            actual, stats = reader.read(plan)
            self.assertEqual(stats.decoded_q_values, 2*sum(q.shape[0]*q.shape[1] for i in images for q in i.coefficients))
            self.assertEqual(stats.requested_q_values, 6)
            self.assertEqual(stats.unique_requested_q_values, 3)
            for c, q in enumerate(actual[0].coefficients):
                y, x, h, w = req.regions[c]
                np.testing.assert_array_equal(q, images[0].coefficients[c].reshape(*images[0].coefficients[c].shape[:2], 64)[y:y+h, x:x+w, 1:2])
            self.assertEqual(stats.file_opens, 2)  # one diagonal, one Q metadata file

    def test_signed_width_edges_common_width_and_overflow(self):
        original = image(0, ((1, 1),))
        original.coefficients[0].fill(0)
        original.coefficients[0][0, 0, 0, 0] = -129
        built = build_store([original], self.root / "fs", "fs_matched_b8")
        self.assertEqual(built["groups"][0]["units"][0]["dtype"], "<i2")
        self.assertEqual(built["groups"][0]["units"][1]["dtype"], "|i1")
        common = build_store([original], self.root / "common", "fs_matched_b8", dtype_policy="int16")
        self.assertTrue(all(u["dtype"] == "<i2" for u in common["groups"][0]["units"]))
        for dtype in (np.int8, np.int16):
            with self.assertRaises(OverflowError):
                signed_cast(np.array([np.iinfo(dtype).max+1], dtype=np.int64), dtype)
        wide = SourceImage(original.metadata, (original.coefficients[0].astype(np.int32),), original.tables)
        wide.coefficients[0].flat[0] = 32768
        with self.assertRaises(OverflowError):
            build_store([wide], self.root / "overflow", "fs_matched_b8")

    def test_bounded_decoded_cache_hits_and_eviction(self):
        build_store([image(0, ((1, 1),)), image(1, ((1, 1),))], self.root / "ic", "ic_matched_shared")
        with StoreReader(self.root / "ic", cache_bytes=384) as reader:
            def read(identifier):
                return reader.read(reader.plan([ReadRequest(identifier, (tuple(range(64)),))]))[1]
            first, second = read("0"), read("0")
            self.assertEqual(first.decoded_q_values, 64)
            self.assertEqual(second.decoded_q_values, 0)
            self.assertEqual(second.application_read_bytes, 0)
            third = read("1")
            self.assertEqual(third.cache_evictions, 1)
            self.assertEqual(read("0").decoded_q_values, 64)

    def test_fixedk_requires_evidence_and_cannot_fabricate_dense_source(self):
        with self.assertRaises(ValueError):
            build_store([image(0)], self.root / "invalid", "ic_matched_fixedk", source_frequencies=[[0]]*3)
        build_store([image(0)], self.root / "fixed", "ic_matched_fixedk", source_frequencies=[[0]]*3, dependency_evidence="frequency-only test")
        with StoreReader(self.root / "fixed") as reader:
            with self.assertRaises(ValueError):
                reader.plan([ReadRequest("0", ((1,),)*3)])
            views, _ = reader.read(reader.plan([ReadRequest("0", ((0,),)*3)]))
            with self.assertRaises(ValueError):
                views[0].dense()

    def test_corrupt_member_fails_without_fallback(self):
        built = build_store([image(0)], self.root / "ic", "ic_matched_shared")
        path = self.root / "ic" / built["groups"][0]["units"][0]["file"]
        path.write_bytes(path.read_bytes()[:32])
        with StoreReader(self.root / "ic") as reader:
            with self.assertRaisesRegex(ValueError, "payload content changed"):
                reader.read(reader.plan([ReadRequest("0", (tuple(range(64)),)*3)]))

    def test_native_and_matched_cannot_be_mixed(self):
        with self.assertRaises(ValueError):
            build_store([image(0)], self.root / "native", "fs_native")


class NativeTests(TemporaryTest):
    def test_normalization_uses_train_only_and_includes_structural_zeros(self):
        value = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
        stats = fit_statistics([value, value+1], split="train")
        normalized = [normalize_mosaic(v, stats) for v in (value, value+1)]
        merged = np.stack(normalized).transpose(1, 0, 2, 3).reshape(3, -1)
        np.testing.assert_allclose(merged.mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(merged.std(axis=1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            fit_statistics([value], split="val")
        with self.assertRaises(ValueError):
            fit_statistics([np.zeros((3, 2, 2))], split="train")

    def test_mosaic_shape_structural_zeros_and_b8_triangle(self):
        coefficients = np.ones((3, 8, 8, 32, 32), dtype=np.int32)
        result = mosaic(coefficients, 3)
        self.assertEqual(result.shape, (3, 24, 24))
        self.assertEqual(np.count_nonzero(result), 3*64*6)
        expected = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
        np.testing.assert_array_equal(result[0, :3, :3], expected)
        self.assertEqual(np.count_nonzero(mosaic(np.ones((3, 1, 1, 8, 8)), 8)), 3*36)

    def test_dct_against_independent_scalar_formula_and_sampling(self):
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        spec = NativeSpec(8, 8, (2, 2), 20)
        actual = quantize_rgb(rgb, spec)[0, 0, 0]
        pixels = gray[::2, ::2].astype(np.float64)-128
        expected = np.empty((8, 8))
        for u in range(8):
            for v in range(8):
                total = sum(pixels[y, x]*math.cos(math.pi*(2*y+1)*u/16)*math.cos(math.pi*(2*x+1)*v/16)
                            for y in range(8) for x in range(8))
                expected[u, v] = np.rint(total*(1/math.sqrt(8) if u == 0 else .5)*(1/math.sqrt(8) if v == 0 else .5)/20)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(candidates()), 6048)

    def test_b32_ic_icb_fs_same_native_tensor_and_63_columns(self):
        rng = np.random.default_rng(4)
        rgbs = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        spec = NativeSpec(32, 3)
        expected = [encode_ic(rgb, spec) for rgb in rgbs]
        for i, rgb in enumerate(rgbs):
            write_ic_archive(self.root / f"image{i}.npz", [rgb], spec)
            np.testing.assert_array_equal(read_ic_archive(self.root / f"image{i}.npz", spec)[0], expected[i])
        write_ic_archive(self.root / "batch.npz", rgbs, spec)
        for actual, target in zip(read_ic_archive(self.root / "batch.npz", spec), expected):
            np.testing.assert_array_equal(actual, target)
        built = build_store([fs_source(rgb, i) for i, rgb in enumerate(rgbs)], self.root / "fs", "fs_native", group_size=2)
        self.assertEqual(len(built["groups"][0]["units"]), 63)
        with StoreReader(self.root / "fs") as reader:
            requests = [ReadRequest(str(i), (tuple(range(1024)),)*3) for i in range(3)]
            views, _ = reader.read(reader.plan(requests))
            for view, target in zip(views, expected):
                np.testing.assert_array_equal(mosaic(np.stack(view.dense()), 3), target)

    def test_diagonals_partition_every_frequency_once(self):
        for b in (8, 32):
            groups = [diagonal_frequencies(b, d) for d in range(2*b-1)]
            self.assertEqual(sorted(f for g in groups for f in g), list(range(b*b)))
            for d, group in enumerate(groups):
                self.assertTrue(all(f//b+f % b == d for f in group))
        self.assertEqual(mask_family([0, 1, 8], 8), "native_triangle_k2")
        self.assertEqual(mask_family([0, 1], 8), "matched_arbitrary_adapter")
        self.assertEqual(mask_family(list(range(64)), 8), "matched_arbitrary_adapter")


class TrainingTests(unittest.TestCase):
    def test_full_backward_updates_and_rng_restoration(self):
        torch.manual_seed(42)
        net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 3))
        batch = [(torch.randn(2, 4),)]
        labels = [torch.tensor([0, 2])]
        state = torch.get_rng_state().clone()
        result = check_updates(net, batch, batch, labels, lambda p: torch.optim.SGD(p, lr=.01, momentum=.9), 10, 0, 0)
        self.assertTrue(result["loss_gradients_and_parameters_match"])
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        with self.assertRaises(AssertionError):
            check_updates(net, [(batch[0][0]+1,)], batch, labels, lambda p: torch.optim.SGD(p, lr=.01), 1, 0, 0)


@unittest.skipUnless((Path.home() / "RGB-no-more/utils/dct_ops.py").exists(), "upstream DCT resize source unavailable")
class MaterializationTests(TemporaryTest):
    def test_aligned_edge_crops_quantization_saturation_and_flip_match_dense_y_oracle(self):
        from fastlens.benchmarks.baselines.dct_baselines.materialize import Materializer
        upstream = Path.home() / "RGB-no-more"
        profile = dict(version=1, representation="jpeg_b8", id="test", model="resnet24", transform="cnn_source512",
                       source_frequencies=[list(range(64))]*3, dependency_derivation="full source",
                       upstream=str(upstream), output_grids=[56]*3, output_channels=[[0, 1], [0, 8]], mean=[.5, -1], std=[2., 4.])
        rng = np.random.default_rng(27)
        coefficients = (rng.integers(-20000, 20000, (64, 64, 8, 8), dtype=np.int16),
                        np.zeros((32, 32, 8, 8), np.int16), np.zeros((32, 32, 8, 8), np.int16))
        tables = (np.full((8, 8), 3, np.uint16), np.ones((8, 8), np.uint16), np.ones((8, 8), np.uint16))
        adapter = Materializer(profile)
        for crop, flip in (([0, 0, 448, 448], False), ([32, 32, 448, 448], True), ([64, 64, 448, 448], True)):
            x, y, w, h = crop
            raw = coefficients[0][y//8:(y+h)//8, x//8:(x+w)//8].astype(np.float32)*tables[0]
            raw = raw.clip(-32768, 32767).reshape(56, 56, 64)
            planes = raw[..., [1, 8]].transpose(2, 0, 1)
            if flip:
                planes = planes[:, :, ::-1]*np.array([-1., 1.], dtype=np.float32)[:, None, None]
            expected = (planes-np.array([.5, -1.], np.float32)[:, None, None])/np.array([2., 4.], np.float32)[:, None, None]
            actual, = adapter(coefficients, tables, dict(crop_box=crop, flip=flip))
            np.testing.assert_array_equal(actual.numpy(), expected)

    @unittest.skipUnless(importlib.util.find_spec("dct_manip"), "libjpeg extension unavailable")
    def test_transformer_rounding_and_separate_tensor_shapes_match_upstream_test_transforms(self):
        from PIL import Image
        from fastlens.benchmarks.baselines.dct_baselines.materialize import Materializer, source_reference
        from fastlens.benchmarks.baselines.dct_baselines.source import read_jpeg
        from fastlens.benchmarks.baselines.system_rgbnomore.inference.model_factory import VITTI_MODEL_ID, SWINV2_T_MODEL_ID
        path = self.root / "source.jpg"
        rng = np.random.default_rng(11)
        Image.fromarray(rng.integers(0, 256, (512, 512, 3), np.uint8)).save(path, quality=82, subsampling=2)
        source = read_jpeg(path, "source", 0, "test")
        for model, model_id, crop, grids in (("vitti", VITTI_MODEL_ID, [32, 32, 448, 448], [28, 14, 14]),
                                              ("swinv2", SWINV2_T_MODEL_ID, [0, 0, 512, 512], [32, 16, 16])):
            profile = dict(version=1, representation="jpeg_b8", id=model, model=model, model_id=model_id,
                           transform="rgbnomore_source512", source_frequencies=[list(range(64))]*3,
                           dependency_derivation="full source", upstream=str(Path.home() / "RGB-no-more"),
                           output_grids=grids, default_crop=crop)
            row = dict(crop_box=crop, flip=False)
            actual = Materializer(profile)(source.coefficients, source.tables, row)
            expected = source_reference(profile, source, row)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class ReplayTests(TemporaryTest):
    def test_real_io_replay_preserves_tail_counts_and_schema_fields(self):
        import time
        from fastlens.benchmarks.baselines.dct_baselines import runner
        images = [image(i, ((1, 1),)) for i in range(3)]
        store_root = self.root / "ic"
        build_store(images, store_root, "ic_matched_batched", group_size=2)
        profile = dict(source_frequencies=[list(range(64))], tensors=[dict(shape=[1, 1, 1])],
                       model="fixture", id="fixture", output_channels=[[0, 0]], checkpoint=artifact(store_root / "store.json"))
        config = dict(model_microbatch_size=2, preparation_pool_size=2, workers=2, torch_threads=1,
                      source_manifest=artifact(store_root / "store.json"), model_profile=artifact(store_root / "store.json"),
                      request_trace=artifact(store_root / "store.json"), label_mapping=artifact(store_root / "store.json"),
                      precision="fp32", decode_budget_bytes=1024**2, reader_memory_budget_bytes=2*1024**2,
                      host_input_budget_bytes=4*1024**2,
                      repository_revision="test")
        config_path = self.root / "config.json"
        write_json(config_path, config)
        with StoreReader(store_root) as reader:
            identity = reader.identity
        evidence = dict(status="passed", resolved_contract=artifact(config_path), store=identity,
                        implementation=implementation_artifacts(), device="cpu", logits_match=True,
                        source_q_exact=True, quantization_tables_exact=True, dependency_closure_verified=True,
                        final_tensor_match=True, training_update_match=False, max_abs_tensor_error=0., max_relative_tensor_error=0.)
        from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
        evidence["loaded_model_state_sha256"] = tensor_state_sha256({})
        evidence_path = self.root / "evidence.json"
        write_json(evidence_path, evidence)
        trace = [dict(image_id=str(i), target_label=i) for i in range(3)]
        class Transform:
            def __init__(self, _):
                pass

            def source_coefficients(self, view):
                return view.dense()

            def __call__(self, q, tables, row, *, cropped=False, source_selected=False):
                time.sleep(.01)  # startup preparation must be inside the timer
                return (torch.as_tensor(q[0][0, 0, :1, :1]).float(),)

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.reshape(len(x), 1).expand(-1, 5)

        source = dict(images=[dict(image_id=str(i), path=str(config_path), jpeg_sha256=artifact(config_path)["sha256"])
                              for i in range(3)])
        with patch.object(runner, "inputs", return_value=(config, source, profile, {}, trace)), patch.object(
            runner, "Materializer", Transform
        ), patch.object(runner, "build_model", return_value=Model()), patch.object(
            runner, "request", side_effect=lambda r, p: ReadRequest(r["image_id"], (tuple(range(64)),))):
            row, detail = benchmark(config_path, store_root, evidence_path, run_kind="inference")
        self.assertEqual(row["counts"]["sample_occurrences"], 3)
        self.assertEqual(row["counts"]["unique_images"], 3)
        self.assertEqual(row["counts"]["physically_decoded_q_values"], 3*64)
        self.assertEqual(row["metrics"]["h2d"]["total_bytes"], 0)
        self.assertGreaterEqual(row["metrics"]["elapsed_wall_s"], .02)
        self.assertFalse(row["headline_eligible"])
        self.assertEqual(row["quality"]["sample_count"], 3)
        self.assertEqual(detail["counters"]["application_read_bytes"],
                         row["metrics"]["io"]["application_read_bytes"])
        self.assertGreaterEqual(row["metrics"]["io"]["physical_storage_read_bytes"], 0)


@unittest.skipUnless(importlib.util.find_spec("dct_manip"), "existing libjpeg coefficient extension unavailable")
class JpegTests(TemporaryTest):
    def test_original_q_qtable_and_odd_geometry_across_supported_sampling(self):
        import dct_manip
        from PIL import Image
        from fastlens.benchmarks.baselines.dct_baselines.source import read_jpeg
        rng = np.random.default_rng(3)
        for mode, sampling in (("RGB", 0), ("RGB", 2), ("L", 0)):
            with self.subTest(mode=mode, sampling=sampling):
                shape = (19, 21, 3) if mode == "RGB" else (19, 21)
                path = self.root / f"{mode}{sampling}.jpg"
                Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(path, quality=82, subsampling=sampling)
                source = read_jpeg(path, "jpeg", 0, "test")
                _, quant, y, chroma = dct_manip.read_coefficients(str(path))
                q = (y[0],) if chroma is None else (y[0], chroma[0], chroma[1])
                for actual, expected, table, qt in zip(source.coefficients, q, source.tables, quant):
                    np.testing.assert_array_equal(actual, expected.numpy())
                    np.testing.assert_array_equal(table, qt.numpy())
                for method in ("ic_matched_shared", "fs_matched_b8"):
                    root = self.root / f"{mode}{sampling}{method}"
                    build_store([source], root, method)
                    with StoreReader(root) as reader:
                        views, _ = reader.read(reader.plan([ReadRequest("jpeg", (tuple(range(64)),)*len(q))]))
                        for actual, expected in zip(views[0].dense(), q):
                            np.testing.assert_array_equal(actual, expected.numpy())


if __name__ == "__main__":
    unittest.main()
