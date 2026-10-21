"""Regressions for the observed round-2 evidence, resource and timing failures."""
import copy
import json
from pathlib import Path
import time
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

from fastlens.benchmarks.baselines.dct_baselines.contracts import ReadRequest, artifact, load_trace, validate_batches, write_json
from fastlens.benchmarks.baselines.dct_baselines.materialize import Materializer, operation_order
from fastlens.benchmarks.baselines.dct_baselines.runner import benchmark, report, SCHEMA
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store
from fastlens.benchmarks.baselines.dct_baselines.training import finish_batch, freeze_trace, oracle_finish_batch, upstream_ops, updates_per_epoch, update_windows
from .test_baselines import TemporaryTest, image
from . import test_baselines as fixtures
from . import test_training as training_fixtures
from .test_training import UPSTREAM


class ScheduleTests(unittest.TestCase):
    def test_population_pool_accumulation_and_retained_tail_counts(self):
        for population, batch, pool, accumulation, expected in ((8,2,2,4,4), (8,2,8,4,1),
                                                              (100,4,12,4,9), (1281167,64,4096,16,1252), (17,4,16,4,2)):
            self.assertEqual(updates_per_epoch(population, batch, pool, accumulation), expected)
            if population <= 100:
                rows = [dict(epoch=0, microbatch_id=i//batch, closed_pool_id=f"0:{i//pool}") for i in range(population)]
                self.assertEqual(len(update_windows(rows, accumulation)), expected)

    def test_both_traces_reject_reused_microbatch_ids_and_wrong_pool(self):
        profile = training_fixtures.FrozenAugmentationTests().profile()
        rows = freeze_trace([dict(image_id=str(i)) for i in range(3)], profile, seed=3, epochs=2,
                            batch_size=2, pool_size=2, labels={str(i): i for i in range(3)}, partition="calibration")
        config = dict(model_microbatch_size=2, preparation_pool_size=2)
        validate_batches(rows, config, training=True)
        changed = copy.deepcopy(rows)
        changed[3]["microbatch_id"] = changed[4]["microbatch_id"] = 0
        with self.assertRaisesRegex(ValueError, "microbatch assignment"):
            validate_batches(changed, config, training=True)
        changed = copy.deepcopy(rows)
        changed[2]["closed_pool_id"] = "0:0"
        with self.assertRaisesRegex(ValueError, "pool boundaries"):
            validate_batches(changed, config, training=True)


class StorageRegressions(TemporaryTest):
    def test_legal_q_or_table_rewrite_rejects_the_old_content_identity(self):
        for member in ("q", "tables"):
            root = self.root/member
            built = build_store([image(0)], root, "ic_matched_shared")
            path = root/built["groups"][0]["units"][0]["file"]
            with StoreReader(root) as reader:
                identity = reader.identity
                reader.verify_content()
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays[member].flat[0] += 1
            np.savez_compressed(path, **arrays)
            with StoreReader(root) as reader:
                self.assertEqual(reader.identity, identity)
                with self.assertRaisesRegex(ValueError, "payload content changed"):
                    reader.verify_content()
                with self.assertRaisesRegex(ValueError, "payload content changed"):
                    reader.read(reader.plan([ReadRequest("0", (tuple(range(64)),)*3)]))

    def test_cross_group_decode_lifetimes_are_bounded_without_cache(self):
        root = self.root/"ic"
        built = build_store([image(i) for i in range(32)], root, "ic_matched_batched", group_size=4)
        member = next(iter(built["files"].values()))["decoded_bytes"]
        references, peak = [], [0]
        original_read, opened = Path.read_bytes, []
        def read_bytes(path):
            opened.append(path)
            return original_read(path)
        with patch.object(Path, "read_bytes", read_bytes):
            reader = StoreReader(root, workers=4, cache_bytes=0, decode_budget_bytes=2*member,
                                 memory_budget_bytes=3*member)
        self.assertEqual(len(opened), reader.initialization_counters.file_opens)
        self.assertEqual(reader.initialization_counters.application_read_bytes, opened[0].stat().st_size)
        with reader:
            decode = reader._decode_file
            def tracked(name):
                arrays, counters = decode(name)
                references.extend(weakref.ref(a) for a in arrays.values())
                peak[0] = max(peak[0], sum(a.nbytes for r in references if (a := r()) is not None))
                return arrays, counters
            reader._decode_file = tracked
            requests = [ReadRequest(str(i), ((0,),)*3, ((0,0,1,1),)*3) for i in range(0,32,4)]
            views, counts = reader.read(reader.plan(requests))
            self.assertLessEqual(peak[0], 2*member)
            self.assertLessEqual(counts.active_decoded_peak_bytes, 2*member)
            self.assertLessEqual(counts.reader_memory_peak_bytes, 3*member)
            self.assertTrue(all(r() is None for r in references))
            self.assertEqual(counts.file_opens, 8)
            for i, view in zip(range(0,32,4), views):
                for actual, expected in zip(view.coefficients, image(i).coefficients):
                    np.testing.assert_array_equal(actual[...,0], expected[:1,:1,0,0])
            with self.assertRaisesRegex(MemoryError, "one whole NPZ"):
                with StoreReader(root, decode_budget_bytes=member-1) as too_small:
                    too_small.read(too_small.plan(requests))


class TraceLabels(TemporaryTest):
    def test_inference_label_mapping_range_type_and_sorted_remapping(self):
        profile = training_fixtures.FrozenAugmentationTests().profile()
        profile["training"] = False
        row = dict(occurrence_id="a", image_id="a", epoch=0, view_id=0, logical_slot=0, microbatch_id=0,
                   profile_id=profile["id"], crop_box=[0,0,512,512], flip=False, resize_geometry=profile["output_grids"],
                   augmentation_parameters={}, mixup_partner_occurrence=None, mixup_lambda=1., target_parameters={}, target_label=999)
        path = self.root/"trace.jsonl"
        path.write_text(json.dumps(row)+"\n")
        # A legitimate checkpoint remapping may differ from the source's label.
        load_trace(path, profile, {"a": {"label": 0}}, {"a": 999})
        for label in (-1, 1000, True, 0):
            row["target_label"] = label
            path.write_text(json.dumps(row)+"\n")
            with self.assertRaisesRegex(ValueError, "label mapping"):
                load_trace(path, profile, {"a": {"label": 0}}, {"a": 999})


class GoldenOrganization(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA normalization regression requires a host GPU")
    def test_cuda_normalization_and_mixup_match_independent_fp32_oracle(self):
        profile = training_fixtures.FrozenAugmentationTests().profile()
        profile.update(output_channels=[[2,8],[0,1],[1,0]],
                       mean=[-.004207899793982506,.014241999946534634,-.007471600081771612],
                       std=[.10321000218391418,2.777600049972534,.4395799934864044])
        values = (torch.arange(28*28*64, device="cuda", dtype=torch.int32)%2041-1024).reshape(28,28,8,8)
        raw = [tuple((values+i+c).clamp(-1024,1016).to(torch.int16) for c in range(3)) for i in range(3)]
        rows = [dict(occurrence_id=str(i), target_label=(2,7,2)[i], augmentation_parameters=[],
                     mixup_partner_occurrence=str((i-1)%3), mixup_lambda=.9778500199317932,
                     target_parameters=dict(lambda_rolled=.02214997261762619)) for i in range(3)]
        dops = upstream_ops(UPSTREAM)
        actual = finish_batch(raw, rows, profile, dops)
        expected = oracle_finish_batch(raw, rows, profile, dops)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_asymmetric_normalization_and_mixup_against_independent_targets(self):
        profile = training_fixtures.FrozenAugmentationTests().profile()
        profile.update(output_grids=[28]*3, output_channels=[[2,8],[0,1],[1,0]], mean=[.125,-.75,2.5], std=[.3,1.25,2.75])
        raw = [tuple(torch.full((28,28,8,8), float(i*9+c*3)) for c in range(3)) for i in range(3)]
        rows = freeze_trace([dict(image_id=str(i)) for i in range(3)], profile, seed=5, epochs=1,
                            batch_size=3, pool_size=3, labels={"0":2,"1":7,"2":2}, partition="golden")
        for row in rows:
            row["augmentation_parameters"] = [dict(operation="Brightness", operation_index=i, magnitude=0.) for i in range(2)]
            row["mixup_lambda"] = .75
            row["target_parameters"]["lambda_rolled"] = .25
        dops = upstream_ops(UPSTREAM)
        actual = finish_batch(raw, rows, profile, dops)
        expected = oracle_finish_batch(raw, rows, profile, dops)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        labels = [r["target_label"] for r in rows]
        golden = torch.zeros(3,1000)
        for i, label in enumerate(labels):
            golden[i,label] += .75
            golden[i,labels[(i-1)%3]] += .25
        torch.testing.assert_close(actual[1], golden, atol=0, rtol=0)

    def test_training_order_instrumentation_and_nonidentity_chroma_resize(self):
        from fastlens.benchmarks.baselines.dct_baselines import training
        from fastlens.benchmarks.baselines.dct_baselines.unified import cpu_raw
        from fastlens.benchmarks.baselines.dct_baselines.source import read_jpeg
        # Existing durable JPEG fixture: no filesystem writes in this check.
        selection = Path(__file__).resolve().parents[4]/"data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
        if not selection.exists():
            self.skipTest("local source512 JPEG fixture unavailable")
        source = json.loads(selection.read_text())["samples"][0]
        image_data = read_jpeg(source["path"], "real", 2, "val")
        dops = upstream_ops(UPSTREAM)
        for cnn in (True, False):
            profile = training_fixtures.FrozenAugmentationTests().profile(cnn)
            profile["upstream"] = str(UPSTREAM)
            profile["operation_order"] = operation_order(profile)
            row = dict(image_id="real", crop_box=[16,32,384,384], flip=True)
            raw = Materializer(profile).raw(image_data.coefficients, image_data.tables, row)
            expected = cpu_raw({"path": source["path"]}, row, profile, dops)
            torch.testing.assert_close(raw, expected, atol=0, rtol=0)
            self.assertTrue(any(torch.count_nonzero(v) for v in raw[1:]))
            rows = freeze_trace([dict(image_id="real")], profile, seed=5, epochs=1, batch_size=1,
                                pool_size=1, labels={"real":2}, partition="order")
            events = []
            augment, organize = training.augment, training.organize
            def observe_augmentation(*args, **kwargs):
                result = augment(*args, **kwargs)
                events.append("augmentation")
                self.assertTrue(all(v.dtype == torch.int16 for v in result))
                self.assertTrue(all(v.min() >= -1024 and v.max() <= 1016 for v in result))
                return result
            def observe_organization(*args):
                events.append("normalization")
                return organize(*args)
            with patch.object(training, "augment", side_effect=observe_augmentation), patch.object(training, "organize", side_effect=observe_organization):
                actual = finish_batch([raw], rows, profile, dops)
            self.assertEqual(events, ["augmentation", "normalization"])
            order = profile["operation_order"]
            normalize = "select_channels" if cnn else "organize_y_cbcr"
            self.assertLess(order.index("published_randaugment_frozen_clamp_after_each_operation"), order.index(normalize))
            torch.testing.assert_close(actual, oracle_finish_batch([expected], rows, profile, dops), atol=0, rtol=0)


class WindowAndReport(TemporaryTest):
    def replay(self):
        from . import test_baselines
        fixture = fixtures.ReplayTests()
        fixture.root = self.root
        results = []
        def capture(*args, **kwargs):
            result = benchmark(*args, **kwargs)
            results.append(result)
            return result
        with patch.object(test_baselines, "benchmark", side_effect=capture):
            fixture.test_real_io_replay_preserves_tail_counts_and_schema_fields()
        return results[0]

    def test_cpu_window_excludes_initialization_and_capacity_accounting(self):
        from fastlens.benchmarks.baselines.dct_baselines import runner
        clock = [0.]
        initialize, capacity, pool = StoreReader.__init__, runner.storage_bytes, runner.Replay.pool
        def initialize_with_outside_work(*args, **kwargs):
            clock[0] += .025
            return initialize(*args, **kwargs)
        def capacity_with_outside_work(*args):
            clock[0] += .075
            return capacity(*args)
        def pool_with_inside_work(*args):
            clock[0] += .05
            return pool(*args)
        with patch.object(StoreReader, "__init__", initialize_with_outside_work), patch.object(runner, "storage_bytes", capacity_with_outside_work), \
                patch.object(runner.Replay, "pool", pool_with_inside_work), patch.object(time, "process_time", side_effect=lambda: clock[0]):
            row, detail = self.replay()
        self.assertAlmostEqual(detail["cpu_core_seconds"], .1)
        self.assertAlmostEqual(row["metrics"]["cpu_core_seconds"], .1)
        self.assertGreater(clock[0], .1)

    def test_report_groups_by_implementation_store_model_and_device(self):
        import jsonschema
        row, _ = self.replay()
        jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).validate(row)
        rows = []
        for index, throughput in enumerate((100.,120.,900.,300.,800.)):
            r = copy.deepcopy(row)
            r["run_id"] = f"fixture-{index}"
            r["metrics"]["throughput_images_s"] = throughput
            if index == 2:
                r["fingerprints"]["implementation"] = "a"*64
                r["fingerprints"]["store"] = "b"*64
                r["artifacts"]["store"]["content_fingerprint"] = "b"*64
            if index == 3:
                r["resolved_parameters"]["model"] = "other-fixture"
                r["environment"]["hardware"]["device"] = "another-test-device"
            if index == 4:
                r["experiment_type"] = "common_processing_ablation"
            rows.append(r)
        path = self.root/"raw.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        summary = report([path], self.root/"summary.json")
        self.assertEqual(len(summary), 4)
        self.assertEqual([r["repeats"] for r in summary], [2,1,1,1])
        self.assertEqual(summary[0]["median_images_s"], 110.)
        self.assertEqual({r["model"] for r in summary}, {"fixture", "other-fixture"})
        self.assertEqual(sum(len(r["run_ids"]) for r in summary), 5)
        self.assertEqual(summary[-1]["experiment_type"], "common_processing_ablation")

    def test_report_rejects_a_completed_run_with_invalid_resource_observations(self):
        row, _ = self.replay()
        path = self.root/"run.jsonl"
        path.write_text(json.dumps(row)+"\n")
        path.with_suffix(".process.json").write_text(json.dumps(dict(valid=False, returncode=0, foreign_gpu_pids=[123])))
        with self.assertRaisesRegex(ValueError, "invalid resource/process"):
            report([path], self.root/"summary.json")
