import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from galp.benchmarks.dct_baselines.contracts import artifact, load_trace, write_json
from galp.benchmarks.dct_baselines.gates import Comparisons, paired_updates, verify_training
from galp.benchmarks.dct_baselines.runner import inputs
from galp.benchmarks.dct_baselines.source import freeze_source, source_images
from galp.benchmarks.dct_baselines.store import build_store
from galp.benchmarks.dct_baselines.training import (
    augment, finish_batch, freeze_trace, make_optimizer, snapshot, update_windows, upstream_ops,
)
from galp.benchmarks.dct_baselines.unified import UnifiedReplay
from galp.benchmarks.system_rgbnomore.training.model_factory import capture_rng_state, rng_state_sha256
from galp.benchmarks.training_pls.recipe import DCT_RANDAUGMENT_OPERATIONS
from galp.benchmarks.training_pls.published_augmentation import _magnitude
from .test_baselines import TemporaryTest

UPSTREAM = Path.home()/"RGB-no-more"
REPO = Path(__file__).resolve().parents[4]
EXACT = {stage: dict(atol=0., rtol=0.) for stage in ("inputs", "model", "state")}


class AccumulatedTrainingTests(unittest.TestCase):
    def test_ten_adamw_updates_restore_rng_and_keep_partial_windows(self):
        torch.manual_seed(13)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 3))
        recipe = dict(total_updates=20, warmup_updates=2, accumulation=2, precision="fp32")
        state = snapshot(model, recipe)
        before = rng_state_sha256(capture_rng_state())
        rows = [dict(microbatch_id=i, epoch=i//2, closed_pool_id=str(i//2)) for i in range(20)]
        batches = {i: ((torch.randn(1 if i % 2 else 2, 4),), torch.tensor([0] if i % 2 else [0, 2])) for i in range(20)}
        windows = update_windows(rows, 2)
        before = rng_state_sha256(capture_rng_state())
        compare = Comparisons(EXACT)
        results = paired_updates(model, batches, batches, windows, state, recipe, compare)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r["samples"] == 3 for r in results))
        self.assertEqual(results[-1]["scheduler"]["completed_updates"], 10)
        self.assertEqual(rng_state_sha256(capture_rng_state()), before)
        self.assertEqual(set(compare.maximum), {"logits", "loss", "gradients", "parameters", "optimizer"})
        altered = dict(batches)
        altered[0] = ((batches[0][0][0]+1,), batches[0][1])
        with self.assertRaises(AssertionError):
            paired_updates(copy.deepcopy(model), altered, batches, windows[:1], state, recipe, Comparisons(EXACT))

    def test_accumulation_weight_is_actual_samples_and_closes_at_pool_boundary(self):
        torch.manual_seed(9)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        reference = copy.deepcopy(model)
        recipe = dict(total_updates=20, warmup_updates=2, precision="fp32")
        x, targets = torch.randn(3, 4), torch.tensor([0, 1, 2])
        from galp.benchmarks.dct_baselines.training import train_window
        opt = make_optimizer(model, recipe)
        opt_ref = make_optimizer(reference, recipe)
        train_window(model, *opt, [((x[:2],), targets[:2]), ((x[2:],), targets[2:])], recipe, lambda *_: None)
        train_window(reference, *opt_ref, [((x,), targets)], recipe, lambda *_: None)
        torch.testing.assert_close(model[0].weight, reference[0].weight, atol=1e-7, rtol=1e-6)
        rows = [dict(epoch=0, microbatch_id=i, closed_pool_id=i//2) for i in range(5)]
        self.assertEqual([len(w) for w in update_windows(rows, 16)], [2, 2, 1])


@unittest.skipUnless((UPSTREAM/"utils/dct_ops.py").exists(), "upstream unavailable")
class FrozenAugmentationTests(unittest.TestCase):
    def profile(self, cnn=True):
        return dict(version=1, representation="jpeg_b8", id="test", training=True,
                    transform="cnn_source512" if cnn else "rgbnomore_source512",
                    output_grids=[56]*3 if cnn else [28, 14, 14], source_frequencies=[list(range(64))]*3,
                    dependency_derivation="full source", output_channels=[[0, 1], [1, 8], [2, 9]],
                    mean=[0.]*3, std=[1.]*3)

    def test_every_frozen_operation_against_scalar_upstream_and_no_rng_consumption(self):
        dops = upstream_ops(UPSTREAM)
        torch.manual_seed(12)
        for cnn in (True, False):
            profile = self.profile(cnn)
            raw = [torch.randint(-1000, 1000, (g, g, 8, 8)).float() for g in profile["output_grids"]]
            for name in DCT_RANDAUGMENT_OPERATIONS:
                magnitude, signed = _magnitude(name)
                for sign in ((1, -1) if signed else (1,)):
                    op = dict(operation=name, magnitude=magnitude*sign, operation_index=0)
                    if name == "Cutout":
                        op.update(centers=[[g//2, g//2] for g in profile["output_grids"]], pads=[2*g//28 for g in profile["output_grids"]])
                    elif name.startswith("Translate"):
                        import math
                        blocks = math.floor(op["magnitude"]/2)*2
                        op["blocks"] = [blocks*g//28 for g in profile["output_grids"]]
                    elif name == "ChromaDrop":
                        op["channel"] = 1
                    before = rng_state_sha256(capture_rng_state())
                    actual = augment(raw, [op], dops)
                    expected = augment(raw, [op], dops, oracle=True)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=f"{name}, cnn={cnn}, sign={sign}")
                    self.assertEqual(rng_state_sha256(capture_rng_state()), before)

    def test_integer_augmentation_keeps_float_normalization_and_explicit_mixup_targets(self):
        from types import SimpleNamespace
        profile = self.profile()
        profile.update(output_channels=[[0, 0]], mean=[.5], std=[.25])
        rows = [dict(occurrence_id=str(i), target_label=i, augmentation_parameters=[],
                     mixup_partner_occurrence=str(1-i), mixup_lambda=.75,
                     target_parameters=dict(lambda_rolled=.25)) for i in range(2)]
        raw = [tuple(torch.full((2, 2, 8, 8), value, dtype=torch.int16) for _ in range(3)) for value in (1, 2)]
        (mixed,), targets = finish_batch(raw, rows, profile, SimpleNamespace())
        torch.testing.assert_close(mixed[:, 0, 0, 0], torch.tensor([3., 5.]), atol=0, rtol=0)
        torch.testing.assert_close(targets[:, :2], torch.tensor([[.75, .25], [.25, .75]]), atol=0, rtol=0)


class FrozenTraceTests(TemporaryTest):
    def test_frozen_trace_preserves_epochs_duplicates_and_tail_mixup(self):
        profile = FrozenAugmentationTests().profile()
        images = [dict(image_id=str(i)) for i in range(3)]
        options = dict(seed=19, epochs=2, batch_size=2, pool_size=2, labels={str(i): i for i in range(3)}, partition="verify")
        rows = freeze_trace(images, profile, **options)
        self.assertEqual(rows, freeze_trace(images, profile, **options))
        path = self.root/"trace.jsonl"
        def write():
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        write()
        self.assertEqual(len(load_trace(path, profile, {str(i): {} for i in range(3)}, options["labels"])), 6)
        self.assertEqual([len(w) for w in update_windows(rows, 16)], [2, 1, 2, 1])
        tail = rows[2]
        self.assertEqual(tail["mixup_partner_occurrence"], tail["occurrence_id"])
        tail["mixup_partner_occurrence"] = rows[0]["occurrence_id"]
        write()
        with self.assertRaisesRegex(ValueError, "Mixup partner"):
            load_trace(path, profile, {str(i): {} for i in range(3)}, options["labels"])


@unittest.skipUnless((UPSTREAM/"utils/dct_ops.py").exists(), "upstream unavailable")
class UnifiedTrainingTests(TemporaryTest):
    def test_real_jpeg_ic_fs_cpu_trace_and_calibration_gates(self):
        from PIL import Image
        rng = np.random.default_rng(31)
        entries = []
        for i in range(3):
            path = self.root/f"{i}.jpg"
            Image.fromarray(rng.integers(0, 256, (512, 512, 3), np.uint8)).save(path, quality=82, subsampling=2)
            entries.append(dict(logical_sample_id=f"train/c{i}/{i}", label=i, path=str(path)))
        selection = self.root/"selection.json"
        write_json(selection, dict(split="train", samples=entries))
        source = freeze_source(selection, self.root/"source.json", dict(resize="synthetic512", encoder="Pillow", quality=82, subsampling=2))
        # This integration fixture tests storage/trace/gates with Tiny below.
        # Real DCTNet architectures are checked in separate processes, because
        # their top-level datasets/models packages collide with RGB-no-more.
        checkpoint = self.root/"fixture_weights.pt"
        torch.save({"fixture": torch.tensor([1.])}, checkpoint)
        profile = dict(version=1, representation="jpeg_b8", id="fixture24", model="resnet24", training=True,
                       transform="cnn_source512", upstream=str(UPSTREAM), output_grids=[56]*3,
                       source_frequencies=[list(range(64))]*3, dependency_derivation="full source",
                       output_channels=[[0, f] for f in range(16)]+[[c, f] for c in (1, 2) for f in (0, 1, 8, 9)],
                       mean=[.5]*24, std=[.25]*24, checkpoint=artifact(checkpoint),
                       resize_implementation=artifact(UPSTREAM/"utils/dct_ops.py"),
                       model_adapter=artifact(__file__), dependency_evidence=artifact(__file__))
        profile["label_policy"] = "source_label"
        from galp.benchmarks.dct_baselines.materialize import operation_order
        profile["operation_order"] = operation_order(profile)
        write_json(self.root/"profile.json", profile)
        labels = {r["image_id"]: r["label"] for r in source["images"]}
        write_json(self.root/"labels.json", dict(version=1, policy="source_label", classes={}, labels=labels))
        for name, images in (("calibration_trace", source["images"][:1]), ("trace", source["images"][1:])):
            rows = freeze_trace(images, profile, seed=10, epochs=10, batch_size=2, pool_size=2, labels=labels, partition=name)
            (self.root/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        recipe = dict(precision="fp32", accumulation=16, total_updates=20, warmup_updates=2,
                      population_size=3, updates_per_epoch=2, total_updates_override=True, schedule_epochs=300)
        config = dict(version=1, source_manifest=artifact(self.root/"source.json"), model_profile=artifact(self.root/"profile.json"),
                      request_trace=artifact(self.root/"trace.jsonl"), calibration_trace=artifact(self.root/"calibration_trace.jsonl"),
                      label_mapping=artifact(self.root/"labels.json"), decode_budget_bytes=16*1024**2,
                      reader_memory_budget_bytes=32*1024**2,
                      host_input_budget_bytes=64*1024**2,
                      model_microbatch_size=2, preparation_pool_size=2, workers=1, torch_threads=1,
                      precision="fp32", tail_policy="keep", training=recipe, tolerance=dict(atol=0., rtol=0.))
        config_path = self.root/"matched.json"
        write_json(config_path, config)
        config, source, profile, _, trace = inputs(config_path)
        schedule = copy.deepcopy(config)
        schedule["training"]["total_updates_override"] = False
        schedule_path = self.root/"schedule.json"
        write_json(schedule_path, schedule)
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            inputs(schedule_path)
        schedule["training"]["total_updates"] = 600
        write_json(schedule_path, schedule)
        inputs(schedule_path)

        inference_profile = dict(profile, training=False)
        inference_profile["operation_order"] = operation_order(inference_profile)
        write_json(self.root/"inference-profile.json", inference_profile)
        inference_rows = [dict(row, augmentation_parameters={}, mixup_partner_occurrence=None,
                               mixup_lambda=1., target_parameters={}) for row in trace]
        (self.root/"inference-trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in inference_rows))
        inference = dict(config, model_profile=artifact(self.root/"inference-profile.json"),
                         request_trace=artifact(self.root/"inference-trace.jsonl"), precision="bf16-autocast")
        write_json(self.root/"inference.json", inference)
        with self.assertRaisesRegex(ValueError, "supports fp32 only"):
            inputs(self.root/"inference.json")
        inference["precision"] = "fp32"
        write_json(self.root/"inference.json", inference)
        inputs(self.root/"inference.json")
        stores = []
        for kind, method in (("ic", "ic_matched_shared"), ("ic", "ic_matched_batched"), ("fs", "fs_matched_b8")):
            root = self.root/("icb" if method == "ic_matched_batched" else kind)
            build_store(source_images(source), root, method, source_artifact=config["source_manifest"])
            stores.append((kind, root))
        for kind, root in [("cpu_dct", None), *stores]:
            with UnifiedReplay(kind, root, config, source, profile) as replay:
                replay.verify_source()
                actual, target = replay.batch(trace[:2])
                expected, expected_target = replay.batch(trace[:2], oracle=True)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                torch.testing.assert_close(target, expected_target, atol=0, rtol=0)
        # The established CNN adapter imports DCTNet's top-level packages.
        # A fresh process is its supported execution contract.
        script = '''
import json, sys, torch
from pathlib import Path
from unittest.mock import patch
from galp.benchmarks.dct_models.training_jpeg import JpegDctDataset, StoreDctDataset
from galp.benchmarks.dct_baselines.store import training_store_identity
channels = [(0,0),(0,1),(0,8),(1,0),(1,8),(2,0)]
for directory in sys.argv[1:]:
    root = Path(directory)
    images = [r['metadata'] for r in json.loads((root/'store.json').read_text())['images']]
    records = [(r['image_id'],r['label'],r['path']) for r in images]
    training_store_identity(root, records)
    for policy in ('per-sample','per-pls'):
        args = (records,[2,0,1],1,channels,56)
        jpeg = JpegDctDataset(*args,crop_policy=policy)
        store = StoreDctDataset(*args,crop_policy=policy,store=root)
        for index in range(3):
            expected, actual = jpeg[index], store[index]
            torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
            assert actual[1:4] == expected[1:4]
            assert actual[-2] > 0
        # PyTorch DataLoader invokes __getitems__ for a whole microbatch.
        # Repeated members of one IC-b/FS group must be read only once.
        with patch.object(store.reader, '_decode_file', wraps=store.reader._decode_file) as decode:
            items = store.__getitems__([2,0,2])
            expected = [jpeg[i] for i in (2,0,2)]
            for a,b in zip(items,expected):
                torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
                assert a[1:4] == b[1:4]
            names = [call.args[0] for call in decode.call_args_list]
            assert len(names) == len(set(names))
        store.reader.close()
'''
        completed = subprocess.run([sys.executable, "-B", "-c", script, *(str(root) for _, root in stores)],
                                   cwd=REPO, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        from galp.benchmarks.system_rgbnomore.training.pipeline import _RgbNoMoreDctDataset, TrainingSample
        from galp.benchmarks.training_pls.published_augmentation import published_training_augmentation
        plan = []
        for i, row in enumerate(source["images"]):
            sample = TrainingSample(logical_sample_id=row["image_id"], path=Path(row["path"]),
                                    label=row["label"], width=512, height=512, galp_image_id=i)
            decision = published_training_augmentation(training_seed=11997733, epoch=1,
                logical_sample_id=row["image_id"], virtual_pls_id=0, crop_policy="per-sample",
                source_width=512, source_height=512).decision
            plan.append((sample, None, decision))
        jpeg = _RgbNoMoreDctDataset(plan, [], UPSTREAM)
        for _, root in stores:
            store = _RgbNoMoreDctDataset(plan, [], UPSTREAM, baseline_store=root)
            for i in (2, 0, 2):
                expected, actual = jpeg[i], store[i]
                for a, b in zip(actual[:2], expected[:2]):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                self.assertEqual(actual[2:4], expected[2:4])
            with patch.object(store.store_reader, "_decode_file", wraps=store.store_reader._decode_file) as decode:
                items = store.__getitems__([2,0,2])
                for a, b in zip(items, [jpeg[i] for i in (2,0,2)]):
                    torch.testing.assert_close(a[:2], b[:2], atol=0, rtol=0)
                    self.assertEqual(a[2:4], b[2:4])
                names = [call.args[0] for call in decode.call_args_list]
                self.assertEqual(len(names), len(set(names)))
            store.store_reader.close()
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(torch.nn.Linear(24, 16), torch.nn.Dropout(.2), torch.nn.Linear(16, 1000))
            def forward(self, value):
                return self.layers(value.mean((2, 3)))
        model = Tiny()
        state = snapshot(model, recipe)
        from galp.benchmarks.system_rgbnomore.training.artifacts import tensor_state_sha256
        state.update(profile=config["model_profile"], recipe=recipe, initial_model_sha256=tensor_state_sha256(model.state_dict()))
        state_path = self.root/"initial.pt"
        torch.save(state, state_path)
        with patch("galp.benchmarks.dct_baselines.gates.build_training_model", side_effect=lambda *_: Tiny()):
            calibration = self.root/"calibration.json"
            result = verify_training(config_path, state_path, "fs", self.root/"fs", calibration, tolerance=EXACT)
            self.assertEqual(result["updates"], 10)
            result = verify_training(config_path, state_path, "fs", self.root/"fs", self.root/"verified.json", calibration_path=calibration)
            self.assertEqual(result["updates"], 10)
            self.assertTrue(all(error == 0 for error in result["max_abs_errors"].values()))
            changed = json.loads(calibration.read_text())
            changed["snapshot"]["sha256"] = "stale"
            stale_calibration = self.root/"stale-calibration.json"
            write_json(stale_calibration, changed)
            with self.assertRaisesRegex(ValueError, "stale"):
                verify_training(config_path, state_path, "fs", self.root/"fs", self.root/"bad.json", calibration_path=stale_calibration)


class CompatibilityReportTests(TemporaryTest):
    def test_existing_performance_report_preserves_ic_fs_names_and_metrics(self):
        from galp.benchmarks.system_rgbnomore.tests.test_equal_image_epoch_benchmark import EqualImageReportTest
        from galp.benchmarks.training_pls.contracts import (
            BACKEND_IMPLEMENTATIONS, EXECUTION_MODES, IC_DCT_BACKEND, FS_DCT_BACKEND, condition_identity_hash,
        )
        from galp.benchmarks.training_pls.recipe import sha256_json
        from galp.benchmarks.training_pls.report_equal_image_performance import build_report
        fixture = EqualImageReportTest()
        standard, native = fixture._fixture(self.root)
        reference = fixture._rgbnomore_fixture(self.root, native)
        for backend, method in ((IC_DCT_BACKEND, "ic_matched_shared"), (FS_DCT_BACKEND, "fs_matched_b8")):
            contract = json.loads((reference/"run_manifest.json").read_text())
            contract.pop("condition_hash")
            contract.pop("run_manifest_hash")
            contract.update(execution_backend=backend, execution_mode=EXECUTION_MODES[backend],
                            backend_implementation=BACKEND_IMPLEMENTATIONS[backend], headline_eligible=False)
            contract["standard_dct_reference"]["baseline_store"] = {"method": method}
            contract["condition_hash"] = condition_identity_hash(contract)
            contract["run_manifest_hash"] = sha256_json(contract)
            write_json(reference/"run_manifest.json", contract)
            for filename, field, value in (("environment.json", "execution_mode", EXECUTION_MODES[backend]),
                                           ("model_only_dct.json", "condition_hash", contract["condition_hash"])):
                record = json.loads((reference/filename).read_text())
                record[field] = value
                write_json(reference/filename, record)
            report, rows = build_report(standard_root=standard, native_run=native, rgbnomore_run=reference)
            warm = {r["pipeline"]: r for r in rows if r["epoch"] == 2}
            self.assertEqual(len(rows), 10)
            self.assertEqual(warm[method]["images_per_second"], 750.)
            self.assertEqual(warm[method]["input_domain"], "dct")
            self.assertNotIn("rgbnomore_dct", report["included_pipelines"])
            self.assertEqual(report["baseline_store"]["method"], method)
            self.assertFalse(report["headline_eligible"])


@unittest.skipUnless((REPO/"galp/data/compressed/imagenet512_val_block_major/manifest.bin").exists(), "local block-major source unavailable")
class NativeSourceTests(unittest.TestCase):
    def test_block_major_integer_q_tables_and_repeated_source_ids(self):
        from galp.benchmarks.dct_models.storage import Reader
        from galp.benchmarks.dct_baselines.source import read_jpeg
        selection = REPO/"galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
        rows = json.loads(selection.read_text())["samples"][:3]
        manifest = REPO/"galp/data/compressed/imagenet512_val_block_major/manifest.bin"
        with Reader(manifest, command="read-source-manifest") as reader:
            for index in (2, 0, 2):
                row = rows[index]
                image = read_jpeg(row["path"], row["logical_sample_id"], row["label"], "val")
                coefficients, tables = reader.read(row["galp_image_id"])
                for a, b, qa, qb in zip(coefficients, image.coefficients, tables, image.tables):
                    np.testing.assert_array_equal(a.reshape(*a.shape[:2], 8, 8), b)
                    np.testing.assert_array_equal(qa.reshape(8, 8), qb)
                info = reader.last_stats["metadata"]
                self.assertEqual([info["width"], info["height"]], [image.metadata["width"], image.metadata["height"]])
                for a, b in zip(info["components"], image.metadata["components"]):
                    self.assertEqual(a, {key: b[key] for key in a})


if __name__ == "__main__":
    unittest.main()
