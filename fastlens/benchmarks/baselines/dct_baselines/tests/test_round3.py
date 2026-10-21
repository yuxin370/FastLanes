"""Round-3 cache admission, matched pairing, and frozen native-search behavior."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from fastlens.benchmarks.baselines.dct_baselines.contracts import ReadRequest
from fastlens.benchmarks.baselines.dct_baselines.native_search import isolated_splits, subset96, validate_task
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store, fingerprint
from fastlens.benchmarks.dct_models.matched_training import check_pair, training_identity
from .test_baselines import TemporaryTest, image


class CacheAdmissionTests(TemporaryTest):
    def test_all_hit_request_fits_without_room_for_a_decode(self):
        root = self.root/"cache"
        manifest = build_store([image(0, ((1,1),)*3)], root, "ic_matched_shared")
        cache_bytes = sum(f["decoded_bytes"] for f in manifest["files"].values())
        request = ReadRequest("0", (tuple(range(64)),)*3)
        with StoreReader(root, cache_bytes=cache_bytes) as reader:
            plan = reader.plan([request])
            expected, cold = reader.read(plan)
            self.assertEqual(reader.cache_size, cache_bytes)
            reader.memory_budget_bytes = cache_bytes+cold.output_peak_bytes
            actual, warm = reader.read(plan)
            self.assertEqual(warm.file_opens, 0)
            self.assertEqual(warm.decoded_q_bytes, 0)
            self.assertEqual(warm.active_decoded_peak_bytes, 0)
            self.assertEqual(warm.decoded_cache_peak_bytes, cache_bytes)
            self.assertEqual(warm.reader_memory_peak_bytes, reader.memory_budget_bytes)
            for a, b in zip(actual[0].coefficients, expected[0].coefficients):
                np.testing.assert_array_equal(a, b)
            with self.assertRaises(MemoryError):
                reader.read(reader.plan([request, request]))
            reader.cache.clear()
            reader.cache_size = 0
            with self.assertRaises(MemoryError):
                reader.read(plan)


class MatchedPairTests(unittest.TestCase):
    def result(self):
        args = SimpleNamespace(mapping_sha256="mapping", condition="B6", no_compile=False,
                               compile_mode="default", segments_per_pool=4, epochs=2, max_pools=0,
                               data_only=False, resume=None, short_warmup=False)
        recipe = dict(training=dict(physical_microbatch=64, gradient_accumulation=16))
        identity = training_identity(args, profile="resnet24", channels=[[0,0]], grid=28,
            model_code=["fixture"],
            initial_model_hash="initial", recipe=recipe, source=Path(__file__),
            runtime=dict(device="H100", device_uuid="same", precision="bf16-autocast"))
        return dict(matched_identity=identity, matched_identity_sha256=fingerprint(identity), epochs=[
            dict(epoch=0, samples=4096, unique_samples=4096, full_epoch=False, optimizer_updates=4,
                 sample_order_digest="actual-order", pools=[dict(pool_index=0, images=4096)], seconds=10.)])

    def test_same_conditions_pass_and_a0_b6_or_state_runtime_mismatches_fail(self):
        left = self.result()
        self.assertEqual(check_pair(left, copy.deepcopy(left))["status"], "matched")
        for field, value in (("condition", "A0"), ("crop_policy", "per-sample"), ("seed", 2),
                             ("initial_model_hash", "other"), ("precision", "fp32"),
                             ("runtime", {"device_uuid":"other"}), ("accumulation", 1)):
            right = copy.deepcopy(left)
            right["matched_identity"][field] = value
            right["matched_identity_sha256"] = fingerprint(right["matched_identity"])
            with self.assertRaisesRegex(ValueError, "matched identity differs"):
                check_pair(left, right)

    def test_actual_schedule_must_match_and_legacy_identity_cannot_supply_speedup(self):
        left = self.result()
        for field, value in (("sample_order_digest", "different-order"), ("optimizer_updates", 1)):
            right = copy.deepcopy(left)
            right["epochs"][0][field] = value
            with self.assertRaisesRegex(ValueError, "consumed occurrence"):
                check_pair(left, right)
        right = copy.deepcopy(left)
        del right["matched_identity"]["augmentation"]
        right["matched_identity_sha256"] = fingerprint(right["matched_identity"])
        with self.assertRaisesRegex(ValueError, "missing fields"):
            check_pair(left, right)
        with self.assertRaisesRegex(ValueError, "legacy results"):
            check_pair({}, left)


class NativeSearchTests(unittest.TestCase):
    def test_candidate_sample_repeats_and_contains_all_block_endpoints(self):
        rows = subset96()
        self.assertEqual(rows, subset96())
        pairs = {(r["spec"]["block_size"], r["spec"]["k"]) for r in rows}
        self.assertEqual(len(pairs), 96)
        self.assertTrue({(b,k) for b in (8,16,32,64,128,256) for k in (1,b)} <= pairs)
        self.assertTrue(all(r["spec"]["sampling"] == (1,1) and r["spec"]["scalar_quantization"] == 50 for r in rows))

    def test_split_is_disjoint_reproducible_and_rejects_final_leakage(self):
        def source(ids):
            return dict(samples=[dict(logical_sample_id=str(i), path=f"/dataset/{i}.jpg", label=i) for i in ids])
        train, final = source(range(10)), source(range(10,13))
        split = isolated_splits(train, final, search_val_count=3, seed=7)
        self.assertEqual(split, isolated_splits(train, final, search_val_count=3, seed=7))
        self.assertEqual(len(split["train_fit"]), 7)
        self.assertEqual(len(split["search_val"]), 3)
        self.assertFalse(set(split["train_fit"]) & set(split["search_val"]))
        self.assertEqual(split["final_evaluation"], ["10","11","12"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            isolated_splits(train, source([1]), search_val_count=3, seed=7)
        final["samples"][0]["path"] = train["samples"][0]["path"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            isolated_splits(train, final, search_val_count=3, seed=7)

    def test_task_requires_explicit_budgets_and_valid_accuracy_units(self):
        task = dict(model=dict(family="resnet50", classes=1000, stem="declared", small_input_policy="declared"),
                    rgb_preprocessing=dict(source_recipe="fixture", train="crop/flip", evaluation="center crop"),
                    hardware=dict(gpu_uuid="fixture", gpu_name="fixture", cpu_threads=1,
                                  host_memory_bytes=1024, cuda_memory_bytes=1024),
                    training=dict(scratch_epochs=1, transfer_epochs=1, seed=7, optimizer="fixture", transfer_state="declared"),
                    final_training=dict(epochs=1, confirmation_seeds=[7]),
                    runtime_candidates=dict(batch_sizes=[64], data_workers=[0], decode_workers=[1], cache_bytes=[0]))
        task.update(split_seed=7, objective=dict(metric="min_e2e_latency_ms", accuracy_unit="percentage_points",
                    accuracy_delta_pp=.5, reference_candidate_id="B008_k008_s1x1_q050"),
                    budgets=dict(search_gpu_hours=8, latency_limit_ms=None, storage_limit_bytes=None))
        validate_task(task)
        task["budgets"]["search_gpu_hours"] = -1
        with self.assertRaisesRegex(ValueError, "search_gpu_hours"):
            validate_task(task)
