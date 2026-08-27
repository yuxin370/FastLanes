#!/usr/bin/env python3
"""Focused tests for the four-condition core PLS model-effect experiment."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from training_pls.contracts import (
    TRAINING_SOURCE_SCOPE,
    blocking_code_identity,
    code_provenance_differences,
    condition_identity_hash,
    validate_seed_block_contracts,
)
from training_pls.core_schedule import (
    crop_key,
    epoch_position_pools,
    flip_key,
    iter_epoch_pools,
    summarize_epoch,
    update_windows,
)
from training_pls.layout import create_layout_plan, load_layout_mapping, sha256_file
from training_pls.matrix import (
    BALANCED_EXECUTION_ORDER,
    CORE_CONDITION_IDS,
    PAIRED_SEEDS,
    SUPPLEMENTAL_CONDITION_IDS,
    core_matrix,
    experiment_matrix,
    execution_order,
    resolve_condition,
)
from training_pls.published_augmentation import (
    apply_published_randaugment,
    apply_published_randaugment_scalar_reference,
    normalized_to_published_int16,
    published_training_augmentation,
    published_validation_augmentation,
)
from training_pls.recipe import (
    RECIPE_NAME,
    recipe_contract,
    sha256_json,
    validation_epochs,
)
from training_pls.report import aggregate, mean_ci, normalized_auc
from training_pls.published_optimizer import build_published_optimizer
from training_pls.run_matrix import _parse_args as _parse_matrix_args
from training_pls.run_matrix import _seed_devices, build_plan
from training_pls.train import (
    MetricsWriter,
    _checkpoint_payload,
    _parse_args as _parse_train_args,
    _restore_checkpoint,
    _train_native_physical_epoch,
    _validate_contract,
)


@dataclass
class FixtureLayout:
    sample_count: int
    logical_sample_ids: tuple[str, ...]
    virtual_pls_ids: np.ndarray
    positions_by_pls: tuple[np.ndarray, ...]


def fixture_layout(count: int, segment_images: int = 1024) -> FixtureLayout:
    pls_ids = np.asarray(
        [position // segment_images for position in range(count)], dtype=np.int32
    )
    segments = tuple(
        np.flatnonzero(pls_ids == pls_id).astype(np.int64)
        for pls_id in range(int(pls_ids[-1]) + 1)
    )
    return FixtureLayout(
        sample_count=count,
        logical_sample_ids=tuple(f"sample-{index}" for index in range(count)),
        virtual_pls_ids=pls_ids,
        positions_by_pls=segments,
    )


class CoreScheduleTests(unittest.TestCase):
    def test_matrix_is_exact_frozen_two_by_two(self) -> None:
        matrix = core_matrix()
        self.assertEqual(
            tuple(condition["condition_id"] for condition in matrix["conditions"]),
            CORE_CONDITION_IDS,
        )
        self.assertEqual(resolve_condition("A0")["crop_policy"], "per-sample")
        self.assertEqual(resolve_condition("A1")["crop_policy"], "per-pls")
        self.assertEqual(resolve_condition("B2")["order_policy"], "closed-pool")
        self.assertEqual(resolve_condition("B6")["segments_per_pool"], 4)
        self.assertEqual(tuple(BALANCED_EXECUTION_ORDER), PAIRED_SEEDS)

    def test_no_shuffle_is_supplemental_not_a_factorial_cell(self) -> None:
        self.assertEqual(SUPPLEMENTAL_CONDITION_IDS, ("N6", "N2"))
        condition = resolve_condition("N6")
        self.assertEqual(condition["crop_policy"], "per-pls")
        self.assertEqual(condition["order_policy"], "physical-order")
        self.assertEqual(condition["segments_per_pool"], 4)
        self.assertEqual(
            tuple(value["condition_id"] for value in core_matrix()["conditions"]),
            CORE_CONDITION_IDS,
        )
        extended = experiment_matrix(("N6",))
        self.assertEqual(extended["conditions"][-1]["condition_id"], "N6")
        self.assertEqual(len(extended["estimands"]), len(core_matrix()["estimands"]))
        self.assertEqual(
            extended["supplemental_estimands"][0]["formula"], "B6 - N6"
        )
        no_shared_crop = resolve_condition("N2")
        self.assertEqual(no_shared_crop["crop_policy"], "per-sample")
        self.assertEqual(no_shared_crop["crop_key_scope"], "logical_sample_id")
        self.assertEqual(no_shared_crop["order_policy"], "physical-order")
        self.assertEqual(no_shared_crop["segments_per_pool"], 4)
        supplemental_rows = [
            row for row in execution_order([11997733]) if row[2] == "N6"
        ]
        self.assertEqual(supplemental_rows, [(11997733, 5, "N6")])

    def test_n2_and_n6_share_order_but_not_crop_scope(self) -> None:
        layout = fixture_layout(4097)
        n2 = list(
            epoch_position_pools(
                layout, condition_id="N2", seed=11997733, epoch=4
            )
        )
        n6 = list(
            epoch_position_pools(
                layout, condition_id="N6", seed=11997733, epoch=4
            )
        )
        self.assertEqual(n2, n6)
        self.assertNotEqual(
            crop_key(
                training_seed=11997733,
                epoch=4,
                logical_sample_id="a",
                virtual_pls_id=0,
                crop_policy="per-sample",
            ),
            crop_key(
                training_seed=11997733,
                epoch=4,
                logical_sample_id="a",
                virtual_pls_id=0,
                crop_policy="per-pls",
            ),
        )

    def test_required_sizes_have_exact_coverage_and_no_drop(self) -> None:
        for count in (359, 1024, 1025, 4096, 4097):
            layout = fixture_layout(count)
            for condition in CORE_CONDITION_IDS:
                summary = summarize_epoch(
                    layout,
                    condition_id=condition,
                    seed=11997733,
                    epoch=0,
                )
                self.assertEqual(summary.sample_count, count)
                emitted = [
                    item.planned_position
                    for pool in iter_epoch_pools(
                        layout,
                        condition_id=condition,
                        seed=11997733,
                        epoch=0,
                    )
                    for microbatch in pool.microbatches
                    for item in microbatch.items
                ]
                self.assertEqual(len(emitted), count)
                self.assertEqual(set(emitted), set(range(count)))
                self.assertEqual(len(emitted), len(set(emitted)))

    def test_paired_order_controls_hold(self) -> None:
        layout = fixture_layout(4097)

        def order(condition: str) -> list[int]:
            return [
                value
                for _pool, _pls, positions in epoch_position_pools(
                    layout,
                    condition_id=condition,
                    seed=11997733,
                    epoch=7,
                )
                for value in positions
            ]

        self.assertEqual(order("A0"), order("A1"))
        self.assertEqual(order("B2"), order("B6"))
        self.assertNotEqual(order("A0"), order("B2"))
        self.assertEqual(order("B2"), order("B2"))

    def test_no_shuffle_uses_fixed_physical_order_at_every_epoch(self) -> None:
        layout = fixture_layout(4097)

        def pools(epoch: int) -> list[tuple[tuple[int, ...], list[int]]]:
            return [
                (pls_ids, positions)
                for _pool, pls_ids, positions in epoch_position_pools(
                    layout,
                    condition_id="N6",
                    seed=11997733,
                    epoch=epoch,
                )
            ]

        epoch_zero = pools(0)
        self.assertEqual(epoch_zero, pools(7))
        self.assertEqual(epoch_zero[0][0], (0, 1, 2, 3))
        self.assertEqual(epoch_zero[0][1], list(range(4096)))
        self.assertEqual(epoch_zero[1], ((4,), [4096]))

    def test_closed_pool_microbatches_and_updates_do_not_cross_pool(self) -> None:
        layout = fixture_layout(4097)
        pools = list(
            iter_epoch_pools(
                layout,
                condition_id="B6",
                seed=11997733,
                epoch=0,
            )
        )
        self.assertEqual(len(pools), 2)
        self.assertEqual(sum(pool.sample_count for pool in pools), 4097)
        self.assertTrue(all(len(pool.virtual_pls_ids) <= 4 for pool in pools))
        self.assertEqual(
            sorted(pls_id for pool in pools for pls_id in pool.virtual_pls_ids),
            list(range(5)),
        )
        for pool in pools:
            for microbatch in pool.microbatches:
                self.assertTrue(all(item.pool_index == pool.pool_index for item in microbatch.items))
            for window in update_windows(pool):
                self.assertTrue(
                    all(microbatch.pool_index == pool.pool_index for microbatch in window)
                )

    def test_crop_and_flip_key_scopes(self) -> None:
        common = dict(training_seed=9, epoch=3, virtual_pls_id=2)
        sample_a = crop_key(
            **common, logical_sample_id="a", crop_policy="per-sample"
        )
        sample_b = crop_key(
            **common, logical_sample_id="b", crop_policy="per-sample"
        )
        pls_a = crop_key(**common, logical_sample_id="a", crop_policy="per-pls")
        pls_b = crop_key(**common, logical_sample_id="b", crop_policy="per-pls")
        self.assertNotEqual(sample_a, sample_b)
        self.assertEqual(pls_a, pls_b)
        self.assertNotEqual(
            flip_key(training_seed=9, epoch=3, logical_sample_id="a"),
            flip_key(training_seed=9, epoch=3, logical_sample_id="b"),
        )

    def test_published_shared_crop_and_validation_geometry(self) -> None:
        first = published_training_augmentation(
            training_seed=7,
            epoch=0,
            logical_sample_id="a",
            virtual_pls_id=3,
            crop_policy="per-pls",
            source_width=512,
            source_height=512,
        )
        second = published_training_augmentation(
            training_seed=7,
            epoch=0,
            logical_sample_id="b",
            virtual_pls_id=3,
            crop_policy="per-pls",
            source_width=512,
            source_height=512,
        )
        crop = lambda value: (
            value.decision.crop_x,
            value.decision.crop_y,
            value.decision.crop_width,
            value.decision.crop_height,
        )
        self.assertEqual(crop(first), crop(second))
        self.assertEqual(first.crop_key, second.crop_key)
        self.assertNotEqual(first.flip_key, second.flip_key)
        self.assertEqual(first.decision.crop_x % 16, 0)
        self.assertIn(first.decision.crop_width // 8, (2, 4, 14, 28, 56))
        validation = published_validation_augmentation(
            logical_sample_id="v", source_width=512, source_height=512, epoch=7
        )
        self.assertEqual(validation.epoch, 7)
        self.assertEqual(
            (
                validation.crop_x,
                validation.crop_y,
                validation.crop_width,
                validation.crop_height,
            ),
            (32, 32, 448, 448),
        )

    def test_randaugment_entry_conversion_clamps_to_published_range(self) -> None:
        values = torch.tensor([-2.0, -1.1, -1.0, 1.0, 2.0])
        converted = normalized_to_published_int16(values)
        self.assertEqual(converted.dtype, torch.int16)
        self.assertEqual(converted.tolist(), [-1024, -1024, -1024, 1016, 1016])

    def test_grouped_randaugment_is_exactly_scalar_equivalent(self) -> None:
        rgbnomore_root = Path("/home/tangyuxin/RGB-no-more")
        generator = torch.Generator().manual_seed(20260811)
        y_raw = torch.randint(-1300, 1300, (32, 1, 28, 28, 8, 8), generator=generator)
        cbcr_raw = torch.randint(
            -1300, 1300, (32, 2, 14, 14, 8, 8), generator=generator
        )
        inputs = tuple(
            (value.clamp(-1024, 1016).float() + 4.0) / 1020.0
            for value in (y_raw, cbcr_raw)
        )
        logical_ids = [f"randaugment-{index}" for index in range(32)]
        grouped, grouped_records = apply_published_randaugment(
            inputs,
            training_seed=11997733,
            epoch=3,
            logical_sample_ids=logical_ids,
            rgbnomore_root=rgbnomore_root,
        )
        scalar, scalar_records = apply_published_randaugment_scalar_reference(
            inputs,
            training_seed=11997733,
            epoch=3,
            logical_sample_ids=logical_ids,
            rgbnomore_root=rgbnomore_root,
        )
        self.assertEqual(grouped_records, scalar_records)
        self.assertTrue(torch.equal(grouped[0], scalar[0]))
        self.assertTrue(torch.equal(grouped[1], scalar[1]))


class RecipeAndContractTests(unittest.TestCase):
    def test_native_physical_epoch_consumes_native_order_and_updates_model(self) -> None:
        class Batch:
            def __init__(self, image_ids: list[int]) -> None:
                self.global_image_ids = image_ids
                self.labels = [value % 2 for value in image_ids]
                y = torch.tensor([[float(value)] for value in image_ids])
                cbcr = torch.zeros_like(y)
                targets = torch.nn.functional.one_hot(
                    torch.tensor(self.labels), num_classes=2
                ).float()
                self.tensors = (y, cbcr, targets)

        class Pool:
            epoch = 0
            pool_index = 0
            virtual_pls_ids = [0]
            image_count = 4
            microbatch_count = 2
            execution_stats = {
                "selected_vector_count": 4,
                "full_vector_count": 8,
                "compressed_payload_bytes_read": 16,
                "uses_planless_fixed_transform": True,
            }

            def __init__(self) -> None:
                self._batches = iter((Batch([2, 0]), Batch([3, 1])))

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._batches)

        class Pipeline:
            def __init__(self) -> None:
                self.pending = False
                self.reclaims = 0

            def start_epoch(self, epoch: int) -> None:
                self.pending = True

            @property
            def has_next_pool(self) -> bool:
                return self.pending

            def next_pool(self) -> Pool:
                self.pending = False
                return Pool()

            def reclaim_finished_pools(self) -> int:
                self.reclaims += 1
                return 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
                return self.linear(torch.cat((y, cbcr), dim=1))

        class Scheduler:
            def __init__(self) -> None:
                self.updates = 0

            def prepare_next_update(self) -> float:
                return 0.1

            def complete_update(self) -> None:
                self.updates += 1

        class WeightDecay:
            def step(self, learning_rate: float) -> None:
                self.learning_rate = learning_rate

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline()
            model = Model()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = Scheduler()
            metrics = MetricsWriter(Path(temporary) / "metrics.jsonl")
            with mock.patch("torch.cuda.synchronize"):
                result = _train_native_physical_epoch(
                    pipeline=pipeline,
                    execution_model=model,
                    model=model,
                    optimizer=optimizer,
                    weight_decayer=WeightDecay(),
                    scheduler=scheduler,
                    device=torch.device("cpu"),
                    recipe={
                        "training": {
                            "physical_microbatch": 64,
                            "gradient_accumulation": 16,
                        },
                        "optimizer": {"gradient_clipping_norm": 1.0},
                        "logging": {"train_loss_every_optimizer_updates": 100},
                    },
                    metrics=metrics,
                    loader_totals={},
                    integration_checks={"coverage_counters_valid": True},
                    condition_id="B6",
                    seed=11997733,
                    epoch=0,
                    expected_sample_count=4,
                    global_update=0,
                    processed_images=0,
                    loss_since_log=0.0,
                    samples_since_log=0,
                    last_logged_update=0,
                    integration_check_first_100=False,
                )
            self.assertEqual(result["global_update"], 1)
            self.assertEqual(result["processed_images"], 4)
            self.assertEqual(result["epoch_record"]["epoch_microbatches"], 2)
            self.assertEqual(result["epoch_record"]["native_pool_count"], 1)
            self.assertEqual(
                result["epoch_record"]["native_execution_stats"][
                    "selected_vector_count"
                ],
                4,
            )
            self.assertEqual(scheduler.updates, 1)
            self.assertEqual(pipeline.reclaims, 1)

    def test_seed_device_mapping_is_exact_and_defaults_cleanly(self) -> None:
        seeds = (11997733, 11997734)
        self.assertEqual(
            _seed_devices(None, seeds=seeds, default_device="cuda:0"),
            {11997733: "cuda:0", 11997734: "cuda:0"},
        )
        self.assertEqual(
            _seed_devices(
                "11997733=cuda:1,11997734=cuda:0",
                seeds=seeds,
                default_device="cuda:2",
            ),
            {11997733: "cuda:1", 11997734: "cuda:0"},
        )
        with self.assertRaises(ValueError):
            _seed_devices(
                "11997733=cuda:1", seeds=seeds, default_device="cuda:0"
            )

    def test_epoch_checkpoint_restores_auditable_rolling_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = torch.nn.Sequential(torch.nn.Linear(2, 2))
            optimizer, weight_decayer, scheduler = build_published_optimizer(
                model,
                learning_rate=3e-3,
                weight_decay=1e-4,
                warmup_updates=2,
                total_updates=10,
            )
            metrics = MetricsWriter(root / "metrics.jsonl")
            metrics.append({"record_type": "train", "optimizer_update": 100})
            checkpoint = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                weight_decayer=weight_decayer,
                scheduler=scheduler,
                completed_epoch=1,
                global_optimizer_update=1252,
                processed_images=1281167,
                recipe_hash="recipe",
                layout_hash="layout",
                condition_hash="condition",
                initial_model_hash="initial",
                metrics_cursor=metrics.cursor(),
                logging_state={
                    "loss_since_log": 12.5,
                    "samples_since_log": 53248,
                    "last_logged_update": 1200,
                },
                integration_checks={"required_updates": 100, "checked_updates": 100},
                loader_totals={"reader.wait": 3.5},
                elapsed_runtime_seconds=99.0,
                pending_validation_epoch=1,
            )
            checkpoint_path = root / "latest.pt"
            torch.save(checkpoint, checkpoint_path)
            metrics.append({"record_type": "train", "optimizer_update": 101})
            completed, updates, images, bookkeeping = _restore_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                weight_decayer=weight_decayer,
                scheduler=scheduler,
                recipe_hash="recipe",
                layout_hash="layout",
                condition_hash="condition",
                initial_model_hash="initial",
                metrics=metrics,
            )
            self.assertEqual((completed, updates, images), (1, 1252, 1281167))
            self.assertEqual(metrics.lines, 1)
            self.assertEqual(bookkeeping["logging_state"]["samples_since_log"], 53248)
            self.assertEqual(bookkeeping["integration_checks"]["checked_updates"], 100)
            self.assertEqual(bookkeeping["loader_totals"]["reader.wait"], 3.5)
            self.assertEqual(bookkeeping["elapsed_runtime_seconds"], 99.0)
            self.assertEqual(bookkeeping["pending_validation_epoch"], 1)

    def test_layout_plan_writes_real_parquet_and_explicit_pls_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "galp-rgbnomore-training-manifest-v1",
                        "population_count": 5,
                        "galp_manifest": "/fixture/manifest.bin",
                        "samples": [
                            {
                                "logical_sample_id": f"sample-{index}",
                                "label": index % 2,
                                "galp_image_id": image_id,
                                "width": 512,
                                "height": 512,
                            }
                            for index, image_id in enumerate((4, 1, 3, 0, 2))
                        ],
                    }
                ),
                encoding="utf-8",
            )
            plan = create_layout_plan(manifest, root / "layout", segment_images=2)
            parquet = root / "layout" / "physical_layout_samples.parquet"
            raw = parquet.read_bytes()
            self.assertEqual(raw[:4], b"PAR1")
            self.assertEqual(raw[-4:], b"PAR1")
            mapping = load_layout_mapping(root / "layout" / "physical_layout_plan.json")
            self.assertEqual(mapping.sample_count, 5)
            self.assertEqual(mapping.virtual_pls_ids.tolist(), [0, 0, 1, 1, 2])
            self.assertEqual(mapping.galp_image_ids.tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(plan["sample_mapping_hash"], plan["sample_mapping_hash"])

    def test_recipe_is_locked(self) -> None:
        recipe = recipe_contract(RECIPE_NAME)
        self.assertEqual(recipe["training"]["epochs"], 300)
        self.assertEqual(recipe["training"]["physical_microbatch"], 64)
        self.assertEqual(recipe["training"]["gradient_accumulation"], 16)
        self.assertEqual(recipe["optimizer"]["learning_rate"], 3e-3)
        self.assertEqual(recipe["optimizer"]["weight_decay"]["coefficient"], 1e-4)
        self.assertEqual(recipe["augmentation"]["mixup"]["alpha"], 0.2)
        self.assertEqual(recipe["augmentation"]["randaugment"]["num_operations"], 2)
        self.assertEqual(recipe["augmentation"]["randaugment"]["magnitude"], 3)
        self.assertTrue(recipe["execution"]["model_compile"]["enabled"])
        self.assertEqual(
            recipe["execution"]["randaugment_dispatch"],
            "exact-keyed-operation-grouped-v1",
        )
        self.assertEqual(validation_epochs()[:4], (0, 1, 2, 5))
        self.assertEqual(validation_epochs()[-1], 300)

    def test_condition_whitelist_accepts_only_registered_fields(self) -> None:
        invariant = {
            "schema_version": "x",
            "training_seed": 11,
            "recipe_hash": "r",
            "layout_hash": "l",
            "condition_hash": "ignored",
        }
        contracts = []
        for condition in CORE_CONDITION_IDS:
            resolved = resolve_condition(condition)
            contracts.append(
                {
                    **invariant,
                    "condition_id": condition,
                    "crop_policy": resolved["crop_policy"],
                    "crop_key_scope": resolved["crop_key_scope"],
                    "order_policy": resolved["order_policy"],
                    "segments_per_pool": resolved["segments_per_pool"],
                    "pool_membership_digest": resolved["order_policy"],
                    "sample_order_digest": resolved["order_policy"],
                    "crop_key_digest": resolved["crop_policy"],
                }
            )
        self.assertTrue(validate_seed_block_contracts(contracts)["valid"])
        contracts[-1]["recipe_hash"] = "changed"
        with self.assertRaises(ValueError):
            validate_seed_block_contracts(contracts)

    def test_scoped_code_identity_ignores_unrelated_repository_changes(self) -> None:
        planned = {
            "commit": "old",
            "working_tree_clean": True,
            "working_tree_status_sha256": "status-old",
            "tracked_diff_sha256": "diff-old",
            "experiment_source_tree_sha256": "broad-old",
            "training_source_scope": TRAINING_SOURCE_SCOPE,
            "training_runtime_source_tree_sha256": "runtime-stable",
        }
        current = {
            **planned,
            "commit": "new",
            "working_tree_clean": False,
            "working_tree_status_sha256": "status-new",
            "tracked_diff_sha256": "diff-new",
            "experiment_source_tree_sha256": "broad-new",
        }
        self.assertEqual(
            blocking_code_identity(planned), blocking_code_identity(current)
        )
        self.assertEqual(
            condition_identity_hash({"code_version": planned, "condition_id": "B6"}),
            condition_identity_hash({"code_version": current, "condition_id": "B6"}),
        )
        self.assertEqual(
            set(code_provenance_differences(planned, current)),
            {
                "commit",
                "working_tree_clean",
                "working_tree_status_sha256",
                "tracked_diff_sha256",
                "experiment_source_tree_sha256",
            },
        )
        current["training_runtime_source_tree_sha256"] = "runtime-changed"
        self.assertNotEqual(
            blocking_code_identity(planned), blocking_code_identity(current)
        )

    def test_contract_validation_treats_repository_provenance_as_non_blocking(
        self,
    ) -> None:
        planned_code = {
            "commit": "planned",
            "working_tree_clean": True,
            "working_tree_status_sha256": "planned-status",
            "tracked_diff_sha256": "planned-diff",
            "experiment_source_tree_sha256": "planned-broad",
            "training_source_scope": TRAINING_SOURCE_SCOPE,
            "training_runtime_source_tree_sha256": "runtime-stable",
        }
        current_code = {
            **planned_code,
            "commit": "current",
            "working_tree_clean": False,
            "working_tree_status_sha256": "current-status",
            "tracked_diff_sha256": "current-diff",
            "experiment_source_tree_sha256": "current-broad",
        }
        payload = {
            "condition_id": "B6",
            "training_seed": 11997733,
            "recipe_hash": "recipe",
            "layout_hash": "layout",
            "execution_mode": "semantic_emulation",
            "semantic_emulation": True,
            "physical_fls_observed": False,
            "physical_gpu_pool": False,
            "backend_implementation": "shared-galp-direct-dct-semantic-backend-v2",
            "code_version": planned_code,
        }
        payload["condition_hash"] = condition_identity_hash(payload)
        payload["run_manifest_hash"] = sha256_json(payload)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch(
                "training_pls.train.code_version", return_value=current_code
            ):
                observed, validation = _validate_contract(
                    path,
                    condition_id="B6",
                    seed=11997733,
                    recipe_hash="recipe",
                    layout_hash="layout",
                    execution_backend="semantic-emulation",
                    physical_galp_manifest=None,
                    premixed_mapping_csv=None,
                    expected_mapping_sha256=None,
                )
            self.assertEqual(observed["condition_hash"], payload["condition_hash"])
            self.assertTrue(validation["blocking_code_identity_matches"])
            self.assertTrue(validation["non_blocking_code_provenance_changed"])

            changed_runtime = {
                **current_code,
                "training_runtime_source_tree_sha256": "runtime-changed",
            }
            with mock.patch(
                "training_pls.train.code_version", return_value=changed_runtime
            ):
                with self.assertRaisesRegex(
                    ValueError, "training runtime sources changed"
                ):
                    _validate_contract(
                        path,
                        condition_id="B6",
                        seed=11997733,
                        recipe_hash="recipe",
                        layout_hash="layout",
                        execution_backend="semantic-emulation",
                        physical_galp_manifest=None,
                        premixed_mapping_csv=None,
                        expected_mapping_sha256=None,
                    )

    def test_manifest_cli_and_matrix_epoch_boundary_pause(self) -> None:
        train_args = _parse_train_args(
            [
                "--train-manifest",
                "train.json",
                "--val-manifest",
                "val.json",
                "--layout-plan",
                "layout.json",
                "--run-manifest",
                "run_manifest.json",
                "--condition",
                "B6",
                "--seed",
                "11997733",
                "--output-dir",
                "output",
                "--stop-after-epoch",
                "2",
            ]
        )
        self.assertEqual(train_args.run_manifest, Path("run_manifest.json"))
        self.assertIsNone(train_args.condition_contract)
        self.assertEqual(train_args.stop_after_epoch, 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("train.json", "val.json", "layout.json")]
            for path in paths:
                path.touch()
            matrix_args = _parse_matrix_args(
                [
                    "--output-dir",
                    str(root / "output"),
                    "--train-manifest",
                    str(paths[0]),
                    "--val-manifest",
                    str(paths[1]),
                    "--layout-plan",
                    str(paths[2]),
                    "--stop-after-epoch",
                    "2",
                ]
            )
        self.assertEqual(matrix_args.stop_after_epoch, 2)

    def test_matrix_plan_writes_canonical_run_manifest_and_forwards_pause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_manifest = root / "train.json"
            val_manifest = root / "val.json"
            layout_path = root / "layout.json"
            for path in (train_manifest, val_manifest, layout_path):
                path.write_text("{}", encoding="utf-8")
            output = root / "output"
            args = _parse_matrix_args(
                [
                    "--output-dir",
                    str(output),
                    "--train-manifest",
                    str(train_manifest),
                    "--val-manifest",
                    str(val_manifest),
                    "--layout-plan",
                    str(layout_path),
                    "--conditions",
                    "B6",
                    "--seeds",
                    "11997733",
                    "--stop-after-epoch",
                    "2",
                ]
            )
            positions = tuple(
                np.arange(index * 1024, (index + 1) * 1024, dtype=np.int64)
                for index in range(4)
            )
            mapping = mock.Mock(
                sample_count=4096,
                positions_by_pls=positions,
                plan={"target_pls_size": 1024},
                layout_hash="layout-hash",
            )
            layout_plan = {
                "target_pls_size": 1024,
                "dataset_manifest_hash": sha256_file(train_manifest),
                "layout_hash": "layout-hash",
            }
            version = {
                "commit": "commit",
                "working_tree_clean": True,
                "working_tree_status_sha256": "status",
                "tracked_diff_sha256": "diff",
                "experiment_source_files": [],
                "experiment_source_tree_sha256": "broad",
                "training_source_scope": TRAINING_SOURCE_SCOPE,
                "training_runtime_source_files": [],
                "training_runtime_source_tree_sha256": "runtime",
            }
            with (
                mock.patch(
                    "training_pls.run_matrix.load_layout_plan",
                    return_value=layout_plan,
                ),
                mock.patch(
                    "training_pls.run_matrix.load_layout_mapping",
                    return_value=mapping,
                ),
                mock.patch(
                    "training_pls.run_matrix._initial_model_hashes",
                    return_value={11997733: "initial-model"},
                ),
                mock.patch(
                    "training_pls.run_matrix.code_version", return_value=version
                ),
                mock.patch(
                    "training_pls.run_matrix._environment", return_value={}
                ),
            ):
                built = build_plan(args)

            command = built["plan"]["commands"][0]
            run_manifest = (
                output / "runs" / "B6" / "seed_11997733" / "run_manifest.json"
            )
            self.assertEqual(Path(command["run_manifest"]), run_manifest)
            self.assertTrue(run_manifest.is_file())
            self.assertIn("--run-manifest", command["argv"])
            self.assertNotIn("--condition-contract", command["argv"])
            stop_index = command["argv"].index("--stop-after-epoch")
            self.assertEqual(command["argv"][stop_index + 1], "2")
            self.assertTrue(
                (output / "contracts" / "seed_11997733" / "B6.json").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "runs"
                    / "B6"
                    / "seed_11997733"
                    / "condition_contract.json"
                ).is_file()
            )


class ReportTests(unittest.TestCase):
    def test_statistics_and_complete_synthetic_report(self) -> None:
        summary = mean_ci([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["mean"], 2.5)
        auc = normalized_auc(
            [
                {"processed_images": 0, "validation_top1": 10.0},
                {"processed_images": 50, "validation_top1": 20.0},
                {"processed_images": 100, "validation_top1": 30.0},
            ],
            100,
        )
        self.assertAlmostEqual(auc, 20.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs-root"
            shifts = {"A0": 0.0, "A1": 0.1, "B2": -0.2, "B6": 0.05}
            for condition in CORE_CONDITION_IDS:
                for seed_index, seed in enumerate(PAIRED_SEEDS):
                    run = root / "runs" / condition / f"seed_{seed}"
                    run.mkdir(parents=True)
                    final = 70.0 + seed_index * 0.02 + shifts[condition]
                    events = [
                        {
                            "record_type": "validation",
                            "condition": condition,
                            "seed": seed,
                            "epoch": 0,
                            "optimizer_update": 0,
                            "processed_images": 0,
                            "validation_top1": 1.0,
                            "validation_top5": 5.0,
                            "validation_loss": 7.0,
                            "validation_latency_seconds": 1.0,
                        },
                        {
                            "record_type": "train",
                            "scope": "epoch",
                            "condition": condition,
                            "seed": seed,
                            "epoch": 300,
                            "optimizer_update": 10,
                            "processed_images": 100,
                            "train_loss": 0.5,
                            "learning_rate": 0.0,
                        },
                        {
                            "record_type": "validation",
                            "condition": condition,
                            "seed": seed,
                            "epoch": 300,
                            "optimizer_update": 10,
                            "processed_images": 100,
                            "validation_top1": final,
                            "validation_top5": final + 20.0,
                            "validation_loss": 1.0,
                            "validation_latency_seconds": 1.0,
                        },
                    ]
                    (run / "metrics.jsonl").write_text(
                        "".join(json.dumps(event) + "\n" for event in events),
                        encoding="utf-8",
                    )
                    result = {
                        "state": "completed",
                        "condition": condition,
                        "seed": seed,
                        "final_top1": final,
                        "final_top5": final + 20.0,
                        "final_validation_loss": 1.0,
                        "total_processed_images": 100,
                        "total_optimizer_updates": 10,
                        "training_runtime_seconds": 2.0,
                        "images_per_second": 50.0,
                        "cuda_memory": {
                            "peak_allocated_bytes": 1,
                            "peak_reserved_bytes": 2,
                        },
                        "execution_mode": "semantic_emulation",
                        "semantic_emulation": True,
                        "physical_fls_observed": False,
                        "physical_gpu_pool": False,
                        "layout_hash": "layout",
                        "recipe_hash": "recipe",
                        "condition_hash": condition,
                        "initial_model_hash": str(seed),
                    }
                    (run / "final_result.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
                    (run / "run_status.json").write_text(
                        json.dumps({"state": "completed"}), encoding="utf-8"
                    )
            output = Path(temporary) / "report"
            result = aggregate(
                root,
                output,
                conditions=CORE_CONDITION_IDS,
                require_complete=True,
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_runs"], 16)
            interaction = next(
                row
                for row in result["effects"]
                if row["effect_id"] == "crop_x_shuffle"
                and row["endpoint"] == "final_top1"
            )
            self.assertEqual(interaction["paired_seed_count"], 4)
            for name in (
                "convergence_curves.csv",
                "final_metrics.csv",
                "paired_effects.csv",
                "factorial_effects.csv",
                "model_results.json",
                "report.md",
                "top1_convergence.png",
                "top5_convergence.png",
                "train_loss_curves.png",
                "final_top1_by_seed.png",
                "paired_differences.png",
                "factorial_effects.png",
                "normalized_auc.png",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
