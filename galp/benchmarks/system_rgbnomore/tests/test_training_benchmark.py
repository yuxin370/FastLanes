#!/usr/bin/env python3
"""CPU audit tests for the formal four-pipeline training benchmark."""

from __future__ import annotations

import copy
import csv
import gc
import json
import ast
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from galp.benchmarks.training_audit_policy import (
    TrainingAuditPolicy,
    TrainingAuditState,
)


BENCHMARK_DIR = Path(__file__).resolve().parents[1]

from galp.benchmarks.system_rgbnomore.training.artifacts import (  # noqa: E402
    nested_state_sha256,
    repository_provenance,
    sha256_file,
    tensor_state_sha256,
    verify_artifact_hashes,
    write_artifact_hashes,
)
from galp.benchmarks.system_rgbnomore.training.augmentation import (  # noqa: E402
    derive_augmentation,
    derive_shard_shared_crop_augmentation,
    horizontal_flip_dct,
)
from galp.benchmarks.system_rgbnomore.training.metrics import gradient_summary, process_memory, tensor_is_finite  # noqa: E402
from galp.benchmarks.system_rgbnomore.training.direct_dct_reader import (  # noqa: E402
    DirectDctTrainingReader,
    NativeExecutionStatsAccumulator,
    merge_native_counter_snapshot,
    native_allocation_stability,
)
from galp.benchmarks.system_rgbnomore.training.manifest_preflight import (  # noqa: E402
    ManifestPreflightError,
    preflight_manifest,
)
from galp.benchmarks.system_rgbnomore.training.generate_imagenet_manifests import main as generate_training_manifests  # noqa: E402
from galp.benchmarks.system_rgbnomore.training.model_factory import (  # noqa: E402
    EXPECTED_PARAMETER_COUNTS,
    MODEL_IDS,
    SWINV2_T_MODEL_ID,
    build_model,
    capture_rng_state,
    capture_training_state,
    initialize_model,
    model_configuration,
    reset_training_state,
    restore_rng_state,
    seed_everything,
)
from galp.benchmarks.system_rgbnomore.training.optimizer import build_optimizer, build_scheduler  # noqa: E402
from galp.benchmarks.system_rgbnomore.training.pipeline import (  # noqa: E402
    DaliTrainingAdapter,
    GalpTrainingAdapter,
    PyTorchTrainingAdapter,
    RgbNoMoreTrainingAdapter,
    _RgbNoMoreDctDataset,
    TrainingBatch,
    TrainingSample,
    _uniform_dali_tensor,
    load_training_manifest,
    validate_dataset_separation,
)
from galp.benchmarks.system_rgbnomore.training.run import (  # noqa: E402
    _adapter_pipeline_config,
    _aggregate_step_repeats,
    _augmentation_batches,
    _collect_batches,
    _parse_args,
    _load_resume,
    _first_step_probe_policy,
    _resolve_runtime_record,
    _save_training_checkpoint,
    _SyncLedger,
    _sync,
    _semantic_compare,
    _train_one_step,
    _train_one_step_runtime,
    _validate_galp_dataset_binding,
    _validate_args,
    run,
)
from galp.benchmarks.system_rgbnomore.training.sample_order import (  # noqa: E402
    SampleIdentity,
    SampleOrderLedger,
    canonical_epoch_order,
)
from galp.benchmarks.system_rgbnomore.training.select_imagenet_canary import select_canary  # noqa: E402
from galp.benchmarks.system_rgbnomore.training.schema import (  # noqa: E402
    TRAINING_CONTRACT_SCHEMA,
    empty_status,
    validate_required_group_coverage,
    validate_training_document,
)
from galp.benchmarks.system_rgbnomore.training.validate import validate_output  # noqa: E402


RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")


def _optimizer_config() -> dict[str, object]:
    return {
        "type": "sgd",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "momentum": 0.0,
        "nesterov": False,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
    }


def _write_jpeg(path: Path, value: int) -> None:
    pixels = np.full((256, 256, 3), value, dtype=np.uint8)
    pixels[32:224, 48:208, :] = (value + 37) % 255
    Image.fromarray(pixels, mode="RGB").save(path, quality=95, subsampling=2)


def _write_jpeg_sampling(path: Path, value: int, subsampling: int) -> None:
    pixels = np.full((32, 32, 3), value, dtype=np.uint8)
    pixels[8:24, 8:24, :] = (value + 37) % 255
    Image.fromarray(pixels, mode="RGB").save(
        path, quality=95, subsampling=subsampling
    )


def _samples(root: Path, count: int = 4) -> list[TrainingSample]:
    result = []
    for index in range(count):
        path = root / f"sample_{index}.jpg"
        _write_jpeg(path, 20 + index * 30)
        result.append(TrainingSample(f"id-{index}", path, index, 256, 256, index))
    return result


def _write_mock_galp_manifest(
    root: Path,
    *,
    version: int,
    image_count: int = 4,
    include_v3_extension: bool = True,
    v3_descriptor_kind: str = "galp-compact-v1",
    v3_spatial_order: str = "tiled-z32",
    v3_spatial_order_id: int = 3,
    rowgroup_vectors: int | None = None,
) -> Path:
    fls = root / "shard_000000.fls"
    metadata = root / "shard_000000.meta.bin"
    fls.write_bytes(b"fls-payload")
    metadata.write_bytes(b"metadata-payload")

    def encoded(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<I", len(raw)) + raw

    payload = bytearray(b"GJDCTSH1")
    effective_rowgroup_vectors = (
        rowgroup_vectors if rowgroup_vectors is not None else (1 if version == 3 else 1024)
    )
    payload.extend(
        struct.pack(
            "<IHIIQI",
            version,
            1,
            effective_rowgroup_vectors,
            8192,
            image_count,
            1,
        )
    )
    payload.extend(
        struct.pack(
            "<IQIQQQIIQQ",
            0,
            0,
            image_count,
            image_count,
            0,
            image_count,
            1,
            1,
            fls.stat().st_size,
            metadata.stat().st_size,
        )
    )
    payload.extend(encoded(fls.name))
    payload.extend(encoded(metadata.name))
    if version == 3 and include_v3_extension:
        payload.extend(b"GJDCCV31")
        payload.extend(encoded("image-major-vector-rowgroups"))
        payload.extend(encoded(v3_descriptor_kind))
        payload.extend(struct.pack("<I", 1024))
        payload.extend(encoded(v3_spatial_order))
        payload.extend(struct.pack("<HI", v3_spatial_order_id, 1))
        payload.extend(struct.pack("<IQQQQ", 0, fls.stat().st_size, 0, 64, 128))
    manifest = root / f"manifest-v{version}.bin"
    manifest.write_bytes(payload)
    return manifest


class _OneBatchAdapter:
    def __init__(self, batch: TrainingBatch) -> None:
        self.batch = batch

    def next_batch(self) -> TrainingBatch:
        return self.batch


class _FakeNativeTrainingBatch:
    def __init__(
        self,
        image_ids: list[int],
        transforms: list[dict[str, object]],
        execution_stats: dict[str, object] | None,
    ) -> None:
        self.y = torch.zeros(len(image_ids), 1, 28, 28, 8, 8)
        self.cbcr = torch.zeros(len(image_ids), 2, 14, 14, 8, 8)
        self.global_image_ids = image_ids
        self.transform_descriptors = [
            {**transform, "global_image_id": image_id}
            for image_id, transform in zip(image_ids, transforms)
        ]
        self._execution_stats = execution_stats
        self.complete_stats_reads = 0
        self.snapshot_stats_reads = 0

    @property
    def execution_stats(self):
        self.complete_stats_reads += 1
        return self._execution_stats

    @property
    def execution_stats_snapshot(self):
        self.snapshot_stats_reads += 1
        return self._execution_stats

    @property
    def tensors(self):
        return self.y, self.cbcr

    @property
    def metrics(self):
        return SimpleNamespace(
            consumer_wait_ms=0.25,
            producer_ms=2.0,
            planning_ms=0.5,
            io_ms=0.75,
            decode_ms=1.0,
            transform_ms=0.25,
            logical_bytes=1024,
            physical_bytes=512,
            peak_transient_bytes=4096,
        )

    def native_execution_stats(self):
        self.complete_stats_reads += 1
        return {} if self._execution_stats is None else dict(self._execution_stats)

    def native_execution_stats_observation(self):
        self.snapshot_stats_reads += 1
        return {
            "stats": self._execution_stats or {},
            "host_snapshot_taken": True,
            "gpu_timings_finalized": False,
        }


class _FakeDirectDctTrainingReader:
    def __init__(self, execution_stats: dict[str, object] | None = None) -> None:
        self.execution_stats = execution_stats
        self.requests: list[tuple[list[int], list[dict[str, object]], dict[str, object]]] = []
        self.batches: list[_FakeNativeTrainingBatch] = []
        self.start_requests: list[list[list[int]]] = []
        self.lifecycle: list[str] = []

        self._next_batch = 0

    def start(self, image_id_batches, *, transforms_by_batch):
        self.requests = []
        self.batches = []
        self._next_batch = 0
        for image_ids, transforms in zip(image_id_batches, transforms_by_batch):
            ids = [int(value) for value in image_ids]
            descriptors = [dict(value) for value in transforms]
            self.requests.append((ids, descriptors, {}))
            self.batches.append(
                _FakeNativeTrainingBatch(ids, descriptors, self.execution_stats)
            )
        self.start_requests.append(
            [list(request[0]) for request in self.requests]
        )
        self.lifecycle.append("start")

    def next_batch(self):
        if self._next_batch >= len(self.batches):
            raise StopIteration
        batch = self.batches[self._next_batch]
        self._next_batch += 1
        return batch

    def close(self):
        self.lifecycle.append("close")
        return None

    def metrics(self):
        consumed = self._next_batch
        return SimpleNamespace(
            complete=True,
            consumer_wait_ms=0.25 * consumed,
            producer_ms=2.0 * consumed,
            planning_ms=0.5 * consumed,
            io_ms=0.75 * consumed,
            decode_ms=1.0 * consumed,
            transform_ms=0.25 * consumed,
            logical_bytes=1024 * consumed,
            physical_bytes=512 * consumed,
            peak_transient_bytes=4096 if consumed else 0,
        )

    def metrics_snapshot(self):
        return vars(self.metrics())

    def aggregate_metrics_snapshots(self, snapshots):
        return {
            key: (all(snapshot[key] for snapshot in snapshots) if key == "complete"
                  else max(snapshot[key] for snapshot in snapshots) if key == "peak_transient_bytes"
                  else sum(snapshot[key] for snapshot in snapshots))
            for key in snapshots[0]
        }

    def prefetched_batch_count(self):
        return min(len(self.batches), self._next_batch + 2)


class _FakeDaliTensorList:
    class _Tensor:
        def shape(self):
            return [np.int64(2), np.int32(3), 4, 4]

    def __init__(self) -> None:
        self.tensor = self._Tensor()

    def as_tensor(self):
        return self.tensor


class TrainingBenchmarkTest(unittest.TestCase):
    def test_official_split_generator_preserves_manifest_local_image_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots: dict[str, Path] = {}
            indexes: dict[str, Path] = {}
            manifests: dict[str, Path] = {}
            for split, count in (("train", 3), ("val", 2)):
                jpeg_root = root / f"jpeg-{split}"
                class_root = jpeg_root / "n00000001"
                class_root.mkdir(parents=True)
                rows = []
                for image_id in range(count):
                    filename = f"{split}-{image_id}.JPEG"
                    _write_jpeg_sampling(class_root / filename, 20 + image_id, 2)
                    rows.append(
                        {
                            "Filepath": f"{split}/n00000001/{filename}",
                            "Label": image_id,
                        }
                    )
                index = root / f"index-{split}.csv"
                with index.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("Filepath", "Label"))
                    writer.writeheader()
                    writer.writerows(rows)
                dct_root = root / f"dct-{split}"
                dct_root.mkdir()
                roots[split] = jpeg_root
                indexes[split] = index
                manifests[split] = _write_mock_galp_manifest(
                    dct_root, version=3, image_count=count
                )

            output = root / "manifests"
            self.assertEqual(
                generate_training_manifests(
                    [
                        "--jpeg-root",
                        str(roots["train"]),
                        "--index-csv",
                        str(indexes["train"]),
                        "--galp-manifest",
                        str(manifests["train"]),
                        "--validation-jpeg-root",
                        str(roots["val"]),
                        "--validation-index-csv",
                        str(indexes["val"]),
                        "--galp-validation-manifest",
                        str(manifests["val"]),
                        "--output-dir",
                        str(output),
                        "--train-count",
                        "0",
                        "--val-count",
                        "0",
                        "--no-probe-dimensions",
                    ]
                ),
                0,
            )
            train = json.loads((output / "train.json").read_text(encoding="utf-8"))
            validation = json.loads((output / "val.json").read_text(encoding="utf-8"))
            self.assertEqual(train["validation_semantics"], "official-imagenet-validation")
            self.assertEqual(validation["validation_semantics"], "official-imagenet-validation")
            self.assertEqual(
                [sample["galp_image_id"] for sample in train["samples"]],
                [0, 1, 2],
            )
            self.assertEqual(
                [sample["galp_image_id"] for sample in validation["samples"]],
                [0, 1],
            )
            self.assertTrue(
                all(
                    sample["logical_sample_id"].startswith("train/")
                    for sample in train["samples"]
                )
            )
            self.assertTrue(
                all(
                    sample["logical_sample_id"].startswith("val/")
                    for sample in validation["samples"]
                )
            )

    def test_canary_selector_is_balanced_exact_and_sampling_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tar_root = root / "class-tars"
            source = root / "source"
            tar_root.mkdir()
            source.mkdir()
            index_rows: list[dict[str, object]] = []
            for label, wnid in enumerate(("n00000001", "n00000002")):
                class_source = source / wnid
                class_source.mkdir()
                values = (
                    (f"{wnid}_unsupported.JPEG", 1),
                    (f"{wnid}_444.JPEG", 0),
                    (f"{wnid}_420.JPEG", 2),
                )
                for ordinal, (filename, subsampling) in enumerate(values):
                    path = class_source / filename
                    _write_jpeg_sampling(path, 30 + label * 40 + ordinal, subsampling)
                    index_rows.append(
                        {"Filepath": f"train/{wnid}/{filename}", "Label": label}
                    )
                with tarfile.open(tar_root / f"{wnid}.tar", "w") as archive:
                    for path in sorted(class_source.iterdir()):
                        archive.add(path, arcname=path.name)
            index_csv = root / "index.csv"
            with index_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Filepath", "Label"])
                writer.writeheader()
                writer.writerows(index_rows)

            hashes = []
            for run_number in (1, 2):
                output = root / f"jpeg-{run_number}"
                result = select_canary(
                    class_tar_root=tar_root,
                    index_csv=index_csv,
                    output_dir=output,
                    selected_index_csv=root / f"selected-{run_number}.csv",
                    output_json=root / f"selection-{run_number}.json",
                    image_count=4,
                    class_count=2,
                )
                self.assertEqual(result["image_count"], 4)
                self.assertEqual(result["images_per_class"], 2)
                self.assertEqual(result["selected_sampling_counts"], {"4:2:0": 2, "4:4:4": 2})
                self.assertEqual(result["rejected_sampling_counts"], {"4:2:2": 2})
                self.assertEqual(len(list(output.rglob("*.JPEG"))), 4)
                self.assertFalse(list(output.rglob("*unsupported*")))
                hashes.append(result["selection_sha256"])
            self.assertEqual(hashes[0], hashes[1])

    def test_manifest_preflight_supports_v2_v3_and_fails_expected_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v2_root = root / "v2"
            v3_root = root / "v3"
            v2_root.mkdir()
            v3_root.mkdir()
            v2 = _write_mock_galp_manifest(v2_root, version=2, image_count=4)
            v3 = _write_mock_galp_manifest(v3_root, version=3, image_count=7)
            (v2_root / "shard_000000.svb").write_bytes(b"legacy-v2-native-input")
            # An undeclared sidecar must not leak a guessed physical-layout
            # dependency into the v3 training preflight contract. Compact-v2's
            # legacy public-reader naming convention remains explicit.
            (v3_root / "shard_000000.svb").write_bytes(b"unannounced-sidecar")
            v2_result = preflight_manifest(v2)
            v3_result = preflight_manifest(v3)
            self.assertEqual(v2_result.physical_layout, "image-major")
            self.assertIsNone(v2_result.declared_physical_layout)
            self.assertIsNone(v2_result.descriptor_kind)
            self.assertEqual(v2_result.as_dict()["payload_file_count"], 3)
            self.assertEqual(
                v2_result.as_dict()["legacy_v2_implicit_payload_file_count"], 1
            )
            self.assertEqual(v3_result.physical_layout, "image-major-vector-rowgroups")
            self.assertEqual(v3_result.declared_physical_layout, v3_result.physical_layout)
            self.assertEqual(v3_result.descriptor_kind, "galp-compact-v1")
            self.assertEqual(v3_result.vector_size, 1024)
            self.assertEqual(v3_result.spatial_order, "tiled-z32")
            self.assertEqual(v3_result.spatial_order_id, 3)
            self.assertEqual(v3_result.rowgroup_vectors, 1)
            self.assertEqual(v3_result.image_count, 7)
            self.assertEqual(v3_result.as_dict()["payload_file_count"], 2)
            self.assertEqual(
                v3_result.as_dict()["legacy_v2_implicit_payload_file_count"], 0
            )
            with self.assertRaisesRegex(ManifestPreflightError, "version mismatch"):
                preflight_manifest(v3, expected_manifest_version=2)
            with self.assertRaisesRegex(ManifestPreflightError, "physical layout mismatch"):
                preflight_manifest(v2, expected_physical_layout="image-major-vector-rowgroups")
            with self.assertRaisesRegex(ManifestPreflightError, "spatial order mismatch"):
                preflight_manifest(v3, expected_spatial_order="raster")
            with self.assertRaisesRegex(ManifestPreflightError, "image count mismatch"):
                preflight_manifest(v3, expected_image_count=8)

    def test_manifest_preflight_requires_the_canonical_v3_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_root = root / "missing"
            invalid_root = root / "invalid"
            v2_root = root / "v2"
            missing_root.mkdir()
            invalid_root.mkdir()
            v2_root.mkdir()
            manifest = _write_mock_galp_manifest(
                missing_root,
                version=3,
                include_v3_extension=False,
            )
            with self.assertRaisesRegex(
                ManifestPreflightError,
                "version 3 requires.*Compact-v3 descriptor extension",
            ):
                preflight_manifest(manifest)
            invalid = _write_mock_galp_manifest(
                invalid_root,
                version=3,
                v3_descriptor_kind="noncanonical-v3",
            )
            with self.assertRaisesRegex(
                ManifestPreflightError,
                "non-canonical descriptor kind",
            ):
                preflight_manifest(invalid)
            v2 = _write_mock_galp_manifest(v2_root, version=2)
            v2.write_bytes(v2.read_bytes() + b"GJDCCV31")
            with self.assertRaisesRegex(
                ManifestPreflightError,
                "unexpected GALP manifest extension for version 2",
            ):
                preflight_manifest(v2)

    def test_manifest_preflight_rejects_corruption_before_reader_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_mock_galp_manifest(root, version=3)
            data = bytearray(manifest.read_bytes())
            data[:8] = b"BADMAGIC"
            manifest.write_bytes(data)
            with self.assertRaisesRegex(ManifestPreflightError, "magic mismatch"):
                preflight_manifest(manifest)

    def test_galp_layout_manifest_rebinding_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_root = root / "samples"
            v2_root = root / "v2"
            v3_root = root / "v3"
            sample_root.mkdir()
            v2_root.mkdir()
            v3_root.mkdir()
            samples = _samples(sample_root)
            v2 = _write_mock_galp_manifest(v2_root, version=2, image_count=4)
            v3 = _write_mock_galp_manifest(v3_root, version=3, image_count=4)
            metadata = {"declared_galp_manifest": str(v3)}
            preflight = preflight_manifest(v2)

            with self.assertRaisesRegex(ValueError, "generated for GALP manifest"):
                _validate_galp_dataset_binding(
                    samples, metadata, v2, preflight, split="train"
                )
            _validate_galp_dataset_binding(
                samples,
                metadata,
                v2,
                preflight,
                split="train",
                allow_layout_manifest_rebinding=True,
            )

    def test_cli_manifest_expectation_mismatch_fails_during_argument_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_mock_galp_manifest(root, version=3)
            args = _parse_args(
                [
                    "--pipeline",
                    "galp",
                    "--train-manifest",
                    str(root / "train.json"),
                    "--val-manifest",
                    str(root / "val.json"),
                    "--galp-manifest",
                    str(manifest),
                    "--expected-manifest-version",
                    "2",
                    "--output-dir",
                    str(root / "output"),
                    "--workers",
                    "1",
                    "--device",
                    "cpu",
                    "--dry-run-contract",
                ]
            )
            with self.assertRaisesRegex(ManifestPreflightError, "version mismatch"):
                _validate_args(args)

    def test_independent_galp_train_and_validation_manifests_are_preflighted_and_routed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_root = root / "train-dct"
            validation_root = root / "validation-dct"
            train_root.mkdir()
            validation_root.mkdir()
            train_manifest = _write_mock_galp_manifest(
                train_root, version=3, image_count=7
            )
            validation_manifest = _write_mock_galp_manifest(
                validation_root, version=3, image_count=5
            )
            args = _parse_args(
                [
                    "--pipeline",
                    "galp",
                    "--train-manifest",
                    str(root / "train.json"),
                    "--val-manifest",
                    str(root / "val.json"),
                    "--galp-manifest",
                    str(train_manifest),
                    "--galp-validation-manifest",
                    str(validation_manifest),
                    "--expected-manifest-version",
                    "3",
                    "--expected-physical-layout",
                    "image-major-vector-rowgroups",
                    "--expected-spatial-order",
                    "tiled-z32",
                    "--expected-image-count",
                    "7",
                    "--expected-validation-image-count",
                    "5",
                    "--output-dir",
                    str(root / "output"),
                    "--workers",
                    "1",
                    "--device",
                    "cpu",
                    "--dry-run-contract",
                ]
            )
            _validate_args(args)
            self.assertEqual(args.galp_manifest_preflight.image_count, 7)
            self.assertEqual(args.galp_validation_manifest_preflight.image_count, 5)
            contract = {
                "pipelines": {
                    "galp_manifest": str(train_manifest),
                    "galp_validation_manifest": str(validation_manifest),
                }
            }
            self.assertEqual(
                _adapter_pipeline_config(contract, args, split="train")[
                    "galp_manifest"
                ],
                str(train_manifest),
            )
            self.assertEqual(
                _adapter_pipeline_config(contract, args, split="validation")[
                    "galp_manifest"
                ],
                str(validation_manifest),
            )

    def test_public_direct_dct_training_reader_is_layout_blind(self) -> None:
        constructed: list[str] = []

        class NativeReader:
            def __init__(self, path: str) -> None:
                constructed.append(path)
                self.image_count = 9

            def pipeline(self, profile_id, *, dct_coeffs="all"):
                self.assert_dct_coeffs = dct_coeffs
                reader = self

                class NativePipeline:
                    ready = True
                    started = True
                    prefetch_metrics = {
                        "producer_ms": 0.0,
                        "planning_ms": 0.0,
                        "io_ms": 0.0,
                        "ordered_submission_ms": 0.0,
                    }

                    def reset(self, batches, *, transforms_by_batch):
                        reader.assert_profile_id = profile_id
                        self.batch = _FakeNativeTrainingBatch(
                            list(batches[0]),
                            list(transforms_by_batch[0]),
                            {"future_counter": 11},
                        )
                        self.consumed = False

                    def __next__(self):
                        if self.consumed:
                            raise StopIteration
                        self.consumed = True
                        return self.batch

                    def close(self):
                        return 0

                return NativePipeline()

        native_module = SimpleNamespace(
            DirectDctReader=NativeReader,
            DIRECT_DCT_BINDING_SCHEMA="galp-direct-dct-binding-v2",
            DIRECT_DCT_PROFILE_SCHEMA="galp-direct-dct-profile-v1",
            DIRECT_DCT_METRICS_SCHEMA="galp-direct-dct-metrics-v2",
            direct_dct_profile_info=lambda profile_id: {
                "schema": "galp-direct-dct-profile-v1",
                "id": profile_id,
                "runtime_policy_id": "compact-v3-planless-limited-o512-c512-v1",
            },
        )
        for name in ("v2.bin", "v3.bin"):
            reader = DirectDctTrainingReader(Path(name), native_module=native_module)
            self.assertEqual(reader.image_count, 9)
            reader.start(
                [[3]], transforms_by_batch=[[{"global_image_id": 3}]]
            )
            batch = reader.next_batch()
            self.assertEqual(batch.global_image_ids, [3])
            self.assertEqual(batch.native_execution_stats()["future_counter"], 11)
            self.assertEqual(
                batch.native_execution_stats_snapshot()["future_counter"], 11
            )
        self.assertEqual(len(constructed), 2)

    def test_public_direct_dct_training_reader_rejects_runtime_policy_mismatch(
        self,
    ) -> None:
        class NativeReader:
            def __init__(self, path: str) -> None:
                self.image_count = 9

        native_module = SimpleNamespace(
            DirectDctReader=NativeReader,
            DIRECT_DCT_BINDING_SCHEMA="galp-direct-dct-binding-v2",
            DIRECT_DCT_PROFILE_SCHEMA="galp-direct-dct-profile-v1",
            DIRECT_DCT_METRICS_SCHEMA="galp-direct-dct-metrics-v2",
            direct_dct_profile_info=lambda profile_id: {
                "schema": "galp-direct-dct-profile-v1",
                "id": profile_id,
                "runtime_policy_id": "alternate-runtime-policy-v1",
            },
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "native profile does not match the training contract",
        ):
            DirectDctTrainingReader(Path("v3.bin"), native_module=native_module)

    def test_repository_v2_v3_manifests_open_through_same_public_reader(self) -> None:
        repository = BENCHMARK_DIR.parents[2]
        module_path = repository / "build/galp/torch"
        manifests = (
            (
                2,
                "image-major",
                50_000,
                repository / "galp/data/compressed/imagenet_original_val_image_major_v2/manifest.bin",
            ),
            (
                3,
                "image-major-vector-rowgroups",
                1_000,
                repository
                / "galp/data/compressed/fixtures/imagenet512_train_1000_compact_v3/dct/manifest.bin",
            ),
        )
        if not module_path.is_dir() or any(not path.is_file() for *_fields, path in manifests):
            self.skipTest("repository v2/v3 data or built public reader is unavailable")
        for version, layout, image_count, path in manifests:
            preflight = preflight_manifest(
                path,
                expected_manifest_version=version,
                expected_physical_layout=layout,
                expected_image_count=image_count,
            )
            reader = DirectDctTrainingReader(path, module_path=module_path)
            self.assertEqual(preflight.image_count, image_count)
            self.assertEqual(reader.image_count, image_count)

    def test_v2_and_v3_preflight_share_one_galp_training_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = _samples(root, 2)
            identities = [
                SampleIdentity(0, index, sample.logical_sample_id)
                for index, sample in enumerate(samples)
            ]
            decisions = [
                derive_augmentation(
                    seed=29,
                    epoch=0,
                    logical_sample_id=sample.logical_sample_id,
                    source_width=sample.width,
                    source_height=sample.height,
                    domain="dct",
                )
                for sample in samples
            ]
            semantic_outputs = []
            for version in (2, 3):
                manifest_root = root / f"v{version}"
                manifest_root.mkdir()
                manifest = _write_mock_galp_manifest(
                    manifest_root, version=version, image_count=2
                )
                preflight = preflight_manifest(manifest)
                reader = _FakeDirectDctTrainingReader(
                    {"manifest_version_for_test_only": version}
                )
                adapter = GalpTrainingAdapter(
                    samples,
                    batch_size=2,
                    workers=1,
                    device=torch.device("cpu"),
                    config={
                        "_direct_dct_training_reader": reader,
                        "execution_mode": "audit",
                    },
                )
                adapter.begin(identities, decisions, [2])
                batch = adapter.next_batch()
                semantic_outputs.append(
                    (
                        [identity.as_dict() for identity in batch.identities],
                        batch.augmentations,
                        [tuple(tensor.shape) for tensor in batch.inputs],
                    )
                )
                self.assertEqual(preflight.version, version)
                self.assertEqual(reader.requests[0][0], [0, 1])
                adapter.close()
            self.assertEqual(semantic_outputs[0], semantic_outputs[1])

    def test_galp_training_adapter_uses_same_reader_surface_and_optional_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 2)
            identities = [
                SampleIdentity(0, index, sample.logical_sample_id)
                for index, sample in enumerate(samples)
            ]
            decisions = [
                derive_augmentation(
                    seed=17,
                    epoch=0,
                    logical_sample_id=sample.logical_sample_id,
                    source_width=sample.width,
                    source_height=sample.height,
                    domain="dct",
                )
                for sample in samples
            ]
            for stats in (None, {"decode_ms": 2.5, "future_counter": 13}):
                reader = _FakeDirectDctTrainingReader(stats)
                adapter = GalpTrainingAdapter(
                    samples,
                    batch_size=2,
                    workers=1,
                    device=torch.device("cpu"),
                    config={
                        "_direct_dct_training_reader": reader,
                        "execution_mode": "audit",
                    },
                )
                adapter.begin(identities, decisions, [2])
                batch = adapter.next_batch()
                self.assertEqual(batch.identities, identities)
                self.assertEqual([tuple(value.shape) for value in batch.inputs], [
                    (2, 1, 28, 28, 8, 8),
                    (2, 2, 14, 14, 8, 8),
                ])
                self.assertEqual(batch.native_execution_stats, stats or {})
                self.assertEqual(reader.requests[0][0], [0, 1])
                adapter.close()

    def test_galp_runtime_metrics_use_nonblocking_native_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 2)
            identities = [
                SampleIdentity(0, index, sample.logical_sample_id)
                for index, sample in enumerate(samples)
            ]
            decisions = [
                derive_augmentation(
                    seed=19,
                    epoch=0,
                    logical_sample_id=sample.logical_sample_id,
                    source_width=sample.width,
                    source_height=sample.height,
                    domain="dct",
                )
                for sample in samples
            ]
            reader = _FakeDirectDctTrainingReader(
                {
                    "planning_ms": 3.5,
                    "projection_ms": 1.25,
                    "galp_native_device_cuda_allocation_count": 7,
                }
            )
            adapter = GalpTrainingAdapter(
                samples,
                batch_size=2,
                workers=1,
                device=torch.device("cpu"),
                config={
                    "_direct_dct_training_reader": reader,
                    "execution_mode": "runtime",
                },
            )
            adapter.begin(identities, decisions, [2])
            batch = adapter.next_batch()
            self.assertFalse(batch.native_gpu_timings_finalized)
            adapter.snapshot_batch_metrics(batch)
            self.assertTrue(batch.native_host_snapshot_taken)
            self.assertFalse(batch.native_gpu_timings_finalized)
            self.assertEqual(batch.native_execution_stats["planning_ms"], 3.5)
            self.assertEqual(batch.stage_seconds["preprocess"], 0.00125)
            self.assertEqual(reader.batches[0].complete_stats_reads, 0)
            self.assertEqual(reader.batches[0].snapshot_stats_reads, 1)
            adapter.finalize_batch_metrics(batch)
            self.assertTrue(batch.native_gpu_timings_finalized)
            self.assertEqual(reader.batches[0].complete_stats_reads, 1)
            adapter.close()

    def test_galp_prefetch_evidence_tracks_only_native_accepted_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 8)
            identities = [
                SampleIdentity(0, index, sample.logical_sample_id)
                for index, sample in enumerate(samples)
            ]
            decisions = [
                derive_augmentation(
                    seed=23,
                    epoch=0,
                    logical_sample_id=sample.logical_sample_id,
                    source_width=sample.width,
                    source_height=sample.height,
                    domain="dct",
                )
                for sample in samples
            ]
            adapter = GalpTrainingAdapter(
                samples,
                batch_size=2,
                workers=0,
                device=torch.device("cpu"),
                config={
                    "_direct_dct_training_reader": _FakeDirectDctTrainingReader(),
                    "execution_mode": "runtime",
                },
            )
            adapter.begin(identities, decisions, [2, 2, 2, 2])
            adapter.next_batch()
            self.assertEqual(adapter.prefetched_read_identities(), identities[:6])
            self.assertNotEqual(adapter.prefetched_read_identities(), identities)
            adapter.close()
            self.assertEqual(adapter.prefetched_read_identities(), identities[:6])
            self.assertTrue(adapter.loader_metrics()["native_metrics_complete"])
            self.assertTrue(adapter.loader_metrics()["closed"])


    def test_galp_workers_do_not_change_native_pipeline_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 6)
            identities = canonical_epoch_order(
                [sample.logical_sample_id for sample in samples], 71, 0
            )
            by_id = {sample.logical_sample_id: sample for sample in samples}
            decisions = [
                derive_augmentation(
                    seed=71,
                    epoch=identity.epoch,
                    logical_sample_id=identity.logical_sample_id,
                    source_width=by_id[identity.logical_sample_id].width,
                    source_height=by_id[identity.logical_sample_id].height,
                    domain="dct",
                )
                for identity in identities
            ]
            observed = []
            for workers in (1, 4):
                reader = _FakeDirectDctTrainingReader(
                    {"future_counter": workers}
                )
                adapter = GalpTrainingAdapter(
                    samples,
                    batch_size=2,
                    workers=workers,
                    device=torch.device("cpu"),
                    config={
                        "_direct_dct_training_reader": reader,
                        "execution_mode": "audit",
                    },
                )
                adapter.begin(identities, decisions, [2, 2, 2])
                emitted = [adapter.next_batch() for _ in range(3)]
                observed.append(
                    (
                        [
                            identity.as_dict()
                            for batch in emitted
                            for identity in batch.identities
                        ],
                        [
                            descriptor
                            for batch in emitted
                            for descriptor in batch.augmentations
                        ],
                    )
                )
                self.assertTrue(adapter.loader_metrics()["native_pipeline_owned"])
                self.assertEqual(
                    adapter.loader_metrics()["max_queue_depth_batches"], 0
                )
                adapter.close()
            self.assertTrue(all(value == observed[0] for value in observed[1:]))

    def test_native_stats_accept_missing_and_future_fields(self) -> None:
        accumulator = NativeExecutionStatsAccumulator()
        accumulator.observe(None)
        accumulator.observe({})
        accumulator.observe({"decode_ms": 1.25, "future_counter": 7, "note": "ok"})
        accumulator.observe({"decode_ms": 2.75, "future_counter": 9})
        result = accumulator.as_dict()
        self.assertTrue(result["optional"])
        self.assertFalse(result["correctness_dependency"])
        self.assertEqual(result["missing_batches"], 2)
        self.assertEqual(result["numeric_aggregates"]["future_counter"]["sum"], 16.0)

    def test_native_allocator_snapshot_accounting_does_not_sum_global_totals(self) -> None:
        aggregate: dict[str, float] = {}
        merge_native_counter_snapshot(
            aggregate,
            {
                "galp_native_device_cuda_allocation_count": 7,
                "compact_batch_buffer_growth_count": 1,
            },
        )
        merge_native_counter_snapshot(
            aggregate,
            {
                "galp_native_device_cuda_allocation_count": 9,
                "compact_batch_buffer_growth_count": 2,
            },
        )
        self.assertEqual(aggregate["galp_native_device_cuda_allocation_count"], 9)
        self.assertEqual(aggregate["compact_batch_buffer_growth_count"], 3)

    def test_native_allocation_stability_is_measured_from_warmup_boundary(self) -> None:
        warmup = [
            {
                "galp_native_device_allocation_requests": 11,
                "galp_native_device_cuda_allocation_count": 4,
                "galp_native_device_cuda_allocation_bytes": 4096,
                "galp_native_pinned_allocation_requests": 7,
                "galp_native_pinned_cuda_allocation_count": 3,
                "galp_native_pinned_cuda_allocation_bytes": 2048,
                "compact_batch_buffer_growth_count": 1,
                "compact_batch_buffer_pageable_fallback_count": 0,
                "decode_workset_output_arena_growth_count": 1,
                "decode_workset_chunk_arena_growth_count": 1,
                "planless_axis_program_device_growth_count": 1,
                "planless_axis_program_pinned_growth_count": 1,
                "planless_axis_program_capacity_contract_complete": True,
                "compact_batch_pool_capacity_contract_complete": True,
            }
        ]
        measured = [
            {
                **warmup[-1],
                "galp_native_device_allocation_requests": 15,
                "galp_native_pinned_allocation_requests": 9,
                "compact_batch_buffer_growth_count": 0,
                "decode_workset_output_arena_growth_count": 0,
                "decode_workset_chunk_arena_growth_count": 0,
                "planless_axis_program_device_growth_count": 0,
                "planless_axis_program_pinned_growth_count": 0,
            },
            {
                **warmup[-1],
                "galp_native_device_allocation_requests": 17,
                "galp_native_pinned_allocation_requests": 10,
                "compact_batch_buffer_growth_count": 0,
                "decode_workset_output_arena_growth_count": 0,
                "decode_workset_chunk_arena_growth_count": 0,
                "planless_axis_program_device_growth_count": 0,
                "planless_axis_program_pinned_growth_count": 0,
            },
        ]
        stable = native_allocation_stability(warmup, measured)
        self.assertTrue(stable["verifiable"])
        self.assertTrue(stable["stable_after_warmup"])
        self.assertTrue(stable["capacity_contract_complete"])
        self.assertEqual(
            stable["global_counter_deltas"]["galp_native_device_allocation_requests"],
            6.0,
        )

        measured[-1]["galp_native_pinned_cuda_allocation_count"] = 4
        unstable = native_allocation_stability(warmup, measured)
        self.assertFalse(unstable["stable_after_warmup"])

        measured[-1]["galp_native_pinned_cuda_allocation_count"] = 3
        measured[-1]["planless_axis_program_capacity_contract_complete"] = False
        incomplete = native_allocation_stability(warmup, measured)
        self.assertTrue(incomplete["verifiable"])
        self.assertFalse(incomplete["capacity_contract_complete"])
        self.assertFalse(incomplete["stable_after_warmup"])


    def test_process_memory_includes_repeat_resource_gauges(self) -> None:
        snapshot = process_memory()
        self.assertIn("rss_bytes", snapshot)
        self.assertIn("peak_rss_bytes", snapshot)
        self.assertIn("virtual_memory_bytes", snapshot)
        self.assertIn("thread_count", snapshot)
        self.assertIn("open_fd_count", snapshot)
        self.assertIn("memory_mapping_count", snapshot)
        if Path("/proc/self/status").is_file():
            self.assertGreater(int(snapshot["rss_bytes"] or 0), 0)
            self.assertGreater(int(snapshot["open_fd_count"] or 0), 0)
            self.assertGreater(int(snapshot["memory_mapping_count"] or 0), 0)





    def test_training_import_boundary_does_not_reach_core_internals(self) -> None:
        training_root = BENCHMARK_DIR / "training"
        banned_prefixes = (
            "galp.src.format",
            "galp.src.jpeg",
            "galp.jpeg",
            "galp.format",
        )
        for path in training_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                any(name.startswith(banned_prefixes) for name in imported),
                f"{path.name} imports a core format/jpeg internal: {imported}",
            )
        generator_source = (training_root / "generate_imagenet_manifests.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from galp.benchmarks.system_rgbnomore.training.manifest_preflight import preflight_manifest",
            generator_source,
        )
        self.assertNotIn("GJDCTSH1", generator_source)

    def test_dali_tensor_list_is_collapsed_before_torch_allocation(self) -> None:
        tensor_list = _FakeDaliTensorList()
        tensor, shape = _uniform_dali_tensor(tensor_list)
        self.assertIs(tensor, tensor_list.tensor)
        self.assertEqual(shape, [np.int64(2), np.int32(3), 4, 4])
        self.assertEqual(tuple(torch.empty(shape).shape), (2, 3, 4, 4))

    def test_execution_mode_cli_default_and_explicit_runtime(self) -> None:
        default = _parse_args([])
        runtime = _parse_args(["--execution-mode", "runtime"])
        self.assertEqual(default.execution_mode, "audit")
        self.assertFalse(default.execution_mode_explicit)
        self.assertGreaterEqual(default.dct_semantic_atol, 1.0 / 1020.0)
        self.assertLess(default.dct_semantic_atol, 2.0 / 1020.0)
        self.assertEqual(runtime.execution_mode, "runtime")
        self.assertTrue(runtime.execution_mode_explicit)
        self.assertFalse(hasattr(default, "prefetch_depth"))
        with self.assertRaises(SystemExit):
            _parse_args(["--prefetch-depth", "4"])
        self.assertEqual(_first_step_probe_policy("audit"), "audit")
        self.assertEqual(_first_step_probe_policy("runtime"), "audit")

    def test_semantic_and_performance_gate_arguments_reject_invalid_values(self) -> None:
        base = [
            "--pipeline",
            "pytorch",
            "--train-manifest",
            "/unused-train.json",
            "--val-manifest",
            "/unused-val.json",
            "--output-dir",
            "/unused-output",
            "--device",
            "cpu",
            "--dry-run-contract",
        ]
        for option, value, message in (
            ("--dct-semantic-atol", "nan", "finite and non-negative"),
            ("--throughput-cv-limit", "-0.1", "finite and non-negative"),
            ("--semantic-gradient-cosine-dct", "1.1", "finite and in"),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, message):
                _validate_args(_parse_args([*base, option, value]))
        with self.assertRaisesRegex(ValueError, "must be at least"):
            _validate_args(
                _parse_args(
                    [
                        *base,
                        "--rgb-semantic-warning-atol",
                        "0.5",
                        "--rgb-semantic-failure-atol",
                        "0.1",
                    ]
                )
            )

    def test_sync_ledger_records_every_explicit_reason(self) -> None:
        ledger = _SyncLedger()
        device = SimpleNamespace(type="cuda")
        with mock.patch("galp.benchmarks.system_rgbnomore.training.run.torch.cuda.synchronize") as synchronize:
            _sync(device, ledger, "measured_region_start")
            _sync(device, ledger, "measured_region_end")
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(
            ledger.as_dict(),
            {
                "count": 2,
                "reasons": {
                    "measured_region_end": 1,
                    "measured_region_start": 1,
                },
            },
        )

    def test_resume_rejects_cross_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "contract.json").write_text(
                json.dumps({"execution": {"mode": "audit"}}), encoding="utf-8"
            )
            args = _parse_args(
                ["--resume-run", str(root), "--execution-mode", "runtime"]
            )
            with self.assertRaisesRegex(ValueError, "immutable contract mode"):
                _load_resume(args)

    def test_runtime_step_uses_shared_finite_gate_without_deep_scans(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1000)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        identities = [SampleIdentity(0, 0, "a"), SampleIdentity(0, 1, "b")]
        batch = TrainingBatch(
            inputs=(torch.randn(2, 3, 224, 224),),
            labels=torch.tensor([1, 2]),
            identities=identities,
            augmentations=[{}, {}],
            on_device=False,
        )
        before = model[-1].weight.detach().clone()
        with (
            mock.patch("galp.benchmarks.system_rgbnomore.training.run.gradient_summary") as gradient_scan,
            mock.patch("galp.benchmarks.system_rgbnomore.training.run.parameter_update_summary") as parameter_scan,
            mock.patch("galp.benchmarks.system_rgbnomore.training.run._sync") as explicit_sync,
        ):
            record, emitted = _train_one_step_runtime(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=_OneBatchAdapter(batch),
                expected=identities,
                domain="rgb",
                device=torch.device("cpu"),
                label_smoothing=0.0,
                gradient_clipping=None,
                audit=TrainingAuditState(
                    TrainingAuditPolicy(mode="benchmark", strict_updates=100),
                    completed_updates=100,
                    device=torch.device("cpu"),
                ),
            )
        gradient_scan.assert_not_called()
        parameter_scan.assert_not_called()
        explicit_sync.assert_not_called()
        self.assertIs(emitted, batch)
        self.assertEqual(record["deep_parameter_scans"], 0)
        self.assertEqual(record["host_scalar_materializations_in_step"], 1)
        self.assertEqual(record["training_audit"]["gradient_gate_reads"], 1)
        self.assertIn("loss_tensor", record)
        _resolve_runtime_record(record)
        self.assertFalse(torch.equal(before, model[-1].weight.detach()))

    def test_audit_and_runtime_use_identical_first_step_probe_outputs(self) -> None:
        torch.manual_seed(37)
        source = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1000)
        )
        initial = copy.deepcopy(source.state_dict())
        identities = [SampleIdentity(0, 0, "probe")]
        inputs = torch.randn(1, 3, 224, 224)
        outputs = []
        for mode in ("audit", "runtime"):
            self.assertEqual(_first_step_probe_policy(mode), "audit")
            model = copy.deepcopy(source)
            model.load_state_dict(initial)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _step: 1.0
            )
            batch = TrainingBatch(
                inputs=(inputs.clone(),),
                labels=torch.tensor([7]),
                identities=identities,
                augmentations=[{"augmentation_key": "same"}],
                on_device=False,
            )
            record, *_ = _train_one_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=_OneBatchAdapter(batch),
                expected=identities,
                domain="rgb",
                device=torch.device("cpu"),
                label_smoothing=0.0,
                gradient_clipping=None,
                collect_numerics=True,
            )
            outputs.append(
                (
                    record["loss"],
                    record["gradients"]["global_l2_norm"],
                    record["parameter_update"]["global_l2_norm"],
                    tensor_state_sha256(model.state_dict()),
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_galp_workers_no_longer_configure_native_rowgroup_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.bin"
            manifest.write_bytes(b"fixture")
            args = _parse_args(
                [
                    "--pipeline",
                    "galp",
                    "--train-manifest",
                    str(root / "train.json"),
                    "--val-manifest",
                    str(root / "val.json"),
                    "--galp-manifest",
                    str(manifest),
                    "--output-dir",
                    str(root / "out"),
                    "--workers",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run-contract",
                ]
            )
            self.assertEqual(args.workers, 0)

    def test_formal_rgb_and_dct_models_execute_real_optimizer_steps(self) -> None:
        if not RGBNOMORE_ROOT.is_dir():
            self.skipTest("external RGB-no-more checkout is unavailable")
        seed_everything(7)
        for domain in ("rgb", "dct"):
            model = build_model(RGBNOMORE_ROOT, domain, torch.device("cpu"))
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                EXPECTED_PARAMETER_COUNTS[domain],
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            before = next(model.parameters()).detach().clone()
            if domain == "rgb":
                logits = model(torch.randn(1, 3, 224, 224))
            else:
                logits = model(
                    torch.randn(1, 1, 28, 28, 8, 8),
                    torch.randn(1, 2, 14, 14, 8, 8),
                )
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
            loss.backward()
            summary = gradient_summary(model)
            optimizer.step()
            self.assertEqual(tuple(logits.shape), (1, 1000))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(summary["finite"])
            self.assertTrue(summary["nonzero"])
            self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
            del model, optimizer, logits, loss
            gc.collect()

    def test_training_model_registry_exposes_swinv2_rgb_and_dct_pair(self) -> None:
        self.assertIn(SWINV2_T_MODEL_ID, MODEL_IDS)
        rgb = model_configuration("rgb", SWINV2_T_MODEL_ID)
        dct = model_configuration("dct", SWINV2_T_MODEL_ID)
        self.assertEqual(rgb["architecture"], dct["architecture"])
        self.assertEqual(rgb["window_size"], 7)
        self.assertEqual(rgb["image_size"], 224)
        self.assertEqual(rgb["expected_trainable_parameters"], 28_347_154)
        self.assertEqual(dct["expected_trainable_parameters"], 28_344_850)
        self.assertIsNone(rgb["dct_stem"])
        self.assertEqual(dct["dct_stem"], "grouped-subblock-ycbcr-v1")

    def test_tiny_training_step_checks_loss_gradient_and_update(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1000)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        identities = [SampleIdentity(0, 0, "a"), SampleIdentity(0, 1, "b")]
        batch = TrainingBatch(
            inputs=(torch.randn(2, 3, 224, 224),),
            labels=torch.tensor([1, 2]),
            identities=identities,
            augmentations=[{}, {}],
            on_device=False,
        )
        record, *_rest = _train_one_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            adapter=_OneBatchAdapter(batch),
            expected=identities,
            domain="rgb",
            device=torch.device("cpu"),
            label_smoothing=0.0,
            gradient_clipping=None,
            collect_numerics=True,
        )
        self.assertTrue(record["loss_finite"])
        self.assertTrue(record["gradients"]["finite"])
        self.assertTrue(record["gradients"]["nonzero"])
        self.assertTrue(record["parameter_update"]["changed"])
        self.assertFalse(record["batch_failures"])

    def test_optimizer_scheduler_and_rng_reset_are_exact(self) -> None:
        seed_everything(19)
        model = torch.nn.Linear(4, 2)
        optimizer, groups = build_optimizer(model, _optimizer_config())
        scheduler = build_scheduler(optimizer, {"type": "cosine", "warmup_steps": 1}, total_steps=4)
        initial = capture_training_state(model, optimizer, scheduler)
        initial.update(sample_order_cursor={"epoch": 0, "position": -1}, augmentation_state={"stateless_keyed": True})
        expected_after_reset = torch.rand(4)
        torch.nn.functional.cross_entropy(model(torch.randn(2, 4)), torch.tensor([0, 1])).backward()
        optimizer.step()
        scheduler.step()
        reset_training_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, initial=initial)
        self.assertEqual(tensor_state_sha256(model.state_dict()), tensor_state_sha256(initial["model"]))
        self.assertEqual(nested_state_sha256(optimizer.state_dict()), nested_state_sha256(initial["optimizer"]))
        self.assertEqual(nested_state_sha256(scheduler.state_dict()), nested_state_sha256(initial["scheduler"]))
        torch.testing.assert_close(torch.rand(4), expected_after_reset)
        self.assertEqual([group["stable_id"] for group in groups], ["decay", "no_decay"])

    def test_weight_checkpoint_load_is_strict_and_full_checkpoint_does_not_degrade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pt"
            source = torch.nn.Linear(4, 2)
            torch.save(
                {
                    "model_architecture": "rgbnomore-vitti-v1",
                    "model_domain": "rgb",
                    "model_state_dict": source.state_dict(),
                },
                path,
            )
            target = torch.nn.Linear(4, 2)
            provenance = initialize_model(
                target, init_mode="weights", checkpoint=path, domain="rgb"
            )
            self.assertTrue(provenance["checkpoint"]["strict_load"])
            self.assertEqual(provenance["checkpoint"]["missing_keys"], [])
            self.assertEqual(tensor_state_sha256(target.state_dict()), tensor_state_sha256(source.state_dict()))
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                initialize_model(
                    torch.nn.Linear(4, 2),
                    init_mode="full-checkpoint",
                    checkpoint=path,
                    domain="rgb",
                )
            optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
            full_path = Path(temporary) / "full.pt"
            torch.save(
                {
                    "model_architecture": "rgbnomore-vitti-v1",
                    "model_domain": "rgb",
                    "model_configuration": model_configuration("rgb"),
                    "optimizer_configuration": {"type": "sgd"},
                    "scheduler_configuration": {"type": "constant"},
                    "scheduler_total_steps": 10,
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": None,
                    "global_step": 8,
                    "epoch": 2,
                    "rng_state": capture_rng_state(),
                    "augmentation_state": {"stateless_keyed": True},
                    "sample_order_cursor": {
                        "epoch": 2,
                        "position": 3,
                        "logical_sample_id": "fixture",
                    },
                },
                full_path,
            )
            full_provenance = initialize_model(
                torch.nn.Linear(4, 2),
                init_mode="full-checkpoint",
                checkpoint=full_path,
                domain="rgb",
            )
            self.assertEqual(full_provenance["convergence_classification"], "resumed_training")
            self.assertEqual(full_provenance["_full_checkpoint_payload"]["global_step"], 8)
            with self.assertRaises(RuntimeError):
                initialize_model(
                    torch.nn.Linear(5, 2), init_mode="weights", checkpoint=path, domain="rgb"
                )

    def test_full_checkpoint_resume_matches_uninterrupted_training_exactly(self) -> None:
        samples = [
            TrainingSample(
                f"id-{index}",
                Path(f"/unused-{index}.jpg"),
                index % 3,
                256,
                256,
                index,
            )
            for index in range(8)
        ]
        batches, _ = _collect_batches(
            samples,
            seed=53,
            batch_size=2,
            batch_count=4,
            drop_last=True,
        )
        cursor = batches[1][-1].as_dict()
        resumed_batches, _ = _collect_batches(
            samples,
            seed=53,
            batch_size=2,
            batch_count=2,
            drop_last=True,
            start_cursor=cursor,
        )
        self.assertEqual(resumed_batches, batches[2:])

        sample_map = {sample.logical_sample_id: sample for sample in samples}
        uninterrupted_decisions = _augmentation_batches(
            batches, sample_map, seed=53, domain="rgb"
        )
        resumed_decisions = _augmentation_batches(
            resumed_batches, sample_map, seed=53, domain="rgb"
        )
        self.assertEqual(resumed_decisions, uninterrupted_decisions[2:])

        def make_training_state():
            model = torch.nn.Linear(4, 3)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.05, momentum=0.9
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9
            )
            return model, optimizer, scheduler

        def advance(model, optimizer, scheduler, planned_batches):
            for planned in planned_batches:
                inputs = torch.randn(len(planned), 4)
                labels = torch.tensor(
                    [identity.position % 3 for identity in planned], dtype=torch.long
                )
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(inputs), labels).backward()
                optimizer.step()
                scheduler.step()

        seed_everything(20260731)
        model, optimizer, scheduler = make_training_state()
        initial = capture_training_state(model, optimizer, scheduler)

        advance(model, optimizer, scheduler, batches)
        uninterrupted = (
            tensor_state_sha256(model.state_dict()),
            nested_state_sha256(optimizer.state_dict()),
            nested_state_sha256(scheduler.state_dict()),
        )
        uninterrupted_next_random = torch.rand(8)

        reset_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            initial=initial,
        )
        advance(model, optimizer, scheduler, batches[:2])

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "resume.pt"
            _save_training_checkpoint(
                checkpoint,
                domain="rgb",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=2,
                epoch=cursor["epoch"],
                sample_order_cursor=cursor,
                optimizer_configuration={"type": "sgd"},
                scheduler_configuration={"type": "step"},
                scheduler_total_steps=4,
            )

            resumed_model, resumed_optimizer, resumed_scheduler = make_training_state()
            provenance = initialize_model(
                resumed_model,
                init_mode="full-checkpoint",
                checkpoint=checkpoint,
                domain="rgb",
            )
            payload = provenance["_full_checkpoint_payload"]
            resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
            resumed_scheduler.load_state_dict(payload["scheduler_state_dict"])
            restore_rng_state(payload["rng_state"])
            self.assertEqual(payload["sample_order_cursor"], cursor)
            self.assertEqual(payload["global_step"], 2)

            advance(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                resumed_batches,
            )
            resumed = (
                tensor_state_sha256(resumed_model.state_dict()),
                nested_state_sha256(resumed_optimizer.state_dict()),
                nested_state_sha256(resumed_scheduler.state_dict()),
            )
            self.assertEqual(resumed, uninterrupted)
            self.assertTrue(torch.equal(torch.rand(8), uninterrupted_next_random))

    def test_pytorch_and_rgbnomore_adapters_use_one_planned_interface(self) -> None:
        if not RGBNOMORE_ROOT.is_dir():
            self.skipTest("external RGB-no-more checkout is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 2)
            identities = [SampleIdentity(0, index, sample.logical_sample_id) for index, sample in enumerate(samples)]
            for adapter_class, domain, expected_shapes in (
                (PyTorchTrainingAdapter, "rgb", ((2, 3, 224, 224),)),
                (RgbNoMoreTrainingAdapter, "dct", ((2, 1, 28, 28, 8, 8), (2, 2, 14, 14, 8, 8))),
            ):
                decisions = [
                    derive_augmentation(
                        seed=5,
                        epoch=0,
                        logical_sample_id=sample.logical_sample_id,
                        source_width=sample.width,
                        source_height=sample.height,
                        domain=domain,
                    )
                    for sample in samples
                ]
                adapter = adapter_class(
                    samples,
                    batch_size=2,
                    workers=0,
                    device=torch.device("cpu"),
                    config={"rgbnomore_root": str(RGBNOMORE_ROOT)},
                )
                adapter.begin(identities, decisions, [2])
                batch = adapter.next_batch()
                self.assertEqual([tuple(value.shape) for value in batch.inputs], list(expected_shapes))
                self.assertEqual(batch.identities, identities)
                self.assertTrue(all(value.dtype == torch.float32 for value in batch.inputs))
                self.assertTrue(adapter.prefetched_read_identities())
                adapter.close()

    def test_gpu_only_adapters_reject_cpu_as_environment_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 1)
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                GalpTrainingAdapter(
                    samples,
                    batch_size=1,
                    workers=0,
                    device=torch.device("cpu"),
                    config={},
                )
            dali = DaliTrainingAdapter(
                samples,
                batch_size=1,
                workers=0,
                device=torch.device("cpu"),
                config={},
            )
            decision = derive_augmentation(
                seed=1,
                epoch=0,
                logical_sample_id="id-0",
                source_width=256,
                source_height=256,
                domain="rgb",
            )
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                dali.begin([SampleIdentity(0, 0, "id-0")], [decision], [1])

    def test_augmentation_is_keyed_worker_independent_and_dct_aligned(self) -> None:
        first = derive_augmentation(
            seed=9, epoch=2, logical_sample_id="x", source_width=320, source_height=288, domain="dct"
        )
        again = derive_augmentation(
            seed=9, epoch=2, logical_sample_id="x", source_width=320, source_height=288, domain="dct"
        )
        changed = derive_augmentation(
            seed=9, epoch=3, logical_sample_id="x", source_width=320, source_height=288, domain="dct"
        )
        self.assertEqual(first, again)
        self.assertNotEqual(first.augmentation_key, changed.augmentation_key)
        for value in (first.crop_x, first.crop_y, first.crop_width, first.crop_height):
            self.assertEqual(value % 16, 0)
        self.assertLessEqual(first.crop_x + first.crop_width, first.source_width)
        self.assertLessEqual(first.crop_y + first.crop_height, first.source_height)
        descriptor = first.native_dct_descriptor()
        self.assertEqual(descriptor["crop"]["unit"], "source_pixels")
        self.assertEqual(descriptor["augmentation_key"], first.augmentation_key)

    def test_dct_aligned_crops_stay_within_non_aligned_sources(self) -> None:
        for seed in range(16):
            decisions = (
                derive_augmentation(
                    seed=seed,
                    epoch=0,
                    logical_sample_id="edge",
                    source_width=16,
                    source_height=22,
                    domain="dct",
                ),
                derive_shard_shared_crop_augmentation(
                    seed=seed,
                    epoch=0,
                    physical_shard_id=0,
                    logical_sample_id="edge",
                    source_width=16,
                    source_height=22,
                    domain="dct",
                ),
            )
            for decision in decisions:
                for value in (
                    decision.crop_x,
                    decision.crop_y,
                    decision.crop_width,
                    decision.crop_height,
                ):
                    self.assertEqual(value % 16, 0)
                self.assertGreater(decision.crop_width, 0)
                self.assertGreater(decision.crop_height, 0)
                self.assertLessEqual(
                    decision.crop_x + decision.crop_width,
                    decision.source_width,
                )
                self.assertLessEqual(
                    decision.crop_y + decision.crop_height,
                    decision.source_height,
                )

    def test_dct_augmentation_rejects_sources_smaller_than_alignment(self) -> None:
        for width, height in ((15, 32), (32, 15), (8, 8)):
            with self.assertRaisesRegex(ValueError, "at least one alignment unit"):
                derive_augmentation(
                    seed=9,
                    epoch=0,
                    logical_sample_id=f"{width}x{height}",
                    source_width=width,
                    source_height=height,
                    domain="dct",
                )

    def test_dct_horizontal_flip_uses_block_reverse_and_odd_u_sign(self) -> None:
        y = torch.arange(1 * 1 * 2 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 1, 2, 3, 8, 8)
        cbcr = y.repeat(1, 2, 1, 1, 1, 1)
        actual_y, actual_cbcr = horizontal_flip_dct(y, cbcr)
        expected = y.flip(-3)
        expected[..., 1::2] *= -1
        torch.testing.assert_close(actual_y, expected)
        torch.testing.assert_close(actual_cbcr[:, 0:1], expected)

    def test_rgbnomore_pixel_crop_matches_native_component_block_mapping(self) -> None:
        decision = derive_augmentation(
            seed=3,
            epoch=0,
            logical_sample_id="mapping-fixture",
            source_width=500,
            source_height=375,
            domain="dct",
        )
        decision = replace(
            decision,
            crop_x=48,
            crop_y=0,
            crop_width=304,
            crop_height=304,
        )
        self.assertEqual(
            _RgbNoMoreDctDataset._component_crop(
                decision, block_width=63, block_height=47
            ),
            (0, 6, 39, 39),
        )
        self.assertEqual(
            _RgbNoMoreDctDataset._component_crop(
                decision, block_width=32, block_height=24
            ),
            (0, 3, 20, 20),
        )
        self.assertEqual(
            _RgbNoMoreDctDataset._component_crop(
                decision, block_width=64, block_height=48
            ),
            (0, 6, 39, 40),
        )

    def test_order_cursor_drop_last_prefetch_and_duplicate_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 5)
            batches, dropped = _collect_batches(
                samples, seed=11, batch_size=2, batch_count=3, drop_last=True
            )
            self.assertEqual([batch[0].epoch for batch in batches], [0, 0, 1])
            self.assertEqual(dropped, {0: 1})
            cursor = batches[0][-1].as_dict()
            resumed, _ = _collect_batches(
                samples, seed=11, batch_size=2, batch_count=1, drop_last=True, start_cursor=cursor
            )
            self.assertEqual(resumed[0], batches[1])
            ledger = SampleOrderLedger("pytorch")
            expected = batches[0]
            ledger.record_requested(expected + batches[1])
            ledger.record_prefetched(expected + batches[1])
            ledger.record_emitted(expected)
            ledger.record_consumed(expected)
            self.assertTrue(ledger.validate(expected)["ok"])
            self.assertEqual(ledger.validate(expected)["prefetch_overrun"], 2)
            ledger.record_consumed([expected[0]])
            result = ledger.validate(expected)
            self.assertFalse(result["ok"])
            self.assertTrue(result["duplicate_consumed"])
            missing = SampleOrderLedger("pytorch")
            missing.record_requested(expected)
            missing.record_prefetched(expected)
            missing.record_emitted(expected[:1])
            missing.record_consumed(expected[:1])
            missing_result = missing.validate(expected)
            self.assertFalse(missing_result["ok"])
            self.assertTrue(
                any("canonical order" in value for value in missing_result["failures"])
            )
            with self.assertRaisesRegex(ValueError, "cannot produce a batch"):
                _collect_batches(samples[:1], seed=1, batch_size=2, batch_count=1, drop_last=True)

    def test_rank_partition_and_extra_lookahead_preserve_logical_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = [
                TrainingSample(
                    f"id-{index}",
                    Path(temporary) / f"unused-{index}.jpg",
                    index % 1000,
                    256,
                    256,
                    index,
                )
                for index in range(12)
            ]
            ids = [sample.logical_sample_id for sample in samples]
            global_order = canonical_epoch_order(ids, 41, 0)
            rank_orders = [
                canonical_epoch_order(
                    ids,
                    41,
                    0,
                    distributed_rank=rank,
                    distributed_world_size=4,
                )
                for rank in range(4)
            ]
            self.assertEqual(
                [[value.logical_sample_id for value in order] for order in rank_orders],
                [
                    [value.logical_sample_id for value in global_order[rank::4]]
                    for rank in range(4)
                ],
            )
            self.assertEqual(
                len({value.logical_sample_id for order in rank_orders for value in order}),
                len(global_order),
            )
            shallow, _ = _collect_batches(
                samples,
                seed=41,
                batch_size=2,
                batch_count=2,
                drop_last=True,
                distributed_rank=0,
                distributed_world_size=2,
            )
            deep, _ = _collect_batches(
                samples,
                seed=41,
                batch_size=2,
                batch_count=5,
                drop_last=True,
                distributed_rank=0,
                distributed_world_size=2,
            )
            self.assertEqual(shallow, deep[:2])

    def test_manifest_labels_hashes_and_train_validation_separation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = _samples(root, 2)
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    {
                        "split": "train",
                        "samples": [
                            {
                                "logical_sample_id": sample.logical_sample_id,
                                "path": sample.path.name,
                                "label": sample.label,
                                "width": 256,
                                "height": 256,
                                "payload_sha256": sha256_file(sample.path),
                            }
                            for sample in samples
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded, metadata = load_training_manifest(manifest, root=None, expected_split="train")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(metadata["sample_count"], 2)
            self.assertFalse(validate_dataset_separation(loaded, loaded)["ok"])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["samples"][0]["label"] = 1000
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid ImageNet-1K label"):
                load_training_manifest(manifest, root=None, expected_split="train")

    def test_semantic_gates_reject_nan_and_gate_first_update(self) -> None:
        identity = [SampleIdentity(0, 0, "a")]
        base = {
            "initial_hash": "same",
            "initial_optimizer_hash": "same-optimizer",
            "initial_scheduler_hash": "same-scheduler",
            "model_configuration": {"architecture": "fixture"},
            "optimizer_configuration_sha256": "same-optimizer-config",
            "sample_ids": identity,
            "augmentation_decisions": [{"key": "same"}],
            "labels": torch.tensor([0]),
            "inputs": [torch.ones(1, 3)],
            "logits": torch.tensor([[1.0, 0.0]]),
            "loss": 0.5,
            "gradients": {"w": torch.ones(3)},
            "updates": {"w": torch.ones(3)},
        }
        gates = {
            "dct_input_atol": 1e-5,
            "dct_rtol": 1e-4,
            "rgb_input_warning_atol": 0.1,
            "rgb_input_failure_atol": 0.5,
            "gradient_cosine_dct": 0.999,
            "gradient_cosine_rgb": 0.99,
        }
        self.assertEqual(_semantic_compare("rgb", "a", base, "b", copy.deepcopy(base), gates)["status"], "passed")
        invalid = copy.deepcopy(base)
        invalid["inputs"][0][0, 0] = torch.nan
        result = _semantic_compare("rgb", "a", base, "b", invalid, gates)
        self.assertEqual(result["status"], "failed")
        opposite = copy.deepcopy(base)
        opposite["updates"]["w"] *= -1
        result = _semantic_compare("rgb", "a", base, "b", opposite, gates)
        self.assertTrue(any("update cosine" in value for value in result["failures"]))
        nan_loss = copy.deepcopy(base)
        nan_loss["loss"] = float("nan")
        result = _semantic_compare("rgb", "a", base, "b", nan_loss, gates)
        self.assertTrue(any("loss contains NaN" in value for value in result["failures"]))
        self.assertFalse(tensor_is_finite(torch.tensor(float("nan"))))

    def test_numerical_and_performance_statuses_are_independent(self) -> None:
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.zeros_like(model.weight)
        model.bias.grad = torch.zeros_like(model.bias)
        summary = gradient_summary(model)
        self.assertTrue(summary["finite"])
        self.assertFalse(summary["nonzero"])
        model.bias.grad = torch.full_like(model.bias, float("inf"))
        self.assertFalse(gradient_summary(model)["finite"])
        repeats = [
            {"repeat": index, "throughput_images_per_s": value}
            for index, value in enumerate((100.0, 100.0, 200.0, 100.0, 200.0))
        ]
        aggregate = _aggregate_step_repeats(repeats, 0.01)
        self.assertEqual(aggregate["included_repeats"], [1, 2, 3, 4])
        self.assertEqual(aggregate["performance_status"], "failed")
        self.assertNotIn("correctness", aggregate)

    def test_schema_group_coverage_and_not_run_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing pipelines"):
            validate_required_group_coverage(["pytorch"], ["rgb"])
        self.assertTrue(all(value == "not_run" for value in empty_status("not_run").values()))
        minimal = {"schema_version": TRAINING_CONTRACT_SCHEMA}
        self.assertIn("contract missing model", validate_training_document(minimal, TRAINING_CONTRACT_SCHEMA))

    def test_artifact_hashes_and_dirty_source_provenance_are_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Training Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "runtime.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            (root / "unrelated.bin").write_bytes(b"user data")
            provenance = repository_provenance(root, [runtime])
            self.assertTrue(provenance["dirty"])
            self.assertFalse(provenance["untracked_runtime_files"])
            output = root / "artifacts"
            output.mkdir()
            (output / "result.json").write_text("{}\n", encoding="utf-8")
            hashes = write_artifact_hashes(output)
            self.assertFalse(verify_artifact_hashes(output, hashes))
            (output / "result.json").write_text("tampered\n", encoding="utf-8")
            self.assertTrue(verify_artifact_hashes(output, hashes))

    def test_cpu_dry_run_emits_single_pipeline_contract_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = _samples(root, 2)
            val_root = root / "val"
            val_root.mkdir()
            val = _samples(val_root, 1)
            manifests = []
            for name, split, values in (("train", "train", train), ("val", "val", val)):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "split": split,
                            "samples": [
                                {
                                    "logical_sample_id": f"{split}-{sample.logical_sample_id}",
                                    "path": str(sample.path),
                                    "label": sample.label,
                                    "width": 256,
                                    "height": 256,
                                }
                                for sample in values
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(path)
            args = _parse_args(
                [
                    "--pipeline",
                    "pytorch",
                    "--phase",
                    "smoke",
                    "--train-manifest",
                    str(manifests[0]),
                    "--val-manifest",
                    str(manifests[1]),
                    "--output-dir",
                    str(root / "out"),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--execution-mode",
                    "runtime",
                    "--dry-run-contract",
                ]
            )
            contract = run(args)
            self.assertEqual(contract["enabled_pipelines"], ["pytorch"])
            self.assertEqual(contract["model"]["architecture"], "rgbnomore-vitti-v1")
            self.assertEqual(contract["execution"]["mode"], "runtime")
            self.assertEqual(
                contract["execution"]["measurement_policy"]["selected"], "runtime"
            )
            self.assertFalse(contract["claims"]["cross_dct_rgb_tensor_equivalence"])
            self.assertEqual(contract["validation_augmentation"]["horizontal_flip"], False)

    def test_cpu_single_pipeline_smoke_run_writes_validated_artifacts(self) -> None:
        if not RGBNOMORE_ROOT.is_dir():
            self.skipTest("external RGB-no-more checkout is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = _samples(root, 2)
            val_root = root / "val"
            val_root.mkdir()
            val = _samples(val_root, 1)
            manifests = []
            for name, split, values in (("train", "train", train), ("val", "val", val)):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "split": split,
                            "samples": [
                                {
                                    "logical_sample_id": f"{split}-{sample.logical_sample_id}",
                                    "path": str(sample.path),
                                    "label": sample.label,
                                    "width": 256,
                                    "height": 256,
                                }
                                for sample in values
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(path)
            output = root / "run"
            args = _parse_args(
                [
                    "--pipeline",
                    "pytorch",
                    "--phase",
                    "smoke",
                    "--train-manifest",
                    str(manifests[0]),
                    "--val-manifest",
                    str(manifests[1]),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--execution-mode",
                    "runtime",
                    "--warmup-steps",
                    "1",
                    "--measured-steps",
                    "1",
                    "--repeats",
                    "3",
                ]
            )
            result = run(args)
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            pipeline = json.loads((output / "pipeline_pytorch.json").read_text(encoding="utf-8"))
            semantic = json.loads((output / "semantic_comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["ok"], validation["failures"])
            self.assertEqual(result["pipeline_status"]["galp"]["overall"], "not_run")
            self.assertEqual(pipeline["status"]["correctness"], "passed")
            self.assertEqual(pipeline["execution_mode"], "runtime")
            self.assertEqual(semantic["rgb"]["status"], "not_run")
            self.assertEqual(semantic["dct"]["status"], "not_run")
            self.assertEqual(len(pipeline["phase_results"]["smoke"]["repeats"]), 3)
            self.assertTrue(
                all(
                    repeat["optimizer_steps"] == 2
                    for repeat in pipeline["phase_results"]["smoke"]["repeats"]
                )
            )
            repeat = pipeline["phase_results"]["smoke"]["repeats"][0]
            self.assertEqual(
                repeat["measurement_instrumentation"][
                    "deep_gradient_scans_in_measured_path"
                ],
                0,
            )
            self.assertEqual(
                repeat["synchronization_accounting"]["explicit_host_device"]["count"],
                0,
            )
            self.assertTrue((output / "artifact_hashes.json").is_file())
            contract_before = (output / "contract.json").read_bytes()
            initial_path = output / "initial_state_rgb_seed11997733.pt"
            initial_hash_before = sha256_file(initial_path)
            resumed = run(_parse_args(["--resume-run", str(output)]))
            validation_after = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(validation_after["ok"], validation_after["failures"])
            self.assertEqual(resumed["contract_sha256"], result["contract_sha256"])
            self.assertEqual((output / "contract.json").read_bytes(), contract_before)
            self.assertEqual(sha256_file(initial_path), initial_hash_before)

            pipeline_path = output / "pipeline_pytorch.json"
            pipeline_before = pipeline_path.read_bytes()
            malformed_pipeline = json.loads(pipeline_before)
            malformed_pipeline["phase_results"]["smoke"]["repeats"][0][
                "common_statistics"
            ]["samples"] += 1
            pipeline_path.write_text(
                json.dumps(malformed_pipeline, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_artifact_hashes(
                output, excluded=("artifact_hashes.json", "validation.json")
            )
            malformed = validate_output(output, write_result=False)
            self.assertFalse(malformed["ok"])
            self.assertTrue(
                any(
                    "common samples disagree with processed_images" in value
                    for value in malformed["failures"]
                )
            )
            pipeline_path.write_bytes(pipeline_before)
            write_artifact_hashes(
                output, excluded=("artifact_hashes.json", "validation.json")
            )

            (output / "commands.json").write_text("{}\n", encoding="utf-8")
            tampered = validate_output(output, write_result=False)
            self.assertFalse(tampered["ok"])
            self.assertTrue(
                any("artifact hash mismatch" in value for value in tampered["failures"])
            )



if __name__ == "__main__":
    unittest.main()
