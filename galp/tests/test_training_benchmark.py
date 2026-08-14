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


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks/system_rgbnomore"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from training.artifacts import (  # noqa: E402
    nested_state_sha256,
    repository_provenance,
    sha256_file,
    tensor_state_sha256,
    verify_artifact_hashes,
    write_artifact_hashes,
)
from training.augmentation import (  # noqa: E402
    derive_augmentation,
    derive_shard_shared_crop_augmentation,
    horizontal_flip_dct,
)
from training.metrics import gradient_summary, process_memory, tensor_is_finite  # noqa: E402
from training.direct_dct_reader import (  # noqa: E402
    DirectDctTrainingReader,
    NativeExecutionStatsAccumulator,
    merge_native_counter_snapshot,
    native_allocation_stability,
)
from training.gate2_acceptance import evaluate_gate2  # noqa: E402
from training.gate3_acceptance import (  # noqa: E402
    _EXACT_REPEAT_RESOURCE_FIELDS,
    evaluate_gate3,
)
from training.strict1k_acceptance import evaluate_strict1k  # noqa: E402
from training.select_gate3_prefetch import select_gate3_prefetch  # noqa: E402
from training.manifest_preflight import (  # noqa: E402
    ManifestPreflightError,
    preflight_manifest,
)
from training.generate_imagenet_manifests import main as generate_training_manifests  # noqa: E402
from training.model_factory import (  # noqa: E402
    EXPECTED_PARAMETER_COUNTS,
    build_model,
    capture_rng_state,
    capture_training_state,
    initialize_model,
    model_configuration,
    reset_training_state,
    restore_rng_state,
    seed_everything,
)
from training.optimizer import build_optimizer, build_scheduler  # noqa: E402
from training.pipeline import (  # noqa: E402
    DaliTrainingAdapter,
    GalpTrainingAdapter,
    PyTorchTrainingAdapter,
    OrderedAsyncPrefetchQueue,
    RgbNoMoreTrainingAdapter,
    _RgbNoMoreDctDataset,
    TrainingBatch,
    TrainingSample,
    _uniform_dali_tensor,
    load_training_manifest,
    validate_dataset_separation,
)
from training.run import (  # noqa: E402
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
from training.sample_order import (  # noqa: E402
    SampleIdentity,
    SampleOrderLedger,
    canonical_epoch_order,
)
from training.v3_acceptance import _gpu_semantic_checks, _normalize_contract, build_report  # noqa: E402
from training.select_imagenet_canary import select_canary  # noqa: E402
from training.schema import (  # noqa: E402
    TRAINING_CONTRACT_SCHEMA,
    empty_status,
    validate_required_group_coverage,
    validate_training_document,
)
from training.validate import validate_output  # noqa: E402


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


class _FakeAsyncHandle:
    def __init__(self, value: object = None, *, ready: bool = True, error: Exception | None = None) -> None:
        self.value = value
        self.ready = ready
        self.started = True
        self.producer_active_ms = 2.0
        self.error = error
        self.read_count = 0
        self.cancel_count = 0

    def read(self):
        self.read_count += 1
        if self.error is not None:
            raise self.error
        return self.value

    def cancel(self) -> bool:
        self.cancel_count += 1
        return False


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

    def native_execution_stats(self):
        return {} if self._execution_stats is None else dict(self._execution_stats)


class _FakeDirectDctTrainingReader:
    def __init__(self, execution_stats: dict[str, object] | None = None) -> None:
        self.execution_stats = execution_stats
        self.requests: list[tuple[list[int], list[dict[str, object]], dict[str, object]]] = []
        self.batches: list[_FakeNativeTrainingBatch] = []

    def prefetch_batch(self, image_ids, *, transforms, **options):
        ids = [int(value) for value in image_ids]
        descriptors = [dict(value) for value in transforms]
        self.requests.append((ids, descriptors, dict(options)))
        batch = _FakeNativeTrainingBatch(ids, descriptors, self.execution_stats)
        self.batches.append(batch)
        return _FakeAsyncHandle(batch)


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

            def prefetch_batch(self, image_ids, *, transforms, **_options):
                return _FakeAsyncHandle(
                    _FakeNativeTrainingBatch(
                        list(image_ids), list(transforms), {"future_counter": 11}
                    )
                )

        native_module = SimpleNamespace(DirectDctReader=NativeReader)
        for name in ("v2.bin", "v3.bin"):
            reader = DirectDctTrainingReader(Path(name), native_module=native_module)
            self.assertEqual(reader.image_count, 9)
            handle = reader.prefetch_batch(
                [3], transforms=[{"global_image_id": 3}], cache_capacity_mib=0
            )
            batch = handle.read()
            self.assertEqual(batch.global_image_ids, [3])
            self.assertEqual(batch.native_execution_stats()["future_counter"], 11)
            self.assertEqual(
                batch.native_execution_stats_snapshot()["future_counter"], 11
            )
        self.assertEqual(len(constructed), 2)

    def test_repository_v2_v3_manifests_open_through_same_public_reader(self) -> None:
        repository = BENCHMARK_DIR.parents[2]
        module_path = repository / "build/galp/torch"
        manifests = (
            (
                2,
                "image-major",
                50_000,
                repository / "galp/data/system_rgbnomore/e2e_v2/dct/manifest.bin",
            ),
            (
                3,
                "image-major-vector-rowgroups",
                1_000,
                repository
                / "galp/data/system_rgbnomore/e2e_v3/galp-v3-train-canary/train-1k-v3c/dct/manifest.bin",
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
                        "prefetch_depth": 1,
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
                        "prefetch_depth": 2,
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
                    "prefetch_depth": 1,
                    "execution_mode": "runtime",
                },
            )
            adapter.begin(identities, decisions, [2])
            batch = adapter.next_batch()
            self.assertFalse(batch.native_stats_finalized)
            adapter.snapshot_batch_metrics(batch)
            self.assertTrue(batch.native_stats_finalized)
            self.assertEqual(batch.native_execution_stats["planning_ms"], 3.5)
            self.assertEqual(batch.stage_seconds["preprocess"], 0.00125)
            self.assertEqual(reader.batches[0].complete_stats_reads, 0)
            self.assertEqual(reader.batches[0].snapshot_stats_reads, 1)
            adapter.close()

    def test_galp_workers_and_prefetch_depth_do_not_change_semantics(self) -> None:
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
            for workers, depth in ((1, 0), (1, 3), (4, 0), (4, 3)):
                reader = _FakeDirectDctTrainingReader(
                    {"future_counter": workers * 10 + depth}
                )
                adapter = GalpTrainingAdapter(
                    samples,
                    batch_size=2,
                    workers=workers,
                    device=torch.device("cpu"),
                    config={
                        "_direct_dct_training_reader": reader,
                        "prefetch_depth": depth,
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
                self.assertLessEqual(
                    adapter.loader_metrics()["max_queue_depth_batches"], depth + 1
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

    def test_gate2_acceptance_requires_capacity_contract_and_zero_measured_growth(self) -> None:
        zero_aggregate_names = (
            "descriptor_map_count",
            "descriptor_open_ms",
            "schema_plan_build_ms",
            "static_metadata_cache_miss_count",
            "host_expanded_transform_items_created",
            "host_output_block_source_lists_created",
            "host_global_transform_sort_items",
        )
        growth_names = (
            "compact_batch_buffer_growth_count",
            "compact_batch_buffer_pageable_fallback_count",
            "decode_workset_output_arena_growth_count",
            "decode_workset_chunk_arena_growth_count",
            "planless_axis_program_device_growth_count",
            "planless_axis_program_pinned_growth_count",
        )
        repeat = {
            "throughput_images_per_s": 1600.0,
            "loader_measured_metrics": {
                "submitted_batches": 10,
                "producer_planning_seconds": 0.05,
                "consumer_wait_fraction_of_measured_wall": 0.01,
                "queue_hit_rate": 1.0,
            },
            "native_allocation_stability": {
                "verifiable": True,
                "stable_after_warmup": True,
                "capacity_contract_complete": True,
                "global_counter_deltas": {
                    "galp_native_device_cuda_allocation_count": 0,
                    "galp_native_pinned_cuda_allocation_count": 0,
                },
                "measured_per_batch_totals": {name: 0 for name in growth_names},
            },
            "native_execution_stats_by_phase": {
                "measured": {
                    "latest": {
                        "planless_axis_program_capacity_contract_complete": True,
                        "planless_axis_program_capacity_contract_bytes": 571648,
                        "planless_axis_program_device_capacity_bytes": 1048576,
                        "planless_axis_program_pinned_capacity_bytes": 1048576,
                        "compact_batch_pool_capacity_contract_complete": True,
                        "compact_batch_pool_capacity_contract_images": 64,
                        "compact_batch_pool_capacity_contract_groups": 64,
                        "compact_batch_pool_capacity_contract_batches": 4,
                        "compact_batch_pool_capacity_contract_bytes": 32 * 1024 * 1024,
                        "compact_batch_pool_prewarmed_slots": 256,
                        "compact_batch_pool_prewarmed_bytes": 32 * 1024 * 1024,
                    },
                    "numeric_aggregates": {
                        name: {"sum": 0.0, "count": 10} for name in zero_aggregate_names
                    },
                }
            },
            "sample_order": {"validation": {"ok": True}},
        }
        passed_status = {
            "artifact": "passed",
            "correctness": "passed",
            "semantic": "passed",
            "overall": "passed",
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            pipeline = {
                "status": passed_status,
                "phase_results": {"smoke": {"repeats": [repeat]}},
            }
            results = {"pipeline_status": {"galp": passed_status}}
            for name, value in (
                ("pipeline_galp.json", pipeline),
                ("results.json", results),
                ("contract.json", {}),
                ("sample_order.json", {}),
            ):
                (run_dir / name).write_text(json.dumps(value), encoding="utf-8")

            accepted = evaluate_gate2(run_dir)
            self.assertTrue(accepted["complete"])
            self.assertTrue(accepted["ok"])
            self.assertEqual(accepted["metrics"]["repeats"][0]["planning_ms_per_batch"], 5.0)

            strict_repeat = copy.deepcopy(repeat)
            strict_repeat["native_execution_stats_by_phase"]["measured"]["latest"].update(
                {
                    "uses_planless_fixed_transform": True,
                    "host_expanded_transform_items_created": 0,
                    "host_output_block_source_lists_created": 0,
                    "host_global_transform_sort_items": 0,
                }
            )
            strict_status = {**passed_status, "semantic": "not_applicable"}
            strict_pipeline = {
                "status": strict_status,
                "first_step_semantic_probe": {
                    "before_warmup": True,
                    "fresh_clone": True,
                    "formal_repeat_polluted": False,
                    "failures": [],
                },
                "phase_results": {
                    "smoke": {
                        "repeats": [copy.deepcopy(strict_repeat) for _ in range(3)]
                    }
                },
            }
            (run_dir / "pipeline_galp.json").write_text(
                json.dumps(strict_pipeline), encoding="utf-8"
            )
            (run_dir / "results.json").write_text(
                json.dumps({"pipeline_status": {"galp": strict_status}}),
                encoding="utf-8",
            )
            strict = evaluate_strict1k(run_dir)
            self.assertTrue(strict["ok"])
            self.assertEqual(strict["strict1k"]["throughput_median_images_per_s"], 1600.0)
            strict_pipeline["phase_results"]["smoke"]["repeats"][-1][
                "native_execution_stats_by_phase"
            ]["measured"]["latest"]["host_global_transform_sort_items"] = 1
            (run_dir / "pipeline_galp.json").write_text(
                json.dumps(strict_pipeline), encoding="utf-8"
            )
            strict_rejected = evaluate_strict1k(run_dir)
            self.assertFalse(strict_rejected["ok"])
            self.assertFalse(
                strict_rejected["checks"]["strict1k.repeat_2.compact_planless_path"]["ok"]
            )

            repeat["native_allocation_stability"]["stable_after_warmup"] = False
            repeat["native_allocation_stability"]["global_counter_deltas"][
                "galp_native_device_cuda_allocation_count"
            ] = 1
            (run_dir / "pipeline_galp.json").write_text(json.dumps(pipeline), encoding="utf-8")
            (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
            rejected = evaluate_gate2(run_dir)
            self.assertFalse(rejected["ok"])
            self.assertIn("repeat_0.allocation_stability", rejected["checks"])
            self.assertFalse(rejected["checks"]["repeat_0.allocation_stability"]["ok"])
            self.assertFalse(
                rejected["checks"]["repeat_0.galp_native_device_cuda_allocation_count_delta"]["ok"]
            )

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

    def test_gate3_acceptance_requires_repeat_and_process_resource_stability(self) -> None:
        zero_aggregate_names = (
            "descriptor_map_count",
            "descriptor_open_ms",
            "schema_plan_build_ms",
            "static_metadata_cache_miss_count",
            "host_expanded_transform_items_created",
            "host_output_block_source_lists_created",
            "host_global_transform_sort_items",
        )
        growth_names = (
            "compact_batch_buffer_growth_count",
            "compact_batch_buffer_pageable_fallback_count",
            "decode_workset_output_arena_growth_count",
            "decode_workset_chunk_arena_growth_count",
            "planless_axis_program_device_growth_count",
            "planless_axis_program_pinned_growth_count",
        )

        def repeat(index: int, throughput: float) -> dict[str, object]:
            latest = {name: 4096 for name in _EXACT_REPEAT_RESOURCE_FIELDS}
            latest.update(
                {
                    "planless_axis_program_capacity_contract_complete": True,
                    "planless_axis_program_capacity_contract_bytes": 571648,
                    "planless_axis_program_device_capacity_bytes": 1048576,
                    "planless_axis_program_pinned_capacity_bytes": 1048576,
                }
            )
            return {
                "repeat": index,
                "ok": True,
                "failures": [],
                "throughput_images_per_s": throughput,
                "loader_measured_metrics": {
                    "submitted_batches": 500,
                    "producer_planning_seconds": 2.5,
                    "consumer_wait_fraction_of_measured_wall": 0.01,
                    "queue_hit_rate": 1.0,
                },
                "sample_order": {"validation": {"ok": True}},
                "native_allocation_stability": {
                    "verifiable": True,
                    "stable_after_warmup": True,
                    "capacity_contract_complete": True,
                    "global_counter_deltas": {
                        "galp_native_device_cuda_allocation_count": 0,
                        "galp_native_pinned_cuda_allocation_count": 0,
                    },
                    "measured_per_batch_totals": {name: 0 for name in growth_names},
                },
                "native_execution_stats_by_phase": {
                    "measured": {
                        "latest": latest,
                        "numeric_aggregates": {
                            name: {"sum": 0.0, "count": 500}
                            for name in zero_aggregate_names
                        },
                    }
                },
                "host_memory": {
                    "rss_bytes": 1024 * 1024 * (1000 + index),
                    "peak_rss_bytes": 1024 * 1024 * (1100 + index),
                    "open_fd_count": 300,
                    "memory_mapping_count": 2400,
                },
            }

        passed_status = {
            "artifact": "passed",
            "correctness": "passed",
            "semantic": "passed",
            "performance": "passed",
            "overall": "passed",
        }
        repeats = [repeat(0, 1600.0), repeat(1, 1601.0), repeat(2, 1599.0)]
        pipeline = {
            "status": passed_status,
            "phase_results": {
                "step": {
                    "repeats": repeats,
                    "aggregate": {"performance_status": "passed"},
                }
            },
        }
        results = {"pipeline_status": {"galp": passed_status}}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            for name, value in (
                ("pipeline_galp.json", pipeline),
                ("results.json", results),
                ("contract.json", {}),
                ("sample_order.json", {}),
            ):
                (run_dir / name).write_text(json.dumps(value), encoding="utf-8")

            accepted = evaluate_gate3(run_dir)
            self.assertTrue(accepted["complete"])
            self.assertTrue(accepted["ok"])
            self.assertEqual(accepted["metrics"]["repeat_count"], 3)
            self.assertEqual(
                accepted["metrics"]["hot_repeat_throughput"]["median"],
                1600.0,
            )

            repeats[-1]["native_execution_stats_by_phase"]["measured"]["latest"][
                "galp_native_pinned_cuda_allocation_count"
            ] = 5
            (run_dir / "pipeline_galp.json").write_text(
                json.dumps(pipeline), encoding="utf-8"
            )
            rejected = evaluate_gate3(run_dir)
            self.assertFalse(rejected["ok"])
            self.assertFalse(
                rejected["checks"][
                    "repeat_resource_stability.galp_native_pinned_cuda_allocation_count"
                ]["ok"]
            )

    def test_gate3_prefetch_selector_requires_gates_and_two_percent_gain(self) -> None:
        def report(median: float, *, ok: bool = True) -> dict[str, object]:
            return {
                "complete": True,
                "ok": ok,
                "failures": [] if ok else ["resource growth"],
                "metrics": {"hot_repeat_throughput": {"median": median}},
            }

        below_margin = select_gate3_prefetch(
            {2: report(1000.0), 4: report(1019.0), 8: report(1100.0, ok=False)}
        )
        self.assertTrue(below_margin["ok"])
        self.assertEqual(below_margin["selection"]["prefetch_depth_batches"], 2)

        above_margin = select_gate3_prefetch(
            {2: report(1000.0), 4: report(1021.0), 8: report(1010.0)}
        )
        self.assertTrue(above_margin["ok"])
        self.assertEqual(above_margin["selection"]["prefetch_depth_batches"], 4)

        missing = select_gate3_prefetch({2: report(1000.0), 4: report(1100.0)})
        self.assertFalse(missing["complete"])
        self.assertFalse(missing["ok"])

    def test_v3_acceptance_report_combines_planner_reader_and_training_gates(self) -> None:
        def write(path: Path, value: dict[str, object]) -> None:
            path.write_text(json.dumps(value), encoding="utf-8")

        def pipeline(
            throughput: float, syncs: float, compute_upper_bound: float = 1100.0
        ) -> dict[str, object]:
            numeric = {
                "planning_ms": {"sum": 100.0, "count": 10},
                "sync_rowgroup_read_ms": {"sum": 200.0, "count": 10},
                "decode_ms": {"sum": 2.0, "count": 10},
                "fixed_transform_ms": {"sum": 40.0, "count": 10},
                "internal_sync_count": {"sum": syncs * 10, "count": 10},
                "coalesced_read_run_count": {"sum": 20.0, "count": 10},
                "preadv_count": {"sum": 10.0, "count": 10},
            }
            latest = {
                "uses_planless_fixed_transform": True,
                "host_expanded_transform_items_created": 0,
                "host_output_block_source_lists_created": 0,
                "host_global_transform_sort_items": 0,
                "compact_plan_bytes": 4096,
            }
            repeat = {
                "ok": True,
                "throughput_images_per_s": throughput,
                "measured_region_wall_clock_seconds": 0.64,
                "measured_steps": 10,
                "loader_measured_metrics": {
                    "consumer_wait_fraction_of_measured_wall": 0.1,
                },
                "compute_only_diagnostic": {
                    "upper_bound_images_per_s": compute_upper_bound
                },
                "native_execution_stats_by_phase": {
                    "measured": {"numeric_aggregates": numeric, "latest": latest},
                },
                "native_allocation_stability": {
                    "verifiable": True,
                    "stable_after_warmup": True,
                },
            }
            return {
                "status": {"semantic": "passed", "correctness": "passed"},
                "first_step_semantic_probe": {
                    "failures": [],
                    "before_warmup": True,
                    "fresh_clone": True,
                    "formal_repeat_polluted": False,
                    "sample_ids": [{"logical_sample_id": "image-0", "epoch": 0, "position": 0}],
                    "labels": [3],
                    "augmentation_decisions": [{"crop": [0, 0, 16, 16], "horizontal_flip": False}],
                    "inputs": [
                        {
                            "shape": [1, 1, 2, 2, 8, 8],
                            "dtype": "torch.float32",
                            "sha256": "abc",
                            "finite": True,
                        }
                    ],
                    "initial_logits": {"sha256": "def", "finite": True},
                    "first_step_loss": 1.25,
                    "reproducibility": {"transform_descriptor_sha256": "ghi"},
                },
                "phase_results": {
                    "smoke": {
                        "repeats": [
                            {**copy.deepcopy(repeat), "repeat": index}
                            for index in range(3)
                        ]
                    }
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v2_dir = root / "v2"
            v3_dir = root / "v3"
            v2_dir.mkdir()
            v3_dir.mkdir()
            common_contract = {"execution": {"batch_size": 64}, "pipelines": {"x": 1}}
            write(v2_dir / "contract.json", common_contract)
            write(v3_dir / "contract.json", common_contract)
            write(v2_dir / "pipeline_galp.json", pipeline(950.0, 2.0))
            write(v3_dir / "pipeline_galp.json", pipeline(1000.0, 1.0))
            planner_common = {
                "batch_size": 64,
                "crop": [0, 0, 112, 112],
                "horizontal_flip": True,
                "iterations": 100,
                "warmup_iterations": 10,
                "fixed_transform_source_block_count": 100,
                "fixed_transform_output_block_count": 80,
                "planned_selected_vector_count": 7,
                "full_vector_count": 8,
            }
            planner_v2 = root / "planner_v2.json"
            planner_v3 = root / "planner_v3.json"
            reader = root / "reader.json"
            gpu_xml = root / "gpu.xml"
            write(planner_v2, {**planner_common, "planning_ms_mean": 1.0})
            write(
                planner_v3,
                {
                    **planner_common,
                    "planning_ms_mean": 1.1,
                    "uses_planless_fixed_transform": True,
                    "host_expanded_transform_items_created": 0,
                    "host_output_block_source_lists_created": 0,
                    "host_global_transform_sort_items": 0,
                },
            )
            write(
                reader,
                {
                    "actual_saved_rowgroups": 1,
                    "compressed_bytes_avoided": 1024,
                    "planned_decode_vectors": 7,
                    "full_rowgroups": 8,
                    "source_output_block_amplification": 1.2,
                },
            )
            gpu_xml.write_text(
                """<?xml version="1.0"?>
<testsuites tests="2" failures="0" disabled="0" errors="0">
  <testsuite name="JpegDct" tests="2" failures="0" disabled="0" errors="0">
    <testcase name="ManifestV3PlanlessMatchesLegacyAcrossRaggedShardsAndSampling" classname="JpegDct" status="run" result="completed"/>
    <testcase name="PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix" classname="JpegDct" status="run" result="completed"/>
  </testsuite>
</testsuites>
""",
                encoding="utf-8",
            )
            report = build_report(
                baseline_dir=None,
                v2_dir=v2_dir,
                v3_dir=v3_dir,
                planner_v2_path=planner_v2,
                planner_v3_path=planner_v3,
                reader_v3_path=reader,
                gpu_gtest_xml_path=gpu_xml,
            )
            self.assertEqual(report["overall_status"], "pass")
            self.assertFalse(report["blocking_failures"])
            self.assertFalse(report["blocking_unverified"])

            write(v3_dir / "pipeline_galp.json", pipeline(1000.0, 1.0, 700.0))
            unfair = build_report(
                baseline_dir=None,
                v2_dir=v2_dir,
                v3_dir=v3_dir,
                planner_v2_path=planner_v2,
                planner_v3_path=planner_v3,
                reader_v3_path=reader,
                gpu_gtest_xml_path=gpu_xml,
            )
            statuses = {item["name"]: item["status"] for item in unfair["checks"]}
            self.assertEqual(
                statuses["training.v2_v3_compute_conditions_comparable"], "fail"
            )
            self.assertEqual(statuses["training.v3_not_slower_than_v2"], "unverified")

    def test_v3_contract_normalization_uses_semantic_initial_state_identity(self) -> None:
        common = {
            "execution": {"batch_size": 64},
            "initial_states": [
                {
                    "artifact": {"path": "/tmp/v2/initial.pt", "sha256": "artifact-sha"},
                    "rng_state": {
                        "encoding": "pickle-v5-base64",
                        "data": "noncanonical-v2-pickle",
                        "sha256": "rng-sha",
                    },
                }
            ],
        }
        other = copy.deepcopy(common)
        other["initial_states"][0]["artifact"]["path"] = "/tmp/v3/initial.pt"
        other["initial_states"][0]["rng_state"]["data"] = "noncanonical-v3-pickle"
        self.assertEqual(_normalize_contract(common), _normalize_contract(other))

        other["initial_states"][0]["rng_state"]["sha256"] = "different-rng-sha"
        self.assertNotEqual(_normalize_contract(common), _normalize_contract(other))

    def test_v3_gpu_semantic_failures_are_attributed_to_the_exact_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            xml = Path(temporary) / "gpu.xml"
            xml.write_text(
                """<testsuites><testsuite name="JpegDct">
<testcase name="ManifestV3PlanlessMatchesLegacyAcrossRaggedShardsAndSampling" classname="JpegDct"><failure/></testcase>
<testcase name="PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix" classname="JpegDct"/>
</testsuite></testsuites>""",
                encoding="utf-8",
            )
            checks: list[dict[str, object]] = []
            _gpu_semantic_checks(checks, xml)
            statuses = {item["name"]: item["status"] for item in checks}
            self.assertEqual(
                statuses["semantics.gpu_manifest_v3_planless_matches_legacy_ragged"],
                "fail",
            )
            self.assertEqual(
                statuses["semantics.gpu_planless_matches_legacy_generality_matrix"],
                "pass",
            )

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
            "from training.manifest_preflight import preflight_manifest",
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
            ("--galp-cache-capacity-mib", "-1", "must be non-negative"),
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
        with mock.patch("training.run.torch.cuda.synchronize") as synchronize:
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

    def test_ordered_async_queue_order_backpressure_exception_and_close(self) -> None:
        queue = OrderedAsyncPrefetchQueue(2)
        first = _FakeAsyncHandle("first", ready=True)
        second = _FakeAsyncHandle("second", ready=False)
        queue.submit("meta-1", lambda: first)
        queue.submit("meta-2", lambda: second)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            queue.submit("overflow", lambda: _FakeAsyncHandle())
        self.assertEqual(queue.pop(), ("meta-1", "first"))
        self.assertEqual(queue.pop(), ("meta-2", "second"))
        metrics = queue.metrics()
        self.assertEqual(metrics["queue_hit_batches"], 1)
        self.assertEqual(metrics["queue_miss_batches"], 1)
        self.assertEqual(metrics["backpressure_events"], 1)

        failing = OrderedAsyncPrefetchQueue(2)
        error_handle = _FakeAsyncHandle(error=RuntimeError("producer failed"))
        drain_handle = _FakeAsyncHandle("drain")
        failing.submit("bad", lambda: error_handle)
        failing.submit("pending", lambda: drain_handle)
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            failing.pop()
        failing.close()
        self.assertEqual(drain_handle.read_count, 1)
        self.assertEqual(drain_handle.cancel_count, 1)
        self.assertTrue(failing.metrics()["closed"])
        self.assertEqual(failing.metrics()["current_queue_depth_batches"], 0)

    def test_runtime_step_avoids_audit_scans_syncs_and_scalar_materialization(self) -> None:
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
            mock.patch("training.run.gradient_summary") as gradient_scan,
            mock.patch("training.run.parameter_update_summary") as parameter_scan,
            mock.patch("training.run._sync") as explicit_sync,
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
            )
        gradient_scan.assert_not_called()
        parameter_scan.assert_not_called()
        explicit_sync.assert_not_called()
        self.assertIs(emitted, batch)
        self.assertEqual(record["deep_parameter_scans"], 0)
        self.assertEqual(record["host_scalar_materializations_in_step"], 0)
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

    def test_galp_zero_workers_has_explicit_failure_semantics(self) -> None:
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
            with self.assertRaisesRegex(ValueError, "rowgroup-prefetch workers"):
                _validate_args(args)
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
                    config={"rgbnomore_root": str(RGBNOMORE_ROOT), "prefetch_depth": 0},
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

    def test_rank_partition_and_prefetch_depth_preserve_logical_order(self) -> None:
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
                    "--prefetch-depth",
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

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA unavailable; run: cd /home/tangyuxin/gfastlanes/FastLanes && "
        "CUDA_VISIBLE_DEVICES=0 /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python "
        "-m unittest -v "
        "galp.tests.test_training_benchmark.TrainingBenchmarkTest.test_real_gpu_formal_model_step",
    )
    def test_real_gpu_formal_model_step(self) -> None:
        model = build_model(RGBNOMORE_ROOT, "rgb", torch.device("cuda:0"))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss = torch.nn.functional.cross_entropy(
            model(torch.randn(1, 3, 224, 224, device="cuda:0")),
            torch.tensor([1], device="cuda:0"),
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
