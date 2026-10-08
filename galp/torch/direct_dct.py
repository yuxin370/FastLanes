"""Public Python facade for GALP's private Direct-DCT extension."""

from __future__ import annotations

import importlib
import operator
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from galp.profiles import DctModelProfile, DirectDctProfile

if TYPE_CHECKING:
    from .experimental import DirectDctPlsMicrobatch


PROFILE_SCHEMA = "galp-direct-dct-profile-v1"
METRICS_SCHEMA = "galp-direct-dct-metrics-v2"
BINDING_SCHEMA = "galp-direct-dct-binding-v2"


class _NotProvided:
    """Sentinel that lets the facade distinguish omitted compatibility args."""

    __slots__ = ("_public_default",)

    def __init__(self, public_default: str) -> None:
        self._public_default = public_default

    def __repr__(self) -> str:
        return self._public_default


_COEFFICIENTS_NOT_PROVIDED = _NotProvided("None")
_DCT_COEFFS_NOT_PROVIDED = _NotProvided("'all'")
_CoefficientInput = Iterable[int] | None
_INVALID_COEFFICIENTS = (
    "invalid DCT coefficient selection; expected 1 to 64 unique indices in [0, 64)"
)


@dataclass(frozen=True, slots=True)
class TrainingPolicy:
    """Sample ordering and the existing PLS RandAugment/Mixup recipe.

    Mapping and seed retain their native meanings. Model representation and
    execution tuning are deliberately absent; this does not introduce a new
    augmentation implementation.
    """

    mapping: str | Path
    seed: int
    expected_mapping_sha256: str
    segments_per_pool: int = 4
    segment_images: int = 1024
    crop_policy: str = "per-pls"
    order_policy: str = "closed-pool"
    model_classes: int = 1000


def _cnn_pipeline_options(profile: DctModelProfile) -> dict[str, Any]:
    """Existing source512 CNN inference configuration, resolved once.

    These are the options formerly composed by native_options,
    source_options and apply_b6_runtime in the CNN benchmark. Keep the
    execution policy here private, independent of the model descriptor.
    """
    selection = profile.source_frequency_policy
    return dict(
        layout="transformed_dct_grid",
        dct_coeffs="all" if selection == "all" else _normalize_coefficients(coefficients=selection),
        enable_planless_execution=True,
        cache_capacity_mib=0,
        plan_cache_capacity=0,
        decode_batch_rowgroups=64,
        decode_workset_capacity_mib=512,
        enable_rowgroup_prefetch=True,
        rowgroup_prefetch_depth=16,
        rowgroup_prefetch_workers=8,
        rowgroup_prefetch_min_decode_batches=1,
        scheduling_policy="limited-overlap",
        transform_blocks_per_launch=32768,
        transform_ctas_per_launch=512,
        use_low_priority_streams=True,
        async_planless_completion=True,
        block_major_double_buffer="auto",
        crop_execution_mode="bounded-io-uring-range-read-selected-decode",
        bounded_read_amplification_cap=1.1,
        bounded_read_local_amplification_cap=0.0,
        bounded_read_max_run_bytes=0,
        grid_transform=dict(
            y_output_width_blocks=profile.output_grid_size,
            y_output_height_blocks=profile.output_grid_size,
            cbcr_output_width_blocks=profile.output_grid_size,
            cbcr_output_height_blocks=profile.output_grid_size,
            crop_reference_width_blocks=64,
            crop_reference_height_blocks=64,
            crop_origin_alignment_blocks=2,
            chroma_crop_scale_x=2,
            chroma_crop_scale_y=2,
            allowed_chroma_sampling_ratios=[[1, 2, 1, 2]],
            clamp_min=-32768,
            clamp_max=32767,
            output_dtype="float32",
            output_add=0.0,
            output_scale=1.0,
            dequantize=True,
            require_all_coefficients=selection == "all",
            output_channels=profile._native_channels(),
        ),
    )


def _normalize_coefficients(
    *,
    coefficients: _CoefficientInput | _NotProvided = _COEFFICIENTS_NOT_PROVIDED,
    dct_coeffs: str | _NotProvided = _DCT_COEFFS_NOT_PROVIDED,
) -> str:
    """Normalize both Python APIs to the binding's existing string contract."""

    if (
        coefficients is not _COEFFICIENTS_NOT_PROVIDED
        and dct_coeffs is not _DCT_COEFFS_NOT_PROVIDED
    ):
        raise TypeError("coefficients and dct_coeffs are mutually exclusive")
    if dct_coeffs is not _DCT_COEFFS_NOT_PROVIDED:
        # Preserve the legacy path exactly: validation remains in the existing
        # binding/native coefficient parser.
        return dct_coeffs  # type: ignore[return-value]
    if coefficients is _COEFFICIENTS_NOT_PROVIDED or coefficients is None:
        return "all"
    if isinstance(coefficients, (str, bytes, bytearray)):
        raise TypeError("coefficients must be None or an iterable of integer indices")
    try:
        raw_values = list(islice(iter(coefficients), 65))
    except TypeError as error:
        raise TypeError(
            "coefficients must be None or an iterable of integer indices"
        ) from error
    if not 1 <= len(raw_values) <= 64:
        raise ValueError(_INVALID_COEFFICIENTS)

    values: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, bool):
            raise TypeError("coefficient indices must be integers")
        try:
            value = operator.index(raw_value)
        except TypeError as error:
            raise TypeError("coefficient indices must be integers") from error
        if not 0 <= value < 64 or value in seen:
            raise ValueError(_INVALID_COEFFICIENTS)
        seen.add(value)
        values.append(value)

    if values == list(range(64)):
        return "all"
    if values == list(range(len(values))):
        return f"first:{len(values)}"
    return "list:" + ",".join(str(value) for value in values)


def _normalize_image_ids(image_ids: Iterable[int], image_count: int) -> list[int]:
    values: list[int] = []
    for raw_value in image_ids:
        if isinstance(raw_value, bool):
            raise TypeError("image ids must be integers")
        try:
            value = operator.index(raw_value)
        except TypeError as error:
            raise TypeError("image ids must be integers") from error
        if not 0 <= value < image_count or value >= 1 << 32:
            raise ValueError(f"image id {value} is out of range")
        values.append(value)
    return values


def _load_native_module(module_path: Path | None) -> ModuleType:
    if module_path is not None:
        resolved = str(module_path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        return importlib.import_module("_galp_direct_dct")
    importlib.import_module("torch")
    try:
        return importlib.import_module("galp.torch._galp_direct_dct")
    except ModuleNotFoundError as error:
        if error.name != "galp.torch._galp_direct_dct":
            raise
    return importlib.import_module("_galp_direct_dct")


def _profile_id(profile: DirectDctProfile | str) -> str:
    if isinstance(profile, DirectDctProfile):
        return profile.id
    if isinstance(profile, str) and profile:
        return profile
    raise TypeError("profile must be a DirectDctProfile or non-empty profile id")


@dataclass(frozen=True, slots=True)
class DirectDctMetrics:
    """Versioned, implementation-independent Direct-DCT observations.

    ``complete`` is false while native GPU timing events are still pending.
    Reading metrics is always non-blocking; use a natural application
    synchronization boundary before requiring final timing values.
    """

    complete: bool
    consumer_wait_ms: float
    submit_to_ready_ms: float
    producer_ms: float
    planning_ms: float
    io_ms: float
    decode_ms: float
    transform_ms: float
    logical_bytes: int
    physical_bytes: int
    peak_transient_bytes: int

    @classmethod
    def _from_native(cls, values: Mapping[str, Any]) -> "DirectDctMetrics":
        if values.get("schema") != METRICS_SCHEMA:
            raise RuntimeError("native Direct-DCT metrics schema is incompatible")
        return cls(
            complete=bool(values["complete"]),
            consumer_wait_ms=float(values["consumer_wait_ms"]),
            submit_to_ready_ms=float(values["submit_to_ready_ms"]),
            producer_ms=float(values["producer_ms"]),
            planning_ms=float(values["planning_ms"]),
            io_ms=float(values["io_ms"]),
            decode_ms=float(values["decode_ms"]),
            transform_ms=float(values["transform_ms"]),
            logical_bytes=int(values["logical_bytes"]),
            physical_bytes=int(values["physical_bytes"]),
            peak_transient_bytes=int(values["peak_transient_bytes"]),
        )


class DirectDctBatch:
    """Model-ready tensors plus stable request metadata.

    When tensors are consumed on a CUDA stream other than the stream where
    their properties were accessed, call :meth:`record_stream` before
    submitting that consumer work. This explicitly forwards the true consumer
    dependency to GALP's native-backed storage lifetime mechanism.
    """

    __slots__ = ("_native", "profile_id")

    def __init__(self, native_batch: Any, profile_id: str) -> None:
        self._native = native_batch
        self.profile_id = profile_id

    @property
    def y(self) -> Any:
        return self._native.y

    @property
    def cbcr(self) -> Any:
        return self._native.cbcr

    @property
    def coefficients(self) -> Any:
        return self._native.coefficients

    @property
    def projected(self) -> Any:
        """CNN CUDA NCHW output; retains the native storage lease."""
        return self._native.projected

    def projected_range(self, first: int, count: int) -> Any:
        """Return a zero-copy view, waiting only for this range's producers."""
        return self._native.projected_range(first, count)

    @property
    def tensors(self) -> tuple[Any, Any]:
        return self.y, self.cbcr

    def record_stream(self, stream: Any | None = None) -> None:
        """Register the actual CUDA consumer stream before submitting work.

        With no argument, the current PyTorch CUDA stream is registered. An
        explicit ``torch.cuda.Stream`` can be supplied from any host context.
        This method is required when tensors were obtained on one stream and
        will be consumed on another; it never synchronizes the host. Work
        submitted to the registered stream before the final related
        Tensor/Storage reference is released is protected. Work submitted
        after that release is outside the lifetime contract.
        """

        if stream is None:
            self._native.record_stream()
            return
        try:
            stream_identity = int(stream.cuda_stream)
            cuda_device = int(stream.device_index)
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("stream must be a torch.cuda.Stream") from error
        self._native.record_stream(stream_identity, cuda_device)

    @property
    def global_image_ids(self) -> list[int]:
        return [int(value) for value in self._native.global_image_ids]

    @property
    def sample_ids(self) -> list[int]:
        """Pythonic alias for :attr:`global_image_ids`.

        This does not touch Tensor or native backing storage; it has the same
        metadata materialization behavior as the compatibility property.
        """

        return self.global_image_ids

    @property
    def transform_descriptors(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._native.transform_descriptors]

    @property
    def layout(self) -> str:
        return str(self._native.layout)

    @property
    def metrics(self) -> DirectDctMetrics:
        """Return a non-blocking snapshot of this batch's stable metrics.

        Event-derived GPU timings are valid when ``complete`` is true. Reading
        this property never adds a synchronization point to the model hot path.
        """

        return DirectDctMetrics._from_native(dict(self._native.metrics))


class _ProjectedBatch:
    """A model batch within one native-owned CNN preparation batch."""

    __slots__ = ("_native", "_first", "_count", "_ids")

    def __init__(self, native: Any, ids: list[int], first: int, count: int) -> None:
        self._native = native
        self._first, self._count, self._ids = first, count, ids

    @property
    def projected(self) -> Any:
        return self._native.projected_range(self._first, self._count)

    def projected_range(self, first: int, count: int) -> Any:
        if first < 0 or count < 0 or first + count > self._count:
            raise IndexError("projected range exceeds this model batch")
        return self._native.projected_range(self._first + first, count)

    @property
    def global_image_ids(self) -> list[int]:
        return self._ids[self._first:self._first + self._count]

    @property
    def sample_ids(self) -> list[int]:
        return self.global_image_ids

    def record_stream(self, stream: Any | None = None) -> None:
        """Register a consumer of this view with the native backing lease."""
        DirectDctBatch.record_stream(self, stream)


def _projected_batches(native_pipeline: Any, batch_size: int) -> Iterator[_ProjectedBatch]:
    # Only split output views. Submission, lookahead and backing ownership
    # remain native; never split the preparation requests into smaller reads.
    for native_batch in native_pipeline:
        ids = list(native_batch.global_image_ids)
        for first in range(0, len(ids), batch_size):
            yield _ProjectedBatch(native_batch, ids, first, min(batch_size, len(ids) - first))


class DirectDctPipeline:
    """Model-facing iterator over existing inference and training backends.

    ``reset`` accepts explicit preparation batches, just like the reader API.
    CNN inference yields at most ``batch_size`` images using native range
    views, without changing those preparation batches. Registered Transformer
    profiles retain their explicit batch boundaries. Training delegates epoch
    ordering, preparation pools and microbatches to PLS.
    """

    __slots__ = ("_native", "_image_count", "profile_id", "_training", "_batch_size", "_views")

    def __init__(
        self,
        manifest: str | Path,
        *,
        profile: DctModelProfile | DirectDctProfile | str,
        batch_size: int = 64,
        training: TrainingPolicy | None = None,
        module_path: str | Path | None = None,
        native_module: ModuleType | Any | None = None,
    ) -> None:
        batch_size = operator.index(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._training = training is not None
        self._batch_size = 0
        self._views = None
        if training is not None:
            from .experimental import DirectDctPlsPipeline

            projected = isinstance(profile, DctModelProfile)
            if projected and profile.source_frequency_policy != "all":
                raise ValueError("CNN PLS training requires all source frequencies")
            self.profile_id = "cnn-source512-v1" if projected else _profile_id(profile)
            self._native = DirectDctPlsPipeline(
                manifest, training.mapping,
                training_seed=training.seed,
                expected_mapping_sha256=training.expected_mapping_sha256,
                crop_policy=training.crop_policy,
                order_policy=training.order_policy,
                segments_per_pool=training.segments_per_pool,
                segment_images=training.segment_images,
                microbatch_images=batch_size,
                model_classes=training.model_classes,
                profile="rgbnomore-training-pls-v1" if projected else profile,
                output_grid_size=profile.output_grid_size if projected else 0,
                output_channels=profile._native_channels() if projected else None,
                module_path=module_path, native_module=native_module,
            )
            self._image_count = self._native.sample_count
        else:
            reader = DirectDctReader(manifest, module_path=module_path, native_module=native_module)
            self._image_count = reader.image_count
            if isinstance(profile, DctModelProfile):
                self.profile_id = "cnn-source512-v1"
                self._batch_size = batch_size
                self._native = reader._native.pipeline_batch_options(
                    **_cnn_pipeline_options(profile), output_batch_images=batch_size,
                )
            else:
                self.profile_id = reader.profile_info(profile)["id"]
                self._native = reader._native.pipeline(self.profile_id, dct_coeffs="all")

    @classmethod
    def _from_native(cls, native_pipeline: Any, profile_id: str, image_count: int) -> "DirectDctPipeline":
        pipeline = cls.__new__(cls)
        pipeline._native = native_pipeline
        pipeline._image_count = image_count
        pipeline.profile_id = profile_id
        pipeline._training = False
        pipeline._batch_size = 0
        pipeline._views = None
        return pipeline

    def start(
        self,
        image_id_batches: Iterable[Sequence[int]],
        *,
        transforms_by_batch: Sequence[Sequence[Mapping[str, Any]] | None] | None = None,
    ) -> "DirectDctPipeline":
        """Consume the complete finite schedule before the first batch is returned."""

        if self._training:
            raise ValueError("training pipelines use start_epoch(), not explicit batch schedules")
        batches = [
            _normalize_image_ids(batch, self._image_count)
            for batch in image_id_batches
        ]
        if any(not batch for batch in batches):
            raise ValueError("Direct-DCT batches must not be empty")
        native_transforms: list[list[dict[str, Any]] | None] | None = None
        if transforms_by_batch is not None:
            if len(transforms_by_batch) != len(batches):
                raise ValueError(
                    "transforms_by_batch must contain exactly one entry per batch"
                )
            native_transforms = [
                None if transforms is None else [dict(value) for value in transforms]
                for transforms in transforms_by_batch
            ]
        if self._views is not None:
            self._views.close()
        self._native.reset(batches, transforms_by_batch=native_transforms)
        if self._batch_size:
            self._views = _projected_batches(self._native, self._batch_size)
        return self

    reset = start

    def start_epoch(self, epoch: int) -> "DirectDctPipeline":
        if not self._training:
            raise ValueError("inference pipelines use reset(), not start_epoch()")
        self._native.start_epoch(epoch)
        return self

    def __iter__(self) -> "DirectDctPipeline":
        return self

    def __next__(self) -> DirectDctBatch | _ProjectedBatch | DirectDctPlsMicrobatch:
        if self._training:
            return next(self._native)
        if self._batch_size:
            if self._views is None:
                raise RuntimeError("call reset() before iterating the pipeline")
            return next(self._views)
        return DirectDctBatch(next(self._native), self.profile_id)

    def close(self) -> None:
        if self._views is not None:
            self._views.close()
        self._native.close()

    @property
    def metrics(self) -> DirectDctMetrics:
        """Return non-blocking cumulative metrics for consumed batches.

        The native pipeline retains only the batches whose timing events are
        pending and finalizes them opportunistically. After the application's
        normal CUDA synchronization boundary, ``complete`` must be true.
        """

        if self._training:
            raise AttributeError("aggregate metrics are not exposed by the PLS backend")
        return DirectDctMetrics._from_native(dict(self._native.metrics))

    def __enter__(self) -> "DirectDctPipeline":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class _DirectDctBatchIterator(Iterator[DirectDctBatch]):
    """Context-manageable iterator owning one existing native pipeline."""

    __slots__ = ("_closed", "_pipeline")

    def __init__(self, pipeline: DirectDctPipeline) -> None:
        self._pipeline = pipeline
        self._closed = False

    def __iter__(self) -> "_DirectDctBatchIterator":
        return self

    def __next__(self) -> DirectDctBatch:
        if self._closed:
            raise StopIteration
        try:
            return next(self._pipeline)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pipeline.close()

    def __enter__(self) -> "_DirectDctBatchIterator":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            # Destructors cannot safely report close errors. Explicit ``with``
            # or ``close`` remains the deterministic early-exit contract.
            pass


class DirectDctReader:
    """Read model-ready Direct-DCT batches from a GALP manifest.

    A profile defines stable processing semantics and the output contract;
    explicit input selections such as ``coefficients`` complete the request.
    Native scheduling, allocator, I/O, cache, and launch settings are
    intentionally absent from this API.

    ``module_path`` and ``native_module`` are development/testing compatibility
    parameters. Installed applications should normally pass only the manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        module_path: str | Path | None = None,
        native_module: ModuleType | Any | None = None,
    ) -> None:
        path = Path(manifest_path).resolve()
        binding_started = time.perf_counter()
        module = native_module or _load_native_module(
            Path(module_path) if module_path is not None else None
        )
        self._binding_import_ms = (time.perf_counter() - binding_started) * 1000.0
        if (
            getattr(module, "DIRECT_DCT_BINDING_SCHEMA", None) != BINDING_SCHEMA
            or getattr(module, "DIRECT_DCT_PROFILE_SCHEMA", None) != PROFILE_SCHEMA
            or getattr(module, "DIRECT_DCT_METRICS_SCHEMA", None) != METRICS_SCHEMA
        ):
            raise RuntimeError(
                "GALP native binding does not implement the public Direct-DCT pipeline API; "
                "rebuild the binding"
            )
        self._module = module
        self._native = module.DirectDctReader(str(path))
        self._validated_profiles: dict[str, dict[str, Any]] = {}

    @property
    def image_count(self) -> int:
        return int(self._native.image_count)

    def profile_info(self, profile: DirectDctProfile | str) -> dict[str, Any]:
        profile_id = _profile_id(profile)
        if profile_id not in self._validated_profiles:
            info = dict(self._module.direct_dct_profile_info(profile_id))
            if info.get("schema") != PROFILE_SCHEMA or info.get("id") != profile_id:
                raise RuntimeError(
                    f"native Direct-DCT profile metadata is invalid for {profile_id!r}"
                )
            self._validated_profiles[profile_id] = info
        return dict(self._validated_profiles[profile_id])

    def pipeline(
        self,
        profile: DirectDctProfile | str,
        *,
        coefficients: Iterable[int] | None = cast(
            Any, _COEFFICIENTS_NOT_PROVIDED
        ),
        dct_coeffs: str = cast(Any, _DCT_COEFFS_NOT_PROVIDED),
    ) -> DirectDctPipeline:
        """Create a reusable native pipeline for a semantic profile.

        ``coefficients`` is the recommended Python API. ``None`` selects all
        coefficients; an iterable preserves its explicit order. ``dct_coeffs``
        remains supported as the legacy string API. Supplying both is an error.
        """

        canonical_coefficients = _normalize_coefficients(
            coefficients=coefficients, dct_coeffs=dct_coeffs
        )
        profile_id = self.profile_info(profile)["id"]
        return DirectDctPipeline._from_native(
            self._native.pipeline(
                profile_id, dct_coeffs=canonical_coefficients
            ),
            profile_id,
            self.image_count,
        )

    def read(
        self,
        image_ids: Sequence[int],
        profile: DirectDctProfile | str,
        *,
        coefficients: Iterable[int] | None = cast(
            Any, _COEFFICIENTS_NOT_PROVIDED
        ),
        dct_coeffs: str = cast(Any, _DCT_COEFFS_NOT_PROVIDED),
        transforms: Sequence[Mapping[str, Any]] | None = None,
    ) -> DirectDctBatch:
        """Read one logical batch through the existing native read path.

        ``coefficients`` is normalized to the binding's established
        ``all``/``first:N``/``list:...`` representation. ``dct_coeffs`` remains
        a compatibility parameter and cannot be combined with it.
        """

        canonical_coefficients = _normalize_coefficients(
            coefficients=coefficients, dct_coeffs=dct_coeffs
        )
        profile_id = self.profile_info(profile)["id"]
        native_transforms = (
            None if transforms is None else [dict(value) for value in transforms]
        )
        native_batch = self._native.read(
            _normalize_image_ids(image_ids, self.image_count),
            profile_id,
            dct_coeffs=canonical_coefficients,
            transforms=native_transforms,
        )
        return DirectDctBatch(native_batch, profile_id)

    def iter_batches(
        self,
        logical_batches: Iterable[Sequence[int]],
        *,
        profile: DirectDctProfile | str,
        coefficients: Iterable[int] | None = cast(
            Any, _COEFFICIENTS_NOT_PROVIDED
        ),
        dct_coeffs: str = cast(Any, _DCT_COEFFS_NOT_PROVIDED),
        transforms_by_batch: Sequence[Sequence[Mapping[str, Any]] | None]
        | None = None,
    ) -> Iterator[DirectDctBatch]:
        """Iterate a finite schedule using exactly ``pipeline()`` + ``start()``.

        The input is fully consumed before the first batch is returned.
        The returned iterator also supports ``with`` and ``close`` for
        deterministic early-exit cleanup. It introduces no producer, queue,
        submission protocol, or execution path of its own.
        """

        pipeline = self.pipeline(
            profile,
            coefficients=coefficients,
            dct_coeffs=dct_coeffs,
        )
        try:
            pipeline.start(
                logical_batches, transforms_by_batch=transforms_by_batch
            )
        except BaseException:
            pipeline.close()
            raise
        return _DirectDctBatchIterator(pipeline)

__all__ = [
    "TrainingPolicy",
    "DirectDctBatch",
    "DirectDctMetrics",
    "DirectDctPipeline",
    "DirectDctReader",
    "METRICS_SCHEMA",
    "PROFILE_SCHEMA",
]
