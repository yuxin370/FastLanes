"""Public Python facade for GALP's private Direct-DCT extension."""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from galp.profiles import DirectDctProfile


PROFILE_SCHEMA = "galp-direct-dct-profile-v1"
METRICS_SCHEMA = "galp-direct-dct-metrics-v2"
BINDING_SCHEMA = "galp-direct-dct-binding-v2"


def _load_native_module(module_path: Path | None) -> ModuleType:
    if module_path is not None:
        resolved = str(module_path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
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


class DirectDctPipeline:
    """Native-owned bounded iterator for one semantic Direct-DCT profile."""

    __slots__ = ("_native", "profile_id")

    def __init__(self, native_pipeline: Any, profile_id: str) -> None:
        self._native = native_pipeline
        self.profile_id = profile_id

    def start(
        self,
        image_id_batches: Sequence[Sequence[int]],
        *,
        transforms_by_batch: Sequence[Sequence[Mapping[str, Any]] | None] | None = None,
    ) -> "DirectDctPipeline":
        batches = [[int(value) for value in batch] for batch in image_id_batches]
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
        self._native.reset(batches, transforms_by_batch=native_transforms)
        return self

    def __iter__(self) -> "DirectDctPipeline":
        return self

    def __next__(self) -> DirectDctBatch:
        return DirectDctBatch(next(self._native), self.profile_id)

    def close(self) -> None:
        self._native.close()

    @property
    def metrics(self) -> DirectDctMetrics:
        """Return non-blocking cumulative metrics for consumed batches.

        The native pipeline retains only the batches whose timing events are
        pending and finalizes them opportunistically. After the application's
        normal CUDA synchronization boundary, ``complete`` must be true.
        """

        return DirectDctMetrics._from_native(dict(self._native.metrics))

    def __enter__(self) -> "DirectDctPipeline":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class DirectDctReader:
    """Read model-ready Direct-DCT batches from a GALP manifest.

    The caller chooses only a semantic profile.  Native scheduling, allocator,
    I/O, cache, and launch settings are intentionally absent from this API.
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
        dct_coeffs: str = "all",
    ) -> DirectDctPipeline:
        """Create a reusable native pipeline for a semantic profile.

        ``dct_coeffs`` selects raw JPEG zigzag columns before dequantization
        and frequency mixing.  It does not expose or alter the profile's
        native runtime policy.
        """

        profile_id = self.profile_info(profile)["id"]
        return DirectDctPipeline(
            self._native.pipeline(profile_id, dct_coeffs=dct_coeffs), profile_id
        )

    def read(
        self,
        image_ids: Sequence[int],
        profile: DirectDctProfile | str,
        *,
        dct_coeffs: str = "all",
        transforms: Sequence[Mapping[str, Any]] | None = None,
    ) -> DirectDctBatch:
        profile_id = self.profile_info(profile)["id"]
        native_transforms = (
            None if transforms is None else [dict(value) for value in transforms]
        )
        native_batch = self._native.read(
            [int(value) for value in image_ids],
            profile_id,
            dct_coeffs=dct_coeffs,
            transforms=native_transforms,
        )
        return DirectDctBatch(native_batch, profile_id)

__all__ = [
    "DirectDctBatch",
    "DirectDctMetrics",
    "DirectDctPipeline",
    "DirectDctReader",
    "METRICS_SCHEMA",
    "PROFILE_SCHEMA",
]
