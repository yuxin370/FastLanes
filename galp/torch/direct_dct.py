"""Public Python facade for GALP's private Direct-DCT extension."""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from galp.profiles import DirectDctProfile


PROFILE_SCHEMA = "galp-direct-dct-profile-v1"


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


class DirectDctBatch:
    """Model-ready tensors plus stable request metadata."""

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

    @property
    def global_image_ids(self) -> list[int]:
        return [int(value) for value in self._native.global_image_ids]

    @property
    def transform_descriptors(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._native.transform_descriptors]

    @property
    def layout(self) -> str:
        return str(self._native.layout)

    def record_stream(self) -> None:
        """Keep this batch alive for work already queued on the current stream."""

        self._native.record_stream()


class DirectDctFuture:
    """Single-consumer future returned by :meth:`DirectDctReader.prefetch`."""

    __slots__ = ("_native", "profile_id")

    def __init__(self, native_future: Any, profile_id: str) -> None:
        self._native = native_future
        self.profile_id = profile_id

    @property
    def ready(self) -> bool:
        return bool(self._native.ready)

    @property
    def started(self) -> bool:
        return bool(self._native.started)

    @property
    def active(self) -> bool:
        return bool(self._native.active)

    @property
    def finished(self) -> bool:
        return bool(self._native.finished)

    def read(self) -> DirectDctBatch:
        return DirectDctBatch(self._native.read(), self.profile_id)

    def cancel(self) -> bool:
        return bool(self._native.cancel())

    def _release_submission(self) -> bool:
        """Private compatibility hook for the benchmark overlap coordinator."""

        return bool(self._native.release_submission())


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
        if getattr(module, "DIRECT_DCT_PROFILE_SCHEMA", None) != PROFILE_SCHEMA:
            raise RuntimeError(
                "GALP native binding does not implement the public Direct-DCT profile API; "
                "rebuild the binding"
            )
        self._module = module
        self._native = module.DirectDctReader(str(path))
        self._validated_profiles: dict[str, dict[str, Any]] = {}

    @property
    def image_count(self) -> int:
        return int(self._native.image_count)

    @property
    def binding_import_ms(self) -> float:
        return self._binding_import_ms

    @property
    def initialization_stats(self) -> dict[str, Any]:
        return dict(self._native.initialization_stats)

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

    def plan(
        self,
        image_ids: Sequence[int],
        profile: DirectDctProfile | str,
        *,
        transforms: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        profile_id = self.profile_info(profile)["id"]
        native_transforms = (
            None if transforms is None else [dict(value) for value in transforms]
        )
        return dict(
            self._native.plan(
                [int(value) for value in image_ids],
                profile_id,
                transforms=native_transforms,
            )
        )

    def prefetch(
        self,
        image_ids: Sequence[int],
        profile: DirectDctProfile | str,
        *,
        transforms: Sequence[Mapping[str, Any]] | None = None,
    ) -> DirectDctFuture:
        profile_id = self.profile_info(profile)["id"]
        native_transforms = (
            None if transforms is None else [dict(value) for value in transforms]
        )
        native_future = self._native.prefetch(
            [int(value) for value in image_ids],
            profile_id,
            transforms=native_transforms,
        )
        return DirectDctFuture(native_future, profile_id)

    def read(
        self,
        image_ids: Sequence[int],
        profile: DirectDctProfile | str,
        *,
        transforms: Sequence[Mapping[str, Any]] | None = None,
    ) -> DirectDctBatch:
        profile_id = self.profile_info(profile)["id"]
        native_transforms = (
            None if transforms is None else [dict(value) for value in transforms]
        )
        native_batch = self._native.read(
            [int(value) for value in image_ids],
            profile_id,
            transforms=native_transforms,
        )
        return DirectDctBatch(native_batch, profile_id)

    def image_metadata(self, global_image_index: int) -> dict[str, Any]:
        return dict(self._native.image_metadata(int(global_image_index)))

    def rowgroup_storage_bytes(
        self, shard_id: int, rowgroup_indices: Sequence[int]
    ) -> int:
        return int(
            self._native.rowgroup_storage_bytes(
                int(shard_id), [int(value) for value in rowgroup_indices]
            )
        )


__all__ = ["DirectDctBatch", "DirectDctFuture", "DirectDctReader", "PROFILE_SCHEMA"]
