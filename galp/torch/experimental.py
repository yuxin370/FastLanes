"""Explicitly experimental PyTorch adapters for GALP Direct-DCT extensions.

Names in this module are not part of :mod:`galp.torch`'s stable umbrella.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from galp.profiles import DirectDctProfile

from .direct_dct import _load_native_module, _profile_id


class DirectDctPlsMicrobatch:
    """Zero-copy model boundary for one native PLS microbatch."""

    __slots__ = ("_native",)

    def __init__(self, native_microbatch: Any) -> None:
        self._native = native_microbatch

    @property
    def y(self) -> Any:
        return self._native.y

    @property
    def cbcr(self) -> Any:
        return self._native.cbcr

    @property
    def targets(self) -> Any:
        return self._native.targets

    @property
    def tensors(self) -> tuple[Any, Any, Any]:
        return self.y, self.cbcr, self.targets

    def record_stream(self, stream: Any | None = None) -> None:
        """Register the actual stream before submitting cross-stream work.

        Work submitted after the final related tensor reference is released is
        outside this lifetime contract.
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
    def epoch(self) -> int:
        return int(self._native.epoch)

    @property
    def pool_index(self) -> int:
        return int(self._native.pool_index)

    @property
    def microbatch_index_in_pool(self) -> int:
        return int(self._native.microbatch_index_in_pool)

    @property
    def pool_offset(self) -> int:
        return int(self._native.pool_offset)

    @property
    def image_count(self) -> int:
        return int(self._native.image_count)

    @property
    def is_pool_end(self) -> bool:
        return bool(self._native.is_pool_end)

    @property
    def global_image_ids(self) -> list[int]:
        return [int(value) for value in self._native.global_image_ids]

    @property
    def labels(self) -> list[int]:
        return [int(value) for value in self._native.labels]


class DirectDctPlsPool:
    """One native-owned, closed GPU pool containing physical PLS segments."""

    __slots__ = ("_native",)

    def __init__(self, native_pool: Any) -> None:
        self._native = native_pool

    def __iter__(self) -> "DirectDctPlsPool":
        return self

    def __next__(self) -> DirectDctPlsMicrobatch:
        return DirectDctPlsMicrobatch(next(self._native))

    def microbatch(self, index: int) -> DirectDctPlsMicrobatch:
        return DirectDctPlsMicrobatch(self._native.microbatch(int(index)))

    def retire(self) -> None:
        """Retire this scheduling context after submitting its model work.

        Tensor backing remains protected independently by the native
        producer/consumer completion contract.
        """

        self._native.retire()

    @property
    def epoch(self) -> int:
        return int(self._native.epoch)

    @property
    def pool_index(self) -> int:
        return int(self._native.pool_index)

    @property
    def image_count(self) -> int:
        return int(self._native.image_count)

    @property
    def microbatch_count(self) -> int:
        return int(self._native.microbatch_count)

    @property
    def virtual_pls_ids(self) -> list[int]:
        return [int(value) for value in self._native.virtual_pls_ids]

    @property
    def execution_stats(self) -> dict[str, Any]:
        return dict(self._native.execution_stats)


class DirectDctPlsPipeline:
    """Experimental thin adapter over the native physical-PLS pipeline."""

    __slots__ = ("_module", "_native")

    def __init__(
        self,
        manifest_path: str | Path,
        premixed_mapping_csv: str | Path,
        *,
        training_seed: int,
        expected_mapping_sha256: str,
        crop_policy: str = "per-pls",
        order_policy: str = "closed-pool",
        segments_per_pool: int = 4,
        microbatch_images: int = 64,
        segment_images: int = 1024,
        model_classes: int = 1000,
        profile: DirectDctProfile | str = "rgbnomore-training-pls-v1",
        module_path: str | Path | None = None,
        native_module: ModuleType | Any | None = None,
    ) -> None:
        module = native_module or _load_native_module(
            Path(module_path) if module_path is not None else None
        )
        if not hasattr(module, "DirectDctPlsPipeline"):
            raise RuntimeError(
                "GALP native binding does not implement the physical PLS pipeline; "
                "rebuild the binding"
            )
        self._module = module
        self._native = module.DirectDctPlsPipeline(
            str(Path(manifest_path).resolve()),
            str(Path(premixed_mapping_csv).resolve()),
            int(training_seed),
            str(expected_mapping_sha256),
            crop_policy=crop_policy,
            order_policy=order_policy,
            segments_per_pool=int(segments_per_pool),
            microbatch_images=int(microbatch_images),
            segment_images=int(segment_images),
            model_classes=int(model_classes),
            profile_id=_profile_id(profile),
        )

    def start_epoch(self, epoch: int) -> "DirectDctPlsPipeline":
        self._native.start_epoch(int(epoch))
        return self

    def next_pool(self) -> DirectDctPlsPool:
        return DirectDctPlsPool(self._native.next_pool())

    def close(self) -> None:
        self._native.close()

    def reclaim_finished_pools(self) -> int:
        """Compatibility hook; native leases remain the reclaim authority."""

        reclaim = getattr(self._module, "reclaim_direct_dct_pls_pools", None)
        if reclaim is None:
            raise RuntimeError(
                "GALP native binding does not expose PLS pool reclamation; rebuild the binding"
            )
        return int(reclaim())

    def __iter__(self) -> "DirectDctPlsPipeline":
        return self

    def __next__(self) -> DirectDctPlsMicrobatch:
        return DirectDctPlsMicrobatch(next(self._native))

    def __enter__(self) -> "DirectDctPlsPipeline":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def sample_count(self) -> int:
        return int(self._native.sample_count)

    @property
    def has_next_pool(self) -> bool:
        return bool(self._native.has_next_pool)

    @property
    def pls_count(self) -> int:
        return int(self._native.pls_count)

    @property
    def segment_images(self) -> int:
        return int(self._native.segment_images)

    @property
    def prefetch_stats(self) -> dict[str, Any]:
        return dict(self._native.prefetch_stats)


__all__ = [
    "DirectDctPlsMicrobatch",
    "DirectDctPlsPipeline",
    "DirectDctPlsPool",
]
