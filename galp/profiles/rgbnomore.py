"""RGB-no-more Direct-DCT semantic profiles.

Only model-facing geometry and numeric semantics are represented by these
objects.  Storage reads, prefetching, stream policy, and kernel launch tuning
are selected by the native implementation.
"""

from __future__ import annotations

from ._base import DirectDctProfile


VALIDATION = DirectDctProfile(
    id="rgbnomore-validation-v1",
)

VALIDATION_CENTER_CROP_512 = DirectDctProfile(
    id="rgbnomore-validation-center-crop-512-v1",
)

__all__ = ["VALIDATION", "VALIDATION_CENTER_CROP_512"]
