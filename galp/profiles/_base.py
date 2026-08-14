"""Common immutable profile descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DirectDctProfile:
    """A stable semantic Direct-DCT output contract.

    The native runtime owns all scheduling, cache, I/O, and kernel launch
    details. They are intentionally absent from this application-facing type.
    """

    id: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("DirectDctProfile.id must not be empty")
