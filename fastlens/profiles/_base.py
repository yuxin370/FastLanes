"""Common immutable profile descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DirectDctProfile:
    """Stable Direct-DCT processing semantics and output contract.

    A final request combines this profile with explicit input selections made
    by the caller. The native runtime owns all scheduling, cache, I/O, and
    kernel launch details; they are intentionally absent from this type.
    """

    id: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("DirectDctProfile.id must not be empty")
