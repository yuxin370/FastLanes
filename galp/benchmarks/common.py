"""Shared file and JPEG metadata helpers for benchmark workloads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


DEFAULT_RGBNOMORE_ROOT = Path(os.environ.get("RGBNOMORE_ROOT", Path.home() / "RGB-no-more"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampling_mode(reader: Any, image_id: int) -> str:
    components = reader.image_metadata(int(image_id)).get("components", [])
    by_slot = {
        int(component.get("semantic_slot_id")): component
        for component in components
        if component.get("present") and int(component.get("semantic_slot_id", -1)) in (0, 1, 2)
    }
    if 0 not in by_slot:
        by_local = {
            int(component.get("local_component_index")): component
            for component in components
            if component.get("present") and int(component.get("local_component_index", -1)) in (0, 1, 2)
        }
        by_slot = by_local
    if 0 in by_slot and 1 not in by_slot and 2 not in by_slot:
        return "grayscale"
    if not {0, 1, 2}.issubset(by_slot):
        return "unknown"
    y = by_slot[0]
    cb = by_slot[1]
    cr = by_slot[2]
    if (
        int(cb.get("h_samp_factor", 0)) != int(cr.get("h_samp_factor", 0))
        or int(cb.get("v_samp_factor", 0)) != int(cr.get("v_samp_factor", 0))
    ):
        return "unsupported_mismatched_chroma"
    if (
        int(cb.get("h_samp_factor", 0)) == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) == int(y.get("v_samp_factor", 0))
    ):
        return "4:4:4"
    if (
        int(cb.get("h_samp_factor", 0)) * 2 == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) * 2 == int(y.get("v_samp_factor", 0))
    ):
        return "4:2:0"
    if (
        int(cb.get("h_samp_factor", 0)) * 2 == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) == int(y.get("v_samp_factor", 0))
    ):
        return "4:2:2"
    if (
        int(cb.get("h_samp_factor", 0)) == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) * 2 == int(y.get("v_samp_factor", 0))
    ):
        return "4:4:0"
    if (
        int(cb.get("h_samp_factor", 0)) * 4 == int(y.get("h_samp_factor", 0))
        and int(cb.get("v_samp_factor", 0)) == int(y.get("v_samp_factor", 0))
    ):
        return "4:1:1"
    return "unsupported"
