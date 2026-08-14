#!/usr/bin/env python3
"""CPU-only tests for persistent RGB-no-more payload fingerprints."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks/system_rgbnomore"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from shared.common import cached_file_fingerprints, file_identity  # noqa: E402


class PayloadFingerprintCacheTest(unittest.TestCase):
    _CACHE_FORMAT = "galp_shard_payload_fingerprints_v1"

    def _files(self, payload: Path) -> list[dict[str, object]]:
        return [
            {
                "kind": "fls",
                "relative_path": payload.name,
                "path": payload,
                "expected_size": payload.stat().st_size,
            }
        ]

    def test_device_remap_reuses_digest_without_rewriting_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "shard.fls"
            payload.write_bytes(b"stable payload")
            cache = root / "manifest.bin.payload_fingerprints.json"
            files = self._files(payload)
            cached_file_fingerprints(files, cache, cache_format=self._CACHE_FORMAT)

            current_identity = file_identity(payload)
            remapped_identity = dict(current_identity)
            remapped_identity["device"] += 1
            cached = json.loads(cache.read_text(encoding="utf-8"))
            cached["files"][0]["file_identity"] = remapped_identity
            cache.write_text(json.dumps(cached), encoding="utf-8")
            cache_before = cache.read_bytes()

            with mock.patch(
                "shared.common.fingerprint_file",
                side_effect=AssertionError("device remap must not rehash payload bytes"),
            ):
                fingerprints = cached_file_fingerprints(
                    files,
                    cache,
                    cache_format=self._CACHE_FORMAT,
                    allow_hash_misses=False,
                )

            self.assertEqual(fingerprints[0]["file_identity"], remapped_identity)
            self.assertEqual(cache.read_bytes(), cache_before)

    def test_content_snapshot_change_still_requires_explicit_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "shard.fls"
            payload.write_bytes(b"original payload")
            cache = root / "manifest.bin.payload_fingerprints.json"
            files = self._files(payload)
            cached_file_fingerprints(files, cache, cache_format=self._CACHE_FORMAT)

            cached = json.loads(cache.read_text(encoding="utf-8"))
            cached["files"][0]["file_identity"]["mtime_ns"] -= 1
            cache.write_text(json.dumps(cached), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing or stale"):
                cached_file_fingerprints(
                    files,
                    cache,
                    cache_format=self._CACHE_FORMAT,
                    allow_hash_misses=False,
                )


if __name__ == "__main__":
    unittest.main()
