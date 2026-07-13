#!/usr/bin/env python3
"""Convenience entry point for the GALP image-order benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent / "image_order_benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run import main  # noqa: E402


if __name__ == "__main__":
    main()
