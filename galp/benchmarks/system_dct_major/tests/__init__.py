"""Tests for the isolated DCT-major benchmark."""

from pathlib import Path
import sys


# The formal command executes this package from the repository root while the
# benchmark's operational entry points also support ``python -m training_pls``.
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
