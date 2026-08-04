from __future__ import annotations

import sys
import unittest
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from diagnostics.crop_ab import _mean_native  # noqa: E402


class CropAbTest(unittest.TestCase):
    def test_native_counter_aliases_use_first_available_counter(self) -> None:
        result = {
            "execution": {"aggregate_exclude_first_repeat": False},
            "repeats": [
                {"native_totals": {"selected_vector_count": 10}},
                {"native_totals": {"actual_vector_count": 14, "selected_vector_count": 99}},
            ],
        }
        self.assertEqual(
            _mean_native(result, "actual_vector_count", "selected_vector_count"),
            12.0,
        )


if __name__ == "__main__":
    unittest.main()
