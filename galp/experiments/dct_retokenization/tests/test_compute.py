#!/usr/bin/env python3

from __future__ import annotations

import unittest

from galp.experiments.dct_retokenization.compute import mac_table, prototype_macs


class ComputeAccountingTest(unittest.TestCase):
    def test_expected_order_and_approximate_values(self) -> None:
        rows = {int(row["tokens"]): row for row in mac_table()}
        self.assertGreater(prototype_macs(196), prototype_macs(98))
        self.assertGreater(prototype_macs(98), prototype_macs(49))
        self.assertAlmostEqual(float(rows[196]["prototype_total_gmac"]), 1.232, delta=0.002)
        self.assertAlmostEqual(float(rows[98]["prototype_total_gmac"]), 0.586, delta=0.002)
        self.assertAlmostEqual(float(rows[49]["prototype_total_gmac"]), 0.293, delta=0.002)
        self.assertAlmostEqual(float(rows[98]["theoretical_reduction_x"]), 2.10, delta=0.02)
        self.assertAlmostEqual(float(rows[49]["theoretical_reduction_x"]), 4.21, delta=0.03)


if __name__ == "__main__":
    unittest.main()
