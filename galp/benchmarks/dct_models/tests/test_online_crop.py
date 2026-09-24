"""CPU checks of online geometry and source-frequency dependencies."""
import unittest
from pathlib import Path

import torch

from galp.benchmarks.dct_models.online_crop import SourceReference

from galp.benchmarks.common import DEFAULT_RGBNOMORE_ROOT


class OnlineCropTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.upstream = DEFAULT_RGBNOMORE_ROOT

    def inputs(self):
        return ([torch.zeros(h, h, 64, dtype=torch.int16) for h in (64, 32, 32)],
                [torch.ones(64, dtype=torch.int16) for _ in range(3)])

    def test_constant_survives_crop_dequantization_and_resize(self):
        q, tables = self.inputs()
        for plane, table in zip(q, tables):
            plane[..., 0] = 80
            table.fill_(2)
        for grid in (28, 56, 112):
            for actual in SourceReference(self.upstream, grid)(q, tables):
                expected = torch.zeros(grid, grid, 64)
                expected[..., 0] = 160
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)

    def test_crop_discards_border_without_resizing_it_into_output(self):
        q, tables = self.inputs()
        for plane, border in zip(q, (4, 2, 2)):
            plane[:border] = 100
            plane[-border:] = 100
            plane[:, :border] = 100
            plane[:, -border:] = 100
        for grid in (28, 56, 112):
            for actual in SourceReference(self.upstream, grid)(q, tables):
                self.assertEqual(int(torch.count_nonzero(actual)), 0)

    def test_discarded_source_frequency_contributes_to_output_dc(self):
        q, tables = self.inputs()
        # Chroma (0,7) is absent from DCT24's four output chroma channels.
        # It nevertheless contributes to output DC after splitting a block.
        q[1][16, 16, 7] = 100
        for grid in (56, 112):
            actual = SourceReference(self.upstream, grid)(q, tables)[1]
            self.assertGreater(float(actual[..., 0].abs().max()), 1.)

    def test_rejects_precomputed_target_as_full_source(self):
        q, tables = self.inputs()
        q[0] = q[0][4:-4, 4:-4]
        with self.assertRaisesRegex(ValueError, "full source component"):
            SourceReference(self.upstream, 56)(q, tables)

    def test_multi_shard_activation_preserves_noncontiguous_ids_and_tail(self):
        from galp.benchmarks.dct_models.evaluate_shards import activation_groups
        shards = [dict(shard_id=i, first_global_image_index=i*1024,
                       image_count=1024, rowgroup_count=48) for i in (0, 3, 4, 9, 11)]
        groups = activation_groups(shards, 4)
        self.assertEqual(groups[0]["shard_ids"], [0, 3, 4, 9])
        self.assertEqual(groups[0]["image_ids"][1024], 3072)
        self.assertEqual(groups[0]["image_count"], 4096)
        self.assertEqual(groups[1]["image_ids"], list(range(11264, 12288)))
        self.assertEqual(sum(g["rowgroup_count"] for g in groups), 240)


if __name__ == "__main__":
    unittest.main()
