"""CPU allocation optimizations preserve values and caller-owned inputs."""
from pathlib import Path
import unittest

import numpy as np
import torch

from fastlens.benchmarks.baselines.dct_baselines.materialize import Materializer, organize


class MaterializeOptimizationTests(unittest.TestCase):
    def test_normalization_preserves_inputs_and_matches_separate_fp32_operations(self):
        for dtype in (torch.int16, torch.float32):
            output = tuple(torch.arange(-64, 64, dtype=dtype).reshape(2, 1, 8, 8) + c
                           for c in range(3))
            before = tuple(value.clone() for value in output)
            profiles = [dict(transform='rgbnomore_source512', training=training) for training in (False, True)]
            profiles.append(dict(transform='cnn_source512', output_channels=[[2, 9], [0, 7], [2, 3], [1, 1]],
                                 mean=[.123, -3.2, .77, 9.], std=[2.7, .17, 4.3, .9]))
            for profile in profiles:
                with self.subTest(dtype=dtype, profile=profile):
                    if profile['transform'] == 'cnn_source512':
                        selected = torch.stack([output[c].flatten(-2)[..., f] for c, f in profile['output_channels']]).float()
                        expected = ((selected - torch.tensor(profile['mean'])[:, None, None]) /
                                    torch.tensor(profile['std'])[:, None, None],)
                    else:
                        normalized = tuple((v.float()+4)/1020 if profile['training'] else
                                           -1.0+((v.float()+1024)/2040)*2.0 for v in output)
                        expected = (normalized[0][None], torch.stack(normalized[1:]))
                    for _ in range(2):
                        actual = organize(profile, output)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        torch.testing.assert_close(output, before, atol=0, rtol=0)

    @unittest.skipUnless((Path.home()/'RGB-no-more/utils/dct_ops.py').exists(), 'upstream resize unavailable')
    def test_repeated_resize_and_flip_do_not_modify_coefficients_tables_or_cached_matrices(self):
        rng = np.random.default_rng(704)
        for transform, grids in [('cnn_source512', [56]*3), ('rgbnomore_source512', [28, 14, 14])]:
            profile = dict(version=1, representation='jpeg_b8', transform=transform,
                           source_frequencies=[list(range(64))]*3, dependency_derivation='full source',
                           upstream=str(Path.home()/'RGB-no-more'), output_grids=grids,
                           output_channels=[[0, 1], [1, 8]], mean=[.5, -1.], std=[2., 4.])
            adapter = Materializer(profile)
            for dtype in (np.int16, np.int32):
                with self.subTest(transform=transform, dtype=dtype):
                    coefficients = tuple(rng.integers(-2048, 2048, (n, n, 8, 8), dtype=dtype)
                                         for n in (56, 28, 28))
                    tables = tuple(rng.integers(1, 65536, (8, 8), dtype=np.uint16) for _ in range(3))
                    before = tuple(value.copy() for value in (*coefficients, *tables))
                    row = dict(crop_box=[32, 32, 448, 448], flip=False)
                    expected = adapter(coefficients, tables, row, cropped=True)
                    matrices = {key: value.clone() for key, value in adapter.matrices.items()}
                    adapter(coefficients, tables, dict(row, flip=True), cropped=True)
                    actual = adapter(coefficients, tables, row, cropped=True)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for value, saved in zip((*coefficients, *tables), before):
                        np.testing.assert_array_equal(value, saved)
                    torch.testing.assert_close(adapter.matrices, matrices, atol=0, rtol=0)
