"""Reader optimizations retain frequency order, ownership and physical reads."""
import math

import numpy as np

from fastlens.benchmarks.baselines.dct_baselines.contracts import ReadRequest
from fastlens.benchmarks.baselines.dct_baselines.runner import inference_loader, inputs
from fastlens.benchmarks.baselines.dct_baselines.source import source_images
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store
from .test_baselines import TemporaryTest, image
from . import test_round4


class ReaderOptimizationTests(TemporaryTest):
    def test_dense_shares_view_memory_without_aliasing_cache_or_other_requests(self):
        original = image(0)
        request = ReadRequest('0', (tuple(range(64)),)*3)
        for method in ('ic_matched_shared', 'ic_matched_batched', 'fs_matched_b8'):
            with self.subTest(method=method):
                root = self.root/method
                build_store([original], root, method)
                with StoreReader(root, cache_bytes=1024**2) as reader:
                    views, _ = reader.read(reader.plan([request, request]))
                    dense = views[0].dense()
                    for value, backing, expected in zip(dense, views[0].coefficients, original.coefficients):
                        self.assertTrue(np.shares_memory(value, backing))
                        np.testing.assert_array_equal(value, expected)
                    dense[0].fill(0)
                    np.testing.assert_array_equal(views[1].dense()[0], original.coefficients[0])
                    again, counters = reader.read(reader.plan([request]))
                    self.assertEqual(counters.decoded_q_values, 0)
                    np.testing.assert_array_equal(again[0].dense()[0], original.coefficients[0])

    def test_reordered_full_and_partial_frequencies_keep_crop_and_whole_member_reads(self):
        originals = [image(0), image(1)]
        selections = (tuple(reversed(range(64))), (63, 0, 8, 1, 27), (1, 8, 15, 22), (9,))
        regions = ((1, 1, 2, 2), (0, 1, 2, 1), (1, 0, 1, 2))
        for method in ('ic_matched_batched', 'fs_matched_b8'):
            root = self.root/method
            build_store(originals, root, method, group_size=2)
            with StoreReader(root) as reader:
                for frequencies in selections:
                    with self.subTest(method=method, frequencies=frequencies):
                        request = ReadRequest('0', (frequencies,)*3, regions)
                        plan = reader.plan([request, request])
                        views, counters = reader.read(plan)
                        expected_decoded = sum(reader.units[name]['values'] for name in plan.units)
                        self.assertEqual(counters.decoded_q_values, expected_decoded)
                        for c, (y, x, h, w) in enumerate(regions):
                            cropped = originals[0].coefficients[c][y:y+h, x:x+w]
                            expected = cropped.reshape(h, w, 64)[..., list(frequencies)]
                            np.testing.assert_array_equal(views[0].coefficients[c], expected)
                            if len(frequencies) == 64:
                                np.testing.assert_array_equal(views[0].dense()[c], cropped)
                        if len(frequencies) != 64:
                            with self.assertRaises(ValueError):
                                views[0].dense()

    def test_fixedk_reordered_storage_mask_keeps_requested_values(self):
        original = image(0, ((2, 3),))
        root = self.root/'fixed'
        build_store([original], root, 'ic_matched_fixedk', source_frequencies=[[63, 0, 8, 1]],
                    dependency_evidence='fixture frequency-only request')
        with StoreReader(root) as reader:
            for frequencies in ((63, 0, 8, 1), (1, 8, 0), (63,)):
                views, _ = reader.read(reader.plan([ReadRequest('0', (frequencies,))]))
                np.testing.assert_array_equal(views[0].coefficients[0],
                                              original.coefficients[0].reshape(2, 3, 64)[..., list(frequencies)])


class GroupAlignedReadTests(TemporaryTest):
    def test_aligned_pools_decode_each_group_once_without_dropping_the_tail(self):
        _, config_path, source, _ = test_round4.CheckpointInferenceTests.fixture(
            self, cpu_loader='dataloader', request_order='source', batch_size=1,
            pool_size=1, decoded_cache_mib=0, prefetch_factor=2)
        config, _, profile, records, trace = inputs(config_path)
        build_store(source_images(source), self.root/'ic', 'ic_matched_batched', group_size=2,
                    source_artifact=config['source_manifest'])
        values_per_image = sum(math.prod(c['block_grid'])*64 for c in source['images'][0]['components'])
        for store in ('ic', 'fs'):
            for pool in (1, 2):
                with self.subTest(store=store, pool=pool):
                    loader = inference_loader(self.root/store, profile, dict(config, preparation_pool_size=pool),
                                              records, trace, reader_only=True)
                    try:
                        batches = list(loader)
                    finally:
                        del loader
                    self.assertEqual(sum(count for count, _ in batches), 3)
                    self.assertEqual([count for count, _ in batches], [1, 1, 1] if pool == 1 else [2, 1])
                    self.assertEqual(sum(c.decoded_q_values for _, c in batches),
                                     values_per_image*(5 if pool == 1 else 3))
                    self.assertEqual(sum(c.file_opens for _, c in batches),
                                     (3 if pool == 1 else 2)*(1 if store == 'ic' else 16))
