import numpy as np

from fastlens.benchmarks.baselines.dct_baselines.contracts import ReadRequest
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store
from .test_baselines import TemporaryTest, image


class StaticGroupTests(TemporaryTest):
    def test_fixed_frequency_group_size_controls_storage_without_changing_selected_values(self):
        sources = [image(i) for i in range(5)]
        mask = (0, 1, 2, 8, 9, 16)
        indices = (4, 0, 2, 0, 1, 3)
        for size, members in ((1, [1, 1, 1, 1, 1]), (4, [4, 1])):
            with self.subTest(group_size=size):
                root = self.root/f'group{size}'
                manifest = build_store(sources, root, 'ic_matched_fixedk', group_size=size,
                                       source_frequencies=[mask]*3, dependency_evidence='D3 source-q prefix')
                self.assertEqual(manifest['storage_group_images'], size)
                self.assertEqual([len(group['images']) for group in manifest['groups']], members)
                with StoreReader(root) as reader:
                    requests = [ReadRequest(str(i), (mask,)*3) for i in indices]
                    views, counters = reader.read(reader.plan(requests))
                    self.assertEqual(counters.file_opens, len(members))
                    for index, view in zip(indices, views):
                        for actual, original, table, expected_table in zip(view.coefficients,
                                sources[index].coefficients, view.tables, sources[index].tables):
                            np.testing.assert_array_equal(actual, original.reshape(*original.shape[:2], 64)[..., mask])
                            np.testing.assert_array_equal(table, expected_table)
