"""The reproduction entry checks the actual streaming reader, including its tail."""
import unittest
from unittest.mock import patch

from fastlens.benchmarks.baselines.dct_baselines.compact_benchmark import check_inputs
from .test_streamed_b8_compact import dataset


class CompactBenchmarkTests(unittest.TestCase):
    @patch('torch.cuda.current_stream', return_value='consumer')
    def test_streamed_reader_checks_all_samples_and_remains_reusable(self, _):
        actual, reference = dataset(), dataset(streamed=False)
        for _ in range(2):
            result = check_inputs(actual, reference, batch_size=2, pool_size=4)
            self.assertEqual(result['samples'], 9)
            self.assertFalse(actual.producer.pipeline.closed)

    @patch('torch.cuda.current_stream', return_value='consumer')
    def test_label_mismatch_fails_and_releases_the_stream(self, _):
        actual, reference = dataset(), dataset(streamed=False)
        actual.labels[0] = 999
        with self.assertRaises(AssertionError):
            check_inputs(actual, reference, batch_size=2, pool_size=4)
        self.assertTrue(actual.producer.pipeline.closed)
