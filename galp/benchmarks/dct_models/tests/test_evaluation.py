"""Prediction comparisons require the same evaluation samples in the same order."""
import unittest

from galp.benchmarks.dct_models.evaluate import prediction_agreement


class PredictionAgreementTest(unittest.TestCase):
    def test_matching_samples_are_compared(self):
        self.assertEqual(prediction_agreement(
            dict(sample_ids=["a", "b"], predictions=[1, 2]),
            dict(sample_ids=["a", "b"], predictions=[1, 3])), .5)

    def test_different_order_or_subset_is_rejected(self):
        result = dict(sample_ids=["a", "b"], predictions=[1, 2])
        for ids in (["b", "a"], ["a", "c"]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "sample IDs or order differ"):
                prediction_agreement(result, dict(sample_ids=ids, predictions=[1, 2]))


if __name__ == "__main__":
    unittest.main()
