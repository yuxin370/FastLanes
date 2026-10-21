import unittest

from fastlens.benchmarks.development.historical.replan import strict_prediction_pass
from fastlens.benchmarks.development.historical.replan import summaries


class ReplanTest(unittest.TestCase):
    def test_failed_observations_never_enter_summary(self):
        base = dict(gpu='H100', model='Swin', domain='DCT', pipeline='FastLens',
                    task='inference', precision='fp32', batch_size=50, samples=50000,
                    valid=True, elapsed_s=25, raw_result='raw.json', notes='')
        result, = summaries([base, dict(base, valid=False, elapsed_s=1)])
        self.assertEqual(result['trials'], 1)
        self.assertEqual(result['median_images_per_s'], 2000)
        self.assertEqual(result['raw_results'], 'raw.json')

    def test_rounding_disagreement_not_promoted_to_strict_equality(self):
        gate = dict(sample_order_equal=True, prediction_agreement=.99998, numerical_control_pass=True)
        self.assertFalse(strict_prediction_pass(gate))
        gate['prediction_agreement'] = 1.0
        self.assertTrue(strict_prediction_pass(gate))
        gate['sample_order_equal'] = False
        self.assertFalse(strict_prediction_pass(gate))

    def test_transformer_requires_full_prediction_gate(self):
        comparison = dict(enforcement='strict', ok=True,
                          full_prediction=dict(count=8, top1_agreement=1.0))
        gate = dict(semantic_validation=dict(comparisons=[comparison]))
        self.assertFalse(strict_prediction_pass(gate))
        comparison['full_prediction']['count'] = 50000
        self.assertTrue(strict_prediction_pass(gate))

    def test_vit_legacy_tolerance_does_not_relax_new_gate(self):
        comparison = dict(enforcement='strict', ok=True, sample_identity_match=True,
                          full_top1_prediction_agreement=.9999)
        gate = dict(semantic_comparisons=[comparison])
        self.assertFalse(strict_prediction_pass(gate))
        comparison['full_top1_prediction_agreement'] = 1.0
        self.assertTrue(strict_prediction_pass(gate))


if __name__ == '__main__':
    unittest.main()
