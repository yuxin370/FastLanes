import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastlens.benchmarks.development.historical.run_p1_swin import validate_epoch


class FullEpochGateTest(unittest.TestCase):
    def setUp(self):
        self.epoch = dict(scope='epoch', epoch=1, epoch_samples=1281167,
                          epoch_optimizer_updates=1252, train_loss=6.5)
        self.result = dict(precision='bf16-autocast', integration_checks=dict(
            coverage_counters_valid=True, gradients_finite=True, loss_finite=True, parameters_finite=True))
        self.warm = dict(initial_state_restored=True, initial_model_hash='initial')
        self.contract = dict(initial_model_hash='initial', recipe_hash='recipe')
        from fastlens.benchmarks.development.historical.run_p1_swin import ROOT
        self.output = ROOT/'results/p1/raw/H100/imagelanes'

    def validate(self):
        with patch('fastlens.benchmarks.development.historical.run_p1_swin.read', side_effect=[self.result, self.warm, self.contract]), \
             patch.object(Path, 'read_text', return_value=json.dumps(self.epoch)):
            return validate_epoch(self.output)

    def test_complete_first_epoch_is_accepted(self):
        accepted = self.validate()
        self.assertTrue(accepted['correctness_passed'])
        self.assertEqual(accepted['epoch']['epoch_samples'], 1281167)

    def test_incomplete_coverage_or_updates_rejected(self):
        for field, wrong in [('epoch_samples', 65536), ('epoch_optimizer_updates', 1251),
                             ('epoch', 2), ('train_loss', float('nan'))]:
            with self.subTest(field=field):
                original = self.epoch[field]
                self.epoch[field] = wrong
                with self.assertRaises(ValueError):
                    self.validate()
                self.epoch[field] = original

    def test_warmup_must_restore_initialization(self):
        self.warm['initial_model_hash'] = 'changed'
        with self.assertRaises(ValueError):
            self.validate()

    def test_failed_finite_gate_rejected(self):
        self.result['integration_checks']['gradients_finite'] = False
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
