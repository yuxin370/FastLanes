"""Reusable inference references, independent of the measured implementation."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from fastlens.benchmarks.baselines.dct_baselines.__main__ import prepare
from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact
from fastlens.benchmarks.baselines.dct_baselines.runner import benchmark, inputs
from fastlens.benchmarks.baselines.dct_baselines.unified import UnifiedReplay
from fastlens.benchmarks.baselines.dct_baselines.verification import batch_signature, reference_identities, verify_inference
from .test_baselines import TemporaryTest
from . import test_round4


class ReferenceIdentityTests(unittest.TestCase):
    def test_share_preprocessing_but_bind_model_source_labels_order_and_normalization(self):
        config = dict(source_manifest={'sha256': 'source'}, model_microbatch_size=2,
                      precision='fp32', torch_threads=1)
        profile = dict(id='a', model='a', checkpoint={'sha256': 'weights-a'}, mean=[1.], std=[2.])
        trace = [dict(profile_id='a', image_id=i, target_label=i) for i in (0, 1)]
        def keys(c=config, p=profile, t=trace):
            return reference_identities(c, p, t, {'device': 'cpu'}, [], [])
        original = keys()
        other = dict(profile, id='b', model='b', checkpoint={'sha256': 'weights-b'})
        different_model = keys(p=other, t=[dict(r, profile_id='b') for r in trace])
        self.assertEqual(original[0], different_model[0])
        self.assertNotEqual(original[1], different_model[1])
        for c, p, t in ((dict(config, source_manifest={'sha256': 'other'}), profile, trace),
                        (config, dict(profile, std=[3.]), trace),
                        (config, profile, list(reversed(trace))),
                        (config, profile, [dict(trace[0], target_label=2), trace[1]])):
            self.assertNotEqual(original[0], keys(c, p, t)[0])

    def test_tensor_signature_includes_shapes_values_and_occurrence_labels(self):
        rows = [dict(occurrence_id='0', image_id='x', target_label=1)]
        tensor = torch.arange(6.).reshape(1, 2, 3)
        signature = batch_signature(rows, (tensor,))
        self.assertEqual(signature, batch_signature(rows, (tensor.clone(),)))
        self.assertNotEqual(signature, batch_signature(rows, (tensor.reshape(1, 3, 2),)))
        self.assertNotEqual(signature, batch_signature(rows, (tensor+1,)))
        self.assertNotEqual(signature, batch_signature([dict(rows[0], target_label=2)], (tensor,)))


class VerificationReuseTests(TemporaryTest):
    fixture = test_round4.CheckpointInferenceTests.fixture
    load_model = staticmethod(test_round4.CheckpointInferenceTests.load_model)

    def check(self, config, backend, output, **options):
        return verify_inference(config, backend, self.root/'fs' if backend == 'fs' else None,
                                self.root/output, cache_dir=self.root/'references', **options)

    def test_reuses_source_inputs_model_and_accepts_evidence_for_timing(self):
        _, config, _, _ = self.fixture()
        with patch('fastlens.benchmarks.baselines.dct_baselines.verification.build_model', side_effect=self.load_model), patch('builtins.print'):
            reference = self.check(config, 'cpu_dct', 'cpu.json')
            self.assertEqual(reference['reuse']['independent_reference_batches'], 2)
            self.assertEqual(reference['reuse']['model_forward_calls'], 2)
            with patch('fastlens.benchmarks.baselines.dct_baselines.verification.source_reference', side_effect=AssertionError('oracle repeated')), \
                    patch('fastlens.benchmarks.baselines.dct_baselines.verification.build_model', side_effect=AssertionError('model repeated')):
                store = self.check(config, 'fs', 'fs.json')
                self.assertEqual(store['reuse'], dict(input_batches=2, model_batches=2,
                                                     independent_reference_batches=0, model_forward_calls=0))
                self.assertEqual(store['metrics'], reference['metrics'])
                with patch.object(UnifiedReplay, 'verify_source', side_effect=AssertionError('q/Q repeated')):
                    again = self.check(config, 'fs', 'fs-again.json')
                self.assertIsNotNone(again['source_evidence_reused'])
            with patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=self.load_model):
                row, _ = benchmark(config, self.root/'fs', self.root/'fs.json', backend='fs', run_kind='inference')
                self.assertEqual(row['quality']['sample_count'], 3)

    def test_new_checkpoint_reuses_inputs_and_source_but_computes_new_model_results(self):
        args, config, _, _ = self.fixture()
        with patch('fastlens.benchmarks.baselines.dct_baselines.verification.build_model', side_effect=self.load_model), patch('builtins.print'):
            first = self.check(config, 'cpu_dct', 'first.json')
            frozen, _, profile, _, _ = inputs(config)
            replacement = self.load_model(profile, 'cpu')
            with torch.no_grad():
                replacement.head.bias.copy_(torch.arange(4, -1, -1).float())
            torch.save(replacement.state_dict(), self.root/'replacement.pt')
            profile = dict(profile, checkpoint=artifact(self.root/'replacement.pt'))
            args = copy.copy(args)
            args.output = self.root/'second-contract'
            args.source_manifest = Path(frozen['source_manifest']['path'])
            args.source_recipe = None
            with patch('fastlens.benchmarks.baselines.dct_baselines.__main__.resolve_profile', return_value=profile):
                prepare(args)
            with patch('fastlens.benchmarks.baselines.dct_baselines.verification.source_reference', side_effect=AssertionError('oracle repeated')), \
                    patch.object(UnifiedReplay, 'verify_source', side_effect=AssertionError('q/Q repeated')):
                second = self.check(args.output/'matched.json', 'cpu_dct', 'second.json')
            self.assertEqual(first['reference_cache']['inputs'], second['reference_cache']['inputs'])
            self.assertNotEqual(first['reference_cache']['model'], second['reference_cache']['model'])
            self.assertEqual(second['reuse']['independent_reference_batches'], 0)
            self.assertEqual(second['reuse']['model_forward_calls'], 2)
            self.assertNotEqual(first['metrics']['top1_fraction'], second['metrics']['top1_fraction'])

    def test_existing_gate_supplies_model_results_without_new_forward_calls(self):
        from fastlens.benchmarks.baselines.dct_baselines.gates import verify_inference as original_verify
        _, config, _, _ = self.fixture()
        prior = self.root/'existing.json'
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model), patch('builtins.print'):
            existing = original_verify(config, 'cpu_dct', None, prior)
        with patch('fastlens.benchmarks.baselines.dct_baselines.verification.build_model', side_effect=AssertionError('model repeated')), \
                patch('builtins.print'):
            result = self.check(config, 'fs', 'from-existing.json', model_evidence=prior)
        self.assertEqual(result['reuse']['model_forward_calls'], 0)
        self.assertEqual(result['reuse']['model_batches'], 2)
        self.assertEqual(result['metrics'], existing['metrics'])
        self.assertEqual(result['model_evidence_reused'], artifact(prior))

    def test_corrupt_cache_and_wrong_actual_inputs_cannot_receive_a_pass_marker(self):
        _, config, _, _ = self.fixture()
        with patch('fastlens.benchmarks.baselines.dct_baselines.verification.build_model', side_effect=self.load_model), patch('builtins.print'):
            reference = self.check(config, 'cpu_dct', 'reference.json')
            path = Path(reference['reference_cache']['inputs']['path'])
            text = path.read_text()
            data = json.loads(text)
            data['batches'][0]['tensors'][0]['sha256'] = '0'*64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'cache is incomplete, changed'):
                self.check(config, 'fs', 'corrupt-cache.json')
            path.write_text(text)
            original = UnifiedReplay.inference_batches
            def wrong(replay, trace):
                for rows, tensors in original(replay, trace):
                    yield rows, (tensors[0]+1, *tensors[1:])
            with patch.object(UnifiedReplay, 'inference_batches', wrong), self.assertRaises(AssertionError):
                self.check(config, 'fs', 'wrong-input.json')
        self.assertFalse((self.root/'corrupt-cache.json').exists())
        self.assertFalse((self.root/'wrong-input.json').exists())

    def test_differing_batch_uses_original_tolerance_and_direct_model_comparison(self):
        _, config, _, _ = self.fixture(atol=1e-4)
        with patch('fastlens.benchmarks.baselines.dct_baselines.verification.build_model', side_effect=self.load_model), patch('builtins.print'):
            self.check(config, 'cpu_dct', 'reference.json')
            original = UnifiedReplay.inference_batches
            def perturb(replay, trace):
                for rows, tensors in original(replay, trace):
                    yield rows, (tensors[0]+1e-6, *tensors[1:])
            with patch.object(UnifiedReplay, 'inference_batches', perturb):
                result = self.check(config, 'fs', 'close-input.json')
            self.assertEqual(result['reuse']['model_batches'], 0)
            self.assertEqual(result['reuse']['model_forward_calls'], 4)
            self.assertEqual(result['tolerance']['atol'], 1e-4)
            self.assertGreater(result['max_abs_tensor_error'], 0)
