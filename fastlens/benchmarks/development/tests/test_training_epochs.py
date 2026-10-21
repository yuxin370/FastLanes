"""CPU regression checks for excluded warmup and saved-epoch pairing."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from fastlens.benchmarks.development.diagnostics.check_epoch_pair import AUGMENTATION_DIGESTS
from fastlens.benchmarks.development.diagnostics.check_epoch_pair import CONTRACT_FIELDS
from fastlens.benchmarks.development.diagnostics.check_epoch_pair import NATIVE_MARKER
from fastlens.benchmarks.development.diagnostics.check_epoch_pair import replay_digests
from fastlens.benchmarks.development.diagnostics.check_epoch_pair import verify_transformer_pair
from fastlens.benchmarks.dct_models.training_pls import warmup_training_step
from fastlens.benchmarks.training_pls.published_optimizer import build_published_optimizer


class MonitorLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory(dir=Path.home()/'tmp')
        self.addCleanup(directory.cleanup)
        self.output=Path(directory.name)

    def test_monitor_failure_stops_only_owned_process_group(self):
        import signal
        from fastlens.benchmarks.execution.inference_load import run
        process=Mock(pid=12345,returncode=None)
        process.poll.return_value=None
        guard=Mock(observations=[])
        guard.blockers.side_effect=[set(),RuntimeError('monitor unavailable')]
        with patch('fastlens.benchmarks.execution.inference_load.ResourceGuard',return_value=guard), \
             patch('fastlens.benchmarks.execution.inference_load.subprocess.Popen',return_value=process), \
             patch('fastlens.benchmarks.execution.inference_load.os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'monitor unavailable'):
                run(['unused'],self.output,{'CUDA_VISIBLE_DEVICES':'GPU-test'},None)
        kill.assert_called_once_with(12345,signal.SIGTERM)
        process.wait.assert_called_once()
        self.assertNotIn('process_finished',(self.output/'concurrent_load.jsonl').read_text())

    def test_success_records_monitor_completion_without_stopping_child(self):
        from fastlens.benchmarks.execution.inference_load import run
        process=Mock(pid=12345,returncode=0)
        process.poll.side_effect=[None,0,0]
        guard=Mock(observations=[])
        guard.blockers.return_value=set()
        with patch('fastlens.benchmarks.execution.inference_load.ResourceGuard',return_value=guard), \
             patch('fastlens.benchmarks.execution.inference_load.subprocess.Popen',return_value=process), \
             patch('fastlens.benchmarks.execution.inference_load.os.killpg') as kill, \
             patch('fastlens.benchmarks.execution.inference_load.time.sleep'):
            run(['unused'],self.output,{'CUDA_VISIBLE_DEVICES':'GPU-test'},None)
        kill.assert_not_called()
        records=[json.loads(line) for line in (self.output/'concurrent_load.jsonl').read_text().splitlines()]
        self.assertEqual(records[-1]['phase'],'process_finished')
        self.assertEqual(records[-1]['returncode'],0)

    def test_gpu_binding_applies_to_cache_and_measurement(self):
        from fastlens.benchmarks.execution.inference_load import run
        guard=Mock(observations=[],numa_node=1)
        guard.blockers.return_value=set()
        process=Mock(pid=12345,returncode=0)
        process.poll.return_value=0
        with patch('fastlens.benchmarks.execution.inference_load.ResourceGuard',return_value=guard), \
             patch('fastlens.benchmarks.execution.inference_load.subprocess.check_output',
                   return_value='{"resident_fraction":1.0}') as warm, \
             patch('fastlens.benchmarks.execution.inference_load.subprocess.Popen',return_value=process) as launch:
            run(['python','measure.py'],self.output,{'CUDA_VISIBLE_DEVICES':'GPU-test'},
                [Path('/dataset/input.fls')],bind_to_gpu=True)
        prefix=['numactl','--cpunodebind=1','--membind=1']
        self.assertEqual(warm.call_args.args[0][:3],prefix)
        self.assertEqual(json.loads(warm.call_args.kwargs['input']),['/dataset/input.fls'])
        measured=launch.call_args.args[0]
        self.assertEqual(measured[measured.index('--')+1:],prefix+['python','measure.py'])
        cache=json.loads((self.output/'warm_cache.json').read_text())
        self.assertEqual(cache['warmup_numa_node'],1)


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 2))
        self.optimizer, self.decay, self.scheduler = build_published_optimizer(self.model, total_updates=20000)
        self.x = torch.randn(4, 3)
        self.target = torch.tensor([0, 1, 0, 1])

    def step(self):
        warmup_training_step(self.model, self.model, self.optimizer, self.decay, self.scheduler,
            self.x, self.target, set_to_none=True, context='test step=0')

    def test_bad_backward_fails_before_optimizer_mutation(self):
        before = copy.deepcopy(self.model.state_dict())
        self.model[0].weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        self.optimizer.step = Mock(wraps=self.optimizer.step)
        with self.assertRaisesRegex(FloatingPointError, 'test step=0: gradient 0.weight'):
            self.step()
        self.optimizer.step.assert_not_called()
        self.assertEqual(self.scheduler.completed_updates, 0)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_finite_update_uses_schedule_and_independent_decay(self):
        self.decay.step = Mock(wraps=self.decay.step)
        self.step()
        self.assertEqual(self.scheduler.completed_updates, 1)
        self.assertEqual(self.optimizer.param_groups[0]['lr'], 6e-7)
        self.decay.step.assert_called_once_with(6e-7)
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in self.model.parameters()))

    def test_input_failure_is_distinguished_from_backward(self):
        self.x[0, 0] = float('inf')
        with self.assertRaisesRegex(FloatingPointError, 'test step=0: input'):
            self.step()


class SavedPairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.home()/'tmp')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        mapping = root/'mapping.csv'
        ids = [f'sample-{i}' for i in range(128)]
        with mapping.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['logical_sample_id'])
            writer.writeheader()
            writer.writerows({'logical_sample_id': value} for value in ids)
        sha = hashlib.sha256(mapping.read_bytes()).hexdigest()
        contract = dict.fromkeys(CONTRACT_FIELDS, 'same')
        contract.update(training_seed=11997733, condition_id='B6', microbatch_size=64,
            crop_policy='per-pls', mixup={'alpha': .2}, precision='fp32', initial_model_hash='initial')
        replay, first_ids = replay_digests(SimpleNamespace(sample_count=128,
            logical_sample_ids=ids, segment_images=1024, positions_by_pls=[range(128)]), contract)
        report = dict(passed=True, samples=128, cuda_visible_devices='GPU-test',
            scope='B6 epoch index 0 first two microbatches',
            observations=[dict(sample_ids=batch, targets={'allclose': True}, tensors=[{'allclose': True}])
                          for batch in first_ids],
            update_check=dict(initial_model_hash='initial', precision='fp32', observations=[
                dict(gradients_finite=True, parameters_changed=True, optimizer_updates=1) for _ in range(2)]))
        self.report = root/'check.json'
        self.report.write_text(json.dumps(report))
        self.jobs, self.values, self.manifests = [], [], []
        for pipeline in ('imagelanes', 'cpu_dct'):
            out = root/pipeline
            directory = out/'runs/B6/seed_11997733'
            directory.mkdir(parents=True)
            manifest = directory/'run_manifest.json'
            current = dict(contract)
            if pipeline == 'imagelanes':
                current['physical_execution'] = dict(premixed_mapping_csv=str(mapping),
                    premixed_mapping_sha256=sha, segment_images=1024)
            manifest.write_text(json.dumps(current))
            self.manifests.append(manifest)
            (directory/'metrics.jsonl').write_text(json.dumps(dict(scope='epoch', epoch=1)))
            (out/'run.command.json').write_text(json.dumps(dict(environment={'CUDA_VISIBLE_DEVICES': 'GPU-test'},
                command=['--premixed-mapping-csv', str(mapping), '--expected-mapping-sha256', sha])))
            self.jobs.append(dict(output=str(out), pipeline=pipeline, correctness_report=str(self.report)))
            digests = dict(replay)
            if pipeline == 'imagelanes':
                digests.update(dict.fromkeys(AUGMENTATION_DIGESTS, NATIVE_MARKER))
            self.values.append(dict(samples=128, initial_model_hash='initial', precision='fp32', matching_digests=digests))

    def test_markers_require_logical_replay_and_independent_gate(self):
        evidence = verify_transformer_pair(self.jobs, self.values)
        self.assertTrue(evidence['passed'])
        self.assertIn('not observed hashes', evidence['native_augmentation'])
        report = json.loads(self.report.read_text())
        report['observations'][0]['tensors'][0]['allclose'] = False
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'tensor/target comparison failed'):
            verify_transformer_pair(self.jobs, self.values)

    def test_changed_configuration_blocks_pair(self):
        contract = json.loads(self.manifests[1].read_text())
        contract['mixup']['alpha'] = .4
        self.manifests[1].write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, 'contract mismatch: mixup'):
            verify_transformer_pair(self.jobs, self.values)

    def test_changed_augmentation_digest_blocks_pair(self):
        self.values[1]['matching_digests']['mixup_digest'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'replay differs'):
            verify_transformer_pair(self.jobs, self.values)


if __name__ == '__main__':
    unittest.main()
