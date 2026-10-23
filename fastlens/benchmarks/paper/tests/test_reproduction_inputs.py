"""Fresh-checkout reference and H2D replay regressions; no GPU required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastlens.benchmarks.paper import end_to_end, prepare_references
from fastlens.benchmarks.paper.shared.references import reference_inputs, expected_predictions
from fastlens.benchmarks.paper.section_6_3.h2d_traffic.run_section42_resources import jobs, verify
from fastlens.benchmarks.paper.section_6_3.h2d_traffic import profile_section42


class ReproductionInputsTest(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def test_resolves_explicit_reference_and_contract_relative_to_config(self):
        reference = self.write('outputs/result.json', {})
        contract = self.write('inputs/contract.json', {})
        config = self.write('references.json', {'4090': {'swinv2': {
            'fastlens': {'reference': 'outputs/result.json', 'contract': 'inputs/contract.json'}}}})
        self.assertEqual(reference_inputs(config, '4090', 'swinv2', 'imagelanes'),
                         dict(reference=reference, contract=contract))

    def test_missing_input_fails_before_gpu_selection_or_measurement(self):
        config = self.write('references.json', {'4090': {'mobilenet24': {
            'cpu_dct': {'reference': 'absent.json'}}}})
        argv = ['run', 'run', '--gpus', '4090', '--models', 'mobilenet24',
                '--phases', 'inference', '--references', str(config)]
        with patch.object(sys, 'argv', argv), patch.object(end_to_end, 'gpu_environment') as gpu:
            with self.assertRaises(FileNotFoundError):
                end_to_end.main('matched', 'test', 'unused.py')
            gpu.assert_not_called()

    def test_preparation_writes_new_references_without_using_existing_predictions(self):
        config = self.write('references.json', {'4090': {'mobilenet24': {
            'cpu_dct': {'reference': 'cpu/result.json'},
            'fastlens': {'reference': 'native/result.json'}}}})
        argv = ['prepare', '--gpus', '4090', '--references', str(config), '--dry-run']
        with patch.object(sys, 'argv', argv), patch.object(prepare_references, 'execute') as execute:
            prepare_references.main()
        self.assertEqual(execute.call_count, 2)
        for call in execute.call_args_list:
            command = list(map(str, call.args[0]))
            self.assertIn('--prepare-reference', command)
            self.assertNotIn('--reference', command)
            self.assertNotIn('--check', command)
        self.write('cpu/result.json', {})
        with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
            prepare_references.main()

    def test_chapter_forwards_inputs_and_records_replayable_commands(self):
        reference = self.write('reference.json', {})
        contract = self.write('contract.json', {})
        config = self.write('references.json', {'4090': {'mobilenet24': {
            name: {'reference': 'reference.json', 'contract': 'contract.json'}
            for name in ('cpu_dct', 'fastlens')}}})
        argv = ['run', 'run', '--gpus', '4090', '--models', 'mobilenet24',
                '--phases', 'inference', '--repeats', '1', '--references', str(config),
                '--output', str(self.root / 'runs')]
        with patch.object(sys, 'argv', argv), patch.object(end_to_end, 'gpu_environment', return_value={}), \
                patch.object(end_to_end, 'execute') as execute:
            end_to_end.main('matched', 'test', 'unused.py')
        self.assertEqual(execute.call_count, 4)
        for call in execute.call_args_list:
            command = list(map(str, call.args[0]))
            self.assertEqual(command[command.index('--reference') + 1], str(reference))
            self.assertEqual(command[command.index('--contract') + 1], str(contract))
            self.assertNotIn('--prepare-reference', command)
            output = Path(command[command.index('--output') + 1])
            metadata = json.loads((output / 'run.command.json').read_text())
            self.assertEqual(metadata['command'], command)

    def test_swin_native_predictions_follow_recorded_physical_order(self):
        samples = [dict(sample_id=f'image-{i}', galp_image_id=i, label=i % 1000) for i in range(50000)]
        manifest = self.write('samples.json', dict(samples=samples))
        contract = self.write('contract.json', dict(dataset=dict(sample_manifest=str(manifest))))
        ids = list(reversed(range(50000)))
        reference = self.write('reference.json', dict(image_ids=ids, predictions=[[i % 1000] for i in ids]))
        predictions = expected_predictions(reference, contract)
        self.assertEqual(len(predictions), 50000)
        self.assertEqual(predictions['image-1234'], 234)
        reference.write_text(json.dumps(dict(image_ids=ids[:-1], predictions=[[i % 1000] for i in ids[:-1]])))
        with self.assertRaises(AssertionError):
            expected_predictions(reference, contract)

    def test_training_forwards_prepared_l3_and_jpeg_inputs(self):
        argv = ['run', 'run', '--gpus', '4090', '--models', 'vitti',
                '--phases', 'training', '--repeats', '1',
                '--l3-root', str(self.root / 'l3'), '--l3-library', str(self.root / 'l3.so'),
                '--ffcv-jpeg-root', str(self.root / 'jpeg'), '--output', str(self.root / 'runs')]
        with patch.object(sys, 'argv', argv), patch.object(end_to_end, 'gpu_environment', return_value={}), \
                patch.object(end_to_end, 'execute') as execute:
            end_to_end.main('deployment', 'test', 'unused.py')
        for call in execute.call_args_list:
            command = list(map(str, call.args[0]))
            pipeline = command[command.index('--path') + 1]
            if pipeline == 'l3':
                self.assertEqual(command[command.index('--l3-root') + 1], str(self.root / 'l3'))
                self.assertEqual(command[command.index('--l3-library') + 1], str(self.root / 'l3.so'))
            elif pipeline == 'ffcv_jpeg':
                self.assertEqual(command[command.index('--ffcv-root') + 1], str(self.root / 'jpeg'))
            if '--check' not in command:
                self.assertIn('--correctness-report', command)

    def test_h2d_uses_completed_trials_and_deduplicates_workloads(self):
        metadata = dict(phase='inference', gpu='4090', model='mobilenet24', pipeline='fastlens',
                        command=['python', 'worker.py'], environment={})
        for trial in ('check', 'trial_0', 'trial_1'):
            self.write(f'runs/{trial}/run.command.json', metadata)
            self.write(f'runs/{trial}/result.json', dict(images=50000))
        selected = list(jobs(self.root / 'runs'))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['pipeline'], 'FastLens')
        self.assertEqual(selected[0]['source'], str(self.root / 'runs/trial_0/result.json'))
        (self.root / 'runs/trial_0/result.json').unlink()
        with self.assertRaises(FileNotFoundError):
            list(jobs(self.root / 'runs'))

    def test_h2d_replay_preserves_prediction_and_checkpoint_validation(self):
        result = dict(images=50000, top1=75., predictions=[[1, 2]], sample_ids=['sample'], checkpoint='model.pth')
        source = self.write('result.json', result)
        job = dict(phase='inference', source=str(source))
        self.assertEqual(verify(job, dict(result)), 50000)
        with self.assertRaises(AssertionError):
            verify(job, dict(result, predictions=[[2, 1]]))
        with self.assertRaises(AssertionError):
            verify(job, dict(result, checkpoint='different.pth'))

    def test_h2d_replay_rejects_changed_training_order(self):
        result = dict(images=65536, updates=64, correctness_passed=True,
                      initial_model_hash='model', sample_order_sha256='samples',
                      precision='fp32', batch=64, accumulation=16, warmup_updates=4)
        source = self.write('training.json', result)
        job = dict(phase='training', source=str(source))
        self.assertEqual(verify(job, dict(result)), 65536)
        with self.assertRaises(AssertionError):
            verify(job, dict(result, sample_order_sha256='different'))

    def test_profiling_instruments_each_worker_at_its_existing_timer(self):
        for relative, (start, end) in profile_section42.BOUNDARIES.items():
            with self.subTest(worker=relative):
                path = profile_section42.ROOT / relative
                source = path.read_text()
                instrumented = profile_section42.instrument(source, path)
                compile(instrumented, str(path), 'exec')
                lines = instrumented.splitlines()
                self.assertTrue(lines[lines.index(start) - 1].endswith('.begin()'))
                self.assertTrue(lines[lines.index(end) + 1].endswith('.end()'))


if __name__ == '__main__':
    unittest.main()
