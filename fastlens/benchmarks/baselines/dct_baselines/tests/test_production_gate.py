"""Production numerical fixture: complete accumulation and actual source-PLS crop keys."""
import unittest
from pathlib import Path
from unittest.mock import patch

from fastlens.benchmarks.baselines.dct_baselines.production_gate import COUNT, production_traces
from fastlens.benchmarks.baselines.dct_baselines.contracts import validate_batches
from fastlens.benchmarks.baselines.dct_baselines.training import update_windows
from fastlens.benchmarks.training_pls.published_augmentation import published_training_augmentation
from . import test_training as fixtures
from fastlens.benchmarks.baselines.dct_baselines import production_gate


class ProductionTraceTests(unittest.TestCase):
    def test_disjoint_full_pools_sixteen_microbatches_and_tail_match_physical_crop_keys(self):
        images = [dict(image_id=str(i), galp_image_id=i) for i in range(COUNT)]
        labels = {str(i): i%1000 for i in range(COUNT)}
        profile = fixtures.FrozenAugmentationTests().profile()
        traces = production_traces(images, profile, labels, 11997733)
        config = dict(model_microbatch_size=64, preparation_pool_size=4096)
        for trace in traces.values():
            validate_batches(trace, config, training=True)
            windows = update_windows(trace, 16)
            self.assertEqual([len({r['microbatch_id'] for r in w}) for w in windows], [16,16,16,16,1])
            self.assertEqual([len(w) for w in windows], [1024,1024,1024,1024,1])
            self.assertEqual(len(trace), 4097)
            self.assertTrue(any(r['flip'] for r in trace))
            self.assertTrue(any(r['crop_box'][2] != 512 for r in trace))
            for row in trace[::256]:
                expected = published_training_augmentation(training_seed=11997733, epoch=0,
                    logical_sample_id=row['image_id'], virtual_pls_id=int(row['image_id'])//1024,
                    crop_policy='per-pls', source_width=512, source_height=512).decision
                self.assertEqual(row['crop_box'], [expected.crop_x,expected.crop_y,expected.crop_width,expected.crop_height])
        self.assertFalse({r['image_id'] for r in traces['trace']} & {r['image_id'] for r in traces['calibration_trace']})


class ProductionCommandTests(unittest.TestCase):
    def test_fastlens_calibration_and_verification_receive_native_module_directory(self):
        output = Path('/fixture-output')
        with patch.dict(production_gate.os.environ), patch.object(production_gate, 'prepare', return_value=output/'contract/matched.json'), \
                patch.object(production_gate, 'command') as command:
            production_gate.run(output, model='vitti')
        calls = [call.args for call in command.call_args_list if call.args[1].startswith('fastlens_')]
        self.assertEqual([call[1] for call in calls], ['fastlens_calibrate','fastlens_verify'])
        for call in calls:
            self.assertEqual(call[call.index('--module-path')+1], production_gate.REPO/'build/fastlens/torch')
            self.assertEqual(call[call.index('--source-workers')+1], 48)

    def test_prepared_run_reuses_snapshot_and_stores_and_writes_fresh_evidence(self):
        prepared, output = Path('/prepared'), Path('/fresh-output')
        with patch.dict(production_gate.os.environ), patch.object(production_gate, 'prepare') as prepare, \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.inputs', return_value=(None,None,{'model':'vitti'},None,None)), \
                patch.object(Path, 'mkdir') as mkdir, patch.object(production_gate, 'command') as command:
            production_gate.run(output, model='vitti', prepared=prepared)
        prepare.assert_not_called()
        mkdir.assert_called_once_with(parents=True, exist_ok=False)
        calls = [call.args for call in command.call_args_list]
        self.assertEqual(len(calls), 10)
        self.assertFalse(any(call[1]=='snapshot' for call in calls))
        for call in calls:
            self.assertEqual(call[0], output)
            self.assertEqual(call[call.index('--snapshot')+1], prepared/'initial.pt')
        fs = next(call for call in calls if call[1]=='fs_calibrate')
        self.assertEqual(fs[fs.index('--root')+1], prepared/'fs_matched_b8')
