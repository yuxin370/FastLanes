"""Small CPU crop/sequence tests; no ImageNet scan, native load, or GPU work."""
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from fastlens.benchmarks.baselines.system_rgbnomore.training.augmentation import derive_augmentation
from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingSample
from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import PlannedRgbDataset, check_rgb_input
from fastlens.benchmarks.baselines.system_rgbnomore.training.sample_order import SampleIdentity


class FullRgbEpochTest(unittest.TestCase):
    def test_partial_update_has_same_gradient_as_unpartitioned_batch(self):
        from fastlens.benchmarks.paper.shared.training_window_rgb import update_microbatch_sizes
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 2)
        x, y = torch.randn(143, 3), torch.randn(143, 2)
        torch.nn.functional.mse_loss(model(x), y).backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        offset = 0
        for n in update_microbatch_sizes(len(x)):
            loss = torch.nn.functional.mse_loss(model(x[offset:offset+n]), y[offset:offset+n])
            (loss * (n / len(x))).backward()
            offset += n
        self.assertEqual(offset, 143)
        for parameter, reference in zip(model.parameters(), expected):
            torch.testing.assert_close(parameter.grad, reference)
        batches = [update_microbatch_sizes(min(1024, 1281167-i))
                   for i in range(0, 1281167, 1024)]
        self.assertEqual(len(batches), 1252)
        self.assertEqual(sum(map(sum, batches)), 1281167)
        self.assertEqual(batches[-1], [64, 64, 15])



class TrainingResourceAndCacheTest(unittest.TestCase):
    def test_interrupted_logger_reaps_measurement_before_returning(self):
        import os
        import signal
        import subprocess
        import sys
        import time
        import psutil
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            output = Path(temporary)
            ready = output/'ready'
            child = (
                'import os,signal,time,sys; from pathlib import Path; '
                'signal.signal(signal.SIGTERM, lambda *_: (time.sleep(.3), sys.exit(0))); '
                f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)'
            )
            env = dict(os.environ)
            env.pop('FASTLENS_EVALUATION_SNAPSHOT', None)
            with subprocess.Popen([sys.executable, 'fastlens/benchmarks/execution/run_logged.py', '--log',
                    str(output/'run.log'), '--', sys.executable, '-c', child],
                    env=env, start_new_session=True, stdout=subprocess.DEVNULL) as wrapper:
                try:
                    deadline = time.monotonic()+10
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(ready.exists())
                    child_pid = int(ready.read_text())
                    os.killpg(wrapper.pid, signal.SIGTERM)
                    wrapper.wait(timeout=10)
                    self.assertFalse(psutil.pid_exists(child_pid))
                    self.assertTrue((output/'run.command.json').exists())
                finally:
                    if wrapper.poll() is None:
                        os.killpg(wrapper.pid, signal.SIGKILL)
                        wrapper.wait()

    def test_new_owned_child_is_not_external_contention(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from fastlens.benchmarks.execution.resource_guard import ResourceGuard
        child = MagicMock()
        child.parents.return_value = [SimpleNamespace(pid=100)]
        child.cmdline.return_value = ['python', 'training_window_rgb.py']
        child.create_time.return_value = 123.0
        with patch.dict('os.environ', {'FASTLENS_ALLOWED_RESIDENT_PIDS': ''}), \
             patch('fastlens.benchmarks.execution.resource_guard.gpu_numa_nodes', return_value={'GPU-test':0}), \
             patch('fastlens.benchmarks.execution.resource_guard.subprocess.check_output', return_value='200, GPU-test\n'), \
             patch('fastlens.benchmarks.execution.resource_guard.psutil.Process', return_value=child), \
             patch.object(ResourceGuard, 'host_blockers', return_value=set()):
            guard = ResourceGuard('GPU-test')
            self.assertEqual(guard.blockers({100}), set())
            self.assertTrue(guard.observations[0]['owned_descendant'])
            child.parents.return_value = [SimpleNamespace(pid=999)]
            self.assertEqual(guard.blockers({100}), {200})
            self.assertEqual(guard.observations[0]['parent_pids'], [999])

    def test_cleanup_only_removes_private_cache_and_preserves_source(self):
        import os
        from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as temporary:
            output = Path(temporary)
            source = output/'jpeg-source'
            source.mkdir()
            (source/'sample.JPEG').write_bytes(b'unchanged JPEG source')
            namespace = Path(tempfile.mkdtemp(prefix='fastlens-coordl-', dir=Path.home()/'tmp'))
            (namespace/'source').symlink_to(source, target_is_directory=True)
            cached = Path('/dev/shm/cache')/namespace.relative_to('/')
            cached.mkdir(parents=True)
            (cached/'partial-tmp').write_bytes(b'partial cache')
            (output/'coordl_cache_namespace.json').write_text(json.dumps(dict(namespace=str(namespace), owner_pid=os.getpid())))
            cleanup_coordl_cache(output)
            cleanup_coordl_cache(output)
            self.assertFalse(namespace.exists())
            self.assertFalse(cached.exists())
            self.assertEqual((source/'sample.JPEG').read_bytes(), b'unchanged JPEG source')
            (output/'coordl_cache_namespace.json').write_text(json.dumps(dict(namespace=str(source), owner_pid=os.getpid())))
            with self.assertRaisesRegex(ValueError, 'Not an owned'):
                cleanup_coordl_cache(output)
            self.assertTrue(source.exists())


class TrainingInputGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.home()/'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.samples = []
        for i in range(4):
            path = self.root/f'{i}.png'
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
            self.samples.append(TrainingSample(str(i), path, i, 32, 32, i))
        self.decisions = [derive_augmentation(seed=11997733, epoch=2, logical_sample_id=s.logical_sample_id,
                          source_width=32, source_height=32, domain='rgb') for s in self.samples]
        self.identities = [SampleIdentity(2, i, s.logical_sample_id) for i, s in enumerate(self.samples)]
        self.labels = torch.arange(4)
        self.native_pixels = [np.full((32, 32, 3), 64, dtype=np.uint8) for _ in self.samples]
        self.images = torch.full((4, 3, 224, 224), 64/127.5-1)

    def check(self, images=None, labels=None, identities=None, pipeline='dali'):
        return check_rgb_input(self.images if images is None else images,
            self.labels if labels is None else labels, self.identities if identities is None else identities,
            self.samples, self.decisions, pipeline=pipeline, device=torch.device('cpu'),
            dali_config={}, output=self.root/'check')

    def test_native_reference_accepts_decoder_difference_and_records_pil_error(self):
        with patch('fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters.decode_native_rgb_reference', return_value=self.native_pixels) as decode:
            report = self.check()
        self.assertEqual(decode.call_args.args[0], self.samples)
        self.assertTrue(report['order_matches'] and report['labels_match'])
        self.assertTrue(all(r['max_error'] < 1e-7 and r['pil_max_error'] > 8/255 for r in report['samples']))
        self.assertEqual(json.loads((self.root/'check/rgb_input_check.json').read_text()), report)

    def test_native_reference_still_rejects_input_label_and_order_errors(self):
        with patch('fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters.decode_native_rgb_reference', return_value=self.native_pixels):
            # Below the maximum-error limit, but above the unchanged mean-error limit.
            with self.assertRaisesRegex(ValueError, 'normalization gate failed'):
                self.check(images=self.images+0.01)
            with self.assertRaisesRegex(ValueError, 'normalization gate failed'):
                self.check(images=torch.full_like(self.images, float('nan')))
            with self.assertRaisesRegex(ValueError, 'label mismatch'):
                self.check(labels=self.labels.flip(0))
            with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
                self.check(identities=list(reversed(self.identities)))

    def test_pytorch_keeps_pil_reference(self):
        with patch('fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters.decode_native_rgb_reference') as decode:
            report = self.check(images=torch.full_like(self.images, -1), pipeline='pytorch')
        decode.assert_not_called()
        self.assertTrue(all(r['max_error'] == 0 for r in report['samples']))

    def test_old_report_cannot_authorize_native_decoder_performance_run(self):
        from fastlens.benchmarks.paper.shared.training_window_rgb import main
        path = self.root/'old-result.json'
        path.write_text(json.dumps(dict(correctness_passed=True, mode='correctness', model='mobilenet24',
                                        arm='dali', gpu='NVIDIA GeForce RTX 4090')))
        argv = ['training_window_rgb.py', '--gpu', '4090', '--model', 'mobilenet24', '--pipeline', 'dali',
                '--trial', '0', '--output', str(self.root/'performance'), '--correctness-report', str(path)]
        with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'standard native RGB decoder'):
            main()


class TrainingCropTest(unittest.TestCase):
    def test_ffcv_jpeg_matches_keyed_pytorch_crops_and_labels(self):
        self.check_ffcv_crops('jpg')

    def test_ffcv_matches_keyed_pytorch_crops_and_labels(self):
        self.check_ffcv_crops('raw')

    def check_ffcv_crops(self, mode):
        import warnings
        from numba.core.errors import NumbaPerformanceWarning
        from ffcv.writer import DatasetWriter
        from ffcv.fields import RGBImageField, IntField
        from ffcv.loader import Loader, OrderOption
        from ffcv.fields.decoders import IntDecoder
        from fastlens.benchmarks.baselines.system_rgbnomore.training.ffcv_crop import PlannedCropDecoder
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            pixels=np.random.default_rng(7).integers(0,256,(4,512,512,3),dtype=np.uint8)
            samples=[]
            for i,array in enumerate(pixels):
                path=root/f'{i}.{"jpg" if mode == "jpg" else "png"}';Image.fromarray(array).save(path)
                samples.append(TrainingSample(str(i),path,i,512,512,i))
            decisions=[derive_augmentation(seed=11997733,epoch=2,logical_sample_id=s.logical_sample_id,
                source_width=512,source_height=512,domain='rgb') for s in samples]
            expected=PlannedRgbDataset(samples,decisions)
            if mode == 'jpg':
                from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import decode_native_rgb_reference
                decoded = decode_native_rgb_reference(samples, pipeline='ffcv-jpeg', device=None, dali_config={})
                reference_samples = []
                for i, array in enumerate(decoded):
                    path = root/f'decoded-{i}.png'
                    Image.fromarray(array).save(path)
                    reference_samples.append(TrainingSample(str(i),path,i,512,512,i))
                expected = PlannedRgbDataset(reference_samples, decisions)
            class Raw:
                def __len__(self):return 4
                def __getitem__(self,i):return (str(samples[i].path) if mode == 'jpg' else pixels[i]),i
            from fastlens.benchmarks.baselines.ffcv.prepare_compressed_ffcv import original_jpeg_field
            field = original_jpeg_field() if mode == 'jpg' else RGBImageField(write_mode='raw')
            DatasetWriter(str(root/'test.beton'),{'image':field,'label':IntField()},
                          num_workers=1).from_indexed_dataset(Raw())
            outputs=[]
            with warnings.catch_warnings():
                warnings.simplefilter('error', NumbaPerformanceWarning)
                for workers in (1, 2):
                    loader=Loader(str(root/'test.beton'),batch_size=3,num_workers=workers,order=OrderOption.SEQUENTIAL,
                        drop_last=False,pipelines={'image':[PlannedCropDecoder(decisions)],'label':[IntDecoder()]})
                    outputs.append([(x.copy(),y.copy()) for x,y in loader])
            actual=outputs[1]
            for serial,parallel in zip(outputs[0],actual):
                np.testing.assert_array_equal(serial[0],parallel[0])
                np.testing.assert_array_equal(serial[1],parallel[1])
            self.assertEqual([len(x) for x,_ in actual],[3,1])
            self.assertEqual(np.concatenate([y.reshape(-1) for _,y in actual]).tolist(),[0,1,2,3])
            for i,value in enumerate(np.concatenate([x for x,_ in actual])):
                tensor=torch.from_numpy(value.copy()).permute(2,0,1).float().div_(127.5).sub_(1)
                # Bilinear implementations may round uint8 at different stages: one LSB.
                torch.testing.assert_close(tensor,expected[i][0],rtol=0,atol=2/255+1e-7)


if __name__=='__main__':unittest.main()
