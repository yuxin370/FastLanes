"""Round-4 current-source binding and pretrained inference semantics."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from fastlens.benchmarks.baselines.dct_baselines.__main__ import prepare
from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact, write_json
from fastlens.benchmarks.baselines.dct_baselines.gates import verify_inference
from fastlens.benchmarks.baselines.dct_baselines.materialize import resolve_profile
from fastlens.benchmarks.baselines.dct_baselines.runner import SCHEMA, Replay, benchmark, evict_input_files, inputs, report
from fastlens.benchmarks.baselines.dct_baselines.source import source_images, verify_source_files, freeze_source, validate_population
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store, training_store_identity
from fastlens.benchmarks.baselines.dct_baselines.unified import UnifiedReplay
from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
from .test_baselines import TemporaryTest


class TinyCheckpoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 5)

    def forward(self, y, cbcr):
        return self.head(torch.stack((y.flatten(1).mean(1), cbcr.flatten(1).mean(1)), 1))


class PopulationTests(unittest.TestCase):
    def test_unique_coverage_class_count_and_tail_population(self):
        source = dict(images=[dict(image_id=name) for name in ('val/a/1', 'val/a/2', 'val/b/3')])
        trace = list(reversed(source['images']))
        self.assertEqual(validate_population(source, trace, expected_images=3, expected_classes=2),
                         dict(unique_images=3, classes=2))
        for wrong in (trace[:-1], trace+trace[:1], trace[:2]+trace[:1]):
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                validate_population(source, wrong)
        with self.assertRaisesRegex(ValueError, 'classes'):
            validate_population(source, trace, expected_classes=1000)
        with self.assertRaisesRegex(ValueError, 'unique images'):
            validate_population(source, trace, expected_images=50000)


class WindowResourceTests(unittest.TestCase):
    def test_existing_and_new_workers_count_only_window_cpu_and_io(self):
        from fastlens.benchmarks.dct_models.measure_process import ProcessWindow
        state = {1: (2., 10, 100), 2: (3., 20, 80)}
        processes = {}
        for pid in (1, 2, 3):
            process = MagicMock(pid=pid)
            process.cpu_times.side_effect = lambda pid=pid: SimpleNamespace(user=state[pid][0], system=0.)
            process.memory_info.side_effect = lambda pid=pid: SimpleNamespace(rss=state[pid][2])
            process.num_threads.return_value = 2
            process.io_counters.side_effect = lambda pid=pid: SimpleNamespace(read_bytes=state[pid][1])
            processes[pid] = process
        root = processes[1]
        root.children.side_effect = lambda **_: [processes[pid] for pid in state if pid != 1]
        with patch('fastlens.benchmarks.dct_models.measure_process.psutil.Process', return_value=root):
            with ProcessWindow() as window:
                state.update({1: (5., 15, 160), 2: (6., 27, 90), 3: (4., 11, 120)})
        self.assertEqual(window.result['cpu_core_seconds'], 10.)
        self.assertEqual(window.result['physical_read_bytes'], 23)
        self.assertEqual(window.result['tree_rss_peak_bytes'], 370)


class CachePageTests(TemporaryTest):
    def test_empty_lock_does_not_abort_residency_observation(self):
        import os
        from fastlens.benchmarks.profiling.input_motivation import cache_pages
        lock, data = self.root/'schedule.bin.lock', self.root/'schedule.bin'
        lock.touch()
        size = os.sysconf('SC_PAGE_SIZE') + 1
        data.write_bytes(b'x' * size)
        observed = cache_pages([lock, data])
        self.assertEqual(observed['files'], 2)
        self.assertEqual(observed['bytes'], size)
        self.assertEqual(observed['pages'], 2)
        self.assertGreaterEqual(observed['resident_fraction'], 0.)
        self.assertLessEqual(observed['resident_fraction'], 1.)
        self.assertEqual(cache_pages([lock]), dict(files=1, bytes=0, pages=0,
                         resident_pages=0, resident_fraction=None))


class CheckpointInferenceTests(TemporaryTest):
    def fixture(self, **options):
        samples = []
        rng = np.random.default_rng(119)
        for i in range(3):
            path = self.root/f'{i}.jpg'
            Image.fromarray(rng.integers(0, 256, (512, 512, 3), np.uint8)).save(path, quality=75, subsampling=2)
            samples.append(dict(logical_sample_id=f'val/c{i}/{i}', label=i, path=str(path)))
        write_json(self.root/'selection.json', dict(split='val', samples=samples))
        write_json(self.root/'recipe.json', dict(resize='512 synthetic fixture', encoder='Pillow', quality=75, subsampling=2))
        model = TinyCheckpoint()
        with torch.no_grad():
            model.head.weight.fill_(.375)
            model.head.bias.copy_(torch.arange(5).float())
        torch.save(model.state_dict(), self.root/'weights.pt')
        profile = resolve_profile('vitti', Path.home()/'RGB-no-more', self.root/'weights.pt')
        args = SimpleNamespace(batch_size=2, pool_size=2, workers=2, torch_threads=1, atol=0., rtol=0.,
            output=self.root/'contract', source_manifest=None, selection=self.root/'selection.json',
            source_recipe=self.root/'recipe.json', count=None, training=False, model='vitti',
            upstream=Path.home()/'RGB-no-more', checkpoint=self.root/'weights.pt', seed=17,
            decode_budget_mib=16, reader_budget_mib=32, host_input_budget_mib=64)
        vars(args).update(options)
        with patch('fastlens.benchmarks.baselines.dct_baselines.__main__.resolve_profile', return_value=profile):
            prepare(args)
        config_path = args.output/'matched.json'
        config = json.loads(config_path.read_text())
        source = json.loads((args.output/'source.json').read_text())
        build_store(source_images(source), self.root/'fs', 'fs_matched_b8', group_size=2,
                    source_artifact=config['source_manifest'])
        return args, config_path, source, model

    @staticmethod
    def load_model(profile, device):
        model = TinyCheckpoint()
        model.load_state_dict(torch.load(profile['checkpoint']['path'], weights_only=True), strict=True)
        return model.to(device).eval()

    def test_dataloader_source_order_batched_decode_once_and_tail(self):
        import jsonschema
        _, config_path, source, _ = self.fixture(cpu_loader='dataloader', prefetch_factor=2,
                                                request_order='source', host_input_budget_mib=128)
        build_store(source_images(source), self.root/'ic', 'ic_matched_batched', group_size=2,
                    source_artifact=json.loads(config_path.read_text())['source_manifest'])
        _, _, _, _, trace = inputs(config_path)
        self.assertEqual([r['image_id'] for r in trace], [r['image_id'] for r in source['images']])
        rows = []
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=self.load_model), patch('builtins.print'):
            for backend, root in [('cpu_dct', None), ('ic', self.root/'ic')]:
                evidence_path = self.root/f'{backend}.json'
                verify_inference(config_path, backend, root, evidence_path)
                row, detail = benchmark(config_path, root, evidence_path, backend=backend, run_kind='inference')
                jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).validate(row)
                self.assertEqual(row['counts']['sample_occurrences'], 3)
                self.assertEqual(detail['dataset_passes'], 1)
                self.assertGreaterEqual(row['metrics']['cpu_core_seconds'], 0.)
                self.assertGreater(row['metrics']['memory']['host_rss_peak_bytes'], 0)
                if root is not None:
                    store = json.loads((root/'store.json').read_text())
                    self.assertEqual(row['metrics']['io']['application_read_bytes'],
                                     sum(f['size_bytes'] for f in store['files'].values()))
                rows.append(row)
        self.assertEqual(rows[0]['quality'], rows[1]['quality'])

    def test_warmup_and_preflight_excluded_complete_passes_and_no_timed_hashing(self):
        _, config, _, _ = self.fixture()
        evidence_path = self.root/'fs.json'
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model), patch('builtins.print'):
            verify_inference(config, 'fs', self.root/'fs', evidence_path)
        clock = [0.]
        timed_pool = [False]
        hashed_files = []
        original_pool, original_check = Replay.pool, StoreReader._check_content
        def pool(replay, rows):
            timed_pool[0] = True
            try:
                return original_pool(replay, rows)
            finally:
                timed_pool[0] = False
                clock[0] += .75
        def check(reader, name, encoded):
            self.assertFalse(timed_pool[0], 'content hashing belongs before timing')
            hashed_files.append(name)
            return original_check(reader, name, encoded)
        def evict(paths):
            clock[0] += 100.  # must not contribute to measured duration
            return dict(resident_pages=0, pages=100)
        with patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=self.load_model), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.time.perf_counter', side_effect=lambda: clock[0]), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.evict_input_files', side_effect=evict) as eviction, \
                patch.object(Replay, 'pool', pool), patch.object(StoreReader, '_check_content', check), patch('builtins.print'):
            row, detail = benchmark(config, self.root/'fs', evidence_path, backend='fs', run_kind='inference',
                                    warmup_seconds=2., min_seconds=2., cold_cache=True)
        self.assertEqual(detail['warmup'], dict(dataset_passes=2, elapsed_wall_s=3.))
        self.assertEqual(detail['dataset_passes'], 2)
        self.assertEqual(row['counts']['sample_occurrences'], 6)
        self.assertEqual(row['quality']['sample_count'], 3)
        self.assertEqual(row['quality']['top1_fraction'], json.loads(evidence_path.read_text())['metrics']['top1_fraction'])
        self.assertEqual(row['metrics']['first_pass_wall_s'], 1.5)
        self.assertEqual(row['metrics']['pass_wall_s'], [1.5, 1.5])
        self.assertEqual(row['metrics']['elapsed_wall_s'], 3.)
        self.assertEqual(eviction.call_count, 2)
        store = json.loads((self.root/'fs/store.json').read_text())
        self.assertCountEqual(hashed_files, store['files'])

    def test_dataloader_cache_survives_warmup_and_preserves_tail(self):
        import jsonschema
        _, config, source, _ = self.fixture(cpu_loader='dataloader', prefetch_factor=2,
            request_order='source', decoded_cache_mib=8, host_input_budget_mib=128)
        build_store(source_images(source), self.root/'ic', 'ic_matched_batched', group_size=2,
                    source_artifact=json.loads(config.read_text())['source_manifest'])
        evidence = self.root/'ic.json'
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=self.load_model), patch('builtins.print'):
            verify_inference(config, 'ic', self.root/'ic', evidence)
            row, detail = benchmark(config, self.root/'ic', evidence, backend='ic', run_kind='inference',
                                    warmup_seconds=.001)
        jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).validate(row)
        self.assertEqual(row['quality']['sample_count'], 3)
        self.assertEqual(row['counts']['sample_occurrences'], 3)
        self.assertEqual(row['metrics']['io']['application_read_bytes'], 0)
        self.assertEqual(detail['counters']['decoded_cache_hits'], 2)
        self.assertEqual(detail['counters']['decoded_q_values'], 0)
        self.assertLessEqual(detail['counters']['decoded_cache_peak_bytes'], 8*1024**2)

    def test_natural_cache_is_observed_without_eviction(self):
        _, config, _, _ = self.fixture()
        evidence = self.root/'fs.json'
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=self.load_model), patch('builtins.print'), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.evict_input_files') as evict, \
                patch('fastlens.benchmarks.profiling.input_motivation.cache_pages',
                      return_value=dict(pages=100, resident_pages=40, resident_fraction=.4)):
            verify_inference(config, 'fs', self.root/'fs', evidence)
            row, detail = benchmark(config, self.root/'fs', evidence, backend='fs', observe_cache=True,
                                    warmup_seconds=.001)
        evict.assert_not_called()
        self.assertEqual(row['cache_mode'], 'mixed_measured')
        self.assertEqual(detail['cache_preparation'][0]['resident_pages'], 40)
        self.assertEqual(row['counts']['dataset_passes'], 1)

    def test_parallel_source_freeze_preserves_order_q_tables_and_identity(self):
        args, _, source, _ = self.fixture()
        parallel = freeze_source(args.selection, self.root/'parallel.json', source['preprocessing_recipe'], workers=2)
        self.assertEqual(parallel, source)

    def test_reader_only_uses_worker_cache_and_keeps_all_requests(self):
        from fastlens.benchmarks.baselines.dct_baselines.runner import benchmark_reader
        _, config, source, _ = self.fixture(cpu_loader='dataloader', prefetch_factor=2,
            request_order='source', decoded_cache_mib=8, host_input_budget_mib=128)
        build_store(source_images(source), self.root/'ic', 'ic_matched_batched', group_size=2,
                    source_artifact=json.loads(config.read_text())['source_manifest'])
        with patch.object(Replay, 'pool', side_effect=AssertionError('reader tuning must not materialize model inputs')):
            result = benchmark_reader(config, self.root/'ic', repeats=2)
        self.assertFalse(result['headline_eligible'])
        self.assertEqual(len(result['runs']), 2)
        for run in result['runs']:
            self.assertEqual(run['sample_occurrences'], 3)
            self.assertEqual(run['counters']['decoded_cache_hits'], 2)
            self.assertEqual(run['counters']['inflated_members'], 0)

    def test_cold_cache_requires_observed_zero_residency(self):
        path = self.root/'input.bin'
        path.write_bytes(b'input')
        with patch('fastlens.benchmarks.baselines.dct_baselines.runner.os.posix_fadvise'), \
                patch('fastlens.benchmarks.profiling.input_motivation.cache_pages') as pages:
            pages.return_value = dict(resident_pages=1, pages=1)
            with self.assertRaisesRegex(RuntimeError, 'cache eviction incomplete'):
                evict_input_files([path])
            pages.return_value = dict(resident_pages=0, pages=1)
            self.assertEqual(evict_input_files([path])['resident_pages'], 0)

    def test_checkpoint_state_cpu_preprocessing_predictions_and_tail(self):
        _, config, _, saved = self.fixture()
        def load(profile, device):
            model = TinyCheckpoint()
            model.load_state_dict(torch.load(profile['checkpoint']['path'], weights_only=True), strict=True)
            return model.to(device)
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=load), patch('builtins.print'):
            for backend, root in [('cpu_dct', None), ('fs', self.root/'fs')]:
                result = verify_inference(config, backend, root, self.root/f'{backend}.json')
                self.assertEqual(result['loaded_model_state_sha256'], tensor_state_sha256(saved.state_dict()))
                self.assertEqual(result['occurrence_count'], 3)
                self.assertEqual(result['max_abs_tensor_error'], 0)
                self.assertEqual(result['max_abs_logits_error'], 0)
                self.assertTrue(result['predictions_match'])
                self.assertEqual(result['execution_partition']['augmentation'], 'disabled')
                self.assertEqual(result['execution_partition']['geometry_normalization'], 'cpu')
                self.assertEqual(result['metrics']['cross_entropy'], result['metrics']['reference_cross_entropy'])
            batches = UnifiedReplay.inference_batches
            def corrupt(replay, trace):
                for rows, tensors in batches(replay, trace):
                    yield rows, (tensors[0]+1, *tensors[1:])
            with patch.object(UnifiedReplay, 'inference_batches', corrupt), self.assertRaises(AssertionError):
                verify_inference(config, 'fs', self.root/'fs', self.root/'must-not-pass.json')
        self.assertFalse((self.root/'must-not-pass.json').exists())

    def test_timed_jpeg_and_fs_paths_keep_tail_accuracy_and_report_identity(self):
        import jsonschema
        _, config, source, _ = self.fixture()
        def load(profile, device):
            model = TinyCheckpoint()
            model.load_state_dict(torch.load(profile['checkpoint']['path'], weights_only=True), strict=True)
            return model.to(device).eval()
        rows = []
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=load), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=load), patch('builtins.print'):
            for backend, root in [('cpu_dct', None), ('fs', self.root/'fs')]:
                evidence_path = self.root/f'{backend}.json'
                evidence = verify_inference(config, backend, root, evidence_path)
                row, detail = benchmark(config, root, evidence_path, backend=backend, run_kind='inference')
                jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).validate(row)
                self.assertEqual(row['counts']['sample_occurrences'], 3)
                self.assertEqual(row['quality']['top1_fraction'], evidence['metrics']['top1_fraction'])
                self.assertEqual(row['quality']['top5_fraction'], evidence['metrics']['top5_fraction'])
                self.assertEqual(row['artifacts']['backend'], evidence['backend'])
                self.assertEqual(detail['source_verification']['images'], 3)
                self.assertGreater(row['metrics']['elapsed_wall_s'], 0)
                self.assertFalse(row['headline_eligible'])
                if backend == 'cpu_dct':
                    self.assertIsNone(row['artifacts']['store'])
                    self.assertIsNone(row['metrics']['io']['application_read_bytes'])
                else:
                    self.assertGreater(row['metrics']['io']['application_read_bytes'], 0)
                rows.append(row)
            path = self.root/'runs.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            summary = report([path], self.root/'summary.json')
            self.assertEqual({r['method'] for r in summary}, {'cpu_dct', 'fs_matched_b8'})
            self.assertTrue(all(r['repeats'] == 1 for r in summary))
            Image.new('RGB', (512, 512), 'red').save(source['images'][0]['path'], quality=75, subsampling=2)
            with self.assertRaisesRegex(ValueError, 'current JPEG content'):
                benchmark(config, None, self.root/'cpu_dct.json', backend='cpu_dct', run_kind='inference')

    def test_source_gate_reuse_still_checks_inputs_and_rejects_changed_jpeg(self):
        _, config, source, _ = self.fixture()
        prior = self.root/'prior.json'
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model), patch('builtins.print'):
            verify_inference(config, 'fs', self.root/'fs', prior)
            with patch.object(UnifiedReplay, 'verify_source', side_effect=AssertionError('do not decode q again')):
                result = verify_inference(config, 'fs', self.root/'fs', self.root/'reused.json', source_evidence=prior)
                self.assertEqual(result['source_evidence_reused'], artifact(prior))
                self.assertEqual(result['occurrence_count'], 3)
                Path(source['images'][0]['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'JPEG content changed'):
                    verify_inference(config, 'fs', self.root/'fs', self.root/'invalid.json', source_evidence=prior)

    def test_gate_rejects_wrong_worker_labels_with_unchanged_inputs(self):
        _, config, _, _ = self.fixture()
        original = Replay.pool
        def wrong_labels(replay, rows):
            batches, counters = original(replay, rows)
            return [(tensors, labels+1) for tensors, labels in batches], counters
        with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model), \
                patch.object(Replay, 'pool', wrong_labels), patch('builtins.print'), self.assertRaises(AssertionError):
            verify_inference(config, 'fs', self.root/'fs', self.root/'invalid.json')

    def test_reuse_frozen_source_and_reject_same_path_jpeg_replacement(self):
        args, config_path, source, _ = self.fixture()
        config = json.loads(config_path.read_text())
        args.output = self.root/'second-contract'
        args.source_manifest = Path(config['source_manifest']['path'])
        args.source_recipe = None
        args.selection = None
        prepare(args)
        second = json.loads((args.output/'matched.json').read_text())
        self.assertEqual(second['source_manifest'], config['source_manifest'])
        records = [(r['image_id'], r['label'], r['path']) for r in source['images']]
        before = training_store_identity(self.root/'fs', records, 'fs-matched-dct')
        self.assertEqual(before['experiment_type'], 'shared_partial_processing_training')
        self.assertEqual(verify_source_files(source)['images'], 3)
        Image.new('RGB', (512, 512), 'red').save(records[0][2], quality=75, subsampling=2)
        with self.assertRaisesRegex(ValueError, 'current JPEG content'):
            training_store_identity(self.root/'fs', records, 'fs-matched-dct')
        with self.assertRaisesRegex(ValueError, 'current JPEG content'):
            verify_source_files(source)


class NativePreflightReuseTests(unittest.TestCase):
    def test_full_gate_binds_native_input_content(self):
        tensors = (torch.ones(1, 1), torch.ones(1, 1))
        trace = [dict(image_id='image', target_label=0)]
        records = {'image': {}}
        config = dict(torch_threads=1, tolerance=dict(atol=0., rtol=0.), source_manifest=dict(path='/source', sha256='source'))
        profile = dict(checkpoint={})
        native = MagicMock()
        native.__enter__.return_value = native
        native.identity = dict(kind='fastlens')
        native.reader = None
        native.inference_batches.return_value = iter([(trace, tensors)])
        ref = lambda path: dict(path=str(path), sha256='unchanged')
        with patch('fastlens.benchmarks.baselines.dct_baselines.gates.inputs', return_value=(config, {}, profile, records, trace)), \
                patch('fastlens.benchmarks.baselines.dct_baselines.gates.UnifiedReplay', return_value=native), \
                patch('fastlens.benchmarks.baselines.dct_baselines.gates.artifact', side_effect=ref), \
                patch('fastlens.benchmarks.baselines.dct_baselines.gates.write_json'), \
                patch('fastlens.benchmarks.baselines.dct_baselines.runner.cold_input_files', return_value=[Path('/input.fls'), Path('/access.bin')]), \
                patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', return_value=TinyCheckpoint()), \
                patch('fastlens.benchmarks.baselines.dct_baselines.materialize.source_reference', return_value=tuple(t[0] for t in tensors)), \
                patch('fastlens.benchmarks.baselines.dct_baselines.source.source_images', return_value=[object()]), patch('builtins.print'):
            evidence = verify_inference('/config', 'fastlens', '/manifest', '/unused-output', source_workers=4)
        native.verify_source.assert_called_once_with(workers=4)
        self.assertEqual(evidence['native_input_files'], [ref('/input.fls'), ref('/access.bin')])
        self.assertEqual(evidence['status'], 'passed')
        self.assertTrue(evidence['logits_match'])

    def test_timing_reuses_gate_and_rejects_changed_or_unbound_inputs(self):
        from dataclasses import make_dataclass
        from fastlens.benchmarks.dct_models.storage import BRIDGE
        from fastlens.benchmarks.baselines.dct_baselines import runner
        hashes = {}
        ref = lambda path: dict(path=str(path), sha256=hashes.get(str(path), 'unchanged'))
        config = dict(torch_threads=1, model_microbatch_size=1, preparation_pool_size=1,
                      reader_memory_budget_bytes=1024, host_input_budget_bytes=4096)
        profile = dict(tensors=[dict(shape=[1])])
        trace = [dict(image_id='image', target_label=0)]
        files = [Path('/input.fls'), Path('/access.bin')]
        identity = dict(kind='fastlens', manifest=ref('/manifest'), binding=ref('/binding'))
        evidence = dict(status='passed', resolved_contract=ref('/config'), implementation=[], device='cpu',
                        native_sources=[ref('/native.cpp')], native_input_files=[ref(p) for p in files],
                        backend=dict(identity, coefficient_bridge=ref(BRIDGE)))
        native = MagicMock()
        native.__enter__.return_value = native
        native.identity = identity
        native.pipeline.metrics = make_dataclass('CompleteMetrics', [('complete', bool)])(True)
        native.inference_batches.side_effect = lambda _: iter([(trace, (torch.ones(1, 1),))])
        native.verify_source.side_effect = AssertionError('performance repeats must not decode source q/Q again')
        with patch.object(runner, 'inputs', return_value=(config, {}, profile, {}, trace)), \
                patch.object(Path, 'read_text', side_effect=lambda *a, **k: json.dumps(evidence)), \
                patch.object(runner, 'artifact', side_effect=ref), \
                patch.object(runner, 'implementation_artifacts', return_value=[]), \
                patch.object(runner, 'cold_input_files', return_value=files), \
                patch.object(runner, 'verify_source_files', return_value=dict(images=1)), \
                patch.object(runner, 'result_record', side_effect=lambda *a, **k: dict(
                    resolved_parameters={}, notes=[], counts={}, quality={}, metrics=dict(memory={}, io={}))), \
                patch('fastlens.benchmarks.baselines.dct_baselines.unified.UnifiedReplay', return_value=native), patch('builtins.print'):
            for _ in range(2):
                benchmark('/config', '/manifest', '/evidence', backend='fastlens')
            for path in ('/input.fls', '/access.bin'):
                hashes[path] = 'changed'
                with self.assertRaisesRegex(ValueError, 'native input files changed'):
                    benchmark('/config', '/manifest', '/evidence', backend='fastlens')
                hashes.clear()
            hashes[str(BRIDGE)] = 'changed'
            with self.assertRaisesRegex(ValueError, 'native manifest or binary'):
                benchmark('/config', '/manifest', '/evidence', backend='fastlens')
            hashes.clear()
            del evidence['native_input_files']
            with self.assertRaisesRegex(ValueError, 'lacks content binding'):
                benchmark('/config', '/manifest', '/evidence', backend='fastlens')
        native.verify_source.assert_not_called()
