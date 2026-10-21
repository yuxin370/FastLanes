"""Explicit source-frequency budgets preserve the independent JPEG oracle."""
import copy
from contextlib import nullcontext
import json
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from fastlens.benchmarks.baselines.dct_baselines.__main__ import prepare
from fastlens.benchmarks.baselines.dct_baselines.contracts import (
    SourceImage, request_for_row, source_diagonal_frequencies, validate_profile,
)
from fastlens.benchmarks.baselines.dct_baselines.materialize import (
    Materializer, source_diagonal_profile, source_reference,
)
from fastlens.benchmarks.baselines.dct_baselines.runner import benchmark, inputs, verify
from fastlens.benchmarks.baselines.dct_baselines.source import source_images
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store
from .test_baselines import TemporaryTest
from . import test_round4


class PartialInferenceTests(TemporaryTest):
    fixture = test_round4.CheckpointInferenceTests.fixture
    load_model = staticmethod(test_round4.CheckpointInferenceTests.load_model)

    def test_zero_fill_precedes_resize_and_leaves_source_unchanged(self):
        _, config_path, source, _ = self.fixture()
        _, _, full, _, trace = inputs(config_path)
        row = trace[0]
        image = next(image for image in source_images(source) if image.metadata['image_id'] == row['image_id'])
        original = tuple(q.copy() for q in image.coefficients)
        for diagonals in (4, 8, 15):
            profile = source_diagonal_profile(full, diagonals)
            q = tuple(value.copy() for value in image.coefficients)
            # Independent source masking, before the upstream test transform.
            for value in q:
                for u in range(8):
                    for v in range(8):
                        if u+v >= diagonals:
                            value[..., u, v] = 0
            expected = source_reference(full, SourceImage(image.metadata, q, image.tables), row)
            actual = Materializer(profile)(image.coefficients, image.tables, row)
            oracle = source_reference(profile, image, row)
            for candidate in (actual, oracle):
                for a, b in zip(candidate, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(image.coefficients, original):
            np.testing.assert_array_equal(a, b)
        self.assertIs(source_diagonal_profile(full, 15), full)
        wrong = dict(full, source_frequencies=[list(source_diagonal_frequencies(4))]*3)
        with self.assertRaisesRegex(ValueError, 'all 64'):
            validate_profile(wrong)
        wrong = source_diagonal_profile(full, 4)
        wrong['source_frequencies'] = [list(source_diagonal_frequencies(6))]*3
        with self.assertRaisesRegex(ValueError, 'exact diagonal prefix'):
            validate_profile(wrong)

    def test_full_ic_static_ic_and_partial_fs_match_with_real_column_reads(self):
        _, config_path, source, _ = self.fixture(source_diagonals=6)
        config, _, profile, _, trace = inputs(config_path)
        for name, method in [('full', 'ic_matched_shared'), ('static', 'ic_matched_fixedk')]:
            options = (dict(source_frequencies=profile['source_frequencies'],
                            dependency_evidence=config['model_profile']) if name == 'static' else {})
            build_store(source_images(source), self.root/name, method,
                        source_artifact=config['source_manifest'], **options)
        with patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=self.load_model):
            for root in (self.root/'fs', self.root/'full', self.root/'static'):
                evidence = verify(config_path, root, self.root/f'{root.name}.json', check_model=True)
                self.assertEqual(evidence['max_abs_tensor_error'], 0.)
                self.assertTrue(evidence['logits_match'])
                self.assertEqual(evidence['occurrence_count'], 3)  # retained one-sample tail
        partial = request_for_row(trace[0], profile)
        full = request_for_row(trace[0], dict(profile, source_frequencies=[list(range(64))]*3))
        with StoreReader(self.root/'fs') as reader:
            views, selected = reader.read(reader.plan([partial]))
            _, complete = reader.read(reader.plan([full]))
            self.assertLess(selected.application_read_bytes, complete.application_read_bytes)
            self.assertLess(selected.decoded_q_values, complete.decoded_q_values)
            with self.assertRaisesRegex(ValueError, 'complete source dependency'):
                views[0].dense()
            dense = Materializer(profile).source_coefficients(views[0])
            for q in dense:
                self.assertFalse(np.any(q.reshape(*q.shape[:2], 64)[..., [f for f in range(64) if f//8+f%8 >= 6]]))
        with StoreReader(self.root/'static') as reader:
            with self.assertRaises(ValueError):
                reader.plan([full])

    def test_unified_partial_source_verification_inputs_and_model_gate(self):
        from fastlens.benchmarks.baselines.dct_baselines.unified import UnifiedReplay, cpu_raw
        from fastlens.benchmarks.baselines.dct_baselines.gates import verify_inference
        _, config_path, source, _ = self.fixture(source_diagonals=6)
        config, _, profile, records, trace = inputs(config_path)
        static = self.root/'static'
        build_store(source_images(source), static, 'ic_matched_fixedk',
                    source_artifact=config['source_manifest'], source_frequencies=profile['source_frequencies'],
                    dependency_evidence=config['model_profile'])
        for backend, root in (('cpu_dct', None), ('ic', static), ('fs', self.root/'fs')):
            with UnifiedReplay(backend, root, config, source, profile) as replay:
                # The independent JPEG reference must mask before multiplying Q/resize.
                row = trace[0]
                reference = cpu_raw(records[row['image_id']], row, profile, replay.dops)
                actual = replay.raw([row])[0]
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            with patch('fastlens.benchmarks.baselines.dct_baselines.materialize.build_model', side_effect=self.load_model):
                evidence = verify_inference(config_path, backend, root, self.root/f'{backend}.unified.json')
            self.assertTrue(evidence['source_q_exact'])
            self.assertTrue(evidence['final_tensor_match'])
            self.assertTrue(evidence['predictions_match'])
            self.assertEqual(evidence['occurrence_count'], 3)
            self.assertEqual(evidence['max_abs_tensor_error'], 0.)
            self.assertEqual(evidence['max_abs_logits_error'], 0.)
        from fastlens.benchmarks.baselines.dct_baselines.verification import verify_inference as verify_reused
        with patch('fastlens.benchmarks.baselines.dct_baselines.unified.UnifiedReplay.verify_source',
                   side_effect=AssertionError('source q audit repeated')), \
                patch('fastlens.benchmarks.baselines.dct_baselines.verification.build_model', side_effect=self.load_model):
            reused = verify_reused(config_path, 'fs', self.root/'fs', self.root/'fs.reused.json',
                                   cache_dir=self.root/'cache', source_evidence=self.root/'fs.unified.json')
        self.assertTrue(reused['source_q_exact'])
        with patch('fastlens.benchmarks.baselines.dct_baselines.runner.build_model', side_effect=self.load_model):
            row, detail = benchmark(config_path, self.root/'fs', self.root/'fs.reused.json',
                                    device='cpu', run_kind='inference')
        self.assertEqual(row['counts']['sample_occurrences'], 3)

    def test_all64_evidence_cannot_authorize_partial_timing(self):
        args, config_path, _, _ = self.fixture()
        _, _, full_profile, _, _ = inputs(config_path)
        evidence_path = self.root/'all64.json'
        verify(config_path, self.root/'fs', evidence_path)
        args = copy.copy(args)
        args.source_manifest = args.output/'source.json'
        args.source_recipe = None
        args.output = self.root/'partial-contract'
        args.source_diagonals = 4
        with patch('fastlens.benchmarks.baselines.dct_baselines.__main__.resolve_profile', return_value=full_profile):
            prepare(args)
        with self.assertRaisesRegex(ValueError, 'another contract'):
            benchmark(args.output/'matched.json', self.root/'fs', evidence_path)


class NativePartialSelectionTests(TemporaryTest):
    def test_storage_column_order_and_mixed_shard_rejection(self):
        from fastlens.benchmarks.baselines.dct_baselines.unified import native_source_frequencies
        frequencies = list(source_diagonal_frequencies(4))
        profile = dict(source_diagonals=4, source_frequencies=[frequencies]*3)
        paths = [self.root/'natural.meta.bin', self.root/'zigzag.meta.bin']
        for version, offset in ((1, 14), (3, 16)):
            for path, zigzag in zip(paths, (False, True)):
                header = bytearray(18)
                header[:8] = f'GJDCTMD{version}'.encode()
                header[8:10] = version.to_bytes(2, 'little')
                header[offset:offset+2] = int(zigzag).to_bytes(2, 'little')
                path.write_bytes(header)
            for path, expected in zip(paths, (tuple(frequencies), tuple(range(len(frequencies))))):
                with patch('fastlens.benchmarks.system_dct_major.common.parse_manifest',
                           return_value=dict(shards=[dict(metadata_path=str(path))])):
                    self.assertEqual(native_source_frequencies('/unused', profile), expected)
        with patch('fastlens.benchmarks.system_dct_major.common.parse_manifest',
                   return_value=dict(shards=[dict(metadata_path=str(path)) for path in paths])):
            with self.assertRaisesRegex(ValueError, 'common coefficient order'):
                native_source_frequencies('/unused', profile)
        with patch('fastlens.benchmarks.system_dct_major.common.parse_manifest', side_effect=AssertionError('all64 changed')):
            self.assertEqual(native_source_frequencies('/unused', {}), 'all')

    def test_cnn_and_transformer_pass_partial_columns_to_native(self):
        from fastlens.benchmarks.baselines.dct_baselines.unified import UnifiedReplay
        from fastlens.torch.direct_dct import _cnn_pipeline_options
        selected = tuple(range(10))
        profile = dict(version=1, representation='jpeg_b8', source_diagonals=4,
                       source_frequencies=[list(source_diagonal_frequencies(4))]*3,
                       source_zero_fill='quantized_before_dequantization', dependency_derivation='explicit lossy test',
                       upstream=str(self.root), output_grids=[28]*3, model='resnet24',
                       output_channels=[[0, 0]], mean=[0.], std=[1.])
        manifest = self.root/'manifest.bin'
        manifest.write_bytes(b'fixture')
        for transform in ('cnn_source512', 'rgbnomore_source512'):
            pipeline = MagicMock()
            with patch('fastlens.benchmarks.baselines.dct_baselines.unified.upstream_ops'), \
                    patch('fastlens.benchmarks.baselines.dct_baselines.materialize.load_upsample_dct'), \
                    patch('fastlens.benchmarks.baselines.dct_baselines.unified.native_source_frequencies', return_value=selected), \
                    patch('fastlens.torch.direct_dct.DirectDctPipeline', return_value=pipeline) as cnn, \
                    patch('fastlens.torch.direct_dct.DirectDctReader') as transformer, \
                    patch('torch.cuda.device', side_effect=lambda *_: nullcontext()):
                transformer.return_value.pipeline.return_value = pipeline
                UnifiedReplay('fastlens', manifest, dict(model_microbatch_size=1),
                              dict(images=[dict(image_id='0', galp_image_id=0)]),
                              dict(profile, transform=transform), device='cuda:0')
                if transform == 'cnn_source512':
                    native = cnn.call_args.kwargs['profile']
                    self.assertEqual(native.source_frequency_policy, selected)
                    self.assertFalse(_cnn_pipeline_options(native)['grid_transform']['require_all_coefficients'])
                    transformer.assert_not_called()
                else:
                    cnn.assert_not_called()
                    self.assertEqual(transformer.return_value.pipeline.call_args.kwargs['coefficients'], selected)
