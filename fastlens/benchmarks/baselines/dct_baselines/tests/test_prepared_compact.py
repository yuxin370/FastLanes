"""Offline prepared IC mosaics preserve B8 compact inputs and reusable content checks."""
import json
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from fastlens.benchmarks.baselines.dct_baselines.b8_compact import B8CompactDataset, make_contract
from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact, write_json
from fastlens.benchmarks.baselines.dct_baselines.prepared_compact import PreparedCompactDataset, prepare
from fastlens.benchmarks.baselines.dct_baselines.source import freeze_source, source_images
from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store, fingerprint
from .test_baselines import TemporaryTest


class PreparedCompactTests(TemporaryTest):
    def fixture(self):
        rng = np.random.default_rng(170)
        samples = []
        for index in range(5):
            path = self.root/f'{index}.JPEG'
            Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).save(path, quality=75, subsampling=2)
            samples.append(dict(path=str(path), logical_sample_id=f'train/{index}', label=index % 3))
        selection, source_path, profile = self.root/'selection.json', self.root/'source.json', self.root/'d3.json'
        write_json(selection, dict(split='train', samples=samples))
        freeze_source(selection, source_path, dict(resize='none', encoder='Pillow', quality=75, subsampling=2))
        from fastlens.benchmarks.common import DEFAULT_RGBNOMORE_ROOT
        contract = make_contract(source_path, 3, DEFAULT_RGBNOMORE_ROOT)
        contract['normalization'] = dict(split='train', mean=[-13., .2, 6.1], std=[.51, 2.6, 23.7])
        contract['data_identity'] = fingerprint(contract)
        write_json(profile, contract)
        store = self.root/'source-store'
        build_store(source_images(json.loads(source_path.read_text())), store, 'ic_matched_fixedk', group_size=2,
                    source_frequencies=contract['profile']['source_frequencies'],
                    dependency_evidence=artifact(profile), source_artifact=artifact(source_path))
        with StoreReader(store) as reader:
            evidence = reader.verify_content()
        return profile, store, evidence

    def test_parallel_preparation_int32_mosaics_normalization_order_duplicates_and_tail(self):
        profile, store, evidence = self.fixture()
        outputs = prepare(profile, store, self.root/'prepared', group_size=2, workers=2, content_evidence=evidence)
        indices = [4, 0, 2, 0]
        with B8CompactDataset(profile, 'ic', store, batch_size=2, pool_size=4, workers=2,
                              content_evidence=evidence) as reference:
            expected, labels, _ = reference.read(indices)
        with B8CompactDataset(profile, 'ic', store, batch_size=2, pool_size=4, workers=2,
                              normalize=False, content_evidence=evidence) as reference:
            raw, _, _ = reference.read([4])
        for name, manifest in outputs.items():
            with self.subTest(method=name), PreparedCompactDataset(manifest, workers=2) as dataset:
                actual, actual_labels, counters = dataset.read(indices)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual_labels.tolist(), labels.tolist())
                self.assertEqual(dataset.sample_ids, [f'train/{i}' for i in range(5)])
                self.assertEqual([len(group['indices']) for group in dataset.groups],
                                 [1]*5 if name.endswith('shared') else [2, 2, 1])
                self.assertEqual(counters['file_opens'], 3)
                with np.load(dataset.root/dataset.groups[-1]['file']) as archive:
                    values = archive['q']
                self.assertEqual(values.dtype, np.dtype('<i4'))
                np.testing.assert_array_equal(values[-1], raw[0].numpy())
                with patch.object(PreparedCompactDataset, '_check_content', side_effect=AssertionError('repeated hash')):
                    with PreparedCompactDataset(manifest, workers=2, content_evidence=dataset.content_verification) as reused:
                        self.assertTrue(reused.content_verification['reused'])
                        torch.testing.assert_close(reused.read(indices)[0], expected, atol=0, rtol=0)

    def test_changed_prepared_archive_rejects_reused_evidence_and_new_audit(self):
        profile, store, evidence = self.fixture()
        outputs = prepare(profile, store, self.root/'prepared', group_size=2, workers=1, content_evidence=evidence)
        path = outputs['ic_prepared_batched']
        with PreparedCompactDataset(path) as dataset:
            verified = dataset.content_verification
            archive = dataset.root/dataset.groups[0]['file']
        with archive.open('ab') as output:
            output.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'file state changed'):
            PreparedCompactDataset(path, content_evidence=verified)
        with self.assertRaisesRegex(ValueError, 'payload content changed'):
            PreparedCompactDataset(path)
