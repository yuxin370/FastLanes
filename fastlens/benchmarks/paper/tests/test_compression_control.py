"""CPU checks for Figure 15 input selection and preparation; no GPU measurements."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastlens import cli
from fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor import prepare_compression_control as prepare
from fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor import run
from fastlens.tests.test_cli_native import NativeCliFixture


class CompressionControlTests(unittest.TestCase):
    def test_selects_exact_contiguous_whole_partitions(self):
        shards = [dict(shard_id=i, first_global_image_index=start, image_count=count)
                  for i, (start, count) in enumerate(((0, 2), (2, 3), (5, 4)))]
        manifest = dict(shards=shards)
        self.assertEqual(prepare.select_shards(manifest, 5), shards[:2])
        self.assertEqual(prepare.select_shards(manifest, 3, 2), shards[1:2])
        self.assertEqual(prepare.select_shards(manifest, 3, shard_ids=[1]), shards[1:2])
        for count, start, ids in ((4, 0, None), (2, 1, None), (6, 0, [0, 2]), (4, 0, [0, 0])):
            with self.subTest(count=count, start=start, ids=ids), self.assertRaises(ValueError):
                prepare.select_shards(manifest, count, start, ids)

    def test_run_forwards_explicit_preparation_path(self):
        argv = ['run', 'run', '--preparation', 'prepared/preparation.json', '--dry-run']
        with patch.object(sys, 'argv', argv), patch.object(run, 'execute') as execute:
            run.main()
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index('--preparation') + 1], Path('prepared/preparation.json'))

    def test_rejects_mismatched_prepared_sample_order(self):
        with tempfile.TemporaryDirectory(prefix='fastlens-prepared-pair-') as directory:
            root = Path(directory)
            fixture = cli.REPO_ROOT / 'fastlens/tests/fixtures/formats/block-major'
            for name in ('compressed', 'raw_int16'):
                shutil.copytree(fixture, root / name)
                (root / name / 'samples.json').write_text('[{"sample_id":"one"}]')
                (root / name / 'profile.json').write_text('{}')
            path = root / 'preparation.json'
            path.write_text(json.dumps(dict(samples=1, compressed=str(root / 'compressed'), raw_int16=str(root / 'raw_int16'))))
            self.assertEqual(prepare.prepared_controls(path)['samples'], 1)
            (root / 'raw_int16/samples.json').write_text('[{"sample_id":"different"}]')
            with self.assertRaisesRegex(ValueError, 'physical sample order'):
                prepare.prepared_controls(path)


class CompressionControlNativeTests(NativeCliFixture):
    def test_prepares_matching_raw_and_compressed_controls(self):
        self.convert()
        build = self.tools[cli.TOOL_NAMES[0]].parents[3]
        raw_tool = build / 'fastlens/benchmarks/fastlens_raw_dct_control'
        merge_tool = build / 'fastlens/benchmarks/fastlens_dct_storage'
        if not raw_tool.is_file() or not merge_tool.is_file():
            if os.environ.get('FASTLENS_REQUIRE_PREPARATION_TESTS') == '1':
                self.fail('build fastlens_raw_dct_control and fastlens_dct_storage first')
            self.skipTest('native paper preparation targets unavailable')
        samples = [dict(sample_id=path.name) for path in self.paths]
        (self.output / 'samples.json').write_text(json.dumps(samples))
        (self.output / 'profile.json').write_text('{"geometry":"fixture"}')
        output = self.root / 'controls'
        result = subprocess.run([sys.executable, '-B', '-m',
            'fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.prepare_compression_control',
            '--source-root', str(self.output), '--output-root', str(output), '--build-dir', str(build),
            '--image-count', '3', '--workers', '2'], cwd=cli.REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        prepared = prepare.prepared_controls(output / 'preparation.json')
        self.assertEqual((prepared['samples'], prepared['source_shard_ids']), (3, [0, 1]))
        for name in ('compressed', 'raw_int16'):
            self.assertEqual(json.loads((output / name / 'samples.json').read_text()), samples)
            verified = subprocess.run([str(self.tools[cli.TOOL_NAMES[0]]), '--verify-manifest',
                str(output / name / 'manifest.bin'), '--input-list', str(self.output / 'conversion/sources.txt')],
                capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
