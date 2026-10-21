import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastlens.benchmarks.execution.revision_results import select_current
from fastlens.benchmarks.execution.revision_results import verify_snapshot


class RevisionResultsTest(unittest.TestCase):
    def test_native_versions_are_not_pooled_and_raw_rows_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)/'snapshot.json'
            snapshot.write_text(json.dumps(dict(native_sha256='new')))
            rows = [dict(gpu='H100', model='Mobile', pipeline='FastLens',
                         implementation_cohort=c, timestamp=str(i), raw_result=str(i))
                    for i,c in enumerate(['native_old', 'native_new', 'native_new'])]
            with patch('fastlens.benchmarks.execution.revision_results.SNAPSHOT', snapshot):
                selected = select_current(rows)
            self.assertEqual([r['raw_result'] for r in selected], ['1', '2'])
            self.assertEqual(len(rows), 3)

    def test_decoder_configs_are_not_pooled(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)/'snapshot.json'
            snapshot.write_text(json.dumps(dict(native_sha256='new')))
            rows = [dict(gpu='4090', model='Swin', pipeline='CoorDL', implementation_cohort=c,
                         timestamp=str(i), raw_result=str(i))
                    for i,c in enumerate(['coordl_cpu', 'coordl_mixed', 'coordl_mixed'])]
            with patch('fastlens.benchmarks.execution.revision_results.SNAPSHOT', snapshot):
                self.assertEqual(len(select_current(rows)), 2)

    def test_changed_code_rejects_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source.py').write_text('changed')
            snapshot = root/'snapshot.json'
            snapshot.write_text(json.dumps(dict(files={'source.py':'old-hash'}, fingerprint='old')))
            with patch('fastlens.benchmarks.execution.revision_results.ROOT', root), patch('fastlens.benchmarks.execution.revision_results.SNAPSHOT', snapshot):
                with self.assertRaisesRegex(RuntimeError, 'source.py'):
                    verify_snapshot()


if __name__ == '__main__': unittest.main()
