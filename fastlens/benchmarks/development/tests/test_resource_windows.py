"""Synthetic unit fixtures only; these values never enter experiment outputs."""
import datetime
import io
import json
import unittest
from unittest.mock import patch

from fastlens.benchmarks.development.diagnostics.analyze_resource_windows import ROOT
from fastlens.benchmarks.development.diagnostics.analyze_resource_windows import memory_row


class MemoryWindowTest(unittest.TestCase):
    def test_residency_scope_and_contemporaneous_pool_counters(self):
        stamp=datetime.datetime(2026,9,27,12)
        start=int(stamp.timestamp()*1e9)
        rows=[]
        for used,cached in [(10,1),(1,10)]:
            rows.append(dict(host_rss_bytes=used,tree_rss_bytes=used*2,pinned_allocated_bytes=3,
                native_pools=dict(device_pool=dict(in_use_bytes=used,cached_bytes=cached,peak_in_use_bytes=999),
                                  pinned_pool=dict(in_use_bytes=used,cached_bytes=cached))))
        source=dict(pid=42,start_unix_ns=start,end_unix_ns=start+1_000_000_000,
                    torch_peak_allocated_bytes=30,torch_peak_reserved_bytes=40,measurements=rows)
        gpu='2026/09/27 11:59:59.000,42,gpu,99\n2026/09/27 12:00:00.100,42,gpu,2\n2026/09/27 12:00:00.100,43,gpu,88\n'
        result=dict(profile=False,memory_diagnostic=True,images=16384,correctness_passed=True,
                    timestamp='synthetic',git_commit='synthetic',gpu='synthetic',model='synthetic',domain='DCT',arm='imagelanes')
        with patch('pathlib.Path.read_text',return_value=json.dumps(source)),patch('pathlib.Path.open',return_value=io.StringIO(gpu)):
            row=memory_row(ROOT/'synthetic-unit-fixture',result)
        self.assertEqual(row['gpu_peak_process_bytes'],2*1024*1024)
        self.assertEqual(row['gpu_residency_samples'],1)
        self.assertEqual(row['native_peak_bytes'],11)
        self.assertEqual(row['native_pinned_peak_bytes'],11)
        self.assertEqual(row['torch_peak_allocated_bytes'],30)

    def test_profiled_memory_is_not_primary_memory(self):
        with self.assertRaisesRegex(ValueError,'Unprofiled'):
            memory_row(ROOT,dict(profile=True))


if __name__=='__main__':
    unittest.main()
