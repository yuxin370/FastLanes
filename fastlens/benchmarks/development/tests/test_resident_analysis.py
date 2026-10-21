import unittest
from pathlib import Path
import sqlite3
import tempfile

from fastlens.benchmarks.development.diagnostics.analyze_resident_trace import analyze
from fastlens.benchmarks.development.diagnostics.analyze_resident_trace import category
from fastlens.benchmarks.development.diagnostics.analyze_resident_trace import subtract
from fastlens.benchmarks.development.diagnostics.analyze_resident_trace import union
from fastlens.benchmarks.development.diagnostics.profile_resident_study import select


class ResidentAnalysisTest(unittest.TestCase):
    def test_overlapping_wait_intervals_count_once_and_preserve_unknown_gaps(self):
        waits = [[10, 20], [15, 30], [30, 35], [50, 60]]
        self.assertEqual(union(waits), [[10, 35], [50, 60]])
        self.assertEqual(subtract([[0, 100]], waits), [[0, 10], [35, 50], [60, 100]])

    def test_profile_selection_uses_gap_and_predeclared_ffcv_tie(self):
        rows = [dict(gpu=gpu, model=model, g_DCT=gap, ffcv_raw=raw, ffcv_jpeg=jpeg)
                for gpu, model, gap, raw, jpeg in (
                    ('4090', 'vitti', -.02, 101, 100), ('4090', 'swinv2', .2, 150, 100),
                    ('H100', 'vitti', .04, 120, 100), ('H100', 'swinv2', .08, 95, 100))]
        largest, closest = select(rows)
        self.assertEqual((largest['gpu'], largest['model'], largest['ffcv']), ('4090', 'swinv2', 'ffcv_raw'))
        self.assertEqual((closest['gpu'], closest['model'], closest['ffcv']), ('4090', 'vitti', 'ffcv_jpeg'))
        self.assertEqual(closest['g_DCT'], -.02)

    def test_fused_kernel_is_not_reported_as_pure_mixup(self):
        self.assertEqual(category('projected_normalize_mixup_kernel', ''),
                         'Fused normalization / Mixup (full or basic)')
        self.assertEqual(category('ampere_sgemm', 'model.backward u=1 m=2 b=18'), 'Model backward')

    def test_event_reuse_and_autograd_worker_keep_causal_attribution(self):
        with tempfile.TemporaryDirectory(prefix='resident-trace-test-', dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            db = sqlite3.connect(root/'trace.sqlite')
            db.executescript('''
                CREATE TABLE StringIds(id,value);
                CREATE TABLE NVTX_EVENTS(start,end,globalTid,text,textId);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,streamId,correlationId,shortName);
                CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT(globalPid,contextId,eventId,eventSyncId,timestamp,streamId);
                CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(globalPid,contextId,eventId,eventSyncId,streamId,syncType,correlationId);
            ''')
            db.executemany('INSERT INTO StringIds VALUES (?,?)', [(1,'ampere_sgemm'),(2,'normalize_mixup_kernel')])
            db.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)', [
                (0,1000,1,'resident_study.measured_window'), (0,50,1,'optimizer.zero u=0'),
                (100,200,1,'input.request u=0 m=0 b=0'), (200,250,1,'model.forward u=0 m=0 b=0'),
                (250,300,1,'model.backward u=0 m=0 b=0'), (400,450,1,'input.request u=0 m=1 b=1'),
                (450,500,1,'model.forward u=0 m=1 b=1')])
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)', [
                (1,2,1,1), (30,40,3,2), (110,120,1,3), (210,220,1,4),
                (260,270,2,5), (410,420,1,6), (460,470,1,7)])
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)', [
                (10,100,7,1,1), (40,300,19,2,2), (350,400,7,4,1), (400,500,7,5,1), (600,650,7,7,1)])
            # Same event handle recorded twice; eventSyncId selects the right completion.
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_CUDA_EVENT VALUES (?,?,?,?,?,?)',
                           [(1,1,10,1,300,19),(1,1,10,2,550,19)])
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (?,?,?,?,?,?,?)',
                           [(1,1,10,1,7,2,3),(1,1,10,2,7,2,6)])
            db.commit()
            db.close()
            report = analyze(root/'trace.sqlite', root)
            self.assertAlmostEqual(report['confirmed_input_wait_ms'], 130/1e6)
            backward, = [r for r in report['categories'] if r['category']=='Model backward']
            self.assertEqual(backward['calls'], 1)
            self.assertAlmostEqual(backward['cumulative_gpu_ms'], 100/1e6)


if __name__ == '__main__':
    unittest.main()
