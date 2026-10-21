import unittest
from unittest.mock import patch, Mock

import psutil
from fastlens.benchmarks.development.historical.run_p0_after_wait import external_pending
from fastlens.benchmarks.development.historical.run_p0_after_wait import compute_pids


class WaitTest(unittest.TestCase):
    def test_wait_tracks_original_process_not_reused_pid(self):
        process = Mock()
        process.create_time.return_value = 100.0
        process.status.return_value = psutil.STATUS_RUNNING
        with patch('fastlens.benchmarks.development.historical.run_p0_after_wait.psutil.Process', return_value=process):
            self.assertTrue(external_pending(123, 100.0))
            self.assertFalse(external_pending(123, 99.0))
            process.status.return_value = psutil.STATUS_ZOMBIE
            self.assertFalse(external_pending(123, 100.0))

    def test_completed_process_releases_wait(self):
        with patch('fastlens.benchmarks.development.historical.run_p0_after_wait.psutil.Process', side_effect=psutil.NoSuchProcess(123)):
            self.assertFalse(external_pending(123, 100.0))

    def test_compute_guard_includes_all_gpu_processes(self):
        with patch('fastlens.benchmarks.development.historical.run_p0_after_wait.subprocess.check_output', return_value='100\n200\n100\n'):
            self.assertEqual(compute_pids(), {100, 200})
        with patch('fastlens.benchmarks.development.historical.run_p0_after_wait.subprocess.check_output', return_value=''):
            self.assertEqual(compute_pids(), set())


if __name__ == '__main__':
    unittest.main()

class ResourceGuardTest(unittest.TestCase):
    def setUp(self):
        topology=patch('fastlens.benchmarks.execution.resource_guard.gpu_numa_nodes',
                       return_value={'local-gpu':0, 'H100':0, 'RTX4090':0, 'remote-gpu':1})
        self.topology=topology.start()
        self.addCleanup(topology.stop)
        patcher=patch('fastlens.benchmarks.execution.resource_guard.ResourceGuard.host_blockers',return_value=set())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_same_locality_job_blocks(self):
        from fastlens.benchmarks.execution.resource_guard import ResourceGuard
        with patch('fastlens.benchmarks.execution.resource_guard.subprocess.check_output', return_value='7, local-gpu\n'):
            self.assertEqual(ResourceGuard('local-gpu').blockers(), {7})
            self.assertEqual(ResourceGuard('local-gpu').blockers({7}), set())

    def test_remote_job_is_annotated_without_blocking(self):
        from fastlens.benchmarks.execution.resource_guard import ResourceGuard
        process = Mock()
        process.pid = 7
        process.children.return_value = []
        process.threads.return_value = []
        process.cpu_affinity.return_value = [24, 25]
        process.create_time.return_value = 10
        process.io_counters.return_value = Mock(read_bytes=0, write_bytes=0, read_chars=10)
        guard = ResourceGuard('local-gpu')
        with patch('fastlens.benchmarks.execution.resource_guard.subprocess.check_output', return_value='7, remote-gpu\n'), \
             patch('fastlens.benchmarks.execution.resource_guard.psutil.Process', return_value=process), \
             patch('fastlens.benchmarks.execution.resource_guard.Path.read_text', return_value='anon=10 N1=10\n'):
            self.assertEqual(guard.blockers(), set())
            self.assertEqual(guard.blockers(), set())
            self.assertFalse(guard.observations[-1]['blocking'])
            process.io_counters.return_value.read_chars = 20
            self.assertEqual(guard.blockers(), set())
            process.cpu_affinity.return_value = [0, 24]
            self.assertEqual(guard.blockers(), set())

    def test_other_local_gpu_can_run_with_remote_host_resources(self):
        from fastlens.benchmarks.execution.resource_guard import ResourceGuard
        process = Mock(pid=7)
        process.children.return_value = []
        process.threads.return_value = [Mock(id=8)]
        process.cpu_affinity.return_value = [24]
        process.create_time.return_value = 10
        process.io_counters.return_value = Mock(read_bytes=0, write_bytes=0, read_chars=10)
        guard = ResourceGuard('H100')
        with patch('fastlens.benchmarks.execution.resource_guard.subprocess.check_output', return_value='7, RTX4090\n'), \
             patch('fastlens.benchmarks.execution.resource_guard.psutil.Process', return_value=process), \
             patch('fastlens.benchmarks.execution.resource_guard.Path.read_text', return_value='anon=10 N1=10\n'), \
             patch('fastlens.benchmarks.execution.resource_guard.resident_model_window', return_value=True) as resident, \
             patch('fastlens.benchmarks.execution.resource_guard.os.sched_getaffinity', return_value={24}) as affinity:
            self.assertEqual(guard.blockers(), set())
            self.assertEqual(guard.blockers(), set())
            affinity.return_value = {0}
            self.assertEqual(guard.blockers(), set())
            affinity.return_value = {24}
            resident.return_value = False
            self.assertEqual(guard.blockers(), set())

    def test_moved_gpu_uses_new_node_and_recognizes_its_neighbor(self):
        from fastlens.benchmarks.execution.resource_guard import ResourceGuard
        self.topology.return_value['RTX4090']=1
        with patch('fastlens.benchmarks.execution.resource_guard.Path.read_text',return_value='24-47,72-95'):
            guard=ResourceGuard('RTX4090')
        self.assertEqual(guard.numa_node,1)
        self.assertEqual(guard.local_cpus,set(range(24,48))|set(range(72,96)))
        process=Mock(pid=7)
        process.children.return_value=[]
        process.threads.return_value=[]
        process.cpu_affinity.return_value=[0]
        process.create_time.return_value=10
        process.io_counters.return_value=Mock(read_bytes=0,write_bytes=0,read_chars=10)
        with patch('fastlens.benchmarks.execution.resource_guard.subprocess.check_output',return_value='7, remote-gpu\n'), \
             patch('fastlens.benchmarks.execution.resource_guard.psutil.Process',return_value=process), \
             patch('fastlens.benchmarks.execution.resource_guard.Path.read_text',return_value='anon=10 N0=10\n'), \
             patch('fastlens.benchmarks.execution.resource_guard.resident_model_window',return_value=False):
            guard.blockers()
            guard.blockers()
        observed=guard.observations[-1]
        self.assertFalse(observed['independent_pcie_or_resident_window'])
        self.assertEqual(observed['reason'],'shared or unverified host resources')


class HostLoadTest(unittest.TestCase):
    def test_cpu_only_bulk_job_is_annotated(self):
        from fastlens.benchmarks.execution.resource_guard import ResourceGuard
        process=Mock()
        process.info=dict(pid=42,create_time=1,cpu_times=Mock(user=10,system=0),cpu_affinity=[0,48])
        with patch('fastlens.benchmarks.execution.resource_guard.gpu_numa_nodes',return_value={'gpu':0}):
            guard=ResourceGuard('gpu')
        with patch('fastlens.benchmarks.execution.resource_guard.psutil.process_iter',return_value=[process]), \
             patch('fastlens.benchmarks.execution.resource_guard.time.monotonic',side_effect=[1,2,3,4]):
            self.assertEqual(guard.host_blockers(set()),set())
            process.info['cpu_times'].user=34
            self.assertEqual(guard.host_blockers(set()),set())
            self.assertEqual(guard.observations[-1]['pid'],42)
            process.info['cpu_times'].user=58
            process.info['cpu_affinity']=[24,72]
            self.assertEqual(guard.host_blockers(set()),set())
            self.assertEqual(guard.host_blockers({42}),set())


class NativeWindowTailTest(unittest.TestCase):
    def test_partial_pool_tail_is_recorded_and_full_batch_follows(self):
        from types import SimpleNamespace
        from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import NativeWindow
        tail=SimpleNamespace(global_image_ids=[0,1],is_pool_end=True)
        full=SimpleNamespace(global_image_ids=list(range(2,66)),is_pool_end=False,projected='tensor',targets='targets',epoch=2)
        # A real iterator exercises next(pool), with an observable retirement.
        class Pool:
            def __init__(self,batch): self.batch=batch; self.retired=False
            def __next__(self): return self.batch
            def retire(self): self.retired=True
        old_pool=Pool(tail)
        adapter=NativeWindow.__new__(NativeWindow)
        adapter.pool=old_pool
        adapter.previous_end=False
        adapter.pipeline=Mock()
        adapter.pipeline.next_pool.return_value=Pool(full)
        adapter.model='mobilenet'
        adapter.records=[{'logical_sample_id':str(i)} for i in range(66)]
        adapter.skipped_partial_ids=[]
        with patch('fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters.torch.cuda.synchronize'):
            batch=adapter.next_batch()
        self.assertTrue(old_pool.retired)
        self.assertEqual(adapter.skipped_partial_ids,['0','1'])
        self.assertEqual([i.logical_sample_id for i in batch.identities],[str(i) for i in range(2,66)])
        self.assertEqual({i.epoch for i in batch.identities},{2})
        self.assertEqual(batch.inputs,('tensor',))
