"""MPS ownership, monitoring and idle-window regressions; no GPU required."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from fastlens.benchmarks.dct_models import gpu_occupancy as gpu
from fastlens.benchmarks.dct_models import measure_process


class MpsOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.processes = {
            100: Mock(cmdline=lambda: ['/usr/bin/nvidia-cuda-mps-server']),
            200: Mock(cmdline=lambda: ['python'], uids=lambda: SimpleNamespace(real=os.getuid())),
            300: Mock(cmdline=lambda: ['python'], uids=lambda: SimpleNamespace(real=os.getuid()+1)),
        }

    def check(self, clients, visible=(100,), own=()):
        replies = {'get_server_list\n': '100\n', 'get_client_list 100\n': clients}
        with patch.object(gpu, '_query', side_effect=lambda cmd, **kw: replies[kw['input']]), \
                patch.object(gpu.psutil, 'Process', side_effect=self.processes.__getitem__):
            return gpu.gpu_occupancy('GPU-test', processes=visible, own_pids=own)

    def test_idle_server_and_same_uid_clients_do_not_block(self):
        self.assertEqual(self.check('')['blocking_pids'], [])
        result = self.check('200\n', visible=(100, 200))
        self.assertEqual(result['blocking_pids'], [])
        self.assertEqual(result['own_mps_client_pids'], [200])
        self.assertEqual(result['mps_servers'], [100])

    def test_other_uid_mps_and_external_non_mps_block(self):
        self.assertEqual(self.check('200\n300\n')['blocking_pids'], [300])
        self.assertEqual(self.check('', visible=(100, 200))['blocking_pids'], [200])
        self.assertEqual(self.check('', visible=(100, 200), own=(200,))['blocking_pids'], [])

    def test_other_gpu_processes_are_not_queried(self):
        with patch.object(gpu, '_query', return_value='GPU-other, 300\n'), \
                patch.object(gpu.psutil, 'Process') as process:
            self.assertEqual(gpu.gpu_occupancy('GPU-test')['blocking_pids'], [])
            process.assert_not_called()

    def test_control_failure_malformed_reply_and_wrong_server_fail_closed(self):
        with patch.object(gpu.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'mps')):
            with self.assertRaises(gpu.GpuCheckError):
                gpu._control_pids('get_server_list')
        for reply in ('Server not found', '200\nerror', ''):
            with self.subTest(reply=reply), patch.object(gpu, '_query', return_value=reply), \
                    patch.object(gpu.psutil, 'Process', side_effect=self.processes.__getitem__):
                with self.assertRaises(gpu.GpuCheckError):
                    gpu.gpu_occupancy('GPU-test', processes=[100])

    def test_unreadable_client_uid_fails_closed(self):
        self.processes[200].uids = Mock(side_effect=psutil.AccessDenied(200))
        with self.assertRaises(gpu.GpuCheckError):
            self.check('200\n')

    def test_missing_client_requires_confirmed_departure(self):
        self.processes[200].uids = Mock(side_effect=psutil.NoSuchProcess(200))
        with self.assertRaises(gpu.GpuCheckError):
            self.check('200\n')
        with patch.object(gpu, '_query', side_effect=['100\n', '200\n', '']), \
                patch.object(gpu.psutil, 'Process', side_effect=self.processes.__getitem__):
            self.assertEqual(gpu.gpu_occupancy('GPU-test', processes=[100])['blocking_pids'], [])


class MpsMonitorTests(unittest.TestCase):
    def test_monitor_records_sharing_and_invalidates_foreign_clients_or_query_failure(self):
        for blocked, failure in (([], None), ([300], None), ([], gpu.GpuCheckError('permission denied'))):
            with self.subTest(blocked=blocked, failure=failure), \
                    tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                output = Path(directory)/'process.json'
                child = Mock(pid=400, poll=Mock(side_effect=[None, 0, 0]), wait=Mock(return_value=0))
                monitor = Mock(poll=Mock(return_value=None))
                root = Mock(pid=400, children=Mock(return_value=[]),
                            uids=Mock(return_value=SimpleNamespace(real=os.getuid())),
                            ppid=Mock(return_value=1), cmdline=Mock(return_value=['fixture']),
                            memory_info=Mock(return_value=SimpleNamespace(rss=1024)),
                            io_counters=Mock(return_value=SimpleNamespace(read_bytes=0, write_bytes=0, read_chars=0)))
                root.name.return_value = 'fixture'

                def launch(command, **kwargs):
                    if command[0] == 'nvidia-smi':
                        kwargs['stdout'].write('100, 20, GPU-test\n')
                        kwargs['stdout'].flush()
                        return monitor
                    return child

                occupancy = dict(blocking_pids=blocked, mps_servers=[100], own_mps_client_pids=[200])
                with patch('sys.argv', ['measure_process', '--output', str(output), '--gpu-uuid', 'GPU-test', 'fixture']), \
                        patch.object(measure_process.subprocess, 'Popen', side_effect=launch), \
                        patch.object(measure_process.psutil, 'Process', return_value=root), \
                        patch.object(measure_process.time, 'sleep'), \
                        patch.object(measure_process, 'gpu_occupancy', return_value=occupancy, side_effect=failure), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    measure_process.main()
                result = json.loads(output.read_text())
                self.assertEqual(result['valid'], not blocked and failure is None)
                self.assertEqual(stopped.exception.code, 0 if result['valid'] else 1)
                if failure:
                    self.assertIn('permission denied', result['gpu_check_error'])
                else:
                    self.assertEqual(result['foreign_gpu_pids'], blocked)
                    self.assertEqual(result['mps_server_pids'], [100])
                    self.assertEqual(result['same_uid_mps_client_pids'], [200])

    def test_final_identity_removes_stale_classification_but_keeps_unknown_and_foreign_pids(self):
        def state(blocked=(), servers=(), clients=()):
            return dict(blocking_pids=list(blocked), mps_servers=list(servers), own_mps_client_pids=list(clients))

        for pid, states, expected in (
            (100, [state(servers=[100]), state(blocked=[100])], []),
            (401, [state(blocked=[401]), state()], []),
            (200, [state(blocked=[200]), state(clients=[200])], []),
            (300, [state(blocked=[300]), state(blocked=[300])], [300]),
            (900, [state(blocked=[900]), state(blocked=[900])], [900]),
        ):
            with self.subTest(pid=pid), tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                output = Path(directory)/'process.json'
                child = Mock(pid=400, poll=Mock(side_effect=[None, None, 0, 0]), wait=Mock(return_value=0))
                monitor = Mock(poll=Mock(return_value=None))
                worker = Mock(pid=401, memory_info=Mock(return_value=SimpleNamespace(rss=1024)),
                              io_counters=Mock(return_value=SimpleNamespace(read_bytes=0, write_bytes=0, read_chars=0)))
                root = Mock(pid=400, children=Mock(side_effect=[[], [worker]]),
                            memory_info=worker.memory_info, io_counters=worker.io_counters)
                observed = Mock(uids=Mock(return_value=SimpleNamespace(real=os.getuid()+int(pid == 300))),
                                ppid=Mock(return_value=400), cmdline=Mock(return_value=['fixture', 'worker']))
                observed.name.return_value = 'fixture'
                snapshots = []

                def process(process_id):
                    if process_id == 400:
                        return root
                    snapshots.append(process_id)
                    if process_id == 900:
                        raise psutil.NoSuchProcess(process_id)
                    return observed

                def launch(command, **kwargs):
                    if command[0] == 'nvidia-smi':
                        kwargs['stdout'].write(f'{pid}, 20, GPU-test\n')
                        kwargs['stdout'].flush()
                        return monitor
                    return child

                def next_sample(_):
                    with output.with_suffix('.gpu.csv').open('a') as samples:
                        samples.write(f'{pid}, 20, GPU-test\n')

                with patch('sys.argv', ['measure_process', '--output', str(output), '--gpu-uuid', 'GPU-test', 'fixture']), \
                        patch.object(measure_process.subprocess, 'Popen', side_effect=launch), \
                        patch.object(measure_process.psutil, 'Process', side_effect=process), \
                        patch.object(measure_process.time, 'sleep', side_effect=next_sample), \
                        patch.object(measure_process, 'gpu_occupancy', side_effect=states), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    measure_process.main()
                result = json.loads(output.read_text())
                self.assertEqual(result['foreign_gpu_pids'], expected)
                self.assertEqual(result['valid'], not expected)
                self.assertEqual(stopped.exception.code, bool(expected))
                self.assertEqual(snapshots, [pid])
                snapshot = result['extra_gpu_processes'][str(pid)]
                self.assertEqual(snapshot['status'], 'gone' if pid == 900 else 'observed')
                if pid != 900:
                    self.assertEqual(snapshot['uid'], os.getuid()+int(pid == 300))
                    self.assertEqual(snapshot['ppid'], 400)
                    self.assertEqual(snapshot['cmdline'], ['fixture', 'worker'])


class IdleWindowTests(unittest.TestCase):
    def test_both_waiters_reset_the_sixty_second_window_after_contention(self):
        scripts = Path(__file__).resolve().parents[1]
        for name, marker, argv in (
            ('complete_workloads_h100.sh', '"$PYTHON" -B - "$CUDA_VISIBLE_DEVICES" "$previous"', ['wait', 'GPU-test', '/previous']),
            ('run_inference_pilot.sh', '"${python[@]}" - "$CUDA_VISIBLE_DEVICES" "$wait_for_gpu"', ['wait', 'GPU-test', 'true']),
        ):
            body = (scripts/name).read_text().split(marker, 1)[1].split("<<'PY'\n", 1)[1].split('\nPY', 1)[0]
            # The boundary waiter first sees contention; the initial queue
            # also sees one idle poll before contention resets its timer.
            states = ([[]] if name.startswith('complete') else []) + [[300]] + [[]]*7
            elapsed = [0]

            def sleep(seconds):
                elapsed[0] += seconds

            with self.subTest(script=name), patch('sys.argv', argv), \
                    patch.object(gpu, 'gpu_occupancy', side_effect=[dict(blocking_pids=pids) for pids in states]), \
                    patch('psutil.process_iter', return_value=[]), patch('subprocess.run'), \
                    patch('time.monotonic', side_effect=lambda: elapsed[0]), patch('time.sleep', side_effect=sleep), \
                    contextlib.redirect_stdout(io.StringIO()):
                exec(compile(body, name, 'exec'), {})
            self.assertEqual(elapsed[0], 80 if name.startswith('complete') else 70)


if __name__ == '__main__':
    unittest.main()
