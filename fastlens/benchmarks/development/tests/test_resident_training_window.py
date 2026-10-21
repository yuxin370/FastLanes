import unittest
from types import SimpleNamespace
from threading import Event, Thread
from unittest.mock import patch

import torch

from fastlens.benchmarks.paper.section_6_2_4.resident_training.resident_training_window import InitialExecutionState
from fastlens.benchmarks.paper.section_6_2_4.resident_training.resident_training_window import RealResident
from fastlens.benchmarks.training_pls.published_optimizer import build_published_optimizer


class ResidentTrainingTest(unittest.TestCase):
    def test_ffcv_close_stops_producer_before_releasing_memory(self):
        from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import FfcvWindow
        stopped = Event()
        released = []
        iterator = Thread(target=stopped.wait)
        iterator.terminate_event = stopped
        def release():
            self.assertFalse(iterator.is_alive())
            released.append(True)
        iterator.close = release
        iterator.start()
        source = FfcvWindow.__new__(FfcvWindow)
        source.iterator = iterator
        with patch('torch.cuda.synchronize'):
            source.close()
        self.assertEqual(released, [True])
        self.assertIsNone(source.iterator)

    def test_restore_undoes_optimizer_buffers_model_buffers_and_rng(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8),
                                    torch.nn.Dropout(.2), torch.nn.Linear(8, 2)).train()
        optimizer, decay, scheduler = build_published_optimizer(model, total_updates=20000)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        state = InitialExecutionState(model, optimizer, decay, scheduler)
        initial = state.hashes(model, optimizer, decay, scheduler)
        loss = model(torch.randn(6, 4)).square().mean()
        loss.backward()
        lr = scheduler.prepare_next_update()
        optimizer.step()
        decay.step(lr)
        scheduler.complete_update()
        self.assertTrue(optimizer.state)
        state.restore(model, optimizer, decay, scheduler)
        self.assertEqual(initial, state.hashes(model, optimizer, decay, scheduler))
        self.assertFalse(optimizer.state)
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))

    def test_real_batches_are_copied_and_replayed_in_fixed_order(self):
        storage = torch.randn(64, 3, 4, 4).to(memory_format=torch.channels_last)
        class Source:
            position = 0
            def next_batch(self):
                i = self.position
                self.position += 1
                return SimpleNamespace(inputs=(storage,), labels=torch.full((64,), i, dtype=torch.int64),
                                       identities=[i])
        resident = RealResident(Source())
        storage.zero_()
        for i in range(19):
            b = resident.next_batch()
            self.assertEqual(b.identities, [i % 8])
            self.assertTrue(torch.all(b.labels == i % 8))
            self.assertEqual(b.inputs[0].stride(), storage.stride())
            self.assertGreater(torch.count_nonzero(b.inputs[0]), 0)
        resident.reset()
        self.assertEqual(resident.next_batch().identities, [0])


if __name__ == '__main__':
    unittest.main()
