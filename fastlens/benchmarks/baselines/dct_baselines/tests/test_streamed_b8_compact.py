"""Native-style streaming preserves compact inputs, pool boundaries and real SGD updates."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fastlens.benchmarks.baselines.dct_baselines.b8_compact import B8CompactDataset
from fastlens.benchmarks.baselines.dct_baselines.compact_training import _train_pass, batch_plan
from fastlens.benchmarks.baselines.dct_baselines.streamed_b8_compact import StreamedB8CompactDataset
from fastlens.benchmarks.baselines.dct_baselines.unified import UnifiedReplay


class NativeBatch:
    def __init__(self, pipeline, ids):
        self.pipeline, self.global_image_ids = pipeline, ids
        self.recorded = False

    def record_stream(self, stream):
        self.pipeline.streams.append(stream)
        self.recorded = True

    @property
    def projected(self):
        assert self.recorded
        self.pipeline.range_reads += 1
        return torch.stack([self.pipeline.values[index] for index in self.global_image_ids])


class NativePipeline:
    def __init__(self, values, batch_size):
        self.values, self.batch_size = values, batch_size
        self.starts, self.streams = [], []
        self.range_reads = 0
        self.closed = False

    def start(self, ids, *, transforms_by_batch):
        assert not self.closed
        self.starts.append((ids, transforms_by_batch))
        self.pending = iter([group[start:start+self.batch_size] for group in ids
                             for start in range(0, len(group), self.batch_size)])

    def __iter__(self):
        return self

    def __next__(self):
        return NativeBatch(self, next(self.pending))

    def close(self):
        self.closed = True


def dataset(count=9, *, streamed=True):
    kind = StreamedB8CompactDataset if streamed else B8CompactDataset
    data = kind.__new__(kind)
    data.image_count, data.input_shape = count, (3, 84, 84)
    data.manifest = dict(split='train')
    data.sample_ids = [f'train/{index}' for index in range(count)]
    data.labels = [index % 2 for index in range(count)]
    data.batch_size, data.pool_size, data.side = 2, 4, 3
    data.device, data.normalize, data.cpu = torch.device('cpu'), True, None
    data.profile = dict(id='fixture', default_crop=[32, 32, 448, 448], transform='cnn_source512')
    data.u, data.v = torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([0, 1, 2, 0, 1, 0])
    data.mean = torch.tensor([-1.7, .3, 2.5])[None, :, None, None]
    data.std = torch.tensor([2., 1.3, 4.7])[None, :, None, None]
    records = {name: dict(galp_image_id=70+index*2) for index, name in enumerate(data.sample_ids)}
    values = {70+index*2: ((torch.arange(18*28*28).reshape(18, 28, 28)+index*7) % 23-11).float()
              for index in range(count)}
    pipeline = NativePipeline(values, data.batch_size)
    producer = SimpleNamespace(records=records, pipeline=pipeline, profile=data.profile, device=data.device,
                               config=dict(model_microbatch_size=data.batch_size, preparation_pool_size=data.pool_size))
    producer.inference_batches = lambda trace: UnifiedReplay.inference_batches(producer, trace)
    data.producer = producer
    return data


class StreamedCompactTests(unittest.TestCase):
    @patch('torch.cuda.current_stream', return_value='consumer')
    def test_one_start_multiple_pools_first_batch_is_immediate_and_repeated_pass_tail_is_retained(self, _):
        data = dataset()
        plan = batch_plan(data.image_count, 2, 4, 2)
        for repeat in range(2):
            before = data.producer.pipeline.range_reads
            stream = data.iter_training_batches(plan)
            first, labels, _ = next(stream)
            self.assertEqual(first.shape, (2, 3, 84, 84))
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(data.producer.pipeline.range_reads-before, 1)
            self.assertEqual(len(data.producer.pipeline.starts), repeat+1)
            self.assertEqual([len(pool) for pool in data.producer.pipeline.starts[-1][0]], [4, 5])
            self.assertTrue(torch.is_grad_enabled())
            rest = list(stream)
            self.assertEqual([len(first), *[len(values) for values, _, _ in rest]], [2, 2, 2, 3])
            self.assertEqual(torch.cat([labels, *[labels for _, labels, _ in rest]]).tolist(), data.labels)
            self.assertFalse(data.producer.pipeline.closed)
            self.assertEqual(data.producer.pipeline.streams, ['consumer']*(5*(repeat+1)))

    @patch('torch.cuda.current_stream', return_value='consumer')
    def test_inputs_and_real_sgd_updates_equal_pool_path_with_accumulation_and_bn_safe_tail(self, _):
        old, new = dataset(streamed=False), dataset()
        plan = batch_plan(new.image_count, 2, 4, 2)
        expected = torch.cat([old.read(list(range(start, stop)))[0] for start, stop, _ in plan])
        actual = torch.cat([values for values, _, _ in new.iter_training_batches(plan)])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.manual_seed(13)
        a = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.BatchNorm2d(3),
                                torch.nn.Flatten(), torch.nn.Linear(3, 2))
        b = copy.deepcopy(a)
        a_opt = torch.optim.SGD(a.parameters(), lr=.03, momentum=.9, weight_decay=4e-5)
        b_opt = torch.optim.SGD(b.parameters(), lr=.03, momentum=.9, weight_decay=4e-5)
        for _ in range(2):
            old_result = _train_pass(old, a, a_opt, plan, torch.device('cpu'))
            new_result = _train_pass(new, b, b_opt, plan, torch.device('cpu'))
            self.assertEqual(old_result, new_result)
        for name, expected in a.state_dict().items():
            torch.testing.assert_close(b.state_dict()[name], expected, atol=0, rtol=0)
        for state_a, state_b in zip(a_opt.state.values(), b_opt.state.values()):
            torch.testing.assert_close(state_a['momentum_buffer'], state_b['momentum_buffer'], atol=0, rtol=0)

    @patch('torch.cuda.current_stream', return_value='consumer')
    def test_early_close_order_failure_and_training_exception_close_native_ownership(self, _):
        plan = batch_plan(9, 2, 4, 1)
        data = dataset()
        stream = data.iter_training_batches(plan)
        next(stream)
        stream.close()
        self.assertTrue(data.producer.pipeline.closed)
        data = dataset()
        def fail(_):
            raise RuntimeError('model failed')
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with patch.object(model, 'forward', fail), self.assertRaisesRegex(RuntimeError, 'model failed'):
            _train_pass(data, model, optimizer, plan, torch.device('cpu'))
        self.assertTrue(data.producer.pipeline.closed)
        data = dataset()
        original_start = data.producer.pipeline.start
        def reversed_start(ids, **kwargs):
            original_start([list(reversed(group)) for group in ids], **kwargs)
        with patch.object(data.producer.pipeline, 'start', reversed_start), self.assertRaisesRegex(AssertionError, 'order differs'):
            list(data.iter_training_batches(plan))
        self.assertTrue(data.producer.pipeline.closed)
