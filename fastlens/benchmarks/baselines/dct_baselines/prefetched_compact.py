"""Ordered CPU preparation pools with persistent spawn workers for IC/FS training."""
from __future__ import annotations

import torch

from .compact_training import batch_plan


def _worker_init(_):
    torch.set_num_threads(1)


class _Pools(torch.utils.data.Dataset):
    def __init__(self, factory, ranges):
        self.factory, self.ranges = factory, ranges
        self.dataset = None

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = self.factory()
        start, stop = self.ranges[index]
        values, labels, counters = self.dataset.read(list(range(start, stop)))
        if values.device.type != "cpu" or labels.device.type != "cpu":
            raise ValueError("prefetch workers must produce CPU tensors")
        return values, labels, counters


class PrefetchedCompactDataset:
    """Adapt existing CPU read(indices) datasets without changing the training loop.

    parent supplies metadata only. factory is a picklable callable constructing
    one single-worker reader in each child, with the existing content evidence.
    The caller retains ownership of parent. read() accepts the same ordered pool
    schedule as compact_training; subsequent complete passes reuse the workers.
    """
    def __init__(self, parent, factory, *, batch_size=64, pool_size=256, workers=8,
                 prefetch_factor=2, pin_memory=True):
        if workers < 1 or prefetch_factor < 1:
            raise ValueError("prefetch requires positive workers and prefetch_factor")
        self.manifest, self.image_count, self.input_shape = parent.manifest, parent.image_count, parent.input_shape
        self.execution_partition = dict(read_decode="cpu", normalization="cpu", preparation="persistent spawn workers")
        self.ranges = [(start, stop) for start, stop, _ in batch_plan(self.image_count, batch_size, pool_size, 1)]
        self.loader_parameters = dict(workers=workers, worker_torch_threads=1, prefetch_factor=prefetch_factor,
                                      persistent_workers=True, pin_memory=pin_memory, multiprocessing_context="spawn")
        generator = torch.Generator().manual_seed(0)  # do not consume the model's RNG
        self.loader = torch.utils.data.DataLoader(_Pools(factory, self.ranges), batch_size=None,
            num_workers=workers, prefetch_factor=prefetch_factor, persistent_workers=True,
            pin_memory=pin_memory, multiprocessing_context="spawn", worker_init_fn=_worker_init, generator=generator)
        self.iterator = None
        self.position = 0
        self.closed = False

    def read(self, indices):
        if self.closed:
            raise RuntimeError("prefetch dataset is closed")
        start, stop = self.ranges[self.position]
        if list(indices) != list(range(start, stop)):
            raise ValueError("read must follow the frozen preparation-pool order")
        if self.iterator is None:
            self.iterator = iter(self.loader)  # first worker start is inside the first requested read
        result = next(self.iterator)
        self.position += 1
        if self.position == len(self.ranges):
            try:
                next(self.iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("prefetch produced an unexpected extra pool")
            self.iterator = None
            self.position = 0
        return result

    def close(self):
        if not self.closed:
            # PyTorch exposes no public DataLoader.close for persistent workers.
            # Use the same iterator shutdown used by its destructor, explicitly.
            if self.loader._iterator is not None:
                self.loader._iterator._shutdown_workers()
            self.iterator = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
