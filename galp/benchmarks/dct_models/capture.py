"""Bounded, profiling-only NVTX/CUDA-profiler window shared by the runners."""
from contextlib import nullcontext
import json
import time
import torch


class Capture:
    def __init__(self, enabled, first=256, steps=64, images_per_step=64):
        self.enabled, self.first, self.stop = enabled, first, first + steps
        self.images = steps * images_per_step
        self.initialization_seconds = self.wait_seconds = 0.

    def step(self, index):
        if not self.enabled:
            return
        if index == self.first:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("dctnet.capture")
        elif index == self.stop:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            print(json.dumps(dict(profile_complete=True, captured_images=self.images)), flush=True)
            raise SystemExit(0)

    def range(self, name):
        return torch.cuda.nvtx.range(name) if self.enabled else nullcontext()


def batches(loader, capture):
    started = time.perf_counter()
    iterator = iter(loader)
    capture.initialization_seconds = time.perf_counter()-started
    index = 0
    while True:
        capture.step(index)
        started = time.perf_counter()
        with capture.range("input.wait"):
            try:
                batch = next(iterator)
            except StopIteration:
                capture.wait_seconds += time.perf_counter()-started
                return
        capture.wait_seconds += time.perf_counter()-started
        yield batch
        index += 1
