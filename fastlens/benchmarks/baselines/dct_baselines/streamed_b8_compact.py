"""Stream native projected microbatches into the unchanged B8 compact training recipe."""
from __future__ import annotations

import torch

from .b8_compact import B8CompactDataset


class StreamedB8CompactDataset(B8CompactDataset):
    def __init__(self, contract, root, *, device='cuda:0', module_path=None,
                 batch_size=64, pool_size=256, workers=8, decode_budget_mib=1024,
                 reader_memory_mib=2048, normalize=True):
        super().__init__(contract, 'fastlens', root, device=device, module_path=module_path,
                         batch_size=batch_size, pool_size=pool_size, workers=workers,
                         decode_budget_mib=decode_budget_mib, reader_memory_mib=reader_memory_mib,
                         normalize=normalize)

    def _mosaic(self, projected):
        # End no_grad before yielding: the caller's forward/backward must retain autograd.
        with torch.no_grad():
            n = len(projected)
            output = torch.zeros((n, *self.input_shape), dtype=torch.float32, device=self.device)
            blocks = output.reshape(n, 3, 28, self.side, 28, self.side).permute(0, 1, 2, 4, 3, 5)
            selected = projected.reshape(n, 3, len(self.u), 28, 28).permute(0, 1, 3, 4, 2)
            blocks[..., self.u, self.v] = selected
            if self.normalize:
                output.sub_(self.mean).div_(self.std)
            return output

    def iter_training_batches(self, plan):
        """Submit a whole pass once; retain the existing pool/batch and singleton-tail plan."""
        ids = [[self.producer.records[self.sample_ids[index]]['galp_image_id'] for index in range(start, stop)]
               for start, stop, _ in plan]
        transforms = [[dict(crop=dict(zip(('x', 'y', 'width', 'height'), self.profile['default_crop'])),
                            horizontal_flip=False, logical_sample_id=self.sample_ids[index])
                       for index in range(start, stop)] for start, stop, _ in plan]
        pipeline = self.producer.pipeline
        completed = False
        owner = projected = values = None
        parts = []
        try:
            pipeline.start(ids, transforms_by_batch=transforms)
            native = iter(pipeline)
            consumed = offset = 0
            for pool_start, _, windows in plan:
                for window in windows:
                    for row in window:
                        left, right = pool_start+row['start'], pool_start+row['stop']
                        needed = right-left
                        while needed:
                            if projected is None:
                                owner = next(native)
                                batch_ids = owner.global_image_ids
                                expected = [self.producer.records[name]['galp_image_id']
                                            for name in self.sample_ids[consumed:consumed+len(batch_ids)]]
                                if not batch_ids or batch_ids != expected:
                                    raise AssertionError('native compact occurrence order differs from training plan')
                                # Storage-backed projected ranges retain their lease; record the stream
                                # before enqueueing the compact copy on that same consumer stream.
                                owner.record_stream(torch.cuda.current_stream(self.device))
                                projected = owner.projected
                                offset = 0
                                consumed += len(batch_ids)
                            count = min(needed, len(projected)-offset)
                            parts.append((owner, projected[offset:offset+count]))
                            offset += count
                            needed -= count
                            if offset == len(projected):
                                owner = projected = None
                        # Concatenate only when the unchanged BN-safe tail spans native ranges.
                        values = parts[0][1] if len(parts) == 1 else torch.cat([part for _, part in parts])
                        output = self._mosaic(values)
                        values = None
                        parts.clear()  # record_stream protects the already enqueued native-to-mosaic copy.
                        labels = torch.tensor(self.labels[left:right], dtype=torch.int64)
                        yield output, labels, None
            if consumed != self.image_count or projected is not None:
                raise AssertionError('native compact pass does not cover the training population')
            try:
                next(native)
            except StopIteration:
                completed = True
            else:
                raise AssertionError('native compact pipeline yielded extra occurrences')
        finally:
            parts.clear()
            owner = projected = values = None
            if not completed:
                pipeline.close()
