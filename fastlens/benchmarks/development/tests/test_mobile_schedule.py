import os
os.environ.setdefault('DCTNET_PROFILE','mobilenet24')
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from fastlens.benchmarks.dct_models.training_jpeg import JpegTrainingPipeline


class MatchedPoolScheduleTest(unittest.TestCase):
    def test_partial_segment_keeps_pool_and_microbatch_boundaries(self):
        p=JpegTrainingPipeline.__new__(JpegTrainingPipeline)
        p.sample_count=17*1024+143
        p.records=[('id',0,'unused')]*p.sample_count
        p.augment=SimpleNamespace(input_channels=[])
        p.grid,p.workers,p.microbatch,p.pool_images=112,0,64,4096
        p.condition='B6'
        p.store=None
        p.iterator=None
        with patch('fastlens.benchmarks.dct_models.training_jpeg.DataLoader',return_value=[]) as loader:
            p.start_epoch(0)
        dataset=loader.call_args.args[0]
        batches=loader.call_args.kwargs['batch_sampler']
        self.assertEqual(sorted(dataset.order),list(range(p.sample_count)))
        self.assertEqual([i for batch in batches for i in batch],list(range(p.sample_count)))
        offset=0
        for index,length in enumerate(p.pool_lengths):
            begin=p.microbatch_offsets[index]
            stop=p.microbatch_offsets[index+1] if index+1<len(p.pool_lengths) else len(batches)
            selected=batches[begin:stop]
            self.assertEqual(sum(map(len,selected)),length)
            self.assertTrue(all(len(b)==64 for b in selected[:-1]))
            self.assertEqual(selected[0][0],offset)
            self.assertEqual(selected[-1][-1],offset+length-1)
            offset+=length
        self.assertEqual(sum(len(b)<64 for b in batches),1)
        self.assertEqual(dataset.crop_policy,'per-pls')


if __name__=='__main__': unittest.main()
