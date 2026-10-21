"""Exercise 64+64+15 RGB batches with the actual four deployment readers."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',choices=['pytorch','dali','coordl','ffcv'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import cv2
    cv2.setNumThreads(1)
    if args.pipeline=='ffcv':
        from ffcv.loader import Loader
    if args.pipeline=='coordl':
        from nvidia.dali import backend
    import torch
    from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingSample,DaliTrainingAdapter,resolve_dali_variant
    from fastlens.benchmarks.baselines.system_rgbnomore.training.sample_order import SampleIdentity
    from fastlens.benchmarks.baselines.system_rgbnomore.training.augmentation import derive_augmentation
    from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import PlannedRgbWindow,FfcvWindow,CoorDLWindow,check_rgb_input
    torch.cuda.set_device(0);torch.set_num_threads(8);device=torch.device('cuda:0')
    entries=json.loads((ROOT/'fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json').read_text())['samples']
    entries=entries[:128]+entries[-15:]
    samples=[TrainingSample(e['logical_sample_id'],Path(e['path']),e['label'],e['width'],e['height'],e['galp_image_id']) for e in entries]
    decisions=[derive_augmentation(seed=11997733,epoch=2,logical_sample_id=s.logical_sample_id,source_width=s.width,source_height=s.height,domain='rgb') for s in samples]
    variant=dict(resolve_dali_variant('d2'),num_threads=16,prefetch_queue_depth=2,seed=11997733,resize_antialias=False)
    if args.pipeline=='ffcv':
        from ffcv.writer import DatasetWriter
        from ffcv.fields import RGBImageField,IntField
        from fastlens.benchmarks.baselines.ffcv.prepare_p0_ffcv import WindowDataset
        import tempfile
        temporary=tempfile.TemporaryDirectory(prefix='fastlens-ffcv-tail-',dir=Path.home()/'tmp')
        root=Path(temporary.name)
        (root/'samples.json').write_text(json.dumps(entries))
        DatasetWriter(str(root/'train_window.beton'),{'image':RGBImageField(write_mode='raw'),'label':IntField(),'ordinal':IntField()},num_workers=1).from_indexed_dataset(WindowDataset(entries))
        adapter=FfcvWindow(samples,device,decisions,root=root)
    elif args.pipeline=='coordl':
        adapter=CoorDLWindow(samples,device,args.output,decisions)
    else:
        cls=DaliTrainingAdapter if args.pipeline=='dali' else PlannedRgbWindow
        adapter=cls(samples,batch_size=64,workers=16,device=device,config={'phase':'train','dali':variant})
        adapter.begin([SampleIdentity(2,i,s.logical_sample_id) for i,s in enumerate(samples)],decisions)
    offset=0;counts=[];ids=[]
    for n in (64,64,15):
        batch=adapter.next_batch()
        assert batch.inputs[0].shape==(n,3,224,224) and batch.labels.shape==(n,)
        check_rgb_input(batch.inputs[0],batch.labels,batch.identities,samples[offset:offset+n],decisions[offset:offset+n],
            pipeline=args.pipeline,device=device,dali_config=variant,output=args.output/f'batch_{offset}')
        counts.append(n);ids.extend(i.logical_sample_id for i in batch.identities);offset+=n
    assert ids==[s.logical_sample_id for s in samples]
    adapter.close()
    if args.pipeline=='ffcv':
        del adapter
        temporary.cleanup()
    result=dict(passed=True,pipeline=args.pipeline,samples=offset,batch_sizes=counts,tail_input_verified=True,sample_order_verified=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
