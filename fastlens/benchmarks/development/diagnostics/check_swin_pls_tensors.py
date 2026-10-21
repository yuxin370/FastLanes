#!/usr/bin/env python3
"""Non-performance B6 native/CPU-DCT crop+RandAugment+Mixup tensor gate."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=['vitti','swinv2'],default='swinv2')
    parser.add_argument('--epoch',type=int,default=2)
    parser.add_argument('--augmentation',choices=['full','basic'],default='full')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from PIL import Image
    from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import NativeWindow
    from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingSample,RgbNoMoreTrainingAdapter
    from fastlens.benchmarks.training_pls.published_augmentation import (
        published_training_augmentation,apply_published_randaugment,apply_published_mixup)
    from fastlens.benchmarks.training_pls.sanity_check import _tensor_difference
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    native=NativeWindow(args.model,args.augmentation)
    native.pipeline.start_epoch(args.epoch)
    observations=[]
    update_check=None
    for microbatch_index in range(2):
        batch=native.next_batch()
        samples=[]
        decisions=[]
        for position in batch.native_stats_source.global_image_ids:
            record=native.records[position]
            path=Path(record['source_path'])
            with Image.open(path) as source:
                width,height=source.size
            sample=TrainingSample(record['logical_sample_id'],path,int(record['label']),width,height,int(record['galp_image_id']))
            samples.append(sample)
            decisions.append(published_training_augmentation(training_seed=11997733,epoch=args.epoch,
                logical_sample_id=sample.logical_sample_id,virtual_pls_id=int(record['virtual_pls_id']),
                crop_policy='per-pls',source_width=width,source_height=height).decision)
        reference=RgbNoMoreTrainingAdapter(samples,batch_size=64,workers=0,device=device,
            config={'rgbnomore_root':str(Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more')))})
        reference.begin(batch.identities,decisions)
        ref=reference.next_batch()
        inputs=tuple(x.to(device) for x in ref.inputs)
        labels=ref.labels.to(device)
        ids=[s.logical_sample_id for s in samples]
        if args.augmentation=='full':
            inputs,_=apply_published_randaugment(inputs,training_seed=11997733,epoch=args.epoch,
                logical_sample_ids=ids,rgbnomore_root=(Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more'))))
            inputs,targets,_=apply_published_mixup(inputs,labels,training_seed=11997733,epoch=args.epoch,
                microbatch_index=microbatch_index,alpha=.2,classes=1000)
        else:
            inputs=tuple(((x*1020.-4.).round().clamp(-1024,1016)+4.)/1020. for x in inputs)
            targets=labels
            assert batch.labels.dtype==torch.int64 and batch.labels.ndim==1
            torch.testing.assert_close(batch.labels,targets,rtol=0,atol=0)
        if update_check is None:
            from fastlens.benchmarks.paper.shared.training_window_rgb import build_window_model
            from fastlens.benchmarks.development.diagnostics.check_training_update import check_update
            model,_,_=build_window_model(args.model,'dct',device)
            update_check=check_update(model,inputs,targets,batch.inputs,batch.labels)
            del model
        differences=[_tensor_difference(a,b,atol=1/1020+1e-7,rtol=1e-4) for a,b in zip(batch.inputs,inputs)]
        target_difference=_tensor_difference(batch.labels.float(),targets.float(),atol=1e-7,rtol=1e-6)
        observations.append(dict(microbatch=microbatch_index,sample_ids=ids,tensors=differences,targets=target_difference))
        reference.close()
        del batch,ref,inputs,targets,labels
    native.close()
    passed=all(all(d['allclose'] for d in row['tensors']) and row['targets']['allclose'] for row in observations)
    result=dict(gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        samples=128,passed=passed,augmentation=args.augmentation,observations=observations,update_check=update_check,
        scope=f'Non-performance check, B6 epoch index {args.epoch} first two microbatches; inherited one-coefficient-unit input tolerance; no convergence claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
