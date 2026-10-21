"""Check matched CNN/eFUN B6 samples, tensors, targets, loss and updates before timing."""
import argparse
import json
import os
from pathlib import Path
import sys
import cv2

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(os.environ.get('FASTLENS_TRAINING_NATIVE_DIR',ROOT/'build/fastlens/torch')))
os.environ.setdefault('DCTNET_PROFILE','mobilenet24')
if os.environ['DCTNET_PROFILE']=='efun':
    import fastlens.benchmarks.dct_models as package
    from fastlens.benchmarks.dct_models import efun_backend
    sys.modules['fastlens.benchmarks.dct_models.backend']=efun_backend
    package.backend=efun_backend
import torch
from fastlens.benchmarks.dct_models.training_pls import channels
from fastlens.benchmarks.dct_models.training_jpeg import JpegTrainingPipeline
from fastlens.torch.experimental import DirectDctPlsPipeline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--augmentation',choices=['full','basic'],default='full')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    root=ROOT/'fastlens/data/compressed/imagenet512_train_block_major_premixed'
    material=json.loads((root/'materialization_contract.json').read_text())
    mapping=Path(material['ordered_mapping'])
    from fastlens.benchmarks.dct_models import backend as B
    cpu=JpegTrainingPipeline(mapping,output_channels=channels(),grid=B.GRID,workers=8,microbatch=64,pool_images=4096,condition='B6')
    if args.augmentation=='basic':
        class BasicAugmentation:
            input_channels=[(int(c),int(f)) for c,f,_,_ in channels()]
            def apply(self,x,logical_ids,labels,seed,epoch,microbatch):
                means=B.MEAN.to(x.device).reshape(1,-1,1,1)
                stds=B.STD.to(x.device).reshape(1,-1,1,1)
                return (x.round().clamp(-1024,1016)-means)/stds,torch.tensor(labels,device=x.device,dtype=torch.int64)
        cpu.augment=BasicAugmentation()
    native=DirectDctPlsPipeline(root/'dct/manifest.bin',mapping,training_seed=11997733,
        expected_mapping_sha256=material['ordered_mapping_sha256'],output_grid_size=B.GRID,output_channels=channels(),
        enable_randaugment=args.augmentation=='full',enable_mixup=args.augmentation=='full',
        transform_blocks_per_launch=32768,transform_ctas_per_launch=512)
    cpu.start_epoch(0)
    native.start_epoch(0)
    observations=[]
    update_check=None
    try:
        cp,np=cpu.next_pool(),native.next_pool()
        ci,ni=iter(cp),iter(np)
        for _ in range(2):
            c,n=next(ci),next(ni)
            assert c.global_image_ids==list(n.global_image_ids)
            if args.augmentation=='basic':
                assert n.targets.dtype==torch.int64 and n.targets.ndim==1
            torch.testing.assert_close(c.targets,n.targets,rtol=0,atol=0)
            # Predetermined numerical input gate, before any new performance run.
            torch.testing.assert_close(c.projected,n.projected,rtol=1e-5,atol=1e-5)
            if update_check is None:
                from fastlens.benchmarks.paper.shared.training_window_rgb import build_window_model
                from fastlens.benchmarks.development.diagnostics.check_training_update import check_update
                model,_,_=build_window_model(B.PROFILE,'dct',torch.device('cuda:0'))
                update_check=check_update(model,(c.projected,),c.targets,(n.projected,),n.targets)
                del model
            observations.append(dict(sample_ids=[cpu.records[i][0] for i in c.global_image_ids],
                input_max_abs=float((c.projected-n.projected).abs().max()),targets_exact=True))
            del c,n
        np.retire()
        del ci,ni,cp,np
    finally:
        cpu.close()
        native.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,samples=128,gpu=torch.cuda.get_device_name(),
        condition='B6',epoch=0,augmentation=args.augmentation,input_atol=1e-5,input_rtol=1e-5,targets_atol=0,targets_rtol=0,
        observations=observations,update_check=update_check),indent=2)+'\n')


if __name__=='__main__': main()
