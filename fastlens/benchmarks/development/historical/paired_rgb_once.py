#!/usr/bin/env python3
"""Reuse the working CoorDL/FFCV readers with the registered RGB family checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--profile',required=True,choices=['swinv2','mobilenet24','resnet24','efun'])
parser.add_argument('--pipeline',required=True,choices=['ffcv','coordl'])
parser.add_argument('--contract',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.pipeline=='ffcv':
    from ffcv.loader import Loader
else:
    from nvidia.dali import backend
import torch
from fastlens.benchmarks.system_dct_major import pipeline as P
contract=json.loads(args.contract.read_text())
cnn=args.profile!='swinv2'
torch.set_num_threads(8 if cnn else 1)
if cnn:
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.profile=='efun':
        import fastlens.benchmarks.dct_models as package
        from fastlens.benchmarks.dct_models import efun_backend
        sys.modules['fastlens.benchmarks.dct_models.backend']=efun_backend
        package.backend=efun_backend
    from fastlens.benchmarks.dct_models import rgb
    assert rgb.CHECKPOINT.resolve()==Path(contract['models']['rgb']['checkpoint']).resolve()
    def build(**kwargs): return rgb.model().to(kwargs['device'])
    base=P.CoorDLAdapter if args.pipeline=='coordl' else P.FfcvAdapter
    class Normalized(base):
        def load(self,expected):
            batch=super().load(expected)
            x=batch.inputs[0].add(1).mul(.5)
            mean=x.new_tensor(rgb.MEAN)[None,:,None,None]
            std=x.new_tensor(rgb.STD)[None,:,None,None]
            batch.inputs=((x-mean)/std,)
            return batch
    setattr(P,'CoorDLAdapter' if args.pipeline=='coordl' else 'FfcvAdapter',Normalized)
else:
    from fastlens.benchmarks.baselines.system_rgbnomore.inference.model_factory import build_rgb_model,SWINV2_T_MODEL_ID
    def build(**kwargs):
        return build_rgb_model(kwargs['rgbnomore_root'],kwargs['checkpoint'],kwargs['device'],model_id=SWINV2_T_MODEL_ID)
P.build_workload_model=build
def prime(model,domain,batch_size,device,expected_width):
    x=torch.zeros(batch_size,3,*contract['preprocess']['rgb']['crop_size'],device=device)
    for _ in range(20): P._forward(model,(x,),expected_width)
    torch.cuda.synchronize()
P._prime_model_for_cold_start=prime
P.run_pipeline(args.pipeline,args.contract,args.output)
