#!/usr/bin/env python3
"""Prime resident Swin tensors outside the existing complete-pass timer."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from fastlens.benchmarks.baselines.system_rgbnomore.inference import pipeline as P


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pipeline',required=True,choices=['pytorch','dali','galp','rgbnomore'])
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    domain='rgb' if args.pipeline in ('pytorch','dali') else 'dct'
    original=getattr(P,'_build_'+domain+'_model')
    def build(contract,device):
        model=original(contract,device)
        batch=contract['execution']['batch_size']
        if domain=='rgb':
            shapes=[(3,*contract['preprocess']['rgb']['crop_size'])]
        else:
            shapes=[contract['preprocess']['dct'][k] for k in ('y_shape','cbcr_shape')]
        inputs=tuple(torch.zeros((batch,*shape),device=device) for shape in shapes)
        for _ in range(20):
            logits=P._forward(model,inputs,contract,device)
        assert torch.isfinite(logits).all()
        torch.cuda.synchronize()
        (args.output.parent/'model_warmup.json').write_text(json.dumps(dict(batches=20,
            samples_excluded=20*batch,input_shapes=[list(x.shape) for x in inputs],
            timing='before run_pipeline repeat timer',finite=True),indent=2)+'\n')
        return model
    setattr(P,'_build_'+domain+'_model',build)
    P.run_pipeline(args.pipeline,args.contract,args.output)


if __name__=='__main__':
    main()
