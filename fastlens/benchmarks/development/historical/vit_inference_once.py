#!/usr/bin/env python3
"""Run the existing ViT evaluator with explicit short resident-input warmup."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--pipeline',required=True,choices=['pytorch','dali','ffcv','coordl','dct_major_pushdown'])
parser.add_argument('--contract',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.pipeline=='ffcv':
    from ffcv.loader import Loader
elif args.pipeline=='coordl':
    from nvidia.dali import backend
from fastlens.benchmarks.system_dct_major import pipeline as P
original=P._prime_model_for_cold_start
def prime(*values,**kwargs):
    for _ in range(20): original(*values,**kwargs)
P._prime_model_for_cold_start=prime
P.run_pipeline(args.pipeline,args.contract,args.output)
