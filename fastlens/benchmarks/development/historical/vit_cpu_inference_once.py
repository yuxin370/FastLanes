"""Use the existing ViT CPU-DCT evaluator with explicit resident warm-up."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from fastlens.benchmarks.system_dct_major import pipeline as P

parser = argparse.ArgumentParser()
parser.add_argument('--contract', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
original = P._prime_model_for_cold_start


def prime(*values, **kwargs):
    for _ in range(20):
        original(*values, **kwargs)


P._prime_model_for_cold_start = prime
P.run_pipeline('rgbnomore', args.contract, args.output)
