"""Verify new inference configurations using shared source/input/model references."""
import argparse
from pathlib import Path

from . import verify_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu_dct", "ic", "fs"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-evidence", type=Path)
    parser.add_argument("--model-evidence", type=Path, help="reuse a passed same-workload model result, including existing gates")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend != "cpu_dct" and args.root is None:
        parser.error("IC/FS requires --root")
    result = verify_inference(args.config, args.backend, args.root, args.output,
        cache_dir=args.cache_dir, device=args.device, source_evidence=args.source_evidence,
        model_evidence=args.model_evidence)
    print(f"Passed: {result['occurrence_count']} inference samples; reuse={result['reuse']}. Evidence: {args.output}")


if __name__ == "__main__":
    main()
