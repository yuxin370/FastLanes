"""Run the existing prepared-input driver; optionally capture a separate CUDA diagnostic."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--diagnostic-window', type=Path)
    args, remaining = parser.parse_known_args()
    from fastlens.benchmarks.dct_models import evaluate_shards
    if args.diagnostic_window:
        import torch
        import time
        sys.path.insert(0, str(ROOT / 'build/fastlens/torch'))
        import _fastlens_direct_dct as native
        original = native.DirectDctReader
        window = {}

        def reader(*positional, **keywords):
            torch.cuda.synchronize()
            window['start_unix_ns'] = time.time_ns()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('control.measured_pass')
            return original(*positional, **keywords)

        native.DirectDctReader = reader
        evaluate_shards.Capture.range = lambda self, name: torch.cuda.nvtx.range(name)
    sys.argv = [str(Path(evaluate_shards.__file__)), *remaining]
    evaluate_shards.main()
    if args.diagnostic_window:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        window['end_unix_ns'] = time.time_ns()
        args.diagnostic_window.write_text(json.dumps(window, indent=2) + '\n')


if __name__ == '__main__':
    main()
