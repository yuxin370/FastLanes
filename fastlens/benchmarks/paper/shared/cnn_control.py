#!/usr/bin/env python3
"""Matched CNN controls using the existing reader/model, with an explicit source crop."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--crop-size', type=int, choices=(512, 448, 352, 256, 160), default=448)
    parser.add_argument('--source-coefficients', type=int, choices=(64, 48, 32, 24, 16, 8), default=64,
                        help='raw zigzag prefix before resize; hold fixed for causal controls')
    parser.add_argument('--read-amplification-cap', type=float,
                        help='override the B6 bounded range-merge cap for a matched I/O experiment')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--diagnostic-window', type=Path,
                        help='enable a separate full-pass CUPTI/NVTX capture and write its scope')
    args, remaining = parser.parse_known_args()
    from fastlens.benchmarks.dct_models import online_crop
    from fastlens.benchmarks.system_dct_major.common import resolve_coefficient_selection

    border = (512 - args.crop_size) // 2
    crop = [border, border, args.crop_size, args.crop_size]
    online_crop.CROP = crop
    selection = resolve_coefficient_selection(f'first:{args.source_coefficients}')
    mask = torch.zeros(64)
    mask[selection['resolved_natural_indices']] = 1
    original_options = online_crop.source_options

    def source_options(options, execution_mode):
        options = original_options(options, execution_mode)
        options['dct_coeffs'] = f'first:{args.source_coefficients}'
        options['grid_transform']['require_all_coefficients'] = args.source_coefficients == 64
        return options

    online_crop.source_options = source_options

    class CropReference(online_crop.SourceReference):
        def __call__(self, coefficients, tables):
            components = []
            for component, (values, table) in enumerate(zip(coefficients, tables)):
                block_pixels = 8 if component == 0 else 16
                start = border // block_pixels
                blocks = args.crop_size // block_pixels
                raw = torch.as_tensor(values).float() * mask
                raw *= torch.as_tensor(table).reshape(64)
                raw = raw[start:start + blocks, start:start + blocks].reshape(1, blocks, blocks, 8, 8)
                components.append(self.resize(raw, self.grid, dtype_out=torch.float32, conv_mxs={})
                                  .reshape(self.grid, self.grid, 64))
            return components

    online_crop.SourceReference = CropReference
    if os.environ.get('DCTNET_PROFILE') == 'efun':
        import fastlens.benchmarks.dct_models as package
        from fastlens.benchmarks.dct_models import efun_backend
        sys.modules['fastlens.benchmarks.dct_models.backend'] = efun_backend
        package.backend = efun_backend
    from fastlens.benchmarks.dct_models import evaluate_shards
    if args.read_amplification_cap is not None:
        original_runtime = evaluate_shards.apply_b6_runtime

        def apply_runtime(options):
            original_runtime(options)
            options['bounded_read_amplification_cap'] = args.read_amplification_cap
            return options

        evaluate_shards.apply_b6_runtime = apply_runtime
    window = dict(crop=crop, retained_fraction=(args.crop_size / 512) ** 2,
                  source_coefficients=args.source_coefficients,
                  mask_stage='raw coefficients before dequantization/resize',
                  profiler_enabled=bool(args.diagnostic_window))
    if args.diagnostic_window and ('--verify' in remaining or '--profile' in remaining):
        parser.error('diagnostic capture is separate from correctness and the artifact short capture')
    sys.path.insert(0, str(ROOT / 'build/fastlens/torch'))
    import _fastlens_direct_dct as native
    original_reader = native.DirectDctReader

    def process_io():
        return {key: int(value) for key, value in
                (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}

    def reader(*reader_args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        window.update(start_unix_ns=time.time_ns(), pid=os.getpid(), gpu=torch.cuda.get_device_name(),
                      process_io_start=process_io())
        if args.diagnostic_window:
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('control.measured_pass')
        return original_reader(*reader_args, **kwargs)

    native.DirectDctReader = reader
    if args.diagnostic_window:
        evaluate_shards.Capture.range = lambda self, name: torch.cuda.nvtx.range(name)
    sys.argv = [str(Path(evaluate_shards.__file__)), *remaining, '--output-dir', str(args.output_dir)]
    evaluate_shards.main()
    torch.cuda.synchronize()
    window.update(end_unix_ns=time.time_ns(),
                  process_io_end=process_io(),
                  torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if args.diagnostic_window:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        args.diagnostic_window.write_text(json.dumps(window, indent=2) + '\n')
    result_path, = args.output_dir.glob('N_*.json')
    result = json.loads(result_path.read_text())
    result['control'] = window
    result_path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
