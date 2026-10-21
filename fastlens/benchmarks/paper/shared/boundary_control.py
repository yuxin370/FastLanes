"""Run the existing CNN control with experimental staging or shared read-completion bandwidth.

The simulator paces .fls reads before their buffers become available to consumers.
All reader threads/files share one rate, with no accumulated idle credit. It models
application-read completion bandwidth, not a remote protocol, RTT, or cold disk.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'build/fastlens/torch')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--control-library', type=Path, required=True)
    parser.add_argument('--staging', choices=('compact', 'source'), default='compact')
    parser.add_argument('--read-mbps', type=float, default=0, help='decimal MB/s; zero means unlimited')
    args, remaining = parser.parse_known_args()
    if args.read_mbps < 0:
        parser.error('read-mbps must be nonnegative')
    import torch
    library_path, = (ROOT / 'build/fastlens/torch').glob('_fastlens_direct_dct*.so')
    os.environ['FASTLENS_NATIVE_LIBRARY'] = str(library_path)
    if args.staging == 'source':
        os.environ['FASTLENS_SOURCE_LAYOUT_CONTROL'] = '1'
    else:
        os.environ.pop('FASTLENS_SOURCE_LAYOUT_CONTROL', None)
    control = ctypes.CDLL(str(args.control_library.resolve()), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    control.input_boundary_begin.argtypes = [ctypes.c_double]
    control.input_boundary_begin.restype = None
    control.input_boundary_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 3 + [ctypes.POINTER(ctypes.c_double)]
    control.input_boundary_stats.restype = None
    import _fastlens_direct_dct as native
    original = native.DirectDctReader

    def reader(*reader_args, **kwargs):
        control.input_boundary_begin(args.read_mbps)
        return original(*reader_args, **kwargs)

    native.DirectDctReader = reader
    from fastlens.benchmarks.paper.shared import cnn_control
    sys.argv = ['fastlens/benchmarks/paper/shared/cnn_control.py', *remaining]
    cnn_control.main()
    counts = [ctypes.c_uint64() for _ in range(3)]
    span = ctypes.c_double()
    control.input_boundary_stats(*(ctypes.byref(x) for x in counts), ctypes.byref(span))
    output = Path(remaining[remaining.index('--output-dir') + 1])
    path, = output.glob('N_*.json')
    result = json.loads(path.read_text())
    result['input_boundary_control'] = dict(staging=args.staging, read_limit_MB_s=args.read_mbps,
        read_bytes=counts[0].value, read_calls=counts[1].value, compact_entry_calls=counts[2].value,
        read_completion_span_s=span.value, control_library=str(args.control_library.resolve()),
        native_library=str(library_path), simulation='shared application-read completion bandwidth; warm source pages; no RTT')
    assert counts[0].value > 0, 'Read interposition was not active'
    if result['native_options']['crop_execution_mode'] != 'full-source-decode':
        assert counts[2].value > 0, 'Compact staging interposition was not active'
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
