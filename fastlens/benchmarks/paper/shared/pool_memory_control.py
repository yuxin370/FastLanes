"""Load one postprocessing implementation with microbatch or pool-sized allocations."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'build/fastlens/torch')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--control-library', type=Path, required=True)
    parser.add_argument('--allocation', choices=('microbatch', 'pool'), required=True)
    parser.add_argument('--input-check-output', type=Path)
    args, remaining = parser.parse_known_args()
    import torch
    if args.allocation == 'pool':
        os.environ['FASTLENS_POOL_WORKSPACE_CONTROL'] = '1'
    else:
        os.environ.pop('FASTLENS_POOL_WORKSPACE_CONTROL', None)
    # The control's native pool references resolve after the original library is loaded globally.
    library = ctypes.CDLL(str(args.control_library.resolve()), mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | os.RTLD_GLOBAL)
    import _fastlens_direct_dct
    sys.setdlopenflags(flags)
    if args.input_check_output:
        from fastlens.torch.experimental import DirectDctPlsPipeline
        source = ROOT / 'fastlens/data/compressed/imagenet512_train_block_major_premixed'
        contract = json.loads((source / 'materialization_contract.json').read_text())
        torch.set_num_threads(8)
        results = []
        for micro in (64, 63):
            pipeline = DirectDctPlsPipeline(source / 'dct/manifest.bin', source / 'ordered_mapping.csv',
                training_seed=11997733, expected_mapping_sha256=contract['ordered_mapping_sha256'],
                segments_per_pool=4, microbatch_images=micro, module_path=ROOT / 'build/fastlens/torch').start_epoch(2)
            pool = pipeline.next_pool()
            rows = []
            for batch in pool:
                tensors = batch.tensors
                rows.append(dict(ids=batch.global_image_ids,
                    tensors=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in tensors]))
                del tensors, batch
            results.append(dict(microbatch=micro, images=pool.image_count, rows=rows))
            pool.retire()
            del pool
            pipeline.reclaim_finished_pools()
            pipeline.close()
        args.input_check_output.write_text(json.dumps(results) + '\n')
    else:
        sys.argv = ['fastlens/benchmarks/paper/shared/training_window_rgb.py', *remaining]
        runpy.run_path(str(ROOT / 'fastlens/benchmarks/paper/shared/training_window_rgb.py'), run_name='__main__')
    library.pool_allocation_control_calls.restype = ctypes.c_uint64
    calls = library.pool_allocation_control_calls()
    assert calls > 0, 'Allocation control did not intercept postprocessing'
    output = args.input_check_output.parent if args.input_check_output else Path(remaining[remaining.index('--output') + 1])
    (output / 'allocation_control.json').write_text(json.dumps(dict(
        allocation=args.allocation, postprocess_calls=calls, library=str(args.control_library.resolve())), indent=2) + '\n')


if __name__ == '__main__':
    main()
