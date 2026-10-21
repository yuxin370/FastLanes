"""Run a Section 4.2 workload with spatial selection enabled or disabled."""
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


class TrainingVectorCounts:
    """Count each executed pool once and record which window consumes it."""
    def __init__(self):
        self.pools = {}

    def observe(self, pool, measured, images):
        key = pool.pool_index
        if key not in self.pools:
            stats = pool.execution_stats
            self.pools[key] = dict(pool_index=key, pool_images=pool.image_count,
                planned_vectors=int(stats['planned_selected_vector_count']),
                actual_vectors=int(stats['selected_vector_count']),
                full_vectors=int(stats['full_vector_count']),
                warmup_images=0, measured_images=0)
        self.pools[key]['measured_images' if measured else 'warmup_images'] += images

    def result(self):
        pools = list(self.pools.values())
        measured = [p for p in pools if p['measured_images']]
        return dict(pools=pools, measured_pool_count=len(measured),
            measured_images=sum(p['measured_images'] for p in measured),
            boundary_pool_count=sum(p['warmup_images'] > 0 or p['measured_images'] < p['pool_images']
                                    for p in measured),
            **{key: sum(p[key] for p in measured)
               for key in ('planned_vectors', 'actual_vectors', 'full_vectors')},
            scope='Whole executed pools serving the measured window, counted once; includes any boundary-pool rows, excludes warmup-only pools and unconsumed prefetched pools')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--arm', choices=['selected', 'full'], required=True)
    parser.add_argument('--command', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resource', choices=['memory', 'profile', 'cold'])
    parser.add_argument('--input-paths', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--native-dir', type=Path, default=ROOT / 'build/fastlens/torch')
    parser.add_argument('--training-vectors', action='store_true')
    args = parser.parse_args()
    import torch
    native_path, = args.native_dir.glob('_fastlens_direct_dct.*.so')
    sys.path.insert(0, str(args.native_dir.resolve()))
    os.environ['FASTLENS_NATIVE_LIBRARY'] = str(native_path)
    library = ctypes.CDLL(str(args.library.resolve()), mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | os.RTLD_GLOBAL)
    import _fastlens_direct_dct
    sys.setdlopenflags(flags)
    library.crop_control_begin.argtypes = [ctypes.c_int]
    library.crop_control_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 3
    library.crop_control_begin(args.arm == 'full')
    command = json.loads(args.command.read_text())
    training = command[0] == 'fastlens/benchmarks/paper/shared/training_window_rgb.py'
    input_hashes = []
    if args.training_vectors:
        assert training
        global training_vectors
        training_vectors = TrainingVectorCounts()
        sys.modules['fastlens.benchmarks.development.diagnostics.crop_workload_control'] = sys.modules[__name__]
    if args.check_inputs and not training:
        depth = 0

        def before(module, inputs):
            nonlocal depth
            if depth == 0 and len(input_hashes) < 40:
                tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
                if tensors:
                    input_hashes.append([hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
                                         for x in tensors])
            depth += 1

        def after(module, inputs, output):
            nonlocal depth
            depth -= 1

        torch.nn.modules.module.register_module_forward_pre_hook(before)
        torch.nn.modules.module.register_module_forward_hook(after)

    def training_source(source):
        if args.training_vectors:
            anchor = '            if measure:\n                consumed += int(target.shape[0])'
            assert source.count(anchor) == 1
            source = source.replace(anchor,
                "            __import__('fastlens.benchmarks.development.diagnostics.crop_workload_control', fromlist=['training_vectors']).training_vectors.observe(adapter.pool, measure, int(target.shape[0]))\n" + anchor)
        anchor = '    result = dict(timestamp=timestamp,'
        assert source.count(anchor) == 1
        source = source.replace(anchor,
            "    from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256\n"
            "    final_model_hash = tensor_state_sha256(model.state_dict())\n"
            "    if args.check:\n"
            "        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, args.output/'final_state.pt')\n"
            "    result = dict(final_model_hash=final_model_hash, timestamp=timestamp,")
        anchor = "            with span('model.forward'):"
        assert source.count(anchor) == 1
        return source.replace(anchor,
            "            if args.check:\n"
            "                __crop_input_check(x, target)\n" + anchor)

    def capture_training_inputs(inputs, targets):
        args.output.mkdir(parents=True, exist_ok=True)
        tensors = [t.detach().cpu().clone() for t in (*inputs, targets)]
        torch.save(tensors, args.output / f'input_{len(input_hashes):02d}.pt')
        input_hashes.append([hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for t in tensors])

    if args.resource:
        from fastlens.benchmarks.paper.section_6_3.h2d_traffic import profile_section42 as profile
        original = profile.instrument

        def instrument(source, filename):
            source = original(source, filename)
            if str(filename).endswith('fastlens/benchmarks/paper/shared/training_window_rgb.py'):
                source = training_source(source)
            return source

        profile.instrument = instrument
        sys.argv = ['fastlens/benchmarks/paper/section_6_3/h2d_traffic/profile_section42.py', '--command', str(args.command),
                    '--mode', args.resource, '--output', str(args.output)]
        if args.input_paths:
            sys.argv += ['--input-paths', str(args.input_paths)]
        # profile.main expects its own __name__ alias for injected timer calls.
        profile.main()
    elif training:
        script = ROOT / command[0]
        sys.argv = command
        exec(compile(training_source(script.read_text()), str(script), 'exec'),
             {'__name__': '__main__', '__file__': str(script), '__crop_input_check': capture_training_inputs})
    elif command[0] == '-m':
        sys.argv = command[1:]
        runpy.run_module(command[1], run_name='__main__')
    elif command[0] == '-c':
        sys.argv = ['-c', *command[2:]]
        exec(compile(command[1], '<original workload>', 'exec'), {'__name__': '__main__'})
    else:
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
    counts = [ctypes.c_uint64() for _ in range(3)]
    library.crop_control_stats(*(ctypes.byref(x) for x in counts))
    assert counts[0].value > 0, 'Spatial control did not intercept the native planner'
    if args.arm == 'full':
        assert counts[1].value == counts[2].value, 'Full-source control still selected vectors'
    else:
        assert counts[1].value < counts[2].value, 'Selected control did not reduce vectors'
    record = dict(arm=args.arm, prepare_calls=counts[0].value, selected_vectors=counts[1].value,
                  full_vectors=counts[2].value, model_input_hashes=input_hashes,
                  scope='All prepared requests, including warmup and prefetch beyond the timed window')
    (args.output / 'crop_control.json').write_text(json.dumps(record, indent=2)+'\n')
    if args.training_vectors:
        (args.output / 'training_vectors.json').write_text(json.dumps(training_vectors.result(), indent=2)+'\n')


if __name__ == '__main__':
    main()
