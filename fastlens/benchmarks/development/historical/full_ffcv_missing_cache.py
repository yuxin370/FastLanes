"""Fill nonresident Beton ranges; retain the original full-epoch entry."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
MEMORY_NODE = int(os.environ.get('FASTLENS_FFCV_CACHE_MEMORY_NODE', '0'))


def query_cache(path):
    """Count the same mincore bits without a full-corpus scratch allocation."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = os.sysconf('SC_PAGE_SIZE')
    resident = 0
    with path.open('rb') as source:
        size = os.fstat(source.fileno()).st_size
        address = libc.mmap(None, size, 0, 1, source.fileno(), 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), 'mmap')
        try:
            for offset in range(0, size, 1 << 26):
                length = min(1 << 26, size - offset)
                vector = (ctypes.c_ubyte * ((length + page - 1) // page))()
                if libc.mincore(address + offset, length, vector):
                    raise OSError(ctypes.get_errno(), 'mincore')
                resident += int(np.count_nonzero(np.frombuffer(vector, dtype=np.uint8) & 1))
        finally:
            libc.munmap(address, size)
    pages = (size + page - 1) // page
    return dict(files=1, bytes=size, pages=pages, resident_pages=resident,
                resident_fraction=resident / pages)


def fill_missing(path):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = os.sysconf('SC_PAGE_SIZE')
    issued = 0
    with path.open('rb') as source:
        size = os.fstat(source.fileno()).st_size
        address = libc.mmap(None, size, 0, 1, source.fileno(), 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), 'mmap')
        try:
            for offset in range(0, size, 1 << 26):
                length = min(1 << 26, size - offset)
                vector = (ctypes.c_ubyte * ((length + page - 1) // page))()
                if libc.mincore(address + offset, length, vector):
                    raise OSError(ctypes.get_errno(), 'mincore')
                missing = np.flatnonzero((np.frombuffer(vector, dtype=np.uint8) & 1) == 0)
                # Contiguous missing pages are read together; cached ranges are skipped.
                if not len(missing):
                    continue
                cuts = np.flatnonzero(np.diff(missing) != 1) + 1
                starts = missing[np.r_[0, cuts]]
                ends = missing[np.r_[cuts - 1, len(missing) - 1]] + 1
                for first, last in zip(starts, ends):
                    position = offset + int(first) * page
                    end = min(offset + int(last) * page, size)
                    while position < end:
                        block = os.pread(source.fileno(), min(1 << 20, end - position), position)
                        if not block:
                            raise EOFError('Beton truncated during excluded prewarming')
                        position += len(block)
                        issued += len(block)
        finally:
            libc.munmap(address, size)
    return issued


def cache_pages(paths, warm=False):
    if len(paths) != 1 or Path(paths[0]).suffix != '.beton':
        raise ValueError('Only the full-training Beton is supported')
    command = ['numactl', '--cpunodebind=1', f'--membind={MEMORY_NODE}', sys.executable,
               str(Path(__file__).resolve()), '--cache-helper', str(paths[0])]
    if warm:
        command.append('--warm')
    result = json.loads(subprocess.check_output(command, text=True, cwd=ROOT))
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    with (output / 'cache_queries.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(command=command, result=result)) + '\n')
    return result


def main():
    from fastlens.benchmarks.profiling import input_motivation
    if sys.argv[1] == '--cache-helper':
        path = Path(sys.argv[2])
        passes = []
        if '--warm' in sys.argv:
            # A previous one-pass fill lost pages before its final query.
            # Retry missing ranges only; never relax the residency requirement.
            for _ in range(3):
                issued = fill_missing(path)
                result = query_cache(path)
                passes.append(dict(result, issued_read_bytes=issued))
                if result['resident_pages'] == result['pages']:
                    break
        else:
            result = query_cache(path)
        result['prewarm_issued_read_bytes'] = sum(p['issued_read_bytes'] for p in passes)
        result['prewarm_passes'] = passes
        print(json.dumps(result))
        return
    if sys.argv[sys.argv.index('--pipeline') + 1] != 'ffcv':
        raise ValueError('Only the FFCV full-epoch retry uses this entry')
    from fastlens.benchmarks.development.historical.frozen_rgb_epoch_20261006 import SOURCE
    source = SOURCE.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    assert sha == '1f8166530d77bcbfdfd94e7d6c8f7e898ad01c9d530646a4d18facf596221cdf'
    module = types.ModuleType('fastlens.benchmarks.paper.shared.training_window_rgb')
    module.__file__ = str(ROOT / 'fastlens/benchmarks/paper/shared/training_window_rgb.py')
    sys.modules[module.__name__] = module
    exec(compile(source, str(SOURCE), 'exec'), module.__dict__)
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'executed_entry.json').write_text(json.dumps(dict(source=str(SOURCE), sha256=sha,
        model_stream_policy='Original default stream', shared_entry_replaced=False), indent=2) + '\n')
    (output / 'cache_placement.json').write_text(json.dumps(dict(
        helper_cpu_node=1, helper_memory_node=MEMORY_NODE, training_cpu_node=0, training_memory_node=0,
        policy='Read missing page ranges only; subsequent full mincore gate remains unchanged.',
        notes='Excluded prewarming only. Existing pages are not migrated; placement differs from older runs. Not isolated GPU scaling.'), indent=2) + '\n')
    input_motivation.cache_pages = cache_pages
    module.main()


if __name__ == '__main__':
    main()
