"""Query full-Beton residency by file range without reading payload pages."""
import argparse
import csv
import ctypes
import datetime
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    page = os.sysconf('SC_PAGE_SIZE')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    rows = []
    started = datetime.datetime.now().astimezone().isoformat()
    with args.source.open('rb') as source:
        size = os.fstat(source.fileno()).st_size
        address = libc.mmap(None, size, 0, 1, source.fileno(), 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), 'mmap')
        try:
            for offset in range(0, size, 1 << 30):
                length = min(1 << 30, size - offset)
                pages = (length + page - 1) // page
                vector = (ctypes.c_ubyte * pages)()
                if libc.mincore(address + offset, length, vector):
                    raise OSError(ctypes.get_errno(), 'mincore')
                resident = int((np.frombuffer(vector, dtype=np.uint8) & 1).sum())
                rows.append(dict(offset_bytes=offset, length_bytes=length, pages=pages, resident_pages=resident))
        finally:
            libc.munmap(address, size)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(source=str(args.source), started=started,
                    completed=datetime.datetime.now().astimezone().isoformat(),
                    scope='Read-only sequential mincore observations, not one atomic snapshot or a performance result.',
                    pages=sum(r['pages'] for r in rows), resident_pages=sum(r['resident_pages'] for r in rows))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
