"""Binary IPC with the experiment's native FASTLENS bridge."""
import struct
import os
import json
import subprocess
from pathlib import Path

import numpy as np

BRIDGE = Path(os.environ.get("FASTLENS_DCT_STORAGE", Path(__file__).resolve().parents[3] / "build/fastlens/benchmarks/fastlens_dct_storage"))


def pack_image(path, coefficients, tables):
    name = str(path).encode()
    parts = [struct.pack("<I", len(name)), name]
    for c, q in zip(coefficients, tables):
        parts.extend([np.asarray(q, dtype="<u2").tobytes(), np.asarray(c, dtype="<i2").tobytes()])
    return b"".join(parts)


class Reader:
    def __init__(self, manifest, *, command="read-manifest"):
        self.process = subprocess.Popen([str(BRIDGE), command, str(manifest)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self, image_id):
        self.process.stdin.write(struct.pack("<I", image_id))
        self.process.stdin.flush()
        def read(n):
            data = self.process.stdout.read(n)
            if len(data) != n:
                raise RuntimeError(self.process.stderr.read().decode())
            return data
        count, = struct.unpack("<I", read(4))
        coefficients, tables = [], []
        for expected in range(count):
            component, height, width = struct.unpack("<III", read(12))
            if component != expected:
                raise ValueError("unexpected FASTLENS component order")
            tables.append(np.frombuffer(read(128), dtype="<u2").copy())
            coefficients.append(np.frombuffer(read(height * width * 128), dtype="<i2")
                                .reshape(height, width, 64).copy())
        self.last_stats = json.loads(self.process.stderr.readline())
        return coefficients, tables

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()
        if self.process.wait() != 0:
            raise RuntimeError("native FASTLENS reader failed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
