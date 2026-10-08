"""L3 encoding with the paper's 32x32 patches for images below HD resolution."""
import ctypes
import struct

import numpy as np
import torch
from nvidia.dali import plugin_manager

PATCH = 32
PATCHES_PER_SIDE = 512 // PATCH


def load_plugin(path):
    plugin_manager.load_library(str(path))


class Encoder:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.encode_rows = self.library.l3_encode_rows
        self.encode_rows.argtypes = [ctypes.c_void_p] * 4
        self.encode_rows.restype = ctypes.c_int
        self.input = torch.empty((3, 512, 512), dtype=torch.uint8, device="cuda")
        self.packed = torch.empty((3, 512, PATCHES_PER_SIDE, PATCH + 2), dtype=torch.uint8, device="cuda")
        self.lengths = torch.empty((3, 512, PATCHES_PER_SIDE), dtype=torch.uint8, device="cuda")

    def encode(self, rgb):
        if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
            raise ValueError("L3 benchmark encoding requires 512x512 uint8 RGB")
        self.input.copy_(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
        status = self.encode_rows(self.input.data_ptr(), self.packed.data_ptr(), self.lengths.data_ptr(),
                                  torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"L3 encoder CUDA error {status}")
        packed, lengths = self.packed.cpu().numpy(), self.lengths.cpu().numpy()
        patches, sizes = [], []
        for c in range(3):
            for y in range(PATCHES_PER_SIDE):
                for x in range(PATCHES_PER_SIDE):
                    rows = packed[c, y * PATCH:(y + 1) * PATCH, x]
                    count = lengths[c, y * PATCH:(y + 1) * PATCH, x]
                    patch = rows[np.arange(PATCH + 2)[None, :] < count[:, None]].tobytes()
                    patches.append(patch)
                    sizes.append(len(patch))
        return (struct.pack("<4siiB", b"LLL.", 512, 512, PATCH)
                + struct.pack(f"<{3 * PATCHES_PER_SIDE ** 2}H", *sizes) + b"".join(patches))


def decode_batch(blobs, library):
    """Correctness helper; benchmark pipelines keep the decoded images on the GPU."""
    from nvidia.dali import fn, pipeline_def, types
    load_plugin(library)

    @pipeline_def
    def pipeline():
        encoded = fn.external_source(source=lambda: [np.frombuffer(b, dtype=np.uint8) for b in blobs],
                                     batch=True, dtype=types.UINT8, ndim=1)
        return fn.l3_decoder(encoded, device="mixed")

    pipe = pipeline(batch_size=len(blobs), num_threads=2, device_id=0)
    pipe.build()
    return pipe.run()[0].as_cpu().as_array()
