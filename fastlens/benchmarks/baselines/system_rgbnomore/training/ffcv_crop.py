"""Keyed training crop/flip on raw RGB or JPEG-backed FFCV images."""
import numpy as np
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
from ffcv.pipeline.compiler import Compiler
from ffcv.libffcv import imdecode
import math


def bilinear_crop(source, top, left, height, width, destination):
    """Half-pixel bilinear sampling without the native FFCV AREA resize."""
    for oy in range(224):
        y = max(0., (oy + .5) * height / 224 - .5)
        y0 = int(math.floor(y)); y1 = min(y0 + 1, height - 1)
        fy = y - y0
        for ox in range(224):
            x = max(0., (ox + .5) * width / 224 - .5)
            x0 = int(math.floor(x)); x1 = min(x0 + 1, width - 1)
            fx = x - x0
            for c in range(3):
                a = source[top+y0,left+x0,c] * (1-fx) + source[top+y0,left+x1,c] * fx
                b = source[top+y1,left+x0,c] * (1-fx) + source[top+y1,left+x1,c] * fx
                destination[oy,ox,c] = min(255, max(0, int(a*(1-fy) + b*fy + .5)))


class PlannedCropDecoder(RandomResizedCropRGBImageDecoder):
    def __init__(self, decisions):
        super().__init__((224, 224))
        self.crops = np.array([(d.crop_y, d.crop_x, d.crop_height, d.crop_width,
                               d.horizontal_flip) for d in decisions], dtype=np.int64)

    def generate_code(self):
        read = self.memory_read
        iterator = Compiler.get_iterator()
        resize = Compiler.compile(bilinear_crop)
        decode_jpeg = Compiler.compile(imdecode)
        crops = self.crops

        def decode(indices, storage, metadata, storage_state):
            destination, scratch = storage
            for dst in iterator(len(indices)):
                src = indices[dst]
                field = metadata[src]
                data = read(field['data_ptr'], storage_state)
                height, width = np.uint32(field['height']), np.uint32(field['width'])
                if field['mode'] == 0:
                    decode_jpeg(data, scratch[dst], height, width, height, width,
                                0, 0, 1, 1, False, False)
                    pixels = scratch[dst].reshape(-1)[:3 * height * width].reshape(height, width, 3)
                else:
                    pixels = data.reshape(height, width, 3)
                y, x, h, w, flip = crops[src]
                resize(pixels, y, x, h, w, destination[dst])
                if flip:
                    destination[dst] = destination[dst, :, ::-1, :].copy()
            return destination[:len(indices)]
        decode.is_parallel = True
        return decode
