"""GPU regression tests; set L3_LIBRARY to the built DALI plugin."""
import os
import unittest
from itertools import cycle

import numpy as np

from .codec import Encoder, decode_batch


class CodecTests(unittest.TestCase):
    def test_lossless_bit_widths_and_large_batch(self):
        library = os.environ["L3_LIBRARY"]
        encoder = Encoder(library)
        rng = np.random.default_rng(47)
        inputs = [rng.integers(0, 1 << bits, (512, 512, 3), dtype=np.uint8) for bits in range(9)]
        inputs += [np.full((512, 512, 3), 255, dtype=np.uint8)]
        blobs = [encoder.encode(pixels) for pixels in inputs]
        # The original DALI patch allocated only 32 streams and indexed by sample.
        order = [i % len(inputs) for i in range(64)]
        actual = decode_batch([blobs[i] for i in order], library)
        for output, index in zip(actual, order):
            np.testing.assert_array_equal(output, inputs[index])
        np.testing.assert_array_equal(decode_batch(blobs[:3], library), np.stack(inputs[:3]))

    def test_reject_wrong_dimensions(self):
        encoder = Encoder(os.environ["L3_LIBRARY"])
        with self.assertRaisesRegex(ValueError, "512x512"):
            encoder.encode(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_reject_previous_patch_geometry(self):
        library = os.environ["L3_LIBRARY"]
        blob = bytearray(Encoder(library).encode(np.zeros((512, 512, 3), dtype=np.uint8)))
        blob[12] = 64
        with self.assertRaisesRegex(RuntimeError, "32x32 patches"):
            decode_batch([bytes(blob)], library)

    def test_prefetched_batches_keep_their_offsets(self):
        from nvidia.dali import fn, pipeline_def, types
        from .codec import load_plugin
        library = os.environ["L3_LIBRARY"]
        load_plugin(library)
        encoder = Encoder(library)
        rng = np.random.default_rng(93)
        inputs = [rng.integers(0, 1 << bits, (512, 512, 3), dtype=np.uint8) for bits in (1, 4, 6, 8)]
        encoded = [np.frombuffer(encoder.encode(x), dtype=np.uint8) for x in inputs]
        orders = [[(i + offset) % 4 for i in range(64)] for offset in range(4)]

        @pipeline_def
        def pipeline():
            source = fn.external_source(source=cycle([[encoded[i] for i in order] for order in orders]),
                                        batch=True, dtype=types.UINT8, ndim=1)
            return fn.l3_decoder(source, device="mixed")

        pipe = pipeline(batch_size=64, num_threads=2, device_id=0, prefetch_queue_depth=2)
        pipe.build()
        for batch in range(12):
            actual = pipe.run()[0].as_cpu().as_array()
            np.testing.assert_array_equal(actual, np.stack([inputs[i] for i in orders[batch % 4]]))


if __name__ == "__main__":
    unittest.main()
