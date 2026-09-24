"""CPU integration checks; requires the FUN checkout and published eFUN checkpoint."""
import io
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import galp.benchmarks.dct_models.efun_backend as B
from galp.benchmarks.dct_models.dct_geometry import load_upsample_dct


class EfunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        pixels = np.random.default_rng(11997733).integers(0, 256, (277, 341, 3), dtype=np.uint8)
        cls.image = Image.fromarray(pixels)

    def encoded_image(self):
        result = io.BytesIO()
        self.image.save(result, format="PNG")
        result.seek(0)
        return result

    def test_stored_input_matches_author_transform(self):
        reference = B.Reference()
        q, tables, components = reference.coefficients(self.encoded_image(), verify=True)
        actual = B.organize([v.astype(np.float32) * table for v, table in zip(q, tables)])
        expected = reference(self.encoded_image())
        self.assertEqual(tuple(actual.shape), (192, 28, 28))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(B.organize(components), expected, rtol=0, atol=0)

    def test_published_checkpoint_and_training_step(self):
        net = B.model()
        x = B.Reference()(self.encoded_image())
        with torch.inference_mode():
            logits = net(x[None])
        self.assertEqual(tuple(logits.shape), (1, 1000))
        self.assertTrue(torch.isfinite(logits).all())
        # Exercise the shared GALP driver's scratch initialization contract.
        torch.manual_seed(11997733)
        for module in net.modules():
            if list(module.parameters(recurse=False)):
                module.reset_parameters()
        from timm.data import create_transform
        from timm.optim import RMSpropTF
        transform = create_transform((3, 224, 224), is_training=True, dct=True,
                                     use_prefetcher=False, color_jitter=.4)
        inputs = torch.stack([transform(self.image), transform(self.image)])
        net.train().requires_grad_(True)
        optimizer = RMSpropTF(net.parameters(), lr=.001, eps=.001)
        before = net.classifier.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(net(inputs), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, net.classifier.weight))

    def test_existing_cnn_upsampling_unchanged(self):
        upsample = load_upsample_dct(B.DEFAULT_RGBNOMORE_ROOT)
        resize = load_upsample_dct(B.DEFAULT_RGBNOMORE_ROOT, resize=True)
        x = torch.randn(1, 14, 28, 8, 8)
        expected, _, _ = upsample(x, L=4, M=2)
        actual = resize(x, 56, dtype_out=torch.float32, conv_mxs={})
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_downsample_preserves_constant_signal(self):
        resize = load_upsample_dct(B.DEFAULT_RGBNOMORE_ROOT, resize=True)
        x = torch.zeros(1, 56, 14, 8, 8)
        x[..., 0, 0] = 80  # Every spatial block represents the same constant signal.
        actual = resize(x, 28, dtype_out=torch.float32, conv_mxs={})
        expected = torch.zeros(1, 28, 28, 8, 8)
        expected[..., 0, 0] = 80
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)

    def test_rgb_checkpoint_and_input(self):
        with patch.dict(sys.modules, {"galp.benchmarks.dct_models.backend": B}), \
                patch("galp.benchmarks.dct_models.backend", B, create=True):
            import galp.benchmarks.dct_models.rgb as rgb
        encoded = self.encoded_image().getvalue()
        entries = [dict(path="synthetic.png", model_label=7, ordinal=0)]
        with patch("galp.benchmarks.dct_models.rgb.Path.read_bytes", return_value=encoded):
            x, label, ordinal, *_, size = rgb.Inputs(entries)[0]
        from torchvision.models import EfficientNet_B0_Weights
        expected = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()(self.image)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)
        self.assertEqual((label, ordinal, size), (7, 0, len(encoded)))
        with torch.inference_mode():
            logits = rgb.model()(x[None])
        self.assertEqual(tuple(logits.shape), (1, 1000))
        self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipUnless(os.environ.get("EFUN_RGB_TEST_CUDA") == "1", "explicit CUDA test selection required")
    def test_rgb_dali_partial_batch(self):
        with patch.dict(sys.modules, {"galp.benchmarks.dct_models.backend": B}), \
                patch("galp.benchmarks.dct_models.backend", B, create=True):
            import galp.benchmarks.dct_models.rgb as rgb
            from galp.benchmarks.dct_models.evaluate import samples
        entries = samples(17)
        net = rgb.model().cuda()
        seen = []
        sizes = []
        with torch.inference_mode():
            for batch in rgb.dali_loader(entries, batch_size=8, workers=2, prefetch_queue_depth=4):
                x = batch[0]["image"]
                seen.extend(batch[0]["ordinal"].flatten().tolist())
                sizes.append(len(x))
                self.assertEqual(tuple(x.shape[1:]), (3, 224, 224))
                self.assertEqual(x.dtype, torch.float32)
                self.assertTrue(torch.isfinite(net(x)).all())
        self.assertEqual(seen, list(range(17)))
        self.assertEqual(sizes, [8, 8, 1])


if __name__ == "__main__":
    unittest.main()
