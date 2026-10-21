import unittest

import torch
from torch.nn import functional as F

from fastlens.benchmarks.baselines.dct_baselines.compact_models import build_model
from fastlens.benchmarks.baselines.dct_baselines.compact_training import benchmark
from .test_compact_training import TensorDataset


class CompactModelTests(unittest.TestCase):
    def test_three_models_update_at_both_real_input_sizes_with_same_initial_weights(self):
        for name in ("resnet50", "mobilenet_v2", "vitti"):
            results = []
            for size in (84, 224):
                with self.subTest(model=name, size=size):
                    result = benchmark(TensorDataset(2, size), model_name=name, device="cpu",
                                       batch_size=2, pool_size=2, warmup_seconds=0.,
                                       epochs=1, repeats=1, seed=20261014)
                    self.assertEqual(result["input_shape"], [3, size, size])
                    self.assertEqual(result["repeats"][0]["optimizer_updates"], 1)
                    self.assertEqual(result["repeats"][0]["samples"], 2)
                    self.assertNotEqual(result["initial_model_sha256"], result["repeats"][0]["final_model_sha256"])
                    self.assertIsNone(result["checkpoint"])
                    self.assertEqual(result["model_input_policy"]["image_resize"], "none")
                    results.append(result)
            self.assertEqual(results[0]["initial_model_sha256"], results[1]["initial_model_sha256"])

    def test_vit_padding_preserves_pixels_and_position_interpolation_trains(self):
        torch.manual_seed(37)
        model = build_model("vitti").train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = model.encoder.pos_embedding.detach().clone()
        observed, token_counts = [], []
        patch_hook = model.conv_proj.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach().clone()))
        layer_hook = model.encoder.layers.register_forward_pre_hook(lambda _, args: token_counts.append(args[0].shape[1]))
        # The standard torchvision head starts at zero. The second step must
        # also train the patch/position/encoder path after the first head update.
        for _ in range(2):
            inputs = torch.randn(2, 3, 84, 84)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (2, 1000))
            F.cross_entropy(logits, torch.tensor([1, 7])).backward()
            optimizer.step()
            torch.testing.assert_close(observed[-1][..., :84, :84], inputs, atol=0, rtol=0)
            self.assertEqual(tuple(observed[-1].shape), (2, 3, 96, 96))
            self.assertEqual(int(torch.count_nonzero(observed[-1][..., 84:, :])), 0)
            self.assertEqual(int(torch.count_nonzero(observed[-1][..., :, 84:])), 0)
        self.assertFalse(torch.equal(before, model.encoder.pos_embedding))
        with torch.no_grad():
            model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(observed[-1].shape), (2, 3, 224, 224))
        self.assertEqual(token_counts, [37, 37, 197])
        patch_hook.remove()
        layer_hook.remove()


if __name__ == "__main__":
    unittest.main()
