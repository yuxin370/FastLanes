"""Explicit real-GPU smoke test for the RGB training model."""

import unittest

import torch

from galp.benchmarks.common import DEFAULT_RGBNOMORE_ROOT
from galp.benchmarks.system_rgbnomore.training.model_factory import build_model


class TrainingGpuTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_gpu_formal_model_step(self) -> None:
        model = build_model(DEFAULT_RGBNOMORE_ROOT, "rgb", torch.device("cuda:0"))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss = torch.nn.functional.cross_entropy(
            model(torch.randn(1, 3, 224, 224, device="cuda:0")),
            torch.tensor([1], device="cuda:0"),
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
