"""Run eFUN through the existing CNN inference, generation, and training drivers."""
from __future__ import annotations

import argparse
import os
import runpy
import sys

import efun_backend as backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["evaluate", "evaluate_rgb", "generate", "validate", "evaluate_shards",
                                         "training_pls", "training_model_only", "training_input_probe", "official_train"])
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.modules["backend"] = backend
    if args.task == "official_train":
        backend.upstream_timm()
        script = backend.UPSTREAM / "train.py"
        # Defaults follow the author's README. Later CLI arguments can override them.
        defaults = ["--model", "efun", "--dct", "--no-prefetcher", "--batch-size", "128",
                    "--epochs", "450", "--sched", "step", "--decay-epochs", "2.4",
                    "--decay-rate", ".97", "--opt", "rmsproptf", "--opt-eps", ".001",
                    "--warmup-lr", "1e-6", "--weight-decay", "1e-5", "--lr", ".048",
                    "--drop", ".2", "--drop-path", ".2", "--model-ema", "--model-ema-decay", ".9999",
                    "--remode", "pixel", "--reprob", ".2"]
        # Upstream resume uses torch.load without weights_only, for its own checkpoints.
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    else:
        script = backend.HERE / f"{args.task}.py"
        defaults = []
        if args.task == "training_pls":
            defaults = ["--validation-data", str(backend.VALIDATION_DATA)]
        if args.task == "training_model_only":
            defaults = ["--domain", "dct", "--required-gpu-name", ""]
        if args.task == "training_input_probe":
            defaults = ["--required-gpu-name", ""]
    sys.argv = [str(script), *defaults, *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
