"""Check native coefficient selection against the shared raw-JPEG reference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from galp.benchmarks.coefficient_mask_evaluator.evaluate import (
    BatchedDctPreprocessor,
    RawDctDataset,
)
from galp.benchmarks.system_dct_major.common import (
    load_contract,
    load_sample_manifest,
    resolve_coefficient_selection,
    write_json,
)
from galp.benchmarks.system_dct_major.feature_model import build_workload_model
from galp.profiles.rgbnomore import VALIDATION_CENTER_CROP_512
from galp.torch import DirectDctReader


def verify(contract_path: Path, output: Path, sample_count: int) -> None:
    contract = load_contract(contract_path)
    all_samples = load_sample_manifest(Path(contract["dataset"]["sample_manifest"]))
    samples = all_samples[:sample_count]
    device = torch.device(contract["execution"]["device"])
    root = Path(contract["models"]["rgbnomore_root"])
    config = contract["pipelines"]["dct_major_pushdown"]
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(config["block_major_access_dir"])
    os.environ["GALP_PHASE6_NATIVE_PHYSICAL"] = "1"
    reader = DirectDctReader(config["manifest"], module_path=config["torch_binding_dir"])
    dataset = RawDctDataset(samples, root)
    raw = [dataset[index] for index in range(len(samples))]
    y, cbcr, quant = (torch.stack([item[index] for item in raw]) for index in range(3))
    preprocessor = BatchedDctPreprocessor(root, device)
    model = build_workload_model(
        domain="dct", workload="evaluation", rgbnomore_root=root,
        checkpoint=Path(contract["models"]["dct"]["checkpoint"]), device=device,
    )
    thresholds = contract["semantic_validation"]
    # The scheduled native path activates complete manifest shards, even when
    # the semantic comparison only consumes the first small logical batch.
    logical_batches = [
        [int(sample["galp_image_id"]) for sample in all_samples[offset : offset + len(samples)]]
        for offset in range(0, len(all_samples), len(samples))
    ]
    results = []
    for spec in ("all", "first:32", "first:16", "list:0,2,5,9"):
        selection = resolve_coefficient_selection(spec)
        mask = torch.zeros(64, dtype=torch.bool)
        mask[selection["resolved_natural_indices"]] = True
        reference = tuple(value[0] for value in preprocessor(y, cbcr, quant, mask.reshape(1, 8, 8)))
        image_ids = [int(sample["galp_image_id"]) for sample in samples]
        with reader.pipeline(VALIDATION_CENTER_CROP_512, dct_coeffs=spec).start(logical_batches) as pipeline:
            batch = next(pipeline)
            if list(batch.global_image_ids) != image_ids:
                raise ValueError(f"{spec}: native sample order differs from the reference")
            actual = (batch.y, batch.cbcr)
            errors = []
            for observed, expected in zip(actual, reference, strict=True):
                torch.testing.assert_close(observed, expected, rtol=0, atol=thresholds["input_max_abs"])
                error = (observed - expected).abs()
                mean_error = float(error.mean())
                if mean_error > thresholds["input_mean_abs"]:
                    raise ValueError(f"{spec}: mean DCT error exceeds the contract")
                errors.append({"max_abs": float(error.max()), "mean_abs": mean_error})
            with torch.inference_mode():
                expected_logits = model(*reference)
                actual_logits = model(*actual)
            cosine = float(torch.nn.functional.cosine_similarity(actual_logits, expected_logits).min())
            agreement = float((actual_logits.argmax(1) == expected_logits.argmax(1)).float().mean())
            if not cosine >= thresholds["logit_cosine_min"] or agreement < thresholds["semantic_top1_agreement_min"]:
                raise ValueError(f"{spec}: model predictions differ from the raw-mask reference")
            results.append({"selection": selection, "input_errors": errors, "logit_cosine_min": cosine,
                            "top1_agreement": agreement})
    write_json(output, {"ok": True, "samples": len(samples), "cases": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error("--sample-count must be positive")
    verify(args.contract, args.output, args.sample_count)


if __name__ == "__main__":
    main()
