#!/usr/bin/env python3
"""Immutable recipe for the core PLS model-effect experiment."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from .model_registry import (
    DEFAULT_MODEL_ID,
    SWINV2_T_MODEL_ID,
    VITTI_MODEL_ID,
    model_configuration,
    recipe_for_model,
)


RECIPE_NAME = "rgbnomore-vitti-dct-published-v1"
SWINV2_RECIPE_NAME = "rgbnomore-swinv2-t-dct-224-v1"
RECIPE_NAMES = (RECIPE_NAME, SWINV2_RECIPE_NAME)
RECIPE_SCHEMA = "galp-pls-recipe-contract-v2"

# This order is copied from RGB-no-more ``generate_config(modelarch='vitti',
# domain='dct')``.  It is part of the recipe and must not vary by condition.
DCT_RANDAUGMENT_OPERATIONS = (
    "AutoContrast",
    "Posterize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "MidfreqAug",
    "Cutout",
    "TranslateX",
    "TranslateY",
    "Rotate90",
    "AutoSaturation",
    "Grayscale",
    "ChromaDrop",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def recipe_contract(name: str = RECIPE_NAME) -> dict[str, Any]:
    recipe_models = {
        RECIPE_NAME: VITTI_MODEL_ID,
        SWINV2_RECIPE_NAME: SWINV2_T_MODEL_ID,
    }
    if name not in recipe_models:
        raise ValueError(f"unknown recipe {name!r}; expected one of {RECIPE_NAMES}")
    model_id = recipe_models[name]
    payload: dict[str, Any] = {
        "schema_version": RECIPE_SCHEMA,
        "recipe": name,
        "reference": {
            "project": "RGB-no-more",
            "local_commit": "dce075711991a5d2e7668e5137f37fb74e1dc4f2",
            "reference_topology": "8gpu_ddp",
            "execution_topology": "single_gpu_accum16",
            "distributed_bitwise_equivalence_claim": False,
        },
        "model": {
            **model_configuration(model_id),
            "initialization_device": "canonical-cpu-state-loaded-into-device-native-model",
        },
        "execution": {
            "model_compile": {
                "enabled": True,
                "backend": "inductor",
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
            },
            "float32_matmul_precision": "highest",
            "randaugment_dispatch": "exact-keyed-operation-grouped-v1",
            "autocast": {
                "enabled": False,
                "device_type": "cuda",
                "dtype": None,
            },
        },
        "training": {
            "epochs": 300,
            "physical_microbatch": 64,
            "gradient_accumulation": 16,
            "effective_update_batch": 1024,
            "precision": "fp32",
            "drop_samples": False,
            "accumulation_crosses_epoch": False,
            "accumulation_crosses_closed_pool": False,
            "partial_update_normalization": "actual_samples_in_accumulation_window",
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 3.0e-3,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "adamw_weight_decay": 0.0,
            "gradient_clipping_norm": 1.0,
            "weight_decay": {
                "type": "rgbnomore-independent-weight-decay",
                "coefficient": 1.0e-4,
                "parameter_selection": "name contains '.weight' and not 'lrnorm'",
                "update": "p *= 1 - (current_lr/base_lr)*coefficient",
            },
        },
        "scheduler": {
            "type": "rgbnomore-warmup-then-cosine",
            "warmup_optimizer_updates": 10_000,
            "step_unit": "optimizer_update",
            "cosine_minimum_learning_rate": 0.0,
            "total_updates_source": "epoch-aware schedule",
        },
        "augmentation": {
            "training_crop": {
                "algorithm": "RGB-no-more RandomResizedCrop_DCT",
                "output_luma_blocks": 28,
                "output_chroma_blocks": 14,
                "scale": [0.05, 1.0],
                "ratio": [1.0, 1.0],
                "luma_crop_size_choices": [2, 4, 14, 28],
                "origin_alignment_luma_blocks": 2,
            },
            "horizontal_flip": {
                "probability": 0.5,
                "scope": "sample",
                "dct_semantics": "reverse block columns and negate odd horizontal frequencies",
            },
            "randaugment": {
                "implementation": "RGB-no-more RandAugment_dct",
                "num_operations": 2,
                "magnitude": 3,
                "magnitude_bins": 11,
                "operations": list(DCT_RANDAUGMENT_OPERATIONS),
                "scope": "sample",
            },
            "mixup": {
                "implementation": "RGB-no-more RandomMixup_DCT",
                "alpha": 0.2,
                "pairing": "roll batch by one",
                "lambda_order": "larger beta component weights original sample",
                "scope": "microbatch",
            },
            "normalization": {
                "input_range": [-1024.0, 1016.0],
                "output_range": [-1.0, 1.0],
                "formula": "(coefficient + 4) / 1020",
            },
        },
        "validation": {
            "transform": "ResizedCenterCrop_DCT(32,28)",
            "luma_reference_resize_blocks": 32,
            "luma_output_blocks": 28,
            "chroma_output_blocks": 14,
            "shuffle": False,
        },
        "logging": {
            "validation_epochs": [0, 1, 2]
            + list(range(5, 301, 5)),
            "train_loss_every_optimizer_updates": 100,
            "per_epoch_train_record": True,
        },
        "scientific_policy": {
            "strategy_selection": False,
            "throughput_is_explanatory": True,
            "memory_is_explanatory": True,
            "loader_metrics_are_explanatory": True,
            "primary_endpoint": "final-checkpoint validation top-1",
            "practical_top1_equivalence_margin_percentage_points": 0.3,
        },
    }
    if model_id == SWINV2_T_MODEL_ID:
        payload["reference"].update(
            {
                "configuration_source": "RGB-no-more SwinV2-T DCT adapted to 224 input",
                "official_256_checkpoint_compatible": False,
                "training_recipe_policy": (
                    "B6-only model-comparison recipe; ViT schedule retained and "
                    "BF16 autocast enabled for the larger architecture"
                ),
            }
        )
        payload["execution"]["autocast"] = {
            "enabled": True,
            "device_type": "cuda",
            "dtype": "bfloat16",
        }
        payload["training"]["precision"] = "bf16-autocast"
        payload["scientific_policy"].update(
            {
                "primary_endpoint": "B6 convergence and equal-image runtime",
                "official_rgbnomore_accuracy_claim": False,
                "model_sweep_recipe": True,
            }
        )
    payload["recipe_hash"] = sha256_json(payload)
    return deepcopy(payload)


def assert_recipe_overrides(
    *,
    recipe: str | None,
    epochs: int | None = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    expected_recipe = recipe_for_model(model_id)
    if recipe is None:
        recipe = expected_recipe
    if recipe != expected_recipe:
        raise ValueError(
            f"model {model_id!r} requires recipe {expected_recipe!r}; got {recipe!r}"
        )
    contract = recipe_contract(recipe)
    fixed_epochs = int(contract["training"]["epochs"])
    if epochs is not None and int(epochs) != fixed_epochs:
        raise ValueError(
            f"recipe {recipe!r} fixes epochs={fixed_epochs}; got --epochs={epochs}"
        )
    return contract


def validation_epochs(recipe: str = RECIPE_NAME) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in recipe_contract(recipe)["logging"]["validation_epochs"]
    )
