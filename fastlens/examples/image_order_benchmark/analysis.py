#!/usr/bin/env python3
"""Quantitative comparison and report generation for image-order experiments."""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "stdev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
        "max": max(numeric),
    }


def _aggregate_records(records: Sequence[dict[str, Any]], exclude_first: bool) -> dict[str, Any]:
    selected = list(records[1:] if exclude_first and len(records) > 1 else records)
    return {
        "repeat_indices": [int(record["repeat"]) for record in selected],
        "throughput_images_per_s": distribution([record["throughput_images_per_s"] for record in selected]),
        "mean_latency_ms": distribution([record["latency_ms"]["mean"] for record in selected]),
        "p95_latency_ms": distribution([record["latency_ms"]["p95"] for record in selected]),
        "accuracy_top1": distribution([record["accuracy_top1"] for record in selected]),
        "accuracy_top5": distribution([record["accuracy_top5"] for record in selected]),
        "rowgroups": distribution([record["native_counters"].get("rowgroups", 0) for record in selected]),
        "worksets": distribution([record["native_counters"].get("worksets", 0) for record in selected]),
        "decode_kernels": distribution([record["native_counters"].get("decode_kernels", 0) for record in selected]),
    }


def _paired_speedup(
    numerator: Sequence[dict[str, Any]],
    denominator: Sequence[dict[str, Any]],
    exclude_first: bool,
) -> dict[str, Any]:
    left = {int(record["repeat"]): record for record in numerator}
    right = {int(record["repeat"]): record for record in denominator}
    repeat_ids = sorted(set(left) & set(right))
    if exclude_first and len(repeat_ids) > 1:
        repeat_ids = repeat_ids[1:]
    ratios = [
        float(left[repeat]["throughput_images_per_s"]) / float(right[repeat]["throughput_images_per_s"])
        for repeat in repeat_ids
    ]
    latency_ratios = [
        float(right[repeat]["latency_ms"]["mean"]) / float(left[repeat]["latency_ms"]["mean"])
        for repeat in repeat_ids
    ]
    return {
        "repeat_indices": repeat_ids,
        "throughput_speedup": distribution(ratios),
        "latency_reduction_factor": distribution(latency_ratios),
        "throughput_percent_change_p50": (percentile(ratios, 0.5) - 1.0) * 100.0,
    }


def compare_semantic_artifacts(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)
    left_ids = left["image_ids"].astype(np.int64)
    right_ids = right["image_ids"].astype(np.int64)
    left_order = np.argsort(left_ids)
    right_order = np.argsort(right_ids)
    ids_equal = np.array_equal(left_ids[left_order], right_ids[right_order])
    if not ids_equal:
        raise ValueError("semantic artifacts do not contain the same image-ID cohort")
    left_labels = left["labels"][left_order]
    right_labels = right["labels"][right_order]
    if not np.array_equal(left_labels, right_labels):
        raise ValueError("semantic artifact labels differ after image-ID alignment")
    left_logits = left["logits"][left_order].astype(np.float64)
    right_logits = right["logits"][right_order].astype(np.float64)
    difference = np.abs(left_logits - right_logits)
    numerator = np.sum(left_logits * right_logits, axis=1)
    denominator = np.linalg.norm(left_logits, axis=1) * np.linalg.norm(right_logits, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-30)
    left_top5 = np.argpartition(left_logits, -5, axis=1)[:, -5:]
    right_top5 = np.argpartition(right_logits, -5, axis=1)[:, -5:]
    left_predictions = np.argmax(left_logits, axis=1)
    right_predictions = np.argmax(right_logits, axis=1)
    left_correct1 = left_predictions == left_labels
    right_correct1 = right_predictions == right_labels
    left_correct5 = np.asarray([label in predictions for label, predictions in zip(left_labels, left_top5)])
    right_correct5 = np.asarray([label in predictions for label, predictions in zip(right_labels, right_top5)])
    left_only_correct = int(np.sum(left_correct1 & ~right_correct1))
    right_only_correct = int(np.sum(~left_correct1 & right_correct1))
    discordant_correctness = left_only_correct + right_only_correct
    if discordant_correctness:
        tail = sum(
            math.comb(discordant_correctness, index)
            for index in range(min(left_only_correct, right_only_correct) + 1)
        ) / (2**discordant_correctness)
        mcnemar_pvalue = min(1.0, 2.0 * tail)
    else:
        mcnemar_pvalue = 1.0
    top1_delta_pp = float((right_correct1.mean() - left_correct1.mean()) * 100.0)
    result = {
        "sample_count": int(len(left_ids)),
        "image_ids_equal_after_alignment": True,
        "labels_equal_after_alignment": True,
        "logit_max_abs": float(difference.max(initial=0.0)),
        "logit_mean_abs": float(difference.mean()),
        "logit_cosine_mean": float(cosine.mean()),
        "top1_prediction_agreement": float(np.mean(left_predictions == right_predictions)),
        "top5_set_agreement": float(
            np.mean(
                [
                    set(left_predictions.tolist()) == set(right_predictions.tolist())
                    for left_predictions, right_predictions in zip(left_top5, right_top5)
                ]
            )
        ),
        "left_accuracy_top1": float(left_correct1.mean()),
        "right_accuracy_top1": float(right_correct1.mean()),
        "accuracy_top1_delta_percentage_points": top1_delta_pp,
        "top1_correctness_left_only": left_only_correct,
        "top1_correctness_right_only": right_only_correct,
        "top1_correctness_discordant": discordant_correctness,
        "top1_exact_mcnemar_pvalue": mcnemar_pvalue,
        "practical_significance_threshold_percentage_points": 0.5,
        "practically_large_accuracy_change": abs(top1_delta_pp) >= 0.5,
        "left_accuracy_top5": float(left_correct5.mean()),
        "right_accuracy_top5": float(right_correct5.mean()),
        "accuracy_top5_delta_percentage_points": float((right_correct5.mean() - left_correct5.mean()) * 100.0),
    }
    left.close()
    right.close()
    return result


def _bootstrap_mean_ci(values: Sequence[float], seed: int, samples: int = 20000) -> list[float]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot bootstrap an empty sequence")
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choice(numeric) for _ in numeric) for _ in range(samples)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def _paired_sign_flip_pvalue(values: Sequence[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot test an empty sequence")
    observed = abs(statistics.fmean(numeric))
    if len(numeric) <= 16:
        means = [
            abs(statistics.fmean(sign * value for sign, value in zip(signs, numeric)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(numeric))
        ]
        return sum(value >= observed - 1e-15 for value in means) / len(means)
    rng = random.Random(0x51A7E)
    trials = 100000
    extreme = 0
    for _ in range(trials):
        candidate = abs(statistics.fmean((1.0 if rng.getrandbits(1) else -1.0) * value for value in numeric))
        extreme += int(candidate >= observed - 1e-15)
    return extreme / trials


def summarize_training_probe(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs", [])
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_seed.setdefault(int(run["seed"]), {})[str(run["condition"])] = run
    complete = {
        seed: conditions
        for seed, conditions in by_seed.items()
        if {"paired_scattered", "contiguous"}.issubset(conditions)
    }
    if not complete:
        return {"available": False, "reason": "no complete paired training runs"}

    def differences(metric: str) -> list[float]:
        return [
            float(conditions["contiguous"]["final_evaluation"][metric])
            - float(conditions["paired_scattered"]["final_evaluation"][metric])
            for conditions in complete.values()
        ]

    top1_pp = [value * 100.0 for value in differences("accuracy_top1")]
    top5_pp = [value * 100.0 for value in differences("accuracy_top5")]
    loss_delta = differences("loss")
    top1_ci = _bootstrap_mean_ci(top1_pp, int(payload.get("seed", 11997733)))
    practical_threshold_pp = float(payload.get("practical_significance_threshold_percentage_points", 0.5))
    sign_flip_pvalue = _paired_sign_flip_pvalue(top1_pp)
    ci_excludes_zero = bool(top1_ci[0] > 0.0 or top1_ci[1] < 0.0)
    return {
        "available": True,
        "paired_seed_count": len(complete),
        "seeds": sorted(complete),
        "contiguous_minus_scattered_top1_percentage_points": distribution(top1_pp),
        "contiguous_minus_scattered_top1_mean_bootstrap_95_ci": top1_ci,
        "contiguous_minus_scattered_top1_sign_flip_pvalue": sign_flip_pvalue,
        "contiguous_minus_scattered_top5_percentage_points": distribution(top5_pp),
        "contiguous_minus_scattered_loss": distribution(loss_delta),
        "practical_significance_threshold_percentage_points": practical_threshold_pp,
        "bootstrap_ci_excludes_zero": ci_excludes_zero,
        "statistically_detected": bool(ci_excludes_zero and sign_flip_pvalue < 0.05),
        "practically_large_mean": abs(statistics.fmean(top1_pp)) >= practical_threshold_pp,
        "scope": payload.get("scope"),
    }


def summarize(
    *,
    selection: dict[str, Any],
    performance: dict[str, Any],
    semantic_paths: dict[str, Path],
    training: dict[str, Any] | None,
    training_order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records_by_condition = {
        condition: sorted(records, key=lambda item: int(item["repeat"]))
        for condition, records in performance["records"].items()
    }
    exclude_first = bool(performance.get("exclude_first_repeat", True))
    aggregates = {
        condition: _aggregate_records(records, exclude_first)
        for condition, records in records_by_condition.items()
    }
    paired_effect = _paired_speedup(
        records_by_condition["contiguous"],
        records_by_condition["paired_scattered"],
        exclude_first,
    )
    operational_effect = _paired_speedup(
        records_by_condition["contiguous"],
        records_by_condition["current_random"],
        exclude_first,
    )
    inference = compare_semantic_artifacts(
        semantic_paths["paired_scattered"], semantic_paths["contiguous"]
    )
    training_summary = summarize_training_probe(training) if training is not None else {
        "available": False,
        "reason": "training-order probe skipped",
    }
    return {
        "schema_version": "galp_image_order_summary_v1",
        "selection_sha256": performance["selection_sha256"],
        "selection_invariants": selection["paired_invariants"],
        "access_patterns": {
            condition: payload["access"] for condition, payload in selection["conditions"].items()
        },
        "performance": {
            "exclude_first_repeat": exclude_first,
            "aggregates": aggregates,
            "paired_same_cohort_contiguous_vs_scattered": paired_effect,
            "operational_contiguous_vs_current_random": operational_effect,
            "timing_boundary": performance["timing_boundary"],
        },
        "inference_semantics": inference,
        "training_order_probe": training_summary,
        "full_training_label_order": training_order,
        "interpretation_guards": [
            "The paired scattered/contiguous comparison uses exactly the same warmup and measured image-ID sets.",
            "The current-random/contiguous comparison is operational but uses different image cohorts and is not a causal estimate by itself.",
            "Inference accuracy is evaluated after aligning outputs by FASTLENS image ID, not by batch ordinal.",
            "The training result is a short deterministic fine-tuning sensitivity probe, not a replacement for full ImageNet retraining.",
        ],
    }


def write_csv(path: Path, performance: dict[str, Any]) -> None:
    columns = (
        "condition",
        "repeat",
        "execution_order",
        "images",
        "seconds",
        "throughput_images_per_s",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "accuracy_top1",
        "accuracy_top5",
        "rowgroups",
        "worksets",
        "decode_kernels",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for condition, records in performance["records"].items():
            for record in records:
                writer.writerow(
                    {
                        "condition": condition,
                        "repeat": record["repeat"],
                        "execution_order": record["execution_order"],
                        "images": record["images"],
                        "seconds": record["seconds"],
                        "throughput_images_per_s": record["throughput_images_per_s"],
                        "latency_mean_ms": record["latency_ms"]["mean"],
                        "latency_p50_ms": record["latency_ms"]["p50"],
                        "latency_p95_ms": record["latency_ms"]["p95"],
                        "accuracy_top1": record["accuracy_top1"],
                        "accuracy_top5": record["accuracy_top5"],
                        "rowgroups": record["native_counters"].get("rowgroups", 0),
                        "worksets": record["native_counters"].get("worksets", 0),
                        "decode_kernels": record["native_counters"].get("decode_kernels", 0),
                    }
                )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    performance = summary["performance"]
    aggregates = performance["aggregates"]
    paired = performance["paired_same_cohort_contiguous_vs_scattered"]
    operational = performance["operational_contiguous_vs_current_random"]
    inference = summary["inference_semantics"]
    training = summary["training_order_probe"]
    training_order = summary.get("full_training_label_order")
    access = summary["access_patterns"]
    lines = [
        "# FastLens random and contiguous image-ID evaluation",
        "",
        "## Results",
        "",
        (
            f"For the same measured cohort, median contiguous-batch throughput relative to scattered batches is "
            f"**{paired['throughput_speedup']['p50']:.2f}×** ({paired['throughput_percent_change_p50']:+.1f}%); "
            f"Relative to the globally random benchmark, the throughput ratio is **{operational['throughput_speedup']['p50']:.2f}×**."
        ),
        (
            f"For {inference['sample_count']} identical images aligned by image ID, the top-1/top-5 accuracy changes are "
            f"**{inference['accuracy_top1_delta_percentage_points']:+.4f} / "
            f"{inference['accuracy_top5_delta_percentage_points']:+.4f} percentage points**, "
            f"top-1 prediction agreement {inference['top1_prediction_agreement'] * 100:.4f}%, "
            f"exact McNemar p={inference['top1_exact_mcnemar_pvalue']:.4f}."
        ),
        "",
        "## End-to-end performance",
        "",
        "| Condition | Throughput p50 (img/s) | Mean batch latency p50 (ms) | rowgroups / measured window | top-1 | top-5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in ("current_random", "paired_scattered", "contiguous"):
        item = aggregates[condition]
        lines.append(
            f"| {condition} | {item['throughput_images_per_s']['p50']:.3f} | "
            f"{item['mean_latency_ms']['p50']:.3f} | {item['rowgroups']['p50']:.0f} | "
            f"{item['accuracy_top1']['p50']:.4f} | {item['accuracy_top5']['p50']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Timing boundary: `{performance['timing_boundary']}`. Exclude repeat 0 from the aggregate: "
            f"`{performance['exclude_first_repeat']}`.",
            "",
            "## Access patterns and class distribution",
            "",
            "| Condition | Mean physical rowgroups/batch | Adjacent ID fraction | Mean unique classes/batch | Mean label entropy (bits) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in ("current_random", "paired_scattered", "contiguous"):
        item = access[condition]
        lines.append(
            f"| {condition} | {item['rowgroups_per_batch']['mean']:.2f} | "
            f"{item['forward_adjacent_pair_fraction'] * 100:.2f}% | "
            f"{item['unique_classes_per_batch']['mean']:.2f} | "
            f"{item['label_entropy_bits_per_batch']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Physical IDs follow class-sorted paths, so contiguous batches have lower class diversity. "
            "The measured prediction agreement and accuracy differences are reported below; training order also changes the composition of each gradient update.",
            "",
            "## Inference accuracy and per-sample semantics",
            "",
            f"- Logit max/mean absolute difference: `{inference['logit_max_abs']:.8g}` / `{inference['logit_mean_abs']:.8g}`.",
            f"- Logit cosine mean: `{inference['logit_cosine_mean']:.10f}`.",
            f"- Top-1 / top-5 set agreement: `{inference['top1_prediction_agreement']:.6f}` / "
            f"`{inference['top5_set_agreement']:.6f}`.",
            f"- Top-1 correctness discordant pairs: left-only `{inference['top1_correctness_left_only']}`, "
            f"right-only `{inference['top1_correctness_right_only']}`; exact McNemar "
            f"p=`{inference['top1_exact_mcnemar_pvalue']:.4f}`.",
            f"- Accuracy change exceeds the 0.5 percentage-point practical threshold: `{inference['practically_large_accuracy_change']}`.",
            "",
            "## Training-order sensitivity probe",
            "",
        ]
    )
    if training_order:
        contiguous_order = training_order["contiguous"]
        random_order = training_order["random"]
        lines.extend(
            [
                f"In the full ImageNet training index ({training_order['image_count']:,} images, batch={training_order['batch_size']}), "
                f"contiguous batches contain a mean of **{contiguous_order['unique_classes_mean']:.2f}** classes, "
                f"**{contiguous_order['single_class_batch_fraction'] * 100:.2f}%** of batches contain a single class; "
                f"global shuffling produces a mean of **{random_order['unique_classes_mean']:.2f}** classes and "
                f"{random_order['single_class_batch_fraction'] * 100:.2f}% single-class batches.",
                "",
            ]
        )
    if training.get("available"):
        top1 = training["contiguous_minus_scattered_top1_percentage_points"]
        ci = training["contiguous_minus_scattered_top1_mean_bootstrap_95_ci"]
        lines.extend(
            [
                f"- Paired seeds: {training['paired_seed_count']}.",
                f"- Held-out top-1, contiguous minus scattered: mean `{top1['mean']:+.4f}` percentage points, "
                f"bootstrap 95% CI `[{ci[0]:+.4f}, {ci[1]:+.4f}]`, "
                f"paired sign-flip p=`{training['contiguous_minus_scattered_top1_sign_flip_pvalue']:.4f}`.",
                f"- Prespecified practical significance threshold: `{training['practical_significance_threshold_percentage_points']:.3f}` percentage points; "
                f"Mean exceeds threshold: `{training['practically_large_mean']}`; "
                f"Bootstrap CI excludes zero: `{training['bootstrap_ci_excludes_zero']}`; "
                f"Statistically detected by the exact paired test: `{training['statistically_detected']}`.",
                f"- Scope: {training.get('scope')}.",
            ]
        )
    else:
        lines.append(f"Not run: {training.get('reason', 'unknown')}.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    lines.extend(f"- {guard}" for guard in summary["interpretation_guards"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_artifacts(output_dir: Path, summary: dict[str, Any], performance: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "performance.csv", performance)
    write_report(output_dir / "report.md", summary)
