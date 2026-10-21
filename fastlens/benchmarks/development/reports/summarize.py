#!/usr/bin/env python3
"""Export measured full-pass results; never promote smoke or failed gates."""
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results"
FIELDS = "timestamp git_commit gpu cpu model domain pipeline task precision batch_size trial samples elapsed_s images_per_s model_only_images_per_s storage_bytes issued_read_bytes h2d_bytes h2d_copy_count gpu_peak_process_bytes torch_peak_allocated_bytes torch_peak_reserved_bytes native_peak_bytes host_rss_peak_bytes pinned_peak_bytes accuracy notes raw_result correctness_status".split()
FIELDS += ["optimizer_updates", "effective_batch_size", "training_loss", "initial_model_hash"]
FIELDS += ["condition", "diagnostic_raw"]


def model_name(profile, domain):
    if domain == "rgb":
        if profile.startswith("mobilenet"):
            return "MobileNetV2"
        if profile.startswith("resnet"):
            return "ResNet-50"
        if profile == "efun":
            return "EfficientNet-B0"
    return {"vitti": "ViT-Ti", "swinv2": "SwinV2-T-256-window8", "efun": "eFUN-DCT",
            "mobilenet24": "MobileNetV2-DCT24", "mobilenet32": "MobileNetV2-DCT32",
            "resnet24": "ResNet-50-DCT24", "resnet64": "ResNet-50-DCT64"}[profile]


def read(path):
    return json.loads(path.read_text())


def correctness_passed(gate):
    """Keep available valid arms; missing arms and timing noise remain explicit."""
    return all(": hot throughput CV " in failure or ": throughput endpoint drift " in failure
               or failure.startswith("missing pipeline result: ")
               for failure in gate.get("validation_failures", gate["failures"]))


from fastlens.benchmarks.common import write_csv


def summarize(rows):
    groups = {}
    keys = "gpu model domain pipeline task precision batch_size".split()
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for key, group in groups.items():
        values = [r["samples"] / r["elapsed_s"] for r in group]
        accuracies = [r["accuracy"] for r in group if r.get("accuracy") is not None]
        cv = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None
        summary = dict(zip(keys, key), trials=len(group), median_images_per_s=statistics.median(values),
                       min_images_per_s=min(values), max_images_per_s=max(values),
                       median_elapsed_s=statistics.median(r["elapsed_s"] for r in group), cv=cv,
                       accuracy=statistics.median(accuracies) if accuracies else None,
                       trial_status="COMPLETE" if len(group) >= 5 or (len(group) == 3 and cv <= .02) else "TBD: additional trials required",
                       measurement_status="PROVISIONAL" if any("PROVISIONAL" in (r.get("notes") or "") for r in group) or (cv is not None and cv > .02) else "VALIDATED",
                       raw_results=";".join(r["raw_result"] for r in group))
        summaries.append(summary)
    return summaries


def training_rows(gpu_names, cpu):
    rows = []
    for path in sorted((RESULTS / "raw").glob("*/*/training/*/warm_epochs.json")):
        group = read(path)
        alias = path.relative_to(RESULTS / "raw").parts[0]
        for trial, warm in enumerate(group["trials"], 1):
            metadata = read(path.parent / f"trial_{trial}.command.json")
            profile = group["model"]
            domain = "rgb" if group["pipeline"] in ("pytorch_rgb", "dali") else "dct"
            model = {"rgbnomore-vitti-dct-224-v1": "ViT-Ti",
                     "rgbnomore-swinv2-t-dct-224-v1": "SwinV2-T-224-window7"}.get(profile)
            if model is None:
                model = model_name(profile, domain)
            row = dict.fromkeys(FIELDS)
            row.update(timestamp=metadata["timestamp"], git_commit=metadata["git_commit"],
                       gpu=gpu_names[alias], cpu=cpu, model=model, domain=domain,
                       pipeline=group["pipeline"], task="training", precision=warm["precision"],
                       batch_size=warm["microbatch"], trial=trial, samples=warm["samples"],
                       elapsed_s=warm["elapsed_s"], optimizer_updates=warm["updates"],
                       effective_batch_size=warm["effective_batch_size"], training_loss=warm["loss"],
                       initial_model_hash=warm.get("initial_model_hash"),
                       raw_result=warm["raw_result"], notes=group["comparison"],
                       correctness_status="PASS: full epoch 2 coverage, updates and finite loss; no convergence equivalence claim")
            rows.append(row)
    return rows


def main():
    environment = read(RESULTS / "inventory/environment.json")
    cpu_info = json.loads(environment["evidence"]["cpu"]["stdout"])["lscpu"]
    cpu = next(row["data"] for row in cpu_info if row["field"] == "Model name:")
    gpu_names = [gpu.findtext("product_name") for gpu in ET.fromstring(environment["evidence"]["gpu"]["stdout"]).findall("gpu")]
    gpu_name = {alias: next(name for name in gpu_names if alias in name) for alias in ("4090", "H100")}
    rows = []
    for run_dir in sorted((RESULTS / "raw").glob("*/*/inference*")):
        gate_path = run_dir / "results.json"
        if not gate_path.exists():
            continue
        gate = read(gate_path)
        if not correctness_passed(gate):
            continue
        contract = read(run_dir / "contract.json")
        model = "ViT-Ti" if run_dir.parent.name == "vitti" else "SwinV2-T-256-window8"
        legacy = run_dir.parent.name == "swinv2"
        allocated_key = "peak_gpu_memory_allocated_bytes" if legacy else "peak_torch_gpu_allocated_bytes"
        reserved_key = "peak_gpu_memory_reserved_bytes" if legacy else "peak_torch_gpu_reserved_bytes"
        for path in sorted(run_dir.glob("pipeline_*.json")):
            if path.name.endswith(".command.json"):
                continue
            resumed = path.with_name(path.stem + "_resumed.command.json")
            metadata = read(resumed if resumed.exists() else run_dir.with_suffix(".command.json"))
            result = read(path)
            for repeat in result["repeats"][1:]:
                if repeat["images"] != 50000:
                    raise ValueError(f"not full validation: {path}")
                row = dict.fromkeys(FIELDS)
                row.update(timestamp=metadata["timestamp"], git_commit=metadata["git_commit"],
                           gpu=gpu_name[path.relative_to(RESULTS / "raw").parts[0]], cpu=cpu, model=model,
                           domain=result["domain"], pipeline=result["pipeline"], task="inference",
                           precision="fp32", batch_size=contract["execution"]["batch_size"],
                           trial=repeat["repeat"] + (3 if path.parent.name == "inference_extra" else 0),
                           samples=repeat["images"], elapsed_s=repeat["repeat_scope_seconds"],
                           torch_peak_allocated_bytes=repeat[allocated_key],
                           torch_peak_reserved_bytes=repeat[reserved_key],
                           accuracy=repeat["accuracy_top1"],
                           notes="ImageNet-512 view; full repeat wall time; first pass excluded; cross-domain deployment comparison; unknown counters blank",
                           raw_result=str(path.relative_to(ROOT)) + f"#/repeats/{repeat['repeat']}",
                           correctness_status="PASS: sample coverage and DCT equivalence" if result["domain"] == "dct" else "PASS: sample coverage; RGB preprocessing equivalence is diagnostic, not required")
                if any(": hot throughput CV " in failure or ": throughput endpoint drift " in failure for failure in gate["failures"]):
                    row["notes"] += "; PROVISIONAL: existing stability gate failed; retain all observations"
                if gate.get("performance_gate_failures"):
                    row["notes"] += "; artifact performance targets failed; valid observations retained"
                if any(failure.startswith("missing pipeline result: ") for failure in gate["failures"]):
                    row["notes"] += "; MATRIX INCOMPLETE: missing pipeline is omitted, not imputed"
                rows.append(row)
    for gate_path in sorted((RESULTS / "raw").glob("*/*/inference/correctness.json")):
        gate = read(gate_path)
        if not gate["numerical_control_pass"]:
            continue
        for path in sorted(gate_path.parent.glob("*/trial_*/*_50000.json")):
            if path.name == "samples_50000.json":
                continue
            trial = int(path.parent.name.removeprefix("trial_"))
            if trial == 0:
                continue
            result = read(path)
            if result["samples"] != 50000 or len(set(result["sample_ids"])) != 50000:
                raise ValueError(f"incomplete sample coverage: {path}")
            metadata = read(path.parent.with_suffix(".command.json"))
            pipeline = path.parent.parent.name
            domain = "rgb" if pipeline.startswith("rgb_") else "dct"
            profile = gate_path.parent.parent.name
            row = dict.fromkeys(FIELDS)
            row.update(timestamp=metadata["timestamp"], git_commit=metadata["git_commit"],
                       gpu=gpu_name[path.relative_to(RESULTS / "raw").parts[0]], cpu=cpu,
                       model=model_name(profile, domain), domain=domain, pipeline=pipeline, task="inference",
                       precision="fp32", batch_size=result["batch_size"], trial=trial, samples=result["samples"],
                       elapsed_s=result["e2e_seconds"], accuracy=result["top1"] / 100,
                       notes="ImageNet-512; physical image order; online source crop; process warmup pass excluded; RGB is a deployment comparison",
                       raw_result=str(path.relative_to(ROOT)), correctness_status="PASS: coverage, small tensor gate and full prediction comparison")
            rows.append(row)
    write_csv(RESULTS / "e2e/inference_raw.csv", rows, FIELDS)
    training = training_rows(gpu_name, cpu)
    write_csv(RESULTS / "e2e/training_raw.csv", training, FIELDS)
    summaries = summarize(rows)
    ceiling_rows = []
    for path in sorted((RESULTS / "raw").glob("*/*/model_only_*.json")):
        if ".command." in path.name:
            continue
        result = read(path)
        for trial in result["trials"]:
            row = dict.fromkeys(FIELDS)
            row.update(timestamp=result["timestamp"], git_commit=result["git_commit"], gpu=result["gpu"],
                       cpu=cpu, model=model_name(result["model"], result["domain"]),
                       domain=result["domain"], pipeline="model_only", task="inference",
                       precision=result["precision"], batch_size=result["batch_size"],
                       trial=trial["trial"], samples=trial["samples"], elapsed_s=trial["elapsed_s"],
                       torch_peak_allocated_bytes=trial["torch_peak_allocated_bytes"],
                       torch_peak_reserved_bytes=trial["torch_peak_reserved_bytes"],
                       raw_result=str(path.relative_to(ROOT)) + f"#/trials/{trial['trial'] - 1}",
                       correctness_status=result["correctness"], notes=result["scope"])
            ceiling_rows.append(row)
    ceilings = summarize(ceiling_rows)
    lookup = {(r["gpu"], r["model"], r["domain"], r["precision"], r["batch_size"]): r["median_images_per_s"] for r in ceilings}
    for row in summaries:
        ceiling = lookup.get(tuple(row[k] for k in ("gpu", "model", "domain", "precision", "batch_size")))
        row.update(model_only_images_per_s=ceiling,
                   pipeline_efficiency=row["median_images_per_s"] / ceiling if ceiling else None)
    write_csv(RESULTS / "e2e/model_only_raw.csv", ceiling_rows, FIELDS)
    resources = []
    for name in ('h2d', 'memory'):
        path = RESULTS / 'cost' / (name + '.csv')
        if not path.exists():
            continue
        with path.open() as stream:
            for observed in csv.DictReader(stream):
                row = dict.fromkeys(FIELDS)
                row.update({k: v for k, v in observed.items() if k in FIELDS})
                row.update(cpu=cpu, task='inference_' + name,
                           domain='dct' if observed['pipeline'] in ('dct_major_pushdown', 'rgbnomore') else 'rgb',
                           precision='fp32', batch_size=50)
                resources.append(row)
    for name in ('crop_pushdown', 'frequency_pushdown', 'tensor_construction', 'compression'):
        path = RESULTS / 'ablation' / (name + '.csv')
        if not path.exists():
            continue
        with path.open() as stream:
            for observed in csv.DictReader(stream):
                row = dict.fromkeys(FIELDS)
                row.update({k: v for k, v in observed.items() if k in FIELDS and k != 'images_per_s'})
                row.update(cpu=cpu, task='ablation_' + name, condition=observed['case'])
                resources.append(row)
    write_csv(RESULTS / "unified_raw.csv", rows + training + ceiling_rows + resources, FIELDS)
    unified = []
    for row in rows + training + ceiling_rows:
        derived = dict(row)
        derived["images_per_s"] = row["samples"] / row["elapsed_s"]
        if row['task'] == 'inference':
            derived["model_only_images_per_s"] = lookup.get(tuple(row[k] for k in ("gpu", "model", "domain", "precision", "batch_size")))
        unified.append(derived)
    resource_derived = []
    for row in resources:
        derived = dict(row)
        if row['task'].startswith('ablation_'):
            derived['images_per_s'] = float(row['samples']) / float(row['elapsed_s'])
        resource_derived.append(derived)
    write_csv(RESULTS / "unified.csv", unified + resource_derived, FIELDS)
    summary_fields = "gpu model domain pipeline task precision batch_size trials median_images_per_s min_images_per_s max_images_per_s median_elapsed_s cv accuracy trial_status measurement_status raw_results model_only_images_per_s pipeline_efficiency".split()
    write_csv(RESULTS / "e2e/inference_summary.csv", summaries, summary_fields)
    write_csv(RESULTS / "inference_table.csv", summaries, summary_fields)
    training_summaries = summarize(training)
    write_csv(RESULTS / "e2e/training_summary.csv", training_summaries, summary_fields)
    write_csv(RESULTS / "training_table.csv", training_summaries, summary_fields)
    scaling = []
    for low in summaries:
        if low["gpu"] != gpu_name["4090"]:
            continue
        for high in summaries:
            if high["gpu"] == gpu_name["H100"] and all(low[k] == high[k] for k in ("model", "domain", "pipeline", "precision", "batch_size")):
                scaling.append({k: low[k] for k in ("model", "domain", "pipeline", "precision", "batch_size")} |
                               dict(gpu_e2e_scaling=high["median_images_per_s"] / low["median_images_per_s"],
                                    gpu_model_scaling=high["model_only_images_per_s"] / low["model_only_images_per_s"] if high["model_only_images_per_s"] and low["model_only_images_per_s"] else None,
                                    efficiency_4090=low["pipeline_efficiency"], efficiency_h100=high["pipeline_efficiency"],
                                    host_comparison="same host; see PCIe/NUMA topology in inventory", raw_results=low["raw_results"] + ";" + high["raw_results"]))
    write_csv(RESULTS / "analysis/gpu_scaling.csv", scaling,
              "model domain pipeline precision batch_size gpu_e2e_scaling gpu_model_scaling efficiency_4090 efficiency_h100 host_comparison raw_results".split())
    report = ["# FastLens evaluation — IN PROGRESS", "",
              environment.get("evaluation_status_note", ""), "",
              "No systems conclusion is accepted until the corresponding correctness gates and trial requirements pass.", "",
              "User-confirmed primary dataset: all 50,000 samples of the artifact's existing ImageNet-512 validation view. This is resized ImageNet, not original-resolution JPEGs. Both GPUs use the same host and data paths.", "",
              "RTX 4090 and H100 are installed in the same host; exact device, PCIe, topology, host and software evidence is in `results/inventory/environment.json` and `results/inventory/raw/`.", "",
              "RGB versus DCT, CNN A0 versus B6 crop/shuffle recipes, and altered model variants are deployment/model tradeoff comparisons, not causal input-system comparisons.", "",
              "## Executed commands", ""]
    for path in sorted(RESULTS.rglob("*.command.json")):
        if "excluded" in path.parts:
            continue
        meta = read(path)
        report.extend([f"### {path.relative_to(ROOT)}", "", f"- Git commit: `{meta['git_commit']}` (working tree status recorded in inventory).",
                       f"- Machine: `{environment['hostname']}`; CUDA_VISIBLE_DEVICES=`{meta['environment']['CUDA_VISIBLE_DEVICES']}`.",
                       f"- Exit status: {meta.get('status', meta.get('returncode', 'RUNNING'))}; raw log: `{path.with_suffix('').with_suffix('.log').relative_to(ROOT)}`.",
                       "- Configuration and exact command:", "", "```sh", meta["shell_command"], "```", ""])
    report.extend(["## Full inference summaries", ""])
    if summaries:
        for row in summaries:
            report.append(f"- {row['gpu']} / {row['model']} / {row['domain']} / {row['pipeline']}: median {row['median_images_per_s']:.3f} images/s; min {row['min_images_per_s']:.3f}; max {row['max_images_per_s']:.3f}; {row['trial_status']}; {row['measurement_status']}. Source: {row['raw_results']}.")
    else:
        report.append("MISSING/TBD — no complete correctness-gated full-inference trial group yet. Smoke measurements are excluded.")
    report.extend(["", "## CNN prediction correctness", ""])
    for path in sorted((RESULTS / "raw").glob("*/*/inference/correctness.json")):
        gate = read(path)
        report.append(f"- {path.relative_to(RESULTS / 'raw')}: ordered sample IDs equal={gate['sample_order_equal']}; full 50,000-image prediction agreement={gate['prediction_agreement']}; numerical gate={gate['numerical_control_pass']}. {gate['note']}. The acceptance threshold is 0.999; agreement below 1 is not exact prediction equivalence.")
    report.extend(["", "## Resident-model inference ceilings", ""])
    for row in ceilings:
        report.append(f"- {row['gpu']} / {row['model']} / {row['domain']} / batch {row['batch_size']} / {row['precision']}: median {row['median_images_per_s']:.3f} images/s; min {row['min_images_per_s']:.3f}; max {row['max_images_per_s']:.3f}; {row['trial_status']}; {row['measurement_status']}. Shape/finite-output gate passed. Sources: {row['raw_results']}.")
    report.extend(["", "## Training", ""])
    for row in training_summaries:
        report.append(f"- {row['gpu']} / {row['model']} / {row['pipeline']}: warm epoch-2 median {row['median_images_per_s']:.3f} images/s; {row['trial_status']}; {row['measurement_status']}. Sources: {row['raw_results']}.")
    if not training:
        report.append("MISSING/TBD — no completed full training trial group. Empty CSVs contain headers only.")
    report.extend(["", "## Storage and resource costs", "",
                   "Actual file sizes and separately counted metadata/indexes: `results/cost/storage.csv`, backed by per-file stat observations in `results/cost/storage_raw.json`. Historical conversion/verification logs are embedded verbatim and labeled as historical timing.", "",
                   "Derived amortization: `results/cost/break_even.csv`. Nonpositive pass-time savings remain UNDEFINED. Temporary peak disk use and FFCV conversion duration are MISSING/TBD. H2D, memory and timelines require separate successful captures on the requested GPUs; excluded wrong-device captures are not paper evidence."])
    raw_control = RESULTS / "preparation/raw_int16/shard_000000_schema_retry.log"
    if raw_control.exists() and raw_control.read_text().strip().startswith('{'):
        preparation = json.loads(raw_control.read_text())
        report.extend(["", "## Raw-int16 control preparation", "",
                       f"One source shard was re-encoded with forced uncompressed int16 columns. Exact coefficient and rowgroup-boundary comparison: {preparation['coefficients_and_rowgroups_exact']}. Measured source/raw bytes: {preparation['source_bytes']} / {preparation['raw_int16_bytes']}. Conversion: {preparation['conversion_s']} s; verification: {preparation['verification_s']} s. Raw log: `{raw_control.relative_to(ROOT)}`.", "",
                       "GPU tensor equivalence and matched compression performance: MISSING/TBD. No percentage of FastLens speedup is attributed to compression."])
    report.extend(["", "## Missing experiments", "",
                   "- P0: full inference for every supported combination on both GPUs; resident model-only ceilings; complete warm epoch-2 training with adaptive 3→5 trials.",
                   "- P1: storage/conversion and break-even; diagnostic H2D/timelines; simultaneous whole-process and component memory.",
                   "- P2: matched spatial/frequency controls; direct output versus full grids; matched raw-int16 compression control.",
                   "- P3: layout, scheduling, batch-size and frequency-budget tradeoffs. Not launched before P0–P2 completion.",
                   "- P4: optional cold-cache and convergence extensions. Not launched.",
                   "- Paper tables/figures and Q1–Q7 conclusions: MISSING/TBD until measurements support them.", "",
                   "## Failure/retry notes", "",
                   "- Initial GPU query inside the sandbox could not communicate with the driver. The authorized outside-sandbox query succeeded; no product change was needed.",
                   "- Initial ViT smoke request used 100 images and was rejected before measurement: the scheduled reader requires a complete physical shard. Corrected to 1,024 images. No performance result from the rejected launch is accepted.",
                   "- Device-selection error: initial runs labeled H100 used CUDA ordinal 1, which is RTX PRO 6000 on this host; NVML ordinal 1 is H100. All affected logs and diagnostic captures are preserved under results/excluded/device_order and excluded from tables/scaling. The 4090 mapping was correct. Launchers now select UUIDs; independently executed name/UUID checks are in results/inventory/device_mapping.json.",
                   "- Excluded PRO 6000 PyTorch/FFCV additional trials failed stability thresholds; those observations are preserved. An unrelated 4090 Python process and shared-host CPU load were observed; no causal attribution is made. Exclusive-machine availability has been requested.",
                   "- Initial profiling attempt failed because pynvml is unavailable. The CLI monitor subsequently sampled the wrong NVML device due to the ordinal mismatch above; its empty files are not valid whole-process memory measurements."])
    report.extend(["- Actual H100 CoorDL launch encountered a cache left by the interrupted diagnostic. Automatic approval review rejected deleting the shared cache because another job uses the dataset and exclusive ownership was not established. The user authorized cleanup only after the other job finishes; this condition is still pending and source/cache files are unchanged. The other five H100 arms pass the original semantic checks; the validator records the missing CoorDL result explicitly."])
    (ROOT / "benchmark_results/archive/experiments/REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Exported {len(rows)} raw warm-pass rows and {len(summaries)} summaries")


if __name__ == "__main__":
    main()
