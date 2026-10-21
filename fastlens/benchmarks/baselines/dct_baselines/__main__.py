"""python -m fastlens.benchmarks.baselines.dct_baselines prepare/build/verify/bench/report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import subprocess

from .contracts import VERSION, artifact, load_artifact, write_json
from .materialize import MODELS, operation_order, resolve_profile, source_diagonal_profile
from .runner import benchmark, inputs, report, verify
from .source import freeze_source
from .store import METHODS, DEFAULT_DECODE_BUDGET, DEFAULT_MEMORY_BUDGET, build_store, storage_bytes


def prepare(args):
    if args.batch_size <= 0 or args.pool_size < args.batch_size or args.pool_size % args.batch_size:
        raise ValueError("pool size must be a positive multiple of model batch size")
    if min(args.workers, args.torch_threads) <= 0 or any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        raise ValueError("invalid worker/thread/tolerance configuration")
    if args.training and (args.epochs <= 0 or args.accumulation <= 0):
        raise ValueError("training epochs/accumulation must be positive")
    diagonals = getattr(args, "source_diagonals", 15)
    if args.training and diagonals != 15:
        raise ValueError("source-diagonals currently supports checkpoint inference; compact training uses its own representation")
    args.output.mkdir(parents=True, exist_ok=False)
    if getattr(args, "source_manifest", None) is not None:
        if args.count is not None or args.source_recipe is not None:
            raise ValueError("an existing frozen source keeps its complete population and producer recipe")
        source_ref = artifact(args.source_manifest)
        source = load_artifact(source_ref)
    else:
        if args.source_recipe is None:
            raise ValueError("a new source selection requires --source-recipe")
        recipe = json.loads(args.source_recipe.read_text())
        source = freeze_source(args.selection, args.output / "source.json", recipe, args.count,
                               workers=getattr(args, "source_workers", 1))
        source_ref = artifact(args.output / "source.json")
    profile = source_diagonal_profile(resolve_profile(args.model, args.upstream, args.checkpoint), diagonals)
    if args.training:
        if not 0 < args.calibration_count < len(source["images"]):
            raise ValueError("training requires a positive calibration-count smaller than the source population")
        profile["training"] = True
        profile["checkpoint_role"] = "inference reference only; training starts from the separately hashed complete snapshot"
        if args.model == "swinv2":
            profile["output_grids"] = [28, 14, 14]
            profile["tensors"][0]["shape"] = [1, 28, 28, 8, 8]
            profile["tensors"][1]["shape"] = [2, 14, 14, 8, 8]
        profile["operation_order"] = operation_order(profile)
        from fastlens.benchmarks.training_pls import published_augmentation, published_optimizer, recipe as training_recipe
        from fastlens.benchmarks.baselines.system_rgbnomore.training import model_factory
        profile["training_dependencies"] = [artifact(module.__file__) for module in
                                            (published_augmentation, published_optimizer, training_recipe, model_factory)]
    write_json(args.output / "profile.json", profile)
    # Import the repository's checkpoint label policy from the full population,
    # before choosing a small development subset.
    population = load_artifact(source["selection"])["samples"]
    classes = sorted({row["logical_sample_id"].split("/")[1] for row in population})
    class_labels = {name: index for index, name in enumerate(classes)}
    order = list(source["images"])
    request_order = getattr(args, "request_order", "shuffle")
    if request_order == "shuffle":
        random.Random(args.seed).shuffle(order)
    elif args.training:
        raise ValueError("source order is an inference protocol; training retains its frozen shuffle")
    labels = {image["image_id"]: class_labels[image["image_id"].split("/")[1]]
              if profile["label_policy"] == "sorted_class_names" else image["label"] for image in order}
    mapping = dict(version=VERSION, policy=profile["label_policy"], classes=class_labels, labels=labels)
    write_json(args.output / "labels.json", mapping)
    if args.training:
        from .training import freeze_trace
        for name, selected in (("calibration_trace", order[:args.calibration_count]), ("trace", order[args.calibration_count:])):
            rows = freeze_trace(selected, profile, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                                pool_size=args.pool_size, labels=labels, partition=name)
            with (args.output / f"{name}.jsonl").open("w") as stream:
                stream.writelines(json.dumps(row, sort_keys=True)+"\n" for row in rows)
    else:
        with (args.output / "trace.jsonl").open("w") as stream:
            for slot, image in enumerate(order):
                row = dict(occurrence_id=f"0:{slot}:0", image_id=image["image_id"], epoch=0, view_id=0,
                           logical_slot=slot, microbatch_id=slot//args.batch_size,
                           crop_box=profile["default_crop"], resize_geometry=profile["output_grids"], flip=False,
                           augmentation_parameters={}, mixup_partner_occurrence=None, mixup_lambda=1.,
                           target_parameters={}, target_label=labels[image["image_id"]], profile_id=profile["id"])
                stream.write(json.dumps(row, sort_keys=True)+"\n")
    config = dict(version=VERSION, source_manifest=source_ref,
                  model_profile=artifact(args.output / "profile.json"), request_trace=artifact(args.output / "trace.jsonl"),
                  label_mapping=artifact(args.output / "labels.json"),
                  repository_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  model_microbatch_size=args.batch_size, preparation_pool_size=args.pool_size,
                  workers=args.workers, torch_threads=args.torch_threads, precision="fp32", tail_policy="keep",
                  cpu_loader=getattr(args, "cpu_loader", "threadpool"), prefetch_factor=getattr(args, "prefetch_factor", 2),
                  trace_generation_seed=args.seed, request_order=request_order, tolerance=dict(atol=args.atol, rtol=args.rtol,
                  rationale="predeclared input/logit FP32 thresholds; exact by default; nonzero limits need independent justification, never post-failure widening"))
    config.update(decode_budget_bytes=args.decode_budget_mib*1024**2, reader_memory_budget_bytes=args.reader_budget_mib*1024**2,
                  host_input_budget_bytes=args.host_input_budget_mib*1024**2,
                  decoded_cache_bytes=getattr(args, "decoded_cache_mib", 0)*1024**2)
    if args.training:
        from fastlens.benchmarks.training_pls.recipe import RECIPE_NAME, SWINV2_RECIPE_NAME
        precision = "fp32" if args.model == "vitti" else "bf16-autocast"
        from .training import updates_per_epoch
        per_epoch = updates_per_epoch(len(population), args.batch_size, args.pool_size, args.accumulation)
        total = args.total_updates if args.total_updates is not None else per_epoch*300
        if args.epochs <= 0 or args.accumulation <= 0 or not 0 <= args.warmup_updates < total:
            raise ValueError("invalid training epochs/accumulation/schedule")
        config.update(precision=precision, calibration_trace=artifact(args.output / "calibration_trace.jsonl"),
                      training=dict(recipe=RECIPE_NAME if args.model == "vitti" else SWINV2_RECIPE_NAME,
                                    precision=precision, accumulation=args.accumulation, total_updates=total,
                                    warmup_updates=args.warmup_updates, population_size=len(population),
                                    updates_per_epoch=per_epoch, total_updates_override=args.total_updates is not None,
                                    accumulation_closes_at="pool_and_epoch", target_effective_batch=args.batch_size*args.accumulation,
                                    schedule_epochs=300, initialization="from_scratch", compile=False))
    write_json(args.output / "matched.json", config)
    # Resolve every profile/trace field now, not when a measurement has begun.
    inputs(args.output / "matched.json")
    return dict(contract=str(args.output / "matched.json"), source_images=len(order), model=args.model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="freeze source q/Q identities, real profile and occurrence trace")
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--selection", type=Path, help="existing repository sample manifest")
    sources.add_argument("--source-manifest", type=Path, help="reuse a frozen source and its existing IC/FS stores")
    p.add_argument("--source-recipe", type=Path, help="actual common JPEG preprocessing; required for a new selection")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--upstream", type=Path, required=True, help="RGB-no-more source root")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--source-diagonals", type=int, choices=range(1, 16), default=15,
                   help="retain complete B8 low-frequency diagonals and zero-fill source q; 15 preserves all64")
    p.add_argument("--count", type=int)
    p.add_argument("--source-workers", type=int, default=8, help="parallel JPEG coefficient extraction when freezing a new source")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--pool-size", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--cpu-loader", choices=("threadpool", "dataloader"), default="threadpool")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader tasks prefetched per worker; each task is a preparation pool")
    p.add_argument("--torch-threads", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--request-order", choices=("shuffle", "source"), default="shuffle",
                   help="source traverses stored image groups without the pilot's global shuffle; inference only")
    p.add_argument("--atol", type=float, default=0.)
    p.add_argument("--rtol", type=float, default=0.)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training", action="store_true", help="freeze published augmentation, Mixup and training recipe")
    p.add_argument("--epochs", type=int, default=10, help="finite training trace epochs; no implicit cycling")
    p.add_argument("--calibration-count", type=int, default=1, help="disjoint source images reserved for numerical calibration")
    p.add_argument("--accumulation", type=int, default=16)
    p.add_argument("--total-updates", type=int, help="explicit schedule override for diagnostic fixtures")
    p.add_argument("--warmup-updates", type=int, default=10000)
    p.add_argument("--decode-budget-mib", type=int, default=DEFAULT_DECODE_BUDGET//1024**2)
    p.add_argument("--reader-budget-mib", type=int, default=DEFAULT_MEMORY_BUDGET//1024**2)
    p.add_argument("--host-input-budget-mib", type=int, default=4096)
    p.add_argument("--decoded-cache-mib", type=int, default=0, help="per-reader decoded cache, included in reader and host budgets")
    p = commands.add_parser("build", help="stream source images into an adapted IC/FS store")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=[m for m in METHODS if m != "fs_native"], required=True)
    p.add_argument("--group-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8, help="offline build processes; each extracts and compresses a whole storage group")
    p.add_argument("--dtype-policy", choices=("paper", "int16", "safe"), default="paper")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("snapshot", help="freeze complete initial training state, including optimizer/schedule/decay/RNG")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for command in ("calibrate", "verify-training"):
        p = commands.add_parser(command, help="paired published-recipe G1/G2/G3 updates against an independent CPU-DCT oracle")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--snapshot", type=Path, required=True)
        p.add_argument("--backend", choices=("cpu_dct", "fastlens", "ic", "fs"), required=True)
        p.add_argument("--root", type=Path, help="IC/FS store directory or FastLens manifest.bin; unused for cpu_dct")
        p.add_argument("--module-path", type=Path)
        p.add_argument("--device", default="cpu", help="cpu or an explicit CUDA device, e.g. cuda:0")
        p.add_argument("--updates", type=int, default=10)
        p.add_argument("--source-workers", type=int, default=1, help="independent CPU readers for FastLens source q/Q verification")
        p.add_argument("--output", type=Path, required=True)
        if command == "calibrate":
            p.add_argument("--input-atol", type=float, required=True)
            p.add_argument("--model-atol", type=float, required=True)
            p.add_argument("--state-atol", type=float, required=True)
            p.add_argument("--rtol", type=float, required=True)
        else:
            p.add_argument("--calibration", type=Path, required=True)
    p = commands.add_parser("verify-inference", help="pretrained checkpoint equivalence on CPU baseline/native CUDA paths")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu_dct", "fastlens", "ic", "fs"), required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--module-path", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--source-workers", type=int, default=1)
    p.add_argument("--source-evidence", type=Path, help="reuse an unchanged passed source q/Q gate; final inputs/logits are still checked in full")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("verify", help="G1/G2 and optional checkpoint/backward semantic checks")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--check-model", action="store_true")
    p.add_argument("--training-steps", type=int, default=0, help="SGD semantic smoke; not a full published-recipe epoch")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("bench", help="diagnostic closed-window replay, never auto-qualified for a headline table")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--store", "--root", dest="store", type=Path, help="IC/FS directory or native manifest; unused for cpu_dct")
    p.add_argument("--backend", choices=("cpu_dct", "fastlens", "ic", "fs"), help="default: existing IC/FS store replay")
    p.add_argument("--module-path", type=Path)
    p.add_argument("--source-workers", type=int, default=1)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--device", default="cpu", help="cpu or an explicit CUDA device, e.g. cuda:0")
    p.add_argument("--run-kind", choices=("input_only", "inference"), default="input_only")
    p.add_argument("--cache-bytes", type=int, help="override the contract's per-reader decoded cache; IC/FS only")
    p.add_argument("--cold-cache", action="store_true", help="evict and verify input file pages before each dataset pass")
    p.add_argument("--min-seconds", type=float, default=0., help="repeat complete dataset passes until this timed duration is reached")
    p.add_argument("--warmup-seconds", type=float, default=0., help="excluded warmup before measuring; complete dataset passes")
    p.add_argument("--observe-cache", action="store_true", help="observe actual input page residency before the measured window")
    p.add_argument("--profile-stages", action="store_true", help="separate profiling run; collect CPU stage durations and CUDA events")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True, help="new JSONL results file; per-run counters are saved alongside")
    p = commands.add_parser("report", help="validate the supplied draft result schema and summarize raw repeats")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("reader-bench", help="CPU reader-only tuning; independent source/shuffle trace, no model work")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "build":
        config, source, profile, _, _ = inputs(args.config)
        fixed = args.method == "ic_matched_fixedk"
        from fastlens.benchmarks.dct_models.measure_process import ProcessWindow
        with ProcessWindow() as monitor:
            store = build_store(source, args.output, args.method, group_size=args.group_size, workers=args.workers,
                                source_frequencies=profile["source_frequencies"] if fixed else None,
                                dependency_evidence=profile["dependency_evidence"] if fixed else None,
                                dtype_policy=args.dtype_policy, source_artifact=config["source_manifest"])
        result = dict(method=store["method"], workers=args.workers, storage=storage_bytes(args.output),
                      conversion_wall_s=store["conversion_wall_s"], conversion_cpu_s=store["conversion_cpu_s"],
                      resources=monitor.result)
    elif args.command == "reader-bench":
        from .runner import benchmark_reader
        if args.output.exists():
            raise FileExistsError(args.output)
        result = benchmark_reader(args.config, args.root, repeats=args.repeats)
        write_json(args.output, result)
    elif args.command == "verify-inference":
        from .gates import verify_inference
        if args.backend != "cpu_dct" and args.root is None:
            parser.error("this backend requires --root")
        result = verify_inference(args.config, args.backend, args.root, args.output, device=args.device,
                                  module_path=args.module_path, source_workers=args.source_workers,
                                  source_evidence=args.source_evidence)
    elif args.command == "verify":
        if args.training_steps < 0:
            parser.error("training-steps cannot be negative")
        result = verify(args.config, args.store, args.output, device=args.device,
                        check_model=args.check_model, training_steps=args.training_steps)
    elif args.command == "bench":
        if args.repeats < 1:
            parser.error("repeats must be positive")
        if args.backend != "cpu_dct" and args.store is None:
            parser.error("this backend requires --root/--store")
        if args.source_workers < 1:
            parser.error("source-workers must be positive")
        with args.output.open("x") as stream:
            for run in range(args.repeats):
                row, detail = benchmark(args.config, args.store, args.evidence, device=args.device,
                                        run_kind=args.run_kind, cache_bytes=args.cache_bytes, backend=args.backend,
                                        module_path=args.module_path, source_workers=args.source_workers,
                                        min_seconds=args.min_seconds, cold_cache=args.cold_cache, warmup_seconds=args.warmup_seconds,
                                        observe_cache=args.observe_cache, profile_stages=args.profile_stages)
                detail_path = args.output.with_name(f"{args.output.stem}.run{run}.counters.json")
                write_json(detail_path, detail)
                if args.cold_cache or args.observe_cache:
                    row["artifacts"]["cache_evidence"] = artifact(detail_path)
                stream.write(json.dumps(row)+"\n")
                stream.flush()
        result = dict(results=str(args.output), repeats=args.repeats, headline_eligible=False)
    elif args.command == "snapshot":
        from .gates import create_snapshot
        result = create_snapshot(args.config, args.output)
    elif args.command in ("calibrate", "verify-training"):
        from .gates import verify_training
        if args.backend != "cpu_dct" and args.root is None:
            parser.error("this backend requires --root")
        tolerance = None
        if args.command == "calibrate":
            values = (args.input_atol, args.model_atol, args.state_atol, args.rtol)
            if any(not math.isfinite(v) or v < 0 for v in values):
                parser.error("calibration limits must be finite and nonnegative")
            tolerance = {key: dict(atol=value, rtol=args.rtol) for key, value in
                         (("inputs", args.input_atol), ("model", args.model_atol), ("state", args.state_atol))}
        result = verify_training(args.config, args.snapshot, args.backend, args.root, args.output,
                                 device=args.device, module_path=args.module_path, updates=args.updates,
                                 source_workers=args.source_workers,
                                 calibration_path=getattr(args, "calibration", None), tolerance=tolerance)
    else:
        result = report(args.results, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
