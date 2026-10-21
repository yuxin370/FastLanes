"""Launch CNN B6 storage comparisons and reject incompatible speedup pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact
from fastlens.benchmarks.baselines.dct_baselines.store import fingerprint


def training_identity(args, *, profile, channels, grid, model_code, initial_model_hash, recipe, source, runtime):
    """Shared semantics; backend storage and worker implementation stay outside it."""
    return dict(
        source_manifest=artifact(source), mapping_sha256=args.mapping_sha256,
        condition=args.condition, seed=11997733,
        crop_policy="per-sample" if args.condition == "A0" else "per-pls",
        order_policy="global" if args.condition == "A0" else "closed-pool",
        profile=profile, channels=channels, grid=grid, model_code=model_code, initial_model_hash=initial_model_hash,
        recipe=recipe, precision="bf16-autocast", tf32=False,
        resumed=args.resume is not None, short_warmup=args.short_warmup,
        audit_policy=artifact(Path(__file__).parents[1]/"training_audit_policy.py"),
        training_driver=artifact(Path(__file__).with_name("training_pls.py")),
        compile=not args.no_compile, compile_mode=args.compile_mode,
        microbatch=recipe["training"]["physical_microbatch"],
        accumulation=recipe["training"]["gradient_accumulation"],
        pool_images=args.segments_per_pool*1024, epochs=args.epochs, max_pools=args.max_pools,
        data_only=args.data_only, runtime=runtime,
        augmentation=[artifact(Path(__file__).parents[1]/"training_pls"/name) for name in
                      ("published_augmentation.py", "core_schedule.py", "published_optimizer.py")],
    )


def check_pair(left, right):
    """Require new identities plus the actually consumed occurrence/pool schedule."""
    if "matched_identity" not in left or "matched_identity" not in right:
        raise ValueError("legacy results lack the matched identity; append them to the existing report without a strict paired speedup")
    a, b = left["matched_identity"], right["matched_identity"]
    required = {"source_manifest", "mapping_sha256", "condition", "seed", "crop_policy", "order_policy",
                "profile", "channels", "grid", "initial_model_hash", "recipe", "precision", "tf32",
                "compile", "compile_mode", "microbatch", "accumulation", "pool_images", "epochs", "max_pools",
                "data_only", "runtime", "augmentation", "resumed", "short_warmup", "audit_policy", "model_code", "training_driver"}
    for result, identity in ((left, a), (right, b)):
        if required - identity.keys():
            raise ValueError(f"matched identity missing fields: {sorted(required-identity.keys())}")
        if result["matched_identity_sha256"] != fingerprint(identity):
            raise ValueError("matched identity hash differs")
        if identity["initial_model_hash"] is None or identity["data_only"]:
            raise ValueError("matched speedup requires model training")
        if identity["resumed"]:
            raise ValueError("matched launcher requires fresh runs; resumed optimizer state is not bound here")
    if a != b:
        raise ValueError(f"matched identity differs: {sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))}")
    fields = ("epoch", "samples", "unique_samples", "full_epoch", "optimizer_updates", "sample_order_digest")
    def schedule(result):
        return [([r[k] for k in fields], [(p["pool_index"], p["images"]) for p in r["pools"]]) for r in result["epochs"]]
    if not left["epochs"] or schedule(left) != schedule(right):
        raise ValueError("matched consumed occurrence/pool/update schedule differs")
    return dict(status="matched", matched_identity_sha256=fingerprint(a), headline_eligible=False,
                claim_boundary="same declared recipe and consumed schedule; production numerical and global resource gates still required")


def launch(args):
    from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader
    variants = [("jpeg", "jpeg", None), ("native", "native", None),
                ("ic", "ic", args.ic_store), ("ic_batched", "ic", args.ic_batched_store),
                ("fs", "fs", args.fs_store)]
    for name, backend, store in variants:
        if store is not None:
            with StoreReader(store) as reader:
                expected = {"ic": "ic_matched_shared", "ic_batched": "ic_matched_batched", "fs": "fs_matched_b8"}[name]
                if reader.manifest["method"] != expected:
                    raise ValueError(f"{name} requires {expected}")
    for name, backend, store in variants:
        command = [sys.executable, "-B", "-m", "fastlens.benchmarks.dct_models.training_pls",
                   "--condition", "B6", "--input-backend", backend,
                   "--manifest", str(args.manifest), "--mapping", str(args.mapping),
                   "--mapping-sha256", args.mapping_sha256, "--segments-per-pool", "4",
                   "--workers", str(args.workers), "--epochs", str(args.epochs),
                   "--validation-count", str(args.validation_count), "--output-dir", str(args.output/name)]
        if args.max_pools:
            command += ["--max-pools", str(args.max_pools)]
        if store is not None:
            command += ["--baseline-store", str(store)]
        if args.plan:
            print(f"DCTNET_PROFILE={shlex.quote(args.profile)} {shlex.join(command)}")
        else:
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise ValueError("set CUDA_VISIBLE_DEVICES to the common GPU")
            subprocess.run(command, env=dict(os.environ, DCTNET_PROFILE=args.profile), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("launch")
    p.add_argument("--profile", choices=("mobilenet24", "mobilenet32", "resnet24", "resnet64"), required=True)
    for key in ("manifest", "mapping", "ic-store", "ic-batched-store", "fs-store", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--mapping-sha256", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-pools", type=int, default=0)
    p.add_argument("--validation-count", type=int, default=50000)
    p.add_argument("--plan", action="store_true")
    p = commands.add_parser("check")
    p.add_argument("training", type=Path, nargs=2, help="two completed training.json files")
    args = parser.parse_args()
    if args.command == "launch":
        launch(args)
    else:
        left, right = (json.loads(path.read_text()) for path in args.training)
        result = check_pair(left, right)
        result.update(training=[artifact(path) for path in args.training],
                      epoch_speedup_left_over_right=[a["seconds"]/b["seconds"] for a, b in zip(left["epochs"], right["epochs"])])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
