"""Matched resident-input model ceilings and warm-page-cache input pipelines.

Run with --suite for three independent trials and separate Nsight captures.
Training and inference are separate studies, not convergence experiments.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("mobilenet", "resnet", "vitti", "swinv2", "efficientnet")
ARMS = ("model-only", "pytorch", "dali")
DCT_MODELS = ("mobilenet24", "mobilenet32", "resnet24", "resnet64", "vitti-dct", "swinv2-dct",
              "efun", "efun-l", "efun-s", "efun-s-plus")
DCT_ARMS = ("model-only", "pytorch")
SEED = 11997733


def cache_pages(paths, warm=False):
    """Read source bytes when requested; query actual residency with mincore."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = os.sysconf("SC_PAGE_SIZE")
    resident = pages = size_bytes = 0
    for path in paths:
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            # Empty lock files have no cache pages; mmap rejects a zero length.
            if size == 0:
                continue
            if warm:
                while source.read(1024 * 1024):
                    pass
            n = (size + page - 1) // page
            address = libc.mmap(None, size, 0, 1, source.fileno(), 0)
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), f"mmap: {path}")
            vector = (ctypes.c_ubyte * n)()
            try:
                if libc.mincore(address, size, vector):
                    raise OSError(ctypes.get_errno(), f"mincore: {path}")
                resident += sum(value & 1 for value in vector)
            finally:
                libc.munmap(address, size)
            pages += n
            size_bytes += size
    return dict(files=len(paths), bytes=size_bytes, pages=pages,
                resident_pages=resident, resident_fraction=resident / pages if pages else None)


def run(args):
    import torch
    from fastlens.benchmarks.baselines.system_rgbnomore.training.augmentation import derive_augmentation
    from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import (
        TrainingSample, PyTorchTrainingAdapter, DaliTrainingAdapter, resolve_dali_variant,
    )
    from fastlens.benchmarks.baselines.system_rgbnomore.training.sample_order import SampleIdentity
    from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import (
        build_model, seed_everything, VITTI_MODEL_ID, SWINV2_T_MODEL_ID,
    )
    from fastlens.benchmarks.training_pls.published_optimizer import build_published_optimizer

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(SEED)
    device = torch.device("cuda:0")
    if "RTX 4090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("This experiment is specified for the RTX 4090")
    if args.domain == "dct":
        from fastlens.benchmarks.profiling.input_motivation_dct import build_dct_model, dct_adapter, preprocessing
        model = build_dct_model(args.model, device)
    elif args.model in ("mobilenet", "resnet"):
        os.environ["DCTNET_PROFILE"] = "mobilenet24" if args.model == "mobilenet" else "resnet24"
        from fastlens.benchmarks.dct_models.rgb import RGBMobileNetV2, RGBResNet
        model = (RGBMobileNetV2() if args.model == "mobilenet" else RGBResNet()).to(device)
    elif args.model == "efficientnet":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None).to(device)
    else:
        model = build_model((Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more'))), "rgb", device,
                            model_id=VITTI_MODEL_ID if args.model == "vitti" else SWINV2_T_MODEL_ID)
    model.train(args.mode == "training")
    precision = "fp32" if args.mode == "inference" or args.model in ("vitti", "vitti-dct") else "bf16-autocast"
    execute = torch.compile(model, backend="inductor", options={"triton.cudagraphs": False}, dynamic=False)
    if args.domain == "dct":
        compiled = execute
        execute = lambda inputs: compiled(*inputs)
    if args.mode == "training":
        optimizer, decay, scheduler = build_published_optimizer(model, total_updates=375600)
        # Keep gradient allocation and zeroing identical across all three arms.
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
    manifest = ROOT / "fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json"
    population = json.loads(manifest.read_text())["samples"]
    # Extra batches allow normal asynchronous prefetch at the timing boundary.
    images_per_update = 1024 if args.mode == "training" else 64
    count = (args.warmup + args.updates) * images_per_update + 256
    entries = random.Random(SEED).sample(population, count)
    del population
    samples = [TrainingSample(e["logical_sample_id"], Path(e["path"]), e["label"],
                              e["width"], e["height"], e["galp_image_id"]) for e in entries]
    paths = [s.path for s in samples]
    print(f"{args.model}/{args.arm}: warming {count} JPEG files", flush=True)
    cache_warm = cache_pages(paths, warm=True)
    if args.arm == "model-only":
        samples = samples[:64]
    identities = [SampleIdentity(2 if args.mode == "training" else 0, i, s.logical_sample_id)
                  for i, s in enumerate(samples)]
    if args.mode == "inference":
        from fastlens.benchmarks.baselines.system_rgbnomore.training.equal_image_epoch_benchmark import _center_validation_decision
        decisions = [_center_validation_decision(s) for s in samples]
    else:
        decisions = [] if args.arm == "dali" else [derive_augmentation(
            seed=SEED, epoch=2, logical_sample_id=s.logical_sample_id,
            source_width=s.width, source_height=s.height, domain=args.domain) for s in samples]
    variant = dict(resolve_dali_variant("d3" if args.mode == "training" else "d2"),
                   num_threads=16, prefetch_queue_depth=2, seed=SEED)
    if args.domain == "dct":
        adapter = dct_adapter(args.model, args.mode, samples, device)
    else:
        cls = DaliTrainingAdapter if args.arm == "dali" else PyTorchTrainingAdapter
        adapter = cls(samples, batch_size=64, workers=16, device=device,
                      config={"phase": "train" if args.mode == "training" else "validation", "dali": variant})
    adapter.begin(identities, decisions)
    if args.arm == "model-only":
        batch = adapter.next_batch()
        resident = (tuple(x.to(device) for x in batch.inputs) if args.domain == "dct"
                    else batch.inputs[0].to(device)), batch.labels.to(device)
        adapter.close()
        del batch
    finite = torch.ones((), device=device, dtype=torch.bool)

    def span(name):
        return torch.cuda.nvtx.range(name) if args.profile else contextlib.nullcontext()

    def training_update():
        nonlocal finite
        with span("training.optimizer.zero"):
            optimizer.zero_grad(set_to_none=False)
            learning_rate = scheduler.prepare_next_update()
        for _ in range(16):
            if args.arm == "model-only":
                x, target = resident
            else:
                with span("training.loader.next_batch"):
                    batch = adapter.next_batch()
                with span("training.input_handoff"):
                    x = (tuple(value.to(device, non_blocking=True) for value in batch.inputs)
                         if args.domain == "dct" else batch.inputs[0].to(device, non_blocking=True))
                    target = batch.labels.to(device, non_blocking=True)
            with span("training.model.forward"):
                with (torch.autocast("cuda", dtype=torch.bfloat16)
                      if precision != "fp32" else contextlib.nullcontext()):
                    logits = execute(x)
                    loss = torch.nn.functional.cross_entropy(logits, target)
            with span("training.model.backward"):
                (loss / 16).backward()
            with span("training.model.finite"):
                finite &= torch.isfinite(loss.detach())
            del logits, loss, x, target
            if args.arm != "model-only":
                del batch
        with span("training.optimizer.step"):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            finite &= torch.isfinite(norm)
            optimizer.step()
            decay.step(learning_rate)
            scheduler.complete_update()

    def inference_update():
        nonlocal finite
        if args.arm == "model-only":
            x, _ = resident
        else:
            with span("training.loader.next_batch"):
                batch = adapter.next_batch()
            with span("training.input_handoff"):
                x = (tuple(value.to(device, non_blocking=True) for value in batch.inputs)
                     if args.domain == "dct" else batch.inputs[0].to(device, non_blocking=True))
        with span("training.model.forward"):
            logits = execute(x)
        with span("training.model.finite"):
            finite &= torch.isfinite(logits).all()

    update = training_update if args.mode == "training" else torch.inference_mode()(inference_update)
    print(f"{args.model}/{args.arm}: compile and {args.warmup} warmup updates", flush=True)
    for _ in range(args.warmup):
        update()
    torch.cuda.synchronize()
    before = cache_pages(paths)
    if before["resident_pages"] != before["pages"]:
        raise RuntimeError(f"Source JPEG pages are not fully resident: {before}")
    print(f"{args.model}/{args.arm}: measuring {args.updates} updates", flush=True)
    if args.profile:
        torch.cuda.profiler.start()
    with span("motivation.capture"):
        started = time.perf_counter()
        for index in range(args.updates):
            with span(f"update.{index}"):
                update()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    if args.profile:
        torch.cuda.profiler.stop()
    after = cache_pages(paths)
    if not bool(finite.item()):
        raise FloatingPointError("Non-finite training loss or gradient norm")
    if args.arm != "model-only":
        adapter.close()
    result = dict(model=args.model, arm=args.arm, trial=args.trial, profile=args.profile, mode=args.mode, domain=args.domain,
                  seconds=seconds, images=args.updates * images_per_update,
                  images_per_second=args.updates * images_per_update / seconds,
                  batch=64, accumulation=16 if args.mode == "training" else 1, warmup_updates=args.warmup, updates=args.updates,
                  seed=SEED, precision=precision, compile="inductor/default", cuda_graphs=False, tf32=False,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  model_parameters=sum(p.numel() for p in model.parameters()),
                  pytorch_workers=16, dali=variant if args.domain == "rgb" else None, host_threads=8,
                  cache_warm=cache_warm, cache_before=before, cache_after=after,
                  initialization=f"same seeded random {args.domain.upper()} model in each arm",
                  training=("forward, CE, backward, clipping, AdamW, independent decay, scheduler; deferred loss/gradient-norm finite check"
                            if args.mode == "training" else "eval forward and deferred logit finite check; no backward or optimizer"),
                  scope=f"bounded warm-cache {args.mode}; excludes compilation, warmup, adapter setup/closure; not a complete epoch",
                  augmentation=(preprocessing(args.model, args.mode) if args.domain == "dct" else
                                "RGB random crop/flip, resize224, normalize .5; DALI D3 native order/crop; no pixel-equality claim"
                                if args.mode == "training" else "system validation: center square, bilinear resize224, normalize .5, no flip; planned DALI D2"),
                  command=sys.argv)
    if args.arm == "dali":
        import nvidia.dali
        result["dali_version"] = nvidia.dali.__version__
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("model", "arm", "images_per_second", "seconds")}), flush=True)


def suite(args):
    args.output.mkdir(parents=True, exist_ok=True)
    models = DCT_MODELS if args.domain == "dct" else MODELS
    arms = DCT_ARMS if args.domain == "dct" else ARMS
    tasks = [(model, arm, trial, False) for trial in range(3) for model in models
             for arm in arms[trial:] + arms[:trial]]
    trace_models = ("mobilenet24", "vitti-dct") if args.domain == "dct" else ("mobilenet", "vitti")
    tasks += [(model, arm, 0, True) for model in trace_models for arm in arms if arm != "model-only"]
    for model, arm, trial, profile in tasks:
        destination = args.output / (f"{model}-{arm}-trace" if profile else f"{model}-{arm}-r{trial}")
        if (destination / "result.json").exists() and (not profile or (destination / "trace.sqlite").exists()):
            continue
        destination.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "fastlens.benchmarks.profiling.input_motivation",
                   "--model", model, "--arm", arm, "--trial", str(trial),
                   "--domain", args.domain,
                   "--mode", args.mode,
                   "--warmup", str(args.warmup), "--updates", str((8 if args.mode == "training" else 128) if profile else args.updates),
                   "--output", str(destination)]
        if profile:
            command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                       "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                       "--force-overwrite=true", "--output", str(destination / "trace")] + command + ["--profile"]
        print(f"START {destination.name}", flush=True)
        with (destination / "run.log").open("w") as log, (destination / "gpu.csv").open("w") as telemetry:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            while True:
                subprocess.run(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
                                "--query-compute-apps=timestamp,pid,process_name,used_memory", "--format=csv,noheader"],
                               stdout=telemetry, stderr=telemetry, check=True)
                try:
                    status = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if status:
            raise RuntimeError(f"Experiment failed ({status}): {destination / 'run.log'}")
        if profile:
            subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "--output",
                            str(destination / "trace.sqlite"), str(destination / "trace.nsys-rep")], check=True)
        print(f"DONE {destination.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--model", choices=MODELS + DCT_MODELS)
    parser.add_argument("--domain", choices=("rgb", "dct"), default="rgb")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--mode", choices=("training", "inference"), default="training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.updates < 1:
        parser.error("warmup and updates must be positive")
    if not args.suite and (args.model is None or args.arm is None):
        parser.error("a single run requires --model and --arm")
    if args.model is not None and args.model not in (DCT_MODELS if args.domain == "dct" else MODELS):
        parser.error("model does not match the selected input domain")
    if args.domain == "dct" and args.arm == "dali":
        parser.error("the repository DALI adapter only supports RGB input; no DCT DALI arm is implemented")
    (suite if args.suite else run)(args)


if __name__ == "__main__":
    main()
