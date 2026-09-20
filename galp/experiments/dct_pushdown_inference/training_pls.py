"""DCTNet training on the native PLS data plane; A0/B6 share model and augmentation math."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

import backend as B
from capture import Capture

sys.path.insert(0, str(B.REPO / "build/galp/torch"))
sys.path.insert(0, str(B.REPO))
sys.path.insert(0, str(B.REPO / "galp/benchmarks/system_dct_major"))
from galp.torch.experimental import DirectDctPlsPipeline
from galp.benchmarks.training_audit_policy import build_audit_policy, decision_for_next_update
from training_pls.published_optimizer import build_published_optimizer
from training_pls.recipe import SWINV2_RECIPE_NAME, recipe_contract
from galp.benchmarks.system_rgbnomore.training.model_factory import capture_rng_state, restore_rng_state


def channels():
    pairs = [(c, f) for c, indices in enumerate(B.INDICES) for f in indices]
    return [[c, f, mean, std] for (c, f), mean, std in
            zip(pairs, B.MEAN.flatten().tolist(), B.STD.flatten().tolist())]


@torch.inference_mode()
def validate(net, count, root):
    from evaluate import samples
    from evaluate_shards import native_options, parse_manifest
    import _galp_direct_dct as native
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(root / "access")
    selected = {s["galp_image_id"]: s["model_label"] for s in samples(count)}
    stored = json.loads((root / "samples.json").read_text())
    manifest = parse_manifest(root / "manifest.bin")
    reader = native.DirectDctReader(str(root / "manifest.bin"))
    total = top1 = top5 = 0
    ce = 0.
    net.eval()
    started = time.perf_counter()
    for shard in manifest["shards"]:
        first = shard["first_global_image_index"]
        positions = list(range(first, first + shard["image_count"]))
        local = [i for i, position in enumerate(positions) if stored[position]["galp_image_id"] in selected]
        if not local:
            continue
        batch = reader.read_batch(positions, **native_options(pushdown=True, projected=True))
        x = batch.projected
        for offset in range(0, len(local), 64):
            indices = local[offset:offset + 64]
            target = torch.tensor([selected[stored[first+i]["galp_image_id"]] for i in indices], device="cuda")
            logits = net(x[indices])
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("non-finite validation logits")
            rank = logits.topk(5, dim=1).indices
            top1 += int((rank[:, 0] == target).sum())
            top5 += int((rank == target[:, None]).any(1).sum())
            ce += float(torch.nn.functional.cross_entropy(logits, target, reduction="sum"))
            total += len(indices)
        del x, batch
    torch.cuda.synchronize()
    assert total == count
    net.train()
    return dict(samples=total, top1=100*top1/total, top5=100*top5/total, ce=ce/total,
                seconds=time.perf_counter()-started, precision="fp32", input="N stored reference target, fixed validation subset")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--mapping-sha256", required=True)
    p.add_argument("--condition", choices=["A0", "B6", "RGB"], required=True)
    p.add_argument("--input-backend", choices=["native", "jpeg", "rgb_pytorch", "rgb_d2", "rgb_d3"], default="native")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dali-prefetch-depth", type=int, choices=[2, 4], default=2)
    p.add_argument("--segments-per-pool", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--transform-blocks-per-launch", type=int, default=0, help="0 retains the registered runtime policy")
    p.add_argument("--transform-ctas-per-launch", type=int, default=0, help="0 retains the registered runtime policy")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead"], default="default")
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--max-pools", type=int, default=0, help="bounded diagnostic, not a full epoch")
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--validation-count", type=int, default=50000)
    p.add_argument("--validation-data", type=Path, default=B.DEFAULT_E2E_V3_ROOT /
                   ("dct_major_dctnet_mobilenet32" if B.PROFILE.startswith("mobilenet") else "dct_major_dctnet_static64"))
    p.add_argument("--profile", action="store_true", help="capture 16384 images at complete pool boundaries")
    p.add_argument("--profile-warmup-images", type=int, default=4096)
    p.add_argument("--resume", type=Path, help="completed-epoch checkpoint; training restarts at the next epoch")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--initial-validation", action="store_true", help="evaluate scratch initialization before epoch 1")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    pool_images = args.segments_per_pool * 1024
    if pool_images <= 0:
        p.error("segments-per-pool must be positive")
    if args.profile and (args.profile_warmup_images < 0 or
                         args.profile_warmup_images % pool_images or 16384 % pool_images):
        p.error("profile warmup and the 16384-image capture must contain complete pools")
    initialization_started = time.perf_counter()
    rgb_backend = args.input_backend.startswith("rgb_")
    if rgb_backend != (args.condition == "RGB"):
        raise ValueError("RGB condition requires an RGB input backend")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "training.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    torch.set_num_threads(args.torch_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(11997733)
    print(json.dumps(dict(stage="initialization", device=torch.cuda.get_device_name(), profile=B.PROFILE)), flush=True)
    source_path = B.DEFAULT_E2E_V3_ROOT / "training_manifests_official_v3/train.json"
    source = json.loads(source_path.read_text())
    names = sorted({s["logical_sample_id"].split("/")[1] for s in source["samples"]})
    old_by_name = {s["logical_sample_id"].split("/")[1]: s["label"] for s in source["samples"]}
    if sorted(old_by_name.values()) != list(range(1000)):
        raise ValueError("training manifest labels are not a class permutation")
    label_columns = torch.tensor([old_by_name[n] for n in names], device="cuda")
    rgb_labels = label_columns.argsort()
    population = source["population_count"]
    del source
    recipe = recipe_contract(SWINV2_RECIPE_NAME)
    microbatch, accumulation = recipe["training"]["physical_microbatch"], recipe["training"]["gradient_accumulation"]
    effective = microbatch * accumulation
    audit = build_audit_policy()
    net = optimizer = decayer = scheduler = execution = None
    if not args.data_only:
        if rgb_backend:
            from rgb import model
        else:
            model = B.model
        net = model().requires_grad_(True).train()
        # Every parameter-bearing official CNN layer is reset; checkpoint values do not initialize training.
        torch.manual_seed(11997733)
        for module in net.modules():
            if list(module.parameters(recurse=False)):
                module.reset_parameters()
        net.cuda()
        if args.compile_mode == "reduce-overhead" and not args.no_compile:
            # Accumulation must not retain gradients owned by a previous graph replay.
            for param in net.parameters():
                param.grad = torch.zeros_like(param)
        optimizer, decayer, scheduler = build_published_optimizer(net, total_updates=math.ceil(population / effective) * 300)
        execution = net if args.no_compile else torch.compile(net, **{
            k: args.compile_mode if k == "mode" else v
            for k, v in recipe["execution"]["model_compile"].items() if k != "enabled"})
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(args.manifest.parent / "block_major_access_v1")
    if rgb_backend:
        from training_rgb import RgbTrainingPipeline
        pipeline = RgbTrainingPipeline(variant=args.input_backend.removeprefix("rgb_"), workers=args.workers,
            microbatch=microbatch, pool_images=args.segments_per_pool*1024,
            dali_prefetch_depth=args.dali_prefetch_depth)
    elif args.input_backend == "jpeg":
        if args.condition != "A0":
            raise ValueError("JPEG reference uses A0 per-sample crop and global shuffle")
        from training_jpeg import JpegTrainingPipeline
        pipeline = JpegTrainingPipeline(args.mapping, output_channels=channels(), grid=B.GRID,
            workers=args.workers, microbatch=microbatch, pool_images=args.segments_per_pool*1024)
    else:
        pipeline = DirectDctPlsPipeline(args.manifest, args.mapping, training_seed=11997733,
            expected_mapping_sha256=args.mapping_sha256,
            crop_policy="per-sample" if args.condition == "A0" else "per-pls",
            order_policy="global" if args.condition == "A0" else "closed-pool",
            segments_per_pool=args.segments_per_pool, microbatch_images=microbatch,
            output_grid_size=B.GRID, output_channels=channels(), transform_blocks_per_launch=args.transform_blocks_per_launch,
            transform_ctas_per_launch=args.transform_ctas_per_launch)
    if pipeline.sample_count != population:
        raise ValueError("physical data and training manifest population differ")
    monitor_file = (args.output_dir / "gpu_process_memory.csv").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory",
                                "--format=csv,noheader,nounits", "-lms", "100"], stdout=monitor_file)
    records = []
    tree_rss_peak = 0
    torch.cuda.reset_peak_memory_stats()
    complete_epochs = 0
    if args.resume:
        # RNG states must remain CPU byte tensors; optimizer.load_state_dict moves its tensors to parameter devices.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["profile"] != B.PROFILE or state["condition"] != args.condition or state["segments_per_pool"] != args.segments_per_pool:
            raise ValueError("resume profile, condition, or pool size differs")
        if state.get("input_backend", "native") != args.input_backend:
            raise ValueError("resume input backend differs")
        net.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        decayer.load_state_dict(state["decayer"])
        scheduler.load_state_dict(state["scheduler"])
        complete_epochs = state["completed_epochs"]
        restore_rng_state(state["rng_state"])
        del state
    initialization_seconds = time.perf_counter()-initialization_started
    try:
        if args.initial_validation and net is not None and not args.resume:
            if rgb_backend:
                from training_rgb import validate_rgb
                initial_validation = validate_rgb(net, args.validation_count, args.workers)
            else:
                initial_validation = validate(net, args.validation_count, args.validation_data)
                os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(args.manifest.parent / "block_major_access_v1")
            (args.output_dir / "initial_validation.json").write_text(json.dumps(initial_validation, indent=2) + "\n")
        for epoch in range(complete_epochs, args.epochs):
            finite = torch.ones((), device="cuda", dtype=torch.bool)
            loss_sum = torch.zeros((), device="cuda")
            seen = np.zeros(population, dtype=np.bool_)
            count, model_ms, input_wait = 0, 0., 0.
            pool_records = []
            capture = Capture(args.profile, first=args.profile_warmup_images // pool_images,
                              steps=16384 // pool_images, images_per_step=pool_images)
            torch.cuda.synchronize()
            started = time.perf_counter()
            epoch_started_unix_seconds = time.time()
            pipeline.start_epoch(epoch)
            epoch_setup_seconds = time.perf_counter() - started
            while pipeline.has_next_pool:
                capture.step(len(pool_records))
                pool_input_wait_start = input_wait
                t = time.perf_counter()
                with torch.cuda.nvtx.range("input.pool_wait"):
                    pool = pipeline.next_pool()
                input_wait += time.perf_counter() - t
                events = []
                batches = iter(pool)
                while True:
                    t = time.perf_counter()
                    with capture.range("input.next_batch"):
                        batch = next(batches, None)
                    input_wait += time.perf_counter() - t
                    if batch is None:
                        break
                    ids = np.asarray(batch.global_image_ids)
                    if seen[ids].any() or len(np.unique(ids)) != len(ids):
                        raise ValueError("duplicate physical image position in epoch")
                    seen[ids] = True
                    x = batch.projected
                    target = rgb_labels[batch.targets] if rgb_backend else batch.targets[:, label_columns]
                    shape = (3, 224, 224) if rgb_backend else (len(B.CHANNELS), B.GRID, B.GRID)
                    if tuple(x.shape) != (batch.image_count, *shape):
                        raise ValueError(f"wrong model input shape {x.shape}")
                    count += batch.image_count
                    if args.data_only:
                        finite &= torch.isfinite(x).all() & torch.isfinite(target).all()
                    else:
                        batch_index = batch.microbatch_index_in_pool
                        if batch_index % accumulation == 0:
                            optimizer.zero_grad(set_to_none=args.compile_mode != "reduce-overhead" or args.no_compile)
                            lr = scheduler.prepare_next_update()
                            window_images = min(effective, pool.image_count - batch.pool_offset)
                            decision = decision_for_next_update(audit, scheduler.completed_updates)
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        with torch.cuda.nvtx.range("model.forward_loss_backward"):
                            with capture.range("detail.forward"):
                                if args.compile_mode == "reduce-overhead" and not args.no_compile:
                                    torch.compiler.cudagraph_mark_step_begin()
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    logits = execution(x)
                            with capture.range("detail.loss_checks"):
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    loss = torch.nn.functional.cross_entropy(logits, target)
                                ok = torch.isfinite(logits).all() & torch.isfinite(loss)
                                finite &= ok
                                if decision.synchronous_loss_logits:
                                    if not bool(ok):
                                        raise FloatingPointError("non-finite loss/logits")
                                    float(loss.detach())
                                loss_sum += loss.detach() * batch.image_count
                            with capture.range("detail.backward"):
                                (loss * (batch.image_count / window_images)).backward()
                        if (batch_index + 1) % accumulation == 0 or batch.is_pool_end:
                            with torch.cuda.nvtx.range("model.optimizer_audit"):
                                for param in net.parameters():
                                    if param.grad is not None:
                                        ok = torch.isfinite(param.grad).all()
                                        finite &= ok
                                        if decision.synchronous_gradients and not bool(ok):
                                            raise FloatingPointError("non-finite gradient")
                                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                                optimizer.step()
                                decayer.step(lr)
                                scheduler.complete_update()
                                if decision.synchronous_parameters and not all(bool(torch.isfinite(v).all()) for v in net.parameters()):
                                    raise FloatingPointError("non-finite parameter")
                        end.record()
                        events.append((begin, end))
                        del loss, logits
                    del x, target, batch
                with capture.range("detail.pool_sync"):
                    torch.cuda.synchronize()
                pool_model_ms = sum(a.elapsed_time(b) for a, b in events)
                model_ms += pool_model_ms
                pool_records.append(dict(pool_index=pool.pool_index, images=pool.image_count,
                                         completed_epoch_seconds=time.perf_counter()-started,
                                         input_wait_seconds=input_wait-pool_input_wait_start,
                                         model_stream_seconds=pool_model_ms/1000,
                                         native=dict(pool.execution_stats)))
                del pool
                pipeline.reclaim_finished_pools()
                root_process = psutil.Process()
                tree_rss = 0
                for process in [root_process, *root_process.children(recursive=True)]:
                    try:
                        tree_rss += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                tree_rss_peak = max(tree_rss_peak, tree_rss)
                print(json.dumps(dict(epoch=epoch, pools=len(pool_records), images=count,
                    seconds=time.perf_counter()-started, updates=0 if scheduler is None else scheduler.completed_updates,
                    torch_allocated=torch.cuda.memory_allocated(), device_used=torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])), flush=True)
                if args.max_pools and len(pool_records) >= args.max_pools:
                    break
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            if not bool(finite) or (net is not None and not all(bool(torch.isfinite(v).all()) for v in net.parameters())):
                raise FloatingPointError("epoch finite check failed")
            full = count == population
            if not args.max_pools and not full:
                raise ValueError(f"incomplete epoch: {count}/{population}")
            if full:
                complete_epochs += 1
            record = dict(epoch=epoch, samples=count, full_epoch=full, seconds=wall, images_per_second=count/wall,
                started_unix_seconds=epoch_started_unix_seconds,
                epoch_setup_seconds=epoch_setup_seconds,
                input_wait_seconds=input_wait, model_stream_seconds=model_ms/1000, ce=None if args.data_only else float(loss_sum)/count,
                optimizer_updates=0 if scheduler is None else scheduler.completed_updates,
                unique_samples=int(seen.sum()), pools=pool_records, native_prefetch=dict(pipeline.prefetch_stats))
            if net is not None:
                if rgb_backend:
                    from training_rgb import validate_rgb
                    record["validation"] = validate_rgb(net, args.validation_count, args.workers)
                else:
                    record["validation"] = validate(net, args.validation_count, args.validation_data)
                os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(args.manifest.parent / "block_major_access_v1")
            records.append(record)
            (args.output_dir / f"epoch_{epoch}.json").write_text(json.dumps(record, indent=2)+"\n")
            if full and net is not None:
                torch.save(dict(model=net.state_dict(), optimizer=optimizer.state_dict(), decayer=decayer.state_dict(),
                    scheduler=scheduler.state_dict(), completed_epochs=complete_epochs, profile=B.PROFILE, condition=args.condition,
                    input_backend=args.input_backend, segments_per_pool=args.segments_per_pool, rng_state=capture_rng_state()),
                    args.output_dir / f"epoch_{epoch}.pth")
            if args.max_pools:
                break
    finally:
        pipeline.close()
        torch.cuda.synchronize()
        monitor.terminate()
        monitor.wait()
        monitor_file.close()
    values = [int(line.split(",")[-1].strip()) for line in (args.output_dir / "gpu_process_memory.csv").read_text().splitlines()
              if line.split(",")[1].strip() == str(os.getpid())]
    result = dict(profile=B.PROFILE, condition=args.condition, device=torch.cuda.get_device_name(),
        device_uuid=str(torch.cuda.get_device_properties(0).uuid),
        backbone=("EfficientNet-B0" if rgb_backend else "eFUN") if B.PROFILE == "efun" else
                 ("MobileNetV2" if B.PROFILE.startswith("mobilenet") else "ResNet-50"), representation="RGB" if rgb_backend else "DCT",
        initialization="from scratch; official CNN layers reset with seed 11997733", full_training_epochs_completed=complete_epochs,
        resumed_from=str(args.resume) if args.resume else None,
        manifest=str(args.manifest) if args.input_backend == "native" else None,
        mapping=None if rgb_backend else str(args.mapping), training_manifest=str(source_path),
        input_backend=args.input_backend, workers=args.workers if args.input_backend != "native" else None,
        output_channels=None if rgb_backend else channels(), grid_size=224 if rgb_backend else B.GRID, segments_per_pool=args.segments_per_pool, segment_images=1024,
        transform_blocks_per_launch=args.transform_blocks_per_launch or 512,
        transform_ctas_per_launch=args.transform_ctas_per_launch or 512,
        compile_mode=args.compile_mode, torch_threads=args.torch_threads,
        microbatch=microbatch, accumulation=accumulation, precision="bf16-autocast", compile=not args.no_compile,
        data_only=args.data_only, audit_policy=audit, epochs=records, torch_peak_allocated=torch.cuda.max_memory_allocated(),
        initialization_seconds=initialization_seconds, sampled_process_tree_rss_peak_bytes=tree_rss_peak,
        cpu_tree_memory_scope="RSS summed at pool boundaries; includes workers but double-counts shared pages, not PSS",
        torch_peak_reserved=torch.cuda.max_memory_reserved(), nvml_process_peak_mib=max(values) if values else None,
        cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        largest_finished_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        timing="cold epoch E2E includes first compile and pool boundaries; CUDA model stream intervals include audit host gaps; overlapping stages are not additive",
        augmentation=("existing Transformer RGB crop/flip, bilinear resize 224, mean/std .5; hard labels, no DCT RandAugment/Mixup" if rgb_backend else
                      "source DCT crop/resize/flip; round to integer; RandAugment clamp [-1024,1016], two operations; spatial magnitudes scaled from 28-grid; channel normalization then Mixup"),
        validation=("per-epoch RGB center-crop validation" if rgb_backend else "per-epoch N validation") + "; bounded training probes are not convergence evidence")
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("epochs", "audit_policy", "output_channels")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
