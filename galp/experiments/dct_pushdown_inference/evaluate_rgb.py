"""Compare PyTorch/PIL and DALI input pipelines with one official RGB checkpoint."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import backend as B
import rgb
from evaluate import samples
from capture import Capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", choices=["pytorch", "dali"], required=True)
    p.add_argument("--count", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--dali-prefetch-depth", type=int, choices=[2, 4], default=2)
    p.add_argument("--model-threads", type=int, default=8)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.model_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    entries = samples(args.count)
    net = rgb.model().cuda()
    reference = rgb.Inputs(entries)
    with torch.inference_mode():
        warm = net(reference[0][0][None].cuda())
    assert warm.shape == (1,1000) and torch.isfinite(warm).all()
    torch.cuda.synchronize()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = Capture(args.profile, images_per_step=args.batch_size)
    predictions = [-1] * args.count
    read_work = decode_work = transform_work = read_bytes = 0
    top1 = top5 = count = 0
    ce = wait = handoff = model_time = init = 0.
    expected_ordinal = 0
    with torch.inference_mode():
        start = time.perf_counter()
        if args.route == "pytorch":
            loader = DataLoader(reference, batch_size=args.batch_size, num_workers=args.workers,
                                worker_init_fn=rgb.worker_init, pin_memory=True, shuffle=False,
                                prefetch_factor=2 if args.workers else None)
        else:
            loader = rgb.dali_loader(entries, args.batch_size, args.workers, args.dali_prefetch_depth)
        iterator = iter(loader)
        init = time.perf_counter()-start
        step = 0
        while True:
            capture.step(step)
            t = time.perf_counter()
            with capture.range("input.wait"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
            wait += time.perf_counter()-t
            t = time.perf_counter()
            with capture.range("input.handoff"):
                if args.route == "pytorch":
                    x, labels, ordinals, reads, decodes, transforms, sizes = batch
                    read_work += float(reads.sum())
                    decode_work += float(decodes.sum())
                    transform_work += float(transforms.sum())
                    read_bytes += int(sizes.sum())
                    ordinals = ordinals.tolist()
                    x, labels = x.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                else:
                    x = batch[0]["image"]
                    ordinals = batch[0]["ordinal"].reshape(-1).tolist()
                    labels = torch.tensor([entries[o]["model_label"] for o in ordinals], device="cuda")
                assert ordinals == list(range(expected_ordinal, expected_ordinal + len(ordinals)))
                expected_ordinal += len(ordinals)
                torch.cuda.current_stream().synchronize()
            handoff += time.perf_counter()-t
            t = time.perf_counter()
            with capture.range("model.forward"):
                logits = net(x)
                torch.cuda.current_stream().synchronize()
            model_time += time.perf_counter()-t
            with capture.range("metrics"):
                assert torch.isfinite(logits).all()
                ranked = logits.topk(5, dim=1).indices
                top1 += int((ranked[:,0] == labels).sum())
                top5 += int((ranked == labels[:,None]).any(dim=1).sum())
                ce += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
                for o, pred in zip(ordinals, ranked[:,0].tolist()):
                    predictions[o] = pred
            count += len(ordinals)
            step += 1
            if step % 64 == 0:
                print(f"RGB {args.route} {count}/{args.count}", flush=True)
        torch.cuda.synchronize()
        wall = time.perf_counter()-start
    assert count == args.count and min(predictions) >= 0
    result = dict(route="RGB_"+args.route, samples=count, top1=100*top1/count, top5=100*top5/count,
                  ce=ce/count, e2e_seconds=wall, initialization_seconds=init, input_wait_seconds=wait,
                  handoff_seconds=handoff, model_seconds=model_time, batch_size=args.batch_size,
                  workers=args.workers, model_threads=args.model_threads, precision="float32", tf32=False,
                  dali_prefetch_queue_depth=args.dali_prefetch_depth if args.route == "dali" else None,
                  device_name=torch.cuda.get_device_name(), model=type(net).__name__, checkpoint=str(rgb.CHECKPOINT),
                  source_root=str(B.DEFAULT_DATA_ROOT), source_version="imagenet_512",
                  preprocessing=dict(resize=256, crop=224, color="RGB", mean=rgb.MEAN, std=rgb.STD,
                                     interpolation=rgb.INTERPOLATION.value, antialias=True,
                                     note="Torchvision EfficientNet_B0_Weights.IMAGENET1K_V1" if B.PROFILE == "efun"
                                     else "legacy PyTorch RGB reference; no independent RGB eval script in DCTNet"),
                  read_worker_seconds=read_work if args.route=="pytorch" else None,
                  decode_worker_seconds=decode_work if args.route=="pytorch" else None,
                  transform_worker_seconds=transform_work if args.route=="pytorch" else None,
                  source_bytes=read_bytes if args.route=="pytorch" else sum(Path(e["path"]).stat().st_size for e in entries),
                  timing_scope="file input through logits and metrics; reader initialization included; model warmup excluded",
                  predictions=predictions)
    baseline = args.output_dir / f"RGB_pytorch_{count}.json"
    if args.route=="dali" and baseline.exists():
        previous = json.loads(baseline.read_text())
        result["prediction_agreement_with_pytorch"] = sum(a==b for a,b in zip(predictions,previous["predictions"]))/count
        result["top1_delta_pp_vs_pytorch"] = result["top1"]-previous["top1"]
    (args.output_dir/f"RGB_{args.route}_{count}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="predictions"},indent=2))


if __name__ == "__main__":
    main()
