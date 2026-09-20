"""Frozen-checkpoint evaluation on the existing validation selection rule."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import backend as B
from capture import Capture, batches


def samples(count):
    manifest = json.loads((B.DEFAULT_E2E_V3_ROOT / "training_manifests_official_v3/val.json").read_text())
    population = manifest["samples"]
    classes = sorted({s["logical_sample_id"].split("/")[1] for s in population})
    labels = {name: index for index, name in enumerate(classes)}
    # Preserve stored IDs/labels; model_label is an explicit checkpoint mapping.
    order = list(range(len(population)))
    random.Random(11997733).shuffle(order)
    return [dict(population[i], ordinal=j,
                 model_label=labels[population[i]["logical_sample_id"].split("/")[1]])
            for j, i in enumerate(order[:count])]


class Inputs(Dataset):
    def __init__(self, entries, route="R", manifest=None, new_ids=None):
        self.entries = entries
        self.reference = None
        self.route, self.manifest, self.new_ids = route, manifest, new_ids

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if self.reference is None:
            torch.set_num_threads(1)
            if self.route == "R":
                self.reference = B.Reference()
            else:
                from storage import Reader
                from old_adapter import OldDCTAdapter
                from multiprocessing.util import Finalize
                self.reference = Reader(self.manifest)
                Finalize(self, self.reference.close, exitpriority=10)
                if self.route == "O":
                    self.adapter = OldDCTAdapter(Path("/home/tangyuxin/RGB-no-more"), target_grid=B.GRID)
                elif self.route == "S":
                    from online_crop import SourceReference
                    self.adapter = SourceReference(B.DEFAULT_RGBNOMORE_ROOT, B.GRID)
        entry = self.entries[index]
        start = time.perf_counter()
        if self.route == "R":
            x = self.reference(entry["path"])
            size = Path(entry["path"]).stat().st_size
            native, adaptation = 0., 0.
        else:
            image_id = entry["galp_image_id"]
            if self.route == "N":
                image_id = self.new_ids[image_id]
            q, qt = self.reference.read(image_id)
            native = self.reference.last_stats["read_decode_seconds"]
            size = self.reference.last_stats["payload_read_bytes"]
            t = time.perf_counter()
            components = self.adapter(q, qt) if self.route in ("O", "S") else [c.astype("float32") * table for c,table in zip(q,qt)]
            if self.route == "S":
                components = [c.round().clamp(-32768, 32767) for c in components]
            x = B.organize(components)
            adaptation = time.perf_counter()-t
        return x, entry["model_label"], time.perf_counter() - start, size, native, adaptation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-threads", type=int, default=8)
    parser.add_argument("--route", choices=["R", "N"] if B.PROFILE == "efun" else ["R", "N", "O", "S"], default="R")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--new-data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=B.DEFAULT_E2E_V3_ROOT / "runs" / B.RUN_NAME)
    args = parser.parse_args()
    torch.set_num_threads(args.model_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = samples(args.count)
    (args.output_dir / "profile.json").write_text(json.dumps(B.profile(), indent=2))
    (args.output_dir / f"samples_{args.count}.json").write_text(json.dumps(selected, indent=2))
    net = B.model().to(device)
    # Real source image warmup also checks storage target definition against R.
    ref = B.Reference()
    q, qt, components = ref.coefficients(selected[0]["path"])
    x = ref(selected[0]["path"])
    torch.testing.assert_close(x, B.organize(components), rtol=0, atol=0)
    with torch.inference_mode():
        warmup = net(x[None].to(device))
    assert tuple(warmup.shape) == (1, 1000) and torch.isfinite(warmup).all()
    manifest, new_ids = None, None
    if args.route == "N":
        if args.new_data is None:
            parser.error("N requires --new-data")
        manifest = args.new_data / "manifest.bin"
        new_ids = {s["galp_image_id"]:i for i,s in enumerate(json.loads((args.new_data/"samples.json").read_text()))}
    elif args.route in ("O", "S"):
        manifest = B.DEFAULT_GALP_MANIFEST
    loader = DataLoader(Inputs(selected, args.route, manifest, new_ids), batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, prefetch_factor=2 if args.workers else None,
                        pin_memory=device.type == "cuda")
    top1 = top5 = count = 0
    ce = input_work = model_time = read_bytes = 0
    native_time = adaptation_time = 0
    h2d_time = 0
    predictions = []
    capture = Capture(args.profile, images_per_step=args.batch_size)
    start = time.perf_counter()
    with torch.inference_mode():
        for x, labels, construction, sizes, native, adaptation in batches(loader, capture):
            if device.type == "cuda":
                t = time.perf_counter()
                with capture.range("input.handoff"):
                    x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    torch.cuda.synchronize()
                h2d_time += time.perf_counter()-t
            t = time.perf_counter()
            with capture.range("model.forward"):
                logits = net(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            model_time += time.perf_counter() - t
            assert torch.isfinite(logits).all()
            ranked = logits.topk(5, dim=1).indices
            top1 += int((ranked[:, 0] == labels).sum())
            top5 += int((ranked == labels[:, None]).any(dim=1).sum())
            ce += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
            input_work += float(construction.sum())
            native_time += float(native.sum())
            adaptation_time += float(adaptation.sum())
            read_bytes += int(sizes.sum())
            count += len(labels)
            predictions.extend(ranked[:, 0].tolist())
            print(f"{args.route} {count}/{args.count} top1={top1/count:.4f}", flush=True)
    wall = time.perf_counter() - start
    result = dict(path=args.route, samples=count, top1=100*top1/count, top5=100*top5/count,
                  ce=ce/count, input_worker_seconds=input_work, model_seconds=model_time,
                  e2e_seconds=wall, logical_read_bytes=read_bytes, physical_read_bytes=None,
                  initialization_seconds=capture.initialization_seconds,
                  input_wait_seconds=capture.wait_seconds,
                  read_bytes_kind="source JPEG bytes" if args.route == "R" else "requested GALP payload bytes",
                  native_read_decode_worker_seconds=native_time if args.route != "R" else None,
                  adaptation_worker_seconds=adaptation_time if args.route != "R" else None,
                  manifest=str(manifest) if manifest else None,
                  timing_scope="source files through logits; warmup excluded; includes worker startup",
                  device=str(device), precision="float32", batch_size=args.batch_size,
                  h2d_seconds=h2d_time if device.type == "cuda" else None,
                  workers=args.workers, model_threads=args.model_threads,
                  source_root=str(B.DEFAULT_DATA_ROOT), checkpoint=str(B.CHECKPOINT),
                  predictions=predictions)
    if args.route == "S":
        result["numerical_contract"] = "full source read/decode on CPU, crop/resize, round-even, int16 saturation, normalize"
        result["comparison_note"] = "same source content as online native; different physical layout and CPU resize arithmetic"
    baseline = args.output_dir / f"R_{count}.json"
    if args.route != "R" and baseline.exists():
        previous = json.loads(baseline.read_text())
        result["prediction_agreement_with_R"] = sum(a==b for a,b in zip(predictions,previous["predictions"]))/count
        result["top1_delta_pp_vs_R"] = result["top1"]-previous["top1"]
    (args.output_dir / f"{args.route}_{count}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
