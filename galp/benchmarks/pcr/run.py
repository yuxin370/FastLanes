"""Fixed-checkpoint RGB inference with PCR or original JPEG input.

Bandwidth limits pace aggregate read completion across all workers, with warm
pages and no idle credit, matching experiments/boundary_control.py's model.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
from pathlib import Path
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .storage import read_record, read_size


class ReadLimiter:
    def __init__(self, mbps):
        self.bytes_per_second = mbps * 1_000_000
        self.deadline = multiprocessing.Value("d", 0.)

    def reserve(self, size):
        with self.deadline.get_lock():
            end = max(time.monotonic(), self.deadline.value) + size / self.bytes_per_second
            self.deadline.value = end
        return end

    def finish(self, end):
        delay = end - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class PacedJPEGInputs(Dataset):
    """The RGB deployment reader with pacing immediately after each file read."""
    def __init__(self, dataset, limiter, sizes):
        self.dataset, self.limiter, self.sizes = dataset, limiter, sizes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, number):
        entry = self.dataset.entries[number]
        started = time.perf_counter()
        deadline = self.limiter.reserve(self.sizes[number])
        blob = Path(entry["path"]).read_bytes()
        self.limiter.finish(deadline)
        read_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with Image.open(io.BytesIO(blob)) as image:
            pixels = image.convert("RGB")
        decode_seconds = time.perf_counter() - started
        started = time.perf_counter()
        inputs = self.dataset.transform(pixels)
        return (inputs, entry["model_label"], entry["ordinal"], read_seconds,
                decode_seconds, time.perf_counter() - started, len(blob))


class RecordInputs(Dataset):
    def __init__(self, root, index, route, scans, transform, limiter):
        self.root, self.index, self.route = root, index, route
        self.scans, self.transform, self.limiter = scans, transform, limiter

    def __len__(self):
        return len(self.index["records"])

    def __getitem__(self, number):
        record = self.index["records"][number]
        first, count = record["first"], record["count"]
        entries = self.index["entries"][first:first + count]
        started = time.perf_counter()
        scans = self.scans if self.route == "pcr" else 1
        blobs, labels, size = read_record(self.root, record[self.route], scans, self.limiter)
        if labels != [entry["model_label"] for entry in entries]:
            raise ValueError("PCR labels differ from the sample selection")
        read_seconds = time.perf_counter() - started
        inputs, decode_seconds, transform_seconds = [], 0., 0.
        for blob in blobs:
            started = time.perf_counter()
            with Image.open(io.BytesIO(blob)) as image:
                pixels = image.convert("RGB")
            decode_seconds += time.perf_counter() - started
            started = time.perf_counter()
            inputs.append(self.transform(pixels))
            transform_seconds += time.perf_counter() - started
        return (torch.stack(inputs), torch.tensor(labels), first, size,
                read_seconds, decode_seconds, transform_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--route", choices=("jpeg_files", "jpeg_record", "pcr"), required=True)
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--model", choices=("mobilenet24", "resnet24"), default="mobilenet24")
    parser.add_argument("--read-mbps", type=float, default=0.)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Original-JPEG result on the identical selection/model")
    args = parser.parse_args()
    if args.read_mbps < 0 or args.workers < 0 or args.batch_size < 1:
        parser.error("Use a nonnegative read rate/workers and a positive batch size")
    if args.output.exists():
        parser.error("Output already exists; use a fresh trial path")
    os.environ["DCTNET_PROFILE"] = args.model
    from galp.benchmarks.dct_models import rgb
    from galp.benchmarks.profiling.input_motivation import cache_pages

    index = json.loads((args.data / "index.json").read_text())
    entries, records = index["entries"], index["records"]
    if any(record["first"] % args.batch_size for record in records):
        parser.error("Record boundaries must align with model batches")
    if args.route == "jpeg_files":
        paths = [Path(entry["path"]) for entry in entries]
        jpeg_sizes = [path.stat().st_size for path in paths]
        expected_bytes = sum(jpeg_sizes)
    else:
        paths = [args.data / record[args.route]["path"] for record in records]
        expected_bytes = sum(read_size(record[args.route], args.scans if args.route == "pcr" else 1)
                             for record in records)
    cache = cache_pages(paths, warm=True)
    if cache["resident_fraction"] != 1:
        raise RuntimeError("Input pages are not fully warm")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    model = rgb.model().cuda()
    transform = rgb.Inputs(entries).transform
    limiter = ReadLimiter(args.read_mbps) if args.read_mbps else None
    predictions, sample_ids = [], [entry["logical_sample_id"] for entry in entries]
    count = correct1 = correct5 = issued_bytes = h2d_tensor_bytes = 0
    ce = read_work = decode_work = transform_work = 0.
    source = None

    def batches():
        nonlocal source
        if args.route == "jpeg_files":
            dataset = rgb.Inputs(entries)
            if limiter is not None:
                dataset = PacedJPEGInputs(dataset, limiter, jpeg_sizes)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=False, pin_memory=True, worker_init_fn=rgb.worker_init,
                                prefetch_factor=2 if args.workers else None)
            source = iter(loader)
            for first in range(0, len(entries), args.batch_size):
                images, labels, ordinals, reads, decodes, transforms, sizes = next(source)
                if ordinals.tolist() != list(range(first, first + len(labels))):
                    raise ValueError("JPEG sample order changed")
                yield images, labels, first, int(sizes.sum()), float(reads.sum()), float(decodes.sum()), float(transforms.sum())
        else:
            dataset = RecordInputs(args.data, index, args.route, args.scans, transform, limiter)
            loader = DataLoader(dataset, batch_size=None, num_workers=args.workers,
                                pin_memory=True, worker_init_fn=rgb.worker_init,
                                prefetch_factor=2 if args.workers else None)
            source = iter(loader)
            for _ in records:
                yield next(source)

    with torch.inference_mode():
        dummy = torch.zeros(args.batch_size, 3, 224, 224, device="cuda")
        for _ in range(20):
            model(dummy)
        torch.cuda.synchronize()
        del dummy
        started = time.perf_counter()
        for images, labels, first, size, reads, decodes, transforms in batches():
            if first != count:
                raise ValueError("Record/sample order changed")
            issued_bytes += size
            read_work += reads
            decode_work += decodes
            transform_work += transforms
            for offset in range(0, len(labels), args.batch_size):
                x = images[offset:offset + args.batch_size]
                y = labels[offset:offset + args.batch_size]
                h2d_tensor_bytes += x.numel() * x.element_size() + y.numel() * y.element_size()
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                logits = model(x)
                ce += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
                ranked = logits.topk(5, dim=1).indices
                matches = ranked.eq(y[:, None])
                correct1 += int(matches[:, :1].sum())
                correct5 += int(matches.sum())
                predictions.extend(ranked.cpu().tolist())
                count += len(y)
                stream.synchronize()
                del x, y, logits, ranked, matches
        elapsed = time.perf_counter() - started
    del source
    if count != len(entries) or issued_bytes != expected_bytes:
        raise ValueError("Incomplete sample coverage or unexpected read byte count")
    result = dict(route=args.route, scans=args.scans if args.route == "pcr" else None,
                  model=args.model, checkpoint=str(rgb.CHECKPOINT), samples=count,
                  e2e_seconds=elapsed, images_per_s=count / elapsed, top1=100 * correct1 / count,
                  top5=100 * correct5 / count, ce=ce / count, read_mbps=args.read_mbps,
                  issued_read_bytes=issued_bytes, storage_bytes=sum(path.stat().st_size for path in paths),
                  external_index_bytes=(args.data / "index.json").stat().st_size,
                  h2d_input_tensor_bytes=h2d_tensor_bytes,
                  h2d_note="Input images and labels only; calculated tensor bytes, not a CUPTI measurement",
                  read_worker_seconds=read_work, decode_worker_seconds=decode_work,
                  transform_worker_seconds=transform_work, workers=args.workers, batch_size=args.batch_size,
                  record_images=index["record_images"], model_threads=8, warmup_batches=20,
                  physical_order=index.get("physical_order", False),
                  precision="float32", tf32=False, cache=cache, device=torch.cuda.get_device_name(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), data=str(args.data.resolve()),
                  upstream_commit=index["upstream_commit"],
                  implementation="raw PCR layout + Python/PyTorch reader; not the original TensorFlow runtime",
                  timing_scope="reader creation/startup through final synchronized logits, CE, top1/top5 and prediction copies; excludes worker shutdown, index parsing, model load and 20-batch model warmup",
                  bandwidth_scope="shared warm-page application-read completion, decimal MB/s, no idle credit or RTT",
                  sample_ids=sample_ids, predictions=predictions)
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if (set(sample_ids) != set(reference["sample_ids"]) or result["model"] != reference["model"]
                or Path(result["checkpoint"]).resolve() != Path(reference["checkpoint"]).resolve()
                or args.batch_size != reference.get("batch_size", reference.get("batch"))):
            raise ValueError("Reference has a different sample/model contract")
        expected = dict(zip(reference["sample_ids"], reference["predictions"], strict=True))
        result["prediction_agreement"] = sum(actual[0] == (expected[sid][0] if isinstance(expected[sid], list) else expected[sid])
                                             for sid, actual in zip(sample_ids, predictions, strict=True)) / count
        if isinstance(reference["predictions"][0], list):
            result["top5_prediction_agreement"] = sum(actual == expected[sid]
                                                     for sid, actual in zip(sample_ids, predictions, strict=True)) / count
        result["top1_delta_pp"] = result["top1"] - reference["top1"]
        result["reference"] = str(args.reference.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sample_ids", "predictions")}), flush=True)


if __name__ == "__main__":
    main()
