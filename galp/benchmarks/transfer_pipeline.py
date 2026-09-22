"""Small end-to-end control for transfer-lock experiments; uses the real extension.

Run with PYTHONPATH containing the built native module and repository root.
This includes scheduling, reading, decode, transform and consumer completion.
It is not a model-training throughput benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
import torch
from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("--batches", type=int, default=64)
parser.add_argument("--pipelines", type=int, default=1)
args = parser.parse_args()
if args.batches <= 0 or args.pipelines <= 0:
    raise ValueError("positive batches and pipelines required")


def consume():
    torch.cuda.set_device(0)
    reader = DirectDctReader(args.manifest)
    ids = list(range(min(2, reader.image_count)))
    if not ids:
        raise ValueError("nonempty manifest required")
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        with reader.pipeline(VALIDATION).start([ids] * args.batches) as pipeline:
            count = 0
            for batch in pipeline:
                # Touch both outputs and complete their consumer work.
                value = batch.y.sum() + batch.cbcr.sum()
                count += len(ids)
            consumer.synchronize()
            if not torch.isfinite(value).item():
                raise RuntimeError("nonfinite pipeline output")
    return count


with ThreadPoolExecutor(max_workers=args.pipelines) as workers:
    for repeat in range(4):
        start = time.perf_counter()
        jobs = [workers.submit(consume) for _ in range(args.pipelines)]
        count = sum(job.result() for job in jobs)
        seconds = time.perf_counter() - start
        print(f"repeat={repeat},pipelines={args.pipelines},images={count},seconds={seconds:.6f},images_s={count/seconds:.3f}", flush=True)
