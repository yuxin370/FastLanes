#!/usr/bin/env python3
"""Diagnostic-only NVTX/CUPTI window and simultaneous memory samples, after one full warm pass."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.system_dct_major import pipeline as benchmark


def ranged(name, function):
    def wrapped(*args, **kwargs):
        with torch.cuda.nvtx.range(name):
            return function(*args, **kwargs)
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=benchmark.PIPELINES, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True, help="NVML-backed nvidia-smi CSV from the existing process monitor")
    parser.add_argument("--memory-only", action="store_true", help="collect memory without Nsight/CUPTI or NVTX instrumentation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(args.source_contract.read_text())
    contract["execution"]["repeats"] = 2
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    rows = []
    stopping = threading.Event()
    window = {}
    process = psutil.Process()
    executor = ThreadPoolExecutor(max_workers=1)

    def sample_memory():
        while not stopping.is_set():
            rss = process.memory_info().rss
            tree_rss = rss
            for child in process.children(recursive=True):
                try:
                    tree_rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            pinned = torch.cuda.memory.host_memory_stats()
            rows.append(dict(timestamp_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(), host_rss_bytes=rss,
                tree_rss_bytes=tree_rss, torch_pinned_active_bytes=pinned.get("active_bytes.current"),
                torch_pinned_allocated_bytes=pinned.get("allocated_bytes.current")))
            stopping.wait(.1)

    original_factory = benchmark.make_adapter

    def make_adapter(*factory_args, **factory_kwargs):
        adapter = original_factory(*factory_args, **factory_kwargs)
        original_begin, original_end = adapter.begin_repeat, adapter.end_repeat
        repeats = 0
        sampler = None

        def begin():
            nonlocal repeats, sampler
            repeats += 1
            if repeats == 2:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                window["start_unix_ns"] = time.time_ns()
                sampler = executor.submit(sample_memory)
                if not args.memory_only:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("fastlens.measured_pass")
            return original_begin()

        def end():
            result = original_end()
            if repeats == 2:
                torch.cuda.synchronize()
                if not args.memory_only:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
                stopping.set()
                sampler.result()
                window["end_unix_ns"] = time.time_ns()
            return result

        adapter.begin_repeat, adapter.end_repeat = begin, end
        if not args.memory_only:
            adapter.load = ranged("input.prepare", adapter.load)
        return adapter

    benchmark.make_adapter = make_adapter
    if not args.memory_only:
        benchmark._to_device = ranged("input.handoff", benchmark._to_device)
        benchmark._forward = ranged("model.forward", benchmark._forward)
    try:
        result = benchmark.run_pipeline(args.pipeline, contract_path, args.output_dir / "pipeline.json")
    finally:
        stopping.set()
        executor.shutdown(wait=True)
    if not rows:
        raise RuntimeError("no whole-process memory samples captured")
    (args.output_dir / "memory_samples.json").write_text(json.dumps(dict(
        pipeline=args.pipeline, pid=os.getpid(), gpu=torch.cuda.get_device_name(),
        gpu_samples=str(args.gpu_samples), profiler_enabled=not args.memory_only,
        samples=result["repeats"][1]["images"], window=window, measurements=rows,
        scope="diagnostic warm full pass; 100 ms samples; tree RSS can double-count shared pages; native counters remain separate; profiler overhead excluded from primary throughput"), indent=2) + "\n")
    print(json.dumps(dict(pipeline=args.pipeline, captured_images=result["repeats"][1]["images"], memory_samples=len(rows))))


if __name__ == "__main__":
    main()
