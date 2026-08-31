#!/usr/bin/env python3
"""Deterministic bounded one-pool-lookahead contract for experimental PLS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from galp.torch.experimental import DirectDctPlsPipeline


def _wait_stats(
    pipeline: DirectDctPlsPipeline,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: float,
    description: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        stats = pipeline.prefetch_stats
        if predicate(stats):
            return stats
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}: {stats!r}")
        time.sleep(0.005)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pipeline = DirectDctPlsPipeline(
        args.manifest,
        args.mapping,
        training_seed=11997733,
        expected_mapping_sha256=args.mapping_sha256,
        segments_per_pool=1,
        module_path=args.module_path,
    )
    pool_a = None
    pool_b = None
    try:
        pipeline.start_epoch(0)
        pool_a = pipeline.next_pool()
        after_a = _wait_stats(
            pipeline,
            lambda value: int(value["prepare_completed_count"]) >= 2,
            timeout_seconds=args.timeout_seconds,
            description="Pool B prepare completion while Pool A remains active",
        )
        if pool_a.pool_index != 0:
            raise AssertionError(f"first pool index changed: {pool_a.pool_index}")
        if int(after_a["live_context_count"]) != 2:
            raise AssertionError(f"Pool A/B contexts are not both live: {after_a!r}")

        pool_b = pipeline.next_pool()
        if pool_b.pool_index != 1:
            raise AssertionError(f"second pool index changed: {pool_b.pool_index}")
        bounded = _wait_stats(
            pipeline,
            lambda value: int(value["context_waiter_count"]) == 1,
            timeout_seconds=args.timeout_seconds,
            description="third prepare waiting for one of two context permits",
        )
        if int(bounded["prepare_started_count"]) != 2:
            raise AssertionError(f"third prepare started before a context retired: {bounded!r}")
        if int(bounded["peak_live_context_count"]) > 2:
            raise AssertionError(f"pool context bound exceeded: {bounded!r}")

        pool_a.retire()
        after_retire = _wait_stats(
            pipeline,
            lambda value: int(value["prepare_started_count"]) >= 3,
            timeout_seconds=args.timeout_seconds,
            description="third prepare admission after Pool A retirement",
        )
        if int(after_retire["peak_live_context_count"]) > 2:
            raise AssertionError(f"pool context bound exceeded after retirement: {after_retire!r}")

        ids_a = pool_a.microbatch(0).global_image_ids
        ids_b = pool_b.microbatch(0).global_image_ids
        if not ids_a or not ids_b or set(ids_a).intersection(ids_b):
            raise AssertionError("adjacent PLS pools emitted empty or overlapping first microbatches")

        result = {
            "schema": "galp-pls-pool-lookahead-v1",
            "result": "PASS",
            "pool_indices": [pool_a.pool_index, pool_b.pool_index],
            "pool_a_first_microbatch_ids": ids_a,
            "pool_b_first_microbatch_ids": ids_b,
            "after_pool_a_activation": after_a,
            "while_two_contexts_live": bounded,
            "after_pool_a_retire": after_retire,
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print("PLS POOL LOOKAHEAD PASS")
        print(rendered, end="")
        return 0
    finally:
        if pool_a is not None:
            pool_a.retire()
        if pool_b is not None:
            pool_b.retire()
        pool_a = None
        pool_b = None
        pipeline.reclaim_finished_pools()
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
