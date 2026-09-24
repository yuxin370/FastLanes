"""DCTNet: one native block-major activation per shard, then GPU microbatches.

Uses B6's existing planless reader and bounded native prefetch. Supports stored
target geometry or full source512 coefficients with online center crop/resize.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import galp.benchmarks.dct_models.backend as B
from galp.benchmarks.dct_models.evaluate import samples
from galp.benchmarks.dct_models.capture import Capture
from galp.benchmarks.dct_models.online_crop import CROP, SOURCE_PROFILE, SourceReference, read_source_jpeg, source_options

from galp.benchmarks.system_dct_major.common import parse_manifest, ZIGZAG_COLUMN_TO_NATURAL_INDEX


def native_options(pushdown=False, projected=False):
    required = set().union(*map(set, B.INDICES))
    columns = [i for i, natural in enumerate(ZIGZAG_COLUMN_TO_NATURAL_INDEX) if natural in required]
    selection = "list:" + ",".join(map(str, columns)) if pushdown else "all"
    options = dict(layout="transformed_dct_grid", dct_coeffs=selection,
                enable_planless_execution=True, cache_capacity_mib=0,
                plan_cache_capacity=0,
                decode_batch_rowgroups=64, rowgroup_prefetch_workers=4,
                decode_workset_capacity_mib=512,
                grid_transform=dict(y_output_width_blocks=B.GRID, y_output_height_blocks=B.GRID,
                                    cbcr_output_width_blocks=B.GRID, cbcr_output_height_blocks=B.GRID,
                                    crop_reference_width_blocks=B.GRID, crop_reference_height_blocks=B.GRID,
                                    crop_origin_alignment_blocks=1,
                                    chroma_crop_scale_x=1, chroma_crop_scale_y=1,
                                    allowed_chroma_sampling_ratios=[[1, 1, 1, 1]],
                                    # Q100 tables are all 1; stored int16 values
                                    # cannot be changed by this native API bound.
                                    clamp_min=-32768, clamp_max=32767,
                                    output_dtype="float32", output_add=0., output_scale=1.,
                                    dequantize=True, require_all_coefficients=not pushdown))
    if projected:
        mean, std = B.MEAN.flatten().tolist(), B.STD.flatten().tolist()
        channels = [(component, frequency) for component, indices in enumerate(B.INDICES) for frequency in indices]
        options["grid_transform"]["output_channels"] = [
            [component, frequency, mean[i], std[i]] for i, (component, frequency) in enumerate(channels)]
    return options


def apply_b6_runtime(options):
    """B6 runtime from profiles/direct_dct.hpp, with the measured CNN launch size.

    Geometry and crop-off remain experiment inputs. No training augmentation or
    shuffle is imported into inference.
    """
    options.update(enable_rowgroup_prefetch=True, rowgroup_prefetch_depth=16,
                   rowgroup_prefetch_workers=8, rowgroup_prefetch_min_decode_batches=1,
                   scheduling_policy="limited-overlap", transform_blocks_per_launch=32768,
                   transform_ctas_per_launch=512, use_low_priority_streams=True,
                   async_planless_completion=True, block_major_double_buffer="auto",
                   bounded_read_amplification_cap=1.1,
                   bounded_read_local_amplification_cap=0., bounded_read_max_run_bytes=0)
    if options["crop_execution_mode"] != "full-source-decode":
        options["crop_execution_mode"] = "bounded-io-uring-range-read-selected-decode"
    return options


def activation_groups(shards, size):
    groups = []
    for first in range(0, len(shards), size):
        members = shards[first:first + size]
        ids = [i for s in members for i in range(s["first_global_image_index"],
                                                s["first_global_image_index"] + s["image_count"])]
        groups.append(dict(shard_id=members[0]["shard_id"], shard_ids=[s["shard_id"] for s in members],
                           image_ids=ids, image_count=len(ids),
                           rowgroup_count=sum(s["rowgroup_count"] for s in members)))
    return groups



def organize_batch(y, cbcr):
    n = len(y)
    if tuple(y.shape) != (n, 1, B.GRID, B.GRID, 8, 8) or tuple(cbcr.shape) != (n, 2, B.GRID, B.GRID, 8, 8):
        raise ValueError(f"unexpected target geometry: {y.shape}, {cbcr.shape}")
    planes = [y[:, 0].flatten(-2), cbcr[:, 0].flatten(-2), cbcr[:, 1].flatten(-2)]
    chosen = [c.permute(0, 3, 1, 2)[:, indices] for c, indices in zip(planes, B.INDICES)]
    x = torch.cat(chosen, dim=1)
    return x.sub_(B.MEAN.to(y.device)).div_(B.STD.to(y.device))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-data", type=Path, required=True)
    p.add_argument("--count", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model-threads", type=int, default=8)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path,
                   default=B.DEFAULT_E2E_V3_ROOT / "runs" / B.RUN_NAME / "gpu")
    p.add_argument("--verify", action="store_true", help="separate small correctness run; not performance")
    p.add_argument("--profile", action="store_true", help="capture 4 shards after 16 warmup shards")
    p.add_argument("--pushdown", choices=["on", "off"], default="on")
    p.add_argument("--output-layout", choices=["grid", "projected"], default="projected")
    p.add_argument("--input-geometry", choices=["precomputed", "source512"], default="precomputed")
    p.add_argument("--crop-execution-mode", default="auto",
                   choices=["auto", "full-source-decode", "full-rowgroup-decode", "vector-range-read-selected-decode"])
    p.add_argument("--native-runtime", choices=["legacy", "b6"], default="legacy")
    p.add_argument("--shards-per-activation", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--physical-prefix", action="store_true", help="calibration: use the first count stored images")
    args = p.parse_args()
    online = args.input_geometry == "source512"
    if online and args.pushdown != "off":
        p.error("source512 resize needs all source frequencies: pass --pushdown off")
    if args.native_runtime == "b6" and not online:
        p.error("B6 inference runtime requires --input-geometry source512")
    if not 1 <= args.count <= 50000:
        p.error("count must be in [1, 50000]")
    torch.set_num_threads(args.model_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(B.REPO / "build/galp/torch"))
    import _galp_direct_dct as native

    root = args.new_data.resolve()
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(root / "access")
    manifest = parse_manifest(root / "manifest.bin")
    if manifest["physical_layout"] != "dct-major/spatial-major-image-minor":
        raise ValueError("requires true spatial-major/image-minor storage")
    # Mother data stores all frequencies before model selection/normalization.
    # Compare its numerical storage contract, not the checkpoint that first used it.
    model_fields = {"model", "checkpoint", "shape", "indices", "mean", "std"}
    stored_profile = json.loads((root / "profile.json").read_text())
    if online:
        if stored_profile != SOURCE_PROFILE:
            raise ValueError("online crop requires full, unresized source512 data")
    elif ({k: v for k, v in stored_profile.items() if k not in model_fields}
            != {k: v for k, v in B.profile().items() if k not in model_fields}):
        raise ValueError("stored target profile differs from model input contract")
    stored = json.loads((root / "samples.json").read_text())
    if len(stored) != manifest["image_count"] or (args.count == 50000 and len(stored) != 50000):
        raise ValueError("stored sample mapping does not cover the requested dataset")
    selected = samples(args.count)
    if args.physical_prefix:
        by_id = {s["galp_image_id"]: s for s in samples(50000)}
        selected = [dict(by_id[s["galp_image_id"]], ordinal=i) for i, s in enumerate(stored[:args.count])]
    positions = {s["galp_image_id"]: i for i, s in enumerate(stored)}
    if len(positions) != len(stored):
        raise ValueError("duplicate stored image IDs")
    ordinal_by_position = {positions[s["galp_image_id"]]: s["ordinal"] for s in selected}
    shards = [s for s in manifest["shards"] if any(i in ordinal_by_position for i in
              range(s["first_global_image_index"], s["first_global_image_index"] + s["image_count"]))]
    source_shard_count = len(shards)
    shards = activation_groups(shards, args.shards_per_activation)
    net = B.model().cuda()
    reference = B.Reference()
    if online and args.verify:
        source_reference = SourceReference(B.DEFAULT_RGBNOMORE_ROOT, B.GRID)
    with torch.inference_mode():
        net(reference(selected[0]["path"])[None].cuda())
    torch.cuda.synchronize()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = [-1] * args.count
    top1 = top5 = count = decoded_images = 0
    ce = wait_seconds = organize_seconds = model_seconds = 0.
    shard_stats = []
    checks = []
    capture = Capture(args.profile, first=16 // args.shards_per_activation,
                      steps=4 // args.shards_per_activation, images_per_step=1024 * args.shards_per_activation)
    started = time.perf_counter()
    reader = native.DirectDctReader(str(root / "manifest.bin"))
    init_seconds = time.perf_counter() - started
    if not reader.initialization_stats["block_major_metadata_lazy"]:
        raise RuntimeError("native block-major sidecars not active")

    options = native_options(args.pushdown == "on", args.output_layout == "projected")
    if online:
        source_options(options, args.crop_execution_mode)
    if args.native_runtime == "b6":
        apply_b6_runtime(options)

    def requests(shard):
        ids = shard["image_ids"]
        transforms = [dict(crop=CROP, horizontal_flip=False) for _ in ids] if online else None
        return ids, transforms

    def submit(shard):
        ids, transforms = requests(shard)
        return reader.prefetch_batch(ids, transforms=transforms, **options)

    pending = submit(shards[0])
    with torch.inference_mode():
        for shard_index, shard in enumerate(shards):
            capture.step(shard_index)
            t = time.perf_counter()
            with capture.range("input.wait"):
                batch = reader.read_prefetched(pending)
                if args.output_layout == "projected":
                    x = batch.projected
                    y = cbcr = None
                else:
                    y, cbcr = batch.y, batch.cbcr
                torch.cuda.current_stream().synchronize()
            wait_seconds += time.perf_counter() - t
            decoded_images += len(x) if args.output_layout == "projected" else len(y)
            # Native manages the next preparation; at most one active + one next shard.
            with capture.range("input.prefetch_submit"):
                pending = submit(shards[shard_index + 1]) if shard_index + 1 < len(shards) else None
            t = time.perf_counter()
            with capture.range("input.organize"):
                if args.output_layout == "grid":
                    x = organize_batch(y, cbcr)
                torch.cuda.current_stream().synchronize()
            organize_seconds += time.perf_counter() - t
            image_ids = shard["image_ids"]
            local = [i for i in range(len(x)) if image_ids[i] in ordinal_by_position]
            if args.verify:
                if online:
                    # Identical online transform after complete source decoding.
                    # Verify requested samples; do not allocate a third full M4
                    # grid alongside the active and prefetched activations.
                    ids = [image_ids[i] for i in local]
                    transforms = [dict(crop=CROP, horizontal_flip=False) for _ in ids]
                    comparison = reader.read_batch(ids, transforms=transforms,
                        **source_options(native_options(False, False), "full-source-decode"))
                    torch.testing.assert_close(x[local], organize_batch(comparison.y, comparison.cbcr), atol=0, rtol=0)
                    del comparison
                for i in local[:3]:
                    ordinal = ordinal_by_position[image_ids[i]]
                    if online:
                        q, qt = read_source_jpeg(selected[ordinal]["path"])
                        components = source_reference(q, qt)
                        expected = torch.cat([c.permute(2, 0, 1)[indices]
                                              for c, indices in zip(components, B.INDICES)]).clamp(-32768, 32767)
                        actual = (x[i].cpu() * B.STD + B.MEAN)
                        error = float((actual - expected).abs().max())
                        if error > 0.505:
                            raise AssertionError(f"source resize error exceeds native rounding: {error}")
                        torch.testing.assert_close(actual, actual.round(), atol=0.005, rtol=0)
                        checks.append(dict(ordinal=ordinal, coefficient_max_abs_before_rounding=error,
                                           rounded_coefficient_disagreements=int((actual.round() != expected.round()).sum()),
                                           native_grid_and_full_source_input_exact=True))
                        continue
                    q, qt, components = reference.coefficients(selected[ordinal]["path"])
                    if args.output_layout == "grid":
                        actual = torch.cat((y[i:i+1], cbcr[i:i+1]), dim=1).cpu().reshape(3, B.GRID, B.GRID, 64)
                        target = torch.stack([torch.as_tensor(c) for c in components]).float()
                        if args.pushdown == "on":
                            omitted = sorted(set(range(64)) - set().union(*map(set, B.INDICES)))
                            target[..., omitted] = 0
                        torch.testing.assert_close(actual, target, atol=0, rtol=0)
                    expected = B.organize(components).cuda()
                    torch.testing.assert_close(x[i], expected, atol=0, rtol=0)
                    a, b = net(x[i:i+1]), net(expected[None])
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                    checks.append(dict(ordinal=ordinal, input_max_abs=float((x[i]-expected).abs().max()),
                                       logits_max_abs=float((a-b).abs().max())))
            # Select a requested subset once, then use zero-copy batch views.
            ordinals_in_shard = [ordinal_by_position[image_ids[i]] for i in local]
            if len(local) != len(x):
                x = x[local]
            shard_labels = torch.tensor([selected[o]["model_label"] for o in ordinals_in_shard], device="cuda")
            for offset in range(0, len(local), args.batch_size):
                ordinals = ordinals_in_shard[offset:offset+args.batch_size]
                labels = shard_labels[offset:offset+args.batch_size]
                inputs = x[offset:offset+args.batch_size]
                torch.cuda.current_stream().synchronize()
                t = time.perf_counter()
                with capture.range("model.forward"):
                    logits = net(inputs)
                    torch.cuda.current_stream().synchronize()
                model_seconds += time.perf_counter() - t
                assert torch.isfinite(logits).all()
                ranks = logits.topk(5, dim=1).indices
                top1 += int((ranks[:, 0] == labels).sum())
                top5 += int((ranks == labels[:, None]).any(dim=1).sum())
                ce += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
                count += len(ordinals)
                for o, pred in zip(ordinals, ranks[:, 0].tolist()):
                    if predictions[o] != -1:
                        raise RuntimeError("duplicate sample ordinal")
                    predictions[o] = pred
            stats = dict(batch.execution_stats)
            if not stats["uses_planless_fixed_transform"]:
                raise RuntimeError("expected native planless decode")
            if not online and stats["rowgroup_count"] != shard["rowgroup_count"]:
                raise RuntimeError("expected every target shard rowgroup")
            if any(stats[k] for k in ("duplicate_physical_read_count", "rowgroup_revisit_count",
                                      "vector_run_revisit_count", "physical_read_order_inversions")):
                raise RuntimeError("native reader repeated physical work")
            if not online and stats["actual_vector_count"] != stats["full_vector_count"]:
                raise RuntimeError("whole-shard vector decode coverage differs")
            if online and args.crop_execution_mode == "full-source-decode":
                if stats["rowgroup_count"] != shard["rowgroup_count"] or stats["actual_vector_count"] != stats["full_vector_count"]:
                    raise RuntimeError("crop-off did not decode the complete source shard")
            shard_stats.append(dict(shard_id=shard["shard_id"], images=shard["image_count"],
                                    shard_ids=shard["shard_ids"],
                                    source_rowgroups=shard["rowgroup_count"], **stats))
            print(f"N block-major {count}/{args.count}, shard {shard_index+1}/{len(shards)}", flush=True)
            del inputs, x, y, cbcr, batch
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    assert count == args.count and all(p >= 0 for p in predictions)
    result = dict(path="N", execution="native-block-major-whole-shard", samples=count,
                  decoded_images=decoded_images, shard_activations=source_shard_count,
                  activation_count=len(shard_stats), shards_per_activation=args.shards_per_activation,
                  native_runtime=args.native_runtime, physical_prefix=args.physical_prefix,
                  top1=100*top1/count, top5=100*top5/count, ce=ce/count,
                  e2e_seconds=wall, reader_initialization_seconds=init_seconds,
                  input_wait_seconds=wait_seconds, organization_seconds=organize_seconds,
                  model_seconds=model_seconds, precision="float32", device="cuda",
                  device_name=torch.cuda.get_device_name(),
                  batch_size=args.batch_size, model_threads=args.model_threads,
                  prefetch="one active + one preparing activation; native futures",
                  timing_scope="files through logits, includes reader startup; model warmup excluded",
                  verification_run=args.verify, checks=checks,
                  pushdown=args.pushdown,
                  input_geometry=args.input_geometry, online_crop=CROP if online else None,
                  crop_execution_mode=options["crop_execution_mode"] if online else None,
                  numerical_contract="dequantize, online crop/resize, round-even, int16 saturation, normalize"
                      if online else "stored target coefficients, dequantize, normalize",
                  output_layout=args.output_layout,
                  native_options=options, manifest=str(root/"manifest.bin"),
                  checkpoint=str(B.CHECKPOINT), native_shards=shard_stats, predictions=predictions,
                  model_profile=B.profile(), stored_profile=stored_profile,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  prediction_order="original fixed evaluation ordinal; shard order only affects execution")
    baseline = args.baseline_dir / f"R_{args.count}.json"
    if baseline.exists():
        r = json.loads(baseline.read_text())
        result["prediction_agreement_with_R"] = sum(a == b for a,b in zip(predictions,r["predictions"]))/count
        result["top1_delta_pp_vs_R"] = result["top1"] - r["top1"]
    (args.output_dir/f"N_{args.count}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("native_shards", "predictions")}, indent=2))


if __name__ == "__main__":
    main()
