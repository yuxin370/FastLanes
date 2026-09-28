"""Capture one source512 prediction mismatch, optionally in its original batch of 64.

This is a diagnostic, not a correctness gate or performance measurement. Native
float32 grids already contain rounded integers; the public reader does not expose
the native pre-rounding accumulator. No full-shard activation or warmup is run.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("DCTNET_PROFILE", "mobilenet24")

from galp.benchmarks.dct_models import backend as B
from galp.benchmarks.dct_models.evaluate import samples
from galp.benchmarks.dct_models.evaluate_shards import apply_b6_runtime, native_options, organize_batch
from galp.benchmarks.dct_models.online_crop import CROP, SOURCE_PROFILE, SourceReference, read_source_jpeg, source_options


def difference(left, right):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("non-finite diagnostic tensor")
    return dict(exact=torch.equal(left, right), disagreements=int((left != right).sum()),
                max_abs=float((left.double() - right.double()).abs().max()))


def selected_coefficients(components):
    return torch.cat([c.permute(2, 0, 1)[indices] for c, indices in zip(components, B.INDICES)])


def logit_summary(logits):
    values, indices = logits.topk(5)
    return dict(top5=indices.tolist(), top5_logits=values.tolist(),
                top1_top2_margin=float(values[0] - values[1]),
                classes={str(c): float(logits[c]) for c in (496, 514, 457)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", type=int, default=22858, help="galp_image_id, never the stored ordinal")
    parser.add_argument("--batch-size", type=int, choices=(1, 64), default=1)
    parser.add_argument("--new-data", type=Path, default=B.REPO / "galp/data/compressed/imagenet512_val_block_major")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true", help="capture the CPU reference without accessing CUDA")
    parser.add_argument("--verify", action="store_true", help="require exact native coefficients, inputs and logits")
    args = parser.parse_args()
    if args.verify and args.cpu_only:
        parser.error("--verify requires native GPU execution")
    if B.PROFILE != "mobilenet24":
        parser.error("this diagnostic requires DCTNET_PROFILE=mobilenet24")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    population = samples(50000, physical_order=True)  # metadata only
    target_ordinal = next(i for i, s in enumerate(population) if s["galp_image_id"] == args.image_id)
    first = target_ordinal // args.batch_size * args.batch_size
    entries = population[first:first + args.batch_size]
    target = target_ordinal - first
    root = args.new_data.resolve()
    if json.loads((root / "profile.json").read_text()) != SOURCE_PROFILE:
        raise ValueError("expected source512 storage")
    stored = json.loads((root / "samples.json").read_text())
    positions = {s["galp_image_id"]: i for i, s in enumerate(stored)}
    ids = [positions[s["galp_image_id"]] for s in entries]
    for entry, position in zip(entries, ids):
        if entry["logical_sample_id"] != stored[position]["logical_sample_id"]:
            raise ValueError("source and compressed sample identities differ")
    report = dict(sample=entries[target], stored_position=ids[target],
                  stored_ordinal=stored[ids[target]]["ordinal"], batch_target_index=target,
                  batch_sample_ids=[s["logical_sample_id"] for s in entries],
                  batch_storage_positions=ids, batch_size=len(entries), checkpoint=str(B.CHECKPOINT),
                  torch_version=torch.__version__, tf32=False, cpu_only=args.cpu_only,
                  native_pre_rounding="unavailable: public reader finalizes before returning tensors")
    tensors = {}
    reference = SourceReference(B.DEFAULT_RGBNOMORE_ROOT, B.GRID)
    cpu_inputs = []
    for i, entry in enumerate(entries):
        q, tables = read_source_jpeg(entry["path"])
        unrounded = torch.stack(reference(q, tables))
        rounded = unrounded.round().clamp(-32768, 32767)
        cpu_inputs.append(B.organize(rounded))
        if i == target:
            tensors.update(source_quantized=[torch.as_tensor(c) for c in q],
                           source_tables=torch.stack([torch.as_tensor(t).int() for t in tables]),
                           cpu_unrounded=unrounded, cpu_rounded=rounded)
    tensors["cpu_input"] = torch.stack(cpu_inputs)
    if not args.cpu_only:
        sys.path.insert(0, str(B.REPO / "build/galp/torch"))
        import _galp_direct_dct as native

        os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(root / "access")
        reader = native.DirectDctReader(str(root / "manifest.bin"))
        metadata = reader.image_metadata(ids[target])
        table_by_id = {t["table_id"]: t["values"] for t in metadata["quant_tables"]}
        tensors["native_source_tables"] = torch.tensor(
            [table_by_id[c["quant_tbl_no"]] for c in metadata["components"]], dtype=torch.int32)
        raw = reader.read_batch([ids[target]], layout="ycbcr_dct_grid", dct_coeffs="all",
                                cache_capacity_mib=0, plan_cache_capacity=0)
        tensors["native_source_quantized"] = [
            raw.y[0, 0].flatten(-2).cpu(), raw.cbcr[0, 0].flatten(-2).cpu(), raw.cbcr[0, 1].flatten(-2).cpu()]
        del raw
        report["source_quantized"] = [difference(a, b) for a, b in
                                      zip(tensors["source_quantized"], tensors["native_source_quantized"])]
        report["source_tables"] = difference(tensors["source_tables"], tensors["native_source_tables"])
        report["native_options"] = {}
        for name, projected, mode in (("projected", True, "vector-range-read-selected-decode"),
                                      ("grid", False, "vector-range-read-selected-decode"),
                                      ("full_source", False, "full-source-decode")):
            options = apply_b6_runtime(source_options(native_options(False, projected), mode))
            report["native_options"][name] = options
            batch = reader.read_prefetched(reader.prefetch_batch(
                ids, transforms=[dict(crop=CROP, horizontal_flip=False) for _ in ids], **options))
            if projected:
                tensors[name + "_input"] = batch.projected.cpu()
            else:
                tensors[name + "_input"] = organize_batch(batch.y, batch.cbcr).cpu()
                tensors[name + "_rounded"] = torch.cat(
                    (batch.y[target], batch.cbcr[target]), dim=0).flatten(-2).cpu()
            del batch
        report["cpu_vs_native_rounded"] = difference(tensors["cpu_rounded"], tensors["grid_rounded"])
        report["inputs"] = {name: difference(tensors["cpu_input"], tensors[name + "_input"])
                            for name in ("grid", "projected", "full_source")}
        report["projected_vs_grid"] = difference(tensors["projected_input"], tensors["grid_input"])
        report["grid_vs_full_source"] = difference(tensors["grid_input"], tensors["full_source_input"])
        cpu_values = selected_coefficients(tensors["cpu_unrounded"])
        cpu_rounded = selected_coefficients(tensors["cpu_rounded"])
        native_rounded = selected_coefficients(tensors["grid_rounded"])
        channels = [(c, f) for c, indices in enumerate(B.INDICES) for f in indices]
        report["selected_rounding_disagreements"] = [
            dict(channel=c, component=channels[c][0], frequency=channels[c][1], y=y, x=x,
                 cpu_unrounded=float(cpu_values[c, y, x]), cpu_rounded=float(cpu_rounded[c, y, x]),
                 native_rounded=float(native_rounded[c, y, x]))
            for c, y, x in (cpu_rounded != native_rounded).nonzero().tolist()]
        net = B.model().cuda()
        report["device"] = torch.cuda.get_device_name()
        report["logits"] = {}
        with torch.inference_mode():
            for name in ("cpu", "grid", "projected", "full_source"):
                # Separate forwards with identical shape, strides, dtype and model.
                logits = net(tensors[name + "_input"].cuda().contiguous()).cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError("non-finite model logits")
                tensors[name + "_logits"] = logits
                report["logits"][name] = logit_summary(logits[target])
        report["cpu_vs_projected_logits"] = difference(tensors["cpu_logits"], tensors["projected_logits"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, args.output_dir / "tensors.pt")
    (args.output_dir / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.verify:
        for actual, expected in zip(tensors["native_source_quantized"], tensors["source_quantized"]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(tensors["native_source_tables"], tensors["source_tables"], atol=0, rtol=0)
        for name in ("grid", "full_source"):
            torch.testing.assert_close(tensors[name + "_rounded"], tensors["cpu_rounded"], atol=0, rtol=0)
        for name in ("grid", "projected", "full_source"):
            torch.testing.assert_close(tensors[name + "_input"], tensors["cpu_input"], atol=0, rtol=0)
            torch.testing.assert_close(tensors[name + "_logits"], tensors["cpu_logits"], atol=0, rtol=0)
        print("PASS: source coefficients/tables, resized coefficients, inputs and logits are exactly equal")


if __name__ == "__main__":
    main()
