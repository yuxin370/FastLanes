"""Check committed counts/order and a few real reference/roundtrip samples."""
import argparse
import json
from pathlib import Path
import subprocess
import os
import sys

import numpy as np
import torch

import galp.benchmarks.dct_models.backend as B
from galp.benchmarks.dct_models.storage import BRIDGE, Reader


def validate_block_major(data, samples, info, ref, net):
    from galp.benchmarks.dct_models.evaluate_shards import native_options, organize_batch
    sys.path.insert(0, str(B.REPO / "build/galp/torch"))
    import _galp_direct_dct as native
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(data.resolve() / "access")
    reader = native.DirectDctReader(str(data.resolve() / "manifest.bin"))
    net = net.cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks = []
    # First, middle and final shard, including the short final shard boundary.
    shards = info["shards"]
    with torch.inference_mode():
        for j in sorted({0, len(shards)//2, len(shards)-1}):
            shard = shards[j]
            first, count = shard["first_global_image_index"], shard["image_count"]
            batch = reader.read_batch(list(range(first, first+count)), **native_options())
            y, cbcr = batch.y, batch.cbcr
            x = organize_batch(y, cbcr)
            for local in sorted({0, count-1}):
                index = first + local
                q, qt, expected = ref.coefficients(samples[index]["path"], verify=True)
                md = reader.image_metadata(index)
                table_map = {t["table_id"]:t["values"] for t in md["quant_tables"]}
                tables = [table_map[c["quant_tbl_no"]] for c in md["components"]]
                np.testing.assert_array_equal(tables, qt)
                np.testing.assert_array_equal(qt, np.ones_like(qt))
                actual = torch.cat((y[local:local+1], cbcr[local:local+1]), 1).cpu().reshape(3,B.GRID,B.GRID,64)
                torch.testing.assert_close(actual, torch.as_tensor(q).float(), rtol=0, atol=0)
                r = ref(samples[index]["path"]).cuda()
                torch.testing.assert_close(x[local], r, rtol=0, atol=0)
                a, b = net(x[local:local+1]), net(r[None])
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                checks.append(dict(index=index, input_max_abs=float((x[local]-r).abs().max()),
                                   logits_max_abs=float((a-b).abs().max())))
            stats = batch.execution_stats
            assert stats["uses_planless_fixed_transform"] and stats["duplicate_physical_read_count"] == 0
            del x, y, cbcr, batch
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    info = json.loads(subprocess.check_output([str(BRIDGE), "info-manifest", str(args.data/"manifest.bin")]))
    samples = json.loads((args.data/"samples.json").read_text())
    labels = json.loads((args.data/"labels.json").read_text())
    assert info["image_count"] == len(samples) == labels["image_count"]
    assert len({s["galp_image_id"] for s in samples}) == len(samples)
    assert labels["labels"] == [s["label"] for s in samples]
    assert labels["sample_ids"] == [s["logical_sample_id"] for s in samples]
    population = json.loads((B.DEFAULT_E2E_V3_ROOT/"training_manifests_official_v3/val.json").read_text())["samples"]
    for sample in samples:
        original = population[sample["galp_image_id"]]
        for key in ("path", "label", "logical_sample_id"):
            assert sample[key] == original[key]
    cursor = 0
    for shard in info["shards"]:
        assert shard["first_global_image_index"] == cursor
        assert shard["image_count"] == shard["metadata_image_count"]
        cursor += shard["image_count"]
    assert cursor == len(samples)
    torch.set_num_threads(8)
    ref, net = B.Reference(), B.model()
    if info["version"] == 1:
        checks = validate_block_major(args.data, samples, info, ref, net)
        result = dict(samples=len(samples), shards=info["shard_count"], failed_shards=0,
                      physical_layout=info["physical_layout"], mapping_matches_existing_manifest=True,
                      coefficients_and_qtables_exact=True, sampled_reads=checks,
                      fls_bytes=sum(s["fls_bytes"] for s in info["shards"]),
                      metadata_bytes=sum(s["metadata_bytes"] for s in info["shards"]))
        (args.data/"validation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    indices = sorted({0, min(511,len(samples)-1), min(512,len(samples)-1),
                      len(samples)//3, len(samples)//2, len(samples)-1})
    checks = []
    with Reader(args.data/"manifest.bin") as reader, torch.inference_mode():
        for index in indices:
            sample = samples[index]
            actual, tables = reader.read(index)
            expected, qt, _ = ref.coefficients(sample["path"], verify=True)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(tables, qt)
            r = ref(sample["path"])
            n = B.organize([a.astype("float32")*q for a,q in zip(actual,tables)])
            torch.testing.assert_close(r,n,rtol=0,atol=0)
            output = net(torch.stack([r,n]))
            torch.testing.assert_close(output[0],output[1],rtol=1e-5,atol=1e-5)
            checks.append(dict(index=index, galp_image_id=sample["galp_image_id"],
                               input_max_abs=float((r-n).abs().max()),
                               logits_max_abs=float((output[0]-output[1]).abs().max())))
    result = dict(samples=len(samples), shards=info["shard_count"], failed_shards=0,
                  mapping_matches_existing_manifest=True, coefficients_and_qtables_exact=True,
                  sampled_reads=checks, fls_bytes=sum(s["fls_bytes"] for s in info["shards"]),
                  metadata_bytes=sum(s["metadata_bytes"] for s in info["shards"]),
                  payload_bytes=sum(s["payload_size"] for s in info["shards"]))
    (args.data/"validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
