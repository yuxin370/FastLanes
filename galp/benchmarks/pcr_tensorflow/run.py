"""Native author PCR reader + author TF preprocessing + PyTorch inference.

This is a separate pipeline reproduction, not the PIL-preprocessed paper table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--op", type=Path, required=True)
    parser.add_argument("--upstream-release", type=Path, required=True)
    parser.add_argument("--route", choices=["pcr", "jpeg_files"], required=True)
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new result output path")
    os.environ["DCTNET_PROFILE"] = "mobilenet24"
    import tensorflow as tf
    import torch

    from galp.benchmarks.dct_models import rgb
    from .dataset import interleaved_ordinals, pcr_dataset, split_record

    tf.config.threading.set_inter_op_parallelism_threads(args.workers)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    upstream = args.upstream_release / "tensorflow"
    preprocessing = import_file("pcr_preprocessing", upstream / "imagenet_preprocessing_input.py")
    prefetcher = import_file("pcr_prefetcher", upstream / "data_prefetcher.py")
    index = json.loads((args.data / "index.json").read_text())
    entries = index["entries"]
    if args.route == "pcr":
        order = interleaved_ordinals(index["records"], args.workers)
        filenames = [str(args.data / record["path"]) for record in index["records"]]
        dataset = pcr_dataset(args.op, filenames, args.data / "PCR_index.pb", args.scans, args.workers)
        dataset = dataset.map(split_record, num_parallel_calls=args.workers)
        read_bytes = sum(sum(record["sizes"][:args.scans + 1]) for record in index["records"])
    else:
        order = list(range(len(entries)))
        filenames = [entry["path"] for entry in entries]
        labels = [entry["model_label"] for entry in entries]
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(lambda path, label: (tf.io.read_file(path), label),
                              num_parallel_calls=args.workers)
        read_bytes = sum(Path(path).stat().st_size for path in filenames)

    def preprocess(jpeg, label):
        image = preprocessing.preprocess_image_fn(jpeg, is_training=False, image_size=224)
        image = (image / 255.0 - tf.constant(rgb.MEAN)) / tf.constant(rgb.STD)
        return image, tf.cast(label, tf.int64)

    dataset = dataset.map(preprocess, num_parallel_calls=args.workers)
    dataset = dataset.batch(args.batch_size).prefetch(2)
    options = tf.data.Options()
    options.threading.private_threadpool_size = args.workers
    dataset = dataset.with_options(options)
    model = rgb.model().cuda()
    with torch.inference_mode():
        dummy = torch.zeros(args.batch_size, 3, 224, 224, device="cuda")
        for _ in range(20):
            model(dummy)
        torch.cuda.synchronize()
        del dummy
        started = time.perf_counter()
        loader = prefetcher.data_prefetcher(
            dataset, math.ceil(len(entries) / args.batch_size), permute_channel=True,
        )
        correct1 = correct5 = count = 0
        predictions = []
        for images, labels in loader:
            logits = model(images)
            ranked = logits.topk(5, dim=1).indices
            predictions.extend(ranked.cpu().tolist())
            matches = ranked.eq(labels[:, None])
            correct1 += int(matches[:, :1].sum())
            correct5 += int(matches.sum())
            count += len(labels)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    if count != len(entries):
        raise AssertionError("Incomplete validation dataset")
    result = dict(
        implementation="author C++ PCR DatasetOp; TensorFlow 2.15.1 API adapter",
        route=args.route, scans=args.scans if args.route == "pcr" else None,
        preprocessing="author TF decode_and_crop_jpeg + bicubic; ImageNet mean/std",
        model="mobilenet24 RGB", checkpoint=str(rgb.CHECKPOINT),
        samples=count, top1=100 * correct1 / count, top5=100 * correct5 / count,
        e2e_seconds=seconds, images_per_s=count / seconds, requested_read_bytes=read_bytes,
        h2d_image_bytes=count * 3 * 224 * 224 * 4,
        gpu=torch.cuda.get_device_name(), tensorflow=tf.__version__, torch=torch.__version__,
        workers=args.workers, batch_size=args.batch_size,
        source_raw=index["source_raw"], store=str(args.data),
        timing_scope="single-run pipeline check; no bandwidth cap; cache state not controlled",
        top5_by_sample_id=dict(zip(
            (entries[i]["logical_sample_id"] for i in order), predictions, strict=True)),
    )
    if args.reference:
        reference = json.loads(args.reference.read_text())["top5_by_sample_id"]
        actual = result["top5_by_sample_id"]
        if actual.keys() != reference.keys():
            raise AssertionError("Reference sample IDs differ")
        result["top1_prediction_agreement"] = sum(
            actual[sid][0] == reference[sid][0] for sid in actual) / count
        result["top5_prediction_agreement"] = sum(
            actual[sid] == reference[sid] for sid in actual) / count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "top5_by_sample_id"}), flush=True)


if __name__ == "__main__":
    main()
