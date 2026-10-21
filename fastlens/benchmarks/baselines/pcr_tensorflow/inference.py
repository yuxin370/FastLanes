"""Author-native PCR deployment inference and source-crop bandwidth controls."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import time

from .run import import_file

ROOT = Path(__file__).resolve().parents[4]


class ReadControl:
    def __init__(self, op):
        self.library = ctypes.CDLL(str(op.resolve()))
        self.library.pcr_boundary_begin.argtypes = [ctypes.c_double]
        self.library.pcr_boundary_begin.restype = ctypes.c_int
        self.library.pcr_boundary_reserve.argtypes = [ctypes.c_uint64]
        self.library.pcr_boundary_reserve.restype = ctypes.c_double
        self.library.pcr_boundary_finish.argtypes = [ctypes.c_double]
        self.library.pcr_boundary_finish.restype = None
        self.library.pcr_boundary_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64),
                                                   ctypes.POINTER(ctypes.c_uint64),
                                                   ctypes.POINTER(ctypes.c_double)]
        self.library.pcr_boundary_stats.restype = ctypes.c_int

    def begin(self, mbps):
        assert self.library.pcr_boundary_begin(mbps * 1_000_000) == 0

    def stats(self):
        size, calls, span = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_double()
        assert self.library.pcr_boundary_stats(ctypes.byref(size), ctypes.byref(calls),
                                               ctypes.byref(span)) == 0
        return dict(read_bytes=size.value, read_calls=calls.value, read_completion_span_s=span.value)


def model_configuration(name, device):
    """Reuse the paper's model constructors and checkpoints."""
    if name in ("vitti", "swinv2"):
        from fastlens.benchmarks.baselines.system_rgbnomore.inference import pipeline
        contract_path = ROOT / ("results/remaining_inference_recheck_20260929/inference/"
                                f"4090_{name}_pytorch/trial_1/contract.json")
        contract = json.loads(contract_path.read_text())
        if name == "vitti":
            from fastlens.benchmarks.system_dct_major import pipeline as major
            model = major.build_workload_model(
                rgbnomore_root=Path(contract["models"]["rgbnomore_root"]),
                checkpoint=Path(contract["models"]["rgb"]["checkpoint"]), device=device,
                domain="rgb", workload=contract["workload"]["kind"],
                feature_stage=contract["workload"].get("feature_stage", "penultimate"))
        else:
            model = pipeline._build_rgb_model(contract, device)
        return model, Path(contract["models"]["rgb"]["checkpoint"]), 256 if name == "swinv2" else 224
    if name == "efun":
        import sys
        import fastlens.benchmarks.dct_models as package
        from fastlens.benchmarks.dct_models import efun_backend
        sys.modules["fastlens.benchmarks.dct_models.backend"] = efun_backend
        package.backend = efun_backend
    from fastlens.benchmarks.dct_models import rgb
    return rgb.model().to(device), rgb.CHECKPOINT, 224


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--op", type=Path, required=True)
    parser.add_argument("--upstream-release", type=Path, required=True)
    parser.add_argument("--model", choices=("vitti", "swinv2", "mobilenet24", "efun"), required=True)
    parser.add_argument("--route", choices=("pcr", "jpeg_record", "jpeg_files"), required=True)
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--protocol", choices=("deployment", "bandwidth"), default="deployment")
    parser.add_argument("--source-crop", type=int, choices=(448, 160))
    parser.add_argument("--read-mbps", type=float, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--check", action="store_true", help="Excluded short check retaining full Top-5 predictions")
    parser.add_argument("--reference", type=Path, help="Same native-TF JPEG preprocessing reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new result output path")
    if args.protocol == "bandwidth" and (args.model != "mobilenet24" or args.source_crop is None):
        parser.error("Bandwidth controls require MobileNetV2 and an explicit source crop")
    if args.read_mbps < 0:
        parser.error("Read bandwidth must be nonnegative")
    if args.route == "jpeg_files" and args.read_mbps:
        parser.error("Limited JPEG controls use the native single-scan jpeg_record route")
    os.environ["DCTNET_PROFILE"] = args.model
    import cv2
    cv2.setNumThreads(1)
    import tensorflow as tf
    import torch
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    from .dataset import pcr_dataset, split_record

    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    tf.load_op_library(str(args.op))
    control = ReadControl(args.op)
    index = json.loads((args.data / "index.json").read_text())
    count = args.count
    if count is None and args.check:
        count = 100 if args.model == "swinv2" else 128
    count = count or (16384 if args.protocol == "bandwidth" else 50000)
    entries = index["entries"][:count]
    assert len(entries) == count
    records = [r for r in index["records"] if r["first"] < count]
    batch_size = 50 if args.model == "swinv2" else 64
    sample_ids = [e["logical_sample_id"] for e in entries]
    paths = ([args.data / r["path"] for r in records] if args.route != "jpeg_files"
             else [Path(e["path"]) for e in entries])
    sizes = [p.stat().st_size for p in paths]
    cache = cache_pages(paths, warm=True)
    assert cache["resident_fraction"] == 1
    model, checkpoint, image_size = model_configuration(args.model, device)
    model.eval()
    upstream = args.upstream_release / "tensorflow"
    preprocessing = import_file("pcr_preprocessing", upstream / "imagenet_preprocessing_input.py")
    prefetcher = import_file("pcr_prefetcher", upstream / "data_prefetcher.py")

    @tf.function(input_signature=[tf.TensorSpec([], tf.string), tf.TensorSpec([], tf.int32)])
    def preprocess(jpeg, label):
        if args.source_crop is not None or args.model == "swinv2":
            crop = args.source_crop or 512
            border = (512 - crop) // 2
            image = tf.image.decode_and_crop_jpeg(jpeg, [border, border, crop, crop], channels=3)
            image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]
            image = tf.reshape(image, [image_size, image_size, 3])
        else:
            image = preprocessing.preprocess_image_fn(jpeg, is_training=False, image_size=image_size)
        if args.model in ("vitti", "swinv2"):
            image = image / 127.5 - 1
        else:
            image = (image / 255.0 - tf.constant([.485, .456, .406])) / tf.constant([.229, .224, .225])
        return image, tf.cast(label, tf.int64)

    preprocess.get_concrete_function()

    def dataset():
        if args.route != "jpeg_files":
            source = pcr_dataset(args.op, [str(p) for p in paths], args.data / "PCR_index.pb",
                                 args.scans if args.route == "pcr" else 1,
                                 readers=8, block_length=index["records"][0]["count"])
            source = source.map(split_record, num_parallel_calls=8).take(count)
        else:
            source = tf.data.Dataset.from_tensor_slices(
                ([str(p) for p in paths], [e["model_label"] for e in entries], sizes))

            def read_jpeg(path, label, size):
                return tf.io.read_file(path), tf.cast(label, tf.int32)

            source = source.map(read_jpeg, num_parallel_calls=8)
        source = source.map(preprocess, num_parallel_calls=8).batch(batch_size).prefetch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 8
        options.threading.max_intra_op_parallelism = 1
        options.experimental_deterministic = True
        return source.with_options(options)

    with torch.inference_mode():
        dummy = torch.zeros(batch_size, 3, image_size, image_size, device=device).contiguous(memory_format=torch.channels_last)
        for _ in range(20):
            model(dummy)
        del dummy
        torch.cuda.synchronize()
        predictions = []
        correct1 = correct5 = observed = 0
        ce = 0.
        control.begin(args.read_mbps)
        started = time.perf_counter()
        loader = prefetcher.data_prefetcher(dataset(), math.ceil(count / batch_size), permute_channel=True)
        for _ in range(math.ceil(count / batch_size)):
            images, labels = next(loader)
            logits = model(images)
            ranks = logits.topk(5, dim=1).indices
            matches = ranks.eq(labels[:, None])
            correct1 += int(matches[:, :1].sum())
            correct5 += int(matches.sum())
            ce += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
            predictions.extend(ranks.cpu().tolist() if args.protocol == "deployment" or args.check
                               else ranks[:, 0].cpu().tolist())
            observed += len(labels)
            stream.synchronize()
            del images, labels, logits, ranks, matches
        if args.protocol == "bandwidth":
            del loader
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if args.protocol == "deployment":
            del loader
    assert observed == count
    boundary = control.stats()
    if args.route == "jpeg_files" and not args.read_mbps:
        boundary.update(read_bytes=sum(sizes), read_calls=count, read_completion_span_s=None)
    result = dict(model=args.model, route=args.route, scans=args.scans if args.route == "pcr" else None,
        protocol=args.protocol, correctness_check=args.check, samples=count, sample_ids=sample_ids, predictions=predictions,
        checkpoint=str(checkpoint), top1=100*correct1/count, top5=100*correct5/count, ce=ce/count,
        e2e_seconds=seconds, images_per_s=count/seconds, batch_size=batch_size, workers=8,
        host_threads=8, precision="fp32", tf32=False, compile=False, warmup_batches=20,
        cpu_affinity=sorted(os.sched_getaffinity(0)), gpu=torch.cuda.get_device_name(),
        tensorflow=tf.__version__, torch=torch.__version__, source_crop=args.source_crop,
        preprocessing="TF decode_and_crop_jpeg + author bicubic; model-specific normalization; Swin full source",
        implementation="author C++ PCR DatasetOp, TF2.15 API adapter; author TF-to-PyTorch prefetcher",
        storage_bytes=sum(sizes) if args.route == "jpeg_files" else sum(sum(r["sizes"]) for r in index["records"]),
        h2d_input_tensor_bytes=count*(3*image_size*image_size*4+8), cache=cache,
        read_mbps=args.read_mbps, input_boundary_control=boundary,
        jpeg_read_control="native single-scan JPEG records for bandwidth; tf.io.read_file for deployment references",
        data=str(args.data.resolve()), source_raw=index["source_raw"],
        timing_scope=("reader construction/startup through synchronized logits, CE, top1/top5 and "
                      + ("Top-5 copies; reader shutdown excluded" if args.protocol == "deployment" else
                         "Top-1 copies and reader shutdown") + "; model load, TF tracing and 20 model warmup batches excluded"))
    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert sample_ids == reference["sample_ids"] and result["checkpoint"] == reference["checkpoint"]
        assert args.source_crop == reference["source_crop"]
        result["prediction_agreement"] = sum(a == b for a, b in zip(predictions, reference["predictions"], strict=True))/count
        result["top1_delta_pp"] = result["top1"] - reference["top1"]
        if args.route == "jpeg_record" or (args.route == "pcr" and args.scans == 10):
            assert predictions == reference["predictions"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: result[k] for k in ("model", "route", "samples", "top1", "top5", "images_per_s")}), flush=True)


if __name__ == "__main__":
    main()
