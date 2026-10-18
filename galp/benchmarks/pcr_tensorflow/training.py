"""Native PCR inputs for the existing restored-state RGB training window.

The author native reader and TF decode/crop/resize path use the existing
training window's keyed boxes, flips, bilinear resize and [-1, 1] normalization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from galp.benchmarks.system_rgbnomore.training.pipeline import TrainingBatch, TrainingSample
from galp.benchmarks.system_rgbnomore.training.sample_order import SampleIdentity

from .run import import_file

SEED = 11997733

def preprocess_image(jpeg, decision):
    """TF JPEG/crop/resize with the original training window's requested box."""
    import tensorflow as tf
    image = tf.image.decode_and_crop_jpeg(jpeg, decision[:4], channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.image.resize(image, [224, 224], method="bilinear", antialias=False)
    image = tf.cond(decision[4] != 0, lambda: tf.reverse(image, [1]), lambda: image)
    return tf.reshape(image, [224, 224, 3]) / 127.5 - 1.


def crop_parameters(decision):
    return [decision.crop_y, decision.crop_x, decision.crop_height, decision.crop_width,
            int(decision.horizontal_flip)]


class NativeTrainingWindow:
    def __init__(self, index, root, op, upstream, samples, decisions, workers):
        import tensorflow as tf
        from .dataset import pcr_dataset, split_record
        self.samples = samples
        self.position = 0
        records = index["records"]
        dataset = pcr_dataset(op, [str(root / row["path"]) for row in records],
                              root / "PCR_index.pb", scans=10, readers=workers,
                              block_length=records[0]["count"])
        dataset = dataset.map(split_record, num_parallel_calls=workers, deterministic=True)
        boxes = tf.data.Dataset.from_tensor_slices(np.asarray(
            [crop_parameters(d) for d in decisions], dtype=np.int32))
        dataset = tf.data.Dataset.zip((dataset, boxes)).map(
            lambda item, box: (preprocess_image(item[0], box), tf.cast(item[1], tf.int64)),
            num_parallel_calls=workers, deterministic=True,
        ).batch(64).prefetch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = workers
        options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        prefetcher = import_file("pcr_training_prefetcher", upstream / "tensorflow/data_prefetcher.py")
        self.loader = prefetcher.data_prefetcher(dataset, len(samples) // 64, permute_channel=True)

    def next_batch(self):
        images, labels = next(self.loader)
        first = self.position
        self.position += len(labels)
        identities = [SampleIdentity(2, i, self.samples[i].logical_sample_id)
                      for i in range(first, self.position)]
        return TrainingBatch(inputs=(images,), labels=labels, identities=identities,
                             augmentations=[], on_device=True, stage_seconds={})

    def close(self):
        self.loader.stream.synchronize()
        self.loader = None


def training_inputs(args, device):
    import tensorflow as tf
    from galp.benchmarks.system_rgbnomore.training.augmentation import derive_augmentation
    tf.config.threading.set_inter_op_parallelism_threads(16)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    root = args.pcr_data
    index = json.loads((root / "index.json").read_text())
    entries = index["entries"]
    samples = [TrainingSample(e["logical_sample_id"], Path(e["path"]), e["model_label"],
                              e["width"], e["height"], e["galp_image_id"]) for e in entries]
    canonical = json.loads((Path(__file__).resolve().parents[3] /
                           "results/preparation/p0_ffcv/samples.json").read_text())
    assert [(s.logical_sample_id, s.label) for s in samples] == [
        (e["logical_sample_id"], e["label"]) for e in canonical]
    decisions = [derive_augmentation(seed=SEED, epoch=2, logical_sample_id=s.logical_sample_id,
                                     source_width=s.width, source_height=s.height, domain="rgb")
                 for s in samples]

    paths = [root / row["path"] for row in index["records"]]
    variant = dict(implementation="author C++ PCR DatasetOp; TensorFlow 2.15.1 API adapter",
                   preprocessing="native TF decode_and_crop_jpeg INTEGER_ACCURATE; bilinear without antialias; [-1,1]",
                   augmentation="original window per-sample keyed boxes/flip; seed 11997733, epoch 2",
                   scans=10, input_workers=16, source=str(root), op=str(args.pcr_op),
                   upstream_release=str(args.pcr_upstream_release), seed=SEED, epoch=2,
                   tensorflow=tf.__version__, source_samples=len(samples))
    first_batch_hash = None

    def make():
        nonlocal first_batch_hash
        source = NativeTrainingWindow(index, root, args.pcr_op, args.pcr_upstream_release, samples, decisions, 16)
        digest = hashlib.sha256()
        for tensor in (source.loader.next_input, source.loader.next_target):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        actual = digest.hexdigest()
        if first_batch_hash is None:
            first_batch_hash = actual
        else:
            assert actual == first_batch_hash, "TF input reconstruction changed the first batch"
            variant["input_restart_verified"] = True
        variant["first_batch_sha256"] = actual
        return source

    return make, paths, samples, decisions, variant


def check_input(images, labels, identities, samples, decisions, output):
    """Compare the native scan-10 path with original JPEGs under the same TF ops."""
    import tensorflow as tf
    from experiments.p0_adapters import check_rgb_input
    check_rgb_input(images, labels, identities, samples, decisions, pipeline="pcr",
                    device=images.device, dali_config={}, output=output)
    differences = []
    for i, sample in enumerate(samples[:4]):
        expected = preprocess_image(tf.constant(sample.path.read_bytes()),
                                    tf.constant(crop_parameters(decisions[i]), tf.int32))
        expected = torch.from_numpy(expected.numpy()).permute(2, 0, 1)
        delta = (images[i].detach().cpu() - expected).abs()
        row = dict(sample_id=sample.logical_sample_id, max_error=float(delta.max()), mean_error=float(delta.mean()))
        assert row["max_error"] <= 8 / 255 and row["mean_error"] <= 1 / 255, row
        differences.append(row)
    report = dict(passed=True, reference="original JPEG + identical keyed TF preprocessing",
                  samples=len(identities), tensor_checks=differences,
                  normalization_mean=(0.5, 0.5, 0.5), normalization_std=(0.5, 0.5, 0.5))
    (output / "pcr_input_check.json").write_text(json.dumps(report, indent=2) + "\n")
