"""Thin Python handle for the authors' C++ DatasetOp (no Python record loader)."""
from pathlib import Path

import tensorflow as tf


def interleaved_ordinals(records, readers):
    """Order of deterministic interleave for the prepared fixed-size shards."""
    shard_size = records[0]["count"]
    if any(record["count"] != shard_size for record in records[:-1]):
        raise ValueError("Expected fixed-size PCR shards followed by one final shard")
    return [record["first"] + i
            for start in range(0, len(records), readers)
            for i in range(shard_size)
            for record in records[start:start + readers] if i < record["count"]]


def pcr_dataset(op_path: Path, filenames, index_path: Path, scans: int, readers: int = 8,
                block_length: int = 1):
    op = tf.load_op_library(str(op_path))

    def source(filename):
        variant = op.progressive_compressed_record_dataset(
            filenames=filename, compression_type="", buffer_size=tf.constant(0, tf.int64),
            scan_groups=scans, index_source_filename=str(index_path),
            metadata_output_type="labels_first",
        )
        return tf.data.experimental.from_variant(variant, tf.TensorSpec([], tf.string))

    # Match the original wrapper's parallel file interleave, retaining its C++
    # scan-prefix read, protobuf parsing, buffer reuse and image reconstruction.
    return tf.data.Dataset.from_tensor_slices(filenames).interleave(
        source, cycle_length=readers, block_length=block_length,
        num_parallel_calls=readers, deterministic=True,
    )


def split_record(record):
    label = tf.io.decode_raw(tf.strings.substr(record, 0, 4), tf.int32)[0]
    jpeg = tf.strings.substr(record, 4, -1)
    return jpeg, label
