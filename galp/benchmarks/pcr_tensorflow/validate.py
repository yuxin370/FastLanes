"""Check every native reconstructed JPEG and label against the existing adapter."""
import argparse
import json
from pathlib import Path

import tensorflow as tf

from galp.benchmarks.pcr.storage import read_record
from .dataset import interleaved_ordinals, pcr_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--op", type=Path, required=True)
    parser.add_argument("--scans", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--readers", type=int, default=8)
    args = parser.parse_args()
    tf.config.threading.set_inter_op_parallelism_threads(args.readers)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    index = json.loads((args.data / "index.json").read_text())
    raw_root = Path(index["source_raw"])
    raw_index = json.loads((raw_root / "index.json").read_text())
    files = [str(args.data / record["path"]) for record in index["records"]]
    for scans in args.scans:
        expected = []
        for record in raw_index["records"]:
            images, labels, _ = read_record(raw_root, record["pcr"], scans)
            expected.extend((label.to_bytes(4, "little", signed=True) + image)
                            for label, image in zip(labels, images, strict=True))
        dataset = pcr_dataset(args.op, files, args.data / "PCR_index.pb", scans, args.readers)
        options = tf.data.Options()
        options.threading.private_threadpool_size = args.readers
        dataset = dataset.with_options(options)
        order = interleaved_ordinals(index["records"], args.readers)
        seen = 0
        for value in dataset.as_numpy_iterator():
            if value != expected[order[seen]]:
                raise AssertionError(f"Unexpected native output at scan {scans}, sample {seen}")
            seen += 1
        if seen != len(index["entries"]):
            raise AssertionError("Incomplete sample coverage")
        print(json.dumps(dict(scans=scans, samples=seen, jpeg_bytes_and_labels_exact=True,
                              sample_order_exact=True)), flush=True)


if __name__ == "__main__":
    main()
