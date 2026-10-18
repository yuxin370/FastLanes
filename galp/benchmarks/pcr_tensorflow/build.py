"""Build the authors' native PCR DatasetOp against TensorFlow CPU 2.15.1."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import urllib.request

COMMIT = "c3cc6a9bad115925cd398d01cedd85af68aa1be2"
SOURCES = [
    "tensorflow/core/example/progressive_compressed_record.proto",
    "tensorflow/core/kernels/data/progressive_compressed_record_dataset_op.cc",
    "tensorflow/core/kernels/data/progressive_compressed_record_dataset_op.h",
    "tensorflow/core/lib/io/progressive_compressed_record_reader.cc",
    "tensorflow/core/lib/io/progressive_compressed_record_reader.h",
    "tensorflow/core/util/pcr_parsing.cc",
    "tensorflow/core/util/pcr_parsing.h",
    "tensorflow/core/ops/dataset_ops.cc",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--protoc", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root = args.build_dir.resolve()
    upstream = root / "upstream"
    adapted = root / "adapted"

    def fetch(name):
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://raw.githubusercontent.com/mkuchnik/TF_PCR/{COMMIT}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(fetch, SOURCES))
    for name in SOURCES:
        if name.endswith("dataset_ops.cc"):
            continue
        text = (upstream / name).read_text()
        # TensorFlow 2.15 moved Status and file interfaces to TSL/absl.
        text = text.replace("Status::OK()", "OkStatus()")
        text = text.replace("s.error_message()", "s.message()")
        text = text.replace(
            '"tensorflow/core/common_runtime/metrics.h"',
            '"tensorflow/core/framework/metrics.h"',
        )
        text = text.replace(
            '"tensorflow/core/kernels/data/name_utils.h"',
            '"tensorflow/core/data/name_utils.h"',
        )
        text = text.replace('class RandomAccessFile;', '')
        text = text.replace(
            '#include "tensorflow/core/platform/types.h"',
            '#include "tensorflow/core/platform/types.h"\n'
            '#include "tensorflow/core/platform/file_system.h"',
        )
        if name.endswith("progressive_compressed_record_reader.cc"):
            text = text.replace("namespace tensorflow {", (
                'extern "C" double pcr_boundary_reserve(uint64_t);\n'
                'extern "C" void pcr_boundary_finish(double);\n\nnamespace tensorflow {'
            ), 1)
            text = text.replace(
                "  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(n, result));",
                "  const double deadline = pcr_boundary_reserve(n);\n"
                "  Status read_status = input_stream_->ReadNBytes(n, result);\n"
                "  pcr_boundary_finish(deadline);\n"
                "  TF_RETURN_IF_ERROR(read_status);",
            )
        path = adapted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    original_ops = (upstream / SOURCES[-1]).read_text()
    registration = original_ops.split('REGISTER_OP("ProgressiveCompressedRecordDataset")', 1)[1]
    registration = registration.split('\nREGISTER_OP("Iterator")', 1)[0]
    registration_path = adapted / "registration.cc"
    registration_path.write_text(
        '#include "tensorflow/core/framework/op.h"\n'
        '#include "tensorflow/core/framework/common_shape_fns.h"\n'
        'namespace tensorflow {\nREGISTER_OP("ProgressiveCompressedRecordDataset")'
        + registration + "\n}\n"
    )
    import tensorflow as tf

    subprocess.run([
        str(args.protoc), f"--proto_path={adapted}",
        f"--proto_path={tf.sysconfig.get_include()}", f"--cpp_out={adapted}",
        f"--python_out={adapted}", str(adapted / SOURCES[0]),
    ], check=True)

    if tf.__version__ != "2.15.1":
        raise RuntimeError(f"This API adapter targets TensorFlow 2.15.1, got {tf.__version__}")
    compiler = os.environ.get("CXX", "clang++")
    flags = ["-O3", "-DNDEBUG", "-fPIC", f"-I{adapted}", *tf.sysconfig.get_compile_flags()]
    units = [adapted / name for name in SOURCES[:-1] if name.endswith(".cc")]
    units += [adapted / SOURCES[0].replace(".proto", ".pb.cc"), registration_path]
    boundary = adapted / "read_boundary.cc"
    boundary.write_text(Path(__file__).with_name("read_boundary.cc").read_text())
    units.append(boundary)

    def compile_one(source):
        obj = source.with_suffix(".o")
        subprocess.run([compiler, *flags, "-c", str(source), "-o", str(obj)], check=True)
        return obj

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        objects = list(pool.map(compile_one, units))
    output = root / "pcr_dataset.so"
    subprocess.run([
        compiler, "-shared", *map(str, objects), *tf.sysconfig.get_link_flags(),
        "-l:libtensorflow_cc.so.2",
        f"-Wl,-rpath,{tf.sysconfig.get_lib()}", "-o", str(output),
    ], check=True)
    tf.load_op_library(str(output))
    print(f"Loaded native PCR DatasetOp: {output}")


if __name__ == "__main__":
    main()
