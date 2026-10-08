"""Build the L3 codec and DALI operator against the active DALI installation."""
import argparse
import subprocess
from pathlib import Path

from nvidia.dali import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(args.nvcc), "-shared", "-std=c++20", "-O3", "-Xcompiler", "-fPIC",
        "-gencode=arch=compute_89,code=sm_89", "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
        *sysconfig.get_compile_flags(), "-I" + str(source / "upstream"),
        str(source / "upstream/encoder.cu"), str(source / "upstream/decoder.cu"),
        str(source / "codec.cu"), str(source / "dali_decoder.cc"), "-rdc=true",
        *sysconfig.get_link_flags(), "-Xlinker", "-rpath", "-Xlinker", sysconfig.get_lib_dir(),
        "-o", str(args.output),
    ], check=True)


if __name__ == "__main__":
    main()
