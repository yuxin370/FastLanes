# ────────────────────────────────────────────────────────
# |                      FastLanes                       |
# ────────────────────────────────────────────────────────
# galp/scripts/codegen/generate_multicolumn_kernels.py
# ────────────────────────────────────────────────────────
#!/usr/bin/env python3

import os
import sys

import argparse
import logging

FILE_HEADER = """
// galp/benchmarks/include/galp_bench/generated/multi_column_device_kernels.cuh
#include "codecs/decode/alp.cuh"
#include "cuda/device_utils.cuh"

#ifndef GALP_BENCH_GENERATED_MULTI_COLUMN_DEVICE_KERNELS_CUH
#define GALP_BENCH_GENERATED_MULTI_COLUMN_DEVICE_KERNELS_CUH

namespace galp::bench::multi_column {
"""

FILE_FOOTER = """
}
#endif // GALP_BENCH_GENERATED_MULTI_COLUMN_DEVICE_KERNELS_CUH
"""

HOST_FILE_HEADER = """// galp/benchmarks/include/galp_bench/generated/multi_column_host_kernels.cuh
#ifndef GALP_BENCH_GENERATED_MULTI_COLUMN_HOST_KERNELS_CUH
#define GALP_BENCH_GENERATED_MULTI_COLUMN_HOST_KERNELS_CUH

#include "galp_bench/data.cuh"
#include "cuda/device_utils.cuh"
#include "galp_bench/generated/multi_column_device_kernels.cuh"

namespace galp::kernels { namespace host {

template <typename T, unsigned UNPACK_N_VECS, unsigned UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__host__ bool query_multi_column(const ColumnT& column, const T magic_value, const uint32_t n_samples) {
\tusing DeviceColumnT          = typename ColumnT::DeviceColumnT;
\tconstexpr int32_t MAX_N_COLS = 10;
\tDeviceColumnT     device_columns[MAX_N_COLS];

\tfor (int32_t c {0}; c < MAX_N_COLS; ++c) {
\t\t// INFO: Bitwidths are shuffled to ensure that the vectors that
\t\t// are unpacked in the same loop do not have an identical bitwidth,
\t\t// as this might benefit branched unpackers
\t\tgalp::bench::columns::shuffle_bit_widths(column);
\t\tdevice_columns[c] = column.copy_to_device();
\t}

\tbool                        result = false;
\tGPUArray<bool>              d_out(1, &result);
\tconst ThreadblockMapping<T> mapping(UNPACK_N_VECS, column.get_n_vecs());
"""

HOST_FILE_FOOTER = """
\tfor (int32_t c {0}; c < MAX_N_COLS; ++c) {
\t\tgalp::codec::host::free_column(device_columns[c]);
\t}

\td_out.copy_to_host(&result);
\treturn result;
}
}} // namespace galp::kernels::host

#endif // GALP_BENCH_GENERATED_MULTI_COLUMN_HOST_KERNELS_CUH
"""

def generate_global_function(n_cols: int):
    col_range = range(n_cols)
    return "\n".join(
        [
            "template <typename T, unsigned UNPACK_N_VECS, unsigned UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>",
            "__global__ void query_multi_column("
            + ",".join([f"const ColumnT column_{i}" for i in col_range])
            + ", const T value, bool *out) {",
            f"constexpr int32_t N_COLS = {n_cols};",
            "const auto mapping = VectorToWarpMapping<T, UNPACK_N_VECS>();",
            "const int32_t vector_index = mapping.get_vector_index();",
            "const lane_t lane = mapping.get_lane();",
            f"T registers[UNPACK_N_VALUES * UNPACK_N_VECS * {n_cols}];",
            "bool all_columns_equal = true;",
            "\n".join(
                [
                    f"DecompressorT decompressor_{i} = DecompressorT(column_{i}, vector_index, lane);"
                    for i in col_range
                ]
            ),
            "for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {",
            "\n".join(
                [
                    f"decompressor_{i}.unpack_next_into(registers + {i} * (UNPACK_N_VALUES * UNPACK_N_VECS));"
                    for i in col_range
                ]
            ),
            "#pragma unroll",
            "for (int c{1}; c < N_COLS; ++c) {",
            "#pragma unroll",
            "for (int va{0}; va < UNPACK_N_VALUES; ++va) {",
            "#pragma unroll",
            "for (int v{0}; v < UNPACK_N_VECS; ++v) {",
            "all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * UNPACK_N_VECS * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * UNPACK_N_VECS * UNPACK_N_VALUES];",
            "}",
            "}",
            "}",
            "#pragma unroll",
            "for (int va{0}; va < UNPACK_N_VALUES; ++va) {",
            "#pragma unroll",
            "for (int v{0}; v < UNPACK_N_VECS; ++v) {",
            "all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;",
            "}",
            "}",
            "}",
            "",
            "if (all_columns_equal) {",
            "*out = true;",
            "}}",
        ]
    )


def generate_host_launch(n_cols: int):
    columns = ", ".join([f"device_columns[{idx}]" for idx in range(n_cols)])
    args = ", ".join([columns, "magic_value", "d_out.get()"])
    return "\n".join(
        [
            "\tfor (uint32_t repeat {0}; repeat < n_samples; ++repeat) {",
            "\t\tgalp::bench::multi_column::query_multi_column<T, UNPACK_N_VECS, UNPACK_N_VALUES, DecompressorT, DeviceColumnT>",
            f"\t\t    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>({args});",
            "\t\tCUDA_SAFE_CALL(cudaDeviceSynchronize());",
            "\t}",
        ]
    )


def main(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    n_cols = 10
    device_str = "\n\n".join(
        [
            FILE_HEADER,
            *[generate_global_function(n) for n in range(1, n_cols + 1)],
            FILE_FOOTER,
        ]
    )
    host_str = "\n".join(
        [
            HOST_FILE_HEADER,
            *[generate_host_launch(n) for n in range(1, n_cols + 1)],
            HOST_FILE_FOOTER,
        ]
    )

    with open(os.path.join(out_dir, "multi_column_device_kernels.cuh"), "w") as file:
        file.write(device_str)
    with open(os.path.join(out_dir, "multi_column_host_kernels.cuh"), "w") as file:
        file.write(host_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "-ll",
        "--logging-level",
        type=int,
        default=logging.INFO,
        choices=[logging.CRITICAL, logging.ERROR, logging.INFO, logging.DEBUG],
        help=f"logging level to use: {logging.CRITICAL}=CRITICAL, {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
        + f"{logging.DEBUG}=DEBUG, higher number means less output",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="Directory to write generated multi-column benchmark headers into.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )
    main(args)
