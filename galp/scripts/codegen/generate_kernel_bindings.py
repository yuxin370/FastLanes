# ────────────────────────────────────────────────────────
# |                      FastLanes                       |
# ────────────────────────────────────────────────────────
# galp/scripts/codegen/generate_kernel_bindings.py
# ────────────────────────────────────────────────────────
#!/usr/bin/python3

import json
import os
import sys
from pathlib import Path

import argparse
import logging

# Resolved from argparse at startup; see __main__ block.
GENERATED_BINDINGS_DIR: str = ""
GENERATED_HEADERS_DIR: str | None = None
WRITTEN_BINDINGS: set[str] = set()
WRITTEN_HEADERS: set[str] = set()

MANIFEST_PATH = Path(__file__).with_name("generated_bindings_manifest.json")


def load_generated_bindings_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("schema") != "galp-generated-bindings-v1":
        raise RuntimeError(f"unsupported generated binding manifest: {MANIFEST_PATH}")
    for field in ("bindings", "headers", "optional_bindings"):
        if field not in manifest:
            raise RuntimeError(f"generated binding manifest lacks {field!r}: {MANIFEST_PATH}")
    return manifest


GENERATED_BINDINGS_MANIFEST = load_generated_bindings_manifest()

FILE_HEADER = """
#include "cuda/launch/dispatch.cuh"
#include "galp_bench/generated/kernel_bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {
"""

FILE_FOOTER = """
}
"""

KERNEL_BINDINGS_HEADER = """// galp/benchmarks/include/galp_bench/generated/kernel_bindings.cuh
#ifndef GALP_BENCH_GENERATED_KERNEL_BINDINGS_CUH
#define GALP_BENCH_GENERATED_KERNEL_BINDINGS_CUH

#include "core/enums.cuh"
#include "codecs/decode/alp.cuh"
#include <cstdint>

namespace galp::bench::bindings {

template <typename T, typename ColumnT>
T* decompress_column(const ColumnT column,
                     const unsigned unpack_n_vectors,
                     const unsigned unpack_n_values,
                     const galp::format::Unpacker unpacker,
                     const galp::format::Patcher patcher,
                     const galp::format::Expander expander,
                     const uint32_t n_samples);

template <typename T, typename ColumnT>
T* decompress_column(const ColumnT column,
                     const unsigned unpack_n_vectors,
                     const unsigned unpack_n_values,
                     const galp::format::Unpacker unpacker,
                     const galp::format::Patcher patcher,
                     const galp::format::Expander expander,
                     const uint32_t n_samples,
                     const bool use_shuffle);

template <typename T, typename ColumnT>
bool query_column(const ColumnT column,
                  const unsigned unpack_n_vectors,
                  const unsigned unpack_n_values,
                  const galp::format::Unpacker unpacker,
                  const galp::format::Patcher patcher,
                  const T magic_value,
                  const uint32_t n_samples);

template <typename T, typename ColumnT>
bool compute_column(const ColumnT column,
                    const unsigned unpack_n_vectors,
                    const unsigned unpack_n_values,
                    const galp::format::Unpacker unpacker,
                    const galp::format::Patcher patcher,
                    const unsigned n_repetitions,
                    const uint32_t n_samples);

} // namespace galp::bench::bindings

#endif // GALP_BENCH_GENERATED_KERNEL_BINDINGS_CUH
"""


DATA_TYPES = [
    "int8_t",
    "int16_t",
    "uint32_t",
    "uint64_t",
    "float",
    "double",
]

FUNCTIONS = [
    "decompress_column",
    "query_column",
    "compute_column",
]

ENCODINGS = [
    "BP",
    "FFOR",
    "DELTA",
    "ALP",
    "ALPExtended",
    "FREQ",
    "FREQExtended",
    "DICT",
    "DICTSLPATCH",
    "DICTShfl32",
    "CROSSRLE",
    "CROSSRLEExtended",
    "CROSSRLELaneMask",
    "SLPATCH",
    "RLE",
    "CONSTANT",
]

UNPACKERS = [
    "None",
    "Dummy",
    "OldFls",
    #"SwitchCase",
    #"Stateless",
    #"StatelessBranchless",
    #"StatefulCache",
    #"StatefulLocal1",
    #"StatefulLocal2",
    #"StatefulLocal4",
    #"StatefulShared1",
    #"StatefulShared2",
    #"StatefulShared4",
    #"StatefulRegister1",
    #"StatefulRegister2",
    #"StatefulRegister4",
    #"StatefulRegisterBranchless1",
    #"StatefulRegisterBranchless2",
    #"StatefulRegisterBranchless4",
    "StatefulBranchless",
]

EXPANDERS = [
    "None",
    "Dummy",
    "Stateful",
    "StatefulCache",
    "PrefetchStateful",
    "StatefulShuffle",
    "StatefulAdvance",
    "StatefulExtended",
    "Branchless",
    "PrefetchBranchless"
]

PATCHERS = [
    "None",
    "Dummy",
    "Stateless",
    "Stateful",
    "Naive",
    "NaiveBranchless",
    "PrefetchAll",
    "PrefetchAllBranchless",
]
def get_column_t(encoding: str, data_type: str, function: str) -> str:
    column_t = f"BPColumn<{data_type}>"
    if "CROSSRLEExtended" in encoding:
        column_t = f"CROSSRLEExtendedColumn<{data_type}>"
    elif "CROSSRLELaneMask" in encoding:
        column_t = f"CROSSRLELaneMaskColumn<{data_type}>"
    elif "CROSSRLE" in encoding:
        column_t = f"CROSSRLEColumn<{data_type}>"
    elif "DICTSLPATCH" in encoding:
        column_t = f"DICTSLPATCHColumn<{data_type}>"
    elif encoding == "DELTA":
        column_t = f"DELTAColumn<{data_type}>"
    elif "FFOR" in encoding:
        column_t = f"FFORColumn<{data_type}>"
    elif "SLPATCH" in encoding:
        column_t = f"SLPATCHColumn<{data_type}>"
    elif "ALPExtended" in encoding:
        column_t = f"ALPExtendedColumn<{data_type}>"
    elif "ALP" in encoding:
        column_t = f"ALPColumn<{data_type}>"
    elif "FREQExtended" in encoding:
        column_t = f"FREQExtendedColumn<{data_type}>"
    elif "FREQ" in encoding:
        column_t = f"FREQColumn<{data_type}>"
    elif "DICT" in encoding or "DICTShfl32" in encoding:
        column_t = f"DICTFFORColumn<{data_type}>"
    elif encoding == "RLE":
        column_t = f"RLEColumn<{data_type}, {data_type}>"
    elif "CONSTANT" in encoding:
        column_t = f"CONSTANTColumn<{data_type}>"
    return "galp::codec::device::" + column_t


def get_decompressor_type(
    encoding: str,
    data_type: str,
    function: str,
    unpacker: str,
    patcher: str,
    expander: str,
    n_vec: int,
    n_val: int,
) -> str:
    column_t = get_column_t(encoding, data_type, function)
    functor = f"BPFunctor<{data_type}>"
    patcher_t = f""
    decompressor_t = f"{encoding}Decompressor"
    expander_t = f""
    if encoding == "DELTA":
        uint_t = f"typename galp::codec::utils::same_width_uint<{data_type}>::type"
        functor = f"FFORFunctor<{uint_t}, {n_vec}>"
        decompressor_t = "DELTADecompressor"
    elif "FFOR" in encoding:
        functor = f"FFORFunctor<{data_type}, {n_vec}>"
    elif "SLPATCH" in encoding and "DICTSLPATCH" not in encoding:
        functor = f"FFORFunctor<{data_type}, {n_vec}>"
        patcher_t = f"galp::codec::device::{patcher}SLPATCHExceptionPatcher<{data_type}, {n_vec}, {n_val}>,"
        decompressor_t = "SLPATCHDecompressor"
    elif "ALPExtended" in encoding:
        functor = f"ALPFunctor<{data_type}, {n_vec}>"
        patcher_t = f"galp::codec::device::{patcher}ALPExceptionPatcher<{data_type}, {n_vec}, {n_val}>,"
        decompressor_t = "ALPDecompressor"
    elif "ALP" in encoding:
        functor = f"ALPFunctor<{data_type}, {n_vec}>"
        patcher_t = f"galp::codec::device::{patcher}ALPExceptionPatcher<{data_type}, {n_vec}, {n_val}>,"
    elif "FREQExtended" in encoding:
        patcher_t = f"galp::codec::device::{patcher}FREQExceptionPatcher<{data_type}, {n_vec}, {n_val}>,"
        decompressor_t = "FREQDecompressor"
    elif "FREQ" in encoding:
        patcher_t = f"galp::codec::device::{patcher}FREQExceptionPatcher<{data_type}, {n_vec}, {n_val}>,"
    elif "DICTSLPATCH" in encoding:
        functor = f"DICTFunctor<{data_type}, {n_vec}>"
        index_t = f"galp::codec::utils::same_width_uint<{data_type}>::type"
        patcher_t = (
            f"galp::codec::device::{patcher}SLPATCHDictExceptionPatcher<"
            f"{data_type}, {index_t}, {n_vec}, {n_val}>,"
        )
        decompressor_t = "DICTSLPATCHDecompressor"
    elif "DICTShfl32" in encoding:
        functor = f"DICTShfl32Functor<{data_type}, {n_vec}>"
    elif "DICT" in encoding:
        functor = f"DICTFunctor<{data_type}, {n_vec}>"
    elif encoding == "RLE":
        code_t = data_type
        functor = f"FFORFunctor<{code_t}, {n_vec}>"
        decompressor_t = "RLEDecompressor"
        expander_name = "Dummy" if expander == "None" else expander
    elif "CONSTANT" in encoding:
        decompressor_t = "CONSTANTDecompressor"
    elif "CROSSRLEExtended" in encoding:
        expander_t = f"galp::codec::device::{expander}CROSSRLEExpander<{data_type}, {n_vec}, {n_val}>,"
        decompressor_t = "CROSSRLEDecompressor"
    elif "CROSSRLELaneMask" in encoding:
        expander_t = f"galp::codec::device::{expander}CROSSRLEExpander<{data_type}, {n_vec}, {n_val}>,"
        decompressor_t = "CROSSRLEDecompressor"
    elif "CROSSRLE" in encoding:
        expander_t = f"galp::codec::device::{expander}CROSSRLEExpander<{data_type}, {n_vec}, {n_val}>,"

    loader_t = ""
    if "Stateful" in unpacker and "StatefulBranchless" not in unpacker:
        loader_t = ", galp::codec::device::"
        if "Cache" in unpacker:
            loader_t += f"CacheLoader<{data_type}, {n_vec}>"
        elif "Local" in unpacker:
            loader_t += f"LocalMemoryLoader<{data_type}, {n_vec}, {unpacker[-1]}>"
        elif "Shared" in unpacker:
            loader_t += f"SharedMemoryLoader<{data_type}, {n_vec}, {unpacker[-1]}>"
        elif "RegisterBranchless" in unpacker:
            loader_t += (
                f"RegisterBranchlessLoader<{data_type}, {n_vec}, {unpacker[-1]}>"
            )
        elif "Register" in unpacker:
            loader_t += f"RegisterLoader<{data_type}, {n_vec}, {unpacker[-1]}>"
        unpacker = "Stateful"

    unpacker_data_type = (
        f"typename galp::codec::utils::same_width_uint<{data_type}>::type"
        if encoding == "DELTA"
        else data_type
    )
    unpacker_t = f"galp::codec::device::BitUnpacker{unpacker}<{unpacker_data_type}, {n_vec}, {n_val},  galp::codec::device::{functor} {loader_t}>,"

    if "FREQ" in encoding or "FREQExtended" in encoding or "CROSSRLE" in encoding or "CROSSRLEExtended" in encoding or "CROSSRLELaneMask" in encoding:
        unpacker_t = f""
    if "CONSTANT" in encoding:
        return f"galp::codec::device::{decompressor_t}<{data_type}, {n_vec}, {n_val}, {column_t}>"
    if encoding == "RLE":
        rle_expander_t = (
            f"galp::codec::device::{expander_name}RLEExpander<"
            f"{data_type}, {code_t}, {n_vec}, {n_val}>,"
        )
        return f"galp::codec::device::{decompressor_t}<{data_type}, {code_t}, {n_vec}, {n_val}, {unpacker_t} {rle_expander_t} {column_t}>"
    if "DICTSLPATCH" in encoding:
        return f"galp::codec::device::{decompressor_t}<{data_type}, {n_vec}, {n_val}, {unpacker_t} {patcher_t} {column_t}>"
    return f"galp::codec::device::{decompressor_t}<{data_type}, {n_vec}, {unpacker_t} {patcher_t} {expander_t} {column_t}>"


def get_if_statement(
    encoding: str,
    data_type: str,
    function: str,
    n_vec: int,
    n_val: int,
    unpacker: str,
    patcher: str,
    expander: str = "None",
    is_query_column: bool = False,
    n_repetitions: int | None = None,
) -> str:

    assert data_type in DATA_TYPES
    assert function in FUNCTIONS
    assert n_vec in [1, 2, 4, 8]
    assert n_val in [1, 32]
    assert unpacker in UNPACKERS
    assert patcher in PATCHERS
    assert expander in EXPANDERS

    column_t = get_column_t(encoding, data_type, function)
    decompressor_t = get_decompressor_type(
        encoding, data_type, function, unpacker, patcher, expander, n_vec, n_val
    )
    extra_param = ", magic_value" if is_query_column else ""
    untransposer_t = (
        ", galp::codec::device::FastLanes1024InputUntransposer"
        if encoding == "DELTA" and function == "decompress_column"
        else ""
    )

    if encoding == "CROSSRLE" or encoding == "CROSSRLEExtended" or encoding == "CROSSRLELaneMask":
        return (
            f"if (unpack_n_vectors == {n_vec} && unpack_n_values == {n_val} && expander == galp::format::Expander::{expander}) "
            + "{"  # }
            f"return galp::kernels::host::{function}<{data_type}, {n_vec}, {n_val}, {decompressor_t}, {column_t} {',' + str(n_repetitions) if n_repetitions else ''}>(column {extra_param}, n_samples);"
            "}"
        )
    if encoding == "CONSTANT":
        return (
            f"if (unpack_n_vectors == {n_vec} && unpack_n_values == {n_val}) "
            + "{"  # }
            f"return galp::kernels::host::{function}<{data_type}, {n_vec}, {n_val}, {decompressor_t}, {column_t} {',' + str(n_repetitions) if n_repetitions else ''}>(column {extra_param}, n_samples);"
            "}"
        )
    if encoding == "FREQ" or encoding == "FREQExtended":
        return (
            f"if (unpack_n_vectors == {n_vec} && unpack_n_values == {n_val} && patcher == galp::format::Patcher::{patcher}) "
            + "{"  # }
            f"return galp::kernels::host::{function}<{data_type}, {n_vec}, {n_val}, {decompressor_t}, {column_t} {',' + str(n_repetitions) if n_repetitions else ''}>(column {extra_param}, n_samples);"
            "}"
        )
    return (
        f"if (unpack_n_vectors == {n_vec} && unpack_n_values == {n_val} && unpacker == galp::format::Unpacker::{unpacker} && patcher == galp::format::Patcher::{patcher}) "
        + "{"  # }
        f"return galp::kernels::host::{function}<{data_type}, {n_vec}, {n_val}, {decompressor_t}, {column_t}{untransposer_t} {',' + str(n_repetitions) if n_repetitions else ''}>(column {extra_param}, n_samples);"
        "}"
    )


def get_function(
    encoding: str,
    data_type: str,
    function: str,
    return_type: str,
    content: list[str],
    is_query_column: bool = False,
    is_compute_column: bool = False,
) -> str:
    column_t = get_column_t(encoding, data_type, function)
    expander_param = ", const galp::format::Expander expander" if function == "decompress_column" else ""
    magic_param = f", const {data_type} magic_value" if is_query_column else ""
    repetitions_param = ", const unsigned n_repetitions" if is_compute_column else ""
    return (
        f"template<> {return_type} {function}<{data_type},{column_t}>(const {column_t} column, const unsigned unpack_n_vectors, const unsigned unpack_n_values, const galp::format::Unpacker unpacker, const galp::format::Patcher patcher{expander_param}{magic_param}{repetitions_param}, const uint32_t n_samples)"
        + "{"
        + "\n".join(content)
        + f'throw std::invalid_argument("Could not find correct binding in {function} {encoding}<{data_type}>");'
        + "}"
    )


def write_file(
    file_name: str,
    functions: list[str],
):
    if file_name in WRITTEN_BINDINGS:
        raise RuntimeError(f"duplicate generated binding output: {file_name}")
    WRITTEN_BINDINGS.add(file_name)
    logging.info(f"Writing file {file_name}")
    out_path = os.path.join(GENERATED_BINDINGS_DIR, file_name)
    with open(out_path, "w") as f:
        f.write("\n".join([FILE_HEADER] + functions + [FILE_FOOTER]))


def write_kernel_bindings_header():
    if GENERATED_HEADERS_DIR is None:
        return
    file_name = "kernel_bindings.cuh"
    if file_name in WRITTEN_HEADERS:
        raise RuntimeError(f"duplicate generated header output: {file_name}")
    WRITTEN_HEADERS.add(file_name)
    os.makedirs(GENERATED_HEADERS_DIR, exist_ok=True)
    out_path = os.path.join(GENERATED_HEADERS_DIR, file_name)
    with open(out_path, "w") as f:
        f.write(KERNEL_BINDINGS_HEADER)


def get_if_statement_check_wrapper(
    disable_unnecessary: bool,
    encoding: str,
    data_type: str,
    function: str,
    n_vec: int,
    n_val: int,
    unpacker: str,
    patcher: str,
    expander: str = "None",
    is_query_column: bool = False,
    n_repetitions: int | None = None,
) -> str:
    # du handling
    is_necessary = (
        data_type == "float"
        and function == "decompress_column"
        and unpacker == "OldFls"
    )

    # OldFls handling
    if unpacker == "OldFls":
        n_val = 32

    # Filters
    unnessary_filter = disable_unnecessary and not is_necessary
    switch_case_filter = unpacker == "SwitchCase" and (
        n_vec != 1
        or n_val != 1
        or "uint" not in data_type
        or function == "compute_column"
    )
    legacy_fastlanes_filter = unpacker == "OldFls" and (
        n_vec != 1 or data_type not in ["uint32_t", "float"]
    )
    is_filtered = (
        unnessary_filter
        or switch_case_filter
        or legacy_fastlanes_filter
    )
    if is_filtered:
        return ""

    return get_if_statement(
        encoding,
        data_type,
        function,
        n_vec,
        n_val,
        unpacker,
        patcher,
        expander,
        is_query_column,
        n_repetitions,
    )


def main(args):
    WRITTEN_BINDINGS.clear()
    WRITTEN_HEADERS.clear()
    for encoding, patchers_per_encoding in zip(
        ["FREQ", "FREQExtended"], [PATCHERS[1:4], PATCHERS[4:]]
    ):
        for data_type in ["int8_t", "int16_t", "uint32_t", "uint64_t"]:
            for binding in ["decompress_column"]:
                is_query_column = binding == "query_column"
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    "None",
                                    patcher,
                                    is_query_column=is_query_column,
                                    n_repetitions=None,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for patcher in patchers_per_encoding
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    for encoding, expander_per_encoding in zip(
        ["CROSSRLE", "CROSSRLEExtended","CROSSRLELaneMask"], [EXPANDERS[1:7], EXPANDERS[7:8],EXPANDERS[8:]]
    ):
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, is_query_column in zip(
                ["decompress_column"], [False]
            ):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    "None",
                                    "None",
                                    expander,
                                    is_query_column=is_query_column,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for expander in expander_per_encoding
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )


    for encoding in ["DICT"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, is_query_column in zip(
                ["decompress_column"], [False]
            ):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "None",
                                    is_query_column=is_query_column,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS[1:]
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    for encoding in ["DICTSLPATCH"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, is_query_column in zip(
                ["decompress_column"], [False]
            ):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "Stateful",
                                    is_query_column=is_query_column,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS[1:]
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    if args.enable_dictshfl32:
        for encoding in ["DICTShfl32"]:
            for data_type in ["uint32_t", "uint64_t"]:
                for binding, is_query_column in zip(
                    ["decompress_column"], [False]
                ):
                    write_file(
                        f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                        [
                            get_function(
                                encoding,
                                data_type,
                                binding,
                                "bool" if is_query_column else data_type + "*",
                                [
                                    get_if_statement_check_wrapper(
                                        args.disable_unnecessary,
                                        encoding,
                                        data_type,
                                        binding,
                                        n_vec,
                                        n_val,
                                        unpacker,
                                        "None",
                                        is_query_column=is_query_column,
                                    )
                                    for n_vec in [1, 4]
                                    for n_val in [1]
                                    for unpacker in UNPACKERS[1:]
                                ],
                                is_query_column=is_query_column,
                            )
                        ],
                    )

    for encoding in ["BP", "FFOR"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, is_query_column in zip(
                ["decompress_column", "query_column"], [False, True]
            ):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "None",
                                    is_query_column=is_query_column,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS[1:]
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    for encoding in ["FFOR"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding in ["compute_column"]:
                is_compute_column = binding == "compute_column"
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "None",
                                    n_repetitions=10 if is_compute_column else None,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS[1:]
                            ],
                            is_compute_column=is_compute_column,
                        )
                    ],
                )

    for encoding in ["FFOR", "DELTA"]:
        for data_type in ["int8_t", "int16_t"]:
            binding = "decompress_column"
            write_file(
                f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                [
                    get_function(
                        encoding,
                        data_type,
                        binding,
                        data_type + "*",
                        [
                            get_if_statement_check_wrapper(
                                args.disable_unnecessary,
                                encoding,
                                data_type,
                                binding,
                                n_vec,
                                1,
                                "StatefulBranchless",
                                "None",
                            )
                            for n_vec in [1, 2, 4]
                        ],
                    )
                ],
            )

    for encoding in ["SLPATCH"]:
        for data_type in ["int16_t", "uint32_t", "uint64_t"]:
            for binding, is_query_column in zip(["decompress_column"], [False]):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "Stateful",
                                    is_query_column=is_query_column,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS[1:]
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    for encoding in ["RLE"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, is_query_column in zip(["decompress_column"], [False]):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    "None",
                                    is_query_column=is_query_column,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS[1:]
                                
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    for encoding in ["CONSTANT"]:
        for data_type in ["uint32_t", "uint64_t"]:
            for binding, is_query_column in zip(["decompress_column"], [False]):
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    "None",
                                    "None",
                                    is_query_column=is_query_column,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    for encoding, patchers_per_encoding in zip(
        ["ALP", "ALPExtended"], [PATCHERS[1:4], PATCHERS[4:]]
    ):
        for data_type in ["float", "double"]:
            for binding in ["decompress_column", "query_column"]:
                is_query_column = binding == "query_column"
                write_file(
                    f"{encoding.lower()}-{data_type}-{binding}-bindings.cu",
                    [
                        get_function(
                            encoding,
                            data_type,
                            binding,
                            "bool" if is_query_column else data_type + "*",
                            [
                                get_if_statement_check_wrapper(
                                    args.disable_unnecessary,
                                    encoding,
                                    data_type,
                                    binding,
                                    n_vec,
                                    n_val,
                                    unpacker,
                                    patcher,
                                    is_query_column=is_query_column,
                                    n_repetitions=None,
                                )
                                for n_vec in [1, 4]
                                for n_val in [1]
                                for unpacker in UNPACKERS[2:]
                                for patcher in patchers_per_encoding
                            ],
                            is_query_column=is_query_column,
                        )
                    ],
                )

    write_kernel_bindings_header()

    expected_bindings = set(GENERATED_BINDINGS_MANIFEST["bindings"])
    if args.enable_dictshfl32:
        expected_bindings.update(GENERATED_BINDINGS_MANIFEST["optional_bindings"]["dictshfl32"])
    if WRITTEN_BINDINGS != expected_bindings:
        missing = sorted(expected_bindings - WRITTEN_BINDINGS)
        unexpected = sorted(WRITTEN_BINDINGS - expected_bindings)
        raise RuntimeError(
            f"generated binding outputs disagree with {MANIFEST_PATH}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if GENERATED_HEADERS_DIR is not None:
        expected_headers = set(GENERATED_BINDINGS_MANIFEST["headers"])
        if WRITTEN_HEADERS != expected_headers:
            raise RuntimeError(
                f"generated header outputs disagree with {MANIFEST_PATH}: "
                f"expected={sorted(expected_headers)}, actual={sorted(WRITTEN_HEADERS)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "-du",
        "--disable-unnecessary",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
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
        help="Directory to write the generated binding .cu files into.",
    )
    parser.add_argument(
        "--header-out-dir",
        help="Directory to write generated benchmark binding headers into.",
    )
    parser.add_argument(
        "--enable-dictshfl32",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Also generate experimental DICTShfl32 binding translation units.",
    )

    args = parser.parse_args()
    GENERATED_BINDINGS_DIR = os.path.abspath(args.out_dir)
    GENERATED_HEADERS_DIR = (
        os.path.abspath(args.header_out_dir) if args.header_out_dir else None
    )
    os.makedirs(GENERATED_BINDINGS_DIR, exist_ok=True)
    if GENERATED_HEADERS_DIR is not None:
        os.makedirs(GENERATED_HEADERS_DIR, exist_ok=True)
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )
    main(args)
