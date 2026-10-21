#!/usr/bin/env python3
"""Write/verify the Parquet mirror using a local Python that has pyarrow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = np.load(args.runtime_mapping, allow_pickle=False)
    schema_version = str(archive["schema_version"].item())
    writer_version = str(archive["writer_contract_version"].item())
    schema = pa.schema(
        [
            ("logical_sample_id", pa.string()),
            ("label", pa.int32()),
            ("galp_image_id", pa.int64()),
            ("manifest_index", pa.int64()),
            ("planned_physical_position", pa.int64()),
            ("virtual_pls_id", pa.int32()),
            ("position_in_pls", pa.int32()),
            ("width", pa.int32()),
            ("height", pa.int32()),
        ],
        metadata={
            b"schema_version": schema_version.encode("ascii"),
            b"writer_contract_version": writer_version.encode("ascii"),
        },
    )
    table = pa.Table.from_pydict(
        {
            name: archive[name]
            for name in (
                "logical_sample_id",
                "label",
                "galp_image_id",
                "manifest_index",
                "planned_physical_position",
                "virtual_pls_id",
                "position_in_pls",
                "width",
                "height",
            )
        },
        schema=schema,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=["logical_sample_id"],
            write_statistics=True,
            row_group_size=64 * 1024,
        )
        verified = pq.read_table(temporary)
        if verified.num_rows != table.num_rows or verified.schema.names != schema.names:
            raise RuntimeError("Parquet verification changed row count or schema")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
